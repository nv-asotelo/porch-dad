/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Standalone diagnostic: includes the pinned public kernel, retaining its NVIDIA/MIT HAN Lab licenses.
// Compare NPerBlock=2 and 4 for batch-one Cosmos MLP shapes; no model or backend mutation.
// Check the advertised source SHA externally before compiling. Link CUDA runtime only, not backend archives.
// CUDA event batches use synthetic packed weights repeatedly; they do not measure whole-model latency.

#define gemv_forward_cuda_new gemv_forward_cuda_new_baseline
#include "kernels/int4GroupwiseGemmKernels/int4WoQGemvCuda.cu"
#undef gemv_forward_cuda_new

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

namespace
{
constexpr char kSOURCE_SHA[] = "3a55142d894fef9ae50ff8f609d385a7b2fbee9be09d73b69a3ea97fee7e595c";
constexpr int kGUARD_HALVES = 128;
constexpr uint16_t kGUARD_BITS = 0x55AA;
constexpr uint16_t kBASELINE_POISON = 0x7E01;
constexpr uint16_t kCANDIDATE_POISON = 0x7E02;

void cudaCheck(cudaError_t result, char const* operation)
{
    if (result != cudaSuccess)
    {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

class Stream
{
public:
    Stream() { cudaCheck(cudaStreamCreateWithFlags(&mValue, cudaStreamNonBlocking), "create stream"); }
    ~Stream() { cudaStreamDestroy(mValue); }
    Stream(Stream const&) = delete;
    Stream& operator=(Stream const&) = delete;
    cudaStream_t get() const { return mValue; }
private:
    cudaStream_t mValue{};
};

class Event
{
public:
    Event() { cudaCheck(cudaEventCreate(&mValue), "create event"); }
    ~Event() { cudaEventDestroy(mValue); }
    Event(Event const&) = delete;
    Event& operator=(Event const&) = delete;
    cudaEvent_t get() const { return mValue; }
private:
    cudaEvent_t mValue{};
};

template <typename T>
class DeviceBuffer
{
public:
    explicit DeviceBuffer(size_t count) { cudaCheck(cudaMalloc(&mData, count * sizeof(T)), "allocate device buffer"); }
    ~DeviceBuffer() { cudaFree(mData); }
    DeviceBuffer(DeviceBuffer const&) = delete;
    DeviceBuffer& operator=(DeviceBuffer const&) = delete;
    T* get() const { return mData; }
private:
    T* mData{};
};

struct Shape
{
    int n;
    int k;
};

struct Settings
{
    int warmup{100};
    int iterations{200};
    int trials{12};
    bool graph{false};
    bool checkOnly{false};
};

uint32_t randomNext(uint32_t& state)
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

float randomSigned(uint32_t& state)
{
    return (static_cast<int>(randomNext(state) & 65535U) - 32768) / 32768.0F;
}

std::string fingerprint(void const* data, size_t bytes)
{
    uint64_t hash = 14695981039346656037ULL;
    auto const* cursor = static_cast<uint8_t const*>(data);
    for (size_t i = 0; i < bytes; ++i)
    {
        hash = (hash ^ cursor[i]) * 1099511628211ULL;
    }
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << hash;
    return out.str();
}

half halfFromBits(uint16_t bits)
{
    half value;
    static_assert(sizeof(value) == sizeof(bits));
    std::memcpy(&value, &bits, sizeof(bits));
    return value;
}

struct Fixture
{
    std::vector<half> inputs;
    std::vector<uint32_t> weights;
    std::vector<half> scales;
};

constexpr std::array<char const*, 9> kPATTERNS{
    "signed_random", "positive_relu2", "sparse_boundaries", "alternating_cancellation", "signed_zero",
    "zero_weights", "unit_weights", "wide_finite", "subnormal_inputs"};

Fixture makeFixture(Shape shape, int pattern, uint32_t seed)
{
    Fixture fixture;
    fixture.inputs.resize(shape.k);
    fixture.weights.resize(static_cast<size_t>(shape.n) * shape.k / 8);
    fixture.scales.resize(static_cast<size_t>(shape.k / 128) * shape.n);
    uint32_t state = seed;
    for (int i = 0; i < shape.k; ++i)
    {
        float value = randomSigned(state);
        switch (pattern)
        {
        case 1:
            value = std::max(value, 0.0F);
            value *= value;
            break;
        case 2:
            value = (i % 128 == 0 || i % 128 == 127 || i == shape.k - 1) ? value : 0.0F;
            break;
        case 3:
            value = (i % 2 == 0) ? 1.0F : -1.0F;
            break;
        case 4:
            value = (i % 2 == 0) ? 0.0F : -0.0F;
            break;
        case 7:
            value = std::ldexp(value, (i % 17) - 8);
            break;
        default:
            break;
        }
        fixture.inputs[i] = pattern == 8
            ? halfFromBits(static_cast<uint16_t>((i % 1023 + 1) | ((i % 2) ? 0x8000 : 0)))
            : __float2half_rn(value);
    }
    // Native V1 packed nibbles store q+8. Random physical packed bytes are a valid permutation of a dense INT4 matrix.
    // This tests launch equivalence, not the checkpoint's quantization or packing conversion.
    for (size_t i = 0; i < fixture.weights.size(); ++i)
    {
        uint32_t word = 0;
        for (int nibble = 0; nibble < 8; ++nibble)
        {
            int q = static_cast<int>(randomNext(state) % 15) - 7;
            if (pattern == 3)
            {
                q = ((i + nibble) % 2) ? -7 : 7;
            }
            else if (pattern == 5)
            {
                q = 0;
            }
            else if (pattern == 6)
            {
                q = 1;
            }
            word |= static_cast<uint32_t>(q + 8) << (4 * nibble);
        }
        fixture.weights[i] = word;
    }
    for (size_t i = 0; i < fixture.scales.size(); ++i)
    {
        float const base = pattern == 7 ? 0.00025F : 0.015625F;
        float const scale = base * (1.0F + static_cast<float>(randomNext(state) % 32) / 16.0F);
        fixture.scales[i] = __float2half_rn(scale);
    }
    return fixture;
}

class Buffers
{
public:
    explicit Buffers(Shape shape)
        : mShape(shape), mInput(shape.k), mWeight(static_cast<size_t>(shape.n) * shape.k / 8),
          mScale(static_cast<size_t>(shape.k / 128) * shape.n),
          mBaseline(shape.n + 2 * kGUARD_HALVES), mCandidate(shape.n + 2 * kGUARD_HALVES)
    {
    }

    void upload(Fixture const& fixture)
    {
        cudaCheck(cudaMemcpy(mInput.get(), fixture.inputs.data(), fixture.inputs.size() * sizeof(half),
            cudaMemcpyHostToDevice), "upload activations");
        cudaCheck(cudaMemcpy(mWeight.get(), fixture.weights.data(), fixture.weights.size() * sizeof(uint32_t),
            cudaMemcpyHostToDevice), "upload weights");
        cudaCheck(cudaMemcpy(mScale.get(), fixture.scales.data(), fixture.scales.size() * sizeof(half),
            cudaMemcpyHostToDevice), "upload scales");
        std::vector<uint16_t> initial(mShape.n + 2 * kGUARD_HALVES, kGUARD_BITS);
        std::fill(initial.begin() + kGUARD_HALVES, initial.end() - kGUARD_HALVES, kBASELINE_POISON);
        cudaCheck(cudaMemcpy(mBaseline.get(), initial.data(), initial.size() * sizeof(uint16_t),
            cudaMemcpyHostToDevice), "initialize baseline output");
        std::fill(initial.begin() + kGUARD_HALVES, initial.end() - kGUARD_HALVES, kCANDIDATE_POISON);
        cudaCheck(cudaMemcpy(mCandidate.get(), initial.data(), initial.size() * sizeof(uint16_t),
            cudaMemcpyHostToDevice), "initialize candidate output");
    }

    void launch(bool candidate, cudaStream_t stream) const
    {
        using namespace trt_edgellm::kernel;
        if (candidate)
        {
            gemv_kernel<4, 1, 256, 128><<<mShape.n / 16, 256, 0, stream>>>(
                mInput.get(), mWeight.get(), mScale.get(), mCandidate.get() + kGUARD_HALVES, mShape.k, mShape.n);
        }
        else
        {
            gemv_forward_cuda_new_baseline(mInput.get(), reinterpret_cast<int8_t const*>(mWeight.get()), mScale.get(),
                mBaseline.get() + kGUARD_HALVES, 1, mShape.n, mShape.k, 128, stream);
        }
    }

    void compare(int pattern, uint32_t seed, bool emit) const
    {
        std::vector<uint16_t> baseline(mShape.n + 2 * kGUARD_HALVES);
        std::vector<uint16_t> candidate(baseline.size());
        cudaCheck(cudaMemcpy(baseline.data(), mBaseline.get(), baseline.size() * sizeof(uint16_t),
            cudaMemcpyDeviceToHost), "download baseline output");
        cudaCheck(cudaMemcpy(candidate.data(), mCandidate.get(), candidate.size() * sizeof(uint16_t),
            cudaMemcpyDeviceToHost), "download candidate output");
        int nonzero = 0;
        for (size_t i = 0; i < baseline.size(); ++i)
        {
            bool const guard = i < kGUARD_HALVES || i >= baseline.size() - kGUARD_HALVES;
            if (baseline[i] != candidate[i] || (guard && baseline[i] != kGUARD_BITS))
            {
                std::ostringstream error;
                error << "Bitwise/guard mismatch N=" << mShape.n << " K=" << mShape.k << " pattern="
                      << kPATTERNS.at(pattern) << " seed=" << seed << " buffer_index=" << i << " baseline=0x"
                      << std::hex << baseline[i] << " candidate=0x" << candidate[i];
                throw std::runtime_error(error.str());
            }
            if (!guard)
            {
                if ((baseline[i] & 0x7C00U) == 0x7C00U)
                {
                    throw std::runtime_error("Nonfinite output or unwritten poison in bounded-finite fixture");
                }
                nonzero += (baseline[i] & 0x7FFFU) != 0;
            }
        }
        if ((pattern == 4 || pattern == 5) && nonzero != 0)
        {
            throw std::runtime_error("Zero-input/zero-weight fixture produced nonzero output");
        }
        if ((pattern == 0 || pattern == 1 || pattern == 2 || pattern == 6 || pattern == 7) && nonzero == 0)
        {
            throw std::runtime_error("Nonzero fixture unexpectedly produced all zero outputs");
        }
        if (emit)
        {
            std::cout << "{\"kind\":\"equivalence\",\"n\":" << mShape.n << ",\"k\":" << mShape.k
                      << ",\"pattern\":" << std::quoted(kPATTERNS.at(pattern)) << ",\"seed\":" << seed
                      << ",\"compared_fp16_bits\":" << mShape.n << ",\"nonzero_outputs\":" << nonzero
                      << ",\"output_fnv1a64\":\""
                      << fingerprint(baseline.data() + kGUARD_HALVES, mShape.n * sizeof(uint16_t))
                      << "\",\"guards_pass\":true,\"bitwise_pass\":true}" << std::endl;
        }
    }

private:
    Shape mShape;
    DeviceBuffer<half> mInput;
    DeviceBuffer<uint32_t> mWeight;
    DeviceBuffer<half> mScale;
    DeviceBuffer<half> mBaseline;
    DeviceBuffer<half> mCandidate;
};

class GraphBatch
{
public:
    GraphBatch(Buffers const& buffers, bool candidate, int iterations, cudaStream_t stream)
    {
        cudaCheck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin graph capture");
        for (int i = 0; i < iterations; ++i)
        {
            buffers.launch(candidate, stream);
        }
        cudaCheck(cudaStreamEndCapture(stream, &mGraph), "end graph capture");
        cudaCheck(cudaGraphInstantiate(&mExecutable, mGraph, nullptr, nullptr, 0), "instantiate graph");
    }
    ~GraphBatch()
    {
        cudaGraphExecDestroy(mExecutable);
        cudaGraphDestroy(mGraph);
    }
    GraphBatch(GraphBatch const&) = delete;
    GraphBatch& operator=(GraphBatch const&) = delete;
    void launch(cudaStream_t stream) const { cudaCheck(cudaGraphLaunch(mExecutable, stream), "launch graph batch"); }
private:
    cudaGraph_t mGraph{};
    cudaGraphExec_t mExecutable{};
};

double median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    return (values[(values.size() - 1) / 2] + values[values.size() / 2]) / 2.0;
}

void printArray(std::vector<double> const& values)
{
    std::cout << '[';
    for (size_t i = 0; i < values.size(); ++i)
    {
        std::cout << (i ? "," : "") << values[i];
    }
    std::cout << ']';
}

std::array<double, 2> benchmark(Buffers const& buffers, Shape shape, Settings const& settings, cudaStream_t stream)
{
    for (int i = 0; i < settings.warmup; ++i)
    {
        buffers.launch(false, stream);
        buffers.launch(true, stream);
    }
    cudaCheck(cudaGetLastError(), "warmup launches");
    cudaCheck(cudaStreamSynchronize(stream), "warmup synchronize");
    // Capture both modes so setup order cannot distinguish baseline/candidate. Used only with --graph.
    GraphBatch baselineGraph(buffers, false, settings.iterations, stream);
    GraphBatch candidateGraph(buffers, true, settings.iterations, stream);
    if (settings.graph)
    {
        baselineGraph.launch(stream);
        candidateGraph.launch(stream);
        cudaCheck(cudaStreamSynchronize(stream), "graph upload warmup");
    }
    Event start;
    Event end;
    std::array<std::vector<double>, 2> results;
    for (int trial = 0; trial < settings.trials; ++trial)
    {
        for (int order = 0; order < 2; ++order)
        {
            int const candidate = (trial + order) % 2;
            cudaCheck(cudaEventRecord(start.get(), stream), "record start");
            if (settings.graph)
            {
                (candidate ? candidateGraph : baselineGraph).launch(stream);
            }
            else
            {
                for (int i = 0; i < settings.iterations; ++i)
                {
                    buffers.launch(candidate != 0, stream);
                }
            }
            cudaCheck(cudaGetLastError(), "timed launches");
            cudaCheck(cudaEventRecord(end.get(), stream), "record end");
            cudaCheck(cudaEventSynchronize(end.get()), "synchronize end");
            float elapsed = 0.0F;
            cudaCheck(cudaEventElapsedTime(&elapsed, start.get(), end.get()), "elapsed time");
            results[candidate].push_back(elapsed / settings.iterations);
        }
    }
    std::array<double, 2> medians{median(results[0]), median(results[1])};
    std::cout << "{\"kind\":\"timing\",\"n\":" << shape.n << ",\"k\":" << shape.k
              << ",\"mode\":\"" << (settings.graph ? "graph_batch" : "host_launch_batch")
              << "\",\"warmup_per_variant\":" << settings.warmup << ",\"calls_per_trial\":" << settings.iterations
              << ",\"trials_per_variant\":" << settings.trials << ",\"order\":\"alternating_AB_BA\""
              << ",\"baseline_ms_per_call_trials\":";
    printArray(results[0]);
    std::cout << ",\"candidate_ms_per_call_trials\":";
    printArray(results[1]);
    std::cout << ",\"baseline_median_batch_mean_ms\":" << medians[0]
              << ",\"candidate_median_batch_mean_ms\":" << medians[1]
              << ",\"median_batch_mean_reduction_percent\":" << 100.0 * (1.0 - medians[1] / medians[0])
              << "}" << std::endl;
    return medians;
}

Settings parseSettings(int argc, char** argv)
{
    Settings settings;
    for (int i = 1; i < argc; ++i)
    {
        std::string const option = argv[i];
        if (option == "--graph")
        {
            settings.graph = true;
        }
        else if (option == "--check-only")
        {
            settings.checkOnly = true;
        }
        else if ((option == "--warmup" || option == "--iterations" || option == "--trials") && i + 1 < argc)
        {
            int const value = std::stoi(argv[++i]);
            if (option == "--warmup")
            {
                settings.warmup = value;
            }
            else if (option == "--iterations")
            {
                settings.iterations = value;
            }
            else
            {
                settings.trials = value;
            }
        }
        else
        {
            throw std::invalid_argument("Usage: profile_mlp_gemv [--graph] [--check-only] "
                "[--warmup N>=25] [--iterations N>=100] [--trials N>=4]");
        }
    }
    if (settings.warmup < 25 || settings.iterations < 100 || settings.trials < 4)
    {
        throw std::invalid_argument("Require warmup>=25, iterations>=100, trials>=4");
    }
    return settings;
}

template <int NPerBlock>
void printKernelAttributes()
{
    cudaFuncAttributes attributes{};
    cudaCheck(cudaFuncGetAttributes(&attributes, trt_edgellm::kernel::gemv_kernel<NPerBlock, 1, 256, 128>),
        "query kernel attributes");
    int blocksPerSm = 0;
    cudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocksPerSm, trt_edgellm::kernel::gemv_kernel<NPerBlock, 1, 256, 128>, 256, 0), "query occupancy");
    std::cout << "{\"kind\":\"kernel_attributes\",\"n_per_block\":" << NPerBlock
              << ",\"registers_per_thread\":" << attributes.numRegs
              << ",\"local_bytes_per_thread\":" << attributes.localSizeBytes
              << ",\"shared_bytes_per_block\":" << attributes.sharedSizeBytes
              << ",\"maximum_active_blocks_per_sm\":" << blocksPerSm << "}" << std::endl;
}
} // namespace

int main(int argc, char** argv)
{
    try
    {
        Settings const settings = parseSettings(argc, argv);
        cudaDeviceProp properties{};
        cudaCheck(cudaGetDeviceProperties(&properties, 0), "query device");
        cudaCheck(cudaSetDevice(0), "select device");
        int runtimeVersion = 0;
        int driverVersion = 0;
        cudaCheck(cudaRuntimeGetVersion(&runtimeVersion), "runtime version");
        cudaCheck(cudaDriverGetVersion(&driverVersion), "driver version");
        std::cout << std::fixed << std::setprecision(9);
        std::cout << "{\"kind\":\"environment\",\"source_sha256_required\":\"" << kSOURCE_SHA
                  << "\",\"source_sha_checked_by_harness\":false,\"device\":" << std::quoted(properties.name)
                  << ",\"sm\":" << properties.major * 10 + properties.minor
                  << ",\"multiprocessors\":" << properties.multiProcessorCount
                  << ",\"cuda_runtime_version\":" << runtimeVersion << ",\"cuda_driver_version\":" << driverVersion
                  << ",\"cuda_compiler_major\":" << __CUDACC_VER_MAJOR__
                  << ",\"cuda_compiler_minor\":" << __CUDACC_VER_MINOR__
                  << ",\"batch\":1,\"group_size\":128,\"threads_per_block\":256,\"baseline_n_per_block\":2"
                  << ",\"candidate_n_per_block\":4,\"weights\":\"synthetic native V1 q+8 packed INT4\""
                  << ",\"accumulation\":\"unchanged upstream half2 MAC and warp/interwarp reductions\""
                  << ",\"limitation\":\"repeated synthetic weights; full-model latency and RAM unmeasured\"}"
                  << std::endl;
        printKernelAttributes<2>();
        printKernelAttributes<4>();
        Stream stream;
        constexpr std::array<Shape, 2> kSHAPES{{{9216, 2048}, {2048, 9216}}};
        constexpr std::array<uint32_t, 2> kSEEDS{0xC05A05U, 0x9E3779B9U};
        int cases = 0;
        int64_t outputBitsCompared = 0;
        std::array<double, 2> projected{};
        // Finish all correctness cases before emitting any timing result.
        for (Shape const shape : kSHAPES)
        {
            Buffers buffers(shape);
            for (int pattern = 0; pattern < static_cast<int>(kPATTERNS.size()); ++pattern)
            {
                for (uint32_t const seed : kSEEDS)
                {
                    Fixture const fixture = makeFixture(shape, pattern, seed);
                    buffers.upload(fixture);
                    buffers.launch(false, stream.get());
                    buffers.launch(true, stream.get());
                    cudaCheck(cudaGetLastError(), "equivalence launches");
                    cudaCheck(cudaStreamSynchronize(stream.get()), "equivalence synchronize");
                    buffers.compare(pattern, seed, true);
                    ++cases;
                    outputBitsCompared += shape.n;
                }
            }
        }
        std::cout << "{\"kind\":\"equivalence_summary\",\"cases\":" << cases
                  << ",\"compared_fp16_values_bitwise\":" << outputBitsCompared << ",\"pass\":true}" << std::endl;
        if (!settings.checkOnly)
        {
            for (Shape const shape : kSHAPES)
            {
                Buffers buffers(shape);
                Fixture const fixture = makeFixture(shape, 0, kSEEDS[0]);
                buffers.upload(fixture);
                std::cout << "{\"kind\":\"timing_fixture\",\"n\":" << shape.n << ",\"k\":" << shape.k
                          << ",\"seed\":" << kSEEDS[0] << ",\"input_fnv1a64\":\""
                          << fingerprint(fixture.inputs.data(), fixture.inputs.size() * sizeof(half))
                          << "\",\"packed_weight_fnv1a64\":\""
                          << fingerprint(fixture.weights.data(), fixture.weights.size() * sizeof(uint32_t))
                          << "\",\"scales_fnv1a64\":\""
                          << fingerprint(fixture.scales.data(), fixture.scales.size() * sizeof(half)) << "\"}"
                          << std::endl;
                auto const times = benchmark(buffers, shape, settings, stream.get());
                buffers.compare(0, kSEEDS[0], false);
                projected[0] += 28 * times[0];
                projected[1] += 28 * times[1];
            }
            std::cout << "{\"kind\":\"synthetic_56_gemv_projection\",\"baseline_ms\":" << projected[0]
                      << ",\"candidate_ms\":" << projected[1] << ",\"saved_ms\":" << projected[0] - projected[1]
                      << ",\"is_model_latency_measurement\":false}" << std::endl;
        }
        return 0;
    }
    catch (std::exception const& error)
    {
        std::cerr << "profile_mlp_gemv: " << error.what() << std::endl;
        return 1;
    }
}
