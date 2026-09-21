// SPDX-License-Identifier: Apache-2.0
// Original bounded diagnostic for the public TensorRT-Edge-LLM sampler.
// Link with the pinned build's edgellmCore static library (sampling.cu, Tensor,
// logger and their normal dependencies), CUDA runtime, and TensorRT headers.
// Include roots: TensorRT-Edge-LLM/cpp and the installed TensorRT/CUDA headers.
// Example invocation: ./profile_sampler --greedy-compare --stress
// Supply --implementation-label when linked against a modified sampler archive.
// No model, checkpoint, input image, or serving parameter is changed.
// Synthetic fixed logits characterize this sampler only. CUDA event intervals
// include GPU work and any host-submission gaps; they are not whole-model timings.

#include "sampler/sampling.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
constexpr int kVocab = 131072;
constexpr int kBatch = 1;
constexpr int kTopK = 50;
constexpr float kTemperature = 0.7F;
constexpr float kTopP = 0.9F;
constexpr uint64_t kSamplingSeed = 42;
constexpr uint64_t kSamplingOffset = 0;
using Clock = std::chrono::steady_clock;

void cudaCheck(cudaError_t result, char const* operation)
{
    if (result != cudaSuccess)
    {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

struct Stream
{
    cudaStream_t value{};
    Stream() { cudaCheck(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking), "create stream"); }
    ~Stream() { if (value) cudaStreamDestroy(value); }
    Stream(Stream const&) = delete;
    Stream& operator=(Stream const&) = delete;
};

struct Event
{
    cudaEvent_t value{};
    Event() { cudaCheck(cudaEventCreate(&value), "create timing event"); }
    ~Event() { if (value) cudaEventDestroy(value); }
    Event(Event const&) = delete;
    Event& operator=(Event const&) = delete;
};

struct Settings
{
    int iterations{100}; // Per batch; at least 100 measured launches even with one batch.
    int batches{10};
    int warmup{25};
    bool greedyCompare{false};
    bool stress{false};
    std::string implementationLabel{"unmodified upstream"};
};

double quantile(std::vector<double> values, double fraction)
{
    std::sort(values.begin(), values.end());
    size_t const index = static_cast<size_t>(std::ceil(fraction * values.size())) - 1;
    return values[std::min(index, values.size() - 1)];
}

void stressCurrentSampler(std::vector<float> const& original, trt_edgellm::rt::Tensor& logits,
    trt_edgellm::rt::Tensor& indices, trt_edgellm::rt::Tensor& workspace, cudaStream_t stream)
{
    using namespace trt_edgellm;
    struct Case { char const* label; float temperature; int pattern; };
    std::vector<Case> const cases{{"random_t0.7", 0.7F, 0}, {"random_t1.0", 1.0F, 0},
        {"random_t0.1", 0.1F, 0}, {"equal_logits", 0.7F, 1},
        {"many_ties", 0.7F, 2}, {"dominant_token", 0.7F, 3}};
    for (auto const& fixture : cases)
    {
        std::vector<float> values = original;
        if (fixture.pattern == 1) std::fill(values.begin(), values.end(), 0.0F);
        if (fixture.pattern == 2)
            for (size_t i = 0; i < values.size(); ++i) values[i] = static_cast<int>(i % 13) - 6.0F;
        if (fixture.pattern == 3)
        {
            std::fill(values.begin(), values.end(), -24.0F);
            values[4242] = 24.0F;
        }
        cudaCheck(cudaMemcpyAsync(logits.rawPointer(), values.data(), values.size() * sizeof(float),
            cudaMemcpyHostToDevice, stream), "stress upload");
        SamplingParams const params(kBatch, kVocab, fixture.temperature, kTopK, kTopP);
        int32_t selected[2]{-1, -1};
        for (int repeat = 0; repeat < 2; ++repeat)
        {
            topKtopPSamplingFromLogits(logits, indices, params, workspace, stream,
                kSamplingSeed, kSamplingOffset);
            cudaCheck(cudaGetLastError(), "stress sampler launch");
            cudaCheck(cudaMemcpyAsync(&selected[repeat], indices.rawPointer(), sizeof(int32_t),
                cudaMemcpyDeviceToHost, stream), "stress download");
            cudaCheck(cudaStreamSynchronize(stream), "stress synchronize");
        }
        if (selected[0] < 0 || selected[0] >= kVocab || selected[0] != selected[1])
            throw std::runtime_error(std::string("Invalid/nonrepeatable fixed-seed stress output: ") + fixture.label);
        // Membership check permits ties at rank 50; it does not impose a new tie-breaking policy.
        std::vector<float> scaled = values;
        float const inverseTemperature = 1.0F / fixture.temperature;
        for (float& value : scaled) value *= inverseTemperature;
        float const selectedValue = scaled[selected[0]];
        std::nth_element(scaled.begin(), scaled.begin() + kTopK - 1, scaled.end(), std::greater<float>());
        if (selectedValue < scaled[kTopK - 1] || (fixture.pattern == 3 && selected[0] != 4242))
            throw std::runtime_error(std::string("Sampler violated stress fixture support: ") + fixture.label);
        std::cout << "{\"stress_case\":\"" << fixture.label << "\",\"temperature\":" << fixture.temperature
                  << ",\"selected_token_id\":" << selected[0]
                  << ",\"fixed_seed_repeat_equal\":true,\"within_top50_support\":true,"
                     "\"limitation\":\"support and repeatability only; not a distribution or candidate-equivalence test\"}"
                  << std::endl;
    }
    cudaCheck(cudaMemcpyAsync(logits.rawPointer(), original.data(), original.size() * sizeof(float),
        cudaMemcpyHostToDevice, stream), "restore timed fixture");
    cudaCheck(cudaStreamSynchronize(stream), "restore fixture synchronize");
}

void measure(char const* label, bool greedy, Settings const& settings,
    trt_edgellm::rt::Tensor const& logits, trt_edgellm::rt::Tensor& indices,
    trt_edgellm::rt::Tensor& workspace, cudaStream_t stream)
{
    using namespace trt_edgellm;
    SamplingParams const params(kBatch, kVocab, kTemperature, kTopK, kTopP);
    auto launch = [&]() {
        if (greedy)
        {
            // Diagnostic comparison only: this does not replace serving topK=50.
            selectAllTopK(logits, std::nullopt, indices, 1, workspace, stream);
        }
        else
        {
            // Same seed/offset defaults used by the current vanilla decoder.
            topKtopPSamplingFromLogits(logits, indices, params, workspace, stream,
                kSamplingSeed, kSamplingOffset);
        }
    };

    for (int i = 0; i < settings.warmup; ++i) launch();
    cudaCheck(cudaGetLastError(), "warmup launch");
    cudaCheck(cudaStreamSynchronize(stream), "warmup synchronize");

    Event begin;
    Event end;
    std::vector<double> gpuMeans;
    std::vector<double> enqueueMeans;
    for (int batch = 0; batch < settings.batches; ++batch)
    {
        // Allocation, input upload and output download are outside this interval.
        cudaCheck(cudaEventRecord(begin.value, stream), "record begin");
        auto const enqueueStart = Clock::now();
        for (int i = 0; i < settings.iterations; ++i) launch();
        auto const enqueueEnd = Clock::now();
        cudaCheck(cudaGetLastError(), "measured launch");
        cudaCheck(cudaEventRecord(end.value, stream), "record end");
        cudaCheck(cudaEventSynchronize(end.value), "measured synchronize");
        float elapsedMs{};
        cudaCheck(cudaEventElapsedTime(&elapsedMs, begin.value, end.value), "elapsed events");
        gpuMeans.push_back(static_cast<double>(elapsedMs) / settings.iterations);
        enqueueMeans.push_back(std::chrono::duration<double, std::milli>(enqueueEnd - enqueueStart).count()
            / settings.iterations);
    }

    int32_t selected{-1};
    cudaCheck(cudaMemcpyAsync(&selected, indices.rawPointer(), sizeof(selected),
        cudaMemcpyDeviceToHost, stream), "copy selected token");
    cudaCheck(cudaStreamSynchronize(stream), "selected-token synchronize");
    if (selected < 0 || selected >= kVocab) throw std::runtime_error("Invalid sampled token index");

    double const average = std::accumulate(gpuMeans.begin(), gpuMeans.end(), 0.0) / gpuMeans.size();
    double const enqueueAverage = std::accumulate(enqueueMeans.begin(), enqueueMeans.end(), 0.0)
        / enqueueMeans.size();
    std::cout << "{\"method\":\"" << label << "\",\"measured_calls\":"
              << settings.iterations * settings.batches << ",\"warmup_calls\":" << settings.warmup
              << ",\"iterations_per_batch\":" << settings.iterations << ",\"batches\":" << settings.batches
              << ",\"cuda_event_mean_ms_per_call\":" << average
              << ",\"cuda_event_min_batch_mean_ms\":" << *std::min_element(gpuMeans.begin(), gpuMeans.end())
              << ",\"cuda_event_p50_batch_mean_ms\":" << quantile(gpuMeans, 0.50)
              << ",\"cuda_event_p95_batch_mean_ms\":" << quantile(gpuMeans, 0.95)
              << ",\"host_enqueue_mean_ms_per_call\":" << enqueueAverage
              << ",\"selected_token_id\":" << selected << ",\"batch_mean_ms\":[";
    for (size_t i = 0; i < gpuMeans.size(); ++i)
    {
        if (i) std::cout << ',';
        std::cout << gpuMeans[i];
    }
    std::cout << "]}" << std::endl;
}
} // namespace

int main(int argc, char** argv)
{
    try
    {
        Settings settings;
        for (int i = 1; i < argc; ++i)
        {
            std::string const arg = argv[i];
            if (arg == "--greedy-compare") settings.greedyCompare = true;
            else if (arg == "--stress") settings.stress = true;
            else if (arg == "--implementation-label" && i + 1 < argc)
            {
                settings.implementationLabel = argv[++i];
            }
            else if ((arg == "--iterations" || arg == "--batches" || arg == "--warmup") && i + 1 < argc)
            {
                int const value = std::stoi(argv[++i]);
                if (arg == "--iterations") settings.iterations = value;
                else if (arg == "--batches") settings.batches = value;
                else settings.warmup = value;
            }
            else throw std::invalid_argument("Usage: profile_sampler [--iterations 100] [--batches 10] "
                "[--warmup 25] [--greedy-compare] [--stress] [--implementation-label LABEL]");
        }
        if (settings.iterations < 100 || settings.iterations > 10000 || settings.batches < 1
            || settings.batches > 100 || settings.warmup < 1 || settings.warmup > 1000)
            throw std::invalid_argument("Bounds: iterations 100..10000, batches 1..100, warmup 1..1000");
        if (settings.implementationLabel.empty() || settings.implementationLabel.size() > 128
            || std::any_of(settings.implementationLabel.begin(), settings.implementationLabel.end(),
                [](unsigned char value) { return value < 32 || value == 127; }))
            throw std::invalid_argument("Implementation label must contain 1..128 bytes without control characters");

        int device{};
        int runtimeVersion{};
        cudaDeviceProp properties{};
        cudaCheck(cudaGetDevice(&device), "get device");
        cudaCheck(cudaGetDeviceProperties(&properties, device), "get device properties");
        cudaCheck(cudaRuntimeGetVersion(&runtimeVersion), "get runtime version");
        Stream stream;
        using namespace trt_edgellm;
        using namespace trt_edgellm::rt;
        SamplingParams const params(kBatch, kVocab, kTemperature, kTopK, kTopP);
        size_t const workspaceBytes = std::max(getTopKtopPSamplingWorkspaceSize(kBatch, kVocab, params),
            getSelectAllTopKWorkspaceSize(kBatch, kVocab, 1));
        Tensor logits({kBatch, kVocab}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "profile.logits");
        Tensor indices({kBatch, 1}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "profile.indices");
        Tensor workspace({static_cast<int64_t>(workspaceBytes)}, DeviceType::kGPU,
            nvinfer1::DataType::kINT8, "profile.workspace");
        std::vector<float> values(kVocab);
        uint32_t state = 0xC05A05U;
        for (float& value : values)
        {
            state = state * 1664525U + 1013904223U;
            value = (static_cast<float>(state >> 8) / 16777216.0F) * 16.0F - 8.0F;
        }
        cudaCheck(cudaMemcpyAsync(logits.rawPointer(), values.data(), values.size() * sizeof(float),
            cudaMemcpyHostToDevice, stream.value), "upload logits");
        cudaCheck(cudaMemsetAsync(indices.rawPointer(), 0xFF, sizeof(int32_t), stream.value), "initialize output");
        cudaCheck(cudaStreamSynchronize(stream.value), "input synchronize");

        std::cerr << "Synthetic fixed logits only: no model-quality or end-to-end speedup claim. "
                     "Batch-mean quantiles are not individual-token latency quantiles.\n";
        std::cout << std::fixed << std::setprecision(6)
                  << "{\"fixture\":\"lcg32_seed_0xC05A05_uniform_minus8_plus8\",\"batch_size\":1,"
                     "\"vocab_size\":131072,\"logits_dtype\":\"FP32\",\"top_k\":50,"
                     "\"temperature\":0.7,\"top_p\":0.9,\"sampling_seed\":42,\"sampling_offset\":0,"
                     "\"sampling_implementation\":" << std::quoted(settings.implementationLabel)
                  << ",\"cuda_runtime_version\":" << runtimeVersion
                  << ",\"compute_capability\":\"" << properties.major << '.' << properties.minor
                  << "\",\"sm_count\":" << properties.multiProcessorCount
                  << ",\"workspace_bytes\":" << workspaceBytes << "}" << std::endl;
        if (settings.stress) stressCurrentSampler(values, logits, indices, workspace, stream.value);
        measure("topk50_t0.7_p0.9", false, settings, logits, indices, workspace, stream.value);
        if (settings.greedyCompare)
            measure("greedy_top1_diagnostic_only", true, settings, logits, indices, workspace, stream.value);
        return 0;
    }
    catch (std::exception const& error)
    {
        cudaDeviceSynchronize();
        std::cerr << "Sampler diagnostic failed: " << error.what() << '\n';
        return 1;
    }
}
