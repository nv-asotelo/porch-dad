// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Paired GPU equivalence diagnostic for the public sampler and this project's candidate.
// Compile against an UNPATCHED pinned cpp/sampler/sampling.cu; the macro gives its
// original stage-1 kernel a distinct symbol. The candidate below matches the patch.
// The complete original source retains its NVIDIA Apache-2.0 header on inclusion.

#define topKStage1 topKStage1Baseline
#include "sampler/sampling.cu"
#undef topKStage1

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace trt_edgellm
{
template <typename T, int32_t BLOCK_SIZE_, int32_t BLOCKS_PER_BEAM_>
__global__ void topKStage1Retained(T const* __restrict__ logits, float* tmpLogits, int32_t* topKTmpIdBuf, float* topKTmpValBuf,
    SamplingParams const params)
{
    typedef cub::BlockReduce<TopK_2, BLOCK_SIZE_> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    __shared__ int32_t selectedIndex;

    auto const tid = static_cast<int32_t>(threadIdx.x);
    auto const bid = static_cast<int32_t>(blockIdx.x);

    auto const batchId = bid / BLOCKS_PER_BEAM_;
    auto const blockLane = bid % BLOCKS_PER_BEAM_;

    if (batchId >= params.batchSize)
        return;

    auto const vocabSize = params.vocabSize;
    auto const k = params.topK;
    auto const temperature = params.temperature;
    auto const invTemp = invTemp_device(temperature);

    auto const tmpLogBufIndex = batchId * vocabSize;
    auto const tmpTopKBufIndex = batchId * BLOCKS_PER_BEAM_ * k + blockLane * k;

    TopK_2 partial;
    float const MAX_T_VAL = FLT_MAX;

    // Copy logits to temporary buffer and apply temperature
    for (auto elemId = tid + blockLane * BLOCK_SIZE_; elemId < vocabSize; elemId += BLOCK_SIZE_ * BLOCKS_PER_BEAM_)
    {
        auto localIndex = elemId + tmpLogBufIndex;
        float logit = logitToFloat(logits[localIndex]);
        tmpLogits[localIndex] = logit * invTemp;
    }

    partial.init();
#pragma unroll
    for (auto elemId = tid + blockLane * BLOCK_SIZE_; elemId < vocabSize; elemId += BLOCK_SIZE_ * BLOCKS_PER_BEAM_)
    {
        auto index = elemId + tmpLogBufIndex;
        partial.insert(tmpLogits[index], index);
    }

    for (int32_t ite = 0; ite < k; ite++)
    {
        TopK_2 total = BlockReduce(tempStorage).Reduce(partial, topk2MaxOpFunctor());

        if (tid == 0)
        {
            auto const index = tmpTopKBufIndex + ite;
            topKTmpIdBuf[index] = total.index;
            topKTmpValBuf[index] = total.value;
            selectedIndex = total.index;

            if (total.index >= 0)
            {
                tmpLogits[total.index] = -MAX_T_VAL;
            }
        }
        __syncthreads();

        // Only the winning thread's partition changed; retain every other local maximum.
        if (ite + 1 < k && selectedIndex >= 0 && partial.index == selectedIndex)
        {
            partial.init();
#pragma unroll
            for (auto elemId = tid + blockLane * BLOCK_SIZE_; elemId < vocabSize;
                 elemId += BLOCK_SIZE_ * BLOCKS_PER_BEAM_)
            {
                auto index = elemId + tmpLogBufIndex;
                partial.insert(tmpLogits[index], index);
            }
        }
    }
}

} // namespace trt_edgellm

namespace
{
constexpr int32_t kTHREADS = 256;
constexpr int32_t kPARTITIONS = 8;
constexpr int64_t kGUARD_ELEMENTS = 16;
constexpr unsigned char kGUARD_BYTE = 0xA5;
using trt_edgellm::rt::DeviceType;
using trt_edgellm::rt::Tensor;

void cudaCheck(cudaError_t result, char const* operation)
{
    if (result != cudaSuccess)
    {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

struct OutputBuffers
{
    Tensor temporary;
    Tensor indices;
    Tensor values;
    Tensor hostTemporary;
    Tensor hostIndices;
    Tensor hostValues;

    OutputBuffers(int64_t logitsCount, int64_t topCount)
        : temporary({logitsCount + kGUARD_ELEMENTS}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT)
        , indices({topCount + kGUARD_ELEMENTS}, DeviceType::kGPU, nvinfer1::DataType::kINT32)
        , values({topCount + kGUARD_ELEMENTS}, DeviceType::kGPU, nvinfer1::DataType::kFLOAT)
        , hostTemporary({logitsCount + kGUARD_ELEMENTS}, DeviceType::kCPU, nvinfer1::DataType::kFLOAT)
        , hostIndices({topCount + kGUARD_ELEMENTS}, DeviceType::kCPU, nvinfer1::DataType::kINT32)
        , hostValues({topCount + kGUARD_ELEMENTS}, DeviceType::kCPU, nvinfer1::DataType::kFLOAT)
    {
    }

    void initialize(cudaStream_t stream)
    {
        for (Tensor* tensor : {&temporary, &indices, &values})
        {
            cudaCheck(cudaMemsetAsync(tensor->rawPointer(), kGUARD_BYTE, tensor->getMemoryCapacity(), stream),
                "initialize stage-1 buffers");
        }
    }

    void download(cudaStream_t stream)
    {
        cudaCheck(cudaMemcpyAsync(hostTemporary.rawPointer(), temporary.rawPointer(), temporary.getMemoryCapacity(),
            cudaMemcpyDeviceToHost, stream), "download temporary logits");
        cudaCheck(cudaMemcpyAsync(hostIndices.rawPointer(), indices.rawPointer(), indices.getMemoryCapacity(),
            cudaMemcpyDeviceToHost, stream), "download top-K indices");
        cudaCheck(cudaMemcpyAsync(hostValues.rawPointer(), values.rawPointer(), values.getMemoryCapacity(),
            cudaMemcpyDeviceToHost, stream), "download top-K values");
    }

    void restoreValues(cudaStream_t stream)
    {
        cudaCheck(cudaMemcpyAsync(values.rawPointer(), hostValues.rawPointer(), values.getMemoryCapacity(),
            cudaMemcpyHostToDevice, stream), "restore stage-2 input values");
    }
};

void requireEqual(Tensor const& first, Tensor const& second, int64_t elements, std::string const& label)
{
    auto const* a = static_cast<unsigned char const*>(first.rawPointer());
    auto const* b = static_cast<unsigned char const*>(second.rawPointer());
    constexpr size_t kELEMENT_BYTES = sizeof(uint32_t);
    for (int64_t i = 0; i < elements; ++i)
    {
        if (std::memcmp(a + i * kELEMENT_BYTES, b + i * kELEMENT_BYTES, kELEMENT_BYTES) != 0)
        {
            uint32_t bitsA{};
            uint32_t bitsB{};
            std::memcpy(&bitsA, a + i * kELEMENT_BYTES, kELEMENT_BYTES);
            std::memcpy(&bitsB, b + i * kELEMENT_BYTES, kELEMENT_BYTES);
            throw std::runtime_error(label + " differs at element " + std::to_string(i)
                + ": baseline_bits=" + std::to_string(bitsA) + ", candidate_bits=" + std::to_string(bitsB));
        }
    }
    for (int64_t i = elements * kELEMENT_BYTES; i < (elements + kGUARD_ELEMENTS) * kELEMENT_BYTES; ++i)
    {
        if (a[i] != kGUARD_BYTE || b[i] != kGUARD_BYTE)
        {
            throw std::runtime_error(label + " overran its trailing guard");
        }
    }
}

template <typename T>
T convertValue(float value)
{
    if constexpr (std::is_same_v<T, half>)
    {
        return __float2half(value);
    }
    else
    {
        return value;
    }
}

template <typename T>
float toFloat(T value)
{
    if constexpr (std::is_same_v<T, half>)
    {
        return __half2float(value);
    }
    else
    {
        return value;
    }
}

template <typename T>
float fixtureValue(int64_t index, int32_t pattern, uint32_t& randomState)
{
    float const maximum = std::is_same_v<T, half> ? 65504.0F : FLT_MAX;
    float value{};
    switch (pattern)
    {
    case 0:
    {
        randomState = randomState * 1664525U + 1013904223U;
        value = static_cast<float>(randomState >> 8) / 16777216.0F * 32.0F - 16.0F;
        break;
    }
    case 1:
    {
        value = 2.0F;
        break;
    }
    case 2:
    {
        value = static_cast<int32_t>(index % 17) - 8.0F;
        break;
    }
    case 3:
    {
        value = (index % 2 == 0) ? 0.0F : -0.0F;
        break;
    }
    case 4:
    {
        float const extremes[]{maximum, -maximum, 1.0F, -1.0F,
            std::numeric_limits<float>::min(), std::numeric_limits<float>::denorm_min(), 0.0F};
        value = extremes[index % 7];
        break;
    }
    case 5:
    {
        float const nonfinite[]{std::numeric_limits<float>::quiet_NaN(), INFINITY, -INFINITY,
            maximum, -maximum, 1.0F, -1.0F};
        value = nonfinite[index % 7];
        break;
    }
    case 6:
    {
        float const invalid[]{std::numeric_limits<float>::quiet_NaN(), -INFINITY, -FLT_MAX};
        value = invalid[index % 3];
        break;
    }
    default:
    {
        throw std::invalid_argument("Unknown fixture pattern");
    }
    }
    return value;
}

struct Counters
{
    int64_t cases{};
    int64_t stage1Indices{};
    int64_t stage1ValueBits{};
    int64_t scratchValueBits{};
    int64_t sampledTokens{};
};

template <typename T>
void checkCase(int32_t batch, int32_t vocabulary, int32_t topK, float temperature,
    int32_t pattern, cudaStream_t stream, Counters& counters)
{
    using namespace trt_edgellm;
    int64_t const logitsCount = static_cast<int64_t>(batch) * vocabulary;
    int64_t const topCount = static_cast<int64_t>(batch) * kPARTITIONS * topK;
    nvinfer1::DataType const dtype = std::is_same_v<T, half> ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT;
    Tensor input({logitsCount}, DeviceType::kGPU, dtype);
    Tensor hostInput({logitsCount}, DeviceType::kCPU, dtype);
    OutputBuffers baseline(logitsCount, topCount);
    OutputBuffers candidate(logitsCount, topCount);
    SamplingParams const params(batch, vocabulary, temperature, topK, 0.9F);
    if (params.topK != topK)
    {
        throw std::runtime_error("Fixture unexpectedly triggered SamplingParams top-K normalization");
    }
    uint32_t randomState = 0xC05A05U;
    bool stage2Safe = topK <= vocabulary && pattern != 5 && pattern != 6;
    float const inverseTemperature = temperature < 1e-3F ? 1000.0F : 1.0F / temperature;
    T* hostData = hostInput.dataPointer<T>();
    for (int64_t i = 0; i < logitsCount; ++i)
    {
        hostData[i] = convertValue<T>(fixtureValue<T>(i, pattern, randomState));
        stage2Safe = stage2Safe && std::isfinite(toFloat(hostData[i]) * inverseTemperature);
    }
    cudaCheck(cudaMemcpyAsync(input.rawPointer(), hostInput.rawPointer(), input.getMemoryCapacity(),
        cudaMemcpyHostToDevice, stream), "upload fixture");
    baseline.initialize(stream);
    candidate.initialize(stream);
    topKStage1Baseline<T, kTHREADS, kPARTITIONS><<<batch * kPARTITIONS, kTHREADS, 0, stream>>>(
        input.dataPointer<T>(), baseline.temporary.dataPointer<float>(), baseline.indices.dataPointer<int32_t>(),
        baseline.values.dataPointer<float>(), params);
    cudaCheck(cudaGetLastError(), "baseline stage-1 launch");
    topKStage1Retained<T, kTHREADS, kPARTITIONS><<<batch * kPARTITIONS, kTHREADS, 0, stream>>>(
        input.dataPointer<T>(), candidate.temporary.dataPointer<float>(), candidate.indices.dataPointer<int32_t>(),
        candidate.values.dataPointer<float>(), params);
    cudaCheck(cudaGetLastError(), "candidate stage-1 launch");
    baseline.download(stream);
    candidate.download(stream);
    cudaCheck(cudaStreamSynchronize(stream), "stage-1 comparison synchronize");
    std::string const label = "dtype=" + std::string(std::is_same_v<T, half> ? "FP16" : "FP32")
        + " batch=" + std::to_string(batch) + " vocab=" + std::to_string(vocabulary)
        + " K=" + std::to_string(topK) + " T=" + std::to_string(temperature)
        + " pattern=" + std::to_string(pattern);
    requireEqual(baseline.hostIndices, candidate.hostIndices, topCount, label + " indices");
    requireEqual(baseline.hostValues, candidate.hostValues, topCount, label + " value bits");
    requireEqual(baseline.hostTemporary, candidate.hostTemporary, logitsCount, label + " temporary logits bits");
    counters.stage1Indices += topCount;
    counters.stage1ValueBits += topCount;
    counters.scratchValueBits += logitsCount;

    if (stage2Safe)
    {
        Tensor firstTokens({batch}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
        Tensor secondTokens({batch}, DeviceType::kGPU, nvinfer1::DataType::kINT32);
        Tensor hostFirst({batch}, DeviceType::kCPU, nvinfer1::DataType::kINT32);
        Tensor hostSecond({batch}, DeviceType::kCPU, nvinfer1::DataType::kINT32);
        uint64_t const seeds[]{42, 0, 123456789};
        uint64_t const offsets[]{0, 3, 17};
        for (int32_t trial = 0; trial < 3; ++trial)
        {
            baseline.restoreValues(stream);
            candidate.restoreValues(stream);
            size_t const sharedBytes = topK * (sizeof(int32_t) + sizeof(float));
            topKStage2Sampling<kTHREADS><<<batch, kTHREADS, sharedBytes, stream>>>(
                baseline.indices.dataPointer<int32_t>(), baseline.values.dataPointer<float>(),
                firstTokens.dataPointer<int32_t>(), params, seeds[trial], offsets[trial]);
            topKStage2Sampling<kTHREADS><<<batch, kTHREADS, sharedBytes, stream>>>(
                candidate.indices.dataPointer<int32_t>(), candidate.values.dataPointer<float>(),
                secondTokens.dataPointer<int32_t>(), params, seeds[trial], offsets[trial]);
            cudaCheck(cudaGetLastError(), "paired stage-2 launch");
            cudaCheck(cudaMemcpyAsync(hostFirst.rawPointer(), firstTokens.rawPointer(), hostFirst.getMemoryCapacity(),
                cudaMemcpyDeviceToHost, stream), "download baseline token");
            cudaCheck(cudaMemcpyAsync(hostSecond.rawPointer(), secondTokens.rawPointer(), hostSecond.getMemoryCapacity(),
                cudaMemcpyDeviceToHost, stream), "download candidate token");
            cudaCheck(cudaStreamSynchronize(stream), "stage-2 comparison synchronize");
            if (std::memcmp(hostFirst.rawPointer(), hostSecond.rawPointer(), hostFirst.getMemoryCapacity()) != 0)
            {
                throw std::runtime_error(label + " final sampled tokens differ");
            }
            for (int32_t row = 0; row < batch; ++row)
            {
                int32_t const token = hostFirst.dataPointer<int32_t>()[row];
                if (token < 0 || token >= vocabulary)
                {
                    throw std::runtime_error(label + " sampled token out of range");
                }
            }
            counters.sampledTokens += batch;
        }
    }
    ++counters.cases;
    std::cout << "{\"case\":\"" << label << "\",\"stage1_bitwise_equal\":true,\"guards_intact\":true,"
              << "\"stage2_compared\":" << (stage2Safe ? "true" : "false") << "}" << std::endl;
}

template <typename T>
void runCases(cudaStream_t stream, Counters& counters)
{
    struct Shape { int32_t batch; int32_t vocabulary; int32_t topK; };
    Shape const shapes[]{{1, 1, 1}, {1, 7, 50}, {2, 257, 50}, {3, 2051, 50},
        {2, 8193, 17}, {1, 131072, 1}, {1, 131072, 50}, {2, 131071, 50}};
    float const temperatures[]{0.7F, 1.0F, 0.001F};
    for (Shape const shape : shapes)
    {
        for (float const temperature : temperatures)
        {
            for (int32_t pattern = 0; pattern < 7; ++pattern)
            {
                checkCase<T>(shape.batch, shape.vocabulary, shape.topK, temperature, pattern, stream, counters);
            }
        }
    }
    checkCase<T>(2, 2051, 1, 0.0F, 0, stream, counters);
}
} // namespace

int main()
{
    cudaStream_t stream{};
    try
    {
        cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
        Counters counters;
        std::cout << "{\"baseline_source_sha256_required\":\"90e73755fffe25a9c381283c83481813681b1b11284463730eae0681b6a0aa0f\","
                     "\"patterns\":[\"random\",\"equal\",\"ties\",\"signed_zero\",\"finite_extremes\","
                     "\"nonfinite_stage1_only\",\"invalid_stage1_only\"],\"candidate_installed\":false}" << std::endl;
        runCases<float>(stream, counters);
        runCases<half>(stream, counters);
        cudaCheck(cudaStreamSynchronize(stream), "final synchronize");
        cudaCheck(cudaStreamDestroy(stream), "destroy stream");
        stream = nullptr;
        std::cout << "{\"status\":\"passed\",\"cases\":" << counters.cases
                  << ",\"stage1_indices_compared\":" << counters.stage1Indices
                  << ",\"stage1_float_bit_patterns_compared\":" << counters.stage1ValueBits
                  << ",\"scratch_float_bit_patterns_compared\":" << counters.scratchValueBits
                  << ",\"final_sampled_tokens_compared\":" << counters.sampledTokens
                  << ",\"limitation\":\"GPU fixture equivalence; not model quality or end-to-end speedup\"}" << std::endl;
        return 0;
    }
    catch (std::exception const& error)
    {
        if (stream)
        {
            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
        }
        std::cerr << "Sampler equivalence FAILED: " << error.what() << std::endl;
        return 1;
    }
}
