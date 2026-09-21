# Cosmos3-Edge / TensorRT Edge-LLM feasibility

Public-source audit: 2026-09-19. No model weights were downloaded beyond three bounded 1 MiB HTTP range reads used to inspect safetensors headers. No device inference has run: the coordinating agent reports no enumerated NVIDIA USB device and no reachable Jetson SSH endpoint.

## Decision

The requested combination has an upstream implementation. Start with the Cosmos3-Edge **reasoner**, which accepts images and text and emits text, using TensorRT Edge-LLM's resident C++ runtime and experimental OpenAI streaming server. The generator is outside this runtime's image/video synthesis contract and would consume unnecessary resources for live scene descriptions. [NVIDIA model design](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/models/cosmos3.html)

| Component | Pin / verified fact |
|---|---|
| Backend | NVIDIA/TensorRT-Edge-LLM `v0.10.1`, commit `e8b29522938901f6df19ebeedd4b69bc8edbcd97` |
| Model | `nvidia/Cosmos3-Edge`, revision `344d602b128d1bbdacb43b08d0a3626f46343e29`; public, ungated |
| Orin stack | Official matrix lists JetPack 7.2, CUDA 13.2, platform TensorRT 10, SM87 |
| Orin model precision | FP16, INT8, INT4; FP8/MXFP8/FP4/NVFP4 unsupported |
| Code license | Apache-2.0 |
| Model license | OpenMDW-1.1 |

Sources: [backend release](https://github.com/NVIDIA/TensorRT-Edge-LLM/tree/v0.10.1), [model revision](https://huggingface.co/nvidia/Cosmos3-Edge/tree/344d602b128d1bbdacb43b08d0a3626f46343e29), [official platform matrix](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/support-matrix.html). The backend matrix names the 7.2 family; the installation plan uses the coordinating agent's JetPack 7.2.1 selection, subject to checking actual installed packages.

OpenMDW-1.1 permits use, modification, and redistribution, with retention of the agreement and applicable copyright/origin notices in distributions. It imposes no output-specific restrictions. Preserve model and dependency license notices in deployment artifacts. [License text](https://openmdw.ai/license/1-1/)

## Architecture and memory arithmetic

The root config declares `Cosmos3EdgeForConditionalGeneration` / `cosmos3_edge`. The language tower has 28 layers, hidden size 2048, 16 query heads, 8 KV heads, head dimension 128, 9216-wide squared-ReLU FFNs, and a 131072-token vocabulary. The SigLIP2 vision tower has 27 layers, hidden size 1152, 16 heads, 16-pixel patches, and 2×2 spatial merging. The reasoner is separate from diffusion generation. [Pinned model config](https://huggingface.co/nvidia/Cosmos3-Edge/blob/344d602b128d1bbdacb43b08d0a3626f46343e29/config.json), [Transformers architecture documentation](https://huggingface.co/docs/transformers/main/model_doc/cosmos3_edge)

Calculated by summing safetensors header shapes for the exact tensors named in the pinned root reasoner index:

| Reasoner component | Parameters | BF16/FP16 weight bytes |
|---|---:|---:|
| Language, embedding, output head | 1,946,277,376 | 3,892,554,752 |
| Vision encoder | 412,649,712 | 825,299,424 |
| Vision-to-language projector | 76,692,992 | 153,385,984 |
| Total | 2,435,620,080 | 4,871,240,160 = 4.54 GiB |

This is weight arithmetic, **not measured allocated VRAM**. The model card's 4B describes the broader checkpoint, while reasoning uses this smaller subset. The physical reasoner-referenced shard files are approximately 7.72 GB because the two transformer shards also contain generator tensors; the full snapshot is approximately 9.18 GB. The root index selectively names 698 tensors across those transformer shards and `vision_encoder/model.safetensors`. [Pinned reasoner index](https://huggingface.co/nvidia/Cosmos3-Edge/blob/344d602b128d1bbdacb43b08d0a3626f46343e29/model.safetensors.index.json)

Further calculations, excluding engine workspaces, activations, metadata, OS, and alignment:

- FP16 KV storage = `2 × 28 × 8 × 128 × 2` = 114688 bytes/token = 112 KiB/token. Capacity 1024/2048/4096 costs 112/224/448 MiB respectively, at batch one.
- INT4 language linear weights, including the output head, with the input embedding and full visual path retained in FP16: approximately 2.19 GiB plus quantization scales and unquantized norms. This is an engineering estimate, not an available certified artifact or device measurement.
- Leaving the output head FP16 adds approximately 384 MiB versus INT4. FFN-only INT4 has a higher weight floor near 3.06 GiB.
- Orin uses shared system DRAM. Device allocation and total system memory must both be reported; neither is interchangeable with a discrete GPU's dedicated VRAM figure.

## Build and quantization constraints

CPU ONNX export plus device engine building is the documented supported workflow. The experimental direct builder avoids PyTorch/ONNX in the resident server and supports externalized checkpoint weights. The server caches a complete profile and admits one inference at a time; SSE ends with `[DONE]`. Its CLI accepts a checkpoint, not an existing engine path. [Reasoner workflow](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/examples/vla/cosmos3.html), [server contract](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/examples/experimental-server.html)

The direct server builds all physically declared Cosmos components. To serve only reasoning, create a separate checkpoint directory with the root reasoner config/index/tokenizer/processor files and their referenced shards, **without** `transformer/config.json` and `vae/config.json`. Never mutate the original snapshot. The component resolver then exposes only LLM and VISUAL. This conclusion comes from source inspection and awaits a device smoke test. [Component resolver at the release pin](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/experimental/builder/models/cosmos3/configuration.py)

`int4_awq` is the upstream supported quantizer choice; an INT4 LM head is separately selectable. No native RTN (round-to-nearest) checkpoint quantization switch was found in the pinned direct builder or quantizer. The builder consumes previously quantized checkpoints. Generic AWQ loads the source model and defaults to text calibration batches of 16; the documented minimum GPU memory is the FP16 checkpoint size, and activations increase this. Treat 8 GB on-device calibration as unproven. A separate NVIDIA GPU with 16–24 GB is a practical engineering preference, not a measured requirement. The source Mac cannot run this CUDA calibration. [Quantizer implementation](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/tensorrt_edgellm/quantization/quantize.py), [installation requirements](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/installation.html)

No official NVIDIA prequantized INT4 Cosmos3-Edge checkpoint was identified in the inspected model list. Cosmos-specific AWQ export and visual-projector handling need validation. The generic exclusion prefixes include `visual` but do not include the Cosmos `projector` name; do not assume the projector stays FP16 without inspecting the exported tensors. [Quantization configuration](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/tensorrt_edgellm/quantization/quantization_configs.py)

For low-memory INT4 builds, upstream specifically recommends `--externalize-weights int4_ffn` on Orin Nano. The direct server already selects externalization of all supported weight kinds. The direct builder is experimental; INT4 and Cosmos multimodal combinations lack comprehensive automatic direct-builder CI. [Supported model precision notes](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/supported-models.html), [direct-builder validation status](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/direct-engine-builder.html)

## Bounded optimization candidates

1. Establish text then image correctness with reasoner-only FP16, batch one, 1024 input / 2048 KV capacity, one resident backend, and capped 64-token answers.
2. Reduce visual input resolution and compare equal-content prompts; allow at most one pending camera frame. Time capture-to-first-token separately from backend prefill and decode.
3. Attempt AWQ once a suitable CUDA calibration host is available. Compare against FP16 on fixed images, OCR, and counts before accepting the artifact.
4. Follow `GOAL.md`: at most 12 measured candidates, with 5 warmups and at least 30 measured requests each. Stop after 3 consecutive candidates without at least 5% improvement in p50 latency or peak memory, with no greater than 5% regression in the counterpart or p95 latency. Require equivalent image-grounding quality, reject OOM/errors/swapping/backlog, and retain the best passing Pareto configurations. Select the fastest that preserves at least 512 MiB system MemAvailable throughout a 10-minute sustained run.

Do not propose FP8 KV or FP8 embeddings for SM87: upstream requires SM89+. DART's published guard names Qwen mRoPE families, so do not claim Cosmos pruning without checking actual pruned-token counters. [FP8 KV constraints](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/features/FP8KV.html), [FP8 embedding constraints](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/features/fp8-embedding.html), [DART limitations](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/features/visual-token-pruning.html)

Public Cosmos benchmarks include eager Transformers on **AGX Orin 64 GB**, e.g. image decode 12.3 tokens/s, but no corresponding Orin Nano Edge-LLM measurement. Those numbers cannot substitute for this project's measurements. [Published benchmark table](https://github.com/NVIDIA/cosmos/blob/main/inference_benchmarks.md#embedded-platform-eager-transformers)
