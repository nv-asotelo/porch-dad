# CuTe groups required by the exact reasoner

Read-only source audit at TensorRT-Edge-LLM `e8b29522938901f6df19ebeedd4b69bc8edbcd97`, using the verified Cosmos3-Edge checkpoint. This selects build components; no source patch, GPU build or inference result is claimed.

The checkpoint has a dense 28-layer decoder, hidden size 2048, 16 query/8 KV heads and head dimension 128, plus SigLIP2 vision with head dimension 72. Its reasoner has no MoE, GDN, SSD or talker path. The supported `fmha` group covers the decoder's paged FP16 context attention and the visual encoder's dimension 72 attention. XQA decode has a separate JIT path. Dense FP16 projections use TensorRT matrix multiplication, and normalization is decomposed into TensorRT operations. [FMHA runtime guards](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/cpp/kernels/contextAttentionKernels/cuteDslFMHAV2Runner.cpp#L110), [builder operations](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/experimental/builder/ops/backend.py#L669).

| Intended build | Generator group selection | CMake selection | SM87 variants |
| --- | --- | --- | ---: |
| FP16 reasoner | `--kernels fmha` | `-DENABLE_CUTE_DSL=fmha` | 25 |
| FP16 plus default INT4 V2 | `--kernels fmha,int4_fp16_gemm` | `'-DENABLE_CUTE_DSL=fmha;int4_fp16_gemm'` | 49 |
| All supported model groups | `--kernels ALL` | `-DENABLE_CUTE_DSL=ALL` | 81 |

The default INT4 V2 plugin requires the INT4 compile guard; without it, enqueue returns an error. The selected group supplies 16 GEMM tactics and 8 decode GEMV variants. Preserve FP16 vision/projector weights. [INT4 plugin](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/cpp/plugins/int4GroupwiseGemmPluginV2/int4GroupwiseGemmPluginV2.cpp#L354).

The generator supports comma-separated groups and merges compatible existing artifact metadata, archive members and headers. Thus an FP16 baseline can generate `fmha` first; a later INT4 build can generate only `int4_fp16_gemm` in the same artifact directory, without `--clean`, and reconfigure CMake to enable both. CMake accepts semicolon-separated groups from artifact metadata. [Generator](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/kernelSrcs/build_cutedsl.py#L1878), [CMake selector](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/cmake/CuteDsl.cmake#L336).

Explicit `--gpu_arch sm_87` bypasses automatic architecture detection, but not GPU use. FMHA checks device count before its export-only branch, allocates CuPy tensors, obtains a CUDA stream and queries device properties. Other groups also allocate GPU memory. Normal boot and a working GPU driver are required before source generation. CMake compilation itself can be driverless with matching prebuilt artifacts, but this checkout contains only checksum files and upstream describes the matching archive as unpublished. A preliminary `ENABLE_CUTE_DSL=OFF` build would not provide the intended runtime, so it is not part of this deployment. [FMHA generation](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/kernelSrcs/fmha_v2_cutedsl/fmha.py#L2687).
