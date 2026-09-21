# Cosmos3 patch-embedding layout repair

The pinned experimental checkpoint-direct builder feeds the shared C++ runtime's CHW patches into an unmodified HWC patch-projection matrix. This changes color and spatial features. RGB decoding and normalization match the model configuration; patch sequence order also matches.

`cosmos3-patch-embedding-chw.patch` targets public NVIDIA TensorRT-Edge-LLM commit `e8b29522938901f6df19ebeedd4b69bc8edbcd97`. It applies the column permutation already used by that same repository's main exporter to its lightweight experimental builder. No checkpoint values are trained or modified. Only the adapted patch-projection matrix and bias are embedded into the visual engine; other engine weights continue using the original external-weight policy. The matrix is 1,769,472 bytes in FP16 and its bias is 2,304 bytes.

The patch rejects a quantized patch projector, because permuting packed quantized weights would need a separate implementation. The current FP16 vision tower and language-only AWQ candidates preserve this projector in FP16.

## Evidence and attribution

- [NVIDIA main exporter: HWC→CHW patch-weight adaptation](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/tensorrt_edgellm/models/cosmos3_reasoner/modeling_cosmos3_reasoner_visual.py#L104), invoked by its weight loader at line 748. Apache-2.0; NVIDIA Corporation and affiliates.
- [Shared C++ patchifier](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/cpp/kernels/preprocessKernels/imageUtilKernels.cu#L118): channel-major flattened values.
- [Official Transformers Cosmos3 processor](https://github.com/huggingface/transformers/blob/c587bc884db2c2e31fc2b8102314656b17aa07b1/src/transformers/models/cosmos3_edge/image_processing_cosmos3_edge.py#L142): block-major patches, HWC values inside each patch. Apache-2.0; NVIDIA and Hugging Face.
- The experimental `Linear` previously produced an identity external checkpoint recipe. Clearing that recipe is essential: otherwise the runtime reloads the unpermuted matrix. `ops/backend.py:942` folds the adapted FP16 matrix into the engine when the recipe is absent.

`results/cosmos3-patch-layout-diagnostic.json` compares the actual checkpoint patch matrix against actual frozen JPEG fixture pixels. Normalization and matrix values are cast to FP16, then the CPU computes the dot products in FP64 to isolate layout from accumulation error. Uncorrected maximum absolute error was 6.3207227 and cosine similarity 0.720724; the corrected projection exactly matched this reference. This verifies the first projection only, not end-to-end inference.

## Apply and verify

On a clean checkout at the pin:

```sh
git -C external/TensorRT-Edge-LLM apply --check ../../patches/cosmos3-patch-embedding-chw.patch
git -C external/TensorRT-Edge-LLM apply ../../patches/cosmos3-patch-embedding-chw.patch
python -m unittest discover -s tests -p test_cosmos3_patch_layout.py -v
```

Four CPU regressions cover color/spatial projection parity, column mapping and source immutability, invalid dimensions, and removal of the identity runtime recipe. TensorRT/CUDA are not imported by these tests. Syntax compilation and `git diff --check` also passed locally.

A **fresh visual engine must be built**. Editing metadata alone cannot apply this fix. No native extension rebuild is required. The language engine is unaffected, but using a fresh `COSMOS_CACHE_DIR` for the complete existing build script provides clean provenance and preserves all earlier evidence. The cache key does not establish which source patch built an old cache, so do not reuse the previous bundle. Then rerun the unchanged six-image quality screen before treating the candidate as valid.

Original work here is diagnosis, the small experimental-builder integration, and regression/evidence collection. The permutation and correct layout contract come from the credited public upstream sources.
