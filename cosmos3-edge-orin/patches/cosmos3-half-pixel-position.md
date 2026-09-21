# Cosmos3 learned-position interpolation repair

The public Cosmos3 processor/model use `align_corners=False` when interpolating the learned 16×16 position table. The C++ Cosmos3 runner inherited Qwen3's endpoint-aligned coordinates, `(side - 1) * index / (size - 1)`. The required half-pixel coordinates are `clamp((index + 0.5) * side / size - 0.5, 0, side - 1)`.

`cosmos3-half-pixel-position.patch` targets public TensorRT-Edge-LLM commit `e8b29522938901f6df19ebeedd4b69bc8edbcd97`. It adds an optional `alignCorners=true` argument to the shared kernel API, retains the Qwen default, and selects `false` only in Cosmos3's runner. The existing four-tap gather, merge-block ordering, image offsets, and temporal repetition remain in use.

The patch changes the native runtime only. Rebuild `_edgellm_runtime` and its affected dependencies before starting a fresh server. This patch alone does not require rebuilding visual or language TensorRT engines. The separate patch-projection layout repair does require a new visual engine.

## Sources and validation

- [Official Transformers Cosmos3 learned-position interpolation](https://github.com/huggingface/transformers/blob/c587bc884db2c2e31fc2b8102314656b17aa07b1/src/transformers/models/cosmos3_edge/modeling_cosmos3_edge.py#L392), credited to NVIDIA and Hugging Face under Apache-2.0.
- [NVIDIA's main exporter uses half-pixel coordinates](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/tensorrt_edgellm/models/cosmos3_reasoner/modeling_cosmos3_reasoner_visual.py#L547), Apache-2.0.
- [The original shared C++ implementation uses endpoint coordinates](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/cpp/kernels/preprocessKernels/imageUtilKernels.cu#L711).

Run `python -m unittest discover -s tests -p test_cosmos3_position_layout.py -v` with NumPy and a host C++ compiler. The test compiles and executes the actual edited kernel body and host scale expressions, substituting host CUDA-index/half stand-ins. It compares all indices and FP16 weights exactly to independently constructed reference grids: 16×16 identity, 32×32 upsample, 16×32, 24×40, 40×24, and 64×16. Each uses two frames and a nonzero output offset, with untouched sentinel regions. Four Qwen grids also retain the previous formulas, including 8×10. Both tests passed locally; this is a host formula/packing check, not CUDA execution validation.

Using the actual checkpoint position table in a separate CPU arithmetic diagnostic, the original endpoint formula differed from the half-pixel formula by a maximum 14.8521 and RMS 0.131445 at 32×32. At 16×16 they match. These are intermediate embedding differences, not end-to-end quality scores.

## Remaining interpolation and resize limits

This patch establishes the reference's four-tap coordinate convention. It does **not** implement the wider antialias filter needed when either output patch-grid axis is smaller than the learned 16-patch side. The official model uses `antialias=True`; the main exporter also documents that its four-tap approximation differs for those downsampled axes. General reference parity is therefore unverified for a resized image axis below 256 pixels.

The active backend profile uses `min_image_tokens=4`, `max_image_tokens_per_image=512`, and `patch_size*merge_size=32`. `qwenSmartResize3D` therefore permits 4,096–524,288 pixels per still image, aligning each axis to 32. This constrains area, not a 256-pixel minimum per axis. The official checkpoint processor's minimum area is 65,536 pixels, which also does not guarantee a minimum axis for elongated images.

Examples under the active backend: a 128×128 image stays 128×128; a 384×216 frame rounds to 384×224; both enter the antialias limitation. A 384×288 4:3 frame, a 512×288 16:9 frame, and the unchanged 512×512 quality fixtures keep every axis at least 256. Very wide images can still have short axes after upper-area resizing. These examples describe the current profile, not a promise for every future memory-optimized profile. No UI, resize policy, or fixture was changed by this patch.

Original work is the Cosmos-specific native dispatch and validation. The interpolation convention comes from the credited public references.
