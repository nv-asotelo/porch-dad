# 04 — Quantize the text tower to INT4 and build both engines

This is the step that produced the largest single win in this deployment: decode went from
43.29 ms/token to 13.38 ms/token (Round 3, −69.1%), the LLM engine went from 3.135 GB to 0.818 GB
on disk, and resident RSS dropped by 2.32 GB. Round 2 had already established *why* this was the
only remaining lever: at FP16 the decoder was moving 3.36 GB of weights per token in 43.29 ms
≈ 77.6 GB/s against the Orin Nano's ~102 GB/s peak, i.e. **76% of theoretical bandwidth**. No
scheduling or caching trick helps at that point. The weights themselves have to get smaller.

The work splits into four parts:

1. Why the three obvious quantization routes are closed on this model and toolchain.
2. The route that works: calibration-free RTN with MSE-optimal clipping.
3. Four verification gates, because a silent quantization bug is invisible until quality collapses.
4. ONNX export (host) and engine builds (device).

Target platform for everything below: Jetson Orin Nano Super 8 GB, JetPack 7.2.1 / L4T R39.2.1,
CUDA 13.2, TensorRT 10.16.2.10, TensorRT-Edge-LLM 0.10.1.

All `ssh`/`scp` examples use `JETSON_HOST`, which defaults to `orin@jetson.local` throughout this
repo:

```bash
export JETSON_HOST="${JETSON_HOST:-orin@jetson.local}"
```

---

## Part 1 — The three closed routes

Each of these was verified by reading the toolchain source, not assumed. File:line citations refer
to the TensorRT-Edge-LLM 0.10.1 Python package as installed.

### Route A — `tensorrt-edgellm-export --quantization int4_awq`

The flag exists and the CLI accepts it without complaint. It is gated behind
`_needs_moe_quantization` (`tensorrt_edgellm/scripts/export.py:1046,1074`) and is dropped earlier
still at `tensorrt_edgellm/scripts/export.py:4129-4131`. **It applies to Mixture-of-Experts models
only.** Cosmos3-Edge is dense, so passing the flag is a **silent no-op**: the export succeeds, the
engine builds, and the weights are still FP16. Nothing in the log says otherwise.

This is the most expensive of the three failures precisely because it does not fail. Check the
resulting engine size and the build log's `Total Weights Memory` (see Part 4) rather than trusting
the flag.

Related trap from the same CLI family: `tensorrt-edgellm-quantize` aborts with
`KeyError: 'cosmos3_edge'` on this model. Use `tensorrt-edgellm-export --task reasoning` instead.

### Route B — modelopt AWQ calibration

AWQ derives its per-channel scales from *observed activations*, which requires an eager PyTorch
forward pass over calibration data. In TensorRT-Edge-LLM, `attention_plugin` is a
`torch.library.custom_op` whose body is literally `return torch.zeros(...)`
(`tensorrt_edgellm/models/ops.py:87-193`) — a shape-only stub that exists so ONNX export can infer
tensor shapes, not a real attention implementation.

Consequently an eager forward pass yields zeros, and **no meaningful activation statistics can ever
be collected. AWQ calibration is structurally impossible in this toolchain**, independent of how
much calibration data or time you throw at it. (A separate attempt on a Qwen VLM, where the path is
nominally supported, confirmed the cost even when it does work: roughly 4 hours of calibration and
a GPU OOM.)

### Route C — the community pre-quantized checkpoint

`ubr-physical-ai/Cosmos3-Edge-INT4-AWQ` on HuggingFace is a W4A16_AWQ, group_size 128 quantization
of Cosmos3-Edge with ONNX published alongside, produced for a Jetson Orin Nano during the NVIDIA /
OpenHackathons / Oracle Open Models Codefest 2026. It is the closest existing artifact to what this
deployment needed.

**It was declined.** Its own documentation states that the checkpoint **has never been executed**.
An unexecuted checkpoint is a hypothesis, not a deliverable — and it is exactly the kind of artifact
whose failure mode (all-zero or NaN weights from a broken quantizer backend) is invisible until you
run it.

Only the checkpoint's **safetensors header metadata** was read — tensor names, shapes, and dtypes,
no weight data — to cross-check the on-disk packing contract described in Part 2. That is the only
way this artifact was used here.

---

## Part 2 — Calibration-free RTN with MSE-optimal clipping

Round-to-nearest (RTN) needs no activation data at all, which sidesteps Route B's blocker entirely.
Plain RTN is, however, noticeably worse than it needs to be on outlier-heavy LLM weights, so the
quantizer adds a clipping search that still uses **only the weights**.

### The algorithm

For each `[N, K]` weight matrix, reshaped into groups of `GROUP = 128` along the input dimension:

1. Compute `amax` per group (absolute max, clamped to a `1e-8` floor).
2. Sweep a per-group scale multiplier `alpha` over `torch.linspace(0.55, 1.0, 19)`. For each alpha:
   - `s_try = (amax * alpha) / 7.0`
   - `q_try = round(g / s_try).clamp(-8, 7)`
   - `err = ((q_try * s_try - g) ** 2).sum(...)` — squared reconstruction error, per group.
3. Keep, **per group independently**, the `alpha` (i.e. the scale) that minimizes `err`. Selection is
   done with `torch.where` over the whole tensor at once, so all groups are searched in parallel.
4. Quantize once more with the winning scale to produce the final `q`.

`alpha = 1.0` is plain RTN and is included in the sweep, so the search can never do worse than RTN on
its own objective. Shrinking the range (`alpha < 1`) clips the largest outliers but gives every other
weight in the group finer resolution; on LLM weight distributions that trade is a consistent win.

Measured effect on mean relative weight error across the 169 quantized linears:

| Method | Mean relative weight error |
|---|---|
| Plain RTN (`alpha = 1.0`) | 13.31% |
| RTN + MSE-optimal clipping | **11.06%** |

Worst individual layer after clipping: `to_k` at **20.5%**. That is high, and it is the basis for
the quality caveat at the end of this document — the improvement is real, but 11.06% is not the
error level of a calibrated method.

### modelopt's actual role: format oracle, not quantizer

The on-disk W4A16 layout that `tensorrt-edgellm-export` consumes is a contract with no tolerance for
guesswork — get the scale convention or nibble order wrong and you get a model that loads, runs, and
emits garbage. modelopt was used to *read out* that contract rather than to perform the quantization.

On the development host, modelopt's CUDA extension is broken on an sm_120 GPU (it emits zeros in
fp32 and NaN in fp16), so `INT4_BLOCKWISE_WEIGHT_ONLY_CFG` was run on **CPU**, purely to extract
ground truth:

| Contract element | Value extracted |
|---|---|
| Scale convention | `scale = amax / 7`; `q = round(w / scale).clamp(-8, 7)` |
| Agreement with the reimplementation | matched to **3.7e-09** |
| Packed weight | `[N//2, K]` uint8, two int4 nibbles per byte — **even row = low nibble, odd row = high nibble** |
| Companion tensor | `<linear>.weight_scale`, fp32 `[N, K//128]`, per-output-channel per-group |

That 3.7e-09 agreement is what licenses replacing modelopt's quantizer with a ~45-line function
(`quantize_weight` in [`../scripts/rtn_int4_quantize.py`](../scripts/rtn_int4_quantize.py)): the
format is identical, only the scale-selection policy differs.

### Scope: what gets quantized, what does not

169 linears are quantized, selected by this regex in
[`../scripts/rtn_int4_quantize.py`](../scripts/rtn_int4_quantize.py):

```
^layers\.\d+\.(self_attn\.(to_q|to_k|to_v|to_out)|mlp\.(up_proj|down_proj))\.weight$|^lm_head\.weight$
```

Everything else stays FP16: the vision tower, the projector, `embed_tokens`, and all norms. This
mirrors the reference recipe's `exclude_modules`, which the script also writes into the emitted
config:

```json
["embed_tokens", "model.projector*", "model.visual*", "norm"]
```

Two reasons this exclude list matters beyond convention. First, `embed_tokens` is 536 MB of
`embedding.safetensors` that is read sparsely — quantizing it buys little bandwidth and costs
accuracy on rare tokens. Second, the vision tower is a separate engine entirely (938 MB FP16), so
text-tower quantization does not touch it; see Part 4.

### The 64-alignment rule

TensorRT-Edge-LLM skips INT4 for a linear when
`out_features % 64 != 0 or in_features % 64 != 0`
(`tensorrt_edgellm/quantization/quantize.py:1017-1029`). **Non-aligned layers are silently skipped**
— no warning, no error, just an FP16 layer inside what you believe is an INT4 model.

All Cosmos3-Edge text-tower linears pass this check, so nothing was dropped here. On a different
model, verify it explicitly: count `Int4GroupwiseGemmPluginV2` nodes in the exported ONNX (Gate 3)
and compare against the number of linears you expected to quantize.

The quantizer itself asserts a stricter pair of preconditions and will abort rather than emit a
malformed tensor:

- `N % 2 == 0` (required for nibble packing)
- `K % 128 == 0` (required for group-wise scales)

### The quantizer: real interface and outputs

```bash
python3 scripts/rtn_int4_quantize.py \
  --src /path/to/Cosmos3-Edge \
  --dst /path/to/cosmos3_int4_ckpt
```

Two arguments, both required, both directories. No GPU is used and no calibration data is needed;
it runs on CPU in fp32. No wall-clock runtime was recorded for this step in this deployment, so none
is quoted here.

What it writes into `--dst`:

| Output | Detail |
|---|---|
| Rewritten `*.safetensors` shards | Matched weights replaced by `uint8 [N//2, K]` packed tensors, each with a sibling `<name>.weight_scale` fp32 `[N, K//128]`. Unmatched tensors are copied through unchanged. Saved with `metadata={"format": "pt"}`. |
| All non-`.safetensors` files | Copied verbatim from `--src` (files via `shutil.copy2`, directories via `copytree`), so tokenizer, `visual/`, and configs come along. |
| `model.safetensors.index.json` | Rewritten so each new `weight_scale` key maps to the same shard as its weight. Skipped if the source has no index. |
| `hf_quant_config.json` | `{"quantization": {...}}` with `quant_algo: "W4A16_AWQ"`, `group_size: 128`, `has_zero_point: false`, `pre_quant_scale: false`, `kv_cache_quant_algo: null`, and the exclude list above. |
| `config.json` | The same dict injected as `quantization_config`. |

Note the declared `quant_algo` is `W4A16_AWQ`. That is the **format** name the loader dispatches on
(4-bit weights, 16-bit activations, group-wise scales, no zero point), and it is what makes the
checkpoint loadable. It is not a claim that AWQ's activation-aware scale search was performed — it
was not, and per Route B it cannot be in this toolchain. Stating this plainly matters if you publish
the checkpoint: anyone reading `W4A16_AWQ` will otherwise assume calibration happened.

Console output is the primary record of the run: a per-tensor line (first three, then every 40th)
of the form `[n] <key>: (N, K) -> packed (N//2, K) scale (N, K//128) relerr X.XX%`, and a final
`quantized 169 linears; mean rel err 11.06%`. Capture it — Gate 1 lives in that output.

---

## Part 3 — Four verification gates

A wrong quantizer does not crash. It produces a checkpoint of the right size and shape that loads
cleanly, builds cleanly, serves requests, and degrades quality in ways you will attribute to the
model rather than to your own tooling. Two of the failure modes actually encountered in this project
(modelopt's broken CUDA extension emitting zeros; `--quantization int4_awq` silently doing nothing)
were both *completely silent*. Each gate below is cheap; skipping them is not.

### Gate 1 — Dequant round-trip error, per tensor, before packing

Computed inside `quantize_weight` immediately after quantization and before any bit manipulation:

```python
rel = ((deq - w32).abs().mean() / w32.abs().mean().clamp(min=1e-12)).item()
```

This is the honest accuracy signal, and it is measured at the only point where the original fp32
tensor is still in hand. Expect per-tensor values in the low teens for this model; the aggregate was
11.06% mean, 20.5% worst (`to_k`). A tensor reporting ~0% or ~100% is the tell for an all-zeros or
all-clipped bug.

### Gate 2 — Nibble pack/unpack must reproduce the quantized tensor exactly

The script packs, then immediately unpacks with logic mirroring
`tensorrt_edgellm/checkpoint/loader.py` `_unpack_awq_prepacked`, and asserts bit-exact equality:

```python
u16 = packed.to(torch.int16) & 0xFF
back = torch.zeros(N, K, dtype=torch.int16)
back[0::2] = u16 & 0xF
back[1::2] = (u16 >> 4) & 0xF
assert torch.equal(back, qi), "nibble pack/unpack round-trip failed"
```

This is what catches a transposed nibble order — the single most likely mistake in this format, and
one that costs nothing at build time and everything at inference time. An assertion here is worth
more than any amount of downstream eyeballing, because swapped nibbles produce plausible-looking
noise, not an obvious failure.

### Gate 3 — Inspect the exported ONNX before building anything

After `tensorrt-edgellm-export` (Part 4) and before spending device time on a build, confirm on the
exported graph:

| Check | Expected for this model |
|---|---|
| `Int4GroupwiseGemmPluginV2` nodes | **169** — one per quantized linear |
| INT8 initializers at the cuteDSL fragment shape | 169 |
| Non-zero fraction of weight bytes | **100%** |
| `model.onnx.data` size | **0.807 GB** (FP16 reference: 3.36 GB) |

The node count is the direct test for the 64-alignment rule: any silently skipped linear shows up as
a missing plugin node. The non-zero check exists specifically because a broken quantizer backend
writes a perfectly well-formed file full of zeros.

### Gate 4 — Read `Total Weights Memory` from the engine build log

The final gate is the build log itself:

| Field | FP16 | INT4 | Ratio |
|---|---|---|---|
| TensorRT `Total Weights Memory` | 3,355,696,384 B | **865,480,704 B** | 3.88× |
| `llm.engine` on disk | 3,366,196,508 B (3.135 GB) | **878,552,564 B (0.818 GB)** | 3.83× |

If the engine is still ~3.1 GB, quantization did not take effect — which is exactly what Route A
produces. This is the check that makes the silent no-op loud.

---

## Part 4 — ONNX export and engine builds

Division of labor: **ONNX export runs on the host**, **engine builds run on the Jetson**. TensorRT
compiles per-GPU, so an engine built elsewhere is not portable to sm_87.

### Step 1 — Export ONNX (host)

```bash
tensorrt-edgellm-export \
  /path/to/cosmos3_int4_ckpt \
  /path/to/cosmos3_int4_onnx \
  --task reasoning \
  --skip-visual
```

`--task reasoning` is required for Cosmos3-Edge; this is also the workaround for the
`tensorrt-edgellm-quantize` → `KeyError: 'cosmos3_edge'` failure noted in Part 1.

The two positional arguments are `model` then `output_dir`, and both `--skip-visual` and `--skip-llm`
exist on this release — read from `tensorrt-edgellm-export --help` on the install used here. The
exporter creates `llm/`, `visual/` and `audio/` **subdirectories under `output_dir`** as needed, which
is where the `llm/` level in every path below comes from.

`--skip-visual` is the right call **when only the text tower changed**, which is the case here: the
vision tower is excluded from quantization, so its ONNX and its engine are bit-identical to what you
already have. Skipping it avoids re-exporting ~985 MB of external data (the vision tower's ONNX
external data is 984,940,544 bytes) and avoids a redundant ~938 MB engine build on the device. Drop
the flag on a first-time export, or whenever anything in `visual/` changes.

Then copy the **contents** of the ONNX output directory to the device, into the layout the deployed
shim expects (the same destination [`03-install-trt-edge-llm.md`](./03-install-trt-edge-llm.md) uses):

```bash
ssh "$JETSON_HOST" 'mkdir -p ~/tensorrt-edgellm-workspace/Cosmos3-Edge/onnx/reasoning'
rsync -avP /path/to/cosmos3_int4_onnx/ \
  "$JETSON_HOST":~/tensorrt-edgellm-workspace/Cosmos3-Edge/onnx/reasoning/
```

The trailing slash on the source matters: it copies `llm/` (and `visual/`, if you exported it) *into*
`.../onnx/reasoning/`, so the exporter's `llm/` lands at
`/home/orin/tensorrt-edgellm-workspace/Cosmos3-Edge/onnx/reasoning/llm` — exactly the `CHECKPOINT_DIR`
constant in [`../serve/cosmos3_shim.py`](../serve/cosmos3_shim.py). Copy the whole tree: ONNX external
data (`model.onnx.data`) is referenced by relative path from `model.onnx`.

Run Gate 3 against the exported ONNX before this copy — it is much cheaper to catch a bad export on
the host than after a device build.

### Step 2 — Build the LLM engine (on the Jetson)

```bash
ssh "$JETSON_HOST"
llm_build \
  --onnxDir   ~/tensorrt-edgellm-workspace/Cosmos3-Edge/onnx/reasoning/llm \
  --engineDir ~/tensorrt-edgellm-workspace/Cosmos3-Edge-INT4/engines/reasoning \
  --maxBatchSize 1 \
  --maxKVCacheCapacity 2048
```

`--onnxDir` points at the `llm/` subdirectory the exporter created, not at the export root. Install
the resulting `llm.engine` where the shim looks for it — `ENGINE_DIR` in
[`../serve/cosmos3_shim.py`](../serve/cosmos3_shim.py) is `/opt/tensorrt-edgellm/models/default` — or
build directly into that directory.

**On those two numbers.** They are not defaults chosen for taste — an engine built at
`--maxBatchSize 4 --maxKVCacheCapacity 4096` **OOMs at engine load** on this 8 GB board, even though
the build itself completes. The rebuild at batch 1 / KV 2048 is what fits. (For reference, the
Jetson AI Lab TensorRT-Edge-LLM Orin Nano tutorial recommends being even more conservative:
`maxInputLen 512`, `maxKVCacheCapacity 1024`.) These are recorded as the values used for the
deployed engine; the Jetson went offline before a final re-read of the built `config.json`, so treat
them as the build inputs rather than as freshly verified output.

If peak *build* memory is the binding constraint, the documented lever is:

```bash
llm_build ... --externalize-weights int4_ffn
```

This is the Jetson AI Lab tutorial's recommended way to cut peak build memory by keeping FFN weights
out of the in-memory build graph. It was **not needed in this deployment** — peak build CPU memory
was 3,884 MiB, which fit — so it is documented here as the known remedy rather than as something
measured on this model.

Expected success signals, in order of appearance:

| Signal | Expected value |
|---|---|
| Plugin nodes consumed from ONNX | 169 `Int4GroupwiseGemmPluginV2` |
| Build log `Total Weights Memory` | **865,480,704** bytes |
| Resulting `llm.engine` | 878,552,564 bytes ≈ **0.818 GB** |
| Peak build CPU memory | 3,884 MiB |

### Step 3 — Build the vision engine (on the Jetson)

Only required on a first build, or when something in `visual/` actually changes. Text-tower
quantization does not affect it.

```bash
visual_build \
  --onnxDir   ~/tensorrt-edgellm-workspace/Cosmos3-Edge/onnx/reasoning/visual \
  --engineDir ~/tensorrt-edgellm-workspace/Cosmos3-Edge-INT4/engines/reasoning
# defaults on this release: --minImageTokens 4  --maxImageTokens 1024  --maxImageTokensPerImage 512
# the engine is written to <engineDir>/visual/
```

The flag set and those defaults are read from `visual_build --help` on the Jetson. The precise
argument *values* used for the deployed vision engine were not recorded in this deployment's notes,
so they are not reproduced here rather than guessed. The artifact it produces is known: an FP16
engine of **938 MB** from **984,940,544 bytes** of ONNX external data, for a
SigLIP-So400m-class tower (27 layers, hidden 1152, intermediate 4304, 16 heads, patch 16, merge 2,
~411M params).

Two things not to attempt on this platform, both verified dead ends:

- `USE_TRT_NATIVE_ATTN=1` for fused ViT attention. It exports cleanly (27 `TRT_Attention` ops plus a
  `kv_lengths` input) but **requires TensorRT ≥ 11** (docstring at
  `tensorrt_edgellm/models/ops.py:384`); JetPack 7.2.1 ships 10.16.2.10. `visual_build` then fails
  with the misleading `[6] creator && "Plugin not found, are the plugin name, version, and namespace
  correct?"`. `libNvInfer_edgellm_plugin.so` provides `ViTAttentionPlugin`, not `TRT_Attention`.
- Narrowing the ViT optimization profile. `optHW = (minImageTokens + maxImageTokens)/2 × 4`
  (`cpp/builder/visualBuilder.cpp:319`), so with defaults (min 4, max 1024) the opt point is already
  514 image tokens — essentially the 512 operating point. It is a no-op.

### Step 4 — Plugin path at build and load time

A `Plugin not found` error at either engine build or engine load usually means the plugin library
was not located. Point the runtime at it:

```bash
export EDGELLM_PLUGIN_PATH=/home/orin/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so
# or, if that is not honored in your context:
export LD_PRELOAD=/home/orin/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so
```

The serving shim sets `EDGELLM_PLUGIN_PATH` itself via `os.environ.setdefault` — see
[`../serve/cosmos3_shim.py`](../serve/cosmos3_shim.py).

Note the distinction from the `TRT_Attention` case above: there, the identical error message is
*not* a path problem, and setting these variables will not help.

### Step 5 — Sanity check on load

With the INT4 engine in place, the resident runtime reports a shorter construction time than FP16 —
a weak but free signal that the smaller engine is the one being loaded:

| Engine | `LLMRuntime constructed in` |
|---|---|
| FP16 | 7.79 s |
| INT4 | **6.28 s** |

Serving, CUDA-graph capture, and the image-token budget are covered in the serving and measurement
documents; the full write-up is in [`../docs/report.md`](../docs/report.md).

---

## Caveats

State these alongside the numbers, not in a footnote.

- **Quantization quality is not fully characterized.** 11.06% mean relative weight error (worst
  layer `to_k` at 20.5%) from an uncalibrated method. Spot checks at greedy decoding produce
  coherent, grounded output with no repetition collapse or word salad — but **no benchmark suite was
  run**. Evaluate on a real task set before production use.
- **No FP16-vs-INT4 side-by-side on identical prompts.** Both engines cannot be resident within 8 GB
  simultaneously, so the quality comparison is INT4 spot checks against historical FP16 samples, not
  a controlled A/B.
- **`W4A16_AWQ` in the emitted config is a format label, not a provenance claim.** No activation
  calibration was performed, and per Route B none is possible in this toolchain.
- **`--maxBatchSize 1 --maxKVCacheCapacity 2048` are the values used**, recorded from the build
  invocation. The device went offline before the built `config.json` could be re-read, so they are
  not presented as freshly re-verified output.
- **`--externalize-weights int4_ffn` was documented, not exercised here.** It was not needed — peak
  build CPU memory was 3,884 MiB, which fit — so it is carried as the Jetson AI Lab tutorial's
  recommended remedy rather than as something measured on this model.
- **The `visual_build` flag set and its defaults come from `--help` on the device; the argument
  values used for the deployed vision engine were not recorded**, so the invocation above shows the
  surface, not a transcript of the build that produced the shipped 938 MB engine.
- **No wall-clock runtime was recorded for `scripts/rtn_int4_quantize.py`.** The only timings
  measured around this step are peak build CPU memory (3,884 MiB) and `LLMRuntime constructed in`
  (6.28 s INT4 / 7.79 s FP16).
