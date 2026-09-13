# Running Cosmos3-Edge Fast on a Jetson Orin Nano Super (8 GB)

**A measured optimization campaign: 22.5× end-to-end latency reduction and 2.32 GB of RAM reclaimed**

Author: asotelo@nvidia.com · Date: 2026-09-13
Platform: Jetson Orin Nano Super 8 GB · JetPack 7.2.1 (L4T R39.2.1) · CUDA 13.2 · TensorRT 10.16.2.10
Model: `nvidia/Cosmos3-Edge` (reasoner) · Runtime: TensorRT-Edge-LLM 0.10.1
Workload: Live VLM WebUI streaming a webcam into an OpenAI-compatible endpoint

---

## 1. Executive summary

Cosmos3-Edge is listed as supported on the Orin Nano 8 GB, but the official guidance stops at
BF16 through HuggingFace Transformers and publishes no performance numbers for that device.
Starting from a working-but-slow deployment, four measured optimization rounds produced:

| Metric | Before | After | Change |
|---|---|---|---|
| End-to-end request latency (mean) | 13.91 s | **0.619 s** | **22.5× faster** |
| Decode rate | 22.5 tok/s (44.43 ms/tok) | **76.0 tok/s** (13.16 ms/tok) | **3.38× faster** |
| Fixed cost (ViT encode + prefill) | ~348 ms | **287 ms** | −18% |
| Resident process RSS | 6.02 GB | **3.70 GB** | **−2.32 GB (−38.5%)** |
| System memory available | 472 MB | **2,705 MB** | **5.7×** |
| LLM engine on disk | 3.135 GB | **0.818 GB** | **3.83× smaller** |
| TensorRT weights memory | 3.125 GB | **0.806 GB** | **3.88× smaller** |

> The end-to-end **mean** is sample-dependent — it tracks the response-length distribution of the
> traffic in the window (619 ms over the clean n=88 post-change window; 626 ms over the larger
> n=699 window that also contains probe traffic). The regression coefficients are the stable
> quantities: **13.13–13.16 ms/token** marginal and **287–289 ms** fixed.


> **Read §8 before quoting these numbers.** The 22.5× is measured against a naive
> process-per-request baseline; against a competently configured FP16 resident baseline the
> honest figure is **3.38× decode + 2.32 GB RAM**. INT4 output quality was spot-checked at
> greedy decoding but **not benchmarked on a task suite**.

For context, the Jetson AI Lab model page reports **44.1 tok/s** decode for Cosmos3-Edge on an
**AGX Orin 64 GB** at BF16 via vLLM. This deployment reaches **76.0 tok/s on an Orin Nano 8 GB** —
the smaller, cheaper device — by moving to INT4 weights on the TensorRT-Edge-LLM runtime.

The headline RAM result matters more than the headline latency result. Going from **472 MB**
to **2.7 GB** of available system memory is the difference between a board that OOMs when you
add a second workload and one with real headroom. During this work a Home Assistant Core process
was observed resident alongside the VLM; DeepStream 9.1 is installed on the device but was not
exercised concurrently, so treat multi-workload co-hosting as plausible-and-partly-demonstrated
rather than benchmarked.

---

## 2. The system under test

```
Jetson Orin Nano Super 8 GB
├─ GPU      Ampere sm_87, 1020 MHz (MAXN_SUPER)
├─ Memory   8 GB LPDDR5 unified, ~102 GB/s peak, EMC 3199 MHz
├─ Software JetPack 7.2.1 / L4T R39.2.1, Ubuntu 24.04, CUDA 13.2,
│           TensorRT 10.16.2.10, cuDNN 9.20, DeepStream 9.1
└─ Model    Cosmos3-Edge reasoner
   ├─ Vision tower  SigLIP-So400m class: 27 layers, hidden 1152,
   │                intermediate 4304, 16 heads, patch 16, merge 2 (~411M params)
   └─ Text tower    ~1.68B params, 131072 vocab
```

Both towers run as separate TensorRT engines (`llm.engine`, `visual/visual.engine`) driven by a
single resident `LLMRuntime` instance.

---

## 3. Measurement methodology (and two traps that invalidate naive benchmarks)

Getting trustworthy numbers was harder than getting the speedups, and two specific traps produced
confidently wrong results before being caught.

### Trap 1: the encoder embedding cache

TensorRT-Edge-LLM caches vision-encoder embeddings keyed on **raw pixel bytes**
(`llmRankRuntime.cpp:1967`). Benchmarking by looping the same image skips the entire ViT — roughly
250 ms of real work — and reports a fantasy number. Every synthetic probe in this campaign
generates a **unique** image per iteration.

### Trap 2: contention with live traffic

The Live VLM WebUI streamed continuously throughout. The runtime serializes requests, so probe
latencies are inflated by whatever the WebUI is doing concurrently. A "context cache regression"
was initially reported and then retracted for exactly this reason — it was lock contention, not
the cache. Under contention, **minimums** are the honest statistic, because contention can only
ever add time.

### The metric that actually works

Response length varies request to request, so raw mean latency is not comparable across runs.
Instead, regress observed latency against generated-token count:

```
elapsed_ms = fixed_ms + marginal_ms_per_token × gen_tok
```

This cleanly separates the two things that respond to different optimizations:

- **`fixed`** — image preprocessing + ViT encode + prefill (attacked by Round 4)
- **`marginal`** — per-token decode (attacked by Round 3)

Final fit quality: **R² = 0.999** over n=699 live WebUI requests. When the WebUI emits
constant-length replies the regression degenerates, so `compare_perf.py` falls back to matching
on `gen_tok` — a stronger comparison when it is available.

Tooling written for this: `collect_perf.py` (regression) and `compare_perf.py` (matched-`gen_tok` A/B).

### Decomposing the fixed term

Knowing `fixed` was 47% of mean latency did not say *which* part to attack. Splitting it required two
probes, each using freshly generated images so the embedding cache could not interfere, and taking
**minimums** because live traffic was contending:

| Probe | Best latency |
|---|---|
| Text-only, prefill + 1 token | **39 ms** |
| Image (640×480, 300 image tokens), prefill + 1 token | **287 ms** |
| → implied **ViT encode** | **≈248 ms** |

So the ViT was **55% of the fixed term and 26% of mean latency** — the single largest remaining line
item, which is what made the image-token budget the right Round 4 target.

### Why the ViT was not kernel-tuned

Before attacking the ViT it was worth asking whether it was simply running badly. At 512 image tokens
(2048 patches) the vision tower does ≈**2.21 TFLOP** (1.68 TFLOP of GEMM + 0.52 TFLOP of attention).
At 248 ms that is **8.9 TFLOPS achieved, ≈53% of the Orin Nano Super's ~16.7 TFLOPS dense FP16 peak**.

Roughly half of peak is respectable for a real transformer encoder, which means the ViT was **not**
misconfigured and kernel-level tuning had little headroom to recover. That is why Round 4 reduced the
*amount of work* (fewer image tokens) rather than trying to make the same work faster — and it also
correctly predicted that narrowing the TensorRT optimization profile would be a no-op (see §6).

---

## 4. The optimization rounds

The campaign ran as a goal-seek loop: keep optimizing while each round beats a threshold
(started at 25%, lowered to 5%, raised to 10%).

| # | Change | Result | vs. bar |
|---|---|---|---|
| 0 | Baseline: `llm_inference` CLI spawned per request | 13.91 s/req | — |
| 1 | Resident runtime + CUDA graph + MAXN_SUPER + `jetson_clocks` | 2.07 s/req | **−85.1%** ✅ |
| 2 | Context-cache reuse, greedy decode, `top_k` bug fix | 43.29 ms/tok | −3.5…4.3% ❌ |
| 3 | INT4 W4A16 self-quantization (RTN + MSE-optimal clipping) | 13.38 ms/tok | **−69.1%** ✅ |
| 4 | Image-token budget 512 → 320 | 287 ms fixed | **−20.8%** ✅ |

### Round 1 — Stop paying startup costs (−85.1%)

The original path shelled out to the `llm_inference` binary per request, paying engine
deserialization (~6–8 s) every single time. Replacing it with a resident Python shim
(`serve/cosmos3_shim.py`) that holds one `LLMRuntime` for the process lifetime, plus:

- `capture_decoding_cuda_graph()` to eliminate per-token kernel launch overhead
- `nvpmodel -m 2` (MAXN_SUPER) + `jetson_clocks`: GPU 306→1020 MHz, CPU→1728 MHz, EMC 2133→3199 MHz

**13.91 s → 2.07 s.** This is the single largest win in the campaign and required no model changes.

### Round 2 — The micro-optimizations that didn't pay (−4%)

Context-cache reuse (−3.5%), greedy decode (−4.3%), and a genuine bug fix (`req.top_k` was never
set, leaving it zero-initialized instead of a validated 50) all landed below the threshold.

This round is worth reporting precisely *because* it failed: it established that decode was
**memory-bandwidth-bound**, not overhead-bound. At FP16 the model moved 3.36 GB per token in
43.29 ms ≈ **77.6 GB/s** against a ~102 GB/s theoretical peak. At ~76% of peak bandwidth, no
amount of scheduling cleverness helps. **The only lever with real headroom was making the
weights smaller.**

### Round 3 — INT4 self-quantization (−69.1%)

This is the technically substantial round. See §5 for the full path, including why the obvious
approaches were unavailable.

Result: **43.29 → 13.38 ms/token (3.24× at this round's endpoint; 3.38× against the final
13.13 ms/token measured after Round 4)**, engine 3.135 → 0.818 GB, RSS 6.02 → 3.70 GB.

### Round 4 — Right-sizing the image token budget (−20.8%)

With decode 3.2× faster, the profile inverted: the **FP16 vision tower became the bottleneck**,
at a 938 MB engine — larger than the INT4 text tower. The fixed term as a whole was 47% of mean
latency, of which the ViT encode alone was ~248 ms: **55% of the fixed term and 26% of mean latency**.

Measuring latency against image-token count showed a clean linear relationship, and — critically —
that the WebUI was pinned at the 512-token cap:

| Probe resolution | Image tokens | Latency |
|---|---|---|
| 320×240 | 80 | 90 ms |
| 448×336 | 140 | 152 ms |
| 640×480 | 300 | 288 ms |
| 896×672 | 512 (capped) | 499 ms |
| 1280×960 | 512 (capped) | 512 ms |

`max_image_tokens_per_image` is read **at runtime** from the engine's `visual/config.json` and
feeds `smartResize` (`imageUtils.cpp:145`, `maxPixels = tokens × 32²`). Lowering it 512 → 320 needs
**no engine rebuild** and is reversible in ~4 seconds.

Effect: a **constant −158 ms across every `gen_tok` bucket** — the exact signature of a fixed-cost
reduction, confirming the change touched prefill/ViT and not decode.

| gen_tok | n(before) | n(after) | before | after | delta |
|---|---|---|---|---|---|
| 17 | 13 | 3 | 672 ms | 516 ms | −156 ms |
| 20 | 60 | 4 | 714 ms | 556 ms | −158 ms |
| 22 | 70 | 5 | 738 ms | 582 ms | −156 ms |
| 34 | 31 | 8 | 899 ms | 740 ms | −159 ms |

**Weighted improvement: 20.8%.**

#### The quality tradeoff — measured, not assumed

Unlike rounds 1–3, this one changes model *input fidelity*, so it was A/B'd at greedy decoding:

- **Natural camera scene** (the actual Live VLM use case): output **byte-identical** at 320 and 512
  across three prompts, including one specifically probing fine detail.
- **Dense text screenshot**: 512 quoted on-screen annotations verbatim ("Large left lateral
  deviation", "Markers appear to be lagging behind in both video panel and 3D panel"); 320 drifted
  to a vaguer, partly-hallucinated description ("flight simulation or navigation system",
  generic X/Y/Z axes).

**Recommendation: 320 for live camera work, 512 for document/screenshot reading.** Currently
deployed at 320, with the original config preserved at `config.json.512.bak`.

---

## 5. How the INT4 quantization was actually done

The interesting part is that all three obvious routes were closed, and why.

### Route A — `--quantization int4_awq` on the export CLI ❌

The flag exists but is gated behind `_needs_moe_quantization` (`export.py:1046,1074`) and dropped
earlier still at `export.py:4129-4131`. **It only applies to Mixture-of-Experts models.**
Cosmos3-Edge is dense, so the flag silently does nothing.

### Route B — modelopt AWQ calibration ❌

AWQ derives per-channel scales from observed *activations*, which requires running the model
forward in eager PyTorch. In TensorRT-Edge-LLM, `attention_plugin` is a `torch.library.custom_op`
whose body is **`return torch.zeros(...)`** (`ops.py:87-193`) — a shape-only stub for ONNX export.
Eager forward produces zeros, so no meaningful activation statistics can ever be collected.
**AWQ calibration is structurally impossible in this toolchain.** (A separate attempt on Qwen
confirmed the cost even where it does work: ~4 hours and a GPU OOM.)

### Route C — download a pre-quantized community checkpoint ❌ (declined)

`ubr-physical-ai/Cosmos3-Edge-INT4-AWQ` on HuggingFace is a W4A16_AWQ g128 quantization of
Cosmos3-Edge produced for a Jetson Orin Nano during the NVIDIA/OpenHackathons/Oracle Open Models
Codefest 2026. It was declined for this deployment, and the repo's own documentation vindicates
that: **the checkpoint has never been executed.**

Its safetensors **header metadata** was read (tensor names, shapes, dtypes — no weight data) to
confirm the expected on-disk layout. That is the only way this artifact was used.

### Route D — calibration-free RTN with MSE-optimal clipping ✅

Round-to-nearest needs no activation data at all, sidestepping the blocker entirely.

**modelopt's role here was as the format oracle, not the quantizer.** Its CUDA extension is broken
on the host's sm_120 GPU (emits zeros in fp32, NaN in fp16), so `INT4_BLOCKWISE_WEIGHT_ONLY_CFG`
was run on **CPU** purely to extract ground truth for the on-disk contract:

- Scale convention: `scale = amax / 7`, `q = round(w/scale).clamp(-8, 7)` — **matched to 3.7e-09**
- Packing: `[N//2, K]` uint8, two int4 nibbles per byte (even row low, odd row high)
- Companion `weight_scale`: fp32 `[N, K//128]`, per-output-channel per-group

Plain RTN gave 13.31% mean relative weight error. Adding a **calibration-free MSE-optimal clipping
search** — sweeping a per-group scale multiplier α ∈ [0.55, 1.0] and keeping whichever minimizes
squared reconstruction error — cut that to **11.06%**. Shrinking the range trades clipping error
for finer resolution, which is a strict win on outlier-heavy LLM weights.

169 linears were quantized (attention `to_q/to_k/to_v/to_out`, MLP `up_proj/down_proj`, `lm_head`).
The vision tower, projector, embeddings, and norms stayed FP16 — mirroring the reference
`exclude_modules` recipe.

Verification at each stage, because a silent quantization bug is invisible until quality collapses:

1. Dequant round-trip error computed per tensor before packing
2. Nibble pack/unpack asserted to reproduce the quantized tensor **exactly**
3. Exported ONNX confirmed to carry **169 `Int4GroupwiseGemmPluginV2` nodes** and **100% non-zero**
   weight bytes (guarding against the all-zeros failure mode seen with the broken CUDA ext)
4. Engine build reported `Total Weights Memory: 865,480,704` vs `3,355,696,384` — a **3.88×** reduction

**Quality outcome:** coherent, well-grounded descriptions with no repetition collapse or word
salad. 11.06% mean weight error is high enough to warrant the caveat in §8.

---

## 6. Negative results worth publishing

These cost real time to establish and would save the next person the same effort.

| Attempt | Outcome | Root cause |
|---|---|---|
| `USE_TRT_NATIVE_ATTN=1` for fused ViT attention | **Blocked** | Emits `TRT_Attention`, which **requires TensorRT ≥ 11** (`ops.py:384`). JetPack 7.2.1 ships **10.16.2.10**. Export succeeds; the build fails with the misleading *"Plugin not found, are the plugin name, version, and namespace correct?"* |
| KV-cache quantization | **Not applicable** | Only FP8 is implemented; Orin is sm_87 Ampere with **no FP8 hardware** |
| Narrowing the ViT optimization profile | **No-op** | `optHW = (minImageTokens + maxImageTokens)/2 × 4` → with defaults, opt is already **514 image tokens** ≈ the 512 operating point |
| Cosmos3-Edge-Policy-DROID | **Non-viable** | GEN engine OOMs on 8 GB even with no other model resident |
| Qwen VLM alternatives | **Abandoned** | `--quantization` no-op on dense models, GPU OOM, ~4 h AWQ calibration |
| Engine at batch=4 / KV=4096 | **OOM** | Rebuilt at `maxBatchSize 1`, `maxKVCacheCapacity 2048` |

The first row is the most actionable: a supported, documented environment variable that produces a
clean export and then fails at build time with an error naming the wrong root cause. A TensorRT
version precondition check at export time would have saved the entire round.

---

## 7. Where the optimizations came from

The optimizations came from three distinct sources with very different reliability: NVIDIA's own
platform and runtime, modelopt, and a community artifact.

### NVIDIA official (highest value, well documented)
- **`nvpmodel` MAXN_SUPER + `jetson_clocks`** — free, no model changes, large effect
- **TensorRT-Edge-LLM resident C++ runtime** — the `LLMRuntime` object, CUDA graph capture,
  `Int4GroupwiseGemmPluginV2` (cuteDSL), the encoder embedding cache
- **Engine build tuning** — `maxBatchSize`, `maxKVCacheCapacity`, `--externalize-weights int4_ffn`
- **Jetson AI Lab tutorial** — the INT4-AWQ-on-Orin-Nano workflow (demonstrated on Qwen3-4B)

### modelopt (essential, but not in the role you'd expect)
Used as an **executable specification** of the W4A16 on-disk format rather than as the quantizer.
Given that its CUDA path was broken on the host GPU and its AWQ path was blocked by the zero-stub
attention op, its durable value here was providing a verifiable ground truth for scale convention
and packing layout. That is a genuinely useful role, and an underdocumented one.

### Community (informative, not deployable)
`ubr-physical-ai/Cosmos3-Edge-INT4-AWQ` proved the *shape* of the answer — W4A16_AWQ, group size
128, ONNX published alongside, engine built on-device. But an unexecuted checkpoint is a
hypothesis, not a result. Reading its header metadata to confirm the packing contract was the
appropriate level of trust to extend.

---

## 8. Caveats and honest limits

- **Quantization quality is not fully characterized.** 11.06% mean relative weight error (worst
  layer: `to_k` at 20.5%) from uncalibrated RTN. Spot checks at greedy decoding look clean, and
  the model remains coherent and grounded, but **no benchmark suite was run**. Before production
  use, evaluate on a real task set.
- **No FP16-vs-INT4 side-by-side on identical prompts.** Both engines cannot be resident within
  8 GB simultaneously, and tearing down the running deployment to A/B was not justified. The
  quality evidence is INT4-only plus historical FP16 samples.
- **The 20.8% from Round 4 is a quality tradeoff**, not a free win — demonstrated to be lossless
  on natural scenes and lossy on dense text (§4).
- **Measurements come from live WebUI traffic**, which is the right workload but is subject to
  contention; minimums and matched-`gen_tok` comparisons were used to control for it.
- The 22.5× headline compares against a genuinely naive baseline (process-per-request). Against a
  competently configured FP16 resident baseline, the honest figure is the **3.38× decode speedup**
  plus **2.32 GB RAM**.

---

## 9. Repository landscape for Cosmos3-Edge optimization

The guidance a developer needs is real, but it is scattered across at least six properties, and
the one repo named "framework" contains none of it.

| Repository | Role | Cosmos3-Edge optimization content |
|---|---|---|
| [`nvidia/cosmos`](https://github.com/nvidia/cosmos) | Cosmos 3 platform hub | Documents Cosmos3-Edge (4B) and Jetson AGX Orin; FP8/NVFP4 listed as *coming soon*. Closest thing to a home. |
| [`nvidia/cosmos-framework`](https://github.com/nvidia/cosmos-framework) | Training + inference framework | **None.** No Jetson, no quantization, no TensorRT, no edge. Inference backends are Diffusers / Transformers / vLLM. |
| [`NVIDIA/TensorRT-edge-llm`](https://github.com/NVIDIA/TensorRT-edge-llm) | The actual edge runtime | **Where the real capability lives**: Cosmos3-Edge support, INT4/NVFP4/FP8, CUDA graphs, speculative decoding, plugins |
| [`NVIDIA/TensorRT-Model-Optimizer`](https://github.com/NVIDIA/TensorRT-Model-Optimizer) | modelopt | INT4 AWQ / blockwise weight-only / NVFP4 / FP8. Edge + Jetson not called out. |
| [jetson-ai-lab.com](https://www.jetson-ai-lab.com/models/cosmos3-edge/) | Deployment guidance | Lists Orin Nano 8 GB, but **BF16 via HF Transformers, vLLM recommended, no Orin Nano perf numbers, no quantized path** |
| [`nvidia-cosmos/*`](https://github.com/nvidia-cosmos) (predict2.5, transfer2.5, reason2, cosmos-cookbook) | Superseded by Cosmos 3 | Limited maintenance. Cookbook has one adjacent recipe: Cosmos-Reason2 on **Jetson AGX Thor** with FP8 + TensorRT-Edge-LLM. |

**The concrete gap this project hit:** the Jetson AI Lab Cosmos3-Edge page lists Orin Nano 8 GB as
supported but gives BF16-only guidance and no numbers; the TensorRT-Edge-LLM Orin Nano tutorial
demonstrates INT4 AWQ but on **Qwen3-4B**, not Cosmos3-Edge. Nobody documents the intersection —
*Cosmos3-Edge, quantized, on Orin Nano* — which is exactly the configuration an 8 GB board needs,
and the one an unexecuted community checkpoint was attempting to fill.

---

## 10. Recommendations for `github.com/nvidia/cosmos-framework`

How could cosmos-framework become the single place for Cosmos3-Edge optimization guidance?
A caveat first, then the recommendations.

**Caveat on placement.** cosmos-framework is currently a *training and inference framework*
(FSDP/TP/CP/PP, DCP checkpoints, vLLM serving). Edge deployment guidance is a genuinely different
audience, and the capability itself lives in TensorRT-Edge-LLM. The strongest version of this is
**not** to relocate the runtime docs, but to make cosmos-framework the **authoritative index and
contract owner** — the place that tells you what is possible, what it costs, and where to go —
with deep links out. A thin, accurate, well-maintained index beats a thick copy that drifts.

### 10.1 Publish a per-device support matrix with real numbers

The single highest-value addition. Today a developer cannot answer "will this fit and how fast
will it be" without doing what this project did.

| Device | RAM | Precision | Runtime | Weights | Peak RSS | Decode | Status |
|---|---|---|---|---|---|---|---|
| Orin Nano Super | 8 GB | INT4 W4A16 | TRT-Edge-LLM | 0.81 GB | 3.70 GB | 76 tok/s | ✅ this report |
| Orin Nano Super | 8 GB | FP16 | TRT-Edge-LLM | 3.13 GB | 6.02 GB | 22.5 tok/s | ⚠️ 472 MB headroom |
| AGX Orin | 64 GB | BF16 | vLLM | — | — | 44.1 tok/s | official |

Include the **failures**: Cosmos3-Edge-Policy-DROID does not fit on 8 GB. Knowing what *doesn't*
work is as valuable as knowing what does.

### 10.2 Ship a validated, executed INT4 Cosmos3-Edge checkpoint

The community filled this vacuum with an artifact that has **never been run**. A first-party
checkpoint with published provenance (method, calibration data or explicitly none, measured
weight error, benchmark scores on a named task set) would displace it immediately.

If shipping weights is not viable, ship the **recipe plus a verification script** that asserts the
packing contract and reports reconstruction error — the checks in §5 that turned a plausible
quantization into a verified one.

### 10.3 Document the hardware/software preconditions per optimization

Every technique should carry an explicit precondition table, because three of this project's dead
ends were preconditions discovered only at failure time:

| Technique | Requires | Fails on |
|---|---|---|
| `USE_TRT_NATIVE_ATTN=1` | **TensorRT ≥ 11** | JetPack 7.2.1 (TRT 10.16) — misleading "Plugin not found" |
| FP8 weights / KV cache | Ada / Hopper / Blackwell / Thor | Orin (sm_87) — no FP8 hardware |
| NVFP4 | Blackwell / Thor | Orin |
| INT4 W4A16 | `out%64==0 && in%64==0` | non-aligned layers silently skipped |
| `--quantization int4_awq` | **MoE models only** | dense models — **silent no-op** |

Two of these fail *silently* and one fails with an error naming the wrong cause. Upstream, these
should be hard errors at export time.

### 10.4 Document the measurement traps

Publish the benchmarking guidance from §3 — the embedding cache defeating repeated-frame probes,
and the fixed-vs-marginal decomposition. Without these, users will generate confident, wrong
numbers, and any support burden that follows will be spent relitigating measurements.

### 10.5 Make the edge path discoverable from where people land

Add to cosmos-framework's README an "Edge / Jetson deployment" section that routes to
TensorRT-Edge-LLM and the Jetson AI Lab pages. Today the framework's inference story is
Diffusers/Transformers/vLLM, which points an Orin Nano user at a path that will not fit.

### 10.6 Close the specific Orin Nano × Cosmos3-Edge gap

Extend the Jetson AI Lab Cosmos3-Edge page with the Orin Nano 8 GB quantized path, and add a
Cosmos3-Edge variant of the TensorRT-Edge-LLM Orin Nano tutorial (which currently demonstrates
Qwen3-4B). Between them, that is the exact intersection nobody documents today.

---

## 11. Reproducing this

```bash
# 1. Platform (free, do this first — largest effort-to-payoff ratio after the resident runtime)
sudo nvpmodel -m 2 && sudo jetson_clocks          # MAXN_SUPER

# 2. Self-quantize the text tower to INT4 W4A16 (calibration-free, CPU, ~minutes)
python3 rtn_int4_quantize.py --src <Cosmos3-Edge ckpt> --dst cosmos3_int4_ckpt

# 3. Export INT4 ONNX (host), then build the engine ON the Jetson (TRT compiles per-GPU)
tensorrt-edgellm-export cosmos3_int4_ckpt cosmos3_int4_onnx --task reasoning --skip-visual
llm_build --onnxDir .../llm --engineDir .../reasoning \
          --maxBatchSize 1 --maxKVCacheCapacity 2048

# 4. Serve from a resident runtime, never process-per-request
systemctl start cosmos3-edge-shim.service

# 5. Right-size the image budget (runtime config, no rebuild, reversible)
#    visual/config.json -> builder_config.max_image_tokens_per_image = 320

# 6. Measure honestly
python3 collect_perf.py "10 min ago"                       # fixed vs marginal
python3 compare_perf.py <A_start> <A_end> <B_start>        # matched-gen_tok A/B
```

**Artifacts:** `rtn_int4_quantize.py` (quantizer), `collect_perf.py` / `compare_perf.py`
(measurement), `serve/cosmos3_shim.py` (resident OpenAI-compatible server).

---

## Sources

- [NVIDIA Cosmos](https://github.com/nvidia/cosmos)
- [NVIDIA cosmos-framework](https://github.com/nvidia/cosmos-framework)
- [NVIDIA TensorRT-edge-llm](https://github.com/NVIDIA/TensorRT-edge-llm)
- [NVIDIA TensorRT-Model-Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
- [Jetson AI Lab — TensorRT Edge-LLM on Jetson](https://www.jetson-ai-lab.com/tutorials/tensorrt-edge-llm/)
- [Jetson AI Lab — Cosmos3 Edge](https://www.jetson-ai-lab.com/models/cosmos3-edge/)
- [nvidia-cosmos organization](https://github.com/nvidia-cosmos)
- [nvidia-cosmos/cosmos-cookbook](https://github.com/nvidia-cosmos/cosmos-cookbook)
- [ubr-physical-ai/Cosmos3-Edge-INT4-AWQ](https://huggingface.co/ubr-physical-ai/Cosmos3-Edge-INT4-AWQ)
