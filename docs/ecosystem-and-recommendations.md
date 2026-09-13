# Ecosystem Survey and Recommendations for `nvidia/cosmos-framework`

This document does two things:

1. Maps the repositories and documentation properties a developer must cross to deploy
   **Cosmos3-Edge on a Jetson Orin Nano Super 8 GB**, and states precisely what each one does and
   does not contain.
2. Proposes six concrete, actionable changes, led by an argument about *where* the guidance should
   live.

> **This is an independent engineering assessment, not an official NVIDIA position.** It was written
> by an engineer who deployed Cosmos3-Edge on an Orin Nano Super 8 GB and hit every gap described
> here. Nothing in it has been reviewed or endorsed by the teams that own the repositories discussed.
> The landscape survey reflects the state of those properties as observed in **September 2026**;
> upstream repos move, so re-check before citing.

All performance numbers referenced below come from this deployment and are documented, with
methodology and caveats, in [the full report](./report.md).

---

## Part 1 — The repository landscape

The guidance a developer needs to answer *"will Cosmos3-Edge fit on my 8 GB Jetson, and how fast
will it run?"* exists, but it is spread across at least six properties — and the repository named
"framework" contains none of it.

| Property | Role | Cosmos3-Edge edge/optimization content |
|---|---|---|
| [`nvidia/cosmos`](https://github.com/nvidia/cosmos) | Cosmos 3 platform hub | Documents Cosmos3-Edge (4B), Cosmos3-Nano (16B), Cosmos3-Super (64B) and the Reasoner/Generator runtime modes. Jetson coverage is **AGX Orin**; FP8/NVFP4 are listed as *coming soon*. Closest thing to a home for edge guidance today. |
| [`nvidia/cosmos-framework`](https://github.com/nvidia/cosmos-framework) | Training + inference framework | **None.** No Jetson, no quantization, no TensorRT, no edge content. Inference backends are Diffusers / Transformers / vLLM. Top-level dirs: `cosmos_framework/`, `examples/`, `packages/`, `docs/`. |
| [`NVIDIA/TensorRT-edge-llm`](https://github.com/NVIDIA/TensorRT-edge-llm) | The edge runtime | **Where the capability actually lives.** Targets Jetson / DRIVE / DGX Spark; Cosmos3-Edge is listed as supported. Provides INT4, NVFP4 and FP8 paths, CUDA graphs, custom plugins, speculative decoding (EAGLE, DDTree, DSpark) and multi-turn KV-cache reuse. |
| [`NVIDIA/TensorRT-Model-Optimizer`](https://github.com/NVIDIA/TensorRT-Model-Optimizer) | modelopt | INT4 AWQ, INT4 blockwise weight-only, NVFP4, FP8, sparsity, distillation, speculative decoding. Edge and Jetson are **not** called out as targets. |
| [jetson-ai-lab.com — Cosmos3 Edge](https://www.jetson-ai-lab.com/models/cosmos3-edge/) | Deployment guidance | Lists **Orin Nano 8 GB** as supported, but the guidance is **BF16 via HF Transformers** with **vLLM recommended**. Publishes AGX Orin 64 GB at **44.1 tok/s**; publishes **no Orin Nano performance numbers** and **no quantized path**. Correctly notes that single-stream decode is memory-bandwidth-bound. |
| [jetson-ai-lab.com — TensorRT Edge-LLM tutorial](https://www.jetson-ai-lab.com/tutorials/tensorrt-edge-llm/) | Runtime tutorial | The Orin Nano 8 GB worked example is **Qwen3-4B-Instruct with INT4 AWQ** — not Cosmos3-Edge. Recommends `maxInputLen 512` and `maxKVCacheCapacity 1024` for Orin Nano, and `--externalize-weights int4_ffn` to cut peak build memory. States engines must be built **on the target device**. |
| [`nvidia-cosmos/*`](https://github.com/nvidia-cosmos) — predict2.5, transfer2.5, reason2, [cosmos-cookbook](https://github.com/nvidia-cosmos/cosmos-cookbook) | Superseded by Cosmos 3 | Limited maintenance. The cookbook holds one adjacent recipe: Cosmos-Reason2 on Jetson AGX **Thor** with FP8 + TensorRT-Edge-LLM — a different model on a different, FP8-capable SoC. |

### What this means in practice

- The **model** docs (`nvidia/cosmos`) and the **runtime** docs (`TensorRT-edge-llm`) are each
  correct in isolation and neither is wrong. The problem is that the join between them —
  *this model, this precision, this board* — is not written down anywhere.
- The **framework** repo is the name a developer searching "cosmos framework inference" is most
  likely to land on, and it is the one property with zero edge content. Its inference story
  (Diffusers / Transformers / vLLM) routes an Orin Nano user toward a path that will not fit in
  8 GB.
- The **cookbook's** Jetson recipe uses FP8 on AGX Thor. FP8 is a hardware capability Orin
  (sm_87, Ampere) does not have, so that recipe is not transferable to Orin Nano even as a
  template — a distinction the reader has to already know to make.

---

## Part 2 — The documented gap, stated crisply

> The Jetson AI Lab Cosmos3-Edge page **lists Orin Nano 8 GB as supported** but gives **BF16-only
> guidance with no Orin Nano numbers and no quantized path**. The TensorRT-Edge-LLM Orin Nano
> tutorial **demonstrates INT4** — but on **Qwen3-4B**, not Cosmos3-Edge. **Nobody documents the
> intersection: Cosmos3-Edge, quantized, on Orin Nano** — which is exactly the configuration an
> 8 GB board requires.

Two independent signals that this gap is real and costly:

1. **BF16 does not leave usable headroom on 8 GB.** In this deployment, FP16 Cosmos3-Edge weights
   were 3.135 GB on disk, the resident process sat at 6.02 GB RSS, and the system had **472 MB**
   of RAM available with nothing else running. That configuration boots and answers, so it looks
   "supported", but it has no room for a camera pipeline, a web UI, or a second model. After INT4
   self-quantization the same process holds at 3.70 GB RSS with 2,705 MB available.
2. **The vacuum got filled by an unvalidated artifact.** A community checkpoint,
   [`ubr-physical-ai/Cosmos3-Edge-INT4-AWQ`](https://huggingface.co/ubr-physical-ai/Cosmos3-Edge-INT4-AWQ),
   was produced for a Jetson Orin Nano specifically to fill this gap. Its own documentation states
   the checkpoint **has never been executed**. This project declined to use it and read only its
   safetensors header metadata (tensor names, shapes, dtypes — no weight data) to confirm the
   on-disk packing contract. An unrun checkpoint is not a substitute for first-party guidance;
   it is evidence of demand for it.

The gap is not theoretical: closing it on this board produced **76.0 tok/s decode on an Orin Nano
8 GB at INT4 W4A16**, against the **44.1 tok/s** the Jetson AI Lab publishes for **AGX Orin 64 GB
at BF16 via vLLM**. Those are different boards, different runtimes, and different precisions — the
comparison is not apples to apples and should not be read as one. It does establish that the Orin
Nano configuration nobody documents is not a marginal one.

---

## Part 3 — Recommendations

### Recommendation 0 (leading): make `cosmos-framework` the index and contract owner, not a second copy

State the placement problem before the recommendations, because it changes what "put the guidance in
cosmos-framework" should mean.

`cosmos-framework` is currently a **training and inference framework**. Edge deployment is a
different audience with a different toolchain, and the capability itself — engine build, INT4
kernels, CUDA graphs, plugins — lives in `TensorRT-edge-llm`. Relocating runtime documentation into
`cosmos-framework` would create a second copy of fast-moving material maintained by a team that does
not own the code it describes. That copy will drift, and a drifted copy is worse than no copy,
because it is authoritative-looking and wrong.

The stronger move is to make `cosmos-framework` the **authoritative index and contract owner**:

- **Index:** the one page that answers *what is possible, on what hardware, at what cost, and where
  to go next*, with deep links into `TensorRT-edge-llm` and Jetson AI Lab for the how-to.
- **Contract owner:** the place that owns the things that are *model* properties rather than
  *runtime* properties, and therefore genuinely belong with the model — the per-device support
  matrix, the precision support matrix, the checkpoint provenance format, and the precondition
  table in Recommendation 3. These change when the model changes, not when the runtime changes.

This is an argument, not a hedge: a thin, accurate, well-maintained index that owns the
model-side contracts beats a thick copy of someone else's runtime docs. The two proposed artifacts
below (the support matrix and the precondition table) are deliberately chosen to be maintainable by
the framework repo without tracking runtime internals.

---

### 1. Publish a per-device support matrix with real numbers — including what does *not* fit

The single highest-value addition. Today a developer cannot answer "will this fit and how fast will
it be" without repeating the entire exercise this repository documents.

Proposed artifact (rows below are this deployment's measurements plus the one published official
number; an official version would be filled in by the owning team):

| Device | RAM | Precision | Runtime | LLM engine on disk | Process RSS | Decode | Source / status |
|---|---|---|---|---|---|---|---|
| Orin Nano Super | 8 GB | INT4 W4A16 | TensorRT-Edge-LLM | 0.818 GB | 3.70 GB | 76.0 tok/s | This deployment; self-quantized, RTN + MSE clipping |
| Orin Nano Super | 8 GB | FP16 | TensorRT-Edge-LLM | 3.135 GB | 6.02 GB | 22.5 tok/s | This deployment; runs, but only 472 MB RAM left |
| AGX Orin | 64 GB | BF16 | vLLM | not published | not published | 44.1 tok/s | Jetson AI Lab (official) |
| Orin Nano Super — Cosmos3-Edge-**Policy-DROID** | 8 GB | — | TensorRT-Edge-LLM | — | — | — | **Does not fit.** GEN engine OOMs with no other model resident |

Two properties make this table useful rather than decorative:

- **It records failures.** The Policy-DROID row is as valuable as any success row. A developer who
  learns in one line that a variant does not fit on 8 GB has been saved a multi-hour build.
- **It separates "boots" from "deployable."** The FP16 Orin Nano row runs. It also leaves 472 MB of
  system RAM, which is not a configuration anyone should ship a camera pipeline on. A matrix that
  reports only "supported / unsupported" cannot express that; one with an RSS column can.

Caveat to carry with any such table: these numbers are single-stream, batch-1, on the specific
JetPack/TensorRT/runtime versions recorded in [the report](./report.md). They are not a portable
performance guarantee.

---

### 2. Ship a validated, *executed* INT4 Cosmos3-Edge checkpoint — with provenance

The community filled this vacuum with an artifact that has, by its own admission, never been run.
A first-party checkpoint would displace it immediately, provided it publishes provenance rather
than just weights:

- **Method** (RTN / AWQ / GPTQ / other) and **group size**
- **Calibration data**, named — or an explicit statement that the method is calibration-free
- **Measured weight reconstruction error**, mean and worst layer
- **Benchmark scores on a named task suite**, so the accuracy cost is a number rather than a vibe
- **Exactly which modules are quantized** and which stay in higher precision

For reference, the equivalent disclosure for this deployment: calibration-free RTN with MSE-optimal
per-group clipping (α swept over `linspace(0.55, 1.0, 19)`), group size 128, **169 linears**
quantized (attention `to_q/to_k/to_v/to_out`, MLP `up_proj/down_proj`, `lm_head`), vision tower /
projector / `embed_tokens` / norms left FP16; mean relative weight error **11.06%**, worst layer
`to_k` at **20.5%**; **no benchmark suite was run** — that last item is a real gap in this work and
is exactly the field a first-party artifact should fill.

**If shipping weights is not viable, ship the recipe plus a verification script.** The quantizer
used here is [`scripts/rtn_int4_quantize.py`](../scripts/rtn_int4_quantize.py); the checks that
turned a plausible quantization into a verified one were:

1. Per-tensor dequantization round-trip relative error, computed before packing.
2. Nibble pack/unpack asserted to reproduce the quantized tensor **exactly**, mirroring the
   runtime's own `_unpack_awq_prepacked` loader path.
3. Exported ONNX inspected — node count and type, non-zero weight-byte fraction, external data size.
4. Engine build log's reported total weights memory compared against the FP16 build.

A shipped `verify_int4_checkpoint.py` that performs checks 1–3 against any candidate checkpoint
would let the community produce artifacts that are *provably* loadable, instead of hopefully
loadable.

---

### 3. Document hardware/software preconditions per optimization — and make them hard errors

Three of this project's dead ends were preconditions discovered only at failure time. Two failed
**silently**; one failed with an error message naming the wrong cause.

Proposed artifact — a precondition table attached to every documented technique:

| Technique | Requires | Fails on | Failure mode today |
|---|---|---|---|
| `USE_TRT_NATIVE_ATTN=1` (fused ViT attention) | TensorRT ≥ 11 | JetPack 7.2.1 (TensorRT 10.16) | Exports cleanly, then the visual build dies with `Plugin not found, are the plugin name, version, and namespace correct?` — which names the wrong cause |
| FP8 weights / FP8 KV cache | Ada / Hopper / Blackwell / Thor | Orin (sm_87) — no FP8 hardware | Not applicable; only FP8 KV quantization is implemented |
| NVFP4 | Blackwell / Thor | Orin | Not applicable |
| INT4 W4A16 | `out_features % 64 == 0 && in_features % 64 == 0` | Non-aligned layers | **Silently skipped** — layer stays FP16, no warning |
| `--quantization int4_awq` on the export CLI | Mixture-of-Experts models only | Dense models (Cosmos3-Edge is dense) | **Silent no-op** — the flag is accepted and dropped |
| AWQ calibration via modelopt | An eager PyTorch forward that produces real activations | TensorRT-Edge-LLM, where the attention plugin is a shape-only stub returning zeros | **Structurally impossible** — eager forward returns zeros from the stub, so no activation statistics can ever be collected and calibration cannot produce meaningful scales |

The documentation change is cheap. The upstream fix is the valuable half: **each of these should be
a hard error at export time**, not a silent pass. `--quantization int4_awq` on a dense model should
refuse to run rather than emit an unquantized checkpoint that looks quantized; a 64-alignment skip
should at minimum log per-layer; `USE_TRT_NATIVE_ATTN=1` should check the TensorRT version and say
so by name.

---

### 4. Document the measurement traps, or users will publish confident wrong numbers

Two traps in this stack invalidate naive benchmarks, and both produce *better-looking* numbers, so
neither announces itself:

- **The encoder embedding cache.** The runtime caches vision-encoder embeddings keyed on **raw pixel
  bytes**. A benchmark loop that sends the same test image repeatedly skips the entire ViT — roughly
  **248 ms** of real work in this configuration — from the second iteration onward. Every probe must
  generate a **unique image per iteration**;
  [`scripts/profile_fixed.py`](../scripts/profile_fixed.py) and
  [`scripts/tok_vs_res.py`](../scripts/tok_vs_res.py) do this deliberately.
- **Contention with live traffic.** The runtime serializes requests, so probes run alongside a live
  stream are inflated by concurrent work. In this project a "context-cache regression" was initially
  reported and then **retracted** — it was lock contention, not the cache. Under contention,
  **minimums** are the honest statistic.

Alongside the traps, publish the metric that works. Response length varies per request, so raw mean
latency is not comparable across runs. Regress instead:

```text
elapsed_ms = fixed_ms + marginal_ms_per_token × gen_tok
```

The `fixed` term is image preprocessing + ViT encode + prefill; the `marginal` term is per-token
decode. They respond to different optimizations, and an A/B that reports only mean latency cannot
tell you which one moved. When the client emits constant-length replies the regression degenerates
and R² collapses — fall back to **matching on `gen_tok`**, which is the stronger comparison anyway.
A measurement window spanning a config change mixes two fixed-cost regimes and will show a low R²;
that is expected, not a fault.

Reference implementations, both reading the target host from the `JETSON_HOST` environment variable
(default `orin@jetson.local`, never a hardcoded address):

```bash
export JETSON_HOST=orin@jetson.local

# Fit fixed_ms and marginal_ms/token over a window of live requests
python3 scripts/collect_perf.py "10 min ago"

# Matched-gen_tok A/B between two windows
python3 scripts/compare_perf.py "<A_start>" "<A_end>" "<B_start>"
```

See [`scripts/collect_perf.py`](../scripts/collect_perf.py) and
[`scripts/compare_perf.py`](../scripts/compare_perf.py).

---

### 5. Make the edge path discoverable from where developers actually land

`cosmos-framework`'s inference story is Diffusers / Transformers / vLLM. For an Orin Nano 8 GB user
that is a dead end: the BF16 path leaves 472 MB of system RAM in this deployment, and vLLM is not
the runtime that gets Cosmos3-Edge into 8 GB.

Concrete change: an **"Edge / Jetson deployment"** section in the `cosmos-framework` README that
(a) names the supported edge boards, (b) routes to `TensorRT-edge-llm` and the Jetson AI Lab pages,
and (c) states in one sentence that memory-constrained boards need a quantized path rather than the
default backends.

The physical reason this matters, not just the ergonomic one: single-stream decode on this class of
device is **memory-bandwidth-bound**. At FP16, decode moved 3.36 GB per token in 43.29 ms ≈
**77.6 GB/s** against the Orin Nano's ~102 GB/s peak — **76% of theoretical**. No scheduler or
sampling trick recovers meaningful time at that point; this project measured two such attempts at
−3.5% and −4.3% and rejected both. Shrinking the weights is the only lever with real headroom, which
is why precision selection belongs at the top of the edge guidance rather than in a footnote.

---

### 6. Close the Orin Nano × Cosmos3-Edge gap at both ends

Two specific documentation edits, one per property:

- **Jetson AI Lab — Cosmos3-Edge page:** add the Orin Nano 8 GB quantized path with real numbers, or
  state plainly that Orin Nano support means BF16 with very little headroom. Listing the board as
  supported while publishing no numbers for it is the ambiguity that sends developers down the vLLM
  path.
- **Jetson AI Lab — TensorRT-Edge-LLM tutorial:** add a **Cosmos3-Edge variant** of the existing
  Orin Nano walkthrough, which today demonstrates Qwen3-4B. The tutorial already carries the
  Orin-Nano-specific build advice (`maxInputLen 512`, `maxKVCacheCapacity 1024`,
  `--externalize-weights int4_ffn` for peak build memory, and building engines on the target device);
  a Cosmos3-Edge variant would mostly be swapping the model and adding the vision-tower steps.

Between those two edits, the intersection nobody documents today would be documented once, in the
place people already look.

---

## Scope and limits of this assessment

- **Independent, not official.** No part of this has been reviewed by NVIDIA. Treat the
  recommendations as a field report from one deployment, not as a roadmap.
- **Single device, single configuration.** Everything measured here is one Jetson Orin Nano Super
  8 GB on one JetPack/TensorRT/runtime combination, batch 1, single stream.
- **Quantization quality is not fully characterized.** The INT4 checkpoint behind the numbers cited
  above shows 11.06% mean relative weight error and was spot-checked for coherence at greedy
  decoding, but **no benchmark suite was run**. Anyone reusing this recipe should evaluate on a real
  task set first. This is the same disclosure gap Recommendation 2 asks upstream to close, and it
  applies to this work too.
- **The landscape survey is a point-in-time observation** (September 2026) of public repositories and
  documentation pages. Any of them may have changed since.

Full methodology, per-round results, negative results and caveats: [`docs/report.md`](./report.md).
Deployed artifacts: [`serve/cosmos3_shim.py`](../serve/cosmos3_shim.py),
[`systemd/cosmos3-edge-shim.service`](../systemd/cosmos3-edge-shim.service),
[`scripts/`](../scripts/).
