# Optimization catalogue

Every optimization that made it into the deployed configuration, grouped by **where it came from**.
The grouping is deliberate: a technique published by NVIDIA in the runtime source, a technique
borrowed from a library used outside its intended role, and a technique read off an artifact that
was never executed do not deserve the same level of trust, and the difference should be visible in
the documentation rather than buried in a commit message.

Target reader: you have a Jetson, you have never used TensorRT-Edge-LLM, and you want to know which
of these levers is worth your afternoon.

## Ground rules for the numbers below

- Every measurement comes from one device: **Jetson Orin Nano Super 8 GB**, JetPack 7.2.1 /
  L4T R39.2.1, CUDA 13.2, TensorRT 10.16.2.10, TensorRT-Edge-LLM 0.10.1, model
  `nvidia/Cosmos3-Edge` (reasoner), batch 1, single stream. Nothing here is a multi-device result.
- The workload is live Live-VLM-WebUI traffic (a webcam streaming into an OpenAI-compatible
  endpoint). That is the right workload for this deployment, but the runtime serializes requests,
  so numbers taken under concurrent traffic are inflated. Minimums and matched-`gen_tok`
  comparisons are used to control for that; see [report.md](./report.md) §3.
- Where a detail was not verified in this deployment, it says so in the same sentence as the claim.
- Commands reference the board through `JETSON_HOST`. Set it once:

```bash
export JETSON_HOST=orin@jetson.local   # or your board's user@host
```

`collect_perf.py` and `compare_perf.py` read the same variable and default to
`orin@jetson.local`; `profile_fixed.py`, `tok_vs_res.py` and `quality.py` run on the device
against loopback.

---

## 1. The four rounds

The campaign ran as a goal-seek loop: keep going while each round beats a threshold. The threshold
started at 25%, was lowered to 5%, and was later raised to 10%. Round 2 is below every
version of that bar.

| # | Change | Result | Verdict |
|---|---|---|---|
| 0 | Baseline: `llm_inference` CLI spawned per request | 13.91 s/req | baseline |
| 1 | Resident runtime + CUDA graph + MAXN_SUPER + `jetson_clocks` | 2.07 s/req | **−85.1% PASS** |
| 2 | Context-cache reuse, greedy decode, `top_k` fix | 43.29 ms/tok | −3.5…4.3% **FAIL** |
| 3 | INT4 W4A16 self-quantization (RTN + MSE-optimal clipping) | 13.38 ms/tok | **−69.1% PASS** |
| 4 | Image-token budget 512 → 320 | 287 ms fixed | **−20.8% PASS** |

End state, against the round-0 baseline:

| Metric | Before | After | Change |
|---|---|---|---|
| End-to-end mean latency | 13.91 s | 619 ms | 22.5× |
| Decode marginal | 44.43 ms/tok (22.5 tok/s) | 13.13–13.16 ms/tok (76.0 tok/s) | 3.38× |
| Fixed cost (ViT + prefill) | ~348 ms | 287–289 ms | −18% |
| Resident process RSS | 6.02 GB | 3.70 GB | −2.32 GB (−38.5%) |
| System RAM available | 472 MB | 2,705 MB | 5.7× |
| LLM engine on disk | 3,366,196,508 B | 878,552,564 B | 3.83× |
| TensorRT "Total Weights Memory" | 3,355,696,384 B | 865,480,704 B | 3.88× |

Final regression fit: marginal **13.13 ms/token**, fixed **289 ms**, R² = 0.999 over n = 699 live
requests.

**Read the 22.5× honestly.** It is measured against a naive process-per-request baseline that
re-deserialized the engines on every request. Against a competently configured FP16 *resident*
baseline the honest figure is **3.38× on decode plus 2.32 GB of RAM reclaimed** — still the result
that matters on an 8 GB board, but a much smaller number than the headline.

### Round 2 failed its threshold, and is the most useful round in the table

Context-cache reuse measured −3.5%. Greedy decode measured −4.3%. A genuine bug was also fixed
along the way (`req.top_k` was never set on the request object, so it was zero-initialized instead
of carrying a validated 50). All of it landed below the acceptance bar, and none of it was retained
as a "win".

It is reported here because it is what told us where to go next. At FP16 the model moved **3.36 GB
of weights per decoded token in 43.29 ms ≈ 77.6 GB/s**, against the board's **~102 GB/s peak** —
**76% of theoretical bandwidth**. Once a decode loop is at 76% of memory bandwidth, scheduling
tricks, cache reuse and sampler tweaks have nothing left to recover; the only remaining lever is
moving fewer bytes per token. Round 2's failure is what made Round 3 (INT4) the obvious next step
rather than a guess, and it is also what rules out a whole family of "try harder at scheduling"
ideas for anyone reproducing this.

A second lesson from the same round: a "context-cache regression (4.98 s)" was reported and then
**retracted**. It was lock contention from concurrent live traffic, not the cache.

---

## 2. Provenance at a glance

| Optimization | Source | Round | Measured effect | Quality |
|---|---|---|---|---|
| `nvpmodel -m 2` (MAXN_SUPER) + `jetson_clocks` | NVIDIA platform | 1 | bundled in −85.1% | neutral (power/thermal cost) |
| Resident `LLMRuntime` (no process-per-request) | NVIDIA runtime | 1 | bundled in −85.1% | neutral |
| `capture_decoding_cuda_graph()` | NVIDIA runtime | 1 | bundled in −85.1% | neutral |
| Engine build sizing (`maxBatchSize 1`, `maxKVCacheCapacity 2048`) | NVIDIA runtime | 1/3 | made the engine loadable at all | capacity limit |
| `Int4GroupwiseGemmPluginV2` (cuteDSL) | NVIDIA runtime | 3 | mechanism behind −69.1% decode | see INT4 caveat |
| Encoder embedding cache | NVIDIA runtime | — | skips ~248 ms ViT on byte-identical frames | neutral; benchmarking hazard |
| W4A16 on-disk format contract | modelopt (as oracle) | 3 | no runtime effect; correctness | neutral |
| Packing/scale layout confirmation | Community checkpoint (metadata only) | 3 | no runtime effect; corroboration | neutral |
| Calibration-free RTN + MSE-optimal clipping | This project | 3 | 43.29 → 13.38 ms/tok; 3.88× weights | **tradeoff, uncharacterized** |
| Image-token budget 512 → 320 | This project | 4 | constant −158 ms fixed cost (−20.8%) | **tradeoff, characterized** |

"Bundled in −85.1%" is literal: Round 1 changed three things at once and the individual
contributions were never separated. The resident runtime is certainly the dominant term — it
removes a ~6–8 s engine deserialization from every request — but this campaign cannot tell you how
much of Round 1 was the clocks.

---

## 3. Source 1 — NVIDIA official (platform and runtime)

This is the highest-trust group: the code is public, the mechanism is inspectable in the runtime
source, and every item here was exercised directly.

### 3.1 `nvpmodel -m 2` (MAXN_SUPER) + `jetson_clocks`

**What it does.** Selects the board's maximum power profile and then pins clocks to their caps
instead of letting the governors ramp. Recorded transitions on this board: GPU **306 → 1020 MHz**,
CPU → **1728 MHz**, EMC **2133 → 3199 MHz**. The EMC change matters most here, because decode is
memory-bandwidth-bound (§1).

**Measured effect.** Not isolated — applied together with the resident runtime in Round 1
(13.91 s → 2.07 s). Treat this as "free and certainly positive", not as a quantified win.

**Tradeoff.** Higher sustained power draw and a hotter board; `jetson_clocks` does not survive
reboot unless you arrange for it. No effect on model output.

**Verify.**

```bash
ssh "$JETSON_HOST" 'sudo nvpmodel -q'          # expect mode 2 / MAXN_SUPER
ssh "$JETSON_HOST" 'sudo jetson_clocks --show' # GPU should report 1020 MHz, EMC 3199 MHz
```

### 3.2 The resident C++ runtime (one `LLMRuntime` for the process lifetime)

**What it does.** TensorRT-Edge-LLM ships a `llm_inference` CLI that is convenient and a set of
pybind11 bindings that are not advertised as loudly. The original deployment shelled out to the
CLI per request, paying engine deserialization (~6–8 s) every single time. The replacement is a
small FastAPI shim ([`../serve/cosmos3_shim.py`](../serve/cosmos3_shim.py)) that constructs one
`rt.LLMRuntime(...)` at startup and serves every request against it, behind a single-slot executor
and an `asyncio` lock because the runtime serializes anyway.

**Measured effect.** The dominant term in Round 1's 13.91 s → 2.07 s. Construction cost is now paid
once per process: **6.28 s** for the INT4 engine (7.79 s for the FP16 one).

**Tradeoff.** Quality-neutral. Operationally it trades crash isolation for speed — a bad request
that takes down the runtime takes down the service, which is why the unit file uses
`Restart=on-failure`. Requests are serialized, so concurrency is queueing, not parallelism.

**Verify.**

```bash
ssh "$JETSON_HOST" "journalctl -u cosmos3-edge-shim.service --no-pager | grep '\[shim\]'"
```

Expect the startup sequence the shim prints: `LLMRuntime constructed in <n>s`, `CUDA graphs
captured in <n>s`, `warm-up inference in <n>s`, `ready` — once, at process start, and never again
per request. If you see it repeat, something is restarting the process and you are back on the slow
path.

Note that [`../systemd/cosmos3-edge-shim.service`](../systemd/cosmos3-edge-shim.service) launches
the deployed copy at `/opt/tensorrt-edgellm/cosmos3_shim_v1.py`; the file in this repo is the same
program, published under a cleaner name.

### 3.3 CUDA graph capture for the decode loop

**What it does.** `_runtime.capture_decoding_cuda_graph()` records the per-token decode kernel
sequence once and replays it as a single graph launch, removing per-token launch overhead from the
critical path. One line, called immediately after construction.

**Measured effect.** Not isolated from the rest of Round 1. The later bandwidth analysis (§1) shows
why it could not have been large on its own: by Round 2 the decode loop was already at 76% of peak
memory bandwidth, which bounds how much launch overhead could still have been present.

**Tradeoff.** Quality-neutral. Costs a capture step at startup (logged) and fixes the decode shape,
which is a non-issue at batch 1.

**Verify.** The `CUDA graphs captured in <n>s` line in the shim journal (same command as §3.2).

### 3.4 `Int4GroupwiseGemmPluginV2` (cuteDSL) — the thing that makes INT4 fast

**What it does.** This is the runtime-side half of Round 3. The export path replaces each quantized
linear with a plugin node that consumes the packed 4-bit weights and their per-group FP32 scales
directly, dequantizing inside the GEMM rather than materializing an FP16 copy of the weights. It is
the reason a W4A16 checkpoint turns into a smaller *resident* model and not just a smaller file on
disk.

**Preconditions worth knowing before you plan a quantization.** INT4 is silently skipped for a
layer when `out_features % 64 != 0 or in_features % 64 != 0`
(`tensorrt_edgellm/quantization/quantize.py:1017-1029`). All Cosmos3-Edge layers pass, so this cost
nothing here — but "silently skipped" means a model with unaligned shapes will quantize, build, and
simply be bigger than you expected with no error.

**Measured effect.** With 169 plugin nodes in the graph: engine **3,366,196,508 → 878,552,564 B**
on disk, TensorRT "Total Weights Memory" **3,355,696,384 → 865,480,704 B** (3.88×), decode
**43.29 → 13.38 ms/tok**, process RSS 6.02 → 3.70 GB.

**Tradeoff.** The plugin itself is not lossy beyond the weights you hand it; the quality question
belongs to the quantizer (§6.1).

**Verify.** Count the plugin nodes in the exported ONNX before you build anything:

```python
import collections, onnx
m = onnx.load("cosmos3_int4_onnx/llm/model.onnx", load_external_data=False)
print(collections.Counter(n.op_type for n in m.graph.node)["Int4GroupwiseGemmPluginV2"])
# expected here: 169 — one per quantized linear
```

Then confirm the build actually shrank the weights, from the `llm_build` log:

```
Total Weights Memory: 865,480,704     # INT4 here; 3,355,696,384 at FP16
```

If the node count is lower than the number of linears you quantized, the alignment rule above is
the first thing to check.

### 3.5 The encoder embedding cache

**What it does.** The runtime caches vision-encoder embeddings keyed on **raw pixel bytes**
(`cpp/.../llmRankRuntime.cpp:1967`). A byte-identical frame skips the entire ViT.

**Measured effect.** The ViT encode it skips is worth **≈248 ms** (measured as the difference
between a text-only prefill + 1 token at **39 ms** and the image path at 640×480 / 300 image tokens
+ 1 token at **287 ms**).

**Tradeoff.** Quality-neutral by construction — it is an exact-input cache. In production on a live
camera it rarely fires, because consecutive frames differ in at least one byte. Its practical
significance in this project was as a **benchmarking hazard**: any probe that loops the same image
skips ~248 ms of real work and reports a number that cannot be reproduced by the real workload.
Every synthetic probe here generates a unique image per iteration — see
[`../scripts/profile_fixed.py`](../scripts/profile_fixed.py) and
[`../scripts/tok_vs_res.py`](../scripts/tok_vs_res.py).

**Verify.** Send the same image twice and then a fresh one, and watch `[perf] elapsed_ms=` in the
journal: the repeat should drop by roughly the ViT cost, the fresh one should not.
[`../scripts/quality.py`](../scripts/quality.py) deliberately reuses one file
(`bench_frame.jpg`) — that is correct for a deterministic quality A/B and wrong for any latency
claim.

### 3.6 Engine build sizing

**What it does.** Engine build parameters bound the runtime's memory reservation. This deployment
uses:

```bash
llm_build --onnxDir .../llm --engineDir .../reasoning \
          --maxBatchSize 1 --maxKVCacheCapacity 2048
```

**Measured effect.** An engine built at `--maxBatchSize 4 --maxKVCacheCapacity 4096` **OOMed at
engine load** on this 8 GB board and had to be rebuilt at the values above. So the "effect" is
binary: with the larger settings there is no deployment. Peak **CPU** memory during the INT4 build
was **3,884 MiB**.

**Tradeoff.** A hard capacity limit, not a quality change: batch 1 means no request parallelism, and
2048 is the KV-cache ceiling. Both are adequate for a single webcam stream and would not be for a
multi-client service.

**Related upstream lever, not verified in this deployment.** The Jetson AI Lab TensorRT-Edge-LLM
tutorial recommends `--externalize-weights int4_ffn` to reduce peak build memory, and suggests
`maxInputLen 512` / `maxKVCacheCapacity 1024` for Orin Nano. This deployment ran at
`maxKVCacheCapacity 2048` and did not separately measure the `--externalize-weights` flag.

**Verify.**

```bash
ssh "$JETSON_HOST" 'ls -l /opt/tensorrt-edgellm/models/default/llm.engine'
# expected: 878552564 bytes (INT4)
ssh "$JETSON_HOST" 'systemctl status cosmos3-edge-shim.service | grep -i memory'
ssh "$JETSON_HOST" 'free -m'   # ~2,705 MB available with the service resident
```

Honest caveat: the build parameters are recorded as the values that were used. The Jetson went
offline before `config.json` could be re-read one final time, so treat them as the recorded build
command, not as freshly re-verified output.

### 3.7 NVIDIA techniques that were evaluated and did not apply

Cataloguing these is part of the point — three of them fail silently or with a misleading error.

| Technique | Outcome | Root cause |
|---|---|---|
| `USE_TRT_NATIVE_ATTN=1` (fused ViT attention) | Blocked | Exports cleanly (27 `TRT_Attention` ops + a `kv_lengths` input) but **requires TensorRT ≥ 11** (`tensorrt_edgellm/models/ops.py:384`); JetPack 7.2.1 ships 10.16.2.10. `visual_build` fails with `[6] creator && "Plugin not found, are the plugin name, version, and namespace correct?"` — an error that names the wrong cause. `libNvInfer_edgellm_plugin.so` provides `ViTAttentionPlugin`, not `TRT_Attention`. |
| KV-cache quantization | Not applicable | Only FP8 is implemented (`FP8_KV_CFG`); Orin is sm_87 Ampere with no FP8 hardware. |
| NVFP4 weights | Not applicable | Blackwell / Thor only. |
| `tensorrt-edgellm-export --quantization int4_awq` | **Silent no-op** | Gated behind `_needs_moe_quantization` (`export.py:1046,1074`, dropped again at `4129-4131`) — Mixture-of-Experts models only. Cosmos3-Edge is dense. |
| Narrowing the ViT optimization profile | No-op | `optHW = (minImageTokens + maxImageTokens)/2 × 4` (`cpp/builder/visualBuilder.cpp:319`); with defaults (min 4, max 1024) the optimum is already **514** image tokens ≈ the 512 operating point. |

For completeness on why the ViT was not attacked at the kernel level: at 512 image tokens
(2048 patches) it performs ~2.21 TFLOP in 248 ms = **8.9 TFLOPS achieved, ≈53% of this board's
~16.7 TFLOPS dense FP16 peak**. That is already respectable, which is exactly why the lever that
worked was the token budget (§6.2) and not kernel tuning.

---

## 4. Source 2 — modelopt, used as a format oracle rather than as the quantizer

### What it was *not* used for

NVIDIA TensorRT-Model-Optimizer is the natural tool for this job, and two independent things
prevented it from doing the job here:

1. **Its CUDA path was broken on the host GPU.** On the sm_120 host used for export, the modelopt
   CUDA extension produced zeros in fp32 and NaN in fp16. Unusable as a quantizer on that machine.
2. **Its AWQ path was structurally blocked, not merely inconvenient.** AWQ derives scales from
   observed *activations*, which requires an eager PyTorch forward pass. In TensorRT-Edge-LLM the
   `attention_plugin` is a `torch.library.custom_op` whose body is literally
   `return torch.zeros(...)` (`tensorrt_edgellm/models/ops.py:87-193`) — a shape-only stub that
   exists so the model can be traced to ONNX. An eager forward therefore yields zeros, and **no
   meaningful activation statistics can ever be collected in this toolchain**. This is not a bug you
   wait out; it is a property of how the runtime models attention for export.

### What it *was* used for

`INT4_BLOCKWISE_WEIGHT_ONLY_CFG` was run **on CPU** — sidestepping the broken CUDA extension —
against the same weights, purely to read off the on-disk contract that
`tensorrt-edgellm-export` will later consume:

| Contract item | Ground truth extracted |
|---|---|
| Scale convention | `scale = amax / 7`, `q = round(w/scale).clamp(-8, 7)` |
| Weight packing | `[N//2, K]` uint8, two int4 nibbles per byte; even row = low nibble, odd row = high |
| Companion tensor | `weight_scale`, fp32 `[N, K//128]`, per-output-channel per-group |

The independently written quantizer in this repo reproduces modelopt's scales to **3.7e-09**.

**Why this is a genuinely useful role and not a consolation prize.** The W4A16 on-disk format is a
contract with several independent ways to be silently wrong: the divisor could be 7 or 8, the clamp
could be symmetric or asymmetric, the nibble order could be reversed, the scale could be
per-channel or per-group, stored as fp16 or fp32, transposed or not. Every one of those mistakes
produces a file that loads, exports, builds, and generates fluent garbage. Reading the format out of
a reference implementation *as executable code* — and then matching it numerically to 3.7e-09 —
converts "I read the docs and think this is the layout" into "I can demonstrate my file is
bit-compatible with the reference producer." A specification you can execute and diff against is
worth more here than a specification you can read.

**Measured effect on the deployment.** None directly: modelopt contributed zero bytes of shipped
weights. Its contribution is that Round 3 worked on the first engine build instead of producing a
plausible-looking model that had to be debugged from output quality backwards.

**Tradeoff.** None to the model. The cost is a CPU-only reference run and the dependency on
modelopt at development time only, not at deploy time.

**Verify.** Re-run `INT4_BLOCKWISE_WEIGHT_ONLY_CFG` on CPU over one layer and diff its scales
against those produced by [`../scripts/rtn_int4_quantize.py`](../scripts/rtn_int4_quantize.py) with
the MSE search disabled (α fixed at 1.0, i.e. plain RTN); the two should agree to ~4e-9 (3.7e-09
was measured here). If your modelopt version changes the convention, this diff is what tells you
before the engine does.

---

## 5. Source 3 — the community checkpoint (read, not used)

`ubr-physical-ai/Cosmos3-Edge-INT4-AWQ` on HuggingFace is a W4A16_AWQ, group-size-128 quantization
of Cosmos3-Edge, published with ONNX alongside, produced for a Jetson Orin Nano during the
NVIDIA / OpenHackathons / Oracle Open Models Codefest 2026.

**What it demonstrated.** The *shape* of the answer, which was worth knowing: that W4A16 at group
size 128 is the form the ecosystem converges on for this model class on this device, that the ONNX
is published next to the checkpoint, and that the engine is expected to be built on-device. When
you are about to spend a day on a quantization route, independent evidence that the route's
end-state is the right shape has real value.

**Why it was declined.** Its own documentation states that the checkpoint **has never been
executed.** An unexecuted checkpoint is a hypothesis with a file attached, not a result. The
failure mode this protects against is not hypothetical: it is exactly the failure mode observed
with the broken modelopt CUDA extension, which produced weight files that were the right shape,
the right dtype, and entirely zeros. Nothing short of running it distinguishes the two, and the
publisher had not run it. Preferring to self-quantize from official weights over adopting an
unvalidated third-party artifact is the general policy here, and this is a clean example of why.

**The level of trust extended.** Header metadata only — tensor **names, shapes and dtypes**. No
weight data was downloaded, loaded, or used. The safetensors format makes this precise: the file
begins with an 8-byte little-endian header length followed by that many bytes of JSON, so the
metadata can be read without touching a single weight byte:

```bash
# Read only the safetensors header (names/shapes/dtypes), never the tensor payload.
SAFETENSORS_URL=https://huggingface.co/<repo>/resolve/main/model-00001-of-0000N.safetensors
curl -sL -r 0-7 "$SAFETENSORS_URL" \
  | python3 -c "import sys; print(int.from_bytes(sys.stdin.buffer.read(8), 'little'))"
# then fetch exactly that many bytes starting at offset 8 and json.loads() them
```

That was enough to corroborate the packing contract obtained from modelopt (§4) — a second,
independent witness to the layout — and it was the appropriate ceiling on trust for an artifact
whose author had not run it.

**Measured effect on the deployment.** Zero bytes shipped; corroboration only. This entry exists
because "we looked at it and deliberately did not use it" is a decision that deserves to be on the
record as much as the things that were used.

---

## 6. Original to this project

### 6.1 Calibration-free RTN with MSE-optimal clipping (Round 3)

**What it does.** Round-to-nearest weight quantization needs no activation data at all, which
sidesteps the zero-stub blocker in §4 entirely. Plain RTN, however, wastes resolution on outliers:
the group's `amax` sets the scale, so a single large weight coarsens the quantization of every
other weight in the group. The addition here is a **per-group clipping search that uses only the
weights**: sweep a scale multiplier α over `linspace(0.55, 1.0, 19)`, quantize the group at each α,
compute the squared reconstruction error, and keep the α that minimizes it. Shrinking the range
trades a little clipping error for finer resolution everywhere else, which is a net win on
outlier-heavy LLM weight distributions.

The whole quantizer is ~150 lines:
[`../scripts/rtn_int4_quantize.py`](../scripts/rtn_int4_quantize.py).

**Scope.** 169 linears, matched by

```
^layers\.\d+\.(self_attn\.(to_q|to_k|to_v|to_out)|mlp\.(up_proj|down_proj))\.weight$|^lm_head\.weight$
```

The vision tower, the projector, `embed_tokens` and all norms stay FP16, mirroring the reference
recipe's `exclude_modules` (`["embed_tokens","model.projector*","model.visual*","norm"]`).

**Measured effect.**

| Quantity | Value |
|---|---|
| Mean relative weight error, plain RTN (α = 1.0) | 13.31% |
| Mean relative weight error, MSE-optimal clipping | **11.06%** |
| Worst layer | `to_k`, 20.5% |
| Decode | 43.29 → 13.38 ms/tok (−69.1%) |
| Engine on disk | 3,366,196,508 → 878,552,564 B (3.83×) |
| TensorRT weights memory | 3,355,696,384 → 865,480,704 B (3.88×) |
| Process RSS | 6.02 → 3.70 GB |
| `LLMRuntime` construction | 7.79 s (FP16) → 6.28 s (INT4) |
| Peak CPU memory during build | 3,884 MiB |

The MSE search is pure profit relative to plain RTN — same format, same runtime, same cost at
inference, 2.25 percentage points less weight error — but note what it is *not*: it is still
uncalibrated, and it does not close the gap to a properly calibrated AWQ.

**Tradeoff — and this is the biggest caveat in the repo.** This is **not quality-neutral and not
fully characterized**. 11.06% mean relative weight error is substantial. The observed output is
coherent and grounded, with no repetition collapse or word salad, and greedy-decode spot checks
look clean — but **no benchmark suite was run**, and no FP16-vs-INT4 side-by-side on identical
prompts exists, because both engines cannot be resident within 8 GB simultaneously. Evaluate on a
real task set before putting this in front of anything that matters.

**Verify.** Four gates, all of which passed, in the order they catch problems:

1. **Per-tensor dequant round-trip error**, computed before packing. The script prints it for the
   first three tensors, every fortieth tensor, and as a mean at the end:
   `quantized 169 linears; mean rel err 11.06%`.
2. **Nibble pack/unpack asserted exact.** The script unpacks its own output and asserts equality
   with the pre-pack tensor, mirroring `tensorrt_edgellm/checkpoint/loader.py`
   `_unpack_awq_prepacked`. This is the check that catches a reversed nibble order, which is
   otherwise invisible until generation quality collapses.
3. **ONNX inspection after export**: 169 `Int4GroupwiseGemmPluginV2` nodes, 169 INT8 tensors at the
   cuteDSL fragment shape, **100% non-zero weight bytes**, and `model.onnx.data` at **0.807 GB**
   versus 3.36 GB at FP16. The non-zero check exists specifically because the broken modelopt CUDA
   extension's failure mode was all-zero weights that look structurally perfect.
4. **Engine build log**: `Total Weights Memory: 865,480,704` versus `3,355,696,384`.

```bash
python3 scripts/rtn_int4_quantize.py --src <Cosmos3-Edge ckpt> --dst cosmos3_int4_ckpt
# gates 1 and 2 run inline and abort on failure
```

### 6.2 Right-sizing the image-token budget (Round 4)

**What it does.** After Round 3 made decode 3× faster, the profile inverted: the FP16 vision tower
(938 MB engine, ~411M params) became the dominant fixed cost. Measured decomposition, using
minimums under contention:

- text-only prefill + 1 token: **39 ms**
- image path (640×480, 300 image tokens) + 1 token: **287 ms**
- therefore ViT encode ≈ **248 ms** — **55% of the fixed term** and **26% of mean latency** before
  this round.

Sweeping latency against image-token count (each probe using a **unique** generated image, to defeat
the embedding cache) showed a clean linear relationship and, more importantly, that the WebUI was
pinned at the 512-token cap:

| Probe resolution | Image tokens | Best latency |
|---|---|---|
| 320×240 | 80 | 90 ms |
| 448×336 | 140 | 152 ms |
| 640×480 | 300 | 288 ms |
| 896×672 | 512 (capped) | 499 ms |
| 1280×960 | 512 (capped) | 512 ms |

`max_image_tokens_per_image` is read **at runtime** from the engine's `visual/config.json`
(`builder_config`) and consumed by `smartResize` in `cpp/multimodal/common/imageUtils.cpp:145`,
where `maxPixels = max_image_tokens_per_image × 32²` (each LLM image token is a 32×32 px region:
patch 16 × spatial merge 2). Lowering it from 512 to 320 therefore needs **no engine rebuild** and
is reversible in about 4 seconds by restarting the shim.

**Measured effect.** A **constant −158 ms in every matched-`gen_tok` bucket** — the signature of a
fixed-cost reduction rather than a decode-rate change, which is what confirms the change landed
where it was supposed to:

| gen_tok | n(before) | n(after) | before | after | delta | improve |
|---|---|---|---|---|---|---|
| 17 | 13 | 3 | 672 ms | 516 ms | −156 ms | 23.2% |
| 18 | 11 | 2 | 687 ms | 528 ms | −158 ms | 23.1% |
| 20 | 60 | 4 | 714 ms | 556 ms | −158 ms | 22.2% |
| 21 | 15 | 7 | 727 ms | 567 ms | −160 ms | 22.0% |
| 22 | 70 | 5 | 738 ms | 582 ms | −156 ms | 21.1% |
| 23 | 9 | 12 | 756 ms | 594 ms | −162 ms | 21.5% |
| 27 | 12 | 2 | 808 ms | 646 ms | −161 ms | 19.9% |
| 34 | 31 | 8 | 899 ms | 740 ms | −159 ms | 17.7% |

**Weighted improvement: 20.8%.**

**Tradeoff — real, measured, and workload-dependent.** Unlike Rounds 1–3 this one changes the
model's *input fidelity*, so it was A/B'd at temperature 0 (greedy):

- **Natural camera scene** (the actual Live VLM workload): output **byte-identical** at 320 and 512
  across three prompts, including one specifically probing fine detail. Free win on this workload.
- **Dense text screenshot**: at 512 the model quoted on-screen annotations verbatim ("Large left
  lateral deviation", "Markers appear to be lagging behind in both video panel and 3D panel"); at
  320 it drifted vaguer and partly hallucinated ("flight simulation or navigation system", generic
  X/Y/Z axes). Clear regression on this workload.

**Recommendation: 320 for live camera work, 512 for document and screenshot reading.** Because the
setting is runtime-only, this can be a deployment choice rather than a build choice.

**Verify.**

```bash
# 1. the configured budget
ssh "$JETSON_HOST" "grep -o '\"max_image_tokens_per_image\"[^,]*' \
  /opt/tensorrt-edgellm/models/default/visual/config.json"

# 2. what the live client actually sends (the number that matters)
ssh "$JETSON_HOST" "journalctl -u cosmos3-edge-shim.service --since '10 min ago' --no-pager" \
  | grep '\[perf\]'
# prompt_tok was pinned at 510/511 before the change; 330 after
```

The `prompt_tok` field is the honest check: the cap only helps if the client was actually hitting
it. Re-run the sweep with [`../scripts/tok_vs_res.py`](../scripts/tok_vs_res.py) to reproduce the
resolution table on your own board: it sweeps the same ladder as the table above (320×240, 448×336,
640×480, 896×672, 1280×960) against a text-only baseline, and reads the token count from the HTTP
response's `usage.prompt_tokens` field, which the shim published here populates. The figures in this
document were gathered with an earlier snapshot of the shim that returned `usage: null` and reported
`prompt_tok` on the `[perf]` journal line instead — if you are running that older copy, read the
count from the journal.

---

## 7. How to verify the whole stack in one pass

```bash
export JETSON_HOST=orin@jetson.local

# platform
ssh "$JETSON_HOST" 'sudo nvpmodel -q; sudo jetson_clocks --show'

# resident runtime, one construction, graphs captured
ssh "$JETSON_HOST" "journalctl -u cosmos3-edge-shim.service --no-pager | grep '\[shim\]'"

# INT4 engine actually in place
ssh "$JETSON_HOST" 'ls -l /opt/tensorrt-edgellm/models/default/llm.engine'   # 878552564 B

# memory headroom
ssh "$JETSON_HOST" 'free -m'                                                 # ~2705 MB available

# fixed vs marginal cost, fitted from live traffic
python3 scripts/collect_perf.py "10 min ago"        # expect ~289 ms fixed, ~13.1 ms/token

# A/B two windows at matched gen_tok (the stronger comparison)
python3 scripts/compare_perf.py <A_start> <A_end> <B_start>
```

A caution on `collect_perf.py`: a window that spans a configuration change mixes two fixed-cost
regimes and will report a low R². That is expected, not a fault in the fit — bound your windows to
one configuration, or use the matched-`gen_tok` comparison instead.

---

## 8. What is deliberately not optimized

- **The vision tower is still FP16.** It is the largest single engine at 938 MB, larger than the
  INT4 text tower. Quantizing it was out of scope for this campaign, and §3.7 shows the fused
  attention route is blocked by the TensorRT version on JetPack 7.2.1.
- **No KV-cache quantization.** Only FP8 is implemented and Orin sm_87 has no FP8 hardware.
- **No speculative decoding.** TensorRT-Edge-LLM supports EAGLE / DDTree / DSpark; none was
  evaluated here, so nothing is claimed about it on this board.
- **Batch 1 only.** The engine is built at `maxBatchSize 1`; multi-client throughput is not a
  property this deployment has.

For the full write-up, including the measurement methodology and the repository-landscape analysis,
see [report.md](./report.md).
