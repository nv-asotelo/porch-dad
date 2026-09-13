# Negative results

Six things that did not work, and exactly why.

Every item below was established on the real deployment — Jetson Orin Nano Super 8 GB,
JetPack 7.2.1 (L4T R39.2.1), CUDA 13.2, TensorRT 10.16.2.10, TensorRT-Edge-LLM 0.10.1,
`nvidia/Cosmos3-Edge` (reasoner) — and each one cost hours that the next person should not have
to spend. Two of them failed *silently*, and one failed with an error message that names the
wrong cause. That is the reason this page exists as a first-class document rather than a
footnote in [the report](./report.md).

| # | Attempt | Outcome | Root cause (one line) |
|---|---|---|---|
| 1 | `USE_TRT_NATIVE_ATTN=1` — fused ViT attention | Blocked at **build** time | `TRT_Attention` requires TensorRT ≥ 11; this image ships 10.16.2.10 |
| 2 | KV-cache quantization | Not applicable | Only FP8 is implemented; Orin sm_87 has no FP8 hardware |
| 3 | Narrowing the ViT optimization profile | No-op | `opt` is already 514 image tokens, i.e. the 512 operating point |
| 4 | `Cosmos3-Edge-Policy-DROID` | Non-viable on 8 GB | GEN engine OOMs with nothing else resident |
| 5 | Qwen VLM alternatives | Abandoned | Same dense-model no-op, plus GPU OOM and ~4 h calibration |
| 6 | Engine at `batch=4` / `KV=4096` | OOM at engine load | Build succeeds, load does not fit in 8 GB unified memory |

---

## 1. `USE_TRT_NATIVE_ATTN=1` — the most actionable failure

This is the one worth reading in full. It is a documented, supported environment variable that
produces a **clean, correct-looking export** and then fails at engine build with an error that
sends you to debug the wrong subsystem.

### What was attempted

TensorRT-Edge-LLM can emit the vision tower's attention as a single fused native TensorRT
operator instead of the decomposed MatMul/Softmax/MatMul pattern, gated on the
`USE_TRT_NATIVE_ATTN` environment variable. The motivation was the measured fixed cost: ViT
encode was **≈ 248 ms**, **55% of the fixed term** and **26% of mean latency** before Round 4.
A fused attention kernel is the obvious first thing to try against that number.

```bash
# On the HOST workstation, during ONNX export of the vision tower
USE_TRT_NATIVE_ATTN=1 tensorrt-edgellm-export <model> <output_dir> --task reasoning
```

### What happened — export succeeds, and looks right

The export completes without warnings and the resulting graph is exactly what the feature
promises:

- **27 `TRT_Attention` nodes** — one per vision-tower layer, matching the
  27 layers in `visual/config.json`
- a new **`kv_lengths` graph input**, which is the signature that the native op is wired up with
  its sequence-length metadata rather than left dangling

Both of those are positive signals. Nothing at export time indicates a problem. Then the engine
build (`visual_build`) fails:

```text
[6] creator && "Plugin not found, are the plugin name, version, and namespace correct?"
```

### Root cause

`TRT_Attention` **requires TensorRT ≥ 11**. The requirement is stated in the docstring of
the op definition at `tensorrt_edgellm/models/ops.py:384`. JetPack 7.2.1 ships TensorRT
**10.16.2.10**. There is no combination of plugin paths, namespaces, or rebuilds that fixes this
on JetPack 7.2.1 — the operator does not exist in the runtime you have.

### Why the error message is actively misleading

Three things conspire here, and each one costs time:

1. **"Plugin not found" is the generic TensorRT plugin-registry lookup failure.** It is emitted
   whenever the builder cannot resolve an op to a registered creator, regardless of *why*. It
   never mentions a version requirement.

2. **The same message has a real, different, common cause on this stack.** During bring-up,
   genuine "Plugin not found" failures at engine build/load were fixed by pointing
   `EDGELLM_PLUGIN_PATH` and/or `LD_PRELOAD` at `libNvInfer_edgellm_plugin.so` — see
   [troubleshooting §6a](./troubleshooting.md#6a-genuinely-missing-plugin-path). So the message
   is *correct-looking* and the standard remedy is known to work, which is precisely the trap:
   you will spend your first hour on plugin search paths.

3. **The plugin library really does contain an attention plugin — the wrong one.**
   `libNvInfer_edgellm_plugin.so` provides **`ViTAttentionPlugin`**, but **not** `TRT_Attention`.
   A quick `strings | grep -i attention` over the library therefore "confirms" that attention
   plugins are present and loaded, which reads as evidence that your plugin path is fine and the
   problem must be elsewhere. It isn't. `TRT_Attention` is not an Edge-LLM plugin at all; it is a
   **native TensorRT operator** that the core library is expected to resolve, and TensorRT 10.16
   does not have it.

### How to recognize this in under a minute

```bash
# 1. What TensorRT does this device actually have?
# the $ must reach the REMOTE dpkg-query unexpanded, hence the backslashes
ssh "${JETSON_HOST:-orin@jetson.local}" \
  "dpkg-query -W -f='\${Package} \${Version}\\n' 'libnvinfer*' 2>/dev/null | sort"
```

If the major version is **10**, stop here: `USE_TRT_NATIVE_ATTN=1` cannot work, and no amount of
export-side debugging will change that. As of JetPack 7.2.1 this is every Orin.

```bash
# 2. Confirm the op is in the exported graph (it will be) — export success proves nothing
python3 - <<'PY'
import onnx
m = onnx.load("visual/model.onnx", load_external_data=False)
print("TRT_Attention nodes:", sum(1 for n in m.graph.node if n.op_type == "TRT_Attention"))
# print every (domain, op_type) pair so a zero count above is not mistaken for a failed export —
# the op may simply be spelled differently on your release
print("ops present:", sorted({(n.domain, n.op_type) for n in m.graph.node}))
print("graph inputs:", [i.name for i in m.graph.input])
PY

# 3. Confirm the plugin library does NOT provide it (it does not) — this is the misleading check
ssh "${JETSON_HOST:-orin@jetson.local}" \
  'strings /home/orin/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so | grep -i attention'
```

Step 3 is included specifically because it is the check most people run first, and because its
result — an attention plugin, just not that one — is what makes the failure feel like a
configuration problem.

**The general rule this establishes: a clean export is not evidence of a supported path.** In
this toolchain, export and build are separate programs with separate notions of what exists.
Version preconditions are enforced (if at all) only at build time, by whichever component
happens to fail first.

### What to do instead, and the honest limit of this finding

Leave `USE_TRT_NATIVE_ATTN` unset. The default decomposed-attention path builds and runs
correctly, and it is what produced every number in this repo.

The gain that was forfeited is **unknown and was never measured on this device** — the engine
could not be built, so there is no A/B. What *is* measured is that the ViT was not
grossly inefficient to begin with: at 512 image tokens (2048 patches) it achieved
**8.9 TFLOPS in 248 ms**, roughly **53% of the Orin Nano Super's ~16.7 TFLOPS dense FP16 peak**.
That is a respectable fraction for an unfused attention pattern, and it is why the eventual
lever against fixed cost was the **image-token budget** (512 → 320, −158 ms constant, 20.8%
weighted improvement) rather than kernel fusion. Kernel-level ViT work had a smaller ceiling
than the token-count work, and the token-count work needed no rebuild.

---

## 2. KV-cache quantization — FP8-only on hardware with no FP8

### What was attempted

Shrinking the KV cache is the standard second move after weight quantization, especially on a
memory-bandwidth-bound decode. Decode here was measured at **77.6 GB/s of 102 GB/s peak (76% of
theoretical)** at FP16, so anything that reduces bytes moved per token is worth evaluating.

### What happened

The only KV-cache quantization implemented in this toolchain is **FP8** (`FP8_KV_CFG`). There is
no INT8 or INT4 KV path to select.

### Root cause

Orin is **sm_87 (Ampere)** and has **no FP8 hardware** — nor NVFP4. The precondition is the
device, not the software, so there is nothing to configure around. This is a hard architectural
stop, not a missing build flag.

### How to recognize it quickly

Check the compute capability before reading any quantization docs. FP8 requires
Ada / Hopper / Blackwell / Thor; NVFP4 requires Blackwell / Thor. If you are on Orin (sm_87),
both are off the table for weights *and* KV cache, and the only viable weight format in this
stack is **INT4 W4A16** (which is what this deployment used, at 3.88× weight-memory reduction).

A practical consequence: NVIDIA's own Cosmos-adjacent edge recipe — Cosmos-Reason2 on Jetson AGX
**Thor** with FP8 + TensorRT-Edge-LLM, in `nvidia-cosmos/cosmos-cookbook` — is not portable to
Orin for exactly this reason. Recipes are silently device-specific.

---

## 3. Narrowing the ViT optimization profile — already optimal, so a no-op

### What was attempted

TensorRT optimization profiles have `min`/`opt`/`max` shapes, and kernels are tuned for `opt`.
The deployment operated at ~512 image tokens while the profile defaults span a much wider range,
so narrowing the range to bracket the real operating point looked like free performance.

### What happened

Nothing would have changed. Reading the builder shows `opt` is already sitting on the operating
point:

```text
optHW = (minImageTokens + maxImageTokens) / 2 × 4      cpp/builder/visualBuilder.cpp:319
```

With the defaults (`min` = 4, `max` = 1024 image tokens), `opt` lands at
**(4 + 1024) / 2 = 514 image tokens** — within 2 tokens of the 512 operating point. The ×4 is the
patch expansion (`spatial_merge_size` 2 ⇒ 4 patches per image token).

### Root cause

The midpoint formula plus the default bounds happen to coincide with the workload. Narrowing
`min`/`max` symmetrically around 512 would recompute an `opt` of ~512 instead of 514 — the same
kernel selection, for the cost of a full ViT engine rebuild.

Stated plainly: this is a **source-level determination, not a benchmarked A/B**. The arithmetic
is deterministic and the conclusion follows from it, but no narrowed-profile engine was built and
timed to confirm the null result.

### How to recognize it quickly

Before rebuilding an engine to narrow a profile, compute `opt` from the builder's own formula and
compare it against your measured operating point. If they already agree, the rebuild buys
nothing. Note that this reasoning is specific to the default `min`/`max` here — change the image
token budget far from 512 (Round 4 moved it to 320) and the gap to `opt` = 514 widens, at which
point the question is worth revisiting. It was not revisited in this deployment.

---

## 4. `Cosmos3-Edge-Policy-DROID` — does not fit on 8 GB

### What was attempted

The Policy-DROID variant was evaluated as a second workload alongside the reasoner, to see
whether the RAM freed by INT4 (**−2.32 GB RSS**, 472 MB → 2,705 MB system memory available) was
enough to host a policy model too.

### What happened

The **GEN engine OOMs on 8 GB even with no other model resident** — i.e. the failure is not a
co-residency problem that better scheduling or a smaller KV cache on the reasoner could solve.
It does not fit on its own.

### Root cause

Model size against 8 GB of unified LPDDR5, where the same pool serves CPU, GPU, and the display
stack. No configuration change in this repo's scope addresses it.

### How to recognize it quickly

Treat "supported model family" as a family-level claim, not a per-variant one. A support matrix
that lists Cosmos3-Edge for Orin Nano 8 GB says nothing about Policy-DROID on the same board.
**What does not fit is as much a deployment fact as what does**, which is why publishing per-device
non-viability is one of the recommendations in [the report](./report.md).

---

## 5. The Qwen detour — right instinct, wrong turn

### What was attempted

When INT4 on Cosmos3-Edge looked blocked, the obvious fallback was a different VLM with a
known-good quantized path — the Jetson AI Lab TensorRT-Edge-LLM tutorial for Orin Nano 8 GB
demonstrates exactly that, on **Qwen3-4B-Instruct with INT4 AWQ**.

### What happened

Three separate walls, in order:

1. **`--quantization int4_awq` was a silent no-op** for the same reason as on Cosmos3-Edge: the
   flag is gated behind `_needs_moe_quantization` (`tensorrt_edgellm/scripts/export.py:1046,1074`,
   dropped again at `export.py:4129-4131`) and applies to **Mixture-of-Experts models only**.
   Dense models accept the flag and ignore it. No error, no warning.
2. **GPU OOM** during the calibration attempt.
3. **~4 hours** of AWQ calibration wall-clock for the path that does work.

### Root cause

Changing the *model* did not change the *toolchain constraint*. The dense-vs-MoE gate and the
cost of activation-based calibration are properties of the export tooling, not of Cosmos3-Edge.
Swapping models to route around a toolchain limitation is almost always wasted motion.

### What it was still worth

One genuinely useful data point: it established the **real cost of AWQ even where it is
supported** (~4 h plus a GPU that has to hold the calibration forward pass). That number is what
made calibration-free **RTN + MSE-optimal clipping** obviously correct rather than merely
convenient — the final route runs on CPU, needs no calibration data, and landed at
**11.06% mean relative weight error** (worst layer `to_k` at 20.5%) versus 13.31% for plain RTN.
Those are **weight-reconstruction errors only — no task-suite benchmark was run** on the INT4
engine; greedy-decode spot checks looked clean, but quality is not characterized. Evaluate on a
real task set before production use (see [caveats](./report.md#8-caveats-and-honest-limits)).

### How to recognize it quickly

Before switching models, ask whether the wall you hit is in the model or in the tooling. If it is
in the tooling — a gated flag, a stubbed op, a missing operator — the new model hits the same
wall one afternoon later.

---

## 6. Engine at `batch=4` / `KV=4096` — OOM at engine **load**

### What was attempted

An LLM engine built with headroom for concurrency and long contexts:
`maxBatchSize 4`, `maxKVCacheCapacity 4096`.

### What happened

**OOM at engine load** — not at build. The engine file exists and is valid; the runtime cannot
instantiate it in 8 GB of unified memory alongside its own allocations. The engine was rebuilt at:

```bash
llm_build --onnxDir .../llm --engineDir .../reasoning \
          --maxBatchSize 1 --maxKVCacheCapacity 2048
```

These are recorded as the parameters used. The Jetson went offline before a final re-read of
`config.json`, so treat them as the values that were set, not as freshly re-verified output.

### Root cause

KV-cache working set scales with `maxBatchSize × maxKVCacheCapacity`, and on unified memory that
allocation competes with weights, activations, the CUDA context, and the rest of the system.
Going from 4×4096 to 1×2048 is an **8× reduction** in the reserved KV footprint. For a
single-stream webcam workload the batch dimension was pure waste: the runtime serializes requests
anyway, which is also why concurrent traffic inflates probe latencies (see the contention trap in
[the report](./report.md)).

### How to recognize it quickly

- The failure surfaces at **load / runtime construction**, so a successful build is not
  reassurance. Watch for it when you first start the resident shim
  ([`serve/cosmos3_shim.py`](../serve/cosmos3_shim.py) constructs `LLMRuntime` at startup).
- Size the batch dimension to the *actual* concurrency of the client. One camera = batch 1.
- For reference, the Jetson AI Lab Orin Nano tutorial recommends `maxInputLen 512` and
  `maxKVCacheCapacity 1024` for this class of device — more conservative than the 2048 used here,
  which worked at batch 1. It also documents `--externalize-weights int4_ffn` to cut peak build
  memory; peak build CPU memory in this deployment was **3,884 MiB**.

---

## How to avoid wasting time — checklist

Derived entirely from the six failures above. Run these *before* starting an optimization, not
after it fails.

1. **Check the hardware preconditions first, from the device.** Compute capability and TensorRT
   major version decide more than any flag. sm_87 ⇒ no FP8, no NVFP4, no FP8 KV cache.
   TensorRT 10.x ⇒ no `TRT_Attention`, so no `USE_TRT_NATIVE_ATTN=1`.
2. **Never treat a clean export as proof of support.** Export and build enforce different things.
   `USE_TRT_NATIVE_ATTN=1` produces 27 correct-looking ops and a new graph input, and still
   cannot be built.
3. **Distrust the error message; verify the version.** "Plugin not found, are the plugin name,
   version, and namespace correct?" has at least two causes on this stack — a genuinely missing
   `EDGELLM_PLUGIN_PATH`/`LD_PRELOAD`, and an operator your TensorRT is too old to have. Check
   the version before you touch plugin paths.
4. **Assume quantization flags may be silent no-ops, and verify the artifact.** `--quantization
   int4_awq` does nothing on dense models; INT4 is silently skipped when
   `out_features % 64 != 0 or in_features % 64 != 0`
   (`tensorrt_edgellm/quantization/quantize.py:1017-1029`). Count the quantized nodes in the
   exported ONNX and read `Total Weights Memory` from the build log — those are the only
   statements that cannot lie.
5. **Read the builder source before rebuilding to tune a profile.** The ViT `opt` shape was
   already 514 image tokens against a 512 operating point; the rebuild would have been pure cost.
6. **Ask whether the wall is in the model or in the tooling.** If it is the tooling, changing
   models reproduces the wall at a later hour.
7. **Prefer changes that need no rebuild.** The image-token budget is read at runtime from
   `visual/config.json` and is reversible in ~4 s by restarting the shim; it delivered 20.8%.
   Engine rebuilds cost far more per experiment and are harder to revert.
8. **Size engines for the real workload, not for hypothetical concurrency.** `batch=4` bought
   nothing on a single camera stream and cost an OOM at load; the runtime serializes requests
   regardless.
9. **Expect OOM to appear at load, not at build.** Budget the check accordingly, and verify by
   starting the resident process rather than by a successful `llm_build`.
10. **Write down the null results.** Three of the six items here look like promising ideas on
    paper and are still worth *rechecking* on a newer JetPack, a different board, or a newer
    TensorRT — which is only possible if the reason they failed was recorded precisely enough to
    re-test.

---

Related: [full report](./report.md) ·
[troubleshooting](./troubleshooting.md) ·
[INT4 quantizer](../scripts/rtn_int4_quantize.py) ·
[measurement scripts](../scripts/) ·
[resident shim](../serve/cosmos3_shim.py)
