# Cosmos3-Edge on a Jetson Orin Nano Super 8 GB

This repository is a reproducible recipe plus the measurements behind it for running
[`nvidia/Cosmos3-Edge`](https://github.com/nvidia/cosmos) (reasoner task) fast on a **Jetson Orin
Nano Super 8 GB**, using the **TensorRT-Edge-LLM 0.10.1** runtime with **self-quantized INT4 (W4A16)**
weights. The published guidance for this device stops at BF16 and publishes no numbers for it, and
the only INT4 Cosmos3-Edge checkpoint we could find on HuggingFace has, by its own documentation,
never been executed. So the weights here were quantized from the official checkpoint locally with
calibration-free RTN plus MSE-optimal clipping, the engines were built on the device, and every
number below was measured on a live workload (Live VLM WebUI streaming a webcam into an
OpenAI-compatible endpoint). The repo contains the quantizer, the serving shim, the systemd units,
the measurement tooling, and the deployment walkthrough — it does **not** contain model weights.

> **Read the [caveats](#caveats-and-limits) before quoting any number here.** In particular: the
> INT4 quality was spot-checked but **never run against a benchmark suite**, and the 22.5×
> end-to-end headline is measured against a deliberately naive baseline. The honest figure against
> a competently configured FP16 resident baseline is **3.38× decode and 2.32 GB of RAM**.

---

## Measured results

| Metric | Before | After | Change |
|---|---|---|---|
| End-to-end mean latency | 13.91 s | **619 ms** | 22.5× |
| Decode, marginal cost | 44.43 ms/tok (22.5 tok/s) | **13.13–13.16 ms/tok (76.0 tok/s)** | 3.38× |
| Fixed cost (ViT encode + prefill) | ~348 ms | **287–289 ms** | −18% |
| Resident process RSS | 6.02 GB | **3.70 GB** | −2.32 GB (−38.5%) |
| System RAM available | 472 MB | **2,705 MB** | 5.7× |
| LLM engine on disk | 3,366,196,508 B (3.135 GB) | **878,552,564 B (0.818 GB)** | 3.83× |
| TensorRT `Total Weights Memory` | 3,355,696,384 B | **865,480,704 B** | 3.88× |

> The end-to-end **mean** is sample-dependent — it tracks the response-length distribution of the
> traffic in the window (619 ms over the clean n=88 post-change window; 626 ms over the larger
> n=699 window that also contains probe traffic). The regression coefficients are the stable
> quantities: **13.13–13.16 ms/token** marginal and **287–289 ms** fixed.


Final regression fit over live traffic: **marginal 13.13 ms/token, fixed 289 ms, R² = 0.999,
n = 699 requests**. Latency is regressed against generated-token count rather than averaged,
because response length varies per request and raw means are not comparable across runs — see
[docs/methodology.md](docs/methodology.md).

The RAM result is arguably the more useful one on an 8 GB board: 472 MB → 2,705 MB of available
system memory is the difference between a device that OOMs when you add a second workload and one
that has room for it.

## Comparison against the published AGX Orin number

| Configuration | Device | RAM | Precision | Runtime | Decode |
|---|---|---|---|---|---|
| Jetson AI Lab, official | AGX Orin | 64 GB | BF16 | vLLM | 44.1 tok/s |
| This repo | Orin Nano Super | 8 GB | INT4 W4A16 | TensorRT-Edge-LLM 0.10.1 | **76.0 tok/s** |

**This is not an apples-to-apples hardware comparison, and it should not be presented as one.**
The two rows differ in precision (INT4 weight-only vs BF16) *and* in runtime (TensorRT-Edge-LLM vs
vLLM) *and* in device. INT4 weight-only quantization is precisely what buys the decode rate here —
single-stream decode on this model is memory-bandwidth-bound, so shrinking the weights 3.88× is
close to a direct multiplier — and that comes with a quality cost this repo has **not** fully
measured. The comparison's actual value is narrower and more practical: it shows that the smaller,
cheaper 8 GB device is not disqualified from this model, which is not obvious from the published
material. Jetson AI Lab publishes **no** Orin Nano performance numbers for Cosmos3-Edge at all,
so there is no same-runtime, same-precision baseline to compare against.

## Why this repo exists

There is a specific, documentable gap in the published material as of September 2026:

- **[jetson-ai-lab.com/models/cosmos3-edge](https://www.jetson-ai-lab.com/models/cosmos3-edge/)**
  lists **Orin Nano 8 GB** as a supported device, but the guidance is **BF16 via HuggingFace
  Transformers** with **vLLM recommended**, and it publishes **no Orin Nano performance numbers and
  no quantized path**. The AGX Orin 64 GB number (44.1 tok/s) is the only figure given.
- **[jetson-ai-lab.com/tutorials/tensorrt-edge-llm](https://www.jetson-ai-lab.com/tutorials/tensorrt-edge-llm/)**
  *does* demonstrate INT4 on an Orin Nano 8 GB — but on **Qwen3-4B-Instruct**, not Cosmos3-Edge.
- **[`nvidia/cosmos-framework`](https://github.com/nvidia/cosmos-framework)** contains no Jetson,
  quantization, TensorRT, or edge content; its inference backends are Diffusers / Transformers /
  vLLM. The edge capability lives in
  [`NVIDIA/TensorRT-edge-llm`](https://github.com/NVIDIA/TensorRT-edge-llm).

Nobody documents the intersection — **Cosmos3-Edge, quantized, on Orin Nano** — which is exactly
the configuration an 8 GB board requires. A community checkpoint
(`ubr-physical-ai/Cosmos3-Edge-INT4-AWQ`) exists and was **declined** for this deployment: its own
documentation states it has never been executed. Only its safetensors *header metadata* (tensor
names, shapes, dtypes — no weight data) was read, to confirm the on-disk packing contract.

Closing that gap also meant working around three closed routes: the export CLI's
`--quantization int4_awq` flag is a silent no-op on dense models, modelopt AWQ calibration is
structurally impossible in this toolchain (the eager attention op is a zeros-returning stub), and
the community checkpoint was unvalidated. See [docs/optimizations.md](docs/optimizations.md) and
[docs/negative-results.md](docs/negative-results.md).

## What is in the repo

| Path | What it is |
|---|---|
| [`deploy/01-hardware-and-flashing.md`](deploy/01-hardware-and-flashing.md) | Board, recovery mode, SDK Manager flashing to NVMe |
| [`deploy/02-platform-tuning.md`](deploy/02-platform-tuning.md) | `nvpmodel` MAXN_SUPER, `jetson_clocks`, swap, clock verification |
| [`deploy/03-install-trt-edge-llm.md`](deploy/03-install-trt-edge-llm.md) | Building/installing TensorRT-Edge-LLM 0.10.1 and its plugin library |
| [`deploy/04-quantize-and-build-engines.md`](deploy/04-quantize-and-build-engines.md) | INT4 quantization, ONNX export, on-device engine build |
| [`deploy/05-serve-and-webui.md`](deploy/05-serve-and-webui.md) | Running the shim + Live VLM WebUI under systemd |
| [`deploy/06-reachy-mini-sdk.md`](deploy/06-reachy-mini-sdk.md) | Pointing a [Reachy Mini](https://huggingface.co/docs/reachy_mini) robot's camera at the shim, via Live VLM WebUI and then natively |
| [`reachy/cosmos_bridge.py`](reachy/cosmos_bridge.py) | Native Reachy Mini camera → Cosmos3-Edge bridge (no browser) |
| [`reachy/release_camera_for_webui.py`](reachy/release_camera_for_webui.py) | Frees Reachy Mini's camera so a browser/Live VLM WebUI can open it directly |
| [`scripts/rtn_int4_quantize.py`](scripts/rtn_int4_quantize.py) | The quantizer: weight-only INT4 RTN with MSE-optimal clipping, emits the ModelOpt W4A16 on-disk format |
| [`scripts/collect_perf.py`](scripts/collect_perf.py) | Fits `elapsed = fixed + marginal × gen_tok` over shim journal logs |
| [`scripts/compare_perf.py`](scripts/compare_perf.py) | Matched-`gen_tok` A/B between two time windows |
| [`scripts/profile_fixed.py`](scripts/profile_fixed.py) | Splits fixed cost into ViT encode vs prefill, using unique images per probe |
| [`scripts/tok_vs_res.py`](scripts/tok_vs_res.py) | Latency vs image resolution / image-token count; sweeps the published ladder (320×240, 448×336, 640×480, 896×672, 1280×960) and subtracts a text-only prompt-token baseline |
| [`scripts/quality.py`](scripts/quality.py) | Greedy-decode (temperature 0) quality probe |
| [`serve/cosmos3_shim.py`](serve/cosmos3_shim.py) | Resident OpenAI-compatible shim: one `LLMRuntime` for process lifetime, CUDA-graph decode, `[perf]` instrumentation |
| [`systemd/cosmos3-edge-shim.service`](systemd/cosmos3-edge-shim.service) | Unit for the shim |
| [`systemd/live-vlm-webui.service`](systemd/live-vlm-webui.service) | Unit for Live VLM WebUI pointed at the shim |
| [`docs/report.md`](docs/report.md) | The full written report |
| [`docs/methodology.md`](docs/methodology.md) | How the numbers were taken, and two traps that invalidate naive benchmarks |
| [`docs/optimizations.md`](docs/optimizations.md) | The four rounds in detail, including the INT4 derivation |
| [`docs/negative-results.md`](docs/negative-results.md) | What was tried and did not work, with root causes |
| [`docs/troubleshooting.md`](docs/troubleshooting.md) | Failures hit during bring-up and their fixes |
| [`docs/ecosystem-and-recommendations.md`](docs/ecosystem-and-recommendations.md) | Where this capability is documented today, and what would close the gap |
| `web/index.html`, `web/slides.html` | Self-contained report page and slide deck |

The two host-side scripts — [`collect_perf.py`](scripts/collect_perf.py) and
[`compare_perf.py`](scripts/compare_perf.py) — read the target host from the **`JETSON_HOST`**
environment variable, defaulting to `orin@jetson.local`; they `ssh` in to read the shim's journal.
The other three (`profile_fixed.py`, `tok_vs_res.py`, `quality.py`) run **on the device** against
`http://127.0.0.1:8000`, and `quality.py` additionally expects a fixed test frame at
`/home/orin/bench_frame.jpg`. No addresses or credentials are stored in this repo.

## Quickstart

The shortest path, in order. Each step links to the document that explains it; do not skip 02, the
power mode alone is a large fraction of Round 1.

```bash
export JETSON_HOST=orin@jetson.local    # or your own user@host
```

1. **Flash the board** — [deploy/01-hardware-and-flashing.md](deploy/01-hardware-and-flashing.md).
   JetPack 7.2.1 to **NVMe** (SDK Manager defaults to the SD slot). Recovery mode needs an
   FC REC-to-GND jumper on most cases.
2. **Tune the platform** — [deploy/02-platform-tuning.md](deploy/02-platform-tuning.md).
   ```bash
   ssh "$JETSON_HOST" 'sudo nvpmodel -m 2 && sudo jetson_clocks'
   ```
   MAXN_SUPER takes the GPU 306 → 1020 MHz, CPU → 1728 MHz, EMC 2133 → 3199 MHz.
3. **Install TensorRT-Edge-LLM 0.10.1** —
   [deploy/03-install-trt-edge-llm.md](deploy/03-install-trt-edge-llm.md). Note the
   `EDGELLM_PLUGIN_PATH` / `LD_PRELOAD` requirement for `libNvInfer_edgellm_plugin.so`; missing it
   produces a "Plugin not found" error at build or load time.
4. **Quantize and export on the host, build the engines on the device** —
   [deploy/04-quantize-and-build-engines.md](deploy/04-quantize-and-build-engines.md).
   ```bash
   python3 scripts/rtn_int4_quantize.py --src <fp16-checkpoint-dir> --dst <int4-checkpoint-dir>
   ```
   Then export with `tensorrt-edgellm-export <int4-checkpoint-dir> <onnx-out-dir> --task reasoning`
   (the `tensorrt-edgellm-quantize` entry point fails with `KeyError: 'cosmos3_edge'`) and build.
   The exporter creates `llm/`, `visual/` and `audio/` subdirectories under the output directory, so
   the LLM build's `--onnxDir` is `<onnx-out-dir>/llm`. Engine build parameters used here
   were `--maxBatchSize 1 --maxKVCacheCapacity 2048`; batch 4 / KV 4096 OOMs at engine load on 8 GB.
   Peak build CPU memory was 3,884 MiB (on the device, where the build runs).
5. **Serve it** — [deploy/05-serve-and-webui.md](deploy/05-serve-and-webui.md).
   ```bash
   ssh "$JETSON_HOST" 'sudo systemctl enable --now cosmos3-edge-shim.service live-vlm-webui.service'
   ```
   The shim exposes `GET /v1/models` and `POST /v1/chat/completions` on port 8000; the WebUI runs
   on port 8090 and talks to the shim. Startup logs `LLMRuntime constructed in 6.28 s` for the INT4
   engines (7.79 s for FP16), then captures CUDA graphs and runs a warm-up inference.
6. **Measure it** — [docs/methodology.md](docs/methodology.md).
   ```bash
   python3 scripts/collect_perf.py "10 min ago"
   ```
   Read the methodology first. Benchmarking by looping one image silently skips the entire vision
   tower (the runtime caches encoder embeddings on raw pixel bytes) and will report a number that
   is not real.

## The four optimization rounds

Run as a goal-seek loop: keep going while each round beats a threshold (started at 25%, lowered to
5%, then raised to 10%).

| # | Change | Result | Verdict |
|---|---|---|---|
| 0 | Baseline: `llm_inference` CLI spawned per request | 13.91 s/req | baseline |
| 1 | Resident runtime + CUDA graph + MAXN_SUPER + `jetson_clocks` | 2.07 s/req | **−85.1%** pass |
| 2 | Context-cache reuse, greedy decode, `top_k` fix | 43.29 ms/tok | −3.5…4.3% **fail** |
| 3 | INT4 W4A16 self-quantization (RTN + MSE-optimal clipping) | 13.38 ms/tok | **−69.1%** pass |
| 4 | Image-token budget 512 → 320 | 287 ms fixed | **−20.8%** pass |

Round 2 is kept in the record because its failure is what made Round 3 the obvious move: at FP16
the model moved 3.36 GB per token in 43.29 ms ≈ **77.6 GB/s** against a ~102 GB/s peak, i.e. 76% of
theoretical bandwidth. No scheduling trick helps at that point; the only lever left was shrinking
the weights. Round 2 also fixed a real bug — `req.top_k` was never set, so it was zero-initialized
rather than the validated default of 50.

Round 4 needs **no engine rebuild**: `max_image_tokens_per_image` is read at runtime from the
engine's `visual/config.json`, and the change is reversible in about 4 s by restarting the shim.
Its effect was a constant **−158 ms in every `gen_tok` bucket**, which is the signature of a
fixed-cost reduction rather than a decode-rate change. It is also the one round with a quality
cost — see below.

Full derivations, including the INT4 format contract and the four verification gates, are in
[docs/optimizations.md](docs/optimizations.md).

## Caveats and limits

These are not footnotes. If you are deciding whether to adopt this recipe, these determine the
answer.

- **Quantization quality is not fully characterized.** Uncalibrated RTN with MSE-optimal clipping
  gives **11.06% mean relative weight error** (worst layer: `to_k` at **20.5%**). Spot checks at
  greedy decoding are coherent and grounded, with no repetition collapse or word salad — but **no
  benchmark suite was run**. Evaluate on a real task set before any production use.
- **There is no FP16-vs-INT4 side-by-side on identical prompts.** Both engines cannot be resident
  within 8 GB simultaneously, so the direct comparison was never made. The quality evidence is
  INT4-only.
- **Round 4's 20.8% is a quality tradeoff, not a free win.** At temperature 0, output was
  byte-identical at 320 vs 512 image tokens on a natural camera scene across three prompts,
  including one probing fine detail. On a **dense text screenshot** the 512-token run quoted
  on-screen annotations verbatim while the 320-token run drifted vaguer and partly hallucinated.
  Use **320 for live camera work, 512 for document or screenshot reading**.
- **Measurements come from live Live VLM WebUI traffic.** That is the right workload, but the
  runtime serializes requests, so probes are inflated by concurrent work. Minimums and
  matched-`gen_tok` comparisons are used to control for it; one "regression" was reported and then
  retracted after it turned out to be lock contention.
- **The 22.5× headline compares against a naive baseline** — a process spawned per request, paying
  engine deserialization every time. Against a competently configured FP16 resident baseline, the
  honest figure is **3.38× decode plus 2.32 GB of reclaimed RAM**.
- **Engine build parameters are recorded, not re-verified.** They are `--maxBatchSize 1
  --maxKVCacheCapacity 2048`; the device went offline before a final re-read of `config.json`, so
  treat these as the values used rather than as freshly confirmed output.
- **Everything here is specific to sm_87 Ampere.** FP8 weights and FP8 KV-cache quantization need
  Ada/Hopper/Blackwell/Thor; NVFP4 needs Blackwell/Thor; the fused ViT attention path
  (`USE_TRT_NATIVE_ATTN=1`) needs TensorRT ≥ 11 and fails on JetPack 7.2.1 with a misleading
  "Plugin not found" error. See [docs/negative-results.md](docs/negative-results.md).

## Requirements

Exact versions of the deployment these numbers come from. Other combinations may work; they were
not tested here.

| Component | Version / value |
|---|---|
| Device | Jetson Orin Nano Super, 8 GB LPDDR5 unified, ~102 GB/s peak bandwidth |
| GPU | Ampere **sm_87** — no FP8 hardware, no NVFP4 hardware |
| JetPack / L4T | **7.2.1** / **R39.2.1** (Ubuntu 24.04) |
| CUDA | **13.2** |
| TensorRT | **10.16.2.10** |
| cuDNN | **9.20** |
| Runtime | **TensorRT-Edge-LLM 0.10.1** |
| Model | `nvidia/Cosmos3-Edge`, reasoner task (vision tower ~411M params, text tower ~1.68B, vocab 131072) |
| Storage | NVMe SSD (915 GB) — flash to NVMe, not the SD slot |
| Swap | 2 GB swapfile |
| Power mode | MAXN_SUPER (`nvpmodel -m 2`) + `jetson_clocks` |

DeepStream 9.1 is also present on this image but is not used by this recipe. Engines are built on
the target device, per the TensorRT-Edge-LLM tutorial guidance; peak build CPU memory for the INT4
build was 3,884 MiB. This repository ships **no model weights** — you supply the official checkpoint and
quantize it locally.

## License and attribution

This repository is licensed under **Apache-2.0**.

`nvidia/Cosmos3-Edge` model weights are distributed by NVIDIA under their own terms and are not
included or redistributed here; the quantizer in `scripts/` operates on a checkpoint you obtain
yourself. Referenced third-party projects — TensorRT-Edge-LLM, TensorRT-Model-Optimizer, Live VLM
WebUI, and the Jetson AI Lab documentation — remain under their respective licenses.

Work by asotelo@nvidia.com. This is a personal engineering write-up of one deployment, not an
official NVIDIA product, release, or support commitment. The measurements describe this specific
board and software stack and carry the caveats listed above.
