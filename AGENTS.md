# One-shot: Cosmos3-Edge + Live VLM WebUI on a Jetson Orin Nano Super 8 GB

**You are an agent executing this runbook.** It takes a freshly flashed Orin Nano Super 8 GB to a
running Live VLM WebUI serving INT4-quantized Cosmos3-Edge at ~250 ms per caption.

Read this whole file before running anything. Every phase has a **gate** — a command with a stated
pass condition. **Do not proceed past a failed gate.** The gates exist because most failures in this
stack are silent: an export succeeds and the engine build fails, a build succeeds and the load OOMs,
a load succeeds and the captions are wrong. Each gate was added because that specific failure
actually happened during the original deployment.

## Ground truth before you start

| | |
|---|---|
| Board | Jetson Orin Nano Super **8 GB** (sm_87) |
| Software | JetPack 7.2.1 / L4T R39.2.1, CUDA 13.2, TensorRT 10.16.2.10 |
| Runtime | TensorRT-Edge-LLM **v0.10.1** |
| Model | `nvidia/Cosmos3-Edge` |
| Expected result | ~250 ms typical caption, ~4.8 GB of 7.4 GB RAM, ~15% CPU |

**This board has no NVENC and no FP8.** Both are sm_89+ features. If a step wants either, the step
is wrong — do not go looking for a driver fix.

**Two hosts are involved.** ONNX export runs on an x86 workstation; quantization, engine builds and
serving run on the Jetson. The export step is the only one that is not on-device, because
`tensorrt-edgellm-export` is not packaged for aarch64 in this release.

---

## Phase 0 — Preflight

```bash
bash oneshot/preflight.sh
```

**Gate:** exits 0. It refuses to continue on a board that is not 8 GB, on a JetPack other than 7.x,
or with CUDA/TensorRT majors that do not match. A version mismatch here produces failures three
phases later that look like something else entirely.

## Phase 1 — Platform tuning

```bash
sudo nvpmodel -m 2      # MAXN_SUPER
sudo jetson_clocks
```

**Gate:** `nvpmodel -q` reports `MAXN_SUPER`. GPU should clock to 1020 MHz and EMC to 3199 MHz.

This is part of the single largest win in the campaign (13.91 s → 2.07 s), bundled with the resident
runtime in Phase 5 — the two were never measured separately, so do not attribute the delta to either
alone.

## Phase 2 — Install TensorRT-Edge-LLM v0.10.1

On **both** the Jetson and the x86 host:

```bash
git clone https://github.com/NVIDIA/TensorRT-edge-llm.git ~/TensorRT-Edge-LLM
cd ~/TensorRT-Edge-LLM
git checkout v0.10.1
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
python3 -m venv .venv && .venv/bin/pip install -e .
```

**Use v0.10.1 specifically.** Cosmos3-Edge support landed in v0.10.0 (PR #171) and v0.10.1 (PR #196)
fixed the Cosmos3 Reasoner **visual patch layout** — the vision path this deployment depends on.

**Gate:** `ls ~/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so` succeeds, and

```bash
export EDGELLM_PLUGIN_PATH=$HOME/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so
```

resolves to a real file. **`EDGELLM_PLUGIN_PATH` defaults to a *relative* path.** If you leave it
unset, TensorRT fails with `Plugin not found, are the plugin name, version, and namespace correct?`
at build *or* at deserialization — two different phases, same unhelpful message. `serve/cosmos3_shim.py`
sets it via `os.environ.setdefault`, so serving is covered; interactive builds are not.

## Phase 3 — Fetch the checkpoint

```bash
huggingface-cli download nvidia/Cosmos3-Edge --local-dir ~/Cosmos3-Edge
```

**Gate:** `~/Cosmos3-Edge` contains `transformer/` and `vision_encoder/` subdirectories. This is a
**modular checkpoint**, which matters in Phase 4.

> `vision_encoder/model.safetensors` may be a **symlink into the HF blob store**. Do not `rm -f` it
> and write in place — you will delete the blob the snapshot shares. Write to a new path.

## Phase 4 — INT4 quantize (Jetson or host, CPU only)

```bash
python3 scripts/rtn_int4_quantize.py \
  --src ~/Cosmos3-Edge \
  --dst ~/cosmos3_int4_ckpt
```

Calibration-free round-to-nearest with an MSE-optimal clipping search. No GPU, no calibration data,
runs in fp32 on CPU.

**Do not substitute modelopt here.** Its CUDA path emits weights of the right *shape* with wrong
values on some hosts, and its AWQ path needs activation statistics this toolchain cannot produce —
the attention plugin is a shape-only stub returning `torch.zeros(...)` for ONNX export. modelopt is
useful in this project only as a *format oracle*; see `docs/optimizations.md`.

**Gate:** mean relative error prints in the **10–12%** range and every quantized tensor is non-zero.
An all-zero or near-zero weight file is the known failure mode and it will still export and build
cleanly — it only shows up as garbage captions at the very end.

Only 64-aligned linears are quantized: the kernel needs `out_features % 64 == 0` **and**
`in_features % 64 == 0`. The ViT MLP is 1152↔4304 and `4304 % 64 == 16`, so those layers stay FP16
by design. That is expected output, not a bug.

## Phase 5 — Export and build engines

**Export (x86 host):**

```bash
tensorrt-edgellm-export ~/cosmos3_int4_ckpt ~/cosmos3_int4_onnx \
  --task reasoning --skip-visual
```

`--task reasoning` is required for Cosmos3-Edge. It is also the workaround for
`tensorrt-edgellm-quantize` failing with `KeyError: 'cosmos3_edge'`.

**Gate:** the exported LLM ONNX contains **169 `Int4GroupwiseGemmPluginV2` nodes**, one per
quantized linear, and its INT8 initializers are 100% non-zero. Verify both before building — this is
the last point where a bad quantization is cheap to catch.

**Build the LLM engine (Jetson):**

```bash
llm_build --onnxDir ~/cosmos3_int4_onnx/llm \
          --engineDir /opt/tensorrt-edgellm/models/default \
          --maxBatchSize 1 --maxKVCacheCapacity 2048
```

`--maxBatchSize 4 --maxKVCacheCapacity 4096` **builds fine and then OOMs at engine load** on 8 GB.
Batch 1 / KV 2048 is what fits.

**Build the visual engine (Jetson):**

```bash
visual_build --onnxDir ~/cosmos3_int4_onnx/visual \
             --engineDir /opt/tensorrt-edgellm/models/default
```

> **Known gap, stated rather than guessed:** the exact `visual_build` argument values used in the
> original deployment were never recorded. Release defaults are `--minImageTokens 4
> --maxImageTokens 1024 --maxImageTokensPerImage 512`. Build with defaults — Phase 7 tunes the value
> that actually matters at runtime, so you do not need to get it right at build time.

**Gate:** `llm.engine` is **~0.818 GB** (down from 3.135 GB at FP16) and the visual engine is
**~938 MB**. The visual engine being larger than the INT4 text engine is correct and is exactly why
the vision tower becomes the next bottleneck.

## Phase 6 — Install the shim and the WebUI

```bash
sudo mkdir -p /opt/tensorrt-edgellm
sudo cp serve/cosmos3_shim.py /opt/tensorrt-edgellm/cosmos3_shim_v1.py
~/TensorRT-Edge-LLM/.venv/bin/pip install live-vlm-webui
sudo install -m 0644 systemd/cosmos3-edge-shim.service \
                     systemd/live-vlm-webui.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now cosmos3-edge-shim.service live-vlm-webui.service
```

The shim holds ONE `LLMRuntime` for the process lifetime and calls
`capture_decoding_cuda_graph()`. The original path shelled out to `llm_inference` per request,
paying ~6–8 s of engine deserialization every single time — that is the cost this removes.

The WebUI needs no adapter: it speaks the OpenAI chat-completions API, so
`--api-base http://localhost:8000/v1 --model nvidia/Cosmos3-Edge` is the entire integration.
`--model` must match the id the shim advertises.

**Gate:** both units are `active`, `curl localhost:8000/v1/models` returns
`nvidia/Cosmos3-Edge`, and `https://<jetson-ip>:8090/` serves the UI.

## Phase 7 — Tune the image-token budget

```bash
sudo python3 - <<'PY'
import json
p = "/opt/tensorrt-edgellm/models/default/visual/config.json"
c = json.load(open(p))
c["max_image_tokens_per_image"] = 320
json.dump(c, open(p, "w"), indent=1)
PY
sudo systemctl restart cosmos3-edge-shim.service
```

`max_image_tokens_per_image` is read **at runtime**, so this needs no rebuild. It feeds `smartResize`
(`maxPixels = tokens × 32²`). Lowering 512 → 320 removed a constant ~158 ms from the fixed cost.

**Gate:** latency drops and captions stay coherent. This is a **characterized tradeoff**, not a free
win — read `docs/report.md` §"Round 4 quality tradeoff" before pushing it lower.

## Phase 8 — Verify

```bash
bash oneshot/verify.sh
```

**Gate — all of these, on an otherwise idle board:**

| Measure | Expected |
|---|---|
| Best single caption | ~245 ms |
| Typical (uncontended) | ~253 ms |
| Latency model | `elapsed_ms ≈ 200 + 13.5 × generated_tokens` |
| Steady-state decode | ~13.5 ms/token (~74 tok/s) |
| System RAM | ~4.8 GB of 7.4 GB |
| CPU | ~15% **system-wide across 6 cores** (GPU-bound, so this is correct, not suspicious) |
| GPU | ~98% |

**Two traps that will corrupt this measurement:**

1. **The encoder embedding cache.** The runtime caches vision embeddings keyed on frame content.
   Benchmarking the same image repeatedly measures the cache, not the model — it silently skips
   ~248 ms of ViT. Vary the input.
2. **Concurrent clients.** A live WebUI browser tab drives ~1 request/sec on its own. Close it, or
   you are measuring contention. Check GPU utilisation is 0% before you start.

---

## If a caption is wrong rather than slow

Latency and correctness fail independently here. A fast, confident, wrong caption means Phase 4 —
go back and check the quantized weights are non-zero, then check the 169-node ONNX gate. Everything
downstream of a bad quantization succeeds.

## Scope

This branch is **frozen** and covers Cosmos3-Edge + Live VLM WebUI only. The NVR layer (Frigate,
ring-mqtt, alert policy) lives on `main` and is deliberately absent here. Full measurement
methodology, negative results and the ecosystem survey are in `docs/`; step-by-step prose with far
more detail than this runbook is in `deploy/`.
