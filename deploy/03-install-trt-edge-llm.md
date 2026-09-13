# 03 — Installing TensorRT-Edge-LLM 0.10.1

Target: **Jetson Orin Nano Super, 8 GB**, JetPack 7.2.1 / L4T R39.2.1, Ubuntu 24.04,
CUDA 13.2, TensorRT 10.16.2.10, cuDNN 9.20. Runtime version used throughout this repo is
**TensorRT-Edge-LLM 0.10.1**.

By the end of this document you have two working environments:

| Environment | Machine | Produces |
|---|---|---|
| C++ runtime + tools | the Jetson (target device) | `llm_build`, `visual_build`, `llm_inference`, `libNvInfer_edgellm_plugin.so`, the `_edgellm_runtime` pybind module |
| Python export package | a host workstation | ONNX graphs (`model.onnx` + external `model.onnx.data`) for the text and vision towers |

Nothing here is model-specific yet. Quantization and engine building are covered in the
later steps; this document is only about getting the toolchain to exist and to load its
plugins.

---

## 1. Why the work is split across two machines

This is the single most important structural fact about the toolchain, and getting it wrong
costs a full rebuild cycle.

**ONNX export is device-agnostic. TensorRT engine building is not.**

TensorRT does not emit portable binaries. `llm_build` and `visual_build` run tactic
selection against the GPU that is physically present — timing candidate kernels, choosing
layouts, and serializing the winners into the `.engine` file along with a hard compatibility
record of the SM architecture and the TensorRT version. An engine serialized on a desktop
Blackwell card will not deserialize on Orin's sm_87, and an engine built against a different
TensorRT minor version will not deserialize either.

So the pipeline is:

```
host workstation                          Jetson Orin Nano (sm_87)
────────────────                          ────────────────────────
HF checkpoint
   │
   ├─ scripts/rtn_int4_quantize.py  (CPU)
   │     → W4A16 checkpoint
   │
   └─ tensorrt-edgellm-export             copy ONNX over the network
         → model.onnx + .data       ───▶     │
                                             ├─ llm_build     → LLM engine
                                             ├─ visual_build  → vision engine
                                             └─ llm_inference / resident shim
```

Two secondary reasons reinforce the split, both specific to an 8 GB board:

- Export loads the full-precision checkpoint into host RAM and writes multi-gigabyte ONNX
  external-data files. On this project the FP16 text tower's `model.onnx.data` was **3.36 GB**
  and the INT4 one **0.807 GB**; the vision tower's ONNX external data was **984,940,544 bytes**.
- Engine building on the Jetson is itself memory-hungry. The INT4 LLM engine build peaked at
  **3,884 MiB** of CPU memory on a device with 8 GB unified LPDDR5 and a 2 GB swapfile. Doing
  export on the same box as well is asking for the OOM killer.

The one thing you cannot move to the host is the engine build. That has to happen on the
Jetson.

> Host GPU note: the host used for this project had an **sm_120** GPU, and modelopt's CUDA
> extension was broken on it (emitting zeros in fp32 and NaN in fp16). The quantizer in this
> repo therefore runs entirely on **CPU** and needs no working host GPU at all. If your host
> has a healthy GPU, that is a convenience, not a requirement.

---

## 2. Clone the repository

Do this on **both** machines. The Jetson needs the `cpp/` tree; the host needs the
`tensorrt_edgellm/` Python package. Keeping the same commit on both sides is what guarantees
the ONNX your host emits matches the graph the device's `llm_build` expects.

On the Jetson (this project used `/home/orin/TensorRT-Edge-LLM`, which is the path the
systemd units and the serving shim in this repo assume):

```bash
git clone https://github.com/NVIDIA/TensorRT-edge-llm.git ~/TensorRT-Edge-LLM
cd ~/TensorRT-Edge-LLM
git tag -l            # confirm the exact tag spelling for 0.10.1 before checking out
git checkout <0.10.1-tag>
```

The release used here is **0.10.1**. The exact tag string was not recorded during this
deployment, so check `git tag -l` rather than assuming a `v` prefix. Pin it explicitly — do
not build from a moving `main`, because the ONNX plugin node names and the plugin library's
registered creators must agree between the export package and the runtime.

Repeat the clone and checkout on the host workstation.

---

## 3. Build the C++ runtime, tools, and Python bindings (Jetson)

The build is a standard out-of-source CMake configure-and-build:

```bash
cd ~/TensorRT-Edge-LLM
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
```

> The exact CMake option set used in this deployment was not recorded, so treat the repo's
> own `README` / `docs` as authoritative for flags — in particular any option that gates the
> pybind11 bindings, which you **do** need (see below). The generic invocation above is shown
> so the artifact paths in the rest of this repo make sense, not as a verified flag list.

### What you need out of the build

| Artifact | Used by | Purpose |
|---|---|---|
| `build/llm_build` | step 04 | Compiles the text-tower ONNX into a TensorRT engine |
| `build/visual_build` | step 04 | Compiles the vision-tower ONNX into a TensorRT engine |
| `build/llm_inference` | smoke tests | One-shot CLI: loads engines, runs one request, exits |
| `build/libNvInfer_edgellm_plugin.so` | both builders **and** the runtime | Custom TensorRT plugins (see section 4) |
| `build/pybind/_edgellm_runtime*.so` | [`serve/cosmos3_shim.py`](../serve/cosmos3_shim.py) | Python bindings for the resident `LLMRuntime` |

The pybind module is not optional for this deployment. `llm_inference` is a **process-per-request**
interface — it deserializes ~4.1 GB of engines on every invocation, which is exactly the
baseline this project measured at **13.91 s per request**. The entire Round 1 win (down to
2.07 s/req) comes from holding one `LLMRuntime` resident for the life of a process via the
bindings, so if `build/pybind/` is empty you cannot reproduce any result in this repo. See
[`serve/cosmos3_shim.py`](../serve/cosmos3_shim.py), which does:

```python
sys.path.insert(0, "/home/orin/TensorRT-Edge-LLM/build/pybind")
import _edgellm_runtime as rt
```

### Memory during compilation

Compiling CUDA/C++ with one job per core on an 8 GB board puts several heavyweight compiler
processes in flight against the same unified memory the GPU uses. This deployment did not
record a compile-time OOM, so treat this as precaution rather than a measured failure: if
the build dies with a killed compiler, drop to `-j4` or `-j2` and retry. The compile is a
one-time cost; the recurring memory pressure that *was* measured is at engine-build time
(3,884 MiB peak, section 1).

---

## 4. The plugin library — `libNvInfer_edgellm_plugin.so`

TensorRT-Edge-LLM does not express everything in stock ONNX ops. Two plugins matter for this
project:

- **`Int4GroupwiseGemmPluginV2`** (cuteDSL) — the W4A16 matmul. The INT4 text-tower ONNX
  exported here contained **169** of these nodes, one per quantized linear.
- **`ViTAttentionPlugin`** — the vision tower's attention.

Both live in `libNvInfer_edgellm_plugin.so`. TensorRT resolves plugin creators by
(name, version, namespace) at **both** build time and deserialization time, so the library
must be discoverable in both phases. If it is not, you get:

```
[6] creator && "Plugin not found, are the plugin name, version, and namespace correct?"
```

### Making it discoverable

```bash
export EDGELLM_PLUGIN_PATH=$HOME/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so
# belt and braces — some entrypoints only pick the library up if it is already mapped:
export LD_PRELOAD=$EDGELLM_PLUGIN_PATH
```

Put this in the environment of every process that touches an engine: `llm_build`,
`visual_build`, `llm_inference`, and the serving process. The shim in this repo sets it
in-process before importing the bindings, which is the more robust pattern because it
survives being launched by a supervisor that scrubs the environment:

```python
os.environ.setdefault(
    "EDGELLM_PLUGIN_PATH", "/home/orin/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so"
)
```

For a systemd-managed service, set it with an `Environment=` line in the unit rather than
relying on a login shell profile — units do not source `~/.bashrc`. See
[`systemd/cosmos3-edge-shim.service`](../systemd/cosmos3-edge-shim.service) for the unit as
deployed.

### The error message lies about the cause at least once

"Plugin not found" is also what you get for a completely unrelated problem. Setting
`USE_TRT_NATIVE_ATTN=1` to export fused ViT attention produces a clean ONNX export — 27
`TRT_Attention` ops plus a `kv_lengths` input — and then `visual_build` fails with the same
string. The real cause there is that `TRT_Attention` **requires TensorRT ≥ 11**
(docstring at `tensorrt_edgellm/models/ops.py:384`), and JetPack 7.2.1 ships **10.16.2.10**.
`libNvInfer_edgellm_plugin.so` provides `ViTAttentionPlugin` but not `TRT_Attention`, so no
amount of `LD_PRELOAD` will fix it.

Triage rule: if the plugin path is set and you still get "Plugin not found", stop looking at
the path and start looking at whether the ONNX contains an op your TensorRT version cannot
support.

---

## 5. Verify the device install

The cheapest end-to-end check is to load an existing engine directory with `llm_inference`
and confirm it reaches generation. If you have no engine yet, this check moves to step 04 —
there is nothing useful to verify before the first engine exists.

Once a resident runtime is in play, the load line from the shim is the practical health
signal. This deployment observed:

| Engine | `LLMRuntime` construction time |
|---|---|
| INT4 W4A16 | 6.28 s |
| FP16 | 7.79 s |

A construction that throws instead of printing a time is almost always the plugin path
(section 4) or an engine/TensorRT version mismatch (section 1). A construction that succeeds
but is dramatically slower than the above usually means the device is not in its performance
mode — `sudo nvpmodel -m 2 && sudo jetson_clocks` for MAXN_SUPER.

---

## 6. Host-side export environment

On the host workstation, create an isolated virtualenv and install the Python package from
the checked-out tree:

```bash
cd ~/TensorRT-Edge-LLM
python3 -m venv .venv-export
source .venv-export/bin/activate
pip install -e .            # confirm the install target against the repo's own instructions
pip list | grep -i edgellm  # confirm the version is 0.10.1
```

This gives you the `tensorrt_edgellm` Python package and its console entrypoints. The two
that matter are `tensorrt-edgellm-export` and `tensorrt-edgellm-quantize`.

You also need `torch` and `safetensors` for [`scripts/rtn_int4_quantize.py`](../scripts/rtn_int4_quantize.py),
which is pure CPU tensor work (round-to-nearest with an MSE-optimal clipping sweep, no
calibration data, no GPU kernels). A CPU-only torch wheel is sufficient.

`nvidia-modelopt` is **optional** and, in this project, was not used as a quantizer at all —
only as a format oracle, run on CPU, to extract the ground-truth W4A16 on-disk contract
(`scale = amax/7`, `q = round(w/scale).clamp(-8, 7)`, `[N//2, K]` uint8 nibble packing, fp32
`[N, K//128]` companion scales). If you trust the contract as implemented in
`scripts/rtn_int4_quantize.py`, you can skip installing modelopt entirely.

### Gotcha: `tensorrt-edgellm-quantize` does not know this model

The obvious entrypoint fails:

```
KeyError: 'cosmos3_edge'
```

`tensorrt-edgellm-quantize` has no registration for the `cosmos3_edge` architecture. The
working entrypoint is the exporter with an explicit task. It takes two positional arguments,
in this order: `model` (a local checkpoint directory or a Hugging Face model ID) and
`output_dir`:

```bash
tensorrt-edgellm-export <model> <output_dir> --task reasoning
```

`output_dir` is a **root** directory, not the ONNX directory itself: the exporter creates
`llm/`, `visual/` and `audio/` subdirectories underneath it as needed. That is where the
`llm/` level in every downstream path comes from — `<output_dir>/llm` is what you later hand
to `llm_build --onnxDir`, and it is what has to land at the shim's `CHECKPOINT_DIR`
(section 7).

When you have already built the vision engine and only want to re-export the text tower, add
`--skip-visual` (there is a matching `--skip-llm` for the reverse case):

```bash
tensorrt-edgellm-export cosmos3_int4_ckpt cosmos3_int4_onnx --task reasoning --skip-visual
```

A related trap, since it lives on the same CLI: **`--quantization int4_awq` is a silent no-op
on this model.** The flag exists, but it is gated behind `_needs_moe_quantization`
(`tensorrt_edgellm/scripts/export.py:1046,1074`, dropped again at `export.py:4129-4131`) and
applies to Mixture-of-Experts models only. Cosmos3-Edge is dense, so passing the flag
produces an FP16 export with no error and no warning. That is why this repo quantizes the
checkpoint itself before export rather than asking the exporter to do it. Full reasoning in
[`../docs/report.md`](../docs/report.md).

### Disk budget

Both sides need real space. Measured sizes from this deployment:

| Artifact | Size |
|---|---|
| Vision-tower ONNX external data | 984,940,544 B |
| Vision-tower engine (FP16) | 938 MB |
| Text-tower ONNX external data, FP16 | 3.36 GB |
| Text-tower ONNX external data, INT4 | 0.807 GB |
| Text-tower engine, FP16 | 3,366,196,508 B (3.135 GB) |
| Text-tower engine, INT4 W4A16 | 878,552,564 B (0.818 GB) |

The Jetson in this project ran off an NVMe SSD (915 GB), which is far more than enough; an
SD-card install is not recommended for holding checkpoints, ONNX and engines simultaneously.

---

## 7. Moving ONNX from the host to the Jetson

Reference the device through the `JETSON_HOST` environment variable, the same convention the
two host-side measurement scripts in this repo use ([`scripts/collect_perf.py`](../scripts/collect_perf.py)
and [`scripts/compare_perf.py`](../scripts/compare_perf.py), both defaulting to
`orin@jetson.local`; the remaining scripts run on the device against loopback):

```bash
export JETSON_HOST=${JETSON_HOST:-orin@jetson.local}
ssh "$JETSON_HOST" 'mkdir -p ~/tensorrt-edgellm-workspace/Cosmos3-Edge/onnx/reasoning'
rsync -avP --info=progress2 cosmos3_int4_onnx/ "$JETSON_HOST":~/tensorrt-edgellm-workspace/Cosmos3-Edge/onnx/reasoning/
```

Copy the **whole directory**. ONNX external data (`model.onnx.data`) is referenced by
relative path from `model.onnx`; moving the `.onnx` file alone yields a graph that loads and
then fails on missing initializers.

Mind the trailing slash on the source. `cosmos3_int4_onnx/` is the exporter's `output_dir`,
so what gets copied is its *contents* — the `llm/` and `visual/` subdirectories the exporter
created (section 6) — landing inside `.../onnx/reasoning/`. The text tower therefore ends up
at `/home/orin/tensorrt-edgellm-workspace/Cosmos3-Edge/onnx/reasoning/llm`, which is exactly
the `CHECKPOINT_DIR` constant in [`serve/cosmos3_shim.py`](../serve/cosmos3_shim.py). Drop the
trailing slash and you get a nested `reasoning/cosmos3_int4_onnx/llm` that the shim will not
find. Built engines are installed separately, under `/opt/tensorrt-edgellm/models/default`.

---

## 8. Gotcha: the `--system-site-packages` venv ABI mismatch

On JetPack, TensorRT, CUDA Python bindings and several other components arrive as system
`.deb` packages, not PyPI wheels. The usual workaround is to create the device-side venv with
`--system-site-packages` so the venv can see them:

```bash
python3 -m venv --system-site-packages ~/TensorRT-Edge-LLM/.venv
```

This works, and it is what the units in this repo assume
([`systemd/cosmos3-edge-shim.service`](../systemd/cosmos3-edge-shim.service) and
[`systemd/live-vlm-webui.service`](../systemd/live-vlm-webui.service) both run
`/home/orin/TensorRT-Edge-LLM/.venv/bin/...`). But it creates a specific failure mode:

`pip install` inside the venv can pull a **newer `numpy`** into the venv while `pandas` and
`matplotlib` continue to resolve to the **system** copies compiled against the older numpy
ABI. Import then fails deep inside a C extension.

The symptom is nastier than a traceback: **the Gradio UI silently returns nothing.** Gradio
catches the exception on the event-handler path, so the browser just sits there with no
output and no error, while the model, the engines and the shim are all completely healthy.
It is easy to spend an hour blaming the runtime for a plotting-library ABI break.

Fix — install matching `pandas` and `matplotlib` *inside* the venv so all three come from the
same ABI generation:

```bash
source ~/TensorRT-Edge-LLM/.venv/bin/activate
pip install --force-reinstall pandas matplotlib
python -c "import numpy, pandas, matplotlib; print(numpy.__version__, pandas.__version__, matplotlib.__version__)"
```

If that one-liner prints three versions without throwing, the UI will render. Run it directly
rather than trusting the web UI to report the problem — it will not.

---

## 9. Known failure modes on this platform

| Symptom | Cause | Action |
|---|---|---|
| `Plugin not found, are the plugin name, version, and namespace correct?` at build or load | `libNvInfer_edgellm_plugin.so` not discoverable | Set `EDGELLM_PLUGIN_PATH` and/or `LD_PRELOAD` (section 4) |
| Same message, path already set, `USE_TRT_NATIVE_ATTN=1` in use | `TRT_Attention` needs TensorRT ≥ 11; JetPack 7.2.1 has 10.16.2.10 | Do not use `USE_TRT_NATIVE_ATTN` on Orin |
| `KeyError: 'cosmos3_edge'` | `tensorrt-edgellm-quantize` has no entry for this architecture | Use `tensorrt-edgellm-export --task reasoning` |
| Gradio returns nothing, no error | numpy/pandas or numpy/matplotlib ABI mismatch in a `--system-site-packages` venv | Reinstall pandas/matplotlib in the venv (section 8) |
| Engine deserialization fails on the Jetson | Engine built on the host, or against a different TensorRT version | Rebuild the engine on the device (section 1) |
| OOM at engine load | Build parameters too large; batch=4 / KV=4096 OOMed on this board | Rebuild at `--maxBatchSize 1 --maxKVCacheCapacity 2048` |
| Script exits with rc=127 referencing `time` | `/usr/bin/time` does not exist on this Jetson image | Use shell builtins or Python timing instead |

The `--maxBatchSize 1 --maxKVCacheCapacity 2048` values are the ones used for the engines
behind this repo's measurements. The Jetson went offline before a final re-read of the
engine's `config.json`, so treat those as the parameters passed, not as freshly verified
output. For reference, the Jetson AI Lab TensorRT-Edge-LLM tutorial recommends
`maxInputLen 512` / `maxKVCacheCapacity 1024` for Orin Nano, and
`--externalize-weights int4_ffn` to cut peak build memory — neither was exercised here.

---

## 10. Next

With the toolchain built and the plugin loading, the remaining steps are quantizing the
checkpoint, exporting ONNX on the host, and building the engines on the device. The
quantizer is [`scripts/rtn_int4_quantize.py`](../scripts/rtn_int4_quantize.py); the full
rationale for every choice above, including the routes that were tried and closed, is in
[`docs/report.md`](../docs/report.md).
