# Troubleshooting

Symptom-first guide for bringing up Cosmos3-Edge on a Jetson Orin Nano Super (8 GB) with
TensorRT-Edge-LLM. Entries are ordered by what you see, not by subsystem. Most entries below were
hit during this deployment; entries that were instead derived from reading the shipped artifacts, or
that are preventive rather than an observed failure, say so in the entry itself.

**Deployment this was observed on:** JetPack 7.2.1 / L4T R39.2.1, Ubuntu 24.04, CUDA 13.2,
TensorRT 10.16.2.10, cuDNN 9.20, TensorRT-Edge-LLM 0.10.1, NVMe root, 2 GB swapfile.
Behaviour on other JetPack or runtime versions was not tested here; where a fix depends on a
version, that is called out in the entry.

**Conventions used in all commands**

```bash
export JETSON_HOST="${JETSON_HOST:-orin@jetson.local}"   # never hardcode an IP
```

Commands prefixed with `ssh "$JETSON_HOST"` run from your workstation; bare commands run in a
shell on the Jetson itself. Paths such as `/home/orin/TensorRT-Edge-LLM` and
`/opt/tensorrt-edgellm/models/default` are the ones used by
[`serve/cosmos3_shim.py`](../serve/cosmos3_shim.py) and the
[systemd units](../systemd/cosmos3-edge-shim.service) in this repo; adjust for your layout.

## Index

| You see | Go to |
|---|---|
| Host `lsusb` shows no Jetson at all | [1](#1-jetson-does-not-appear-in-lsusb) |
| SDK Manager flash aborts early, network-ish error | [2](#2-sdk-manager-flash-fails-because-ipv6-is-disabled-on-the-host) |
| Device boots but the NVMe is empty / rootfs is tiny | [3](#3-flashed-to-the-sd-slot-instead-of-nvme) |
| Device boots, but no CUDA / TensorRT / DeepStream on it | [4](#4-sdk-manager-never-pushes-the-target-components) |
| `KeyError: 'cosmos3_edge'` | [5](#5-keyerror-cosmos3_edge-from-tensorrt-edgellm-quantize) |
| `Plugin not found, are the plugin name, version, and namespace correct?` | [6](#6-plugin-not-found--two-different-causes) |
| Engine build dies, OOM killer, or machine freezes during build | [7](#7-oom-while-building-the-engine) |
| Runtime OOMs at engine load / service never reaches "ready" | [8](#8-oom-at-engine-load) |
| Service restarts in a loop, `No such file or directory` | [9](#9-cosmos3-edge-shimservice-restart-loop) |
| `ModuleNotFoundError: No module named '_edgellm_runtime'` | [10](#10-modulenotfounderror-_edgellm_runtime) |
| numpy ABI error, or Gradio silently returns nothing | [11](#11-numpy-abi-mismatch-in-a---system-site-packages-venv) |
| Script exits `rc=127`, `/usr/bin/time: No such file or directory` | [12](#12-usrbintime-does-not-exist-rc127) |
| Benchmark numbers that are too good to be true | [13](#13-benchmark-results-that-look-impossibly-fast) |
| A "regression" that disappears when you re-run it | [14](#14-a-regression-that-vanishes-on-re-measurement) |
| `all gen_tok identical - cannot separate fixed/marginal`, or R-squared collapse | [15](#15-regression-fit-degenerates-or-r-squared-collapses) |
| `prompt_tok` stuck at 510/511 regardless of camera resolution | [16](#16-prompt_tok-pinned-at-510511) |
| `TypeError: 'NoneType' object is not subscriptable` in `tok_vs_res.py` | [17](#17-typeerror-nonetype-object-is-not-subscriptable-in-tok_vs_respy) |
| It fails and it is *supposed* to fail on this hardware | [Not bugs](#not-bugs-things-that-cannot-work-on-orin-sm_87) |

---

## Flashing and bring-up

### 1. Jetson does not appear in `lsusb`

**Symptom.** With the device connected to the host by USB and powered, `lsusb` on the host lists no
NVIDIA device, and SDK Manager reports no target found.

**Cause.** The board is not in Force Recovery Mode. On the Elecrow case used here, the case's
recovery button does nothing without the corresponding firmware support, so pressing it is not a
substitute.

**Fix.** Power off, jumper **FC REC to GND** on the carrier board header, then power on with the
jumper in place.

**Verify.**

```bash
lsusb | grep -i 0955
```

| USB ID | Meaning |
|---|---|
| `0955:7523` | APX device — in Force Recovery Mode, ready to flash |
| `0955:7020` | Normally booted device — **not** in recovery, flashing will not start |

### 2. SDK Manager flash fails because IPv6 is disabled on the host

**Symptom.** Flash aborts during the host-side setup phase on a machine where IPv6 has been
disabled by policy.

**Cause.** SDK Manager requires IPv6 on the host.

**Fix.** Re-enable it before launching SDK Manager:

```bash
sudo sysctl -w net.ipv6.conf.all.disable_ipv6=0
sudo sysctl -w net.ipv6.conf.default.disable_ipv6=0
```

**Verify.** Both values read `0`, then re-run the flash:

```bash
sysctl net.ipv6.conf.all.disable_ipv6 net.ipv6.conf.default.disable_ipv6
```

### 3. Flashed to the SD slot instead of NVMe

**Symptom.** Flash reports success, but the device boots from (or fails to boot from) the SD slot
and the NVMe SSD is untouched.

**Cause.** SDK Manager defaulted the storage target to `/dev/mmcblk0` (the SD slot). The target is
not inferred from what is physically fastest or largest — it must be selected.

**Fix.** In SDK Manager's target configuration, explicitly select the **NVMe** device as the storage
target, then re-flash.

**Verify.** On the booted device, the root filesystem should be on the NVMe, not on `mmcblk0`:

```bash
ssh "$JETSON_HOST" 'findmnt -no SOURCE / ; lsblk -o NAME,SIZE,TYPE,MOUNTPOINT'
```

This deployment runs root on a 915 GB NVMe SSD.

### 4. SDK Manager never pushes the target components

**Symptom.** The device boots a working L4T image, but CUDA, cuDNN, TensorRT and DeepStream are
absent on the target; SDK Manager shows the host-side steps complete and the target-side component
install either skipped or failed.

**Cause.** Observed in this bring-up; the root cause was not isolated. SDK Manager completed the
flash but never performed the on-target component installation.

**Fix.** Copy the component `.deb` packages that SDK Manager downloaded on the host to the device
and install them there manually. On the host the downloads live under SDK Manager's download
directory (default `~/Downloads/nvidia/sdkm_downloads`); copy them over and `dpkg -i` /
`apt install ./*.deb` on the Jetson.

**Verify.** On the device:

```bash
ssh "$JETSON_HOST" 'nvcc --version; dpkg -l | grep -E "cuda|cudnn|tensorrt|deepstream" | head -40'
```

Expected on this deployment: CUDA 13.2, cuDNN 9.20, TensorRT 10.16.2.10, DeepStream 9.1.

---

## Export, quantization and engine build

### 5. `KeyError: 'cosmos3_edge'` from `tensorrt-edgellm-quantize`

**Symptom.**

```
KeyError: 'cosmos3_edge'
```

raised by `tensorrt-edgellm-quantize`.

**Cause.** The quantize entry point has no registered handler for the `cosmos3_edge` model type.

**Fix.** Do not use `tensorrt-edgellm-quantize` for this model. Use the export CLI with the
reasoning task instead:

```bash
tensorrt-edgellm-export <ckpt_dir> <onnx_dir> --task reasoning
```

If you want INT4 weights, quantize the checkpoint yourself first with
[`scripts/rtn_int4_quantize.py`](../scripts/rtn_int4_quantize.py) and export the result. Note that
`tensorrt-edgellm-export --quantization int4_awq` is **not** an alternative here: the flag is gated
behind Mixture-of-Experts handling and is a silent no-op on a dense model like Cosmos3-Edge. See
[Not bugs](#not-bugs-things-that-cannot-work-on-orin-sm_87).

The two positional arguments are `model output_dir`, and the exporter creates `llm/`, `visual/` and
`audio/` subdirectories under `output_dir` as needed — so the LLM ONNX lands at `<onnx_dir>/llm/`
and the vision tower at `<onnx_dir>/visual/`. Everything downstream (`llm_build --onnxDir`, the
verification snippets below) points at those subdirectories, not at `<onnx_dir>` itself.

**Verify.** Export completes and `<onnx_dir>/llm/` contains the LLM subgraph plus its external data
file. For an INT4 export, `<onnx_dir>/llm/model.onnx.data` was **0.807 GB** here versus **3.36 GB**
for FP16.

### 6. "Plugin not found" — two different causes

**Symptom.**

```
[6] creator && "Plugin not found, are the plugin name, version, and namespace correct?"
```

at engine build or engine load. The message names the wrong problem in one of the two cases below,
so identify which one you have *before* changing anything.

**Triage first.** Determine whether the missing plugin is a genuinely unloaded plugin library or a
`TRT_Attention` op that your TensorRT cannot provide:

```bash
# (a) is the plugin library loaded / on the path?
echo "$EDGELLM_PLUGIN_PATH"
ls -l /home/orin/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so

# (b) heuristic: which attention plugins does that library carry?
#     (C++ symbols are mangled; look for ViTAttentionPlugin, and note the absence of
#      TRT_Attention -- it is a TensorRT-native op, never an Edge-LLM plugin)
nm -D --defined-only /home/orin/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so \
  | grep -i attention | head

# (c) does the ONNX you are building contain TRT_Attention nodes?
#     the vision tower lives in the exporter's visual/ subdirectory
grep -ac TRT_Attention <onnx_dir>/visual/model.onnx   # approximate: byte match, not graph parse
```

#### 6a. Genuinely missing plugin path

**Cause.** TensorRT never loaded `libNvInfer_edgellm_plugin.so`, so none of the Edge-LLM plugin
creators are registered.

**Fix.** Point the runtime at the library, by env var and/or `LD_PRELOAD`:

```bash
export EDGELLM_PLUGIN_PATH=/home/orin/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so
export LD_PRELOAD="$EDGELLM_PLUGIN_PATH"
```

The shim in this repo sets `EDGELLM_PLUGIN_PATH` itself via `os.environ.setdefault` before importing
`_edgellm_runtime`, so it is immune to this as long as the path on that line is correct for your
install ([`serve/cosmos3_shim.py`](../serve/cosmos3_shim.py)).

**Verify.** Re-run the failing build/load; the error disappears and the engine deserializes. From
the shim, a successful load prints `[shim] LLMRuntime constructed in <n>s` followed by
`[shim] ready`.

#### 6b. `TRT_Attention` requiring TensorRT >= 11 — **not fixable by setting the plugin path**

**Symptom.** The same "Plugin not found" text, but specifically from `visual_build`, after exporting
the vision tower with `USE_TRT_NATIVE_ATTN=1`. The export itself succeeds and looks healthy: 27
`TRT_Attention` ops and a `kv_lengths` input.

**Cause.** `TRT_Attention` is a TensorRT-native op, not an Edge-LLM plugin. It requires
**TensorRT >= 11** (docstring at `tensorrt_edgellm/models/ops.py:384`). JetPack 7.2.1 ships
TensorRT **10.16.2.10**. `libNvInfer_edgellm_plugin.so` provides `ViTAttentionPlugin` but **not**
`TRT_Attention`, so no value of `EDGELLM_PLUGIN_PATH` or `LD_PRELOAD` can ever satisfy it. The error
message is misleading: it reports a plugin lookup failure for what is really a version
precondition failure.

**Fix.** Unset `USE_TRT_NATIVE_ATTN` and export the vision tower with the default (plugin) attention
path. This was tried here and abandoned; there is no workaround on TensorRT 10.x.

```bash
unset USE_TRT_NATIVE_ATTN
```

**Verify.** The re-exported ONNX contains no `TRT_Attention` nodes, and `visual_build` completes.
The FP16 vision engine built this way is **938 MB** on disk (ONNX external data 984,940,544 bytes).

**How to tell 6a from 6b in one line:** if you set `USE_TRT_NATIVE_ATTN=1` and the failure is in
`visual_build`, it is 6b and the plugin path is irrelevant. If you did not, and any engine fails,
start with 6a.

### 7. OOM while building the engine

**Preventive entry — this was not hit here.** The INT4 build completed, peaking at **3,884 MiB** of
CPU memory. The OOMs actually observed in this deployment were at engine *load*
([8](#8-oom-at-engine-load)) and on models that do not fit at all (Policy-DROID). This entry is the
precaution that kept the build inside its headroom, written up because the margin is thin.

**Symptom.** `llm_build` is killed mid-build, the OOM killer fires, or the board becomes
unresponsive and the build never finishes.

**Cause.** The Orin Nano has 8 GB of **unified** memory shared by CPU and GPU. Engine build is
memory-hungry on top of whatever is already resident. The 3,884 MiB peak above fits only if the
serving stack is not also resident (the FP16 shim's RSS alone was 6.02 GB, the INT4 shim's 3.70 GB).

**Fix.**

1. Stop everything resident before building:

   ```bash
   sudo systemctl stop live-vlm-webui.service cosmos3-edge-shim.service
   ```

2. Build with weights externalized, which the Jetson AI Lab TensorRT-Edge-LLM tutorial recommends
   specifically to cut peak build memory: `--externalize-weights int4_ffn`. Recommended by that
   tutorial, **not exercised here** — the build fit without it.
3. Keep swap available (2 GB swapfile on this image). Swap makes a marginal build complete slowly
   rather than fail; it is not a substitute for freeing RAM.
4. Build **on the target device**. TensorRT compiles per-GPU, so a host-built engine is not
   portable to the Jetson — cross-building is not a way around the memory limit.

**Verify.** The build completes and prints a weights-memory line. For the INT4 text tower here:

```
Total Weights Memory: 865,480,704
```

versus `3,355,696,384` for FP16 — a 3.88x reduction. Watch peak memory live from a second shell
with `tegrastats`.

### 8. OOM at engine load

**Symptom.** The engine builds fine but OOMs when the runtime deserializes it; the shim never
reaches `[shim] ready` and systemd restarts it in a loop.

**Cause.** Engine build parameters that are fine on a big board do not fit in 8 GB. Here, an engine
built at `batch=4` with `KV=4096` OOMed at load.

**Fix.** Rebuild with the batch and KV-cache capacity the board can actually hold. The values used
in this deployment:

```bash
llm_build --onnxDir <onnx_dir>/llm --engineDir <engine_dir> \
          --maxBatchSize 1 --maxKVCacheCapacity 2048
```

The flag names are attested (`llm_build`'s `--onnxDir`/`--engineDir` from the upstream Cosmos3
example, which passes the exporter's `llm/` subdirectory); the two *values* are recorded as the ones
used, and the Jetson went offline before a final re-read of `config.json`, so treat those as the
build inputs rather than as freshly verified output. The Jetson AI Lab Orin Nano tutorial
is more conservative still — `maxInputLen 512`, `maxKVCacheCapacity 1024` — which is where to go next
if 2048 does not fit for your prompt lengths.

**Verify.**

```bash
ssh "$JETSON_HOST" 'journalctl -u cosmos3-edge-shim.service -n 50 --no-pager'
```

Look for `[shim] LLMRuntime constructed in <n>s`, `[shim] CUDA graphs captured in <n>s`, and
`[shim] ready`. Reference timings here: **6.28 s** construction for INT4, 7.79 s for FP16.

---

## Serving

### 9. `cosmos3-edge-shim.service` restart loop

**Symptom.** `systemctl status cosmos3-edge-shim.service` shows repeated
`Failed with result 'exit-code'`, journal shows a Python
`No such file or directory` or `can't open file` error.

**Cause.** (Derived from reading the shipped unit file, not from a recorded failure during this
bring-up.) The unit's `ExecStart` points at an absolute path that does not exist on your device.
The unit as shipped runs
`/home/orin/TensorRT-Edge-LLM/.venv/bin/python /opt/tensorrt-edgellm/cosmos3_shim_v1.py`, i.e. it
expects the shim to have been copied to `/opt/tensorrt-edgellm/cosmos3_shim_v1.py` and a venv at
`/home/orin/TensorRT-Edge-LLM/.venv`. The file in this repo is
[`serve/cosmos3_shim.py`](../serve/cosmos3_shim.py).

**Fix.** Either copy the shim to the path the unit expects, or edit the unit to match your layout
(and `systemctl daemon-reload`). The same applies to
[`live-vlm-webui.service`](../systemd/live-vlm-webui.service), which expects `live-vlm-webui` inside
that venv and the shim reachable at `http://localhost:8000/v1`.

**Verify.**

```bash
ssh "$JETSON_HOST" 'systemctl is-active cosmos3-edge-shim.service && \
  curl -s http://127.0.0.1:8000/v1/models'
```

The models endpoint should return a list containing `nvidia/Cosmos3-Edge`.

### 10. `ModuleNotFoundError: _edgellm_runtime`

**Symptom.**

```
ModuleNotFoundError: No module named '_edgellm_runtime'
```

on shim startup.

**Cause.** (Derived from reading the shipped shim, not from a recorded failure during this bring-up.)
The pybind11 bindings are imported from a hardcoded build directory. The shim does
`sys.path.insert(0, "/home/orin/TensorRT-Edge-LLM/build/pybind")` before importing; if your
TensorRT-Edge-LLM build lives elsewhere, or the pybind module was not built, the import fails.

**Fix.** Correct that path in [`serve/cosmos3_shim.py`](../serve/cosmos3_shim.py) (and
`EDGELLM_PLUGIN_PATH` on the line above it) to your build tree, or build the pybind target.

**Verify.**

```bash
# on the Jetson
/home/orin/TensorRT-Edge-LLM/.venv/bin/python - <<'EOF'
import sys
sys.path.insert(0, "/home/orin/TensorRT-Edge-LLM/build/pybind")
import _edgellm_runtime as rt
print(rt.__file__)
EOF
```

### 11. numpy ABI mismatch in a `--system-site-packages` venv

**Symptom.** A numpy ABI error at import time — typically a compiled module built against a
different numpy major version — from `pandas` or `matplotlib`. The user-visible form is worse:
**Gradio silently returns nothing**, with the traceback swallowed rather than surfaced in the UI.

**Cause.** A venv created with `--system-site-packages` mixes the system numpy with venv-local
`pandas`/`matplotlib` (or vice versa) that were compiled against a different numpy ABI.

**Fix.** Install matching `pandas` and `matplotlib` **inside** the venv so all three come from one
consistent set.

```bash
/home/orin/TensorRT-Edge-LLM/.venv/bin/pip install --upgrade pandas matplotlib
```

**Verify.** Import all three in the venv interpreter and check that numpy resolves to a single
location:

```bash
/home/orin/TensorRT-Edge-LLM/.venv/bin/python -c \
  "import numpy, pandas, matplotlib; print(numpy.__version__, numpy.__file__)"
```

Then reload the Gradio page and confirm output appears.

---

## Measurement and benchmarking

### 12. `/usr/bin/time` does not exist (rc=127)

**Symptom.** A benchmark or wrapper script exits with `rc=127` and
`/usr/bin/time: No such file or directory`.

**Cause.** The Jetson image does not ship the GNU `time` binary. Bash's `time` is a shell keyword,
so `\time`/`/usr/bin/time` invocations from scripts fail while an interactive `time cmd` appears to
work — which makes this look intermittent.

**Fix.** Either install it (`sudo apt install time`) or, preferably, time in-process. The scripts in
this repo do the latter: they measure with `time.time()` around the HTTP call
([`scripts/profile_fixed.py`](../scripts/profile_fixed.py)) or parse the shim's own `[perf]` journal
lines ([`scripts/collect_perf.py`](../scripts/collect_perf.py)).

**Verify.** `command -v /usr/bin/time` returns a path, or the script no longer references it and
exits 0.

### 13. Benchmark results that look impossibly fast

**Symptom.** An image benchmark reports latencies far below the known fixed cost — e.g. well under
the ~287 ms image-path floor measured here — and the numbers barely move when you change image
resolution.

**Cause.** **The encoder embedding cache.** TensorRT-Edge-LLM caches vision-encoder embeddings keyed
on **raw pixel bytes** (`cpp/.../llmRankRuntime.cpp:1967`). A loop that posts the *same* image every
iteration skips the entire ViT after the first call — roughly **248 ms** of real work on this
platform — and reports a fantasy number.

**Fix.** Generate a **unique image per iteration**. Both probes in this repo do this deliberately:
[`scripts/profile_fixed.py`](../scripts/profile_fixed.py) and
[`scripts/tok_vs_res.py`](../scripts/tok_vs_res.py) seed a fresh random image per call. A unique
filename is not enough — the key is the pixel bytes, so re-encoding the same picture does not help.

**Verify.** The measured latency should scale with image token count. The reference sweep from this
deployment (unique image per probe, `max_tokens=1`, best-of):

| Probe resolution | Image tokens | Best latency |
|---|---|---|
| 320x240 | 80 | 90 ms |
| 448x336 | 140 | 152 ms |
| 640x480 | 300 | 288 ms |
| 896x672 | 512 (capped) | 499 ms |
| 1280x960 | 512 (capped) | 512 ms |

[`scripts/tok_vs_res.py`](../scripts/tok_vs_res.py) sweeps exactly these five resolutions and
subtracts a text-only baseline request from each `prompt_tokens` reading, so its `img_tok` column is
directly comparable with the table above.

If your curve is flat across those resolutions, the cache is still in play.

### 14. A "regression" that vanishes on re-measurement

**Symptom.** A change appears to make things dramatically worse — here, a context-cache
"regression" of **4.98 s/req** was initially reported — and then cannot be reproduced.

**Cause.** **Contention with live traffic.** The runtime serializes requests. A probe fired while
the Live VLM WebUI is streaming waits behind other work, and the probe's mean latency measures the
queue, not the model. The 4.98 s figure was **retracted**: it was lock contention, not the cache.

**Fix.** Either quiesce the workload before probing, or report **minimums**, which are the honest
statistic under contention. The fixed-cost decomposition in this repo (39 ms text-only, 287 ms
image path) is a minimum-based measurement taken under contention, and
[`scripts/profile_fixed.py`](../scripts/profile_fixed.py) takes medians over repeats for the same
reason.

**Verify.** Re-run the probe with the WebUI stopped:

```bash
ssh "$JETSON_HOST" 'sudo systemctl stop live-vlm-webui.service'
# re-run probe, then restart the WebUI when done
```

If the delta disappears, it was contention.

### 15. Regression fit degenerates or R-squared collapses

**Symptom.** [`scripts/collect_perf.py`](../scripts/collect_perf.py) prints
`all gen_tok identical (<n>) - cannot separate fixed/marginal`, or reports a low R² that looks like
a broken measurement.

**Cause.** Two distinct situations:

1. The client emits constant-length replies, so there is no spread in `gen_tok` and the fit
   `elapsed_ms = fixed_ms + marginal_ms_per_token * gen_tok` has zero variance in x.
2. The time window **spans a configuration change**, so it mixes two fixed-cost regimes. A low R²
   there is expected, not a fault.

**Fix.** Fall back to matching on `gen_tok` — which is the stronger comparison anyway — using
[`scripts/compare_perf.py`](../scripts/compare_perf.py), and make sure each window sits entirely on
one side of the config change:

```bash
JETSON_HOST="$JETSON_HOST" python3 scripts/compare_perf.py "<A_start>" "<A_end>" "<B_start>"
```

**Verify.** A healthy single-regime fit looks like the final one from this deployment:
**marginal 13.13 ms/token, fixed 289 ms, R² = 0.999, n = 699**. A healthy matched-`gen_tok` A/B
shows a consistent delta per bucket — the Round 4 result was a constant **-158 ms across every
bucket**, which is the signature of a fixed-cost reduction rather than a decode-rate change.

### 16. `prompt_tok` pinned at 510/511

**Symptom.** The `[perf]` journal lines report `prompt_tok` of 510 or 511 no matter what the camera
sends, and per-request latency will not drop below roughly 500 ms on the image path.

**Cause.** Not an error — the image-token budget is capped at 512 and your frames exceed it, so
every frame is resized to the cap. `max_image_tokens_per_image` is read at runtime from the
engine's `visual/config.json` (`builder_config`) and consumed by `smartResize` in
`cpp/multimodal/common/imageUtils.cpp:145`, where `maxPixels = max_image_tokens_per_image * 32^2`
(each LLM image token is 32x32 px: patch 16 x merge 2).

**Fix (a tradeoff, not a free win).** Lower the budget in the engine's `visual/config.json`:

```jsonc
// <engine_dir>/visual/config.json
"builder_config": { "max_image_tokens_per_image": 320 }
```

No engine rebuild is needed and the change is reversible in about 4 s by restarting the shim.
Measured here: **320 for live camera work, 512 for document/screenshot reading.** At temperature 0
the output was byte-identical at 320 vs 512 on a natural camera scene across three prompts, but on a
dense text screenshot 320 drifted vaguer and partly hallucinated where 512 quoted on-screen text
verbatim. Choose per workload.

**Verify.** After restarting the shim, `prompt_tok` in the journal drops — it went from 510/511 to
**330** here:

```bash
ssh "$JETSON_HOST" 'journalctl -u cosmos3-edge-shim.service -n 20 --no-pager | grep "\[perf\]"'
```

### 17. `TypeError: 'NoneType' object is not subscriptable` in `tok_vs_res.py`

**Symptom.** [`scripts/tok_vs_res.py`](../scripts/tok_vs_res.py) raises
`TypeError: 'NoneType' object is not subscriptable` at
`r.json()["usage"]["prompt_tokens"]`.

**Cause.** The probe reads token counts from the OpenAI-style `usage` object and the server it is
pointed at returns `"usage": null`. (Diagnosed by reading the two files in this repo, not from a
recorded failure during the original bring-up.)

**Fix.** [`serve/cosmos3_shim.py`](../serve/cosmos3_shim.py) in this repo returns a real `usage`
object — `prompt_tokens`, `completion_tokens`, `total_tokens`, built from the same
`resp.prompt_token_counts[0]` and `len(resp.output_ids[0])` that feed the `[perf]` line — so the
sweep works against the shim as shipped. The earlier snapshot used to gather the figures in this
report returned `null` there and logged those counts only to the `[perf]` journal line. If you are
running that older copy, or a different OpenAI-compatible server that omits `usage`, either populate
it the same way or read `prompt_tok` from the shim journal instead of the HTTP response.

**Verify.** `curl` the endpoint and confirm `usage` is non-null before running the sweep:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"nvidia/Cosmos3-Edge","messages":[{"role":"user","content":"hi"}],"max_tokens":4}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["usage"])'
```

---

## Not bugs: things that cannot work on Orin (sm_87)

These fail by design on this hardware or in this toolchain. Two of them fail **silently**, which is
worse than an error, so check them explicitly rather than assuming they took effect.

| What you tried | What happens | Why |
|---|---|---|
| `USE_TRT_NATIVE_ATTN=1` fused ViT attention | Misleading `Plugin not found` at `visual_build` | Needs TensorRT >= 11; JetPack 7.2.1 ships 10.16.2.10. See [6b](#6b-trt_attention-requiring-tensorrt--11--not-fixable-by-setting-the-plugin-path) |
| FP8 weights or FP8 KV cache | Not applicable | Only `FP8_KV_CFG` is implemented, and Orin sm_87 has no FP8 hardware |
| NVFP4 | Not applicable | Blackwell / Thor only |
| `tensorrt-edgellm-export --quantization int4_awq` on a dense model | **Silent no-op** — engine comes out FP16-sized | Gated behind MoE handling (`export.py:1046,1074`, dropped again at `export.py:4129-4131`) |
| INT4 on a layer with `out_features % 64 != 0` or `in_features % 64 != 0` | **Silently skipped** | Alignment rule at `quantization/quantize.py:1017-1029`. All Cosmos3-Edge layers pass, so all 169 targeted linears were quantized here |
| modelopt AWQ calibration in this toolchain | Produces no usable statistics | `attention_plugin` is a `torch.library.custom_op` whose body returns `torch.zeros(...)` (`models/ops.py:87-193`) — a shape-only stub for ONNX export, so eager forward yields zeros |
| Narrowing the ViT optimization profile | No-op | `optHW = (minImageTokens + maxImageTokens)/2 * 4` (`cpp/builder/visualBuilder.cpp:319`); with defaults the opt point is already ~514 image tokens |
| Cosmos3-Edge-Policy-DROID on 8 GB | GEN engine OOMs even with nothing else resident | Does not fit; this is a capacity limit, not a configuration error |

The silent failures are why [sanity check 5](#5-confirm-the-engine-actually-loaded-int4-weights) below
checks for INT4 by *size on disk*, not by the flag you passed.

---

## Sanity checks

Run these before believing any measurement. They take under a minute.

### 1. Power mode and clocks

```bash
ssh "$JETSON_HOST" 'sudo nvpmodel -q; sudo jetson_clocks --show | head -30'
```

Expected on this deployment: power mode **MAXN_SUPER** (`nvpmodel -m 2`) with `jetson_clocks`
applied, giving GPU **1020 MHz** (from 306), CPU **1728 MHz**, EMC **3199 MHz** (from 2133). The
exact output formatting varies by L4T release and is not reproduced here — check the values, not the
layout. If clocks are at their idle floors, every latency number you take will be wrong in the same
direction.

To (re)apply:

```bash
ssh "$JETSON_HOST" 'sudo nvpmodel -m 2 && sudo jetson_clocks'
```

### 2. Memory headroom

```bash
ssh "$JETSON_HOST" 'free -m; swapon --show'
```

Reference points from this deployment, with the serving stack resident: system RAM **available**
was 472 MB under FP16 and **2,705 MB** under INT4. A 2 GB swapfile is configured. If available
memory is in the low hundreds of MB, expect OOM at the next engine build
([7](#7-oom-while-building-the-engine)) and inflated latencies from paging.

Live view during a build or a run:

```bash
ssh "$JETSON_HOST" 'tegrastats --interval 1000'
```

### 3. Service health

```bash
ssh "$JETSON_HOST" 'systemctl is-active cosmos3-edge-shim.service live-vlm-webui.service'
ssh "$JETSON_HOST" 'journalctl -u cosmos3-edge-shim.service -n 40 --no-pager'
```

A healthy startup sequence prints, in order:

```
[shim] LLMRuntime constructed in <n>s
[shim] CUDA graphs captured in <n>s
[shim] warm-up inference in <n>s
[shim] ready
```

`warm-up skipped: ...` is non-fatal by design (the warm-up is best-effort), but it means the first
real request pays first-call cost — do not include that request in a benchmark.

### 4. End-to-end reachability

```bash
ssh "$JETSON_HOST" 'curl -s http://127.0.0.1:8000/v1/models'
```

Should list `nvidia/Cosmos3-Edge`. The WebUI unit points at `http://localhost:8000/v1` on port 8090;
if the model list is empty or the request hangs, fix the shim before looking at the WebUI.

### 5. Confirm the engine actually loaded INT4 weights

Because `--quantization int4_awq` is a silent no-op on dense models, check the artifact, not the
invocation. Any one of these is sufficient; the first two are the strongest.

**a. Engine size on disk** — the decisive check:

```bash
ssh "$JETSON_HOST" 'ls -l /opt/tensorrt-edgellm/models/default'
```

| Build | LLM engine bytes |
|---|---|
| FP16 | 3,366,196,508 (3.135 GB) |
| INT4 W4A16 | **878,552,564 (0.818 GB)** |

If the engine is ~3.1 GB, it is FP16 regardless of what you passed to the exporter.

**b. Build log weights memory** — from the build you kept:

```
Total Weights Memory: 865,480,704     # INT4  (vs 3,355,696,384 FP16)
```

**c. Resident RSS of the shim process:**

```bash
ssh "$JETSON_HOST" 'systemctl show -p MainPID cosmos3-edge-shim.service; \
  ps -o rss=,cmd= -p $(systemctl show -p MainPID --value cosmos3-edge-shim.service)'
```

Reference: **3.70 GB** INT4 versus 6.02 GB FP16.

**d. Decode rate from live traffic** — the behavioural check:

```bash
JETSON_HOST="$JETSON_HOST" python3 scripts/collect_perf.py "10 min ago"
```

Reference: **13.13-13.16 ms/token (76.0 tok/s)** INT4 versus 44.43 ms/token (22.5 tok/s) FP16. A
marginal near 44 ms/token means you are running FP16 weights.

**e. Checkpoint-side check**, if you still have the quantized checkpoint:

```bash
ssh "$JETSON_HOST" 'cat <ckpt_dir>/hf_quant_config.json'
```

[`scripts/rtn_int4_quantize.py`](../scripts/rtn_int4_quantize.py) writes `quant_algo: "W4A16_AWQ"`,
`group_size: 128`, `has_zero_point: false`, and the `exclude_modules` list that keeps the vision
tower, projector, `embed_tokens` and norms in FP16. The corresponding INT4 ONNX contained
**169 `Int4GroupwiseGemmPluginV2` nodes** and a `model.onnx.data` of 0.807 GB.

### 6. Quality spot check after any quantization or token-budget change

```bash
# prerequisite: a fixed test frame at /home/orin/bench_frame.jpg
# (the path is hardcoded in scripts/quality.py -- edit it if yours lives elsewhere)
python3 scripts/quality.py int4     # runs on the Jetson against 127.0.0.1:8000
```

[`scripts/quality.py`](../scripts/quality.py) uses `temperature: 0.0` so two runs are directly
comparable — provided both runs use the *same* frame, which is why the path is fixed rather than
generated. Without that file the script dies with `FileNotFoundError` before it ever reaches the
shim. **Caveat, stated plainly:** this is a spot check, not an evaluation. The INT4 weights
here carry **11.06% mean relative weight error** (worst layer `to_k` at **20.5%**) from uncalibrated
RTN with MSE-optimal clipping, and **no benchmark suite was run**. Output was coherent and grounded
with no repetition collapse, but evaluate on a real task set before production use. Note also that
no FP16-vs-INT4 side-by-side on identical prompts was possible, because both engines cannot be
resident within 8 GB at once.

---

## Related documents

- [Full written report](report.md) — methodology, the four optimization rounds, and the
  INT4 derivation in detail.
- [`scripts/`](../scripts/) — the measurement and quantization tools referenced above.
- [`serve/cosmos3_shim.py`](../serve/cosmos3_shim.py), [`systemd/`](../systemd/) — the deployed
  serving stack.
