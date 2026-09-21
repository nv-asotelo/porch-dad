# Backend build and launch

Status (2026-09-20): **corrected FP16 is selected and serving actual image answers through enabled, active backend/UI systemd services**. The selected profile is input 1024 / KV 2048, batch one, FP16 weights/activation/KV, zero encoder embedding cache, no text-context reuse, and stock 25 W with restored dynamic clocks. Swap is disabled persistently; the existing `/swapfile` and original fstab are preserved. See [selected environment](../deployment/selected.env), [configuration/provenance](../deployment/selected-config.json), [selection decision](../results/optimization-selection.json), and [memory-mode receipt](../results/deployment-memory-mode/receipt.json).

The corrected six-image screen scores [18/19 required facts](../results/quality-fp16-corrected-01-review.json), up from the initial 7/19. The official unmodified model reproduces the sole remaining shape error, so the absolute screen still fails. The selected model meets the frozen [reference-equivalence policy](../results/optimization-quality-policy.json), and all six final-service outputs match corrected FP16 exactly. Browser incremental display, Stop and restart passed [against the actual Orin service](../results/browser-live-ui.json).

The final service benchmark, after five warmups and 30 measured requests, reports **265.540 ms p50 time to first text, 1,825.899 ms p50 complete-answer latency, and 6,909,468,672 bytes (6.435 GiB) sampled peak system RAM unavailable**. There were no request errors, OOM events, or workload swap activity. RAM unavailable is `MemTotal - MemAvailable`, a shared system-memory observation, not dedicated VRAM or a CUDA allocation counter. [Raw final service benchmark](../results/raw/benchmark-final-service-01.jsonl). The bounded search stopped after four evaluated candidates and three consecutive non-improvements. The [API streaming soak](../results/raw/soak-final-service-01.summary.json) passed **600.701 seconds and 366 requests**, with zero errors, truncation, OOM or swap-in/out, at least **1,011,798,016 bytes available RAM**, and only **159,744 bytes median RSS growth** within the frozen tolerance. System RAM unavailable median growth was 2,562,048 bytes, reported separately; observed active requests peaked at one and queued requests at zero. This validates sequential API streaming stability, not semantic quality or browser-camera behavior.

The [final service audit](../results/final-service-audit.json) records both services enabled and active with zero restarts and backend ready. A post-install physical reboot was not tested. The [final browser upload check](../results/browser-live-ui-final.json) passed incremental text, Stop/restart and no JavaScript errors; the [final synthetic moving-camera check](../results/browser-camera-ui-final.json) also passed fresh frames, one active request, Stop and restart.

The [selected-model natural-image review](../results/natural-smoke-final-01-review.json) found **5/9 observations with generous identity credit, 4/9 complete required facts, and 0/3 fully correct cases**. The [official unmodified PyTorch/Transformers reference](../results/reference-natural-smoke.json) also scored 5/9 generous and 4/9 strict, with the market and dog answers matching the Orin verbatim. The cap/mitt wording differs but both omit the required relation. This limited comparison supports a model/task limitation on these examples; it does not prove full backend equivalence, representative accuracy, or natural-video quality. Mac reference timings are not Orin performance measurements.

The [final all-boot kernel audit](../results/final-service-audit.json) records **nine NVMe `completion polled` timeout warnings**, the latest at uptime **10025.646445 seconds** during cold startup. This extends the earlier [cold-reload observation](../results/nvme-cold-reload-observation.json); the earlier 30-minute window without recurrence did not establish a fix. The backend reached ready and completed the measured final benchmark and soak. These warnings do not establish a new steady-inference performance problem or its cause. The root cause remains unresolved; no kernel or NVMe power-management fix is claimed.

The original native build completed both required targets and all 25 selected SM87 FMHA variants with exit code 0 in a recorded wrapper interval of 45m44s. The [native-build receipt](../results/native-backend-build.json) preserves raw-log references and sampled RAM/swap/temperature separately from engine construction and inference. The initial engine receipt is [model-cache-20260920T052832Z-36e5d1ee.json](../results/model-cache-20260920T052832Z-36e5d1ee.json). Build completion and a streamed response do not by themselves pass the visual quality gate.

The official Super QSPI/NVMe flash completed over USB-C on P3767-0005, followed by normal boot and a successful CUDA calculation. The Orin runs Ubuntu 24.04.4 / L4T 39.2.1 with CUDA compiler 13.2.86 and TensorRT 10.16.2.10; its root partition/ext4 filesystem has been expanded to use the NVMe capacity. Python dependency and tokenizer checks passed on the target CPU. The pinned backend and digest-verified reasoner snapshot are installed under `/home/jetson/cosmos-edge`. Commands below were checked against public TensorRT Edge-LLM `v0.10.1`. See [normal boot](../results/normal-boot.json), [CUDA preflight](../results/cuda-preflight.json), [storage expansion](../results/storage-expansion.json), and [Python environment](../results/python-environment.json).

The target is Cosmos3-Edge's image/text **reasoner** on an Orin Nano. Install the selected official JetPack 7.2.1 image first, then check that its CUDA/TensorRT packages match the upstream JetPack 7.2 / CUDA 13.2 / TensorRT 10 row. Flashing and storage selection are handled separately; this document never guesses a disk to overwrite. [Official backend support matrix](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/support-matrix.html)

## 1. Inventory the real target

Run from this project's directory on the Jetson:

```bash
export COSMOS_PROJECT_DIR="$PWD"
bash scripts/device_inventory.sh
```

Confirm the actual board model, memory capacity, OS, `nvidia-l4t-core`, CUDA 13.2, TensorRT 10, and free storage. Reserve at least 30 GB of working storage for source, checkpoint shards, build products, and temporary engines; an NVMe SSD is preferable. The script only queries state and does not change power modes, clocks, packages, or files. Capture an idle memory baseline before loading the model.

## 2. Build the pinned runtime on Orin

Use the existing public checkout under `external/TensorRT-Edge-LLM`, or clone it once if it was not copied with the project:

```bash
git clone --branch v0.10.1 --depth 1 \
  https://github.com/NVIDIA/TensorRT-Edge-LLM.git \
  "$COSMOS_PROJECT_DIR/external/TensorRT-Edge-LLM"
```

Do not run that clone command over an existing checkout. Verify the immutable pin, then initialize its public submodules:

```bash
cd "$COSMOS_PROJECT_DIR/external/TensorRT-Edge-LLM"
test "$(git rev-parse HEAD)" = e8b29522938901f6df19ebeedd4b69bc8edbcd97
git submodule update --init --recursive
sudo apt update
sudo apt install -y build-essential binutils cmake git python3-venv python3-dev
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'pybind11==3.0.4' \
  'nvidia-cutlass-dsl[cu13]==4.7.0' 'cupy-cuda13x==13.6.0' \
  'cuda-python>=12.6,<14'
export PATH="/usr/local/cuda-13.2/bin:$PATH"
export TRT_PACKAGE_DIR=/usr
export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu:/usr/local/cuda-13.2/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
nvcc --version
python -c 'import tensorrt; print(tensorrt.__version__)'
```

For this deployment, `bash scripts/build_backend.sh fp16` runs a real CUDA array check, the kernel generation/configuration below, and the required runtime/plugin build targets with one compiler worker. It saves a separate receipt directory with logs, exit status and a `tegrastats` trace when available. This wrapper completed successfully on the normally booted Orin, including the native runtime import. Its optional `int4` mode also generates the INT4 V2 group. Model-engine and image-inference validation remain separate steps; avoid overlapping engine construction, native compilation and resident serving on this memory-constrained board.

The source checkout does not contain the generated CuTe DSL artifacts. Generate the SM87 artifacts before CMake; a single compile worker reduces build memory pressure:

```bash
python kernelSrcs/build_cutedsl.py \
  --gpu_arch sm_87 --arch aarch64 --cuda-version 13.2 \
  --kernels fmha --jobs 1
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DTRT_PACKAGE_DIR=/usr \
  -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
  -DEMBEDDED_TARGET=jetson-orin \
  -DCUDA_CTK_VERSION=13.2 \
  -DENABLE_CUTE_DSL=fmha \
  -DBUILD_PYTHON_BINDINGS=ON \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build build --parallel 1
python -m pip install -e '.[server,server-tools]'
python -m pip install --only-binary=:all: \
  'numpy==2.2.6' 'scipy==1.15.3' 'cffi==2.0.0'
python -m pip check
python -m experimental.server --help
```

These commands combine the documented Orin source build with the pinned kernel builder's dependency checks. A missing compatible wheel or failed kernel build is a real installation blocker; retain its log and resolve it before moving on. The `server,server-tools` extras avoid installing the full PyTorch exporter into the serving environment. The native runtime still depends on JetPack's TensorRT libraries. [Installation guide](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/installation.html), [pinned build dependency checks](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/kernelSrcs/build_cutedsl.py), [package extras](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/pyproject.toml)

The `--system-site-packages` environment exposes JetPack's apt-installed TensorRT bindings. On the installed Ubuntu image it also exposes SciPy 1.11.4, which rejects the server's required NumPy 2.2.6, and PyNaCl 1.5.0 with an unmet CFFI dependency. The wheel-only command above installs compatible SciPy and CFFI **inside the venv**, shadowing the older SciPy without changing system Python. SciPy 1.15.3 accepts NumPy 1.23.5 through versions below 2.5; both selected packages publish CPython 3.12 ARM64 manylinux wheels. Their metadata and exact wheel hashes are recorded in [the dependency audit](../research/python-system-site-compatibility.md). Require a clean `pip check` and CPU import checks before proceeding; package installation alone does not demonstrate GPU execution.

Transformers 5.14.1 and Jinja2 3.1.6 from `server-tools` are needed to derive this checkpoint's processed chat template. The earlier [tokenizer preflight](../results/cosmos3-tokenizer-preflight.json) verified text formatting and rendered model image delimiters, but did not catch missing `content_types` mappings in the generated artifact. Actual image requests exposed that omission. The [media repair](../scripts/repair_cosmos_chat_template.py) now restores the exact image/video delimiters from the checkpoint template; the [RoPE repair](../scripts/repair_cosmos_runtime_config.py) supplies existing checkpoint values under the aliases consumed by the C++ vision runner. Both helpers retain before/after receipts and are run by `build_model_cache.py`. The current [artifact preflight](../scripts/preflight_cosmos_artifacts.py) rejects missing repairs, generic fallback templates and unsupported cache state. These checks support launch correctness; actual image inference remains the acceptance test.

Apply the task's exact source patches to the pinned public checkout before reproducing the corrected candidate: [HWC→CHW patch projection](../patches/cosmos3-patch-embedding-chw.patch), [Cosmos half-pixel position alignment](../patches/cosmos3-half-pixel-position.patch), and [encoder embedding-cache budget API](../patches/tensorrt-edge-llm-v0.10.1-encoder-cache-budget.patch). Use `git apply --check` on a clean checkout first and do not apply a patch twice. The projection repair requires a fresh visual engine; the position and cache-control patches require a native runtime rebuild. Preserve the original cache as diagnostic evidence and use a new `COSMOS_CACHE_DIR` for the corrected engine. Exact design, tests and upstream credit are recorded in the [contribution ledger](../research/contribution-ledger.md).

The exact dense reasoner uses the supported `fmha` group for FP16 decoder and vision attention; its dense matmuls and normalization do not require the other CuTe groups. This selects 25 SM87 variants instead of all 81. Actual generation still requires a working GPU; it cannot run in the driverless recovery chroot. See [the source audit](../research/cosmos3-kernel-selection.md).

Before an INT4 V2 candidate, extend the same artifact directory without `--clean`:

```bash
python kernelSrcs/build_cutedsl.py --gpu_arch sm_87 --arch aarch64 \
  --cuda-version 13.2 --kernels int4_fp16_gemm --jobs 1
cmake -S . -B build '-DENABLE_CUTE_DSL=fmha;int4_fp16_gemm'
cmake --build build --parallel 1
```

The generator merges compatible prior FMHA metadata/artifacts. The INT4 group is mandatory for the default INT4 V2 plugin; merely changing checkpoints on an FMHA-only build is insufficient.

## 3. Use the verified reasoner-only snapshot

The local project already contains `models/cosmos3-edge-reasoner` at public NVIDIA revision `344d602b128d1bbdacb43b08d0a3626f46343e29`: **21 files totaling 7,735,525,563 bytes**, including 7,718,115,464 bytes of physical weight shards. Every file passed its public Hugging Face Git blob SHA-1 or LFS SHA-256 check; details are in [model-download.json](../results/model-download.json). Generator component configurations remain absent. The indexed reasoner tensors occupy about 4.87 GB, but the referenced shard files contain additional tensors; downloaded bytes are not a memory measurement.

This snapshot has already been transferred to `/home/jetson/cosmos-edge/models/cosmos3-edge-reasoner` and verified on this Orin; see [transfer evidence](../results/deployment-transfer.json). For a fresh deployment, copy it while preserving the directory structure and original notices. Set:

```bash
export COSMOS_REASONER_DIR="$COSMOS_PROJECT_DIR/models/cosmos3-edge-reasoner"
```

No second download is needed for this prepared workspace. For reproduction on a fresh machine **without the snapshot**, the following alternative downloads the same indexed weight shards and root metadata into a new empty directory. The recorded prepared download includes all original root files, including `.gitattributes`; this alternative selects the root metadata extensions listed below. Preserve the public revision and verify downloaded files before building.

```bash
cd "$COSMOS_PROJECT_DIR/external/TensorRT-Edge-LLM"
source .venv/bin/activate
export COSMOS_REASONER_DIR="$COSMOS_PROJECT_DIR/models/cosmos3-edge-reasoner"
python - <<'PY'
import json
import os
from pathlib import Path
from huggingface_hub import hf_hub_download, model_info, snapshot_download

repo = 'nvidia/Cosmos3-Edge'
revision = '344d602b128d1bbdacb43b08d0a3626f46343e29'
target = Path(os.environ['COSMOS_REASONER_DIR'])
if target.exists() and any(target.iterdir()):
    raise SystemExit('Choose a new empty reasoner directory; existing data is preserved.')
index_path = hf_hub_download(repo, 'model.safetensors.index.json', revision=revision)
index = json.loads(Path(index_path).read_text())
root_files = [
    entry.rfilename for entry in model_info(repo, revision=revision).siblings
    if '/' not in entry.rfilename
    and entry.rfilename.endswith(('.json', '.jinja', '.md', '.txt'))
]
allowed = sorted(set(root_files) | set(index['weight_map'].values()))
snapshot_download(repo, revision=revision, local_dir=target, allow_patterns=allowed)
for path in ('transformer/config.json', 'vae/config.json'):
    assert not (target / path).exists(), path
print(target.resolve())
PY
```

This preserves the provider's root config, processor, tokenizer, chat template, index, and referenced weights. It deliberately never downloads generator component configs or VAE weights into this new folder. The upstream component resolver then has only LLM and VISUAL to build. Keep the original model's license/origin notices and the [OpenMDW-1.1 agreement](https://openmdw.ai/license/1-1/) with any redistributed model artifacts. [Pinned component-selection source](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/experimental/builder/models/cosmos3/configuration.py)

## 4. Reproduce the selected build and services

The prepared Orin already has the corrected native runtime and selected engine. These build commands are for reproduction after applying the three source patches described above; stop the task's resident services before any native/engine rebuild. A changed visual projection requires a fresh visual engine, so use a new cache directory for a changed implementation and update both selected deployment files only after validation. The helper preserves existing profiles and refuses incomplete/stale matching bundles instead of deleting them.

From a normal boot on the Orin, with the pinned backend Python environment prepared:

```bash
cd /home/jetson/cosmos-edge
sudo systemctl stop cosmos-edge-ui.service cosmos-edge-backend.service
export COSMOS_PROJECT_DIR="$PWD"
export COSMOS_REASONER_DIR="$PWD/models/cosmos3-edge-reasoner"
export COSMOS_CACHE_DIR="$PWD/data/engine-cache-chw"
export COSMOS_MAX_INPUT_LEN=1024
export COSMOS_MAX_KV_CAPACITY=2048
export COSMOS_ENCODER_CACHE_BYTES=0
unset COSMOS_MAX_IMAGE_TOKENS COSMOS_MAX_IMAGE_TOKENS_PER_IMAGE
bash scripts/build_backend.sh fp16
python3 scripts/measure_command.py --output "results/engine-reproduction-$(date -u +%Y%m%dT%H%M%SZ)" -- \
  external/TensorRT-Edge-LLM/.venv/bin/python scripts/build_model_cache.py \
  --model "$COSMOS_REASONER_DIR" --cache-dir "$COSMOS_CACHE_DIR"
bash scripts/run_backend.sh --preflight-only "$COSMOS_REASONER_DIR"
```

For a fresh deployment without installed units, omit the initial service-stop command. Install Ubuntu's `time` package if `/usr/bin/time` is absent. Native compilation, engine construction and resident serving must run sequentially on this board. The measurement wrapper saves output, sampled system RAM/swap, `tegrastats`, GNU time and terminal status. A cache hit records the current environment without claiming a new engine build; compile/build peaks are distinct from inference peaks.

The build helper runs the checkpoint-specific media and RoPE repairs. Direct server startup requires a complete normalized bundle matching the model, input/KV and visual profile; **startup does not build missing engines**. Preserve the original checkpoint and its notices. Build success still requires actual image-quality and streaming validation.

The checked-in `deployment/selected.env` contains only task paths and explicit runtime settings, using systemd `NAME=value` syntax. It selects `COSMOS_PROFILE=fp16`, the original reasoner checkpoint, `data/engine-cache-chw`, input 1024, KV 2048, encoder cache zero and `COSMOS_STATIC_CLOCKS=0`. The companion `deployment/selected-config.json` records source/model pins, patches and observed deployment state; copied provenance does not establish that a newly built engine has been validated.

Install and start the selected services after the cache is ready and any task-owned foreground servers have stopped:

```bash
cd /home/jetson/cosmos-edge
sudo bash scripts/install_services.sh
sudo systemctl start cosmos-edge-backend.service cosmos-edge-ui.service
systemctl is-enabled cosmos-edge-backend.service cosmos-edge-ui.service
systemctl is-active cosmos-edge-backend.service cosmos-edge-ui.service
journalctl -u cosmos-edge-backend.service -u cosmos-edge-ui.service -n 50 --no-pager
curl --fail http://127.0.0.1:8090/health/ready
```

The installer verifies units and enables boot startup but never starts them itself. Readiness appears after model loading. The backend binds `127.0.0.1:8000`; the UI binds `127.0.0.1:8090` and proxies streaming image requests to it. The backend service runs as `jetson`; the installer does not source its environment file as root. On the Mac, an SSH forward to the Orin's loopback UI makes browser camera access available at [http://localhost:8090](http://localhost:8090):

```bash
ssh -N -L 8090:127.0.0.1:8090 jetson@YOUR_JETSON_ADDRESS
```

The selected service uses one active request, at most one queued request, no prefix reuse or speculation, and zero encoder embedding cache. The pinned upstream server's default visual profile is 1024 total image tokens and 512 per image. The task wrapper supports explicit `COSMOS_MAX_IMAGE_TOKENS` and `COSMOS_MAX_IMAGE_TOKENS_PER_IMAGE` through the pinned Python `BuildOptions` API; if used, identical values must be present during build and serve. The compact-profile candidate was not run in the original FP16 phase because its stop rule had fired. The later user-selected MLP phase did build and select a compact profile; see section 6 below. [Pinned BuildOptions](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/experimental/server/runtime/engine_build.py)

The Cosmos positional repair aligns the learned grid for the fixed 512×512 quality fixtures. When a processed image axis is below 256 pixels, its four-tap approximation differs from the official model's wider antialiased downsampling filter. The UI preserves image content/aspect and centers it on black padding when either resized axis would be below 256; it never crops, distorts or upscales small images. Direct API clients must account for this [documented interpolation boundary](../patches/cosmos3-half-pixel-position.md). UI padding is not a claim of general backend parity.

## 5. Quantization findings and the unselected AWQ route

There is no upstream RTN switch in this pin. This task implemented an isolated [CPU INT4 round-to-nearest converter](../scripts/quantize_cosmos3_rtn.py) and [backend integration](../scripts/rtn_backend.py), built both candidates and measured their actual outputs on the Orin. The broad decoder candidate scored [13/19 required facts](../results/quality-rtn-v1-01-review.json). The MLP-only candidate scored [18/19](../results/quality-rtn-mlp-01-review.json), but introduced a new color error while fixing the permitted reference shape error. Both failed the original frozen FP16-equivalence rule. The user later selected MLP-only RTN as the new starting point, accepting its documented color error. Section 6 describes the current MLP deployment. Packed checkpoint size alone does not establish runtime memory or dedicated VRAM usage.

The fourth trial used static maximum clocks within unchanged stock 25 W. It preserved the FP16 outputs, but missed the required 5% latency improvement and recorded workload swap activity. The search stopped at the declared three consecutive non-improvements; stock dynamic clocks were restored and independently verified. [Selection receipt](../results/optimization-selection.json), [clock restoration check](../results/deployment-memory-mode/clock-restore-verification.json).

ModelOpt AWQ remains an unexecuted alternative requiring calibration and new quality validation. Direct FP16 building externalizes weights but does not quantize them. The following public upstream commands are retained for reference, not as a selected or verified Cosmos3 AWQ deployment.

**Calibration dependency gate:** the exact public Transformers 5.14.1 wheel lacks Cosmos3-Edge model/processor registration, so the unmodified `.[tools]` recipe below cannot currently calibrate this checkpoint. Its command is a record of the upstream flag surface, not a validated Cosmos3 run. The official implementation exists at an immutable newer Transformers commit; isolate that dependency override from the Orin serving environment and follow the [source audit](../research/transformers-registration-audit.md) and [guarded AWQ candidate](../research/cosmos3-awq-candidate.md) before execution.

Use a separate Linux NVIDIA GPU for calibration if the Nano cannot accommodate source weights plus activations. A 16–24 GB GPU is a practical preference, not a measured minimum. Do not install CUDA quantization packages on the Mac and expect GPU execution. In a separate calibration environment with the same pinned checkout:

```bash
python3 -m venv .venv-quantize
source .venv-quantize/bin/activate
python -m pip install -e '.[tools]'
tensorrt-edgellm-quantize llm \
  --model_dir /absolute/path/to/Cosmos3-Edge-reasoner \
  --output_dir /absolute/path/to/Cosmos3-Edge-reasoner-int4 \
  --quantization int4_awq \
  --lm_head_quantization int4_awq \
  --dtype fp16 \
  --num_samples 128
```

This is the **candidate command**, not a verified Cosmos AWQ artifact. The source pins Torch 2.13.0, Transformers 5.14.1, and ModelOpt 0.45.0. Generic AWQ defaults to calibration batches of 16. Review the exported weight names, visual projector precision, model metadata, and final model quality; a model-specific failure must be fixed and tested, not concealed. Keep FP16 visual weights and FP16 KV cache for the initial Orin path; FP8 KV/embedding require SM89+ and are unavailable on SM87. [Quantizer code](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/tensorrt_edgellm/quantization/quantize.py), [precision support](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/supported-models.html)

An AWQ checkpoint would need a separate compatible cache-preparation route and actual device validation: the selected `build_model_cache.py` helper explicitly rejects quantized metadata, and `run_backend.sh` refuses missing engines. The RTN helper is specific to this task's RTN artifacts and is not an AWQ adapter. If the experimental direct path fails for this model/precision, the documented fallback is CPU ONNX export with `--task reasoning`, followed by device `llm_build`, `visual_build`, and `llm_inference`. That fallback does not automatically provide the existing experimental HTTP server, so a resident C++ streaming adapter would then be additional work. Do not replace the resident backend with one process launch per camera frame. [Supported Cosmos reasoner commands](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/examples/vla/cosmos3.html)

## 6. Measurement and acceptance

Use the project's benchmark harness and the exact stopping rule in `GOAL.md`. Compare fixed JPEGs, prompts, output caps, and power settings after warmup. Collect median/p95 time to first token, request completion latency, genuine completion-token throughput, device allocated-memory peak where native instrumentation permits, total DRAM use, and an image-grounding/OCR/counting quality check. The client throughput includes the whole request; isolated decode speed requires native timing. Label cold compilation separately. FP16 and INT4 weight arithmetic is available in `research/backend-feasibility.md`; none of it is measured VRAM or an observed speedup.

## 7. Preserve or restore the selected memory and clock state

The final resident deployment uses physical RAM only: `/proc/swaps` contains just its header, the `/swapfile` fstab entry is commented out, and the existing 2 GiB file is retained. The original fstab is saved at `/home/jetson/cosmos-edge/results/deployment-memory-mode/fstab.before`; its before/after hashes and RAM observations are in the [receipt](../results/deployment-memory-mode/receipt.json). This final operational change was followed by the service benchmark and completed 600.701-second API soak above; it is not an extra search candidate. `install_services.sh` does not alter swap or the power-mode selection.

To inspect this state on the Orin:

```bash
cat /proc/swaps
free -h
sudo /usr/sbin/nvpmodel -q
sudo /usr/bin/jetson_clocks --show
```

To restore the preserved swap for future build work, stop the task's services, review the backup and edit only the task's commented swap entry. Preserve unrelated subsequent fstab changes:

```bash
sudo systemctl stop cosmos-edge-ui.service cosmos-edge-backend.service
sudo diff -u /home/jetson/cosmos-edge/results/deployment-memory-mode/fstab.before /etc/fstab
sudoedit /etc/fstab
# Restore this exact original entry: /swapfile none swap sw 0 0
sudo systemctl daemon-reload
sudo /sbin/swapon /swapfile
cat /proc/swaps
```

The diff is expected to show the commented entry. Restoring swap changes the measured deployment configuration; record that change before comparing performance. To return to no-swap operation, first stop the task's resident services and verify enough available RAM to page in any occupied swap, then run `sudo /sbin/swapoff /swapfile`, comment that same fstab entry, and reload systemd. Preserve `/swapfile` and the saved original fstab.

Dynamic clocks are already restored and `COSMOS_STATIC_CLOCKS=0` is selected. Only exact `COSMOS_STATIC_CLOCKS=1` makes the installer add the root oneshot `cosmos-edge-clocks.service`, ordered after `nvpmodel.service` and required before the backend; no fan or power-mode changes are requested. If a future clock trial enables it, restoring dynamic operation requires setting the selection back to zero and reinstalling units to remove the backend dependency, then disabling the clock unit and restoring the saved settings:

```bash
cd /home/jetson/cosmos-edge
# Set COSMOS_STATIC_CLOCKS=0 in deployment/selected.env first.
sudo bash scripts/install_services.sh
sudo systemctl disable --now cosmos-edge-clocks.service
sudo /usr/bin/jetson_clocks --restore /home/jetson/cosmos-edge/data/power/stock25w.before.conf
```

The disable command applies only if a clock unit was previously installed; none is needed for the selected dynamic configuration. Merely disabling a clock unit while leaving the backend's `Requires` dependency in place would let a later backend start activate it again.


## 6. Selected MLP-only RTN deployment

The current [selection](../deployment/selected-config.json) uses the same pinned source model and a task-generated MLP-only INT4 derivative. Attention, LM head, vision, projector, embeddings, activations and KV remain FP16. The known yellow-to-orange error remains. The first new candidate saved 163.45 MiB of sampled peak shared RAM and showed 1.19% lower aggregate p50. The search stopped below the user's 10% continuation threshold and retained the smaller configuration. Read the [measurement and policy record](../research/mlp-goal.md) before interpreting those figures.

On the configured Orin the model, engine cache and native binaries already exist. Start the enabled services as above. The selected launcher uses `COSMOS_PROFILE=rtn-v1`, model `models/cosmos3-edge-rtn-mlp-int4`, cache `data/engine-cache-mlp-compact`, input capacity 1024, KV capacity 1664, total image capacity 512 and per-image capacity 512. These preserve the UI's single-image input and 512-output-token envelope. The first build used the original V1 plugin, followed by the task's native kernel rebuild. Runtime receipts identify the final rebuilt plugin and extension.

For a fresh reconstruction after the pinned dependencies and earlier compatibility patches in this document, stop the task's backend before allocating build/model memory. Preserve existing model/cache directories; the converter and engine wrapper validate their receipts. The converter's `--apply` writes a new derivative, without calibration:

```bash
cd /home/jetson/cosmos-edge
sudo systemctl stop cosmos-edge-backend
external/TensorRT-Edge-LLM/.venv/bin/python scripts/quantize_cosmos3_rtn.py \
  --source /home/jetson/cosmos-edge/models/cosmos3-edge-reasoner \
  --output /home/jetson/cosmos-edge/models/cosmos3-edge-rtn-mlp-int4 \
  --quantization-scope mlp-only --apply
```

Apply the new launch patch once to the pinned upstream checkout, preserving its original license notices, then rebuild both targets. An already patched checkout fails `apply --check`; do not apply it twice.

```bash
export PATH="/home/jetson/cosmos-edge/external/TensorRT-Edge-LLM/.venv/bin:/usr/local/cuda-13.2/bin:$PATH"
export TRT_PACKAGE_DIR=/usr CUDA_PATH=/usr/local/cuda-13.2
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/usr/local/cuda-13.2/lib64
export TMPDIR=/home/jetson/cosmos-edge/data/tmp
export XDG_CACHE_HOME=/home/jetson/cosmos-edge/data/cache
export CUDA_CACHE_PATH=/home/jetson/cosmos-edge/data/cache/cuda
git -C external/TensorRT-Edge-LLM apply --check /home/jetson/cosmos-edge/patches/int4-gemv-cosmos-mlp-n4.patch
git -C external/TensorRT-Edge-LLM apply /home/jetson/cosmos-edge/patches/int4-gemv-cosmos-mlp-n4.patch
cmake --build external/TensorRT-Edge-LLM/build --parallel 1 \
  --target NvInfer_edgellm_plugin _edgellm_runtime
external/TensorRT-Edge-LLM/.venv/bin/python scripts/rtn_backend.py build \
  --model /home/jetson/cosmos-edge/models/cosmos3-edge-rtn-mlp-int4 \
  --cache-dir /home/jetson/cosmos-edge/data/engine-cache-mlp-compact \
  --max-input-len 1024 --max-kv-capacity 1664 \
  --max-image-tokens 512 --max-image-tokens-per-image 512
sudo systemctl start cosmos-edge-backend
```

Build/serve options must agree. A reconstruction is a new build with new binary hashes and may choose different TensorRT tactics; do not reuse historical receipts as proof of its identity or performance. Validate real inference, quality and runtime memory before treating a rebuilt engine as equivalent. The [launch patch](../patches/int4-gemv-cosmos-mlp-n4.patch), [GPU equivalence harness](../scripts/profile_mlp_gemv.cu), and [recorded build receipt](../results/rtn-build-20260921T040030Z-41c97631.json) document the measured version.
