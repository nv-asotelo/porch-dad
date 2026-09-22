# Agent deployment runbook for Cosmos3-Edge on Jetson Orin

Deploy the selected MLP INT4 backend and the existing **Local Vision Lab / Cosmos3 Edge Live View** interface from the public release. This is an installation task with a finite acceptance gate, not another optimization search. It does not install Porch Dad, Frigate, robots or the separate Live VLM WebUI application. No auxiliary GPU, PyTorch calibration environment or private repository is needed.

The commands below run **on the target Jetson after normal boot**, as its own non-root account, unless a step explicitly says client computer. Use a new writable checkout with a plain absolute path without spaces, `%` or shell metacharacters. Keep unrelated directories, processes and services intact. The public repository does not include reusable device credentials, a public demo endpoint, model weights or portable prebuilt engines.

Run the command blocks in one persistent Bash/SSH session with `set -euo pipefail`. If an agent tool creates a fresh shell or SSH process for each call, explicitly restore the earlier `COSMOS_*` variables and build environment before that stage; exports do not survive separate calls. Keep the same absolute receipt directory when resuming, and inspect completed stages rather than rerunning the fresh-clone or patch blocks.

## Compatibility and scope

| Target | Status for this release |
| --- | --- |
| Jetson Orin Nano 8 GB | Hardware used for the recorded deployment and inference evidence. A fresh installation still needs its own acceptance results. |
| Jetson Orin NX 8 GB or 16 GB | Build candidate only after the checks below pass. No deployment or performance result on these boards is claimed. |
| Jetson AGX Orin 32 GB or 64 GB | Build candidate only after the checks below pass. Keep the same engine profile for this installation; extra RAM is not a reason to change it silently. |
| Orin Nano 4 GB | Outside this selected profile's supported deployment scope. Its recorded shared-memory use exceeds this board's capacity. |
| Jetson Thor, Xavier, TX2 or original Jetson Nano | Outside this release's build path. Stop and report that a separate port/build/validation is required. Thor support elsewhere in TensorRT-Edge-LLM does not make this SM87-specific recipe portable. |

The **tested reference stack** is JetPack **7.2.1**, Jetson Linux/L4T **39.2.1**, CUDA compiler **13.2.86**, TensorRT **10.16.2.10**, Python **3.12**, on Linux aarch64 with an installed ext4 root filesystem. CUDA compiler build `13.2.86` is more precise than the advertised CUDA release label.

The machine-readable gate accepts the supported Orin family only with SM87, at least 7 GiB Linux-visible RAM, at least 30 GiB free working storage, L4T 39.2.x, CUDA 13.2, TensorRT 10.16.x and Python 3.12. An accepted patch-level variant is a **compatible build candidate**, not an already validated installation. An 8 GB module normally exposes less than 8 GiB to Linux. Larger models in the table do not inherit Nano latency, memory or accuracy numbers.

A matching JetPack name alone is insufficient: check the installed packages and execute the GPU probe. Do not change these gates to make a different board pass. Public context: [pinned backend support matrix](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/docs/source/user_guide/getting_started/support-matrix.md) and [NVIDIA GPU compute capabilities](https://developer.nvidia.com/cuda/gpus).

If the board needs OS installation or recovery, stop this runbook at that boundary. The [USB-C installation record](jetpack-install.md) describes one identified Nano and disk, not permission to overwrite another device. Flashing, repartitioning or reusing existing partitions requires a separate instruction identifying the actual board, target storage and data-disposition decision. Never infer a target disk, force flash or reuse the original task's disk layout.

## 1. Resolve the public release and record the exact commit

The public release is `nv-asotelo/porch-dad`, tag `cosmos3-edge-mlp-freeze-2026-09-21`, project subtree `cosmos3-edge-orin/`. The user explicitly authorized a deployment-documentation/helper update to that freeze. Resolve the tag once, record its full commit SHA, and use that SHA for subsequent repeats. A moving branch or tag label alone is not an immutable reproduction identifier. See [the freeze record](../deployment/FROZEN.md).

On the target, with Git already available:

```bash
set -euo pipefail
export COSMOS_RELEASE_CHECKOUT="$HOME/cosmos-edge-release"
export COSMOS_RELEASE_REF=cosmos3-edge-mlp-freeze-2026-09-21
test ! -e "$COSMOS_RELEASE_CHECKOUT"
git clone --filter=blob:none --no-checkout \
  https://github.com/nv-asotelo/porch-dad.git "$COSMOS_RELEASE_CHECKOUT"
git -C "$COSMOS_RELEASE_CHECKOUT" sparse-checkout init --cone
git -C "$COSMOS_RELEASE_CHECKOUT" sparse-checkout set cosmos3-edge-orin
git -C "$COSMOS_RELEASE_CHECKOUT" fetch origin "refs/tags/$COSMOS_RELEASE_REF"
export COSMOS_RELEASE_COMMIT="$(git -C "$COSMOS_RELEASE_CHECKOUT" rev-parse 'FETCH_HEAD^{commit}')"
git -C "$COSMOS_RELEASE_CHECKOUT" checkout --detach "$COSMOS_RELEASE_COMMIT"
export COSMOS_PROJECT_DIR="$COSMOS_RELEASE_CHECKOUT/cosmos3-edge-orin"
cd "$COSMOS_PROJECT_DIR"
test -f scripts/jetson_preflight.py
test -f scripts/configure_deployment.py
mkdir -p data/deployment-receipts
export COSMOS_RECEIPTS="$(mktemp -d "$COSMOS_PROJECT_DIR/data/deployment-receipts/$(date -u +%Y%m%dT%H%M%SZ).XXXXXX")"
printf '%s\n' "$COSMOS_RELEASE_COMMIT" > "$COSMOS_RECEIPTS/release-commit.txt"
printf '%s\n' "$COSMOS_RELEASE_REF" > "$COSMOS_RECEIPTS/release-tag.txt"
git show --no-patch --format=fuller "$COSMOS_RELEASE_COMMIT" > "$COSMOS_RECEIPTS/release.txt"
sha256sum sources.lock.json deployment/selected.env > "$COSMOS_RECEIPTS/release-inputs.sha256"
```

If the new checkout path already exists, inspect and resume it deliberately rather than deleting it or cloning over it. If a handoff supplies an expected full release SHA, compare it with `COSMOS_RELEASE_COMMIT` and stop on a mismatch. The `data/` receipt directory is ignored by Git. Build helpers also create uniquely named receipts under `results/`; treat new device paths, logs and inventory as local operational data and do not publish them automatically.

For an agent connecting remotely, use the target address/account supplied for this deployment and verify its SSH host fingerprint. Do not copy an address, password, account name or SSH key from historical evidence.

## 2. Inventory and gate the installed system

```bash
cd "$COSMOS_PROJECT_DIR"
python3 scripts/jetson_preflight.py --stage system \
  --output "$COSMOS_RECEIPTS/preflight-system.json"
bash scripts/device_inventory.sh > "$COSMOS_RECEIPTS/device-inventory.txt"
findmnt -n -o SOURCE,FSTYPE / > "$COSMOS_RECEIPTS/root-filesystem.txt"
ss -ltnp > "$COSMOS_RECEIPTS/listeners-before.txt"
cat /proc/swaps > "$COSMOS_RECEIPTS/swaps-before.txt"
```

The preflight exits 0 for a compatible candidate and 2 for a failed gate. Read the JSON; successful system inventory is not GPU inference acceptance. The native wrapper additionally requires the installed ext4 root and normally booted Orin, not a driverless recovery chroot.

Inspect ports 8000, 8090 and, if requested, 8443, plus any existing `cosmos-edge-backend.service` and `cosmos-edge-ui.service`. Do not take over another workload's ports or units. An upgrade of an existing task-owned deployment needs a recorded rollback and a deliberate service stop before rebuilding; the fresh-install commands here assume those names are available.

Leave system swap, existing cooling policy and unrelated services unchanged. Do not clear page caches or disable global services to manufacture headroom. If memory is insufficient, preserve the failure and report the active workload/resource constraint.

Record the current power and clock state:

```bash
sudo /usr/sbin/nvpmodel -q > "$COSMOS_RECEIPTS/power-before.txt"
sudo /usr/bin/jetson_clocks --show > "$COSMOS_RECEIPTS/clocks-before.txt"
```

The recorded controlled reference used stock 25 W with dynamic clocks. **Do not hardcode power-mode ID 2, or assume one ID means MAXN on every Orin.** Preserve the current power mode unless the deployment request explicitly selects another. If MAXN or MAXN_SUPER is requested, inspect the target's configuration and choose the numeric ID corresponding to the supported label on that exact module/carrier:

```bash
readlink -f /etc/nvpmodel.conf
sed -n '/< POWER_MODEL/p' /etc/nvpmodel.conf
sudo /usr/sbin/nvpmodel -h
# Only after selecting and recording the actual supported ID:
# sudo /usr/sbin/nvpmodel -m "$CHOSEN_NVP_ID"
```

A power-mode change may require a reboot. Follow the target's message rather than forcing through it. MAXN and fixed clocks are different settings: this release keeps `COSMOS_STATIC_CLOCKS=0` and does not call `jetson_clocks` to lock clocks. That environment value does not undo a pre-existing clock lock. If this is not a fresh dynamic-clock state, restore only a verified target-local snapshot under the appropriate ownership/authorization, then record the observed state; never use a snapshot copied from another board.

## 3. Prepare the pinned backend and apply all six patches

Install build prerequisites, without replacing the already gated CUDA/TensorRT stack:

```bash
sudo apt update
sudo apt install -y build-essential binutils cmake git python3-venv python3-dev curl openssl
cd "$COSMOS_PROJECT_DIR"
mkdir -p external
test ! -e external/TensorRT-Edge-LLM
git clone --branch v0.10.1 --depth 1 \
  https://github.com/NVIDIA/TensorRT-Edge-LLM.git external/TensorRT-Edge-LLM
test "$(git -C external/TensorRT-Edge-LLM rev-parse HEAD)" = \
  e8b29522938901f6df19ebeedd4b69bc8edbcd97
git -C external/TensorRT-Edge-LLM submodule update --init --recursive
git -C external/TensorRT-Edge-LLM diff --exit-code
```

Apply the complete selected patch set once to this clean pin, in order:

```bash
for patch_name in \
  cosmos3-patch-embedding-chw.patch \
  cosmos3-half-pixel-position.patch \
  tensorrt-edge-llm-v0.10.1-encoder-cache-budget.patch \
  int4-gemv-cosmos-mlp-n4.patch \
  cosmos-runtime-image-token-budget.patch \
  cosmos-encoder-cache-bypass.patch; do
  sha256sum "patches/$patch_name" >> "$COSMOS_RECEIPTS/patches.sha256"
  git -C external/TensorRT-Edge-LLM apply --check "$COSMOS_PROJECT_DIR/patches/$patch_name"
  git -C external/TensorRT-Edge-LLM apply "$COSMOS_PROJECT_DIR/patches/$patch_name"
done
git -C external/TensorRT-Edge-LLM diff --binary > "$COSMOS_RECEIPTS/backend-task-changes.patch"
git -C external/TensorRT-Edge-LLM submodule status --recursive > "$COSMOS_RECEIPTS/backend-submodules.txt"
```

The first two patches repair the Cosmos visual projection/position handling. The remaining patches supply encoder-cache control, selected N4 MLP kernel tiling and bounded runtime image budgets/cache bypass. Later patches depend on earlier ones. A failed patch check is a real mismatch: do not skip it, apply with fuzz, or switch the backend to `main`. On resume, inspect saved progress and the actual diff; do not run this fresh-checkout patch block again over an already patched tree.

The separate Live VLM WebUI repository is a credited workflow/prompt reference, not a runtime prerequisite. Do not run a broad source-fetch script to install optional applications for this recipe.

## 4. Install the Python dependencies before invoking the native wrapper

This ordering is deliberate. `build_backend.sh` begins with `pip check`; installing the server extras or correcting the exposed system SciPy/CFFI only after that call is too late.

```bash
cd "$COSMOS_PROJECT_DIR/external/TensorRT-Edge-LLM"
python3 -m venv --system-site-packages .venv
export COSMOS_PY="$COSMOS_PROJECT_DIR/external/TensorRT-Edge-LLM/.venv/bin/python"
"$COSMOS_PY" -m pip install --upgrade pip
"$COSMOS_PY" -m pip install -e '.[server,server-tools]'
"$COSMOS_PY" -m pip install 'pybind11==3.0.4' \
  'nvidia-cutlass-dsl[cu13]==4.7.0' 'cupy-cuda13x==13.6.0' \
  'cuda-python>=12.6,<14'
"$COSMOS_PY" -m pip install --only-binary=:all: \
  'numpy==2.2.6' 'scipy==1.15.3' 'cffi==2.0.0'
"$COSMOS_PY" -m pip check
"$COSMOS_PY" -m pip freeze > "$COSMOS_RECEIPTS/python-packages.txt"
export PATH="$COSMOS_PROJECT_DIR/external/TensorRT-Edge-LLM/.venv/bin:/usr/local/cuda-13.2/bin:$PATH"
export TRT_PACKAGE_DIR=/usr
export CUDA_PATH=/usr/local/cuda-13.2
export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu:/usr/local/cuda-13.2/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$COSMOS_PROJECT_DIR"
```

`--system-site-packages` exposes the JetPack TensorRT bindings. The compatible NumPy/SciPy/CFFI wheels above live inside the venv and avoid changing system Python. The pinned upstream server extras include the required server and tokenizer/template packages. Do not substitute the full `tools`/AWQ environment or install PyTorch to perform this CPU RTN conversion. A missing compatible wheel or a nonzero `pip check` is a blocker to resolve in this environment, not a reason to suppress the check. The runtime preflight runs after native compilation in the next step because it also verifies the compiled binding.

## 5. Build the native runtime on the target

Do not run the model service, engine construction, native compilation or other heavy tasks simultaneously on the shared-memory device.

```bash
cd "$COSMOS_PROJECT_DIR"
bash scripts/build_backend.sh fp16
python3 scripts/jetson_preflight.py --stage runtime --python "$COSMOS_PY" \
  --output "$COSMOS_RECEIPTS/preflight-runtime.json"
```

Here `fp16` names the wrapper's **FMHA kernel-group build mode**, not the selected model precision. The selected MLP INT4 path uses the V1 plugin and the applied N4 patch; the optional V2 kernel group from wrapper mode `int4` is not required for this deployment.

The wrapper uses one compiler worker, runs a GPU preflight, generates the SM87 FMHA kernels, builds `_edgellm_runtime` and `NvInfer_edgellm_plugin`, and verifies the native import. The subsequent runtime preflight checks the compiled image-budget/cache-bypass binding, GPU count, actual SM87 identity, a real CuPy GPU calculation and TensorRT. Neither check proves model inference. Preserve its new `results/build-fp16.*` directory and exit status. The historical 45-minute build is context, not a time limit or guaranteed build duration. Wait for actual progress/completion; do not count a launched compiler or a completed Python package install as a successful native build.

## 6. Download and verify the reasoner snapshot

Download the root metadata/notices and only the shards referenced by the reasoner index at NVIDIA revision `344d602b128d1bbdacb43b08d0a3626f46343e29`. Keep the generator/VAE component configurations out of this model directory. The frozen `results/model-download.json` supplies expected file hashes at that pin; it is a **digest catalog**, not proof that this new device downloaded anything.

The allowlisted downloader includes all 21 files, including notices and `.gitattributes`, verifies their SHA-256 digests, and writes a fresh local receipt:

```bash
cd "$COSMOS_PROJECT_DIR"
export COSMOS_REASONER_DIR="$COSMOS_PROJECT_DIR/models/cosmos3-edge-reasoner"
"$COSMOS_PY" scripts/fetch_reasoner.py \
  --output "$COSMOS_REASONER_DIR" \
  --receipt "$COSMOS_RECEIPTS/reasoner-download.json"
```

The output must be a new directory. If a download is interrupted, use `--resume` only for this helper's partial directory with its matching marker/reference; choose a new receipt filename. `--verify-only` hashes an existing snapshot without network access and also needs a new receipt path. Do not reuse another application's mixed model folder. No existing receipt is overwritten. The expected download totals about 7.74 GB; download size is not GPU memory use. Retain NVIDIA's source model card and [OpenMDW-1.1 terms](https://openmdw.ai/license/1-1/). Network/authentication or digest failure must be reported honestly; do not substitute another revision, omit files or fabricate the receipt.

## 7. Convert only the MLP weights on the CPU

The converter reads the original checkpoint, verifies its recorded source hashes, and creates a separate derivative. It uses symmetric group-128 round-to-nearest packing, not AWQ/GPTQ calibration. Specify `mlp-only` explicitly; its default scope would quantize more layers.

```bash
cd "$COSMOS_PROJECT_DIR"
export COSMOS_MLP_DIR="$COSMOS_PROJECT_DIR/models/cosmos3-edge-rtn-mlp-int4"
"$COSMOS_PY" scripts/quantize_cosmos3_rtn.py \
  --source "$COSMOS_REASONER_DIR" --output "$COSMOS_MLP_DIR" \
  --quantization-scope mlp-only > "$COSMOS_RECEIPTS/mlp-conversion-plan.json"
"$COSMOS_PY" scripts/quantize_cosmos3_rtn.py \
  --source "$COSMOS_REASONER_DIR" --output "$COSMOS_MLP_DIR" \
  --quantization-scope mlp-only --apply \
  > "$COSMOS_RECEIPTS/mlp-conversion.log" 2>&1
"$COSMOS_PY" - <<'PY'
import hashlib
import json
import os
from pathlib import Path
model = Path(os.environ['COSMOS_MLP_DIR'])
record = json.loads((model / 'rtn-conversion.json').read_text())
assert record['complete'] is True
assert record['quantization_scope'] == 'mlp-only'
assert record['quantized_linear_count'] == 56
assert record['source_revision'] == '344d602b128d1bbdacb43b08d0a3626f46343e29'
digest = hashlib.sha256()
with (model / 'model.safetensors').open('rb') as source:
    for block in iter(lambda: source.read(4 << 20), b''):
        digest.update(block)
assert digest.hexdigest() == record['model_safetensors_sha256']
output = Path(os.environ['COSMOS_RECEIPTS']) / 'mlp-conversion-verified.json'
with output.open('x') as stream:
    json.dump(record, stream, indent=2)
    stream.write('\n')
print(output)
PY
```

The output directory must not already exist. Preserve an incomplete candidate and its log before deciding how to resume. The `rtn-conversion.json` hash receipt describes **this conversion**; copying the historical derivative's receipt is not acceptable. Its `inference_validated: false` is intentional. Attention, LM head, visual/projector/embedding components and other unquantized tensors remain FP16; runtime activations/KV also remain FP16.

## 8. Build this target's engine with the selected capacities

Do not copy a Nano's TensorRT engine to an NX/AGX and declare it portable. Build on the actual target with its gated libraries. Even a rebuild on another Nano can select different tactics and must pass fresh inference checks.

```bash
cd "$COSMOS_PROJECT_DIR"
export COSMOS_ENGINE_CACHE="$COSMOS_PROJECT_DIR/data/engine-cache-mlp-compact"
export TMPDIR="$COSMOS_PROJECT_DIR/data/tmp"
export XDG_CACHE_HOME="$COSMOS_PROJECT_DIR/data/cache"
export CUDA_CACHE_PATH="$COSMOS_PROJECT_DIR/data/cache/cuda"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH"
"$COSMOS_PY" scripts/rtn_backend.py build \
  --model "$COSMOS_MLP_DIR" --cache-dir "$COSMOS_ENGINE_CACHE" \
  --max-input-len 1024 --max-kv-capacity 1664 \
  --max-image-tokens 512 --max-image-tokens-per-image 512 \
  --encoder-embedding-cache-budget-bytes 0 \
  > "$COSMOS_RECEIPTS/engine-build.log" 2>&1
```

Use the **literal** input/KV/image capacities `1024 / 1664 / 512 / 512`; omitting the visual arguments or relying on the helper's KV default selects another profile. Batch size is one. The wrapper records model/native/TensorRT provenance and validates normalized chat-template/media/RoPE artifacts. Preserve the emitted `results/rtn-build-*.json` receipt and its actual bundle path. Do not overwrite an existing unrelated cache or bypass a hash/profile mismatch. Build completion is not image-inference acceptance.

## 9. Generate target-local configuration and install services

```bash
cd "$COSMOS_PROJECT_DIR"
python3 scripts/configure_deployment.py \
  --model-dir "$COSMOS_MLP_DIR" --cache-dir "$COSMOS_ENGINE_CACHE"
cp deployment/local.env "$COSMOS_RECEIPTS/local.env"
sudo bash scripts/install_services.sh --user "$(id -un)" \
  --env-file "$COSMOS_PROJECT_DIR/deployment/local.env" --dry-run
# After checking that the rendered paths/account/units belong to this deployment:
sudo bash scripts/install_services.sh --user "$(id -un)" \
  --env-file "$COSMOS_PROJECT_DIR/deployment/local.env"
sudo systemctl start cosmos-edge-backend.service
```

`configure_deployment.py` runs as the target user, creates the task cache/temp directories and writes private `deployment/local.env`. It rewrites paths without editing the frozen `deployment/selected.env` evidence. It does not change power, clocks or performance settings. Existing local configuration is preserved unless deliberately replaced with its `--force` option. The service installer accepts the real non-root account, derives the project directory, installs/enables units and does **not** start them. The commands above then start only the selected backend.

Confirm these values in `local.env` before installation: profile `rtn-v1`; input 1024; KV 1664; built total/per-image limits 512/512; runtime image default 512; top-p 1; encoder-cache budget 0; static clocks 0. The UI defaults to output cap **64**, temperature 0, the original concise-scene prompt and Lightweight capture. Input image tokens and output tokens are independent; an editable output ceiling of 512 is not the first-load default. Do not silently switch to the intermediate 320/cache256MiB/static-clock settings.

The launcher validates the completed checkpoint, native binding and exact cache profile at startup. It is intended to load an existing engine, not build missing engines in a service restart loop. Wait for readiness with a finite deadline:

```bash
"$COSMOS_PY" - <<'PY'
import time
from urllib.request import urlopen
end = time.monotonic() + 300
last = None
while time.monotonic() < end:
    try:
        with urlopen('http://127.0.0.1:8000/health/ready', timeout=3) as response:
            if response.status == 200:
                print(response.read().decode())
                break
    except Exception as error:
        last = str(error)
    time.sleep(2)
else:
    raise SystemExit('Backend did not become ready in 300 seconds: ' + str(last))
PY
systemctl is-active cosmos-edge-backend.service
export COSMOS_BACKEND_INVOCATION="$(systemctl show cosmos-edge-backend.service -p InvocationID --value)"
test -n "$COSMOS_BACKEND_INVOCATION"
sudo journalctl -u cosmos-edge-backend.service \
  "_SYSTEMD_INVOCATION_ID=$COSMOS_BACKEND_INVOCATION" --no-pager -o cat \
  > "$COSMOS_RECEIPTS/backend-startup.log"
```

A deadline failure means inspect the actual build/startup logs and storage/kernel state. Do not repeatedly restart a model still paging/loading, extend the deadline indefinitely, run concurrent builders, or report readiness from systemd's `active` state alone.

Validate the **newly emitted** build and serve receipts before using the API. `/api/runtime` provides a runtime engine ID and live controls; it does not expose the selected model path, exact cache bundle or input/KV capacities. Those must be checked against the new startup receipt, not inferred from a friendly profile name:

```bash
"$COSMOS_PY" - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

root = Path(os.environ['COSMOS_PROJECT_DIR']).resolve()
receipts = Path(os.environ['COSMOS_RECEIPTS'])
def load_emitted(log_name, expected_mode):
    matches = re.findall(r'^RTN receipt: (.+)$', (receipts / log_name).read_text(), re.M)
    assert len(matches) == 1, (log_name, 'Expected one receipt from this invocation')
    path = Path(matches[0].strip()).resolve()
    assert path.is_relative_to(root / 'results') and path.is_file()
    record = json.loads(path.read_text())
    assert record['mode'] == expected_mode
    for key, expected in {'max_input_len': 1024, 'max_kv_cache_capacity': 1664,
                          'max_batch_size': 1, 'max_image_tokens': 512,
                          'max_image_tokens_per_image': 512,
                          'int4_gemm_plugin_version': 1}.items():
        assert record['build_options'][key] == expected, (key, record['build_options'])
    assert Path(record['model_dir']).resolve() == Path(os.environ['COSMOS_MLP_DIR']).resolve()
    assert Path(record['cache_dir']).resolve() == Path(os.environ['COSMOS_ENGINE_CACHE']).resolve()
    assert record['checkpoint_provenance']['quantization_scope'] == 'mlp-only'
    assert record['checkpoint_provenance']['quantized_linear_count'] == 56
    saved = receipts / ('engine-' + expected_mode + '.json')
    assert not saved.exists()
    shutil.copyfile(path, saved)
    return record, hashlib.sha256(saved.read_bytes()).hexdigest()

built, build_hash = load_emitted('engine-build.log', 'build')
served, serve_hash = load_emitted('backend-startup.log', 'serve')
assert served['runtime_initialized'] is True
assert served['bundle_dir'] == built['bundle_dir']
with (receipts / 'startup-cache-validation.json').open('x') as output:
    json.dump({'validated': True, 'bundle_dir': served['bundle_dir'],
               'build_receipt_sha256': build_hash, 'serve_receipt_sha256': serve_hash,
               'model_inference_validated': False}, output, indent=2)
    output.write('\n')
PY
```

## 10. Run fresh functional acceptance, then open the UI

Close other clients and keep the newly started backend idle before this bounded functional suite:

```bash
cd "$COSMOS_PROJECT_DIR"
"$COSMOS_PY" scripts/validate_runtime_controls.py --policy lightweight \
  --base-url http://127.0.0.1:8000 \
  --request-log "$COSMOS_PROJECT_DIR/data/logs/native-requests.jsonl" \
  --output "$COSMOS_RECEIPTS/runtime-controls.json"
sudo systemctl start cosmos-edge-ui.service
curl --fail --silent --show-error http://127.0.0.1:8090/health/ready \
  > "$COSMOS_RECEIPTS/ui-ready.json"
curl --fail --silent --show-error http://127.0.0.1:8090/api/runtime \
  > "$COSMOS_RECEIPTS/runtime-loaded.json"
"$COSMOS_PY" scripts/benchmark.py run \
  --url http://127.0.0.1:8090/v1/chat/completions --model Cosmos3-Edge \
  --image benchmarks/fixtures/jpeg/01-left-right.jpg \
  --candidate-id installation-smoke --requests 1 --warmup 0 \
  --max-tokens 64 --temperature 0 --top-p 1 \
  --output "$COSMOS_RECEIPTS/image-smoke.jsonl"
```

The runtime suite checks load defaults, requested image budgets, invalid-input rejection, cache behavior, native timings and actual streaming against new server request IDs/log rows. It deliberately makes more than one inference and is **functional acceptance**, not a benchmark or accuracy score. The final single-image command exercises the UI's proxy too; its fewer-than-30-samples warning is expected for this smoke test. Check its successful stream and inspect the actual caption against the red-circle/blue-square fixture. Record any wrong object, color or relation rather than hiding it. The selected MLP model already has documented quality limitations; a successful HTTP response is not semantic correctness.

On the **client computer**, replace the placeholders with this deployment's supplied SSH account/address:

```bash
ssh -N -L 127.0.0.1:8090:127.0.0.1:8090 YOUR_JETSON_USER@YOUR_JETSON_ADDRESS
```

Open **http://localhost:8090**. This is the default access route: UI loopback 8090, backend loopback 8000. Verify the `LOCAL VISION LAB` page, backend-ready state, uploaded-image manual inference and streamed caption. Check CPU/GPU/shared-RAM telemetry and that browser timing is separate from server TTFT/Latency. A client browser may use its camera on localhost; only use a real camera when that input is authorized. Use a synthetic upload otherwise.

### Optional direct HTTPS on the target

Only if direct LAN access is requested, obtain the target's own IPv4 address, then use the target-local helper. Do not reuse the previous device's certificate or address:

```bash
cd "$COSMOS_PROJECT_DIR"
export JETSON_IPV4=YOUR_DEVICE_IPV4
sudo bash scripts/enable_lan_ui.sh --user "$(id -un)" \
  --env-file "$COSMOS_PROJECT_DIR/deployment/local.env" --dry-run "$JETSON_IPV4"
sudo bash scripts/enable_lan_ui.sh --user "$(id -un)" \
  --env-file "$COSMOS_PROJECT_DIR/deployment/local.env" "$JETSON_IPV4"
sudo systemctl restart cosmos-edge-ui.service
openssl x509 -in deployment/tls/orin.crt -noout -fingerprint -sha256
```

The helper verifies a local IPv4 and creates a **fresh device certificate/key**; each actual invocation rotates them. It changes listeners to HTTP 8090 plus HTTPS 8443 on all interfaces, but does not restart the UI itself. Visit `https://YOUR_DEVICE_IPV4:8443` and verify its certificate fingerprint through your trusted SSH session before granting browser camera access. The retained filename `orin.crt` does not imply the old certificate is reused.

This demo has no application authentication. HTTPS protects transport but does not restrict who can invoke it, and HTTP API/health routes remain reachable on 8090. Keep these listeners on an appropriately restricted trusted network; use an authenticated front end for wider access. The loopback backend stays on 8000. On an existing installation, neither helper automatically removes old LAN exposure/drop-ins; inspect them rather than claiming an SSH tunnel made the device loopback-only.

## Completion, reporting and stopping

Finish when the local evidence shows the correct target/pins, successful native and target-engine builds, validated source/derivative hashes, correct loaded capacities/defaults, successful fresh runtime acceptance and a working browser upload/stream. Confirm both units are enabled and active, record the actual access URL, and check the backend has no unexpected active/queued requests when the smoke test is over. Report whether reboot persistence was actually tested; enabled units alone do not prove a reboot test.

The handoff should contain:

- Actual board/RAM/SM and installed software versions, resolved release commit, backend/model pins, selected power mode and observed clocks.
- New model-download/conversion/build receipt paths and hashes, loaded engine ID, input/KV/visual capacities, service account and private configuration path.
- Fresh runtime acceptance and smoke/browser evidence, actual captions and limitations, resource-counter definitions, and the usable URL or SSH forwarding instruction.
- Any failures, manual interventions, remaining deviations and rollback information. Distinguish compile success, engine build success, readiness, functional inference and quality review.

Keep these new receipts local unless publication is requested. Historical files under `results/` describe the original device; never copy them into a new acceptance report as if this board produced them. A single smoke timing is not comparable performance evidence. Server TTFT/Latency exclude JPEG encode/decode, admission and network; browser timings measure a broader experience. Shared system RAM includes the OS and applications and is not dedicated GPU VRAM.

Stop after acceptance. Do not change precision scope, profile capacities, power policy, image detail, prompts or token limits to pursue marginal gains. If a gate fails, preserve the failure/log, identify the specific incompatibility and resolve only the deployment issue within the authorized scope. A required platform port, destructive reflash, conflicting existing service or unapproved asset/source is a boundary to report, not a reason to silently switch implementations.

## Copy-paste handoff prompt

Replace the two connection placeholders and optionally supply a trusted full release SHA. Do not put a password in the prompt or repository.

```text
Deploy the selected Cosmos3-Edge MLP INT4 backend and Local Vision Lab streaming UI on my Jetson at YOUR_JETSON_ADDRESS as YOUR_JETSON_USER. Use the public nv-asotelo/porch-dad release tag cosmos3-edge-mlp-freeze-2026-09-21, subtree cosmos3-edge-orin, and follow docs/agent-deployment.md chronologically. Resolve and record the full release commit before proceeding; use a supplied expected full SHA if present.

Inventory the actual device and run the machine-readable system and runtime preflights. The tested hardware is Orin Nano 8 GB; Orin NX 8/16 GB and AGX Orin 32/64 GB are only build candidates after the matching SM87/JetPack stack and memory/storage gates. Stop on unsupported hardware or stack mismatch. Do not flash, repartition, reuse existing partitions or alter unrelated workloads without a separately explicit instruction.

Use backend e8b29522938901f6df19ebeedd4b69bc8edbcd97 with all six release patches and model revision 344d602b128d1bbdacb43b08d0a3626f46343e29. Install server extras and the pinned NumPy/SciPy/CFFI compatibility wheels before the native build wrapper. Download and hash-check the reasoner files, perform the 56-MLP-only CPU RTN conversion with its new receipt, and build the engine on this target. No auxiliary GPU, AWQ/PyTorch calibration, internal repository, Porch Dad or separate Live VLM WebUI installation is needed.

Keep input1024/KV1664/built visual512 per-image512, image default512, encoder cache0, static clocks0, top-p1 and UI output64/temperature0/lightweight capture. Generate target-local paths/account configuration. Preserve the board's current power mode unless I explicitly request a change; if MAXN is requested, identify its actual supported local mode ID rather than assuming a number. Never enable fixed clocks implicitly.

Use SSH-forwarded localhost access by default. If I request direct LAN access, generate a fresh target-local HTTPS certificate and report its fingerprint and the unauthenticated listener scope. Never inherit a prior device address, credentials or certificate.

Validate startup cache identity, run fresh runtime-control acceptance, one synthetic-image request through the UI proxy, and a browser upload/stream check. Store new local receipts and report the actual caption and limitations. Do not represent historical results as this device's tests. Converge on a working installation, then stop; do not start an optimization search. Return the usable webpage/tunnel instructions, resolved source pins, actual hardware/software/profile, receipt paths, manual interventions and any unresolved failure.
```
