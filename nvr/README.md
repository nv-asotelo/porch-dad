# NV-livestream-reachy-cosmos-demo — Live Vision + Reachy Mini + Cosmos3-Edge

A code-frozen demo branch of porch-dad: **Live Vision** (a lightweight, low-RAM camera/VLM web
UI) plus **Reachy Mini** motor/app/speech control, both talking to a locally-hosted
**Cosmos3-Edge** model on a Jetson Orin. Everything specific to the full NVR project this branch
was cut from — Frigate, Ring, Scout, the porch-feed command centre and its docker-compose stack —
has been removed outright, not just disabled. See `main` for that project.

```
USB webcam ──┐
             ├─▶ browser (getUserMedia) ──▶ Live Vision :8090/:8091 ──▶ Cosmos3-Edge shim :8000
Reachy Mini ─┘        │                           │
  camera/mic           │                    /api/reachy/* ──▶ Reachy Mini daemon :8000 (on the robot)
  (needs the bridge,    │                           │
   see below)           ▼                           ▼
                   /api/engines/*              motors, apps, speaker (Piper TTS)
                   (TensorRT engine swap)
```

## What's here

| Path | What it is |
|---|---|
| `nvr/ui/` | Live Vision: `scripts/serve_ui.py` (stdlib-only `http.server`, no framework) + `web/`. Camera/VLM UI, Reachy motor/app/TTS control, engine switching. |
| `nvr/reachy/reachy.py` | Thin REST client for the Reachy Mini daemon (motors, pose, apps, volume, sound upload/play). Zero NVR dependencies - only `requests`. |
| `nvr/reachy/reachy_mjpeg_bridge.py` | WebRTC-to-HTTP bridge for the robot's camera/mic (needed only for "switch to Reachy Mini" as a *video* source - motor/app/TTS control works without it). |
| `nvr/shim/cosmos3_shim_v1.py`, `nvr/bin/cosmos3-edge-infer` | The Cosmos3-Edge TensorRT-Edge-LLM serving shim. |
| `systemd/cosmos3-edge-shim.service`, `nvr/systemd/cosmos-edge-ui.service` | The two units Live Vision needs. |
| `nvr/systemd/dropins/` | `vm.swappiness=10` and `MemorySwapMax=0` for the shim - the model is ~3.8 GB resident and must not get paged out (see deploy/04 on main for the swap-thrashing writeup this fixes). |

## Running the camera/mic bridge (optional - only for "switch to Reachy Mini" video)

```
python3 -m venv reachy_env && reachy_env/bin/pip install -r nvr/reachy/requirements.txt
reachy_env/bin/python3 nvr/reachy/reachy_mjpeg_bridge.py \
  --robot-host <reachy-ip> --listen 127.0.0.1 --listen-port 8099 --fps 5
```

Installs cleanly from prebuilt aarch64 wheels on JetPack/Ubuntu 24.04 + Python 3.12 - no
compilation needed, confirmed on the clone below. `requirements.txt`'s own header explains the
pinned `aiortc==1.10.1` (a newer aiortc has the same RTX-decoding bug this version works around).

## Running serve_ui.py

```
python3 nvr/ui/scripts/serve_ui.py \
  --host 0.0.0.0 --port 8091 --backend-port 8000 --allow-insecure-lan \
  --reachy-daemon-url http://<reachy-ip>:8000 \
  --piper-bin /path/to/piper --piper-model /path/to/voice.onnx \
  --engine-link /path/to/engine-symlink --engines-config /path/to/engines.json
```

Every one of `--reachy-daemon-url`, `--piper-bin`/`--piper-model`, and `--engine-link`/
`--engines-config` is independently optional - omit any of them and that feature's UI simply
doesn't appear, rather than erroring. `--reachy-url` (the camera/mic bridge, default
`http://127.0.0.1:8099`) is separate from `--reachy-daemon-url` (the robot's own control API) -
video/audio and motor/app/TTS control can be deployed independently.

`--piper-bin`/`--piper-model`: see deploy/07 section 9 on main for why Piper (a self-contained
native binary, no Python/torch stack) over a larger model - the same reasoning applies here, this
branch's whole point is the smallest RAM footprint.

`--engine-link`/`--engines-config`: `EngineSwitcher` does an atomic `sudo -n ln -sfn` onto the
chosen engine directory, then `sudo -n systemctl restart <--shim-service>`. Needs passwordless
sudo for exactly those two commands (see deploy/03 on main for the sudoers line). The JSON config
is shaped like main's `config.yaml` `engines:` dict: `{"id": {"name", "path", "profile"}, ...}`.

## Deployed and smoke-tested on the bootable clone (2026-09-26)

Deployed to `jetson@<clone-ip>:/home/jetson/porch-dad-demo/` (mirroring this repo's `nvr/`
layout - `serve_ui.py` locates `reachy.py` via a relative path, see its own comment) and run as
`porch-dad-demo-ui.service` on port 8091, **alongside** that box's own pre-existing Live-Vision-
style setup on port 8090 - additive, nothing on the clone was replaced or disabled.

Verified against the real robot over the LAN:
- `GET /api/reachy/state` - live telemetry (pose, battery-relevant fields, motor mode).
- `POST /api/reachy/action/center` - real motor movement (`{"ok": true, "message": "centred"}`).
- `GET /api/reachy/apps` - the robot's 12 installed onboard apps.
- `POST /api/reachy/speak` - real Piper synthesis, uploaded and played through the robot's own
  speaker (`spoke in 1.76s (synth 0.77s, play 0.99s)`).
- The camera/mic bridge (`reachy_mjpeg_bridge.py`, its own `reachy_env` venv - `pip install -r
  nvr/reachy/requirements.txt` installed cleanly from prebuilt aarch64 wheels, no compilation
  needed), run as `reachy-mjpeg-bridge.service`: `/healthz` reports `"state": "live"` with real
  video (1280x720 JPEGs, confirmed by eye - a Charmander figurine on a desk, not noise) and real
  audio frames flowing, and a still fetched **through Live Vision's own relay**
  (`/reachy/still.jpg?token=...`, the same path the browser UI uses) came back correctly - the
  full "switch to Reachy Mini" video path is confirmed working end to end, not just the bridge in
  isolation. Unlike main's unit, this one drops `--push-url` (no Live VLM WebUI push target on
  this box) and `--recover-ssh`/`--recover-key` (no recovery key provisioned here).
- All 26 of `nvr/ui/tests/test_reachy_proxy.py` still pass unmodified.

- **The "Slow" (v3) engine: built, deployed, and serving real captions.** See the full build
  story below - this section only covers what's running now.
  - `GET /api/engines` (through Live Vision's own proxy) reports `v3`/"Slow" as `active`.
  - `POST /v1/chat/completions` against the real deployed shim, with a real camera frame, returned
    `"A Pokemon toy is sitting next to a laptop on a desk."` - accurate (it's a Charmander figurine
    on a desk next to a laptop) - in the standard OpenAI response shape, `usage` block included.
  - Engine switching's `EngineSwitcher` mechanism is live and correctly wired to this box's actual
    layout (`--engine-link /home/jetson/porch-dad-demo/engine-link`, `--engines-config
    /home/jetson/porch-dad-demo/engines.json`, `--shim-service porch-dad-shim-v3`) - not just
    verified against fake directories anymore.
  - All four relevant services active together with 3.3 GB still available: `cosmos-edge-ui`
    (the box's own, untouched), `porch-dad-demo-ui` (this branch's Live Vision), `porch-dad-
    shim-v3` (serves the v3 engine), `reachy-mjpeg-bridge`.

**One deliberate, documented tradeoff**: `cosmos-edge-backend.service` (the box's own default MLP
backend) is stopped and disabled, because its ~6.6 GB resident footprint and the v3 engine's
footprint cannot both fit in 8 GB - confirmed by two OOM kills hitting it directly while building
and testing v3 alongside it. This is not a bug to fix later; it is the actual point of choosing
the smallest-footprint engine; a board this size runs one VLM backend at a time. `cosmos-edge-ui`
(the box's own webUI on :8090) still loads but its inference calls will fail until that service is
restarted - reversible with `systemctl enable --now cosmos-edge-backend` (and stopping
`porch-dad-shim-v3` first, for the same memory reason in reverse).

## How the v3 engine was actually built (for reproducing this, or building v1/v2 the same way)

**The earlier "wrong checkpoint" conclusion during this work was wrong - a path error, corrected
below.** `nvidia/Cosmos3-Edge` is a Mixture-of-Transformers Omni model (per its own model card):
one `transformer/` checkpoint with two complementary towers, an autoregressive tower for text and
a diffusion tower for image/video/action, selected at export time by `--task {policy,reasoning}`.
Pointing `--src` at the snapshot root (rather than `transformer/` specifically) made the quantizer
find zero flat `.safetensors` files and silently no-op; the config fields that looked
diffusion-only (`action_dim`, `latent_channel`) belong to the *other* tower in the same Omni
checkpoint, not evidence this was the wrong model.

1. **Quantize**, pointed at the `transformer/` subfolder specifically:
   `scripts/rtn_int4_quantize.py --src .../transformer --dst cosmos3_int4_ckpt`. Reproduced
   deploy/04's documented numbers on main exactly: 169 linears, 11.06% mean relative weight error.
2. **Fix the checkpoint's `config.json` and index filename** before export - two issues the
   quantizer doesn't handle because it copies the source through unchanged:
   - Add `"model_type": "cosmos3_edge"` and the four multimodal token IDs (`image_token_id`,
     `video_token_id`, `vision_start_token_id`, `vision_end_token_id`) from the *snapshot root's*
     `config.json` (not `transformer/config.json`, which lacks them).
   - Rename `diffusion_pytorch_model.safetensors.index.json` to `model.safetensors.index.json`
     (the name the exporter's loader requires) and add the quantizer's `.weight_scale` index
     entries by hand (it would have added them automatically had it found the right name).
   - Copy `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`,
     `chat_template.jinja` from the snapshot root into the checkpoint dir - also not carried by
     `transformer/` alone, and required for serving.
3. **Export with `--int4-gemm-plugin-version 1`, not the default 2.** This was the real blocker
   this work hit and eventually solved: `tensorrt-edgellm-export --task reasoning --skip-visual`
   with the *default* V2 (cuteDSL fragment-layout) plugin produced an ONNX that builds up through
   graph optimization and then fails at `IBuilder::buildSerializedNetwork` with `Error Code 9:
   could not find any supported formats consistent with input/output data types` on an ordinary
   `Int4GroupwiseGemmPluginV2` node - reproducible regardless of KV cache capacity, and not
   resolved by re-exporting with this box's own git-checkout `tensorrt_edgellm` instead of the
   build workstation's pip-installed one (ruling out a version-skew explanation). V1 (the legacy
   AWQ-swizzled plugin) exports and builds cleanly with no further changes.
4. **Build the vision engine too, from the same source checkpoint** (`vision_encoder/` +
   the snapshot root's `config.json`, exported with `--skip-llm`) - the pre-built vision engine
   already on this clone (from its own MLP build) turned out to use externalized/refit weights
   tied to that specific build's checkpoint path and would not load against v3
   (`missing tensor model.projector.linear_fc1.bias`). Building fresh from the same source instead
   produced a self-contained engine matching deploy/04's documented size almost exactly (938 MB).
5. **Manually populate `content_types` in `processed_chat_template.json`** after export. The
   exporter's automatic chat-template extraction (`tensorrt_edgellm.chat_template.process_chat_template`,
   which tries `AutoProcessor`/`AutoTokenizer` with `trust_remote_code=True`) silently falls back
   to a minimal stub with `"content_types": {}` for this checkpoint - never raises, just produces
   a template the C++ tokenizer can't recognize `image` content in
   (`EDGELLM_BAD_MEDIA_COUNT: pad count is smaller than this request's media count`). The correct
   value, confirmed against the checkpoint's own `chat_template.jinja` and matching the
   `qwen3_omni.json` reference template in the exporter's own template library (Cosmos3-Edge's
   text tower is Qwen3-VL-based): `{"image": {"format": "<|vision_start|><|image_pad|><|vision_end|>"},
   "video": {"format": "<|vision_start|><|video_pad|><|vision_end|>"}}`.
6. **`llm_build`/`visual_build` had to be compiled on the clone first** - only their CMake targets
   existed (`cmake --build build --target llm_build -j$(nproc)`, same for `visual_build`; both
   link against the already-built `libNvInfer_edgellm_plugin.so`, so this was fast).
7. **Serve it**: this clone has no `cosmos3_shim_v1.py` deployment of its own (it uses a different
   serving script, `rtn_backend.py`, tied to its own cache-bundle layout that doesn't accept an
   arbitrary engine directory directly) - copied main's shim, patched four hardcoded
   `/home/orin/...` paths for this box's layout, changed its hardcoded port from 8000 (taken by
   the box's own backend) to 8001, and ran it as `porch-dad-shim-v3.service`.

Every fix above was applied to `/home/jetson/porch-dad-demo/ckpt/`'s `config.json` and to the
already-exported ONNX/engine directories directly on the clone - not yet folded back into
`scripts/rtn_int4_quantize.py` itself on this branch. Doing that (so a fresh quantize run needs
none of these manual steps) is the natural next cleanup, not yet done.
