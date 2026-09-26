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

**Not yet deployed/verified there:**
- Engine switching against that box's actual engine layout - its existing engines use a different
  selection mechanism (`deployment/selected.env` sourced by `run_selected_backend.sh`) than this
  branch's `EngineSwitcher` (a stable symlink `ln -sfn` swap, matching main's porch-feed
  convention). `--engine-link`/`--engines-config` were left unset for this smoke test; the
  mechanism itself was verified separately against fake local engine directories.
- The "Slow" (v3) engine itself: **blocked**. The `nvidia/Cosmos3-Edge` checkpoint cached on the
  build workstation is `Cosmos3OmniPipeline` (video/image diffusion generation), not the
  reasoning/captioning causal-LM the shim actually serves - same repo name, wrong variant, and the
  repo has only one branch (`main`) with no tag preserving an older layout. Needs either the
  already-built engine from main's original SD card, or the correct source checkpoint identified.
