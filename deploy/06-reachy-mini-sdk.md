# 06 — Reachy Mini SDK: pointing its camera at Cosmos3-Edge

> **On the Orin, the robot is wired differently: see
> [`deploy/07`](07-reachy-homeassistant-and-orin-changes.md).** There, `reachy-mjpeg-bridge.service`
> holds one WebRTC session to the robot and serves its camera and microphone over HTTP to Frigate,
> the Live VLM WebUI, Live Vision and the command centre (§1, §5–§7), and §8 is the boot checklist.
> This page is the SDK route, for a machine next to the robot. Two parts of it touch the Orin's
> path: `release_camera_for_webui.py` (§2 below) makes the daemon release its media, which the
> bridge treats as a reason to go dormant until the media is back; and an SDK client using the
> camera over the network is one more WebRTC session on a daemon that leaks sockets per session
> (`deploy/07` §1 and §7).

This step is separate from the core recipe in `deploy/01`–`05`: it does not change the shim or
the engines, it wires a second camera source — [Reachy Mini](https://huggingface.co/docs/reachy_mini)'s
onboard camera, accessed through its Python SDK — into the same Cosmos3-Edge shim those steps
stood up. Two phases, in order: first through Live VLM WebUI (no new code, reuses the browser
pipeline from `deploy/05`), then natively (no browser at all — a small Python script talks to the
shim directly). Both assume `cosmos3-edge-shim.service` from `deploy/05` is already running.

Reference: [Reachy Mini SDK installation guide](https://huggingface.co/docs/reachy_mini/SDK/installation)
and the [Python SDK reference](https://huggingface.co/docs/reachy_mini/SDK/python-sdk).

---

## 1. Install the SDK

Run this on whichever machine is physically connected to (or on the same network as) Reachy Mini.
That does **not** have to be the Jetson — the SDK only needs network/USB reach to the robot and to
the shim's port 8000; see §4 for the two-machine case.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh          # uv: package/interpreter manager
uv python install 3.12 --default                         # SDK supports 3.10-3.12; 3.12 recommended
sudo apt install git git-lfs && git lfs install

uv venv reachy_mini_env --python 3.12
source reachy_mini_env/bin/activate
uv pip install "reachy-mini"
```

Linux additionally needs GStreamer (see the SDK's
[GStreamer installation guide](https://huggingface.co/docs/reachy_mini/SDK/gstreamer-installation))
and USB permissions for the daemon link:

```bash
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", MODE="0666", GROUP="dialout"
SUBSYSTEM=="usb", ATTRS{idVendor}=="38fb", ATTRS{idProduct}=="1001", MODE="0666", GROUP="dialout"' \
  | sudo tee /etc/udev/rules.d/99-reachy-mini.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo usermod -aG dialout "$USER"   # log out and back in for this to take effect
```

Then install this repo's bridge dependencies into the same environment:

```bash
uv pip install -r reachy/requirements.txt
```

Verify the camera works before wiring anything to Cosmos3-Edge:

```python
from reachy_mini import ReachyMini
with ReachyMini() as mini:
    frame = mini.media.get_frame()
    print(frame.shape, frame.dtype)   # (height, width, 3) uint8
```

---

## 2. Phase 1 — Reachy Mini's camera through Live VLM WebUI

The WebUI (`systemd/live-vlm-webui.service`, port 8090) picks up whatever camera the browser
exposes to it — it has no idea Reachy Mini exists, and it does not need to. The only problem to
solve is that the reachy_mini **daemon** normally holds the camera device for itself (for
`mini.media.get_frame()`, head tracking, etc.), so the OS-level camera picker in the browser won't
see it unless that hold is released.

`media_backend="no_media"` does exactly that: it tells the daemon to give up the camera (and
audio) hardware for as long as the process holds it open, and reacquire it automatically on exit.
This repo's [`../reachy/release_camera_for_webui.py`](../reachy/release_camera_for_webui.py) is
that hold, as a standalone script:

```bash
python3 reachy/release_camera_for_webui.py
```

Leave it running, then open `http://<webui-host>:8090/` in a browser **on the machine physically
connected to Reachy Mini** and select the Reachy Mini camera from the browser's device picker.
From here it is exactly the pipeline `deploy/05-serve-and-webui.md` §6 already describes: the
browser base64-encodes frames as `image_url` data URLs and POSTs them to the shim.

Two things to check if the camera does not appear or the page can't get frames, both called out in
`deploy/05` §6 and worth repeating here because Reachy Mini's daemon usage makes them more likely:

- **Secure-context restriction.** Browsers generally only grant camera access on `https://` or
  `localhost`. If the WebUI host and the browser host differ, the browser may block camera access
  outright — open the WebUI from the same machine Reachy Mini is attached to, or put a reverse
  proxy with TLS in front of it.
- **The daemon must actually have released the camera.** If `release_camera_for_webui.py` is not
  running (or was killed), the OS device is back in the daemon's hands and the browser will not see
  it, or will see it as busy. Ctrl+C hands it back to the daemon.

Stop `release_camera_for_webui.py` (Ctrl+C) when done with this phase — the daemon reacquires the
camera and things like head tracking work again.

---

## 3. Phase 2 — natively, no browser

[`../reachy/cosmos_bridge.py`](../reachy/cosmos_bridge.py) replaces the browser entirely: it pulls
frames straight from the SDK with `mini.media.get_frame()`, resizes and JPEG-encodes them, and
POSTs to the shim's `/v1/chat/completions` on a fixed interval — the same request shape
`nvr/bridge/porch_dad.py` uses for Frigate/Ring clips, just with one live frame instead of several
sampled ones.

```bash
python3 reachy/cosmos_bridge.py                 # reads reachy/config.yaml
# or: python3 reachy/cosmos_bridge.py /path/to/other-config.yaml
```

It prints one line per successful description:

```
[cosmos-bridge] 612ms :: A person stands in front of the camera, waving.
```

Relevant knobs in [`../reachy/config.yaml`](../reachy/config.yaml):

| Key | Purpose |
|---|---|
| `cosmos3_url` | shim address — `http://127.0.0.1:8000` if this script runs on the Jetson, `http://<jetson-ip>:8000` otherwise |
| `frame_width` | resize target before JPEG encode; 640 lands near the 320-token image cap, matching the "320 for live camera work" recommendation in the top-level README |
| `interval_seconds` | gap between captures — the shim is single-flight (`deploy/05` §2.4), so this should comfortably exceed one round trip or requests queue up |
| `connection_mode` | forwarded to `ReachyMini()` — `"localhost_only"` for USB/same-host, `"network"` for Wi-Fi, `null` to auto-detect |

`media_backend="default"` is used (not `"no_media"`) — unlike Phase 1, this script does not need
the OS-level camera device, it goes through the SDK's own media pipeline, so it can coexist with
head tracking or other daemon features that also use the camera.

---

## 4. Running the bridge on a different machine than the shim

Nothing here requires the Reachy Mini SDK and the shim to share a host. `cosmos_bridge.py` only
needs HTTP reach to the shim's port 8000 and SDK reach (USB or Wi-Fi, per `connection_mode`) to the
robot; point `cosmos3_url` at `http://<jetson-ip>:8000` and run the script wherever the robot is.
The shim binds `0.0.0.0` with no authentication (`deploy/05` §4) — treat this exactly like the
WebUI's own security note: trusted network only, or behind a reverse proxy / firewall rule.

---

## 5. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Browser camera picker doesn't list Reachy Mini | `release_camera_for_webui.py` isn't running, or the daemon reacquired the camera because the script exited |
| `cosmos_bridge.py` never prints a line | check `cosmos3_url` is reachable (`curl http://<host>:8000/v1/models`) and that `cosmos3-edge-shim.service` is up |
| Every `cosmos_bridge.py` frame is skipped with a `requests` timeout/connection error | shim not running, wrong `cosmos3_url`, or a firewall blocking port 8000 between the two machines |
| `ModuleNotFoundError: reachy_mini` or `cv2` | activate the venv from §1, or re-run `uv pip install -r reachy/requirements.txt` |
| `cosmos_bridge.py` requests pile up / latency climbs steadily | `interval_seconds` is shorter than the shim's round trip; the shim is single-flight (`deploy/05` §2.4) — raise the interval |
