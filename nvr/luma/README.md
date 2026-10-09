# VITURE Luma Ultra camera bridge

`luma_camera_bridge.py` serves the camera of the VITURE Luma Ultra XR glasses, plugged into the Orin's
USB-C port, to everything on the Jetson from one capture - the way `../reachy/reachy_mjpeg_bridge.py`
serves the Reachy Mini's. Live Vision relays it under `/luma/` and offers it as **Use VITURE Luma
Ultra** (`../README.md`, "VITURE Luma Ultra"). How the glasses connect, and what else they present, is
in [`../../deploy/11-viture-luma-ultra.md`](../../deploy/11-viture-luma-ultra.md).

| Route | |
|---|---|
| `/mjpeg` | multipart MJPEG for previews: the camera's own JPEG frames, not re-encoded |
| `/still.jpg` | the newest frame; 503, with the reason, when there is no fresh one (5 s) |
| `/healthz` | `state` (`idle`, `starting`, `live`, `no_camera`, `error`) and `reason`, plus `live`, `has_frame`, `frames`, `last_frame_age_s`, `device`, `size`, `fps`, `mjpeg_clients` |

- **Which camera.** The glasses' camera is a plain UVC webcam - a Sonix "USB 2.0 Camera", `0c45:636b` -
  behind the glasses' own USB hub, next to their `35ca` VITURE devices. The bridge picks the capture
  node of a video device whose hub also holds a `35ca` device, so another webcam on the Orin is never
  taken for it and `/dev/videoN` can move with plug order. `--device /dev/videoN` names one outright.
- **Only while asked.** The camera runs while something asks for it - a `/healthz` poll, an open
  `/mjpeg`, a `/still.jpg` - and stops `--idle` seconds (30) after the last request. A request after
  that starts it again: live in about 1.4 s.
- **Unplugged.** `/healthz` says `no_camera` and why, and the bridge finds the camera again when the
  glasses come back. A pipeline that stops delivering frames is restarted.
- **Needs** Python 3's standard library and `gst-launch-1.0` (`v4l2src`, `multipartmux`), which
  JetPack ships. It runs as `orin`, which is in the `video` group that owns `/dev/video*`.

## Install

On the Orin, from a copy of this repo:

```bash
install -D -m 0755 nvr/luma/luma_camera_bridge.py /home/orin/nvr/luma/luma_camera_bridge.py
sudo install -m 0644 nvr/systemd/luma-camera-bridge.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now luma-camera-bridge
curl -s http://127.0.0.1:8103/healthz; echo     # "starting" (this request woke it), then:
sleep 2; curl -s http://127.0.0.1:8103/healthz; echo   # "live": true
```

Then Live Vision needs the code that knows the source, and the flag:

```bash
install -m 0644 nvr/ui/scripts/serve_ui.py /home/orin/nvr/ui/scripts/serve_ui.py
install -m 0644 nvr/ui/web/app.js nvr/ui/web/index.html /home/orin/nvr/ui/web/
# The drop-in carries --luma-url http://127.0.0.1:8103: reinstall it as ../README.md says
# ("Robot control and speech: the drop-in", the sed pipeline), then check it took:
systemctl cat cosmos-edge-ui | grep -o -- '--luma-url [^ ]*'
sudo systemctl daemon-reload && sudo systemctl restart cosmos-edge-ui
```

The page shows **Use VITURE Luma Ultra** once it knows the server relays the bridge. `no_camera` means
the glasses are not on USB: see deploy/11, "When nothing appears".

Options: `--size` (an MJPEG size the camera offers, up to `1920x1080`; default `1280x720`), `--fps`
(5-30, default 15; Live Vision reads it from `/healthz` to pace its Live VLM sampling at every 30
frames), `--idle` (seconds, default 30), `--listen` and `--port` (default `127.0.0.1:8103`).

Tests, without the glasses: `python3 -m unittest discover -s nvr/luma/tests`.
