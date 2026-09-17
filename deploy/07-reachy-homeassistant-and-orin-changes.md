# 07 — Reachy Mini on the Orin: bridge, Home Assistant, and system changes

Everything in this document was applied to the running Orin. It exists because several of these
changes are **not** recoverable from the code alone: package pins, a disabled systemd unit, a
Docker runtime registration, and two third-party integrations patched in place. Re-flashing the
board without this list would leave a working-looking system that fails in non-obvious ways.

---

## 1. Reachy Mini camera → MJPEG bridge

`reachy-mjpeg-bridge.service` pulls the robot's WebRTC stream and re-serves it as MJPEG so
anything ffmpeg-based (Frigate, go2rtc) can read it.

| | |
|---|---|
| Code | [`../nvr/reachy/reachy_mjpeg_bridge.py`](../nvr/reachy/reachy_mjpeg_bridge.py) |
| Unit | [`../systemd/reachy-mjpeg-bridge.service`](../systemd/reachy-mjpeg-bridge.service) |
| Listens | `172.17.0.1:8099` — the Docker gateway |
| Cost | ~115 MB RSS, `MemoryMax=300M` |

**Why it exists.** The robot's camera is reachable only through the SDK's WebRTC or a local IPC
socket. ffmpeg can open neither, so Frigate cannot consume it directly.

**Why aiortc and not the Reachy SDK.** The SDK's WebRTC backend needs GStreamer's `webrtcsrc` from
`gst-plugins-rs`, which is not packaged for Linux and must be compiled from Rust source. aiortc is
pure Python. An earlier arrangement ran that GStreamer stack in a container on a *workstation* and
pushed frames to the Orin, which meant a desktop sat in the NVR's path and the camera died whenever
that machine slept.

**Why it binds `172.17.0.1` and not `127.0.0.1` or `0.0.0.0`.** Frigate runs in a bridged
container, so the container's `127.0.0.1` is not the host's. `host.docker.internal` resolves to
`172.17.0.1` via the `extra_hosts` entry the stack already uses. Binding the gateway rather than
`0.0.0.0` keeps the robot's camera off the LAN.

**Vendored dependency.** [`../nvr/reachy/reachy_stream/`](../nvr/reachy/reachy_stream/) is a copy of
`stream.py` and `const.py` from Pollen's Home Assistant integration, with a stub `__init__.py`. The
upstream package's `__init__` imports `homeassistant`, so it cannot be imported outside HA; those
two modules have no HA imports. Vendoring keeps Pollen's signalling logic rather than
reimplementing a WebRTC client, at the cost of needing a refresh when they change it.

---

## 2. Home Assistant integrations

Three were installed into `/home/orin/homeassistant-config/custom_components/`. **Two are patched
in place and HACS will overwrite those patches on update.**

### HACS 2.0.5
Installed host-side into the bind-mounted config directory rather than by running the official
script inside the container, which avoids depending on what tooling that Jetson HA image ships.
Still needs a GitHub device authorisation in the UI to be usable.

### Reachy Mini (`pollen-robotics/reachy_mini_homeassistant`) — PATCHED
Gives a camera entity, motor-mode selects, wake/sleep buttons, volume sliders and DoA sensors,
consuming the robot's WebRTC stream server-side.

* Patch: [`../nvr/homeassistant/patches/reachy_mini-manifest-aiortc.patch`](../nvr/homeassistant/patches/reachy_mini-manifest-aiortc.patch)
* **Also requires `pip install aiortc==1.10.1` inside the container.**

Upstream pins a bundled wheel, `aiortc 1.14.0+av17`, which requires `pyopenssl>=25.0.0`. Home
Assistant 2024.12 pins `pyopenssl==24.2.1`, so the requirement is unresolvable and the config flow
returns HTTP 500 before rendering. The pin is real, not conservative: aiortc 1.14 hands a
`cryptography` certificate to `use_certificate()`, and only pyopenssl ≥ 25 accepts it —

```
TypeError: cert must be an X509 instance
```

That failure appears only during the DTLS handshake, so importing the module and constructing an
`RTCPeerConnection` both succeed while the camera never produces a frame. Stock `aiortc 1.10.1`
matches what this HA actually has (`av 13.1.0`, `pyopenssl 24.2.1`) and works, verified by pulling
a real 1280×720 frame. Nothing in HA's own pins was changed.

### YoLink Local (`borenstein/yolink-local-ha`) — PATCHED
* Patches: [`yolocal-paho-mqtt-1.6-compat.patch`](../nvr/homeassistant/patches/yolocal-paho-mqtt-1.6-compat.patch),
  [`yolocal-manifest-paho.patch`](../nvr/homeassistant/patches/yolocal-manifest-paho.patch)

Same class of failure: it requires `paho-mqtt>=2.0.0` while HA 2024.12 pins `1.6.1`. The patch
selects the callback API at runtime and accepts both `on_disconnect` signatures, so it keeps
working if HA is ever upgraded. Forcing paho 2.x instead would break HA's own MQTT integration,
which carries the Frigate and ring-mqtt traffic.

**Known-unresolved:** the hub at `192.168.6.253` accepts the MQTT connection (`rc=0`) and then
refuses *every* subscription, including `$SYS/#`, disconnecting with `rc=7`. HTTP auth works and
`Home.getDeviceList` returns devices, so this is an authorisation or hub-side setting, not the
`net_id`. Worth raising upstream.

Also note: the HA container's pip is hard-wired to `https://pypi.jetson-ai-lab.dev/jp6/cu126`,
which carries no general packages. Any install needs `--index-url https://pypi.org/simple`.

---

## 3. System-level changes

These leave no trace in any repo file.

| Change | Why |
|---|---|
| `apt install nvidia-container-toolkit` + `nvidia-ctk runtime configure --runtime=docker` | Docker had no `nvidia` runtime, a prerequisite for any GPU container. Needed before Frigate can ever use a GPU detector. |
| `systemctl disable live-vlm-webui.service` | The fork's unit carries `Conflicts=live-vlm-webui.service`. With the original still enabled, it could stop the fork. |
| `systemctl enable live-vlm-webui-fork.service` | So the fork, not the original, survives a reboot. |
| `systemctl enable reachy-mjpeg-bridge.service` | The camera bridge should come back on boot. |

**Live VLM WebUI is deliberately kept off** much of the time: it serves Cosmos optimisation work,
not the NVR, and costs load. The command centre records that intent
(`/home/orin/nvr/feed/service_intent.json`) and shows a "stopped by user" badge, so neither a human
nor an automation silently restarts it. Nothing in the NVR path depends on it.

---

## 4. Frigate

The `reachy_mini` camera and its go2rtc stream are in
[`../nvr/frigate/config.yml`](../nvr/frigate/config.yml), shipped **disabled**. Toggle it from the
command centre (Camera power), which flips the flag over MQTT with no restart.

Two things to know:

* While on it costs roughly a core of detection plus an h264 transcode, on a board whose CPU
  detector already runs ~100 ms per inference. Enabling it took system load from ~3.8 to ~5.9.
* Frigate 0.18 persists camera enable/disable state **separately from this file and restores it
  over the config at startup**, so the value here only governs a fresh deployment.

### Why not a GPU detector
Frigate supports `rfdetr` and `dfine` model types, but this image has no TensorRT and its
onnxruntime offers only `['AzureExecutionProvider', 'CPUExecutionProvider']`. Jetson support ships
as a separate `-tensorrt-jp6` image, and this board runs **JetPack 7 / L4T R39.2.1**; upstream
PR #23584 adds a `-tensorrt-jp7` image but was deferred to Frigate 0.19. Selecting `onnx` today
would silently run on CPU; selecting `tensorrt` would fail outright.
