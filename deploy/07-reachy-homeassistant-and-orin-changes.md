# 07 — Reachy Mini on the Orin: bridge, Home Assistant, and system changes

Everything in this document was applied to the running Orin, except where a section is marked
*contract*: those describe components built alongside the bridge (Live Vision's Reachy source,
Frigate audio, the command centre's boot intent) by the interface they were built to, and say what
was and was not checked. The document exists because several of these changes are **not**
recoverable from the code alone: package pins, unit enablement, container restart policies, a
Docker runtime registration, and two third-party integrations patched in place. Re-flashing or
cloning the board without this list leaves a working-looking system that fails in non-obvious
ways. §8 is the checklist, and `scripts/reachy_smoke.py` checks it.

---

## 1. Reachy Mini camera and microphone bridge

`reachy-mjpeg-bridge.service` holds **one** WebRTC session to the robot and fans it out over plain
HTTP to everything on the Orin that wants the camera or the microphone: Frigate (through go2rtc),
the Live VLM WebUI, Live Vision, and the porch-feed command centre.

| | |
|---|---|
| Code | [`../nvr/reachy/reachy_mjpeg_bridge.py`](../nvr/reachy/reachy_mjpeg_bridge.py) |
| Unit | [`../systemd/reachy-mjpeg-bridge.service`](../systemd/reachy-mjpeg-bridge.service) |
| Listens | `127.0.0.1:8099` (this host) and `172.17.0.1:8099` (the Docker gateway, for Frigate). Never `0.0.0.0` |
| Robot side | One WebRTC session (signalling on `192.168.6.162:8443`), plus REST GETs on `:8000` |
| Runs in | `/home/orin/reachy_env`: aiortc 1.10.1, av 13.1.0, aiohttp 3.14.3 |
| Memory | `MemoryMax=300M`. The video-only predecessor ran at ~115 MB RSS; this version, with audio, has not been re-measured |

**Why it exists.** The robot's camera and microphone are reachable only through the SDK's WebRTC
or a local IPC socket. ffmpeg can open neither, and a browser page cannot be pointed at the robot
without opening a WebRTC session of its own.

| Endpoint | What | Read by |
|---|---|---|
| `GET /mjpeg` | `multipart/x-mixed-replace; boundary=frame`, 1280×720 JPEG, ~5 fps (`--fps 5`). A fresh frame at once, then only NEW frames | go2rtc's ffmpeg (Frigate), browser previews |
| `GET /still.jpg` | The newest frame. `503`, with the state and reason as text, when none is fresher than 5 s | porch-feed (preview, anomaly look), Live Vision (§5) |
| `GET /audio.mp3` | Endless `audio/mpeg`: the robot's microphone, 48 kHz stereo, 96 kb/s. Encoded only while someone listens | go2rtc (Frigate audio, §6), browsers |
| `GET /healthz` | State, and why (below) | porch-feed, `scripts/reachy_smoke.py` |
| `--push-url` | POSTs each new frame into the Live VLM WebUI's push API, session `reachy` | Live VLM WebUI (§5) |

`/healthz` says *why*, not just *what*, because the answer decides what a human should do:

| Field | Meaning |
|---|---|
| `state` | `starting` → `connecting` → `live`; `reconnecting` between sessions; `dormant` when the camera is unavailable on purpose; `error` if the supervisor itself failed |
| `reason` | Why it is not `live`. `camera held by the robot app 'x'` and `robot daemon unreachable` call for different responses |
| `live`, `has_frame` | Both vouch only for a frame under 5 s old. The command centre reads `has_frame` as "connected"; a picture from minutes ago is not a connection |
| `frames`, `last_frame_age_s` | Frames published, and the newest one's age (`stale_s` is the same number) |
| `sessions`, `restarts`, `failed_streak` | Robot sessions opened; sessions that ended; sessions **in a row** that never delivered video |
| `blocked_by` | The robot app holding the camera, when that is why it is dormant |
| `mjpeg_clients` | Open `/mjpeg` readers |
| `audio` | `{live, frames, last_frame_age_s, listeners}`: microphone frames arriving, and `/audio.mp3` listeners |
| `push` | `{enabled, state, pushed, detail}`: the WebUI push loop (§5) |

**Why exactly one session.** Daemon 1.11 gives every WebRTC consumer its own hardware H.264
encoder and its own TURN allocations. Frigate, two Live UIs and the command centre each opening a
session would cost the robot four encoders for one camera, and every session also leaks (below).
So the bridge holds one session whether or not anyone downstream is watching: the robot encodes for
it continuously, which is the price of one stable session instead of churn. The official Reachy
app is a separate WebRTC consumer and coexists with it (§7).

**Why the answer is H.264 without RTX.** The robot's offer (daemon 1.11.0, kept as
[`../nvr/tests/fixtures/reachy_offer_daemon_1.11.0.sdp`](../nvr/tests/fixtures/reachy_offer_daemon_1.11.0.sdp))
carries H.264 with `red`, `ulpfec` and two `rtx` payload types, plus Opus. aiortc 1.10.1, and
still 1.14.0, mishandles an RTX packet: after unwrapping it, `_handle_rtp_packet` keeps the RTX
codec, so a frame completed by a retransmission is queued to the decoder as `video/rtx`. The
decoder thread dies with `No decoder found for MIME type video/rtx` and the track goes silent
seconds into every session, while the signalling still looks healthy. So the bridge adds its
transceivers *before* applying the offer (video restricted to H.264 with `setCodecPreferences`,
plus a recvonly audio transceiver), because aiortc negotiates codecs inside `setRemoteDescription`
against the preferences of whichever transceiver claims each m-line. RTX is never in the answer,
so the robot never sends it; losses are recovered by the keyframe requests aiortc already sends.
Upgrading aiortc is not a fix: 1.14.0 has the same bug.

**The daemon's socket leak, and why retries are slow.** Daemon 1.11 puts TURN relays on every
consumer, LAN ones included, and libnice leaves their refresh sockets open when a session ends
(`We still have alive TURN refreshes` in the robot's log). One failed session every ~26 s overnight
exhausted the daemon's file descriptors (`Too many open files`), after which it rejected every
consumer, the official app included, until the robot was power-cycled. A retry that cannot succeed
must not be allowed to use up the robot, so the pacing is set for the robot's sake:

| | |
|---|---|
| First retry | 3 s, doubling after each session that ends |
| Cap | **300 s**: about a dozen attempts an hour, against the ~140 that exhausted it |
| Reset | Once a session has stayed live for 60 s |
| Dead session | A live session with no video for 8 s, whatever the signalling says. A new one gets 20 s to its first frame |
| `failed_streak` | Sessions in a row with no video. At 4, `reason` turns into advice: `N sessions in a row without video (last: …). If this persists, restart the robot: its daemon can run out of file descriptors.` |

A `failed_streak` that keeps climbing while the daemon still answers REST is this failure:
power-cycle the robot. Restarting the bridge does not help; the daemon is refusing everyone, and
each restart is one more session.

**Dormant: the camera is unavailable on purpose.** Before every session the bridge asks the
daemon's REST API, not WebRTC, whether a session could succeed. From the WebRTC side "no frames"
looks the same whatever the cause, and every session into a camera the bridge cannot have is one
more encoder spun up and one more leak.

| Daemon answer | `reason` |
|---|---|
| `/api/daemon/status` unreachable, or `state` not `running` | `robot daemon unreachable (…)` / `robot daemon is 'x', not running` |
| `/api/media/status` has `no_media`, `released`, or `available: false` | `daemon runs without media` / `daemon has released the camera and microphone` / `daemon reports the camera unavailable` |
| `/api/daemon/robot-app-lock-status` has `state: local_app` | `camera held by the robot app 'x'`, with `blocked_by` set |
| The lock was held and is now free, for under 30 s | `robot app 'x' just released the camera; waiting Ns in case it is restarting`: an app restart frees the lock for seconds, and dialling into that gap once reached an exhausted daemon and stopped the app's camera as it came back |

Any of these: `state: dormant`, a REST poll every 10 s, no session until the answer changes, and
then it resumes by itself. None of it is an error. A camera taken *mid*-session shows up first as
a stall or as the robot ending the session; the check before the next session then finds the
reason. Daemons without the media or lock endpoints are tolerated, and the session attempt decides.

**Stopping: SIGTERM ends the session.** systemd stops and restarts with SIGTERM, whose default
action kills Python on the spot. The robot then keeps encoding for a consumer that no longer exists
until ICE gives up on it, and a second session opened in that window can get audio but no video.
The bridge turns SIGTERM and SIGINT into cancellation, which unwinds the session and sends the
robot an `endSession`; the unit allows `TimeoutStopSec=10` for it.

**Audio.** The robot's session carries Opus alongside the video, and the track has to be read even
when no one is listening: aiortc queues every decoded frame for a reader that never comes, and the
process grows until it is killed. So the bridge always drains it and encodes only on demand:

* **MP3** (libmp3lame, 48 kHz stereo, 96 kb/s), because it is the one format every consumer here
  reads as an endless HTTP stream: ffmpeg (go2rtc, Frigate) and a browser `<audio>` alike.
* The encoder is created with the first `/audio.mp3` listener and dropped with the last, so an
  unheard microphone costs only the draining.
* Each listener gets a queue of 64 MP3 frames (~1.5 s). A slow listener loses its oldest audio
  rather than stalling the others.
* An audio error is logged and never takes the video down.

**Why it binds `127.0.0.1` and `172.17.0.1`, and never `0.0.0.0`.** Loopback serves this host.
Frigate runs in a bridged container, whose `127.0.0.1` is not the host's; `host.docker.internal`
resolves to `172.17.0.1` via the `extra_hosts` entry the stack already uses. Not binding `0.0.0.0`
means nothing on the LAN talks to the bridge directly: the camera and microphone reach LAN browsers
only through the services that re-serve them, each with its own rules, listed under "What reaches
the LAN" below. `172.17.0.1` exists only once dockerd is up,
and at boot the bridge can start first, so it waits for the address (retrying every 5 s) rather
than crash-looping. The unit orders itself `After=docker.service` without `Wants=`, so it cannot
drag docker in. Any other bind failure (a port conflict) exits, and `Restart=always` brings it
back.

**What reaches the LAN.** Watching and listening from another machine is the point of the Live
UIs, so the robot is not kept off the LAN; each route that re-serves it has its own rule:

| Route | Video | Microphone | Rule |
|---|---|---|---|
| Live Vision `:8092` / `:8443`, `/reachy/*` (§5) | yes | yes, when Listen is on | Per-start token handed only to its own page, plus a `Host` check: keeps other web pages and DNS rebinding out, not a determined LAN user |
| Live VLM WebUI `:8090`, session `reachy` (§5) | yes (pushed frames, `/api/push/latest.jpg`) | no | None: the WebUI has no authentication |
| Frigate `:5000` (§6) | yes, `reachy_mini` | only as the "Video + mic" live choice | None: Frigate's `:5000` is unauthenticated |
| Command centre `:8096` | the preview still | no | Same as its other read-only routes |

**Why aiortc and not the Reachy SDK.** The SDK's WebRTC backend needs GStreamer's `webrtcsrc` from
`gst-plugins-rs`, which is not packaged for Linux and must be compiled from Rust source. aiortc is
pure Python. An earlier arrangement ran that GStreamer stack in a container on a *workstation* and
pushed frames to the Orin, which meant a desktop sat in the NVR's path and the camera died whenever
that machine slept.

**Vendored dependency.** [`../nvr/reachy/reachy_stream/`](../nvr/reachy/reachy_stream/) is a copy of
`stream.py` and `const.py` from Pollen's Home Assistant integration, with a stub `__init__.py`. The
upstream package's `__init__` imports `homeassistant`, so it cannot be imported outside HA; those
two modules have no HA imports. The bridge subclasses its `ReachyMiniStreamClient` (adding frame
timestamps, the audio track and the H.264-only answer) rather than reimplementing a WebRTC client,
at the cost of needing a refresh when Pollen change their signalling.

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
| `systemctl enable live-vlm-webui-fork.service` | So the fork, not the original, comes back after a reboot. Its boot state now follows the command centre's recorded intent (§8). |
| `systemctl enable reachy-mjpeg-bridge.service` | The camera bridge should come back on boot. Lost on the SD-card clone (§8). |
| `systemctl enable docker.service containerd.service porch-feed.service` | Found disabled on the SD-card clone (§8). Without docker, nothing in `docker-compose.yml` starts and `172.17.0.1` never appears, so the bridge cannot serve Frigate. |
| `docker update --restart unless-stopped frigate mosquitto ring-mqtt` | The clone had them on `restart=no` (§8). `docker-compose.yml` sets `unless-stopped`; a container created or restored any other way does not inherit it. |
| `nvr/ui/scripts/enable_lan_ui.sh` → `/etc/systemd/system/cosmos-edge-ui.service.d/lan.conf` | Live Vision on `0.0.0.0:8092` (HTTP; LAN requests for `/` are redirected) and `:8443` (HTTPS, self-signed cert in `/home/orin/nvr/ui/deployment/tls/`). Browsers grant cameras only in a secure context. The unit in the repo binds loopback only; this drop-in is the deliberate exception. |

**Live VLM WebUI is deliberately kept off** much of the time: it serves Cosmos optimisation work,
not the NVR, and costs load. The command centre records that intent
(`/home/orin/nvr/feed/service_intent.json`) and shows a "stopped by user" badge, so neither a human
nor an automation silently restarts it. Nothing in the NVR path depends on it: when it is off, the
bridge notes it in `/healthz` and keeps serving everything else (§5).

---

## 4. Frigate

The `reachy_mini` camera and its go2rtc stream are in
[`../nvr/frigate/config.yml`](../nvr/frigate/config.yml), shipped **disabled**. Toggle it from the
command centre (Camera power), which flips the flag over MQTT with no restart.

Two things to know:

* While on it costs roughly a core of detection plus an h264 transcode, on a board whose CPU
  detector already runs ~100 ms per inference. Enabling it took system load from ~3.8 to ~5.9.
* Frigate 0.18 persists camera enable/disable state **separately from this file and restores it
  over the config at startup**, so the value here only governs a fresh deployment. Its
  `/api/config` shows both: `enabled` is the running state, `enabled_in_config` the file's. On
  2026-09-24 `reachy_mini` was disabled at runtime while the live config file said `enabled: true`;
  `scripts/reachy_smoke.py` reports exactly that and changes neither.

The stream Frigate reads is the bridge's `/mjpeg`; §6 adds the microphone.

### Why not a GPU detector
Frigate supports `rfdetr` and `dfine` model types, but this image has no TensorRT and its
onnxruntime offers only `['AzureExecutionProvider', 'CPUExecutionProvider']`. Jetson support ships
as a separate `-tensorrt-jp6` image, and this board runs **JetPack 7 / L4T R39.2.1**; upstream
PR #23584 adds a `-tensorrt-jp7` image but was deferred to Frigate 0.19. Selecting `onnx` today
would silently run on CPU; selecting `tensorrt` would fail outright.

---

## 5. Feeding the Live UIs

Both Live UIs get the robot through the bridge. Neither opens a session of its own (§1).

### Live VLM WebUI: pushed into its `reachy` session

The unit's `--push-url https://127.0.0.1:8090/api/push/frame?session_id=reachy&source_name=reachy-mini`
makes the bridge POST each new JPEG into the push API of the WebUI (the nv-asotelo fork, not in this
repo). It is watched at `https://<orin>:8090/?session=reachy`, the command centre's "Reachy Mini
feed" link. TLS is not verified on that hop: the WebUI's certificate is self-signed, and the hop is
loopback.

The WebUI is off much of the time on purpose (§3), so its absence is a state in `/healthz`
`push.state`, not an error:

| `push.state` | Meaning | What the bridge does |
|---|---|---|
| `pushing` | Frames accepted | One POST per new frame |
| `waiting for video` | No fresh frame: the bridge is not live | Waits for one |
| `WebUI not reachable (stopped?)` | Refused or timed out | Probes every 10 s |
| `stopped in the WebUI; press Start there to resume` | HTTP 409: someone pressed Stop on the `reachy` session | Waits, retrying every 5 s. Re-creating the session behind their back would silently keep the VLM running |
| `WebUI answered HTTP n` | Anything else; the body is in `push.detail` | Retries every 5 s |

The WebUI's side of it: `GET https://127.0.0.1:8090/api/push/status` lists `streams[]` with
`session_id`, `connected` and `frames_received`. Healthy is `reachy` connected with
`frames_received` rising.

Checked 2026-09-24 with real robot frames while the robot's WebRTC was down (§7): 40 captures from
the robot's own camera, taken through the Reachy testbench app, pushed into the `reachy` session.
The WebUI sent frame 30 to the shim on the Fast (v2) engine and answered "A Jetson Orin Nano
Developer Kit box sits on a desk next to a computer and a Charmander figurine" in 1.9 s. The same
kind of frame through Live Vision (`reachy_smoke.py --image`): first text 710 ms, done 973 ms.

### Live Vision: the Reachy source

Live Vision (`nvr/ui`, `cosmos-edge-ui.service`; `:8092` HTTP and `:8443` HTTPS with the §3
drop-in) captures from the browser with `getUserMedia`. Its Reachy source, built alongside this
bridge, is specified as:

* `serve_ui.py` proxies `/reachy/<path>` to the bridge's `/<path>` on `127.0.0.1:8099`; for
  example `/reachy/healthz` answers with the bridge's `/healthz`. The proxy is not a convenience.
  The page's Content-Security-Policy allows only `'self'` (plus `blob:` and `data:`) for
  `connect-src`, `img-src` and `media-src`, and the bridge listens on loopback and the Docker
  gateway, never the LAN, so Live Vision's own origin is the only way a browser can reach the
  robot's frames.
* Every `/reachy/` route needs `?token=`, a secret `serve_ui.py` makes at each start and hands
  out only in its `/api/access` answer, which only a same-origin page can read; without it the
  answer is `401`. `/reachy/` and `/api/access` also answer only to a `Host` that is an IP
  address, `localhost`, or a name given with `--allowed-host`; anything else gets `421`, which
  is what a DNS-rebound name would send. Like the command centre's control token, this keeps
  other web pages off the robot's camera and microphone, not a determined LAN attacker.
* `/reachy/healthz` carries `X-Reachy-Slots: <open>/<max>`, the streams open against the
  server-wide cap of 4.
* `?source=reachy` on the page selects the robot instead of a local camera.

Whatever the source, inference goes through the same `/v1/chat/completions` rules
(`validate_request` in `serve_ui.py`): `stream: true`; exactly one user message holding one text
part and one `data:image/jpeg;base64,` image; temperature 0 or 0.7; `max_tokens` 1–512; no `seed`;
a body of at most 2 MiB; no cross-origin `Origin`. One generation runs at a time and a second gets
`429`.

Deployed 2026-09-24 and checked from the Orin against `https://192.168.6.252:8443`: `/reachy/healthz`
with no token or a wrong one answered `401`, as did `/reachy/audio.mp3`; `/api/access` with
`Host: evil.example` answered `421`; with the page's token `/reachy/healthz` relayed the bridge's
state; plain HTTP `/` from a LAN host still redirected to HTTPS; the token never appeared in the
journal. The inference path was checked the same day with a Frigate snapshot instead of a robot
frame: first text in 176 ms, 16 tokens complete in 438 ms (428 ms native), on the Fast (v2)
engine. A live robot frame through this path waits on the robot (§7, descriptor exhaustion).

---

## 6. Frigate audio

Frigate's `reachy_mini` stays **video only**, read from the bridge's `/mjpeg` (§4). The microphone
is a separate go2rtc stream, `reachy_mini_mic`, offered as a choice in the camera's live view:

```yaml
go2rtc:
  streams:
    reachy_mini:
      - "ffmpeg:http://host.docker.internal:8099/mjpeg#video=h264"
    reachy_mini_mic:
      - "rtsp://127.0.0.1:8554/reachy_mini"        # the same picture, not encoded twice
      - "ffmpeg:http://host.docker.internal:8099/audio.mp3#audio=opus"
cameras:
  reachy_mini:
    live:
      streams:
        Video: reachy_mini
        Video + mic: reachy_mini_mic
```

Why not a second source on `reachy_mini` itself: Frigate 0.18's live player asks for audio on every
view, muted or not, and so does its stream-metadata probe. A microphone source on `reachy_mini`
would start an MP3-to-Opus ffmpeg, and the bridge's MP3 encoder, whenever anyone looked at the
camera. As its own stream it runs only when "Video + mic" is picked, and detection never touches it.

Applied to the live config on 2026-09-24 after `--validate-config` passed on a candidate copy
(`docker exec -e CONFIG_FILE=/config/config.yml.reachy-candidate frigate python3 -m frigate
--validate-config`); the previous file is kept beside it as `config.yml.bak-pre-reachy-mic-*`. Every
camera's runtime enabled state read the same before and after the restart.

---

## 7. Swapping the camera and mic between systems

The robot has one camera and one microphone, and several systems want them. What makes swapping
safe is that on the Orin **the bridge should be the only thing that talks WebRTC to the robot**.
Everything else on the Orin reads the bridge, so taking the camera away is one event in one place,
and giving it back needs nothing restarted. The exception is Home Assistant's Reachy Mini camera
entity (row below), which opens sessions of its own.

| System | How it gets the camera and mic | Cost to the robot | What happens to the others | Recovery |
|---|---|---|---|---|
| Frigate (`reachy_mini` on) | go2rtc reads the bridge's `/mjpeg`; `/audio.mp3` only while "Video + mic" is being watched (§6) | Nothing beyond the bridge's session | Nothing: one more reader of the bridge | Turning the camera off stops go2rtc's reads; the bridge's session carries on for everyone else |
| Live VLM WebUI | The bridge pushes into its `reachy` session (§5) | Nothing | Nothing | WebUI stopped: the bridge probes every 10 s and resumes when frames are accepted. Stop pressed on the session: press Start there |
| Live Vision, Reachy source (*contract*, §5) | Browser → Live Vision `/reachy/*` → bridge | Nothing | Nothing | Shows what the bridge has; when it has nothing, the bridge's state says why |
| Official Reachy app | Its own WebRTC session | Its own H.264 encoder and TURN allocations, and every session adds to the daemon's socket leak (§1) | Nothing: it coexists with the bridge | Nothing to recover, unless the daemon runs out of descriptors (below) |
| Home Assistant's Reachy Mini camera (§2) | Its own WebRTC session, server-side, through the same Pollen client the bridge vendors | Worse than the official app: that client tears its session down after 10 s idle and opens a new one on the next image request, so a dashboard left open churns sessions into the daemon's socket leak (§1) | Nothing directly | Keep `camera.reachy_mini_12d8_camera` disabled, or point HA at the bridge's `/still.jpg` / `/mjpeg` with a generic or MJPEG camera instead. HA's container was stopped on 2026-09-24 |
| An on-robot app that takes the camera | Daemon app lock, `state: local_app` | — | The bridge's session ends or stalls (8 s) and it goes `dormant`, `blocked_by` naming the app. Everything on the Orin loses video and audio: `/still.jpg` 503, command centre "no frames", WebUI push `waiting for video`, Frigate's `reachy_mini` input idle | Automatic. The bridge polls REST every 10 s and opens a session once the lock clears |
| Daemon media release: an SDK client with `media_backend="no_media"`, such as [`../reachy/release_camera_for_webui.py`](../reachy/release_camera_for_webui.py) | The client holds the camera and microphone devices | — | As above, with `reason: daemon has released the camera and microphone` | Automatic, once the client exits and the daemon takes the media back |

### Forced recovery from porch-dad

The daemon runs its camera, WebRTC and REST in one process, and when that process runs out of file
descriptors nothing it offers can fix it: `/api/daemon/restart` and a media release both stay in
the same process. Measured 2026-09-24: 1014 of the default 1024 descriptors in use, 778 of them
sockets. So porch-dad restarts the process itself, over SSH, with a key that can do nothing else.

[`../nvr/reachy/setup_robot_recovery.sh`](../nvr/reachy/setup_robot_recovery.sh), run once on the
Orin (it asks for `pollen`'s password, `root` on the stock image, and stores nothing):

| Where | What |
|---|---|
| Orin | `/home/orin/.ssh/reachy_recover_ed25519`, and the robot's host key in `known_hosts` |
| Robot `~pollen/.ssh/authorized_keys` | That key, with `command="sudo -n /usr/bin/systemctl restart reachy-mini-daemon.service",no-pty,no-port-forwarding,no-X11-forwarding,no-agent-forwarding`: whatever a client asks for, the key runs that and only that |
| Robot `/etc/systemd/system/reachy-mini-daemon.service.d/porch-dad-nofile.conf` | `LimitNOFILE=16384`: the leak takes 16x as long to wedge the daemon. Applies from the next daemon restart |

The bridge unit passes `--recover-ssh pollen@192.168.6.162 --recover-key <that key>`; `/healthz`
reports `recovery: {configured, attempts, last_result}`. Both robot-side files survive daemon
updates (they are outside its venv) but not a reinstall of the robot's OS: run the script again.

Verified 2026-09-24/25 on the device: the bridge's own recovery call restarted the daemon (exit 0,
7.6 s); its REST API answered 10 s later; the running bridge was live again ~9 s after that, pushing
to the WebUI, without being touched; the new process runs with `Max open files 16384` and held 157
descriptors. The first recovery that day was not this: the wedged daemon restarted by itself
(most likely a crash and systemd's restart), and the bridge picked the camera up on its own.

When the robot itself is the problem:

| Situation | Bridge | Recovery |
|---|---|---|
| Robot off or rebooting, daemon not `running` | `dormant`: `robot daemon unreachable (…)` or `robot daemon is 'x', not running` | Automatic, within one 10 s poll of the daemon answering `running` |
| Daemon out of file descriptors | After the first session without video it reads the daemon's journal, finds `Too many open files`, and stops dialling | **Automatic.** It restarts the daemon over SSH (below) at most once per 15 min, then resumes as soon as the daemon is a new process. Without the key, or if the restart fails, it rebuilds the robot's camera pipeline for its local apps and waits for a power-cycle, saying so in `reason` |

The bridge rows follow from its code (§1). The app-lock, media-release and descriptor rows were not
staged against the running bridge for this document; the descriptor row is the failure seen under
~26 s retries (§1). While the bridge is not live, open `/mjpeg` connections stay open and idle
rather than being closed. How quickly Frigate's own input picks up again once frames return is
go2rtc's and Frigate's reconnect logic, not measured here.

---

## 8. Boot and bootable-clone checklist

The Orin boots from an SD-card clone, and a clone carries files, not guarantees. On this one
**docker, containerd, reachy-mjpeg-bridge, porch-feed and live-vlm-webui-fork were not enabled at
boot, and frigate, mosquitto and ring-mqtt were on `restart=no`**, although
[`../nvr/README.md`](../nvr/README.md) ("What survives a reboot") records the shim, WebUI,
porch-feed and frigate-notify as `boot=enabled` and the containers as `restart=unless-stopped`.
That table was verified before a planned restart of the original board; the clone did not inherit
it. While everything was running nothing looked wrong. The next reboot would have come back without
Frigate, MQTT, ring-mqtt, the camera bridge or the command centre. How the clone
lost these states was not established, which is why the check is a script rather than a note.

| What | Must be | Why |
|---|---|---|
| `docker.service`, `containerd.service` | enabled | Everything in `docker-compose.yml`, and the `172.17.0.1` address the bridge serves Frigate on |
| `frigate`, `mosquitto`, `ring-mqtt` | `restart=unless-stopped` | With `restart=no` they stay down after a reboot even with docker up |
| `cosmos3-edge-shim.service` | enabled | The model: every caption and both Live UIs |
| `reachy-mjpeg-bridge.service` | enabled | The robot's camera and microphone, for everything else |
| `porch-feed.service` | enabled | The command centre, including the Reachy preview and anomaly watch |
| `cosmos-edge-ui.service`, `live-vlm-webui-fork.service` | **follows intent** | Below |

If a clone comes up without them:

```bash
sudo systemctl enable docker.service containerd.service cosmos3-edge-shim.service \
  reachy-mjpeg-bridge.service porch-feed.service
sudo docker update --restart unless-stopped frigate mosquitto ring-mqtt
```

On 2026-09-24 the device had been put right: the smoke check reported every row above as enabled
or `unless-stopped`. It was then rebooted cold from the SD clone to prove it. SSH was back in
91 s; the shim answered 25 s later on the default Fast (v2) engine; the bridge bound both addresses
and went straight to its correct state (dormant, a robot app held the camera); the command centre,
both Live UIs, Frigate, MQTT and ring-mqtt all came back unattended; every Frigate camera's runtime
state was the same as before; and the smoke check read exactly as it had before the reboot
(16 PASS, 0 FAIL). `porch-dad.service`, running but deliberately not enabled, stayed down as its
boot state says and was started again by hand.

**Boot follows intent for the two Live UIs.** Both are switched off on purpose at times (§3), so
"always enabled" would be as wrong as "always disabled". The command centre records every start and
stop in `service_intent.json`, under the service keys `livevision` and `vlm`, and both carry
`boot_follows_intent: true` in its config: a verified ON also runs `systemctl enable`, a verified
OFF runs `disable` (also when the stop worked but something still holds the port), and RESTART is
disabled while a service is stopped so it cannot quietly turn a deliberately stopped UI back on at
boot. Each such service card shows "starts at boot" or "off at boot", and says so when that
disagrees with the recorded intent. The smoke check reports `is-enabled` beside the recorded
intent too. On 2026-09-24 both were `enabled` with a recorded intent of `running`.

`homeassistant` is deliberately not in the table: it is not in `docker-compose.yml`, it was stopped
with `restart=no` when this was written, and while its Reachy camera entity is enabled it is a
second WebRTC client of the robot (§7). The smoke check reports it without ever failing on it.

### After any clone boot: `scripts/reachy_smoke.py`

[`../scripts/reachy_smoke.py`](../scripts/reachy_smoke.py) is one stdlib-only file, so it runs on a
fresh clone with nothing installed. Run it with `/home/orin/reachy_env/bin/python` to add the
microphone level (PyAV); plain `python3` does everything else.

```bash
scp scripts/reachy_smoke.py orin@192.168.6.252:/home/orin/nvr/reachy/
ssh orin@192.168.6.252 /home/orin/reachy_env/bin/python /home/orin/nvr/reachy/reachy_smoke.py
```

It changes nothing and never opens a WebRTC session to the robot (§1 is why); the robot only sees
REST GETs. Its cost is a few seconds of bridge streams (the MP3 encoder runs while it listens) and
one 16-token inference. Exit status is 1 on any FAIL. `--json` for scripts, `--no-inference` to
skip the inference, `--image FILE` to test inference without the robot.

| Check | PASS | Otherwise |
|---|---|---|
| `robot daemon`, `robot media`, `robot app lock` | Daemon `running`, media with the daemon, no app lock | Unreachable FAIL. Media released or app lock WARN: the bridge is dormant on purpose |
| `bridge 127.0.0.1`, `bridge 172.17.0.1` | `/healthz` live, with microphone audio | `dormant` or a first connect WARN; anything else FAIL, with `reason` and `failed_streak`. Only the gateway failing means docker is not up |
| `bridge /still.jpg` | A JPEG, dimensions read from its SOF header | |
| `bridge /mjpeg` | ≥ 3 JPEG parts in 3 s | |
| `bridge /audio.mp3` | ≥ 8 KB of chained MPEG audio frames in 3 s. With PyAV, decoded and its RMS level reported | Below −70 dBFS WARN. That catches a dead or muted capture path, not a quiet room |
| `webui push (reachy)` | `reachy` connected, `frames_received` rising | SKIP when the WebUI is off. WARN when Stop was pressed there |
| `live vision /reachy` | `/reachy/healthz` relays the bridge's `/healthz` | SKIP when Live Vision is off. 404 FAIL: a build without the Reachy source |
| `live vision inference` | One streamed request, shaped to `validate_request`, returns SSE text | SKIP when Live Vision is off or there is no frame. 429 WARN, not retried |
| `shim`, `shim engine` | `/v1/models` and `/health/ready` answer. `/opt/tensorrt-edgellm/models/default` resolves to the engine whose notes in `nvr/feed/config.yaml` call it "the default choice" (v2, Fast) | Still loading WARN. Another engine WARN |
| `frigate reachy_mini` | go2rtc lists the stream. Reports its bridge sources and the camera's runtime and config-file state | Never changes the camera |
| `boot …`, `restart …` | The table above | |

SKIP means off on purpose, or not judgeable because something upstream already failed: one root
cause reads as one FAIL, not ten. The run of 2026-09-24, with the bridge stopped and the deployed
Live Vision predating its Reachy source (`--image` supplied a Frigate snapshot; caption elided):

```
CHECK                     STATUS  DETAIL
robot daemon              PASS    state 'running', version 1.11.0
robot media               PASS    camera and microphone with the daemon
robot app lock            PASS    no on-robot app holds the camera (state 'free')
bridge 127.0.0.1          FAIL    :8099/healthz connection refused - is reachy-mjpeg-bridge.service running?
bridge 172.17.0.1         FAIL    :8099/healthz connection refused - same as above
bridge /still.jpg         SKIP    no fresh frame to serve: the bridge is not answering
bridge /mjpeg             SKIP    the bridge is not answering
bridge /audio.mp3         SKIP    the bridge has no microphone audio: the bridge is not answering
webui push (reachy)       SKIP    the WebUI is up, but the bridge is not answering
live vision /reachy       FAIL    Live Vision answers, but has no /reachy/ proxy: a build without the Reachy source
live vision inference     PASS    '…' - first text 176 ms, all 438 ms (native 428 ms), image from --image
shim                      PASS    answers, model nvidia/Cosmos3-Edge, ready
shim engine               PASS    -> Fast (v2), the config's default
frigate reachy_mini       PASS    go2rtc stream listed, bridge sources: /mjpeg (no audio source: Frigate gets no Reachy audio); camera disabled at runtime (config file says enabled) (reported, not changed)
boot docker               PASS    enabled
boot containerd           PASS    enabled
boot cosmos3-edge-shim    PASS    enabled
boot reachy-mjpeg-bridge  PASS    enabled
boot porch-feed           PASS    enabled
boot cosmos-edge-ui       PASS    enabled, follows intent (last started from the command centre)
boot live-vlm-webui-fork  PASS    enabled, follows intent (last started from the command centre)
restart frigate           PASS    restart=unless-stopped, running
restart mosquitto         PASS    restart=unless-stopped, running
restart ring-mqtt         PASS    restart=unless-stopped, running

24 checks: 17 PASS, 0 WARN, 3 FAIL, 4 SKIP
```

Its tests, [`../nvr/tests/test_reachy_smoke.py`](../nvr/tests/test_reachy_smoke.py), build JPEG,
MPEG audio and SSE bytes from their specs, run the checks against fake servers on loopback, and run
the inference check through the real `serve_ui.py`. On the Orin with `reachy_env` two more run:
tones pushed through the bridge's own `AudioHub` come back as valid MP3 at the right level.

```bash
python3 -m unittest discover -s nvr/tests -p 'test_reachy_smoke.py'
```
