# porch-dad — Frigate NVR + Ring + Cosmos3-Edge scene descriptions

A home security stack on a single **Jetson Orin Nano Super (8 GB)**: Frigate does recording and
object detection, ring-mqtt brings Ring cameras in as RTSP, and every completed event is summarised
in one sentence by the locally-hosted **Cosmos3-Edge** VLM. Results land on a phone-friendly web feed.

```
Ring cameras ──ring-mqtt──▶ RTSP :8554 ──▶ Frigate ──▶ MQTT frigate/events
                                              │                   │
                                         clips + API              ▼
                                              └────────▶ porch-dad bridge
                                                          │  3 sampled frames
                                                          ▼
                                                    Cosmos3-Edge shim :8000
                                                          │
                                                          ▼
                                            SQLite + MQTT + web feed :8095
```

## Services and ports

| Service | Port | What it is |
|---|---|---|
| porch-dad feed | **8095** | The mobile web UI. This is the one you open on your phone. |
| Frigate | 5000 | NVR UI + API (the bridge pulls event clips from here) |
| ring-mqtt setup | 55123 | **Ring account authentication happens here** |
| ring-mqtt RTSP | 8554 | Ring camera streams, consumed by Frigate |
| Mosquitto | 1883 | MQTT bus shared by all three |
| Cosmos3-Edge shim | 8000 | OpenAI-compatible endpoint serving the INT4 model |

## Ring authentication

Ring has no official API, so ring-mqtt performs an interactive login and stores a long-lived
**refresh token**. Do this yourself — the credentials go from your browser straight to ring-mqtt and
are never stored in this repo.

1. Open **`http://<jetson-ip>:55123/`** from any device on your LAN.
2. Enter your Ring **email and password**.
3. Enter the **2FA code** Ring sends (SMS or email). 2FA is mandatory on Ring accounts.
4. ring-mqtt writes the token to `ring-mqtt/data/ring-state.json`, which is bind-mounted so it
   survives container restarts.

Verify it worked:

```bash
docker logs ring-mqtt | grep -iE 'located|found|device'
mosquitto_sub -h <jetson-ip> -t 'ring/#' -C 5      # should show your devices
```

**Use a dedicated Ring account.** Create a Shared User in the Ring app with access only to the
cameras you want monitored, and authenticate with that. The token is long-lived and sits on the
device, so scoping it limits what a compromise reaches.

**When the token stops working** — after a password change or a session revoke — ring-mqtt logs an
authentication failure. Redo the same web-UI flow; nothing else needs touching.

## How Ring cameras are consumed (and why not through Frigate)

**Ring's live stream cannot be used as a continuous NVR source, and no subscription changes that.**

ring-mqtt exposes each camera at `ring-mqtt:8554/<device_id>_live`, but that is an on-demand
**WebRTC** session transcoded to RTSP, not a persistent feed like an ONVIF camera. Pointed at
Frigate it fails:

```
ffmpeg.front_door.detect ERROR : rtsp://ring-mqtt:8554/90486cee0ffc_live:
                                 Invalid data found when processing input
```

…even while ring-mqtt reports the session healthy (`WebRTC session is connected`, `new consumer
stream=..._live`). ffprobe with a 20s analyze window fails identically. Two further problems make
this the wrong shape regardless of the transport bug:

- **Battery drain.** Frigate's continuous detect holds the stream open permanently. A
  battery-powered Ring camera lasts hours under that, not months.
- **No paid tier fixes it.** Ring's 24/7 Continuous Recording records to *Ring's cloud* for the Ring
  app. It does not expose a persistent local RTSP stream, and it requires wired power anyway. Do not
  buy a subscription expecting it to make this work.

### The event-driven path (what this repo actually does)

Ring is event-driven by design, so the bridge consumes what Ring naturally emits:

```
Ring motion/ding ──MQTT──▶ porch-dad ──▶ snapshot(s) from ring/+/camera/+/snapshot/image
                                          └──▶ Cosmos3-Edge ──▶ feed
```

This works for **every** camera, wired or battery, adds no continuous load, and needs no live
stream. Configure it under `ring_*` in `bridge/config.yaml`; map device IDs to friendly names in
`ring_cameras` so `{camera}` renders readably in the prompt.

Verified against real cameras and a real person walking past:

```
[ROUTINE] Front Driveway · motion · 1f · 902ms
  [03:57] [ROUTINE] A silver SUV is parked under a covered driveway. A person wearing a black
  shirt and shorts is standing next to the vehicle, holding a device, possibly a phone.

[ROUTINE] Office · motion · 1f · 717ms
  [04:55:17] Human male, bald, wearing black t-shirt, black shorts, black flip-flops,
  walking on a porch.
```

**Known limitation: one frame per Ring event.** Unlike the Frigate path, which samples 3 frames from
a clip, Ring publishes a single snapshot per motion event, so descriptions capture one moment rather
than a sequence. The bridge degrades gracefully (1..3 frames) rather than discarding the event, and
one snapshot still produces the quality shown above. A per-camera `ring_cooldown` (default 45s)
stops a busy camera flooding the feed.

Frigate remains in the stack for genuine continuous-stream cameras (ONVIF/RTSP), where it does the
recording and object detection it is good at. Its `front_door` entry ships **disabled** as a
documented example of the Ring input that does not work.

## Wiring a real RTSP camera into Frigate

## Platform constraints that shaped this setup

These are measured facts about this board, not preferences. Changing them has consequences.

**Orin Nano has no hardware video encoder.** NVIDIA states plainly that "the Jetson Orin Nano does
not have the NVENC engine" and directs users to software libx264. Consequences: Frigate records by
**stream copy** and never re-encodes, and **birdseye is disabled** because it composites and
re-encodes. `h264_nvenc` fails on this board with "Invalid argument".

**JetPack's bundled ffmpeg is a stripped decode-only build.** `7:8.0.1-nvidia1` has **no `scale`
filter, no `lavfi` input, and no software encoders** — only NVENC encoders that this board cannot
use. Frame extraction therefore uses a full static ffmpeg installed separately:

```bash
curl -fsSL -o ff.tar.xz https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-arm64-static.tar.xz
tar xf ff.tar.xz && cd ffmpeg-*-static
sudo install -m755 ffmpeg  /usr/local/bin/ffmpeg-full
sudo install -m755 ffprobe /usr/local/bin/ffprobe-full
```

The bridge points at these via `ffmpeg:` / `ffprobe:` in `bridge/config.yaml`.

**Object detection runs on CPU.** Frigate has no official JetPack 7.2 build — `onnxruntime-gpu`
has no prebuilt Orin wheel for CUDA 13.2, and the classic TensorRT detector was removed in JP7.
Community images exist ([discussion #23388](https://github.com/blakeblackshear/frigate/discussions/23388))
and are the upgrade path if you want GPU detection.

This is a deliberate choice, not only a compatibility workaround: GPU detection would contend with
Cosmos3-Edge for both the GPU and the scarce 8 GB of unified memory, and the detector only has to be
good enough to *trigger* an event — the scene understanding is the VLM's job. Measured CPU detector
inference: **84 ms** at 5 fps detect, which is ample for trigger duty.

**Three frames per event is a hard ceiling.** The visual engine is built with
`max_image_tokens: 1024`, and each frame costs ~302 prompt tokens at the current 320-token per-image
cap:

| Frames | Prompt tokens | Result |
|---|---|---|
| 1 | 329 | OK |
| 2 | 631 | OK |
| 3 | 933 | OK |
| 4 | 1235 | **rejected** — "Failed to handle generation request" |

Raising `num_frames` above 3 requires rebuilding the visual engine with a larger
`max_image_tokens`, not just editing the config.

## Memory

The model holds ~3.8 GB of the board's 8 GB, so every container has a hard `mem_limit`. A container
OOM is recoverable; a host OOM would take the model down with it.

| Component | Limit | Typical |
|---|---|---|
| Cosmos3-Edge shim | — | ~3.8 GB |
| Frigate | 1400 MB | ~400–900 MB |
| ring-mqtt | 400 MB | ~50 MB |
| Mosquitto | 64 MB | ~6 MB |
| porch-dad bridge | 320 MB | ~120 MB |

With the full stack running, roughly **1.6–1.7 GB remains available**. Check pressure with
`vmstat 1 5` and look at `si`/`so` — nonzero values mean active swapping, which will badly hurt
inference latency. Swap *usage* alone is not a problem; swap *traffic* is.

If you need more headroom, in order of value: quantize the visual tower to INT4 (it is still FP16 at
938 MB, ~470 MB recoverable), drop `maxKVCacheCapacity` from 2048 to 1024 (~100 MB), or stop the
desktop session (~150–250 MB, but that kills VNC access).

## The prompts

Both live in `bridge/config.yaml` so they can be edited without touching code. The system prompt
defines the monitoring policy — ignore environmental noise, treat normal deliveries as routine, flag
loitering, obscured faces, unauthorized entry, smoke or fire. The user prompt is templated with
`{num_frames}`, `{stream_id}` and `{camera}`.

The system role **is** honoured by this model — verified with a discriminator probe, where a system
instruction changed both behaviour and response style versus no instruction.

`max_tokens` is 512. That is a ceiling, not a target: the prompt asks for text-message-length output,
so replies are typically 5–60 tokens. The cap costs nothing unless the model actually runs long.

## Operational notes

**Timestamps.** The system prompt asks for a `[MM:SS]` bracket, but the model cannot know the wall
clock or its position in the clip, so treat anything it emits there as decorative. The **feed shows
the real event time from Frigate**, which is authoritative.

**Categories.** `[ALERT]` and `[ROUTINE]` are parsed from the reply. A reply with neither tag is
filed as **UNSCORED** rather than silently assumed routine — an untagged reply is not evidence of
safety. Events that could not be analysed at all appear as **ERROR** with the reason.

**Failures are never silent.** If a clip cannot be downloaded or no frames extract, the event still
appears in the feed as ERROR. Reporting nothing is indistinguishable from "nothing happened", which
is the one outcome a security monitor must never produce. Relatedly, the bridge refuses to caption
zero frames — asked with no images, the model confidently answers "No activity detected", which
would be a fabricated all-clear.

**Latency.** ~1.0 s per event idle; ~4 s with Frigate actively decoding and detecting. Dominated by
the 3-frame prefill, not decode.

## What is verified, and what is not

Verified on hardware, end to end:

- **Frigate 0.18.0 runs on JetPack 7** — healthy, 5.1 fps capture, CPU detector at 84 ms, publishing
  to MQTT, no errors.
- **The complete pipeline runs unattended**: real webcam footage -> Frigate detects `person` ->
  event closes -> MQTT -> bridge -> 3 sampled frames -> Cosmos3-Edge -> classification -> SQLite ->
  MQTT -> web feed. Measured 1387 ms and 1453 ms per event.
- **Descriptions are accurate.** Real output: *"A man wearing a black shirt is stretching by raising
  both arms above his head"* and *"standing in the middle of a room, flexing his muscles"* — both
  matched what the subject was actually doing.
- **The ALERT path works on real footage.** With a subject covering their face, the model produced
  *"Man wearing black shirt, black pants, white watch, ... covering his face with both hands"* and
  the rules classified it **ALERT (face obscured)** in 921–1482 ms — from a clip where only 1 of the
  3 sampled frames contained the subject.
- **The ERROR path works**: a synthetic event with no retrievable clip surfaced as a visible ERROR
  row rather than being dropped.

### Why categorisation is rule-based, not model-based

The model does **not** reliably emit the `[ROUTINE]`/`[ALERT]` tag the system prompt asks for —
measured **0 of 3** runs on real footage. Two alternatives were tested and rejected:

| Approach | Result |
|---|---|
| Reinforce the prompt to demand a tag | 3/3 tag compliance, but descriptions collapsed to *"Human activity is present. [ALERT]"*, it emitted the literal placeholder `[MM:SS]`, and it flagged a person merely standing there as ALERT |
| Second text-only classification pass (~100 ms) | Fast but **unsafe**: labelled *"smoke is rising from the garage"* and *"hooded figure, face covered, looking around"* both ROUTINE, and returned non-answers (`A\nROUT`, an emoji) on others |

False negatives on smoke and on a concealed face are the worst failures this system can produce, so
the policy is encoded as explicit patterns instead. Rules score **12/12** on a fixture set covering
loitering, obscured faces, climbing, forced entry, smoke, wildlife, delivery drivers and empty
replies. They are auditable and cannot drift. An explicit tag from the model is still honoured when
present.

**Not verified:** Ring authentication end to end, since that needs live credentials — everything
downstream of the RTSP URL is proven, but the Ring login itself is untested. INT4 quantization also
shows occasional small artifacts (one run described "white watch, black watch" where there was one
watch), consistent with the ~11% mean weight error documented in the main report.

## Security

Everything binds `0.0.0.0` for LAN access, and **the Cosmos3-Edge shim has no authentication** —
anyone on your network can drive the model. Frigate generates a random admin password on first start
(`docker logs frigate | grep -A2 'Password:'`); change it in the Frigate UI. Mosquitto allows
anonymous connections. That is all reasonable on a trusted home LAN and inadequate if this board is
exposed more broadly.

## Running it

```bash
cd nvr
docker compose up -d                    # mosquitto, ring-mqtt, frigate
sudo systemctl enable --now porch-dad   # the bridge + web feed
curl localhost:8095/healthz
```

Then open **`http://<jetson-ip>:8095`** on your phone.

## Prompt tuning: what a 4B model actually does

The prompts in `frigate/config.yml` were hill-climbed against real frames from these cameras, not
written from intuition. Findings worth keeping:

| Attempt | Result |
|---|---|
| Structured fields (`GATE:` / `HAZARD:` / `PEOPLE:` …) | **Echoes the template back verbatim** instead of filling it in. Unusable. |
| Very terse (<25 words) | **Hallucinates.** Invented "a small, open wooden gate" on a camera with no gate. |
| "Never state what is absent" | Output degraded to literally `No, no, no.` |
| Hazard-first numbered list | Gets the gate right but **ignores the rest of the scene** (missed a parked SUV entirely) |
| Scene description + explicit coverage list | **Best.** Accurate, no hallucination, text-message length. |

Two general lessons: this model needs an *explicit* instruction list to stay grounded, and any
negative instruction ("don't mention X", "never say what's missing") tends to derail it.

### Gate state is not a reliable signal

Asked directly, with `NONE` offered as an option:

| Camera | Ground truth | Model answer (5 runs, temp 0) |
|---|---|---|
| front_driveway | gate **closed** | CLOSED ×5 ✓ |
| front_entryway | **no gate in frame** | CLOSED ×5 ✗ |

The model is perfectly stable but **biased to CLOSED** — it answers CLOSED for a camera with no gate
at all, and never returns NONE. "Closed" therefore carries almost no information: it is the default.

For a "gate left open too long" alert this is the **dangerous** failure direction: a genuinely open
gate would most likely still read CLOSED, producing silence instead of an alert. Gate language is
therefore kept out of the shared prompts entirely, and gate alerting is **not implemented** until
OPEN detection is measured against a genuinely open gate.

Note also that "open too long" is inherently **stateful**, which Frigate's GenAI is not — it captions
each event independently and has no memory between them. It needs a small watcher tracking last-known
state per camera over time, which should only be built once the underlying signal is trustworthy.

## Push notifications to your phone

| Route | Effort | Notes |
|---|---|---|
| **Frigate native web push** | lowest | Built into Frigate 0.14+. Install the Frigate UI as a PWA on the phone and enable notifications in Settings. Works on Android and on iOS 16.4+ (must be added to the Home Screen). No extra service, no third party. |
| **ntfy / Pushover** | low | Subscribe to MQTT `frigate/events`, POST title/body/image. ntfy is free and self-hostable; both have good iOS apps and support attaching the snapshot. Best choice if the Frigate PWA is not enough. |
| **Telegram** | low | Create a bot with @BotFather, get the token and chat id, then POST to `api.telegram.org/bot<TOKEN>/sendPhoto`. Free, no approval, supports images. |
| **WhatsApp** | high | Requires the WhatsApp Business Cloud API (Meta developer account, business verification, a dedicated number) or a paid relay such as Twilio. Business-initiated messages need pre-approved templates. Unofficial libraries violate the ToS. |
| **Apple Messages** | not directly possible | iMessage has no public send API and cannot be driven from Linux. Practical options: an Apple Shortcuts automation triggered by a webhook, or simply use ntfy/Pushover, which deliver native iOS notifications. |

For an iPhone the honest ranking is: **Frigate's own web push first**, then **ntfy or Pushover**.
Both beat WhatsApp and iMessage here on effort and reliability, and neither sends your camera imagery
to a third-party messaging platform.

## Gate / door state: measured, and not viable on this model

The request was "alert when the gate has been open too long". It was tested properly and **not
built**, because the underlying signal does not exist. Across ~9 probes on a genuinely open entryway,
spanning 256 to 522 prompt tokens (including a tight upscaled crop of the doorway):

| Probe | Answer | Truth |
|---|---|---|
| "gate open or closed?" ×3 | CLOSED | OPEN |
| "door open or closed?" ×3 | CLOSED | OPEN |
| "any door or gate standing open?" | "No." | OPEN |
| upscaled door crop, 522 tokens | CLOSED | OPEN |
| free-form description | "a wooden gate that is currently open" | OPEN |

Earlier, on a camera with **no gate in frame at all**, it answered CLOSED 5/5 even when offered NONE.

Two compounding causes, both real:

1. **The model has a hard CLOSED bias.** It never declines to answer. Tellingly, asked to describe the
   doorway it said *"the bright light streaming through the doorway creates an illusion of depth,
   making it appear as though I could walk through"* — it perceives light through an open doorway and
   still answers CLOSED to the binary question. Its free-form text contradicts its direct answers on
   the same image, so neither is usable.
2. **The pixels lack the information.** `latest.jpg` serves the *detect* stream, which is 640x360
   here, and that doorway is backlit with highlights clipped to pure white. The state is not reliably
   readable by a human from that frame either.

For "open too long" — which is also inherently **stateful**, something Frigate's per-event GenAI is
not — use a **physical contact sensor** (YoLink/Zigbee/Z-Wave) into Home Assistant. That yields a
deterministic boolean with a real timestamp, and HA does "open for > N minutes" natively. Route its
alert to the same ntfy topic and gate alerts land beside the camera descriptions.

Use the model for what it demonstrably does well on this hardware: *"male, wearing black shirt,
carrying black backpack"*, *"bald, wearing black t-shirt, black shorts, black flip-flops, walking on
a porch"*, and *"covering his face with both hands"* correctly escalated to ALERT.

## Token budget: the limits that actually bind

Frigate sends roughly six frames per GenAI request even with `use_snapshot: true`. Two separate
engine limits apply, and both produce failures that look like generic errors:

```
EDGELLM_INPUT_TOO_LONG: input length 1800 exceeds engine max_input_len 1536
Failed to handle generation request          <- silently: exceeded max_image_tokens 1024
```

Measured at a 150-token per-image cap:

| Images | Prompt tokens | Result |
|---|---|---|
| 1 | 240 | OK |
| 2 | 386 | OK |
| 4 | 678 | OK |
| 6 | 970 | OK |
| 8 | — | fails |

So `max_image_tokens_per_image: 150` in the engine's `visual/config.json` (runtime-read, no rebuild)
keeps Frigate's payload inside both limits.

**But fewer frames give better text.** At one image: *"a delivery worker standing in the front
driveway, wearing a black shirt"*. At two: *"Person detected in front driveway, San Diego,
California"* - a hallucinated location. At four: *"back view."* If descriptions look thin, reduce the
frame count rather than raising the token budget.

## Getting Cosmos captions into notifications: the two bugs

**1. Frigate stores the description at `data.description`, not `description`.**

The top-level `description` field stays `null` forever. Reading only that makes a fully working
GenAI pipeline look completely broken - every notification falls back to "Person detected (no
description generated)" while the model is in fact producing good output. Always check both:

```python
d = (event.get("data") or {}).get("description") or event.get("description")
```

**2. Review GenAI starves object GenAI on a small engine.**

Two different subsystems call the model with wildly different payloads:

| Caller | Payload | Tokens | Fits? |
|---|---|---|---|
| `objects.genai` (use_snapshot: true) | one snapshot | ~240 | yes |
| `review.genai` | many thumbnails | ~1866 | **no** |

`review.genai` overruns both `max_input_len: 1536` and `max_image_tokens: 1024`, and its failures
look identical to object-description failures in the log. Measured headroom at various caps:

| Per-image cap | Images that fit | First failure |
|---|---|---|
| 150 | 6 (970 tok) | 8 |
| 110 | 10 (998 tok) | 12 |

`max_image_tokens: 1024` is baked into the visual engine at build time and cannot be raised from
config. So `review.genai` is disabled here; re-enable only after rebuilding the visual engine with a
larger `max_image_tokens` and the LLM with a larger `--maxInputLen`.

Verified working end to end - a real Ring event, captioned locally and pushed to a phone:

> **Pinky** - *The person is a delivery worker, wearing a black t-shirt and dark pants, carrying a
> black bag. They are opening a glass door to enter a building.*

## Door state: it works when the image carries the information

An earlier conclusion here was too broad. On a well-exposed frame the model reads door state
correctly and even localises it:

| Image | Probe | Answer | Truth |
|---|---|---|---|
| Patio, clear light | "open or closed?" | **OPEN** | OPEN ✓ |
| Patio, clear light | "which one?" | "the **right** glass door is standing open" | ✓ |
| Entryway, backlit | "open or closed?" ×4 | CLOSED | OPEN ✗ |

The entryway failures were not a model limitation but an **image** limitation: that doorway is
backlit with highlights clipped to pure white, and the state is not reliably readable by a human
either. Vision-based door state is workable on a well-exposed camera. For "open too long" a physical
contact sensor is still the better instrument, because that requirement is stateful and deterministic.
