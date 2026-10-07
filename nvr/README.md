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
not have the NVENC engine" and directs users to software libx264. **birdseye stays disabled** for
that reason - it composites and re-encodes every camera continuously. Re-tested inside the Frigate
container, both hardware encoders the ffmpeg build advertises fail at runtime:

```
h264_nvenc    -> Cannot load libcuda.so.1
h264_v4l2m2m  -> Could not find a valid device
```

An encoder appearing in `ffmpeg -encoders` is not evidence the silicon exists.

Recording, however, **does** re-encode - see [Recordings that will not play](#recordings-that-will-
not-play-in-a-browser) below. Stream copy was the original choice, and it turned out to be the cause
of unplayable recordings; the encode was measured before being accepted rather than assumed
unaffordable.

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

## Engine sizing: measured, and smaller won

Three engine builds were measured against the workload that actually produces good captions.

| Engine | maxInputLen | max_image_tokens | KV | Shim RSS | vs v1 |
|---|---|---|---|---|---|
| v1 | 1536 | 1024 | 2048 | 3.410 GB | — |
| v2 | 3072 | 2048 | 3072 | 3.692 GB | **+8.3%** |
| **v3 (deployed)** | **1024** | **640** | **1024** | **3.395 GB** | −0.44% |

v2 was built to unblock `review.genai`, and it worked: 8 images at 1334 tokens pass where 6 failed
before. But it cost **+282 MB to enable frame counts that degrade output**:

| Images | Prompt tokens | Output |
|---|---|---|
| 1 | 249 | "Black male carrying black bag in left hand" |
| 2 | 404 | "male wearing a black hoodie" |
| 6 | 1024 | `person: 63% Confidence.\nperson: (n/d)` |
| 8 | 1334 | "Person walking in front of home." |

That is the third independent replication that **more frames make this model worse**. So v3 went the
other way - sized for one snapshot (~250 prompt tokens) plus 512 generated, with
`maxKVCacheCapacity 1024` per Jetson AI Lab's own Orin Nano guidance. It is the smallest of the
three and loses nothing usable.

**All three engines are equally accurate.** The names describe speed, because speed is the only axis
that separates them. Each engine re-ran the identical 23-frame set, scored by the same classifier
against the same hand-checked truth set:

| Engine | Alert accuracy | Avg latency | RAM |
|---|---|---|---|
| **Fast (v2)** | 95.7% (22/23) | **208 ms** | 3.692 GB |
| Medium (v1) | 95.7% (22/23) | 443 ms | 3.410 GB |
| Slow (v3) | 95.7% (22/23) | 482 ms | 3.395 GB |

Identical scores, and all three miss the *same* frame (a small dog at a person's feet) - a
perception limit of the shared INT4 weights, not an engine difference. The engines differ only in
context budget, which a single-snapshot workload never exercises.

The latency ordering is the counter-intuitive part: the **largest-context build is the fastest**,
because it spends the fewest tokens per frame (170 vs 320) and so does the least prefill work.
Context size and per-frame token spend are set independently, and it is the token spend that drives
latency. Anyone reasoning "bigger context = slower" or "more tokens = more accurate" gets both
backwards here.

Caveat worth stating: 23 frames is a small set, and a tie across three engines is exactly what too
little resolving power looks like. The honest claim is that no accuracy difference is detectable at
this sample size - not that none exists.

**RAM is dominated by fixed costs**, which is why the tunable limits barely move it: 0.818 GB LLM
engine + 0.915 GB visual engine + 0.500 GB embeddings = 2.23 GB of the 3.4 GB. Engine limits are
worth roughly 300 MB in total. The only remaining large lever is INT4 on the vision tower.

### Vision tower INT4: quantized, but not loadable

The quantization works - 110 linears, 11.52% mean relative error, 0.911 GB -> 0.611 GB - and the
weights are kept for later. The export pipeline cannot consume them:

- This is a **modular checkpoint** (`transformer/` + `vision_encoder/` subdirs), so
  `_load_all_weights(model_dir)` returns nothing at the root.
- `_tower_model_config` decides a tower is quantized only if it finds `.weight_scale` in that root
  dict, finds none, and resets the tower to FP16.
- The exporter then builds `FP16Linear` against packed `[N//2, K]` weights:
  `RuntimeError: The size of tensor a (576) must match the size of tensor b (1152)`.

Fixing it needs a vendor source patch. Note also that the ViT MLP can **never** be INT4 here: it is
1152 <-> 4304 and `4304 % 64 == 16` fails the kernel's alignment rule, so ~268M of ~488M params are
permanently FP16 and any future saving is capped near a third of the tower.

## Detector budget and motion gating

The CPU detector is a shared, finite resource: at ~86-116 ms per inference it sustains roughly
**9-11 detections/sec across all cameras combined**. Six cameras at 5 fps demand 30/sec, so every
camera starves and real people go undetected - which is exactly what happened.

Following the DeepStream/VSS principle of gating inference cheaply rather than running the detector
on everything:

- `motion.contour_area` 15 -> 40 (primary) / 60 (secondary) so leaf-scale movement is ignored
- `motion.threshold` 30 -> 40 / 45 so subtle luminance shifts (shadow, cloud) do not trigger
- `detect.fps` split by importance: primary 4, secondary 2

Result: detector inference **116 ms -> 85.6 ms**, primary cameras actually detecting (4.1-4.2
detection_fps) while idle secondaries sit at 0.0. This also enforces "ignore wind, trees and
shadows" at the cheap layer instead of asking the VLM to do it.

On frame counts, this deliberately diverges from [NVIDIA VSS](https://docs.nvidia.com/vss/3.1.0/real-time-vlm.html),
which samples ~8 frames per 30-second chunk. VSS targets 8B+ models such as Cosmos-Reason2; on this
4B INT4 model more frames measurably degrade output, so one snapshot is used instead.

## Prompt findings: positive framing beats prohibition

Measured on real frames from these cameras:

| Prompt | Result |
|---|---|
| "Include delivery and service workers" | Invented a **"delivery boy"** with a **"white receipt"** that do not exist |
| "Do not guess an occupation" | **"There is no visible person"** on a frame that clearly contains one |
| (neutral, no guidance) | Invented an **"old man"** |
| **"Describe only what is visible"** | **"A person is leaning against the wall, wearing a black jacket and dark pants, with a black backpack on."** |

Two rules follow. **Naming a role primes it** - mention delivery workers and the model will find one.
And **prohibitions destabilise this model**: "do not guess" produced a false negative, the same
failure class as "never state what is absent" collapsing to "No, no, no." Prefer positive
constraints throughout.

## Alert quality: 87% -> 95.7%, judged against the frames

Alerts were noisy: phantom delivery workers, a black cat that did not exist, the model narrating
itself ("I am Pinky, a delivery assistant"). 23 captioned events from front_driveway,
front_entryway and pinky were re-run and judged by reading each actual frame.

### Three defects, all caused by the prompt

| Defect | Cause | Evidence |
|---|---|---|
| Roleplay | `Look at home camera '{camera}'` made the name become content | *"I am Pinky, a delivery assistant"*, *"a security guard, pinky-shaped"* |
| Overlay read as scene | `snapshots.bounding_box: true` burns boxes and `car: 84%` onto the image | *"a green bounding box around the vehicle"*, *"a 70% detected license plate"* |
| Category manufacturing | the prompt enumerated the alert categories | see below |

The third is the important one. **Every category named in the prompt came back as an observation
whether or not it was present:**

| Prompt listed | Model produced | Reality |
|---|---|---|
| "delivery and service workers" | "delivery worker" on 9 frames | one resident |
| "a gate standing open" | "a gate that is standing open" | gate closed |
| "a person or animal and what it is doing" | "A person is walking through a gated area" | empty driveway |

So the prompt written to *narrow* alerting was **generating** false alerts in every listed category.

### The fix: the model describes, code decides

The prompt now enumerates nothing and names no camera:

```
Describe only what is visible in this image, in one short sentence.
```

Every alert category lives in `nvr/feed/alert_policy.py`. Two refinements mattered:

- **Fuse Frigate's detector label.** It is a trained object model and beat the VLM on presence - it
  labelled one event `dog` that the caption missed entirely.
- **Suppress parked vehicles.** These driveways always contain a parked car, so firing on every
  mention of it alerted on the background in 3 of 23 frames. Scope is *movement* of vehicles, so a
  parked car is background unless Frigate itself raised a vehicle event.

### Result

**95.7% (22/23) alert accuracy, with zero false alarms.** The single miss is a small dog at a
person's feet in a wide-angle frame; the person was still alerted, so the event was not dropped.

Two rejected alternatives, both measured:

- *"What is happening in this image?"* fixed the camera-narration failures but scored **worse
  overall** - it invented a car crash and brought the cat hallucination back. Tuning on the failing
  subset regressed the whole set.
- Enumerating the scope per-camera (gate only on driveway cameras) still produced false gate-open
  reports, because naming the category is what triggers it.

Headlines are also cleaned before they reach the phone: *"Two cars are parked in a driveway, viewed
through a fisheye lens that distorts the perspective"* becomes *"Two cars are parked in a driveway."*


## Recordings that will not play in a browser

Symptom, from Chromium's media pipeline while scrubbing the Frigate recordings timeline:

```
Failed to play recordings (error 3): PipelineStatus::PIPELINE_ERROR_DECODE:
Failed to send video packet for decoding:
{timestamp=114367333 duration=45188 size=28752 is_key_frame=0 encrypted=0}
```

**Cause: Ring cameras change resolution mid-session, and stream copy preserves that faithfully.**
Fourteen consecutive `front_entryway` segments held four different resolutions:

| Resolution | H.264 level |
|---|---|
| 880x494 | 3.2 |
| 1312x736 | 4.0 |
| 1968x1104 | 5.0 |
| 2624x1472 | 5.0 |

Playback concatenates 10-second segments into one HLS stream whose init segment declares a single
SPS/PPS. A mid-stream resolution change violates it and the browser rejects the packet.

The control case was already in the deployment: **`pinky` never adapts** (1280x720 Constrained
Baseline on every segment) and is the one camera whose recordings always played cleanly. Per-camera
decode-error counts made it plain - the adapting cameras threw 1-5 errors per segment, `pinky` zero.

**Why this hides from the command line:** `ffmpeg -f null -` conceals the discontinuity and reports
success. The files look fine from a shell and fail in a browser, because ffmpeg's decoder is
permissive and Chromium's is not. Verifying with ffmpeg alone would have cleared a broken file.

### Two hypotheses that measurement killed

Worth recording, because both sounded right:

1. **"`+discardcorrupt` is punching holes in the GOP."** Plausible - it drops damaged packets, so
   surviving P-frames reference frames that are gone. Tested by capturing the live stream with and
   without it: **removing it made things worse** (2/6/0 decode errors vs 0/0/0). It was helping.
2. **"The timestamps are non-monotonic."** A remux did warn `non monotonically increasing dts`. But
   probing the packets on disk showed **zero** violations - the warnings came from the remux
   rounding into a coarser timebase, not from the files. A diagnostic can manufacture the very
   symptom it is used to look for.

### The fix

Normalize every recorded segment to one set of stream parameters (`ffmpeg.output_args.record`):

```yaml
ffmpeg:
  output_args:
    record: >-
      -f segment -segment_time 10 -segment_format mp4 -reset_timestamps 1 -strftime 1
      -vf fps=15,scale=1280:720 -c:v libx264 -preset ultrafast -tune zerolatency
      -profile:v high -level 4.0 -pix_fmt yuv420p -g 30 -crf 28 -an
```

| Flag | Why |
|---|---|
| `fps=15` | source is variable-rate; measured packet gaps ran 0.2 ms to 0.63 s |
| `scale=1280:720` | one resolution for every camera and every segment |
| `-profile`/`-level` | pinned, so every segment emits a byte-identical SPS |
| `-pix_fmt yuv420p` | Ring sends full-range `yuvj420p`; browsers expect `yuv420p` |
| `-g 30` | a keyframe every 2 s, so segments always cut on one |

**Cost, measured before accepting it:** libx264 ultrafast at 1280x720 encodes a 7.3 s segment in
0.93 s - 0.13x realtime, roughly 25% of one core per camera. ffmpeg already decodes the stream to
produce the detect output, so the incremental cost is the scale and encode alone. The CPU detector
still dominates at ~150%. Load average returned to its pre-change level after the restart transient.

Result - every new segment across all four cameras, where before there were four resolutions:

```
front_entryway   Constrained Baseline,1280,720,yuv420p   10.000s
front_driveway   Constrained Baseline,1280,720,yuv420p   10.000s
office           Constrained Baseline,1280,720,yuv420p   10.000s
pinky            Constrained Baseline,1280,720,yuv420p   10.000s
```

Durations are now exactly 10.000 s rather than ragged 7.3-13.5 s, because the stream is CFR.
`-tune zerolatency` disables CABAC and B-frames, so the output lands on Constrained Baseline
despite `-profile:v high` - which is the most compatible profile, and exactly what `pinky` was
already producing.

**This does not repair recordings made before the change.** They still hold mixed resolutions and
will still fail to play. Only segments written after the restart are normalized.


## Alert accuracy: the label fusion was overruling the caption

Re-ran Cosmos3-Edge over **all 35 stored events** and classified each with the live policy. Every
single one alerted - a **100% alert rate**, which is what "the alerts are a little goofy" looks like
from the inside. A security system that fires on everything conveys nothing.

**Cause: `FRIGATE_LABEL_MAP` fusion ran unconditionally, after the parked-vehicle suppression.**
The classifier suppressed a parked car correctly, and then the detector label put it straight back:

| Caption | Frigate label | Verdict |
|---|---|---|
| "A silver SUV is **parked**…" | *none* | no alert |
| "A silver SUV is **parked**…" | `car` | **ALERT** |
| "An **empty** driveway with a **closed** gate." | `car` | **ALERT** |

Frigate labels every driveway event `car`, so the suppression was dead code in production - it
could only ever fire in the one case that never happens. Six frames alerted whose own caption said
the cars were parked.

**The fusion still earns its place**, which is why it was not simply deleted. Two frames were
checked by eye and the detector was right where the caption was wrong:

- Frigate said `dog`; the caption said "two women sitting at a table". There **is** a black dog at
  the woman's feet.
- Frigate said `person`; the caption described only trees and a driveway. There **is** a person at
  the bottom-right edge of frame.

So the rule is now: **the label may add what the caption failed to see; it may not overrule what the
caption actually saw.** One line:

```python
if mapped and mapped not in cats and not (mapped == "vehicle" and vehicle_parked):
    cats.append(mapped)
```

### Result, per camera

Truth was set by looking at all 35 frames directly - not by trusting the caption, and not by
trusting the label. Where they disagreed, the image decided.

| Camera | Before | After | n |
|---|---|---|---|
| `front_driveway` | 58.3% | **91.7%** | 12 |
| `front_entryway` | 100% | **100%** | 3 |
| `pinky` | 100% | **100%** | 17 |
| `driveway_2nd_floor` | 66.7% | **100%** | 3 |
| **Priority cameras** | 84.4% | **96.9%** | 32 |
| **All cameras** | 82.9% | **97.1%** | 35 |

**False alarms: 6 -> 0.** The goal was 95% and it is met on both the three priority cameras (96.9%)
and the full set (97.1%).

### The one remaining miss, stated honestly

A red car **arriving** - headlights on, occupant visible - is captioned "a red car and a silver SUV
parked side by side". Its own caption says "parked", so a caption-driven classifier cannot separate
it from the two re-detections of that same car sitting still 17 minutes later.

`data.path_data` was tested as a way to tell arrival from parked and **does not work**: the parked
SUV scores 0.165 max displacement and the arriving car 0.183. The tracker jitters on a stationary
car about as much as a car creeps in. Recorded here so it is not re-attempted.

The trade is deliberate and favourable: five recurring false alarms removed for one missed arrival
of a resident's own car, which is not the "out of the ordinary" event this system exists to catch.

### Reproducing

```bash
python3 nvr/feed/rerun_all_events.py   # fresh inference over every stored event
python3 nvr/feed/eval_alert_policy.py  # score before/after against the judged truth set
```


## Demo mode: measuring the model, not the NVR

The NVR stack and the model compete for the same 8 GB and the same 6 cores. With everything
running, the board sat at **6.73 GB used, 0.80 GB available, load average 7.56** - benchmarking the
Live VLM WebUI in that state measures Frigate's CPU detector as much as it measures Cosmos3-Edge.

```bash
bash nvr/demo-mode.sh on                      # keep the desktop/VNC session alive
bash nvr/demo-mode.sh on --remote-recording   # also stop VNC, if recording from another machine
bash nvr/demo-mode.sh status
bash nvr/demo-mode.sh off                     # full stack back
```

Demo mode keeps four units:
- `cosmos3-edge-shim.service`, which holds the engine;
- `cosmos-edge-ui.service`, Live Vision;
- `reachy-mjpeg-bridge.service`, Live Vision's Reachy source;
- `pokemon-classifier.service`, its classifiers.

It stops everything else in the NVR:
- the Frigate, ring-mqtt, Home Assistant and MQTT containers;
- the three Scout bridges;
- porch-feed, porch-dad, frigate-notify and breed-classifier.

`on` records what it stopped in `/home/orin/nvr/.demo-mode`, and `off` brings back exactly that. A
container that was already down, such as Home Assistant outside the boot set, stays down. It runs
one `systemctl` call per unit, so a sudoers file can allow exactly those lines
(`ui/config/sudoers-live-vision-engines`). `json` reports the state for Live Vision.

**From Live Vision.** The page's **Demo mode** button runs the same script, and also covers engines
that only fit on a quiesced board:
- An engine marked `"needs_demo_mode": true` in `engines.json`, today LocateAnything 3B, says
  "Needs demo mode" under its button.
- Clicking it shows a warning: what will stop, and why the model does not fit beside the NVR.
- Only after the user confirms does the page send the switch with `?demo=1`. The server then
  enters demo mode and loads the model. Stopping the NVR is never a side effect.
- **End demo mode** first swaps such a model back for one that fits: the one active before, or the
  registry's `"default"`. Only then does the NVR come back, so Frigate never restarts beside a
  model that crowds it out.

Verified on 2026-10-05 with LocateAnything:
- Refused without consent (HTTP 409).
- With consent, loaded in 138 s, with 3.46 GB still free.
- Answered a Bernese query.
- Ended in 50 s, with Cosmos3-Edge v3 back first and every container, service and camera running
  again. Frigate's runtime camera state was unchanged.

The table below predates the current script. It was measured when demo mode kept the shim and
`live-vlm-webui.service`:

| | Full stack | Demo mode |
|---|---|---|
| RAM used | 6.73 GB | **4.83 GB** |
| RAM available | 0.80 GB | **2.53 GB** |
| Load average | 7.56 | **0.69** |

`--remote-recording` exists because stopping `x11vnc`/`gnome-remote-desktop` kills the Orin's own
desktop session. Only pass it when the recording is driven from another machine's browser.

### Measured on the quiesced board

Fitted over **2818 real requests** from a live WebUI session:

```
elapsed_ms = 200 + 13.49 x generated_tokens          (r = 0.854)
```

| | |
|---|---|
| Best end-to-end | **245 ms** |
| Typical, uncontended | **253 ms** (14-token caption) |
| Typical, live streaming ~1 req/s | **442 ms** (18-token caption) |
| p90 | 607 ms |
| Steady-state decode | **13.49 ms/token = 74.1 tok/s** |
| CPU | **15.0% median, 23.5% peak** - system-wide across 6 cores |
| GPU | 97.7% median, 99.6% peak |
| System RAM | **4.95 GB of 7.37 GB**, shim RSS 3.85 GB, 2.60 GB free |

**The latency spread is caption length, not instability.** 12 generated tokens takes 322 ms, 41
takes 702 ms, and the relationship in between is linear. Quoting a single mean for this workload
hides that entirely - the mean just tracks whatever caption-length distribution the traffic had.

The 13.49 ms/token measured here independently reproduces the campaign's converged
13.13 ms/token, on a different day and a differently loaded board.

Reproduce with `bench/bench_live_webui.py`; the recorded run is `bench/live-webui-quiesced.json`.


## Stabilising CPU: what actually costs what

The board hit **load average 10.23 on 6 cores** with Frigate at **450%**. Measured the floor first,
then worked down to it.

**Baseline — porch-dad fully loaded, zero camera feeds:** shim, WebUI, Frigate, ring-mqtt, MQTT and
Home Assistant all running, every camera disabled. **~11% system-wide**, load settling to 2.66.
That is the floor, and it is what any camera cost is measured against.

| | Peak | After | Baseline |
|---|---|---|---|
| Frigate | 450% | **132–163%** | 30% |
| Load average | 10.23 | **3.32** | 2.66 |
| System-wide CPU | ~89% busy | **33%** | ~11% |

### The four things that were actually expensive

**1. The record re-encode — my own measurement error.** Normalizing recordings to fix browser
playback cost far more than estimated. The original figure (~25% of one core per camera) came from
re-encoding an **already-normalized 1280x720 segment**. In production `front_entryway` streams
**2624x1472 at 24 fps** — 4.3x the pixels — and those ffmpeg processes sat at **116–124%** each.
Dropped the record output to `fps=5, scale=960:540`. Detect only consumes 2 fps; 5 fps is plenty to
review an event.

**2. Live view was transcoding per viewer.** With no `go2rtc:` or `live:` section, Frigate 0.18
falls back to **jsmpeg**: it takes the 640x360 detect stream, *upscales* it to 1280x720 and encodes
MPEG-1 on the CPU, per viewer. That is the "CPU spikes when I open live view" symptom exactly.
Adding a `go2rtc` stream **named after each camera** lets Frigate hand the browser H.264 over
WebRTC/MSE with no server-side transcode.

**3. Motion sensitivity, not detect fps, drives the detector.** CPU inference costs **94.8 ms per
region** on this board, so the dial that matters is how many regions motion produces.
`contour_area: 40` meant every leaf edge and shadow spawned its own region and the detector ran
near-continuously at **110%**. Raising it to `threshold: 50, contour_area: 150` took the detector to
**52.6%** — a bigger win than halving detect fps, and it aligns with the alert policy, which already
declares trees in wind out of scope.

**4. `truck` was in `objects.track` and the model does not support it.** Frigate logged a warning
per camera on every start. Removed.

### Hardware decode is not available, and the reason is not the obvious one

`h264_cuvid` is compiled into both ffmpeg builds, so this looks reachable. It is not:

```
container:  Cannot load libcuda.so.1          # libcuda is not mounted in
host:       Cannot load libnvcuvid.so.1       # not on the linker path
host + LD_LIBRARY_PATH=/opt/nvidia/l4t-gpu-libs/openrm:
            CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
```

`libnvcuvid.so.1` **does** exist, under `openrm` — but `openrm` is the discrete-GPU stack. Jetson
reaches NVDEC through **V4L2**, not the CUVID API: `gst-inspect-1.0` finds `nvv4l2decoder`, and
`/usr/lib/aarch64-linux-gnu/nvidia/libv4l2_nvcuvidvideocodec.so` is the Tegra path. So an ffmpeg
pipeline cannot hardware-decode here regardless of how the libraries are wired. A GStreamer
`nvv4l2decoder` pipeline is the remaining untested route.

Software fallback that was measured but **not deployed**: `-skip_frame nonref` cut a decode+encode
pass from 0.86 s to 0.38 s (2.3x). It was left out because it drops frames from the record path, and
recording playback had only just been fixed — not worth risking for a second-order win.

### What is left, and why

The remaining ~130% is two ffmpeg processes decoding **2624x1472@24** and **1968x1104@15**. That
decode is unavoidable in software: every frame must be decoded to produce even a 2 fps detect
stream. Ring chooses that resolution and does not expose a substream, so the only real fix is
hardware decode via GStreamer.

Costs scale directly with what Ring decides to send, so expect this number to move on its own —
Frigate was observed at 132%, 163% and 217% within ten minutes with no configuration change at all.


## Night-time false alerts: three bugs, none of them CPU

Low light makes Ring drop resolution, and the stream streaks on the change. That produced
erroneous alerts on a parked car overnight. The CPU work above helps only sideways — detect at
2 fps halves exposure to glitch frames and `contour_area: 150` kills small artefacts — but a
resolution change is a full-frame event with a huge contour, so it still fires. The real causes
were in the classifier.

Measured over one night on `front_driveway`, with the caption each event actually produced:

| Score | Label | Caption | Was |
|---|---|---|---|
| 0.70 | person | "A **distorted, pixelated** image of a dimly lit room with a person" | ALERT |
| 0.71 | person | "A black SUV is **parked** under a concrete overpass at night" | ALERT |
| 0.72 | person | "A **blurry view** from a moving vehicle" | ALERT |
| 0.82 | car | "**No people are visible.** A black cat is walking across the driveway" | ALERT person |
| 0.71 | car | "**No exterior lamps are lit**" | ALERT lights |

### 1. Negation read as sighting

Matching the whole caption at once reads a word inside a denial as an observation.
**Four of six `lights` matches in one night were "No exterior lamps are lit"** — the exact opposite
of the thing being alerted on. The same applied to "There are no vehicles visible" and "No people
are visible". The model volunteers these denials readily, so this was not an edge case.

Fixed by matching **clause by clause** and dropping any clause containing a negation, splitting on
sentence boundaries *and* on commas that precede a negation (the model writes
"No people are visible, no animals are visible, no vehicles are visible" as one sentence).

### 2. A degraded frame is evidence about the frame, not the scene

When the caption says the image is blurry or pixelated, nothing else in it is trustworthy —
including a person the model hallucinated into the artefact, and including the detector label,
which is what actually fired these. A degraded frame now suppresses the whole event.

The distinction that makes this safe: the degradation must describe the **image**, not a subject.
A first attempt matched the bare adjective and suppressed *"A blurry person in a white shirt"* — a
real person on `pinky`, seen imperfectly. Requiring an image noun (`image`, `view`, `footage`,
`frame`) within 30 characters separates "the picture is broken" from "the subject is indistinct".

### 3. Uncorroborated detector labels need confidence

Every uncorroborated false positive sat at **0.70–0.72**; every real sighting the caption also
described scored **0.74–0.84**. So a category that comes *only* from the detector label, with
nothing in the caption backing it, now requires `top_score >= 0.75`. Caption-corroborated
categories are deliberately **not** gated — a caption that names the thing is its own evidence at
any score, which is what keeps the 0.71 and 0.72 real sightings on `pinky`.

### Result

Overnight false alerts **5 → 0**, with all nine real sightings preserved. Re-checked against the
judged 35-event set from the earlier accuracy work: **still 97.1% overall, 96.9% on the priority
cameras, 0 false alarms** — no regression, same single known miss (an arriving car whose own
caption says "parked").


## "The Live VLM WebUI is crashing" — it was not crashing

Symptom: the WebUI appeared to die during live streaming. It never did.

```
Active: active (running) since 12:39:30 PDT; 2h 33min ago
NRestarts=0          Result=success
```

Zero restarts, zero errors in its log, zero OOM kills, and it was answering requests at ~265 ms
throughout. What looked like a crash was a **stall**.

### Cause: the model was being paged out to a disk-backed swapfile

| | |
|---|---|
| Free RAM | 90 MB of 7.4 GB |
| Swap in use | 1120 MB of 2048 MB (a `/swapfile`, i.e. disk) |
| **Of that, model pages** | **566 MB** |
| `vm.swappiness` | 60 (the image default) |

Every inference that touched a swapped page had to fault it back from disk. That stalls the
request long enough for the browser's WebRTC session to give up, and the client reads a dropped
connection as a dead server.

Model weights are the single worst thing on this box to page out, so:

```ini
# /etc/systemd/system/cosmos3-edge-shim.service.d/10-no-swap.conf
[Service]
MemorySwapMax=0
```

plus `vm.swappiness=10` and stopping the unused desktop session (gdm, 243 MB — the demo is driven
from another machine's browser). Result: **swap 1120 → 335 MB, model swap 0, available RAM
785 → 1008 MB.**

### What was pinned, and what deliberately was not

The same drop-in was applied to `live-vlm-webui` and then **reverted**. Its RSS under sustained
streaming goes 212 MB idle → 622 MB peak → 544 MB steady. That looked like a leak at first and is
not one — it grows and then reclaims. Pinning it would cost ~600 MB of hard RAM for frame buffers
that tolerate paging perfectly well, on a board with ~400 MB spare.

So only the model is pinned. The rule that falls out: **pin what is latency-critical and
re-read constantly; let large, elastic, latency-tolerant buffers page.**

### The tradeoff this introduces, stated plainly

`MemorySwapMax=0` means the shim can no longer swap under pressure — so if the board genuinely runs
out of RAM, the shim gets **OOM-killed** rather than degrading slowly. That is the better failure
mode here (a 2 s restart beats minutes of thrash), but it is a real change in behaviour. Headroom
is ~800 MB; the obvious reclaim if that tightens is Home Assistant, which is resident at ~318 MB
and logged **zero lines in 30 minutes**.

Note `/proc/pressure/memory` does not exist on this image, so `systemd-oomd` is disabled and there
is no userspace OOM protection — the kernel OOM killer is the only backstop.


### What survives a reboot, and the one thing that does not

Verified before a planned restart, then updated for the boot set of 2026-10-03, in which Live
Vision replaces the Live VLM WebUI as the Live UI that comes back. That boot set was applied on
the board that day and has not yet been through a reboot:

| | |
|---|---|
| `MemorySwapMax=0` drop-in | present in `/etc/systemd/system/cosmos3-edge-shim.service.d/` |
| `vm.swappiness=10` | present in `/etc/sysctl.d/99-porchdad-swap.conf` |
| cosmos3-edge-shim, porch-feed, porch-dad, frigate-notify, reachy-mjpeg-bridge, cosmos-edge-ui (Live Vision), pokemon-classifier (its classifiers) | all `boot=enabled`, as are docker and containerd ([deploy/07 §8](../deploy/07-reachy-homeassistant-and-orin-changes.md)) |
| live-vlm-webui-fork (Live VLM WebUI) | `boot=disabled` and stopped: still installed, started on demand ("Live Vision" below) |
| frigate, ring-mqtt, mosquitto, scout-bridge, scout-bridge-first-floor | all `restart=unless-stopped` |
| homeassistant | stopped, `restart=no`: not part of the boot set (deploy/07 §8) |
| camera on/off state | persisted by Frigate in `/config/.runtime_state.json` — `front_entryway` ON, `front_driveway` ON, `office` OFF, `pinky` OFF |

The two Live UIs' boot state follows the command centre (deploy/07 §8): ON or OFF there also
enables or disables the unit. So their two rows are the default until someone changes it there,
and the intent it records should match them ("Live Vision" below).

**The exception: `gdm` comes back.** It is `boot=static`, pulled in by the default target:

```
$ systemctl get-default
graphical.target
```

So the ~243 MB desktop session that was stopped to make room returns on every boot. On a board
where the model is now pinned in RAM and cannot swap, that 243 MB is the difference between
comfortable headroom and an OOM kill. If the Orin is always driven headless — the demo runs from
another machine's browser — this is the persistent fix:

```bash
sudo systemctl set-default multi-user.target   # revert: set-default graphical.target
```

Left unset here rather than changed silently, because it removes the local GUI entirely and that
should be a deliberate choice, not a side effect of a memory fix.


## Detection modes: a second, targeted look after Frigate's detection

Frigate's detector names COCO classes, so every dog in a doggy daycare is just "dog". A detection
mode ([`feed/detection_modes.py`](feed/detection_modes.py)) is a second look that asks one
question. On the cameras it is on, when Frigate reports one of its labels, porch-feed answers
that question about the camera's frame. It uses a specialist where there is one, and the loaded
VLM otherwise. The answer replaces the stock description in that camera's notification.

**The Bernese mountain dog mode** is the first. It is on for the Reachy Mini, which watches a
doggy-daycare feed on a monitor. Each look goes like this:

1. **Detection.** Frigate tracks `dog` on `reachy_mini`. Its generic GenAI description stays
   person-only there.
2. **Trigger.** porch-feed looks on the event's `new` or `update`. It does not wait for `end`,
   because a dog asleep at daycare can be one event for an hour.
3. **Identify, on Frigate's boxes.** It takes the robot's own 1280x720 frame and every dog box
   Frigate is tracking on the camera. `breed-classifier.service` scores each box with ImageNet
   ViT-B/16 ([`feed/breed_service.py`](feed/breed_service.py): INT8, CPU, onnxruntime, its own
   venv) against the Swiss mountain dog family that a blurry Bernese is taken for: Bernese,
   EntleBucher, Appenzeller and Greater Swiss. The best box at 0.5 or more is the Bernese, and
   its box is the 2D grounding.
4. **Or ask the VLM.** With no dog tracked, as in a Look now, or with no breed service, it asks
   the loaded VLM the benchmark's yes/no question with logprobs: "Is there a Bernese mountain dog
   in this image? Answer with only yes or no." It gates on the model's own P(yes) of at least 0.6.
   Only then does it ask for the box in that model's own dialect: `bbox_2d`, `box_2d` or `<ref>`.
   LocateAnything takes the phrase itself.
5. **Notification.** When found, porch-feed does four things:
   - draws the box on the frame and saves a Feed entry;
   - sets Frigate's sub_label on the event, which then shows as "dog · Bernese mountain dog";
   - pushes to the phone, for example "Reachy Mini · Bernese mountain dog: Bernese mountain dog
     spotted in the middle of the frame (99% sure).", with the boxed frame attached;
   - optionally says it aloud on the Reachy.

   frigate-notify stands its stock push down for any label a mode owns, so there is one
   notification, not two.
6. **Cooldown.** It looks at most once per camera every 300 s.

Measured on 35 frames of the daycare footage the Reachy recorded on 2026-10-04 (332 dog crops).
These are the frames the gates were set on:

| Engine | Frames with the Bernese found | False alarms |
|---|---|---|
| Breed scores on the dog boxes | 4 of 4, scoring 0.70-0.99; one of them Qwen3-VL missed | none: no other dog above 0.09 |
| Qwen3-VL-2B | 3, with tight boxes | a curled-up husky at P(yes) 0.54, under the 0.6 gate now in use |
| Cosmos3-Edge v3 | | yes on 24 of the 35 frames, boxing doodles and huskies |

The breed path runs whatever VLM is loaded, so the mode works beside stock Cosmos3-Edge:
- **Speed:** about 0.5 s per dog on two capped cores.
- **Memory:** it loads its 89 MB model on demand and drops it after 15 idle minutes (27 MB idle,
  231 MB loaded).
- **Engine trust:** for its VLM fallback, each mode still lists the engines it trusts. For this
  one that is Qwen3-VL and LocateAnything, in `MODES["bernese"]["engines"]`. With another model
  loaded, a VLM verdict is recorded in the Feed for comparison, but nothing is pushed or spoken.

The command centre's **Detection modes** card has three controls:
- one chip per Frigate camera, to turn the mode on there;
- toggles for the push and for the Reachy's voice;
- a Look now button per camera, which spends one look whatever the cooldown.

Settings persist in `/home/orin/nvr/feed/detection_modes.json`, seeded by `config.yaml`'s
`detection_modes:` block. `vlmbench/mode_eval.py` runs a mode over a folder of frames on any engine
and draws a contact sheet. That is how the numbers above were taken.

**Different prompts and engines per camera: what this board can do.**
- **Prompts per camera, the Frigate way.** Frigate 0.18 takes per-camera
  `objects.genai.object_prompts`. A `reachy_mini`-only dog prompt was validated: it resolves while
  every other camera keeps the global one. Frigate fills in `{camera}`, `{label}` and
  `{sub_label}`; literal braces must be doubled. It can switch prompt sets at runtime with profiles,
  over MQTT `frigate/profile/set`. All of this changes only the wording of the stock description:
  there is no gate, no box and no probability, and the text goes out as-is.
- **Prompts per camera, the detection-mode way.** A mode can ask any question, gated on the
  model's own probability, with a box drawn and its own notification.
- **Engines per camera: one VLM, plus specialists.** One VLM fits beside the NVR (2.2-3.1 GiB
  resident, with 0.8-1.3 GB left), and switching is global and takes 45-55 s. So a camera's mode
  gets an engine of its own in one of two ways:
  - a small specialist beside the VLM, like the breed service;
  - a `url` in `config.yaml` pointing at a second box. It is config-only, so the control plane can
    never be told to send camera frames somewhere new.

**Recommended way to customize detection per camera:**
1. **Keep Frigate as the first stage** on every camera, tuned per camera: labels, zones, masks and
   fps. It is the cheap gate that decides whether a VLM is asked anything at all.
2. **Change wording with Frigate's per-camera `object_prompts`**, with no code. For example, give the
   Reachy's indoor view and the driveway different prompts. Use profiles for sets that follow the
   time of day or presence.
3. **Put identity and location questions in detection modes:** whose dog it is, which car, whether
   the gate is open. Make one mode per question, on the cameras that need it. Give each mode a
   measured gate and the engines it was validated on, and let its verdict be the notification.
4. **Choose the one loaded engine by the modes the house actually runs.** Cosmos3-Edge v3 is the
   better person detector on the dark indoor cameras, but it cannot tell breeds. Qwen3-VL-2B can.
   Post-training Qwen3-VL on these cameras, the Bernese included, is how one engine does both.
5. **A second engine has to be small, or elsewhere.** The breed service is the working example: a
   co-resident specialist on Frigate's crops, a few hundred MB on the CPU only while it is
   looking. A mode's `url` can also point at a second device. A second VLM does not fit on this
   Orin.

## Live Vision: models, classifiers, Reachy Mini control and Piper speech

Live Vision (`nvr/ui`, `cosmos-edge-ui.service`) captions a browser camera, or the Reachy Mini's
through the bridge (deploy/07 §5), with the shim. The unit binds loopback `:8092`; with the LAN
drop-in from `enable_lan_ui.sh` it serves the LAN on `:8092` and `:8443`, and a LAN request for
`/` on `:8092` is redirected to HTTPS, the only place a browser grants a camera (deploy/07 §3).
Since 2026-10-03 it is the Live UI the board starts at boot, in place of the Live VLM WebUI, and it
also drives the robot: motors, head pose, antennas, onboard apps, volume, and speech through the
robot's speaker. Its model panel switches the shim between the board's Cosmos3-Edge engines and peer
VLMs, or picks one of five image classifiers - two for Pokémon, three for dog breeds - which draw a
saliency overlay over the picture. Its sample sets, one tab each, score a model as the answers land:
189 Pokémon photos, and on this Orin the doggy-daycare frames, where the dog classifiers try to spot
the Bernese ("Models" below).

Its code is the [live-vision-cosmos-demo](https://github.com/nv-asotelo/live-vision-cosmos-demo)
repo at `e84cd37`, plus this repo's model switching and classifiers. `nvr/ui/scripts/engine_backends.py`,
`nvr/reachy/reachy.py` and `nvr/reachy/reachy_mjpeg_bridge.py` are byte-identical to the demo.
`serve_ui.py`, `app.js`, `index.html` and `style.css` carry the additions in "Models"; beyond those,
two files differ only where the demo names its own layout:

| File | Changed from `e84cd37` | Why |
|---|---|---|
| `nvr/ui/scripts/serve_ui.py` | `--shim-service` defaults to `cosmos3-edge-shim`, and its help names `systemd/cosmos3-edge-shim.service` | The demo's shim unit is `live-vision-cosmos-demo-shim`. Read only with `--engine-link` |
| `nvr/ui/scripts/serve_ui.py` | The module docstring, which `--help` prints, the comment on the `reachy.py` import and `EngineSwitcher`'s docstring: `nvr/` paths, a bridge that listens on loopback and the Docker gateway (deploy/07 §1), no `setup-orin.sh` | The demo's paths, its loopback-only bridge and its setup script are not this repo's |
| `nvr/ui/web/index.html` | The image-token help names `max_image_tokens_per_image` in the engine's `visual/config.json`, read when the shim starts (deploy/05 §5). The clocks help names `jetson_clocks` (deploy/02) | The demo fixes the budget at 512 when its `setup-orin.sh` builds the engine, and pins clocks with a unit that script installs |

### Release the shared Reachy bridge

Both Porch Dad's Reachy card and Live Vision now provide **Release bridge** and
**Resume bridge**. This ports the explicit bridge-stop behavior from the standalone
Live Vision setup controls (`151d67f`), retaining Porch Dad's existing robot settings
and its model/classifier controls.

Release stops only this Orin's `reachy-mjpeg-bridge.service`, ending its WebRTC
camera/microphone session for **all local viewers** (including Frigate and Live VLM
WebUI). A browser's **Stop** button only closes that browser's preview. Robot motor,
speaker and onboard-app controls are separate; release does not put the robot to
sleep, reboot it, change its address or stop another computer's connection.

Resume starts the same service with its existing configuration. The UI reports
service startup separately from fresh camera frames; choose **Use Reachy Mini** to
open a Live Vision preview again. Status polls and page reloads never start the
bridge. Release preserves boot enablement: rebooting the Orin or explicitly starting
the service (including Demo mode) can reconnect it.

For human or agent deployment, copy `nvr/reachy/bridge_control.py` along with the
updated Porch Feed and Live Vision files, then restart those web services. The
service account needs only the exact `systemctl start/stop reachy-mjpeg-bridge.service`
sudo commands in `nvr/ui/config/sudoers-live-vision-engines`. Do not replace private
robot settings or rebuild engines. Both interfaces use a process-shared lock, record the requested state privately
for the service-status panel, and verify the unit/PID after an action; a timeout or failed stop must remain a visible
error, not a successful release.

Verify release gives an inactive unit with PID zero, and resume gives a fresh
`/healthz` frame and usable preview. Keep any unrelated Orin untouched. These changes
add no dependency or vendored third-party source; Pollen's stream-client notices are
unchanged. The standalone demo's discovery/installation pipeline is not copied here.

Compared with the UI this repo carried before (`b3f069c`):

* **Auto-speak keeps the selected rate.** It no longer switches to 2x when turned on (`b3f069c`),
  or to 2.5x when captions arrive under 1.5 s apart (`ab7508a`). A caption that lands while the
  last one is still being spoken is still skipped, not queued.
* **Each Reachy capture fetches a fresh `/reachy/still.jpg`** instead of drawing the
  `/reachy/mjpeg` preview, whose decoded picture can stop advancing in the browser while new frames
  keep arriving.
* **Device metrics update 4 times a second**, and GPU load is the peak of 50 ms readings since the
  last update: one reading a second missed inference bursts and showed a busy GPU near 0%.
* **No engine switching to brockone.** [deploy/10](../deploy/10-brockone-live-vision.md) is
  superseded. The registry example keeps only its `cosmos` entry, because a `brockone` entry now
  stops the UI at startup.
* **The bridge changed only in wording, and in one default.** Its push states say "receiver" where
  they said "WebUI" (deploy/07 §5), and `--robot-host` now defaults to `$REACHY_MINI_IP` and refuses
  to start without an address, where it used to default to a fixed one. The unit passes
  `--robot-host`, so nothing changes on the Orin. The bridge's comment on that option, about
  `reachy.env`, describes the demo's unit.

### Robot control and speech: the drop-in

The unit starts without robot flags, and `lan.conf` adds only the listeners. Robot control, speech
and a larger memory cap come from
[`systemd/dropins/cosmos-edge-ui.service.d-zz-live-vision-demo.conf`](systemd/dropins/cosmos-edge-ui.service.d-zz-live-vision-demo.conf),
which repeats `lan.conf`'s command and adds:

| Flags | What they do |
|---|---|
| `--reachy-daemon-url http://<reachy-ip>:8000` | The robot daemon's REST API: motors, pose, apps, volume, sound upload and play. Not the camera: video and audio come from the bridge, relayed under `/reachy/` (`--reachy-url`, default `http://127.0.0.1:8099`) with or without this flag. Needs `requests`, which this interpreter has: porch-feed runs on it too |
| `--piper-bin`, `--piper-model` | Speech: Piper synthesizes on the Orin and the robot's speaker plays it ("Piper" below). Refused without `--reachy-daemon-url`, so with no robot leave out all three |
| `--engine-link /opt/tensorrt-edgellm/models/default`, `--engines-config /home/orin/nvr/ui/config/engines.json` | The model buttons, from [`ui/config/engines.orin.json`](ui/config/engines.orin.json) ("Models" below) |
| `--classifier-url http://127.0.0.1:8094`, `--samples-dir /home/orin/nvr/classifier/samples` | The classifiers' service and the labelled Pokémon sample set ([`classifier/README.md`](classifier/README.md)). Each further `--samples-dir` is one more tab: this Orin adds `/home/orin/nvr/classifier/samples-daycare`, a private set that is not in git ("Sample sets" below) |
| `--services-config /home/orin/nvr/ui/config/services.json` | Adds the classifier service to the page's status bar, from [`ui/config/services.orin.json`](ui/config/services.orin.json) |

Without `--reachy-daemon-url` the robot panel is not shown at all, which is also what a drop-in that
never took effect looks like.

Install it on the Orin from a copy of this repo, after `enable_lan_ui.sh` has issued the cert and
written `lan.conf` (deploy/07 §3), and after the classifiers (`classifier/README.md`):

```bash
install -D -m 0644 nvr/ui/config/engines.orin.json /home/orin/nvr/ui/config/engines.json
install -m 0644 nvr/ui/config/services.orin.json /home/orin/nvr/ui/config/services.json
read -rp "Reachy Mini address: " REACHY_IP     # its LAN IP or hostname - never commit it
DAYCARE=/home/orin/nvr/classifier/samples-daycare   # a private spot set, this Orin only ("Sample sets")
EXTRA=; [ -f "$DAYCARE/manifest.json" ] && EXTRA=" --samples-dir $DAYCARE"
sudo install -d /etc/systemd/system/cosmos-edge-ui.service.d
sed -e "s/<reachy-ip>/$REACHY_IP/g" \
    -e "/^ExecStart=\//s# --samples-dir /home/orin/nvr/classifier/samples # --samples-dir /home/orin/nvr/classifier/samples$EXTRA #" \
    nvr/systemd/dropins/cosmos-edge-ui.service.d-zz-live-vision-demo.conf \
  | sudo tee /etc/systemd/system/cosmos-edge-ui.service.d/zz-live-vision-demo.conf >/dev/null
sudo systemctl daemon-reload
systemctl cat cosmos-edge-ui | grep -c '<reachy-ip>'   # must print 0
systemctl cat cosmos-edge-ui | grep -o -- '--samples-dir [^ ]*'   # one line per sample-set tab
sudo systemctl restart cosmos-edge-ui
```

The `grep` matters: `serve_ui.py` does not check the address at startup, so a drop-in installed
with the placeholder starts normally, and its robot panel just reads "Robot daemon unreachable".
The committed file lists only the Pokémon set; the second `sed` adds the daycare set where it exists,
since that set is not in the repo.

**Why `zz-`.** `lan.conf` and this file both reset `ExecStart=`, systemd applies a unit's drop-ins
in file-name order, and the last command wins. `zz-` sorts after `lan.conf`. The numeric prefix the
other drop-ins here use would sort before it, and then `lan.conf`'s command, which has no robot or
speech flags, wins without an error. A reset replaces the whole command, so this file repeats the
listener flags: change ports here. Re-running `enable_lan_ui.sh` is harmless (it rewrites
`lan.conf` and keeps an existing cert), but its ports have no effect while this file exists.

**Why `MemoryMax=1G`, when the unit says a UI must never compete with the shim.** Piper runs as a
child of `serve_ui.py`, so it lives in the UI's cgroup from the first speak until the UI stops. It
was measured there at 105-135 MB RSS, and with the UI's own memory that is more than the unit's
256M, which the kernel enforces by OOM-killing inside the unit. The tradeoff, stated plainly: 1G is
a ceiling, not a reservation, but it is more than the ~800 MB of headroom measured above, so the cap
no longer guarantees the shim outlives a runaway UI. What the board runs by default changed with it:
this UI instead of the Live VLM WebUI, which measured 212 MB idle to 622 MB peak RSS ("What was
pinned" above). This UI with Piper beside the full NVR has not been measured. Read it while Piper
is warm before tightening the cap: `systemctl show cosmos-edge-ui -p MemoryCurrent`.

**What reaches the LAN.** The `/reachy/` camera and microphone routes need the page's token
(deploy/07 §5), but robot control (`POST /api/reachy/*`) checks only that a request is not
cross-origin, so any LAN client can move the robot or make it speak through Live Vision. The robot's
daemon answers on the LAN at `:8000` anyway, which is how this UI reaches it: this adds a browser
route to the robot, not new reach.

### Models: VLMs and classifiers

The model buttons come from `engines.json`. There are 17 of them, so the panel groups them: one tab
per kind, "Vision-language models" and "Classifiers", each with its count and the active model's tab
marked, and inside a tab the registry's own groups, in its order (`"group"`: Cosmos3-Edge, Other
VLMs, Grounding; Pokémon, Dog breeds). A button drops a leading group name from its registry name ("Skshmjn ViT-B/16",
not "Pokémon · Skshmjn ViT-B/16"), and its tooltip has the full profile. The panel shows the active
model's tab until you pick another, and an entry without a `"group"` falls in "Models" or
"Classifiers". Its Cosmos entries are the command centre's engines
under the command centre's names, and switching one does what the command centre's switch does:
relink `/opt/tensorrt-edgellm/models/default` and restart `cosmos3-edge-shim`. That restart also
restarts `porch-dad.service` (`Requires=`), and both UIs read the same link, so either shows the
other's switch. Live Vision waits up to 180 s for the shim to answer, and puts the previous engine
back if the new one does not. Measured through Live Vision on 2026-10-03, beside the full NVR:

| Button | Engine | Load | First answer, a new image | Free memory after load |
|---|---|---:|---:|---:|
| Cosmos v2 · Fast | Cosmos3-Edge-INT4-v2 | 37-40 s | 1.10 s | ~0.9 GB |
| Cosmos v1 · Medium | Cosmos3-Edge-INT4 | 47 s | 1.12 s | ~1.2 GB |
| Cosmos v3 · Slow (the default) | Cosmos3-Edge-INT4-v3 | 37-48 s | 1.15 s | ~0.65-1.4 GB |
| Cosmos MLP-INT4 | Cosmos3-Edge-RTN-MLP-INT4, weight-stripped: refits from its `model/` | 64 s | 1.58 s | **~0.33 GB** |
| Cosmos FP16 | Cosmos3-Edge | shown, refused | | |

FP16 is listed with `"enabled": false`, so its button says why it cannot be chosen: its engine
files are 4.8 GB and the board had 3.6 GB free with the shim stopped. Answers were a 48-token
description of the same frame, and the free memory moves by hundreds of MB with Frigate's load.

**sudo.** The switch runs `sudo -n ln -sfn <engine path> /opt/tensorrt-edgellm/models/default` and
`sudo -n systemctl restart cosmos3-edge-shim`. On this board `orin` has `NOPASSWD: ALL` from the
image's cloud-init, so nothing more is needed. Elsewhere, install
[`ui/config/sudoers-live-vision-engines`](ui/config/sudoers-live-vision-engines), which allows
exactly those command lines and no others: a wildcard such as `ln -sfn * <link>` also admits extra
options like `-t <dir>`, and with them a root-owned link anywhere. A new engine is a new line there.

**Classifiers** are the registry's `"kind": "classifier"` entries, served by
`pokemon-classifier.service` (loopback `:8094`, [`classifier/README.md`](classifier/README.md)).
Choosing one asks that service to load it, in seconds, and leaves the shim and its engine running,
so porch-dad keeps captioning; choosing a Cosmos engine again frees the classifier's memory, and
returns to the engine the shim still holds without a restart. With a classifier selected, every
frame, upload or sample gets its most likely label - the species or the breed - the next four, and
a saliency overlay, which **Saliency overlay** turns off. None of the classifiers outputs boxes,
points or masks: the overlay is saliency, where the evidence came from, and the page says so under
the picture.

**Sample sets.** The **Sample sets** panel has a tab per `--samples-dir`, in their order; click one, or
move between them with the arrow keys, as in the model panel. The tab opens on the set for the
selected classifier's subject until you pick one.
- **Pokémon (189):** labelled photos. Choose one to show it and run the selected model on it, or run
  the selected classifier over all of them and read its top-1 and top-5 score as the answers land.
  Samples are classified from the server's copy of each file, so the score does not depend on the
  browser's capture settings.
- **Doggy daycare (132), this Orin only:** the frames Reachy filmed of the daycare feed on
  2026-10-04, each labelled with whether the Bernese is in it and where (the labels of
  [`vlmbench/README.md`](vlmbench/README.md), "Bernese mountain dog"). It is a *spot* set
  ([`classifier/make_spot_set.py`](classifier/make_spot_set.py)), and stays out of git. **Show where
  your dog is** draws the labelled box, dashed amber. With a dog-breed classifier selected, every box
  the RT-DETR detector drew in the frame is classified on its own, cropped as the Bernese mode's breed
  scorer crops it, and each box is drawn with its breed; the ones named Bernese turn green. A VLM
  answers a Name the dog breed preset with one name and one box. Run scores the whole set: spotted
  (Bernese, on the right dog), named on the wrong dog, named without a box or point (a VLM's answer
  that says nowhere), missed, and false alarms in frames without it.

Run on the Orin on 2026-10-05, through this page:

| Model | Spotted the Bernese, of the 50 frames with it | Named it on the wrong dog | False alarms, of the 82 frames without it | Run |
|---|---:|---:|---:|---:|
| wesleyacheng ViT-B/16 | 18 | 0 | 1 | 38 s |
| ImageNet ViT-B/16 | 12 | 0 | 0 | 39 s |
| Dog-Breed-120 SigLIP2 | 20 | 1 | 15 | 36 s |
| Cosmos v3, Name the dog breed · box | 0 | 0 | 0 | 126 s |

The detector's boxes covered the Bernese in only 34 of the 50 frames, so no classifier could spot it
in more. Cosmos v3 never answered "Bernese Mountain Dog" in any of the 132 frames, so it spotted
nothing and raised no false alarm. The run scores the name, and a box only once the name is right, so
it does not show which dog Cosmos picked; on the one frame checked by hand it named and boxed a golden
retriever. wesleyacheng is the one to watch with: it spots the Bernese in a third of the frames and
almost never anywhere else.

**Dog-breed classifiers.** The "Dog breeds" group has three: the ImageNet ViT-B/16 the Bernese mode
scores with (1,000 ImageNet classes, 121 of them dogs), wesleyacheng's ViT-B/16 fine-tuned on Stanford
Dogs, and Dog-Breed-120, a SigLIP2-base fine-tuned on undisclosed data (120 breeds each). On the
daycare footage the ImageNet ViT and wesleyacheng ranked the Bernese's crops about equally and
Dog-Breed-120 significantly worse; as a top label, which is what this page shows, wesleyacheng named
the Bernese most reliably ([`vlmbench/README.md`](vlmbench/README.md), "Which breed classifier"). On
the Orin each holds about 265 MB of GPU memory while selected, and its inference takes 15-19 ms
(median) after 10-13 ms of preprocessing ([`classifier/README.md`](classifier/README.md), "Measured on
the Orin").

**Peer VLMs.** Seven more buttons put the models benchmarked against Cosmos3-Edge one click away
([`vlmbench/README.md`](vlmbench/README.md) has the method and the numbers): Qwen3-VL 2B, Qwen3.5 2B,
InternVL3.5 2B and Cosmos-Reason2 2B are TensorRT-Edge-LLM engines built with Cosmos3-Edge's own
recipe; Gemma 4 E2B and Nemotron 3 Nano 4B run in NVIDIA's Jetson llama.cpp container, and
LocateAnything 3B in PyTorch, because none of the three can be a TensorRT engine on this board. Those
three are **proxy engines**: their directory holds a `proxy.json` instead of engine files
([`vlmbench/proxies/`](vlmbench/proxies/) has all three), and the
shim, restarted by the same switch, starts that model's own server and relays the same OpenAI API to
it - so Frigate's GenAI, porch-dad and this page reach every model the same way, and switching away
stops the server before the next model loads. Nemotron 3 Nano 4B reads text only: the shim drops the
image, so it is there for text prompts. LocateAnything answers boxes, not prose: a short phrase
("Bernese mountain dog", `person</c>dog`) is its query, and anything longer falls back to people,
pets and vehicles; its license is non-commercial.

LocateAnything now runs in llama.cpp (PR #24749, Q4_K_M), with its adapter (`la_server.py
--upstream`) in front. Beside the NVR, a 960x540 frame pushed it past 3 GB and the kernel ran out of
memory, so its button says "Needs demo mode": clicking it warns that the NVR will stop, then loads
it at full resolution (Demo mode, above). Gemma 4 E2B was greyed out until it ran on a newer
llama.cpp. That build reads its
per-layer embedding table from disk on demand (`--lazy-mode`), which NVIDIA's Jetson image
predates; `vlmbench/build_llama_upstream.sh` builds it on the workstation, and it runs inside the
same image.

**Name presets.** Four quick presets, after the Live VLM WebUI ones, ask for a Pokémon's species or
a dog's breed, and a box or a point. Cosmos3-Edge answers with JSON - `{"name": ..., "bbox_2d": [x1, y1, x2, y2]}` or
`"point_2d": [x, y]`, normalized 0-1000 over the picture it was sent - which the page parses
tolerantly (first value of a repeated key, fences, lists and cut-off answers; checked against 378 real
answers) and draws over the picture, the caption reading "Pikachu · 72%". The percentage is the
model's own probability for the name it wrote: the presets decode greedily and ask the shim for
per-token logprobs (`logprobs`, `top_logprobs`; nvr/shim, `47ca180`), and the name's tokens are
multiplied. It is a probability of that exact spelling, not calibrated. The presets are worded from a
measured comparison of eleven prompts on the 189 samples; Cosmos3-Edge names the species on 35 of them
(18.5%; 33 of the 120 Gen-1 images, 2 of the other 69). Through the page's own captures a run scored
32/189, its probability averaging 60% on right answers and 10% on wrong ones. The two dog-breed
presets were added on 2026-10-05 in the same form and were not compared that way; on a Commons photo
of a Bernese, Cosmos v3 answered `{"name": "Bernese Mountain Dog", "bbox_2d": [0, 195, 998, 948]}`, at
74%. A preset scores a sample set about its own subject: the Pokémon presets the Pokémon set, the
dog-breed presets the daycare set.

A classifier answers the presets for its own subject - the registry's `"subject"`, `"pokemon"` or
`"dog"` - with its top label and a box or point derived from its saliency map, drawn dashed and
labelled as an estimate (classifier/README.md, "What the overlay shows"). For the dog-breed ViTs the
page says the mark shows where the classifier looked, which need not be the dog: on the daycare
footage it landed on a dog no more often than the frame's centre. Dog-Breed-120's map is nearly flat,
so it gets no mark. The other presets are disabled while a classifier is selected, and selecting one
swaps a preset for the other subject to its own, box for box and point for point.

### Piper

Live Vision's Piper is separate from porch-feed's, which
[deploy/07 §9](../deploy/07-reachy-homeassistant-and-orin-changes.md) covers along with why Piper at
all. It is an older release (v1.2.0 against porch-feed's 2023.11.14-2), with another voice and its
own directory, kept warm by `serve_ui.py` the same way (`--json-input`, `--length_scale 0.85`).
v1.2.0 is the release the demo pins; its aarch64 asset is `piper_arm64.tar.gz`, and
`piper_linux_aarch64.tar.gz` is a 404 under that tag. Neither file is in git (a 25 MB bundle, a
63 MB voice). As `orin`, on the Orin:

```bash
curl -fsSL -o /tmp/piper_arm64.tar.gz \
  https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_arm64.tar.gz
echo "34b298f6b3e55b55e81f05c6157310f9ec4df3fdd3d73e4c85eb80e218c54d2c  /tmp/piper_arm64.tar.gz" | sha256sum -c
tar -xzf /tmp/piper_arm64.tar.gz -C /home/orin/nvr   # -> /home/orin/nvr/piper/piper, own onnxruntime + espeak-ng
for ext in onnx onnx.json; do
  curl -fsSL -o /home/orin/nvr/piper/en_US-ljspeech-medium.$ext \
    https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ljspeech/medium/en_US-ljspeech-medium.$ext
done
```

Rates other than 1x run the WAV through `sox tempo`; without sox every rate quietly plays at 1x
(`sudo apt install sox`). `en_US-ljspeech-medium` is the voice the rates were tuned against by ear:
intelligible up to 2.5x, mumbling at 3x (`cb9cdfd`). Its model card says it was trained from
scratch on the public-domain LJ Speech dataset; read a substitute voice's model card first, because
some Piper voices carry non-commercial licenses.

### At boot, and on demand

Since 2026-10-03 Live Vision starts at boot and the Live VLM WebUI does not ("What survives a
reboot" above has the whole set). Both are boot-follows-intent services in the command centre
(deploy/07 §8): ON there also enables the unit at boot, and OFF disables it. Setting the two there,
Live Vision ON and the WebUI OFF, makes the recorded intent agree with the boot state; OFF is
offered only while a service is running. Set with `systemctl` instead, the boot state is the same,
but where an older recorded intent says otherwise the command centre's cards show the disagreement,
and `scripts/reachy_smoke.py` warns on the boot state and, while the WebUI's recorded intent is
still ON, fails `webui push (reachy)`, until it is set there.

The WebUI is still installed. `sudo systemctl start live-vlm-webui-fork` runs it until the next
reboot; ON in the command centre also brings it back at every boot until it is turned OFF there.
`reachy-mjpeg-bridge` keeps its `--push-url` to the WebUI's `reachy` session: with the WebUI off
that target is usually down, which the bridge treats as a state, not an error (`push.state`
`receiver not reachable (stopped?)`, probed every 10 s), and it resumes by itself once the WebUI
accepts frames again (deploy/07 §5).

### Verified, and what was not

Checked 2026-10-03 on the Orin, through Live Vision's own API with the drop-in installed:
`POST /api/reachy/action/center` answered `{"ok": true, "message": "centred"}`, and
`POST /api/reachy/speak` answered `spoke in 3.21s (synth 2.77s, play 0.45s)`, the first utterance
after a start, so the synthesis time includes Piper's cold load. Piper ran at 105-135 MB RSS in the
unit's cgroup. The Orin keeps the UI it ran before (branch reachy-feed's) as
`/home/orin/nvr/ui.bak-reachy-feed-82775ab`, and the bridge it ran before as
`/home/orin/nvr/reachy/reachy_mjpeg_bridge.py.bak-d86c1ba`.

Model switching was checked through the same API: FP16 refused at once with its reason; v3 to v2
in 39.7 s and back in 37.2 s, each followed by a caption of a test frame through
`/v1/chat/completions`, with `porch-dad` restarted and active after each.

**Not verified:** speech at rates other than 1x, and whether sox is installed there; the 2026-10-03
boot set across a reboot; and this UI with Piper beside the full NVR, or beside the WebUI.
