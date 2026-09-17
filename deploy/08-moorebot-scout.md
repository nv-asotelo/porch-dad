# 08 — Moorebot Scout: prepared, not deployed

Support for the Moorebot Scout is built and tested but **nothing is running**. There was no Scout
on the network when this was written, so this documents what exists, what was proved without the
hardware, and the short list of things that can only be settled with a robot in front of you.

Detail lives in [`../nvr/scout/README.md`](../nvr/scout/README.md); this is the deployment view.

---

## What was added

| | |
|---|---|
| Bridge | [`../nvr/scout/scout_mjpeg_bridge.py`](../nvr/scout/scout_mjpeg_bridge.py) — ROS1 camera → MJPEG, plus a `/cmd_vel` relay |
| Container | [`../nvr/scout/Dockerfile`](../nvr/scout/Dockerfile), `ros:noetic-ros-base` (arm64) |
| Messages | [`../nvr/scout/roller_eye/`](../nvr/scout/roller_eye/) — msgs-only catkin package, **MIT**, vendored from the robot vendor |
| Test fixture | [`../nvr/scout/fake_scout.py`](../nvr/scout/fake_scout.py) — a fake robot, so this is testable with no hardware |
| Compose | `scout-bridge` service in [`../nvr/docker-compose.yml`](../nvr/docker-compose.yml), behind `profiles: ["scout"]` |
| Frigate | `scout` go2rtc stream + camera in [`../nvr/frigate/config.yml`](../nvr/frigate/config.yml), shipped `enabled: false` |
| Command centre | `scout` service entry and a health link in [`../nvr/feed/config.yaml`](../nvr/feed/config.yaml) |

## Everything is off by default, in three independent ways

Deliberate, on a board that was sitting at ~628 MB available while this was written:

1. The compose service is behind a profile, so `docker compose up` does not start it.
2. The Frigate camera ships `enabled: false`.
3. The command centre lists it as a service you can start, and will report it down until you do.

Starting it is three commands, and all three are a decision:

```bash
cd nvr
docker compose --profile scout build scout-bridge
docker compose --profile scout up -d scout-bridge
# then enable the `scout` camera from the command centre
```

## Cost

Measured on the Orin against the fake robot, not estimated:

| | |
|---|---|
| Bridge RSS with frames flowing | **36.5 MB** (`mem_limit` 128 M) |
| CPU | 0.3% |
| Image on disk | ~3 GB, on a 722 GB-free NVMe |

The earlier estimate for this work was 250–400 MB. That assumed a transcode; there is none, because
the robot publishes frames it has already JPEG-encoded and the bridge only copies bytes. Two Scouts
would be roughly 75 MB, which this board can afford — enabling their Frigate *cameras* is the
expensive part, not running the bridges.

## What was proved without a robot

Using `fake_scout.py` on the Jetson: the message class builds and imports; a real JPEG survives the
round trip and comes out of `/mjpeg` renderable at 5 fps; `/healthz` answers and `/drive` refuses
safely while no robot is connected; a 1.0 s drive command publishes exactly 10 messages and then
stops; a 99 m/s request clamps to 0.30 m/s.

Also established, from the vendor's own firmware source rather than from documentation:

* **`+linear.y` is forward, `+linear.x` is strafe** — the opposite of the usual ROS1 base.
  `avoidObstacle()` clamps only `y`, against the forward-facing ToF sensor.
* **The robot runs its own motion dead-man.** If `cmd_vel` stops arriving it zeroes velocity and
  then cuts motor power. So stopping the bridge, killing the container or losing the network all
  stop the robot, and nothing here has to be trusted to send a final zero.
* **The Scout has no RTSP or ONVIF server**, so there is no simpler integration being missed.

## What still needs the hardware

* Whether the ROS master is reachable from the LAN without SSH or a developer mode.
* Whether the vendor's own AI service holds the camera exclusively.
* **Whether the camera topic is `/CoreNode/jpg` or `/CoreNode/h264`.** The vendor README documents
  h264; every working community project uses jpg. If a firmware publishes only h264 this needs a
  transcode, which is a different design and a different RAM budget — not a config change. The
  bridge detects this case at startup and says so in `/healthz` instead of showing black.
* Whether the shipping firmware's `frame.msg` hashes the same as the public repo's — if not, the
  subscription connects and silently delivers nothing.
* Whether the robot stays on the network while docked.
* Name resolution: the firmware pins no `ROS_IP`, so its nodes advertise by hostname. See the
  `extra_hosts` note in `docker-compose.yml`; this is the most likely first failure.

## A licensing note

`kingardor/noir-rover` (GPL-3.0) solves this problem already and was read closely, but no code from
it is here — porch-dad is Apache-2.0, and that project has no container anyway, being macOS/Homebrew
native. What was taken from it is knowledge: topic names and the axis convention, both re-verified
against the vendor's firmware. The only vendored artefact is `frame.msg`, which comes from
`Pilot-Labs-Dev/Scout-open-source` under **MIT** with its licence retained alongside it.

## Live-system state

**None.** Nothing was installed, enabled or started on the Orin for this, and the running Frigate
config was not touched — the `scout` stream and camera exist only in the repo's
`nvr/frigate/config.yml` and reach the box on the next deployment of that file. The image
`porch-scout-bridge:latest` was built there during testing and the test containers were removed.
