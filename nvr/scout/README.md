# Moorebot Scout → Frigate

A container that subscribes to a Moorebot Scout's ROS1 camera topic, re-serves the frames as
MJPEG for Frigate, and relays drive commands back to the robot.

**Nothing here is running.** It is built and tested but ships switched off, behind a compose
profile, with its Frigate camera disabled. At the time of writing there was no Scout on the
network; everything below was verified against a fake robot (`fake_scout.py`) and against the
vendor's own firmware source.

---

## Why this exists rather than an RTSP URL

Frigate can only ingest what ffmpeg can open. The Scout publishes its camera on the ROS1 topic
`/CoreNode/jpg`, which ffmpeg cannot open. This is the same problem the Reachy Mini posed, and it
has the same shape of answer — a small bridge on the Jetson that speaks the robot's transport and
re-serves MJPEG on the Docker gateway, with nothing outside this box in the path.

**The Scout has no RTSP or ONVIF server**, so there is no simpler route that skips all of this.
That was checked rather than assumed, because if it were false none of this code would be needed:
the vendor firmware contains no RTSP or ONVIF implementation (the only apparent matches are a
motor variable spelled `rtSp`), and its one streaming library is `srs_librtmp.cpp` — RTMP *push*
to the vendor's cloud, which is not something an NVR can pull from. Independently, every community
project that displays Scout video — `go-scout`, `remotv-node-for-scout`, `noir-rover` — reads
`/CoreNode/jpg` rather than opening a stream URL.

It is cheaper than the Reachy bridge in one important way: the robot has **already JPEG-encoded**
the frames, and they arrive as raw bytes in the message's `data` field. Serving them is a copy.
There is no numpy, no OpenCV and no image library in this container at all.

## Why not kingardor/noir-rover

[`kingardor/noir-rover`](https://github.com/kingardor/noir-rover) already bridges the Scout and was
the obvious candidate. It was rejected for four independent reasons:

* **Licence.** It is GPL-3.0; porch-dad is Apache-2.0. Vendoring it would relicense this tree.
* **There is nothing to lift.** It has no Dockerfile. It runs natively on macOS via Homebrew
  micromamba (`/opt/homebrew/opt/micromamba`, `brew install ffmpeg espeak-ng`, MLX models).
  Containerising it for arm64 meant writing a Dockerfile from scratch regardless.
* **Weight.** Its bridge pulls in FastAPI and Redis to support an LLM and knowledge-graph stack we
  are not deploying, on a board with a few hundred MB free.
* **Size of the real problem.** What we needed was ~200 lines matching a file already in this repo.

What that project was genuinely worth was *knowledge* — the topic names and the axis convention.
Those are facts, and they were re-verified against the vendor's firmware rather than trusted.

`roller_eye/msg/frame.msg` is vendored from
[`Pilot-Labs-Dev/Scout-open-source`](https://github.com/Pilot-Labs-Dev/Scout-open-source), which is
**MIT** — not from noir-rover. See `roller_eye/LICENSE`.

---

## The facts this depends on

Each was verified from a primary source, because each is the kind of thing that fails silently.

| Fact | Where it was verified |
|---|---|
| Camera is `/CoreNode/jpg`, type `roller_eye/frame`, `data` is raw JPEG | `frame.msg`; `type` field 1 = JPG, 0 = H264 |
| **`+linear.y` is FORWARD, `+linear.x` is strafe** | firmware `motor_node.cpp` — `avoidObstacle()` clamps *only* `y`, against the forward-facing ToF reading |
| The robot stops itself if commands stop arriving | firmware `start_motor_daemon()` — clears a keep-alive flag, waits `STOP_TIME_THRESHOLD`, then `setX_Y_Wz(0,0,0)` and cuts motor power |
| Firmware clamps every axis and ignores commands when motors are disabled | `checkVolecity(..., MACC_MAX_SPEED_*)`, `mMotorEnable` gate |
| The generated message MD5 is `bce5a3441e8f21e02d2b9d7ce432bea2` | built in this container |

Two of these change the design:

**The axis convention is not the ROS default.** Standard ROS1 bases drive on `linear.x`. This one
drives on `linear.y`. Getting it backwards would strafe the robot sideways — in the one direction
its obstacle sensor is not watching.

**The dead-man already exists, in firmware.** So the bridge sustains motion by *repeating* the
command at 10 Hz for a bounded duration, and stopping simply means stopping publishing. A crashed
bridge, a killed container, a pulled network cable and a deliberate stop are all the same event to
the robot, and all of them stop it. Nothing in this container has to be trusted to send a final
zero.

### The MD5 caveat

ROS refuses a connection when the two ends hash a message definition differently. `frame.msg` is
copied byte for byte from the vendor repo and must stay that way — reformatting it, even
whitespace, changes the hash and breaks the camera with a confusing error. If the robot's firmware
ships a different `frame.msg` than the public repo, the symptom will be a subscription that
connects and never delivers; compare hashes before assuming the bridge is at fault.

---

## Running it

Build and start (it is not started by a plain `docker compose up`):

```bash
cd nvr
docker compose --profile scout build scout-bridge
docker compose --profile scout up -d scout-bridge
```

Set `ROS_MASTER_URI` to the robot and `ROS_IP` to this Jetson in `docker-compose.yml` first. Both
matter: `ROS_IP` is the address this node *advertises*, so if it is wrong the robot will accept the
subscription and then be unable to route frames back.

Then, to actually see it in Frigate, enable the `scout` camera — it ships `enabled: false`.

| Endpoint | |
|---|---|
| `http://172.17.0.1:8098/mjpeg` | the stream go2rtc consumes |
| `http://172.17.0.1:8098/still.jpg` | one frame |
| `http://172.17.0.1:8098/healthz` | `ros_connected`, `stale_s`, `live`, `driving` |
| `POST /drive` | `{"x": strafe, "y": forward, "yaw": rad/s, "duration": s}` |
| `POST /stop` | |

Clamps: 0.3 m/s linear, 1.0 rad/s angular, 3 s per request. These are a courtesy bound against a
typo — the real limits are enforced by the robot.

## Testing without a robot

`fake_scout.py` publishes real JPEGs on the camera topic and prints whatever arrives on
`/cmd_vel`, labelled by axis. That last part is the point: it is how you confirm "forward" moves
`y` without watching a robot drive into a wall.

```bash
# three terminals, or -d
docker run -d --name scout-master --network host --entrypoint /bin/bash porch-scout-bridge:latest \
  -lc '. /opt/ros/noetic/setup.bash && exec roscore -p 11399'
docker run -d --name scout-fake --network host \
  -v "$PWD/fake_scout.py:/opt/scout/fake_scout.py:ro" \
  --entrypoint /bin/bash porch-scout-bridge:latest \
  -lc '. /opt/ros/noetic/setup.bash && . /catkin_ws/devel/setup.bash && \
       ROS_MASTER_URI=http://127.0.0.1:11399 exec python3 -u /opt/scout/fake_scout.py'
docker run -d --name scout-test --network host porch-scout-bridge:latest \
  --master-uri http://127.0.0.1:11399 --listen 127.0.0.1 --listen-port 8098

curl -s http://127.0.0.1:8098/healthz
curl -s -X POST http://127.0.0.1:8098/drive -H 'Content-Type: application/json' \
     -d '{"y":0.2,"duration":1.0}'
docker logs scout-fake | grep cmd_vel     # must show forward(y)=+0.200
docker rm -f scout-master scout-fake scout-test
```

Measured on the Orin with that fixture running:

| | |
|---|---|
| Bridge RSS, frames flowing | **36.5 MB** (`mem_limit` is 128 M) |
| Bridge CPU | 0.3% |
| A 1.0 s drive command | published exactly 10 messages, then stopped |
| 99 m/s request | clamped to 0.30 m/s |
| No robot present | `/healthz` answers, `/still.jpg` 503s, `/drive` refuses |

The 37 MB is worth stating plainly because the earlier estimate for this work was 250–400 MB. That
estimate assumed a transcode. There is no transcode.

## Why ROS Noetic, and why that is less alarming than it sounds

The Scout's firmware is ROS1 and speaks TCPROS, so ROS1 is not a preference. Noetic went EOL in
May 2025, but the base image's apt source is `snapshots.ros.org/noetic/final` — a frozen post-EOL
snapshot rather than a rolling repo — and the only package this build installs is `python3-aiohttp`
from Ubuntu. Built and run successfully on this board in 2026.

## Still unknown until a Scout is on the network

* Whether the robot exposes its ROS master to the LAN without SSH access or a developer mode, and
  what the user must do on the robot first. The firmware pins no `ROS_MASTER_URI`, `ROS_IP` or
  `ROS_HOSTNAME` anywhere, so it runs with ROS defaults — `roscore` on `0.0.0.0:11311`, nodes
  advertising by hostname. That is encouraging for reachability but means **name resolution is the
  first thing to suspect**: see the `extra_hosts` note in `docker-compose.yml`.
* Whether the vendor's own AI/streaming services hold the camera exclusively, so that a second
  subscriber gets no frames.
* Whether the shipping firmware's `frame.msg` hashes the same as the public repo's.
* Whether the robot stays on the network while docked.

The bridge is written to fail visibly rather than silently on all of these: `/healthz` reports
`ros_connected`, the reason it is not connected, and `stale_s` for the case where frames stop
without the connection dropping.
