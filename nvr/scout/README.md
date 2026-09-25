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

### Settled against hardware: BOTH `/CoreNode/jpg` and `/CoreNode/h264` exist

The vendor README documents `/CoreNode/h264`; every community project reads `/CoreNode/jpg`.
Measured on a live Scout, **both are published simultaneously** — so neither source was wrong:

| topic | type | rate | bandwidth | payload |
|---|---|---|---|---|
| `/CoreNode/jpg` | `roller_eye/frame` | 6.7 fps | 756 kB/s | JPEG SOI at offset 41, 1920×1080, `type=1` |
| `/CoreNode/h264` | `roller_eye/frame` | 23.8 fps | 196 kB/s | Annex-B `00 00 00 01`, `type=0` |

This bridge reads `jpg`, which is why it costs ~48 MB and does no transcoding. The startup check
below is therefore belt-and-braces rather than load-bearing, but it stays: firmware versions
differ, and a topic that silently never publishes is the hardest failure to diagnose.

**Worth revisiting:** `h264` carries 3.5× the framerate at a quarter of the bandwidth, and the
pipeline downstream re-encodes MJPEG to h264 in go2rtc anyway — so passing h264 straight through
would skip a decode and an encode. That is a design change, not a config change, and it is not
done here.

### The one that actually bit: the robot advertises itself by hostname

The ROS master returns `http://linaro-alip:11311/` and hands out peers as
`['TCPROS', 'linaro-alip', 50218]`. rospy uses those strings verbatim, and nothing on this network
resolves `linaro-alip` — no DNS, no mDNS. Without a hosts entry the subscription is accepted and
then no frame ever arrives, which looks exactly like a broken camera. Hence `extra_hosts` in
`docker-compose.yml`. Verified both ways in this image: without it, `socket.gaierror`; with it,
frames immediately.

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

## Confirmed against a live Scout (2026-09-17)

Robot at `192.168.7.6`, hostname `linaro-alip`, SSH banner `OpenSSH_7.4p1 Debian-10+deb9u7`
(Debian 9, the stock Linaro ALIP image). Only **two ports open: 22 and 11311** — a full scan of
23/80/443/554/1883/5000/8000-8100/8554/9090/37020 found nothing else, which independently confirms
there is no RTSP, no ONVIF, no web UI and no MQTT on the robot.

* `roller_eye/frame` md5 on the robot is `bce5a3441e8f21e02d2b9d7ce432bea2` — **identical** to the
  one this container generates, and the robot's `message_definition` is byte-for-byte the vendored
  `msg/frame.msg`. The md5 caveat above is satisfied on this firmware.
* `/cmd_vel` is stock `geometry_msgs/Twist`, md5 `9f195f881246fdfa2798d1d3eebca84a`.
* The camera works: a frame pulled through this bridge decoded as a valid 1920×1080 JFIF.
* Bridge cost against the real robot: **47.8 MB RSS, ~2% CPU** — close to the 36.5 MB measured
  against the fixture, the difference being 1920×1080 frames rather than 160×120.

**The cloud login path**, from the node graph rather than speculation: nothing listens for the
vendor, so it must be robot-initiated outbound. `/CloudNode` publishes `status` and offers
`cloud_cmd_send`, a persistent outbound command channel; `/AppNode` publishes `sock_status` and
`p2p_status` with `P2P_AV_PLAYING/STOP/ERROR` constants — a signalling socket plus a peer-to-peer
A/V channel, idle when no phone is viewing — and offers `/sys/get_userid`, so the robot holds a
cloud account binding. `/RTMPNode` and `/S3Node` offer `rtmp_start`/`rtmp_stop` and `s3_setting`.
None of that is needed for this bridge, which talks only to the local ROS master.

## Incident: `cloud_node`/`app_node`/`s3_node` went missing (2026-09-17), and the rule that follows

Confirmed running above, same robot, same day: `/CloudNode`, `/AppNode`, `/S3Node`. By
2026-09-25 they were gone — not disabled, not crashed, **absent from disk**. `dpkg -V roller-eye`
on both Scouts flags identical missing files: `app_node`, `cloud_node`, `s3_node`,
`upgrader_node`, four `test_*` diagnostic binaries, and `/var/roller_eye/config/p2p_user` (the
per-account P2P credential) — `dpkg` still believes they are installed, so this was never an
`apt`/`dpkg` transaction; something deleted files dpkg had already placed.
`/opt/ros/melodic/lib/roller_eye/`'s mtime and `AppNode.log`'s last line (mid-response to a
normal app `getOtaStatus` poll) both stop at the same second, `2026-09-17 16:36:35`, on
`robot_room`'s Scout; the
first-floor Scout shows the same pattern at `18:07` the same day. Root had a shell open on the
robot around that time running `git pull` / `./build.sh` in `~/ros/src/roller_eye` - a fuller
source checkout than the one this project vendors (see below), now itself deleted, so the exact
command that did it cannot be recovered.

**The vendored `nvr/scout/roller_eye/` in this repo did not do this.** Its `CMakeLists.txt` says
outright "nothing is built for the robot" - it exists only to generate this container's own
Python message bindings (`frame.msg`, `status.msg`) for the Dockerfile above, and is never
installed on the Scout itself. Whatever touched `/opt/ros/melodic/lib/roller_eye/` on the robot
was a **separate, fuller checkout that lived directly on the robot's own filesystem**, almost
certainly pulled there to get `.srv` definitions this project needed and didn't yet have a safer
way to reach. `roller_eye_srv.py` (hand-rolled TCPROS, no rebuild required) exists specifically
so that need never has to touch the robot's install again.

**The consequence turned out worse than "no app control": `app_node` is not just what lets the
app control an already-paired robot, it appears to be what answers the pairing/binding handshake
itself**, even in the robot's own local setup AP mode - a phone can join that AP at the WiFi
layer with `app_node` absent, but the Moorebot app's bind step then has nothing on the robot side
to answer it. A restart cannot fix a missing binary; re-pairing cannot fix a missing binary either
if pairing itself depends on that binary. Restoration requires either a genuine vendor `.deb` for
this exact build (`/var/roller_eye/config/version`: `01_HW32_MO200A_020150`) or Moorebot's own
recovery path - both out of reach once the vendor is out of business and no backup was taken.

**The rule, going forward, for any Scout with root access:** before running anything that touches
`/opt/ros/melodic/lib/roller_eye/`, `/var/roller_eye/config/`, or any `dpkg -i`/rebuild-and-install
step on the robot itself -

1. `dpkg -V roller-eye` first, and keep the output. That is the cheapest possible tripwire and
   the only reason this incident was even diagnosable after the fact.
2. `tar czf` a backup of `/opt/ros/melodic/lib/roller_eye/` and `/var/roller_eye/config/` to the
   Orin (`scp` it off the robot entirely) before touching either directory, every time, no
   exceptions - this is the one step that would have made this incident a non-event.
3. Never install or rebuild anything under `/opt/ros/melodic/lib/roller_eye/` on the robot to get
   a `.srv`/`.msg` definition. Extend `roller_eye_srv.py` by hand instead (see its own docstring
   for why TCPROS-by-hand is not the workaround it looks like - it is the version of this that
   cannot delete a vendor binary).
4. If a factory-fresh or recently-purchased-used unit is ever reachable with root **and its
   official app still binds successfully**, back up its `/opt/ros/melodic/lib/roller_eye/` and
   `/var/roller_eye/config/` immediately, before any other work touches it. That backup is the
   only way a future loss on another unit stays recoverable - copy the binaries across (matching
   `/var/roller_eye/config/version`) rather than depending on the app a second time.

## Recovering a used (eBay) unit to factory defaults, and its role: test bench, not a third feed

This third Scout is **not** going into the fleet as a third live camera - that would mean a third
bridge container, a third Frigate camera, a third `scouts:` entry, its own go2rtc stream, real
ongoing plumbing. Given how the other two ended up with `app_node`/`cloud_node`/etc. missing with
no clean way to trace exactly what happened, the decision instead is to keep this one as a
dedicated **test bench**: the one Scout root/porch-dad changes get tried against first, so a
mistake costs a bench unit, not a robot that's live in Frigate and the command centre. It only
joins the fleet later, deliberately, if that's decided separately - nothing here should assume it
will.

Moorebot is out of business, so whatever the official app can still do today is the last chance
to do it - there is no vendor to ask later. A used unit almost certainly still shows a previous
owner's account binding.

1. **Before anything else touches this robot: get root, run `dpkg -V roller-eye`, and back up
   `/opt/ros/melodic/lib/roller_eye/` and `/var/roller_eye/config/` to the Orin.** This is rule 4
   above, and it is the actual point of doing this now while a working unit exists - a clean copy
   of `app_node`/`cloud_node`/`s3_node`/`upgrader_node`/the `test_*` tools/`p2p_user`'s *shape*
   (not another unit's credential, but confirmation of what the intact file set and permissions
   look like) is worth more than the robot working stand-alone, because it is the one thing that
   could restore the two already-broken Scouts without needing the app at all - **if**
   `/var/roller_eye/config/version` matches theirs (`01_HW32_MO200A_020150`). Check that version
   file first; if it differs, the binaries may not be ABI-compatible and copying them across is a
   separate judgement call, not an automatic win.
2. Try unbinding from the previous owner's account through the app's normal "remove device" flow
   first, if the eBay seller is reachable and willing - this is the clean path and leaves
   `app_node` etc. untouched.
3. If that is not available, the robot's own physical factory-reset (button or button-combo on
   the unit itself - **not verified against this hardware from this session**, since the only two
   units with root access both already had `app_node` missing before this was investigated;
   check the unit's underside/manual, or Moorebot's last-published documentation if archived
   anywhere, e.g. the Wayback Machine) should force it back into its own setup AP regardless of
   prior binding, the same AP mode the app was just tried against on the other robot.
4. Bind it to your own account through the app while on that AP. Confirm in the app that video,
   drive and status all work - that confirms `app_node`/`cloud_node` are genuinely functional, not
   just present.
5. Only after step 1's backup exists and step 4 confirms the unit is fully app-controllable should
   any root/porch-dad integration work start on this unit, and every step of it follows the rule
   above: back up again immediately before the first thing that touches
   `/opt/ros/melodic/lib/roller_eye/`.

## Still unknown

Three of the four original unknowns are now answered above: the ROS master **is** reachable on the
LAN with no SSH or developer mode (port 11311, open), the md5 **does** match, and the vendor's own
services do **not** hold the camera exclusively — `/CloudNode`, `/AppNode` and the rest were all
running while this bridge pulled frames.

What is still open:

* Whether the robot stays on the network while docked or idle, and whether the ROS master survives
  a firmware update.
* Whether driving it works. **`/cmd_vel` has been read, never written** — no motion command has
  been sent to this robot. The axis convention (`+linear.y` forward) comes from the vendor's
  firmware source, not from watching it move, so the first drive should be a small one in clear
  space with a hand near the robot.

The bridge is written to fail visibly rather than silently on all of these: `/healthz` reports
`ros_connected`, the reason it is not connected, and `stale_s` for the case where frames stop
without the connection dropping.
