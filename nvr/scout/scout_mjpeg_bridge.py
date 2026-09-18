#!/usr/bin/env python3
"""Serve a Moorebot Scout's camera as MJPEG, and relay drive commands, from the Jetson.

Same shape and the same reasons as reachy_mjpeg_bridge.py next door: Frigate can only ingest what
ffmpeg can open, and the Scout's camera is a ROS1 topic, which ffmpeg cannot open. This subscribes
to that topic and re-serves the frames as MJPEG on the Docker gateway so Frigate/go2rtc can read
them locally, with nothing outside this box in the path.

Two things make this cheaper than the Reachy bridge:

* /CoreNode/jpg carries frames that the robot has ALREADY JPEG-encoded, in the `data` field of a
  roller_eye/frame message. Serving them is a copy, not a decode-and-re-encode, so there is no
  numpy, no OpenCV and no image library in this process at all.
* The robot's firmware runs the motion watchdog for us (see drive() below), so this holds no
  safety-critical timer of its own.

Why our own bridge rather than running kingardor/noir-rover, which already does this: that project
is GPL-3.0 (porch-dad is Apache-2.0), is built for macOS via Homebrew micromamba with no container
of any kind, and pulls in Redis and FastAPI to support an LLM stack we are not deploying. What we
actually needed from it was knowledge, not code - the topic names and the axis convention below -
and those are facts, verified against the vendor's own firmware source. Only the message
definition is vendored, and that comes from the vendor's MIT repo, not from noir-rover.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import threading
import time
from collections import deque

from aiohttp import web

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG = logging.getLogger("scout-mjpeg")

CAMERA_TOPIC = "/CoreNode/jpg"
CMD_VEL_TOPIC = "/cmd_vel"

# Forward-facing time-of-flight rangefinder, stock sensor_msgs/Range, 0.03-2.0 m.
#
# This is the obstacle sensor, and it is the one that gets believed. A vision model was tried for
# the same job and is not fit for it: asked whether the robot could drive straight ahead with a
# large dog lying 25 cm in front, Cosmos3-Edge answered "CLEAR" three times out of three, while
# this sensor read 0.252 m. The model is excellent at saying WHAT is there ("a large dog is lying
# on the floor", every time) and unreliable at judging what to do about it, so perception and
# judgement are split: the rangefinder decides, the model narrates.
TOF_TOPIC = "/SensorNode/tof"

# A reading older than this is not trustworthy as an obstacle check.
TOF_STALE_S = 3.0

# Battery, roller_eye/status: an int32[] where [0] is the charge state and [1] is percent.
# Constants from status.msg: CHARGING=0, UNCHARGE=1, FULL=2, UNKNOWN=3.
BATTERY_TOPIC = "/SensorNode/simple_battery_status"
BATTERY_STATE = {0: "charging", 1: "discharging", 2: "full", 3: "unknown"}

# The vendor's own README documents the video stream as /CoreNode/h264 and never mentions
# /CoreNode/jpg, while every working community project reads /CoreNode/jpg. Both appear to exist on
# current firmware, with jpg being the already-encoded preview, but the documentation conflict is
# unresolved and the robot was not available to settle it.
#
# Settled on hardware: the firmware publishes BOTH, so neither source was wrong.
#   /CoreNode/jpg    6.7 fps   756 kB/s   1920x1080 JPEG, type=1
#   /CoreNode/h264  23.8 fps   196 kB/s   Annex-B, type=0
#
# Both are subscribed. They are used for different jobs and the difference is measured, not
# aesthetic. Serving JPEG to Frigate means go2rtc must ENCODE h264, which cost 33.8% of a CPU on
# this board (no NVENC, so libx264 in software) and pushed detector inference from 100 ms to
# 178 ms, making every other camera drop frames. Passing the robot's own h264 through means go2rtc
# only remuxes. The jpg topic stays because stills are still wanted: the command centre's camera
# card and the Cosmos description both need a single frame, and pulling one from h264 would mean
# decoding - exactly the cost being avoided.
H264_TOPIC = "/CoreNode/h264"

# roller_eye/frame multiplexes codecs on one topic and tags each with `type`.
FRAME_TYPE_H264 = 0
FRAME_TYPE_JPG = 1

# How much recent h264 to keep so a client that connects mid-stream has something to start on.
# A decoder cannot begin at an arbitrary NAL: it needs the parameter sets and then an IDR. Those
# are cached separately and replayed to every new client, which is what makes joining work at all.
H264_RING = 512

# Restart the subscription after this long with a byte-identical frame. Learned from the Reachy
# bridge: a stalled stream keeps the frame counter climbing and keeps returning HTTP 200 while
# showing a frozen picture, and only comparing the bytes catches it.
STALE_AFTER = 12.0

# Motion. The robot's own MotorNode clamps every axis to its MACC_MAX_SPEED_* limits and refuses
# commands entirely unless its motors are enabled, so these are a courtesy bound to keep a typo in
# a curl command from requesting a lurch, not the real safety layer.
MAX_LINEAR = 0.3     # m/s
# The firmware's MACC_MAX is ~7 rad/s; this ceiling is what the UI is allowed to ask for. Raised
# from 1.0 to give the TRACKED Scout enough authority for an in-place tank turn: spinning treads
# scrub sideways against the floor and 1.0 rad/s barely broke that static friction. Mecanum still
# only ever asks for ~0.7, so this headroom changes nothing for it.
MAX_ANGULAR = 2.0    # rad/s
MAX_DURATION = 3.0   # s, per request
CMD_HZ = 10.0


class Bridge:
    def __init__(self, fps: float):
        self.interval = 1.0 / fps if fps > 0 else 0.2
        self.latest: bytes | None = None
        self._lock = threading.Lock()
        self._frames = 0
        self._last_digest: str | None = None
        self._digest_since = 0.0
        self._resubs = 0
        self._ros_ready = False
        self._ros_error: str | None = None
        self._sub = None
        self._pub = None
        self._twist_cls = None
        self._drive_until = 0.0
        self._drive_cmd = (0.0, 0.0, 0.0)
        # Latest rangefinder reading and when it arrived. None means nothing within its 2 m range.
        self._tof_m: float | None = None
        self._tof_at = 0.0
        self._batt_pct: int | None = None
        self._batt_state: str | None = None
        self._batt_at = 0.0
        self._pub_recreates = 0
        self._rospy = None
        self._range_cls = None
        self._status_cls = None
        self._tof_sub = None
        self._batt_sub = None
        self._sub_resubs = 0
        # h264 passthrough. `_h264_ring` holds recent access units keyed by a monotonic sequence
        # so each client can track its own position without a per-client queue; `_h264_params`
        # holds the most recent SPS+PPS and `_h264_idr` the most recent keyframe, which together
        # are what a joining decoder needs before anything else will decode.
        self._h264_ring: deque = deque(maxlen=H264_RING)
        self._h264_seq = 0
        self._h264_params = b""
        self._h264_idr: bytes = b""
        self._h264_frames = 0
        self._h264_at = 0.0

    # ---------------------------------------------------------------------- ROS
    def start_ros(self, master_uri: str) -> None:
        """Bring ROS up on a background thread.

        Deliberately not on the main thread: rospy.init_node blocks until it can reach the master,
        and the Scout is a battery robot that spends much of its life asleep or docked. Blocking
        here would mean the bridge never binds its port, so /healthz could not report WHY it was
        down - which is the one thing worth knowing when the robot is missing.
        """
        threading.Thread(target=self._ros_main, args=(master_uri,), daemon=True).start()

    def _ros_main(self, master_uri: str) -> None:
        try:
            import rospy
            from geometry_msgs.msg import Twist
            from sensor_msgs.msg import Range
            from roller_eye.msg import frame as RollerFrame
            from roller_eye.msg import status as RollerStatus
        except ImportError as e:
            self._ros_error = (f"ROS imports failed ({e}); the roller_eye messages are probably "
                               f"not built - see nvr/scout/README.md")
            _LOG.error(self._ros_error)
            return

        try:
            _LOG.info("connecting to ROS master at %s", master_uri)
            # init_node does not raise when the master is absent - it logs "will keep trying" and
            # blocks indefinitely. Without this line /healthz would sit at ros_connected=false with
            # error=null, which reads like "fine, no data" rather than "the robot is not there".
            self._ros_error = f"connecting to {master_uri} (robot asleep, docked or off-network?)"
            rospy.init_node("porch_scout_bridge", anonymous=True, disable_signals=True)
            self._twist_cls = Twist
            self._pub = rospy.Publisher(CMD_VEL_TOPIC, Twist, queue_size=1)
            self._sub = rospy.Subscriber(CAMERA_TOPIC, RollerFrame, self._on_frame, queue_size=1)
            # Kept as handles and the classes stored, so the watchdog can resubscribe them: when
            # the app takes control it restarts the robot's SensorNode, and rospy does not reliably
            # re-establish these, leaving battery and ToF frozen.
            self._rospy = rospy
            self._range_cls, self._status_cls = Range, RollerStatus
            self._tof_sub = rospy.Subscriber(TOF_TOPIC, Range, self._on_tof, queue_size=1)
            self._batt_sub = rospy.Subscriber(BATTERY_TOPIC, RollerStatus, self._on_battery, queue_size=1)
            # Separate subscription, same message type: h264 for Frigate, jpg for stills.
            rospy.Subscriber(H264_TOPIC, RollerFrame, self._on_h264, queue_size=4)
            self._ros_ready = True
            self._ros_error = self._check_camera_topic(rospy)
            _LOG.info("subscribed to %s, publishing %s", CAMERA_TOPIC, CMD_VEL_TOPIC)
        except Exception as e:
            self._ros_error = f"ROS init failed: {e}"
            _LOG.error(self._ros_error)
            return

        threading.Thread(target=self._drive_loop, daemon=True).start()
        threading.Thread(target=self._stale_watchdog, args=(rospy, RollerFrame),
                         daemon=True).start()
        threading.Thread(target=self._pub_watchdog, args=(rospy,), daemon=True).start()
        rospy.spin()

    def _pub_connections(self) -> int:
        try:
            return self._pub.get_num_connections() if self._pub else 0
        except Exception:
            return 0

    def _resubscribe_stale_sensors(self) -> None:
        """Re-establish the battery/ToF subscriptions if they have gone quiet.

        A restart of the robot's SensorNode (which the app triggers when it takes control) breaks
        these subscriptions and rospy does not reliably recover them, freezing battery and ToF.
        The topics publish at ~0.6-2 Hz, so more than 15 s of silence means the connection is dead,
        not merely idle.
        """
        if self._rospy is None:
            return
        now = time.monotonic()
        if self._batt_at and now - self._batt_at > 15 and self._status_cls is not None:
            try:
                if self._batt_sub is not None:
                    self._batt_sub.unregister()
                self._batt_sub = self._rospy.Subscriber(
                    BATTERY_TOPIC, self._status_cls, self._on_battery, queue_size=1)
                self._batt_at = now  # reset so it does not churn every pass while genuinely gone
                self._sub_resubs += 1
                _LOG.warning("battery subscription was stale - resubscribed")
            except Exception as e:
                _LOG.error("battery resubscribe failed: %s", e)
        if self._tof_at and now - self._tof_at > 15 and self._range_cls is not None:
            try:
                if self._tof_sub is not None:
                    self._tof_sub.unregister()
                self._tof_sub = self._rospy.Subscriber(
                    TOF_TOPIC, self._range_cls, self._on_tof, queue_size=1)
                self._tof_at = now
                self._sub_resubs += 1
                _LOG.warning("ToF subscription was stale - resubscribed")
            except Exception as e:
                _LOG.error("tof resubscribe failed: %s", e)

    def _pub_watchdog(self, rospy) -> None:
        """Recreate the /cmd_vel publisher if it loses its subscriber.

        The robot's own nodes restart - on low battery, when the app takes control, on a firmware
        hiccup - and when they do, this publisher drops off the ROS master and every drive command
        silently goes nowhere while /drive still answers "moving". That is exactly how the robot
        became undriveable in the field. MotorNode is always subscribed on a healthy robot, so a
        sustained zero-subscriber count means our registration is stale; recreating the publisher
        re-registers it with the master and MotorNode reconnects.
        """
        missing = 0.0
        while not rospy.is_shutdown():
            time.sleep(3)
            pub = self._pub
            if pub is None or self._twist_cls is None:
                continue
            try:
                connected = pub.get_num_connections() > 0
            except Exception:
                continue
            self._resubscribe_stale_sensors()
            if connected:
                missing = 0.0
                continue
            missing += 3
            if missing >= 9:
                _LOG.warning("/cmd_vel has had no subscriber for ~%.0fs - recreating publisher", missing)
                try:
                    pub.unregister()
                except Exception:
                    pass
                try:
                    self._pub = rospy.Publisher(CMD_VEL_TOPIC, self._twist_cls, queue_size=1)
                    self._pub_recreates += 1
                    missing = 0.0
                except Exception as e:
                    _LOG.error("publisher recreate failed: %s", e)

    def _check_camera_topic(self, rospy) -> str | None:
        """Say plainly whether the robot publishes the topic we just subscribed to.

        Subscribing to a topic nobody publishes is not an error in ROS - it succeeds and waits
        forever. Given the vendor/community disagreement about jpg vs h264 (see the constants
        above), that silence is the single most likely way this ends up showing a black camera, so
        it is worth one query to turn it into a sentence.
        """
        try:
            published = {name for name, _ in rospy.get_published_topics()}
        except Exception:
            return None                      # master busy; not worth failing over
        if CAMERA_TOPIC in published:
            return None
        if H264_TOPIC in published:
            return (f"the robot publishes {H264_TOPIC} but NOT {CAMERA_TOPIC}; this bridge streams "
                    f"pre-encoded JPEG and cannot use h264 without a transcode - see README")
        cams = sorted(t for t in published if "CoreNode" in t) or sorted(published)[:12]
        return (f"{CAMERA_TOPIC} is not being published; topics seen: {', '.join(cams) or 'none'}")

    def _stale_watchdog(self, rospy, frame_cls) -> None:
        """Re-subscribe when the picture stops changing.

        A ROS subscription can go quiet without erroring - the robot's CoreNode restarts, or its
        camera is grabbed by the vendor's own AI service - and nothing raises. The Reachy bridge
        learned this the expensive way: the feed looked healthy from every angle except the
        picture, which was of an empty chair while someone sat in it. Comparing frame bytes is the
        only signal that catches it, so having computed it, act on it.
        """
        while not rospy.is_shutdown():
            time.sleep(2.0)
            if not self._digest_since or self.latest is None:
                continue
            if time.monotonic() - self._digest_since <= STALE_AFTER:
                continue
            _LOG.warning("identical frame for >%.0fs - resubscribing to %s",
                         STALE_AFTER, CAMERA_TOPIC)
            try:
                if self._sub is not None:
                    self._sub.unregister()
                self._sub = rospy.Subscriber(CAMERA_TOPIC, frame_cls, self._on_frame,
                                             queue_size=1)
                self._resubs += 1
                # Reset the clock, else every pass re-fires until a genuinely new frame lands.
                self._digest_since = time.monotonic()
                self._last_digest = None
            except Exception as e:
                _LOG.error("resubscribe failed: %s", e)

    @staticmethod
    def _nal_types(buf: bytes) -> set[int]:
        """NAL unit types present in an Annex-B buffer.

        Only the type nibble is read - this never parses or rewrites the bitstream, because the
        whole point is to hand the robot's bytes to ffmpeg untouched.
        """
        types, i, n = set(), 0, len(buf)
        while True:
            j = buf.find(b"\x00\x00\x01", i)
            if j < 0 or j + 3 >= n:
                break
            types.add(buf[j + 3] & 0x1F)
            i = j + 3
        return types

    def _on_h264(self, msg) -> None:
        """Buffer an h264 access unit for passthrough. No decode, no re-encode, no parsing."""
        if getattr(msg, "type", FRAME_TYPE_H264) != FRAME_TYPE_H264:
            return
        data = bytes(msg.data)
        if not data:
            return
        kinds = self._nal_types(data)
        with self._lock:
            # 7 = SPS, 8 = PPS. Cache whenever they appear; they rarely change but a client that
            # connects later has no other way to get them.
            if 7 in kinds or 8 in kinds:
                self._h264_params = data if (5 not in kinds) else self._h264_params or data
            if 5 in kinds:                      # 5 = IDR, the only safe place to start decoding
                self._h264_idr = data
            self._h264_seq += 1
            self._h264_ring.append((self._h264_seq, data))
            self._h264_frames += 1
            self._h264_at = time.monotonic()

    def _on_battery(self, msg) -> None:
        """roller_eye/status.status is [charge_state, percent, ...]."""
        st = list(getattr(msg, "status", []) or [])
        if len(st) >= 2:
            self._batt_state = BATTERY_STATE.get(int(st[0]), "unknown")
            self._batt_pct = int(st[1])
            self._batt_at = time.monotonic()

    def _on_tof(self, msg) -> None:
        """Keep the newest range reading.

        The sensor reports "nothing in range" as inf or -inf rather than a number, so those are
        normalised to None. That is a meaningful value here - it means clear to 2 m - and is not
        the same as never having had a reading, which `tof_age_s` distinguishes.
        """
        try:
            r = float(msg.range)
        except (TypeError, ValueError, AttributeError):
            return
        self._tof_m = None if (r != r or r in (float("inf"), float("-inf"))) else r
        self._tof_at = time.monotonic()

    def tof(self) -> tuple[float | None, float | None]:
        """(metres, age_seconds); metres is None when nothing is within range."""
        if not self._tof_at:
            return None, None
        return self._tof_m, round(time.monotonic() - self._tof_at, 1)

    def _on_frame(self, msg) -> None:
        if getattr(msg, "type", FRAME_TYPE_JPG) != FRAME_TYPE_JPG:
            return
        data = bytes(msg.data)
        if not data:
            return
        digest = hashlib.md5(data).hexdigest()
        now = time.monotonic()
        if digest != self._last_digest:
            self._last_digest = digest
            self._digest_since = now
        with self._lock:
            self.latest = data
            self._frames += 1

    # ---------------------------------------------------------------------- motion
    def drive(self, x: float, y: float, yaw: float, duration: float) -> tuple[bool, str]:
        """Request motion for a bounded time.

        AXES ARE NOT THE ROS CONVENTION. On this robot +linear.y is FORWARD and +linear.x is a
        sideways strafe - the opposite of the usual ROS1 base. That is not folklore: the vendor's
        MotorNode runs its time-of-flight obstacle avoidance by clamping `y` alone
        (avoidObstacle() compares y against a ceiling derived from the forward ToF reading), which
        only makes sense if y is the direction of travel. Swapping these would drive the robot
        sideways into whatever its obstacle sensor is not looking at.

        Motion is sustained by REPEATING the command, not by latching it. MotorNode's daemon
        thread clears a keep-alive flag, waits STOP_TIME_THRESHOLD, and if no further cmd_vel
        arrived it calls setX_Y_Wz(0,0,0) and eventually cuts motor power. So the dead-man already
        exists in firmware and is free: to stop the robot this bridge simply stops publishing, and
        a bridge crash, a dropped network or a killed container all stop it too. Nothing here needs
        to be trusted to send a final zero.
        """
        if not self._ros_ready:
            return False, self._ros_error or "not connected to the robot's ROS master"
        x = max(-MAX_LINEAR, min(MAX_LINEAR, float(x)))
        y = max(-MAX_LINEAR, min(MAX_LINEAR, float(y)))
        yaw = max(-MAX_ANGULAR, min(MAX_ANGULAR, float(yaw)))
        duration = max(0.0, min(MAX_DURATION, float(duration)))
        self._drive_cmd = (x, y, yaw)
        self._drive_until = time.monotonic() + duration
        return True, f"strafe {x:+.2f} forward {y:+.2f} yaw {yaw:+.2f} m/s for {duration:.1f}s"

    def stop(self) -> tuple[bool, str]:
        self._drive_until = 0.0
        self._drive_cmd = (0.0, 0.0, 0.0)
        if not self._ros_ready:
            return True, "not connected; the robot is not being driven by this bridge"
        # One explicit zero so the robot stops now rather than at the end of the firmware's
        # watchdog interval. Not required for safety - see drive() - just for responsiveness.
        self._publish(0.0, 0.0, 0.0)
        return True, "stopped"

    def _publish(self, x: float, y: float, yaw: float) -> None:
        pub, twist_cls = self._pub, self._twist_cls
        if pub is None or twist_cls is None:
            return
        try:
            msg = twist_cls()
            msg.linear.x, msg.linear.y, msg.angular.z = x, y, yaw
            pub.publish(msg)
        except Exception as e:
            _LOG.warning("publish failed: %s", e)

    def _drive_loop(self) -> None:
        """Repeat the current command while it is still in date. See drive()."""
        period = 1.0 / CMD_HZ
        while True:
            if time.monotonic() < self._drive_until:
                self._publish(*self._drive_cmd)
            time.sleep(period)

    # ---------------------------------------------------------------------- HTTP
    def _stale_s(self) -> float | None:
        return round(time.monotonic() - self._digest_since, 1) if self._digest_since else None

    async def mjpeg(self, request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(
            status=200,
            headers={"Content-Type": "multipart/x-mixed-replace; boundary=frame",
                     "Cache-Control": "no-store"},
        )
        await resp.prepare(request)
        try:
            while True:
                with self._lock:
                    frame = self.latest
                if frame:
                    await resp.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(frame)}\r\n\r\n".encode()
                        + frame
                        + b"\r\n"
                    )
                await asyncio.sleep(self.interval)
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        return resp

    async def h264(self, request: web.Request) -> web.StreamResponse:
        """The robot's own h264, byte for byte, as a raw Annex-B elementary stream.

        ffmpeg reads this with `-f h264 -i http://...`, and go2rtc with `#video=copy` then only
        remuxes it - no encode. That is the entire point: encoding this 1920x1080 stream in
        software cost 33.8% of a CPU and made every other Frigate camera drop frames.

        A new client is sent the cached parameter sets and the most recent IDR before anything
        else. Without that a decoder sits on undecodable inter frames until the next keyframe,
        which on this robot is about a second - long enough to look broken.
        """
        resp = web.StreamResponse(
            status=200,
            headers={"Content-Type": "video/h264", "Cache-Control": "no-store"},
        )
        await resp.prepare(request)

        with self._lock:
            params, idr, cursor = self._h264_params, self._h264_idr, self._h264_seq
        try:
            if params:
                await resp.write(params)
            if idr and idr != params:
                await resp.write(idr)
            while True:
                with self._lock:
                    pending = [(s, d) for s, d in self._h264_ring if s > cursor]
                if pending:
                    for s, d in pending:
                        await resp.write(d)
                        cursor = s
                else:
                    # Poll rather than signal: the producer is a ROS callback on another thread,
                    # and a short sleep is simpler and safer than waking the loop from it.
                    await asyncio.sleep(0.02)
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        return resp

    async def still(self, request: web.Request) -> web.Response:
        with self._lock:
            frame = self.latest
        if not frame:
            raise web.HTTPServiceUnavailable(text="no frame yet")
        return web.Response(body=frame, content_type="image/jpeg")

    async def health(self, request: web.Request) -> web.Response:
        # `frames` counts callbacks, not distinct pictures, so `stale_s` is the honest signal.
        stale = self._stale_s()
        tof_m, tof_age = self.tof()
        with self._lock:
            has = self.latest is not None
            frames = self._frames
        return web.json_response({
            "ros_connected": self._ros_ready,
            "error": self._ros_error,
            "frames": frames,
            "has_frame": has,
            "stale_s": stale,
            "resubscribes": self._resubs,
            "live": bool(has and (stale is None or stale < STALE_AFTER)),
            "driving": time.monotonic() < self._drive_until,
            # The obstacle sensor. null means nothing within its 2 m range, which is "clear" -
            # not "unknown". tof_age_s distinguishes those: a null with a fresh age is clear, a
            # null with no age at all means no reading has ever arrived.
            "tof_m": (round(tof_m, 3) if tof_m is not None else None),
            "tof_age_s": tof_age,
            "tof_fresh": bool(tof_age is not None and tof_age < TOF_STALE_S),
            "battery_pct": self._batt_pct,
            "battery_state": self._batt_state,
            "battery_fresh": bool(self._batt_at and (time.monotonic() - self._batt_at) < 30.0),
            "cmd_vel_connected": bool(self._pub is not None and self._pub_connections()),
            "pub_recreates": self._pub_recreates,
            "sub_resubs": self._sub_resubs,
            "h264_frames": self._h264_frames,
            "h264_ready": bool(self._h264_idr),
            "h264_age_s": (round(time.monotonic() - self._h264_at, 1) if self._h264_at else None),
        })

    async def http_drive(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError):
            body = {}
        ok, msg = self.drive(body.get("x", 0.0), body.get("y", 0.0),
                             body.get("yaw", 0.0), body.get("duration", 0.6))
        return web.json_response({"ok": ok, "message": msg}, status=200 if ok else 503)

    async def http_stop(self, request: web.Request) -> web.Response:
        ok, msg = self.stop()
        return web.json_response({"ok": ok, "message": msg})


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--master-uri", default=None,
                   help="ROS master on the robot, e.g. http://192.168.6.170:11311 "
                        "(defaults to $ROS_MASTER_URI)")
    p.add_argument("--listen", default="172.17.0.1",
                   help="bind address; the docker gateway keeps the camera off the LAN")
    p.add_argument("--listen-port", type=int, default=8098)
    p.add_argument("--fps", type=float, default=5.0,
                   help="rate frames are served at; the robot's publish rate is unaffected")
    args = p.parse_args()

    master = args.master_uri or os.environ.get("ROS_MASTER_URI")
    if not master:
        raise SystemExit("no ROS master: pass --master-uri or set ROS_MASTER_URI")
    os.environ["ROS_MASTER_URI"] = master

    bridge = Bridge(args.fps)
    bridge.start_ros(master)

    app = web.Application()
    app.router.add_get("/mjpeg", bridge.mjpeg)
    app.router.add_get("/h264", bridge.h264)
    app.router.add_get("/still.jpg", bridge.still)
    app.router.add_get("/healthz", bridge.health)
    app.router.add_post("/drive", bridge.http_drive)
    app.router.add_post("/stop", bridge.http_stop)

    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, args.listen, args.listen_port).start()
    _LOG.info("MJPEG on http://%s:%d/mjpeg (ROS master %s)",
              args.listen, args.listen_port, master)
    await asyncio.Event().wait()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
