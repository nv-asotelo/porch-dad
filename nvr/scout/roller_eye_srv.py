"""Call roller_eye ROS1 services on the Moorebot Scout over raw TCPROS.

Why this exists: the vendored roller_eye package in nvr/scout/roller_eye generates only
roller_eye/frame, so rospy.ServiceProxy cannot talk to any of the robot's services without
adding every .srv and rebuilding the bridge image. TCPROS service calls are simple enough
to do by hand - a length-prefixed header exchange, then a length-prefixed request body - so
this module speaks it directly with nothing but the standard library.

Signatures below are quoted from the vendor source, MIT licensed:
https://github.com/Pilot-Labs-Dev/Scout-open-source/tree/main/roller_eye/srv
Every md5 here was checked against the live robot on 2026-09-17 and matched the GitHub
source exactly, so the shipped firmware and the public source agree on these types.

The robot advertises services as rosrpc://linaro-alip:PORT, a hostname that does not
resolve off the robot, so the host is rewritten to the master's IP.
"""

from __future__ import annotations

import socket
import struct
import urllib.parse
import xmlrpc.client

DEFAULT_MASTER = "http://192.168.7.6:11311"
CALLER_ID = "/porch_scout_srv"

# roller_eye/record.msg type constants
RECORD_TYPE_ALL = 0
RECORD_TYPE_SNAPSHOT = 1
RECORD_TYPE_RECORD = 2
RECORD_TYPE_THUMB = 3
RECORD_TYPE_SCHED_RECORD = 4


class ServiceError(RuntimeError):
    """The service was reachable but returned failure (ok byte 0)."""


def _enc_header(fields: dict[str, str]) -> bytes:
    body = b"".join(
        struct.pack("<I", len(f)) + f
        for f in ((k + "=" + v).encode() for k, v in fields.items())
    )
    return struct.pack("<I", len(body)) + body


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("connection closed mid-message")
        buf += chunk
    return buf


def _read_header(sock: socket.socket) -> dict[str, str]:
    (n,) = struct.unpack("<I", _recv_exactly(sock, 4))
    buf = _recv_exactly(sock, n)
    out, i = {}, 0
    while i < len(buf):
        (flen,) = struct.unpack("<I", buf[i : i + 4])
        i += 4
        key, _, val = buf[i : i + flen].decode(errors="replace").partition("=")
        out[key] = val
        i += flen
    return out


def _lookup(service: str, master: str) -> tuple[str, int]:
    code, msg, uri = xmlrpc.client.ServerProxy(master).lookupService(CALLER_ID, service)
    if code != 1:
        raise LookupError(f"{service}: {msg}")
    host, _, port = uri[len("rosrpc://") :].rstrip("/").rpartition(":")
    try:
        socket.getaddrinfo(host, None)
    except socket.gaierror:
        host = urllib.parse.urlparse(master).hostname or host
    return host, int(port)


def call(service: str, payload: bytes = b"", master: str = DEFAULT_MASTER,
         timeout: float = 10.0) -> bytes:
    """Call `service` with an already-serialised request, returning the response body."""
    host, port = _lookup(service, master)
    with socket.create_connection((host, port), timeout) as sock:
        sock.settimeout(timeout)
        # probe=1 asks for the type header without invoking the service, which is how we
        # learn the md5 the server expects.
        sock.sendall(_enc_header(
            {"callerid": CALLER_ID, "service": service, "md5sum": "*", "probe": "1"}))
        md5 = _read_header(sock).get("md5sum", "*")
    with socket.create_connection((host, port), timeout) as sock:
        sock.settimeout(timeout)
        sock.sendall(_enc_header(
            {"callerid": CALLER_ID, "service": service, "md5sum": md5, "persistent": "0"}))
        _read_header(sock)
        sock.sendall(struct.pack("<I", len(payload)) + payload)
        ok = _recv_exactly(sock, 1)[0]
        (n,) = struct.unpack("<I", _recv_exactly(sock, 4))
        body = _recv_exactly(sock, n) if n else b""
    if not ok:
        raise ServiceError(f"{service} returned false: {body.decode(errors='replace')!r}")
    return body


def _lookup_node(node: str, master: str) -> str:
    code, msg, uri = xmlrpc.client.ServerProxy(master).lookupNode(CALLER_ID, node)
    if code != 1:
        raise LookupError(f"{node}: {msg}")
    # Same hostname gotcha as _lookup(): the robot advertises itself as linaro-alip, which does
    # not resolve off the robot, so rewrite to the master's own host if that's where we got it.
    parsed = urllib.parse.urlparse(uri)
    try:
        socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror:
        real_host = urllib.parse.urlparse(master).hostname or parsed.hostname
        uri = uri.replace(parsed.hostname, real_host, 1)
    return uri


def publish_once(topic: str, node: str, msg_type: str, md5sum: str, payload: bytes,
                  master: str = DEFAULT_MASTER, timeout: float = 10.0) -> None:
    """Publish one message on `topic` directly to `node`'s existing subscription, bypassing the
    master's own registerPublisher/publisherUpdate dance entirely.

    Every other function here is a service CALL, where we are the client and the robot is the
    server that already knows how to answer - `call()`'s two-socket probe-then-invoke shape only
    makes sense for that direction. Publishing is the other way around: we are the one deciding
    what to send, and topic delivery in ROS1 normally goes through the master registering us as a
    publisher, subscribers being told we exist via a publisherUpdate callback to THEIR node, and
    only then connecting to us - which needs us to run an XML-RPC server nothing else here does.
    Skipped entirely: `node` is already a known, already-subscribed listener (e.g. CoreNode is
    always subscribed to `testBackup` - see alg_backing_up.cpp's constructor), so this calls that
    node's own `requestTopic` XML-RPC method directly, exactly as if we were a legitimate
    publisher it already knew about, then does the raw TCPROS publisher handshake by hand.
    """
    node_uri = _lookup_node(node, master)
    code, msg, proto = xmlrpc.client.ServerProxy(node_uri).requestTopic(
        CALLER_ID, topic, [["TCPROS"]])
    if code != 1:
        raise LookupError(f"{node} requestTopic({topic}): {msg}")
    _proto_name, host, port = proto
    # Same hostname gotcha, third time over: requestTopic's own response also advertises
    # linaro-alip, not an address reachable off the robot.
    try:
        socket.getaddrinfo(host, None)
    except socket.gaierror:
        host = urllib.parse.urlparse(master).hostname or host
    with socket.create_connection((host, int(port)), timeout) as sock:
        sock.settimeout(timeout)
        sock.sendall(_enc_header({
            "callerid": CALLER_ID, "topic": topic, "type": msg_type,
            "md5sum": md5sum, "latching": "0",
        }))
        _read_header(sock)   # the subscriber's own header - not inspected, just drained
        sock.sendall(struct.pack("<I", len(payload)) + payload)


# roller_eye/testBackup uses the well-known std_msgs/Int8 wire type (one signed byte), not a
# roller_eye message - its md5sum is the fixed, standard one for that core ROS type.
STD_MSGS_INT8_MD5 = "27ffa0c9c4b8fb8492252bcad9e5c57b"


def test_backup(value: int, master: str = DEFAULT_MASTER) -> None:
    """CoreNode/testBackup: std_msgs/Int8. Not a real .srv - a vendor debug hook, but the only
    known way to trigger the camera-guided dock approach directly (see alg_backing_up.cpp's
    onTestBackup): 0 does nothing, 2 calls doBackup() (the actual approach-and-dock routine) as
    long as canBackup() thinks the last-seen dock pose is close enough, 4 runs detection only.
    There is no ROS-level way to ask it to search from scratch if the dock is not already in
    view - point the robot at the dock (or near where it last was) before calling this with 2."""
    if value not in (0, 2, 4):
        raise ValueError("value must be 0 (noop), 2 (do backup) or 4 (detect only)")
    publish_once("/CoreNode/testBackup", "/CoreNode", "std_msgs/Int8", STD_MSGS_INT8_MD5,
                 struct.pack("<b", value), master)


def _str(buf: bytes, i: int) -> tuple[str, int]:
    (n,) = struct.unpack("<I", buf[i : i + 4])
    i += 4
    return buf[i : i + n].decode(errors="replace"), i + n


def snapshot(master: str = DEFAULT_MASTER) -> str:
    """One JPEG to the robot's own storage. Returns the record id.

    roller_eye/record_start: int8 type, int8 mode, uint32 duration, int32 count
                         --- int8 status, string id
    count=1 fires a 10ms one-shot timer; the file lands in
    /userdata/roller_eye/media_files/<timestamp>_<id>-1.jpg and is indexed in MEDIA_FILES.DB.
    """
    body = call("/RecorderAgentNode/record_start",
                struct.pack("<bbIi", RECORD_TYPE_SNAPSHOT, 0, 0, 1), master)
    return _str(body, 1)[0]


def record_start(duration_s: int, count: int = 1, master: str = DEFAULT_MASTER) -> str:
    """Record `count` segments of `duration_s` seconds each to the robot's storage.

    duration must be >= 3 (VIDEO_MIN_DURATION). count=0 means UNLIMITED - it keeps rolling
    new segments until record_stop, so never pass 0 from a UI button.
    """
    if duration_s < 3:
        raise ValueError("duration must be >= 3 seconds (VIDEO_MIN_DURATION)")
    if count < 1:
        raise ValueError("count=0 records forever; pass an explicit segment count")
    body = call("/RecorderAgentNode/record_start",
                struct.pack("<bbIi", RECORD_TYPE_RECORD, 0, duration_s, count), master)
    return _str(body, 1)[0]


def record_stop(master: str = DEFAULT_MASTER) -> int:
    """roller_eye/record_stop: int8 type --- int8 status"""
    return struct.unpack("<b", call("/RecorderAgentNode/record_stop",
                                    struct.pack("<b", RECORD_TYPE_RECORD), master)[:1])[0]


def record_status(rec_type: int = RECORD_TYPE_RECORD, master: str = DEFAULT_MASTER) -> int:
    """0 = idle, 1 = a task of this type is running."""
    return struct.unpack("<b", call("/RecorderAgentNode/record_get_status",
                                    struct.pack("<b", rec_type), master)[:1])[0]


def record_file_num(rec_type: int, master: str = DEFAULT_MASTER) -> int:
    return struct.unpack("<i", call("/RecorderAgentNode/record_get_file_num",
                                    struct.pack("<b", rec_type), master)[:4])[0]


def record_files(rec_type: int, start: int = 0, size: int = 20,
                 master: str = DEFAULT_MASTER) -> list[dict]:
    """roller_eye/record_get_files: int8 type, string id, int32 start, int32 size
                                --- record[] files

    roller_eye/record.msg: string id, string name, uint32 dur, int8 type, time create,
    uint32 size.  Note `name` comes back as the full on-robot path, not a bare filename.
    """
    req = struct.pack("<b", rec_type) + struct.pack("<I", 0) + struct.pack("<ii", start, size)
    body = call("/RecorderAgentNode/record_get_files", req, master)
    (count,) = struct.unpack("<I", body[:4])
    i, files = 4, []
    for _ in range(count):
        rid, i = _str(body, i)
        name, i = _str(body, i)
        dur, ftype = struct.unpack("<Ib", body[i : i + 5])
        i += 5
        secs, _nsecs = struct.unpack("<II", body[i : i + 8])
        i += 8
        (size_bytes,) = struct.unpack("<I", body[i : i + 4])
        i += 4
        files.append({"id": rid, "path": name, "dur": dur, "type": ftype,
                      "created": secs, "size": size_bytes})
    return files


def file_path(record_id: str, master: str = DEFAULT_MASTER) -> str:
    """roller_eye/record_get_file_path: string id --- string path"""
    req = struct.pack("<I", len(record_id)) + record_id.encode()
    return _str(call("/RecorderAgentNode/record_get_file_path", req, master), 0)[0]


def night_get(master: str = DEFAULT_MASTER) -> tuple[int, int]:
    """roller_eye/night_get: (empty) --- int8 isNight, int32 brightness (IR led, 0-99)."""
    body = call("/CoreNode/night_get", b"", master)
    return struct.unpack("<bi", body[:5])


def adjust_light(cmd: int, master: str = DEFAULT_MASTER) -> None:
    """roller_eye/adjust_ligth (vendor's spelling): int32 cmd --- (empty)

    This is the infrared illuminator for night vision, not a white lamp and not the status
    LEDs. The .srv comment claims "0-down,1-up, 3,-max, 4-auto" but media_core_node.cpp
    implements: 0 = dim (10% of current, or 50% below 6), 1 = restore the pre-dim value,
    2 = max (99), anything else = max.

    Raises ServiceError whenever night mode is not active: adjust() opens with
    `if (!mIRActive) return false;`, and mIRActive is set by the ambient light watcher on
    SensorNode/light, not by this call. In daylight there is no way to force the IR on.
    """
    call("/CoreNode/adjust_light", struct.pack("<i", cmd), master)


def led_all_on(master: str = DEFAULT_MASTER) -> None:
    """roller_eye/led_all_on: empty request, empty response (md5 d41d8cd9... = empty).

    Lights all four battery LEDs plus both wifi LEDs - a lamp test / "locate the robot",
    not illumination. There is no led_all_off; ui_node repaints the battery LEDs to the
    real charge level on the next SensorNode/simple_battery_status message.
    """
    call("/led_all_on", b"", master)


def system_event(event: int, master: str = DEFAULT_MASTER) -> None:
    """roller_eye/system_event: int32 event --- (empty)

    SpeakerNode's SoundEffectsMgr shells out to `aplay` for a fixed wav per event id, so
    this is the only ROS-visible way to make the robot emit a sound of its own. Playback is
    gated on the platform config's soundEffect.activate flag, so a successful call does not
    guarantee audible output.

    Valid ids are 0..14. Do NOT send 15: SYSEVT_TRACEDONE indexes s_sound_files[15] but
    that array has only 15 entries (a missing comma merged ids 14 and 15's paths), so 15
    reads off the end of the array.
    """
    if not 0 <= event <= 14:
        raise ValueError("event must be 0..14; 15 reads past the end of s_sound_files")
    call("/system_event", struct.pack("<i", event), master)


def _enc_str(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<I", len(b)) + b


# ---------------------------------------------------------------------------------- navigation
# Investigated 2026-09-25 while looking for a way to stop a Scout auto-seeking its dock on low
# battery and, separately, retrace a marked path back to it. `/CoreNode/nav_cancel` belongs to
# BackingUp (roller_eye/src/nodes/media_core/alg_backing_up.cpp) - the dock-return algorithm
# itself, vision-aligned via the camera for final approach. It is triggered by an Int32 on the
# `backing_up` topic (media_core_node.cpp's `DataPulisher<std_msgs::Int32,BackingUp>`), which
# nothing in the open-source nodes publishes - the trigger is almost certainly app_node or
# cloud_node, both missing on these two robots (see nvr/scout/README.md's incident writeup), so
# the native auto-dock-seek may already be orphaned. Calling nav_cancel here is a cheap, harmless
# belt-and-suspenders stop regardless: BackingUp.cancel() just quits its algorithm and goes
# BACK_UP_INACTIVE if nothing was running.
#
# `/NavPathNode/*` is a separate, general path-record/replay system (roller_eye/src/nodes/
# nav_path_node.cpp - source not fetched, only the .srv shapes, but the naming and BackingUp's
# own use of `roller_eye/track_trace.h` for its final-approach retrace strongly suggest the same
# underlying trace mechanism). nav_path_start begins recording odometry under a name; nav_path_save
# finalizes it; nav_patrol replays a saved path by name. This is the "path trace back to the dock"
# capability - genuinely exposed, no missing binary required.


def nav_cancel_backup(master: str = DEFAULT_MASTER) -> None:
    """roller_eye/nav_cancel: (empty) --- (empty). Stops CoreNode's BackingUp (dock-return) if
    one is in progress. Safe to call speculatively - a no-op if nothing was running."""
    call("/CoreNode/nav_cancel", b"", master)


def nav_cancel_path(master: str = DEFAULT_MASTER) -> None:
    """Same empty nav_cancel shape, but NavPathNode's copy - stops a patrol/path-record in
    progress there. Separate service from nav_cancel_backup; call both to be sure."""
    call("/NavPathNode/nav_cancel", b"", master)


def nav_exit(master: str = DEFAULT_MASTER) -> None:
    """roller_eye/nav_exit: (empty) --- (empty). Exits NavPathNode's nav mode entirely."""
    call("/NavPathNode/nav_exit", b"", master)


def nav_get_status(master: str = DEFAULT_MASTER) -> int:
    """roller_eye/nav_get_status: (empty) --- int32 status."""
    return struct.unpack("<i", call("/NavPathNode/nav_get_status", b"", master)[:4])[0]


def save_tmp_pic_for_start_path(name: str, master: str = DEFAULT_MASTER) -> None:
    """roller_eye/saveTmpPicForStartPath: string name --- (empty). CoreNode, not NavPathNode -
    saves a reference photo at the current position, associated with a path name about to be
    recorded. Call this first, at the dock, before nav_path_start."""
    call("/CoreNode/saveTmpPicForStartPath", _enc_str(name), master)


def nav_path_start(name: str, is_from_out_start: bool = False, master: str = DEFAULT_MASTER) -> None:
    """roller_eye/nav_path_start: int8 isFromOutStart, string name --- (empty). Begins recording
    odometry under `name` - drive the robot (our own /api/scout/{sid}/drive works fine for this)
    from the dock to wherever it should be able to find its way back from, then nav_path_save."""
    call("/NavPathNode/nav_path_start", struct.pack("<b", int(is_from_out_start)) + _enc_str(name), master)


def nav_path_save(name: str, master: str = DEFAULT_MASTER) -> None:
    """roller_eye/nav_path_save: string name --- (empty). Finalizes the path nav_path_start began
    recording. `name` here is documented as matching the one passed to nav_path_start."""
    call("/NavPathNode/nav_path_save", _enc_str(name), master)


def nav_patrol(name: str, is_from_out_start: bool = True, master: str = DEFAULT_MASTER) -> int:
    """roller_eye/nav_patrol: int8 isFromOutStart, string name --- int32 ret. Replays a saved
    path - this is "trace back to the dock". isFromOutStart defaults True here (unlike
    nav_path_start's False): retracing normally starts from wherever the robot currently is,
    not from the path's own recorded start point."""
    body = call("/NavPathNode/nav_patrol",
                struct.pack("<b", int(is_from_out_start)) + _enc_str(name), master)
    return struct.unpack("<i", body[:4])[0]


def nav_patrol_stop(master: str = DEFAULT_MASTER) -> None:
    """roller_eye/nav_patrol_stop: (empty) --- (empty)."""
    call("/NavPathNode/nav_patrol_stop", b"", master)


def nav_delete_path(names: list[str], master: str = DEFAULT_MASTER) -> None:
    """roller_eye/nav_delete_path: string[] names --- (empty). ROS array wire format: uint32
    count, then each element length-prefixed the same as a lone string."""
    body = struct.pack("<I", len(names)) + b"".join(_enc_str(n) for n in names)
    call("/NavPathNode/nav_delete_path", body, master)


SYSEVT = {
    0: "power_on.wav", 1: "power_down.wav", 2: "Iot2WifiDirect.wav",
    3: "WifiDirect2Iot.wav", 4: "connect_wifi.wav", 5: "success.wav", 6: "fail.wav",
    7: "charging.wav", 8: "leave_home.wav", 9: "battery_low.wav",
    10: "got_new_instrction.wav", 11: "obstacle_detected.wav",
    12: "rest_to_factory_condition.wav", 13: "alert2.wav",
    14: "(broken path - concatenated string literal, plays nothing)",
}


if __name__ == "__main__":
    print("night mode (isNight, ir_brightness):", night_get())
    print("recording now:", record_status())
    for t, label in ((RECORD_TYPE_SNAPSHOT, "snapshots"), (RECORD_TYPE_RECORD, "videos")):
        print(f"{label} on robot:", record_file_num(t))
    for f in record_files(RECORD_TYPE_RECORD, 0, 3):
        print(f"  {f['path']}  {f['dur']}s  {f['size']}B")
