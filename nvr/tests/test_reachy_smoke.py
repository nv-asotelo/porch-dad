"""Tests for scripts/reachy_smoke.py. Stdlib unittest, no robot, no network beyond loopback.

    python3 -m unittest discover -s nvr/tests -p 'test_reachy_smoke.py' -v

The parsers are tested on bytes built here from the specs, so a pass means the parser matches the
format rather than one captured sample. The checks are then run against small fake servers on
loopback - and the inference check against the REAL Live Vision server (nvr/ui/scripts/serve_ui.py)
in front of a fake shim, because the request shape it must satisfy is serve_ui's validate_request
and origin rules, and a copy of those rules in a test would drift.

Two tests need the Orin's /home/orin/reachy_env (PyAV, and the bridge's own imports) and skip
anywhere else: they push audio through the bridge's real AudioHub and check that what it emits is
what the smoke check accepts.
"""
from __future__ import annotations

import importlib.util
import json
import math
import random
import socket
import sys
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module          # dataclasses look their module up while it executes
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


smoke = _load("reachy_smoke", ROOT / "scripts" / "reachy_smoke.py")
serve_ui = _load("serve_ui", ROOT / "nvr" / "ui" / "scripts" / "serve_ui.py")


# ----------------------------------------------------------------------------------- builders
def segment(marker: int, payload: bytes) -> bytes:
    return bytes([0xFF, marker]) + (len(payload) + 2).to_bytes(2, "big") + payload


def jpeg(width: int = 1280, height: int = 720, sof: int = 0xC0, before_sof: bytes = b"") -> bytes:
    """Header-accurate JPEG: SOI, APP0, DQT, DHT, SOFn, SOS, a little scan data, EOI."""
    app0 = segment(0xE0, b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00")
    dqt = segment(0xDB, b"\x00" + bytes(64))
    dht = segment(0xC4, b"\x00" + bytes(16))          # C4 sits in the SOF range and is not one
    sof_seg = segment(sof, b"\x08" + height.to_bytes(2, "big") + width.to_bytes(2, "big")
                      + b"\x03\x01\x22\x00\x02\x11\x01\x03\x11\x01")
    sos = segment(0xDA, b"\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00")
    return b"\xff\xd8" + app0 + before_sof + dqt + dht + sof_seg + sos + b"\x12\x34" + b"\xff\xd9"


def mp3_frame(bitrate_index: int = 7, rate_index: int = 1, padding: int = 0, mode: int = 1,
              version_bits: int = 3, fill: int = 0) -> bytes:
    """One MPEG audio layer III frame with a zeroed body; defaults are 96 kb/s 48 kHz joint stereo,
    which is what the bridge's libmp3lame settings produce."""
    b1 = 0xE0 | (version_bits << 3) | (1 << 1) | 1               # layer III, no CRC
    b2 = (bitrate_index << 4) | (rate_index << 2) | (padding << 1)
    b3 = mode << 6
    header = bytes([0xFF, b1, b2, b3])
    length = smoke.mp3_header(header)["length"]
    return header + bytes([fill]) * (length - 4)


def mjpeg_part(body: bytes, length: bool = True, boundary: bytes = b"frame") -> bytes:
    head = b"--" + boundary + b"\r\nContent-Type: image/jpeg\r\n"
    if length:
        head += f"Content-Length: {len(body)}\r\n".encode()
    return head + b"\r\n" + body + b"\r\n"


SHIM_EVENTS = [
    {"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
    {"choices": [{"index": 0, "delta": {"content": "A desk "}, "finish_reason": None}]},
    {"choices": [{"index": 0, "delta": {"content": "with a café "}, "finish_reason": None}]},
    {"choices": [{"index": 0, "delta": {"content": "mug."}, "finish_reason": None}]},
    {"cosmos_metrics": {"timing_boundary": "native_inference", "native_inference_ms": 412.5},
     "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
]


def shim_sse(events=SHIM_EVENTS) -> bytes:
    """Byte-for-byte the shape nvr/shim/cosmos3_shim_v1.py streams."""
    out = "".join("data: " + json.dumps(e, separators=(",", ":")) + "\n\n" for e in events)
    return (out + "data: [DONE]\n\n").encode()


# ----------------------------------------------------------------------------------- parsers
class JpegDimensions(unittest.TestCase):
    def test_baseline(self):
        self.assertEqual(smoke.jpeg_dimensions(jpeg(1280, 720)), (1280, 720))

    def test_progressive(self):
        self.assertEqual(smoke.jpeg_dimensions(jpeg(640, 360, sof=0xC2)), (640, 360))

    def test_sof_bytes_inside_an_earlier_segment_are_not_the_sof(self):
        # An EXIF thumbnail carries its own SOF; a byte search would report the thumbnail's size.
        exif = segment(0xE1, b"Exif\x00\x00" + b"\xff\xc0\x00\x11\x08\x00\x10\x00\x10")
        self.assertEqual(smoke.jpeg_dimensions(jpeg(1280, 720, before_sof=exif)), (1280, 720))

    def test_fill_bytes_before_a_marker(self):
        data = jpeg(320, 240)
        padded = data[:2] + b"\xff\xff\xff" + data[2:]
        self.assertEqual(smoke.jpeg_dimensions(padded), (320, 240))

    def test_not_a_jpeg(self):
        self.assertIsNone(smoke.jpeg_dimensions(b"\x89PNG\r\n\x1a\n" + bytes(64)))
        self.assertIsNone(smoke.jpeg_dimensions(b""))
        self.assertIsNone(smoke.jpeg_dimensions(b"no fresh frame (dormant: robot off)"))

    def test_truncated_or_malformed(self):
        data = jpeg()
        sof_at = data.index(b"\xff\xc0")
        self.assertIsNone(smoke.jpeg_dimensions(data[:sof_at + 6]))
        self.assertIsNone(smoke.jpeg_dimensions(b"\xff\xd8" + b"\x00" * 8))
        # Scan data before any SOF: there is no frame header to read.
        self.assertIsNone(smoke.jpeg_dimensions(b"\xff\xd8" + segment(0xDA, b"\x00" * 4)))
        self.assertIsNone(smoke.jpeg_dimensions(jpeg(0, 720)))


class SplitMjpeg(unittest.TestCase):
    def test_parts_by_content_length(self):
        frames = [jpeg(1280, 720), jpeg(640, 360), jpeg(320, 240)]
        parts, rest = smoke.split_mjpeg(b"".join(mjpeg_part(f) for f in frames))
        self.assertEqual(parts, frames)
        self.assertEqual(rest, b"\r\n")

    def test_a_split_part_waits_for_the_rest(self):
        frames = [jpeg(), jpeg(), jpeg()]
        stream = b"".join(mjpeg_part(f) for f in frames)
        cut = len(stream) - 40
        parts, rest = smoke.split_mjpeg(stream[:cut])
        self.assertEqual(len(parts), 2)
        more, _ = smoke.split_mjpeg(rest + stream[cut:])
        self.assertEqual(more, frames[2:])

    def test_without_content_length_ends_at_the_next_delimiter(self):
        frames = [jpeg(), jpeg(640, 360)]
        stream = b"".join(mjpeg_part(f, length=False) for f in frames) + b"--frame\r\n"
        parts, _ = smoke.split_mjpeg(stream)
        self.assertEqual(parts, frames)

    def test_other_boundary(self):
        parts, _ = smoke.split_mjpeg(mjpeg_part(jpeg(), boundary=b"xyz"), b"xyz")
        self.assertEqual(len(parts), 1)
        self.assertEqual(smoke.split_mjpeg(mjpeg_part(jpeg(), boundary=b"xyz"))[0], [])


class Mp3Scan(unittest.TestCase):
    def test_header_of_the_bridge_format(self):
        h = smoke.mp3_header(mp3_frame())
        self.assertEqual((h["version"], h["layer"], h["kbps"], h["sample_rate"], h["channels"]),
                         ("1", 3, 96, 48000, 2))
        self.assertEqual(h["length"], 288)          # 144 * 96000 / 48000

    def test_frame_lengths(self):
        # 128 kb/s at 44.1 kHz with the padding bit: 144 * 128000 / 44100 = 417, + 1.
        self.assertEqual(smoke.mp3_header(mp3_frame(9, 0, padding=1))["length"], 418)
        # MPEG-2 layer III halves the coefficient: 72 * 64000 / 24000 = 192.
        h = smoke.mp3_header(mp3_frame(8, 1, version_bits=2))
        self.assertEqual((h["version"], h["sample_rate"], h["kbps"], h["length"], h["samples"]),
                         ("2", 24000, 64, 192, 576))

    def test_invalid_headers(self):
        good = bytearray(mp3_frame()[:4])
        for byte, value in ((2, 0xF4), (2, 0x0C | 0x70), (1, 0xE0 | (1 << 3) | 3), (1, 0xF9)):
            bad = bytearray(good)
            bad[byte] = value                   # bitrate 15, rate 3, reserved version, layer 00
            self.assertIsNone(smoke.mp3_header(bytes(bad)), bytes(bad).hex())
        self.assertIsNone(smoke.mp3_header(b"\xff\xfb\x74"))            # short
        self.assertIsNone(smoke.mp3_header(b"ID3\x04"))

    def test_counts_a_chain(self):
        scan = smoke.scan_mp3(mp3_frame() * 10)
        self.assertEqual(scan["frames"], 10)
        self.assertEqual(scan["bytes"], 2880)
        self.assertEqual((scan["sample_rate"], scan["channels"], scan["kbps"]), (48000, 2, 96))
        self.assertAlmostEqual(scan["seconds"], 10 * 1152 / 48000)

    def test_resyncs_past_garbage_and_false_syncs(self):
        # A valid-looking header whose "next frame" is not there must not count.
        garbage = b"\x00\x11" + b"\xff\xfb\x74\x40" + bytes(50) + b"\xff\xf3"
        scan = smoke.scan_mp3(garbage + mp3_frame() * 5)
        self.assertEqual((scan["frames"], scan["first_offset"]), (5, len(garbage)))

    def test_partial_last_frame_is_not_counted(self):
        data = mp3_frame() * 4
        self.assertEqual(smoke.scan_mp3(data + mp3_frame()[:100])["frames"], 4)

    def test_mono(self):
        self.assertEqual(smoke.scan_mp3(mp3_frame(mode=3) * 3)["channels"], 1)

    def test_random_bytes_are_not_audio(self):
        rng = random.Random(20260924)
        noise = bytes(rng.getrandbits(8) for _ in range(64 * 1024))
        self.assertEqual(smoke.scan_mp3(noise)["frames"], 0)
        self.assertEqual(smoke.scan_mp3(b"")["frames"], 0)


class Sse(unittest.TestCase):
    def check(self, sse):
        self.assertEqual(sse.text, "A desk with a café mug.")
        self.assertTrue(sse.done)
        self.assertIsNone(sse.error)
        self.assertEqual(sse.finish_reason, "stop")
        self.assertEqual(sse.metrics["native_inference_ms"], 412.5)

    def test_whole(self):
        sse = smoke.SseStream()
        sse.feed(shim_sse())
        self.check(sse)
        self.assertEqual(sse.events, 6)

    def test_byte_by_byte_including_a_split_character(self):
        sse = smoke.SseStream()
        for b in shim_sse():
            sse.feed(bytes([b]))
        self.check(sse)

    def test_crlf(self):
        sse = smoke.SseStream()
        data = shim_sse().replace(b"\n", b"\r\n")
        for i in range(0, len(data), 7):
            sse.feed(data[i:i + 7])
        self.check(sse)

    def test_live_vision_error_event_after_the_200(self):
        # serve_ui writes this when the backend fails mid-stream; note the leading newline.
        sse = smoke.SseStream()
        sse.feed(shim_sse(SHIM_EVENTS[:2]).replace(b"data: [DONE]\n\n", b""))
        sse.feed(b'\nevent: error\n'
                 b'data: {"error": {"message": "Backend stream exceeded 1 MiB."}}\n\n')
        self.assertEqual(sse.text, "A desk ")
        self.assertEqual(sse.error, "Backend stream exceeded 1 MiB.")

    def test_shim_inference_error_chunk(self):
        sse = smoke.SseStream()
        sse.feed(shim_sse([{"choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
                            "error": {"message": "inference failed: boom"}}]))
        self.assertEqual(sse.error, "inference failed: boom")
        self.assertEqual(sse.finish_reason, "error")
        self.assertTrue(sse.done)

    def test_comments_other_fields_and_an_unterminated_last_event(self):
        sse = smoke.SseStream()
        sse.feed(b": keep-alive\n\nid: 7\nretry: 100\n\n")
        sse.feed(b'data: {"choices":[{"delta":{"content":"hi"}}]}')
        self.assertEqual(sse.text, "")
        sse.close()
        self.assertEqual((sse.text, sse.events, sse.done), ("hi", 1, False))

    def test_multi_line_data(self):
        sse = smoke.SseStream()
        sse.feed(b'data: {"choices":\ndata: [{"delta":{"content":"x"}}]}\n\n')
        self.assertEqual(sse.text, "x")


class FeedConfig(unittest.TestCase):
    def test_repo_config(self):
        text = (ROOT / "nvr" / "feed" / "config.yaml").read_text()
        engines = smoke.config_engines(text)
        self.assertEqual(list(engines), ["v2", "v1", "v3", "mlp"])
        self.assertEqual(engines["v2"]["path"], "/home/orin/tensorrt-edgellm-workspace/"
                                                "Cosmos3-Edge-INT4-v2/engines/reasoning")
        self.assertEqual(engines["v2"]["name"], "Fast")
        eid, how = smoke.default_engine(engines)
        self.assertEqual(eid, "v2")
        self.assertIn("default choice", how)
        self.assertEqual(smoke.config_scalar(text, "db_path"), "/home/orin/nvr/feed/feed.db")
        self.assertIsNone(smoke.config_scalar(text, "service_intent_path"))

    def test_folded_notes_and_no_marked_default(self):
        text = ("engines:\n"
                "  # comment\n"
                "  b:\n"
                "    name: \"B\"\n"
                "    path: \"/e/b\"\n"
                "    notes: >-\n"
                "      first line,\n"
                "      second line.\n"
                "  a:\n"
                "    name: A\n"
                "    path: /e/a\n"
                "scouts:\n"
                "  - id: x\n")
        engines = smoke.config_engines(text)
        self.assertEqual(engines, {"b": {"name": "B", "path": "/e/b",
                                         "notes": "first line, second line."},
                                   "a": {"name": "A", "path": "/e/a"}})
        self.assertEqual(smoke.default_engine(engines)[0], "b")
        self.assertEqual(smoke.default_engine({})[0], None)

    def test_never_reads_outside_the_engines_block(self):
        text = 'control_token: "s3cret"\nengines:\n  v2:\n    path: /p\nother: 1\n'
        self.assertNotIn("s3cret", json.dumps(smoke.config_engines(text)))


class Verdicts(unittest.TestCase):
    LIVE = {"state": "live", "reason": None, "live": True, "has_frame": True, "frames": 900,
            "last_frame_age_s": 0.1, "sessions": 1, "failed_streak": 0,
            "audio": {"live": True, "last_frame_age_s": 0.0},
            "push": {"enabled": True, "state": "pushing", "detail": None}}

    def test_bridge(self):
        self.assertEqual(smoke.bridge_verdict(self.LIVE)[0], smoke.PASS)
        no_audio = {**self.LIVE, "audio": {"live": False, "last_frame_age_s": 30.0}}
        self.assertEqual(smoke.bridge_verdict(no_audio)[0], smoke.WARN)
        dormant = {"state": "dormant", "reason": "camera held by the robot app 'hand_tracker'",
                   "live": False, "blocked_by": "hand_tracker"}
        status, detail = smoke.bridge_verdict(dormant)
        self.assertEqual(status, smoke.WARN)
        self.assertIn("hand_tracker", detail)
        failing = {"state": "reconnecting", "reason": "no video within 20s of connecting",
                   "live": False, "sessions": 5, "failed_streak": 5}
        status, detail = smoke.bridge_verdict(failing)
        self.assertEqual(status, smoke.FAIL)
        self.assertIn("failed_streak 5", detail)
        self.assertEqual(smoke.bridge_verdict({"state": "connecting", "live": False, "sessions": 1,
                                               "failed_streak": 0})[0], smoke.WARN)
        self.assertEqual(smoke.bridge_verdict({"state": "connecting", "live": False, "sessions": 6,
                                               "failed_streak": 5})[0], smoke.FAIL)
        stale = {**self.LIVE, "live": False, "last_frame_age_s": 7.5}
        self.assertEqual(smoke.bridge_verdict(stale)[0], smoke.FAIL)

    def test_boot(self):
        self.assertEqual(smoke.boot_verdict("enabled")[0], smoke.PASS)
        for state in ("disabled", "masked", "not-found", "enabled-runtime"):
            self.assertEqual(smoke.boot_verdict(state)[0], smoke.FAIL, state)
        self.assertEqual(smoke.boot_verdict("static")[0], smoke.WARN)

    def test_intent(self):
        v = smoke.intent_verdict
        self.assertEqual(v("enabled", "running")[0], smoke.PASS)
        self.assertEqual(v("disabled", "stopped")[0], smoke.PASS)
        self.assertEqual(v("disabled", "running")[0], smoke.WARN)
        self.assertEqual(v("enabled", "stopped")[0], smoke.WARN)
        self.assertEqual(v("disabled", None)[0], smoke.PASS)
        self.assertEqual(v("not-found", "running")[0], smoke.FAIL)

    def test_restart(self):
        self.assertEqual(smoke.restart_verdict("unless-stopped")[0], smoke.PASS)
        self.assertEqual(smoke.restart_verdict("always")[0], smoke.WARN)
        self.assertEqual(smoke.restart_verdict("no")[0], smoke.FAIL)
        self.assertEqual(smoke.restart_verdict("")[0], smoke.FAIL)

    def test_ui_down(self):
        v = smoke.ui_down_verdict
        # Turned off from the command centre, or no record and no crash: off on purpose.
        self.assertEqual(v("stopped", "inactive")[0], smoke.SKIP)
        self.assertEqual(v("stopped", None)[0], smoke.SKIP)
        self.assertEqual(v(None, "inactive")[0], smoke.SKIP)
        self.assertEqual(v(None, None)[0], smoke.SKIP)
        # Last started there and now not running: what a clone boot that lost the unit looks like.
        status, detail = v("running", "inactive")
        self.assertEqual(status, smoke.FAIL)
        self.assertIn("did it come back after the boot?", detail)
        self.assertEqual(v("running", None)[0], smoke.FAIL)
        # A crash is not a decision, recorded intent or not.
        self.assertEqual(v(None, "failed")[0], smoke.FAIL)
        self.assertEqual(v("running", "failed")[0], smoke.FAIL)
        # Up but not listening yet.
        self.assertEqual(v("running", "activating")[0], smoke.WARN)
        self.assertEqual(v(None, "active")[0], smoke.WARN)

    def test_shim_down(self):
        v, now = smoke.shim_down_verdict, 20000.0

        def unit(active="active", sub="running", age=None, restarts="0", load="loaded"):
            u = {"LoadState": load, "ActiveState": active, "SubState": sub, "NRestarts": restarts,
                 "ActiveEnterTimestampMonotonic": "0"}
            if age is not None:
                u["ActiveEnterTimestampMonotonic"] = str(int((now - age) * 1e6))
            return u

        # Right after a start the port refuses while the engine loads: a WARN, not a FAIL.
        status, detail = v(unit(age=35), now)
        self.assertEqual(status, smoke.WARN)
        self.assertIn("running for 35 s but not listening yet", detail)
        self.assertEqual(v(unit(), now)[0], smoke.WARN)               # no timestamp: benefit of doubt
        # Far longer than any load: stuck.
        status, detail = v(unit(age=smoke.SHIM_LOAD_GRACE_S + 600), now)
        self.assertEqual(status, smoke.FAIL)
        self.assertIn("15 min", detail)
        # Loading again after systemd restarted it says so.
        self.assertIn("restarted it 2 times", v(unit(age=10, restarts="2"), now)[1])
        self.assertEqual(v(unit("activating", "auto-restart", restarts="4"), now)[0], smoke.FAIL)
        self.assertEqual(v(unit("activating", "start"), now)[0], smoke.WARN)
        self.assertEqual(v(unit("failed", "failed"), now)[0], smoke.FAIL)
        self.assertEqual(v(unit("inactive", "dead"), now)[0], smoke.FAIL)
        self.assertIn("not installed", v(unit("inactive", "dead", load="not-found"), now)[1])
        # systemd not asked (not the Orin, or a test): the old answer.
        status, detail = v({}, now)
        self.assertEqual(status, smoke.FAIL)
        self.assertIn("cosmos3-edge-shim.service running?", detail)


# ----------------------------------------------------------------------------------- fakes
@contextmanager
def serve(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    server.block_on_close = False
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", server
    finally:
        server.shutdown()
        server.server_close()


def closed_port() -> int:
    """A loopback port nothing listens on, so a connection is refused at once."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Quiet(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send_json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class FakeBridge(Quiet):
    """The bridge's HTTP surface as reachy_mjpeg_bridge.py serves it: chunked endless streams."""
    protocol_version = "HTTP/1.1"
    health = Verdicts.LIVE

    def chunk(self, data: bytes) -> None:
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def endless(self, content_type: str, piece, interval: float) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        deadline = time.monotonic() + 5
        try:
            while time.monotonic() < deadline:
                self.chunk(piece())
                time.sleep(interval)
        except (BrokenPipeError, ConnectionResetError):
            pass
        self.close_connection = True

    def do_GET(self):
        if self.path == "/healthz":
            self.send_json(self.health)
        elif self.path == "/still.jpg":
            if not self.health.get("has_frame"):
                self.send_json({"error": "no fresh frame"}, 503)
                return
            body = jpeg()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/mjpeg":
            self.endless("multipart/x-mixed-replace; boundary=frame",
                         lambda: mjpeg_part(jpeg()), 0.1)
        elif self.path == "/audio.mp3":
            # ~14 KB/s of frames, a little over the real 96 kb/s, in packet-sized writes.
            self.endless("audio/mpeg", lambda: mp3_frame() * 2, 0.04)
        else:
            self.send_json({"error": "not found"}, 404)


def dead(port=None):
    return f"http://127.0.0.1:{port or closed_port()}"


def only(smoke_obj, *names):
    return {r.check: r for r in smoke_obj.results if r.check in names or not names}


def feed_config(directory: Path, intent: dict | None = None) -> Path:
    """A porch-feed config whose service_intent.json is in `directory`.

    The repo's config points at /home/orin/nvr/feed/, and on the Orin the file there records what
    a human last did to each service: a test that read it would pass or fail with the house.
    """
    if intent is not None:
        (directory / "service_intent.json").write_text(json.dumps(intent))
    path = directory / "config.yaml"
    path.write_text(f'db_path: "{directory / "feed.db"}"\n')
    return path


class BridgeChecks(unittest.TestCase):
    def setUp(self):
        self.saved_av, smoke.av = smoke.av, None      # level measurement has its own test
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        smoke.av = self.saved_av
        self.tmp.cleanup()

    def run_smoke(self, base, intent=None):
        port = int(base.rsplit(":", 1)[1])
        s = smoke.Smoke(bridge_hosts=("127.0.0.1",), bridge_port=port, webui_url=dead(),
                        live_vision_url=dead(), shim_url=dead(), frigate_url=dead(),
                        engine_link="/nonexistent/default", sample_s=1.0, inference=False,
                        robot=False, system=False,
                        feed_config=feed_config(Path(self.tmp.name), intent))
        s.run()
        return s

    def test_live_bridge(self):
        with serve(FakeBridge) as (base, _):
            s = self.run_smoke(base)
        r = only(s)
        self.assertEqual(r["bridge 127.0.0.1"].status, smoke.PASS, r["bridge 127.0.0.1"].detail)
        self.assertEqual(r["bridge /still.jpg"].status, smoke.PASS)
        self.assertIn("1280x720", r["bridge /still.jpg"].detail)
        self.assertEqual(r["bridge /mjpeg"].status, smoke.PASS, r["bridge /mjpeg"].detail)
        self.assertEqual(r["bridge /audio.mp3"].status, smoke.PASS, r["bridge /audio.mp3"].detail)
        self.assertIn("48000 Hz stereo 96 kb/s", r["bridge /audio.mp3"].detail)
        # Deliberately-off services SKIP; the shim and Frigate are expected on, so they FAIL.
        self.assertEqual(r["webui push (reachy)"].status, smoke.SKIP)
        self.assertEqual(r["live vision /reachy"].status, smoke.SKIP)
        self.assertEqual(r["live vision inference"].status, smoke.SKIP)
        self.assertEqual(r["shim"].status, smoke.FAIL)
        self.assertEqual(r["shim engine"].status, smoke.SKIP)
        self.assertEqual(r["frigate reachy_mini"].status, smoke.FAIL)
        self.assertEqual(s.still, jpeg())
        # The shim is judged before the inference that depends on it.
        names = [x.check for x in s.results]
        self.assertLess(names.index("shim"), names.index("live vision inference"))

    def test_live_ui_down_against_the_recorded_intent(self):
        # Both last started from the command centre and neither answering: that is not "off on
        # purpose", and after a clone boot it is exactly what the check is for.
        started = {"desired": "running", "action": "start", "by": "user", "at": 1789662049.0}
        with serve(FakeBridge) as (base, _):
            s = self.run_smoke(base, intent={"vlm": started, "livevision": started})
        r = only(s)
        for name in ("webui push (reachy)", "live vision /reachy"):
            self.assertEqual(r[name].status, smoke.FAIL, r[name].detail)
            self.assertIn("last started from the command centre", r[name].detail)
            self.assertIn("intent: running", r[name].detail)

    def test_bridge_gone_between_healthz_and_the_streams(self):
        # Restart=always, a MemoryMax kill: /healthz answered, the next request is refused. Each
        # check must say so under its own name, not as a crash of the script.
        s = smoke.Smoke(bridge_hosts=(), robot=False, system=False, sample_s=0.5)
        s.health, s.bridge_url = Verdicts.LIVE, dead()
        for check in (s.check_still, s.check_mjpeg, s.check_audio):
            check()
        self.assertEqual([r.check for r in s.results],
                         ["bridge /still.jpg", "bridge /mjpeg", "bridge /audio.mp3"])
        for r in s.results:
            self.assertEqual(r.status, smoke.FAIL)
            self.assertIn("stopped answering mid-check (connection refused)", r.detail)

    def test_dormant_bridge_is_one_warning_not_a_cascade(self):
        class Dormant(FakeBridge):
            health = {"state": "dormant", "reason": "daemon has released the camera and microphone",
                      "live": False, "has_frame": False, "blocked_by": None,
                      "audio": {"live": False}, "push": {"enabled": True}}
        with serve(Dormant) as (base, _):
            s = self.run_smoke(base)
        r = only(s)
        self.assertEqual(r["bridge 127.0.0.1"].status, smoke.WARN)
        for name in ("bridge /still.jpg", "bridge /mjpeg", "bridge /audio.mp3"):
            self.assertEqual(r[name].status, smoke.SKIP, name)
            self.assertIn("released", r[name].detail)

    def test_bridge_down(self):
        s = self.run_smoke(dead())
        r = only(s)
        self.assertEqual(r["bridge 127.0.0.1"].status, smoke.FAIL)
        self.assertIn("connection refused", r["bridge 127.0.0.1"].detail)
        self.assertIn("reachy-mjpeg-bridge.service", r["bridge 127.0.0.1"].detail)
        self.assertEqual(r["bridge /mjpeg"].status, smoke.SKIP)

    def test_gateway_only_down_points_at_docker(self):
        with serve(FakeBridge) as (base, _):
            port = int(base.rsplit(":", 1)[1])
            # 127.0.0.2 is loopback too, and nothing listens there: it stands in for the gateway.
            s = smoke.Smoke(bridge_hosts=("127.0.0.1", "127.0.0.2"), bridge_port=port,
                            robot=False, system=False)
            s.check_bridge_health()
        self.assertEqual([r.status for r in s.results], [smoke.PASS, smoke.FAIL])
        self.assertIn("docker not up?", s.results[1].detail)


class ConsumerChecks(unittest.TestCase):
    def make(self, **kw):
        kw.setdefault("system", False)
        s = smoke.Smoke(bridge_hosts=(), robot=False, sample_s=0.5,
                        feed_config=ROOT / "nvr" / "feed" / "config.yaml", **kw)
        return s

    def test_webui_push(self):
        received = {"n": 100}

        class WebUI(Quiet):
            session = "reachy"

            def do_GET(self):
                received["n"] += 3
                self.send_json({"streams": [{"session_id": self.session, "connected": True,
                                             "frames_received": received["n"]}]})
        with serve(WebUI) as (base, _):
            s = self.make(webui_url=base)
            s.health = Verdicts.LIVE
            s.check_webui_push()
            self.assertEqual(s.results[-1].status, smoke.PASS, s.results[-1].detail)

            # The session exists under another name only: the bridge's reason decides the verdict.
            WebUI.session = "other"
            stopped = "stopped in the WebUI; press Start there to resume"
            s.health = {**Verdicts.LIVE,
                        "push": {"enabled": True, "detail": "HTTP 409", "state": stopped}}
            s.check_webui_push()
            self.assertEqual(s.results[-1].status, smoke.WARN)
            s.health = {**Verdicts.LIVE,
                        "push": {"enabled": True, "state": "WebUI answered HTTP 500"}}
            s.check_webui_push()
            self.assertEqual(s.results[-1].status, smoke.FAIL)
            self.assertIn("HTTP 500", s.results[-1].detail)

    def test_webui_down_is_a_skip_with_the_recorded_intent(self):
        s = self.make(webui_url=dead())
        s.intent = {"vlm": {"desired": "stopped", "by": "user", "at": 1789662049.0}}
        s.check_webui_push()
        self.assertEqual(s.results[-1].status, smoke.SKIP)
        self.assertIn("intent: stopped", s.results[-1].detail)

    def test_ui_down_asks_systemd_which_unit(self):
        asked = []

        def activity(unit):
            asked.append(unit)
            return {"LoadState": "loaded", "ActiveState": "failed", "SubState": "failed"}
        with mock.patch.object(smoke, "unit_activity", activity):
            s = self.make(webui_url=dead(), live_vision_url=dead(), system=True)
            s.check_webui_push()
            s.check_live_vision_proxy()
        self.assertEqual(asked, ["live-vlm-webui-fork.service", "cosmos-edge-ui.service"])
        self.assertEqual([r.status for r in s.results], [smoke.FAIL, smoke.FAIL])
        self.assertIn("crashed", s.results[0].detail)

    def test_live_vision_proxy(self):
        class LiveVision(Quiet):
            has_proxy = True

            def do_GET(self):
                if self.path == "/reachy/healthz" and self.has_proxy:
                    self.send_json(Verdicts.LIVE)
                else:
                    self.send_json({"error": {"message": "Not found."}}, 404)
        with serve(LiveVision) as (base, _):
            s = self.make(live_vision_url=base)
            s.check_live_vision_proxy()
            self.assertEqual(s.results[-1].status, smoke.PASS)
            LiveVision.has_proxy = False
            s.check_live_vision_proxy()
            self.assertEqual(s.results[-1].status, smoke.FAIL)
            self.assertIn("no /reachy/ proxy", s.results[-1].detail)

    def test_live_vision_proxy_carries_the_relay_token(self):
        # Live Vision answers /reachy/ only with the token its own page reads from /api/access.
        class LiveVision(Quiet):
            def do_GET(self):
                if self.path == "/api/access":
                    self.send_json({"https_port": 8443, "reachy_token": "tok-en_1"})
                elif self.path == "/reachy/healthz?token=tok-en_1":
                    self.send_json(Verdicts.LIVE)
                else:
                    self.send_json({"error": {"message": "Missing or out-of-date relay token."}}, 401)
        with serve(LiveVision) as (base, _):
            s = self.make(live_vision_url=base)
            s.check_live_vision_proxy()
            self.assertEqual(s.results[-1].status, smoke.PASS, s.results[-1].detail)

    def test_shim_loading_is_one_warning_and_no_inference(self):
        # cosmos3_shim_v1 loads its engine before uvicorn opens :8000, so for ~40-70 s after a
        # start the port refuses. With the unit active that is loading, not down; and the
        # inference is not sent, because Live Vision would only relay the shim's absence as a 503.
        calls = []

        class LiveVision(Quiet):
            def do_GET(self):
                calls.append(self.path)
                self.send_json({"error": {"message": "backend is not ready."}}, 503)

        loading = {"LoadState": "loaded", "ActiveState": "active", "SubState": "running",
                   "NRestarts": "0",
                   "ActiveEnterTimestampMonotonic": str(int((time.monotonic() - 30) * 1e6))}
        with serve(LiveVision) as (base, _), \
                mock.patch.object(smoke, "unit_activity", lambda unit: loading):
            s = self.make(shim_url=dead(), live_vision_url=base, image=jpeg(), system=True)
            s.check_shim()
            s.check_inference()
        shim, inference = s.results
        self.assertEqual(shim.status, smoke.WARN, shim.detail)
        self.assertIn("loading the engine", shim.detail)
        self.assertEqual(inference.status, smoke.SKIP, inference.detail)
        self.assertIn("not sent: the shim is not listening", inference.detail)
        self.assertEqual(calls, [])

    def test_frigate(self):
        class Frigate(Quiet):
            def do_GET(self):
                if self.path == "/api/go2rtc/streams":
                    self.send_json({"reachy_mini": {"producers": [
                        {"url": "ffmpeg:http://host.docker.internal:8099/mjpeg#video=h264"},
                        {"url": "ffmpeg:http://host.docker.internal:8099/audio.mp3#audio=opus"}]},
                        "front_driveway": {"producers": []}})
                else:
                    self.send_json({"cameras": {"reachy_mini": {"enabled": False,
                                                                "enabled_in_config": True}}})
        with serve(Frigate) as (base, _):
            s = self.make(frigate_url=base)
            s.check_frigate()
        r = s.results[-1]
        self.assertEqual(r.status, smoke.PASS)
        self.assertIn("/audio.mp3, /mjpeg", r.detail)
        self.assertIn("camera disabled at runtime (config file says enabled) "
                      "(reported, not changed)", r.detail)

    def test_frigate_mic_as_its_own_stream(self):
        # As deployed: reachy_mini video only, the microphone in reachy_mini_mic (deploy/07 §6).
        class Frigate(Quiet):
            def do_GET(self):
                if self.path == "/api/go2rtc/streams":
                    self.send_json({
                        "reachy_mini": {"producers": [
                            {"url": "ffmpeg:http://host.docker.internal:8099/mjpeg#video=h264"}]},
                        "reachy_mini_mic": {"producers": [
                            {"url": "rtsp://127.0.0.1:8554/reachy_mini"},
                            {"url": "ffmpeg:http://host.docker.internal:8099/audio.mp3#audio=opus"}]}})
                else:
                    self.send_json({"cameras": {"reachy_mini": {"enabled": False}}})
        with serve(Frigate) as (base, _):
            s = self.make(frigate_url=base)
            s.check_frigate()
        r = s.results[-1]
        self.assertEqual(r.status, smoke.PASS)
        self.assertIn("bridge sources: /mjpeg; mic as reachy_mini_mic (/audio.mp3)", r.detail)
        self.assertNotIn("no audio source", r.detail)


class HomeAssistant(unittest.TestCase):
    """HA's registries as HA 2024.12 writes them (.storage/core.config_entries, entity_registry)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.config = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def registries(self, *, entry_disabled=None, camera_disabled=None):
        storage = self.config / ".storage"
        storage.mkdir(exist_ok=True)
        entries = [{"entry_id": "R1", "domain": "reachy_mini", "disabled_by": entry_disabled,
                    "title": "Reachy Mini (12d81245)", "data": {"host": "192.168.6.162"}},
                   {"entry_id": "Y1", "domain": "yolocal", "disabled_by": None,
                    "data": {"secret": "s3cret"}}]
        entities = [{"entity_id": "camera.reachy_mini_12d8_camera", "platform": "reachy_mini",
                     "config_entry_id": "R1", "disabled_by": camera_disabled},
                    {"entity_id": "select.reachy_mini_12d8_motor_mode", "platform": "reachy_mini",
                     "config_entry_id": "R1", "disabled_by": None},
                    {"entity_id": "camera.front_driveway", "platform": "mqtt",
                     "config_entry_id": "M1", "disabled_by": None}]
        (storage / "core.config_entries").write_text(json.dumps(
            {"version": 1, "minor_version": 4, "key": "core.config_entries",
             "data": {"entries": entries}}))
        (storage / "core.entity_registry").write_text(json.dumps(
            {"version": 1, "minor_version": 15, "key": "core.entity_registry",
             "data": {"entities": entities, "deleted_entities": []}}))

    def test_cameras(self):
        self.assertIsNone(smoke.ha_reachy_cameras(self.config))
        self.registries()
        self.assertEqual(smoke.ha_reachy_cameras(self.config), ["camera.reachy_mini_12d8_camera"])
        self.registries(camera_disabled="user")
        self.assertEqual(smoke.ha_reachy_cameras(self.config), [])
        # A disabled integration loads none of its entities.
        self.registries(entry_disabled="user")
        self.assertEqual(smoke.ha_reachy_cameras(self.config), [])

    def run_check(self, container):
        s = smoke.Smoke(bridge_hosts=(), robot=False, ha_config=self.config)
        with mock.patch.object(smoke.shutil, "which", lambda name: "/usr/bin/" + name), \
                mock.patch.object(smoke, "container_state", lambda name: container):
            s.check_home_assistant()
        return only(s)

    def test_never_a_failure(self):
        self.registries()
        r = self.run_check(("no", "exited"))
        # Off with restart=no may be deliberate; a camera that cannot open sessions now is a SKIP.
        self.assertEqual(r["restart homeassistant"].status, smoke.WARN)
        self.assertIn("restart=no", r["restart homeassistant"].detail)
        self.assertIn("off on purpose", r["restart homeassistant"].detail)
        self.assertEqual(r["ha reachy camera"].status, smoke.SKIP)
        self.assertIn("before starting HA", r["ha reachy camera"].detail)

        r = self.run_check(("unless-stopped", "running"))
        self.assertEqual(r["restart homeassistant"].status, smoke.PASS)
        self.assertEqual(r["ha reachy camera"].status, smoke.WARN)
        self.assertIn("camera.reachy_mini_12d8_camera enabled", r["ha reachy camera"].detail)
        self.assertNotIn("s3cret", json.dumps([vars(x) for x in r.values()]))

        self.registries(camera_disabled="user")
        self.assertEqual(self.run_check(("unless-stopped", "running"))["ha reachy camera"].status,
                         smoke.PASS)
        r = self.run_check((None, "Error: No such object: homeassistant"))
        self.assertEqual(r["restart homeassistant"].status, smoke.SKIP)


class LiveVisionInference(unittest.TestCase):
    """The real serve_ui.py in front of a fake shim: the request must survive validate_request."""

    def setUp(self):
        self.seen = seen = []

        class Shim(Quiet):
            def do_GET(self):
                self.send_json({"object": "list", "data": [{"id": "nvidia/Cosmos3-Edge"}]})

            def do_POST(self):
                seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for line in shim_sse().split(b"\n\n")[:-1]:
                    self.wfile.write(line + b"\n\n")
                    self.wfile.flush()
                    time.sleep(0.01)

        self.shim_cm = serve(Shim)
        shim_base, _ = self.shim_cm.__enter__()
        class QuietUi(serve_ui.Handler):
            def log_message(self, *args):
                pass

        self.ui = serve_ui.Server(("127.0.0.1", 0), QuietUi)
        self.ui.backend_port = int(shim_base.rsplit(":", 1)[1])
        self.ui.https_port = None
        threading.Thread(target=self.ui.serve_forever, daemon=True).start()
        self.ui_base = f"http://127.0.0.1:{self.ui.server_address[1]}"

    def tearDown(self):
        self.ui.shutdown()
        self.ui.server_close()
        self.shim_cm.__exit__(None, None, None)

    def make(self, **kw):
        return smoke.Smoke(bridge_hosts=(), robot=False, system=False,
                           live_vision_url=self.ui_base, **kw)

    def test_one_streamed_inference(self):
        s = self.make(image=jpeg())
        s.check_inference()
        r = s.results[-1]
        self.assertEqual(r.status, smoke.PASS, r.detail)
        self.assertIn("A desk with a café mug.", r.detail)
        self.assertIn("native 412 ms", r.detail)
        self.assertEqual(len(self.seen), 1)
        sent = self.seen[0]
        serve_ui.validate_request(sent)            # what reached the shim is what the UI allows
        self.assertEqual((sent["model"], sent["stream"], sent["temperature"], sent["max_tokens"]),
                         ("nvidia/Cosmos3-Edge", True, 0, 16))

    def test_busy_is_a_warning_and_not_retried(self):
        s = self.make(image=jpeg())
        with serve_ui.GENERATION_LOCK:
            s.check_inference()
        self.assertEqual(s.results[-1].status, smoke.WARN, s.results[-1].detail)
        self.assertIn("429", s.results[-1].detail)
        self.assertEqual(self.seen, [])

    def test_no_image_skips(self):
        s = self.make()
        s.check_inference()
        self.assertEqual(s.results[-1].status, smoke.SKIP)
        self.assertIn("--image", s.results[-1].detail)
        self.assertEqual(self.seen, [])


# ----------------------------------------------------------------------------------- on the Orin
def _bridge_module():
    try:
        return _load("reachy_mjpeg_bridge", ROOT / "nvr" / "reachy" / "reachy_mjpeg_bridge.py")
    except (ImportError, SystemExit):
        return None


@unittest.skipIf(smoke.av is None, "needs PyAV (the Orin's /home/orin/reachy_env)")
class BridgeAudioContract(unittest.TestCase):
    """Audio frames shaped like aiortc's (s16 stereo 48 kHz, 960 samples) through the bridge's own
    AudioHub, then read back the way the smoke check reads /audio.mp3."""

    def encode(self, amplitude: float, seconds: float = 1.0) -> bytes:
        bridge = _bridge_module()
        if bridge is None:
            self.skipTest("the bridge's imports (aiohttp, aiortc) are not installed here")
        import array
        import av
        hub = bridge.AudioHub()
        q = hub.subscribe()
        for n in range(int(seconds * 50)):
            pcm = array.array("h")
            for i in range(960):
                v = int(amplitude * 32767 * math.sin(2 * math.pi * 440 * (n * 960 + i) / 48000))
                pcm.extend((v, v))
            frame = av.AudioFrame(format="s16", layout="stereo", samples=960)
            frame.sample_rate = 48000
            frame.planes[0].update(pcm.tobytes())
            hub.feed(frame)
        out = b""
        while not q.empty():
            out += q.get_nowait()
        return out

    def test_tone_is_valid_mp3_at_its_level(self):
        data = self.encode(0.5)                    # -6 dBFS peak: a sine's RMS is 3 dB lower
        scan = smoke.scan_mp3(data)
        self.assertGreater(scan["frames"], 30)
        self.assertEqual(scan["first_offset"], 0)
        self.assertEqual((scan["sample_rate"], scan["channels"], scan["kbps"]), (48000, 2, 96))
        self.assertAlmostEqual(smoke.mp3_level_dbfs(data), -9.0, delta=1.0)

    def test_silence_reads_as_silent(self):
        data = self.encode(0.0)
        self.assertGreater(smoke.scan_mp3(data)["frames"], 30)
        self.assertLess(smoke.mp3_level_dbfs(data), smoke.SILENT_DBFS)


if __name__ == "__main__":
    unittest.main()
