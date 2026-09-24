"""Tests for nvr/reachy/reachy_mjpeg_bridge.py that never go near the robot.

The robot is the scarce thing here: every WebRTC session costs it a hardware encoder and, on daemon
1.11, sockets it never gives back. So nothing below opens a session to it. The daemon, the
signalling server and the Live VLM WebUI are all faked on 127.0.0.1, and a guard fails any test
whose HTTP client tries another host.

Run with the bridge's own venv, from a copy of nvr/ (stdlib unittest, no pytest):

    cd nvr/tests && /home/orin/reachy_env/bin/python -m unittest -v test_reachy_bridge
"""
from __future__ import annotations

import asyncio
import errno
import io
import json
import logging
import os
import signal
import socket
import sys
import tempfile
import time
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
BRIDGE = HERE.parent / "reachy" / "reachy_mjpeg_bridge.py"
OFFER = HERE / "fixtures" / "reachy_offer_daemon_1.11.0.sdp"
sys.path.insert(0, str(BRIDGE.parent))

import aiohttp  # noqa: E402
import av  # noqa: E402
import numpy as np  # noqa: E402
import yarl  # noqa: E402
from aiohttp import web  # noqa: E402
from aiortc import RTCSessionDescription  # noqa: E402

import reachy_mjpeg_bridge as rb  # noqa: E402
from reachy_stream.stream import StreamUnavailableError  # noqa: E402

# Everything fake listens here. The bridge's own default robot host is never used.
LOCAL = "127.0.0.1"

logging.getLogger().setLevel(logging.ERROR if not os.environ.get("BRIDGE_TEST_LOGS") else logging.INFO)

_real_request = aiohttp.ClientSession._request


async def _local_only(self, method, str_or_url, *args, **kwargs):
    host = yarl.URL(str(str_or_url)).host
    if host != LOCAL:
        raise AssertionError(f"a test tried to reach {host}; only {LOCAL} fakes are allowed")
    return await _real_request(self, method, str_or_url, *args, **kwargs)


def setUpModule():
    # ws_connect goes through _request too, so this covers signalling as well as REST.
    aiohttp.ClientSession._request = _local_only


def tearDownModule():
    aiohttp.ClientSession._request = _real_request


# ---------------------------------------------------------------------------------------- helpers
def free_port() -> int:
    """A port nothing listens on: connecting to it is refused, as with a stopped service."""
    with socket.socket() as s:
        s.bind((LOCAL, 0))
        return s.getsockname()[1]


async def serve(app: web.Application, port: int = 0) -> tuple[web.AppRunner, int]:
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, LOCAL, port).start()
    return runner, runner.addresses[0][1]


async def wait_until(predicate, timeout: float = 3.0, what: str = "condition") -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out after {timeout}s waiting for {what}")
        await asyncio.sleep(0.01)


def app_for(bridge: rb.Bridge) -> web.Application:
    """The bridge's routes, as main() wires them."""
    app = web.Application()
    app.router.add_get("/mjpeg", bridge.mjpeg)
    app.router.add_get("/still.jpg", bridge.still)
    app.router.add_get("/audio.mp3", bridge.audio_mp3)
    app.router.add_get("/healthz", bridge.health)
    return app


class Spinning(BaseException):
    """Raised by a guard when a loop runs without ever yielding; BaseException so no handler eats it."""


class FakeDaemon:
    """The robot daemon's REST endpoints the bridge polls. None as a body answers 404."""

    def __init__(self) -> None:
        self.status: dict | None = {"state": "running"}
        self.media: dict | None = {"available": True, "released": False, "no_media": False}
        self.lock: dict | None = {"state": "none"}
        # The journal websocket (/logs/ws/daemon): these lines, then silence. None answers 404.
        self.log_lines: list[str] | None = None
        self.requests: list[str] = []

    def app(self) -> web.Application:
        def endpoint(attr: str):
            async def handler(request: web.Request) -> web.Response:
                self.requests.append(request.path)
                body = getattr(self, attr)
                if body is None:
                    raise web.HTTPNotFound()
                return web.json_response(body)
            return handler

        async def logs(request: web.Request) -> web.StreamResponse:
            self.requests.append(request.path)
            if self.log_lines is None:
                raise web.HTTPNotFound()
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            for line in self.log_lines:
                await ws.send_str(line)
            try:
                await ws.receive()   # hold it open like journalctl -f, until the client leaves
            finally:
                await ws.close()
            return ws

        app = web.Application()
        app.router.add_get("/api/daemon/status", endpoint("status"))
        app.router.add_get("/api/media/status", endpoint("media"))
        app.router.add_get("/api/daemon/robot-app-lock-status", endpoint("lock"))
        async def media_action(request: web.Request) -> web.Response:
            self.requests.append(request.path)
            return web.json_response({"status": "ok"})

        app.router.add_get("/logs/ws/daemon", logs)
        app.router.add_post("/api/media/release", media_action)
        app.router.add_post("/api/media/acquire", media_action)
        return app


def journal(pid: int, *messages: str) -> list[str]:
    return [f"2026-09-24T21:21:36+01:00 reachy-mini launcher.sh[{pid}]: {m}" for m in messages]


EMFILE_LINE = "Error creating GUPnP context: Could not create socketUnable to create socket: Too many open files"


class FakeSignalling:
    """Just enough gst-webrtc-signalling to hand out a session id. It never sends an SDP offer, so no
    media is negotiated and nothing is decoded; the point is what the bridge says on the way out."""

    def __init__(self, producers: list[dict] | None = None) -> None:
        self.producers = [{"id": "producer-1", "meta": {"name": "reachymini"}}] \
            if producers is None else producers
        self.connections = 0
        self.received: list[dict] = []
        self.started = asyncio.Event()

    def app(self) -> web.Application:
        async def handler(request: web.Request) -> web.WebSocketResponse:
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            self.connections += 1
            await ws.send_json({"type": "welcome", "peerId": "bridge-peer"})
            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    break
                data = json.loads(msg.data)
                self.received.append(data)
                if data.get("type") == "list":
                    await ws.send_json({"type": "list", "producers": self.producers})
                elif data.get("type") == "startSession":
                    await ws.send_json({"type": "sessionStarted", "peerId": "producer-1",
                                        "sessionId": "session-1"})
                    self.started.set()
            return ws

        app = web.Application()
        app.router.add_get("/", handler)
        return app


class FakeWebUI:
    """The Live VLM WebUI push API. Records every request, so a test can prove what was NOT called."""

    def __init__(self) -> None:
        self.status = 200
        self.posts = 0
        self.paths: list[str] = []

    def app(self) -> web.Application:
        async def anything(request: web.Request) -> web.Response:
            self.paths.append(request.path)
            if request.method == "POST" and request.path == "/api/push/frame":
                await request.read()
                self.posts += 1
                return web.Response(status=self.status,
                                    text="session stopped" if self.status == 409 else "ok")
            return web.Response(status=404)

        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", anything)
        return app


def mic_frames(n: int, start_pts: int = 10**9):
    """Frames shaped like aiortc's decoded Opus: s16 interleaved stereo, 48 kHz, 20 ms, running pts."""
    t = np.arange(n * 960)
    tone = (np.sin(2 * np.pi * 440 * t / 48000) * 8000).astype(np.int16)
    for i in range(n):
        interleaved = np.repeat(tone[i * 960:(i + 1) * 960], 2).reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(interleaved, format="s16", layout="stereo")
        frame.sample_rate = 48000
        frame.time_base = Fraction(1, 48000)
        frame.pts = start_pts + i * 960
        yield frame


# -------------------------------------------------------------------------------- SDP negotiation
def m_sections(sdp: str) -> list[list[str]]:
    sections: list[list[str]] = []
    for line in sdp.splitlines():
        if line.startswith("m="):
            sections.append([])
        if sections and line:
            sections[-1].append(line)
    return sections


def by_kind(sdp: str) -> dict[str, list[str]]:
    return {s[0].split()[0][2:]: s for s in m_sections(sdp)}


def reordered(offer: str, kinds: list[str]) -> str:
    """The same offer with its m-sections in another order. As a real offer would, the new first
    section carries the transport (port 9) and the rest are bundle-only on port 0."""
    lines = offer.strip().splitlines()
    first_m = next(i for i, line in enumerate(lines) if line.startswith("m="))
    sections = by_kind(offer)
    out = []
    mids = []
    for n, kind in enumerate(kinds):
        section = [line for line in sections[kind] if line != "a=bundle-only"]
        m = section[0].split(" ")
        m[1] = "9" if n == 0 else "0"
        section[0] = " ".join(m)
        if n:
            section.insert(2, "a=bundle-only")
        mids.append(next(line[len("a=mid:"):] for line in section if line.startswith("a=mid:")))
        out.extend(section)
    session = ["a=group:BUNDLE " + " ".join(mids) if line.startswith("a=group:BUNDLE") else line
               for line in lines[:first_m]]
    return "\r\n".join(session + out) + "\r\n"


def rtpmap(section: list[str]) -> dict[str, str]:
    """payload type -> codec name, from a section's a=rtpmap lines."""
    return {line.split()[0][len("a=rtpmap:"):]: line.split()[1].split("/")[0].lower()
            for line in section if line.startswith("a=rtpmap:")}


class H264OnlyAnswerTest(unittest.IsolatedAsyncioTestCase):
    """The answer to daemon 1.11's recorded offer must accept H.264 alone, so the robot never sends
    the RTX packets that kill aiortc's decoder."""

    async def answer_to(self, offer: str) -> str:
        pc = rb.h264_only_pc()
        try:
            await pc.setRemoteDescription(RTCSessionDescription(offer, "offer"))
            return (await pc.createAnswer()).sdp
        finally:
            await pc.close()

    def assert_h264_only(self, offer: str, answer: str) -> None:
        # The answer mirrors the offer's m-line order, whatever that order is.
        self.assertEqual([s[0].split()[0] for s in m_sections(answer)],
                         [s[0].split()[0] for s in m_sections(offer)])
        sections = by_kind(answer)

        video = sections["video"]
        codecs = rtpmap(video)
        self.assertEqual(sorted(set(codecs.values())), ["h264"], video)
        self.assertEqual(video[0].split()[3:], list(codecs), "m=video lists payloads it did not map")
        for unwanted in ("rtx", "red", "ulpfec", "apt="):
            self.assertFalse([line for line in video if unwanted in line.lower()], unwanted)
        self.assertIn("a=recvonly", video)
        # Without RTX, loss recovery rests on keyframe requests: PLI must survive the answer.
        pt = next(iter(codecs))
        self.assertIn(f"a=rtcp-fb:{pt} nack pli", video)

        audio = sections["audio"]
        self.assertEqual(list(rtpmap(audio).values()), ["opus"], audio)
        self.assertIn("a=recvonly", audio)

    async def test_recorded_offer(self):
        offer = OFFER.read_text()
        self.assertIn("rtx/90000", offer, "fixture no longer offers RTX; the test proves nothing")
        self.assert_h264_only(offer, await self.answer_to(offer))

    async def test_audio_first_offer(self):
        # The robot has been seen offering audio before video. The pre-added transceivers must
        # still claim the right m-lines, or video would be negotiated without the H.264 filter.
        offer = reordered(OFFER.read_text(), ["audio", "video", "application"])
        self.assertTrue(m_sections(offer)[0][0].startswith("m=audio 9 "))
        self.assert_h264_only(offer, await self.answer_to(offer))


# --------------------------------------------------------------------------------------- FrameHub
class FrameHubTest(unittest.IsolatedAsyncioTestCase):
    async def test_next_waits_for_a_newer_frame(self):
        hub = rb.FrameHub()
        self.assertFalse(await hub.next(0, timeout=0.05))
        waiters = [asyncio.create_task(hub.next(0, timeout=2)) for _ in range(3)]
        await asyncio.sleep(0.05)
        self.assertFalse(any(w.done() for w in waiters))
        hub.publish(b"one")
        self.assertEqual(await asyncio.gather(*waiters), [True, True, True])
        self.assertEqual(hub.seq, 1)
        # Already newer: no wait at all, even with no time to wait.
        self.assertTrue(await hub.next(0, timeout=0))
        # Nothing newer than the newest: times out.
        self.assertFalse(await hub.next(hub.seq, timeout=0.05))

    async def test_fresh_and_age(self):
        hub = rb.FrameHub()
        self.assertIsNone(hub.age)
        self.assertFalse(hub.fresh)
        hub.publish(b"jpeg")
        self.assertTrue(hub.fresh)
        self.assertLess(hub.age, 1.0)
        hub.at -= rb.FRESH_FOR + 0.1
        self.assertFalse(hub.fresh)
        self.assertGreater(hub.age, rb.FRESH_FOR)
        self.assertEqual(hub.jpeg, b"jpeg")   # stale is not gone; it is just not vouched for


# --------------------------------------------------------------------------------------- AudioHub
class AudioHubTest(unittest.IsolatedAsyncioTestCase):
    async def test_running_clock_pts_encodes_to_valid_mp3(self):
        hub = rb.AudioHub()
        q = hub.subscribe()
        mp3 = bytearray()
        with self.assertNoLogs("reachy-bridge", level="WARNING"):
            for frame in mic_frames(100):   # 2 s of microphone, pts starting at 10^9
                hub.feed(frame)
                while not q.empty():
                    mp3 += q.get_nowait()
        self.assertGreater(len(mp3), 10_000)
        with av.open(io.BytesIO(bytes(mp3)), format="mp3") as container:
            decoded = list(container.decode(audio=0))
        self.assertEqual(decoded[0].sample_rate, 48000)
        self.assertEqual(len(decoded[0].layout.channels), 2)
        self.assertGreater(sum(f.samples for f in decoded), 48000)   # over 1 s survived the trip

    async def test_no_encoder_without_listeners(self):
        hub = rb.AudioHub()
        frames = mic_frames(6)
        for frame in (next(frames), next(frames)):
            hub.feed(frame)
        self.assertIsNone(hub._encoder)
        self.assertEqual(hub.frames_in, 2)
        q = hub.subscribe()
        hub.feed(next(frames))
        self.assertIsNotNone(hub._encoder)
        hub.unsubscribe(q)
        hub.feed(next(frames))
        self.assertIsNone(hub._encoder)
        self.assertEqual((hub.frames_in, hub.listeners), (4, 0))

    async def test_full_listener_drops_its_oldest_and_never_holds_up_others(self):
        hub = rb.AudioHub()
        slow, fast = hub.subscribe(), hub.subscribe()
        got = []
        for i in range(hub.QUEUE_MAX + 10):
            hub._broadcast(b"%d" % i)
            got.append(fast.get_nowait())
        self.assertEqual(got, [b"%d" % i for i in range(hub.QUEUE_MAX + 10)])
        self.assertEqual(slow.qsize(), hub.QUEUE_MAX)
        self.assertEqual(slow.get_nowait(), b"10")   # the ten oldest went, the newest stayed


# ---------------------------------------------------------------------------- camera_unavailable
class CameraUnavailableTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.daemon = FakeDaemon()
        self.runner, port = await serve(self.daemon.app())
        self.bridge = rb.Bridge(LOCAL, free_port(), fps=5, daemon_port=port)
        self.bridge.session = aiohttp.ClientSession()

    async def asyncTearDown(self):
        await self.bridge.session.close()
        await self.runner.cleanup()

    async def test_running_and_free(self):
        self.assertEqual(await self.bridge.camera_unavailable(), (None, None))

    async def test_media_released(self):
        self.daemon.media = {"available": False, "released": True, "no_media": False}
        reason, holder = await self.bridge.camera_unavailable()
        self.assertIn("released", reason)
        self.assertIsNone(holder)

    async def test_no_media(self):
        self.daemon.media = {"no_media": True}
        reason, holder = await self.bridge.camera_unavailable()
        self.assertIn("without media", reason)
        self.assertIsNone(holder)

    async def test_media_endpoint_missing_is_not_a_reason(self):
        # Older daemons lack /api/media/status; the lock check still runs.
        self.daemon.media = None
        self.assertEqual(await self.bridge.camera_unavailable(), (None, None))
        self.daemon.lock = {"state": "local_app", "holder_name": "marionette"}
        self.assertEqual((await self.bridge.camera_unavailable())[1], "marionette")

    async def test_a_released_lock_must_stay_free(self):
        self.daemon.lock = {"state": "local_app", "holder_name": "reachy_mini_testbench"}
        await self.bridge.camera_unavailable()
        self.daemon.lock = {"state": "free", "holder_name": None}
        with mock.patch.object(rb, "LOCK_FREE_GRACE_S", 0.3):
            reason, holder = await self.bridge.camera_unavailable()
            self.assertIn("'reachy_mini_testbench' just released the camera", reason)
            self.assertIsNone(holder)
            await asyncio.sleep(0.35)
            self.assertEqual(await self.bridge.camera_unavailable(), (None, None))

    async def test_local_app_holds_the_camera(self):
        self.daemon.lock = {"state": "local_app", "holder_name": "marionette"}
        reason, holder = await self.bridge.camera_unavailable()
        self.assertIn("'marionette'", reason)
        self.assertEqual(holder, "marionette")

    async def test_daemon_down(self):
        bridge = rb.Bridge(LOCAL, free_port(), fps=5, daemon_port=free_port())
        bridge.session = self.bridge.session
        reason, holder = await bridge.camera_unavailable()
        self.assertIn("unreachable", reason)
        self.assertIsNone(holder)

    async def test_daemon_not_running(self):
        self.daemon.status = {"state": "stopped"}
        reason, holder = await self.bridge.camera_unavailable()
        self.assertIn("'stopped', not running", reason)
        self.assertIsNone(holder)
        self.assertEqual(self.daemon.requests, ["/api/daemon/status"])   # asks nothing further


# ------------------------------------------------------------------------------------ _run_session
class ScriptedClient(rb.AVStreamClient):
    """The real AVStreamClient with the robot replaced by a script run as its session task.

    Only acquire() (which would dial the robot) and the JPEG encode are faked, so session_over,
    end_reason and close() are the bridge's real code.
    """

    script = None
    instances: list[ScriptedClient] = []

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.closed = False
        ScriptedClient.instances.append(self)

    async def acquire(self) -> None:
        self.signalling_reached, self._session_id = True, "session-1"
        self._task = asyncio.get_running_loop().create_task(type(self).script(self))

    async def async_get_image(self, timeout: float = 10.0) -> bytes:
        if self.session_over:
            raise StreamUnavailableError("camera stream is not available")
        return b"jpeg-%d" % self.video_frames

    async def close(self) -> None:
        try:
            await super().close()
        finally:
            self.closed = True   # only reached if close() was awaited, not merely called


async def frames_for(client: ScriptedClient, seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        client.last_video_at = time.monotonic()
        client.video_frames += 1
        await asyncio.sleep(0.01)


class RunSessionTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        ScriptedClient.instances = []
        self.bridge = rb.Bridge(LOCAL, free_port(), fps=50, daemon_port=free_port())
        for patch in (mock.patch.object(rb, "AVStreamClient", ScriptedClient),
                      mock.patch.object(rb, "FIRST_FRAME_TIMEOUT", 0.3),
                      mock.patch.object(rb, "STALL_AFTER", 0.3)):
            patch.start()
            self.addCleanup(patch.stop)

    async def run_script(self, script) -> float:
        ScriptedClient.script = staticmethod(script)
        lived = await asyncio.wait_for(self.bridge._run_session(), 5)
        (client,) = ScriptedClient.instances
        self.assertTrue(client.closed, "close() was not awaited")
        self.assertTrue(client.session_over, "the session task outlived close()")
        self.assertIsNone(self.bridge.client)
        self.assertEqual(self.bridge.state, "reconnecting")
        return lived

    async def test_first_frame_timeout(self):
        async def silent(client):
            await asyncio.sleep(3600)
        self.assertEqual(await self.run_script(silent), 0.0)
        self.assertIn("no video within", self.bridge.reason)
        self.assertEqual(self.bridge.frames.seq, 0)

    async def test_stall_after_live(self):
        async def then_silent(client):
            await frames_for(client, 0.2)
            await asyncio.sleep(3600)
        lived = await self.run_script(then_silent)
        self.assertGreater(lived, 0.0)
        self.assertIn("video stalled", self.bridge.reason)
        self.assertGreater(self.bridge.frames.seq, 0)

    async def test_session_over_without_exception(self):
        async def then_ends(client):
            await frames_for(client, 0.1)
        self.assertGreater(await self.run_script(then_ends), 0.0)
        self.assertEqual(self.bridge.reason, ScriptedClient.instances[0].end_reason)
        self.assertIn("robot ended the session", self.bridge.reason)

    async def test_session_over_with_exception(self):
        async def then_fails(client):
            await frames_for(client, 0.1)
            raise ValueError("bad sdp")
        await self.run_script(then_fails)
        self.assertEqual(self.bridge.reason, "session failed: ValueError: bad sdp")

    async def test_last_frame_and_session_end_together(self):
        # The robot's last frame and the end of its session land between two polls. The upstream
        # client drops its frame when the session ends, so fetching that frame fails; the reason
        # must still be why the session ended, not the failed fetch.
        async def last_frame_then_ends(client):
            await frames_for(client, 0.1)
            client.last_video_at = time.monotonic()
            client.video_frames += 1
        self.assertGreater(await self.run_script(last_frame_then_ends), 0.0)
        self.assertIn("robot ended the session", self.bridge.reason)

    async def test_session_fails_before_any_video(self):
        async def fails(client):
            raise ValueError("bad sdp")
        self.assertEqual(await self.run_script(fails), 0.0)
        self.assertEqual(self.bridge.reason, "session failed: ValueError: bad sdp")

    async def test_acquire_error_still_closes(self):
        class BrokenAcquire(ScriptedClient):
            async def acquire(self):
                raise RuntimeError("no route")
        with mock.patch.object(rb, "AVStreamClient", BrokenAcquire):
            self.assertEqual(await self.bridge._run_session(), 0.0)
        self.assertTrue(ScriptedClient.instances[0].closed)
        self.assertEqual(self.bridge.reason, "session error: RuntimeError: no route")

    async def test_cancelled_session_still_closes(self):
        # SIGTERM lands here: cancellation must unwind through close(), which ends the session.
        async def forever(client):
            await frames_for(client, 3600)
        ScriptedClient.script = staticmethod(forever)
        task = asyncio.create_task(self.bridge._run_session())
        await wait_until(lambda: self.bridge.state == "live", what="live")
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(ScriptedClient.instances[0].closed)


class EndReasonTest(unittest.IsolatedAsyncioTestCase):
    """The real client, against fakes: why a session ended is what /healthz tells a human."""

    async def run_client(self, port: int) -> str:
        async with aiohttp.ClientSession() as http:
            client = rb.AVStreamClient(LOCAL, session=http, port=port, on_audio=lambda frame: None)
            try:
                await client.acquire()
                await wait_until(lambda: client.session_over, what="the session to end")
                return client.end_reason
            finally:
                await client.close()

    async def test_signalling_unreachable(self):
        # The upstream client swallows the connection error. "The robot ended the session" would
        # send a human to look at the robot's apps when its signalling port is not even answering.
        port = free_port()
        reason = await self.run_client(port)
        self.assertIn(f"ws://{LOCAL}:{port}", reason)
        self.assertIn("unreachable", reason)

    async def test_producer_not_advertised(self):
        signalling = FakeSignalling(producers=[])
        runner, port = await serve(signalling.app())
        try:
            reason = await self.run_client(port)
        finally:
            await runner.cleanup()
        self.assertEqual(signalling.connections, 1)
        self.assertIn("started no session", reason)
        self.assertIn("not advertised", reason)


# ------------------------------------------------------------------------------------- supervisor
class SupervisorTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        ScriptedClient.instances = []
        self.daemon = FakeDaemon()
        self.runner, port = await serve(self.daemon.app())
        self.bridge = rb.Bridge(LOCAL, free_port(), fps=50, daemon_port=port)
        self.bridge.session = aiohttp.ClientSession()

    async def asyncTearDown(self):
        await self.bridge.session.close()
        await self.runner.cleanup()

    async def test_dormant_opens_no_session_and_resumes_on_its_own(self):
        async def forever(client):
            await frames_for(client, 3600)
        ScriptedClient.script = staticmethod(forever)
        self.daemon.lock = {"state": "local_app", "holder_name": "marionette"}
        with mock.patch.object(rb, "AVStreamClient", ScriptedClient), \
                mock.patch.object(rb, "DORMANT_POLL_S", 0.05), \
                mock.patch.object(rb, "LOCK_FREE_GRACE_S", 0.5):
            task = asyncio.create_task(self.bridge._supervise())
            try:
                await wait_until(lambda: self.daemon.requests.count("/api/daemon/status") >= 3,
                                 what="three dormant polls")
                self.assertEqual((self.bridge.state, self.bridge.blocked_by), ("dormant", "marionette"))
                self.assertEqual((self.bridge.sessions, ScriptedClient.instances), (0, []))
                self.daemon.lock = {"state": "none"}
                # A freed lock is not dialled into at once: an app restarting frees it briefly.
                await wait_until(lambda: "just released" in (self.bridge.reason or ""),
                                 what="the grace period")
                self.assertEqual(self.bridge.sessions, 0)
                await wait_until(lambda: self.bridge.state == "live", what="live after the app left")
                self.assertEqual(self.bridge.sessions, 1)
                self.assertIsNone(self.bridge.blocked_by)
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self.assertTrue(ScriptedClient.instances[0].closed)

    async def test_stop_during_a_session_teardown_opens_no_further_session(self):
        # SIGTERM can land while close() waits for an ended session to tear down (endSession, the
        # peer connection closing). The upstream client awaits that with CancelledError suppressed,
        # which swallowed the stop too: the supervisor slept its backoff and dialled the robot again.
        async def stalls_then_slow_teardown(client):
            try:
                await frames_for(client, 0.1)
                await asyncio.sleep(3600)
            finally:
                client.tearing_down = True
                await asyncio.sleep(0.5)
        ScriptedClient.script = staticmethod(stalls_then_slow_teardown)
        self.bridge._backoff = 0.05
        with mock.patch.object(rb, "AVStreamClient", ScriptedClient), \
                mock.patch.object(rb, "STALL_AFTER", 0.2):
            task = asyncio.create_task(self.bridge._supervise())
            try:
                await wait_until(lambda: ScriptedClient.instances
                                 and getattr(ScriptedClient.instances[0], "tearing_down", False),
                                 what="the session's teardown")
                task.cancel()
                await asyncio.wait([task], timeout=2)
                await asyncio.sleep(0.3)   # time enough for a swallowed stop to dial again
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(self.bridge.sessions, 1)
        self.assertTrue(task.cancelled())
        self.assertTrue(ScriptedClient.instances[0].closed)


class JournalParsingTest(unittest.TestCase):
    def test_pid_from_the_newest_line_that_has_one(self):
        lines = journal(76389, "old") + ["no pid here"] + journal(81234, "new")
        self.assertEqual(rb.daemon_pid(lines), "81234")
        self.assertEqual(rb.daemon_pid(["2026 reachy-mini python[76389]: x"]), "76389")
        self.assertIsNone(rb.daemon_pid(["nothing", ""]))

    def test_fd_exhaustion_is_the_exact_libc_message(self):
        self.assertTrue(rb.fd_exhausted(journal(1, "consumer added", EMFILE_LINE)))
        self.assertFalse(rb.fd_exhausted(journal(1, "consumer added", "too many consumers")))


class FdExhaustionTest(unittest.IsolatedAsyncioTestCase):
    """A daemon out of descriptors: stop dialling (each dial kills the robot's camera pipeline for its
    local apps too), and start again by itself once the daemon is a new process."""

    async def asyncSetUp(self):
        ScriptedClient.instances = []
        self.daemon = FakeDaemon()
        self.runner, port = await serve(self.daemon.app())
        self.bridge = rb.Bridge(LOCAL, free_port(), fps=50, daemon_port=port)
        self.bridge.session = aiohttp.ClientSession()
        self.bridge._backoff = 0.01
        for patch in (mock.patch.object(rb, "AVStreamClient", ScriptedClient),
                      mock.patch.object(rb, "DORMANT_POLL_S", 0.05),
                      mock.patch.object(rb, "FD_EXHAUSTED_RECHECK_S", 0.2),
                      mock.patch.object(rb, "MEDIA_REBUILD_PAUSE_S", 0.01)):
            patch.start()
            self.addCleanup(patch.stop)

    async def asyncTearDown(self):
        await self.bridge.session.close()
        await self.runner.cleanup()

    async def test_holds_off_until_the_daemon_restarts(self):
        async def refused_or_live(client):
            if rb.fd_exhausted(self.daemon.log_lines or []):
                return                      # the robot ends the session before any video
            await frames_for(client, 3600)
        ScriptedClient.script = staticmethod(refused_or_live)
        self.daemon.log_lines = journal(76389, "consumer added", EMFILE_LINE, "consumer removed")
        task = asyncio.create_task(self.bridge._supervise())
        try:
            await wait_until(lambda: self.bridge.fd_exhausted_pid == "76389", what="exhaustion noted")
            await asyncio.sleep(0.5)        # several dormant polls and re-checks
            self.assertEqual(self.bridge.sessions, 1, "dialled an exhausted daemon again")
            # The failed session stopped the robot's camera pipeline; with no app running, the
            # bridge rebuilt it, exactly once.
            self.assertEqual([p for p in self.daemon.requests if p.startswith("/api/media/r")
                              or p.endswith("/acquire")], ["/api/media/release", "/api/media/acquire"])
            self.assertEqual(self.bridge.state, "dormant")
            self.assertIn("out of file descriptors", self.bridge.reason)
            self.assertIn("Power-cycle", self.bridge.reason)
            # The robot comes back as a new process with a clean journal.
            self.daemon.log_lines = journal(90001, "Media hardware re-acquired.")
            await wait_until(lambda: self.bridge.state == "live", timeout=3, what="live after restart")
            self.assertIsNone(self.bridge.fd_exhausted_pid)
            self.assertEqual(self.bridge.sessions, 2)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_no_media_rebuild_while_an_app_runs(self):
        self.daemon.lock = {"state": "local_app", "holder_name": "reachy_mini_testbench"}
        self.daemon.log_lines = journal(76389, EMFILE_LINE)
        await self.bridge._repair_robot_camera()
        self.assertNotIn("/api/media/release", self.daemon.requests)

    async def test_a_plain_failure_is_not_mistaken_for_exhaustion(self):
        async def refused(client):
            return
        ScriptedClient.script = staticmethod(refused)
        self.daemon.log_lines = journal(76389, "consumer added", "consumer removed")
        task = asyncio.create_task(self.bridge._supervise())
        try:
            await wait_until(lambda: self.bridge.sessions >= 2, what="an ordinary retry")
            self.assertIsNone(self.bridge.fd_exhausted_pid)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_no_journal_endpoint_falls_back_to_backoff(self):
        async def refused(client):
            return
        ScriptedClient.script = staticmethod(refused)
        self.daemon.log_lines = None        # an older daemon without /logs/ws/daemon
        task = asyncio.create_task(self.bridge._supervise())
        try:
            await wait_until(lambda: self.bridge.sessions >= 2, what="an ordinary retry")
            self.assertIsNone(self.bridge.fd_exhausted_pid)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


# ------------------------------------------------------------------------------ Live VLM WebUI push
class PushLoopTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.webui = FakeWebUI()
        self.webui_port = free_port()
        self.runner = None
        self.bridge = rb.Bridge(
            LOCAL, free_port(), fps=20, daemon_port=free_port(),
            push_url=f"http://{LOCAL}:{self.webui_port}/api/push/frame?session_id=reachy")
        self.bridge.session = aiohttp.ClientSession()
        for patch in (mock.patch.object(rb, "PUSH_RETRY_S", 0.2),
                      mock.patch.object(rb, "PUSH_UNREACHABLE_RETRY_S", 0.2)):
            patch.start()
            self.addCleanup(patch.stop)
        self.tasks: list[asyncio.Task] = []

    async def asyncTearDown(self):
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        await self.bridge.session.close()
        if self.runner:
            await self.runner.cleanup()

    async def start_webui(self):
        self.runner, _ = await serve(self.webui.app(), self.webui_port)

    def start(self, video: bool = True):
        async def camera():
            while True:
                self.bridge.frames.publish(b"jpeg")
                await asyncio.sleep(0.05)
        if video:
            self.tasks.append(asyncio.create_task(camera()))
        self.tasks.append(asyncio.create_task(self.bridge._push_loop()))

    async def test_200_counts_pushes(self):
        await self.start_webui()
        self.start()
        await wait_until(lambda: self.bridge.push["pushed"] >= 3, what="three pushes")
        self.assertEqual(self.bridge.push["state"], "pushing")
        self.assertEqual(self.bridge.push["pushed"], self.webui.posts)

    async def test_409_waits_for_start_and_never_starts_the_session_itself(self):
        self.webui.status = 409
        await self.start_webui()
        self.start()
        await wait_until(lambda: self.webui.posts >= 1, what="a first push")
        await asyncio.sleep(1.0)
        self.assertIn("Start", self.bridge.push["state"])
        self.assertEqual(self.bridge.push["pushed"], 0)
        # It waits between tries instead of posting at the frame rate (20 per second here).
        self.assertLessEqual(self.webui.posts, 8)
        self.assertEqual(set(self.webui.paths), {"/api/push/frame"})   # never /api/push/start
        self.webui.status = 200
        await wait_until(lambda: self.bridge.push["state"] == "pushing", what="pushing after Start")

    async def test_refused_then_recovers(self):
        self.start()
        await wait_until(lambda: self.bridge.push["state"] == "WebUI not reachable (stopped?)",
                         what="the unreachable state")
        self.assertEqual(self.bridge.push["pushed"], 0)
        await self.start_webui()
        await wait_until(lambda: self.bridge.push["state"] == "pushing", what="recovery")
        self.assertGreater(self.bridge.push["pushed"], 0)

    async def test_a_stale_unpushed_frame_does_not_spin_the_event_loop(self):
        # The WebUI was down while the last frames arrived, and the video stopped more than
        # FRESH_FOR before the retry wait ended: the newest frame is newer than the last push but
        # too old to send. Every await in the loop then completes without suspending, so unless
        # that frame is consumed the loop never yields and the whole bridge freezes.
        await self.start_webui()
        self.bridge.frames.publish(b"jpeg")
        self.bridge.frames.at -= rb.FRESH_FOR + 1
        real_next, calls = self.bridge.frames.next, []

        async def guarded_next(after_seq, timeout):
            calls.append(after_seq)
            if len(calls) > 1000:
                raise Spinning("push loop spun without yielding")
            return await real_next(after_seq, timeout)

        self.bridge.frames.next = guarded_next
        self.start(video=False)
        await asyncio.sleep(0.3)
        push = self.tasks[-1]
        if push.done() and isinstance(push.exception(), Spinning):
            self.fail(f"push loop spun: {len(calls)} calls to next() without yielding")
        self.assertLess(len(calls), 5)
        self.assertEqual(self.bridge.push["state"], "waiting for video")
        self.assertEqual(self.webui.posts, 0)   # a stale frame is never pushed
        self.bridge.frames.publish(b"fresh")
        await wait_until(lambda: self.bridge.push["pushed"] == 1, what="the fresh frame pushed")


# ---------------------------------------------------------------------------------------- binding
class BindTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.runner = web.AppRunner(web.Application())
        await self.runner.setup()

    async def asyncTearDown(self):
        await self.runner.cleanup()

    async def test_missing_address_waits(self):
        # 192.0.2.1 (TEST-NET-1) is on no interface here: EADDRNOTAVAIL, as for the docker gateway
        # before dockerd is up.
        with mock.patch.object(rb, "BIND_RETRY_S", 0.02):
            task = asyncio.create_task(rb._bind(self.runner, "192.0.2.1", free_port()))
            await asyncio.sleep(0.3)
            try:
                self.assertFalse(task.done(), task.done() and task.exception())
                # Each failed try registered a site with the runner; they must not pile up while
                # waiting, which at boot without docker is indefinitely.
                self.assertEqual(len(self.runner.sites), 0)
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def test_address_in_use_raises(self):
        with socket.socket() as taken:
            taken.bind((LOCAL, 0))
            taken.listen()
            with self.assertRaises(OSError) as cm:
                await asyncio.wait_for(rb._bind(self.runner, LOCAL, taken.getsockname()[1]), 5)
        self.assertEqual(cm.exception.errno, errno.EADDRINUSE)

    async def test_binds(self):
        port = free_port()
        await asyncio.wait_for(rb._bind(self.runner, LOCAL, port), 5)
        self.assertEqual(self.runner.addresses, [(LOCAL, port)])


# ------------------------------------------------------------------------------------ HTTP clients
class HttpTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.bridge = rb.Bridge(LOCAL, free_port(), fps=5, daemon_port=free_port())
        self.runner, self.port = await serve(app_for(self.bridge))
        patch = mock.patch.object(rb, "CLIENT_IDLE_CHECK_S", 0.1)
        patch.start()
        self.addCleanup(patch.stop)

    async def asyncTearDown(self):
        await self.runner.cleanup()

    async def open(self, path: str):
        reader, writer = await asyncio.open_connection(LOCAL, self.port)
        writer.write(f"GET {path} HTTP/1.1\r\nHost: bridge\r\n\r\n".encode())
        await writer.drain()
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 2)
        self.assertIn(b" 200 ", head.split(b"\r\n")[0])
        return reader, writer

    async def test_mjpeg_client_leaving_while_no_video_is_released(self):
        # Dormant: nothing is published, so nothing is written, and a write failing is the only
        # other way the handler learns its client left.
        _, writer = await self.open("/mjpeg")
        await wait_until(lambda: self.bridge.mjpeg_clients == 1, what="the client counted")
        writer.close()
        await writer.wait_closed()
        await wait_until(lambda: self.bridge.mjpeg_clients == 0, timeout=2,
                         what="the departed MJPEG client released")

    async def test_audio_listener_leaving_while_no_audio_is_released(self):
        _, writer = await self.open("/audio.mp3")
        await wait_until(lambda: self.bridge.audio.listeners == 1, what="the listener counted")
        writer.close()
        await writer.wait_closed()
        await wait_until(lambda: self.bridge.audio.listeners == 0, timeout=2,
                         what="the departed audio listener released")

    async def test_mjpeg_new_client_gets_the_fresh_frame_then_only_new_ones(self):
        self.bridge.frames.publish(b"FRAME-1")
        reader, writer = await self.open("/mjpeg")
        try:
            part = await asyncio.wait_for(reader.readuntil(b"FRAME-1\r\n"), 2)
            self.assertIn(b"Content-Type: image/jpeg", part)
            self.bridge.frames.publish(b"FRAME-2")
            await asyncio.wait_for(reader.readuntil(b"FRAME-2\r\n"), 2)
        finally:
            writer.close()

    async def test_mjpeg_new_client_is_not_sent_a_stale_frame(self):
        self.bridge.frames.publish(b"OLD-FRAME")
        self.bridge.frames.at -= rb.FRESH_FOR + 60
        reader, writer = await self.open("/mjpeg")
        try:
            with self.assertRaises(asyncio.TimeoutError):
                await asyncio.wait_for(reader.readuntil(b"OLD-FRAME\r\n"), 0.5)
            self.bridge.frames.publish(b"NEW-FRAME")
            data = await asyncio.wait_for(reader.readuntil(b"NEW-FRAME\r\n"), 2)
            self.assertNotIn(b"OLD-FRAME", data)
        finally:
            writer.close()

    async def test_still_is_503_without_a_fresh_frame(self):
        async with aiohttp.ClientSession() as http:
            url = f"http://{LOCAL}:{self.port}/still.jpg"
            async with http.get(url) as r:
                self.assertEqual(r.status, 503)
            self.bridge.frames.publish(b"jpeg")
            async with http.get(url) as r:
                self.assertEqual((r.status, r.content_type, await r.read()),
                                 (200, "image/jpeg", b"jpeg"))

    async def test_health_fields(self):
        # porch-feed, the Live UIs and the docs read these names; adding is fine, renaming is not.
        async with aiohttp.ClientSession() as http:
            async with http.get(f"http://{LOCAL}:{self.port}/healthz") as r:
                health = await r.json()
        for key in ("state", "reason", "live", "has_frame", "frames", "last_frame_age_s", "stale_s",
                    "sessions", "restarts", "failed_streak", "blocked_by", "mjpeg_clients",
                    "audio", "push"):
            self.assertIn(key, health)
        self.assertLessEqual({"live", "frames", "last_frame_age_s", "listeners"}, set(health["audio"]))
        self.assertLessEqual({"enabled", "state", "pushed", "detail"}, set(health["push"]))
        self.assertEqual((health["state"], health["live"], health["has_frame"]),
                         ("starting", False, False))


# ------------------------------------------------------------------------------ the whole process
class ProcessTest(unittest.IsolatedAsyncioTestCase):
    """The real bridge as systemd runs it, against a fake daemon and fake signalling."""

    async def asyncSetUp(self):
        self.daemon, self.signalling = FakeDaemon(), FakeSignalling()
        self.daemon_runner, self.daemon_port = await serve(self.daemon.app())
        self.sig_runner, self.sig_port = await serve(self.signalling.app())
        self.log = tempfile.TemporaryFile()

    async def asyncTearDown(self):
        await self.sig_runner.cleanup()
        await self.daemon_runner.cleanup()
        self.log.close()

    async def spawn(self, listen_port: int) -> asyncio.subprocess.Process:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(BRIDGE), "--robot-host", LOCAL,
            "--robot-port", str(self.sig_port), "--daemon-port", str(self.daemon_port),
            "--listen", LOCAL, "--listen-port", str(listen_port),
            stdout=self.log, stderr=self.log, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        self.addCleanup(lambda: proc.returncode is None and proc.kill())
        return proc

    def output(self) -> str:
        self.log.seek(0)
        return self.log.read().decode(errors="replace")

    async def test_sigterm_ends_the_robot_session(self):
        proc = await self.spawn(free_port())
        await asyncio.wait_for(self.signalling.started.wait(), 20)
        proc.send_signal(signal.SIGTERM)
        # The unit gives it TimeoutStopSec=10 before SIGKILL, which would skip endSession.
        rc = await asyncio.wait_for(proc.wait(), 10)
        out = self.output()
        self.assertIn({"type": "endSession", "sessionId": "session-1"}, self.signalling.received, out)
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.signalling.connections, 1, out)
        self.assertNotIn("Traceback", out)
        self.assertNotIn("Unclosed", out)

    async def test_port_conflict_exits_without_a_robot_session(self):
        # A port conflict exits the process and systemd restarts it every 5 s. If each start got as
        # far as a robot session, that loop would be the session churn that exhausts the daemon.
        with socket.socket() as taken:
            taken.bind((LOCAL, 0))
            taken.listen()
            proc = await self.spawn(taken.getsockname()[1])
            rc = await asyncio.wait_for(proc.wait(), 10)
        out = self.output()
        self.assertNotEqual(rc, 0, out)
        self.assertIn("address already in use", out.lower())
        self.assertEqual(self.signalling.connections, 0, out)


if __name__ == "__main__":
    unittest.main()
