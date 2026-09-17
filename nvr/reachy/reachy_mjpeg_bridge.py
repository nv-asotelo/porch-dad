#!/usr/bin/env python3
"""Serve the Reachy Mini camera as MJPEG, on the Jetson.

Frigate can only ingest what ffmpeg can open, and the robot's camera speaks WebRTC. This pulls
that stream with aiortc and re-serves it as MJPEG on loopback, so Frigate/go2rtc can read it
locally.

Why here and not on the workstation: the previous arrangement had a desktop transcoding for the
NVR, so the cameras died whenever that machine slept. Nothing outside this box is in the path now.

Why aiortc and not the Reachy SDK: the SDK's WebRTC backend needs GStreamer's `webrtcsrc` from
gst-plugins-rs, which is not packaged for Linux and must be built from Rust source. aiortc is pure
Python. It is pinned to 1.10.1 deliberately -- 1.14 hands a `cryptography` certificate to
pyopenssl's use_certificate(), which only pyopenssl >= 25 accepts, and Home Assistant 2024.12
pins 24.2.1.

The heavy lifting is Pollen's own ReachyMiniStreamClient, imported from the Home Assistant custom
component rather than reimplemented, so this tracks their signalling handling.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import sys
import time
from pathlib import Path

from aiohttp import web

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG = logging.getLogger("reachy-mjpeg")

# Restart the session after this long with a byte-identical frame.
STALE_AFTER = 12.0

# Pollen's stream client ships inside the HA custom component and has no HA imports.
HA_COMPONENTS = Path("/home/orin/nvr/reachy")
sys.path.insert(0, str(HA_COMPONENTS))

try:
    import aiohttp

    from reachy_stream.stream import ReachyMiniStreamClient
except ImportError as e:  # pragma: no cover
    sys.exit(f"cannot import the vendored stream client ({e})")


class Bridge:
    def __init__(self, host: str, port: int, fps: float):
        self.host = host
        self.port = port
        self.interval = 1.0 / fps if fps > 0 else 0.2
        self.session: aiohttp.ClientSession | None = None
        self.client: ReachyMiniStreamClient | None = None
        self.latest: bytes | None = None
        self._frames = 0
        # Staleness watchdog. async_get_image() waits on the client's frame event and returns its
        # cached frame; if the WebRTC track stalls while that event stays set, it hands back the
        # SAME frame indefinitely and never raises. The feed then looks alive - frame counter
        # climbing, HTTP 200s - while showing a still picture. Only comparing the bytes catches it.
        self._last_digest: str | None = None
        self._digest_since = 0.0
        self._restarts = 0

    async def start(self) -> None:
        self.session = aiohttp.ClientSession()
        self.client = ReachyMiniStreamClient(self.host, session=self.session, port=self.port)
        await self.client.acquire()
        asyncio.create_task(self._pump())

    async def _pump(self) -> None:
        """Poll stills into `latest`.

        The client hands out JPEG stills rather than raw frames, so this is a copy, not a
        re-encode. On failure it re-acquires instead of exiting: a WebRTC stream can drop without
        raising, and a bridge that dies silently is worse than one that reconnects.
        """
        while True:
            try:
                img = await self.client.async_get_image()
                if img:
                    digest = hashlib.md5(img).hexdigest()
                    now = time.monotonic()
                    if digest != self._last_digest:
                        self._last_digest = digest
                        self._digest_since = now
                    elif now - self._digest_since > STALE_AFTER:
                        _LOG.warning(
                            "identical frame for %.0fs - stream stalled, restarting session",
                            now - self._digest_since,
                        )
                        await self._restart()
                        continue
                    self.latest = img
                    self._frames += 1
                    if self._frames % 200 == 0:
                        _LOG.info("served %d frames (%d restarts)", self._frames, self._restarts)
                else:
                    await asyncio.sleep(0.5)
            except Exception as e:
                _LOG.warning("stream error (%s), re-acquiring", e)
                try:
                    await self.client.release()
                except Exception:
                    pass
                await asyncio.sleep(3)
                try:
                    await self.client.acquire()
                except Exception as e2:
                    _LOG.error("re-acquire failed: %s", e2)
                    await asyncio.sleep(5)
            await asyncio.sleep(self.interval)

    async def _restart(self) -> None:
        """Tear the session down and build a new one. Release alone is not enough: the client
        caches the last frame, so a fresh acquire is what actually re-establishes the track."""
        self._restarts += 1
        self._last_digest = None
        self._digest_since = time.monotonic()
        try:
            await self.client.release()
            await self.client.async_shutdown()
        except Exception as e:
            _LOG.debug("release during restart: %s", e)
        await asyncio.sleep(2)
        try:
            self.client = ReachyMiniStreamClient(self.host, session=self.session, port=self.port)
            await self.client.acquire()
            _LOG.info("stream session restarted")
        except Exception as e:
            _LOG.error("restart failed: %s", e)
            await asyncio.sleep(5)

    async def mjpeg(self, request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "multipart/x-mixed-replace; boundary=frame",
                "Cache-Control": "no-store",
            },
        )
        await resp.prepare(request)
        try:
            while True:
                if self.latest:
                    await resp.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(self.latest)}\r\n\r\n".encode()
                        + self.latest
                        + b"\r\n"
                    )
                await asyncio.sleep(self.interval)
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        return resp

    async def still(self, request: web.Request) -> web.Response:
        if not self.latest:
            raise web.HTTPServiceUnavailable(text="no frame yet")
        return web.Response(body=self.latest, content_type="image/jpeg")

    async def health(self, request: web.Request) -> web.Response:
        # `stale_s` is the honest liveness signal: frames counts calls, not new pictures.
        stale = round(time.monotonic() - self._digest_since, 1) if self._digest_since else None
        return web.json_response({
            "frames": self._frames,
            "has_frame": self.latest is not None,
            "stale_s": stale,
            "restarts": self._restarts,
            "live": bool(self.latest is not None and (stale is None or stale < STALE_AFTER)),
        })


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--robot-host", default="192.168.6.162")
    p.add_argument("--robot-port", type=int, default=8443)
    p.add_argument("--listen", default="127.0.0.1", help="bind address; loopback keeps it local")
    p.add_argument("--listen-port", type=int, default=8099)
    p.add_argument("--fps", type=float, default=5.0)
    args = p.parse_args()

    bridge = Bridge(args.robot_host, args.robot_port, args.fps)
    await bridge.start()

    app = web.Application()
    app.router.add_get("/mjpeg", bridge.mjpeg)
    app.router.add_get("/still.jpg", bridge.still)
    app.router.add_get("/healthz", bridge.health)

    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, args.listen, args.listen_port).start()
    _LOG.info("MJPEG on http://%s:%d/mjpeg (robot %s:%d)",
              args.listen, args.listen_port, args.robot_host, args.robot_port)
    await asyncio.Event().wait()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
