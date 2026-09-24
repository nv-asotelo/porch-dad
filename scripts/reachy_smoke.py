#!/usr/bin/env python3
"""Read-only smoke check of the Reachy Mini path on the Orin: the robot, its bridge, both Live UIs,
the shim, Frigate, and what will come back after a reboot.

Run it on the Orin after a deploy, and after EVERY boot of a cloned SD card. A clone once came up
looking healthy while docker, containerd, the camera bridge, porch-feed and the Live VLM WebUI were
disabled at boot and frigate, mosquitto and ring-mqtt were on restart=no. Nothing was visibly wrong
until the next reboot, when most of the NVR would simply not have come back.

    python3 scripts/reachy_smoke.py                             # table; exit 1 if anything FAILs
    python3 scripts/reachy_smoke.py --json                      # the same, for a script
    /home/orin/reachy_env/bin/python scripts/reachy_smoke.py    # adds the microphone level (PyAV)
    python3 scripts/reachy_smoke.py --image some.jpg            # inference without the robot

Every check says PASS, WARN, FAIL or SKIP with a one-line reason. SKIP is for things that are off on
purpose (the Live VLM WebUI often is; the command centre records when a human turned it off) or
cannot be judged because something upstream already failed: one root cause should read as one
FAIL, not ten.

It changes nothing. It never opens a WebRTC session to the robot - the daemon leaks sockets on every
session (deploy/07 section 1), so a smoke check that connected would slowly use up the thing it
checks - and asks the robot only GETs on its REST API. What it does cost: a few seconds of MJPEG and
MP3 from the bridge (whose MP3 encoder runs only while someone listens), and ONE 16-token inference
through Live Vision (--no-inference skips it).

Stdlib only, so it runs on a fresh clone with nothing installed. PyAV is used if it happens to
import, for the one thing the stdlib cannot do: decode the MP3 and say whether the microphone hears
anything.
"""
from __future__ import annotations

import argparse
import array
import codecs
import http.client
import io
import json
import math
import os
import re
import shutil
import socket
import ssl
import subprocess
import sys
import time
from base64 import b64encode
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, urlsplit

try:   # optional; /home/orin/reachy_env has it, the system python3 does not
    import av
    # FFmpeg's own chatter ("Estimating duration from bitrate") would land in the middle of the
    # table. A real decode failure still surfaces, as a WARN on the audio check.
    av.logging.set_level(av.logging.ERROR)
except ImportError:
    av = None

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"

ROBOT_HOST = "192.168.6.162"
# REST only. Never the signalling port (8443): a WebRTC session is the one thing this must not open.
ROBOT_REST_PORT = 8000
BRIDGE_PORT = 8099
# The bridge binds both. Loopback serves this host (porch-feed, Live Vision); the docker gateway
# serves Frigate, whose container cannot see the host's 127.0.0.1. It is one process, so if only
# the gateway fails, docker is not up and Frigate is blind while everything else looks fine.
BRIDGE_HOSTS = ("127.0.0.1", "172.17.0.1")
WEBUI_URL = "https://127.0.0.1:8090"
WEBUI_SESSION = "reachy"
LIVE_VISION_URL = "http://127.0.0.1:8092"
SHIM_URL = "http://127.0.0.1:8000"
FRIGATE_URL = "http://127.0.0.1:5000"
FRIGATE_CAMERA = "reachy_mini"
# The microphone is its own go2rtc stream, offered as a live-view choice (deploy/07 section 6),
# so that Frigate's live player, which asks for audio on every view, does not start it.
FRIGATE_MIC_STREAM = "reachy_mini_mic"
ENGINE_LINK = "/opt/tensorrt-edgellm/models/default"
FEED_CONFIG = Path("/home/orin/nvr/feed/config.yaml")
REPO_FEED_CONFIG = Path(__file__).resolve().parents[1] / "nvr" / "feed" / "config.yaml"

SAMPLE_S = 3.0
# The bridge publishes ~5 fps, so 3 s should carry ~15 frames. Three is "it is moving", not a rate.
MIN_MJPEG_FRAMES = 3
# 96 kb/s is ~12 KB/s; 8 KB in 3 s allows for the encoder's start-up delay and nothing more.
MIN_AUDIO_BYTES = 8 * 1024
# Not a measured room level. Digital silence decodes to -inf and 16-bit dither sits near -90, while
# a live microphone in a quiet room reads well above this - so this catches a dead or muted capture
# path, not a quiet house.
SILENT_DBFS = -70.0
# Live Vision's own ceiling is 120 s (serve_ui.MAX_SECONDS); a healthy 16-token answer takes ~1 s.
INFERENCE_TIMEOUT_S = 90.0
LIVE_VISION_MAX_BODY = 2 * 1024 * 1024   # serve_ui.MAX_BODY: larger requests are refused with 413

SHIM_UNIT = "cosmos3-edge-shim.service"
# The shim loads its engine inside uvicorn's startup hook, and uvicorn opens the port only once
# that hook returns. So for the ~40-70 s of a load, :8000 refuses connections exactly as a dead
# shim would, and only systemd can tell the two apart. Well past a load, a unit that is active but
# still not listening is stuck, not loading.
SHIM_LOAD_GRACE_S = 300

# Must be enabled, or the NVR does not come back after a reboot.
BOOT_UNITS = ("docker.service", "containerd.service", SHIM_UNIT,
              "reachy-mjpeg-bridge.service", "porch-feed.service")
# The two Live UIs are switched off on purpose at times, so "enabled" is not the right answer for
# them; the right answer is whatever the command centre last recorded. unit -> its key in
# porch-feed's `services:` list, which is the key service_intent.json is written under.
INTENT_UNITS = {"cosmos-edge-ui.service": "livevision", "live-vlm-webui-fork.service": "vlm"}
UI_UNITS = {key: unit for unit, key in INTENT_UNITS.items()}
CONTAINERS = ("frigate", "mosquitto", "ring-mqtt")
# Home Assistant is reported but never FAILs. It is not in nvr/docker-compose.yml, and while its
# Reachy Mini camera entity is enabled it is a second WebRTC client of the robot (deploy/07 §7),
# so it may well be off on purpose.
HA_CONTAINER = "homeassistant"
HA_CONFIG = Path("/home/orin/homeassistant-config")


@dataclass
class Result:
    check: str
    status: str
    detail: str


# ----------------------------------------------------------------------------------- parsers
def jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    """(width, height) from the first SOF segment, or None if this is not a well-formed JPEG.

    Walks the segment chain from SOI rather than searching for FFC0: a search also matches bytes
    inside an EXIF thumbnail or a DHT table, and misses progressive (FFC2) images entirely.
    """
    if len(data) < 4 or data[0] != 0xFF or data[1] != 0xD8:
        return None
    i = 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            return None
        marker = data[i + 1]
        if marker == 0xFF:                                   # fill byte before a marker
            i += 1
            continue
        if marker == 0x01 or 0xD0 <= marker <= 0xD8:         # TEM, RSTn, SOI: no length field
            i += 2
            continue
        if marker in (0xD9, 0xDA):                           # EOI or scan data before any SOF
            return None
        length = int.from_bytes(data[i + 2:i + 4], "big")
        if length < 2:
            return None
        # C0-CF are the SOF markers, except C4 (DHT), C8 (reserved) and CC (DAC).
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            if i + 9 > len(data):
                return None
            height = int.from_bytes(data[i + 5:i + 7], "big")
            width = int.from_bytes(data[i + 7:i + 9], "big")
            return (width, height) if width and height else None
        i += 2 + length
    return None


def split_mjpeg(buf: bytes, boundary: bytes = b"frame") -> tuple[list[bytes], bytes]:
    """Complete parts of a multipart/x-mixed-replace body, and whatever is left over.

    Trusts Content-Length when a part has one (the bridge always sends it) and otherwise ends the
    part at the next delimiter, which is what a JPEG-agnostic reader such as ffmpeg does.
    """
    delim = b"--" + boundary
    parts: list[bytes] = []
    while True:
        start = buf.find(delim)
        if start < 0:
            return parts, buf
        head_end = buf.find(b"\r\n\r\n", start)
        if head_end < 0:
            return parts, buf[start:]
        headers = buf[start + len(delim):head_end].decode("latin-1")
        body_start = head_end + 4
        m = re.search(r"(?im)^content-length:\s*(\d+)\s*$", headers)
        if m:
            end = body_start + int(m.group(1))
            if end > len(buf):
                return parts, buf[start:]
        else:
            end = buf.find(b"\r\n" + delim, body_start)
            if end < 0:
                return parts, buf[start:]
        parts.append(buf[body_start:end])
        buf = buf[end:]


# MPEG audio frame header tables (ISO 11172-3 / 13818-3), kb/s by bitrate index. Index 0 is "free
# format" and 15 is invalid; neither is accepted. Keyed by (MPEG-1?, layer).
_MP3_KBPS = {
    (True, 1): (0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448),
    (True, 2): (0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384),
    (True, 3): (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320),
    (False, 1): (0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256),
    (False, 2): (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160),
    (False, 3): (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160),
}
# Sample rates by the header's version bits: 3 = MPEG-1, 2 = MPEG-2, 0 = MPEG-2.5 (1 is reserved).
_MP3_RATES = {3: (44100, 48000, 32000), 2: (22050, 24000, 16000), 0: (11025, 12000, 8000)}
_MP3_VERSION = {3: "1", 2: "2", 0: "2.5"}


def mp3_header(data: bytes, i: int = 0) -> dict | None:
    """Decode the 4-byte MPEG audio frame header at data[i], or None if there is not one."""
    if i < 0 or i + 4 > len(data) or data[i] != 0xFF or data[i + 1] & 0xE0 != 0xE0:
        return None
    b1, b2, b3 = data[i + 1], data[i + 2], data[i + 3]
    version = (b1 >> 3) & 3
    layer = 4 - ((b1 >> 1) & 3)          # header bits 11 = layer I, 10 = II, 01 = III, 00 reserved
    index, rate_index, padding = b2 >> 4, (b2 >> 2) & 3, (b2 >> 1) & 1
    if version == 1 or layer == 4 or index in (0, 15) or rate_index == 3 or b3 & 3 == 2:
        return None
    mpeg1 = version == 3
    kbps = _MP3_KBPS[(mpeg1, layer)][index]
    rate = _MP3_RATES[version][rate_index]
    if layer == 1:
        length, samples = (12000 * kbps // rate + padding) * 4, 384
    elif layer == 3 and not mpeg1:
        length, samples = 72000 * kbps // rate + padding, 576
    else:
        length, samples = 144000 * kbps // rate + padding, 1152
    return {"version": _MP3_VERSION[version], "layer": layer, "kbps": kbps, "sample_rate": rate,
            "channels": 1 if b3 >> 6 == 3 else 2, "length": length, "samples": samples}


def scan_mp3(data: bytes) -> dict:
    """Count the complete MPEG audio frames in a byte stream by walking the header chain.

    An 11-bit sync pattern turns up by chance every few KB of compressed audio, so a header only
    counts if it is chained: the next frame's header sits exactly where this one's length says,
    or this frame follows directly on from one that was already counted, or it ends the data.
    """
    out = {"frames": 0, "bytes": 0, "first_offset": None, "seconds": 0.0,
           "sample_rate": None, "channels": None, "kbps": None, "version": None, "layer": None}
    i, prev_end, n = 0, -1, len(data)
    while i + 4 <= n:
        h = mp3_header(data, i)
        if h is not None and i + h["length"] <= n:
            end = i + h["length"]
            if i == prev_end or end == n or mp3_header(data, end) is not None:
                if out["first_offset"] is None:
                    out["first_offset"] = i
                    out.update({k: h[k] for k in ("sample_rate", "channels", "kbps",
                                                  "version", "layer")})
                out["frames"] += 1
                out["bytes"] += h["length"]
                out["seconds"] += h["samples"] / h["sample_rate"]
                prev_end = i = end
                continue
        i += 1
    out["seconds"] = round(out["seconds"], 3)
    return out


class SseStream:
    """Incremental reader for the OpenAI-style event stream the shim emits and Live Vision relays.

    Events end at a blank line; only `event:` and `data:` matter here. Live Vision reports a failure
    that happens mid-stream as an `event: error`, because by then it has already sent its 200.
    """

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._buf = ""
        self.text = ""
        self.events = 0
        self.done = False
        self.error: str | None = None
        self.finish_reason: str | None = None
        self.metrics: dict | None = None

    def feed(self, data: bytes | str) -> None:
        # A chunk can end mid-character or between \r and \n; both are handled by buffering.
        self._buf += self._decoder.decode(data) if isinstance(data, bytes) else data
        self._buf = self._buf.replace("\r\n", "\n")
        while "\n\n" in self._buf:
            block, self._buf = self._buf.split("\n\n", 1)
            self._event(block)

    def close(self) -> None:
        self._buf += self._decoder.decode(b"", final=True)
        if self._buf.strip():
            self._event(self._buf)
        self._buf = ""

    def _event(self, block: str) -> None:
        name, data = "message", []
        for line in block.split("\n"):
            if not line or line.startswith(":"):
                continue
            field, _, value = line.partition(":")
            value = value[1:] if value.startswith(" ") else value
            if field == "event":
                name = value
            elif field == "data":
                data.append(value)
        if not data:
            return
        self.events += 1
        payload = "\n".join(data)
        if payload == "[DONE]":
            self.done = True
            return
        try:
            obj = json.loads(payload)
        except ValueError:
            if name == "error":
                self.error = payload
            return
        if not isinstance(obj, dict):
            return
        if name == "error" or obj.get("error"):
            err = obj.get("error")
            self.error = str(err.get("message") if isinstance(err, dict) else err or payload)
        for choice in obj.get("choices") or []:
            delta = choice.get("delta") or {}
            if isinstance(delta.get("content"), str):
                self.text += delta["content"]
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]
        if isinstance(obj.get("cosmos_metrics"), dict):
            self.metrics = obj["cosmos_metrics"]


def config_engines(text: str) -> dict[str, dict[str, str]]:
    """The `engines:` block of porch-feed's config.yaml, read without a YAML library.

    Deliberately narrow - that block only, plain and folded (>-) scalars only. The live config also
    holds the control_token, and nothing here should be able to pick that up and print it.
    """
    engines: dict[str, dict[str, str]] = {}
    inside, current, folding = False, None, None
    for line in text.splitlines():
        stripped = line.strip()
        if not inside:
            inside = line.rstrip() == "engines:"
            continue
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent == 0:
            break                                             # the next top-level key
        if folding and indent > 4:
            current[folding] = (current[folding] + " " + stripped).strip()
            continue
        folding = None
        m = re.fullmatch(r"([\w.-]+):\s*(.*)", stripped)
        if not m:
            continue
        key, value = m.group(1), m.group(2).strip()
        if indent == 2 and not value:
            current = engines.setdefault(key, {})
        elif indent == 4 and current is not None:
            if value in (">", ">-", "|", "|-"):
                current[key], folding = "", key
            else:
                current[key] = value.strip("\"'")
    return engines


def default_engine(engines: dict[str, dict[str, str]]) -> tuple[str | None, str]:
    """(engine id, how it was picked).

    The config has no `default:` key. It marks the default in prose, in the notes of the build it
    recommends ("...so this is the default choice"), and lists that build first; so the notes win
    and the order is the fallback.
    """
    for eid, e in engines.items():
        if re.search(r"\bdefault choice\b", e.get("notes", ""), re.I):
            return eid, "its notes call it the default choice"
    if engines:
        return next(iter(engines)), "no engine is marked; the first listed"
    return None, "the config lists no engines"


def config_scalar(text: str, key: str) -> str | None:
    """A top-level `key: value` from the config, or None."""
    m = re.search(rf"(?m)^{re.escape(key)}:\s*(.+?)\s*(?:#.*)?$", text)
    return m.group(1).strip("\"'") if m else None


def bridge_verdict(h: dict) -> tuple[str, str]:
    """PASS/WARN/FAIL for one bridge /healthz document."""
    state, reason = h.get("state"), h.get("reason")
    audio, push = h.get("audio") or {}, h.get("push") or {}
    if h.get("live"):
        bits = [f"live, {h.get('frames')} frames, newest {h.get('last_frame_age_s')} s old",
                f"session {h.get('sessions')}"]
        if push.get("enabled"):
            bits.append(f"push: {push.get('state')}")
        if not audio.get("live"):
            return WARN, "; ".join(bits + [f"video live but NO microphone audio "
                                           f"(last {audio.get('last_frame_age_s')} s ago)"])
        return PASS, "; ".join(bits + ["audio live"])
    if state == "dormant":
        # The bridge does this on purpose when the camera is knowably unavailable. Not a fault
        # here - but nothing downstream can be checked either.
        held = f" (held by {h['blocked_by']!r})" if h.get("blocked_by") else ""
        return WARN, f"dormant on purpose{held}: {reason}"
    if state == "live":
        return FAIL, f"state live but no fresh frame (newest {h.get('last_frame_age_s')} s old)"
    if state in ("starting", "connecting") and not h.get("failed_streak"):
        # A first session takes up to 20 s (ICE, DTLS, the first keyframe); right after a boot this
        # is the expected answer, not a fault. A retry after a failure says reconnecting instead.
        return WARN, (f"{state} (session {h.get('sessions')}, no failures yet): "
                      f"run again in a minute")
    extra = f"; sessions {h.get('sessions')}, failed_streak {h.get('failed_streak')}"
    return FAIL, f"{state}: {reason or 'no reason given'}{extra}"


def boot_verdict(state: str) -> tuple[str, str]:
    """For units that must come back at boot."""
    if state == "enabled":
        return PASS, "enabled"
    if state == "enabled-runtime":
        return FAIL, "enabled-runtime: enabled only until the next reboot"
    if state in ("disabled", "masked"):
        return FAIL, f"{state}: will not start at boot"
    if state == "not-found":
        return FAIL, "not installed"
    return WARN, f"{state}: starts only if something else pulls it in"


def intent_verdict(state: str, desired: str | None) -> tuple[str, str]:
    """For the two Live UIs, whose boot state should follow the command centre's recorded intent."""
    if state == "not-found":
        return FAIL, "not installed"
    enabled = state == "enabled"
    if desired is None:
        return PASS, f"{state}; no intent recorded by the command centre, so either is fine"
    if desired == "running":
        if enabled:
            return PASS, "enabled, follows intent (last started from the command centre)"
        return WARN, (f"{state}, but last started from the command centre: "
                      f"it will stay off after a reboot")
    if not enabled:
        return PASS, f"{state}, follows intent (last stopped from the command centre)"
    return WARN, "enabled, but last stopped from the command centre: a reboot starts it anyway"


def restart_verdict(policy: str) -> tuple[str, str]:
    if policy == "unless-stopped":
        return PASS, "restart=unless-stopped"
    if policy == "always":
        return WARN, "restart=always: also restarts a container someone stopped on purpose"
    return FAIL, f"restart={policy or 'no'}: will not come back after a reboot"


def ui_down_verdict(desired: str | None, active: str | None) -> tuple[str, str]:
    """For a Live UI that does not answer at all.

    Off is fine when a human turned it off, and the command centre's recorded intent says whether
    one did. Last started there and now not running is what a clone boot that lost the unit's
    enablement looks like, and that must not hide behind "off on purpose". `active` is the unit's
    ActiveState, None when systemd was not asked; `failed` is a crash, whatever the intent.
    """
    if desired == "stopped":
        return SKIP, "off, as last set from the command centre"
    if active == "failed":
        return FAIL, "the unit failed: it crashed rather than being turned off"
    if active in ("active", "activating", "reloading"):
        return WARN, f"the unit is {active} but not answering yet: run again in a minute"
    if desired == "running":
        return FAIL, ("last started from the command centre" + (f", but {active}" if active else "")
                      + ": did it come back after the boot?")
    return SKIP, "deliberately off at times, and the command centre has recorded no intent"


def shim_down_verdict(unit: dict[str, str], now: float) -> tuple[str, str]:
    """For a shim whose port refuses connections: still loading its engine, or down?

    `unit` is unit_activity()'s answer, empty when systemd was not asked. `now` is
    time.monotonic(), which on Linux is the CLOCK_MONOTONIC that systemd stamps
    ActiveEnterTimestampMonotonic (microseconds) with.
    """
    active, sub = unit.get("ActiveState"), unit.get("SubState")
    if not active:
        return FAIL, f"is {SHIM_UNIT} running?"
    if unit.get("LoadState") == "not-found":
        return FAIL, f"{SHIM_UNIT} is not installed"
    restarts = unit.get("NRestarts", "0")
    again = f"; systemd has restarted it {restarts} times" if restarts not in ("", "0") else ""
    if active == "active":
        since = unit.get("ActiveEnterTimestampMonotonic", "")
        age = now - int(since) / 1e6 if since.isdigit() and int(since) else None
        if age is not None and age > SHIM_LOAD_GRACE_S:
            return FAIL, (f"{SHIM_UNIT} active for {age / 60:.0f} min and still not listening: "
                          f"stuck loading the engine?{again}")
        took = f" for {age:.0f} s" if age is not None else ""
        return WARN, (f"{SHIM_UNIT} running{took} but not listening yet: loading the engine "
                      f"(~40-70 s after a start), run again in a minute{again}")
    if sub == "auto-restart":
        # Exited, and waiting out RestartSec before the next start: between crashes, not loading.
        return FAIL, f"{SHIM_UNIT} exited and is waiting to be restarted{again}: crash-looping?"
    if active in ("activating", "reloading"):
        return WARN, f"{SHIM_UNIT} {active}: run again in a minute"
    return FAIL, f"{SHIM_UNIT} {active}" + (f" ({sub})" if sub else "") + ": not running"


# ----------------------------------------------------------------------------------- HTTP
class Unreachable(Exception):
    """Nothing answered - refused, timed out, no route. Distinct from an HTTP error status."""


def _why(e: BaseException) -> str:
    if isinstance(e, ConnectionRefusedError):
        return "connection refused"
    if isinstance(e, (TimeoutError, socket.timeout)):
        return "timed out"
    return f"{type(e).__name__}: {e}".rstrip(": ")


def _connection(url: str, timeout: float) -> tuple[http.client.HTTPConnection, str]:
    u = urlsplit(url)
    if u.scheme == "https":
        # Loopback services with self-signed certificates (the WebUI's own). Verifying would only
        # prove this script does not carry their CA.
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        conn = http.client.HTTPSConnection(u.hostname, u.port or 443, timeout=timeout, context=ctx)
    else:
        conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    return conn, (u.path or "/") + (f"?{u.query}" if u.query else "")


def stream(url: str, seconds: float, *, method: str = "GET", body: bytes | None = None,
           headers: dict | None = None, timeout: float = 4.0, on_chunk=None,
           max_bytes: int = 32 * 2**20) -> tuple[int, dict, bytes]:
    """(status, headers, body), reading the body for at most `seconds` after the headers arrive.

    For endless responses (MJPEG, MP3, SSE) as much as for ordinary ones. `timeout` bounds the
    connect and the wait for headers. `on_chunk` sees each chunk as it lands and can end the read
    early by returning True.
    """
    conn, path = _connection(url, timeout)
    try:
        try:
            conn.connect()
            # Kept, because http.client drops conn.sock once it hands a Connection: close response
            # to the caller; the read deadline below still needs it.
            sock = conn.sock
            conn.request(method, path, body=body, headers=headers or {})
            resp = conn.getresponse()
        except (OSError, http.client.HTTPException) as e:
            raise Unreachable(_why(e)) from e
        hdrs = {k.lower(): v for k, v in resp.getheaders()}
        chunks, size = [], 0
        deadline = time.monotonic() + seconds
        # isclosed(): from Python 3.12 read1 closes the response, and with it the socket, as soon
        # as the last byte of a Content-Length body is read; the next settimeout would be EBADF.
        while size < max_bytes and not resp.isclosed():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                sock.settimeout(remaining)
                chunk = resp.read1(65536)
            except (TimeoutError, socket.timeout):
                break
            except (OSError, http.client.HTTPException):
                break                                          # the server hung up mid-stream
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if on_chunk is not None and resp.status == 200 and on_chunk(chunk):
                break
        return resp.status, hdrs, b"".join(chunks)
    finally:
        conn.close()


def get(url: str, timeout: float = 4.0) -> tuple[int, dict, bytes]:
    return stream(url, timeout, timeout=timeout)


def get_json(url: str, timeout: float = 4.0):
    """Parsed JSON from a 200, else HttpStatus; Unreachable if nothing answers."""
    status, _, body = get(url, timeout)
    if status != 200:
        raise HttpStatus(status, body)
    return json.loads(body)


class HttpStatus(Exception):
    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self.message = error_message(body)
        super().__init__(f"HTTP {status}" + (f": {self.message}" if self.message else ""))


def error_message(body: bytes) -> str:
    """The human part of an error body: {"error": {"message": ...}}, {"detail": ...}, or text."""
    text = body.decode("utf-8", "replace").strip()
    try:
        obj = json.loads(text)
        err = obj.get("error") if isinstance(obj, dict) else None
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
        if isinstance(obj, dict) and obj.get("detail"):
            return str(obj["detail"])
    except ValueError:
        pass
    return " ".join(text.split())[:160]


# ----------------------------------------------------------------------------------- audio
def mp3_level_dbfs(data: bytes) -> float | None:
    """RMS level of MP3 bytes in dBFS (-inf for digital silence); None if it cannot be decoded.

    PyAV only. The samples go through array('h') rather than numpy so this adds no dependency
    beyond av itself.
    """
    if av is None:
        return None
    total, count = 0, 0
    try:
        with av.open(io.BytesIO(data), format="mp3") as container:
            resampler = av.AudioResampler(format="s16")   # packed int16, same layout and rate
            for frame in container.decode(audio=0):
                for f in resampler.resample(frame):
                    n = f.samples * len(f.layout.channels)
                    pcm = array.array("h")
                    pcm.frombytes(bytes(f.planes[0])[:n * 2])
                    total += sum(s * s for s in pcm)
                    count += n
    except Exception:
        pass                              # a truncated last frame is fine once samples exist
    if not count:
        return None
    rms = math.sqrt(total / count) / 32768
    return 20 * math.log10(rms) if rms > 0 else float("-inf")


# ----------------------------------------------------------------------------------- system
def unit_enabled(unit: str) -> str:
    """`systemctl is-enabled`, as one word. Read from stdout: it exits non-zero unless enabled."""
    try:
        r = subprocess.run(["systemctl", "is-enabled", unit], capture_output=True, text=True,
                           timeout=10)
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"unknown ({type(e).__name__})"
    out = r.stdout.strip()
    if out:
        return out.splitlines()[-1].strip()
    return "not-found" if "No such file" in r.stderr or "not found" in r.stderr else "unknown"


def unit_activity(unit: str) -> dict[str, str]:
    """What the unit is doing now, from `systemctl show`; empty if systemd cannot be asked.

    `show` answers even for a unit that does not exist, with LoadState=not-found.
    """
    try:
        r = subprocess.run(["systemctl", "show", unit,
                            "--property=LoadState,ActiveState,SubState,NRestarts,"
                            "ActiveEnterTimestampMonotonic"],
                           capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if r.returncode != 0:
        return {}
    return dict(line.split("=", 1) for line in r.stdout.splitlines() if "=" in line)


def ha_reachy_cameras(config_dir: Path) -> list[str] | None:
    """Enabled camera entities of Home Assistant's Reachy Mini integration, from HA's registries.

    None when there are no registries to read. Only ids, domains and disabled_by are looked at:
    the config entries also hold every integration's credentials, and none of that belongs in
    this script's output.
    """
    storage = config_dir / ".storage"
    try:
        entries = json.loads((storage / "core.config_entries").read_text())["data"]["entries"]
        entities = json.loads((storage / "core.entity_registry").read_text())["data"]["entities"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    # An entity of a disabled config entry is not loaded, whatever its own disabled_by says.
    loaded = {e.get("entry_id") for e in entries if isinstance(e, dict)
              and e.get("domain") == "reachy_mini" and not e.get("disabled_by")}
    return sorted(e["entity_id"] for e in entities if isinstance(e, dict)
                  and e.get("platform") == "reachy_mini" and e.get("config_entry_id") in loaded
                  and str(e.get("entity_id", "")).startswith("camera.")
                  and not e.get("disabled_by"))


def container_state(name: str) -> tuple[str | None, str]:
    """(restart policy, run state) from docker inspect, or (None, why it could not be read)."""
    try:
        r = subprocess.run(["docker", "inspect", "--format",
                            "{{.HostConfig.RestartPolicy.Name}} {{.State.Status}}", name],
                           capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, f"docker inspect failed ({type(e).__name__})"
    if r.returncode == 0:
        policy, _, status = r.stdout.strip().partition(" ")
        return policy, status
    return None, " ".join(r.stderr.split())[:160] or f"docker inspect exited {r.returncode}"


# ----------------------------------------------------------------------------------- checks
class Smoke:
    def __init__(self, *, robot_host: str = ROBOT_HOST, bridge_hosts=BRIDGE_HOSTS,
                 bridge_port: int = BRIDGE_PORT, webui_url: str = WEBUI_URL,
                 live_vision_url: str = LIVE_VISION_URL, shim_url: str = SHIM_URL,
                 frigate_url: str = FRIGATE_URL, engine_link: str = ENGINE_LINK,
                 feed_config: Path | None = None, ha_config: Path = HA_CONFIG,
                 sample_s: float = SAMPLE_S, image: bytes | None = None,
                 inference: bool = True, robot: bool = True, system: bool = True) -> None:
        self.robot_host = robot_host
        self.bridge_hosts = tuple(bridge_hosts)
        self.bridge_port = bridge_port
        self.webui_url = webui_url.rstrip("/")
        self.live_vision_url = live_vision_url.rstrip("/")
        self.shim_url = shim_url.rstrip("/")
        self.frigate_url = frigate_url.rstrip("/")
        self.engine_link = engine_link
        if feed_config is None:
            feed_config = FEED_CONFIG if FEED_CONFIG.is_file() else REPO_FEED_CONFIG
        self.feed_config = Path(feed_config)
        self.ha_config = Path(ha_config)
        self.sample_s = sample_s
        self.image = image
        self.inference = inference
        self.robot = robot
        self.system = system
        self.results: list[Result] = []
        self.health: dict | None = None          # the first bridge /healthz that answered
        self.bridge_url: str | None = None       # ...and where
        self.still: bytes | None = None
        self.intent: dict = {}
        self.shim_problem: str | None = None     # set by check_shim when the model cannot answer

    def add(self, check: str, status: str, detail: str) -> None:
        self.results.append(Result(check, status, detail))

    def run(self) -> list[Result]:
        self.intent = self._load_intent()
        # The shim before the inference: a shim still loading its engine is one WARN, not a WARN
        # plus an inference FAIL for the same cause.
        steps = [self.check_bridge_health, self.check_still, self.check_mjpeg, self.check_audio,
                 self.check_webui_push, self.check_live_vision_proxy, self.check_shim,
                 self.check_engine, self.check_inference, self.check_frigate]
        if self.robot:
            steps.insert(0, self.check_robot)
        if self.system:
            steps += [self.check_boot, self.check_containers, self.check_home_assistant]
        for step in steps:
            try:
                step()
            except Exception as e:   # a bug in one check must not hide what the others found
                self.add(step.__name__.replace("check_", ""), FAIL,
                         f"the check itself crashed: {type(e).__name__}: {e}")
        return self.results

    # ------------------------------------------------------------------ helpers
    def _load_intent(self) -> dict:
        """porch-feed's record of what a human last did to each service, keyed like its config."""
        try:
            text = self.feed_config.read_text()
        except OSError:
            return {}
        path = config_scalar(text, "service_intent_path")
        if not path:
            db = config_scalar(text, "db_path")
            path = str(Path(db).parent / "service_intent.json") if db else None
        try:
            data = json.loads(Path(path).read_text()) if path else {}
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _intent_note(self, key: str) -> str:
        i = self.intent.get(key) or {}
        if not i.get("desired"):
            return ""
        when = (datetime.fromtimestamp(i["at"]).strftime("%b %d %H:%M")
                if isinstance(i.get("at"), (int, float)) else "?")
        return f"; command centre intent: {i['desired']} ({i.get('by') or 'user'}, {when})"

    def _ui_down(self, key: str) -> tuple[str, str]:
        """ui_down_verdict for the Live UI under `key`, with its unit asked only on a system run."""
        desired = (self.intent.get(key) or {}).get("desired")
        active = unit_activity(UI_UNITS[key]).get("ActiveState") if self.system else None
        return ui_down_verdict(desired if desired in ("running", "stopped") else None, active)

    @property
    def bridge_live(self) -> bool:
        return bool(self.health and self.health.get("live"))

    def _bridge_down_reason(self) -> str:
        if self.health is None:
            return "the bridge is not answering"
        return f"the bridge is not live ({self.health.get('state')}: {self.health.get('reason')})"

    @staticmethod
    def _gone(e: Unreachable) -> str:
        # /healthz answered moments ago, so this is the bridge going away mid-run (Restart=always,
        # a MemoryMax kill, someone restarting it), not a bridge that was never up.
        return f"the bridge stopped answering mid-check ({e}); run again once /healthz answers"

    # ------------------------------------------------------------------ robot
    def check_robot(self) -> None:
        """Only GETs on port 8000. The same three the bridge asks before it opens a session."""
        base = f"http://{self.robot_host}:{ROBOT_REST_PORT}"
        try:
            status = get_json(base + "/api/daemon/status")
        except (Unreachable, HttpStatus, ValueError) as e:
            self.add("robot daemon", FAIL, f"{self.robot_host}:{ROBOT_REST_PORT} {e} - robot off, "
                                           f"booting, or off the network")
            for name in ("robot media", "robot app lock"):
                self.add(name, SKIP, "robot daemon not answering")
            return
        state = status.get("state") if isinstance(status, dict) else None
        version = status.get("version") if isinstance(status, dict) else None
        detail = f"state {state!r}" + (f", version {version}" if version else "")
        if state == "running":
            self.add("robot daemon", PASS, detail)
        else:
            self.add("robot daemon", FAIL, f"{detail}: the bridge stays dormant until running")

        try:
            media = get_json(base + "/api/media/status")
            if media.get("no_media"):
                self.add("robot media", WARN, "daemon runs without media: no camera for anyone")
            elif media.get("released"):
                self.add("robot media", WARN, "daemon has released the camera and microphone "
                                              "(an SDK client holds them); bridge dormant until "
                                              "they are back")
            elif media.get("available") is False:
                self.add("robot media", WARN, "daemon reports the camera unavailable")
            else:
                self.add("robot media", PASS, "camera and microphone with the daemon")
        except HttpStatus as e:
            self.add("robot media", SKIP if e.status == 404 else FAIL,
                     "no /api/media/status on this daemon" if e.status == 404 else str(e))
        except (Unreachable, ValueError, AttributeError) as e:
            self.add("robot media", FAIL, f"/api/media/status: {e}")

        try:
            lock = get_json(base + "/api/daemon/robot-app-lock-status")
            if lock.get("state") == "local_app":
                self.add("robot app lock", WARN, f"camera held by the robot app "
                                                 f"{lock.get('holder_name') or '?'!r}; "
                                                 f"bridge dormant until it exits")
            else:
                self.add("robot app lock", PASS, f"no on-robot app holds the camera "
                                                 f"(state {lock.get('state')!r})")
        except HttpStatus as e:
            self.add("robot app lock", SKIP if e.status == 404 else FAIL,
                     "no app-lock endpoint on this daemon" if e.status == 404 else str(e))
        except (Unreachable, ValueError, AttributeError) as e:
            self.add("robot app lock", FAIL, f"/api/daemon/robot-app-lock-status: {e}")

    # ------------------------------------------------------------------ bridge
    def check_bridge_health(self) -> None:
        down = []
        for host in self.bridge_hosts:
            base = f"http://{host}:{self.bridge_port}"
            name = f"bridge {host}"
            try:
                h = get_json(base + "/healthz", timeout=3)
            except Unreachable as e:
                # Only the gateway down means docker (and so the address) is missing: the bridge
                # waits for 172.17.0.1 to appear rather than failing. Both down: the bridge itself.
                hint = ("is reachy-mjpeg-bridge.service running?" if not self.health and not down
                        else "same as above" if down
                        else "docker not up? Frigate cannot reach the bridge without this address")
                self.add(name, FAIL, f":{self.bridge_port}/healthz {e} - {hint}")
                down.append(host)
                continue
            except (HttpStatus, ValueError) as e:
                self.add(name, FAIL, f"/healthz: {e}")
                continue
            if self.health is None:
                self.health, self.bridge_url = h, base
            self.add(name, *bridge_verdict(h))

    def check_still(self) -> None:
        if not (self.health and self.health.get("has_frame")):
            self.add("bridge /still.jpg", SKIP,
                     f"no fresh frame to serve: {self._bridge_down_reason()}")
            return
        try:
            status, hdrs, body = get(self.bridge_url + "/still.jpg", timeout=5)
        except Unreachable as e:
            self.add("bridge /still.jpg", FAIL, self._gone(e))
            return
        if status != 200:
            self.add("bridge /still.jpg", FAIL, f"HTTP {status}: {error_message(body)}")
            return
        dims = jpeg_dimensions(body)
        if dims is None:
            self.add("bridge /still.jpg", FAIL, f"{len(body)} bytes of {hdrs.get('content-type')}, "
                                                f"but not a JPEG (no SOI/SOF)")
            return
        self.still = body
        ctype = hdrs.get("content-type", "")
        self.add("bridge /still.jpg", PASS if ctype.startswith("image/jpeg") else WARN,
                 f"{dims[0]}x{dims[1]} JPEG, {len(body) // 1024} KB"
                 + ("" if ctype.startswith("image/jpeg") else f", but served as {ctype!r}"))

    def check_mjpeg(self) -> None:
        if not self.bridge_live:
            self.add("bridge /mjpeg", SKIP, self._bridge_down_reason())
            return
        try:
            status, hdrs, data = stream(self.bridge_url + "/mjpeg", self.sample_s)
        except Unreachable as e:
            self.add("bridge /mjpeg", FAIL, self._gone(e))
            return
        if status != 200:
            self.add("bridge /mjpeg", FAIL, f"HTTP {status}: {error_message(data)}")
            return
        m = re.search(r'boundary="?([^";]+)', hdrs.get("content-type", ""))
        parts, _ = split_mjpeg(data, (m.group(1) if m else "frame").encode())
        dims = [d for d in map(jpeg_dimensions, parts) if d]
        detail = (f"{len(dims)} frames in {self.sample_s:g} s "
                  f"({len(dims) / self.sample_s:.1f} fps)" + (f", {dims[-1][0]}x{dims[-1][1]}"
                                                                if dims else ""))
        if len(parts) > len(dims):
            detail += f"; {len(parts) - len(dims)} parts were not JPEGs"
        if len(dims) >= MIN_MJPEG_FRAMES and len(parts) == len(dims):
            self.add("bridge /mjpeg", PASS, detail)
        else:
            self.add("bridge /mjpeg", FAIL if not dims else WARN,
                     detail + f" (want >= {MIN_MJPEG_FRAMES})")

    def check_audio(self) -> None:
        audio = (self.health or {}).get("audio") or {}
        if not audio.get("live"):
            self.add("bridge /audio.mp3", SKIP, "the bridge has no microphone audio: "
                     + (self._bridge_down_reason() if not self.bridge_live
                        else f"last audio {audio.get('last_frame_age_s')} s ago"))
            return
        try:
            status, _, data = stream(self.bridge_url + "/audio.mp3", self.sample_s)
        except Unreachable as e:
            self.add("bridge /audio.mp3", FAIL, self._gone(e))
            return
        if status != 200:
            self.add("bridge /audio.mp3", FAIL, f"HTTP {status}: {error_message(data)}")
            return
        scan = scan_mp3(data)
        detail = f"{len(data) // 1024} KB, {scan['frames']} MP3 frames in {self.sample_s:g} s"
        if len(data) < MIN_AUDIO_BYTES or not scan["frames"]:
            self.add("bridge /audio.mp3", FAIL, detail + f" (want >= {MIN_AUDIO_BYTES // 1024} KB "
                                                         f"of valid frames)")
            return
        detail += (f" = {scan['seconds']:.1f} s of {scan['sample_rate']} Hz "
                   f"{'stereo' if scan['channels'] == 2 else 'mono'} {scan['kbps']} kb/s")
        if av is None:
            self.add("bridge /audio.mp3", PASS, detail + "; level not measured (no PyAV here: "
                                                         "use /home/orin/reachy_env/bin/python)")
            return
        level = mp3_level_dbfs(data)
        if level is None:
            self.add("bridge /audio.mp3", WARN, detail + "; PyAV could not decode it")
        elif level < SILENT_DBFS:
            self.add("bridge /audio.mp3", WARN, detail + f"; SILENT ({level:.0f} dBFS RMS): "
                                                         f"microphone muted or dead?")
        else:
            self.add("bridge /audio.mp3", PASS, detail + f"; level {level:.0f} dBFS RMS")

    # ------------------------------------------------------------------ Live UIs
    def check_webui_push(self) -> None:
        name = f"webui push ({WEBUI_SESSION})"
        url = self.webui_url + "/api/push/status"
        try:
            before = get_json(url)
        except Unreachable as e:
            verdict, why = self._ui_down("vlm")
            self.add(name, verdict, f"Live VLM WebUI not answering ({e}): {why}"
                                    f"{self._intent_note('vlm')}")
            return
        except (HttpStatus, ValueError) as e:
            self.add(name, FAIL, f"/api/push/status: {e}")
            return
        if not self.bridge_live:
            self.add(name, SKIP, f"the WebUI is up, but {self._bridge_down_reason()}")
            return
        time.sleep(self.sample_s)
        try:
            after = get_json(url)
        except (Unreachable, HttpStatus, ValueError) as e:
            self.add(name, FAIL, f"/api/push/status stopped answering mid-check: {e}")
            return
        s0, s1 = self._push_session(before), self._push_session(after)
        push = self.health.get("push") or {}
        bridge_side = (f"bridge push: {push.get('state')}"
                       + (f" ({push['detail']})" if push.get("detail") else "")
                       if push.get("enabled") else "the bridge runs without --push-url")
        # A 409 means someone pressed Stop on the session in the WebUI. That is a decision, and the
        # bridge deliberately waits for Start rather than re-creating the session behind their back.
        stopped_there = "stopped in the WebUI" in str(push.get("state"))
        if not s1 or not s1.get("connected"):
            self.add(name, WARN if stopped_there else FAIL,
                     f"no connected {WEBUI_SESSION!r} session; {bridge_side}")
            return
        n0 = (s0 or {}).get("frames_received") or 0
        n1 = s1.get("frames_received") or 0
        if n1 > n0:
            self.add(name, PASS, f"connected, +{n1 - n0} frames in {self.sample_s:g} s "
                                 f"({n1} received)")
        else:
            self.add(name, WARN if stopped_there else FAIL,
                     f"connected, but frames_received stuck at {n1}; {bridge_side}")

    @staticmethod
    def _push_session(status) -> dict | None:
        for s in (status or {}).get("streams") or []:
            if isinstance(s, dict) and s.get("session_id") == WEBUI_SESSION:
                return s
        return None

    def _relay_token(self) -> str | None:
        """Live Vision's per-process /reachy/ token, read the way its own page reads it.

        None when there is none to read: a build that predates the token, or one not answering,
        which the /reachy/healthz request that follows reports on its own.
        """
        try:
            status, _, body = get(self.live_vision_url + "/api/access", timeout=5)
            token = json.loads(body).get("reachy_token") if status == 200 else None
        except (Unreachable, ValueError, AttributeError):
            return None
        return token if isinstance(token, str) and token else None

    def check_live_vision_proxy(self) -> None:
        name = "live vision /reachy"
        token = self._relay_token()
        query = f"?token={quote(token, safe='')}" if token else ""
        try:
            status, _, body = get(self.live_vision_url + "/reachy/healthz" + query, timeout=5)
        except Unreachable as e:
            verdict, why = self._ui_down("livevision")
            self.add(name, verdict, f"Live Vision not answering ({e}): {why}"
                                    f"{self._intent_note('livevision')}")
            return
        if status == 404:
            self.add(name, FAIL, "Live Vision answers, but has no /reachy/ proxy: "
                                 "a build without the Reachy source")
            return
        if status != 200:
            if self.health is None:
                self.add(name, SKIP, f"HTTP {status}, and the bridge itself is down, "
                                     f"so there is nothing to relay")
            else:
                self.add(name, FAIL, f"HTTP {status}: {error_message(body)}")
            return
        try:
            h = json.loads(body)
            state = h["state"]
        except (ValueError, KeyError, TypeError):
            self.add(name, FAIL, "answered 200, but not with the bridge's /healthz")
            return
        self.add(name, PASS, f"relays the bridge: state {state}, {h.get('frames')} frames")

    def check_inference(self) -> None:
        """ONE request, shaped exactly as serve_ui.validate_request insists.

        Streaming only, one user message holding one text part and one embedded JPEG, temperature
        0 or 0.7, no seed, no extra keys, and no Origin header (the UI refuses cross-origin
        requests; a request with no Origin is what a same-host tool looks like).
        """
        name = "live vision inference"
        if not self.inference:
            self.add(name, SKIP, "--no-inference")
            return
        if self.shim_problem:
            # Live Vision would only relay the shim's absence as a 503; the shim row has the cause.
            self.add(name, SKIP, f"not sent: the shim {self.shim_problem}")
            return
        try:
            status, _, body = get(self.live_vision_url + "/v1/models", timeout=6)
        except Unreachable as e:
            self.add(name, SKIP, f"Live Vision not answering ({e}); "
                                 f"its state is under 'live vision /reachy'")
            return
        if status != 200:
            self.add(name, FAIL, f"/v1/models through Live Vision: HTTP {status}: "
                                 f"{error_message(body)}")
            return
        try:
            model = json.loads(body)["data"][0]["id"]
        except (ValueError, KeyError, IndexError, TypeError):
            self.add(name, FAIL, "/v1/models through Live Vision lists no model")
            return
        image = self.image or self.still
        if image is None:
            self.add(name, SKIP, f"no image to send: {self._bridge_down_reason()} "
                                 f"(--image FILE tests this path without the robot)")
            return
        payload = {
            "model": model, "stream": True, "temperature": 0, "max_tokens": 16,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Describe what the camera sees in one short sentence."},
                {"type": "image_url",
                 "image_url": {"url": "data:image/jpeg;base64," + b64encode(image).decode()}},
            ]}],
        }
        data = json.dumps(payload).encode()
        if len(data) > LIVE_VISION_MAX_BODY:
            self.add(name, FAIL, f"image too large: the request would be {len(data) // 1024} KB, "
                                 f"Live Vision takes {LIVE_VISION_MAX_BODY // 1024} KB")
            return
        sse = SseStream()
        t0 = time.monotonic()
        first: list[float] = []

        def on_chunk(chunk: bytes) -> bool:
            sse.feed(chunk)
            if sse.text and not first:
                first.append(time.monotonic() - t0)
            return sse.done or sse.error is not None

        try:
            status, _, rest = stream(self.live_vision_url + "/v1/chat/completions",
                                     INFERENCE_TIMEOUT_S, method="POST", body=data,
                                     headers={"Content-Type": "application/json",
                                              "Accept": "text/event-stream"},
                                     timeout=30, on_chunk=on_chunk)
        except Unreachable as e:
            self.add(name, FAIL, f"no response to the inference request ({e})")
            return
        total = time.monotonic() - t0
        if status == 429:
            self.add(name, WARN, "Live Vision is busy with another inference (HTTP 429); "
                                 "not retried - this check sends exactly one")
            return
        if status != 200:
            self.add(name, FAIL, f"HTTP {status}: {error_message(rest)}")
            return
        sse.close()
        if sse.error:
            self.add(name, FAIL, f"stream error after {total:.1f} s: {sse.error}")
            return
        if not sse.text.strip():
            ended = "stream ended" if sse.done else f"no [DONE] within {INFERENCE_TIMEOUT_S:.0f} s"
            self.add(name, FAIL, f"no text: {sse.events} events, {ended}")
            return
        said = " ".join(sse.text.split())
        said = said if len(said) <= 60 else said[:57] + "..."
        native = (sse.metrics or {}).get("native_inference_ms")
        first_s = first[0] if first else total
        self.add(name, PASS, f"{said!r} - first text {first_s * 1000:.0f} ms, "
                             f"all {total * 1000:.0f} ms"
                             + (f" (native {native:.0f} ms)" if isinstance(native, (int, float))
                                else "")
                             + (", image from --image" if self.image else ", bridge still"))

    # ------------------------------------------------------------------ shim and engine
    def check_shim(self) -> None:
        try:
            status, _, body = get(self.shim_url + "/v1/models", timeout=5)
        except Unreachable as e:
            # Refused is also what a shim loading its engine looks like (SHIM_LOAD_GRACE_S), which
            # right after a boot is the expected answer; systemd tells loading from down.
            unit = unit_activity(SHIM_UNIT) if self.system else {}
            verdict, why = shim_down_verdict(unit, time.monotonic())
            self.shim_problem = f"is not listening ({e})"
            self.add("shim", verdict, f"{urlsplit(self.shim_url).netloc} {e} - {why}")
            return
        if status != 200:
            self.shim_problem = f"answers /v1/models with HTTP {status}"
            self.add("shim", FAIL, f"/v1/models: HTTP {status}: {error_message(body)}")
            return
        try:
            model = json.loads(body)["data"][0]["id"]
        except (ValueError, KeyError, IndexError, TypeError):
            model = "?"
        try:
            ready, _, _ = get(self.shim_url + "/health/ready", timeout=5)
        except Unreachable:
            ready = None
        if ready == 503:
            # Not seen with cosmos3_shim_v1, which loads before it listens (SHIM_LOAD_GRACE_S); a
            # shim that loaded in the background would answer the static /v1/models meanwhile.
            self.shim_problem = "is still loading the engine"
            self.add("shim", WARN, f"answers (model {model}), but /health/ready says it is still "
                                   f"loading the engine")
        else:
            self.add("shim", PASS, f"answers, model {model}"
                                   + (", ready" if ready == 200 else f", /health/ready {ready}"))

    def check_engine(self) -> None:
        name = "shim engine"
        link = self.engine_link
        if not os.path.lexists(link):
            self.add(name, SKIP, f"{link} does not exist here (not the Orin?)")
            return
        target = os.path.realpath(link)
        try:
            engines = config_engines(self.feed_config.read_text())
        except OSError as e:
            self.add(name, WARN, f"{link} -> {target}; cannot read {self.feed_config}: {e}")
            return
        want, how = default_engine(engines)
        if not os.path.isfile(os.path.join(target, "llm.engine")):
            self.add(name, FAIL, f"{link} -> {target}, which has no llm.engine")
            return
        active = next((k for k, e in engines.items()
                       if e.get("path") and os.path.realpath(e["path"]) == target), None)

        def label(k: str) -> str:
            return f"{engines[k].get('name', k)} ({k})"

        if want is None:
            self.add(name, WARN, f"-> {target}; {how} in {self.feed_config}")
        elif active == want:
            self.add(name, PASS, f"-> {label(active)}, the config's default")
        else:
            self.add(name, WARN, f"-> {label(active) if active else target}, but the config's "
                                 f"default is {label(want)} ({how}); switch it from the "
                                 f"command centre if that was not deliberate")

    # ------------------------------------------------------------------ Frigate
    def check_frigate(self) -> None:
        name = f"frigate {FRIGATE_CAMERA}"
        try:
            streams = get_json(self.frigate_url + "/api/go2rtc/streams", timeout=6)
        except Unreachable as e:
            self.add(name, FAIL, f"Frigate {urlsplit(self.frigate_url).netloc} {e} - "
                                 f"is the frigate container up?")
            return
        except (HttpStatus, ValueError) as e:
            self.add(name, FAIL, f"/api/go2rtc/streams: {e}")
            return
        if not isinstance(streams, dict) or FRIGATE_CAMERA not in streams:
            listed = ", ".join(sorted(streams)) if isinstance(streams, dict) else "?"
            self.add(name, FAIL, f"go2rtc has no {FRIGATE_CAMERA} stream (has: {listed})")
            return
        def bridge_sources(stream: str) -> list[str]:
            producers = (streams.get(stream) or {}).get("producers") or []
            return sorted({m.group(1) for p in producers if isinstance(p, dict)
                           for m in [re.search(rf":{self.bridge_port}(/[^#?\s\"]*)",
                                               str(p.get("url", "")))] if m})
        sources = bridge_sources(FRIGATE_CAMERA)
        detail = f"go2rtc stream listed, bridge sources: {', '.join(sources) or 'none'}"
        if FRIGATE_MIC_STREAM in streams:
            mic = bridge_sources(FRIGATE_MIC_STREAM)
            detail += (f"; mic as {FRIGATE_MIC_STREAM} ({', '.join(mic) or 'no bridge source'})"
                       if "/audio.mp3" in mic else f"; {FRIGATE_MIC_STREAM} has no /audio.mp3 source")
        elif "/audio.mp3" not in sources:
            detail += " (no audio source: Frigate gets no Reachy audio)"
        # Reported, never changed: camera on/off is a deliberate human choice on this box.
        # `enabled` is the running state; Frigate 0.18 persists toggles apart from the config file
        # and restores them over it at startup, so enabled_in_config can legitimately disagree.
        try:
            cam = (get_json(self.frigate_url + "/api/config", timeout=8).get("cameras") or {}) \
                .get(FRIGATE_CAMERA)
            if cam is None:
                enabled = "not configured as a camera"
            else:
                enabled = f"camera {'enabled' if cam.get('enabled') else 'disabled'}"
                if "enabled_in_config" in cam and cam["enabled_in_config"] != cam.get("enabled"):
                    enabled += (f" at runtime (config file says "
                                f"{'enabled' if cam['enabled_in_config'] else 'disabled'})")
        except (Unreachable, HttpStatus, ValueError, AttributeError):
            enabled = "camera state unknown"
        self.add(name, PASS, f"{detail}; {enabled} (reported, not changed)")

    # ------------------------------------------------------------------ boot
    def check_boot(self) -> None:
        if shutil.which("systemctl") is None:
            self.add("boot", SKIP, "no systemctl here")
            return
        for unit in BOOT_UNITS:
            self.add(f"boot {unit.removesuffix('.service')}", *boot_verdict(unit_enabled(unit)))
        for unit, key in INTENT_UNITS.items():
            desired = (self.intent.get(key) or {}).get("desired")
            self.add(f"boot {unit.removesuffix('.service')}",
                     *intent_verdict(unit_enabled(unit), desired))

    def check_containers(self) -> None:
        if shutil.which("docker") is None:
            self.add("restart", SKIP, "no docker CLI here")
            return
        for name in CONTAINERS:
            policy, status = container_state(name)
            if policy is None:
                if "permission denied" in status.lower():
                    self.add(f"restart {name}", SKIP, f"cannot read docker: {status}")
                else:
                    self.add(f"restart {name}", FAIL, status)
                continue
            verdict, detail = restart_verdict(policy)
            self.add(f"restart {name}", verdict, f"{detail}, {status}")

    def check_home_assistant(self) -> None:
        """Its restart policy and its Reachy camera, both reported and neither FAILed (HA_CONTAINER).

        The camera is the one thing on the Orin besides the bridge that opens WebRTC sessions to
        the robot, and it opens a fresh one for each image asked for after 10 s idle - the kind of
        churn that ran the daemon out of file descriptors (deploy/07 §1, §7).
        """
        running = None
        name = f"restart {HA_CONTAINER}"
        if shutil.which("docker") is None:
            self.add(name, SKIP, "no docker CLI here")
        else:
            policy, status = container_state(HA_CONTAINER)
            if policy is None:
                self.add(name, SKIP, f"cannot read it: {status}")
            else:
                running = status == "running"
                verdict, detail = restart_verdict(policy)
                if verdict == FAIL:
                    verdict, detail = WARN, (f"{detail} ({status}). nvr/README.md records "
                                             f"unless-stopped; leave it if HA is off on purpose")
                else:
                    detail = f"{detail}, {status}"
                self.add(name, verdict, detail)

        name = "ha reachy camera"
        cameras = ha_reachy_cameras(self.ha_config)
        if cameras is None:
            self.add(name, SKIP, f"no Home Assistant registries readable under {self.ha_config}")
        elif not cameras:
            self.add(name, PASS, "HA's Reachy Mini camera is disabled: HA opens no WebRTC "
                                 "sessions to the robot")
        elif running is False:
            self.add(name, SKIP, f"{', '.join(cameras)} enabled, but HA is not running, so it "
                                 f"opens no sessions now; deal with it before starting HA "
                                 f"(deploy/07 §7)")
        else:
            self.add(name, WARN, f"{', '.join(cameras)} enabled"
                                 + ("" if running else " (HA's state unknown)")
                                 + ": HA opens WebRTC sessions of its own to the robot, a new one "
                                   "per image after 10 s idle, adding to the daemon's socket leak. "
                                   "Disable it, or point HA at the bridge (deploy/07 §7)")


# ----------------------------------------------------------------------------------- output
def render(results: list[Result], host: str) -> str:
    width = max([len(r.check) for r in results] + [5])
    lines = [f"Reachy smoke check on {host}, {datetime.now():%Y-%m-%d %H:%M:%S}", "",
             f"{'CHECK':<{width}}  STATUS  DETAIL"]
    lines += [f"{r.check:<{width}}  {r.status:<6}  {r.detail}" for r in results]
    counts = {s: sum(r.status == s for r in results) for s in (PASS, WARN, FAIL, SKIP)}
    lines += ["", f"{len(results)} checks: " + ", ".join(f"{n} {s}" for s, n in counts.items())]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", action="store_true", help="print JSON instead of a table")
    p.add_argument("--no-inference", action="store_true",
                   help="skip the one inference through Live Vision")
    p.add_argument("--image", type=Path,
                   help="JPEG to send for the inference instead of the bridge's still")
    p.add_argument("--robot-host", default=ROBOT_HOST, help="robot address (REST GETs only)")
    p.add_argument("--feed-config", type=Path,
                   help=f"porch-feed config (default {FEED_CONFIG}, else the repo copy)")
    p.add_argument("--seconds", type=float, default=SAMPLE_S,
                   help="how long to sample each stream (default %(default)s)")
    args = p.parse_args(argv)

    image = None
    if args.image:
        try:
            image = args.image.read_bytes()
        except OSError as e:
            p.error(f"cannot read {args.image}: {e}")
        if jpeg_dimensions(image) is None:
            p.error(f"{args.image} is not a JPEG")

    smoke = Smoke(robot_host=args.robot_host, feed_config=args.feed_config,
                  sample_s=args.seconds, image=image, inference=not args.no_inference)
    results = smoke.run()
    host = socket.gethostname()
    failed = any(r.status == FAIL for r in results)
    if args.json:
        print(json.dumps({"host": host, "at": datetime.now().isoformat(timespec="seconds"),
                          "ok": not failed, "pyav": av is not None,
                          "checks": [asdict(r) for r in results]}, indent=2))
    else:
        print(render(results, host))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
