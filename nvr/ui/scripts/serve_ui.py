#!/usr/bin/env python3
"""Serve the original camera UI and proxy only the loopback Edge-LLM API.

No model, cloud fallback, downloaded dependency or synthetic inference lives here.
Run on the Jetson alongside tensorrt-edgellm-serve. See research/ui-feasibility.md.

It also relays the Reachy Mini bridge (nvr/reachy/reachy_mjpeg_bridge.py) under /reachy/.
The bridge listens on loopback and the Docker gateway only, and a same-origin relay is what
lets the page draw the robot's frames into a canvas for inference without tainting it.
Those routes need this process's token, which only the page itself can read (/api/access),
and a Host that DNS rebinding cannot fake: see RELAY_TOKEN and host_allowed().
"""

import argparse
import base64
from datetime import datetime, timezone
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import math
from pathlib import Path
import re
import secrets
import select
import socket
import ssl
import threading
import time
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1] / "web"
MAX_BODY = 2 * 1024 * 1024
MAX_RESPONSE = 1024 * 1024
MAX_SECONDS = 120
GENERATION_LOCK = threading.Lock()
GENERATION_WAIT_LOCK = threading.Lock()
HANDOFF_SECONDS = 0.75

# Reachy bridge routes: page path -> (bridge path, the only Content-Type accepted from it).
REACHY_FETCHES = {"/reachy/healthz": ("/healthz", "application/json"),
                  "/reachy/still.jpg": ("/still.jpg", "image/jpeg")}
REACHY_STREAMS = {"/reachy/mjpeg": ("/mjpeg", "multipart/x-mixed-replace"),
                  "/reachy/audio.mp3": ("/audio.mp3", "audio/mpeg")}
REACHY_SECONDS = 3
# Each open stream pins a handler thread (and a disconnect watcher) for as long as a tab
# watches or listens, in a process capped at MemoryMax=256M with a listen queue of 8. The cap
# is per process, so the HTTP and HTTPS listeners share it, and inference never waits on it.
MAX_REACHY_STREAMS = 4
REACHY_STREAM_SLOTS = threading.BoundedSemaphore(MAX_REACHY_STREAMS)
# Replacing an <img> src closes the old stream and opens the new one at once, but the old
# stream's watcher needs up to 0.2 s to notice and hand its slot back. A full cap waits this
# long for that handoff before it refuses, so a tab reopening its own stream is not turned away.
STREAM_HANDOFF_SECONDS = 0.5
# The bridge writes nothing while it has no new frame or no microphone audio. A stream silent
# this long gives its slot back; the page reopens it when the bridge reports live again.
STREAM_IDLE_SECONDS = 20
STREAM_CHUNK = 16384
# Open streams by the page's ?stream= token -> [monotonic time of the last byte relayed].
# An MJPEG <img> tells the page nothing when its stream ends cleanly (a restart of this
# server, say) and keeps showing the last frame, so the page asks here instead: each
# /reachy/healthz?stream=<token> answer carries X-Reachy-Stream: closed | idle_ms=<n>.
RELAYED_STREAMS = {}
RELAYED_STREAMS_LOCK = threading.Lock()
STREAM_TOKEN = re.compile(r"[A-Za-z0-9_-]{8,64}")
# Streams holding a slot, for X-Reachy-Slots on /reachy/healthz. The semaphore cannot be asked
# how many it has given out, so this is counted beside it, under RELAYED_STREAMS_LOCK: taken
# after the slot, given up before it, so it never shows more open than there really are.
OPEN_REACHY_STREAMS = 0
# The relay's key, new in every process. The robot's camera and microphone must not be open to
# every page a LAN browser visits: an <img> or <audio> on another site carries no Origin, and a
# DNS-rebound name looks same-origin to the browser. /api/access hands this to the page, which
# only a same-origin page can read, and every /reachy/ URL must carry it as ?token=. As with the
# command centre's control token, this stops drive-by pages, not a determined LAN attacker.
RELAY_TOKEN = secrets.token_urlsafe(24)
HOSTNAME = re.compile(r"[a-zA-Z0-9](?:[a-zA-Z0-9.-]*[a-zA-Z0-9])?\.?")


class DeviceTelemetry:
    """One cheap, shared sampler; HTTP requests never trigger hardware probes.

    CPU is aggregate busy time across all cores (0–100%, not per-core summed).
    Orin's GPU load node is in thousandths. Memory is system RAM unavailable,
    including CPU/GPU/OS use, not a CUDA allocator or dedicated VRAM counter.
    """

    GPU_LOAD_PATHS = (
        "/sys/devices/platform/bus@0/17000000.gpu/load",
        "/sys/devices/platform/17000000.gpu/load",
        "/sys/class/devfreq/17000000.gpu/device/load",
    )

    def __init__(self, proc_root=Path("/proc"), gpu_paths=None, interval=1.0):
        self.proc_root = Path(proc_root)
        self.gpu_paths = tuple(Path(p) for p in (self.GPU_LOAD_PATHS if gpu_paths is None else gpu_paths))
        self.interval = interval
        self.previous_cpu = None
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.thread = None
        self.sample_time = None
        self.sample = self.empty_sample()

    @staticmethod
    def empty_sample():
        return {"sampled_at": None, "status": "unavailable",
                "cpu": {"utilization_percent": None},
                "gpu": {"utilization_percent": None},
                "memory": {"used_bytes": None, "total_bytes": None,
                           "utilization_percent": None, "shared": True,
                           "measurement": "MemTotal - MemAvailable"}}

    @staticmethod
    def read_small(path):
        with path.open(encoding="ascii") as source:
            return source.read(8192)

    def collect(self):
        sample = self.empty_sample()
        try:
            fields = self.read_small(self.proc_root / "stat").splitlines()[0].split()
            if fields[0] != "cpu" or len(fields) < 9:
                raise ValueError("Missing aggregate CPU counters")
            # guest/guest_nice are already included in user/nice: do not add them.
            counters = tuple(int(value) for value in fields[1:9])
            if any(value < 0 for value in counters):
                raise ValueError("Negative CPU counter")
            previous, self.previous_cpu = self.previous_cpu, counters
            if previous is not None:
                delta = [now - before for now, before in zip(counters, previous)]
                total = sum(delta)
                if total > 0 and min(delta) >= 0:
                    idle = delta[3] + delta[4]
                    sample["cpu"]["utilization_percent"] = round(100 * (total - idle) / total, 1)
        except (OSError, ValueError, IndexError, UnicodeError):
            self.previous_cpu = None

        for path in self.gpu_paths:
            try:
                load = int(self.read_small(path).strip())
                if not 0 <= load <= 1000:
                    raise ValueError("GPU load outside permille range")
                sample["gpu"]["utilization_percent"] = load / 10
                break
            except (OSError, ValueError, UnicodeError):
                continue

        try:
            values = {}
            for line in self.read_small(self.proc_root / "meminfo").splitlines():
                fields = line.split()
                if fields and fields[0] in {"MemTotal:", "MemAvailable:"}:
                    if len(fields) != 3 or fields[2] != "kB":
                        raise ValueError("Unexpected memory units")
                    values[fields[0]] = int(fields[1]) * 1024
            total, available = values["MemTotal:"], values["MemAvailable:"]
            if not 0 <= available <= total or total <= 0:
                raise ValueError("Invalid memory counters")
            sample["memory"].update(used_bytes=total - available, total_bytes=total,
                                    utilization_percent=round(100 * (total - available) / total, 1))
        except (OSError, ValueError, KeyError, UnicodeError):
            pass

        present = [sample[name]["utilization_percent"] is not None for name in ("cpu", "gpu", "memory")]
        sample["status"] = "ok" if all(present) else "partial" if any(present) else "unavailable"
        sample["sampled_at"] = datetime.now(timezone.utc).isoformat()
        with self.lock:
            self.sample, self.sample_time = sample, time.monotonic()

    def snapshot(self):
        with self.lock:
            # Nested data is never mutated after publication by the sampler.
            return {**self.sample, "age_ms": None if self.sample_time is None else
                    round(max(0, time.monotonic() - self.sample_time) * 1000),
                    "sample_interval_ms": round(self.interval * 1000)}

    def start(self):
        def run():
            while not self.stopped.is_set():
                self.collect()
                if self.stopped.wait(self.interval):
                    break
        self.thread = threading.Thread(target=run, name="device-telemetry", daemon=True)
        self.thread.start()
        return self

    def close(self):
        self.stopped.set()
        if self.thread:
            self.thread.join(timeout=2)


def validate_request(payload):
    """Accept one bounded image and prompt, with no URLs or conversation history."""
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object.")
    allowed = {"model", "messages", "stream", "temperature", "max_tokens", "top_p", "stream_options",
               "max_image_tokens_per_image"}
    if "seed" in payload:
        raise ValueError("Upstream TensorRT-Edge-LLM does not support seed. Use temperature 0.")
    if set(payload) - allowed or payload.get("stream") is not True:
        raise ValueError("Only the supported streaming image request is accepted.")
    if not isinstance(payload.get("model"), str) or not 1 <= len(payload["model"]) <= 256:
        raise ValueError("A model name is required.")
    tokens = payload.get("max_tokens")
    if type(tokens) is not int or not 1 <= tokens <= 512:
        raise ValueError("max_tokens must be from 1 to 512.")
    if type(payload.get("temperature")) not in (int, float) or payload["temperature"] not in (0, 0.7):
        raise ValueError("Use temperature 0 (Lightweight) or 0.7 (Live VLM WebUI).")
    if "top_p" in payload and (type(payload["top_p"]) not in (int, float)
            or not math.isfinite(payload["top_p"]) or not 0 < payload["top_p"] <= 1):
        raise ValueError("top_p must be greater than 0 and at most 1.")
    if "max_image_tokens_per_image" in payload and (type(payload["max_image_tokens_per_image"]) is not int
            or not 4 <= payload["max_image_tokens_per_image"] <= 512):
        raise ValueError("Image token budget must be an integer from 4 to 512.")
    if "stream_options" in payload:
        options = payload["stream_options"]
        if not isinstance(options, dict) or set(options) != {"include_usage"} or options["include_usage"] is not True:
            raise ValueError("Only stream_options.include_usage=true is supported.")
    messages = payload.get("messages")
    if not isinstance(messages, list) or len(messages) != 1:
        raise ValueError("Send exactly one user message; history is disabled.")
    message = messages[0]
    if not isinstance(message, dict) or set(message) != {"role", "content"} or message["role"] != "user":
        raise ValueError("A single user message is required.")
    content = message["content"]
    if not isinstance(content, list) or len(content) != 2 or not all(isinstance(x, dict) for x in content):
        raise ValueError("Send one text prompt and one JPEG image.")
    texts = [x for x in content if x.get("type") == "text"]
    images = [x for x in content if x.get("type") == "image_url"]
    if len(texts) != 1 or len(images) != 1:
        raise ValueError("Send one text prompt and one JPEG image.")
    text = texts[0].get("text")
    if set(texts[0]) != {"type", "text"} or not isinstance(text, str) or not 1 <= len(text.strip()) <= 2000:
        raise ValueError("Prompt must contain 1–2000 characters.")
    image = images[0].get("image_url")
    if set(images[0]) != {"type", "image_url"} or not isinstance(image, dict) or set(image) != {"url"}:
        raise ValueError("Only an embedded JPEG is accepted.")
    url = image["url"]
    prefix = "data:image/jpeg;base64,"
    if not isinstance(url, str) or not url.startswith(prefix):
        raise ValueError("External image URLs are disabled. Send an embedded JPEG.")
    raw = base64.b64decode(url[len(prefix):], validate=True)
    if not raw.startswith(b"\xff\xd8\xff") or len(raw) > MAX_BODY * 3 // 4:
        raise ValueError("Invalid or oversized JPEG image.")


def parse_host(host_header):
    """(host, IP address or None) named by a Host header, which may carry a port."""
    if not isinstance(host_header, str) or not host_header or any(
            ord(char) <= 32 or ord(char) == 127 or char == "\\" for char in host_header):
        raise ValueError("Invalid Host header.")
    try:
        parsed = urlsplit("//" + host_header)
        host = parsed.hostname
        parsed.port  # Validate the caller's port before anything relies on the authority.
        if not host or parsed.username is not None or parsed.password is not None or \
                parsed.path or parsed.query or parsed.fragment:
            raise ValueError("Invalid host authority")
        try:
            return host, ipaddress.ip_address(host)
        except ValueError:
            if len(host) > 253 or not HOSTNAME.fullmatch(host):
                raise ValueError("Invalid hostname")
            return host, None
    except ValueError as exc:
        raise ValueError("Invalid Host header.") from exc


def is_localhost_name(host):
    name = host.rstrip(".").lower()
    return name == "localhost" or name.endswith(".localhost")


def camera_redirect_url(host_header, https_port):
    """Upgrade the same requested LAN host, preserving secure-context localhost.

    Never use the peer address: a forwarded browser request can arrive over a
    loopback connection while its actual document origin is still a LAN host.
    """
    if not https_port:
        return None
    host, address = parse_host(host_header)
    if address is None:
        if is_localhost_name(host):
            return None
        authority = host
    else:
        if address.is_loopback:
            return None
        authority = f"[{host}]" if address.version == 6 else host
    return f"https://{authority}:{https_port}/"


def host_allowed(host_header, allowed_hosts=frozenset()):
    """Whether the browser asked for this server by a name that cannot be DNS-rebound.

    Rebinding points an attacker's own name at this address: the browser then treats the
    attacker's page as same-origin here, sends Sec-Fetch-Site: same-origin, and would let it
    read /api/access. The one thing it cannot change is the name it asked for, in Host. An IP
    literal or localhost resolves nowhere else; any other name must be one the operator listed.
    """
    try:
        host, address = parse_host(host_header)
    except ValueError:
        return False
    return address is not None or is_localhost_name(host) or host.rstrip(".").lower() in allowed_hosts


def allowed_host_name(value):
    """An --allowed-host value: one DNS name, no scheme or port, as Host will be compared."""
    name = value.strip().rstrip(".").lower()
    if not name or len(name) > 253 or not HOSTNAME.fullmatch(name):
        raise ValueError(f"--allowed-host takes a bare host name such as orin.local, not {value!r}")
    return name


def bridge_address(url):
    """(host, port) of the Reachy bridge. Plain HTTP only: it never leaves the Jetson."""
    message = "--reachy-url must look like http://127.0.0.1:8099"
    try:
        parsed = urlsplit(url)
        port = parsed.port or 80
    except ValueError as exc:
        raise ValueError(message) from exc
    if parsed.scheme != "http" or not parsed.hostname or parsed.username is not None or \
            parsed.password is not None or parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError(message)
    return parsed.hostname, port


def media_type(response):
    return response.getheader("Content-Type", "").split(";")[0].strip().lower()


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 8

    def __init__(self, *args, telemetry=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.owns_telemetry = telemetry is None
        self.telemetry = telemetry if telemetry is not None else DeviceTelemetry().start()
        self.https_port = None
        # Names besides IP literals and localhost that /reachy/ and /api/access answer to.
        self.allowed_hosts = frozenset()

    def server_close(self):
        if getattr(self, "owns_telemetry", False):
            self.telemetry.close()
        super().server_close()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_request(self, code="-", size="-"):
        # Polling is routine; do not write a system journal line every second or two.
        if self.path.partition("?")[0] in {"/api/metrics", "/reachy/healthz"} and code == 200:
            return
        # Nor the relay token that every /reachy/ URL carries: more can read the journal than this page.
        self.log_message('"%s" %s %s', re.sub(r"([?&]token=)[^&\s]*", r"\1<redacted>", self.requestline),
                         str(getattr(code, "value", code)), str(size))

    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def send_headers(self, code, content_type, length=None, extra_headers=None):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
        self.send_header("Connection", "close")
        if length is not None:
            self.send_header("Content-Length", str(length))
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.close_connection = True

    def json_error(self, code, message):
        body = json.dumps({"error": {"message": message}}).encode()
        self.send_headers(code, "application/json", len(body))
        self.wfile.write(body)

    def origin_allowed(self):
        origin = self.headers_in.get("Origin")
        if not origin:
            return True
        try:
            parsed = urlsplit(origin)
            return parsed.scheme in {"http", "https"} and parsed.netloc == self.headers_in.get("Host")
        except ValueError:
            return False

    def fetch_site_allowed(self):
        # An <img> or <audio> request carries no Origin header, so origin_allowed() alone cannot
        # stop another site embedding the robot's camera or microphone and using up its stream
        # slots. Browsers send Fetch Metadata over HTTPS: only this page, or a typed URL, passes.
        # Plain HTTP gets no Fetch Metadata from most browsers, which is why the token exists.
        return self.headers_in.get("Sec-Fetch-Site", "same-origin") in {"same-origin", "none"}

    def host_refused(self):
        """Answer 421 unless Host names this server in a way DNS rebinding cannot fake."""
        if host_allowed(self.headers_in.get("Host"), self.server.allowed_hosts):
            return False
        self.json_error(421, "Live Vision does not answer to this host name. Open it by IP address, "
                             "or start it with --allowed-host for this name.")
        return True

    @staticmethod
    def relay_token_valid(query):
        supplied = parse_qs(query).get("token") or []
        # Bytes: compare_digest refuses str with non-ASCII characters, which a query can carry.
        return len(supplied) == 1 and secrets.compare_digest(supplied[0].encode(), RELAY_TOKEN.encode())

    def do_GET(self):
        self.headers_in = self.headers
        if not self.origin_allowed():
            self.json_error(403, "Cross-origin requests are disabled.")
            return
        if self.path == "/" and not isinstance(self.connection, ssl.SSLSocket):
            try:
                target = camera_redirect_url(self.headers.get("Host"), self.server.https_port)
            except ValueError as exc:
                self.json_error(400, str(exc))
                return
            if target:
                self.send_headers(302, "text/plain; charset=utf-8", 0, {"Location": target})
                return
        if self.path in {"/health/ready", "/v1/models", "/api/runtime"}:
            self.proxy(self.path)
            return
        # The page makes every stream URL it opens unique, so a reopened stream is really
        # fetched again; the video's ?stream= token also names it for X-Reachy-Stream.
        # Every route also needs the relay's ?token= (RELAY_TOKEN), checked after Host and Fetch
        # Metadata so those refusals keep saying what is wrong. Queries never reach the bridge.
        route, _, query = self.path.partition("?")
        if route in REACHY_FETCHES or route in REACHY_STREAMS:
            stream = (parse_qs(query).get("stream") or [""])[0]
            stream = stream if STREAM_TOKEN.fullmatch(stream) else None
            if self.host_refused():
                return
            if not self.fetch_site_allowed():
                self.json_error(403, "Cross-site requests are disabled.")
            elif not self.relay_token_valid(query):
                # Also what a page still open across a restart of this server gets: it asks
                # /api/access again for the new token.
                self.json_error(401, "Missing or out-of-date relay token. Reload Live Vision.")
            elif route in REACHY_STREAMS:
                self.reachy_stream(*REACHY_STREAMS[route], stream)
            else:
                self.reachy_fetch(*REACHY_FETCHES[route], stream)
            return
        if self.path == "/api/metrics":
            body = json.dumps(self.server.telemetry.snapshot()).encode()
            self.send_headers(200, "application/json", len(body))
            self.wfile.write(body)
            return
        if self.path == "/api/access":
            # The relay token goes only to a page that can read this answer: a cross-origin page
            # cannot, and a rebound name is refused here before it gets the chance.
            if self.host_refused():
                return
            body = json.dumps({"https_port": self.server.https_port, "reachy_token": RELAY_TOKEN}).encode()
            self.send_headers(200, "application/json", len(body))
            self.wfile.write(body)
            return
        # Only the bare page is upgraded to HTTPS above, for the camera. A link such as
        # /?source=reachy is served where it was asked for: the robot needs no secure
        # context, and a detour through a self-signed certificate would only add a warning.
        assets = {"/": ("index.html", "text/html; charset=utf-8"),
                  "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                  "/style.css": ("style.css", "text/css; charset=utf-8")}
        if route not in assets:
            self.json_error(404, "Not found.")
            return
        name, content_type = assets[route]
        data = (ROOT / name).read_bytes()
        self.send_headers(200, content_type, len(data))
        self.wfile.write(data)

    def do_POST(self):
        self.headers_in = self.headers
        if not self.origin_allowed():
            self.json_error(403, "Cross-origin requests are disabled.")
            return
        if self.path != "/v1/chat/completions":
            self.json_error(404, "Not found.")
            return
        try:
            if self.headers.get("Transfer-Encoding"):
                raise ValueError("Chunked uploads are disabled.")
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                raise ValueError("Content-Type must be application/json.")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY:
                self.json_error(413, "The image request must be smaller than 2 MiB.")
                return
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError("Incomplete request body.")
            payload = json.loads(body)
            validate_request(payload)
        except socket.timeout:
            self.json_error(408, "Request upload timed out.")
            return
        except (ValueError, TypeError, KeyError) as exc:
            self.json_error(400, str(exc))
            return
        if not self.acquire_generation():
            if self.client_disconnected():
                return
            self.json_error(429, "One inference is already active. Retry after it finishes.")
            return
        try:
            if self.client_disconnected():
                return
            self.proxy(self.path, body)
        finally:
            GENERATION_LOCK.release()

    def client_disconnected(self):
        # The complete body has been consumed and responses close the connection;
        # EOF or unexpected pipelined bytes mean this request must not start work.
        try:
            return bool(select.select([self.connection], [], [], 0)[0])
        except (OSError, ValueError):
            return True

    def acquire_generation(self):
        if GENERATION_LOCK.acquire(blocking=False):
            return True
        # A browser abort settles before the 200 ms disconnect watcher releases
        # the active slot. Admit at most one short handoff waiter, not a backlog.
        if not GENERATION_WAIT_LOCK.acquire(blocking=False):
            return False
        try:
            deadline = time.monotonic() + HANDOFF_SECONDS
            while not self.client_disconnected():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                if GENERATION_LOCK.acquire(timeout=min(0.05, remaining)):
                    return True
            return False
        finally:
            GENERATION_WAIT_LOCK.release()

    def proxy(self, path, body=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.backend_port,
                                               timeout=MAX_SECONDS if body else 3)
        finished = threading.Event()
        expired = threading.Event()
        stream_started = False
        response = None
        watcher = None
        try:
            connection.connect()
            upstream_socket = connection.sock
            if body:
                def watch_disconnect():
                    deadline = time.monotonic() + MAX_SECONDS
                    while not finished.wait(0.2):
                        if time.monotonic() >= deadline:
                            expired.set()
                            break
                        try:
                            ready, _, _ = select.select([self.connection], [], [], 0)
                            if ready:
                                # Requests do not pipeline on this close-after-response
                                # connection. EOF (or new unexpected bytes) cancels work.
                                self.connection.recv(1)
                                break
                        except (OSError, ValueError):
                            break
                    if not finished.is_set():
                        try:
                            upstream_socket.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass
                watcher = threading.Thread(target=watch_disconnect, daemon=True)
                watcher.start()
            connection.request("POST" if body else "GET", path, body=body,
                               headers={"Content-Type": "application/json", "Accept": "text/event-stream" if body else "application/json"})
            response = connection.getresponse()
            if response.status != 200:
                raw = response.read(MAX_RESPONSE + 1)
                try:
                    message = json.loads(raw).get("error", {}).get("message", "Backend request failed.")
                except (ValueError, AttributeError):
                    message = "Backend request failed."
                self.json_error(response.status, message)
                return
            if body:
                if "text/event-stream" not in response.getheader("Content-Type", ""):
                    raise ValueError("Backend did not return SSE token streaming.")
                self.send_headers(200, "text/event-stream; charset=utf-8")
                stream_started = True
                size = 0
                while True:
                    data = response.read1(4096)
                    if not data:
                        break
                    size += len(data)
                    if size > MAX_RESPONSE:
                        raise ValueError("Backend stream exceeded 1 MiB.")
                    self.wfile.write(data)
                    self.wfile.flush()
                if expired.is_set():
                    raise TimeoutError("Inference exceeded the 120-second request limit.")
            else:
                data = response.read(MAX_RESPONSE + 1)
                if len(data) > MAX_RESPONSE:
                    raise ValueError("Backend response exceeded the size limit.")
                self.send_headers(200, "application/json", len(data))
                self.wfile.write(data)
        except (OSError, http.client.HTTPException, ValueError) as exc:
            message = "Local TensorRT-Edge-LLM backend is not ready." if not body else str(exc)
            try:
                if stream_started:
                    self.wfile.write(("\nevent: error\ndata: " + json.dumps({"error": {"message": message}}) + "\n\n").encode())
                    self.wfile.flush()
                else:
                    self.json_error(503 if not body else 502, message)
            except OSError:
                pass
        finally:
            finished.set()
            if response:
                response.close()
            connection.close()
            if watcher:
                watcher.join(timeout=0.5)

    def reachy_refusal(self, status, raw):
        # Pass the bridge's own status and words through: its 503 says why there is no frame
        # ("no fresh frame (dormant: camera held by the robot app 'x')"), which is the useful part.
        text = raw[:MAX_RESPONSE].decode("utf-8", "replace").strip()[:300]
        self.json_error(status if 400 <= status <= 599 else 502,
                        f"Reachy bridge: {text or f'HTTP {status}'}")

    def reachy_fetch(self, path, content_type, token=None):
        """One bounded answer from the bridge: its health, or its newest frame."""
        extra_headers = {}
        with RELAYED_STREAMS_LOCK:
            entry = RELAYED_STREAMS.get(token) if token else None
            idle = None if entry is None else time.monotonic() - entry[0]
            slots = OPEN_REACHY_STREAMS
        if token:
            extra_headers["X-Reachy-Stream"] = "closed" if idle is None else f"idle_ms={round(idle * 1000)}"
        if path == "/healthz":
            # A refused <img> cannot see its 503, so health says whether the cap is why.
            extra_headers["X-Reachy-Slots"] = f"{slots}/{MAX_REACHY_STREAMS}"
        connection = http.client.HTTPConnection(*self.server.reachy_address, timeout=REACHY_SECONDS)
        response = None
        try:
            try:
                connection.request("GET", path, headers={"Accept": content_type})
                response = connection.getresponse()
                data = response.read(MAX_RESPONSE + 1)
            except (OSError, http.client.HTTPException):
                self.json_error(502, "Reachy bridge is not reachable.")
                return
            if len(data) > MAX_RESPONSE:
                self.json_error(502, "Reachy bridge response exceeded the size limit.")
            elif response.status != 200:
                self.reachy_refusal(response.status, data)
            elif media_type(response) != content_type:
                self.json_error(502, "Reachy bridge returned an unexpected content type.")
            else:
                self.send_headers(200, content_type, len(data), extra_headers)
                self.wfile.write(data)
        except OSError:
            pass  # The browser has gone; nobody is left to tell.
        finally:
            if response:
                response.close()
            connection.close()

    def reachy_stream(self, path, content_type, token=None):
        """Relay an endless bridge stream (MJPEG or MP3) until the browser or the bridge stops.

        Never takes GENERATION_LOCK: watching or listening to the robot must neither wait for
        inference nor hold it up. Every way out - browser gone, bridge restarted, idle timeout -
        ends the same way: both sockets closed, the slot returned, the token forgotten.
        """
        global OPEN_REACHY_STREAMS
        if not REACHY_STREAM_SLOTS.acquire(timeout=STREAM_HANDOFF_SECONDS):
            self.json_error(503, f"{MAX_REACHY_STREAMS} Reachy streams are already open on this server. "
                                 "Close another Live Vision tab or turn Listen off.")
            return
        with RELAYED_STREAMS_LOCK:
            OPEN_REACHY_STREAMS += 1
        connection = http.client.HTTPConnection(*self.server.reachy_address, timeout=REACHY_SECONDS)
        finished = threading.Event()
        response = None
        watcher = None
        started = False
        entry = [time.monotonic()]
        if token:
            with RELAYED_STREAMS_LOCK:
                RELAYED_STREAMS[token] = entry
        try:
            connection.connect()
            upstream_socket = connection.sock
            connection.request("GET", path, headers={"Accept": content_type})
            response = connection.getresponse()
            if response.status != 200:
                self.reachy_refusal(response.status, response.read(MAX_RESPONSE + 1))
                return
            # Relayed whole, because the multipart boundary lives in it.
            upstream_type = response.getheader("Content-Type", "")
            if media_type(response) != content_type or not re.fullmatch(r"[\x20-\x7e]{1,200}", upstream_type):
                self.json_error(502, "Reachy bridge returned an unexpected content type.")
                return
            upstream_socket.settimeout(STREAM_IDLE_SECONDS)

            def watch_disconnect():
                # The browser sends nothing after its GET on this close-after-response
                # connection, so readable means gone: Stop, img.src = "", a closed tab.
                # Shutting the bridge socket unblocks the relay now, not at the next frame,
                # which may be minutes away while the robot is dormant.
                while not finished.wait(0.2):
                    if self.client_disconnected():
                        break
                if not finished.is_set():
                    try:
                        upstream_socket.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
            watcher = threading.Thread(target=watch_disconnect, name="reachy-stream-watch", daemon=True)
            watcher.start()
            self.send_headers(200, upstream_type)
            started = True
            while True:
                data = response.read1(STREAM_CHUNK)
                if not data:
                    break
                self.wfile.write(data)
                entry[0] = time.monotonic()
        except (OSError, http.client.HTTPException):
            if not started:
                try:
                    self.json_error(502, "Reachy bridge is not reachable.")
                except OSError:
                    pass
        finally:
            finished.set()
            if token:
                with RELAYED_STREAMS_LOCK:
                    if RELAYED_STREAMS.get(token) is entry:
                        del RELAYED_STREAMS[token]
            if response:
                response.close()
            connection.close()
            if watcher:
                watcher.join(timeout=0.5)
            with RELAYED_STREAMS_LOCK:
                OPEN_REACHY_STREAMS -= 1
            REACHY_STREAM_SLOTS.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--backend-port", type=int, default=8000)
    parser.add_argument("--cert", type=Path)
    parser.add_argument("--key", type=Path)
    parser.add_argument("--https-port", type=int,
                        help="Add HTTPS on this port while the primary port serves HTTP.")
    parser.add_argument("--allow-insecure-lan", action="store_true",
                        help="Explicitly allow LAN HTTP for image uploads. Cameras need HTTPS.")
    parser.add_argument("--reachy-url", default="http://127.0.0.1:8099",
                        help="Reachy Mini bridge relayed under /reachy/ (plain HTTP on the Jetson).")
    parser.add_argument("--allowed-host", action="append", default=[], metavar="NAME",
                        help="A host name (such as orin.local) that /reachy/ and /api/access answer "
                             "to, besides IP addresses and localhost. Repeat for more names.")
    args = parser.parse_args()
    try:
        allowed_hosts = frozenset(allowed_host_name(name) for name in args.allowed_host)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        loopback = args.host == "localhost" or ipaddress.ip_address(args.host).is_loopback
    except ValueError:
        parser.error("--host must be an IP address or localhost.")
    try:
        reachy_address = bridge_address(args.reachy_url)
    except ValueError as exc:
        parser.error(str(exc))
    if any(not 1 <= port <= 65535 for port in
           (args.port, args.backend_port, *([args.https_port] if args.https_port is not None else []))):
        parser.error("Ports must be from 1 to 65535.")
    if bool(args.cert) != bool(args.key):
        parser.error("Provide both --cert and --key.")
    if args.https_port is not None and (not args.cert or args.https_port == args.port):
        parser.error("--https-port requires a certificate and a different port from --port.")
    primary_tls = bool(args.cert) and args.https_port is None
    if not loopback and not primary_tls and not args.allow_insecure_lan:
        parser.error("LAN HTTP requires explicit --allow-insecure-lan. Cameras need HTTPS.")
    if ":" in args.host:
        Server.address_family = socket.AF_INET6
    context = None
    if args.cert:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(args.cert, args.key)
    servers = []
    threads = []
    try:
        server = Server((args.host, args.port), Handler)
        servers.append(server)
        server.backend_port = args.backend_port
        server.reachy_address = reachy_address
        server.https_port = args.https_port
        server.allowed_hosts = allowed_hosts
        if primary_tls:
            server.socket = context.wrap_socket(server.socket, server_side=True)
        if args.https_port is not None:
            secure = Server((args.host, args.https_port), Handler, telemetry=server.telemetry)
            servers.append(secure)
            secure.backend_port = args.backend_port
            secure.reachy_address = reachy_address
            secure.allowed_hosts = allowed_hosts
            secure.socket = context.wrap_socket(secure.socket, server_side=True)
            worker = threading.Thread(target=secure.serve_forever, name="https-ui", daemon=True)
            worker.start()
            threads.append((secure, worker))
            print(f"Camera UI: https://{args.host}:{args.https_port}", flush=True)
        scheme = "https" if primary_tls else "http"
        print(f"UI: {scheme}://{args.host}:{args.port}; backend: http://127.0.0.1:{args.backend_port}", flush=True)
        print(f"Reachy Mini bridge: {args.reachy_url}, relayed under /reachy/", flush=True)
        print("UI availability does not imply model or Jetson readiness.", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for running, worker in threads:
            running.shutdown()
            worker.join(timeout=2)
        for server in reversed(servers):
            server.server_close()


if __name__ == "__main__":
    main()
