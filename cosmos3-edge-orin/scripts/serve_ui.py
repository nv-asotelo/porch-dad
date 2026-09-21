#!/usr/bin/env python3
"""Serve the original camera UI and proxy only the loopback Edge-LLM API.

No model, cloud fallback, downloaded dependency or synthetic inference lives here.
Run on the Jetson alongside tensorrt-edgellm-serve. See research/ui-feasibility.md.
"""

import argparse
import base64
from datetime import datetime, timezone
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
from pathlib import Path
import re
import select
import socket
import ssl
import threading
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1] / "web"
MAX_BODY = 2 * 1024 * 1024
MAX_RESPONSE = 1024 * 1024
MAX_SECONDS = 120
GENERATION_LOCK = threading.Lock()
GENERATION_WAIT_LOCK = threading.Lock()
HANDOFF_SECONDS = 0.75


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
    allowed = {"model", "messages", "stream", "temperature", "max_tokens", "top_p", "stream_options"}
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
    if "top_p" in payload and (type(payload["top_p"]) not in (int, float) or payload["top_p"] != 1):
        raise ValueError("This deployment uses top_p 1.")
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


def camera_redirect_url(host_header, https_port):
    """Upgrade the same requested LAN host, preserving secure-context localhost.

    Never use the peer address: a forwarded browser request can arrive over a
    loopback connection while its actual document origin is still a LAN host.
    """
    if not https_port:
        return None
    if not isinstance(host_header, str) or not host_header or any(
            ord(char) <= 32 or ord(char) == 127 or char == "\\" for char in host_header):
        raise ValueError("Invalid Host header.")
    try:
        parsed = urlsplit("//" + host_header)
        host = parsed.hostname
        parsed.port  # Validate the caller's port before building our authority.
        if not host or parsed.username is not None or parsed.password is not None or \
                parsed.path or parsed.query or parsed.fragment:
            raise ValueError("Invalid host authority")
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            if len(host) > 253 or not re.fullmatch(r"[a-zA-Z0-9](?:[a-zA-Z0-9.-]*[a-zA-Z0-9])?\.?", host):
                raise ValueError("Invalid hostname")
            if host.rstrip(".").lower() == "localhost" or host.rstrip(".").lower().endswith(".localhost"):
                return None
            authority = host
        else:
            if address.is_loopback:
                return None
            authority = f"[{host}]" if address.version == 6 else host
    except ValueError as exc:
        raise ValueError("Invalid Host header.") from exc
    return f"https://{authority}:{https_port}/"


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 8

    def __init__(self, *args, telemetry=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.owns_telemetry = telemetry is None
        self.telemetry = telemetry if telemetry is not None else DeviceTelemetry().start()
        self.https_port = None

    def server_close(self):
        if getattr(self, "owns_telemetry", False):
            self.telemetry.close()
        super().server_close()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_request(self, code="-", size="-"):
        # Polling is routine; do not write a system journal line every second.
        if self.path == "/api/metrics" and code == 200:
            return
        super().log_request(code, size)

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
        if self.path in {"/health/ready", "/v1/models"}:
            self.proxy(self.path)
            return
        if self.path == "/api/metrics":
            body = json.dumps(self.server.telemetry.snapshot()).encode()
            self.send_headers(200, "application/json", len(body))
            self.wfile.write(body)
            return
        if self.path == "/api/access":
            body = json.dumps({"https_port": self.server.https_port}).encode()
            self.send_headers(200, "application/json", len(body))
            self.wfile.write(body)
            return
        assets = {"/": ("index.html", "text/html; charset=utf-8"),
                  "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                  "/style.css": ("style.css", "text/css; charset=utf-8")}
        if self.path not in assets:
            self.json_error(404, "Not found.")
            return
        name, content_type = assets[self.path]
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
    args = parser.parse_args()
    try:
        loopback = args.host == "localhost" or ipaddress.ip_address(args.host).is_loopback
    except ValueError:
        parser.error("--host must be an IP address or localhost.")
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
        server.https_port = args.https_port
        if primary_tls:
            server.socket = context.wrap_socket(server.socket, server_side=True)
        if args.https_port is not None:
            secure = Server((args.host, args.https_port), Handler, telemetry=server.telemetry)
            servers.append(secure)
            secure.backend_port = args.backend_port
            secure.socket = context.wrap_socket(secure.socket, server_side=True)
            worker = threading.Thread(target=secure.serve_forever, name="https-ui", daemon=True)
            worker.start()
            threads.append((secure, worker))
            print(f"Camera UI: https://{args.host}:{args.https_port}", flush=True)
        scheme = "https" if primary_tls else "http"
        print(f"UI: {scheme}://{args.host}:{args.port}; backend: http://127.0.0.1:{args.backend_port}", flush=True)
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
