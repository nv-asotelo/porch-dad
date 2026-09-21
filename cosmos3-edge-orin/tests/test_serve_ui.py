"""Loopback transport fixtures only; these tests perform no model inference."""
import base64
import copy
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import socket
import threading
import time
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("serve_ui", Path(__file__).resolve().parents[1] / "scripts/serve_ui.py")
ui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ui)


class TelemetryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.gpu = self.root / "gpu-load"
        self.monitor = ui.DeviceTelemetry(self.root, [self.gpu])
        (self.root / "meminfo").write_text("MemTotal: 8000 kB\nMemAvailable: 2000 kB\n")
        (self.root / "stat").write_text("cpu 100 0 50 800 50 0 0 0 100 0\n")
        self.gpu.write_text("756\n")

    def test_real_units_and_aggregate_cpu_delta_without_guest_double_count(self):
        self.monitor.collect()
        self.assertIsNone(self.monitor.snapshot()["cpu"]["utilization_percent"])
        (self.root / "stat").write_text("cpu 150 0 50 850 50 0 0 0 150 0\n")
        self.monitor.collect()
        result = self.monitor.snapshot()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["cpu"]["utilization_percent"], 50)
        self.assertEqual(result["gpu"]["utilization_percent"], 75.6)
        self.assertEqual(result["memory"]["used_bytes"], 6000 * 1024)
        self.assertEqual(result["memory"]["total_bytes"], 8000 * 1024)
        self.assertEqual(result["memory"]["utilization_percent"], 75)
        self.assertTrue(result["memory"]["shared"])
        self.assertLess(result["age_ms"], 1000)

    def test_missing_or_invalid_inputs_clear_values_instead_of_reusing_them(self):
        self.monitor.collect()
        (self.root / "stat").unlink()
        self.gpu.write_text("1001")
        (self.root / "meminfo").write_text("MemTotal: 8000 kB\nMemAvailable: 9000 kB\n")
        self.monitor.collect()
        result = self.monitor.snapshot()
        self.assertEqual(result["status"], "unavailable")
        self.assertTrue(all(result[name]["utilization_percent"] is None for name in ("cpu", "gpu", "memory")))
        self.assertIsNone(result["memory"]["used_bytes"])

    def test_counter_reset_cannot_report_fake_busy_percent(self):
        self.monitor.collect()
        (self.root / "stat").write_text("cpu 1 0 1 8 0 0 0 0 0 0\n")
        self.monitor.collect()
        self.assertIsNone(self.monitor.snapshot()["cpu"]["utilization_percent"])
        self.assertEqual(self.monitor.snapshot()["status"], "partial")

    def test_gpu_zero_is_valid_but_unreadable_node_is_not_zero(self):
        self.gpu.write_text("0")
        self.monitor.collect()
        self.assertEqual(self.monitor.snapshot()["gpu"]["utilization_percent"], 0)
        self.gpu.unlink()
        self.monitor.collect()
        self.assertIsNone(self.monitor.snapshot()["gpu"]["utilization_percent"])

    def test_snapshot_reads_only_cache_and_reports_age(self):
        self.monitor.collect()
        self.gpu.write_text("999")
        self.monitor.sample_time -= 6
        result = self.monitor.snapshot()
        self.assertEqual(result["gpu"]["utilization_percent"], 75.6)
        self.assertGreaterEqual(result["age_ms"], 6000)


def payload():
    return {"model": "transport-fixture-no-inference", "stream": True, "temperature": 0,
            "max_tokens": 64, "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Protocol test only."},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," +
                 base64.b64encode(b"\xff\xd8\xff\xd9").decode()}}]}]}


class Fixture(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_GET(self):
        value = {"status": "ready"} if self.path == "/health/ready" else {"data": [{"id": "transport-fixture-no-inference"}]}
        data = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        with self.server.count_lock:
            self.server.posts += 1
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(b": transport test only\n\n")
        self.wfile.flush()
        self.connection.settimeout(3)
        try:
            if self.connection.recv(1) == b"":
                self.server.cancelled.set()
        except OSError:
            pass


class CountingHandler(ui.Handler):
    def proxy(self, path, body=None):
        if not body:
            return super().proxy(path, body)
        with self.server.count_lock:
            self.server.active += 1
            self.server.peak_active = max(self.server.peak_active, self.server.active)
        try:
            return super().proxy(path, body)
        finally:
            with self.server.count_lock:
                self.server.active -= 1


class ProxyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = ThreadingHTTPServer(("127.0.0.1", 0), Fixture)
        cls.backend.cancelled = threading.Event()
        cls.backend.count_lock = threading.Lock()
        cls.backend.posts = 0
        cls.proxy = ui.Server(("127.0.0.1", 0), CountingHandler)
        cls.proxy.count_lock = threading.Lock()
        cls.proxy.active = cls.proxy.peak_active = 0
        cls.proxy.backend_port = cls.backend.server_port
        for server in (cls.backend, cls.proxy):
            threading.Thread(target=server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        for server in (cls.proxy, cls.backend):
            server.shutdown()
            server.server_close()

    def request(self, method, path, value=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.proxy.server_port, timeout=4)
        body = None if value is None else json.dumps(value)
        connection.request(method, path, body, headers or {"Content-Type": "application/json"})
        response = connection.getresponse()
        status, data = response.status, response.read()
        connection.close()
        return status, data

    def wait_for(self, predicate, message, timeout=2):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(predicate(), message)

    def open_stream(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.proxy.server_port, timeout=4)
        connection.request("POST", "/v1/chat/completions", json.dumps(payload()),
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        self.assertEqual(response.readline(), b": transport test only\n")
        return connection, response

    def test_static_and_readiness(self):
        for path in ("/", "/app.js", "/style.css", "/health/ready", "/v1/models"):
            self.assertEqual(self.request("GET", path)[0], 200)
        self.assertEqual(self.request("GET", "/../scripts/serve_ui.py")[0], 404)

    def test_metrics_during_inference_and_same_origin_guard(self):
        with ui.GENERATION_LOCK:
            status, body = self.request("GET", "/api/metrics")
        self.assertEqual(status, 200)
        self.assertIn("sampled_at", json.loads(body))
        status, _ = self.request("GET", "/api/metrics", headers={"Origin": "https://unrelated.invalid"})
        self.assertEqual(status, 403)

    def test_unavailable_is_explicit(self):
        unused = socket.socket()
        unused.bind(("127.0.0.1", 0))
        port = unused.getsockname()[1]
        unused.close()
        try:
            self.proxy.backend_port = port
            status, data = self.request("GET", "/health/ready")
            self.assertEqual(status, 503)
            self.assertIn(b"not ready", data)
        finally:
            self.proxy.backend_port = self.backend.server_port

    def test_request_bounds_and_no_remote_media(self):
        original = payload()
        for tokens in (0, 513, True, 1.5):
            value = copy.deepcopy(original)
            value["max_tokens"] = tokens
            self.assertEqual(self.request("POST", "/v1/chat/completions", value)[0], 400)
        value = copy.deepcopy(original)
        value["max_tokens"] = 512
        ui.validate_request(value)
        value = copy.deepcopy(original)
        value["messages"][0]["content"][1]["image_url"]["url"] = "https://example.org/image.jpg"
        self.assertEqual(self.request("POST", "/v1/chat/completions", value)[0], 400)
        value = copy.deepcopy(original)
        value["messages"].append(copy.deepcopy(value["messages"][0]))
        self.assertEqual(self.request("POST", "/v1/chat/completions", value)[0], 400)
        # Oversized uploads are rejected from headers before their bodies are
        # consumed. Sending only the header avoids an expected broken pipe.
        connection = http.client.HTTPConnection("127.0.0.1", self.proxy.server_port, timeout=4)
        connection.putrequest("POST", "/v1/chat/completions")
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Content-Length", str(ui.MAX_BODY + 1))
        connection.endheaders()
        response = connection.getresponse()
        self.assertEqual(response.status, 413)
        response.read()
        connection.close()

    def test_same_origin(self):
        status, _ = self.request("POST", "/v1/chat/completions", payload(),
                                 {"Content-Type": "application/json", "Origin": "https://example.org"})
        self.assertEqual(status, 403)

    def test_lan_document_redirect_and_loopback_api_preservation(self):
        self.proxy.https_port = 8443
        self.addCleanup(setattr, self.proxy, "https_port", None)
        for host, expected in (("192.168.6.252:8090", "https://192.168.6.252:8443/"),
                               ("orin.local:8090", "https://orin.local:8443/"),
                               ("[fd00::10]:8090", "https://[fd00::10]:8443/")):
            with self.subTest(host=host):
                connection = http.client.HTTPConnection("127.0.0.1", self.proxy.server_port, timeout=4)
                connection.request("GET", "/", headers={"Host": host})
                response = connection.getresponse()
                self.assertEqual(response.status, 302)
                self.assertEqual(response.getheader("Location"), expected)
                self.assertEqual(response.getheader("Cache-Control"), "no-store")
                self.assertEqual(response.getheader("Content-Length"), "0")
                self.assertEqual(response.read(), b"")
                connection.close()
        for host in ("localhost:8090", "127.0.0.1:8090", "127.0.0.2:8090", "[::1]:8090"):
            self.assertEqual(self.request("GET", "/", headers={"Host": host})[0], 200)
        for path in ("/health/ready", "/v1/models", "/api/metrics", "/api/access", "/app.js"):
            self.assertEqual(self.request("GET", path, headers={"Host": "192.168.6.252:8090"})[0], 200)
        for host in ("user@attacker.test:8090", "orin.local/path", "orin.local?x=y",
                     "orin.local#fragment", "orin.local:bogus", "orin.local:99999", "orin\\evil.test"):
            self.assertEqual(self.request("GET", "/", headers={"Host": host})[0], 400)

    def test_safe_usage_options_and_unsupported_seed(self):
        value = payload()
        value.update(top_p=1, stream_options={"include_usage": True})
        ui.validate_request(value)
        for options in ({"include_usage": False}, {"include_usage": 1}, {"include_usage": True, "other": True}):
            value["stream_options"] = options
            with self.assertRaises(ValueError):
                ui.validate_request(value)
        value = payload()
        value["seed"] = 42
        status, data = self.request("POST", "/v1/chat/completions", value)
        self.assertEqual(status, 400)
        self.assertIn(b"does not support seed", data)

    def test_live_vlm_temperature_keeps_backend_sampling_defaults(self):
        value = payload()
        value["temperature"] = 0.7
        value.pop("top_p", None)
        ui.validate_request(value)
        for temperature in (True, "0.7", -1, 0.5, float("nan"), float("inf")):
            value["temperature"] = temperature
            with self.assertRaises(ValueError):
                ui.validate_request(value)

    def test_stream_flush_admission_and_cancellation(self):
        self.backend.cancelled.clear()
        connection = http.client.HTTPConnection("127.0.0.1", self.proxy.server_port, timeout=4)
        started = time.monotonic()
        connection.request("POST", "/v1/chat/completions", json.dumps(payload()), {"Content-Type": "application/json"})
        response = connection.getresponse()
        try:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.readline(), b": transport test only\n")
            self.assertLess(time.monotonic() - started, 1)
            waiting = time.monotonic()
            self.assertEqual(self.request("POST", "/v1/chat/completions", payload())[0], 429)
            self.assertGreaterEqual(time.monotonic() - waiting, ui.HANDOFF_SECONDS * 0.8)
            self.assertLess(time.monotonic() - waiting, ui.HANDOFF_SECONDS + 1)
        finally:
            response.close()
            connection.close()
        self.assertTrue(self.backend.cancelled.wait(2), "disconnect was not forwarded upstream")
        for _ in range(20):
            if not ui.GENERATION_LOCK.locked():
                break
            time.sleep(0.05)
        self.assertFalse(ui.GENERATION_LOCK.locked(), "single-flight lock was not released")

    def test_one_pending_restart_handoffs_after_cancel_and_third_is_rejected(self):
        active, response = self.open_stream()
        pending = http.client.HTTPConnection("127.0.0.1", self.proxy.server_port, timeout=4)
        result, errors = [], []
        finished = threading.Event()
        before_posts = self.backend.posts

        def restart():
            try:
                pending.request("POST", "/v1/chat/completions", json.dumps(payload()),
                                {"Content-Type": "application/json"})
                result.append(pending.getresponse())
            except Exception as error:
                errors.append(error)
            finally:
                finished.set()

        worker = threading.Thread(target=restart, daemon=True)
        try:
            worker.start()
            self.wait_for(ui.GENERATION_WAIT_LOCK.locked, "restart did not occupy the one pending slot")
            self.assertFalse(finished.is_set(), "restart bypassed active request")
            started = time.monotonic()
            self.assertEqual(self.request("POST", "/v1/chat/completions", payload())[0], 429)
            self.assertLess(time.monotonic() - started, ui.HANDOFF_SECONDS / 2)
            response.close()
            active.close()
            self.assertTrue(finished.wait(2), "pending restart did not proceed after cancellation")
            self.assertEqual(errors, [])
            self.assertEqual(result[0].status, 200)
            self.assertEqual(result[0].readline(), b": transport test only\n")
            self.assertEqual(self.backend.posts, before_posts + 1, "rejected third request reached backend")
            self.assertEqual(self.proxy.peak_active, 1, "proxy forwarded simultaneous generations")
            self.assertFalse(ui.GENERATION_WAIT_LOCK.locked())
        finally:
            response.close()
            active.close()
            for item in result:
                item.close()
            pending.close()
            worker.join(timeout=2)
            self.wait_for(lambda: not ui.GENERATION_LOCK.locked(), "handoff test leaked the active lock")

    def test_disconnected_pending_client_never_reaches_backend(self):
        active, response = self.open_stream()
        pending = http.client.HTTPConnection("127.0.0.1", self.proxy.server_port, timeout=4)
        before_posts = self.backend.posts
        try:
            pending.request("POST", "/v1/chat/completions", json.dumps(payload()),
                            {"Content-Type": "application/json"})
            self.wait_for(ui.GENERATION_WAIT_LOCK.locked, "pending request was not admitted")
            pending.sock.shutdown(socket.SHUT_RDWR)
            pending.close()
            self.wait_for(lambda: not ui.GENERATION_WAIT_LOCK.locked(),
                          "cancelled pending request retained the waiting slot", timeout=0.5)
            self.assertTrue(ui.GENERATION_LOCK.locked(), "pending cancellation disturbed active request")
            self.assertEqual(self.backend.posts, before_posts)
        finally:
            pending.close()
            response.close()
            active.close()
            self.wait_for(lambda: not ui.GENERATION_LOCK.locked(), "cancellation test leaked the active lock")


if __name__ == "__main__":
    unittest.main()
