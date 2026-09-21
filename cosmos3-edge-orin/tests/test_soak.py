"""Soak policy and transport fixtures only; never run a real model or short pass."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import signal
import sys
import threading
import time
import json
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import soak


def evidence():
    records = [{"case_id": str(index), "error": None, "stream_done": True,
                "finish_reason": "stop", "output_text": "Transport fixture only."} for index in range(6)]
    samples = [{"kind": "proc", "monotonic_seconds": second, "process_rss_bytes": 3 << 30,
                "system_ram_total_bytes": 8 << 30, "system_ram_unavailable_bytes": 4 << 30,
                "system_swap_occupied_bytes": 0,
                "swap_in_pages": 7, "swap_out_pages": 9, "system_oom_kills": 0} for second in range(601)]
    health = [{"monotonic_seconds": second, "http_status": 200, "error": None,
               "data": {"status": "ready", "active_requests": 1, "queued_requests": 0}} for second in range(601)]
    return records, samples, health


def evaluate(records, samples, health, **overrides):
    options = dict(started=0, ended=600, expected_cases=[str(i) for i in range(6)], requested_seconds=600)
    return soak.evaluate(records, samples, health, **(options | overrides))


class SoakPolicyTests(unittest.TestCase):
    def test_complete_stable_synthetic_evidence_exercises_passing_policy(self):
        result = evaluate(*evidence())
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["minimum_available_bytes"], 4 << 30)
        self.assertEqual(result["counter_deltas"]["swap_in_pages"], 0)
        self.assertEqual(result["growth"]["process_rss_bytes"]["growth_bytes"], 0)

    def test_short_or_missing_evidence_cannot_pass(self):
        for overrides in ({"ended": 599}, {"pid_unchanged": False}, {"provenance_unchanged": False},
                          {"sampler_errors": ["fixture sampling failure"]}, {"run_errors": ["interrupted"]}):
            with self.subTest(overrides=overrides):
                self.assertEqual(evaluate(*evidence(), **overrides)["status"], "failed")
        records, samples, health = evidence()
        for args in (([], samples, health), (records, [], health), (records, samples, []),
                     (records, samples[20:], health), (records, samples, health[:-20])):
            self.assertEqual(evaluate(*args)["status"], "failed")
        del samples[250]["swap_out_pages"]
        self.assertIn("missing_or_invalid_memory_evidence", evaluate(records, samples, health)["failure_reasons"])

    def test_swap_oom_headroom_and_rss_growth_are_detected(self):
        for counter in ("swap_in_pages", "swap_out_pages", "system_oom_kills"):
            records, samples, health = evidence()
            for sample in samples[300:]:
                sample[counter] += 1
            self.assertIn(counter + "_increased", evaluate(records, samples, health)["failure_reasons"])
        records, samples, health = evidence()
        samples[300]["system_ram_unavailable_bytes"] = (8 << 30) - (511 << 20)
        self.assertIn("available_memory_below_512_mib_or_unknown", evaluate(records, samples, health)["failure_reasons"])
        records, samples, health = evidence()
        for sample in samples[540:]:
            sample["process_rss_bytes"] += 200 << 20
        self.assertIn("backend_rss_growth_exceeded_frozen_tolerance", evaluate(records, samples, health)["failure_reasons"])

    def test_system_growth_is_reported_separately_and_counter_reset_is_unknown(self):
        records, samples, health = evidence()
        for sample in samples[540:]:
            sample["system_ram_unavailable_bytes"] += 200 << 20
        result = evaluate(records, samples, health)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["growth"]["system_ram_unavailable_bytes"]["growth_bytes"], 200 << 20)
        self.assertEqual(result["growth"]["process_rss_bytes"]["growth_bytes"], 0)
        samples[-1]["swap_in_pages"] = 0
        self.assertIsNone(evaluate(records, samples, health)["counter_deltas"]["swap_in_pages"])

    def test_errors_truncation_length_and_backlog_refuse_pass(self):
        for change in ({"error": {"message": "backend error"}}, {"stream_done": False},
                       {"finish_reason": "length"}, {"output_text": " "}):
            records, samples, health = evidence()
            records[0].update(change)
            self.assertIn("request_error_truncation_or_output_limit", evaluate(records, samples, health)["failure_reasons"])
        for field in ("active_requests", "queued_requests"):
            records, samples, health = evidence()
            health[300]["data"][field] = 2
            self.assertIn("active_or_queued_requests_exceeded_one_or_unknown", evaluate(records, samples, health)["failure_reasons"])
        records, samples, health = evidence()
        health[300]["error"] = "timed out"
        self.assertIn("missing_unready_or_invalid_health_evidence", evaluate(records, samples, health)["failure_reasons"])

    def test_actual_manifest_is_used_without_modifying_prompt_or_request_settings(self):
        manifest, _, workloads = soak.load_workloads(ROOT / "benchmarks/fixtures/jpeg-manifest.json")
        self.assertEqual(len(workloads), 6)
        for case, workload in zip(manifest["cases"], workloads):
            self.assertEqual(workload["payload"]["messages"][0]["content"][0]["text"], case["prompt"])
            self.assertEqual(workload["payload"]["max_tokens"], 64)
            self.assertEqual(workload["payload"]["temperature"], 0)

    def test_rotation_is_serial_and_stops_after_failed_sample(self):
        workloads = [{"case_id": str(index), "payload": {"index": index}} for index in range(6)]
        clock, calls, active = [0], [], [0]
        def request(_url, payload, _timeout):
            active[0] += 1
            self.assertEqual(active[0], 1)
            calls.append(payload["index"])
            clock[0] += 100  # Unit-test clock only; no short accepted real run.
            active[0] -= 1
            return dict(error=None, stream_done=True, finish_reason="stop", output_text="fixture")
        records = []
        with patch.object(soak.time, "monotonic", side_effect=lambda: clock[0]), \
             patch.object(soak, "bounded_request", side_effect=request):
            soak.stream_sequence("unused", workloads, records, lambda _: None, started=0, duration=650, timeout=120)
        self.assertEqual(calls, [0, 1, 2, 3, 4, 5, 0])
        with patch.object(soak, "bounded_request", return_value={"error": {"message": "fixture"}, "finish_reason": None}):
            failed = []
            soak.stream_sequence("unused", workloads, failed, lambda _: None, started=time.monotonic(), duration=600, timeout=120)
            self.assertEqual(len(failed), 1)


class DeadlineTests(unittest.TestCase):
    @unittest.skipUnless(hasattr(signal, "setitimer"), "Requires POSIX wall timer")
    def test_stalled_sse_has_a_real_wall_deadline(self):
        release = threading.Event()
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.flush()
                release.wait(2)
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            started = time.monotonic()
            result = soak.bounded_request(f"http://127.0.0.1:{server.server_port}/v1/chat/completions", {}, .1)
            self.assertLess(time.monotonic() - started, 1)
            self.assertIsNotNone(result["error"])
            self.assertFalse(result["stream_done"])
            self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0, 0))
        finally:
            release.set()
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_health_body_limit_wall_deadline_and_shutdown_completion(self):
        release, entered = threading.Event(), threading.Event()
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass
            def do_GET(self):
                self.send_response(200)
                data = (b"x" * 65537 if self.path == "/large" else json.dumps({
                    "status": "ready", "active_requests": 1, "queued_requests": 0, "extra": "not retained"}).encode())
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.flush()
                if self.path == "/stall":
                    entered.set()
                    release.wait(2)
                try:
                    self.wfile.write(data)
                except OSError:
                    pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        origin = f"http://127.0.0.1:{server.server_port}"
        try:
            with self.assertRaisesRegex(ValueError, "64 KiB"):
                soak.read_health(origin + "/large")
            ready = soak.read_health(origin + "/ready")
            self.assertEqual(ready["data"], {"status": "ready", "active_requests": 1, "queued_requests": 0})
            started = time.monotonic()
            with self.assertRaises(Exception):
                soak.read_health(origin + "/stall", timeout=.1)
            self.assertLess(time.monotonic() - started, 1)
            writes = []
            observer = soak.Observer(soak.LocalSampler(.5, None, None), origin + "/ready", writes.append)
            observer.start()
            self.assertTrue(observer.stop())
            self.assertFalse(observer.thread.is_alive())
            self.assertEqual(len(writes), 1, "No late health writes may remain after stop returns")
            # Stop while the health response body is still blocked. The poll
            # must finish with failed evidence before output can be closed.
            writes.clear()
            entered.clear()
            observer = soak.Observer(soak.LocalSampler(.5, None, None), origin + "/stall", writes.append)
            observer.thread = threading.Thread(target=observer.capture, daemon=True)
            actual_read = soak.read_health
            with patch.object(soak, "read_health", side_effect=lambda url: actual_read(url, timeout=.1)):
                observer.thread.start()
                self.assertTrue(entered.wait(1))
                self.assertTrue(observer.stop())
            self.assertFalse(observer.thread.is_alive())
            self.assertEqual(len(writes), 1)
            self.assertIsNotNone(writes[0]["error"])
        finally:
            release.set()
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
