"""Protocol and stopping-policy tests. These do not exercise a GPU or Jetson."""

import importlib.util
import argparse
import hashlib
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock


SPEC = importlib.util.spec_from_file_location("benchmark", Path(__file__).parents[1] / "scripts" / "benchmark.py")
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class SSETests(unittest.TestCase):
    def test_every_byte_boundary_unicode_crlf_comments_and_multiline(self):
        wire = '\ufeff: heartbeat\r\nid: 7\r\ndata: café\r\ndata: 世界\r\n\r\ndata: [DONE]\r\n\r\n'.encode()
        parser = benchmark.SSEDecoder()
        events = []
        for byte in wire:
            events.extend(parser.feed(bytes([byte])))
        events.extend(parser.feed(b"", final=True))
        self.assertEqual(events, ["café\n世界", "[DONE]"])

    def test_cr_only_and_incomplete_event(self):
        parser = benchmark.SSEDecoder()
        self.assertEqual(parser.feed(b"data: first\r\rdata: unfinished"), ["first"])
        self.assertEqual(parser.feed(b"", final=True), [])

    def test_empty_data_is_an_event_but_comments_are_not(self):
        parser = benchmark.SSEDecoder()
        self.assertEqual(parser.feed(b": a\n\ndata:\n\ndata:  space\n\n"), ["", " space"])


class StreamTests(unittest.TestCase):
    def test_payload_omits_seed_rejected_by_pinned_backend(self):
        workload = {"model": "Cosmos3-Edge", "max_tokens": 64, "temperature": 0, "top_p": 1,
                    "prompt": "Describe this scene.", "image_mime_type": "image/jpeg"}
        payload = benchmark.build_payload(workload, b"test JPEG bytes")
        self.assertNotIn("seed", payload)
        self.assertEqual(payload["model"], "Cosmos3-Edge")
        self.assertEqual(payload["stream_options"], {"include_usage": True})
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["top_p"], 1)
        self.assertTrue(payload["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_live_payload_omits_top_p_instead_of_sending_null(self):
        workload = {"model": "Cosmos3-Edge", "max_tokens": 512, "temperature": .7, "top_p": None,
                    "prompt": "Describe what you see in this image in one sentence.",
                    "image_mime_type": "image/jpeg"}
        payload = benchmark.build_payload(workload, b"test JPEG bytes")
        self.assertEqual(payload["temperature"], .7)
        self.assertEqual(payload["max_tokens"], 512)
        for key in ("top_p", "top_k", "seed"):
            self.assertNotIn(key, payload)

    def test_explicit_top_p_is_forwarded_without_altering_temperature(self):
        workload = {"model": "Cosmos3-Edge", "max_tokens": 512, "temperature": .7, "top_p": .9,
                    "prompt": "Describe this scene.", "image_mime_type": "image/jpeg"}
        payload = benchmark.build_payload(workload, b"test JPEG bytes")
        self.assertEqual(payload["temperature"], .7)
        self.assertEqual(payload["top_p"], .9)

    def request(self, events, status=200):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                self.send_response(status)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                try:
                    for event in events:
                        data = event if isinstance(event, str) else json.dumps(event)
                        wire = ("data: " + data + "\n\n").encode()
                        # Deliberately split both JSON and UTF-8 across writes.
                        for byte in wire:
                            self.wfile.write(bytes([byte]))
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass  # Expected when the client rejects an error event early.

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            return benchmark.stream_request(f"http://127.0.0.1:{server.server_port}/v1/chat/completions", {}, 3)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_role_events_do_not_count_as_tokens_and_usage_drives_throughput(self):
        result = self.request([
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"content": "A scene."}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"completion_tokens": 4}}, "[DONE]",
        ])
        self.assertIsNone(result["error"])
        self.assertEqual(result["output_text"], "A scene.")
        self.assertEqual(result["completion_tokens"], 4)
        self.assertGreater(result["ttft_ms"], 0)
        self.assertAlmostEqual(result["completion_tokens_per_second"], 4000 / result["total_latency_ms"])

    def test_missing_usage_never_estimates_tokens(self):
        result = self.request([{"choices": [{"delta": {"content": "many words with spaces"},
                                               "finish_reason": "stop"}]}, "[DONE]"])
        self.assertIsNone(result["error"])
        self.assertIsNone(result["completion_tokens"])
        self.assertIsNone(result["completion_tokens_per_second"])

    def test_truncated_stream_is_an_error_even_with_finish_reason(self):
        result = self.request([{"choices": [{"delta": {"content": "partial"}, "finish_reason": "stop"}]}])
        self.assertIn("missing [DONE]", result["error"]["message"])
        self.assertFalse(result["stream_done"])

    def test_malformed_stream_and_http_error_are_recorded(self):
        self.assertEqual(self.request(["not-json"])["error"]["type"], "JSONDecodeError")
        self.assertEqual(self.request([], status=503)["http_status"], 503)

    def test_empty_success_is_rejected(self):
        self.assertIn("without visible output", self.request(["[DONE]"])["error"]["message"])

    def test_native_error_or_cancellation_finish_reason_is_not_success(self):
        for reason in ("error", "cancelled", "canceled"):
            with self.subTest(reason=reason):
                result = self.request([{"choices": [{"delta": {"content": "Partial output"},
                                                       "finish_reason": reason}]}, "[DONE]"])
                self.assertEqual(result["finish_reason"], reason)
                self.assertIn("Backend finished with reason", result["error"]["message"])

    def test_whitespace_only_output_is_not_visible_text(self):
        result = self.request([{"choices": [{"delta": {"content": "   "}}]}, "[DONE]"])
        self.assertIn("without visible output", result["error"]["message"])

    def test_done_without_terminal_reason_is_not_success(self):
        result = self.request([{"choices": [{"delta": {"content": "Partial answer"}}]}, "[DONE]"])
        self.assertIn("terminal finish reason", result["error"]["message"])


class SamplingCLITests(unittest.TestCase):
    def run_fixture(self, extra_args):
        # Exercise CLI-to-receipt/payload wiring without networking or inference.
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "fixture.jpg"
            image.write_bytes(b"test-only-image-bytes-not-decoded")
            output = Path(directory) / "results.jsonl"
            args = ["run", "--url", "http://127.0.0.1:8090/v1/chat/completions",
                    "--model", "Cosmos3-Edge", "--image", str(image), "--candidate-id", "offline-test",
                    "--output", str(output), "--requests", "1", "--warmup", "0", *extra_args]
            result = {"error": None, "total_latency_ms": 10, "ttft_ms": 1,
                      "completion_tokens": 3, "output_text": "Fixture response.", "usage": {"completion_tokens": 3}}
            with mock.patch.object(benchmark, "stream_request", return_value=result) as request, \
                    mock.patch("sys.stdout", new_callable=io.StringIO), \
                    mock.patch("sys.stderr", new_callable=io.StringIO):
                self.assertEqual(benchmark.main(args), 0)
            config = json.loads(output.read_text().splitlines()[0])
            return config, request.call_args.args[1]

    def test_historical_defaults_preserve_payload_and_workload_fingerprint(self):
        config, payload = self.run_fixture([])
        workload = config["workload"]
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["top_p"], 1)
        self.assertEqual(payload["max_tokens"], 64)
        historical = {"model": "Cosmos3-Edge",
                      "prompt": "Describe the visible scene in one concise sentence. Focus on objects and actions.",
                      "max_tokens": 64, "temperature": 0, "top_p": 1, "concurrency": 1,
                      "image_sha256": hashlib.sha256(b"test-only-image-bytes-not-decoded").hexdigest(),
                      "image_mime_type": "image/jpeg", "image_bytes": len(b"test-only-image-bytes-not-decoded"),
                      "stream_options": {"include_usage": True}}
        self.assertEqual(workload, historical)
        self.assertEqual(config["workload_fingerprint"],
                         hashlib.sha256(json.dumps(historical, sort_keys=True).encode()).hexdigest())
        explicit, _ = self.run_fixture(["--temperature", "0.0", "--top-p", "1.0"])
        self.assertEqual(explicit["workload_fingerprint"], config["workload_fingerprint"])

    def test_sampling_values_and_omission_are_part_of_fingerprint(self):
        cases = [[], ["--temperature", ".7"], ["--temperature", ".7", "--top-p", ".9"],
                 ["--temperature", ".7", "--top-p", "omit"]]
        outputs = [self.run_fixture(args) for args in cases]
        self.assertEqual(len({config["workload_fingerprint"] for config, _ in outputs}), 4)
        config, payload = outputs[-1]
        self.assertEqual(config["workload"]["temperature"], .7)
        self.assertIsNone(config["workload"]["top_p"])
        self.assertNotIn("top_p", payload)
        self.assertEqual(outputs[2][1]["top_p"], .9)

    def test_nonfinite_out_of_range_and_invalid_values_fail_before_image_read_or_request(self):
        invalid = {"temperature": ["nan", "inf", "-inf", "-.1", "2.1", "omit"],
                   "top-p": ["nan", "inf", "-inf", "0", "-.1", "1.01", "not-a-number"]}
        for flag, values in invalid.items():
            for value in values:
                with self.subTest(flag=flag, value=value), \
                        mock.patch.object(benchmark, "stream_request") as request, \
                        mock.patch.object(Path, "read_bytes", side_effect=AssertionError("Read image before validation")), \
                        mock.patch("sys.stderr", new_callable=io.StringIO), \
                        self.assertRaises(SystemExit) as raised:
                    benchmark.main(["run", "--url", "http://127.0.0.1:8090/v1/chat/completions",
                                    "--model", "Cosmos3-Edge", "--image", "unused.jpg",
                                    "--candidate-id", "offline-test", "--output", "unused.jsonl",
                                    f"--{flag}={value}"])
                self.assertEqual(raised.exception.code, 2)
                request.assert_not_called()

    def test_programmatic_validation_rejects_boolean_and_accepts_boundaries(self):
        for temperature, top_p in [(True, 1), (0, False), (None, 1), ("0.7", 1)]:
            with self.subTest(temperature=temperature, top_p=top_p), self.assertRaises(ValueError):
                benchmark.validate_sampling(temperature, top_p)
        self.assertEqual(benchmark.validate_sampling(0, 1), (0, 1))
        self.assertEqual(benchmark.validate_sampling(2, .001), (2, .001))
        self.assertEqual(benchmark.validate_sampling(.7, None), (.7, None))


class MemoryTests(unittest.TestCase):
    def test_separate_overlapping_counters_and_swap_activity(self):
        samples = [
            {"kind": "proc", "system_ram_unavailable_bytes": 400, "process_rss_bytes": 100,
             "system_swap_occupied_bytes": 90, "swap_in_pages": 7, "swap_out_pages": 8, "system_oom_kills": 0},
            {"kind": "proc", "system_ram_unavailable_bytes": 450, "process_rss_bytes": 110,
             "system_swap_occupied_bytes": 90, "swap_in_pages": 7, "swap_out_pages": 9, "system_oom_kills": 0},
        ]
        result = benchmark.summarize_memory(samples)
        self.assertEqual(result["local_system_ram_unavailable_peak_bytes"], 450)
        self.assertEqual(result["local_process_rss_peak_bytes"], 110)
        self.assertTrue(result["swap_activity_detected"])
        self.assertFalse(result["oom_detected"])
        self.assertIsNone(result["cuda_allocator_allocated_bytes"])

    def test_unknown_counters_and_counter_reset_are_not_clean_evidence(self):
        self.assertIsNone(benchmark.summarize_memory([])["swap_activity_detected"])
        result = benchmark.summarize_memory([
            {"kind": "proc", "swap_in_pages": 9, "swap_out_pages": 10},
            {"kind": "proc", "swap_in_pages": 0, "swap_out_pages": 0},
        ])
        self.assertIsNone(result["swap_activity_detected"])

    def test_tegrastats_parsing_preserves_separate_ram_and_swap(self):
        result = benchmark.parse_tegrastats("RAM 3120/7620MB (lfb 4x4MB) SWAP 9/4096MB GR3D_FREQ 99%")
        self.assertEqual(result["tegrastats_ram_used_bytes"], 3120 * 1024 ** 2)
        self.assertEqual(result["tegrastats_swap_used_bytes"], 9 * 1024 ** 2)

    def test_warmup_latency_excluded_but_warmup_oom_rejects_run(self):
        records = [
            {"phase": "warmup", "error": None, "total_latency_ms": 999, "ttft_ms": 900,
             "memory": {"oom_detected": True, "swap_activity_detected": False}},
            {"phase": "measured", "error": None, "total_latency_ms": 10, "ttft_ms": 1,
             "memory": {"oom_detected": False, "swap_activity_detected": False,
                        "local_system_ram_unavailable_peak_bytes": 100}},
        ]
        args = argparse.Namespace(candidate_id="test", memory_metric="local_system_ram_unavailable_peak_bytes",
                                  quality="unknown", quality_note=None)
        result = benchmark.summarize_run(records, {"workload_fingerprint": "test"}, args)
        self.assertEqual(result["p50_latency_ms"], 10)
        self.assertEqual(result["measured_requests"], 1)
        self.assertEqual(result["memory_bytes"], 100)
        self.assertTrue(result["oom_detected"])
        self.assertIsNone(result["quality_pass"])


def candidate(name="base", **changes):
    result = {"candidate_id": name, "quality_pass": True, "error_count": 0,
              "oom_detected": False, "swap_activity_detected": False,
              "workload_fingerprint": "fixed-workload", "measured_requests": 30,
              "p50_latency_ms": 100., "p95_latency_ms": 120.,
              "memory_metric": "local_system_ram_unavailable_peak_bytes", "memory_bytes": 1000.}
    result.update(changes)
    return result


class StoppingTests(unittest.TestCase):
    def test_accepts_exact_five_percent_gain_and_regression_boundary(self):
        result = benchmark.evaluate_candidates([candidate(), candidate("faster", p50_latency_ms=95,
                                                p95_latency_ms=126, memory_bytes=1050)])
        self.assertEqual(result["selected_candidate"]["candidate_id"], "faster")

    def test_memory_gain_can_accept_small_latency_cost(self):
        result = benchmark.evaluate_candidates([candidate(), candidate("smaller", p50_latency_ms=104,
                                                memory_bytes=940)])
        self.assertEqual(result["selected_candidate"]["candidate_id"], "smaller")

    def test_p95_regression_rejects_large_p50_improvement(self):
        result = benchmark.evaluate_candidates([candidate(), candidate("spiky", p50_latency_ms=80,
                                                p95_latency_ms=127)])
        self.assertEqual(result["decisions"][1]["reason"], "regression_exceeds_5_percent")

    def test_hard_rejects_errors_oom_swap_and_unknown_safety(self):
        for field, value in [("error_count", 1), ("warmup_error_count", 1), ("oom_detected", True),
                             ("swap_activity_detected", True), ("oom_detected", None), ("quality_pass", False)]:
            with self.subTest(field=field, value=value):
                result = benchmark.evaluate_candidates([candidate(), candidate("bad", p50_latency_ms=50,
                                                       **{field: value})])
                self.assertFalse(result["decisions"][1]["accepted"])

    def test_workload_and_memory_definition_must_match(self):
        for changes, reason in [({"workload_fingerprint": "other"}, "workload_mismatch"),
                                ({"memory_metric": "rss"}, "memory_metric_mismatch")]:
            result = benchmark.evaluate_candidates([candidate(), candidate("different", p50_latency_ms=50,
                                                   **changes)])
            self.assertEqual(result["decisions"][1]["reason"], reason)

    def test_stop_after_three_non_improvements_ignores_later_gain(self):
        result = benchmark.evaluate_candidates([candidate(), candidate("n1"), candidate("n2"),
                                                candidate("n3"), candidate("late", p50_latency_ms=50)])
        self.assertEqual(result["stop_reason"], "consecutive_non_improvement_limit_reached")
        self.assertEqual(result["candidates_evaluated"], 4)
        self.assertEqual(result["candidates_ignored_after_stop"], 1)

    def test_acceptance_resets_non_improvement_count(self):
        result = benchmark.evaluate_candidates([candidate(), candidate("n1"), candidate("n2"),
                                                candidate("gain", p50_latency_ms=90), candidate("n3", p50_latency_ms=90)])
        self.assertEqual(result["decisions"][-1]["consecutive_non_improvements"], 1)

    def test_twelve_candidate_cap_and_reject_looser_limits(self):
        candidates = [candidate(str(i), p50_latency_ms=100 * .9 ** i) for i in range(15)]
        result = benchmark.evaluate_candidates(candidates)
        self.assertEqual(result["candidates_evaluated"], 12)
        self.assertEqual(result["stop_reason"], "candidate_budget_reached")
        with self.assertRaises(ValueError):
            benchmark.evaluate_candidates(candidates, maximum_candidates=13)

    def test_nonfinite_metrics_or_too_few_samples_are_rejected(self):
        for changes in ({"memory_bytes": float("nan")}, {"p50_latency_ms": -1}, {"measured_requests": 29}):
            result = benchmark.evaluate_candidates([candidate(**changes)])
            self.assertIsNone(result["selected_candidate"])


if __name__ == "__main__":
    unittest.main()
