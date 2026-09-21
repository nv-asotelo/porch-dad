"""Synthetic statistical/contract checks, not Orin performance measurements."""

import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SPEC = importlib.util.spec_from_file_location(
    "compare_server_timings", Path(__file__).parents[1] / "scripts" / "compare_server_timings.py")
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


def rows(backend="trt-320", fixed=100, slope=5, budget=320, lengths=(8, 16, 32, 64), count=20):
    result = []
    for i in range(count):
        tokens = lengths[i % len(lengths)]
        result.append({
            "kind": "server_request", "schema_version": 1, "backend_id": backend,
            "request_id": str(i), "status": "completed", "warmup": False,
            "timing_source": "server_monotonic", "timing_boundary": "native_inference",
            "server_elapsed_ms": fixed + slope * tokens, "completion_tokens": tokens,
            "cache_state": "hit", "controls": {
                "image_sha256": hashlib.sha256(b"synthetic-image").hexdigest(),
                "prompt_sha256": hashlib.sha256(b"synthetic-prompt").hexdigest(),
                "max_image_tokens_per_image": budget, "temperature": .7, "top_p": .95,
                "concurrency": 1, "clock_policy": "static", "encoder_cache_bytes": 268435456,
            },
        })
    return result


def report(data, **kwargs):
    return compare.compare_records(data, bootstrap_repeats=100, **kwargs)


class ServerTimingTests(unittest.TestCase):
    def test_known_line_uses_actual_tokens_not_output_cap(self):
        data = rows()
        for row in data:
            row["max_tokens"] = 512
        result = report(data)
        fit = result["strata"][0]["fit"]
        self.assertAlmostEqual(fit["fixed_ms"], 100)
        self.assertAlmostEqual(fit["marginal_ms_per_token"], 5)
        self.assertAlmostEqual(fit["r_squared"], 1)
        for bound in fit["fixed_ms_ci95"]:
            self.assertAlmostEqual(bound, 100)
        self.assertEqual(result["strata"][0]["actual_output_token_range"], [8, 64])

    def test_bootstrap_is_deterministic_and_noisy_intervals_have_width(self):
        data = rows(count=40)
        for i, row in enumerate(data):
            row["server_elapsed_ms"] += ((i * 13) % 19) - 9
        first = report(data, seed=99)
        self.assertEqual(first, report(data, seed=99))
        fit = first["strata"][0]["fit"]
        self.assertLess(fit["fixed_ms_ci95"][0], fit["fixed_ms_ci95"][1])
        self.assertLess(fit["marginal_ms_per_token_ci95"][0], fit["marginal_ms_per_token_ci95"][1])

    def test_matched_backend_differences_and_prediction_overlap(self):
        result = report(rows("left", 100, 5) + rows("right", 90, 4, lengths=(16, 32, 64, 128)))
        pair = result["comparisons"][0]
        self.assertEqual(pair["overlapping_actual_output_token_range"], [16, 64])
        self.assertAlmostEqual(pair["right_minus_left"]["fixed_ms"], -10)
        self.assertAlmostEqual(pair["right_minus_left"]["marginal_ms_per_token"], -1)
        self.assertEqual(pair["predictions_within_observed_overlap"][0]["right_minus_left_ms"], -26)

    def test_changed_budget_requires_explicit_permission_and_is_disclosed(self):
        data = rows("left", budget=512) + rows("right", budget=320)
        strict = report(data)
        self.assertEqual(strict["comparisons"], [])
        self.assertEqual(strict["noncomparisons"][0]["reasons"], ["unmatched_controls"])
        changed = report(data, vary_controls=["max_image_tokens_per_image"])
        pair = changed["comparisons"][0]
        self.assertEqual(pair["comparison_scope"], "configuration change")
        self.assertEqual(pair["differing_controls"]["max_image_tokens_per_image"], {"left": 512, "right": 320})

    def test_prompt_and_cache_states_are_never_pooled_or_waived(self):
        data = rows("left") + rows("right")
        for row in data[20:]:
            row["controls"]["prompt_sha256"] = "1" * 64
            row["cache_state"] = "miss"
        result = report(data)
        self.assertEqual(result["fitted_strata"], 2)
        self.assertEqual(result["comparisons"], [])
        self.assertEqual(result["noncomparisons"][0]["reasons"], ["unmatched_controls", "unmatched_cache_state"])
        with self.assertRaisesRegex(ValueError, "input hashes"):
            report(data, vary_controls=["prompt_sha256"])

    def test_same_loaded_backend_can_compare_explicit_runtime_budget_change(self):
        data = rows("same-native-backend", budget=512)
        second = rows("same-native-backend", budget=320)
        for row in second:
            row["request_id"] = "other-" + row["request_id"]
        data += second
        self.assertEqual(report(data)["comparisons"], [])
        result = report(data, vary_controls=["max_image_tokens_per_image"])
        pair = result["comparisons"][0]
        self.assertEqual(pair["left_backend"], "same-native-backend")
        self.assertEqual(pair["right_backend"], "same-native-backend")
        self.assertEqual(pair["comparison_scope"], "configuration change")
        self.assertNotEqual(pair["left_stratum_id"], pair["right_stratum_id"])

    def test_repeated_single_length_cannot_identify_slope(self):
        result = report(rows(lengths=(21,)))
        self.assertEqual(result["fitted_strata"], 0)
        self.assertIsNone(result["strata"][0]["fit"])
        self.assertIn("fewer_than_three_distinct_actual_output_lengths", result["strata"][0]["reasons"])

    def test_client_jpeg_network_queue_timings_cannot_be_relabelled(self):
        data = [{"kind": "request", "total_latency_ms": 1234, "completion_tokens": 20}]
        for boundary in ("body_received_to_generation_complete", "http_round_trip", "client_capture"):
            item = rows(count=1)[0]
            item["timing_boundary"] = boundary
            data.append(item)
        result = report(data)
        self.assertEqual(result["accepted_measured_requests"], 0)
        self.assertEqual(result["excluded_records"]["client_timing_is_not_server_evidence"], 1)
        self.assertEqual(result["excluded_records"]["unsupported_timing_boundary"], 3)

    def test_warmup_failures_missing_usage_and_unknown_cache_are_counted(self):
        data = rows(count=6)
        data[0]["warmup"] = True
        data[1]["status"] = "failed"
        data[2]["completion_tokens"] = None
        data[3]["cache_state"] = "unknown"
        data[4].pop("warmup")
        data[5]["error"] = "cancelled"
        result = report(data)
        self.assertEqual(result["accepted_measured_requests"], 0)
        self.assertEqual(result["excluded_records"], {
            "warmup": 1, "failed_or_incomplete_request": 2, "missing_actual_completion_tokens": 1,
            "unknown_cache_state": 1, "unknown_warmup_state": 1})
        self.assertEqual(result["backend_record_counts"]["trt-320"]["records"], 6)
        self.assertEqual(result["backend_record_counts"]["trt-320"]["failed_or_incomplete_request"], 2)

    def test_boolean_nonfinite_and_missing_controls_are_rejected(self):
        for key, value in [("server_elapsed_ms", True), ("server_elapsed_ms", float("nan")),
                           ("server_elapsed_ms", float("inf")), ("completion_tokens", True),
                           ("completion_tokens", 10.5), ("completion_tokens", 0)]:
            with self.subTest(key=key, value=value):
                data = rows(count=1)
                data[0][key] = value
                self.assertEqual(report(data)["accepted_measured_requests"], 0)
        for changes in ({"concurrency": 2}, {"encoder_cache_bytes": True}, {"top_p": None},
                        {"temperature": float("nan")}, {"image_sha256": "not-a-sha"}):
            data = rows(count=1)
            data[0]["controls"].update(changes)
            self.assertEqual(report(data)["accepted_measured_requests"], 0)
        data = rows(count=1)
        data[0]["controls"].pop("clock_policy")
        self.assertEqual(report(data)["excluded_records"], {"missing_controls": 1})

    def test_overlapping_files_cannot_double_count_request_ids(self):
        data = rows()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            report(data + copy.deepcopy(data))

    def test_unmatched_additional_control_or_disjoint_ranges_block_comparison(self):
        data = rows("left") + rows("right", lengths=(80, 96, 128, 144))
        data[0]["controls"]["native_revision"] = "extra-separate-stratum"
        result = report(data)
        self.assertEqual(result["comparisons"], [])
        self.assertIn("no_overlapping_actual_output_token_range", result["noncomparisons"][0]["reasons"])

    def test_negative_intercept_is_reported_not_clipped(self):
        result = report(rows(fixed=-10, slope=5))
        self.assertEqual(result["strata"][0]["fit"]["fixed_ms"], -10)
        self.assertTrue(any("negative intercept" in item for item in result["strata"][0]["warnings"]))

    def test_cli_fingerprints_logs_and_protects_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.jsonl"
            output = Path(directory) / "report.json"
            path.write_text("\n".join(json.dumps(row) for row in rows()) + "\n")
            self.assertEqual(compare.main([str(path), "--output", str(output), "--bootstrap", "100"]), 0)
            saved = output.read_bytes()
            result = json.loads(saved)
            self.assertEqual(result["sources"][0]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            with mock.patch("sys.stderr", new_callable=io.StringIO):
                self.assertEqual(compare.main([str(path), "--output", str(output), "--bootstrap", "100"]), 2)
            self.assertEqual(output.read_bytes(), saved)

    def test_client_only_log_yields_unavailable_report_and_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "client.jsonl"
            path.write_text(json.dumps({"kind": "request", "total_latency_ms": 500}) + "\n")
            with mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
                self.assertEqual(compare.main([str(path), "--bootstrap", "100"]), 2)
            self.assertEqual(json.loads(stdout.getvalue())["fitted_strata"], 0)


if __name__ == "__main__":
    unittest.main()
