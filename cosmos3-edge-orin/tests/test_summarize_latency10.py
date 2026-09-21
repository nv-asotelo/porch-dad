"""Frozen-policy tests with invented transport fixtures, never GPU measurements."""

import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("summarize_latency10", ROOT / "scripts/summarize_latency10.py")
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.plan = json.loads((ROOT / "results/latency10/plan.json").read_text())
        self.manifest = json.loads((ROOT / "benchmarks/live-vlm-1280/manifest.json").read_text())
        self.write_group("baseline", [100, 200, 400])
        self.write_group("candidate", [90, 180, 360])

    def write_group(self, prefix, latencies):
        backend = {key: "test-fixture" for key in summary.INVARIANTS}
        backend.update(encoder_embedding_cache_budget_bytes=0, text_context_reuse=False,
                       sampling={"temperature": .7, "top_p": .9, "top_k": 50,
                                 "request_top_p": "omitted", "request_top_k": "omitted",
                                 "native_philox_seed": 42, "native_philox_offset": 0})
        for fixture, latency in zip(self.manifest["fixtures"], latencies):
            workload = {"model": "test-only-model", "prompt": self.plan["prompt"], "max_tokens": 512,
                        "temperature": .7, "top_p": None, "concurrency": 1,
                        "image_sha256": fixture["sha256"], "image_bytes": fixture["size_bytes"],
                        "image_mime_type": "image/jpeg", "stream_options": {"include_usage": True}}
            fingerprint = summary.sha256(json.dumps(workload, sort_keys=True).encode())
            identity = {"candidate_id": prefix, "run_id": prefix + fixture["file"],
                        "workload_fingerprint": fingerprint}
            config = {"kind": "run_config", **identity, "workload": workload, "requests": 30, "warmup": 5,
                      "backend_config": backend, "endpoint": "http://127.0.0.1:8090/v1/chat/completions",
                      "memory_sampling": {"enabled": True, "host": "test-fixture"}}
            sample = {"kind": "proc", "system_ram_total_bytes": 8 * 1024**3,
                      "system_ram_unavailable_bytes": 7 * 1024**3, "process_rss_bytes": 6 * 1024**3,
                      "swap_in_pages": 12, "swap_out_pages": 13, "system_oom_kills": 0}
            records = [config]
            for index in range(35):
                records.append({"kind": "request", **identity, "index": index,
                                "phase": "warmup" if index < 5 else "measured", "error": None,
                                "http_status": 200, "stream_done": True, "finish_reason": "stop",
                                "output_text": "Test fixture answer.", "completion_tokens": 5,
                                "usage": {"completion_tokens": 5}, "total_latency_ms": latency,
                                "ttft_ms": 20, "memory": {"sampler_errors": [], "oom_detected": False,
                                "swap_activity_detected": False, "samples": [dict(sample), dict(sample)]}})
            records.append({"kind": "summary", **identity, "measured_requests": 30, "warmup_requests": 5})
            self.save(prefix, fixture["file"], records)

    def path(self, prefix, fixture=None):
        fixture = fixture or self.manifest["fixtures"][0]["file"]
        return self.directory / (prefix + "-" + Path(fixture).stem + ".jsonl")

    def save(self, prefix, fixture, records):
        self.path(prefix, fixture).write_text("".join(json.dumps(record) + "\n" for record in records))

    def change(self, edit, prefix="candidate", fixture=None):
        fixture = fixture or self.manifest["fixtures"][0]["file"]
        records = [json.loads(line) for line in self.path(prefix, fixture).read_text().splitlines()]
        edit(records)
        self.save(prefix, fixture, records)

    def compare(self):
        before = summary.read_group(self.directory, "baseline", self.manifest, self.plan)
        after = summary.read_group(self.directory, "candidate", self.manifest, self.plan)
        return summary.compare(before, after), before, after

    def test_exact_ten_percent_geomean_gain_accepts_and_memory_is_not_added(self):
        result, before, after = self.compare()
        self.assertTrue(result["accepted"])
        self.assertAlmostEqual(before["geomean_p50_complete_latency_ms"], 200)
        self.assertAlmostEqual(after["geomean_p50_complete_latency_ms"], 180)
        self.assertAlmostEqual(result["latency_reduction_percent"], 10)
        self.assertEqual(after["peak_shared_ram_unavailable_bytes"], 7 * 1024**3)
        self.assertEqual(after["peak_process_rss_bytes"], 6 * 1024**3)
        self.assertEqual(after["minimum_available_shared_ram_bytes"], 1024**3)

    def test_below_ten_percent_stops_even_when_faster(self):
        self.write_group("candidate", [90.01, 180.02, 360.04])
        result, _, _ = self.compare()
        self.assertFalse(result["accepted"])
        self.assertEqual(result["decision"], "stop_and_restore_last_accepted")
        self.assertIn("geomean_complete_latency_gain_below_10_percent", result["rejection_reasons"])

    def test_equal_image_weighting_cannot_hide_a_regressed_fixture(self):
        self.write_group("candidate", [105.01, 100, 200])
        result, _, _ = self.compare()
        self.assertLess(result["candidate_to_baseline_ratio"], .9)
        self.assertTrue(any("regression_exceeds_5_percent" in reason for reason in result["rejection_reasons"]))

    def test_p95_regression_rejects_unchanged_median_gain(self):
        def edit(records):
            for record in records[-4:-1]:
                record["total_latency_ms"] = 120
        self.change(edit)
        result, _, _ = self.compare()
        self.assertFalse(result["accepted"])
        self.assertGreater(result["per_fixture"][0]["ratios"]["p95_latency_ms"], 1.05)

    def test_shorter_answer_is_not_an_optimization(self):
        def edit(records):
            for record in records[1:-1]:
                record.update(output_text="Shorter.", completion_tokens=2, usage={"completion_tokens": 2})
        self.change(edit)
        result, _, _ = self.compare()
        self.assertFalse(result["accepted"])
        self.assertTrue(any("not_equivalent" in reason for reason in result["rejection_reasons"]))

    def test_single_different_measured_answer_rejects_repeatability(self):
        self.change(lambda records: records[7].update(output_text="Different answer."))
        result, _, after = self.compare()
        self.assertFalse(result["accepted"])
        self.assertEqual(after["per_fixture"][0]["unique_measured_answers"], 2)

    def test_real_token_usage_required_and_length_finish_fails(self):
        for change in ({"usage": None}, {"finish_reason": "length"}, {"completion_tokens": True}):
            with self.subTest(change=change):
                self.write_group("candidate", [90, 180, 360])
                self.change(lambda records: records[7].update(change))
                self.assertFalse(self.compare()[0]["accepted"])

    def test_invented_workload_fingerprint_is_detected(self):
        self.change(lambda records: records[0]["workload"].update(temperature=0))
        with self.assertRaisesRegex(ValueError, "invalid workload fingerprint"):
            self.compare()

    def test_valid_but_different_workload_fingerprint_rejects(self):
        def edit(records):
            records[0]["workload"]["model"] = "different-model"
            fingerprint = summary.sha256(json.dumps(records[0]["workload"], sort_keys=True).encode())
            for record in records:
                record["workload_fingerprint"] = fingerprint
        self.change(edit)
        result, _, _ = self.compare()
        self.assertTrue(any("workload_fingerprint_mismatch" in reason for reason in result["rejection_reasons"]))

    def test_warmup_errors_and_between_request_swap_changes_fail(self):
        self.change(lambda records: records[1].update(error={"type": "test-error"}))
        self.assertFalse(self.compare()[0]["accepted"])
        self.write_group("candidate", [90, 180, 360])
        def edit(records):
            # Each request is internally stable and claims false; cross-request check catches this.
            for record in records[15:-1]:
                for sample in record["memory"]["samples"]:
                    sample["swap_in_pages"] += 1
        self.change(edit)
        result, _, _ = self.compare()
        self.assertTrue(any("swap_in_pages_changed_or_reset" in reason for reason in result["rejection_reasons"]))

    def test_available_ram_floor_is_inclusive_and_warmups_count(self):
        def edit(records, offset):
            sample = records[1]["memory"]["samples"][0]
            sample["system_ram_unavailable_bytes"] = sample["system_ram_total_bytes"] - summary.MEMORY_FLOOR + offset
        self.change(lambda records: edit(records, 0))
        self.assertTrue(self.compare()[0]["accepted"])
        self.change(lambda records: edit(records, 1))
        self.assertFalse(self.compare()[0]["accepted"])

    def test_incomplete_or_duplicate_run_set_is_invalid(self):
        self.change(lambda records: records.pop(8))
        with self.assertRaisesRegex(ValueError, "exactly 35"):
            self.compare()
        self.write_group("candidate", [90, 180, 360])
        (self.directory / "candidate-duplicate.jsonl").write_bytes(self.path("candidate").read_bytes())
        with self.assertRaisesRegex(ValueError, "exactly 3 files"):
            self.compare()

    def test_changed_clock_policy_rejects_even_with_fast_timing(self):
        for fixture in self.manifest["fixtures"]:
            self.change(lambda records: records[0]["backend_config"].update(clock_policy="different"),
                        fixture=fixture["file"])
        result, _, _ = self.compare()
        self.assertTrue(any("clock_policy" in reason for reason in result["rejection_reasons"]))

    def test_cli_receipt_is_exclusive_and_reports_real_evidence_hashes(self):
        output = self.directory / "comparison.json"
        args = ["--baseline-dir", str(self.directory), "--candidate-dir", str(self.directory),
                "--output", str(output)]
        with mock.patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(summary.main(args), 0)
        data = json.loads(output.read_text())
        self.assertTrue(data["comparison"]["accepted"])
        self.assertEqual(data["baseline"]["per_fixture"][0]["source_sha256"],
                         summary.sha256(self.path("baseline").read_bytes()))
        original = output.read_bytes()
        with mock.patch("sys.stderr", new_callable=io.StringIO):
            self.assertEqual(summary.main(args), 2)
        self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
