"""New MLP-goal policy checks; all timings and RAM here are offline fixtures."""

import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import summarize_mlp_goal as mlp

# Reuse the existing transport-fixture writer without inheriting/re-running its tests.
SPEC = importlib.util.spec_from_file_location("latency_fixture_helpers", Path(__file__).with_name("test_summarize_latency10.py"))
fixtures = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixtures)


class MlpGoalTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ComparisonTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.directory = self.fixture.directory
        self.manifest = self.fixture.manifest
        self.plan = {**self.fixture.plan, "baseline_candidate": "baseline",
                     "mutable_backend_fields": sorted(mlp.MUTABLE_FIELDS)}
        self.plan_path = self.directory / "mlp-plan.json"
        self.plan_path.write_text(json.dumps(self.plan))
        for prefix in ("baseline", "candidate"):
            self.normalize(prefix)

    def edit_all(self, prefix, edit):
        for fixture in self.manifest["fixtures"]:
            self.fixture.change(edit, prefix=prefix, fixture=fixture["file"])

    def normalize(self, prefix):
        def edit(records):
            records[0]["backend_config"].update(
                weight_sha256=mlp.MLP_WEIGHT_SHA256, quantization_scope="mlp-only", quantized_linear_count=56,
                max_input_len=1024, max_kv_cache_capacity=2048, max_batch_size=1,
                max_image_tokens=1024, max_image_tokens_per_image=512, int4_gemm_plugin_version=1,
                native_extension_sha256="a" * 64, native_extension_current_sha256="a" * 64)
            # This historical-style quality field must not force the new goal back to FP16.
            records[-1].update(quality_pass=False, quality_note="Known yellow-to-orange baseline limitation")
            for request in records[1:-1]:
                for sample in request["memory"]["samples"]:
                    sample["system_ram_unavailable_bytes"] = 4_000_000_000
                    sample["process_rss_bytes"] = 3_000_000_000
        self.edit_all(prefix, edit)

    def read(self, prefix):
        return mlp.read_mlp_group(self.directory, prefix, self.manifest, self.plan)

    def compare(self, initial=None):
        return mlp.compare(self.read("baseline"), self.read("candidate"),
                           mlp.validate_plan(self.plan, self.manifest), initial)

    def test_exact_ten_percent_accepts_mlp_with_disclosed_old_quality_failure(self):
        result = self.compare()
        self.assertTrue(result["accepted"])
        self.assertEqual(result["selected_candidate_id"], "candidate")
        self.assertAlmostEqual(result["latency_reduction_percent"], 10)

    def test_explicit_runtime_changes_are_allowed_and_recorded(self):
        self.edit_all("candidate", lambda records: records[0]["backend_config"].update(
            max_kv_cache_capacity=1664, max_image_tokens=512, int4_gemm_plugin_version=2,
            native_extension_sha256="b" * 64, native_extension_current_sha256="b" * 64))
        result = self.compare()
        self.assertTrue(result["accepted"])
        self.assertEqual(result["per_fixture"][0]["backend_changes"]["max_image_tokens"],
                         {"before": 1024, "after": 512})

    def test_undeclared_runtime_change_rejects_and_cannot_modify_frozen_fields(self):
        self.plan["mutable_backend_fields"].remove("max_kv_cache_capacity")
        self.edit_all("candidate", lambda records: records[0]["backend_config"].update(max_kv_cache_capacity=1664))
        self.assertFalse(self.compare()["accepted"])
        for field in ("clock_policy", "power_mode", "max_input_len", "max_image_tokens_per_image", "weight_sha256"):
            with self.subTest(field=field):
                bad = {**self.plan, "mutable_backend_fields": [field]}
                with self.assertRaisesRegex(ValueError, "unsupported"):
                    mlp.validate_plan(bad, self.manifest)

    def test_input_per_image_scope_weight_and_binary_guards(self):
        for changes in ({"max_input_len": 512}, {"max_image_tokens_per_image": 256},
                        {"quantization_scope": "all-linears"}, {"quantized_linear_count": 169},
                        {"weight_sha256": "b" * 64}, {"native_extension_current_sha256": "missing"},
                        {"native_extension_current_sha256": "b" * 64}, {"max_kv_cache_capacity": 1536}):
            with self.subTest(changes=changes):
                self.normalize("candidate")
                self.edit_all("candidate", lambda records: records[0]["backend_config"].update(changes))
                with self.assertRaises(ValueError):
                    self.read("candidate")
        self.normalize("candidate")
        self.edit_all("candidate", lambda records: records[0]["backend_config"].update(max_kv_cache_capacity=1537))
        self.assertEqual(self.read("candidate")["per_fixture"][0]["backend_config"]["max_kv_cache_capacity"], 1537)

    def test_profile_cannot_change_between_fixture_runs(self):
        self.fixture.change(lambda records: records[0]["backend_config"].update(max_image_tokens=512))
        with self.assertRaisesRegex(ValueError, "changes between fixtures"):
            self.read("candidate")

    def test_clock_change_rejects_without_reverting_to_fp16(self):
        self.edit_all("candidate", lambda records: records[0]["backend_config"].update(clock_policy="fixed"))
        result = self.compare()
        self.assertFalse(result["accepted"])
        self.assertEqual(result["decision"], "stop_and_keep_last_accepted_mlp")
        self.assertEqual(result["selected_candidate_id"], "baseline")

    def test_one_percent_peak_ram_growth_boundary(self):
        def set_ram(records, value):
            for request in records[1:-1]:
                for sample in request["memory"]["samples"]:
                    sample["system_ram_unavailable_bytes"] = value
        self.edit_all("candidate", lambda records: set_ram(records, 4_040_000_000))
        self.assertTrue(self.compare()["accepted"])
        self.edit_all("candidate", lambda records: set_ram(records, 4_040_000_001))
        result = self.compare()
        self.assertFalse(result["accepted"])
        self.assertIn("peak_shared_ram_regression_exceeds_1_percent", result["rejection_reasons"])

    def test_output_change_shortening_and_unknown_real_tokens_reject(self):
        for changes in ({"output_text": "Different answer."}, {"completion_tokens": 2, "usage": {"completion_tokens": 2}},
                        {"usage": None}, {"finish_reason": "length"}):
            with self.subTest(changes=changes):
                self.fixture.write_group("candidate", [90, 180, 360])
                self.normalize("candidate")
                self.fixture.change(lambda records: records[7].update(changes))
                self.assertFalse(self.compare()["accepted"])

    def test_warmup_oom_swap_and_memory_floor_fail(self):
        def low_ram(records):
            sample = records[1]["memory"]["samples"][0]
            sample["system_ram_unavailable_bytes"] = sample["system_ram_total_bytes"] - mlp.core.MEMORY_FLOOR + 1
        for edit in (lambda records: records[1]["memory"].update(oom_detected=True),
                     lambda records: records[1]["memory"].update(swap_activity_detected=True), low_ram):
            self.fixture.write_group("candidate", [90, 180, 360])
            self.normalize("candidate")
            self.fixture.change(edit)
            self.assertFalse(self.compare()["accepted"])

    def test_memory_only_gain_does_not_replace_ten_percent_latency_gate(self):
        before, after = self.read("baseline"), self.read("candidate")
        after["geomean_p50_complete_latency_ms"] = before["geomean_p50_complete_latency_ms"] * .9001
        after["peak_shared_ram_unavailable_bytes"] = 2_000_000_000
        result = mlp.compare(before, after, set(mlp.MUTABLE_FIELDS))
        self.assertFalse(result["accepted"])
        self.assertEqual(result["memory_changes"]["candidate_vs_incumbent"]["shared_ram_unavailable"]["decrease_bytes"], 2_000_000_000)
        self.assertEqual(result["memory_changes"]["selected_vs_incumbent"]["shared_ram_unavailable"]["decrease_bytes"], 0)

    def test_p95_regression_and_invalid_baseline_cannot_be_accepted(self):
        before, after = self.read("baseline"), self.read("candidate")
        after["per_fixture"][0]["p95_latency_ms"] = before["per_fixture"][0]["p95_latency_ms"] * 1.0501
        result = mlp.compare(before, after, set(mlp.MUTABLE_FIELDS))
        self.assertTrue(any("regression_exceeds_5_percent" in reason for reason in result["rejection_reasons"]))
        before["failures"].append("swap activity")
        result = mlp.compare(before, after, set(mlp.MUTABLE_FIELDS))
        self.assertEqual(result["decision"], "invalid_baseline_do_not_accept")
        self.assertIsNone(result["selected_candidate_id"])

    def test_incumbent_and_initial_memory_deltas_keep_counters_separate(self):
        before, after = self.read("baseline"), self.read("candidate")
        initial = copy.deepcopy(before)
        initial["peak_shared_ram_unavailable_bytes"] = 5_000_000_000
        after["peak_shared_ram_unavailable_bytes"] = 3_800_000_000
        after["peak_process_rss_bytes"] = 2_900_000_000
        result = mlp.compare(before, after, set(mlp.MUTABLE_FIELDS), initial)
        self.assertTrue(result["accepted"])
        memory = result["memory_changes"]
        self.assertEqual(memory["candidate_vs_incumbent"]["shared_ram_unavailable"]["decrease_bytes"], 200_000_000)
        self.assertEqual(memory["selected_vs_initial_mlp"]["shared_ram_unavailable"]["decrease_bytes"], 1_200_000_000)
        self.assertEqual(memory["selected_vs_initial_mlp"]["process_rss"]["decrease_bytes"], 100_000_000)
        initial["per_fixture"][0]["answer"] = "Different initial answer."
        with self.assertRaisesRegex(ValueError, "not equivalent"):
            mlp.compare(before, after, set(mlp.MUTABLE_FIELDS), initial)

    def test_cli_output_is_exclusive_and_fingerprints_both_validators(self):
        output = self.directory / "mlp-comparison.json"
        args = ["--baseline-dir", str(self.directory), "--candidate-dir", str(self.directory),
                "--plan", str(self.plan_path), "--output", str(output)]
        with mock.patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(mlp.main(args), 0)
        receipt = json.loads(output.read_text())
        self.assertTrue(receipt["comparison"]["accepted"])
        self.assertIsNotNone(receipt["initial_mlp"])
        self.assertEqual(receipt["raw_validator_sha256"], mlp.core.sha256(Path(mlp.core.__file__).read_bytes()))
        before = output.read_bytes()
        with mock.patch("sys.stderr", new_callable=io.StringIO):
            self.assertEqual(mlp.main(args), 2)
        self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
