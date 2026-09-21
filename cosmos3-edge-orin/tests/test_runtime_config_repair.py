"""Offline artifact repair tests; no TensorRT, CUDA or model weights are loaded."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SPEC = importlib.util.spec_from_file_location(
    "runtime_config_repair", Path(__file__).parents[1] / "scripts/repair_cosmos_runtime_config.py")
repair = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repair)


def fixtures():
    source = {"model_type": "cosmos3_edge", "text_config": {
        "vocab_size": 131072, "rope_parameters": {
            "rope_theta": 100000000, "mrope_section": [24, 20, 20], "rope_type": "default"}},
        "vision_config": {"model_type": "cosmos3_edge_vision", "num_patches": 256}}
    artifact = copy.deepcopy(source)
    artifact["model_type"] = "cosmos3_edge_vision"
    artifact["builder_config"] = {"max_image_tokens": 1024}
    artifact["unrelated"] = {"preserve": [1, "text", None]}
    return source, artifact


class RepairTests(unittest.TestCase):
    def test_adds_aliases_without_mutating_inputs_or_unrelated_fields(self):
        source, artifact = fixtures()
        before_source, before_artifact = copy.deepcopy(source), copy.deepcopy(artifact)
        result, additions = repair.plan_repair(source, artifact)
        self.assertEqual(set(additions), {"text_config.rope_theta", "text_config.rope_scaling"})
        self.assertEqual(result["text_config"]["rope_theta"], 100000000)
        self.assertEqual(result["text_config"]["rope_scaling"]["mrope_section"], [24, 20, 20])
        self.assertEqual(result["text_config"]["rope_scaling"]["type"], "default")
        for key in ("rope_theta", "rope_scaling"):
            result["text_config"].pop(key)
        self.assertEqual(result, artifact)
        self.assertEqual(source, before_source)
        self.assertEqual(artifact, before_artifact)

    def test_rejects_wrong_family_conflicts_and_different_source_fields(self):
        for location, key, value in (("root", "model_type", "qwen3_vl"),
                                     ("text_config", "rope_theta", None),
                                     ("text_config", "rope_scaling", {"mrope_section": [16, 24, 24]}),
                                     ("text_config", "vocab_size", 100),
                                     ("vision_config", "num_patches", 1024)):
            with self.subTest(location=location, key=key):
                source, artifact = fixtures()
                target = artifact if location == "root" else artifact[location]
                target[key] = value
                with self.assertRaises(ValueError):
                    repair.plan_repair(source, artifact)
        source, artifact = fixtures()
        source["model_type"] = "qwen3_vl"
        with self.assertRaises(ValueError):
            repair.plan_repair(source, artifact)

    def test_already_normalized_is_noop(self):
        source, artifact = fixtures()
        result, _ = repair.plan_repair(source, artifact)
        again, additions = repair.plan_repair(source, result)
        self.assertEqual(again, result)
        self.assertEqual(additions, {})

    def test_apply_backs_up_exact_bytes_and_preserves_source_and_engine(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            model, bundle, receipts = base / "model", base / "bundle", base / "receipts"
            model.mkdir()
            (bundle / "visual").mkdir(parents=True)
            source, artifact = fixtures()
            source_bytes = json.dumps(source).encode()
            before = json.dumps(artifact, separators=(",", ":")).encode()
            (model / "config.json").write_bytes(source_bytes)
            target = bundle / "visual/config.json"
            target.write_bytes(before)
            target.chmod(0o640)
            engine = bundle / "visual/visual.engine"
            engine.write_bytes(b"synthetic sentinel, not a TensorRT engine")
            engine_before = engine.read_bytes()
            # Substitute the source pin only for this wholly synthetic filesystem fixture.
            with mock.patch.object(repair, "SOURCE_CONFIG_SHA256", repair.sha256(source_bytes)):
                dry = repair.repair(model, bundle, results_dir=receipts)
                self.assertEqual(dry["status"], "dry_run")
                self.assertEqual(target.read_bytes(), before)
                self.assertFalse(receipts.exists())
                applied = repair.repair(model, bundle, apply=True, results_dir=receipts)
                self.assertEqual(applied["status"], "applied")
                self.assertEqual(Path(applied["backup"]).read_bytes(), before)
                self.assertEqual(json.loads(Path(applied["receipt"]).read_text()), applied)
                self.assertEqual(repair.sha256(target.read_bytes()), applied["after_sha256"])
                self.assertEqual(target.stat().st_mode & 0o777, 0o640)
                after = target.read_bytes()
                files_before = sorted(receipts.iterdir())
                self.assertEqual(repair.repair(model, bundle, apply=True, results_dir=receipts)["status"],
                                 "already_normalized")
                self.assertEqual(target.read_bytes(), after)
                self.assertEqual(sorted(receipts.iterdir()), files_before)
            self.assertEqual((model / "config.json").read_bytes(), source_bytes)
            self.assertEqual(engine.read_bytes(), engine_before)
            with self.assertRaisesRegex(ValueError, "SHA256"):
                repair.repair(model, bundle, apply=True, results_dir=receipts)


if __name__ == "__main__":
    unittest.main()
