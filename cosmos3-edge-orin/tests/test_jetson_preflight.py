"""Compatibility decisions must not turn upstream support into local validation."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("jetson_preflight", Path(__file__).resolve().parents[1] / "scripts/jetson_preflight.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def known_device():
    return {"os": "Linux", "architecture": "aarch64", "device_model": "NVIDIA Jetson Orin Nano Developer Kit",
            "device_compatible": "nvidia,p3767-0005 nvidia,tegra234", "l4t": "39.2.1",
            "ram_total_bytes": int(7.37 * mod.GIB), "ram_available_bytes": 6 * mod.GIB,
            "project_free_bytes": 40 * mod.GIB, "project_writable": True, "root_filesystem": "ext4",
            "cuda_compiler": "13.2.86", "canonical_cuda_toolkit": True,
            "tensorrt": "10.16.2.10", "python": [3, 12, 3],
            "gpu_count": 1, "compute_capability": [8, 7], "cuda_runtime": 13020,
            "cuda_array_sum": 523776, "native_binding": True, "encoder_budget_binding": True,
            "image_budget_binding": True, "encoder_bypass_binding": True}


class CompatibilityTests(unittest.TestCase):
    def test_actual_nano_memory_is_accepted_but_never_inference_validated(self):
        for stage in ("system", "runtime"):
            result = mod.evaluate(known_device(), stage)
            self.assertTrue(result["ready_for_next_step"], result)
            self.assertFalse(result["inference_validated"])

    def test_cupy_cuda13_wheel_runtime_does_not_need_to_equal_toolkit_minor(self):
        self.assertTrue(mod.evaluate({**known_device(), "cuda_runtime": 13000}, "runtime")["ready_for_next_step"])

    def test_other_orin_sizes_remain_candidates(self):
        for model, size in (("NVIDIA Jetson Orin NX", 7.5), ("NVIDIA Jetson AGX Orin", 61)):
            device = known_device()
            device.update(device_model=model, ram_total_bytes=int(size * mod.GIB))
            result = mod.evaluate(device, "runtime")
            self.assertTrue(result["ready_for_next_step"], result)
            self.assertTrue(any("not hardware-tested" in line for line in result["warnings"]))

    def test_wrong_devices_and_incomplete_stacks_fail_closed(self):
        variants = [dict(os="Darwin"), dict(architecture="x86_64"), dict(root_filesystem="overlay"),
                    dict(device_model="Jetson Thor", device_compatible="nvidia,tegra264", compute_capability=[11, 0]),
                    dict(device_model="Jetson AGX Xavier", device_compatible="nvidia,tegra194", compute_capability=[7, 2]),
                    dict(ram_total_bytes=4 * mod.GIB), dict(compute_capability=[8, 6]),
                    dict(gpu_count=0), dict(l4t="36.4.3"), dict(cuda_compiler="12.6.85"),
                    dict(canonical_cuda_toolkit=False), dict(tensorrt="10.3.0"), dict(python=[3, 10, 12]),
                    dict(cuda_array_sum=0), dict(native_binding=False), dict(encoder_budget_binding=False),
                    dict(image_budget_binding=False), dict(encoder_bypass_binding=False), dict(cuda_runtime=12060)]
        for change in variants:
            with self.subTest(change=change):
                self.assertFalse(mod.evaluate({**known_device(), **change}, "runtime")["ready_for_next_step"])

    def test_storage_gate_is_for_fresh_build_and_swap_is_not_changed(self):
        device = {**known_device(), "project_free_bytes": 2 * mod.GIB, "swap_total_bytes": 2 * mod.GIB}
        self.assertFalse(mod.evaluate(device, "system")["ready_for_next_step"])
        result = mod.evaluate(device, "runtime")
        self.assertTrue(result["ready_for_next_step"])
        self.assertTrue(any("Swap" in line for line in result["warnings"]))

    def test_package_patch_variants_are_explicitly_unvalidated(self):
        result = mod.evaluate({**known_device(), "tensorrt": "10.16.3.1", "l4t": "39.2.2"}, "runtime")
        self.assertTrue(result["ready_for_next_step"])
        self.assertTrue(any("Package versions differ" in line for line in result["warnings"]))
        self.assertFalse(result["inference_validated"])

    def test_real_version_output_parsers(self):
        self.assertEqual(mod.parse_l4t("# R39 (release), REVISION: 2.1, GCID: 123"), "39.2.1")
        self.assertEqual(mod.parse_cuda("Cuda compilation tools, release 13.2, V13.2.86"), "13.2.86")
        self.assertIsNone(mod.parse_l4t("not a release"))
        self.assertIsNone(mod.parse_cuda("compiler unavailable"))

    def test_cli_failure_receipt_is_private_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inventory.json"
            with patch.object(mod, "collect", return_value={"os": "Darwin"}), patch("builtins.print"):
                self.assertEqual(mod.main(["--project-dir", directory, "--output", str(path)]), 2)
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
                saved = path.read_bytes()
                with self.assertRaises(SystemExit):
                    mod.main(["--project-dir", directory, "--output", str(path)])
                self.assertEqual(path.read_bytes(), saved)


if __name__ == "__main__":
    unittest.main()
