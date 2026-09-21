"""Offline cache-profile regression checks, with no TensorRT/GPU imports."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "external/TensorRT-Edge-LLM"))
import rtn_backend
import quantize_cosmos3_rtn as converter
import preflight_cosmos_artifacts as preflight
from experimental.server.runtime import engine_build


class RtnProfileTests(unittest.TestCase):
    def test_each_profile_override_changes_cache_and_argv(self):
        variants = [("max_input_len", 768, "--max-input-len"),
                    ("max_kv_capacity", 1024, "--max-kv-cache-capacity"),
                    ("max_image_tokens", 512, "--max-image-tokens"),
                    ("max_image_tokens_per_image", 256, "--max-image-tokens-per-image")]
        default = rtn_backend.build_options()
        self.assertIsNone(default.max_image_tokens)
        self.assertIsNone(default.max_image_tokens_per_image)
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            model.mkdir()
            (model / "config.json").write_text('{}')
            seen = {engine_build.bundle_cache_path(str(model), directory, default)}
            for keyword, value, flag in variants:
                with self.subTest(keyword=keyword):
                    options = rtn_backend.build_options(**{keyword: value})
                    path = engine_build.bundle_cache_path(str(model), directory, options)
                    self.assertNotIn(path, seen)
                    seen.add(path)
                    argv = options.to_argv(str(model), directory)
                    self.assertEqual(argv[argv.index(flag) + 1], str(value))
                    self.assertEqual(argv[-2:], ["--int4-gemm-plugin-version", "1"])
                    self.assertEqual(path, engine_build.bundle_cache_path(str(model), directory,
                        rtn_backend.build_options(**{keyword: value})))
        for invalid in ({"max_input_len": 0}, {"max_kv_capacity": 512},
                        {"max_image_tokens": 256}, {"max_image_tokens_per_image": 0}):
            with self.assertRaises(ValueError):
                rtn_backend.build_options(**invalid)

    def test_scope_validation_checks_exact_modules_counts_and_exclusions(self):
        catalog = {f"layers.{layer}.{module}.weight": {"shape": (128, 256)}
                   for layer in range(28) for module in
                   ("self_attn.to_q", "self_attn.to_k", "self_attn.to_v", "self_attn.to_out",
                    "mlp.up_proj", "mlp.down_proj")}
        catalog["lm_head.weight"] = {"shape": (128, 256)}
        catalog.update({f"model.visual.fixture_{i}.weight": {"shape": (2, 2)} for i in range(529)})
        for scope, count, total in (("all-linears", 169, 1036), ("mlp-only", 56, 810)):
            with self.subTest(scope=scope):
                tensors, _ = converter.output_catalog(catalog, scope)
                index = {name: "model.safetensors" for name in tensors}
                metadata = converter.quantization_metadata(catalog, scope)["quantization"]
                record = {"quantization_scope": scope, "quantized_linear_count": count,
                          "source_tensor_count": 698, "output_tensor_count": total}
                self.assertEqual(rtn_backend.validate_scope_layout(record, metadata, index), (scope, count))
                with self.assertRaises(ValueError):
                    rtn_backend.validate_scope_layout({**record, "quantized_linear_count": count + 1}, metadata, index)
                with self.assertRaisesRegex(ValueError, "exclusions"):
                    rtn_backend.validate_scope_layout(record, {**metadata, "exclude_modules": []}, index)
                wrong = dict(index)
                wrong["unexpected.qweight"] = wrong.pop("layers.0.mlp.up_proj.qweight")
                with self.assertRaisesRegex(ValueError, "declared scope"):
                    rtn_backend.validate_scope_layout(record, metadata, wrong)
                if scope == "all-linears":
                    del record["quantization_scope"]
                    del metadata["quantization_scope"]
                    self.assertEqual(rtn_backend.validate_scope_layout(record, metadata, index), (scope, count))
                else:
                    with self.assertRaisesRegex(ValueError, "inconsistent"):
                        rtn_backend.validate_scope_layout(record, {**metadata, "quantization_scope": "all-linears"}, index)

    def test_replacement_and_fingerprint_keep_explicit_v1(self):
        plain = engine_build.BuildOptions(max_input_len=1024, max_kv_cache_capacity=2048, max_batch_size=1)
        first, second = rtn_backend.build_options(), rtn_backend.build_options()
        self.assertNotIn("int4_gemm_plugin_version", asdict(plain))
        self.assertEqual(asdict(first), asdict(second))
        replaced = replace(first, plugin_path="/test/plugin.so")
        argv = replaced.to_argv("/model", "/cache")
        self.assertEqual(argv[-2:], ["--int4-gemm-plugin-version", "1"])
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            model.mkdir()
            (model / "config.json").write_text('{}')
            first_path = engine_build.bundle_cache_path(str(model), directory, first)
            self.assertEqual(first_path, engine_build.bundle_cache_path(str(model), directory, second))
            self.assertEqual(first_path, engine_build.bundle_cache_path(str(model), directory, replaced))
            self.assertNotEqual(first_path, engine_build.bundle_cache_path(str(model), directory, plain))

    def test_preflight_checks_custom_profile_and_keeps_default(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            model.mkdir()
            (model / "config.json").write_text('{}')
            (model / "processed_chat_template.json").write_text('{}')
            options = rtn_backend.build_options()
            with patch.object(preflight, "source_tokenizer", return_value=object()), \
                 patch.object(preflight, "validate_chat"), \
                 patch.object(preflight, "validate_bundle", return_value={}) as bundle_check:
                result = preflight.validate(model, ROOT / "external/TensorRT-Edge-LLM", directory,
                    max_input_len=1024, max_kv_capacity=2048, preflight_only=False, build_options=options)
                self.assertEqual(result["bundle"], engine_build.bundle_cache_path(str(model), directory, options))
                self.assertTrue(bundle_check.call_args.kwargs["require_ready"])
                with patch.object(engine_build, "_is_ready", return_value=True) as ready:
                    self.assertTrue(bundle_check.call_args.kwargs["is_ready"]())
                    self.assertIs(ready.call_args.args[2], options)
                default = preflight.validate(model, ROOT / "external/TensorRT-Edge-LLM", directory,
                    max_input_len=1024, max_kv_capacity=2048, preflight_only=False)
                self.assertNotEqual(default["bundle"], result["bundle"])
                with self.assertRaisesRegex(ValueError, "must match"):
                    preflight.validate(model, ROOT / "external/TensorRT-Edge-LLM", directory,
                        max_input_len=512, max_kv_capacity=2048, preflight_only=False, build_options=options)


if __name__ == "__main__":
    unittest.main()
