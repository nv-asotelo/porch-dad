"""CPU format and immutability checks; these do not measure model quality."""

import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external/TensorRT-Edge-LLM"))
from experimental.builder.weight_packing import int4
from experimental.builder.core.quantization import algorithm_to_type, QUANT_INT4_GPTQ, parse_quantization
from experimental.builder.core.weights import Weights
cosmos_spec = importlib.util.spec_from_file_location("cosmos_weights_cpu",
    ROOT / "external/TensorRT-Edge-LLM/experimental/builder/models/cosmos3/weights.py")
cosmos_weights = importlib.util.module_from_spec(cosmos_spec)
cosmos_spec.loader.exec_module(cosmos_weights)

spec = importlib.util.spec_from_file_location("rtn_converter", ROOT / "scripts/quantize_cosmos3_rtn.py")
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


class RtnTests(unittest.TestCase):
    def test_mlp_scope_uses_actual_upstream_dispatch_and_weight_loading(self):
        names = ["layers.0.self_attn.to_q.weight", "layers.0.mlp.up_proj.weight",
                 "layers.0.mlp.down_proj.weight", "lm_head.weight", "model.visual.test.weight"]
        values = (np.arange(128 * 256).reshape(128, 256) % 15 - 7).astype(np.float32)
        raw = (values.view(np.uint32) >> 16).astype("<u2").tobytes()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "source.bin").write_bytes(raw * len(names))
            catalog = {name: {"shape": values.shape, "shard": "source.bin", "offset": i * len(raw)}
                       for i, name in enumerate(names)}
            tensors, size = converter.output_catalog(catalog, "mlp-only")
            self.assertEqual(len(tensors), len(names) + 4)
            self.assertEqual(tensors["layers.0.mlp.up_proj.qweight"]["shape"], [32, 128])
            self.assertEqual(tensors["layers.0.mlp.up_proj.qzeros"]["shape"], [2, 16])
            self.assertEqual(tensors["layers.0.mlp.up_proj.scales"]["shape"], [2, 128])
            encoded = json.dumps(tensors).encode()
            destination = folder / "model.safetensors"
            destination.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(size))
            for name, item in catalog.items():
                converter.convert_tensor(folder, name, item, destination, 8 + len(encoded), tensors, "mlp-only")
            self.assertEqual((folder / "source.bin").read_bytes(), raw * len(names))
            (folder / "model.safetensors.index.json").write_text(json.dumps(
                {"weight_map": {name: destination.name for name in tensors}}))
            metadata = converter.quantization_metadata(catalog, "mlp-only")
            self.assertIn("layers.0.self_attn.q_proj", metadata["quantization"]["exclude_modules"])
            (folder / "hf_quant_config.json").write_text(json.dumps(metadata))
            quant = parse_quantization(str(folder), {}, {}, conversion=cosmos_weights)
            weights = Weights(str(folder), group_size=128, quant=quant,
                              conversion=cosmos_weights, int4_gemm_plugin_version=1)
            try:
                for module in ("model.layers.0.self_attn.q_proj", "lm_head", "model.visual.test"):
                    self.assertEqual(quant.module_type(module), "fp16")
                    plain = weights.linear(module, weights.module_quant_type(module))
                    self.assertEqual(plain.quant_type, "fp16")
                    np.testing.assert_array_equal(plain.weight, values.astype(np.float16))
                for module in ("model.layers.0.mlp.up_proj", "model.layers.0.mlp.down_proj"):
                    self.assertEqual(quant.module_type(module), "int4_gptq")
                    packed = weights.linear(module, weights.module_quant_type(module))
                    self.assertEqual(packed.quant_type, "int4_gptq")
                    self.assertEqual(packed.weight.shape, (64, 256))
                    np.testing.assert_array_equal(packed.weight, int4.pack_intweights((values + 8).astype(np.uint8)))
            finally:
                weights.close()

    def test_default_catalog_and_exclusions_preserve_legacy_contract(self):
        catalog = {name: {"shape": shape} for name, shape in [
            ("layers.0.self_attn.to_q.weight", (128, 256)),
            ("layers.0.mlp.up_proj.weight", (128, 256)),
            ("lm_head.weight", (128, 256)),
            ("model.visual.test.weight", (2, 2))]}
        default = converter.output_catalog(catalog)
        self.assertEqual(default, converter.output_catalog(catalog, "all-linears"))
        tensors, size = default
        self.assertEqual(len(tensors), 10)
        self.assertEqual(size, 3 * (32 * 128 * 4 + 2 * 16 * 4 + 2 * 128 * 2) + 8)
        self.assertEqual(converter.excluded_modules(catalog), ["visual.test"])
        with self.assertRaises(ValueError):
            converter.output_catalog(catalog, "unknown")

    def test_actual_upstream_v1_v2_packing(self):
        signed = (np.arange(128 * 256).reshape(128, 256) % 15 - 7).astype(np.float32)
        values = signed * np.tile(np.repeat([0.25, 2.0], 128), (128, 1))
        unsigned, scales, metrics = converter.quantize_block(values)
        self.assertEqual(metrics["squared_error"], 0)
        np.testing.assert_array_equal(unsigned, signed + 8)
        np.testing.assert_array_equal(scales, np.tile([0.25, 2.0], (128, 1)))
        packed = converter.pack_rows(unsigned)
        np.testing.assert_array_equal(int4._unpack_gptq_rows(packed).T, unsigned)
        zeros = np.full((2, 16), 0x77777777, dtype=np.int32)
        for version, reference in ((1, int4.pack_intweights), (2, int4.pack_cutedsl_fragment)):
            actual, permutation = int4.repack_gptq(packed, zeros, zero_point_offset=1, plugin_version=version)
            np.testing.assert_array_equal(actual, reference(unsigned))
            np.testing.assert_array_equal(permutation, np.arange(256))
        self.assertEqual(algorithm_to_type("RTN_GPTQ"), QUANT_INT4_GPTQ)

    def test_stored_scale_error_and_zero_groups(self):
        values = np.random.default_rng(42).normal(size=(8, 256)).astype(np.float32)
        unsigned, scales, metrics = converter.quantize_block(values)
        reconstructed = ((unsigned.astype(np.float32) - 8).reshape(8, 2, 128)
                         * scales.astype(np.float32)[..., None]).reshape(values.shape)
        self.assertLessEqual(metrics["max_error_in_scale_units"], 0.501)
        self.assertAlmostEqual(metrics["squared_error"], float(np.sum((values - reconstructed).astype(np.float64)**2)))
        unsigned, scales, metrics = converter.quantize_block(np.zeros((8, 128), np.float32))
        np.testing.assert_array_equal(unsigned, 8)
        np.testing.assert_array_equal(scales, 1)
        self.assertEqual(metrics["squared_error"], 0)
        with self.assertRaises(ValueError):
            converter.quantize_block(np.full((8, 128), np.nan))
        with self.assertRaises(ValueError):
            converter.quantize_block(np.zeros((8, 127)))

    def test_memmapped_conversion_preserves_source_and_plain_vision(self):
        language_name = "layers.0.self_attn.to_q.weight"
        vision_name = "model.visual.patch_embed.proj.weight"
        language = (np.arange(8 * 128).reshape(8, 128) % 15 - 7).astype(np.float32)
        vision = np.array([[1.0, -2.0], [0.25, 0]], dtype=np.float32)
        payload = b"".join((array.view(np.uint32) >> 16).astype("<u2").tobytes() for array in (language, vision))
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            shard = source / "source.bin"
            shard.write_bytes(payload)
            catalog = {
                language_name: {"shard": shard.name, "offset": 0, "shape": language.shape},
                vision_name: {"shard": shard.name, "offset": language.size * 2, "shape": vision.shape},
            }
            tensors, size = converter.output_catalog(catalog)
            encoded = json.dumps(tensors).encode()
            encoded += b" " * (-len(encoded) % 8)
            destination = source / "model.safetensors"
            destination.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(size))
            for name, item in catalog.items():
                converter.convert_tensor(source, name, item, destination, 8 + len(encoded), tensors)
            self.assertEqual(shard.read_bytes(), payload)
            self.assertNotIn(language_name, tensors)
            self.assertEqual(tensors[vision_name]["dtype"], "F16")
            plain = converter.mapped_output(destination, 8 + len(encoded), tensors[vision_name])
            np.testing.assert_array_equal(plain, vision)
            plain._mmap.close()
            weight = converter.mapped_output(destination, 8 + len(encoded), tensors["layers.0.self_attn.q_proj.qweight"])
            np.testing.assert_array_equal(int4._unpack_gptq_rows(weight).T - 8, language)
            weight._mmap.close()


if __name__ == "__main__":
    unittest.main()
