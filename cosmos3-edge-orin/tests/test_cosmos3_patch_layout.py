"""CPU regression for the pinned lightweight builder's Cosmos3 patch projection."""

import ast
from dataclasses import dataclass, replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "external/TensorRT-Edge-LLM/experimental/builder/models/cosmos3"
SPEC = importlib.util.spec_from_file_location("cosmos3_weight_layout", SOURCE / "weights.py")
weights = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(weights)
adapt = weights.adapt_patch_embedding_weight_for_chw_input


@dataclass
class Descriptor:
    weight: np.ndarray
    bias: np.ndarray
    weight_recipe: dict | None


class PatchLayoutTests(unittest.TestCase):
    def test_distinct_color_and_spatial_values_preserve_projection(self):
        rng = np.random.default_rng(517)
        matrix = rng.normal(size=(11, 3 * 16 * 16)).astype(np.float64)
        # Colored constant patches and spatial variation exercise both channel and pixel order.
        patches = rng.normal(size=(5, 16, 16, 3))
        patches[:3] = np.array([[1, -1, -1], [-1, 1, -1], [-1, -1, 1]])[:, None, None, :]
        reference = patches.reshape(5, -1) @ matrix.T
        runtime_patches = patches.transpose(0, 3, 1, 2).reshape(5, -1)
        self.assertGreater(float(np.max(np.abs(runtime_patches @ matrix.T - reference))), 1)
        corrected = runtime_patches @ adapt(matrix, 16, 3).T
        np.testing.assert_allclose(corrected, reference, rtol=1e-12, atol=1e-12)
        self.assertTrue(adapt(matrix, 16, 3).flags.c_contiguous)

    def test_preserves_dtype_source_and_upstream_column_permutation(self):
        source = np.arange(2 * 3 * 4 * 4, dtype=np.float16).reshape(2, -1)
        original = source.copy()
        actual = adapt(source, 4, 3)
        # Explicit indexing is independent of the implementation's reshape/transpose path.
        expected = np.empty_like(source)
        for output in range(2):
            for channel in range(3):
                for y in range(4):
                    for x in range(4):
                        expected[output, channel * 16 + y * 4 + x] = source[output, (y * 4 + x) * 3 + channel]
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(source, original)
        self.assertEqual(actual.dtype, np.float16)

    def test_wrong_shapes_rejected(self):
        for shape, patch_size, channels in (((3, 7), 2, 3), ((12,), 2, 3), ((3, 0), 0, 3)):
            with self.subTest(shape=shape, patch_size=patch_size):
                with self.assertRaises(ValueError):
                    adapt(np.zeros(shape), patch_size, channels)

    def test_projection_descriptor_drops_identity_runtime_recipe(self):
        # Exercise the production override without importing unavailable TensorRT on the Mac.
        tree = ast.parse((SOURCE / "modeling_cosmos3_reasoner_visual.py").read_text())
        node = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                    and node.name == "Cosmos3ReasonerPatchEmbedding")
        namespace = {"Linear": object, "replace": replace,
                     "adapt_patch_embedding_weight_for_chw_input": adapt}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
        projection = object.__new__(namespace[node.name])
        projection.prefix = "model.visual.embeddings.patch_embedding"
        projection.patch_size, projection.num_channels = 2, 3
        projection.quant_type = lambda: "fp16"
        original = Descriptor(np.arange(48, dtype=np.float16).reshape(4, 12),
                              np.arange(4, dtype=np.float16), {"assemble": "identity"})
        projection.weights = SimpleNamespace(linear=lambda prefix, precision: original)
        descriptor = projection.weight_descriptor()
        self.assertIsNone(descriptor.weight_recipe)
        self.assertIs(descriptor.bias, original.bias)
        self.assertEqual(original.weight_recipe, {"assemble": "identity"})
        np.testing.assert_array_equal(descriptor.weight, adapt(original.weight, 2, 3))
        projection.quant_type = lambda: "int4_awq"
        with self.assertRaises(ValueError):
            projection.weight_descriptor()


if __name__ == "__main__":
    unittest.main()
