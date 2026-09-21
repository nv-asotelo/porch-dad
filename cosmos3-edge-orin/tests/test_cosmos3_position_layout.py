"""Execute the actual position kernel body on CPU against independent grid references.

CUDA indexing and the half conversion are represented by host stand-ins. This
checks the production formulas, packing and offsets; it is not a CUDA execution
test. Antialiased downsampling below the learned 16x16 grid is out of scope.
"""

from io import StringIO
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CPP = ROOT / "external/TensorRT-Edge-LLM/cpp"


def reference(height, width, aligned):
    if aligned:
        y = np.arange(height, dtype=np.float32) * np.float32(15 / (height - 1))
        x = np.arange(width, dtype=np.float32) * np.float32(15 / (width - 1))
    else:
        y = np.clip((np.arange(height, dtype=np.float32) + np.float32(.5)) * np.float32(16 / height) - np.float32(.5), 0, 15)
        x = np.clip((np.arange(width, dtype=np.float32) + np.float32(.5)) * np.float32(16 / width) - np.float32(.5), 0, 15)
    y0, x0 = y.astype(np.int64), x.astype(np.int64)
    y1, x1 = np.minimum(y0 + 1, 15), np.minimum(x0 + 1, 15)
    dy, dx = y - y0.astype(np.float32), x - x0.astype(np.float32)
    indices = np.stack([a[:, None] * 16 + b[None, :] for a, b in ((y0, x0), (y0, x1), (y1, x0), (y1, x1))])
    weights = np.stack([a[:, None] * b[None, :] for a, b in ((1-dy, 1-dx), (1-dy, dx), (dy, 1-dx), (dy, dx))])
    def blocks(array):
        return array.reshape(4, height // 2, 2, width // 2, 2).transpose(0, 1, 3, 2, 4).reshape(4, -1)
    return blocks(indices), blocks(weights).astype(np.float16)


class PositionLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("clang++") or shutil.which("c++")
        if not compiler:
            raise unittest.SkipTest("A host C++ compiler is required")
        source = (CPP / "kernels/preprocessKernels/imageUtilKernels.cu").read_text()
        start = source.index("__global__ void initFastPosEmbedQwenViTKernel(")
        end = source.index("\nvoid initFastPosEmbedQwenViT(", start)
        kernel = source[start:end].replace("__global__ ", "", 1)
        host = source[end:]
        slope_start = host.index("    float const lineSpaceH =")
        slope_end = host.index("\n\n", slope_start)
        slopes = host[slope_start:slope_end]
        cls.scratch = tempfile.TemporaryDirectory()
        cls.executable = Path(cls.scratch.name) / "position-kernel"
        harness = Path(cls.scratch.name) / "position-kernel.cpp"
        harness.write_text("""
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>
using half = float;
float __float2half(float value) { return value; }
struct Dim { int64_t x; } blockIdx{0}, blockDim{1}, threadIdx{0};
""" + kernel + """
int main(int argc, char** argv) {
    int64_t const H = std::atoll(argv[1]), W = std::atoll(argv[2]);
    bool const alignCorners = std::atoi(argv[3]);
    int64_t const numGridPerSide = 16, startIdx = 5, frames = 2;
    int64_t const total = startIdx + frames * H * W + 7;
""" + slopes + """
    std::vector<int64_t> indices(4 * total, -77);
    std::vector<half> weights(4 * total, -88);
    for (int64_t frame = 0; frame < frames; ++frame) {
        for (threadIdx.x = 0; threadIdx.x < H * W + 5; ++threadIdx.x) {
            initFastPosEmbedQwenViTKernel(indices.data(), weights.data(), H/2, W/2,
                2, numGridPerSide, lineSpaceH, lineSpaceW, startIdx + frame*H*W,
                total, alignCorners);
        }
    }
    std::cout << std::setprecision(9);
    for (size_t i = 0; i < indices.size(); ++i) std::cout << indices[i] << ' ' << weights[i] << '\\n';
}
""")
        subprocess.run([compiler, "-std=c++17", "-O0", "-ffp-contract=off", str(harness), "-o", str(cls.executable)], check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.scratch.cleanup()

    def check_grid(self, height, width, aligned):
        output = subprocess.check_output([str(self.executable), str(height), str(width), str(int(aligned))], text=True)
        actual = np.loadtxt(StringIO(output))
        total = 5 + 2 * height * width + 7
        indices = actual[:, 0].astype(np.int64).reshape(4, total)
        weights = actual[:, 1].astype(np.float16).reshape(4, total)
        expected_indices, expected_weights = reference(height, width, aligned)
        for frame in range(2):
            selection = slice(5 + frame * height * width, 5 + (frame + 1) * height * width)
            np.testing.assert_array_equal(indices[:, selection], expected_indices)
            np.testing.assert_array_equal(weights[:, selection], expected_weights)
        np.testing.assert_array_equal(indices[:, :5], -77)
        np.testing.assert_array_equal(indices[:, -7:], -77)
        np.testing.assert_array_equal(weights[:, :5], -88)
        np.testing.assert_array_equal(weights[:, -7:], -88)

    def test_cosmos_identity_upsample_and_nonsquare(self):
        for height, width in ((16, 16), (32, 32), (16, 32), (24, 40), (40, 24), (64, 16)):
            with self.subTest(height=height, width=width):
                self.check_grid(height, width, False)

    def test_qwen_default_formulas_are_preserved(self):
        for height, width in ((16, 16), (32, 32), (24, 40), (8, 10)):
            with self.subTest(height=height, width=width):
                self.check_grid(height, width, True)
        header = (CPP / "kernels/preprocessKernels/imageUtilKernels.h").read_text()
        self.assertIn("cudaStream_t stream, bool const alignCorners = true", header)
        qwen = (CPP / "multimodal/qwen3/qwen3vlViTRunner.h").read_text()
        cosmos = (CPP / "multimodal/cosmos3/cosmos3EdgeViTRunner.h").read_text()
        self.assertIn("fastPosEmbedAlignCorners() const\n    {\n        return true;", qwen)
        self.assertIn("fastPosEmbedAlignCorners() const override\n    {\n        return false;", cosmos)


if __name__ == "__main__":
    unittest.main()
