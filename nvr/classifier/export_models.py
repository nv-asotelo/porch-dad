#!/usr/bin/env python3
"""Export the two Live Vision classifiers to the ONNX files their meta.json names. Run on a workstation.

Downloads each model at its pinned revision, checks the weights' SHA-256, and exports ONNX opset 17
with two outputs: logits, and a saliency map computed in the graph, so the Orin needs no PyTorch:

  skshmjn  ViT-B/16 (Hugging Face, Apache-2.0): attention rollout - the mean of the heads, plus the
           identity, row-normalised, multiplied through the 12 layers - CLS to the 14x14 patches.
           Class-agnostic: where the model looked, whatever it decided.
  bierny   DenseNet121 (ONNX on Hugging Face, MIT): Grad-CAM for the top class at the last
           convolution (1024x7x7). Its head is global average pool then one linear layer, so the
           Grad-CAM channel weights are that class's row of the linear weights.

Then zeroes FP32 weights below the smallest normal float (Bierny carries 77,581, which only slow
CPU kernels down; outputs unchanged), and converts to FP16 with float32 inputs and outputs. The
Orin builds its TensorRT engines from those files (build_engines.py).

Needs torch, transformers, onnx, onnxruntime and huggingface_hub. No Hugging Face token: both
repositories are public.

  python export_models.py [--out-dir nvr/classifier/models] [skshmjn] [bierny]
"""
import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = Path(__file__).resolve().parent
PINS = {
    "skshmjn": ("skshmjn/Pokemon-classifier-gen9-1025", "9e8d54de136b99afc212322eae13ddc07a6fd779",
                {"model.safetensors": "2da26f20523f4f09a74e277fb5d0a336fddf8b53f5b8eaefc7f8f063d9d0699a",
                 "config.json": None, "preprocessor_config.json": None}),
    "bierny": ("BiernyVR/pokemon-classifier-mobilenetv3", "3bf528cf0f6ab8464ea3bac0f70d84886e8dab25",
               {"pokemon_classifier.onnx": "ede506ccbded53eb7867f081417d721dbd32e3aa0f12f73683307f3f6b81d69f"}),
}


def download(mid: str) -> Path:
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    from huggingface_hub import hf_hub_download
    repo, revision, files = PINS[mid]
    paths = {}
    for name, sha in files.items():
        paths[name] = Path(hf_hub_download(repo, name, revision=revision))
        if sha and hashlib.sha256(paths[name].read_bytes()).hexdigest() != sha:
            raise SystemExit(f"{mid}: {name} does not match its pinned SHA-256")
    return paths[next(iter(files))].parent


def export_skshmjn(source: Path, out: Path):
    import torch
    from transformers import ViTForImageClassification

    class WithRollout(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, pixel_values):
            result = self.model(pixel_values=pixel_values, output_attentions=True)
            tokens = result.attentions[0].shape[-1]
            eye = torch.eye(tokens, dtype=pixel_values.dtype, device=pixel_values.device)
            joint = None
            for attention in result.attentions:
                a = 0.5 * attention.mean(1) + 0.5 * eye
                a = a / a.sum(-1, keepdim=True)
                joint = a if joint is None else torch.matmul(a, joint)
            return result.logits, joint[:, 0, 1:].reshape(-1, 14, 14)

    model = ViTForImageClassification.from_pretrained(source, attn_implementation="eager").eval()
    with torch.no_grad():
        torch.onnx.export(WithRollout(model).eval(), (torch.randn(1, 3, 224, 224),), str(out),
                          input_names=["pixel_values"], output_names=["logits", "saliency"],
                          dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}, "saliency": {0: "batch"}},
                          opset_version=17, dynamo=False, do_constant_folding=True)


def export_bierny(source: Path, out: Path):
    from onnx import version_converter
    model = version_converter.convert_version(onnx.load(source / "pokemon_classifier.onnx"), 17)
    graph = model.graph
    for value in list(graph.input) + list(graph.output):
        value.type.tensor_type.shape.dim[0].dim_param = "batch"
    for value in graph.output:
        if value.name == "output":
            value.name = "logits"
    for node in graph.node:
        node.output[:] = ["logits" if x == "output" else x for x in node.output]
    graph.node.extend([
        helper.make_node("ArgMax", ["logits"], ["cam_cls"], axis=1, keepdims=0, name="cam_argmax"),
        helper.make_node("Gather", ["classifier.weight", "cam_cls"], ["cam_alpha"], axis=0, name="cam_gather"),
        helper.make_node("Unsqueeze", ["cam_alpha", "cam_axes23"], ["cam_alpha4"], name="cam_unsq"),
        helper.make_node("Mul", ["/features/norm5/act/Relu_output_0", "cam_alpha4"], ["cam_weighted"], name="cam_mul"),
        helper.make_node("ReduceSum", ["cam_weighted", "cam_axis1"], ["cam_sum"], keepdims=0, name="cam_rsum"),
        helper.make_node("Relu", ["cam_sum"], ["saliency"], name="cam_relu"),
    ])
    graph.initializer.extend([numpy_helper.from_array(np.array([2, 3], np.int64), "cam_axes23"),
                              numpy_helper.from_array(np.array([1], np.int64), "cam_axis1")])
    graph.output.append(helper.make_tensor_value_info("saliency", TensorProto.FLOAT, ["batch", 7, 7]))
    del graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, out)


def flush_denormals(path: Path) -> int:
    model = onnx.load(path)
    count = 0
    for tensor in model.graph.initializer:
        if tensor.data_type == TensorProto.FLOAT:
            w = numpy_helper.to_array(tensor).copy()
            tiny = (w != 0) & (np.abs(w) < np.finfo(np.float32).tiny)
            if tiny.any():
                count += int(tiny.sum())
                w[tiny] = 0.0
                tensor.CopyFrom(numpy_helper.from_array(w, tensor.name))
    onnx.save(model, path)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="*", help=f"any of {', '.join(PINS)}; default both")
    parser.add_argument("--out-dir", type=Path, default=HERE / "models")
    args = parser.parse_args()
    if set(args.models) - set(PINS):
        parser.error(f"unknown model: {', '.join(sorted(set(args.models) - set(PINS)))}")
    from onnxruntime.transformers.float16 import convert_float_to_float16
    for mid in args.models or list(PINS):
        target = args.out_dir / mid
        target.mkdir(parents=True, exist_ok=True)
        fp32 = target / "model_fp32.onnx"
        (export_skshmjn if mid == "skshmjn" else export_bierny)(download(mid), fp32)
        zeroed = flush_denormals(fp32)
        onnx.save(convert_float_to_float16(onnx.load(fp32), keep_io_types=True), target / "model_fp16.onnx")
        fp32.unlink()
        print(f"{mid}: {target / 'model_fp16.onnx'} ({zeroed} denormal weights zeroed)")


if __name__ == "__main__":
    main()
