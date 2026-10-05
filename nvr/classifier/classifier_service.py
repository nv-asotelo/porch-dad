#!/usr/bin/env python3
"""Image classifiers for Live Vision - Pokémon and dog breeds: a loopback HTTP service with one model
resident at a time.

Live Vision (nvr/ui) offers these beside its Cosmos3-Edge engines as "kind": "classifier" registry
entries and relays its /api/classify here. Each model directory holds a TensorRT engine built on the
Orin from an ONNX export with two outputs - logits and a saliency map, Grad-CAM for the CNNs and
attention rollout for the ViTs, a class activation map for SigLIP - and a meta.json with the
model's own preprocessing and the canonical species (or breed) of every output
(nvr/classifier/README.md).

None of these models outputs 2D grounding (boxes, points or masks), so "boxes" is always empty. The
saliency grid shows where the model's evidence came from, not where a Pokémon or a dog is.

Only the model in use is resident: POST /load swaps it in, and Live Vision sends POST /unload when
it goes back to Cosmos3-Edge, so the service holds no model memory while no classifier is selected.

  GET  /health                  {"status": "ok", "loaded": id | null}
  GET  /models                  {"models": [{id, name, labels, species[], saliency, loaded, ...}]}
  POST /load     {model}        {"ok": true, "load_ms": n}
  POST /unload                  {"ok": true}
  POST /classify {model, image (base64 JPEG/PNG/WebP), saliency?: bool, topk?: 1-10}
                 -> {model, species, label, score, topk: [{species, label, score}],
                     saliency: {w, h, cells: [0..1, row-major], method} | null,
                     boxes: [], timing_ms: {preprocess, inference, total}}
"""
import argparse
import base64
import io
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

MAX_BODY = 4 * 1024 * 1024
MAX_PIXELS = 4096 * 4096
ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}\Z")
RESAMPLE = {"bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC, "lanczos": Image.LANCZOS}


def load_meta(directory: Path) -> dict:
    meta = json.loads((directory / "meta.json").read_text())
    if not ID.fullmatch(str(meta.get("id", ""))) or meta["id"] != directory.name:
        raise ValueError(f"{directory}: meta.json id must match the directory name")
    labels = meta.get("labels")
    if not isinstance(labels, list) or not labels or not all(
            isinstance(x, list) and len(x) == 2 and isinstance(x[0], str)
            and (x[1] is None or isinstance(x[1], str)) for x in labels):
        raise ValueError(f"{directory}: labels must be [[raw label, species or null], ...]")
    prep = meta.get("preprocess", {})
    if prep.get("resample") not in RESAMPLE or not all(
            isinstance(prep.get(k), list) and len(prep[k]) == 3 for k in ("mean", "std")):
        raise ValueError(f"{directory}: preprocess needs resample, mean and std")
    return meta


def preprocess(image: Image.Image, prep: dict) -> np.ndarray:
    """Each model's own recipe: a squash resize of the whole image, so a saliency grid covers the
    whole picture, then /255 and per-channel mean/std, as NCHW float32."""
    size = int(prep.get("size", 224))
    resized = image.convert("RGB").resize((size, size), RESAMPLE[prep["resample"]])
    a = np.asarray(resized, np.float32) / 255.0
    a = (a - np.asarray(prep["mean"], np.float32)) / np.asarray(prep["std"], np.float32)
    return np.ascontiguousarray(a.transpose(2, 0, 1)[None], dtype=np.float32)


def rank(logits: np.ndarray, labels: list, k: int) -> list:
    """Distinct species by their best label's logit, forms collapsed. Probability is a softmax over
    all labels, and a species scores its best label's. An unmapped label keeps its rank slot."""
    z = logits.astype(np.float64) - float(np.max(logits))
    p = np.exp(z)
    p /= p.sum()
    out, seen = [], set()
    for i in np.argsort(-logits, kind="stable"):
        raw, species = labels[i]
        key = species if species is not None else f"\0{i}"
        if key in seen:
            continue
        seen.add(key)
        out.append({"species": species, "label": raw, "score": round(float(p[i]), 6)})
        if len(out) == k:
            break
    return out


def saliency_grid(raw: np.ndarray, method: str) -> dict:
    s = np.nan_to_num(np.asarray(raw, np.float64).squeeze(), nan=0.0, posinf=0.0, neginf=0.0)
    if s.ndim != 2:
        raise ValueError("saliency output must be one 2D map")
    s = np.maximum(s, 0.0)
    peak = float(s.max())
    cells = (s / peak if peak > 0 else s).round(4)
    return {"w": int(s.shape[1]), "h": int(s.shape[0]), "cells": cells.ravel().tolist(), "method": method}


def check(result):
    """cuda.bindings calls return (error, *values)."""
    from cuda.bindings import runtime as cudart
    err, *values = result if isinstance(result, tuple) else (result,)
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error {err}")
    return values[0] if len(values) == 1 else values


class TensorRTModel:
    """One deserialized engine with fixed batch-1 buffers; inputs and outputs by tensor name."""

    def __init__(self, plan: Path, input_name: str, size: int):
        import tensorrt as trt
        from cuda.bindings import runtime as cudart
        self.cudart = cudart
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.runtime.deserialize_cuda_engine(plan.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"could not deserialize {plan}")
        self.context = self.engine.create_execution_context()
        self.stream = check(cudart.cudaStreamCreate())
        self.input = input_name
        self.context.set_input_shape(input_name, (1, 3, size, size))
        self.buffers = {}
        try:
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                host = np.empty(tuple(self.context.get_tensor_shape(name)),
                                trt.nptype(self.engine.get_tensor_dtype(name)))
                device = check(cudart.cudaMalloc(host.nbytes))
                self.buffers[name] = (host, device)
                self.context.set_tensor_address(name, device)
        except Exception:
            self.close()
            raise

    def run(self, x: np.ndarray) -> dict:
        cudart = self.cudart
        host, device = self.buffers[self.input]
        np.copyto(host, x, casting="same_kind")
        check(cudart.cudaMemcpyAsync(device, host.ctypes.data, host.nbytes,
                                     cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream))
        if not self.context.execute_async_v3(self.stream):
            raise RuntimeError("TensorRT execution failed")
        outputs = {name: pair for name, pair in self.buffers.items() if name != self.input}
        for host, device in outputs.values():
            check(cudart.cudaMemcpyAsync(host.ctypes.data, device, host.nbytes,
                                         cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream))
        check(cudart.cudaStreamSynchronize(self.stream))
        return {name: host.copy() for name, (host, _) in outputs.items()}

    def close(self):
        for _, device in self.buffers.values():
            self.cudart.cudaFree(device)
        self.buffers = {}
        if getattr(self, "stream", None) is not None:
            self.cudart.cudaStreamDestroy(self.stream)
            self.stream = None
        self.context = self.engine = None


class OnnxRuntimeModel:
    """The same model from its ONNX file, for a host without TensorRT (tests, a workstation)."""

    def __init__(self, onnx_path: Path):
        import onnxruntime as ort
        # CUDA when this onnxruntime has it, else CPU; never its TensorRT provider, which needs libraries
        # this path exists to do without.
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.input = self.session.get_inputs()[0].name
        self.outputs = [o.name for o in self.session.get_outputs()]

    def run(self, x: np.ndarray) -> dict:
        return dict(zip(self.outputs, self.session.run(None, {self.input: x})))

    def close(self):
        self.session = None


class Classifiers:
    def __init__(self, models_dir: Path, backend: str):
        self.backend = backend
        self.meta = {}
        for directory in sorted(p for p in models_dir.iterdir() if (p / "meta.json").is_file()):
            meta = load_meta(directory)
            meta["dir"] = directory
            self.meta[meta["id"]] = meta
        self.loaded_id, self.model = None, None
        self.lock = threading.Lock()

    def describe(self) -> list:
        out = []
        for mid, m in self.meta.items():
            species = sorted({s for _, s in m["labels"] if s})
            artifact = m["dir"] / (m["engine"] if self.backend == "tensorrt" else m["onnx"])
            out.append({"id": mid, "name": m.get("name", mid), "architecture": m.get("architecture", ""),
                        "labels": len(m["labels"]), "species": species,
                        "saliency": m.get("saliency", ""), "grounding": "none",
                        "license": m.get("license", ""), "source": m.get("source", ""),
                        "installed": artifact.is_file(), "loaded": mid == self.loaded_id})
        return out

    def _load(self, mid: str) -> float:
        if mid == self.loaded_id:
            return 0.0
        self._unload()
        m = self.meta[mid]
        t0 = time.perf_counter()
        if self.backend == "tensorrt":
            self.model = TensorRTModel(m["dir"] / m["engine"], m["input"], int(m["preprocess"].get("size", 224)))
        else:
            self.model = OnnxRuntimeModel(m["dir"] / m["onnx"])
        self.loaded_id = mid
        return (time.perf_counter() - t0) * 1000

    def _unload(self):
        if self.model is None:
            return
        self.model.close()
        self.loaded_id, self.model = None, None
        if self.backend == "tensorrt":
            # The CUDA context outlives the engine and holds memory the shim may need: release it
            # too, so an unused service keeps nothing on the GPU. The next load makes a new one.
            from cuda.bindings import runtime as cudart
            cudart.cudaDeviceReset()

    def load(self, mid: str) -> float:
        with self.lock:
            return self._load(mid)

    def unload(self):
        with self.lock:
            self._unload()

    def classify(self, mid: str, image: Image.Image, want_saliency: bool, topk: int) -> dict:
        m = self.meta[mid]
        t0 = time.perf_counter()
        x = preprocess(image, m["preprocess"])
        with self.lock:
            self._load(mid)
            t1 = time.perf_counter()
            outputs = self.model.run(x)
            t2 = time.perf_counter()
        logits = np.asarray(outputs[m["outputs"]["logits"]], np.float32).reshape(-1)
        if logits.size != len(m["labels"]) or not np.all(np.isfinite(logits)):
            raise RuntimeError("model returned unusable logits")
        ranked = rank(logits, m["labels"], topk)
        saliency = (saliency_grid(outputs[m["outputs"]["saliency"]], m.get("saliency", ""))
                    if want_saliency else None)
        return {"model": mid, "species": ranked[0]["species"], "label": ranked[0]["label"],
                "score": ranked[0]["score"], "topk": ranked, "saliency": saliency, "boxes": [],
                "timing_ms": {"preprocess": round((t1 - t0) * 1000, 2),
                              "inference": round((t2 - t1) * 1000, 2),
                              "total": round((time.perf_counter() - t0) * 1000, 2)}}


def decode_image(text) -> Image.Image:
    if not isinstance(text, str) or not text:
        raise ValueError("image must be base64 text")
    raw = base64.b64decode(text, validate=True)
    if len(raw) > MAX_BODY:
        raise ValueError("image too large")
    image = Image.open(io.BytesIO(raw))
    if image.format not in {"JPEG", "PNG", "WEBP"}:
        raise ValueError("send a JPEG, PNG or WebP image")
    if image.width * image.height > MAX_PIXELS:
        raise ValueError("image has too many pixels")
    image.load()
    return image


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def reply(self, code: int, value: dict):
        body = json.dumps(value).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def do_GET(self):
        classifiers = self.server.classifiers
        if self.path == "/health":
            self.reply(200, {"status": "ok", "loaded": classifiers.loaded_id})
        elif self.path == "/models":
            self.reply(200, {"models": classifiers.describe()})
        else:
            self.reply(404, {"error": "not found"})

    def do_POST(self):
        classifiers = self.server.classifiers
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 <= length <= MAX_BODY * 2:
                self.reply(413, {"error": "request too large"})
                return
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("expected a JSON object")
            if self.path == "/unload":
                classifiers.unload()
                self.reply(200, {"ok": True})
                return
            mid = body.get("model")
            if self.path not in {"/load", "/classify"}:
                self.reply(404, {"error": "not found"})
                return
            if mid not in classifiers.meta:
                self.reply(404, {"error": f"unknown model {str(mid)[:64]!r}"})
                return
            if self.path == "/load":
                self.reply(200, {"ok": True, "load_ms": round(classifiers.load(mid), 1)})
                return
            topk = body.get("topk", 5)
            if type(topk) is not int or not 1 <= topk <= 10:
                raise ValueError("topk must be an integer from 1 to 10")
            want = body.get("saliency", False)
            if not isinstance(want, bool):
                raise ValueError("saliency must be true or false")
            image = decode_image(body.get("image"))
        except (ValueError, TypeError, json.JSONDecodeError, OSError, Image.DecompressionBombError) as exc:
            self.reply(400, {"error": str(exc)[:300]})
            return
        try:
            self.reply(200, classifiers.classify(mid, image, want, topk))
        except Exception as exc:  # a CUDA or TensorRT failure: report it, keep serving
            self.reply(500, {"error": f"{type(exc).__name__}: {str(exc)[:300]}"})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8094)
    parser.add_argument("--models-dir", type=Path, default=Path("/home/orin/nvr/classifier/models"))
    parser.add_argument("--backend", choices=("tensorrt", "onnxruntime"), default="tensorrt")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    server.classifiers = Classifiers(args.models_dir, args.backend)
    names = ", ".join(server.classifiers.meta) or "none"
    print(f"classifiers ({args.backend}): {names} on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.classifiers.unload()


if __name__ == "__main__":
    main()
