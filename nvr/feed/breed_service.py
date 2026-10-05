#!/usr/bin/env python3
"""Breed scores for the dogs Frigate is tracking: a loopback service for detection modes.

Frigate's detector finds each dog and its box; this scores every box with an ImageNet classifier
(google/vit-base-patch16-224, INT8, on the CPU through onnxruntime) and returns the probability
that it belongs to a set of classes - for the Bernese mountain dog mode, the four Swiss mountain
dogs (Bernese, EntleBucher, Appenzeller, Greater Swiss), which on a blurry crop are one look-alike
family. Measured on 332 dog crops of the doggy-daycare feed: the Bernese scored 0.70-0.99 in all
four frames it was in, and no other dog scored above 0.09. Neither 2B VLM came close, and this runs
whatever VLM the shim has loaded.

CPU, two threads, its own venv: no CUDA context beside the VLM, and nothing added to the
TensorRT-Edge-LLM environment the shim and Live Vision share. The model loads on the first request
and is dropped after --idle-unload seconds without one: resident it costs ~226 MB, and a mode looks
at most once per camera every few minutes, so most of the day it holds next to nothing.

  GET  /health   {"status": "ok", "model": name}
  POST /score    {"image": base64 JPEG, "boxes": [[x1, y1, x2, y2] in 0-1, ...], "classes": [names]}
                 -> {"scores": [P(any of classes) per box], "top": [[label, p] per box], "ms": n}

  python breed_service.py --model vit-b16-imagenet-int8.onnx --meta vit-b16-imagenet.json --port 8107
"""
import argparse
import base64
import ctypes
import gc
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import onnxruntime as ort
from PIL import Image

MAX_BODY = 8 * 1024 * 1024
MAX_BOXES = 32


class Scorer:
    def __init__(self, model, meta, threads=2, idle_unload=900):
        self.model, self.threads, self.idle_unload = model, threads, idle_unload
        self.meta = json.load(open(meta))
        self.labels = self.meta["labels"]
        self.mean = np.array(self.meta["mean"], np.float32)
        self.std = np.array(self.meta["std"], np.float32)
        self.size = int(self.meta["size"])
        self.lock = threading.Lock()          # one inference at a time: the CPU is Frigate's too
        self.session, self.last_used = None, 0.0
        threading.Thread(target=self._unloader, daemon=True).start()

    def _load(self):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self.threads
        opts.inter_op_num_threads = 1
        # No arena: what an inference allocates goes back when it ends, not into a pool kept for later.
        opts.enable_cpu_mem_arena = False
        self.session = ort.InferenceSession(self.model, opts, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def _unloader(self):
        while True:
            time.sleep(30)
            with self.lock:
                if self.session is not None and time.time() - self.last_used > self.idle_unload:
                    self.session = None
                    gc.collect()
                    try:                          # hand freed heap pages back to the kernel
                        ctypes.CDLL("libc.so.6").malloc_trim(0)
                    except OSError:
                        pass
                    print("[breed] idle: model unloaded", flush=True)

    def crop(self, image, box):
        """The box with a 10% margin, as the model's input tensor."""
        w, h = image.size
        x1, y1, x2, y2 = box[0] * w, box[1] * h, box[2] * w, box[3] * h
        m = 0.1 * max(x2 - x1, y2 - y1)
        c = image.crop((int(max(0, x1 - m)), int(max(0, y1 - m)), int(min(w, x2 + m)), int(min(h, y2 + m))))
        a = np.asarray(c.resize((self.size, self.size), Image.BILINEAR), np.float32) / 255
        return ((a - self.mean) / self.std).transpose(2, 0, 1)

    def score(self, image, boxes, classes):
        idx = [i for i, label in enumerate(self.labels) if label.split(",")[0] in set(classes)]
        if not idx:
            raise ValueError("none of those classes is an ImageNet label")
        if not boxes:
            return [], []
        x = np.stack([self.crop(image, b) for b in boxes])
        with self.lock:
            if self.session is None:
                self._load()
            logits = np.concatenate([self.session.run(None, {self.input: x[i:i + 1]})[0] for i in range(len(x))])
            self.last_used = time.time()
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        top = [[self.labels[int(k)].split(",")[0], round(float(row[k]), 3)] for row, k in zip(p, p.argmax(1))]
        return [round(float(row[idx].sum()), 4) for row in p], top


class Handler(BaseHTTPRequestHandler):
    def reply(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            scorer = self.server.scorer
            self.reply(200, {"status": "ok", "model": scorer.meta.get("source", ""), "loaded": scorer.session is not None})
        else:
            self.reply(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/score":
            self.reply(404, {"error": "not found"})
            return
        n = int(self.headers.get("Content-Length") or 0)
        if not 0 < n <= MAX_BODY:
            self.reply(413, {"error": "body too large or empty"})
            return
        try:
            body = json.loads(self.rfile.read(n))
            image = Image.open(io.BytesIO(base64.b64decode(body["image"]))).convert("RGB")
            boxes = [[min(max(float(v), 0.0), 1.0) for v in b] for b in body.get("boxes") or []][:MAX_BOXES]
            if any(len(b) != 4 or b[2] <= b[0] or b[3] <= b[1] for b in boxes):
                raise ValueError("boxes must be [x1, y1, x2, y2] in 0-1")
            t0 = time.perf_counter()
            scores, top = self.server.scorer.score(image, boxes, body.get("classes") or [])
        except (KeyError, ValueError, TypeError, OSError) as e:
            self.reply(400, {"error": f"{type(e).__name__}: {e}"})
            return
        self.reply(200, {"scores": scores, "top": top, "ms": round((time.perf_counter() - t0) * 1000, 1)})

    def log_message(self, *args):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8107)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--idle-unload", type=float, default=900, help="seconds without a request before the model is dropped")
    args = ap.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.scorer = Scorer(args.model, args.meta, args.threads, args.idle_unload)
    print(f"[breed] {args.model} on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
