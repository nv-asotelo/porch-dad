#!/usr/bin/env python3
"""LocateAnything-3B (INT4) behind an OpenAI-compatible chat API, for the shim's proxy mode.

LocateAnything answers category queries with boxes; it does not write prose. So a request's text is
read as a query: a short phrase ("Bernese mountain dog", "person</c>dog") is the query itself, and
"find / locate / detect / where is X" asks for X. Anything else - Frigate's "Describe only what is
visible..." for one - falls back to people, pets and vehicles, so a caller that expects a sentence
still gets a truthful one-line count. The answer is that line plus the boxes as JSON
({"label", "bbox_2d"} on the 0-1000 grid, x first), the format Live Vision's overlay reads.

Two backends answer the query:
  python la_server.py --model <dir> --weights <int4.safetensors> --port 8105   PyTorch, INT4 (la_int4.py)
  python la_server.py --upstream http://127.0.0.1:8091 --port 8105            llama.cpp (PR #24749)
The second is what fits beside the NVR on the Orin: llama-server, started with --special, decodes
the boxes one token at a time and has none of PyTorch's 0.9 GB CUDA context.
"""
import argparse
import base64
import io
import json
import os
import re
import sys
import time
import uuid

# Not PYTORCH_CUDA_ALLOC_CONF=expandable_segments: on Jetson the first allocation then fails
# (PyTorch asks NVML for GPU fabric info, which the integrated GPU does not have).

DEFAULT_QUERY = "person</c>dog</c>cat</c>car</c>truck"
# LocateAnything's own chat wrapper and query wording (processing_locateanything.py,
# batch_utils/hybrid_runtime.py), which an OpenAI-style server has to be sent verbatim.
SYSTEM = "You are a helpful assistant.\n"
LOCATE = "Locate all the instances that matches the following description: "
ASK = re.compile(r"^\s*(?:please\s+)?(?:locate|find|detect|point (?:to|at)|show me|where (?:is|are))\s+"
                 r"(?:the |a |an |all |every |any )?(.+?)\s*[.?!]*\s*$", re.I)
DATA_URL = re.compile(r"^data:[\w/+.-]+;base64,(.+)$", re.S)


def fit_side(im, max_side):
    """Shrink so the long side is at most max_side px (LA_MAX_SIDE): MoonViT runs at native resolution,
    and on the Orin its activations for a 960x540 frame do not fit beside the NVR."""
    if not max_side or max(im.size) <= max_side:
        return im
    s = max_side / max(im.size)
    return im.resize((max(14, round(im.size[0] * s)), max(14, round(im.size[1] * s))))


def query_for(text):
    t = (text or "").strip()
    if "</c>" in t:
        return t.rstrip(".?! ")
    m = ASK.match(t)
    if m:
        return m.group(1)
    if t and len(t.split()) <= 6 and "?" not in t:
        return t.rstrip(".! ")
    return DEFAULT_QUERY


def parse(text):
    """'<ref>dog</ref><box><1><2><3><4></box>...' -> [{"label", "bbox_2d"}] on 0-1000."""
    out = []
    for label, boxes in re.findall(r"<ref>(.*?)</ref>((?:<box>.*?</box>)+)", text or ""):
        for b in re.findall(r"<box>(.*?)</box>", boxes):
            nums = [int(n) for n in re.findall(r"<(\d+)>", b)]
            if len(nums) == 4:
                out.append({"label": label, "bbox_2d": nums})
            elif len(nums) == 2:
                out.append({"label": label, "point_2d": nums})
    return out


def summary(found, query):
    labels = [q.strip() for q in query.split("</c>") if q.strip()]
    counts = {label: sum(1 for f in found if f["label"] == label) for label in labels}
    if not any(counts.values()):
        return "None found: " + ", ".join(labels) + "."
    return ", ".join(f"{n} {label}" for label, n in counts.items() if n) + "."


def upstream_locate(upstream, image_url, query, timeout=180):
    """One query to a llama.cpp server running LocateAnything: its raw <ref>/<box> answer."""
    import urllib.request
    body = {"model": "local", "temperature": 0, "max_tokens": 512,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url}},
                                                      {"type": "text", "text": LOCATE + query}]}]}
    req = urllib.request.Request(f"{upstream.rstrip('/')}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["choices"][0]["message"]["content"] or ""


def upstream_ready(upstream):
    import urllib.request
    try:
        with urllib.request.urlopen(f"{upstream.rstrip('/')}/health", timeout=2) as r:
            return r.status == 200
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--weights")
    ap.add_argument("--upstream", help="llama.cpp server with LocateAnything loaded, instead of PyTorch")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8105)
    args = ap.parse_args()
    if not args.upstream and not (args.model and args.weights):
        ap.error("give --upstream, or --model and --weights")
    if args.upstream:
        serve(args, None)
        return

    os.environ.setdefault("LA_FLASH_MODEL", args.model)
    os.environ.setdefault("LA_FLASH_ATTN", "sdpa")
    sys.path.insert(0, args.model)
    # This directory's la_int4 wins over any copy that ended up beside the model.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import torch
    from transformers import AutoProcessor, AutoTokenizer
    from batch_utils import generate_batch_hybrid
    from batch_utils import hybrid_runtime as hr
    from la_int4 import load_quantized

    hr._tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    hr._proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    hr._model = load_quantized(args.model, args.weights)
    hr._set_vision_attention_mode(hr._model)
    hr._set_llm_mode(hr._model, "sdpa")
    print("[la] model loaded", flush=True)

    def local(image, query):
        torch.cuda.synchronize()
        raw = generate_batch_hybrid([(image, query)], temperature=0.0, top_p=0.9, top_k=None,
                                    repetition_penalty=1.1, max_new_tokens=512)[0]
        torch.cuda.synchronize()
        return raw

    serve(args, local)


def serve(args, local):
    """The OpenAI-compatible front: local(image, query) runs PyTorch; without it, args.upstream."""
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from PIL import Image

    app = FastAPI()
    model_name = "nvidia/LocateAnything-3B"

    @app.get("/health")
    def health():
        # In upstream mode the shim waits on this, so it must not say ok before llama.cpp does.
        if local is None and not upstream_ready(args.upstream):
            return JSONResponse(status_code=503, content={"status": "loading"})
        return {"status": "ok"}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "local"}]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        image_url, text = None, ""
        for m in body.get("messages") or []:
            if m.get("role") != "user":
                continue
            content = m.get("content")
            if isinstance(content, str):
                text = content
                continue
            for c in content or []:
                if c.get("type") == "text":
                    text = c.get("text") or text
                elif c.get("type") == "image_url":
                    image_url = (c.get("image_url") or {}).get("url", "")
        if not image_url:
            return JSONResponse(status_code=400, content={"error": {"message": "LocateAnything needs an image"}})
        query = query_for(text)
        t0 = time.perf_counter()
        if local is None:
            raw = upstream_locate(args.upstream, image_url, query)
        else:
            d = DATA_URL.match(image_url)
            image = Image.open(io.BytesIO(base64.b64decode(d.group(1))) if d else image_url).convert("RGB")
            raw = local(fit_side(image, int(os.environ.get("LA_MAX_SIDE", "0"))), query)
        found = parse(raw)
        answer = summary(found, query) + "\n" + json.dumps(found)
        print(f"[la] query={query!r} boxes={len(found)} ms={(time.perf_counter() - t0) * 1000:.0f}", flush=True)
        usage = {"prompt_tokens": 0, "completion_tokens": len(found) * 6 + 1, "total_tokens": len(found) * 6 + 1}
        cid = "chatcmpl-" + uuid.uuid4().hex
        if not body.get("stream"):
            return {"id": cid, "object": "chat.completion", "created": int(time.time()), "model": model_name,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
                    "usage": usage}

        def frames():
            head = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()), "model": model_name}
            yield "data: " + json.dumps({**head, "choices": [{"index": 0, "delta": {"content": answer}, "finish_reason": None}]}) + "\n\n"
            yield "data: " + json.dumps({**head, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}) + "\n\n"
            yield "data: " + json.dumps({**head, "choices": [], "usage": usage}) + "\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(frames(), media_type="text/event-stream")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
