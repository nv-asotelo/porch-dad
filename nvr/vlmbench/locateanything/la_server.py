#!/usr/bin/env python3
"""LocateAnything-3B (INT4) behind an OpenAI-compatible chat API, for the shim's proxy mode.

LocateAnything answers category queries with boxes; it does not write prose. So a request's text is
read as a query: a short phrase ("Bernese mountain dog", "person</c>dog") is the query itself, and
"find / locate / detect / where is X" asks for X. Anything else - Frigate's "Describe only what is
visible..." for one - falls back to people, pets and vehicles, so a caller that expects a sentence
still gets a truthful one-line count. The answer is that line plus the boxes as JSON
({"label", "bbox_2d"} on the 0-1000 grid, x first), the format Live Vision's overlay reads.

  python la_server.py --model <dir> --weights <int4.safetensors> --port 8092
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

DEFAULT_QUERY = "person</c>dog</c>cat</c>car</c>truck"
ASK = re.compile(r"^\s*(?:please\s+)?(?:locate|find|detect|point (?:to|at)|show me|where (?:is|are))\s+"
                 r"(?:the |a |an |all |every |any )?(.+?)\s*[.?!]*\s*$", re.I)
DATA_URL = re.compile(r"^data:[\w/+.-]+;base64,(.+)$", re.S)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8092)
    args = ap.parse_args()

    os.environ.setdefault("LA_FLASH_MODEL", args.model)
    os.environ.setdefault("LA_FLASH_ATTN", "sdpa")
    sys.path.insert(0, args.model)
    import torch
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from PIL import Image
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

    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": "nvidia/LocateAnything-3B", "object": "model", "owned_by": "local"}]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        image, text = None, ""
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
                    url = (c.get("image_url") or {}).get("url", "")
                    d = DATA_URL.match(url)
                    image = Image.open(io.BytesIO(base64.b64decode(d.group(1))) if d else url).convert("RGB")
        if image is None:
            return JSONResponse(status_code=400, content={"error": {"message": "LocateAnything needs an image"}})
        query = query_for(text)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        raw = generate_batch_hybrid([(image, query)], temperature=0.0, top_p=0.9, top_k=None,
                                    repetition_penalty=1.1, max_new_tokens=512)[0]
        torch.cuda.synchronize()
        found = parse(raw)
        answer = summary(found, query) + "\n" + json.dumps(found)
        print(f"[la] query={query!r} boxes={len(found)} ms={(time.perf_counter() - t0) * 1000:.0f}", flush=True)
        usage = {"prompt_tokens": 0, "completion_tokens": len(found) * 6 + 1, "total_tokens": len(found) * 6 + 1}
        cid = "chatcmpl-" + uuid.uuid4().hex
        if not body.get("stream"):
            return {"id": cid, "object": "chat.completion", "created": int(time.time()), "model": "nvidia/LocateAnything-3B",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
                    "usage": usage}

        def frames():
            head = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                    "model": "nvidia/LocateAnything-3B"}
            yield "data: " + json.dumps({**head, "choices": [{"index": 0, "delta": {"content": answer}, "finish_reason": None}]}) + "\n\n"
            yield "data: " + json.dumps({**head, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}) + "\n\n"
            yield "data: " + json.dumps({**head, "choices": [], "usage": usage}) + "\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(frames(), media_type="text/event-stream")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
