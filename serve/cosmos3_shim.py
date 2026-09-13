#!/usr/bin/env python3
"""OpenAI-compatible shim for Cosmos3-Edge — RESIDENT runtime (v1).

v0 spawned llm_inference as a subprocess per request, reloading ~4.1GB of engines
every time (~13.9s/req). v1 loads LLMRuntime once at startup via the pybind11
bindings and serves every request against the warm, resident model.
"""
import asyncio
import base64
import os
import re
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault(
    "EDGELLM_PLUGIN_PATH", "/home/orin/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so"
)
sys.path.insert(0, "/home/orin/TensorRT-Edge-LLM/build/pybind")

import _edgellm_runtime as rt  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

MODEL_ID = "nvidia/Cosmos3-Edge"
ENGINE_DIR = "/opt/tensorrt-edgellm/models/default"
CHECKPOINT_DIR = "/home/orin/tensorrt-edgellm-workspace/Cosmos3-Edge/onnx/reasoning/llm"

DATA_URL_RE = re.compile(r"^data:(?P<mime>[\w/+.-]+);base64,(?P<data>.+)$", re.DOTALL)

app = FastAPI()
_lock = asyncio.Lock()
_pool = ThreadPoolExecutor(max_workers=1)
_runtime = None


def _init_runtime():
    global _runtime
    t0 = time.time()
    _runtime = rt.LLMRuntime(ENGINE_DIR, ENGINE_DIR, {}, CHECKPOINT_DIR, rt.ContextCacheConfig())
    print(f"[shim] LLMRuntime constructed in {time.time()-t0:.2f}s", flush=True)
    t0 = time.time()
    _runtime.capture_decoding_cuda_graph()
    print(f"[shim] CUDA graphs captured in {time.time()-t0:.2f}s", flush=True)
    # Warm-up so the first real request doesn't pay first-call cost.
    try:
        t0 = time.time()
        inner = rt.Request([rt.Message("user", [rt.MessageContent("text", "hi")])])
        req = rt.LLMGenerationRequest()
        req.requests = [inner]
        req.max_generate_length = 4
        _runtime.handle_request(req)
        print(f"[shim] warm-up inference in {time.time()-t0:.2f}s", flush=True)
    except Exception as e:  # warm-up is best-effort
        print(f"[shim] warm-up skipped: {e}", flush=True)
    print("[shim] ready", flush=True)


@app.on_event("startup")
def _startup():
    _init_runtime()


def _build_request(messages, max_tokens, temperature, top_p, top_k=50):
    """Translate OpenAI-style messages into an LLMGenerationRequest."""
    images = []
    rt_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content")
        if isinstance(content, str):
            rt_messages.append(rt.Message(role, [rt.MessageContent("text", content)]))
            continue
        parts = []
        for item in content or []:
            itype = item.get("type")
            if itype == "text":
                parts.append(rt.MessageContent("text", item.get("text", "")))
            elif itype == "image_url":
                url = (item.get("image_url") or {}).get("url", "")
                m = DATA_URL_RE.match(url)
                if m:
                    raw = base64.b64decode(m.group("data"))
                    images.append(rt.load_image_from_bytes(raw))
                    parts.append(rt.MessageContent("image", "inline"))
                else:
                    images.append(rt.load_image_from_path(url))
                    parts.append(rt.MessageContent("image", url))
        rt_messages.append(rt.Message(role, parts))

    inner = rt.Request(rt_messages)
    if images:
        inner.image_buffers = images
    req = rt.LLMGenerationRequest()
    req.requests = [inner]
    req.max_generate_length = int(max_tokens)
    req.temperature = float(temperature)
    req.top_p = float(top_p)
    req.top_k = int(top_k)
    return req


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local"}]}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    _t_start = time.time()
    body = await request.json()
    try:
        req = _build_request(
            body.get("messages", []),
            body.get("max_tokens", 256),
            body.get("temperature", 0.7),
            body.get("top_p", 0.95),
            body.get("top_k", 50),
        )
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": {"message": f"bad request: {e}"}})

    loop = asyncio.get_running_loop()
    async with _lock:
        try:
            resp = await loop.run_in_executor(_pool, _runtime.handle_request, req)
        except Exception as e:
            return JSONResponse(status_code=502, content={"error": {"message": f"inference failed: {e}"}})

    text = resp.output_texts[0] if resp.output_texts else ""
    # Token counts feed both the [perf] log line and the OpenAI `usage` field, so they are
    # computed outside the instrumentation try-block.
    _gen = len(resp.output_ids[0]) if resp.output_ids else 0
    _prompt = resp.prompt_token_counts[0] if resp.prompt_token_counts else 0
    # --- latency instrumentation: normalize by generated tokens ---
    try:
        _elapsed_ms = (time.time() - _t_start) * 1000.0
        _mspt = (_elapsed_ms / _gen) if _gen else float("nan")
        print(f"[perf] elapsed_ms={_elapsed_ms:.0f} prompt_tok={_prompt} "
              f"gen_tok={_gen} ms_per_tok={_mspt:.1f}", flush=True)
    except Exception as _e:
        print(f"[perf] instrumentation error: {_e}", flush=True)
    reason = "stop"
    try:
        fr = resp.finish_reasons[0]
        reason = "length" if fr == rt.FinishReason.LENGTH else "stop"
    except Exception:
        pass

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": reason}
        ],
        "usage": {
            "prompt_tokens": _prompt,
            "completion_tokens": _gen,
            "total_tokens": _prompt + _gen,
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
