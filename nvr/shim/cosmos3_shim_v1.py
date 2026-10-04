#!/usr/bin/env python3
"""OpenAI-compatible shim for Cosmos3-Edge — RESIDENT runtime (v1).

v0 spawned llm_inference as a subprocess per request, reloading ~4.1GB of engines
every time (~13.9s/req). v1 loads LLMRuntime once at startup via the pybind11
bindings and serves every request against the warm, resident model.
"""
import asyncio
import base64
import json
import math
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
from fastapi.responses import JSONResponse, StreamingResponse  # noqa: E402

MODEL_ID = "nvidia/Cosmos3-Edge"
ENGINE_DIR = "/opt/tensorrt-edgellm/models/default"

# The checkpoint that v1/v2/v3 all share. Those are W4A16 builds with their weights
# serialised into llm.engine, so one checkpoint serves all three and this was hardcoded.
SHARED_CHECKPOINT_DIR = "/home/orin/tensorrt-edgellm-workspace/Cosmos3-Edge/onnx/reasoning/llm"

DATA_URL_RE = re.compile(r"^data:(?P<mime>[\w/+.-]+);base64,(?P<data>.+)$", re.DOTALL)

# OpenAI's own ceiling for top_logprobs. The runtime would go to kMaxLogprobsK = 50, but its
# top-K kernel makes one pass over the 131072-entry vocabulary per alternative, on every
# generated token, so the cost grows with K - and no OpenAI client asks for more than 20.
MAX_TOP_LOGPROBS = 20

app = FastAPI()
_lock = asyncio.Lock()
_pool = ThreadPoolExecutor(max_workers=1)
_runtime = None


def _checkpoint_for(engine_dir: str) -> str:
    """Pick the checkpoint belonging to whichever engine is symlinked in right now.

    Hardcoding one checkpoint was fine while every build was W4A16 with its weights inside
    llm.engine. A weight-stripped build is not: its llm.engine is ~16 MB and refits from its
    own model/ directory at load time. Handing such an engine the shared checkpoint aborts in
    externalWeightManager with "missing tensor embed_tokens.weight" - which is misleading,
    because that tensor is present in the build's own checkpoint and simply absent from the
    one it was wrongly given.

    Layout convention: <build>/engines/reasoning is the engine, <build>/model is its
    checkpoint. Builds without that directory keep the shared checkpoint, so v1/v2/v3 are
    unaffected.
    """
    real = os.path.realpath(engine_dir)
    own = os.path.realpath(os.path.join(real, os.pardir, os.pardir, "model"))
    return own if os.path.isdir(own) else SHARED_CHECKPOINT_DIR


def _init_runtime():
    global _runtime
    checkpoint_dir = _checkpoint_for(ENGINE_DIR)
    print(f"[shim] engine     = {os.path.realpath(ENGINE_DIR)}", flush=True)
    print(f"[shim] checkpoint = {checkpoint_dir}", flush=True)
    t0 = time.time()
    _runtime = rt.LLMRuntime(ENGINE_DIR, ENGINE_DIR, {}, checkpoint_dir, rt.ContextCacheConfig())
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


@app.get("/health/ready")
def health_ready():
    """Readiness, as the Live Vision UI defines it.

    That UI refuses to run inference until this returns 200 with status "ready" - it polls
    /health/ready, /v1/models and /api/runtime together and treats a failure of the first two
    as "backend unavailable". This shim only ever served /v1/models and /v1/chat/completions,
    so the UI sat there reporting the backend down while the model was loaded and answering.

    Reports the real thing: _runtime is None until _init_runtime() finishes constructing
    LLMRuntime, which takes ~40-70s after a restart, so 503 during that window is accurate
    rather than a formality.

    /api/runtime is deliberately NOT implemented. The UI tolerates it (null-checked, and
    applyRuntime is try/caught), and answering it means publishing engine_id, an image-token
    limit, a static_clocks flag and an encoder cache size. The first is knowable here; the
    rest are not, and inventing them would push wrong limits into the UI's controls. Missing
    endpoint costs only the image-token presets falling back to defaults.
    """
    if _runtime is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return JSONResponse({"status": "ready"})


def _logprobs_request(body):
    """OpenAI `logprobs` / `top_logprobs` -> (num_logprobs for the runtime, top_logprobs to return).

    (0, None) unless the client asks, and with 0 the request is built exactly as before: the
    runtime only runs its per-token log-softmax and top-K pass when num_logprobs > 0, so
    Frigate, porch-feed and the Live Vision UI keep both their responses and their latency.
    A client that wants no alternatives still costs K = 1, because the chosen token's own
    logprob is read out of that list.
    """
    want, top = body.get("logprobs"), body.get("top_logprobs")
    if not want:
        if top:
            raise ValueError("top_logprobs needs logprobs: true")
        return 0, None
    if want is not True:
        raise ValueError("logprobs must be true or false")
    if top is None:
        top = 0
    if type(top) is not int or not 0 <= top <= MAX_TOP_LOGPROBS:
        raise ValueError(f"top_logprobs must be an integer from 0 to {MAX_TOP_LOGPROBS}")
    return max(1, top), top


def _build_request(messages, max_tokens, temperature, top_p, top_k=50, num_logprobs=0):
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
    if num_logprobs:
        # Only set when asked, so every other request is the one it always was.
        req.num_logprobs = int(num_logprobs)
    return req


def _lp_entry(e):
    """One runtime LogprobEntry -> OpenAI's {token, logprob, bytes}, plus its token_id.

    `piece` is the token's raw bytes, and in this byte-level vocabulary a token can be part of
    a character - "Nidoran♀" tokenizes with the 3-byte "♀" split over two tokens. So `token`
    is decoded with errors="replace" and is for display only; `bytes` is exact, and anything
    that lines tokens up with the answer text has to use `bytes`. Special tokens, such as the
    <|im_end|> that ends every answer, have an empty piece and are told apart by token_id.
    A non-finite logprob becomes -9999.0, OpenAI's own value for "too unlikely to matter":
    FastAPI refuses to serialise -inf, turning a good answer into a 500, and the SSE path
    would write -Infinity, which JSON.parse rejects.
    """
    piece = bytes(e.piece)
    lp = float(e.logprob)
    return {"token": piece.decode("utf-8", "replace"),
            "logprob": lp if math.isfinite(lp) else -9999.0,
            "bytes": list(piece),
            "token_id": int(e.token_id)}


def _lp_content(token_ids, steps, top_logprobs):
    """Runtime logprobs -> OpenAI logprobs.content, one entry per generated token.

    steps[i] is the runtime's top-K list for token_ids[i], best first. Its values are the
    log-softmax of the raw logits - temperature 1, before top-k/top-p, the OpenAI convention -
    so they are the model's own confidence whatever the request decodes with. The chosen token
    is looked up by id rather than assumed to be first: under greedy decoding (temperature 0
    or top_k 1) it always is, but a sampled token can fall outside the top K, and then nothing
    about it is known, so it gets null logprob and null bytes rather than a guess.
    """
    content = []
    for tid, step in zip(token_ids, steps):
        top = [_lp_entry(e) for e in step]
        chosen = next((t for t in top if t["token_id"] == tid), None)
        if chosen is None:
            chosen = {"token": "", "logprob": None, "bytes": None, "token_id": int(tid)}
        content.append({**chosen, "top_logprobs": top[:top_logprobs]})
    return content


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local"}]}


def _sse(payload: dict) -> str:
    return "data: " + json.dumps(payload, separators=(",", ":")) + "\n\n"


async def _stream_completion(req, body, t_start, top_logprobs=None):
    """Server-sent events, for clients that will not accept a single JSON body.

    The Live Vision UI only ever sends stream=true and refuses anything that is not SSE, so
    without this the model would answer correctly and its proxy would discard the result as a
    502. Frigate GenAI and porch-feed send no stream flag and must keep getting exactly the
    body they always got, which is why this is a separate path rather than a rewrite.

    The runtime streams through a StreamChannel attached to the request: handle_request runs
    on the same single-slot pool as before, while this coroutine pops chunks as they are
    produced. wait_pop is blocking, so it goes to the default executor rather than the
    inference pool - putting it on _pool would deadlock against the generation it is waiting on.

    With logprobs asked for, each chunk carries the entries for exactly its own tokens: the
    runtime's chunk.logprobs[i] belongs to chunk.token_ids[i], and a token holding part of a
    character arrives in the same chunk as the token that completes it.
    """
    channel = rt.StreamChannel.create()
    channel.set_stream_interval(1)
    req.stream_channels = [channel]

    cid = "chatcmpl-" + uuid.uuid4().hex
    created = int(time.time())
    head = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID}
    want_usage = bool((body.get("stream_options") or {}).get("include_usage"))

    async def frames():
        loop = asyncio.get_running_loop()
        gen_tok = prompt_tok = 0
        reason = "stop"
        await _lock.acquire()
        try:
            yield _sse({**head, "choices": [{"index": 0, "delta": {"role": "assistant"},
                                             "finish_reason": None}]})
            # The native boundary the UI insists on. Started here, after _build_request has
            # already decoded and loaded the JPEG, so this measures generation and not image
            # work or transport - which is exactly the distinction its comments demand.
            t_native = time.monotonic()
            first_text_at = None
            fut = loop.run_in_executor(_pool, _runtime.handle_request, req)
            while True:
                chunk = await loop.run_in_executor(None, channel.wait_pop, 250)
                if chunk is None:
                    # No chunk within the timeout. Only stop if generation is actually over,
                    # otherwise this is just a slow token and the stream continues.
                    if fut.done() and channel.is_finished():
                        break
                    continue
                if getattr(chunk, "prompt_token_count", 0):
                    prompt_tok = chunk.prompt_token_count
                lp = None
                if top_logprobs is not None:
                    lp = {"content": _lp_content(chunk.token_ids, chunk.logprobs, top_logprobs)}
                if chunk.text:
                    if first_text_at is None:
                        first_text_at = time.monotonic()
                    gen_tok += len(chunk.token_ids) if chunk.token_ids else 1
                    choice = {"index": 0, "delta": {"content": chunk.text}, "finish_reason": None}
                    if lp is not None:
                        choice["logprobs"] = lp
                    yield _sse({**head, "choices": [choice]})
                elif lp and lp["content"]:
                    # Tokens with no text of their own - the <|im_end|> that ends the answer -
                    # still ship, on an empty delta, so the stream's entries add up to exactly
                    # the list a non-streaming request gets.
                    yield _sse({**head, "choices": [{"index": 0, "delta": {}, "logprobs": lp,
                                                     "finish_reason": None}]})
                if chunk.finished:
                    if getattr(chunk, "reason", None) == rt.FinishReason.LENGTH:
                        reason = "length"
                    break
            try:
                await fut          # surface an inference error rather than ending the stream silently
            except Exception as e:
                yield _sse({**head, "choices": [{"index": 0, "delta": {},
                                                 "finish_reason": "error"}],
                            "error": {"message": f"inference failed: {e}"}})
                yield "data: [DONE]\n\n"
                return
            # Live Vision reads data.cosmos_metrics and validates every field before it will
            # display a latency at all - wrong types or a missing key silently render "-".
            # timing_source must say server_monotonic because these come from time.monotonic().
            native_ms = (time.monotonic() - t_native) * 1000.0
            metrics = {"timing_boundary": "native_inference",
                       "timing_source": "server_monotonic",
                       "request_id": cid,
                       "completion_tokens": int(gen_tok),
                       "prompt_tokens": int(prompt_tok),
                       "native_inference_ms": round(native_ms, 3)}
            if first_text_at is not None:
                metrics["first_text_timing_boundary"] = "native_start_to_server_text"
                metrics["server_first_text_ms"] = round((first_text_at - t_native) * 1000.0, 3)
            yield _sse({**head, "cosmos_metrics": metrics,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": reason}]})
            if want_usage:
                yield _sse({**head, "choices": [],
                            "usage": {"prompt_tokens": prompt_tok, "completion_tokens": gen_tok,
                                      "total_tokens": prompt_tok + gen_tok}})
            yield "data: [DONE]\n\n"
            elapsed = (time.time() - t_start) * 1000.0
            mspt = (elapsed / gen_tok) if gen_tok else float("nan")
            # Tagged so logprobs requests, which cost a little more per token, can be told
            # apart in the journal; untagged lines are exactly what they always were.
            lp_tag = f" logprobs={req.num_logprobs}" if top_logprobs is not None else ""
            print(f"[perf] stream elapsed_ms={elapsed:.0f} prompt_tok={prompt_tok} "
                  f"gen_tok={gen_tok} ms_per_tok={mspt:.1f}{lp_tag}", flush=True)
        finally:
            if not channel.is_finished():
                channel.cancel()
            _lock.release()

    return StreamingResponse(frames(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    _t_start = time.time()
    body = await request.json()
    try:
        num_logprobs, top_logprobs = _logprobs_request(body)
        req = _build_request(
            body.get("messages", []),
            body.get("max_tokens", 256),
            body.get("temperature", 0.7),
            body.get("top_p", 0.95),
            body.get("top_k", 50),
            num_logprobs,
        )
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": {"message": f"bad request: {e}"}})

    if body.get("stream"):
        return await _stream_completion(req, body, _t_start, top_logprobs)

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
        _lp_tag = f" logprobs={req.num_logprobs}" if top_logprobs is not None else ""
        print(f"[perf] elapsed_ms={_elapsed_ms:.0f} prompt_tok={_prompt} "
              f"gen_tok={_gen} ms_per_tok={_mspt:.1f}{_lp_tag}", flush=True)
    except Exception as _e:
        print(f"[perf] instrumentation error: {_e}", flush=True)
    reason = "stop"
    try:
        fr = resp.finish_reasons[0]
        reason = "length" if fr == rt.FinishReason.LENGTH else "stop"
    except Exception:
        pass

    choice = {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": reason}
    if top_logprobs is not None:
        # One entry per generated token, the closing <|im_end|> included. Each property read
        # copies the whole nested vector out of C++, so each is read once.
        ids, steps = resp.output_ids, resp.logprobs
        choice["logprobs"] = {"content": _lp_content(ids[0] if ids else [],
                                                     steps[0] if steps else [], top_logprobs)}

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [choice],
        "usage": {
            "prompt_tokens": _prompt,
            "completion_tokens": _gen,
            "total_tokens": _prompt + _gen,
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
