#!/usr/bin/env python3
"""Task-owned request controls and native-only timing for the pinned public server.

The existing native engine retains its built capacity. Each decoded image carries
an independently validated runtime token budget. Timing surrounds only the native
inference call: JPEG decode, request preparation/admission and response transport
are outside that interval. No client clock is used to score inference.
"""
from __future__ import annotations

from contextvars import ContextVar
from datetime import datetime, timezone
import hashlib
import copy
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
from pathlib import Path
import threading
import time
import types
import uuid

ROOT = Path(__file__).resolve().parents[1]
ACTIVE_TRACE = ContextVar('cosmos_request_trace', default=None)


def image_budget(value, limit=512, minimum=4):
    if type(value) is not int or not minimum <= value <= limit:
        raise ValueError(f'Image token budget must be an integer from {minimum} to {limit}.')
    return value


def sampling_top_p(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError('top_p must be greater than 0 and at most 1.')
    return value


def request_controls(payload, default_budget, limit, minimum=4, default_top_p=1.0):
    """Consume original extension fields before upstream schema validation."""
    body = dict(payload)
    budget = image_budget(body.pop('max_image_tokens_per_image', default_budget), limit, minimum)
    body['top_p'] = sampling_top_p(body.get('top_p', default_top_p))
    benchmark = body.pop('cosmos_benchmark', {})
    if (not isinstance(benchmark, dict) or set(benchmark) - {'warmup', 'run_id', 'cache_mode'}
            or type(benchmark.get('warmup', False)) is not bool
            or not isinstance(benchmark.get('run_id', ''), str)
            or len(benchmark.get('run_id', '')) > 100
            or benchmark.get('cache_mode', 'default') not in ('default', 'bypass')):
        raise ValueError('Invalid benchmark annotation.')
    return body, budget, benchmark


def content_hashes(messages):
    # Store content hashes, never images, prompts, passwords or output text.
    images, prompt = [], copy.deepcopy(messages)
    for message in prompt:
        content = message.get('content', [])
        if isinstance(content, str):
            continue
        if not isinstance(content, list) or not all(isinstance(item, dict) for item in content):
            raise ValueError('Message content must be text or a list of content objects.')
        for item in content:
            if item.get('type') == 'image_url':
                image = item.get('image_url')
                if not isinstance(image, dict) or not isinstance(image.get('url'), str):
                    raise ValueError('Expected image_url.url.')
                images.append(image['url'])
                image['url'] = '<image>'
    digest = lambda value: hashlib.sha256(json.dumps(value, ensure_ascii=False,
                                    separators=(',', ':'), sort_keys=True).encode()).hexdigest()
    return digest(images), digest(prompt), len(images)



class _FirstTextIterator:
    """Observe server text without adding a queue or changing cancellation."""
    def __init__(self, source, timer, trace):
        self.source, self.timer, self.trace = source, timer, trace

    def __iter__(self):
        return self

    def __next__(self):
        try:
            delta = next(self.source)
        except BaseException:
            self.timer.stream_closed(self.trace)
            raise
        # Match the pinned no-reasoning-parser HTTP path and historical first
        # nonempty-text definition, including whitespace but excluding markers.
        if getattr(delta, 'text', '').replace('<|im_end|>', ''):
            started = self.trace.get('_native_started_ns')
            if started is not None and 'server_first_text_ms' not in self.trace:
                self.trace.update(server_first_text_ms=(self.timer.clock() - started) / 1e6,
                                  first_text_timing_boundary='native_start_to_server_text')
                self.timer.publish(self.trace)
        return delta

    def close(self):
        # Forward directly even while another thread is blocked in __next__.
        # A generator-only wrapper would lose this upstream cancellation signal.
        try:
            self.source.close()
        finally:
            self.timer.stream_closed(self.trace)


class NativeTimer:
    """Instrument this LLM instance, preserving its admission and cancellation.

    Request preparation runs in an asyncio worker which propagates ContextVars.
    The native generation thread uses the prepared request's identity instead of
    a global current-request value. No timer spans a network yield.
    """
    def __init__(self, llm, runtime, emit, clock=time.perf_counter_ns):
        self.llm, self.runtime, self.emit, self.clock = llm, runtime, emit, clock
        self.pending = {}
        self.lock = threading.Lock()
        original_make, original_handle = llm._make_generation_request, llm._handle_request

        def make(_llm, *args, **kwargs):
            request = original_make(*args, **kwargs)  # JPEG decoding is here, before the timer.
            trace = ACTIVE_TRACE.get()
            if trace is not None:
                rows = request.requests
                for row in rows:
                    images = row.image_buffers
                    for image in images:
                        image.max_image_tokens_per_image = trace['controls']['max_image_tokens_per_image']
                        image.skip_encoder_cache = trace['controls'].get('encoder_cache_mode') == 'bypass'
                    row.image_buffers = images
                request.requests = rows  # pybind's STL getters return copies.
                with self.lock:
                    self.pending[id(request)] = trace
            return request

        def handle(_llm, request):
            with self.lock:
                trace = self.pending.pop(id(request), None)
            if trace is None:
                return original_handle(request)
            elapsed = None
            try:
                with llm._infer_guard():
                    llm._ensure_open()
                    before = self.observed_images()
                    started = self.clock()
                    trace['_native_started_ns'] = started
                    try:
                        response = llm._runtime.handle_request(request)
                    finally:
                        elapsed = (self.clock() - started) / 1e6
                    after = self.observed_images()
            except BaseException:
                trace.update(status='error', server_elapsed_ms=elapsed)
                self.publish(trace)
                raise
            if getattr(llm, 'context_cache_enabled', False):
                llm._log_context_reuse_metrics()
            reasons = getattr(response, 'finish_reasons', [])
            from experimental.server.runtime.engine import finish_reason_name
            reason = finish_reason_name(llm._rt, reasons[0]) if reasons else None
            ids = getattr(response, 'output_ids', [])
            prompts = getattr(response, 'prompt_token_counts', [])
            cache = 'unknown'
            if trace['controls']['encoder_cache_bytes'] == 0 or trace['controls'].get('encoder_cache_mode') == 'bypass':
                cache = 'disabled'
            elif trace['image_count'] == 1 and before is not None and after is not None:
                cache = 'miss' if after[0] > before[0] else 'hit'
            trace.update(status='completed' if reason in ('stop', 'length') else 'canceled',
                         server_elapsed_ms=elapsed, native_inference_ms=elapsed,
                         completion_tokens=len(ids[0]) if ids else 0,
                         prompt_tokens=prompts[0] if prompts else None,
                         finish_reason=reason, cache_state=cache,
                         observed_image_tokens=(after[1] - before[1]) if before and after and after[0] > before[0] else None)
            self.publish(trace)
            return response

        llm._make_generation_request = types.MethodType(make, llm)
        llm._handle_request = types.MethodType(handle, llm)
        if hasattr(llm, 'generate_stream'):
            original_stream = llm.generate_stream

            def generate_stream(_llm, *args, **kwargs):
                request = kwargs.get('prebuilt_request')
                with self.lock:
                    trace = self.pending.get(id(request))
                source = original_stream(*args, **kwargs)
                if trace is None:
                    return source
                trace['_streaming'] = True
                return _FirstTextIterator(source, self, trace)

            llm.generate_stream = types.MethodType(generate_stream, llm)

    def publish(self, trace):
        # A short generation may finish before Python observes its first chunk.
        # Preserve the native duration, but wait for text (or stream closure)
        # before logging once. No consumer delay is folded into native total.
        with self.lock:
            if trace.get('_logged') or trace.get('status') not in ('completed', 'error', 'canceled'):
                return
            if trace.get('_streaming') and not (trace.get('_stream_closed') or 'server_first_text_ms' in trace):
                return
            trace['_logged'] = True
            record = {key: value for key, value in trace.items() if not key.startswith('_')}
        self.emit(record)

    def stream_closed(self, trace):
        trace['_stream_closed'] = True
        self.publish(trace)

    def observed_images(self):
        try:
            metrics = self.llm._runtime.get_multimodal_metrics()
            return int(metrics.observed_image_runs), int(metrics.observed_image_tokens)
        except (AttributeError, RuntimeError):
            return None

    def discard_trace(self, trace):
        with self.lock:
            for key in [key for key, value in self.pending.items() if value is trace]:
                del self.pending[key]

    def discard(self, request):
        with self.lock:
            self.pending.pop(id(request), None)


def create_app(client, config, *, log_path=None):
    from fastapi import APIRouter
    from experimental.server.api.app import create_app as upstream_app
    from experimental.server.api.routes import _ReleasingStreamingResponse, router as upstream_router
    from experimental.server.api.protocol import ChatCompletionRequest
    from experimental.server.api.errors import InvalidRequestError
    from fastapi.responses import JSONResponse
    from pydantic import ValidationError

    app = upstream_app(client, config)
    llm = client.llm
    builder = llm._visual_config().get('builder_config', {})
    limit = int(builder.get('max_image_tokens_per_image', 512))
    minimum = int(builder.get('min_image_tokens', 4))
    default = image_budget(int(os.environ.get('COSMOS_MAX_IMAGE_TOKENS_PER_IMAGE', '512')), limit, minimum)
    default_top_p = sampling_top_p(float(os.environ.get('COSMOS_TOP_P', '1')))
    if not hasattr(llm._rt.ImageData(), 'max_image_tokens_per_image'):
        raise RuntimeError('Runtime image-budget binding missing; apply the task patch and rebuild _edgellm_runtime.')
    if not hasattr(llm._rt.ImageData(), 'skip_encoder_cache'):
        raise RuntimeError('Runtime encoder-cache bypass binding missing; apply the task patch and rebuild _edgellm_runtime.')
    engine_id = uuid.uuid4().hex
    cache_bytes = llm._context_cache_config.encoder_embedding_cache_budget_bytes
    runtime = dict(engine_id=engine_id, max_image_tokens_per_image=default,
                   max_image_tokens_per_image_limit=limit, max_image_tokens_per_image_min=minimum,
                   encoder_cache_bytes=cache_bytes, static_clocks=os.environ.get('COSMOS_STATIC_CLOCKS') == '1',
                   top_p=default_top_p, timing_boundary='native_inference', timing_source='server_monotonic')
    target = Path(log_path or os.environ.get('COSMOS_REQUEST_LOG', str(ROOT / 'data/logs/native-requests.jsonl')))
    target.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('cosmos.requests.' + engine_id)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    log_handler = RotatingFileHandler(target, maxBytes=10 << 20, backupCount=3)
    log_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(log_handler)

    def emit(trace):
        logger.info(json.dumps(trace, separators=(',', ':'), allow_nan=False))

    timer = NativeTimer(llm, runtime, emit)
    app.state.cosmos_runtime = runtime
    app.state.cosmos_timer = timer
    # Retain upstream's health, media policy, authentication and lifecycle. Only
    # this chat route consumes the task's two validated extension fields.
    def is_chat(route):
        return (getattr(route, 'path', None) == '/v1/chat/completions'
                and 'POST' in getattr(route, 'methods', set()))

    # Newer FastAPI keeps include_router lazy, so its app-level entry has no
    # path. Replace that inclusion with a task-local router instead of leaving
    # the upstream schema first in dispatch order or mutating its global router.
    lazy_inclusions = [route for route in app.router.routes
                       if getattr(route, 'original_router', None) is upstream_router]
    app.router.routes[:] = [route for route in app.router.routes
                           if route not in lazy_inclusions and not is_chat(route)]
    if lazy_inclusions:
        retained = APIRouter()
        retained.routes.extend(route for route in upstream_router.routes if not is_chat(route))
        app.include_router(retained)

    @app.get('/api/runtime')
    async def settings():
        return runtime

    @app.post('/v1/chat/completions')
    async def chat(payload: dict):
        try:
            body, budget, benchmark = request_controls(payload, default, limit, minimum, default_top_p)
            request = ChatCompletionRequest.model_validate(body)
        except (ValueError, TypeError, ValidationError) as exc:
            raise InvalidRequestError(str(exc)) from exc
        try:
            image_hash, prompt_hash, image_count = content_hashes(request.messages)
        except (ValueError, TypeError, AttributeError) as exc:
            raise InvalidRequestError(str(exc)) from exc
        handler = app.state.openai_serving_chat
        prepared = handler.prepare_request(request)
        trace = dict(kind='server_request', schema_version=1, request_id=uuid.uuid4().hex,
                     at_utc=datetime.now(timezone.utc).isoformat(), engine_id=engine_id,
                     backend_id=os.environ.get('COSMOS_BACKEND_ID', os.environ.get('COSMOS_PROFILE', 'tensorrt-edge-llm')),
                     status='preparing', _streaming=request.stream,
                     timing_source='server_monotonic', timing_boundary='native_inference',
                     warmup=benchmark.get('warmup', False), run_id=benchmark.get('run_id', ''),
                     image_count=image_count, requested_max_tokens=request.effective_max_tokens,
                     controls=dict(image_sha256=image_hash, prompt_sha256=prompt_hash,
                         max_image_tokens_per_image=budget, temperature=prepared.sampling.temperature,
                         top_p=prepared.sampling.top_p, top_k=prepared.sampling.top_k, concurrency=1,
                         clock_policy='static' if runtime['static_clocks'] else 'dynamic',
                         encoder_cache_mode=benchmark.get('cache_mode', 'default'),
                         encoder_cache_bytes=cache_bytes, text_context_reuse=client.capabilities.context_reuse))
        token = ACTIVE_TRACE.set(trace)
        try:
            if not request.stream:
                result = await handler.create_chat_completion(request)
                data = result.model_dump(exclude_none=True)
                data['cosmos_metrics'] = public_metrics(trace)
                return JSONResponse(data)
            engine_request = await handler.prepare_engine_request(request, prepared)
        except BaseException:
            timer.discard_trace(trace)
            raise
        finally:
            ACTIVE_TRACE.reset(token)

        release = engine_request.release
        def release_request():
            timer.discard(engine_request.request)
            release()
        engine_request.release = release_request

        async def stream():
            try:
                async for chunk in handler.stream_chat_completion(request, prepared, engine_request):
                    if chunk.strip() == 'data: [DONE]' and trace.get('status') == 'completed':
                        yield 'data: ' + json.dumps({'choices': [], 'cosmos_metrics': public_metrics(trace)}) + '\n\n'
                    yield chunk
            finally:
                timer.discard(engine_request.request)
                engine_request.release()
        return _ReleasingStreamingResponse(stream(), prepared_stream=engine_request,
                    media_type='text/event-stream', headers={'Cache-Control': 'no-cache'})
    return app


def public_metrics(trace):
    return {key: trace.get(key) for key in ('request_id', 'server_elapsed_ms', 'native_inference_ms',
             'completion_tokens', 'prompt_tokens', 'timing_source', 'timing_boundary', 'cache_state',
             'observed_image_tokens', 'server_first_text_ms', 'first_text_timing_boundary')}


def run_http_server(client, config):
    import uvicorn
    uvicorn.run(create_app(client, config), host=config.host, port=config.port, log_level=config.log_level)
