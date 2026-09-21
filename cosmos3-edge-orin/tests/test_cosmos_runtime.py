"""CPU-only contracts for per-request controls and transport-free native timing."""

import asyncio
import copy
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock


SPEC = importlib.util.spec_from_file_location(
    "cosmos_runtime", Path(__file__).parents[1] / "scripts/cosmos_runtime.py")
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


class Clock:
    def __init__(self):
        self.ns = 0

    def __call__(self):
        return self.ns

    def advance(self, milliseconds):
        self.ns += milliseconds * 1_000_000


class CopiedRow:
    """Model pybind STL getters: edits require assigning the returned list back."""
    def __init__(self, count=1):
        self._images = [SimpleNamespace(max_image_tokens_per_image=0, skip_encoder_cache=False)
                        for _ in range(count)]

    @property
    def image_buffers(self):
        return copy.deepcopy(self._images)

    @image_buffers.setter
    def image_buffers(self, value):
        self._images = copy.deepcopy(value)


class CopiedRequest:
    def __init__(self, image_count=1):
        self._rows = [CopiedRow(image_count)]

    @property
    def requests(self):
        return copy.deepcopy(self._rows)

    @requests.setter
    def requests(self, value):
        self._rows = copy.deepcopy(value)


class FakeNative:
    def __init__(self, clock):
        self.clock = clock
        self.runs, self.tokens = 10, 3200
        self.cache_hit = False
        self.metrics_missing = False
        self.error = None
        self.reason = "stop"
        self.received_budgets = None
        self.received_bypass = None

    def get_multimodal_metrics(self):
        self.clock.advance(13)  # Reading metrics is also outside native timing.
        if self.metrics_missing:
            raise RuntimeError("metrics unavailable")
        return SimpleNamespace(observed_image_runs=self.runs, observed_image_tokens=self.tokens)

    def handle_request(self, request):
        self.received_budgets = [image.max_image_tokens_per_image
                                 for row in request.requests for image in row.image_buffers]
        self.received_bypass = [image.skip_encoder_cache
                               for row in request.requests for image in row.image_buffers]
        self.clock.advance(42)
        if self.error:
            raise self.error
        if not self.cache_hit:
            self.runs += 1
            self.tokens += 300
        return SimpleNamespace(finish_reasons=[self.reason], output_ids=[[3, 4, 5, 6]],
                               prompt_token_counts=[345])


class FakeLLM:
    context_cache_enabled = False

    def __init__(self, clock):
        self.clock = clock
        self._runtime = FakeNative(clock)
        self._rt = object()
        self.original_handle_calls = 0
        self.guard_error = None
        self.preparation_error = None

    def _make_generation_request(self, image_count=1):
        self.clock.advance(71)  # JPEG decode and model request preparation.
        if self.preparation_error:
            raise self.preparation_error
        return CopiedRequest(image_count)

    @contextmanager
    def _infer_guard(self):
        self.clock.advance(89)  # Wait for the existing native admission lock.
        if self.guard_error:
            raise self.guard_error
        try:
            yield
        finally:
            self.clock.advance(97)  # Admission cleanup after native completion.

    def _ensure_open(self):
        self.clock.advance(11)

    def _handle_request(self, request):
        self.original_handle_calls += 1
        return self._runtime.handle_request(request)


class ScriptedStream:
    """Server-side iterator with controlled delivery delays and close state."""
    def __init__(self, clock, chunks=()):
        self.clock = clock
        self.chunks = iter(chunks)
        self.closed = False
        self.close_calls = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.closed:
            raise StopIteration
        delay_ms, text = next(self.chunks)
        self.clock.advance(delay_ms)
        return SimpleNamespace(text=text, token_ids=[1] if text else [])

    def close(self):
        self.close_calls += 1
        self.closed = True


class FakeStreamingLLM(FakeLLM):
    def __init__(self, clock):
        super().__init__(clock)
        self.stream = ScriptedStream(clock)
        self.stream_calls = []

    def generate_stream(self, *args, **kwargs):
        self.stream_calls.append((args, kwargs))
        return self.stream


def trace(cache_bytes=268435456, image_count=1, budget=320, cache_mode="default"):
    return {"controls": {"max_image_tokens_per_image": budget, "encoder_cache_bytes": cache_bytes,
                         "encoder_cache_mode": cache_mode},
            "image_count": image_count}


class NativeTimerTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.llm = FakeLLM(self.clock)
        self.emitted = []
        self.timer = runtime.NativeTimer(self.llm, {}, lambda value: self.emitted.append(copy.deepcopy(value)),
                                         clock=self.clock)
        engine = ModuleType("experimental.server.runtime.engine")
        engine.finish_reason_name = lambda binding, value: value
        self.module_patch = mock.patch.dict(sys.modules, {engine.__name__: engine})
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

    def prepare(self, current=None):
        current = trace() if current is None else current
        token = runtime.ACTIVE_TRACE.set(current)
        try:
            request = self.llm._make_generation_request(current["image_count"])
        finally:
            runtime.ACTIVE_TRACE.reset(token)
        return current, request

    def test_decode_queue_metrics_and_transport_do_not_enter_native_duration(self):
        current, request = self.prepare()
        self.clock.advance(150)  # HTTP scheduling and any delay before generation.
        response = self.llm._handle_request(request)
        self.clock.advance(900)  # Simulated client/network drain after response.
        self.assertEqual(response.output_ids, [[3, 4, 5, 6]])
        self.assertEqual(current["server_elapsed_ms"], 42)
        self.assertEqual(current["native_inference_ms"], 42)
        self.assertGreater(self.clock.ns / 1_000_000, 1000)
        self.assertEqual(current["completion_tokens"], 4)
        self.assertEqual(current["prompt_tokens"], 345)
        self.assertEqual(current["status"], "completed")
        self.assertEqual(current["cache_state"], "miss")
        self.assertEqual(current["observed_image_tokens"], 300)
        self.assertEqual(self.llm._runtime.received_budgets, [320])
        self.assertEqual(self.llm._runtime.received_bypass, [False])
        self.assertEqual(self.llm.original_handle_calls, 0)
        self.assertEqual(self.timer.pending, {})
        self.assertEqual(self.emitted, [{key: value for key, value in current.items()
                                        if not key.startswith("_")}])

    def test_copy_returning_pybind_rows_and_images_preserve_each_request_budget(self):
        first, request512 = self.prepare(trace(budget=512, image_count=2))
        second, request320 = self.prepare(trace(budget=320))
        self.llm._handle_request(request320)
        self.assertEqual(self.llm._runtime.received_budgets, [320])
        self.llm._handle_request(request512)
        self.assertEqual(self.llm._runtime.received_budgets, [512, 512])
        self.assertEqual(first["cache_state"], "unknown", "Multiple images do not establish a single cache hit")
        self.assertEqual(second["cache_state"], "miss")
        self.assertEqual(self.timer.pending, {})

    def test_cache_hit_disabled_missing_metrics_and_length_completion(self):
        for cache_bytes, hit, missing, expected in ((268435456, True, False, "hit"),
                (0, False, False, "disabled"), (268435456, False, True, "unknown")):
            with self.subTest(expected=expected):
                self.llm._runtime.cache_hit = hit
                self.llm._runtime.metrics_missing = missing
                self.llm._runtime.reason = "length"
                current, request = self.prepare(trace(cache_bytes=cache_bytes))
                self.llm._handle_request(request)
                self.assertEqual(current["cache_state"], expected)
                self.assertEqual(current["status"], "completed")
                self.assertEqual(current["finish_reason"], "length")
                if hit or missing:
                    self.assertIsNone(current["observed_image_tokens"])

    def test_native_failure_and_cancellation_preserve_exception_and_clear_pending(self):
        for error in (RuntimeError("native failed"), asyncio.CancelledError()):
            with self.subTest(error=type(error).__name__):
                self.llm._runtime.error = error
                current, request = self.prepare()
                with self.assertRaises(type(error)) as caught:
                    self.llm._handle_request(request)
                self.assertIs(caught.exception, error)
                self.assertEqual(current["status"], "error")
                self.assertEqual(current["server_elapsed_ms"], 42)
                self.assertEqual(self.timer.pending, {})

    def test_cache_bypass_reaches_native_copy_and_reports_disabled_with_vision_work(self):
        current, request = self.prepare(trace(cache_mode="bypass"))
        self.llm._handle_request(request)
        self.assertEqual(self.llm._runtime.received_bypass, [True])
        self.assertEqual(current["cache_state"], "disabled")
        self.assertEqual(current["observed_image_tokens"], 300)
        self.assertEqual(current["controls"]["encoder_cache_bytes"], 268435456)

    def test_preparation_failure_does_not_leave_pending_trace(self):
        self.llm.preparation_error = ValueError("invalid image")
        with self.assertRaisesRegex(ValueError, "invalid image"):
            self.prepare()
        self.assertEqual(self.timer.pending, {})
        self.assertEqual(self.emitted, [])
        self.assertIsNone(runtime.ACTIVE_TRACE.get())

    def test_canceled_finish_reason_is_never_success(self):
        self.llm._runtime.reason = "cancelled"
        current, request = self.prepare()
        self.llm._handle_request(request)
        self.assertEqual(current["status"], "canceled")
        self.assertEqual(self.timer.pending, {})

    def test_failed_admission_has_no_native_duration(self):
        self.llm.guard_error = RuntimeError("admission closed")
        current, request = self.prepare()
        with self.assertRaisesRegex(RuntimeError, "admission closed"):
            self.llm._handle_request(request)
        self.assertIsNone(current["server_elapsed_ms"])
        self.assertEqual(current["status"], "error")
        self.assertEqual(self.timer.pending, {})

    def test_discard_request_and_trace_do_not_leak_or_remove_other_requests(self):
        current = trace()
        _, first = self.prepare(current)
        _, second = self.prepare(current)
        other, third = self.prepare()
        self.timer.discard(first)
        self.assertNotIn(id(first), self.timer.pending)
        self.assertIn(id(second), self.timer.pending)
        self.timer.discard_trace(current)
        self.assertEqual(self.timer.pending, {id(third): other})
        self.timer.discard(third)
        self.timer.discard(third)  # Release and finalizer may both invoke cleanup.
        self.assertEqual(self.timer.pending, {})
        self.assertEqual(self.emitted, [])

    def test_untraced_requests_preserve_original_handler(self):
        request = self.llm._make_generation_request()
        self.llm._handle_request(request)
        self.assertEqual(self.llm.original_handle_calls, 1)
        self.assertEqual(self.emitted, [])

    def test_preparation_worker_context_and_generation_thread_use_request_identity(self):
        current = trace()

        async def prepare_in_worker():
            token = runtime.ACTIVE_TRACE.set(current)
            try:
                return await asyncio.to_thread(self.llm._make_generation_request)
            finally:
                runtime.ACTIVE_TRACE.reset(token)

        request = asyncio.run(prepare_in_worker())
        thread = threading.Thread(target=self.llm._handle_request, args=(request,))
        thread.start()
        thread.join()
        self.assertEqual(current["server_elapsed_ms"], 42)
        self.assertEqual(self.timer.pending, {})
        self.assertIsNone(runtime.ACTIVE_TRACE.get())


class NativeFirstTextTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.llm = FakeStreamingLLM(self.clock)
        self.emitted = []
        self.timer = runtime.NativeTimer(self.llm, {},
            lambda value: self.emitted.append(copy.deepcopy(value)), clock=self.clock)
        engine = ModuleType("experimental.server.runtime.engine")
        engine.finish_reason_name = lambda binding, value: value
        patch = mock.patch.dict(sys.modules, {engine.__name__: engine})
        patch.start()
        self.addCleanup(patch.stop)

    def prepare_stream(self, chunks=()):
        current = trace(cache_bytes=0)
        token = runtime.ACTIVE_TRACE.set(current)
        try:
            request = self.llm._make_generation_request()
        finally:
            runtime.ACTIVE_TRACE.reset(token)
        self.llm.stream = ScriptedStream(self.clock, chunks)
        iterator = self.llm.generate_stream([], prebuilt_request=request)
        self.assertIs(self.llm.stream_calls[-1][1]["prebuilt_request"], request)
        return current, request, iterator

    def assert_one_final_row(self, first_text_ms=None, status="completed"):
        self.assertEqual(len(self.emitted), 1)
        row = self.emitted[0]
        self.assertEqual(row["status"], status)
        self.assertEqual(row.get("server_first_text_ms"), first_text_ms)
        if first_text_ms is not None:
            self.assertEqual(row["first_text_timing_boundary"], "native_start_to_server_text")
        self.assertFalse(any(key.startswith("_") for key in row),
                         "Private timestamps/publication state must not enter request logs")
        return row

    def test_first_visible_text_excludes_decode_admission_and_transport(self):
        current, request, iterator = self.prepare_stream(
            [(5, ""), (3, "<|im_end|>"), (4, "A red circle"), (6, " is visible.")])
        self.clock.advance(200)  # HTTP preparation/scheduling before native admission.
        native = self.llm._runtime.handle_request

        def generate(prepared):
            self.assertEqual(next(iterator).text, "")
            self.assertIsNone(current.get("server_first_text_ms"))
            self.assertEqual(next(iterator).text, "<|im_end|>")
            self.assertIsNone(current.get("server_first_text_ms"))
            self.assertEqual(next(iterator).text, "A red circle")
            self.assertEqual(current["server_first_text_ms"], 12)
            return native(prepared)

        self.llm._runtime.handle_request = generate
        self.llm._handle_request(request)
        self.clock.advance(900)  # Proxy/browser/network delay after first server text.
        self.assertEqual(next(iterator).text, " is visible.")
        iterator.close()
        self.assertTrue(self.llm.stream.closed)
        row = self.assert_one_final_row(12)
        self.assertEqual(row["native_inference_ms"], 54)
        self.assertGreater(self.clock.ns / 1e6, 1000)
        public = runtime.public_metrics(current)
        self.assertEqual(public["server_first_text_ms"], 12)
        self.assertEqual(public["first_text_timing_boundary"], "native_start_to_server_text")
        self.assertNotIn("_native_started_ns", public)

    def test_native_completion_before_first_drain_waits_to_publish_and_does_not_clamp(self):
        current, request, iterator = self.prepare_stream([(17, "Caption")])
        self.llm._handle_request(request)
        self.assertEqual(self.timer.pending, {}, "Worker has already consumed the trace mapping")
        self.assertEqual(current["native_inference_ms"], 42)
        self.assertEqual(self.emitted, [], "Completed streaming requests await first-text observation")
        self.clock.advance(29)  # Server thread scheduling before it drains buffered text.
        self.assertEqual(next(iterator).text, "Caption")
        # Native42 + after-metrics13 + lock cleanup97 + scheduling29 + drain17.
        expected = 42 + 13 + 97 + 29 + 17
        self.assertEqual(current["server_first_text_ms"], expected)
        iterator.close()
        iterator.close()
        row = self.assert_one_final_row(expected)
        self.assertGreater(row["server_first_text_ms"], row["native_inference_ms"],
                           "A consumer-observed first-text metric must not be fabricated by clamping")

    def test_exhaustion_without_visible_text_logs_once_and_leaves_ttft_unavailable(self):
        _, request, iterator = self.prepare_stream([(2, ""), (3, "<|im_end|>")])
        self.llm._handle_request(request)
        self.assertEqual(self.emitted, [])
        self.assertEqual([chunk.text for chunk in iterator], ["", "<|im_end|>"])
        iterator.close()
        iterator.close()
        row = self.assert_one_final_row()
        self.assertEqual(row["native_inference_ms"], 42)

    def test_close_reaches_blocked_iterator_and_native_cancellation_logs_once(self):
        current, request, _ = self.prepare_stream()
        closed = threading.Event()
        entered_next = threading.Event()
        native_started = threading.Event()
        errors = []

        class BlockingStream:
            close_calls = 0

            def __iter__(inner):
                return inner

            def __next__(inner):
                entered_next.set()
                if not closed.wait(2):
                    raise AssertionError("close did not unblock stream consumption")
                raise StopIteration

            def close(inner):
                inner.close_calls += 1
                closed.set()

        self.llm.stream = BlockingStream()
        iterator = self.llm.generate_stream([], prebuilt_request=request)
        native = self.llm._runtime.handle_request
        self.llm._runtime.reason = "cancelled"

        def handle(prepared):
            native_started.set()
            if not closed.wait(2):
                raise AssertionError("close did not cancel native generation")
            return native(prepared)

        def consume():
            try:
                next(iterator)
            except StopIteration:
                pass
            except BaseException as exc:
                errors.append(exc)

        def run_native():
            try:
                self.llm._handle_request(request)
            except BaseException as exc:
                errors.append(exc)

        self.llm._runtime.handle_request = handle
        worker = threading.Thread(target=run_native, daemon=True)
        consumer = threading.Thread(target=consume, daemon=True)
        worker.start()
        try:
            self.assertTrue(native_started.wait(1))
            consumer.start()
            self.assertTrue(entered_next.wait(1))
            iterator.close()  # Must forward even though another thread is in next().
        finally:
            closed.set()
            worker.join(2)
            if consumer.ident is not None:
                consumer.join(2)
        self.assertFalse(worker.is_alive())
        self.assertFalse(consumer.is_alive())
        self.assertEqual(errors, [])
        self.assertGreaterEqual(self.llm.stream.close_calls, 1)
        iterator.close()
        self.assert_one_final_row(status="canceled")
        self.assertIsNone(current.get("server_first_text_ms"))
        self.assertEqual(self.timer.pending, {})

    def test_nonstreaming_requests_log_without_waiting_for_first_text(self):
        current = trace(cache_bytes=0)
        token = runtime.ACTIVE_TRACE.set(current)
        try:
            request = self.llm._make_generation_request()
        finally:
            runtime.ACTIVE_TRACE.reset(token)
        self.llm._handle_request(request)
        self.assert_one_final_row()
        self.assertEqual(self.llm.stream_calls, [])

    def test_untraced_stream_forwards_iterator_without_inventing_timing(self):
        request = self.llm._make_generation_request()
        self.llm.stream = ScriptedStream(self.clock, [(3, "Plain caller")])
        iterator = self.llm.generate_stream([], prebuilt_request=request)
        self.assertEqual(next(iterator).text, "Plain caller")
        iterator.close()
        self.assertTrue(self.llm.stream.closed)
        self.assertEqual(self.emitted, [])


class ControlTests(unittest.TestCase):
    def test_default_and_explicit_image_budget_are_separate_from_output_cap(self):
        payload = {"max_tokens": 512, "messages": [{"role": "user", "content": "Describe this image."}]}
        original = copy.deepcopy(payload)
        body, budget, annotation = runtime.request_controls(payload, 320, 512)
        self.assertEqual(budget, 320)
        self.assertEqual(body["max_tokens"], 512)
        self.assertEqual(body["top_p"], 1.0)
        self.assertEqual(annotation, {})
        self.assertEqual(payload, original)
        payload.update(max_image_tokens_per_image=512, top_p=.8, cosmos_benchmark={"warmup": True, "run_id": "fixture"})
        body, budget, annotation = runtime.request_controls(payload, 320, 512)
        self.assertEqual(budget, 512)
        self.assertEqual(body["top_p"], .8)
        self.assertEqual(annotation, {"warmup": True, "run_id": "fixture"})
        self.assertNotIn("max_image_tokens_per_image", body)
        self.assertNotIn("cosmos_benchmark", body)

    def test_budget_range_rejects_bool_float_string_and_built_capacity_excess(self):
        for value in (True, False, 0, 3, -1, 513, 320.0, "320", None, float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                runtime.request_controls({"max_image_tokens_per_image": value}, 320, 512)
        self.assertEqual(runtime.image_budget(4), 4)
        self.assertEqual(runtime.image_budget(512), 512)
        with self.assertRaises(ValueError):
            runtime.request_controls({}, 320, 256)

    def test_top_p_rejects_nonfinite_none_boolean_and_out_of_range(self):
        for value in (True, False, 0, -1, 1.001, None, "0.95", float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                runtime.request_controls({"top_p": value}, 320, 512)
        self.assertEqual(runtime.sampling_top_p(1), 1)
        self.assertEqual(runtime.sampling_top_p(.00001), .00001)

    def test_benchmark_annotation_is_small_typed_and_cannot_override_controls(self):
        for value in (None, [], "warmup", {"warmup": 1}, {"warmup": "false"},
                      {"run_id": True}, {"run_id": "a" * 101}, {"cache_state": "hit"},
                      {"cache_mode": "hit"}, {"cache_mode": None}, {"cache_mode": True}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                runtime.request_controls({"cosmos_benchmark": value}, 320, 512)
        for mode in ("default", "bypass"):
            body, _, annotation = runtime.request_controls({"cosmos_benchmark": {"cache_mode": mode}}, 320, 512)
            self.assertEqual(annotation["cache_mode"], mode)
            self.assertNotIn("cosmos_benchmark", body)

    def test_hashes_track_image_and_prompt_independently_without_mutating_messages(self):
        first = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": [
            {"type": "text", "text": "Describe this image."},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}}]}]
        original = copy.deepcopy(first)
        image, prompt, count = runtime.content_hashes(first)
        self.assertEqual(count, 1)
        self.assertEqual(first, original)
        self.assertEqual(len(image), 64)
        self.assertEqual(len(prompt), 64)
        second = copy.deepcopy(first)
        second[1]["content"][1]["image_url"]["url"] = "data:image/jpeg;base64,BBBB"
        image2, prompt2, _ = runtime.content_hashes(second)
        self.assertNotEqual(image, image2)
        self.assertEqual(prompt, prompt2)
        second[1]["content"][0]["text"] = "Count the people."
        image3, prompt3, _ = runtime.content_hashes(second)
        self.assertEqual(image2, image3)
        self.assertNotEqual(prompt2, prompt3)

    def test_invalid_multimodal_hash_inputs_are_refused(self):
        for content in (1, ["not-an-object"], [{"type": "image_url", "image_url": "bad"}],
                        [{"type": "image_url", "image_url": {"url": None}}]):
            with self.subTest(content=content), self.assertRaises(ValueError):
                runtime.content_hashes([{"role": "user", "content": content}])

    def test_public_metrics_excludes_prompt_images_and_internal_controls(self):
        current = trace()
        current.update(request_id="fixture", server_elapsed_ms=42, native_inference_ms=42,
                       completion_tokens=4, prompt_tokens=345, timing_source="server_monotonic",
                       timing_boundary="native_inference", cache_state="miss", observed_image_tokens=300)
        result = runtime.public_metrics(current)
        self.assertNotIn("controls", result)
        self.assertEqual(result["server_elapsed_ms"], 42)
        self.assertEqual(result["timing_boundary"], "native_inference")


if __name__ == "__main__":
    unittest.main()
