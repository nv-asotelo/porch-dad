"""Real FastAPI/upstream routing regressions, without CUDA or model loading.

Run in the server environment with httpx installed:
    python -m unittest discover -s tests -p test_cosmos_runtime_routes.py -v

This intentionally uses public upstream's real app, schemas, serving adapter,
middleware and lifespan. Only the engine client is fake. Missing optional server
dependencies skip this suite; a routing/schema failure does not.
"""

import copy
import importlib.util
import logging
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "external/TensorRT-Edge-LLM"
sys.path.insert(0, str(BACKEND))
SPEC = importlib.util.spec_from_file_location(
    "cosmos_runtime_routes_under_test", ROOT / "scripts/cosmos_runtime.py")
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


class FakeLLM:
    def __init__(self, model_dir):
        self.model_dir = str(model_dir)
        self._rt = SimpleNamespace(ImageData=lambda: SimpleNamespace(
            max_image_tokens_per_image=0, skip_encoder_cache=False))
        self._context_cache_config = SimpleNamespace(
            encoder_embedding_cache_budget_bytes=268435456)

    def _visual_config(self):
        return {"builder_config": {
            "max_image_tokens_per_image": 512, "min_image_tokens": 4}}

    def _make_generation_request(self, *args, **kwargs):
        raise AssertionError("routing tests must not prepare a native request")

    def _handle_request(self, request):
        raise AssertionError("routing tests must not invoke a native engine")


class FakeEngineClient:
    model_name = "route-test-cosmos"
    active_requests = 0
    queued_requests = 0

    def __init__(self, model_dir, capabilities_type, output_type):
        self.llm = FakeLLM(model_dir)
        self.capabilities = capabilities_type(
            chat=True, transcription=False, speech=False,
            input_modalities=("text", "image"), output_modalities=("text",),
            max_model_len=1664, max_input_len=1024, max_batch_size=1,
            max_num_seqs=1, kv_cache_dtype="fp16", speculative_decoding=False,
            speculative_method="none", context_reuse=False)
        self.output_type = output_type
        self.prepared = []
        self.generated = []
        self.closed = False

    async def prepare_request(self, messages, sampling, **kwargs):
        self.prepared.append((copy.deepcopy(messages), sampling))
        return SimpleNamespace(request=object(), release=lambda: None)

    async def generate(self, messages, sampling, **kwargs):
        trace = runtime.ACTIVE_TRACE.get()
        self.generated.append({"trace": copy.deepcopy(trace), "sampling": sampling})
        if trace is not None:
            # Synthetic values are solely a route-selection marker, not a
            # native performance measurement or a substitute for hardware QA.
            trace.update(status="completed", server_elapsed_ms=42.0,
                         native_inference_ms=42.0, completion_tokens=2,
                         prompt_tokens=11, cache_state="disabled")
        kwargs["prepared"].release()
        return self.output_type(text="The extension route ran.", token_ids=[1, 2],
                                prompt_tokens=11, finish_reason="stop")

    async def close(self):
        self.closed = True


class CosmosRuntimeRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from fastapi.testclient import TestClient
            from experimental.server.api.app import create_app
            from experimental.server.api.routes import router
            from experimental.server.config import ApiConfig
            from experimental.server.runtime.engine import CompletionOutput
            from experimental.server.runtime.engine_client import EngineCapabilities
        except ModuleNotFoundError as exc:
            raise unittest.SkipTest(f"Optional public server dependency missing: {exc}") from exc
        except RuntimeError as exc:
            if "httpx" in str(exc) and "install" in str(exc):
                raise unittest.SkipTest(str(exc)) from exc
            raise
        cls.TestClient = TestClient
        cls.upstream_app = staticmethod(create_app)
        cls.router = router
        cls.ApiConfig = ApiConfig
        cls.CompletionOutput = CompletionOutput
        cls.EngineCapabilities = EngineCapabilities

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cosmos-routes-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.original_routes = tuple(self.router.routes)
        self.original_endpoints = tuple(getattr(r, "endpoint", None)
                                        for r in self.original_routes)
        self.environment = mock.patch.dict(os.environ, {
            "COSMOS_MAX_IMAGE_TOKENS_PER_IMAGE": "320", "COSMOS_STATIC_CLOCKS": "1",
            "COSMOS_BACKEND_ID": "routing-test", "COSMOS_TOP_P": "0.95"})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.clients = []

    def tearDown(self):
        self.assertEqual(tuple(self.router.routes), self.original_routes)
        self.assertEqual(tuple(getattr(r, "endpoint", None) for r in self.router.routes),
                         self.original_endpoints)
        # create_app's per-instance loggers must not retain temporary files.
        for client, app in self.clients:
            if hasattr(app.state, "cosmos_runtime"):
                logger = logging.getLogger("cosmos.requests." + app.state.cosmos_runtime["engine_id"])
                for handler in logger.handlers[:]:
                    handler.close()
                    logger.removeHandler(handler)

    def app(self, *, upstream=False, api_key=""):
        client = FakeEngineClient(self.directory, self.EngineCapabilities, self.CompletionOutput)
        config = self.ApiConfig(api_key=api_key, reasoning_parser="none")
        app = (self.upstream_app(client, config) if upstream else
               runtime.create_app(client, config,
                   log_path=self.directory / f"requests-{len(self.clients)}.jsonl"))
        self.clients.append((client, app))
        return client, app

    @staticmethod
    def payload(**updates):
        body = {"model": FakeEngineClient.model_name,
                "messages": [{"role": "user", "content": "Describe this scene."}],
                "max_tokens": 16, "temperature": 0.7}
        body.update(updates)
        return body

    def test_extensions_dispatch_to_task_route_not_upstream_schema(self):
        body = self.payload(max_image_tokens_per_image=384,
                            cosmos_benchmark={"warmup": True, "run_id": "route-regression",
                                              "cache_mode": "bypass"})
        original_client, original_app = self.app(upstream=True)
        with self.TestClient(original_app) as http:
            rejected = http.post("/v1/chat/completions", json=body)
        self.assertEqual(rejected.status_code, 400, rejected.text)
        self.assertEqual(original_client.prepared, [])

        client, app = self.app()
        with self.TestClient(app) as http:
            response = http.post("/v1/chat/completions", json=body)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(client.generated), 1)
        trace = client.generated[0]["trace"]
        self.assertEqual(trace["controls"]["max_image_tokens_per_image"], 384)
        self.assertEqual(trace["controls"]["encoder_cache_mode"], "bypass")
        self.assertEqual(trace["controls"]["top_p"], 0.95)
        self.assertTrue(trace["warmup"])
        self.assertEqual(trace["run_id"], "route-regression")
        self.assertEqual(response.json()["cosmos_metrics"]["native_inference_ms"], 42.0)
        self.assertEqual(response.json()["usage"]["completion_tokens"], 2)
        self.assertTrue(client.closed)

    def test_preserves_health_authentication_and_extension_validation(self):
        client, app = self.app(api_key="route-test-key")
        headers = {"Authorization": "Bearer route-test-key"}
        with self.TestClient(app) as http:
            health = http.get("/health")
            self.assertEqual(health.status_code, 200, health.text)
            self.assertEqual(health.json()["capabilities"]["max_input_len"], 1024)
            self.assertEqual(http.get("/health/ready").status_code, 200)
            settings = http.get("/api/runtime").json()
            self.assertEqual(settings["max_image_tokens_per_image"], 320)
            self.assertEqual(settings["encoder_cache_bytes"], 268435456)
            self.assertTrue(settings["static_clocks"])
            self.assertEqual(http.get("/v1/models").status_code, 401)
            models = http.get("/v1/models", headers=headers)
            self.assertEqual(models.status_code, 200, models.text)
            self.assertEqual(models.json()["data"][0]["id"], client.model_name)
            denied = http.post("/v1/chat/completions", json=self.payload())
            self.assertEqual(denied.status_code, 401, denied.text)
            self.assertEqual(denied.headers["www-authenticate"], "Bearer")
            invalid = http.post("/v1/chat/completions", headers=headers,
                                json=self.payload(max_image_tokens_per_image=3))
            self.assertEqual(invalid.status_code, 400, invalid.text)
            self.assertEqual(client.prepared, [])
            valid = http.post("/v1/chat/completions", headers=headers, json=self.payload())
            self.assertEqual(valid.status_code, 200, valid.text)
            self.assertEqual(client.generated[0]["trace"]["controls"]["max_image_tokens_per_image"], 320)
            self.assertEqual(valid.json()["cosmos_metrics"]["timing_boundary"], "native_inference")
        self.assertTrue(client.closed)

    def test_multiple_task_apps_leave_global_upstream_routes_intact(self):
        first, first_app = self.app()
        second, second_app = self.app()
        with self.TestClient(first_app) as http:
            response = http.post("/v1/chat/completions", json=self.payload(max_image_tokens_per_image=320))
            self.assertEqual(response.status_code, 200, response.text)
        with self.TestClient(second_app) as http:
            response = http.post("/v1/chat/completions", json=self.payload(max_image_tokens_per_image=512))
            self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(first.generated[0]["trace"]["controls"]["max_image_tokens_per_image"], 320)
        self.assertEqual(second.generated[0]["trace"]["controls"]["max_image_tokens_per_image"], 512)
        self.assertNotEqual(first_app.state.cosmos_runtime["engine_id"],
                            second_app.state.cosmos_runtime["engine_id"])
        # Create upstream after both task apps: a global-router mutation would
        # lose this original route or change its strict extension-field policy.
        original, app = self.app(upstream=True)
        with self.TestClient(app) as http:
            response = http.post("/v1/chat/completions", json=self.payload())
            self.assertEqual(response.status_code, 200, response.text)
            self.assertNotIn("cosmos_metrics", response.json())
            self.assertIsNone(original.generated[0]["trace"])
            rejected = http.post("/v1/chat/completions", json=self.payload(max_image_tokens_per_image=320))
            self.assertEqual(rejected.status_code, 400, rejected.text)
        self.assertEqual(len(original.generated), 1)


if __name__ == "__main__":
    unittest.main()
