"""CPU-only config/translation checks; no native binding is compiled or loaded."""

import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).parents[1]
UPSTREAM = ROOT / "external/TensorRT-Edge-LLM"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


config = load("encoder_budget_server_config", UPSTREAM / "experimental/server/config.py")
launcher = load("encoder_budget_launcher", ROOT / "scripts/serve_backend.py")
# Exercise the exact translator without importing GPU-dependent engine modules.
tree = ast.parse((UPSTREAM / "experimental/server/runtime/engine.py").read_text())
node = next(item for item in tree.body if isinstance(item, ast.FunctionDef)
            and item.name == "_native_context_cache_config")
scope = {"ContextCacheConfig": config.ContextCacheConfig}
exec(compile(ast.Module(body=[node], type_ignores=[]), "native_config_translator", "exec"), scope)
translate = scope["_native_context_cache_config"]


class NativeConfig:
    __slots__ = ("enabled", "max_records", "recurrent_snapshot_pool_bytes",
                 "partial_kv_snapshot_pool_bytes", "encoder_embedding_cache_budget_bytes")

    def __init__(self):
        self.enabled = False
        self.max_records = 1024
        self.recurrent_snapshot_pool_bytes = 0
        self.partial_kv_snapshot_pool_bytes = 0
        self.encoder_embedding_cache_budget_bytes = 256 * 1024 * 1024


class EncoderCacheBudgetTests(unittest.TestCase):
    def test_upstream_default_unchanged_and_zero_works_without_context_reuse(self):
        self.assertEqual(config.ContextCacheConfig().encoder_embedding_cache_budget_bytes, 256 * 1024 * 1024)
        for budget in (0, 16 * 1024 * 1024):
            parsed = config.ContextCacheConfig.parse({"encoder_embedding_cache_budget_bytes": budget})
            self.assertFalse(parsed.enabled)
            native = translate(SimpleNamespace(ContextCacheConfig=NativeConfig), parsed)
            self.assertFalse(native.enabled)
            self.assertEqual(native.encoder_embedding_cache_budget_bytes, budget)

    def test_config_rejects_invalid_values(self):
        for value in (True, -1, 2**63, 1.5, "1024"):
            with self.subTest(value=value), self.assertRaises(config.ServerConfigError):
                config.ContextCacheConfig(encoder_embedding_cache_budget_bytes=value)

    def test_upstream_cli_accepts_zero_independently_and_retains_own_default(self):
        baseline = config.parse_server_config(["/absolute/model"])
        self.assertEqual(baseline.model.context_cache_config.encoder_embedding_cache_budget_bytes, 256 * 1024 * 1024)
        parsed = config.parse_server_config(["/absolute/model", launcher.BUDGET_FLAG, "0"])
        self.assertFalse(parsed.model.context_cache_config.enabled)
        self.assertEqual(parsed.model.context_cache_config.encoder_embedding_cache_budget_bytes, 0)

    def test_original_launcher_defaults_to_disabled_cache_and_preserves_other_arguments(self):
        original = ["/absolute/model", "--host", "127.0.0.1", "--max-batch-size", "1"]
        budget, args = launcher.server_arguments(original)
        self.assertEqual(budget, 0)
        self.assertEqual(args, original + [launcher.BUDGET_FLAG, "0"])
        budget, args = launcher.server_arguments(original + [launcher.BUDGET_FLAG, "0"])
        self.assertEqual(budget, 0)
        self.assertEqual(args, original + [launcher.BUDGET_FLAG, "0"])
        budget, args = launcher.server_arguments(original + [launcher.BUDGET_FLAG, "16777216"])
        self.assertEqual(budget, 16777216)
        self.assertEqual(args, original + [launcher.BUDGET_FLAG, "16777216"])

    def test_old_binding_rejected_and_patched_config_roundtrips(self):
        with self.assertRaisesRegex(RuntimeError, "rebuild _edgellm_runtime"):
            launcher.verify_native_binding(SimpleNamespace(ContextCacheConfig=object), 0)
        launcher.verify_native_binding(SimpleNamespace(ContextCacheConfig=NativeConfig), 0)


if __name__ == "__main__":
    unittest.main()
