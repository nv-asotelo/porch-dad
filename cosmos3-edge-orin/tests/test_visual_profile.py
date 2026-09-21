"""Actual upstream profile identity across build/preflight/serve; no GPU imports."""

import ast
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "external/TensorRT-Edge-LLM"))
import build_model_cache
import serve_backend
from experimental.server.runtime import engine_build


def assigned_options(source, scope):
    tree = ast.parse(source)
    node = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "options" for target in node.targets))
    exec(compile(ast.Module(body=[node], type_ignores=[]), "actual-build-options", "exec"), scope)
    return scope["options"]


class VisualProfileTests(unittest.TestCase):
    def test_actual_worker_preflight_and_shell_serve_match_cache_identity(self):
        source = (ROOT / "scripts/run_backend.sh").read_text()
        embedded = source.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
        shell_tail = source[source.index("visual_args=()"):]
        environment_cases = [({}, None, None),
            ({"COSMOS_MAX_IMAGE_TOKENS": "", "COSMOS_MAX_IMAGE_TOKENS_PER_IMAGE": ""}, None, None),
            ({"COSMOS_MAX_IMAGE_TOKENS": "256", "COSMOS_MAX_IMAGE_TOKENS_PER_IMAGE": "256"}, 256, 256),
            ({"COSMOS_MAX_IMAGE_TOKENS": "1024"}, 1024, None),
            ({"COSMOS_MAX_IMAGE_TOKENS": "512", "COSMOS_ENGINE_MAX_IMAGE_TOKENS_PER_IMAGE": "512",
              "COSMOS_MAX_IMAGE_TOKENS_PER_IMAGE": "320"}, 512, 512),
            ({"COSMOS_MAX_IMAGE_TOKENS": "512", "COSMOS_ENGINE_MAX_IMAGE_TOKENS_PER_IMAGE": "512",
              "COSMOS_MAX_IMAGE_TOKENS_PER_IMAGE": "512"}, 512, 512),
            ({"COSMOS_ENGINE_MAX_IMAGE_TOKENS_PER_IMAGE": "", "COSMOS_MAX_IMAGE_TOKENS_PER_IMAGE": "512"}, None, None)]
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "scripts").mkdir()
            (folder / "scripts/serve_backend.py").write_text("import json,sys\nprint(json.dumps(sys.argv[1:]))\n")
            model = folder / "model"
            model.mkdir()
            (model / "config.json").write_text("{}")
            paths = []
            for additions, total, per_image in environment_cases:
                with self.subTest(additions=additions), patch.dict(os.environ, additions, clear=True):
                    preflight = assigned_options(embedded, {"os": os, "BuildOptions": engine_build.BuildOptions})
                    built = assigned_options(build_model_cache.WORKER,
                        {"os": os, "builder": engine_build, "max_input": "1024", "max_kv": "2048"})
                    env = dict(os.environ, PATH=os.defpath, python_bin=sys.executable,
                               project_dir=str(folder), checkpoint=str(model), cache_dir=str(folder / "cache"))
                    output = subprocess.check_output(["bash", "-c", shell_tail], env=env, text=True)
                    argv = json.loads(output.splitlines()[-1])
                    visual, remaining = serve_backend.visual_arguments(argv)
                    budget, upstream_args = serve_backend.server_arguments(remaining)
                    config, kwargs = serve_backend.configured_model(upstream_args, visual)
                    served = kwargs["build_options"]
                    self.assertEqual(asdict(preflight), asdict(built))
                    self.assertEqual(asdict(built), asdict(served))
                    self.assertEqual((served.max_image_tokens, served.max_image_tokens_per_image), (total, per_image))
                    self.assertEqual(budget, 0)
                    self.assertFalse(config.model.context_cache_config.enabled)
                    self.assertEqual(config.api.max_queued_requests, 1)
                    self.assertEqual(config.api.reasoning_parser, "none")
                    cache_paths = {engine_build.bundle_cache_path(str(model), str(folder / "cache"), value)
                                   for value in (preflight, built, served)}
                    self.assertEqual(len(cache_paths), 1)
                    paths.append(cache_paths.pop())
            self.assertEqual(paths[0], paths[1], "Empty overrides must retain the original None fingerprint")
            self.assertNotEqual(paths[0], paths[2], "Compact profile must select a different bundle")
            self.assertNotEqual(paths[0], paths[3], "Explicit values participate in cache identity")
            self.assertEqual(paths[4], paths[5], "Runtime 320/512 must reuse the same built 512-capacity engine")
            self.assertEqual(paths[0], paths[6], "Explicit empty engine override must retain original FP16 cache identity")

    def test_optional_flags_are_consumed_locally_and_bad_capacities_refused(self):
        visual, remaining = serve_backend.visual_arguments(["/model", "--max-image-tokens", "256", "--max-batch-size", "1"])
        self.assertEqual(visual, {"max_image_tokens": 256, "max_image_tokens_per_image": None})
        self.assertEqual(remaining, ["/model", "--max-batch-size", "1"])
        for value in ("0", "-1", "nope"):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                serve_backend.visual_arguments(["/model", "--max-image-tokens", value])


if __name__ == "__main__":
    unittest.main()
