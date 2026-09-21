"""CPU-only checks against the unchanged local public tokenizer, when available."""

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import preflight_cosmos_artifacts as preflight


@unittest.skipUnless(importlib.util.find_spec("transformers") and
                     (ROOT / "models/cosmos3-edge-reasoner/tokenizer.json").is_file(),
                     "Requires the task-local tokenizer and Transformers CPU environment")
class CosmosPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = ROOT / "models/cosmos3-edge-reasoner"
        cls.source = json.loads((cls.model / "config.json").read_text())
        cls.tokenizer = preflight.source_tokenizer(cls.model)
        path = ROOT / "external/TensorRT-Edge-LLM/experimental/builder/core/artifacts/chat_template.py"
        spec = importlib.util.spec_from_file_location("pinned_template_test", path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        with tempfile.TemporaryDirectory() as directory:
            helper.write_processed_chat_template(str(cls.model), directory)
            cls.generated = json.loads((Path(directory) / "processed_chat_template.json").read_text())
        cls.normalized, _, _ = preflight.chat_repair.plan_repair(cls.generated)

    def test_real_jinja_and_tokens_allow_known_build_repair_without_mutation(self):
        before = copy.deepcopy(self.generated)
        self.assertEqual(self.generated["content_types"], {})  # Reproduces upstream omission.
        additions = preflight.validate_chat(self.generated, self.tokenizer)
        self.assertEqual(len(additions), 2)
        self.assertEqual(self.generated, before)
        with self.assertRaisesRegex(ValueError, "media formats need normalization"):
            preflight.validate_chat(self.generated, self.tokenizer, require_normalized=True)
        self.assertEqual(preflight.validate_chat(self.normalized, self.tokenizer, require_normalized=True), {})

    def test_bad_media_thinking_or_whitespace_semantics_are_refused(self):
        mutations = [
            lambda data: data["content_types"]["image"].update(format="<wrong_image_pad>"),
            lambda data: data.update(generation_prompt_thinking=data["generation_prompt"]),
            lambda data: data["roles"]["user"].update(trim_content=True),
            lambda data: data.update(prompt_prefix="unexpected prefix"),
            lambda data: data.update(default_system_prompt="unexpected system text"),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                data = copy.deepcopy(self.normalized)
                mutation(data)
                with self.assertRaises(ValueError):
                    preflight.validate_chat(data, self.tokenizer)

    def test_source_hash_guard_precedes_tokenizer_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            for name in preflight.chat_repair.SOURCE_HASHES:
                data = (self.model / name).read_bytes()
                (model / name).write_bytes(data + b" " if name == "config.json" else data)
            with self.assertRaisesRegex(ValueError, "config.json SHA256 differs"):
                preflight.source_tokenizer(model)

    def test_missing_or_incomplete_bundle_cannot_trigger_direct_start_build(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "missing"
            args = (self.model, bundle, self.source, self.tokenizer)
            self.assertEqual(preflight.validate_bundle(*args, require_ready=False,
                                                       is_ready=lambda: self.fail("Build preflight must not load readiness dependencies")), {})
            with self.assertRaisesRegex(ValueError, "No ready engine bundle"):
                preflight.validate_bundle(*args, require_ready=True, is_ready=lambda: True)
            bundle.mkdir()
            with self.assertRaisesRegex(ValueError, "No ready engine bundle"):
                preflight.validate_bundle(*args, require_ready=True, is_ready=lambda: False)

    def test_rope_repair_allowed_for_build_but_required_before_serving(self):
        # The readiness callback is an explicit metadata-test double, never a real engine.
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            (bundle / "visual").mkdir()
            chat = bundle / "processed_chat_template.json"
            chat.write_text(json.dumps(self.normalized))
            visual = copy.deepcopy(self.source)
            visual["model_type"] = "cosmos3_edge_vision"
            path = bundle / "visual/config.json"
            path.write_text(json.dumps(visual))
            before = {file: file.read_bytes() for file in (path, chat)}
            args = (self.model, bundle, self.source, self.tokenizer)
            changes = preflight.validate_bundle(*args, require_ready=False, is_ready=lambda: True)
            self.assertEqual(len(changes["rope"]), 2)
            with self.assertRaisesRegex(ValueError, "RoPE aliases need normalization"):
                preflight.validate_bundle(*args, require_ready=True, is_ready=lambda: True)
            self.assertEqual({file: file.read_bytes() for file in before}, before)
            normalized, _ = preflight.runtime_repair.plan_repair(self.source, visual)
            path.write_text(json.dumps(normalized))
            changes = preflight.validate_bundle(*args, require_ready=True, is_ready=lambda: True)
            self.assertEqual(changes, {"media": {}, "rope": {}})
            normalized["text_config"]["rope_theta"] = 10000
            path.write_text(json.dumps(normalized))
            with self.assertRaisesRegex(ValueError, "Conflicting existing"):
                preflight.validate_bundle(*args, require_ready=False, is_ready=lambda: True)


if __name__ == "__main__":
    unittest.main()
