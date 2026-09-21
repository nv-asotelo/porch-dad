"""Offline Cosmos3 sidecar checks; synthetic files are never model outputs."""

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
try:
    import repair_cosmos_chat_template as repair
finally:
    sys.path.pop(0)


def template():
    return {"model_path": "/synthetic/model", "roles": {
        "user": {"prefix": "\n<|im_start|>user\n", "suffix": "<|im_end|>\n"}},
        "generation_prompt": "<|im_start|>assistant\n<think></think>",
        "generation_prompt_thinking": "<|im_start|>assistant\n<think>\n",
        "content_types": {"custom": {"format": "preserve me"}}}


def source_files():
    config = {"model_type": "cosmos3_edge"}
    decoder = {}
    for key, (token_id, text) in repair.SPECIAL_TOKENS.items():
        config[key] = token_id
        decoder[str(token_id)] = {"content": text, "special": True}
    return {"config.json": json.dumps(config).encode(),
            "tokenizer_config.json": json.dumps({"added_tokens_decoder": decoder}).encode(),
            "chat_template.jinja": b"synthetic test fixture; not the official Jinja template"}


class ChatRepairTests(unittest.TestCase):
    def test_missing_formats_added_and_unrelated_fields_preserved(self):
        before = template()
        original = copy.deepcopy(before)
        after, changes, _ = repair.plan_repair(before)
        self.assertEqual(before, original)
        self.assertEqual(len(changes), 2)
        self.assertEqual(after["content_types"]["image"]["format"],
                         "<|vision_start|><|image_pad|><|vision_end|>")
        self.assertEqual(after["content_types"]["video"]["format"],
                         "<|vision_start|><|video_pad|><|vision_end|>")
        for kind in ("image", "video"):
            del after["content_types"][kind]
        self.assertEqual(after, original)

    def test_empty_format_repaired_but_nonempty_conflicts_refused(self):
        for empty in (None, ""):
            before = template()
            before["content_types"]["image"] = {"format": empty, "extra": "keep"}
            after, _, previous = repair.plan_repair(before)
            self.assertEqual(after["content_types"]["image"]["extra"], "keep")
            self.assertEqual(previous["content_types.image.format"], {"present": True, "value": empty})
        for bad in ("<wrong_pad>", {"format": "wrong"}, 19):
            before = template()
            before["content_types"]["image"] = {"format": bad}
            with self.assertRaises(ValueError):
                repair.plan_repair(before)

    def test_apply_backup_hashes_idempotence_and_source_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            model, bundle, receipts = base / "model", base / "bundle", base / "receipts"
            model.mkdir()
            bundle.mkdir()
            originals = source_files()
            for name, data in originals.items():
                (model / name).write_bytes(data)
            expected = {name: repair.common.sha256(data) for name, data in originals.items()}
            target = bundle / "processed_chat_template.json"
            before = json.dumps(template(), separators=(",", ":")).encode()
            target.write_bytes(before)
            # Pin only the synthetic fixture for this isolated filesystem test.
            with mock.patch.object(repair, "SOURCE_HASHES", expected):
                self.assertEqual(repair.repair(model, bundle, results_dir=receipts)["status"], "dry_run")
                self.assertEqual(target.read_bytes(), before)
                self.assertFalse(receipts.exists())
                record = repair.repair(model, bundle, apply=True, results_dir=receipts)
                self.assertEqual(record["status"], "applied")
                self.assertEqual(Path(record["backup"]).read_bytes(), before)
                self.assertEqual(json.loads(Path(record["receipt"]).read_text()), record)
                self.assertEqual(repair.common.sha256(target.read_bytes()), record["after_sha256"])
                after = target.read_bytes()
                receipt_files = sorted(receipts.iterdir())
                self.assertEqual(repair.repair(model, bundle, apply=True, results_dir=receipts)["status"],
                                 "already_normalized")
                self.assertEqual(target.read_bytes(), after)
                self.assertEqual(sorted(receipts.iterdir()), receipt_files)
                for name, data in originals.items():
                    self.assertEqual((model / name).read_bytes(), data)
                (model / "chat_template.jinja").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    repair.repair(model, bundle, apply=True, results_dir=receipts)
                self.assertEqual(target.read_bytes(), after)

    def test_source_token_mismatch_rejected_even_with_matching_file_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            originals = source_files()
            config = json.loads(originals["config.json"])
            config["image_token_id"] = 999
            originals["config.json"] = json.dumps(config).encode()
            for name, data in originals.items():
                (model / name).write_bytes(data)
            with mock.patch.object(repair, "SOURCE_HASHES", {
                    name: repair.common.sha256(data) for name, data in originals.items()}):
                with self.assertRaisesRegex(ValueError, "image_token_id"):
                    repair.validate_source(model)


if __name__ == "__main__":
    unittest.main()
