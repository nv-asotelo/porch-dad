import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("fetch_reasoner", Path(__file__).resolve().parents[1] / "scripts/fetch_reasoner.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class ReasonerDownloadTests(unittest.TestCase):
    def test_checked_in_reference_catalog_has_all_21_verified_files(self):
        files = mod.reference_files(json.loads(mod.REFERENCE.read_text()))
        self.assertEqual(len(files), 21)
        self.assertEqual(sum(f["size_bytes"] for f in files.values()), 7735525563)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.payloads = {
            "config.json": json.dumps({"model_type": "cosmos3_edge", "vision_config": {"hidden": 1}}).encode(),
            "model.safetensors.index.json": json.dumps({"weight_map": {"weight": "weights.safetensors"}}).encode(),
            "weights.safetensors": b"test tensor bytes", "README.md": b"original notices",
        }
        self.files = {name: {"verified": True, "size_bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
                      for name, body in self.payloads.items()}
        self.reference = {"repository": mod.REPOSITORY, "revision": mod.REVISION,
                          "complete": True, "files": self.files}
        self.reference_path = self.root / "reference.json"
        self.reference_path.write_text(json.dumps(self.reference))

    def materialize(self, target):
        target.mkdir(exist_ok=True)
        for name, body in self.payloads.items():
            (target / name).write_bytes(body)

    def test_reference_rejects_wrong_revision_and_traversal(self):
        self.assertEqual(mod.reference_files(self.reference), self.files)
        with self.assertRaises(ValueError):
            mod.reference_files({**self.reference, "revision": "moving-main"})
        for name in ("../outside", "/outside", "a/../outside", "a\\outside"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                mod.reference_files({**self.reference, "files": {**self.files, name: self.files["README.md"]}})

    def test_exact_hashes_and_generator_exclusion(self):
        target = self.root / "model"
        self.materialize(target)
        self.assertEqual(len(mod.verify(target, self.files)), 4)
        (target / "README.md").write_bytes(b"different notice")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            mod.verify(target, self.files)
        self.materialize(target)
        (target / "vae").mkdir()
        (target / "vae/config.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "generator"):
            mod.verify(target, self.files)

    def test_existing_user_directory_preserved_and_owned_resume_allowed(self):
        target = self.root / "model"
        self.materialize(target)
        for resume in (True, False):
            with self.assertRaises(ValueError):
                mod.prepare_target(target, self.files, resume)
        target2 = self.root / "owned"
        mod.prepare_target(target2, self.files, False)
        self.materialize(target2)
        mod.prepare_target(target2, self.files, True)
        with self.assertRaises(ValueError):
            mod.prepare_target(target2, {**self.files, "extra": self.files["README.md"]}, True)

    def test_symlinked_model_file_is_never_downloaded_through_or_verified(self):
        target = self.root / "owned"
        mod.prepare_target(target, self.files, False)
        outside = self.root / "user-file"
        outside.write_bytes(self.payloads["README.md"])
        self.materialize(target)
        (target / "README.md").unlink()
        (target / "README.md").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "symlink"):
            mod.prepare_target(target, self.files, True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            mod.verify(target, self.files)

    def test_download_cli_pins_all_files_then_writes_fresh_private_receipt(self):
        calls = []
        def download(repo, **kwargs):
            calls.append((repo, kwargs))
            self.materialize(Path(kwargs["local_dir"]))
        fake = types.SimpleNamespace(snapshot_download=download)
        target, receipt = self.root / "model", self.root / "receipt.json"
        with patch.object(mod, "REFERENCE", self.reference_path), patch.dict(sys.modules, huggingface_hub=fake), patch("builtins.print"):
            self.assertEqual(mod.main(["--output", str(target), "--receipt", str(receipt)]), 0)
        self.assertEqual(calls[0][0], mod.REPOSITORY)
        self.assertEqual(calls[0][1]["revision"], mod.REVISION)
        self.assertEqual(calls[0][1]["allow_patterns"], sorted(self.files))
        self.assertEqual(receipt.stat().st_mode & 0o777, 0o600)
        record = json.loads(receipt.read_text())
        self.assertTrue(record["complete"])
        self.assertFalse(record["inference_validated"])

    def test_verify_only_does_not_import_downloader_and_failure_is_recorded(self):
        target, receipt = self.root / "model", self.root / "receipt.json"
        self.materialize(target)
        (target / "README.md").write_bytes(b"changed")
        with patch.object(mod, "REFERENCE", self.reference_path), patch.dict(sys.modules, huggingface_hub=None), patch("builtins.print"):
            self.assertEqual(mod.main(["--output", str(target), "--receipt", str(receipt), "--verify-only"]), 2)
            with self.assertRaises(SystemExit):
                mod.main(["--output", str(target), "--receipt", str(receipt), "--verify-only"])
        self.assertFalse(json.loads(receipt.read_text())["complete"])


if __name__ == "__main__":
    unittest.main()
