"""Local archive checks; no model transfer, SSH, device access or GPU execution."""

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("deploy_transfer", ROOT / "scripts/deploy_transfer.py")
deploy_transfer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(deploy_transfer)
ARCHIVE = ROOT / deploy_transfer.ARCHIVE_SOURCE


def receiver_backend_code():
    """Execute the actual receiver extraction/checks without SSH/account setup."""
    receiver = ast.parse(deploy_transfer.REMOTE_RECEIVER)
    body = []
    extracting_backend = False
    for node in receiver.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            body.append(node)
        elif isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "backend" for target in node.targets):
            extracting_backend = True
            body.append(node)
        elif extracting_backend:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and (
                    isinstance(node.value.func, ast.Attribute)
                    and isinstance(node.value.func.value, ast.Name)
                    and node.value.func.value.id == "incomplete"):
                break
            body.append(node)
    if not extracting_backend:
        raise AssertionError("Receiver lost backend extraction")
    return compile(ast.Module(body=body, type_ignores=[]), "receiver-backend-check", "exec")


class DeploymentAllowlistTests(unittest.TestCase):
    def test_final_service_dependencies_are_included_as_regular_files(self):
        required = {
            "scripts/install_services.sh", "scripts/run_selected_backend.sh", "scripts/run_backend.sh",
            "scripts/rtn_backend.py", "scripts/serve_backend.py", "scripts/serve_ui.py",
            "scripts/build_model_cache.py", "scripts/preflight_cosmos_artifacts.py",
            "scripts/repair_cosmos_runtime_config.py", "scripts/repair_cosmos_chat_template.py",
            "tests/test_selected_backend.py", "docs/deployment-transfer.md",
            "research/reference-quality-diagnostic.md",
        }
        self.assertTrue(required.issubset(deploy_transfer.ORIGINAL_FILES))
        self.assertEqual(len(deploy_transfer.ORIGINAL_FILES), len(set(deploy_transfer.ORIGINAL_FILES)))
        for relative in deploy_transfer.ORIGINAL_FILES:
            self.assertTrue(deploy_transfer.local_file(relative).is_file())
            self.assertNotIn(".qa", Path(relative).parts)

    def test_selected_profile_is_explicit_and_contains_only_reviewed_settings(self):
        self.assertEqual({path for path in deploy_transfer.ORIGINAL_FILES if path.startswith("deployment/")},
                         {"deployment/selected.env", "deployment/selected-config.json"})
        settings = dict(line.split("=", 1) for line in
                        deploy_transfer.local_file("deployment/selected.env").read_text().splitlines())
        self.assertEqual(set(settings), {
            "COSMOS_PROFILE", "COSMOS_MODEL_DIR", "COSMOS_CACHE_DIR", "COSMOS_MAX_INPUT_LEN",
            "COSMOS_MAX_KV_CAPACITY", "COSMOS_ENCODER_CACHE_BYTES", "COSMOS_STATIC_CLOCKS",
            "HF_HUB_OFFLINE", "HF_HUB_DISABLE_IMPLICIT_TOKEN", "PYTHONUNBUFFERED", "TMPDIR",
            "XDG_CACHE_HOME", "CUDA_CACHE_PATH",
        })
        self.assertEqual(settings["COSMOS_PROFILE"], "fp16")
        self.assertEqual(settings["COSMOS_STATIC_CLOCKS"], "0")
        for name in ("COSMOS_MODEL_DIR", "COSMOS_CACHE_DIR", "TMPDIR", "XDG_CACHE_HOME", "CUDA_CACHE_PATH"):
            self.assertTrue(settings[name].startswith(deploy_transfer.DESTINATION + "/"))
            self.assertNotIn("$", settings[name])
        config = json.loads(deploy_transfer.local_file("deployment/selected-config.json").read_text())
        self.assertEqual(config["backend_revision"], deploy_transfer.BACKEND)
        self.assertEqual(config["model_revision"], deploy_transfer.MODEL)
        self.assertEqual(config["cache"], settings["COSMOS_CACHE_DIR"])
        for key, variable in (("max_input_len", "COSMOS_MAX_INPUT_LEN"),
                              ("max_kv_cache_capacity", "COSMOS_MAX_KV_CAPACITY"),
                              ("encoder_embedding_cache_budget_bytes", "COSMOS_ENCODER_CACHE_BYTES")):
            self.assertEqual(config[key], int(settings[variable]))
        self.assertIs(config["selected"], True)
        self.assertFalse(config["text_context_reuse"])

    def test_only_verified_model_receipt_is_allowlisted_from_results(self):
        self.assertEqual([path for path in deploy_transfer.ORIGINAL_FILES if path.startswith("results/")],
                         ["results/model-download.json"])
        record = json.loads(deploy_transfer.local_file("results/model-download.json").read_text())
        self.assertEqual(record["revision"], deploy_transfer.MODEL)
        self.assertIs(record["complete"], True)
        self.assertEqual(record["errors"], [])
        self.assertTrue(all(entry["verified"] is True for entry in record["files"].values()))

    def test_natural_smoke_images_and_saved_cc_by_attribution_are_complete(self):
        folder = ROOT / "benchmarks/natural-smoke"
        manifest = json.loads((folder / "manifest.json").read_text())
        files = {str(path.relative_to(ROOT)) for path in folder.rglob("*") if path.is_file()}
        self.assertEqual(len(files), 10)
        self.assertTrue(files.issubset(deploy_transfer.ORIGINAL_FILES))
        self.assertEqual(len(manifest["cases"]), 3)
        for case in manifest["cases"]:
            image = folder / case["filename"]
            self.assertEqual(image.stat().st_size, case["size_bytes"])
            self.assertEqual(deploy_transfer.sha256(image), case["sha256"])
            attribution = case["provenance"]
            evidence = folder / attribution["flickr_attribution_evidence"]
            self.assertEqual(deploy_transfer.sha256(evidence), case["flickr_attribution_evidence_sha256"])
            public_metadata = json.loads(evidence.read_text())["oembed"]
            self.assertEqual(attribution["license"], "CC BY 2.0")
            self.assertEqual(public_metadata["license"], attribution["license"])
            self.assertEqual(public_metadata["license_url"], attribution["license_url"])
            self.assertEqual(public_metadata["author_name"], attribution["creator"])


@unittest.skipUnless(ARCHIVE.is_file(), "Run deploy_transfer.py prepare to create the backend archive")
class BackendArchiveTests(unittest.TestCase):
    def test_regular_file_archive_preserves_each_repository_identity(self):
        # Execute the actual receiver's extraction and Git verification statements;
        # leave its account/SSH/outer-transfer setup out of this local-only test.
        code = receiver_backend_code()

        with tarfile.open(ARCHIVE) as archive:
            members = archive.getmembers()
            self.assertTrue(all(member.isfile() for member in members))
            # This is the shape that triggered the real transfer bug: empty refs
            # directories cannot be represented by the regular-file-only archive.
            names = {member.name for member in members}
            for relative, _, _ in deploy_transfer.REPOSITORIES[1:]:
                self.assertFalse(any(name.startswith(relative + "/.git/refs/") for name in names))

        with tempfile.TemporaryDirectory(prefix="deployment-roundtrip-") as temporary:
            root = Path(temporary).resolve()
            (root / ".transfer").mkdir()
            shutil.copyfile(ARCHIVE, root / deploy_transfer.ARCHIVE_TARGET)
            manifest = {"backend": {"repositories": [
                {"path": path, "revision": revision}
                for path, revision, _ in deploy_transfer.REPOSITORIES
            ]}}
            exec(code, {"ROOT": root, "manifest": manifest})
            backend = root / "external/TensorRT-Edge-LLM"
            for relative, revision, _ in deploy_transfer.REPOSITORIES:
                checkout = backend / relative
                self.assertEqual(deploy_transfer.git(checkout, "rev-parse", "HEAD"), revision)
                self.assertEqual(Path(deploy_transfer.git(checkout, "rev-parse", "--show-toplevel")), checkout)
                self.assertEqual(deploy_transfer.git(checkout, "status", "--porcelain", "--untracked-files=no"), "")
            status = subprocess.check_output(
                ["git", "-C", str(backend), "submodule", "status", "--recursive"], text=True)
            self.assertEqual(len(status.splitlines()), len(deploy_transfer.REPOSITORIES) - 1)
            self.assertTrue(all(line.startswith(" ") for line in status.splitlines()))


@unittest.skipUnless((ROOT / "external/TensorRT-Edge-LLM/.git").exists(), "Requires the local public pinned backend")
class PatchedBackendArchiveTests(unittest.TestCase):
    def test_fresh_archive_contains_exact_documented_patches_and_receiver_checks_hashes(self):
        with tempfile.TemporaryDirectory(prefix="patched-backend-roundtrip-") as temporary:
            folder = Path(temporary).resolve()
            archive_path = folder / "patched.tar"
            with patch.object(deploy_transfer, "PACKAGE", folder):
                state = deploy_transfer.backend_snapshot(archive_path)
            self.assertEqual(state["source_revision"], deploy_transfer.BACKEND)
            self.assertEqual([item["path"] for item in state["patches"]], deploy_transfer.BACKEND_PATCHES)
            self.assertEqual(len(state["patched_files"]), 10)
            with tarfile.open(archive_path) as archive:
                for item in state["patched_files"]:
                    member = archive.getmember(item["path"])
                    self.assertTrue(member.isfile())
                    self.assertEqual(hashlib.sha256(archive.extractfile(member).read()).hexdigest(), item["after_sha256"])
                    self.assertNotEqual(item["before_sha256"], item["after_sha256"])
            manifest = {"backend": {"revision": deploy_transfer.BACKEND, "source_state": state,
                "repositories": [{"path": path, "revision": revision}
                                 for path, revision, _ in deploy_transfer.REPOSITORIES]}}
            for mode in ("valid", "tampered"):
                root = folder / mode
                (root / ".transfer").mkdir(parents=True)
                # A hardlink avoids duplicating the immutable archive in this local test.
                (root / deploy_transfer.ARCHIVE_TARGET).hardlink_to(archive_path)
                checking = copy.deepcopy(manifest)
                if mode == "tampered":
                    checking["backend"]["source_state"]["patched_files"][0]["after_sha256"] = "0" * 64
                    with self.assertRaisesRegex(ValueError, "Patched backend file SHA-256 mismatch"):
                        exec(receiver_backend_code(), {"ROOT": root, "manifest": checking})
                else:
                    exec(receiver_backend_code(), {"ROOT": root, "manifest": checking})
                    backend = root / "external/TensorRT-Edge-LLM"
                    self.assertEqual(hashlib.sha256(deploy_transfer.git(backend, *deploy_transfer.DIFF_ARGUMENTS,
                                                                         raw=True)).hexdigest(), state["tracked_diff_sha256"])


if __name__ == "__main__":
    unittest.main()
