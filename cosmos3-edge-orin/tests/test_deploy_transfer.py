"""Local archive checks; no model transfer, SSH, device access or GPU execution."""

import ast
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
from types import SimpleNamespace
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
            "scripts/cosmos_runtime.py", "scripts/benchmark_native.py", "scripts/benchmark_ttft.py", "scripts/validate_runtime_controls.py",
            "scripts/compare_server_timings.py", "tests/test_cosmos_runtime.py",
            "tests/test_compare_server_timings.py", "tests/test_runtime_image_budget_patch.py",
            "tests/test_ui_metrics.js", "tests/test_capture_presets.js",
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
            "COSMOS_MAX_IMAGE_TOKENS", "COSMOS_MAX_IMAGE_TOKENS_PER_IMAGE",
            "COSMOS_ENGINE_MAX_IMAGE_TOKENS_PER_IMAGE", "COSMOS_BACKEND_ID",
            "COSMOS_TOP_P",
        })
        self.assertEqual(settings["COSMOS_PROFILE"], "rtn-v1")
        self.assertEqual(settings["COSMOS_MODEL_DIR"], deploy_transfer.DESTINATION + "/models/cosmos3-edge-rtn-mlp-int4")
        self.assertEqual(settings["COSMOS_CACHE_DIR"], deploy_transfer.DESTINATION + "/data/engine-cache-mlp-compact")
        self.assertEqual(settings["COSMOS_MAX_INPUT_LEN"], "1024")
        self.assertEqual(settings["COSMOS_MAX_KV_CAPACITY"], "1664")
        self.assertEqual(settings["COSMOS_MAX_IMAGE_TOKENS"], "512")
        self.assertEqual(settings["COSMOS_STATIC_CLOCKS"], "0")
        self.assertEqual(settings["COSMOS_ENCODER_CACHE_BYTES"], "0")
        self.assertEqual(settings["COSMOS_MAX_IMAGE_TOKENS_PER_IMAGE"], "512")
        self.assertEqual(settings["COSMOS_TOP_P"], "1")
        self.assertEqual(settings["COSMOS_ENGINE_MAX_IMAGE_TOKENS_PER_IMAGE"], "512")
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
        self.assertEqual(config["max_image_tokens_per_image"], 512)
        self.assertEqual(config["sampling"]["top_p"], 1)
        self.assertEqual(config["engine_max_image_tokens_per_image"], 512)
        self.assertEqual(config["quantization_scope"], "mlp-only")
        self.assertEqual(config["quantized_linear_count"], 56)
        self.assertEqual(config["sampling"]["temperature"], 0)
        self.assertEqual(config["ui_defaults"]["capture_preset"], "lightweight")
        self.assertEqual(config["ui_defaults"]["max_output_tokens"], 64)
        self.assertEqual(config["ui_defaults"]["frame_longest_side"], 512)

    def test_current_six_patch_policy_and_runtime_dependency_order_are_allowlisted(self):
        self.assertEqual(deploy_transfer.BACKEND_PATCHES, [
            "patches/cosmos3-patch-embedding-chw.patch", "patches/cosmos3-half-pixel-position.patch",
            "patches/tensorrt-edge-llm-v0.10.1-encoder-cache-budget.patch", "patches/int4-gemv-cosmos-mlp-n4.patch",
            "patches/cosmos-runtime-image-token-budget.patch", "patches/cosmos-encoder-cache-bypass.patch"])
        self.assertTrue(set(deploy_transfer.BACKEND_PATCHES).issubset(deploy_transfer.ORIGINAL_FILES))
        config = json.loads(deploy_transfer.local_file("deployment/selected-config.json").read_text())
        self.assertEqual(set(config["patch_sha256"]), {Path(path).name for path in deploy_transfer.BACKEND_PATCHES})

    def test_complete_tracked_diff_rejects_missing_or_undocumented_patches(self):
        deploy_transfer.require_reviewed_backend_diff(b"all six patches", b"all six patches")
        for actual in (b"three old patches", b"all six patches plus undocumented change", b""):
            with self.subTest(actual=actual), self.assertRaisesRegex(ValueError, "6 reviewed patches"):
                deploy_transfer.require_reviewed_backend_diff(actual, b"all six patches")

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

    def test_runtime_acceptance_default_manifest_and_exact_inputs_are_packaged(self):
        folder = ROOT / "benchmarks/live-vlm-1280"
        manifest = json.loads((folder / "manifest.json").read_text())
        for name in ("manifest.json", "README.md", *(item["file"] for item in manifest["fixtures"])):
            self.assertIn(str((folder / name).relative_to(ROOT)), deploy_transfer.ORIGINAL_FILES)
        for item in manifest["fixtures"]:
            path = folder / item["file"]
            self.assertEqual(deploy_transfer.sha256(path), item["sha256"])
            self.assertEqual(path.stat().st_size, item["size_bytes"])
        self.assertIn("research/runtime-controls-server-timing.md", deploy_transfer.ORIGINAL_FILES)


class DeploymentScopeTests(unittest.TestCase):
    def fixture(self, root, profile="rtn-v1", model="models/cosmos3-edge-rtn-mlp-int4"):
        (root / "deployment").mkdir()
        selection = root / "deployment/selected.env"
        selection.write_text("COSMOS_PROFILE=" + profile + "\nCOSMOS_MODEL_DIR="
                             + deploy_transfer.DESTINATION + "/" + model + "\n")
        return selection

    def test_mlp_plain_prepare_refuses_before_model_hashing_or_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            self.fixture(root)
            with patch.object(deploy_transfer, "ROOT", root), \
                    patch.object(deploy_transfer, "verified_model_files") as models, \
                    patch.object(deploy_transfer, "backend_snapshot") as snapshot:
                with self.assertRaisesRegex(ValueError, "prepare --source-stage"):
                    deploy_transfer.prepare()
                models.assert_not_called()
                snapshot.assert_not_called()

    def test_explicit_source_stage_is_never_selected_service_ready(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            self.fixture(root)
            with patch.object(deploy_transfer, "ROOT", root):
                scope = deploy_transfer.deployment_scope(source_stage=True)
            self.assertEqual(scope["kind"], "source_stage")
            self.assertEqual(scope["selected_profile"], "rtn-v1")
            self.assertEqual(scope["included_model_path"], deploy_transfer.DESTINATION + "/models/cosmos3-edge-reasoner")
            for field in ("selected_weights_included", "engine_caches_included", "native_binaries_included",
                          "ready_to_start_selected_service"):
                self.assertIs(scope[field], False)
            self.assertIn("MLP INT4 weights", scope["required_followup"][0])

    def test_original_fp16_checkpoint_selection_keeps_plain_prepare_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            self.fixture(root, profile="fp16", model="models/cosmos3-edge-reasoner")
            with patch.object(deploy_transfer, "ROOT", root):
                scope = deploy_transfer.deployment_scope()
            self.assertEqual(scope["kind"], "fp16_checkpoint_and_sources")
            self.assertIs(scope["selected_weights_included"], True)
            self.assertIs(scope["ready_to_start_selected_service"], False)
            self.assertIs(scope["engine_caches_included"], False)

    def test_unknown_duplicate_or_outside_selected_model_fails_without_shell_execution(self):
        cases = ["COSMOS_PROFILE=rtn-v1\nCOSMOS_PROFILE=fp16\n",
                 "COSMOS_PROFILE=$(exit 87)\nCOSMOS_MODEL_DIR=/home/jetson/cosmos-edge/models/model\n",
                 "COSMOS_PROFILE=rtn-v1\nCOSMOS_MODEL_DIR=/other/place/model\n",
                 "COSMOS_PROFILE=rtn-v1\nCOSMOS_MODEL_DIR=/home/jetson/cosmos-edge/models/../../secret\n"]
        for contents in cases:
            with self.subTest(contents=contents), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                selection = self.fixture(root)
                selection.write_text(contents)
                with patch.object(deploy_transfer, "ROOT", root), self.assertRaises(ValueError):
                    deploy_transfer.deployment_scope(source_stage=True)

    def test_prepared_manifest_records_real_model_scope_and_patch_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            self.fixture(root)
            (root / "scripts").mkdir()
            (root / "scripts/cosmos_runtime.py").write_text("# synthetic packaging fixture\n")
            weight = root / deploy_transfer.MODEL_DIRECTORY / "fixture.weights"
            weight.parent.mkdir(parents=True)
            weight.write_bytes(b"synthetic bytes, not real model weights")
            relative = str(weight.relative_to(root))
            model_files = {relative: {"sha256": deploy_transfer.sha256(weight), "size_bytes": weight.stat().st_size}}
            state = {"source_revision": deploy_transfer.BACKEND,
                     "patches": [{"path": path} for path in deploy_transfer.BACKEND_PATCHES]}

            def fake_snapshot(destination):
                destination.write_bytes(b"synthetic backend archive fixture")
                return state

            with patch.object(deploy_transfer, "ROOT", root), \
                    patch.object(deploy_transfer, "PACKAGE", root / "data/deployment"), \
                    patch.object(deploy_transfer, "ORIGINAL_FILES", ["deployment/selected.env", "scripts/cosmos_runtime.py"]), \
                    patch.object(deploy_transfer, "verified_model_files", return_value=model_files), \
                    patch.object(deploy_transfer, "backend_snapshot", side_effect=fake_snapshot), \
                    patch("sys.stdout", new_callable=io.StringIO):
                deploy_transfer.prepare(source_stage=True)
            result = json.loads((root / "data/deployment/manifest.json").read_text())
            self.assertEqual(result["backend"]["source_state"], state)
            self.assertEqual(result["model"]["path"], deploy_transfer.MODEL_DIRECTORY)
            self.assertEqual(result["deployment_scope"]["kind"], "source_stage")
            self.assertIs(result["deployment_scope"]["selected_weights_included"], False)
            self.assertIn("generated MLP INT4 model weights", result["exclusions"])
            self.assertEqual({item["source"] for item in result["files"]},
                {"deployment/selected.env", "scripts/cosmos_runtime.py", relative, deploy_transfer.ARCHIVE_SOURCE})

    def test_send_rejects_scope_claiming_missing_mlp_weights_before_transport(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            self.fixture(root)
            package = root / "data/deployment"
            package.mkdir(parents=True)
            with patch.object(deploy_transfer, "ROOT", root):
                scope = deploy_transfer.deployment_scope(source_stage=True)
                scope["selected_weights_included"] = True
                (package / "manifest.json").write_text(json.dumps({"deployment_scope": scope}))
                args = SimpleNamespace(host="example.invalid", port=22,
                                       identity_file="unused-key", known_hosts="unused-hosts")
                with patch.object(deploy_transfer, "PACKAGE", package), \
                        patch.object(deploy_transfer, "private_file", return_value=root / "unused"), \
                        patch.object(deploy_transfer.subprocess, "Popen") as transport, \
                        patch.object(deploy_transfer, "verified_model_files") as models:
                    with self.assertRaisesRegex(ValueError, "selected-artifact staging scope"):
                        deploy_transfer.send(args)
                    transport.assert_not_called()
                    models.assert_not_called()


@unittest.skipUnless(os.environ.get("COSMOS_RUN_ARCHIVE_INTEGRATION") == "1" and ARCHIVE.is_file(),
                     "Optional existing archive integration: set COSMOS_RUN_ARCHIVE_INTEGRATION=1 with a prepared archive")
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
                if relative:  # The parent intentionally carries reviewed patches; submodules stay clean.
                    self.assertEqual(deploy_transfer.git(checkout, "status", "--porcelain", "--untracked-files=no"), "")
            status = subprocess.check_output(
                ["git", "-C", str(backend), "submodule", "status", "--recursive"], text=True)
            self.assertEqual(len(status.splitlines()), len(deploy_transfer.REPOSITORIES) - 1)
            self.assertTrue(all(line.startswith(" ") for line in status.splitlines()))


@unittest.skipUnless(os.environ.get("COSMOS_RUN_ARCHIVE_INTEGRATION") == "1"
                     and (ROOT / "external/TensorRT-Edge-LLM/.git").exists(),
                     "Optional fresh archive integration: set COSMOS_RUN_ARCHIVE_INTEGRATION=1 with the exact six-patch local checkout")
class PatchedBackendArchiveTests(unittest.TestCase):
    def test_fresh_archive_contains_exact_documented_patches_and_receiver_checks_hashes(self):
        with tempfile.TemporaryDirectory(prefix="patched-backend-roundtrip-") as temporary:
            folder = Path(temporary).resolve()
            archive_path = folder / "patched.tar"
            with patch.object(deploy_transfer, "PACKAGE", folder):
                state = deploy_transfer.backend_snapshot(archive_path)
            self.assertEqual(state["source_revision"], deploy_transfer.BACKEND)
            self.assertEqual([item["path"] for item in state["patches"]], deploy_transfer.BACKEND_PATCHES)
            expected_paths = set()
            for relative in deploy_transfer.BACKEND_PATCHES:
                for line in (ROOT / relative).read_text().splitlines():
                    if line.startswith("+++ b/"):
                        expected_paths.add(line.removeprefix("+++ b/").split("\t", 1)[0])
            self.assertEqual({item["path"] for item in state["patched_files"]}, expected_paths)
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
