"""Selected service launcher checks using inert local scripts; no device calls."""

import configparser
import json
import os
import pwd
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SelectedBackendTests(unittest.TestCase):
    def launch(self, profile, *, cache="/selected/cache", model="/selected/model", overrides=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = root / "scripts"
            scripts.mkdir()
            launcher = scripts / "run_selected_backend.sh"
            shutil.copyfile(ROOT / "scripts/run_selected_backend.sh", launcher)
            # Both endpoints only print their received argv and selected cache.
            stub = '#!/usr/bin/env bash\nprintf "%s\\n" "$COSMOS_CACHE_DIR" "$@"\n'
            (scripts / "run_backend.sh").write_text(stub)
            python = root / "external/TensorRT-Edge-LLM/.venv/bin/python"
            python.parent.mkdir(parents=True)
            python.write_text(stub)
            python.chmod(0o755)
            env = {key: value for key, value in os.environ.items() if not key.startswith("COSMOS_")}
            env["COSMOS_PROFILE"] = profile
            if cache is not None:
                env["COSMOS_CACHE_DIR"] = cache
            if model is not None:
                env["COSMOS_MODEL_DIR"] = model
            env.update(overrides or {})
            return subprocess.run(["bash", str(launcher)], env=env, capture_output=True, text=True)

    def test_fp16_passes_explicit_selection(self):
        result = self.launch("fp16")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["/selected/cache", "/selected/model"])

    def test_rtn_passes_explicit_selection_and_fixed_loopback_port(self):
        result = self.launch("rtn-v1")
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        self.assertEqual(lines[0], "/selected/cache")
        self.assertTrue(lines[1].endswith("/scripts/rtn_backend.py"))
        self.assertEqual(lines[2:], ["serve", "--model", "/selected/model", "--cache-dir",
                                    "/selected/cache", "--max-input-len", "1024", "--max-kv-capacity", "2048",
                                    "--host", "127.0.0.1", "--port", "8000",
                                    "--encoder-embedding-cache-budget-bytes", "0"])

    def test_rtn_compact_profile_forwards_all_selected_limits(self):
        result = self.launch("rtn-v1", overrides={"COSMOS_MAX_INPUT_LEN": "512", "COSMOS_MAX_KV_CAPACITY": "1024",
                            "COSMOS_MAX_IMAGE_TOKENS": "256", "COSMOS_MAX_IMAGE_TOKENS_PER_IMAGE": "256"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines()[2:], ["serve", "--model", "/selected/model", "--cache-dir",
            "/selected/cache", "--max-input-len", "512", "--max-kv-capacity", "1024", "--host", "127.0.0.1",
            "--port", "8000", "--max-image-tokens", "256", "--max-image-tokens-per-image", "256",
            "--encoder-embedding-cache-budget-bytes", "0"])

    def test_rtn_empty_optional_limits_preserve_default_profile(self):
        default = self.launch("rtn-v1")
        empty = self.launch("rtn-v1", overrides={"COSMOS_MAX_IMAGE_TOKENS": "", "COSMOS_MAX_IMAGE_TOKENS_PER_IMAGE": ""})
        self.assertEqual(empty.returncode, 0, empty.stderr)
        self.assertEqual(empty.stdout.splitlines()[2:], default.stdout.splitlines()[2:])

    def test_rtn_engine_capacity_is_separate_from_runtime_image_budget(self):
        for budget in ("320", "512"):
            with self.subTest(runtime_image_budget=budget):
                result = self.launch("rtn-v1", overrides={"COSMOS_MAX_IMAGE_TOKENS": "512",
                    "COSMOS_ENGINE_MAX_IMAGE_TOKENS_PER_IMAGE": "512",
                    "COSMOS_MAX_IMAGE_TOKENS_PER_IMAGE": budget})
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.splitlines()[-6:-2],
                    ["--max-image-tokens", "512", "--max-image-tokens-per-image", "512"])

    def test_rtn_explicit_cache_budget_overrides_lightweight_default(self):
        result = self.launch("rtn-v1", overrides={"COSMOS_ENCODER_CACHE_BYTES": "268435456"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines()[-2:], ["--encoder-embedding-cache-budget-bytes", "268435456"])

    def test_missing_selection_or_unknown_profile_refuses_before_dispatch(self):
        for profile in ("fp16", "rtn-v1"):
            for field in ("cache", "model"):
                with self.subTest(profile=profile, missing=field):
                    result = self.launch(profile, **{field: None})
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("Selected " + field + " is required", result.stderr)
                    self.assertEqual(result.stdout, "")
        self.assertNotEqual(self.launch("unvalidated").returncode, 0)


class ServiceInstallTests(unittest.TestCase):
    def install(self, selection, *, existing_clocks=False, local_selection=None, args=(), verify_fail=False):
        """Execute the real installer with all privileged paths/commands inert."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            scripts = root / "scripts"
            scripts.mkdir()
            units = root / "units"
            units.mkdir()
            (root / "deployment").mkdir()
            (root / "models/selected").mkdir(parents=True)
            (root / "data/selected-cache").mkdir(parents=True)
            base = (f"COSMOS_MODEL_DIR={root}/models/selected\n"
                    f"COSMOS_CACHE_DIR={root}/data/selected-cache\n")
            if "COSMOS_PROFILE=" not in selection:
                base += "COSMOS_PROFILE=rtn-v1\n"
            (root / "deployment/selected.env").write_text(base + selection)
            if local_selection is not None:
                (root / "deployment/local.env").write_text(base + local_selection)
            shutil.copyfile(ROOT / "scripts/configure_deployment.py", scripts / "configure_deployment.py")
            for name in ("run_selected_backend.sh", "run_backend.sh", "rtn_backend.py",
                         "serve_backend.py", "serve_ui.py", "preflight_cosmos_artifacts.py",
                         "repair_cosmos_runtime_config.py", "repair_cosmos_chat_template.py",
                         "build_model_cache.py", "cosmos_runtime.py"):
                (scripts / name).touch()
            python = root / "external/TensorRT-Edge-LLM/.venv/bin/python"
            python.parent.mkdir(parents=True)
            python.write_text("#!/bin/sh\nexit 99\n")
            python.chmod(0o755)
            commands = root / "commands"
            commands.mkdir()
            recorder = root / "record.py"
            recorder.write_text("import json, os, sys\n"
                                "with open(os.environ['SERVICE_TEST_LOG'], 'a') as log:\n"
                                "    log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
                                "if os.environ.get('SERVICE_VERIFY_FAIL') and sys.argv[1] == 'systemd-analyze': sys.exit(9)\n")
            for name in ("systemctl", "systemd-analyze", "jetson_clocks"):
                executable = commands / name
                executable.write_text("#!/bin/sh\nexec " + shlex.quote(sys.executable) + " " +
                                      shlex.quote(str(recorder)) + " " + name + ' "$@"\n')
                executable.chmod(0o755)
            clock_path = commands / "jetson_clocks"
            old_unit = "existing task-owned clock unit\n"
            if existing_clocks:
                (units / "cosmos-edge-clocks.service").write_text(old_unit)
            source = (ROOT / "scripts/install_services.sh").read_text()
            guard = "[[ $EUID == 0 && $(uname -m) == aarch64 ]] || { echo 'Run as root on the Orin.' >&2; exit 2; }"
            self.assertEqual(source.count(guard), 1)
            source = source.replace(guard, ": # Test-only platform guard bypass")
            source = source.replace("/etc/systemd/system", str(units))
            source = source.replace("/usr/bin/jetson_clocks", str(clock_path))
            source = source.replace("/usr/bin/python3", shlex.quote(sys.executable))
            installer = scripts / "install_services.sh"
            installer.write_text(source)
            command_log = root / "commands.jsonl"
            account = pwd.getpwuid(os.getuid()) if os.getuid() else pwd.getpwnam("nobody")
            if os.getuid() == 0:
                for path in [root, *root.rglob("*")]:
                    os.chown(path, account.pw_uid, account.pw_gid)
            env = dict(os.environ, SUDO_USER=account.pw_name, PATH=str(commands) + os.pathsep + os.environ["PATH"],
                       SERVICE_TEST_LOG=str(command_log), SERVICE_VERIFY_FAIL="1" if verify_fail else "")
            result = subprocess.run(["bash", str(installer), *args], env=env, capture_output=True, text=True)
            calls = [json.loads(line) for line in command_log.read_text().splitlines()] if command_log.exists() else []
            # Never execute clocks or start/stop/restart services, even in the inert fixture.
            self.assertFalse(any(call[0] == "jetson_clocks" for call in calls))
            self.assertFalse(any(call[0] == "systemctl" and any(value in call[1:] for value in
                             ("start", "stop", "restart", "--now")) for call in calls))
            return result, {path.name: path.read_text() for path in units.iterdir()}, calls, str(clock_path)

    def test_local_environment_selected_and_actual_nonroot_account_rendered(self):
        result, units, _, _ = self.install("COSMOS_STATIC_CLOCKS=0\n", local_selection="COSMOS_STATIC_CLOCKS=0\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        account = pwd.getpwuid(os.getuid()) if os.getuid() else pwd.getpwnam("nobody")
        backend = units["cosmos-edge-backend.service"]
        self.assertIn(f"User={account.pw_name}\n", backend)
        self.assertIn("/deployment/local.env\n", backend)
        self.assertNotIn("/home/jetson/cosmos-edge", backend)
        self.assertIn("--host 127.0.0.1 --port 8090", units["cosmos-edge-ui.service"])

    def test_dry_run_prints_units_and_invalid_account_writes_nothing(self):
        result, units, calls, _ = self.install("COSMOS_STATIC_CLOCKS=0\n", args=("--dry-run",))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("EnvironmentFile=", result.stdout)
        self.assertEqual(units, {})
        self.assertEqual(calls, [])
        result, units, calls, _ = self.install("COSMOS_STATIC_CLOCKS=0\n", args=("--user", "root"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must not be root", result.stderr)
        self.assertEqual(units, {})
        self.assertEqual(calls, [])

    def test_failed_staged_verification_writes_no_units(self):
        result, units, calls, _ = self.install("COSMOS_STATIC_CLOCKS=0\n", verify_fail=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(units, {})
        self.assertEqual([call[0] for call in calls], ["systemd-analyze"])

    def test_static_clocks_unit_and_backend_dependency_are_opt_in(self):
        result, units, calls, clock_path = self.install("COSMOS_STATIC_CLOCKS=1\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        clocks = configparser.ConfigParser(interpolation=None)
        clocks.read_string(units["cosmos-edge-clocks.service"])
        self.assertEqual(clocks["Unit"]["After"], "nvpmodel.service")
        self.assertEqual(clocks["Unit"]["Before"], "cosmos-edge-backend.service")
        self.assertEqual(clocks["Service"]["Type"], "oneshot")
        self.assertEqual(clocks["Service"]["User"], "root")
        self.assertEqual(clocks["Service"]["ExecStart"], clock_path)
        self.assertEqual(clocks["Service"]["RemainAfterExit"], "yes")
        backend = units["cosmos-edge-backend.service"]
        self.assertIn("After=network.target cosmos-edge-clocks.service", backend)
        self.assertIn("Requires=cosmos-edge-clocks.service", backend)
        self.assertIn(["systemctl", "enable", "cosmos-edge-backend.service",
                       "cosmos-edge-ui.service", "cosmos-edge-clocks.service"], calls)
        verify = next(call for call in calls if call[0] == "systemd-analyze")
        self.assertEqual(verify[1], "verify")
        self.assertEqual({Path(value).name for value in verify[2:]}, set(units))

    def test_unset_and_zero_do_not_generate_or_enable_clock_service(self):
        for selection in ("COSMOS_PROFILE=fp16\n", "COSMOS_STATIC_CLOCKS=0\n"):
            with self.subTest(selection=selection):
                result, units, calls, _ = self.install(selection)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn("cosmos-edge-clocks.service", units)
                self.assertNotIn("cosmos-edge-clocks.service", units["cosmos-edge-backend.service"])
                self.assertIn(["systemctl", "enable", "cosmos-edge-backend.service", "cosmos-edge-ui.service"], calls)

    def test_zero_disables_old_boot_activation_without_changing_current_clocks(self):
        result, units, calls, _ = self.install("COSMOS_STATIC_CLOCKS=0\n", existing_clocks=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(units["cosmos-edge-clocks.service"], "existing task-owned clock unit\n")
        self.assertNotIn("cosmos-edge-clocks.service", units["cosmos-edge-backend.service"])
        self.assertIn(["systemctl", "disable", "cosmos-edge-clocks.service"], calls)

    def test_invalid_or_duplicate_clock_values_refuse_before_writing_units(self):
        for assignment in ("2", '"1"', " 1", "", "$(exit 87)"):
            with self.subTest(assignment=assignment):
                result, units, calls, _ = self.install("COSMOS_STATIC_CLOCKS=" + assignment + "\n")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("exact unquoted line", result.stderr)
                self.assertEqual(units, {})
                self.assertEqual(calls, [])
        result, units, calls, _ = self.install("COSMOS_STATIC_CLOCKS=1\nCOSMOS_STATIC_CLOCKS=0\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Duplicate", result.stderr)
        self.assertEqual(units, {})
        self.assertEqual(calls, [])

    def test_selected_environment_is_never_executed(self):
        result, _, _, _ = self.install("UNRELATED=$(exit 87)\nexit 88\nCOSMOS_STATIC_CLOCKS=0\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("literal", result.stderr)


if __name__ == "__main__":
    unittest.main()
