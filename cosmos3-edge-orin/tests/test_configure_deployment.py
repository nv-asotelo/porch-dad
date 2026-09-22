"""Portable deployment configuration checks; no systemd or device changes."""
import grp
import importlib.util
import json
import os
from pathlib import Path
import pwd
import stat
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("configure_deployment", ROOT / "scripts/configure_deployment.py")
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)


class ConfigureDeploymentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        (self.root / "deployment").mkdir()
        self.frozen = (ROOT / "deployment/selected.env").read_text()
        (self.root / "deployment/selected.env").write_text(self.frozen)
        self.model = self.root / "models/model"
        self.model.mkdir(parents=True)
        self.cache = self.root / "data/engines"

    def generate(self, **kwargs):
        with patch.object(config.os, "geteuid", return_value=1001):
            return config.configure(self.root, str(self.model), str(self.cache), **kwargs)

    def test_private_local_paths_preserve_every_frozen_tuning_value(self):
        self.cache.mkdir(parents=True)
        marker = self.cache / "existing.engine"
        marker.write_bytes(b"unchanged")
        target = self.generate()
        values = config.parse_env(target)
        frozen_values = config.parse_env(self.root / "deployment/selected.env")
        for key in frozen_values.keys() - config.PATH_KEYS:
            self.assertEqual(values[key], frozen_values[key], key)
        self.assertEqual(values["COSMOS_MAX_IMAGE_TOKENS_PER_IMAGE"], "512")
        self.assertEqual(values["COSMOS_MAX_INPUT_LEN"], "1024")
        self.assertEqual(values["COSMOS_MAX_KV_CAPACITY"], "1664")
        self.assertEqual(values["COSMOS_ENCODER_CACHE_BYTES"], "0")
        self.assertEqual(values["COSMOS_STATIC_CLOCKS"], "0")
        self.assertEqual(values["COSMOS_TOP_P"], "1")
        self.assertEqual(values["COSMOS_MODEL_DIR"], str(self.model))
        for key in config.PATH_KEYS:
            self.assertTrue(Path(values[key]).is_dir(), key)
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o600)
        self.assertEqual(marker.read_bytes(), b"unchanged")
        self.assertEqual((self.root / "deployment/selected.env").read_text(), self.frozen)

    def test_requires_normal_user_existing_model_and_explicit_overwrite(self):
        with patch.object(config.os, "geteuid", return_value=0):
            with self.assertRaisesRegex(ValueError, "without sudo"):
                config.configure(self.root, str(self.model), str(self.cache))
        self.assertFalse((self.root / "deployment/local.env").exists())
        self.model.rmdir()
        with self.assertRaisesRegex(ValueError, "existing local model"):
            self.generate()
        self.assertFalse(self.cache.exists())
        self.model.mkdir()
        first = self.generate().read_bytes()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.generate()
        self.assertEqual((self.root / "deployment/local.env").read_bytes(), first)
        self.generate(data_dir=str(self.root / "alternate-data"), force=True)
        self.assertEqual(config.parse_env(self.root / "deployment/local.env")["TMPDIR"], str(self.root / "alternate-data/tmp"))

    def test_unsafe_path_refused_before_any_directory_creation(self):
        for bad in ("relative", "/tmp/with space", "/tmp/percent%u", "/tmp/line\nbreak", "/tmp/$(command)", "/tmp/quote'", "/tmp/semi;"):
            with self.subTest(path=bad), patch.object(config.os, "geteuid", return_value=1001):
                with self.assertRaisesRegex(ValueError, "absolute path"):
                    config.configure(self.root, str(self.model), bad)
        self.assertFalse((self.root / "deployment/local.env").exists())
        self.assertFalse(self.cache.exists())
        unsafe = self.root / "unsafe space"
        unsafe.mkdir()
        (self.root / "safe-link").symlink_to(unsafe)
        with self.assertRaisesRegex(ValueError, "resolves"):
            config.safe_path(self.root / "safe-link", "link")

    def test_literal_parser_rejects_execution_duplicates_and_systemd_injection(self):
        target = self.root / "deployment/bad.env"
        cases = ("UNRELATED=$(touch /tmp/unwanted)\n", "export COSMOS_PROFILE=rtn-v1\n", "COSMOS_PROFILE=rtn-v1\nCOSMOS_PROFILE=fp16\n",
                 "COSMOS_MODEL_DIR=/tmp/hello world\n", "LD_PRELOAD=/tmp/evil.so\n", "COSMOS_STATIC_CLOCKS=\"1\"\n")
        for text in cases:
            with self.subTest(text=text):
                target.write_text(text)
                with self.assertRaises(ValueError):
                    config.parse_env(target)
        values = config.parse_env(self.generate())
        values["COSMOS_CACHE_DIR"] = "/tmp/%u"
        with self.assertRaisesRegex(ValueError, "absolute path"):
            config.validate_values(values)

    def test_legacy_fp16_unset_builder_overrides_remain_supported(self):
        values = config.parse_env(self.generate())
        values.update(COSMOS_PROFILE="fp16", COSMOS_MAX_IMAGE_TOKENS="", COSMOS_ENGINE_MAX_IMAGE_TOKENS_PER_IMAGE="")
        config.validate_values(values)
        values["COSMOS_PROFILE"] = "rtn-v1"
        with self.assertRaisesRegex(ValueError, "integer"):
            config.validate_values(values)

    def test_account_is_existing_nonroot_and_has_safe_primary_group(self):
        for bad in (None, "", "a%u", "a\nb", "a;whoami"):
            with self.subTest(name=bad), self.assertRaises(ValueError):
                config.service_account(bad)
        with self.assertRaisesRegex(ValueError, "not be root"):
            config.service_account("root")
        with patch.object(config.pwd, "getpwnam", side_effect=KeyError):
            with self.assertRaisesRegex(ValueError, "does not exist"):
                config.service_account("missing")
        entry = Mock(pw_uid=1001, pw_gid=1002, pw_name="operator")
        with patch.object(config.pwd, "getpwnam", return_value=entry), \
             patch.object(config.grp, "getgrgid", return_value=Mock(gr_name="operators")):
            self.assertEqual(config.service_account("operator"), ("operator", "operators"))

    def test_local_environment_precedes_provenance_and_explicit_override_wins(self):
        local = self.generate()
        override = self.root / "deployment/override.env"
        override.write_bytes(local.read_bytes())
        with patch.object(config, "service_account", return_value=("operator", "operators")), \
             patch.object(config, "check_service_access") as access:
            self.assertEqual(config.validated_install(self.root, "operator")[2], local)
            self.assertEqual(config.validated_install(self.root, "operator", str(override))[2], override)
            (self.root / "deployment/selected.env").write_bytes(local.read_bytes())
            local.unlink()
            self.assertEqual(config.validated_install(self.root, "operator")[2], self.root / "deployment/selected.env")
            self.assertEqual(access.call_count, 3)

    def test_lan_ip_must_be_an_actual_local_ipv4(self):
        result = Mock(stdout=json.dumps([{"addr_info": [{"local": "192.0.2.7"}]}]))
        with patch.object(config.shutil, "which", return_value="/usr/sbin/ip"), \
             patch.object(config.subprocess, "run", return_value=result) as run:
            self.assertEqual(config.local_ipv4("192.0.2.7"), "192.0.2.7")
            run.assert_called_once_with(["/usr/sbin/ip", "-j", "-4", "address", "show"], check=True, capture_output=True, text=True)
            for bad in ("192.0.2.8", "127.0.0.1", "0.0.0.0", "224.0.0.1", "169.254.1.2", "::1", "1.2.3.4;id"):
                with self.subTest(ip=bad), self.assertRaises(ValueError):
                    config.local_ipv4(bad)

    def test_service_probe_checks_runtime_logs_even_with_custom_data_directory(self):
        values = config.parse_env(self.generate(data_dir=str(self.root / "elsewhere")))
        (self.root / "results").mkdir()
        (self.root / "data/logs").mkdir(parents=True)
        entry = Mock(pw_uid=1001)
        with patch.object(config.pwd, "getpwnam", return_value=entry), \
             patch.object(config.os, "geteuid", return_value=1001), \
             patch.object(config.os, "access", side_effect=lambda path, mode: path != self.root / "data/logs"):
            with self.assertRaisesRegex(ValueError, "data/logs"):
                config.check_service_access(self.root, "operator", values)


if __name__ == "__main__":
    unittest.main()
