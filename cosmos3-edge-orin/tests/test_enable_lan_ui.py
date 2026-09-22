"""Execute LAN helper against inert commands and temporary unit paths only."""
import grp
import json
import os
from pathlib import Path
import pwd
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LanUiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.scripts = self.root / "scripts"
        self.scripts.mkdir()
        self.units = self.root / "units"
        self.units.mkdir()
        self.commands = self.root / "commands"
        self.commands.mkdir()
        self.log = self.root / "commands.jsonl"
        self.account = pwd.getpwuid(os.getuid()) if os.getuid() else pwd.getpwnam("nobody")
        self.group = grp.getgrgid(self.account.pw_gid).gr_name
        for relative in ("deployment", "models/selected", "data/engines"):
            (self.root / relative).mkdir(parents=True, exist_ok=True)
        python = self.root / "external/TensorRT-Edge-LLM/.venv/bin/python"
        python.parent.mkdir(parents=True)
        python.write_text("#!/bin/sh\nexit 99\n")
        python.chmod(0o755)
        (self.root / "deployment/local.env").write_text(
            f"COSMOS_PROFILE=rtn-v1\nCOSMOS_MODEL_DIR={self.root}/models/selected\n"
            f"COSMOS_CACHE_DIR={self.root}/data/engines\nCOSMOS_STATIC_CLOCKS=0\n")
        shutil.copyfile(ROOT / "scripts/configure_deployment.py", self.scripts / "configure_deployment.py")
        self.unit = self.units / "cosmos-edge-ui.service"
        self.unit.write_text(f"[Service]\nUser={self.account.pw_name}\nGroup={self.group}\nExecStart=/usr/bin/python3 placeholder.py\n")
        source = (ROOT / "scripts/enable_lan_ui.sh").read_text()
        guard = "[[ $EUID == 0 && $(uname -m) == aarch64 ]] || { echo 'Run as root on the Orin.' >&2; exit 2; }"
        self.assertEqual(source.count(guard), 1)
        source = source.replace(guard, ": # Test-only platform guard bypass")
        source = source.replace("/etc/systemd/system", str(self.units))
        source = source.replace("/usr/bin/python3", shlex.quote(sys.executable))
        self.helper = self.scripts / "enable_lan_ui.sh"
        self.helper.write_text(source)
        recorder = self.root / "fake.py"
        recorder.write_text('''import json, os, pathlib, shutil, sys, uuid
name, *args = sys.argv[1:]
with open(os.environ['LAN_TEST_LOG'], 'a') as output:
    output.write(json.dumps([name, *args]) + '\\n')
if name == 'ip':
    print(json.dumps([{'addr_info': [{'local': '192.0.2.7'}]}]))
elif name == 'openssl' and args[0] == 'req':
    identity = str(uuid.uuid4())
    for flag in ('-keyout', '-out'):
        pathlib.Path(args[args.index(flag)+1]).write_text(identity)
elif name == 'install':
    dirs = '-d' in args
    paths = []
    mode = 0o755
    i = 0
    while i < len(args):
        if args[i] in ('-o', '-g', '-m'):
            if args[i] == '-m': mode = int(args[i+1], 8)
            i += 2
        elif args[i] == '-d': i += 1
        else: paths.append(args[i]); i += 1
    if dirs:
        for value in paths:
            pathlib.Path(value).mkdir(parents=True, exist_ok=True)
            pathlib.Path(value).chmod(mode)
    else:
        shutil.copyfile(*paths)
        pathlib.Path(paths[-1]).chmod(mode)
elif name == 'systemd-analyze' and os.environ.get('LAN_VERIFY_FAIL'):
    sys.exit(9)
''')
        for name in ("ip", "openssl", "install", "systemd-analyze", "systemctl"):
            executable = self.commands / name
            executable.write_text("#!/bin/sh\nexec " + shlex.quote(sys.executable) + " " + shlex.quote(str(recorder)) + " " + name + ' "$@"\n')
            executable.chmod(0o755)
        if os.getuid() == 0:
            for path in [self.root, *self.root.rglob("*")]:
                os.chown(path, self.account.pw_uid, self.account.pw_gid)
        self.env = dict(os.environ, SUDO_USER=self.account.pw_name,
                        PATH=str(self.commands) + os.pathsep + os.environ["PATH"], LAN_TEST_LOG=str(self.log))

    def launch(self, *args, **env):
        return subprocess.run(["bash", str(self.helper), *args], env=dict(self.env, **env), capture_output=True, text=True)

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def test_explicit_lan_setup_always_generates_new_identity_without_restart(self):
        cert = self.root / "deployment/tls/orin.crt"
        first = self.launch("192.0.2.7")
        self.assertEqual(first.returncode, 0, first.stderr)
        identity = cert.read_text()
        second = self.launch("--user", self.account.pw_name, "192.0.2.7")
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertNotEqual(cert.read_text(), identity)
        requests = [call for call in self.calls() if call[:2] == ["openssl", "req"]]
        self.assertEqual(len(requests), 2)
        self.assertTrue(all("subjectAltName=IP:192.0.2.7,DNS:localhost,IP:127.0.0.1" in call for call in requests))
        self.assertTrue(all("jetson.local" not in " ".join(call) for call in requests))
        control = [call for call in self.calls() if call[0] == "systemctl"]
        self.assertEqual(control, [["systemctl", "daemon-reload"], ["systemctl", "daemon-reload"]])
        dropin = (self.units / "cosmos-edge-ui.service.d/lan.conf").read_text()
        self.assertIn("--host 0.0.0.0 --port 8090 --allow-insecure-lan --https-port 8443", dropin)
        self.assertIn(str(self.root), dropin)
        self.assertNotIn("/home/jetson", dropin)

    def test_dry_run_and_invalid_ip_never_write_identity_or_units(self):
        dry = self.launch("--dry-run", "192.0.2.7")
        self.assertEqual(dry.returncode, 0, dry.stderr)
        self.assertIn("fresh certificate", dry.stdout)
        wrong = self.launch("192.0.2.8")
        self.assertNotEqual(wrong.returncode, 0)
        self.assertIn("not assigned", wrong.stderr)
        self.assertFalse((self.root / "deployment/tls").exists())
        self.assertFalse((self.units / "cosmos-edge-ui.service.d").exists())
        self.assertTrue(all(call[0] == "ip" for call in self.calls()))

    def test_account_mismatch_and_verification_failure_preserve_installed_unit(self):
        before = self.unit.read_bytes()
        result = self.launch("192.0.2.7", LAN_VERIFY_FAIL="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.unit.read_bytes(), before)
        self.assertFalse((self.units / "cosmos-edge-ui.service.d").exists())
        self.assertFalse((self.root / "deployment/tls").exists())
        self.unit.write_text("[Service]\nUser=someone-else\nGroup=other\n")
        result = self.launch("192.0.2.7")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("account differs", result.stderr)
        self.assertFalse((self.units / "cosmos-edge-ui.service.d").exists())


if __name__ == "__main__":
    unittest.main()
