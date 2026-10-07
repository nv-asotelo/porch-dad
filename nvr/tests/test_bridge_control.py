# SPDX-License-Identifier: Apache-2.0
"""Hermetic service lifecycle checks: no systemd, robot, network or hardware."""
from __future__ import annotations

import os
import json
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reachy"))
from bridge_control import BridgeControl, SUDO, SYSTEMCTL, UNIT


def service(state="active", pid=123, *, result="success", load="loaded"):
    return subprocess.CompletedProcess([], 0, (
        f"LoadState={load}\nActiveState={state}\nSubState={state}\n"
        f"MainPID={pid}\nResult={result}\n"
    ), "")


class ScriptedRunner:
    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


class BridgeControlTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.lock = Path(self.directory.name) / ".bridge-control.lock"

    def controller(self, *answers, timeout=1):
        self.runner = ScriptedRunner(*answers)
        return BridgeControl(runner=self.runner, lock_path=self.lock, timeout=timeout)

    def test_active_does_not_claim_frame_readiness(self):
        result = self.controller(service()).status()
        self.assertEqual(result["state"], "active")
        self.assertTrue(result["active"])
        self.assertFalse(result["released"])
        self.assertIn("not yet verified", result["message"])
        self.assertEqual(len(self.runner.calls), 1)
        command, options = self.runner.calls[0]
        self.assertEqual(command[:3], [SYSTEMCTL, "show", UNIT])
        self.assertNotIn("shell", options)

    def test_missing_unit_is_not_released(self):
        result = self.controller(service("inactive", 0, load="not-found")).status()
        self.assertEqual(result["state"], "unavailable")
        self.assertEqual(result["load_state"], "not-found")
        self.assertFalse(result["released"])

    def test_release_uses_only_fixed_stop_command(self):
        result = self.controller(service(), service("inactive", 0)).change("release")
        self.assertTrue(result["ok"])
        self.assertTrue(result["released"])
        self.assertEqual(self.runner.calls[0][0], [SUDO, "-n", SYSTEMCTL, "stop", UNIT])
        self.assertEqual(len(self.runner.calls), 2)
        self.assertEqual(stat.S_IMODE(self.lock.stat().st_mode), 0o600)
        self.assertIn("boot enablement are unchanged", result["message"])

    def test_resume_verifies_positive_pid(self):
        result = self.controller(service(), service()).change("resume")
        self.assertTrue(result["ok"])
        self.assertEqual(self.runner.calls[0][0], [SUDO, "-n", SYSTEMCTL, "start", UNIT])
        self.assertIn("does not verify video", result["message"])

    def test_no_pid_is_not_confirmed_resume(self):
        result = self.controller(service(), service("active", 0)).change("resume")
        self.assertFalse(result["ok"])
        self.assertEqual(result["state"], "transitioning")

    def test_inactive_with_pid_is_not_released(self):
        result = self.controller(service(), service("inactive", 42)).change("release")
        self.assertFalse(result["ok"])
        self.assertFalse(result["released"])

    def test_failed_stop_even_if_process_exited_is_not_success(self):
        failure = subprocess.CompletedProcess([], 1, "", "stop failed")
        result = self.controller(failure, service("inactive", 0)).change("release")
        self.assertFalse(result["ok"])
        self.assertFalse(result["released"])
        self.assertEqual(result["state"], "failed")

    def test_forced_stop_result_is_not_clean_release(self):
        result = self.controller(service(), service("inactive", 0, result="timeout")).change("release")
        self.assertFalse(result["ok"])
        self.assertEqual(result["state"], "failed")
        self.assertFalse(result["released"])

    def test_failed_unit_is_not_released(self):
        result = self.controller(service("failed", 0, result="signal")).status()
        self.assertEqual(result["state"], "failed")
        self.assertFalse(result["released"])

    def test_incomplete_status_fails_closed(self):
        malformed = subprocess.CompletedProcess([], 0, "LoadState=loaded\nMainPID=nope\n", "")
        result = self.controller(malformed).status()
        self.assertEqual(result["state"], "unavailable")

    def test_invalid_action_runs_nothing(self):
        control = self.controller()
        for action in ["restart", "start; reboot", "release --now", ""]:
            with self.assertRaises(ValueError):
                control.change(action)
        self.assertFalse(self.runner.calls)
        self.assertFalse(self.lock.exists())

    def test_sudo_error_is_bounded(self):
        denied = subprocess.CompletedProcess([], 1, "", "denied\n" * 1000)
        result = self.controller(denied, service()).change("release")
        self.assertFalse(result["ok"])
        self.assertLess(len(result["message"]), 300)
        self.assertNotIn("\n", result["message"])

    def test_action_timeout_never_reports_success(self):
        timeout = subprocess.TimeoutExpired("systemctl", 1)
        control = self.controller(timeout)
        result = control.change("release")
        self.assertFalse(result["ok"])
        self.assertFalse(result["released"])
        self.assertEqual(result["state"], "failed")
        self.assertEqual(control.intent()["desired"], "stopped")

    def test_status_os_error_is_unavailable(self):
        result = self.controller(FileNotFoundError("systemctl missing")).status()
        self.assertEqual(result["state"], "unavailable")

    def test_lock_symlink_is_rejected_without_touching_target(self):
        target = Path(self.directory.name) / "important"
        target.write_text("leave alone")
        target.chmod(0o644)
        self.lock.symlink_to(target)
        control = self.controller()
        result = control.change("release")
        self.assertFalse(result["ok"])
        self.assertFalse(self.runner.calls)
        self.assertEqual(target.read_text(), "leave alone")
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o644)

    def test_hardlinked_lock_is_rejected(self):
        target = Path(self.directory.name) / "important"
        target.write_text("leave alone")
        os.link(target, self.lock)
        result = self.controller().change("resume")
        self.assertFalse(result["ok"])
        self.assertFalse(self.runner.calls)

    def test_two_controllers_serialize_mutations(self):
        started = threading.Event()
        unblock = threading.Event()
        second_started = threading.Event()
        first_result = []
        second_result = []

        def first_runner(command, **kwargs):
            if command[0] == SUDO:
                started.set()
                self.assertTrue(unblock.wait(1))
                return service()
            return service("inactive", 0)

        def second_runner(command, **kwargs):
            if command[0] == SUDO:
                second_started.set()
            return service()

        first = BridgeControl(runner=first_runner, lock_path=self.lock, timeout=2)
        second = BridgeControl(runner=second_runner, lock_path=self.lock, timeout=2)
        t1 = threading.Thread(target=lambda: first_result.append(first.change("release")))
        t2 = threading.Thread(target=lambda: second_result.append(second.change("resume")))
        t1.start()
        self.assertTrue(started.wait(1))
        t2.start()
        self.assertFalse(second_started.wait(0.1))
        unblock.set()
        t1.join(2)
        t2.join(2)
        self.assertFalse(t1.is_alive())
        self.assertFalse(t2.is_alive())
        self.assertTrue(first_result[0]["ok"])
        self.assertTrue(second_result[0]["ok"])

    def test_lock_wait_is_bounded_and_does_not_run_command(self):
        import fcntl
        self.lock.touch(mode=0o600)
        with self.lock.open("r+") as held:
            fcntl.flock(held, fcntl.LOCK_EX)
            control = self.controller(timeout=0.1)
            started = time.monotonic()
            result = control.change("resume")
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertEqual(result["state"], "transitioning")
        self.assertFalse(result["ok"])
        self.assertFalse(self.runner.calls)

    def test_intent_is_saved_before_service_action(self):
        observed = []
        control = BridgeControl(lock_path=self.lock, runner=lambda *args, **kw: None)

        def runner(command, **kwargs):
            observed.append(control.intent())
            return service("inactive", 0)

        control._runner = runner
        before = time.time()
        self.assertTrue(control.change("release")["ok"])
        self.assertEqual(observed[0]["action"], "stop")
        self.assertEqual(observed[0]["desired"], "stopped")
        self.assertEqual(observed[0]["by"], "bridge controls")
        self.assertGreaterEqual(observed[0]["at"], before)
        self.assertEqual(stat.S_IMODE(self.lock.with_suffix(".json").stat().st_mode), 0o600)
        self.assertEqual(list(Path(self.directory.name).glob("*.tmp")), [])

    def test_intent_failure_prevents_service_action_and_cleans_temporary(self):
        control = self.controller()
        with mock.patch("bridge_control.os.replace", side_effect=OSError("read-only filesystem")):
            result = control.change("resume")
        self.assertFalse(result["ok"])
        self.assertFalse(self.runner.calls)
        self.assertEqual(control.intent(), {})
        self.assertEqual(list(Path(self.directory.name).glob("*.tmp")), [])

    def test_intent_symlink_is_never_followed_or_overwritten(self):
        target = Path(self.directory.name) / "important"
        target.write_text("leave alone")
        sidecar = self.lock.with_suffix(".json")
        sidecar.symlink_to(target)
        control = self.controller()
        self.assertEqual(control.intent(), {})
        self.assertFalse(control.change("release")["ok"])
        self.assertFalse(self.runner.calls)
        self.assertEqual(target.read_text(), "leave alone")
        self.assertTrue(sidecar.is_symlink())

    def test_intent_missing_or_invalid_has_no_valid_record(self):
        control = self.controller(service(), service())
        self.assertEqual(control.intent(), {})
        self.assertTrue(control.change("resume")["ok"])
        good = control.intent()
        self.assertEqual(good["desired"], "running")
        self.assertEqual(good["action"], "start")
        sidecar = self.lock.with_suffix(".json")
        invalid_records = [
            {**good, "at": float("nan")}, {**good, "at": float("inf")},
            {**good, "at": True}, {**good, "at": -1},
            {**good, "at": 10 ** 1000},
            {**good, "at": "2026-10-06"}, {**good, "desired": "stopped"},
            {**good, "by": "someone else"}, {**good, "note": "arbitrary text"},
            {**good, "address": "must not be present"}, [], None,
        ]
        for record in invalid_records:
            sidecar.write_text(json.dumps(record))
            self.assertEqual(control.intent(), {}, record)
        for raw in ["not json", " " * 4097]:
            sidecar.write_text(raw)
            self.assertEqual(control.intent(), {})
        sidecar.write_text(json.dumps(good))
        sidecar.chmod(0o644)
        self.assertEqual(control.intent(), {})

    def test_intent_fifo_does_not_block_reader(self):
        os.mkfifo(self.lock.with_suffix(".json"), 0o600)
        control = self.controller()
        self.assertEqual(control.intent(), {})
        self.assertFalse(control.change("release")["ok"])
        self.assertFalse(self.runner.calls)

    def test_resume_replaces_previous_stop_intent(self):
        control = self.controller(service(), service("inactive", 0), service(), service())
        self.assertTrue(control.change("release")["ok"])
        old = control.intent()
        self.assertTrue(control.change("resume")["ok"])
        new = control.intent()
        self.assertEqual(new["desired"], "running")
        self.assertGreaterEqual(new["at"], old["at"])


if __name__ == "__main__":
    unittest.main()
