# SPDX-License-Identifier: Apache-2.0
"""Control the local shared camera bridge, never the Reachy robot daemon.

Stopping this fixed systemd unit sends its configured SIGTERM so the bridge can
close its WebRTC session. It preserves the robot address and boot enablement.
Both front ends share a private flock file, so conflicting requests serialize.
"""
from __future__ import annotations

import fcntl
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import time


UNIT = "reachy-mjpeg-bridge.service"
SYSTEMCTL = "/usr/bin/systemctl"
SUDO = "/usr/bin/sudo"
_PROPERTIES = "LoadState,ActiveState,SubState,MainPID,Result"
_INTENTS = {
    "release": {
        "desired": "stopped", "action": "stop", "by": "bridge controls",
        "note": "Bridge released; robot settings and boot enablement are preserved.",
    },
    "resume": {
        "desired": "running", "action": "start", "by": "bridge controls",
        "note": "Bridge resumed; fresh camera frames still need verification.",
    },
}


def _bounded_error(error: object) -> str:
    # systemctl/sudo diagnostics can be multiline; keep UI responses bounded.
    return " ".join(str(error).split())[:240]


class BridgeControl:
    """Small synchronous helper; HTTP servers should call it off the event loop."""

    def __init__(self, *, runner=None, lock_path=None, intent_path=None, timeout=20.0):
        self._runner = runner or subprocess.run
        self._lock_path = Path(lock_path) if lock_path is not None else (
            Path(__file__).resolve().parent / ".bridge-control.lock"
        )
        self._intent_path = Path(intent_path) if intent_path is not None else self._lock_path.with_suffix(".json")
        self._timeout = max(0.1, float(timeout))

    def intent(self):
        """Read the last requested state; an intent is not proof the action succeeded."""
        try:
            fd = os.open(self._intent_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as stream:
                info = os.fstat(stream.fileno())
                if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or
                        info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o600):
                    return {}
                raw = stream.read(4097)
            if len(raw) > 4096:
                return {}
            record = json.loads(raw)
            if not isinstance(record, dict) or set(record) != {"desired", "action", "at", "by", "note"}:
                return {}
            timestamp = record["at"]
            if (isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or
                    not math.isfinite(timestamp) or timestamp <= 0):
                return {}
            fields = {key: value for key, value in record.items() if key != "at"}
            return record if fields in _INTENTS.values() else {}
        except (OSError, ValueError, TypeError, OverflowError):
            return {}

    def _record_intent(self, action):
        """Atomic private metadata, written while the shared action lock is held."""
        try:
            existing = self._intent_path.lstat()
        except FileNotFoundError:
            existing = None
        if existing is not None and (
                not stat.S_ISREG(existing.st_mode) or existing.st_nlink != 1 or
                existing.st_uid != os.geteuid()):
            raise OSError("Bridge intent must be a private regular file owned by this user")
        record = dict(_INTENTS[action], at=time.time())
        fd, temporary = tempfile.mkstemp(prefix=self._intent_path.name + ".", suffix=".tmp",
                                         dir=self._intent_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(record, stream, separators=(",", ":"), allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self._intent_path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    @staticmethod
    def _result(state, message, *, active=False, released=False, main_pid=0,
                load_state="unknown", **extra):
        return dict(state=state, active=active, released=released, message=message,
                    unit=UNIT, main_pid=main_pid, load_state=load_state, **extra)

    def _run(self, command, timeout):
        return self._runner(command, capture_output=True, text=True,
                            check=False, timeout=max(0.01, timeout))

    def _status(self, timeout):
        try:
            completed = self._run(
                [SYSTEMCTL, "show", UNIT, "--no-pager", f"--property={_PROPERTIES}"],
                timeout,
            )
        except (OSError, subprocess.SubprocessError) as error:
            return self._result("unavailable", "Cannot inspect the bridge: " + _bounded_error(error))
        if completed.returncode:
            detail = _bounded_error(completed.stderr or completed.stdout or "systemctl failed")
            return self._result("unavailable", "Cannot inspect the bridge: " + detail)
        values = {}
        for line in completed.stdout.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key] = value
        load = values.get("LoadState", "unknown")
        if load != "loaded":
            return self._result("unavailable", "The local Reachy bridge service is not loaded.",
                                load_state=load)
        try:
            pid = int(values["MainPID"])
            if pid < 0:
                raise ValueError("negative PID")
            active_state = values["ActiveState"]
            result = values["Result"]
        except (KeyError, ValueError):
            return self._result("unavailable", "The bridge service returned incomplete status.",
                                load_state=load)
        extra = dict(main_pid=pid, load_state=load, active_state=active_state,
                     sub_state=values.get("SubState", ""), service_result=result)
        if active_state == "failed" or (active_state == "inactive" and result != "success"):
            return self._result("failed", "The bridge service failed; inspect its service log.", **extra)
        if active_state == "active" and pid > 0:
            return self._result("active", "Bridge running; fresh camera frames are not yet verified.",
                                active=True, **extra)
        if active_state == "inactive" and pid == 0 and result == "success":
            return self._result("released", "Bridge stopped. It stays released until an explicit start or reboot.",
                                released=True, **extra)
        if active_state in {"activating", "deactivating", "reloading", "refreshing", "active", "inactive"}:
            return self._result("transitioning", "The bridge service is changing state.", **extra)
        return self._result("unavailable", "The bridge service returned an unknown state.", **extra)

    def status(self):
        """Read process state, without opening a robot connection or checking frames."""
        return self._status(min(3.0, self._timeout))

    def _acquire(self, deadline):
        # Do not follow an attacker-created symlink or lock a shared/hardlinked file.
        fd = os.open(self._lock_path,
                     os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.geteuid():
                raise OSError("Bridge lock must be a private regular file owned by this user")
            os.fchmod(fd, 0o600)
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    current = os.stat(self._lock_path, follow_symlinks=False)
                    if (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino):
                        raise OSError("Bridge lock changed while acquiring it")
                    return fd
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Another bridge action is still running")
                    time.sleep(min(0.05, max(0, deadline - time.monotonic())))
        except BaseException:
            os.close(fd)
            raise

    def change(self, action):
        """Release or resume the shared bridge. Success never implies frame readiness."""
        if action not in {"release", "resume"}:
            raise ValueError("Bridge action must be 'release' or 'resume'")
        deadline = time.monotonic() + self._timeout
        try:
            fd = self._acquire(deadline)
        except TimeoutError:
            return self._result("transitioning", "Another bridge action is still running; retry shortly.", ok=False)
        except OSError as error:
            return self._result("unavailable", "Cannot lock the bridge controls: " + _bounded_error(error), ok=False)
        try:
            try:
                # Record what the user asked before attempting it, including partial failures.
                self._record_intent(action)
            except (OSError, ValueError) as error:
                return self._result("unavailable", "Cannot save bridge intent: " + _bounded_error(error), ok=False)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return self._result("transitioning", "Bridge action timed out before starting.", ok=False)
            verb = "stop" if action == "release" else "start"
            try:
                # Leave time for the status verification inside the total deadline.
                completed = self._run([SUDO, "-n", SYSTEMCTL, verb, UNIT], max(0.01, remaining - 1.0))
            except (OSError, subprocess.SubprocessError) as error:
                return self._result("failed", "Bridge action did not complete: " + _bounded_error(error), ok=False)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return self._result("failed", "Bridge action exceeded its verification deadline.", ok=False)
            observed = self._status(min(3.0, remaining))
            if completed.returncode:
                detail = _bounded_error(completed.stderr or completed.stdout or "systemctl failed")
                # Even if the PID disappeared, a forced/failed stop is not a clean release.
                return dict(observed, ok=False, state="failed", released=False,
                            message="Bridge action failed: " + detail)
            target = "released" if action == "release" else "active"
            ok = observed["state"] == target
            observed["ok"] = ok
            if ok and action == "release":
                observed["message"] = (
                    "Bridge released. Robot address and boot enablement are unchanged; "
                    "it stays stopped until an explicit start or reboot."
                )
            elif ok:
                observed["message"] = "Bridge resumed. Waiting for fresh camera frames; process state alone does not verify video."
            elif observed["state"] not in {"failed", "unavailable"}:
                observed["message"] = "The requested bridge state is not yet verified; refresh status before retrying."
            return observed
        finally:
            os.close(fd)


BRIDGE_CONTROL = BridgeControl()
