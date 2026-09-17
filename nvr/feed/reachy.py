"""Reachy Mini control for the porch-dad command centre.

Talks to the robot's own daemon REST API rather than proxying Home Assistant. HA's Reachy
integration exposes the same capabilities, but routing through it would make this page depend on
Home Assistant being up - and the command centre's job is to work when other things are off. The
camera preview already learned that lesson the hard way: it was wired to the Live VLM WebUI and
went dark whenever that was switched off to save load.

Everything here is a thin wrapper over documented daemon endpoints. Angles are degrees at this
boundary and radians on the wire, matching the robot's published limits.
"""
from __future__ import annotations

import math

import requests

MOTOR_MODES = ("enabled", "disabled", "gravity_compensation")

# Published safety limits (Reachy Mini "Core Concepts").
LIMITS_DEG = {"pitch": (-40.0, 40.0), "roll": (-40.0, 40.0),
              "yaw": (-180.0, 180.0), "body_yaw": (-160.0, 160.0)}
# Antennas park here rather than 0: at the null position a servo hunts and visibly twitches.
ANTENNA_PARK_DEG = 10.0


class Reachy:
    """Minimal client for one Reachy Mini daemon."""

    def __init__(self, base_url: str, timeout: float = 8.0):
        self.base = (base_url or "").rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------------ plumbing
    def _get(self, path: str, default=None):
        try:
            r = requests.get(f"{self.base}{path}", timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError):
            return default

    def _post(self, path: str, payload=None) -> tuple[bool, str]:
        try:
            r = requests.post(f"{self.base}{path}", json=payload, timeout=self.timeout)
            if r.status_code >= 400:
                return False, f"HTTP {r.status_code}: {r.text[:120]}"
            return True, "ok"
        except requests.RequestException as e:
            return False, f"robot unreachable: {e}"

    # ------------------------------------------------------------------ read
    def state(self) -> dict:
        """Everything the panel shows, in one call-set. Never raises: the panel degrades instead."""
        status = self._get("/api/daemon/status") or {}
        if not status:
            return {"reachable": False}

        backend = status.get("backend_status") or {}
        out = {
            "reachable": True,
            "name": status.get("robot_name"),
            "version": status.get("version"),
            "state": status.get("state"),
            # `ready` is the robot's awake flag: False until woken, not an error.
            "awake": bool(backend.get("ready")),
            "motor_mode": backend.get("motor_control_mode"),
            "wireless": status.get("wireless_version"),
            "hardware_id": status.get("hardware_id"),
        }

        full = self._get("/api/state/full") or {}
        pose = full.get("head_pose") or {}
        if pose:
            out["pose_deg"] = {k: round(math.degrees(float(pose.get(k, 0.0))), 1)
                               for k in ("roll", "pitch", "yaw")}
            out["body_yaw_deg"] = round(math.degrees(float(full.get("body_yaw") or 0.0)), 1)
            out["antennas_deg"] = [round(math.degrees(float(a)), 1)
                                   for a in (full.get("antennas_position") or [])]

        doa = self._get("/api/state/doa") or {}
        if doa:
            # 0 rad is left, pi/2 front/back, pi right.
            out["doa_deg"] = round(math.degrees(float(doa.get("angle") or 0.0)), 1)
            out["speech_detected"] = bool(doa.get("speech_detected"))

        vol = self._get("/api/volume/current") or {}
        mic = self._get("/api/volume/microphone/current") or {}
        out["speaker_volume"] = vol.get("volume")
        out["mic_volume"] = mic.get("volume")

        app = self._get("/api/apps/current-app-status")
        out["app"] = (app or {}).get("name") if isinstance(app, dict) else None
        out["move_running"] = bool(self._get("/api/move/running") or [])

        loop = backend.get("control_loop_stats") or {}
        out["control_hz"] = round(float(loop.get("mean_control_loop_frequency") or 0.0), 1)
        out["control_errors"] = loop.get("nb_error")
        return out


    def wake(self):
        return self._post("/api/move/play/wake_up")

    def sleep(self):
        return self._post("/api/move/play/goto_sleep")


    def set_motor_mode(self, mode: str):
        if mode not in MOTOR_MODES:
            return False, f"mode must be one of {', '.join(MOTOR_MODES)}"
        return self._post(f"/api/motors/set_mode/{mode}")

    def set_volume(self, which: str, value: int):
        value = max(0, min(100, int(value)))
        path = "/api/volume/set" if which == "speaker" else "/api/volume/microphone/set"
        ok, msg = self._post(path, {"volume": value})
        return ok, (f"{which} volume {value}%" if ok else msg)


    def look(self, pitch=None, yaw=None, roll=None, body_yaw=None, duration=1.0):
        """Move the head. Degrees in, radians out. Omitted axes hold their current value.

        Positive pitch tilts the head DOWN - verified from the camera, not assumed - so a caller
        meaning "look up" sends a negative number.
        """
        st = self.state()
        if not st.get("reachable"):
            return False, "robot unreachable"
        if (st.get("motor_mode") or "").lower() == "disabled":
            # The daemon accepts moves with motors off and reports success while nothing turns.
            return False, "motors are disabled - enable them first"

        cur = st.get("pose_deg") or {}
        vals = {
            "pitch": cur.get("pitch", 0.0) if pitch is None else float(pitch),
            "roll": cur.get("roll", 0.0) if roll is None else float(roll),
            "yaw": cur.get("yaw", 0.0) if yaw is None else float(yaw),
            "body_yaw": (st.get("body_yaw_deg") or 0.0) if body_yaw is None else float(body_yaw),
        }
        notes = []
        for k, v in vals.items():
            lo, hi = LIMITS_DEG[k]
            c = max(lo, min(hi, v))
            if c != v:
                notes.append(f"{k} clamped to {c:g}°")
            vals[k] = c

        payload = {
            "head_pose": {"x": 0.0, "y": 0.0, "z": 0.0,
                          "roll": math.radians(vals["roll"]),
                          "pitch": math.radians(vals["pitch"]),
                          "yaw": math.radians(vals["yaw"])},
            "body_yaw": math.radians(vals["body_yaw"]),
            "duration": max(0.2, min(5.0, float(duration))),
            "interpolation": "minjerk",
        }
        ok, msg = self._post("/api/move/goto", payload)
        if not ok:
            return False, msg
        return True, "; ".join(notes) or "moving"

    def center(self):
        st = self.state()
        if (st.get("motor_mode") or "").lower() == "disabled":
            return False, "motors are disabled - enable them first"
        payload = {
            "head_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            "body_yaw": 0.0,
            "antennas": [math.radians(ANTENNA_PARK_DEG)] * 2,
            "duration": 1.0,
            "interpolation": "minjerk",
        }
        ok, msg = self._post("/api/move/goto", payload)
        return ok, ("centred" if ok else msg)

    def look_at_voice(self):
        """Turn the head toward the last detected speaker.

        DoA is reported as 0 rad = left, pi/2 = front/back, pi = right, so the useful head yaw is
        the angle mapped onto the robot's own yaw convention.
        """
        st = self.state()
        if st.get("doa_deg") is None:
            return False, "no direction-of-arrival reading available"
        yaw = round(90.0 - float(st["doa_deg"]), 1)
        ok, msg = self.look(yaw=yaw, duration=0.8)
        return ok, (f"turning to {yaw:g}° (voice at {st['doa_deg']:g}°)" if ok else msg)
