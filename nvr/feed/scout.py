"""Moorebot Scout control for the porch-dad command centre.

Talks to `scout-bridge` on this box, not to the robot. The Scout speaks ROS1, and putting a ROS
client in this process would drag the whole ROS runtime into a FastAPI app on a board that counts
megabytes. The bridge already holds that, costs ~48 MB, and exposes plain HTTP.

Everything here is deliberately thin. The bridge does the clamping and owns the drive loop; this
just forwards and shapes the replies for the panel.
"""
from __future__ import annotations

import requests

# Matches the bridge's own limits (nvr/scout/scout_mjpeg_bridge.py). Duplicated rather than
# imported because that module lives in a ROS container, not in this venv - but the bridge clamps
# independently, so these are a UI courtesy and not the safety boundary.
MAX_LINEAR = 0.3
MAX_ANGULAR = 1.0
MAX_DURATION = 3.0


class Scout:
    """Minimal client for one scout-bridge."""

    def __init__(self, base_url: str, timeout: float = 6.0):
        self.base = (base_url or "").rstrip("/")
        self.timeout = timeout

    def state(self) -> dict:
        """Bridge health, reshaped for the panel. Never raises; the card degrades instead."""
        if not self.base:
            return {"enabled": False}
        try:
            r = requests.get(f"{self.base}/healthz", timeout=self.timeout)
            r.raise_for_status()
            h = r.json()
        except (requests.RequestException, ValueError) as e:
            return {"enabled": True, "reachable": False, "error": str(e)[:120]}

        # `frames` counts callbacks, not distinct pictures, so `live` (which the bridge derives
        # from a frame-content digest) is the honest signal that the picture is actually moving.
        return {
            "enabled": True,
            "reachable": True,
            "ros_connected": bool(h.get("ros_connected")),
            "live": bool(h.get("live")),
            "frames": h.get("frames"),
            "stale_s": h.get("stale_s"),
            "driving": bool(h.get("driving")),
            "blocked_by": h.get("blocked_by"),
            "error": h.get("error"),
        }

    def drive(self, x=0.0, y=0.0, yaw=0.0, duration=0.6) -> tuple[bool, str]:
        """Bounded motion.

        AXES ARE NOT THE ROS CONVENTION: +y is FORWARD and +x is a sideways strafe. Verified on the
        robot - a +0.10 y command for 0.6 s moved the camera bodily toward the scene - and before
        that from the vendor firmware, where obstacle avoidance clamps y alone against the
        forward-facing ToF sensor.

        Motion is sustained by repetition, so there is no latch to leave on: the robot's own
        MotorNode zeroes velocity and cuts motor power when cmd_vel stops arriving.
        """
        if not self.base:
            return False, "scout_bridge_url is not configured"
        body = {
            "x": max(-MAX_LINEAR, min(MAX_LINEAR, float(x))),
            "y": max(-MAX_LINEAR, min(MAX_LINEAR, float(y))),
            "yaw": max(-MAX_ANGULAR, min(MAX_ANGULAR, float(yaw))),
            "duration": max(0.0, min(MAX_DURATION, float(duration))),
        }
        return self._post("/drive", body)

    def stop(self) -> tuple[bool, str]:
        return self._post("/stop", None)

    def _post(self, path: str, body) -> tuple[bool, str]:
        try:
            r = requests.post(f"{self.base}{path}", json=body, timeout=self.timeout)
            msg = ""
            try:
                msg = (r.json() or {}).get("message") or ""
            except ValueError:
                msg = r.text[:120]
            if r.status_code >= 400:
                return False, msg or f"HTTP {r.status_code}"
            return True, msg or "ok"
        except requests.RequestException as e:
            return False, f"bridge unreachable: {e}"
