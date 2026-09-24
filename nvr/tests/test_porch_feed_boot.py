#!/usr/bin/env python3
"""Unit tests for porch-feed's pure helpers: boot state that follows intent, and the Reachy card.

    python3 nvr/tests/test_porch_feed_boot.py        (stdlib only; runs anywhere, no Orin needed)

porch_feed.py cannot simply be imported here. It reads /home/orin/nvr/feed/config.yaml at import
time and needs the FastAPI stack, and an import that "works" against the live config is exactly
how a test ends up touching the live box. So the helpers under test are compiled straight out of
the source with ast, one function at a time, into an empty namespace. They are pure on purpose: a
helper that starts reaching for a module global fails here with a NameError, which is the point.
"""
from __future__ import annotations

import __future__
import ast
import unittest
from pathlib import Path

NVR = Path(__file__).resolve().parents[1]
FEED = NVR / "feed" / "porch_feed.py"
FEED_CONFIG = NVR / "feed" / "config.yaml"


def load_helpers(*names: str) -> dict:
    """Compile the named top-level functions from porch_feed.py, and nothing else."""
    tree = ast.parse(FEED.read_text(), filename=str(FEED))
    found = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names}
    missing = sorted(set(names) - set(found))
    if missing:
        raise AssertionError(f"not defined at top level in porch_feed.py: {missing}")
    module = ast.Module(body=[found[n] for n in names], type_ignores=[])
    ns: dict = {}
    # The module relies on postponed annotations (`list[str] | None`); keep that behaviour.
    exec(compile(module, str(FEED), "exec",
                 flags=__future__.annotations.compiler_flag, dont_inherit=True), ns)
    return ns


H = load_helpers("boot_command", "boot_enabled_from", "boot_note", "lan_link",
                 "reachy_health_summary")

VLM = {"key": "vlm", "kind": "systemd", "unit": "live-vlm-webui-fork.service", "port": 8090,
       "boot_follows_intent": True}
LIVE_VISION = {"key": "livevision", "kind": "systemd", "unit": "cosmos-edge-ui.service",
               "port": 8092, "boot_follows_intent": True}


class BootCommand(unittest.TestCase):
    def test_start_and_restart_enable(self):
        for action in ("start", "restart"):
            self.assertEqual(H["boot_command"](VLM, action),
                             ["sudo", "-n", "systemctl", "enable", "live-vlm-webui-fork.service"])

    def test_stop_disables(self):
        self.assertEqual(H["boot_command"](LIVE_VISION, "stop"),
                         ["sudo", "-n", "systemctl", "disable", "cosmos-edge-ui.service"])

    def test_never_without_the_flag(self):
        shim = {"key": "shim", "kind": "systemd", "unit": "cosmos3-edge-shim.service"}
        off = dict(VLM, boot_follows_intent=False)
        for svc in (shim, off):
            for action in ("start", "stop", "restart"):
                self.assertIsNone(H["boot_command"](svc, action), (svc["key"], action))

    def test_never_for_this_page_itself(self):
        # Disabling porch-feed would remove the only browser route back after the next boot.
        feed = {"key": "feed", "kind": "systemd", "unit": "porch-feed.service", "self": True,
                "boot_follows_intent": True}
        for action in ("start", "stop", "restart"):
            self.assertIsNone(H["boot_command"](feed, action))

    def test_never_for_docker(self):
        frigate = {"key": "frigate", "kind": "docker", "name": "frigate",
                   "boot_follows_intent": True}
        for action in ("start", "stop", "restart"):
            self.assertIsNone(H["boot_command"](frigate, action))

    def test_unknown_action_or_missing_unit(self):
        self.assertIsNone(H["boot_command"](VLM, "reload"))
        self.assertIsNone(H["boot_command"](dict(VLM, unit=""), "start"))


class BootEnabledFrom(unittest.TestCase):
    def test_plain_states(self):
        f = H["boot_enabled_from"]
        self.assertIs(f("enabled\n"), True)
        self.assertIs(f("disabled\n"), False)
        self.assertIs(f("masked"), False)

    def test_runtime_enablement_does_not_survive_a_reboot(self):
        self.assertIs(H["boot_enabled_from"]("enabled-runtime\n"), False)

    def test_unclear_is_none_not_off(self):
        f = H["boot_enabled_from"]
        for out in ("static", "indirect", "", None,
                    "Failed to get unit file state for nope.service: No such file or directory"):
            self.assertIsNone(f(out), out)


class BootNote(unittest.TestCase):
    def test_success(self):
        self.assertEqual(H["boot_note"]("start", True), "; starts at boot")
        self.assertEqual(H["boot_note"]("restart", True), "; starts at boot")
        self.assertEqual(H["boot_note"]("stop", True), "; off at boot")

    def test_failure_is_reported_with_its_detail(self):
        note = H["boot_note"]("stop", False, "sudo: a password is required\n")
        self.assertIn("could not disable it at boot", note)
        self.assertIn("sudo: a password is required", note)
        self.assertIn("could not enable it at boot", H["boot_note"]("start", False, ""))
        self.assertTrue(H["boot_note"]("start", False, "").endswith("no detail"))


class LanLink(unittest.TestCase):
    def test_live_vision_keeps_port_path_and_query(self):
        self.assertEqual(H["lan_link"]("https://127.0.0.1:8443/?source=reachy", "192.168.6.252"),
                         "https://192.168.6.252:8443/?source=reachy")

    def test_webui_link_is_unchanged_by_the_refactor(self):
        # What the old hand-built f-string produced for the shipped config.
        self.assertEqual(H["lan_link"]("https://127.0.0.1:8090/?session=reachy", "192.168.6.252"),
                         "https://192.168.6.252:8090/?session=reachy")

    def test_no_port_and_no_path(self):
        self.assertEqual(H["lan_link"]("http://127.0.0.1", "h"), "http://h/")

    def test_unset_or_malformed_gives_no_link(self):
        self.assertIsNone(H["lan_link"]("", "h"))
        self.assertIsNone(H["lan_link"]("https://127.0.0.1:84o3/", "h"))
        self.assertIsNone(H["lan_link"]("https://127.0.0.1:8443/", ""))

    def test_missing_scheme_gives_no_link(self):
        # Parses with no hostname at all; a guessed link would be wrong, so there is none.
        self.assertIsNone(H["lan_link"]("127.0.0.1:8443/?source=reachy", "h"))
        self.assertIsNone(H["lan_link"]("ftp://127.0.0.1:8443/", "h"))

    def test_ipv6_links_host_is_bracketed(self):
        self.assertEqual(H["lan_link"]("https://127.0.0.1:8443/?source=reachy", "fd00::5"),
                         "https://[fd00::5]:8443/?source=reachy")
        self.assertEqual(H["lan_link"]("https://127.0.0.1:8443/", "[fd00::5]"),
                         "https://[fd00::5]:8443/")


class ReachyHealthSummary(unittest.TestCase):
    NEW = {"state": "live", "reason": None, "live": True, "has_frame": True, "frames": 812,
           "last_frame_age_s": 0.2, "stale_s": 0.2, "sessions": 1, "restarts": 0,
           "failed_streak": 0, "blocked_by": None, "mjpeg_clients": 1,
           "audio": {"live": True, "frames": 4000, "last_frame_age_s": 0.0, "listeners": 2},
           "push": {"enabled": True, "state": "pushing", "pushed": 90, "detail": None}}

    def test_new_bridge_live(self):
        s = H["reachy_health_summary"](self.NEW)
        self.assertTrue(s["connected"])
        self.assertEqual((s["state"], s["reason"], s["frames"], s["failed_streak"]),
                         ("live", None, 812, 0))
        self.assertEqual(s["audio"], {"live": True, "listeners": 2})
        self.assertEqual(s["push"], {"state": "pushing"})

    def test_connected_follows_live_not_has_frame(self):
        d = dict(self.NEW, state="reconnecting", live=False, has_frame=True,
                 reason="video stalled for 9s", failed_streak=2)
        s = H["reachy_health_summary"](d)
        self.assertFalse(s["connected"])
        self.assertEqual((s["reason"], s["failed_streak"]), ("video stalled for 9s", 2))

    def test_dormant_reports_why(self):
        d = dict(self.NEW, state="dormant", live=False, has_frame=False,
                 reason="camera held by the robot app 'conversation'", blocked_by="conversation")
        s = H["reachy_health_summary"](d)
        self.assertFalse(s["connected"])
        self.assertEqual(s["reason"], "camera held by the robot app 'conversation'")

    def test_old_bridge_falls_back_to_has_frame(self):
        s = H["reachy_health_summary"]({"has_frame": True, "frames": 3})
        self.assertTrue(s["connected"])
        self.assertIsNone(s["live"])
        self.assertIsNone(s["state"])
        self.assertIsNone(s["audio"])
        self.assertIsNone(s["push"])
        self.assertFalse(H["reachy_health_summary"]({"has_frame": False})["connected"])


class ShippedConfig(unittest.TestCase):
    """The repo config flags exactly the two UIs, and never the page itself."""

    def setUp(self):
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML not installed")
        self.cfg = yaml.safe_load(FEED_CONFIG.read_text())

    def test_boot_follows_intent_services(self):
        flagged = {s["key"] for s in self.cfg["services"] if s.get("boot_follows_intent")}
        self.assertEqual(flagged, {"vlm", "livevision"})
        for s in self.cfg["services"]:
            if s.get("self"):
                self.assertIsNone(H["boot_command"](s, "stop"))

    def test_live_vision_link_from_config(self):
        self.assertEqual(H["lan_link"](self.cfg["reachy_live_vision_url"], self.cfg["links_host"]),
                         f"https://{self.cfg['links_host']}:8443/?source=reachy")


if __name__ == "__main__":
    unittest.main(verbosity=2)
