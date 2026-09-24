#!/usr/bin/env python3
"""porch-feed — comparison feed for Cosmos3-Edge captions on Jetson Orin Nano.

Captures every Frigate GenAI description along with the engine that produced it and the peak
CPU / unified-memory / GPU utilisation observed during that inference, then serves them as both a
browsable page and an RSS 2.0 feed so different engine builds can be compared on identical traffic.

Control plane (all guarded, all reversible):
    * switch the active Cosmos3-Edge engine (v1 / v2 / v3)
    * on / off / restart for every process in the stack, systemd unit or docker container alike
    * per-camera power mode: "powered" keeps Frigate's continuous RTSP stream open, "saver" disables
      it so battery Ring cameras are not drained (events still arrive via ring-mqtt motion)

Every control action is verified against the service's PORT rather than against the supervisor's
exit code. `systemctl is-active` reporting "active" only means the unit has not exited: the shim is
"active" for several seconds while it deserializes engines and cannot answer a request, and a unit
that has exited can still leave something holding its port. The port is the honest signal, and
after a stop it is also the evidence that the process really let go of its memory - so stops report
the megabytes actually reclaimed, which on an 8 GB board is the entire reason to stop anything.

Memory discipline matters here: the model holds ~3.3 GB of 8 GB. This service samples /proc and
sysfs directly rather than spawning tegrastats per sample, keeps only a bounded in-memory window,
and its unit sets MemoryMax. Before starting a heavy service it refuses if free memory is too low.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import secrets
import sqlite3
import subprocess
import threading
import time
from contextlib import closing
from email.utils import formatdate
from pathlib import Path
from xml.sax.saxutils import escape

import alert_policy
import base64
import requests
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from reachy import (ANTENNA_LIMIT_RAD, ANTENNA_PARK_DEG, LIMITS_M, LIMITS_RAD, MOTOR_MODES,
                    Reachy as ReachyClient)
from scout import Scout as ScoutClient

CFG = yaml.safe_load(Path(os.environ.get("PORCH_FEED_CONFIG",
                                         "/home/orin/nvr/feed/config.yaml")).read_text())
FRIGATE = CFG["frigate_url"].rstrip("/")
DB_PATH = CFG["db_path"]
SNAP_DIR = Path(CFG.get("snapshot_dir", "/home/orin/nvr/feed/snapshots"))
SNAP_KEEP_MB = float(CFG.get("snapshot_keep_mb", 512))
ENGINES: dict = CFG["engines"]
ENGINE_LINK = CFG["engine_symlink"]
DESC_WAIT = float(CFG.get("description_wait_seconds", 40))
DESC_POLL = float(CFG.get("description_poll_seconds", 0.5))
ALWAYS_POWERED = set(CFG.get("always_powered_cameras") or [])
FRIGATE_CONFIG = CFG["frigate_config_path"]
MIN_FREE_MB = int(CFG.get("min_free_mb_to_start_service", 400))
# Shared secret for the mutating endpoints. Read-only views stay open so the feed is easy to read
# on a phone; anything that runs sudo requires this.
CONTROL_TOKEN = str(CFG.get("control_token") or "").strip()
# Records why a service is in the state it is in. Written whenever the control plane starts or
# stops something, so "off" can be distinguished from "off on purpose".
#
# This exists because automation (and assistants) kept restarting services a human had
# deliberately stopped, reading a clean shutdown as a fault. A clean `systemctl stop` and a crash
# look almost identical in the journal after the fact; the difference is intent, and intent has to
# be recorded at the moment of the action or it is gone.
INTENT_PATH = Path(CFG.get("service_intent_path")
                   or str(Path(DB_PATH).parent / "service_intent.json"))
LINKS = CFG.get("links") or []
LINKS_HOST = CFG.get("links_host") or "127.0.0.1"
REACHY_WEBUI = str(CFG.get("reachy_webui_url") or "").rstrip("/")
# Live Vision opened on the Reachy source. Like reachy_webui_url it is written as loopback; only its
# scheme, port, path and query are used, re-pointed at links_host (see lan_link).
REACHY_LIVE_VISION = str(CFG.get("reachy_live_vision_url") or "").strip()
REACHY_CAM = str(CFG.get("reachy_camera_url") or "").rstrip("/")
REACHY_DAEMON = str(CFG.get("reachy_daemon_url") or "").rstrip("/")
_reachy = ReachyClient(REACHY_DAEMON) if REACHY_DAEMON else None
# Scout audio goes over SSH (the robot has no ROS/RTSP audio). Key auth, unprivileged linaro.
SCOUT_SSH_KEY = str(CFG.get("scout_ssh_key") or "/home/orin/.ssh/scout_ed25519")
SCOUT_MIC_DEV = str(CFG.get("scout_mic_device") or "hw:0,1")
SCOUT_SPK_DEV = str(CFG.get("scout_speaker_device") or "hw:0,0")
AUDIO_RATE = 16000

# One robot per entry. The list drives everything downstream - endpoints, cards, Frigate - so a
# third Scout is a config line, not code. Falls back to the single scout_bridge_url (old single-
# robot config) and then to the two known robots, so a live box with neither still works.
DEFAULT_SCOUTS = [
    {"id": "robot_room", "name": "Robot room · 2F",
     "bridge_url": "http://172.17.0.1:8098", "ssh": "linaro@192.168.7.6", "kinematics": "mecanum"},
    {"id": "first_floor", "name": "First floor · tracked",
     "bridge_url": "http://172.17.0.1:8100", "ssh": "linaro@192.168.7.7", "kinematics": "tracked"},
]
_scouts_cfg = CFG.get("scouts")
if not _scouts_cfg:
    if CFG.get("scout_bridge_url"):
        _scouts_cfg = [{"id": "scout", "name": "Moorebot Scout",
                        "bridge_url": CFG["scout_bridge_url"],
                        "ssh": CFG.get("scout_ssh", "linaro@192.168.7.6"), "kinematics": "mecanum"}]
    else:
        _scouts_cfg = DEFAULT_SCOUTS


class _ScoutEntry:
    """One robot: its bridge client, ssh target for audio, and display metadata."""
    def __init__(self, d: dict):
        self.id = str(d["id"])
        self.name = str(d.get("name") or self.id)
        self.kinematics = str(d.get("kinematics") or "mecanum")
        self.bridge = str(d.get("bridge_url") or "").rstrip("/")
        self.ssh = str(d.get("ssh") or "")
        # The docker container running THIS robot's bridge, so a ROS restart can bounce it to
        # reconnect to the fresh roscore. Empty = skip the bridge bounce (bridge stays as-is).
        self.container = str(d.get("container") or "")
        self.client = ScoutClient(self.bridge) if self.bridge else None


SCOUTS = {}
for _d in _scouts_cfg:
    try:
        _e = _ScoutEntry(_d)
        SCOUTS[_e.id] = _e
    except (KeyError, TypeError) as _err:
        print(f"[feed] bad scout config entry {_d}: {_err}", flush=True)


def _scout_entry(sid: str) -> "_ScoutEntry":
    e = SCOUTS.get(sid)
    if e is None:
        raise HTTPException(404, f"unknown scout '{sid}'")
    return e
# Anomaly watch: the microphone triggers the eye.
#
# Continuous captioning of the robot's view would mean a ~650 ms VLM call every few seconds,
# competing with the NVR for a GPU that already holds a 3.6 GB model. Instead the cheap signal
# gates the expensive one: poll direction-of-arrival (a tiny JSON GET, no decode, no GPU) and only
# spend an inference when the robot actually hears something, rate-limited by a cooldown.
REACHY_WATCH = bool(CFG.get("reachy_watch_enabled", True))
REACHY_WATCH_COOLDOWN = float(CFG.get("reachy_watch_cooldown", 60))
# Which alert_policy categories count as an anomaly for the robot's indoor view. See the filter in
# reachy_look_and_describe() for why this is narrower than the exterior policy.
REACHY_ALERT_CATEGORIES = set(CFG.get("reachy_alert_categories") or ["person", "animal"])
# Frigate labels that also trigger a look. Empty camera list means any camera.
REACHY_TRIGGER_LABELS = {str(x).lower() for x in
                         (CFG.get("reachy_trigger_labels") or ["person", "dog", "cat"])}
REACHY_TRIGGER_CAMERAS = {str(x) for x in (CFG.get("reachy_trigger_cameras") or [])}
_reachy_alert: dict = {"at": None, "description": None, "categories": [], "headline": None,
                       "trigger": None, "checked": 0, "last_check": 0.0}
REACHY_SESSION = str(CFG.get("reachy_session") or "reachy")

_lock = threading.Lock()
_cfg_lock = threading.Lock()   # serialises read-modify-write of the Frigate YAML
_ctl_lock = threading.Lock()   # serialises privileged control actions

# --------------------------------------------------------------------------- telemetry
GPU_LOAD_PATHS = [
    "/sys/devices/platform/bus@0/17000000.gpu/load",
    "/sys/class/devfreq/17000000.gpu/device/load",
]


def _cpu_times() -> tuple[int, int]:
    with open("/proc/stat") as f:
        parts = f.readline().split()[1:]
    vals = [int(x) for x in parts]
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
    return sum(vals), idle


def _mem_pct() -> float:
    """Unified-memory utilisation. On Jetson this IS the VRAM figure - CPU and GPU share one pool."""
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, _, v = line.partition(":")
            info[k] = int(v.split()[0])
    total = info.get("MemTotal", 1)
    avail = info.get("MemAvailable", 0)
    return (total - avail) * 100.0 / total


def _gpu_pct() -> float:
    for p in GPU_LOAD_PATHS:
        try:
            raw = int(Path(p).read_text().strip())
            # Some kernels report 0-1000, others 0-100.
            return raw / 10.0 if raw > 100 else float(raw)
        except Exception:
            continue
    return -1.0


class PeakSampler:
    """Samples CPU / unified-memory / GPU and records the peak over a window.

    Runs only while an inference is outstanding, so idle cost is zero.
    """

    def __init__(self, hz: float = 5.0):
        self.interval = 1.0 / hz
        self._stop = threading.Event()
        self.peak_cpu = 0.0
        self.peak_mem = 0.0
        self.peak_gpu = 0.0
        self.cpu_samples = 0      # 0 means the window was too short to measure
        self.inst_samples = 0
        self._t: threading.Thread | None = None

    def _sample_instant(self):
        self.peak_mem = max(self.peak_mem, _mem_pct())
        g = _gpu_pct()
        if g >= 0:
            self.peak_gpu = max(self.peak_gpu, g)
        self.inst_samples += 1

    def _run(self):
        prev_total, prev_idle = _cpu_times()
        while not self._stop.wait(self.interval):
            total, idle = _cpu_times()
            dt, di = total - prev_total, idle - prev_idle
            prev_total, prev_idle = total, idle
            if dt > 0:
                self.peak_cpu = max(self.peak_cpu, (dt - di) * 100.0 / dt)
                self.cpu_samples += 1
            self._sample_instant()

    def result(self) -> dict:
        """None where the window was too short to measure, never a misleading 0.0.

        A backfilled event whose description already exists resolves in ~10ms, which is under one
        sample interval. Recording 0% CPU there would corrupt the per-engine averages this feed
        exists to produce, so unmeasured is stored as NULL and rendered as a dash.
        """
        return {
            "peak_cpu": round(self.peak_cpu, 1) if self.cpu_samples else None,
            "peak_mem": round(self.peak_mem, 1) if self.inst_samples else None,
            "peak_gpu": round(self.peak_gpu, 1) if (self.inst_samples and self.peak_gpu > 0) else None,
        }

    def __enter__(self):
        self._sample_instant()      # guarantee at least one memory/GPU reading
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *_a):
        self._stop.set()
        if self._t:
            self._t.join(timeout=2)


# --------------------------------------------------------------------------- service links
_link_state: list[dict] = []
_link_lock = threading.Lock()


def _probe(link: dict) -> dict:
    """Probe one service. TCP connect for non-HTTP; a real request otherwise.

    Polled on a background thread rather than per request, so opening the page never blocks on a
    dead host's connect timeout.
    """
    import socket
    host, port = LINKS_HOST, int(link["port"])
    up = False
    if link.get("tcp_only"):
        try:
            with socket.create_connection((host, port), timeout=2):
                up = True
        except OSError:
            up = False
        url = f"{host}:{port}"
    else:
        scheme = link.get("scheme", "http")
        url = f"{scheme}://{host}:{port}{link.get('path', '/')}"
        try:
            r = requests.get(url, timeout=3, verify=False,
                             allow_redirects=False)
            up = r.status_code < 500
        except requests.RequestException:
            up = False
        # The probe hits `path`; the link shown to the user is `url_path`, which may carry query
        # parameters that would make the health check session-specific.
        url = f"{scheme}://{host}:{port}{link.get('url_path', '')}"
    return {"name": link["name"], "port": port, "url": url, "up": up,
            "browsable": not link.get("tcp_only", False)}


def reachy_look_and_describe(trigger: str) -> dict:
    """One frame, one VLM call, classified by the existing alert policy.

    Deliberately reuses alert_policy rather than asking the model to judge: the prompt enumerates
    nothing, because this 4B model reports back whatever the prompt lists.
    """
    global _reachy_alert
    if not REACHY_CAM:
        return {"error": "no camera bridge configured"}
    try:
        img = requests.get(f"{REACHY_CAM}/still.jpg", timeout=6)
        if img.status_code != 200 or not img.content:
            return {"error": "no frame available"}
        b64 = base64.b64encode(img.content).decode()
        payload = {
            "model": "cosmos3-edge",
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
                {"type": "text", "text": alert_policy.DESCRIBE_PROMPT},
            ]}],
            "max_tokens": 128,
            "temperature": 0.0,
        }
        r = requests.post(f"{CFG['cosmos3_url'].rstrip('/')}/v1/chat/completions",
                          json=payload, timeout=120)
        r.raise_for_status()
        desc = r.json()["choices"][0]["message"]["content"].strip()
    except (requests.RequestException, KeyError, ValueError) as e:
        return {"error": f"{type(e).__name__}: {e}"}

    verdict = alert_policy.classify(desc)          # a dict, not a list
    cats = verdict.get("categories") or []

    # Narrow the policy for an indoor robot. alert_policy is written for exterior security, where a
    # package on the ground or a vehicle is meaningful; pointed at a desk it fires on cardboard.
    # Indoors the anomaly worth surfacing is a living thing that should not be there, so only
    # those categories count. Everything else is still recorded, just not raised as an alert.
    cats = [c for c in cats if c in REACHY_ALERT_CATEGORIES]

    _reachy_alert = {
        "at": time.time(),
        "description": desc,
        "categories": cats,
        "alert": bool(cats),
        "all_categories": verdict.get("categories") or [],
        "headline": verdict.get("headline") or desc,
        "trigger": trigger,
        "checked": _reachy_alert.get("checked", 0) + 1,
        "last_check": time.time(),
    }
    if cats:
        print(f"[feed] reachy anomaly ({trigger}) {cats}: {desc[:110]}", flush=True)
    return _reachy_alert


def reachy_watcher() -> None:
    """Poll the robot's ear; spend an inference only when it hears something."""
    if not (REACHY_WATCH and _reachy and REACHY_CAM):
        return
    while True:
        try:
            doa = _reachy._get("/api/state/doa") or {}
            if doa.get("speech_detected"):
                since = time.time() - (_reachy_alert.get("last_check") or 0)
                if since >= REACHY_WATCH_COOLDOWN:
                    reachy_look_and_describe("speech")
        except Exception as e:
            print(f"[feed] reachy watcher: {e}", flush=True)
        time.sleep(3)


def link_poller() -> None:
    global _link_state
    import urllib3
    try:
        urllib3.disable_warnings()      # self-signed certs on Frigate/VLM are expected here
    except Exception:
        pass
    while True:
        state = [_probe(l) for l in LINKS]
        with _link_lock:
            _link_state = state
        time.sleep(20)


# --------------------------------------------------------------------------- storage
def db() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    with closing(db()) as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS entries (
                id           TEXT PRIMARY KEY,
                ts           REAL,
                camera       TEXT,
                label        TEXT,
                description  TEXT,
                engine_id    TEXT,
                engine_name  TEXT,
                latency_ms   INTEGER,
                peak_cpu     REAL,
                peak_mem     REAL,
                peak_gpu     REAL
            )""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_ts ON entries(ts DESC)")
        c.commit()


def save(row: dict) -> None:
    with _lock, closing(db()) as c:
        c.execute("""INSERT OR REPLACE INTO entries
            (id,ts,camera,label,description,engine_id,engine_name,latency_ms,peak_cpu,peak_mem,peak_gpu)
            VALUES (:id,:ts,:camera,:label,:description,:engine_id,:engine_name,:latency_ms,
                    :peak_cpu,:peak_mem,:peak_gpu)""", row)
        c.commit()


# --------------------------------------------------------------------------- snapshot cache
def snap_path(eid: str) -> Path:
    # eids come from Frigate and look like "1789352873.693748-uu7zc8", but they land on the
    # filesystem, so anything that could climb out of SNAP_DIR is stripped rather than trusted.
    safe = "".join(ch for ch in eid if ch.isalnum() or ch in "-_.")
    return SNAP_DIR / f"{safe}.jpg"


def cache_snapshot(eid: str) -> Path | None:
    """Save the event's best frame locally. Returns the path, or None if Frigate has no image.

    snapshot.jpg is the frame Frigate scored highest for the event, which is the one worth showing.
    thumbnail.jpg is the fallback: much smaller and always present, so a feed item still gets a
    picture even when snapshots are off for that camera.
    """
    dst = snap_path(eid)
    if dst.exists() and dst.stat().st_size > 1024:
        return dst
    for endpoint in ("snapshot.jpg", "thumbnail.jpg"):
        try:
            r = requests.get(f"{FRIGATE}/api/events/{eid}/{endpoint}", timeout=20)
        except requests.RequestException:
            continue
        if r.status_code == 200 and len(r.content) > 1024:
            SNAP_DIR.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(".part")
            tmp.write_bytes(r.content)
            tmp.replace(dst)          # atomic, so a reader never sees a half-written jpeg
            return dst
    return None


def prune_snapshots() -> None:
    """Keep the cache under snapshot_keep_mb, oldest first. Unbounded growth on a device with one
    SD card is its own outage."""
    try:
        files = sorted(SNAP_DIR.glob("*.jpg"), key=lambda f: f.stat().st_mtime, reverse=True)
    except OSError:
        return
    total = 0
    for f in files:
        try:
            total += f.stat().st_size
            if total > SNAP_KEEP_MB * 2**20:
                f.unlink()
        except OSError:
            pass


# --------------------------------------------------------------------------- engine control
def active_engine() -> dict:
    try:
        target = os.path.realpath(ENGINE_LINK)
    except OSError:
        target = ""
    for eid, e in ENGINES.items():
        if os.path.realpath(e["path"]) == target:
            return {"id": eid, **e}
    return {"id": "unknown", "name": "Unknown", "path": target,
            "profile": "", "notes": "active engine does not match any configured build"}


def run(cmd: list[str], timeout: int = 180) -> tuple[bool, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0, (r.stderr or r.stdout)[-400:]
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def switch_engine(eid: str) -> tuple[bool, str]:
    if not _ctl_lock.acquire(blocking=False):
        return False, "another control action is in progress"
    try:
        return _switch_engine(eid)
    finally:
        _ctl_lock.release()


def _switch_engine(eid: str) -> tuple[bool, str]:
    if eid not in ENGINES:
        return False, f"unknown engine {eid}"
    path = ENGINES[eid]["path"]
    if not os.path.isfile(os.path.join(path, "llm.engine")):
        return False, f"engine not built at {path}"
    ok, msg = run(["sudo", "-n", "ln", "-sfn", path, ENGINE_LINK])
    if not ok:
        return False, f"symlink failed: {msg}"
    ok, msg = run(["sudo", "-n", "systemctl", "restart", "cosmos3-edge-shim.service"])
    if not ok:
        return False, f"restart failed: {msg}"
    # wait for readiness so the UI does not report success before the model can serve
    for _ in range(60):
        try:
            if requests.get(f"{CFG['cosmos3_url']}/v1/models", timeout=2).status_code == 200:
                return True, f"switched to {ENGINES[eid]['name']}"
        except requests.RequestException:
            pass
        time.sleep(2)
    return False, "engine swapped but shim did not become ready in 120s"


# --------------------------------------------------------------------------- service control
SERVICES = {s["key"]: s for s in (CFG.get("services") or [])}
ACTIONS = ("start", "stop", "restart")


# Boot state that follows intent.
#
# The Live VLM WebUI and Live Vision are switched off on purpose much of the time, so any FIXED boot
# state is wrong half the time: enabled resurrects a UI somebody stopped, disabled loses one they
# started. For services marked boot_follows_intent the control plane makes the unit's enablement
# match the last deliberate ON/OFF, so a reboot keeps the choice instead of undoing it.
#
# The three helpers below are pure (no globals, no I/O) so nvr/tests/test_porch_feed_boot.py can
# exercise them without importing this module, which reads the live config at import time.
def boot_command(service: dict, action: str) -> list[str] | None:
    """The `systemctl enable|disable` that carries this action across a reboot, or None.

    None for anything not opted in, and always for this page itself: disabling porch-feed would
    remove the only browser route to turning anything back on after the next boot. Docker services
    are left alone too - their restart policy lives in the container, not in systemd.
    """
    if not service.get("boot_follows_intent") or service.get("self"):
        return None
    if service.get("kind") != "systemd" or not service.get("unit"):
        return None
    if action in ("start", "restart"):
        verb = "enable"
    elif action == "stop":
        verb = "disable"
    else:
        return None
    return ["sudo", "-n", "systemctl", verb, service["unit"]]


def boot_enabled_from(is_enabled_output: str) -> bool | None:
    """Will the unit start at the next boot, read from `systemctl is-enabled`. None if unclear.

    enabled-runtime counts as off: its symlink lives in /run, which a reboot wipes. States such as
    static or indirect do not answer the question either way, so they stay None rather than being
    rendered as a confident "off at boot".
    """
    lines = (is_enabled_output or "").strip().splitlines()
    state = lines[-1].strip() if lines else ""
    return {"enabled": True, "disabled": False, "masked": False, "masked-runtime": False,
            "enabled-runtime": False}.get(state)


def boot_note(action: str, ok: bool, detail: str = "") -> str:
    """Suffix for the action's message. A failure is reported, but the start/stop itself stands:
    the service really did start or stop, and saying otherwise would invite a pointless retry."""
    on = action != "stop"
    if ok:
        return "; starts at boot" if on else "; off at boot"
    return (f"; but could not {'enable' if on else 'disable'} it at boot, so a reboot may undo "
            f"this: {(detail or '').strip()[:200] or 'no detail'}")


def load_intent() -> dict:
    try:
        return json.loads(INTENT_PATH.read_text())
    except (OSError, ValueError):
        return {}


def record_intent(key: str, action: str, actor: str = "user", note: str = "") -> None:
    """Remember that someone deliberately put this service into this state."""
    data = load_intent()
    data[key] = {
        "desired": "stopped" if action == "stop" else "running",
        "action": action,
        "by": actor,
        "at": time.time(),
        "note": note,
    }
    try:
        INTENT_PATH.parent.mkdir(parents=True, exist_ok=True)
        INTENT_PATH.write_text(json.dumps(data, indent=2))
    except OSError as e:
        print(f"[feed] could not record intent for {key}: {e}", flush=True)


def free_mb() -> int:
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, _, v = line.partition(":")
            info[k] = int(v.split()[0])
    return info.get("MemAvailable", 0) // 1024


def port_open(port: int | None, timeout: float = 1.0) -> bool | None:
    """Is anything accepting connections on this port right now?

    None when the service has no port to probe, which is a different answer from False and must
    not be rendered as "down".
    """
    if not port:
        return None
    import socket
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def unit_state(key: str) -> str:
    """What systemd or docker claims. Not proof that the service is serving - see port_open()."""
    s = SERVICES[key]
    if s["kind"] == "systemd":
        ok, out = run(["systemctl", "is-active", s["unit"]], timeout=15)
        return out.strip() or ("active" if ok else "inactive")
    ok, out = run(["sudo", "-n", "docker", "inspect", "-f", "{{.State.Status}}", s["name"]], 20)
    return out.strip() if ok else "absent"


def service_status(key: str) -> dict:
    """Combined view: what the supervisor claims, and what the port actually shows.

    The two disagree in exactly the cases worth surfacing: `starting` while the shim spends seconds
    deserializing engines before it can answer, and `stuck` when a unit has exited but something is
    still holding its port - which on an 8 GB board usually means the memory has not come back yet.
    """
    s = SERVICES[key]
    state = unit_state(key)
    running = state in ("active", "running")
    listening = port_open(s.get("port"))

    if listening is None:
        health = "up" if running else "down"          # no port to check; supervisor is all we have
    elif running and listening:
        health = "up"
    elif running and not listening:
        health = "starting"
    elif not running and listening:
        health = "stuck"
    else:
        health = "down"

    # Only asked for services whose boot state this page manages: /api/status is polled every 10 s,
    # and is-enabled is cheap but not free. None means "not managed here" or "unclear".
    follows = bool(s.get("boot_follows_intent")) and s["kind"] == "systemd"
    boot_state = boot_enabled = None
    if follows:
        _ok, out = run(["systemctl", "is-enabled", s["unit"]], timeout=10)
        boot_enabled = boot_enabled_from(out)
        boot_state = (out.strip().splitlines() or ["unknown"])[-1][:60]

    intent = load_intent().get(key) or {}
    return {
        "key": key,
        "label": s.get("label", key),
        # `intent` answers "was this meant to be off?" - check it before starting anything.
        "intent": intent.get("desired"),
        "intent_by": intent.get("by"),
        "intent_at": intent.get("at"),
        "intent_note": intent.get("note"),
        # True when something is running that a human deliberately stopped, or vice versa.
        "contradicts_intent": bool(
            intent.get("desired")
            and ((intent["desired"] == "stopped") == (state in ("active", "running")))
        ),
        "kind": s["kind"],
        "unit": s.get("unit") or s.get("name"),
        "port": s.get("port"),
        "state": state,
        "running": running,
        "listening": listening,
        "health": health,
        "self": bool(s.get("self")),
        "note": s.get("note", ""),
        "heavy_mb": s.get("heavy_mb"),
        "boot_follows_intent": follows,
        "boot_enabled": boot_enabled,
        "boot_state": boot_state,
    }


def _await_port(port: int | None, want_open: bool, timeout: float = 25.0) -> bool:
    """Wait until the port reaches the expected state. True if it got there."""
    if not port:
        time.sleep(1.0)     # nothing to observe; give the supervisor a moment to settle
        return True
    deadline = time.time() + timeout
    while time.time() < deadline:
        if port_open(port) is want_open:
            return True
        time.sleep(0.5)
    return False


def _deferred_self_action(unit: str, action: str) -> tuple[bool, str]:
    """Act on porch-feed itself, after this request has already been answered.

    Running `systemctl stop porch-feed` in-process kills uvicorn mid-request, so the browser sees a
    connection reset instead of a result. Scheduling it as a transient unit lets the response land
    first.
    """
    ok, msg = run([
        "sudo", "-n", "systemd-run", "--on-active=2",
        f"--unit=porch-feed-{action}-{int(time.time())}",
        "systemctl", action, unit,
    ])
    if not ok:
        return False, msg
    verb = "stopping" if action == "stop" else f"{action}ing"
    return True, (f"{verb} porch-feed in 2s — this page will go offline"
                  + ("; start it again over SSH." if action == "stop" else "; reload in ~10s."))


def _follow_intent_at_boot(s: dict, action: str) -> str:
    """Make the unit's boot state match a verified action. Returns a note for the message.

    Runs only after verification, so a start that never came up does not also get enabled at boot.
    """
    cmd = boot_command(s, action)
    if not cmd:
        return ""
    ok, msg = run(cmd, timeout=30)
    if not ok:
        print(f"[feed] {' '.join(cmd[2:])} failed: {msg.strip()[:200]}", flush=True)
    return boot_note(action, ok, msg)


def set_service(key: str, action: str) -> tuple[bool, str]:
    """Run a control action and then *verify* it, rather than trusting the exit code.

    Verification is the whole point: it confirms the port reached the expected state, and reports
    how much memory actually came back. On an 8 GB board a "successful" stop that frees nothing is
    a failure worth seeing. Once verified, a boot_follows_intent service also has its boot state
    set to match (see boot_command), so the next reboot keeps what was chosen here.
    """
    if key not in SERVICES or action not in ACTIONS:
        return False, "bad request"
    s = SERVICES[key]

    # Guard: the model needs ~3.3GB of 8GB. Starting another heavy service on a nearly-full box is
    # how you OOM-kill the model, so refuse rather than let the kernel decide what dies.
    if action in ("start", "restart"):
        fm = free_mb()
        if fm < MIN_FREE_MB:
            return False, (f"refused: only {fm} MB free, need {MIN_FREE_MB} MB. "
                           f"Stop another service first.")

    # Record intent before acting, so a stop that half-succeeds is still known to be deliberate.
    record_intent(key, action, actor="user")

    if s.get("self"):
        return _deferred_self_action(s["unit"], action)

    before = free_mb()
    if s["kind"] == "systemd":
        ok, msg = run(["sudo", "-n", "systemctl", action, s["unit"]])
    else:
        ok, msg = run(["sudo", "-n", "docker", action, s["name"]])
    if not ok:
        return False, msg or f"{action} failed"

    want_open = action != "stop"
    reached = _await_port(s.get("port"), want_open)
    after = free_mb()
    delta = after - before

    label = s.get("label", key)
    port = s.get("port")

    if action == "stop":
        if not reached:
            # The stop itself succeeded and the intent is recorded as stopped, so the boot state
            # follows it even though something still holds the port. Otherwise the page could not
            # fix it: OFF is disabled for a service that is no longer running.
            return False, (f"{label}: {s['kind']} reported stopped, but port {port} is still "
                           f"accepting connections — something is still holding it. "
                           f"Memory is at {after} MB free." + _follow_intent_at_boot(s, action))
        freed = f"reclaimed {delta} MB" if delta > 0 else "no memory reclaimed"
        expect = s.get("heavy_mb")
        warn = ""
        if expect and delta < expect * 0.5:
            warn = (f" — expected about {expect} MB back; the kernel may still be releasing "
                    f"page cache, re-check the memory readout in a few seconds")
        return True, (f"{label} stopped, port {port or '—'} closed, {freed} ({after} MB free){warn}"
                      + _follow_intent_at_boot(s, action))

    if not reached:
        return False, (f"{label}: {action} issued but port {port} never started listening. "
                       f"Check `journalctl -u {s.get('unit') or s.get('name')}`.")
    cost = f"used {-delta} MB" if delta < 0 else f"memory unchanged ({delta:+d} MB)"
    return True, (f"{label} {action}ed, port {port or '—'} listening, {cost} ({after} MB free)"
                  + _follow_intent_at_boot(s, action))


# --------------------------------------------------------------------------- camera power
def read_frigate_cfg() -> dict:
    return yaml.safe_load(Path(FRIGATE_CONFIG).read_text())


def camera_power() -> dict:
    """powered = Frigate holds a continuous RTSP stream; saver = it does not.

    Continuous detect is what drains a battery Ring camera, so 'saver' simply disables the camera in
    Frigate. Motion events still arrive through ring-mqtt, which costs the camera nothing extra.
    """
    cfg = read_frigate_cfg()
    out = {}
    for name, cam in (cfg.get("cameras") or {}).items():
        out[name] = {
            "mode": "powered" if cam.get("enabled") else "saver",
            "always_powered": name in ALWAYS_POWERED,
        }
    return out


def set_camera_power(name: str, mode: str) -> tuple[bool, str]:
    if mode not in ("powered", "saver"):
        return False, "mode must be powered|saver"
    if mode == "saver" and name in ALWAYS_POWERED:
        return False, f"{name} is configured always-powered"
    cfg = read_frigate_cfg()
    if name not in (cfg.get("cameras") or {}):
        return False, f"unknown camera {name}"
    want = (mode == "powered")
    if bool(cfg["cameras"][name].get("enabled")) == want:
        return True, f"{name} already {mode}"

    # Rewrite only the one `enabled:` line inside that camera's block, so comments and formatting
    # in the hand-maintained config survive a round-trip.
    with _cfg_lock:
        text = Path(FRIGATE_CONFIG).read_text()
        lines = text.split("\n")

        # Anchor inside the cameras: block only. `  genai:` also exists under `objects:`, so a bare
        # search for "  {name}:" can match the wrong section.
        try:
            cam_sec = next(i for i, l in enumerate(lines) if l.rstrip() == "cameras:")
        except StopIteration:
            return False, "cameras: section not found"
        start = None
        for i in range(cam_sec + 1, len(lines)):
            l = lines[i]
            if l and not l.startswith(" "):
                break                      # left the cameras: block entirely
            if l.startswith(f"  {name}:"):
                start = i
                break
        if start is None:
            return False, f"camera block for {name} not found under cameras:"

        # Only the camera-level key counts. Matching any `enabled:` would hit the nested ones under
        # detect:/record:/snapshots: and silently disable DETECTION while claiming to save battery.
        CAM_LEVEL = "    "
        for i in range(start + 1, len(lines)):
            l = lines[i]
            if l.strip() and not l.startswith(CAM_LEVEL):
                break                      # next camera, or a top-level key
            if l.startswith(CAM_LEVEL) and l[len(CAM_LEVEL):].lstrip() != l[len(CAM_LEVEL):]:
                continue                   # deeper than camera level
            if l.strip().startswith("enabled:"):
                lines[i] = CAM_LEVEL + f"enabled: {'true' if want else 'false'}"
                Path(FRIGATE_CONFIG).write_text("\n".join(lines))
                ok, msg = run(["sudo", "-n", "docker", "restart", "frigate"], 240)
                return ok, (f"{name} -> {mode}" if ok
                            else f"config updated but restart failed: {msg}")
        return False, f"no camera-level enabled: found for {name}"


# --------------------------------------------------------------------------- ingest
def event_detail(eid: str) -> dict:
    try:
        r = requests.get(f"{FRIGATE}/api/events/{eid}", timeout=15)
        if r.status_code == 200:
            return r.json()
    except requests.RequestException:
        pass
    return {}


def handle_event(after: dict) -> None:
    eid = after.get("id")
    if not eid:
        return
    with _lock, closing(db()) as c:
        if c.execute("SELECT 1 FROM entries WHERE id=?", (eid,)).fetchone():
            return
        # Claim the id inside the same lock so a second delivery cannot also pass the check.
        c.execute("INSERT OR IGNORE INTO entries (id, ts, camera, label, engine_id) "
                  "VALUES (?,?,?,?,?)",
                  (eid, after.get("start_time") or time.time(), after.get("camera", "?"),
                   after.get("label", "object"), "pending"))
        c.commit()

    eng_before = active_engine()
    t0 = time.time()
    desc = None
    with PeakSampler() as sampler:
        deadline = t0 + DESC_WAIT
        while time.time() < deadline:
            ev = event_detail(eid) or {}
            # Frigate 0.18 stores the generated text at data.description; the top-level field is null.
            d = (ev.get("data") or {}).get("description") or ev.get("description")
            if d:
                desc = d.strip()
                break
            # Poll interval bounds the resolution of latency_ms. At 2 s every measurement landed
            # on a 2 s boundary, which is far coarser than the difference between engines this
            # feed exists to detect.
            time.sleep(DESC_POLL)
    if not desc:
        return  # nothing to compare; the notifier already surfaces un-described events

    # Re-read the engine after the wait. If it changed mid-flight the attribution is ambiguous, and
    # a wrongly-attributed row is worse than a missing one when the whole point is comparison.
    eng = active_engine()
    if eng["id"] != eng_before["id"]:
        print(f"[feed] {eid} dropped: engine changed {eng_before['id']}->{eng['id']} mid-inference",
              flush=True)
        return

    # Grab the still now, while Frigate certainly still holds the event.
    if cache_snapshot(eid):
        threading.Thread(target=prune_snapshots, daemon=True).start()

    save({
        "id": eid, "ts": after.get("start_time") or t0,
        "camera": after.get("camera", "?"), "label": after.get("label", "object"),
        "description": desc, "engine_id": eng["id"], "engine_name": eng["name"],
        "latency_ms": int((time.time() - t0) * 1000),
        **sampler.result(),
    })
    r = sampler.result()
    fmt = lambda v: "n/a" if v is None else f"{v:.0f}%"
    print(f"[feed] {eid} [{eng['id']}] cpu={fmt(r['peak_cpu'])} mem={fmt(r['peak_mem'])} "
          f"gpu={fmt(r['peak_gpu'])} :: {desc[:70]}", flush=True)


def mqtt_loop() -> None:
    import paho.mqtt.client as mqtt

    def on_connect(c, *_a):
        c.subscribe("frigate/events")
        print("[feed] subscribed to frigate/events", flush=True)

    def on_message(_c, _u, msg):
        try:
            data = json.loads(msg.payload.decode())
        except Exception:
            return
        after = data.get("after") or {}

        # Detection trigger for the robot's anomaly check. Frigate fires "new" as soon as it has a
        # tracked object, which is the moment worth looking - waiting for "end" would describe a
        # scene after whatever caused it has gone. The same cooldown as the speech trigger applies,
        # so a busy camera cannot turn this into continuous inference.
        if data.get("type") in ("new", "update"):
            label = (after.get("label") or "").lower()
            cam = after.get("camera") or ""
            if label in REACHY_TRIGGER_LABELS and (
                not REACHY_TRIGGER_CAMERAS or cam in REACHY_TRIGGER_CAMERAS
            ):
                since = time.time() - (_reachy_alert.get("last_check") or 0)
                if since >= REACHY_WATCH_COOLDOWN:
                    threading.Thread(
                        target=reachy_look_and_describe,
                        args=(f"{label}@{cam}",),
                        daemon=True,
                    ).start()

        if data.get("type") != "end":
            return
        threading.Thread(target=handle_event, args=(after,), daemon=True).start()

    c = mqtt.Client()
    c.on_connect, c.on_message = on_connect, on_message
    while True:
        try:
            c.connect(CFG.get("mqtt_host", "127.0.0.1"), int(CFG.get("mqtt_port", 1883)), 60)
            c.loop_forever()
        except Exception as e:
            print(f"[feed] mqtt reconnect after {e}", flush=True)
            time.sleep(5)


# --------------------------------------------------------------------------- web
app = FastAPI(title="porch-feed")


def require_control(request: Request) -> None:
    """Guard every privileged endpoint.

    These routes run `sudo -n` (symlink swap, systemctl, docker) and the service binds 0.0.0.0, so
    without this any LAN host - or any web page a browser on the LAN happens to load - could stop
    the NVR, disable a camera, or unload the model. A header-based secret is used rather than a
    cookie precisely because cookies are what make drive-by CSRF work: a cross-origin form POST
    cannot set a custom header.
    """
    if not CONTROL_TOKEN:
        raise HTTPException(503, "control_token is not configured; control plane disabled")
    supplied = request.headers.get("X-Porch-Token", "")
    if not secrets.compare_digest(supplied, CONTROL_TOKEN):
        raise HTTPException(401, "missing or invalid X-Porch-Token")


@app.get("/api/entries")
def api_entries(limit: int = 100, engine: str | None = None):
    # Reservation rows (engine_id='pending', no description) are bookkeeping, not captions.
    q = "SELECT * FROM entries WHERE engine_id <> 'pending' AND description IS NOT NULL"
    args: list = []
    if engine and engine != "all":
        q += " AND engine_id=?"
        args.append(engine)
    q += " ORDER BY ts DESC LIMIT ?"
    args.append(limit)
    with closing(db()) as c:
        return JSONResponse([dict(r) for r in c.execute(q, args).fetchall()])


@app.get("/api/status")
def api_status():
    return {
        "active_engine": active_engine(),
        "engines": {k: {"name": v["name"], "profile": v.get("profile", ""),
                        "notes": v.get("notes", ""),
                        "built": os.path.isfile(os.path.join(v["path"], "llm.engine"))}
                    for k, v in ENGINES.items()},
        "services": {k: service_status(k) for k in SERVICES},
        "cameras": camera_power(),
        "memory": {"free_mb": free_mb(), "used_pct": round(_mem_pct(), 1),
                   "min_free_to_start_mb": MIN_FREE_MB},
        # Same-origin UI needs it to call the guarded routes. This stops drive-by CSRF (a
        # cross-origin page cannot read this response), not a determined LAN attacker.
        "control_token": CONTROL_TOKEN,
    }


@app.post("/api/engine/{eid}")
def api_engine(eid: str, request: Request):
    require_control(request)
    ok, msg = switch_engine(eid)
    if not ok:
        raise HTTPException(400, msg)
    return {"ok": True, "message": msg}


@app.post("/api/service/{key}/{action}")
def api_service(key: str, action: str, request: Request):
    require_control(request)
    ok, msg = set_service(key, action)
    if not ok:
        raise HTTPException(400, msg or "failed")
    return {"ok": True, "message": msg, "service": service_status(key),
            "memory": {"free_mb": free_mb()}}


@app.get("/api/service/{key}")
def api_service_status(key: str):
    """Poll one service without running anything privileged."""
    if key not in SERVICES:
        raise HTTPException(404, "no such service")
    return {"service": service_status(key), "memory": {"free_mb": free_mb()}}


def publish_camera_enabled(name: str, on: bool) -> None:
    """Set Frigate's RUNTIME enable state over MQTT.

    Editing `enabled:` in config.yml is not sufficient on Frigate 0.18. It persists a runtime
    enable/disable state separately and RESTORES IT OVER THE CONFIG at startup:

        frigate.comms.dispatcher INFO : Restored runtime state: pinky.enabled=OFF

    Measured consequence: after a reboot this page reported two cameras as "powered" (it reads the
    config) while Frigate had them switched off and served no stream. The config edit alone is a
    toggle that silently does nothing once the runtime state has ever been set.
    """
    try:
        import paho.mqtt.publish as publish
        publish.single(f"frigate/{name}/enabled/set", "ON" if on else "OFF",
                       hostname=CFG.get("mqtt_host", "127.0.0.1"),
                       port=int(CFG.get("mqtt_port", 1883)))
    except Exception as e:
        print(f"[feed] mqtt enable publish failed for {name}: {type(e).__name__}: {e}", flush=True)


@app.post("/api/camera/{name}/{mode}")
def api_camera(name: str, mode: str, request: Request):
    require_control(request)
    ok, msg = set_camera_power(name, mode)
    if not ok:
        raise HTTPException(400, msg)
    # Config first (so it survives a Frigate config reload), then runtime state (so it takes
    # effect now and survives a restart). Both are required; neither alone is enough.
    publish_camera_enabled(name, mode == "powered")
    return {"ok": True, "message": msg}


@app.get("/api/compare")
def api_compare():
    """Per-engine aggregates - the point of the whole exercise.

    Two filters, both load-bearing:

    `engine_id <> 'pending'` drops the reservation rows written by handle_event() to claim an
    event id. A row stays `pending` forever when Frigate never publishes a description, and those
    rows carry no engine, no description and no metrics - averaged in, they appeared as a nameless
    fourth engine with a 0 ms latency, i.e. the fastest thing on the page.

    `description IS NOT NULL` is belt and braces for the same class of half-written row.

    The engine NAME is resolved from config rather than read back from the row. Names are a
    labelling decision that changes (v1 was shipped as "Balanced", it is "Medium" now), and a
    stored name means the same engine shows up under two different labels depending on when its
    captions happened to be recorded.
    """
    with closing(db()) as c:
        rows = c.execute("""
            SELECT engine_id, engine_name, COUNT(*) n,
                   AVG(latency_ms) lat, AVG(peak_cpu) cpu, AVG(peak_mem) mem, AVG(peak_gpu) gpu,
                   SUM(peak_cpu IS NOT NULL) measured,
                   AVG(LENGTH(description)) desc_len
            FROM entries
            WHERE engine_id <> 'pending' AND description IS NOT NULL
            GROUP BY engine_id ORDER BY engine_id""").fetchall()

    out = []
    for r in rows:
        d = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in dict(r).items()}
        cfg = ENGINES.get(d["engine_id"])
        if cfg:
            d["engine_name"] = cfg["name"]
        d["engine_name"] = d.get("engine_name") or d["engine_id"]
        out.append(d)
    return JSONResponse(out)


@app.get("/rss")
def rss(limit: int = 50, engine: str | None = None):
    q = "SELECT * FROM entries"
    args: list = []
    if engine and engine != "all":
        q += " WHERE engine_id=?"
        args.append(engine)
    q += " ORDER BY ts DESC LIMIT ?"
    args.append(limit)
    with closing(db()) as c:
        rows = [dict(r) for r in c.execute(q, args).fetchall()]
    base = CFG.get("public_url", "http://localhost:8096").rstrip("/")
    items = []
    for r in rows:
        title = f"[{r['engine_name']}] {r['camera'].replace('_',' ').title()} — {r['label']}"
        img = f"{base}/img/{r['id']}.jpg"
        have_img = snap_path(r["id"]).exists()
        # The still leads: in a feed reader the picture is what identifies the event at a glance,
        # and the caption is what explains it.
        body = ((f'<p><img src="{img}" alt="{escape(r["camera"])}" '
                 f'style="max-width:100%;height:auto"/></p>' if have_img else "")
                + f"<p>{r['description']}</p>"
                + f"<p><b>Engine:</b> {r['engine_name']} ({r['engine_id']})<br/>"
                  f"<b>Event latency (Frigate pipeline, not inference):</b> "
                  f"{r['latency_ms']} ms<br/>"
                  f"<b>Peak CPU:</b> {r['peak_cpu']}%<br/>"
                  f"<b>Peak unified memory (VRAM):</b> {r['peak_mem']}%<br/>"
                  f"<b>Peak GPU:</b> {r['peak_gpu']}%</p>")
        # CDATA, not escape(): this body is HTML meant to render. Escaping it made readers display
        # the literal tags instead of the formatting, and would have done the same to the <img>.
        body = body.replace("]]>", "]]&gt;")
        enclosure = ""
        if have_img:
            size = snap_path(r["id"]).stat().st_size
            enclosure = (f"<enclosure url='{escape(img)}' length='{size}' type='image/jpeg'/>"
                         # media:content as well - readers split about evenly on which they honour
                         f"<media:content url='{escape(img)}' medium='image' type='image/jpeg'/>"
                         f"<media:thumbnail url='{escape(img)}'/>")
        items.append(
            "<item>"
            f"<title>{escape(title)}</title>"
            f"<description><![CDATA[{body}]]></description>"
            f"{enclosure}"
            f"<pubDate>{formatdate(r['ts'], usegmt=True)}</pubDate>"
            f"<guid isPermaLink='false'>{escape(r['id'])}</guid>"
            f"<category>{escape(r['engine_id'])}</category>"
            f"<link>{escape(base)}/#{escape(r['id'])}</link>"
            "</item>")
    xml = ("<?xml version='1.0' encoding='UTF-8'?>"
           "<rss version='2.0' xmlns:media='http://search.yahoo.com/mrss/'><channel>"
           "<title>porch-dad — Cosmos3-Edge caption feed</title>"
           f"<link>{escape(base)}</link>"
           "<description>Locally generated scene descriptions with engine and resource telemetry"
           "</description>"
           f"<lastBuildDate>{formatdate(time.time(), usegmt=True)}</lastBuildDate>"
           + "".join(items) + "</channel></rss>")
    return Response(xml, media_type="application/rss+xml")


@app.get("/img/{eid}.jpg")
def api_image(eid: str):
    """Serve the event's best still from the local cache, backfilling from Frigate on a miss.

    The feed proxies the image rather than pointing readers at Frigate directly: Frigate's own port
    is authenticated and prunes on its own schedule, so a direct link would break for anyone reading
    the feed later or from outside.
    """
    path = snap_path(eid)
    if not (path.exists() and path.stat().st_size > 1024):
        if not cache_snapshot(eid):
            raise HTTPException(404, "no snapshot for this event")
        path = snap_path(eid)
    return Response(path.read_bytes(), media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/links")
def api_links():
    with _link_lock:
        return JSONResponse(_link_state)


@app.get("/reachy/latest.jpg")
def reachy_latest():
    """Proxy the newest frame pushed from the Reachy Mini.

    Server-side rather than an <img> pointed straight at the WebUI: that origin is HTTPS with a
    self-signed certificate, and a browser on this plain-HTTP page will not render an image from an
    untrusted origin. Fetching here and re-serving same-origin avoids asking the user to accept a
    certificate just to see a thumbnail.
    """
    if not REACHY_CAM:
        raise HTTPException(404, "reachy_camera_url is not configured")
    try:
        r = requests.get(f"{REACHY_CAM}/still.jpg", timeout=5)
    except requests.RequestException as e:
        raise HTTPException(502, f"camera bridge unreachable: {e}")
    if r.status_code != 200 or not r.content:
        raise HTTPException(404, "no frame available yet")
    return Response(r.content, media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


# lan_link and reachy_health_summary are pure for the same reason as boot_command: the tests
# compile them out of this file rather than import it. Hence the local urllib import.
def lan_link(url: str, host: str) -> str | None:
    """The same URL with its host swapped for one a browser elsewhere on the LAN can reach.

    Config writes these as loopback because that is what this service itself talks to; handed to a
    browser on another machine, 127.0.0.1 points it at its own localhost. Scheme, port, path and
    query are kept, so a new link is a config line rather than another hand-built f-string.
    """
    from urllib.parse import urlsplit, urlunsplit
    if not url or not host:
        return None
    u = urlsplit(url)
    try:
        port = u.port
    except ValueError:          # a typo such as ":84o3" - no link beats a wrong one
        return None
    if u.scheme not in ("http", "https") or not u.hostname:
        return None             # "127.0.0.1:8443/x" parses with no host at all; same rule
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"      # an IPv6 links_host needs brackets before a port can follow
    return urlunsplit((u.scheme, f"{host}:{port}" if port else host,
                       u.path or "/", u.query, u.fragment))


def reachy_health_summary(d: dict) -> dict:
    """The bridge's /healthz, reduced to what the Reachy card shows.

    `connected` follows the bridge's own `live` (a session that is delivering fresh frames). A
    bridge from before that field only reports has_frame, which is the best answer it has.
    `reason` is the part worth reading when it is not live: "camera held by the robot app" and
    "robot daemon unreachable" call for different responses, and neither is fixed by a restart.
    """
    audio = d.get("audio") if isinstance(d.get("audio"), dict) else None
    push = d.get("push") if isinstance(d.get("push"), dict) else None
    return {
        "connected": bool(d["live"]) if "live" in d else bool(d.get("has_frame")),
        "live": d.get("live"),
        "frames": d.get("frames", 0),
        "state": d.get("state"),
        "reason": d.get("reason"),
        "failed_streak": d.get("failed_streak"),
        "audio": ({"live": bool(audio.get("live")), "listeners": audio.get("listeners", 0)}
                  if audio is not None else None),
        "push": {"state": push.get("state")} if push is not None else None,
    }


@app.get("/api/reachy")
def api_reachy():
    """Whether the Reachy Mini camera bridge is live, and why not when it is not."""
    webui = f"{REACHY_WEBUI}/?session={REACHY_SESSION}" if REACHY_WEBUI else ""
    out = {"configured": bool(REACHY_CAM), "session": REACHY_SESSION,
           "connected": False, "live": None, "fps": None, "frames": 0, "resolution": None,
           "state": None, "reason": None, "failed_streak": None, "audio": None, "push": None,
           "source": "reachy-mjpeg-bridge",
           "link": lan_link(webui, LINKS_HOST),
           "live_vision_link": lan_link(REACHY_LIVE_VISION, LINKS_HOST)}
    if not REACHY_CAM:
        return out
    try:
        d = requests.get(f"{REACHY_CAM}/healthz", timeout=4).json()
        if not isinstance(d, dict):
            raise ValueError("healthz did not answer a JSON object")
        out.update(reachy_health_summary(d))
    except (requests.RequestException, ValueError) as e:
        # Distinct from any state the bridge reports about itself: here there is no bridge to ask.
        out.update(state="unreachable",
                   reason=(f"reachy-mjpeg-bridge not answering at {REACHY_CAM} "
                           f"({type(e).__name__}) - is reachy-mjpeg-bridge.service running?"))
    return out


@app.get("/api/reachy/state")
def api_reachy_state():
    """Full robot state for the control panel. Degrades rather than erroring when it is off."""
    if not _reachy:
        return {"enabled": False}
    st = _reachy.state()
    st["enabled"] = True
    return JSONResponse(st)


def _reachy_result(ok: bool, msg: str):
    """Turn the client's (ok, message) into a response, surfacing refusals as 400 not 500."""
    if not ok:
        raise HTTPException(400, msg)
    return {"ok": True, "message": msg}


@app.post("/api/reachy/action/{name}")
def api_reachy_action(name: str, request: Request):
    """One-shot buttons. Guarded like every other control that touches hardware."""
    require_control(request)
    if not _reachy:
        raise HTTPException(503, "reachy_daemon_url is not configured")
    actions = {
        "wake": _reachy.wake,
        "sleep": _reachy.sleep,
        "center": _reachy.center,
        "look-at-voice": _reachy.look_at_voice,
    }
    fn = actions.get(name)
    if not fn:
        raise HTTPException(404, f"unknown action {name}")
    ok, msg = fn()
    return _reachy_result(ok, msg)


@app.post("/api/reachy/motors/{mode}")
def api_reachy_motors(mode: str, request: Request):
    require_control(request)
    if not _reachy:
        raise HTTPException(503, "reachy_daemon_url is not configured")
    return _reachy_result(*_reachy.set_motor_mode(mode))


@app.post("/api/reachy/volume/{which}/{value}")
def api_reachy_volume(which: str, value: int, request: Request):
    require_control(request)
    if not _reachy:
        raise HTTPException(503, "reachy_daemon_url is not configured")
    if which not in ("speaker", "mic"):
        raise HTTPException(400, "which must be speaker or mic")
    return _reachy_result(*_reachy.set_volume(which, value))


@app.post("/api/reachy/look/{axis}/{deg}")
def api_reachy_look_axis(axis: str, deg: float, request: Request):
    """Nudge one axis. Positive pitch tilts DOWN - the robot's own convention, kept honest here."""
    require_control(request)
    if not _reachy:
        raise HTTPException(503, "reachy_daemon_url is not configured")
    if axis not in ("pitch", "yaw", "roll", "body_yaw"):
        raise HTTPException(400, "axis must be pitch, yaw, roll or body_yaw")
    return _reachy_result(*_reachy.look(**{axis: deg}))


@app.post("/api/reachy/target")
async def api_reachy_target(request: Request):
    """Live pose control, mirroring the sliders and pads in the robot's own desktop app.

    Units are the daemon's: metres and radians. The panel shows radians because the app does, and
    converting for display only to convert back before sending would be two chances to get a sign
    wrong on a robot whose pitch is already positive-downward.
    """
    require_control(request)
    if not _reachy:
        raise HTTPException(503, "reachy_daemon_url is not configured")
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(400, "expected a JSON body")
    if not isinstance(body, dict):
        raise HTTPException(400, "expected a JSON object")

    pose = {k: body[k] for k in ("x", "y", "z", "roll", "pitch", "yaw") if body.get(k) is not None}
    antennas = body.get("antennas")
    if antennas is not None and (not isinstance(antennas, (list, tuple)) or len(antennas) != 2):
        raise HTTPException(400, "antennas must be [left, right]")
    try:
        ok, msg = _reachy.set_target(pose=pose or None, body_yaw=body.get("body_yaw"),
                                     antennas=antennas)
    except (TypeError, ValueError) as e:
        raise HTTPException(400, f"bad target: {e}")
    return _reachy_result(ok, msg)


@app.get("/api/scouts")
def api_scouts():
    """The robots to render, in config order. Read-only so the page can build its cards."""
    return {"scouts": [{"id": e.id, "name": e.name, "kinematics": e.kinematics}
                       for e in SCOUTS.values()]}


@app.get("/api/scout/{sid}/state")
def api_scout_state(sid: str):
    """Bridge health. Read-only, so no token: the panel needs it to render at all."""
    e = _scout_entry(sid)
    if not e.client:
        return {"enabled": False}
    return e.client.state()


@app.get("/scout/{sid}/latest.jpg")
def scout_latest(sid: str):
    """Proxy the bridge's still so the browser only ever talks to this origin."""
    e = _scout_entry(sid)
    if not e.bridge:
        raise HTTPException(404, "bridge not configured")
    try:
        r = requests.get(f"{e.bridge}/still.jpg", timeout=6)
        if r.status_code != 200 or not r.content:
            raise HTTPException(503, "no frame")
        return Response(content=r.content, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})
    except requests.RequestException as ex:
        raise HTTPException(503, f"bridge unreachable: {ex}")


@app.get("/scout/{sid}/mjpeg")
def scout_mjpeg(sid: str):
    """Live MJPEG, proxied so the browser talks only to this origin.

    The bridge binds the docker gateway (172.17.0.1), which the browser cannot reach, so the
    command centre relays the multipart stream. An <img> pointed here updates at the stream's rate
    with essentially no added latency - unlike the old 3-second still poll that made the robot
    impossible to drive.
    """
    e = _scout_entry(sid)
    if not e.bridge:
        raise HTTPException(404, "bridge not configured")
    try:
        up = requests.get(f"{e.bridge}/mjpeg", stream=True, timeout=(5, 30))
    except requests.RequestException as ex:
        raise HTTPException(503, f"bridge unreachable: {ex}")
    ct = up.headers.get("Content-Type", "multipart/x-mixed-replace; boundary=frame")

    def relay():
        try:
            for chunk in up.iter_content(chunk_size=16384):
                if chunk:
                    yield chunk
        except requests.RequestException:
            pass
        finally:
            up.close()

    return StreamingResponse(relay(), media_type=ct, headers={"Cache-Control": "no-store"})


def _scout_ssh(ssh_target: str, remote_cmd: str) -> list[str]:
    return ["ssh", "-i", SCOUT_SSH_KEY, "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=6", ssh_target, remote_cmd]


async def _audio_ws_capture(ws: WebSocket, ssh_target: str):
    """Mic -> browser: arecord over SSH, one channel of the stereo PDM mic forwarded as PCM16."""
    await ws.accept()
    import numpy as np
    cmd = _scout_ssh(ssh_target, f"exec arecord -q -D {SCOUT_MIC_DEV} -f S16_LE -c 2 -r {AUDIO_RATE} -t raw")
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    try:
        while True:
            raw = await proc.stdout.readexactly(1280)   # 20 ms stereo S16
            await ws.send_bytes(np.frombuffer(raw, dtype="<i2").reshape(-1, 2)[:, 0].tobytes())
    except (asyncio.IncompleteReadError, WebSocketDisconnect, RuntimeError, ConnectionError):
        pass
    finally:
        await _kill_proc(proc)


async def _audio_ws_playback(ws: WebSocket, ssh_target: str):
    """Browser -> speaker: mono PCM16 in, duplicated to stereo (the sink rejects mono), to aplay."""
    await ws.accept()
    import numpy as np
    cmd = _scout_ssh(ssh_target, f"exec aplay -q -D {SCOUT_SPK_DEV} -f S16_LE -c 2 -r {AUDIO_RATE} -t raw")
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
    try:
        while True:
            msg = await ws.receive_bytes()
            if not msg:
                continue
            proc.stdin.write(np.repeat(np.frombuffer(msg, dtype="<i2"), 2).tobytes())
            await proc.stdin.drain()
    except (WebSocketDisconnect, RuntimeError, ConnectionError, BrokenPipeError):
        pass
    finally:
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass
        await _kill_proc(proc)


async def _kill_proc(proc):
    """Kill the local ssh; the remote arecord/aplay exits on the broken pipe."""
    try:
        proc.terminate()
        await asyncio.wait_for(proc.wait(), timeout=3)
    except (ProcessLookupError, asyncio.TimeoutError):
        try:
            proc.kill()
        except ProcessLookupError:
            pass


@app.websocket("/api/audio/scout/{sid}/listen")
async def scout_audio_listen(ws: WebSocket, sid: str):
    """Hear a Scout's mic. Token as a query param, since a room mic is privacy-relevant."""
    e = SCOUTS.get(sid)
    if e is None or not e.ssh:
        await ws.close(code=4404)
        return
    if CONTROL_TOKEN and ws.query_params.get("token") != CONTROL_TOKEN:
        await ws.close(code=4401)
        return
    await _audio_ws_capture(ws, e.ssh)


@app.websocket("/api/audio/scout/{sid}/talk")
async def scout_audio_talk(ws: WebSocket, sid: str):
    """Talk through a Scout's speaker. Push-to-talk on the browser keeps it half-duplex (no AEC)."""
    e = SCOUTS.get(sid)
    if e is None or not e.ssh:
        await ws.close(code=4404)
        return
    if CONTROL_TOKEN and ws.query_params.get("token") != CONTROL_TOKEN:
        await ws.close(code=4401)
        return
    await _audio_ws_playback(ws, e.ssh)


@app.post("/api/scout/{sid}/drive")
async def api_scout_drive(sid: str, request: Request):
    """Bounded motion. +y is FORWARD, +x strafes (ignored by the tracked model) - see scout.drive()."""
    require_control(request)
    e = _scout_entry(sid)
    if not e.client:
        raise HTTPException(503, "bridge not configured")
    try:
        b = await request.json()
    except (json.JSONDecodeError, ValueError):
        b = {}
    # A tracked robot cannot strafe; drop x so a stray gamepad axis cannot scrub the treads.
    x = 0.0 if e.kinematics == "tracked" else b.get("x", 0.0)
    return _reachy_result(*e.client.drive(x=x, y=b.get("y", 0.0),
                                          yaw=b.get("yaw", 0.0), duration=b.get("duration", 0.6)))


@app.post("/api/scout/{sid}/stop")
def api_scout_stop(sid: str, request: Request):
    require_control(request)
    e = _scout_entry(sid)
    if not e.client:
        raise HTTPException(503, "bridge not configured")
    return _reachy_result(*e.client.stop())


@app.post("/api/scout/{sid}/check")
def api_scout_check(sid: str, request: Request):
    """Describe what a Scout sees. Description only - the rangefinder judges obstacles, not the VLM."""
    require_control(request)
    e = _scout_entry(sid)
    if not e.bridge:
        raise HTTPException(503, "bridge not configured")
    try:
        img = requests.get(f"{e.bridge}/still.jpg", timeout=8)
        if img.status_code != 200 or not img.content:
            raise HTTPException(503, "no frame available")
    except requests.RequestException as ex:
        raise HTTPException(503, f"bridge unreachable: {ex}")

    payload = {
        "model": "cosmos3-edge",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(img.content).decode()}},
            {"type": "text", "text": alert_policy.DESCRIBE_PROMPT},
        ]}],
        "max_tokens": 128,
        "temperature": 0.0,
    }
    try:
        r = requests.post(f"{CFG['cosmos3_url'].rstrip('/')}/v1/chat/completions", json=payload, timeout=180)
        r.raise_for_status()
        desc = r.json()["choices"][0]["message"]["content"].strip()
    except (requests.RequestException, KeyError, ValueError) as ex:
        raise HTTPException(503, f"cosmos: {ex}")

    cats = sorted((alert_policy.classify(desc).get("categories") or []))
    state = e.client.state() if e.client else {}
    return {"description": desc, "categories": cats,
            "alert": bool(cats), "tof_m": state.get("tof_m"), "at": time.time()}


_bg_tasks: set = set()


async def _handoff_to_app(host: str, container: str, name: str) -> None:
    """Restart the robot's ROS stack, then bounce our bridge so it rejoins the fresh roscore.

    Runs detached from the request. Two things this learned the hard way:

    1. The ssh call must not gate the HTTP response. Restarting roller_eye relaunches ~14 ROS nodes
       and the robot is CPU-starved for ~20 s, during which the ssh channel close hangs well past
       any sane request timeout - so the endpoint used to return a 504 even though the restart had
       already happened. Here it is fire-and-forget with a generous wait.

    2. The bridge does NOT self-heal from a *roscore* restart. Its watchdogs recover a restarted
       NODE (SensorNode/CoreNode) while the master stays up, but when the master itself restarts,
       rospy's registration is stale and resubscribing never reconnects (observed: frozen 60 s+).
       A container bounce is the reliable fix - init_node then blocks until the NEW master answers.
       The bridge is bounced AFTER a delay so it binds the new roscore, not the old one mid-death.
    """
    ssh = ["ssh", "-i", SCOUT_SSH_KEY, "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
           "-o", "ConnectTimeout=10", f"root@{host}", "systemctl restart --no-block roller_eye.service"]
    try:
        proc = await asyncio.create_subprocess_exec(
            *ssh, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        try:
            await asyncio.wait_for(proc.wait(), timeout=45)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        print(f"[feed] handoff: issued roller_eye restart on {name} ({host})", flush=True)
    except OSError as ex:
        print(f"[feed] handoff: ssh to {host} failed: {ex}", flush=True)
        return
    if not container:
        return
    await asyncio.sleep(22)   # let roscore come fully back before the bridge reconnects
    ok, msg = await asyncio.get_event_loop().run_in_executor(
        None, lambda: run(["sudo", "-n", "docker", "restart", container], 60))
    print(f"[feed] handoff: bounced bridge {container} ok={ok} {msg[:120]}", flush=True)


@app.post("/api/scout/{sid}/restart-ros")
async def api_scout_restart_ros(sid: str, request: Request):
    """Hand a Scout back to the official Moorebot app by restarting its ROS stack.

    The app reaches the robot through its /CloudNode and /AppNode, which come up with the roller_eye
    ROS graph; bouncing the graph re-establishes them and clears any state left by our drive relay.
    Needs root (sudo is deliberately crippled on this robot), so it uses the durable root key, not
    the linaro audio login. The actual work - restart robot ROS, then bounce our bridge so our own
    card recovers - runs in the background (see _handoff_to_app) so this returns at once.
    """
    require_control(request)
    e = _scout_entry(sid)
    if not e.ssh:
        raise HTTPException(503, "no ssh target for this scout")
    host = e.ssh.split("@")[-1]
    t = asyncio.create_task(_handoff_to_app(host, e.container, e.name))
    _bg_tasks.add(t)
    t.add_done_callback(_bg_tasks.discard)
    return {"message": f"Handing {e.name} back to the Moorebot app — restarting its ROS and "
                       f"reconnecting the bridge. This feed drops out and returns in ~30–40 s."}


@app.get("/api/reachy/apps")
def api_reachy_apps():
    """Installed robot apps, which one is running, and which starts at boot."""
    if not _reachy:
        raise HTTPException(503, "reachy_daemon_url is not configured")
    return _reachy.apps()


@app.post("/api/reachy/apps/start/{name}")
def api_reachy_app_start(name: str, request: Request):
    require_control(request)
    if not _reachy:
        raise HTTPException(503, "reachy_daemon_url is not configured")
    return _reachy_result(*_reachy.start_app(name))


@app.post("/api/reachy/apps/stop")
def api_reachy_app_stop(request: Request):
    require_control(request)
    if not _reachy:
        raise HTTPException(503, "reachy_daemon_url is not configured")
    return _reachy_result(*_reachy.stop_app())


@app.get("/api/reachy/limits")
def api_reachy_limits():
    """Ranges for the control panel, so the sliders cannot ask for something unreachable.

    Served rather than hard-coded in the page because they were measured against this robot - in
    particular the head saturates at about +0.019 m upward while the daemon still answers 200.
    """
    return {
        "translation_m": LIMITS_M,
        "rotation_rad": {k: list(v) for k, v in LIMITS_RAD.items()},
        "antenna_rad": [-ANTENNA_LIMIT_RAD, ANTENNA_LIMIT_RAD],
        "antenna_park_rad": math.radians(ANTENNA_PARK_DEG),
    }


@app.get("/api/reachy/alert")
def api_reachy_alert():
    """Most recent anomaly check: what it saw, and whether the policy called it an alert."""
    return JSONResponse({**_reachy_alert, "watching": bool(REACHY_WATCH and _reachy and REACHY_CAM),
                         "cooldown_s": REACHY_WATCH_COOLDOWN,
                         "triggers": {"speech": True,
                                      "labels": sorted(REACHY_TRIGGER_LABELS),
                                      "cameras": sorted(REACHY_TRIGGER_CAMERAS) or ["any"]}})


@app.post("/api/reachy/check")
def api_reachy_check(request: Request):
    """Look now, on demand. Costs one VLM call."""
    require_control(request)
    res = reachy_look_and_describe("manual")
    if res.get("error"):
        raise HTTPException(502, res["error"])
    return res


@app.get("/healthz")
def healthz():
    with closing(db()) as c:
        n = c.execute("SELECT COUNT(*) n FROM entries").fetchone()["n"]
    return {"ok": True, "entries": n, "engine": active_engine()["id"], "free_mb": free_mb()}


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>porch dad command center</title>
<!-- Added to the iOS home screen this runs standalone (no Safari chrome), which is the intended
     way to reach the control plane from a phone. -->
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="porch dad">
<meta name="mobile-web-app-capable" content="yes">
<meta name="theme-color" content="#0b0d0c">
<style>
:root{--bg:#0b0d0c;--card:#161a15;--line:#2a2f28;--fg:#e8ece7;--mut:#9aa396;--g:#76b900;
      --r:#ff5c5c;--y:#ffb020;--b:#4aa3ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:16px}
header{border-bottom:1px solid var(--line);position:sticky;top:0;background:rgba(11,13,12,.96);
       backdrop-filter:blur(8px);z-index:9}
h1{font-size:18px;margin:0}h1 span{color:var(--g)}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:var(--mut);margin:20px 0 8px}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
button{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:9px;
       padding:7px 13px;font-size:13.5px;cursor:pointer}
button:hover{border-color:var(--g)}
button.on{background:rgba(118,185,0,.16);border-color:var(--g);color:var(--g)}
button.warn{border-color:var(--y);color:var(--y)}
button.busy{opacity:.5;pointer-events:none}
button.locked{opacity:.55;cursor:not-allowed;border-style:dashed}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px;margin-bottom:10px}
.tag{font-size:11px;font-weight:700;border-radius:999px;padding:2px 9px;letter-spacing:.04em}
.e-v1{background:rgba(74,163,255,.15);color:var(--b)}
.e-v2{background:rgba(255,176,32,.15);color:var(--y)}
.e-v3{background:rgba(118,185,0,.15);color:var(--g)}
.meta{color:var(--mut);font-size:12px;display:flex;gap:12px;flex-wrap:wrap;margin-top:7px}
.desc{margin:6px 0 0}
.still{width:100%;max-height:320px;object-fit:cover;border-radius:9px;margin-top:8px;
       background:#0b0d0c;display:block}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{border-bottom:1px solid var(--line);padding:7px 9px;text-align:left}
th{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.06em}
.msg{padding:8px 12px;border-radius:9px;margin:8px 0;font-size:13px;display:none}
.msg.ok{background:rgba(118,185,0,.12);color:var(--g);display:block}
.msg.err{background:rgba(255,92,92,.12);color:var(--r);display:block}
.hint{color:var(--mut);font-size:12px;margin:4px 0 0}
a{color:var(--g);text-decoration:none}
.chip{display:inline-flex;align-items:center;gap:7px;background:var(--card);color:var(--fg);
      border:1px solid var(--line);border-radius:9px;padding:7px 13px;font-size:13.5px;
      cursor:pointer;text-decoration:none}
a.chip:hover{border-color:var(--g);background:rgba(118,185,0,.10)}
.chip.on{border-color:var(--g)}
.chip.down{opacity:.6;border-style:dashed}

/* Reachy controller - the same controls the robot's own desktop app offers (antennas, head
   X/Y + Z, pitch/yaw, roll, body yaw), in this page's colours rather than the app's orange.
   Values are shown in radians and metres because that is what the app shows and what the
   daemon takes; converting for display only would add a place for a sign error. */
.sec{font-size:11px;text-transform:uppercase;letter-spacing:.09em;color:var(--mut);
     margin:14px 0 6px;display:flex;align-items:center;gap:7px}
.ctlrow{display:flex;gap:8px;flex-wrap:wrap}
.ctl{background:rgba(255,255,255,.02);border:1px solid var(--line);border-radius:10px;
     padding:9px 11px;flex:1 1 170px;min-width:0}
.ctl h4{margin:0 0 6px;font-size:12px;font-weight:600;display:flex;justify-content:space-between;
        align-items:baseline;gap:8px}
.ctl h4 b{color:var(--mut);font-weight:400;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
          font-size:11.5px}
/* 2D pad: drag the dot, both axes at once. */
.pad{position:relative;width:100%;aspect-ratio:1;max-width:150px;margin:0 auto;
     border:1px solid var(--line);border-radius:10px;background:
     linear-gradient(var(--line),var(--line)) center/1px 100% no-repeat,
     linear-gradient(var(--line),var(--line)) center/100% 1px no-repeat;
     touch-action:none;cursor:crosshair}
.pad i{position:absolute;width:15px;height:15px;border-radius:50%;background:var(--g);
       transform:translate(-50%,-50%);left:50%;top:50%;box-shadow:0 0 0 4px rgba(118,185,0,.18)}
.padwrap{display:flex;gap:9px;align-items:stretch}
.vwrap{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:5px}
input[type=range]{width:100%;accent-color:var(--g)}
input[type=range].vert{writing-mode:vertical-lr;direction:rtl;width:22px;height:120px}
.ctl.wide{flex-basis:100%}

/* Scout drive: a D-pad grid plus a rotate pair, laid out like the robot's own app. */
.sxy{display:flex;gap:18px;align-items:center;margin-top:10px;flex-wrap:wrap}
.dpad{display:grid;grid-template-columns:repeat(3,46px);grid-template-rows:repeat(3,46px);gap:5px}
.dpad button{width:46px;height:46px;padding:0;font-size:17px;border-radius:9px}
.d-u{grid-area:1/2}.d-l{grid-area:2/1}.d-c{grid-area:2/2}.d-r{grid-area:2/3}.d-d{grid-area:3/2}
.dpad button:active{background:rgba(118,185,0,.18);border-color:var(--g)}
/* Holding a drive button on iOS otherwise pops the text-selection magnifier / copy callout and
   starts a selection. Suppress selection and the callout on the controls, and touch-action:none on
   the pads stops a press-drag from scrolling the page. The caption box re-enables selection so the
   Cosmos description can still be copied. */
.scoutCard button{-webkit-user-select:none;user-select:none;-webkit-touch-callout:none;touch-action:manipulation}
.scoutCard .dpad,.scoutCard .rotpad{-webkit-user-select:none;user-select:none;-webkit-touch-callout:none;touch-action:none}
.scoutCard .ralert{-webkit-user-select:text;user-select:text}
.rotpad{display:flex;flex-direction:column;gap:4px;align-items:center}
.rotpad button{width:52px;height:46px;font-size:19px}
.batt{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.chip.locked{opacity:.55;cursor:not-allowed;border-style:dashed}
.chip .hint{margin:0;font-size:11.5px;opacity:.8}
.callout{background:rgba(74,163,255,.09);border:1px solid rgba(74,163,255,.35);border-left:3px solid var(--b);
         border-radius:9px;padding:10px 13px;margin:0 0 10px;font-size:13px;line-height:1.5}
.callout b{color:var(--b)}
button:disabled{opacity:.45;cursor:default}
button:disabled:hover{border-color:var(--line)}
button.off{background:rgba(255,92,92,.14);border-color:var(--r);color:var(--r)}
.svc{display:flex;gap:12px;align-items:center;justify-content:space-between;flex-wrap:wrap;
     background:var(--card);border:1px solid var(--line);border-radius:12px;padding:10px 12px;margin-bottom:8px}
.svc-id{min-width:0;flex:1}
.svc-name{font-size:14.5px}
.svc.held{border-left:3px solid var(--y)}
.intent{margin-top:4px;font-size:12px;color:var(--y)}
.intent.off b{color:var(--r)}
.ralert{display:none;border-radius:9px;padding:9px 12px;margin-bottom:8px;font-size:13px}
.ralert.hit{display:block;background:rgba(255,92,92,.13);border:1px solid var(--r);color:var(--r)}
.ralert.clear{display:block;background:rgba(118,185,0,.10);border:1px solid var(--line);color:var(--mut)}
.vol{margin-top:10px}
.vol label{display:block;margin-top:6px;font-size:12px;color:var(--mut)}
.svc .hint{margin:2px 0 0;word-break:break-word}
.selfurl{display:flex;gap:10px;align-items:center;flex-wrap:wrap;background:var(--card);
         border:1px solid var(--line);border-left:3px solid var(--g);border-radius:10px;
         padding:9px 12px;margin-bottom:10px;font-size:13px}
.selfurl a{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13.5px;word-break:break-all}
.selfurl>span:first-child{color:var(--mut);text-transform:uppercase;font-size:11px;letter-spacing:.06em}
button.mini{padding:3px 9px;font-size:11.5px}
</style></head><body>
<header><div class="wrap"><div class="row" style="justify-content:space-between">
  <h1>porch dad <span>command center</span></h1>
  <div class="row"><span id="mem" class="hint"></span><a href="/rss">RSS</a></div>
</div></div></header>
<div class="wrap">

  <h2>Open a service</h2>
  <div class="selfurl">
    <span>This page</span>
    <a id="selfurl" href="/">…</a>
    <button class="mini" onclick="copySelf()">copy</button>
    <span class="hint">Add to Home Screen on iOS to run it as an app.</span>
  </div>
  <div class="row" id="links"></div>
  <p class="hint">Every service with its port, polled every 20&nbsp;s — links stay clickable even
     when a service is down (🟢 responding, ⚪ not responding). Dashed entries are not browsable
     web UIs (MQTT, RTSP, VNC).</p>

  <h2>Moorebot Scouts</h2>
  <!-- One card per robot, built by buildScoutCards() from /api/scouts. Empty until that resolves;
       a robot whose bridge is down renders a card that reports the bridge is not running, rather
       than silently vanishing - the same honest-state rule the rest of the page follows. -->
  <div id="scoutCards"></div>

  <!-- Action feedback banner. Placed BELOW the Scout controls, not at the top, so that a message
       appearing or clearing never reflows the drive pad the user is holding. -->
  <div id="msg" class="msg"></div>

  <h2>Reachy Mini</h2>
  <div class="card" id="reachyCard">
    <div class="row" style="justify-content:space-between;align-items:center">
      <span id="reachyState" class="hint">checking…</span>
      <!-- hrefs come from /api/reachy, built on links_host (see lan_link): the URLs in config are
           loopback, which on a phone would point at the phone. -->
      <div class="row">
        <a id="reachyLink" class="chip" href="#" target="_blank" rel="noreferrer">Open in Live VLM WebUI</a>
        <a id="reachyVisionLink" class="chip" href="#" target="_blank" rel="noreferrer"
           style="display:none">Open in Live Vision</a>
      </div>
    </div>
    <img id="reachyImg" class="still" alt="Reachy Mini camera" style="display:none">
    <p class="hint" id="reachyHint" style="display:none">
      No frames. This reads <code>reachy-mjpeg-bridge.service</code> on this box, which pulls the
      robot's WebRTC stream directly — it does <b>not</b> need the Live VLM WebUI running.
    </p>
    <!-- The Cosmos caption belongs with the picture it describes, not buried in the controls. -->
    <div id="reachyAlert" class="ralert"></div>
    <div class="row" style="margin-top:8px">
      <button onclick="rq('/api/reachy/check')"
              title="Grabs one frame from the camera above and sends it to Cosmos3-Edge. This is the button that writes a new caption.">
        Look &amp; describe</button>
    </div>
    <p class="hint" id="reachyWatchHint"></p>
  </div>

  <div class="card" id="reachyCtl" style="display:none">
    <div class="hint" id="reachyStatus2">checking…</div>
    <!-- What currently owns the robot. The daemon gives one app the lock at a time and that app
         takes the camera and microphone with it, so this is the line that explains a dark preview
         and a deaf anomaly watcher. -->
    <div class="hint" id="reachyBusy"></div>

    <div class="sec">▷ Apps <span class="hint" id="reachyAppNow" style="text-transform:none;letter-spacing:0">—</span></div>
    <div class="row" id="reachyApps"></div>
    <p class="hint">One app at a time; starting one stops the other. A conversation app takes the
       microphone and camera, so the anomaly watcher goes quiet while it runs.</p>

    <div class="row" style="margin-top:10px">
      <button onclick="rq('/api/reachy/action/wake')">Wake</button>
      <button onclick="rq('/api/reachy/action/sleep')">Sleep</button>
    </div>

    <!-- Pose, motor modes and volumes are occasional; collapsed so they stop dominating the page.
         Native <details> rather than a scripted toggle, so it survives the 10 s refresh without
         any state to keep. -->
    <details id="reachyPose" style="margin-top:12px">
      <summary style="cursor:pointer;color:var(--mut);font-size:12px;text-transform:uppercase;
                      letter-spacing:.08em">Pose, motors &amp; volume</summary>

    <div class="row" style="margin-top:10px">
      <button onclick="rq('/api/reachy/action/center')">Centre</button>
      <!-- Label, not route: the endpoint name stays put so anything already calling it keeps
           working. "Look at voice" read like it produced a caption, which it does not. -->
      <button onclick="rq('/api/reachy/action/look-at-voice')"
              title="Turns the head toward the last sound the microphone array heard. No camera, no caption.">
        Face the last sound</button>
    </div>
    <div class="row" style="margin-top:8px">
      <span class="hint">Motors</span>
      <button onclick="rq('/api/reachy/motors/enabled')">Stiff</button>
      <button onclick="rq('/api/reachy/motors/gravity_compensation')">Soft</button>
      <button onclick="rq('/api/reachy/motors/disabled')">Limp</button>
    </div>

    <div class="sec">⌇⌇ Antennas</div>
    <div class="ctlrow">
      <div class="ctl">
        <h4>Left <b id="antLv">0.000 rad</b></h4>
        <input type="range" id="antL" min="-3.14" max="3.14" step="0.01" value="0.175">
      </div>
      <div class="ctl">
        <h4>Right <b id="antRv">0.000 rad</b></h4>
        <input type="range" id="antR" min="-3.14" max="3.14" step="0.01" value="0.175">
      </div>
    </div>

    <div class="sec">◉◉ Head</div>
    <div class="ctlrow">
      <div class="ctl">
        <h4>Position X/Y <b id="xyv">0.000 0.000</b></h4>
        <div class="padwrap">
          <div class="pad" id="padXY"><i></i></div>
          <div class="vwrap">
            <span class="hint" style="font-size:10px">Z</span>
            <input type="range" class="vert" id="posZ" min="-0.018" max="0.018" step="0.001" value="0">
            <b class="hint" id="zv" style="font-size:10.5px">0.000</b>
          </div>
        </div>
      </div>
      <div class="ctl">
        <h4>Pitch / Yaw <b id="pyv">0.000 0.000</b></h4>
        <div class="pad" id="padPY"><i></i></div>
      </div>
    </div>
    <div class="ctlrow" style="margin-top:8px">
      <div class="ctl wide">
        <h4>Roll <b id="rollv">0.000 rad</b></h4>
        <input type="range" id="roll" min="-0.7" max="0.7" step="0.01" value="0">
      </div>
    </div>

    <div class="sec">▭ Body</div>
    <div class="ctlrow">
      <div class="ctl wide">
        <h4>Yaw <b id="byawv">0.000 rad</b></h4>
        <input type="range" id="byaw" min="-2.79" max="2.79" step="0.01" value="0">
      </div>
    </div>
    <p class="hint">
      Live control: these set the target the robot's 50&nbsp;Hz loop is already chasing, so they
      track your finger rather than queueing moves. Pitch is positive <b>downward</b> — the pad is
      inverted so dragging up looks up. Head travel is ±20&nbsp;mm in X/Y and ±18&nbsp;mm in Z; the
      platform saturates at about +19&nbsp;mm however far you ask.
    </p>
    <div class="vol">
      <label>Microphone <span id="micVal" class="hint"></span></label>
      <input type="range" id="micVol" min="0" max="100" step="5"
             oninput="document.getElementById('micVal').textContent=this.value+'%'"
             onchange="rq('/api/reachy/volume/mic/'+this.value)">
      <label>Speaker <span id="spkVal" class="hint"></span></label>
      <input type="range" id="spkVol" min="0" max="100" step="5"
             oninput="document.getElementById('spkVal').textContent=this.value+'%'"
             onchange="rq('/api/reachy/volume/speaker/'+this.value)">
    </div>
    </details>
  </div>

  <h2>Feed <span id="filter" class="hint"></span></h2>
  <div class="row" id="filters"></div>
  <div id="feed" style="margin-top:10px"></div>

  <h2>Cosmos3-Edge engine</h2>
  <div class="callout">
    <b>All three engines are equally accurate — 95.7% (22/23).</b>
    Measured on one identical 23-frame set scored against a hand-checked truth set: every engine
    returned the same score and missed the same frame. They share identical INT4 weights and differ
    only in context budget, which a single snapshot never exercises.
    <b>Pick on speed and RAM, not accuracy</b> — and note the fastest engine is also the one that
    spends the fewest tokens per frame.
  </div>
  <div class="row" id="engines"></div>
  <p class="hint" id="enginehint"></p>

  <h2>Service power</h2>
  <div id="services"></div>
  <p class="hint"><b>Services stopped on purpose stay stopped.</b> Turning one off here records
     who did it and when; the card then shows a "stopped by user" badge, turning it back on asks
     for confirmation, and <code>/api/status</code> exposes the same flag so automation can check
     before restarting something deliberately shut down.</p>
  <p class="hint"><b>Some choices also survive a reboot.</b> Services showing <i>starts at boot</i> /
     <i>off at boot</i> follow the last ON/OFF here: once the action is verified, ON enables the
     unit at boot and OFF disables it, so a reboot neither brings back a UI you stopped nor loses
     one you started.</p>
  <p class="hint">Every action is verified against the <b>port</b>, not against systemd — a unit can
     report <i>active</i> while the shim is still loading engines and cannot answer. Stops report
     how much RAM actually came back; starting is refused when free memory is low, because the
     model needs ~3.3&nbsp;GB of 8&nbsp;GB and an OOM kill would take the model down.</p>

  <h2>Camera power</h2>
  <div class="row" id="cameras"></div>
  <p class="hint"><b>Powered</b> keeps a continuous RTSP stream for full NVR (drains battery cameras).
     <b>Saver</b> disables the stream; motion events still arrive via ring-mqtt at no battery cost.</p>

  <h2>Comparison</h2>
  <table id="cmp"><thead><tr><th>Engine</th><th>Captions</th><th>Measured</th>
    <th>Avg event latency</th>
    <th>Avg peak CPU</th><th>Avg peak VRAM</th><th>Avg peak GPU</th><th>Avg length</th></tr></thead>
    <tbody></tbody></table>
  <p class="hint"><b>Read the latency column carefully.</b> It is the whole Frigate pipeline —
     event end, clip finalise, Frigate's own GenAI call, then this service noticing the description
     — so it runs to seconds and is <i>not</i> the model's inference time. It is dominated by
     Frigate, not by the engine, and <b>cannot be used to rank engines</b>. The per-inference
     figures are the ones on the engine buttons above (recorded from a dedicated eval), and the
     live per-request number is the shim's own <code>[perf] elapsed_ms</code> line in
     <code>journalctl -u cosmos3-edge-shim</code>. The columns that <i>do</i> compare engines
     meaningfully here are peak CPU / VRAM / GPU and caption length.</p>

</div>
<script>
let FILTER='all';
const fmt = t => new Date(t*1000).toLocaleString([], {month:'short',day:'numeric',hour:'2-digit',minute:'2-digit',second:'2-digit'});
const esc = s => (s||'').replace(/[<>&]/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]));
function say(t, ok){ const m=document.getElementById('msg'); m.textContent=t; m.className='msg '+(ok?'ok':'err');
  setTimeout(()=>{m.className='msg'},6000); }
let TOKEN='';
// The address to hand to iOS "Add to Home Screen". Taken from the browser's own location so it is
// always the reachable one -- on this box 0.0.0.0 in config is not a thing a phone can open.
function showSelfUrl(){
  const u = window.location.origin + '/';
  const a = document.getElementById('selfurl');
  a.textContent = u; a.href = u;
}
async function copySelf(){
  try{ await navigator.clipboard.writeText(window.location.origin + '/'); say('Address copied', true); }
  catch(e){ say('Copy failed - long-press the link instead', false); }
}
// ---------------------------------------------------------------- Reachy live controller
// One target object, one throttled sender. The robot is already running a 50 Hz loop chasing
// whatever target it last received, so the job here is only to keep that target fresh: send at
// most every 60 ms while dragging, and always send once more on release so the final resting
// value is never the one that got throttled away.
const T = {x:0,y:0,z:0,roll:0,pitch:0,yaw:0,body_yaw:0,antennas:[0.175,0.175]};
let tBusy=false, tPend=false, tLast=0, tGrabbed=0;
const D2R = Math.PI/180;

function tTouch(){ tGrabbed = Date.now(); }          // suppress state sync while the user drives

async function tSend(){
  if(tBusy){ tPend=true; return; }
  const now=Date.now();
  if(now-tLast < 60){ if(!tPend){ tPend=true; setTimeout(()=>{tPend=false;tSend();}, 60-(now-tLast)); } return; }
  tBusy=true; tLast=now;
  try{
    const tok = await tokenReady();
    const r = await fetch('/api/reachy/target',{method:'POST',
      headers:{'Content-Type':'application/json','X-Porch-Token':tok},
      body:JSON.stringify(T)});
    if(!r.ok){ const j=await r.json().catch(()=>({})); say(j.detail||j.message||'target refused', false); }
  }catch(e){ /* a dropped frame of control is not worth a banner */ }
  tBusy=false;
  if(tPend){ tPend=false; tSend(); }
}

// The control token arrives with the first /api/status poll. Clicking before that landed used to
// fail with a bare 401 that looked like "the controls are broken" - which is exactly how this was
// first reported - so wait for it instead of sending an empty header.
async function tokenReady(){
  if(TOKEN) return TOKEN;
  try{ const st = await (await fetch('/api/status',{cache:'no-store'})).json();
       TOKEN = st.control_token || ''; }catch(e){}
  return TOKEN;
}

// Not `fmt`: the page already has a const fmt for dates, and redeclaring a const is a hard
// SyntaxError that would kill every script on the page, not just this panel.
function rcNum(v,n){ return (v<0?'':' ') + v.toFixed(n===undefined?3:n); }

function bindSlider(id, lblId, key, unit, idx){
  const el=document.getElementById(id), lbl=document.getElementById(lblId);
  if(!el) return;
  const paint=()=>{ const v=parseFloat(el.value);
    lbl.textContent = rcNum(v) + (unit||''); };
  el.addEventListener('input', ()=>{ tTouch();
    const v=parseFloat(el.value);
    if(idx===undefined) T[key]=v; else T.antennas[idx]=v;
    paint(); tSend(); });
  paint();
}

// A pad maps the two axes of a square onto two target fields. `inv` flips the vertical axis for
// pitch, which is positive downward on this robot: without it, dragging up would look down.
function bindPad(id, lblId, kx, ky, rx, ry, inv){
  const pad=document.getElementById(id); if(!pad) return;
  const dot=pad.querySelector('i'), lbl=document.getElementById(lblId);
  let down=false;
  const paint=()=>{
    const fx=(T[kx]/rx+1)/2, fy=((inv?-T[ky]:T[ky])/ry+1)/2;
    dot.style.left=(Math.min(1,Math.max(0,fx))*100)+'%';
    dot.style.top=(100-Math.min(1,Math.max(0,fy))*100)+'%';
    lbl.textContent=rcNum(T[kx])+' '+rcNum(T[ky]);
  };
  const at=(ev)=>{
    const b=pad.getBoundingClientRect();
    const fx=Math.min(1,Math.max(0,(ev.clientX-b.left)/b.width));
    const fy=Math.min(1,Math.max(0,(ev.clientY-b.top)/b.height));
    T[kx]=(fx*2-1)*rx;
    const vy=((1-fy)*2-1)*ry;
    T[ky]=inv?-vy:vy;
    tTouch(); paint(); tSend();
  };
  pad.addEventListener('pointerdown',e=>{down=true;pad.setPointerCapture(e.pointerId);at(e);});
  pad.addEventListener('pointermove',e=>{if(down)at(e);});
  pad.addEventListener('pointerup',  e=>{down=false;at(e);});
  pad.addEventListener('pointercancel',()=>{down=false;});
  pad._paint=paint; paint();
}

// Pull the live pose back into the widgets, but only when the user is not driving them - snapping
// a slider out from under a finger is worse than a stale reading.
function syncCtl(rs){
  if(Date.now()-tGrabbed < 1500) return;
  const p=rs.pose_deg||{}, m=rs.pos_m||{};
  const set=(id,lbl,v,unit)=>{const e=document.getElementById(id);
    if(e&&document.activeElement!==e&&v!=null){e.value=v;
      const b=document.getElementById(lbl); if(b) b.textContent=rcNum(v)+(unit||'');}};
  if(p.roll!=null){ T.roll=p.roll*D2R; set('roll','rollv',T.roll,' rad'); }
  if(p.pitch!=null) T.pitch=p.pitch*D2R;
  if(p.yaw!=null)   T.yaw=p.yaw*D2R;
  if(rs.body_yaw_deg!=null){ T.body_yaw=rs.body_yaw_deg*D2R; set('byaw','byawv',T.body_yaw,' rad'); }
  if(m.x!=null) T.x=m.x; if(m.y!=null) T.y=m.y;
  if(m.z!=null){ T.z=m.z; set('posZ','zv',T.z,''); }
  const a=rs.antennas_deg;
  if(a&&a.length===2){ T.antennas=[a[0]*D2R,a[1]*D2R];
    set('antL','antLv',T.antennas[0],' rad'); set('antR','antRv',T.antennas[1],' rad'); }
  const px=document.getElementById('padXY'), pp=document.getElementById('padPY');
  if(px&&px._paint) px._paint();
  if(pp&&pp._paint) pp._paint();
}

// ---------------------------------------------------------------- Moorebot Scouts
// One card per robot, built from /api/scouts. Every DOM id is suffixed with the robot's sid
// (sc-<part>-<sid>) and every bit of runtime state lives in SC[sid], so two robots never share a
// timer, WebSocket or audio context. +y is FORWARD and +x strafes on this firmware - not the ROS
// convention - so the axis mapping lives here and the buttons read the human way. A tracked robot
// cannot strafe: its strafe buttons are omitted and the backend drops any x regardless.
const SC = {};                 // sid -> runtime state (vec,timer,gp*,ws,ac,talk*,meta)
function scEl(sid, part){ return document.getElementById('sc-'+part+'-'+sid); }

function scoutCardHTML(m){
  const sid = m.id, mec = m.kinematics !== 'tracked';
  // Tracked robots turn in place by scrubbing both treads sideways, which needs far more command
  // than a mecanum spin - 0.7 rad/s barely moved them. Tunable up to the bridge's 2.0 ceiling.
  const rot = mec ? 0.7 : 1.5;
  // A held button repeats its command through scoutHold; releasing (or leaving) stops. touchstart
  // is defaulted-prevented so a press does not also scroll or fire a synthetic mouse event.
  const held = (dx,dy,dyaw,cls,title,glyph) =>
    `<button class="${cls}" title="${title}"`
    + ` onmousedown="scoutHold('${sid}',${dx},${dy},${dyaw})" onmouseup="scoutRelease('${sid}')" onmouseleave="scoutRelease('${sid}')"`
    + ` ontouchstart="event.preventDefault();scoutHold('${sid}',${dx},${dy},${dyaw})" ontouchend="scoutRelease('${sid}')">${glyph}</button>`;
  const kind = mec ? 'mecanum · strafes' : 'tracked · no strafe';
  return `
  <div class="card scoutCard" id="sc-card-${sid}" style="display:none">
    <div class="row" style="justify-content:space-between;align-items:center">
      <b>${esc(m.name)} <span class="hint" style="font-weight:400">· ${kind}</span></b>
      <span id="sc-state-${sid}" class="hint">checking…</span>
    </div>
    <img id="sc-img-${sid}" class="still" alt="${esc(m.name)} camera" style="display:none">
    <p class="hint" id="sc-hint-${sid}" style="display:none"></p>
    <!-- Latest Cosmos caption, right under the video it describes. Persists between refreshes. -->
    <div id="sc-alert-${sid}" class="ralert"></div>
    <div class="sxy">
      <div class="dpad">
        ${held(0,0.15,0,'d-u','forward','▲')}
        ${mec ? held(-0.15,0,0,'d-l','strafe left','◀') : ''}
        <button class="d-c warn" onclick="post('/api/scout/${sid}/stop')" title="stop">■</button>
        ${mec ? held(0.15,0,0,'d-r','strafe right','▶') : ''}
        ${held(0,-0.15,0,'d-d','back','▼')}
      </div>
      <div class="rotpad">
        <span class="hint">rotate</span>
        <div class="row">
          ${held(0,0,rot,'','rotate left (CCW)','⟲')}
          ${held(0,0,-rot,'','rotate right (CW)','⟳')}
        </div>
      </div>
    </div>
    <!-- Gamepad: any HID controller the browser sees (Luna, Xbox, PS). One stick drives one robot -
         enabling it here disables it on the other card. Left stick = strafe + forward (forward only
         on a tracked robot), right stick X = rotate. -->
    <div class="row" style="margin-top:10px;align-items:center">
      <button id="sc-gpbtn-${sid}" onclick="scoutGamepadToggle('${sid}')">🎮 Use gamepad</button>
      <span id="sc-gpstate-${sid}" class="hint"></span>
    </div>
    <div class="row" style="margin-top:8px">
      <button id="sc-listenbtn-${sid}" onclick="scoutListenToggle('${sid}')"
              title="Hear this robot's microphone in your browser">🔊 Listen</button>
      <button id="sc-talkbtn-${sid}" title="Hold to speak through this robot's speaker (needs https or localhost)"
              onmousedown="scoutTalkStart('${sid}')" onmouseup="scoutTalkStop('${sid}')" onmouseleave="scoutTalkStop('${sid}')"
              ontouchstart="event.preventDefault();scoutTalkStart('${sid}')" ontouchend="scoutTalkStop('${sid}')">🎙 Hold to talk</button>
      <button onclick="scoutSnapshot('${sid}')" title="Save the current frame to your device">📷 Snapshot</button>
      <button onclick="scoutCheck('${sid}',this)"
              title="Sends one frame to Cosmos3-Edge for a description. It is not asked what to do — the rangefinder decides that.">
        Look &amp; describe</button>
      <button class="warn" onclick="scoutRestartRos('${sid}')"
              title="Restart the robot's ROS stack (roller_eye.service) so the official Moorebot app can take control. Drive and video on this card pause ~20 s while it comes back, then recover on their own.">
        ♻ Restart ROS → app</button>
    </div>
    <p class="hint">The <b>range</b> in the status line is the forward time-of-flight sensor and is
       what to trust for obstacles — and it only guards <b>forward</b>: strafe, reverse and rotate
       are unprotected, and there is no rear sensor. Watch the video. Held motion drives 0.6&nbsp;s
       at a time and stops itself when you let go.</p>
  </div>`;
}

// Build a card per robot once, on load. If /api/scouts is empty or fails the panel stays empty,
// which is the honest state - no robots configured.
async function buildScoutCards(){
  const box = document.getElementById('scoutCards');
  if(!box) return;
  let list = [];
  try{ list = (await (await fetch('/api/scouts',{cache:'no-store'})).json()).scouts || []; }
  catch(e){ return; }
  box.innerHTML = list.map(scoutCardHTML).join('');
  list.forEach(m => { SC[m.id] = {meta:m, vec:{x:0,y:0,yaw:0}, timer:null,
                                  gpOn:false, gpRAF:null, gpConn:null,
                                  ws:null, ac:null, playAt:0,
                                  talkWS:null, talkStream:null, talkNode:null, talkCtx:null, talkResume:false}; });
}

// Held motion: repeat the command while a button (or a gamepad stick) is engaged, so the robot
// keeps moving smoothly instead of lurching once per click. The firmware zeroes velocity when
// commands stop, so releasing = stopping with nothing latched.
function scoutHold(sid, x, y, yaw){
  const s = SC[sid]; if(!s) return;
  s.vec = {x, y, yaw};
  if(s.timer) return;
  const tick = () => {
    if(s.vec.x || s.vec.y || s.vec.yaw){
      // duration 0.4 > the 0.15 s tick, so motion never gaps between commands. Quiet: no banner.
      postJSON(`/api/scout/${sid}/drive`, {x:s.vec.x, y:s.vec.y, yaw:s.vec.yaw, duration:0.4}, true);
    }
  };
  tick();
  s.timer = setInterval(tick, 150);
}
function scoutRelease(sid){
  const s = SC[sid]; if(!s) return;
  s.vec = {x:0, y:0, yaw:0};
  if(s.timer){ clearInterval(s.timer); s.timer = null; }
  // One explicit stop so it halts now rather than at the end of the firmware's watchdog window.
  postJSON(`/api/scout/${sid}/stop`, null, true);
}

// Gamepad: deadzoned axes fed through the same held-motion loop. Only one robot owns the single
// stick at a time (gpOwner) - enabling it on one card disables it on the other.
let gpOwner = null;
function scoutGamepadToggle(sid){
  const s = SC[sid]; if(!s) return;
  const btn = scEl(sid,'gpbtn'), st = scEl(sid,'gpstate');
  if(s.gpOn){ scoutGamepadOff(sid); return; }
  if(!('getGamepads' in navigator)){
    st.textContent = 'this browser blocks the gamepad API on http — open the page over https or localhost';
    return;
  }
  if(gpOwner && gpOwner !== sid) scoutGamepadOff(gpOwner);   // one stick, one robot
  gpOwner = sid; s.gpOn = true;
  btn.classList.add('on'); btn.textContent = '🎮 Gamepad on';
  s.gpConn = e => {};   // a connect event just makes the pad visible to getGamepads(); loop reads it
  window.addEventListener('gamepadconnected', s.gpConn);
  scoutGpLoop(sid);
}
function scoutGamepadOff(sid){
  const s = SC[sid]; if(!s) return;
  s.gpOn = false;
  const btn = scEl(sid,'gpbtn'), st = scEl(sid,'gpstate');
  if(btn){ btn.classList.remove('on'); btn.textContent = '🎮 Use gamepad'; }
  if(st) st.textContent = '';
  if(s.gpRAF) cancelAnimationFrame(s.gpRAF);
  if(s.gpConn){ window.removeEventListener('gamepadconnected', s.gpConn); s.gpConn = null; }
  if(gpOwner === sid) gpOwner = null;
  scoutRelease(sid);
}
function scoutGpLoop(sid){
  const s = SC[sid]; if(!s || !s.gpOn) return;
  const pads = navigator.getGamepads ? navigator.getGamepads() : [];
  let gp = null;
  for(const p of pads){ if(p){ gp = p; break; } }
  const st = scEl(sid,'gpstate');
  if(gp){
    const dz = v => Math.abs(v) < 0.15 ? 0 : v;
    // left-stick X -> strafe(+x right, mecanum only); left-stick Y up -> forward(+y);
    // right-stick X right -> rotate CW(-yaw). Same gentle limits as the on-screen pad.
    const tracked = s.meta.kinematics === 'tracked';
    const x   = tracked ? 0 : dz(gp.axes[0] || 0) * 0.2;
    const y   = -dz(gp.axes[1] || 0) * 0.2;
    // Tracked robots need more yaw to break tread friction (matches the on-screen rotate buttons).
    const yaw = -dz(gp.axes[2] || 0) * (tracked ? 1.5 : 0.8);
    if(st) st.textContent = `${gp.id.slice(0,28)} · x${x.toFixed(2)} y${y.toFixed(2)} yaw${yaw.toFixed(2)}`;
    if(x || y || yaw) scoutHold(sid, x, y, yaw);
    else if(s.timer) scoutRelease(sid);
  } else if(st){
    st.textContent = 'press a button on the controller to connect it';
  }
  s.gpRAF = requestAnimationFrame(() => scoutGpLoop(sid));
}

// Listen: stream the robot mic (16 kHz mono PCM16 over a WebSocket) and play it back through Web
// Audio, scheduling each chunk after the last so it plays gaplessly. No secure context needed -
// only mic CAPTURE (talk) requires https/localhost; playback works on plain http.
async function scoutListenToggle(sid){
  const s = SC[sid]; if(!s) return;
  const btn = scEl(sid,'listenbtn');
  if(s.ws){ scoutListenStop(sid); return; }
  const tok = await tokenReady();
  s.ac = new (window.AudioContext || window.webkitAudioContext)();
  s.playAt = 0;
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  s.ws = new WebSocket(`${proto}://${location.host}/api/audio/scout/${sid}/listen?token=${encodeURIComponent(tok)}`);
  s.ws.binaryType = 'arraybuffer';
  s.ws.onopen = () => { btn.classList.add('on'); btn.textContent = '🔊 Listening'; };
  s.ws.onmessage = ev => {
    const pcm = new Int16Array(ev.data);
    if(!pcm.length || !s.ac) return;
    const buf = s.ac.createBuffer(1, pcm.length, 16000);
    const ch = buf.getChannelData(0);
    for(let i=0;i<pcm.length;i++) ch[i] = pcm[i] / 32768;
    const src = s.ac.createBufferSource();
    src.buffer = buf; src.connect(s.ac.destination);
    const now = s.ac.currentTime;
    // Keep a small lead; if we fall behind (tab throttled), resync rather than pile up latency.
    if(s.playAt < now + 0.02 || s.playAt > now + 0.5) s.playAt = now + 0.08;
    src.start(s.playAt);
    s.playAt += buf.duration;
  };
  s.ws.onclose = () => scoutListenStop(sid);
  s.ws.onerror = () => say('listen: connection failed', false);
}
function scoutListenStop(sid){
  const s = SC[sid]; if(!s) return;
  const btn = scEl(sid,'listenbtn');
  if(btn){ btn.classList.remove('on'); btn.textContent = '🔊 Listen'; }
  if(s.ws){ try{s.ws.close();}catch(e){} s.ws = null; }
  if(s.ac){ try{s.ac.close();}catch(e){} s.ac = null; }
}

// Talk: capture the browser mic, downsample to 16 kHz mono PCM16, stream it to the robot speaker
// while the button is held. getUserMedia needs a secure context (https/localhost), so on plain http
// this reports that instead of silently failing. This robot's Listen is paused while talking so its
// mic does not loop the speaker back - half-duplex push-to-talk, no on-robot echo cancellation.
async function scoutTalkStart(sid){
  const s = SC[sid]; if(!s) return;
  if(s.talkWS || s.talkNode) return;                   // already talking (button repeat)
  if(!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia){
    say('Talk needs https or localhost — the browser blocks mic capture on plain http. '
       + 'Open via an ssh -L localhost forward.', false);
    return;
  }
  const tok = await tokenReady();
  s.talkResume = !!s.ws;
  if(s.talkResume) scoutListenStop(sid);               // avoid feedback on THIS robot
  try{
    s.talkStream = await navigator.mediaDevices.getUserMedia(
      {audio: {channelCount: 1, echoCancellation: true, noiseSuppression: true}});
  }catch(e){ say('microphone permission denied', false); return; }
  s.talkCtx = new (window.AudioContext || window.webkitAudioContext)();
  const src = s.talkCtx.createMediaStreamSource(s.talkStream);
  s.talkNode = s.talkCtx.createScriptProcessor(4096, 1, 1);
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  s.talkWS = new WebSocket(`${proto}://${location.host}/api/audio/scout/${sid}/talk?token=${encodeURIComponent(tok)}`);
  s.talkWS.binaryType = 'arraybuffer';
  const ratio = s.talkCtx.sampleRate / 16000;
  s.talkNode.onaudioprocess = e => {
    if(!s.talkWS || s.talkWS.readyState !== 1) return;
    const inp = e.inputBuffer.getChannelData(0);
    const n = Math.floor(inp.length / ratio);
    const out = new Int16Array(n);
    for(let i=0;i<n;i++){ const v = inp[Math.floor(i*ratio)]; out[i] = Math.max(-32768, Math.min(32767, v*32768)); }
    s.talkWS.send(out.buffer);
  };
  // Route through a muted gain so the ScriptProcessor runs without playing the user's own mic back.
  const mute = s.talkCtx.createGain(); mute.gain.value = 0;
  src.connect(s.talkNode); s.talkNode.connect(mute); mute.connect(s.talkCtx.destination);
  const b = scEl(sid,'talkbtn'); if(b){ b.classList.add('on'); b.textContent = '🎙 Talking…'; }
}
function scoutTalkStop(sid){
  const s = SC[sid]; if(!s) return;
  const b = scEl(sid,'talkbtn'); if(b){ b.classList.remove('on'); b.textContent = '🎙 Hold to talk'; }
  if(s.talkNode){ try{s.talkNode.disconnect();}catch(e){} s.talkNode = null; }
  if(s.talkStream){ s.talkStream.getTracks().forEach(t=>t.stop()); s.talkStream = null; }
  if(s.talkCtx){ try{s.talkCtx.close();}catch(e){} s.talkCtx = null; }
  if(s.talkWS){ try{s.talkWS.close();}catch(e){} s.talkWS = null; }
  if(s.talkResume){ s.talkResume = false; scoutListenToggle(sid); }
}

// Look & describe: the endpoint returns {description, categories, alert} - it has no "message"
// field, so routing it through post() showed a bare green "done" and threw the caption away. This
// renders it into the alert box under the video and keeps it there.
async function scoutCheck(sid, btn){
  if(btn){ btn.classList.add('busy'); btn.textContent = 'Describing…'; }
  const el = scEl(sid,'alert');
  try{
    const r = await fetch(`/api/scout/${sid}/check`, {method:'POST', headers:{'X-Porch-Token': await tokenReady()}});
    const j = await r.json().catch(()=>({}));
    if(!r.ok){ say(j.detail || j.message || 'describe failed', false); return; }
    const when = new Date((j.at ? j.at*1000 : Date.now())).toLocaleTimeString();
    el.className = 'ralert ' + (j.alert ? 'hit' : 'clear');
    el.innerHTML = `<b>${when}</b> — ${esc(j.description || '(no description)')}`
      + (j.categories && j.categories.length ? ` <span class="hint">[${esc(j.categories.join(', '))}]</span>` : '')
      + (j.tof_m != null ? ` <span class="hint">· range ${(+j.tof_m).toFixed(2)} m</span>` : '');
  }catch(e){ say(String(e), false); }
  finally{ if(btn){ btn.classList.remove('busy'); btn.textContent = 'Look & describe'; } }
}

// Restart the robot's ROS stack so the official Moorebot app can take over. Disruptive - our own
// video and drive pause while roscore restarts - so it confirms first, and stops any motion we own
// before bouncing the stack. The card recovers on its own via the bridge watchdogs.
async function scoutRestartRos(sid){
  const s = SC[sid]; if(!s) return;
  const name = (s.meta && s.meta.name) || sid;
  if(!confirm(`Restart the ROS stack on "${name}"?\n\n`
    + `This hands control back to the official Moorebot app. Live video and drive on this card `
    + `pause for ~20 s while the robot's ROS restarts, then recover on their own.`)) return;
  scoutRelease(sid);                 // stop anything we are driving before the bounce
  scoutListenStop(sid);              // its mic/audio pipes die with the stack anyway
  await post(`/api/scout/${sid}/restart-ros`);
}

async function scoutSnapshot(sid){
  try{
    const r = await fetch(`/scout/${sid}/latest.jpg?t=` + Date.now(), {cache:'no-store'});
    if(!r.ok){ say('no frame', false); return; }
    const b = await r.blob(), u = URL.createObjectURL(b), a = document.createElement('a');
    a.href = u; a.download = `scout-${sid}-` + new Date().toISOString().replace(/[:.]/g,'-') + '.jpg';
    a.click(); URL.revokeObjectURL(u); say('snapshot saved', true);
  }catch(e){ say(String(e), false); }
}

// quiet=true reports only failures. Drive and stop fire many times a second while a control is
// held; without this they would spam the feedback banner and (before it was moved) reflow the page.
async function postJSON(url, body, quiet){
  try{
    const r = await fetch(url, {method:'POST',
      headers:{'Content-Type':'application/json','X-Porch-Token': await tokenReady()},
      body: JSON.stringify(body)});
    if(!r.ok){ const j = await r.json().catch(()=>({})); say(j.detail || j.message || 'failed', false); }
    else if(!quiet){ const j = await r.json().catch(()=>({})); say(j.message || 'done', true); }
  }catch(e){ if(!quiet) say(String(e), false); }
}

// Poll one robot's state and update its card. Same body as the single-robot version, scoped to sid.
async function refreshScout(sid){
  const s = SC[sid]; if(!s) return;
  const card = scEl(sid,'card');
  if(!card) return;
  try{
    const d = await (await fetch(`/api/scout/${sid}/state`,{cache:'no-store'})).json();
    if(!d.enabled){ card.style.display='none'; return; }
    card.style.display='block';
    const img  = scEl(sid,'img');
    const hint = scEl(sid,'hint');
    const st   = scEl(sid,'state');

    if(d.live){
      // Status strip, app-style: battery + charge state, then range. The charge state is spelled
      // out and coloured on its own span (inline colour wins over the line's ToF colour) so it is
      // unambiguous whether the robot is actually charging - green ⚡ when it is, red 🔻 when it is
      // low and running the battery down.
      let batt = '';
      if(d.battery_fresh && d.battery_pct != null){
        const bs = d.battery_state;
        const label = bs === 'charging' ? '⚡ charging'
                    : bs === 'full'     ? '🔋 full'
                    : bs === 'discharging' ? '🔻 on battery' : bs || '';
        const col = (bs === 'charging' || bs === 'full') ? 'var(--g)'
                  : (d.battery_pct < 20 ? 'var(--r)' : (d.battery_pct < 40 ? 'var(--y)' : 'var(--fg)'));
        batt = `<span class="batt" style="color:${col}">${d.battery_pct}% ${label}</span> · `;
      } else if(d.reachable) {
        batt = `<span class="batt hint">battery —</span> · `;
      }
      let range;
      if(!d.tof_fresh)        range = 'range —';
      else if(d.tof_m == null) range = 'range >2 m clear';
      else                     range = `range ${d.tof_m.toFixed(2)} m`;
      st.innerHTML = `<span class="batt">${batt}</span>● live · ${range} · ${d.frames} frames`
                   + (d.driving ? ' · driving' : '');
      // Amber inside a body-length, red when it is about to touch something.
      st.style.color = (d.tof_fresh && d.tof_m != null)
        ? (d.tof_m < 0.20 ? 'var(--r)' : d.tof_m < 0.50 ? 'var(--y)' : '') : '';
      // Point at the live MJPEG stream ONCE, not a fresh still every poll. An MJPEG <img> holds a
      // persistent connection; if it dies (e.g. porch-feed restarts) the browser keeps the dead
      // socket occupying one of its ~6 per-host slots, and enough of those stall every fetch on
      // the page - which is how the controls once appeared "broken". onerror reconnects with a
      // cache-bust so a broken stream frees its slot and reopens instead of lingering.
      if(!img.dataset.streaming){
        img.dataset.streaming = '1';
        img.onerror = () => { if(img.dataset.streaming) setTimeout(() => {
          img.src = `/scout/${sid}/mjpeg?t=` + Date.now(); }, 800); };
        img.src = `/scout/${sid}/mjpeg`;
      }
      img.style.display='block'; hint.style.display='none';
    } else {
      img.style.display='none'; hint.style.display='block';
      // Drop the dead stream so the next live poll reconnects cleanly, and free its connection slot.
      if(img.dataset.streaming){ img.onerror = null; delete img.dataset.streaming; img.removeAttribute('src'); }
      // Distinguish the three ways this goes quiet, because they need different fixes.
      if(!d.reachable){
        st.textContent = '○ bridge not running';
        hint.innerHTML = `Start it with <code>docker compose --profile scout up -d</code> in <code>/home/orin/nvr</code>.`;
      } else if(!d.ros_connected){
        st.textContent = '○ bridge up, robot not reachable';
        hint.textContent = d.error || 'The bridge cannot reach the robot’s ROS master.';
      } else {
        st.textContent = '○ connected, no frames';
        hint.textContent = d.blocked_by ? `Camera held by ${d.blocked_by}.` : (d.error || 'No frames yet.');
      }
    }
  }catch(e){ /* leave the card as it was */ }
}
function refreshScouts(){ for(const sid in SC) refreshScout(sid); }

// Installed robot apps. Rendered as one button each rather than a dropdown so the running one can
// show as pressed - which app has the robot is the thing you actually want to see at a glance.
async function refreshApps(){
  const box = document.getElementById('reachyApps');
  if(!box) return;
  try{
    const a = await (await fetch('/api/reachy/apps',{cache:'no-store'})).json();
    const now = document.getElementById('reachyAppNow');
    // An app that died reports state "error" with a Python traceback. Showing the first line beats
    // "nothing running", which is what this said while an app was crash-looping.
    now.textContent = a.error ? ('✕ ' + a.running + ': ' + a.error.split('\n')[0].slice(0,90))
                    : a.running ? ('running: ' + a.running + (a.state && a.state!=='running' ? ' ('+a.state+')' : ''))
                    : 'nothing running';
    now.style.color = a.error ? 'var(--r)' : '';
    box.innerHTML = (a.installed||[]).map(app=>{
      const on = app.name === a.running;
      // The app's own settings page - where a conversation app takes its API key and persona.
      const link = (on && app.url) ? ` <a href="${esc(app.url)}" target="_blank" rel="noreferrer">↗</a>` : '';
      const href = '/api/reachy/apps/start/' + encodeURIComponent(app.name);
      return `<button class="${on?'on':''}" onclick="post('${href}')">${esc(app.name)}</button>${link}`;
    }).join('') + (a.running
      ? `<button class="warn" onclick="post('/api/reachy/apps/stop')">Stop app</button>` : '');
  }catch(e){ /* the panel keeps whatever it last showed */ }
}

function initCtl(){
  bindSlider('antL','antLv',null,' rad',0);
  bindSlider('antR','antRv',null,' rad',1);
  bindSlider('roll','rollv','roll',' rad');
  bindSlider('byaw','byawv','body_yaw',' rad');
  bindSlider('posZ','zv','z','');
  bindPad('padXY','xyv','x','y',0.02,0.02,false);
  // Pitch is positive downward, so the pad is inverted: drag up, look up.
  bindPad('padPY','pyv','yaw','pitch',Math.PI,0.7,true);
}

async function rq(url){ return post(url); }
async function post(url){
  // tokenReady(), not TOKEN: the token only arrives with the first status poll, so a click during
  // the first second used to send an empty header and fail with a bare 401.
  try{ const r=await fetch(url,{method:'POST',headers:{'X-Porch-Token':await tokenReady()}});
       const j=await r.json().catch(()=>({}));
       say(j.message || j.detail || (r.ok?'done':'failed'), r.ok); }
  catch(e){ say(String(e), false); }
  await load();
}
async function load(){
  const st = await (await fetch('/api/status',{cache:'no-store'})).json();
  TOKEN = st.control_token || '';
  document.getElementById('mem').textContent =
    `${st.memory.free_mb} MB free · ${st.memory.used_pct}% used`;

  document.getElementById('engines').innerHTML = Object.entries(st.engines).map(([id,e])=>
    `<button class="${st.active_engine.id===id?'on':''} ${e.built?'':'locked'}"
      ${e.built?`onclick="post('/api/engine/${id}')"`:''} title="${esc(e.notes)}">
      ${esc(e.name)} <span class="hint">(${id})</span></button>`).join('');
  const ae = st.engines[st.active_engine.id];
  document.getElementById('enginehint').textContent =
    ae ? `${st.active_engine.name} — ${ae.profile}. ${ae.notes}` : '';

  document.getElementById('services').innerHTML = Object.values(st.services).map(s=>{
    // health: up | starting | stuck | down. `starting` and `stuck` are the cases where systemd and
    // the port disagree, which is exactly what the port poll exists to surface.
    const dot = {up:'🟢', starting:'🟡', stuck:'🟠', down:'⚪'}[s.health] || '⚪';
    const portTxt = s.port
      ? `:${s.port} ${s.listening ? 'listening' : 'closed'}`
      : 'no port to poll';
    const confirm = s.self
      ? `if(!window.confirm('This stops the page you are using. You will need SSH to start it again. Continue?'))return;`
      : '';
    // "Stopped by you" is the important state to surface: it is the one that must not be
    // quietly undone by anybody - human or automation - without asking first.
    const when = s.intent_at ? new Date(s.intent_at*1000).toLocaleString([], {month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'}) : '';
    let intentBadge = '';
    if (s.intent === 'stopped') {
      intentBadge = `<div class="intent off">⏸ stopped by ${esc(s.intent_by||'user')} · ${when}`
                  + (s.contradicts_intent ? ' — <b>but it is running again</b>' : '')
                  + `${s.intent_note ? ' · '+esc(s.intent_note) : ''}</div>`;
    } else if (s.intent === 'running' && s.contradicts_intent) {
      intentBadge = `<div class="intent off">▶ started by ${esc(s.intent_by||'user')} · ${when} — <b>but it is down</b></div>`;
    }
    // Boot state, only for services whose ON/OFF here also enables/disables them at boot. A
    // mismatch with the recorded intent is the case this exists for: the next reboot would
    // quietly undo the choice (enable/disable failed, or someone changed it over SSH).
    let boot = '';
    if (s.boot_follows_intent) {
      boot = s.boot_enabled === true ? 'starts at boot'
           : s.boot_enabled === false ? 'off at boot' : 'boot: ' + (s.boot_state || 'unknown');
      if (s.intent && s.boot_enabled != null && (s.intent === 'running') !== s.boot_enabled)
        boot += s.boot_enabled ? ' — a reboot would bring it back' : ' — a reboot would leave it off';
      boot = ' · ' + esc(boot);
    }
    return `<div class="svc${s.intent==='stopped'?' held':''}">
      <div class="svc-id">
        <div class="svc-name">${dot} ${esc(s.label)}</div>
        <div class="hint">${esc(s.unit||'')} · ${s.state} · ${portTxt}${boot}${s.note?' · '+esc(s.note):''}</div>
        ${intentBadge}
      </div>
      <div class="row">
        <button class="${s.running?'on':''}" ${s.running?'disabled':''}
                onclick="${s.intent==='stopped'?`if(!window.confirm('${esc(s.label)} was stopped on purpose. Turn it back on?'))return;`:''}post('/api/service/${s.key}/start')">ON</button>
        <button class="${!s.running?'off':'warn'}" ${!s.running?'disabled':''}
                onclick="${confirm}post('/api/service/${s.key}/stop')">OFF</button>
        <button ${!s.running?'disabled title="not running - use ON"':''}
                onclick="${confirm}post('/api/service/${s.key}/restart')">RESTART</button>
      </div></div>`;}).join('');

  document.getElementById('cameras').innerHTML = Object.entries(st.cameras).map(([n,c])=>{
    const on = c.mode==='powered';
    if(c.always_powered) return `<button class="on locked" title="configured always-powered">
      ${esc(n)}: POWERED 🔒</button>`;
    return `<button class="${on?'on':'warn'}" onclick="post('/api/camera/${n}/${on?'saver':'powered'}')">
      ${esc(n)}: ${on?'POWERED':'SAVER'}</button>`;}).join('');

  try{
    const lk = await (await fetch('/api/links',{cache:'no-store'})).json();
    document.getElementById('links').innerHTML = lk.length ? lk.map(l=>{
      const dot = l.up ? '🟢' : '⚪';
      if(!l.browsable) return `<span class="chip locked" title="not a browsable web UI">${dot} ${esc(l.name)} <span class="hint">${esc(l.url)}</span></span>`;
      return `<a class="chip ${l.up?'on':'down'}" href="${esc(l.url)}" target="_blank" rel="noreferrer"
                 title="${l.up?'open':'not responding — link still shown'} ${esc(l.url)}">${dot} ${esc(l.name)}
              <span class="hint">${esc(l.url)}</span></a>`;
    }).join('') : '<span class="hint">probing…</span>';
  }catch(e){}

  try{
    const rc = await (await fetch('/api/reachy',{cache:'no-store'})).json();
    const img = document.getElementById('reachyImg');
    const st  = document.getElementById('reachyState');
    const hint= document.getElementById('reachyHint');
    document.getElementById('reachyLink').href = rc.link || '#';
    const lv = document.getElementById('reachyVisionLink');
    lv.href = rc.live_vision_link || '#';
    lv.style.display = rc.live_vision_link ? '' : 'none';
    // The retry streak goes in the tooltip, not the line: the bridge already writes it into
    // `reason` once it gets long enough to mean the robot needs a power-cycle.
    st.title = rc.state ? `bridge ${rc.state}` + (rc.failed_streak ? ` · ${rc.failed_streak} `
                          + `session(s) in a row without video` : '') : '';
    if(rc.connected){
      const a = rc.audio, p = rc.push;
      st.textContent = `● live · ${rc.frames} frames · via ${rc.source||'bridge'}`
        + (a ? (a.live ? ' · 🎙 mic live' + (a.listeners ? ` (${a.listeners} listening)` : '')
                       : ' · 🎙 no mic audio') : '')
        // "idle" is a bridge started without --push-url; nothing to report.
        + (p && p.state && p.state !== 'idle' ? ` · WebUI push: ${p.state}` : '');
      st.style.color = 'var(--g)';
      img.style.display = 'block'; hint.style.display = 'none';
      // Only fetch the still while frames are actually arriving; otherwise this proxies a 404
      // every second for no reason.
      img.src = `/reachy/latest.jpg?t=${Date.now()}`;
    } else {
      // Say WHY it is not live. "Camera held by the robot app", "daemon unreachable" and "bridge
      // not running" each need a different fix, and none of them is "wait longer".
      st.textContent = !rc.configured ? '○ not configured'
        : rc.reason ? `○ ${rc.state || 'not live'} — ${rc.reason}`
        // Between FRESH_FOR (5 s) and the bridge's stall check (8 s) it can still say "live".
        : rc.state === 'live' ? '○ live, waiting for a fresh frame…'
        : rc.state  ? `○ ${rc.state}…`
        : '○ no frames yet — is reachy-mjpeg-bridge.service running?';
      st.style.color = (rc.state === 'error' || rc.state === 'unreachable') ? 'var(--r)'
                     : rc.state === 'dormant' ? 'var(--y)' : 'var(--mut)';
      img.style.display = 'none'; img.removeAttribute('src');
      hint.style.display = 'block';
    }
  }catch(e){}

  try{
    const rs = await (await fetch('/api/reachy/state',{cache:'no-store'})).json();
    const box = document.getElementById('reachyCtl');
    if(rs.enabled && rs.reachable){
      box.style.display='block';
      const p = rs.pose_deg||{};
      document.getElementById('reachyStatus2').textContent =
        `${rs.awake?'● awake':'○ asleep'} · motors ${rs.motor_mode||'?'} · `
        + `yaw ${p.yaw??'—'}° pitch ${p.pitch??'—'}° body ${rs.body_yaw_deg??'—'}°`
        + (rs.speech_detected?' · 🔊 hearing speech':'')
        + (rs.doa_deg!=null?` · voice at ${rs.doa_deg}°`:'');

      // What is using the robot right now. control_hz is the honest load signal: the 50 Hz loop is
      // the highest-priority work on the robot's CM4, so it sags when something is eating the CPU.
      // Measured on this robot: ~46 Hz idle, ~49 with a thin conversation app, ~31 with a heavy one.
      const busy = document.getElementById('reachyBusy');
      const hz = rs.control_hz;
      const bits = [];
      if(rs.lock_holder) bits.push(`🔒 ${rs.lock_holder} holds the robot — it owns the camera and mic`);
      else bits.push('🔓 no app holding the robot');
      if(hz!=null) bits.push(`control loop ${hz} Hz${hz<40?' (loaded)':''}`);
      if(rs.control_errors) bits.push(`${rs.control_errors} loop errors`);
      if(rs.move_running) bits.push('a move is playing');
      busy.textContent = bits.join(' · ');
      busy.style.color = (hz!=null && hz<35) ? 'var(--y)' : '';
      const setv=(id,v,lbl)=>{const e=document.getElementById(id);
        if(e&&document.activeElement!==e&&v!=null){e.value=v;document.getElementById(lbl).textContent=v+'%';}};
      setv('micVol', rs.mic_volume, 'micVal');
      setv('spkVol', rs.speaker_volume, 'spkVal');
      syncCtl(rs);
      refreshApps();
    } else { box.style.display = rs.enabled ? 'block' : 'none';
             if(rs.enabled) document.getElementById('reachyStatus2').textContent='robot unreachable'; }

    const al = await (await fetch('/api/reachy/alert',{cache:'no-store'})).json();
    const el = document.getElementById('reachyAlert');
    if(al.at){
      const when = new Date(al.at*1000).toLocaleTimeString();
      const hit = !!al.alert;
      el.className = 'ralert ' + (hit?'hit':'clear');
      el.innerHTML = (hit?'⚠️ <b>'+esc(al.headline||'anomaly')+'</b><br>':'')
                   + esc(al.description||'') + ` <span class="hint">(${esc(al.trigger||'')} · ${when})</span>`;
    } else { el.className='ralert'; }
    document.getElementById('reachyWatchHint').textContent = al.watching
      ? `Watching: a look is spent when the robot hears speech, or Frigate detects `
        + `${(al.triggers&&al.triggers.labels||[]).join('/')} on ${(al.triggers&&al.triggers.cameras||['any']).join(', ')}. `
        + `At most once per ${al.cooldown_s}s — ${al.checked||0} checks so far.`
      : 'Anomaly watch is off.';
  }catch(e){}

  const cmp = await (await fetch('/api/compare',{cache:'no-store'})).json();
  document.querySelector('#cmp tbody').innerHTML = cmp.length ? cmp.map(r=>
    `<tr><td><span class="tag e-${r.engine_id}">${esc(r.engine_name)}</span></td><td>${r.n}</td><td>${r.measured||0}</td>
     <td>${r.lat!=null?Math.round(r.lat)+' ms':'—'}</td><td>${r.cpu??'—'}${r.cpu!=null?'%':''}</td><td>${r.mem??'—'}${r.mem!=null?'%':''}</td><td>${r.gpu??'—'}${r.gpu!=null?'%':''}</td>
     <td>${r.desc_len!=null?Math.round(r.desc_len)+' ch':'—'}</td></tr>`).join('')
    : `<tr><td colspan="8" class="hint">No captions recorded yet.</td></tr>`;

  const ids = ['all', ...Object.keys(st.engines)];
  document.getElementById('filters').innerHTML = ids.map(i=>
    `<button class="${FILTER===i?'on':''}" onclick="FILTER='${i}';load()">${i==='all'?'All engines':esc(st.engines[i].name)}</button>`).join('');

  const ev = await (await fetch(`/api/entries?limit=60&engine=${FILTER}`,{cache:'no-store'})).json();
  document.getElementById('feed').innerHTML = ev.length ? ev.map(e=>
    `<div class="card" id="${esc(e.id)}">
       <div class="row" style="justify-content:space-between">
         <span class="tag e-${e.engine_id}">${esc(e.engine_name)}</span>
         <span class="hint">${fmt(e.ts)}</span></div>
       <img class="still" loading="lazy" src="/img/${encodeURIComponent(e.id)}.jpg" alt=""
            onerror="this.remove()"/>
       <p class="desc">${esc(e.description)}</p>
       <div class="meta"><span>${esc(e.camera)} · ${esc(e.label)}</span>
         <span title="end-to-end Frigate pipeline, not model inference time">${e.latency_ms} ms e2e</span><span>CPU ${e.peak_cpu??'—'}${e.peak_cpu!=null?'%':''}</span>
         <span>VRAM ${e.peak_mem??'—'}${e.peak_mem!=null?'%':''}</span><span>GPU ${e.peak_gpu??'—'}${e.peak_gpu!=null?'%':''}</span></div>
     </div>`).join('')
    : `<p class="hint">No captions yet for this filter. Trigger motion on a camera.</p>`;
}
// The Scouts refresh on their own timer: their cameras are worth seeing at a higher rate than the
// 10 s whole-page poll, and they must keep updating even when the robot sections are hidden. Cards
// are built once from /api/scouts, then each polls on the shared 3 s tick.
showSelfUrl(); initCtl(); load(); setInterval(load, 10000);
buildScoutCards().then(() => { refreshScouts(); setInterval(refreshScouts, 3000); });
</script></body></html>"""


def main() -> None:
    init_db()
    threading.Thread(target=mqtt_loop, daemon=True).start()
    threading.Thread(target=link_poller, daemon=True).start()
    threading.Thread(target=reachy_watcher, daemon=True).start()
    print(f"[feed] engine={active_engine()['id']} port={CFG.get('web_port', 8096)}", flush=True)
    uvicorn.run(app, host=CFG.get("web_host", "0.0.0.0"),
                port=int(CFG.get("web_port", 8096)),
                log_level=str(CFG.get("web_log_level", "warning")), access_log=True)


if __name__ == "__main__":
    main()
