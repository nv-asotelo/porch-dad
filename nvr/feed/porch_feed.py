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

import json
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

import requests
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

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
LINKS = CFG.get("links") or []
LINKS_HOST = CFG.get("links_host") or "127.0.0.1"
REACHY_WEBUI = str(CFG.get("reachy_webui_url") or "").rstrip("/")
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

    return {
        "key": key,
        "label": s.get("label", key),
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


def set_service(key: str, action: str) -> tuple[bool, str]:
    """Run a control action and then *verify* it, rather than trusting the exit code.

    Verification is the whole point: it confirms the port reached the expected state, and reports
    how much memory actually came back. On an 8 GB board a "successful" stop that frees nothing is
    a failure worth seeing.
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
            return False, (f"{label}: {s['kind']} reported stopped, but port {port} is still "
                           f"accepting connections — something is still holding it. "
                           f"Memory is at {after} MB free.")
        freed = f"reclaimed {delta} MB" if delta > 0 else "no memory reclaimed"
        expect = s.get("heavy_mb")
        warn = ""
        if expect and delta < expect * 0.5:
            warn = (f" — expected about {expect} MB back; the kernel may still be releasing "
                    f"page cache, re-check the memory readout in a few seconds")
        return True, f"{label} stopped, port {port or '—'} closed, {freed} ({after} MB free){warn}"

    if not reached:
        return False, (f"{label}: {action} issued but port {port} never started listening. "
                       f"Check `journalctl -u {s.get('unit') or s.get('name')}`.")
    cost = f"used {-delta} MB" if delta < 0 else f"memory unchanged ({delta:+d} MB)"
    return True, f"{label} {action}ed, port {port or '—'} listening, {cost} ({after} MB free)"


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
        if data.get("type") != "end":
            return
        threading.Thread(target=handle_event, args=(data.get("after") or {},), daemon=True).start()

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
    if not REACHY_WEBUI:
        raise HTTPException(404, "reachy_webui_url is not configured")
    try:
        r = requests.get(f"{REACHY_WEBUI}/api/push/latest.jpg",
                         params={"session_id": REACHY_SESSION}, timeout=5, verify=False)
    except requests.RequestException as e:
        raise HTTPException(502, f"Live VLM WebUI unreachable: {e}")
    if r.status_code != 200 or not r.content:
        raise HTTPException(404, "no frame available yet")
    return Response(r.content, media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.get("/api/reachy")
def api_reachy():
    """Whether the Reachy Mini push source is currently delivering frames."""
    out = {"configured": bool(REACHY_WEBUI), "session": REACHY_SESSION,
           "connected": False, "fps": None, "frames": 0, "resolution": None,
           "link": f"{REACHY_WEBUI}/?session={REACHY_SESSION}" if REACHY_WEBUI else None}
    if not REACHY_WEBUI:
        return out
    try:
        r = requests.get(f"{REACHY_WEBUI}/api/push/status", timeout=4, verify=False)
        for st in (r.json().get("streams") or []):
            if st.get("session_id") == REACHY_SESSION:
                out.update(connected=bool(st.get("connected")), fps=st.get("fps"),
                           frames=st.get("frames_received", 0),
                           resolution=(f"{st.get('width')}\u00d7{st.get('height')}"
                                       if st.get("width") else None))
                break
    except (requests.RequestException, ValueError):
        pass
    return out


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
  <div id="msg" class="msg"></div>

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

  <h2>Reachy Mini camera</h2>
  <div class="card" id="reachyCard">
    <div class="row" style="justify-content:space-between;align-items:center">
      <span id="reachyState" class="hint">checking…</span>
      <a id="reachyLink" class="chip" href="#" target="_blank" rel="noreferrer">Open in Live VLM WebUI</a>
    </div>
    <img id="reachyImg" class="still" alt="Reachy Mini camera" style="display:none">
    <p class="hint" id="reachyHint" style="display:none">
      No frames on this session. The push client feeds it — see
      <code>examples/push_reachy_mini.py</code> in the live-vlm-webui fork.
    </p>
  </div>

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

  <h2>Feed <span id="filter" class="hint"></span></h2>
  <div class="row" id="filters"></div>
  <div id="feed" style="margin-top:10px"></div>
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
async function post(url){
  try{ const r=await fetch(url,{method:'POST',headers:{'X-Porch-Token':TOKEN}});
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
    return `<div class="svc">
      <div class="svc-id">
        <div class="svc-name">${dot} ${esc(s.label)}</div>
        <div class="hint">${esc(s.unit||'')} · ${s.state} · ${portTxt}${s.note?' · '+esc(s.note):''}</div>
      </div>
      <div class="row">
        <button class="${s.running?'on':''}" ${s.running?'disabled':''}
                onclick="post('/api/service/${s.key}/start')">ON</button>
        <button class="${!s.running?'off':'warn'}" ${!s.running?'disabled':''}
                onclick="${confirm}post('/api/service/${s.key}/stop')">OFF</button>
        <button onclick="${confirm}post('/api/service/${s.key}/restart')">RESTART</button>
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
    if(rc.connected){
      st.textContent = `● live · ${rc.resolution||'—'} · ${rc.fps||'—'} fps · ${rc.frames} frames`;
      st.style.color = 'var(--g)';
      img.style.display = 'block'; hint.style.display = 'none';
      // Only fetch the still while frames are actually arriving; otherwise this proxies a 404
      // every second for no reason.
      img.src = `/reachy/latest.jpg?t=${Date.now()}`;
    } else {
      st.textContent = rc.configured ? '○ no frames on session "'+rc.session+'"' : '○ not configured';
      st.style.color = 'var(--mut)';
      img.style.display = 'none'; img.removeAttribute('src');
      hint.style.display = 'block';
    }
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
showSelfUrl(); load(); setInterval(load, 10000);
</script></body></html>"""


def main() -> None:
    init_db()
    threading.Thread(target=mqtt_loop, daemon=True).start()
    threading.Thread(target=link_poller, daemon=True).start()
    print(f"[feed] engine={active_engine()['id']} port={CFG.get('web_port', 8096)}", flush=True)
    uvicorn.run(app, host=CFG.get("web_host", "0.0.0.0"),
                port=int(CFG.get("web_port", 8096)), log_level="warning")


if __name__ == "__main__":
    main()
