#!/usr/bin/env python3
"""porch-dad — Frigate events -> Cosmos3-Edge scene description -> mobile web feed.

Flow
    Frigate --(MQTT frigate/events, type=end)--> this service
        -> download the event clip from the Frigate API
        -> sample N frames evenly with ffmpeg (N is capped by the engine's image-token budget)
        -> POST to the Cosmos3-Edge shim with a system + user turn
        -> persist to SQLite, republish on MQTT, serve a phone-friendly feed

Engine constraint that drives the design
    The visual engine is built with max_image_tokens=1024 and ~302 prompt tokens are consumed per
    frame at the current 320-token per-image cap. Three frames fit (933 tokens); a fourth is
    rejected by the runtime with "Failed to handle generation request". NUM_FRAMES is therefore
    clamped to MAX_FRAMES. Raising it means rebuilding the visual engine with a larger
    max_image_tokens, not just editing this file.
"""
from __future__ import annotations

import base64
import json
import os
import re
import sqlite3
import subprocess
import tempfile
import threading
import time
from contextlib import closing
from pathlib import Path

import requests
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response

CFG_PATH = os.environ.get("PORCH_DAD_CONFIG", "/home/orin/nvr/bridge/config.yaml")
CFG = yaml.safe_load(Path(CFG_PATH).read_text())

FRIGATE = CFG["frigate_url"].rstrip("/")
SHIM = CFG["cosmos3_url"].rstrip("/")
DB_PATH = CFG["db_path"]
MAX_FRAMES = int(CFG.get("max_frames", 3))
NUM_FRAMES = min(int(CFG.get("num_frames", 3)), MAX_FRAMES)
MAX_TOKENS = int(CFG.get("max_tokens", 512))
FRAME_WIDTH = int(CFG.get("frame_width", 640))
MIN_CLIP_WAIT = float(CFG.get("clip_wait_seconds", 3.0))
# JetPack ships a stripped NVIDIA ffmpeg (no scale filter, no software encoders). Point these at
# a full build; see nvr/README.md.
FFMPEG = CFG.get("ffmpeg", "ffmpeg")
FFPROBE = CFG.get("ffprobe", "ffprobe")
CAMERA_LABELS = CFG.get("camera_labels") or {}

# Ring event source. Ring's *live* stream is an on-demand WebRTC session that ring-mqtt transcodes
# to RTSP; it cannot be consumed as a continuous NVR feed (ffmpeg reports "Invalid data found when
# processing input" even while ring-mqtt reports the WebRTC session connected), and holding it open
# would drain battery cameras. Ring is event-driven by design, so we consume motion/ding events and
# the JPEG snapshots ring-mqtt publishes instead. This works for every camera, wired or battery.
RING_ENABLED = bool(CFG.get("ring_enabled", True))
RING_CAMERAS = CFG.get("ring_cameras") or {}          # device id -> friendly name
RING_WINDOW = float(CFG.get("ring_snapshot_window", 8.0))
RING_COOLDOWN = float(CFG.get("ring_cooldown", 45.0))  # per-camera, avoids event storms
SYSTEM_PROMPT = CFG["system_prompt"]
USER_PROMPT_TMPL = CFG["user_prompt"]

_db_lock = threading.Lock()


# --------------------------------------------------------------------------- storage
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with closing(db()) as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id           TEXT PRIMARY KEY,
                camera       TEXT,
                label        TEXT,
                start_time   REAL,
                end_time     REAL,
                category     TEXT,
                description  TEXT,
                raw          TEXT,
                num_frames   INTEGER,
                latency_ms   INTEGER,
                created_at   REAL
            )""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_events_start ON events(start_time DESC)")
        c.commit()


def save_event(row: dict) -> None:
    with _db_lock, closing(db()) as c:
        c.execute("""
            INSERT OR REPLACE INTO events
            (id,camera,label,start_time,end_time,category,description,raw,num_frames,latency_ms,created_at)
            VALUES (:id,:camera,:label,:start_time,:end_time,:category,:description,:raw,
                    :num_frames,:latency_ms,:created_at)""", row)
        c.commit()


# --------------------------------------------------------------------------- frames
def fetch_clip(event_id: str, dest: str) -> bool:
    """Frigate finalizes the clip a moment after the event ends; retry briefly."""
    url = f"{FRIGATE}/api/events/{event_id}/clip.mp4"
    for attempt in range(6):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200 and len(r.content) > 1024:
                Path(dest).write_bytes(r.content)
                return True
        except requests.RequestException:
            pass
        time.sleep(MIN_CLIP_WAIT * (attempt + 1) / 2)
    return False


def clip_duration(path: str) -> float:
    try:
        out = subprocess.run(
            [FFPROBE, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path],
            capture_output=True, text=True, timeout=30)
        return float(out.stdout.strip())
    except Exception:
        return 0.0


def sample_frames(clip: str, n: int) -> list[bytes]:
    """Evenly spaced frames, scaled so each lands within the per-image token budget."""
    dur = clip_duration(clip)
    if dur <= 0:
        fracs = [0.0] * n
    else:
        # interior points: avoids the black/partial frames common at clip boundaries
        fracs = [dur * (i + 1) / (n + 1) for i in range(n)]
    frames: list[bytes] = []
    with tempfile.TemporaryDirectory() as td:
        for i, ts in enumerate(fracs):
            out = os.path.join(td, f"f{i}.jpg")
            cmd = [FFMPEG, "-v", "error", "-y"]
            if dur > 0:
                cmd += ["-ss", f"{ts:.3f}"]
            cmd += ["-i", clip, "-frames:v", "1", "-q:v", "3",
                    "-vf", f"scale={FRAME_WIDTH}:-2", out]
            try:
                subprocess.run(cmd, check=True, capture_output=True, timeout=60)
                data = Path(out).read_bytes()
                if data:
                    frames.append(data)
            except Exception as e:
                print(f"[porch-dad] frame {i} extract failed: {e}", flush=True)
    return frames


# --------------------------------------------------------------------------- model
CATEGORY_RE = re.compile(r"\[(ALERT|ROUTINE)\]", re.I)

# The alert criteria from the monitoring policy, as deterministic patterns.
#
# Why rules and not a second model pass: a text-only classification call is cheap (~100ms) but was
# measured to be badly unreliable on exactly the cases that matter - it labelled "smoke is rising
# from the garage" and "hooded figure with face covered, looking around repeatedly" both ROUTINE,
# and emitted non-answers ("A\nROUT", an emoji) on others. False negatives on smoke and on a
# concealed face are the worst failures this system can produce, so the policy is encoded
# explicitly instead. Rules are auditable and cannot drift.
ALERT_PATTERNS = [
    (re.compile(r"\b(loiter\w*|lingering|pacing|waiting around|standing (?:around|there) for)", re.I), "loitering"),
    (re.compile(r"\b(mask(?:ed|ing)?|balaclava|hood(?:ed|ie)?\b.*\b(?:face|cover)|face (?:is )?(?:cover\w*|conceal\w*|obscur\w*|hidden))", re.I), "face obscured"),
    (re.compile(r"\b(cover\w*|conceal\w*|obscur\w*|hiding)\b.{0,24}\bface", re.I), "face obscured"),
    (re.compile(r"\b(climb\w*|jump\w*|vault\w*)\b.{0,30}\b(fence|wall|gate|window|railing)", re.I), "climbing"),
    (re.compile(r"\b(pry\w*|forc\w*|break\w*|smash\w*|kick\w*)\b.{0,30}\b(door|window|lock|gate)", re.I), "forced entry"),
    (re.compile(r"\b(smoke|fire|flames?|burning|blaze)\b", re.I), "smoke or fire"),
    (re.compile(r"\b(backyard|back yard|rear yard)\b.{0,40}\b(enter\w*|inside|climb\w*|access\w*)", re.I), "yard entry"),
]

NO_ACTIVITY_RE = re.compile(r"no activity detected", re.I)


def categorize(text: str) -> tuple[str, str]:
    """Return (category, reason).

    Order matters. An explicit tag from the model is honoured first; otherwise the policy rules
    decide. A reply that is empty or unintelligible becomes UNSCORED rather than ROUTINE, because
    "we could not tell" must never be presented as "nothing is wrong".
    """
    t = (text or "").strip()
    if not t:
        return "UNSCORED", "empty response"

    m = CATEGORY_RE.search(t)
    if m:
        return m.group(1).upper(), "tagged by model"

    for pat, reason in ALERT_PATTERNS:
        if pat.search(t):
            return "ALERT", reason

    if NO_ACTIVITY_RE.search(t):
        return "ROUTINE", "no activity"

    # Something was described and nothing matched an alert criterion. Per the policy a person
    # simply being present is not an anomaly, so this is routine.
    return "ROUTINE", "no alert criteria matched"


def describe(frames: list[bytes], camera: str, stream_id: str) -> tuple[str, int]:
    # Guard: with no images the model still returns a confident "No activity detected",
    # which would be a fabricated all-clear. Never let that reach the feed.
    if not frames:
        raise ValueError("refusing to describe zero frames")
    content = [{"type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(f).decode()}}
               for f in frames]
    content.append({"type": "text",
                    "text": USER_PROMPT_TMPL.format(num_frames=len(frames),
                                                    stream_id=stream_id,
                                                    camera=camera)})
    payload = {
        "model": "cosmos3-edge",
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": content}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
    }
    t0 = time.time()
    r = requests.post(f"{SHIM}/v1/chat/completions", json=payload, timeout=300)
    ms = int((time.time() - t0) * 1000)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip(), ms


# --------------------------------------------------------------------------- pipeline
def handle_event(after: dict, publish) -> None:
    eid = after.get("id")
    camera = after.get("camera", "unknown")
    label = after.get("label", "object")
    if not eid:
        return
    with closing(db()) as c:
        if c.execute("SELECT 1 FROM events WHERE id=?", (eid,)).fetchone():
            return  # already processed

    print(f"[porch-dad] event {eid} camera={camera} label={label}", flush=True)

    def fail(reason: str) -> None:
        """Surface the failure instead of dropping it.

        Reporting nothing is indistinguishable from "nothing happened", which is the one
        outcome a security monitor must never produce. Failures get a row so they appear
        in the feed as ERROR rather than vanishing.
        """
        print(f"[porch-dad] {eid}: {reason}", flush=True)
        save_event({
            "id": eid, "camera": camera, "label": label,
            "start_time": after.get("start_time") or time.time(),
            "end_time": after.get("end_time") or time.time(),
            "category": "ERROR", "description": f"Event captured but not analyzed: {reason}",
            "raw": json.dumps(after), "num_frames": 0, "latency_ms": 0,
            "created_at": time.time(),
        })
        publish("porchdad/events", json.dumps(
            {"id": eid, "camera": camera, "category": "ERROR", "description": reason}))

    with tempfile.TemporaryDirectory() as td:
        clip = os.path.join(td, "clip.mp4")
        if not fetch_clip(eid, clip):
            return fail("clip could not be downloaded from Frigate")
        frames = sample_frames(clip, NUM_FRAMES)

    if not frames:
        return fail("no frames could be extracted from the clip")

    try:
        text, ms = describe(frames, CAMERA_LABELS.get(camera, camera), eid)
    except Exception as e:
        return fail(f"Cosmos3-Edge inference failed: {type(e).__name__}: {e}")

    cat, reason = categorize(text)
    row = {
        "id": eid, "camera": camera, "label": label,
        "start_time": after.get("start_time") or time.time(),
        "end_time": after.get("end_time") or time.time(),
        "category": cat, "description": text, "raw": json.dumps(after),
        "num_frames": len(frames), "latency_ms": ms, "created_at": time.time(),
    }
    save_event(row)
    print(f"[porch-dad] {eid} [{cat}/{reason}] {ms}ms {len(frames)}f :: {text[:100]}", flush=True)
    publish("porchdad/events", json.dumps({k: row[k] for k in
            ("id", "camera", "label", "category", "description", "start_time", "latency_ms")}))


# --------------------------------------------------------------------------- ring
_ring_snaps: dict[str, list] = {}      # device id -> [(ts, jpeg bytes)]
_ring_last: dict[str, float] = {}      # device id -> last handled event ts
_ring_lock = threading.Lock()


def ring_snapshot(dev: str, payload: bytes) -> None:
    if not payload or len(payload) < 2048:
        return
    with _ring_lock:
        buf = _ring_snaps.setdefault(dev, [])
        buf.append((time.time(), payload))
        del buf[:-6]


def ring_recent(dev: str, window: float) -> list[bytes]:
    """Most recent distinct snapshots within the window, newest last."""
    now = time.time()
    with _ring_lock:
        buf = list(_ring_snaps.get(dev, []))
    out, seen = [], set()
    for ts, data in buf:
        if now - ts > window:
            continue
        h = (len(data), data[:64])
        if h in seen:
            continue
        seen.add(h)
        out.append(data)
    return out[-NUM_FRAMES:]


def handle_ring_event(dev: str, kind: str, publish) -> None:
    """Ring motion/ding -> collect snapshots -> describe.

    Unlike the Frigate path there is no clip, so the frame count is whatever snapshots arrived
    during the window (1..NUM_FRAMES). One good snapshot still yields a useful description, so we
    degrade gracefully rather than discarding the event.
    """
    now = time.time()
    if now - _ring_last.get(dev, 0) < RING_COOLDOWN:
        return
    _ring_last[dev] = now
    name = RING_CAMERAS.get(dev, dev)
    eid = f"ring-{dev}-{int(now)}"
    print(f"[porch-dad] ring {kind} on {name} ({dev})", flush=True)

    deadline = now + RING_WINDOW
    while time.time() < deadline and len(ring_recent(dev, RING_WINDOW + 5)) < NUM_FRAMES:
        time.sleep(0.5)
    frames = ring_recent(dev, RING_WINDOW + 5)

    def fail(reason: str) -> None:
        print(f"[porch-dad] {eid}: {reason}", flush=True)
        save_event({"id": eid, "camera": name, "label": kind,
                    "start_time": now, "end_time": time.time(),
                    "category": "ERROR",
                    "description": f"Ring {kind} captured but not analyzed: {reason}",
                    "raw": json.dumps({"device": dev, "kind": kind}), "num_frames": 0,
                    "latency_ms": 0, "created_at": time.time()})

    if not frames:
        return fail("no snapshot arrived from ring-mqtt within the window")
    try:
        text, ms = describe(frames, name, eid)
    except Exception as e:
        return fail(f"Cosmos3-Edge inference failed: {type(e).__name__}: {e}")

    cat, reason = categorize(text)
    save_event({"id": eid, "camera": name, "label": kind,
                "start_time": now, "end_time": time.time(), "category": cat,
                "description": text, "raw": json.dumps({"device": dev, "kind": kind}),
                "num_frames": len(frames), "latency_ms": ms, "created_at": time.time()})
    print(f"[porch-dad] {eid} [{cat}/{reason}] {ms}ms {len(frames)}f :: {text[:100]}", flush=True)
    publish("porchdad/events", json.dumps(
        {"id": eid, "camera": name, "label": kind, "category": cat,
         "description": text, "start_time": now, "latency_ms": ms}))


# --------------------------------------------------------------------------- mqtt
def mqtt_loop() -> None:
    import paho.mqtt.client as mqtt

    def on_connect(client, *_a):
        client.subscribe(CFG["mqtt_topic"])
        print(f"[porch-dad] subscribed to {CFG['mqtt_topic']}", flush=True)
        if RING_ENABLED:
            for t in ("ring/+/camera/+/motion/state",
                      "ring/+/camera/+/ding/state",
                      "ring/+/camera/+/snapshot/image"):
                client.subscribe(t)
            print(f"[porch-dad] subscribed to ring motion/ding/snapshot "
                  f"({len(RING_CAMERAS)} named cameras)", flush=True)

    def on_message(client, _u, msg):
        parts = msg.topic.split("/")
        if RING_ENABLED and len(parts) >= 5 and parts[2] == "camera":
            dev = parts[3]
            leaf = "/".join(parts[4:])
            if leaf == "snapshot/image":
                ring_snapshot(dev, msg.payload)
                return
            if leaf in ("motion/state", "ding/state"):
                if msg.payload.decode(errors="ignore").strip().upper() == "ON":
                    kind = "motion" if leaf.startswith("motion") else "ding"
                    threading.Thread(target=handle_ring_event,
                                     args=(dev, kind, client.publish), daemon=True).start()
                return
        try:
            data = json.loads(msg.payload.decode())
        except Exception:
            return
        if data.get("type") != "end":
            return
        after = data.get("after") or {}
        cams = CFG.get("cameras") or []
        if cams and after.get("camera") not in cams:
            return
        threading.Thread(target=handle_event, args=(after, client.publish), daemon=True).start()

    c = mqtt.Client()
    c.on_connect = on_connect
    c.on_message = on_message
    while True:
        try:
            c.connect(CFG["mqtt_host"], int(CFG["mqtt_port"]), 60)
            c.loop_forever()
        except Exception as e:
            print(f"[porch-dad] mqtt reconnect after error: {e}", flush=True)
            time.sleep(5)


# --------------------------------------------------------------------------- web
app = FastAPI(title="porch-dad")


@app.get("/api/events")
def api_events(limit: int = 50):
    with closing(db()) as c:
        rows = c.execute(
            "SELECT * FROM events ORDER BY start_time DESC LIMIT ?", (limit,)).fetchall()
    return JSONResponse([dict(r) for r in rows])


@app.get("/thumb/{event_id}")
def thumb(event_id: str):
    try:
        r = requests.get(f"{FRIGATE}/api/events/{event_id}/thumbnail.jpg", timeout=15)
        if r.status_code == 200:
            return Response(r.content, media_type="image/jpeg",
                            headers={"Cache-Control": "max-age=86400"})
    except requests.RequestException:
        pass
    raise HTTPException(404, "thumbnail unavailable")


@app.get("/clip/{event_id}")
def clip(event_id: str):
    return Response(status_code=302,
                    headers={"Location": f"{CFG['frigate_public_url'].rstrip('/')}"
                                         f"/api/events/{event_id}/clip.mp4"})


@app.get("/healthz")
def healthz():
    with closing(db()) as c:
        n = c.execute("SELECT COUNT(*) AS n FROM events").fetchone()["n"]
    return {"ok": True, "events": n, "num_frames": NUM_FRAMES, "max_tokens": MAX_TOKENS}


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


INDEX_HTML = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#0b0d0c">
<meta name="apple-mobile-web-app-capable" content="yes">
<title>porch-dad</title>
<style>
:root{--bg:#0b0d0c;--card:#161a15;--line:#2a2f28;--fg:#e8ece7;--mut:#9aa396;--g:#76b900;--r:#ff5c5c;--y:#ffb020}
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
body{margin:0;background:var(--bg);color:var(--fg);
 font:16px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
 padding:env(safe-area-inset-top) env(safe-area-inset-right) env(safe-area-inset-bottom) env(safe-area-inset-left)}
header{position:sticky;top:0;z-index:5;background:rgba(11,13,12,.94);backdrop-filter:blur(8px);
 border-bottom:1px solid var(--line);padding:14px 16px;display:flex;align-items:center;gap:10px}
h1{font-size:17px;margin:0;letter-spacing:-.01em}
h1 span{color:var(--g)}
.dot{width:8px;height:8px;border-radius:50%;background:var(--g);box-shadow:0 0 8px var(--g)}
.meta{margin-left:auto;color:var(--mut);font-size:12px}
.wrap{padding:12px;max-width:720px;margin:0 auto}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;overflow:hidden;margin-bottom:12px}
.card.alert{border-color:var(--r);box-shadow:0 0 0 1px rgba(255,92,92,.25)}
.row{display:flex;gap:12px;padding:12px}
img.th{width:104px;height:78px;object-fit:cover;border-radius:9px;background:#222;flex:none}
.body{min-width:0;flex:1}
.top{display:flex;align-items:center;gap:8px;margin-bottom:5px;flex-wrap:wrap}
.tag{font-size:11px;font-weight:700;letter-spacing:.06em;padding:2px 8px;border-radius:999px}
.tag.ALERT{background:rgba(255,92,92,.15);color:var(--r)}
.tag.ROUTINE{background:rgba(118,185,0,.13);color:var(--g)}
.tag.UNSCORED{background:rgba(255,176,32,.13);color:var(--y)}
.tag.ERROR{background:rgba(255,176,32,.2);color:var(--y)}
.card.error{border-color:var(--y)}
.cam{font-size:12px;color:var(--mut)}
.desc{font-size:14.5px;margin:0;word-wrap:break-word}
.foot{display:flex;gap:10px;align-items:center;padding:0 12px 11px;font-size:11.5px;color:var(--mut)}
.foot a{color:var(--g);text-decoration:none;margin-left:auto;border:1px solid var(--line);
 padding:5px 11px;border-radius:8px}
.empty{text-align:center;color:var(--mut);padding:60px 20px;font-size:14px}
</style></head><body>
<header><span class="dot"></span><h1>porch<span>-dad</span></h1><span class="meta" id="meta">loading…</span></header>
<div class="wrap" id="feed"><div class="empty">Waiting for events…</div></div>
<script>
const fmt = t => new Date(t*1000).toLocaleString([], {month:'short',day:'numeric',hour:'2-digit',minute:'2-digit',second:'2-digit'});
async function load(){
  try{
    const r = await fetch('/api/events?limit=60',{cache:'no-store'});
    const ev = await r.json();
    document.getElementById('meta').textContent = ev.length ? ev.length+' events' : 'no events yet';
    const f = document.getElementById('feed');
    if(!ev.length){ f.innerHTML = '<div class="empty">No events yet.<br>Frigate has not reported a completed event.</div>'; return; }
    f.innerHTML = ev.map(e => `
      <div class="card ${e.category==='ALERT'?'alert':(e.category==='ERROR'?'error':'')}">
        <div class="row">
          <img class="th" loading="lazy" src="/thumb/${e.id}" onerror="this.style.visibility='hidden'">
          <div class="body">
            <div class="top">
              <span class="tag ${e.category}">${e.category}</span>
              <span class="cam">${e.camera} · ${e.label}</span>
            </div>
            <p class="desc">${(e.description||'').replace(/[<>&]/g, c=>({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]))}</p>
          </div>
        </div>
        <div class="foot"><span>${fmt(e.start_time)}</span><span>· ${e.num_frames}f · ${e.latency_ms}ms</span>
          <a href="/clip/${e.id}">clip</a></div>
      </div>`).join('');
  }catch(err){ document.getElementById('meta').textContent = 'offline'; }
}
load(); setInterval(load, 5000);
</script></body></html>"""


def main() -> None:
    init_db()
    threading.Thread(target=mqtt_loop, daemon=True).start()
    print(f"[porch-dad] frames={NUM_FRAMES} (max {MAX_FRAMES}) max_tokens={MAX_TOKENS} "
          f"frigate={FRIGATE} shim={SHIM}", flush=True)
    uvicorn.run(app, host=CFG.get("web_host", "0.0.0.0"),
                port=int(CFG.get("web_port", 8095)), log_level="warning")


if __name__ == "__main__":
    main()
