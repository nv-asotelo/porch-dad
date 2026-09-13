#!/usr/bin/env python3
"""Push Frigate events (with the Cosmos3-Edge description) to a phone via ntfy or Pushover.

Frigate's GenAI description is produced *after* the event ends, so this waits for it rather than
firing immediately with an empty body. If it never arrives the notification still goes out with the
detection label - a security notification that never sends is worse than one lacking prose.

Provider is chosen by `provider:` in config.yaml. Both attach the snapshot image.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import requests
import yaml

CFG = yaml.safe_load(Path(os.environ.get("FRIGATE_NOTIFY_CONFIG",
                                         "/home/orin/nvr/notify/config.yaml")).read_text())
FRIGATE = CFG["frigate_url"].rstrip("/")
PROVIDER = CFG.get("provider", "ntfy").lower()
CAMERAS = CFG.get("cameras") or []
MIN_GAP = float(CFG.get("min_seconds_between", 30))
DESC_WAIT = float(CFG.get("description_wait_seconds", 25))
ALERT_WORDS = [w.lower() for w in (CFG.get("alert_keywords") or [])]

_last: dict[str, float] = {}


def event_detail(eid: str) -> dict:
    try:
        r = requests.get(f"{FRIGATE}/api/events/{eid}", timeout=15)
        if r.status_code == 200:
            return r.json()
    except requests.RequestException:
        pass
    return {}


def wait_for_description(eid: str, timeout: float) -> str | None:
    """Poll until Frigate attaches the GenAI description, or give up."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        d = (event_detail(eid) or {}).get("description")
        if d:
            return d.strip()
        time.sleep(2.0)
    return None


def snapshot(eid: str) -> bytes | None:
    try:
        r = requests.get(f"{FRIGATE}/api/events/{eid}/snapshot.jpg", timeout=20)
        if r.status_code == 200 and len(r.content) > 1024:
            return r.content
    except requests.RequestException:
        pass
    return None


def is_alert(text: str) -> bool:
    t = (text or "").lower()
    return any(w in t for w in ALERT_WORDS)


# --------------------------------------------------------------------------- providers
def send_ntfy(title: str, body: str, img: bytes | None, alert: bool, url: str) -> None:
    c = CFG["ntfy"]
    headers = {
        "Title": title,
        "Priority": "high" if alert else "default",
        "Tags": "rotating_light" if alert else "house",
        "Click": url,
    }
    if c.get("token"):
        headers["Authorization"] = f"Bearer {c['token']}"
    topic = f"{c['server'].rstrip('/')}/{c['topic']}"
    if img:
        # Attaching the image as the body makes it render inline in the ntfy app.
        headers["Filename"] = "snapshot.jpg"
        headers["Message"] = body
        requests.put(topic, data=img, headers=headers, timeout=30).raise_for_status()
    else:
        requests.post(topic, data=body.encode("utf-8"), headers=headers, timeout=30).raise_for_status()


def send_pushover(title: str, body: str, img: bytes | None, alert: bool, url: str) -> None:
    c = CFG["pushover"]
    data = {
        "token": c["api_token"],
        "user": c["user_key"],
        "title": title,
        "message": body,
        "priority": 1 if alert else 0,
        "url": url,
        "url_title": "Open in Frigate",
    }
    files = {"attachment": ("snapshot.jpg", img, "image/jpeg")} if img else None
    r = requests.post("https://api.pushover.net/1/messages.json",
                      data=data, files=files, timeout=30)
    r.raise_for_status()


def notify(title: str, body: str, img: bytes | None, alert: bool, url: str) -> None:
    if PROVIDER == "ntfy":
        send_ntfy(title, body, img, alert, url)
    elif PROVIDER == "pushover":
        send_pushover(title, body, img, alert, url)
    else:
        raise ValueError(f"unknown provider {PROVIDER!r}")


# --------------------------------------------------------------------------- pipeline
def handle(after: dict) -> None:
    eid = after.get("id")
    cam = after.get("camera", "camera")
    label = after.get("label", "object")
    if not eid:
        return
    if CAMERAS and cam not in CAMERAS:
        return
    now = time.time()
    if now - _last.get(cam, 0) < MIN_GAP:
        return
    _last[cam] = now

    desc = wait_for_description(eid, DESC_WAIT)
    body = desc or f"{label.title()} detected (no description generated)."
    alert = is_alert(body)
    pretty = cam.replace("_", " ").title()
    title = f"{'ALERT - ' if alert else ''}{pretty}"
    url = f"{CFG.get('frigate_public_url', FRIGATE).rstrip('/')}/events?event_id={eid}"

    try:
        notify(title, body, snapshot(eid), alert, url)
        print(f"[notify] {eid} -> {PROVIDER} [{'ALERT' if alert else 'info'}] {body[:80]}",
              flush=True)
    except Exception as e:
        print(f"[notify] {eid} FAILED: {type(e).__name__}: {e}", flush=True)


def main() -> None:
    import paho.mqtt.client as mqtt

    def on_connect(c, *_a):
        c.subscribe("frigate/events")
        print(f"[notify] provider={PROVIDER} cameras={CAMERAS or 'all'} "
              f"desc_wait={DESC_WAIT}s - subscribed", flush=True)

    def on_message(_c, _u, msg):
        try:
            data = json.loads(msg.payload.decode())
        except Exception:
            return
        if data.get("type") != "end":
            return
        threading.Thread(target=handle, args=(data.get("after") or {}, ), daemon=True).start()

    c = mqtt.Client()
    c.on_connect = on_connect
    c.on_message = on_message
    while True:
        try:
            c.connect(CFG.get("mqtt_host", "127.0.0.1"), int(CFG.get("mqtt_port", 1883)), 60)
            c.loop_forever()
        except Exception as e:
            print(f"[notify] mqtt reconnect after {e}", flush=True)
            time.sleep(5)


if __name__ == "__main__":
    main()
