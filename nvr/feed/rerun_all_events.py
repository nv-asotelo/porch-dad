#!/usr/bin/env python3
"""Re-run Cosmos3-Edge over every stored event and classify with the live alert policy.

Fresh inference on every run - nothing cached, nothing carried over from an earlier session.
Writes both the image and the new description so the result can be judged against the picture.
"""
import base64
import json
import os
import sys
import time

import requests

sys.path.insert(0, "/home/orin/nvr/feed")
from alert_policy import DESCRIBE_PROMPT, classify  # noqa: E402

FRIGATE = "http://127.0.0.1:5000"
SHIM = "http://127.0.0.1:8000/v1/chat/completions"
OUT = "/home/orin/rerun_out"
os.makedirs(OUT, exist_ok=True)

events = requests.get(f"{FRIGATE}/api/events?limit=500", timeout=30).json()
events = [e for e in events if (e.get("data") or {}).get("description")]
print(f"events with stored descriptions: {len(events)}", flush=True)

rows = []
for i, e in enumerate(events, 1):
    eid, cam, lab = e["id"], e["camera"], e["label"]

    img = None
    for endpoint in ("snapshot.jpg", "thumbnail.jpg"):
        r = requests.get(f"{FRIGATE}/api/events/{eid}/{endpoint}", timeout=25)
        if r.status_code == 200 and len(r.content) > 1024:
            img = r.content
            break
    if not img:
        print(f"  [{i}/{len(events)}] {eid} NO IMAGE - skipped", flush=True)
        continue

    safe = eid.replace(".", "_")
    with open(os.path.join(OUT, safe + ".jpg"), "wb") as fh:
        fh.write(img)

    b64 = base64.b64encode(img).decode()
    t0 = time.time()
    resp = requests.post(SHIM, timeout=180, json={
        "model": "nvidia/Cosmos3-Edge",
        "max_tokens": 512,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
            {"type": "text", "text": DESCRIBE_PROMPT},
        ]}],
    })
    ms = int((time.time() - t0) * 1000)
    if resp.status_code != 200:
        print(f"  [{i}/{len(events)}] {eid} HTTP {resp.status_code}", flush=True)
        continue

    desc = resp.json()["choices"][0]["message"]["content"].strip()
    v = classify(desc, lab)
    rows.append({
        "id": eid, "file": safe + ".jpg", "camera": cam, "label": lab,
        "desc": desc, "stored_desc": (e.get("data") or {}).get("description"),
        "alert": v["alert"], "cats": v["categories"], "moving": v.get("moving"),
        "headline": v["headline"], "ms": ms,
    })
    cats = ",".join(v["categories"]) or "-"
    print(f"  [{i}/{len(events)}] {cam:18s} {lab:7s} alert={str(v['alert']):5s} "
          f"{cats:22s} {ms:5d}ms :: {desc[:58]}", flush=True)

with open(os.path.join(OUT, "rerun.json"), "w") as fh:
    json.dump(rows, fh, indent=1)
print(f"\nwrote {len(rows)} rows -> {OUT}/rerun.json")
