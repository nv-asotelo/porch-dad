#!/usr/bin/env python3
"""Cut the VLM benchmark's frame set out of Frigate.

Two sources per camera:
  events      Frigate's own tracked objects: the clean snapshot (no box, no clock) at the
              detect resolution, with the detector's label and box. These are the frames
              Frigate's GenAI hands Cosmos3-Edge in production (genai.use_snapshot).
  recordings  a frame from a random 10 s recording segment, at the recording resolution.
              Segments with no tracked object supply the negatives - the empty porch, the
              dark hallway - that an event-only set would never contain.

Picks are stratified so one busy camera or one label cannot crowd out the rest: events
round-robin over (label, day/night) groups, recording segments over (has objects,
day/night). Everything is seeded, so the same --seed against the same Frigate gives the
same set. Frames are written as JPEG (quality 95) under <out>/images/ and described in
<out>/manifest.json. The images show the household; they stay on the LAN and out of git.
"""
import argparse
import datetime
import io
import json
import os
import random
import sys
import time
import urllib.request

from PIL import Image

NIGHT_HOURS = set(range(20, 24)) | set(range(0, 6))


def get(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def get_json(url):
    return json.loads(get(url))


def period(ts):
    return "night" if datetime.datetime.fromtimestamp(ts).hour in NIGHT_HOURS else "day"


def round_robin(groups, n, rng):
    """Take n items, one group at a time, in a seeded order inside each group."""
    pools = {k: rng.sample(v, len(v)) for k, v in sorted(groups.items())}
    picked = []
    while len(picked) < n and any(pools.values()):
        for k in sorted(pools):
            if pools[k] and len(picked) < n:
                picked.append(pools[k].pop())
    return picked


def save_jpeg(raw, path):
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    im.save(path, "JPEG", quality=95)
    return im.size


def xywh_to_box(b):
    """Frigate's normalized [x, y, w, h] -> [x1, y1, x2, y2], clamped to the frame."""
    x, y, w, h = b
    return [round(max(0.0, x), 4), round(max(0.0, y), 4), round(min(1.0, x + w), 4), round(min(1.0, y + h), 4)]


def collect_events(api, camera, n, rng, out):
    events = get_json(f"{api}/events?cameras={camera}&limit=2000&has_snapshot=1")
    groups = {}
    for e in events:
        groups.setdefault((e["label"], period(e["start_time"])), []).append(e)
    items = []
    for e in round_robin(groups, n * 2, rng):  # spares for snapshots that fail to fetch
        if len(items) == n:
            break
        name = f"{camera}-ev-{e['id'].split('-')[0].replace('.', '_')}.jpg"
        try:
            size = save_jpeg(get(f"{api}/events/{e['id']}/snapshot-clean.webp"), os.path.join(out, "images", name))
        except Exception as err:  # an expired or half-written snapshot: take the next one
            print(f"  skip event {e['id']}: {err}", file=sys.stderr)
            continue
        data = e.get("data") or {}
        items.append({
            "id": name[:-4], "camera": camera, "source": "event", "image": f"images/{name}",
            "time": data.get("snapshot_frame_time") or e["start_time"], "period": period(e["start_time"]),
            "width": size[0], "height": size[1],
            "frigate": {"event_id": e["id"], "label": e["label"], "score": data.get("top_score"),
                        "box": xywh_to_box(data["box"]) if data.get("box") else None,
                        "description": data.get("description")},
        })
    return items


def collect_recordings(api, camera, n, rng, out, hours):
    now = time.time()
    segments = get_json(f"{api}/{camera}/recordings?after={int(now - hours * 3600)}&before={int(now - 60)}")
    groups = {}
    for s in segments:
        groups.setdefault((s.get("objects", 0) > 0, period(s["start_time"])), []).append(s)
    items = []
    for s in round_robin(groups, n * 2, rng):
        if len(items) == n:
            break
        t = round(s["start_time"] + rng.uniform(1.0, max(1.5, s.get("duration", 10) - 1.0)), 2)
        name = f"{camera}-rec-{str(t).replace('.', '_')}.jpg"
        try:
            size = save_jpeg(get(f"{api}/{camera}/recordings/{t}/snapshot.png"), os.path.join(out, "images", name))
        except Exception as err:
            print(f"  skip segment {s['id']}: {err}", file=sys.stderr)
            continue
        items.append({
            "id": name[:-4], "camera": camera, "source": "recording", "image": f"images/{name}",
            "time": t, "period": period(t), "width": size[0], "height": size[1],
            "frigate": {"segment_id": s["id"], "objects": s.get("objects", 0), "motion": s.get("motion", 0)},
        })
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frigate", default="http://127.0.0.1:5000", help="Frigate base URL")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hours", type=float, default=72, help="how far back to look for recording segments")
    ap.add_argument("--plan", nargs="+", required=True, metavar="CAMERA=EVENTS:RECORDINGS",
                    help="frames per camera, e.g. front_driveway=20:10")
    args = ap.parse_args()
    api = args.frigate.rstrip("/") + "/api"
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    items = []
    for spec in args.plan:
        camera, counts = spec.split("=")
        n_events, n_recordings = (int(x) for x in counts.split(":"))
        rng = random.Random(f"{args.seed}:{camera}")
        got_e = collect_events(api, camera, n_events, rng, args.out) if n_events else []
        got_r = collect_recordings(api, camera, n_recordings, rng, args.out, args.hours) if n_recordings else []
        print(f"{camera}: {len(got_e)}/{n_events} events, {len(got_r)}/{n_recordings} recording frames")
        items += got_e + got_r
    manifest = {"created": datetime.datetime.now().isoformat(timespec="seconds"), "seed": args.seed,
                "frigate": args.frigate, "plan": args.plan, "items": items}
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"{len(items)} frames -> {args.out}/manifest.json")


if __name__ == "__main__":
    main()
