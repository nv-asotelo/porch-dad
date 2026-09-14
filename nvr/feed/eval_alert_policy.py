#!/usr/bin/env python3
"""Score the alert policy against a judged truth set, per camera.

Truth was established by looking at each of the 35 stored frames directly - not by trusting the
caption, and not by trusting Frigate's label. Where the two disagreed the image decided.
"""
import importlib.util
import json
import sys
from collections import OrderedDict


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Events that must NOT alert. Every one is a parked car that lives in this driveway, re-detected
# by Frigate because light or shadow moved across it. Verified frame by frame:
#   1789340942  silver SUV parked, vehicle gate closed; the blue shape by the gate is foliage,
#               confirmed by zooming in - there is no person
#   1789343252  two cars parked, daylight
#   1789347348  silver SUV parked; the only change since 16:09 is the porch lamp coming on at dusk
#   1789352863  same two cars, parked, night
#   1789352873  same two cars, parked, night, 11 s after the previous one
#   1789343348  red car parked in the drive, second-floor view
NO_ALERT = {
    "1789340942.19479-jdt1yf",
    "1789343252.658581-ts5g63",
    "1789347348.487696-1u81v8",
    "1789352863.50231-fv97yv",
    "1789352873.693748-uu7zc8",
    "1789343348.425568-p56476",
}

# Everything else must alert. Two are worth naming because the caption alone would have missed them:
#   1789347766  Frigate said "dog"; the caption said "two women at a table". There IS a black dog
#               at the woman's feet - the label was right and the caption was wrong.
#   1789343538  Frigate said "person"; the caption described only trees and a driveway. There IS a
#               person at the bottom-right edge of frame.
# And one is a genuine miss we accept:
#   1789351797  the red car is ARRIVING - headlights on, occupant visible - but its own caption
#               says "parked side by side", so a caption-driven classifier cannot tell it from the
#               two re-detections that follow it 17 minutes later.
KNOWN_HARD = {
    "1789351797.726124-8qpvgj": "arriving car described as 'parked' by the caption",
}

rows = json.load(open("rerun/rerun.json"))
policies = OrderedDict([
    ("before (fusion overrides suppression)", load("alert_policy_live.py", "ap_before")),
    ("after  (fusion may only add)", load(
        "/home/asotelo/jetson-orin-nano-cosmos3-edge-optimization/nvr/feed/alert_policy.py",
        "ap_after")),
])

for name, ap in policies.items():
    per_cam = {}
    fp, fn = [], []
    for r in rows:
        want = r["id"] not in NO_ALERT
        got = ap.classify(r["desc"], r["label"])["alert"]
        c = per_cam.setdefault(r["camera"], [0, 0])
        c[1] += 1
        if got == want:
            c[0] += 1
        elif got and not want:
            fp.append(r)
        else:
            fn.append(r)

    tot_ok = sum(v[0] for v in per_cam.values())
    tot_n = sum(v[1] for v in per_cam.values())
    print(f"\n=== {name} ===")
    print(f"  {'camera':20s} {'accuracy':>9s}   {'n':>3s}")
    for cam in ("front_driveway", "front_entryway", "pinky", "driveway_2nd_floor"):
        if cam not in per_cam:
            continue
        ok, n = per_cam[cam]
        star = "  <- priority" if cam != "driveway_2nd_floor" else ""
        print(f"  {cam:20s} {ok/n*100:8.1f}%  {ok:2d}/{n}{star}")
    prio = [per_cam[c] for c in ("front_driveway", "front_entryway", "pinky") if c in per_cam]
    pok, pn = sum(v[0] for v in prio), sum(v[1] for v in prio)
    print(f"  {'-'*44}")
    print(f"  {'PRIORITY CAMERAS':20s} {pok/pn*100:8.1f}%  {pok:2d}/{pn}")
    print(f"  {'ALL CAMERAS':20s} {tot_ok/tot_n*100:8.1f}%  {tot_ok:2d}/{tot_n}")
    print(f"  false alarms: {len(fp)}   missed: {len(fn)}")
    for r in fp:
        print(f"    FALSE ALARM  {r['camera']:15s} [{r['label']}] {r['desc'][:62]}")
    for r in fn:
        why = KNOWN_HARD.get(r["id"], "")
        print(f"    MISSED       {r['camera']:15s} [{r['label']}] {r['desc'][:62]}"
              + (f"\n                 ^ {why}" if why else ""))
