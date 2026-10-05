"""Reconcile the two labelling passes: agreement is accepted, disagreement is listed for review.

  python reconcile.py   -> labels-agreed.json, review.json (frames to settle by hand)

A label is the set of candidate ids marked Bernese (plus an unboxed Bernese, if any) and the daycare
flag. Boxes come from the detector candidates the labellers picked, so ground truth never depends on
any model under test.
"""
import glob
import json
import os

D = "/home/asotelo/vlm-sweep/bernese"
cands = json.load(open(f"{D}/eval-cands/candidates.json"))
passes = {}
for f in sorted(glob.glob(f"{D}/labels/batch-*-*.json")):
    who = os.path.basename(f)[:-5].split("-")[-1]          # A or B
    for fr in json.load(open(f))["frames"]:
        passes.setdefault(fr["id"], {})[who] = fr

agreed, review = {}, {}
for fid in sorted(cands):
    p = passes.get(fid, {})
    if set(p) != {"A", "B"}:
        review[fid] = {"why": f"labelled by {sorted(p) or 'nobody'}", **p}
        continue
    a, b = p["A"], p["B"]
    key = lambda x: (bool(x["daycare"]), tuple(sorted(x.get("bernese_ids") or [])), bool(x.get("unboxed")))
    if key(a) == key(b):
        boxes = [c["box"] for c in cands[fid] if c["id"] in set(a.get("bernese_ids") or [])]
        if a.get("unboxed") and b.get("unboxed"):
            boxes.append([round((u + v) / 2, 4) for u, v in zip(a["unboxed"], b["unboxed"])])
        agreed[fid] = {"daycare": bool(a["daycare"]), "boxes": boxes,
                       "confidence": [a.get("confidence"), b.get("confidence")]}
    else:
        review[fid] = {"why": "disagree", "A": a, "B": b}
json.dump(agreed, open(f"{D}/labels-agreed.json", "w"), indent=1)
json.dump(review, open(f"{D}/review.json", "w"), indent=1)
pos = sum(bool(v["boxes"]) for v in agreed.values())
print(f"{len(agreed)} agreed ({pos} with a Bernese, {sum(v['daycare'] for v in agreed.values())} daycare), "
      f"{len(review)} to review: {sorted(review)}")
