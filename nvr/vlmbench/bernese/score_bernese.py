"""Score Bernese mountain dog presence and grounding against the reconciled labels.

  python score_bernese.py labels.json run.jsonl [run.jsonl ...]

labels.json: {frame: {"daycare": bool, "boxes": [[x1, y1, x2, y2] in 0-1, ...]}} (the Bernese boxes).
A run answers per frame with "found", "box" (the one a notification would draw) and optionally
"boxes" (every box it gave). Frames without the daycare feed are left out.
"""
import json
import sys


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def best(box, gts):
    return max((iou(box, g) for g in gts), default=0.0)


def score(labels, path):
    rows = [json.loads(l) for l in open(path)]
    meta = next((r for r in rows if r.get("kind") == "meta"), {})
    ans = {r["id"]: r for r in rows if r.get("kind") == "answer"}
    tp = fp = fn = tn = hit = wrong_dog = any_hit = 0
    ious, ms = [], []
    for fid, lab in labels.items():
        if not lab["daycare"] or fid not in ans:
            continue
        a, gts = ans[fid], lab["boxes"]
        if a.get("ms") is not None:
            ms.append(a["ms"])
        if gts and a["found"]:
            tp += 1
            if a.get("box"):
                o = best(a["box"], gts)
                ious.append(o)
                hit += o >= 0.5
                wrong_dog += o < 0.1
            any_hit += any(best(b, gts) >= 0.5 for b in (a.get("boxes") or ([a["box"]] if a.get("box") else [])))
        elif gts:
            fn += 1
        elif a["found"]:
            fp += 1
        else:
            tn += 1
    pos = tp + fn
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / pos if pos else None
    ms.sort()
    return {"run": meta.get("label") or path, "frames": tp + fp + fn + tn, "bernese_frames": pos,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec and rec else 0.0,
            # what a notification shows: found, and its box on the Bernese
            "boxed_right": hit, "boxed_right_of_bernese_frames": hit / pos if pos else None,
            "wrong_dog": wrong_dog, "any_box_right": any_hit,
            "mean_iou_when_found": sum(ious) / len(ious) if ious else None,
            "p50_ms": ms[len(ms) // 2] if ms else None}


def main():
    labels = json.load(open(sys.argv[1]))
    pct = lambda v: "-" if v is None else f"{v * 100:.0f}%"
    print("| Run | Bernese frames | Said Bernese (right / wrong) | Precision | Recall | Box on the Bernese | Wrong dog boxed | Mean IoU when found | p50 ms |")
    print("|---|---|---|---|---|---|---|---|---|")
    for path in sys.argv[2:]:
        s = score(labels, path)
        print(f"| {s['run']} | {s['bernese_frames']} of {s['frames']} | {s['tp'] + s['fp']} ({s['tp']} / {s['fp']}) "
              f"| {pct(s['precision'])} | {pct(s['recall'])} | {s['boxed_right']} ({pct(s['boxed_right_of_bernese_frames'])})"
              f" | {s['wrong_dog']} | {'-' if s['mean_iou_when_found'] is None else round(s['mean_iou_when_found'], 2)}"
              f" | {s['p50_ms'] if s['p50_ms'] is not None else '-'} |")
        print(json.dumps(s), file=sys.stderr)


if __name__ == "__main__":
    main()


def frigate_coverage(labels, frigate_dogs, min_score=0.7):
    """How often Frigate's own detector (probe) boxes the labelled Bernese, at a track-worthy score."""
    seen = pos = 0
    for fid, lab in labels.items():
        if not lab["daycare"] or not lab["boxes"]:
            continue
        pos += 1
        dets = [d[:4] for d in frigate_dogs.get(fid, []) if d[4] >= min_score]
        seen += any(best(d, lab["boxes"]) >= 0.3 for d in dets)
    return seen, pos
