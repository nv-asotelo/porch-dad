#!/usr/bin/env python3
"""Score the VLM benchmark: accuracy against the ground-truth labels, latency and memory from the runs.

  python3 score.py --data DIR --labels labels.json results/*.jsonl [--judgments judgments.json] [--json out.json]

Presence questions are scored only where the labellers did not mark that question ambiguous for that
frame. Grounding is scored only on frames with exactly one person whose Frigate detector box the
labellers confirmed as tight ("good"): that box is the reference, and IoU >= 0.5 counts as a hit.
Captions are scored by a separate judge pass (judgments.json, 0-2 per caption); without it the
caption column is left empty rather than guessed.
"""
import argparse
import json
import re
import statistics
import sys

YES = re.compile(r"^\W*(yes|yeah|yep|true)\b", re.I)
NO = re.compile(r"^\W*(no|nope|false|none)\b", re.I)
WORDS = {"zero": 0, "none": 0, "no": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
         "seven": 7, "eight": 8, "nine": 9, "ten": 10}
NUM = re.compile(r"-?\d+(?:\.\d+)?")


def yes_no(text):
    if text is None:
        return None
    if YES.search(text):
        return True
    if NO.search(text):
        return False
    return None


def count(text):
    if text is None:
        return None
    m = NUM.search(text)
    if m:
        return int(float(m.group(0)))
    for w in re.findall(r"[a-z]+", text.lower()):
        if w in WORDS:
            return WORDS[w]
    return None


def parse_box(text, task):
    """First box in an answer, as normalized [x1, y1, x2, y2] - or None.

    Qwen-family and Cosmos answers are JSON bbox_2d on a 0-1000 grid, x first. Gemma's native
    convention is box_2d as [ymin, xmin, ymax, xmax] on the same grid. InternVL writes
    [[x1, y1, x2, y2]] on 0-1000 after the runtime strips its <ref>/<box> tokens. So: take the four
    numbers that follow a box key if there is one, else the first four numbers in the answer.
    """
    if not text:
        return None
    m = re.search(r'"?(bbox_2d|box_2d|bbox|box)"?\s*[:=]\s*\[+\s*([^\]]+)', text)
    key = m.group(1) if m else None
    nums = [float(x) for x in NUM.findall(m.group(2) if m else text)][:4]
    if len(nums) < 4:
        return None
    if key == "box_2d" or (task == "ground_native" and key is None and "box_2d" in text):
        nums = [nums[1], nums[0], nums[3], nums[2]]
    scale = 1000.0 if max(nums) > 1.5 else 1.0
    x1, y1, x2, y2 = (min(max(v / scale, 0.0), 1.0) for v in nums)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else None


def load_run(path):
    meta, answers = {}, []
    for line in open(path):
        rec = json.loads(line)
        if rec["kind"] == "meta":
            meta.update(rec)
        else:
            answers.append(rec)
    return meta, answers


def score_run(meta, answers, labels, boxes, judgments):
    by_task = {}
    for a in answers:
        by_task.setdefault(a["task"], []).append(a)
    out = {"label": meta.get("label"), "answers": len(answers),
           "errors": sum(1 for a in answers if a.get("error"))}

    def presence(task, field, flag):
        right = total = tp = fp = fn = unparsed = 0
        for a in by_task.get(task, []):
            gt = labels.get(a["id"])
            if not gt or gt.get(flag):
                continue
            truth = gt[field] > 0 if isinstance(field, str) else sum(gt[f] for f in field) > 0
            said = yes_no(a.get("text"))
            unparsed += said is None
            total += 1
            right += said == truth
            tp += bool(said) and truth
            fp += bool(said) and not truth
            fn += (said is False or said is None) and truth
        res = {"acc": right / total if total else None, "n": total, "unparsed": unparsed}
        if task == "person":
            res["precision"] = tp / (tp + fp) if tp + fp else None
            res["recall"] = tp / (tp + fn) if tp + fn else None
        return res

    out["person"] = presence("person", "people", "person_ambiguous")
    out["vehicle"] = presence("vehicle", "vehicles", "vehicle_ambiguous")
    out["animal"] = presence("animal", ("dogs", "cats"), "animal_ambiguous")

    exact = total = unparsed = 0
    errs = []
    for a in by_task.get("count", []):
        gt = labels.get(a["id"])
        if not gt or gt.get("person_ambiguous"):
            continue
        n = count(a.get("text"))
        total += 1
        unparsed += n is None
        exact += n == gt["people"]
        errs.append(abs((n if n is not None else 0) - gt["people"]))
    out["count"] = {"acc": exact / total if total else None, "mae": statistics.mean(errs) if errs else None,
                    "n": total, "unparsed": unparsed}

    for task in ("ground", "ground_native"):
        ious, unparsed = [], 0
        for a in by_task.get(task, []):
            ref = boxes.get(a["id"])
            if ref is None:
                continue
            b = parse_box(a.get("text"), task)
            unparsed += b is None
            ious.append(iou(b, ref) if b else 0.0)
        if ious:
            out[task] = {"miou": statistics.mean(ious), "acc50": sum(i >= 0.5 for i in ious) / len(ious),
                         "n": len(ious), "unparsed": unparsed}

    lab = meta.get("label")
    if judgments and lab in judgments:
        scores = [s for s in judgments[lab].values() if s is not None]
        out["caption"] = {"mean": statistics.mean(scores) if scores else None,
                          "good": sum(s == 2 for s in scores) / len(scores) if scores else None,
                          "n": len(scores)}

    lat = {}
    for task, items in by_task.items():
        ok = [a for a in items if not a.get("error")]
        tot = [a["total_ms"] for a in ok]
        ttft = [a["ttft_ms"] for a in ok if a.get("ttft_ms") is not None]
        decode = []
        for a in ok:
            u = a.get("usage") or {}
            gen = u.get("completion_tokens") or 0
            if gen > 2 and a.get("ttft_ms") is not None:
                decode.append((a["total_ms"] - a["ttft_ms"]) / (gen - 1))
        lat[task] = {"p50_ms": statistics.median(tot) if tot else None, "p90_ms": pct(tot, 0.9),
                     "ttft_p50_ms": statistics.median(ttft) if ttft else None,
                     "decode_ms_per_tok": statistics.median(decode) if decode else None,
                     "prompt_tokens_p50": statistics.median([(a.get("usage") or {}).get("prompt_tokens") or 0 for a in ok]) if ok else None}
    out["latency"] = lat
    mb, ma = meta.get("memory_before") or {}, meta.get("memory_after") or {}
    out["memory"] = {"peak_mib": round(ma["shim_peak"] / 2**20) if ma.get("shim_peak") else None,
                     "current_mib": round(ma["shim_current"] / 2**20) if ma.get("shim_current") else None,
                     "board_available_mib": round(ma["mem_available"] / 2**20) if ma.get("mem_available") else None,
                     "board_available_before_mib": round(mb["mem_available"] / 2**20) if mb.get("mem_available") else None}
    return out


def fmt(v, kind="pct"):
    if v is None:
        return "-"
    return f"{v * 100:.0f}%" if kind == "pct" else (f"{v:.0f}" if kind == "ms" else f"{v:.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--labels", required=True)
    ap.add_argument("--judgments")
    ap.add_argument("--json", help="write the full scores here")
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    lab_items = json.load(open(args.labels))["items"]
    labels = {i: v for i, v in lab_items.items()}
    boxes = {i: v["person_box"] for i, v in lab_items.items() if v.get("person_box")}
    judgments = json.load(open(args.judgments)) if args.judgments else None
    rows = [score_run(*load_run(p), labels, boxes, judgments) for p in args.runs]

    head = ("model", "person", "P/R", "vehicle", "animal", "count", "MAE", "ground", "IoU", "caption",
            "caption p50", "yes/no p50", "TTFT", "ms/tok", "peak MiB")
    print(" | ".join(head))
    print(" | ".join("---" for _ in head))
    for r in rows:
        g = r.get("ground_native") if (r.get("ground_native") or {}).get("acc50", -1) > (r.get("ground") or {}).get("acc50", -1) else r.get("ground")
        lat = r["latency"]
        print(" | ".join([
            r["label"], fmt(r["person"]["acc"]),
            f"{fmt(r['person'].get('precision'))}/{fmt(r['person'].get('recall'))}",
            fmt(r["vehicle"]["acc"]), fmt(r["animal"]["acc"]), fmt(r["count"]["acc"]), fmt(r["count"]["mae"], "f"),
            fmt((g or {}).get("acc50")), fmt((g or {}).get("miou"), "f"),
            fmt((r.get("caption") or {}).get("mean"), "f"),
            fmt((lat.get("caption") or {}).get("p50_ms"), "ms"), fmt((lat.get("person") or {}).get("p50_ms"), "ms"),
            fmt((lat.get("caption") or {}).get("ttft_p50_ms"), "ms"), fmt((lat.get("caption") or {}).get("decode_ms_per_tok"), "f"),
            str(r["memory"]["peak_mib"] or "-")]))
    if args.json:
        json.dump(rows, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
