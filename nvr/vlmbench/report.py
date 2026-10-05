#!/usr/bin/env python3
"""Markdown results tables for vlmbench/README.md from score.py --json output (plus caption judgments).

  python report.py --scores scores.json --labels labels.json --results DIR [--judgments judgments.json]
                   [--names label=Name,...] [--off-board label,...]

Rows named in --off-board were measured on another machine: they get accuracy rows only.
"""
import argparse
import collections
import json
import os


def pct(v):
    return "-" if v is None else f"{v * 100:.0f}%"


def ms(v):
    return "-" if v is None else f"{v:.0f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--judgments")
    ap.add_argument("--names", help="label=display name,...")
    ap.add_argument("--off-board", default="", help="labels measured off the Orin (no speed/memory row)")
    args = ap.parse_args()
    rows = json.load(open(args.scores))
    labels = json.load(open(args.labels))["items"]
    judg = json.load(open(args.judgments)) if args.judgments else {}
    names = dict(p.split("=", 1) for p in args.names.split(",")) if args.names else {}
    name = lambda r: names.get(r["label"], r["label"])
    off_board = set(filter(None, args.off_board.split(",")))
    # The model's own box format counts when it scores better than the shared bbox_2d prompt.
    native = lambda r: (r.get("ground_native") or {}).get("acc50", -1) > (r.get("ground") or {}).get("acc50", -1)

    print("### Accuracy\n")
    print("| Model | Person (P / R) | Vehicle | Dog or cat | Count exact (MAE) | Box IoU>=0.5 (mean IoU) | Caption (0-2) | Hallucinated captions |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        g = r.get("ground_native") if native(r) else r.get("ground")
        j = judg.get(r["label"]) or {}
        cap = [v["score"] for v in j.values()]
        hall = [v["hallucination"] for v in j.values()]
        mae = "-" if r["count"]["mae"] is None else "%.2f" % r["count"]["mae"]
        miou = "-" if not g else "%.2f" % g["miou"]
        capm = "-" if not cap else "%.2f" % (sum(cap) / len(cap))
        hallp = "-" if not hall else pct(sum(hall) / len(hall))
        person = r["person"]
        print(f"| {name(r)} | {pct(person['acc'])} ({pct(person.get('precision'))} / {pct(person.get('recall'))}) "
              f"| {pct(r['vehicle']['acc'])} | {pct(r['animal']['acc'])} | {pct(r['count']['acc'])} ({mae}) "
              f"| {pct((g or {}).get('acc50'))} ({miou}) | {capm} | {hallp} |")

    print("\n### Speed and memory (on the Orin, beside the running NVR)\n")
    print("| Model | Caption p50 / p90 ms | First token ms | Decode ms/token | Yes/no p50 ms | Box p50 ms | Resident GiB |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        if r["label"] in off_board:
            continue
        lat = r["latency"]
        c, p = lat.get("caption") or {}, lat.get("person") or {}
        g = lat.get("ground_native" if native(r) else "ground") or {}
        mem = r["memory"].get("resident_mib")
        dec = c.get("decode_ms_per_tok")
        decs = "-" if dec is None else "%.1f" % dec
        mems = "-" if not mem else "%.2f" % (mem / 1024)
        print(f"| {name(r)} | {ms(c.get('p50_ms'))} / {ms(c.get('p90_ms'))} | {ms(c.get('ttft_p50_ms'))} "
              f"| {decs} | {ms(p.get('p50_ms'))} | {ms(g.get('p50_ms'))} | {mems} |")

    print("\n### False \"person\" answers, by camera (frames with nobody in them)\n")
    cams = sorted({v["camera"] for v in labels.values()})
    print("| Model | " + " | ".join(cams) + " |")
    print("|---|" + "---|" * len(cams))
    for r in rows:
        fp, neg = collections.Counter(), collections.Counter()
        for line in open(os.path.join(args.results, f"{r['label']}.jsonl")):
            a = json.loads(line)
            if a.get("kind") != "answer" or a.get("task") != "person":
                continue
            gt = labels.get(a["id"]) or {}
            if gt.get("person_ambiguous") or gt.get("people", 1) != 0:
                continue
            neg[gt["camera"]] += 1
            fp[gt["camera"]] += (a.get("text") or "").strip().lower().startswith("yes")
        print(f"| {name(r)} | " + " | ".join(f"{fp[c]}/{neg[c]}" for c in cams) + " |")


if __name__ == "__main__":
    main()
