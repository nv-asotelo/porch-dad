#!/usr/bin/env python3
"""Group every model's caption of each frame into blind judging batches.

Captions are shuffled per frame under neutral letters, so a judge sees "A", "B", "C" and never a
model name; the key that maps letters back to models stays in this file's output, not the prompt.

  python judge_inputs.py --data DIR --labels labels.json --batch 12 --out judge.json results/*.jsonl
"""
import argparse
import json
import os
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--out", required=True)
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()

    labels = json.load(open(args.labels))["items"]
    captions = {}
    for path in args.runs:
        for line in open(path):
            r = json.loads(line)
            if r.get("kind") == "answer" and r.get("task") == "caption" and not r.get("error"):
                captions.setdefault(r["id"], {})[r["label"]] = (r.get("text") or "").strip()

    frames = []
    for fid in sorted(captions):
        rng = random.Random(fid)
        models = sorted(captions[fid])
        rng.shuffle(models)
        letters = [chr(ord("A") + i) for i in range(len(models))]
        gt = labels.get(fid) or {}
        frames.append({"id": fid, "image": os.path.join(os.path.abspath(args.data), "images", f"{fid}.jpg"),
                       "reference": gt.get("caption_ref"), "key_facts": gt.get("key_facts"),
                       "counts": {k: gt.get(k) for k in ("people", "vehicles", "dogs", "cats")},
                       "captions": {letter: captions[fid][m] for letter, m in zip(letters, models)},
                       "key": dict(zip(letters, models))})
    batches = [frames[i:i + args.batch] for i in range(0, len(frames), args.batch)]
    json.dump({"batches": batches}, open(args.out, "w"), indent=1)
    models = sorted({m for f in frames for m in f["key"].values()})
    print(f"{len(frames)} frames, {len(batches)} batches, models: {models}")


if __name__ == "__main__":
    main()
