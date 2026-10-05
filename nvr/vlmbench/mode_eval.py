#!/usr/bin/env python3
"""Run a detection mode (nvr/feed/detection_modes.py) over a folder of frames, on whatever engine
answers at --url, and save every verdict - plus, optionally, a contact sheet with the boxes drawn.

  python3 mode_eval.py --url http://127.0.0.1:8000 --mode bernese --images DIR --out run.jsonl \
      [--always-ground] [--sheet sheet.jpg] [--label NAME]

--always-ground also asks for the box when the model says no, so a grounding score can be taken
on every frame a labeller marked positive, whatever the presence answer was.
"""
import argparse
import glob
import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "feed"))
import detection_modes as dm  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--mode", default="bernese")
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--always-ground", action="store_true")
    ap.add_argument("--sheet")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.images, "*.jpg")))
    engine = dm.model_id(args.url)
    fam = dm.family(engine)
    name = dm.MODES[args.mode]["name"]
    rows = []
    with open(args.out, "w") as out:
        out.write(json.dumps({"kind": "meta", "label": args.label, "engine": engine, "family": fam,
                              "mode": args.mode, "items": len(files), "started": time.time()}) + "\n")
        for f in files:
            img = open(f, "rb").read()
            v = dm.check(args.url, img, args.mode, min_confidence=0.5)
            if args.always_ground and fam not in ("locateanything", "text") and not v["found"] and not v["error"]:
                t0 = time.time()
                ans = dm._chat(args.url, img, dm.ground_prompt(name, fam), 48, False, 60)["message"]["content"]
                v["answers"].append(ans)
                v["box_if_asked"] = dm.parse_box(ans)
                v["ms"] += int((time.time() - t0) * 1000)
            v.update(kind="answer", id=os.path.basename(f))
            out.write(json.dumps(v) + "\n")
            rows.append(v)
            print(f"{v['id']}: found={v['found']} p={v['confidence']} box={v['box']} {v['ms']} ms "
                  f"{v['error'] or ''}", flush=True)

    if args.sheet and rows:
        from PIL import Image
        thumbs = []
        for f, v in zip(files, rows):
            box = v["box"] or v.get("box_if_asked")
            tag = f"{'YES' if v['found'] else 'no'} {'' if v['confidence'] is None else round(v['confidence'], 2)}"
            im = Image.open(io.BytesIO(dm.annotate(open(f, "rb").read(), box, f"{v['id'][:-4]} {tag}")))
            thumbs.append(im.resize((640, 360)))
        cols = 3
        sheet = Image.new("RGB", (cols * 640, ((len(thumbs) + cols - 1) // cols) * 360))
        for i, im in enumerate(thumbs):
            sheet.paste(im, ((i % cols) * 640, (i // cols) * 360))
        sheet.save(args.sheet, quality=80)


if __name__ == "__main__":
    main()
