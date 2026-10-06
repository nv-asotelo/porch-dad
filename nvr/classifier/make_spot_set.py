#!/usr/bin/env python3
"""Build a "spot" sample set for Live Vision from labelled frames: whether one target is in each
picture, and where. Run it where the frames are, copy the output directory to the Orin, and add it
to Live Vision's --samples-dir: it shows as a tab beside the Pokémon set. The frames and labels are
yours, so they stay out of git.

  python make_spot_set.py FRAMES_DIR LABELS.json OUT_DIR --target bernese-mountain-dog \
      --target-name "your dog" --title "Doggy daycare" --subject dog \
      [--candidates CANDIDATES.json] [--where daycare] [--id daycare] [--credit TEXT] [--license TEXT]

LABELS.json:     {file: {"boxes": [[x1, y1, x2, y2], ...], ...}}, the target's boxes as fractions of
                 the picture, empty where it is not in it. --where KEY keeps only the files whose
                 label has KEY true.
CANDIDATES.json: {file: [{"box": [x1, y1, x2, y2]}, ...]}, every box a detector drew around something
                 of the target's kind. The page classifies each on its own, as the Bernese mode does.

A caption is the frame's time when its name ends in a valid HHMMSS ("e-131430.jpg" -> "13:14:30"),
else its name. OUT_DIR must be new, empty or an earlier spot set; it gets a .gitignore of its own, so
the frames stay out of git wherever it is written. The manifest is checked with Live Vision's own
loader before the script finishes.
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def caption(stem):
    match = re.search(r"(\d\d)(\d\d)(\d\d)$", stem)
    if match and int(match[1]) < 24 and int(match[2]) < 60 and int(match[3]) < 60:
        return ":".join(match.groups())
    return stem


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames", type=Path)
    ap.add_argument("labels", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--target", required=True, help="the target's species key, as the classifiers name it")
    ap.add_argument("--target-name", default="the target", help='how the page names it, such as "your dog"')
    ap.add_argument("--title", required=True, help="the tab's title")
    ap.add_argument("--subject", default=None, help='"dog", "pokemon": which presets and classifiers it is for')
    ap.add_argument("--id", default=None, help="the set's id (default: the output directory's name)")
    ap.add_argument("--candidates", type=Path, default=None)
    ap.add_argument("--where", default=None, help="keep only the files whose label has this key true")
    ap.add_argument("--credit", default="", help="shown when you hover a tile")
    ap.add_argument("--license", default="private: not for redistribution")
    args = ap.parse_args()
    set_id = args.id or args.out.name
    labels = json.loads(args.labels.read_text())
    candidates = json.loads(args.candidates.read_text()) if args.candidates else {}
    def boxes(raw):
        """Clamped into the picture (detectors overshoot its edges by a hair) and rounded; a box that
        collapses is dropped."""
        out = []
        for box in raw:
            x1, y1, x2, y2 = (round(min(1.0, max(0.0, float(v))), 4) for v in box)
            if x1 < x2 and y1 < y2:
                out.append([x1, y1, x2, y2])
        return out
    if args.out.is_dir() and any(args.out.iterdir()) and not (args.out / "manifest.json").is_file():
        sys.exit(f"{args.out} holds other files: give a new directory, or an earlier spot set's")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / ".gitignore").write_text("# A spot set: your own frames and labels, never committed.\n*\n")
    images = []
    for name in sorted(labels):
        label = labels[name]
        if args.where and not label.get(args.where):
            continue
        source = args.frames / name
        if not source.is_file():
            sys.exit(f"{source} is missing")
        shutil.copy2(source, args.out / name)
        stem = Path(name).stem
        images.append({"id": re.sub(r"[^a-z0-9_.-]", "-", f"{set_id}-{stem}".lower()), "file": name,
                       "caption": caption(stem), "boxes": boxes(label.get("boxes") or []),
                       "candidates": boxes(c["box"] for c in candidates.get(name, [])),
                       "credit": args.credit, "license": args.license})
    manifest = {"id": set_id, "title": args.title, "task": "spot", "target": args.target,
                "target_name": args.target_name, "images": images}
    if args.subject:
        manifest["subject"] = args.subject
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    sys.path.insert(0, str(HERE.parent / "ui" / "scripts"))
    import serve_ui
    loaded = serve_ui.load_sample_set(args.out)       # Live Vision's own check
    present = sum(bool(image["boxes"]) for image in images)
    print(f"{args.out}: {len(loaded['images'])} pictures, {args.target_name} in {present}, "
          f"{sum(len(image['candidates']) for image in images)} candidate boxes")


if __name__ == "__main__":
    main()
