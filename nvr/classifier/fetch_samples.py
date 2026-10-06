#!/usr/bin/env python3
"""Fetch the labelled Pokémon sample set into a directory Live Vision serves (--samples-dir).

samples.json lists 189 openly licensed photos (CC0, Public Domain Mark, CC BY, CC BY-SA) of 110
species, with each one's creator, licence and source. The images are not in git: this downloads
each from its source, prepares it the way the set was built - RGB, transparency on white, at most
640 px on the longest side (LANCZOS), JPEG quality 92 - and writes the manifest.json serve_ui.py
reads, carrying each attribution. A file whose SHA-256 differs from the one the set was scored with
is kept and reported: a different Pillow can encode the same picture differently.

  python fetch_samples.py /home/orin/nvr/classifier/samples
"""
import argparse
import hashlib
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent


def prepare(raw: bytes) -> bytes:
    image = Image.open(io.BytesIO(raw))
    image.load()
    if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        image = Image.new("RGB", rgba.size, (255, 255, 255))
        image.paste(rgba, mask=rgba.getchannel("A"))
    image = image.convert("RGB")
    longest = max(image.size)
    if longest > 640:
        scale = 640 / longest
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.LANCZOS)
    out = io.BytesIO()
    image.save(out, "JPEG", quality=92)
    return out.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", type=Path, help="directory to fill, such as /home/orin/nvr/classifier/samples")
    parser.add_argument("--source", type=Path, default=HERE / "samples.json")
    args = parser.parse_args()
    source = json.loads(args.source.read_text())
    args.target.mkdir(parents=True, exist_ok=True)
    images, failed, differs = [], [], []
    for record in source["images"]:
        path = args.target / record["file"]
        if not path.is_file():
            try:
                request = urllib.request.Request(record["image_url"], headers={"User-Agent": "porch-dad sample fetch"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    path.write_bytes(prepare(response.read()))
                time.sleep(0.2)
            except Exception as exc:  # one dead link must not stop the rest
                failed.append(f"{record['file']}: {exc}")
                continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != record.get("sha256"):
            differs.append(record["file"])
        images.append({"id": path.stem.lower(), "file": record["file"], "species": record["species"],
                       "credit": record.get("attribution", ""), "license": record.get("license", ""),
                       "source": record.get("source_url", "")})
    # title: the set's tab in Live Vision; subject: which Name presets and classifiers it is for.
    manifest = {"name": source.get("name", ""), **{k: source[k] for k in ("title", "subject") if k in source},
                "description": source.get("description", ""), "license_note": source.get("license_note", ""),
                "images": images}
    (args.target / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(f"{len(images)} of {len(source['images'])} images in {args.target}")
    if differs:
        print(f"{len(differs)} differ from the scored files (scores may shift slightly): {', '.join(differs[:10])}")
    if failed:
        print("not fetched:\n  " + "\n  ".join(failed), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
