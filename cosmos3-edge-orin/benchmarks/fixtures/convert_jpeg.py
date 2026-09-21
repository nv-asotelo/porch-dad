#!/usr/bin/env python3
"""Create fixed JPEG transport fixtures; preserve original PNGs and answer facts.

Host preparation only: Pillow 12.1.1 (MIT-CMU), installed in a task-local venv.
No model, network or device calls occur here. Use --check to verify frozen bytes.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path

import PIL
from PIL import Image, features


ROOT = Path(__file__).resolve().parent
SETTINGS = {"quality": 90, "subsampling": 0, "optimize": False, "progressive": False}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def converted_files():
    source_bytes = (ROOT / "manifest.json").read_bytes()
    source = json.loads(source_bytes)
    files, cases = {}, []
    for case in source["cases"]:
        png_path = ROOT / case["filename"]
        if png_path.resolve().parent != ROOT or png_path.suffix != ".png":
            raise ValueError("Expected an original PNG directly inside the fixture directory")
        png = png_path.read_bytes()
        if digest(png) != case["sha256"]:
            raise ValueError("Original PNG changed: " + case["filename"])
        with Image.open(io.BytesIO(png)) as image:
            if image.size != (case["width_px"], case["height_px"]):
                raise ValueError("Original PNG dimensions disagree with the source manifest")
            # A fresh RGB image drops metadata without resizing or recoloring.
            rgb = Image.frombytes("RGB", image.size, image.convert("RGB").tobytes())
        encoded = io.BytesIO()
        rgb.save(encoded, format="JPEG", exif=b"", icc_profile=None, **SETTINGS)
        jpeg = encoded.getvalue()
        filename = "jpeg/" + png_path.stem + ".jpg"
        files[filename] = jpeg
        cases.append({"id": case["id"], "filename": filename, "mime_type": "image/jpeg",
                      "width_px": rgb.width, "height_px": rgb.height, "size_bytes": len(jpeg),
                      "sha256": digest(jpeg),
                      "source_png": {"filename": case["filename"], "sha256": case["sha256"]},
                      "prompt": case["prompt"], "expected_facts": case["expected_facts"], "skills": case["skills"]})
    manifest = {
        "schema_version": 1, "suite_id": source["suite_id"] + "-jpeg-v1",
        "source_manifest": {"filename": "manifest.json", "sha256": digest(source_bytes)},
        "authorship": source["authorship"], "fixture_license": source["license"],
        "provenance": "Deterministic JPEG codec derivatives of this project's original synthetic PNG inputs. No model was run; expected facts are answer keys, not observed results.",
        "converter": {"filename": "convert_jpeg.py", "sha256": digest(Path(__file__).read_bytes()),
                      "pillow_version": PIL.__version__, "pillow_license": "MIT-CMU",
                      "jpeg_api_version": features.version_codec("jpg"),
                      "libjpeg_turbo_version": features.version_feature("libjpeg_turbo")},
        "encoding": {"format": "JPEG", "input_mode": "RGB", "dimensions_px": [512, 512],
                     "resize": False, "metadata": "No EXIF or ICC profile", "chroma_sampling": "4:4:4",
                     **SETTINGS},
        "request_settings": source["request_settings"], "quality_policy": source["quality_policy"],
        "comparison_rule": "Use these exact JPEG bytes and the same case prompt across all candidates. Preserve the PNGs. Browser canvas re-encoding produces a different workload.",
        "cases": cases,
    }
    files["jpeg-manifest.json"] = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode()
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify all JPEGs and the manifest without modifying files")
    args = parser.parse_args()
    files = converted_files()
    for name, data in files.items():
        path = ROOT / name
        if args.check:
            if not path.is_file() or path.read_bytes() != data:
                raise SystemExit("Missing or changed derivative/manifest: " + name)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    total = sum(len(data) for name, data in files.items() if name.endswith(".jpg"))
    print(("VERIFIED" if args.check else "GENERATED") + f": six 512x512 JPEGs, {total} bytes total; originals and 19 expected facts unchanged. No model run.")


if __name__ == "__main__":
    main()
