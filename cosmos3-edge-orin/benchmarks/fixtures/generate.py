#!/usr/bin/env python3
"""Render original deterministic geometric quality fixtures with Python stdlib."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import zlib


ROOT = Path(__file__).resolve().parent
SIZE = 512
COLORS = {
    "white": (255, 255, 255),
    "red": (220, 42, 46),
    "blue": (32, 102, 220),
    "green": (31, 154, 77),
    "orange": (238, 126, 30),
    "yellow": (246, 201, 39),
    "purple": (132, 57, 190),
}


class Raster:
    def __init__(self, width=SIZE, height=SIZE, background=(255, 255, 255)):
        self.width, self.height = width, height
        self.pixels = bytearray(bytes(background) * width * height)

    def span(self, y, left, right, color):
        left, right = max(0, left), min(self.width, right)
        if 0 <= y < self.height and right > left:
            offset = 3 * (y * self.width + left)
            self.pixels[offset:offset + 3 * (right - left)] = bytes(color) * (right - left)

    def rectangle(self, box, color):
        left, top, right, bottom = box
        for y in range(top, bottom):
            self.span(y, left, right, color)

    def circle(self, center, radius, color):
        cx, cy = center
        for y in range(cy - radius, cy + radius):
            half_width = math.sqrt(max(0, radius ** 2 - (y + .5 - cy) ** 2))
            self.span(y, math.ceil(cx - half_width - .5), math.ceil(cx + half_width - .5), color)

    def polygon(self, points, color):
        for y in range(math.floor(min(point[1] for point in points)), math.ceil(max(point[1] for point in points))):
            intersections = []
            for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]):
                if min(y1, y2) <= y + .5 < max(y1, y2):
                    intersections.append(x1 + (y + .5 - y1) * (x2 - x1) / (y2 - y1))
            intersections.sort()
            for left, right in zip(intersections[::2], intersections[1::2]):
                self.span(y, math.ceil(left - .5), math.ceil(right - .5), color)

    def paste(self, image, left, top):
        for y in range(image.height):
            source = y * image.width * 3
            target = ((top + y) * self.width + left) * 3
            self.pixels[target:target + image.width * 3] = image.pixels[source:source + image.width * 3]

    def png(self):
        def chunk(kind, data):
            return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
        rows = b"".join(b"\x00" + self.pixels[y * self.width * 3:(y + 1) * self.width * 3]
                        for y in range(self.height))
        header = struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0)
        return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b"")


def scenes():
    specifications = [
        {
            "id": "01-left-right",
            "filename": "01-left-right.png",
            "prompt": "Describe the shape and color on the left and on the right. Which shape is left of the other?",
            "expected_facts": ["The left shape is a red circle.", "The right shape is a blue square.",
                               "The red circle is to the left of the blue square."],
            "objects": [
                {"shape": "circle", "color": "red", "center_px": [142, 256], "radius_px": 70},
                {"shape": "square", "color": "blue", "box_px": [306, 186, 446, 326]},
            ],
            "skills": ["shape", "color", "left-right"],
        },
        {
            "id": "02-counts",
            "filename": "02-counts.png",
            "prompt": "How many green circles and how many orange squares are visible? How many shapes are there in total?",
            "expected_facts": ["There are three green circles.", "There are two orange squares.",
                               "There are five shapes in total."],
            "objects": [
                {"shape": "circle", "color": "green", "center_px": [x, 168], "radius_px": 44}
                for x in [112, 256, 400]
            ] + [{"shape": "square", "color": "orange", "box_px": [x - 44, 308, x + 44, 396]}
                 for x in [184, 328]],
            "skills": ["counting", "shape", "color"],
        },
        {
            "id": "03-above-below",
            "filename": "03-above-below.png",
            "prompt": "Identify the shape and color at the top and at the bottom. Is the yellow shape above or below the purple shape?",
            "expected_facts": ["The upper shape is a yellow triangle.", "The lower shape is a purple circle.",
                               "The yellow triangle is above the purple circle."],
            "objects": [
                {"shape": "triangle", "color": "yellow", "points_px": [[256, 66], [170, 216], [342, 216]]},
                {"shape": "circle", "color": "purple", "center_px": [256, 364], "radius_px": 70},
            ],
            "skills": ["shape", "color", "above-below"],
        },
        {
            "id": "04-inside-outside",
            "filename": "04-inside-outside.png",
            "prompt": "Is the red circle inside or outside the blue outlined square? Is the green triangle inside or outside it, and on which side?",
            "expected_facts": ["The red circle is inside the blue outlined square.",
                               "The green triangle is outside the blue outlined square.",
                               "The green triangle is to the right of the blue outlined square."],
            "objects": [
                {"shape": "outlined_square", "color": "blue", "box_px": [48, 132, 296, 380], "stroke_px": 12},
                {"shape": "circle", "color": "red", "center_px": [172, 256], "radius_px": 46},
                {"shape": "triangle", "color": "green", "points_px": [[396, 196], [340, 300], [452, 300]]},
            ],
            "skills": ["containment", "left-right"],
        },
        {
            "id": "05-relative-size",
            "filename": "05-relative-size.png",
            "prompt": "What shape and color do both objects share? Is the left object or the right object larger?",
            "expected_facts": ["Both objects are circles.", "Both objects are purple.",
                               "The circle on the left is larger than the circle on the right."],
            "objects": [
                {"shape": "circle", "color": "purple", "center_px": [152, 256], "radius_px": 88},
                {"shape": "circle", "color": "purple", "center_px": [382, 256], "radius_px": 38},
            ],
            "skills": ["shape", "color", "relative-size"],
        },
        {
            "id": "06-quadrants",
            "filename": "06-quadrants.png",
            "prompt": "Give the color and shape in each position: top left, top right, bottom left, and bottom right.",
            "expected_facts": ["The top-left shape is a red square.", "The top-right shape is a green circle.",
                               "The bottom-left shape is a blue triangle.", "The bottom-right shape is an orange star."],
            "objects": [
                {"shape": "square", "color": "red", "box_px": [78, 78, 178, 178]},
                {"shape": "circle", "color": "green", "center_px": [384, 128], "radius_px": 53},
                {"shape": "triangle", "color": "blue", "points_px": [[128, 322], [70, 426], [186, 426]]},
                {"shape": "star", "color": "orange", "points_px": [[384, 320], [399, 363], [445, 364],
                  [409, 392], [422, 436], [384, 410], [346, 436], [359, 392], [323, 364], [369, 363]]},
            ],
            "skills": ["shape", "color", "quadrant-position"],
        },
    ]
    for case in specifications:
        image = Raster()
        for item in case["objects"]:
            color = COLORS[item["color"]]
            if item["shape"] == "circle":
                image.circle(item["center_px"], item["radius_px"], color)
            elif item["shape"] in {"square", "outlined_square"}:
                image.rectangle(item["box_px"], color)
                if item["shape"] == "outlined_square":
                    x1, y1, x2, y2 = item["box_px"]
                    inset = item["stroke_px"]
                    image.rectangle((x1 + inset, y1 + inset, x2 - inset, y2 - inset), COLORS["white"])
            else:
                image.polygon(item["points_px"], color)
        yield case, image


def contact_sheet(images):
    sheet = Raster(1616, 1140, (228, 234, 240))
    digits = ["010101001001111", "110001010100111", "110001010001110", "101101111001001",
              "111100110001110", "011100110101010"]
    for index, image in enumerate(images):
        left, top = 20 + (index % 3) * 532, 20 + (index // 3) * 560
        for y in range(5):
            for x in range(3):
                if digits[index][y * 3 + x] == "1":
                    sheet.rectangle((left + x * 3, top + y * 3, left + x * 3 + 3, top + y * 3 + 3), (35, 49, 65))
        sheet.paste(image, left, top + 28)
    return sheet


def generate():
    files = {}
    entries = []
    images = []
    for case, image in scenes():
        encoded = image.png()
        files[case["filename"]] = encoded
        entries.append({**case, "width_px": SIZE, "height_px": SIZE, "mime_type": "image/png",
                        "sha256": hashlib.sha256(encoded).hexdigest(),
                        "rgb_pixel_sha256": hashlib.sha256(image.pixels).hexdigest()})
        images.append(image)
    files["contact-sheet.png"] = contact_sheet(images).png()
    manifest = {
        "schema_version": 1,
        "suite_id": "original-geometric-quality-v1",
        "title": "Original geometric visual quality fixtures",
        "authorship": "Original scenes, ground-truth facts and renderer authored by Codex for this Jetson deployment project on 2026-09-19.",
        "license": "Apache-2.0; see the repository LICENSE",
        "provenance": {
            "source": "Original deterministic procedural geometry; no downloaded artwork, dataset images or model outputs.",
            "generator": "generate.py",
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "dependencies": "Python standard library only; no image-generation model or runtime imaging package.",
            "randomness": "None. Fixed coordinates, colors and prompts; no timestamps or random seeds in generated images.",
        },
        "request_settings": {"model_alias": "Cosmos3-Edge", "max_tokens": 64,
                             "temperature": 0, "top_p": 1, "concurrency": 1,
                             "stream": True, "stream_options": {"include_usage": True}},
        "quality_policy": {
            "required_facts": sum(len(case["expected_facts"]) for case in entries),
            "scoring": "Human semantic review: award one point per fully correct expected fact, regardless of wording; missing, hedged or contradicted required facts receive zero.",
            "pass_rule": "All 19 required facts correct, all six responses nonempty and completed without errors, and no materially false added visual claim.",
            "accepted_equivalents": ["circle / circular disk", "square / equal-sided box", "violet / purple", "upper / top", "lower / bottom"],
            "review_evidence": "Record candidate/revision, fixture hashes, exact submitted prompts, raw outputs, per-fact scores, reviewer and false claims. Do not substitute expected facts for model output.",
            "limitations": "A basic geometric regression screen. Passing does not prove natural-image/video understanding, temporal reasoning, safety, general accuracy or GPU performance.",
        },
        "contact_sheet": {"filename": "contact-sheet.png", "sha256": hashlib.sha256(files["contact-sheet.png"]).hexdigest(),
                          "layout": "Cases 01, 02, 03 across the top; 04, 05, 06 across the bottom. Do not submit this sheet as a case."},
        "cases": entries,
    }
    files["manifest.json"] = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode()
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Compare existing artifacts with a fresh deterministic rendering")
    args = parser.parse_args()
    files = generate()
    failures = []
    for name, data in files.items():
        path = ROOT / name
        if args.check:
            if not path.is_file() or path.read_bytes() != data:
                failures.append(name)
        else:
            path.write_bytes(data)
    if failures:
        print("MISSING OR CHANGED: " + ", ".join(failures))
        return 1
    print(("VERIFIED" if args.check else "GENERATED") + ": six 512x512 fixtures, contact sheet and hashed manifest; no model was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
