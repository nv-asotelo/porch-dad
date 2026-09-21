#!/usr/bin/env python3
"""Restore source-faithful Cosmos3 image/video template formats; default dry-run.

Use --apply only while the server and cache builders are stopped. The original
Jinja, model and tokenizer metadata are verified against their downloaded pins.
Only the bundle's processed_chat_template.json is changed; model files, roles,
generation prompts and engines are preserved.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import stat
import sys
import uuid

import repair_cosmos_runtime_config as common


# Verified against official Git blob hashes in results/model-download.json.
SOURCE_HASHES = {
    "config.json": common.SOURCE_CONFIG_SHA256,
    "chat_template.jinja": "7120ee6666468d4e9b2dc11e133ac5c2fa765fa5907706bf0f906270aa5510c8",
    "tokenizer_config.json": "85577f01be64ea03ee9aacc0536088e47b96971e4aff21e0aa604900b1410386",
}
FORMATS = {
    "image": "<|vision_start|><|image_pad|><|vision_end|>",
    "video": "<|vision_start|><|video_pad|><|vision_end|>",
}
SPECIAL_TOKENS = {
    "video_token_id": (18, "<|video_pad|>"), "image_token_id": (19, "<|image_pad|>"),
    "vision_start_token_id": (20, "<|vision_start|>"), "vision_end_token_id": (21, "<|vision_end|>"),
}


def validate_source(model):
    originals = {name: (model / name).read_bytes() for name in SOURCE_HASHES}
    for name, data in originals.items():
        if common.sha256(data) != SOURCE_HASHES[name]:
            raise ValueError(f"Source {name} SHA256 differs from the verified official pin")
    config = json.loads(originals["config.json"])
    tokenizer = json.loads(originals["tokenizer_config.json"])
    if config.get("model_type") != "cosmos3_edge":
        raise ValueError("Source model_type must be cosmos3_edge")
    decoder = tokenizer.get("added_tokens_decoder", {})
    for key, (token_id, text) in SPECIAL_TOKENS.items():
        token = decoder.get(str(token_id), {})
        if (config.get(key) != token_id or token.get("content") != text
                or token.get("special") is not True):
            raise ValueError(f"Source special token contract differs for {key}")
    return originals


def plan_repair(artifact):
    if not isinstance(artifact.get("roles"), dict) or not isinstance(artifact.get("generation_prompt"), str):
        raise ValueError("Processed template must already contain roles and generation_prompt")
    content_types = artifact.get("content_types", {})
    if not isinstance(content_types, dict):
        raise ValueError("content_types must be an object")
    repaired = copy.deepcopy(artifact)
    additions, previous = {}, {}
    for kind, expected in FORMATS.items():
        entry = content_types.get(kind, {})
        if not isinstance(entry, dict):
            raise ValueError(f"content_types.{kind} must be an object")
        actual = entry.get("format")
        if actual == expected:
            continue
        if actual not in (None, ""):
            raise ValueError(f"Conflicting nonempty content_types.{kind}.format; nothing changed")
        key = f"content_types.{kind}.format"
        previous[key] = {"present": "format" in entry, "value": actual}
        repaired.setdefault("content_types", {}).setdefault(kind, {})["format"] = expected
        additions[key] = expected
    return repaired, additions, previous


def repair(model, bundle, *, apply=False, results_dir=None):
    model = common.absolute_directory(model, "--model")
    bundle = common.absolute_directory(bundle, "--bundle")
    if bundle == model or model in bundle.parents or bundle in model.parents:
        raise ValueError("Model and bundle must be separate, non-nested directories")
    target = bundle / "processed_chat_template.json"
    if target.is_symlink() or target.resolve(strict=True).parent != bundle:
        raise ValueError("Refusing a symlinked processed template")
    originals = validate_source(model)
    before = target.read_bytes()
    repaired, additions, previous = plan_repair(json.loads(before))
    after = common.json_bytes(repaired) if additions else before
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "dry_run" if additions else "already_normalized",
        "source_model": str(model), "source_reference": "nvidia/Cosmos3-Edge",
        "source_revision": common.SOURCE_REVISION,
        "source_sha256": {name: common.sha256(data) for name, data in originals.items()},
        "special_tokens": {key: {"id": value[0], "token": value[1]} for key, value in SPECIAL_TOKENS.items()},
        "artifact": str(target), "before_sha256": common.sha256(before), "after_sha256": common.sha256(after),
        "additions": additions, "previous_values": previous, "engine_files_modified": False,
        "note": "Formats match the pinned Jinja with add_vision_id disabled. Roles/generation prompts are preserved. No inference validation is performed.",
    }
    if not apply or not additions:
        return record
    results = Path(results_dir) if results_dir is not None else common.ROOT / "results/compatibility"
    results.mkdir(parents=True, exist_ok=True)
    stem = "cosmos-media-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex
    backup = results / (stem + ".processed-template.before.json")
    receipt = results / (stem + ".receipt.json")
    record.update(status="prepared", backup=str(backup.resolve()), receipt=str(receipt.resolve()))
    common.exclusive_write(backup, before)
    common.exclusive_write(receipt, common.json_bytes(record))
    if target.is_symlink() or target.read_bytes() != before or any(
            (model / name).read_bytes() != data for name, data in originals.items()):
        raise ValueError("Source/template changed during preparation; no replacement; prepared receipt retained")
    common.atomic_write(target, after, stat.S_IMODE(target.stat().st_mode))
    record.update(status="applied", applied_utc=datetime.now(timezone.utc).isoformat())
    common.atomic_write(receipt, common.json_bytes(record))
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Absolute unchanged official source checkpoint directory")
    parser.add_argument("--bundle", required=True, help="Absolute prepared engine bundle directory")
    parser.add_argument("--apply", action="store_true", help="Back up and apply the exact source-faithful formats")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(repair(args.model, args.bundle, apply=args.apply), indent=2))
        return 0
    except (OSError, ValueError, TypeError, AttributeError) as error:
        print("Cosmos chat template repair refused: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
