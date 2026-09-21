#!/usr/bin/env python3
"""Normalize two missing Cosmos3 visual runtime aliases; dry-run unless --apply.

Stop the server and cache builders before applying. Only bundle/visual/config.json
is replaced; source checkpoints and engines are never written. Backups and receipts
are saved under this project's results/compatibility directory.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import uuid


ROOT = Path(__file__).resolve().parents[1]
SOURCE_REVISION = "344d602b128d1bbdacb43b08d0a3626f46343e29"
# Verified against the official Git blob in results/model-download.json.
SOURCE_CONFIG_SHA256 = "ffb75b654e65d10e337ea07f20ac4eadc3775dd22c9a8194c87c3b68df5c8f5f"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def plan_repair(source, artifact):
    """Return a copy and explicit additions, refusing foreign or conflicting data."""
    if source.get("model_type") != "cosmos3_edge":
        raise ValueError("Source model_type must be cosmos3_edge")
    if artifact.get("model_type") != "cosmos3_edge_vision":
        raise ValueError("Artifact model_type must be cosmos3_edge_vision")
    text = source.get("text_config")
    actual_text = artifact.get("text_config")
    vision = source.get("vision_config")
    actual_vision = artifact.get("vision_config")
    if not all(isinstance(value, dict) for value in (text, actual_text, vision, actual_vision)):
        raise ValueError("Source and artifact require text_config and vision_config objects")
    for section, expected, actual in (("text_config", text, actual_text),
                                       ("vision_config", vision, actual_vision)):
        for key, value in expected.items():
            if key not in actual or actual[key] != value:
                raise ValueError(f"Artifact {section}.{key} differs from source")
    rope = text.get("rope_parameters")
    if not isinstance(rope, dict) or rope != {
            "mrope_section": [24, 20, 20], "rope_theta": 100000000, "rope_type": "default"}:
        raise ValueError("Source rope_parameters do not match the pinned Cosmos3 contract")
    if vision.get("num_patches") != 256:
        raise ValueError("Source vision num_patches must be 256")
    scaling = copy.deepcopy(rope)
    scaling["type"] = scaling["rope_type"]  # Same alias as upstream normalize_rope_scaling.
    aliases = {"rope_theta": rope["rope_theta"], "rope_scaling": scaling}
    repaired = copy.deepcopy(artifact)
    additions = {}
    for key, value in aliases.items():
        if key in actual_text:
            if actual_text[key] != value:
                raise ValueError(f"Conflicting existing text_config.{key}; nothing changed")
        else:
            repaired["text_config"][key] = value
            additions["text_config." + key] = value
    return repaired, additions


def absolute_directory(value, label):
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(label + " must be an absolute path")
    path = path.resolve(strict=True)
    if not path.is_dir():
        raise ValueError(label + " must be a directory")
    return path


def atomic_write(path, data, mode=0o600):
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            os.fchmod(output.fileno(), mode)
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def exclusive_write(path, data):
    with path.open("xb") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())


def repair(model, bundle, *, apply=False, results_dir=None):
    model = absolute_directory(model, "--model")
    bundle = absolute_directory(bundle, "--bundle")
    if bundle == model or model in bundle.parents or bundle in model.parents:
        raise ValueError("Model and bundle must be separate, non-nested directories")
    source_path = model / "config.json"
    target = bundle / "visual/config.json"
    if target.is_symlink() or target.resolve(strict=True).parent != bundle / "visual":
        raise ValueError("Refusing a symlinked visual config or visual directory")
    source_bytes = source_path.read_bytes()
    before = target.read_bytes()
    source, artifact = json.loads(source_bytes), json.loads(before)
    repaired, additions = plan_repair(source, artifact)
    if sha256(source_bytes) != SOURCE_CONFIG_SHA256:
        raise ValueError("Source config SHA256 differs from the verified official pinned config")
    after = json_bytes(repaired) if additions else before
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "dry_run" if additions else "already_normalized",
        "source_config": str(source_path), "source_config_sha256": sha256(source_bytes),
        "source_reference": "nvidia/Cosmos3-Edge", "source_revision": SOURCE_REVISION,
        "artifact": str(target), "before_sha256": sha256(before), "after_sha256": sha256(after),
        "additions": additions, "engine_files_modified": False,
        "note": "Only config aliases are normalized. No engine rebuild or inference validation is performed.",
    }
    if not apply or not additions:
        return record
    results = Path(results_dir) if results_dir is not None else ROOT / "results/compatibility"
    results.mkdir(parents=True, exist_ok=True)
    stem = "cosmos-rope-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex
    backup = results / (stem + ".visual-config.before.json")
    receipt = results / (stem + ".receipt.json")
    record.update(status="prepared", backup=str(backup.resolve()), receipt=str(receipt.resolve()))
    exclusive_write(backup, before)
    exclusive_write(receipt, json_bytes(record))
    # Preserve concurrent edits rather than overwriting them. Call only while cache writers are stopped.
    if target.is_symlink() or target.read_bytes() != before or source_path.read_bytes() != source_bytes:
        raise ValueError("Config changed during preparation; no artifact replacement; prepared receipt retained")
    atomic_write(target, after, stat.S_IMODE(target.stat().st_mode))
    record.update(status="applied", applied_utc=datetime.now(timezone.utc).isoformat())
    atomic_write(receipt, json_bytes(record))
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Absolute directory of the unchanged official source checkpoint")
    parser.add_argument("--bundle", required=True, help="Absolute prepared engine bundle directory containing visual/config.json")
    parser.add_argument("--apply", action="store_true", help="Back up and apply the exact displayed additions")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(repair(args.model, args.bundle, apply=args.apply), indent=2))
        return 0
    except (OSError, ValueError, TypeError, AttributeError) as error:
        print("Cosmos runtime config repair refused: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
