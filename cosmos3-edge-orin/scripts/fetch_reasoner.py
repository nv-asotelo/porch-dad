#!/usr/bin/env python3
"""Fetch or verify only the frozen Cosmos3-Edge reasoner files, with exact hashes.

Uses the public model provider via huggingface_hub in the backend environment.
No shell commands, model execution, TensorRT building or service changes occur.
Receipts are private and created exclusively. Keep original model notices.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "results/model-download.json"
REPOSITORY = "nvidia/Cosmos3-Edge"
REVISION = "344d602b128d1bbdacb43b08d0a3626f46343e29"
MARKER = ".cosmos-reasoner-download.json"


def reference_files(reference: dict) -> dict:
    if (reference.get("repository") not in (REPOSITORY, "https://huggingface.co/" + REPOSITORY)
            or reference.get("revision") != REVISION or reference.get("complete") is not True):
        raise ValueError("Require the complete frozen NVIDIA reasoner reference manifest")
    files = reference["files"]
    if not files or not {"config.json", "model.safetensors.index.json"}.issubset(files):
        raise ValueError("Missing reasoner metadata in reference manifest")
    for name, entry in files.items():
        path = PurePosixPath(name)
        if (not name or path.is_absolute() or ".." in path.parts or "\\" in name
                or str(path) != name or not entry.get("verified")
                or not re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", ""))
                or not isinstance(entry.get("size_bytes"), int) or entry["size_bytes"] < 0):
            raise ValueError("Unsafe or unverified model reference entry: " + name)
    if {"transformer/config.json", "vae/config.json"} & set(files):
        raise ValueError("Reference includes generator component configurations")
    return files


def verify(target: Path, files: dict) -> dict:
    verified = {}
    for name, expected in files.items():
        path = target / name
        if path.is_symlink() or path.resolve() != path or not path.is_file():
            raise ValueError("Missing or symlinked model file: " + name)
        if path.stat().st_size != expected["size_bytes"]:
            raise ValueError("Model file size mismatch: " + name)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
        actual = digest.hexdigest()
        if actual != expected["sha256"]:
            raise ValueError("Model SHA-256 mismatch: " + name)
        verified[name] = {"size_bytes": expected["size_bytes"], "sha256": actual, "verified": True}
    for name in ("transformer/config.json", "vae/config.json"):
        if (target / name).exists():
            raise ValueError("Unexpected generator configuration: " + name)
    config = json.loads((target / "config.json").read_text())
    if config.get("model_type") != "cosmos3_edge" or not config.get("vision_config"):
        raise ValueError("Expected the image/text reasoner config")
    shards = set(json.loads((target / "model.safetensors.index.json").read_text())["weight_map"].values())
    if not shards.issubset(files):
        raise ValueError("Unverified shard named by model index")
    return verified


def prepare_target(target: Path, files: dict, resume: bool) -> None:
    identity = {"repository": REPOSITORY, "revision": REVISION,
                "reference_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}
    if target.is_symlink():
        raise ValueError("Download directory must not be a symlink")
    if target.exists() and any(target.iterdir()):
        marker = target / MARKER
        if not resume or marker.is_symlink() or not marker.is_file() or json.loads(marker.read_text()) != identity:
            raise ValueError("Use a new empty directory, --verify-only, or --resume for this helper's matching partial download")
        if any(path.is_symlink() for path in target.rglob("*")):
            raise ValueError("Refusing symlink in resumed model directory")
        # A resumed download must never overwrite through a supplied symlink.
        for name in files:
            path = target / name
            if path.resolve() != path or path.is_symlink():
                raise ValueError("Refusing symlink in resumed model path: " + name)
    else:
        target.mkdir(parents=True, exist_ok=True)
        (target / MARKER).write_text(json.dumps(identity, indent=2) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New model directory, or matching existing model with --verify-only")
    parser.add_argument("--receipt", type=Path, required=True, help="New private JSON receipt path")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--verify-only", action="store_true", help="No network; hash existing files")
    modes.add_argument("--resume", action="store_true", help="Resume only a matching download created by this helper")
    args = parser.parse_args(argv)
    if args.receipt.exists():
        parser.error("--receipt must be new; preserve previous evidence")
    if args.output.is_symlink():
        parser.error("--output must not be a symlink")
    target = args.output.resolve()
    reference = json.loads(REFERENCE.read_text())
    files = reference_files(reference)
    record = {"repository": REPOSITORY, "revision": REVISION, "complete": False,
              "inference_validated": False, "started_utc": datetime.now(timezone.utc).isoformat(),
              "reference_manifest_sha256": hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
              "files": {}, "errors": []}
    old_umask = os.umask(0o077)
    try:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        # Reserve the receipt before any download; never overwrite a prior run.
        with args.receipt.open("x") as receipt:
            receipt.write(json.dumps(record, indent=2) + "\n")
        try:
            if not args.verify_only:
                prepare_target(target, files, args.resume)
                from huggingface_hub import snapshot_download
                snapshot_download(REPOSITORY, revision=REVISION, local_dir=target,
                                  allow_patterns=sorted(files), max_workers=2)
            record["files"] = verify(target, files)
            record["verified_total_bytes"] = sum(e["size_bytes"] for e in record["files"].values())
            record["complete"] = True
        except Exception as exc:
            record["errors"].append(type(exc).__name__ + ": " + str(exc))
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        args.receipt.write_text(json.dumps(record, indent=2) + "\n")
    finally:
        os.umask(old_umask)
    print(json.dumps({"complete": record["complete"], "verified_files": len(record["files"]),
                      "verified_total_bytes": record.get("verified_total_bytes", 0),
                      "errors": record["errors"], "inference_validated": False}, indent=2))
    return 0 if record["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
