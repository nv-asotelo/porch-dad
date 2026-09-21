#!/usr/bin/env python3
"""Build the pinned FP16 reasoner engine cache in a process which then exits.

Run with the backend environment's Python, before launching run_backend.sh.
No server is started. Existing published profiles are preserved: use a fresh
COSMOS_CACHE_DIR if another profile or an incomplete matching bundle exists.
Do not run another builder/server cache manager concurrently.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_REVISION = "e8b29522938901f6df19ebeedd4b69bc8edbcd97"
CACHE_CAP = 12 << 30


def absolute_path(value, label, resolve=True):
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(label + " must be an absolute path, matching the server invocation")
    return path.resolve() if resolve else path


def preserve_existing_cache(builder, model, cache, options):
    """Avoid upstream's replacement and LRU deletion paths for existing bundles."""
    bundle = Path(builder.bundle_cache_path(str(model), str(cache), options))
    existing = builder._bundle_directories(str(cache))
    if any(path.resolve() != bundle.resolve() for path in existing):
        raise ValueError("Other cached profiles exist; choose a fresh COSMOS_CACHE_DIR to preserve them")
    if bundle.exists() and not builder._is_ready(str(model), str(bundle), options):
        raise ValueError("Matching cache exists but is incomplete/stale; use a fresh COSMOS_CACHE_DIR; nothing was deleted")
    if sum(builder._directory_size(path) for path in existing) > CACHE_CAP:
        raise ValueError("Existing published cache exceeds 12 GiB; it has been preserved")
    return bundle


def hash_stream(source):
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
        digest.update(chunk)
        size += len(chunk)
    return {"sha256": digest.hexdigest(), "size_bytes": size}


def normalize_runtime_artifacts(model, bundle):
    """Repair the two verified Cosmos3 sidecar omissions after build or cache hit."""
    records = []
    for script in ("repair_cosmos_runtime_config.py", "repair_cosmos_chat_template.py"):
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), "--model", str(model),
             "--bundle", str(bundle), "--apply"],
            capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(script + " failed: " + completed.stderr.strip())
        records.append(json.loads(completed.stdout))
    return records


def build_provenance(upstream):
    """Observe build inputs without allocating a GPU array or changing options."""
    import cupy
    import tensorrt

    upstream = Path(upstream).resolve()
    plugin = upstream / "build/libNvInfer_edgellm_plugin.so"
    with plugin.open("rb") as source:
        plugin_record = {"path": str(plugin.resolve()), **hash_stream(source)}
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    # Include staged and unstaged tracked changes relative to HEAD, including
    # tracked submodule diffs. Keep the patch off stdout and out of the receipt.
    with tempfile.TemporaryFile() as patch:
        subprocess.run(
            ["git", "-C", str(upstream), "diff", "--binary", "--full-index",
             "--no-color", "--no-renames", "--no-ext-diff", "--no-textconv",
             "--src-prefix=a/", "--dst-prefix=b/", "--submodule=diff",
             "--ignore-submodules=untracked", "HEAD", "--"],
            stdout=patch, stderr=subprocess.PIPE, check=True)
        patch.seek(0)
        tracked_diff = hash_stream(patch)
    tracked_diff.update(dirty=bool(tracked_diff["size_bytes"]), base="HEAD",
                        scope="Tracked working tree versus HEAD, including staged changes and tracked submodule diffs; untracked files excluded")
    return {
        "tensorrt_version": tensorrt.__version__, "cupy_version": cupy.__version__,
        "cuda_driver_version": int(cupy.cuda.runtime.driverGetVersion()),
        "cuda_runtime_version": int(cupy.cuda.runtime.runtimeGetVersion()),
        "backend_revision": revision, "backend_tracked_diff": tracked_diff,
        "built_plugin": plugin_record,
        "plugin_environment_override": os.environ.get("EDGELLM_PLUGIN_PATH"),
    }


WORKER = r'''
import importlib.util, json, pathlib, sys
from dataclasses import asdict
from datetime import datetime, timezone
from experimental.server.runtime import engine_build as builder
model, cache, script, receipt, max_input, max_kv = sys.argv[1:]
spec = importlib.util.spec_from_file_location('cosmos_build_model_cache', script)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
import os
options = builder.BuildOptions(max_input_len=int(max_input), max_kv_cache_capacity=int(max_kv), max_batch_size=1,
    max_image_tokens=int(os.environ['COSMOS_MAX_IMAGE_TOKENS']) if os.environ.get('COSMOS_MAX_IMAGE_TOKENS') else None,
    max_image_tokens_per_image=int(os.environ['COSMOS_MAX_IMAGE_TOKENS_PER_IMAGE']) if os.environ.get('COSMOS_MAX_IMAGE_TOKENS_PER_IMAGE') else None)
helper.preserve_existing_cache(builder, pathlib.Path(model), pathlib.Path(cache), options)
started = datetime.now(timezone.utc).isoformat()
provenance_before = helper.build_provenance(pathlib.Path.cwd())
prepared = builder.prepare_model(model, cache, options, max_cache_size_bytes=helper.CACHE_CAP, clear_cache=False)
compatibility_repairs = helper.normalize_runtime_artifacts(model, prepared.bundle_dir)
provenance_after = helper.build_provenance(pathlib.Path.cwd())
cache_bytes = sum(builder._directory_size(path) for path in builder._bundle_directories(cache))
if cache_bytes > helper.CACHE_CAP:
    raise SystemExit('Published cache exceeds 12 GiB; build artifacts were preserved, but no success receipt is issued')
record = {'started_utc':started, 'completed_utc':datetime.now(timezone.utc).isoformat(),
    'backend_revision':helper.EXPECTED_REVISION, 'model_dir':model, 'cache_dir':cache,
    'build_mode':'unquantized FP16 reasoner baseline', 'build_options':asdict(options),
    'max_cache_size_bytes':helper.CACHE_CAP, 'published_cache_bytes':cache_bytes,
    'prepared_model':asdict(prepared), 'server_started':False, 'inference_validated':False,
    'compatibility_repairs':compatibility_repairs,
    'build_provenance':{'before':provenance_before, 'after':provenance_after,
        'changed_during_prepare':provenance_before != provenance_after,
        'note':'Versions use CUDA integer encoding. built_plugin identifies the source-build artifact; any EDGELLM_PLUGIN_PATH override is recorded separately. On a cache hit (built=false), these observations do not establish the original engine build environment.'},
    'note':'The 12 GiB limit concerns published engine files, not GPU memory or temporary compiler disk usage.'}
with pathlib.Path(receipt).open('x') as output:
    output.write(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2),flush=True)
'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(ROOT / "models/cosmos3-edge-reasoner"))
    parser.add_argument("--cache-dir", default=os.environ.get("COSMOS_CACHE_DIR", str(ROOT / "data/engine-cache-chw")))
    args = parser.parse_args(argv)
    try:
        model = absolute_path(args.model, "Model directory")
        cache = absolute_path(args.cache_dir, "Cache directory")
        upstream = absolute_path(os.environ.get("EDGE_LLM_DIR", str(ROOT / "external/TensorRT-Edge-LLM")), "Backend directory")
        # Preserve the venv executable path: resolving its symlink would select
        # the base interpreter and lose the backend's installed Python packages.
        python = absolute_path(os.environ.get("EDGE_LLM_PYTHON", str(upstream / ".venv/bin/python")),
                               "Backend Python", resolve=False)
        max_input = int(os.environ.get("COSMOS_MAX_INPUT_LEN", "1024"))
        max_kv = int(os.environ.get("COSMOS_MAX_KV_CAPACITY", "2048"))
        if max_input < 1 or max_kv < max_input:
            raise ValueError("Require positive input tokens and KV capacity at least as large as the input limit")
        config = json.loads((model / "config.json").read_text())
        if (model / "hf_quant_config.json").exists() or any(
                component.get("quantization_config") for component in
                (config, config.get("text_config") or {}, config.get("vision_config") or {})):
            raise ValueError("This FP16 baseline helper refuses quantized checkpoint metadata")
        environment = dict(os.environ)
        environment.update(EDGE_LLM_DIR=str(upstream), EDGE_LLM_PYTHON=str(python), COSMOS_CACHE_DIR=str(cache),
                           TRT_PACKAGE_DIR="/usr", PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1")
        environment["LD_LIBRARY_PATH"] = "/usr/lib/aarch64-linux-gnu:/usr/local/cuda-13.2/lib64" + (
            ":" + environment["LD_LIBRARY_PATH"] if environment.get("LD_LIBRARY_PATH") else "")
        # The same launcher validation checks Jetson/architecture, exact backend
        # pin, reasoner-only shards, real processed template and TensorRT major.
        subprocess.run(["bash", str(ROOT / "scripts/run_backend.sh"), "--preflight-only", str(model)],
                       env=environment, check=True)
        results = ROOT / "results"
        results.mkdir(exist_ok=True)
        receipt = results / ("model-cache-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8] + ".json")
        subprocess.run([str(python), "-c", WORKER, str(model), str(cache), str(Path(__file__).resolve()),
                        str(receipt), str(max_input), str(max_kv)], cwd=upstream, env=environment, check=True)
        print("Engine-cache process exited successfully. Receipt: " + str(receipt))
        print("Start the server separately with the same absolute model/cache paths and input/KV limits. Inference remains unvalidated.")
        return 0
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print("Model-cache preparation failed: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
