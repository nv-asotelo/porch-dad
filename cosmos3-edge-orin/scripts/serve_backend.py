#!/usr/bin/env python3
"""Launch the pinned public server with a bounded encoder embedding cache.

Invoked by run_backend.sh after its device/checkpoint/template preflight. All other
arguments are passed to experimental.server unchanged, except optional visual
profile flags translated into the public Python BuildOptions API. Requires the encoder-cache
API patch and rebuilt native binding; no engine rebuild is required by this option.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
BACKEND_REVISION = "e8b29522938901f6df19ebeedd4b69bc8edbcd97"
BUDGET_FLAG = "--encoder-embedding-cache-budget-bytes"


def budget_bytes(value):
    try:
        number = int(value)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("Budget must be a non-negative integer byte count") from error
    if number < 0 or number > 2**63 - 1:
        raise argparse.ArgumentTypeError("Budget must fit a non-negative signed 64-bit integer")
    return number


def server_arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(BUDGET_FLAG, type=budget_bytes, default=0,
                        help="Encoder output cache budget in bytes (default: 0, disabled)")
    args, upstream_args = parser.parse_known_args(argv)
    return args.encoder_embedding_cache_budget_bytes, upstream_args + [
        BUDGET_FLAG, str(args.encoder_embedding_cache_budget_bytes)]


def visual_arguments(argv):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    def positive(value):
        number = int(value)
        if number < 1:
            raise argparse.ArgumentTypeError("Visual profile capacities must be positive")
        return number
    parser.add_argument("--max-image-tokens", type=positive)
    parser.add_argument("--max-image-tokens-per-image", type=positive)
    args, remaining = parser.parse_known_args(argv)
    return vars(args), remaining


def configured_model(upstream_args, visual):
    """Translate the two builder-only flags without changing the upstream CLI."""
    from experimental.server.config import parse_server_config
    from experimental.server.runtime.engine_build import BuildOptions
    config = parse_server_config(upstream_args)
    context = config.model.context_cache_config
    if (context.enabled
            or config.api.max_queued_requests != 1 or config.api.reasoning_parser != "none"
            or config.model.clear_engine_cache or config.model.speculative_config):
        raise ValueError("Compact serving requires text context reuse disabled, one queued request, "
                         "reasoning-parser none and an existing non-speculative bundle")
    kwargs = config.model.llm_kwargs()
    kwargs["build_options"] = BuildOptions(max_input_len=config.model.max_input_len,
        max_kv_cache_capacity=config.model.max_kv_cache_capacity,
        max_batch_size=config.model.max_batch_size, **visual)
    return config, kwargs


def serve_visual_profile(upstream_args, visual):
    config, kwargs = configured_model(upstream_args, visual)
    from experimental.server.runtime import engine_build
    bundle = engine_build.bundle_cache_path(config.model.model, config.model.cache_dir, kwargs["build_options"])
    if not engine_build._is_ready(config.model.model, bundle, kwargs["build_options"]):
        raise ValueError("Selected compact engine bundle is not ready; prepare it with build_model_cache.py first")
    from experimental.server.runtime.engine import load_model
    from experimental.server.runtime.engine_client import EngineClient
    from cosmos_runtime import run_http_server
    logging.basicConfig(level=getattr(logging, config.api.log_level.upper()))
    llm = load_model(**kwargs)
    try:
        if Path(llm.bundle_dir).resolve() != Path(bundle).resolve():
            raise RuntimeError("Runtime selected a different visual profile bundle")
        run_http_server(EngineClient(llm, config.api), config.api)
    finally:
        llm.close()


def verify_native_binding(runtime, budget):
    config = runtime.ContextCacheConfig()
    if not hasattr(config, "encoder_embedding_cache_budget_bytes"):
        raise RuntimeError("Native binding lacks encoder_embedding_cache_budget_bytes; apply the pinned patch and rebuild _edgellm_runtime before serving")
    config.encoder_embedding_cache_budget_bytes = budget
    if config.encoder_embedding_cache_budget_bytes != budget:
        raise RuntimeError("Native encoder embedding cache budget did not round-trip")


def main(argv=None):
    visual, remaining = visual_arguments(sys.argv[1:] if argv is None else argv)
    budget, upstream_args = server_arguments(remaining)
    upstream = Path(os.environ.get("EDGE_LLM_DIR", str(ROOT / "external/TensorRT-Edge-LLM"))).resolve()
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != BACKEND_REVISION:
        raise RuntimeError("Backend checkout differs from the inspected v0.10.1 pin")
    sys.path.insert(0, str(upstream))
    from experimental.server.runtime.engine import _import_runtime
    verify_native_binding(_import_runtime(), budget)
    print(f"Requested encoder_embedding_cache_budget_bytes={budget}; independent of text context reuse.",
          file=sys.stderr, flush=True)
    # Even the default profile uses the same native timing/control adapter.
    serve_visual_profile(upstream_args, visual)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print("Backend launch refused: " + str(error), file=sys.stderr)
        raise SystemExit(1)
