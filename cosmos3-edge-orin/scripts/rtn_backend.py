#!/usr/bin/env python3
"""Build or serve the original CPU RTN candidate with native INT4 V1 kernels.

Run with the backend venv Python on Jetson Orin. Build and serve share an explicit
BuildOptions subclass; its plugin version participates in the cache fingerprint.
Serving requires a ready normalized bundle and never intentionally auto-builds.
The FP16 launcher and its cache profile remain unchanged. No calibration or GPU
quality claim is implied by this helper. Run only one cache manager at a time.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
BACKEND_REVISION = "e8b29522938901f6df19ebeedd4b69bc8edbcd97"
SOURCE_REVISION = "344d602b128d1bbdacb43b08d0a3626f46343e29"
CACHE_CAP = 12 << 30


def build_options(*, max_input_len=1024, max_kv_capacity=2048,
                  max_image_tokens=None, max_image_tokens_per_image=None):
    """One factory used by preflight, builder and serving; no global patch."""
    from experimental.server.runtime.engine_build import BuildOptions

    @dataclass(frozen=True)
    class RtnBuildOptions(BuildOptions):
        int4_gemm_plugin_version: int = 1

        def to_argv(self, model_dir, bundle_dir):
            return super().to_argv(model_dir, bundle_dir) + [
                "--int4-gemm-plugin-version", str(self.int4_gemm_plugin_version)]

    if max_input_len <= 0 or max_kv_capacity < max_input_len:
        raise ValueError("Require 0 < max input length <= max KV capacity")
    image_max = 1024 if max_image_tokens is None else max_image_tokens
    image_per = 512 if max_image_tokens_per_image is None else max_image_tokens_per_image
    if not 4 <= image_per <= image_max:
        raise ValueError("Require 4 <= max image tokens per image <= max image tokens")
    return RtnBuildOptions(max_input_len=max_input_len, max_kv_cache_capacity=max_kv_capacity,
                           max_batch_size=1, max_image_tokens=max_image_tokens,
                           max_image_tokens_per_image=max_image_tokens_per_image)


def validate_scope_layout(record, metadata, index):
    """Verify exact module identities, not only a claimed INT4 tensor count."""
    from quantize_cosmos3_rtn import SCOPE_COUNTS, output_name

    # Existing completed all-linears candidates predate the explicit scope field.
    scope = record.get("quantization_scope", "all-linears")
    if scope not in SCOPE_COUNTS or metadata.get("quantization_scope", "all-linears") != scope:
        raise ValueError("Unknown or inconsistent RTN quantization scope")
    expected = {f"layers.{layer}.mlp.{projection}" for layer in range(28)
                for projection in ("up_proj", "down_proj")}
    if scope == "all-linears":
        expected |= {f"layers.{layer}.self_attn.{projection}" for layer in range(28)
                     for projection in ("q_proj", "k_proj", "v_proj", "o_proj")}
        expected.add("lm_head")
    count = SCOPE_COUNTS[scope]
    if (record.get("quantized_linear_count") != count or record.get("source_tensor_count") != 698
            or record.get("output_tensor_count") != 698 + 2 * count
            or len(index) != 698 + 2 * count or set(index.values()) != {"model.safetensors"}):
        raise ValueError("Unexpected RTN checkpoint tensor counts/layout for " + scope)
    for suffix in (".qweight", ".qzeros", ".scales"):
        if {name[:-len(suffix)] for name in index if name.endswith(suffix)} != expected:
            raise ValueError("Packed RTN modules do not match declared scope: " + suffix)
    excluded = {output_name(name)[:-7].removeprefix("model.") for name in index if name.endswith(".weight")}
    if set(metadata.get("exclude_modules", [])) != excluded:
        raise ValueError("FP16 exclusions do not match actual checkpoint modules")
    if expected & excluded:
        raise ValueError("RTN checkpoint contains both plain and packed forms of a module")
    return scope, count


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_candidate(model):
    """Verify conversion receipt, weight bytes and actual parser dispatch."""
    from experimental.builder.core.quantization import parse_quantization
    import repair_cosmos_chat_template

    repair_cosmos_chat_template.validate_source(model)
    source = json.loads((model / "config.json").read_text())
    record = json.loads((model / "rtn-conversion.json").read_text())
    if (not record.get("complete") or record.get("algorithm") != "round_to_nearest"
            or record.get("dispatch_identifier") != "RTN_GPTQ"
            or record.get("source_revision") != SOURCE_REVISION
            or record.get("backend_revision") != BACKEND_REVISION
            or record.get("calibration") is not None or record.get("group_size") != 128
            or not record.get("source_shards")):
        raise ValueError("Require the complete, pinned original CPU RTN conversion receipt")
    if any((model / part / "config.json").exists() for part in ("transformer", "vae")):
        raise ValueError("Only the Cosmos3 reasoner checkpoint is supported")
    metadata = json.loads((model / "hf_quant_config.json").read_text())["quantization"]
    if (metadata.get("quant_algo") != "RTN_GPTQ" or metadata.get("quantization_algorithm") != "round_to_nearest"
            or metadata.get("calibration") is not None or metadata.get("kv_cache_quant_algo") is not None
            or metadata.get("group_size") != 128 or metadata.get("sym") is not True
            or metadata.get("zero_point_offset") != 1 or metadata.get("checkpoint_format") != "gptq"):
        raise ValueError("Expected uncalibrated RTN_GPTQ storage and FP16 KV cache")
    weights = model / "model.safetensors"
    if (weights.stat().st_size != record["model_safetensors_bytes"]
            or sha256_file(weights) != record["model_safetensors_sha256"]):
        raise ValueError("Candidate weights differ from the completed conversion")
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    scope, count = validate_scope_layout(record, metadata, index)
    quant = parse_quantization(str(model), source, source["text_config"])
    if (quant.quant_type, quant.group_size, quant.gptq_zero_point_offset) != ("int4_gptq", 128, 1):
        raise ValueError("Pinned builder did not select the expected RTN storage contract")
    from quantize_cosmos3_rtn import output_name
    for name in index:
        if name.endswith(".weight") and quant.module_type(output_name(name)[:-7]) != "fp16":
            raise ValueError("Unpacked tensor must remain FP16: " + name)
        if name.endswith(".qweight") and (name.startswith(("model.visual.", "model.projector."))
                                         or quant.module_type(name[:-8]) != "int4_gptq"):
            raise ValueError("Invalid INT4 module dispatch: " + name)
    return {"source_revision": SOURCE_REVISION, "source_shards": record["source_shards"],
            "algorithm": record["algorithm"], "calibration": None,
            "quantization_scope": scope, "quantized_linear_count": count, "output_tensor_count": len(index),
            "candidate_shard": {"path": str(weights), "sha256": record["model_safetensors_sha256"],
                                "size_bytes": weights.stat().st_size},
            "metadata_sha256": {name: sha256_file(model / name) for name in
                ("config.json", "hf_quant_config.json", "model.safetensors.index.json", "rtn-conversion.json")}}


def write_receipt(record):
    folder = ROOT / "results"
    folder.mkdir(exist_ok=True)
    path = folder / ("rtn-" + record["mode"] + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-")
                     + uuid.uuid4().hex[:8] + ".json")
    with path.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("RTN receipt: " + str(path), flush=True)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "serve"))
    parser.add_argument("--model", type=Path, default=ROOT / "models/cosmos3-edge-rtn-int4")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/engine-cache-rtn-v1")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--served-model-name", default="Cosmos3-Edge")
    parser.add_argument("--max-input-len", type=int, default=1024)
    parser.add_argument("--max-kv-capacity", type=int, default=2048)
    parser.add_argument("--max-image-tokens", type=int)
    parser.add_argument("--max-image-tokens-per-image", type=int)
    from serve_backend import budget_bytes
    parser.add_argument("--encoder-embedding-cache-budget-bytes", type=budget_bytes, default=0)
    args = parser.parse_args(argv)
    if not args.model.is_absolute() or not args.cache_dir.is_absolute():
        raise ValueError("Model and cache paths must be absolute")
    model, cache = args.model.resolve(), args.cache_dir.resolve()
    upstream = Path(os.environ.get("EDGE_LLM_DIR", str(ROOT / "external/TensorRT-Edge-LLM"))).resolve()
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise ValueError("This build/serve helper requires Jetson Orin Linux aarch64")
    identity = Path("/proc/device-tree/model").read_bytes().replace(b"\0", b" ").decode()
    if "Jetson" not in identity or "Orin" not in identity:
        raise ValueError("Expected Jetson Orin device-tree identity")
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != BACKEND_REVISION:
        raise ValueError("Backend differs from the pinned v0.10.1 source")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TRT_PACKAGE_DIR="/usr")
    sys.path.insert(0, str(upstream))
    # Upstream resolves its default plugin and native extension from the checkout,
    # matching the FP16 build worker and launcher. Task paths above are absolute.
    os.chdir(upstream)
    import build_model_cache as baseline
    from preflight_cosmos_artifacts import validate
    from experimental.server.runtime import engine_build as builder
    options = build_options(max_input_len=args.max_input_len, max_kv_capacity=args.max_kv_capacity,
                            max_image_tokens=args.max_image_tokens,
                            max_image_tokens_per_image=args.max_image_tokens_per_image)
    candidate = validate_candidate(model)
    # Same explicit subclass selects and verifies the same cache in both modes.
    selected = baseline.preserve_existing_cache(builder, model, cache, options)
    validate(model, upstream, cache, max_input_len=args.max_input_len, max_kv_capacity=args.max_kv_capacity,
             preflight_only=args.mode == "build", build_options=options)
    provenance = baseline.build_provenance(upstream)
    if provenance["tensorrt_version"].split(".")[0] != "10":
        raise ValueError("Expected TensorRT 10 from the installed JetPack")
    record = {"mode": args.mode, "started_utc": datetime.now(timezone.utc).isoformat(),
              "model_dir": str(model), "cache_dir": str(cache), "bundle_dir": str(selected),
              "build_options": asdict(options), "checkpoint_provenance": candidate,
              "provenance_before": provenance, "inference_validated": False,
              "note": "RTN without calibration; GPTQ describes storage only. Vision/projector/embed and KV are FP16."}
    if args.mode == "build":
        prepared = builder.prepare_model(str(model), str(cache), options,
                                         max_cache_size_bytes=CACHE_CAP, clear_cache=False)
        record["compatibility_repairs"] = baseline.normalize_runtime_artifacts(model, prepared.bundle_dir)
        validate(model, upstream, cache, max_input_len=args.max_input_len, max_kv_capacity=args.max_kv_capacity,
                 preflight_only=False, build_options=options)
        record.update(prepared_model=asdict(prepared), server_started=False)
    else:
        # Preflight above requires a ready bundle. LLM receives identical options;
        # no default V2 profile is substituted and no global API is monkeypatched.
        from experimental.server.runtime.engine import load_model, _import_runtime
        from experimental.server.runtime.engine_client import EngineClient
        from cosmos_runtime import run_http_server
        from experimental.server.config import ApiConfig, ContextCacheConfig
        from serve_backend import verify_native_binding
        verify_native_binding(_import_runtime(), args.encoder_embedding_cache_budget_bytes)
        api = ApiConfig(host=args.host, port=args.port, served_model_name=args.served_model_name,
                        max_queued_requests=1)
        context = ContextCacheConfig(enabled=False,
            encoder_embedding_cache_budget_bytes=args.encoder_embedding_cache_budget_bytes)
        logging.basicConfig(level=logging.INFO)
        llm = load_model(model=str(model), cache_dir=str(cache), build_options=options,
                         max_input_len=args.max_input_len, max_kv_cache_capacity=args.max_kv_capacity, max_batch_size=1,
                         engine_cache_max_size_gb=12, clear_engine_cache=False,
                         context_cache_config=context)
        if Path(llm.bundle_dir).resolve() != selected:
            llm.close()
            raise RuntimeError("Runtime selected a different bundle")
        record.update(runtime_initialized=True, http_server_started=False,
                      context_cache=asdict(context))
    record["provenance_after"] = baseline.build_provenance(upstream)
    record["completed_utc"] = datetime.now(timezone.utc).isoformat()
    record["published_cache_bytes"] = sum(builder._directory_size(path) for path in builder._bundle_directories(str(cache)))
    if record["published_cache_bytes"] > CACHE_CAP:
        raise RuntimeError("Published cache exceeds 12 GiB; preserved without a success receipt")
    write_receipt(record)
    if args.mode == "serve":
        try:
            run_http_server(EngineClient(llm, api), api)
        finally:
            llm.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print("RTN backend refused: " + str(error), file=sys.stderr)
        raise SystemExit(1)
