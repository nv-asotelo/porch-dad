#!/usr/bin/env python3
"""One guarded Cosmos3 AWQ candidate using the pinned public quantizer.

Default is a CPU configuration/processor preflight. --run loads the GPU model
only after actual baseline evidence and a local, licensed image corpus exist.
All task paths must resolve inside COSMOS_AUX_DIR. No packages are installed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SOURCE_PIN = "e8b29522938901f6df19ebeedd4b69bc8edbcd97"
MODEL_PIN = "344d602b128d1bbdacb43b08d0a3626f46343e29"
PARAMETERS = 2_435_620_080


def contained(root, value):
    path = Path(value).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f"Path is outside the task directory: {value}")
    return path


def json_file(root, value):
    return json.loads(contained(root, value).read_text())


def corpus_rows(root, manifest, samples):
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if len(rows) < samples:
        raise ValueError(f"Need at least {samples} distinct calibration images")
    selected, seen = [], set()
    for row in rows[:samples]:
        image = contained(root, manifest.parent / row["image"])
        if not row.get("source") or not row.get("license") or not row.get("question", "").strip():
            raise ValueError("Each image needs a source, license, and question")
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        if digest != row.get("sha256") or digest in seen:
            raise ValueError("Calibration digest mismatch or duplicate image")
        seen.add(digest)
        selected.append((image, row["question"]))
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--gate", help="JSON containing actual FP16 success or memory-failure evidence")
    parser.add_argument("--images", help="Local calibration JSONL, never the held-out evaluation set")
    parser.add_argument("--samples", type=int, default=32, choices=range(32, 129), metavar="32..128")
    args = parser.parse_args()
    root = Path(args.task_dir).resolve()
    if not root.is_dir() or root == Path("/") or os.environ.get("COSMOS_AUX_DIR") != str(root):
        raise ValueError("Source awq_isolated_env.sh with this exact existing task directory first")
    project = contained(root, Path(__file__).resolve().parent.parent)
    contained(root, Path.cwd())
    # The isolated venv's interpreter symlink may point to system Python; its
    # prefix, site packages, and all mutable environment paths must be task-local.
    contained(root, sys.prefix)
    for name in ("TMPDIR", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "HF_HOME", "HF_HUB_CACHE",
                 "HF_TOKEN_PATH", "HF_DATASETS_CACHE", "HF_MODULES_CACHE", "TORCH_HOME",
                 "TORCH_EXTENSIONS_DIR", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR",
                 "CUDA_CACHE_PATH", "PIP_CACHE_DIR", "PYTHONPYCACHEPREFIX", "GIT_CONFIG_GLOBAL"):
        contained(root, os.environ[name])
    source = contained(root, project / "external/TensorRT-Edge-LLM")
    model = contained(root, project / "models/cosmos3-edge-reasoner")
    for path in model.rglob("*"):
        contained(root, path)
    pin = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if pin != SOURCE_PIN:
        raise ValueError("Upstream source pin differs")
    lock = json_file(root, project / "sources.lock.json")
    if not any(s.get("name") == "Cosmos3-Edge" and s.get("commit") == MODEL_PIN for s in lock["sources"]):
        raise ValueError("Model provenance pin differs")
    cfg = json_file(root, model / "config.json")
    if cfg.get("model_type") != "cosmos3_edge" or cfg.get("auto_map"):
        raise ValueError("Expected the pinned built-in Cosmos3 configuration without remote modeling code")
    if any((model / p).exists() for p in ("transformer/config.json", "vae/config.json")):
        raise ValueError("Generator configurations are forbidden in this reasoner snapshot")
    for shard in set(json_file(root, model / "model.safetensors.index.json")["weight_map"].values()):
        if not contained(root, model / shard).is_file():
            raise ValueError(f"Missing indexed shard: {shard}")
    report_dir = contained(root, project / "results")
    report_dir.mkdir(exist_ok=True)
    report = {"source_pin": pin, "model_pin": MODEL_PIN, "quantization_pipeline_invoked": False,
              "status": "preflight", "samples": args.samples, "started_unix": time.time()}
    destination = report_dir / ("awq-candidate.json" if args.run else "awq-preflight.json")
    torch = None
    try:
        import importlib.metadata as metadata
        from transformers import AutoConfig, AutoProcessor, AutoModelForImageTextToText
        for package, expected in (("torch", "2.13.0"), ("transformers", "5.14.1"), ("nvidia-modelopt", "0.45.0")):
            version = metadata.version(package)
            report[package] = version
            if version.split("+")[0] != expected:
                raise ValueError(f"{package} must match upstream pin {expected}; found {version}")
        config = AutoConfig.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)
        if type(config) not in AutoModelForImageTextToText._model_mapping:
            raise RuntimeError("Installed Transformers has no Cosmos3 ImageTextToText calibration class")
        processor = AutoProcessor.from_pretrained(str(model), local_files_only=True, trust_remote_code=False)
        if not callable(getattr(processor, "apply_chat_template", None)):
            raise RuntimeError("Cosmos3 processor does not provide multimodal chat preprocessing")
        report["config_class"] = type(config).__name__
        report["processor_class"] = type(processor).__name__
        if not args.run:
            report["status"] = "configuration_preflight_passed_gpu_unverified"
            return
        if not args.gate or not args.images:
            raise ValueError("--run needs actual --gate evidence and --images JSONL")
        gate = json_file(root, args.gate)
        if not (gate.get("fp16_image_grounding_passed") is True or gate.get("fp16_memory_failure_documented") is True):
            raise ValueError("Optimization remains gated on real FP16 signs of life or a documented memory failure")
        if not gate.get("evidence_file"):
            raise ValueError("Gate must link to the actual device evidence_file")
        evidence = contained(root, Path(args.gate).resolve().parent / gate["evidence_file"])
        if not evidence.is_file() or evidence.stat().st_size == 0:
            raise ValueError("Missing actual device evidence")
        rows = corpus_rows(root, contained(root, args.images), args.samples)
        # Recheck transferred input bytes against the already verified source
        # manifest before consuming expensive calibration GPU time.
        download = json_file(root, project / "results/model-download.json")
        if download.get("revision") != MODEL_PIN or download.get("complete") is not True:
            raise ValueError("Missing verified pinned model download manifest")
        for filename, record in download["files"].items():
            path = contained(root, model / filename)
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            if path.stat().st_size != record["size_bytes"] or digest.hexdigest() != record["sha256"]:
                raise ValueError(f"Transferred model file failed checksum: {filename}")
        output = contained(root, project / "models/cosmos3-edge-awq-candidate-01")
        if output.exists():
            raise ValueError("Candidate output already exists; preserve it and review rather than silently rerunning")
        import torch as torch_module
        torch = torch_module
        import tensorrt_edgellm.quantization.quantize as quant
        if not Path(quant.__file__).resolve().is_relative_to(source):
            raise ValueError("Quantizer import is not from the pinned local checkout")
        if not torch.cuda.is_available():
            raise RuntimeError("No usable CUDA GPU")
        torch.manual_seed(42)
        torch.cuda.reset_peak_memory_stats()
        report["gpu"] = torch.cuda.get_device_name(0)
        report["gpu_total_bytes"] = torch.cuda.get_device_properties(0).total_memory
        report["gpu_free_bytes_before"] = torch.cuda.mem_get_info()[0]
        report["gate"] = gate
        report["calibration_manifest_sha256"] = hashlib.sha256(contained(root, args.images).read_bytes()).hexdigest()
        original_config = quant.build_quant_config
        original_load = quant._load_model
        original_batches = quant._multimodal_calib_dataloader

        def build_config(*positional, **kwargs):
            result = original_config(*positional, **kwargs)
            # Upstream's visual globs cover model.visual, but miss model.projector.
            result["quant_cfg"].append({"quantizer_name": "*projector.*", "enable": False})
            return result

        def checked_load(*positional, **kwargs):
            loaded, tokenizer, proc = original_load(*positional, **kwargs)
            if sum(p.numel() for p in loaded.parameters()) != PARAMETERS:
                raise RuntimeError("HF calibration model parameter count differs from the indexed reasoner")
            if not any("visual" in n for n, _ in loaded.named_modules()) or not any("projector" in n for n, _ in loaded.named_modules()):
                raise RuntimeError("Unexpected vision/projector module naming; audit precision exclusions before proceeding")
            if not quant._is_image_blind_calibration(loaded, build_config("int4_awq", "int4_awq")):
                raise RuntimeError("AWQ would silently select text-only calibration")
            return loaded, tokenizer, proc

        def checked_batches(*positional, **kwargs):
            batches = original_batches(*positional, **kwargs)
            if len(batches) != args.samples:
                raise RuntimeError("Calibration sample count changed")
            for batch in batches:
                if batch["input_ids"].shape[-1] > 768 or "pixel_values" not in batch:
                    raise RuntimeError("Calibration must contain images and at most 768 total tokens")
            return batches

        def images():
            from PIL import Image
            for image, question in rows:
                with Image.open(image) as opened:
                    # A fixed bounded candidate, not an adaptive quality search.
                    bounded = opened.convert("RGB")
                    bounded.thumbnail((256, 256))
                    yield bounded.copy(), question

        def forbidden_text():
            raise RuntimeError("Text-only AWQ calibration is forbidden for this image-grounded candidate")
            yield ""  # generator contract

        quant.build_quant_config = build_config
        quant._load_model = checked_load
        quant._multimodal_calib_dataloader = checked_batches
        report["quantization_pipeline_invoked"] = True
        quant.quantize_and_export(model_dir=str(model), output_dir=str(output),
                                  quantization="int4_awq", lm_head_quantization="int4_awq",
                                  visual_quantization=None, visual_mha_quantization=None,
                                  kv_cache_quantization=None, dtype="fp16", device="cuda",
                                  image_dataset=images, text_dataset=forbidden_text,
                                  num_samples=args.samples)
        from safetensors import safe_open
        keys, visual_tensors, head_scales = [], 0, 0
        for shard in output.rglob("*.safetensors"):
            with safe_open(str(shard), framework="pt", device="cpu") as sf:
                for key in sf.keys():
                    keys.append(key)
                    if ".visual." in key or ".projector." in key:
                        visual_tensors += 1
                        if "scale" in key or sf.get_slice(key).get_dtype() != "F16":
                            raise RuntimeError("Exported visual/projector tensor is quantized or not FP16")
                    if key.endswith("lm_head.weight_scale"):
                        head_scales += 1
        if not visual_tensors or not head_scales:
            raise RuntimeError("Export lost vision tensors or lacks quantized LM-head scales")
        report.update(status="candidate_exported_not_validated_on_orin", output=str(output),
                      tensor_count=len(keys), visual_fp16_tensor_count=visual_tensors)
    except Exception as exc:
        report.update(status="blocked_or_failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if torch is not None and torch.cuda.is_available():
            report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
        report["finished_unix"] = time.time()
        destination.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
