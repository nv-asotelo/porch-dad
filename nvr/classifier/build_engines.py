#!/usr/bin/env python3
"""Build each classifier's TensorRT engine on the Orin, from the ONNX file its meta.json names.

Run with the interpreter classifier_service.py runs under (the TensorRT-Edge-LLM venv): an engine
only loads in the TensorRT version that built it, and this builds with that same library, where
/usr/bin/trtexec may not be. Strongly typed, so the FP16 export runs in FP16 exactly where its ONNX
says so, with batch 1 at the model's own input size. Skips an engine that is newer than its ONNX.

  python build_engines.py [--models-dir /home/orin/nvr/classifier/models] [--force] [model ...]
"""
import argparse
import json
import os
import threading
import time
from pathlib import Path


def meminfo_mb() -> dict:
    with open("/proc/meminfo") as f:
        return {k: int(v.split()[0]) // 1024 for k, _, v in (line.partition(":") for line in f)}


def guard(available_mb: int, swap_mb: int):
    """Abort the build before it starves the NVR: the board has no memory to spare in full porch-dad
    mode, and an unguarded ViT-B/16 build there drove MemAvailable to 272 MB and swap to full. Low
    MemAvailable alone is survivable while swap has room; both running out is what ends in the OOM
    killer, whose first pick is the shim."""
    def watch():
        while True:
            info = meminfo_mb()
            if info["MemAvailable"] < available_mb or info["SwapTotal"] and info["SwapFree"] < swap_mb:
                print(f"aborting: MemAvailable {info['MemAvailable']} MB, SwapFree {info['SwapFree']} MB "
                      f"(floors {available_mb} and {swap_mb} MB) - free some memory first", flush=True)
                os._exit(3)
            time.sleep(0.5)
    threading.Thread(target=watch, daemon=True).start()


def build(directory: Path, workspace_mb: int, level) -> Path:
    import tensorrt as trt
    meta = json.loads((directory / "meta.json").read_text())
    onnx, plan = directory / meta["onnx"], directory / meta["engine"]
    size = int(meta["preprocess"].get("size", 224))
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx)):
        raise RuntimeError("; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)
    config.max_aux_streams = 0
    if level is not None:  # lower: fewer tactics timed, a shorter build, a slightly slower engine
        config.builder_optimization_level = level
    profile = builder.create_optimization_profile()
    shape = (1, 3, size, size)
    profile.set_shape(meta["input"], shape, shape, shape)
    config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT could not build the engine")
    partial = plan.with_suffix(".partial")
    partial.write_bytes(serialized)
    partial.replace(plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="*", help="model ids (directory names); default all")
    parser.add_argument("--models-dir", type=Path, default=Path("/home/orin/nvr/classifier/models"))
    parser.add_argument("--workspace-mb", type=int, default=128,
                        help="builder scratch memory; small, because the board is shared with the NVR")
    parser.add_argument("--level", type=int, default=None, choices=range(6),
                        help="TensorRT builder optimization level, 0-5 (default TensorRT's, 3)")
    parser.add_argument("--min-available-mb", type=int, default=150,
                        help="abort if MemAvailable falls below this")
    parser.add_argument("--min-swap-free-mb", type=int, default=300,
                        help="abort if free swap falls below this")
    parser.add_argument("--force", action="store_true", help="rebuild engines that look current")
    args = parser.parse_args()
    guard(args.min_available_mb, args.min_swap_free_mb)
    directories = [args.models_dir / m for m in args.models] if args.models else sorted(
        p for p in args.models_dir.iterdir() if (p / "meta.json").is_file())
    for directory in directories:
        meta = json.loads((directory / "meta.json").read_text())
        onnx, plan = directory / meta["onnx"], directory / meta["engine"]
        if plan.is_file() and plan.stat().st_mtime > onnx.stat().st_mtime and not args.force:
            print(f"{directory.name}: {plan.name} is current", flush=True)
            continue
        t0 = time.time()
        built = build(directory, args.workspace_mb, args.level)
        print(f"{directory.name}: built {built.name} ({built.stat().st_size / 1e6:.1f} MB) "
              f"in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
