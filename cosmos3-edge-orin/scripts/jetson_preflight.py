#!/usr/bin/env python3
"""Read-only gates for the frozen Cosmos MLP deployment; never an inference acceptance.

Run system before installing the backend and runtime after compiling it. Exit 0
means a build/runtime candidate, not a validated model. Exit 2 reports blockers.
No packages, clocks, disks or services are changed. --output creates a private,
new receipt; keep it in ignored data/ rather than publishing device inventory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024 ** 3
MIN_RAM = 7 * GIB  # An 8 GB Nano exposes approximately 7.37 GiB to Linux.
MIN_BUILD_FREE = 30 * GIB
REFERENCE = {
    "board": "Jetson Orin Nano 8 GB (P3767-0005)",
    "jetpack": "7.2.1", "l4t": "39.2.1", "cuda_compiler": "13.2.86",
    "tensorrt": "10.16.2.10", "python": "3.12", "compute_capability": [8, 7],
}

PYTHON_PROBE = r'''
import json, sys
result = {"python": list(sys.version_info[:3])}
try:
    import tensorrt
    result["tensorrt"] = tensorrt.__version__
except Exception as exc:
    result["tensorrt_error"] = type(exc).__name__ + ": " + str(exc)
if sys.argv[1] == "runtime":
    try:
        import cupy as cp
        count = cp.cuda.runtime.getDeviceCount()
        result["gpu_count"] = count
        if count == 1:
            props = cp.cuda.runtime.getDeviceProperties(0)
            name = props["name"]
            result["gpu_name"] = name.decode() if isinstance(name, bytes) else name
            result["compute_capability"] = [props["major"], props["minor"]]
            result["cuda_runtime"] = cp.cuda.runtime.runtimeGetVersion()
            result["cuda_array_sum"] = int(cp.arange(1024, dtype=cp.int32).sum().get())
            cp.cuda.runtime.deviceSynchronize()
        from experimental.server.runtime.engine import _import_runtime
        rt = _import_runtime()
        result["native_binding"] = bool(hasattr(rt, "LLMRuntime"))
        result["encoder_budget_binding"] = hasattr(rt.ContextCacheConfig(), "encoder_embedding_cache_budget_bytes")
        result["image_budget_binding"] = hasattr(rt.ImageData(), "max_image_tokens_per_image")
        result["encoder_bypass_binding"] = hasattr(rt.ImageData(), "skip_encoder_cache")
    except Exception as exc:
        result["runtime_error"] = type(exc).__name__ + ": " + str(exc)
print("COSMOS_PREFLIGHT_JSON=" + json.dumps(result))
'''


def command(argv: list[str], timeout: int = 15) -> str:
    result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{Path(argv[0]).name} exited {result.returncode}: "
                           + (result.stderr.strip() or result.stdout.strip())[-1500:])
    return result.stdout.strip()


def read_text(path: str) -> str:
    try:
        return Path(path).read_text().replace("\0", " ").strip()
    except (OSError, UnicodeError):
        return ""


def parse_l4t(text: str) -> str | None:
    match = re.search(r"# R(\d+).*?REVISION:\s*([\d.]+)", text)
    return f"{match[1]}.{match[2]}" if match else None


def parse_cuda(text: str) -> str | None:
    match = re.search(r"\bV(\d+\.\d+\.\d+)\b", text)
    return match[1] if match else None


def collect(project: Path, python: str, stage: str) -> dict:
    inventory = {
        "os": platform.system(), "architecture": platform.machine(),
        "device_model": read_text("/proc/device-tree/model"),
        "device_compatible": read_text("/proc/device-tree/compatible"),
        "l4t": parse_l4t(read_text("/etc/nv_tegra_release")),
    }
    mem = dict(re.findall(r"^(\w+):\s+(\d+)\s+kB", read_text("/proc/meminfo"), re.M))
    inventory["ram_total_bytes"] = int(mem.get("MemTotal", 0)) * 1024
    inventory["ram_available_bytes"] = int(mem.get("MemAvailable", 0)) * 1024
    inventory["swap_total_bytes"] = int(mem.get("SwapTotal", 0)) * 1024
    inventory["project_free_bytes"] = shutil.disk_usage(project).free
    inventory["project_writable"] = os.access(project, os.W_OK)
    try:
        inventory["root_filesystem"] = command(["findmnt", "-n", "-o", "FSTYPE", "/"])
    except (OSError, RuntimeError, subprocess.TimeoutExpired):
        inventory["root_filesystem"] = None
    nvcc = Path("/usr/local/cuda-13.2/bin/nvcc")
    inventory["canonical_cuda_toolkit"] = nvcc.is_file() and Path("/usr/local/cuda-13.2/lib64").is_dir()
    executable = str(nvcc) if nvcc.is_file() else shutil.which("nvcc")
    if executable:
        try:
            inventory["cuda_compiler"] = parse_cuda(command([executable, "--version"]))
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            inventory["cuda_probe_error"] = str(exc)
    try:
        output = command([python, "-c", PYTHON_PROBE, stage], timeout=60)
        records = [line.removeprefix("COSMOS_PREFLIGHT_JSON=") for line in output.splitlines()
                   if line.startswith("COSMOS_PREFLIGHT_JSON=")]
        if len(records) != 1:
            raise ValueError("Python probe did not return exactly one inventory record")
        inventory.update(json.loads(records[0]))
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        inventory["python_probe_error"] = str(exc)
    nvpmodel = shutil.which("nvpmodel")
    if nvpmodel:
        try:
            inventory["power_mode_query"] = command([nvpmodel, "-q"])
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            inventory["power_mode_query_error"] = str(exc)
    return inventory


def evaluate(inventory: dict, stage: str) -> dict:
    errors, warnings = [], []
    if inventory.get("os") != "Linux" or inventory.get("architecture") != "aarch64":
        errors.append("This release requires Linux aarch64 on a Jetson Orin.")
    if inventory.get("root_filesystem") != "ext4":
        errors.append("The native wrapper requires a normally booted installed ext4 root filesystem, not a recovery chroot.")
    model = str(inventory.get("device_model", "")).lower()
    compatible = str(inventory.get("device_compatible", "")).lower()
    if not ("jetson" in model and "orin" in model and "tegra234" in compatible):
        errors.append("Require Jetson Orin device-tree identity and Tegra234; Thor, Xavier and original Nano are outside this release.")
    if inventory.get("ram_total_bytes", 0) < MIN_RAM:
        errors.append("Require an 8 GB-class or larger Orin (at least 7 GiB Linux MemTotal); Nano 4 GB is unsupported for this profile.")
    if inventory.get("project_free_bytes", 0) < MIN_BUILD_FREE and stage == "system":
        errors.append("Reserve at least 30 GiB free on the project filesystem before downloading and building.")
    if not inventory.get("project_writable"):
        errors.append("Project directory must be writable by the deployment account.")
    if not re.fullmatch(r"39\.2\.\d+(?:\.\d+)*", str(inventory.get("l4t", ""))):
        errors.append("This frozen recipe targets L4T 39.2.x / JetPack 7.2.x. Match that stack; do not auto-upgrade or flash a disk.")
    if not str(inventory.get("cuda_compiler", "")).startswith("13.2."):
        errors.append("Require the CUDA 13.2 compiler; install the matching JetPack compute packages.")
    if not inventory.get("canonical_cuda_toolkit"):
        errors.append("Build and launch helpers require the JetPack toolkit at /usr/local/cuda-13.2 with bin/nvcc and lib64.")
    if not str(inventory.get("tensorrt", "")).startswith("10.16."):
        errors.append("Require platform TensorRT 10.16.x visible to the selected Python; do not substitute an x86/SBSA wheel.")
    if inventory.get("python", [])[:2] != [3, 12]:
        errors.append("This frozen Python dependency recipe requires Python 3.12.")
    for key in ("cuda_probe_error", "python_probe_error", "tensorrt_error"):
        if inventory.get(key):
            errors.append(f"{key}: {inventory[key]}")
    if stage == "runtime":
        if inventory.get("gpu_count") != 1 or inventory.get("compute_capability") != [8, 7]:
            errors.append("The native release requires exactly one CUDA device with SM87 (compute capability 8.7).")
        if inventory.get("cuda_array_sum") != 523776:
            errors.append("The real CUDA array calculation did not pass.")
        # CuPy's CUDA-13 wheel can report its bundled 13.0 runtime even on the
        # tested JetPack 13.2 toolkit. The compiler/platform gate remains 13.2.
        if inventory.get("cuda_runtime", 0) // 1000 != 13:
            errors.append("The CuPy GPU probe must use a CUDA 13.x runtime with the required platform CUDA 13.2 toolkit.")
        for key in ("native_binding", "encoder_budget_binding", "image_budget_binding", "encoder_bypass_binding"):
            if inventory.get(key) is not True:
                errors.append(f"Missing {key}; build the native runtime with all six release patches.")
        if inventory.get("runtime_error"):
            errors.append("runtime_error: " + str(inventory["runtime_error"]))
    else:
        warnings.append("Actual CUDA compute capability and native bindings remain unchecked until --stage runtime.")
    if any(inventory.get(key) != REFERENCE[key] for key in ("l4t", "cuda_compiler", "tensorrt")):
        warnings.append("Package versions differ from the recorded reference; local compilation and inference acceptance are required.")
    if "nano" not in model:
        warnings.append("This Orin variant was not hardware-tested for this release. Shared SM87 is a compatibility candidate, not performance or quality validation.")
    if inventory.get("ram_available_bytes", 0) < 5 * GIB:
        warnings.append("Less than 5 GiB RAM is currently available; stop only identified deployment workloads before memory-heavy build steps.")
    if inventory.get("swap_total_bytes", 0):
        warnings.append("Swap is configured; record its state and do not compare memory/latency to the no-swap reference without accounting for it.")
    return {"ready_for_next_step": not errors,
            "classification": "blocked" if errors else f"{stage}_candidate",
            "inference_validated": False, "errors": errors, "warnings": warnings}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("system", "runtime"), default="system")
    parser.add_argument("--python", default=sys.executable, help="Backend venv Python for runtime stage")
    parser.add_argument("--project-dir", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, help="New private JSON receipt; existing files are preserved")
    args = parser.parse_args(argv)
    project = args.project_dir.resolve()
    if not project.is_dir():
        parser.error("--project-dir must already exist")
    if args.output and args.output.exists():
        parser.error("--output must be a new receipt path")
    inventory = collect(project, args.python, args.stage)
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "stage": args.stage,
              "reference": REFERENCE, "inventory": inventory, **evaluate(inventory, args.stage)}
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        old_umask = os.umask(0o077)
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as stream:
                stream.write(text)
        finally:
            os.umask(old_umask)
    print(text, end="")
    return 0 if report["ready_for_next_step"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
