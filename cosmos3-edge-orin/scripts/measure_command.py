#!/usr/bin/env python3
"""Run one build command and record its separate, sampled memory footprint.

This observes local Linux system RAM and swap counters. GNU time separately
records maximum RSS in KiB. These overlapping measurements must not be added.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def utc():
    return datetime.now(timezone.utc).isoformat()


def counters(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        fields = line.replace(":", " ", 1).split()
        if len(fields) >= 2 and fields[1].isdigit():
            result[fields[0]] = int(fields[1]) * (1024 if fields[-1] == "kB" else 1)
    return result


def cleanup_group(process, signum, timeout):
    """Bound shutdown even if the group leader exits before its descendants."""
    if process is None:
        return None
    result = {"initial_signal": signum, "signal_sent": False, "sigkill_sent": False}
    process.poll()  # Reap an exited leader before checking its remaining group.
    try:
        os.killpg(process.pid, signum)
        result["signal_sent"] = True
        deadline = time.monotonic() + timeout
        while True:
            process.poll()
            os.killpg(process.pid, 0)
            if time.monotonic() >= deadline:
                os.killpg(process.pid, signal.SIGKILL)
                result["sigkill_sent"] = True
                break
            time.sleep(0.1)
    except ProcessLookupError:
        pass
    except OSError as exc:
        result["error"] = str(exc)
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        result["error"] = "Process leader did not exit after bounded group cleanup"
    result["leader_returncode"] = process.returncode
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New result directory")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Supply a command after --")
    if sys.platform != "linux" or not Path("/usr/bin/time").is_file():
        parser.error("Requires Linux and /usr/bin/time")
    args.output.mkdir(parents=False, exist_ok=False)
    output = args.output.resolve()
    status = {"started_utc": utc(), "command": command, "cwd": str(Path.cwd()),
              "phase": "build", "inference_validated": False, "sample_interval_s": 0.5,
              "swap_page_bytes": os.sysconf("SC_PAGE_SIZE"),
              "memory_note": "System RAM includes CPU and GPU shared memory on Jetson. JSON RAM/swap sizes are bytes; pswpin/pswpout are page counts. System peaks are sampled. GNU time maximum RSS is a separate kernel high-water mark in KiB, not a sum of simultaneous process-tree RSS. These observations overlap; do not add them."}
    (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    child = None
    tegra = None
    requested_signal = None

    def stop(signum, _frame):
        nonlocal requested_signal
        if requested_signal is None:
            requested_signal = signum

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    first = last = None
    peak_unavailable = None
    minimum_available = None
    peak_swap = None
    samples = 0
    errors = []
    started = time.monotonic()
    exit_code = 1
    try:
        with (output / "output.log").open("x") as log, (output / "memory.jsonl").open("x") as memory, (output / "tegrastats.log").open("x") as tegra_log:
            if shutil.which("tegrastats"):
                tegra = subprocess.Popen(["tegrastats", "--interval", "500"], stdout=tegra_log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
            if requested_signal is not None:
                raise InterruptedError("Interrupted before starting the build command")
            child = subprocess.Popen(["/usr/bin/time", "-v", "-o", str(output / "time.txt"), "--", *command],
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            (output / "pid").write_text(str(child.pid) + "\n")
            while True:
                record = {"utc": utc(), "elapsed_s": time.monotonic() - started}
                try:
                    mem = counters("/proc/meminfo")
                    vm = counters("/proc/vmstat")
                    record.update(mem_available_bytes=mem["MemAvailable"], system_ram_unavailable_bytes=mem["MemTotal"]-mem["MemAvailable"],
                                  swap_occupied_bytes=mem["SwapTotal"]-mem["SwapFree"], pswpin=vm["pswpin"], pswpout=vm["pswpout"], oom_kill=vm.get("oom_kill"))
                    first = first or record.copy()
                    last = record.copy()
                    peak_unavailable = max(peak_unavailable, record["system_ram_unavailable_bytes"]) if peak_unavailable is not None else record["system_ram_unavailable_bytes"]
                    minimum_available = min(minimum_available, record["mem_available_bytes"]) if minimum_available is not None else record["mem_available_bytes"]
                    peak_swap = max(peak_swap, record["swap_occupied_bytes"]) if peak_swap is not None else record["swap_occupied_bytes"]
                except (OSError, KeyError) as exc:
                    record["error"] = str(exc)
                    errors.append(str(exc))
                memory.write(json.dumps(record) + "\n")
                memory.flush()
                samples += 1
                if requested_signal is not None:
                    break  # The finally block bounds whole-group termination.
                if child.poll() is not None:
                    exit_code = child.returncode
                    break
                time.sleep(0.5)
    except Exception as exc:
        status["wrapper_error"] = str(exc)
        exit_code = 1
    finally:
        status["command_cleanup"] = cleanup_group(child, requested_signal or signal.SIGTERM, 10)
        status["tegrastats_cleanup"] = cleanup_group(tegra, signal.SIGTERM, 3)
        if requested_signal is not None:
            exit_code = 128 + requested_signal
        elif exit_code < 0:
            exit_code = 128 - exit_code
        if any(item and item.get("error") for item in (status["command_cleanup"], status["tegrastats_cleanup"])) and exit_code == 0:
            exit_code = 1
        status.update(completed_utc=utc(), elapsed_s=time.monotonic()-started, exit_code=exit_code,
                      child_returncode=child.returncode if child is not None else None,
                      requested_signal=requested_signal,
                      samples=samples, sampling_errors=errors, system_ram_unavailable_peak_bytes=peak_unavailable,
                      mem_available_minimum_bytes=minimum_available, swap_occupied_peak_bytes=peak_swap,
                      swap_in_pages_delta=last["pswpin"]-first["pswpin"] if first and last else None,
                      swap_out_pages_delta=last["pswpout"]-first["pswpout"] if first and last else None,
                      oom_kills_delta=last["oom_kill"]-first["oom_kill"] if first and last and first["oom_kill"] is not None and last["oom_kill"] is not None else None)
        (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        (output / "exit-code").write_text(str(exit_code) + "\n")
    print(json.dumps(status, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
