#!/usr/bin/env python3
"""Read-only Linux storage/IRQ sampling; the only write is a new JSONL file."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import signal
import sys
import threading
import time


PROC = Path("/proc")
CPU_FIELDS = "user nice system idle iowait irq softirq steal guest guest_nice".split()
DISK_FIELDS = (
    "reads_completed reads_merged sectors_read read_ms writes_completed "
    "writes_merged sectors_written write_ms io_in_progress io_ms weighted_io_ms "
    "discards_completed discards_merged sectors_discarded discard_ms "
    "flushes_completed flush_ms"
).split()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def named_counters(names: list[str], fields: list[str]) -> dict:
    values = [int(value) for value in fields]
    result = dict(zip(names, values))
    if len(values) > len(names):
        result["additional_values"] = values[len(names):]
    return result


def sample(proc: Path = PROC) -> dict:
    started = time.monotonic()
    result = {"type": "sample", "utc": utc_now(), "monotonic_s": started,
              "uptime_s": None, "all_cpu_idle_s": None, "cpu0_ticks": None,
              "interrupt_cpus": [], "interrupts": [], "nvme0n1": None,
              "errors": {}}
    for filename in ("uptime", "interrupts", "stat", "diskstats"):
        try:
            lines = (proc / filename).read_text().splitlines()
            if filename == "uptime":
                result["uptime_s"], result["all_cpu_idle_s"] = map(float, lines[0].split())
            elif filename == "interrupts":
                cpus = lines[0].split()
                if not cpus or any(not cpu.startswith("CPU") for cpu in cpus):
                    raise ValueError("Missing interrupt CPU header")
                result["interrupt_cpus"] = cpus
                for line in lines[1:]:
                    if "nvme" not in line.lower() and "14160000.pcie" not in line.lower():
                        continue
                    irq, rest = line.split(":", 1)
                    fields = rest.split()
                    if len(fields) < len(cpus):
                        raise ValueError("Incomplete interrupt counters")
                    result["interrupts"].append({
                        "irq": irq.strip(),
                        "counts": dict(zip(cpus, map(int, fields[:len(cpus)]))),
                        "description": " ".join(fields[len(cpus):]),
                    })
                if not result["interrupts"]:
                    raise ValueError("No matching NVMe/14160000.pcie interrupts")
            elif filename == "stat":
                fields = next(line.split()[1:] for line in lines if line.startswith("cpu0 "))
                if len(fields) < 4:
                    raise ValueError("Incomplete cpu0 counters")
                result["cpu0_ticks"] = named_counters(CPU_FIELDS, fields)
            else:
                fields = next(line.split() for line in lines if len(line.split()) >= 3
                              and line.split()[2] == "nvme0n1")
                if len(fields) < 14:
                    raise ValueError("Incomplete nvme0n1 counters")
                result["nvme0n1"] = {"major": int(fields[0]), "minor": int(fields[1]),
                                     **named_counters(DISK_FIELDS, fields[3:])}
        except (OSError, ValueError, IndexError, StopIteration) as exc:
            result["errors"][filename] = f"{type(exc).__name__}: {exc}"
    result["read_duration_s"] = time.monotonic() - started
    return result


def positive_seconds(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New JSONL file; parent directory must exist")
    parser.add_argument("--interval", type=positive_seconds, default=1.0, help="Seconds (default: 1)")
    parser.add_argument("--duration", type=positive_seconds, default=1800.0, help="Seconds (default: 1800)")
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("Run on Linux; this sampler reads local /proc only")

    stopped = threading.Event()
    received_signal = 0

    def stop(signum, _frame):
        nonlocal received_signal
        received_signal = signum
        stopped.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        output = args.output.open("x", encoding="utf-8")
    except OSError as exc:
        parser.error(str(exc))

    with output:
        def emit(record: dict):
            output.write(json.dumps(record, separators=(",", ":")) + "\n")

        started = time.monotonic()
        deadline = started + args.duration
        emit({"type": "metadata", "schema_version": 1, "utc": utc_now(),
              "kernel": platform.release(), "interval_s": args.interval,
              "duration_s": args.duration, "clock_ticks_per_second": os.sysconf("SC_CLK_TCK"),
              "disk_sector_bytes": 512, "proc_sources": ["interrupts", "stat", "uptime", "diskstats"],
              "note": "Cumulative counters except io_in_progress; CPU guest counters overlap user/nice. "
                      "Reads are sequential, not an atomic snapshot. No kernel log capture or settings changes."})
        output.flush()
        next_sample = started
        last_flush = started
        count = 0
        while not stopped.is_set() and time.monotonic() < deadline:
            now = time.monotonic()
            if now < next_sample:
                stopped.wait(min(next_sample, deadline) - now)
                continue
            record = sample(PROC)
            record["elapsed_s"] = record["monotonic_s"] - started
            emit(record)
            count += 1
            now = time.monotonic()
            if now - last_flush >= 5:
                output.flush()
                last_flush = now
            # Skip missed slots instead of issuing a burst after a stalled read/write.
            next_sample = started + (math.floor((now - started) / args.interval) + 1) * args.interval
        emit({"type": "end", "utc": utc_now(), "samples": count,
              "elapsed_s": time.monotonic() - started,
              "reason": signal.Signals(received_signal).name if received_signal else "duration"})
        output.flush()
    return 128 + received_signal if received_signal else 0


if __name__ == "__main__":
    raise SystemExit(main())
