#!/usr/bin/env python3
"""Bounded local API streaming soak; browser/camera behavior and quality are separate.

Rotates the six frozen JPEG workloads with one request in flight for at least
600 seconds. Records sampled, overlapping system RAM and backend RSS separately.
No missing resource/health evidence can produce a passing result.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import http.client
import ipaddress
import json
from pathlib import Path
import platform
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit, urlunsplit

from benchmark import LocalSampler, build_payload, safe_url, stream_request, summarize_memory


ROOT = Path(__file__).resolve().parents[1]
POLICY = {
    "minimum_duration_seconds": 600,
    "minimum_available_bytes": 512 << 20,
    "maximum_active_requests": 1, "maximum_queued_requests": 1,
    "rss_growth_baseline_window_seconds": [60, 120], "rss_growth_final_window_seconds": 60,
    "rss_growth_absolute_tolerance_bytes": 64 << 20, "rss_growth_relative_tolerance": 0.05,
    "growth_rule": "Fail if final-window median RSS minus baseline median exceeds max(64 MiB, 5% baseline RSS).",
    "system_ram_growth_rule": "Report the same windows separately; system unavailable RAM is not added to RSS.",
    "length_finish_reason_is_failure": True,
    "quality_rule": "No semantic grading or exact answer matching; the frozen quality screen is separate.",
}
PROC_FIELDS = ("system_ram_total_bytes", "system_ram_unavailable_bytes", "process_rss_bytes",
               "system_swap_occupied_bytes", "swap_in_pages", "swap_out_pages", "system_oom_kills")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(data):
    return hashlib.sha256(data).hexdigest()


def local_url(value):
    value = safe_url(value)
    host = urlsplit(value).hostname
    if host != "localhost" and not ipaddress.ip_address(host).is_loopback:
        raise ValueError("Run the soak on the backend host against a loopback URL")
    return value


def load_workloads(manifest_path):
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    settings = manifest["request_settings"]
    if (len(manifest["cases"]) != 6 or len({case["id"] for case in manifest["cases"]}) != 6
            or settings != {"model_alias": "Cosmos3-Edge", "max_tokens": 64, "temperature": 0,
                            "top_p": 1, "concurrency": 1, "stream": True,
                            "stream_options": {"include_usage": True}}):
        raise ValueError("Require the frozen six-case Cosmos3 JPEG settings")
    workloads = []
    for case in manifest["cases"]:
        path = (manifest_path.parent / case["filename"]).resolve()
        if not path.is_relative_to(manifest_path.parent.resolve()) or case["mime_type"] != "image/jpeg":
            raise ValueError("Fixture must be a local JPEG within the manifest directory")
        image = path.read_bytes()
        if fingerprint(image) != case["sha256"]:
            raise ValueError("Frozen fixture SHA256 mismatch: " + case["id"])
        workload = dict(model=settings["model_alias"], prompt=case["prompt"], max_tokens=settings["max_tokens"],
                        temperature=settings["temperature"], top_p=settings["top_p"], image_mime_type=case["mime_type"])
        workloads.append({"case_id": case["id"], "image_sha256": case["sha256"],
                          "workload": workload, "payload": build_payload(workload, image)})
    return manifest, fingerprint(raw), workloads


def pid_identity(pid):
    folder = Path(f"/proc/{pid}")
    fields = (folder / "stat").read_text().rpartition(") ")[2].split()
    return {"pid": pid, "start_ticks": int(fields[19]),
            "command": (folder / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip(),
            "executable": str((folder / "exe").resolve(strict=True))}


def provenance(config_path):
    raw = config_path.read_bytes()
    config = json.loads(raw)
    if not isinstance(config, dict) or not config:
        raise ValueError("--backend-config must be a nonempty JSON provenance object")
    upstream = ROOT / "external/TensorRT-Edge-LLM"
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    diff = subprocess.check_output(["git", "-C", str(upstream), "diff", "--binary", "HEAD", "--"])
    return {"backend_config_path": str(config_path.resolve()), "backend_config_sha256": fingerprint(raw),
            "backend_config": config, "backend_revision": revision, "backend_tracked_diff_sha256": fingerprint(diff),
            "soak_script_sha256": fingerprint(Path(__file__).read_bytes()),
            "benchmark_script_sha256": fingerprint((ROOT / "scripts/benchmark.py").read_bytes())}


def bounded_request(url, payload, timeout):
    # benchmark.stream_request catches TimeoutError and retains the failed sample.
    # A real wall timer also bounds a peer that drips bytes without complete SSE events.
    def expired(_signum, _frame):
        raise TimeoutError("Soak request exceeded its wall-clock timeout")
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return stream_request(url, payload, timeout)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def stream_sequence(url, workloads, records, write, *, started, duration, timeout):
    while time.monotonic() - started < duration:
        selected = workloads[len(records) % len(workloads)]
        record = {"kind": "request", "index": len(records), "case_id": selected["case_id"],
                  "observed_utc": utc_now(), "monotonic_seconds": time.monotonic(),
                  **bounded_request(url, selected["payload"], timeout)}
        write(record)
        records.append(record)
        if record["error"] or record["finish_reason"] != "stop":
            break  # Keep failed evidence without adding load to an unhealthy backend.


def read_health(url, timeout=3):
    """Bound health headers/body by wall time and 64 KiB, including slow drips."""
    parts = urlsplit(url)
    factory = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
    connection = factory(parts.hostname, parts.port, timeout=timeout)
    upstream_socket = [None]
    expired = threading.Event()
    def abort():
        expired.set()
        active_socket = upstream_socket[0] or connection.sock
        if active_socket:
            try:
                active_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
    timer = threading.Timer(timeout, abort)
    timer.daemon = True
    timer.start()
    response = None
    try:
        connection.connect()
        upstream_socket[0] = connection.sock
        if expired.is_set():
            raise TimeoutError("Health request exceeded its wall-clock timeout")
        connection.request("GET", parts.path or "/", headers={"Accept": "application/json"})
        response = connection.getresponse()
        raw = response.read(65537)
        if expired.is_set():
            raise TimeoutError("Health request exceeded its wall-clock timeout")
        if len(raw) > 65536:
            raise ValueError("Health response exceeded 64 KiB")
        data = json.loads(raw)
        # The pinned readiness API exposes exactly these observable counters.
        # Do not retain arbitrary extra response content on every poll.
        if isinstance(data, dict):
            data = {key: data.get(key) for key in ("status", "active_requests", "queued_requests")}
        return {"http_status": response.status, "data": data}
    finally:
        timer.cancel()
        if response is not None:
            response.close()
        connection.close()
        timer.join(timeout=.1)


class Observer:
    def __init__(self, sampler, health_url, write):
        self.sampler, self.health_url, self.write = sampler, health_url, write
        self.health = []
        self.written = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.poll, daemon=True)

    def flush_memory(self):
        with self.sampler.lock:
            unseen = self.sampler.samples[self.written:]
            self.written = len(self.sampler.samples)
        for sample in unseen:
            self.write({"kind": "resource_sample", "sample": sample})

    def capture(self):
        record = {"kind": "health_sample", "monotonic_seconds": time.monotonic(), "error": None}
        try:
            record.update(read_health(self.health_url))
        except Exception as error:
            record["error"] = str(error)
        self.health.append(record)
        self.write(record)
        self.flush_memory()

    def poll(self):
        while not self.stop_event.wait(1):  # At most one health poll per second.
            self.capture()

    def start(self):
        self.capture()
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=4)
        return not self.thread.is_alive()


def evaluate(records, samples, health, *, started, ended, expected_cases, requested_seconds,
             sampler_errors=(), run_errors=(), pid_unchanged=True, provenance_unchanged=True):
    reasons = list(run_errors)
    elapsed = ended - started
    if elapsed < max(POLICY["minimum_duration_seconds"], requested_seconds):
        reasons.append("duration_below_required_minimum")
    failures = [record for record in records if record.get("error") or record.get("stream_done") is not True
                or record.get("finish_reason") != "stop" or not record.get("output_text", "").strip()]
    if failures:
        reasons.append("request_error_truncation_or_output_limit")
    if set(record["case_id"] for record in records) != set(expected_cases):
        reasons.append("incomplete_fixture_coverage")
    if not pid_unchanged:
        reasons.append("backend_pid_identity_changed_or_unavailable")
    if not provenance_unchanged:
        reasons.append("backend_provenance_changed_or_unavailable")
    proc = [sample for sample in samples if sample.get("kind") == "proc"]
    valid_proc = [sample for sample in proc if not sample.get("error") and all(
        type(sample.get(field)) is int and sample[field] >= 0 for field in PROC_FIELDS)
        and sample["system_ram_total_bytes"] >= sample["system_ram_unavailable_bytes"]]
    complete_proc = len(valid_proc) == len(proc) and len(proc) >= 2 and not sampler_errors
    if not complete_proc:
        reasons.append("missing_or_invalid_memory_evidence")
    times = [sample.get("monotonic_seconds", started) for sample in proc]
    if (len(times) < 2 or times[0] > started + 2 or times[-1] < ended - 2
            or any(b - a > 5 for a, b in zip(times, times[1:]))):
        reasons.append("incomplete_memory_sample_coverage")
    available = [sample["system_ram_total_bytes"] - sample["system_ram_unavailable_bytes"] for sample in valid_proc]
    minimum_available = min(available) if available else None
    if minimum_available is None or minimum_available < POLICY["minimum_available_bytes"]:
        reasons.append("available_memory_below_512_mib_or_unknown")
    counter_deltas = {}
    for field in ("swap_in_pages", "swap_out_pages", "system_oom_kills"):
        values = [sample[field] for sample in valid_proc]
        deltas = [b - a for a, b in zip(values, values[1:])]
        counter_deltas[field] = sum(deltas) if len(values) > 1 and min(deltas) >= 0 and complete_proc else None
        if counter_deltas[field] is None:
            reasons.append(field + "_unverified")
        elif counter_deltas[field] > 0:
            reasons.append(field + "_increased")
    growth = {}
    for metric in ("process_rss_bytes", "system_ram_unavailable_bytes"):
        first = [sample[metric] for sample in valid_proc if 60 <= sample["monotonic_seconds"] - started < 120]
        final = [sample[metric] for sample in valid_proc if ended - 60 <= sample["monotonic_seconds"] <= ended]
        initial_median = statistics.median(first) if first else None
        final_median = statistics.median(final) if final else None
        delta = final_median - initial_median if first and final else None
        tolerance = max(64 << 20, initial_median * .05) if first else None
        growth[metric] = {"baseline_median": initial_median, "final_median": final_median,
                          "growth_bytes": delta, "baseline_samples": len(first), "final_samples": len(final),
                          "tolerance_bytes": tolerance if metric == "process_rss_bytes" else None}
        if not first or not final:
            reasons.append(metric + "_growth_unverified")
        elif metric == "process_rss_bytes" and delta > tolerance:
            reasons.append("backend_rss_growth_exceeded_frozen_tolerance")
    valid_health = [sample for sample in health if not sample.get("error") and sample.get("http_status") == 200
                    and isinstance(sample.get("data"), dict) and sample["data"].get("status") == "ready"
                    and all(type(sample["data"].get(field)) is int and sample["data"][field] >= 0
                            for field in ("active_requests", "queued_requests"))]
    if len(valid_health) != len(health) or len(health) < 2:
        reasons.append("missing_unready_or_invalid_health_evidence")
    health_times = [sample["monotonic_seconds"] for sample in health]
    if (len(health_times) < 2 or health_times[0] > started + 3 or health_times[-1] < ended - 5
            or any(b - a > 5 for a, b in zip(health_times, health_times[1:]))):
        reasons.append("incomplete_health_sample_coverage")
    peak_active = max((sample["data"]["active_requests"] for sample in valid_health), default=None)
    peak_queued = max((sample["data"]["queued_requests"] for sample in valid_health), default=None)
    if peak_active is None or peak_active > 1 or peak_queued is None or peak_queued > 1:
        reasons.append("active_or_queued_requests_exceeded_one_or_unknown")
    memory = summarize_memory(samples, list(sampler_errors))
    memory.pop("samples")
    return {"kind": "soak_summary", "status": "passed" if not reasons else "failed",
            "scope": "API streaming soak; not browser/camera validation or semantic quality grading",
            "elapsed_seconds": elapsed, "requested_seconds": requested_seconds, "policy": POLICY,
            "request_count": len(records), "request_failure_count": len(failures),
            "case_counts": {case: sum(record["case_id"] == case for record in records) for case in expected_cases},
            "length_finish_count": sum(record.get("finish_reason") == "length" for record in records),
            "minimum_available_bytes": minimum_available, "counter_deltas": counter_deltas,
            "growth": growth, "memory": memory, "resource_sample_count": len(proc),
            "oom_note": "The sampled oom_kill counter is system-wide. CUDA/backend failures also reject the run as request errors; this is not CUDA allocator instrumentation.",
            "health_sample_count": len(health), "peak_active_requests": peak_active,
            "peak_queued_requests": peak_queued, "failure_reasons": sorted(set(reasons))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8090/v1/chat/completions")
    parser.add_argument("--health-url", help="Default: /health/ready at the request endpoint origin")
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--pid", type=int, required=True, help="Local backend PID, not proxy PID")
    parser.add_argument("--backend-config", type=Path, required=True, help="JSON object recording selected engine/runtime provenance")
    parser.add_argument("--output", type=Path, required=True, help="New raw JSONL; sibling .summary.json is also created exclusively")
    parser.add_argument("--duration", type=float, default=600, help="Seconds, at least 600 and at most 3600 (default 600)")
    parser.add_argument("--timeout", type=float, default=120, help="Hard per-request wall timeout, 1–120 seconds")
    parser.add_argument("--tegrastats", help="Optional tegrastats executable; proc evidence remains mandatory")
    args = parser.parse_args(argv)
    if not 600 <= args.duration <= 3600 or not 1 <= args.timeout <= 120 or args.pid <= 0:
        parser.error("Require 600–3600 seconds duration, 1–120 seconds timeout and a positive backend PID")
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        parser.error("Run on the Orin so sampled RAM/PID counters describe the actual backend host")
    identity = Path("/proc/device-tree/model").read_bytes().replace(b"\0", b" ").decode()
    if "Jetson" not in identity or "Orin" not in identity:
        parser.error("Expected Jetson Orin device-tree identity")
    url = local_url(args.url)
    parts = urlsplit(url)
    health_url = local_url(args.health_url or urlunsplit((parts.scheme, parts.netloc, "/health/ready", "", "")))
    manifest, manifest_hash, workloads = load_workloads(ROOT / "benchmarks/fixtures/jpeg-manifest.json")
    before_pid, before_provenance = pid_identity(args.pid), provenance(args.backend_config)
    config = {"kind": "soak_config", "created_utc": utc_now(), "candidate_id": args.candidate_id,
              "scope": "API streaming soak; frozen semantic quality and real browser/camera checks are separate",
              "endpoint": url, "health_url": health_url, "duration_seconds": args.duration,
              "request_timeout_seconds": args.timeout, "maximum_drain_seconds": args.timeout,
              "policy": POLICY, "suite_id": manifest["suite_id"], "manifest_sha256": manifest_hash,
              "workloads": [{key: value for key, value in item.items() if key != "payload"} for item in workloads],
              "pid_before": before_pid, "provenance_before": before_provenance,
              "host": socket.gethostname(), "device": identity, "python": sys.version,
              "memory_interval_seconds": .5, "health_interval_seconds": 1}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.output.with_suffix(".summary.json")
    if args.output.resolve() == summary_path.resolve():
        raise ValueError("Raw output must have a different name from its .summary.json companion")
    # Reserve both evidence files before making any request; never replace an earlier run.
    with summary_path.open("x") as summary_file, args.output.open("x") as output:
        write_lock = threading.Lock()
        def write(record):
            with write_lock:
                output.write(json.dumps(record, allow_nan=False) + "\n")
                output.flush()
        write(config)  # The policy is frozen on disk before the first request.
        print(json.dumps({"raw": str(args.output), "summary": str(summary_path), "policy": POLICY}), flush=True)
        sampler = LocalSampler(.5, args.pid, args.tegrastats)
        observer = Observer(sampler, health_url, write)
        records, errors = [], []
        started = time.monotonic()
        try:
            sampler.start()
            observer.start()
            started = time.monotonic()
            write({"kind": "soak_started", "monotonic_seconds": started, "observed_utc": utc_now()})
            stream_sequence(url, workloads, records, write, started=started, duration=args.duration,
                            timeout=args.timeout)
        except (Exception, KeyboardInterrupt) as error:
            errors.append(type(error).__name__ + ": " + str(error))
        finally:
            ended = time.monotonic()  # Teardown time is not counted toward the ten minutes.
            if observer.thread.ident is not None and not observer.stop():
                errors.append("health_observer_did_not_stop_within_four_seconds")
            sampler.capture()
            sampler.stop()
            if any(thread.is_alive() for thread in sampler.threads):
                errors.append("resource_observer_did_not_stop")
            observer.flush_memory()
        after_pid = after_provenance = None
        try:
            after_pid, after_provenance = pid_identity(args.pid), provenance(args.backend_config)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            errors.append("Final provenance unavailable: " + str(error))
        summary = evaluate(records, sampler.samples, observer.health, started=started, ended=ended,
                           expected_cases=[item["case_id"] for item in workloads], requested_seconds=args.duration,
                           sampler_errors=sampler.errors, run_errors=errors,
                           pid_unchanged=before_pid == after_pid, provenance_unchanged=before_provenance == after_provenance)
        summary.update(candidate_id=args.candidate_id, completed_utc=utc_now(), raw_jsonl=str(args.output),
                       manifest_sha256=manifest_hash, pid_after=after_pid, provenance_after=after_provenance)
        write(summary)
        json.dump(summary, summary_file, indent=2, allow_nan=False)
        summary_file.write("\n")
    print(json.dumps({"status": summary["status"], "request_count": len(records),
                      "elapsed_seconds": summary["elapsed_seconds"], "failure_reasons": summary["failure_reasons"]}), flush=True)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
