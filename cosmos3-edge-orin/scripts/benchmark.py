#!/usr/bin/env python3
"""Reproducible OpenAI-compatible vision streaming benchmarks, using stdlib only."""

from __future__ import annotations

import argparse
import base64
import codecs
import hashlib
import json
import math
import mimetypes
import os
from pathlib import Path
import platform
import re
import socket
import statistics
import subprocess
import sys
import threading
import time
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen
import uuid


class SSEDecoder:
    """Decode arbitrary UTF-8 chunk boundaries and SSE LF/CRLF/CR line endings."""

    def __init__(self) -> None:
        self.decoder = codecs.getincrementaldecoder("utf-8")()
        self.buffer = ""
        self.data: list[str] = []
        self.first_line = True

    def _line(self, line: str) -> list[str]:
        if self.first_line:
            line = line.removeprefix("\ufeff")
            self.first_line = False
        if not line:
            if self.data:
                event = "\n".join(self.data)
                self.data = []
                return [event]
            return []
        if line.startswith(":"):
            return []
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "data":
            self.data.append(value)
        return []

    def feed(self, chunk: bytes, final: bool = False) -> list[str]:
        self.buffer += self.decoder.decode(chunk, final=final)
        events: list[str] = []
        while True:
            match = re.search(r"[\r\n]", self.buffer)
            if not match:
                break
            pos = match.start()
            if self.buffer[pos] == "\r" and pos + 1 == len(self.buffer) and not final:
                break  # The next chunk may begin with the LF of CRLF.
            width = 2 if self.buffer[pos:pos + 2] == "\r\n" else 1
            events.extend(self._line(self.buffer[:pos]))
            self.buffer = self.buffer[pos + width:]
        if final:
            # SSE dispatches only on an empty line; discard an unfinished event.
            self.buffer = ""
            self.data = []
        return events


def iter_sse(stream: Any):
    decoder = SSEDecoder()
    read = getattr(stream, "read1", stream.read)
    while True:
        chunk = read(4096)
        if not chunk:
            yield from decoder.feed(b"", final=True)
            return
        yield from decoder.feed(chunk)


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content
                       if isinstance(part, dict) and isinstance(part.get("text"), str))
    return ""


def stream_request(url: str, payload: dict, timeout: float, api_key: str | None = None) -> dict:
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    request = Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    started = time.perf_counter()
    first_text = None
    usage = None
    cosmos_metrics = None
    output: list[str] = []
    finished = False
    error = None
    status = None
    finish_reason = None
    try:
        with urlopen(request, timeout=timeout) as response:
            status = response.status
            if "text/event-stream" not in response.headers.get("Content-Type", "").lower():
                raise ValueError("Expected Content-Type text/event-stream")
            for event in iter_sse(response):
                if time.perf_counter() - started > timeout:
                    raise TimeoutError("Request exceeded total timeout")
                if event.strip() == "[DONE]":
                    finished = True
                    break
                if not event.strip():
                    continue
                item = json.loads(event)
                if not isinstance(item, dict):
                    raise ValueError("SSE JSON must be an object")
                if item.get("error"):
                    raise ValueError("Backend returned an error: " + str(item["error"])[:1000])
                if isinstance(item.get("usage"), dict):
                    usage = item["usage"]
                if isinstance(item.get("cosmos_metrics"), dict):
                    cosmos_metrics = item["cosmos_metrics"]
                choices = item.get("choices") or []
                if len(choices) > 1:
                    raise ValueError("Expected a single completion choice")
                for choice in choices:
                    delta = choice.get("delta") or {}
                    text = content_text(delta.get("content"))
                    if text:
                        if first_text is None:
                            first_text = time.perf_counter() - started
                        output.append(text)
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]
                        if finish_reason in {"error", "cancelled", "canceled"}:
                            raise ValueError("Backend finished with reason: " + finish_reason)
            if not finished:
                raise ValueError("Truncated stream: missing [DONE] event")
            if not "".join(output).strip():
                raise ValueError("Stream completed without visible output text")
            if finish_reason not in {"stop", "length"}:
                raise ValueError("Missing or unsupported terminal finish reason: " + str(finish_reason))
    except HTTPError as exc:
        status = exc.code
        error = {"type": "HTTPError", "message": "HTTP " + str(exc.code)}
    except Exception as exc:  # A failed sample must remain in the JSONL evidence.
        error = {"type": type(exc).__name__, "message": str(exc)[:1200]}
    elapsed = time.perf_counter() - started
    tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
        tokens = None
    return {
        "ttft_ms": first_text * 1000 if first_text is not None else None,
        "total_latency_ms": elapsed * 1000,
        "completion_tokens": tokens,
        "completion_tokens_per_second": tokens / elapsed if tokens is not None and not error else None,
        "throughput_definition": "completion_tokens / total_request_seconds",
        "usage": usage,
        "cosmos_metrics": cosmos_metrics,
        "output_text": "".join(output),
        "http_status": status,
        "finish_reason": finish_reason,
        "stream_done": finished,
        "error": error,
    }


def read_key_values(path: Path, kib: bool = False) -> dict[str, int]:
    values = {}
    for line in path.read_text().splitlines():
        fields = line.replace(":", " ", 1).split()
        if len(fields) >= 2 and fields[1].isdigit():
            values[fields[0]] = int(fields[1]) * (1024 if kib and fields[-1] == "kB" else 1)
    return values


def parse_tegrastats(line: str) -> dict[str, int]:
    result = {}
    for name in ("RAM", "SWAP"):
        match = re.search(r"\b" + name + r"\s+(\d+)/(\d+)MB", line)
        if match:
            result["tegrastats_" + name.lower() + "_used_bytes"] = int(match[1]) * 1024 ** 2
            result["tegrastats_" + name.lower() + "_total_bytes"] = int(match[2]) * 1024 ** 2
    return result


class LocalSampler:
    """Observe the host running this script; no inference about a remote device."""

    def __init__(self, interval: float, pid: int | None, tegrastats: str | None) -> None:
        self.interval = interval
        self.pid = pid
        self.tegrastats = tegrastats
        self.samples: list[dict] = []
        self.errors: list[str] = []
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.threads: list[threading.Thread] = []
        self.process = None

    def append(self, sample: dict) -> int:
        with self.lock:
            index = len(self.samples)
            sample.update(sequence=index, monotonic_seconds=time.monotonic())
            self.samples.append(sample)
            return index

    def capture(self) -> int:
        sample: dict[str, Any] = {"kind": "proc"}
        try:
            mem = read_key_values(Path("/proc/meminfo"), kib=True)
            vm = read_key_values(Path("/proc/vmstat"))
            if "MemTotal" in mem and "MemAvailable" in mem:
                sample["system_ram_unavailable_bytes"] = mem["MemTotal"] - mem["MemAvailable"]
            sample["system_ram_total_bytes"] = mem.get("MemTotal")
            if "SwapTotal" in mem and "SwapFree" in mem:
                sample["system_swap_occupied_bytes"] = mem["SwapTotal"] - mem["SwapFree"]
            sample["swap_in_pages"] = vm.get("pswpin")
            sample["swap_out_pages"] = vm.get("pswpout")
            sample["system_oom_kills"] = vm.get("oom_kill")
            if self.pid is not None:
                status = read_key_values(Path(f"/proc/{self.pid}/status"), kib=True)
                sample["process_rss_bytes"] = status.get("VmRSS")
        except OSError as exc:
            sample["error"] = str(exc)
        return self.append(sample)

    def _poll(self) -> None:
        while not self.stop_event.wait(self.interval):
            self.capture()

    def _read_tegra(self) -> None:
        assert self.process and self.process.stdout
        for line in self.process.stdout:
            self.append({"kind": "tegrastats", "raw": line.strip()[:8192], **parse_tegrastats(line)})

    def start(self) -> None:
        self.capture()
        thread = threading.Thread(target=self._poll, daemon=True)
        thread.start()
        self.threads.append(thread)
        if self.tegrastats:
            try:
                self.process = subprocess.Popen(
                    [self.tegrastats, "--interval", str(max(10, round(self.interval * 1000)))],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                )
                thread = threading.Thread(target=self._read_tegra, daemon=True)
                thread.start()
                self.threads.append(thread)
            except OSError as exc:
                self.errors.append(str(exc))

    def since(self, sequence: int) -> dict:
        end = self.capture()
        with self.lock:
            samples = list(self.samples[sequence:end + 1])
        return summarize_memory(samples, self.errors)

    def clear_history(self) -> None:
        # Raw samples have already been written to JSONL. Keep observer memory bounded.
        with self.lock:
            self.samples.clear()

    def stop(self) -> None:
        self.stop_event.set()
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        for thread in self.threads:
            thread.join(timeout=2)


def summarize_memory(samples: list[dict], errors: list[str] | None = None) -> dict:
    metrics = {
        "local_system_ram_unavailable_peak_bytes": "system_ram_unavailable_bytes",
        "local_process_rss_peak_bytes": "process_rss_bytes",
        "tegrastats_ram_used_peak_bytes": "tegrastats_ram_used_bytes",
        "local_swap_occupied_peak_bytes": "system_swap_occupied_bytes",
    }
    result: dict[str, Any] = {"samples": samples, "sampler_errors": list(errors or [])}
    for destination, source in metrics.items():
        values = [s[source] for s in samples if s.get(source) is not None]
        result[destination] = max(values) if values else None
    proc = [sample for sample in samples if sample["kind"] == "proc"]

    def activity(keys: tuple[str, ...]) -> bool | None:
        if len(proc) < 2 or any(sample.get("error") for sample in proc):
            return None
        before, after = proc[0], proc[-1]
        if any(before.get(key) is None or after.get(key) is None for key in keys):
            return None
        deltas = [after[key] - before[key] for key in keys]
        if any(delta < 0 for delta in deltas):
            return None
        return any(delta > 0 for delta in deltas)

    result["swap_activity_detected"] = activity(("swap_in_pages", "swap_out_pages"))
    result["oom_detected"] = activity(("system_oom_kills",))
    result["cuda_allocator_allocated_bytes"] = None
    result["cuda_allocator_reserved_bytes"] = None
    result["cuda_allocator_note"] = "Requires backend instrumentation; cannot be inferred from RAM or RSS."
    result["memory_note"] = (
        "Local host observations. Jetson GPU/CPU share RAM. RAM, RSS and allocator counters overlap; "
        "do not add them or label system RAM as dedicated VRAM. Peaks are sampled, not exact."
    )
    return result


def percentile(values: list[float], fraction: float) -> float | None:
    """Linearly interpolated empirical percentile, including endpoints."""
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def aggregate_flag(records: list[dict], name: str) -> bool | None:
    flags = [record.get("memory", {}).get(name) for record in records]
    if any(flag is True for flag in flags):
        return True
    if flags and all(flag is False for flag in flags):
        return False
    return None


def summarize_run(records: list[dict], config: dict, args: argparse.Namespace) -> dict:
    measured = [record for record in records if record["phase"] == "measured"]
    successful = [record for record in measured if record["error"] is None]
    latencies = [record["total_latency_ms"] for record in successful]
    ttfts = [record["ttft_ms"] for record in successful if record["ttft_ms"] is not None]
    memory_values = [record.get("memory", {}).get(args.memory_metric) for record in measured]
    memory_values = [value for value in memory_values if value is not None]
    return {
        "kind": "summary", "candidate_id": args.candidate_id,
        "workload_fingerprint": config["workload_fingerprint"],
        "measured_requests": len(measured), "successful_requests": len(successful),
        "warmup_requests": len(records) - len(measured),
        "error_count": len(measured) - len(successful),
        "warmup_error_count": sum(record["error"] is not None for record in records if record["phase"] == "warmup"),
        "p50_latency_ms": statistics.median(latencies) if latencies else None,
        "p95_latency_ms": percentile(latencies, .95),
        "p50_ttft_ms": statistics.median(ttfts) if ttfts else None,
        "p95_ttft_ms": percentile(ttfts, .95),
        "memory_metric": args.memory_metric,
        "memory_bytes": max(memory_values) if memory_values else None,
        "oom_detected": aggregate_flag(records, "oom_detected"),
        "swap_activity_detected": aggregate_flag(records, "swap_activity_detected"),
        "quality_pass": {"unknown": None, "pass": True, "fail": False}[args.quality],
        "quality_note": args.quality_note,
        "recommendation": "Use at least 30 measured requests per candidate, after warmup.",
    }


def candidate_rejection(candidate: dict, minimum_requests: int) -> str | None:
    if candidate.get("quality_pass") is not True:
        return "quality_not_verified_pass"
    if candidate.get("error_count") != 0 or candidate.get("warmup_error_count", 0) != 0:
        return "request_errors"
    if candidate.get("oom_detected") is not False:
        return "oom_detected_or_unverified"
    if candidate.get("swap_activity_detected") is not False:
        return "swap_activity_detected_or_unverified"
    if not candidate.get("workload_fingerprint"):
        return "missing_workload_fingerprint"
    if not candidate.get("memory_metric"):
        return "missing_memory_metric"
    if candidate.get("measured_requests", 0) < minimum_requests:
        return "insufficient_measured_requests"
    for key in ("p50_latency_ms", "p95_latency_ms", "memory_bytes"):
        value = candidate.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            return "invalid_" + key
    if candidate["p95_latency_ms"] < candidate["p50_latency_ms"]:
        return "p95_below_p50"
    return None


def evaluate_candidates(candidates: list[dict], maximum_candidates: int = 12,
                        maximum_non_improvements: int = 3, minimum_requests: int = 30) -> dict:
    """Select an incumbent in input order. Baseline counts toward the 12-candidate cap."""
    if not 1 <= maximum_candidates <= 12 or not 1 <= maximum_non_improvements <= 3:
        raise ValueError("Limits may be tightened but cannot exceed 12 candidates / 3 non-improvements")
    if minimum_requests < 1:
        raise ValueError("minimum_requests must be positive")
    incumbent = None
    consecutive = 0
    decisions = []
    stop_reason = "input_exhausted"
    for candidate in candidates:
        if len(decisions) >= maximum_candidates:
            stop_reason = "candidate_budget_reached"
            break
        reason = candidate_rejection(candidate, minimum_requests)
        accepted = False
        if reason is None and incumbent is None:
            accepted, reason = True, "initial_eligible_baseline"
        elif reason is None:
            if candidate["workload_fingerprint"] != incumbent["workload_fingerprint"]:
                reason = "workload_mismatch"
            elif candidate["memory_metric"] != incumbent["memory_metric"]:
                reason = "memory_metric_mismatch"
            else:
                ratios = {key: candidate[key] / incumbent[key] for key in
                          ("p50_latency_ms", "p95_latency_ms", "memory_bytes")}
                # Tiny tolerance avoids floating-point rejection at exactly 5%.
                tolerance = 1e-12
                gain = min(ratios["p50_latency_ms"], ratios["memory_bytes"]) <= .95 + tolerance
                regression = max(ratios.values()) > 1.05 + tolerance
                accepted = gain and not regression
                reason = "accepted_improvement" if accepted else (
                    "regression_exceeds_5_percent" if regression else "improvement_below_5_percent")
        if accepted:
            incumbent = candidate
            consecutive = 0
        else:
            consecutive += 1
        decisions.append({"candidate_id": candidate.get("candidate_id"), "accepted": accepted,
                          "reason": reason, "consecutive_non_improvements": consecutive})
        if consecutive >= maximum_non_improvements:
            stop_reason = "consecutive_non_improvement_limit_reached"
            break
    if len(decisions) >= maximum_candidates and stop_reason == "input_exhausted":
        stop_reason = "candidate_budget_reached"
    return {"selected_candidate": incumbent, "decisions": decisions, "stop_reason": stop_reason,
            "candidates_evaluated": len(decisions), "candidates_ignored_after_stop": len(candidates) - len(decisions),
            "maximum_candidates": maximum_candidates, "maximum_non_improvements": maximum_non_improvements,
            "minimum_measured_requests": minimum_requests,
            "scope": "Best accepted candidate in this bounded experiment; not a global optimum."}


def safe_url(url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
        raise ValueError("Use an explicit HTTP(S) URL without embedded credentials")
    if parts.query or parts.fragment:
        raise ValueError("Put authentication in an environment variable, not the endpoint URL")
    return urlunsplit(parts)


def build_payload(workload: dict, image_bytes: bytes) -> dict:
    # v0.10.1 rejects seed even though its protocol schema exposes the field.
    temperature, top_p = validate_sampling(workload["temperature"], workload["top_p"])
    payload = {key: workload[key] for key in ("model", "max_tokens")}
    payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    payload.update(stream=True, stream_options={"include_usage": True}, messages=[{
        "role": "user", "content": [{"type": "text", "text": workload["prompt"]},
        {"type": "image_url", "image_url": {
            "url": f'data:{workload["image_mime_type"]};base64,' + base64.b64encode(image_bytes).decode()}}]}])
    return payload


def validate_sampling(temperature: float, top_p: float | None) -> tuple[float, float | None]:
    """Validate the pinned server's ranges; None means omit top_p on the wire."""
    for name, value in (("temperature", temperature), ("top_p", top_p)):
        if name == "top_p" and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(name + " must be a finite number")
        if (name == "temperature" and not 0 <= value <= 2) or (name == "top_p" and not 0 < value <= 1):
            raise ValueError("temperature must be in [0, 2]; top-p must be in (0, 1] or 'omit'")
    # Keep historical JSON/fingerprints byte-compatible (0 and 1, not 0.0 and 1.0).
    temperature = int(temperature) if temperature == int(temperature) else temperature
    if top_p is not None and top_p == int(top_p):
        top_p = int(top_p)
    return temperature, top_p


def parse_top_p(value: str) -> float | None:
    return None if value == "omit" else float(value)


def run_benchmark(args: argparse.Namespace) -> int:
    url = safe_url(args.url)
    temperature, top_p = validate_sampling(args.temperature, args.top_p)
    if args.requests < 1 or args.warmup < 0 or args.max_tokens < 1 or args.interval <= 0 or args.timeout <= 0:
        raise ValueError("Requests/max-tokens/interval/timeout must be positive; warmup must be nonnegative")
    if args.quality == "pass" and not args.quality_note:
        raise ValueError("--quality pass requires --quality-note describing the external correctness check")
    if (args.pid or args.tegrastats) and not args.sample_local:
        raise ValueError("--pid and --tegrastats require --sample-local")
    if args.requests < 30:
        print("Warning: fewer than 30 measured requests; treat this as a smoke test.", file=sys.stderr)
    image_path = Path(args.image)
    image_bytes = image_path.read_bytes()
    mime = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    if not mime.startswith("image/"):
        raise ValueError("Image must have a recognized image filename extension")
    workload = {"model": args.model, "prompt": args.prompt, "max_tokens": args.max_tokens,
                "temperature": temperature, "top_p": top_p, "concurrency": 1,
                "image_sha256": hashlib.sha256(image_bytes).hexdigest(), "image_mime_type": mime,
                "image_bytes": len(image_bytes), "stream_options": {"include_usage": True}}
    fingerprint = hashlib.sha256(json.dumps(workload, sort_keys=True).encode()).hexdigest()
    backend_config = json.loads(Path(args.backend_config).read_text()) if args.backend_config else {}
    if not isinstance(backend_config, dict):
        raise ValueError("Backend config must be a JSON object")
    run_id = str(uuid.uuid4())
    config = {"kind": "run_config", "schema_version": 1, "run_id": run_id,
              "candidate_id": args.candidate_id, "endpoint": url, "workload": workload,
              "workload_fingerprint": fingerprint, "backend_config": backend_config,
              "timestamp_unix_seconds": time.time(), "requests": args.requests, "warmup": args.warmup,
              "memory_sampling": {"enabled": args.sample_local, "host": socket.gethostname(),
                                  "pid": args.pid, "interval_seconds": args.interval,
                                  "tegrastats": args.tegrastats, "metric": args.memory_metric},
              "provenance": {"python": sys.version, "platform": platform.platform(),
                             "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    payload = build_payload(workload, image_bytes)
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    if args.api_key_env and not api_key:
        raise ValueError("Requested API-key environment variable is empty or unset")
    sampler = LocalSampler(args.interval, args.pid, args.tegrastats) if args.sample_local else None
    records = []
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves existing experiment evidence.
    with output_path.open("x", encoding="utf-8") as output:
        def write(record: dict) -> None:
            output.write(json.dumps(record, allow_nan=False) + "\n")
            output.flush()
        write(config)
        try:
            if sampler:
                sampler.start()
            for index in range(args.warmup + args.requests):
                phase = "warmup" if index < args.warmup else "measured"
                sequence = sampler.capture() if sampler else None
                result = stream_request(url, payload, args.timeout, api_key)
                memory = sampler.since(sequence) if sampler else {}
                record = {"kind": "request", "run_id": run_id, "candidate_id": args.candidate_id,
                          "phase": phase, "index": index, "workload_fingerprint": fingerprint,
                          "memory": memory, **result}
                write(record)
                records.append({key: value for key, value in record.items()
                                if key not in {"memory", "output_text", "usage"}} | {
                                    "memory": {key: value for key, value in memory.items() if key != "samples"}})
                if sampler:
                    sampler.clear_history()
                print(f"{phase} {index + 1}: " + ("ERROR " + result["error"]["message"] if result["error"]
                      else f'{result["total_latency_ms"]:.1f} ms'), file=sys.stderr)
        finally:
            if sampler:
                sampler.stop()
        summary = {"run_id": run_id, **summarize_run(records, config, args)}
        write(summary)
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 1 if any(record["error"] for record in records) else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Send serial, fixed-workload streamed image requests")
    run.add_argument("--url", required=True, help="Full chat/completions URL")
    run.add_argument("--model", required=True)
    run.add_argument("--image", required=True)
    run.add_argument("--prompt", default="Describe the visible scene in one concise sentence. Focus on objects and actions.")
    run.add_argument("--candidate-id", required=True)
    run.add_argument("--output", required=True, help="New JSONL file; existing files are never overwritten")
    run.add_argument("--requests", type=int, default=30)
    run.add_argument("--warmup", type=int, default=5)
    run.add_argument("--max-tokens", type=int, default=64)
    run.add_argument("--temperature", type=float, default=0, help="Sampling temperature in [0, 2] (default: 0)")
    run.add_argument("--top-p", type=parse_top_p, default=1, metavar="VALUE|omit",
                     help="Top-p in (0, 1], or omit to use server defaults (default: 1). "
                          "Record resolved server defaults in --backend-config when omitting.")
    run.add_argument("--timeout", type=float, default=180)
    run.add_argument("--api-key-env", help="Name of the variable holding the API key; key is never recorded")
    run.add_argument("--backend-config", help="JSON object with engine precision/build/runtime provenance")
    run.add_argument("--sample-local", action="store_true", help="Sample this host, ideally on the Jetson itself")
    run.add_argument("--pid", type=int, help="Local backend PID; RSS excludes child processes")
    run.add_argument("--tegrastats", help="Path/name of tegrastats executable, e.g. /usr/bin/tegrastats")
    run.add_argument("--interval", type=float, default=.1)
    run.add_argument("--memory-metric", choices=["local_system_ram_unavailable_peak_bytes",
                     "local_process_rss_peak_bytes", "tegrastats_ram_used_peak_bytes"],
                     default="local_system_ram_unavailable_peak_bytes")
    run.add_argument("--quality", choices=["unknown", "pass", "fail"], default="unknown")
    run.add_argument("--quality-note", help="Describe the separately performed task correctness check")
    evaluate = commands.add_parser("evaluate", help="Select candidates under the fixed stopping policy")
    evaluate.add_argument("files", nargs="+", help="Run JSONL files, in candidate trial order")
    evaluate.add_argument("--maximum-candidates", type=int, default=12)
    evaluate.add_argument("--maximum-non-improvements", type=int, default=3)
    evaluate.add_argument("--minimum-requests", type=int, default=30)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return run_benchmark(args)
        candidates = []
        for filename in args.files:
            for line in Path(filename).read_text().splitlines():
                item = json.loads(line)
                if item.get("kind") == "summary":
                    candidates.append(item)
        if not candidates:
            raise ValueError("No summary records found")
        result = evaluate_candidates(candidates, args.maximum_candidates, args.maximum_non_improvements,
                                     args.minimum_requests)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result["selected_candidate"] else 1
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
