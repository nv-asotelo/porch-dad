#!/usr/bin/env python3
"""Compare one candidate with the last accepted run under the frozen latency10 plan.

Reads benchmark JSONL only; never runs inference. Exit 0 accepts, 1 stops the
search, and 2 indicates malformed/incomplete evidence. Outputs are exclusive.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


PROJECT = Path(__file__).resolve().parents[1]
MEMORY_FLOOR = 512 * 1024 ** 2
INVARIANTS = ("model", "model_revision", "backend_revision", "precision", "max_input_len",
              "max_kv_cache_capacity", "max_batch_size", "encoder_embedding_cache_budget_bytes",
              "text_context_reuse", "power_mode", "clock_policy", "sampling")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def number(value, *, positive: bool = False) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and (value > 0 if positive else value >= 0))


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def read_run(path: Path, fixture: dict, plan: dict) -> dict:
    raw = path.read_bytes()
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    check(all(isinstance(record, dict) for record in records), f"{path}: non-object record")
    check(records and records[0].get("kind") == "run_config" and records[-1].get("kind") == "summary",
          f"{path}: missing initial config or final summary")
    config, summary, requests = records[0], records[-1], records[1:-1]
    check(all(record.get("kind") == "request" for record in requests), f"{path}: unexpected record kind")
    workload = config.get("workload", {})
    fingerprint = sha256(json.dumps(workload, sort_keys=True).encode())
    check(config.get("workload_fingerprint") == fingerprint, f"{path}: invalid workload fingerprint")
    expected = {"prompt": plan["prompt"], "max_tokens": plan["max_tokens"], "temperature": plan["temperature"],
                "top_p": None, "concurrency": 1, "image_sha256": fixture["sha256"],
                "image_bytes": fixture["size_bytes"], "image_mime_type": "image/jpeg",
                "stream_options": {"include_usage": True}}
    check(all(key in workload and workload[key] == value for key, value in expected.items()),
          f"{path}: workload differs from frozen plan/fixture")
    check("top_k" not in workload and "seed" not in workload, f"{path}: unexpected sampling override")
    check(config.get("requests") == 30 and config.get("warmup") == 5,
          f"{path}: expected 30 measured requests and 5 warmups")
    check(len(requests) == 35, f"{path}: expected exactly 35 request records")
    for index, request in enumerate(requests):
        check(request.get("phase") == ("warmup" if index < 5 else "measured")
              and request.get("index") == index, f"{path}: invalid request sequence at {index}")
    for record in [*requests, summary]:
        check(record.get("run_id") == config.get("run_id")
              and record.get("candidate_id") == config.get("candidate_id")
              and record.get("workload_fingerprint") == fingerprint,
              f"{path}: request/summary identity mismatch")
    check(summary.get("measured_requests") == 30 and summary.get("warmup_requests") == 5,
          f"{path}: summary count mismatch")
    sampling = config.get("memory_sampling", {})
    check(sampling.get("enabled") is True and sampling.get("host"), f"{path}: local sampling unavailable")
    backend = config.get("backend_config", {})
    check(all(key in backend for key in INVARIANTS), f"{path}: incomplete backend invariants")
    actual_sampling = backend["sampling"]
    expected_sampling = {"temperature": .7, "top_p": .9, "top_k": 50,
                         "request_top_p": "omitted", "request_top_k": "omitted",
                         "native_philox_seed": 42, "native_philox_offset": 0}
    check(isinstance(actual_sampling, dict)
          and all(actual_sampling.get(key) == value for key, value in expected_sampling.items()),
          f"{path}: resolved sampling provenance differs from frozen plan")
    check(backend["encoder_embedding_cache_budget_bytes"] == 0 and backend["text_context_reuse"] is False,
          f"{path}: request caches must remain disabled")

    failures = []
    proc = []
    for index, request in enumerate(requests):
        if ("error" not in request or request["error"] is not None or request.get("http_status") != 200
                or request.get("stream_done") is not True or request.get("finish_reason") != "stop"):
            failures.append(f"request_{index}_error_or_incomplete")
        memory = request.get("memory", {})
        if memory.get("sampler_errors") or memory.get("oom_detected") is not False \
                or memory.get("swap_activity_detected") is not False:
            failures.append(f"request_{index}_memory_safety_unverified_or_failed")
        samples = [sample for sample in memory.get("samples", []) if sample.get("kind") == "proc"]
        check(len(samples) >= 2, f"{path}: request {index} lacks raw memory samples")
        for sample in samples:
            check(not sample.get("error"), f"{path}: failed raw memory sample")
            for field in ("system_ram_total_bytes", "system_ram_unavailable_bytes", "process_rss_bytes",
                          "swap_in_pages", "swap_out_pages", "system_oom_kills"):
                check(number(sample.get(field)), f"{path}: missing/invalid {field}")
            check(sample["system_ram_total_bytes"] >= sample["system_ram_unavailable_bytes"],
                  f"{path}: impossible available RAM")
        proc.extend(samples)
    # Check raw counters across requests too; per-request flags alone miss gaps.
    for field in ("swap_in_pages", "swap_out_pages", "system_oom_kills"):
        if len({sample[field] for sample in proc}) != 1:
            failures.append(field + "_changed_or_reset")
    measured = requests[5:]
    for request in measured:
        check(number(request.get("total_latency_ms"), positive=True)
              and number(request.get("ttft_ms"), positive=True)
              and request["ttft_ms"] <= request["total_latency_ms"], f"{path}: invalid measured latency")
    answers = {(request.get("output_text"), request.get("completion_tokens"), request.get("finish_reason"))
               for request in measured}
    token_valid = all(isinstance(request.get("completion_tokens"), int)
                      and not isinstance(request["completion_tokens"], bool)
                      and request["completion_tokens"] > 0
                      and isinstance(request.get("output_text"), str) and request["output_text"].strip()
                      and isinstance(request.get("usage"), dict)
                      and request["usage"].get("completion_tokens") == request["completion_tokens"]
                      for request in measured)
    if len(answers) != 1 or not token_valid:
        failures.append("measured_answers_or_real_token_counts_not_repeatable")
    answer, tokens, finish = next(iter(answers)) if len(answers) == 1 else (None, None, None)
    available = min(sample["system_ram_total_bytes"] - sample["system_ram_unavailable_bytes"] for sample in proc)
    if available < MEMORY_FLOOR:
        failures.append("available_shared_ram_below_512_mib")
    latencies = [request["total_latency_ms"] for request in measured]
    ttfts = [request["ttft_ms"] for request in measured]
    return {"fixture": fixture["file"], "source": str(path), "source_sha256": sha256(raw),
            "run_id": config["run_id"], "candidate_id": config["candidate_id"],
            "workload_fingerprint": fingerprint, "endpoint": config["endpoint"],
            "sampling_host": sampling["host"], "backend_config": backend,
            "measured_requests": 30, "warmup_requests": 5,
            "p50_latency_ms": statistics.median(latencies), "p95_latency_ms": percentile(latencies, .95),
            "p50_ttft_ms": statistics.median(ttfts), "p95_ttft_ms": percentile(ttfts, .95),
            "completion_tokens": tokens, "answer": answer, "finish_reason": finish,
            "unique_measured_answers": len(answers),
            "peak_shared_ram_unavailable_bytes": max(sample["system_ram_unavailable_bytes"] for sample in proc),
            "peak_process_rss_bytes": max(sample["process_rss_bytes"] for sample in proc),
            "minimum_available_shared_ram_bytes": available,
            "raw_counter_start": {key: proc[0][key] for key in ("swap_in_pages", "swap_out_pages", "system_oom_kills")},
            "raw_counter_end": {key: proc[-1][key] for key in ("swap_in_pages", "swap_out_pages", "system_oom_kills")},
            "failures": sorted(set(failures))}


def read_group(directory: Path, prefix: str, manifest: dict, plan: dict) -> dict:
    check(prefix and not any(char in prefix for char in "/\\*?[]"), "Prefix must be a literal filename prefix")
    files = sorted(directory.glob(prefix + "-*.jsonl"))
    fixtures = {fixture["sha256"]: fixture for fixture in manifest["fixtures"]}
    check(len(files) == len(fixtures), f"{directory}/{prefix}-*: expected exactly {len(fixtures)} files")
    by_name = {}
    for path in files:
        with path.open() as handle:
            config = json.loads(next(handle))
        digest = config.get("workload", {}).get("image_sha256")
        check(digest in fixtures, f"{path}: image not in frozen manifest")
        fixture = fixtures[digest]
        check(fixture["file"] not in by_name, f"{path}: duplicate fixture")
        by_name[fixture["file"]] = read_run(path, fixture, plan)
    runs = [by_name[name] for name in plan["fixture_order"]]
    check(len({run["candidate_id"] for run in runs}) == 1, f"{prefix}: mixed candidate IDs")
    failures = [f'{run["fixture"]}: {failure}' for run in runs for failure in run["failures"]]
    for previous, current in zip(runs, runs[1:]):
        if previous["raw_counter_end"] != current["raw_counter_start"]:
            failures.append("swap_or_oom_counter_changed_between_fixtures")
        check(all(previous["backend_config"][key] == current["backend_config"][key] for key in INVARIANTS)
              and previous["endpoint"] == current["endpoint"]
              and previous["sampling_host"] == current["sampling_host"], f"{prefix}: inconsistent run provenance")
    return {"candidate_id": runs[0]["candidate_id"], "per_fixture": runs,
            "geomean_p50_complete_latency_ms": math.exp(statistics.mean(math.log(run["p50_latency_ms"]) for run in runs)),
            "peak_shared_ram_unavailable_bytes": max(run["peak_shared_ram_unavailable_bytes"] for run in runs),
            "peak_process_rss_bytes": max(run["peak_process_rss_bytes"] for run in runs),
            "minimum_available_shared_ram_bytes": min(run["minimum_available_shared_ram_bytes"] for run in runs),
            "failures": sorted(set(failures))}


def compare(baseline: dict, candidate: dict) -> dict:
    failures = [f"baseline: {failure}" for failure in baseline["failures"]]
    failures += [f"candidate: {failure}" for failure in candidate["failures"]]
    comparisons = []
    for before, after in zip(baseline["per_fixture"], candidate["per_fixture"], strict=True):
        check(before["fixture"] == after["fixture"], "Fixture ordering mismatch")
        name = before["fixture"]
        if before["workload_fingerprint"] != after["workload_fingerprint"]:
            failures.append(name + ": workload_fingerprint_mismatch")
        differing = [key for key in INVARIANTS if before["backend_config"][key] != after["backend_config"][key]]
        if differing or before["endpoint"] != after["endpoint"] or before["sampling_host"] != after["sampling_host"]:
            failures.append(name + ": invariant_provenance_mismatch " + ",".join(differing))
        equivalent = (before["unique_measured_answers"] == after["unique_measured_answers"] == 1
                      and before["answer"] == after["answer"] and before["completion_tokens"] == after["completion_tokens"]
                      and before["finish_reason"] == after["finish_reason"] == "stop")
        if not equivalent:
            failures.append(name + ": output_or_token_count_not_equivalent")
        ratios = {key: after[key] / before[key] for key in ("p50_latency_ms", "p95_latency_ms")}
        if any(ratio > 1.05 + 1e-12 for ratio in ratios.values()):
            failures.append(name + ": per_fixture_latency_regression_exceeds_5_percent")
        comparisons.append({"fixture": name, "ratios": ratios, "output_equivalent": equivalent})
    ratio = candidate["geomean_p50_complete_latency_ms"] / baseline["geomean_p50_complete_latency_ms"]
    if ratio > .9 + 1e-12:
        failures.append("geomean_complete_latency_gain_below_10_percent")
    return {"accepted": not failures,
            "decision": "accept_candidate" if not failures else "stop_and_restore_last_accepted",
            "candidate_to_baseline_ratio": ratio, "latency_reduction_percent": 100 * (1 - ratio),
            "per_fixture": comparisons, "rejection_reasons": sorted(set(failures))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--baseline-prefix", default="baseline")
    parser.add_argument("--candidate-prefix", default="candidate")
    parser.add_argument("--plan", type=Path, default=PROJECT / "results/latency10/plan.json")
    parser.add_argument("--manifest", type=Path, default=PROJECT / "benchmarks/live-vlm-1280/manifest.json")
    parser.add_argument("--output", type=Path, help="New JSON receipt; existing files are never overwritten")
    args = parser.parse_args(argv)
    try:
        plan_raw, manifest_raw = args.plan.read_bytes(), args.manifest.read_bytes()
        plan, manifest = json.loads(plan_raw), json.loads(manifest_raw)
        check(plan["warmups_per_fixture"] == 5 and plan["measured_requests_per_fixture"] == 30
              and plan["max_tokens"] == 512 and plan["temperature"] == .7
              and plan["top_p_request"] == plan["top_k_request"] == "omitted"
              and plan["backend_resolved_top_p"] == .9 and plan["backend_resolved_top_k"] == 50,
              "Plan does not match this frozen comparator")
        check(manifest["width"] == 1280 and manifest["height"] == 720
              and plan["fixture_order"] == [fixture["file"] for fixture in manifest["fixtures"]],
              "Manifest dimensions/order mismatch")
        baseline = read_group(args.baseline_dir, args.baseline_prefix, manifest, plan)
        candidate = read_group(args.candidate_dir, args.candidate_prefix, manifest, plan)
        result = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                  "plan": {"path": str(args.plan), "sha256": sha256(plan_raw)},
                  "manifest": {"path": str(args.manifest), "sha256": sha256(manifest_raw)},
                  "comparator_sha256": sha256(Path(__file__).read_bytes()),
                  "primary_metric": plan["primary_metric"], "baseline": baseline, "candidate": candidate,
                  "comparison": compare(baseline, candidate),
                  "memory_scope": "Sampled extrema across warmup and measured requests; shared RAM includes OS/apps; RSS overlaps and is not added.",
                  "quality_scope": "Byte-identical measured answers establish preservation, not correctness. Baseline mistakes remain mistakes.",
                  "queue_scope": "Serial harness concurrency is verified; absence of other clients/queue backlog requires the separate device readiness/isolation evidence.",
                  "limits": plan.get("limits", [])}
        rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x") as handle:
                handle.write(rendered)
        print(rendered, end="")
        return 0 if result["comparison"]["accepted"] else 1
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as exc:
        print(json.dumps({"accepted": False, "decision": "invalid_evidence_do_not_accept", "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
