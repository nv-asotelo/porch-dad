#!/usr/bin/env python3
"""Fit server-only fixed cost and marginal output-token cost from request logs.

Input is JSONL produced by server instrumentation, never a client benchmark:
  {"kind":"server_request", "schema_version":1, "backend_id":"trt-320",
   "request_id":"unique-id", "status":"completed", "warmup":false,
   "timing_source":"server_monotonic", "timing_boundary":"native_inference",
   "server_elapsed_ms":720.4, "completion_tokens":20, "cache_state":"hit",
   "controls":{"image_sha256":"...", "prompt_sha256":"...",
     "max_image_tokens_per_image":320, "temperature":0.7, "top_p":0.95,
     "concurrency":1, "clock_policy":"static", "encoder_cache_bytes":268435456}}

Timing must surround native inference only, after image decoding and queueing,
and finish before network writes. Actual completion tokens come from backend
usage, not the requested cap, words, characters, or SSE event count. Additional
controls are retained and matched. max_tokens is intentionally not a control:
varying this ceiling with otherwise fixed input identifies marginal token cost.

Each backend/input/cache stratum is fitted separately. By default only equal
controls are compared; --vary-control explicitly permits and reports an intended
configuration change. Output is exclusive when --output is used. Exit 0 means
at least one identifiable fit; exit 2 means malformed input or no such fit. A
single fit does not imply that a second backend was measured or is comparable.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import random
import sys


REQUIRED_CONTROLS = (
    "image_sha256", "prompt_sha256", "max_image_tokens_per_image", "temperature",
    "top_p", "concurrency", "clock_policy", "encoder_cache_bytes",
)
# Input identity and sampling cannot be waived to make unrelated workloads match.
VARYABLE_CONTROLS = {"max_image_tokens_per_image", "clock_policy", "encoder_cache_bytes"}
BOUNDARY = "native_inference"


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def number(value, positive=False) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and (value > 0 if positive else value >= 0))


def integer(value, positive=False) -> bool:
    return type(value) is int and (value > 0 if positive else value >= 0)


def invalid_reason(row: dict) -> str | None:
    """Reject unsupported evidence explicitly instead of reinterpreting it."""
    if row.get("kind") != "server_request":
        if row.get("kind") in {"request", "soak_request"} or "total_latency_ms" in row:
            return "client_timing_is_not_server_evidence"
        return "non_request_metadata"
    if type(row.get("schema_version")) is not int or row["schema_version"] != 1:
        return "unsupported_schema"
    if not all(isinstance(row.get(key), str) and row[key].strip()
               for key in ("backend_id", "request_id")):
        return "missing_request_identity"
    if row.get("status") != "completed" or row.get("error") is not None:
        return "failed_or_incomplete_request"
    if type(row.get("warmup")) is not bool:
        return "unknown_warmup_state"
    if row["warmup"]:
        return "warmup"
    if row.get("timing_source") != "server_monotonic" or row.get("timing_boundary") != BOUNDARY:
        return "unsupported_timing_boundary"
    if not number(row.get("server_elapsed_ms"), positive=True):
        return "invalid_server_duration"
    if not integer(row.get("completion_tokens"), positive=True):
        return "missing_actual_completion_tokens"
    if row.get("cache_state") not in {"hit", "miss", "disabled"}:
        return "unknown_cache_state"
    controls = row.get("controls")
    if not isinstance(controls, dict) or any(key not in controls for key in REQUIRED_CONTROLS):
        return "missing_controls"
    for key in ("image_sha256", "prompt_sha256"):
        value = controls[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            return "invalid_input_hash"
    if (not integer(controls["max_image_tokens_per_image"], positive=True)
            or not number(controls["temperature"]) or controls["temperature"] > 2
            or not number(controls["top_p"], positive=True) or controls["top_p"] > 1
            or not integer(controls["concurrency"], positive=True)
            or not integer(controls["encoder_cache_bytes"])
            or not isinstance(controls["clock_policy"], str) or not controls["clock_policy"].strip()):
        return "invalid_controls"
    if controls["concurrency"] != 1:
        return "concurrent_workload_not_supported"
    try:
        canonical(controls)
    except (TypeError, ValueError):
        return "nonfinite_or_unserializable_controls"
    return None


def linear_fit(points: list[tuple[float, float]]) -> dict | None:
    """Centered ordinary least squares; no outlier removal or sign clipping."""
    n = len(points)
    if n < 3:
        return None
    xbar = math.fsum(x for x, _ in points) / n
    ybar = math.fsum(y for _, y in points) / n
    xx = math.fsum((x - xbar) ** 2 for x, _ in points)
    if xx == 0:
        return None
    slope = math.fsum((x - xbar) * (y - ybar) for x, y in points) / xx
    intercept = ybar - slope * xbar
    sse = math.fsum((y - intercept - slope * x) ** 2 for x, y in points)
    sst = math.fsum((y - ybar) ** 2 for _, y in points)
    return {"fixed_ms": intercept, "marginal_ms_per_token": slope,
            "residual_rmse_ms": math.sqrt(sse / (n - 2)),
            "r_squared": 1 - sse / sst if sst else None}


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def interval(values: list[float]) -> list[float]:
    return [percentile(values, .025), percentile(values, .975)]


def bootstrap(points: list[tuple[float, float]], repeats: int, seed: int) -> list[dict]:
    """Circular moving-block bootstrap retains short-range serial correlation.

    Records must remain in request order within each stratum. Block length is
    ceil(n**(1/3)); this is a sensitivity estimate, not a cure for long-run drift.
    """
    rng = random.Random(seed)
    n = len(points)
    width = max(2, math.ceil(n ** (1 / 3)))
    fits = []
    for _ in range(repeats * 10):
        sample = []
        while len(sample) < n:
            start = rng.randrange(n)
            sample.extend(points[(start + i) % n] for i in range(width))
        fitted = linear_fit(sample[:n])
        if fitted is not None:
            fits.append(fitted)
        if len(fits) == repeats:
            return fits
    raise ValueError("Bootstrap could not obtain enough nondegenerate fits")


def compare_records(records: list[dict], *, min_samples=12, min_token_span=8,
                    bootstrap_repeats=2000, seed=2026, vary_controls=()) -> dict:
    if min_samples < 3 or min_token_span < 1 or bootstrap_repeats < 100:
        raise ValueError("Require min_samples>=3, min_token_span>=1, bootstrap_repeats>=100")
    if set(vary_controls) - VARYABLE_CONTROLS:
        raise ValueError("Unsupported varied control; input hashes and sampling must match")
    groups = defaultdict(list)
    excluded = Counter()
    backend_counts = defaultdict(Counter)
    seen = set()
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("Every JSONL record must be an object")
        reason = invalid_reason(row)
        backend = row.get("backend_id")
        if isinstance(backend, str) and backend.strip():
            backend_counts[backend]["records"] += 1
            backend_counts[backend][reason or "accepted_measured_requests"] += 1
        if reason:
            excluded[reason] += 1
            continue
        identity = (row["backend_id"], row["request_id"])
        if identity in seen:
            raise ValueError("Duplicate backend/request identity: " + repr(identity))
        seen.add(identity)
        key = canonical({"backend_id": row["backend_id"], "controls": row["controls"],
                         "cache_state": row["cache_state"]})
        groups[key].append(row)

    strata, fitted_groups = [], []
    for index, (key, rows) in enumerate(sorted(groups.items())):
        identity = json.loads(key)
        points = [(row["completion_tokens"], row["server_elapsed_ms"]) for row in rows]
        lengths = sorted(set(x for x, _ in points))
        result = {**identity, "stratum_id": digest(identity), "measured_requests": len(rows),
                  "actual_output_token_range": [lengths[0], lengths[-1]],
                  "distinct_actual_output_lengths": len(lengths),
                  "actual_output_token_counts": {str(length): sum(x == length for x, _ in points)
                                                  for length in lengths}}
        reasons = []
        if len(rows) < min_samples:
            reasons.append("too_few_measured_requests")
        if len(lengths) < 3:
            reasons.append("fewer_than_three_distinct_actual_output_lengths")
        if lengths[-1] - lengths[0] < min_token_span:
            reasons.append("insufficient_actual_output_token_span")
        if reasons:
            result.update(status="unidentifiable", reasons=reasons, fit=None)
            strata.append(result)
            continue
        fitted = linear_fit(points)
        draws = bootstrap(points, bootstrap_repeats, seed + index)
        for name in ("fixed_ms", "marginal_ms_per_token"):
            fitted[name + "_ci95"] = interval([draw[name] for draw in draws])
        warnings = ["fixed_ms is an extrapolated zero-output intercept, not measured prefill latency"]
        if fitted["fixed_ms"] < 0:
            warnings.append("negative intercept: a literal fixed-time interpretation is unsupported")
        if fitted["marginal_ms_per_token_ci95"][0] <= 0:
            warnings.append("positive marginal token cost is not resolved by the 95% interval")
        if fitted["r_squared"] is not None and fitted["r_squared"] < .8:
            warnings.append("linear output length explains less than 80% of observed timing variation")
        fitted["bootstrap_block_length"] = max(2, math.ceil(len(points) ** (1 / 3)))
        result.update(status="fitted", fit=fitted, warnings=warnings)
        strata.append(result)
        fitted_groups.append((result, draws))

    comparisons, noncomparisons = [], []
    for (left, left_draws), (right, right_draws) in combinations(fitted_groups, 2):
        differing = {name: {"left": left["controls"].get(name), "right": right["controls"].get(name)}
                     for name in sorted(set(left["controls"]) | set(right["controls"]))
                     if (name not in left["controls"] or name not in right["controls"]
                         or left["controls"][name] != right["controls"][name])}
        # A single loaded backend can change an explicitly permitted runtime
        # setting. Preserve its actual identity; never relabel the source logs.
        if left["backend_id"] == right["backend_id"] and not (set(differing) & set(vary_controls)):
            continue
        identity = {"left_backend": left["backend_id"], "right_backend": right["backend_id"],
                    "left_stratum_id": left["stratum_id"], "right_stratum_id": right["stratum_id"],
                    "differing_controls": differing}
        reasons = []
        if set(differing) - set(vary_controls):
            reasons.append("unmatched_controls")
        if left["cache_state"] != right["cache_state"]:
            reasons.append("unmatched_cache_state")
        low = max(left["actual_output_token_range"][0], right["actual_output_token_range"][0])
        high = min(left["actual_output_token_range"][1], right["actual_output_token_range"][1])
        if low >= high:
            reasons.append("no_overlapping_actual_output_token_range")
        if reasons:
            noncomparisons.append({**identity, "reasons": reasons})
            continue
        delta = {}
        for name in ("fixed_ms", "marginal_ms_per_token"):
            delta[name] = right["fit"][name] - left["fit"][name]
            delta[name + "_ci95"] = interval([r[name] - l[name] for l, r in zip(left_draws, right_draws)])
        predictions = []
        for tokens in sorted(set([low, (low + high) / 2, high])):
            prediction = {"actual_output_tokens": tokens}
            for label, result in (("left_ms", left), ("right_ms", right)):
                prediction[label] = result["fit"]["fixed_ms"] + tokens * result["fit"]["marginal_ms_per_token"]
            prediction["right_minus_left_ms"] = prediction["right_ms"] - prediction["left_ms"]
            prediction["right_minus_left_ms_ci95"] = interval([
                r["fixed_ms"] - l["fixed_ms"] + tokens * (r["marginal_ms_per_token"] - l["marginal_ms_per_token"])
                for l, r in zip(left_draws, right_draws)])
            predictions.append(prediction)
        comparisons.append({**identity, "cache_state": left["cache_state"],
                            "comparison_scope": "configuration change" if differing else "matched workload",
                            "overlapping_actual_output_token_range": [low, high],
                            "right_minus_left": delta, "predictions_within_observed_overlap": predictions})
    return {
        "schema_version": 1, "metric": "server native inference milliseconds",
        "model": "server_elapsed_ms = fixed_ms + marginal_ms_per_token * actual_completion_tokens",
        "timing_boundary": BOUNDARY,
        "excludes": ["browser capture/JPEG encoding", "server image byte decoding", "network transport", "request queue"],
        "interval_method": "95% percentile circular moving-block bootstrap, independently resampled per backend/stratum",
        "bootstrap_repeats": bootstrap_repeats, "bootstrap_seed": seed,
        "minimum_measured_requests": min_samples, "minimum_actual_output_token_span": min_token_span,
        "explicitly_varied_controls": sorted(vary_controls),
        "input_records": len(records), "excluded_records": dict(sorted(excluded.items())),
        "backend_record_counts": {backend: dict(sorted(counts.items()))
                                  for backend, counts in sorted(backend_counts.items())},
        "accepted_measured_requests": sum(len(rows) for rows in groups.values()),
        "fitted_strata": len(fitted_groups), "strata": strata,
        "comparisons": comparisons, "noncomparisons": noncomparisons,
        "limitations": [
            "No historical client latency is converted to server latency; missing server logs remain unavailable.",
            "The intercept extrapolates to zero output tokens and is not an isolated vision/prefill measurement.",
            "Marginal cost is an empirical slope over the observed lengths, not hardware-only GPU decode time.",
            "Keep logs in acquisition order. Block bootstrap captures short-range correlation, not arbitrary drift or thermal changes.",
            "Output caps may be varied to obtain lengths; actual backend token counts, not caps, are fitted.",
            "Timing does not establish answer quality; compare quality separately when changing visual tokens.",
        ],
    }


def read_logs(paths: list[Path]) -> tuple[list[dict], list[dict]]:
    records, sources = [], []
    for path in paths:
        raw = path.read_bytes()
        sources.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
        for line_number, line in enumerate(raw.splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except (ValueError, UnicodeError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: record is not an object")
            records.append(row)
    return records, sources


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="+", type=Path, help="Server request JSONL, in acquisition order")
    parser.add_argument("--output", type=Path, help="Exclusive JSON output; defaults to stdout")
    parser.add_argument("--min-samples", type=int, default=12)
    parser.add_argument("--min-token-span", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--vary-control", action="append", default=[], choices=sorted(VARYABLE_CONTROLS),
                        help="Explicitly permit this configuration difference; never silently pooled")
    args = parser.parse_args(argv)
    try:
        records, sources = read_logs(args.logs)
        report = compare_records(records, min_samples=args.min_samples, min_token_span=args.min_token_span,
                                 bootstrap_repeats=args.bootstrap, seed=args.seed, vary_controls=args.vary_control)
        report.update(created_utc=datetime.now(timezone.utc).isoformat(), sources=sources,
                      comparator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        output = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x") as stream:
                stream.write(output)
        else:
            sys.stdout.write(output)
        return 0 if report["fitted_strata"] else 2
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
