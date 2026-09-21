#!/usr/bin/env python3
"""Evaluate one MLP-only RTN latency step without changing historical policy.

The baseline is the last accepted MLP configuration. Reuse the original raw
JSONL validation; allow only runtime changes named in the new frozen plan.
Exit 0 accepts, 1 stops at the incumbent, and 2 means invalid evidence.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

import summarize_latency10 as core


MLP_WEIGHT_SHA256 = "93ac92d48d893aa45009fb327846bdd1052d03b22150ab8b64698d00a23d1854"
MUTABLE_FIELDS = frozenset({"max_kv_cache_capacity", "max_image_tokens", "native_extension_sha256",
                            "native_extension_current_sha256", "int4_gemm_plugin_version"})
EXTRA_FIELDS = ("max_image_tokens", "max_image_tokens_per_image", "quantization_scope",
                "quantized_linear_count", "weight_sha256", "int4_gemm_plugin_version",
                "native_extension_sha256", "native_extension_current_sha256")


def validate_plan(plan: dict, manifest: dict) -> set[str]:
    core.check(plan["warmups_per_fixture"] == 5 and plan["measured_requests_per_fixture"] == 30
               and plan["prompt"] == "Describe what you see in this image in one sentence."
               and plan["max_tokens"] == 512 and plan["temperature"] == .7
               and plan["top_p_request"] == plan["top_k_request"] == "omitted"
               and plan["backend_resolved_top_p"] == .9 and plan["backend_resolved_top_k"] == 50,
               "Plan differs from the fixed MLP workload")
    core.check(manifest["width"] == 1280 and manifest["height"] == 720
               and len(manifest["fixtures"]) == 3
               and plan["fixture_order"] == [fixture["file"] for fixture in manifest["fixtures"]],
               "Manifest dimensions/count/order mismatch")
    fields = plan.get("mutable_backend_fields", [])
    core.check(isinstance(fields, list) and all(isinstance(field, str) for field in fields)
               and len(set(fields)) == len(fields) and set(fields) <= MUTABLE_FIELDS,
               "mutable_backend_fields contains an unsupported or duplicate field")
    return set(fields)


def read_mlp_group(directory: Path, prefix: str, manifest: dict, plan: dict) -> dict:
    group = core.read_group(directory, prefix, manifest, plan)
    first = group["per_fixture"][0]["backend_config"]
    for run in group["per_fixture"]:
        backend = run["backend_config"]
        core.check(backend.get("weight_sha256") == MLP_WEIGHT_SHA256
                   and backend.get("quantization_scope") == "mlp-only"
                   and type(backend.get("quantized_linear_count")) is int
                   and backend["quantized_linear_count"] == 56,
                   f"{run['source']}: not the pinned MLP-only derivative")
        for field, required in (("max_input_len", 1024), ("max_image_tokens_per_image", 512), ("max_batch_size", 1)):
            core.check(type(backend.get(field)) is int and backend[field] == required,
                       f"{run['source']}: {field} must remain {required}")
        # Keep the full UI envelope: 1024 input + 512 output + one reserved slot.
        for field, minimum in (("max_kv_cache_capacity", 1537), ("max_image_tokens", 512)):
            core.check(type(backend.get(field)) is int and backend[field] >= minimum,
                       f"{run['source']}: invalid {field}")
        core.check(type(backend.get("int4_gemm_plugin_version")) is int
                   and backend["int4_gemm_plugin_version"] in (1, 2), f"{run['source']}: invalid INT4 plugin version")
        binary = backend.get("native_extension_current_sha256", backend.get("native_extension_sha256"))
        core.check(isinstance(binary, str) and re.fullmatch(r"[a-f0-9]{64}", binary),
                   f"{run['source']}: missing native binary SHA256")
        if "native_extension_sha256" in backend and "native_extension_current_sha256" in backend:
            core.check(backend["native_extension_sha256"] == backend["native_extension_current_sha256"],
                       f"{run['source']}: stale native binary provenance")
        core.check(all(backend.get(field) == first.get(field) for field in EXTRA_FIELDS),
                   f"{prefix}: MLP/profile/binary provenance changes between fixtures")
    return group


def equivalence(before: dict, after: dict, mutable: set[str]) -> tuple[list[dict], list[str]]:
    failures, comparisons = [], []
    fields = (*core.INVARIANTS, *EXTRA_FIELDS)
    for left, right in zip(before["per_fixture"], after["per_fixture"], strict=True):
        core.check(left["fixture"] == right["fixture"], "Fixture ordering mismatch")
        name = left["fixture"]
        if left["workload_fingerprint"] != right["workload_fingerprint"]:
            failures.append(name + ": workload_fingerprint_mismatch")
        changes = {field: {"before": left["backend_config"].get(field), "after": right["backend_config"].get(field)}
                   for field in fields if left["backend_config"].get(field) != right["backend_config"].get(field)}
        forbidden = sorted(set(changes) - mutable)
        if forbidden:
            failures.append(name + ": invariant_backend_fields_changed " + ",".join(forbidden))
        if left["endpoint"] != right["endpoint"] or left["sampling_host"] != right["sampling_host"]:
            failures.append(name + ": endpoint_or_sampling_host_mismatch")
        same = (left["unique_measured_answers"] == right["unique_measured_answers"] == 1
                and left["answer"] == right["answer"] and left["completion_tokens"] == right["completion_tokens"]
                and left["finish_reason"] == right["finish_reason"] == "stop")
        if not same:
            failures.append(name + ": mlp_output_or_token_count_not_equivalent")
        comparisons.append({"fixture": name, "output_equivalent": same, "backend_changes": changes,
                            "ratios": {field: right[field] / left[field]
                                       for field in ("p50_latency_ms", "p95_latency_ms")}})
    return comparisons, failures


def memory_delta(before: dict, after: dict) -> dict:
    result = {}
    for label, field in (("shared_ram_unavailable", "peak_shared_ram_unavailable_bytes"),
                         ("process_rss", "peak_process_rss_bytes")):
        start, end = before[field], after[field]
        result[label] = {"before_peak_bytes": start, "after_peak_bytes": end,
                         "decrease_bytes": start - end,
                         "decrease_percent": 100 * (start - end) / start if start else None}
    return result


def compare(baseline: dict, candidate: dict, mutable: set[str], initial: dict | None = None) -> dict:
    comparisons, failures = equivalence(baseline, candidate, mutable)
    failures += [f"baseline: {failure}" for failure in baseline["failures"]]
    failures += [f"candidate: {failure}" for failure in candidate["failures"]]
    for comparison in comparisons:
        if any(ratio > 1.05 + 1e-12 for ratio in comparison["ratios"].values()):
            failures.append(comparison["fixture"] + ": per_fixture_latency_regression_exceeds_5_percent")
    ratio = candidate["geomean_p50_complete_latency_ms"] / baseline["geomean_p50_complete_latency_ms"]
    if ratio > .9 + 1e-12:
        failures.append("geomean_complete_latency_gain_below_10_percent")
    memory_ratio = candidate["peak_shared_ram_unavailable_bytes"] / baseline["peak_shared_ram_unavailable_bytes"]
    if memory_ratio > 1.01 + 1e-12:
        failures.append("peak_shared_ram_regression_exceeds_1_percent")
    if initial:
        _, initial_failures = equivalence(initial, baseline, mutable)
        core.check(not initial["failures"] and not initial_failures,
                   "Initial MLP evidence is unsafe or not equivalent to the incumbent")
    accepted = not failures
    selected = candidate if accepted else baseline
    memory = {"candidate_vs_incumbent": memory_delta(baseline, candidate),
              "selected_vs_incumbent": memory_delta(baseline, selected)}
    if initial:
        memory.update(candidate_vs_initial_mlp=memory_delta(initial, candidate),
                      selected_vs_initial_mlp=memory_delta(initial, selected))
    return {"accepted": accepted,
            "decision": ("invalid_baseline_do_not_accept" if baseline["failures"] else
                         "accept_candidate" if accepted else "stop_and_keep_last_accepted_mlp"),
            "selected_candidate_id": selected["candidate_id"] if not baseline["failures"] else None,
            "candidate_to_baseline_ratio": ratio, "latency_reduction_percent": 100 * (1 - ratio),
            "peak_shared_ram_candidate_to_baseline_ratio": memory_ratio,
            "per_fixture": comparisons, "rejection_reasons": sorted(set(failures)), "memory_changes": memory,
            "memory_change_definition": "Decrease in sampled peak memory; positive means lower, negative means higher. Shared RAM and RSS overlap and are never added.",
            "memory_history_scope": "Only the supplied incumbent, candidate and optional initial MLP runs; no claim about an unprovided search history."}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True, help="Last accepted MLP run directory")
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--baseline-prefix", default="baseline")
    parser.add_argument("--candidate-prefix", default="candidate")
    parser.add_argument("--initial-dir", type=Path, help="Optional initial MLP baseline for cumulative memory deltas")
    parser.add_argument("--initial-prefix", default="baseline")
    parser.add_argument("--plan", type=Path, default=core.PROJECT / "results/mlp-goal/plan.json")
    parser.add_argument("--manifest", type=Path, default=core.PROJECT / "benchmarks/live-vlm-1280/manifest.json")
    parser.add_argument("--output", type=Path, help="New JSON receipt; never overwritten")
    args = parser.parse_args(argv)
    try:
        plan_raw, manifest_raw = args.plan.read_bytes(), args.manifest.read_bytes()
        plan, manifest = json.loads(plan_raw), json.loads(manifest_raw)
        mutable = validate_plan(plan, manifest)
        baseline = read_mlp_group(args.baseline_dir, args.baseline_prefix, manifest, plan)
        candidate = read_mlp_group(args.candidate_dir, args.candidate_prefix, manifest, plan)
        initial = read_mlp_group(args.initial_dir, args.initial_prefix, manifest, plan) if args.initial_dir else None
        if initial is not None:
            core.check(initial["candidate_id"] == plan["baseline_candidate"], "Initial MLP candidate ID differs from frozen plan")
        if initial is None and baseline["candidate_id"] == plan.get("baseline_candidate"):
            initial = baseline
        result = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                  "plan": {"path": str(args.plan), "sha256": core.sha256(plan_raw)},
                  "manifest": {"path": str(args.manifest), "sha256": core.sha256(manifest_raw)},
                  "comparator_sha256": core.sha256(Path(__file__).read_bytes()),
                  "raw_validator_sha256": core.sha256(Path(core.__file__).read_bytes()),
                  "primary_metric": plan["primary_metric"], "mutable_backend_fields": sorted(mutable),
                  "baseline": baseline, "candidate": candidate, "initial_mlp": initial,
                  "comparison": compare(baseline, candidate, mutable, initial),
                  "baseline_policy": "MLP-only RTN is the user-selected starting point. Its known yellow-to-orange error remains an accepted limitation; historical FP16 quality gates do not select or reject candidates here.",
                  "quality_scope": "Preserve every measured MLP answer and actual completion-token count. Equal output is not proof of correctness or general accuracy.",
                  "memory_scope": "Sampled extrema across warmup and measured requests; at least 512 MiB available shared RAM, no observed swap/OOM. Shared RAM includes OS/apps; RSS overlaps it.",
                  "queue_scope": "Serial harness concurrency verified; absence of other clients requires separate readiness/isolation evidence.",
                  "limits": plan.get("limits", [])}
        rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x") as handle:
                handle.write(rendered)
        print(rendered, end="")
        return 0 if result["comparison"]["accepted"] else (2 if baseline["failures"] else 1)
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as exc:
        print(json.dumps({"accepted": False, "decision": "invalid_evidence_do_not_accept", "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
