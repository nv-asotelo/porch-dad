# Frozen meeting version, 21 September 2026

The user requested committing and freezing `codex/cosmos3-edge-mlp` in `nv-asotelo/porch-dad`. The final snapshot includes the serving implementation, measured request logs, comparison research, meeting presentation and operational cleanup receipt. Its annotated Git tag is `cosmos3-edge-mlp-freeze-2026-09-21`. The Git commit/tag identifies the exact snapshot; this document does not claim a GitHub administrator-enforced branch-protection rule.

No further changes should be pushed to this frozen branch without a later user instruction. The earlier `codex/cosmos3-edge-checkpoint` remains at `9ed3ffd9263ae887ee6a7aded77368da220525a1`.

## Serving identity

Runtime source was committed before the meeting artifacts as public commit `935ea4887e19038e93be2bcbe19a5ecbe4777ec5`, with project subtree `cosmos3-edge-orin/` exactly matching local commit `796be230dea9cde3531f73ebc4abc6638dc3aeb5`. Subsequent meeting-artifact additions do not change that runtime implementation.

The selected engine is MLP-only RTN INT4, with 56 quantized MLP projections and the remaining documented components in FP16. It uses the original lightweight demo controls: input image cap 512, output cap 64, temperature 0, top-p 1, zero encoder cache and dynamic clocks within stock 25 W. The UI remains at `https://192.168.6.252:8443/` directly on the Orin. Selected artifact hashes and process identity are recorded in `selected-config.json` and `../results/runtime-controls/deployment.json`.

The latest matched test measured server TTFT at 249.50 ms median for original FP16 and 239.32 ms for selected MLP, with 30 measured requests after five warmups each. This is a fixed-fixture result, not a normalized performance ranking against the separate one-shot branch. Timing, concurrency and output-length limits remain in the comparison receipts.

## Operational cleanup

At the user's request, no compiler, benchmark, profiler or duplicate model process remained running. The idle iperf3 server, fwupd service/refresh timer and kerneloops service were stopped for this measurement session. They were not uninstalled, disabled or masked; their previous enablement is preserved and restoration commands are recorded. Core OS, network, SSH, cooling and GPU support remain active.

A 30-second sample of the existing UI telemetry, after five seconds of settling, observed no active or queued inference requests. Median aggregate CPU was 0.5%, GPU was 0%, and shared system RAM unavailable was approximately 4.93 GiB of 7.37 GiB. This is loaded-engine idle state, with SSH and a lightweight sampler present, not inference-load performance or dedicated VRAM. The engine ID and loaded controls were unchanged. No page-cache drop, reboot, model restart or new inference benchmark was used.

Exact observations and service restoration commands: `../results/development-cleanup/receipt.json`, `samples.jsonl` and `restore-services.sh`. Measurements during an active camera stream should be labeled separately from this idle baseline.
