# Cosmos3-Edge live vision on Orin Nano

An original small browser interface for NVIDIA's Cosmos3-Edge reasoner and TensorRT-Edge-LLM v0.10.1, inspired by NVIDIA-AI-IOT's live-vlm-webui. Browser camera snapshots go through a local proxy to the resident TensorRT server, and generated text streams back over SSE. The application keeps one active request and no growing conversation history.

**Deployment status: JetPack is installed via USB-C. The final selected deployment returns to MLP-only RTN INT4 with the original lightweight demo parameters, at the user's request after a temporary FP16 restoration.** The preserved compact MLP engine, later UI controls and server timing remain available. The original FP16 model/cache and its historical evidence are preserved separately. The subsequent matched fixture test measured server TTFT at 249.50 ms for FP16 and 239.32 ms for MLP INT4. Those measurements and their limits are recorded separately below. Backend and UI are managed by systemd services. USB-C flash, normal NVMe boot, and a CUDA GPU calculation passed. The Orin Nano 8 GB developer kit (P3767-0005) runs Ubuntu 24.04.4 / Jetson Linux 39.2.1 with CUDA compiler 13.2.86 and TensorRT 10.16.2.10. Its WD_BLACK SN7100 1 TB NVMe root partition and ext4 filesystem were expanded successfully. Source/checkpoint transfer and Python dependency checks passed. The pinned reasoner snapshot contains 21 digest-verified files totaling 7,735,525,563 bytes.

The original native build completed both required targets and all 25 selected SM87 FMHA variants in 45m44s. Its [build receipt](results/native-backend-build.json) describes compilation, separately from inference. Open the UI directly on the Orin at **[https://192.168.6.252:8443](https://192.168.6.252:8443)**. The [HTTP address](http://192.168.6.252:8090) redirects to HTTPS for browser camera access. HTTPS uses a device-local certificate; accept it and grant camera permission on first use. The selected `data/engine-cache-mlp-compact` engine quantizes 56 MLP weights to RTN INT4 and keeps attention, vision, embeddings, activations and KV in FP16. It uses dynamic clocks within stock 25 W, input capacity 1,024, KV capacity 1,664, aggregate/per-image visual capacities 512/512, batch one and zero encoder-cache budget. The latest native extension supplies the retained controls and timing instrumentation. Swap remains disabled, with its original file and configuration backup preserved.

The preserved [MLP experiment](research/mlp-goal.md) measured **915.676→904.818 ms** geometric-mean complete-answer p50 across three fixed 1280×720 inputs, five warmups and 30 measured requests per image. All 90 measured answers and token counts matched. Sampled peak shared RAM fell **163.45 MiB (3.12%)**, to **4.958 GiB**; process RSS separately fell **220.33 MiB**, to **4.433 GiB**. These counters overlap. The observed **1.19%** latency change may include noise and did not meet the 10% continuation threshold, so the search stopped. A documented [policy clarification](results/mlp-goal/policy-clarification.json) selected the smaller candidate for that phase while preserving the original failed comparison. Its six-image diagnostic remains **18/19**, with the same yellow-to-orange color error. See [historical selection evidence](results/mlp-goal/selection.json); this is not a 10% latency success or a new TTFT comparison.

Earlier **FP16** service verification, after five warmups and 30 requests on the fixed 512×512 image workload, measured **265.5 ms p50 / 267.4 ms p95 time to first text**, **1.826 s p50 / 1.830 s p95 complete-answer latency**, and **6.435 GiB sampled peak system RAM unavailable**. Process RSS peaked at 5.907 GiB; it overlaps system RAM and must not be added to it. These are historical shared-memory observations, not dedicated VRAM or a CUDA allocator measurement. All 30 requests succeeded without OOM or swapping. See [raw FP16 verification](results/raw/benchmark-final-service-01.jsonl); this different workload is not the new MLP comparison denominator.

The new [matched server-TTFT comparison](results/runtime-controls/ttft-comparison.json) measured the original FP16 first and then selected INT4 MLP, with five warmups and 30 measured requests per engine. The fixed 512×512 JPEG, prompt, image cap 512, output cap 64, greedy decoding, zero encoder cache and stock 25 W/dynamic clocks match. Native first-text medians/p95 were **249.50/250.57 ms FP16** and **239.32/240.72 ms MLP**. The observed median difference is **10.18 ms (4.08%) on this fixture**. These server measurements exclude JPEG decode, admission and network, and include server text-consumer scheduling. One additional logged request started during the MLP window. Actual visual work was 256 image tokens. FP16/MLP generated 37/31 output tokens, so their complete-answer durations do not establish equal-token decode speed. The historical 265.5 ms first-text figure above uses the broader loopback HTTP boundary. No cross-hackathon speed ratio follows from these numbers.

The earlier [four-trial search](results/optimization-selection.json) stopped after three consecutive non-improvements. All-linear RTN INT4 reduced memory but failed image quality (13/19 facts). MLP-only RTN also introduced a new color error despite scoring 18/19. Fixed clocks improved median answer latency by only 3.94%, below the frozen 5% threshold; global swap counters also changed. Corrected FP16 remained selected for that historical phase, before the user explicitly chose MLP-only RTN as the new baseline. Neither bounded search establishes a global optimum.

Earlier FP16 image-quality checks also remain limited: [its synthetic screen](results/quality-final-service-01-review.json) scores 18/19, with the circle-to-sphere error exactly reproduced by the official model reference. The strict 19/19 screen did not pass. A separate [three-photo FP16 smoke check](results/natural-smoke-final-01-review.json) earned only 5/9 generously credited observations and no fully correct case. Objects were omitted and spatial descriptions were wrong. The official model reference also scored 5/9, with two of three answers identical; the third still missed the required spatial relation. These historical results do not grade the new MLP configuration. The three images are illustrative, not an accuracy benchmark. [An NVMe timeout recurred during cold loading](research/nvme-timeout-investigation.md); the service recovered, but no storage fix is claimed.

The earlier [600.7-second FP16 streaming soak](results/raw/soak-final-service-01.summary.json) passed all 366 requests with no errors, OOM, swapping or queued backlog. Minimum available RAM was 965 MiB; process RSS grew by 156 KiB between the specified windows. Its [upload/Stop/restart](results/browser-live-ui-final.json) and [synthetic moving-camera](results/browser-camera-ui-final.json) browser checks passed. The dated [service audit](results/final-service-audit.json) records both units enabled and active, with zero restarts. The new MLP [service/browser verification](results/mlp-goal/browser-verification.json) and [600.5-second soak](results/mlp-goal/soak-final.summary.json) passed separately, including 569 requests without errors, OOM, swapping or queued backlog. A physical reboot after service installation was not tested. The direct HTTPS URL above does not require the Mac tunnel.

The restored first-demo defaults use the prompt **“Describe the visible scene in one concise sentence. Focus on objects and actions.”** and an **output token limit of 64**. The user first restored these parameters, then explicitly tried the original FP16 engine, and finally requested a return to MLP INT4 with a matched TTFT check. The final capture/generation defaults stay lightweight through that last engine change. Advanced controls and native server timing remain. The public [Live VLM WebUI reference](https://github.com/NVIDIA-AI-IOT/live-vlm-webui/blob/2fd5ba0b334c334d24bf0f9439d8742b243d22be/src/live_vlm_webui/vlm_service.py#L42) uses “Describe what you see in this image in one sentence.” with a 512-token output ceiling. That prompt remains in Quick presets and the output ceiling remains editable.

The **Live VLM WebUI / Lightweight** capture toggle now defaults to **Lightweight** on every fresh page load: an ideal 640×480/15 FPS camera request, a 512-pixel longest side, a one-second minimum interval, JPEG quality 0.8 and temperature 0. Live VLM remains available and requests ideal 1280×720 video with no FPS constraint, samples every 30 video frames, preserves captured dimensions, skips eligible frames while busy, and uses temperature 0.7. Both modes keep your prompt/token limit when switching, stop active work and clear old timings. Actual camera resolution/FPS and sent-frame dimensions appear in the UI.

For a Live VLM settings comparison, select that capture preset, its Scene Description prompt and a 512-token output limit, then hold model/input/sampling controls equal. Browser versus WebRTC frame counting, JPEG encoding and token streaming still differ. This does not establish an identical transport benchmark. See [the comparison contract](research/capture-preset-comparison.md). Historical performance and presentation figures retain their original workloads.

The UI includes live CPU, GPU and shared-memory usage from the Orin, refreshed once per second. CPU is aggregate utilization across all cores; GPU is the driver-reported load. **VRAM / shared RAM** shows `MemTotal - MemAvailable` and total Linux-visible RAM, including CPU, GPU and OS use. It is not a per-model CUDA allocation. Missing or stale telemetry is shown as unavailable. The cached `/api/metrics` endpoint samples independently of inference and needs no extra package or root access. See [telemetry definitions](research/ui-feasibility.md#live-device-telemetry).

The **Quick presets** menu includes all ten prompt labels and texts from Live VLM WebUI, including both robot-navigation examples. The prompt remains editable and applies to the next request. **Caption position** offers Side window, Above camera and Below camera, with Below camera as the first-visit default and your selection saved in this browser. Above and Below use a compact black closed-caption band attached to the video. The last caption stays visible until new text arrives. On narrow screens, Side window stacks below the camera.

The resource cards show colored **sparklines for the last 60 seconds**, with per-chart scales and gaps for missing readings. In closed-caption modes they fill the right column beneath output info. **TTFT** measures native inference start to the first nonempty server text. **Latency** measures the complete native inference on the Orin. Both exclude JPEG encoding/decoding, queueing and network transport and update after a successful answer. **Average** is the cumulative mean of successful server-timed requests in the current settings group. Browser first-text, round-trip and frame-age diagnostics remain separate. Historical browser-timed results are unchanged.

**Advanced model settings** separates **input image tokens** (512 default, 320, or Custom) from the **output token limit** (64 on page load, adjustable up to 512). The image cap and top-p (1.0) apply to the next request; a new engine load restores these advanced defaults. The selected service uses dynamic clocks within stock 25 W and zero encoder-cache budget. It retains its built 512-image-token capacity for instant switching; a smaller request cap does not promise proportional resident-memory savings. The earlier 320-token / top-p 0.95 / 256 MiB cache / static-clock acceptance remains historical evidence. Its subsequent comparison run was interrupted by the user's restoration request, so no completed fixed/marginal latency comparison is claimed. See [runtime controls and server-side measurement](research/runtime-controls-server-timing.md).

**Run inference** captures one camera frame or analyzes the selected image. **Live streaming: On/Off** controls automatic camera requests while keeping the preview running. Turning it off cancels an automatic request already in progress, preserves any partial caption, and leaves manual requests available. **Stop** stops the camera and cancels the active request. Layout changes keep the camera and current answer intact. These controls retain the selected model and the existing generation limits. See [the UI behavior and upstream source](research/capture-preset-comparison.md#prompt-menu-caption-placement-and-manual-inference).

## Findings presentation

[Meeting PowerPoint](output/cosmos3-edge-two-hackathons-2026-09-21T18-45-11-750Z.pptx) · [PDF](output/cosmos3-edge-two-hackathons-2026-09-21T18-45-11-750Z.pdf) · [5–10 minute presenter guide](output/two-hackathons-presenter-guide-2026-09-21.md). Use the eight main slides for the talk and six appendix slides for technical questions. The presentation combines both hackathons, their public contributions, manual steering, configuration changes and the matched FP16/MLP TTFT measurements. All 14 slides were visually reviewed in both rendered PowerPoint and PDF output; [validation and artifact hashes](results/two-hackathons-presentation.json) record the checks and their limits.

**Frozen meeting version:** see [the freeze record](deployment/FROZEN.md). The user-requested cleanup leaves the same engine and UI running, with no development workload active. A separate 30-second loaded-engine idle sample observed median CPU **0.5%**, GPU **0%**, and shared RAM **4.93 GiB**. [The cleanup receipt](results/development-cleanup/receipt.json) records stopped optional services, unchanged engine identity and restoration commands. These are idle measurements, not inference-load performance.

The [two-hackathon source comparison](research/two-hackathon-branch-comparison.md), [supplied write-up review](research/oneshot-writeup-review.md) and [manual steering record](research/two-hackathon-steering.md) distinguish reported historical performance, live screenshots and current matched measurements. The named one-shot branch quantizes a broader text scope and uses different clocks, caching and timing, so the review does not claim a normalized cross-branch speed ranking.

[Download the earlier detailed 37-slide presentation](output/cosmos3-edge-findings-with-after-and-mlp-goal-2026-09-21T04-28-52-127Z.pptx). It covers the full iteration and steering history, the supplied after-demo statistics and frames, and the final MLP selection. The new matched comparison reduced sampled peak shared RAM by **163.45 MiB** and observed aggregate p50 by **1.19%**. Search stopped below the 10% continuation threshold. The deck explains the agent's after-measurement interpretation correction, preserved quality limitations, public contributions and the **569-request, 600.5-second** stability run. The current model runs directly at [the Orin HTTPS UI](https://192.168.6.252:8443).

[Download the 28-slide checkpoint presentation](output/cosmos3-edge-findings-with-latency-goal-2026-09-21T01-13-52-175Z.pptx). It predates the new [MLP selection report](research/mlp-goal.md). It includes the initial UI assumptions, your steering toward Live VLM WebUI defaults, original “before” frames, the historical four-trial search, and the subsequent FP16 10% latency goal. That FP16 candidate preserved every measured answer but reduced the complete-answer metric by only **0.48%** (1355.410→1348.917 ms), below the required 10%. That search stopped and the verified original FP16 runtime was restored. A **22.30% synthetic sampler improvement** is reported separately and does not imply a comparable inference gain. Installation, original integration, public-source credits, shared-memory constraints, quality limits and human interventions remain covered. The [full intervention record](research/first-version-journey.md) distinguishes user reports from observed actions and omits credentials; the [FP16 follow-up experiment](research/latency10-follow-up.md) links its raw evidence.

The [earlier 18-slide revision](output/cosmos3-edge-findings-with-journey-2026-09-20T18-31-44-270Z.pptx) remains available as a dated record. The new deck preserves its first 17 slides and updates the stale capture-default status.

The [original 14-slide presentation](output/cosmos3-edge-findings-2026-09-20T07-33-43-203Z.pptx) remains available, with its [validation and visual-review receipt](results/final-presentation.json). Its slides and measurements are preserved in the updated deck.

The later user recording now has [six presentation-ready “before” frames](output/before-cosmos3-edge-2026-09-20/README.md), a [visual gallery](output/before-cosmos3-edge-2026-09-20/index.html), and exact source timestamps/checksums. These preserve the user-designated unoptimized Live VLM capture baseline. They show completed answers and individual timings, but the numeric CPU/GPU/shared-memory readings and token-limit control are outside the recorded viewport. They are separate from the earlier benchmark. The updated presentation embeds the action-camera and pen frames with their individual observations and limitations.

## Reproduce the preparation

```bash
bash scripts/fetch_sources.sh
python3 scripts/host_probe.py  # macOS USB/media check
```

Pinned sources and licenses are in [sources.lock.json](sources.lock.json) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md); the [contribution ledger](research/contribution-ledger.md) maps upstream work, public people/team credits, and this task's original integration. Implementation source preparation used only the permitted public repositories, without personal-namespace implementation reuse or internal repositories/communications. The user later explicitly authorized publication to `nv-asotelo/porch-dad` and read-only comparison with its named one-shot branch. That comparison did not change the deployed implementation. Newly authored code in this workspace is under Apache-2.0; upstream code, pretrained weights and platform software retain their separate terms.

## Install and launch on the Jetson

1. For a fresh device, follow the [JetPack installation route](docs/jetpack-install.md). This device has already completed the official Super QSPI/NVMe flash and normal boot. The recorded [Super flash choices](research/confirmed-board-flash-options.md) make target media explicit.
2. Run `bash scripts/device_inventory.sh` on the Jetson and save the result. Verify the actual hardware and installed CUDA/TensorRT stack.
3. Follow [backend build and checkpoint instructions](docs/backend-build.md). The verified `models/cosmos3-edge-reasoner` snapshot and pinned backend are already at `/home/jetson/cosmos-edge` on this Orin. Preserve the checkpoint's original files and notices. Native compilation and FP16 engine construction passed. The model server currently uses the target GPU; stop that owned service before rebuilding or loading another candidate.
4. The selected deployment is already installed as enabled systemd services. On this Orin:

```bash
cd /home/jetson/cosmos-edge
sudo systemctl start cosmos-edge-backend cosmos-edge-ui
systemctl is-active cosmos-edge-backend cosmos-edge-ui
```

The selected launcher validates and loads the existing engine cache. Rebuild explicitly using [backend instructions](docs/backend-build.md) when the model, TensorRT version or engine profile changes. Check `http://127.0.0.1:8090/health/ready` on the Jetson before inference. The UI shows backend unavailability while the model loads.

Direct LAN access is enabled by a persistent systemd drop-in. Visiting the LAN page on HTTP port 8090 redirects to HTTPS port 8443. Loopback HTTP and API/health endpoints remain available. Both listeners share one telemetry sampler and one inference admission lock. The backend remains on loopback port 8000. The certificate covers the current Orin IP and `jetson.local`, expires September 20, 2027, and is stored under `deployment/tls` on the Orin. No client trust store was changed. Camera use requires browser acceptance of the certificate and camera permission. To reproduce the listener configuration:

```bash
sudo bash scripts/enable_lan_ui.sh 192.168.6.252
sudo systemctl restart cosmos-edge-ui
```

The previous Mac tunnel remains an optional camera route without a LAN certificate warning:

```bash
ssh -N -L 8090:127.0.0.1:8090 YOUR_JETSON_USER@YOUR_JETSON_ADDRESS
```

Then open [http://localhost:8090](http://localhost:8090). The browser treats localhost as a secure camera context. Direct LAN hosting does not depend on this tunnel or the Mac. See [UI deployment notes](research/ui-feasibility.md) and [deployment evidence](results/ui-lan-deployment.json). This is a single-user LAN service without authentication.

## Measure and optimize

Use [the benchmark method](research/benchmark-method.md) with the actual service on the Jetson. Start with five warmups and at least 30 measured requests per candidate. Capture actual outputs, prompt/image hashes, engine/model revisions, system memory and thermals. Jetson shares RAM between CPU and GPU; system RAM, RSS and CUDA allocations are distinct overlapping observations.

The initial frozen plan allowed at most five measured candidates including baseline, within the original ceiling of 12. It stopped after four trials because three consecutive candidates failed to deliver an eligible 5% p50-latency or peak-memory improvement while preserving quality and avoiding greater than 5% regressions. The subsequent [MLP goal](research/mlp-goal.md) also stopped after its first candidate missed the 10% continuation threshold; its smaller candidate was selected under the disclosed deployment clarification. After the later temporary FP16 restoration, the user's final selection returns to that compact MLP engine with first-demo lightweight parameters. The requested TTFT check is a bounded comparison, not another optimization search. No further marginal-gain trial is scheduled. An INT4 weight-size estimate is not a measured peak-memory result.

## Local verification and preview

```bash
python3 -m unittest discover -s tests -v
node tests/test_sse.js
python3 scripts/serve_ui.py --port 8090
```

The last command can preview the UI on macOS with the backend unavailable; use another port if the Orin UI tunnel already occupies 8090. Test SSE fixtures exercise transport behavior only and never count as Cosmos3 inference or performance results. Historical FP16 and MLP measurements, browser checks and their separate soaks remain recorded. [The lightweight acceptance receipt](results/runtime-controls/validation-lightweight.json) describes the interim MLP restoration. [The temporary FP16 TTFT receipt](results/runtime-controls/ttft-fp16/receipt.json) records its separate matched timing run; a planned FP16 runtime-control acceptance did not run. Keep final-MLP acceptance and TTFT evidence separate instead of overwriting these records. [Completion evidence](GOAL.md) records the scope and remaining limitations.

## Project files

| Path | Purpose |
| --- | --- |
| `web/` | Original camera/upload interface and incremental response rendering |
| `scripts/serve_ui.py` | Dependency-free local streaming proxy, bounded input and one active inference |
| `scripts/run_backend.sh` | Guarded launcher for the pinned upstream backend on Orin |
| `scripts/benchmark.py` | Reproducible streaming measurements and finite optimization evaluator |
| `research/` | Cited public-source feasibility and measurement method |
| `docs/` | Installation and build procedure |
| `results/` | Observations and experiment evidence; no fabricated GPU data |
| `output/` | Findings presentation and its evidence-backed input data |
