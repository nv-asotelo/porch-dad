# Human steering and measurement normalization across two hackathons

Prepared for the 2026-09-21 meeting. The short talk should take about seven minutes, with detailed evidence available in the appendix. This record distinguishes the first campaign's supplied write-up, this task's recorded user instructions and measurements, and the newly supplied screenshots. It does not claim either campaign was fully autonomous or that unmatched results identify the faster implementation.

## Motivation

Both campaigns ask whether Cosmos3-Edge can provide useful, local camera understanding within an 8 GB Orin Nano's shared-memory budget. The first campaign emphasizes a resident runtime, broad text-layer INT4 conversion, maximum clocks and Live VLM WebUI integration. This independent task begins with a USB-C JetPack installation and builds a working streaming application from public model/runtime components. The user initially prohibited reuse of personal and internal projects. Later instructions explicitly authorized publishing this task to `nv-asotelo/porch-dad`, then inspecting the named one-shot branch for this comparison. No one-shot implementation was reused to build the deployed task branch.

The useful combined result is a working edge demonstration plus a more explicit measurement contract. A marketing audience can see a local camera application and constrained-device headroom. A technical audience can inspect how precision scope, input detail, generated length, power and timer boundaries affect the claim.

## Recorded steering in this task

| Stage | Human instruction or observation | Result and limit |
| --- | --- | --- |
| Physical installation | User reported the FC REC–GND jumper state, confirmed readiness and later said the jumper was absent. User also helped with connectivity and offered an isolated auxiliary GPU. | Official JetPack QSPI/NVMe flashing completed through the USB-C recovery path, with a task-owned Mac USB SSH/NFS bridge. The offered auxiliary GPU did no task work. Physical actions remain user-reported where the receipts cannot independently establish them. |
| First working demo | The agent chose a lightweight camera configuration and finite optimization gates from the original broad request. | 640×480 ideal camera, 15 FPS ideal, 512-pixel transmitted longest side, JPEG 0.8, one-second minimum cadence, 64 output tokens and greedy decoding. These were agent assumptions, not Live VLM WebUI defaults. |
| Fairer UI baseline | User asked for the same sample prompt and maximum tokens as Live VLM WebUI, then asked about capture interval and image size and requested a comparison toggle with Live VLM as default. | Added the upstream prompt, 512 output cap, ideal 1280×720 capture, 30-frame cadence and temperature 0.7 preset. Browser/WebRTC scheduling and transport still differed. Changing a preset did not make the backend or input-token cap identical. |
| Usability and access | User requested direct Orin hosting, HTTPS camera access, resource usage, all prompt presets, manual inference, live toggle, captions above/below/alongside the camera, sparklines and compact placement. | The app now serves on the Orin, defaults to captions below the camera and displays CPU/GPU/shared RAM. These improve the demonstration but are not model speedups. |
| Preserve the baseline | User requested a commit before further goal pursuit and publication on the same INT4 MLP branch. | Frozen checkpoint remains `9ed3ffd9263ae887ee6a7aded77368da220525a1`. Subsequent evidence and UI work remain separately versioned. |
| FP16 quality and stopping policy | The original search protected FP16 reference behavior. The user later required stopping when a next step failed to improve latency by 10%. | Broad RTN quantization failed the diagnostic quality gate. A later FP16 sampler experiment improved whole-answer latency only 0.48%, so the search stopped and the original runtime was restored. |
| MLP selection and policy correction | The user subsequently chose MLP-only INT4 as a new baseline, accepting its known color limitation. The same 10% continuation threshold applied. | Compact MLP improved whole-answer latency 1.19% and saved 163.45 MiB sampled peak shared RAM. It failed the original frozen acceptance rule. After measurement, the agent reinterpreted 10% as the threshold for continuing the search, rather than for retaining a useful smaller candidate. Search stopped and that candidate remained deployed. `results/mlp-goal/policy-clarification.json` preserves the timing of this correction and the unchanged failed comparison. No global optimum or general accuracy rate is claimed. |
| Advanced runtime controls | User requested image-token choices 512, 320 and Custom, with 320 on load, static clocks, a 256 MiB encoder cache and top-p 0.95. | Implemented bounded per-request native image budgets and clear separation from the output-token ceiling. Runtime controls and cache identity were validated. These were an intermediate policy, not the final defaults. |
| Timer correction | User clarified that JPEG decoding and accumulated network delay should not be included in the engine latency comparison. | Native inference and server TTFT now start after JPEG decode/preparation/admission. Browser first-text, round trip and frame age remain separate diagnostics. The server records request IDs and timings for audit. |
| Perceived slowdown | User reported slower behavior and clarified “Very first demo: restore its original lightweight defaults.” | Stopped the active token-length experiment before its planned end and preserved partial logs. Restored the first camera/prompt/output/sampling settings, image cap 512, cache zero and dynamic clocks. Concurrent benchmarking was a possible source of contention, not a proven complete explanation. |
| Engine reversal | User requested the original engine with Saturday's demo parameters, then requested returning to INT4 MLP and checking approximately 250 ms TTFT. | Temporarily loaded the preserved original FP16 engine. The supplied screenshot calls it BF16, but the deployment artifacts identify FP16. The user's observations about which configuration felt faster remain observations, not a controlled engine ranking. |
| Measure before switching | User added “before you do measure now with BF16 up and compare once you switch engines.” | Measured the currently loaded FP16 first, then MLP INT4, using identical input and controls, five warmups and 30 measurements each. Server TTFT medians were 249.50 and 239.32 ms. The final engine is MLP INT4 with the first demo's lightweight defaults. |

The first campaign's HTML reports its own interventions: replacing process-per-request serving, correcting a top-k bug, detecting cache and contention traps, rejecting unexecuted checkpoints, validating INT4 storage/weights, changing clocks and reducing visual detail. It does not contain the original conversation, labor accounting or a measured count of user interventions. Do not describe its frozen runbook as proof of a zero-intervention original build.

## Supplied screenshot observations

The exact unaltered images and SHA-256 hashes are in `output/two-hackathons-evidence/manifest.json`. Filenames describe the user's labels. The advanced controls are outside the visible crop, so these images do not establish the image-token budget, encoder-cache size or clock policy by themselves. All three visible capture labels already show Lightweight, a 640×480 camera at 15 FPS and transmitted 512×384 frames.

| User label | Visible native measurement | Visible browser diagnostics | Visible shared RAM | Interpretation |
| --- | --- | --- | --- | --- |
| INT4 MLP engine Astra Settings | Latency 686 ms, Average 708 ms, 8 timed requests | Collapsed | 4.92 / 7.37 GiB | A user-observed session, not a measured 320-token experiment merely because its filename says Astra settings. |
| BF16 original first version Astra | Latency 1,436 ms, Average 1,307 ms, 53 timed requests | Collapsed | 6.39 / 7.37 GiB | Original engine labeled BF16 by the user, identified as FP16 by deployment receipts. Caption is in flight; displayed native metrics are from the preceding completed answer. |
| INT4 MLP with original demo settings astra | Server TTFT 197 ms, Latency 632 ms, Average 785 ms, 13 timed requests | First text 312 ms, round trip 746 ms, frame age 818 ms | 4.94 / 7.37 GiB | Clear illustration that server time and browser experience are different measurements. It is one displayed observation with a different input from the controlled test. |

The 115 ms gap between the last screenshot's server TTFT and browser first text combines excluded work and delivery/scheduling boundaries. It is not a measured Wi-Fi-only penalty. Session averages have different counts and changing images/answer lengths. Memory snapshots include the OS and applications and overlap process RSS. None of these screenshots alone proves a cross-engine throughput or dedicated-VRAM percentage improvement.

## What is normalized now

Within the new FP16-versus-MLP test: exact JPEG bytes, prompt, temperature zero, top-p one, output cap 64, image cap 512, zero encoder cache, disabled text reuse, serial requests, stock 25 W/dynamic clocks, server timing boundary and warmup policy match. The actual input is 256 visual tokens. Both use current instrumentation and retained engine artifacts. One additional logged request started during the MLP run; the report discloses it and retains the raw row. The sequential order follows the user's request and is not a randomized crossover experiment.

Across the two hackathons: broad versus selective quantization, plugin path, power/clock policy, image workload, cache policy and complete-answer versus first-text timing still differ. The historical one-shot performance is reported evidence, not a newly normalized competitor run. A fair cross-branch ranking would require a shared input/output-quality set, explicit cache policy, identical power/input settings and a common timing contract. The current single-answer-length TTFT experiment cannot identify both fixed milliseconds and marginal milliseconds per output token. The earlier mixed-budget token-length experiment was interrupted and no completed fit is claimed.

## Sources

- Task installation and human handoffs: `research/first-version-journey.md` and its linked event receipts.
- Initial search and quality gates: `GOAL.md`, `results/optimization-selection.json`, `THIRD_PARTY_NOTICES.md`.
- Later finite searches: `research/latency10-follow-up.md`, `research/mlp-goal.md` and unchanged raw receipts.
- Runtime controls, interrupted sweep and matched TTFT: `research/runtime-controls-server-timing.md`, `results/runtime-controls/ttft-comparison.json`.
- Actual serving identity: `deployment/selected-config.json`, `results/runtime-controls/deployment.json`.
- First-campaign write-up review: `research/oneshot-writeup-review.md`.
- Authorized frozen-branch comparison: `research/two-hackathon-branch-comparison.md`.
