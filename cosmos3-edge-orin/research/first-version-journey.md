# First-version goal, user steering and human interventions

Prepared from the task conversation supplied to the coordinating agent and the cited public/task-owned receipts. This appendix supplements the first completed 14-slide presentation, `output/cosmos3-edge-findings-2026-09-20T07-33-43-203Z.pptx`. It preserves that version's measurements and separates later UI requests. It omits passwords, credentials and the auxiliary host address. No internal repositories, communications or personal-namespace projects are sources.

## Original goal and the agent's finite policy

The user requested JetPack installation through the connected Orin's USB-C port, working Cosmos3-Edge inference through public TensorRT-Edge-LLM, a streaming browser UI served on the Orin, measured latency/shared-memory optimization after signs of life, and a presentation with findings and attribution. The user prohibited personal-namespace and internal sources.

Exact safe original instruction:

> Give yourself a proper goal setting so you don't keep iterating on marginal gains for either.

The agent chose the numeric completion and search gates. The original ceiling was 12 candidates, later narrowed to five with the same 5% improvement criterion. Each measured candidate used five warmups and 30 measured requests. Acceptance required at least 5% lower p50 complete-answer latency or sampled peak memory, no greater than 5% regression in the counterpart or p95 latency, equivalent image quality, and no errors, OOM or workload swapping. Three consecutive non-improvements ended the search after four trials. The planned compact fifth trial did not run. These numbers are agent implementation choices, not a verbatim user request or a claim of global optimality.

Evidence: [goal and gates](../GOAL.md), [original recorded budget](../results/deployment-status.json), [tightened plan](../results/optimization-search-plan.json), [actual stop decision](../results/optimization-selection.json).

## Safe transcript excerpts and their role

| User wording | Classification and interpretation |
| --- | --- |
| “Did you install jetpack via the usb-c?” | Progress oversight. This asks for installation status, not a physical intervention. |
| “what have you done thus far?” | Progress oversight. |
| “Poll the USB and see if you see it now, it's jumpered FC REC and GND” | User-reported recovery jumper state and a request for a fresh observation. Device enumeration is a separate agent observation. |
| “done” | User readiness confirmation following a requested fresh recovery power cycle. It does not independently prove the detailed power/cable sequence or physical-action timing. |
| “the jumper isn’t in” | Explicit user-reported absence of the recovery jumper. The structured receipt preserves the same wording with a straight apostrophe. It does not establish who removed it or when. |
| “Isolate your work from anything else on that computer.” | Safe excerpt from the optional auxiliary GPU offer. The user limited remote work to one new task directory. Host and credentials are deliberately omitted. |
| “Do you need the GPU? I took the ethernet offline so you can go faster, can you work on wifi or do you want ethernet” | User-reported connectivity change and a question about the agent's needs. It is not an independently measured network change or demonstrated speed gain. |
| “nvm we'll take all night if you need to, giving the GPU ethernet back and you on wifi” | User-reported connectivity steering and willingness to allow more time. The records do not independently identify the resulting interface medium. |
| “ok was able to ssh in from the terminal inside this codex side chat” | User-reported successful interactive login. This does not establish successful authentication by the agent's SSH clients. |

The supplied auxiliary GPU never performed task work. The [access receipt](../results/aux-gpu-status.json) records a reachable SSH server but rejected agent authentication, including after credential reconfirmation. It records no remote task directory, no created files and no access to other directories. Actual Orin GPU inference and the separate local Mac MPS reference runs must not be confused with use of this auxiliary machine.

## Hardware handoffs and verified automation

Times are UTC recording timestamps. They do not establish the exact instant of a user's physical action.

| Recorded time | Human input / outstanding prerequisite | Agent action and verified outcome |
| --- | --- | --- |
| 2026-09-19 19:21 | User reported FC REC–GND recovery state. | [Host inventory](../results/host-inventory.json) observed APX `0955:7523`; no flash had occurred. |
| 2026-09-20 01:20–01:21 | [Status](../results/deployment-status.json) records physical-reset confirmation; exact detailed sequence is not preserved. | [Official EEPROM/bulk upload](../results/jetson-board.json) succeeded after the recorded cold recovery reset, identifying Orin Nano 8 GB. No persistent write yet. |
| 01:40 | [RAM-boot receipt](../results/ram-recovery-boot.json) requests Mac unlock/accessory approval if a prompt appears. Approval itself remains unconfirmed. | Official RAM recovery image uploaded and USB identity changed. Guest USB networking timed out. |
| 02:00–02:28 | No further manual action is established by these receipts. | [Native Mac USB SSH and NFS bridge](../results/usb-transfer-path.json) worked without changing Mac security policy. [Official QSPI/NVMe flash](../results/jetpack-flash-status.json) completed at 02:28:02, exit 0, using `--no-reboot`. |
| 03:14–03:38 | Confirmation that the recovery jumper was absent remained outstanding. | Agent continued source/model and package preparation. [Native ARM chroot provisioning](../results/chroot-compute-provisioning.json) and [cleanup](../results/chroot-cleanup.json) passed. The [blocker audit](../results/blocked-audit.json) distinguishes that work from a normal boot and GPU execution. |
| 04:33–04:43 | “the jumper isn’t in” resolved the confirmation gate at the receipt's 04:33:44 timestamp. | [Normal NVMe boot and task-key SSH](../results/normal-boot.json), [CUDA calculation](../results/cuda-preflight.json) and [storage expansion](../results/storage-expansion.json) then passed. |

The records do not establish a separate manual login/configuration sequence for the Orin. Agent access used a task-specific deployment key. Normal-boot discovery used mDNS and a LAN endpoint, which does not by itself identify Wi-Fi versus Ethernet. The physical reboot after final systemd service installation was **not tested**, as preserved in the [final audit](../results/final-service-audit.json).

## First-version boundary

The first completed version includes the USB-C install, corrected FP16 Orin inference, streaming browser upload/Stop/restart and synthetic-camera checks, four-trial bounded search, enabled/active services, a 600.701-second API soak with 366 successful requests, and the 14-slide findings presentation. It openly retains quality and storage limits. The geometric screen is 18/19 strict facts, the natural-photo diagnostic is 5/9 generous or 4/9 strict with 0/3 fully correct cases, and NVMe cold-start timeout warnings remain unresolved. The first fixed benchmark used a 512×512 image and **64 output tokens**. No later UI default change alters those saved results.

Evidence: [first presentation receipt](../results/final-presentation.json), [final service benchmark](../results/raw/benchmark-final-service-01.jsonl), [soak](../results/raw/soak-final-service-01.summary.json), [quality](../results/quality-fp16-corrected-01-review.json), [natural-photo review](../results/natural-smoke-final-01-review.json), [service/storage audit](../results/final-service-audit.json).

## Follow-ups after the first version

Exact safe requests:

> add the cpu, gpu, and vram usage to the UI.

> Set the max tokens and the sample prompt to the same default parameters as Web VLM UI.

> Add in any of the goal seek, steering prompts, and manual interventions we had to do to get here for the first version, in the presentation

> Serve the webpage directly on the orin please

The telemetry/default changes passed [deployed browser checks](../results/browser-ui-telemetry.json) and [verification](../results/ui-telemetry-validation.json). The UI now shows CPU and GPU utilization plus shared system RAM (`MemTotal - MemAvailable`), not a dedicated CUDA allocator total. The sample prompt is “Describe what you see in this image in one sentence.” and the UI output default is 512 tokens, matching the pinned upstream service defaults. These are later UI changes, separate from the unchanged 64-token benchmark.

The user then asked:

> Are the capture interval and the frame longest side the same defaults as Web VLM UI?

This is an informational question, not a request to change them. The task UI still uses a 1-second minimum capture interval and 512px longest side. Pinned upstream uses 30 captured frames between inference attempts and an ideal 1280×720 camera request. The scheduling semantics differ, and those defaults have not been matched. See [task capture UI](../web/index.html), [task scheduling](../web/app.js) and the pinned upstream [video processor](https://github.com/NVIDIA-AI-IOT/live-vlm-webui/blob/2fd5ba0b334c334d24bf0f9439d8742b243d22be/src/live_vlm_webui/video_processor.py), [30-frame input](https://github.com/NVIDIA-AI-IOT/live-vlm-webui/blob/2fd5ba0b334c334d24bf0f9439d8742b243d22be/src/live_vlm_webui/static/index.html#L2329) and [ideal camera dimensions](https://github.com/NVIDIA-AI-IOT/live-vlm-webui/blob/2fd5ba0b334c334d24bf0f9439d8742b243d22be/src/live_vlm_webui/static/index.html#L2659).

Direct Orin LAN access passed real image inference over [HTTP](http://192.168.6.252:8090) and [HTTPS](https://192.168.6.252:8443), recorded in the [deployment receipt](../results/ui-lan-deployment.json) and [browser checks](../results/browser-ui-lan.json). One persistent UI process serves both listeners and the backend PID remained unchanged. HTTPS uses a self-signed task-local certificate, so first-visit acceptance/trust is a user setup step. The browser QA bypassed certificate errors only in its isolated test context, confirmed a secure context, and exercised synthetic-camera start/inference/stop. It did not validate a physical camera or modify the user's trust store. This later deployment change does not retroactively change first-version access, which used an SSH tunnel to an Orin-hosted service.

## Audit cautions

Subsequent request after this presentation revision: “can we have a toggle so we have an apples to apples comparison to web vlm ui? Make the Live VLM WebUI default”. The later [capture preset implementation](capture-preset-comparison.md) now defaults to Live VLM camera/generation settings and keeps the prior settings as Lightweight. This supersedes the earlier capture-default status above; the deck remains a dated record of the preceding state.

- `deployment-status.json` preserves an intermediate pending state. The coordinating task reconciled the installation guide with completed receipts during this update. Event-specific receipts establish the later completions.
- `flash-host.json` retains an early timestamp despite later added success fields. `normal-boot.json` points to a CUDA receipt with a later timestamp. Do not infer exact durations from mutable summary records.
- “done” and reported network changes must not become invented cable actions, approval clicks or quantified speed improvements.
- The first-version benchmark and attribution remain unchanged. Later UI behavior requires its own evidence.

## User-recorded “before” evidence

The user supplied `Codex recording unoptimized Sep 20 Cosmos3-Edge.mov` and requested relevant frames for the unoptimized apples-to-apples baseline. [Six full-resolution frames and captions](../output/before-cosmos3-edge-2026-09-20/README.md) preserve the visible Live VLM default capture preset, actual 1280×720 / 30 FPS camera label, sent dimensions, prompt and completed-answer timings. The [manifest](../output/before-cosmos3-edge-2026-09-20/manifest.json) records the source hash and exact presentation timestamps/decoded frame indices. No source recording or evidence-frame pixels were retouched.

The selected frames include a model interpretation in the pen response that the image does not establish. These are curated visual examples, not an accuracy or latency benchmark. Hardware telemetry values and the max-token control are outside the viewport. “Unoptimized” preserves the user's baseline designation; the corrected FP16 integration and bounded-memory configuration were already deployed. The [24-slide presentation revision](../output/cosmos3-edge-findings-with-before-and-steering-2026-09-21T00-37-23-106Z.pptx) now embeds the action-camera and pen frames alongside the initial assumptions, steering, and bounded-search outcome. It preserves the earlier findings as historical measurements.


## Assumptions, steering and goal-seek results in the presentation

The user requested: “Note your initial assumptions at the before, the steering to make you compare fairly to Web VLM UI's default settings, and your results after the goal seek. Put that into the presentation.”

Slides 19–24 record the compact initial UI choices (64 tokens, 512-pixel longest side, one-second minimum cadence, greedy generation), the original overhead hypothesis, the requests to align upstream defaults, two unaltered recording frames, the four-trial search and final-service measurements. Slide 18 now reflects the Live VLM default toggle and HTTPS camera redirect. The earlier 17 slides retain their measured content.

The chronology is explicit: the bounded optimization search and final-service verification came before the later default alignment and user recording. No additional candidate search or paired optimization benchmark ran under the new Live VLM preset. The selected corrected FP16 remains unchanged. The 64-token geometric benchmark and the later stochastic camera observations cannot establish a before/after performance gain against each other.

## Subsequent 10% latency goal

The later active goal explicitly requested at least 10% latency improvement per step and stopping when the next run cannot provide that gain. After the user's Live VLM default correction, the agent established a new matched workload instead of reusing the earlier 512×512/64-token figures. The three-image recording-derived suite, geometric mean, five warmups plus 30 measurements, and first-confirmed-failure interpretation are agent-selected controls; they are recorded before measurement in the frozen plan.

A task-authored retained-partial top-K experiment passed 338 GPU equivalence cases and preserved all measured outputs. Controlled synthetic sampler time fell 22.30%, but complete-answer latency changed only 1355.410→1348.917 ms (0.48%). The search stopped and the original FP16 runtime was restored. A duplicate CUDA-symbol collision required correcting the test's archive copy; this was an agent build correction, not a user action. The [follow-up record](latency10-follow-up.md) separates that experiment, observed results, initial assumptions and known quality limitations.

The [latest 28-slide presentation](../output/cosmos3-edge-findings-with-latency-goal-2026-09-21T01-13-52-175Z.pptx) preserves the original 24 slides and appends the new goal and measured outcome. The direct Orin UI remains at https://192.168.6.252:8443 with the original verified FP16 runtime.
