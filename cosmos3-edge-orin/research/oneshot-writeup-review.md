# First hackathon: review of the supplied write-up

Reviewed 2026-09-21 for the combined 5–10-minute presentation. This is an evidence review of two user-supplied documents describing work dated 2026-09-13 / 2026-09-14. It is not a reproduction of that deployment, an audit of its repository, or a new benchmark. Reported numbers below retain the source's units and timing labels.

## Source provenance

| Supplied file | Bytes | SHA-256 |
| --- | ---: | --- |
| `/Users/asotelo/Desktop/work/Downloads/index.html` | 40,284 | `bbb37170e85d30aa0dd3445a2984460a7f686e092a25ec6f6a55124dbd34b7ff` |
| `/Users/asotelo/Desktop/work/Downloads/slides.html` | 30,072 | `92bfbb9e912f834b2ca775f32867f11293d41c2a8076a72a9fa867fc9df528a5` |

The report is titled *Cosmos3-Edge on Jetson Orin Nano Super — Optimization Report*. The deck is titled *Cosmos3-Edge on Orin Nano Super — Optimization Deck*. Both were read as local HTML/text. No embedded JavaScript, shell example, linked runbook or remote script was executed. The deck contains inline presentation-navigation JavaScript; the report contains no script element. Extracted text in `.qa/user-index-text.txt` and `.qa/user-slides-text.txt` aided review; the supplied HTML files above are the authoritative sources for this note.

The pages link a personal-namespace repository, a frozen one-shot branch/tag, acknowledgements and contributor CSV. Those links were not followed. Their existence in the document is a reported claim, not independent verification of source, reproducibility, contributor identities or feature authorship. The linked public stack pages also were not re-fetched for this review. Dated ecosystem/support claims should be attributed to the supplied report, not presented as a fresh ecosystem audit.

## Motivation and approach

The stated goal was to make a useful local vision-language application fit and respond on an 8 GB Orin Nano Super, retaining memory headroom for other device services. The write-up's broader thesis is that usable capabilities existed across several projects, but the model, board, runtime, quantization and UI were not connected by one discoverable, measured deployment recipe. This is the strongest marketing message supported by the documents: **integration and verification turned available components into a usable edge application**. The stronger universal claim that nobody had previously connected them is not established by the supplied pages.

The approach was iterative and measurement-driven: remove repeated model startup; capture decode CUDA graphs and change the power/clock policy; test smaller overhead changes; identify weight movement as a plausible decode bottleneck; quantize the text tower; then reduce the image-token budget after vision processing became a larger part of the remaining work. The final deliverable is described as an eight-phase runbook with explicit verification gates. This review did not execute that runbook.

Sources: report `#results`, `#engine`, `#rounds`, `#stack`, `#repos`; deck “The actual thesis,” “The gap,” and “Reproduce it.”

## Reported performance and memory

| Reported comparison | Source values | What the presentation can safely say |
| --- | --- | --- |
| Whole campaign, mean request latency | 13.91 s → 619 ms, about 22.5× | Reported improvement from a **process-per-request cold CLI baseline**, across several configuration changes. Not a model-only speedup. |
| Resident serving stage | 13.91 s → 2.07 s, 85.1% lower | Resident runtime, decode CUDA graphs, MAXN_SUPER and static clocks changed together; the source does not isolate their individual effects. |
| Decode throughput | 22.5 → 76.0 tokens/s, 3.38× | Reported historical FP16 versus broadly quantized text-tower result. It is not a matched comparison against the current MLP-only implementation. |
| Process RSS | 6.02 → 3.70 GB, 2.32 GB less | Process resident memory as reported; not dedicated VRAM or a model-only allocation. |
| Available system RAM | 472 → 2,705 MB | Reported headroom in the original result summary. Do not add this to RSS savings. |
| Text engine size | 3.135 → 0.818 GB, 3.83× smaller | Engine-file size, distinct from process and system memory. |
| Image-budget stage | 512 → 320 tokens; about 158 ms lower at matched output-token buckets; 20.8% weighted latency reduction | Reported reduction in fixed work with an input-detail tradeoff. Three natural-scene prompts matched, but dense text degraded. |

The reported round means are 13,910 ms (cold CLI), 2,070 ms (resident stage), approximately 1,990 ms (micro-optimizations), 744 ms (INT4), and 619 ms (image-budget stage). Different tables and later live-demo measurements describe different cohorts. Do not combine them into one synthetic request or use 619 ms and 245 ms as successive optimization rounds.

The later quiesced-board demonstration reports 442 ms median complete-answer latency at a median 18-token caption and roughly one request per second. It separately reports 253 ms median for a 14-token caption, 245 ms best live request, 249 ms best in a controlled 40-run benchmark, and p90 values of 607 ms live / 624 ms benchmark. Its stated decode slope is 13.49 ms/token (74.1 tokens/s).

The live-demo configuration is one 640×360 camera frame and a one-sentence prompt, approximately 184–187 input tokens; INT4 W4A16, 28 decoder layers, hidden size 2048, input capacity 1536, KV capacity 2048, MAXN_SUPER. The campaign also states `jetson_clocks` was applied, with GPU 306→1020 MHz and EMC 2133→3199 MHz. Software is reported as JetPack 7.2.1 / L4T R39.2.1, CUDA 13.2, TensorRT 10.16.2.10 and TensorRT-Edge-LLM 0.10.1.

For that live demonstration, the source reports system RAM 4.95 GB used of 7.37 GB, 2.60 GB available, shim RSS 3.85 GB, CPU median/peak 15.0%/23.5% across all six cores, and GPU median/peak 97.7%/99.6%. These are **shared-memory** observations. A diagram that labels process RSS simply “model” overstates attribution: the process also includes runtime and other resident allocations.

The demonstrated headroom was measured after stopping other services. The report says quiescing reduced system use from 6.73 to 4.95 GB and load average from 7.56 to 0.48. It does **not** demonstrate simultaneous inference with DeepStream and Home Assistant, despite suggesting headroom for them. The 1.8 GB quiescing change and 2.32 GB historical RSS change are separate effects; they must not be added or attributed to one quantization change.

Sources: report `#results`, `#live`, `#rounds`, `#tradeoff`; deck “What changed,” “Live demo,” and “Round 4.”

## Original work, steering and effort

The supplied report describes a task-authored resident HTTP shim, a CPU round-to-nearest INT4 converter with per-group MSE clipping, packing/scale verification, deployment integration, request-log analysis and a gated reproduction runbook. It credits existing runtime, kernels and quantization conventions to their upstream projects. It does not justify claiming invention of INT4, CUDA graphs, the resident runtime, or the upstream model.

The reported quantizer processed **169 text linears**, retaining the vision tower, projector, embeddings and norms in FP16. It searched clipping factors from 0.55 to 1.0 per group and reports relative reconstruction error falling from 13.31% to 11.06%. Model Optimizer on CPU served as a format oracle; the report says the host CUDA extension produced invalid values. Checks covered scale agreement, pack/unpack agreement, nonzero exported weights, 169 quantized plugin nodes, and engine weight-memory accounting. Those checks establish the claimed construction process, not task accuracy.

Documented interventions and corrections include:

- Replacing per-request binary launches with one resident runtime and changing power/clock policy.
- Detecting an encoder-cache measurement artifact when reusing the same image; the report then describes unique input images for probes.
- Retracting a context-cache regression after recognizing serialized contention from live traffic.
- Fixing a zero-initialized `top_k`, testing context reuse and greedy sampling, and rejecting the roughly 4% combined result as below the campaign's threshold. The numeric acceptance threshold is not supplied.
- Abandoning reported blocked AWQ routes and an unexecuted community checkpoint, then implementing RTN plus clipping and verification.
- Reducing batch/KV settings after memory failures and stopping unrelated services for the later benchmark.
- Choosing 320 image tokens after examining the vision cost, while preserving the 512 setting for denser visual information.

The pages do **not** contain a chronological user/agent conversation, individual steering prompts, a count of manual interventions, person-hours, GPU-hours, elapsed end-to-end implementation time, or named participants. They date the work to September 13–14, but that is not a measured two-day labor claim. “CPU, minutes” describes the quantization command and is not an estimate of total project effort. The claim of one-pass gated reproduction is a proposed/frozen artifact, not evidence that the original exploratory work required one prompt or no intervention.

Do not assign these interventions to a particular person or agent. For the combined presentation, distinguish **first-hackathon interventions reported by the supplied write-up** from **second-hackathon user steering captured in this task's actual record**.

Sources: report `#method`, `#rounds`, `#int4`, `#negative`, `#stack`, `#repro`; deck “Before optimizing,” “Trust nothing,” “Negative results,” and “Reproduce it.”

## Contributions and attribution

| Project or component named in the supplied report | Reported contribution |
| --- | --- |
| NVIDIA-AI-IOT/live-vlm-webui | WebRTC capture and OpenAI-compatible UI/client. |
| NVIDIA/TensorRT-Edge-LLM | Resident runtime, decode graph capture, multimodal integration, W4A16 contract and image-token configuration. |
| nvidia/Cosmos3-Edge | The pretrained vision-language model. |
| NVIDIA/CUTLASS / CuTe DSL | The stated INT4 matrix-multiplication implementation underlying the quantized plugin. |
| TensorRT Edge LLM vision plugin | Vision attention and TensorRT plugin integration. |
| NVIDIA/TensorRT-Model-Optimizer | Reference for the quantized on-disk format and scale convention. |
| NVIDIA/TensorRT; CUDA / cuda-python | Engine construction/execution and CUDA access. |
| JetPack / Jetson Linux | Device software, power modes and clock controls. |
| PyTorch; Hugging Face safetensors | Quantizer execution and tensor-file support. |
| aiortc; PyAV; OpenCV; OpenAI client | UI capture/media/client dependencies. |

The report expressly distinguishes repository-level contributors from feature authors. It names no individual contributors in the supplied HTML and instead points to separate files. Its “156 of 156 commits by one author” statement is a dated, unverified repository-history claim; omit the count from a compact presentation unless independently verified from an authorized source. Credit the projects and their maintainers, and use the separately verified contribution ledger for any named individuals. A repository contribution count is not evidence of authorship of a specific kernel or feature.

Sources: report `#stack`, especially “Two things this table does not claim”; deck “Where every piece came from.”

## Timing and configuration caveats for the combined deck

1. **No common timer is established across hackathons.** The supplied first write-up labels request times end-to-end and names preprocessing, vision encode and prefill in the fixed term. Its logging implementation and raw request rows were not supplied. The exact treatment of JPEG encoding/decoding, transport, queueing and response serialization therefore cannot be confirmed. Its complete-answer measurements are not TTFT. Our current `native_start_to_server_text` and `native_inference` fields have explicit, different boundaries; do not subtract or divide across them.
2. **The models are quantized differently.** The first write-up reports 169 text linears under INT4 W4A16 with MSE clipping. The current selected implementation uses 56 MLP linears with its own RTN conversion and N4 kernel change, retaining attention and LM head in FP16. “Both INT4” is insufficient for a speed or quality comparison.
3. **Workload and controls differ.** First-demo live numbers use camera scenes, varying captions, a 320 image cap, MAXN_SUPER/static clocks and input/KV capacities 1536/2048. Our final current profile uses image cap 512, cache budget zero, dynamic stock 25 W clocks and capacities 1024/1664. The bounded TTFT comparison uses one fixed 512×512 geometric image and a different prompt. Actual output length must accompany complete-answer latency.
4. **The first documents disagree on fit sample count.** The report says `R²=0.999` over 699 live requests; the deck says 88. A separate later fit in both uses 2,818 requests and `r=0.854`: `elapsed_ms = 200 + 13.49 × generated_tokens`. These are not interchangeable fits. For ordinary single-predictor regression with an intercept, squaring 0.854 yields about 0.729, not 0.999. Cohorts and raw records are absent, so leave them separate and do not choose the more attractive statistic.
5. **Do not claim caption length explains all variation.** The fitted expression predicts about 389 ms for 14 tokens, while the separate summary reports a 253 ms median at 14 tokens. Regression residuals and differing cohorts could explain disagreement, but the pages do not reconcile it. Their categorical claim of no jitter, contention or thermal contribution is stronger than this evidence supports. A fitted slope is useful; a fitted intercept is not itself a directly measured kernel phase.
6. **Quality is not established broadly.** There is no reported task-suite accuracy benchmark or identical-prompt FP16-versus-INT4 output comparison. Three natural-scene prompts matching at 320/512 do not make 320 universally lossless. The dense-text example degrades. Weight reconstruction error is not an accuracy score. Two engines need not coexist in memory to perform a sequential controlled comparison, so the stated inability to load both simultaneously is not a fundamental comparison barrier.
7. **Hardware superiority is not isolated.** The AGX Orin comparison uses another precision, runtime and unspecified matching workload. The supplied 44.1-token/s external number was not revalidated here. Omit “Nano beats AGX” as a controlled result; if retained, label it as a context example with unmatched software/precision/workload.
8. **Memory terms overlap and units remain source-reported.** Process RSS, system used/available RAM, engine-file size and compiled weight bytes answer different questions. Do not add them or call them dedicated VRAM. The first pages use GB/MB labels without consistently defining decimal versus binary units; preserve that qualification if comparing with this task's explicit GiB/MiB receipts.
9. **Failure/support claims are scoped to the reported setup.** Claims about AWQ export gating, fused-attention TensorRT requirements, FP8 limitations, and a policy model not fitting concern the reported pinned stack. They were not independently retested in this review and should not be generalized to all future versions or deployments.

## Suggested use in a short presentation

Use one first-hackathon slide: the motivation was usable local vision within an 8 GB budget; reported gains came from eliminating cold startup, broadly quantizing text weights, and reducing visual input work. Lead with the reported 6.02→3.70 GB process RSS and 22.5→76 tokens/s, explicitly attributed to the September write-up. Put the 22.5× cold-baseline headline in a footnote or omit it. State the absent accuracy suite and differing timer/configuration in the speaker notes.

Use one comparison/learning slide: the second task independently built a serving system from permitted public components, then exposed how defaults, visual detail, output length, caching, clocks and measurement boundaries change what a speed claim means. Keep first-hackathon reported figures, this task's verified receipts, user-reported observations and proposed next steps visibly distinct. The shared lesson is a need for discoverable deployment recipes, explicit quality gates and reproducible request-level measurements—not a claimed winner between unmatched implementations.
