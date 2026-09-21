# MLP-only INT4 optimization goal

The user chose **MLP-only round-to-nearest (RTN) INT4 as the new starting point**, then asked to improve latency and RAM footprint, stop when p50 could no longer improve by 10%, and load the optimized model. The search has stopped after one measured candidate: its observed latency improvement was **1.19%**, below 10%, while sampled peak shared RAM fell **3.12%**. The smaller `live-vlm-mlp-n4-compact` configuration is selected for deployment. **Service, browser and ten-minute stability checks passed.** The earlier FP16 selection, rejected candidates and completed latency experiment remain historical records.

The agent initially treated 10% as both the search-continuation threshold and a deployment requirement. An independent review of the user's wording corrected that interpretation after measurement: stop further experimentation, but keep the useful memory reduction without material latency or output regression. The [policy clarification](../results/mlp-goal/policy-clarification.json) and [selection receipt](../results/mlp-goal/selection.json) record this decision. The original strict plan and its failed comparison are preserved; no 10% latency success is claimed.

The [new plan](../results/mlp-goal/plan.json) was frozen at **2026-09-21 03:51 UTC**, before baseline measurement. Its source checkpoint is public branch `codex/cosmos3-edge-checkpoint`, commit `9ed3ffd9263ae887ee6a7aded77368da220525a1`; the corresponding local source commit is `a1048ff6c071f6e7477cae544899de97171781eb`. The user explicitly authorized publication to [this repository](https://github.com/nv-asotelo/porch-dad/tree/9ed3ffd9263ae887ee6a7aded77368da220525a1). That exception does not authorize unrelated personal or internal sources.

The immutable reasoner source is NVIDIA Cosmos3-Edge revision `344d602b128d1bbdacb43b08d0a3626f46343e29`, with TensorRT-Edge-LLM revision `e8b29522938901f6df19ebeedd4b69bc8edbcd97`. The original CPU converter quantized 56 MLP linear weights using groupwise RTN, without calibration; it uses GPTQ-compatible storage, not GPTQ optimization or AWQ calibration. Attention, LM head, vision, projector, embeddings, activations and KV remain FP16. The derivative weight file SHA256 is `93ac92d48d893aa45009fb327846bdd1052d03b22150ab8b64698d00a23d1854`. See the [conversion receipt](../results/rtn-mlp-conversion-receipt.json) and [baseline configuration](../results/mlp-goal/baseline-config.json).

The fresh baseline uses INT4 plugin V1, batch one, input capacity 1,024, KV capacity 2,048, aggregate image-token capacity 1,024 and per-image capacity 512. Encoder embedding cache budget is zero and text context reuse is disabled. Stock 25 W power mode and restored dynamic clocks remain fixed.

The [fresh synthetic diagnostic](../results/mlp-goal/quality-baseline-review.json) reproduced all six prior MLP answers exactly: **18/19 required facts**, with the yellow triangle still described as orange. The user accepted this limitation when selecting the new baseline. It remains an error, but it does not trigger a return to FP16 under this new goal. That diagnostic uses greedy decoding with a 64-token ceiling; it is separate from the default-preset latency workload and is not a general accuracy estimate.

## Fresh default-preset baseline

The three [frozen 1280×720 JPEGs](../benchmarks/live-vlm-1280/manifest.json) are action-camera, pen and scissors crops from the user's screen recording, resized and encoded by Chromium canvas at JPEG quality 0.75. They are not original sensor frames. Every configuration receives the same bytes, the prompt **“Describe what you see in this image in one sentence.”**, a 512-token ceiling and temperature 0.7. Requests omit top-p/top-k; the pinned backend resolves them to 0.9/50. The native sampler's existing Philox seed 42/offset 0 is unchanged. Repeatability is verified from actual outputs and usage counts.

Five warmups and 30 measured requests per image produced **90 successful measured requests**. The primary metric is the geometric mean of the three per-image median complete-answer latencies, giving each image equal weight. The [baseline summary](../results/mlp-goal/baseline-summary.json) reports **915.676 ms**. Continuing the search required **824.108 ms or lower**, as well as every other gate.

| Image | Complete p50 / p95 (ms) | First text p50 / p95 (ms) | Completion tokens |
| --- | ---: | ---: | ---: |
| Action camera | 996.912 / 1008.730 | 473.408 / 484.495 | 21 |
| Pen | 877.008 / 882.004 | 484.440 / 487.643 | 16 |
| Scissors | 878.142 / 883.249 | 484.860 / 489.648 | 16 |

All 30 measured outputs for each image were identical:

- Action camera: “A man holds a GoPro camera up to his face as if he is taking a selfie.”
- Pen: “A man holds a pen in front of his face while raising three fingers.”
- Scissors: “A man is holding a pair of scissors in front of his face.”

These responses are the preservation targets, not a claim that every description or inferred intention is correct. Three recording-derived examples cannot establish representative model accuracy. The earlier FP16 default-preset run generated different answers and token counts; its latency is not the denominator for this optimization. The still-earlier 512×512 greedy MLP run also used a different workload and observed swap activity.

Across warmup and measured requests, sampled peak system RAM unavailable was **5,495,009,280 bytes (5.118 GiB)**; peak process RSS was **4,990,783,488 bytes (4.648 GiB)**. Minimum available shared RAM was **2,417,766,400 bytes (2.252 GiB)**. Orin CPU and GPU share RAM: `MemTotal - MemAvailable` includes OS and other applications, while RSS overlaps that counter. They must not be added or described as dedicated VRAM allocations.

Raw samples and [boundary/isolation evidence](../results/mlp-goal/baseline-isolation.json) show no OOM or swap I/O, exactly 105 backend POSTs including warmups, and zero active or queued requests afterward. The measurements include image decoding, inference and Orin-local proxy streaming. Browser capture/encoding, Wi-Fi and the every-30-video-frames scheduling cadence are outside this isolated benchmark.

## Measured candidate and original strict contract

The measured candidate combines a [shape-specific kernel patch](../patches/int4-gemv-cosmos-mlp-n4.patch), increasing batch-one INT4 output tiling from two to four, with a compact engine profile. Aggregate image-token capacity falls from 1,024 to 512 and KV capacity from 2,048 to **1,664**, while input capacity stays 1,024, output ceiling stays 512 and per-image capacity stays 512. The minimum KV capacity is **1,537** including the upstream reserved slot. Actual image bytes, measured outputs and token work remain unchanged. See the [candidate configuration](../results/mlp-goal/candidate-config.json). The compact engine build completed in **245.33 seconds** and the native rebuild in **29.68 seconds**, without observed OOM or swap I/O; these are build measurements, not inference timings.

NVIDIA's public TensorRT-Edge-LLM provides the runtime and CUDA kernel. The task contributes the shape-specific dispatch change, standalone diagnostic harness, controlled measurement and new comparison policy. The [kernel profiler](../scripts/profile_mlp_gemv.cu) compared the two template variants on synthetic packed weights: **36 cases passed**, comparing **202,752 FP16 values bitwise** with output guards intact. The [GPU receipt](../results/mlp-goal/profiling/gemv-graph.jsonl) records median batch-mean kernel times of **0.151483→0.148869 ms** for `(N,K)=(9216,2048)` and **0.126013→0.115485 ms** for `(2048,9216)`, observed reductions of **1.73%** and **8.35%**. Each variant used 12 alternating-order trials of 200 graph-captured calls. The first shape's trial times visibly vary; these percentages describe this diagnostic run, not a general speed guarantee.

The profiler's synthetic 56-GEMV projection is **7.770→7.402 ms**. It is not an actual model trace or an inference-latency measurement. Timing repeatedly uses the same synthetic buffers. The receipt's per-case `compared_fp16_bits` field counts FP16 values, while its summary names that unit correctly; `warmup_per_variant=100` excludes the additional one-graph, 200-call warmup. These metadata limitations do not change the full-model comparison below.

The [MLP comparator](../scripts/summarize_mlp_goal.py) reuses the original raw JSONL validator without changing historical policy. Its original frozen acceptance contract requires:

1. At least 10% lower geometric-mean complete-answer p50 than the last accepted MLP run, with no image's p50 or p95 regressing by more than 5%.
2. Byte-identical measured answer text, actual completion-token counts and normal `stop` completion for every image and repetition. Shorter answers or reduced actual input work do not qualify.
3. No request errors, OOM or swap I/O, at least 512 MiB available shared RAM, and no more than 1% increase in sampled peak shared RAM.
4. The same MLP derivative, prompt, image hashes, sampling, endpoint, power/clock policy, input capacity and per-image capacity. Only runtime fields explicitly listed in the plan may vary. Both native-extension hash fields must agree when present.

Memory changes are reported separately for shared RAM and process RSS, against both the incumbent and the initial MLP baseline. Positive decrease values mean lower sampled peaks; they are not sums of overlapping counters. The comparator cannot infer absence of other clients from request JSONL alone, so device readiness and isolation receipts remain part of the evidence.

The original plan permits one matched baseline/candidate repeat only when the threshold or equivalence is ambiguous. It also instructed restoration of the incumbent after failure. The later policy clarification changes deployment selection, not the recorded measurements or the requirement to stop. The old FP16 strict-quality gate does not override this user-selected MLP starting point. A bounded result does not establish a global optimum.

## Full comparison and deployment selection

The [measured comparison](../results/mlp-goal/comparison-01.json) reports geometric-mean p50 **915.676→904.818 ms**, an observed **1.1857% reduction**. All 90 measured candidate answers and completion counts exactly match the MLP baseline. All p95 values improved; the action-camera median increased by only 0.364%, within the frozen 5% guard. The small aggregate difference may include measurement noise and is not presented as a robust general latency gain.

| Image | Candidate complete p50 / p95 (ms) | Candidate first text p50 / p95 (ms) | Completion tokens |
| --- | ---: | ---: | ---: |
| Action camera | 1000.544 / 1006.562 | 481.411 / 486.821 | 21 |
| Pen | 859.030 / 873.228 | 468.259 / 482.636 | 16 |
| Scissors | 861.865 / 875.330 | 471.060 / 483.113 | 16 |

Peak shared RAM unavailable decreased from **5,495,009,280 to 5,323,616,256 bytes**, saving **171,393,024 bytes (163.45 MiB, 3.119%)**. Process RSS separately decreased from **4,990,783,488 to 4,759,748,608 bytes**, saving **231,034,880 bytes (220.33 MiB, 4.629%)**. The candidate's minimum available shared RAM was **2,589,159,424 bytes (2.411 GiB)**. These overlapping memory savings must not be added. The [candidate isolation receipt](../results/mlp-goal/candidate-isolation.json) confirms 105 expected POSTs, unchanged swap/OOM counters and an idle, empty queue afterward.

The [candidate synthetic diagnostic](../results/mlp-goal/quality-candidate-review.json) also reproduced all six MLP baseline answers exactly, retaining the known **18/19** score and yellow-to-orange error. This is a regression check, not evidence that general visual accuracy improved.

The only failed original acceptance gate was the 10% latency threshold. The search therefore stopped without another performance trial. The deployment decision nevertheless selects the smaller candidate, reflecting the user's footprint objective and the distinction between continuing research and retaining a useful result. The [clarification receipt](../results/mlp-goal/policy-clarification.json) explicitly records its post-measurement timing and hashes of the unchanged plan and comparison. The original comparison still says `accepted: false`; the separate selection receipt names `live-vlm-mlp-n4-compact`.

**Selected-service verification and the new sustained soak passed.** The [service receipt](../results/mlp-goal/verification.json) verifies the actual model/cache arguments, native and plugin hashes, loaded mappings and three identical functional answers. [Direct HTTPS browser checks](../results/mlp-goal/browser-verification.json) confirm real upload inference, streamed text, Live VLM defaults, synthetic moving-camera stop/restart and CPU/GPU/shared-RAM readings. The [current interface snapshot](../output/cosmos3-edge-mlp-current-ui.png) shows one successful upload. The [600.5-second soak](../results/mlp-goal/soak-final.summary.json) completed **569 requests** across six diagnostic images without errors, OOM, swap I/O or queued backlog. Minimum available shared RAM was 2,593,685,504 bytes (2.416 GiB). Median RSS decreased by 1,122,304 bytes across the specified windows. This is API stability evidence, separate from accuracy grading and physical-camera behavior. Both services are enabled and active with no restarts. A physical reboot was not tested. Open [the direct Orin UI](https://192.168.6.252:8443).

## Reproduce the measurements

On the Orin, load the intended MLP profile, verify it is ready and isolated, and record the current binary hashes and profile in a fresh backend configuration. The frozen baseline configuration is appropriate only for the exact baseline binary/profile. Use a new output directory; benchmark and comparison files are created exclusively.

```bash
cd /home/jetson/cosmos-edge
run_dir=results/mlp-goal/reproduction-baseline
mkdir -p "$run_dir"
backend_pid=$(systemctl show cosmos-edge-backend --property=MainPID --value)
for fixture in 01-action-camera 02-pen 03-scissors; do
  python3 scripts/benchmark.py run \
    --url http://127.0.0.1:8090/v1/chat/completions \
    --model Cosmos3-Edge --image "benchmarks/live-vlm-1280/$fixture.jpg" \
    --prompt 'Describe what you see in this image in one sentence.' \
    --candidate-id live-vlm-mlp-baseline \
    --warmup 5 --requests 30 --max-tokens 512 --temperature 0.7 --top-p omit \
    --sample-local --pid "$backend_pid" --tegrastats /usr/bin/tegrastats \
    --backend-config results/mlp-goal/baseline-config.json \
    --output "$run_dir/baseline-$fixture.jsonl"
done
```

Repeat the identical requests for the candidate, with its own candidate ID, backend configuration and output prefix. Then compare the saved groups, using actual run directories and prefixes:

```bash
python3 scripts/summarize_mlp_goal.py \
  --baseline-dir results/mlp-goal/raw --baseline-prefix baseline \
  --candidate-dir results/mlp-goal/raw --candidate-prefix candidate \
  --initial-dir results/mlp-goal/raw --initial-prefix baseline \
  --output results/mlp-goal/reproduced-comparison.json
```

The comparator can identify an incumbent with `--baseline-*` and the original MLP baseline with `--initial-*` for cumulative memory reporting. This search has stopped; those options do not authorize another trial. Exit 0 passes the original strict contract, 1 reports its failed gate, and 2 denotes invalid evidence. Reproducing this comparison should return **1**; the separately recorded deployment clarification is not an override hidden in the evaluator. The comparator performs no device operations or deployment. Its current policy and inherited validator are recorded by hash in every comparison receipt.
