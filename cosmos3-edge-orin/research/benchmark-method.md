# Reproducible latency and memory measurements

`scripts/benchmark.py` uses Python's standard library. Run it on the Jetson against the actual inference endpoint so local memory observations belong to the serving machine. A remote client can measure HTTP latency, but its local RAM measurements do not describe the Jetson. This tool sends one request at a time; measure the web UI's concurrent workload separately if it differs.

```sh
python3 scripts/benchmark.py run \
  --url http://127.0.0.1:8000/v1/chat/completions \
  --model Cosmos3-Edge \
  --image /absolute/path/to/fixed-scene.jpg \
  --candidate-id baseline \
  --warmup 5 --requests 30 --max-tokens 64 \
  --sample-local --pid 1234 \
  --tegrastats /usr/bin/tegrastats \
  --backend-config /absolute/path/to/engine-config.json \
  --output /absolute/path/to/results/baseline.jsonl
```

Replace the example model alias, backend PID, image, and paths with actual values. The default prompt matches the UI: “Describe the visible scene in one concise sentence. Focus on objects and actions.” The same request can target the UI proxy at `http://127.0.0.1:8090/v1/chat/completions` with a JPEG smaller than its 2 MiB request limit. The UI resizes/re-encodes uploaded images, so use the exact submitted JPEG bytes for an equal-workload comparison. `--sample-local` reads `/proc/meminfo` and `/proc/vmstat`; omit `--pid` or `--tegrastats` when unavailable. Without local monitoring, swapping/OOM evidence remains unknown and the optimizer cannot accept a candidate. The sampler starts tegrastats with the current user's permissions and never invokes sudo. `--api-key-env VARIABLE_NAME` supplies authentication without storing the key. Output files are created exclusively to protect previous results.

Keep the image bytes, prompt, output token limit, model alias, temperature, top-p, and serial concurrency fixed across trials. Their workload fingerprint must match. Temperature is zero; seed is omitted because the pinned backend explicitly rejects non-null seed values despite exposing the field in its request schema. [TensorRT-Edge-LLM v0.10.1 request validation](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/experimental/server/api/serving_chat.py#L157) A fixed token limit does **not** guarantee an equal output length: inspect the recorded actual completion counts and output text. Correctness should cover a separate fixed set of representative images/questions before assigning `--quality pass --quality-note 'Describe the actual correctness check'`; the harness cannot grade scene understanding by itself. A default run records quality as unknown.

Record backend provenance in the supplied JSON object: model revision, source repository revisions, TensorRT-Edge-LLM version, JetPack version, precision, engine build flags, input resolution, visual token budget, context/KV-cache limits, batch size, power mode, and clock settings. Record license/attribution in the project report. Do not put secrets in this object. The JSONL also records the endpoint, script SHA-256, Python/platform details, workload hashes, warmup and measured request records, actual output text, and a summary. Warmup measurements are excluded from latency percentiles and the peak-memory selection metric. Warmup request failures, OOM, and swapping are retained and disqualify a candidate.

TTFT means elapsed client time to the first **nonempty visible text content** in a streaming delta, including upload, queue, vision preprocessing, prefill, and transport. Role-only events do not count. Total latency ends at the `[DONE]` event. The recorded throughput is the server's integer `usage.completion_tokens` divided by the entire request duration; it is **end-to-end completion-token throughput**, not isolated GPU decoding speed. No character, word, SSE-chunk, or guessed-token conversion is used. Missing usage produces `null` for token counts and throughput. The server is asked for streaming usage; unsupported options, malformed SSE, missing `[DONE]`, HTTP failures, and empty output are recorded as errors. The timeout applies to socket operations and is also checked when events arrive; it is not a strict process-level wall-clock deadline during a blocked socket read.

Memory counters remain separate:

- `local_system_ram_unavailable_peak_bytes` is sampled `MemTotal - MemAvailable` on the host. It includes the system and other services and excludes RAM the kernel considers reclaimable. On a Jetson this is an observation of unified system RAM, **not dedicated VRAM**.
- `tegrastats_ram_used_peak_bytes` preserves tegrastats' separate RAM-used observation. It is not interchangeable with `MemTotal - MemAvailable`.
- `local_process_rss_peak_bytes` is the selected PID's sampled `VmRSS`, excluding child processes. It does not account for all GPU or system memory.
- CUDA allocated/reserved counters are `null` until the inference backend provides allocator instrumentation. No such values are inferred from RSS or total RAM.

These counters overlap; never add them. Sampling every 100 ms can miss brief peaks. The request records retain samples for auditing. Swap occupancy alone is not proof of workload swapping: an increase in `/proc/vmstat` `pswpin` or `pswpout` disqualifies a candidate. An increase in `oom_kill` also disqualifies it. This counter measures kernel OOM kills, not all GPU allocator failures; backend error responses and error/cancelled finish reasons are rejected separately, and backend logs must also be checked for recovered allocation failures. These are system-wide signals, so unrelated concurrent workloads can cause a conservative rejection. Missing/reset counters are unknown, never a clean bill of health. Keep other workloads and power/thermal conditions stable.

After the quality checks, run candidate files in their actual trial order:

```sh
python3 scripts/benchmark.py evaluate \
  /absolute/path/to/results/baseline.jsonl \
  /absolute/path/to/results/candidate-01.jsonl \
  /absolute/path/to/results/candidate-02.jsonl
```

The first eligible run is the baseline. Each later candidate must pass quality, have zero errors and verified absence of workload swapping/OOM, contain at least 30 measured requests by default, and use the same workload and memory metric. Accept it only if p50 total latency **or** peak memory improves by at least 5%, while the other metric and p95 total latency each regress by no more than 5%, compared with the current accepted candidate. Stop after **12 total candidates including baseline** or **three consecutive non-improvements**, including rejected candidates. Acceptance resets that consecutive counter. The evaluator reports and ignores input after a stopping boundary; it does not launch more experiments. Limits can be tightened but not expanded. `--minimum-requests` can be lowered for protocol smoke tests; such runs should not support performance conclusions. Report the best accepted configuration within this bounded experiment, not a claim of global optimality.

This evaluator covers the bounded candidate experiment only. The project's separate completion gates still require representative-image quality checks, a 10-minute sustained UI run, no growing queue/memory use, and at least 512 MiB `MemAvailable` throughout that run. Retain the raw candidate files for a Pareto comparison and repeated/paired near-threshold checks; the evaluator reports an incumbent, not the complete Pareto frontier. It does not certify those additional deployment gates.

Run parser, local HTTP protocol, memory-accounting, and stopping-policy tests with:

```sh
python3 -m unittest discover -s tests -p test_benchmark.py -v
```

These tests use a local synthetic SSE fixture solely to verify protocol handling. They are not measurements of Cosmos3-Edge, TensorRT-Edge-LLM, a GPU, or a Jetson.
