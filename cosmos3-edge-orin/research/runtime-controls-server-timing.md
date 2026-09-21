# Runtime image budgets and server-side scoring

The current default is the first demo's **Lightweight** behavior, restored at the user's explicit request after an intermediate runtime-controls experiment. The user then explicitly requested the original engine as well. After that temporary FP16 restoration, the user requested returning to INT4 MLP and measuring TTFT, with FP16 measured first. INT4 MLP is now selected with input 1024/KV 1664 and aggregate/per-image visual capacity 512. Advanced image controls, resource charts, closed captions and native-only timing remain. The camera and generation defaults match the first demo; the selected MLP engine, instrumentation and UI are newer.

The intermediate request was image cap320, static clocks,256MiB encoder cache and top-p0.95. Functional acceptance passed for that policy. While a token-length benchmark was active, the user reported slower live behavior and explicitly chose “Very first demo: restore its original lightweight defaults.” The benchmark was stopped immediately. GPU contention from background measurement was possible; no controlled evidence attributes the reported slowdown solely to the settings. The unfinished run is not a completed performance comparison.

## Input tokens versus output tokens

The main Output token limit now defaults to64, matching the first demo; Live VLM WebUI uses512. It caps generated answer length. Advanced Input image token budget defaults to512 and limits visual tokens derived from one image. It changes the amount of visual detail supplied to the model; it is neither a camera-resolution selector nor the number of generated words. Actual image-token count can be below the cap because the native resize must fit its patch grid.

The UI offers 320, 512 and an integer Custom value from 4 to the loaded capacity (512 here). Values apply to the next request, without restarting the camera. A new backend engine ID resets image budget and top-p to the engine-load defaults. First-load capture is Lightweight: ideal640×480 camera at15FPS (maximum30), longest side512, JPEG quality0.8, one-second minimum start-to-start interval and temperature 0. Output is capped at64 tokens. The original prompt is “Describe the visible scene in one concise sentence. Focus on objects and actions.” Greedy decoding resolves top-p/top-k to1/1. Live VLM capture and all ten upstream prompts remain selectable; the capture toggle preserves edited prompt/output/advanced values.

The selected MLP engine supports512 tokens per image, with explicit aggregate/per-image build overrides512/512 and runtime default512. During the temporary FP16 restoration, an explicitly empty `COSMOS_ENGINE_MAX_IMAGE_TOKENS_PER_IMAGE=` preserved its original unset builder override and cache identity (aggregate 1024/per-image 512). The original public API only exposed the build parameter. The task's patch adds a bounded per-image override to the native request and uses the original Cosmos/Qwen CUDA resizing on both cache-miss and cache-hit paths. It does not substitute browser or Pillow resizing. A smaller request budget can reduce work but does not shrink retained engine buffers proportionally.

The encoder-cache identity includes pixels, source geometry, resize mode, modality and image budget, so the same image at 320 and 512 cannot reuse incompatible embeddings. A separate benchmark-only flag bypasses cache lookup and storage without changing image pixels or evicting normal cached entries. It is accepted only by the loopback backend, not the LAN UI proxy.

## Load defaults

| Setting | Selected value | Meaning |
| --- | ---: | --- |
| Runtime input image cap | 512 | Default per-image visual budget; UI can override within the loaded profile |
| Built per-image capacity | 512 | Retained engine capacity; switching does not rebuild the engine |
| Output token cap | 64 | Maximum generated answer length, separately editable |
| top-p | 1 | Engine default; Lightweight uses temperature 0 and greedy decoding |
| `COSMOS_STATIC_CLOCKS` | 0 | Dynamic clocks within the existing25W power mode |
| `COSMOS_ENCODER_CACHE_BYTES` | 0 | Encoder embedding cache disabled, as in the first demo |

Clock/cache values are reported by the loaded backend in the advanced panel. They are read-only in the browser and require a backend restart to change. The later restoration request supersedes the earlier320/static/cache defaults. The saved stock-clock snapshot was restored and the clock service disabled; CPU/GPU/EMC dynamic frequency ranges are checked in the deployment receipt. The vendor restore utility emitted optional GPU persistence-file warnings; frequency restoration is verified independently. Text context reuse stays disabled.

## Timing boundary and logs

`NativeTimer` records a monotonic timestamp immediately before `LLMRuntime.handle_request` and another immediately after it returns, inside the inference lock. Decoding the JPEG into native image data, preparing the request and acquiring admission/locks occur before the timer. SSE serialization, proxy writes and browser delivery occur outside it. Native image preprocessing/vision encoding, prompt tokenization/prefill, token generation and native result handling are included. This is native server wall-clock inference time, not a GPU-event measurement.

The server publishes the duration, actual native completion-token count and a request ID in `cosmos_metrics` before SSE `[DONE]`; non-streaming responses include the same object. The UI accepts only a successful stream with the declared `native_inference` / `server_monotonic` boundary. It does not substitute a browser timer when server metrics are unavailable. Browser first-text, round-trip and frame-age diagnostics remain explicitly separate.

The server also appends bounded, rotated JSONL logs at `data/logs/native-requests.jsonl`. Records contain input hashes, controls, native duration, actual output count, finish reason and observed encoder-cache state. They contain neither images nor prompt/output text. Always-on host counters identify actual vision processing without enabling GPU profiling. Failed/canceled requests are not scored. Warmup labels and run IDs are benchmark annotations, not measurements.

## Interrupted fixed-work and marginal-token experiment

The interrupted experiment planned to use one fixed 1280×720 image and the Live VLM accessibility prompt. It randomizes output caps 8/16/32/64 across input budgets 320/512 and cache modes, with eight repetitions per cell and three warmups per budget/mode: 128 measured requests plus 12 warmups. Native cache-hit measurements and requests that execute the vision encoder are kept separate. Actual output counts, not requested caps, determine the fit.

For each matched input/cache stratum, the comparator fits:

`native inference milliseconds = fixed_ms + marginal_ms_per_token × actual completion_tokens`

It reports the sample count, observed token range, residual error, R² and 95% moving-block bootstrap intervals. `fixed_ms` is the fitted zero-output intercept, not a directly measured prefill duration. It is not clipped if negative. At least three distinct output lengths and enough samples/span are required; a single deterministic answer length cannot identify both terms. Comparisons explicitly disclose the different image budgets and never mix cache hits with misses/bypass.

The run stopped after110 measured responses and12 warmup responses, short of128 planned measurements. The backend logged one additional in-flight completion after its client stopped (111 measured native records total). All partial records and their request IDs are preserved in `results/runtime-controls/token-sweep/`; no fitted comparison or performance gain is published. The lightweight settings were restored on MLP first; the user then requested the original engine, so the preserved FP16 cache was selected as well. Do not run a benchmark concurrently with the interactive demo when assessing responsiveness.

The six historical MLP benchmark logs contain 210 client-timed requests. Those cannot be retroactively converted into native inference measurements or used to rank another backend under this boundary. The comparator rejects them. The interrupted run targeted two image-budget configurations of one MLP INT4 backend. It is neither a new FP16-versus-INT4 result nor evidence of broad caption accuracy.

## Functional evidence and interventions

- Intermediate acceptance: `results/runtime-controls/validation-routes-fixed.json`,17successful requests and4invalid budgets. Image caps320/512/custom384 produced299/480/364 observed visual tokens respectively on the fixed1280×720 input. Cache-hit and bypass captions agreed within each tested budget.
- Held-object smoke checks recognized the action camera, pen and scissors at both320and512, but captions could add unsupported intent (for example “as if they were a sword”). This is not accuracy parity.
- Initial `validation.json` records rejection of the extra API fields: FastAPI0.139.2 retained a lazy included router, so a flat path filter left the upstream route first. The task now replaces that inclusion with a local filtered router, preserving upstream guards and routes. Real FastAPI route tests cover the regression.
- Both incremental native builds completed without OOM or swap use. The second binding is recorded in `native-artifacts.json`; the engine and model artifacts were retained.
- Readiness waits were extended after slow startup. Historical NVMe timeout warnings already documented in the project can affect cold starts; no storage repair or causal performance claim is implied here.
- Interim MLP Lightweight acceptance (`results/runtime-controls/validation-lightweight.json`) passed17 requests. The attempted original-FP16 acceptance command failed at SSH connection before running; no such acceptance receipt is claimed. The subsequent paired TTFT run validates35 successful requests per engine. Final browser checks deliberately delay one real response700 ms and verify that neither native latency nor server TTFT absorbs that delay.

## Reproduction

Apply `patches/cosmos-runtime-image-token-budget.patch`, then `patches/cosmos-encoder-cache-bypass.patch` to the pinned, previously patched public checkout. Rebuild `_edgellm_runtime` and its native dependencies before starting the selected service. Preserve the existing engine cache and the old native binary for rollback. The launchers pass the encoder-cache budget into the real `ContextCacheConfig`; changing the environment alone was insufficient in the prior INT4 launcher.

For a separately scheduled measurement when the demo is idle, run `scripts/benchmark_native.py` on the Orin with the fixed image, a new output directory and the actual server request-log path. It saves its plan before sending requests and extracts authoritative log records by request ID. Run `scripts/compare_server_timings.py` against those extracted records with `--vary-control max_image_tokens_per_image`. The measured client response records are audit material; the comparator reads server logs.

The benchmark plan requires the intermediate static-clock/256MiB-cache policy and validates cache hits. It intentionally does not run unchanged against the restored zero-cache default. `scripts/validate_runtime_controls.py --policy lightweight` checks the current policy; `--policy experimental320` describes the superseded experiment. Do not change the active deployment merely to finish the interrupted benchmark.

## Server TTFT and the requested engine switch

The user asked whether TTFT was still approximately 250 ms and requested returning to INT4 MLP. Before switching, the user added: “before you do measure now with BF16 up and compare once you switch engines.” The preserved engine was FP16, not BF16. The existing FP16 engine was measured first, then INT4 MLP; both use the same current native binary and Python timing instrumentation. Enabling the new TTFT observer required restarting the Python service with the same FP16 engine before baseline measurement. No engines were rebuilt for this comparison.

The new `server_first_text_ms` starts at the same post-admission, post-JPEG-decode native entry point as total inference. It ends when the server generation iterator receives its first nonempty text delta, before SSE serialization and network transport. It includes native visual preprocessing, vision encoding, prefill, generation and server consumer scheduling. It is a server wall-clock measurement, not a CUDA event or first token-ID timestamp. Empty chunks and the end marker do not count; whitespace matches the historical first-nonempty-text definition. A delayed consumer can observe text after native generation finishes; durations are not clamped or rewritten. One final log record correlates to the response ID, and canceled/failed requests are not scored. The UI's TTFT tile updates at successful answer completion; browser first-visible-text remains a separate diagnostic.

`scripts/benchmark_ttft.py` saves a fixed plan, performs five warmups then 30 measured serial requests, and matches every returned request ID and timing to the native server log. The input is the original already-encoded512 × 512 shape fixture, SHA256 `52074a486ce36764d48e86fec209694ade16137318545524a3b51e47a048ca55`; its original left/right shape-and-color prompt, temperature 0, top-p 1, max output 64, image cap 512, cache 0 and dynamic 25 W clocks are held constant. It records other logged requests whose start timestamps fall within the run. Each backend receives the same input and controls; the compact MLP engine's retained capacities differ from FP16 by design. The benchmark fixture has256 actual image tokens under the 512 cap. This fixture and prompt differ from a live camera scene, so the result does not promise250ms on every camera frame.

The historical original FP16 measurement (`results/raw/benchmark-final-service-01.jsonl`) was loopback HTTP request start to first nonempty SSE text on port8090, including proxy/JPEG decoding. Its median was265.540ms and p95 was267.365ms. New server-only TTFT must not be equated directly to that broader historical metric. The paired experiment separately retains the same loopback HTTP diagnostic for an appropriate historical comparison. Requested output caps and actual output counts remain distinct; this single deterministic answer per engine cannot identify both a fixed intercept and marginal milliseconds per token. No fixed/marginal fit is claimed from the TTFT run.

| Engine | Server TTFT median | Server TTFT p95 | Loopback HTTP TTFT median | Native complete-answer median | Actual output tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original FP16 | 249.50 ms | 250.57 ms | 267.67 ms | 1804.56 ms | 37 |
| INT4 MLP | 239.32 ms | 240.72 ms | 261.67 ms | 1008.97 ms | 31 |

Each row has30 measured requests after5 warmups. Observed server TTFT median fell by10.18ms (4.08%). FP16 remains approximately 250 ms under the new server boundary, and INT4 MLP is below 250 ms on this fixture. Original historical FP16 loopback HTTP265.54 ms versus current267.67 ms is a much closer boundary comparison than mixing native TTFT with browser latency. Both models generated deterministic but different-length answers; the total-duration difference does not establish an equivalent throughput speedup.

One additional logged, completed request started during the MLP run, versus zero within the FP16 window, and one loopback HTTP TTFT reached 849.36 ms. This counter uses logged request-start timestamps and does not detect all possible overlapping activity; the outlier is not conclusively attributed to that request. Native timing excludes queueing and network, but the run was not fully exclusive; shared scheduling and dynamic-clock effects remain possible. Engine order was sequential, with FP16 measured first as the user requested. No repeated search or memory benchmark was run. The prescribed 70 requests (including warmups) completed and the measurement stopped. Exact samples, native logs, controls, hashes and limitations are in `results/runtime-controls/ttft-comparison.json` and its two run directories.

The final browser check passed six real inference requests, manual/live operation,512/320/custom 384 budgets, desktop/mobile layout and resource sparklines. Holding one completed response700 ms left both server TTFT and native latency unchanged while increasing browser delay. Final deployment retains INT4 MLP and the original lightweight defaults.
