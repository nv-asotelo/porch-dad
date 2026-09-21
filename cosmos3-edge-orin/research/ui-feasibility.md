# Public streaming UI feasibility and implementation

Research date: 2026-09-19. Public sources only. No projects in the excluded personal namespace, internal communications or internal repositories were used. No model weights were downloaded in this subtask.

## Public model and backend availability

`nvidia/Cosmos3-Edge` is publicly available and ungated. The official model card identifies a 4B Cosmos 3 Edge model, released July 20, 2026, under OpenMDW 1.1; the repository API returned revision `344d602b128d1bbdacb43b08d0a3626f46343e29`. Its reasoner accepts text with image/video input and returns text. The model card distinguishes reasoner and generator modes and says only BF16 has been officially tested. Lower precision deployment needs independent quality checks. These facts establish public availability, not suitability or performance on an Orin Nano. [Model card](https://huggingface.co/nvidia/Cosmos3-Edge), [public model metadata](https://huggingface.co/api/models/nvidia/Cosmos3-Edge).

TensorRT-Edge-LLM provides a Cosmos3 reasoner consisting of a SigLIP2 visual encoder, PatchMerger and autoregressive decoder. Its policy path has separate components. Avoid loading generator/policy/VAE components for the requested scene-to-text interface. [Cosmos3 design](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/v0.10.1/docs/source/developer_guide/models/cosmos3.md).

The public `v0.10.1` experimental server implements `/health/ready`, `/v1/models`, and `/v1/chat/completions`; streaming chat emits SSE. Its documented server holds one native generation state, uses bounded admission, keeps engines resident, and cancels native work on disconnect. `stream_options.include_usage=true` exposes native usage data. `seed` is rejected by upstream; the benchmark and UI use `temperature=0` and may pass `top_p=1`. The server's direct builder accepts model checkpoints rather than prebuilt engine paths. [Pinned routes](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/v0.10.1/experimental/server/api/routes.py), [pinned server guide](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/v0.10.1/docs/source/user_guide/examples/experimental-server.md).

Initial research inspected main `e8b29522938901f6df19ebeedd4b69bc8edbcd97`; integration endpoints were subsequently confirmed in release `v0.10.1`. Actual native Cosmos3 engine build/load on the target remains a hardware verification step.

## What the reference UI contributes

Live VLM WebUI main `2fd5ba0b334c334d24bf0f9439d8742b243d22be` provides an Apache-2.0 webcam/WebRTC interface, OpenAI API configuration, prompt controls and monitoring. Public package metadata credits NVIDIA Corporation. GitHub contribution metadata lists `tokk-nv` (162 commits; public name Chitoku YATO) and `chitoku` (1 commit). Commit counts are a point-in-time API snapshot, not an exhaustive allocation of authorship. [License](https://github.com/NVIDIA-AI-IOT/live-vlm-webui/blob/2fd5ba0b334c334d24bf0f9439d8742b243d22be/LICENSE), [package metadata](https://github.com/NVIDIA-AI-IOT/live-vlm-webui/blob/2fd5ba0b334c334d24bf0f9439d8742b243d22be/pyproject.toml), [contributors](https://github.com/NVIDIA-AI-IOT/live-vlm-webui/graphs/contributors), [Chitoku YATO](https://github.com/tokk-nv).

Its VLM service converts selected frames to JPEG data URLs and calls OpenAI chat without `stream=true`, publishing an answer after completion. Its video processor asynchronously samples incoming video and passes video back to the client, with an application lock skipping inference when busy. The native video still requires a WebRTC media pipeline. These observations motivated a smaller original interface with local browser preview and actual token streaming. [VLM service](https://github.com/NVIDIA-AI-IOT/live-vlm-webui/blob/2fd5ba0b334c334d24bf0f9439d8742b243d22be/src/live_vlm_webui/vlm_service.py), [video processor](https://github.com/NVIDIA-AI-IOT/live-vlm-webui/blob/2fd5ba0b334c334d24bf0f9439d8742b243d22be/src/live_vlm_webui/video_processor.py).

The reference server can auto-discover or fall back to a cloud API when unspecified, and its browser page references external script CDNs and STUN services. This deployment instead supplies its own static assets and a fixed local inference proxy. [Reference server](https://github.com/NVIDIA-AI-IOT/live-vlm-webui/blob/2fd5ba0b334c334d24bf0f9439d8742b243d22be/src/live_vlm_webui/server.py), [reference browser source](https://github.com/NVIDIA-AI-IOT/live-vlm-webui/blob/2fd5ba0b334c334d24bf0f9439d8742b243d22be/src/live_vlm_webui/static/index.html).

## Original work in this workspace

`web/index.html`, `web/style.css`, and `web/app.js` implement an original static interface. `scripts/serve_ui.py` uses Python's standard library to serve it and proxy the three local backend endpoints. No reference-UI source code or assets were copied. Credit the reference UI for the workflow, NVIDIA Cosmos for model weights, and NVIDIA TensorRT-Edge-LLM for model execution, kernels, builders, and the OpenAI server. The new work is interface, transport, admission, instrumentation and validation—not a new model or inference kernel.

The browser displays its local camera stream, sends one prompt/image message and consumes SSE token deltas. There is one request in flight, no retained dialogue and no pending frame queue. The default Live VLM WebUI preset requests ideal 1280×720 without a frame-rate constraint, samples every 30 browser-presented video frames, sends native captured dimensions at nominal browser JPEG quality 0.75, and skips eligible frames while busy. It uses 512 output tokens and temperature 0.7. The optional Lightweight preset restores 640×480/15 FPS capture, JPEG quality 0.8, a 512-pixel longest side and a one-second minimum interval after each answer, with temperature 0. Its controls allow 384/512/768 pixels and alternative time intervals. Both presets allow 1–512 output tokens. Switching presets cancels work and clears timings while preserving prompt/token selections. See [the explicit comparison contract](capture-preset-comparison.md).

The proxy limits requests to 2 MiB, backend responses to 1 MiB, and inference to 120 seconds. A global nonblocking lock rejects concurrent generations with HTTP 429 instead of accumulating work. Closing or stopping a request closes its upstream connection and releases admission. It accepts only an embedded JPEG and one bounded prompt; arbitrary media URLs and cloud backend URLs are not supported. Backend destinations are always `127.0.0.1` with a configurable port.

Visible measurements use the browser's monotonic clock:

- **First visible token:** request submission to first nonempty text delta, including proxy/network time. This is not native GPU prefill latency and does not count hidden reasoning text.
- **Complete answer:** request submission to a correctly terminated `[DONE]` event after visible output. Cancelled, empty, failed and truncated streams never increment the completed counter.
- **Frame age:** time since the browser drew the current request's source into its capture canvas. It includes JPEG encoding, transfer, waiting and generation. It is not camera sensor-to-display latency; uploads use the time sampled in the browser, not file creation time.

The UI reports sampled shared system RAM, CPU and GPU utilization from the Jetson. It does not report a CUDA allocator total, native GPU latency or tokens/sec. Jetson GPU and CPU share memory, so total used/available RAM and process/device allocation are distinct overlapping measurements.

## Run and access

Run the upstream backend on the Jetson at loopback port 8000, then run this from the project directory on the Jetson:

```sh
python3 scripts/serve_ui.py
```

The default URL is `http://127.0.0.1:8090` on that machine. With an SSH tunnel from the viewing computer, use `ssh -N -L 8090:127.0.0.1:8090 USER@JETSON` and open `http://localhost:8090` in the viewing browser. Inference and UI serving still run on the Jetson; the browser supplies the camera. This avoids requiring a locally trusted LAN TLS certificate.

For direct LAN access, provide a certificate trusted by the viewing browser whose SAN covers the Jetson hostname/IP:

```sh
python3 scripts/serve_ui.py --host 0.0.0.0 --port 8090 --cert /path/to/cert.pem --key /path/to/key.pem
```

Then open `https://JETSON-HOSTNAME:8090`. Non-loopback HTTP requires the explicit `--allow-insecure-lan` option. Camera access requires browser permission and a secure context (HTTPS or localhost). Image upload remains a fallback. Camera permissions cannot be granted programmatically on behalf of the user. [Browser camera requirements](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia).

At the user's request, the installed service exposes HTTP at `http://192.168.6.252:8090` and HTTPS at `https://192.168.6.252:8443`. `scripts/enable_lan_ui.sh` creates a task-local certificate and a systemd drop-in. It does not change client trust stores. The first HTTPS visit shows a certificate warning. After the user encountered the camera blocker on HTTP, plaintext LAN `GET /` now returns a temporary, non-cached redirect to the same host's HTTPS port. Host validation prevents malformed redirect authorities. Loopback document requests, assets, API/health endpoints and the HTTPS listener do not redirect. Both listeners run in the same process, share telemetry and bounded admission, and proxy only the loopback backend. This LAN service has no authentication. The original Mac tunnel is optional. See `results/ui-camera-redirect-deployment.json` and `results/browser-ui-camera-redirect.json` for the fix; the older `ui-lan` receipts preserve the original dual-URL behavior.

The page explicitly shows **Backend not ready** when the local engine is absent. A loaded web page or protocol test is not evidence of Jetson inference. Real image inference is recorded separately from synthetic camera transport tests.

## Verification and remaining target checks

Run:

```sh
python3 -m unittest discover -s tests -p test_serve_ui.py -v
node tests/test_sse.js
python3 -m py_compile scripts/serve_ui.py
node --check web/app.js
```

The proxy tests use explicitly labeled local transport fixtures, with no model predictions. They check static serving, health/model forwarding, unavailable-backend status, request bounds, forbidden remote media/history, same-origin enforcement, usage options, unsupported seeds, immediate stream flushing, concurrency rejection and disconnect cancellation. The parser tests split UTF-8 and SSE at every byte boundary and cover CR/LF variants, multiline events and truncated EOF.

A bounded headless Chromium/Playwright contract check also passed through the real browser UI and Python proxy: image upload rendered an incremental text delta before the synthetic fixture was allowed to finish; successful completion incremented the counter; Stop interrupted a second request and caused an upstream disconnect without counting it; a third request completed afterward. No browser JavaScript errors were observed. `results/browser-contract-check.json` records the checks and confirms both temporary listeners stopped. The fixture identified itself as `SYNTHETIC-TRANSPORT-FIXTURE-NO-INFERENCE`; this test performed no model inference, used no GPU and recorded no inference performance measurements.

On target, establish real evidence with changing scenes and the real Cosmos model: record camera scene transitions and changing grounded answers, show a token arriving before `[DONE]`, correlate browser metrics with native timing/usage and sampled RAM, stop mid-generation and verify native admission recovers, and run a sustained single-camera soak without RAM growth or frame backlog. Keep engine build time, cold start and warm requests separate. A finite optimization sweep should change one memory/latency lever at a time and retain only measured improvements that preserve answer quality; exact thresholds belong in the task's agreed benchmark plan.

## License record

TensorRT-Edge-LLM and Live VLM WebUI publish Apache-2.0 source licenses. Cosmos3-Edge publishes OpenMDW 1.1 weights. Retain upstream licenses/notices when distributing copied components and the model license with redistributed weights. Model and software licenses are separate; a repository license does not automatically cover binary SDKs or model weights. [Edge-LLM license](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/v0.10.1/LICENSE), [OpenMDW 1.1](https://openmdw.ai/license/1-1/).


## Live device telemetry

The original UI service samples once per second in one background thread and returns the latest snapshot at `GET /api/metrics`. HTTP requests do not launch processes, import GPU frameworks or wait for a measurement interval. Concurrent browser tabs share the same sample. Sampling and reads never acquire the inference admission lock.

- CPU: aggregate busy-time delta divided by total-time delta from `/proc/stat`, normalized to 0–100% across all cores. Idle plus I/O wait is excluded; guest counters are not counted twice. Counter resets and unavailable reads produce `null` rather than a fabricated zero. [Linux kernel counter definitions](https://docs.kernel.org/filesystems/proc.html#miscellaneous-kernel-statistics-in-proc-stat).
- GPU: read the Orin driver load node at `/sys/devices/platform/bus@0/17000000.gpu/load` (with two known aliases), validate 0–1000 and divide by ten for percent. The node is readable by the existing `jetson` service user on this installed board. This is GPU engine activity, not memory utilization. NVIDIA separately documents the related [GR3D activity metric](https://docs.nvidia.com/jetson/archives/r36.5/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html).
- VRAM / shared RAM: `(MemTotal - MemAvailable)` bytes divided by `MemTotal`, from `/proc/meminfo`; display used/total GiB and percent. Jetson CPU and GPU share physical RAM, and this includes OS and other application pressure. It does not report CUDA allocation or model-only VRAM. It matches the definition used in the earlier performance report. [Linux memory counter definitions](https://docs.kernel.org/filesystems/proc.html#meminfo).

The response includes sample time, monotonic sample age, and `ok`, `partial` or `unavailable` status. Null readings remain unavailable in the browser. Samples older than five seconds are rejected as stale. Telemetry polling has a bounded timeout, does not overlap, and pauses when the page is hidden. It does not replace or influence streaming requests. Successful telemetry polls are omitted from the HTTP access log to avoid continuous journal churn.
