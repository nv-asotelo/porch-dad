# Closed captions, resource history and latency

The user supplied `2026-09-14 08-58-29.mkv` as a visual reference and requested the Live VLM WebUI caption/sparkline presentation, followed by its Latency and Average metrics. Five frames from the 36.966-second recording show white sentences in a black strip at the bottom of the visible camera area, with colored resource charts below. The recording is a layout reference, not a benchmark for this implementation.

## Caption behavior

Below camera is the initial default. A compact black band sits directly against the preview, with white left-aligned text. It occupies its own space and does not cover image pixels. Above camera uses the same band above the preview. Side window moves that one caption into the adjacent information panel; it stacks below the camera on narrow screens. The user's browser remembers the selected position.

There is one caption element and no growing transcript. The previous caption remains visible while the next frame is being processed. The first arriving token replaces it, and later tokens extend the new answer. Long captions wrap and scroll within a bounded band. Repositioning preserves the live video and active inference. Manual capture, the ten prompt presets and live pause/resume remain available.

The public reference implements an absolute overlay. In the supplied recording, the black space around the visible camera makes the lower overlay appear as a separate band. This interface uses a real band outside the image to meet the user's explicit below-camera request. The information panel contains Latency, Average and Count. Detailed first-token, request-to-stream-end and frame-age measurements remain expandable.

## Resource sparklines

Three charts use the existing Orin sampler, independently of inference: CPU in blue, GPU in green and shared system RAM in purple. Numeric readings and thin utilization bars remain visible. The memory chart represents `MemTotal - MemAvailable` as a percentage of Linux-visible shared RAM. It does not invent a separate dedicated VRAM pool.

The browser retains up to 60 seconds of actual device sample timestamps, with a 120-point safety bound. It does not fabricate a history on page load. Repeated cached samples are deduplicated. Null, invalid, failed, stale and hidden-page periods break the line rather than becoming zeros. Genuine zero readings remain valid. A timestamp rollback starts a new history. Samples expire as the window advances.

Each chart auto-scales from zero to its observed maximum, with a 1% minimum scale. Its displayed range makes this explicit. The translucent fill and line scale for the display's pixel density and resize with the page. Accessible chart descriptions include the observed range and most recent plotted value. These choices preserve the reference's compact line-and-fill appearance while keeping missing data distinct from measured inactivity.

## Latency and Average

Live VLM WebUI starts its timer before PIL JPEG encoding and ends after the complete API answer. Its Average is `total_inference_time / total_inferences`, accumulated across successful requests. The original implementation here timed its Complete answer metric after browser JPEG encoding and had no cumulative average.

The new Latency starts immediately before browser canvas capture and JPEG encoding and ends after a complete successful stream. It includes the UI proxy, transport and full response, rather than reporting time to first token. Average is the arithmetic mean of those unrounded durations. Displayed values use whole milliseconds, as upstream does. Count includes successful manual and automatic requests, including valid token-limit completions. Errors, truncated streams and canceled requests do not enter the average. The latest successful values remain while another request is pending or fails.

The timing population begins on page load and resets when the capture preset changes, alongside the existing completed-request counter. Camera Stop, live pause, caption placement and prompt edits do not reset it. Timing details retain the historical request-to-first-text and request-to-stream-end definitions, both starting after JPEG encoding. Historical benchmark records are unchanged.

This matches the complete-response timing concept and cumulative-average formula. Browser canvas encoding and streaming still differ from upstream's server-side PIL encoding and non-streaming response. It is not an identical transport benchmark or proof of model speedup.

## Public sources and validation

- [Live VLM WebUI caption styling](https://github.com/NVIDIA-AI-IOT/live-vlm-webui/blob/2fd5ba0b334c334d24bf0f9439d8742b243d22be/src/live_vlm_webui/static/index.html#L792) and [sparkline rendering](https://github.com/NVIDIA-AI-IOT/live-vlm-webui/blob/2fd5ba0b334c334d24bf0f9439d8742b243d22be/src/live_vlm_webui/static/index.html#L3738) informed the appearance. The caption docks, bounded history and chart drawing here are task-authored.
- [Upstream full-request timer](https://github.com/NVIDIA-AI-IOT/live-vlm-webui/blob/2fd5ba0b334c334d24bf0f9439d8742b243d22be/src/live_vlm_webui/vlm_service.py#L90) and [average formula](https://github.com/NVIDIA-AI-IOT/live-vlm-webui/blob/2fd5ba0b334c334d24bf0f9439d8742b243d22be/src/live_vlm_webui/vlm_service.py#L233) establish the metric definitions.
- [Metric tests](../tests/test_ui_metrics.js) use a controlled clock to verify encoding inclusion, raw averaging, reset behavior, failure/cancellation exclusion and history gaps. [Streaming tests](../tests/test_sse.js) cover caption movement during live/manual requests. Browser checks cover desktop/mobile layout, real rendered chart pixels and manual/live camera controls.
- [Real Orin browser verification](../results/ui-caption-sparklines-browser.json) passed all caption placements, cumulative timing, three rendered resource charts, responsive layouts and manual/live capture. Its four requests and synthetic camera are functional checks, not a performance or quality comparison. [Deployment verification](../results/ui-caption-sparklines-deployment.json) confirms the served asset hashes and the unchanged selected MLP runtime.
