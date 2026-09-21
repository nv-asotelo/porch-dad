# Live VLM WebUI comparison preset

The user requested a toggle for comparison with Live VLM WebUI and asked to make that the default. The default preset matches the pinned public UI's capture and generation settings. The other preset preserves the previous lightweight behavior. This is a settings comparison on the same resident TensorRT model, not an implementation or full-stack performance equivalence claim.

| Setting | Live VLM WebUI preset (default) | Lightweight preset |
| --- | --- | --- |
| Camera request | Ideal 1280×720, no explicit FPS constraint | Ideal 640×480, ideal 15 FPS / max 30 FPS |
| Inference eligibility | Every 30 video frames | After the last answer and a minimum start-to-start time of 1 second |
| Busy behavior | Skip eligible frames, no queue | Wait for completion before capturing |
| Sent image dimensions | Preserve actual source dimensions | Limit longest side to 512 by default |
| Browser JPEG quality | 0.75 nominal | 0.8 |
| Temperature | 0.7 | 0 (greedy) |
| top_p / top_k | Omitted, so this backend resolves 0.9 / 50 | top_p=1; greedy backend resolves top_k=1 |
| Initial prompt | Describe what you see in this image in one sentence. | Same prompt, preserved across toggles |
| Initial output ceiling | 512 tokens | Same ceiling, preserved across toggles |

The preset is explicit and defaults to Live VLM on every new page load. Switching stops camera tracks, aborts the active request, cancels frame callbacks, clears output/timings/counters and prevents late stream cleanup from overwriting the new mode. Prompt and token edits are retained across the toggle so the source workload can be held constant. The camera label shows actual dimensions/FPS from the negotiated track, and the capture label shows actual JPEG dimensions. Preset values are requests, not a promise that every physical camera supplies 1280×720.

## Equivalence boundaries

- Upstream counts frames after WebRTC reception. This UI counts browser-presented video frames with `requestVideoFrameCallback` and its monotonic frame counter. There is no one-second approximation. Delayed callbacks coalesce crossed sample boundaries into at most one fresh frame; they never replay a backlog. Browsers without frame callbacks must use Lightweight or a supported browser.
- Upstream encodes the WebRTC-decoded image with Pillow's JPEG defaults. Browser canvas quality 0.75 is only nominally similar; encoders and source compression are different. JPEG bytes are not claimed identical.
- This UI retains token streaming, while upstream requests a non-streaming answer. Compare complete-answer latency with the same source, model, prompt and token limit; first-token timings are not an upstream equivalent. Our latency starts after browser JPEG capture. Upstream latency includes Pillow encoding. CPU/network pipeline costs differ.
- The model still applies its own image preprocessing. From the pinned resize formula and the selected 512 visual-token per-image budget, 1280×720 predicts 960×512 / 480 visual tokens. That is more input work than the old 512-pixel frame. Larger dimensions alone do not establish a quality or memory improvement.
- All existing four-trial optimization, quality grading and 64-token benchmark records stay unchanged. The new 0.7-temperature setting is stochastic, so exact prior outputs and timings should not be expected. No new optimization search is implied by this preset.

## Runtime bounds

Frames with an axis below 256 pixels are padded to preserve the whole source for the existing model integration. Full-size mode rejects axes over 4096, aspect ratios over 8:1 after padding, and encoded data URLs over 1,900,000 characters. The aspect check avoids very thin post-resize grids outside the validated position repair. These checks ask the user to choose a smaller source or Lightweight instead of silently changing comparison dimensions. The server's 2 MiB request limit, 120-second request deadline and 1,024-token input limit remain. Very long prompts may exceed the remaining input budget with larger images.

## Prompt menu, caption placement and manual inference

The 2026-09-21 UI update copies the ten Quick Presets labels and prompt texts exactly from the same public Live VLM WebUI revision, which was also its verified public HEAD on that date. They are Scene Description, Object Detection, Activity Recognition, Safety Monitoring, Emotion Detection, Accessibility, OCR / Text Reading, Yes/No Question, Robot Navigation (Simple) and Robot Navigation (ROS). Selecting one fills the editable prompt. Editing it to a different value selects Custom prompt. The initial Scene Description prompt and 512-token ceiling remain unchanged. The robot examples generate text only; this interface does not execute commands or control a robot.

The explicit caption selector offers Side window, Above camera and Below camera. Below camera is the first-visit default. The browser remembers a valid selection, falling back to Below if storage is blocked or invalid. Side window stacks below the camera on screens at most 800 CSS pixels wide. The implementation moves only the existing output panel, preserving the video element, active request, streamed text, metrics and reading order. Upstream has separate vertical-order and on-camera-overlay controls, plus automatic side-by-side layout on wide screens. This task's three-position selector implements the requested placement without covering the video.

Run inference submits exactly one current camera frame or uploaded image. Live streaming defaults On and controls automatic camera requests. Off preserves camera tracks, prevents further automatic requests and aborts any current automatic request, retaining partial text. An explicitly requested manual inference survives toggling live streaming off. Switching capture presets still stops the camera and active request, but preserves the live on/off choice. The existing one-request admission guard covers both manual and automatic submissions. Turning live back on uses the existing capture cadence; it does not replay skipped frames.

The prompt labels/text remain upstream Apache-2.0 material. Caption positioning, prompt-menu integration and manual/live request controls are task-authored. Source: [Quick Presets and prompt editor](https://github.com/NVIDIA-AI-IOT/live-vlm-webui/blob/2fd5ba0b334c334d24bf0f9439d8742b243d22be/src/live_vlm_webui/static/index.html#L2349), [layout controls](https://github.com/NVIDIA-AI-IOT/live-vlm-webui/blob/2fd5ba0b334c334d24bf0f9439d8742b243d22be/src/live_vlm_webui/static/index.html#L2053) and [Apache-2.0 license](https://github.com/NVIDIA-AI-IOT/live-vlm-webui/blob/2fd5ba0b334c334d24bf0f9439d8742b243d22be/LICENSE).

[Direct Orin browser verification](../results/ui-controls-browser.json) passed actual upload inference, incremental text, all caption positions, manual camera capture, live pause/resume, prompt submission and mobile layout. Camera frames were Chromium's synthetic feed. [Deployment verification](../results/ui-controls-deployment.json) checks the served asset hashes and unchanged selected runtime. The existing SSE tests now also cover manual/automatic request ownership and caption movement during a stream. These are functional checks, not a new latency or accuracy experiment.

## Original comparison sources and verification

Pinned Live VLM WebUI revision: `2fd5ba0b334c334d24bf0f9439d8742b243d22be`.

- [Camera constraints](https://github.com/NVIDIA-AI-IOT/live-vlm-webui/blob/2fd5ba0b334c334d24bf0f9439d8742b243d22be/src/live_vlm_webui/static/index.html#L4253).
- [Every-30-frame sampling](https://github.com/NVIDIA-AI-IOT/live-vlm-webui/blob/2fd5ba0b334c334d24bf0f9439d8742b243d22be/src/live_vlm_webui/video_processor.py#L135).
- [JPEG, prompt, token limit, temperature, and busy skip](https://github.com/NVIDIA-AI-IOT/live-vlm-webui/blob/2fd5ba0b334c334d24bf0f9439d8742b243d22be/src/live_vlm_webui/vlm_service.py).
- [TensorRT request defaults](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/experimental/server/api/protocol.py#L64) and [image resize](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/cpp/multimodal/common/imageUtils.cpp#L178).

Verification uses `tests/test_capture_presets.js`, the existing SSE/cancellation tests, proxy tests, [deployment receipt](../results/ui-presets-deployment.json), and [direct browser checks](../results/browser-ui-presets.json). The browser's synthetic camera exercises real Orin inference and lifecycle behavior, not physical-camera hardware or natural-video accuracy.
