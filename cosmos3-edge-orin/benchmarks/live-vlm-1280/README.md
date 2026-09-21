# Frozen Live VLM default inference inputs

These three 1280×720 JPEGs come from the camera preview in the user's September 20 screen recording. They are **not** original sensor frames, and they are not guaranteed to be the exact earlier frames used for the answers visible in that recording. Original full screenshots remain in `output/before-cosmos3-edge-2026-09-20`.

The extraction crops only the preview pixels (3348×2096 source: x294, y459, width1348, height758), resizes to 1280×720, and encodes through Chromium canvas JPEG quality0.75. The manifest pins source timestamps, source PNG hashes, output hashes and browser version. No model-output text or UI controls enter these inputs.

The new latency comparison sends these exact same bytes to every candidate, with the Live VLM prompt, 512-token ceiling, temperature0.7 and omitted top-p/top-k fields. It measures inference through the Orin-local UI proxy; browser camera capture intervals, browser encoding and Wi-Fi time are outside that measurement. A crop-derived input may yield different answers from the historical recording. Historical recording timings are not the new baseline denominator.

The three visible object categories—action camera, pen and scissors—are a small controlled comparison. They do not constitute a representative quality benchmark. Baseline object recognition can coexist with unsupported scene or intention claims. The acceptance gate preserves identical answers and token counts; it does not certify the accuracy of those answers.
