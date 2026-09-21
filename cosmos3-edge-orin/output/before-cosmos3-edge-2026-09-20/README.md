# Before — Cosmos3-Edge on Orin

Six original-resolution frames from the user-designated “before / unoptimized” Cosmos3-Edge session on Orin, using the Live VLM WebUI default capture preset. These are curated visual observations, not benchmark aggregates.

[Open the frame gallery](index.html) · [Contact sheet](contact-sheet.jpg) · [Exact timestamps and checksums](manifest.json)

Source: `Codex recording unoptimized Sep 20 Cosmos3-Edge.mov` · 34.847 s · 3348×2096 · variable frame rate. Source SHA-256: `6f4b66d3572edf2faffa617baa77b1d448635354f1874d4960e2c7c87238b438`. The original recording was left unchanged.

All six PNGs preserve the full decoded source frame with no crop, rescaling or annotation. Only the contact sheet uses resized, labeled previews.

| Frame / clip offset | First visible token | Complete answer | Frame age | Sent / skipped / completed |
| --- | ---: | ---: | ---: | ---: |
| [B01 · Scene overview · 00:00:00.950](01-before-00.950s-scene-overview.png) | 559 ms | 1.26 s | 1.67 s | 61 / 71 / 61 |
| [B02 · Action camera · 00:00:05.808](02-before-05.808s-action-camera.png) | 812 ms | 1.44 s | 1.55 s | 63 / 74 / 63 |
| [B03 · Pen demonstration · 00:00:09.842](03-before-09.842s-pen.png) | 612 ms | 1.43 s | 1.56 s | 65 / 76 / 65 |
| [B04 · Toy demonstration · 00:00:21.950](04-before-21.950s-toy.png) | 683 ms | 1.59 s | 1.65 s | 71 / 82 / 71 |
| [B05 · Battery demonstration · 00:00:29.850](05-before-29.850s-battery.png) | 550 ms | 1.43 s | 1.54 s | 75 / 86 / 75 |
| [B06 · Scissors demonstration · 00:00:33.947](06-before-33.947s-scissors.png) | 584 ms | 1.30 s | 1.64 s | 77 / 88 / 77 |

All selected frames show “Answer complete.” Counters include activity before the recording began. The pen answer includes an inferred action (“as if to scratch it”); retain this as a model limitation, not a verified action. The moving toy and battery previews are slightly blurred; their original pixels are preserved.

## Presentation caption

> Before — Cosmos3-Edge on Orin, Live VLM WebUI default capture preset. User-recorded 1280×720 / 30 FPS camera session; full-size samples every 30 frames, temperature 0.7. On-screen timings are individual request observations.

## Comparison boundaries

The selected preset, camera/source dimensions, prompt, sampling description and temperature are visible. CPU/GPU/shared-memory readings and the max-token control are outside the viewport. Do not use these frames to establish VRAM use or to claim the active token limit; the documented 512-token default alone does not prove it.

“Before / unoptimized” preserves the user’s label for this recording. The existing corrected FP16 backend and memory limits were already deployed. This is the comparison capture profile, not a claim of an untouched upstream backend. See the [selected configuration](../../deployment/selected-config.json) and [settings comparison contract](../../research/capture-preset-comparison.md). Browser capture, JPEG encoding and streaming differ from upstream Live VLM WebUI.

Do not combine these observations with the earlier fixed 512×512 / 64-token / temperature-0 benchmark. The preview is live while the answer describes an earlier sampled frame. Frame age is measured since browser capture. A visual mismatch alone is not a grounding-error measurement.

## Re-extract

Run `python3 extract_frames.py /path/to/original.mov /path/to/new-output-directory`. The script verifies the source hash and selects the exact six decoded frame indices using FFmpeg. Original timestamps and indices are preserved in `manifest.json`; the display timecodes round to milliseconds. PNG file bytes can vary between encoder versions.
