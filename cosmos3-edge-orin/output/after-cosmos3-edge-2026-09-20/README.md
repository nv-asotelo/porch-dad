# After — recorded Cosmos3-Edge observations

Twenty completed-response timings and six full frames from the user-designated “after” recording. This clip follows restoration of the original corrected FP16 runtime after the sampler candidate missed the 10% latency goal; it precedes the new MLP optimization goal. “After” and “opted for accuracy” are the user’s labels, not evidence of a speed or accuracy gain.

[Open the frame gallery](index.html) · [Statistics and all 20 timing rows](statistics.json) · [Frame timestamps and checksums](manifest.json) · [504 parsed observations](metric-observations.jsonl) · [Raw OCR](metrics-ocr.jsonl)

Source: `After - Codex opted for accuracy  - ten percent goal.mov` · 49.196667 s · 3348×2096 · 2169 frames · variable frame rate, time base 1/600. SHA-256: `7f0eb89718adba7317b37b93018243c9778503fd75c6e0cf15486c74a3eeb363`. The original recording was left unchanged. All six PNGs preserve the full decoded source frame, with no crop, rescaling, annotation or text replacement.

## Visible timing statistics

All 20 completions, counters **45–64**, are represented exactly once, with no counter gaps. Counter 44 already existed when recording began. These are rounded UI request timings; the first visible completion can belong to a request that began before the clip.

| UI timing | Minimum | Median | p95 | Maximum | Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| First visible token | 715 ms | **975 ms** | 1463 ms | 1900 ms | 1070.2 ms |
| Complete answer | 1.50 s | **1.98 s** | 2.779 s | 2.95 s | 2.0605 s |

Percentiles use linear interpolation at `(n−1)×p`, with equal weight per completion. Derived decimal places do not add precision to the displayed readings. First visible token measures request to first text; complete answer measures request to stream end. These are not server-only inference times.

OCR covered 492 source frames at nominal 10 Hz. Completed counters 46, 52 and 63 existed too briefly to appear with finished timings in that sample. Twelve exact intervening frames recovered all three; their verified evidence frames are source indices 197 / 4.233333 s, 832 / 19.241667 s and 2076 / 47.266667 s. `statistics.json` retains exact integer PTS and frame indices for every observation.

## Visible resource samples

Across the 492 regular samples: **CPU 0.5–3.5%; GPU 0.0–99.9%; shared RAM 6.38–6.39 of 7.37 GiB**. The UI telemetry updates more slowly than the video sampling, so these are repeated visible readings, not 492 independent measurements. Ranges can miss peaks. Shared RAM is system memory including the OS and applications, not a separate dedicated VRAM pool or model-only allocation. The 12 recovery frames are excluded from these resource ranges.

## Six selected full frames

| Frame / clip offset | First text | Complete answer | Completed counter | CPU / GPU / shared RAM |
| --- | ---: | ---: | ---: | ---: |
| [after-01 · Toy recognition · 00:06.992](01-after-06.992s-toy.png) | 1030 ms | 2.43 s | 47 | 2.3% / 82.5% / 6.39 GiB |
| [after-02 · Action camera recognition · 00:14.000](02-after-14.000s-action-camera.png) | 865 ms | 1.66 s | 50 | 1.3% / 0.0% / 6.39 GiB |
| [after-03 · Computer mouse recognition · 00:30.008](03-after-30.008s-mouse.png) | 950 ms | 1.63 s | 56 | 1.0% / 0.0% / 6.39 GiB |
| [after-04 · Watch recognition · 00:35.142](04-after-35.142s-watch.png) | 961 ms | 1.82 s | 58 | 1.0% / 0.0% / 6.39 GiB |
| [after-05 · Scissors recognition · 00:40.008](05-after-40.008s-scissors.png) | 762 ms | 1.53 s | 60 | 2.5% / 99.8% / 6.39 GiB |
| [after-06 · Handheld console specificity limitation · 00:47.267](06-after-47.267s-handheld.png) | 1180 ms | 1.96 s | 63 | 0.7% / 0.0% / 6.39 GiB |

All selected frames show “Answer complete.” The camera example calls a gray/silver-looking device “white.” The watch family and white strap are visible; its exact brand is not independently verified. The handheld answer says “Game Boy Color,” while the next answer uses the broader “silver Game Boy.” The exact hardware or replica identity is unverified; this illustrates variable specificity, not a scored accuracy result.

## What this recording establishes

The displayed profile is Live VLM WebUI; the camera reports 1280×720 / 30.0 FPS and sent images are 1280×720. The selected viewport does not show max tokens, temperature, top-k or top-p. Do not infer them from the deployment defaults. Skipped capture opportunities are not failed requests, and camera FPS is not inference throughput.

The preview remains live while the answer describes an earlier sampled image. These frames do not recover the exact historical inference inputs. Frame age updates separately from answer timing; preserve the visible 1.94 s frame age and 1.96 s complete-answer value in the console example.

These illustrations and timing aggregates are separate from the earlier fixed 512×512 / 64-token benchmark and from the controlled Live VLM workload used for the 10% search. Changing scenes, output lengths, network timing and UI telemetry prevent a causal before/after speed or accuracy comparison from these clips alone. No generated-token count or error/OOM rate is inferred.

The deployment context is supported by the [FP16 restoration receipt](../../results/latency10/restoration.json), [functional verification](../../results/latency10/restoration-verification.json), and [10% trial comparison](../../results/latency10/comparison.json). Runtime identity is task chronology, not a fact visible in recording pixels. The sampler candidate achieved 0.479099% whole-answer reduction, failed the 10% gate, and was removed before this clip.

## Provenance

The user supplied the recording. Task-authored extraction used FFmpeg/ffprobe for exact source-frame selection and macOS Vision OCR for visible text. The six evidence images retain normal FFmpeg YUV-to-RGB decoding at original resolution; checksums identify the exported PNGs. [extract_frames.py](extract_frames.py) verifies the source hash and reproduces the selected frame indices; PNG byte encoding can vary between FFmpeg versions.
