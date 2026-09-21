# Fixed visual quality screen

The six original 512×512 PNGs in `benchmarks/fixtures/` check colors, simple shapes, counting, left/right, above/below, containment and relative size. The [manifest](../benchmarks/fixtures/manifest.json) contains each exact prompt, scene geometry, 19 required factual statements, image hashes, settings and authorship. The [contact sheet](../benchmarks/fixtures/contact-sheet.png) is an inspection aid; submit individual cases. The fixtures and renderer were authored for this project and use the repository's Apache-2.0 license. They contain no downloaded artwork or model output.

Before comparing an optimization candidate, submit all six images with their manifest prompts, one request at a time, using the same model revision, image bytes and generation settings. Use 64 output tokens, temperature 0, top-p 1 and no seed. Save the actual response for every case. A human reviewer awards one point per fully correct required fact, accepting equivalent wording. Missing, contradicted or hedged required facts receive zero. Pass requires **19/19 facts, six successful nonempty completed responses, and no materially false added visual claim**. Record per-fact scores, false claims, reviewer, candidate, request settings and hashes. Ground-truth statements are an answer key, never observed model results. No candidate has been evaluated by creating these files.

Use the frozen JPEGs in `benchmarks/fixtures/jpeg/` for real benchmark requests through the UI proxy. The [JPEG manifest](../benchmarks/fixtures/jpeg-manifest.json) preserves each original prompt and expected fact, and records both source PNG and derivative JPEG hashes. All six derivatives are 512×512 RGB, quality 90, 4:4:4 chroma sampling, non-progressive JPEG, with optimization disabled and no EXIF or ICC profile. The original PNGs remain untouched. The browser's canvas re-encodes uploads, so send the JPEG directly with the benchmark harness to retain these exact bytes; browser-upload measurements are a different workload.

The harness accepts these JPEGs without modification and includes their SHA-256, MIME type, prompt and generation settings in its workload fingerprint. For example, on the Jetson:

```sh
python3 scripts/benchmark.py run \
  --url http://127.0.0.1:8090/v1/chat/completions \
  --model Cosmos3-Edge \
  --image benchmarks/fixtures/jpeg/01-left-right.jpg \
  --prompt 'Describe the shape and color on the left and on the right. Which shape is left of the other?' \
  --candidate-id baseline-left-right --warmup 5 --requests 30 --max-tokens 64 \
  --sample-local --output results/raw/baseline-left-right.jsonl
```

Use the same case and exact prompt for each candidate comparison. Different cases intentionally have different fingerprints; do not pass them to the optimizer as though they were one fixed workload. Review all six responses against the unchanged 19-fact answer key separately. The harness records per-image timings and does not automatically grade quality or combine a suite into one latency metric.

Regenerate or verify the fixtures without installing packages:

```sh
python3 benchmarks/fixtures/generate.py
python3 benchmarks/fixtures/generate.py --check
```

The renderer uses fixed geometry and the Python standard library. The check compares regenerated files byte-for-byte; the manifest also records raw RGB hashes, allowing a pixel-level comparison if a future zlib version changes PNG compression. Creating the fixtures adds no dependency to the device runtime.

JPEG conversion is a host preparation step using Pillow 12.1.1 (MIT-CMU), with the observed JPEG codec version recorded in the derivative manifest. Its isolated environment lives under the ignored task-local `.qa/` directory; it is not a device runtime dependency:

```sh
python3 -m venv .qa/fixture-codec-venv
.qa/fixture-codec-venv/bin/python -m pip install Pillow==12.1.1
.qa/fixture-codec-venv/bin/python benchmarks/fixtures/convert_jpeg.py
.qa/fixture-codec-venv/bin/python benchmarks/fixtures/convert_jpeg.py --check
```

Freeze the supplied JPEG files for comparisons. A different codec build may produce different compressed bytes; `--check` reports that mismatch rather than silently treating it as the same workload.

Passing this small, deliberately simple screen does **not** demonstrate natural-camera or video understanding, motion/temporal reasoning, robustness, or performance. Representative real scenes and the project's sustained streaming and memory tests remain necessary before marking deployment quality as passed.
