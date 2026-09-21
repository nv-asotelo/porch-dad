# Frozen API streaming soak procedure

`scripts/soak.py` tests the selected running backend on the Orin for at least ten minutes. It cycles through all six unchanged JPEGs, prompts and request settings in `benchmarks/fixtures/jpeg-manifest.json`, with one streaming request in flight. This is an **API streaming soak**, separate from the real browser/camera checks and the frozen semantic quality screen. It does not compare answers with an exact expected string or turn a semantic quality failure into a pass.

Run on the Orin after recording the selected backend's engine precision, model/cache paths, source/native-build pins and runtime settings in a JSON configuration receipt. Use the actual backend PID, not the proxy or launcher shell PID:

```sh
python3 scripts/soak.py \
  --candidate-id selected \
  --pid "$BACKEND_PID" \
  --backend-config /home/jetson/cosmos-edge/results/selected-backend-config.json \
  --output results/raw/soak-selected-01.jsonl \
  --duration 600
```

The default endpoint is the local UI proxy on port 8090. `/health/ready` is polled at most once per second, independently of streaming. Health reads have a three-second wall timeout and a 64 KiB response limit; only status and admission counters are retained. The existing benchmark `LocalSampler` records `/proc` memory and VM counters every 500 ms; optional `--tegrastats /usr/bin/tegrastats` adds its separate observations. Only loopback endpoints are accepted, and the CLI requires an Orin device identity so local memory measurements cannot accidentally be attributed to a remote GPU.

The following policy is fixed in source and written to the first JSONL record **before the first request**:

- At least 600 seconds of the request loop, with all six cases represented. Setup and teardown do not count. Requests are sequential. A final request may drain for at most the configured timeout, which cannot exceed 120 seconds.
- Every request must contain visible streamed text, a successful `stop` finish reason and `[DONE]`. HTTP/backend errors, missing terminal events, interruption or the 64-token `length` limit fail the run; the loop stops after a request failure.
- All sampled available system memory must be at least 512 MiB. Any increase in system swap-in pages, swap-out pages or `oom_kill` fails the run. Missing fields, sampler errors and counter resets cannot establish clean evidence.
- Every health sample must return HTTP 200 and `status=ready`, with valid counters. Active requests and queued requests must each remain at or below one. Missing health evidence or sample gaps exceeding five seconds fail the run; resource coverage is checked similarly.
- The first 60 seconds allow initial allocator settling. Compare median backend RSS from seconds 60–120 with the final 60 seconds. Growth exceeding **the larger of 64 MiB or 5% of the baseline median RSS** fails the run. Report the raw delta and sample counts even below this tolerance. System RAM unavailable uses the same median windows and is reported separately, without adding it to RSS or treating it as dedicated VRAM.
- The backend PID/start time/command/executable and recorded backend configuration/source provenance must remain unchanged. Missing final provenance fails the run.

The raw JSONL contains the policy, exact workload/image hashes, backend configuration and PID identity, every response, sampled resource/health data, and a final summary. A sibling `.summary.json` contains the compact result. Both files use exclusive creation; existing runs are never overwritten. No ten-minute hardware run is implied by the local tests.

RAM/RSS/tegrastats peaks are sampled and overlap. The system OOM counter does not measure a CUDA allocator; CUDA/backend errors reject the run through failed requests, and backend logs may be needed to classify their cause. Passing this bounded screen establishes only the observed API stability and resource constraints for this workload and configuration.
