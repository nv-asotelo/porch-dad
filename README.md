# One-shot: Cosmos3-Edge + Live VLM WebUI on Jetson Orin Nano Super 8 GB

> **This branch is frozen.** It is a fixed reference point linked from the report and the
> presentation. Active work continues on
> [`main`](https://github.com/nv-asotelo/porch-dad/tree/main), which also carries the NVR layer
> (Frigate, ring-mqtt, alert policy) that this branch deliberately omits.

Everything needed to take a freshly flashed **Jetson Orin Nano Super 8 GB** to a running
**Live VLM WebUI** serving **INT4-quantized Cosmos3-Edge** — in one pass, executable by an agent.

## Run it

Point a coding agent at this repository and tell it to execute **[`AGENTS.md`](AGENTS.md)**. That
file is the runbook: eight phases, each with a gate that has a stated pass condition.

```bash
bash oneshot/preflight.sh   # Phase 0 — refuses to start on the wrong board or stack
# ... phases 1-7 per AGENTS.md ...
bash oneshot/verify.sh      # Phase 8 — proves it works, and that the measurement is honest
```

## What you should get

| | |
|---|---|
| Best caption | ~245 ms |
| Typical (uncontended) | ~253 ms |
| Latency model | `elapsed_ms ≈ 200 + 13.5 × generated_tokens` |
| Steady-state decode | ~13.5 ms/token (~74 tok/s) |
| System RAM | ~4.8 GB of 7.4 GB |
| CPU | ~15%, system-wide across 6 cores (GPU-bound) |

## Why the gates matter

Most failures in this stack are **silent**. An export succeeds and the engine build fails. A build
succeeds and the load OOMs. A load succeeds and the captions are quietly wrong. Every gate in
`AGENTS.md` exists because that specific failure actually happened during the original deployment —
in particular:

- **An all-zero quantized weight file exports and builds perfectly cleanly.** It only shows up as
  garbage captions at the very end. Phase 4 and Phase 5 both gate on it.
- **`--maxBatchSize 4 --maxKVCacheCapacity 4096` builds fine and then OOMs at engine load** on 8 GB.
- **`EDGELLM_PLUGIN_PATH` defaults to a relative path**, and the resulting error names the wrong
  cause.
- **The encoder embedding cache** makes a repeated-image benchmark look ~2× faster than the model
  is. `oneshot/verify.sh` varies its input specifically to defeat this.

## What is here

| Path | |
|---|---|
| [`AGENTS.md`](AGENTS.md) | the runbook — **start here** |
| [`oneshot/`](oneshot) | Phase 0 and Phase 8 gate scripts |
| [`scripts/rtn_int4_quantize.py`](scripts/rtn_int4_quantize.py) | the INT4 W4A16 quantizer (calibration-free RTN + MSE-optimal clipping) |
| [`serve/cosmos3_shim.py`](serve/cosmos3_shim.py) | OpenAI-compatible shim holding one resident `LLMRuntime` |
| [`systemd/`](systemd) | units for the shim and the WebUI |
| [`deploy/`](deploy) | the same path as prose, with far more detail than the runbook |
| [`docs/`](docs) | methodology, negative results, the ecosystem survey |
| [`bench/`](bench) | the benchmark harness and the recorded quiesced run |
| [`ACKNOWLEDGEMENTS.md`](ACKNOWLEDGEMENTS.md) | the nine NVIDIA projects this is assembled from |

## The point

Not one optimization here required new NVIDIA technology. The resident runtime, the CUDA-graph
decode, the INT4 matmul kernel, the on-disk quantization format contract, the runtime-tunable image
token budget, the power modes — all of it already shipped. What did not exist was the
**intersection**: no published path takes this model, on this board, through these projects, to a
measured number. This branch is that path, in one shot.

## Honest limits

INT4 output quality was **spot-checked at greedy decoding, not benchmarked on a task suite**. The
22.5× headline elsewhere in this repo is against a naive process-per-request baseline; against a
competently configured FP16 resident baseline the honest figure is **3.38× decode + 2.32 GB RAM**.
The exact `visual_build` argument values from the original deployment were never recorded — the
runbook says so and builds with release defaults rather than inventing them.

Licensed Apache-2.0. Cosmos3-Edge weights are distributed by NVIDIA under their own terms and are
not redistributed here. Personal engineering write-up of one deployment — not an official NVIDIA
product, release, or support commitment.
