# 05 — Serving Cosmos3-Edge and wiring up Live VLM WebUI

At this point you have two TensorRT engines on the device (the INT4 text tower and the FP16
vision tower) sitting in an engine directory. This step turns them into a service: a resident
process that holds one `LLMRuntime` for its whole lifetime and exposes an OpenAI-compatible
`/v1/chat/completions` endpoint, plus a browser front end that streams a webcam into it.

Everything below describes the artifacts in this repo as they were actually deployed:

- [`../serve/cosmos3_shim.py`](../serve/cosmos3_shim.py)
- [`../systemd/cosmos3-edge-shim.service`](../systemd/cosmos3-edge-shim.service)
- [`../systemd/live-vlm-webui.service`](../systemd/live-vlm-webui.service)

Paths in the shim and the units are the deployment's own absolute paths (`/home/orin/...`,
`/opt/tensorrt-edgellm/...`). If your layout differs, change them in one place each — they are
plain constants at the top of the shim and plain strings in the units.

Set the host once; every `ssh` command below uses it:

```bash
export JETSON_HOST="${JETSON_HOST:-orin@jetson.local}"
```

---

## 1. Why a resident runtime is the whole ballgame

The first working version of this deployment shelled out to the `llm_inference` CLI binary once
per request. That is the obvious thing to do, it works, and it is catastrophically slow: every
request paid TensorRT engine deserialization from scratch, roughly **6–8 s**, before a single
token of real work happened.

| Configuration | Mean end-to-end latency per request |
|---|---|
| Baseline: `llm_inference` CLI spawned per request | 13.91 s |
| Resident runtime + CUDA graph + MAXN_SUPER + `jetson_clocks` | 2.07 s |

That is **−85.1%**, the single largest win in the campaign, and it required no change to the
model at all. Be honest about what is bundled in that number: the resident runtime is the bulk of
it, but the same round also enabled `capture_decoding_cuda_graph()` and the MAXN_SUPER power mode
with `jetson_clocks` (GPU 306 → 1020 MHz, CPU → 1728 MHz, EMC 2133 → 3199 MHz). The rounds were
not separated further, so treat −85.1% as the combined figure, not as the CLI-elimination figure
alone.

The consequence for everything downstream: once startup cost is paid exactly once, per-request
latency is dominated by real work (ViT encode, prefill, decode), which is what makes the
regression methodology in §3 meaningful. Against a per-request-CLI baseline every measurement is
just measuring `dlopen` and deserialization.

---

## 2. What the shim actually does

### 2.1 Finding the runtime bindings

Before importing anything, the shim wires up the C++ side:

```python
os.environ.setdefault(
    "EDGELLM_PLUGIN_PATH", "/home/orin/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so"
)
sys.path.insert(0, "/home/orin/TensorRT-Edge-LLM/build/pybind")

import _edgellm_runtime as rt
```

`_edgellm_runtime` is the pybind11 module built alongside TensorRT-Edge-LLM; it is not pip
installed, hence the explicit `sys.path` entry. `EDGELLM_PLUGIN_PATH` points at the custom plugin
library — without it, engine load fails with TensorRT's `"Plugin not found, are the plugin name,
version, and namespace correct?"`. Because it is set with `setdefault`, an
`Environment=EDGELLM_PLUGIN_PATH=...` line in the systemd unit (or an exported value in your
shell) wins over the hard-coded default; you do not have to edit the script to relocate the
plugin.

### 2.2 Runtime construction and warm-up

All of the expensive work happens once, in a FastAPI `startup` hook:

```python
_runtime = rt.LLMRuntime(ENGINE_DIR, ENGINE_DIR, {}, CHECKPOINT_DIR, rt.ContextCacheConfig())
```

The five positional arguments, as used here:

| Position | Value in this deployment | Meaning |
|---|---|---|
| 1 | `/opt/tensorrt-edgellm/models/default` | engine directory |
| 2 | `/opt/tensorrt-edgellm/models/default` | second engine-directory argument — the same path is passed twice because the LLM engine and the `visual/` subtree live under one directory here |
| 3 | `{}` | empty options dict; no overrides are passed |
| 4 | `/home/orin/tensorrt-edgellm-workspace/Cosmos3-Edge/onnx/reasoning/llm` | checkpoint directory (tokenizer/config assets that accompany the exported ONNX) |
| 5 | `rt.ContextCacheConfig()` | default context-cache configuration |

Then, still at startup:

```python
_runtime.capture_decoding_cuda_graph()
```

This captures the per-token decode step as a CUDA graph, so each subsequent token replays one
graph launch instead of re-issuing the full kernel sequence. On a single-stream, small-batch
workload the launch overhead is a real fraction of a 13 ms token, and capture is a one-time cost
at startup.

Finally the shim issues a throwaway 4-token `"hi"` request so the first *real* request does not
pay first-call costs (allocator growth, lazy initialization). The warm-up is wrapped in
`try/except` and is best-effort: if it raises, the shim logs `[shim] warm-up skipped: ...` and
still comes up.

Each phase prints its own timing, so the journal tells you what startup cost you are paying:

```
[shim] LLMRuntime constructed in 6.28s
[shim] CUDA graphs captured in ...s
[shim] warm-up inference in ...s
[shim] ready
```

`LLMRuntime constructed in 6.28 s` is the measured INT4 figure; the FP16 engine took 7.79 s. The
CUDA-graph and warm-up durations are printed by the shim but were not recorded as results in this
project, so no number is quoted for them here.

### 2.3 The OpenAI-compatible surface

Two endpoints, and only two:

| Method | Path | Behaviour |
|---|---|---|
| `GET` | `/v1/models` | returns a one-element list containing `nvidia/Cosmos3-Edge` |
| `POST` | `/v1/chat/completions` | non-streaming chat completion |

`_build_request()` translates OpenAI-style `messages` into runtime objects:

- a string `content` becomes `rt.Message(role, [rt.MessageContent("text", content)])`
- a list `content` is walked part by part; `{"type": "text"}` becomes a text part
- `{"type": "image_url"}` is matched against `^data:(mime);base64,(data)$`. A data URL is
  base64-decoded and loaded with `rt.load_image_from_bytes()`; anything else is passed straight to
  `rt.load_image_from_path()`, i.e. a non-data URL is treated as a **local filesystem path**, not
  fetched over HTTP. Decoded images are collected into `inner.image_buffers`.

Sampling parameters map to `max_generate_length`, `temperature`, `top_p`, `top_k`, with defaults
`max_tokens=256`, `temperature=0.7`, `top_p=0.95`, `top_k=50`. The explicit `top_k` assignment is
itself a bug fix: in an earlier revision `req.top_k` was never set, so it was zero-initialized
rather than carrying a validated 50.

The response is a standard `chat.completion` object. Two notes about it:

- **`usage` is populated.** The response returns real `prompt_tokens` / `completion_tokens` /
  `total_tokens`, computed from the same counts the `[perf]` line logs. Be aware of the provenance
  of the numbers in this repo, though: they were gathered with an earlier snapshot of the shim that
  returned `"usage": None` and logged those counts to the journal only, so the server-side `[perf]`
  log line in §3 — not the HTTP response — is what every measurement here was read from.
- **Non-streaming only.** There is no `stream: true` support and no `/v1/completions`.

Errors are mapped explicitly: a request that fails translation returns HTTP 400 with
`bad request: ...`; a failure inside `handle_request` returns HTTP 502 with `inference failed:
...`. `finish_reason` is `"length"` when the runtime reports `rt.FinishReason.LENGTH`, otherwise
`"stop"`.

### 2.4 Concurrency: deliberately single-flight

```python
_lock = asyncio.Lock()
_pool = ThreadPoolExecutor(max_workers=1)
...
async with _lock:
    resp = await loop.run_in_executor(_pool, _runtime.handle_request, req)
```

One lock, one worker thread. Inference runs off the event loop so the server stays responsive, but
exactly one request is in the runtime at a time. This matches the runtime's own behaviour (it
serializes requests anyway) and keeps peak memory bounded on an 8 GB board, where the engine was
built with `--maxBatchSize 1 --maxKVCacheCapacity 2048`.

The measurement consequence is important and is easy to get wrong: **if you run a benchmark probe
while the WebUI is streaming, your probe waits behind the WebUI's requests and reports inflated
latencies.** During this project a "context cache regression (4.98 s)" was reported and then
retracted for exactly this reason — it was lock contention, not the cache. Under live traffic,
minimums are the honest statistic.

---

## 3. The `[perf]` instrumentation line

After every successful completion the shim prints one line to stdout (and therefore to the
journal):

```
[perf] elapsed_ms=582 prompt_tok=330 gen_tok=22 ms_per_tok=26.5
```

That line is **illustrative**, not a single captured log line: it is assembled from measured
values — `prompt_tok` = 330 is the post-change prompt length from §5, and 582 ms is the measured
post-change mean at `gen_tok` = 22 in the matched-`gen_tok` table in §5.

The fields are built from:

```python
_elapsed_ms = (time.time() - _t_start) * 1000.0
_gen = len(resp.output_ids[0]) if resp.output_ids else 0
_prompt = resp.prompt_token_counts[0] if resp.prompt_token_counts else 0
```

The token counts are computed *outside* the `try` because the OpenAI `usage` field depends on
them; only the timing arithmetic and the `print` are guarded, printing `[perf] instrumentation error: ...` rather
than failing the request — telemetry must never break serving.

Know what `elapsed_ms` covers: `_t_start` is taken at the top of the handler, *before*
`await request.json()`, so it includes JSON parsing, base64 decode, image load, and any wait on
`_lock`. It is server-side wall clock for the request, not pure model time.

**Why `prompt_tok` and `gen_tok` are logged is the point of the whole file.** Response length
varies per request, so raw mean latency is not comparable between two configurations. With
`gen_tok` recorded you can fit

```
elapsed_ms = fixed_ms + marginal_ms_per_token × gen_tok
```

which separates the two costs that different optimizations attack:

| Term | What it is | Which round attacks it |
|---|---|---|
| `fixed` | image preprocessing + ViT encode + prefill | image-token budget (§5) |
| `marginal` | per-token decode | INT4 weight quantization |

The final fit for this deployment: **marginal 13.13 ms/token, fixed 289 ms, R² = 0.999,
n = 699** live requests.

`prompt_tok` earns its place separately: it is how the image-token cap was caught. The WebUI was
pinned at `prompt_tok` = 510/511 — i.e. saturating the 512-token image cap on every frame — and
after the change in §5 it dropped to **330**. Without that field the cap would have been invisible.

Two scripts consume these lines over SSH, both reading `JETSON_HOST` (default `orin@jetson.local`):

```bash
# fit fixed + marginal over a time window
python3 scripts/collect_perf.py "30 min ago"

# matched-gen_tok A/B between two windows: A_since A_until B_since
python3 scripts/compare_perf.py "60 min ago" "30 min ago" "30 min ago"
```

See [`../scripts/collect_perf.py`](../scripts/collect_perf.py) and
[`../scripts/compare_perf.py`](../scripts/compare_perf.py). When the client emits constant-length
replies the regression degenerates (no `gen_tok` spread, R² collapses) — fall back to matching on
`gen_tok`, which is the stronger comparison anyway. A window that spans a config change mixes two
fixed-cost regimes and will also show a low R²; that is expected, not a fault.

---

## 4. Installing the systemd units

Install the shim script where the unit expects it, then the units themselves. The deployed script
name (`cosmos3_shim_v1.py`) differs from the repo filename; keep them consistent or edit
`ExecStart`.

```bash
scp serve/cosmos3_shim.py "$JETSON_HOST":/tmp/cosmos3_shim_v1.py
scp systemd/cosmos3-edge-shim.service systemd/live-vlm-webui.service "$JETSON_HOST":/tmp/

ssh "$JETSON_HOST" 'sudo install -m 0644 /tmp/cosmos3_shim_v1.py /opt/tensorrt-edgellm/cosmos3_shim_v1.py \
  && sudo install -m 0644 /tmp/cosmos3-edge-shim.service /tmp/live-vlm-webui.service /etc/systemd/system/ \
  && sudo systemctl daemon-reload \
  && sudo systemctl enable --now cosmos3-edge-shim.service live-vlm-webui.service'
```

Watch it come up — the shim is not ready until it prints `[shim] ready`:

```bash
ssh "$JETSON_HOST" 'journalctl -u cosmos3-edge-shim.service -f'
```

### What each directive is for

**`cosmos3-edge-shim.service`**

| Directive | Why |
|---|---|
| `After=network.target cosmos3-edge-default.service` / `Wants=...` | ordering + weak dependency on the unit that provisions the `models/default` engine directory. That unit is **not part of this repo**; if you have no equivalent, delete both references rather than leaving a dangling name. `Wants=` (not `Requires=`) means the shim still starts if it is absent. |
| `User=orin` | runs unprivileged as the owner of the engine and venv paths; nothing here needs root |
| `WorkingDirectory=/home/orin/TensorRT-Edge-LLM` | relative lookups (notably the `build/` tree holding the pybind module and the plugin `.so`) resolve from the source checkout |
| `Environment=PATH=/home/orin/TensorRT-Edge-LLM/.venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin` | systemd starts services with a minimal environment and **does not read your login shell profile**, so `PATH` must be stated explicitly. The venv's `bin` comes first so any subprocess resolves the venv interpreter and console scripts; `/usr/local/cuda/bin` is added so CUDA tooling is on `PATH`. The rest is the stock system `PATH`. |
| `ExecStart=/home/orin/TensorRT-Edge-LLM/.venv/bin/python /opt/tensorrt-edgellm/cosmos3_shim_v1.py` | the venv interpreter is named by absolute path — the unit never depends on `PATH` resolution for the thing it is actually launching |
| `Restart=on-failure` / `RestartSec=2` | a crash (OOM, engine load failure) is retried after 2 s; a clean exit is not restarted |

Note what is *not* here: there is no `Environment=EDGELLM_PLUGIN_PATH=...`, because the shim sets
it itself with `setdefault` (§2.1). Adding one is the supported way to override the plugin
location without touching the script.

**`live-vlm-webui.service`**

| Directive | Why |
|---|---|
| `After=network.target cosmos3-edge-shim.service` / `Wants=cosmos3-edge-shim.service` | start the backend first. This is ordering only — it does not wait for `[shim] ready`, so the WebUI can come up while the runtime is still deserializing. Early requests fail until the shim finishes startup. |
| `Environment=PATH=/home/orin/TensorRT-Edge-LLM/.venv/bin:...` | same reason as above: the venv `bin` must precede the system path so the `live-vlm-webui` console script and its interpreter come from the venv. This unit's `PATH` deliberately omits `/usr/local/cuda/bin` — the WebUI process does no GPU work; it is a web front end talking HTTP to the shim. |
| `ExecStart=... live-vlm-webui --host 0.0.0.0 --port 8090 --model nvidia/Cosmos3-Edge --api-base http://localhost:8000/v1 --api-key EMPTY` | serves the UI on 8090 and points it at the shim on loopback 8000. `--model` must match the shim's advertised id. |
| `Restart=on-failure` / `RestartSec=3` | same policy, slightly longer backoff |

**Security, stated plainly:** the shim has **no authentication of any kind** and binds `0.0.0.0`,
and the WebUI binds `0.0.0.0` too. `--api-key EMPTY` is a placeholder to satisfy the client's
OpenAI plumbing; the shim never inspects it. Run this on a trusted network, or put it behind a
reverse proxy / firewall rule. Do not expose port 8000 or 8090 to an untrusted network.

---

## 5. Tuning the image-token budget

This is the highest-leverage knob in the whole deployment for a camera workload, and it costs
nothing to try.

### Where the knob lives and why no rebuild is needed

`max_image_tokens_per_image` lives in the engine's `visual/config.json`, under `builder_config`.
It is read **at runtime**, not baked into the engine: it is consumed by `smartResize` in
`cpp/multimodal/common/imageUtils.cpp:145`, where

```
maxPixels = max_image_tokens_per_image × 32²
```

(each LLM image token corresponds to a 32×32 px region: patch size 16 × spatial merge 2). The
input image is resized down until it fits that pixel budget, and the token count follows.

Practical consequences:

- **No engine rebuild.** You edit a JSON file.
- **Reversible in about 4 seconds** — the time to restart the shim.
- The runtime must re-read the file, so a shim restart is required; editing it alone changes
  nothing for the running process.

### Measured latency vs image tokens

Each probe used a **uniquely generated image** per iteration. This is mandatory: the runtime caches
vision-encoder embeddings keyed on raw pixel bytes, so looping one image skips the entire ViT
(~250 ms of real work) and reports a fantasy number.

| Probe resolution | Image tokens | Best latency |
|---|---|---|
| 320×240 | 80 | 90 ms |
| 448×336 | 140 | 152 ms |
| 640×480 | 300 | 288 ms |
| 896×672 | 512 (capped) | 499 ms |
| 1280×960 | 512 (capped) | 512 ms |

Cleanly linear up to the cap, then flat — the last two rows are the same 512 tokens, which is the
direct evidence that the WebUI was saturating the cap on every frame.

### Measured effect of 512 → 320, matched on `gen_tok`

| `gen_tok` | n(before) | n(after) | before | after | delta | improvement |
|---|---|---|---|---|---|---|
| 17 | 13 | 3 | 672 ms | 516 ms | −156 ms | 23.2% |
| 18 | 11 | 2 | 687 ms | 528 ms | −158 ms | 23.1% |
| 20 | 60 | 4 | 714 ms | 556 ms | −158 ms | 22.2% |
| 21 | 15 | 7 | 727 ms | 567 ms | −160 ms | 22.0% |
| 22 | 70 | 5 | 738 ms | 582 ms | −156 ms | 21.1% |
| 23 | 9 | 12 | 756 ms | 594 ms | −162 ms | 21.5% |
| 27 | 12 | 2 | 808 ms | 646 ms | −161 ms | 19.9% |
| 34 | 31 | 8 | 899 ms | 740 ms | −159 ms | 17.7% |

**Weighted improvement: 20.8%.** The delta is a constant ≈ −158 ms in *every* bucket, which is the
signature of a fixed-cost reduction rather than a change in decode rate — precisely what you expect
if the change touched ViT/prefill and left decode alone. Note the small `n` in several "after"
buckets; the per-bucket deltas are consistent enough to carry the conclusion, but they are not
large samples.

### The quality A/B — this is a tradeoff, not a free win

Compared at temperature 0 (greedy), same prompts, 320 vs 512:

| Input type | Result at 320 vs 512 |
|---|---|
| Natural camera scene (the actual Live VLM use case) | Output **byte-identical** across three prompts, including one specifically probing fine detail |
| Dense text screenshot | **Degraded.** At 512 the model quoted on-screen annotations verbatim ("Large left lateral deviation", "Markers appear to be lagging behind in both video panel and 3D panel"). At 320 it drifted vaguer and partly hallucinated — "flight simulation or navigation system", generic X/Y/Z axes. |

**Recommendation: 320 for live camera work, 512 for document/screenshot reading.** If your workload
is reading text off a screen or a page, take the 158 ms and keep 512.

### Changing it

Back up first, edit, restart:

```bash
ENGINE_VISUAL=/opt/tensorrt-edgellm/models/default/visual/config.json

ssh "$JETSON_HOST" "sudo cp $ENGINE_VISUAL $ENGINE_VISUAL.512.bak"

ssh "$JETSON_HOST" "sudo python3 - <<'PY'
import json
p = '$ENGINE_VISUAL'
cfg = json.load(open(p))
cfg['builder_config']['max_image_tokens_per_image'] = 320
json.dump(cfg, open(p, 'w'), indent=2)
print('max_image_tokens_per_image =', cfg['builder_config']['max_image_tokens_per_image'])
PY"

ssh "$JETSON_HOST" 'sudo systemctl restart cosmos3-edge-shim.service'
```

Confirm it took effect from the `[perf]` line — `prompt_tok` should fall from ~510/511 to ~330:

```bash
ssh "$JETSON_HOST" 'journalctl -u cosmos3-edge-shim.service -f | grep --line-buffered "\[perf\]"'
```

### Reverting

```bash
ssh "$JETSON_HOST" "sudo cp $ENGINE_VISUAL.512.bak $ENGINE_VISUAL && sudo systemctl restart cosmos3-edge-shim.service"
```

Only the values **512 (default) and 320** were exercised in this deployment. Lower token counts
appear on the resolution sweep above, but those were produced by shrinking the input image at a
512 cap — no cap value below 320 was exercised here. Raising the cap **above** 512 was not tested
either, and the ViT optimization profile was built with an `opt` shape near 514 image tokens, so
larger budgets would move you off the profile's sweet spot — treat anything above 512 as
unverified.

---

## 6. Connecting Live VLM WebUI

If you installed `live-vlm-webui.service` in §4 it is already connected: it was started with
`--api-base http://localhost:8000/v1` and `--model nvidia/Cosmos3-Edge`, i.e. it talks to the shim
over loopback as if it were an OpenAI server. To run it by hand instead (useful while debugging):

```bash
ssh "$JETSON_HOST"
/home/orin/TensorRT-Edge-LLM/.venv/bin/live-vlm-webui \
  --host 0.0.0.0 --port 8090 \
  --model nvidia/Cosmos3-Edge \
  --api-base http://localhost:8000/v1 \
  --api-key EMPTY
```

Before pointing a browser at it, verify the backend directly — this isolates shim problems from
WebUI problems:

```bash
# 1. is the model advertised?
ssh "$JETSON_HOST" 'curl -s http://localhost:8000/v1/models'

# 2. text-only round trip
ssh "$JETSON_HOST" 'curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"nvidia/Cosmos3-Edge\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":16}"'
```

Then open `http://<jetson-host>:8090/` and select your camera. Browsers generally restrict camera
access to secure contexts (HTTPS or `localhost`), so a plain-HTTP page served from another machine
may be blocked by the browser rather than by anything on the Jetson — this constraint was not
specifically characterized in this deployment, so treat it as a thing to check, not a documented
result.

Images arrive from the WebUI as base64 `data:` URLs in the `image_url` content parts, which is the
path the shim handles with `rt.load_image_from_bytes()` (§2.3).

### The WebUI's latency number is not the model's latency

This trips people up. The figure the WebUI reports is a full client-side round trip: frame capture,
JPEG encode in the browser, network transfer, queueing behind any in-flight request, the shim's own
work, then the response render. The shim's `[perf]` line covers only the server side.

In this deployment, the two were roughly:

| Measured where | Mean | Provenance |
|---|---|---|
| Shim side (`[perf] elapsed_ms`) | 626–627 ms | **directly measured** — `collect_perf.py` over the n = 699 post-change requests reported `mean elapsed=626ms` |
| Reported by Live VLM WebUI | ~1851 ms | **operator-reported** — observed in the WebUI during the session, not independently instrumented by this project |

so roughly two thirds of what the browser displays is **not** model execution. Treat that ratio as
indicative rather than as a measured result: only the shim-side figure comes from this repo's
tooling. The mean itself is sample-dependent — it moves with the response-length distribution of
whatever traffic is in the window (619 ms over the clean n=88 post-change window, 626 ms over the
larger n=699 window that also contains probe traffic) — which is precisely why the regression
coefficients, stable at 13.13–13.16 ms/token and 287–289 ms fixed, are the figures to compare. Use the
shim-side `[perf]` line for any optimization decision —
the WebUI number is a user-experience metric, and it is a legitimate one to care about, but it is
not the number you regress against.

---

## 7. Quick troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Plugin not found, are the plugin name, version, and namespace correct?` at startup | `EDGELLM_PLUGIN_PATH` is wrong or the plugin `.so` is missing. Add `Environment=EDGELLM_PLUGIN_PATH=/path/to/libNvInfer_edgellm_plugin.so` to the unit (it overrides the shim's `setdefault`), or `LD_PRELOAD` it. |
| `ModuleNotFoundError: _edgellm_runtime` | the `sys.path.insert` target (`build/pybind`) does not exist — the pybind module was not built, or `WorkingDirectory`/paths point at the wrong checkout |
| Unit starts, then requests 502 for several seconds | expected: the WebUI unit is only ordered after the shim, not gated on `[shim] ready`. Wait for `[shim] ready` in the journal. |
| Every request takes seconds and `[perf]` shows sane `gen_tok` | you are probably measuring contention — the shim is single-flight. Stop the WebUI before probing, or use minimums. |
| Probe latency suspiciously low and flat | the vision-encoder embedding cache is keyed on raw pixel bytes; you are re-sending an identical image and skipping the ViT entirely. Generate a unique image per iteration. |
| `collect_perf.py` prints `insufficient samples` | no `[perf]` lines in the window, or `JETSON_HOST` is unset/wrong (it defaults to `orin@jetson.local`) |
| `R²` collapses across a window | the window spans a config change (two fixed-cost regimes) or the client emitted constant-length replies. Use `compare_perf.py` matched on `gen_tok` instead. |

Full write-up of the campaign, including the quantization work behind the INT4 engine, is in
[`../docs/report.md`](../docs/report.md).
