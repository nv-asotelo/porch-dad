# Measurement Methodology

How to benchmark a Cosmos3-Edge deployment on a Jetson Orin Nano Super without fooling yourself.

This document is written to stand on its own. It describes the traps that produced confidently
wrong numbers during this campaign, the metric that survived contact with a live workload, the
statistical failure modes of that metric, and the two roofline checks that decided which
optimizations were worth attempting at all. The specific numbers come from this deployment
(Jetson Orin Nano Super 8 GB, JetPack 7.2.1, TensorRT 10.16.2.10, TensorRT-Edge-LLM 0.10.1,
`nvidia/Cosmos3-Edge` reasoner); the reasoning transfers to any single-stream VLM on an edge device.

The short version:

1. A vision-encoder embedding cache keyed on raw pixel bytes will silently delete ~250 ms of real
   work from any benchmark that loops the same image.
2. The runtime serializes requests. Any probe fired at a busy server measures the queue, not the model.
3. Response length varies per request, so mean latency is not comparable between runs. Regress
   instead: `elapsed_ms = fixed_ms + marginal_ms_per_token × gen_tok`.
4. Before optimizing anything, compute what fraction of the relevant hardware peak you already
   achieve. It tells you whether to tune the kernel or delete the work.

Related: [full report](./report.md) · [`scripts/collect_perf.py`](../scripts/collect_perf.py) ·
[`scripts/compare_perf.py`](../scripts/compare_perf.py) · [`serve/cosmos3_shim.py`](../serve/cosmos3_shim.py)

---

## 1. Two traps

### 1.1 Trap 1 — the encoder embedding cache

TensorRT-Edge-LLM caches vision-encoder embeddings, keyed on **raw pixel bytes**
(`cpp/.../llmRankRuntime.cpp:1967`). Submit the same image twice and the second request skips the
entire ViT forward pass.

On this device the ViT encode is approximately **248 ms** (Section 6). That is 55% of the fixed
per-request cost and 26% of mean end-to-end latency before the image-token budget was reduced.
A benchmark loop of the form

```python
# WRONG — measures the cache, not the model
img = open("test.jpg", "rb").read()
for _ in range(50):
    post(img, "Describe.")
```

will report a number that is roughly a quarter of a second too fast on every iteration after the
first, and it will do so silently. There is no warning, no log line, and no change in the response.

**Defeating it.** Every synthetic probe in this repo generates a **unique image per iteration**.
[`scripts/profile_fixed.py`](../scripts/profile_fixed.py) and
[`scripts/tok_vs_res.py`](../scripts/tok_vs_res.py) both build a fresh noise image from a per-call
random seed and JPEG-encode it:

```python
def img_b64(seed, w=640, h=480):
    # unique image each call -> defeats the encoder embedding cache
    random.seed(seed)
    ...
```

Because the key is **decoded pixel bytes**, it is not enough to vary the container: re-muxing,
changing EXIF, or renaming a file may leave the decoded pixels identical. Change actual pixels.

**Detecting that you fell in anyway.** Run the probe and compare iteration 1 against iteration 2
with the *same* image. If the second is roughly 250 ms faster, the cache is answering. A second
signature: an elapsed-time distribution with two tight clusters separated by ~250 ms and nothing
in between.

**What is and is not affected.** Live camera traffic largely sidesteps this trap, because
consecutive frames from a real sensor differ byte-for-byte after re-encoding — which is why the
live-traffic measurements in this repo are trustworthy on this axis. Replayed files, a paused or
frozen stream, a synthetic still, and any fixed test corpus all put you straight back into it.
Treat "the stream was live" as an assumption to verify, not a guarantee.

### 1.2 Trap 2 — contention with live traffic

The serving path is single-threaded by construction. In
[`serve/cosmos3_shim.py`](../serve/cosmos3_shim.py):

```python
_lock = asyncio.Lock()
_pool = ThreadPoolExecutor(max_workers=1)
```

and in the handler:

```python
async with _lock:
    resp = await loop.run_in_executor(_pool, _runtime.handle_request, req)
```

One request is in flight at a time. A probe fired while the Live VLM WebUI is streaming does not
run concurrently with the WebUI request — it waits for it, and the wall-clock time it reports
includes that wait.

This is compounded by where the instrumentation starts. In the same file:

```python
async def chat_completions(request: Request):
    _t_start = time.time()
    body = await request.json()
```

`_t_start` is taken on arrival, **before** the lock is acquired. The `elapsed_ms` value in the
`[perf]` journal line therefore spans *queue wait + inference*, not inference alone. For the
intended workload — one WebUI client, one request at a time — queue wait is negligible and
`elapsed_ms` is a clean latency measurement. Add a second concurrent client and the regression's
intercept starts absorbing queue time, at which point `fixed_ms` is no longer a property of the
hardware. If you benchmark under concurrency, say so explicitly and expect the intercept to be
meaningless.

**Consequence for statistics.** Contention can only ever *add* time. It has no mechanism to make a
request faster. Under contention, therefore, the **minimum** is the honest statistic and the mean
is a measurement of how busy the box was. Medians are a compromise and are only safe when the
interfering load is light relative to the sample count.

### 1.3 Worked example: the "context cache regression" that was not real

This is included because being wrong in public, and then correcting it, is the only part of
benchmarking methodology that is genuinely hard to teach.

**The claim.** During Round 2, enabling context-cache reuse was measured and reported as a
**regression: 4.98 s per request**, against a Round 1 baseline of roughly 2 s. The conclusion drawn
at the time was that the context cache was actively harmful on this model and should be reverted.

**Why it was wrong.** The probe ran against the live shim while the Live VLM WebUI was still
streaming webcam frames into the same endpoint. Given the `asyncio.Lock` plus single-worker
executor above, the probe request sat behind one or more in-flight WebUI requests, and `_t_start`
had already been taken. The 4.98 s was the probe's own inference *plus* the tail of somebody
else's. Nothing about it was attributable to the context cache. The sample size was small enough,
and the contention heavy enough, that one bad draw dominated the reported figure.

**How it was corrected.** Re-measure with the contention controlled rather than ignored:

- take **minimums** over repeated probes instead of means, since contention is strictly additive;
- or quiesce the interfering client (`systemctl stop live-vlm-webui.service`) for the duration;
- or abandon synthetic probing entirely and read the WebUI's own `[perf]` lines out of the journal,
  comparing **matched `gen_tok` buckets** across two time windows (Section 4). Requests logged by
  the WebUI are the workload, so they are not competing with a separate probe.

**The honest result.** Context-cache reuse was a **−3.5%** improvement, and greedy decode was
**−4.3%**. Both were real gains and both were below the campaign's improvement bar, so Round 2 was
recorded as a **failure to clear the threshold — not a regression**. The retraction changed the
sign of the finding, not just its magnitude.

**The transferable lesson.** A single slow sample against a serialized server is a measurement of
the queue. Before attributing a latency change to the thing you just changed, establish that the
measurement window contained no other traffic, or use a statistic (minimum) and a comparison
(matched workload) that contention cannot corrupt.

---

## 2. The metric

### 2.1 Why mean latency is not comparable

The workload is a webcam-driven VLM answering open-ended prompts. Response length varies request to
request — in the live samples analysed here, generated-token counts ranged from 17 to 34 tokens in
the dense buckets alone. At a decode rate of ~13.13 ms/token, a 17-token spread is ~223 ms of
latency difference caused by nothing but how talkative the model felt. That is the same order of
magnitude as the entire fixed cost of the request.

So a before/after comparison of mean latency is a comparison of two different workloads. It can
show an improvement that is entirely an artifact of shorter replies, or hide a real one behind
longer ones.

### 2.2 The model

```
elapsed_ms = fixed_ms + marginal_ms_per_token × gen_tok
```

Both terms map to something physical and, critically, to **different optimizations**:

| Term | Physical meaning | Scales with | Attacked by |
|---|---|---|---|
| `fixed_ms` | image decode and preprocessing, ViT encode, LLM prefill over the full prompt | image token count | Round 4 (image-token budget 512 → 320) |
| `marginal_ms_per_token` | one autoregressive decode step: stream the weights, update KV cache, sample | weight bytes moved per token | Round 3 (INT4 W4A16 quantization) |

That separation is the entire value of the metric. A change that halves decode cost and a change
that halves ViT cost look identical in a mean-latency table and completely different here.

Final fit for this deployment, over live WebUI traffic:

| Quantity | Value |
|---|---|
| `marginal_ms_per_token` | 13.13 ms/token (76.0 tok/s) |
| `fixed_ms` | 289 ms |
| R² | 0.999 |
| n | 699 requests |

### 2.3 Do not use the per-request `ms_per_tok` field

The shim also logs `ms_per_tok = elapsed_ms / gen_tok`. That is a convenience field, **not** the
decode rate — it amortizes the fixed cost across however many tokens happened to be generated. For
a 20-token response taking 556 ms it reports 27.8 ms/token, more than double the true marginal rate
of 13.13 ms/token, and it will drift as response lengths drift. Use it for eyeballing a live
journal; never quote it as a decode rate.

### 2.4 The intercept is an extrapolation — validate it independently

`fixed_ms` is the fitted value at `gen_tok = 0`, and no observation exists there. With live samples
clustered between 17 and 34 generated tokens, the intercept is extrapolated roughly 17 tokens
beyond the nearest data point, so a small error in the slope swings it substantially. A high R² does
not protect you from this: R² measures fit over the observed range, not the credibility of an
extrapolation outside it.

The fix is triangulation. This deployment's intercept was cross-checked against a direct
measurement of the same quantity:

| Method | Value |
|---|---|
| Regression intercept over 699 live requests | 289 ms |
| Direct probe: image path (640×480, 300 image tokens) + 1 generated token, minimum | 287 ms |

Two independent methods, 2 ms apart. That agreement — not the R² — is what makes the fixed-cost
number believable. Run the direct probe (Section 5) any time the intercept moves.

---

## 3. When the regression breaks

Three distinct failure modes, with different responses. Two of them are not bugs.

### 3.1 Degenerate case — constant response length

If the client is configured so every reply is the same length (a fixed `max_tokens` that is always
hit, or a prompt template that always terminates at the same point), then every `gen_tok` is
identical, the variance of the regressor is zero, and the fit has no unique solution: infinitely
many `(fixed, marginal)` pairs reproduce the data exactly. R² collapses or is undefined.

`collect_perf.py` detects this explicitly rather than emitting nonsense:

```python
den = sum((x - mx) ** 2 for x in xs)
...
if den == 0:
    print(f"  all gen_tok identical ({xs[0]:.0f}) - cannot separate fixed/marginal")
    sys.exit()
```

**Response:** fall back to matched-`gen_tok` comparison (Section 4). This is not a downgrade —
comparing like-for-like response lengths is the stronger comparison whenever it is available,
because it assumes no model at all.

### 3.2 Narrow `gen_tok` range

A milder version of the same problem. If `gen_tok` spans 20–22 rather than 17–34, the slope is
estimated from very little leverage and the intercept is extrapolated far outside the data. The fit
will still report a high R² — it is fitting a short segment well — while the intercept is close to
unconstrained. Always print the `gen_tok` range alongside the fit, which `collect_perf.py` does:

```
n=699  gen_tok range 14-49  mean elapsed=626ms
```

**Response:** widen the range (vary `max_tokens` or the prompt), or accept the slope and get
`fixed_ms` from the direct probe instead.

### 3.3 A window that spans a configuration change

This is the case most likely to be misread as a broken tool. If your time window straddles a shim
restart that changed a fixed-cost parameter — for example the image-token budget going from 512 to
320 — the sample contains **two populations with the same slope and different intercepts**. In this
deployment the two regimes are separated by a constant **158 ms** (Section 4).

Fitting one line through both produces residuals on the order of that 158 ms offset, which is
comparable to the entire latency spread generated by the observed `gen_tok` range. R² drops,
sometimes dramatically.

**That low R² is correct behaviour, not a fault.** The data genuinely is not one line. The tool is
telling you your window is contaminated.

**Response:** split the window at the change boundary and use
[`compare_perf.py`](../scripts/compare_perf.py) with an explicit `until` for the "before" window,
so the restart falls in neither sample. A useful sanity check is that the split fits should show a
roughly *unchanged* slope and a shifted intercept — if the slope moved too, the restart changed
more than you thought it did.

---

## 4. Matched-`gen_tok` A/B

Rather than modelling the length dependence, eliminate it: bucket requests by exact `gen_tok`,
compare only buckets present in both windows, and report the per-bucket median.

`compare_perf.py` weights the summary by `min(n_A, n_B)` per bucket, so a bucket with 60 before-
samples and 4 after-samples contributes 4 units of weight, not 60:

```python
w = min(len(da[g]), len(db[g]))
tot_a += ma * w; tot_b += mb * w; wt += w
```

**Reading the shape of the delta is the point.** The per-bucket deltas carry a signature:

- **A constant delta across all `gen_tok` buckets** ⇒ the change moved `fixed_ms`. The extra work
  is paid once per request regardless of response length.
- **A delta proportional to `gen_tok`** ⇒ the change moved `marginal_ms_per_token`. Longer replies
  benefit proportionally more.
- **Both** ⇒ two things changed, or something changed that you did not intend to change.

The Round 4 result (image-token budget 512 → 320) is a textbook constant-delta case:

| gen_tok | n (before) | n (after) | before | after | delta | improvement |
|---|---|---|---|---|---|---|
| 17 | 13 | 3 | 672 ms | 516 ms | −156 ms | 23.2% |
| 18 | 11 | 2 | 687 ms | 528 ms | −158 ms | 23.1% |
| 20 | 60 | 4 | 714 ms | 556 ms | −158 ms | 22.2% |
| 21 | 15 | 7 | 727 ms | 567 ms | −160 ms | 22.0% |
| 22 | 70 | 5 | 738 ms | 582 ms | −156 ms | 21.1% |
| 23 | 9 | 12 | 756 ms | 594 ms | −162 ms | 21.5% |
| 27 | 12 | 2 | 808 ms | 646 ms | −161 ms | 19.9% |
| 34 | 31 | 8 | 899 ms | 740 ms | −159 ms | 17.7% |

Weighted improvement: **20.8%**. The delta is flat between −156 and −162 ms (mean −158.75 ms)
across a 2× range of response lengths, which is exactly what a fixed-cost reduction looks like and
is inconsistent with a decode speedup. Note also that the percentage improvement *declines* with
`gen_tok` (23.2% → 17.7%) purely because the denominator grows — another reason percentages of
mean latency are a poor way to report this class of change.

**Caveat on bucket sample counts.** Several "after" buckets have n = 2–4. A median of two samples
is a weak statistic. The confidence here comes from the *consistency of the constant across eight
independent buckets*, not from any single bucket. Do not report a single sparse bucket alone.

---

## 5. The tools, and how they are actually run

### 5.1 The log-line contract

Everything downstream depends on one `print` in [`serve/cosmos3_shim.py`](../serve/cosmos3_shim.py):

```python
print(f"[perf] elapsed_ms={_elapsed_ms:.0f} prompt_tok={_prompt} "
      f"gen_tok={_gen} ms_per_tok={_mspt:.1f}", flush=True)
```

`gen_tok` is `len(resp.output_ids[0])` — the tokens the runtime actually produced, not `max_tokens`.
`prompt_tok` is the runtime's prompt token count, which is how the image-token budget change is
observed directly (`prompt_tok` 510/511 before Round 4, 330 after). The service runs under systemd,
so these lines land in journald under `cosmos3-edge-shim.service`.

Both analysis scripts parse exactly this format:

```python
pat = re.compile(r"elapsed_ms=(\d+) prompt_tok=(\d+) gen_tok=(\d+)")
```

If you change the log line, change the regex. Requests with `gen_tok == 0` are filtered out by both
scripts — an empty completion carries no decode information and would produce an infinite
`ms_per_tok`.

### 5.2 `JETSON_HOST`

`collect_perf.py` and `compare_perf.py` run **on a workstation** and SSH into the Jetson. Both read
the target from the `JETSON_HOST` environment variable, defaulting to `orin@jetson.local`:

```python
JETSON = os.environ.get("JETSON_HOST", "orin@jetson.local")
```

```bash
export JETSON_HOST=orin@jetson.local   # user@host or an ssh_config alias
```

Key-based SSH auth is assumed; both scripts invoke `ssh -o BatchMode=yes` and discard stderr, so
ssh never prompts — a missing key or an unknown host key surfaces as an empty result set rather
than an error, instead of hanging on a prompt you cannot see. If you
get `insufficient samples: 0`, verify the SSH path first:

```bash
ssh "$JETSON_HOST" journalctl -u cosmos3-edge-shim.service --since '5 min ago' --no-pager | tail
```

### 5.3 `collect_perf.py` — fit the model

```bash
export JETSON_HOST=orin@jetson.local
python3 scripts/collect_perf.py "30 min ago"
```

One optional positional argument: a window start, passed verbatim to `journalctl --since`
(default `"10 min ago"`). Any string `journalctl` accepts works — `"2026-09-13 14:00:00"`,
`"yesterday"`, `"-1h"`.

It pulls the journal, parses `[perf]` lines, and does an ordinary least-squares fit of `elapsed_ms`
on `gen_tok`. Output:

```
n=699  gen_tok range 14-49  mean elapsed=626ms
  marginal = 13.13 ms/token
  fixed    = 289 ms
  R^2      = 0.999
```

Behaviour worth knowing:

- fewer than 2 parsed samples → `insufficient samples: N` and exit;
- zero `gen_tok` variance → the degenerate-case message from Section 3.1 and exit;
- no outlier rejection, no trimming. That is deliberate: with the contention trap in play, silently
  discarding slow samples would hide exactly the problem you need to see. Inspect the range and
  mean it prints before trusting the fit.

### 5.4 `compare_perf.py` — matched-`gen_tok` A/B

```bash
export JETSON_HOST=orin@jetson.local
python3 scripts/compare_perf.py "14:00:00" "14:25:00" "14:30:00"
```

Three **required** positional arguments, in order:

| Position | Meaning |
|---|---|
| 1 | `A_since` — start of the "before" window |
| 2 | `A_until` — end of the "before" window |
| 3 | `B_since` — start of the "after" window (runs to now) |

The "after" window has no end bound, which makes the intended usage clear: change the config,
restart, let traffic accumulate, then compare against a bounded pre-change window. Put the restart
**between** `A_until` and `B_since` so neither window straddles it (Section 3.3).

Output is the two window summaries, the per-bucket table (top 8 shared `gen_tok` values, ranked by
combined sample count), and the weighted improvement:

```
=== BEFORE ===
A: n=... prompt_tok=510-511 gen_tok=17-34 mean_elapsed=...ms
=== AFTER  ===
B: n=... prompt_tok=330-330 gen_tok=17-34 mean_elapsed=...ms

 gen_tok   nA   nB     A_ms     B_ms    delta  improve
 ...

weighted matched-gen_tok improvement: 20.8%
```

The `prompt_tok` range in the window summaries is a free correctness check: if you changed the
image-token budget and `prompt_tok` did not move, you are comparing two identical configurations.

**Known rough edge:** `compare_perf.py` contains a `fit()` function that is never called and whose
R² expression is vestigial (it is annotated `# unused` in the source). Ignore it; use
`collect_perf.py` for regression. Similarly `describe()` returns an empty dict that no caller uses.

### 5.5 On-device probes

These three run **on the Jetson**, against `http://127.0.0.1:8000`, and are the tools for questions
the journal cannot answer.

[`scripts/profile_fixed.py`](../scripts/profile_fixed.py) — splits the fixed cost into ViT versus
prefill. Two warm-up calls, then six image requests and six text-only requests interleaved, all at
`max_tokens=1` so that `elapsed ≈ fixed`, each with a **unique** 640×480 noise image:

```bash
python3 scripts/profile_fixed.py
```

[`scripts/tok_vs_res.py`](../scripts/tok_vs_res.py) — sweeps input resolution over the published
ladder (320×240 through 1280×960) and reports `prompt_tokens`, the image-only token count (raw
`prompt_tokens` minus a text-only baseline request), and the **best** (minimum) of four timings per
resolution, at `max_tokens=1`. This
is the measurement that located the image-token cap and motivated Round 4.

[`scripts/quality.py`](../scripts/quality.py) — three fixed prompts against
`/home/orin/bench_frame.jpg` at `temperature=0.0`, tagged by a mandatory argv label:

```bash
python3 scripts/quality.py before   # ... change config, restart shim ...
python3 scripts/quality.py after
diff <(python3 scripts/quality.py after) ...
```

Greedy decoding is what makes this useful: at temperature 0 the comparison is byte-level. If the
outputs are identical, the change was quality-neutral on that input, full stop. It is a *probe*, not
a benchmark — three prompts on one frame. No task suite was run in this deployment, and that is a
real gap, stated as such in the report.

### 5.6 Discrepancies between the committed probes and the recorded results

Stated plainly rather than papered over, because a reader re-running these will notice:

- **`profile_fixed.py` prints medians; the recorded decomposition used minimums.** Under contention
  the minimum is the defensible statistic (Section 1.2). If you run this against live traffic,
  change `med()` to `min()` or quiesce the stream first.
- **The recorded numbers predate the committed shim's `usage` field.** `tok_vs_res.py` reads
  `r.json()["usage"]["prompt_tokens"]`, and the committed `serve/cosmos3_shim.py` returns a real
  `usage` object (`prompt_tokens` / `completion_tokens` / `total_tokens`). The figures in this repo
  were gathered with an earlier snapshot that returned `"usage": None` and logged the same counts on
  the `[perf]` line instead, so reading `prompt_tok` from the journal remains an equivalent
  fallback.
- **The committed sweep matches the published ladder** — 320×240, 448×336, 640×480, 896×672,
  1280×960 — and subtracts a real text-only baseline, so its `img_tok` column is an image-only
  count rather than raw `prompt_tok`. What carries the conclusion is still the shape of the result:
  linear growth until the token cap, flat afterwards.

---

## 6. Fixed-cost decomposition

The regression gives one number for everything that happens before the first token. To act on it
you need to know how it splits. The method is subtraction:

1. Send an **image** request with `max_tokens=1`. Elapsed ≈ image decode + preprocessing + ViT
   encode + prefill over the image and text tokens + one decode step.
2. Send a **text-only** request with `max_tokens=1`. Elapsed ≈ prefill over text tokens + one
   decode step.
3. The difference is the image path: preprocessing plus ViT encode plus the extra prefill over the
   image tokens.

Every image must be unique (Trap 1) and, under contention, take minimums (Trap 2).

| Measurement (minimums, under contention) | Value |
|---|---|
| text-only prefill + 1 token | 39 ms |
| image path (640×480, 300 image tokens) + 1 token | 287 ms |
| **⇒ ViT encode + image prefill** | **≈ 248 ms** |

Before Round 4, that 248 ms was **55% of the fixed term** and **26% of mean end-to-end latency** —
which is what justified spending a round on it.

**Independent cross-check.** The resolution sweep measured 288 ms at the same 640×480 / 300-token
operating point via a different script and a different sampling rule. Subtracting the same 39 ms
text-only floor gives 249 ms. Two paths to the same number within 1 ms.

**The curve is super-linear in image tokens.** Subtracting the 39 ms floor from the sweep (derived
arithmetic on the measured values, not separately measured):

| Image tokens | Measured best | minus 39 ms floor | per image token |
|---|---|---|---|
| 80 | 90 ms | 51 ms | 0.64 ms |
| 140 | 152 ms | 113 ms | 0.81 ms |
| 300 | 288 ms | 249 ms | 0.83 ms |
| 512 (at cap) | 499 ms | 460 ms | 0.90 ms |

Cost per image token rises with token count, which is what a transformer encoder does: the
projection GEMMs scale linearly with sequence length while the attention score matrices scale
quadratically. The practical consequence is that cutting the token budget buys slightly *more* than
its proportional share — and that probes at or above the cap (512 here) measure the cap, not the
resolution, so two different resolutions can return the same latency and the same `prompt_tok`.

---

## 7. Roofline checks: decide what to optimize before optimizing it

The most useful thing done in this campaign was, twice, to compute the achieved fraction of the
relevant hardware peak *before* choosing an optimization. Both times it redirected the work.

### 7.1 Decode is memory-bandwidth-bound

Single-stream autoregressive decode reads essentially every weight once per token. At FP16, the
engine's TensorRT-reported weights memory was 3,355,696,384 B ≈ 3.36 GB, and the measured decode
rate was 43.29 ms/token:

```text
3.36 GB / 0.04329 s = 77.6 GB/s achieved
77.6 / ~102 GB/s peak LPDDR5 = 76% of theoretical bandwidth
```

At 76% of peak bandwidth there is no scheduling trick left. This is precisely why Round 2 — context
cache reuse, greedy decode, a `top_k` fix — could only produce 3.5–4.3%: it was optimizing
scheduling on a bandwidth-bound kernel. **The failed round was informative because of this
calculation.** The only remaining lever was to move fewer bytes, which is Round 3 (INT4 W4A16,
weights memory 865,480,704 B, a 3.88× reduction) and which delivered −69.1%.

Repeating the arithmetic post-quantization (0.865 GB at 13.13 ms/token ≈ 66 GB/s, about 65% of
peak) suggests decode is still bandwidth-dominated but with more headroom than before. *This figure
is derived from the two recorded values above, not separately measured, and it ignores KV-cache and
activation traffic, which matter relatively more once the weights shrink.*

### 7.2 The ViT is already reasonably efficient

The same question for the vision tower: is 248 ms slow, or is it simply a lot of arithmetic?

From the vision tower config — 27 layers, hidden 1152, intermediate 4304, ~411M parameters,
patch 16 with spatial merge 2, so 512 image tokens correspond to 2048 patches:

```text
GEMM work      = 2 × params × patches
               = 2 × 411e6 × 2048                   ≈ 1.68 TFLOP
attention work = 27 layers × 4 × patches² × hidden
               = 27 × 4 × 2048² × 1152              ≈ 0.52 TFLOP
total                                               ≈ 2.21 TFLOP

2.21 TFLOP / 0.248 s = 8.9 TFLOPS achieved
8.9 / ~16.7 TFLOPS dense FP16 peak = ~53% of peak
```

**53% of dense FP16 peak on a real transformer encoder is respectable.** The absolute ceiling from
perfect kernel tuning is therefore under 2×, and realistically much less — a large engineering
effort for a bounded return, on kernels inside a closed runtime that this project does not own.

Meanwhile, the token budget is a **runtime configuration value**: `max_image_tokens_per_image` is
read at runtime from the engine's `visual/config.json` (`builder_config`) and consumed by
`smartResize` in `cpp/multimodal/common/imageUtils.cpp:145`, where
`maxPixels = max_image_tokens_per_image × 32²`. No engine rebuild, reversible in about 4 seconds by
restarting the shim. Cutting 512 → 320 removed 37.5% of the tokens and, per Section 6's
super-linear curve, somewhat more than 37.5% of the ViT arithmetic.

**The rule this produces:** high achieved-fraction-of-peak means *do less work*; low
achieved-fraction-of-peak means *do the same work better*. The ViT was at 53% and the answer was to
shrink the input; decode was at 76% and the answer was to shrink the weights. In neither case was
the answer to write a faster kernel.

**The tradeoff, stated in the same breath:** the token-budget reduction is not free. At greedy
decoding it was byte-identical on natural camera scenes across three prompts, but on a dense text
screenshot the 512-token configuration quoted on-screen annotations verbatim while the 320-token
configuration drifted vaguer and partly hallucinated. 320 is the right setting for live camera work;
512 is the right setting for document and screenshot reading. A benchmark that only used camera
frames would have reported this as a pure win, which is the fourth trap and the reason
`quality.py` exists.

---

## 8. Procedure

To reproduce or to evaluate your own change:

```bash
# 0. Baseline conditions. Record them; they change results by more than most optimizations.
ssh "$JETSON_HOST" 'sudo nvpmodel -q && sudo jetson_clocks --show | head'

# 1. Establish the current model over real traffic.
export JETSON_HOST=orin@jetson.local
python3 scripts/collect_perf.py "30 min ago"
#    Check: n large, gen_tok range wide, R^2 high. If R^2 is low, find the config change.

# 2. Validate the intercept independently (run on the Jetson).
ssh "$JETSON_HOST" 'cd /opt/tensorrt-edgellm && python3 profile_fixed.py'
#    The image-path minimum should agree with the fitted fixed_ms.

# 3. Quality reference BEFORE the change, greedy decode (run on the Jetson).
ssh "$JETSON_HOST" 'cd /opt/tensorrt-edgellm && python3 quality.py before' > /tmp/q_before.txt

# 4. Note the wall-clock time, make the change, restart, note the time again.
ssh "$JETSON_HOST" 'sudo systemctl restart cosmos3-edge-shim.service'

# 5. Let traffic accumulate, then compare with the restart strictly between the windows.
python3 scripts/compare_perf.py "14:00:00" "14:25:00" "14:30:00"
#    Read the SHAPE of the delta, not just the headline percentage.

# 6. Quality AFTER, and diff it.
ssh "$JETSON_HOST" 'cd /opt/tensorrt-edgellm && python3 quality.py after' > /tmp/q_after.txt
diff /tmp/q_before.txt /tmp/q_after.txt
```

Paths on the Jetson depend on where you installed the probes; the repo's own deployment layout is
described in the [report](./report.md). The systemd unit names used above match
[`systemd/cosmos3-edge-shim.service`](../systemd/cosmos3-edge-shim.service) and
[`systemd/live-vlm-webui.service`](../systemd/live-vlm-webui.service).

**Checklist before publishing a number:**

- [ ] Every synthetic probe used a unique image per iteration.
- [ ] The measurement window is free of a config change (check `prompt_tok` range and R²).
- [ ] No competing client was hitting the endpoint, or the statistic is a minimum, or the
      comparison is matched-`gen_tok`.
- [ ] The comparison controls for response length — matched `gen_tok`, or a regression with a
      reported `gen_tok` range and R².
- [ ] The fitted intercept was cross-checked against a direct fixed-cost probe.
- [ ] The delta's shape (constant vs proportional) is consistent with the mechanism you claim.
- [ ] Power mode and clocks are recorded and identical between the two windows.
- [ ] A quality check was run at temperature 0, and any degradation is reported alongside the speedup.

---

## 9. What this methodology does not establish

Stated plainly, because a methodology document that only lists its strengths is marketing.

- **No task-suite quality benchmark was run.** Quality evidence here is greedy-decode spot checks on
  a handful of prompts. The INT4 weights carry 11.06% mean relative weight error (worst layer 20.5%)
  from uncalibrated RTN with MSE-optimal clipping. Evaluate on a real task set before production use.
- **No FP16-versus-INT4 side-by-side on identical prompts**, because both engines cannot be resident
  within 8 GB simultaneously. The quality comparison that would be most convincing is the one this
  hardware cannot run.
- **Single-stream only.** Every number here is one request at a time. Nothing in this document
  characterizes throughput under concurrency, and as Section 1.2 explains, the instrumentation's
  `elapsed_ms` would conflate queue time with inference time if you tried.
- **Measurements come from live Live-VLM-WebUI traffic**, which is the right workload but is subject
  to contention. Minimums and matched-`gen_tok` comparisons control for it; they do not eliminate it.
- **The headline 22.5× is against a naive baseline** (a CLI process spawned per request, reloading
  engines every time). Against a competently configured FP16 resident baseline the honest figure is
  3.38× on decode plus 2.32 GB of RAM reclaimed.
- **Engine build parameters** are recorded as `--maxBatchSize 1 --maxKVCacheCapacity 2048`. The
  device went offline before a final re-read of `config.json`, so treat these as the values used
  rather than as freshly verified output.
