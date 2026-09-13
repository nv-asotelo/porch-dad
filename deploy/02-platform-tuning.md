# 02 — Platform tuning: power mode, clocks, swap, memory headroom

This is the cheapest change in the entire project. It requires no model changes, no rebuild, no
code, and it is reversible in one command. Do it before you measure anything else, because every
benchmark taken at the default power mode is measuring the wrong machine.

Two commands do the work:

```bash
sudo nvpmodel -m 2      # MAXN_SUPER
sudo jetson_clocks      # pin clocks to the maximum allowed by the active mode
```

`nvpmodel -m 2` selects the power profile (it raises the *ceiling*). `jetson_clocks` disables the
dynamic frequency governors and pins every clock to that ceiling (it removes the *ramp*). You want
both: the profile alone still lets DVFS idle the GPU and memory controller down between requests,
which shows up as a slow first token on every request after a pause.

## What actually changes

Measured on this deployment (Jetson Orin Nano Super 8 GB, JetPack 7.2.1 / L4T R39.2.1):

| Clock domain | Default | MAXN_SUPER + `jetson_clocks` | Ratio |
|---|---|---|---|
| GPU | 306 MHz | 1020 MHz | 3.33× |
| CPU | — | 1728 MHz | — |
| EMC (memory controller) | 2133 MHz | 3199 MHz | 1.50× |

The CPU default is recorded as "raised to 1728 MHz" in this deployment's notes without a
corresponding idle-state figure, so the ratio is left blank rather than guessed.

**Honest scoping note:** these settings were applied as part of optimization Round 1, together with
two other changes (a resident runtime replacing a process-per-request CLI, and a captured decoding
CUDA graph). That bundle took end-to-end latency from 13.91 s/request to 2.07 s/request (−85.1%).
The isolated contribution of `nvpmodel`/`jetson_clocks` alone was **not separately measured** here.
Treat the clock table as verified and the attribution of the Round 1 speedup as shared.

## Why EMC is the clock that matters

The intuition most people bring from desktop GPUs — "raise the GPU core clock" — is the wrong
intuition for single-stream LLM decode. Decode with batch size 1 does almost no arithmetic per byte
loaded: every generated token requires streaming the *entire* weight set from memory through the
GPU once. It is bandwidth-bound, not FLOP-bound.

This was not assumed. It was measured, and it is the reason optimization Round 2 (scheduling and
cache tricks) failed while Round 3 (shrinking the weights) succeeded:

| Quantity | Value | Source |
|---|---|---|
| Bytes moved per decoded token (FP16 text tower) | 3.36 GB | measured |
| Decode time per token at that point | 43.29 ms | measured |
| Implied achieved bandwidth | 77.6 GB/s | 3.36 GB ÷ 43.29 ms |
| Orin Nano Super peak LPDDR5 bandwidth | ~102 GB/s | platform spec |
| Fraction of theoretical peak | 76% | 77.6 ÷ 102 |

At 76% of theoretical peak memory bandwidth there is no scheduling trick left to find. Context-cache
reuse bought −3.5% and greedy decode bought −4.3%, both below the improvement bar, because neither
one reduces the bytes that have to cross the memory bus. The only levers that move a number like
that are (a) raising the bus clock, which is what EMC 2133 → 3199 MHz does, and (b) making the
weights smaller, which is what the INT4 W4A16 round did (3.135 GB → 0.818 GB of engine on disk).

Those two levers compose. INT4 cut the resident weights by 3.88× (TensorRT-reported weights memory,
3,355,696,384 B → 865,480,704 B), and the EMC clock sets how
fast the remaining bytes move. Leaving the memory controller at its default clock throws away part
of the quantization win you paid for in weight error.

The GPU core clock is not irrelevant — the vision tower *is* compute-bound (the ViT achieved
~8.9 TFLOPS against a ~16.7 TFLOPS dense FP16 peak, about 53% of peak, during a 248 ms encode) — but
on a VLM workload where decode dominates total latency, EMC is where the leverage is.

## Applying and verifying

Run these on the Jetson. If you are driving it over SSH, reference it through the same environment
variable the measurement scripts use (`JETSON_HOST`, default `orin@jetson.local`):

```bash
export JETSON_HOST=orin@jetson.local

ssh "$JETSON_HOST" 'sudo nvpmodel -m 2 && sudo jetson_clocks'
```

Verify the mode actually took effect. `nvpmodel -m` will silently keep the previous mode if the
requested mode ID does not exist in `/etc/nvpmodel.conf` for your board, and mode IDs are **not**
portable between Jetson modules — mode 2 is MAXN_SUPER on this Orin Nano Super, and you should
confirm the name rather than trusting the number:

```bash
# Which mode is active, by name and ID
ssh "$JETSON_HOST" 'sudo nvpmodel -q'

# Which modes this board actually offers
ssh "$JETSON_HOST" 'grep -E "^< POWER_MODEL" /etc/nvpmodel.conf'

# Current vs min vs max for every pinned clock domain
ssh "$JETSON_HOST" 'sudo jetson_clocks --show'
```

Expect `nvpmodel -q` to name `MAXN_SUPER`, and `jetson_clocks --show` to report GPU at 1020 MHz and
EMC at 3199 MHz with current equal to max for each domain. If current is below max on EMC, the
profile is set but the clocks are not pinned — re-run `jetson_clocks`.

For a live view under load, `tegrastats` streams per-domain frequencies and utilisation while a
request is in flight. Its exact field names vary across L4T releases and were not re-verified for
this document beyond the clock values in the table above, so read it as a monitor, not as a contract.

A useful sanity check: run one of the measurement scripts before and after, not a synthetic loop.

```bash
python3 scripts/collect_perf.py "10 min ago"
```

See [../scripts/collect_perf.py](../scripts/collect_perf.py). Note the measurement traps documented
in [../docs/report.md](../docs/report.md) — in particular, the runtime caches vision-encoder
embeddings keyed on raw pixel bytes, so any probe that reuses the same image skips ~250 ms of real
ViT work and reports a fantasy number.

## Making it persist across reboot

The two settings do **not** behave the same way:

| Setting | Survives reboot? | Notes |
|---|---|---|
| `nvpmodel -m 2` | Yes | The selection is written to persistent configuration and re-applied at boot. |
| `jetson_clocks` | **No** | Clocks return to governor control on every boot. |

So `jetson_clocks` needs a boot-time hook. A minimal systemd unit, modelled on the two units this
repo ships in [../systemd/](../systemd):

```ini
[Unit]
Description=Pin Jetson clocks to the active nvpmodel ceiling
After=nvpmodel.service
Wants=nvpmodel.service

[Service]
Type=oneshot
ExecStart=/usr/bin/jetson_clocks
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

Write it to the device, then enable it — `systemctl enable` fails with *"Unit file
jetson-clocks.service does not exist"* if the file was never created:

```bash
ssh "$JETSON_HOST" 'sudo tee /etc/systemd/system/jetson-clocks.service >/dev/null' <<'UNIT'
[Unit]
Description=Pin Jetson clocks to the active nvpmodel ceiling
After=nvpmodel.service
Wants=nvpmodel.service

[Service]
Type=oneshot
ExecStart=/usr/bin/jetson_clocks
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
UNIT

ssh "$JETSON_HOST" 'sudo systemctl daemon-reload && sudo systemctl enable --now jetson-clocks.service'
```

This unit is an **example**, not one of the artifacts deployed and measured in this project — the
repo ships only `cosmos3-edge-shim.service` and `live-vlm-webui.service`. Check the unit name of the
`nvpmodel` service on your image before relying on the ordering dependency; if it differs, the
`After=` line is a no-op and the unit may run before the power mode is applied, in which case
`jetson_clocks` pins to the *old* ceiling.

To undo everything, pick a lower-power mode and let the governors take over again:

```bash
ssh "$JETSON_HOST" 'sudo jetson_clocks --restore || true; sudo nvpmodel -m 0; sudo reboot'
```

`jetson_clocks --store` / `--restore` save and replay the pre-pin clock state; storing before you
first pin is the clean way to get an exact revert. A reboot is the reliable fallback.

## The cost: power and heat

MAXN_SUPER is, by construction, the mode with no power cap. Pinning clocks on top of it means the
board draws its maximum whether or not a request is in flight — there is no idle ramp-down any more.
Three practical consequences:

- **Power supply.** Use a supply that meets the module's MAXN_SUPER rating with margin. An
  undersized or marginal USB-C supply typically manifests as brownout resets under load, which look
  like random crashes of the inference process rather than like a power problem.
- **Cooling.** Sustained clocks mean sustained thermals. If the module throttles, your measurements
  become a function of ambient temperature and enclosure airflow, and a long benchmark run will
  drift slower than a short one. Active cooling is the safe choice for a board that serves
  continuously.
- **Duty cycle.** For a board that handles occasional requests, the default profile with DVFS may be
  the better engineering tradeoff. Pinned clocks pay a constant power bill to remove a per-request
  ramp-up latency.

**Not verified in this deployment:** no power draw, junction temperature, or thermal-throttling
measurements were recorded. The above is the standard tradeoff, stated so you budget for it — not a
measurement from this board.

## Swap

This deployment ran with a 2 GB swapfile on the NVMe SSD alongside the 915 GB root filesystem.

Be clear about what swap does and does not do on a Jetson:

- It **does** help with host-side CPU memory peaks, which is a real problem here: TensorRT engine
  builds must happen on the target device, and the INT4 build peaked at **3,884 MiB** of CPU memory.
  On an 8 GB board with a model and a desktop session also resident, that peak is what pushes you
  into the OOM killer mid-build.
- It **does not** give you more room for the model. Orin memory is unified — the GPU and CPU share
  the same 8 GB — and engine weights are pinned device memory. If a model does not fit, swap will
  not make it fit; it will make the failure slower and harder to diagnose. A GEN engine for
  Cosmos3-Edge-Policy-DROID OOMed on this 8 GB board with nothing else resident, and no amount of
  swap changes that.
- A resident inference process that is actively swapping is a broken deployment, not a tuned one.
  Decode reads the whole weight set per token; paging any of it to disk is catastrophic, not slow.

Check what you have:

```bash
ssh "$JETSON_HOST" 'swapon --show; free -m'
```

Create a 2 GB swapfile on NVMe if there is none (adjust the path to your NVMe mount):

```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab   # persist across reboot
```

Put the swapfile on the NVMe SSD, not on a microSD card. The `/etc/fstab` line is what makes it
survive a reboot; `swapon` alone does not.

## Checking memory headroom before you load a model

On Orin there is no separate VRAM pool to inspect. `free -m` is the whole budget — CPU allocations,
GPU allocations, page cache, and your desktop session all come out of the same 8 GB.

```bash
ssh "$JETSON_HOST" 'free -m'
```

What this project measured, as a calibration point for what "enough headroom" looks like:

| Configuration | Resident process RSS | System RAM available afterwards |
|---|---|---|
| FP16 text tower | 6.02 GB | 472 MB |
| INT4 W4A16 text tower | 3.70 GB | 2,705 MB |

The FP16 row is the instructive one. 472 MB available is technically "working", and it is also a
board that OOMs the moment you add a second workload, open a browser on the desktop session, or
start an engine build. The INT4 row is what makes the same board able to co-host other services.

Practical headroom guidance, grounded in what was hit on this hardware:

| Activity | Free RAM you want before starting | Why |
|---|---|---|
| Loading the INT4 serving engine | ≥ 4.0 GB | Process settles at 3.70 GB RSS; leave room for the load transient. |
| Loading the FP16 serving engine | ≥ 6.5 GB | Process settles at 6.02 GB RSS; this is the configuration that left 472 MB. |
| Building an engine on device | ≥ 4.0 GB of *CPU-side* room | Measured peak build memory 3,884 MiB. Stop the serving shim first. |

Two failure modes worth recognising, both observed here:

- **OOM at engine load, not at build time.** An engine built with `--maxBatchSize 4` and
  `--maxKVCacheCapacity 4096` built successfully and then OOMed when the runtime tried to load it,
  because the KV cache is allocated at load. It was rebuilt at `--maxBatchSize 1
  --maxKVCacheCapacity 2048`. If you see a clean build followed by a load failure, suspect the KV
  cache sizing, not the weights.
- **A model that simply does not fit.** Cosmos3-Edge-Policy-DROID was non-viable on 8 GB regardless
  of tuning. Platform tuning changes speed, not capacity.

Stop the serving process before a build so the two peaks do not overlap:

```bash
ssh "$JETSON_HOST" 'sudo systemctl stop cosmos3-edge-shim.service && free -m'
```

See [../systemd/cosmos3-edge-shim.service](../systemd/cosmos3-edge-shim.service) for the unit as
deployed, and [../serve/cosmos3_shim.py](../serve/cosmos3_shim.py) for the resident runtime it
starts.

## Summary

| Action | Effort | Effect | Risk |
|---|---|---|---|
| `nvpmodel -m 2` | one command, persists | Raises the clock ceiling; EMC 2133 → 3199 MHz is the one that matters for decode | Higher power draw and heat |
| `jetson_clocks` | one command, needs a boot unit | Removes DVFS ramp; pins GPU 1020 MHz / EMC 3199 MHz | Constant power draw at idle |
| 2 GB swapfile | one-time setup | Absorbs the 3,884 MiB CPU peak of on-device engine builds | None if the serving process never swaps |
| `free -m` before load | free | Catches the "built fine, OOMs at load" class of failure early | None |

Everything above is free in the sense that it costs no model quality. It is the correct first step,
and it is also not sufficient on its own: on this board the change that actually moved decode from
44.43 ms/token to 13.13 ms/token was shrinking the weights, not raising the clocks. Platform tuning
makes the bandwidth-bound measurement honest so that you can see that.
