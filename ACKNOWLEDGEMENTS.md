# Acknowledgements

This deployment is assembled almost entirely from NVIDIA work done by other people. What follows
records, as precisely as the evidence supports, which project contributed what — including the
cases where a project or an artifact was deliberately *not* used the way it is normally used.

Contributor lists are taken from the public repositories (`git log` and the GitHub contributors
API) on 2026-09-14, ordered by commit count. Names are the ones those contributors publish on their
own GitHub profiles. A blank cell means the profile publishes a handle, an org name, or no name at
all — the handle is the credit in those rows, and nobody's name is reconstructed from commit
metadata they did not choose to publish.

---

## Runtime — `NVIDIA/TensorRT-Edge-LLM`

<https://github.com/NVIDIA/TensorRT-Edge-LLM> · Apache-2.0 · version 0.10.1

*"High-performance, light-weight C++ LLM and VLM Inference Software for Physical AI."*

**This project is the substrate the whole optimization campaign runs on.** Round 1 is purely using
its API correctly; Rounds 3 and 4 are this project's own work landing on primitives this runtime
provides:

| What we did | What the runtime provided |
|---|---|
| Round 1 — stop paying startup costs (13.91 s → 2.07 s, bundled with the platform clock change) | `LLMRuntime` held resident, plus `capture_decoding_cuda_graph()` to remove per-token kernel launch overhead |
| Round 3 — INT4 self-quantization (43.29 → 13.38 ms/token) | the W4A16 kernel and its on-disk checkpoint contract, including the `out_features % 64 == 0 && in_features % 64 == 0` alignment rule — layers that miss it are *silently* left FP16, which is what caps the ViT MLP (1152 ↔ 4304, `4304 % 64 == 16`) |
| Round 4 — right-sizing the image token budget | `max_image_tokens_per_image` being read **at runtime** from `visual/config.json`, so the vision tower could be re-tuned without a rebuild |

### Which release introduced what

`NVIDIA/TensorRT-Edge-LLM` publishes **squashed release drops** — 26 commits for the whole project,
each one an entire release — so there is no granular feature commit to cite. The finest citable
unit is the release. Traced with `git log --reverse -S<symbol>` against the cloned repo at v0.10.1:

| Capability used here | First shipped in | Commit |
|---|---|---|
| `capture_decoding_cuda_graph()` | 0.5.0 (MLPerf Inference) | `fc577b8` |
| W4A16 kernel + on-disk checkpoint contract | 0.4.0 | `4e97b81` |
| `max_image_tokens_per_image` (read at runtime) | 0.4.0 | `4e97b81` |
| `maxKVCacheCapacity` | 0.4.0 | `4e97b81` |
| `EDGELLM_PLUGIN_PATH` | 0.4.0 | `4e97b81` |
| `ContextCacheConfig` | 0.9.1 (MLPerf Inference) | `bb8a1d1` |

Deployed here: **v0.10.1**, merge `e8b2952`, PR #196, 2026-09-03. Release PRs across the project's
history were opened by `@nvluxiaoz` (#3, #34, #49, #53, #69, #196), `@nvxingkaiz` (#76, #90),
`@JCalafato` (#101), `@jhalabi-nv` (#122, #136) and `@ever-wong` (#171) — that is the authorship
the public history actually records, and it is release authorship, not a statement about who wrote
any particular feature.

Contributors:

| | GitHub | Name |
|---|---|---|
| 18 | [@nvluxiaoz](https://github.com/nvluxiaoz) | Luxiao Zheng |
| 2 | [@ever-wong](https://github.com/ever-wong) | Ever Wong |
| 2 | [@jhalabi-nv](https://github.com/jhalabi-nv) | |
| 2 | [@nvxingkaiz](https://github.com/nvxingkaiz) | Xingkai Zhou |
| 1 | [@JCalafato](https://github.com/JCalafato) | |
| 1 | [@poweiw](https://github.com/poweiw) | Po-Wei (Vincent) |

## Front end — `NVIDIA-AI-IOT/live-vlm-webui`

<https://github.com/NVIDIA-AI-IOT/live-vlm-webui> · Apache-2.0 · version 0.4.0

*"Real-time Vision Language Model interaction via webcam — WebRTC-based web interface."*

This is the interface in the demo recording, and it generated the live workload behind the latency
measurements. The shim's own `[perf]` log holds **5,900 requests all-time** across every client —
this WebUI and the Frigate event bridge both — and it is the **2,818-request quiesced-board
session** driven from this WebUI that the fit over generated-token count
(`elapsed_ms = 200 + 13.49 × generated_tokens`, r = 0.854) is built on, rather than a quoted
average.

**Effectively a one-person project** — 156 of 156 commits:

| | GitHub | Name |
|---|---|---|
| 156 | [@tokk-nv](https://github.com/tokk-nv) | Chitoku YATO, NVIDIA |

## Model — `nvidia/Cosmos3-Edge`

<https://huggingface.co/nvidia/Cosmos3-Edge> · NVIDIA Cosmos team

The 4B VLM every caption in this project comes from. Distributed under NVIDIA's own terms; the
weights are **not** redistributed in this repository — the quantizer operates on a checkpoint you
obtain yourself.

## Platform — JetPack 7.2.1 / L4T R39.2.1, CUDA 13.2, TensorRT 10.16.2.10

NVIDIA Jetson platform software. `nvpmodel -m 2` (MAXN_SUPER) and `jetson_clocks` — GPU 306 → 1020
MHz, EMC 2133 → 3199 MHz — were one of the three changes bundled into Round 1's −85.1%. Round 1
changed all three at once and they were never separated: the resident runtime is the dominant term,
because it removes a ~6–8 s engine deserialization from every request.

## Guidance — Jetson AI Lab

<https://www.jetson-ai-lab.com/tutorials/tensorrt-edge-llm/>

The Orin Nano TensorRT-Edge-LLM tutorial is the closest published engine-sizing guidance for this
class of device — `maxInputLen 512`, `maxKVCacheCapacity 1024`, and `--externalize-weights int4_ffn`
to cut peak build memory. This deployment did **not** start there: it built at `maxInputLen 1536` /
`maxKVCacheCapacity 2048`, and only the last NVR engine (v3) converged on the tutorial's
`maxKVCacheCapacity 1024`. The `--externalize-weights` lever was never measured here. Separately,
the Jetson AI Lab **Cosmos3-Edge model page's 44.1 tok/s on an AGX Orin 64 GB (BF16, vLLM)** is the
only external reference point this work had to orient against — the tutorial itself demonstrates
INT4 on Qwen3-4B, not on this model.

The tutorial and the Cosmos3-Edge model page are publications with no personal byline, so no
individual is credited for them here.

## `dusty-nv/jetson-containers`

<https://github.com/dusty-nv/jetson-containers>

Not a dependency of this deployment — the engines are built and the shim runs directly on the
device as systemd units, outside any container — and none of the guidance above is attributable to
it. Recorded as an acknowledgement of the ecosystem this work sits in, crediting its contributors
for that project alone. Contributors (7 of 8; `dependabot[bot]` omitted):

| | GitHub | Name |
|---|---|---|
| 2691 | [@dusty-nv](https://github.com/dusty-nv) | Dustin Franklin, NVIDIA |
| 1698 | [@johnnynunez](https://github.com/johnnynunez) | |
| 448 | [@tokk-nv](https://github.com/tokk-nv) | Chitoku YATO, NVIDIA |
| 243 | [@ms1design](https://github.com/ms1design) | |
| 100 | [@OriNachum](https://github.com/OriNachum) | |
| 51 | [@D-G-Dimitrov](https://github.com/D-G-Dimitrov) | |
| 7 | [@kbenkhaled](https://github.com/kbenkhaled) | |

---

## Three sources credited for something other than their main purpose

Recorded this way deliberately — crediting them for what they are normally used for would
misrepresent what happened here.

### `NVIDIA/TensorRT-Model-Optimizer` (modelopt) — used as a **format oracle**, not as the quantizer

<https://github.com/NVIDIA/TensorRT-Model-Optimizer> · Apache-2.0

modelopt did **not** quantize this model, for two independent reasons and only one of them is
modelopt's. Its CUDA path was broken on the sm_120 host used for export, producing weight files that
were the right *shape* but wrong. And its AWQ path needs real activations, which this toolchain
cannot supply: TensorRT-Edge-LLM's `attention_plugin` is a shape-only stub returning
`torch.zeros(...)` so the model can be traced to ONNX, so an eager forward yields no meaningful
activation statistics anywhere in this toolchain. The INT4 W4A16 quantization here is hand-written
round-to-nearest with an MSE-optimal clipping search.

What modelopt was genuinely used for is narrower and still load-bearing: run **on CPU**,
sidestepping the broken extension, it served as an **executable specification of the runtime's
on-disk W4A16 contract** — the scale convention (`scale = amax / 7`, clamp `[-8, 7]`), the packed
nibble layout, and the `weight_scale` companion tensor. The hand-written quantizer reproduces its
scales to **3.7e-09**, which is what turned "I think this is the layout" into a demonstrable match.
Development time only; nothing from modelopt ships in the deployed path.

The CUDA breakage above was specific to the sm_120 export host used here; it is not a claim about
the project generally, and nobody below touched that host. The credit is for publishing a format
contract precise enough to check a hand-written quantizer against. Top contributors:
[@kevalmorabia97](https://github.com/kevalmorabia97) (292),
[@cjluo-nv](https://github.com/cjluo-nv) (101), [@ajrasane](https://github.com/ajrasane) (55),
[@ChenhanYu](https://github.com/ChenhanYu) (53), [@h-guo18](https://github.com/h-guo18) (50),
[@Edwardf0t1](https://github.com/Edwardf0t1) (49).

### NVIDIA VSS and DeepStream — a reference this project reasoned **against**

The DeepStream/VSS principle of gating inference cheaply rather than running it on everything is
followed here directly, at two levels: motion gating (`contour_area`, `motion.threshold`,
`detect.fps`) keeps the CPU detector off idle frames, and the VLM runs only on completed Frigate
events. The principle is DeepStream's and VSS's; what implements it here is Frigate plus
`nvr/feed/alert_policy.py`, not DeepStream.

On frame counts this deployment **deliberately diverges** from [NVIDIA
VSS](https://docs.nvidia.com/vss/3.1.0/real-time-vlm.html), which samples ~8 frames per 30-second
chunk. VSS targets 8B+ models such as Cosmos-Reason2; on this 4B **INT4** model, feeding more
frames was measured three separate times to make captions *worse*, so it sends one frame per Ring
event and at most three on the Frigate clip path. That is not a defect in VSS guidance — it is a
model-scale difference, and having an explicit published number to diverge from is what made the
divergence measurable rather than arbitrary.

DeepStream 9.1 is installed on the device but was **not** in the measured inference path.

### `ubr-physical-ai/Cosmos3-Edge-INT4-AWQ` — read for layout, declined as an artifact

Not NVIDIA work: a community W4A16_AWQ group-128 quantization of this model, published for a Jetson
Orin Nano. Its safetensors **header metadata** — tensor names, shapes, dtypes — corroborated that
W4A16 at group size 128 is the form this model class converges on for this device, which was worth
knowing before spending a day on the route. Header bytes only; no weight data was downloaded,
loaded, or used.

It was **declined as a checkpoint** because its own documentation states it has never been executed,
and an unexecuted checkpoint is a hypothesis with a file attached. That is the same failure mode
modelopt's broken CUDA extension produced here: files of the right shape and dtype, and wrong.
Credit for the signal, not for the weights.

---

## Not NVIDIA

The NVR layer is community open source and deserves its own credit: [Frigate](https://frigate.video)
(@blakeblackshear and contributors), [ring-mqtt](https://github.com/tsightler/ring-mqtt)
(@tsightler), [go2rtc](https://github.com/AlexxIT/go2rtc) (@AlexxIT), and Eclipse Mosquitto. Home
Assistant is not part of this stack — it is referenced as the recommended path for the stateful
door-sensor case the VLM should not be asked to cover.

---

## Standing

Work by asotelo@nvidia.com. A personal engineering write-up of one deployment — not an official
NVIDIA product, release, or support commitment. Every project above is used under its own license;
nothing here implies endorsement by the teams credited.
