*Cosmos3-Edge running real-time on a Jetson Orin Nano Super 8 GB dev kit*

I put `nvidia/Cosmos3-Edge` on the entry-level board in the Orin line, quantized it to *INT4* myself, and pointed it at my house. The quantizer and the optimization rounds are mine; the runtime, the front end, the model and the platform are other people's, and that is most of the stack.

*Real numbers, straight off the Live VLM WebUI while it streams*

• Latency *361 ms*, Avg *500 ms*
• System RAM *4.8 / 7.4 GB* — whole model resident, ~2.6 GB still free
• *~42 tokens/sec* typical, *~72 tokens/sec* best observed on a long generation
• *15% CPU* — system-wide across all 6 Cortex-A78AE cores, because the work is GPU-bound
• Fastest single caption measured: *245 ms*; median under live streaming on a quiesced board: *442 ms*

*The INT4 quantization is the whole story*

Single-stream decode on this model is *memory-bandwidth-bound*. At FP16 it moved 3.36 GB per token in 43.29 ms — about 77.6 GB/s against this board's ~102 GB/s ceiling. At 76% of theoretical peak there is no scheduling trick left; the only lever is making the weights smaller.

So: *INT4 W4A16 weight-only*, self-quantized from the official checkpoint, round-to-nearest with an MSE-optimal clipping search, calibration-free. *43.29 → 13.38 ms/token*, LLM engine *3.135 GB → 0.818 GB* (3.83× smaller). That is what makes it fit in 8 GB at all.

*The repos, and specifically what I used from each*

*`NVIDIA/TensorRT-Edge-LLM`* — deployed at v0.10.1 (`e8b2952`, PR #196). Heads-up for anyone tracing this: the repo publishes *squashed release drops*, 26 commits total, so the finest citable unit is the release, not a feature commit. The four things I leaned on, traced to the release that introduced each:

• `capture_decoding_cuda_graph()` — first in *0.5.0*. Removes per-token kernel launch overhead. Part of the change that took per-request latency 13.91 s → 2.07 s, bundled with holding `LLMRuntime` resident and the platform clock change — I never separated the three, so please don't read that delta as any one of them.
• the *W4A16 kernel and its on-disk checkpoint contract* — first in *0.4.0*. What my quantizer rides on. Its constraint is the interesting part: a layer is quantizable only if `out_features % 64 == 0` *and* `in_features % 64 == 0`. Cosmos3-Edge's ViT MLP is 1152↔4304 and 4304 % 64 = 16, so those layers cannot be INT4 and stay FP16 — which is exactly why the vision tower became the next bottleneck.
• `max_image_tokens_per_image` — first in *0.4.0*, and critically read *at runtime* from the engine's `visual/config.json`. That let me re-tune the image token budget 512 → 320 with no rebuild.
• `maxKVCacheCapacity` and `EDGELLM_PLUGIN_PATH` — both *0.4.0*. The plugin path defaults to a *relative* directory, which is an easy way to lose an afternoon.

*`NVIDIA-AI-IOT/live-vlm-webui`* v0.4.0 — the front end in the demo. WebRTC webcam capture plus an OpenAI-compatible client (`--api-base` / `--model`), which is why it dropped onto my shim with zero glue. Effectively a one-person project: 156 of 156 commits by `@tokk-nv`. Highest leverage-per-line dependency in the stack.

*`NVIDIA/TensorRT-Model-Optimizer`* — credited for something other than its job. It did *not* quantize this model, for two independent reasons and only one is modelopt's: its CUDA path was broken on my sm_120 export host and emitted weights of the right shape with wrong values, and its AWQ path needs real activation statistics this toolchain cannot supply (the attention plugin is a shape-only stub returning `torch.zeros(...)` for ONNX export). What it *did* provide, and this was load-bearing: run on CPU to sidestep the broken extension, it is an executable spec of the runtime's on-disk W4A16 contract — packed nibble layout and `weight_scale` tensors — and my quantizer matches its scales to *3.7e-09*. I checked against a reference instead of against my own assumptions. Dev-time only; nothing from it ships.

*`nvidia/Cosmos3-Edge`* — the model. Every caption comes out of this 4B VLM. Weights under NVIDIA's own terms and not redistributed; my quantizer runs on a checkpoint you fetch yourself.

*JetPack 7.2.1 / L4T R39.2.1 / CUDA 13.2 / TensorRT 10.16.2.10* — `nvpmodel -m 2` (MAXN_SUPER) and `jetson_clocks` move GPU 306 → 1020 MHz and EMC 2133 → 3199 MHz.

*Jetson AI Lab* — the Orin Nano TensorRT-Edge-LLM tutorial and the Cosmos3-Edge model page were the only external material I had to orient against, and I should be straight that I did *not* follow the tutorial's sizing: it suggests `maxInputLen 512` / `maxKVCacheCapacity 1024`, I built at 1536 / 2048, and only the final NVR engine landed on KV 1024.

*One framing note before anyone quotes this*

Jetson AI Lab publishes 44.1 tok/s for this model on an *AGX Orin 64 GB* at BF16 via vLLM. My best is 71.9 tok/s on an 8 GB Nano — but that is *not* apples-to-apples: different precision, different runtime, different device. And 71.9 is the single fastest of 5,900 logged requests, not a typical one; the all-time median is ~42 tok/s, which is *below* the published AGX figure. The claim is not "faster than an AGX." It is narrower and more useful: the cheap 8 GB board is not disqualified from this model, which is not obvious from the published material — there are no Orin Nano numbers published for Cosmos3-Edge at all. INT4 is what buys the decode rate, and it carries a quality cost I have not fully benchmarked.

*On names* — I have not resolved every contributor and I am not going to guess. The runtime's release PRs are authored by `@nvluxiaoz`, `@nvxingkaiz`, `@JCalafato`, `@jhalabi-nv`, `@ever-wong`, `@poweiw` (GitHub handles, not Slack). If I have mis-scoped who owns what, point me at the right teams and I will correct it.

Writeup, measurements and the INT4 quantizer: https://github.com/nv-asotelo/porch-dad

Personal engineering write-up of one deployment — not an official NVIDIA product, release, or support commitment, and nothing here implies endorsement by the teams named.
