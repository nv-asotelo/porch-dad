# Backend preflight source audit

2026-09-19. Inspected only the local public NVIDIA TensorRT-Edge-LLM v0.10.1 checkout, verified HEAD `e8b29522938901f6df19ebeedd4b69bc8edbcd97`, and this project's launcher/UI. No GPU code, model load, or server was executed. Conclusions establish source compatibility, not device success or numerical quality.

## Result

The original unquantized BF16 Cosmos3-Edge reasoner checkpoint follows an FP16 weights/activations and FP16 KV path with the current launcher. The one-image OpenAI request and SSE response contract match the UI. One material preflight gap was found and corrected: unsupported quantization nested under `text_config` or `vision_config` could bypass the original top-level-only check.

## Resident server and component selection

The launcher uses real, accepted v0.10.1 flags:

```text
python -m experimental.server /absolute/reasoner-checkpoint
  --host 127.0.0.1 --port 8000 --served-model-name Cosmos3-Edge
  --cache-dir /absolute/engine-cache --engine-cache-max-size-gb 12
  --max-input-len 1024 --max-kv-cache-capacity 2048
  --max-batch-size 1 --max-queued-requests 1 --queue-timeout 10
  --reasoning-parser none
```

This is a bounded initial configuration, not the smallest proven memory configuration. The server has no `--dtype`, `--kv-cache-dtype`, or image-token-limit CLI option. Do not add invented switches. Engine cache size is a **disk** limit. The current CLI retains builder defaults of 1024 total image tokens and 512 per image; a later measured smaller vision profile requires `LLM(build_options=BuildOptions(...))` or extending the server configuration. [Launcher, line117](/Users/asotelo/Documents/ChatGPT/jetson-orin-streaming-codes/scripts/run_backend.sh:117), [server parser, line275](/Users/asotelo/Documents/ChatGPT/jetson-orin-streaming-codes/external/TensorRT-Edge-LLM/experimental/server/config.py:275), [builder CLI defaults, line304](/Users/asotelo/Documents/ChatGPT/jetson-orin-streaming-codes/external/TensorRT-Edge-LLM/experimental/builder/cli.py:304)

The server forwards `--components all --externalize-weights all`. For Cosmos, component availability is determined by the configs physically present. Root text/vision configs select LLM+VISUAL. `transformer/config.json` adds generator and understanding-prefill; `vae/config.json` adds VAE. A reasoner snapshot must retain the shared shard files referenced by its root index while omitting those generator component configs. The launcher already rejects their presence. [BuildOptions, line65](/Users/asotelo/Documents/ChatGPT/jetson-orin-streaming-codes/external/TensorRT-Edge-LLM/experimental/server/runtime/engine_build.py:65), [Cosmos component detection, line46](/Users/asotelo/Documents/ChatGPT/jetson-orin-streaming-codes/external/TensorRT-Edge-LLM/experimental/builder/models/cosmos3/configuration.py:46)

The server loads its runtime once and captures decoding CUDA graphs at startup. It does not reload weights for every frame. Speculation is disabled by default and explicitly unsupported by the Cosmos builder. Context reuse is disabled by launch default, despite individual requests defaulting to allowing reuse. [Resident runtime, line789](/Users/asotelo/Documents/ChatGPT/jetson-orin-streaming-codes/external/TensorRT-Edge-LLM/experimental/server/runtime/engine.py:789), [Cosmos speculative guard, line105](/Users/asotelo/Documents/ChatGPT/jetson-orin-streaming-codes/external/TensorRT-Edge-LLM/experimental/builder/models/cosmos3/configuration.py:105), [context-cache default, line100](/Users/asotelo/Documents/ChatGPT/jetson-orin-streaming-codes/external/TensorRT-Edge-LLM/experimental/server/config.py:100)

## BF16 checkpoint conversion and unsupported precision

Absent quantization metadata, `QuantConfig` defaults to `fp16` with no KV quantization. Cosmos language input embeddings are FP16; KV inputs use FP16 unless effective metadata selects FP8. FP8 embeddings require the builder's explicit opt-in, which this server does not supply. This precision is selected by graph/weight contracts, not the model card's BF16 dtype string. [Quantization defaults, line50](/Users/asotelo/Documents/ChatGPT/jetson-orin-streaming-codes/external/TensorRT-Edge-LLM/experimental/builder/core/quantization.py:50), [Cosmos tensor types, line66](/Users/asotelo/Documents/ChatGPT/jetson-orin-streaming-codes/external/TensorRT-Edge-LLM/experimental/builder/models/cosmos3/modeling_cosmos3_reasoner_text.py:66), [FP8 embedding opt-in, line317](/Users/asotelo/Documents/ChatGPT/jetson-orin-streaming-codes/external/TensorRT-Edge-LLM/experimental/builder/cli.py:317)

BF16 is explicitly converted to FP16 on both paths:

- Baked constants: `get_f16()` decodes BF16 via FP32 and casts to NumPy FP16. [safetensors reader, line391](/Users/asotelo/Documents/ChatGPT/jetson-orin-streaming-codes/external/TensorRT-Edge-LLM/experimental/builder/core/safetensors_np.py:391)
- Externalized weights: FP16 parameter metadata is emitted, and native assembly detects BF16 sources for FP16 output bindings; the CUDA conversion uses BF16→float→half. [External FP16 weight contract, line617](/Users/asotelo/Documents/ChatGPT/jetson-orin-streaming-codes/external/TensorRT-Edge-LLM/experimental/builder/core/weights.py:617), [native cast detection, line169](/Users/asotelo/Documents/ChatGPT/jetson-orin-streaming-codes/external/TensorRT-Edge-LLM/cpp/runtime/weight/checkpointWeightAssemble.cpp:169), [conversion kernel, line81](/Users/asotelo/Documents/ChatGPT/jetson-orin-streaming-codes/external/TensorRT-Edge-LLM/cpp/kernels/weightsTransform/fp16/linear/fp16LayoutConvert.cu:81)

The corrected launcher mirrors metadata precedence: `hf_quant_config.json` overrides all embedded configurations; otherwise each text/vision component uses its nonempty `quantization_config`, then the root fallback. It checks effective metadata recursively for FP8/FP4 families, rejects nonempty `kv_cache_scheme` (the upstream parser can map that to FP8 without a literal FP8 value), and rejects nonempty KV algorithms other than FP16. Legitimate INT4/AWQ/GPTQ and INT8 metadata can pass to upstream validation/calibration. This guard does not assert that every quantized checkpoint is otherwise valid. [Upstream precedence, line91](/Users/asotelo/Documents/ChatGPT/jetson-orin-streaming-codes/external/TensorRT-Edge-LLM/experimental/builder/core/quantization.py:91), [embedded KV selection, line190](/Users/asotelo/Documents/ChatGPT/jetson-orin-streaming-codes/external/TensorRT-Edge-LLM/experimental/builder/core/quantization.py:190), [corrected guard, line63](/Users/asotelo/Documents/ChatGPT/jetson-orin-streaming-codes/scripts/run_backend.sh:63)

For the first run, use the original pinned unquantized reasoner. After successful load, require `/health/ready` to report `capabilities.kv_cache_dtype = fp16`, image input support, batch1, and speculation disabled. Inspect generated engine metadata before recording the baseline. [Health capability fields, line45](/Users/asotelo/Documents/ChatGPT/jetson-orin-streaming-codes/external/TensorRT-Edge-LLM/experimental/server/api/routes.py:45)

## Current image and streaming request

The UI submits one user message with text plus `image_url.url = data:image/jpeg;base64,...`, model name, `temperature:0`, `top_p:1`, `max_tokens`, `stream:true`, and `stream_options.include_usage:true`. All are accepted by the server schema. The server decodes the data URL, loads bytes as an image, and creates a matching image placeholder in the message. No local-media allowance is needed for data URLs. [UI payload, line120](/Users/asotelo/Documents/ChatGPT/jetson-orin-streaming-codes/web/app.js:120), [request schema, line51](/Users/asotelo/Documents/ChatGPT/jetson-orin-streaming-codes/external/TensorRT-Edge-LLM/experimental/server/api/protocol.py:51), [data-URL loader, line166](/Users/asotelo/Documents/ChatGPT/jetson-orin-streaming-codes/external/TensorRT-Edge-LLM/experimental/server/media/media_source.py:166), [image bytes, line1961](/Users/asotelo/Documents/ChatGPT/jetson-orin-streaming-codes/external/TensorRT-Edge-LLM/experimental/server/runtime/engine.py:1961)

The native multimodal dispatch includes Cosmos3EdgeViTRunner, which implements the Cosmos SigLIP2/PatchMerger path. Streaming emits content deltas, a terminal finish reason, optional real token usage, then `[DONE]`. `--reasoning-parser none` sends visible output through `delta.content`, matching the UI. That option disables output parsing; it is not a model-level guarantee that no reasoning tokens are generated. [Cosmos dispatch, line191](/Users/asotelo/Documents/ChatGPT/jetson-orin-streaming-codes/external/TensorRT-Edge-LLM/cpp/multimodal/common/multimodalRunner.cpp:191), [SSE handling, line415](/Users/asotelo/Documents/ChatGPT/jetson-orin-streaming-codes/external/TensorRT-Edge-LLM/experimental/server/api/serving_chat.py:415)

## Verification completed

`bash -n scripts/run_backend.sh` passed. Fourteen temporary offline fixtures executed the **actual embedded launcher preflight**, stopping before its TensorRT import:

- Accepted: original BF16/unquantized config; embedded INT4 AWQ; embedded INT8 SQ; ModelOpt INT4 with null KV; ModelOpt INT8 with explicit FP16 KV; both global and component precedence examples where inactive FP8 metadata is overridden.
- Rejected: nested text FP8; nested vision NVFP4; root MXFP8 fallback; nonempty nested KV scheme without an FP8 string; INT8 KV; ModelOpt FP8 KV; mixed-precision FP8 layer.

The only implementation change was the launcher metadata guard. Device build memory, FP16 numerical behavior, real image output, streaming cancellation, and sustained memory/latency remain hardware validation gates. No additional source blocker was found in this bounded audit.
