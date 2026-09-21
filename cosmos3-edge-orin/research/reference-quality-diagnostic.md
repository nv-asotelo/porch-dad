# Official reference for the remaining shape-label failure

On 2026-09-20 UTC, one bounded reference run reproduced the corrected Orin's case-05 answer **verbatim** using the official public Transformers Cosmos3-Edge implementation and the unchanged NVIDIA checkpoint:

> Both objects are purple. The left object, which is a sphere, is larger than the right object, which is a smaller sphere.

This is the task's actual model output, not a quoted source passage. The fixture depicts flat circles. The frozen quality screen still treats the sphere label as incorrect; the reference does not convert an 18/19 result into a 19/19 pass. It establishes that the remaining observed wording error also occurs in the official model path, rather than supplying evidence of a remaining TensorRT-specific defect. Matching text on one request does not establish numerical parity or general model accuracy.

## Controlled comparison

| Field | Orin / TensorRT-Edge-LLM | Mac / official Transformers |
| --- | --- | --- |
| Checkpoint | NVIDIA pin `344d602b128d1bbdacb43b08d0a3626f46343e29` | Same local checkpoint, read-only |
| Image | Original frozen `05-relative-size.jpg` | Same bytes; SHA-256 `f1e76656b44bd8cbd9c25f98b9915d9ca9f0916127f81aeebc97aa64f420ae6d` |
| Prompt | “What shape and color do both objects share? Is the left object or the right object larger?” | Identical; text before image; no added system instruction |
| Generation | Temperature zero, 64-token cap, thinking disabled | Greedy `do_sample=False`, 64-token cap, thinking disabled |
| Prompt tokens | 294 | 294; image grid `[1,32,32]` |
| Completion tokens | 28 | 28 |
| Answer | Sphere wording above | Identical text |

Reference implementation: [Hugging Face Transformers pin `c587bc884db2c2e31fc2b8102314656b17aa07b1`](https://github.com/huggingface/transformers/tree/c587bc884db2c2e31fc2b8102314656b17aa07b1), credited to NVIDIA and Hugging Face under Apache-2.0. The pin includes official [checkpoint-name conversion mappings](https://github.com/huggingface/transformers/blob/c587bc884db2c2e31fc2b8102314656b17aa07b1/src/transformers/conversion_mapping.py#L163), so this test required no custom tensor conversion or model-source patch. Loading reported zero missing, unexpected or mismatched keys.

## Environment and limits

The isolated `.qa/reference-venv` used Torch 2.13.0, torchvision 0.28.0 and Transformers 5.18.0.dev0 from the immutable source pin on an Apple M3 Pro with 36 GiB RAM. MPS availability and arithmetic were verified before loading. Inference used FP16, SDPA, four CPU threads, and an offline checkpoint/cache policy. MPS's supported device checks follow the [official PyTorch documentation](https://docs.pytorch.org/docs/2.14/notes/mps.html). CPU fallback was enabled for unsupported MPS operators; no such warning appeared in the successful log.

The bounded task began at 06:18:43 UTC, with a 15-minute deadline. There was one completed generation. An initial attempt loaded and preprocessed the model but stopped before generation because the task's receipt writer could not serialize a set in Transformers loading metadata. Fixing that logger required no model compatibility change. The successful process exited after 9.07 seconds; generation took 3.34 seconds. This was a diagnostic with a different backend and cold state, **not an Orin performance comparison**.

The successful process recorded maximum RSS of 10,093,281,280 bytes, MPS allocated memory of approximately 4.87 GB, and post-generation MPS driver memory of approximately 6.23 GB. These are overlapping counters and must not be added. No reference model process remains running. No GPU-host or Orin access was used for this diagnostic.

## Evidence

- [Reference output, expanded prompt IDs, timing and memory](../results/reference-official-05.json)
- [Exact image-bearing request](../results/reference-official-05-request.json)
- [Machine-readable comparison](../results/reference-official-05-comparison.json)
- [Environment, source hashes, source-archive provenance and limitations](../results/reference-official-05-environment.json)
- [Exact installed packages](../results/reference-official-05-packages.txt)
- [Successful execution log](../results/reference-official-05-output.log)
- [Original corrected Orin responses](../results/raw/quality-fp16-corrected-01.jsonl)

The task's work here is the controlled diagnostic and evidence capture. Model behavior, architecture, weights and reference implementation remain the credited upstream contributions.
