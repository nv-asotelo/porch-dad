# One isolated Cosmos3 INT4 AWQ candidate

**Prepared, not executed.** No auxiliary-host connection, package install, GPU operation, or calibration was performed for this audit. Run this candidate only after real FP16 image inference succeeds on the Orin, or an actual FP16 build/load/inference memory failure is saved. Remote GPU inference cannot satisfy the user's Orin deployment gate. This candidate is one entry in the existing 12-candidate maximum, with the unchanged stopping and quality rules in [GOAL.md](../GOAL.md).

## What the pinned source supports—and what remains a gate

The pinned `v0.10.1` quantizer accepts `--quantization int4_awq --lm_head_quantization int4_awq --dtype fp16 --device cuda --num_samples 32`. Leave visual, visual-MHA, and KV quantization unset: their exposed quantized choices are FP8, which is unsuitable for the Orin path. INT4 LM-head quantization must be explicitly enabled. There are **no CLI batch-size or calibration-length flags** in this pin. [CLI, lines 55–143](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/tensorrt_edgellm/scripts/quantize.py#L55), [quantization configuration, lines 245–263](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/tensorrt_edgellm/quantization/quantization_configs.py#L245)

Three material constraints prevent claiming this is already a working Cosmos3 AWQ pipeline:

1. The generic quantizer loads Transformers AutoModel/AutoProcessor classes. Its Cosmos3 builder/export implementations are separate and do not register a Cosmos3 Hugging Face calibration model. Our pinned snapshot declares `Cosmos3EdgeForConditionalGeneration` and `Cosmos3EdgeProcessor` but contains neither `auto_map` nor modeling Python. The original runner first checks that installed **Transformers 5.14.1** actually recognizes the model, multimodal model class, and processor. If recognition fails, stop: a correctly validated public Cosmos3 calibration adapter is additional engineering. Do not change `model_type` to a different model family. [Generic loader, lines 227–316](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/tensorrt_edgellm/quantization/quantize.py#L227), [separate Cosmos3 reasoner implementation](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/tensorrt_edgellm/models/cosmos3_reasoner/modeling_cosmos3_reasoner_text.py)
2. AWQ requires **image calibration even when vision stays FP16**. Upstream explicitly detects that text-only activation rescaling can damage image behavior, chooses the multimodal route, and caps it at 128 samples. The fallback text route uses batch 16 for AWQ. Our runner supplies local image/question pairs and rejects any text-only fallback. [Modality detection, lines 739–752](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/tensorrt_edgellm/quantization/quantize.py#L739), [calibration routing, lines 1107–1141](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/tensorrt_edgellm/quantization/quantize.py#L1107)
3. Upstream's visual exclusion globs include `visual` but omit plain **`projector`**. Cosmos3 stores separate `model.projector.linear_fc1/2` weights, both INT4-aligned. Our original wrapper appends `{"quantizer_name":"*projector.*","enable":false}` after upstream configuration construction, preserving vision **and projector** FP16. It also verifies exported visual/projector tensors are FP16 without quantization-scale tensors. This is a narrow local integration correction, not a change to the upstream checkout. [Exclusion prefixes, lines 50–59](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/tensorrt_edgellm/quantization/quantization_configs.py#L50)

The builder accepts ModelOpt `W4A16_AWQ`, but native Cosmos3 attention tensor aliases (`to_q`, etc.) and builder names (`q_proj`, etc.) are different. This family does not supply quantization-name normalization. The proposed candidate uses uniform language INT4; any selective language exclusions or later mixed-precision recipe require a separate alias audit. Export alone does not prove the checkpoint's exclusion metadata is consumed correctly by both builder components. [AWQ format](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/experimental/builder/core/quantization.py#L239), [Cosmos3 tensor aliases](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/experimental/builder/models/cosmos3/weights.py#L36)

## Prepared original files

- [awq_isolated_env.sh](../scripts/awq_isolated_env.sh) redirects known Python, pip, Hugging Face, Torch, CUDA, Triton, compiler, logging and temporary directories into the newly created task directory. It leaves `HOME` unchanged, disables implicit Hub authentication and network model/dataset access, disables user Python packages, and uses a task-local Git configuration.
- [quantize_cosmos3_awq.py](../scripts/quantize_cosmos3_awq.py) defaults to a CPU model/processor preflight. `--run` additionally requires actual baseline evidence, a task-local corpus with digests/provenance, the already verified model manifest, and a fresh candidate output path. It rechecks model SHA-256 values, verifies the 2,435,620,080-parameter reasoner, adds the projector exclusion, records CUDA allocator peaks, and leaves all outputs under the approved task directory.

These environment settings are **not an OS filesystem sandbox**. Python and CUDA must use installed executables, libraries, drivers and device interfaces. They prevent ordinary task caches from landing in unrelated user directories, but do not prove that every native library avoids every possible host path. The primary agent owns the exact host access/isolation boundary. Do not inspect other projects or host user data, reuse an unrelated environment, or silently relax the user's isolation constraint. No remote commands were run by this subagent.

Expected layout after the primary agent creates the approved directory and copies the pinned public inputs:

```text
<new-task-directory>/
  project/scripts/{awq_isolated_env.sh,quantize_cosmos3_awq.py}
  project/external/TensorRT-Edge-LLM/       # pinned complete checkout
  project/models/cosmos3-edge-reasoner/    # existing verified snapshot
  project/results/model-download.json
  project/sources.lock.json
  calibration/images.jsonl                # approved images and provenance
  calibration/images/...
  evidence/fp16-gate.json                 # only after a real device result
  evidence/<actual-result-or-log>
  venv/                                  # created here, never system install
  cache/ config/ data/ state/ run/ tmp/
```

After host access is delegated and the exact new directory is known, the primary agent can use the following setup **inside it**. Substitute only that authorized path; never search for existing environments. The `tools` extra follows the public upstream version pins and is intentionally separate from the Orin's lightweight serving environment. Pip's explicit cache argument remains effective with `--isolated`.

```bash
export COSMOS_AUX_DIR=/absolute/new-task-directory
source "$COSMOS_AUX_DIR/project/scripts/awq_isolated_env.sh" "$COSMOS_AUX_DIR"
python3 -m venv "$COSMOS_AUX_DIR/venv"
"$COSMOS_AUX_DIR/venv/bin/python" -m pip --isolated \
  --cache-dir "$COSMOS_AUX_DIR/cache/pip" install --no-input \
  --index-url https://pypi.org/simple \
  -e "$COSMOS_AUX_DIR/project/external/TensorRT-Edge-LLM[tools]"
"$COSMOS_AUX_DIR/venv/bin/python" -m pip --isolated freeze \
  > "$COSMOS_AUX_DIR/project/results/awq-python-packages.txt"
"$COSMOS_AUX_DIR/venv/bin/python" \
  "$COSMOS_AUX_DIR/project/scripts/quantize_cosmos3_awq.py" \
  --task-dir "$COSMOS_AUX_DIR"
```

Preflight failure is a material result; preserve `project/results/awq-preflight.json` and stop before GPU model load. A missing public dependency wheel, insufficient existing driver capability, or missing Cosmos3 registration is not permission to install system software or access other directories. [Pinned tools dependencies](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/pyproject.toml#L50)

## Calibration and the single candidate

Use **32 distinct representative image/question pairs**, batch one. They must be original task captures or public images with suitable provenance and licensing; do not automatically pull the default MMMU validation set. Keep calibration disjoint from the fixed held-out evaluation suite. Tiny synthetic-only data can validate plumbing but cannot support image-quality conclusions.

Each JSONL row has `image` (relative to the manifest), `question`, `sha256`, `source`, and `license`. For example, the *schema* is:

```json
{"image":"images/example.jpg","question":"Describe the visible objects and their spatial relationships.","sha256":"<actual SHA-256>","source":"<original capture or public source URL>","license":"<actual license or original ownership>"}
```

No fabricated corpus or passing gate is supplied. The gate JSON must have either `fp16_image_grounding_passed: true` or `fp16_memory_failure_documented: true`, plus `evidence_file` pointing to the actual saved Orin result/log, relative to the gate file. Once these exist:

```bash
"$COSMOS_AUX_DIR/venv/bin/python" \
  "$COSMOS_AUX_DIR/project/scripts/quantize_cosmos3_awq.py" \
  --task-dir "$COSMOS_AUX_DIR" --run --samples 32 \
  --gate "$COSMOS_AUX_DIR/evidence/fp16-gate.json" \
  --images "$COSMOS_AUX_DIR/calibration/images.jsonl" \
  > "$COSMOS_AUX_DIR/project/results/awq-candidate-01.log" 2>&1
```

The wrapper bounds the input image's longest side to 256 pixels and rejects processor outputs over **768 total tokens** before calibration. Upstream materializes every processed batch in CPU memory; its generic multimodal `max_length` argument does not cap ordinary model inputs. The checks therefore matter even at batch one. The existing dense-AWQ export workaround is retained by calling the upstream pipeline; it skips a lossy ModelOpt resmoothing pass. [Loader, lines 150–224](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/tensorrt_edgellm/quantization/quantize.py#L150), [export workaround, lines 680–736](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/e8b29522938901f6df19ebeedd4b69bc8edbcd97/tensorrt_edgellm/quantization/quantize.py#L680)

Successful export produces `project/models/cosmos3-edge-awq-candidate-01` and a report explicitly marked **not validated on Orin**. Before transfer, audit the exported quantization metadata and tensor names against the Cosmos3 builder; after transfer, build on SM87 and evaluate the same real images, quality rubric, memory counters, and latency workload as FP16. Do not transfer CUDA engines built for the auxiliary GPU and assume they are portable to Orin.

## Memory estimate, not a benchmark

The original indexed reasoner contains **4,871,240,160 bytes (4.54 GiB)** of 16-bit weights, independent of its larger physical shard downloads. Calibration also needs activations, attention/logit tensors, quantizer statistics, candidate scales, temporary copies, CUDA context and allocator headroom. At the 768-token cap, a full FP16 `[1,768,131072]` logits tensor alone is **192 MiB**. One full extra 16-bit weight copy would add another **4.54 GiB**. These are arithmetic components, not a measured peak or guaranteed allocation pattern.

Treat a **16–24 GB GPU as a planning preference**, not a verified requirement or promise; actual architecture, free memory and ModelOpt behavior are still unknown. Save calibration/export peak allocated and reserved CUDA bytes separately from eventual Orin inference memory. Theoretical INT4-language-plus-head / FP16-vision/projector/embedding weight storage is roughly **2.19 GiB before scales and runtime overhead**; it does not establish that the 8 GB Orin meets the final sustained-run memory gate. See [weight evidence](../results/model-download-validation.json) and [backend estimates](backend-feasibility.md).
