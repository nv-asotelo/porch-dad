# Pokémon classifiers for Live Vision

Live Vision (`nvr/ui`) lists two public Pokémon classifiers beside its Cosmos3-Edge engines. Choose
one, and each frame, upload or sample image gets the most likely species, the next four, and a
saliency overlay over the picture. They run on the Orin as TensorRT FP16 engines, in
`pokemon-classifier.service` on loopback `:8094`. Selecting one leaves the shim and its engine
running, so porch-dad's captions carry on, and going back to Cosmos frees the classifier's memory.

A sample set of 189 labelled photos lets the page run the selected classifier over every image and
score it as the answers land.

## Which two, and why

All five candidates in brocktwo's
[classifier comparison](https://github.com/nv-asotelo/brocktwo/blob/codex/brocktwo/docs/pokemon-classifier-comparison.md)
were loaded at their pinned revisions and run, each with its own preprocessing, on the same 189
images: openly licensed real-world photos of 110 species (figures, plush toys, costumes, drawings,
Pokémon GO photos, crafts), 120 of them of the 61 Gen-1 species in the set. Near-duplicates of the
candidates' downloadable training images were removed first. Labels map to one species each, forms
collapsed; a species a model cannot name counts as wrong.

| Model (pinned source) | Labels | Top-1, all 189 | Top-5 | Top-1, 120 Gen-1 | Licence |
|---|---|---:|---:|---:|---|
| **Skshmjn ViT-B/16** ([9e8d54de](https://huggingface.co/skshmjn/Pokemon-classifier-gen9-1025/tree/9e8d54de136b99afc212322eae13ddc07a6fd779)) | 1,025 species | **67.7%** (128) | **82.0%** | 63.3% | Apache-2.0 |
| **Bierny DenseNet121** ([3bf528cf](https://huggingface.co/BiernyVR/pokemon-classifier-mobilenetv3/tree/3bf528cf0f6ab8464ea3bac0f70d84886e8dab25)) | 151, Gen 1 only | 40.7% (77) | 49.2% | **64.2%** | MIT, though its card also calls it non-commercial |
| JJMack ViT-B/16 ([b5955cf3](https://huggingface.co/JJMack/pokemon_gen1_9_classifier/tree/b5955cf36287b4c9019e3041c8c0b0becb91963e)) | 1,025 | 36.5% (69) | 47.6% | 43.3% | CC-BY-NC-SA-4.0 |
| Dima806 ViT-B/16 ([52d72780](https://huggingface.co/dima806/pokemons_1000_types_image_detection/tree/52d727808e21ec0cf553bfe488e581b5c503d0a0)) | 1,000 | 6.3% (12) | 11.6% | 4.2% | Apache-2.0 |
| Ram MobileNetV3-Large ([4bb229bc](https://github.com/ram-singhh/Poketwo-Auto-Catcher/tree/4bb229bc78a98d9da19634338c97298506a38e0f)) | 936 labels, 763 species | 4.8% (9) | 10.6% | 3.3% | MIT |

**Skshmjn** is the best by a wide margin: right on 65 images where Bierny is wrong against 14 the
other way, and ahead of every other candidate (exact McNemar p < 0.001 for each pair).
**Bierny** is second on all 189 images although it can only name Gen-1 species, and level with
Skshmjn on the Gen-1 images. Its lead over JJMack is not significant on all 189 (77 against 69,
p = 0.32); it is kept for its Gen-1 accuracy, its size (a 15.8 MB engine, against 173 MB for each
ViT on a board with little memory to spare) and JJMack's non-commercial share-alike licence.
Dima806 and Ram fail on photos of real objects: both were trained on game art and sprites.

189 images is a sample, not a benchmark: Skshmjn's 95% interval is 60.8-74.0%, Bierny's 34.0-47.9%.
The set was chosen by searching for each species by name, so it leans to well-known species and
photogenic objects, and its labels were checked by eye one image at a time.

## What the overlay shows

None of the five outputs 2D grounding - no boxes, points or masks - so the service always returns
`"boxes": []`. The overlay is saliency, the closest honest picture of where a whole-image classifier
found its evidence, and the page says so beside it:

- **Skshmjn: attention rollout** (Abnar and Zuidema, 2020): the 12 layers' attention, heads averaged
  and the identity added, multiplied through, from the class token to the 14×14 image patches. It
  shows where the model looked, whatever it decided.
- **Bierny: Grad-CAM** for the top species at the last convolution (7×7). Its head is global average
  pooling then one linear layer, so the channel weights are that species' row of the linear layer.

Both are computed inside the exported ONNX graph, so the Orin needs no PyTorch. Each map is scaled
to 0-1 and drawn over the picture the classifier was sent, brighter and more opaque where the
evidence is stronger.

## Files

| | |
|---|---|
| `classifier_service.py` | The service: `GET /health`, `GET /models`, `POST /load`, `POST /unload`, `POST /classify` (its docstring has the shapes). TensorRT on the Orin; `--backend onnxruntime` runs the same models from their ONNX files anywhere else |
| `models/<id>/meta.json` | Each model's preprocessing, output names, the species every output maps to, its pinned source and licence. The ONNX and engine files beside it are not in git |
| `export_models.py` | Workstation step: download both models at their pinned revisions, check their SHA-256, export the ONNX files with the saliency output, convert to FP16 |
| `build_engines.py` | Orin step: build each TensorRT engine from its ONNX file, with the TensorRT the service loads |
| `samples.json`, `fetch_samples.py` | The sample set: each photo's source, creator, licence and attribution, and the script that fetches and prepares them for Live Vision. The photos are not in git |
| `../systemd/pokemon-classifier.service` | The unit |

## Install

On a workstation with `torch`, `transformers`, `onnx`, `onnxruntime` and `huggingface_hub` (no
Hugging Face token is needed: both repositories are public):

```bash
python nvr/classifier/export_models.py          # writes models/<id>/model_fp16.onnx beside meta.json
scp -r nvr/classifier/models orin@<orin>:/home/orin/nvr/classifier/
scp nvr/classifier/{classifier_service.py,build_engines.py,fetch_samples.py,samples.json} orin@<orin>:/home/orin/nvr/classifier/
```

On the Orin, as `orin`:

```bash
cd /home/orin/nvr/classifier
/home/orin/TensorRT-Edge-LLM/.venv/bin/python build_engines.py      # see "Building the engines"
/home/orin/TensorRT-Edge-LLM/.venv/bin/python fetch_samples.py /home/orin/nvr/classifier/samples
sudo install -m 0644 ~/porch-dad/nvr/systemd/pokemon-classifier.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now pokemon-classifier
curl -s http://127.0.0.1:8094/models | python3 -m json.tool | grep -E '"id"|installed'
```

Then Live Vision needs the two classifier entries in its registry
(`nvr/ui/config/engines.orin.json` → `/home/orin/nvr/ui/config/engines.json`) and
`--classifier-url http://127.0.0.1:8094 --samples-dir /home/orin/nvr/classifier/samples`, which
the drop-in `nvr/systemd/dropins/cosmos-edge-ui.service.d-zz-live-vision-demo.conf` carries
([`../README.md`](../README.md), "Live Vision").

### Building the engines

An engine only loads in the TensorRT that built it, on the GPU it was built for, so it is built on
the Orin, by `build_engines.py` with the venv's TensorRT. It aborts if MemAvailable falls below
150 MB or free swap below 300 MB, before the OOM killer picks a victim, whose first pick would be
the shim.

Bierny's engine builds beside the full NVR (273 s, at TensorRT's default optimization level).
Skshmjn's does not: a ViT-B/16 build needs about 1.5 GB more than the board has free in full
porch-dad mode. Unguarded, it held MemAvailable near 270 MB with swap full for 12 minutes; guarded,
at optimization levels 0 and 1, and with `vm.swappiness` raised for the build, it stopped itself
every time. With the shim stopped it takes 28 s at level 1, the engine the Orin runs, so build it
in a short window - porch-dad's captions pause, nothing else does:

```bash
cd /home/orin/nvr/classifier
sudo systemctl stop cosmos3-edge-shim            # porch-dad stops with it (Requires=)
/home/orin/TensorRT-Edge-LLM/.venv/bin/python build_engines.py skshmjn --level 1
sudo systemctl start porch-dad                   # starts the shim first (After=, Requires=)
```

On 2026-10-03 that window lasted 76 s, the shim's reload included.

## Measured on the Orin

2026-10-03, beside the full NVR with the shim on its default engine, through the service and
through Live Vision:

| | Bierny DenseNet121 | Skshmjn ViT-B/16 |
|---|---:|---:|
| Engine file | 15.8 MB | 174.5 MB |
| GPU memory while loaded (nvmap's ledger) | 106 MB | 267 MB |
| Load: first, then again | 0.4 s | 3.7 s, then 0.4 s |
| Inference, median | 13.6 ms | 14.9 ms |
| Decoding and preprocessing, median | 4.1 ms | 3.8 ms |
| All 189 samples through Live Vision's "Run" | 7 s | 7 s |
| Top-1 on the samples, TensorRT FP16 | 77/189 (40.7%) | 127/189 (67.2%) |

The scores match the FP32 reference above, except one Skshmjn image FP16 tips the other way.
Unloaded, the service holds 128 KB of GPU memory and about 45 MB of RAM. Its cgroup keeps showing
kernel memory charged by the last model (nvmap's page pool), but restarting it frees only those
45 MB, so that charge is not memory the NVR is missing. GPU memory does count against the unit's
`MemoryMax`: at 512M every Skshmjn load hit the cap and took 10 s of reclaim, hence 1G.
