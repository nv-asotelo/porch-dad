# Peer VLMs against Cosmos3-Edge, on this Orin Nano, on this house's cameras

Question: which open VLM is the best to post-train and run on the Jetson Orin Nano 8 GB beside the
NVR - one that answers about as fast as Cosmos3-Edge (~500 ms), in about the same memory (<4 GB),
on frames from the Ring cameras, the Reachy Mini and the Scouts.

## Candidates, and why these

A research sweep (Jetson AI Lab, TensorRT-Edge-LLM's own Orin Nano tables, leaderboards, model
cards; 2026-10-04) narrowed the ~2B class to models with a real Orin Nano path:

| Model | Why | Runtime here |
|---|---|---|
| Cosmos3-Edge v3 (baseline) | what porch-dad runs today | TensorRT-Edge-LLM, INT4 text tower |
| Qwen3-VL-2B-Instruct | measured on Orin Nano by NVIDIA (36 tok/s); best-documented grounding and the most mature fine-tuning tooling; Apache-2.0 | TensorRT-Edge-LLM |
| Qwen3.5-2B | newest Apache-2.0 2B, natively multimodal, best card scores in the class; on NVIDIA's Orin Nano table | TensorRT-Edge-LLM |
| Cosmos-Reason2-2B | NVIDIA's physical-AI post-train of Qwen3-VL-2B: tells whether that post-training helps on these frames | TensorRT-Edge-LLM |
| InternVL3.5-2B | the strongest different architecture (InternViT + Qwen3-1.7B); Apache-2.0 | TensorRT-Edge-LLM |
| Gemma 4 E2B | the "Gemma E2B" Jetson AI Lab lists for Orin Nano | llama.cpp (Jetson AI Lab's path): TensorRT-Edge-LLM loads its 4.7 GB per-layer embedding table whole |
| LocateAnything-3B | NVIDIA's grounding model (in Nemotron 3 Nano Omni); non-commercial license | PyTorch, INT4 weight-only (no TensorRT path exists) |
| Nemotron 3 Nano 4B | requested; text only, so it gets Cosmos3-Edge's caption instead of the image | llama.cpp (Jetson AI Lab's path): no INT4 TensorRT build, 3136-wide projections fit no 128-group |
| Nemotron 3 Nano 30B-A3B | requested exploration; text only, 17 GB at its smallest | contained llama.cpp mmap test (explore_30b.sh) |

## Same recipe for every TensorRT model

Every TensorRT-Edge-LLM candidate was given Cosmos3-Edge's exact treatment, so differences are the
model's and not the recipe's: `scripts/rtn_int4_quantize.py` (RTN INT4 W4A16, group 128, MSE-optimal
clipping, lm_head quantized - tied ones untied), `tensorrt-edgellm-export`, then `build_engines.sh`
on the Orin with v3's limits (batch 1, 1024 input, 1024 KV, image tokens 64-640 with 320 per image;
InternVL 256:512:320 = one 448 px tile, 256 tokens). Mean INT4 reconstruction error: Cosmos3-Edge
11.06%, Qwen3-VL-2B 11.09%, Cosmos-Reason2-2B 11.09%, InternVL3.5-2B 11.08%, Qwen3.5-2B 11.44%.

## The frames and the ground truth

`collect.py` cut 122 frames from Frigate (seeded, stratified by camera, label and day/night): 80
event snapshots (the 640x360 frames Frigate's GenAI hands Cosmos3-Edge, with the detector's box) and
42 frames from recordings (960x540; the quiet hours, empty rooms, night). Two independent labellers
per frame, disagreements reconciled by a third (105 agreed, 17 reconciled): people / vehicles / dogs /
cats counts with per-question ambiguity flags, a reference caption and key facts, and a verdict on
the detector's box. Grounding is scored on the 41 frames with exactly one person whose detector box
was judged tight. The detector itself was wrong on 13 of its 80 boxes (a Charmander figure as a
person, an SUV as a person) - presence questions are scored against the labels, not the detector.
The frames show the household and are not in git.

## Tasks (bench.py)

Exactly the prompts, temperature 0, one image per request, streamed, timed next to the model:

| Task | Prompt |
|---|---|
| caption | Frigate's own GenAI prompt: "Describe only what is visible in this image, in one short sentence." |
| person / vehicle / animal | "Is there a person / car or other vehicle / dog or a cat in this image? Answer with only yes or no." |
| count | "How many people are visible in this image? Answer with only a number." |
| ground | JSON `bbox_2d` on a 0-1000 grid; InternVL and Gemma also get their own documented format |

Captions are judged blind (letters, shuffled per frame) against the image, the reference caption and
the key facts: 2 accurate, 1 partly right or vague, 0 wrong or hallucinated main content.

## Results

Measured 2026-10-04 on the Orin with the whole NVR running (Frigate recording every enabled camera,
detection on). Each model ran alone in the shim's slot under a 3.3 GB memory cap
(`run_trt_bench.sh`, `run_llama_bench.sh`), so an overrun kills the model, never Frigate; no model
in the speed table was OOM-killed. "Resident" is the model's own memory at the end of its run:
anonymous, GPU (NvMap, which Jetson's kernel books as kernel memory) and shared. It leaves out the
page cache that reading the engine files leaves behind. With 122 frames (110 for counting, 41 for
boxes), differences of a few points are noise. `report.py` renders these tables from
`score.py --json` and the caption judgments.

### Accuracy

| Model | Person (P / R) | Vehicle | Dog or cat | Count exact (MAE) | Box IoU>=0.5 (mean IoU) | Caption (0-2) | Hallucinated captions |
|---|---|---|---|---|---|---|---|
| Cosmos3-Edge v3 (today) | 95% (91% / 100%) | 100% | 99% | 65% (0.79) | 98% (0.80) | 1.64 | 30% |
| Cosmos3-Edge v2 | 90% (86% / 98%) | 99% | 97% | 55% (1.16) | 93% (0.73) | 1.57 | 34% |
| Qwen3-VL-2B | 88% (83% / 100%) | 100% | 95% | 83% (0.25) | 85% (0.68) | 1.86 | 16% |
| Qwen3.5-2B | 89% (84% / 100%) | 100% | 85% | 90% (0.17) | 44% (0.46) | 1.89 | 10% |
| Cosmos-Reason2-2B | 75% (70% / 100%) | 100% | 97% | 71% (0.30) | 93% (0.72) | 1.37 | 57% |
| InternVL3.5-2B | 83% (78% / 98%) | 100% | 97% | 51% (0.50) | 37% (0.44) | 1.51 | 46% |
| Gemma 4 E2B (llama.cpp) | 83% (96% / 73%) | 99% | 92% | 58% (0.78) | 37% (0.47) | - | - |
| Nemotron 3 Nano 4B on v3's caption | 97% (100% / 95%) | 99% | 97% | 95% (0.06) | - (-) | - | - |
| LocateAnything-3B INT4 (RTX 5070) | 66% (63% / 98%) | 99% | 71% | 62% (0.49) | 98% (0.85) | - | - |

### Speed and memory (on the Orin, beside the running NVR)

| Model | Caption p50 / p90 ms | First token ms | Decode ms/token | Yes/no p50 ms | Box p50 ms | Resident GiB |
|---|---|---|---|---|---|---|
| Cosmos3-Edge v3 (today) | 530 / 666 | 225 | 14.7 | 133 | 690 | 2.85 |
| Cosmos3-Edge v2 | 444 / 558 | 179 | 14.6 | 114 | 647 | 2.80 |
| Qwen3-VL-2B | 574 / 725 | 205 | 15.0 | 129 | 615 | 2.61 |
| Qwen3.5-2B | 669 / 850 | 274 | 16.7 | 202 | 894 | 2.83 |
| Cosmos-Reason2-2B | 677 / 886 | 207 | 14.9 | 132 | 564 | 2.74 |
| InternVL3.5-2B | 508 / 635 | 238 | 15.4 | 172 | 518 | 2.22 |
| Gemma 4 E2B (llama.cpp) | 1341 / 1797 | 438 | 40.4 | 495 | 2173 | 3.04 |
| Nemotron 3 Nano 4B on v3's caption | - / - | - | - | 381 | - | 2.63 |

### False "person" answers, by camera (frames with nobody in them)

| Model | front_driveway | front_entryway | reachy_mini | scout | scout_first_floor |
|---|---|---|---|---|---|
| Cosmos3-Edge v3 (today) | 1/13 | 0/5 | 2/14 | 0/6 | 3/9 |
| Cosmos3-Edge v2 | 2/13 | 0/5 | 1/14 | 1/6 | 6/9 |
| Qwen3-VL-2B | 0/13 | 0/5 | 8/14 | 0/6 | 5/9 |
| Qwen3.5-2B | 2/13 | 0/5 | 8/14 | 0/6 | 2/9 |
| Cosmos-Reason2-2B | 7/13 | 1/5 | 8/14 | 4/6 | 7/9 |
| InternVL3.5-2B | 2/13 | 3/5 | 5/14 | 2/6 | 6/9 |
| Gemma 4 E2B (llama.cpp) | 0/13 | 0/5 | 1/14 | 0/6 | 1/9 |
| Nemotron 3 Nano 4B on v3's caption | 0/13 | 0/5 | 0/14 | 0/6 | 0/9 |
| LocateAnything-3B INT4 (RTX 5070) | 10/13 | 2/5 | 14/14 | 2/6 | 8/9 |

The box column uses the shared `bbox_2d` prompt, or the model's own documented format where that
scored better. InternVL3.5's own `<ref>` prompt got 37%; with `bbox_2d` it got 0%. Two rows are not
like the others:
- **Nemotron 3 Nano 4B** cannot see. It answered the same questions from Cosmos3-Edge v3's caption
  of the frame, so its time is added to v3's 530 ms caption.
- **LocateAnything** was measured on the workstation's RTX 5070, because it does not fit on the Orin
  beside the NVR (below). It finds things rather than describing them, so it has no caption.
- **Gemma 4 E2B** runs in llama.cpp, from a newer build than NVIDIA's Jetson image (below), measured
  2026-10-05 under the same 3.3 GB cap. Its captions were not judged.

### What the numbers say

- **Cosmos3-Edge v3 is still the best detector here.** Among the VLMs it has the best person
  accuracy (95%: precision 91% at 100% recall), boxes (98% of boxes at IoU 0.5 or better, mean IoU
  0.80) and dog-or-cat answers (99%). It does this at 530 ms a caption in 2.85 GiB. It is weakest at
  counting and in caption detail:
  - It answered "2" or "4" on 21 of the 47 empty frames and 15 of the 57 one-person frames.
  - 30% of its captions assert something that is not in the frame.
- **Qwen3-VL-2B runs in the same envelope.** It takes 574 ms a caption and 129 ms for a yes/no, in
  2.61 GiB.
  - It writes better captions: 1.86 against 1.64, and only 16% of them hallucinate.
  - It counts better: 83% exact, mean absolute error 0.25.
  - It is worse where the NVR needs it most. It said "person" on 8 of 14 empty Reachy frames (the
    dark shelf) and 5 of 9 empty first-floor Scout frames, and its boxes are looser (85%, mean IoU
    0.68).
- **Qwen3.5-2B describes best** (1.89, 10% hallucinated) and counts best of the VLMs (90%).
  - It is about a quarter slower: 669 ms a caption and 202 ms for a yes/no. Its decode costs
    16.7 ms a token, against 14.6-15.4 ms for the rest, and its first token arrives at 274 ms.
  - It is weak at boxes (44%) and at dogs and cats (85%).
- **Cosmos-Reason2-2B is NVIDIA's physical-AI post-train of Qwen3-VL-2B.** On these cameras it lost
  more than it gained over its base.
  - Its boxes improved, from 85% to 93%.
  - It says "person" in empty rooms and empty driveways far more often (precision 70%).
  - Its captions got worse: 1.37, with 57% hallucinating. They are also longer, so they take 677 ms
    at the same 14.9 ms a token.
- **InternVL3.5-2B is the smallest** (2.22 GiB), with fast captions, but it is less accurate than v3
  on people, counts, boxes and captions.
- **Cosmos3-Edge v3 beats v2** on every accuracy column, for 86 ms more per caption.
- **Gemma 4 E2B runs, but last.** It rarely calls an empty room a person (precision 96%), but it
  misses people (recall 73%). It is 2.5-3.7 times slower than the TensorRT models: 495 ms for a
  yes/no, 1341 ms a caption, 40 ms a token. Its boxes need its own `box_2d` format (37%).

### Outside the TensorRT path

**Nemotron 3 Nano 4B (text only).** There is no INT4 TensorRT build: its 3136-wide projections fit
no 128 group. It ran instead in NVIDIA's Jetson llama.cpp container (Q4_K_M) as a second stage,
reading Cosmos3-Edge v3's captions.
- **Accuracy.** As a second stage it scores best on presence and counting: person 97% (precision
  100%, recall 95%) and counts 95% (MAE 0.06). Cosmos3-Edge's captions carry more truth than its own
  one-word answers.
- **Cost.** It adds 381 ms per question to the caption, and it needs its own 2.63 GiB. That fits
  only by switching models, because 0.8-1.3 GB is left free once a VLM is loaded.
- **Alerts.** Asked "should the homeowner be alerted?", it said yes to all 122 frames, 54 of which
  had nobody in them.
- **Verdict.** It is a reasonable text model to ask questions about Frigate's descriptions. It is
  not an alert filter as prompted.

**Gemma 4 E2B: it fits once llama.cpp stops holding its embedding table.** Of its 2.9 GB GGUF,
1.54 GiB is the per-layer embedding table and 0.21 GiB the token embedding. Both are lookups that
llama.cpp keeps on the CPU side as a memory map, and NVIDIA's Jetson build (b10373, 2026-08-12)
pre-faults the whole map at load.
- **Why it failed.** It was OOM-killed under the 3.3 GB cap twice, at 2.27 + 1.81 GB and
  2.56 + 1.81 GB (anonymous + mapped file). The 1.81 GB of file is those two tables, held resident
  although a lookup touches a few KB per token. TensorRT-Edge-LLM fares worse: it loads the table
  whole, in FP16, at 4.7 GB.
- **The fix is upstream.** llama.cpp added lazy tensor reads on 2026-08-27 (`--lazy-mode on`): the
  table's rows are read from disk when a token needs them. Its default `auto` only applies to
  tensors over 4 GiB, so it has to be forced on for E2B. `build_llama_upstream.sh` cross-compiles
  current llama.cpp for the Orin on the workstation. The binary runs inside NVIDIA's image, which
  supplies the CUDA 13 runtime.
- **Result.** It loads in about 40 s with 23 MiB of the GGUF mapped instead of 1.73 GiB. It holds
  3.04 GiB resident (0.5 GiB anonymous, 2.4 GiB NvMap) and leaves about 1 GB free on the board. It
  is a Live Vision button again (proxy engine `proxies/Gemma-4-E2B-it-llamacpp.proxy.json`). Its
  numbers are in the tables above.

**LocateAnything-3B: the best boxes, in llama.cpp, and only in demo mode.** Nothing claims it runs on
an Orin Nano:
- Jetson AI Lab has no page for it. Its parent there, Nemotron 3 Nano Omni, lists commands only for
  Thor and AGX Orin 64GB.
- Its model card says Transformers only ("TensorRT, TensorRT-LLM, and Triton are not yet supported"),
  tested on A100 and H100, with Thor "possible with additional model optimization".

Two attempts:
1. **PyTorch, INT4 (`locateanything/la_int4.py`).** This shrinks 7.7 GB to 2.2 GB: tinygemm INT4
   weights, an INT4 copy of the tied head, and an INT8 embedding.
   - On the RTX 5070 it drew the best boxes measured: 98% at IoU 0.5 or better, mean IoU 0.85.
   - It finds rather than classifies. Asked for "person" it boxed something in all 14 empty Reachy
     frames (precision 63%).
   - On the Orin it was OOM-killed under caps of 3.3 and 3.6 GB. PyTorch's CUDA context alone is
     0.9 GB there, and JetPack 7.2 has no sm_87 PyTorch build.
2. **llama.cpp, the jetson-device-skills recipe for an 8 GB board.** It comes from NVlabs/Eagle's
   Embodied branch: Kimi-VL's MoonViT-400M vision encoder, Eagle's MLP projector and a plain
   Qwen2.5-3B, so neither TensorRT-Edge-LLM nor Eagle's runtimes can load it.
   - Its parallel box decoding can be switched off. Next-token ("slow") decoding is plain causal
     Qwen2, and Eagle's results rate it the most accurate mode.
   - llama.cpp's draft PR #24749 adds the projector. `build_llama_upstream.sh`'s cross-build, pointed
     at the PR, makes the Orin binary.
   - The PR's own converter made a Q4_K_M text model (2.0 GiB) and a Q8_0 projector (0.6 GB).
     `locateanything/la_http_bench.py` benchmarks it over HTTP.

What the llama.cpp build measured:
- **Beside the NVR**, on 69 of the benchmark frames (640x360 event snapshots), under the 3.3 GB cap:
  - 2.8 GiB resident, about 1.4 s a query;
  - boxes 100% at IoU 0.5, mean 0.85;
  - dog or cat 93%, person 73% (precision 67%).
- **Then a global OOM.** On the Reachy's 960x540 recordings (672 image tokens instead of about 300)
  it grew past 3 GB while the board had 3.5 GB free. The kernel ran out of memory globally and
  killed `llama-server`, the largest process; Frigate survived.
- **Result: demo mode only.** Live Vision's LocateAnything button now needs demo mode (nvr/README.md),
  where it runs at full resolution with 3.46 GB to spare. `run_llama_bench.sh` now also holds its
  cap 600 MiB under what is free.

**Nemotron 3 Nano 30B-A3B does not run on this board.** Jetson AI Lab's catalogue ticks "Orin Nano
8GB", but the model's own page asks for 32 GB of RAM, and the smallest GGUF is 17 GB.
- `explore_30b.sh` ran it fenced: CPU only, a 2.5 GB cap, two cores and a 5-minute deadline, so
  llama.cpp could only demand-page the file from the SD card.
- In 5 minutes it read 23 GB from the SD card and never finished loading, so no token came.
- Frigate logged no recording warnings during the run. The GGUF was deleted afterwards.

## Bernese mountain dog: finding one dog among many

**The question:** how well does LocateAnything ground one specific dog, and how does it compare with
the other ways the Bernese mode can identify it?

**The footage:** the Reachy Mini filmed a monitor showing a doggy-daycare feed on 2026-10-04.
- Frames: one per minute from Frigate's recordings, 149 in all (960x540), 30 s off the frames the
  mode's gates were tuned on.
- Of these, 132 show the daycare feed, and the Bernese is in 50 of them.

**Ground truth:** two independent labelling passes per frame, with disagreements (11 of 149)
settled by hand from enlarged crops.
- Labellers picked the Bernese from numbered boxes drawn by a strong COCO detector (RT-DETR v2, on
  the frame and on 2x tiles), or drew a box where none covered it.
- So no model under test made the ground truth.

**The runs:**
- LocateAnything: the llama.cpp build, queried with "Bernese mountain dog" at full resolution (the
  RTX 5070 ran the same GGUF and code the Orin runs in demo mode).
- The two VLMs: the mode's own two-step question on the Orin, gated at P(yes) 0.6.
- The breed scores: the mode's breed service run over the RT-DETR boxes.

"Box on the Bernese" counts frames where the box a notification would draw overlaps the labelled
Bernese at IoU 0.5 or more.

| Method | Said "Bernese" (right / wrong) | Precision | Recall | Box on the Bernese | Wrong dog boxed | Time |
|---|---|---|---|---|---|---|
| LocateAnything-3B (llama.cpp) | 121 (50 / 71) | 41% | 100% | 8 of 50 | 40 | 0.4 s (RTX 5070) |
| Breed scores (ViT-B/16) on detector boxes | 16 (15 / 1) | 94% | 30% | 15 of 50 | 0 | ~0.5 s a dog (Orin CPU) |
| LocateAnything boxes, then breed scores | 8 (8 / 0) | 100% | 16% | 8 of 50 | 0 | |
| Qwen3-VL-2B, two-step | 12 (9 / 3) | 75% | 18% | 6 of 50 | 3 | ~1 s (Orin) |
| Cosmos3-Edge v3, two-step | 60 (24 / 36) | 40% | 48% | 5 of 50 | 19 | ~1 s (Orin) |

**What it says:**
- **LocateAnything grounds "a dog", not "the Bernese".** It answered "Bernese mountain dog" with a
  box on 121 of 132 frames, every frame with dogs in it. Its first box was on the Bernese in only 8
  of the 50 Bernese frames, and on another dog 40 times; any of its boxes hit the Bernese in 13.
  On this footage it is an excellent dog finder (its person boxes in the benchmark were the
  tightest measured) and a poor identifier of a breed among look-alikes.
- **Identity is a classifier's job.** The breed scores on the detector's boxes were wrong once in 16
  calls. The one miss, at 17:49, is a black dog the two labellers split on; it was settled as not a
  Bernese (white socks, no rust). They are the only method precise enough to push to a phone.
  - Recall is 30%. The detector boxed the Bernese in 34 of the 50 frames, and the classifier
    identified 15 of those 34 crops. The misses are tiny, side-on or half-hidden.
  - The breed rows share their boxes with the ground truth, so their localization is the
    detector's by construction. Their precision and recall are not.
- **The VLMs** sit between: Qwen3-VL is precise but finds few, and Cosmos3-Edge v3 cannot tell breeds.
- **The real bottleneck is upstream.** In Frigate's own CPU detector (SSD MobileNet, 320x320 regions)
  run over the same frames:
  - At the reachy_mini detect size (640x360), it found a track-worthy dog (score 0.7 or more) on the
    Bernese in 1 of 50 frames.
  - At 1280x720, it found one in 2.
  - The Bernese is a dark dog on a dark teal floor, a few dozen pixels high on a filmed monitor.
  - A mode triggered by Frigate's detection will therefore rarely fire on this camera, however good
    the identifier.
  - For a camera like this, the look should be scheduled while the camera is on, with a stronger dog
    detector in front of the breed scores. RT-DETR, which boxed the Bernese in 34 of 50 frames,
    runs on the Orin's CPU in about a second a frame.

### Which breed classifier

**The question:** would a dedicated dog-breed classifier identify the Bernese better than the
ImageNet ViT-B/16 the mode scores with?

**The candidates**, each at its pinned revision (`nvr/classifier/export_models.py`):
- The mode's own ImageNet ViT-B/16, in FP32 and as the INT8 ONNX the breed service runs.
- wesleyacheng's ViT-B/16, fine-tuned on Stanford Dogs.
- prithivMLmods' Dog-Breed-120, a SigLIP2-base fine-tuned on undisclosed data. Its card's 86.8%
  accuracy appears to be measured on its training images.

**The crops:** every RT-DETR dog box in the 132 daycare frames, cut as `breed_service.py` cuts them
(the box plus 10%). A crop counts as the Bernese if it overlaps the Bernese's labelled box at IoU 0.5 or more,
and as another dog below 0.1; the 4 in between (merged or partial boxes) are left out. That leaves 36
crops of the Bernese and 955 of other dogs.

**The score:** the probability of any of the four Swiss mountain dogs, as the mode reads it. The 95%
intervals come from a bootstrap over frames, because crops in one frame are not independent.

| Model | PR-AUC [95% CI] | ROC-AUC | At the mode's 0.5 gate: right / wrong | Frames said "Bernese" at 0.5 (right / wrong) | Gate set on the dev frames | At that gate: right / wrong | Top label "Bernese": right / wrong |
|---|---|---:|---:|---:|---:|---:|---:|
| ImageNet ViT-B/16 (FP32) | 0.74 [0.60-0.86] | 0.952 | 15 / 2 | 16 (15 / 1) | 0.47 | 15 / 2 | 11 / 0 |
| ImageNet ViT-B/16, INT8 (the breed service) | 0.74 [0.60-0.87] | 0.942 | 15 / 2 | 16 (15 / 1) | 0.40 | 15 / 2 | 11 / 0 |
| wesleyacheng ViT-B/16 | 0.71 [0.55-0.85] | 0.916 | 0 / 0 | 0 | 0.27 | 11 / 0 | 18 / 1 |
| Dog-Breed-120 SigLIP2 | 0.59 [0.42-0.73] | 0.910 | 15 / 11 | 26 (16 / 10) | 0.91 | 4 / 0 | 19 / 26 |

Each dev gate sits halfway between the best-scoring other dog on the 35 frames the mode's gate was
first set on and the next Bernese crop above it, so dev precision is 100%. It was then applied
unchanged to the crops above. The last column needs no gate: a crop counts when the model's single most
likely breed is the Bernese mountain dog, which is what Live Vision shows.

**What it says:**
- **The ImageNet ViT stays, for now.** Dog-Breed-120 ranks the crops significantly worse: PR-AUC 0.15 lower,
  95% interval 0.02 to 0.28. At 0.5 it called 11 other dogs Swiss mountain dogs, and on the dev
  frames another dog (0.89) outscored two of the four Bernese crops.
- **wesleyacheng ranks about as well** (PR-AUC 0.03 lower, interval -0.11 to +0.16), but its
  probabilities run low. The four dev Bernese crops scored 0.40-0.49, so at 0.5 it never fires; at its
  own gate it was right 11 times and never wrong, against the ImageNet ViT's 15.
- **As a top label it is the best of the three.** It named the Bernese on 18 of its 36 crops with one
  false call among 955 other dogs; the ImageNet ViT named it on 11, and Dog-Breed-120 on 19 but on 26
  other dogs too. A top-label rule for the mode is worth trying on the next footage; 36 crops are too
  few to switch on.
- **INT8 costs nothing measurable:** PR-AUC difference 0.000, interval -0.04 to +0.04.

All three are in Live Vision's "Dog breeds" group, so any picture can be tried on each
(`nvr/classifier/README.md`, "Dog breeds"; its "What the overlay shows" has how their saliency maps
fared on this footage).

## Jetson AI Lab's "runs on Orin Nano", and NVIDIA's jetson-device-skills

**What the listings mean.** Jetson AI Lab's Gemma 4 E2B page lists the Orin Nano 8GB for llama.cpp
only:
- Its page data has an empty vLLM command for the Nano, and its benchmark says "No data available".
- Its command is the bare `llama-server -hf unsloth/gemma-4-E2B-it-GGUF:Q4_K_S`, which assumes the
  whole board.
- That board is an otherwise idle Orin Nano with about 6 GB to spend. This one also runs Frigate,
  go2rtc, porch-dad, Live Vision and three Scout bridges. That leaves 3.3-3.9 GB with the shim
  stopped, and each model here is capped at 3.3 GB so that an overrun kills the model, not the NVR.
- LocateAnything has no listing at all (above).

**The skills, one by one.** These are the techniques in
[NVIDIA-AI-IOT/jetson-device-skills](https://github.com/NVIDIA-AI-IOT/jetson-device-skills), read
2026-10-05. Its own `audit.sh` and headless `plan.sh` were run on this board.

| Technique (skill) | Here | Measured |
|---|---|---|
| Headless: `multi-user.target`, no display manager (headless-mode) | already in place: the desktop session was ~243 MB ([nvr/README.md](../README.md)) | the skill's plan finds 3 MB left to take (avahi) |
| MAXN power mode (llm-serve) | already `MAXN_SUPER` | |
| `jetson_clocks` (llm-serve) | not used | no gain on Qwen3-VL-2B, which was loaded (details below) |
| INT4 W4A16 on the Orin Nano (inference-mem-tune) | every TensorRT model | |
| llama.cpp: small context, one slot (inference-mem-tune) | `-c` 1024-2048, `-np 1`, `--cache-ram 0` | |
| llama.cpp `--flash-attn` (inference-mem-tune) | `on` for Gemma; before, the default `auto` (on with CUDA) | |
| llama.cpp `--no-mmap` (inference-mem-tune) | not used, on purpose | it would make Gemma worse: 1.7 GiB of embedding tables would become anonymous memory the kernel cannot reclaim. What worked is newer than the skills: `--lazy-mode on` (above) |
| TensorRT Edge-LLM: batch, input length, KV (inference-mem-tune) | batch 1, 1024 input, 1024 KV | |
| Memory audit with NvMap attribution (memory-audit) | each run's cgroup `memory.stat` (anonymous + NvMap kernel + shared), the same idea | `audit.sh` agrees: 2.87 GB of NvMap in the shim with Cosmos-Reason2 loaded |
| `drop_caches` (memory-audit) | not needed | the skill scopes its stuck-memory bug to JetPack before 7.2; this is L4T 39.2.1 |
| llama.cpp + 4-bit GGUF for a model with no TensorRT path (inference-mem-tune) | LocateAnything-3B, via llama.cpp PR #24749 | fits beside the NVR only at about 300 image tokens; at full resolution, demo mode |
| vLLM / SGLang, speculative decoding (llm-serve, speculative-decoding) | not applicable | Jetson AI Lab gives Gemma 4 E2B no vLLM command on the Nano, and the skills' own recommender picks llama.cpp for 8 GB boards |

**The `jetson_clocks` A/B.** Qwen3-VL-2B was loaded; each arm ran 15 frames for captions and
yes/no, and 5 single requests:

| Request | Default clocks (DVFS) | Pinned |
|---|---|---|
| Warm caption | 548 ms | 544-550 ms |
| Single request after 25 s idle (median) | 396 ms | 420 ms |
| Yes/no | 130-134 ms | 130-134 ms |

DVFS ramps fast enough, so the clocks stay at their defaults, with no extra heat or power on a box
that runs all day.

## Recommendation

**Inference: keep Cosmos3-Edge v3 as the default.** On these cameras it is the only model that is
right about presence and boxes at the target speed and memory. On the indoor cameras it gave the fewest false
"person" answers of any VLM tested (5 on 29 empty frames), and its boxes are the tightest on the Orin. Its captions and counts are
its weak side.

**Post-training: start from Qwen3-VL-2B.**
- **It already fits.** It runs in v3's envelope (574 ms, 129 ms, 2.61 GiB) through the same export
  and engine build, verified here. NVIDIA publishes Orin Nano numbers for it.
- **It is strong where training is expensive.** It writes the best captions and counts of the models
  that run at full speed. Its mistakes are the kind that house-labelled fine-tuning fixes cheaply:
  - false "person" on the dark Reachy shelf and the first-floor Scout's empty rooms (more negatives);
  - loose boxes (Frigate's detector boxes, checked by the labellers, make the targets).
- **It is the easiest base to train.** It is Apache-2.0, writes 0-1000 boxes natively, and has the
  most widely used LoRA recipes in the 2B class.
- **Generic post-training will not do.** Cosmos-Reason2-2B is that same base after NVIDIA's generic
  physical-AI post-training, and here it made more false alarms, not fewer. Training data has to
  come from this house's cameras.

**Identity questions need Qwen3-VL's world knowledge.** In the Bernese mountain dog mode
(nvr/feed/detection_modes.py), on 35 frames of the doggy-daycare feed the Reachy watched:
- Qwen3-VL-2B said yes on 4 frames. Three were the Bernese, with a tight box; one was a
  curled-up husky at P 0.54.
- Cosmos3-Edge v3 said yes on 24 frames, boxing doodles and huskies.

So the engine that runs a camera's modes can matter more than the one that is best at "is there a
person". That is one more reason to post-train Qwen3-VL.

Choose Qwen3.5-2B instead if descriptions matter more than boxes. It gives the best captions and
counts, for about a quarter more latency, and its boxes need the most training.

To fine-tune, start from this benchmark's labels, with more empty frames from the Reachy and the
Scouts, and re-run `bench.py` and `score.py` after training.
