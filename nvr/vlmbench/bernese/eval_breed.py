"""The breed service's verdict on every eval frame: the detector's dog boxes scored with the same
INT8 ViT-B/16 and the same crop/normalisation as breed_service.py; best box wins at >= 0.5."""
import json, os, sys, time
sys.path.insert(0, "/tmp/porch-dad-main/nvr/feed")
from PIL import Image
from breed_service import Scorer
cands = json.load(open("/home/asotelo/vlm-sweep/bernese/eval-cands/candidates.json"))
scorer = Scorer("/home/asotelo/vlm-sweep/breed/vit-b16-imagenet-int8.onnx", "/home/asotelo/vlm-sweep/breed/vit-b16-imagenet.json", threads=4)
classes = ["Bernese mountain dog", "EntleBucher", "Appenzeller", "Greater Swiss Mountain dog"]
with open("/home/asotelo/vlm-sweep/bernese/eval-breed.jsonl", "w") as out:
    out.write(json.dumps({"kind": "meta", "label": "breed-vit-b16-int8", "engine": "ImageNet ViT-B/16 INT8 on detector boxes"}) + "\n")
    for f in sorted(cands):
        im = Image.open(f"/home/asotelo/vlm-sweep/bernese/eval/{f}").convert("RGB")
        boxes = [c["box"] for c in cands[f]]
        t0 = time.perf_counter()
        scores, top = scorer.score(im, boxes, classes) if boxes else ([], [])
        k = max(range(len(scores)), key=scores.__getitem__) if scores else None
        conf = scores[k] if k is not None else None
        out.write(json.dumps({"kind": "answer", "id": f, "found": bool(conf is not None and conf >= 0.5),
                              "confidence": conf, "box": boxes[k] if conf is not None and conf >= 0.5 else None,
                              "best_box": boxes[k] if k is not None else None, "scores": scores,
                              "ms": round((time.perf_counter() - t0) * 1000, 1)}) + "\n")
print("done")
