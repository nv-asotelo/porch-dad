#!/usr/bin/env python3
"""Write meta.json for the dog-breed classifiers from their pinned Hugging Face configs. Run on a
workstation, after or before export_models.py (it downloads the same pinned files).

labels is one [raw label, species] pair per logit, in id order. The species is the breed as the
page's own kebab key (speciesKey in nvr/ui/web/app.js), so all three models name a breed the same
way and their answers can be compared: "bernese-mountain-dog" whichever model says it. Labels that
are not dog breeds get null - ImageNet's other 879 classes, and Dog-Breed-120's stray "test" class.
Each file is checked with classifier_service.load_meta before it is written, because one bad
meta.json keeps the whole classifier service from starting.

  python make_dog_meta.py [--out-dir nvr/classifier/models]
"""
import argparse
import json
import re
import sys
import tempfile
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from export_models import PINS, download  # noqa: E402

ROLLOUT = "attention rollout (class-agnostic: where the model looked, whatever it decided)"
MODELS = {
    "imagenet-vit": {
        "name": "ImageNet ViT-B/16",
        "architecture": "ViT-B/16 · ImageNet-1k, 121 of its 1,000 classes are dogs",
        "saliency": ROLLOUT,
        "license": "Apache-2.0",
        # ImageNet's dogs: classes 151-268 (domestic breeds) and 273-275 (dingo, dhole, African
        # hunting dog), the same 120 breeds as Stanford Dogs plus the dalmatian.
        "dog_ids": set(range(151, 269)) | {273, 274, 275},
    },
    "wesleyacheng": {
        "name": "wesleyacheng ViT-B/16",
        "architecture": "ViT-B/16 (ImageNet-21k) fine-tuned on Stanford Dogs · 120 breeds",
        "saliency": ROLLOUT,
        "license": "MIT (model card); fine-tuned on the Stanford Dogs dataset",
        # Its own labels were cut at the first hyphen of the Stanford Dogs folder names
        # ("n02099429-curly-coated_retriever" -> "curly"). Each stub names exactly one breed.
        "aliases": {"curly": "curly-coated_retriever", "wire": "wire-haired_fox_terrier",
                    "soft": "soft-coated_wheaten_terrier", "flat": "flat-coated_retriever", "shih": "shih-tzu",
                    "black": "black-and-tan_coonhound", "german_short": "german_short-haired_pointer"},
    },
    "dogbreed120": {
        "name": "Dog-Breed-120 SigLIP2",
        "architecture": "SigLIP2-base/16 fine-tuned on dog breeds · 120 breeds and a stray 'test'",
        "saliency": "class activation map of the top breed: each patch token's share of its score",
        "license": "Apache-2.0 (model card); training data undisclosed",
        "not_breeds": {"test"},
    },
}
SOURCES = {
    "imagenet-vit": ("google/vit-base-patch16-224", "3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3",
                     "1cea07110a4a47edc51420b2dda6f3b8b58e7256e8f44b4ea6aa9696162ccb5d"),
}


def species_key(name):
    """speciesKey in nvr/ui/web/app.js, in Python."""
    key = name.strip().replace("♀", "-f").replace("♂", "-m")
    key = "".join(c for c in unicodedata.normalize("NFKD", key) if not unicodedata.combining(c)).lower()
    key = re.sub(r"['’.]", "", key).replace(":", "-")
    key = re.sub(r"[\s_]+", "-", key)
    key = re.sub(r"-+", "-", key).strip("-")
    return key or None


def labels_for(mid, id2label):
    spec = MODELS[mid]
    out = []
    for i in range(len(id2label)):
        raw = id2label[str(i)] if str(i) in id2label else id2label[i]
        first = raw.split(",")[0]
        if "dog_ids" in spec:
            breed = species_key(first) if i in spec["dog_ids"] else None
        else:
            first = spec.get("aliases", {}).get(first, first)
            breed = None if first in spec.get("not_breeds", ()) else species_key(first)
        out.append([raw, breed])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=HERE / "models")
    args = ap.parse_args()
    import classifier_service
    keys = {}
    for mid, spec in MODELS.items():
        repo, revision, files = PINS[mid]
        source = download(mid)
        config = json.loads((source / "config.json").read_text())
        weights_sha = files["model.safetensors"]
        labels = labels_for(mid, config["id2label"])
        assert len(labels) == len(config["id2label"]), mid
        meta = {
            "id": mid, "name": spec["name"], "architecture": spec["architecture"], "input": "pixel_values",
            "preprocess": {"size": 224, "resample": "bilinear", "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
            "saliency": spec["saliency"], "onnx": "model_fp16.onnx", "engine": "model_fp16.plan",
            "outputs": {"logits": "logits", "saliency": "saliency"}, "labels": labels,
            "source": f"https://huggingface.co/{repo}/tree/{revision}",
            "sha256_of_original_weights": weights_sha, "license": spec["license"],
        }
        assert len(meta["saliency"]) <= 80, (mid, len(meta["saliency"]))
        assert all(len(raw) <= 128 and (s is None or len(s) <= 64) for raw, s in labels), mid
        text = json.dumps(meta, indent=1, ensure_ascii=False) + "\n"
        with tempfile.TemporaryDirectory() as scratch:    # load_meta wants <dir named as the id>/meta.json
            staged = Path(scratch) / mid
            staged.mkdir()
            (staged / "meta.json").write_text(text)
            classifier_service.load_meta(staged)        # the service's own check, before anything is written
        target = args.out_dir / mid
        target.mkdir(parents=True, exist_ok=True)
        tmp = target / "meta.json.tmp"
        tmp.write_text(text)
        tmp.replace(target / "meta.json")
        keys[mid] = {s for _, s in labels if s}
        print(f"{mid}: {len(labels)} labels, {len(keys[mid])} breeds -> {target / 'meta.json'}")
    common = keys["wesleyacheng"] & keys["dogbreed120"] & keys["imagenet-vit"]
    print(f"breeds all three name the same way: {len(common)}; "
          f"only some: {sorted((keys['wesleyacheng'] | keys['dogbreed120'] | keys['imagenet-vit']) - common)}")


if __name__ == "__main__":
    main()
