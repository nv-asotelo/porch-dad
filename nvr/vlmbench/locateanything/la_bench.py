#!/usr/bin/env python3
"""Run the VLM benchmark on LocateAnything-3B (INT4), writing the same JSONL as bench.py.

LocateAnything is a grounding model: it answers category queries ("person", "dog</c>cat") with
boxes, not prompts with prose. So the benchmark's questions become queries, and one "person" call
answers presence, count (number of boxes) and grounding (the first box) at once - which is how it
would be used. Captions are not something it does and are not asked. Each answer records the
latency of the call that produced it; answers sharing a call say so (shared_call).

  python la_bench.py --model <dir> --weights <int4.safetensors> --data <dir> --labels <labels.json> --out <jsonl>
"""
import argparse
import json
import os
import re
import sys
import time

# Not PYTORCH_CUDA_ALLOC_CONF=expandable_segments: on Jetson the first allocation then fails
# (PyTorch asks NVML for GPU fabric info, which the integrated GPU does not have).

BOX = re.compile(r"<ref>(.*?)</ref>((?:<box>.*?</box>)+)")
NUMS = re.compile(r"<(\d+)>")
QUERIES = {"person": "person", "vehicle": "car</c>truck</c>bus</c>motorcycle", "animal": "dog</c>cat"}


def parse(text):
    """'<ref>person</ref><box><614><305><716><611></box>...' -> {label: [[x1,y1,x2,y2], ...]} on 0-1000."""
    out = {}
    for label, boxes in BOX.findall(text or ""):
        out.setdefault(label, [])
        for b in re.findall(r"<box>(.*?)</box>", boxes):
            nums = [int(n) for n in NUMS.findall(b)]
            if len(nums) == 4:
                out[label].append(nums)
            elif len(nums) == 2:  # a point
                out[label].append(nums + nums)
    return out


def fit_side(im, max_side):
    """Shrink so the long side is at most max_side px (LA_MAX_SIDE): MoonViT runs at native resolution,
    and on the Orin its activations for a 960x540 frame do not fit beside the NVR."""
    if not max_side or max(im.size) <= max_side:
        return im
    s = max_side / max(im.size)
    return im.resize((max(14, round(im.size[0] * s)), max(14, round(im.size[1] * s))))


def cgroup_memory():
    path = "/sys/fs/cgroup" + open("/proc/self/cgroup").read().strip().split(":")[-1]
    def rd(name):
        try:
            return int(open(f"{path}/{name}").read().split()[0])
        except (OSError, ValueError):
            return None
    stat = {}
    try:
        for line in open(f"{path}/memory.stat"):
            k, v = line.split()
            if k in ("anon", "file", "kernel", "shmem", "file_mapped"):
                stat[k] = int(v)
    except OSError:
        pass
    avail = None
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            avail = int(line.split()[1]) * 1024
    return {"shim_current": rd("memory.current"), "shim_peak": rd("memory.peak"), "stat": stat, "mem_available": avail}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--labels")
    ap.add_argument("--label", default="locateanything-3b-int4")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.environ.setdefault("LA_FLASH_MODEL", args.model)
    os.environ.setdefault("LA_FLASH_ATTN", "sdpa")
    sys.path.insert(0, args.model)
    # This directory's la_int4 wins over any copy that ended up beside the model.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import torch
    from transformers import AutoProcessor, AutoTokenizer
    from batch_utils import generate_batch_hybrid
    from batch_utils import hybrid_runtime as hr
    from batch_utils.hybrid_runtime import load_pil
    from la_int4 import load_quantized

    t0 = time.time()
    hr._tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    hr._proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    hr._model = load_quantized(args.model, args.weights)
    hr._set_vision_attention_mode(hr._model)
    hr._set_llm_mode(hr._model, "sdpa")
    load_s = time.time() - t0

    def run(im, query):
        torch.cuda.synchronize()
        t = time.perf_counter()
        # top_p 0.9 is the README default and not optional: with top_p=None the hybrid decoder's
        # acceptance step degenerates into dozens of repeated boxes.
        text = generate_batch_hybrid([(im, query)], temperature=0.0, top_p=0.9, top_k=None,
                                     repetition_penalty=1.1, max_new_tokens=256)[0]
        torch.cuda.synchronize()
        return text, round((time.perf_counter() - t) * 1000, 1)

    manifest = json.load(open(os.path.join(args.data, "manifest.json")))
    items = manifest["items"][: args.limit]
    labels = json.load(open(args.labels))["items"] if args.labels else {}
    max_side = int(os.environ.get("LA_MAX_SIDE", "0"))
    warm = fit_side(load_pil(os.path.join(args.data, items[0]["image"])), max_side)
    for _ in range(2):
        run(warm, "person")

    mem_before = cgroup_memory()
    with open(args.out, "w") as out:
        out.write(json.dumps({"kind": "meta", "label": args.label, "family": "locateanything", "memory_before": mem_before,
                              "load_seconds": round(load_s, 1), "items": len(items), "max_side": max_side,
                              "tasks": ["person", "count", "vehicle", "animal", "ground"]}) + "\n")
        n, t_start = 0, time.time()
        for item in items:
            im = fit_side(load_pil(os.path.join(args.data, item["image"])), max_side)
            for qname, query in QUERIES.items():
                text, ms = run(im, query)
                found = parse(text)
                rec = {"kind": "answer", "label": args.label, "id": item["id"], "raw": text, "total_ms": ms,
                       "ttft_ms": ms, "usage": None, "query": query}
                if qname == "person":
                    boxes = found.get("person", [])
                    out.write(json.dumps({**rec, "task": "person", "text": "yes" if boxes else "no"}) + "\n")
                    out.write(json.dumps({**rec, "task": "count", "text": str(len(boxes)), "shared_call": True}) + "\n")
                    if not labels or (labels.get(item["id"]) or {}).get("person_box"):
                        ground = json.dumps({"bbox_2d": boxes[0]}) if boxes else "none"
                        out.write(json.dumps({**rec, "task": "ground", "text": ground, "shared_call": True}) + "\n")
                else:
                    hit = any(found.get(k) for k in found)
                    out.write(json.dumps({**rec, "task": qname, "text": "yes" if hit else "no"}) + "\n")
                n += 1
            out.flush()
            print(f"\r{args.label}: {n} calls, {time.time() - t_start:.0f}s", end="", file=sys.stderr, flush=True)
        out.write(json.dumps({"kind": "meta", "label": args.label, "memory_after": cgroup_memory(),
                              "seconds": round(time.time() - t_start, 1), "answers": n}) + "\n")
    print(f"\n{args.label}: {n} calls in {time.time() - t_start:.0f}s -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
