#!/usr/bin/env python3
"""Run the VLM benchmark's prompts against whatever model the shim is serving.

Runs ON the Orin, next to the shim, so the timings are the model's and not the LAN's: every
request streams, time to first token and total time are taken here, and the token counts
come back in the shim's usage chunk. Memory is read from the shim's own cgroup before and
after - on Jetson the GPU's nvmap allocations are charged there too, so memory.peak is the
whole model, weights, KV cache and activations - and from /proc/meminfo for the board.

Stdlib only, so it runs under the system python3 with nothing installed.

  python3 bench.py --data /home/orin/nvr/vlmbench/data --label qwen3-vl-2b --out results/qwen3-vl-2b.jsonl
"""
import argparse
import base64
import json
import os
import sys
import time
import urllib.request

# The production prompt is Frigate's GenAI prompt, word for word (genai.prompt in
# nvr/frigate/config.yml): it is what Cosmos3-Edge answers dozens of times an hour.
TASKS = {
    "caption": ("Describe only what is visible in this image, in one short sentence.", 64),
    "person": ("Is there a person in this image? Answer with only yes or no.", 4),
    "count": ("How many people are visible in this image? Answer with only a number.", 4),
    "vehicle": ("Is there a car or other vehicle in this image? Answer with only yes or no.", 4),
    "animal": ("Is there a dog or a cat in this image? Answer with only yes or no.", 4),
    "ground": ('Locate the person in this image. Reply with only a JSON object with "bbox_2d": its bounding '
               'box [x1, y1, x2, y2], with coordinates from 0 to 1000.', 48),
}
# Each family's own documented grounding prompt, run alongside the common one so that a model
# is not marked down for a format it was never trained on.
NATIVE_GROUND = {
    "internvl": ("Please provide the bounding box coordinate of the region this sentence describes: "
                 "<ref>the person</ref>", 48),
    "gemma": ("Detect the person in this image. Reply with only a JSON object with \"box_2d\": "
              "[ymin, xmin, ymax, xmax], with coordinates from 0 to 1000.", 48),
}
SHIM_CGROUP = "/sys/fs/cgroup/system.slice/cosmos3-edge-shim.service"
# For a text-only model (Nemotron 3 Nano 4B): no image, just a VLM's caption of the frame. What it
# can answer from that is what a text LLM behind the VLM would add.
CASCADE = 'A home camera frame was described by a vision model as: "{caption}"\n{question}'
CASCADE_TASKS = {"person", "count", "vehicle", "animal", "brief"}
TASKS_TEXT = {"brief": ("In one sentence, should the homeowner be alerted about this scene, and why?", 48)}


def read_int(path):
    try:
        with open(path) as f:
            return int(f.read().split()[0])
    except (OSError, ValueError):
        return None


def memory(cgroup):
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":")
            if k in ("MemTotal", "MemAvailable", "SwapFree"):
                info[k] = int(v.split()[0]) * 1024
    stat = {}
    try:
        with open(f"{cgroup}/memory.stat") as f:
            for line in f:
                k, v = line.split()
                if k in ("anon", "file", "kernel", "shmem", "file_mapped"):
                    stat[k] = int(v)
    except OSError:
        pass
    # anon + kernel (nvmap's GPU buffers are charged as kernel memory) + shmem is what the model
    # actually holds; "file" is page cache from reading the engine files, which the kernel drops at
    # will - it inflates memory.current and memory.peak without being needed.
    return {"shim_current": read_int(f"{cgroup}/memory.current"), "shim_peak": read_int(f"{cgroup}/memory.peak"),
            "stat": stat,
            "mem_available": info.get("MemAvailable"), "swap_free": info.get("SwapFree")}


def ask(url, image_b64, prompt, max_tokens, timeout=120):
    content = [{"type": "text", "text": prompt}]
    if image_b64 is not None:
        content.insert(0, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}})
    body = {
        "model": "local",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens, "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft, text, usage, finish = None, [], None, None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                piece = (choice.get("delta") or {}).get("content")
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    text.append(piece)
                finish = choice.get("finish_reason") or finish
    total = time.perf_counter() - t0
    return {"text": "".join(text), "ttft_ms": round(ttft * 1000, 1) if ttft is not None else None,
            "total_ms": round(total * 1000, 1), "usage": usage, "finish_reason": finish}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    ap.add_argument("--data", required=True, help="directory holding manifest.json and images/")
    ap.add_argument("--label", required=True, help="name of the model being measured")
    ap.add_argument("--family", default="", help="internvl or gemma adds that family's native grounding prompt")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--labels", help="labels.json; grounding then runs only where a person box is known")
    ap.add_argument("--cgroup", default=SHIM_CGROUP,
                    help="cgroup whose memory.current/peak is the model's (a llama.cpp container's docker-<id>.scope)")
    ap.add_argument("--context-from", metavar="RUN.jsonl",
                    help="text-only model: send this run's caption of each frame instead of the image")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    manifest = json.load(open(os.path.join(args.data, "manifest.json")))
    items = manifest["items"][: args.limit]
    labels = json.load(open(args.labels))["items"] if args.labels else {}
    tasks = [(t, *TASKS[t]) for t in args.tasks.split(",")]
    if args.family in NATIVE_GROUND and "ground" in args.tasks:
        tasks.append(("ground_native", *NATIVE_GROUND[args.family]))
    captions = None
    if args.context_from:
        captions = {r["id"]: r.get("text") for r in map(json.loads, open(args.context_from))
                    if r.get("kind") == "answer" and r.get("task") == "caption" and r.get("text")}
        tasks = [t for t in tasks if t[0] in CASCADE_TASKS] + [("brief", *TASKS_TEXT["brief"])]

    warm = None if captions is not None else base64.b64encode(open(os.path.join(args.data, items[0]["image"]), "rb").read()).decode()
    for _ in range(3):  # first-call costs (CUDA graphs, allocator growth) stay out of the numbers
        ask(args.url, warm, TASKS["caption"][0], 16)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    mem_before = memory(args.cgroup)
    n, t_start = 0, time.time()
    with open(args.out, "w") as out:
        out.write(json.dumps({"kind": "meta", "label": args.label, "family": args.family, "memory_before": mem_before,
                              "context_from": args.context_from,
                              "started": t_start, "items": len(items), "tasks": [t[0] for t in tasks]}) + "\n")
        for item in items:
            image_b64 = None
            if captions is None:
                image_b64 = base64.b64encode(open(os.path.join(args.data, item["image"]), "rb").read()).decode()
            elif item["id"] not in captions:
                continue
            for task, prompt, max_tokens in tasks:
                if captions is not None:
                    prompt = CASCADE.format(caption=captions[item["id"]], question=prompt)
                if task.startswith("ground") and labels:
                    gt = labels.get(item["id"], {})
                    if not gt.get("person_box"):
                        continue
                try:
                    res = ask(args.url, image_b64, prompt, max_tokens)
                except Exception as err:  # one failed request is a result, not the end of the run
                    res = {"error": f"{type(err).__name__}: {err}"}
                out.write(json.dumps({"kind": "answer", "label": args.label, "id": item["id"], "task": task, **res}) + "\n")
                out.flush()
                n += 1
            print(f"\r{args.label}: {n} answers, {time.time() - t_start:.0f}s", end="", file=sys.stderr, flush=True)
        out.write(json.dumps({"kind": "meta", "label": args.label, "memory_after": memory(args.cgroup),
                              "seconds": round(time.time() - t_start, 1), "answers": n}) + "\n")
    print(f"\n{args.label}: {n} answers in {time.time() - t_start:.0f}s -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
