#!/usr/bin/env python3
"""The LocateAnything benchmark of la_bench.py, asked of an OpenAI-compatible server instead of
PyTorch - here llama.cpp built from PR #24749, which adds LocateAnything's vision projector and
decodes boxes one token at a time (its "slow" mode; parallel box decoding needs PyTorch).

Same queries and records as la_bench.py, so score.py reads both. Takes bench.py's arguments, so
run_llama_bench.sh can drive it (BENCH_SCRIPT=locateanything/la_http_bench.py). The server must run
with --special, or the <ref>/<box> tokens are stripped from the answer.

  python3 la_http_bench.py --url http://127.0.0.1:8091/v1/chat/completions --data DIR --label L --out OUT.jsonl
"""
import argparse
import base64
import io
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from bench import SHIM_CGROUP, memory  # noqa: E402
from la_bench import QUERIES, parse  # noqa: E402  (no torch at import time)

# The processor's own wrapper (processing_locateanything.py py_apply_chat_template) and the query
# wording of its batch runtime (batch_utils/hybrid_runtime.py _PROMPT).
SYSTEM = "You are a helpful assistant.\n"
PROMPT = "Locate all the instances that matches the following description: "


def locate(url, image_b64, query, timeout=180):
    body = {"model": "local", "temperature": 0, "max_tokens": 256, "stream": True,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": [
                             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                             {"type": "text", "text": PROMPT + query}]}]}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft, text = None, []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            for choice in json.loads(line[5:]).get("choices") or []:
                piece = (choice.get("delta") or {}).get("content")
                if piece:
                    ttft = ttft if ttft is not None else time.perf_counter() - t0
                    text.append(piece)
    return "".join(text), round((time.perf_counter() - t0) * 1000, 1), round((ttft or 0) * 1000, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8091/v1/chat/completions")
    ap.add_argument("--data", required=True)
    ap.add_argument("--labels")
    ap.add_argument("--label", default="locateanything-3b-gguf")
    ap.add_argument("--cgroup", default=SHIM_CGROUP)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--family", default="")      # accepted for bench.py compatibility
    args = ap.parse_args()

    manifest = json.load(open(os.path.join(args.data, "manifest.json")))
    items = manifest["items"][: args.limit]
    labels = json.load(open(args.labels))["items"] if args.labels else {}
    b64 = lambda item: base64.b64encode(open(os.path.join(args.data, item["image"]), "rb").read()).decode()
    locate(args.url, b64(items[0]), "person")     # warm-up, as la_bench.py does

    with open(args.out, "w") as out:
        out.write(json.dumps({"kind": "meta", "label": args.label, "family": "locateanything",
                              "memory_before": memory(args.cgroup), "items": len(items),
                              "tasks": ["person", "count", "vehicle", "animal", "ground"]}) + "\n")
        n, t_start = 0, time.time()
        for item in items:
            image = b64(item)
            for qname, query in QUERIES.items():
                text, ms, ttft = locate(args.url, image, query)
                found = parse(text)
                rec = {"kind": "answer", "label": args.label, "id": item["id"], "raw": text, "total_ms": ms,
                       "ttft_ms": ttft, "usage": None, "query": query}
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
        out.write(json.dumps({"kind": "meta", "label": args.label, "memory_after": memory(args.cgroup),
                              "seconds": round(time.time() - t_start, 1), "answers": n}) + "\n")
    print(f"\n{args.label}: {n} calls in {time.time() - t_start:.0f}s -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
