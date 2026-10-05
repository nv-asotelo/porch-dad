"""LocateAnything (GGUF, llama.cpp PR #24749) on every eval frame, query "Bernese mountain dog"."""
import base64
import glob
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, "/tmp/porch-dad-main/nvr/vlmbench/locateanything")
from la_server import LOCATE, SYSTEM, parse  # noqa: E402

url, out_path, label = sys.argv[1], sys.argv[2], sys.argv[3]
with open(out_path, "w") as out:
    out.write(json.dumps({"kind": "meta", "label": label}) + "\n")
    for f in sorted(glob.glob("/home/asotelo/vlm-sweep/bernese/eval/*.jpg")):
        b64 = base64.b64encode(open(f, "rb").read()).decode()
        body = {"temperature": 0, "max_tokens": 256, "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
                                         {"type": "text", "text": LOCATE + "Bernese mountain dog"}]}]}
        t0 = time.perf_counter()
        r = json.load(urllib.request.urlopen(urllib.request.Request(
            url, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=300))
        raw = r["choices"][0]["message"]["content"]
        boxes = [[v / 1000 for v in b["bbox_2d"]] for b in parse(raw) if "bbox_2d" in b]
        out.write(json.dumps({"kind": "answer", "id": os.path.basename(f), "found": bool(boxes),
                              "box": boxes[0] if boxes else None, "boxes": boxes, "raw": raw,
                              "ms": round((time.perf_counter() - t0) * 1000, 1)}) + "\n")
        out.flush()
print("done")
