"""Save a labelling agent's final JSON answer from its transcript, printing only a summary."""
import json, sys
path, out = sys.argv[1], sys.argv[2]
last = None
for line in open(path):
    try:
        rec = json.loads(line)
    except ValueError:
        continue
    msg = rec.get("message") or {}
    if msg.get("role") == "assistant":
        for c in msg.get("content") or []:
            if isinstance(c, dict) and c.get("type") == "text" and "{" in c.get("text", ""):
                last = c["text"]
text = last[last.index("{"): last.rindex("}") + 1]
data = json.loads(text)
json.dump(data, open(out, "w"), indent=1)
print(out.split("/")[-1], len(data["frames"]), "frames;", sum(bool(f.get("bernese_ids") or f.get("unboxed")) for f in data["frames"]), "with a Bernese")
