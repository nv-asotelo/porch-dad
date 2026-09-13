#!/usr/bin/env python3
"""Fit elapsed = fixed + marginal*gen_tok over WebUI-driven shim requests.

Usage: python3 collect_perf.py "10 min ago"
Reads [perf] lines from the Jetson shim journal and regresses latency against
generated-token count, so response-length variation doesn't corrupt A/B comparisons.
"""
import os
import re
import subprocess
import sys

JETSON = os.environ.get("JETSON_HOST", "orin@jetson.local")
window = sys.argv[1] if len(sys.argv) > 1 else "10 min ago"
out = subprocess.run(
    ["ssh", "-o", "BatchMode=yes", JETSON,
     f"journalctl -u cosmos3-edge-shim.service --since '{window}' --no-pager"],
    capture_output=True, text=True,
).stdout

pat = re.compile(r"elapsed_ms=(\d+) prompt_tok=(\d+) gen_tok=(\d+)")
rows = [(float(a), float(b), float(c)) for a, b, c in pat.findall(out)]
rows = [r for r in rows if r[2] > 0]

if len(rows) < 2:
    print(f"insufficient samples: {len(rows)}")
    sys.exit()

xs = [r[2] for r in rows]
ys = [r[0] for r in rows]
n = len(rows)
mx = sum(xs) / n
my = sum(ys) / n
den = sum((x - mx) ** 2 for x in xs)

print(f"n={n}  gen_tok range {min(xs):.0f}-{max(xs):.0f}  mean elapsed={my:.0f}ms")
if den == 0:
    print(f"  all gen_tok identical ({xs[0]:.0f}) - cannot separate fixed/marginal")
    sys.exit()

m = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
b = my - m * mx
ss_tot = sum((y - my) ** 2 for y in ys)
ss_res = sum((y - (m * x + b)) ** 2 for x, y in zip(xs, ys))
r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")

print(f"  marginal = {m:.2f} ms/token")
print(f"  fixed    = {b:.0f} ms")
print(f"  R^2      = {r2:.3f}")
