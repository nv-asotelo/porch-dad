#!/usr/bin/env python3
"""Compare two shim time windows at matched gen_tok.

Regression needs gen_tok spread; when the WebUI emits constant-length replies
that degenerates. Matching on gen_tok is valid either way and is the stronger
comparison, so prefer it and fall back to the fitted model only for coverage.
"""
import os
import re, subprocess, sys, statistics as st

JETSON = os.environ.get("JETSON_HOST", "orin@jetson.local")
pat = re.compile(r"elapsed_ms=(\d+) prompt_tok=(\d+) gen_tok=(\d+)")

def pull(since, until=None):
    cmd = f"journalctl -u cosmos3-edge-shim.service --since '{since}'"
    if until: cmd += f" --until '{until}'"
    out = subprocess.run(["ssh", "-o", "BatchMode=yes", JETSON,cmd+" --no-pager"],
                         capture_output=True,text=True).stdout
    return [(int(a),int(b),int(c)) for a,b,c in pat.findall(out) if int(c)>0]

def fit(rows):
    xs=[r[2] for r in rows]; ys=[r[0] for r in rows]; n=len(rows)
    mx=sum(xs)/n; my=sum(ys)/n; den=sum((x-mx)**2 for x in xs)
    if den==0: return None,my,0.0
    m=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/den
    return m, my-m*mx, 1-sum((y-(m*x+my+m*mx-m*mx-(my-m*mx))+0)**2 for x,y in zip(xs,ys))/1  # unused

def describe(label, rows):
    xs=[r[2] for r in rows]; ys=[r[0] for r in rows]; ps=[r[1] for r in rows]
    print(f"{label}: n={len(rows)} prompt_tok={min(ps)}-{max(ps)} "
          f"gen_tok={min(xs)}-{max(xs)} mean_elapsed={sum(ys)/len(ys):.0f}ms")
    return {}

def by_gentok(rows):
    d={}
    for e,p,g in rows: d.setdefault(g,[]).append(e)
    return d

a_since, a_until, b_since = sys.argv[1], sys.argv[2], sys.argv[3]
A=pull(a_since,a_until); B=pull(b_since)
if not A or not B: sys.exit(f"empty window (A={len(A)} B={len(B)})")

print("=== BEFORE ==="); describe("A", A)
print("=== AFTER  ==="); describe("B", B)

da, db = by_gentok(A), by_gentok(B)
shared = sorted(set(da) & set(db), key=lambda g: -(len(da[g])+len(db[g])))
print(f"\n{'gen_tok':>8} {'nA':>4} {'nB':>4} {'A_ms':>8} {'B_ms':>8} {'delta':>8} {'improve':>8}")
tot_a=tot_b=0; wt=0
for g in shared[:8]:
    ma, mb = st.median(da[g]), st.median(db[g])
    w=min(len(da[g]),len(db[g]))
    tot_a+=ma*w; tot_b+=mb*w; wt+=w
    print(f"{g:>8} {len(da[g]):>4} {len(db[g]):>4} {ma:>8.0f} {mb:>8.0f} "
          f"{mb-ma:>8.0f} {(ma-mb)/ma*100:>7.1f}%")
if wt:
    print(f"\nweighted matched-gen_tok improvement: {(tot_a-tot_b)/tot_a*100:.1f}%")
else:
    print("\nno shared gen_tok values; compare via fitted model instead")
