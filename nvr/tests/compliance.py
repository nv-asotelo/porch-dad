import sys, time; sys.path.insert(0,'/home/orin/nvr/bridge')
import porch_dad as pd
frames = pd.sample_frames('/tmp/porch_test.mp4', 3)
BASE = pd.USER_PROMPT_TMPL
REINFORCE = ("\nYou MUST end your reply with exactly one category tag on the same line: "
             "[ROUTINE] or [ALERT]. Never omit it.")
def run(tag, tmpl, n=3):
    pd.USER_PROMPT_TMPL = tmpl
    cats=[]
    for i in range(n):
        t,ms = pd.describe(frames, "Front Porch", f"cmp-{i}")
        cats.append(pd.categorize(t))
        if i==0: print(f"  [{tag}] sample: {t[:150]!r}")
    print(f"  [{tag}] categories over {n} runs: {cats}\n")
print("=== as-specified prompt ===")
run("as-spec", BASE)
print("=== with explicit tag reinforcement ===")
run("reinforced", BASE + REINFORCE)
pd.USER_PROMPT_TMPL = BASE
