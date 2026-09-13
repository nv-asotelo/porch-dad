import sys; sys.path.insert(0,'/home/orin/nvr/bridge')
import porch_dad as pd
CASES = [
 ("Human activity. A man wearing a black shirt is gesturing with both hands.", "ROUTINE"),
 ("No activity detected.", "ROUTINE"),
 ("A person in a hooded jacket is standing by the door with their face covered, looking around repeatedly.", "ALERT"),
 ("A delivery driver in a uniform places a cardboard box by the door and walks away.", "ROUTINE"),
 ("Smoke is rising from the left side of the garage.", "ALERT"),
 ("A person is climbing over the backyard fence.", "ALERT"),
 ("A man has been loitering near the porch for several minutes.", "ALERT"),
 ("Someone is prying at the front door lock.", "ALERT"),
 ("A squirrel runs across the porch railing.", "ROUTINE"),
 ("[ALERT] Person concealing their face approaches.", "ALERT"),
 ("[ROUTINE] Mail carrier drops off letters.", "ROUTINE"),
 ("", "UNSCORED"),
]
ok=0
for text, want in CASES:
    got, reason = pd.categorize(text)
    mark = "PASS" if got==want else "FAIL"
    ok += got==want
    print(f"  {mark}  {got:8s} (want {want:8s}) [{reason:26s}] {text[:58]}")
print(f"\n{ok}/{len(CASES)} correct")
