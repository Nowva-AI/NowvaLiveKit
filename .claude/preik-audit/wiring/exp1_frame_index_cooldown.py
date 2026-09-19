"""E1: triangulated frame_index is always 0 -> frame-index cooldowns suppress every fault after the first."""
from collections import Counter
from harness import *

REPS = 6

def run(mode: str) -> tuple[Counter, int]:
    clock = FakeClock()
    pipe, prov = build_pipeline(clock)
    prov.frame_index_mode = mode
    counts: Counter = Counter()
    per_rep: list[Counter] = []
    for rep in range(REPS):
        c = Counter()
        for d in rep_depth_profile():
            res = step(pipe, prov, clock, squat_points(d, valgus_m=0.30))
            for f in res.faults:
                counts[f.fault_type] += 1
                c[f.fault_type] += 1
        per_rep.append(c)
    return counts, pipe.rep_count, per_rep

for mode in ("zero", "incrementing"):
    counts, reps, per_rep = run(mode)
    print(f"frame_index={mode:12s} reps={reps} faults={dict(counts)}")
    print("   per rep:", [dict(c) for c in per_rep])
