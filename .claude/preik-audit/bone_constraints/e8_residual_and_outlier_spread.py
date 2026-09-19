"""E8: (1) constraint residuals left after one production enforce pass (cycle in the torso); (2) spread of a single 20 cm outlier."""
from __future__ import annotations
import logging
import numpy as np
logging.disable(logging.WARNING)
from synth import sequence, add_correlated_noise, pose, body, recenter
from methods import PAIRS19, lengths_from, current_oracle, run_current, pbd_e
from detect import RIGID
from biomechanics.utils.types import CocoKeypoints as CK

truth, s = sequence(420)
L = lengths_from(truth[0])
bc = current_oracle(L)
rng = np.random.default_rng(11)
noisy, _ = add_correlated_noise(truth, rng, 0.02)
res = {pair: [] for pair in PAIRS19}
for p in noisy[60:]:
    out = run_current(bc, p)
    for pair in PAIRS19:
        res[pair].append(abs(np.linalg.norm(out[pair[1]] - out[pair[0]]) - L[pair]) * 1000)
print("residual |len - calibrated| after ONE production enforce pass (tolerance 0), iid 2 cm noise, mm: mean / max")
print("  " + ", ".join(f"{p}-{d}: {np.mean(v):.1f}/{np.max(v):.1f}" for (p, d), v in res.items()))

print("\nSingle 20 cm outlier (random direction) on one joint at the squat bottom, 300 trials: mean displacement (cm) of OTHER joints vs truth")
b = body()
t = recenter(pose(1.0, b))
Lb = lengths_from(t)
bcb = current_oracle(Lb)
names = {CK.LEFT_SHOULDER: "LSho", CK.RIGHT_SHOULDER: "RSho", CK.LEFT_HIP: "LHip", CK.RIGHT_HIP: "RHip", CK.LEFT_KNEE: "LKnee",
         CK.LEFT_ANKLE: "LAnk", CK.LEFT_FOOT_INDEX: "LToe", CK.RIGHT_KNEE: "RKnee"}
for src in (CK.LEFT_SHOULDER, CK.LEFT_HIP, CK.LEFT_KNEE):
    acc = {"current": np.zeros(19), "pbd_rigid": np.zeros(19)}
    for _ in range(300):
        d = rng.normal(size=3); d /= np.linalg.norm(d)
        x = t.copy(); x[src] += 0.20 * d
        acc["current"] += np.linalg.norm(run_current(bcb, x) - t, axis=1) / 300
        acc["pbd_rigid"] += np.linalg.norm(pbd_e(x, Lb, 5, pairs=RIGID) - t, axis=1) / 300
    for m, v in acc.items():
        others = [j for j in names if j != src]
        print(f"  outlier on {names[src]:6s} {m:10s}: self {100*v[src]:5.1f} | " + " ".join(f"{names[j]} {100*v[j]:4.1f}" for j in others))
