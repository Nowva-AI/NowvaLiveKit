"""E2: single-joint 3 cm perturbation -> displacement of every joint and diagnosis-metric change under each method (oracle lengths)."""
from __future__ import annotations
import json
import numpy as np
from synth import pose, body, recenter
from metrics import frame_metrics
from methods import PAIRS19, lengths_from, current_oracle, run_current, tree_a, two_bone_b, symmetric_c, pbd_e, soft_d
from biomechanics.utils.types import CocoKeypoints as CK

rng = np.random.default_rng(0)
b = body()
NAMES = {CK.LEFT_SHOULDER: "LSho", CK.RIGHT_SHOULDER: "RSho", CK.LEFT_HIP: "LHip", CK.RIGHT_HIP: "RHip",
         CK.LEFT_KNEE: "LKnee", CK.LEFT_ANKLE: "LAnk", CK.LEFT_FOOT_INDEX: "LToe", CK.RIGHT_KNEE: "RKnee",
         CK.RIGHT_ANKLE: "RAnk", CK.LEFT_WRIST: "LWri"}
METHODS = {
    "none": lambda p, L: p,
    "current": lambda p, L: run_current(current_oracle(L), p),
    "a_tree": tree_a,
    "b_twobone": two_bone_b,
    "c_symm": symmetric_c,
    "e_pbd5": lambda p, L: pbd_e(p, L, iters=5),
}
MKEYS = ["knee_flex_l", "valgus_l", "knee_dev_cm_l", "hip_shift_cm", "heel_rise_cm_l", "depth_cm", "trunk_flex", "pelvis_list"]
results = {}
for pose_name, s in (("standing", 0.0), ("bottom", 1.0)):
    truth = recenter(pose(s, b, valgus_deg_l=25.0))
    L = lengths_from(truth)
    tm = frame_metrics(truth)
    for src in (CK.LEFT_SHOULDER, CK.LEFT_HIP, CK.LEFT_KNEE, CK.LEFT_ANKLE):
        disp = {m: np.zeros(19) for m in METHODS}
        mdelta = {m: {k: [] for k in MKEYS} for m in METHODS}
        n = 200
        for _ in range(n):
            d = rng.normal(size=3); d /= np.linalg.norm(d)
            noisy = truth.copy(); noisy[src] += 0.03 * d
            for m, fn in METHODS.items():
                out = fn(noisy, L)
                disp[m] += np.linalg.norm(out - truth, axis=1) / n
                fm = frame_metrics(out)
                for k in MKEYS:
                    mdelta[m][k].append(abs(fm[k] - tm[k]))
        key = f"{pose_name}:{NAMES[src]}"
        results[key] = {}
        print(f"\n== {pose_name}: 3 cm impulse on {NAMES[src]} — mean |displacement| cm per joint, and mean |metric change|")
        print("method      " + " ".join(f"{NAMES[j]:>6s}" for j in NAMES) + "  sumAll | " + " ".join(f"{k[:10]:>10s}" for k in MKEYS))
        for m in METHODS:
            row = " ".join(f"{100*disp[m][j]:6.2f}" for j in NAMES)
            mrow = " ".join(f"{np.mean(mdelta[m][k]):10.2f}" for k in MKEYS)
            print(f"{m:11s} {row}  {100*disp[m].sum():6.1f} | {mrow}")
            results[key][m] = {"disp_cm": {NAMES[j]: round(100*disp[m][j], 3) for j in NAMES},
                               "sum_disp_cm": round(100*disp[m].sum(), 2),
                               "metric_abs_change": {k: round(float(np.mean(mdelta[m][k])), 3) for k in MKEYS}}
json.dump(results, open("e2_impulse.json", "w"), indent=1)
