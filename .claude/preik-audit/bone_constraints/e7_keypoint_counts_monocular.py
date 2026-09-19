"""E7: (1) calibration with 17/19/21-keypoint skeletons; (2) single-camera (depth-noisy) variant: sphere projection vs depth-only solve."""
from __future__ import annotations
import logging
import math
import numpy as np
logging.disable(logging.WARNING)
from synth import sequence, add_correlated_noise, to_skel, body, pose, recenter
from metrics import summarize, seq_metrics
from methods import lengths_from, current_oracle, run_current, _length
from biomechanics.utils.bone_constraints import BoneLengthConstraints, BONE_PAIRS
from biomechanics.utils.types import CocoKeypoints as CK
from biomechanics.pipeline_process import _extract_athlete_params


class _P:  # stub with the attribute _extract_athlete_params reads
    pass


truth, s = sequence(60)
for n_kpts in (17, 19, 21):
    pts = truth[0]
    if n_kpts == 21:
        pts = np.vstack([pts, pts[CK.LEFT_ANKLE] + [0, 0.03, -0.06], pts[CK.RIGHT_ANKLE] + [0, 0.03, -0.06]])
    else:
        pts = pts[:n_kpts]
    bc = BoneLengthConstraints(calibration_frames=5)
    for _ in range(5):
        bc.enforce(to_skel(pts, conf=np.full(n_kpts, 0.9)))
    stub = _P(); stub._bone_constraints = bc
    ap = _extract_athlete_params(stub)
    print(f"{n_kpts} kpts: calibrated pairs {len(bc._calibrated_lengths)}/{len(BONE_PAIRS)}; heel pairs present {any(p[1] >= 19 for p in bc._calibrated_lengths)}; "
          f"foot_avg_m {ap['foot_avg_m']:.3f} (0.26 = silent default)")
    out = bc.enforce(to_skel(pts, conf=np.full(n_kpts, 0.9)))
    print(f"   enforce ok, output kpts {len(out.keypoints)}")

LEG_TREE = [(CK.LEFT_HIP, CK.LEFT_KNEE), (CK.LEFT_KNEE, CK.LEFT_ANKLE), (CK.LEFT_ANKLE, CK.LEFT_FOOT_INDEX),
            (CK.RIGHT_HIP, CK.RIGHT_KNEE), (CK.RIGHT_KNEE, CK.RIGHT_ANKLE), (CK.RIGHT_ANKLE, CK.RIGHT_FOOT_INDEX)]


def depth_only(pts, L):
    out = pts.copy()
    for p, d in LEG_TREE:
        target = _length(L, p, d)
        dx, dy = out[d, 0] - out[p, 0], out[d, 1] - out[p, 1]
        dz_noisy = pts[d, 2] - pts[p, 2]
        rem = target * target - dx * dx - dy * dy
        dz = math.copysign(math.sqrt(rem), dz_noisy) if rem > 0 else 0.0
        out[d, 2] = out[p, 2] + dz
    return out


print("\nSingle-camera-like noise: sigma_xy 1 cm, sigma_z 6 cm (rho 0.6), oracle lengths, 3 seeds, all frames>=60")
truth, s = sequence(420, valgus_deg_l=25.0)
tm = seq_metrics(truth)
frames = np.arange(60, 420)
L = lengths_from(truth[0])
acc = {"none": [], "current_sphere": [], "depth_only_legs": []}
for seed in range(3):
    rng = np.random.default_rng(500 + seed)
    noisy, _ = add_correlated_noise(truth, rng, 0.01, rho=0.6, z_scale=6.0)
    bc = current_oracle(L)
    acc["none"].append(summarize(noisy, truth, tm, frames))
    acc["current_sphere"].append(summarize(np.stack([run_current(bc, p) for p in noisy]), truth, tm, frames))
    acc["depth_only_legs"].append(summarize(np.stack([depth_only(p, L) for p in noisy]), truth, tm, frames))
cols = ["mpjpe_cm", "knee_cm", "ankle_cm", "knee_flex_l_rmse", "valgus_l_rmse", "hip_add_l_rmse", "depth_cm_rmse", "hip_shift_cm_rmse", "heel_rise_cm_l_rmse", "dorsi_l_rmse", "trunk_flex_rmse"]
print(f"{'method':18s}" + "".join(f"{c.replace('_rmse','')[:10]:>11s}" for c in cols))
for m, rows in acc.items():
    print(f"{m:18s}" + "".join(f"{np.mean([r[c] for r in rows]):11.2f}" for c in cols))
