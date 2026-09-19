"""Sanity: synth kinematics vs repo IK, velocities, noise levels."""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/Users/naiahoard/NowvaLiveKit/src")
sys.path.insert(0, "/Users/naiahoard/NowvaLiveKit/.claude/preik-audit/smoothing")

from biomechanics.kinematics.analytical_ik import AnalyticalIKSolver  # noqa: E402
from biomechanics.kinematics.valgus import TriangulatedValgusEstimator  # noqa: E402
from biomechanics.utils.types import CocoKeypoints as CK, Skeleton3D  # noqa: E402

import kin  # noqa: E402
import synth  # noqa: E402

world, s, phase, windows = synth.session()
truth = synth.recenter(world)
T = len(truth)
print("frames", T, "duration s", T / 30, "reps", len(windows))

ik = AnalyticalIKSolver()
ve = TriangulatedValgusEstimator()
ang = kin.all_angles(truth)
max_diff = {}
for i in range(0, T, 7):
    sk = Skeleton3D.from_numpy(truth[i], confidences=[0.9] * 19, timestamp=i / 30, frame_index=i)
    ja = ik.solve(sk)
    vr = ve.estimate(None, sk)
    for name, val in (("knee_l", ja.knee_flexion_l), ("hip_l", ja.hip_flexion_l), ("knee_r", ja.knee_flexion_r),
                      ("trunk", ja.trunk_flexion), ("valgus_l", vr.valgus_l), ("valgus_r", vr.valgus_r)):
        max_diff[name] = max(max_diff.get(name, 0), abs(val - ang[name][i]))
print("max |vectorised - repo| deg:", {k: round(v, 6) for k, v in max_diff.items()})

print("knee flex range", ang["knee_l"].min().round(1), ang["knee_l"].max().round(1))
print("hip flex range", ang["hip_l"].min().round(1), ang["hip_l"].max().round(1))
print("rep signal cm range", kin.rep_signal_cm(truth).min().round(1), kin.rep_signal_cm(truth).max().round(1))
for r, (a, b0, b1, e) in enumerate(windows):
    seg = slice(a, e + 1)
    kv = np.abs(np.gradient(ang["knee_l"][seg]) * 30).max()
    knee_rel_v = np.linalg.norm(np.gradient(truth[seg, CK.LEFT_KNEE], axis=0) * 30, axis=1).max()
    ank_rel_v = np.linalg.norm(np.gradient(truth[seg, CK.LEFT_ANKLE], axis=0) * 30, axis=1).max()
    sh_rel_v = np.linalg.norm(np.gradient(truth[seg, CK.LEFT_SHOULDER], axis=0) * 30, axis=1).max()
    print(f"rep{r}: peak knee {ang['knee_l'][seg].max():.1f}/{ang['knee_r'][seg].max():.1f} deg, "
          f"peak knee vel {kv:.0f} deg/s, peak |v| knee {knee_rel_v:.2f} ankle {ank_rel_v:.2f} "
          f"shoulder {sh_rel_v:.2f} m/s (hip-centred); peak valgus L {ang['valgus_l'][seg].max():.1f} "
          f"R {ang['valgus_r'][seg].max():.1f}")

rng = np.random.default_rng(0)
for sig in (2.0, 4.0, 8.0):
    noisy, conf, reproj = synth.triangulated_noise(world, rng, sigma_px=sig)
    err = noisy - truth
    legs = [CK.LEFT_KNEE, CK.RIGHT_KNEE, CK.LEFT_ANKLE, CK.RIGHT_ANKLE, CK.LEFT_HIP]
    print(f"sigma2D {sig}px: 3D err std per axis (x,y,z) cm legs:", (err[:, legs].reshape(-1, 3).std(0) * 100).round(2),
          "conf p5/p50/p95", np.percentile(conf, [5, 50, 95]).round(3), "reproj p50", np.median(reproj).round(2))
    na = kin.all_angles(noisy)
    idle = np.array([p == "idle" for p in phase])
    print("   knee flex err std deg (idle):", (na["knee_l"] - ang["knee_l"])[idle].std().round(2),
          " valgus err std:", (na["valgus_l"] - ang["valgus_l"])[idle].std().round(2),
          " rep signal err std cm:", (kin.rep_signal_cm(noisy) - kin.rep_signal_cm(truth)).std().round(2))
noisy_q0, _, _ = synth.triangulated_noise(world, np.random.default_rng(1), sigma_px=0.0, quantize=True)
print("quantisation only: 3D err std cm", ((noisy_q0 - truth)[:, legs].reshape(-1, 3).std(0) * 100).round(2))
