"""Exp 5: propagate the MEASURED full-frame-squash 2D error (FF minus tracked-crop keypoints from exp1, rescaled to the
synthetic person size) through 3-view DLT. Compare with crop-like iid 1.5 px noise. Reports 3D error, knee flexion,
hip width, and how often the production confidence gets the x0.1 penalty (mean reprojection >= 15 px), which with
view conf <= 1 puts the keypoint below AnalyticalIKSolver.MIN_CONFIDENCE (0.1) -> joint angle returns 0.0."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import synth
sys.path.insert(0, str(Path(__file__).parent))
from exp1_analyze import dec_parabola, to_canvas_crop, to_canvas_ff  # noqa: E402

OUT = Path(__file__).parent
RNG = np.random.default_rng(13)
HALPE_TO_21 = list(range(17)) + [20, 21, 24, 25]
d = np.load(OUT / "exp1_logits.npz")
kff = to_canvas_ff(dec_parabola(d["ff_x"]), dec_parabola(d["ff_y"]), d["ff_scale"])
kcr = to_canvas_crop(dec_parabola(d["cr_x"]), dec_parabola(d["cr_y"]), d["centers"], d["scales"])
err2d = (kff - kcr)[:, HALPE_TO_21] * (540.0 / 470.0)  # (T, 21, 2) scaled to synthetic person height
P = synth.rig()
seq, _ = synth.squat_sequence()


def kflex(X):
    u, v = X[11] - X[13], X[15] - X[13]
    return 180 - np.degrees(np.arccos(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)))


res = {}
for mode in ("crop_iid_1.5px", "ff_same_frame_error_all_views", "ff_independent_frames_per_view"):
    e3, kf, hw, pen, eb = [], [], [], [], []
    for f in range(len(seq)):
        gt = seq[f]
        uv, _ = synth.project(P, gt)
        if mode.startswith("crop"):
            uv = uv + RNG.normal(0, 1.5, uv.shape)
        else:
            T = err2d.shape[0]
            idx = [int(RNG.integers(0, T))] * 3 if "same" in mode else list(RNG.integers(0, T, 3))
            for v in range(3):
                uv[v] = uv[v] + err2d[idx[v]]
        X = synth.dlt(P, uv)
        e3.append(np.linalg.norm(X - gt, axis=1).mean() * 1000)
        eb.append(np.linalg.norm(X[5:17] - gt[5:17], axis=1).mean() * 1000)
        kf.append(abs(kflex(X) - kflex(gt)))
        hw.append((np.linalg.norm(X[11] - X[12]) - np.linalg.norm(gt[11] - gt[12])) * 1000)
        pen.append((synth.reproj_err(P, X, uv).mean(0)[:17] >= 15).mean())
    res[mode] = {"kpt_err_mm_mean": float(np.mean(e3)), "body_kpt_err_mm_mean(5-16)": float(np.mean(eb)), "knee_flex_err_deg_mean": float(np.mean(kf)),
                 "knee_flex_err_deg_p95": float(np.percentile(kf, 95)), "hip_width_err_mm_mean(signed)": float(np.mean(hw)),
                 "frac_body_kpts_conf_x0.1": float(np.mean(pen))}
(OUT / "exp5_results.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
