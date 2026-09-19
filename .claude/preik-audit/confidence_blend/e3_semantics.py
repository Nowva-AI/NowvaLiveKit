"""E3: confidence semantics.
(a) Simulated triangulation (3 views and 2 views) with 2D outliers: does triangulated confidence predict 3D error?
    Does the blender's weight actually suppress bad points and pass good ones?
(b) Real single-camera MediaPipe runs (user_test_runs pipeline_inspect): back out the blend weight the production
    blender applied per keypoint/frame from the recorded raw and confidence_blend stages.
"""
from __future__ import annotations

import glob
import json

import numpy as np
from scipy.stats import spearmanr

import common as C

CK = C.CK
LEG = [CK.LEFT_HIP, CK.RIGHT_HIP, CK.LEFT_KNEE, CK.RIGHT_KNEE, CK.LEFT_ANKLE, CK.RIGHT_ANKLE, 17, 18]
res: dict = {}

# ------------------------------------------------------------------ (a) simulation
rng = np.random.default_rng(11)
world, s, phase, windows = C.synth.session(fps=30.0)
world = np.concatenate([world] * 4)  # more samples
T = world.shape[0]
true_abs = world
P3 = C.synth.cameras()


def tri_case(P: np.ndarray, sigma_px: float, outlier_p: float, outlier_px: float):
    uv = C.synth.project(P, true_abs)
    uv_noisy = uv + rng.normal(0.0, sigma_px, uv.shape)
    mask = rng.random(uv.shape[:-1]) < outlier_p
    uv_noisy += mask[..., None] * rng.normal(0.0, outlier_px, uv.shape)
    uv_noisy[..., 0] = np.round(uv_noisy[..., 0] / C.synth.QUANT_X_PX) * C.synth.QUANT_X_PX
    uv_noisy[..., 1] = np.round(uv_noisy[..., 1] / C.synth.QUANT_Y_PX) * C.synth.QUANT_Y_PX
    X = C.synth.dlt(P, uv_noisy)
    reproj = np.linalg.norm(C.synth.project(P, X) - uv_noisy, axis=-1).mean(axis=-1)
    vc = C.view_conf(T, rng, P.shape[0])
    base = vc.min(axis=-1)
    conf = np.clip(base * np.where(reproj < 15.0, 1.0 - reproj / 15.0, 0.1), 0, 1)
    err_abs = np.linalg.norm(X - true_abs, axis=-1) * 100  # cm, before recentre (per-point quality)
    any_out = mask.any(axis=-1)
    return conf, reproj, err_abs, any_out


cases = {}
for name, P, op in [("3view_outlier3pct_40px", P3, 0.03), ("2view_outlier3pct_40px", P3[[0, 2]], 0.03),
                    ("2view_adjacent_cams_outlier3pct", P3[[0, 1]], 0.03)]:
    conf, reproj, err, out = tri_case(P, 3.0, op, 40.0)
    c = conf[:, LEG].ravel(); e = err[:, LEG].ravel(); o = out[:, LEG].ravel(); r = reproj[:, LEG].ravel()
    w = C.blend_weights(c)
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]
    by_bin = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (c >= lo) & (c < hi)
        if m.sum() >= 20:
            by_bin[f"{lo:.1f}-{hi:.1f}"] = dict(n=int(m.sum()), err_p50_cm=round(float(np.median(e[m])), 2),
                                                err_p95_cm=round(float(np.percentile(e[m], 95)), 2))
    big = e > 5.0
    good = e < 2.0
    cases[name] = dict(
        spearman_conf_vs_err=round(float(spearmanr(c, e).correlation), 3),
        spearman_reproj_vs_err=round(float(spearmanr(r, e).correlation), 3),
        err_by_conf_bin=by_bin,
        frac_points_err_gt_5cm=round(float(big.mean()), 4),
        of_err_gt_5cm_frac_weight_lt_0p2=round(float((w[big] < 0.2).mean()), 3) if big.any() else None,
        of_err_gt_5cm_weight_p50=round(float(np.median(w[big])), 3) if big.any() else None,
        of_err_gt_10cm_frac_weight_lt_0p2=round(float((w[e > 10] < 0.2).mean()), 3) if (e > 10).any() else None,
        of_err_lt_2cm_frac_weight_lt_0p5=round(float((w[good] < 0.5).mean()), 3),
        of_err_lt_2cm_weight_p50=round(float(np.median(w[good])), 3),
        outlier_points_err_p50_cm=round(float(np.median(e[o])), 2), outlier_points_reproj_p50=round(float(np.median(r[o])), 2),
        outlier_points_conf_p50=round(float(np.median(c[o])), 3),
        max_weight_seen=round(float(w.max()), 3),
        frac_conf_ge_0p9=round(float((c >= 0.9).mean()), 4),
    )
res["sim_conf_vs_error"] = cases

# ------------------------------------------------------------------ (b) real MediaPipe single-cam weights
names21 = ["nose", "leye", "reye", "lear", "rear", "lsho", "rsho", "lelb", "relb", "lwri", "rwri", "lhip", "rhip",
           "lknee", "rknee", "lank", "rank", "ltoe", "rtoe", "lheel", "rheel"]
all_w = {n: [] for n in names21}
consistency = []
for path in sorted(glob.glob("/Users/naiahoard/NowvaLiveKit/user_test_runs/*/output/*/pipeline_inspect/data.json")):
    d = json.load(open(path))
    raw = d["skeletons"]["raw"]; cb = d["skeletons"]["confidence_blend"]; ready = d["series"]["ready"]
    for t in range(1, len(raw)):
        if raw[t] is None or cb[t] is None or cb[t - 1] is None or not ready[t] or not ready[t - 1]:
            continue
        r = np.array(raw[t]).reshape(-1, 3); b = np.array(cb[t]).reshape(-1, 3); bp = np.array(cb[t - 1]).reshape(-1, 3)
        den = r - bp
        for k in range(r.shape[0]):
            ok = np.abs(den[k]) > 0.02  # >2 cm so 0.1 mm rounding is negligible
            if ok.sum() == 0:
                continue
            ws = (b[k] - bp[k])[ok] / den[k][ok]
            if ok.sum() >= 2:
                consistency.append(float(ws.max() - ws.min()))
            all_w[names21[k]].append(float(np.median(ws)))
real = {}
for n, ws in all_w.items():
    if len(ws) < 20:
        continue
    a = np.clip(np.array(ws), -0.2, 1.2)
    real[n] = dict(n=len(a), w_p5=round(float(np.percentile(a, 5)), 3), w_p50=round(float(np.median(a)), 3),
                   frac_w_ge_0p99=round(float((a >= 0.99).mean()), 3), frac_w_lt_0p5=round(float((a < 0.5).mean()), 3),
                   implied_conf_p5=round(float(0.1 + 0.8 * np.percentile(a, 5)), 3))
res["real_mediapipe_backed_out_weights"] = real
res["real_weight_axis_consistency_p50_p95"] = [round(float(np.percentile(consistency, q)), 4) for q in (50, 95)]

(C.HERE / "e3_results.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
