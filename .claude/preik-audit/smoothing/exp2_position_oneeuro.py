"""E2: current position One Euro (0.8 Hz, beta 4, dcut 1): effective cutoff, group delay, lag, depth undershoot,
valgus attenuation, bone chord shortening; derivative variant; bone-direction parameterisation."""
from __future__ import annotations

import json
import math

import numpy as np

import harness as H
import smoothers as sm
from biomechanics.utils.types import CocoKeypoints as CK

SEEDS = [0, 1, 2, 3, 4]
results: dict = {}

# ---- effective cutoff / group delay trace on knee keypoint (sigma 4 px, seed 0)
d = H.make_data(0, 4.0)
f = sm.OneEuroVec(0.8, 4.0, 1.0, "raw")
cut = []
for i in range(len(d["noisy"])):
    f.step(d["noisy"][i], d["ts"][i])
    cut.append(f.last_cutoff[CK.LEFT_KNEE].copy())
cut = np.array(cut)  # (T,3)
phase = np.array(d["phase"])
trace = {}
for ph in ("idle", "descending", "bottom", "ascending"):
    m = phase == ph
    c = cut[m]
    fc = np.median(c, axis=0)
    trace[ph] = dict(cutoff_hz_xyz_median=np.round(fc, 2).tolist(),
                     group_delay_ms_xyz=np.round(1000 / (2 * math.pi * fc), 0).tolist())
# at peak descent speed
mids = []
for (a, b0, b1, e) in d["windows"]:
    vel = np.gradient(d["s"][a:b0 + 1])
    mids.append(a + int(np.argmax(vel)))
fc_mid = np.median(cut[mids], axis=0)
trace["peak_descent_velocity_frames"] = dict(cutoff_hz_xyz_median=np.round(fc_mid, 2).tolist(),
                                             group_delay_ms_xyz=np.round(1000 / (2 * math.pi * fc_mid), 0).tolist())
results["effective_cutoff_knee_sigma4"] = trace


def evaluate(make_filter, sigma, noise="tri", seeds=SEEDS, **kw):
    ms = []
    per_rep_depth, per_rep_sig, per_rep_valg = [], [], []
    for sd in seeds:
        dd = H.make_data(sd, sigma, noise=noise)
        r = H.run_stack(dd, pos_filter=make_filter() if make_filter else None, jaf=False, predictive=False,
                        phase_source="oracle", **kw)
        m = H.angle_metrics(r["angles"], dd)
        m.update(H.position_metrics(r["skel"], dd))
        per_rep_depth.append(m["depth_err_per_rep_deg"]); per_rep_sig.append(m["rep_signal_peak_err_cm"])
        per_rep_valg.append(m["valgus_peak_err_deg"])
        ms.append(m)
    out = H.summarize(ms)
    out["depth_err_per_rep_deg"] = np.round(np.mean(per_rep_depth, axis=0), 1).tolist()
    out["rep_signal_peak_err_per_rep_cm"] = np.round(np.mean(per_rep_sig, axis=0), 2).tolist()
    out["valgus_peak_err_per_rep_deg"] = np.round(np.mean(per_rep_valg, axis=0), 1).tolist()
    return out


configs = {
    "raw": None,
    "oneeuro_current_rawderiv": lambda: sm.OneEuroVec(0.8, 4.0, 1.0, "raw"),
    "oneeuro_current_paperderiv": lambda: sm.OneEuroVec(0.8, 4.0, 1.0, "filtered"),
    "bonedir_oneeuro_current": lambda: sm.BoneDirSmoother(sm.OneEuroVec(0.8, 4.0, 1.0, "raw")),
}
results["noise_free_lag_only"] = {n: evaluate(c, 0.0, noise="none", seeds=[0]) for n, c in configs.items()}
for sigma in (2.0, 4.0, 8.0):
    results[f"sigma2d_{sigma:g}px"] = {n: evaluate(c, sigma) for n, c in configs.items()}
    results[f"sigma2d_{sigma:g}px"]["blend_only"] = evaluate(None, sigma, blend=True)

with open("exp2_results.json", "w") as fh:
    json.dump(results, fh, indent=1)

KEYS = ["pos_rms_cm", "pos_rms_moving_cm", "standing_jitter_mm_per_frame", "knee_rms_deg", "knee_rms_moving_deg",
        "knee_std_standing_deg", "knee_lag_ms", "mid_descent_err_deg", "depth_err_mean_deg", "depth_err_worst_deg",
        "rep_signal_peak_err_mean_cm", "valgus_peak_err_mean_deg", "thigh_bias_moving_cm", "thigh_min_minus_true_cm",
        "shank_bias_moving_cm"]
print(json.dumps(results["effective_cutoff_knee_sigma4"], indent=0))
for block, cfgs in results.items():
    if block.startswith("effective"):
        continue
    print("\n###", block)
    print("config".ljust(30) + "".join(k[:14].rjust(15) for k in KEYS))
    for n, m in cfgs.items():
        print(n.ljust(30) + "".join(f"{m.get(k, float('nan')):15.2f}" for k in KEYS))
    for n, m in cfgs.items():
        print("  ", n, "depth/rep deg", m["depth_err_per_rep_deg"], "sig/rep cm", m["rep_signal_peak_err_per_rep_cm"],
              "valgus/rep", m["valgus_peak_err_per_rep_deg"])
