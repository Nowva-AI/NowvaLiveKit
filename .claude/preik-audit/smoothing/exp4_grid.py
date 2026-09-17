"""E4: grid search position smoothers on triangulated noise.

Objective J (degrees, equal weights): knee RMS error while moving + mean |peak depth error| per rep
+ mean |peak valgus error| per valgus rep + standing knee-angle std.  Fixed-lag outputs are re-aligned
by their lag (accuracy of the estimate) and the latency is reported separately.
"""
from __future__ import annotations

import itertools
import json
import sys
import time

import numpy as np
from scipy.signal import butter, filtfilt

import harness as H
import kin
import smoothers as sm

SEEDS = [0, 1, 2]
SIGMA = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0


def angles_dict(k):
    a = kin.all_angles(k)
    return {"knee_flexion_l": a["knee_l"], "knee_flexion_r": a["knee_r"], "hip_flexion_l": a["hip_l"],
            "hip_flexion_r": a["hip_r"], "knee_valgus_l": a["valgus_l"], "knee_valgus_r": a["valgus_r"]}


def fast_eval(filtered, d, lag_frames=0):
    if lag_frames:
        est = np.concatenate([filtered[lag_frames:], np.repeat(filtered[-1:], lag_frames, axis=0)])
    else:
        est = filtered
    m = H.angle_metrics(angles_dict(est), d)
    m.update(H.position_metrics(est, d))
    m["depth_abs_mean_deg"] = float(np.mean(np.abs(m["depth_err_per_rep_deg"])))
    m["valgus_abs_mean_deg"] = float(np.mean(np.abs(m["valgus_peak_err_deg"])))
    m["J"] = m["knee_rms_moving_deg"] + m["depth_abs_mean_deg"] + m["valgus_abs_mean_deg"] + m["knee_std_standing_deg"]
    m["latency_ms"] = lag_frames * 1000 / 30
    return m


DATA = {kind: [H.make_data(sd, SIGMA, kind=kind) for sd in SEEDS] for kind in ("smooth", "hard")}


def score(make, lag_frames=0, offline=None):
    res = {}
    for kind, datas in DATA.items():
        ms = []
        for d in datas:
            if offline is not None:
                out = offline(d)
            else:
                out = sm.run_causal(make(), d["noisy"], d["ts"], d["conf"])
            ms.append(fast_eval(out, d, lag_frames))
        s = H.summarize(ms)
        s["depth_err_mean_signed_deg"] = s.pop("depth_err_mean_deg")
        res[kind] = s
    out = dict(res["smooth"])
    for k, v in res["hard"].items():
        out["hard_" + k] = v
    out["J_mean"] = (out["J"] + out["hard_J"]) / 2
    return out


rows = []
t0 = time.time()


class _Identity:
    def reset(self):
        pass

    def step(self, x, t, conf=None):
        return x.copy()


rows = [("raw", {}, score(lambda: _Identity()))]

for variant, mc, beta, dc in itertools.product(("raw", "filtered"), (0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0),
                                               (0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0), (0.5, 1.0, 2.0, 4.0)):
    rows.append((f"oneeuro_{variant}", dict(min_cutoff=mc, beta=beta, d_cutoff=dc),
                 score(lambda: sm.OneEuroVec(mc, beta, dc, variant))))
print("one euro grid done", round(time.time() - t0, 1), "s", flush=True)

for q, r, conf_ref, lag in itertools.product((0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0), (0.01, 0.02, 0.04),
                                             (None, 0.5), (0, 1, 2, 3, 5)):
    rows.append(("kalman_cv", dict(q=q, r_m=r, conf_ref=conf_ref, lag=lag),
                 score(lambda: sm.KalmanVec(2, q, r, lag, conf_ref), lag_frames=lag)))
print("kalman cv grid done", round(time.time() - t0, 1), "s", flush=True)

for q, r, lag in itertools.product((3.0, 10.0, 30.0, 100.0, 300.0, 1e3), (0.01, 0.02), (0, 2, 3)):
    rows.append(("kalman_ca", dict(q=q, r_m=r, lag=lag), score(lambda: sm.KalmanVec(3, q, r, lag, None), lag_frames=lag)))
print("kalman ca grid done", round(time.time() - t0, 1), "s", flush=True)

for q, r in itertools.product((0.1, 0.3, 1.0, 3.0, 10.0, 30.0), (0.01, 0.02, 0.04)):
    rows.append(("rts_offline_cv", dict(q=q, r_m=r),
                 score(None, offline=lambda d: sm.rts_offline(d["noisy"], d["ts"], 2, q, r))))
for fc in (1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0):
    b, a = butter(2, fc / 15.0)
    rows.append(("butter_filtfilt_offline", dict(cutoff_hz=fc, order="2x2"),
                 score(None, offline=lambda d: filtfilt(b, a, d["noisy"], axis=0))))
print("offline done", round(time.time() - t0, 1), "s", flush=True)

with open(f"exp4_grid_sigma{SIGMA:g}.json", "w") as fh:
    json.dump([dict(name=n, params=p, **m) for n, p, m in rows], fh, indent=0)

KEYS = ["J_mean", "J", "hard_J", "hard_depth_err_mean_signed_deg", "hard_valgus_peak_err_mean_deg", "hard_knee_rms_moving_deg", "knee_rms_moving_deg", "depth_abs_mean_deg", "depth_err_mean_signed_deg", "valgus_abs_mean_deg",
        "valgus_peak_err_mean_deg", "knee_std_standing_deg", "knee_lag_ms", "latency_ms", "pos_rms_moving_cm",
        "rep_signal_peak_err_mean_cm", "standing_jitter_mm_per_frame", "thigh_min_minus_true_cm"]


def show(title, subset):
    print("\n##", title)
    print("name".ljust(24) + "params".ljust(58) + "".join(k.replace("_mean","").replace("_deg","")[-12:].rjust(13) for k in KEYS))
    for n, p, m in subset:
        print(n.ljust(24) + json.dumps(p).ljust(58) + "".join(f"{m[k]:13.2f}" for k in KEYS))


show("raw", rows[:1])
for fam in ("oneeuro_raw", "oneeuro_filtered", "kalman_cv", "kalman_ca", "rts_offline_cv", "butter_filtfilt_offline"):
    sub = sorted([r for r in rows if r[0] == fam], key=lambda r: r[2]["J_mean"])
    show(f"{fam} top 5 by J", sub[:5])
for lag in (0, 1, 2, 3, 5):
    sub = sorted([r for r in rows if r[0] == "kalman_cv" and r[1]["lag"] == lag], key=lambda r: r[2]["J_mean"])
    show(f"kalman_cv lag={lag} top 3", sub[:3])
cur = [r for r in rows if r[0] == "oneeuro_raw" and r[1] == dict(min_cutoff=0.8, beta=4.0, d_cutoff=1.0)]
show("CURRENT position params (0.8, 4.0, 1.0) raw-deriv", cur)
