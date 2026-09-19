"""E1: is ConfidenceBlender time-aware? Loop-rate dependence, duplicate/skipped frames, dropout-hold frames.

All numeric runs use the PRODUCTION ConfidenceBlender class.
"""
from __future__ import annotations

import json
import math

import numpy as np

import common as C

OUT = C.HERE
res: dict = {}

# ---------------------------------------------------------------- (a) analytic EMA properties vs loop rate
def ema_cutoff_hz(w: float, fs: float) -> float:
    a = 1.0 - w
    # |H|^2 = w^2 / (1 - 2 a cos(om) + a^2) = 1/2
    cos_om = (1.0 + a * a - 2.0 * w * w) / (2.0 * a) if a > 0 else -1.0
    cos_om = max(-1.0, min(1.0, cos_om))
    return math.acos(cos_om) * fs / (2.0 * math.pi)

table = []
for fs in [11.6, 15.0, 20.0, 30.0, 60.0]:
    for w in [0.37, 0.45, 0.49, 0.53, 1.0]:
        lag_frames = (1 - w) / w
        table.append(dict(fps=fs, w=w, lag_frames=round(lag_frames, 2), lag_ms=round(1000 * lag_frames / fs, 1),
                          lag_cm_at_0p8mps=round(80 * lag_frames / fs, 2),
                          noise_std_ratio=round(math.sqrt(w / (2 - w)), 3),
                          cutoff_hz=round(ema_cutoff_hz(w, fs), 2) if w < 1 else None))
res["analytic"] = table

# numeric check with production class: ramp 0.8 m/s on every keypoint, conf chosen so w = 0.49
w_target = 0.49
conf_val = 0.1 + 0.8 * w_target
num = []
for fs in [11.6, 30.0]:
    T = 120
    t = np.arange(T) / fs
    pts = np.zeros((T, 19, 3)); pts[:, :, 1] = 0.8 * t[:, None]
    out = C.run_blender(pts, np.full((T, 19), conf_val), t)
    lag_m = (pts[-1, 0, 1] - out[-1, 0, 1])
    num.append(dict(fps=fs, w=w_target, measured_lag_cm=round(100 * lag_m, 2), measured_lag_ms=round(1000 * lag_m / 0.8, 1)))
res["numeric_ramp_check"] = num

# ---------------------------------------------------------------- (b) duplicates / skips from loop vs camera beat
def run_sampling(loop_period_ms: float, jitter_ms: float, cam_fps: float, rng: np.random.Generator,
                 sigma_m: float = 0.01, speed: float = 0.8, dur_s: float = 20.0):
    cam_period = 1.0 / cam_fps
    n_cam = int(dur_s * cam_fps)
    cam_t = np.arange(n_cam) * cam_period
    noise = rng.normal(0, sigma_m, n_cam)
    # loop iteration times
    lt = [0.05]
    while lt[-1] < dur_s - 0.1:
        lt.append(lt[-1] + max(0.001, rng.normal(loop_period_ms, jitter_ms) / 1000.0))
    lt = np.array(lt[1:])
    idx = np.searchsorted(cam_t, lt, side="right") - 1  # latest captured frame
    dup = np.r_[False, idx[1:] == idx[:-1]]
    skipped = np.r_[0, np.maximum(idx[1:] - idx[:-1] - 1, 0)]
    T = len(idx)
    # ramp for lag: position = speed * capture time; static noise run separately
    ramp = np.zeros((T, 19, 3)); ramp[:, :, 1] = (speed * cam_t[idx])[:, None]
    stat = np.zeros((T, 19, 3)); stat[:, :, 1] = noise[idx][:, None]
    conf = np.full((T, 19), 0.1 + 0.8 * 0.49)
    ts = cam_t[idx]  # production triangulated timestamp = capture ref_ts (duplicates share it)
    r_out = C.run_blender(ramp, conf, ts)
    s_out = C.run_blender(stat, conf, ts)
    k = slice(T // 4, None)
    lag_cm = 100 * (ramp[k, 0, 1] - r_out[k, 0, 1])
    # temporal lag w.r.t. true (continuous) position at loop time
    true_now = speed * lt
    lag_vs_now_ms = 1000 * (true_now[k] - r_out[k, 0, 1]) / speed
    # noise ratio against the per-camera-frame noise std (distinct frames)
    uniq_noise_std = noise[np.unique(idx)].std()
    return dict(loop_period_ms=loop_period_ms, jitter_ms=jitter_ms, cam_fps=cam_fps,
                dup_frac=round(float(dup.mean()), 3), skip_frames_per_iter=round(float(skipped.mean()), 3),
                lag_cm_p50=round(float(np.median(lag_cm)), 2), lag_cm_p5=round(float(np.percentile(lag_cm, 5)), 2),
                lag_cm_p95=round(float(np.percentile(lag_cm, 95)), 2),
                total_latency_vs_now_ms_p50=round(float(np.median(lag_vs_now_ms)), 1),
                noise_std_ratio=round(float(s_out[k, 0, 1].std() / uniq_noise_std), 3))

rng = np.random.default_rng(1)
samp = []
for lp, jit in [(33.33, 0.0), (33.33, 3.0), (30.0, 3.0), (25.0, 3.0), (16.7, 2.0), (40.0, 4.0), (86.0, 10.0)]:
    samp.append(run_sampling(lp, jit, 30.0, rng))
res["loop_vs_camera"] = samp

# ---------------------------------------------------------------- (c) dropout hold at mid-descent
world, s_arr, phase, windows = C.synth.session()
true = C.synth.recenter(world)
start, b0, b1, end = windows[0]
mid = start + (b0 - start) // 2
T = true.shape[0]
fs = 30.0
ts = np.arange(T) / fs
conf = np.full((T, 19), 0.1 + 0.8 * 0.49)
# pipeline dropout hold: last valid RAW skeleton, conf *= 1 - k/5, k=1..5 (pipeline.py:537-552)
held_pts = true.copy(); held_conf = conf.copy()
for k in range(1, 6):
    held_pts[mid + k] = true[mid]
    held_conf[mid + k] = conf[mid] * (1 - k / 5)
out_b = C.run_blender(held_pts, held_conf, ts)
kn = C.CK.LEFT_KNEE
an = C.CK.LEFT_ANKLE
rows = []
for f in range(mid - 1, mid + 12):
    rows.append(dict(frame=f - mid, held=bool(mid < f <= mid + 5), conf=round(float(held_conf[f, an]), 3),
                     ankle_err_raw_hold_cm=round(float(100 * np.linalg.norm(held_pts[f, an] - true[f, an])), 2),
                     ankle_err_blend_cm=round(float(100 * np.linalg.norm(out_b[f, an] - true[f, an])), 2)))
res["dropout_hold_mid_descent_ankle"] = rows
# weights during the hold
res["dropout_hold_weights"] = [round(float(C.blend_weights(np.array(conf[mid, an] * (1 - k / 5)))), 3) for k in range(1, 6)]

(OUT / "e1_results.json").write_text(json.dumps(res, indent=1))
for k, v in res.items():
    print("==", k)
    if isinstance(v, list):
        for r in v:
            print("  ", r)
    else:
        print("  ", v)
