"""E1: OneEuroFilter correctness vs Casiez reference, edge cases (duplicate frames, clock mix, reset)."""
from __future__ import annotations

import math
import sys
import time

import numpy as np

sys.path.insert(0, "/Users/naiahoard/NowvaLiveKit/src")
sys.path.insert(0, "/Users/naiahoard/NowvaLiveKit/.claude/preik-audit/smoothing")

from biomechanics.utils.derivatives import DerivativeTracker  # noqa: E402
from biomechanics.utils.filters import JointAngleFilter, OneEuroFilter  # noqa: E402
from biomechanics.utils.position_filter import KeypointPositionSmoother  # noqa: E402
from biomechanics.utils.predictive_state import PredictiveStateEstimator  # noqa: E402
from biomechanics.utils.types import JointAngles, Skeleton3D  # noqa: E402

import casiez_reference_OneEuroFilter as ref  # noqa: E402
import kin  # noqa: E402
import smoothers as sm  # noqa: E402
import synth  # noqa: E402

rng = np.random.default_rng(0)
world, s, phase, windows = synth.session()
truth = synth.recenter(world)
noisy, conf, _ = synth.triangulated_noise(world, rng, sigma_px=4.0)
T = len(noisy)
ts = np.arange(T) / 30.0 + 1000.0

print("== E1a repo KeypointPositionSmoother vs vectorised OneEuroVec(raw-deriv)")
repo = KeypointPositionSmoother(0.8, 4.0, 1.0)
vec = sm.OneEuroVec(0.8, 4.0, 1.0, "raw")
mx = 0.0
for i in range(T):
    sk = Skeleton3D.from_numpy(noisy[i], confidences=list(conf[i]), timestamp=ts[i], frame_index=i)
    a = repo.smooth(sk).to_numpy()
    b = vec.step(noisy[i], ts[i])
    mx = max(mx, np.abs(a - b).max())
print("max abs diff (m):", mx)

print("== E1b Casiez reference (github main) derivative source")
x = noisy[:, 13, 1]
r = ref.OneEuroFilter(freq=30.0, mincutoff=0.8, beta=4.0, dcutoff=1.0)
out_ref = np.array([r(v, t) for v, t in zip(x, ts)])
f_raw = OneEuroFilter(0.8, 4.0, 1.0)
out_repo = np.array([f_raw.filter(v, t) for v, t in zip(x, ts)])
v_filt = sm.OneEuroVec(0.8, 4.0, 1.0, "filtered")
out_filt = np.array([v_filt.step(np.array([[v]]), t)[0, 0] for v, t in zip(x, ts)])
print("max |reference - repo(raw-prev)| m:", np.abs(out_ref - out_repo).max().round(5),
      " max |reference - vec(filtered-prev)| m:", np.abs(out_ref - out_filt).max())

print("== E1c effective cutoff at standing, current position params, sigma2D=4px")
idle = np.array([p == "idle" for p in phase])
for variant in ("raw", "filtered"):
    f = sm.OneEuroVec(0.8, 4.0, 1.0, variant)
    cut = []
    for i in range(T):
        f.step(noisy[i], ts[i])
        cut.append(f.last_cutoff.copy())
    cut = np.array(cut)
    legs = [13, 14, 15, 16]
    print(f"  deriv={variant}: standing cutoff Hz p50 {np.median(cut[idle][:, legs]):.2f} "
          f"p95 {np.percentile(cut[idle][:, legs], 95):.2f}; moving p50 {np.median(cut[~idle][:, legs]):.2f} "
          f"max {cut[~idle][:, legs].max():.2f}")

print("== E1d duplicate frames (same timestamp, same value) through repo OneEuroFilter")
f = OneEuroFilter(1.0, 0.007, 1.0)
angles = 60 + 50 * np.sin(np.linspace(0, 3, 40))
outs = []
for i, a in enumerate(angles):
    outs.append(f.filter(a, 10 + i / 30))
    if i == 20:
        dup = f.filter(a, 10 + i / 30)
        print("  dup output - previous output:", dup - outs[-1])

print("== E1e duplicate frame where upstream EMA (ConfidenceBlender) changes value slightly")


def run_angle_stack(knee_series, t_series, label):
    jaf = JointAngleFilter(min_cutoff=1.0, beta=0.007)
    der = DerivativeTracker(smoothing_alpha=0.3)
    pred = PredictiveStateEstimator(0.2, 15.0)
    vels, preds, filt, accs = [], [], [], []
    for kv, tt in zip(knee_series, t_series):
        ja = JointAngles(knee_flexion_l=kv, knee_flexion_r=kv, hip_flexion_l=kv, hip_flexion_r=kv,
                         timestamp=tt, frame_index=0)
        jaf.update_phase("descending")
        fa = jaf.filter_angles(ja)
        d = der.update(fa)
        p = pred.predict(fa, d)
        vels.append(d.knee_velocity_l); accs.append(d.knee_acceleration_l); preds.append(p.knee_flexion_l - fa.knee_flexion_l); filt.append(fa.knee_flexion_l)
    vels, preds = np.array(vels), np.array(preds)
    print(f"  {label}: max |knee vel| {np.abs(vels).max():.0f} deg/s, max |acc| {np.abs(np.array(accs)).max():.3g} deg/s2, frames with |pred-cur|>=14.9 deg: "
          f"{int((np.abs(preds) >= 14.9).sum())}, max |pred-cur| {np.abs(preds).max():.1f}")
    return vels, preds


t0 = 5000.0
knee_true = 4 + 116 * 0.5 * (1 - np.cos(np.pi * np.arange(60) / 30.0))  # descent over 1 s then ascent
base_t = t0 + np.arange(60) / 30.0
run_angle_stack(knee_true, base_t, "clean 30 Hz")
# duplicates: frames 10,20,30,40 repeated with identical timestamp, value nudged by 0.3 deg (blender re-blend)
k2, t2 = [], []
for i in range(60):
    k2.append(knee_true[i]); t2.append(base_t[i])
    if i in (10, 20, 30, 40):
        k2.append(knee_true[i] + 0.3); t2.append(base_t[i])
run_angle_stack(np.array(k2), np.array(t2), "dup frames (+0.3 deg)")

print("== E1f dropout-hold frame stamped time.time() amid perf_counter timestamps")
k3, t3 = [], []
now_wall = time.time()
for i in range(60):
    k3.append(knee_true[i]); t3.append(base_t[i])
    if i == 15:
        k3.append(knee_true[i]); t3.append(now_wall)   # held skeleton, same angles, wall clock
vel3, pred3 = run_angle_stack(np.array(k3), np.array(t3), "clock-mix hold frame")
bad = np.where(np.abs(pred3) >= 14.9)[0]
print("  predicted clamp frames index range:", (bad.min(), bad.max()) if len(bad) else None,
      " velocity right after:", np.round(vel3[16:20]).tolist())

print("== E1g no reset of JointAngleFilter/DerivativeTracker across a 90 s rest (pipeline.reset_readiness_gate)")
jaf = JointAngleFilter(); der = DerivativeTracker(0.3); pred = PredictiveStateEstimator()
for i in range(30):
    ja = JointAngles(knee_flexion_l=100.0 - i, knee_flexion_r=100.0 - i, timestamp=100 + i / 30, frame_index=i)
    fa = jaf.filter_angles(ja); d = der.update(fa)
ja = JointAngles(knee_flexion_l=5.0, knee_flexion_r=5.0, timestamp=100 + 1 + 90, frame_index=31)
fa = jaf.filter_angles(ja); d = der.update(fa); p = pred.predict(fa, d)
print(f"  first frame after rest: filtered {fa.knee_flexion_l:.2f} (raw 5.0), vel {d.knee_velocity_l:.2f}, "
      f"pred {p.knee_flexion_l:.2f}")
