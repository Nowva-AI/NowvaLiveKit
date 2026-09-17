"""E7: edge cases — missing-value sentinels through filters, display smoother efficacy, JAF phase params, chord shortening."""
from __future__ import annotations

import json

import numpy as np

import harness as H
import smoothers as sm
import synth
from biomechanics.utils.filters import JointAngleFilter
from biomechanics.utils.position_filter import KeypointPositionSmoother, Skeleton2DSmoother
from biomechanics.utils.types import CocoKeypoints as CK, JointAngles, Keypoint2D, Skeleton2D, Skeleton3D
from fixedlag_fast import FixedLagCV

print("== E7a JointAngleFilter fed IK's missing-keypoint sentinel (knee_flexion=0.0 when a keypoint conf<0.1)")
jaf = JointAngleFilter(1.0, 0.007)
outs = []
seq = [60 + 2.0 * i for i in range(20)]
seq_missing = list(seq)
seq_missing[10] = 0.0
seq_missing[11] = 0.0
for i, v in enumerate(seq_missing):
    jaf.update_phase("descending")
    fa = jaf.filter_angles(JointAngles(knee_flexion_l=v, knee_flexion_r=v, timestamp=10 + i / 30, frame_index=i))
    outs.append(fa.knee_flexion_l)
print("  raw   :", [round(v, 1) for v in seq_missing[8:16]])
print("  output:", [round(v, 1) for v in outs[8:16]], " (truth", [round(v, 1) for v in seq[8:16]], ")")

print("== E7b KeypointPositionSmoother (if re-enabled) fed a non-triangulable ankle = (0,0,0) conf 0 for 2 frames")
d = H.make_data(0, 0.0, noise="none")
X = d["truth"].copy()
C = np.full(X.shape[:2], 0.6)
i0 = d["windows"][0][0] + 15
X[i0:i0 + 2, CK.LEFT_ANKLE] = 0.0
C[i0:i0 + 2, CK.LEFT_ANKLE] = 0.0
ps = KeypointPositionSmoother(0.8, 4.0, 1.0)
kl = []
for i in range(i0 - 2, i0 + 6):
    out = ps.smooth(Skeleton3D.from_numpy(X[i], confidences=list(C[i]), timestamp=d["ts"][i], frame_index=i)).to_numpy()
    kl.append(round(float(np.linalg.norm(out[CK.LEFT_ANKLE] - d["truth"][i, CK.LEFT_ANKLE]) * 100), 1))
print("  ankle error cm frames i0-2..i0+5 (filter warm-started 2 frames early):", kl)
ps.reset()
kl = []
for i in range(0, i0 + 6):
    out = ps.smooth(Skeleton3D.from_numpy(X[i], confidences=list(C[i]), timestamp=d["ts"][i], frame_index=i)).to_numpy()
    if i >= i0 - 2:
        kl.append(round(float(np.linalg.norm(out[CK.LEFT_ANKLE] - d["truth"][i, CK.LEFT_ANKLE]) * 100), 1))
print("  ankle error cm (filter warm):", kl)
fl = FixedLagCV(3.0, 0.01, 2)
kl = []
for i in range(0, i0 + 8):
    r2 = np.where(C[i] > 0, 1e-4, 1e6)[:, None] * np.ones((1, 3))  # conf 0 -> effectively no update
    out = fl.step(X[i], d["ts"][i], r2)
    if i >= i0:
        kl.append(round(float(np.linalg.norm(out[CK.LEFT_ANKLE] - d["truth"][i - 2, CK.LEFT_ANKLE]) * 100), 2))
print("  FixedLagCV with R=inf for conf-0 points, ankle err cm (lag-aligned):", kl)

print("== E7c Skeleton2DSmoother (display) on primary-camera 2D keypoints, 4 px noise")
w, s, phase, windows = synth.session()
P = synth.cameras()
uv = synth.project(P, w)[:, :, 1, :]  # camera 1 (front)
rng = np.random.default_rng(0)
uvn = uv + rng.normal(0, 4.0, uv.shape)
sm2 = Skeleton2DSmoother(1.5, 0.5, 1.0)
out2 = []
for i in range(len(uvn)):
    kps = [Keypoint2D(x=float(uvn[i, k, 0]), y=float(uvn[i, k, 1]), confidence=0.6) for k in range(19)]
    sk = Skeleton2D(keypoints=kps, timestamp=1000 + i / 30, frame_index=i)
    out2.append(sm2.smooth(sk).to_numpy()[:, :2])
out2 = np.array(out2)
idle = np.array([p == "idle" for p in phase]); idle[:30] = False
legs = [11, 12, 13, 14, 15, 16]
jit_raw = np.sqrt(np.mean(np.sum(np.diff(uvn[:, legs], axis=0) ** 2, -1)[idle[1:]]))
jit_f = np.sqrt(np.mean(np.sum(np.diff(out2[:, legs], axis=0) ** 2, -1)[idle[1:]]))
err_mov_raw = np.sqrt(np.mean(np.sum((uvn[:, legs] - uv[:, legs]) ** 2, -1)[~idle]))
err_mov_f = np.sqrt(np.mean(np.sum((out2[:, legs] - uv[:, legs]) ** 2, -1)[~idle]))
print(f"  standing jitter px/frame raw {jit_raw:.2f} -> smoothed {jit_f:.2f}; moving RMS err px raw {err_mov_raw:.2f} -> {err_mov_f:.2f}")
alt = sm.OneEuroVec(1.0, 0.02, 1.0, "filtered")
out3 = np.stack([alt.step(uvn[i], 1000 + i / 30) for i in range(len(uvn))])
jit_a = np.sqrt(np.mean(np.sum(np.diff(out3[:, legs], axis=0) ** 2, -1)[idle[1:]]))
err_a = np.sqrt(np.mean(np.sum((out3[:, legs] - uv[:, legs]) ** 2, -1)[~idle]))
print(f"  px-unit retuned OneEuro(1.0, beta 0.02, dcut 1.0, paper deriv): jitter {jit_a:.2f}, moving RMS err {err_a:.2f}")

print("== E7d JointAngleFilter phase-aware vs fixed params, oracle vs counter phase (sigma 4 px, 5 seeds, smooth)")
res = {}
for label, kw in (("fixed_default(1.0,0.007)", dict(phase_aware=False, phase_source="oracle")),
                  ("phase_aware_oracle", dict(phase_aware=True, phase_source="oracle")),
                  ("phase_aware_counter", dict(phase_aware=True, phase_source="counter"))):
    ms = []
    for sd in range(5):
        dd = H.make_data(sd, 4.0)
        r = H.run_stack(dd, jaf=True, predictive=False, **kw)
        ms.append(H.angle_metrics(r["angles"], dd))
    res[label] = {k: v for k, v in H.summarize(ms).items() if k in ("knee_rms_moving_deg", "knee_lag_ms",
                  "mid_descent_err_deg", "depth_err_mean_deg", "depth_err_worst_deg", "valgus_peak_err_mean_deg",
                  "knee_std_standing_deg")}
    print("  ", label, json.dumps(res[label]))

print("== E7e chord shortening from lag only (noise-free): thigh length min/mean bias cm")
d0 = H.make_data(0, 0.0, noise="none")
for label, f in (("oneeuro_current", sm.OneEuroVec(0.8, 4.0, 1.0, "raw")), ("oneeuro_tuned(1.2,16,2)", sm.OneEuroVec(1.2, 16.0, 2.0, "filtered")),
                 ("kf_cv_lag0 q3", FixedLagCV(3.0, 0.01, 0)), ("kf_cv_lag2 q3", FixedLagCV(3.0, 0.01, 2))):
    f.reset()
    out = np.stack([f.step(d0["noisy"][i], d0["ts"][i]) for i in range(len(d0["noisy"]))])
    lag = 2 if "lag2" in label else 0
    m = H.fast_eval(out, d0, lag)
    print(f"   {label:28s} thigh min {m['thigh_min_minus_true_cm']:.2f} bias {m['thigh_bias_moving_cm']:.3f} "
          f"knee rms moving {m['knee_rms_moving_deg']:.2f} depth {m['depth_err_mean_deg']:.2f} lag {m['knee_lag_ms']:.0f} ms")
