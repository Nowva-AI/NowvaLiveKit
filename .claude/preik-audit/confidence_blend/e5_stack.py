"""E5: interaction with VelocityClamp (runs after the blender) and with the post-IK JointAngleFilter ->
DerivativeTracker -> PredictiveStateEstimator stack. Production classes throughout.
"""
from __future__ import annotations

import json

import numpy as np

import common as C
import filters_alt as F
from biomechanics.kinematics.analytical_ik import AnalyticalIKSolver
from biomechanics.utils.derivatives import DerivativeTracker
from biomechanics.utils.filters import JointAngleFilter
from biomechanics.utils.predictive_state import PredictiveStateEstimator
from biomechanics.utils.types import Skeleton3D

CK = C.CK
res: dict = {}

# ---------------------------------------------------------------- (a) single-frame outlier that is only partly flagged
std = C.synth.recenter(C.synth.pose(0.0, C.synth.body()))
n = 12
ts = np.arange(n) / 30.0
smear = {}
for spike_cm, conf_spike in [(25.0, 0.4), (25.0, 0.25), (15.0, 0.45)]:
    pts = np.repeat(std[None], n, axis=0); conf = np.full((n, 19), 0.5)
    pts[2, CK.LEFT_KNEE, 2] += spike_cm / 100.0; conf[2, CK.LEFT_KNEE] = conf_spike
    rows = {}
    for name, ub, uc in [("raw", False, False), ("clamp_only", False, True), ("blend_only", True, False), ("blend+clamp(prod)", True, True)]:
        out = C.run_blend_clamp(pts, conf, ts, use_blend=ub, use_clamp=uc)
        e = 100 * np.linalg.norm(out[:, CK.LEFT_KNEE] - std[CK.LEFT_KNEE], axis=-1)
        rows[name] = dict(err_cm=[round(float(x), 1) for x in e[1:9]], integrated_cm_frames=round(float(e.sum()), 1),
                          frames_err_gt_1cm=int((e > 1.0).sum()))
    smear[f"spike_{spike_cm}cm_conf_{conf_spike}_w_{float(C.blend_weights(np.array(conf_spike))):.3f}"] = rows
res["a_outlier_smear"] = smear


# ---------------------------------------------------------------- (b) post-IK stack
def post_ik(pts: np.ndarray, conf: np.ndarray, ts: np.ndarray, phase: list) -> dict:
    ik = AnalyticalIKSolver(); jf = JointAngleFilter(min_cutoff=1.0, beta=0.007); dtk = DerivativeTracker(smoothing_alpha=0.3)
    pr = PredictiveStateEstimator(0.2, 15.0)
    raw_k, filt_k, pred_k = [], [], []
    prev_phase = "idle"
    for t in range(len(pts)):
        sk = Skeleton3D.from_numpy(pts[t], confidences=conf[t], timestamp=float(ts[t]))
        a = ik.solve(sk)
        jf.update_phase(prev_phase)  # rep counter phase is one frame late in pipeline.py
        fa = jf.filter_angles(a)
        d = dtk.update(fa)
        pa = pr.predict(fa, d)
        raw_k.append(a.knee_flexion_l); filt_k.append(fa.knee_flexion_l); pred_k.append(pa.knee_flexion_l)
        prev_phase = phase[t]
    return dict(raw=np.array(raw_k), filt=np.array(filt_k), pred=np.array(pred_k))


def summarize(k: dict, kt: np.ndarray, windows: list, fps: float) -> dict:
    kvel = np.gradient(kt) * fps
    fast = np.abs(kvel) > 100
    out = {}
    for stage in ["raw", "filt", "pred"]:
        e = k[stage] - kt
        peaks = [round(float(k[stage][st:en + 1].max() - kt[st:en + 1].max()), 1) for st, b0, b1, en in windows]
        out[stage] = dict(rmse=round(float(np.sqrt(np.mean(e ** 2))), 2), fast_mean_abs=round(float(np.abs(e[fast]).mean()), 2),
                          fast_desc_mean_signed=round(float(e[fast & (kvel > 0)].mean()), 2),
                          peak_err_per_rep=peaks)
    return out


import kin_snapshot as K  # noqa: E402
stack = {}
for fps in [30.0, 11.6]:
    rng = np.random.default_rng(21)
    world, s, phase, windows = C.synth.session(fps=fps)
    T = world.shape[0]; ts = np.arange(T) / fps
    pts, conf, reproj, _ = C.triangulate(world, rng, sigma_px=3.0)
    true = C.synth.recenter(world)
    kt = K.knee_flexion(true, "l")
    variants = {
        "no_preik_filter": pts,
        "blend+clamp(prod)": C.run_blend_clamp(pts, conf, ts),
        "clamp_only": C.run_blend_clamp(pts, conf, ts, use_blend=False),
    }
    stack[f"fps_{fps}"] = {name: summarize(post_ik(p, conf, ts, phase), kt, windows, fps) for name, p in variants.items()}
res["b_post_ik_stack_knee_flexion_l_deg"] = stack

(C.HERE / "e5_results.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res["a_outlier_smear"], indent=1))
for fk, v in stack.items():
    print("==", fk)
    for name, st in v.items():
        for stage, m in st.items():
            print(f"  {name:20s} {stage:5s} rmse {m['rmse']:5.2f} fast|e| {m['fast_mean_abs']:5.2f} fastDescSigned {m['fast_desc_mean_signed']:6.2f} peaks {m['peak_err_per_rep']}")
