"""E4: zero-confidence seeding, long absence/return, IK trust band, NaN/inf poisoning, gaps without reset.
All runs use the PRODUCTION ConfidenceBlender and VelocityClamp classes.
"""
from __future__ import annotations

import json
import warnings

import numpy as np

import common as C
from biomechanics.kinematics.analytical_ik import AnalyticalIKSolver
from biomechanics.utils.confidence_blend import ConfidenceBlender
from biomechanics.utils.types import Skeleton3D
from biomechanics.utils.velocity_clamp import VelocityClamp

CK = C.CK
TOE_L = 17
FPS = 30.0
res: dict = {}
rng = np.random.default_rng(5)


def chain(pts, conf, ts, use_blend=True, use_clamp=True):
    b = ConfidenceBlender(); c = VelocityClamp(2.5, 30)
    out = np.empty_like(pts); oc = np.empty(conf.shape)
    for t in range(len(pts)):
        sk = Skeleton3D.from_numpy(pts[t], confidences=conf[t], timestamp=float(ts[t]))
        if use_blend:
            sk = b.blend(sk)
        if use_clamp:
            sk = c.clamp(sk)
        out[t] = sk.to_numpy(); oc[t] = [k.confidence for k in sk.keypoints]
    return out, oc


# ---------------------------------------------------------------- (a) first frame after reset: toe untriangulated -> (0,0,0) conf 0
n = 40
world = np.repeat(C.synth.pose(0.0, C.synth.body())[None], n, axis=0)
pts, conf, reproj, _ = C.triangulate(world, rng, sigma_px=3.0)
true = C.synth.recenter(world)
ts = np.arange(n) / FPS
pts[0, TOE_L] = 0.0; conf[0, TOE_L] = 0.0  # triangulator leaves untriangulated kpts at exact (0,0,0), not recentred
rows = {}
for name, ub, uc in [("raw", False, False), ("blend_only", True, False), ("clamp_only", False, True), ("blend+clamp(prod)", True, True)]:
    o, oc = chain(pts, conf, ts, ub, uc)
    e = 100 * np.linalg.norm(o[:, TOE_L] - true[:, TOE_L], axis=-1)
    first_ok = next((i for i in range(1, n) if np.all(e[i:i + 3] < 3.0)), None)
    rows[name] = dict(err_cm_frames_0_to_14=[round(float(x), 1) for x in e[:15]], frames_until_err_lt_3cm=first_ok,
                      out_conf_frames_1_3=[round(float(x), 3) for x in oc[1:4, TOE_L]])
res["a_toe_seeded_at_origin_after_reset"] = dict(true_toe_distance_from_hip_cm=round(float(100 * np.linalg.norm(true[0, TOE_L])), 1), variants=rows)

# ---------------------------------------------------------------- (b) ankle lost for 1 s during descent (conf<0.1: reproj>=15px), returns
world, s, phase, windows = C.synth.session(fps=FPS)
T = world.shape[0]; ts = np.arange(T) / FPS
pts, conf, reproj, _ = C.triangulate(world, rng, sigma_px=3.0)
true = C.synth.recenter(world)
st, b0, b1, en = windows[0]
lost = slice(st + 2, st + 2 + 30)
bad = pts.copy()
bad[lost, CK.LEFT_ANKLE] += rng.normal(0, 0.10, (30, 3))  # a bad triangulation (~10 cm) as E3 outliers
conf_b = conf.copy(); conf_b[lost, CK.LEFT_ANKLE] = 0.06    # base*0.1 when reproj >= 15 px
o_raw, _ = chain(bad, conf_b, ts, False, False)
o_bl, oc_bl = chain(bad, conf_b, ts, True, False)
o_pr, oc_pr = chain(bad, conf_b, ts, True, True)
ret = lost.stop
e = lambda o: 100 * np.linalg.norm(o[:, CK.LEFT_ANKLE] - true[:, CK.LEFT_ANKLE], axis=-1)
rs = lambda o: C.synth.recenter(o)  # noqa: E731
import kin_snapshot as K  # noqa: E402
res["b_ankle_lost_1s_during_descent"] = dict(
    during_loss_err_cm_mean=dict(raw_outlier=round(float(e(o_raw)[lost].mean()), 1), blend_hold=round(float(e(o_bl)[lost].mean()), 1)),
    during_loss_err_cm_max=dict(raw_outlier=round(float(e(o_raw)[lost].max()), 1), blend_hold=round(float(e(o_bl)[lost].max()), 1)),
    rep_signal_err_cm_max_during_loss=dict(raw=round(float(np.abs(K.rep_signal_cm(o_raw) - K.rep_signal_cm(true))[lost].max()), 1),
                                           blend=round(float(np.abs(K.rep_signal_cm(o_bl) - K.rep_signal_cm(true))[lost].max()), 1)),
    after_return_err_cm_frames_0_8=dict(raw=[round(float(x), 1) for x in e(o_raw)[ret:ret + 9]],
                                        blend=[round(float(x), 1) for x in e(o_bl)[ret:ret + 9]],
                                        blend_clamp=[round(float(x), 1) for x in e(o_pr)[ret:ret + 9]]),
    out_conf_after_return=[round(float(x), 3) for x in oc_bl[ret:ret + 3, CK.LEFT_ANKLE]],
)

# ---------------------------------------------------------------- (c) IK trust band: blender min 0.1 == IK min 0.1
ik = AnalyticalIKSolver()
b = ConfidenceBlender()
std = C.synth.recenter(C.synth.pose(0.0, C.synth.body()))
bot = C.synth.recenter(C.synth.pose(1.0, C.synth.body()))
b.blend(Skeleton3D.from_numpy(std, confidences=np.full(19, 0.5)))
band = []
for cval in [0.09, 0.10, 0.11, 0.12, 0.14, 0.2, 0.3]:
    bb = ConfidenceBlender(); bb.blend(Skeleton3D.from_numpy(std, confidences=np.full(19, 0.5)))
    confs = np.full(19, 0.5); confs[CK.LEFT_KNEE] = cval; confs[CK.LEFT_ANKLE] = cval
    out = bb.blend(Skeleton3D.from_numpy(bot, confidences=confs))
    a = ik.solve(out); a_true = ik.solve(Skeleton3D.from_numpy(bot, confidences=np.full(19, 0.5)))
    band.append(dict(conf=cval, weight=round(float(C.blend_weights(np.array(cval))), 4), ik_uses_point=bool(cval >= ik.min_confidence),
                     knee_flex_l_out=round(a.knee_flexion_l, 1), knee_flex_l_true=round(a_true.knee_flexion_l, 1)))
reproj_band = [round(float(15 * (1 - c / 0.6)), 1) for c in (0.1, 0.14)]
res["c_ik_trust_band"] = dict(rows=band, reproj_px_giving_conf_0p10_to_0p14_at_base_0p6=reproj_band)

# ---------------------------------------------------------------- (d) NaN / inf poisoning
ik_nan = []
for bad_val in [np.nan, np.inf]:
    bb = ConfidenceBlender(); vc = VelocityClamp(2.5, 30)
    seq = []
    for t in range(6):
        p = std.copy(); c = np.full(19, 0.5)
        if t == 1:
            p[CK.LEFT_KNEE] = bad_val; c[CK.LEFT_KNEE] = 0.06  # would be base*0.1 (reproj NaN -> not < 15 -> 0.1 scale)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o = vc.clamp(bb.blend(Skeleton3D.from_numpy(p, confidences=c, timestamp=t / 30)))
            a = ik.solve(o)
        seq.append(dict(t=t, in_knee=str(p[CK.LEFT_KNEE][0]), out_knee_x=str(round(o.keypoints[CK.LEFT_KNEE].x, 3)),
                        out_conf=round(o.keypoints[CK.LEFT_KNEE].confidence, 2), knee_flex=str(round(a.knee_flexion_l, 1))))
    ik_nan.append(dict(bad=str(bad_val), frames=seq))
# does triangulator's formula send non-finite reproj to the 0.1 branch?
rep = np.array([np.nan, np.inf, 3.0])
res["d_nonfinite"] = dict(sequences=ik_nan,
                          triangulator_scale_for_reproj_nan_inf_3=[float(x) for x in np.where(rep < 15.0, 1 - rep / 15.0, 0.1)])

# ---------------------------------------------------------------- (e) gap without reset (user leaves 10 s, returns mid-squat)
bb = ConfidenceBlender()
for t in range(30):
    bb.blend(Skeleton3D.from_numpy(std, confidences=np.full(19, 0.5), timestamp=t / 30))
half = C.synth.recenter(C.synth.pose(0.6, C.synth.body()))
errs = []
for t in range(5):
    o = bb.blend(Skeleton3D.from_numpy(half, confidences=np.full(19, 0.49), timestamp=10.0 + t / 30)).to_numpy()
    errs.append(dict(frame=t, ankle_err_cm=round(float(100 * np.linalg.norm(o[CK.LEFT_ANKLE] - half[CK.LEFT_ANKLE])), 1),
                     knee_flex_err_deg=round(float(K.knee_flexion(o, "l") - K.knee_flexion(half, "l")), 1)))
res["e_gap_10s_no_reset_return_mid_squat"] = errs

(C.HERE / "e4_results.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
