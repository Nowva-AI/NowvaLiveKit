"""E2: lag / depth undershoot / valgus attenuation / bone-length corruption of the production ConfidenceBlender
on a realistic triangulated squat, vs raw and vs alternatives tuned to EQUAL static noise reduction.
"""
from __future__ import annotations

import json

import numpy as np

import common as C
import filters_alt as F
import kin_snapshot as K

CK = C.CK
LEG = [CK.LEFT_KNEE, CK.RIGHT_KNEE, CK.LEFT_ANKLE, CK.RIGHT_ANKLE, 17, 18]
SIGMA_PX = 3.0


def static_ratio(filter_fn, fps: float, seed: int = 7) -> float:
    rng = np.random.default_rng(seed)
    # NOTE synth.session(reps=[]) falls back to DEFAULT_REPS (falsy list) -> build a standing-only world explicitly
    n = int(20.0 * fps)
    world = np.repeat(C.synth.pose(0.0, C.synth.body())[None], n, axis=0)
    ts = np.arange(world.shape[0]) / fps
    pts, conf, rep, _ = C.triangulate(world, rng, sigma_px=SIGMA_PX)
    true = C.synth.recenter(world)
    out = filter_fn(pts, conf, ts)
    k = slice(int(fps), None)
    e_raw = np.sqrt(np.mean(np.sum((pts[k][:, LEG] - true[k][:, LEG]) ** 2, -1)))
    e_f = np.sqrt(np.mean(np.sum((out[k][:, LEG] - true[k][:, LEG]) ** 2, -1)))
    return float(e_f / e_raw)


def xcorr_lag_ms(a: np.ndarray, b: np.ndarray, fps: float, max_lag: int = 8) -> float:
    a = a - a.mean(); b = b - b.mean()
    lags = np.arange(0, max_lag + 1)
    c = np.array([np.dot(a[: len(a) - L], b[L:]) for L in lags])
    i = int(np.argmax(c))
    if 0 < i < len(c) - 1:
        d = 0.5 * (c[i - 1] - c[i + 1]) / (c[i - 1] - 2 * c[i] + c[i + 1])
    else:
        d = 0.0
    return float((i + d) * 1000.0 / fps)


def metrics(out: np.ndarray, true: np.ndarray, s: np.ndarray, phase: list, windows: list, fps: float) -> dict:
    m = {}
    idle = np.array([p == "idle" for p in phase])
    idle[: int(fps)] = False
    m["static_leg_err_rms_cm"] = round(100 * float(np.sqrt(np.mean(np.sum((out[idle][:, LEG] - true[idle][:, LEG]) ** 2, -1)))), 2)
    vel = np.gradient(true, axis=0) * fps
    speed = np.linalg.norm(vel, axis=-1)
    lag = {}
    for name, k in [("knee_l", CK.LEFT_KNEE), ("ankle_l", CK.LEFT_ANKLE), ("toe_l", 17)]:
        moving = speed[:, k] > 0.5
        u = vel[moving, k] / speed[moving, k][:, None]
        along = np.sum((true[moving, k] - out[moving, k]) * u, -1)
        lag[name] = round(100 * float(np.mean(along)), 2)
    m["lag_along_velocity_cm_when_speed_gt_0p5"] = lag
    kt = K.knee_flexion(true, "l"); ko = K.knee_flexion(out, "l")
    m["knee_flex_rmse_deg_all"] = round(float(np.sqrt(np.mean((ko - kt) ** 2))), 2)
    kvel = np.gradient(kt) * fps
    fast = np.abs(kvel) > 100.0
    m["knee_flex_mean_abs_err_deg_when_gt_100dps"] = round(float(np.mean(np.abs(ko - kt)[fast])), 2) if fast.any() else None
    m["knee_flex_xcorr_lag_ms"] = round(xcorr_lag_ms(kt, ko, fps), 1)
    under, depth_under, valg = [], [], []
    rs_t = K.rep_signal_cm(true); rs_o = K.rep_signal_cm(out)
    for r, (st, b0, b1, en) in enumerate(windows):
        seg = slice(st, en + 1)
        under.append(round(float(ko[seg].max() - kt[seg].max()), 2))
        depth_under.append(round(float(rs_o[seg].max() - rs_t[seg].max()), 2))
    m["peak_knee_flex_err_per_rep_deg"] = under
    m["depth_hip_minus_ankle_err_per_rep_cm"] = depth_under
    for r, side in [(1, "l"), (3, "r"), (4, "l")]:
        st, b0, b1, en = windows[r]
        seg = slice(st, en + 1)
        gt = K.gs_valgus(true[seg], side); go = K.gs_valgus(out[seg], side)
        valg.append(dict(rep=r, side=side, true_peak=round(float(np.max(np.abs(gt))), 2), out_peak=round(float(np.max(np.abs(go))), 2)))
    m["valgus_peak"] = valg
    moving_any = np.array([p in ("descending", "ascending") for p in phase])
    bl = {}
    for name, a, b in [("femur_l", CK.LEFT_HIP, CK.LEFT_KNEE), ("tibia_l", CK.LEFT_KNEE, CK.LEFT_ANKLE),
                       ("foot_l", CK.LEFT_ANKLE, 17), ("upperarm_l", CK.LEFT_SHOULDER, CK.LEFT_ELBOW)]:
        L_t = K.bone_len(true, a, b); L_o = K.bone_len(out, a, b)
        d = 100 * (L_o - L_t)[moving_any]
        bl[name] = dict(mean_cm=round(float(d.mean()), 2), p5_cm=round(float(np.percentile(d, 5)), 2),
                        p95_cm=round(float(np.percentile(d, 95)), 2))
    m["bone_len_err_during_motion"] = bl
    return m


res: dict = {}
for fps in [30.0, 15.0, 11.6]:
    rng = np.random.default_rng(3)
    world, s, phase, windows = C.synth.session(fps=fps)
    T = world.shape[0]
    ts = np.arange(T) / fps
    true = C.synth.recenter(world)
    pts, conf, reproj, _ = C.triangulate(world, rng, sigma_px=SIGMA_PX)
    w = C.blend_weights(conf)
    blend_fn = lambda p, c, t: C.run_blender(p, c, t)
    r_bl = static_ratio(blend_fn, fps)
    w_med = float(np.median(w[:, LEG]))
    # tune alternatives to the blender's static noise ratio
    def tune(make_fn, grid):
        best = None
        for g in grid:
            rr = static_ratio(make_fn(g), fps)
            if best is None or abs(rr - r_bl) < abs(best[1] - r_bl):
                best = (g, rr)
        return best
    ema_g = tune(lambda a: (lambda p, c, t: F.ema_fixed(p, a)), np.linspace(0.3, 0.7, 21))
    ab_g = tune(lambda a: (lambda p, c, t: F.alpha_beta(p, t, a)), np.linspace(0.3, 0.9, 31))
    oe_best = None
    for beta in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]:
        g, rr = tune(lambda mc, beta=beta: (lambda p, c, t: F.one_euro(p, t, mc, beta)), [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0])
        if abs(rr - r_bl) < 0.06:
            lagtest = F.one_euro(true, ts, g, beta)
            vel = np.gradient(true, axis=0) * fps
            sp = np.linalg.norm(vel[:, CK.LEFT_ANKLE], axis=-1)
            mv = sp > 0.5
            lag_cm = float(np.mean(np.sum((true[mv, CK.LEFT_ANKLE] - lagtest[mv, CK.LEFT_ANKLE]) * vel[mv, CK.LEFT_ANKLE] / sp[mv][:, None], -1)))
            if oe_best is None or lag_cm < oe_best[3]:
                oe_best = (g, beta, rr, lag_cm)
    tuned = dict(blender_static_ratio=round(r_bl, 3), median_leg_weight=round(w_med, 3),
                 ema_alpha=round(float(ema_g[0]), 3), ema_ratio=round(ema_g[1], 3),
                 alpha_beta_alpha=round(float(ab_g[0]), 3), alpha_beta_ratio=round(ab_g[1], 3),
                 one_euro_min_cutoff=oe_best[0], one_euro_beta=oe_best[1], one_euro_ratio=round(oe_best[2], 3))
    variants = {
        "raw": lambda p, c: p,
        "blender(prod)": lambda p, c: C.run_blender(p, c, ts),
        "ema_fixed_equal_noise": lambda p, c: F.ema_fixed(p, ema_g[0]),
        "alpha_beta_CV_equal_noise": lambda p, c: F.alpha_beta(p, ts, ab_g[0]),
        "one_euro_equal_noise": lambda p, c: F.one_euro(p, ts, oe_best[0], oe_best[1]),
    }
    per = {}
    for name, fn in variants.items():
        per[name] = dict(noiseless_real_conf=metrics(fn(true, conf), true, s, phase, windows, fps),
                         noisy_sigma3px=metrics(fn(pts, conf), true, s, phase, windows, fps))
    res[f"fps_{fps}"] = dict(tuning=tuned, conf_leg_p5_p50_p95=[round(float(x), 3) for x in np.percentile(conf[:, LEG], [5, 50, 95])],
                             reproj_leg_p5_p50_p95=[round(float(x), 2) for x in np.percentile(reproj[:, LEG], [5, 50, 95])],
                             variants=per)
    print("done fps", fps, tuned)

(C.HERE / "e2_results.json").write_text(json.dumps(res, indent=1))

# compact print
for fk, v in res.items():
    print("=====", fk, v["tuning"], "conf", v["conf_leg_p5_p50_p95"], "reproj", v["reproj_leg_p5_p50_p95"])
    for name, mm in v["variants"].items():
        for inp in ["noiseless_real_conf", "noisy_sigma3px"]:
            m = mm[inp]
            print(f"  {name:28s} {inp:20s} static {m['static_leg_err_rms_cm']:5.2f}cm lag(kn/an/toe) {m['lag_along_velocity_cm_when_speed_gt_0p5']} "
                  f"kneeRMSE {m['knee_flex_rmse_deg_all']:5.2f} fastErr {m['knee_flex_mean_abs_err_deg_when_gt_100dps']} xlag {m['knee_flex_xcorr_lag_ms']}ms")
            print(f"      peakKnee {m['peak_knee_flex_err_per_rep_deg']} depth {m['depth_hip_minus_ankle_err_per_rep_cm']}")
            print(f"      valgus {[(x['true_peak'], x['out_peak']) for x in m['valgus_peak']]} bones {m['bone_len_err_during_motion']}")
