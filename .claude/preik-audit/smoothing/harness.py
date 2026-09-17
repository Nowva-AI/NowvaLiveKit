"""Pipeline-faithful replay of pipeline.process_frame from the pre-IK chain to the RuleEngine, plus metrics.

Order replicated (pipeline.py:620-700): preIK chain -> IK (vectorised copy, verified identical) ->
JointAngleFilter.update_phase(rep_counter.phase) -> filter_angles -> DerivativeTracker ->
bottom-frame buffer (max filtered avg knee flexion while rep_counter.in_rep) -> predict -> RuleEngine.evaluate
-> rep_counter.update(signal from preIK skeleton) -> evaluate_rep_complete on completed reps.
"""
from __future__ import annotations

import os
import sys

import numpy as np

os.environ["NOWVA_MULTI_CAMERA"] = "true"
sys.path.insert(0, "/Users/naiahoard/NowvaLiveKit/src")
sys.path.insert(0, "/Users/naiahoard/NowvaLiveKit/.claude/preik-audit/smoothing")

from biomechanics.config import load_pipeline_config  # noqa: E402
from biomechanics.faults.hip_position_counter import SignalRepCounter  # noqa: E402
from biomechanics.faults.rule_engine import RuleEngine  # noqa: E402
from biomechanics.profiles.squat import SquatProfile  # noqa: E402
from biomechanics.utils.derivatives import DerivativeTracker  # noqa: E402
from biomechanics.utils.filters import JointAngleFilter  # noqa: E402
from biomechanics.utils.predictive_state import PredictiveStateEstimator  # noqa: E402
from biomechanics.utils.types import CocoKeypoints as CK, JointAngles, Skeleton3D  # noqa: E402
from biomechanics.utils.velocity_clamp import VelocityClamp  # noqa: E402

import kin  # noqa: E402
import smoothers as sm  # noqa: E402
import synth  # noqa: E402

CONFIG = load_pipeline_config()
LEGS = [CK.LEFT_HIP, CK.RIGHT_HIP, CK.LEFT_KNEE, CK.RIGHT_KNEE, CK.LEFT_ANKLE, CK.RIGHT_ANKLE]
_SESSION_CACHE: dict = {}


def make_data(seed: int = 0, sigma_px: float = 4.0, noise: str = "tri", sigma_m: float = 0.01, reps=None,
              outlier_p: float = 0.0, kind: str = "smooth"):
    key = kind + str(reps)
    if key not in _SESSION_CACHE:
        _SESSION_CACHE[key] = synth.session(reps=reps) if kind == "smooth" else synth.session_hard()
    world, s, phase, windows = _SESSION_CACHE[key]
    rng = np.random.default_rng(seed)
    if noise == "tri":
        noisy, conf, reproj = synth.triangulated_noise(world, rng, sigma_px=sigma_px, outlier_p=outlier_p)
    elif noise == "none":
        noisy, conf, reproj = synth.recenter(world), np.full(world.shape[:2], 0.6), np.zeros(world.shape[:2])
    else:
        noisy, conf, reproj = synth.gaussian_noise(world, rng, sigma_m)
    ts = 1000.0 + np.arange(len(world)) / synth.FPS
    # oracle phase: 'bottom' = within 5% of that rep's depth (the pure pause label never exists for bounce reps)
    phase = list(phase)
    for (a, b0, b1, e) in windows:
        depth = s[a:e + 1].max()
        for i in range(a, e + 1):
            if s[i] >= 0.95 * depth:
                phase[i] = "bottom"
    return dict(truth=synth.recenter(world), noisy=noisy, conf=conf, ts=ts, phase=phase, windows=windows, s=s,
                reproj=reproj)


def summarize(dicts: list[dict]) -> dict:
    out = {}
    for k in dicts[0]:
        vals = [dd[k] for dd in dicts]
        if isinstance(vals[0], (int, float)):
            out[k] = round(float(np.mean(vals)), 3)
    return out


def _joint_angles(k: np.ndarray, t: float, i: int) -> JointAngles:
    a = kin.all_angles(k)
    return JointAngles(knee_flexion_l=float(a["knee_l"]), knee_flexion_r=float(a["knee_r"]),
                       hip_flexion_l=float(a["hip_l"]), hip_flexion_r=float(a["hip_r"]),
                       knee_valgus_l=float(a["valgus_l"]), knee_valgus_r=float(a["valgus_r"]),
                       foot_confidence_l=0.9, foot_confidence_r=0.9,
                       trunk_flexion=float(a["trunk"]), timestamp=t, frame_index=i)


def run_stack(d: dict, blend: bool = False, vclamp: bool = False, pos_filter=None, jaf: bool = True,
              predictive: bool = True, angle_filter=None, phase_source: str = "counter", phase_aware: bool | None = None):
    phase_aware = predictive if phase_aware is None else phase_aware
    noisy, conf, ts = d["noisy"], d["conf"], d["ts"]
    T = len(noisy)
    blender = sm.BlendVec(CONFIG.confidence_blend.min_confidence, CONFIG.confidence_blend.max_confidence)
    vc = VelocityClamp(max_velocity_m_per_s=CONFIG.velocity_clamp.max_velocity_m_per_s,
                       target_fps=CONFIG.pipeline.target_fps)
    if pos_filter is not None:
        pos_filter.reset()
    angle_f = JointAngleFilter(min_cutoff=1.0, beta=0.007)
    deriv = DerivativeTracker(smoothing_alpha=0.3)
    pred = PredictiveStateEstimator(CONFIG.predictive_state.horizon_seconds,
                                    CONFIG.predictive_state.max_extrapolation_deg)
    profile = SquatProfile()
    engine = RuleEngine(CONFIG, rules=profile.create_fault_rules(CONFIG))
    counter = SignalRepCounter(CONFIG.hip_counter)

    keys = ["knee_flexion_l", "knee_flexion_r", "hip_flexion_l", "hip_flexion_r", "knee_valgus_l", "knee_valgus_r"]
    skel_out = np.empty_like(noisy)
    ang_out = {k: np.empty(T) for k in keys}
    eval_out = {k: np.empty(T) for k in keys}
    phase_out, faults_out, reps_out = [], [], []
    bottom_max, bottom_idx, bottoms = 0.0, None, []
    was_in_rep = False
    for i in range(T):
        x = noisy[i]
        if blend:
            x = blender.step(x, ts[i], conf[i])
        if vclamp:
            sk = Skeleton3D.from_numpy(x, confidences=list(conf[i]), timestamp=ts[i], frame_index=i)
            x = vc.clamp(sk).to_numpy()
        if pos_filter is not None:
            side = (conf[i], d["reproj"][i]) if getattr(pos_filter, "wants_reproj", False) else conf[i]
            x = pos_filter.step(x, ts[i], side)
        skel_out[i] = x
        if phase_source == "oracle":
            cur_phase = d["phase"][max(i - 1, 0)]
            cur_in_rep = cur_phase != "idle"
        else:
            cur_phase, cur_in_rep = counter.phase, counter.in_rep
        raw = _joint_angles(x, ts[i], i)
        if phase_aware:  # pipeline only calls update_phase when preik is enabled (same flag as predictive)
            angle_f.update_phase(cur_phase)
        if angle_filter is not None:
            angles = angle_filter(raw)
        elif jaf:
            angles = angle_f.filter_angles(raw)
        else:
            angles = raw
        derivs = deriv.update(angles)
        if cur_in_rep:
            if angles.avg_knee_flexion > bottom_max:
                bottom_max, bottom_idx = angles.avg_knee_flexion, i
        eval_angles = pred.predict(angles, derivs) if predictive else angles
        faults = engine.evaluate(eval_angles, in_rep=cur_in_rep, rep_number=counter.rep_count + 1,
                                 derivatives=derivs, phase=cur_phase)
        signal = kin.rep_signal_cm(x)
        rep_data, _ = counter.update(signal_value=float(signal), timestamp=float(ts[i]), angles=angles, faults=faults)
        if rep_data is not None:
            faults = faults + engine.evaluate_rep_complete(rep_data.max_depth_angle, angles, rep_data.rep_number)
            reps_out.append((i, rep_data.max_depth_angle))
        if was_in_rep and not cur_in_rep:
            bottoms.append(bottom_idx)
            bottom_max, bottom_idx = 0.0, None
        was_in_rep = cur_in_rep
        for k in keys:
            ang_out[k][i] = getattr(angles, k)
            eval_out[k][i] = getattr(eval_angles, k)
        phase_out.append(cur_phase)
        faults_out.append([(str(getattr(f.fault_type, "value", f.fault_type)), str(getattr(f.severity, "value", f.severity)), i) for f in faults])
    return dict(skel=skel_out, angles=ang_out, eval=eval_out, phase=phase_out, faults=faults_out,
                reps=reps_out, bottoms=bottoms)


def true_angles(d: dict) -> dict:
    a = kin.all_angles(d["truth"])
    return {"knee_flexion_l": a["knee_l"], "knee_flexion_r": a["knee_r"], "hip_flexion_l": a["hip_l"],
            "hip_flexion_r": a["hip_r"], "knee_valgus_l": a["valgus_l"], "knee_valgus_r": a["valgus_r"]}


def lag_ms(est: np.ndarray, truth: np.ndarray, mask: np.ndarray, fps: float = 30.0) -> float:
    t = np.arange(len(truth)) / fps
    best, best_lag = 1e9, 0.0
    for lag in np.arange(-0.05, 0.4, 0.0033):
        shifted = np.interp(t - lag, t, truth)
        e = np.sqrt(np.mean((est[mask] - shifted[mask]) ** 2))
        if e < best:
            best, best_lag = e, lag
    return best_lag * 1000.0


def angle_metrics(est: dict, d: dict, prefix: str = "") -> dict:
    tru = true_angles(d)
    phase = np.array(d["phase"])
    idle = phase == "idle"
    moving = ~idle
    windows = d["windows"]
    knee_e = (est["knee_flexion_l"] + est["knee_flexion_r"]) / 2
    knee_t = (tru["knee_flexion_l"] + tru["knee_flexion_r"]) / 2
    out = {}
    out["knee_rms_deg"] = float(np.sqrt(np.mean((knee_e - knee_t) ** 2)))
    out["knee_rms_moving_deg"] = float(np.sqrt(np.mean((knee_e[moving] - knee_t[moving]) ** 2)))
    settle = idle.copy()
    settle[:30] = False
    out["knee_std_standing_deg"] = float(np.std(est["knee_flexion_l"][settle] - tru["knee_flexion_l"][settle]))
    out["knee_lag_ms"] = lag_ms(knee_e, knee_t, moving)
    depth_err, mid_err, valg_err, valg_true = [], [], [], []
    for (a, b0, b1, e) in windows:
        seg = slice(a, e + 1)
        depth_err.append(knee_e[a:e + 20].max() - knee_t[seg].max())
        vel = np.gradient(knee_t[a:b0 + 1])
        mid = a + int(np.argmax(vel))
        mid_err.append(knee_e[mid] - knee_t[mid])
        for side in ("l", "r"):
            tv = tru[f"knee_valgus_{side}"][seg].max()
            if tv > 5:
                valg_err.append(est[f"knee_valgus_{side}"][a:e + 20].max() - tv)
                valg_true.append(tv)
    out["depth_err_per_rep_deg"] = [round(float(v), 1) for v in depth_err]
    out["depth_err_mean_deg"] = float(np.mean(depth_err))
    out["depth_err_worst_deg"] = float(np.min(depth_err))
    out["mid_descent_err_deg"] = float(np.mean(mid_err))
    out["valgus_peak_err_deg"] = [round(float(v), 1) for v in valg_err]
    out["valgus_peak_err_mean_deg"] = float(np.mean(valg_err))
    return {prefix + k: v for k, v in out.items()}


def position_metrics(est: np.ndarray, d: dict) -> dict:
    truth = d["truth"]
    phase = np.array(d["phase"])
    idle = phase == "idle"
    err = np.linalg.norm(est[:, LEGS] - truth[:, LEGS], axis=-1) * 100
    out = {"pos_rms_cm": float(np.sqrt(np.mean(err ** 2))),
           "pos_rms_moving_cm": float(np.sqrt(np.mean(err[~idle] ** 2)))}
    settle = idle.copy(); settle[:30] = False
    jit = np.linalg.norm(np.diff(est[:, LEGS], axis=0), axis=-1)[settle[1:]] * 1000
    out["standing_jitter_mm_per_frame"] = float(np.sqrt(np.mean(jit ** 2)))
    sig_e, sig_t = kin.rep_signal_cm(est), kin.rep_signal_cm(truth)
    dep, knee_y_bottom = [], []
    for (a, b0, b1, e) in d["windows"]:
        dep.append(sig_e[a:e + 20].max() - sig_t[a:e + 1].max())
        mid_b = (b0 + b1) // 2
        knee_y_bottom.append((est[mid_b, CK.LEFT_KNEE] - truth[mid_b, CK.LEFT_KNEE]) * 100)
    out["rep_signal_peak_err_cm"] = [round(float(v), 2) for v in dep]
    out["rep_signal_peak_err_mean_cm"] = float(np.mean(dep))
    out["knee_kpt_err_at_true_bottom_cm"] = np.round(np.mean(np.abs(knee_y_bottom), axis=0), 2).tolist()
    thigh = kin.bone_len(est, CK.LEFT_HIP, CK.LEFT_KNEE) * 100
    shank = kin.bone_len(est, CK.LEFT_KNEE, CK.LEFT_ANKLE) * 100
    thigh_t = float(kin.bone_len(truth, CK.LEFT_HIP, CK.LEFT_KNEE)[0] * 100)
    shank_t = float(kin.bone_len(truth, CK.LEFT_KNEE, CK.LEFT_ANKLE)[0] * 100)
    out["thigh_bias_moving_cm"] = float(np.mean(thigh[~idle]) - thigh_t)
    out["thigh_min_minus_true_cm"] = float(np.min(thigh[~idle]) - thigh_t)
    out["shank_bias_moving_cm"] = float(np.mean(shank[~idle]) - shank_t)
    out["thigh_std_cm"] = float(np.std(thigh))
    return out


def fault_summary(res: dict, d: dict) -> dict:
    tru = true_angles(d)
    windows = d["windows"]
    rep_of = np.full(len(d["phase"]), -1)
    for r, (a, b0, b1, e) in enumerate(windows):
        rep_of[a:e + 45] = r
    counts: dict = {}
    for fl in res["faults"]:
        for ftype, sev, i in fl:
            r = int(rep_of[i])
            counts.setdefault(ftype, {}).setdefault(r, []).append(sev)
    bottoms = []
    knee_t = (tru["knee_flexion_l"] + tru["knee_flexion_r"]) / 2
    for bi in res["bottoms"]:
        if bi is None:
            continue
        r = int(rep_of[bi])
        if r < 0:
            continue
        a, b0, b1, e = windows[r]
        k_est = kin.all_angles(res["skel"][bi])
        bottoms.append(dict(rep=r, frame_offset_from_true_bottom=int(bi - (b0 + b1) // 2),
                            true_knee_at_frame_minus_true_max=round(float(knee_t[bi] - knee_t[a:e + 1].max()), 1),
                            stored_skel_knee_minus_true_max=round(float((k_est["knee_l"] + k_est["knee_r"]) / 2
                                                                        - knee_t[a:e + 1].max()), 1)))
    return dict(faults=counts, n_reps_counted=len(res["reps"]),
                rep_max_depth=[round(v, 1) for _, v in res["reps"]], bottoms=bottoms)


def fast_eval(filtered: np.ndarray, d: dict, lag_frames: int = 0) -> dict:
    """Same as exp4_grid.fast_eval: vectorised angles from filtered positions, lag-realigned, objective J."""
    est = filtered if not lag_frames else np.concatenate([filtered[lag_frames:],
                                                          np.repeat(filtered[-1:], lag_frames, axis=0)])
    a = kin.all_angles(est)
    ang = {"knee_flexion_l": a["knee_l"], "knee_flexion_r": a["knee_r"], "hip_flexion_l": a["hip_l"],
           "hip_flexion_r": a["hip_r"], "knee_valgus_l": a["valgus_l"], "knee_valgus_r": a["valgus_r"]}
    m = angle_metrics(ang, d)
    m.update(position_metrics(est, d))
    m["depth_abs_mean_deg"] = float(np.mean(np.abs(m["depth_err_per_rep_deg"])))
    m["valgus_abs_mean_deg"] = float(np.mean(np.abs(m["valgus_peak_err_deg"])))
    m["J"] = m["knee_rms_moving_deg"] + m["depth_abs_mean_deg"] + m["valgus_abs_mean_deg"] + m["knee_std_standing_deg"]
    m["latency_ms"] = lag_frames * 1000 / 30
    return m
