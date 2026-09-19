"""Metrics for one chain run against the analytic ground truth.

Angle truth = the same AnalyticalIKSolver + TriangulatedValgusEstimator run on the noiseless hip-centred GT
skeleton (keypoints are exact joint centres in the generator, so for knee/hip/trunk flexion this IS the
analytic angle; see REPORT.md). Positions are compared in the triangulated (calibrated) frame against the
hip-centred truth in the true frame, so tpose-mode numbers include real calibration error.
"""

from __future__ import annotations

import math
import warnings

import numpy as np

from .runner import PreparedRun, RunRecords, _truth_angles, hip_centre

KPT_NAMES = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
             "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle", "l_toe", "r_toe"]
LOWER = list(range(11, 19))
BONES = [(5, 11), (6, 12), (5, 6), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (5, 7), (7, 9), (6, 8),
         (8, 10), (15, 17), (16, 18)]
LEG_BONES = [(11, 13), (13, 15), (12, 14), (14, 16)]
LAG_TAUS_S = np.arange(-0.06, 0.402, 0.002)
BOTTOM_SEARCH_EXTRA_S = 0.35
MIN_VALGUS_AMP_DEG = 3.0
MIN_HEEL_AMP_M = 0.01
MIN_SHIFT_AMP_M = 0.02
MIN_LIST_AMP_DEG = 3.0
MIN_STANCE_CHANGE_M = 0.03
MIN_FOOT_STATE_RISE_CM = 1.0
TRUE_BOTTOM_HALF_WINDOW_S = 0.15
M_TO_CM = 100.0
REP_KEYS = [
    "depth_err_final_deg", "depth_err_ik_deg", "depth_err_final_l_deg", "depth_err_final_r_deg", "bottom_sel_delay_ms",
    "bottom_sel_delay_wall_ms", "foot_state_heel_rise_amp_ratio_tb", "foot_state_false_heel_rise_tb_cm",
    "foot_state_heel_rise_peak_ratio_tb", "heel_rise_peak_ratio_tb", "foot_state_heel_rise_peak_cm_tb",
    "true_ankle_rise_peak_cm_tb",
    "valgus_sel_err_deg", "hip_flex_sel_err_deg", "trunk_flex_sel_err_deg", "knee_asym_sel_err_deg",
    "ik_knee_zero_frac_bottom", "knee_flex_tb_err_deg", "valgus_tb_err_deg", "hip_flex_tb_err_deg", "trunk_flex_tb_err_deg", "knee_asym_tb_err_deg",
] + [f"{name}_{kind}" for name, unit in (("valgus", "deg"), ("heel_rise", "mm"), ("hip_shift", "mm"),
                                          ("pelvis_list", "deg"))
     for kind in ("amp_ratio_tb", "amp_ratio_sel", f"amp_err_tb_{unit}", f"amp_err_sel_{unit}")]


def _knee_flexion(points: np.ndarray, hip: int, knee: int, ankle: int) -> np.ndarray:
    thigh = points[:, hip] - points[:, knee]
    shank = points[:, ankle] - points[:, knee]
    cos = np.sum(thigh * shank, 1) / np.maximum(np.linalg.norm(thigh, axis=1) * np.linalg.norm(shank, axis=1), 1e-12)
    return 180.0 - np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _heel_height(points: np.ndarray) -> np.ndarray:
    # ankle height above big toe (Y down), mean of both feet
    return 0.5 * ((points[:, 17, 1] - points[:, 15, 1]) + (points[:, 18, 1] - points[:, 16, 1]))


def _hip_shift(points: np.ndarray) -> np.ndarray:
    across = points[:, 15] - points[:, 16]
    across[:, 1] = 0.0
    across /= np.maximum(np.linalg.norm(across, axis=1, keepdims=True), 1e-9)
    ankle_mid = 0.5 * (points[:, 15] + points[:, 16])
    hip_mid = 0.5 * (points[:, 11] + points[:, 12])
    return np.sum((hip_mid - ankle_mid) * across, 1)


def _stance_width(points: np.ndarray) -> np.ndarray:
    return np.linalg.norm((points[:, 15] - points[:, 16])[:, [0, 2]], axis=1)


def _lag_ms(times: np.ndarray, est: np.ndarray, dense_t: np.ndarray, dense_sig: np.ndarray) -> float:
    ok = np.isfinite(est)
    times, est = times[ok], est[ok]
    if len(est) < 20 or est.std() < 1e-9:
        return float("nan")
    est_c = est - est.mean()
    scores = np.empty(len(LAG_TAUS_S))
    for k, tau in enumerate(LAG_TAUS_S):
        g = np.interp(times - tau, dense_t, dense_sig)
        g = g - g.mean()
        scores[k] = np.sum(est_c * g) / max(math.sqrt(np.sum(est_c ** 2) * np.sum(g ** 2)), 1e-12)
    best = int(np.argmax(scores))
    tau = LAG_TAUS_S[best]
    if 0 < best < len(scores) - 1:
        denom = scores[best - 1] - 2 * scores[best] + scores[best + 1]
        if abs(denom) > 1e-12:
            tau += 0.5 * (scores[best - 1] - scores[best + 1]) / denom * (LAG_TAUS_S[1] - LAG_TAUS_S[0])
    return float(tau * 1000.0)


def _nanmean(values: list[float]) -> float:
    arr = np.array([v for v in values if v is not None], dtype=np.float64)
    return float(np.nanmean(arr)) if arr.size and np.isfinite(arr).any() else float("nan")


def compute_metrics(prepared: PreparedRun, rec: RunRecords) -> dict:
    with warnings.catch_warnings():
        # runs where the readiness gate never opens legitimately produce all-NaN metrics
        warnings.simplefilter("ignore", RuntimeWarning)
        return _compute_metrics(prepared, rec)


def _compute_metrics(prepared: PreparedRun, rec: RunRecords) -> dict:
    sc = prepared.scenario
    times = prepared.truth_time_s
    processed = rec.status == 2
    still = sc.still_mask(times)
    moving = sc.squat_phase(times) > 0.02
    truth = prepared.truth_hc[:, :19]
    out = rec.out_xyz
    m: dict[str, float] = {}

    # ---- positions ----
    valid = processed[:, None] & (rec.out_conf > 0)
    err = np.where(valid, np.linalg.norm(out - truth, axis=2), np.nan)
    m["mpjpe_mm"] = float(np.nanmean(err)) * 1000
    m["mpjpe_lower_mm"] = float(np.nanmean(err[:, LOWER])) * 1000
    m["p95_err_mm"] = float(np.nanpercentile(err, 95)) * 1000 if np.isfinite(err).any() else float("nan")
    m["mpjpe_moving_mm"] = float(np.nanmean(err[moving])) * 1000 if (moving & processed).any() else float("nan")
    m["mpjpe_still_mm"] = float(np.nanmean(err[still])) * 1000 if (still & processed).any() else float("nan")
    m["missing_kpt_frac"] = float(np.mean(rec.out_conf[processed] <= 0)) if processed.any() else float("nan")
    per_kpt = {name: (round(float(np.nanmean(err[:, k])) * 1000, 2) if np.isfinite(err[:, k]).any() else None)
               for k, name in enumerate(KPT_NAMES)}

    # ---- bones ----
    maes, stds, leg_stds = [], [], []
    for a, b in BONES:
        ok = processed & (rec.out_conf[:, a] > 0) & (rec.out_conf[:, b] > 0)
        if ok.sum() < 3:
            continue
        est_len = np.linalg.norm(out[ok, a] - out[ok, b], axis=1)
        true_len = np.linalg.norm(truth[ok, a] - truth[ok, b], axis=1)
        maes.append(np.mean(np.abs(est_len - true_len)))
        stds.append(np.std(est_len))
        if (a, b) in LEG_BONES:
            leg_stds.append(np.std(est_len))
    m["bone_len_mae_mm"] = float(np.mean(maes)) * 1000 if maes else float("nan")
    m["bone_len_std_mm"] = float(np.mean(stds)) * 1000 if stds else float("nan")
    m["leg_bone_len_std_mm"] = float(np.mean(leg_stds)) * 1000 if leg_stds else float("nan")

    # ---- lag ----
    dense_t, dense = prepared.dense_t_s, prepared.dense_hc
    dense_knee_l = _knee_flexion(dense, 11, 13, 15)
    dense_knee_r = _knee_flexion(dense, 12, 14, 16)
    dense_knee = 0.5 * (dense_knee_l + dense_knee_r)
    region = processed & (times >= sc.reps[0].start_s - 0.3) & (times <= sc.reps[-1].end_s + 0.5)
    t_region = times[region]
    m["lag_knee_y_ms"] = _lag_ms(t_region, 0.5 * (out[region, 13, 1] + out[region, 14, 1]), dense_t,
                                 0.5 * (dense[:, 13, 1] + dense[:, 14, 1]))
    m["lag_ankle_y_ms"] = _lag_ms(t_region, 0.5 * (out[region, 15, 1] + out[region, 16, 1]), dense_t,
                                  0.5 * (dense[:, 15, 1] + dense[:, 16, 1]))
    ik_knee = 0.5 * (rec.ik["knee_flexion_l"] + rec.ik["knee_flexion_r"])
    final_knee = 0.5 * (rec.final["knee_flexion_l"] + rec.final["knee_flexion_r"])
    m["lag_knee_flex_ik_ms"] = _lag_ms(t_region, ik_knee[region], dense_t, dense_knee)
    m["lag_knee_flex_final_ms"] = _lag_ms(t_region, final_knee[region], dense_t, dense_knee)

    # ---- whole-sequence angle errors ----
    ta = prepared.truth_angles
    truth_knee = 0.5 * (ta["knee_flexion_l"] + ta["knee_flexion_r"])
    for label, series in (("ik", rec.ik), ("final", rec.final)):
        knee = 0.5 * (series["knee_flexion_l"] + series["knee_flexion_r"])
        m[f"knee_flex_{label}_mae_deg"] = float(np.nanmean(np.abs(knee - truth_knee)[processed]))
        m[f"knee_flex_{label}_moving_mae_deg"] = float(np.nanmean(np.abs(knee - truth_knee)[processed & moving]))
    m["hip_flex_final_mae_deg"] = float(np.nanmean(np.abs(0.5 * (rec.final["hip_flexion_l"] + rec.final[
        "hip_flexion_r"]) - 0.5 * (ta["hip_flexion_l"] + ta["hip_flexion_r"]))[processed]))
    m["trunk_flex_final_mae_deg"] = float(np.nanmean(np.abs(rec.final["trunk_flexion"] - ta["trunk_flexion"])[
        processed]))
    m["valgus_final_mae_deg"] = float(np.nanmean(np.abs(0.5 * (rec.final["knee_valgus_l"] + rec.final[
        "knee_valgus_r"]) - 0.5 * (ta["knee_valgus_l"] + ta["knee_valgus_r"]))[processed]))

    # ---- per-rep bottom metrics ----
    # "sel" = at the frame the pipeline would pick as the bottom (max of JointAngleFilter avg knee flexion, as the
    #         bottom-frame buffer does) compared with the truth at the TRUE bottom: what diagnosis receives.
    # "tb"  = median over processed frames within +-150 ms of the true bottom, IK angles / chain positions vs truth
    #         at those same instants: isolates the pre-IK chain (no angle filter, no bottom-selection lag).
    reps_used = []
    for rep in sc.reps:
        window = processed & (times >= rep.start_s) & (times <= rep.end_s + BOTTOM_SEARCH_EXTRA_S)
        if window.sum() < 3:
            continue
        dense_win = (dense_t >= rep.start_s) & (dense_t <= rep.end_s)
        j_bottom = np.nonzero(dense_win)[0][int(np.argmax(dense_knee[dense_win]))]
        t_bottom = float(dense_t[j_bottom])
        near = processed & (np.abs(times - t_bottom) <= TRUE_BOTTOM_HALF_WINDOW_S)
        baseline = processed & still & (times >= rep.start_s - 1.2) & (times <= rep.start_s - 0.05)
        reps_used.append((rep, window, t_bottom, float(dense_knee[j_bottom]), float(dense_knee_l[dense_win].max()),
                          float(dense_knee_r[dense_win].max()), near, baseline))
    if reps_used:
        bottom_hc = hip_centre(sc.pose(np.array([r[2] for r in reps_used])))
        bottom_truth = _truth_angles(bottom_hc)
    rows: dict[str, list[float]] = {}

    def add(key: str, value: float) -> None:
        rows.setdefault(key, []).append(float(value))

    pos = np.nan_to_num(out)
    ik_valgus = 0.5 * (rec.ik["knee_valgus_l"] + rec.ik["knee_valgus_r"])
    final_valgus = 0.5 * (rec.final["knee_valgus_l"] + rec.final["knee_valgus_r"])
    true_valgus_series = 0.5 * (ta["knee_valgus_l"] + ta["knee_valgus_r"])
    ik_hip = 0.5 * (rec.ik["hip_flexion_l"] + rec.ik["hip_flexion_r"])
    final_hip = 0.5 * (rec.final["hip_flexion_l"] + rec.final["hip_flexion_r"])
    true_hip_series = 0.5 * (ta["hip_flexion_l"] + ta["hip_flexion_r"])
    heel_est, heel_true = _heel_height(pos), _heel_height(truth)
    knee_l, knee_r = rec.ik["knee_flexion_l"], rec.ik["knee_flexion_r"]
    knee_valid = np.isfinite(knee_l) & np.isfinite(knee_r) & (knee_l != 0.0) & (knee_r != 0.0)
    # FootState heel rise (mean of both feet, NaN-aware) vs the true world ankle rise (Y down: rising = smaller y)
    foot_rise_cm = np.nanmean(rec.foot_heel_rise_cm, axis=1) if np.isfinite(rec.foot_heel_rise_cm).any() else None
    true_ankle_y = 0.5 * (prepared.truth_world[:, 15, 1] + prepared.truth_world[:, 16, 1])
    shift_est, shift_true = _hip_shift(pos), _hip_shift(truth)
    for r_idx, (rep, window, t_bottom, true_depth, true_l, true_r, near, baseline) in enumerate(reps_used):
        idx = np.nonzero(window)[0]
        has_bottom = bool(np.isfinite(final_knee[idx]).any())
        i_star = idx[int(np.nanargmax(final_knee[idx]))] if has_bottom else None
        bt = {k: float(v[r_idx]) for k, v in bottom_truth.items()}
        bottom_pts = bottom_hc[r_idx:r_idx + 1, :19]
        true_bottom_valgus = 0.5 * (bt["knee_valgus_l"] + bt["knee_valgus_r"])
        if has_bottom:
            add("depth_err_final_deg", final_knee[i_star] - true_depth)
            add("depth_err_ik_deg", float(np.nanmax(ik_knee[idx])) - true_depth)
            add("depth_err_final_l_deg", float(np.nanmax(rec.final["knee_flexion_l"][idx])) - true_l)
            add("depth_err_final_r_deg", float(np.nanmax(rec.final["knee_flexion_r"][idx])) - true_r)
            add("bottom_sel_delay_ms", (times[i_star] - t_bottom) * 1000)
            add("bottom_sel_delay_wall_ms", (times[i_star] - t_bottom) * 1000 + rec.latency_ms[i_star])
            add("valgus_sel_err_deg", final_valgus[i_star] - true_bottom_valgus)
            add("hip_flex_sel_err_deg", final_hip[i_star] - 0.5 * (bt["hip_flexion_l"] + bt["hip_flexion_r"]))
            add("trunk_flex_sel_err_deg", rec.final["trunk_flexion"][i_star] - bt["trunk_flexion"])
            add("knee_asym_sel_err_deg", (rec.final["knee_flexion_l"][i_star] - rec.final["knee_flexion_r"][i_star])
                - (bt["knee_flexion_l"] - bt["knee_flexion_r"]))
        n_all = np.nonzero(near)[0]
        if len(n_all):
            add("ik_knee_zero_frac_bottom", float(np.mean(~knee_valid[n_all])))
        # IK/valgus return exactly 0.0 when a keypoint has confidence < 0.1; "tb" accuracy uses valid frames only
        n_idx = n_all[knee_valid[n_all]]
        if len(n_idx):
            add("knee_flex_tb_err_deg", float(np.nanmedian(ik_knee[n_idx] - truth_knee[n_idx])))
            add("valgus_tb_err_deg", float(np.nanmedian(ik_valgus[n_idx] - true_valgus_series[n_idx])))
            add("hip_flex_tb_err_deg", float(np.nanmedian(ik_hip[n_idx] - true_hip_series[n_idx])))
            add("trunk_flex_tb_err_deg", float(np.nanmedian(rec.ik["trunk_flexion"][n_idx] - ta["trunk_flexion"][n_idx])))
            add("knee_asym_tb_err_deg", float(np.nanmedian(
                (rec.ik["knee_flexion_l"][n_idx] - rec.ik["knee_flexion_r"][n_idx])
                - (ta["knee_flexion_l"][n_idx] - ta["knee_flexion_r"][n_idx]))))
        if baseline.sum() < 3:
            continue
        b_idx = np.nonzero(baseline)[0]
        # per-rep PEAK heel rise (what HeelRiseRule consumes: max over the rep) vs the true peak ankle rise
        true_rise_window = (float(np.median(true_ankle_y[b_idx])) - true_ankle_y[idx]) * M_TO_CM
        true_rise_peak = float(np.max(true_rise_window))
        add("true_ankle_rise_peak_cm_tb", true_rise_peak)
        heel_window = heel_est[idx] - float(np.nanmedian(heel_est[b_idx]))
        if true_rise_peak >= MIN_FOOT_STATE_RISE_CM and np.isfinite(heel_window).any():
            add("heel_rise_peak_ratio_tb", float(np.nanmax(heel_window)) * M_TO_CM / true_rise_peak)
        if foot_rise_cm is not None and np.isfinite(foot_rise_cm[idx]).any():
            fs_peak = float(np.nanmax(foot_rise_cm[idx]))
            add("foot_state_heel_rise_peak_cm_tb", fs_peak)
            if true_rise_peak >= MIN_FOOT_STATE_RISE_CM:
                add("foot_state_heel_rise_peak_ratio_tb", fs_peak / true_rise_peak)
        if foot_rise_cm is not None and len(n_all):
            fs_near = foot_rise_cm[n_all]
            if np.isfinite(fs_near).any():
                fs_base = float(np.nanmedian(foot_rise_cm[b_idx])) if np.isfinite(foot_rise_cm[b_idx]).any() else 0.0
                fs_amp = float(np.nanmedian(fs_near)) - fs_base
                true_amp = float(np.median(true_ankle_y[b_idx]) - np.median(true_ankle_y[n_all])) * M_TO_CM
                if abs(true_amp) >= MIN_FOOT_STATE_RISE_CM:
                    add("foot_state_heel_rise_amp_ratio_tb", fs_amp / true_amp)
                else:
                    add("foot_state_false_heel_rise_tb_cm", fs_amp - true_amp)
        if knee_valid[b_idx].sum() >= 3:
            b_idx = b_idx[knee_valid[b_idx]]
        signals = [
            # name, measured tb series, measured sel series, truth series, truth at true bottom, min amp, scale, unit
            ("valgus", ik_valgus, final_valgus, true_valgus_series, true_bottom_valgus, MIN_VALGUS_AMP_DEG, 1.0, "deg"),
            ("heel_rise", heel_est, heel_est, heel_true, float(_heel_height(bottom_pts)[0]), MIN_HEEL_AMP_M, 1000.0,
             "mm"),
            ("hip_shift", shift_est, shift_est, shift_true, float(_hip_shift(bottom_pts)[0]), MIN_SHIFT_AMP_M, 1000.0,
             "mm"),
            ("pelvis_list", rec.ik["pelvis_list"], rec.final["pelvis_list"], ta["pelvis_list"], bt["pelvis_list"],
             MIN_LIST_AMP_DEG, 1.0, "deg"),
        ]
        for name, tb_series, sel_series, true_series, true_bottom, min_amp, unit_scale, unit in signals:
            true_base = float(np.median(true_series[b_idx]))
            if len(n_idx):
                meas_amp = float(np.nanmedian(tb_series[n_idx])) - float(np.nanmedian(tb_series[b_idx]))
                true_amp = float(np.median(true_series[n_idx])) - true_base
                if abs(true_amp) >= min_amp:
                    add(f"{name}_amp_ratio_tb", meas_amp / true_amp)
                else:
                    add(f"{name}_amp_err_tb_{unit}", (meas_amp - true_amp) * unit_scale)
            if not has_bottom:
                continue
            meas_amp = float(sel_series[i_star]) - float(np.nanmedian(sel_series[b_idx]))
            true_amp = true_bottom - true_base
            if abs(true_amp) >= min_amp:
                add(f"{name}_amp_ratio_sel", meas_amp / true_amp)
            else:
                add(f"{name}_amp_err_sel_{unit}", (meas_amp - true_amp) * unit_scale)
    for key in REP_KEYS:
        # per-rep window values are outlier-prone (a swap/outlier burst can own a 150 ms window): take the median
        # across reps for bottom-window metrics, the mean for peak/timing metrics
        values = rows.get(key, [])
        robust = "_tb" in key or "_amp_" in key
        m[key] = (float(np.median(values)) if values else float("nan")) if robust else _nanmean(values)
    m["depth_abs_err_final_deg"] = _nanmean([abs(v) for v in rows.get("depth_err_final_deg", [])])
    m["depth_worst_undershoot_final_deg"] = float(min(rows["depth_err_final_deg"])) if rows.get(
        "depth_err_final_deg") else float("nan")
    m["reps_evaluated"] = float(len(reps_used))

    # ---- stance width ----
    ok = processed & (rec.out_conf[:, 15] > 0) & (rec.out_conf[:, 16] > 0)
    width_est = _stance_width(np.nan_to_num(out))
    width_true = _stance_width(truth)
    m["stance_width_mae_mm"] = float(np.mean(np.abs(width_est - width_true)[ok])) * 1000 if ok.any() else float("nan")
    m["stance_change_ratio"] = float("nan")
    if sc.moving_intervals:
        before = ok & still & (times < sc.moving_intervals[0][0])
        after = ok & still & (times > sc.moving_intervals[-1][1]) & (times < sc.reps[0].start_s)
        if before.sum() >= 3 and after.sum() >= 3:
            true_change = np.median(width_true[after]) - np.median(width_true[before])
            if abs(true_change) >= MIN_STANCE_CHANGE_M:
                m["stance_change_ratio"] = float((np.median(width_est[after]) - np.median(width_est[before]))
                                                 / true_change)

    # ---- jitter while standing still ----
    still_proc = processed & still
    triples = np.nonzero(still_proc[1:-1] & still_proc[:-2] & still_proc[2:])[0] + 1
    if len(triples) >= 5:
        second = out[triples + 1, 11:19] - 2 * out[triples, 11:19] + out[triples - 1, 11:19]
        second_true = truth[triples + 1, 11:19] - 2 * truth[triples, 11:19] + truth[triples - 1, 11:19]
        m["jitter_still_mm"] = float(np.sqrt(np.nanmean(np.sum(second ** 2, axis=2)))) * 1000
        m["jitter_still_p50_mm"] = float(np.nanmedian(np.linalg.norm(second, axis=2))) * 1000
        m["jitter_still_truth_mm"] = float(np.sqrt(np.mean(np.sum(second_true ** 2, axis=2)))) * 1000
        knee_second = final_knee[triples + 1] - 2 * final_knee[triples] + final_knee[triples - 1]
        m["jitter_still_knee_final_deg"] = float(np.sqrt(np.nanmean(knee_second ** 2)))
        knee_second_ik = ik_knee[triples + 1] - 2 * ik_knee[triples] + ik_knee[triples - 1]
        m["jitter_still_knee_ik_deg"] = float(np.sqrt(np.nanmean(knee_second_ik ** 2)))
    else:
        for key in ("jitter_still_mm", "jitter_still_p50_mm", "jitter_still_truth_mm", "jitter_still_knee_final_deg",
                    "jitter_still_knee_ik_deg"):
            m[key] = float("nan")

    # ---- gates / calibration / compute ----
    t0 = prepared.ticks[0].t_call_s

    def first_time(mask: np.ndarray) -> float:
        hits = np.nonzero(mask)[0]
        return float(prepared.ticks[hits[0]].t_call_s - t0) if len(hits) else float("nan")

    m["standing_gate_s"] = first_time(rec.standing_ready)
    m["readiness_gate_s"] = first_time(rec.readiness_ready)
    m["chain_calibrated_s"] = first_time(rec.chain_calibrated == 1) if (rec.chain_calibrated >= 0).any() else \
        float("nan")
    m["chain_us_mean"] = float(np.nanmean(rec.chain_us)) if processed.any() else float("nan")
    m["chain_us_p95"] = float(np.nanpercentile(rec.chain_us[processed], 95)) if processed.any() else float("nan")
    m["frame_us_mean"] = float(np.nanmean(rec.total_us)) if processed.any() else float("nan")
    m["processed_frames"] = float(processed.sum())
    m["held_frac"] = float(rec.held[processed].mean()) if processed.any() else float("nan")
    m["predicted_frac"] = float(rec.predicted[processed].mean()) if processed.any() else float("nan")
    m["analysis_latency_ms"] = float(np.nanmean(rec.latency_ms[processed])) if processed.any() else float("nan")
    for name, values in rec.stage_us.items():
        m[f"stage_{name}_us_mean"] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
    m["foot_state_valid_frac"] = float(rec.foot_valid[processed].mean()) if processed.any() else float("nan")
    # Old IK returned exactly 0.0 knee flexion when hip/knee/ankle confidence < 0.1; the new one returns NaN.
    # `ik_knee_invalid_frac` counts both so old and new tails are comparable.
    if processed.any():
        m["ik_knee_zero_frac"] = float(np.mean((knee_l[processed] == 0.0) | (knee_r[processed] == 0.0)))
        m["ik_knee_nan_frac"] = float(np.mean(np.isnan(knee_l[processed]) | np.isnan(knee_r[processed])))
        m["ik_knee_invalid_frac"] = float(np.mean(~knee_valid[processed]))
    else:
        m["ik_knee_zero_frac"] = m["ik_knee_nan_frac"] = m["ik_knee_invalid_frac"] = float("nan")
    return {"metrics": {k: (round(v, 4) if isinstance(v, float) and math.isfinite(v) else
                            (None if isinstance(v, float) else v)) for k, v in m.items()},
            "per_keypoint_mm": per_kpt}
