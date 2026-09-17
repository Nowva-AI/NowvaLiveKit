"""Run GroundClamp variants vs no clamp vs world-frame contact prototype on synthetic squat scenarios."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict

import numpy as np

from sim import CK, FPS, NoiseModel, Scenario, generate, measure
from proto import FootContactModel, FootContactModelRobust

from biomechanics.kinematics.valgus import TriangulatedValgusEstimator as TVE  # noqa: E402
from biomechanics.utils.bone_constraints import BoneLengthConstraints  # noqa: E402
from biomechanics.utils.ground_clamp import GroundClamp  # noqa: E402
from biomechanics.utils.position_filter import KeypointPositionSmoother  # noqa: E402
from biomechanics.utils.standing_gate import StandingPoseGate  # noqa: E402
from biomechanics.utils.types import Skeleton3D  # noqa: E402

CONFIDENCE = 0.9
N_TRI_KPTS = 19
VARIANTS = ("raw", "gc", "bone_bone", "bone_gc_bone", "full_nogc", "full", "proto", "proto21", "robust", "robust21")


def _gate() -> StandingPoseGate:
    # config/biomechanics.yaml standing_gate values + StandingGateConfig default leg extension
    return StandingPoseGate(min_confidence=0.25, max_knee_flexion_deg=25.0, max_trunk_flexion_deg=25.0,
                            min_torso_length_m=0.25, max_torso_length_m=0.80, min_leg_extension_ratio=0.6,
                            required_consecutive_frames=5)


def run_variant(variant: str, meas: dict) -> tuple[np.ndarray, dict]:
    rel = meas["rel"]
    world = meas["meas_world"]
    center = meas["center"]
    n = rel.shape[0]
    k = 21 if variant.endswith("21") else N_TRI_KPTS
    out = rel[:, :k].copy()
    info: dict = {"ready_frame": None, "gc_calibrated_frame": None, "floor_fire": 0, "eq_fire": 0,
                  "width_fire": 0, "gc_frames": 0, "states": [None] * n}

    if variant.startswith("proto") or variant.startswith("robust"):
        model = FootContactModel(k) if variant.startswith("proto") else FootContactModelRobust(k)
        for i in range(n):
            anchored, state = model.process(world[i, :k])
            out[i] = anchored - center[i]
            info["states"][i] = state
        info["ready_frame"] = 0
        return out, info

    gate = _gate()
    bones = BoneLengthConstraints(calibration_frames=30, tolerance=0.0, standing_gate=gate)
    ground = GroundClamp(calibration_frames=30, stance_width_tolerance_m=0.02, ankle_y_tolerance_m=0.01,
                         min_leg_extension_ratio=0.75, standing_gate=gate)
    smoother = KeypointPositionSmoother(min_cutoff=0.8, beta=4.0, d_cutoff=1.0)
    conf = np.full(k, CONFIDENCE)
    for i in range(n):
        sk = Skeleton3D.from_numpy(rel[i, :k], confidences=conf, timestamp=i / FPS, frame_index=i)
        gate.check(sk)
        if not gate.is_ready or variant == "raw":
            continue
        if info["ready_frame"] is None:
            info["ready_frame"] = i
        if variant in ("bone_bone", "bone_gc_bone", "full_nogc", "full"):
            sk = bones.enforce(sk)
        if variant in ("gc", "bone_gc_bone", "full"):
            if ground.is_calibrated:
                pts = sk.to_numpy()
                yl = min(pts[CK.LEFT_ANKLE, 1], ground._ankle_y_max_l)
                yr = min(pts[CK.RIGHT_ANKLE, 1], ground._ankle_y_max_r)
                info["floor_fire"] += int(pts[CK.LEFT_ANKLE, 1] > ground._ankle_y_max_l
                                          or pts[CK.RIGHT_ANKLE, 1] > ground._ankle_y_max_r)
                info["eq_fire"] += int(abs(yl - yr) > 0.01)
                width = float(np.linalg.norm(pts[CK.LEFT_ANKLE, [0, 2]] - pts[CK.RIGHT_ANKLE, [0, 2]]))
                info["width_fire"] += int(abs(width - ground._stance_width) > 0.02)
                info["gc_frames"] += 1
            elif info["gc_calibrated_frame"] is None:
                pass
            sk = ground.clamp(sk)
            if ground.is_calibrated and info["gc_calibrated_frame"] is None:
                info["gc_calibrated_frame"] = i
                info["gc_cal"] = {"ymax_l": ground._ankle_y_max_l, "ymax_r": ground._ankle_y_max_r,
                                  "width": ground._stance_width}
        if variant in ("full_nogc", "full"):
            sk = smoother.smooth(sk)
        if variant in ("bone_bone", "bone_gc_bone", "full_nogc", "full"):
            sk = bones.enforce(sk)
        out[i] = sk.to_numpy()
    return out, info


def _knee_flex(p: np.ndarray, hip: int, knee: int, ankle: int) -> np.ndarray:
    a = p[:, hip] - p[:, knee]
    b = p[:, ankle] - p[:, knee]
    cos = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return 180.0 - np.degrees(np.arccos(np.clip(cos, -1, 1)))


def _valgus(p: np.ndarray) -> np.ndarray:
    res = np.zeros((len(p), 3))
    for i in range(len(p)):
        ml = p[i, CK.LEFT_HIP] - p[i, CK.RIGHT_HIP]
        ml = ml / np.linalg.norm(ml)
        res[i, 0] = TVE._abduction(p[i, CK.LEFT_HIP], p[i, CK.LEFT_KNEE], p[i, CK.LEFT_ANKLE], ml, -1.0)
        res[i, 1] = TVE._abduction(p[i, CK.RIGHT_HIP], p[i, CK.RIGHT_KNEE], p[i, CK.RIGHT_ANKLE], ml, 1.0)
        res[i, 2] = TVE._kasr_3d(p[i, CK.LEFT_KNEE], p[i, CK.RIGHT_KNEE], p[i, CK.LEFT_ANKLE], p[i, CK.RIGHT_ANKLE])
    return res


def _width(p: np.ndarray) -> np.ndarray:
    return np.linalg.norm(p[:, CK.LEFT_ANKLE][:, [0, 2]] - p[:, CK.RIGHT_ANKLE][:, [0, 2]], axis=1)


def _ankle_above_toe(p: np.ndarray) -> np.ndarray:
    # Y-down: ankle above toe => toe_y - ankle_y > 0. Heel rise raises the ankle relative to the planted toe.
    return np.stack([p[:, CK.LEFT_FOOT_INDEX, 1] - p[:, CK.LEFT_ANKLE, 1],
                     p[:, CK.RIGHT_FOOT_INDEX, 1] - p[:, CK.RIGHT_ANKLE, 1]], axis=1)


def evaluate(gen: dict, meas: dict, out_rel: np.ndarray, info: dict, eval_start: int) -> dict:
    k = out_rel.shape[1]
    ideal = meas["ideal_world"][:, :k]
    est = out_rel + meas["center"][:, None, :]
    s = gen["s"]
    phase = gen["phase"]
    n = len(s)
    frames = np.arange(n)
    mask = frames >= eval_start
    sc = gen["scenario"]
    if sc.initial_width_m is not None:
        t = gen["t"]
        mask &= ~((t >= sc.step_start_s - 0.1) & (t < sc.step_start_s + sc.step_duration_s + 0.5))
    stand = mask & (phase == "stand") & (s == 0)
    bottom = mask & (phase == "bottom")

    def rmse(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(x ** 2))) if x.size else float("nan")

    ankle_err = np.linalg.norm(est[:, [CK.LEFT_ANKLE, CK.RIGHT_ANKLE]] - ideal[:, [CK.LEFT_ANKLE, CK.RIGHT_ANKLE]], axis=2)
    toe_err = np.linalg.norm(est[:, [CK.LEFT_FOOT_INDEX, CK.RIGHT_FOOT_INDEX]]
                             - ideal[:, [CK.LEFT_FOOT_INDEX, CK.RIGHT_FOOT_INDEX]], axis=2)
    ankle_dy = est[:, [CK.LEFT_ANKLE, CK.RIGHT_ANKLE], 1] - ideal[:, [CK.LEFT_ANKLE, CK.RIGHT_ANKLE], 1]

    kf_est = np.stack([_knee_flex(est, CK.LEFT_HIP, CK.LEFT_KNEE, CK.LEFT_ANKLE),
                       _knee_flex(est, CK.RIGHT_HIP, CK.RIGHT_KNEE, CK.RIGHT_ANKLE)], axis=1)
    kf_true = np.stack([_knee_flex(ideal, CK.LEFT_HIP, CK.LEFT_KNEE, CK.LEFT_ANKLE),
                        _knee_flex(ideal, CK.RIGHT_HIP, CK.RIGHT_KNEE, CK.RIGHT_ANKLE)], axis=1)
    kf_err = kf_est - kf_true
    tib_est = np.linalg.norm(est[:, [CK.LEFT_ANKLE, CK.RIGHT_ANKLE]] - est[:, [CK.LEFT_KNEE, CK.RIGHT_KNEE]], axis=2)
    tib_true = np.linalg.norm(ideal[:, [CK.LEFT_ANKLE, CK.RIGHT_ANKLE]] - ideal[:, [CK.LEFT_KNEE, CK.RIGHT_KNEE]], axis=2)

    idx_bottom = frames[bottom]
    v_est = _valgus(est[idx_bottom])
    v_true = _valgus(ideal[idx_bottom])
    w_est = _width(est)
    w_true = _width(ideal)
    at_est = _ankle_above_toe(est)
    at_true = _ankle_above_toe(ideal)
    base_mask = stand
    at_est_rel = at_est - np.median(at_est[base_mask], axis=0)
    at_true_rel = at_true - np.median(at_true[base_mask], axis=0)
    ydiff_est = est[:, CK.LEFT_ANKLE, 1] - est[:, CK.RIGHT_ANKLE, 1]
    ydiff_true = ideal[:, CK.LEFT_ANKLE, 1] - ideal[:, CK.RIGHT_ANKLE, 1]

    result = {
        "ankle_rmse_cm": 100 * rmse(ankle_err[mask]),
        "ankle_rmse_stand_cm": 100 * rmse(ankle_err[stand]),
        "ankle_dy_bias_stand_cm": 100 * float(np.mean(ankle_dy[stand])),
        "toe_rmse_cm": 100 * rmse(toe_err[mask]),
        "kneeflex_rmse_deg": rmse(kf_err[mask]),
        "kneeflex_bias_stand_deg": float(np.mean(kf_err[stand])),
        "kneeflex_rmse_stand_deg": rmse(kf_err[stand]),
        "kneeflex_bias_bottom_deg": float(np.mean(kf_err[bottom])),
        "kneeflex_rmse_bottom_deg": rmse(kf_err[bottom]),
        "tibia_bias_stand_cm": 100 * float(np.mean((tib_est - tib_true)[stand])),
        "tibia_rmse_cm": 100 * rmse((tib_est - tib_true)[mask]),
        "valgus_true_bottom_deg": float(np.mean(v_true[:, :2])),
        "valgus_bias_bottom_deg": float(np.mean(v_est[:, :2] - v_true[:, :2])),
        "valgus_rmse_bottom_deg": rmse(v_est[:, :2] - v_true[:, :2]),
        "kasr_true_bottom": float(np.mean(v_true[:, 2])),
        "kasr_bias_bottom": float(np.mean(v_est[:, 2] - v_true[:, 2])),
        "kasr_rmse_bottom": rmse(v_est[:, 2] - v_true[:, 2]),
        "width_true_cm": 100 * float(np.mean(w_true[mask])),
        "width_bias_cm": 100 * float(np.mean((w_est - w_true)[mask])),
        "width_rmse_cm": 100 * rmse((w_est - w_true)[mask]),
        "ankle_above_toe_true_bottom_cm": 100 * float(np.mean(at_true_rel[bottom], axis=0).max()),
        "ankle_above_toe_est_bottom_cm": 100 * float(np.mean(at_est_rel[bottom], axis=0)[
            int(np.argmax(np.mean(at_true_rel[bottom], axis=0)))]),
        "ankle_ydiff_true_bottom_cm": 100 * float(np.mean(ydiff_true[bottom])),
        "ankle_ydiff_est_bottom_cm": 100 * float(np.mean(ydiff_est[bottom])),
        "ankle_ydiff_rmse_cm": 100 * rmse((ydiff_est - ydiff_true)[mask]),
        "hip_ydiff_true_bottom_cm": 100 * float(np.mean(ideal[bottom, CK.LEFT_HIP, 1] - ideal[bottom, CK.RIGHT_HIP, 1])),
        "hip_ydiff_est_bottom_cm": 100 * float(np.mean(est[bottom, CK.LEFT_HIP, 1] - est[bottom, CK.RIGHT_HIP, 1])),
    }
    # lateral hip offset relative to the ankle midpoint (hip shift observable in a hip-centred frame)
    hip_mid_est = (est[:, CK.LEFT_HIP] + est[:, CK.RIGHT_HIP]) / 2
    hip_mid_true = (ideal[:, CK.LEFT_HIP] + ideal[:, CK.RIGHT_HIP]) / 2
    ank_mid_est = (est[:, CK.LEFT_ANKLE] + est[:, CK.RIGHT_ANKLE]) / 2
    ank_mid_true = (ideal[:, CK.LEFT_ANKLE] + ideal[:, CK.RIGHT_ANKLE]) / 2
    result["hip_shift_true_bottom_cm"] = 100 * float(np.mean((hip_mid_true - ank_mid_true)[bottom, 0]))
    result["hip_shift_est_bottom_cm"] = 100 * float(np.mean((hip_mid_est - ank_mid_est)[bottom, 0]))

    if info.get("gc_frames"):
        result["gc_floor_fire_pct"] = 100 * info["floor_fire"] / info["gc_frames"]
        result["gc_eq_fire_pct"] = 100 * info["eq_fire"] / info["gc_frames"]
        result["gc_width_fire_pct"] = 100 * info["width_fire"] / info["gc_frames"]
        result["gc_calibrated_frame"] = info["gc_calibrated_frame"]
        result["gc_cal"] = info.get("gc_cal")

    states = info["states"]
    if states[0] is not None:
        hr_true = gen["heel_rise_true"]
        rise_est = np.array([[st["ankle_rise_l"], st["ankle_rise_r"]] for st in states])
        side = int(np.argmax(np.mean(hr_true[bottom], axis=0)))
        result["proto_ankle_rise_true_bottom_cm"] = 100 * float(np.mean(hr_true[bottom, side]))
        result["proto_ankle_rise_est_bottom_cm"] = 100 * float(np.mean(rise_est[bottom, side]))
        result["proto_ankle_rise_rmse_cm"] = 100 * rmse((rise_est - hr_true)[mask])
        other = 1 - side
        result["proto_ankle_rise_other_bottom_cm"] = 100 * float(np.mean(rise_est[bottom, other]))
        contact = np.array([[st["ankle_contact_l"], st["ankle_contact_r"]] for st in states])
        result["proto_ankle_contact_stand_pct"] = 100 * float(np.mean(contact[stand]))
        result["proto_ankle_contact_bottom_pct"] = 100 * float(np.mean(contact[bottom]))
        if "heel_rise_l" in states[0]:
            heel_est = np.array([[st["heel_rise_l"], st["heel_rise_r"]] for st in states])
            result["proto_heel_rise_est_bottom_cm"] = 100 * float(np.mean(heel_est[bottom, side]))
        hip_above = np.array([st.get("hip_above_ankle_rest_m", np.nan) for st in states])
        true_hip_above = (np.mean(ideal[:, [CK.LEFT_ANKLE, CK.RIGHT_ANKLE], 1], axis=1) * 0
                          + (gen["floor_y"] - 0.075) - hip_mid_true[:, 1])
        result["proto_hip_height_err_bottom_std_cm"] = 100 * float(np.nanstd((hip_above - true_hip_above)[bottom]))
        result["proto_hip_height_err_bottom_bias_cm"] = 100 * float(np.nanmean((hip_above - true_hip_above)[bottom]))
    # F19 bridge grounding: hip height = max foot y (lowest foot kpt, Y-down) - hip y, per frame
    foot_idx = list(range(CK.LEFT_ANKLE, k))
    bridge_hip = np.max(est[:, foot_idx, 1], axis=1) - hip_mid_est[:, 1]
    bridge_true = np.max(ideal[:, foot_idx, 1], axis=1) - hip_mid_true[:, 1]
    result["bridge_hip_height_err_bottom_std_cm"] = 100 * float(np.std((bridge_hip - bridge_true)[bottom]))
    result["bridge_hip_height_err_bottom_bias_cm"] = 100 * float(np.mean((bridge_hip - bridge_true)[bottom]))
    return result


SCENARIOS = {
    "clean": Scenario("clean"),
    "valgus": Scenario("valgus", valgus_swivel_deg=30.0),
    "heel_rise_both": Scenario("heel_rise_both", heel_rise_deg_l=15.0, heel_rise_deg_r=15.0),
    "heel_rise_right": Scenario("heel_rise_right", heel_rise_deg_r=15.0),
    "hip_shift": Scenario("hip_shift", hip_shift_m=0.06),
    "pelvic_list": Scenario("pelvic_list", pelvic_list_deg=6.0),
    "stance_widen": Scenario("stance_widen", initial_width_m=0.30, stance_width_m=0.42),
    "tilt_roll5": Scenario("tilt_roll5", tilt_roll_deg=5.0),
    "tilt_pitch5": Scenario("tilt_pitch5", tilt_pitch_deg=5.0),
    "early_descent": Scenario("early_descent", initial_stand_s=0.5),
    "soft_knee_calib": Scenario("soft_knee_calib", soft_knee_calib_drop_m=0.04),
    "foot_snap": Scenario("foot_snap", foot_snap_to_knee=True),
    "foot_snap_heel_rise_right": Scenario("foot_snap_heel_rise_right", foot_snap_to_knee=True, heel_rise_deg_r=15.0),
    "valgus_widen": Scenario("valgus_widen", valgus_swivel_deg=30.0, initial_width_m=0.30, stance_width_m=0.42),
}


def main(noise_sigmas: list[float], seeds: list[int], scenario_names: list[str], out_path: str) -> dict:
    results: dict = {}
    for sigma in noise_sigmas:
        noise = NoiseModel(white_sigma_m=sigma, slow_sigma_m=0.4 * sigma)
        for name in scenario_names:
            gen = generate(SCENARIOS[name])
            per_variant: dict = {v: [] for v in VARIANTS}
            for seed in seeds:
                meas = measure(gen, noise, seed)
                outs = {v: run_variant(v, meas) for v in VARIANTS}
                cal_frames = [o[1]["gc_calibrated_frame"] for o in outs.values() if o[1].get("gc_calibrated_frame")]
                eval_start = max(cal_frames) + 1 if cal_frames else 45
                for v, (out_rel, info) in outs.items():
                    per_variant[v].append(evaluate(gen, meas, out_rel, info, eval_start))
            agg = {}
            for v, runs in per_variant.items():
                keys = [key for key in runs[0] if isinstance(runs[0][key], (int, float)) and runs[0][key] is not None]
                agg[v] = {key: float(np.nanmean([r[key] for r in runs if r.get(key) is not None])) for key in keys}
                if "gc_cal" in runs[0] and runs[0]["gc_cal"]:
                    agg[v]["gc_cal_first_seed"] = runs[0]["gc_cal"]
            results[f"sigma{sigma*100:.1f}cm/{name}"] = agg
            print(f"done sigma={sigma} {name}", file=sys.stderr)
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=1)
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigmas", default="0.015")
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--out", default="results.json")
    args = parser.parse_args()
    main([float(x) for x in args.sigmas.split(",")], list(range(args.seeds)),
         args.scenarios.split(","), args.out)
