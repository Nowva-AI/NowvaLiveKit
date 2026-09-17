"""Exp 6: systematic 3D bias from T-pose calibration model error (guessed focal 0.8w, canonical anthropometrics,
planar T-pose). Truth: cameras with focal f_true, a body whose segment ratios differ by +-5%, realistic face depth
offsets, optional imperfect T-pose (arms 10 deg low, feet 5 cm wider). Calibration replicates TPoseCalibrator.calibrate
(solvePnP ITERATIVE + RefineLM against build_tpose_model, K = 0.8w). Squat triangulated with recovered P, hip-centered.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

import synth

sys.path.insert(0, "/Users/naiahoard/NowvaLiveKit/src")
from biomechanics.triangulation import calibration as calib_mod  # noqa: E402
from biomechanics.triangulation.calibration import TPoseCalibrator  # noqa: E402

OUT = Path(__file__).parent
RNG = np.random.default_rng(5)
HEIGHT_M = 1.885
YAWS = (-40.0, 0.0, 40.0)


def perturbed_ratios(rel: float) -> dict:
    return {k: v * (1 + RNG.uniform(-rel, rel)) for k, v in calib_mod.SEGMENT_RATIOS.items()}


def true_tpose(ratios: dict, imperfect: bool) -> np.ndarray:
    saved = calib_mod.SEGMENT_RATIOS
    calib_mod.SEGMENT_RATIOS = ratios
    try:
        model = TPoseCalibrator(pose_estimator=None).build_tpose_model(HEIGHT_M)
    finally:
        calib_mod.SEGMENT_RATIOS = saved
    model = model.copy()
    # face keypoints are in front of the body plane (truth Z-backward -> forward is -Z)
    model[0, 2] -= 0.10; model[1, 2] -= 0.08; model[2, 2] -= 0.08; model[3, 2] += 0.0; model[4, 2] += 0.0
    if imperfect:
        droop = math.radians(10)
        for sh, el, wr in ((5, 7, 9), (6, 8, 10)):
            for j in (el, wr):
                d = model[j] - model[sh]
                r = abs(d[0])
                model[j] = model[sh] + np.array([math.copysign(r * math.cos(droop), d[0]), r * math.sin(droop), 0.0])
        for kn, an, s in ((13, 15, 1), (14, 16, -1)):
            model[an, 0] += s * 0.05
            model[kn, 0] += s * 0.025
    return model


def calibrate(P_true: np.ndarray, model_true: np.ndarray, width: int, height: int) -> np.ndarray:
    cal = TPoseCalibrator(pose_estimator=None, focal_length_factor=0.8)
    K = cal._build_intrinsics((width, height))
    canon = cal.build_tpose_model(HEIGHT_M)
    Ps = []
    for v in range(P_true.shape[0]):
        uv, _ = synth.project(P_true[v:v + 1], model_true)
        img = uv[0] + RNG.normal(0, 1.5 / math.sqrt(30), uv[0].shape)  # 30-frame average
        ok, rvec, tvec = cv2.solvePnP(canon, img, K, np.zeros(4), flags=cv2.SOLVEPNP_ITERATIVE)
        rvec, tvec = cv2.solvePnPRefineLM(canon, img, K, np.zeros(4), rvec, tvec)
        R, _ = cv2.Rodrigues(rvec)
        Ps.append(K @ np.hstack([R, tvec]))
    return np.stack(Ps)


def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    u, w = a - b, c - b
    return float(np.degrees(np.arccos(np.clip(np.dot(u, w) / (np.linalg.norm(u) * np.linalg.norm(w)), -1, 1))))


def metrics(X: np.ndarray) -> dict:
    hip = (X[11] + X[12]) / 2
    sh = (X[5] + X[6]) / 2
    up = np.array([0.0, -1.0, 0.0])
    trunk = sh - hip
    return {
        "knee_flex_l": 180 - angle(X[11], X[13], X[15]),
        "hip_flex_l": 180 - angle(X[5], X[11], X[13]),
        "trunk_lean": float(np.degrees(np.arccos(np.dot(trunk, up) / np.linalg.norm(trunk)))),
        "dorsi_l": float(np.degrees(np.arccos(np.dot(X[13] - X[15], up) / np.linalg.norm(X[13] - X[15])))),
        "femur_l": float(np.linalg.norm(X[11] - X[13])), "tibia_l": float(np.linalg.norm(X[13] - X[15])),
        "hip_width": float(np.linalg.norm(X[11] - X[12])), "shoulder_width": float(np.linalg.norm(X[5] - X[6])),
        "torso": float(np.linalg.norm(trunk)),
        "knee_sep_over_hip_width": float(abs(X[13, 0] - X[14, 0]) / np.linalg.norm(X[11] - X[12])),
    }


def run_case(f_factor: float, rel: float, imperfect: bool, n_draws: int = 12) -> dict:
    agg: dict = {}
    for _ in range(n_draws):
        ratios = perturbed_ratios(rel)
        P_true = synth.rig(YAWS, focal_px=f_factor * synth.IMG_W)
        model_true = true_tpose(ratios, imperfect)
        P_est = calibrate(P_true, model_true, synth.IMG_W, synth.IMG_H)
        # handedness / camera centre check (front camera)
        M = P_est[1][:, :3]
        K_est = np.array([[0.8 * synth.IMG_W, 0, synth.IMG_W / 2], [0, 0.8 * synth.IMG_W, synth.IMG_H / 2], [0, 0, 1]])
        Rt = np.linalg.inv(K_est) @ P_est[1]
        C_est = -Rt[:, :3].T @ Rt[:, 3]
        agg.setdefault("front_cam_center_est", []).append(C_est)
        seq, phase = synth.squat_sequence(ratios=ratios, height_m=HEIGHT_M, n_frames=72, reps=1)
        for f in range(0, 72, 3):
            gt = seq[f]
            uv, _ = synth.project(P_true, gt)
            uv = uv + RNG.normal(0, 1.5, uv.shape)
            X = synth.dlt(P_est, uv)
            reproj = synth.reproj_err(P_est, X, uv).mean()
            Xc = X - (X[11] + X[12]) / 2
            gc = gt - (gt[11] + gt[12]) / 2
            mg, me = metrics(gc), metrics(Xc)
            for k in mg:
                if k in ("femur_l", "tibia_l", "hip_width", "shoulder_width", "torso"):
                    agg.setdefault(k + "_err_pct", []).append(100 * (me[k] - mg[k]) / mg[k])
                elif k == "knee_sep_over_hip_width":
                    agg.setdefault(k + "_err", []).append(me[k] - mg[k])
                else:
                    agg.setdefault(k + "_err_deg", []).append(me[k] - mg[k])
            agg.setdefault("kpt_err_mm", []).append(float(np.linalg.norm(Xc - gc, axis=1).mean() * 1000))
            agg.setdefault("reproj_px", []).append(float(reproj))
            agg.setdefault("z_sign_toe_minus_heel", []).append(float(Xc[17, 2] - Xc[19, 2]))
    out = {}
    for k, v in agg.items():
        if k == "front_cam_center_est":
            out[k] = np.mean(v, axis=0).round(2).tolist()
            continue
        v = np.array(v)
        out[k] = {"mean": round(float(v.mean()), 3), "mean_abs": round(float(np.abs(v).mean()), 3), "p95_abs": round(float(np.percentile(np.abs(v), 95)), 3)}
    return out


def main() -> None:
    cases = {}
    for f_factor in (0.55, 0.65, 0.8, 1.0):
        cases[f"f{f_factor}_ratios_exact"] = run_case(f_factor, 0.0, False, n_draws=2)
        cases[f"f{f_factor}_ratios_pm5pct"] = run_case(f_factor, 0.05, False)
    cases["f0.8_ratios_pm5pct_imperfect_tpose"] = run_case(0.8, 0.05, True)
    cases["f0.65_ratios_pm5pct_imperfect_tpose"] = run_case(0.65, 0.05, True)
    (OUT / "exp6_results.json").write_text(json.dumps(cases, indent=1))
    print("true front camera centre:", (-synth.look_at_camera(0.0)[2].T @ synth.look_at_camera(0.0)[3]).round(2).tolist())
    keys = ["kpt_err_mm", "reproj_px", "knee_flex_l_err_deg", "hip_flex_l_err_deg", "trunk_lean_err_deg", "dorsi_l_err_deg",
            "femur_l_err_pct", "tibia_l_err_pct", "hip_width_err_pct", "shoulder_width_err_pct", "torso_err_pct",
            "knee_sep_over_hip_width_err", "z_sign_toe_minus_heel"]
    for name, c in cases.items():
        print(f"\n== {name}  est front cam centre {c['front_cam_center_est']}")
        for k in keys:
            print(f"   {k:30s} mean {c[k]['mean']:8.3f}  mean|.| {c[k]['mean_abs']:8.3f}  p95|.| {c[k]['p95_abs']:8.3f}")


if __name__ == "__main__":
    main()
