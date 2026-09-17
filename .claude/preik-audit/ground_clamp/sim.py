"""Synthetic world-frame squat generator + triangulation measurement model for the GroundClamp audit.

Y-down, X = subject's left, Z = forward (matches triangulation/calibration.py). Feet are planted on a
floor plane in the TRUE gravity frame; knees are solved from two-sphere intersection so bone lengths are exact.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, "/Users/naiahoard/NowvaLiveKit/src")

from biomechanics.utils.types import CocoKeypoints as CK  # noqa: E402

FPS = 30.0
HEIGHT_M = 1.885
FEMUR_M = 0.245 * HEIGHT_M
TIBIA_M = 0.235 * HEIGHT_M
HIP_HALF_M = 0.0725 * HEIGHT_M
TORSO_M = 0.29 * HEIGHT_M
SHOULDER_HALF_M = 0.105 * HEIGHT_M
ANKLE_HEIGHT_M = 0.075
TOE_FORWARD_M = 0.19
TOE_HEIGHT_M = 0.025
HEEL_BACK_M = 0.06
HEEL_HEIGHT_M = 0.03
MTP_FORWARD_M = 0.14
NUM_KPTS = 21  # COCO17 + big toes (17,18) + heels (19,20)


@dataclass
class Scenario:
    name: str
    stance_width_m: float = 0.38
    toe_out_deg: float = 15.0
    n_reps: int = 5
    initial_stand_s: float = 2.0
    descent_s: float = 1.2
    bottom_s: float = 0.3
    ascent_s: float = 1.0
    between_s: float = 1.2
    hip_drop_m: float = 0.52
    hip_back_m: float = 0.22
    valgus_swivel_deg: float = 0.0       # knee swivel about hip-ankle axis at bottom (medial +)
    heel_rise_deg_l: float = 0.0         # foot pitch about MTP at bottom
    heel_rise_deg_r: float = 0.0
    hip_shift_m: float = 0.0             # lateral pelvis translation at bottom (+ = toward left)
    pelvic_list_deg: float = 0.0         # + = left hip lower at bottom
    initial_width_m: float | None = None  # stance during calibration (walk-out), steps to stance_width_m
    step_start_s: float = 2.0
    step_duration_s: float = 0.6
    post_step_stand_s: float = 1.0
    soft_knee_calib_drop_m: float = 0.0  # hips lowered during initial standing (unrack soft knees)
    tilt_roll_deg: float = 0.0           # measurement frame rotation about Z (forward)
    tilt_pitch_deg: float = 0.0          # measurement frame rotation about X (lateral)
    lockout_rise_m: float = 0.0          # between reps hips end this much higher than during calibration
    foot_snap_to_knee: bool = False      # at each rep bottom one foot's keypoints snap to its knee (real RTMPose failure)


def _smooth_step(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(math.pi * u)


def depth_profile(sc: Scenario) -> tuple[np.ndarray, np.ndarray, float]:
    """Return per-frame squat fraction s in [0,1], phase label array, and rep start time."""
    rep_start = sc.initial_stand_s
    if sc.initial_width_m is not None:
        rep_start = sc.step_start_s + sc.step_duration_s + sc.post_step_stand_s
    rep_len = sc.descent_s + sc.bottom_s + sc.ascent_s + sc.between_s
    total_s = rep_start + sc.n_reps * rep_len
    n = int(round(total_s * FPS))
    t = np.arange(n) / FPS
    s = np.zeros(n)
    phase = np.array(["stand"] * n, dtype=object)
    for rep in range(sc.n_reps):
        t0 = rep_start + rep * rep_len
        t1 = t0 + sc.descent_s
        t2 = t1 + sc.bottom_s
        t3 = t2 + sc.ascent_s
        m = (t >= t0) & (t < t1)
        s[m] = _smooth_step((t[m] - t0) / sc.descent_s)
        phase[m] = "descent"
        m = (t >= t1) & (t < t2)
        s[m] = 1.0
        phase[m] = "bottom"
        m = (t >= t2) & (t < t3)
        s[m] = 1.0 - _smooth_step((t[m] - t2) / sc.ascent_s)
        phase[m] = "ascent"
    return s, phase, rep_start


def _rodrigues(v: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    return (v * math.cos(angle_rad) + np.cross(axis, v) * math.sin(angle_rad)
            + axis * np.dot(axis, v) * (1 - math.cos(angle_rad)))


def _solve_knee(hip: np.ndarray, ankle: np.ndarray, pref: np.ndarray, medial: np.ndarray,
                swivel_rad: float) -> np.ndarray:
    d_vec = ankle - hip
    d = float(np.linalg.norm(d_vec))
    u = d_vec / d
    d = min(d, FEMUR_M + TIBIA_M - 1e-6)
    a = (FEMUR_M ** 2 - TIBIA_M ** 2 + d ** 2) / (2 * d)
    r = math.sqrt(max(FEMUR_M ** 2 - a ** 2, 0.0))
    c = hip + a * u
    n1 = pref - np.dot(pref, u) * u
    n1 /= np.linalg.norm(n1)
    n2 = np.cross(u, n1)
    if np.dot(n2, medial) < 0:
        n2 = -n2
    return c + r * (math.cos(swivel_rad) * n1 + math.sin(swivel_rad) * n2)


def generate(sc: Scenario) -> dict:
    s, phase, rep_start = depth_profile(sc)
    n = len(s)
    t = np.arange(n) / FPS
    toe_out = math.radians(sc.toe_out_deg)
    pts = np.zeros((n, NUM_KPTS, 3))
    heel_rise_true = np.zeros((n, 2))   # ankle vertical rise above its planted rest (m), [L, R]
    foot_pitch_true = np.zeros((n, 2))  # deg
    width_true = np.zeros(n)

    w_final = sc.stance_width_m
    # standing hip-ankle vertical span, legs 0.15% short of straight (~6 deg flexion)
    dx = w_final / 2 - HIP_HALF_M
    span = math.sqrt((0.9985 * (FEMUR_M + TIBIA_M)) ** 2 - dx ** 2)
    floor_y = span + ANKLE_HEIGHT_M

    for i in range(n):
        si = s[i]
        # stance width (with step between walk-out and working stance)
        if sc.initial_width_m is not None:
            u = (t[i] - sc.step_start_s) / sc.step_duration_s
            w = sc.initial_width_m + (w_final - sc.initial_width_m) * _smooth_step(np.array(u)).item()
            lift = 0.05 * math.sin(math.pi * min(max(u, 0.0), 1.0)) if 0 < u < 1 else 0.0
        else:
            w = w_final
            lift = 0.0
        width_true[i] = w

        hip_mid = np.array([sc.hip_shift_m * si, sc.hip_drop_m * si, -sc.hip_back_m * si])
        if t[i] < rep_start and sc.soft_knee_calib_drop_m > 0:
            hip_mid[1] += sc.soft_knee_calib_drop_m
        if t[i] >= rep_start and sc.lockout_rise_m > 0:
            hip_mid[1] -= sc.lockout_rise_m * (1 - si)
        lam = math.radians(sc.pelvic_list_deg * si)
        hip_l = hip_mid + np.array([HIP_HALF_M * math.cos(lam), HIP_HALF_M * math.sin(lam), 0.0])
        hip_r = hip_mid + np.array([-HIP_HALF_M * math.cos(lam), -HIP_HALF_M * math.sin(lam), 0.0])

        for side_idx, side in enumerate((1.0, -1.0)):
            fwd = np.array([side * math.sin(toe_out), 0.0, math.cos(toe_out)])
            ankle0 = np.array([side * w / 2, floor_y - ANKLE_HEIGHT_M - lift, 0.0])
            toe = ankle0 + TOE_FORWARD_M * fwd + np.array([0.0, ANKLE_HEIGHT_M - TOE_HEIGHT_M, 0.0])
            heel = ankle0 - HEEL_BACK_M * fwd + np.array([0.0, ANKLE_HEIGHT_M - HEEL_HEIGHT_M, 0.0])
            mtp = ankle0 + MTP_FORWARD_M * fwd + np.array([0.0, ANKLE_HEIGHT_M, 0.0])
            beta_deg = (sc.heel_rise_deg_l if side > 0 else sc.heel_rise_deg_r) * si
            beta = math.radians(beta_deg)
            ankle = ankle0
            if beta != 0.0:
                axis = np.cross(fwd, np.array([0.0, -1.0, 0.0]))
                cand = mtp + _rodrigues(ankle0 - mtp, axis, beta)
                if cand[1] > ankle0[1]:
                    axis = -axis
                    cand = mtp + _rodrigues(ankle0 - mtp, axis, beta)
                ankle = cand
                heel = mtp + _rodrigues(heel - mtp, axis, beta)
            heel_rise_true[i, side_idx] = ankle0[1] - ankle[1] + lift
            foot_pitch_true[i, side_idx] = beta_deg
            hip = hip_l if side > 0 else hip_r
            medial = np.array([-side, 0.0, 0.0])
            pref = fwd + np.array([0.0, 0.0, 0.0])
            knee = _solve_knee(hip, ankle, pref, medial, math.radians(sc.valgus_swivel_deg * si))
            if side > 0:
                pts[i, CK.LEFT_HIP], pts[i, CK.LEFT_KNEE], pts[i, CK.LEFT_ANKLE] = hip, knee, ankle
                pts[i, CK.LEFT_FOOT_INDEX], pts[i, CK.LEFT_HEEL] = toe, heel
            else:
                pts[i, CK.RIGHT_HIP], pts[i, CK.RIGHT_KNEE], pts[i, CK.RIGHT_ANKLE] = hip, knee, ankle
                pts[i, CK.RIGHT_FOOT_INDEX], pts[i, CK.RIGHT_HEEL] = toe, heel

        lean = math.radians(12.0 + 30.0 * si)
        up = np.array([0.0, -math.cos(lean), math.sin(lean)])
        sh_mid = hip_mid + TORSO_M * up
        pts[i, CK.LEFT_SHOULDER] = sh_mid + [SHOULDER_HALF_M, 0, 0]
        pts[i, CK.RIGHT_SHOULDER] = sh_mid - [SHOULDER_HALF_M, 0, 0]
        pts[i, CK.LEFT_ELBOW] = pts[i, CK.LEFT_SHOULDER] + [0.05, 0.15, -0.12]
        pts[i, CK.RIGHT_ELBOW] = pts[i, CK.RIGHT_SHOULDER] + [-0.05, 0.15, -0.12]
        pts[i, CK.LEFT_WRIST] = pts[i, CK.LEFT_SHOULDER] + [0.12, -0.02, -0.05]
        pts[i, CK.RIGHT_WRIST] = pts[i, CK.RIGHT_SHOULDER] + [-0.12, -0.02, -0.05]
        head = sh_mid + 0.20 * up
        pts[i, CK.NOSE] = head + [0, 0, 0.09]
        pts[i, CK.LEFT_EYE] = head + [0.03, -0.03, 0.08]
        pts[i, CK.RIGHT_EYE] = head + [-0.03, -0.03, 0.08]
        pts[i, CK.LEFT_EAR] = head + [0.07, 0, 0]
        pts[i, CK.RIGHT_EAR] = head + [-0.07, 0, 0]

    return {
        "scenario": sc, "world": pts, "s": s, "phase": phase, "t": t, "rep_start": rep_start,
        "heel_rise_true": heel_rise_true, "foot_pitch_true": foot_pitch_true,
        "width_true": width_true, "floor_y": floor_y,
    }


def tilt_matrix(roll_deg: float, pitch_deg: float) -> np.ndarray:
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    rz = np.array([[math.cos(roll), -math.sin(roll), 0], [math.sin(roll), math.cos(roll), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)], [0, math.sin(pitch), math.cos(pitch)]])
    return rx @ rz


@dataclass
class NoiseModel:
    white_sigma_m: float = 0.015
    slow_sigma_m: float = 0.006
    slow_tau_s: float = 0.5
    foot_scale: float = 1.0


def measure(gen: dict, noise: NoiseModel, seed: int) -> dict:
    """Apply world tilt + triangulation noise, then hip-midpoint re-centring (triangulator.py:134-137)."""
    rng = np.random.default_rng(seed)
    sc: Scenario = gen["scenario"]
    rot = tilt_matrix(sc.tilt_roll_deg, sc.tilt_pitch_deg)
    ideal = gen["world"] @ rot.T
    n, k, _ = ideal.shape
    scale = np.ones((1, k, 1))
    for idx in (CK.LEFT_ANKLE, CK.RIGHT_ANKLE, CK.LEFT_FOOT_INDEX, CK.RIGHT_FOOT_INDEX,
                CK.LEFT_HEEL, CK.RIGHT_HEEL):
        scale[0, idx, 0] = noise.foot_scale
    white = rng.normal(0.0, noise.white_sigma_m, size=(n, k, 3))
    rho = math.exp(-1.0 / (noise.slow_tau_s * FPS))
    slow = np.zeros((n, k, 3))
    innov = rng.normal(0.0, noise.slow_sigma_m * math.sqrt(1 - rho ** 2), size=(n, k, 3))
    slow[0] = rng.normal(0.0, noise.slow_sigma_m, size=(k, 3))
    for i in range(1, n):
        slow[i] = rho * slow[i - 1] + innov[i]
    meas_world = ideal + (white + slow) * scale
    if sc.foot_snap_to_knee:
        s_arr = gen["s"]
        rep = -1
        in_snap = False
        for i in range(n):
            if s_arr[i] > 0.85 and not in_snap:
                in_snap = True
                rep += 1
            elif s_arr[i] <= 0.85:
                in_snap = False
            if in_snap:
                knee, ankle, toe, heel = ((CK.LEFT_KNEE, CK.LEFT_ANKLE, CK.LEFT_FOOT_INDEX, CK.LEFT_HEEL) if rep % 2 == 0
                                          else (CK.RIGHT_KNEE, CK.RIGHT_ANKLE, CK.RIGHT_FOOT_INDEX, CK.RIGHT_HEEL))
                base = meas_world[i, knee]
                meas_world[i, ankle] = base + np.array([0.0, 0.04, 0.02]) + white[i, ankle]
                meas_world[i, toe] = base + np.array([0.0, 0.07, 0.08]) + white[i, toe]
                meas_world[i, heel] = base + np.array([0.0, 0.07, -0.02]) + white[i, heel]
    center = (meas_world[:, CK.LEFT_HIP] + meas_world[:, CK.RIGHT_HIP]) / 2.0
    rel = meas_world - center[:, None, :]
    return {"ideal_world": ideal, "meas_world": meas_world, "center": center, "rel": rel, "rot": rot}
