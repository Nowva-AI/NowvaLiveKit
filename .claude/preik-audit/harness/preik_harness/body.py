"""Ground-truth squat motion generator with rigid bone lengths.

World frame matches the production triangulated frame produced by the T-pose PnP calibration: Y down,
X = subject's left, Z = subject's BACK (the front camera sits at -Z; a planar T-pose PnP forces this, see
REPORT.md), origin = hip midpoint during the T-pose. Legs are solved with 2-link IK from a driven pelvis to
planted (or stepping) feet, so femur/tibia/pelvis/trunk/arm lengths are exact at every instant.
Output keypoints: COCO17 + big toes (17, 18) + heels (19, 20) = 21 points.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

HEIGHT_M = 1.885
NUM_KPTS = 21

# True body (differs on purpose from calibration.py SEGMENT_RATIOS)
FEMUR_M = 0.245 * HEIGHT_M
TIBIA_M = 0.246 * HEIGHT_M
HIP_HALF_M = 0.068 * HEIGHT_M
TORSO_M = 0.300 * HEIGHT_M
SHOULDER_HALF_M = 0.110 * HEIGHT_M
UPPER_ARM_M = 0.186 * HEIGHT_M
FOREARM_M = 0.146 * HEIGHT_M
ANKLE_HEIGHT_M = 0.039 * HEIGHT_M
NOSE_UP_M = 0.100 * HEIGHT_M
NOSE_FWD_M = 0.09
EYE_HALF_M = 0.032
EAR_HALF_M = 0.075
FLOOR_Y_M = FEMUR_M + TIBIA_M + ANKLE_HEIGHT_M
# foot keypoints relative to the ankle, foot frame (x outward, y down, z back)
TOE_FROM_ANKLE = np.array([-0.02, 0.05, -0.195])
HEEL_FROM_ANKLE = np.array([0.0, 0.055, 0.065])
REACH_MARGIN_RATIO = 0.997
TRACK_DT_S = 0.001

L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK, L_TOE, R_TOE, L_HEEL, R_HEEL = 11, 12, 13, 14, 15, 16, 17, 18, 19, 20

SCENARIOS = ["clean", "valgus", "heel_rise", "hip_shift", "asymmetric_depth", "fast_reps", "stance_change",
             "walkout"]


def _rot_x(angle_rad: np.ndarray) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    o, z = np.ones_like(c), np.zeros_like(c)
    return np.stack([np.stack([o, z, z], -1), np.stack([z, c, -s], -1), np.stack([z, s, c], -1)], -2)


def _rot_y(angle_rad: np.ndarray) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    o, z = np.ones_like(c), np.zeros_like(c)
    return np.stack([np.stack([c, z, s], -1), np.stack([z, o, z], -1), np.stack([-s, z, c], -1)], -2)


def _rot_z(angle_rad: np.ndarray) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    o, z = np.ones_like(c), np.zeros_like(c)
    return np.stack([np.stack([c, -s, z], -1), np.stack([s, c, z], -1), np.stack([z, z, o], -1)], -2)


def _apply(rot: np.ndarray, vec: np.ndarray) -> np.ndarray:
    return np.einsum("nij,nj->ni", rot, np.broadcast_to(vec, rot.shape[:-2] + (3,)))


def _normalize(vec: np.ndarray) -> np.ndarray:
    return vec / np.maximum(np.linalg.norm(vec, axis=-1, keepdims=True), 1e-12)


def _min_jerk(tau: np.ndarray) -> np.ndarray:
    tau = np.clip(tau, 0.0, 1.0)
    return tau ** 3 * (10 - 15 * tau + 6 * tau ** 2)


@dataclass
class RepWindow:
    start_s: float
    bottom_start_s: float
    bottom_end_s: float
    end_s: float
    knee_target_deg: float


@dataclass
class Scenario:
    """Dense 1 kHz parameter tracks plus metadata; pose(times) evaluates exact rigid poses."""

    name: str
    seed: int
    duration_s: float
    tracks: dict[str, np.ndarray]
    reps: list[RepWindow]
    moving_intervals: list[tuple[float, float]] = field(default_factory=list)
    tpose_yaw_deg: float = 0.0

    def sample(self, times_s: np.ndarray) -> dict[str, np.ndarray]:
        t = self.tracks["t"]
        return {k: (np.interp(times_s, t, v) if v.ndim == 1 else
                    np.stack([np.interp(times_s, t, v[:, i]) for i in range(v.shape[1])], -1))
                for k, v in self.tracks.items() if k != "t"}

    def squat_phase(self, times_s: np.ndarray) -> np.ndarray:
        return np.interp(times_s, self.tracks["t"], self.tracks["s"])

    def still_mask(self, times_s: np.ndarray, guard_s: float = 0.25) -> np.ndarray:
        """True where the athlete is standing still (no rep, no stepping) within +-guard."""
        mask = np.ones_like(times_s, dtype=bool)
        busy = [(r.start_s, r.end_s) for r in self.reps] + self.moving_intervals
        for start, end in busy:
            mask &= ~((times_s > start - guard_s) & (times_s < end + guard_s))
        return mask

    def pose(self, times_s: np.ndarray) -> np.ndarray:
        return pose_from_params(self.sample(np.asarray(times_s, dtype=np.float64)))


def pose_from_params(p: dict[str, np.ndarray]) -> np.ndarray:
    n = len(p["s"])
    out = np.zeros((n, NUM_KPTS, 3))
    ankles, toes, heels = {}, {}, {}
    for side_name, side in (("l", 1.0), ("r", -1.0)):
        heading = np.radians(p[f"foot_heading_{side_name}"])
        rot_foot = _rot_y(heading)
        ankle_flat = np.stack([p[f"ankle_x_{side_name}"], FLOOR_Y_M - ANKLE_HEIGHT_M - p[f"lift_{side_name}"],
                               p[f"ankle_z_{side_name}"]], -1)
        side_vec = np.array([side, 1.0, 1.0])
        toe = ankle_flat + _apply(rot_foot, TOE_FROM_ANKLE * side_vec)
        beta = np.radians(p[f"heel_rise_{side_name}"])
        rot_rise = _rot_x(beta)
        ankle = toe + _apply(rot_foot, _apply(rot_rise, -TOE_FROM_ANKLE * side_vec))
        heel = toe + _apply(rot_foot, _apply(rot_rise, (HEEL_FROM_ANKLE - TOE_FROM_ANKLE) * side_vec))
        ankles[side_name], toes[side_name], heels[side_name] = ankle, toe, heel

    body_yaw = np.radians(p["body_yaw"])
    rot_body = _rot_y(body_yaw)
    ankle_mid = (ankles["l"] + ankles["r"]) / 2.0
    ankle_l_local = np.einsum("nji,nj->ni", rot_body, ankles["l"] - ankle_mid)
    lateral = np.abs(ankle_l_local[:, 0]) - HIP_HALF_M
    knee_flex = np.radians(p["knee_target"])
    shank_tilt = np.radians(p["shank_tilt"])
    hip_ankle_dist = np.sqrt(FEMUR_M ** 2 + TIBIA_M ** 2 + 2 * FEMUR_M * TIBIA_M * np.cos(knee_flex))
    forward = TIBIA_M * np.sin(shank_tilt) - FEMUR_M * np.sin(knee_flex - shank_tilt)
    vertical = np.sqrt(np.maximum(hip_ankle_dist ** 2 - forward ** 2 - lateral ** 2, 1e-4))
    pelvis = ankle_mid + np.einsum("nij,nj->ni", rot_body, np.stack([p["shift_lat"], -vertical, -forward], -1))
    pelvis = pelvis + p["sway"]

    rot_pelvis = np.einsum("nij,njk->nik", _rot_y(body_yaw + np.radians(p["pelvis_yaw"])),
                           _rot_z(np.radians(p["list"])))
    hip_offset = np.array([HIP_HALF_M, 0.0, 0.0])
    reach = REACH_MARGIN_RATIO * (FEMUR_M + TIBIA_M)
    push_down = np.zeros(n)
    for side_name, side in (("l", 1.0), ("r", -1.0)):
        hip = pelvis + _apply(rot_pelvis, hip_offset * side)
        delta = ankles[side_name] - hip
        horizontal_sq = delta[:, 0] ** 2 + delta[:, 2] ** 2
        needed_y = ankles[side_name][:, 1] - np.sqrt(np.maximum(reach ** 2 - horizontal_sq, 1e-6))
        push_down = np.maximum(push_down, needed_y - hip[:, 1])
    pelvis[:, 1] += push_down

    for side_name, side, hip_idx, knee_idx, ankle_idx, toe_idx, heel_idx in (
        ("l", 1.0, L_HIP, L_KNEE, L_ANK, L_TOE, L_HEEL), ("r", -1.0, R_HIP, R_KNEE, R_ANK, R_TOE, R_HEEL),
    ):
        hip = pelvis + _apply(rot_pelvis, hip_offset * side)
        ankle = ankles[side_name]
        axis = ankle - hip
        dist = np.linalg.norm(axis, axis=1)
        axis_unit = axis / dist[:, None]
        pole = _apply(_rot_y(np.radians(p[f"foot_heading_{side_name}"] + side * p[f"valgus_{side_name}"])),
                      np.array([0.0, 0.0, -1.0]))
        pole = _normalize(pole - np.sum(pole * axis_unit, 1, keepdims=True) * axis_unit)
        along = (FEMUR_M ** 2 - TIBIA_M ** 2 + dist ** 2) / (2 * dist)
        radius = np.sqrt(np.maximum(FEMUR_M ** 2 - along ** 2, 0.0))
        out[:, hip_idx] = hip
        out[:, knee_idx] = hip + along[:, None] * axis_unit + radius[:, None] * pole
        out[:, ankle_idx] = ankle
        out[:, toe_idx] = toes[side_name]
        out[:, heel_idx] = heels[side_name]

    rot_trunk = np.einsum("nij,njk->nik", rot_pelvis, _rot_x(np.radians(p["trunk_lean"])))
    neck = pelvis + _apply(rot_trunk, np.array([0.0, -TORSO_M, 0.0]))
    out[:, 5] = pelvis + _apply(rot_trunk, np.array([SHOULDER_HALF_M, -TORSO_M, 0.0]))
    out[:, 6] = pelvis + _apply(rot_trunk, np.array([-SHOULDER_HALF_M, -TORSO_M, 0.0]))
    rot_head = np.einsum("nij,njk->nik", rot_pelvis, _rot_x(np.radians(0.35 * p["trunk_lean"])))
    out[:, 0] = neck + _apply(rot_head, np.array([0.0, -NOSE_UP_M, -NOSE_FWD_M]))
    out[:, 1] = neck + _apply(rot_head, np.array([EYE_HALF_M, -NOSE_UP_M - 0.03, -NOSE_FWD_M + 0.02]))
    out[:, 2] = neck + _apply(rot_head, np.array([-EYE_HALF_M, -NOSE_UP_M - 0.03, -NOSE_FWD_M + 0.02]))
    out[:, 3] = neck + _apply(rot_head, np.array([EAR_HALF_M, -NOSE_UP_M - 0.02, 0.0]))
    out[:, 4] = neck + _apply(rot_head, np.array([-EAR_HALF_M, -NOSE_UP_M - 0.02, 0.0]))

    abduction = math.radians(10.0)
    for side, shoulder_idx, elbow_idx, wrist_idx in ((1.0, 5, 7, 9), (-1.0, 6, 8, 10)):
        upper = np.radians(p["arm_flex"])
        lower = upper + math.radians(15.0)
        upper_dir = np.stack([np.full(n, side * math.sin(abduction)), np.cos(upper) * math.cos(abduction),
                              -np.sin(upper) * math.cos(abduction)], -1)
        lower_dir = np.stack([np.full(n, side * math.sin(abduction)), np.cos(lower) * math.cos(abduction),
                              -np.sin(lower) * math.cos(abduction)], -1)
        out[:, elbow_idx] = out[:, shoulder_idx] + UPPER_ARM_M * np.einsum("nij,nj->ni", rot_body, upper_dir)
        out[:, wrist_idx] = out[:, elbow_idx] + FOREARM_M * np.einsum("nij,nj->ni", rot_body, lower_dir)
    return out


def tpose_skeleton() -> np.ndarray:
    """True-body T-pose at the world origin (legs straight, arms horizontal), shape (21, 3)."""
    pts = np.zeros((NUM_KPTS, 3))
    shoulder_y = -TORSO_M
    for side, s_idx, e_idx, w_idx, h_idx, k_idx, a_idx, t_idx, hl_idx in (
        (1.0, 5, 7, 9, L_HIP, L_KNEE, L_ANK, L_TOE, L_HEEL), (-1.0, 6, 8, 10, R_HIP, R_KNEE, R_ANK, R_TOE, R_HEEL),
    ):
        pts[s_idx] = [side * SHOULDER_HALF_M, shoulder_y, 0.0]
        pts[e_idx] = [side * (SHOULDER_HALF_M + UPPER_ARM_M), shoulder_y, 0.0]
        pts[w_idx] = [side * (SHOULDER_HALF_M + UPPER_ARM_M + FOREARM_M), shoulder_y, 0.0]
        pts[h_idx] = [side * HIP_HALF_M, 0.0, 0.0]
        pts[k_idx] = [side * HIP_HALF_M, FEMUR_M, 0.0]
        pts[a_idx] = [side * HIP_HALF_M, FEMUR_M + TIBIA_M, 0.0]
        pts[t_idx] = pts[a_idx] + TOE_FROM_ANKLE * np.array([side, 1.0, 1.0])
        pts[hl_idx] = pts[a_idx] + HEEL_FROM_ANKLE
    pts[0] = [0.0, shoulder_y - NOSE_UP_M, -NOSE_FWD_M]
    pts[1] = [EYE_HALF_M, shoulder_y - NOSE_UP_M - 0.03, -NOSE_FWD_M + 0.02]
    pts[2] = [-EYE_HALF_M, shoulder_y - NOSE_UP_M - 0.03, -NOSE_FWD_M + 0.02]
    pts[3] = [EAR_HALF_M, shoulder_y - NOSE_UP_M - 0.02, 0.0]
    pts[4] = [-EAR_HALF_M, shoulder_y - NOSE_UP_M - 0.02, 0.0]
    return pts


# ----------------------------------------------------------------------------------------------------------
# scenario construction
# ----------------------------------------------------------------------------------------------------------

def _foot_positions(half_stance_m: float, toe_out_deg: float, yaw_deg: float, center_xz: np.ndarray
                    ) -> dict[str, tuple[float, float, float]]:
    rot = _rot_y(np.array([math.radians(yaw_deg)]))[0]
    feet = {}
    for side_name, side in (("l", 1.0), ("r", -1.0)):
        local = rot @ np.array([side * half_stance_m, 0.0, 0.0])
        feet[side_name] = (center_xz[0] + local[0], center_xz[1] + local[2], yaw_deg - side * toe_out_deg)
    return feet


def build_scenario(name: str, seed: int) -> Scenario:
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario {name}; choose from {SCENARIOS}")
    rng = np.random.default_rng([seed, SCENARIOS.index(name), 7])
    fast = name == "fast_reps"

    # placement of the squat spot relative to the T-pose spot (defines world frame)
    base_offset_xz = rng.uniform(-0.03, 0.03, size=2)
    base_yaw = float(rng.uniform(-4.0, 4.0))
    half_stance = float(rng.uniform(0.18, 0.20))
    toe_out = float(rng.uniform(12.0, 18.0))

    # --- timeline of foot placements (keyframes) and moving intervals ---
    foot_keys: list[tuple[float, dict]] = []  # (time, feet dict) — feet dict holds positions at that time
    step_events: list[tuple[str, float, float, float]] = []  # (side, t0, t1, lift)
    moving: list[tuple[float, float]] = []
    body_yaw_keys = [(0.0, base_yaw)]

    feet = _foot_positions(half_stance, toe_out, base_yaw, base_offset_xz)
    t_cursor = 2.0 + float(rng.uniform(0.0, 0.3))
    foot_keys.append((0.0, dict(feet)))

    if name == "stance_change":
        t_cursor = 2.6
        wider = half_stance + 0.08
        new_feet = _foot_positions(wider, toe_out + 8.0, base_yaw, base_offset_xz)
        for side_name in ("l", "r"):
            foot_keys.append((t_cursor, dict(feet)))
            feet = dict(feet)
            feet[side_name] = new_feet[side_name]
            step_events.append((side_name, t_cursor, t_cursor + 0.45, 0.05))
            moving.append((t_cursor, t_cursor + 0.45))
            t_cursor += 0.45
            foot_keys.append((t_cursor, dict(feet)))
            t_cursor += 0.15
        t_cursor += 1.3
    elif name == "walkout":
        t_cursor = 1.8
        walk_start = t_cursor
        target_center = base_offset_xz + np.array([0.0, -0.5])  # 0.5 m forward = -Z
        target_yaw = base_yaw + 20.0
        final_feet = _foot_positions(half_stance, toe_out, target_yaw, target_center)
        mid_center = base_offset_xz + np.array([0.0, -0.25])
        mid_feet = _foot_positions(half_stance, toe_out, base_yaw + 10.0, mid_center)
        plan = [("l", mid_feet["l"]), ("r", final_feet["r"]), ("l", final_feet["l"])]
        for side_name, target in plan:
            foot_keys.append((t_cursor, dict(feet)))
            feet = dict(feet)
            feet[side_name] = target
            step_events.append((side_name, t_cursor, t_cursor + 0.5, 0.07))
            t_cursor += 0.5
            foot_keys.append((t_cursor, dict(feet)))
        moving.append((walk_start, t_cursor))
        body_yaw_keys += [(walk_start, base_yaw), (t_cursor, target_yaw)]
        t_cursor += 1.8

    # --- reps ---
    reps: list[RepWindow] = []
    rep_params = []
    for _ in range(5):
        duration = float(rng.uniform(0.8, 1.0) if fast else rng.uniform(1.2, 2.5))
        hold = float(rng.uniform(0.0, 0.06) if fast else rng.uniform(0.0, 0.15))
        descent = 0.47 * (duration - hold)
        start = t_cursor
        bottom_start = start + descent
        bottom_end = bottom_start + hold
        end = start + duration
        knee_target = float(rng.uniform(100.0, 120.0))
        reps.append(RepWindow(start, bottom_start, bottom_end, end, knee_target))
        rep_params.append((knee_target, float(rng.uniform(30.0, 38.0)), float(rng.uniform(35.0, 45.0))))
        t_cursor = end + float(rng.uniform(0.1, 0.3) if fast else rng.uniform(0.3, 0.8))
    duration_total = t_cursor + 1.2

    t = np.arange(0.0, duration_total + TRACK_DT_S, TRACK_DT_S)
    n = len(t)
    s = np.zeros(n)
    knee_bottom = np.full(n, rep_params[0][0])
    shank_bottom = np.full(n, rep_params[0][1])
    trunk_bottom = np.full(n, rep_params[0][2])
    for rep, (knee_b, shank_b, trunk_b) in zip(reps, rep_params):
        in_rep = (t >= rep.start_s) & (t <= rep.end_s)
        down = in_rep & (t < rep.bottom_start_s)
        hold = in_rep & (t >= rep.bottom_start_s) & (t <= rep.bottom_end_s)
        up = in_rep & (t > rep.bottom_end_s)
        s[down] = _min_jerk((t[down] - rep.start_s) / (rep.bottom_start_s - rep.start_s))
        s[hold] = 1.0
        s[up] = 1.0 - _min_jerk((t[up] - rep.bottom_end_s) / (rep.end_s - rep.bottom_end_s))
        after = t >= rep.start_s - 0.05
        knee_bottom[after] = knee_b
        shank_bottom[after] = shank_b
        trunk_bottom[after] = trunk_b

    knee_stand, shank_stand, trunk_stand = 4.0, 2.0, 3.0
    tracks: dict[str, np.ndarray] = {"t": t, "s": s}
    tracks["knee_target"] = knee_stand + s * (knee_bottom - knee_stand)
    tracks["shank_tilt"] = shank_stand + s * (shank_bottom - shank_stand)
    tracks["trunk_lean"] = trunk_stand + s * (trunk_bottom - trunk_stand)
    tracks["arm_flex"] = 5.0 + s * 75.0

    yaw_times = np.array([k[0] for k in body_yaw_keys] + [duration_total + 1.0])
    yaw_values = np.array([k[1] for k in body_yaw_keys] + [body_yaw_keys[-1][1]])
    tracks["body_yaw"] = _min_jerk_keys(t, yaw_times, yaw_values)

    # feet tracks
    for side_name in ("l", "r"):
        key_t = np.array([k[0] for k in foot_keys] + [duration_total + 1.0])
        xs = np.array([k[1][side_name][0] for k in foot_keys] + [foot_keys[-1][1][side_name][0]])
        zs = np.array([k[1][side_name][1] for k in foot_keys] + [foot_keys[-1][1][side_name][1]])
        hs = np.array([k[1][side_name][2] for k in foot_keys] + [foot_keys[-1][1][side_name][2]])
        tracks[f"ankle_x_{side_name}"] = _min_jerk_keys(t, key_t, xs)
        tracks[f"ankle_z_{side_name}"] = _min_jerk_keys(t, key_t, zs)
        tracks[f"foot_heading_{side_name}"] = _min_jerk_keys(t, key_t, hs)
        lift = np.zeros(n)
        for step_side, t0, t1, height in step_events:
            if step_side == side_name:
                mask = (t >= t0) & (t <= t1)
                lift[mask] = height * np.sin(np.pi * (t[mask] - t0) / (t1 - t0))
        tracks[f"lift_{side_name}"] = lift

    fault_s15 = s ** 1.5
    zeros = np.zeros(n)
    tracks["shift_lat"] = zeros.copy()
    tracks["list"] = zeros.copy()
    tracks["pelvis_yaw"] = zeros.copy()
    tracks["valgus_l"] = zeros.copy()
    tracks["valgus_r"] = zeros.copy()
    tracks["heel_rise_l"] = zeros.copy()
    tracks["heel_rise_r"] = zeros.copy()
    if name == "valgus":
        swivel = float(rng.uniform(26.0, 32.0))
        tracks["valgus_l"] = swivel * s ** 2
        tracks["valgus_r"] = swivel * s ** 2
    elif name == "heel_rise":
        rise = _smoothstep((s - 0.45) / 0.55) * math.degrees(math.asin(0.03 / 0.26))
        tracks["heel_rise_l"] = rise
        tracks["heel_rise_r"] = rise
    elif name == "hip_shift":
        tracks["shift_lat"] = 0.04 * fault_s15
        tracks["list"] = -5.0 * fault_s15
    elif name == "asymmetric_depth":
        tracks["shift_lat"] = 0.03 * fault_s15
        tracks["list"] = -12.0 * fault_s15
        tracks["pelvis_yaw"] = -8.0 * fault_s15

    # postural sway (small, always on)
    phases = rng.uniform(0, 2 * np.pi, size=4)
    sway = np.zeros((n, 3))
    sway[:, 0] = 0.002 * np.sin(2 * np.pi * 0.35 * t + phases[0]) + 0.001 * np.sin(2 * np.pi * 0.9 * t + phases[1])
    sway[:, 2] = 0.003 * np.sin(2 * np.pi * 0.25 * t + phases[2]) + 0.0015 * np.sin(2 * np.pi * 0.7 * t + phases[3])
    tracks["sway"] = sway
    return Scenario(name=name, seed=seed, duration_s=duration_total, tracks=tracks, reps=reps,
                    moving_intervals=moving)


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _min_jerk_keys(t: np.ndarray, key_t: np.ndarray, values: np.ndarray) -> np.ndarray:
    # min-jerk blend between successive keyframes (holds value when keys are equal)
    out = np.full_like(t, values[0], dtype=np.float64)
    for i in range(len(key_t) - 1):
        t0, t1 = key_t[i], key_t[i + 1]
        mask = (t >= t0) & (t <= t1)
        if t1 - t0 < 1e-9:
            continue
        out[mask] = values[i] + (values[i + 1] - values[i]) * _min_jerk((t[mask] - t0) / (t1 - t0))
    out[t > key_t[-1]] = values[-1]
    return out
