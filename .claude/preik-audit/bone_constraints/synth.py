"""Synthetic Y-down squat skeletons (19 kpts, triangulated layout) + triangulation-like noise."""
from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, "/Users/naiahoard/NowvaLiveKit/src")

from biomechanics.utils.types import CocoKeypoints as CK, Skeleton3D  # noqa: E402

UP = np.array([0.0, -1.0, 0.0])
DOWN = np.array([0.0, 1.0, 0.0])
FWD = np.array([0.0, 0.0, 1.0])
LEFT = np.array([1.0, 0.0, 0.0])
N_KPTS = 19

# calibration.py SEGMENT_RATIOS
SEG = dict(head_to_shoulder=0.130, shoulder_width_half=0.105, torso=0.290,
           hip_width_half=0.0725, femur=0.245, tibia=0.235, upper_arm=0.175, forearm=0.150)


def body(height_m: float = 1.885, hip_half_m: float | None = None) -> dict:
    h = height_m
    return dict(
        torso=SEG["torso"] * h, sh_half=SEG["shoulder_width_half"] * h,
        hip_half=hip_half_m if hip_half_m is not None else SEG["hip_width_half"] * h,
        femur=SEG["femur"] * h, tibia=SEG["tibia"] * h,
        uarm=SEG["upper_arm"] * h, farm=SEG["forearm"] * h,
        head=SEG["head_to_shoulder"] * h,
        stance_half=0.20, toe_out_deg=15.0, foot_fwd=0.20, foot_drop=0.06,
    )


def _two_bone(hip, ankle, a, b, pole, swivel_rad, medial):
    d_vec = ankle - hip
    dist = np.linalg.norm(d_vec)
    u = d_vec / dist
    dist = min(dist, a + b - 1e-6)
    x = (a * a - b * b + dist * dist) / (2 * dist)
    r = math.sqrt(max(a * a - x * x, 0.0))
    p = pole - np.dot(pole, u) * u
    p /= np.linalg.norm(p)
    m = np.cross(u, p)
    if np.dot(m, medial) < 0:
        m = -m
    return hip + x * u + r * (math.cos(swivel_rad) * p + math.sin(swivel_rad) * m)


def pose(s: float, b: dict, valgus_deg_l: float = 0.0, valgus_deg_r: float = 0.0,
         hip_shift_m: float = 0.0, list_m: float = 0.0, heel_rise_m: float = 0.0,
         knee_min_deg: float = 4.0, knee_max_deg: float = 120.0,
         lean_min_deg: float = 8.0, lean_max_deg: float = 42.0, spine_shorten_m: float = 0.0) -> np.ndarray:
    """World frame, ankles on floor at y=0 (Y down). s in [0,1] = depth fraction."""
    pts = np.zeros((N_KPTS, 3))
    to = math.radians(b["toe_out_deg"])
    feet = {}
    for side, sign in (("l", 1.0), ("r", -1.0)):
        d_f = np.array([sign * math.sin(to), 0.0, math.cos(to)])
        ankle0 = np.array([sign * b["stance_half"], 0.0, 0.0])
        toe = ankle0 + b["foot_fwd"] * d_f + np.array([0.0, b["foot_drop"], 0.0])
        ankle = ankle0.copy()
        if side == "l" and heel_rise_m != 0.0:
            # rotate the foot about the toe so the ankle-toe distance is preserved
            foot_len = math.hypot(b["foot_fwd"], b["foot_drop"])
            rise = min(b["foot_drop"] + heel_rise_m * s, foot_len - 1e-6)
            horiz = math.sqrt(foot_len * foot_len - rise * rise)
            ankle = toe - horiz * d_f + rise * UP
        feet[side] = (ankle, toe, d_f)

    k = math.radians(knee_min_deg + (knee_max_deg - knee_min_deg) * s)
    lf, lt = b["femur"], b["tibia"]
    dist = math.sqrt(lf * lf + lt * lt + 2 * lf * lt * math.cos(k))
    px = hip_shift_m * s
    pz = -0.28 * s + 0.02
    list_sin = max(-0.99, min(0.99, list_m * s / b["hip_half"]))
    hl_off = b["hip_half"] * (math.sqrt(1.0 - list_sin * list_sin) * LEFT + list_sin * DOWN)
    # solve pelvis height from the left leg
    a_l = feet["l"][0]
    dx = px + hl_off[0] - a_l[0]
    dz = pz - a_l[2]
    dy = -math.sqrt(max(dist * dist - dx * dx - dz * dz, 1e-9))
    py = a_l[1] + dy - hl_off[1]
    pelvis = np.array([px, py, pz])
    pts[CK.LEFT_HIP] = pelvis + hl_off
    pts[CK.RIGHT_HIP] = pelvis - hl_off

    for side, hip_i, knee_i, ank_i, toe_i, val, med in (
        ("l", CK.LEFT_HIP, CK.LEFT_KNEE, CK.LEFT_ANKLE, CK.LEFT_FOOT_INDEX, valgus_deg_l, -LEFT),
        ("r", CK.RIGHT_HIP, CK.RIGHT_KNEE, CK.RIGHT_ANKLE, CK.RIGHT_FOOT_INDEX, valgus_deg_r, LEFT),
    ):
        ankle, toe, d_f = feet[side]
        pts[ank_i] = ankle
        pts[toe_i] = toe
        pts[knee_i] = _two_bone(pts[hip_i], ankle, lf, lt, d_f, math.radians(val * s), med)

    lean = math.radians(lean_min_deg + (lean_max_deg - lean_min_deg) * s)
    trunk_dir = math.cos(lean) * UP + math.sin(lean) * FWD
    sh_mid = pelvis + (b["torso"] - spine_shorten_m * s) * trunk_dir
    pts[CK.LEFT_SHOULDER] = sh_mid + b["sh_half"] * LEFT
    pts[CK.RIGHT_SHOULDER] = sh_mid - b["sh_half"] * LEFT
    nose = sh_mid + b["head"] * trunk_dir + 0.08 * FWD
    pts[CK.NOSE] = nose
    pts[CK.LEFT_EYE] = nose + 0.035 * LEFT + 0.035 * UP - 0.02 * FWD
    pts[CK.RIGHT_EYE] = nose - 0.035 * LEFT + 0.035 * UP - 0.02 * FWD
    pts[CK.LEFT_EAR] = nose + 0.075 * LEFT - 0.09 * FWD
    pts[CK.RIGHT_EAR] = nose - 0.075 * LEFT - 0.09 * FWD
    arm_ang = math.radians(55 + 35 * s)
    arm_dir = math.cos(arm_ang) * DOWN + math.sin(arm_ang) * FWD
    for sh_i, el_i, wr_i in ((CK.LEFT_SHOULDER, CK.LEFT_ELBOW, CK.LEFT_WRIST),
                             (CK.RIGHT_SHOULDER, CK.RIGHT_ELBOW, CK.RIGHT_WRIST)):
        pts[el_i] = pts[sh_i] + b["uarm"] * arm_dir
        pts[wr_i] = pts[el_i] + b["farm"] * arm_dir
    return pts


def recenter(pts: np.ndarray) -> np.ndarray:
    return pts - (pts[CK.LEFT_HIP] + pts[CK.RIGHT_HIP]) / 2.0


def depth_profile(n_frames: int, fps: float = 30.0, rep_s: float = 2.6, stand_s: float = 2.0) -> np.ndarray:
    t = np.arange(n_frames) / fps
    s = np.where(t < stand_s, 0.0, 0.5 - 0.5 * np.cos(2 * np.pi * (t - stand_s) / rep_s))
    return s


def sequence(n_frames: int = 420, b: dict | None = None, **fault) -> tuple[np.ndarray, np.ndarray]:
    b = b or body()
    s = depth_profile(n_frames)
    truth = np.stack([recenter(pose(si, b, **fault)) for si in s])
    return truth, s


def add_noise(world_seq: np.ndarray, rng: np.random.Generator, sigma_m: float,
              z_scale: float = 1.0, outlier_p: float = 0.0, outlier_range=(0.10, 0.30),
              recenter_output: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Per-axis Gaussian noise (sigma_m), optional depth anisotropy and outliers. Returns (noisy, outlier_mask)."""
    noise = rng.normal(0.0, sigma_m, world_seq.shape)
    noise[..., 2] *= z_scale
    mask = rng.random(world_seq.shape[:2]) < outlier_p
    if outlier_p > 0:
        dirs = rng.normal(size=world_seq.shape)
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
        mags = rng.uniform(*outlier_range, size=world_seq.shape[:2])
        noise += mask[..., None] * dirs * mags[..., None]
    noisy = world_seq + noise
    if recenter_output:
        noisy = noisy - ((noisy[:, CK.LEFT_HIP] + noisy[:, CK.RIGHT_HIP]) / 2.0)[:, None, :]
    return noisy, mask


def to_skel(pts: np.ndarray, conf: np.ndarray | None = None, t: float = 0.0, i: int = 0) -> Skeleton3D:
    if conf is None:
        conf = np.full(len(pts), 0.9)
    return Skeleton3D.from_numpy(pts, confidences=list(conf), timestamp=t, frame_index=i)


def add_correlated_noise(world_seq: np.ndarray, rng: np.random.Generator, sigma_m: float, rho: float = 0.0,
                         z_scale: float = 1.0, outlier_p: float = 0.0, outlier_range=(0.10, 0.30),
                         outlier_burst: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """AR(1) per joint/axis noise with stationary std sigma_m (z scaled), outliers lasting outlier_burst frames.
    Returns (noisy recentered at noisy hip mid, outlier mask)."""
    n_frames, n_kpts, _ = world_seq.shape
    white = rng.normal(0.0, sigma_m, world_seq.shape)
    noise = np.empty_like(white)
    noise[0] = white[0]
    innov = math.sqrt(max(1.0 - rho * rho, 0.0))
    for t in range(1, n_frames):
        noise[t] = rho * noise[t - 1] + innov * white[t]
    noise[..., 2] *= z_scale
    mask = np.zeros((n_frames, n_kpts), dtype=bool)
    if outlier_p > 0:
        starts = rng.random((n_frames, n_kpts)) < (outlier_p / outlier_burst)
        for t, j in zip(*np.nonzero(starts)):
            d = rng.normal(size=3)
            d /= np.linalg.norm(d)
            mag = rng.uniform(*outlier_range)
            t_end = min(n_frames, t + outlier_burst)
            noise[t:t_end, j] += d * mag
            mask[t:t_end, j] = True
    noisy = world_seq + noise
    noisy = noisy - ((noisy[:, CK.LEFT_HIP] + noisy[:, CK.RIGHT_HIP]) / 2.0)[:, None, :]
    return noisy, mask


def apply_semantic_drift(seq: np.ndarray, s: np.ndarray, hip_fwd_m: float = 0.0, hip_down_m: float = 0.0,
                         knee_fwd_m: float = 0.0, knee_down_m: float = 0.0) -> np.ndarray:
    """Pose-dependent detector keypoint drift (grows with depth s): shifts hip/knee keypoints in world axes."""
    out = seq.copy()
    for hip_i in (CK.LEFT_HIP, CK.RIGHT_HIP):
        out[:, hip_i] += s[:, None] * np.array([0.0, hip_down_m, hip_fwd_m])
    for knee_i in (CK.LEFT_KNEE, CK.RIGHT_KNEE):
        out[:, knee_i] += s[:, None] * np.array([0.0, knee_down_m, knee_fwd_m])
    return out
