"""Synthetic triangulated squat session for the smoothing audit.

World frame = pipeline frame: Y down, X subject's left, Z forward, ankles on y=0.
Noise path mirrors production: 3 pinhole cams (f=0.8*W guessed intrinsics) -> 2D gaussian
jitter -> RTMPose SimCC argmax quantisation (384/512 bins on a 192x256 squash of 1280x720)
-> unweighted DLT (SVD) -> confidence = min view conf * (1 - reproj/15) -> hip-mid recentre.
"""
from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, "/Users/naiahoard/NowvaLiveKit/src")

from biomechanics.utils.types import CocoKeypoints as CK  # noqa: E402

UP = np.array([0.0, -1.0, 0.0])
DOWN = np.array([0.0, 1.0, 0.0])
FWD = np.array([0.0, 0.0, 1.0])
LEFT = np.array([1.0, 0.0, 0.0])
N_KPTS = 19
FPS = 30.0
FRAME_W, FRAME_H = 1280, 720
QUANT_X_PX = FRAME_W / 192 / 2.0   # 3.333 px
QUANT_Y_PX = FRAME_H / 256 / 2.0   # 1.406 px
SEG = dict(head_to_shoulder=0.130, shoulder_width_half=0.105, torso=0.290,
           hip_width_half=0.0725, femur=0.245, tibia=0.235, upper_arm=0.175, forearm=0.150)


def body(height_m: float = 1.885) -> dict:
    h = height_m
    return dict(torso=SEG["torso"] * h, sh_half=SEG["shoulder_width_half"] * h,
                hip_half=SEG["hip_width_half"] * h, femur=SEG["femur"] * h, tibia=SEG["tibia"] * h,
                uarm=SEG["upper_arm"] * h, farm=SEG["forearm"] * h, head=SEG["head_to_shoulder"] * h,
                stance_half=0.20, toe_out_deg=15.0, foot_fwd=0.20, foot_drop=0.06)


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
         knee_min_deg: float = 4.0, knee_max_deg: float = 120.0,
         lean_min_deg: float = 8.0, lean_max_deg: float = 42.0) -> np.ndarray:
    pts = np.zeros((N_KPTS, 3))
    to = math.radians(b["toe_out_deg"])
    feet = {}
    for side, sign in (("l", 1.0), ("r", -1.0)):
        d_f = np.array([sign * math.sin(to), 0.0, math.cos(to)])
        ankle = np.array([sign * b["stance_half"], 0.0, 0.0])
        toe = ankle + b["foot_fwd"] * d_f + np.array([0.0, b["foot_drop"], 0.0])
        feet[side] = (ankle, toe, d_f)
    k = math.radians(knee_min_deg + (knee_max_deg - knee_min_deg) * s)
    lf, lt = b["femur"], b["tibia"]
    dist = math.sqrt(lf * lf + lt * lt + 2 * lf * lt * math.cos(k))
    pz = -0.28 * s + 0.02
    hl_off = np.array([b["hip_half"], 0.0, 0.0])
    a_l = feet["l"][0]
    dx = hl_off[0] - a_l[0]
    dz = pz - a_l[2]
    dy = -math.sqrt(max(dist * dist - dx * dx - dz * dz, 1e-9))
    pelvis = np.array([0.0, a_l[1] + dy, pz])
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
    sh_mid = pelvis + b["torso"] * trunk_dir
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


# (descent_s, pause_s, ascent_s, depth_fraction, valgus_l_deg, valgus_r_deg)
DEFAULT_REPS = [
    (1.2, 0.0, 1.0, 1.0, 0.0, 0.0),      # deep 120 deg, no fault
    (1.0, 0.0, 0.8, 1.0, 32.0, 0.0),     # deep, left valgus (GS peak ~15 deg: mild band 12-17)
    (0.8, 0.0, 0.7, 1.0, 0.0, 0.0),      # fast deep
    (1.5, 0.4, 1.0, 0.784, 0.0, 32.0),   # 95 deg = HALF depth fault band [90,100), paused, right valgus
    (0.7, 0.0, 0.6, 1.0, 32.0, 32.0),    # very fast deep, bilateral valgus
    (1.0, 0.1, 0.9, 0.612, 0.0, 0.0),    # 75 deg = QUARTER depth fault band [60,90)
]


def session(reps=None, stand_s: float = 2.0, gap_s: float = 1.5, fps: float = FPS,
            height_m: float = 1.885):
    """Returns world (T,19,3), s (T,), phase list, rep windows [(start,bottom_start,bottom_end,end)]."""
    reps = reps or DEFAULT_REPS
    b = body(height_m)
    s_list, vl_list, vr_list, phase = [], [], [], []
    windows = []
    dt = 1.0 / fps

    def stand(sec):
        for _ in range(int(round(sec * fps))):
            s_list.append(0.0); vl_list.append(0.0); vr_list.append(0.0); phase.append("idle")

    stand(stand_s)
    for (td, tp, ta, depth, vl, vr) in reps:
        start = len(s_list)
        nd = int(round(td * fps))
        for i in range(nd):
            tau = (i + 1) * dt / td
            s_list.append(depth * 0.5 * (1 - math.cos(math.pi * tau))); phase.append("descending")
            vl_list.append(vl); vr_list.append(vr)
        b0 = len(s_list) - 1
        for _ in range(int(round(tp * fps))):
            s_list.append(depth); phase.append("bottom"); vl_list.append(vl); vr_list.append(vr)
        b1 = len(s_list) - 1
        na = int(round(ta * fps))
        for i in range(na):
            tau = (i + 1) * dt / ta
            s_list.append(depth * 0.5 * (1 + math.cos(math.pi * tau))); phase.append("ascending")
            vl_list.append(vl); vr_list.append(vr)
        end = len(s_list) - 1
        windows.append((start, b0, b1, end))
        stand(gap_s)
    s_arr = np.array(s_list)
    # valgus scaled by s inside pose(); normalise by depth so full valgus at that rep's bottom
    world = np.stack([pose(si, b, vl, vr) for si, vl, vr in zip(s_arr, vl_list, vr_list)])
    return world, s_arr, phase, windows


def recenter(pts: np.ndarray) -> np.ndarray:
    return pts - ((pts[..., CK.LEFT_HIP, :] + pts[..., CK.RIGHT_HIP, :]) / 2.0)[..., None, :]


# ---------------------------------------------------------------- cameras

def _look_at(cam_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    z_axis = target - cam_pos
    z_axis /= np.linalg.norm(z_axis)
    # camera y points "down" in image = world +Y (world is Y-down)
    y_hint = DOWN
    x_axis = np.cross(y_hint, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return np.stack([x_axis, y_axis, z_axis])  # rows: world->cam rotation


def cameras(radius_m: float = 3.0, azimuths_deg=(-40.0, 0.0, 40.0), height_m: float = 1.0):
    f = 0.8 * FRAME_W
    K = np.array([[f, 0, FRAME_W / 2], [0, f, FRAME_H / 2], [0, 0, 1.0]])
    target = np.array([0.0, -0.75, 0.0])
    Ps = []
    for az in azimuths_deg:
        a = math.radians(az)
        pos = np.array([radius_m * math.sin(a), -height_m, radius_m * math.cos(a)])
        R = _look_at(pos, target)
        t = -R @ pos
        Ps.append(K @ np.hstack([R, t[:, None]]))
    return np.stack(Ps)  # (V,3,4)


def project(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    Xh = np.concatenate([X, np.ones(X.shape[:-1] + (1,))], axis=-1)
    uvw = np.einsum("vij,...j->...vi", P, Xh)
    return uvw[..., :2] / uvw[..., 2:3]  # (..., V, 2)


def dlt(P: np.ndarray, uv: np.ndarray) -> np.ndarray:
    # uv (..., V, 2) -> X (..., 3); unweighted SVD DLT as triangulator.py (3 views)
    rows_x = uv[..., 0:1] * P[:, 2, :] - P[:, 0, :]
    rows_y = uv[..., 1:2] * P[:, 2, :] - P[:, 1, :]
    A = np.concatenate([rows_x, rows_y], axis=-2)  # (..., 2V, 4)
    _, _, vt = np.linalg.svd(A)
    Xh = vt[..., -1, :]
    return Xh[..., :3] / Xh[..., 3:4]


def triangulated_noise(world: np.ndarray, rng: np.random.Generator, sigma_px: float = 4.0,
                       quantize: bool = True, outlier_p: float = 0.0, P: np.ndarray | None = None):
    """Returns (recentred noisy (T,19,3), confidence (T,19), mean reprojection error px (T,19))."""
    P = cameras() if P is None else P
    uv = project(P, world)  # (T,19,V,2)
    uv_noisy = uv + rng.normal(0.0, sigma_px, uv.shape)
    if outlier_p > 0:
        mask = rng.random(uv.shape[:-1]) < outlier_p
        uv_noisy += mask[..., None] * rng.normal(0.0, 40.0, uv.shape)
    if quantize:
        uv_noisy[..., 0] = np.round(uv_noisy[..., 0] / QUANT_X_PX) * QUANT_X_PX
        uv_noisy[..., 1] = np.round(uv_noisy[..., 1] / QUANT_Y_PX) * QUANT_Y_PX
    X = dlt(P, uv_noisy)
    reproj = np.linalg.norm(project(P, X) - uv_noisy, axis=-1).mean(axis=-1)
    view_conf = np.clip(rng.normal(0.64, 0.02, uv.shape[:-1]), 0.3, 1.0)
    conf = view_conf.min(axis=-1) * np.clip(1.0 - reproj / 15.0, 0.0, 1.0)
    conf = np.where(reproj >= 15.0, view_conf.min(axis=-1) * 0.1, conf)
    return recenter(X), conf, reproj


def gaussian_noise(world: np.ndarray, rng: np.random.Generator, sigma_m: float = 0.01):
    noisy = world + rng.normal(0.0, sigma_m, world.shape)
    conf = np.full(world.shape[:2], 0.6)
    return recenter(noisy), conf, np.zeros(world.shape[:2])


def _bump(t: np.ndarray, center: float, width: float) -> np.ndarray:
    x = (t - center) / width
    return np.where(np.abs(x) < 0.5, 0.5 * (1 + np.cos(2 * np.pi * x)), 0.0)


def session_hard(fps: float = FPS, height_m: float = 1.885, n_reps: int = 5):
    """Bounce squats: raised-cosine accel 0.25 s, cruise, 0.12 s decel into an immediate 0.12 s reversal (~1.2 g at
    the hip), cruise up, 0.25 s decel.  Transient 300 ms knee cave (swivel 35 deg) mid-ascent on reps 1 and 3 (L),
    rep 2 (R); rep 4 has a 30 cm-shallower 'stutter' (two small reversals 0.2 s apart) at the bottom."""
    b = body(height_m)
    dt = 1.0 / fps
    t_all, s_all, vl_all, vr_all, phase, windows = [], [], [], [], [], []
    t = 0.0

    def add(sv, vl, vr, ph):
        nonlocal t
        s_all.append(sv); vl_all.append(vl); vr_all.append(vr); phase.append(ph); t_all.append(t); t += dt

    for _ in range(int(2.0 * fps)):
        add(0.0, 0.0, 0.0, "idle")
    for r in range(n_reps):
        down_s, up_s = (0.85, 0.75) if r % 2 == 0 else (0.7, 0.6)
        ramp, brake = 0.25, 0.15
        fine = 1000
        # velocity profile in s-units, integrate then normalise so the rep reaches depth 1.0
        tt_d = np.linspace(0, down_s, fine)
        v_d = np.ones_like(tt_d)
        v_d[tt_d < ramp] = 0.5 * (1 - np.cos(np.pi * tt_d[tt_d < ramp] / ramp))
        m = tt_d > down_s - brake
        v_d[m] = 0.5 * (1 + np.cos(np.pi * (tt_d[m] - (down_s - brake)) / brake))
        tt_u = np.linspace(0, up_s, fine)
        v_u = np.ones_like(tt_u)
        m = tt_u < brake
        v_u[m] = 0.5 * (1 - np.cos(np.pi * tt_u[m] / brake))
        m = tt_u > up_s - ramp
        v_u[m] = 0.5 * (1 + np.cos(np.pi * (tt_u[m] - (up_s - ramp)) / ramp))
        s_d = np.cumsum(v_d); s_d /= s_d[-1]
        s_u = 1 - np.cumsum(v_u) / np.sum(v_u)
        start = len(s_all)
        nd, nu = int(round(down_s * fps)), int(round(up_s * fps))
        for i in range(nd):
            add(float(np.interp((i + 1) * dt, tt_d, s_d)), 0.0, 0.0, "descending")
        bottom = len(s_all) - 1
        if r == 4:  # stutter: dip up 4% and back down twice
            for i in range(int(0.4 * fps)):
                add(1.0 - 0.04 * math.sin(2 * math.pi * i * dt / 0.2) ** 2, 0.0, 0.0, "bottom")
        for i in range(nu):
            add(float(np.interp((i + 1) * dt, tt_u, s_u)), 0.0, 0.0, "ascending")
        end = len(s_all) - 1
        # transient valgus mid-ascent
        if r in (1, 2, 3):
            idx = np.arange(bottom + 1, end + 1)
            tloc = (idx - bottom) * dt
            bump = _bump(tloc, up_s * 0.45, 0.30) * 35.0
            for j, bv in zip(idx, bump):
                s_here = max(s_all[j], 0.05)
                if r in (1, 3):
                    vl_all[j] = bv / s_here
                else:
                    vr_all[j] = bv / s_here
        windows.append((start, bottom, bottom, end))
        for _ in range(int(1.2 * fps)):
            add(0.0, 0.0, 0.0, "idle")
    s_arr = np.array(s_all)
    world = np.stack([pose(si, b, vl, vr) for si, vl, vr in zip(s_arr, vl_all, vr_all)])
    return world, s_arr, phase, windows
