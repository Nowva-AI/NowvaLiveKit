"""Synthetic squat kinematics (world frame, Y-down, 19 kpts) + 3-camera DLT simulation using the real triangulator."""
from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, "/Users/naiahoard/NowvaLiveKit/src")

from biomechanics.triangulation.calibration import SEGMENT_RATIOS, CalibrationResult, CameraCalibration  # noqa: E402
from biomechanics.triangulation.triangulator import DLTTriangulator  # noqa: E402
from biomechanics.utils.types import CocoKeypoints as CK, MultiViewPose, Skeleton2D, Skeleton3D  # noqa: E402

N_KPTS = 19
UP = np.array([0.0, -1.0, 0.0])
FWD = np.array([0.0, 0.0, 1.0])
LEFT = np.array([1.0, 0.0, 0.0])
HEIGHT_M = 1.885

TEMPOS = {
    # name: (descent_s, bottom_pause_s, ascent_s); min-jerk on HIP HEIGHT, ~0.6 m travel (1.885 m user, deep squat)
    # peak hip vertical speed = 1.875 * travel / T  (VBT: light-load peak ~1.6-2.0 m/s, 1RM mean ~0.3 m/s)
    "bw_normal": (1.20, 0.10, 0.90),
    "bw_fast": (0.70, 0.0, 0.56),
    "bw_explosive": (0.50, 0.0, 0.40),
    "loaded_light_fast": (0.90, 0.0, 0.70),
    "loaded_heavy": (1.50, 0.20, 2.00),
}


def body(height_m: float = HEIGHT_M) -> dict:
    r = SEGMENT_RATIOS
    h = height_m
    return dict(
        head=r["head_to_shoulder"] * h, sh_half=r["shoulder_width_half"] * h, torso=r["torso"] * h,
        hip_half=r["hip_width_half"] * h, femur=r["femur"] * h, tibia=r["tibia"] * h,
        uarm=r["upper_arm"] * h, farm=r["forearm"] * h,
        stance_half=0.22, toe_out_deg=15.0, foot_fwd=0.19, foot_drop=0.05,
    )


def min_jerk(tau: np.ndarray) -> np.ndarray:
    tau = np.clip(tau, 0.0, 1.0)
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def depth_profile(t: np.ndarray, tempo: str, stand_s: float = 0.5, n_reps: int = 1) -> np.ndarray:
    t_d, t_b, t_a = TEMPOS[tempo]
    period = stand_s + t_d + t_b + t_a
    s = np.zeros_like(t)
    for rep in range(n_reps):
        t0 = rep * period + stand_s
        local = t - t0
        down = min_jerk(local / t_d)
        up = 1.0 - min_jerk((local - t_d - t_b) / t_a)
        seg = np.where(local < 0, 0.0, np.where(local < t_d + t_b, down, np.where(local < t_d + t_b + t_a, up, 0.0)))
        s = np.maximum(s, seg)
    return s


def _two_bone(hip, ankle, a, b, pole):
    d_vec = ankle - hip
    dist = np.linalg.norm(d_vec)
    u = d_vec / dist
    dist = min(dist, a + b - 1e-9)
    x = (a * a - b * b + dist * dist) / (2 * dist)
    r = math.sqrt(max(a * a - x * x, 0.0))
    p = pole - np.dot(pole, u) * u
    p /= np.linalg.norm(p)
    return hip + x * u + r * p


def pose_world(s: float, b: dict, arms: str = "bar", knee_min_deg: float = 3.0, knee_max_deg: float = 125.0,
               lean_min_deg: float = 6.0, lean_max_deg: float = 45.0, hip_back_m: float = 0.30,
               valgus_m: float = 0.0) -> np.ndarray:
    """World frame, origin = standing hip midpoint, Y down. s in [0,1] depth fraction."""
    pts = np.zeros((N_KPTS, 3))
    leg_len = b["femur"] + b["tibia"]
    floor_y = leg_len
    to = math.radians(b["toe_out_deg"])
    ankles, dirs = {}, {}
    for side, sign in (("l", 1.0), ("r", -1.0)):
        dirs[side] = np.array([sign * math.sin(to), 0.0, math.cos(to)])
        ankles[side] = np.array([sign * b["stance_half"], floor_y, 0.0])
    lf, lt = b["femur"], b["tibia"]
    pz = -hip_back_m * s
    dx = b["hip_half"] - b["stance_half"]

    def _dy(knee_deg: float, pz_m: float) -> float:
        kk = math.radians(knee_deg)
        ha = math.sqrt(lf * lf + lt * lt + 2 * lf * lt * math.cos(kk))
        return math.sqrt(max(ha**2 - dx * dx - pz_m * pz_m, 1e-9))

    dy_top, dy_bot = _dy(knee_min_deg, 0.0), _dy(knee_max_deg, -hip_back_m)
    dy = dy_top - s * (dy_top - dy_bot)  # hip height linear in s -> min-jerk hip trajectory
    pelvis = np.array([0.0, floor_y - dy, pz])
    # shift so standing hip mid is at origin
    s0_dy = math.sqrt(max((lf + lt - 1e-3) ** 2 - dx * dx, 1e-9))
    pelvis[1] -= (floor_y - s0_dy)
    for key in ankles:
        ankles[key][1] -= (floor_y - s0_dy)
    hips = {"l": pelvis + b["hip_half"] * LEFT, "r": pelvis - b["hip_half"] * LEFT}
    for side, sign in (("l", 1.0), ("r", -1.0)):
        pole = dirs[side] - sign * LEFT * (valgus_m * s / 0.3)
        knee = _two_bone(hips[side], ankles[side], lf, lt, pole)
        toe = ankles[side] + b["foot_fwd"] * dirs[side] + np.array([0.0, b["foot_drop"], 0.0])
        i_hip, i_knee, i_ank, i_toe = (
            (CK.LEFT_HIP, CK.LEFT_KNEE, CK.LEFT_ANKLE, CK.LEFT_FOOT_INDEX) if side == "l"
            else (CK.RIGHT_HIP, CK.RIGHT_KNEE, CK.RIGHT_ANKLE, CK.RIGHT_FOOT_INDEX))
        pts[i_hip], pts[i_knee], pts[i_ank], pts[i_toe] = hips[side], knee, ankles[side], toe
    lean = math.radians(lean_min_deg + (lean_max_deg - lean_min_deg) * s)
    trunk_up = math.cos(lean) * UP + math.sin(lean) * FWD
    trunk_fwd = -math.sin(lean) * UP + math.cos(lean) * FWD
    sh_mid = pelvis + b["torso"] * trunk_up
    pts[CK.LEFT_SHOULDER] = sh_mid + b["sh_half"] * LEFT
    pts[CK.RIGHT_SHOULDER] = sh_mid - b["sh_half"] * LEFT
    nose = sh_mid + b["head"] * trunk_up + 0.08 * trunk_fwd
    pts[CK.NOSE] = nose
    pts[CK.LEFT_EYE] = nose + 0.03 * LEFT + 0.035 * UP - 0.01 * FWD
    pts[CK.RIGHT_EYE] = nose - 0.03 * LEFT + 0.035 * UP - 0.01 * FWD
    pts[CK.LEFT_EAR] = nose + 0.075 * LEFT - 0.09 * trunk_fwd
    pts[CK.RIGHT_EAR] = nose - 0.075 * LEFT - 0.09 * trunk_fwd
    for side, sign in (("l", 1.0), ("r", -1.0)):
        sh = pts[CK.LEFT_SHOULDER] if side == "l" else pts[CK.RIGHT_SHOULDER]
        if arms == "bar":
            # low-bar grip: hands just outside shoulders, slightly behind, elbows back/down; rigid to trunk
            wrist = sh + sign * 0.22 * LEFT - 0.10 * trunk_fwd + 0.02 * trunk_up
            elbow = sh + sign * 0.12 * LEFT - 0.20 * trunk_fwd - 0.12 * trunk_up
        else:
            # bodyweight counterbalance: straight arms from hanging (0 deg) to horizontal forward (90 deg)
            beta = math.radians(90.0 * s)
            arm_dir = math.cos(beta) * (-UP) + math.sin(beta) * FWD
            elbow = sh + b["uarm"] * arm_dir
            wrist = sh + (b["uarm"] + b["farm"]) * arm_dir
        if side == "l":
            pts[CK.LEFT_ELBOW], pts[CK.LEFT_WRIST] = elbow, wrist
        else:
            pts[CK.RIGHT_ELBOW], pts[CK.RIGHT_WRIST] = elbow, wrist
    return pts


def sequence_world(tempo: str, fps: float = 30.0, n_reps: int = 1, arms: str = "bar", stand_s: float = 0.5,
                   height_m: float = HEIGHT_M, t0: float = 0.0, **pose_kw) -> tuple[np.ndarray, np.ndarray]:
    b = body(height_m)
    t_d, t_b, t_a = TEMPOS[tempo]
    total = n_reps * (stand_s + t_d + t_b + t_a) + stand_s
    t = np.arange(0.0, total, 1.0 / fps)
    s = depth_profile(t, tempo, stand_s, n_reps)
    frames = np.stack([pose_world(float(si), b, arms=arms, **pose_kw) for si in s])
    return t + t0, frames


def hip_center(frames: np.ndarray) -> np.ndarray:
    centre = (frames[:, CK.LEFT_HIP] + frames[:, CK.RIGHT_HIP]) / 2.0
    return frames - centre[:, None, :]


# ---------------------------------------------------------------- cameras

def look_at_camera(cam_pos: np.ndarray, target: np.ndarray, f_px: float, res: tuple[int, int]) -> dict:
    z_axis = target - cam_pos
    z_axis /= np.linalg.norm(z_axis)
    down = np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(down, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    R = np.stack([x_axis, y_axis, z_axis])
    t = -R @ cam_pos
    K = np.array([[f_px, 0, res[0] / 2], [0, f_px, res[1] / 2], [0, 0, 1.0]])
    P = K @ np.hstack([R, t[:, None]])
    return dict(P=P, K=K, R=R, t=t)


def make_rig(radius_m: float = 3.2, azimuths_deg: tuple = (0.0, -50.0, 50.0), cam_height_y: float = -0.1,
             res: tuple[int, int] = (1280, 720), f_factor: float = 0.8) -> tuple[CalibrationResult, dict]:
    f_px = f_factor * res[0]
    cams = {}
    calib = CalibrationResult(athlete_height_m=HEIGHT_M)
    for i, az in enumerate(azimuths_deg):
        a = math.radians(az)
        pos = np.array([radius_m * math.sin(a), cam_height_y, radius_m * math.cos(a)])
        cam = look_at_camera(pos, np.array([0.0, 0.1, 0.0]), f_px, res)
        cid = str(i)
        cams[cid] = cam
        calib.cameras[cid] = CameraCalibration(
            camera_id=cid, projection_matrix=cam["P"], intrinsic_matrix=cam["K"], rotation_matrix=cam["R"],
            translation_vector=cam["t"][:, None], reprojection_error=0.0, resolution=res)
    return calib, cams


def project(P: np.ndarray, pts: np.ndarray) -> np.ndarray:
    h = np.hstack([pts, np.ones((len(pts), 1))]) @ P.T
    return h[:, :2] / h[:, 2:3]


def views_for_frame(world_pts: np.ndarray, cams: dict, rng: np.random.Generator, sigma_px: float = 2.0,
                    conf: float = 0.8, corrupt: dict | None = None, timestamp: float = 0.0) -> MultiViewPose:
    """corrupt: {cam_id: fn(uv (19,2), conf (19,)) -> (uv, conf)} applied to that view."""
    views = {}
    for cid, cam in cams.items():
        uv = project(cam["P"], world_pts) + rng.normal(0.0, sigma_px, size=(N_KPTS, 2))
        c = np.full(N_KPTS, conf)
        if corrupt and cid in corrupt:
            uv, c = corrupt[cid](uv.copy(), c.copy())
        arr = np.hstack([uv, c[:, None]])
        views[cid] = Skeleton2D.from_numpy(arr, timestamp=timestamp)
    return MultiViewPose(views=views, timestamp=timestamp, frame_index=0)


def triangulate_sequence(times: np.ndarray, frames_world: np.ndarray, calib: CalibrationResult, cams: dict,
                         seed: int = 0, sigma_px: float = 2.0, corrupt_at: dict | None = None) -> list[Skeleton3D]:
    """corrupt_at: {frame_idx: {cam_id: fn}}"""
    rng = np.random.default_rng(seed)
    tri = DLTTriangulator(calib, min_views=2, max_reprojection_error=15.0, min_confidence=0.3)
    out = []
    for i, (ts, pts) in enumerate(zip(times, frames_world)):
        corrupt = corrupt_at.get(i) if corrupt_at else None
        mv = views_for_frame(pts, cams, rng, sigma_px=sigma_px, corrupt=corrupt, timestamp=float(ts))
        out.append(tri.triangulate(mv))
    return out


def skel(arr: np.ndarray, ts: float, conf: np.ndarray | float = 0.8) -> Skeleton3D:
    c = np.full(len(arr), conf) if np.isscalar(conf) else conf
    return Skeleton3D.from_numpy(arr, confidences=c, timestamp=ts, frame_index=0)


def to_arr(sk: Skeleton3D) -> np.ndarray:
    return sk.to_numpy()


def confs(sk: Skeleton3D) -> np.ndarray:
    return np.array([kp.confidence for kp in sk.keypoints])
