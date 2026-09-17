"""Shared synthetic squat skeleton + 3-camera rig + triangulation method prototypes (numpy, vectorized).

World frame (ground truth): X = subject's left, Y = down, Z = backward (away from the front camera),
origin = standing hip midpoint. Right-handed (MediaPipe world convention). Cameras sit in front (Z < 0).
Keypoint layout: COCO-17 + 17/18 big toes + 19/20 heels (CK enum).
"""
from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, "/Users/naiahoard/NowvaLiveKit/src")
from biomechanics.triangulation.calibration import SEGMENT_RATIOS  # noqa: E402

NUM_KPTS = 21
L_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18), (19, 20)]
LEG_PAIRS = [(11, 12), (13, 14), (15, 16), (17, 18), (19, 20)]
IMG_W, IMG_H = 1280, 720
ANKLE_HEIGHT_M = 0.08


def lr_perm(pairs: list[tuple[int, int]]) -> np.ndarray:
    perm = np.arange(NUM_KPTS)
    for a, b in pairs:
        perm[a], perm[b] = b, a
    return perm


def squat_skeleton(depth_phase: float, height_m: float = 1.885, ratios: dict | None = None,
                   max_knee_flex_deg: float = 120.0, max_dorsi_deg: float = 35.0, max_trunk_deg: float = 40.0,
                   stance_extra_m: float = 0.08, valgus_m: float = 0.0) -> np.ndarray:
    r = ratios or SEGMENT_RATIOS
    femur, tibia, torso = r["femur"] * height_m, r["tibia"] * height_m, r["torso"] * height_m
    hip_half, sh_half, head = r["hip_width_half"] * height_m, r["shoulder_width_half"] * height_m, r["head_to_shoulder"] * height_m
    upper_arm, forearm = r["upper_arm"] * height_m, r["forearm"] * height_m
    knee_flex = math.radians(max_knee_flex_deg * depth_phase)
    dorsi = math.radians(max_dorsi_deg * depth_phase)
    trunk = math.radians(max_trunk_deg * depth_phase)
    thigh = knee_flex - dorsi
    fwd = np.array([0.0, 0.0, -1.0])
    up = np.array([0.0, -1.0, 0.0])
    ankle_y = femur + tibia  # standing
    pts = np.zeros((NUM_KPTS, 3))
    for side, sign in (("l", 1.0), ("r", -1.0)):
        ankle = np.array([sign * (hip_half + stance_extra_m), ankle_y, 0.0])
        knee = ankle + tibia * (up * math.cos(dorsi) + fwd * math.sin(dorsi))
        knee[0] = sign * (hip_half + stance_extra_m * tibia / (femur + tibia) - valgus_m * depth_phase)
        hip = knee + femur * (up * math.cos(thigh) - fwd * math.sin(thigh))
        hip[0] = sign * hip_half
        i_hip, i_knee, i_ank, i_toe, i_heel = (11, 13, 15, 17, 19) if side == "l" else (12, 14, 16, 18, 20)
        pts[i_hip], pts[i_knee], pts[i_ank] = hip, knee, ankle
        toe_out = math.radians(15.0)
        pts[i_toe] = ankle + np.array([sign * 0.17 * math.sin(toe_out), ANKLE_HEIGHT_M - 0.02, -0.17 * math.cos(toe_out)])
        pts[i_heel] = ankle + np.array([0.0, ANKLE_HEIGHT_M - 0.01, 0.06])
    # normalise so standing hip midpoint is origin (hip y at phase 0 == 0)
    hip_mid = (pts[11] + pts[12]) / 2
    trunk_dir = up * math.cos(trunk) + fwd * math.sin(trunk)
    sh_mid = hip_mid + torso * trunk_dir
    pts[5] = sh_mid + np.array([sh_half, 0, 0])
    pts[6] = sh_mid + np.array([-sh_half, 0, 0])
    nose = sh_mid + head * trunk_dir + fwd * 0.09
    pts[0] = nose
    pts[1] = nose + np.array([0.03, -0.035, 0.03]); pts[2] = nose + np.array([-0.03, -0.035, 0.03])
    pts[3] = nose + np.array([0.075, -0.02, 0.10]); pts[4] = nose + np.array([-0.075, -0.02, 0.10])
    for i_sh, i_el, i_wr, sign in ((5, 7, 9, 1.0), (6, 8, 10, -1.0)):
        el = pts[i_sh] + upper_arm * np.array([sign * 0.25, 0.55, 0.80]) / np.linalg.norm([0.25, 0.55, 0.80])
        wr = el + forearm * np.array([-sign * 0.1, -0.95, -0.3]) / np.linalg.norm([0.1, 0.95, 0.3])
        pts[i_el], pts[i_wr] = el, wr
    return pts


def squat_sequence(n_frames: int = 216, reps: int = 3, **kw) -> np.ndarray:
    t = np.arange(n_frames) / n_frames * reps
    phase = 0.5 - 0.5 * np.cos(2 * np.pi * t)
    return np.stack([squat_skeleton(float(p), **kw) for p in phase]), phase


def look_at_camera(yaw_deg: float, radius_m: float = 3.5, cam_height_above_floor_m: float = 1.0,
                   target_height_above_floor_m: float = 0.9, floor_y: float = 0.98, focal_px: float = 0.8 * IMG_W
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    yaw = math.radians(yaw_deg)
    center = np.array([radius_m * math.sin(yaw), floor_y - cam_height_above_floor_m, -radius_m * math.cos(yaw)])
    target = np.array([0.0, floor_y - target_height_above_floor_m, 0.0])
    z_c = target - center
    z_c /= np.linalg.norm(z_c)
    x_c = np.cross(np.array([0.0, 1.0, 0.0]), z_c)
    x_c /= np.linalg.norm(x_c)
    y_c = np.cross(z_c, x_c)
    R = np.stack([x_c, y_c, z_c])
    t = -R @ center
    K = np.array([[focal_px, 0, IMG_W / 2], [0, focal_px, IMG_H / 2], [0, 0, 1.0]])
    P = K @ np.hstack([R, t[:, None]])
    return P, K, R, t


def rig(yaws: tuple[float, ...] = (-40.0, 0.0, 40.0), **kw) -> np.ndarray:
    return np.stack([look_at_camera(y, **kw)[0] for y in yaws])


def project(P: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # P (V,3,4), X (..., K, 3) -> uv (V, ..., K, 2), depth (V, ..., K)
    Xh = np.concatenate([X, np.ones(X.shape[:-1] + (1,))], -1)
    ph = np.einsum("vij,...kj->v...ki", P, Xh)
    return ph[..., :2] / ph[..., 2:3], ph[..., 2]


# ------------------------------------------------------------------------------------------ solvers
def dlt(P: np.ndarray, uv: np.ndarray, weights: np.ndarray | None = None, row_normalize: bool = False) -> np.ndarray:
    # P (V,3,4); uv (V,K,2); weights (V,K) -> X (K,3)
    V, K = uv.shape[:2]
    A = np.empty((K, 2 * V, 4))
    for v in range(V):
        A[:, 2 * v] = uv[v, :, 0, None] * P[v, 2] - P[v, 0]
        A[:, 2 * v + 1] = uv[v, :, 1, None] * P[v, 2] - P[v, 1]
    if row_normalize:
        A /= np.linalg.norm(A, axis=2, keepdims=True) + 1e-12
    if weights is not None:
        w = np.repeat(weights.T, 2, axis=1)  # (K, 2V)
        A *= w[..., None]
    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[:, -1]
    return Xh[:, :3] / Xh[:, 3:4]


def dlt_normalized(P: np.ndarray, Ks: np.ndarray, uv: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    # Hartley-style: work in K^-1 normalised image coordinates (per camera), then row-normalise.
    Kinv = np.linalg.inv(Ks)
    uvh = np.concatenate([uv, np.ones(uv.shape[:-1] + (1,))], -1)
    n = np.einsum("vij,vkj->vki", Kinv, uvh)[..., :2]
    Pn = np.einsum("vij,vjk->vik", Kinv, P)
    return dlt(Pn, n, weights, row_normalize=True)


def dlt_depth_irls(P: np.ndarray, uv: np.ndarray, weights: np.ndarray | None = None, iters: int = 2) -> np.ndarray:
    # Reweight rows by 1/projective depth so the algebraic residual approximates pixel reprojection error.
    X = dlt(P, uv, weights)
    V, K = uv.shape[:2]
    for _ in range(iters):
        depth = np.einsum("vj,kj->vk", P[:, 2], np.hstack([X, np.ones((K, 1))]))
        w = 1.0 / np.maximum(np.abs(depth), 1e-6)
        if weights is not None:
            w = w * weights
        X = dlt(P, uv, w / w.max())
    return X


def reproj_err(P: np.ndarray, X: np.ndarray, uv: np.ndarray) -> np.ndarray:
    pr, _ = project(P, X)
    return np.linalg.norm(pr - uv, axis=-1)  # (V, K)


def best_subset(P: np.ndarray, uv: np.ndarray, tau_px: float, weights: np.ndarray | None = None,
                solver=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 3 views: triple if max view residual <= tau, else the pair with lowest own residual.
    # Returns X (K,3), n_views_used (K,), rms_px (K,)
    solve = solver or (lambda PP, uu, ww: dlt(PP, uu, ww))
    V, K = uv.shape[:2]
    X3 = solve(P, uv, weights)
    e3 = reproj_err(P, X3, uv)
    X = X3.copy()
    n_used = np.full(K, V)
    rms = np.sqrt((e3 ** 2).mean(0))
    bad = e3.max(0) > tau_px
    if V == 3 and bad.any():
        best_rms = np.full(K, np.inf)
        best_X = X3.copy()
        for drop in range(3):
            keep = [v for v in range(3) if v != drop]
            w = None if weights is None else weights[keep]
            Xp = solve(P[keep], uv[keep], w)
            ep = reproj_err(P[keep], Xp, uv[keep])
            rp = np.sqrt((ep ** 2).mean(0))
            better = bad & (rp < best_rms)
            best_rms[better] = rp[better]
            best_X[better] = Xp[better]
        X[bad] = best_X[bad]
        n_used[bad] = 2
        rms[bad] = best_rms[bad]
    return X, n_used, rms


def swap_check(P: np.ndarray, uv: np.ndarray, solver=None, include_leg_only: bool = True
               ) -> tuple[np.ndarray, int]:
    # Try "no swap" and, per view, full L/R swap (and legs-only swap); keep lowest total reprojection error
    # over bilateral keypoints. Returns corrected uv and hypothesis index (0 = none).
    solve = solver or (lambda PP, uu: dlt(PP, uu))
    V = uv.shape[0]
    hyps = [None]
    for v in range(V):
        hyps.append((v, lr_perm(L_PAIRS)))
        if include_leg_only:
            hyps.append((v, lr_perm(LEG_PAIRS)))
    bil = np.array(sorted({i for pr in L_PAIRS for i in pr}))
    best, best_cost, best_uv = 0, np.inf, uv
    for h_idx, h in enumerate(hyps):
        cand = uv.copy()
        if h is not None:
            cand[h[0]] = uv[h[0]][h[1]]
        X = solve(P, cand[:, bil])
        cost = reproj_err(P, X, cand[:, bil]).sum()
        if cost < best_cost * 0.999:
            best, best_cost, best_uv = h_idx, cost, cand
    return best_uv, best
