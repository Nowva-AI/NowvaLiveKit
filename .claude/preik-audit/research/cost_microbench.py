"""Micro-benchmark (Mac) of candidate pre-IK stages for 19 keypoints x 3 views; multiply ~3-5x for Jetson Orin Nano CPU."""
from __future__ import annotations

import timeit

import numpy as np
from scipy import signal

N_KPTS, N_VIEWS = 19, 3
rng = np.random.default_rng(0)
P = rng.normal(size=(N_VIEWS, 3, 4))
uv = rng.normal(size=(N_VIEWS, N_KPTS, 2))
w = rng.uniform(0.3, 1.0, size=(N_VIEWS, N_KPTS))
SUBSETS = [(0, 1, 2), (0, 1), (0, 2), (1, 2)]


def weighted_dlt_all_subsets() -> None:
    for subset in SUBSETS:
        rows = []
        for v in subset:
            rows.append(w[v, :, None] * (uv[v, :, 0, None] * P[v, 2] - P[v, 0]))
            rows.append(w[v, :, None] * (uv[v, :, 1, None] * P[v, 2] - P[v, 1]))
        A = np.stack(rows, axis=1)  # (K, 2V, 4)
        _, _, vt = np.linalg.svd(A)
        X = vt[:, -1, :3] / vt[:, -1, 3:4]
        Xh = np.hstack([X, np.ones((N_KPTS, 1))])
        proj = np.einsum("vij,nj->vni", P, Xh)
        _ = np.linalg.norm(proj[..., :2] / proj[..., 2:3] - uv, axis=-1)


DT = 1 / 30
F = np.array([[1, DT], [0, 1]])
Q = np.array([[DT**4 / 4, DT**3 / 2], [DT**3 / 2, DT**2]]) * 30.0
state = np.zeros((N_KPTS, 3, 2))
cov = np.tile(np.eye(2), (N_KPTS, 3, 1, 1))
z = rng.normal(size=(N_KPTS, 3))
R = np.full((N_KPTS, 3), 1e-4)


def kalman_cv_step() -> None:
    global state, cov
    sp = state @ F.T
    Pp = F @ cov @ F.T + Q
    S = Pp[..., 0, 0] + R
    innov = z - sp[..., 0]
    gate = (innov**2 / S > 16.0) & (np.abs(innov) > 0.04)
    K = Pp[..., :, 0] / S[..., None]
    K[gate] = 0.0
    state = sp + K * innov[..., None]
    cov = Pp - K[..., :, None] * Pp[..., 0, None, :]


sos = signal.butter(2, 6 / 15, output="sos")
rep_buffer = rng.normal(size=(120, N_KPTS, 3))


def rep_window_filtfilt() -> None:
    signal.sosfiltfilt(sos, rep_buffer, axis=0)


def floor_plane_fit() -> None:
    pts = rng.normal(size=(6 * 30, 3))
    centred = pts - pts.mean(axis=0)
    np.linalg.eigh(centred.T @ centred)


for name, fn, n in (("weighted DLT, 4 subsets + reprojection", weighted_dlt_all_subsets, 2000),
                    ("vectorised CV Kalman step + gate (19x3)", kalman_cv_step, 20000),
                    ("rep-window sosfiltfilt 120 frames x 19x3", rep_window_filtfilt, 2000),
                    ("floor plane fit (180 pts)", floor_plane_fit, 2000)):
    t = timeit.timeit(fn, number=n) / n * 1e6
    print(f"{name:45s} {t:8.1f} us/call (Mac)")
