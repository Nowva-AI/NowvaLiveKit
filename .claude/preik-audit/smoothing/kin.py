"""Vectorised copies of the production angle formulas (verified against the repo IK in check_kin.py)."""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/Users/naiahoard/NowvaLiveKit/src")

from biomechanics.utils.types import CocoKeypoints as CK  # noqa: E402


def _ang(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    c = np.sum(a * b, axis=-1) / np.maximum(na * nb, 1e-12)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def knee_flexion(k: np.ndarray, side: str) -> np.ndarray:
    h, kn, a = (CK.LEFT_HIP, CK.LEFT_KNEE, CK.LEFT_ANKLE) if side == "l" else (CK.RIGHT_HIP, CK.RIGHT_KNEE, CK.RIGHT_ANKLE)
    return 180.0 - _ang(k[..., h, :] - k[..., kn, :], k[..., a, :] - k[..., kn, :])


def hip_flexion(k: np.ndarray, side: str) -> np.ndarray:
    h, kn, s = (CK.LEFT_HIP, CK.LEFT_KNEE, CK.LEFT_SHOULDER) if side == "l" else (CK.RIGHT_HIP, CK.RIGHT_KNEE, CK.RIGHT_SHOULDER)
    return 180.0 - _ang(k[..., s, :] - k[..., h, :], k[..., kn, :] - k[..., h, :])


def trunk_flexion(k: np.ndarray) -> np.ndarray:
    sh = (k[..., CK.LEFT_SHOULDER, :] + k[..., CK.RIGHT_SHOULDER, :]) / 2
    hp = (k[..., CK.LEFT_HIP, :] + k[..., CK.RIGHT_HIP, :]) / 2
    return 180.0 - _ang(sh - hp, np.broadcast_to(np.array([0.0, -1.0, 0.0]), sh.shape))


def gs_valgus(k: np.ndarray, side: str) -> np.ndarray:
    lh, rh = k[..., CK.LEFT_HIP, :], k[..., CK.RIGHT_HIP, :]
    ml = _unit(lh - rh)
    if side == "l":
        hip, knee, ankle, medial_sign = lh, k[..., CK.LEFT_KNEE, :], k[..., CK.LEFT_ANKLE, :], -1.0
    else:
        hip, knee, ankle, medial_sign = rh, k[..., CK.RIGHT_KNEE, :], k[..., CK.RIGHT_ANKLE, :], 1.0
    fa = _unit(knee - hip)
    e_ml = _unit(ml - np.sum(ml * fa, axis=-1, keepdims=True) * fa)
    mag = np.abs(90.0 - _ang(e_ml, _unit(ankle - knee)))
    line = ankle - hip
    proj = hip + (np.sum((knee - hip) * line, -1, keepdims=True) / np.sum(line * line, -1, keepdims=True)) * line
    sign = np.where(np.sum((knee - proj) * (medial_sign * ml), -1) >= 0, 1.0, -1.0)
    return mag * sign


def rep_signal_cm(k: np.ndarray) -> np.ndarray:
    hip_y = (k[..., CK.LEFT_HIP, 1] + k[..., CK.RIGHT_HIP, 1]) / 2
    ank_y = (k[..., CK.LEFT_ANKLE, 1] + k[..., CK.RIGHT_ANKLE, 1]) / 2
    return (hip_y - ank_y) * 100.0


def bone_len(k: np.ndarray, a: int, b: int) -> np.ndarray:
    return np.linalg.norm(k[..., a, :] - k[..., b, :], axis=-1)


def all_angles(k: np.ndarray) -> dict[str, np.ndarray]:
    return dict(
        knee_l=knee_flexion(k, "l"), knee_r=knee_flexion(k, "r"),
        hip_l=hip_flexion(k, "l"), hip_r=hip_flexion(k, "r"),
        valgus_l=gs_valgus(k, "l"), valgus_r=gs_valgus(k, "r"),
        trunk=trunk_flexion(k),
    )
