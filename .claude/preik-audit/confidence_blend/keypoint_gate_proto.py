"""Prototype replacement for ConfidenceBlender's legitimate purpose: a time-aware per-keypoint validity gate.

Valid measurements pass through UNCHANGED (zero lag). Invalid ones (not triangulated, non-finite, reprojection error
above threshold) are replaced by constant-velocity extrapolation from the last valid samples for at most HOLD_MAX_S,
then reported missing (confidence 0) at the last valid position. Invalid positions never enter state.
"""
from __future__ import annotations

import numpy as np

HOLD_MAX_S = 0.10
MAX_EXTRAP_SPEED_M_PER_S = 2.0
STATE_EXPIRY_S = 0.5


class KeypointGate:
    def __init__(self, n_keypoints: int, max_reproj_px: float = 8.0, hold_max_s: float = HOLD_MAX_S):
        self.max_reproj_px = max_reproj_px
        self.hold_max_s = hold_max_s
        self._pos = np.zeros((n_keypoints, 3))
        self._vel = np.zeros((n_keypoints, 3))
        self._ts = np.full(n_keypoints, -np.inf)

    def reset(self) -> None:
        self._ts[:] = -np.inf
        self._vel[:] = 0.0

    def update(self, pts: np.ndarray, conf: np.ndarray, reproj_px: np.ndarray, ts: float) -> tuple[np.ndarray, np.ndarray]:
        valid = (conf > 0) & np.isfinite(pts).all(axis=1) & (reproj_px <= self.max_reproj_px)
        age = ts - self._ts
        age = np.where(age < 0, np.inf, age)  # clock went backwards -> treat as expired
        # state update from valid samples (duplicates: age == 0 -> keep velocity)
        fresh = valid & (age > 1e-6) & (age <= STATE_EXPIRY_S)
        new_vel = np.where(fresh[:, None], (pts - self._pos) / np.maximum(age, 1e-6)[:, None], self._vel)
        speed = np.linalg.norm(new_vel, axis=1, keepdims=True)
        new_vel = new_vel * np.minimum(1.0, MAX_EXTRAP_SPEED_M_PER_S / np.maximum(speed, 1e-9))
        new_vel = np.where((valid & (age > STATE_EXPIRY_S))[:, None], 0.0, new_vel)
        holdable = ~valid & (age <= self.hold_max_s)
        out = pts.copy()
        out_conf = np.where(valid, conf, 0.0)
        extrap = self._pos + self._vel * np.minimum(age, self.hold_max_s)[:, None]
        out[holdable] = extrap[holdable]
        out_conf[holdable] = np.maximum(conf[holdable], 0.2)  # still IK-usable while the hold is fresh
        stale = ~valid & ~holdable & np.isfinite(self._ts)
        out[stale] = self._pos[stale]  # position never (0,0,0)/NaN once seen; confidence 0 -> consumers must gate
        out[~valid & ~np.isfinite(self._ts)] = np.where(np.isfinite(pts[~valid & ~np.isfinite(self._ts)]), pts[~valid & ~np.isfinite(self._ts)], 0.0)
        self._vel = np.where(valid[:, None], new_vel, self._vel)
        self._pos = np.where(valid[:, None], pts, self._pos)
        self._ts = np.where(valid, ts, self._ts)
        return out, out_conf
