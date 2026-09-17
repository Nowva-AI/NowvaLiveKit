"""Vectorised reference filters used as comparison baselines (causal, dt-aware)."""
from __future__ import annotations

import math

import numpy as np


def ema_fixed(pts: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(pts)
    out[0] = pts[0]
    for t in range(1, len(pts)):
        out[t] = alpha * pts[t] + (1 - alpha) * out[t - 1]
    return out


def alpha_beta(pts: np.ndarray, ts: np.ndarray, alpha: float, beta: float | None = None) -> np.ndarray:
    """Steady-state constant-velocity Kalman (g-h filter). beta default = Benedict-Bordner a^2/(2-a)."""
    beta = alpha * alpha / (2 - alpha) if beta is None else beta
    out = np.empty_like(pts)
    x = pts[0].copy()
    v = np.zeros_like(x)
    out[0] = x
    for t in range(1, len(pts)):
        dt = max(ts[t] - ts[t - 1], 1e-3)
        xp = x + v * dt
        r = pts[t] - xp
        x = xp + alpha * r
        v = v + (beta / dt) * r
        out[t] = x
    return out


def one_euro(pts: np.ndarray, ts: np.ndarray, min_cutoff: float, beta: float, d_cutoff: float = 1.0) -> np.ndarray:
    def a_of(cut, dt):
        tau = 1.0 / (2 * math.pi * cut)
        return 1.0 / (1.0 + tau / dt)
    out = np.empty_like(pts)
    x = pts[0].copy()
    dx = np.zeros_like(x)
    out[0] = x
    for t in range(1, len(pts)):
        dt = max(ts[t] - ts[t - 1], 1e-3)
        raw_dx = (pts[t] - x) / dt
        ad = a_of(d_cutoff, dt)
        dx = ad * raw_dx + (1 - ad) * dx
        cut = min_cutoff + beta * np.abs(dx)
        tau = 1.0 / (2 * np.pi * cut)
        a = 1.0 / (1.0 + tau / dt)
        x = a * pts[t] + (1 - a) * x
        out[t] = x
    return out
