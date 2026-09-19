"""Vectorised candidate smoothers operating on (N,3) keypoint arrays (N=19), causal step() API.

OneEuroVec     : identical maths to repo OneEuroFilter (derivative from raw prev) or paper variant.
BlendVec       : repo ConfidenceBlender maths.
KalmanVec      : constant-velocity (order=2) / constant-acceleration (order=3) per scalar, optional
                 confidence-scaled measurement noise, optional fixed-lag RTS output (lag frames).
rts_offline    : full forward-backward RTS over a buffer (between-rep diagnosis).
BoneDirSmoother: smooth unit bone directions + fixed calibrated lengths, rebuild from hip-mid root.
"""
from __future__ import annotations

import math
import sys
from collections import deque

import numpy as np

sys.path.insert(0, "/Users/naiahoard/NowvaLiveKit/src")

from biomechanics.utils.types import CocoKeypoints as CK  # noqa: E402


def _alpha(cutoff, dt):
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroVec:
    def __init__(self, min_cutoff=0.8, beta=4.0, d_cutoff=1.0, deriv_from="raw"):
        self.min_cutoff, self.beta, self.d_cutoff, self.deriv_from = min_cutoff, beta, d_cutoff, deriv_from
        self.reset()

    def reset(self):
        self.x_hat = None
        self.last_raw = None
        self.dx_hat = None
        self.t = None
        self.last_cutoff = None

    def step(self, x, t, conf=None):
        if self.x_hat is None:
            self.x_hat = x.copy(); self.last_raw = x.copy(); self.dx_hat = np.zeros_like(x); self.t = t
            self.last_cutoff = np.full_like(x, self.min_cutoff)
            return x.copy()
        dt = t - self.t
        if dt <= 0:
            dt = 1e-6
        prev = self.last_raw if self.deriv_from == "raw" else self.x_hat
        dx = (x - prev) / dt
        ad = _alpha(self.d_cutoff, dt)
        self.dx_hat = ad * dx + (1 - ad) * self.dx_hat
        cutoff = self.min_cutoff + self.beta * np.abs(self.dx_hat)
        tau = 1.0 / (2.0 * np.pi * cutoff)
        a = 1.0 / (1.0 + tau / dt)
        self.x_hat = a * x + (1 - a) * self.x_hat
        self.last_raw = x.copy()
        self.t = t
        self.last_cutoff = cutoff
        return self.x_hat.copy()


class BlendVec:
    def __init__(self, min_conf=0.1, max_conf=0.9):
        self.lo, self.hi = min_conf, max_conf
        self.prev = None

    def reset(self):
        self.prev = None

    def step(self, x, t, conf):
        if self.prev is None:
            self.prev = x.copy()
            return x.copy()
        w = np.clip((conf - self.lo) / (self.hi - self.lo), 0, 1)[:, None]
        self.prev = w * x + (1 - w) * self.prev
        return self.prev.copy()


def _model(order, dt, q):
    if order == 2:
        F = np.array([[1, dt], [0, 1.0]])
        Q = q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
    else:
        F = np.array([[1, dt, dt * dt / 2], [0, 1, dt], [0, 0, 1.0]])
        Q = q * np.array([[dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6],
                          [dt ** 4 / 8, dt ** 3 / 3, dt ** 2 / 2],
                          [dt ** 3 / 6, dt ** 2 / 2, dt]])
    return F, Q


class KalmanVec:
    """q: white accel (order 2, m^2/s^3) or white jerk (order 3, m^2/s^5) spectral density.
    r_m: measurement std (m). conf_ref: if set, R scaled by (conf_ref/conf)^2 (clipped)."""

    def __init__(self, order=2, q=10.0, r_m=0.01, lag=0, conf_ref=None, z_scale=1.0):
        self.order, self.q, self.r_m, self.lag, self.conf_ref, self.z_scale = order, q, r_m, lag, conf_ref, z_scale
        self.reset()

    def reset(self):
        self.x = None
        self.P = None
        self.t = None
        self.hist = deque(maxlen=self.lag + 1)

    def step(self, z, t, conf=None):
        n = z.shape[0]
        d = self.order
        r2 = np.full((n, 3), self.r_m ** 2)
        r2[:, 2] *= self.z_scale ** 2
        if getattr(self, "reproj_gain", None) is not None and conf is not None:
            # conf carries the per-keypoint mean reprojection error (px) in this mode
            rr = np.clip(self.reproj_gain * conf, self.r_m * 0.3, self.r_m * 8.0)
            if self.reproj_mode == "frame_median":
                rr = np.full_like(rr, np.clip(self.reproj_gain * np.median(conf), self.r_m * 0.3, self.r_m * 8.0))
            r2 = np.repeat((rr ** 2)[:, None], 3, axis=1)
            conf = None
        if self.conf_ref is not None and conf is not None:
            scale = np.clip(self.conf_ref / np.maximum(conf, 1e-3), 0.5, 10.0) ** 2
            r2 = r2 * scale[:, None]
        if self.x is None:
            self.x = np.zeros((n, 3, d)); self.x[..., 0] = z
            self.P = np.zeros((n, 3, d, d))
            self.P[..., 0, 0] = r2
            for i in range(1, d):
                self.P[..., i, i] = 1.0 if i == 1 else 10.0
            self.t = t
            self.hist.append((self.x.copy(), self.P.copy(), self.x.copy(), self.P.copy(), np.eye(d)))
            return z.copy()
        dt = t - self.t
        if dt <= 0:
            # duplicate frame: no new information, return current estimate
            return self._output()
        F, Q = _model(d, dt, self.q)
        xp = self.x @ F.T
        Pp = F @ self.P @ F.T + Q
        S = Pp[..., 0, 0] + r2
        K = Pp[..., :, 0] / S[..., None]
        innov = z - xp[..., 0]
        self.x = xp + K * innov[..., None]
        self.P = Pp - K[..., :, None] * Pp[..., 0:1, :]
        self.t = t
        self.hist.append((xp, Pp, self.x.copy(), self.P.copy(), F))
        return self._output()

    def _output(self):
        if self.lag == 0 or len(self.hist) < 2:
            return self.x[..., 0].copy()
        # RTS backward over the window, return oldest smoothed state
        items = list(self.hist)
        xs, Ps = items[-1][2], items[-1][3]
        for k in range(len(items) - 2, -1, -1):
            xf, Pf = items[k][2], items[k][3]
            xp_next, Pp_next, F_next = items[k + 1][0], items[k + 1][1], items[k + 1][4]
            C = Pf @ F_next.T @ np.linalg.inv(Pp_next)
            xs = xf + ((xs - xp_next)[..., None, :] @ np.swapaxes(C, -1, -2))[..., 0, :]
            Ps = Pf + C @ (Ps - Pp_next) @ np.swapaxes(C, -1, -2)
        return xs[..., 0].copy()


def rts_offline(Z, ts, order=2, q=10.0, r_m=0.01, conf=None, conf_ref=None):
    kf = KalmanVec(order=order, q=q, r_m=r_m, conf_ref=conf_ref)
    kf.hist = deque()  # unbounded
    for i in range(len(Z)):
        kf.step(Z[i], ts[i], None if conf is None else conf[i])
    items = list(kf.hist)
    T = len(items)
    out = np.zeros_like(Z)
    xs, Ps = items[-1][2], items[-1][3]
    out[-1] = xs[..., 0]
    for k in range(T - 2, -1, -1):
        xf, Pf = items[k][2], items[k][3]
        xp_next, Pp_next, F_next = items[k + 1][0], items[k + 1][1], items[k + 1][4]
        C = Pf @ F_next.T @ np.linalg.inv(Pp_next)
        xs = xf + ((xs - xp_next)[..., None, :] @ np.swapaxes(C, -1, -2))[..., 0, :]
        Ps = Pf + C @ (Ps - Pp_next) @ np.swapaxes(C, -1, -2)
        out[k] = xs[..., 0]
    return out


# parent, child  (root = hip midpoint, index -1)
BONES = [
    (-1, CK.LEFT_HIP), (-1, CK.RIGHT_HIP),
    (CK.LEFT_HIP, CK.LEFT_KNEE), (CK.RIGHT_HIP, CK.RIGHT_KNEE),
    (CK.LEFT_KNEE, CK.LEFT_ANKLE), (CK.RIGHT_KNEE, CK.RIGHT_ANKLE),
    (CK.LEFT_ANKLE, CK.LEFT_FOOT_INDEX), (CK.RIGHT_ANKLE, CK.RIGHT_FOOT_INDEX),
    (CK.LEFT_HIP, CK.LEFT_SHOULDER), (CK.RIGHT_HIP, CK.RIGHT_SHOULDER),
    (CK.LEFT_SHOULDER, CK.LEFT_ELBOW), (CK.RIGHT_SHOULDER, CK.RIGHT_ELBOW),
    (CK.LEFT_ELBOW, CK.LEFT_WRIST), (CK.RIGHT_ELBOW, CK.RIGHT_WRIST),
    (CK.LEFT_SHOULDER, CK.NOSE), (CK.NOSE, CK.LEFT_EYE), (CK.NOSE, CK.RIGHT_EYE),
    (CK.NOSE, CK.LEFT_EAR), (CK.NOSE, CK.RIGHT_EAR),
]


def bone_vectors(k):
    root = (k[..., CK.LEFT_HIP, :] + k[..., CK.RIGHT_HIP, :]) / 2
    out = []
    for p, c in BONES:
        parent = root if p == -1 else k[..., p, :]
        out.append(k[..., c, :] - parent)
    return np.stack(out, axis=-2)


def rebuild(vecs, n=19):
    k = np.zeros(vecs.shape[:-2] + (n, 3))
    root = np.zeros(vecs.shape[:-2] + (3,))
    for i, (p, c) in enumerate(BONES):
        parent = root if p == -1 else k[..., p, :]
        k[..., c, :] = parent + vecs[..., i, :]
    return k


class OneEuroShared(OneEuroVec):
    """One Euro whose speed term is shared across the skeleton: mean |velocity| of the leg joints (averages
    out per-joint derivative noise), one cutoff for every joint."""

    SPEED_IDX = [11, 12, 13, 14, 15, 16]

    def step(self, x, t, conf=None):
        if self.x_hat is None:
            return super().step(x, t, conf)
        dt = t - self.t
        if dt <= 0:
            return self.x_hat.copy()
        prev = self.last_raw if self.deriv_from == "raw" else self.x_hat
        dx = (x - prev) / dt
        ad = _alpha(self.d_cutoff, dt)
        self.dx_hat = ad * dx + (1 - ad) * self.dx_hat
        speed = float(np.mean(np.linalg.norm(self.dx_hat[self.SPEED_IDX], axis=-1)))
        cutoff = self.min_cutoff + self.beta * speed
        a = _alpha(cutoff, dt)
        self.x_hat = a * x + (1 - a) * self.x_hat
        self.last_raw = x.copy()
        self.t = t
        self.last_cutoff = np.full_like(x, cutoff)
        return self.x_hat.copy()


class BoneDirSmoother:
    """Filter unit bone directions with an inner vector filter, re-apply calibrated lengths."""

    def __init__(self, inner, calib_frames=30):
        self.inner = inner
        self.calib_frames = calib_frames
        self.reset()

    def reset(self):
        self.inner.reset()
        self.len_buf = []
        self.lengths = None

    def step(self, x, t, conf=None):
        v = bone_vectors(x)
        lens = np.linalg.norm(v, axis=-1)
        if self.lengths is None:
            self.len_buf.append(lens)
            if len(self.len_buf) >= self.calib_frames:
                self.lengths = np.median(np.stack(self.len_buf), axis=0)
        u = v / np.maximum(lens[:, None], 1e-9)
        u_s = self.inner.step(u, t, None)
        u_s = u_s / np.maximum(np.linalg.norm(u_s, axis=-1, keepdims=True), 1e-9)
        use_len = self.lengths if self.lengths is not None else np.median(np.stack(self.len_buf), axis=0)
        return rebuild(u_s * use_len[:, None])


def run_causal(filt, Z, ts, conf=None):
    filt.reset()
    out = np.empty_like(Z)
    for i in range(len(Z)):
        out[i] = filt.step(Z[i], ts[i], None if conf is None else conf[i])
    return out
