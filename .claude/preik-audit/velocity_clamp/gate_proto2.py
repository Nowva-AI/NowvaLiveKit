"""Prototype v2: per-keypoint causal outlier gate for triangulated keypoints.

predict  p_pred = x + v*dt   (alpha-beta state, high gains -> small lag)
gate     |z - p_pred| <= r_noise + 0.5*a_max*tau^2     (tau = time since last accepted measurement)
accept   emit z unchanged (no smoothing, no lag); update alpha-beta state
reject   emit p_pred; conf unchanged-but-flagged (predicted mask); velocity kept for 2 frames then damped
reacq    2 consecutive rejected measurements consistent with each other (|z_k - z_{k-1} - v dt| <= r_cand) -> snap
timeout  tau > max_hold -> snap to measurement
low conf (triangulator conf < conf_reject, i.e. reproj >= 15 px) is treated as a rejected measurement
dt <= 0  (duplicate / clock error) -> re-emit previous output, no state change
"""
from __future__ import annotations
import numpy as np
from synth_vc import Skeleton3D, CK

HIP_L, HIP_R = CK.LEFT_HIP, CK.RIGHT_HIP


class InnovationGate2:
    def __init__(self, r_noise_m: float = 0.08, a_max_m_s2: float = 40.0, conf_reject: float = 0.15,
                 alpha: float = 0.85, beta: float = 0.5, max_hold_s: float = 0.2, r_cand_m: float = 0.10,
                 max_dt_s: float = 0.25, hip_centred_common_mode: bool = False):
        self.r0, self.amax, self.cmin = r_noise_m, a_max_m_s2, conf_reject
        self.alpha, self.beta, self.hold, self.rc, self.max_dt = alpha, beta, max_hold_s, r_cand_m, max_dt_s
        self.cm = hip_centred_common_mode
        self.reset()

    def reset(self):
        self.x = None
        self.stats = dict(rejected=0, reacquired=0, timeouts=0, common_mode=0, frames=0)

    def update(self, z: np.ndarray, c: np.ndarray, ts: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(z)
        self.stats["frames"] += 1
        valid = c >= self.cmin
        if self.x is None:
            self.x = z.copy(); self.v = np.zeros_like(z); self.t = ts; self.tau = np.zeros(n)
            self.init = valid.copy(); self.prev_z = z.copy(); self.cand_n = np.zeros(n, int); self.rej_n = np.zeros(n, int)
            self.out = z.copy(); self.out_c = c.copy()
            return z, c, np.zeros(n, bool)
        dt = ts - self.t
        if dt <= 0:
            return self.out, self.out_c, np.zeros(n, bool)
        dt = min(dt, self.max_dt); self.t = ts
        pred = self.x + self.v * dt
        tau = self.tau + dt
        innov = z - pred
        if self.cm:
            ok = valid & self.init; ok[[HIP_L, HIP_R]] = False
            if ok.sum() >= 6:
                cmv = np.median(innov[ok], axis=0)
                if np.linalg.norm(cmv) > self.r0:
                    hl, hr = innov[HIP_L], innov[HIP_R]
                    one_hip_with_crowd = min(np.linalg.norm(hl - cmv), np.linalg.norm(hr - cmv)) < self.r0 / 2
                    other_opposite = min(np.linalg.norm(hl + cmv), np.linalg.norm(hr + cmv)) < self.r0 / 2
                    if one_hip_with_crowd and other_opposite:
                        z = z - cmv; innov = z - pred; self.stats["common_mode"] += 1
        dist = np.linalg.norm(innov, axis=1)
        radius = self.r0 + 0.5 * self.amax * tau ** 2
        accept = valid & (~self.init | (dist <= radius))
        rejected_meas = self.init & ~accept                     # includes low-confidence measurements
        cand_ok = valid & rejected_meas & (self.rej_n > 0) & (np.linalg.norm(z - self.prev_z - self.v * dt, axis=1) <= self.rc)
        self.cand_n = np.where(cand_ok, self.cand_n + 1, np.where(valid & rejected_meas, 1, 0))
        reacq = cand_ok & (self.cand_n >= 2)
        timeout = valid & rejected_meas & (tau > self.hold) & ~reacq
        take = accept | reacq | timeout
        snap = reacq | timeout | (~self.init & valid)
        resid = z - pred
        new_x = np.where(take[:, None], pred + self.alpha * resid, pred)
        new_v = np.where(take[:, None], self.v + (self.beta / dt) * resid, np.where((self.rej_n >= 2)[:, None], 0.7 * self.v, self.v))
        new_x = np.where(snap[:, None], z, new_x)
        new_v = np.where(snap[:, None], np.where(reacq[:, None], (z - self.prev_z) / dt, 0.0), new_v)
        out = np.where(take[:, None], z, pred)
        out = np.where((~self.init & ~valid)[:, None], z, out)
        predicted = self.init & ~take
        out_c = np.where(predicted, np.maximum(self.out_c, self.cmin), c)
        self.stats["rejected"] += int((valid & rejected_meas & ~take).sum()); self.stats["reacquired"] += int(reacq.sum()); self.stats["timeouts"] += int(timeout.sum())
        self.x, self.v = new_x, new_v
        self.tau = np.where(take, 0.0, tau)
        self.rej_n = np.where(take, 0, self.rej_n + 1)
        self.init = self.init | valid
        self.prev_z = np.where(valid[:, None], z, self.prev_z)
        self.out, self.out_c = out, out_c
        return out, out_c, predicted
