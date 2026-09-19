"""Prototype causal outlier gate for triangulated keypoints: alpha-beta prediction, reject-and-predict, re-acquire."""
from __future__ import annotations
import numpy as np
from synth_vc import Skeleton3D, CK

HIP_L, HIP_R = CK.LEFT_HIP, CK.RIGHT_HIP


class InnovationGate:
    def __init__(self, gate_noise_m: float = 0.06, accel_max_m_s2: float = 30.0, conf_reject: float = 0.15,
                 alpha: float = 0.6, beta: float = 0.2, max_hold_s: float = 0.2, reacquire_n: int = 2,
                 max_dt_s: float = 0.25, common_mode: bool = True):
        self.r0, self.amax, self.conf_reject = gate_noise_m, accel_max_m_s2, conf_reject
        self.alpha, self.beta, self.max_hold, self.reacq_n, self.max_dt = alpha, beta, max_hold_s, reacquire_n, max_dt_s
        self.common_mode = common_mode
        self.reset()

    def reset(self):
        self.p = None; self.v = None; self.t = None
        self.init = None; self.tau = None; self.cand = None; self.cand_n = None; self.last_conf = None
        self.stats = dict(rejected=0, reacquired=0, timeouts=0, common_mode=0)

    def update(self, sk: Skeleton3D) -> tuple[Skeleton3D, np.ndarray]:
        z = sk.to_numpy(); c = np.array([kp.confidence for kp in sk.keypoints]); n = len(z)
        if self.p is None:
            self.p = z.copy(); self.v = np.zeros_like(z); self.t = sk.timestamp
            self.init = c >= self.conf_reject; self.tau = np.zeros(n); self.cand = z.copy(); self.cand_n = np.zeros(n, int)
            self.last_conf = c.copy()
            return sk, np.zeros(n, bool)
        dt = sk.timestamp - self.t
        if dt <= 0:                      # duplicate / clock error: re-emit last output, change nothing
            return Skeleton3D.from_numpy(self.p, confidences=self.last_conf, timestamp=sk.timestamp, frame_index=sk.frame_index), np.zeros(n, bool)
        dt = min(dt, self.max_dt)
        self.t = sk.timestamp
        pred = self.p + self.v * dt
        self.tau += dt
        innov = z - pred
        valid = c >= self.conf_reject
        # common-mode (hip-centre) error: most well-tracked non-hip points jump by the same vector
        if self.common_mode:
            mask = valid & self.init
            mask[[HIP_L, HIP_R]] = False
            if mask.sum() >= 6:
                cm = np.median(innov[mask], axis=0)
                if np.linalg.norm(cm) > self.r0:
                    z = z - cm; innov = z - pred; self.stats["common_mode"] += 1
        radius = self.r0 + 0.5 * self.amax * self.tau ** 2
        dist = np.linalg.norm(innov, axis=1)
        accept = valid & (~self.init | (dist <= radius))
        reject = valid & self.init & ~accept
        # re-acquire: consecutive rejected measurements that agree with each other
        same = reject & (np.linalg.norm(z - (self.cand + self.v * dt), axis=1) <= self.r0 * 1.5)
        self.cand_n = np.where(same, self.cand_n + 1, np.where(reject, 1, 0))
        self.cand = np.where(reject[:, None], z, self.cand)
        reacq = reject & (self.cand_n >= self.reacq_n)
        timeout = (valid | self.init) & ~accept & (self.tau > self.max_hold)
        take = accept | reacq | (timeout & valid)
        self.stats["rejected"] += int((reject & ~reacq).sum()); self.stats["reacquired"] += int(reacq.sum()); self.stats["timeouts"] += int((timeout & valid & ~reacq).sum())
        out = pred.copy()
        # accepted: emit the raw measurement (no smoothing / no lag); state via alpha-beta
        resid = z - pred
        new_p = np.where(take[:, None], pred + self.alpha * resid, pred)
        new_v = np.where(take[:, None], self.v + self.beta * resid / dt, self.v * 0.7)
        snap = (reacq | (timeout & valid)) | (~self.init & valid)
        new_p = np.where(snap[:, None], z, new_p)
        new_v = np.where(snap[:, None], 0.0, new_v)
        out = np.where(take[:, None], z, out)
        # untracked (never valid) keypoints pass through untouched
        out = np.where((~self.init & ~valid)[:, None], z, out)
        self.p, self.v = new_p, new_v
        self.tau = np.where(take, 0.0, self.tau)
        self.init = self.init | valid
        conf_out = np.where(take, c, np.maximum(self.last_conf * 0.8, 0.0))
        self.last_conf = np.where(take, c, self.last_conf)
        predicted = ~take & self.init
        return Skeleton3D.from_numpy(out, confidences=conf_out, timestamp=sk.timestamp, frame_index=sk.frame_index), predicted
