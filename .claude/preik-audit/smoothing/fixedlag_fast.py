"""Edge-oriented fixed-lag CV Kalman smoother: closed-form 2x2 maths on (N,3) arrays, ring buffer, no linalg calls.
Reference design for production (verified against smoothers.KalmanVec in the __main__ block)."""
from __future__ import annotations

import numpy as np


class FixedLagCV:
    def __init__(self, q: float = 3.0, r_m: float = 0.01, lag: int = 2, n: int = 19):
        self.q, self.r2, self.lag, self.n = q, r_m * r_m, lag, n
        self.reset()

    def reset(self) -> None:
        self.t = None
        self.buf = []  # list of dicts (oldest first), length <= lag+1

    def step(self, z: np.ndarray, t: float, r2: np.ndarray | None = None) -> np.ndarray:
        r2 = self.r2 if (r2 is None or np.ndim(r2) != 2) else r2  # a 1-D conf vector is ignored
        if self.t is None:
            st = dict(p=z.copy(), v=np.zeros_like(z), P00=np.full_like(z, self.r2), P01=np.zeros_like(z),
                      P11=np.ones_like(z), dt=0.0)
            self.buf = [st]
            self.t = t
            return z.copy()
        dt = t - self.t
        if dt <= 0:
            return self._smoothed()
        prev = self.buf[-1]
        q = self.q
        pp = prev["p"] + dt * prev["v"]
        vp = prev["v"]
        Pp00 = prev["P00"] + 2 * dt * prev["P01"] + dt * dt * prev["P11"] + q * dt ** 3 / 3
        Pp01 = prev["P01"] + dt * prev["P11"] + q * dt * dt / 2
        Pp11 = prev["P11"] + q * dt
        s = Pp00 + r2
        k0, k1 = Pp00 / s, Pp01 / s
        innov = z - pp
        st = dict(p=pp + k0 * innov, v=vp + k1 * innov, P00=(1 - k0) * Pp00, P01=(1 - k0) * Pp01,
                  P11=Pp11 - k1 * Pp01, pp=pp, vp=vp, Pp00=Pp00, Pp01=Pp01, Pp11=Pp11, dt=dt)
        self.buf.append(st)
        if len(self.buf) > self.lag + 1:
            self.buf.pop(0)
        self.t = t
        return self._smoothed()

    def _smoothed(self) -> np.ndarray:
        if self.lag == 0 or len(self.buf) < 2:
            return self.buf[-1]["p"].copy()
        xs_p, xs_v = self.buf[-1]["p"], self.buf[-1]["v"]
        for k in range(len(self.buf) - 2, -1, -1):
            f, nx = self.buf[k], self.buf[k + 1]
            dt = nx["dt"]
            # C = Pf F^T inv(Pp_next)
            a00, a01 = f["P00"] + dt * f["P01"], f["P01"]
            a10, a11 = f["P01"] + dt * f["P11"], f["P11"]
            det = nx["Pp00"] * nx["Pp11"] - nx["Pp01"] ** 2
            i00, i01, i11 = nx["Pp11"] / det, -nx["Pp01"] / det, nx["Pp00"] / det
            c00, c01 = a00 * i00 + a01 * i01, a00 * i01 + a01 * i11
            c10, c11 = a10 * i00 + a11 * i01, a10 * i01 + a11 * i11
            dp, dv = xs_p - nx["pp"], xs_v - nx["vp"]
            xs_p, xs_v = f["p"] + c00 * dp + c01 * dv, f["v"] + c10 * dp + c11 * dv
        return xs_p.copy()


class ProposedSmoother(FixedLagCV):
    """Recommended pre-IK smoother: fixed-lag CV Kalman, R from the frame-median triangulation reprojection
    error (noise-adaptive), conf-0 keypoints predicted not updated, innovation gate for outliers."""

    wants_reproj = True

    def __init__(self, q: float = 10.0, lag: int = 2, gain_m_per_px: float = 0.005, r_floor_m: float = 0.003,
                 r_ceil_m: float = 0.08, gate_sigma: float = 6.0):
        super().__init__(q=q, r_m=0.01, lag=lag)
        self.gain, self.r_floor, self.r_ceil, self.gate = gain_m_per_px, r_floor_m, r_ceil_m, gate_sigma

    def step(self, z, t, side=None):
        if side is None:
            return super().step(z, t)
        conf, reproj = side
        valid = conf > 0
        r = float(np.clip(self.gain * np.median(reproj[valid]) if valid.any() else self.r_ceil, self.r_floor, self.r_ceil))
        r2 = np.full(z.shape, r * r)
        r2[~valid] = 1e6
        if self.buf and self.gate:
            prev = self.buf[-1]
            dt = max(t - self.t, 1e-3) if self.t is not None else 1 / 30
            pp = prev["p"] + dt * prev["v"]
            s = prev["P00"] + 2 * dt * prev["P01"] + dt * dt * prev["P11"] + self.q * dt ** 3 / 3 + r2
            bad = np.abs(z - pp) > self.gate * np.sqrt(s)
            r2 = np.where(bad.any(axis=1, keepdims=True), 1e6, r2)
        return super().step(z, t, r2)


if __name__ == "__main__":
    import time

    import harness as H
    import smoothers as sm

    d = H.make_data(0, 4.0)
    for lag in (0, 1, 2, 3):
        a = sm.run_causal(sm.KalmanVec(2, 3.0, 0.01, lag), d["noisy"], d["ts"])
        f = FixedLagCV(3.0, 0.01, lag)
        t0 = time.perf_counter()
        b = np.stack([f.step(d["noisy"][i], d["ts"][i]) for i in range(len(d["noisy"]))])
        el = (time.perf_counter() - t0) / len(d["noisy"]) * 1e6
        print(f"lag {lag}: max |fast - reference| = {np.abs(a[5:] - b[5:]).max():.2e} m, {el:.1f} us/frame")
