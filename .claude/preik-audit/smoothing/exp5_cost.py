"""E5: per-frame compute cost of the current smoothing layers vs vectorised replacements (Mac, single core)."""
from __future__ import annotations

import time

import numpy as np

import harness as H
import smoothers as sm
from biomechanics.utils.confidence_blend import ConfidenceBlender
from biomechanics.utils.derivatives import DerivativeTracker
from biomechanics.utils.filters import JointAngleFilter
from biomechanics.utils.position_filter import KeypointPositionSmoother
from biomechanics.utils.predictive_state import PredictiveStateEstimator
from biomechanics.utils.types import Skeleton3D

d = H.make_data(0, 4.0)
X, C, TS = d["noisy"], d["conf"], d["ts"]
T = len(X)
REPEAT = 3


def bench(label, fn):
    best = 1e9
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    print(f"{label:62s} {best / T * 1e6:9.1f} us/frame")
    return best / T * 1e6


skels = [Skeleton3D.from_numpy(X[i], confidences=list(C[i]), timestamp=TS[i], frame_index=i) for i in range(T)]


def repo_pos():
    f = KeypointPositionSmoother(0.8, 4.0, 1.0)
    for sk in skels:
        f.smooth(sk)


def roundtrip():
    for sk in skels:
        Skeleton3D.from_numpy(sk.to_numpy(), confidences=[kp.confidence for kp in sk.keypoints],
                              timestamp=sk.timestamp, frame_index=sk.frame_index)


def vec_oe():
    f = sm.OneEuroVec(0.8, 4.0, 1.0, "filtered")
    for i in range(T):
        f.step(X[i], TS[i])


def repo_blend():
    b = ConfidenceBlender()
    for sk in skels:
        b.blend(sk)


def kf(order, lag):
    def run():
        f = sm.KalmanVec(order, 1.0, 0.02, lag, None)
        for i in range(T):
            f.step(X[i], TS[i])
    return run


class AlphaBeta:
    """Steady-state CV Kalman (constant gains) — what the edge implementation should be at fixed 30 Hz."""

    def __init__(self, q=1.0, r=0.02, dt=1 / 30):
        kfv = sm.KalmanVec(2, q, r, 0, None)
        z = np.zeros((1, 3))
        for i in range(400):
            kfv.step(z, i * dt)
        P = kfv.P[0, 0]
        F, Q = sm._model(2, dt, q)
        Pp = F @ P @ F.T + Q
        self.k_pos = Pp[0, 0] / (Pp[0, 0] + r * r)
        self.k_vel = Pp[1, 0] / (Pp[0, 0] + r * r)
        self.dt = dt
        self.p = None

    def step(self, z):
        if self.p is None:
            self.p = z.copy(); self.v = np.zeros_like(z)
            return self.p
        pp = self.p + self.v * self.dt
        innov = z - pp
        self.p = pp + self.k_pos * innov
        self.v = self.v + self.k_vel * innov
        return self.p


def ab():
    f = AlphaBeta()
    for i in range(T):
        f.step(X[i])


def repo_angles():
    jaf = JointAngleFilter(1.0, 0.007)
    der = DerivativeTracker(0.3)
    pr = PredictiveStateEstimator()
    for i in range(T):
        ja = H._joint_angles(X[i], TS[i], i)
        jaf.update_phase("descending")
        fa = jaf.filter_angles(ja)
        dv = der.update(fa)
        pr.predict(fa, dv)


def angles_only():
    for i in range(T):
        H._joint_angles(X[i], TS[i], i)


print("frames", T)
bench("Skeleton3D to_numpy+from_numpy round trip", roundtrip)
bench("repo ConfidenceBlender.blend (incl. round trip)", repo_blend)
bench("repo KeypointPositionSmoother.smooth (57 OneEuro objs, round trip)", repo_pos)
bench("vectorised OneEuroVec.step (19x3 numpy)", vec_oe)
bench("vectorised CV Kalman step (lag 0, full covariance)", kf(2, 0))
bench("vectorised CV Kalman fixed-lag 2", kf(2, 2))
bench("vectorised CV Kalman fixed-lag 3", kf(2, 3))
bench("vectorised CA Kalman lag 0", kf(3, 0))
bench("steady-state alpha-beta (CV KF constant gain) step", ab)
t_ang = bench("JointAngles construction only (harness vectorised IK)", angles_only)
t_all = bench("repo JointAngleFilter+DerivativeTracker+Predictive (+JointAngles)", repo_angles)
print(f"{'  => JointAngleFilter+Derivative+Predictive alone':62s} {t_all - t_ang:9.1f} us/frame")
ab_obj = AlphaBeta()
print("alpha-beta gains (q=1, r=0.02, 30 Hz): k_pos %.4f k_vel %.4f 1/s" % (ab_obj.k_pos, ab_obj.k_vel))
