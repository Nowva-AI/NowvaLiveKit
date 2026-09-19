"""Replicates pipeline.process_frame temporal stages with the REAL classes, instrumented for dt and outputs."""
from __future__ import annotations
import sys, time
sys.path.insert(0, ".")
import numpy as np
from synth_vc import *
from biomechanics.utils.confidence_blend import ConfidenceBlender
from biomechanics.utils.velocity_clamp import VelocityClamp
from biomechanics.utils.filters import JointAngleFilter
from biomechanics.utils.derivatives import DerivativeTracker
from biomechanics.utils.predictive_state import PredictiveStateEstimator
from biomechanics.kinematics.analytical_ik import AnalyticalIKSolver

PERF_BASE = 412_345.0          # time.perf_counter() on macOS/Linux ~ seconds since boot
WALL_BASE = 1_789_560_000.0    # time.time() on 2026-09-16
MAX_HOLD = 5


class Chain:
    def __init__(self, phase: str = "descending", use_blend: bool = True, use_clamp: bool = True, vmax: float = 2.5):
        self.blender = ConfidenceBlender(0.1, 0.9)
        self.clamp = VelocityClamp(vmax, 30)
        self.ik = AnalyticalIKSolver()
        self.angle_filter = JointAngleFilter(min_cutoff=1.0, beta=0.007)
        self.angle_filter.update_phase(phase)
        self.deriv = DerivativeTracker(0.3)
        self.pred = PredictiveStateEstimator(0.2, 15.0)
        self.use_blend, self.use_clamp = use_blend, use_clamp
        self.last_valid = None
        self.hold = 0

    def step(self, sk3d: Skeleton3D | None, wall_now: float, hold_clock: str = "wall") -> dict | None:
        rec = {}
        if sk3d is None:
            if self.last_valid is None or self.hold >= MAX_HOLD:
                return None
            self.hold += 1
            decay = 1.0 - self.hold / MAX_HOLD
            held = self.last_valid
            ts = wall_now if hold_clock == "wall" else hold_clock_value(wall_now)
            sk3d = Skeleton3D.from_numpy(held.to_numpy(), confidences=[kp.confidence * decay for kp in held.keypoints],
                                         timestamp=ts, frame_index=0)
            rec["held"] = True
        else:
            self.hold = 0
            self.last_valid = sk3d
            rec["held"] = False
        rec["ts"] = sk3d.timestamp
        raw = sk3d.to_numpy()
        if self.use_blend:
            sk3d = self.blender.blend(sk3d)
        pre = sk3d.to_numpy()
        prev_ts = self.clamp._prev_timestamp
        if self.use_clamp:
            rec["clamp_dt"] = None if prev_ts is None else (sk3d.timestamp - prev_ts if sk3d.timestamp - prev_ts > 1e-6 else 1 / 30)
            rec["clamp_raw_dt"] = None if prev_ts is None else sk3d.timestamp - prev_ts
            sk3d = self.clamp.clamp(sk3d)
        post = sk3d.to_numpy()
        rec["clamped_n"] = int((np.linalg.norm(post - pre, axis=1) > 1e-9).sum())
        rec["pos"] = post
        angles = self.ik.solve(sk3d)
        f = self.angle_filter._filters.get("knee_flexion_l")
        rec["oneeuro_dt"] = None if f is None or f.last_time is None else sk3d.timestamp - f.last_time
        rec["raw_knee"] = angles.knee_flexion_l
        fa = self.angle_filter.filter_angles(angles)
        rec["filt_knee"] = fa.knee_flexion_l
        rec["deriv_dt"] = None if self.deriv._prev_timestamp is None else fa.timestamp - self.deriv._prev_timestamp
        d = self.deriv.update(fa)
        rec["knee_vel"] = d.knee_velocity_l
        rec["knee_acc"] = d.knee_acceleration_l
        p = self.pred.predict(fa, d)
        rec["pred_knee"] = p.knee_flexion_l
        return rec


def hold_clock_value(wall_now: float) -> float:
    return wall_now - WALL_BASE + PERF_BASE


def run(skeletons: list, times: np.ndarray, hold_clock: str = "wall", **chain_kw) -> list:
    ch = Chain(**chain_kw)
    out = []
    for sk, t in zip(skeletons, times):
        out.append(ch.step(sk, WALL_BASE + t, hold_clock=hold_clock))
    return out
