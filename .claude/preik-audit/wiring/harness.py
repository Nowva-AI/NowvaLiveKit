"""Shared harness: BiomechanicsPipeline driven by a fake triangulated provider and a fake clock.

No repo files are modified. The multi-camera provider is replaced with a generator of
synthetic Y-down, hip-centred 21-keypoint skeletons (same layout the DLT triangulator emits).
"""

from __future__ import annotations

import math
import os
import sys
import types
from unittest.mock import patch

import numpy as np

sys.path.insert(0, "/Users/naiahoard/NowvaLiveKit/src")

FPS = 30.0
FEMUR_M = 0.45
TIBIA_M = 0.43
TORSO_M = 0.52
HIP_HALF_WIDTH_M = 0.12


class FakeClock:
    """Sim clock: time.time() is epoch-like, perf_counter() is small (like the real clocks)."""

    def __init__(self) -> None:
        self.sim_t = 0.0

    def advance(self, seconds: float) -> None:
        self.sim_t += seconds

    def time(self) -> float:
        return 1.7e9 + self.sim_t

    def perf_counter(self) -> float:
        return 5000.0 + self.sim_t


def squat_points(depth_ratio: float, valgus_m: float = 0.0, lean_extra_deg: float = 0.0) -> np.ndarray:
    """21-kpt skeleton, Y-down, hip midpoint at origin. depth_ratio 0=standing, 1=bottom."""
    from biomechanics.utils.types import CocoKeypoints as CK

    shank_deg = 35.0 * depth_ratio
    thigh_deg = 85.0 * depth_ratio
    trunk_deg = 40.0 * depth_ratio + lean_extra_deg
    s, t, k = map(math.radians, (shank_deg, thigh_deg, trunk_deg))
    # floor frame: ankle at y=0 (Y-down => up is negative y), z forward
    knee = np.array([0.0, -TIBIA_M * math.cos(s), TIBIA_M * math.sin(s)])
    hip = knee + np.array([0.0, -FEMUR_M * math.cos(t), -FEMUR_M * math.sin(t)])
    pts = np.zeros((21, 3))
    for sign, (hip_i, knee_i, ankle_i, sh_i, el_i, wr_i, toe_i, heel_i) in (
        (1.0, (CK.LEFT_HIP, CK.LEFT_KNEE, CK.LEFT_ANKLE, CK.LEFT_SHOULDER, CK.LEFT_ELBOW, CK.LEFT_WRIST, CK.LEFT_FOOT_INDEX, CK.LEFT_HEEL)),
        (-1.0, (CK.RIGHT_HIP, CK.RIGHT_KNEE, CK.RIGHT_ANKLE, CK.RIGHT_SHOULDER, CK.RIGHT_ELBOW, CK.RIGHT_WRIST, CK.RIGHT_FOOT_INDEX, CK.RIGHT_HEEL)),
    ):
        x = sign * HIP_HALF_WIDTH_M
        pts[ankle_i] = [x, 0.0, 0.0]
        pts[knee_i] = [x - sign * valgus_m * depth_ratio, knee[1], knee[2]]
        pts[hip_i] = [x, hip[1], hip[2]]
        shoulder = hip + np.array([0.0, -TORSO_M * math.cos(k), TORSO_M * math.sin(k)])
        pts[sh_i] = [sign * 0.19, shoulder[1], shoulder[2]]
        pts[el_i] = [sign * 0.24, shoulder[1] + 0.25, shoulder[2] + 0.05]
        pts[wr_i] = [sign * 0.24, shoulder[1] + 0.48, shoulder[2] + 0.10]
        pts[toe_i] = [x + sign * 0.03, 0.06, 0.16]
        pts[heel_i] = [x, 0.0, -0.05]
    head = (pts[CK.LEFT_SHOULDER] + pts[CK.RIGHT_SHOULDER]) / 2.0 + np.array([0.0, -0.22, 0.03])
    pts[CK.NOSE] = head
    pts[CK.LEFT_EYE] = head + [0.03, -0.02, 0.0]
    pts[CK.RIGHT_EYE] = head + [-0.03, -0.02, 0.0]
    pts[CK.LEFT_EAR] = head + [0.07, 0.0, -0.05]
    pts[CK.RIGHT_EAR] = head + [-0.07, 0.0, -0.05]
    hip_mid = (pts[CK.LEFT_HIP] + pts[CK.RIGHT_HIP]) / 2.0
    return pts - hip_mid


def rep_depth_profile(stand_s: float = 1.0, down_s: float = 1.0, hold_s: float = 0.4, up_s: float = 1.0) -> list[float]:
    frames: list[float] = []
    frames += [0.0] * int(stand_s * FPS)
    n = int(down_s * FPS)
    frames += [0.5 - 0.5 * math.cos(math.pi * i / n) for i in range(n)]
    frames += [1.0] * int(hold_s * FPS)
    n = int(up_s * FPS)
    frames += [0.5 + 0.5 * math.cos(math.pi * i / n) for i in range(n)]
    return frames


class FakeProvider:
    """Stand-in for MultiCameraPoseProvider.get_pose(). Script is a list of per-call specs."""

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.queue: list[dict] = []
        self.frame = np.zeros((72, 128, 3), dtype=np.uint8)
        self.frame_index_mode = "zero"  # production triangulator: always 0
        self.calls = 0
        self.is_calibrated = True
        self.ts_clock = "perf"  # "perf" = multi_capture.py:94; "time" = MediaPipe/RTMPose time.time()

    def push(self, points: np.ndarray | None, confidences: np.ndarray | None = None, duplicate: bool = False) -> None:
        self.queue.append({"points": points, "conf": confidences, "duplicate": duplicate})

    def get_pose(self):
        from biomechanics.utils.types import Skeleton3D

        spec = self.queue.pop(0)
        self.calls += 1
        if spec["points"] is None:
            return self.frame, None, None
        conf = spec["conf"] if spec["conf"] is not None else np.full(len(spec["points"]), 0.9)
        frame_index = 0 if self.frame_index_mode == "zero" else self.calls
        # capture timestamp: perf_counter clock (multi_capture.py:94); duplicates reuse the last one
        if spec["duplicate"]:
            ts = self._last_ts
        else:
            ts = self.clock.perf_counter() if self.ts_clock == "perf" else self.clock.time()
            self._last_ts = ts
        skeleton = Skeleton3D.from_numpy(spec["points"], confidences=list(conf), timestamp=ts, frame_index=frame_index)
        return self.frame, None, skeleton

    def reset_temporal_state(self) -> None:
        pass

    @property
    def swap_count(self) -> int:
        return 0

    def release(self) -> None:
        pass


def build_pipeline(clock: FakeClock, bilstm: bool = False, multi_camera: bool = True):
    os.environ["NOWVA_MULTI_CAMERA"] = "true" if multi_camera else "false"
    from biomechanics import pipeline as pipeline_module
    from biomechanics.config import load_pipeline_config
    from biomechanics.kinematics.valgus import TriangulatedValgusEstimator

    config = load_pipeline_config()
    config.bilstm.enabled = bilstm
    config.triangulation.calibration_file = None
    with patch("biomechanics.pipeline.cv2.VideoCapture"):
        pipe = pipeline_module.BiomechanicsPipeline(config, defer_capture=True)
    provider = FakeProvider(clock)
    pipe._multi_camera = True
    pipe._multi_camera_provider = provider
    pipe._valgus_estimator = TriangulatedValgusEstimator()
    fake_time = types.SimpleNamespace(time=clock.time, perf_counter=clock.perf_counter)
    pipeline_module.time = fake_time
    return pipe, provider


def step(pipe, provider, clock: FakeClock, points, confidences=None, duplicate: bool = False, dt: float = 1.0 / FPS):
    provider.push(points, confidences, duplicate)
    result = pipe.process_frame()
    clock.advance(dt)
    return result
