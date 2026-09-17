"""Pre-IK chain: the single definition of what happens to a 3D skeleton between
pose estimation and the IK solver. Multi-camera input is the world-frame
triangulated skeleton: fixed-lag Kalman -> foot contact -> hip re-centring.
Single-camera (MediaPipe) input is already hip-centred: Kalman only.
Skeleton3D <-> numpy conversion happens exactly once at entry and exit.
"""

from __future__ import annotations

from collections import deque
from typing import Callable, NamedTuple

import numpy as np

from biomechanics.config import BiomechanicsConfig
from biomechanics.triangulation.triangulator import UNCERTAINTY_SCALE_M
from biomechanics.utils.foot_contact import FootContactModel, FootState
from biomechanics.utils.keypoint_kalman import FixedLagKeypointSmoother, KeypointKalmanOutput
from biomechanics.utils.types import CocoKeypoints as CK
from biomechanics.utils.types import Skeleton3D

STAGE_RAW = "raw"
STAGE_KALMAN = "kalman"
STAGE_FOOT_CONTACT = "foot_contact"
STAGE_RECENTRE = "recentre"

# Timestamps -> frame indices kept for mapping the lagged output back to the
# frame it describes; a few more than the lag so a re-initialised smoother
# (which shortens the lag) still finds its entry.
RECENT_FRAMES = 8

TapFn = Callable[[str, np.ndarray, np.ndarray], None]


class PreIKResult(NamedTuple):
    analysis: Skeleton3D
    display: Skeleton3D | None
    analysis_world: Skeleton3D | None
    foot_state: FootState | None
    velocities: np.ndarray


def _hips_tracked(confidences: np.ndarray) -> bool:
    return bool(confidences[CK.LEFT_HIP] > 0.0 and confidences[CK.RIGHT_HIP] > 0.0)


def _centre_at_hips(points: np.ndarray, confidences: np.ndarray) -> np.ndarray | None:
    if not _hips_tracked(confidences):
        return None
    hip_midpoint = (points[CK.LEFT_HIP] + points[CK.RIGHT_HIP]) / 2.0
    return np.where(confidences[:, None] > 0.0, points - hip_midpoint, 0.0)


class PreIKChain:
    """Runs the pre-IK stages on arrays and reports each stage to an optional inspector tap."""

    def __init__(
        self,
        smoother: FixedLagKeypointSmoother,
        foot_contact: FootContactModel | None,
        recentre: bool,
    ) -> None:
        self._smoother = smoother
        self._foot_contact = foot_contact
        self._recentre = recentre
        self._tap: TapFn | None = None
        self._recent: deque[tuple[float, int]] = deque(maxlen=RECENT_FRAMES)
        stage_names = [STAGE_RAW, STAGE_KALMAN]
        if foot_contact is not None:
            stage_names.append(STAGE_FOOT_CONTACT)
        if recentre:
            stage_names.append(STAGE_RECENTRE)
        self.stage_names: tuple[str, ...] = tuple(stage_names)

    def set_tap(self, tap: TapFn | None) -> None:
        self._tap = tap

    def reset(self) -> None:
        """Clear temporal (Kalman) state only. Foot contact is session-scoped and untouched."""
        self._smoother.reset()
        self._recent.clear()

    def run(self, skeleton: Skeleton3D) -> PreIKResult | None:
        """Process one measured frame. None means the hips are unavailable (dropout path)."""
        points = skeleton.to_numpy()
        confidences = np.array([kp.confidence for kp in skeleton.keypoints], dtype=np.float64)
        self._emit(STAGE_RAW, points, confidences)
        output = self._smoother.update(points, confidences, skeleton.timestamp)
        return self._finish(output, skeleton.timestamp, skeleton.frame_index)

    def predict_missing(self, timestamp: float) -> PreIKResult | None:
        """Advance the smoother without a measurement; None once prediction has run out."""
        output = self._smoother.predict_missing(timestamp)
        frame_index = self._recent[-1][1] if self._recent else 0
        return self._finish(output, timestamp, frame_index)

    def _emit(self, stage: str, points: np.ndarray, confidences: np.ndarray) -> None:
        if self._tap is not None:
            self._tap(stage, points, confidences)

    def _lagged_frame_index(self, lagged_timestamp: float, fallback: int) -> int:
        for timestamp, frame_index in self._recent:
            if timestamp == lagged_timestamp:
                return frame_index
        return fallback

    def _finish(
        self, output: KeypointKalmanOutput | None, timestamp: float, frame_index: int,
    ) -> PreIKResult | None:
        if output is None:
            return None
        if not self._recent or timestamp > self._recent[-1][0]:
            self._recent.append((timestamp, frame_index))
        lagged_frame_index = self._lagged_frame_index(output.lagged_timestamp, frame_index)

        lagged_points = output.lagged_points
        lagged_confidences = output.lagged_confidences
        self._emit(STAGE_KALMAN, lagged_points, lagged_confidences)

        foot_state: FootState | None = None
        if self._foot_contact is not None:
            lagged_points, foot_state = self._foot_contact.update(
                lagged_points, lagged_confidences, output.lagged_timestamp,
            )
            self._emit(STAGE_FOOT_CONTACT, lagged_points, lagged_confidences)

        analysis_world: Skeleton3D | None = None
        if self._recentre:
            analysis_world = Skeleton3D.from_numpy(
                lagged_points, confidences=lagged_confidences,
                timestamp=output.lagged_timestamp, frame_index=lagged_frame_index,
            )
            centred = _centre_at_hips(lagged_points, lagged_confidences)
            if centred is None:
                return None
            lagged_points = centred
            self._emit(STAGE_RECENTRE, lagged_points, lagged_confidences)
            display_points = _centre_at_hips(output.current_points, output.current_confidences)
        else:
            if not _hips_tracked(lagged_confidences):
                return None
            display_points = output.current_points if _hips_tracked(output.current_confidences) else None

        analysis = Skeleton3D.from_numpy(
            lagged_points, confidences=lagged_confidences,
            timestamp=output.lagged_timestamp, frame_index=lagged_frame_index,
        )
        display: Skeleton3D | None = None
        if display_points is not None:
            display = Skeleton3D.from_numpy(
                display_points, confidences=output.current_confidences,
                timestamp=output.current_timestamp, frame_index=frame_index,
            )
        return PreIKResult(
            analysis=analysis,
            display=display,
            analysis_world=analysis_world,
            foot_state=foot_state,
            velocities=output.velocities,
        )


def build_preik_chain(config: BiomechanicsConfig, multi_camera: bool) -> PreIKChain:
    kalman_cfg = config.kalman
    measurement_std_floor_m = (
        kalman_cfg.measurement_std_floor_m
        if multi_camera
        else kalman_cfg.single_camera_measurement_std_floor_m
    )
    smoother = FixedLagKeypointSmoother(
        lag_frames=kalman_cfg.lag_frames,
        process_noise=kalman_cfg.process_noise,
        uncertainty_scale_m=UNCERTAINTY_SCALE_M,
        measurement_std_floor_m=measurement_std_floor_m,
        measurement_std_ceiling_m=kalman_cfg.measurement_std_ceiling_m,
        gate_sigma=kalman_cfg.gate_sigma,
        gate_min_radius_m=kalman_cfg.gate_min_radius_m,
        max_predicted_frames=kalman_cfg.max_predicted_frames,
        min_output_confidence=kalman_cfg.min_output_confidence,
    )
    foot_contact = FootContactModel() if multi_camera and config.foot_contact.enabled else None
    return PreIKChain(smoother=smoother, foot_contact=foot_contact, recentre=multi_camera)
