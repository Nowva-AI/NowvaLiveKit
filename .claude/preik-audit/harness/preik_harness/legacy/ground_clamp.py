"""
Ground Plane Clamping for Ankle Keypoints

Calibrates standing foot placement from the first N frames, then
enforces ground-plane constraints: both ankles at the same Y, stance
width preserved, and ankles never past calibrated leg extension.

Runs AFTER bone length enforcement and BEFORE position smoothing
in the pre-IK filter chain.
"""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

import numpy as np

from biomechanics.utils.types import Skeleton3D, CocoKeypoints as CK

if TYPE_CHECKING:
    from biomechanics.utils.standing_gate import StandingPoseGate

logger = logging.getLogger(__name__)


class GroundClamp:
    """
    Enforces ground-plane constraints on ankle and foot keypoints.

    During calibration (standing), records stance width and ankle Y
    positions. After calibration, enforces flat-floor and anti-slide
    constraints every frame.
    """

    def __init__(
        self,
        calibration_frames: int = 30,
        stance_width_tolerance_m: float = 0.02,
        ankle_y_tolerance_m: float = 0.01,
        min_leg_extension_ratio: float = 0.75,
        standing_gate: Optional["StandingPoseGate"] = None,
    ):
        self._calibration_frames = calibration_frames
        self._stance_width_tol = stance_width_tolerance_m
        self._ankle_y_tol = ankle_y_tolerance_m
        self._min_leg_extension_ratio = min_leg_extension_ratio
        self._standing_gate = standing_gate

        self._calibrated = False
        self._frame_count = 0
        self._rejection_count = 0

        self._stance_widths: list[float] = []
        self._ankle_y_l_obs: list[float] = []
        self._ankle_y_r_obs: list[float] = []
        self._leg_extension_obs: list[float] = []

        self._stance_width: float = 0.0
        self._ankle_y_max_l: float = 0.0
        self._ankle_y_max_r: float = 0.0

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    def clamp(self, skeleton: Skeleton3D) -> Skeleton3D:
        points = skeleton.to_numpy()
        confidences = np.array([kp.confidence for kp in skeleton.keypoints])
        n_kpts = len(points)

        if confidences[CK.LEFT_ANKLE] < 0.1 or confidences[CK.RIGHT_ANKLE] < 0.1:
            return skeleton

        if not self._calibrated:
            if self._standing_gate is not None and not self._standing_gate.is_ready:
                return skeleton

            self._record_calibration(points)
            self._frame_count += 1
            if self._frame_count >= self._calibration_frames:
                self._finalize_calibration()
            return skeleton

        corrected = points.copy()

        # --- Floor penetration: ankle can't extend past calibrated standing ---
        # Y-down convention: larger Y = further below hip = toward floor.
        corrected[CK.LEFT_ANKLE, 1] = min(
            corrected[CK.LEFT_ANKLE, 1], self._ankle_y_max_l,
        )
        corrected[CK.RIGHT_ANKLE, 1] = min(
            corrected[CK.RIGHT_ANKLE, 1], self._ankle_y_max_r,
        )

        # --- Flat floor: equalize ankle Y when they diverge ---
        y_diff = abs(corrected[CK.LEFT_ANKLE, 1] - corrected[CK.RIGHT_ANKLE, 1])
        if y_diff > self._ankle_y_tol:
            avg_y = (corrected[CK.LEFT_ANKLE, 1] + corrected[CK.RIGHT_ANKLE, 1]) / 2.0
            corrected[CK.LEFT_ANKLE, 1] = avg_y
            corrected[CK.RIGHT_ANKLE, 1] = avg_y

        # --- Stance width: lock ankle-to-ankle XZ distance ---
        l_xz = corrected[CK.LEFT_ANKLE, [0, 2]]
        r_xz = corrected[CK.RIGHT_ANKLE, [0, 2]]
        current_width = float(np.linalg.norm(l_xz - r_xz))

        if current_width > 1e-6:
            width_error = abs(current_width - self._stance_width)
            if width_error > self._stance_width_tol:
                mid_xz = (l_xz + r_xz) / 2.0
                direction = (l_xz - r_xz) / current_width
                half_target = self._stance_width / 2.0
                corrected[CK.LEFT_ANKLE, 0] = mid_xz[0] + direction[0] * half_target
                corrected[CK.LEFT_ANKLE, 2] = mid_xz[1] + direction[1] * half_target
                corrected[CK.RIGHT_ANKLE, 0] = mid_xz[0] - direction[0] * half_target
                corrected[CK.RIGHT_ANKLE, 2] = mid_xz[1] - direction[1] * half_target

        # --- Propagate ankle corrections to the foot keypoints ---
        # Toes and heels are rigidly attached to the ankle; moving the ankle
        # without them would tear the foot apart.
        for ankle_idx, foot_idx in [
            (CK.LEFT_ANKLE, CK.LEFT_FOOT_INDEX),
            (CK.RIGHT_ANKLE, CK.RIGHT_FOOT_INDEX),
            (CK.LEFT_ANKLE, CK.LEFT_HEEL),
            (CK.RIGHT_ANKLE, CK.RIGHT_HEEL),
        ]:
            if foot_idx < n_kpts and confidences[foot_idx] > 0:
                delta = corrected[ankle_idx] - points[ankle_idx]
                corrected[foot_idx] += delta

        return Skeleton3D.from_numpy(
            corrected,
            confidences=confidences,
            timestamp=skeleton.timestamp,
            frame_index=skeleton.frame_index,
        )

    def _record_calibration(self, points: np.ndarray) -> None:
        l_ankle = points[CK.LEFT_ANKLE]
        r_ankle = points[CK.RIGHT_ANKLE]
        width = float(np.linalg.norm(l_ankle[[0, 2]] - r_ankle[[0, 2]]))
        self._stance_widths.append(width)
        self._ankle_y_l_obs.append(float(l_ankle[1]))
        self._ankle_y_r_obs.append(float(r_ankle[1]))
        self._leg_extension_obs.append(self._leg_extension(points))

    def _leg_extension(self, points: np.ndarray) -> float:
        hip_mid_y = (points[CK.LEFT_HIP, 1] + points[CK.RIGHT_HIP, 1]) / 2.0
        extensions = []
        for hip_idx, knee_idx, ankle_idx in (
            (CK.LEFT_HIP, CK.LEFT_KNEE, CK.LEFT_ANKLE),
            (CK.RIGHT_HIP, CK.RIGHT_KNEE, CK.RIGHT_ANKLE),
        ):
            leg_length = float(
                np.linalg.norm(points[hip_idx] - points[knee_idx])
                + np.linalg.norm(points[knee_idx] - points[ankle_idx])
            )
            if leg_length < 1e-6:
                extensions.append(0.0)
                continue
            span = abs(float(points[ankle_idx, 1]) - hip_mid_y)
            extensions.append(span / leg_length)
        return min(extensions)

    def _finalize_calibration(self) -> None:
        # A standing person's ankles sit most of a leg-length below the
        # hips. Hallucinated folded legs (ankles near hip height) must not
        # become the calibration, or the clamp enforces the fold all
        # session — reject and keep collecting.
        leg_extension = float(np.median(self._leg_extension_obs))
        if leg_extension < self._min_leg_extension_ratio:
            if self._rejection_count % 10 == 0:
                logger.warning(
                    "[GROUND CLAMP] Calibration rejected (x%d): leg extension "
                    "%.2f < %.2f (folded/hallucinated legs) — recollecting",
                    self._rejection_count + 1,
                    leg_extension, self._min_leg_extension_ratio,
                )
            self._rejection_count += 1
            self._clear_observations()
            return

        self._stance_width = float(np.median(self._stance_widths))
        self._ankle_y_max_l = float(np.median(self._ankle_y_l_obs))
        self._ankle_y_max_r = float(np.median(self._ankle_y_r_obs))
        self._calibrated = True

        self._clear_observations()

        logger.info(
            "[GROUND CLAMP] Calibrated: stance_width=%.3fm, "
            "ankle_y_max L=%.3fm R=%.3fm",
            self._stance_width, self._ankle_y_max_l, self._ankle_y_max_r,
        )

    def _clear_observations(self) -> None:
        self._frame_count = 0
        self._stance_widths.clear()
        self._ankle_y_l_obs.clear()
        self._ankle_y_r_obs.clear()
        self._leg_extension_obs.clear()

    def reset(self) -> None:
        self._calibrated = False
        self._rejection_count = 0
        self._clear_observations()
