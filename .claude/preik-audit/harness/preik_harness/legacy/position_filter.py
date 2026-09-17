"""
One Euro Filter for 3D Keypoint Positions

Applies adaptive temporal smoothing to each keypoint's (x, y, z) coordinates
BEFORE the IK solver. This removes frame-to-frame jitter from MediaPipe's
3D world landmarks while preserving fast intentional movements.

Runs as the final pre-IK filter step — after confidence blending, velocity
clamping, and bone constraints have already cleaned the worst outliers.
"""

import numpy as np
from typing import Optional

from .filters import OneEuroFilter
from biomechanics.utils.types import Skeleton2D, Skeleton3D


class KeypointPositionSmoother:
    """
    Applies One Euro Filters to each keypoint's 3D position.

    Maintains 17 x 3 = 51 independent One Euro filters (one per keypoint
    per axis). The adaptive cutoff ensures heavy smoothing during quiet
    standing and responsive tracking during fast squat descent/ascent.

    Args:
        min_cutoff: Minimum cutoff frequency (Hz). Lower = smoother.
            0.8 is good for 3D positions — aggressive smoothing at rest.
        beta: Speed coefficient. Higher = more responsive to fast movement.
            4.0 maps typical joint velocities (0.1-0.5 m/s) to a useful
            adaptive cutoff range (~0.8-2.8 Hz).
        d_cutoff: Cutoff for the derivative low-pass filter.
    """

    def __init__(
        self,
        min_cutoff: float = 0.8,
        beta: float = 4.0,
        d_cutoff: float = 1.0,
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        # Lazily created: dict[(keypoint_idx, axis)] -> OneEuroFilter
        self._filters: dict = {}

    def _get_filter(self, kp_idx: int, axis: int) -> OneEuroFilter:
        key = (kp_idx, axis)
        if key not in self._filters:
            self._filters[key] = OneEuroFilter(
                min_cutoff=self.min_cutoff,
                beta=self.beta,
                d_cutoff=self.d_cutoff,
            )
        return self._filters[key]

    def smooth(self, skeleton: Skeleton3D) -> Skeleton3D:
        """
        Apply One Euro filtering to all keypoint positions.

        Args:
            skeleton: Current frame's Skeleton3D (ideally already cleaned
                      by confidence blending, velocity clamping, bone constraints).

        Returns:
            New Skeleton3D with temporally smoothed positions.
        """
        positions = skeleton.to_numpy()  # (17, 3)
        confidences = np.array([kp.confidence for kp in skeleton.keypoints])
        t = skeleton.timestamp

        smoothed = np.empty_like(positions)
        for kp_idx in range(positions.shape[0]):
            for axis in range(3):
                filt = self._get_filter(kp_idx, axis)
                smoothed[kp_idx, axis] = filt.filter(positions[kp_idx, axis], t)

        return Skeleton3D.from_numpy(
            smoothed,
            confidences=confidences,
            timestamp=skeleton.timestamp,
            frame_index=skeleton.frame_index,
        )

    def reset(self):
        """Reset all filter state."""
        self._filters.clear()


class Skeleton2DSmoother:
    """One Euro Filter for 2D keypoint pixel coordinates (display-only)."""

    def __init__(
        self,
        min_cutoff: float = 1.5,
        beta: float = 0.5,
        d_cutoff: float = 1.0,
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._filters: dict = {}

    def _get_filter(self, kp_idx: int, axis: int) -> OneEuroFilter:
        key = (kp_idx, axis)
        if key not in self._filters:
            self._filters[key] = OneEuroFilter(
                min_cutoff=self.min_cutoff,
                beta=self.beta,
                d_cutoff=self.d_cutoff,
            )
        return self._filters[key]

    def smooth(self, skeleton: Skeleton2D) -> Skeleton2D:
        positions = skeleton.to_numpy()  # (17, 3) — [x, y, confidence]
        confidences = positions[:, 2]
        t = skeleton.timestamp

        smoothed = np.empty_like(positions)
        for kp_idx in range(positions.shape[0]):
            if confidences[kp_idx] <= 0:
                smoothed[kp_idx] = positions[kp_idx]
                continue
            for axis in range(2):
                filt = self._get_filter(kp_idx, axis)
                smoothed[kp_idx, axis] = filt.filter(positions[kp_idx, axis], t)
            smoothed[kp_idx, 2] = confidences[kp_idx]

        return Skeleton2D.from_numpy(
            smoothed,
            timestamp=skeleton.timestamp,
            frame_index=skeleton.frame_index,
        )

    def reset(self):
        self._filters.clear()
