"""
One Euro Filter for 2D skeleton overlay smoothing (display only).

Pixel-unit parameters: the derivative is in px/s, so beta is tiny compared
with the metre-unit defaults it replaced. This never feeds analysis — the
single-camera valgus estimator reads the raw 2D pose.
"""

from __future__ import annotations

import numpy as np

from biomechanics.utils.filters import OneEuroFilter
from biomechanics.utils.types import Skeleton2D

DEFAULT_MIN_CUTOFF_HZ = 1.0
DEFAULT_BETA_PER_PX = 0.02
DEFAULT_D_CUTOFF_HZ = 1.0


class Skeleton2DSmoother:
    """One Euro Filter for 2D keypoint pixel coordinates (display-only)."""

    def __init__(
        self,
        min_cutoff: float = DEFAULT_MIN_CUTOFF_HZ,
        beta: float = DEFAULT_BETA_PER_PX,
        d_cutoff: float = DEFAULT_D_CUTOFF_HZ,
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._filters: dict[tuple[int, int], OneEuroFilter] = {}

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
        positions = skeleton.to_numpy()  # (N, 3) — [x, y, confidence]
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

    def reset(self) -> None:
        self._filters.clear()
