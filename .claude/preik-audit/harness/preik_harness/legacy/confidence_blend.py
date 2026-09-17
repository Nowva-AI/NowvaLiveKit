"""
Confidence-Weighted Keypoint Blending

Instead of a binary accept/reject threshold, blends each keypoint's
detected position with its previous position based on confidence score.

    blended = confidence * detected + (1 - confidence) * previous

High confidence → trust detection.
Low confidence → rely on previous position.
Zero confidence → fully use previous position.

Operates on Skeleton3D, BEFORE velocity clamping and IK.
"""

import numpy as np
from typing import Optional

from biomechanics.utils.types import Skeleton3D


class ConfidenceBlender:
    """
    Blends keypoint positions with previous frame based on confidence.

    Args:
        min_confidence: Below this confidence, fully use previous position.
            Default 0.1 (effectively zero-confidence floor).
        max_confidence: Above this confidence, fully trust detection.
            Default 0.9. Confidences between min and max are linearly
            interpolated.
    """

    def __init__(
        self,
        min_confidence: float = 0.1,
        max_confidence: float = 0.9,
    ):
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence
        self._prev_positions: Optional[np.ndarray] = None  # (17, 3)

    def blend(self, skeleton: Skeleton3D) -> Skeleton3D:
        """
        Apply confidence-weighted blending.

        First frame: stores positions, returns skeleton unchanged.
        Subsequent frames: blends each keypoint based on its confidence.

        Args:
            skeleton: Current frame's Skeleton3D with per-keypoint confidences.

        Returns:
            New Skeleton3D with blended positions.
        """
        current = skeleton.to_numpy()  # (17, 3)
        confidences = np.array([kp.confidence for kp in skeleton.keypoints])

        if self._prev_positions is None:
            # `current` is freshly allocated by to_numpy() and not aliased into
            # the skeleton (from_numpy copies into Point3D floats), so storing
            # it directly is safe — no defensive copy needed.
            self._prev_positions = current
            return skeleton

        # Compute blend weights: map [min_confidence, max_confidence] → [0, 1]
        range_size = self.max_confidence - self.min_confidence
        if range_size < 1e-6:
            weights = np.where(confidences >= self.max_confidence, 1.0, 0.0)
        else:
            weights = (confidences - self.min_confidence) / range_size
            weights = np.clip(weights, 0.0, 1.0)

        # Blend: weight * current + (1 - weight) * previous
        # weights shape: (17,) → expand to (17, 1) for broadcasting
        blended = (
            weights[:, np.newaxis] * current
            + (1.0 - weights[:, np.newaxis]) * self._prev_positions
        )

        # `blended` is a fresh array each frame and is only read (not mutated)
        # downstream, so it can back _prev_positions without a copy.
        self._prev_positions = blended

        return Skeleton3D.from_numpy(
            blended,
            confidences=confidences,
            timestamp=skeleton.timestamp,
            frame_index=skeleton.frame_index,
        )

    def reset(self):
        """Reset state."""
        self._prev_positions = None
