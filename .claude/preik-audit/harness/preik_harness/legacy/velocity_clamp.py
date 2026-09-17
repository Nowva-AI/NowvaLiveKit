"""
Velocity Clamping for 3D Skeleton Keypoints

Limits per-frame displacement of each keypoint to a physically plausible
maximum. If a keypoint moves further than the threshold in one frame,
its position is clamped to the maximum allowed displacement along the
direction of movement.

This runs BEFORE the IK solver, operating on Skeleton3D objects.
"""

import numpy as np
from typing import Optional

from biomechanics.utils.types import Skeleton3D


class VelocityClamp:
    """
    Clamps per-frame keypoint displacement to a maximum threshold.

    Uses actual inter-frame dt from skeleton timestamps to compute the
    displacement limit each frame, so dropped frames or variable FPS
    don't cause false clamping.

    Args:
        max_velocity_m_per_s: Maximum plausible joint velocity in meters/sec.
            Human joints rarely exceed 3 m/s during heavy barbell lifts.
            Default 2.5 m/s provides headroom without passing spikes.
        target_fps: Fallback frame rate used only when timestamps are
            unavailable. Default 30.
    """

    def __init__(
        self,
        max_velocity_m_per_s: float = 2.5,
        target_fps: int = 30,
    ):
        self.max_velocity = max_velocity_m_per_s
        self._fallback_dt = 1.0 / target_fps
        self._prev_positions: Optional[np.ndarray] = None  # (17, 3)
        self._prev_timestamp: Optional[float] = None

    def clamp(self, skeleton: Skeleton3D) -> Skeleton3D:
        """
        Apply velocity clamping to a skeleton.

        First frame: stores positions, returns skeleton unchanged.
        Subsequent frames: clamps each keypoint displacement using
        actual inter-frame dt so variable FPS doesn't cause false clamping.

        Args:
            skeleton: Current frame's Skeleton3D

        Returns:
            New Skeleton3D with clamped keypoint positions.
            Confidences, timestamp, frame_index are preserved.
        """
        current = skeleton.to_numpy()  # (17, 3)
        confidences = np.array([kp.confidence for kp in skeleton.keypoints])

        if self._prev_positions is None:
            self._prev_positions = current.copy()
            self._prev_timestamp = skeleton.timestamp
            return skeleton

        # Use actual dt when timestamps are available
        dt = self._fallback_dt
        if skeleton.timestamp is not None and self._prev_timestamp is not None:
            actual_dt = skeleton.timestamp - self._prev_timestamp
            if actual_dt > 1e-6:
                dt = actual_dt

        max_displacement = self.max_velocity * dt

        displacement = current - self._prev_positions  # (17, 3)
        distances = np.linalg.norm(displacement, axis=1)  # (17,)

        # Find keypoints that exceed threshold
        exceeded = distances > max_displacement

        if np.any(exceeded):
            # Compute unit direction vectors for exceeded keypoints
            # Avoid division by zero for zero-distance keypoints
            safe_distances = np.where(distances > 1e-10, distances, 1.0)
            unit_dirs = displacement / safe_distances[:, np.newaxis]

            # Clamp: move only max_displacement along original direction
            clamped = self._prev_positions.copy()
            clamped[exceeded] = (
                self._prev_positions[exceeded]
                + unit_dirs[exceeded] * max_displacement
            )
            # Keep non-exceeded keypoints at their detected position
            clamped[~exceeded] = current[~exceeded]

            self._prev_positions = clamped.copy()
            self._prev_timestamp = skeleton.timestamp
            return Skeleton3D.from_numpy(
                clamped,
                confidences=confidences,
                timestamp=skeleton.timestamp,
                frame_index=skeleton.frame_index,
            )

        # No clamping needed
        self._prev_positions = current.copy()
        self._prev_timestamp = skeleton.timestamp
        return skeleton

    def reset(self):
        """Reset state (e.g., new session or exercise)."""
        self._prev_positions = None
        self._prev_timestamp = None
