"""
Single-camera mount: the installation's measured pitch and lens height, and the
delta tripwire that revokes a certified mount when the camera is knocked.

On one camera the world frame is the camera's, so a pitched camera leaks
forward-back motion into every height (Δy·cos θ + Δz·sin θ). A measured mount
lets an exercise level the skeleton (geometry.level_camera_frame); the
pipeline's own standing-pitch estimate includes posture and is only ever used
as a delta from its reading at certification (SHARED-PLUMBING.md §3).
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass

from biomechanics.config import CaptureConfig
from biomechanics.utils.frame_context import MOUNT_INCLINOMETER, MOUNT_NONE, MOUNT_RACK

logger = logging.getLogger(__name__)

PITCH_ENV = "NOWVA_CAMERA_PITCH_DEG"
MEASURED_MOUNT_SOURCES = (MOUNT_RACK, MOUNT_INCLINOMETER)


@dataclass(frozen=True)
class CameraMount:
    """pitch_deg is positive when the lens tilts up; NaN fields are unknown."""
    source: str = MOUNT_NONE
    pitch_deg: float = math.nan
    height_m: float = math.nan

    @property
    def is_measured(self) -> bool:
        return self.source in MEASURED_MOUNT_SOURCES and math.isfinite(self.pitch_deg)


def camera_mount_from_config(capture: CaptureConfig) -> CameraMount:
    """The configured mount; NOWVA_CAMERA_PITCH_DEG overrides the pitch, and a
    pitch entered that way on an undeclared mount counts as an inclinometer reading."""
    source = capture.mount_source
    pitch_deg = capture.mount_pitch_deg if capture.mount_pitch_deg is not None else math.nan
    override = os.getenv(PITCH_ENV)
    if override:
        try:
            pitch_deg = float(override)
        except ValueError:
            logger.warning("[CAMERA MOUNT] %s=%r is not a number — ignored", PITCH_ENV, override)
        else:
            if source == MOUNT_NONE:
                source = MOUNT_INCLINOMETER
    height_m = capture.mount_height_m if capture.mount_height_m is not None else math.nan
    return CameraMount(source=source, pitch_deg=pitch_deg, height_m=height_m)


class PitchDeltaTripwire:
    """Watches a posture-matched pitch reading against the reading recorded at
    certification. Each exercise supplies its own reading and tolerance; the
    first change beyond the tolerance trips it, and it stays tripped."""

    def __init__(self, tolerance_deg: float) -> None:
        self.tolerance_deg = tolerance_deg
        self.reference_deg = math.nan
        self.tripped_delta_deg: float | None = None

    @property
    def tripped(self) -> bool:
        return self.tripped_delta_deg is not None

    def set_reference(self, reading_deg: float) -> None:
        self.reference_deg = reading_deg

    def check(self, reading_deg: float) -> float | None:
        """The change from the reference when this reading trips the wire, else None."""
        if self.tripped or not math.isfinite(self.reference_deg) or not math.isfinite(reading_deg):
            return None
        delta_deg = abs(reading_deg - self.reference_deg)
        if delta_deg <= self.tolerance_deg:
            return None
        self.tripped_delta_deg = delta_deg
        return delta_deg
