"""
Untracked Exercise Profile

Stands in for an exercise no profile models yet (bench press, pull-ups,
good mornings). The pipeline keeps running so the voice agent stays
connected, but no reps are counted and no faults fire: the lifter ends
each set by voice.
"""

import math
from typing import List, Optional

from biomechanics.config import BiomechanicsConfig
from biomechanics.faults.fault_types import FaultRule
from biomechanics.profiles.base import ExerciseProfile
from biomechanics.utils.types import JointAngles, Skeleton3D


class UntrackedProfile(ExerciseProfile):

    name = "untracked"
    movement_pattern = None

    def create_fault_rules(self, config: BiomechanicsConfig) -> List[FaultRule]:
        return []

    def get_rep_signal(
        self, skeleton_3d: Skeleton3D, angles: Optional[JointAngles] = None
    ) -> float:
        # The rep counter ignores NaN, so no rep ever opens.
        return math.nan


class UntrackedVariantProfile(UntrackedProfile):
    """A deadlift variant the conventional deadlift's rules do not model (sumo,
    trap bar, deficit, snatch grip, single leg, rack pulls). Untracked, with the
    deadlift's safety flags: hinged frames with plates hiding the feet never feed
    the squat's camera refines or body measurements."""

    allows_camera_refine = False
    feeds_body_calibration = False
