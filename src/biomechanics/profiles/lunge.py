"""
Lunge Exercise Profile

Walking, reverse, and stationary lunges. Signal is the max
of left/right knee flexion (whichever leg is forward).
"""

import logging
from typing import Dict, List, Optional

from biomechanics.config import BiomechanicsConfig, HipPositionCounterConfig
from biomechanics.faults.fault_types import FaultRule
from biomechanics.faults.rules.range_of_motion import RangeOfMotionRule
from biomechanics.faults.rules.knee_valgus import KneeValgusRule
from biomechanics.faults.rules.forward_lean import ForwardLeanRule
from biomechanics.faults.rules.symmetry import SymmetryRule
from biomechanics.profiles.base import ExerciseProfile
from biomechanics.profiles.registry import register_profile
from biomechanics.utils.types import JointAngles, Skeleton3D

logger = logging.getLogger(__name__)


@register_profile(
    "lunge", "walking_lunge", "reverse_lunge", "stationary_lunge",
    "barbell_lunge", "dumbbell_lunge",
)
class LungeProfile(ExerciseProfile):
    """Profile for all lunge variants."""

    name = "lunge"
    movement_pattern = "lunge"

    def create_fault_rules(self, config: BiomechanicsConfig) -> List[FaultRule]:
        return [
            RangeOfMotionRule(
                metric_getter=lambda a: max(a.knee_flexion_l, a.knee_flexion_r),
                target_threshold=90.0,
                min_threshold=60.0,
                direction="max",
                metric_name="lunge depth",
            ),
            KneeValgusRule(
                mild_threshold=8.0,
                moderate_threshold=13.0,
                severe_threshold=18.0,
            ),
            ForwardLeanRule(
                mild_threshold=155.0,
                moderate_threshold=145.0,
                severe_threshold=135.0,
            ),
            SymmetryRule(
                left_getter=lambda a: a.knee_flexion_l,
                right_getter=lambda a: a.knee_flexion_r,
                joint_name="knee",
                mild_threshold=10.0,
                moderate_threshold=18.0,
                severe_threshold=25.0,
            ),
        ]

    def get_rep_signal(
        self, skeleton_3d: Skeleton3D, angles: Optional[JointAngles] = None
    ) -> float:
        """Front-knee flexion as signal (whichever knee is more flexed)."""
        if angles is None:
            return 0.0
        return max(angles.knee_flexion_l, angles.knee_flexion_r)

    def create_rep_counter_config(
        self, config: BiomechanicsConfig
    ) -> HipPositionCounterConfig:
        return HipPositionCounterConfig(
            entry_vel_threshold=15.0,
            bottom_vel_threshold=8.0,
            ascending_vel_threshold=15.0,
            min_depth_cm=30.0,
            standing_return_cm=10.0,
            min_rep_duration_frames=15,
        )

    def get_depth_metric(self, angles: JointAngles) -> float:
        return max(angles.knee_flexion_l, angles.knee_flexion_r)

    def get_asymmetry_metrics(self, angles: JointAngles) -> Dict[str, float]:
        return {"knee": angles.knee_asymmetry}

    def get_cue_dict(self) -> Optional[Dict[str, str]]:
        return {
            "range_of_motion": "Step deeper — front knee to 90°",
            "knee_valgus": "Push front knee out — don't let it cave",
            "forward_lean": "Stay upright — chest up",
            "bilateral_asymmetry": "Balance both legs evenly",
        }
