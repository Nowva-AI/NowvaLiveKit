"""
Romanian Deadlift (RDL) Exercise Profile

Hip hinge with minimal knee bend. Signal is trunk flexion
(inverted so descent = signal rising toward max for the FSM).
"""

import logging
from typing import Dict, List, Optional

from biomechanics.config import BiomechanicsConfig, HipPositionCounterConfig
from biomechanics.faults.fault_types import FaultRule
from biomechanics.faults.rules.range_of_motion import RangeOfMotionRule
from biomechanics.faults.rules.symmetry import SymmetryRule
from biomechanics.faults.rules.back_rounding import BackRoundingRule
from biomechanics.faults.rules.trunk_stability import TrunkStabilityRule
from biomechanics.profiles.base import ExerciseProfile
from biomechanics.profiles.registry import register_profile
from biomechanics.utils.types import JointAngles, Skeleton3D

logger = logging.getLogger(__name__)


@register_profile("romanian_deadlift", "rdl", "stiff_leg_deadlift")
class RomanianDeadliftProfile(ExerciseProfile):
    """Profile for Romanian deadlift / stiff-leg deadlift."""

    name = "romanian_deadlift"
    movement_pattern = "hip_hinge"

    def create_fault_rules(self, config: BiomechanicsConfig) -> List[FaultRule]:
        return [
            RangeOfMotionRule(
                metric_getter=lambda a: 180.0 - a.trunk_flexion,
                target_threshold=90.0,
                min_threshold=60.0,
                direction="max",
                metric_name="hip hinge",
            ),
            BackRoundingRule(
                mild_threshold=8.0,
                moderate_threshold=15.0,
                severe_threshold=25.0,
            ),
            SymmetryRule(
                left_getter=lambda a: a.hip_flexion_l,
                right_getter=lambda a: a.hip_flexion_r,
                joint_name="hip",
                mild_threshold=6.0,
                moderate_threshold=12.0,
                severe_threshold=18.0,
            ),
            # Warn if knees bend too much (turning it into a conventional DL)
            RangeOfMotionRule(
                metric_getter=lambda a: a.avg_knee_flexion,
                target_threshold=30.0,
                min_threshold=30.0,
                direction="min",
                metric_name="knee bend",
            ),
        ]

    def get_rep_signal(
        self, skeleton_3d: Skeleton3D, angles: Optional[JointAngles] = None
    ) -> float:
        """Inverted trunk flexion: rises as user hinges forward."""
        if angles is None:
            return 0.0
        return -(angles.trunk_flexion)

    def create_rep_counter_config(
        self, config: BiomechanicsConfig
    ) -> HipPositionCounterConfig:
        return HipPositionCounterConfig(
            entry_vel_threshold=8.0,
            bottom_vel_threshold=4.0,
            ascending_vel_threshold=8.0,
            min_depth_cm=30.0,
            standing_return_cm=5.0,
            min_rep_duration_frames=20,
        )

    def get_depth_metric(self, angles: JointAngles) -> float:
        return angles.trunk_flexion

    def get_asymmetry_metrics(self, angles: JointAngles) -> Dict[str, float]:
        return {"hip": angles.hip_asymmetry}

    def get_cue_dict(self) -> Optional[Dict[str, str]]:
        return {
            "range_of_motion": "Hinge deeper — feel the hamstring stretch",
            "back_rounding": "Keep spine neutral — chest up",
            "bilateral_asymmetry": "Even out your hinge — weight balanced",
        }
