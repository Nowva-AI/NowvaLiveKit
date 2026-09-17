"""
Deadlift Exercise Profile

Conventional and sumo deadlift. Signal is inverted hip Y-position
so the FSM sees a normal "descend then ascend" pattern even though
the user starts at the bottom.
"""

import logging
from typing import Dict, List, Optional

from biomechanics.config import BiomechanicsConfig, HipPositionCounterConfig
from biomechanics.faults.fault_types import FaultRule
from biomechanics.faults.rules.range_of_motion import RangeOfMotionRule
from biomechanics.faults.rules.symmetry import SymmetryRule
from biomechanics.faults.rules.back_rounding import BackRoundingRule
from biomechanics.faults.rules.bar_path import BarPathRule
from biomechanics.profiles.base import ExerciseProfile
from biomechanics.profiles.registry import register_profile
from biomechanics.utils.types import CocoKeypoints, JointAngles, Skeleton3D

logger = logging.getLogger(__name__)


@register_profile("deadlift", "conventional_deadlift", "sumo_deadlift")
class DeadliftProfile(ExerciseProfile):
    """Profile for conventional and sumo deadlift."""

    name = "deadlift"
    movement_pattern = "deadlift"

    def create_fault_rules(self, config: BiomechanicsConfig) -> List[FaultRule]:
        return [
            RangeOfMotionRule(
                metric_getter=lambda a: a.avg_knee_flexion,
                target_threshold=40.0,
                min_threshold=20.0,
                direction="max",
                metric_name="hip extension",
            ),
            SymmetryRule(
                left_getter=lambda a: a.knee_flexion_l,
                right_getter=lambda a: a.knee_flexion_r,
                joint_name="knee",
                mild_threshold=8.0,
                moderate_threshold=13.0,
                severe_threshold=18.0,
            ),
            BackRoundingRule(
                mild_threshold=10.0,
                moderate_threshold=20.0,
                severe_threshold=30.0,
            ),
            BarPathRule(
                mild_threshold=8.0,
                moderate_threshold=15.0,
                severe_threshold=22.0,
            ),
        ]

    def get_rep_signal(
        self, skeleton_3d: Skeleton3D, angles: Optional[JointAngles] = None
    ) -> float:
        """Inverted hip Y so bottom-start exercises work with the FSM."""
        kpts = skeleton_3d.to_numpy()
        hip_mid_y = (
            kpts[CocoKeypoints.LEFT_HIP][1] + kpts[CocoKeypoints.RIGHT_HIP][1]
        ) / 2
        ankle_mid_y = (
            kpts[CocoKeypoints.LEFT_ANKLE][1] + kpts[CocoKeypoints.RIGHT_ANKLE][1]
        ) / 2
        return -(hip_mid_y - ankle_mid_y) * 100.0

    def create_rep_counter_config(
        self, config: BiomechanicsConfig
    ) -> HipPositionCounterConfig:
        return HipPositionCounterConfig(
            entry_vel_threshold=3.0,
            bottom_vel_threshold=5.0,
            ascending_vel_threshold=3.0,
            min_depth_cm=20.0,
            standing_return_cm=5.0,
            min_rep_duration_frames=20,
        )

    def get_depth_metric(self, angles: JointAngles) -> float:
        return angles.avg_knee_flexion

    def get_asymmetry_metrics(self, angles: JointAngles) -> Dict[str, float]:
        return {"knee": angles.knee_asymmetry, "hip": angles.hip_asymmetry}

    def get_cue_dict(self) -> Optional[Dict[str, str]]:
        return {
            "range_of_motion": "Stand up fully — hips to lockout",
            "back_rounding": "Brace core — keep spine neutral",
            "bar_path": "Keep the bar close — straight path up",
            "bilateral_asymmetry": "Even out left and right",
        }
