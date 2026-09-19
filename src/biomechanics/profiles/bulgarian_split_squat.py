"""
Bulgarian Split Squat Exercise Profile

Single-leg squat with rear foot elevated. Similar to lunge
but with tighter trunk stability requirements.
"""

import logging
from typing import Dict, List, Optional

from biomechanics.config import BiomechanicsConfig, HipPositionCounterConfig
from biomechanics.faults.fault_types import FaultRule
from biomechanics.faults.rules.range_of_motion import RangeOfMotionRule
from biomechanics.faults.rules.knee_valgus import KneeValgusRule
from biomechanics.faults.rules.trunk_stability import TrunkStabilityRule
from biomechanics.faults.rules.symmetry import SymmetryRule
from biomechanics.profiles.base import ExerciseProfile
from biomechanics.profiles.registry import register_profile
from biomechanics.utils.types import JointAngles, Skeleton3D

logger = logging.getLogger(__name__)


@register_profile("bulgarian_split_squat", "bss", "rear_foot_elevated_split_squat")
class BulgarianSplitSquatProfile(ExerciseProfile):
    """Profile for Bulgarian split squat."""

    name = "bulgarian_split_squat"
    movement_pattern = "lunge"

    def create_fault_rules(self, config: BiomechanicsConfig) -> List[FaultRule]:
        return [
            RangeOfMotionRule(
                metric_getter=lambda a: max(a.knee_flexion_l, a.knee_flexion_r),
                target_threshold=95.0,
                min_threshold=70.0,
                direction="max",
                metric_name="split squat depth",
            ),
            KneeValgusRule(
                mild_threshold=8.0,
                moderate_threshold=13.0,
                severe_threshold=18.0,
            ),
            TrunkStabilityRule(
                mild_threshold=8.0,
                moderate_threshold=15.0,
                severe_threshold=22.0,
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
            min_depth_cm=40.0,
            standing_return_cm=10.0,
            min_rep_duration_frames=18,
        )

    def get_depth_metric(self, angles: JointAngles) -> float:
        return max(angles.knee_flexion_l, angles.knee_flexion_r)

    def get_asymmetry_metrics(self, angles: JointAngles) -> Dict[str, float]:
        return {"knee": angles.knee_asymmetry}

    def get_cue_dict(self) -> Optional[Dict[str, str]]:
        return {
            "range_of_motion": "Go deeper — rear knee toward floor",
            "knee_valgus": "Push front knee out over toes",
            "trunk_stability": "Keep torso stable — no rocking",
            "bilateral_asymmetry": "Balance both sides evenly",
        }
