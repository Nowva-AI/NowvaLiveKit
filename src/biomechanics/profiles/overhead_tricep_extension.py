"""
Overhead Tricep Extension Exercise Profile

Arms overhead, elbow flexion/extension. Signal is average
elbow flexion (higher = more flexed = bottom of rep).
"""

import logging
from typing import Callable, Dict, List, Optional

from biomechanics.config import BiomechanicsConfig, HipPositionCounterConfig
from biomechanics.faults.fault_types import FaultRule
from biomechanics.faults.rules.lockout import LockoutRule
from biomechanics.faults.rules.range_of_motion import RangeOfMotionRule
from biomechanics.faults.rules.shoulder_stability import ShoulderStabilityRule
from biomechanics.faults.rules.symmetry import SymmetryRule
from biomechanics.profiles.base import ExerciseProfile
from biomechanics.profiles.registry import register_profile
from biomechanics.utils.types import JointAngles, Skeleton3D

logger = logging.getLogger(__name__)


@register_profile(
    "overhead_tricep_extension", "oh_tricep_extension",
    "overhead_extension", "tricep_extension",
)
class OverheadTricepExtensionProfile(ExerciseProfile):
    """Profile for overhead tricep extensions."""

    name = "overhead_tricep_extension"
    movement_pattern = "tricep_extension"

    def get_feature_extractor(self):
        from biomechanics.ml.upper_body_feature_extractor import UpperBodyFeatureExtractor
        return UpperBodyFeatureExtractor()

    def create_fault_rules(self, config: BiomechanicsConfig) -> List[FaultRule]:
        return [
            LockoutRule(
                joint_getter=lambda a: a.avg_elbow_flexion,
                threshold_deg=15.0,
                at="top",
                fault_message="Lock out at the top — extend fully",
            ),
            RangeOfMotionRule(
                metric_getter=lambda a: a.avg_elbow_flexion,
                target_threshold=100.0,
                min_threshold=70.0,
                direction="max",
                metric_name="tricep extension ROM",
            ),
            ShoulderStabilityRule(
                getter_l=lambda a: a.shoulder_flexion_l,
                getter_r=lambda a: a.shoulder_flexion_r,
                expected_deg=180.0,
                variance_threshold=15.0,
                fault_message="Keep elbows pinned overhead",
            ),
            SymmetryRule(
                left_getter=lambda a: a.elbow_flexion_l,
                right_getter=lambda a: a.elbow_flexion_r,
                joint_name="elbow",
                mild_threshold=8.0,
                moderate_threshold=13.0,
                severe_threshold=18.0,
            ),
        ]

    def get_rep_signal(
        self, skeleton_3d: Skeleton3D, angles: Optional[JointAngles] = None
    ) -> float:
        if angles is None:
            return 0.0
        return angles.avg_elbow_flexion

    def create_rep_counter_config(
        self, config: BiomechanicsConfig
    ) -> HipPositionCounterConfig:
        return HipPositionCounterConfig(
            entry_vel_threshold=40.0,
            bottom_vel_threshold=15.0,
            ascending_vel_threshold=40.0,
            min_depth_cm=70.0,
            standing_return_cm=10.0,
            min_rep_duration_frames=15,
        )

    def get_depth_metric(self, angles: JointAngles) -> float:
        return angles.avg_elbow_flexion

    def get_asymmetry_metrics(self, angles: JointAngles) -> Dict[str, float]:
        return {"elbow": angles.elbow_asymmetry}

    def get_readiness_check(self) -> Optional[Callable]:
        """Arms must be overhead before starting."""
        def _overhead_ready(skeleton_3d: Skeleton3D, angles: JointAngles) -> bool:
            return angles.avg_shoulder_flexion > 150.0
        return _overhead_ready

    def get_cue_dict(self) -> Optional[Dict[str, str]]:
        return {
            "lockout": "Extend fully at the top",
            "range_of_motion": "Lower the weight deeper behind your head",
            "shoulder_stability": "Keep elbows pinned overhead — don't let them drift",
            "bilateral_asymmetry": "Even out left and right",
        }
