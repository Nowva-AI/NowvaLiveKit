"""
Barbell Row Exercise Profile

Bent-over barbell row. Signal is average elbow flexion
(0 = arms hanging, max at top of row).
"""

import logging
from typing import Callable, Dict, List, Optional

from biomechanics.config import BiomechanicsConfig, HipPositionCounterConfig
from biomechanics.faults.fault_types import FaultRule
from biomechanics.faults.rules.range_of_motion import RangeOfMotionRule
from biomechanics.faults.rules.back_rounding import BackRoundingRule
from biomechanics.faults.rules.trunk_stability import TrunkStabilityRule
from biomechanics.faults.rules.symmetry import SymmetryRule
from biomechanics.profiles.base import ExerciseProfile
from biomechanics.profiles.registry import register_profile
from biomechanics.utils.types import JointAngles, Skeleton3D

logger = logging.getLogger(__name__)


@register_profile("barbell_row", "bent_over_row", "pendlay_row")
class BarbellRowProfile(ExerciseProfile):
    """Profile for barbell row variants."""

    name = "barbell_row"
    movement_pattern = "row"

    def get_feature_extractor(self):
        from biomechanics.ml.upper_body_feature_extractor import UpperBodyFeatureExtractor
        return UpperBodyFeatureExtractor()

    def create_fault_rules(self, config: BiomechanicsConfig) -> List[FaultRule]:
        return [
            RangeOfMotionRule(
                metric_getter=lambda a: a.avg_elbow_flexion,
                target_threshold=90.0,
                min_threshold=60.0,
                direction="max",
                metric_name="row ROM",
            ),
            BackRoundingRule(
                mild_threshold=8.0,
                moderate_threshold=15.0,
                severe_threshold=25.0,
            ),
            TrunkStabilityRule(
                mild_threshold=8.0,
                moderate_threshold=15.0,
                severe_threshold=22.0,
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
        """Average elbow flexion (0 = hanging, higher = rowing up)."""
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
            min_depth_cm=40.0,
            standing_return_cm=10.0,
            min_rep_duration_frames=12,
        )

    def get_depth_metric(self, angles: JointAngles) -> float:
        return angles.avg_elbow_flexion

    def get_asymmetry_metrics(self, angles: JointAngles) -> Dict[str, float]:
        return {"elbow": angles.elbow_asymmetry}

    def get_readiness_check(self) -> Optional[Callable]:
        """Bent-over row starts in a hinged position, not standing."""
        def _bent_over_ready(skeleton_3d: Skeleton3D, angles: JointAngles) -> bool:
            return 60.0 <= angles.trunk_flexion <= 140.0
        return _bent_over_ready

    def get_cue_dict(self) -> Optional[Dict[str, str]]:
        return {
            "range_of_motion": "Pull higher — squeeze shoulder blades",
            "back_rounding": "Keep spine neutral — don't round",
            "trunk_stability": "Keep torso angle fixed — no rocking",
            "bilateral_asymmetry": "Pull evenly on both sides",
        }
