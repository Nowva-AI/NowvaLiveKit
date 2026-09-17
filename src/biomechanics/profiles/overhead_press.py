"""
Overhead Press Exercise Profile

Standing barbell or dumbbell press. Signal is average wrist Y-position
relative to shoulders (+ = above shoulder = lockout).
"""

import logging
from typing import Dict, List, Optional

from biomechanics.config import BiomechanicsConfig, HipPositionCounterConfig
from biomechanics.faults.fault_types import FaultRule
from biomechanics.faults.rules.lockout import LockoutRule
from biomechanics.faults.rules.elbow_flare import ElbowFlareRule
from biomechanics.faults.rules.bar_path import BarPathRule
from biomechanics.faults.rules.symmetry import SymmetryRule
from biomechanics.profiles.base import ExerciseProfile
from biomechanics.profiles.registry import register_profile
from biomechanics.utils.types import JointAngles, Skeleton3D

logger = logging.getLogger(__name__)


@register_profile(
    "overhead_press", "ohp", "barbell_press", "military_press",
    "standing_press", "shoulder_press",
)
class OverheadPressProfile(ExerciseProfile):
    """Profile for overhead / military press."""

    name = "overhead_press"
    movement_pattern = "press"

    def get_feature_extractor(self):
        from biomechanics.ml.upper_body_feature_extractor import UpperBodyFeatureExtractor
        return UpperBodyFeatureExtractor()

    def create_fault_rules(self, config: BiomechanicsConfig) -> List[FaultRule]:
        return [
            LockoutRule(
                joint_getter=lambda a: a.avg_elbow_flexion,
                threshold_deg=10.0,
                at="top",
                fault_message="Lock out overhead — fully extend",
            ),
            ElbowFlareRule(
                mild_threshold=75.0,
                moderate_threshold=85.0,
                severe_threshold=95.0,
            ),
            BarPathRule(
                mild_threshold=8.0,
                moderate_threshold=15.0,
                severe_threshold=22.0,
            ),
            SymmetryRule(
                left_getter=lambda a: a.wrist_y_l,
                right_getter=lambda a: a.wrist_y_r,
                joint_name="wrist height",
                mild_threshold=5.0,
                moderate_threshold=10.0,
                severe_threshold=15.0,
            ),
        ]

    def get_rep_signal(
        self, skeleton_3d: Skeleton3D, angles: Optional[JointAngles] = None
    ) -> float:
        """Average wrist Y (shoulder-relative cm). + = above shoulder."""
        if angles is None:
            return 0.0
        return angles.avg_wrist_y

    def create_rep_counter_config(
        self, config: BiomechanicsConfig
    ) -> HipPositionCounterConfig:
        return HipPositionCounterConfig(
            entry_vel_threshold=15.0,
            bottom_vel_threshold=8.0,
            ascending_vel_threshold=15.0,
            min_depth_cm=25.0,
            standing_return_cm=5.0,
            min_rep_duration_frames=15,
        )

    def get_depth_metric(self, angles: JointAngles) -> float:
        return angles.avg_elbow_flexion

    def get_asymmetry_metrics(self, angles: JointAngles) -> Dict[str, float]:
        return {"wrist_height": abs(angles.wrist_y_l - angles.wrist_y_r)}

    def get_cue_dict(self) -> Optional[Dict[str, str]]:
        return {
            "lockout": "Lock out overhead — fully extend arms",
            "elbow_flare": "Tuck elbows slightly at the bottom",
            "bar_path": "Press straight up — keep bar path vertical",
            "bilateral_asymmetry": "Even out left and right",
        }
