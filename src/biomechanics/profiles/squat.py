"""
Squat Exercise Profile

The squat's fault rules and its rep signal (hip height above the ankles).
Faults are judged once per rep from the shared rep features
(analysis.rep_features), against absolute standards; a personal baseline only
ever flags drift, never moves a threshold.
"""

import logging
import math
from typing import List, Optional

from biomechanics.config import BiomechanicsConfig
from biomechanics.faults.fault_types import FaultRule
from biomechanics.faults.rules.balance import BalanceRule
from biomechanics.faults.rules.bar_tilt_asymmetry import BarTiltAsymmetryRule
from biomechanics.faults.rules.depth import DepthRule
from biomechanics.faults.rules.depth_drift import DepthDriftRule
from biomechanics.faults.rules.descent_control import DescentControlRule
from biomechanics.faults.rules.foot_placement import FootPlacementRule
from biomechanics.faults.rules.heel_rise import HeelRiseRule
from biomechanics.faults.rules.hip_shift import HipShiftRule
from biomechanics.faults.rules.hip_shoot import HipShootRule
from biomechanics.faults.rules.knee_tracking import KneeTrackingRule
from biomechanics.faults.rules.standing_lockout import StandingLockoutRule
from biomechanics.faults.rules.velocity_loss import VelocityLossRule
from biomechanics.profiles.base import ExerciseProfile
from biomechanics.profiles.registry import register_profile
from biomechanics.utils.types import CocoKeypoints, JointAngles, Skeleton3D

logger = logging.getLogger(__name__)

# Keypoints whose absence makes the hip-to-ankle signal meaningless.
_REP_SIGNAL_KEYPOINTS = (
    CocoKeypoints.LEFT_HIP, CocoKeypoints.RIGHT_HIP,
    CocoKeypoints.LEFT_ANKLE, CocoKeypoints.RIGHT_ANKLE,
)
MIN_REP_SIGNAL_CONFIDENCE = 0.0


@register_profile(
    "squat",
    "back_squat",
    "front_squat",
    "goblet_squat",
    "bodyweight_squat",
    "barbell_back_squat",
    "barbell_front_squat",
)
class SquatProfile(ExerciseProfile):
    """Profile for all squat variants."""

    name = "squat"
    movement_pattern = "squat"

    def create_fault_rules(self, config: BiomechanicsConfig) -> List[FaultRule]:
        """Squat fault rules, listed in cue priority order (ties on one frame go first-listed)."""
        fc = config.faults
        bt = config.barbell_tracking
        return [
            KneeTrackingRule(
                mild_threshold=fc.knee_valgus.mild,
                moderate_threshold=fc.knee_valgus.moderate,
                severe_threshold=fc.knee_valgus.severe,
            ),
            HipShootRule(
                mild_threshold=fc.hip_shoot.mild,
                moderate_threshold=fc.hip_shoot.moderate,
                severe_threshold=fc.hip_shoot.severe,
            ),
            # Reads the FootState side channel; no-ops without one (single-camera mode).
            HeelRiseRule(
                mild_cm=fc.heel_rise.mild_cm,
                moderate_cm=fc.heel_rise.moderate_cm,
                severe_cm=fc.heel_rise.severe_cm,
                cooldown_s=fc.heel_rise.cooldown_s,
                min_rise_duration_s=fc.heel_rise.min_rise_duration_s,
            ),
            BalanceRule(
                forward_mild=fc.balance.forward_mild,
                forward_moderate=fc.balance.forward_moderate,
                forward_severe=fc.balance.forward_severe,
                backward_mild=fc.balance.backward_mild,
                backward_moderate=fc.balance.backward_moderate,
                backward_severe=fc.balance.backward_severe,
            ),
            HipShiftRule(
                mild_threshold=fc.hip_shift.mild,
                moderate_threshold=fc.hip_shift.moderate,
                severe_threshold=fc.hip_shift.severe,
            ),
            # Fires only when a barbell detection is present (back/front squats).
            BarTiltAsymmetryRule(
                bar_length_m=bt.bar_length_m,
                mild_deg=bt.tilt_asym_mild_deg,
                moderate_deg=bt.tilt_asym_moderate_deg,
                severe_deg=bt.tilt_asym_severe_deg,
                mild_cm=bt.tilt_asym_mild_cm,
                moderate_cm=bt.tilt_asym_moderate_cm,
                severe_cm=bt.tilt_asym_severe_cm,
            ),
            DepthRule(
                tolerance_ratio=fc.depth.tolerance_ratio,
                moderate_deficit_ratio=fc.depth.moderate_deficit_ratio,
                severe_deficit_ratio=fc.depth.severe_deficit_ratio,
            ),
            FootPlacementRule(
                stagger_mild=fc.foot_placement.stagger_mild,
                stagger_moderate=fc.foot_placement.stagger_moderate,
                stagger_severe=fc.foot_placement.stagger_severe,
                flare_mild_deg=fc.foot_placement.flare_mild_deg,
                flare_moderate_deg=fc.foot_placement.flare_moderate_deg,
                flare_severe_deg=fc.foot_placement.flare_severe_deg,
            ),
            StandingLockoutRule(
                mild_threshold=fc.lockout.mild,
                moderate_threshold=fc.lockout.moderate,
                severe_threshold=fc.lockout.severe,
            ),
            DescentControlRule(
                mild_seconds=fc.descent_control.mild_seconds,
                moderate_seconds=fc.descent_control.moderate_seconds,
                severe_seconds=fc.descent_control.severe_seconds,
            ),
            DepthDriftRule(
                mild_threshold=fc.depth_drift.mild,
                moderate_threshold=fc.depth_drift.moderate,
                severe_threshold=fc.depth_drift.severe,
            ),
            VelocityLossRule(
                mild_pct=fc.velocity_loss.mild_pct,
                moderate_pct=fc.velocity_loss.moderate_pct,
                severe_pct=fc.velocity_loss.severe_pct,
            ),
        ]

    def get_rep_signal(
        self, skeleton_3d: Skeleton3D, angles: Optional[JointAngles] = None
    ) -> float:
        """Hip vertical position relative to ankle (cm), in the Y-down frame.

        More negative = standing, less negative = squat bottom.
        """
        keypoints = skeleton_3d.keypoints
        for idx in _REP_SIGNAL_KEYPOINTS:
            if keypoints[idx].confidence <= MIN_REP_SIGNAL_CONFIDENCE:
                # A missing hip or ankle sits at the origin after re-centring;
                # a NaN signal is ignored by the counter, a fake one starts a rep.
                return math.nan
        kpts = skeleton_3d.to_numpy()
        hip_mid_y = (
            kpts[CocoKeypoints.LEFT_HIP][1] + kpts[CocoKeypoints.RIGHT_HIP][1]
        ) / 2
        ankle_mid_y = (
            kpts[CocoKeypoints.LEFT_ANKLE][1] + kpts[CocoKeypoints.RIGHT_ANKLE][1]
        ) / 2
        return (hip_mid_y - ankle_mid_y) * 100.0
