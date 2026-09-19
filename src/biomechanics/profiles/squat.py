"""
Squat Exercise Profile

Extracts the current hardcoded squat behavior from RuleEngine._create_rules()
and the pipeline's hip-position signal computation into a pluggable profile.
This is a pure extraction refactor — zero behavioral change.
"""

import logging
import math
import os
from typing import Dict, List, Optional

from biomechanics.config import BiomechanicsConfig, FaultsConfig
from biomechanics.faults.fault_types import FaultRule
from biomechanics.faults.rules.depth import DepthRule
from biomechanics.faults.rules.symmetry import SymmetryRule
from biomechanics.faults.rules.forward_lean import ForwardLeanRule
from biomechanics.faults.rules.knee_valgus import KneeValgusRule
from biomechanics.faults.rules.bar_tilt_asymmetry import BarTiltAsymmetryRule
from biomechanics.faults.rules.heel_rise import HeelRiseRule
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
    """Profile for all squat variants.

    Reproduces the exact behavior that was previously hardcoded in
    RuleEngine._create_rules() and pipeline.py lines 314-318.
    """

    name = "squat"
    movement_pattern = "squat"

    def create_fault_rules(self, config: BiomechanicsConfig) -> List[FaultRule]:
        """Create the squat fault rules with config thresholds.

        Identical to the previous RuleEngine._create_rules() implementation.
        """
        fc = config.faults
        bt = config.barbell_tracking
        return [
            DepthRule(
                quarter_threshold=60.0,
                half_threshold=fc.depth.parallel,
                parallel_threshold=fc.depth.below_parallel,
            ),
            SymmetryRule(
                mild_threshold=fc.bilateral_asymmetry.mild,
                moderate_threshold=fc.bilateral_asymmetry.moderate,
                severe_threshold=fc.bilateral_asymmetry.severe,
            ),
            # Fires only when a barbell detection is present (back/front squats).
            # Silently no-ops for bodyweight squats — safe to register unconditionally.
            BarTiltAsymmetryRule(
                bar_length_m=bt.bar_length_m,
                mild_deg=bt.tilt_asym_mild_deg,
                moderate_deg=bt.tilt_asym_moderate_deg,
                severe_deg=bt.tilt_asym_severe_deg,
                mild_cm=bt.tilt_asym_mild_cm,
                moderate_cm=bt.tilt_asym_moderate_cm,
                severe_cm=bt.tilt_asym_severe_cm,
            ),
            ForwardLeanRule(
                mild_threshold=fc.forward_lean.mild,
                moderate_threshold=fc.forward_lean.moderate,
                severe_threshold=fc.forward_lean.severe,
            ),
            KneeValgusRule(
                **self._valgus_thresholds(fc),
            ),
            # Reads the FootState side channel; no-ops without one (single-camera mode).
            HeelRiseRule(
                mild_cm=fc.heel_rise.mild_cm,
                moderate_cm=fc.heel_rise.moderate_cm,
                severe_cm=fc.heel_rise.severe_cm,
                cooldown_s=fc.heel_rise.cooldown_s,
                min_rise_duration_s=fc.heel_rise.min_rise_duration_s,
            ),
        ]

    @staticmethod
    def _valgus_thresholds(fc: FaultsConfig) -> dict[str, float]:
        multi = os.getenv("NOWVA_MULTI_CAMERA", "false").lower() == "true"
        kv = fc.knee_valgus
        # Fallback (hip adduction) is always 3D-scale — its formula doesn't
        # change with capture mode, unlike the primary toe/FPPA metric.
        fallback = dict(
            fallback_mild_threshold=kv.mild,
            fallback_moderate_threshold=kv.moderate,
            fallback_severe_threshold=kv.severe,
        )
        if multi:
            return dict(mild_threshold=kv.mild, moderate_threshold=kv.moderate, severe_threshold=kv.severe, **fallback)
        return dict(
            mild_threshold=kv.mild_2d, moderate_threshold=kv.moderate_2d, severe_threshold=kv.severe_2d, **fallback
        )

    def get_rep_signal(
        self, skeleton_3d: Skeleton3D, angles: Optional[JointAngles] = None
    ) -> float:
        """Compute hip vertical position relative to ankle (cm).

        Identical to the previous pipeline.py lines 314-318.
        Convention: more negative = standing, less negative = squat bottom.
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

    def record_calibration_frame(self, angles: JointAngles, state: Dict) -> None:
        """Track peak squat-specific values during calibration reps."""
        # Initialise state on first call
        if "baseline_set" not in state:
            state["baseline_set"] = True
            state["peak_trunk_flexion"] = 180.0
            state["peak_asymmetry"] = 0.0
            state["current_rep_peak_adduction"] = 0.0
            state["rep_peak_hip_adductions"] = []
            state["current_rep_peak_valgus"] = 0.0
            state["rep_peak_knee_valgus"] = []
            state["toe_available"] = False

        state["peak_trunk_flexion"] = min(
            state["peak_trunk_flexion"], angles.trunk_flexion
        )

        frame_adduction = max(
            abs(angles.hip_adduction_l), abs(angles.hip_adduction_r)
        )
        state["current_rep_peak_adduction"] = max(
            state["current_rep_peak_adduction"], frame_adduction
        )

        foot_conf = min(angles.foot_confidence_l, angles.foot_confidence_r)
        if foot_conf >= 0.3:
            state["toe_available"] = True
            frame_valgus = max(angles.knee_valgus_l, angles.knee_valgus_r)
            state["current_rep_peak_valgus"] = max(
                state["current_rep_peak_valgus"], frame_valgus
            )

        state["peak_asymmetry"] = max(
            state["peak_asymmetry"],
            abs(angles.hip_flexion_l - angles.hip_flexion_r),
            abs(angles.knee_flexion_l - angles.knee_flexion_r),
        )

    def apply_baseline(self, rules: List[FaultRule], state: Dict) -> None:
        """Adjust squat rule thresholds after calibration."""
        if not state:
            return

        # Average per-rep peaks
        if state.get("current_rep_peak_adduction", 0) > 0:
            state["rep_peak_hip_adductions"].append(state["current_rep_peak_adduction"])
        if state.get("current_rep_peak_valgus", 0) > 0:
            state["rep_peak_knee_valgus"].append(state["current_rep_peak_valgus"])

        peak_hip_adduction = 0.0
        if state["rep_peak_hip_adductions"]:
            peak_hip_adduction = (
                sum(state["rep_peak_hip_adductions"])
                / len(state["rep_peak_hip_adductions"])
            )

        for rule in rules:
            ft = rule.fault_type
            ft_val = ft.value if hasattr(ft, "value") else ft

            if ft_val == "forward_lean" and hasattr(rule, "apply_baseline"):
                peak = state["peak_trunk_flexion"]
                rule.apply_baseline(peak)
                logger.info(
                    "[SQUAT] Forward lean baseline: peak=%.1f° → %.1f/%.1f/%.1f",
                    peak, rule.mild_threshold, rule.moderate_threshold,
                    rule.severe_threshold,
                )

            elif ft_val == "knee_valgus" and hasattr(rule, "mild_threshold"):
                if state["toe_available"] and state["rep_peak_knee_valgus"]:
                    peak = (
                        sum(state["rep_peak_knee_valgus"])
                        / len(state["rep_peak_knee_valgus"])
                    )
                    rule.mild_threshold = peak + 5.0
                    rule.moderate_threshold = peak + 10.0
                    rule.severe_threshold = peak + 15.0
                    logger.info(
                        "[SQUAT] Knee valgus baseline (toe): peak=%.1f° → %.1f/%.1f/%.1f",
                        peak, rule.mild_threshold, rule.moderate_threshold,
                        rule.severe_threshold,
                    )
                else:
                    rule.mild_threshold = peak_hip_adduction + 5.0
                    rule.moderate_threshold = peak_hip_adduction + 10.0
                    rule.severe_threshold = peak_hip_adduction + 15.0
                    logger.info(
                        "[SQUAT] Knee valgus baseline (hip): peak=%.1f° → %.1f/%.1f/%.1f",
                        peak_hip_adduction, rule.mild_threshold,
                        rule.moderate_threshold, rule.severe_threshold,
                    )

            elif ft_val == "bilateral_asymmetry" and hasattr(rule, "mild_threshold"):
                peak = state["peak_asymmetry"]
                rule.mild_threshold = max(rule.mild_threshold, peak + 5.0)
                rule.moderate_threshold = max(rule.moderate_threshold, peak + 10.0)
                rule.severe_threshold = max(rule.severe_threshold, peak + 15.0)
                logger.info(
                    "[SQUAT] Symmetry baseline: peak=%.1f° → %.1f/%.1f/%.1f",
                    peak, rule.mild_threshold, rule.moderate_threshold,
                    rule.severe_threshold,
                )
