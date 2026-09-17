"""
Knee Valgus Fault Detection Rule

Detects knee cave (valgus) using the mode-aware knee valgus metric (primary),
with fallback to hip adduction angle when foot landmarks are unavailable.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional

from biomechanics.utils.types import JointAngles, FaultEvent, FaultSeverity
from biomechanics.faults.fault_types import FaultRule, FaultType, FAULT_MESSAGES

# Minimum foot landmark confidence to use the toe-based valgus metric
FOOT_CONFIDENCE_THRESHOLD = 0.3
# Minimum time between two valgus reports (was 30 frames at 30 fps)
KNEE_VALGUS_COOLDOWN_S = 1.0
# Both sides within this many degrees of each other report as "both"
BILATERAL_AGREEMENT_DEG = 2.0


class KneeValgusRule(FaultRule):
    """
    Knee valgus detection using the knee valgus metric with hip adduction fallback.

    Primary metric (when foot landmarks available):
    - knee_valgus_l/r from the pipeline's valgus estimator (2D FPPA or 3D
      knee deviation). Positive = knee medial (valgus).

    Fallback metric (when foot landmarks unavailable):
    - Hip adduction angle (thigh medial deviation from sagittal plane).
      Used when foot_confidence is below FOOT_CONFIDENCE_THRESHOLD.

    The primary metric's scale depends on capture mode (2D FPPA vs. 3D
    deviation), but hip adduction's scale does not — so the fallback has its
    own threshold triple, defaulting to the primary one when not given.

    Thresholds are not scaled by body proportions: the metric has no valid
    hip-width dependence (C8).
    """

    def __init__(
        self,
        mild_threshold: float = 12.0,
        moderate_threshold: float = 17.0,
        severe_threshold: float = 24.0,
        fallback_mild_threshold: Optional[float] = None,
        fallback_moderate_threshold: Optional[float] = None,
        fallback_severe_threshold: Optional[float] = None,
    ):
        self.mild_threshold = mild_threshold
        self.moderate_threshold = moderate_threshold
        self.severe_threshold = severe_threshold
        self.fallback_mild_threshold = (
            fallback_mild_threshold if fallback_mild_threshold is not None else mild_threshold
        )
        self.fallback_moderate_threshold = (
            fallback_moderate_threshold if fallback_moderate_threshold is not None else moderate_threshold
        )
        self.fallback_severe_threshold = (
            fallback_severe_threshold if fallback_severe_threshold is not None else severe_threshold
        )

        self._last_fault_time_s: float = float("-inf")

    @property
    def fault_type(self) -> FaultType:
        return FaultType.KNEE_VALGUS

    def evaluate(
        self,
        angles: JointAngles,
        history: deque,
        in_rep: bool = False,
        rep_number: int = 0,
    ) -> Optional[FaultEvent]:
        """
        Evaluate for knee valgus using the toe-based metric or hip adduction fallback.

        Only fires at the bottom of a rep, when valgus exceeds the mild
        threshold, and skips frames where either side's metric is NaN.
        """
        if not in_rep or self._phase != "bottom":
            return None

        # Cooldown between fault reports
        if angles.timestamp - self._last_fault_time_s < KNEE_VALGUS_COOLDOWN_S:
            return None

        # Prefer toe-based valgus when both feet have sufficient confidence
        foot_conf = min(angles.foot_confidence_l, angles.foot_confidence_r)

        if foot_conf >= FOOT_CONFIDENCE_THRESHOLD:
            valgus_l = angles.knee_valgus_l
            valgus_r = angles.knee_valgus_r
            metric_source = "toe"
            mild, moderate, severe = self.mild_threshold, self.moderate_threshold, self.severe_threshold
        else:
            valgus_l = angles.hip_adduction_l
            valgus_r = angles.hip_adduction_r
            metric_source = "hip_adduction"
            mild, moderate, severe = (
                self.fallback_mild_threshold,
                self.fallback_moderate_threshold,
                self.fallback_severe_threshold,
            )

        if math.isnan(valgus_l) or math.isnan(valgus_r):
            return None

        max_valgus = max(valgus_l, valgus_r)

        if max_valgus < mild:
            return None

        severity, score = self._get_severity(
            max_valgus,
            {"mild": mild, "moderate": moderate, "severe": severe},
        )

        if severity == FaultSeverity.NONE:
            return None

        self._last_fault_time_s = angles.timestamp

        # Determine which side has worse valgus (higher positive = more cave)
        affected_side = "left" if valgus_l > valgus_r else "right"
        if abs(valgus_l - valgus_r) < BILATERAL_AGREEMENT_DEG:
            affected_side = "both"

        message_key = severity.value
        message = FAULT_MESSAGES[FaultType.KNEE_VALGUS].get(
            message_key, "Knee valgus detected"
        )

        return self._create_fault_event(
            severity=severity,
            severity_score=score,
            message=message,
            angles=angles,
            rep_number=rep_number,
            details={
                "knee_valgus_l": valgus_l,
                "knee_valgus_r": valgus_r,
                "max_valgus": max_valgus,
                "affected_side": affected_side,
                "metric_source": metric_source,
                "hip_adduction_l": angles.hip_adduction_l,
                "hip_adduction_r": angles.hip_adduction_r,
            },
        )
