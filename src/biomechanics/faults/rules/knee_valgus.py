"""
Knee Valgus Fault Detection Rule

Detects knee cave (valgus) per frame at the bottom of a rep from the mode-aware
knee valgus metric. Used by the lunge; the squat judges whole reps instead
(knee_tracking.py). Without visible feet it says nothing.
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
    Knee valgus from the knees-over-toes metric (knee_valgus_l/r, positive =
    knee medial). Frames where the feet are not seen well enough are skipped:
    the former hip-adduction fallback measured the thigh against the world
    axis, so it read toe-in as valgus and missed real cave with toe-out.
    """

    def __init__(
        self,
        mild_threshold: float = 12.0,
        moderate_threshold: float = 17.0,
        severe_threshold: float = 24.0,
    ):
        self.mild_threshold = mild_threshold
        self.moderate_threshold = moderate_threshold
        self.severe_threshold = severe_threshold

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
        Evaluate knee valgus at the bottom of a rep.

        Only fires when valgus exceeds the mild threshold, and skips frames
        where the feet are not seen or either side's metric is NaN.
        """
        if not in_rep or self._phase != "bottom":
            return None

        # Cooldown between fault reports
        if angles.timestamp - self._last_fault_time_s < KNEE_VALGUS_COOLDOWN_S:
            return None

        if min(angles.foot_confidence_l, angles.foot_confidence_r) < FOOT_CONFIDENCE_THRESHOLD:
            return None
        valgus_l = angles.knee_valgus_l
        valgus_r = angles.knee_valgus_r
        mild, moderate, severe = self.mild_threshold, self.moderate_threshold, self.severe_threshold

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
            },
        )
