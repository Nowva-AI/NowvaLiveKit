"""
Elbow Flare Fault Detection Rule

Detects excessive shoulder abduction at the bottom of a press,
indicating elbows flaring out rather than tucking.
"""

from collections import deque
from typing import Optional

from biomechanics.utils.types import JointAngles, FaultEvent, FaultSeverity
from biomechanics.faults.fault_types import FaultRule, FaultType

# Minimum time between two elbow-flare reports (was 90 frames at 30 fps)
ELBOW_FLARE_COOLDOWN_S = 3.0


class ElbowFlareRule(FaultRule):
    """
    Rule for detecting elbow flare during pressing movements.

    Monitors shoulder abduction during the rep. Excessive abduction
    at the bottom of a press (where wrist_y is lowest) indicates
    elbows flaring out, increasing shoulder injury risk.

    Thresholds (shoulder abduction degrees):
    - Mild: >75°
    - Moderate: >85°
    - Severe: >95°
    """

    def __init__(
        self,
        mild_threshold: float = 75.0,
        moderate_threshold: float = 85.0,
        severe_threshold: float = 95.0,
    ):
        self.mild_threshold = mild_threshold
        self.moderate_threshold = moderate_threshold
        self.severe_threshold = severe_threshold

        self._last_fault_time_s: float = float("-inf")

    @property
    def fault_type(self) -> FaultType:
        return FaultType.ELBOW_FLARE

    def evaluate(
        self,
        angles: JointAngles,
        history: deque,
        in_rep: bool = False,
        rep_number: int = 0,
    ) -> Optional[FaultEvent]:
        if not in_rep:
            return None

        if angles.timestamp - self._last_fault_time_s < ELBOW_FLARE_COOLDOWN_S:
            return None

        # Check both sides
        max_abduction = max(
            angles.shoulder_abduction_l,
            angles.shoulder_abduction_r,
        )

        if max_abduction < self.mild_threshold:
            return None

        severity, score = self._get_severity(
            max_abduction,
            {
                "mild": self.mild_threshold,
                "moderate": self.moderate_threshold,
                "severe": self.severe_threshold,
            },
        )

        if severity == FaultSeverity.NONE:
            return None

        self._last_fault_time_s = angles.timestamp

        flared_side = (
            "left"
            if angles.shoulder_abduction_l > angles.shoulder_abduction_r
            else "right"
        )

        return self._create_fault_event(
            severity=severity,
            severity_score=score,
            message="Tuck elbows — reduce flare",
            angles=angles,
            rep_number=rep_number,
            details={
                "max_abduction": max_abduction,
                "abduction_l": angles.shoulder_abduction_l,
                "abduction_r": angles.shoulder_abduction_r,
                "flared_side": flared_side,
            },
        )
