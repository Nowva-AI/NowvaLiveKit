"""
Shoulder Stability Fault Detection Rule

Detects the upper arm drifting from its expected position during
isolation exercises (curls, tricep extensions, skull crushers).
"""

from collections import deque
from typing import Callable, Optional

from biomechanics.utils.types import JointAngles, FaultEvent, FaultSeverity
from biomechanics.faults.fault_types import FaultRule, FaultType

# Minimum time between two shoulder-stability reports (was 60 frames at 30 fps)
SHOULDER_STABILITY_COOLDOWN_S = 2.0


class ShoulderStabilityRule(FaultRule):
    """
    Rule for detecting shoulder instability / elbow drift.

    Monitors shoulder flexion variance during the rep. Each exercise
    defines an expected shoulder flexion angle (e.g. 0° for curl,
    180° for OH tricep ext, 90° for skull crusher). If the actual
    shoulder flexion drifts beyond ``variance_threshold`` from the
    expected angle, a fault fires.

    Args:
        getter_l: Extracts left shoulder flexion from JointAngles.
        getter_r: Extracts right shoulder flexion from JointAngles.
        expected_deg: The expected shoulder flexion angle for this exercise.
        variance_threshold: Maximum allowed deviation from expected (degrees).
        fault_message: Custom cue string.
    """

    def __init__(
        self,
        getter_l: Callable[[JointAngles], float] = lambda a: a.shoulder_flexion_l,
        getter_r: Callable[[JointAngles], float] = lambda a: a.shoulder_flexion_r,
        expected_deg: float = 0.0,
        variance_threshold: float = 15.0,
        fault_message: str = "Keep upper arms stable",
    ):
        self._getter_l = getter_l
        self._getter_r = getter_r
        self._expected_deg = expected_deg
        self._variance_threshold = variance_threshold
        self._fault_message = fault_message

        self._last_fault_time_s: float = float("-inf")

    @property
    def fault_type(self) -> FaultType:
        return FaultType.SHOULDER_STABILITY

    def evaluate(
        self,
        angles: JointAngles,
        history: deque,
        in_rep: bool = False,
        rep_number: int = 0,
    ) -> Optional[FaultEvent]:
        if not in_rep:
            return None

        if angles.timestamp - self._last_fault_time_s < SHOULDER_STABILITY_COOLDOWN_S:
            return None

        val_l = self._getter_l(angles)
        val_r = self._getter_r(angles)

        drift_l = abs(val_l - self._expected_deg)
        drift_r = abs(val_r - self._expected_deg)
        max_drift = max(drift_l, drift_r)

        if max_drift < self._variance_threshold:
            return None

        self._last_fault_time_s = angles.timestamp

        excess = max_drift - self._variance_threshold
        if excess > 20.0:
            severity = FaultSeverity.SEVERE
            score = 2.5
        elif excess > 10.0:
            severity = FaultSeverity.MODERATE
            score = 1.5 + (excess / 20.0)
        else:
            severity = FaultSeverity.MILD
            score = 0.5 + (excess / 10.0)

        drifted_side = "left" if drift_l > drift_r else "right"

        return self._create_fault_event(
            severity=severity,
            severity_score=score,
            message=self._fault_message,
            angles=angles,
            rep_number=rep_number,
            details={
                "drift_l": drift_l,
                "drift_r": drift_r,
                "max_drift": max_drift,
                "expected_deg": self._expected_deg,
                "drifted_side": drifted_side,
            },
        )
