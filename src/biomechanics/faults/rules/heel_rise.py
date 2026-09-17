"""Heel-rise fault rule: a planted foot's ankle lifting off its floor reference during a rep.

Reads the FootState side channel (multi-camera only) and is a no-op without a valid one.
"""

from __future__ import annotations

import math
from collections import deque

from biomechanics.faults.fault_types import FaultRule, FaultType
from biomechanics.utils.types import FaultEvent, FaultSeverity, JointAngles

MILD_THRESHOLD_CM = 1.5
MODERATE_THRESHOLD_CM = 3.0
SEVERE_THRESHOLD_CM = 5.0
COOLDOWN_S = 2.0
# About eight consecutive frames at 30 fps: noise excursions of the 0.2 s contact window must not fire the rule
# (measured: 0 false events per clean rep at 1.5 cm triangulation noise, 3.4 cm rises still caught every rep).
MIN_RISE_DURATION_S = 0.23
# Both sides are reported when the two rises agree within this.
SIDE_SYMMETRY_CM = 1.0

HEEL_RISE_MESSAGES: dict[FaultSeverity, str] = {
    FaultSeverity.MILD: "Slight heel lift — push through your midfoot",
    FaultSeverity.MODERATE: "Keep your heels down — push through midfoot",
    FaultSeverity.SEVERE: "Heels coming up — drive them into the floor",
}


def _rise_or_zero(rise_cm: float) -> float:
    return 0.0 if math.isnan(rise_cm) else rise_cm


class HeelRiseRule(FaultRule):
    """Fires during a rep on the larger of the two ankle rises once it has persisted briefly."""

    def __init__(
        self,
        mild_cm: float = MILD_THRESHOLD_CM,
        moderate_cm: float = MODERATE_THRESHOLD_CM,
        severe_cm: float = SEVERE_THRESHOLD_CM,
        cooldown_s: float = COOLDOWN_S,
        min_rise_duration_s: float = MIN_RISE_DURATION_S,
    ) -> None:
        self.mild_cm = mild_cm
        self.moderate_cm = moderate_cm
        self.severe_cm = severe_cm
        self.cooldown_s = cooldown_s
        self.min_rise_duration_s = min_rise_duration_s
        self._last_fault_timestamp = -math.inf
        self._rise_start_timestamp: float | None = None

    @property
    def fault_type(self) -> FaultType:
        return FaultType.HEEL_RISE

    def reset(self) -> None:
        self._last_fault_timestamp = -math.inf
        self._rise_start_timestamp = None

    def evaluate(
        self,
        angles: JointAngles,
        history: deque,
        in_rep: bool = False,
        rep_number: int = 0,
    ) -> FaultEvent | None:
        foot_state = self._foot_state
        if foot_state is None or not foot_state.valid or not in_rep:
            self._rise_start_timestamp = None
            return None

        rise_l_cm = _rise_or_zero(foot_state.heel_rise_l_cm)
        rise_r_cm = _rise_or_zero(foot_state.heel_rise_r_cm)
        max_rise_cm = max(rise_l_cm, rise_r_cm)
        if max_rise_cm < self.mild_cm:
            self._rise_start_timestamp = None
            return None

        if self._rise_start_timestamp is None:
            self._rise_start_timestamp = angles.timestamp
        if angles.timestamp - self._rise_start_timestamp < self.min_rise_duration_s:
            return None
        if angles.timestamp - self._last_fault_timestamp < self.cooldown_s:
            return None

        severity, score = self._get_severity(
            max_rise_cm,
            {"mild": self.mild_cm, "moderate": self.moderate_cm, "severe": self.severe_cm},
        )
        if severity == FaultSeverity.NONE:
            return None

        self._last_fault_timestamp = angles.timestamp
        if abs(rise_l_cm - rise_r_cm) < SIDE_SYMMETRY_CM:
            affected_side = "both"
        else:
            affected_side = "left" if rise_l_cm > rise_r_cm else "right"

        return self._create_fault_event(
            severity=severity,
            severity_score=score,
            message=HEEL_RISE_MESSAGES[severity],
            angles=angles,
            rep_number=rep_number,
            details={
                "heel_rise_l_cm": rise_l_cm,
                "heel_rise_r_cm": rise_r_cm,
                "max_heel_rise_cm": max_rise_cm,
                "affected_side": affected_side,
            },
        )
