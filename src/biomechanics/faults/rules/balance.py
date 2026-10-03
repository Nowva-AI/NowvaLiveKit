"""Balance rule: the load drifting off midfoot at the bottom, judged once per rep.

The load is the bar when one is detected and the body's centre of mass
otherwise; the offset is a fraction of the ankle-to-toe length. Forward drift
(onto the toes) is the common, coachable fault; drifting back onto the heels is
reported with a wider band.
"""

from __future__ import annotations

import math
from typing import Optional

from biomechanics.faults.fault_types import FaultType, RepFaultRule
from biomechanics.utils.types import FaultEvent, JointAngles


class BalanceRule(RepFaultRule):
    def __init__(
        self,
        forward_mild: float = 0.20,
        forward_moderate: float = 0.30,
        forward_severe: float = 0.40,
        backward_mild: float = 0.30,
        backward_moderate: float = 0.40,
        backward_severe: float = 0.50,
    ) -> None:
        self.forward = {"mild": forward_mild, "moderate": forward_moderate, "severe": forward_severe}
        self.backward = {"mild": backward_mild, "moderate": backward_moderate, "severe": backward_severe}

    @property
    def fault_type(self) -> FaultType:
        return FaultType.BALANCE

    def judge_rep(self, features, reference, angles: JointAngles) -> Optional[FaultEvent]:
        offset = features.balance_ratio
        if not math.isfinite(offset):
            return None
        direction = "forward" if offset > 0 else "backward"
        thresholds = self.forward if offset > 0 else self.backward
        return self._rep_fault(
            abs(offset),
            thresholds,
            angles,
            features.rep_number,
            unit="ratio",
            phase="bottom",
            is_drift=reference.is_drift("balance_abs", self.forward["mild"]),
            direction=direction,
            load="bar" if features.bar_detected else "body",
        )
