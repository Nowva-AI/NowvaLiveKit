"""Standing lockout rule: not standing all the way up between squat reps.

Judged from the most upright frame before each rep, against the tallest the
athlete has stood this session, as a fraction of leg length.
"""

from __future__ import annotations

import math
from typing import Optional

from biomechanics.faults.fault_types import FaultType, RepFaultRule
from biomechanics.utils.types import FaultEvent, JointAngles


class StandingLockoutRule(RepFaultRule):
    def __init__(
        self,
        mild_threshold: float = 0.05,
        moderate_threshold: float = 0.08,
        severe_threshold: float = 0.12,
    ) -> None:
        self.mild_threshold = mild_threshold
        self.moderate_threshold = moderate_threshold
        self.severe_threshold = severe_threshold

    @property
    def fault_type(self) -> FaultType:
        return FaultType.LOCKOUT

    def judge_rep(self, features, reference, angles: JointAngles) -> Optional[FaultEvent]:
        deficit = features.lockout_deficit_ratio
        if not math.isfinite(deficit):
            return None
        return self._rep_fault(
            deficit,
            {"mild": self.mild_threshold, "moderate": self.moderate_threshold, "severe": self.severe_threshold},
            angles,
            features.rep_number,
            unit="ratio",
            phase="top",
        )
