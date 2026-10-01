"""Descent control rule: dropping into the bottom of the squat too fast.

Judged from the rep's descent time. Kept at the lowest cue priority: a fast but
controlled descent is fine for trained lifters, so this cue only matters when
nothing more important is wrong.
"""

from __future__ import annotations

import math
from typing import Optional

from biomechanics.faults.fault_types import FaultType, RepFaultRule
from biomechanics.utils.types import FaultEvent, JointAngles


class DescentControlRule(RepFaultRule):
    def __init__(
        self,
        mild_seconds: float = 0.6,
        moderate_seconds: float = 0.45,
        severe_seconds: float = 0.3,
    ) -> None:
        self.mild_seconds = mild_seconds
        self.moderate_seconds = moderate_seconds
        self.severe_seconds = severe_seconds

    @property
    def fault_type(self) -> FaultType:
        return FaultType.TEMPO

    def judge_rep(self, features, reference, angles: JointAngles) -> Optional[FaultEvent]:
        descent = features.descent_time_s
        if not math.isfinite(descent) or descent <= 0.0 or descent >= self.mild_seconds:
            return None
        # Severity grows as the descent gets shorter than the controlled minimum.
        return self._rep_fault(
            self.mild_seconds - descent,
            {
                "mild": 0.0,
                "moderate": self.mild_seconds - self.moderate_seconds,
                "severe": self.mild_seconds - self.severe_seconds,
            },
            angles,
            features.rep_number,
            unit="s",
            phase="descent",
            kind="fast_descent",
            descent_time_s=descent,
        )
