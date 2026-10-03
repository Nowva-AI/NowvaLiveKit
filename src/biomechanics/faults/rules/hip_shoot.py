"""Hip shoot rule: hips rising faster than the chest out of the hole (good-morning squat).

The trunk pitching forward while the hip recovers its first third of height is
the classic loaded-squat breakdown; absolute lean is not, since bar position and
anatomy dictate it.
"""

from __future__ import annotations

import math
from typing import Optional

from biomechanics.faults.fault_types import FaultType, RepFaultRule
from biomechanics.utils.types import FaultEvent, JointAngles


class HipShootRule(RepFaultRule):
    def __init__(
        self,
        mild_threshold: float = 8.0,
        moderate_threshold: float = 12.0,
        severe_threshold: float = 18.0,
    ) -> None:
        self.mild_threshold = mild_threshold
        self.moderate_threshold = moderate_threshold
        self.severe_threshold = severe_threshold

    @property
    def fault_type(self) -> FaultType:
        return FaultType.HIP_SHOOT

    def judge_rep(self, features, reference, angles: JointAngles) -> Optional[FaultEvent]:
        if not math.isfinite(features.hip_shoot_deg):
            return None
        return self._rep_fault(
            features.hip_shoot_deg,
            {"mild": self.mild_threshold, "moderate": self.moderate_threshold, "severe": self.severe_threshold},
            angles,
            features.rep_number,
            unit="deg",
            phase="ascent",
            is_drift=reference.is_drift("hip_shoot_deg", self.mild_threshold),
        )
