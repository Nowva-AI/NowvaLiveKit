"""Hip shift rule: the pelvis sliding sideways during the rep, judged once per rep.

Measured from where the hips started this rep, as a fraction of the distance
between the ankles, so standing slightly off-centre is not a shift.
"""

from __future__ import annotations

import math
from typing import Optional

from biomechanics.faults.fault_types import FaultType, RepFaultRule
from biomechanics.utils.types import FaultEvent, JointAngles


class HipShiftRule(RepFaultRule):
    def __init__(
        self,
        mild_threshold: float = 0.10,
        moderate_threshold: float = 0.15,
        severe_threshold: float = 0.22,
    ) -> None:
        self.mild_threshold = mild_threshold
        self.moderate_threshold = moderate_threshold
        self.severe_threshold = severe_threshold

    @property
    def fault_type(self) -> FaultType:
        return FaultType.HIP_SHIFT

    def judge_rep(self, features, reference, angles: JointAngles) -> Optional[FaultEvent]:
        shift = features.hip_shift_ratio
        if not math.isfinite(shift):
            return None
        return self._rep_fault(
            abs(shift),
            {"mild": self.mild_threshold, "moderate": self.moderate_threshold, "severe": self.severe_threshold},
            angles,
            features.rep_number,
            unit="ratio",
            side="right" if shift > 0 else "left",
            phase=features.hip_shift_peak_phase or None,
            is_drift=reference.is_drift("hip_shift_abs", self.mild_threshold),
            hip_shift_ratio=shift,
        )
