"""Velocity loss rule: rep speed dropping against the fastest reps of the set.

Concentric velocity loss is how strength coaches read proximity to failure
(20% loss ≈ moderate fatigue, 40% ≈ near failure; Sánchez-Medina &
González-Badillo 2011). Velocity is the shoulder midpoint, the bar's proxy.
"""

from __future__ import annotations

import math
from typing import Optional

from biomechanics.faults.fault_types import FaultType, RepFaultRule
from biomechanics.utils.types import FaultEvent, JointAngles


class VelocityLossRule(RepFaultRule):
    def __init__(
        self,
        mild_pct: float = 20.0,
        moderate_pct: float = 30.0,
        severe_pct: float = 40.0,
    ) -> None:
        self.mild_pct = mild_pct
        self.moderate_pct = moderate_pct
        self.severe_pct = severe_pct

    @property
    def fault_type(self) -> FaultType:
        return FaultType.VELOCITY_LOSS

    def judge_rep(self, features, reference, angles: JointAngles) -> Optional[FaultEvent]:
        loss = features.velocity_loss_pct
        if not math.isfinite(loss):
            return None
        return self._rep_fault(
            loss,
            {"mild": self.mild_pct, "moderate": self.moderate_pct, "severe": self.severe_pct},
            angles,
            features.rep_number,
            unit="pct",
            phase="ascent",
            is_drift=True,
            velocity_mps=features.concentric_velocity_mps,
            reference_velocity_mps=reference.velocity_reference_mps,
        )
