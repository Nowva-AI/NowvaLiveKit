"""Knee tracking rule: knees caving inside the toe line, judged once per rep.

Reads the rep's worst sustained (p90) knee deviation from the knees-over-toes
plane over the bottom and the ascent — knees caving on the way up out of the
hole is the pattern a coach watches for, not only the bottom position.
"""

from __future__ import annotations

import math
from typing import Optional

from biomechanics.faults.fault_types import FaultType, RepFaultRule
from biomechanics.utils.types import FaultEvent, JointAngles

# Both knees within this many degrees of each other (and both past mild) report as "both".
BILATERAL_AGREEMENT_DEG = 2.0


class KneeTrackingRule(RepFaultRule):
    def __init__(
        self,
        mild_threshold: float = 8.0,
        moderate_threshold: float = 13.0,
        severe_threshold: float = 18.0,
    ) -> None:
        self.mild_threshold = mild_threshold
        self.moderate_threshold = moderate_threshold
        self.severe_threshold = severe_threshold

    @property
    def fault_type(self) -> FaultType:
        return FaultType.KNEE_VALGUS

    def judge_rep(self, features, reference, angles: JointAngles) -> Optional[FaultEvent]:
        sides = {
            "left": (features.valgus_l, features.valgus_peak_phase_l),
            "right": (features.valgus_r, features.valgus_peak_phase_r),
        }
        measured = {side: value for side, (value, _) in sides.items() if math.isfinite(value)}
        if not measured:
            return None

        worst_side = max(measured, key=measured.get)
        worst = measured[worst_side]
        side = worst_side
        if len(measured) == 2 and min(measured.values()) >= self.mild_threshold:
            if abs(measured["left"] - measured["right"]) < BILATERAL_AGREEMENT_DEG:
                side = "both"

        return self._rep_fault(
            worst,
            {"mild": self.mild_threshold, "moderate": self.moderate_threshold, "severe": self.severe_threshold},
            angles,
            features.rep_number,
            unit="deg",
            side=side,
            phase=sides[worst_side][1] or None,
            is_drift=reference.is_drift("valgus", self.mild_threshold),
            knee_valgus_l=features.valgus_l,
            knee_valgus_r=features.valgus_r,
        )
