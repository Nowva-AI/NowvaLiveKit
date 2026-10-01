"""Foot placement rule: an uneven setup — one foot staggered ahead or flared out more.

Judged from the most upright frame before each rep. The side names the foot to
move: the one ahead for a stagger, the one turned out more for a flare.
"""

from __future__ import annotations

import math
from typing import Optional

from biomechanics.faults.fault_types import FaultType, RepFaultRule
from biomechanics.utils.types import FaultEvent, JointAngles


class FootPlacementRule(RepFaultRule):
    def __init__(
        self,
        stagger_mild: float = 0.15,
        stagger_moderate: float = 0.25,
        stagger_severe: float = 0.35,
        flare_mild_deg: float = 10.0,
        flare_moderate_deg: float = 15.0,
        flare_severe_deg: float = 22.0,
    ) -> None:
        self.stagger = {"mild": stagger_mild, "moderate": stagger_moderate, "severe": stagger_severe}
        self.flare = {"mild": flare_mild_deg, "moderate": flare_moderate_deg, "severe": flare_severe_deg}

    @property
    def fault_type(self) -> FaultType:
        return FaultType.FOOT_PLACEMENT

    def judge_rep(self, features, reference, angles: JointAngles) -> Optional[FaultEvent]:
        candidates = []
        stagger = features.stagger_ratio
        if math.isfinite(stagger):
            _, score = self._get_severity(abs(stagger), self.stagger)
            side = "left" if stagger > 0 else "right"
            candidates.append((score, abs(stagger), self.stagger, "ratio", side, "stagger"))

        flare_l, flare_r = features.toe_out_l_deg, features.toe_out_r_deg
        if math.isfinite(flare_l) and math.isfinite(flare_r):
            difference = flare_l - flare_r
            _, score = self._get_severity(abs(difference), self.flare)
            side = "left" if difference > 0 else "right"
            candidates.append((score, abs(difference), self.flare, "deg", side, "flare"))

        if not candidates:
            return None
        score, value, thresholds, unit, side, kind = max(candidates, key=lambda candidate: candidate[0])
        if score <= 0.0:
            return None
        return self._rep_fault(
            value,
            thresholds,
            angles,
            features.rep_number,
            unit=unit,
            side=side,
            phase="setup",
            kind=kind,
            stagger_ratio=stagger,
            toe_out_l_deg=flare_l,
            toe_out_r_deg=flare_r,
        )
