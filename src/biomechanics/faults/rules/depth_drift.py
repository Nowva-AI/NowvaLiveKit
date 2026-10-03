"""Depth drift rule: reps getting shallower than the depth the athlete owns this session.

Only judged on reps that still met the depth target — a missed target is the
depth rule's verdict. Shallower later reps are a fatigue sign a coach calls out
before the target is ever missed.
"""

from __future__ import annotations

import math
from typing import Optional

from biomechanics.faults.fault_types import FaultType, RepFaultRule
from biomechanics.utils.types import FaultEvent, JointAngles


class DepthDriftRule(RepFaultRule):
    def __init__(
        self,
        mild_threshold: float = 0.15,
        moderate_threshold: float = 0.25,
        severe_threshold: float = 0.35,
    ) -> None:
        self.mild_threshold = mild_threshold
        self.moderate_threshold = moderate_threshold
        self.severe_threshold = severe_threshold

    @property
    def fault_type(self) -> FaultType:
        return FaultType.DEPTH_DRIFT

    def judge_rep(self, features, reference, angles: JointAngles) -> Optional[FaultEvent]:
        best = reference.depth_reference()
        depth = features.depth_ratio
        if not (math.isfinite(best) and math.isfinite(depth)):
            return None
        if not reference.reaches_depth_target(depth):
            return None
        return self._rep_fault(
            depth - best,
            {"mild": self.mild_threshold, "moderate": self.moderate_threshold, "severe": self.severe_threshold},
            angles,
            features.rep_number,
            unit="ratio",
            phase="bottom",
            is_drift=True,
            best_depth_ratio=best,
            depth_ratio=depth,
        )
