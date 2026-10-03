"""D7: the shoulder joints sit behind the bar at setup.

A good start has them 0-6 cm in front of it. Ranked ahead of D4 for cueing: the
hip band assumes the shoulders in their band, so when both fire the shoulders
are the correction.
"""

from __future__ import annotations

import math
from typing import Any

from biomechanics.deadlift.rule_base import DeadliftRepRule
from biomechanics.deadlift.types import DeadliftRepFeatures
from biomechanics.faults.fault_types import FaultType


class DeadliftShouldersBehindRule(DeadliftRepRule):
    phase = "setup"
    unit = "cm"

    @property
    def fault_type(self) -> FaultType:
        return FaultType.DEADLIFT_SHOULDERS_BEHIND

    def measure(self, features: DeadliftRepFeatures) -> tuple[float, dict[str, Any]] | None:
        ahead_cm = features.shoulder_vs_bar_cm
        if not math.isfinite(ahead_cm) or ahead_cm >= 0.0:
            return None
        return -ahead_cm, {}
