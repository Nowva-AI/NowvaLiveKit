"""D9: the arms bend during the pull (pulling with the arms instead of pushing the floor).

Side-view elbow bend at its 90th percentile over the pull, against the athlete's
standing arms.
"""

from __future__ import annotations

import math
from typing import Any

from biomechanics.deadlift.rule_base import DeadliftRepRule
from biomechanics.deadlift.types import DeadliftRepFeatures
from biomechanics.faults.fault_types import FaultType


class DeadliftBentArmsRule(DeadliftRepRule):
    phase = "pull"
    unit = "deg"

    @property
    def fault_type(self) -> FaultType:
        return FaultType.DEADLIFT_BENT_ARMS

    def measure(self, features: DeadliftRepFeatures) -> tuple[float, dict[str, Any]] | None:
        bend_deg = features.elbow_flexion_deg
        if not math.isfinite(bend_deg):
            return None
        return bend_deg, {}
