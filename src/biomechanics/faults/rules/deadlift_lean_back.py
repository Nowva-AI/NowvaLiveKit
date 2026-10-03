"""D5: over-extension at the top — leaning back past standing tall.

Signed sagittal trunk angle at the top against the athlete's standing angle; the
squat's trunk pitch is unsigned and could not tell a lean back from a lean forward.
"""

from __future__ import annotations

import math
from typing import Any

from biomechanics.deadlift.rule_base import DeadliftRepRule
from biomechanics.deadlift.types import DeadliftRepFeatures
from biomechanics.faults.fault_types import FaultType


class DeadliftLeanBackRule(DeadliftRepRule):
    phase = "top"
    unit = "deg"
    gravity_measured = True

    @property
    def fault_type(self) -> FaultType:
        return FaultType.DEADLIFT_LEAN_BACK

    def measure(self, features: DeadliftRepFeatures) -> tuple[float, dict[str, Any]] | None:
        lean_deg = features.lean_back_deg
        if not math.isfinite(lean_deg):
            return None
        return lean_deg, {}
