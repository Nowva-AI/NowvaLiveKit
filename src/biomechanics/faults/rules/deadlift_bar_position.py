"""D1: the bar is not over the midfoot at setup.

Read against the midfoot locked when the setup began, so plates hiding the feet
later cannot move it. The closed-loop foot guidance uses the live value while
standing at the bar; this rule judges where the bar actually was at liftoff.
"""

from __future__ import annotations

import math
from typing import Any

from biomechanics.deadlift.rule_base import DeadliftRepRule
from biomechanics.deadlift.types import DeadliftRepFeatures
from biomechanics.faults.fault_types import FaultType


class DeadliftBarPositionRule(DeadliftRepRule):
    phase = "setup"
    unit = "cm"
    bar_measured = True

    @property
    def fault_type(self) -> FaultType:
        return FaultType.DEADLIFT_BAR_POSITION

    def measure(self, features: DeadliftRepFeatures) -> tuple[float, dict[str, Any]] | None:
        offset_cm = features.bar_midfoot_setup_cm
        if not math.isfinite(offset_cm):
            return None
        # Ahead of midfoot: the bar is too far from the shins (step closer).
        direction = "forward" if offset_cm > 0.0 else "back"
        return abs(offset_cm), {"direction": direction, "offset_cm": offset_cm}
