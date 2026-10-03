"""D8b: the bar tilts during the pull.

The 90th percentile height difference between the two plate hubs (or, without
bar tracking, the two wrists). side names the low end.
"""

from __future__ import annotations

import math
from typing import Any

from biomechanics.deadlift.rule_base import DeadliftRepRule
from biomechanics.deadlift.types import DeadliftRepFeatures
from biomechanics.faults.fault_types import FaultType


class DeadliftBarTiltRule(DeadliftRepRule):
    phase = "pull"
    unit = "cm"
    bar_measured = True

    @property
    def fault_type(self) -> FaultType:
        return FaultType.DEADLIFT_BAR_TILT

    def measure(self, features: DeadliftRepFeatures) -> tuple[float, dict[str, Any]] | None:
        tilt_cm = features.bar_tilt_cm
        if not math.isfinite(tilt_cm):
            return None
        return tilt_cm, {"side": features.bar_low_side or None}
