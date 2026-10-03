"""D8b: the bar tilts during the pull.

The 90th percentile height difference between the two plate hubs. Without bar
tracking it is the hands' height difference carried out to the hubs: ~3x the
hands' keypoint noise, so on the wrist proxy it is cued only when severe. side
names the low end.
"""

from __future__ import annotations

import math
from typing import Any

from biomechanics.deadlift.rule_base import DeadliftRepRule
from biomechanics.deadlift.types import BAR_SOURCE_WRIST_PROXY, DeadliftRepFeatures
from biomechanics.faults.fault_types import FaultType

PROXY_MIN_TIER = "severe"


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

    def tier_floor(self, features: DeadliftRepFeatures) -> str:
        return PROXY_MIN_TIER if features.bar_source == BAR_SOURCE_WRIST_PROXY else "mild"
