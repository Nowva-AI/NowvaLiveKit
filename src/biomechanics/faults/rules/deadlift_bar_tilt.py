"""D8b: the bar tilts during the pull.

The 90th percentile height difference between the two plate hubs. Without bar
tracking it is the hands' height difference carried out to the hubs: ~3x the
hands' keypoint noise, which reaches the severe threshold on a few clean reps
in ten at 2 cm. So on the wrist proxy it is cued only when severe, and read as
the smaller of this rep's tilt and the set's recent reps' (to the same side;
none on a set's first rep): the noise tilts each rep its own way, a lifter's
uneven pull repeats. side names the low end.
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
        if features.bar_source == BAR_SOURCE_WRIST_PROXY:
            set_tilt_cm = features.set_bar_tilt_cm
            same_side = (set_tilt_cm > 0.0) == (features.bar_low_side == "right")
            if not math.isfinite(set_tilt_cm) or not same_side:
                return None
            tilt_cm = min(tilt_cm, abs(set_tilt_cm))
        return tilt_cm, {"side": features.bar_low_side or None}

    def tier_floor(self, features: DeadliftRepFeatures) -> str:
        return PROXY_MIN_TIER if features.bar_source == BAR_SOURCE_WRIST_PROXY else "mild"
