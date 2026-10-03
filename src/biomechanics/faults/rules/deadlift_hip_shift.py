"""D8: the hips shift sideways during the pull.

Sideways travel of the hip midpoint against the locked midfoot, as a fraction of
ankle separation, read as a sustained level (the squat's method) so one jittery
frame cannot make a shift. side is the direction the hips moved. Trunk rotation
is not part of it: a mixed grip rotates the trunk by design.
"""

from __future__ import annotations

import math
from typing import Any

from biomechanics.deadlift.rule_base import DeadliftRepRule
from biomechanics.deadlift.types import DeadliftRepFeatures
from biomechanics.faults.fault_types import FaultType


class DeadliftHipShiftRule(DeadliftRepRule):
    phase = "pull"
    unit = "ratio"
    drift_metric = "hip_shift_abs"

    @property
    def fault_type(self) -> FaultType:
        return FaultType.DEADLIFT_HIP_SHIFT

    def measure(self, features: DeadliftRepFeatures) -> tuple[float, dict[str, Any]] | None:
        shift = features.hip_shift_ratio
        if not math.isfinite(shift):
            return None
        return abs(shift), {"side": "right" if shift > 0.0 else "left", "grip": features.grip}
