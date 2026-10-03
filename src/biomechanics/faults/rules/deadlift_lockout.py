"""D6: the rep ends short of standing tall.

The worse of the hip and knee extension deficits at the top, against the
athlete's own standing angles, which removes per-person keypoint bias.
"""

from __future__ import annotations

import math
from typing import Any

from biomechanics.deadlift.rule_base import DeadliftRepRule
from biomechanics.deadlift.types import DeadliftRepFeatures
from biomechanics.faults.fault_types import FaultType


class DeadliftLockoutRule(DeadliftRepRule):
    phase = "top"
    unit = "deg"

    @property
    def fault_type(self) -> FaultType:
        return FaultType.DEADLIFT_LOCKOUT

    def measure(self, features: DeadliftRepFeatures) -> tuple[float, dict[str, Any]] | None:
        deficits = {
            joint: deficit
            for joint, deficit in (
                ("hip", features.hip_extension_deficit_deg),
                ("knee", features.knee_extension_deficit_deg),
            )
            if math.isfinite(deficit)
        }
        if not deficits:
            return None
        joint = max(deficits, key=deficits.get)
        return deficits[joint], {"joint": joint}
