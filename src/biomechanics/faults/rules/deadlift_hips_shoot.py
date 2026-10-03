"""D2: the hips rise before the chest off the floor.

Read from the liftoff to the bar passing the knees: how much further forward the
trunk ended up than the setup model predicts for this body (normally the chest
rises, so the trunk angle falls). The hip-to-shoulder rise ratio over the same
window is reported as the cross-check (> 1.4 = hips clearly leading).
"""

from __future__ import annotations

import math
from typing import Any

from biomechanics.deadlift.rule_base import DeadliftRepRule
from biomechanics.deadlift.types import DeadliftRepFeatures
from biomechanics.faults.fault_types import FaultType

HIPS_LEAD_RISE_RATIO = 1.4


class DeadliftHipsShootRule(DeadliftRepRule):
    phase = "pull"
    unit = "deg"
    gravity_measured = True
    drift_metric = "trunk_change_liftoff_knee_deg"

    @property
    def fault_type(self) -> FaultType:
        return FaultType.DEADLIFT_HIPS_SHOOT

    def measure(self, features: DeadliftRepFeatures) -> tuple[float, dict[str, Any]] | None:
        excess_deg = features.trunk_change_liftoff_knee_deg
        if not math.isfinite(excess_deg):
            return None
        ratio = features.hip_shoulder_rise_ratio
        return excess_deg, {
            "rise_ratio": ratio,
            "ratio_confirms": bool(math.isfinite(ratio) and ratio > HIPS_LEAD_RISE_RATIO),
            "model_predicted": math.isfinite(features.trunk_change_predicted_deg),
        }
