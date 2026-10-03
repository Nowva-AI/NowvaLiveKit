"""D2: the hips rise before the chest off the floor.

Read from the liftoff to the bar passing the knees: how much further forward the
trunk ended up than the setup model predicts for this body (normally the chest
rises, so the trunk angle falls). The size, and so the thresholds, come from the
model, which real lifts have not validated yet (J6). So below severe the rep's
own, model-free evidence must agree: the hips rose faster than the shoulders
(hip-to-shoulder rise ratio over 1.0; a normal pull reads ~0.5-0.75). A lifter who
holds the back angle (ratio ~0.9-1.0) is recorded but only cued when severe.
"""

from __future__ import annotations

import math
from typing import Any

from biomechanics.deadlift.rule_base import DeadliftRepRule
from biomechanics.deadlift.types import DeadliftRepFeatures
from biomechanics.faults.fault_types import FaultType

HIPS_LEAD_RISE_RATIO = 1.0
# Cue tier for a model-sized excess the rise ratio does not confirm.
UNCONFIRMED_MIN_TIER = "severe"


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

    def tier_floor(self, features: DeadliftRepFeatures) -> str:
        ratio = features.hip_shoulder_rise_ratio
        if math.isfinite(ratio) and ratio > HIPS_LEAD_RISE_RATIO:
            return "mild"
        return UNCONFIRMED_MIN_TIER
