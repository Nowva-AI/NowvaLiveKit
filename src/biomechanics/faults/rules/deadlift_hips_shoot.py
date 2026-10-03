"""D2: the hips rise before the chest off the floor.

Two readings of the liftoff-to-knee-pass window decide it:
- whether the hips out-rose the shoulders: the hip-to-shoulder rise ratio, model
  free (a normal pull reads ~0.5-0.75, a held back angle ~1.0, hips first ~1.2+);
- how much: the trunk's tip beyond what the setup model predicts for this body.

The size, and so the thresholds, come from the model, which real lifts have not
validated yet (J6), and a lifter who holds the back angle reads 6-12 deg against
it. So there is a fault only when the hips decisively led: this rep's ratio over
1.05 and, since 2 cm of correlated keypoint noise scatters one rep's ratio by
~0.1, the set's recent reps over 1.15 too (or this rep alone over 1.30).
"""

from __future__ import annotations

import math
from typing import Any

from biomechanics.deadlift.rule_base import DeadliftRepRule
from biomechanics.deadlift.types import DeadliftRepFeatures
from biomechanics.faults.fault_types import FaultType

# This rep's hips out-rose its shoulders.
HIPS_LEAD_RISE_RATIO = 1.05
# ...and so did the set's recent reps (their median ratio), or this rep alone
# beyond what keypoint noise reaches.
SET_DECISIVE_RISE_RATIO = 1.15
REP_DECISIVE_RISE_RATIO = 1.30


def hips_led(features: DeadliftRepFeatures) -> bool:
    """The hips decisively rose before the chest on this rep."""
    ratio = features.hip_shoulder_rise_ratio
    if not math.isfinite(ratio) or ratio <= HIPS_LEAD_RISE_RATIO:
        return False
    return ratio > REP_DECISIVE_RISE_RATIO or features.set_rise_ratio > SET_DECISIVE_RISE_RATIO


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
        if not math.isfinite(excess_deg) or not hips_led(features):
            return None
        return excess_deg, {
            "rise_ratio": features.hip_shoulder_rise_ratio,
            "set_rise_ratio": features.set_rise_ratio,
            "model_predicted": math.isfinite(features.trunk_change_predicted_deg),
        }
