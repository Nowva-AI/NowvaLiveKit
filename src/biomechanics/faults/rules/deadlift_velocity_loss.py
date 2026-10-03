"""D10: the bar slowed against the set's two fastest reps.

The true bar speed (not the squat's shoulder proxy). Fatigue is load advice for
the recap and the set diagnosis, never a mid-set technique cue: its min tier is
"recap".
"""

from __future__ import annotations

import math
from typing import Any

from biomechanics.deadlift.rule_base import DeadliftRepRule
from biomechanics.deadlift.types import DeadliftRepFeatures
from biomechanics.faults.fault_types import FaultType


class DeadliftVelocityLossRule(DeadliftRepRule):
    phase = "pull"
    unit = "pct"
    bar_measured = True

    @property
    def fault_type(self) -> FaultType:
        return FaultType.DEADLIFT_VELOCITY_LOSS

    def measure(self, features: DeadliftRepFeatures) -> tuple[float, dict[str, Any]] | None:
        loss_pct = features.velocity_loss_pct
        if not math.isfinite(loss_pct):
            return None
        return loss_pct, {"velocity_mps": features.concentric_velocity_mps}
