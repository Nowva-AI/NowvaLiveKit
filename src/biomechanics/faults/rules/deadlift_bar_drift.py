"""D3: the bar drifts forward, away from the legs, during the pull.

The 90th percentile of the bar centre's forward travel from its liftoff
position, measured along measured gravity: a few degrees of tilted vertical
over ~55 cm of travel would otherwise fake centimetres of drift.
"""

from __future__ import annotations

import math
from typing import Any

from biomechanics.deadlift.rule_base import DeadliftRepRule
from biomechanics.deadlift.types import DeadliftRepFeatures
from biomechanics.faults.fault_types import FaultType


class DeadliftBarDriftRule(DeadliftRepRule):
    phase = "pull"
    unit = "cm"
    bar_measured = True
    gravity_measured = True
    drift_metric = "bar_drift_cm"

    @property
    def fault_type(self) -> FaultType:
        return FaultType.DEADLIFT_BAR_DRIFT

    def measure(self, features: DeadliftRepFeatures) -> tuple[float, dict[str, Any]] | None:
        drift_cm = features.bar_drift_cm
        if not math.isfinite(drift_cm):
            return None
        return drift_cm, {}
