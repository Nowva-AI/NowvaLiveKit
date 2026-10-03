"""D4: hips too low or too high at setup, against the setup model's band.

The band is the hip heights this athlete's segments allow with the bar over
midfoot, shins on the bar, arms vertical and the shoulders 0-6 cm ahead of the
bar (deadlift.setup_model). No band (no setup fits the measured body) means no
verdict, never a guess.
"""

from __future__ import annotations

import math
from typing import Any

from biomechanics.deadlift.rule_base import DeadliftRepRule
from biomechanics.deadlift.types import DeadliftRepFeatures
from biomechanics.faults.fault_types import FaultType


class DeadliftSetupHipsRule(DeadliftRepRule):
    phase = "setup"
    unit = "cm"

    @property
    def fault_type(self) -> FaultType:
        return FaultType.DEADLIFT_SETUP_HIPS

    def measure(self, features: DeadliftRepFeatures) -> tuple[float, dict[str, Any]] | None:
        height = features.setup_hip_height_cm
        low = features.setup_hip_band_low_cm
        high = features.setup_hip_band_high_cm
        if not (math.isfinite(height) and math.isfinite(low) and math.isfinite(high)):
            return None
        if height < low:
            return low - height, {"direction": "up", "hip_height_cm": height, "band_cm": [low, high]}
        if height > high:
            return height - high, {"direction": "down", "hip_height_cm": height, "band_cm": [low, high]}
        return None
