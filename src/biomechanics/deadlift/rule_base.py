"""What every deadlift fault rule shares (PLAN.md §2.6, §2.9).

Each rule judges one rep's DeadliftRepFeatures once, at rep end. On top of the
squat's details (side, phase, is_drift, value, unit) every deadlift fault stamps
its effective minimum cue tier: the configured tier, raised to at least moderate
when this rep was measured less directly — from the wrists instead of the
tracked bar, or against the body vertical instead of measured gravity. The fault
is still recorded either way; the delivery side just does not speak it.
"""

from __future__ import annotations

import math
from typing import Any

from biomechanics.config import DeadliftFaultConfig
from biomechanics.faults.fault_types import RepFaultRule
from biomechanics.utils.types import FaultEvent, JointAngles

from .types import BAR_SOURCE_WRIST_PROXY, GRAVITY_SOURCE_BODY, DeadliftRepFeatures

TIER_RANK = {"mild": 1, "moderate": 2, "severe": 3, "recap": 4}
INDIRECT_MIN_TIER = "moderate"


def stricter_tier(first: str, second: str) -> str:
    return first if TIER_RANK[first] >= TIER_RANK[second] else second


class DeadliftRepRule(RepFaultRule):
    """A deadlift rep rule. Subclasses set the class attributes below and return
    (value, extras) from measure(); value NaN or None means not judged."""

    phase = ""
    unit = ""
    # Measured on the bar: wider thresholds and a moderate floor from the wrist proxy.
    bar_measured = False
    # Measured against vertical: a moderate floor without measured gravity.
    gravity_measured = False
    # Session-best metric for the drift channel, or "" for none.
    drift_metric = ""

    def __init__(self, thresholds: DeadliftFaultConfig, wrist_proxy_scale: float = 1.0) -> None:
        self.thresholds = thresholds
        self.wrist_proxy_scale = wrist_proxy_scale

    def measure(self, features: DeadliftRepFeatures) -> tuple[float, dict[str, Any]] | None:
        raise NotImplementedError

    def judge_rep(self, features, reference, angles: JointAngles) -> FaultEvent | None:
        if not isinstance(features, DeadliftRepFeatures):
            return None
        measured = self.measure(features)
        if measured is None:
            return None
        value, extras = measured
        if not math.isfinite(value):
            return None
        scale = 1.0
        min_tier = self.thresholds.min_tier
        if self.bar_measured and features.bar_source == BAR_SOURCE_WRIST_PROXY:
            scale = self.wrist_proxy_scale
            min_tier = stricter_tier(min_tier, INDIRECT_MIN_TIER)
        if self.gravity_measured and features.gravity_source == GRAVITY_SOURCE_BODY:
            min_tier = stricter_tier(min_tier, INDIRECT_MIN_TIER)
        thresholds = {
            "mild": self.thresholds.mild * scale,
            "moderate": self.thresholds.moderate * scale,
            "severe": self.thresholds.severe * scale,
        }
        is_drift = bool(self.drift_metric) and reference.is_drift(self.drift_metric, thresholds["mild"])
        side = extras.pop("side", None)
        return self._rep_fault(
            value,
            thresholds,
            angles,
            features.rep_number,
            unit=self.unit,
            side=side,
            phase=self.phase,
            is_drift=is_drift,
            min_tier=min_tier,
            bar_source=features.bar_source,
            gravity_source=features.gravity_source,
            **extras,
        )
