"""Press bar forward: the bar finishing ahead of mid-foot (ohp.md §5.2). Rig only.
With measured gravity it is judged against the locked mid-foot; with the body-axis
vertical (1° = 3.75 cm at the bar) the short shoulder lever is used instead and the
verdict is capped at approximate."""

from __future__ import annotations

import math

from biomechanics.analysis.press_features import DRIFT_BAR_FORWARD, PHASE_LOCKOUT, PressRepFeatures
from biomechanics.config import SeverityTiers
from biomechanics.faults.fault_types import FaultType, RepFaultRule
from biomechanics.faults.observability import APPROXIMATE
from biomechanics.faults.session_reference import SessionReference
from biomechanics.utils.frame_context import GRAVITY_MEASURED
from biomechanics.utils.types import FaultEvent, JointAngles

METRIC_MIDFOOT = "midfoot"
METRIC_SHOULDER = "shoulder"


class PressBarForwardRule(RepFaultRule):
    def __init__(self, tiers: SeverityTiers) -> None:
        self._thresholds = tiers.model_dump()

    @property
    def fault_type(self) -> FaultType:
        return FaultType.PRESS_BAR_FORWARD

    def judge_rep(
        self, features: PressRepFeatures, reference: SessionReference, angles: JointAngles,
    ) -> FaultEvent | None:
        measured = features.gravity_source == GRAVITY_MEASURED
        value = features.bar_over_midfoot_ratio if measured else features.bar_forward_shoulder_ratio
        if not math.isfinite(value):
            return None
        extra: dict[str, str] = {
            "gravity_source": features.gravity_source,
            "metric": METRIC_MIDFOOT if measured else METRIC_SHOULDER,
        }
        if not measured:
            extra["observability"] = APPROXIMATE
        return self._rep_fault(
            value, self._thresholds, angles, features.rep_number,
            unit="arm_ratio", phase=PHASE_LOCKOUT,
            is_drift=reference.is_drift(DRIFT_BAR_FORWARD, self._thresholds["mild"]),
            **extra,
        )
