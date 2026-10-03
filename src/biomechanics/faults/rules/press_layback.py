"""Press layback: the trunk leaning back under the bar beyond what clearing the chin
needs (ohp.md §5.1). Rig only; without measured gravity the body-axis vertical is
2-4° off, so the verdict is capped at approximate (recap-only)."""

from __future__ import annotations

import math

from biomechanics.analysis.press_features import DRIFT_LAYBACK, PHASE_PRESS, PressRepFeatures
from biomechanics.config import SeverityTiers
from biomechanics.faults.fault_types import FaultType, RepFaultRule
from biomechanics.faults.observability import APPROXIMATE
from biomechanics.faults.session_reference import SessionReference
from biomechanics.utils.frame_context import GRAVITY_MEASURED
from biomechanics.utils.types import FaultEvent, JointAngles


class PressLaybackRule(RepFaultRule):
    def __init__(self, tiers: SeverityTiers) -> None:
        self._thresholds = tiers.model_dump()

    @property
    def fault_type(self) -> FaultType:
        return FaultType.PRESS_LAYBACK

    def judge_rep(
        self, features: PressRepFeatures, reference: SessionReference, angles: JointAngles,
    ) -> FaultEvent | None:
        value = features.layback_deg
        if not math.isfinite(value):
            return None
        extra: dict[str, str] = {"gravity_source": features.gravity_source}
        if features.gravity_source != GRAVITY_MEASURED:
            extra["observability"] = APPROXIMATE
        return self._rep_fault(
            value, self._thresholds, angles, features.rep_number,
            unit="deg", phase=PHASE_PRESS,
            is_drift=reference.is_drift(DRIFT_LAYBACK, self._thresholds["mild"]),
            **extra,
        )
