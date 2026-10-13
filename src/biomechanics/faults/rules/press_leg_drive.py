"""Press leg drive: the hips dropping (knees dipping) to drive the bar up
(ohp.md §5.4). Single camera outside a validated pitch band is capped at
approximate: a tilted camera leaks the hips' forward shift into the drop."""

from __future__ import annotations

import math

from biomechanics.analysis.press_features import DRIFT_LEG_DRIVE, PressRepFeatures, mono_band_observability_cap
from biomechanics.config import SeverityTiers
from biomechanics.faults.fault_types import FaultType, RepFaultRule
from biomechanics.faults.session_reference import SessionReference
from biomechanics.utils.types import FaultEvent, JointAngles


class PressLegDriveRule(RepFaultRule):
    def __init__(self, tiers: SeverityTiers, tilted_band_validated: bool) -> None:
        self._thresholds = tiers.model_dump()
        self._tilted_band_validated = tilted_band_validated

    @property
    def fault_type(self) -> FaultType:
        return FaultType.PRESS_LEG_DRIVE

    def judge_rep(
        self, features: PressRepFeatures, reference: SessionReference, angles: JointAngles,
    ) -> FaultEvent | None:
        value = features.leg_drive_ratio
        if not math.isfinite(value):
            return None
        extra: dict[str, str] = {"pitch_band": features.pitch_band}
        cap = mono_band_observability_cap(features, self._tilted_band_validated)
        if cap is not None:
            extra["observability"] = cap
        return self._rep_fault(
            value, self._thresholds, angles, features.rep_number,
            unit="leg_ratio", phase=features.leg_drive_phase,
            is_drift=reference.is_drift(DRIFT_LEG_DRIVE, self._thresholds["mild"]),
            **extra,
        )
