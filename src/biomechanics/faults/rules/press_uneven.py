"""Press uneven: one hand lagging so the bar tilts (ohp.md §5.5), against the set's
pooled rack tilt; the side named is the lagging hand. Single camera outside a
validated pitch band is capped at approximate."""

from __future__ import annotations

import math

from biomechanics.analysis.press_features import DRIFT_TILT, PressRepFeatures, mono_band_observability_cap
from biomechanics.config import SeverityTiers
from biomechanics.faults.fault_types import FaultType, RepFaultRule
from biomechanics.faults.session_reference import SessionReference
from biomechanics.utils.types import FaultEvent, JointAngles


class PressUnevenRule(RepFaultRule):
    def __init__(self, tiers: SeverityTiers, tilted_band_validated: bool) -> None:
        self._thresholds = tiers.model_dump()
        self._tilted_band_validated = tilted_band_validated

    @property
    def fault_type(self) -> FaultType:
        return FaultType.PRESS_UNEVEN

    def judge_rep(
        self, features: PressRepFeatures, reference: SessionReference, angles: JointAngles,
    ) -> FaultEvent | None:
        tilt = features.bar_tilt_deg
        if not math.isfinite(tilt):
            return None
        extra: dict[str, str | float] = {"pitch_band": features.pitch_band, "signed_value": tilt}
        cap = mono_band_observability_cap(features, self._tilted_band_validated)
        if cap is not None:
            extra["observability"] = cap
        # Positive tilt: the left hand finished higher, so the right one lags.
        return self._rep_fault(
            abs(tilt), self._thresholds, angles, features.rep_number,
            unit="deg", side="right" if tilt > 0.0 else "left", phase=features.bar_tilt_phase,
            is_drift=reference.is_drift(DRIFT_TILT, self._thresholds["mild"]),
            **extra,
        )
