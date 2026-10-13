"""Press lockout: elbows short of straight overhead (ohp.md §5.3), judged on the
worse side; a side is named only when the arms differ by the configured margin."""

from __future__ import annotations

import math

from biomechanics.analysis.press_features import DRIFT_LOCKOUT, PHASE_LOCKOUT, PressRepFeatures
from biomechanics.config import SeverityTiers
from biomechanics.faults.fault_types import FaultType, RepFaultRule
from biomechanics.faults.session_reference import SessionReference
from biomechanics.utils.types import FaultEvent, JointAngles

SIDE_BOTH = "both"


class PressLockoutRule(RepFaultRule):
    def __init__(self, tiers: SeverityTiers, side_min_difference_deg: float) -> None:
        self._thresholds = tiers.model_dump()
        self._side_min_difference_deg = side_min_difference_deg

    @property
    def fault_type(self) -> FaultType:
        return FaultType.PRESS_LOCKOUT

    def judge_rep(
        self, features: PressRepFeatures, reference: SessionReference, angles: JointAngles,
    ) -> FaultEvent | None:
        left, right = features.lockout_elbow_l_deg, features.lockout_elbow_r_deg
        if not (math.isfinite(left) or math.isfinite(right)):
            return None
        if math.isfinite(left) and math.isfinite(right):
            value = max(left, right)
            side = SIDE_BOTH
            if abs(left - right) >= self._side_min_difference_deg:
                side = "left" if left > right else "right"
        else:
            # One arm unmeasured: no claim about which arm.
            value = left if math.isfinite(left) else right
            side = SIDE_BOTH
        return self._rep_fault(
            value, self._thresholds, angles, features.rep_number,
            unit="deg", side=side, phase=PHASE_LOCKOUT,
            is_drift=reference.is_drift(DRIFT_LOCKOUT, self._thresholds["mild"]),
            elbow_l_deg=left, elbow_r_deg=right,
        )
