"""RDL bar tilt: one hand drops or the bar rotates at the bottom (rdl.md §5.4).

The wrists' height difference at the bottom minus the rep's start, so camera
roll and grip asymmetry cancel. The side is the hand that dropped.
"""

from __future__ import annotations

import math

from biomechanics.analysis.hinge_features import HingeRepFeatures
from biomechanics.config import SeverityTiers
from biomechanics.faults.fault_types import FaultType, RepFaultRule
from biomechanics.faults.hinge_reference import DRIFT_TILT, HingeSessionReference
from biomechanics.utils.types import FaultEvent, JointAngles


class RdlBarTiltRule(RepFaultRule):
    def __init__(self, tiers: SeverityTiers) -> None:
        self.tiers = tiers

    @property
    def fault_type(self) -> FaultType:
        return FaultType.RDL_BAR_TILT

    def judge_rep(
        self, features: HingeRepFeatures, reference: HingeSessionReference, angles: JointAngles,
    ) -> FaultEvent | None:
        if not math.isfinite(features.bar_tilt_deg):
            return None
        return self._rep_fault(
            features.bar_tilt_deg,
            self.tiers.model_dump(),
            angles,
            features.rep_number,
            unit="deg",
            side=features.bar_tilt_side,
            phase="bottom",
            is_drift=reference.is_drift(DRIFT_TILT, self.tiers.mild),
        )
