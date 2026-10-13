"""RDL bar drift: the bar leaves the legs, load toward the toes (rdl.md §5.2).

The wrist midpoint ahead of midfoot over the lower half of the rep (p80), in
ankle-to-toe lengths. Along the camera axis, so rig-only, and approximate
until the rig validation measures the wrist-to-bar offset.
"""

from __future__ import annotations

import math

from biomechanics.analysis.hinge_features import HingeRepFeatures
from biomechanics.config import SeverityTiers
from biomechanics.faults.fault_types import FaultType, RepFaultRule
from biomechanics.faults.hinge_reference import DRIFT_BAR_OFFSET, HingeSessionReference
from biomechanics.utils.types import FaultEvent, JointAngles


class RdlBarDriftRule(RepFaultRule):
    def __init__(self, tiers: SeverityTiers) -> None:
        self.tiers = tiers

    @property
    def fault_type(self) -> FaultType:
        return FaultType.RDL_BAR_DRIFT

    def judge_rep(
        self, features: HingeRepFeatures, reference: HingeSessionReference, angles: JointAngles,
    ) -> FaultEvent | None:
        if not math.isfinite(features.bar_offset_ratio_p80):
            return None
        return self._rep_fault(
            features.bar_offset_ratio_p80,
            self.tiers.model_dump(),
            angles,
            features.rep_number,
            unit="foot_lengths",
            phase=features.bar_offset_peak_phase,
            is_drift=reference.is_drift(DRIFT_BAR_OFFSET, self.tiers.mild),
        )
