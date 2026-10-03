"""RDL tempo: lowering the bar too fast (rdl.md §5.6).

Judged on the 5-95% shoulder-drop time, a timing normalised by the rep's own
range, so camera pitch and perspective do not enter. Severity grows as the
descent gets shorter than the controlled minimum.
"""

from __future__ import annotations

import math

from biomechanics.analysis.hinge_features import HingeRepFeatures
from biomechanics.config import RdlTempoConfig
from biomechanics.faults.fault_types import FaultType, RepFaultRule
from biomechanics.faults.hinge_reference import DRIFT_DESCENT_TIME, HingeSessionReference
from biomechanics.utils.types import FaultEvent, JointAngles


class RdlTempoRule(RepFaultRule):
    def __init__(self, tiers: RdlTempoConfig) -> None:
        self.tiers = tiers

    @property
    def fault_type(self) -> FaultType:
        return FaultType.RDL_TEMPO

    def judge_rep(
        self, features: HingeRepFeatures, reference: HingeSessionReference, angles: JointAngles,
    ) -> FaultEvent | None:
        descent_s = features.descent_time_s
        mild_s = self.tiers.mild_seconds
        if not math.isfinite(descent_s) or descent_s <= 0.0 or descent_s >= mild_s:
            return None
        return self._rep_fault(
            mild_s - descent_s,
            {
                "mild": 0.0,
                "moderate": mild_s - self.tiers.moderate_seconds,
                "severe": mild_s - self.tiers.severe_seconds,
            },
            angles,
            features.rep_number,
            unit="s",
            phase="descent",
            is_drift=reference.is_drift(DRIFT_DESCENT_TIME, mild_s),
            kind="fast_descent",
            descent_time_s=descent_s,
        )
