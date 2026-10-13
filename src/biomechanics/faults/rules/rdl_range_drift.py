"""RDL range drift: reps shorter than the athlete's own range (rdl.md §5.5).

The hinge short of the median of the session's deepest counted reps, once
enough exist. Pose biases are constant within a session and cancel; the cue
never asks for more than the athlete's own range.
"""

from __future__ import annotations

import math

from biomechanics.analysis.hinge_features import HIP_SOURCE_MEASURED, HingeRepFeatures
from biomechanics.config import SeverityTiers
from biomechanics.faults.fault_types import FaultType, RepFaultRule
from biomechanics.faults.hinge_reference import HingeSessionReference
from biomechanics.utils.types import FaultEvent, JointAngles


class RdlRangeDriftRule(RepFaultRule):
    def __init__(self, tiers: SeverityTiers) -> None:
        self.tiers = tiers

    @property
    def fault_type(self) -> FaultType:
        return FaultType.RDL_RANGE_DRIFT

    def judge_rep(
        self, features: HingeRepFeatures, reference: HingeSessionReference, angles: JointAngles,
    ) -> FaultEvent | None:
        reference_deg = reference.hinge_reference()
        if features.hinge_hip_source != HIP_SOURCE_MEASURED:
            return None
        if not (math.isfinite(reference_deg) and math.isfinite(features.hinge_deg)):
            return None
        return self._rep_fault(
            reference_deg - features.hinge_deg,
            self.tiers.model_dump(),
            angles,
            features.rep_number,
            unit="deg",
            phase="bottom",
            is_drift=True,
            reference_deg=reference_deg,
            range_label=features.range_label,
        )
