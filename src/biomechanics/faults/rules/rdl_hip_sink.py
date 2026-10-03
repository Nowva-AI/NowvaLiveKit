"""RDL hip sink: the hips sink instead of travelling back, squatting it (rdl.md §5.1).

Measured as the share of standing hip height lost at the bottom, i.e. combined
thigh and shin inclination, not knee travel. Not judged when the bottom hips
were not seen. The start tier (certified mount, lens height) belongs to the
profile's cue_start_tier: this rule emits plain verdicts, downgraded to
approximate only when an uncertified camera's pitch tripped the gross guard.
"""

from __future__ import annotations

import math

from biomechanics.analysis.hinge_features import HIP_SOURCE_MEASURED, HingeRepFeatures
from biomechanics.config import SeverityTiers
from biomechanics.faults.fault_types import FaultType, RepFaultRule
from biomechanics.faults.hinge_reference import DRIFT_HIP_SINK, HingeSessionReference
from biomechanics.faults.observability import APPROXIMATE
from biomechanics.utils.types import FaultEvent, JointAngles


class RdlHipSinkRule(RepFaultRule):
    def __init__(self, tiers: SeverityTiers) -> None:
        self.tiers = tiers

    @property
    def fault_type(self) -> FaultType:
        return FaultType.RDL_HIP_SINK

    def judge_rep(
        self, features: HingeRepFeatures, reference: HingeSessionReference | None, angles: JointAngles,
    ) -> FaultEvent | None:
        if features.hinge_hip_source != HIP_SOURCE_MEASURED or not math.isfinite(features.hip_sink_ratio):
            return None
        fault = self._rep_fault(
            features.hip_sink_ratio,
            self.tiers.model_dump(),
            angles,
            features.rep_number,
            unit="ratio",
            side="both",
            phase="bottom",
            is_drift=reference is not None and reference.is_drift(DRIFT_HIP_SINK, self.tiers.mild),
            hinge_deg=features.hinge_deg,
            knee_flexion_bottom_deg=features.knee_flexion_bottom_deg,
        )
        if fault is not None and features.camera_pitch_untrusted:
            fault.details["observability"] = APPROXIMATE
        return fault
