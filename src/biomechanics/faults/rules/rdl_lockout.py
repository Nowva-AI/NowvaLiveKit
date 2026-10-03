"""RDL lockout: the finish is not tall, hips not through or leaning back (rdl.md §5.3).

Judged on the rep's standing setup, i.e. the previous rep's finish, so a rep
that started at a turnaround (piston) is never judged. Signed trunk pitch is
along the camera axis: rig-only, approximate until the rig validation.
"""

from __future__ import annotations

import math

from biomechanics.analysis.hinge_features import HingeRepFeatures
from biomechanics.config import RdlLockoutConfig
from biomechanics.faults.fault_types import FaultType, RepFaultRule
from biomechanics.faults.hinge_reference import DRIFT_LOCKOUT_ABS, HingeSessionReference
from biomechanics.utils.types import FaultEvent, JointAngles

DIRECTION_FORWARD = "forward"
DIRECTION_BACK = "back"


class RdlLockoutRule(RepFaultRule):
    def __init__(self, tiers: RdlLockoutConfig) -> None:
        self.tiers = tiers

    @property
    def fault_type(self) -> FaultType:
        return FaultType.RDL_LOCKOUT

    def judge_rep(
        self, features: HingeRepFeatures, reference: HingeSessionReference, angles: JointAngles,
    ) -> FaultEvent | None:
        pitch_deg = features.lockout_pitch_deg
        if features.started_at_turnaround or not math.isfinite(pitch_deg):
            return None
        tiers = self.tiers
        if pitch_deg >= 0.0:
            direction = DIRECTION_FORWARD
            thresholds = {"mild": tiers.forward_mild, "moderate": tiers.forward_moderate, "severe": tiers.forward_severe}
        else:
            direction = DIRECTION_BACK
            thresholds = {"mild": tiers.back_mild, "moderate": tiers.back_moderate, "severe": tiers.back_severe}
        return self._rep_fault(
            abs(pitch_deg),
            thresholds,
            angles,
            features.rep_number,
            unit="deg",
            phase="top",
            is_drift=reference.is_drift(DRIFT_LOCKOUT_ABS, thresholds["mild"]),
            direction=direction,
        )
