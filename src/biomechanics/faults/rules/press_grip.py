"""Press grip: forearms off vertical in the rack, from the front (ohp.md §5.6).
A setup note (details["setup_note"]: never makes a rep unclean), judged once per
set on the first rep that faced the camera with a finite reading; never on a
touch-and-go rep, which has no rack setup."""

from __future__ import annotations

import math

from biomechanics.analysis.press_features import PHASE_SETUP, PressRepFeatures
from biomechanics.config import SeverityTiers
from biomechanics.faults.fault_types import FaultType, RepFaultRule
from biomechanics.faults.session_reference import SessionReference
from biomechanics.utils.types import FaultEvent, JointAngles

# The correction the note asks for: wrists outside the elbows -> narrower.
DIRECTION_NARROWER = "narrower"
DIRECTION_WIDER = "wider"


class PressGripRule(RepFaultRule):
    def __init__(self, tiers: SeverityTiers) -> None:
        self._thresholds = tiers.model_dump()
        self._judged_this_set = False

    @property
    def fault_type(self) -> FaultType:
        return FaultType.PRESS_GRIP

    def reset(self) -> None:
        self._judged_this_set = False

    def judge_rep(
        self, features: PressRepFeatures, reference: SessionReference, angles: JointAngles,
    ) -> FaultEvent | None:
        if self._judged_this_set or features.piston or not features.facing_ok:
            return None
        tilt = features.grip_tilt_deg
        if not math.isfinite(tilt):
            return None
        self._judged_this_set = True
        return self._rep_fault(
            abs(tilt), self._thresholds, angles, features.rep_number,
            unit="deg", phase=PHASE_SETUP,
            direction=DIRECTION_NARROWER if tilt > 0.0 else DIRECTION_WIDER,
            signed_value=tilt,
            setup_note=True,
        )
