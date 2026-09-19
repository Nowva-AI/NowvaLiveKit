"""
Back Rounding Fault Detection Rule

Detects spinal flexion during lifts where the trunk should stay
neutral (deadlift, row, RDL). Monitors the difference between
rep-start trunk flexion and the minimum trunk flexion during the rep.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional

from biomechanics.utils.types import JointAngles, FaultEvent, FaultSeverity
from biomechanics.faults.fault_types import FaultRule, FaultType, FAULT_MESSAGES

# Minimum time between two back-rounding reports (was 90 frames at 30 fps)
BACK_ROUNDING_COOLDOWN_S = 3.0


class BackRoundingRule(FaultRule):
    """
    Rule for detecting back rounding during lifts.

    Tracks trunk flexion at rep entry (the user's setup angle) and
    fires if the trunk flexion drops significantly lower during the rep
    (i.e. the back rounds forward under load).

    Trunk flexion uses 180° convention: 180 = upright, lower = more flexed.
    A drop of 10°+ from the setup angle indicates rounding.

    Severity bands (degrees of drop from setup):
    - Mild: 10-20°
    - Moderate: 20-30°
    - Severe: >30°
    """

    def __init__(
        self,
        mild_threshold: float = 10.0,
        moderate_threshold: float = 20.0,
        severe_threshold: float = 30.0,
    ):
        self.mild_threshold = mild_threshold
        self.moderate_threshold = moderate_threshold
        self.severe_threshold = severe_threshold

        self._setup_trunk_flexion: float = 0.0
        self._setup_set: bool = False
        self._last_fault_time_s: float = float("-inf")

    @property
    def fault_type(self) -> FaultType:
        return FaultType.BACK_ROUNDING

    def reset(self) -> None:
        self._setup_trunk_flexion = 0.0
        self._setup_set = False

    def evaluate(
        self,
        angles: JointAngles,
        history: deque,
        in_rep: bool = False,
        rep_number: int = 0,
    ) -> Optional[FaultEvent]:
        if not in_rep:
            if self._setup_set:
                self.reset()
            return None

        trunk_flexion = angles.trunk_flexion
        if math.isnan(trunk_flexion):
            return None

        # Record setup angle at rep entry (first valid frame)
        if not self._setup_set:
            self._setup_trunk_flexion = trunk_flexion
            self._setup_set = True
            return None

        # Cooldown
        if angles.timestamp - self._last_fault_time_s < BACK_ROUNDING_COOLDOWN_S:
            return None

        # Trunk flexion drop (180-convention: drop = setup - current)
        drop = self._setup_trunk_flexion - trunk_flexion
        if drop < self.mild_threshold:
            return None

        severity, score = self._get_severity(
            drop,
            {
                "mild": self.mild_threshold,
                "moderate": self.moderate_threshold,
                "severe": self.severe_threshold,
            },
        )

        if severity == FaultSeverity.NONE:
            return None

        self._last_fault_time_s = angles.timestamp

        message = FAULT_MESSAGES.get(FaultType.BACK_ROUNDING, {}).get(
            severity.value, "Back rounding — keep spine neutral"
        )

        return self._create_fault_event(
            severity=severity,
            severity_score=score,
            message=message,
            angles=angles,
            rep_number=rep_number,
            details={
                "setup_trunk_flexion": self._setup_trunk_flexion,
                "current_trunk_flexion": trunk_flexion,
                "drop": drop,
            },
        )
