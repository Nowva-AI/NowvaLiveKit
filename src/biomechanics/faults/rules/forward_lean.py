"""
Forward Lean Fault Detection Rule

Evaluates trunk flexion angle during squats.
Excessive forward lean can indicate mobility issues or
compensatory patterns.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional

from biomechanics.utils.types import JointAngles, FaultEvent, FaultSeverity
from biomechanics.faults.fault_types import FaultRule, FaultType, FAULT_MESSAGES

# Minimum time between two forward-lean reports (was 150 frames at 30 fps)
FORWARD_LEAN_COOLDOWN_S = 5.0
# Trunk flexion reading of a perfectly upright trunk (180-convention)
UPRIGHT_TRUNK_DEG = 180.0
# Baseline calibration: how far past the lifter's own clean-rep peak each tier sits.
BASELINE_MILD_MARGIN_DEG = 10.0
BASELINE_MODERATE_MARGIN_DEG = 15.0
BASELINE_SEVERE_MARGIN_DEG = 20.0


class ForwardLeanRule(FaultRule):
    """
    Rule for detecting excessive forward lean.

    Monitors trunk flexion angle during reps.  Trunk flexion uses the
    180-convention: 180° = upright, decreasing with lean.

    Some forward lean is normal and necessary in squats,
    but excessive lean can indicate:
    - Poor ankle mobility
    - Weak posterior chain
    - Bar position issues

    Severity thresholds (180-convention — lower = more lean):
    - Mild: below 145°
    - Moderate: below 135°
    - Severe: below 125°

    Body-proportion scaling works in lean space from the base thresholds
    stored at construction: threshold = 180 - (180 - base) * scale, so a
    long-femur lifter (scale > 1) is allowed MORE lean, and re-applying the
    scaling never compounds (C8).
    """

    def __init__(
        self,
        mild_threshold: float = 145.0,
        moderate_threshold: float = 135.0,
        severe_threshold: float = 125.0,
    ):
        self._base_mild_threshold = mild_threshold
        self._base_moderate_threshold = moderate_threshold
        self._base_severe_threshold = severe_threshold
        self.mild_threshold = mild_threshold
        self.moderate_threshold = moderate_threshold
        self.severe_threshold = severe_threshold

        self._last_fault_time_s: float = float("-inf")
        self._baseline_peak_deg: float | None = None
        self._proportions_scale: float = 1.0

    def scale_for_proportions(self, proportions) -> None:
        self._proportions_scale = proportions.forward_lean_scale
        self._rebuild_thresholds()

    def apply_baseline(self, peak_trunk_flexion_deg: float) -> None:
        """Tighten the thresholds to the lifter's own clean-rep peak lean."""
        self._baseline_peak_deg = peak_trunk_flexion_deg
        self._rebuild_thresholds()

    def _rebuild_thresholds(self) -> None:
        scale = self._proportions_scale
        mild = UPRIGHT_TRUNK_DEG - (UPRIGHT_TRUNK_DEG - self._base_mild_threshold) * scale
        moderate = UPRIGHT_TRUNK_DEG - (UPRIGHT_TRUNK_DEG - self._base_moderate_threshold) * scale
        severe = UPRIGHT_TRUNK_DEG - (UPRIGHT_TRUNK_DEG - self._base_severe_threshold) * scale
        if self._baseline_peak_deg is not None:
            peak = self._baseline_peak_deg
            mild = min(mild, peak - BASELINE_MILD_MARGIN_DEG)
            moderate = min(moderate, peak - BASELINE_MODERATE_MARGIN_DEG)
            severe = min(severe, peak - BASELINE_SEVERE_MARGIN_DEG)
        self.mild_threshold = mild
        self.moderate_threshold = moderate
        self.severe_threshold = severe

    @property
    def fault_type(self) -> FaultType:
        return FaultType.FORWARD_LEAN

    def evaluate(
        self,
        angles: JointAngles,
        history: deque,
        in_rep: bool = False,
        rep_number: int = 0,
    ) -> Optional[FaultEvent]:
        """
        Evaluate trunk flexion for excessive forward lean.

        Only fires during reps - standing position lean is not
        a fault. Frames with a NaN trunk angle are skipped.
        """
        if not in_rep:
            return None

        trunk_flexion = angles.trunk_flexion  # 180° = upright, lower = more lean
        if math.isnan(trunk_flexion):
            return None

        # Cooldown between fault reports
        if angles.timestamp - self._last_fault_time_s < FORWARD_LEAN_COOLDOWN_S:
            return None

        # Above mild threshold → upright enough → no fault
        if trunk_flexion > self.mild_threshold:
            return None

        # Invert so _get_severity works (higher = more severe)
        lean_amount = self.mild_threshold - trunk_flexion
        severity, score = self._get_severity(
            lean_amount,
            {
                "mild": 0.0,
                "moderate": self.mild_threshold - self.moderate_threshold,
                "severe": self.mild_threshold - self.severe_threshold,
            },
        )

        if severity == FaultSeverity.NONE:
            return None

        self._last_fault_time_s = angles.timestamp

        message_key = severity.value
        message = FAULT_MESSAGES[FaultType.FORWARD_LEAN].get(
            message_key, "Excessive forward lean"
        )

        return self._create_fault_event(
            severity=severity,
            severity_score=score,
            message=message,
            angles=angles,
            rep_number=rep_number,
            details={
                "trunk_flexion": trunk_flexion,
            },
        )
