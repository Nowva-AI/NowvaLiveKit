"""
Bilateral Asymmetry Fault Detection Rule

Compares left vs right joint angles to detect weight shifts and imbalanced
movement patterns. Parameterized with getter lambdas so one class works for
any joint pair. Evaluates one aggregate per rep over the bottom window, not
every frame, so keypoint noise cannot fire it (S16).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Callable, Optional

import numpy as np

from biomechanics.utils.types import JointAngles, FaultEvent, FaultSeverity
from biomechanics.faults.fault_types import FaultRule, FaultType, FAULT_MESSAGES

# Frames whose mean joint value is within this many degrees of the rep's
# deepest value form the bottom window the L-R difference is aggregated over.
BOTTOM_WINDOW_DEG = 10.0
# Fewer valid bottom-window samples than this and the rep is not judged.
MIN_BOTTOM_SAMPLES = 5


class SymmetryRule(FaultRule):
    """
    Rule for detecting bilateral asymmetry between any left/right joint pair.

    By default compares knee flexion (squat behavior). Override
    ``left_getter`` / ``right_getter`` to compare elbows, wrists, etc.

    While in a rep the per-frame L-R differences are collected; when the rep
    ends, the median difference over the frames near the rep's deepest point
    is evaluated once against the thresholds.

    Severity thresholds:
    - Mild: 5-10° difference
    - Moderate: 10-15° difference
    - Severe: >15° difference
    """

    def __init__(
        self,
        left_getter: Optional[Callable[[JointAngles], float]] = None,
        right_getter: Optional[Callable[[JointAngles], float]] = None,
        joint_name: str = "knee",
        mild_threshold: float = 5.0,
        moderate_threshold: float = 10.0,
        severe_threshold: float = 15.0,
    ):
        # Primary joint pair (defaults to knee flexion for backward compat)
        self._left_getter = left_getter or (lambda a: a.knee_flexion_l)
        self._right_getter = right_getter or (lambda a: a.knee_flexion_r)
        self._joint_name = joint_name

        self.mild_threshold = mild_threshold
        self.moderate_threshold = moderate_threshold
        self.severe_threshold = severe_threshold

        self._rep_depths: list[float] = []
        self._rep_differences: list[float] = []
        self._rep_number_in_rep: int = 0
        self._was_in_rep: bool = False

    @property
    def fault_type(self) -> FaultType:
        return FaultType.BILATERAL_ASYMMETRY

    def reset(self) -> None:
        self._clear_rep()

    def evaluate(
        self,
        angles: JointAngles,
        history: deque,
        in_rep: bool = False,
        rep_number: int = 0,
    ) -> Optional[FaultEvent]:
        """
        Collect L-R differences while in a rep; judge the rep once it ends.

        Only reps are judged, to avoid false positives from asymmetric
        standing positions. Frames where either side is NaN are skipped.
        """
        if in_rep:
            self._was_in_rep = True
            self._rep_number_in_rep = rep_number
            left_val = self._left_getter(angles)
            right_val = self._right_getter(angles)
            if not (math.isnan(left_val) or math.isnan(right_val)):
                self._rep_depths.append((left_val + right_val) / 2.0)
                self._rep_differences.append(left_val - right_val)
            return None

        # A rep that ended without finish_rep() (false start, rejected
        # descent) is discarded; the pipeline judges counted reps explicitly
        # so the fault lands on the same frame as the RepData.
        if self._was_in_rep:
            self._clear_rep()
        return None

    def finish_rep(self, angles: JointAngles, rep_number: int) -> Optional[FaultEvent]:
        """Judge the rep that just completed and clear its samples."""
        if not self._was_in_rep:
            return None
        self._rep_number_in_rep = rep_number
        fault = self._evaluate_rep(angles)
        self._clear_rep()
        return fault

    def discard_rep(self) -> None:
        self._clear_rep()

    def _evaluate_rep(self, angles: JointAngles) -> Optional[FaultEvent]:
        if len(self._rep_depths) < MIN_BOTTOM_SAMPLES:
            return None

        depths = np.asarray(self._rep_depths)
        differences = np.asarray(self._rep_differences)
        bottom_window = depths >= depths.max() - BOTTOM_WINDOW_DEG
        bottom_samples = int(bottom_window.sum())
        if bottom_samples < MIN_BOTTOM_SAMPLES:
            return None

        median_difference = float(np.median(differences[bottom_window]))
        asymmetry = abs(median_difference)

        if asymmetry < self.mild_threshold:
            return None

        severity, score = self._get_severity(
            asymmetry,
            {
                "mild": self.mild_threshold,
                "moderate": self.moderate_threshold,
                "severe": self.severe_threshold,
            },
        )

        if severity == FaultSeverity.NONE:
            return None

        heavier_side = "left" if median_difference > 0 else "right"

        message_key = severity.value
        message = FAULT_MESSAGES[FaultType.BILATERAL_ASYMMETRY].get(
            message_key, "Uneven weight distribution"
        )

        return self._create_fault_event(
            severity=severity,
            severity_score=score,
            message=message,
            angles=angles,
            rep_number=self._rep_number_in_rep,
            details={
                "joint": self._joint_name,
                "asymmetry": asymmetry,
                "heavier_side": heavier_side,
                "bottom_samples": bottom_samples,
            },
        )

    def _clear_rep(self) -> None:
        self._rep_depths = []
        self._rep_differences = []
        self._rep_number_in_rep = 0
        self._was_in_rep = False
