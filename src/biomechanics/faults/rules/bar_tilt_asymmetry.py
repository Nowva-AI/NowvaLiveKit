"""
Barbell-Tilt Bilateral Asymmetry Rule

For barbell back/front squats, a non-horizontal bar during the rep is a much
stronger indicator of bilateral asymmetry than pose-based left/right joint
comparisons. If one side's hip drops faster than the other, the bar tilts.

This rule tracks the worst absolute tilt (and the implied left/right height
differential in cm) during a rep and emits a BILATERAL_ASYMMETRY fault at
rep end. It silently does nothing when no barbell is detected, so it's safe
to register unconditionally for all squat variants — bodyweight squats
simply never see a detection and never fire.

For squat it is the only source of BILATERAL_ASYMMETRY: pose-based knee-angle
asymmetry is dominated by monocular depth error, and hip shift is its own fault.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional

from biomechanics.utils.types import JointAngles, FaultEvent, FaultSeverity
from biomechanics.faults.fault_types import FaultRule, FaultType, FAULT_MESSAGES


class BarTiltAsymmetryRule(FaultRule):
    """
    Detects bilateral asymmetry via the barbell's tilt angle across a rep.

    Set via ``set_frame_context(bar_detection=...)`` each frame by the engine.
    """

    def __init__(
        self,
        bar_length_m: float = 2.2,
        mild_deg: float = 2.0,
        moderate_deg: float = 4.0,
        severe_deg: float = 7.0,
        mild_cm: float = 3.0,
        moderate_cm: float = 6.0,
        severe_cm: float = 10.0,
    ):
        self.bar_length_m = float(bar_length_m)
        self.mild_deg = float(mild_deg)
        self.moderate_deg = float(moderate_deg)
        self.severe_deg = float(severe_deg)
        self.mild_cm = float(mild_cm)
        self.moderate_cm = float(moderate_cm)
        self.severe_cm = float(severe_cm)

        self._max_abs_tilt_deg: float = 0.0
        self._max_abs_height_diff_cm: float = 0.0
        self._signed_diff_sum: float = 0.0  # used to pick heavier side
        self._samples: int = 0
        self._in_rep_prev: bool = False

    @property
    def fault_type(self) -> FaultType:
        return FaultType.BILATERAL_ASYMMETRY

    def reset(self) -> None:
        self._max_abs_tilt_deg = 0.0
        self._max_abs_height_diff_cm = 0.0
        self._signed_diff_sum = 0.0
        self._samples = 0
        self._in_rep_prev = False

    def _rep_metrics_reset(self) -> None:
        self._max_abs_tilt_deg = 0.0
        self._max_abs_height_diff_cm = 0.0
        self._signed_diff_sum = 0.0
        self._samples = 0

    def evaluate(
        self,
        angles: JointAngles,
        history: deque,
        in_rep: bool = False,
        rep_number: int = 0,
    ) -> Optional[FaultEvent]:
        detection = self._bar_detection

        if in_rep:
            if detection is not None:
                tilt_deg = detection.tilt_degrees
                self._max_abs_tilt_deg = max(self._max_abs_tilt_deg, abs(tilt_deg))

                # Height differential in cm from the measured bar length
                bar_px = detection.bar_length_px
                if bar_px > 1.0:
                    cm_per_px = (self.bar_length_m * 100.0) / bar_px
                    dy_px = detection.right_end.y - detection.left_end.y
                    height_diff_cm = dy_px * cm_per_px   # + if right end lower
                    self._max_abs_height_diff_cm = max(
                        self._max_abs_height_diff_cm, abs(height_diff_cm)
                    )
                    self._signed_diff_sum += height_diff_cm
                    self._samples += 1
            self._in_rep_prev = True
        return None

    def finish_rep(self, angles: JointAngles, rep_number: int) -> Optional[FaultEvent]:
        """Judge the rep that just completed, on that rep's own number.

        The verdict used to be emitted on the first frame after the rep, tagged
        with the next rep's number, so it never landed in the rep's own faults.
        """
        if not self._in_rep_prev:
            return None
        self._in_rep_prev = False
        peak_tilt = self._max_abs_tilt_deg
        peak_diff = self._max_abs_height_diff_cm
        signed_sum = self._signed_diff_sum
        samples = self._samples
        self._rep_metrics_reset()

        # No detection ever this rep → a bodyweight squat; stay quiet.
        if samples == 0:
            return None

        severity, score = self._combined_severity(peak_tilt, peak_diff)
        if severity == FaultSeverity.NONE:
            return None

        # Whichever side was most often lower (higher y in image coords) bore more load.
        heavier_side = "right" if signed_sum > 0 else "left"

        message = FAULT_MESSAGES[FaultType.BILATERAL_ASYMMETRY].get(
            severity.value, "Uneven weight distribution"
        )

        return self._create_fault_event(
            severity=severity,
            severity_score=score,
            message=message,
            angles=angles,
            rep_number=rep_number,
            details={
                "side": heavier_side,
                "phase": None,
                "is_drift": False,
                "value": peak_tilt,
                "unit": "deg",
                "source": "barbell_tilt",
                "peak_tilt_deg": peak_tilt,
                "peak_height_diff_cm": peak_diff,
                "heavier_side": heavier_side,
            },
        )

    def discard_rep(self) -> None:
        self._in_rep_prev = False
        self._rep_metrics_reset()

    def _combined_severity(self, peak_tilt: float, peak_diff: float) -> tuple:
        """Take the worst of (tilt, height-diff); return (FaultSeverity, score 0-3).

        We evaluate both axes and keep whichever is more severe. This avoids
        false-negatives when the camera is oblique (pixel→cm calibration may
        be inaccurate but tilt angle is still trustworthy) and vice versa.
        """
        tilt_severity, tilt_score = self._get_severity(
            peak_tilt,
            {"mild": self.mild_deg, "moderate": self.moderate_deg, "severe": self.severe_deg},
        )
        diff_severity, diff_score = self._get_severity(
            peak_diff,
            {"mild": self.mild_cm, "moderate": self.moderate_cm, "severe": self.severe_cm},
        )

        if tilt_score >= diff_score:
            return tilt_severity, tilt_score
        return diff_severity, diff_score
