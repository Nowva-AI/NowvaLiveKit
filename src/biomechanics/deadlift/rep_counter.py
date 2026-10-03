"""The rep counter API the pipeline drives, backed by the deadlift analyser.

The pipeline calls update() once per frame after the analyser has observed it;
the analyser's state machine already decided whether a rep ended, so update()
only hands that rep over as RepData. It never uses the squat counter's standing
baseline EMA.
"""

from __future__ import annotations

import math

from biomechanics.utils.types import FaultEvent, JointAngles, RepData

from .analyzer import DeadliftRepAnalyzer


class DeadliftRepCounter:

    def __init__(self, analyzer: DeadliftRepAnalyzer) -> None:
        self._analyzer = analyzer
        self._current_faults: list[FaultEvent] = []
        self._last_pull_s = 0.0
        self._last_lower_s = 0.0
        # The squat counter's rejected-depth read-out; a deadlift is never rejected for depth.
        self.rejected_rep_max_depth_angle = math.nan

    @property
    def in_rep(self) -> bool:
        return self._analyzer.in_rep

    @property
    def phase(self) -> str:
        return self._analyzer.phase.value

    @property
    def rep_count(self) -> int:
        return self._analyzer.rep_count

    @rep_count.setter
    def rep_count(self, value: int) -> None:
        self._analyzer.rep_count = value

    @property
    def rep_started(self) -> bool:
        return self._analyzer.rep_started

    def set_assessment_mode(self, enabled: bool) -> None:
        """No squat-style assessment for the deadlift: every counted rep counts."""

    def reset(self) -> None:
        self._analyzer.reset_set()
        self._analyzer.rep_count = 0
        self._current_faults.clear()

    def add_fault(self, fault: FaultEvent) -> None:
        self._current_faults.append(fault)

    def clear_current_faults(self) -> None:
        self._current_faults.clear()

    def reject_last_rep(self) -> None:
        self._analyzer.rep_count = max(0, self._analyzer.rep_count - 1)

    def snapshot_rep_metrics(self, now: float | None = None) -> dict:
        return {
            "max_depth_angle": math.nan,
            "min_depth_angle": math.nan,
            "descent_time": self._last_lower_s,
            "ascent_time": self._last_pull_s,
            "faults": list(self._current_faults),
            "asymmetry": {},
            "avg_knee_asymmetry": 0.0,
            "avg_hip_asymmetry": 0.0,
            "in_rep": self.in_rep,
        }

    def update(
        self,
        signal_value: float = math.nan,
        timestamp: float | None = None,
        angles: JointAngles | None = None,
        faults: list[FaultEvent] | None = None,
    ) -> tuple[RepData | None, str | None]:
        """(RepData, None) on the frame a rep was counted; the rep's features stay
        queued in the analyser for the pipeline's finish_rep."""
        if faults:
            self._current_faults.extend(faults)
        completed = self._analyzer.take_completed_rep()
        if completed is None:
            return None, None
        features = completed.features
        self._last_pull_s = features.pull_time_s if math.isfinite(features.pull_time_s) else 0.0
        self._last_lower_s = features.lower_time_s if math.isfinite(features.lower_time_s) else 0.0
        rep = RepData(
            rep_number=features.rep_number,
            start_time=completed.start_time,
            end_time=completed.end_time,
            start_frame=completed.start_frame,
            end_frame=completed.end_frame,
            max_depth_angle=math.nan,
            min_depth_angle=math.nan,
            descent_time=self._last_lower_s,
            ascent_time=self._last_pull_s,
            faults=list(self._current_faults),
        )
        self._current_faults.clear()
        return rep, None
