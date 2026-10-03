"""
The athlete's own RDL range and best reps this session (docs/exercises/rdl.md §3.6, §4.2).

Range is judged only against the athlete's own deepest counted reps, never a
fixed angle, so Nova never pushes past the range their hamstrings allow. The
best rep is the most hips-back one within that range, never the deepest. Both
are session-wide: they survive set boundaries and restart with the exercise.
"""

from __future__ import annotations

import math

import numpy as np

from biomechanics.analysis.hinge_features import HIP_SOURCE_MEASURED
from biomechanics.config import RdlRomConfig

# The range reference is the median of this many deepest counted reps.
HINGE_REFERENCE_REPS = 3

DRIFT_HIP_SINK = "rdl_hip_sink"
DRIFT_BAR_OFFSET = "rdl_bar_offset"
DRIFT_TILT = "rdl_tilt"
DRIFT_LOCKOUT_ABS = "rdl_lockout_abs"
DRIFT_DESCENT_TIME = "rdl_descent_time"

# Drift metrics where a smaller value is the better rep, and the feature each reads.
_LOWER_IS_BETTER = {
    DRIFT_HIP_SINK: "hip_sink_ratio",
    DRIFT_BAR_OFFSET: "bar_offset_ratio_p80",
    DRIFT_TILT: "bar_tilt_deg",
}
# A longer controlled descent is the better rep.
_HIGHER_IS_BETTER = frozenset({DRIFT_DESCENT_TIME})


class HingeSessionReference:
    """Session reference for the RDL, in the engine's SessionReference shape.
    There is no depth target: every rep the analyser counts reaches it."""

    def __init__(self, rom: RdlRomConfig) -> None:
        self._rom = rom
        self._best: dict[str, float] = {}
        self._deepest_hinges: list[float] = []
        self.depth_target_ratio: float | None = None

    def reaches_depth_target(self, depth_ratio: float) -> bool:
        return True

    def hinge_reference(self) -> float:
        """Median of the session's deepest counted hinges, capped; NaN until enough exist."""
        if len(self._deepest_hinges) < HINGE_REFERENCE_REPS:
            return math.nan
        return min(float(np.median(self._deepest_hinges)), self._rom.reference_cap_deg)

    def annotate(self, features) -> None:
        """Stamp the range reference as it stood before this rep, and the
        best-rep score: the most hips-back rep that reached the athlete's range."""
        reference_deg = self.hinge_reference()
        features.hinge_reference_deg = reference_deg
        if not math.isfinite(features.hip_sink_ratio):
            return
        if math.isfinite(reference_deg) and not features.hinge_deg >= reference_deg - self._rom.highlight_range_tolerance_deg:
            return
        features.highlight_score = -features.hip_sink_ratio

    def update(self, features) -> None:
        """Fold a counted rep into the range reference and the bests."""
        hinge_deg = features.hinge_deg
        if features.hinge_hip_source == HIP_SOURCE_MEASURED and math.isfinite(hinge_deg):
            self._deepest_hinges = sorted([*self._deepest_hinges, hinge_deg], reverse=True)[:HINGE_REFERENCE_REPS]
        observed = {metric: getattr(features, field) for metric, field in _LOWER_IS_BETTER.items()}
        observed[DRIFT_LOCKOUT_ABS] = abs(features.lockout_pitch_deg)
        for metric, value in observed.items():
            if math.isfinite(value) and value < self._best.get(metric, math.inf):
                self._best[metric] = value
        descent_s = features.descent_time_s
        if math.isfinite(descent_s) and descent_s > self._best.get(DRIFT_DESCENT_TIME, -math.inf):
            self._best[DRIFT_DESCENT_TIME] = descent_s

    def start_new_set(self) -> None:
        """Range and bests are session-wide."""

    def best(self, metric: str) -> float:
        return self._best.get(metric, math.nan)

    def is_drift(self, metric: str, mild_threshold: float) -> bool:
        """True when the best rep so far was clean on this metric."""
        best = self.best(metric)
        if not math.isfinite(best):
            return False
        if metric in _HIGHER_IS_BETTER:
            return best >= mild_threshold
        return best < mild_threshold
