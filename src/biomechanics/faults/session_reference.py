"""The athlete's own best rep this session, for spotting fatigue drift.

A personal baseline is the right tool for one question only: did this fault
appear as the set got harder? It never moves an absolute threshold — a fault on
rep 1 is still a fault. Velocity is referenced per set, because rep speed
resets with rest; position bests carry across the session.
"""

from __future__ import annotations

import math

# Depth drift is measured against the median of this many deepest reps, so one
# unusually deep (or mis-tracked) rep cannot make every later rep "shallow".
DEPTH_REFERENCE_REPS = 3
# Velocity loss is measured against the mean of this many fastest reps of the
# set: the single fastest is biased upward by frame-timing noise.
VELOCITY_REFERENCE_REPS = 2

# Position metrics where a smaller value is the better rep.
_LOWER_IS_BETTER = (
    "valgus",
    "hip_shoot_deg",
    "hip_shift_abs",
    "balance_abs",
    "depth_ratio",
)


class SessionReference:
    """Also holds the athlete's depth target, so the rep gate and the depth
    rules read one definition of "deep enough"."""

    def __init__(self, depth_tolerance_ratio: float = 0.08) -> None:
        self._best: dict[str, float] = {}
        self._set_best_velocity_mps = math.nan
        self._set_velocities: list[float] = []
        self._depths: list[float] = []
        self.depth_target_ratio: float | None = None
        self.depth_tolerance_ratio = depth_tolerance_ratio

    def reaches_depth_target(self, depth_ratio: float) -> bool:
        """No target (assessment, calibration) means every descent counts."""
        if self.depth_target_ratio is None or math.isnan(depth_ratio):
            return True
        return depth_ratio <= self.depth_target_ratio + self.depth_tolerance_ratio

    def annotate(self, features) -> None:
        """Fill the rep's velocity loss against the fastest earlier reps of this set."""
        reference = self.velocity_reference_mps
        velocity = features.concentric_velocity_mps
        if math.isfinite(reference) and reference > 0.0 and math.isfinite(velocity):
            features.velocity_loss_pct = max(0.0, (reference - velocity) / reference * 100.0)

    @property
    def velocity_reference_mps(self) -> float:
        fastest = sorted(self._set_velocities, reverse=True)[:VELOCITY_REFERENCE_REPS]
        return sum(fastest) / len(fastest) if fastest else math.nan

    def depth_reference(self) -> float:
        """Median of the session's few deepest reps — the depth this athlete owns."""
        deepest = sorted(self._depths)[:DEPTH_REFERENCE_REPS]
        if not deepest:
            return math.nan
        return deepest[len(deepest) // 2]

    def best(self, metric: str) -> float:
        return self._best.get(metric, math.nan)

    @property
    def set_best_velocity_mps(self) -> float:
        return self._set_best_velocity_mps

    def start_new_set(self) -> None:
        self._set_best_velocity_mps = math.nan
        self._set_velocities = []

    def update(self, features) -> None:
        """Fold a completed rep's features into the bests."""
        observed = {
            "valgus": _nanmax(features.valgus_l, features.valgus_r),
            "hip_shoot_deg": features.hip_shoot_deg,
            "hip_shift_abs": abs(features.hip_shift_ratio),
            "balance_abs": abs(features.balance_ratio),
            "depth_ratio": features.depth_ratio,
        }
        for metric in _LOWER_IS_BETTER:
            value = observed[metric]
            if not math.isfinite(value):
                continue
            current = self._best.get(metric)
            if current is None or value < current:
                self._best[metric] = value

        if math.isfinite(features.depth_ratio):
            self._depths.append(features.depth_ratio)

        velocity = features.concentric_velocity_mps
        if math.isfinite(velocity):
            self._set_velocities.append(velocity)
        if math.isfinite(velocity) and (
            math.isnan(self._set_best_velocity_mps) or velocity > self._set_best_velocity_mps
        ):
            self._set_best_velocity_mps = velocity

    def is_drift(self, metric: str, mild_threshold: float) -> bool:
        """True when the best rep so far was clean on this metric."""
        best = self.best(metric)
        return math.isfinite(best) and best < mild_threshold


def _nanmax(first: float, second: float) -> float:
    finite = [value for value in (first, second) if math.isfinite(value)]
    return max(finite) if finite else math.nan
