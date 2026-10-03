"""The deadlift's drift channel (PLAN.md §2.8).

A SessionReference subclass, so the depth-target API the pipeline reads on every
switch and at calibration stays intact; it only changes which metrics it tracks.
Session bests (lower is better) for bar drift, trunk change off the floor and
sideways hip travel; velocity is referenced per set, from the set's two fastest
reps, through the base class.
"""

from __future__ import annotations

import math

from biomechanics.faults.session_reference import SessionReference

_LOWER_IS_BETTER = (
    "bar_drift_cm",
    "trunk_change_liftoff_knee_deg",
    "hip_shift_abs",
)


class DeadliftSessionReference(SessionReference):

    def update(self, features) -> None:
        """Fold a completed deadlift rep into the bests and the set's velocities."""
        observed = {
            "bar_drift_cm": features.bar_drift_cm,
            "trunk_change_liftoff_knee_deg": features.trunk_change_liftoff_knee_deg,
            "hip_shift_abs": features.hip_shift_abs,
        }
        for metric in _LOWER_IS_BETTER:
            value = observed[metric]
            if not math.isfinite(value):
                continue
            current = self._best.get(metric)
            if current is None or value < current:
                self._best[metric] = value

        velocity = features.concentric_velocity_mps
        if math.isfinite(velocity):
            self._set_velocities.append(velocity)
            if math.isnan(self._set_best_velocity_mps) or velocity > self._set_best_velocity_mps:
                self._set_best_velocity_mps = velocity
