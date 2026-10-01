"""
Biomechanical Calibration Module

Measures what the athlete's body can do from calibration reps — ankle
dorsiflexion, hip flexion and the depth they reach — and derives their depth
target from it. It never moves a fault threshold: a knee that caves on the
calibration reps is still caving. A personal baseline only ever flags drift
within a session (faults.session_reference).
"""

import math
import statistics
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Exercise -> Movement Pattern mapping
# ---------------------------------------------------------------------------

EXERCISE_TO_MOVEMENT_PATTERN: Dict[str, str] = {
    # Squat variants
    "Barbell Back Squat": "squat",
    "Barbell Front Squat": "squat",
    "Goblet Squat": "squat",
    # Hip hinge variants (future)
    "Barbell Deadlift": "hip_hinge",
    "Romanian Deadlift": "hip_hinge",
    "Sumo Deadlift": "hip_hinge",
    # Push variants (future)
    "Barbell Bench Press": "horizontal_push",
    "Barbell Overhead Press": "vertical_push",
}


def get_movement_pattern(exercise_name: str) -> Optional[str]:
    """Return the movement pattern for an exercise, or None if not mapped."""
    return EXERCISE_TO_MOVEMENT_PATTERN.get(exercise_name)


# ---------------------------------------------------------------------------
# Calibration profile building
# ---------------------------------------------------------------------------

def depth_target_ratio(capacity_ratio: Optional[float], depth_config) -> float:
    """The athlete's depth target from the depth they reached in calibration.

    Depth is hip height above the knee in femur lengths (0 = parallel). An
    athlete who reaches parallel is held to the configured default; one who
    cannot is held to their own range — never shallower than the cap, since a
    quarter squat is not a squat rep.
    """
    default = depth_config.default_target_ratio
    if capacity_ratio is None or math.isnan(capacity_ratio):
        return default
    return min(max(capacity_ratio, default), depth_config.max_target_ratio)


def build_calibration_profile(peaks: dict, config=None) -> dict:
    """The per-athlete parameters calibration sets: today, only the depth target."""
    from biomechanics.config import DepthFaultConfig

    depth_config = config.faults.depth if config is not None else DepthFaultConfig()
    capacity = peaks.get("depth_capacity_ratio")
    return {
        "depth_capacity_ratio": capacity,
        "depth_target_ratio": depth_target_ratio(capacity, depth_config),
    }


def apply_calibration_to_rule_engine(rule_engine, profile: dict) -> None:
    """Install a calibration profile. Profiles stored before depth targets
    existed carry only per-athlete fault thresholds, which are ignored on
    purpose — thresholds no longer come from observed reps."""
    target = profile.get("depth_target_ratio")
    if target is not None:
        rule_engine.set_depth_target(target)


def extract_thresholds_from_rule_engine(rule_engine) -> dict:
    """Read current thresholds from a rule engine for dashboard visualization."""
    thresholds: dict = {}
    for rule in rule_engine.rules:
        if all(hasattr(rule, name) for name in ("mild_threshold", "moderate_threshold", "severe_threshold")):
            thresholds[rule.fault_type.value] = {
                "mild": rule.mild_threshold,
                "moderate": rule.moderate_threshold,
                "severe": rule.severe_threshold,
            }
    if rule_engine.depth_target_ratio is not None:
        thresholds["depth"] = {"target_ratio": rule_engine.depth_target_ratio}
    return thresholds


# ---------------------------------------------------------------------------
# CalibrationTracker — collects peak data during calibration reps
# ---------------------------------------------------------------------------

class CalibrationTracker:
    """Collects the athlete's capacities over calibration reps.

    Usage:
        tracker = CalibrationTracker(target_reps=5)
        while not tracker.is_complete:
            result = pipeline.process_frame()
            if result.joint_angles:
                tracker.record_frame(result.joint_angles, in_rep=...)
            if result.rep_data:
                tracker.on_rep_complete(result.rep_data.max_depth_angle, result.rep_data.features)
        peaks = tracker.get_peaks()

    Per-rep capacities come from the rep features (ankle and hip peaks are a
    95th percentile over the bottom of the rep, never a single frame) and the
    calibration value is the median rep, so one odd rep cannot set it.
    """

    def __init__(self, target_reps: int = 5):
        self.target_reps = target_reps
        self.reps_completed = 0

        self.rep_depth_peaks: list = []
        self._current_rep_peak_depth = 0.0
        self.rep_depth_ratios: list = []
        self.rep_dorsiflexion_peaks: list = []
        self.rep_hip_flexion_peaks: list = []

        # Standing knee flexion (for gate widening after calibration)
        self.standing_knee_flexion = 0.0

    def record_frame(self, joint_angles, in_rep: bool = False) -> None:
        """Record a single frame's angles during calibration."""
        ja = joint_angles

        # Standing knee flexion: only track when NOT in a rep so the gate
        # threshold reflects actual standing posture, not squat depth.
        if not in_rep:
            standing_knee = max(ja.knee_flexion_l, ja.knee_flexion_r)
            self.standing_knee_flexion = max(self.standing_knee_flexion, standing_knee)

        # Per-rep peak knee flexion, kept for the calibration report.
        self._current_rep_peak_depth = max(
            self._current_rep_peak_depth,
            ja.avg_knee_flexion,
        )

    def on_rep_complete(self, depth_angle: float = 0.0, features: Optional[dict] = None) -> None:
        """Called when a rep completes during calibration."""
        self.reps_completed += 1
        # Prefer self-tracked depth (from record_frame) over external depth_angle,
        # because when BiLSTM is enabled the rep counter may have already reset
        # its depth tracking by the time the BiLSTM rep event fires.
        depth = self._current_rep_peak_depth if self._current_rep_peak_depth > 0.0 else depth_angle
        if depth > 0.0:
            self.rep_depth_peaks.append(depth)
        self._current_rep_peak_depth = 0.0

        if not features:
            return
        _append_finite(self.rep_depth_ratios, features.get("depth_ratio"))
        _append_finite(
            self.rep_dorsiflexion_peaks,
            _nanmax(features.get("dorsiflexion_max_l"), features.get("dorsiflexion_max_r")),
        )
        _append_finite(
            self.rep_hip_flexion_peaks,
            _nanmax(features.get("hip_flexion_max_l"), features.get("hip_flexion_max_r")),
        )

    @property
    def is_complete(self) -> bool:
        return self.reps_completed >= self.target_reps

    def get_peaks(self) -> dict:
        """The athlete's capacities: median rep for each measure (None when unmeasured)."""
        return {
            "peak_dorsiflexion": _median_or_none(self.rep_dorsiflexion_peaks),
            "peak_hip_flexion": _median_or_none(self.rep_hip_flexion_peaks),
            "depth_capacity_ratio": _median_or_none(self.rep_depth_ratios),
            "avg_depth": (
                sum(self.rep_depth_peaks) / len(self.rep_depth_peaks)
                if self.rep_depth_peaks else 0.0
            ),
            "depth_per_rep": self.rep_depth_peaks,
            "depth_ratio_per_rep": self.rep_depth_ratios,
        }


def _append_finite(values: list, value) -> None:
    if value is not None and math.isfinite(value):
        values.append(float(value))


def _nanmax(first, second) -> Optional[float]:
    finite = [value for value in (first, second) if value is not None and math.isfinite(value)]
    return max(finite) if finite else None


def _median_or_none(values: list) -> Optional[float]:
    return statistics.median(values) if values else None
