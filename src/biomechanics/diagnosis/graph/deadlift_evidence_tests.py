"""Evidence tests for the deadlift graph (deadlift_causes.yaml), with the squat's
contract: (features, anthro, rom, set_summary) -> float in [0, 1], where features
is the set's median DeadliftRepSummary (cm in the sagittal frame, deadlift.types).
Ramps and weights are initial estimates, to be fitted on coach-labelled sets
(PLAN.md §8). NaN (not measured) is never evidence."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ..rep_scoring import _compute_trend_slope
from ..types import SetScoreSummary

if TYPE_CHECKING:
    from biomechanics.deadlift.diagnosis import DeadliftRepSummary

# The bar counts as over the midfoot within this (cm); D1 cues from 3 cm.
BAR_OVER_MIDFOOT_CM = 2.0
# Setup-model band for the shoulder joint ahead of the bar, cm (PLAN.md §2.7).
SHOULDER_BAND_LOW_CM = 0.0
SHOULDER_BAND_HIGH_CM = 6.0
# With the bar over the midfoot and the shoulders in their band, the setup
# model fixes the hip height: a hip reading outside its band then says more
# about the band (the lifter's build, or the hip keypoint) than the lifter.
SETUP_MODEL_DOUBT = 0.6
# Not measured directly (no hip range in the deadlift), so a mobility limit
# never outranks the in-session cue on the same reading.
MOBILITY_EVIDENCE_WEIGHT = 0.5
ANTHROPOMETRY_EVIDENCE_WEIGHT = 0.8
# A dead-stop rep may still not be pulled tight; touch-and-go reps never are.
DEAD_STOP_SLACK_WEIGHT = 0.7
# Mean bar speed under which the pull counts as slow, i.e. heavy for the lifter.
SLOW_PULL_MPS = 0.45
SLOW_PULL_RANGE_MPS = 0.20
UNKNOWN_SPEED_WEIGHT = 0.3
# Soft knees at the top are rarely the glutes alone.
KNEE_LOCKOUT_WEIGHT = 0.6
MIN_REPS_FOR_TREND = 3


def _finite(value: float) -> bool:
    return value is not None and math.isfinite(value)


def _ramp(value: float, start: float, span: float) -> float:
    # 0 at or below start, 1 at start + span; 0 when unmeasured.
    if not _finite(value):
        return 0.0
    return max(0.0, min(1.0, (value - start) / span))


def _below_band_cm(features: DeadliftRepSummary) -> float:
    return features.setup_hip_band_low_cm - features.setup_hip_height_cm


def _above_band_cm(features: DeadliftRepSummary) -> float:
    return features.setup_hip_height_cm - features.setup_hip_band_high_cm


def _setup_otherwise_right(features: DeadliftRepSummary) -> float:
    shoulders_in_band = (
        SHOULDER_BAND_LOW_CM <= features.shoulder_vs_bar_cm <= SHOULDER_BAND_HIGH_CM
    )
    bar_over_midfoot = abs(features.bar_midfoot_setup_cm) <= BAR_OVER_MIDFOOT_CM
    return 1.0 if shoulders_in_band and bar_over_midfoot else 0.0


def _hips_low_evidence(features: DeadliftRepSummary) -> float:
    return _ramp(_below_band_cm(features), 1.0, 6.0) * (
        1.0 - SETUP_MODEL_DOUBT * _setup_otherwise_right(features)
    )


def _unexplained_hip_shoot(features: DeadliftRepSummary) -> float:
    # Hips that start too low rise first to reach a pulling position; what
    # that does not explain is left to the pull itself.
    shoot = _ramp(features.trunk_change_liftoff_knee_deg, 5.0, 10.0)
    return shoot * (1.0 - _hips_low_evidence(features))


def _absolute(value: float) -> float:
    return abs(value) if _finite(value) else math.nan


def _asymmetry(features: DeadliftRepSummary) -> float:
    return max(
        _ramp(_absolute(features.hip_shift_ratio), 0.06, 0.12),
        _ramp(_absolute(features.bar_tilt_cm), 2.0, 4.0),
    )


def _symmetry_decline(set_summary: SetScoreSummary | None) -> float:
    # 0-1: how much worse the reps' symmetry got as the set went on.
    if set_summary is None:
        return 0.0
    scores = [
        score.symmetry_score
        for score in set_summary.per_rep_scores
        if _finite(score.symmetry_score)
    ]
    if len(scores) < MIN_REPS_FOR_TREND:
        return 0.0
    return _ramp(-_compute_trend_slope(scores), 0.01, 0.04)


def band_miss_cm(features: DeadliftRepSummary) -> float:
    """How far the setup hip height sits outside the setup model's band, cm
    (0 inside it, NaN when not measured)."""
    below = _below_band_cm(features)
    above = _above_band_cm(features)
    if not (_finite(below) and _finite(above)):
        return math.nan
    return max(below, above, 0.0)


def expected_zero(anthro: dict) -> float:
    return 0.0


def test_feet_position(
    features: DeadliftRepSummary,
    anthro: dict,
    rom: dict,
    set_summary: SetScoreSummary | None,
) -> float:
    # The bar was already off the midfoot while standing at it: the feet were
    # placed there. Without a standing reading the feet are the first suspect.
    offset = features.bar_midfoot_stance_cm
    if not _finite(offset):
        offset = features.bar_midfoot_setup_cm
    return _ramp(abs(offset), 1.5, 4.0)


def test_setup_habit(
    features: DeadliftRepSummary,
    anthro: dict,
    rom: dict,
    set_summary: SetScoreSummary | None,
) -> float:
    # The feet were right, but the bar moved while the lifter got set.
    roll = features.bar_midfoot_setup_cm - features.bar_midfoot_stance_cm
    return _ramp(abs(roll), 1.5, 4.0)


def test_hips_too_low(
    features: DeadliftRepSummary,
    anthro: dict,
    rom: dict,
    set_summary: SetScoreSummary | None,
) -> float:
    return _hips_low_evidence(features)


def test_hips_too_high(
    features: DeadliftRepSummary,
    anthro: dict,
    rom: dict,
    set_summary: SetScoreSummary | None,
) -> float:
    return _ramp(_above_band_cm(features), 1.0, 6.0) * (
        1.0 - SETUP_MODEL_DOUBT * _setup_otherwise_right(features)
    )


def test_hip_hamstring_mobility(
    features: DeadliftRepSummary,
    anthro: dict,
    rom: dict,
    set_summary: SetScoreSummary | None,
) -> float:
    # Lowering the hips needs more hip bend: hips kept high may be a range limit.
    return _ramp(_above_band_cm(features), 1.0, 6.0) * MOBILITY_EVIDENCE_WEIGHT


def test_setup_anthropometry(
    features: DeadliftRepSummary,
    anthro: dict,
    rom: dict,
    set_summary: SetScoreSummary | None,
) -> float:
    return (
        _ramp(band_miss_cm(features), 1.0, 6.0)
        * _setup_otherwise_right(features)
        * ANTHROPOMETRY_EVIDENCE_WEIGHT
    )


def test_shoulder_position(
    features: DeadliftRepSummary,
    anthro: dict,
    rom: dict,
    set_summary: SetScoreSummary | None,
) -> float:
    return _ramp(-features.shoulder_vs_bar_cm, 0.5, 4.0)


def test_slack_not_pulled(
    features: DeadliftRepSummary,
    anthro: dict,
    rom: dict,
    set_summary: SetScoreSummary | None,
) -> float:
    rushed = 1.0 if features.touch_and_go else DEAD_STOP_SLACK_WEIGHT
    return _unexplained_hip_shoot(features) * rushed


def test_weak_off_floor(
    features: DeadliftRepSummary,
    anthro: dict,
    rom: dict,
    set_summary: SetScoreSummary | None,
) -> float:
    velocity = features.concentric_velocity_mps
    if _finite(velocity):
        slow = _ramp(SLOW_PULL_MPS - velocity, 0.0, SLOW_PULL_RANGE_MPS)
    else:
        slow = UNKNOWN_SPEED_WEIGHT
    return _unexplained_hip_shoot(features) * slow


def test_load_too_heavy(
    features: DeadliftRepSummary,
    anthro: dict,
    rom: dict,
    set_summary: SetScoreSummary | None,
) -> float:
    # Bar speed lost over the set says how close it was to failure; a falling
    # rep score catches form fading without the speed signal.
    velocity = _ramp(features.velocity_loss_pct, 15.0, 25.0)
    trend = 0.0
    if set_summary is not None and len(set_summary.per_rep_scores) >= MIN_REPS_FOR_TREND:
        trend = _ramp(-set_summary.trend_slope, 0.01, 0.04)
    return max(velocity, trend)


def test_bar_far_at_setup(
    features: DeadliftRepSummary,
    anthro: dict,
    rom: dict,
    set_summary: SetScoreSummary | None,
) -> float:
    return _ramp(features.bar_midfoot_setup_cm, 1.0, 4.0)


def test_lats_not_engaged(
    features: DeadliftRepSummary,
    anthro: dict,
    rom: dict,
    set_summary: SetScoreSummary | None,
) -> float:
    # Drift the starting position does not explain.
    started_far = test_bar_far_at_setup(features, anthro, rom, set_summary)
    return _ramp(features.bar_drift_cm, 2.0, 5.0) * (1.0 - 0.5 * started_far)


def test_glutes_not_finishing(
    features: DeadliftRepSummary,
    anthro: dict,
    rom: dict,
    set_summary: SetScoreSummary | None,
) -> float:
    return max(
        _ramp(features.hip_extension_deficit_deg, 4.0, 8.0),
        KNEE_LOCKOUT_WEIGHT * _ramp(features.knee_extension_deficit_deg, 4.0, 8.0),
    )


def test_over_extension_habit(
    features: DeadliftRepSummary,
    anthro: dict,
    rom: dict,
    set_summary: SetScoreSummary | None,
) -> float:
    # A lean-back with the hips not through is the glutes, not a habit.
    hips_short = _ramp(features.hip_extension_deficit_deg, 4.0, 8.0)
    return _ramp(features.lean_back_deg, 4.0, 8.0) * (1.0 - 0.5 * hips_short)


def test_uneven_stance(
    features: DeadliftRepSummary,
    anthro: dict,
    rom: dict,
    set_summary: SetScoreSummary | None,
) -> float:
    # Set up uneven: the slide is there from the first rep and stays.
    return _ramp(_absolute(features.hip_shift_ratio), 0.06, 0.12) * (
        1.0 - _symmetry_decline(set_summary)
    )


def test_uneven_grip(
    features: DeadliftRepSummary,
    anthro: dict,
    rom: dict,
    set_summary: SetScoreSummary | None,
) -> float:
    return _ramp(_absolute(features.bar_tilt_cm), 2.0, 4.0) * (
        1.0 - _symmetry_decline(set_summary)
    )


def test_unilateral_weakness(
    features: DeadliftRepSummary,
    anthro: dict,
    rom: dict,
    set_summary: SetScoreSummary | None,
) -> float:
    # A weaker side tires first: the asymmetry grows as the set goes on.
    return _asymmetry(features) * _symmetry_decline(set_summary)
