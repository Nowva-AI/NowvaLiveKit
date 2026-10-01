"""Pure evidence-test functions for the diagnosis hypothesis engine.

Each test takes (features, anthro, rom, set_summary) and returns a float in
[0, 1]: how strongly the evidence supports that cause being active. ``features``
is the set's aggregate rep (the median of every per-rep measure), so a cause is
never judged on one rep that may not show its symptom. set_summary carries
set-level score trends and is None for single-rep sets.

Trunk lean is attributed with the sagittal balance model (diagnosis.lean_model):
the lean the reference lifter needs, the extra the athlete's proportions need,
the extra their ankles need, and whatever is left over.
"""

from __future__ import annotations

import math

from ..lean_model import UNRESTRICTED_SHANK_DEG
from ..types import RepKinematicSummary, SetScoreSummary
from .parameter_deltas import dorsi_driven_targets, foot_angle_target_deg

# Ankle dorsiflexion at the bottom of a squat, measured as shank tilt from
# vertical. Clinically restricted ankles reach under ~20°; ~30° and above is
# a normal, unrestricted squat. Absolute bounds on purpose: the athlete's own
# observed peak was once used as their "capacity", which fired the
# limited-ankle hypothesis on every athlete.
ANKLE_DF_LIMITED_DEG = 20.0
ANKLE_DF_UNRESTRICTED_DEG = UNRESTRICTED_SHANK_DEG

# Hip flexion here is the trunk-thigh closure (180° - angle at the hip). A free
# squat to parallel closes it to ~120°; under ~100° the hips limit depth.
HIP_FLEXION_LIMITED_DEG = 100.0
HIP_FLEXION_UNRESTRICTED_DEG = 120.0

# Left-right differences worth attributing a shift to.
UNILATERAL_ANKLE_GAP_DEG = 4.0
UNILATERAL_HIP_GAP_DEG = 5.0

# Descent initiation: hip-flexion change over knee-flexion change in the first
# fifth of the descent. ~1 is a squat that breaks at both joints together.
KNEE_DOMINANT_INITIATION_RATIO = 0.6
HIP_DOMINANT_INITIATION_RATIO = 1.5

CONTROLLED_DESCENT_S = 0.6
GOOD_TOE_OUT_DEG = 15.0
WIDE_STANCE_RATIO = 1.1


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if math.isnan(value):
        return low
    return max(low, min(high, value))


def _finite(value: float) -> bool:
    return value is not None and math.isfinite(value)


def _mean_toe_out(features: RepKinematicSummary) -> float:
    return (features.foot_direction_angle_l + features.foot_direction_angle_r) / 2.0


def _max_valgus(features: RepKinematicSummary) -> float:
    sides = [v for v in (features.knee_valgus_l, features.knee_valgus_r) if _finite(v)]
    return max(sides) if sides else math.nan


def _excess_lean(features: RepKinematicSummary) -> float:
    return features.trunk_pitch_at_bottom - features.expected_pitch_reference


def _lean_share(part: float, features: RepKinematicSummary) -> float:
    """Fraction of the athlete's excess lean that ``part`` (degrees) accounts for."""
    excess = _excess_lean(features)
    if not (_finite(part) and _finite(excess)) or excess <= 0.0:
        return 0.0
    return _clamp(part / excess)


def ankle_df_limitation(features: RepKinematicSummary) -> float:
    """How restricted the ankles look, 0 (unrestricted) to 1 (clearly limited)."""
    sides = [v for v in (features.ankle_df_l_max, features.ankle_df_r_max) if _finite(v)]
    if not sides:
        return 0.0
    peak_tilt = max(sides)
    if peak_tilt >= ANKLE_DF_UNRESTRICTED_DEG:
        return 0.0
    if peak_tilt <= ANKLE_DF_LIMITED_DEG:
        return 1.0
    return (ANKLE_DF_UNRESTRICTED_DEG - peak_tilt) / (ANKLE_DF_UNRESTRICTED_DEG - ANKLE_DF_LIMITED_DEG)


def hip_flexion_limitation(features: RepKinematicSummary, rom: dict) -> float:
    """How restricted the hips look, from the calibrated capacity and this set."""
    peaks = [v for v in (features.hip_flexion_l_max, features.hip_flexion_r_max) if _finite(v)]
    capacity = rom.get("peak_hip_flexion", HIP_FLEXION_UNRESTRICTED_DEG)
    best = max(peaks + [capacity])
    span = HIP_FLEXION_UNRESTRICTED_DEG - HIP_FLEXION_LIMITED_DEG
    return _clamp((HIP_FLEXION_UNRESTRICTED_DEG - best) / span)


def expected_zero(anthro: dict) -> float:
    return 0.0


def test_femur_torso_ratio(
    features: RepKinematicSummary,
    anthro: dict,
    rom: dict,
    set_summary: SetScoreSummary | None,
) -> float:
    anthro_part = features.expected_pitch_athlete - features.expected_pitch_reference
    return _lean_share(anthro_part, features)


def test_hip_anatomy(
    features: RepKinematicSummary,
    anthro: dict,
    rom: dict,
    set_summary: SetScoreSummary | None,
) -> float:
    # Structural only when the hips are limited and nothing else explains it.
    return hip_flexion_limitation(features, rom) * (1.0 - ankle_df_limitation(features)) * 0.6


def test_narrow_stance(
    features: RepKinematicSummary,
    anthro: dict,
    rom: dict,
    set_summary: SetScoreSummary | None,
) -> float:
    # Evidence is the stance gap to the personalized target alone; whether the
    # stance is causing a problem is established by the implicating symptom.
    dorsi_capacity = rom.get("peak_dorsiflexion", 35.0)
    ideal_ratio, _ = dorsi_driven_targets(dorsi_capacity, anthro)
    current_ratio = features.stance_width_ratio
    if not _finite(current_ratio) or current_ratio >= ideal_ratio:
        return 0.0
    return _clamp((ideal_ratio - current_ratio) / 0.5)


def test_narrow_foot_angle(
    features: RepKinematicSummary,
    anthro: dict,
    rom: dict,
    set_summary: SetScoreSummary | None,
) -> float:
    toe_out = _mean_toe_out(features)
    ideal_minimum = foot_angle_target_deg(anthro, rom)
    if not _finite(toe_out) or toe_out >= ideal_minimum:
        return 0.0
    return _clamp((ideal_minimum - toe_out) / ideal_minimum)


def test_stance_toe_mismatch(
    features: RepKinematicSummary,
    anthro: dict,
    rom: dict,
    set_summary: SetScoreSummary | None,
) -> float:
    toe_out = _mean_toe_out(features)
    stance = features.stance_width_ratio
    if not (_finite(toe_out) and _finite(stance)):
        return 0.0
    wide = _clamp((stance - WIDE_STANCE_RATIO) / 0.4)
    toes_forward = _clamp((GOOD_TOE_OUT_DEG - toe_out) / GOOD_TOE_OUT_DEG)
    return wide * toes_forward


def test_bracing_failure(
    features: RepKinematicSummary,
    anthro: dict,
    rom: dict,
    set_summary: SetScoreSummary | None,
) -> float:
    # Lean left over after anatomy and ankles, plus a chest that drops out of the hole.
    explained = features.expected_pitch_with_ankles
    if not _finite(explained):
        explained = features.expected_pitch_athlete
    residual = features.trunk_pitch_at_bottom - explained
    lean_evidence = _lean_share(residual, features) * _clamp(residual / 10.0)
    shoot_evidence = _clamp((features.hip_shoot_deg - 6.0) / 12.0) * 0.6
    return max(lean_evidence, shoot_evidence)


def test_knee_track_cue(
    features: RepKinematicSummary,
    anthro: dict,
    rom: dict,
    set_summary: SetScoreSummary | None,
) -> float:
    valgus = _max_valgus(features)
    if not _finite(valgus) or valgus < 4.0:
        return 0.0
    explained = _finite(_mean_toe_out(features)) and _mean_toe_out(features) < GOOD_TOE_OUT_DEG
    explained = explained or ankle_df_limitation(features) > 0.5
    return _clamp((valgus - 4.0) / 10.0) * (0.4 if explained else 0.8)


def test_weight_shift(
    features: RepKinematicSummary,
    anthro: dict,
    rom: dict,
    set_summary: SetScoreSummary | None,
) -> float:
    shift = abs(features.hip_shift_ratio)
    return _clamp((shift - 0.06) / 0.12)


def test_depth_unfamiliarity(
    features: RepKinematicSummary,
    anthro: dict,
    rom: dict,
    set_summary: SetScoreSummary | None,
) -> float:
    depth = features.depth_ratio
    if not _finite(depth) or depth <= 0.0:
        return 0.0
    capacity = rom.get("depth_capacity_ratio")
    if capacity is not None and _finite(capacity) and capacity <= depth - 0.10:
        # They went deeper in calibration — the range is there.
        return 0.8
    ankle_ok = ankle_df_limitation(features) < 0.5
    hip_ok = hip_flexion_limitation(features, rom) < 0.5
    base = 0.6 if ankle_ok and hip_ok else (0.3 if ankle_ok or hip_ok else 0.1)
    mechanical = max(
        test_narrow_stance(features, anthro, rom, set_summary),
        test_narrow_foot_angle(features, anthro, rom, set_summary),
    )
    return base * (1.0 - mechanical * 0.8)


def test_heel_elevation(
    features: RepKinematicSummary,
    anthro: dict,
    rom: dict,
    set_summary: SetScoreSummary | None,
) -> float:
    # The in-session fix for restricted ankles; ranks just under the ankle itself.
    return ankle_df_limitation(features) * 0.9


def test_foot_placement_asymmetry(
    features: RepKinematicSummary,
    anthro: dict,
    rom: dict,
    set_summary: SetScoreSummary | None,
) -> float:
    stagger = abs(features.stagger_ratio) if _finite(features.stagger_ratio) else 0.0
    flare = abs(features.foot_direction_angle_l - features.foot_direction_angle_r)
    flare = flare if _finite(flare) else 0.0
    return max(_clamp((stagger - 0.08) / 0.20), _clamp((flare - 6.0) / 12.0))


def test_knee_dominant_initiation(
    features: RepKinematicSummary,
    anthro: dict,
    rom: dict,
    set_summary: SetScoreSummary | None,
) -> float:
    ratio = features.initiation_ratio
    if not _finite(ratio):
        return 0.0
    return _clamp((KNEE_DOMINANT_INITIATION_RATIO - ratio) / 0.4)


def test_hip_dominant_initiation(
    features: RepKinematicSummary,
    anthro: dict,
    rom: dict,
    set_summary: SetScoreSummary | None,
) -> float:
    ratio = features.initiation_ratio
    if not _finite(ratio):
        return 0.0
    return _clamp((ratio - HIP_DOMINANT_INITIATION_RATIO) / 1.5)


def test_weight_forward(
    features: RepKinematicSummary,
    anthro: dict,
    rom: dict,
    set_summary: SetScoreSummary | None,
) -> float:
    balance = _clamp((features.balance_ratio - 0.10) / 0.20) if _finite(features.balance_ratio) else 0.0
    heels = _clamp((features.heel_rise_max_cm - 1.0) / 3.0) if _finite(features.heel_rise_max_cm) else 0.0
    return max(balance, heels)


def test_lockout_cue(
    features: RepKinematicSummary,
    anthro: dict,
    rom: dict,
    set_summary: SetScoreSummary | None,
) -> float:
    return _clamp((features.lockout_deficit_ratio - 0.03) / 0.08)


def test_eccentric_control(
    features: RepKinematicSummary,
    anthro: dict,
    rom: dict,
    set_summary: SetScoreSummary | None,
) -> float:
    descent = features.descent_time_s
    if not _finite(descent) or descent <= 0.0:
        return 0.0
    return _clamp((CONTROLLED_DESCENT_S - descent) / 0.3)


def test_gaze(
    features: RepKinematicSummary,
    anthro: dict,
    rom: dict,
    set_summary: SetScoreSummary | None,
) -> float:
    extension = -features.neck_flexion_deg if _finite(features.neck_flexion_deg) else 0.0
    return _clamp((extension - 20.0) / 20.0)


def test_progressive_degradation(
    features: RepKinematicSummary,
    anthro: dict,
    rom: dict,
    set_summary: SetScoreSummary | None,
) -> float:
    # Velocity loss is how strength coaches read proximity to failure; a
    # declining rep score catches form fading without the speed signal.
    velocity = _clamp((features.velocity_loss_pct - 15.0) / 25.0) if _finite(features.velocity_loss_pct) else 0.0
    trend = 0.0
    if set_summary is not None and len(set_summary.per_rep_scores) >= 3:
        decline_per_rep = -set_summary.trend_slope
        trend = _clamp((decline_per_rep - 0.01) / 0.04)
    return max(velocity, trend)


def test_limited_ankle_df(
    features: RepKinematicSummary,
    anthro: dict,
    rom: dict,
    set_summary: SetScoreSummary | None,
) -> float:
    return ankle_df_limitation(features)


def test_unilateral_ankle_df(
    features: RepKinematicSummary,
    anthro: dict,
    rom: dict,
    set_summary: SetScoreSummary | None,
) -> float:
    gap = abs(features.ankle_df_l_max - features.ankle_df_r_max)
    if not _finite(gap):
        return 0.0
    return _clamp((gap - UNILATERAL_ANKLE_GAP_DEG) / 8.0)


def test_limited_hip_flexion(
    features: RepKinematicSummary,
    anthro: dict,
    rom: dict,
    set_summary: SetScoreSummary | None,
) -> float:
    return hip_flexion_limitation(features, rom)


def test_unilateral_hip(
    features: RepKinematicSummary,
    anthro: dict,
    rom: dict,
    set_summary: SetScoreSummary | None,
) -> float:
    gap = abs(features.hip_flexion_l_max - features.hip_flexion_r_max)
    if not _finite(gap):
        return 0.0
    return _clamp((gap - UNILATERAL_HIP_GAP_DEG) / 10.0)


def test_weak_hip_abductors(
    features: RepKinematicSummary,
    anthro: dict,
    rom: dict,
    set_summary: SetScoreSummary | None,
) -> float:
    valgus = _max_valgus(features)
    if not _finite(valgus) or valgus < 5.0:
        return 0.0
    toe_out = _mean_toe_out(features)
    setup_good = (
        _finite(features.stance_width_ratio) and features.stance_width_ratio >= 0.8
        and _finite(toe_out) and toe_out >= GOOD_TOE_OUT_DEG
        and ankle_df_limitation(features) < 0.5
    )
    if setup_good:
        return _clamp((valgus - 5.0) / 8.0)
    return _clamp((valgus - 5.0) / 12.0) * 0.4


def test_quad_strength_limit(
    features: RepKinematicSummary,
    anthro: dict,
    rom: dict,
    set_summary: SetScoreSummary | None,
) -> float:
    shoot = features.hip_shoot_deg
    if not _finite(shoot):
        return 0.0
    return _clamp((shoot - 6.0) / 10.0) * (1.0 - 0.5 * ankle_df_limitation(features))


def test_strength_asymmetry(
    features: RepKinematicSummary,
    anthro: dict,
    rom: dict,
    set_summary: SetScoreSummary | None,
) -> float:
    # What is left when no mobility or setup difference explains the shift.
    explained = max(
        test_unilateral_ankle_df(features, anthro, rom, set_summary),
        test_unilateral_hip(features, anthro, rom, set_summary),
        test_foot_placement_asymmetry(features, anthro, rom, set_summary),
    )
    return test_weight_shift(features, anthro, rom, set_summary) * (1.0 - explained)
