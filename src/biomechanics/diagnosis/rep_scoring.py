"""Per-rep quality scoring for squat biomechanics.

Scores each rep on five dimensions — depth against the athlete's own target,
trunk control (lean beyond what their build and ankles need, and the chest
dropping out of the hole), knee tracking, side-to-side hip shift, and tempo —
and produces a weighted composite. All scores are in [0, 1], 1.0 = perfect.
Penalties are one-sided: sitting more upright than needed or pushing the
knees out is never scored as a fault. Position control outweighs depth.

Fault dimensions are measured over the loaded portion of the rep rather than a
single bottom frame, using a high percentile so one mistracked frame cannot
sink an otherwise clean rep. Depth uses the low percentile for the same reason.

The scorer never modifies the data it is handed: it does not ground, centre,
scale, or filter keypoints. Every measurement is a within-frame difference or
an angle, so the numbers are translation-invariant and raw pipeline
coordinates score identically to preprocessed ones.
"""

from __future__ import annotations

import math

from .graph.evidence_tests import _clamp
from .types import (
    DeadliftRepScore,
    RepKinematicSummary,
    RepScore,
    RepTrajectory,
    RepTrajectorySample,
    SetScoreSummary,
)

WEIGHT_DEPTH = 0.20
WEIGHT_TRUNK = 0.25
WEIGHT_KNEES = 0.25
WEIGHT_TEMPO = 0.15
WEIGHT_SYMMETRY = 0.15

# Percentile of the per-frame series used for each metric. Fault metrics take
# the high end (worst sustained value), depth takes the low end (deepest
# sustained hip position). Both reject single-frame outliers.
WORST_PERCENTILE = 0.90
DEEPEST_PERCENTILE = 0.10

# Frames counted as "loaded" — those within this fraction of a femur length of
# the deepest hip position. Trunk lean, valgus and pelvic level are only
# meaningful near the bottom; measuring them across the whole rep would score
# every athlete against a bottom-of-squat expectation while they stand upright.
LOADED_WINDOW_RATIO = 0.25

DEFAULT_FEMUR_LENGTH_M = 0.42
MIN_FEMUR_LENGTH_M = 0.20

# Hip-above-knee height as a fraction of femur length: 0.0 is parallel (hip
# joint centre level with knee joint centre), ~1.0 is standing with a vertical
# thigh. A rep within the tolerance of the athlete's target scores full depth;
# beyond it the score decays linearly over DEPTH_DECAY_RATIO.
DEPTH_TARGET_TOLERANCE_RATIO = 0.08
DEPTH_DECAY_RATIO = 0.75

TRUNK_TOLERANCE_DEGREES = 3.0
TRUNK_DECAY_RANGE_DEGREES = 20.0
HIP_SHOOT_TOLERANCE_DEGREES = 4.0
HIP_SHOOT_DECAY_RANGE_DEGREES = 14.0

KNEE_PERFECT_ZONE_DEGREES = 4.0
KNEE_DECAY_RANGE_DEGREES = 12.0
# Knees pushed out past the toes only costs score well beyond normal.
KNEE_LATERAL_TOLERANCE_DEGREES = 15.0

# Sideways hip travel as a fraction of ankle separation.
SYMMETRY_TOLERANCE_RATIO = 0.03
SYMMETRY_DECAY_RANGE_RATIO = 0.15
SYMMETRY_TOLERANCE_CM = 1.0
SYMMETRY_DECAY_RANGE_CM = 5.0

# Controlled descent, then an ascent that is forceful without grinding.
# Outside these windows the score decays over TEMPO_DECAY_RANGE_SECONDS, so
# both dive-bombing and stalling are penalised. The concentric ceiling is the
# looser of the two: a hard final rep legitimately takes time to stand up,
# and only a genuine grind should read as a fault.
ECCENTRIC_IDEAL_MIN_SECONDS = 1.0
ECCENTRIC_IDEAL_MAX_SECONDS = 3.0
CONCENTRIC_IDEAL_MIN_SECONDS = 0.5
CONCENTRIC_IDEAL_MAX_SECONDS = 3.0
TEMPO_DECAY_RANGE_SECONDS = 1.5


def _percentile(values: list[float], fraction: float) -> float:
    # Missing angles are NaN (C9) and would poison the sort; a rep with no
    # finite sample scores as if the value were absent.
    ordered = sorted(value for value in values if not math.isnan(value))
    if not ordered:
        return math.nan
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    weight = position - lower_index
    return ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight


def _femur_length_cm(anthro: dict) -> float:
    femur_m = anthro.get("femur_length_avg", DEFAULT_FEMUR_LENGTH_M)
    return max(femur_m, MIN_FEMUR_LENGTH_M) * 100.0


def _hip_above_knee_ratio(
    hip_y_l: float, hip_y_r: float, knee_y_l: float, knee_y_r: float, femur_cm: float
) -> float:
    hip_mid_y = (hip_y_l + hip_y_r) / 2.0
    knee_mid_y = (knee_y_l + knee_y_r) / 2.0
    return (hip_mid_y - knee_mid_y) / femur_cm


def _sample_ratios(
    trajectory: RepTrajectory, femur_cm: float
) -> list[float]:
    """Depth ratio per frame: the measured one, or hip-above-knee from Y-up heights."""
    return [
        sample.depth_ratio
        if math.isfinite(sample.depth_ratio)
        else _hip_above_knee_ratio(
            sample.hip_y_l, sample.hip_y_r, sample.knee_y_l, sample.knee_y_r, femur_cm
        )
        for sample in trajectory.samples
    ]


def _loaded_samples(
    trajectory: RepTrajectory | None, femur_cm: float
) -> list[RepTrajectorySample]:
    """Frames near the bottom of the rep, where fault kinematics are meaningful."""
    if trajectory is None or not trajectory.samples:
        return []

    ratios = _sample_ratios(trajectory, femur_cm)
    finite = [ratio for ratio in ratios if math.isfinite(ratio)]
    if not finite:
        return []
    cutoff = min(finite) + LOADED_WINDOW_RATIO
    return [
        sample
        for sample, ratio in zip(trajectory.samples, ratios)
        if math.isfinite(ratio) and ratio <= cutoff
    ]


def _expected_pitch(rep: RepKinematicSummary) -> float:
    """The lean this athlete's build and ankles need (balance model), best available."""
    for value in (rep.expected_pitch_with_ankles, rep.expected_pitch_athlete, rep.expected_pitch_reference):
        if math.isfinite(value):
            return value
    return math.nan


def score_depth(
    rep: RepKinematicSummary,
    anthro: dict,
    rom: dict,
    trajectory: RepTrajectory | None = None,
) -> float:
    femur_cm = _femur_length_cm(anthro)

    if trajectory is not None and trajectory.samples:
        ratios = _sample_ratios(trajectory, femur_cm)
    elif math.isfinite(rep.depth_ratio):
        ratios = [rep.depth_ratio]
    else:
        ratios = [
            _hip_above_knee_ratio(
                rep.hip_y_l_at_bottom,
                rep.hip_y_r_at_bottom,
                rep.knee_y_l_at_bottom,
                rep.knee_y_r_at_bottom,
                femur_cm,
            )
        ]

    deepest_ratio = _percentile(ratios, DEEPEST_PERCENTILE)
    if math.isnan(deepest_ratio):
        return math.nan
    target = rom.get("depth_target_ratio", 0.0)
    shortfall = deepest_ratio - target - DEPTH_TARGET_TOLERANCE_RATIO
    return _clamp(1.0 - max(0.0, shortfall) / DEPTH_DECAY_RATIO)


def score_trunk_control(
    rep: RepKinematicSummary,
    anthro: dict,
    rom: dict,
    trajectory: RepTrajectory | None = None,
) -> float:
    expected_lean = _expected_pitch(rep)
    loaded = _loaded_samples(trajectory, _femur_length_cm(anthro))

    if loaded:
        excess = _percentile([sample.trunk_pitch - expected_lean for sample in loaded], WORST_PERCENTILE)
    else:
        excess = rep.trunk_pitch_at_bottom - expected_lean

    if not (math.isfinite(excess) or math.isfinite(rep.hip_shoot_deg)):
        return math.nan
    lean_score = 1.0
    if math.isfinite(excess) and excess > TRUNK_TOLERANCE_DEGREES:
        lean_score = _clamp(1.0 - (excess - TRUNK_TOLERANCE_DEGREES) / TRUNK_DECAY_RANGE_DEGREES)

    shoot_score = 1.0
    if math.isfinite(rep.hip_shoot_deg) and rep.hip_shoot_deg > HIP_SHOOT_TOLERANCE_DEGREES:
        shoot_score = _clamp(
            1.0 - (rep.hip_shoot_deg - HIP_SHOOT_TOLERANCE_DEGREES) / HIP_SHOOT_DECAY_RANGE_DEGREES
        )
    return min(lean_score, shoot_score)


def _score_single_knee(deviation_deg: float) -> float:
    if math.isnan(deviation_deg):
        return math.nan
    if deviation_deg < 0.0:
        # Knees out past the toes: only a long way out is a fault.
        lateral = -deviation_deg
        if lateral <= KNEE_LATERAL_TOLERANCE_DEGREES:
            return 1.0
        return _clamp(1.0 - (lateral - KNEE_LATERAL_TOLERANCE_DEGREES) / KNEE_DECAY_RANGE_DEGREES)
    if deviation_deg <= KNEE_PERFECT_ZONE_DEGREES:
        return 1.0
    return _clamp(
        1.0 - (deviation_deg - KNEE_PERFECT_ZONE_DEGREES) / KNEE_DECAY_RANGE_DEGREES
    )


def score_knee_tracking(
    rep: RepKinematicSummary,
    anthro: dict,
    rom: dict,
    trajectory: RepTrajectory | None = None,
) -> float:
    loaded = _loaded_samples(trajectory, _femur_length_cm(anthro))

    if loaded:
        valgus_l = _percentile([sample.knee_valgus_l for sample in loaded], WORST_PERCENTILE)
        valgus_r = _percentile([sample.knee_valgus_r for sample in loaded], WORST_PERCENTILE)
    else:
        valgus_l = rep.knee_valgus_l
        valgus_r = rep.knee_valgus_r

    sides = [score for score in (_score_single_knee(valgus_l), _score_single_knee(valgus_r)) if math.isfinite(score)]
    return sum(sides) / len(sides) if sides else math.nan


def score_symmetry(
    rep: RepKinematicSummary,
    anthro: dict,
    rom: dict,
    trajectory: RepTrajectory | None = None,
) -> float:
    if math.isfinite(rep.hip_shift_ratio):
        shift = abs(rep.hip_shift_ratio)
        if shift <= SYMMETRY_TOLERANCE_RATIO:
            return 1.0
        return _clamp(1.0 - (shift - SYMMETRY_TOLERANCE_RATIO) / SYMMETRY_DECAY_RANGE_RATIO)

    # Without a measured hip shift, fall back to pelvic level at the bottom.
    loaded = _loaded_samples(trajectory, _femur_length_cm(anthro))
    if loaded:
        asymmetry_cm = _percentile(
            [abs(sample.hip_y_l - sample.hip_y_r) for sample in loaded],
            WORST_PERCENTILE,
        )
    else:
        asymmetry_cm = abs(rep.hip_y_l_at_bottom - rep.hip_y_r_at_bottom)

    if math.isnan(asymmetry_cm):
        return math.nan
    if asymmetry_cm <= SYMMETRY_TOLERANCE_CM:
        return 1.0
    return _clamp(
        1.0 - (asymmetry_cm - SYMMETRY_TOLERANCE_CM) / SYMMETRY_DECAY_RANGE_CM
    )


def _score_phase_duration(
    duration_seconds: float, ideal_min: float, ideal_max: float
) -> float:
    # A missing or non-positive duration means the phase was never timed:
    # unmeasured, neither a fault nor a perfect score.
    if not math.isfinite(duration_seconds) or duration_seconds <= 0.0:
        return math.nan
    if ideal_min <= duration_seconds <= ideal_max:
        return 1.0

    if duration_seconds < ideal_min:
        excess = ideal_min - duration_seconds
    else:
        excess = duration_seconds - ideal_max
    return _clamp(1.0 - excess / TEMPO_DECAY_RANGE_SECONDS)


def score_tempo(
    rep: RepKinematicSummary,
    anthro: dict,
    rom: dict,
    trajectory: RepTrajectory | None = None,
) -> float:
    eccentric = _score_phase_duration(
        rep.descent_time_s, ECCENTRIC_IDEAL_MIN_SECONDS, ECCENTRIC_IDEAL_MAX_SECONDS
    )
    concentric = _score_phase_duration(
        rep.ascent_time_s, CONCENTRIC_IDEAL_MIN_SECONDS, CONCENTRIC_IDEAL_MAX_SECONDS
    )
    # Weakest phase wins: a dive-bombed descent is a bad rep however clean the
    # ascent was, and averaging would let the good half hide it.
    timed = [score for score in (eccentric, concentric) if math.isfinite(score)]
    return min(timed) if timed else math.nan


def score_rep(
    rep: RepKinematicSummary,
    anthro: dict,
    rom: dict,
    trajectory: RepTrajectory | None = None,
) -> RepScore:
    depth = score_depth(rep, anthro, rom, trajectory)
    trunk_control = score_trunk_control(rep, anthro, rom, trajectory)
    knee_tracking = score_knee_tracking(rep, anthro, rom, trajectory)
    symmetry = score_symmetry(rep, anthro, rom, trajectory)
    tempo = score_tempo(rep, anthro, rom, trajectory)

    # Unmeasured dimensions are left out of the composite, never scored
    # perfect: a rep whose knees were not seen earns no knee credit.
    weighted = [
        (score, weight)
        for score, weight in (
            (depth, WEIGHT_DEPTH),
            (trunk_control, WEIGHT_TRUNK),
            (knee_tracking, WEIGHT_KNEES),
            (symmetry, WEIGHT_SYMMETRY),
            (tempo, WEIGHT_TEMPO),
        )
        if math.isfinite(score)
    ]
    total_weight = sum(weight for _, weight in weighted)
    composite = (
        sum(score * weight for score, weight in weighted) / total_weight
        if total_weight > 0.0
        else math.nan
    )

    return RepScore(
        rep_number=rep.rep_number,
        depth_score=round(depth, 3),
        trunk_control_score=round(trunk_control, 3),
        knee_tracking_score=round(knee_tracking, 3),
        symmetry_score=round(symmetry, 3),
        tempo_score=round(tempo, 3),
        composite_score=round(composite, 3),
    )


def score_set(
    reps: list[RepKinematicSummary],
    anthro: dict,
    rom: dict,
    trajectories: list[RepTrajectory | None] | None = None,
) -> SetScoreSummary:
    if trajectories is None:
        trajectories = [None] * len(reps)

    per_rep_scores = [
        score_rep(rep, anthro, rom, trajectory)
        for rep, trajectory in zip(reps, trajectories)
    ]
    return summarize_rep_scores(per_rep_scores)


def summarize_rep_scores(
    per_rep_scores: list[RepScore | DeadliftRepScore],
) -> SetScoreSummary:
    """Set statistics over a set's rep scores, whatever the exercise."""
    # Reps with nothing measurable are left out of the set statistics.
    scored = [score for score in per_rep_scores if math.isfinite(score.composite_score)] or per_rep_scores
    composites = [score.composite_score for score in scored]

    mean_score = sum(composites) / len(composites)
    best = max(scored, key=lambda s: s.composite_score)
    worst = min(scored, key=lambda s: s.composite_score)

    trend_slope = _compute_trend_slope(composites)

    return SetScoreSummary(
        mean_score=round(mean_score, 3),
        best_rep_number=best.rep_number,
        worst_rep_number=worst.rep_number,
        trend_slope=round(trend_slope, 4),
        per_rep_scores=per_rep_scores,
    )


def _compute_trend_slope(values: list[float]) -> float:
    num_values = len(values)
    if num_values < 2:
        return 0.0

    mean_x = (num_values - 1) / 2.0
    mean_y = sum(values) / num_values

    covariance = sum(
        (index - mean_x) * (value - mean_y)
        for index, value in enumerate(values)
    )
    variance_x = sum((index - mean_x) ** 2 for index in range(num_values))

    if variance_x == 0.0:
        return 0.0
    return covariance / variance_x
