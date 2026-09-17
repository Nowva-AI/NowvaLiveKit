"""Per-rep quality scoring for squat biomechanics.

Scores each rep on the five dimensions the squat profile actually detects —
depth, trunk control, knee tracking, symmetry, tempo — and produces a weighted
composite score. All scores are in [0, 1] where 1.0 = perfect.

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

from .graph.evidence_tests import _clamp, expected_trunk_lean_geometric
from .types import RepKinematicSummary, RepScore, RepTrajectory, RepTrajectorySample, SetScoreSummary

WEIGHT_DEPTH = 0.42
WEIGHT_TRUNK = 0.20
WEIGHT_KNEES = 0.16
WEIGHT_TEMPO = 0.12
WEIGHT_SYMMETRY = 0.10

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
# thigh. Depth decays linearly across that span.
DEPTH_DECAY_RATIO = 1.0

TRUNK_TOLERANCE_DEGREES = 3.0
TRUNK_DECAY_RANGE_DEGREES = 20.0

KNEE_PERFECT_ZONE_DEGREES = 4.0
KNEE_DECAY_RANGE_DEGREES = 12.0

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
    return [
        _hip_above_knee_ratio(
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
    cutoff = min(ratios) + LOADED_WINDOW_RATIO
    return [
        sample
        for sample, ratio in zip(trajectory.samples, ratios)
        if ratio <= cutoff
    ]


def score_depth(
    rep: RepKinematicSummary,
    anthro: dict,
    rom: dict,
    trajectory: RepTrajectory | None = None,
) -> float:
    femur_cm = _femur_length_cm(anthro)

    if trajectory is not None and trajectory.samples:
        ratios = _sample_ratios(trajectory, femur_cm)
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
    return _clamp(1.0 - deepest_ratio / DEPTH_DECAY_RATIO)


def score_trunk_control(
    rep: RepKinematicSummary,
    anthro: dict,
    rom: dict,
    trajectory: RepTrajectory | None = None,
) -> float:
    expected_lean = expected_trunk_lean_geometric(anthro)
    loaded = _loaded_samples(trajectory, _femur_length_cm(anthro))

    if loaded:
        deviations = [abs(sample.trunk_pitch - expected_lean) for sample in loaded]
        deviation = _percentile(deviations, WORST_PERCENTILE)
    else:
        deviation = abs(rep.trunk_pitch_at_bottom - expected_lean)

    if deviation <= TRUNK_TOLERANCE_DEGREES:
        return 1.0
    return _clamp(
        1.0 - (deviation - TRUNK_TOLERANCE_DEGREES) / TRUNK_DECAY_RANGE_DEGREES
    )


def _score_single_knee(deviation_deg: float) -> float:
    abs_deviation = abs(deviation_deg)
    if abs_deviation <= KNEE_PERFECT_ZONE_DEGREES:
        return 1.0
    return _clamp(
        1.0 - (abs_deviation - KNEE_PERFECT_ZONE_DEGREES) / KNEE_DECAY_RANGE_DEGREES
    )


def score_knee_tracking(
    rep: RepKinematicSummary,
    anthro: dict,
    rom: dict,
    trajectory: RepTrajectory | None = None,
) -> float:
    loaded = _loaded_samples(trajectory, _femur_length_cm(anthro))

    if loaded:
        valgus_l = _percentile(
            [abs(sample.knee_valgus_l) for sample in loaded], WORST_PERCENTILE
        )
        valgus_r = _percentile(
            [abs(sample.knee_valgus_r) for sample in loaded], WORST_PERCENTILE
        )
    else:
        valgus_l = rep.knee_valgus_l
        valgus_r = rep.knee_valgus_r

    return 0.5 * _score_single_knee(valgus_l) + 0.5 * _score_single_knee(valgus_r)


def score_symmetry(
    rep: RepKinematicSummary,
    anthro: dict,
    rom: dict,
    trajectory: RepTrajectory | None = None,
) -> float:
    loaded = _loaded_samples(trajectory, _femur_length_cm(anthro))

    if loaded:
        asymmetry_cm = _percentile(
            [abs(sample.hip_y_l - sample.hip_y_r) for sample in loaded],
            WORST_PERCENTILE,
        )
    else:
        asymmetry_cm = abs(rep.hip_y_l_at_bottom - rep.hip_y_r_at_bottom)

    if asymmetry_cm <= SYMMETRY_TOLERANCE_CM:
        return 1.0
    return _clamp(
        1.0 - (asymmetry_cm - SYMMETRY_TOLERANCE_CM) / SYMMETRY_DECAY_RANGE_CM
    )


def _score_phase_duration(
    duration_seconds: float, ideal_min: float, ideal_max: float
) -> float:
    # A missing or non-positive duration means the phase was never timed.
    # Scoring it zero would punish the athlete for a pipeline gap.
    if duration_seconds <= 0.0:
        return 1.0
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
    return min(eccentric, concentric)


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

    composite = (
        depth * WEIGHT_DEPTH
        + trunk_control * WEIGHT_TRUNK
        + knee_tracking * WEIGHT_KNEES
        + symmetry * WEIGHT_SYMMETRY
        + tempo * WEIGHT_TEMPO
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
    composites = [score.composite_score for score in per_rep_scores]

    mean_score = sum(composites) / len(composites)
    best = max(per_rep_scores, key=lambda s: s.composite_score)
    worst = min(per_rep_scores, key=lambda s: s.composite_score)

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
