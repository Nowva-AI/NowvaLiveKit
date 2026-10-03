"""Set diagnosis for the conventional deadlift (docs/deadlift/PLAN.md §7): the
HypothesisEngine on the deadlift graph, per-rep summaries of DeadliftRepFeatures,
rep and set scores on the deadlift's five dimensions, and DeadliftSetDiagnosis,
which re-diagnoses the set after every rep and at set end (squat contract shape).
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel

from biomechanics.diagnosis.engine import LOAD_REDUCTION_PCT, DiagnosisGraph, HypothesisEngine
from biomechanics.diagnosis.graph import deadlift_evidence_tests, deadlift_parameter_deltas
from biomechanics.diagnosis.graph.deadlift_evidence_tests import band_miss_cm
from biomechanics.diagnosis.graph.deadlift_parameter_deltas import (
    delta_bar_closer,
    delta_bar_start_closer,
    delta_feet_position,
    delta_hips_down,
    delta_hips_up,
    delta_level_bar,
    delta_setup_habit,
    delta_shoulders_forward,
)
from biomechanics.diagnosis.graph.loader import load_graph
from biomechanics.diagnosis.rep_scoring import summarize_rep_scores
from biomechanics.diagnosis.types import DeadliftRepScore, DiagnosisResult, SetScoreSummary

from .types import NAN, DeadliftRepFeatures

WEIGHT_SETUP = 0.25
WEIGHT_COORDINATION = 0.25
WEIGHT_BAR_PATH = 0.20
WEIGHT_LOCKOUT = 0.15
WEIGHT_SYMMETRY = 0.15

# Each measure scores 1 up to its tolerance, then falls to 0 over its decay
# range. Tolerances sit under the faults' mild thresholds (PLAN.md §2.6);
# initial values, fitted together with the thresholds at J6.
BAR_MIDFOOT_TOLERANCE_CM = 2.0
BAR_MIDFOOT_DECAY_CM = 8.0
HIP_BAND_TOLERANCE_CM = 2.0
HIP_BAND_DECAY_CM = 10.0
SHOULDERS_BEHIND_TOLERANCE_CM = 1.0
SHOULDERS_BEHIND_DECAY_CM = 6.0
TRUNK_CHANGE_TOLERANCE_DEG = 5.0
TRUNK_CHANGE_DECAY_DEG = 20.0
BAR_DRIFT_TOLERANCE_CM = 2.0
BAR_DRIFT_DECAY_CM = 8.0
LOCKOUT_TOLERANCE_DEG = 4.0
LOCKOUT_DECAY_DEG = 16.0
HIP_SHIFT_TOLERANCE_RATIO = 0.03
HIP_SHIFT_DECAY_RATIO = 0.15
BAR_TILT_TOLERANCE_CM = 1.5
BAR_TILT_DECAY_CM = 6.0

# The hero measurements (PLAN.md §1); confidence scales with how many the set measured.
_CORE_MEASURES = (
    "bar_midfoot_setup_cm",
    "bar_drift_cm",
    "trunk_change_liftoff_knee_deg",
    "setup_hip_height_cm",
)
# Measured from the side: blanked before causes are scored when no camera sees it.
_SIDE_VIEW_FIELDS = (
    "setup_hip_height_cm",
    "setup_hip_band_low_cm",
    "setup_hip_band_high_cm",
    "shoulder_vs_bar_cm",
    "trunk_change_liftoff_knee_deg",
    "hip_extension_deficit_deg",
    "knee_extension_deficit_deg",
    "lean_back_deg",
)
# Fatigue shows at the end of a set, not in its middle.
_TAIL_FIELDS = ("velocity_loss_pct",)


def _penalty_score(value: float, tolerance: float, decay: float) -> float:
    if not math.isfinite(value):
        return math.nan
    return max(0.0, min(1.0, 1.0 - max(0.0, value - tolerance) / decay))


def _weakest(*scores: float) -> float:
    measured = [score for score in scores if math.isfinite(score)]
    return min(measured) if measured else math.nan


def _extract_feature(rep: DeadliftRepSummary, feature_name: str) -> float:
    if feature_name == "bar_midfoot_offset_cm":
        return abs(rep.bar_midfoot_setup_cm)
    if feature_name == "setup_hip_band_miss_cm":
        return band_miss_cm(rep)
    if feature_name == "shoulders_behind_bar_cm":
        if not math.isfinite(rep.shoulder_vs_bar_cm):
            return math.nan
        return max(0.0, -rep.shoulder_vs_bar_cm)
    if feature_name == "lockout_deficit_deg":
        deficits = [
            value
            for value in (rep.hip_extension_deficit_deg, rep.knee_extension_deficit_deg)
            if math.isfinite(value)
        ]
        return max(deficits) if deficits else math.nan
    if feature_name == "hip_shift_abs":
        return abs(rep.hip_shift_ratio)
    if feature_name == "bar_tilt_abs_cm":
        return abs(rep.bar_tilt_cm)
    return getattr(rep, feature_name)


def _fill_values(
    rep: DeadliftRepSummary,
    anthro: dict,
    rom: dict,
    set_summary: SetScoreSummary | None,
) -> dict[str, Any]:
    feet_cm = delta_feet_position(rep, anthro, rom)["feet_toward_bar_cm"]
    roll_cm = delta_setup_habit(rep, anthro, rom)["bar_toward_shins_cm"]
    if rep.hip_shift_ratio > 0.0:
        shift_side_phrase = "toward your right foot"
    elif rep.hip_shift_ratio < 0.0:
        shift_side_phrase = "toward your left foot"
    else:
        shift_side_phrase = "to one side"
    return {
        "feet_move_cm": abs(feet_cm),
        "feet_direction": "closer to the bar" if feet_cm >= 0.0 else "back from the bar",
        "bar_roll_cm": abs(roll_cm),
        "bar_roll_direction": "away from you" if roll_cm >= 0.0 else "in toward you",
        "hips_up_cm": delta_hips_up(rep, anthro, rom)["hips_up_cm"],
        "hips_down_cm": delta_hips_down(rep, anthro, rom)["hips_down_cm"],
        "shoulders_forward_cm": delta_shoulders_forward(rep, anthro, rom)["shoulders_forward_cm"],
        "bar_start_closer_cm": delta_bar_start_closer(rep, anthro, rom)["bar_start_closer_cm"],
        "bar_closer_cm": delta_bar_closer(rep, anthro, rom)["bar_closer_cm"],
        "bar_level_cm": delta_level_bar(rep, anthro, rom)["bar_level_cm"],
        "shift_side_phrase": shift_side_phrase,
        "reduction_pct": LOAD_REDUCTION_PCT,
    }


class DeadliftRepSummary(BaseModel):
    """The per-rep measures the deadlift diagnosis reads, all numeric so the
    engine can take the set's median rep. NaN = not measured."""
    rep_number: int
    touch_and_go: bool = False
    bar_midfoot_stance_cm: float = NAN
    bar_midfoot_setup_cm: float = NAN
    setup_hip_height_cm: float = NAN
    setup_hip_band_low_cm: float = NAN
    setup_hip_band_high_cm: float = NAN
    shoulder_vs_bar_cm: float = NAN
    trunk_change_liftoff_knee_deg: float = NAN
    bar_drift_cm: float = NAN
    hip_extension_deficit_deg: float = NAN
    knee_extension_deficit_deg: float = NAN
    lean_back_deg: float = NAN
    hip_shift_ratio: float = NAN
    bar_tilt_cm: float = NAN
    concentric_velocity_mps: float = NAN
    velocity_loss_pct: float = NAN

    @classmethod
    def from_features(cls, features: DeadliftRepFeatures | dict[str, Any]) -> DeadliftRepSummary:
        """From the analyser's features, or their dict as sent over IPC (null = not measured)."""
        if isinstance(features, dict):
            features = DeadliftRepFeatures.model_validate(
                {name: value for name, value in features.items() if value is not None}
            )
        return cls(**{name: getattr(features, name) for name in cls.model_fields})


def score_deadlift_rep(rep: DeadliftRepSummary) -> DeadliftRepScore:
    setup = _weakest(
        _penalty_score(abs(rep.bar_midfoot_setup_cm), BAR_MIDFOOT_TOLERANCE_CM, BAR_MIDFOOT_DECAY_CM),
        _penalty_score(band_miss_cm(rep), HIP_BAND_TOLERANCE_CM, HIP_BAND_DECAY_CM),
        _penalty_score(-rep.shoulder_vs_bar_cm, SHOULDERS_BEHIND_TOLERANCE_CM, SHOULDERS_BEHIND_DECAY_CM),
    )
    coordination = _penalty_score(
        rep.trunk_change_liftoff_knee_deg, TRUNK_CHANGE_TOLERANCE_DEG, TRUNK_CHANGE_DECAY_DEG
    )
    bar_path = _penalty_score(rep.bar_drift_cm, BAR_DRIFT_TOLERANCE_CM, BAR_DRIFT_DECAY_CM)
    lockout = _weakest(
        _penalty_score(rep.hip_extension_deficit_deg, LOCKOUT_TOLERANCE_DEG, LOCKOUT_DECAY_DEG),
        _penalty_score(rep.knee_extension_deficit_deg, LOCKOUT_TOLERANCE_DEG, LOCKOUT_DECAY_DEG),
        _penalty_score(rep.lean_back_deg, LOCKOUT_TOLERANCE_DEG, LOCKOUT_DECAY_DEG),
    )
    symmetry = _weakest(
        _penalty_score(abs(rep.hip_shift_ratio), HIP_SHIFT_TOLERANCE_RATIO, HIP_SHIFT_DECAY_RATIO),
        _penalty_score(abs(rep.bar_tilt_cm), BAR_TILT_TOLERANCE_CM, BAR_TILT_DECAY_CM),
    )

    # Unmeasured dimensions are left out of the composite and the rest
    # renormalised, never scored perfect (as rep_scoring.score_rep).
    weighted = [
        (score, weight)
        for score, weight in (
            (setup, WEIGHT_SETUP),
            (coordination, WEIGHT_COORDINATION),
            (bar_path, WEIGHT_BAR_PATH),
            (lockout, WEIGHT_LOCKOUT),
            (symmetry, WEIGHT_SYMMETRY),
        )
        if math.isfinite(score)
    ]
    total_weight = sum(weight for _, weight in weighted)
    composite = (
        sum(score * weight for score, weight in weighted) / total_weight
        if total_weight > 0.0
        else math.nan
    )

    return DeadliftRepScore(
        rep_number=rep.rep_number,
        setup_score=round(setup, 3),
        coordination_score=round(coordination, 3),
        bar_path_score=round(bar_path, 3),
        lockout_score=round(lockout, 3),
        symmetry_score=round(symmetry, 3),
        composite_score=round(composite, 3),
    )


def score_deadlift_set(reps: list[DeadliftRepSummary]) -> SetScoreSummary:
    return summarize_rep_scores([score_deadlift_rep(rep) for rep in reps])


_SYMPTOM_GRAPH, _CAUSE_GRAPH = load_graph(
    "deadlift_symptoms.yaml",
    "deadlift_causes.yaml",
    deadlift_evidence_tests,
    deadlift_parameter_deltas,
)

DEADLIFT_GRAPH = DiagnosisGraph(
    symptoms=_SYMPTOM_GRAPH,
    causes=_CAUSE_GRAPH,
    rep_model=DeadliftRepSummary,
    extract_feature=_extract_feature,
    fill_values=_fill_values,
    score_set=lambda reps, anthro, rom: score_deadlift_set(reps),
    core_measures=_CORE_MEASURES,
    side_view_fields=_SIDE_VIEW_FIELDS,
    tail_fields=_TAIL_FIELDS,
)


class DeadliftSetDiagnosis:
    """The deadlift's set diagnosis: the set so far after every counted rep,
    and the whole set when it ends. Results serialise through
    ipc_bridge.serialize_diagnosis like the squat's."""

    def __init__(self, capture_mode: str) -> None:
        self.capture_mode = capture_mode
        self._engine = HypothesisEngine(DEADLIFT_GRAPH)
        self._reps: list[DeadliftRepSummary] = []

    def on_rep(self, features: DeadliftRepFeatures | dict[str, Any]) -> DiagnosisResult | None:
        """Adds a counted rep (its features, or their dict as RepData carries them)
        and re-diagnoses the set so far. None while the set has measured none of
        the core measures (nothing to say yet)."""
        summary = DeadliftRepSummary.from_features(features)
        self._reps.append(summary)
        result = self._diagnose(f"rolling_rep_{summary.rep_number}")
        return result if result.confidence > 0.0 else None

    def finish_set(self, set_id: str) -> tuple[DiagnosisResult, SetScoreSummary | None] | None:
        """The whole set's diagnosis and score, then a fresh set. None for a set
        with no reps; the score is None when no rep measured any dimension."""
        if not self._reps:
            return None
        result = self._diagnose(set_id)
        score_summary = score_deadlift_set(self._reps)
        self._reps = []
        if not math.isfinite(score_summary.mean_score):
            return result, None
        return result, score_summary

    def reset(self) -> None:
        self._reps = []

    def _diagnose(self, set_id: str) -> DiagnosisResult:
        # The deadlift graph reads no anthropometry or range of motion.
        return self._engine.diagnose_reps(set_id, list(self._reps), {}, {}, self.capture_mode)
