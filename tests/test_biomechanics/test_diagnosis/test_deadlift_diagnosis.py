"""Tests for the deadlift set diagnosis: the deadlift graph on the parameterised
HypothesisEngine (symptoms, cause attribution, one-number deltas and one-sentence
explanations), deadlift rep/set scoring, observability by capture mode, the
contract shape of its results, and the squat graph staying the engine's default."""

from __future__ import annotations

import json
import math
import re
import textwrap

import pytest

from biomechanics.coaching.ipc_bridge import IPCBridge, per_dimension_means, serialize_diagnosis
from biomechanics.deadlift.diagnosis import (
    DEADLIFT_GRAPH,
    WEIGHT_BAR_PATH,
    WEIGHT_COORDINATION,
    WEIGHT_LOCKOUT,
    WEIGHT_SETUP,
    WEIGHT_SYMMETRY,
    DeadliftRepSummary,
    DeadliftSetDiagnosis,
    score_deadlift_rep,
    score_deadlift_set,
)
from biomechanics.deadlift.types import DeadliftRepFeatures
from biomechanics.diagnosis.engine import SQUAT_GRAPH, HypothesisEngine
from biomechanics.diagnosis.graph import deadlift_evidence_tests, deadlift_parameter_deltas, loader
from biomechanics.diagnosis.graph.loader import CAUSE_GRAPH, SYMPTOM_GRAPH, load_graph
from biomechanics.diagnosis.types import (
    DeadliftRepScore,
    DiagnosisResult,
    RepKinematicSummary,
    RepScore,
    SetFeatures,
    SetScoreSummary,
)
from biomechanics.faults import observability
from biomechanics.faults.observability import (
    APPROXIMATE,
    NOT_OBSERVABLE,
    OBSERVABLE,
    SINGLE_CAMERA,
    TRIANGULATED,
    measurement_observability,
)
from biomechanics.utils.json_safe import nan_to_none

SCORE_TOLERANCE = 1e-3
CONFIDENCE_TOLERANCE = 1e-3

# A clean rep: bar over the midfoot, hips mid-band with the shoulders just in
# front of the bar, chest and hips rising together, bar kept close, a full
# lockout, even side to side, and a fast pull with no speed lost.
CLEAN_REP: dict = dict(
    bar_midfoot_stance_cm=0.5,
    bar_midfoot_setup_cm=0.5,
    setup_hip_height_cm=50.0,
    setup_hip_band_low_cm=46.0,
    setup_hip_band_high_cm=54.0,
    shoulder_vs_bar_cm=3.0,
    trunk_change_liftoff_knee_deg=1.0,
    bar_drift_cm=1.0,
    hip_extension_deficit_deg=1.0,
    knee_extension_deficit_deg=1.0,
    lean_back_deg=1.0,
    hip_shift_ratio=0.01,
    bar_tilt_cm=0.5,
    concentric_velocity_mps=0.6,
    velocity_loss_pct=5.0,
)
# Every measure past its fault, so every explanation has a real value to quote.
FAULTY_REP: dict = dict(
    bar_midfoot_stance_cm=4.0,
    bar_midfoot_setup_cm=6.0,
    setup_hip_height_cm=38.0,
    setup_hip_band_low_cm=46.0,
    setup_hip_band_high_cm=54.0,
    shoulder_vs_bar_cm=-3.0,
    trunk_change_liftoff_knee_deg=15.0,
    bar_drift_cm=6.0,
    hip_extension_deficit_deg=12.0,
    knee_extension_deficit_deg=9.0,
    lean_back_deg=10.0,
    hip_shift_ratio=0.15,
    bar_tilt_cm=5.0,
    concentric_velocity_mps=0.3,
    velocity_loss_pct=30.0,
)

# One synthetic set per symptom: per-rep overrides of the clean rep.
SYMPTOM_SETS: dict[str, list[dict]] = {
    "bar_off_midfoot": [dict(bar_midfoot_stance_cm=6.0, bar_midfoot_setup_cm=6.0)] * 3,
    # Hips that start too low leave the shoulders at the bar, not in front of it.
    "setup_hips_off": [dict(setup_hip_height_cm=38.0, shoulder_vs_bar_cm=-1.0)] * 3,
    "shoulders_behind_bar": [dict(shoulder_vs_bar_cm=-5.0)] * 3,
    "hips_shoot": [dict(trunk_change_liftoff_knee_deg=20.0)] * 3,
    "bar_drift": [dict(bar_drift_cm=7.0)] * 3,
    "incomplete_lockout": [dict(hip_extension_deficit_deg=16.0)] * 3,
    "lean_back": [dict(lean_back_deg=16.0)] * 3,
    "hip_shift": [dict(hip_shift_ratio=0.20)] * 3,
    "bar_tilt": [dict(bar_tilt_cm=7.0)] * 3,
    "velocity_loss": [dict(velocity_loss_pct=loss) for loss in (0.0, 10.0, 40.0, 50.0)],
}
# The cause each synthetic set is about: the top hypothesis its symptom implicates.
EXPECTED_CAUSE: dict[str, str] = {
    "bar_off_midfoot": "feet_position",
    "setup_hips_off": "hips_too_low",
    "shoulders_behind_bar": "shoulder_position",
    "hips_shoot": "slack_not_pulled",
    "bar_drift": "lats_not_engaged",
    "incomplete_lockout": "glutes_not_finishing",
    "lean_back": "over_extension_habit",
    "hip_shift": "uneven_stance",
    "bar_tilt": "uneven_grip",
    "velocity_loss": "load_too_heavy",
}
SIDE_VIEW_SYMPTOMS = {
    "setup_hips_off", "shoulders_behind_bar", "hips_shoot", "incomplete_lockout", "lean_back",
}
BAR_SYMPTOMS = {"bar_off_midfoot", "bar_drift", "bar_tilt", "velocity_loss"}
DIAGNOSIS_KEYS = {
    "confidence", "detected_symptoms", "immediate_causes", "session_causes",
    "longterm_causes", "contextual_notes", "combined_perturbation",
}
CAUSE_KEYS = {
    "cause_id", "tier", "score", "explanation", "parameter_delta", "implicated_by", "observability",
}
DEADLIFT_DIMENSIONS = {"setup", "coordination", "bar_path", "lockout", "symmetry"}

_PLACEHOLDER = re.compile(r"[{}]")
_NAN_WORD = re.compile(r"\bnan\b", re.IGNORECASE)
_CM_PLACEHOLDER = re.compile(r"\{\w+_cm:")
_SENTENCE_END = re.compile(r"[.!?](\s|$)")
# A small local model reads these out: plain words, nothing medical, and never
# a claim about back rounding, which no keypoint measures.
_BANNED_WORDS = re.compile(
    r"\b(round(ed|ing)?|spine|spinal|lumbar|thoracic|lats?|injur\w*|pain\w*|disc)\b",
    re.IGNORECASE,
)


def _features(rep_number: int, base: dict = CLEAN_REP, **overrides) -> DeadliftRepFeatures:
    return DeadliftRepFeatures(rep_number=rep_number, **{**base, **overrides})


def _set_from(per_rep_overrides: list[dict], base: dict = CLEAN_REP) -> list[DeadliftRepFeatures]:
    return [
        _features(rep_number, base, **overrides)
        for rep_number, overrides in enumerate(per_rep_overrides, start=1)
    ]


def _clean_set(count: int = 3) -> list[DeadliftRepFeatures]:
    return _set_from([{}] * count)


def _finish(
    features: list[DeadliftRepFeatures], capture_mode: str = TRIANGULATED
) -> tuple[DiagnosisResult, SetScoreSummary | None]:
    diagnosis = DeadliftSetDiagnosis(capture_mode)
    for rep_features in features:
        diagnosis.on_rep(rep_features)
    return diagnosis.finish_set("set-1")


def _diagnose(
    features: list[DeadliftRepFeatures], capture_mode: str = TRIANGULATED
) -> DiagnosisResult:
    return _finish(features, capture_mode)[0]


def _all_hypotheses(result: DiagnosisResult) -> list:
    return (
        result.immediate_causes
        + result.session_causes
        + result.longterm_causes
        + result.contextual_notes
    )


def _cause_ids(result: DiagnosisResult) -> list[str]:
    return [hypothesis.cause_id for hypothesis in _all_hypotheses(result)]


def _hypothesis(result: DiagnosisResult, cause_id: str):
    return next(h for h in _all_hypotheses(result) if h.cause_id == cause_id)


def _symptom_ids(result: DiagnosisResult) -> list[str]:
    return [symptom.symptom_id for symptom in result.detected_symptoms]


def _summaries(features: list[DeadliftRepFeatures]) -> list[DeadliftRepSummary]:
    return [DeadliftRepSummary.from_features(rep_features) for rep_features in features]


def _aggregate(features: list[DeadliftRepFeatures]) -> tuple[DeadliftRepSummary, SetScoreSummary]:
    summaries = _summaries(features)
    set_summary = score_deadlift_set(summaries)
    return HypothesisEngine(DEADLIFT_GRAPH)._aggregate_rep(summaries, set_summary), set_summary


def _tier_one_cause_ids() -> list[str]:
    return sorted(
        cause_id for cause_id, cause_def in DEADLIFT_GRAPH.causes.items() if cause_def["tier"] == 1
    )


def _single_delta(delta: dict) -> float:
    assert len(delta) == 1, delta
    return next(iter(delta.values()))


def _register_bar_3d(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        observability._MEASUREMENT_LEVELS,
        "bar_3d",
        {SINGLE_CAMERA: APPROXIMATE, TRIANGULATED: OBSERVABLE},
    )


def _squat_rep(rep_number: int, valgus_deg: float) -> RepKinematicSummary:
    return RepKinematicSummary(
        rep_number=rep_number,
        trunk_pitch_at_bottom=30.0,
        knee_valgus_l=valgus_deg,
        knee_valgus_r=valgus_deg,
        ankle_df_l_max=32.0,
        ankle_df_r_max=32.0,
        hip_y_l_at_bottom=43.0,
        hip_y_r_at_bottom=43.0,
        knee_y_l_at_bottom=45.0,
        knee_y_r_at_bottom=45.0,
        stance_width_ratio=1.3,
        foot_direction_angle_l=20.0,
        foot_direction_angle_r=20.0,
        depth_class_int=4,
        depth_ratio=-0.05,
        hip_shift_ratio=0.0,
    )


class _RecordingClient:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    def send_message(self, message: dict) -> None:
        self.messages.append(message)


class TestSymptoms:
    """Each symptom fires on a set showing it, alone, and a clean set is quiet."""

    def test_every_symptom_has_a_synthetic_set(self):
        assert set(SYMPTOM_SETS) == set(DEADLIFT_GRAPH.symptoms)
        assert set(EXPECTED_CAUSE) == set(DEADLIFT_GRAPH.symptoms)

    @pytest.mark.parametrize("symptom_id", sorted(SYMPTOM_SETS))
    def test_symptom_fires_alone_on_its_set(self, symptom_id: str):
        result = _diagnose(_set_from(SYMPTOM_SETS[symptom_id]))
        assert _symptom_ids(result) == [symptom_id]

    @pytest.mark.parametrize("symptom_id", sorted(SYMPTOM_SETS))
    def test_symptom_implicates_its_expected_cause_first(self, symptom_id: str):
        result = _diagnose(_set_from(SYMPTOM_SETS[symptom_id]))
        implicated = [h for h in _all_hypotheses(result) if symptom_id in h.implicated_by]
        assert implicated, symptom_id
        assert max(implicated, key=lambda h: h.score).cause_id == EXPECTED_CAUSE[symptom_id]

    def test_clean_set_is_quiet(self):
        result = _diagnose(_clean_set())
        assert result.detected_symptoms == []
        assert _all_hypotheses(result) == []
        assert result.combined_perturbation == {}
        assert result.confidence == pytest.approx(1.0, abs=CONFIDENCE_TOLERANCE)

    def test_fault_just_under_its_threshold_stays_quiet(self):
        result = _diagnose(_set_from([dict(bar_midfoot_stance_cm=2.9, bar_midfoot_setup_cm=2.9)] * 3))
        assert result.detected_symptoms == []


class TestCauseAttribution:
    """Evidence picks between a symptom's candidate causes."""

    def test_bar_off_midfoot_from_the_stance_is_the_feet(self):
        result = _diagnose(_set_from(SYMPTOM_SETS["bar_off_midfoot"]))
        assert "feet_position" in [h.cause_id for h in result.immediate_causes]
        assert "setup_habit" not in _cause_ids(result)

    def test_bar_moved_while_getting_set_is_the_habit(self):
        result = _diagnose(_set_from([dict(bar_midfoot_stance_cm=0.5, bar_midfoot_setup_cm=6.5)] * 3))
        assert "setup_habit" in [h.cause_id for h in result.immediate_causes]
        assert "feet_position" not in _cause_ids(result)

    def test_hips_outside_the_band_with_everything_else_right_points_at_the_build(self):
        # Bar over the midfoot and shoulders in their band: the setup model
        # fixes the hip height, so the band, not the lifter, is in doubt.
        result = _diagnose(_set_from([dict(setup_hip_height_cm=38.0)] * 3))
        assert "setup_anthropometry" in [h.cause_id for h in result.contextual_notes]
        behind = _diagnose(_set_from(SYMPTOM_SETS["setup_hips_off"]))
        assert "setup_anthropometry" not in _cause_ids(behind)
        assert (
            _hypothesis(result, "hips_too_low").score < _hypothesis(behind, "hips_too_low").score
        )

    def test_hips_set_high_are_lowered(self):
        result = _diagnose(_set_from([dict(setup_hip_height_cm=62.0, shoulder_vs_bar_cm=9.0)] * 3))
        assert "hips_too_high" in [h.cause_id for h in result.immediate_causes]
        assert "hips_too_low" not in _cause_ids(result)

    def test_hips_shoot_from_a_low_start_is_the_setup(self):
        result = _diagnose(
            _set_from([dict(trunk_change_liftoff_knee_deg=20.0, **SYMPTOM_SETS["setup_hips_off"][0])] * 3)
        )
        hips_low = _hypothesis(result, "hips_too_low")
        assert "hips_shoot" in hips_low.implicated_by
        assert "slack_not_pulled" not in _cause_ids(result)

    def test_touch_and_go_strengthens_the_slack_cause(self):
        dead_stop = _diagnose(_set_from(SYMPTOM_SETS["hips_shoot"]))
        touch_and_go = _diagnose(
            _set_from([dict(trunk_change_liftoff_knee_deg=20.0, touch_and_go=True)] * 3)
        )
        assert (
            _hypothesis(touch_and_go, "slack_not_pulled").score
            > _hypothesis(dead_stop, "slack_not_pulled").score
        )

    def test_slow_pull_with_hips_shooting_points_at_leg_strength(self):
        slow = _diagnose(
            _set_from([dict(trunk_change_liftoff_knee_deg=20.0, concentric_velocity_mps=0.25)] * 3)
        )
        assert "weak_off_floor" in [h.cause_id for h in slow.longterm_causes]
        fast = _diagnose(_set_from(SYMPTOM_SETS["hips_shoot"]))
        assert "weak_off_floor" not in _cause_ids(fast)

    def test_drift_from_a_bar_set_too_far_is_the_setup(self):
        result = _diagnose(_set_from([dict(bar_drift_cm=7.0, bar_midfoot_setup_cm=5.0)] * 3))
        drift_causes = [h for h in _all_hypotheses(result) if "bar_drift" in h.implicated_by]
        assert max(drift_causes, key=lambda h: h.score).cause_id == "bar_far_at_setup"

    def test_lean_back_with_hips_short_also_points_at_the_glutes(self):
        result = _diagnose(
            _set_from([dict(lean_back_deg=16.0, hip_extension_deficit_deg=10.0)] * 3)
        )
        assert "lean_back" in _hypothesis(result, "glutes_not_finishing").implicated_by
        hips_through = _diagnose(_set_from(SYMPTOM_SETS["lean_back"]))
        assert "glutes_not_finishing" not in _cause_ids(hips_through)

    def test_shift_that_grows_over_the_set_is_one_side_tiring(self):
        growing = _diagnose(_set_from([dict(hip_shift_ratio=ratio) for ratio in (0.10, 0.16, 0.22, 0.28)]))
        assert "unilateral_weakness" in [h.cause_id for h in growing.longterm_causes]
        steady = _diagnose(_set_from(SYMPTOM_SETS["hip_shift"]))
        assert "unilateral_weakness" not in _cause_ids(steady)

    def test_speed_lost_over_the_set_is_load_advice_not_a_cue(self):
        result = _diagnose(_set_from(SYMPTOM_SETS["velocity_loss"]))
        assert "load_too_heavy" in [h.cause_id for h in result.session_causes]
        assert result.immediate_causes == []


class TestCauseOutputs:
    """A small local model relays the outputs: each immediate cause carries one
    number and one plain sentence, and the sentence quotes that same number."""

    def test_only_tier_one_causes_have_a_delta_function(self):
        for cause_id, cause_def in DEADLIFT_GRAPH.causes.items():
            assert (cause_def["tier"] == 1) == (cause_def["parameter_delta_fn"] is not None), cause_id

    @pytest.mark.parametrize("cause_id", _tier_one_cause_ids())
    def test_tier_one_cause_has_exactly_one_numeric_delta(self, cause_id: str):
        aggregate, _ = _aggregate(_set_from([{}] * 3, base=FAULTY_REP))
        delta = DEADLIFT_GRAPH.causes[cause_id]["parameter_delta_fn"](aggregate, {}, {})
        value = _single_delta(delta)
        assert isinstance(value, float)
        assert math.isfinite(value)

    @pytest.mark.parametrize("cause_id", _tier_one_cause_ids())
    def test_tier_one_delta_is_a_number_for_an_unmeasured_rep(self, cause_id: str):
        delta = DEADLIFT_GRAPH.causes[cause_id]["parameter_delta_fn"](
            DeadliftRepSummary(rep_number=1), {}, {}
        )
        assert math.isfinite(_single_delta(delta))

    def test_tier_one_delta_keys_are_distinct(self):
        # The engine sums deltas sharing a key across causes; no two
        # corrections may be added together.
        aggregate, _ = _aggregate(_set_from([{}] * 3, base=FAULTY_REP))
        keys = [
            key
            for cause_id in _tier_one_cause_ids()
            for key in DEADLIFT_GRAPH.causes[cause_id]["parameter_delta_fn"](aggregate, {}, {})
        ]
        assert len(keys) == len(set(keys))

    @pytest.mark.parametrize("cause_id", sorted(DEADLIFT_GRAPH.causes))
    def test_template_fills_every_placeholder(self, cause_id: str):
        aggregate, set_summary = _aggregate(_set_from([{}] * 3, base=FAULTY_REP))
        text = HypothesisEngine(DEADLIFT_GRAPH)._fill_template(
            DEADLIFT_GRAPH.causes[cause_id]["explanation_template"], aggregate, {}, {}, set_summary,
        )
        assert not _PLACEHOLDER.search(text), text
        assert not _NAN_WORD.search(text), text

    @pytest.mark.parametrize("cause_id", sorted(DEADLIFT_GRAPH.causes))
    def test_template_fills_for_a_single_unmeasured_rep(self, cause_id: str):
        text = HypothesisEngine(DEADLIFT_GRAPH)._fill_template(
            DEADLIFT_GRAPH.causes[cause_id]["explanation_template"],
            DeadliftRepSummary(rep_number=1), {}, {}, None,
        )
        assert not _PLACEHOLDER.search(text), text
        assert not _NAN_WORD.search(text), text

    @pytest.mark.parametrize("cause_id", sorted(DEADLIFT_GRAPH.causes))
    def test_explanation_is_one_plain_sentence(self, cause_id: str):
        template = DEADLIFT_GRAPH.causes[cause_id]["explanation_template"]
        assert len(_SENTENCE_END.findall(template)) == 1, template
        assert template.endswith("."), template
        assert not _BANNED_WORDS.search(template), template

    @pytest.mark.parametrize("symptom_id", sorted(SYMPTOM_SETS))
    def test_explanation_quotes_the_delta(self, symptom_id: str):
        result = _diagnose(_set_from(SYMPTOM_SETS[symptom_id]))
        for hypothesis in result.immediate_causes:
            value = _single_delta(hypothesis.parameter_delta)
            template = DEADLIFT_GRAPH.causes[hypothesis.cause_id]["explanation_template"]
            if _CM_PLACEHOLDER.search(template):
                assert f"about {abs(value):.0f} cm" in hypothesis.explanation, hypothesis.explanation

    def test_feet_cue_names_the_direction(self):
        closer = _diagnose(_set_from(SYMPTOM_SETS["bar_off_midfoot"]))
        back = _diagnose(_set_from([dict(bar_midfoot_stance_cm=-6.0, bar_midfoot_setup_cm=-6.0)] * 3))
        assert "6 cm closer to the bar" in _hypothesis(closer, "feet_position").explanation
        assert "6 cm back from the bar" in _hypothesis(back, "feet_position").explanation
        assert _hypothesis(back, "feet_position").parameter_delta == pytest.approx(
            {"feet_toward_bar_cm": -6.0}, abs=SCORE_TOLERANCE
        )

    def test_combined_perturbation_holds_the_immediate_deltas(self):
        result = _diagnose(_set_from(SYMPTOM_SETS["bar_drift"]))
        assert result.combined_perturbation == pytest.approx(
            {"bar_closer_cm": 7.0}, abs=SCORE_TOLERANCE
        )


class TestDeadliftScoring:
    """Setup 25 %, coordination 25 %, bar path 20 %, lockout 15 %, symmetry 15 %;
    unmeasured dimensions are dropped and the rest renormalised."""

    def test_weights_sum_to_one(self):
        total = WEIGHT_SETUP + WEIGHT_COORDINATION + WEIGHT_BAR_PATH + WEIGHT_LOCKOUT + WEIGHT_SYMMETRY
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_clean_rep_scores_full_marks(self):
        score = score_deadlift_rep(DeadliftRepSummary(rep_number=1, **CLEAN_REP))
        for dimension in DEADLIFT_DIMENSIONS:
            assert getattr(score, f"{dimension}_score") == pytest.approx(1.0, abs=SCORE_TOLERANCE)
        assert score.composite_score == pytest.approx(1.0, abs=SCORE_TOLERANCE)

    def test_fault_lowers_only_its_dimension(self):
        score = score_deadlift_rep(DeadliftRepSummary(rep_number=1, **{**CLEAN_REP, "bar_drift_cm": 6.0}))
        assert score.bar_path_score == pytest.approx(0.5, abs=SCORE_TOLERANCE)
        assert score.setup_score == pytest.approx(1.0, abs=SCORE_TOLERANCE)
        assert score.composite_score == pytest.approx(1.0 - WEIGHT_BAR_PATH * 0.5, abs=SCORE_TOLERANCE)

    def test_unmeasured_dimensions_are_dropped_and_the_rest_renormalised(self):
        score = score_deadlift_rep(
            DeadliftRepSummary(rep_number=1, bar_midfoot_setup_cm=6.0, bar_drift_cm=4.0)
        )
        assert score.setup_score == pytest.approx(0.5, abs=SCORE_TOLERANCE)
        assert score.bar_path_score == pytest.approx(0.75, abs=SCORE_TOLERANCE)
        for unmeasured in (score.coordination_score, score.lockout_score, score.symmetry_score):
            assert math.isnan(unmeasured)
        expected = (WEIGHT_SETUP * 0.5 + WEIGHT_BAR_PATH * 0.75) / (WEIGHT_SETUP + WEIGHT_BAR_PATH)
        assert score.composite_score == pytest.approx(expected, abs=SCORE_TOLERANCE)

    def test_one_measured_dimension_is_the_composite(self):
        score = score_deadlift_rep(DeadliftRepSummary(rep_number=1, lean_back_deg=12.0))
        assert score.composite_score == pytest.approx(score.lockout_score, abs=SCORE_TOLERANCE)

    def test_rep_with_nothing_measured_has_no_composite(self):
        assert math.isnan(score_deadlift_rep(DeadliftRepSummary(rep_number=1)).composite_score)

    def test_set_summary_finds_best_and_worst_reps(self):
        summaries = _summaries(_set_from([{}, dict(bar_drift_cm=8.0), {}]))
        summary = score_deadlift_set(summaries)
        assert summary.worst_rep_number == 2
        assert summary.best_rep_number == 1
        assert all(isinstance(score, DeadliftRepScore) for score in summary.per_rep_scores)

    def test_per_dimension_means_report_the_deadlift_dimensions(self):
        summaries = [
            DeadliftRepSummary(rep_number=1, bar_midfoot_setup_cm=6.0, bar_drift_cm=4.0),
            DeadliftRepSummary(rep_number=2, bar_midfoot_setup_cm=2.0, bar_drift_cm=2.0),
        ]
        means = per_dimension_means(score_deadlift_set(summaries).per_rep_scores)
        assert means == {
            "setup": pytest.approx(0.75, abs=SCORE_TOLERANCE),
            "bar_path": pytest.approx(0.875, abs=SCORE_TOLERANCE),
        }

    def test_per_dimension_means_keep_the_squat_dimensions(self):
        squat_score = RepScore(
            rep_number=1, depth_score=0.5, trunk_control_score=0.6, knee_tracking_score=0.7,
            symmetry_score=0.8, tempo_score=0.9, composite_score=0.7,
        )
        assert set(per_dimension_means([squat_score])) == {
            "depth", "trunk_control", "knee_tracking", "symmetry", "tempo",
        }


class TestObservability:
    """Side-view symptoms are blanked on a single camera; bar symptoms carry
    the bar measurement's observability (approximate on one camera once
    bar_3d is registered)."""

    @pytest.mark.parametrize("symptom_id", sorted(SIDE_VIEW_SYMPTOMS))
    def test_side_view_symptom_never_fires_on_a_single_camera(self, symptom_id: str):
        result = _diagnose(_set_from(SYMPTOM_SETS[symptom_id]), capture_mode=SINGLE_CAMERA)
        assert result.detected_symptoms == []
        assert _all_hypotheses(result) == []

    def test_single_camera_blind_spots_are_the_side_view_symptoms(self):
        blind = {
            symptom_id
            for symptom_id, symptom_def in DEADLIFT_GRAPH.symptoms.items()
            if measurement_observability(symptom_def["measurement"], SINGLE_CAMERA) == NOT_OBSERVABLE
        }
        assert blind == SIDE_VIEW_SYMPTOMS

    def test_every_symptom_is_observable_when_triangulated(self):
        for symptom_id, symptom_def in DEADLIFT_GRAPH.symptoms.items():
            observability_level = measurement_observability(symptom_def["measurement"], TRIANGULATED)
            assert observability_level == OBSERVABLE, symptom_id

    @pytest.mark.parametrize("symptom_id", sorted(BAR_SYMPTOMS))
    def test_bar_symptom_carries_the_bar_measurement_observability(self, symptom_id: str):
        assert DEADLIFT_GRAPH.symptoms[symptom_id]["measurement"] == "bar_3d"
        result = _diagnose(_set_from(SYMPTOM_SETS[symptom_id]), capture_mode=SINGLE_CAMERA)
        assert result.detected_symptoms[0].observability == measurement_observability(
            "bar_3d", SINGLE_CAMERA
        )

    def test_registered_bar_3d_is_approximate_on_a_single_camera(self, monkeypatch: pytest.MonkeyPatch):
        _register_bar_3d(monkeypatch)
        single = _diagnose(_set_from(SYMPTOM_SETS["bar_off_midfoot"]), capture_mode=SINGLE_CAMERA)
        assert single.detected_symptoms[0].observability == APPROXIMATE
        assert _hypothesis(single, "feet_position").observability == APPROXIMATE
        assert single.confidence == pytest.approx(0.5, abs=CONFIDENCE_TOLERANCE)
        rig = _diagnose(_set_from(SYMPTOM_SETS["bar_off_midfoot"]), capture_mode=TRIANGULATED)
        assert rig.detected_symptoms[0].observability == OBSERVABLE
        assert rig.confidence == pytest.approx(1.0, abs=CONFIDENCE_TOLERANCE)

    def test_side_view_fields_are_blanked_before_causes_on_a_single_camera(self):
        engine = HypothesisEngine(DEADLIFT_GRAPH)
        rep = DeadliftRepSummary(rep_number=1, **FAULTY_REP)
        blanked = engine._without_unseen_side_view(rep, SINGLE_CAMERA)
        side_view_names = (
            "setup_hip_height_cm", "shoulder_vs_bar_cm", "trunk_change_liftoff_knee_deg", "lean_back_deg",
        )
        for name in side_view_names:
            assert math.isnan(getattr(blanked, name)), name
        assert blanked.bar_midfoot_setup_cm == pytest.approx(FAULTY_REP["bar_midfoot_setup_cm"])
        assert blanked.bar_drift_cm == pytest.approx(FAULTY_REP["bar_drift_cm"])
        assert engine._without_unseen_side_view(rep, TRIANGULATED) is rep


class TestDeadliftSetDiagnosis:
    """A rolling diagnosis after each rep and a final one at set end, in the
    squat's contract shape."""

    def test_on_rep_returns_a_rolling_update_after_each_rep(self):
        diagnosis = DeadliftSetDiagnosis(TRIANGULATED)
        confidences = []
        for rep_features in _clean_set():
            result = diagnosis.on_rep(rep_features)
            assert result.set_id == f"rolling_rep_{rep_features.rep_number}"
            confidences.append(result.confidence)
        assert confidences == pytest.approx([1.0 / 3.0, 2.0 / 3.0, 1.0], abs=CONFIDENCE_TOLERANCE)

    def test_first_rep_already_names_the_fix(self):
        diagnosis = DeadliftSetDiagnosis(TRIANGULATED)
        result = diagnosis.on_rep(_features(1, bar_midfoot_stance_cm=6.0, bar_midfoot_setup_cm=6.0))
        assert [h.cause_id for h in result.immediate_causes] == ["feet_position"]

    def test_on_rep_returns_none_while_nothing_is_measured(self):
        diagnosis = DeadliftSetDiagnosis(TRIANGULATED)
        assert diagnosis.on_rep(DeadliftRepFeatures(rep_number=1)) is None
        assert diagnosis.on_rep(_features(2)) is not None

    def test_finish_set_without_reps_returns_none(self):
        assert DeadliftSetDiagnosis(TRIANGULATED).finish_set("set-1") is None

    def test_finish_set_scores_the_set_and_starts_a_new_one(self):
        diagnosis = DeadliftSetDiagnosis(TRIANGULATED)
        for rep_features in _set_from(SYMPTOM_SETS["bar_drift"]):
            diagnosis.on_rep(rep_features)
        result, score_summary = diagnosis.finish_set("set-7")
        assert result.set_id == "set-7"
        assert "bar_drift" in _symptom_ids(result)
        assert len(score_summary.per_rep_scores) == 3
        assert diagnosis.finish_set("set-8") is None

    def test_finish_set_scores_a_single_heavy_rep(self):
        result, score_summary = _finish(_clean_set(count=1))
        assert score_summary.mean_score == pytest.approx(1.0, abs=SCORE_TOLERANCE)

    def test_set_with_nothing_measured_has_no_score(self):
        diagnosis = DeadliftSetDiagnosis(TRIANGULATED)
        diagnosis.on_rep(DeadliftRepFeatures(rep_number=1))
        result, score_summary = diagnosis.finish_set("set-1")
        assert result.confidence == pytest.approx(0.0, abs=CONFIDENCE_TOLERANCE)
        assert score_summary is None

    def test_reset_drops_the_current_set(self):
        diagnosis = DeadliftSetDiagnosis(TRIANGULATED)
        diagnosis.on_rep(_features(1))
        diagnosis.reset()
        assert diagnosis.finish_set("set-1") is None

    def test_summary_reads_the_ipc_features_dict(self):
        features = _features(3, bar_drift_cm=4.0, touch_and_go=True)
        from_model = DeadliftRepSummary.from_features(features)
        from_ipc = DeadliftRepSummary.from_features(nan_to_none(features.model_dump()))
        assert nan_to_none(from_ipc.model_dump()) == nan_to_none(from_model.model_dump())
        unmeasured = DeadliftRepSummary.from_features({"rep_number": 4, "bar_drift_cm": None})
        assert math.isnan(unmeasured.bar_drift_cm)

    def test_result_serialises_in_the_contract_shape(self):
        result, _ = _finish(_set_from(SYMPTOM_SETS["bar_off_midfoot"]))
        payload = serialize_diagnosis(result)
        assert set(payload) == DIAGNOSIS_KEYS
        cause = payload["immediate_causes"][0]
        assert set(cause) == CAUSE_KEYS
        assert cause["parameter_delta"] == pytest.approx(
            {"feet_toward_bar_cm": 6.0}, abs=SCORE_TOLERANCE
        )
        json.dumps(nan_to_none(payload))

    def test_diagnosis_complete_carries_the_deadlift_dimensions(self):
        client = _RecordingClient()
        result, score_summary = _finish(_set_from(SYMPTOM_SETS["bar_drift"]))
        IPCBridge(client).send_diagnosis_complete(1, result, score_summary)
        message = client.messages[-1]
        assert message["type"] == "diagnosis_complete"
        assert set(message["scoring"]["per_dimension"]) == DEADLIFT_DIMENSIONS
        assert "bar_path_score" in message["scoring"]["per_rep_scores"][0]
        json.dumps(message)


class TestGraphs:
    """The deadlift graph loads through the same loader and checks as the
    squat's, and the squat graph stays the engine's default."""

    def test_deadlift_priors_sum_to_one_per_symptom(self):
        for symptom_id, symptom_def in DEADLIFT_GRAPH.symptoms.items():
            total = sum(candidate["prior"] for candidate in symptom_def["candidate_causes"])
            assert total == pytest.approx(1.0, abs=1e-9), symptom_id

    def test_deadlift_symptoms_use_the_deadlift_measurement_classes(self):
        for symptom_id, symptom_def in DEADLIFT_GRAPH.symptoms.items():
            assert symptom_def["measurement"] in {"bar_3d", "side_view", "lateral_travel"}, symptom_id

    def test_load_graph_rejects_a_cause_missing_from_its_causes_file(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ):
        (tmp_path / "symptoms.yaml").write_text(textwrap.dedent("""
            some_symptom:
              description: "A symptom whose cause is missing"
              detection: { feature: bar_drift_cm, aggregation: median }
              expected_value_fn: expected_zero
              severity_scoring: relative_excess
              threshold: 1.0
              measurement: bar_3d
              candidate_causes:
                - { cause_id: missing_cause, prior: 1.0 }
        """))
        (tmp_path / "other_causes.yaml").write_text("{}\n")
        monkeypatch.setattr(loader, "_GRAPH_DIR", tmp_path)
        with pytest.raises(ValueError, match="missing_cause.*other_causes.yaml"):
            load_graph(
                "symptoms.yaml", "other_causes.yaml",
                deadlift_evidence_tests, deadlift_parameter_deltas,
            )

    def test_squat_graph_is_the_loaded_squat_yaml(self):
        assert SQUAT_GRAPH.symptoms is SYMPTOM_GRAPH
        assert SQUAT_GRAPH.causes is CAUSE_GRAPH
        assert not set(DEADLIFT_GRAPH.causes) & {"weight_too_heavy", "lockout_cue"}

    def test_default_engine_diagnoses_squats_as_the_squat_graph(self):
        set_features = SetFeatures(
            user_id=1,
            set_id="squat-set",
            rep_count=3,
            per_rep_kinematics=[_squat_rep(rep_number, 12.0) for rep_number in (1, 2, 3)],
            anthropometry={"femur_torso_ratio": 0.93, "shoulder_width": 0.40},
            rom={"peak_dorsiflexion": 35.0},
        )
        default = HypothesisEngine().diagnose(set_features)
        explicit = HypothesisEngine(SQUAT_GRAPH).diagnose(set_features)
        assert "knee_not_tracking_toes" in _symptom_ids(default)
        assert default.model_dump() == explicit.model_dump()
