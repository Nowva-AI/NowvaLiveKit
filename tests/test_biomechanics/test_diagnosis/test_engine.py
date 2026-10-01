"""Tests for the hypothesis engine: the set's median aggregate rep, evidence-weighted
cause scoring, balance-model trunk lean attribution, observability by capture
mode, measurement confidence, and explanation/correction consistency."""

from __future__ import annotations

import math
import re

import pytest

from biomechanics.diagnosis.engine import HypothesisEngine
from biomechanics.diagnosis.graph.evidence_tests import (
    ANKLE_DF_UNRESTRICTED_DEG,
    test_knee_track_cue as knee_track_cue_evidence,
)
from biomechanics.diagnosis.graph.loader import CAUSE_GRAPH, SYMPTOM_GRAPH
from biomechanics.diagnosis.graph.parameter_deltas import (
    dorsi_driven_targets,
    foot_angle_target_deg,
    magnitude_widen_stance,
    stance_target_ratio,
)
from biomechanics.diagnosis.lean_model import expected_pitches
from biomechanics.diagnosis.rep_scoring import score_set
from biomechanics.diagnosis.types import RepKinematicSummary, SetFeatures
from biomechanics.faults.observability import (
    APPROXIMATE,
    NOT_OBSERVABLE,
    OBSERVABLE,
    SINGLE_CAMERA,
    TRIANGULATED,
    measurement_observability,
)

ANTHRO = {
    "femur_torso_ratio": 0.42 / 0.45,
    "hip_width": 0.30,
    "shoulder_width": 0.40,
    "femur_length_avg": 0.42,
    "tibia_length_avg": 0.43,
    "torso_length": 0.45,
    "foot_length": 0.26,
}
# Proportions of the 188.5 cm reference user: femur/torso ~0.85, the
# population reference, so their own balanced lean is the reference lean.
REFERENCE_BUILD_ANTHRO = {
    "femur_torso_ratio": 0.462 / 0.543,
    "hip_width": 0.30,
    "shoulder_width": 0.40,
    "femur_length_avg": 0.462,
    "tibia_length_avg": 0.464,
    "torso_length": 0.543,
    "foot_length": 0.20,
}
LONG_FEMUR_ANTHRO = {
    "femur_torso_ratio": 0.53 / 0.48,
    "hip_width": 0.30,
    "shoulder_width": 0.40,
    "femur_length_avg": 0.53,
    "tibia_length_avg": 0.43,
    "torso_length": 0.48,
    "foot_length": 0.20,
}

ROM = {"peak_dorsiflexion": 35.0, "avg_depth": 120.0}

# A clean rep: just below parallel, ankles moving freely, natural toe-out,
# controlled tempo, trunk at the lean its balance needs.
CLEAN_DEPTH_RATIO = -0.05
UNRESTRICTED_ANKLE_DEG = 32.0
NATURAL_TOE_OUT_DEG = 20.0
KNEE_HEIGHT_CM = 45.0
SHALLOW_DEPTH_RATIO = 0.30

CONFIDENCE_TOLERANCE = 1e-3
EVIDENCE_TOLERANCE = 1e-6
DELTA_TOLERANCE_M = 1e-6

# Lean far enough past the balanced pitch to be a genuine fault.
FAULTY_EXTRA_LEAN_DEG = 20.0
VALGUS_FAULT_DEG = 12.0

_PLACEHOLDER = re.compile(r"[{}]")
_NAN_WORD = re.compile(r"\bnan\b", re.IGNORECASE)


def _make_rep(
    rep_number: int, anthro: dict = ANTHRO, **overrides
) -> RepKinematicSummary:
    """A rep summary as the bridge builds it: expected pitches from the
    balance model at this rep's depth and ankle tilt, unless overridden."""
    depth_ratio = overrides.get("depth_ratio", CLEAN_DEPTH_RATIO)
    ankle_l = overrides.get("ankle_df_l_max", UNRESTRICTED_ANKLE_DEG)
    ankle_r = overrides.get("ankle_df_r_max", UNRESTRICTED_ANKLE_DEG)
    reference, athlete, with_ankles = expected_pitches(
        anthro, depth_ratio, (ankle_l + ankle_r) / 2.0,
    )
    hip_height_cm = KNEE_HEIGHT_CM + depth_ratio * anthro["femur_length_avg"] * 100.0
    values: dict = dict(
        rep_number=rep_number,
        trunk_pitch_at_bottom=with_ankles,
        knee_valgus_l=0.0,
        knee_valgus_r=0.0,
        ankle_df_l_max=ankle_l,
        ankle_df_r_max=ankle_r,
        hip_y_l_at_bottom=hip_height_cm,
        hip_y_r_at_bottom=hip_height_cm,
        knee_y_l_at_bottom=KNEE_HEIGHT_CM,
        knee_y_r_at_bottom=KNEE_HEIGHT_CM,
        stance_width_ratio=dorsi_driven_targets(ROM["peak_dorsiflexion"], anthro)[0],
        foot_direction_angle_l=NATURAL_TOE_OUT_DEG,
        foot_direction_angle_r=NATURAL_TOE_OUT_DEG,
        depth_class_int=4,
        descent_time_s=2.0,
        ascent_time_s=1.0,
        depth_ratio=depth_ratio,
        expected_pitch_reference=reference,
        expected_pitch_athlete=athlete,
        expected_pitch_with_ankles=with_ankles,
        hip_shift_ratio=0.0,
    )
    values.update(overrides)
    return RepKinematicSummary(**values)


def _set_of(count: int, **overrides) -> list[RepKinematicSummary]:
    return [_make_rep(rep_number, **overrides) for rep_number in range(1, count + 1)]


def _lean_set(
    anthro: dict,
    extra_lean_deg: float,
    ankle_deg: float = ANKLE_DF_UNRESTRICTED_DEG,
    count: int = 3,
    **overrides,
) -> list[RepKinematicSummary]:
    """Reps leaning ``extra_lean_deg`` past the pitch the athlete's own build
    and ankles need to keep the shoulders over midfoot."""
    _, _, with_ankles = expected_pitches(anthro, CLEAN_DEPTH_RATIO, ankle_deg)
    return [
        _make_rep(
            rep_number,
            anthro=anthro,
            trunk_pitch_at_bottom=with_ankles + extra_lean_deg,
            ankle_df_l_max=ankle_deg,
            ankle_df_r_max=ankle_deg,
            **overrides,
        )
        for rep_number in range(1, count + 1)
    ]


def _fully_measured_rep(rep_number: int) -> RepKinematicSummary:
    """A rep with every optional measure filled, as the bridge builds it from
    whole-rep features."""
    return _make_rep(
        rep_number,
        knee_valgus_l=9.0,
        knee_valgus_r=6.0,
        ankle_df_l_max=24.0,
        ankle_df_r_max=31.0,
        hip_flexion_l_max=104.0,
        hip_flexion_r_max=117.0,
        hip_shift_ratio=0.09,
        hip_shoot_deg=12.0,
        balance_ratio=0.2,
        heel_rise_max_cm=2.5,
        velocity_loss_pct=32.0,
        initiation_ratio=1.8,
        neck_flexion_deg=-28.0,
        lockout_deficit_ratio=0.08,
        stagger_ratio=0.15,
        concentric_velocity_mps=0.45,
        stance_width_ratio=1.1,
        foot_direction_angle_l=12.0,
        foot_direction_angle_r=4.0,
        descent_time_s=0.4,
    )


def _diagnose(
    reps: list[RepKinematicSummary],
    rom: dict = ROM,
    anthro: dict = ANTHRO,
    capture_mode: str = SINGLE_CAMERA,
):
    set_features = SetFeatures(
        user_id=1,
        set_id="set-1",
        rep_count=len(reps),
        per_rep_kinematics=reps,
        anthropometry=anthro,
        rom=rom,
        capture_mode=capture_mode,
    )
    return HypothesisEngine().diagnose(set_features)


def _all_hypotheses(result) -> list:
    return (
        result.immediate_causes
        + result.session_causes
        + result.longterm_causes
        + result.contextual_notes
    )


def _all_cause_ids(result) -> list[str]:
    return [h.cause_id for h in _all_hypotheses(result)]


def _hypothesis(result, cause_id: str):
    return next(h for h in _all_hypotheses(result) if h.cause_id == cause_id)


def _symptom_ids(result) -> list[str]:
    return [s.symptom_id for s in result.detected_symptoms]


def _symptom(result, symptom_id: str):
    return next(s for s in result.detected_symptoms if s.symptom_id == symptom_id)


class TestAggregateRep:
    """Causes are judged on the set's median rep, labelled with the worst
    rep's number so explanations still point at a real rep — whatever the
    numbering convention (live path starts at 1, offline replay at 2, rolling
    windows at arbitrary offsets)."""

    def _aggregate(self, reps: list[RepKinematicSummary]) -> RepKinematicSummary:
        summary = score_set(reps, ANTHRO, ROM)
        return HypothesisEngine()._aggregate_rep(reps, summary)

    def test_live_numbering_from_one(self):
        # depth_ratio 0.9 → barely descended → worst depth score
        reps = [_make_rep(1), _make_rep(2, depth_ratio=0.9), _make_rep(3)]
        assert self._aggregate(reps).rep_number == 2

    def test_offline_numbering_from_two(self):
        reps = [_make_rep(2), _make_rep(3, depth_ratio=0.9), _make_rep(4)]
        assert self._aggregate(reps).rep_number == 3

    def test_rolling_window_numbering(self):
        reps = [_make_rep(4), _make_rep(5, depth_ratio=0.9), _make_rep(6)]
        assert self._aggregate(reps).rep_number == 5

    def test_single_rep_short_circuit(self):
        reps = [_make_rep(7)]
        assert HypothesisEngine()._aggregate_rep(reps, None) is reps[0]

    def test_numeric_fields_are_the_median_not_the_worst(self):
        reps = [
            _make_rep(1, knee_valgus_l=2.0),
            _make_rep(2, knee_valgus_l=4.0),
            _make_rep(3, knee_valgus_l=30.0),
        ]
        assert self._aggregate(reps).knee_valgus_l == pytest.approx(4.0)

    def test_fatigue_measure_takes_the_last_reps(self):
        # Velocity loss shows up at the end of a set, not in its middle — and
        # the last two reps, not one noisy rep, say how close it was.
        reps = [
            _make_rep(1, velocity_loss_pct=5.0),
            _make_rep(2, velocity_loss_pct=40.0),
            _make_rep(3, velocity_loss_pct=10.0),
            _make_rep(4, velocity_loss_pct=30.0),
        ]
        assert self._aggregate(reps).velocity_loss_pct == pytest.approx(20.0)

    def test_missing_values_are_ignored(self):
        reps = [
            _make_rep(1, hip_shift_ratio=math.nan),
            _make_rep(2, hip_shift_ratio=0.02),
            _make_rep(3, hip_shift_ratio=0.04),
        ]
        aggregate = self._aggregate(reps)
        assert aggregate.hip_shift_ratio == pytest.approx(0.03)
        assert math.isnan(aggregate.heel_rise_max_cm)

    def test_bar_detected_on_any_rep(self):
        reps = [_make_rep(1), _make_rep(2, bar_detected=True), _make_rep(3)]
        assert self._aggregate(reps).bar_detected is True


class TestEvidenceUsesMedianRep:
    """Regression: evidence tests used to read the single worst rep, so one
    outlier could implicate a cause the rest of the set never showed."""

    def _valgus_set_with_one_stiff_ankle_rep(self) -> list[RepKinematicSummary]:
        reps = [
            _make_rep(n, knee_valgus_l=VALGUS_FAULT_DEG, knee_valgus_r=VALGUS_FAULT_DEG)
            for n in (1, 2, 3, 4)
        ]
        # The outlier is also the worst-scoring rep, which the old engine picked.
        outlier = _make_rep(
            5, knee_valgus_l=20.0, knee_valgus_r=20.0,
            ankle_df_l_max=12.0, ankle_df_r_max=12.0,
        )
        return reps + [outlier]

    def test_outlier_rep_is_the_worst_rep(self):
        reps = self._valgus_set_with_one_stiff_ankle_rep()
        assert score_set(reps, ANTHRO, ROM).worst_rep_number == 5

    def test_single_stiff_ankle_rep_does_not_implicate_the_ankle(self):
        result = _diagnose(self._valgus_set_with_one_stiff_ankle_rep())
        assert "knee_not_tracking_toes" in _symptom_ids(result)
        cause_ids = _all_cause_ids(result)
        assert "limited_ankle_df" not in cause_ids
        assert "heel_elevation" not in cause_ids

    def test_cause_evidence_is_computed_on_the_median_rep(self):
        reps = self._valgus_set_with_one_stiff_ankle_rep()
        result = _diagnose(reps)
        median_rep = _make_rep(
            5, knee_valgus_l=VALGUS_FAULT_DEG, knee_valgus_r=VALGUS_FAULT_DEG,
        )
        expected = knee_track_cue_evidence(median_rep, ANTHRO, ROM, None)
        worst_rep_evidence = knee_track_cue_evidence(reps[-1], ANTHRO, ROM, None)
        evidence = _hypothesis(result, "knee_track_cue").evidence_score
        assert evidence == pytest.approx(expected, abs=EVIDENCE_TOLERANCE)
        assert evidence != pytest.approx(worst_rep_evidence, abs=0.05)


class TestEvidenceWeightedScoring:
    """A detected symptom must not distribute a full unit of probability mass
    among its candidate causes when all evidence is weak: cause scores are
    weighted by symptom severity and an unexplained-leak term absorbs mass."""

    def test_mild_symptom_with_weak_evidence_yields_no_hypotheses(self):
        # Hips slide 10% of the stance toward one side: barely past the
        # symptom threshold (severity ~0.13) with weak evidence for every
        # candidate cause. Must NOT be coached as a confident diagnosis.
        result = _diagnose(_set_of(3, hip_shift_ratio=0.10))

        assert "hip_shift" in _symptom_ids(result)
        assert _all_cause_ids(result) == []

    def test_severe_symptom_with_strong_evidence_is_diagnosed(self):
        # 16% hip shift with one hip clearly tighter: strong evidence for
        # both the in-session cue and the unilateral hip restriction.
        result = _diagnose(
            _set_of(
                3, hip_shift_ratio=0.16,
                hip_flexion_l_max=110.0, hip_flexion_r_max=125.0,
            )
        )

        cause_ids = _all_cause_ids(result)
        assert "weight_shift_cue" in cause_ids
        assert "unilateral_hip_mobility_limit" in cause_ids


class TestTrunkLeanAgainstBalanceModel:
    """Regression: expected lean was ~12° for real proportions, so every
    textbook squat read as excessive lean and was blamed on bracing. Lean is
    now judged against the pitch that keeps the shoulders over midfoot."""

    def test_lean_at_balanced_pitch_does_not_fire(self):
        result = _diagnose(_lean_set(REFERENCE_BUILD_ANTHRO, 0.0), anthro=REFERENCE_BUILD_ANTHRO)
        assert "excessive_trunk_lean" not in _symptom_ids(result)

    def test_lean_just_inside_threshold_does_not_fire(self):
        result = _diagnose(_lean_set(REFERENCE_BUILD_ANTHRO, 5.0), anthro=REFERENCE_BUILD_ANTHRO)
        assert "excessive_trunk_lean" not in _symptom_ids(result)

    def test_lean_well_beyond_balance_fires_and_blames_bracing(self):
        result = _diagnose(
            _lean_set(REFERENCE_BUILD_ANTHRO, FAULTY_EXTRA_LEAN_DEG),
            anthro=REFERENCE_BUILD_ANTHRO,
        )

        assert _symptom(result, "excessive_trunk_lean").severity == pytest.approx(1.0)
        assert "bracing_failure" in [h.cause_id for h in result.immediate_causes]
        assert "anthropometric_femur_torso_ratio" not in _all_cause_ids(result)

    def test_textbook_45_degree_rep_with_stiff_ankles_is_not_a_bracing_fault(self):
        # 20° of shin tilt needs ~43° of lean at parallel for this build; a
        # 45° rep leans past the reference lifter, but the ankles explain it.
        reps = [
            _make_rep(
                n, anthro=REFERENCE_BUILD_ANTHRO, trunk_pitch_at_bottom=45.0,
                ankle_df_l_max=20.0, ankle_df_r_max=20.0,
            )
            for n in (1, 2, 3)
        ]
        result = _diagnose(reps, anthro=REFERENCE_BUILD_ANTHRO)

        cause_ids = _all_cause_ids(result)
        assert "bracing_failure" not in cause_ids
        assert "limited_ankle_df" in cause_ids

    def test_long_femur_lean_matching_own_balance_is_anatomy(self):
        result = _diagnose(_lean_set(LONG_FEMUR_ANTHRO, 0.0), anthro=LONG_FEMUR_ANTHRO)

        # The lean exceeds the reference lifter's, so the symptom fires...
        assert "excessive_trunk_lean" in _symptom_ids(result)
        # ...and the build, not the brace, explains it.
        assert "bracing_failure" not in _all_cause_ids(result)
        anthropometric = _hypothesis(result, "anthropometric_femur_torso_ratio")
        assert anthropometric in result.contextual_notes
        lean_causes = [
            h for h in _all_hypotheses(result) if "excessive_trunk_lean" in h.implicated_by
        ]
        assert max(lean_causes, key=lambda h: h.score) is anthropometric

    def test_anatomy_note_quotes_the_athletes_balanced_lean(self):
        reps = _lean_set(LONG_FEMUR_ANTHRO, 0.0)
        result = _diagnose(reps, anthro=LONG_FEMUR_ANTHRO)
        athlete_lean = reps[0].expected_pitch_athlete
        explanation = _hypothesis(result, "anthropometric_femur_torso_ratio").explanation
        assert f"about {athlete_lean:.0f}° of forward lean" in explanation


class TestObservability:
    """A coach who cannot see something says nothing about it: symptoms the
    capture mode cannot measure are skipped, and side-view symptoms read from
    one frontal camera are flagged approximate."""

    def test_heel_rise_never_fires_on_a_single_camera(self):
        result = _diagnose(_set_of(3, heel_rise_max_cm=10.0), capture_mode=SINGLE_CAMERA)
        assert "heel_rise" not in _symptom_ids(result)

    def test_heel_rise_fires_when_triangulated(self):
        result = _diagnose(_set_of(3, heel_rise_max_cm=4.0), capture_mode=TRIANGULATED)
        heel_rise = _symptom(result, "heel_rise")
        assert heel_rise.severity > 0.0
        assert heel_rise.observability == OBSERVABLE

    def test_sagittal_symptom_is_approximate_on_a_single_camera(self):
        result = _diagnose(
            _lean_set(REFERENCE_BUILD_ANTHRO, FAULTY_EXTRA_LEAN_DEG),
            anthro=REFERENCE_BUILD_ANTHRO, capture_mode=SINGLE_CAMERA,
        )
        assert _symptom(result, "excessive_trunk_lean").observability == APPROXIMATE
        assert _hypothesis(result, "bracing_failure").observability == APPROXIMATE

    def test_sagittal_symptom_is_observable_when_triangulated(self):
        result = _diagnose(
            _lean_set(REFERENCE_BUILD_ANTHRO, FAULTY_EXTRA_LEAN_DEG),
            anthro=REFERENCE_BUILD_ANTHRO, capture_mode=TRIANGULATED,
        )
        assert _symptom(result, "excessive_trunk_lean").observability == OBSERVABLE
        assert _hypothesis(result, "bracing_failure").observability == OBSERVABLE

    def test_frontal_symptom_is_observable_on_a_single_camera(self):
        result = _diagnose(
            _set_of(3, knee_valgus_l=VALGUS_FAULT_DEG, knee_valgus_r=VALGUS_FAULT_DEG)
        )
        assert _symptom(result, "knee_not_tracking_toes").observability == OBSERVABLE
        assert _hypothesis(result, "knee_track_cue").observability == OBSERVABLE

    def test_every_symptom_is_observable_when_triangulated(self):
        for symptom_id, symptom_def in SYMPTOM_GRAPH.items():
            assert measurement_observability(
                symptom_def["measurement"], TRIANGULATED
            ) == OBSERVABLE, symptom_id

    def test_heel_rise_is_the_only_single_camera_blind_spot(self):
        blind = [
            symptom_id
            for symptom_id, symptom_def in SYMPTOM_GRAPH.items()
            if measurement_observability(symptom_def["measurement"], SINGLE_CAMERA)
            == NOT_OBSERVABLE
        ]
        assert blind == ["heel_rise"]


class TestMeasurementConfidence:
    """Regression: confidence used to be mean symptom severity. It now says
    how well the set was measured — rep count times observability."""

    def test_clean_three_rep_set_is_fully_confident(self):
        assert _diagnose(_set_of(3)).confidence == pytest.approx(1.0, abs=CONFIDENCE_TOLERANCE)

    def test_fewer_than_three_reps_scale_confidence_down(self):
        assert _diagnose(_set_of(2)).confidence == pytest.approx(2.0 / 3.0, abs=CONFIDENCE_TOLERANCE)
        assert _diagnose(_set_of(1)).confidence == pytest.approx(1.0 / 3.0, abs=CONFIDENCE_TOLERANCE)

    def test_observable_symptom_keeps_full_confidence(self):
        result = _diagnose(
            _set_of(3, knee_valgus_l=VALGUS_FAULT_DEG, knee_valgus_r=VALGUS_FAULT_DEG)
        )
        assert result.confidence == pytest.approx(1.0, abs=CONFIDENCE_TOLERANCE)

    def test_approximate_symptom_halves_confidence(self):
        result = _diagnose(
            _lean_set(REFERENCE_BUILD_ANTHRO, FAULTY_EXTRA_LEAN_DEG),
            anthro=REFERENCE_BUILD_ANTHRO,
        )
        assert _symptom_ids(result) == ["excessive_trunk_lean"]
        assert result.confidence == pytest.approx(0.5, abs=CONFIDENCE_TOLERANCE)

    def test_unmeasured_knees_lower_confidence(self):
        """A clean verdict on a set whose knees were never seen is not a confident one."""
        result = _diagnose(_set_of(3, knee_valgus_l=math.nan, knee_valgus_r=math.nan))
        assert result.detected_symptoms == []
        assert result.confidence == pytest.approx(0.75, abs=CONFIDENCE_TOLERANCE)

    def test_same_set_triangulated_is_fully_confident(self):
        result = _diagnose(
            _lean_set(REFERENCE_BUILD_ANTHRO, FAULTY_EXTRA_LEAN_DEG),
            anthro=REFERENCE_BUILD_ANTHRO, capture_mode=TRIANGULATED,
        )
        assert result.confidence == pytest.approx(1.0, abs=CONFIDENCE_TOLERANCE)

    def test_mixed_observability_averages_the_weights(self):
        reps = _lean_set(
            REFERENCE_BUILD_ANTHRO, FAULTY_EXTRA_LEAN_DEG,
            knee_valgus_l=VALGUS_FAULT_DEG, knee_valgus_r=VALGUS_FAULT_DEG,
        )
        result = _diagnose(reps, anthro=REFERENCE_BUILD_ANTHRO)
        assert set(_symptom_ids(result)) == {"excessive_trunk_lean", "knee_not_tracking_toes"}
        assert result.confidence == pytest.approx(0.75, abs=CONFIDENCE_TOLERANCE)

    def test_short_set_with_approximate_symptom_compounds(self):
        result = _diagnose(
            _lean_set(REFERENCE_BUILD_ANTHRO, FAULTY_EXTRA_LEAN_DEG, count=2),
            anthro=REFERENCE_BUILD_ANTHRO,
        )
        assert result.confidence == pytest.approx(1.0 / 3.0, abs=CONFIDENCE_TOLERANCE)

    def test_confidence_does_not_track_severity(self):
        mild = _diagnose(_set_of(3, knee_valgus_l=8.0, knee_valgus_r=8.0))
        severe = _diagnose(_set_of(3, knee_valgus_l=25.0, knee_valgus_r=25.0))
        mild_severity = _symptom(mild, "knee_not_tracking_toes").severity
        severe_severity = _symptom(severe, "knee_not_tracking_toes").severity
        assert mild_severity < severe_severity
        assert mild.confidence == pytest.approx(severe.confidence, abs=CONFIDENCE_TOLERANCE)


class TestExplanationMatchesCorrection:
    """The spoken explanation, the spoken magnitude and the geometric
    parameter delta must all derive from the same personalized target."""

    STRAIGHT_TOES_DEG = 0.0
    NARROW_STANCE_RATIO = 1.2

    def _shallow_narrow_feet_reps(self) -> list[RepKinematicSummary]:
        # Shallow depth with a narrow stance and toes pointing straight ahead:
        # depth_limit implicates both narrow_stance and narrow_foot_angle.
        return _set_of(
            3,
            depth_ratio=SHALLOW_DEPTH_RATIO,
            stance_width_ratio=self.NARROW_STANCE_RATIO,
            foot_direction_angle_l=self.STRAIGHT_TOES_DEG,
            foot_direction_angle_r=self.STRAIGHT_TOES_DEG,
            depth_class_int=2,
        )

    def test_foot_angle_explanation_matches_parameter_delta(self):
        result = _diagnose(self._shallow_narrow_feet_reps())
        hypothesis = next(
            h for h in result.immediate_causes if h.cause_id == "narrow_foot_angle"
        )

        target_angle = foot_angle_target_deg(ANTHRO, ROM)
        assert f"try about {target_angle:.0f}°" in hypothesis.explanation
        assert f"point out about {self.STRAIGHT_TOES_DEG:.0f}°" in hypothesis.explanation

        delta_degrees = math.degrees(abs(hypothesis.parameter_delta["L_ankle.ry"]))
        assert delta_degrees == pytest.approx(
            target_angle - self.STRAIGHT_TOES_DEG, abs=0.01
        )

    def test_stance_explanation_matches_parameter_delta(self):
        result = _diagnose(self._shallow_narrow_feet_reps())
        hypothesis = next(
            h for h in result.immediate_causes if h.cause_id == "narrow_stance"
        )

        target_ratio = stance_target_ratio(self.NARROW_STANCE_RATIO, ANTHRO, ROM)
        expected_per_side_m = (
            (target_ratio - self.NARROW_STANCE_RATIO) * ANTHRO["shoulder_width"] / 2.0
        )
        delta = hypothesis.parameter_delta
        assert delta["__width_increase_per_side_m"] == pytest.approx(
            expected_per_side_m, abs=DELTA_TOLERANCE_M
        )
        assert delta["__foot_target_delta"][5] == pytest.approx(
            expected_per_side_m, abs=DELTA_TOLERANCE_M
        )

        per_side_cm = delta["__width_increase_per_side_m"] * 100.0
        assert f"each foot about {per_side_cm:.0f} cm wider" in hypothesis.explanation
        assert f"about {self.NARROW_STANCE_RATIO:.1f} times shoulder width" in hypothesis.explanation
        # The spoken magnitude names the same distance as the explanation.
        assert f"each foot about {per_side_cm:.0f} centimeters wider" == magnitude_widen_stance(delta)


class TestNarrowStanceSurfacesUnderDepthLimit:
    """When depth_limit fires from hip-vs-knee height, a narrow stance must
    surface as an immediate cause even if the depth class and trunk pitch look
    fine (regression from session 2026-07-22_11-39-49, where the toe cue fired
    six rounds running while the stance cue was structurally silenced)."""

    def _depth_limited_rep(
        self, rep_number: int, stance_width_ratio: float
    ) -> RepKinematicSummary:
        # Hip ~10 cm above the knee fires depth_limit; the contradictory
        # depth_class_int=4 must not gate the stance evidence.
        return _make_rep(
            rep_number,
            depth_ratio=0.25,
            stance_width_ratio=stance_width_ratio,
            foot_direction_angle_l=27.5,
            foot_direction_angle_r=27.5,
            depth_class_int=4,
        )

    def test_narrow_stance_is_diagnosed(self):
        reps = [self._depth_limited_rep(n, 0.77) for n in (1, 2, 3)]
        result = _diagnose(reps)
        assert "depth_limit" in _symptom_ids(result)
        assert "narrow_stance" in [h.cause_id for h in result.immediate_causes]

    def test_wide_stance_is_not_diagnosed(self):
        target_ratio, _ = dorsi_driven_targets(ROM["peak_dorsiflexion"], ANTHRO)
        reps = [
            self._depth_limited_rep(n, target_ratio + 0.1) for n in (1, 2, 3)
        ]
        result = _diagnose(reps)
        assert "narrow_stance" not in [
            h.cause_id for h in result.immediate_causes
        ]


class TestFootAngleCompetesOnEvidence:
    """The toe-out cue used to be force-surfaced past the hypothesis threshold
    as "a safe, universally beneficial cue", on top of a target floored at 30
    deg. Natural toe-out is 5-15 deg, so it fired for nearly everyone and
    prescribed 30-40 deg of external rotation that no measured hip ROM
    supported. It now competes on evidence like every other cause."""

    def _immediate_ids(self, result) -> list[str]:
        return [h.cause_id for h in result.immediate_causes]

    def _shallow_feet_at(self, angle: float) -> list[RepKinematicSummary]:
        # Shallow depth (fires depth_limit), otherwise clean form.
        return _set_of(
            3,
            depth_ratio=SHALLOW_DEPTH_RATIO,
            foot_direction_angle_l=angle,
            foot_direction_angle_r=angle,
            depth_class_int=2,
        )

    def test_target_is_the_personalized_value_not_a_floor(self):
        target = foot_angle_target_deg(ANTHRO, ROM)
        _, personalized = dorsi_driven_targets(ROM["peak_dorsiflexion"], ANTHRO)
        assert target == pytest.approx(personalized)

    def test_target_can_fall_below_thirty_degrees(self):
        """A max(30.0, ...) floor discarded the bottom half of the 15-40 range."""
        assert foot_angle_target_deg(ANTHRO, ROM) < 30.0

    def test_barely_narrow_feet_are_not_force_cued(self):
        """Just under target gives near-zero evidence, so the cause must not
        clear the hypothesis threshold on its own."""
        target = foot_angle_target_deg(ANTHRO, ROM)
        result = _diagnose(self._shallow_feet_at(target - 1.0))
        assert "narrow_foot_angle" not in self._immediate_ids(result)

    def test_clearly_narrow_feet_still_surface(self):
        result = _diagnose(self._shallow_feet_at(2.0))
        assert "narrow_foot_angle" in self._immediate_ids(result)

    def test_adequately_turned_out_feet_are_not_cued(self):
        target = foot_angle_target_deg(ANTHRO, ROM)
        result = _diagnose(self._shallow_feet_at(target + 3.0))
        assert "narrow_foot_angle" not in self._immediate_ids(result)

    def test_narrow_feet_without_supporting_evidence_are_not_cued(self):
        # Deep rep with knee valgus only: no depth or lean symptom implicates
        # the foot angle, so it stays silent.
        reps = _set_of(
            3,
            knee_valgus_l=25.0,
            knee_valgus_r=25.0,
            foot_direction_angle_l=29.0,
            foot_direction_angle_r=29.0,
        )
        result = _diagnose(reps)
        assert "narrow_foot_angle" not in self._immediate_ids(result)


class TestExplanationTemplates:
    """Every cause template is spoken to the athlete, so every placeholder
    must be filled by engine._fill_template — a leftover brace or NaN is read
    out loud."""

    @pytest.mark.parametrize("cause_id", sorted(CAUSE_GRAPH))
    def test_template_fills_every_placeholder(self, cause_id: str):
        reps = [_fully_measured_rep(n) for n in (1, 2, 3)]
        summary = score_set(reps, ANTHRO, ROM)
        aggregate = HypothesisEngine()._aggregate_rep(reps, summary)
        text = HypothesisEngine()._fill_template(
            CAUSE_GRAPH[cause_id]["explanation_template"], aggregate, ANTHRO, ROM, summary,
        )
        assert not _PLACEHOLDER.search(text), text
        assert not _NAN_WORD.search(text), text

    @pytest.mark.parametrize("cause_id", sorted(CAUSE_GRAPH))
    def test_template_fills_for_a_single_rep_without_features(self, cause_id: str):
        # One rep, no whole-rep features: optional measures are NaN and there
        # is no set summary.
        rep = _make_rep(1)
        text = HypothesisEngine()._fill_template(
            CAUSE_GRAPH[cause_id]["explanation_template"], rep, ANTHRO, ROM, None,
        )
        assert not _PLACEHOLDER.search(text), text
        assert not _NAN_WORD.search(text), text

    def test_ankle_template_states_the_unrestricted_range_not_the_athletes_own(self):
        # Regression: expected_df was filled with the athlete's own peak,
        # producing "your ankles bend 25°; you need 25°".
        rep = _make_rep(1, ankle_df_l_max=22.0, ankle_df_r_max=22.0)
        rom_with_low_peak = {"peak_dorsiflexion": 22.0, "avg_depth": 120.0}
        text = HypothesisEngine()._fill_template(
            CAUSE_GRAPH["limited_ankle_df"]["explanation_template"],
            rep, ANTHRO, rom_with_low_peak, None,
        )
        assert "about 22°" in text
        assert f"about {ANKLE_DF_UNRESTRICTED_DEG:.0f}°" in text


class TestGraphContent:
    """Coaching content a professional would catch immediately (audit B6)."""

    def test_no_template_prescribes_copenhagen_planks(self):
        # Copenhagen planks train the adductors, not the abductors.
        for cause_id, cause_def in CAUSE_GRAPH.items():
            assert "copenhagen" not in cause_def["explanation_template"].lower(), cause_id

    def test_no_template_prescribes_hip_flexor_work(self):
        # Tight hip flexors limit hip extension, not the flexion a squat needs.
        for cause_id, cause_def in CAUSE_GRAPH.items():
            assert "hip flexor work" not in cause_def["explanation_template"].lower(), cause_id

    def test_unobservable_arch_collapse_cause_is_gone(self):
        assert "foot_collapse_arch" not in CAUSE_GRAPH

    def test_candidate_priors_sum_to_one_per_symptom(self):
        for symptom_id, symptom_def in SYMPTOM_GRAPH.items():
            total = sum(candidate["prior"] for candidate in symptom_def["candidate_causes"])
            assert total == pytest.approx(1.0, abs=1e-9), symptom_id
