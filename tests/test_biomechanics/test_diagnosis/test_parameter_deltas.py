"""Tests that geometric corrections reach the shared anthropometry-driven targets."""

from __future__ import annotations

import math

import pytest

from biomechanics.diagnosis.graph.parameter_deltas import (
    delta_brace_trunk,
    delta_center_weight,
    delta_widen_foot_angle,
    delta_widen_stance,
    foot_angle_target_deg,
    magnitude_widen_stance,
    stance_target_ratio,
)
from biomechanics.diagnosis.types import RepKinematicSummary

ANTHRO = {"femur_torso_ratio": 0.93, "shoulder_width": 0.40, "hip_width": 0.30}
ROM = {"peak_dorsiflexion": 35.0, "avg_depth": 120.0}

DELTA_TOLERANCE = 1e-6
# Fraction of the unexplained lean the brace correction removes, and its cap.
BRACE_CORRECTION_FRACTION = 0.4
MAX_BRACE_CORRECTION_DEG = 8.0
MAX_PELVIS_SHIFT_M = 0.04


def _make_rep(**overrides) -> RepKinematicSummary:
    values: dict = dict(
        rep_number=1,
        trunk_pitch_at_bottom=21.6,
        knee_valgus_l=0.0,
        knee_valgus_r=0.0,
        ankle_df_l_max=25.0,
        ankle_df_r_max=25.0,
        hip_y_l_at_bottom=40.0,
        hip_y_r_at_bottom=40.0,
        knee_y_l_at_bottom=44.0,
        knee_y_r_at_bottom=44.0,
        stance_width_ratio=1.2,
        foot_direction_angle_l=5.0,
        foot_direction_angle_r=5.0,
        depth_class_int=2,
    )
    values.update(overrides)
    return RepKinematicSummary(**values)


class TestSharedTargets:
    def test_widen_stance_delta_reaches_shared_target(self):
        rep = _make_rep()
        delta = delta_widen_stance(rep, ANTHRO, ROM)

        target_ratio = stance_target_ratio(rep.stance_width_ratio, ANTHRO, ROM)
        expected_per_side = (
            (target_ratio - rep.stance_width_ratio) * ANTHRO["shoulder_width"] / 2.0
        )
        foot_target = delta["__foot_target_delta"]
        assert foot_target[5] == pytest.approx(expected_per_side, abs=DELTA_TOLERANCE)
        assert foot_target[2] == pytest.approx(-expected_per_side, abs=DELTA_TOLERANCE)

    def test_widen_foot_angle_delta_reaches_shared_target(self):
        rep = _make_rep()
        delta = delta_widen_foot_angle(rep, ANTHRO, ROM)

        target_angle = foot_angle_target_deg(ANTHRO, ROM)
        avg_current = 5.0
        delta_degrees = math.degrees(delta["L_ankle.ry"])
        assert delta_degrees == pytest.approx(
            target_angle - avg_current, abs=DELTA_TOLERANCE
        )

    def test_stance_target_is_independent_of_current_stance(self):
        """The target used to be max(dorsi_target, current + 0.15), which
        recommended widening by >=0.15 shoulder-widths regardless of
        measurement — including for athletes already wider than their own
        target, where the evidence test that justified the cue returns zero."""
        narrow = stance_target_ratio(0.9, ANTHRO, ROM)
        wide = stance_target_ratio(2.4, ANTHRO, ROM)
        assert narrow == pytest.approx(wide)

    def test_already_wide_stance_gets_no_widening(self):
        rep = _make_rep(stance_width_ratio=2.4)
        delta = delta_widen_stance(rep, ANTHRO, ROM)
        assert delta["__foot_target_delta"] == [0.0] * 6

    def test_stance_target_is_capped(self):
        assert stance_target_ratio(0.9, ANTHRO, ROM) <= 2.5


class TestWidenStanceSpeaksPerSide:
    """Athletes move each foot a distance; they do not think in stance ratios."""

    def test_per_side_width_matches_the_foot_target(self):
        delta = delta_widen_stance(_make_rep(), ANTHRO, ROM)
        assert delta["__width_increase_per_side_m"] == pytest.approx(
            delta["__foot_target_delta"][5], abs=DELTA_TOLERANCE
        )

    def test_magnitude_names_centimeters_per_foot(self):
        assert magnitude_widen_stance({"__width_increase_per_side_m": 0.04}) == (
            "each foot about 4 centimeters wider"
        )

    def test_magnitude_singular_centimeter(self):
        assert magnitude_widen_stance({"__width_increase_per_side_m": 0.01}) == (
            "each foot about 1 centimeter wider"
        )

    def test_no_widening_has_no_phrase(self):
        assert magnitude_widen_stance({"__width_increase_per_side_m": 0.0}) is None
        assert magnitude_widen_stance({}) is None


class TestBraceTrunk:
    """Only lean the athlete's own build and ankles do not explain is
    correctable by bracing (diagnosis.lean_model)."""

    def test_corrects_a_fraction_of_the_unexplained_lean(self):
        rep = _make_rep(trunk_pitch_at_bottom=50.0, expected_pitch_with_ankles=40.0)
        delta = delta_brace_trunk(rep, ANTHRO, ROM)
        assert delta["trunk.rx"] == pytest.approx(
            -math.radians(10.0 * BRACE_CORRECTION_FRACTION), abs=DELTA_TOLERANCE
        )

    def test_falls_back_to_the_free_ankle_expectation(self):
        rep = _make_rep(trunk_pitch_at_bottom=50.0, expected_pitch_athlete=40.0)
        delta = delta_brace_trunk(rep, ANTHRO, ROM)
        assert delta["trunk.rx"] == pytest.approx(
            -math.radians(10.0 * BRACE_CORRECTION_FRACTION), abs=DELTA_TOLERANCE
        )

    def test_balanced_lean_needs_no_correction(self):
        # Regression: a ~12° geometric expectation made every textbook rep
        # look like it needed bracing.
        rep = _make_rep(trunk_pitch_at_bottom=45.0, expected_pitch_with_ankles=45.0)
        assert delta_brace_trunk(rep, ANTHRO, ROM)["trunk.rx"] == pytest.approx(0.0)

    def test_upright_rep_is_never_pushed_further_upright(self):
        rep = _make_rep(trunk_pitch_at_bottom=30.0, expected_pitch_with_ankles=45.0)
        assert delta_brace_trunk(rep, ANTHRO, ROM)["trunk.rx"] == pytest.approx(0.0)

    def test_correction_is_capped(self):
        rep = _make_rep(trunk_pitch_at_bottom=80.0, expected_pitch_with_ankles=30.0)
        delta = delta_brace_trunk(rep, ANTHRO, ROM)
        assert delta["trunk.rx"] == pytest.approx(
            -math.radians(MAX_BRACE_CORRECTION_DEG), abs=DELTA_TOLERANCE
        )

    def test_unknown_expectation_gives_no_correction(self):
        rep = _make_rep(trunk_pitch_at_bottom=60.0)
        assert delta_brace_trunk(rep, ANTHRO, ROM)["trunk.rx"] == pytest.approx(0.0)


class TestCenterWeight:
    """The pelvis is moved back toward the midline by the measured hip shift."""

    def test_undoes_the_measured_shift(self):
        # Ankles 1.0 x biacromial (0.40 / 0.8 = 0.5 m) apart; a 4% shift is 2 cm.
        rep = _make_rep(hip_shift_ratio=0.04, stance_width_ratio=1.0)
        delta = delta_center_weight(rep, ANTHRO, ROM)
        assert delta["pelvis.tx"] == pytest.approx(-0.02, abs=DELTA_TOLERANCE)

    def test_opposite_shift_moves_the_other_way(self):
        rep = _make_rep(hip_shift_ratio=-0.04, stance_width_ratio=1.0)
        delta = delta_center_weight(rep, ANTHRO, ROM)
        assert delta["pelvis.tx"] == pytest.approx(0.02, abs=DELTA_TOLERANCE)

    def test_shift_is_capped(self):
        rep = _make_rep(hip_shift_ratio=0.5, stance_width_ratio=1.5)
        delta = delta_center_weight(rep, ANTHRO, ROM)
        assert delta["pelvis.tx"] == pytest.approx(-MAX_PELVIS_SHIFT_M, abs=DELTA_TOLERANCE)

    def test_unmeasured_shift_moves_nothing(self):
        delta = delta_center_weight(_make_rep(), ANTHRO, ROM)
        assert delta["pelvis.tx"] == pytest.approx(0.0)
