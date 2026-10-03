"""Tests for set-level evidence tests in the diagnosis graph."""

from __future__ import annotations

import pytest

from biomechanics.diagnosis.graph.evidence_tests import (
    ankle_df_limitation,
    test_bracing_failure as bracing_failure_evidence,
    test_femur_torso_ratio as femur_torso_ratio_evidence,
    test_limited_ankle_df as limited_ankle_df_evidence,
    test_narrow_stance as narrow_stance_evidence,
    test_progressive_degradation as progressive_degradation_evidence,
)
from biomechanics.diagnosis.graph.parameter_deltas import dorsi_driven_targets
from biomechanics.diagnosis.lean_model import UNRESTRICTED_SHANK_DEG, expected_pitches
from biomechanics.diagnosis.types import (
    RepKinematicSummary,
    RepScore,
    SetScoreSummary,
)

ANTHRO = {"femur_torso_ratio": 0.93}
ROM = {"peak_dorsiflexion": 35.0, "avg_depth": 120.0}

EVIDENCE_TOLERANCE = 1e-6

LONG_FEMUR_ANTHRO = {
    "torso_length": 0.48,
    "femur_length_avg": 0.53,
    "tibia_length_avg": 0.43,
    "foot_length": 0.20,
}
REFERENCE_BUILD_ANTHRO = {
    "torso_length": 0.543,
    "femur_length_avg": 0.462,
    "tibia_length_avg": 0.464,
    "foot_length": 0.20,
}
PARALLEL_RATIO = 0.0


def _make_rep(rep_number: int = 1, **overrides) -> RepKinematicSummary:
    values: dict = dict(
        rep_number=rep_number,
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
        foot_direction_angle_l=20.0,
        foot_direction_angle_r=20.0,
        depth_class_int=4,
    )
    values.update(overrides)
    return RepKinematicSummary(**values)


def _lean_rep(anthro: dict, shank_deg: float, extra_lean_deg: float, **overrides) -> RepKinematicSummary:
    """A rep leaning ``extra_lean_deg`` past the balanced pitch its build and ankles need."""
    reference, athlete, with_ankles = expected_pitches(anthro, PARALLEL_RATIO, shank_deg)
    return _make_rep(
        trunk_pitch_at_bottom=with_ankles + extra_lean_deg,
        ankle_df_l_max=shank_deg,
        ankle_df_r_max=shank_deg,
        depth_ratio=PARALLEL_RATIO,
        expected_pitch_reference=reference,
        expected_pitch_athlete=athlete,
        expected_pitch_with_ankles=with_ankles,
        **overrides,
    )


def _make_summary(
    composite_scores: list[float], trend_slope: float
) -> SetScoreSummary:
    per_rep_scores = [
        RepScore(
            rep_number=index + 1,
            depth_score=score,
            trunk_control_score=score,
            knee_tracking_score=score,
            symmetry_score=score,
            tempo_score=score,
            composite_score=score,
        )
        for index, score in enumerate(composite_scores)
    ]
    best = max(per_rep_scores, key=lambda s: s.composite_score)
    worst = min(per_rep_scores, key=lambda s: s.composite_score)
    return SetScoreSummary(
        mean_score=sum(composite_scores) / len(composite_scores),
        best_rep_number=best.rep_number,
        worst_rep_number=worst.rep_number,
        trend_slope=trend_slope,
        per_rep_scores=per_rep_scores,
    )


class TestNarrowStance:
    """narrow_stance evidence must measure the stance-width gap against the
    anthropometry-driven target alone. Whether the stance is causing a
    problem is established by the implicating symptom, not re-derived here
    from depth class or trunk pitch — those can contradict the symptom's own
    depth measure and silently zero the evidence."""

    def _ideal_ratio(self, rom: dict = ROM) -> float:
        ideal, _ = dorsi_driven_targets(rom["peak_dorsiflexion"], ANTHRO)
        return ideal

    def test_stance_at_target_gives_zero(self):
        rep = _make_rep(stance_width_ratio=self._ideal_ratio())
        evidence = narrow_stance_evidence(rep, ANTHRO, ROM, None)
        assert evidence == pytest.approx(0.0, abs=EVIDENCE_TOLERANCE)

    def test_stance_above_target_gives_zero(self):
        rep = _make_rep(stance_width_ratio=self._ideal_ratio() + 0.2)
        evidence = narrow_stance_evidence(rep, ANTHRO, ROM, None)
        assert evidence == pytest.approx(0.0, abs=EVIDENCE_TOLERANCE)

    def test_slightly_narrow_gives_partial_evidence(self):
        rep = _make_rep(stance_width_ratio=self._ideal_ratio() - 0.25)
        evidence = narrow_stance_evidence(rep, ANTHRO, ROM, None)
        assert evidence == pytest.approx(0.5, abs=EVIDENCE_TOLERANCE)

    def test_very_narrow_saturates_at_one(self):
        rep = _make_rep(stance_width_ratio=self._ideal_ratio() - 0.6)
        evidence = narrow_stance_evidence(rep, ANTHRO, ROM, None)
        assert evidence == pytest.approx(1.0, abs=EVIDENCE_TOLERANCE)

    def test_deep_depth_class_does_not_gate_evidence(self):
        # Regression (session 2026-07-22_11-39-49): knee-flexion depth class
        # said "below parallel" while the depth_limit symptom fired from
        # hip-vs-knee height, so a 0.77 stance ratio produced zero evidence
        # and narrow_stance could never surface.
        rep = _make_rep(
            stance_width_ratio=0.77,
            depth_class_int=4,
            trunk_pitch_at_bottom=8.1,
        )
        evidence = narrow_stance_evidence(rep, ANTHRO, ROM, None)
        assert evidence == pytest.approx(1.0, abs=EVIDENCE_TOLERANCE)


class TestProgressiveDegradation:
    """weight_too_heavy evidence must come from the score trend across the
    set, not from a single rep's kinematics."""

    def test_steep_decline_gives_full_evidence(self):
        summary = _make_summary([0.9, 0.8, 0.7, 0.6, 0.5], trend_slope=-0.1)
        evidence = progressive_degradation_evidence(_make_rep(), ANTHRO, ROM, summary)
        assert evidence == pytest.approx(1.0, abs=EVIDENCE_TOLERANCE)

    def test_gentle_decline_gives_partial_evidence(self):
        summary = _make_summary([0.9, 0.87, 0.84], trend_slope=-0.03)
        evidence = progressive_degradation_evidence(_make_rep(), ANTHRO, ROM, summary)
        assert evidence == pytest.approx(0.5, abs=EVIDENCE_TOLERANCE)

    def test_flat_set_gives_zero(self):
        summary = _make_summary([0.8, 0.8, 0.8, 0.8], trend_slope=0.0)
        evidence = progressive_degradation_evidence(_make_rep(), ANTHRO, ROM, summary)
        assert evidence == pytest.approx(0.0, abs=EVIDENCE_TOLERANCE)

    def test_improving_set_gives_zero(self):
        summary = _make_summary([0.6, 0.7, 0.8], trend_slope=0.1)
        evidence = progressive_degradation_evidence(_make_rep(), ANTHRO, ROM, summary)
        assert evidence == pytest.approx(0.0, abs=EVIDENCE_TOLERANCE)

    def test_missing_summary_gives_zero(self):
        evidence = progressive_degradation_evidence(_make_rep(), ANTHRO, ROM, None)
        assert evidence == pytest.approx(0.0, abs=EVIDENCE_TOLERANCE)

    def test_too_few_reps_gives_zero(self):
        summary = _make_summary([0.9, 0.7], trend_slope=-0.2)
        evidence = progressive_degradation_evidence(_make_rep(), ANTHRO, ROM, summary)
        assert evidence == pytest.approx(0.0, abs=EVIDENCE_TOLERANCE)


class TestAnkleDorsiflexionLimitation:
    """Ankle limitation is measured against absolute anatomical bounds. It used
    to be observed_peak / capacity where capacity WAS the athlete's own
    observed peak, so utilization was ~1.0 and limited_ankle_df fired at full
    evidence for essentially every athlete."""

    def test_restricted_ankle_reads_fully_limited(self):
        rep = _make_rep(ankle_df_l_max=18.0, ankle_df_r_max=18.0)
        assert ankle_df_limitation(rep) == pytest.approx(1.0)

    def test_normal_ankle_reads_unrestricted(self):
        rep = _make_rep(ankle_df_l_max=34.0, ankle_df_r_max=34.0)
        assert ankle_df_limitation(rep) == pytest.approx(0.0)

    def test_borderline_ankle_scales_linearly(self):
        rep = _make_rep(ankle_df_l_max=25.0, ankle_df_r_max=25.0)
        assert ankle_df_limitation(rep) == pytest.approx(0.5)

    def test_uses_the_better_ankle(self):
        """One restricted ankle is not a bilateral ROM limit."""
        rep = _make_rep(ankle_df_l_max=15.0, ankle_df_r_max=34.0)
        assert ankle_df_limitation(rep) == pytest.approx(0.0)

    def test_typical_athlete_is_not_flagged(self):
        """The default rep reaches 25 deg with a 35 deg observed peak — the
        old ratio gave 0.71 utilization here and any deeper rep gave 1.0."""
        assert limited_ankle_df_evidence(_make_rep(), ANTHRO, ROM, None) < 1.0

    def test_evidence_is_independent_of_observed_peak(self):
        rep = _make_rep(ankle_df_l_max=22.0, ankle_df_r_max=22.0)
        shallow_baseline = {"peak_dorsiflexion": 22.0, "avg_depth": 120.0}
        deep_baseline = {"peak_dorsiflexion": 45.0, "avg_depth": 120.0}
        assert limited_ankle_df_evidence(
            rep, ANTHRO, shallow_baseline, None,
        ) == pytest.approx(
            limited_ankle_df_evidence(rep, ANTHRO, deep_baseline, None)
        )


class TestLeanAttribution:
    """Lean past the reference lifter's is split into anatomy (the athlete's
    proportions), ankles, and a residual that only bracing explains."""

    def test_long_femur_lean_at_own_balance_is_all_anatomy(self):
        rep = _lean_rep(LONG_FEMUR_ANTHRO, UNRESTRICTED_SHANK_DEG, 0.0)
        assert femur_torso_ratio_evidence(rep, ANTHRO, ROM, None) == pytest.approx(1.0)
        assert bracing_failure_evidence(rep, ANTHRO, ROM, None) == pytest.approx(
            0.0, abs=EVIDENCE_TOLERANCE
        )

    def test_reference_build_lean_is_not_anatomy(self):
        rep = _lean_rep(REFERENCE_BUILD_ANTHRO, UNRESTRICTED_SHANK_DEG, 20.0)
        assert femur_torso_ratio_evidence(rep, ANTHRO, ROM, None) < 0.05

    def test_anatomy_share_shrinks_as_unexplained_lean_grows(self):
        at_balance = _lean_rep(LONG_FEMUR_ANTHRO, UNRESTRICTED_SHANK_DEG, 0.0)
        beyond = _lean_rep(LONG_FEMUR_ANTHRO, UNRESTRICTED_SHANK_DEG, 20.0)
        anthro_part = at_balance.expected_pitch_athlete - at_balance.expected_pitch_reference
        expected_share = anthro_part / (anthro_part + 20.0)
        assert femur_torso_ratio_evidence(beyond, ANTHRO, ROM, None) == pytest.approx(
            expected_share, abs=EVIDENCE_TOLERANCE
        )

    def test_stiff_ankle_lean_at_own_balance_is_not_bracing(self):
        rep = _lean_rep(REFERENCE_BUILD_ANTHRO, 18.0, 0.0)
        assert bracing_failure_evidence(rep, ANTHRO, ROM, None) == pytest.approx(
            0.0, abs=EVIDENCE_TOLERANCE
        )

    def test_unexplained_lean_is_bracing(self):
        rep = _lean_rep(REFERENCE_BUILD_ANTHRO, UNRESTRICTED_SHANK_DEG, 20.0)
        assert bracing_failure_evidence(rep, ANTHRO, ROM, None) > 0.9

    def test_chest_dropping_out_of_the_hole_is_bracing_evidence(self):
        rep = _lean_rep(REFERENCE_BUILD_ANTHRO, UNRESTRICTED_SHANK_DEG, 0.0, hip_shoot_deg=18.0)
        assert bracing_failure_evidence(rep, ANTHRO, ROM, None) == pytest.approx(0.6)
