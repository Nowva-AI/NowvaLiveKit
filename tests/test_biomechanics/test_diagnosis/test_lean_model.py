"""Tests for the sagittal balance model: the trunk pitch that keeps the
shoulders over midfoot for a given build, ankle tilt and depth."""

from __future__ import annotations

import math

import pytest

from biomechanics.diagnosis.lean_model import (
    MAX_PITCH_DEG,
    UNRESTRICTED_SHANK_DEG,
    balanced_trunk_pitch_deg,
    expected_pitches,
)

# The 188.5 cm reference user (FINDINGS.md: torso 0.543, femur 0.462, tibia 0.464 m).
REFERENCE_ANTHRO = {
    "torso_length": 0.543,
    "femur_length_avg": 0.462,
    "tibia_length_avg": 0.464,
    "foot_length": 0.20,
}
PARALLEL_RATIO = 0.0
STANDING_RATIO = 1.0

# Published high-bar back squats sit ~30-45° at parallel; this build with free
# ankles lands at ~33°.
REFERENCE_PITCH_AT_PARALLEL_DEG = 33.3
PITCH_TOLERANCE_DEG = 0.5
EXACT_TOLERANCE_DEG = 1e-9


def _anthro(**overrides) -> dict:
    values = dict(REFERENCE_ANTHRO)
    values.update(overrides)
    return values


class TestBalancedTrunkPitch:

    def test_closed_form_geometry(self):
        # Vertical shin, thigh parallel: hip sits one femur behind the ankle.
        # With no foot, the shoulders must reach femur/torso = 0.5 → 30°.
        pitch = balanced_trunk_pitch_deg(
            femur_m=0.3, tibia_m=0.4, torso_m=0.6, shank_deg=0.0,
            depth_ratio=PARALLEL_RATIO, foot_len_m=0.0,
        )
        assert pitch == pytest.approx(30.0, abs=EXACT_TOLERANCE_DEG)

    def test_hip_ahead_of_midfoot_needs_no_lean(self):
        # Standing with the thigh vertical over a forward knee: the hip is
        # already ahead of midfoot, so the pitch clamps at upright.
        pitch = balanced_trunk_pitch_deg(
            femur_m=0.46, tibia_m=0.46, torso_m=0.54, shank_deg=30.0,
            depth_ratio=STANDING_RATIO, foot_len_m=0.20,
        )
        assert pitch == pytest.approx(0.0, abs=EXACT_TOLERANCE_DEG)

    def test_extreme_proportions_cap_at_max_pitch(self):
        pitch = balanced_trunk_pitch_deg(
            femur_m=0.80, tibia_m=0.30, torso_m=0.40, shank_deg=0.0,
            depth_ratio=PARALLEL_RATIO, foot_len_m=0.20,
        )
        assert pitch == pytest.approx(MAX_PITCH_DEG, abs=1e-6)


class TestExpectedPitches:

    def test_reference_at_parallel_with_free_ankles(self):
        reference, athlete, with_ankles = expected_pitches(
            REFERENCE_ANTHRO, PARALLEL_RATIO, UNRESTRICTED_SHANK_DEG,
        )
        assert reference == pytest.approx(REFERENCE_PITCH_AT_PARALLEL_DEG, abs=PITCH_TOLERANCE_DEG)
        # This build is the reference build, so its own lean matches.
        assert athlete == pytest.approx(reference, abs=PITCH_TOLERANCE_DEG)

    def test_unrestricted_ankles_add_no_lean(self):
        _, athlete, with_ankles = expected_pitches(
            REFERENCE_ANTHRO, PARALLEL_RATIO, UNRESTRICTED_SHANK_DEG,
        )
        assert with_ankles == pytest.approx(athlete, abs=EXACT_TOLERANCE_DEG)

    def test_stiffer_ankles_raise_expected_pitch(self):
        pitches = [
            expected_pitches(REFERENCE_ANTHRO, PARALLEL_RATIO, shank_deg)[2]
            for shank_deg in (35.0, 30.0, 25.0, 20.0, 15.0)
        ]
        assert pitches == sorted(pitches)
        assert len(set(pitches)) == len(pitches)

    def test_ankles_do_not_change_reference_or_athlete_pitch(self):
        stiff = expected_pitches(REFERENCE_ANTHRO, PARALLEL_RATIO, 15.0)
        free = expected_pitches(REFERENCE_ANTHRO, PARALLEL_RATIO, 35.0)
        assert stiff[0] == pytest.approx(free[0], abs=EXACT_TOLERANCE_DEG)
        assert stiff[1] == pytest.approx(free[1], abs=EXACT_TOLERANCE_DEG)

    def test_longer_femurs_raise_expected_pitch(self):
        pitches = [
            expected_pitches(_anthro(femur_length_avg=femur_m), PARALLEL_RATIO, UNRESTRICTED_SHANK_DEG)[1]
            for femur_m in (0.40, 0.44, 0.48, 0.52, 0.56)
        ]
        assert pitches == sorted(pitches)
        assert len(set(pitches)) == len(pitches)

    def test_reference_ignores_the_athletes_own_femur(self):
        # The reference is population proportions on this torso: it is the
        # yardstick the athlete's own build is compared against.
        short = expected_pitches(_anthro(femur_length_avg=0.40), PARALLEL_RATIO, UNRESTRICTED_SHANK_DEG)
        long = expected_pitches(_anthro(femur_length_avg=0.56), PARALLEL_RATIO, UNRESTRICTED_SHANK_DEG)
        assert short[0] == pytest.approx(long[0], abs=EXACT_TOLERANCE_DEG)

    def test_never_above_eighty_degrees(self):
        assert MAX_PITCH_DEG == pytest.approx(80.0)
        for femur_m in (0.40, 0.50, 0.60, 0.70, 0.90):
            for shank_deg in (0.0, 10.0, 20.0, 30.0, 40.0):
                for depth_ratio in (-0.5, -0.2, 0.0, 0.3, 0.6):
                    pitches = expected_pitches(
                        _anthro(femur_length_avg=femur_m), depth_ratio, shank_deg,
                    )
                    for pitch in pitches:
                        assert 0.0 <= pitch <= MAX_PITCH_DEG + 1e-9

    def test_very_long_femur_saturates_at_the_cap(self):
        _, athlete, _ = expected_pitches(
            _anthro(femur_length_avg=0.90), PARALLEL_RATIO, UNRESTRICTED_SHANK_DEG,
        )
        assert athlete == pytest.approx(MAX_PITCH_DEG, abs=1e-6)

    def test_standing_needs_no_lean(self):
        pitches = expected_pitches(REFERENCE_ANTHRO, STANDING_RATIO, UNRESTRICTED_SHANK_DEG)
        assert pitches == pytest.approx((0.0, 0.0, 0.0), abs=EXACT_TOLERANCE_DEG)

    def test_missing_ankle_tilt_leaves_only_the_ankle_pitch_unknown(self):
        reference, athlete, with_ankles = expected_pitches(
            REFERENCE_ANTHRO, PARALLEL_RATIO, math.nan,
        )
        assert math.isfinite(reference)
        assert math.isfinite(athlete)
        assert math.isnan(with_ankles)

    def test_unmeasured_depth_gives_no_expectation(self):
        pitches = expected_pitches(REFERENCE_ANTHRO, math.nan, UNRESTRICTED_SHANK_DEG)
        assert all(math.isnan(pitch) for pitch in pitches)

    @pytest.mark.parametrize("missing_key", ["torso_length", "femur_length_avg", "tibia_length_avg"])
    def test_missing_segment_gives_no_expectation(self, missing_key: str):
        anthro = dict(REFERENCE_ANTHRO)
        del anthro[missing_key]
        pitches = expected_pitches(anthro, PARALLEL_RATIO, UNRESTRICTED_SHANK_DEG)
        assert all(math.isnan(pitch) for pitch in pitches)
