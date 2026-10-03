"""
Tests for exercise profile resolution: names from the program library and
the voice agent reach the right profile, unknown exercises are untracked
rather than squats, and only squats are calibrated or offered for coaching.
"""

from __future__ import annotations

import math

import pytest

from biomechanics.calibration import get_movement_pattern
from biomechanics.config import BiomechanicsConfig
from biomechanics.profiles import UntrackedProfile, coaching_ready_profiles, find_profile_class, get_profile


class TestNameResolution:
    @pytest.mark.parametrize(
        ("exercise_name", "profile_name"),
        [
            ("Barbell Back Squat", "squat"),
            ("Barbell Front Squat", "squat"),
            ("Dumbbell Goblet Squat", "squat"),
            ("Pause Squat", "squat"),
            ("squat", "squat"),
            ("Barbell Romanian Deadlift", "romanian_deadlift"),
            ("Single-Leg RDL", "romanian_deadlift"),
            # Gated until the deadlift is validated (docs/deadlift/PLAN.md §3.1); back
            # to "deadlift" at J8. Sumo is never the conventional deadlift's profile.
            ("Barbell Conventional Deadlift", "untracked"),
            ("Barbell Sumo Deadlift", "untracked"),
            ("Barbell Overhead Press", "overhead_press"),
            ("Split Squat", "lunge"),
            ("Bulgarian Split Squat", "bulgarian_split_squat"),
            ("Reverse Lunge", "lunge"),
        ],
    )
    def test_exercise_resolves_to_its_profile(self, exercise_name: str, profile_name: str):
        assert get_profile(exercise_name).name == profile_name

    @pytest.mark.parametrize(
        "exercise_name",
        ["Barbell Bench Press", "Good Morning", "Hip Thrust", "Pull Up", "Leg Curl", "Calf Raise"],
    )
    def test_unmodelled_exercise_is_untracked_not_a_squat(self, exercise_name: str):
        assert find_profile_class(exercise_name) is None
        assert isinstance(get_profile(exercise_name), UntrackedProfile)


class TestUntrackedProfile:
    def test_untracked_profile_has_no_rules_and_never_starts_a_rep(self):
        profile = UntrackedProfile()
        assert profile.create_fault_rules(BiomechanicsConfig()) == []
        assert math.isnan(profile.get_rep_signal(skeleton_3d=None))
        assert profile.get_fault_to_cue_map() == {}


class TestProfileFlags:
    def test_only_squats_are_offered_for_coaching(self):
        assert [profile.name for profile in coaching_ready_profiles()] == ["squat"]

    def test_only_squats_use_the_bilstm_and_diagnosis_engine(self):
        for exercise_name in ("Barbell Overhead Press", "Barbell Romanian Deadlift", "Reverse Lunge"):
            profile = get_profile(exercise_name)
            assert not profile.uses_bilstm_counter
            assert not profile.uses_diagnosis_engine
        squat = get_profile("Barbell Back Squat")
        assert squat.uses_bilstm_counter
        assert squat.uses_diagnosis_engine


class TestMovementPattern:
    @pytest.mark.parametrize("exercise_name", ["Barbell Back Squat", "Dumbbell Goblet Squat", "squat"])
    def test_squat_variants_share_the_squat_calibration(self, exercise_name: str):
        assert get_movement_pattern(exercise_name) == "squat"

    @pytest.mark.parametrize(
        "exercise_name",
        ["Barbell Romanian Deadlift", "Barbell Overhead Press", "Barbell Bench Press", "Split Squat"],
    )
    def test_uncalibrated_exercises_have_no_calibration_row(self, exercise_name: str):
        assert get_movement_pattern(exercise_name) is None
