"""
Tests for exercise profile resolution: names from the program library and
the voice agent reach the right profile, unknown exercises are untracked
rather than squats, and only squats are calibrated or offered for coaching.
Profiles that gate themselves until coaching-ready (press, RDL) resolve to an
untracked stand-in, and unmodelled variants to UntrackedVariantProfile.
"""

from __future__ import annotations

import math

import pytest

from biomechanics.calibration import get_movement_pattern
from biomechanics.config import BiomechanicsConfig
from biomechanics.profiles import (
    PROFILE_REGISTRY,
    ExerciseProfile,
    UntrackedProfile,
    coaching_ready_profiles,
    find_profile_class,
    get_profile,
)
from biomechanics.profiles import registry
from biomechanics.profiles.overhead_press import OverheadPressProfile
from biomechanics.profiles.romanian_deadlift import RomanianDeadliftProfile
from biomechanics.profiles.squat import SquatProfile
from biomechanics.profiles.untracked_variant import (
    DEADLIFT_VARIANTS,
    PRESS_VARIANTS,
    RDL_VARIANTS,
    UntrackedVariantProfile,
)

GATED_PROFILES = (OverheadPressProfile, RomanianDeadliftProfile)


@pytest.fixture
def gated_profiles_ready(monkeypatch) -> None:
    for profile_class in GATED_PROFILES:
        monkeypatch.setattr(profile_class, "coaching_ready", True)


class TestNameResolution:
    @pytest.mark.parametrize(
        ("exercise_name", "profile_name"),
        [
            ("Barbell Back Squat", "squat"),
            ("Barbell Front Squat", "squat"),
            ("Dumbbell Goblet Squat", "squat"),
            ("Pause Squat", "squat"),
            ("squat", "squat"),
            ("Barbell Conventional Deadlift", "deadlift"),
            ("Split Squat", "lunge"),
            ("Bulgarian Split Squat", "bulgarian_split_squat"),
            ("Reverse Lunge", "lunge"),
        ],
    )
    def test_exercise_resolves_to_its_profile(self, exercise_name: str, profile_name: str):
        assert get_profile(exercise_name).name == profile_name

    @pytest.mark.parametrize(
        ("exercise_name", "profile_name"),
        [
            ("Barbell Romanian Deadlift", "romanian_deadlift"),
            ("RDL", "romanian_deadlift"),
            ("Barbell Overhead Press", "overhead_press"),
            ("Military Press", "overhead_press"),
        ],
    )
    def test_gated_exercise_resolves_to_its_profile_once_ready(
        self, gated_profiles_ready, exercise_name: str, profile_name: str,
    ):
        assert get_profile(exercise_name).name == profile_name

    @pytest.mark.parametrize("exercise_name", ["Barbell Romanian Deadlift", "Barbell Overhead Press"])
    def test_gated_exercise_is_untracked_until_ready(self, exercise_name: str):
        profile = get_profile(exercise_name)
        assert type(profile) is UntrackedProfile
        assert profile.create_fault_rules(BiomechanicsConfig()) == []

    @pytest.mark.parametrize("squat_alias", ["Barbell Back Squat", "Front Squat", "Goblet Squat", "Bodyweight Squat"])
    def test_every_squat_alias_resolves_to_the_squat(self, squat_alias: str):
        assert find_profile_class(squat_alias) is SquatProfile


class TestUntrackedVariants:
    @pytest.mark.parametrize("variant", DEADLIFT_VARIANTS + RDL_VARIANTS + PRESS_VARIANTS)
    def test_every_variant_name_is_an_untracked_variant(self, gated_profiles_ready, variant: str):
        assert find_profile_class(variant) is UntrackedVariantProfile
        assert isinstance(get_profile(variant), UntrackedProfile)

    @pytest.mark.parametrize(
        "exercise_name",
        [
            "Barbell Sumo Deadlift",
            "Single-Leg RDL",
            "Dumbbell Single Leg Romanian Deadlift",
            "Stiff Leg Deadlift",
            "Push Press",
            "Seated Barbell Overhead Press",
            "Dumbbell Shoulder Press",
        ],
    )
    def test_variant_wins_over_the_profile_it_contains(self, gated_profiles_ready, exercise_name: str):
        assert find_profile_class(exercise_name) is UntrackedVariantProfile

    def test_shoulder_press_is_not_the_standing_press(self):
        assert find_profile_class("Shoulder Press") is not OverheadPressProfile

    def test_variants_are_never_offered_and_feed_nothing(self):
        assert not UntrackedVariantProfile.coaching_ready
        assert not UntrackedVariantProfile.allows_camera_refine
        assert not UntrackedVariantProfile.feeds_body_calibration
        assert UntrackedVariantProfile not in coaching_ready_profiles()


class TestGating:
    def test_gated_stand_in_keeps_the_profiles_safety_flags(self, monkeypatch):
        monkeypatch.setattr(RomanianDeadliftProfile, "allows_camera_refine", False)
        monkeypatch.setattr(RomanianDeadliftProfile, "feeds_body_calibration", False)
        profile = get_profile("Barbell Romanian Deadlift")
        assert profile.name == "untracked"
        assert not profile.allows_camera_refine
        assert not profile.feeds_body_calibration

    def test_only_the_press_and_rdl_gate_themselves(self):
        gated = {cls for cls in PROFILE_REGISTRY.values() if cls.gate_until_ready}
        assert gated == set(GATED_PROFILES)

    def test_dev_override_marks_a_profile_ready_at_registration(self, monkeypatch):
        monkeypatch.setenv(registry.DEV_COACHING_READY_ENV, "dev_lift, other_lift")
        monkeypatch.setattr(registry, "PROFILE_REGISTRY", dict(PROFILE_REGISTRY))

        @registry.register_profile("dev_lift")
        class DevLiftProfile(ExerciseProfile):
            name = "dev_lift"
            gate_until_ready = True

        assert DevLiftProfile.coaching_ready
        assert registry.get_profile("dev lift").name == "dev_lift"
        assert DevLiftProfile in registry.coaching_ready_profiles()

    def test_dev_override_names_are_parsed_and_trimmed(self, monkeypatch):
        monkeypatch.setenv(registry.DEV_COACHING_READY_ENV, " overhead_press ,romanian_deadlift,,")
        assert registry.dev_coaching_ready_names() == {"overhead_press", "romanian_deadlift"}

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
