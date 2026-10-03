"""Deadlift profile, gate and registry variants (docs/deadlift/PLAN.md §3.1-3.3)."""

from __future__ import annotations

import math

import pytest

from biomechanics.coaching.cue_cache import FAULT_TO_CUE_MAP
from biomechanics.config import BiomechanicsConfig
from biomechanics.deadlift.analyzer import DeadliftRepAnalyzer
from biomechanics.deadlift.rep_counter import DeadliftRepCounter
from biomechanics.deadlift.session_reference import DeadliftSessionReference
from biomechanics.faults.rules.depth import DepthRule
from biomechanics.profiles import (
    UntrackedProfile,
    coaching_ready_profiles,
    find_profile_class,
    get_profile,
)
from biomechanics.profiles.base import ExerciseProfile
from biomechanics.profiles.deadlift import (
    DEV_COACHING_READY_ENV,
    DeadliftProfile,
    GatedDeadliftProfile,
    _dev_coaching_ready,
)
from biomechanics.profiles.squat import SquatProfile
from biomechanics.profiles.untracked import UntrackedVariantProfile

DEADLIFT_NAMES = ["deadlift", "deadlifts", "Barbell Deadlift", "conventional deadlift", "Barbell Conventional Deadlift"]
VARIANT_NAMES = [
    "Barbell Sumo Deadlift", "Trap Bar Deadlift", "Hex Bar Deadlift", "Deficit Deadlift",
    "Snatch Grip Deadlift", "Single Leg Deadlift", "Rack Pull", "Dumbbell Deadlift", "Kettlebell Deadlift",
]
SQUAT_NAMES = ["squat", "back squat", "front squat", "goblet squat", "bodyweight squat",
               "Barbell Back Squat", "Barbell Front Squat"]
# CONTRACT.md §1, the static min tiers.
CONTRACT_MIN_TIERS = {
    "deadlift_bar_position": "mild",
    "deadlift_shoulders_behind": "moderate",
    "deadlift_setup_hips": "severe",
    "deadlift_hips_shoot": "moderate",
    "deadlift_bar_drift": "moderate",
    "deadlift_lockout": "moderate",
    "deadlift_lean_back": "moderate",
    "deadlift_hip_shift": "moderate",
    "deadlift_bar_tilt": "moderate",
    "deadlift_bent_arms": "mild",
    "deadlift_velocity_loss": "recap",
}


@pytest.fixture
def deadlift_ready(monkeypatch) -> None:
    monkeypatch.setattr(DeadliftProfile, "coaching_ready", True)


class TestGate:
    @pytest.mark.parametrize("exercise_name", DEADLIFT_NAMES)
    def test_deadlift_names_are_untracked_until_coaching_ready(self, exercise_name: str):
        profile = get_profile(exercise_name)
        assert isinstance(profile, GatedDeadliftProfile)
        assert profile.create_fault_rules(BiomechanicsConfig()) == []

    @pytest.mark.parametrize("exercise_name", DEADLIFT_NAMES)
    def test_deadlift_names_resolve_to_the_profile_when_ready(self, exercise_name: str, deadlift_ready):
        assert isinstance(get_profile(exercise_name), DeadliftProfile)

    def test_the_gated_stand_in_keeps_the_deadlifts_safety_flags(self):
        gated = GatedDeadliftProfile()
        assert isinstance(gated, UntrackedProfile)
        assert not gated.allows_camera_refine
        assert not gated.feeds_body_calibration

    def test_menu_lists_the_deadlift_only_when_ready(self, deadlift_ready):
        assert [profile.name for profile in coaching_ready_profiles()] == ["squat", "deadlift"]

    @pytest.mark.parametrize(
        ("listed", "expected"),
        [("deadlift", True), ("squat, Deadlift ", True), ("rdl", False), ("", False)],
    )
    def test_the_dev_override_reads_a_comma_list(self, monkeypatch, listed: str, expected: bool):
        monkeypatch.setenv(DEV_COACHING_READY_ENV, listed)
        assert _dev_coaching_ready("deadlift") is expected


class TestRegistry:
    @pytest.mark.parametrize("exercise_name", VARIANT_NAMES)
    def test_variants_are_never_the_conventional_deadlift(self, exercise_name: str, deadlift_ready):
        assert find_profile_class(exercise_name) is UntrackedVariantProfile
        profile = get_profile(exercise_name)
        assert not profile.allows_camera_refine
        assert not profile.feeds_body_calibration

    @pytest.mark.parametrize("exercise_name", SQUAT_NAMES)
    def test_every_squat_alias_is_still_the_squat(self, exercise_name: str, deadlift_ready):
        assert isinstance(get_profile(exercise_name), SquatProfile)

    @pytest.mark.parametrize(
        "exercise_name", ["Barbell Romanian Deadlift", "RDL", "Stiff Leg Deadlift", "Stiff-Legged Deadlift"],
    )
    def test_romanian_deadlifts_stay_on_their_profile(self, exercise_name: str, deadlift_ready):
        assert get_profile(exercise_name).name == "romanian_deadlift"


class TestHooks:
    def test_every_hook_defaults_to_the_squat_path(self):
        squat = SquatProfile()
        config = BiomechanicsConfig()
        assert squat.create_rep_analyzer(config) is None
        assert squat.create_session_reference() is None
        assert squat.create_set_diagnosis("triangulated") is None
        assert squat.min_cue_tiers(config) == {}
        assert (squat.needs_bar_3d, squat.allows_camera_refine, squat.feeds_body_calibration) == (False, True, True)
        assert (squat.set_idle_timeout_s, squat.waits_for_diagnosis, squat.closed_loop_cue) == (None, None, "")
        assert squat.tracking_keypoints is None
        assert not ExerciseProfile.gate_until_ready

    def test_the_deadlift_brings_its_analyser_reference_and_diagnosis(self):
        profile = DeadliftProfile()
        config = BiomechanicsConfig()
        analyzer = profile.create_rep_analyzer(config)
        assert isinstance(analyzer, DeadliftRepAnalyzer)
        assert isinstance(analyzer.rep_counter, DeadliftRepCounter)
        assert analyzer.rep_counter is analyzer.rep_counter
        assert isinstance(profile.create_session_reference(), DeadliftSessionReference)
        assert profile.create_set_diagnosis("triangulated") is not None
        assert profile.set_idle_timeout_s == pytest.approx(30.0, abs=1e-9)
        assert profile.waits_for_diagnosis is True
        assert math.isnan(profile.get_rep_signal(None))

    def test_no_depth_rule_so_counting_is_never_depth_gated(self):
        rules = DeadliftProfile().create_fault_rules(BiomechanicsConfig())
        assert not any(isinstance(rule, DepthRule) for rule in rules)
        assert len(rules) == len(CONTRACT_MIN_TIERS)

    def test_min_cue_tiers_follow_the_contract(self):
        assert DeadliftProfile().min_cue_tiers(BiomechanicsConfig()) == CONTRACT_MIN_TIERS

    def test_cues_cover_every_mapped_fault_and_the_guidance(self):
        profile = DeadliftProfile()
        cues = profile.get_cue_dict()
        fault_to_cue = profile.get_fault_to_cue_map()
        assert set(fault_to_cue) == set(CONTRACT_MIN_TIERS) - {"deadlift_velocity_loss"}
        assert set(fault_to_cue.values()) <= set(cues)
        for key in ("deadlift_hips_up", "deadlift_hips_down", "deadlift_even_feet_left",
                    "deadlift_step_closer", "deadlift_closer", "deadlift_back", "adjust_good", "rep_1"):
            assert key in cues
        assert "deadlift_flat_back" not in cues
        assert not set(fault_to_cue) & set(FAULT_TO_CUE_MAP)
