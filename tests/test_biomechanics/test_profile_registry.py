"""
Tests for exercise profile resolution: names from the program library and
the voice agent reach the right profile, unknown exercises are untracked
rather than squats, and only squats are calibrated or offered for coaching.
Profiles that gate themselves until coaching-ready (press, RDL) resolve to an
untracked stand-in, and unmodelled variants to UntrackedVariantProfile. Every
registered profile's readiness check, and its analyser's, accepts a pre-latch
frame, whose context has no angles.
"""

from __future__ import annotations

import math

import numpy as np
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
from biomechanics.utils.frame_context import FrameContext
from biomechanics.utils.types import CocoKeypoints as CK
from biomechanics.utils.types import Skeleton3D
from conftest import (
    SYNTHETIC_CLOCK_START_S,
    SYNTHETIC_CONFIDENCE,
    SYNTHETIC_MIN_DEPTH_RATIO,
    SyntheticSession,
    world_squat_points,
)

GATED_PROFILES = (OverheadPressProfile, RomanianDeadliftProfile)
# Every class a scheduled program can reach, gated ones included.
PROFILE_CLASSES = sorted({*PROFILE_REGISTRY.values(), UntrackedProfile}, key=lambda cls: cls.__name__)
# Profiles whose pre-plumbing readiness predicates took (skeleton_3d, angles).
LEGACY_READINESS_EXERCISES = ("Barbell Row", "Skull Crusher", "Overhead Tricep Extension")
PRE_LATCH_FRAMES = 20


def _pre_latch_context(capture_mode: str, confidence: float) -> FrameContext:
    """What the pipeline hands a readiness check: the raw hip-centred skeleton, no angles."""
    world_points = world_squat_points(SYNTHETIC_MIN_DEPTH_RATIO)
    centred = world_points - (world_points[CK.LEFT_HIP] + world_points[CK.RIGHT_HIP]) / 2.0
    confidences = np.full(len(world_points), confidence)
    skeleton_world = None
    if capture_mode == "triangulated":
        skeleton_world = Skeleton3D.from_numpy(world_points, confidences=confidences)
    return FrameContext(
        skeleton_3d=Skeleton3D.from_numpy(centred, confidences=confidences),
        angles=None,
        timestamp=SYNTHETIC_CLOCK_START_S,
        phase="idle",
        in_rep=False,
        capture_mode=capture_mode,
        raw_confidences=confidences,
        skeleton_world=skeleton_world,
    )


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


class TestReadinessBeforeTheLatch:
    """The pipeline runs these on every pre-latch frame; one that raises ends
    the pose loop for the rest of the session."""

    @pytest.mark.parametrize("profile_class", PROFILE_CLASSES, ids=lambda cls: cls.__name__)
    @pytest.mark.parametrize(
        ("capture_mode", "confidence"),
        [("single_camera", SYNTHETIC_CONFIDENCE), ("triangulated", SYNTHETIC_CONFIDENCE), ("single_camera", 0.0)],
        ids=["mono", "rig", "keypoints_lost"],
    )
    def test_readiness_checks_accept_a_context_without_angles(
        self, profile_class: type[ExerciseProfile], capture_mode: str, confidence: float,
    ):
        profile = profile_class()
        ctx = _pre_latch_context(capture_mode, confidence)

        profile_check = profile.get_readiness_check()
        if profile_check is not None:
            assert profile_check(ctx) in (True, False)
        analyzer = profile.create_rep_analyzer(BiomechanicsConfig())
        analyzer_check = getattr(analyzer, "is_ready", None)
        if analyzer_check is not None:
            assert analyzer_check(ctx) in (True, False)

    @pytest.mark.parametrize("exercise_name", LEGACY_READINESS_EXERCISES)
    def test_a_scheduled_legacy_exercise_survives_its_pre_latch_frames(self, monkeypatch, exercise_name: str):
        session = SyntheticSession(monkeypatch, multi_camera=False, exercise_name=exercise_name)
        assert session.pipeline.profile.get_readiness_check() is None

        session.run([0.0] * PRE_LATCH_FRAMES)

        assert session.pipeline.is_ready


class TestProfileFlags:
    def test_only_squats_are_offered_for_coaching(self):
        assert [profile.name for profile in coaching_ready_profiles()] == ["squat"]

    @pytest.mark.parametrize("profile_class", PROFILE_CLASSES, ids=lambda cls: cls.__name__)
    def test_only_squats_report_depth(self, profile_class: type[ExerciseProfile]):
        assert profile_class.reports_depth is (profile_class is SquatProfile)

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
