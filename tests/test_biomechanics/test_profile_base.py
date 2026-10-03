"""
The shared exercise hooks on ExerciseProfile (SHARED-PLUMBING.md §1, §4, §9):
every default is today's squat path, and cue_start_tier reads min_cue_tiers.
"""

from __future__ import annotations

import math
from types import MappingProxyType

import pytest

from biomechanics.config import BiomechanicsConfig
from biomechanics.profiles import get_profile
from biomechanics.profiles.base import DEFAULT_SET_IDLE_TIMEOUT_S, ExerciseProfile
from biomechanics.utils.frame_context import FrameContext
from biomechanics.utils.types import (
    LEG_KEYPOINTS,
    FaultEvent,
    FaultSeverity,
    JointAngles,
    create_empty_skeleton_3d,
)

PROFILE_NAMES = ("Barbell Back Squat", "Reverse Lunge", "Barbell Curl", "Barbell Bench Press")


def _context() -> FrameContext:
    return FrameContext(
        skeleton_3d=create_empty_skeleton_3d(), angles=None, timestamp=0.0,
        phase="idle", in_rep=False, capture_mode="single_camera",
    )


def _fault() -> FaultEvent:
    return FaultEvent(fault_type="press_uneven", severity=FaultSeverity.MILD, severity_score=1.0, message="m")


class TestHookDefaults:
    def test_every_hook_defaults_to_the_squat_path(self):
        profile = ExerciseProfile()
        config = BiomechanicsConfig()

        assert profile.create_rep_analyzer(config) is None
        assert profile.observe_frame(_context()) is None
        assert profile.create_session_reference() is None
        assert profile.create_set_diagnosis() is None
        assert profile.get_readiness_check() is None
        assert profile.on_rep_rejected("partial", None, JointAngles()) == []
        assert profile.shallow_rep_cue(_fault(), streak=3, cued_this_set=0) is None
        assert profile.get_hedged_recap_faults(config) == frozenset()

    def test_flag_defaults_keep_the_squat_unchanged(self):
        profile = ExerciseProfile()
        assert profile.allows_camera_refine
        assert not profile.needs_bar_3d
        assert profile.feeds_body_calibration
        assert not profile.gate_until_ready
        # Only the squat speaks about depth (it sets reports_depth itself).
        assert not profile.reports_depth
        assert profile.readiness_hint == ""
        assert dict(profile.min_cue_tiers) == {}
        assert profile.tracking_keypoints == LEG_KEYPOINTS
        assert profile.set_idle_timeout_s == pytest.approx(DEFAULT_SET_IDLE_TIMEOUT_S)

    @pytest.mark.parametrize("exercise_name", PROFILE_NAMES)
    def test_existing_profiles_keep_the_defaults(self, exercise_name: str):
        profile = get_profile(exercise_name)
        assert profile.create_rep_analyzer(BiomechanicsConfig()) is None
        assert profile.cue_start_tier("knee_valgus", _context()) == FaultSeverity.MILD


class TestStartTier:
    def test_a_missing_fault_starts_at_mild(self):
        assert ExerciseProfile().cue_start_tier("press_uneven", _context()) == FaultSeverity.MILD

    def test_min_cue_tiers_set_the_static_start(self):
        class _Tiered(ExerciseProfile):
            min_cue_tiers = MappingProxyType({"press_uneven": FaultSeverity.SEVERE})

        assert _Tiered().cue_start_tier("press_uneven", _context()) == FaultSeverity.SEVERE
        assert _Tiered().cue_start_tier("press_grip", _context()) == FaultSeverity.MILD

    def test_the_default_table_cannot_be_mutated_by_accident(self):
        with pytest.raises(TypeError):
            ExerciseProfile.min_cue_tiers["press_uneven"] = FaultSeverity.SEVERE


class TestFrameContextDefaults:
    def test_up_defaults_to_world_up_from_the_body(self):
        ctx = _context()
        assert list(ctx.up) == [0.0, -1.0, 0.0]
        assert ctx.gravity_source == "body"
        assert not ctx.pitch_certified
        assert math.isnan(ctx.camera_pitch_deg) and math.isnan(ctx.mount_pitch_deg)

    def test_each_context_owns_its_up_vector(self):
        first, second = _context(), _context()
        first.up[1] = 1.0
        assert second.up[1] == -1.0
