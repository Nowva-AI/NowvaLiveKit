"""
OverheadPressProfile (docs/exercises/ohp.md §7.1 Step 4): names and longest-match
variants, gating until coaching-ready, profile flags, the counter config, the
depth metric, and cue keys that exist in the agent's cue text.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from biomechanics.analysis.press_features import PressRepAnalyzer, PressSessionReference
from biomechanics.config import BiomechanicsConfig
from biomechanics.profiles import find_profile_class, get_profile
from biomechanics.profiles.overhead_press import OverheadPressProfile
from biomechanics.profiles.untracked import UntrackedProfile
from biomechanics.utils.types import LEG_KEYPOINTS, CocoKeypoints as CK, JointAngles

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from agent.services.coaching_constants import CUE_TEXT_MAP, FIXED_CUE_TEXT  # noqa: E402

PRESS_NAMES = [
    "overhead_press", "ohp", "military_press", "standing_press", "strict_press", "barbell_overhead_press",
    "Barbell Overhead Press", "Standing Barbell Overhead Press", "Strict Press",
]
NOT_THE_PRESS = [
    "Shoulder Press", "Barbell Press", "Push Press", "Seated Barbell Overhead Press", "Seated Military Press",
    "Dumbbell Overhead Press", "Kettlebell Overhead Press", "Single Arm Overhead Press",
    "Smith Machine Overhead Press", "Arnold Press", "Z Press", "Push Jerk", "Behind The Neck Press",
]


@pytest.fixture
def press_ready(monkeypatch) -> None:
    monkeypatch.setattr(OverheadPressProfile, "coaching_ready", True)


class TestRegistry:
    @pytest.mark.parametrize("name", PRESS_NAMES)
    def test_press_names_resolve_to_the_press(self, name: str):
        assert find_profile_class(name) is OverheadPressProfile

    @pytest.mark.parametrize("name", NOT_THE_PRESS)
    def test_other_presses_are_never_the_standing_press(self, press_ready, name: str):
        assert find_profile_class(name) is not OverheadPressProfile
        assert isinstance(get_profile(name), UntrackedProfile)

    def test_gated_until_coaching_ready(self):
        assert not OverheadPressProfile.coaching_ready
        assert OverheadPressProfile.gate_until_ready
        assert isinstance(get_profile("Barbell Overhead Press"), UntrackedProfile)

    def test_gated_stand_in_never_feeds_body_calibration(self):
        stand_in = get_profile("Barbell Overhead Press")
        assert isinstance(stand_in, UntrackedProfile)
        assert not stand_in.feeds_body_calibration

    def test_ready_press_resolves_to_its_profile(self, press_ready):
        assert isinstance(get_profile("Barbell Overhead Press"), OverheadPressProfile)


class TestProfileHooks:
    def test_flags(self):
        profile = OverheadPressProfile()
        assert not profile.reports_depth
        assert not profile.uses_bilstm_counter
        assert profile.create_set_diagnosis() is None
        assert profile.get_readiness_check() is None
        assert profile.readiness_hint
        assert not profile.feeds_body_calibration
        assert set(LEG_KEYPOINTS) < set(profile.tracking_keypoints)
        assert {CK.LEFT_WRIST, CK.RIGHT_WRIST} <= set(profile.tracking_keypoints)

    def test_analyser_reference_and_counter_come_from_the_press_config(self):
        config = BiomechanicsConfig()
        profile = OverheadPressProfile()
        assert isinstance(profile.create_rep_analyzer(config), PressRepAnalyzer)
        assert isinstance(profile.create_session_reference(config), PressSessionReference)
        assert profile.create_rep_counter_config(config) is config.faults.press.counter

    def test_depth_metric_is_the_elbow_angle(self):
        angles = JointAngles(elbow_flexion_l=10.0, elbow_flexion_r=30.0)
        assert OverheadPressProfile().get_depth_metric(angles) == pytest.approx(160.0)

    def test_hedged_recap_faults_come_from_config(self):
        config = BiomechanicsConfig()
        config.faults.press.hedged_recap_faults = {"press_uneven"}
        assert OverheadPressProfile().get_hedged_recap_faults(config) == frozenset({"press_uneven"})


class TestCueKeys:
    def test_every_cue_key_has_its_text(self):
        cues = OverheadPressProfile().get_cue_dict()
        own = {key for key in cues if key.startswith("press_")}
        assert own == {
            "press_layback", "press_bar_forward", "press_lockout", "press_lockout_left", "press_lockout_right",
            "press_leg_drive", "press_uneven", "press_uneven_left", "press_uneven_right", "press_velocity_loss",
            "press_setup",
        }
        assert own <= set(CUE_TEXT_MAP)
        assert "rep_1" in cues and "knees_out" not in cues

    def test_every_cued_fault_has_a_fixed_praise_line(self):
        for cue in OverheadPressProfile().get_fault_to_cue_map().values():
            assert f"{cue}_fixed" in FIXED_CUE_TEXT
