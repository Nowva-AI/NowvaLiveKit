"""
RomanianDeadliftProfile wiring (docs/exercises/rdl.md §3, §6, §7 step 10): flags,
rules in priority order, cue keys the voice agent already has, the rep signal,
readiness and the counter config.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

from biomechanics.analysis.hinge_features import RdlRepAnalyzer
from biomechanics.coaching.cue_cache import FAULT_CUE_PRIORITY, GENERIC_POSITIVE_CUE_KEYS
from biomechanics.config import BiomechanicsConfig
from biomechanics.faults.hinge_reference import HingeSessionReference
from biomechanics.faults.observability import FAULT_MEASUREMENT
from biomechanics.faults.rules.back_rounding import BackRoundingRule
from biomechanics.faults.rules.range_of_motion import RangeOfMotionRule
from biomechanics.faults.rules.symmetry import SymmetryRule
from biomechanics.profiles import find_profile_class, get_profile
from biomechanics.profiles.romanian_deadlift import RDL_FAULT_TO_CUE, RomanianDeadliftProfile
from biomechanics.utils.types import CocoKeypoints as CK
from biomechanics.utils.types import Skeleton3D
from test_rdl_poses import CLEAN_BOTTOM, NUM_KEYPOINTS, STANDING

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from agent.services.coaching_constants import (  # noqa: E402
    AGENT_SIDE_CUE_KEYS,
    CUE_DISPLAY_LABELS,
    CUE_TEXT_MAP,
    FIXED_CUE_TEXT,
)

RDL_FAULT_TYPES = (
    "rdl_hip_sink", "rdl_bar_drift", "rdl_lockout", "rdl_bar_tilt", "rdl_range_drift", "rdl_tempo", "rdl_partial",
)
RDL_PRIORITY_BLOCK = range(40, 50)
SETUP_CUE_KEYS = ("rdl_setup", "rdl_back_note", "level_camera")
# rdl.md §3.3 for 188.5 cm: shoulders ~147 cm above the ankles standing, ~105 at the bottom.
STANDING_SHOULDER_CM = 147.0
BOTTOM_SHOULDER_CM = 105.0
SHOULDER_TOLERANCE_CM = 2.0


@pytest.fixture
def rdl_profile() -> RomanianDeadliftProfile:
    profile = RomanianDeadliftProfile()
    profile.create_fault_rules(BiomechanicsConfig())
    return profile


class TestFlags:
    def test_flags_follow_the_plan(self):
        assert RomanianDeadliftProfile.name == "romanian_deadlift"
        assert RomanianDeadliftProfile.gate_until_ready
        assert not RomanianDeadliftProfile.coaching_ready
        assert not RomanianDeadliftProfile.reports_depth
        assert not RomanianDeadliftProfile.allows_camera_refine
        assert not RomanianDeadliftProfile.feeds_body_calibration
        assert not RomanianDeadliftProfile.uses_bilstm_counter
        assert not RomanianDeadliftProfile.uses_diagnosis_engine
        assert RomanianDeadliftProfile.readiness_hint == "HANDS IN VIEW"
        assert RomanianDeadliftProfile.display_name

    @pytest.mark.parametrize("exercise_name", ["Barbell Romanian Deadlift", "Romanian Deadlift", "RDL"])
    def test_names_resolve_to_the_rdl(self, exercise_name: str):
        assert find_profile_class(exercise_name) is RomanianDeadliftProfile

    def test_dev_override_serves_the_rdl(self, monkeypatch):
        monkeypatch.setattr(RomanianDeadliftProfile, "coaching_ready", True)
        assert isinstance(get_profile("Barbell Romanian Deadlift"), RomanianDeadliftProfile)

    def test_gated_stand_in_keeps_the_rdl_safety_flags(self):
        stand_in = get_profile("Barbell Romanian Deadlift")
        assert stand_in.name == "untracked"
        assert not stand_in.allows_camera_refine
        assert not stand_in.feeds_body_calibration


class TestRulesAndCues:
    def test_rules_are_in_priority_order(self, rdl_profile: RomanianDeadliftProfile):
        rules = rdl_profile.create_fault_rules(BiomechanicsConfig())
        priorities = [FAULT_CUE_PRIORITY[rule.fault_type.value] for rule in rules]
        assert priorities == sorted(priorities)
        assert all(priority in RDL_PRIORITY_BLOCK for priority in priorities)

    def test_scaffold_rules_are_gone(self, rdl_profile: RomanianDeadliftProfile):
        rules = rdl_profile.create_fault_rules(BiomechanicsConfig())
        assert not any(isinstance(rule, (BackRoundingRule, RangeOfMotionRule, SymmetryRule)) for rule in rules)

    def test_every_fault_type_is_mapped_registered_and_prioritised(self, rdl_profile: RomanianDeadliftProfile):
        cue_map = rdl_profile.get_fault_to_cue_map()
        assert tuple(cue_map) == RDL_FAULT_TYPES
        for fault_type in RDL_FAULT_TYPES:
            assert fault_type in FAULT_MEASUREMENT
            assert FAULT_CUE_PRIORITY[fault_type] in RDL_PRIORITY_BLOCK

    def test_cue_keys_are_the_ones_the_agent_has(self, rdl_profile: RomanianDeadliftProfile):
        for cue_key in rdl_profile.get_fault_to_cue_map().values():
            assert cue_key.startswith("rdl_")
            assert cue_key in CUE_TEXT_MAP
            assert cue_key in CUE_DISPLAY_LABELS
            if cue_key != RDL_FAULT_TO_CUE["rdl_partial"]:
                assert f"{cue_key}_fixed" in FIXED_CUE_TEXT

    def test_cue_dict_has_corrections_positives_and_rep_counts_but_no_agent_side_clips(
        self, rdl_profile: RomanianDeadliftProfile,
    ):
        cue_dict = rdl_profile.get_cue_dict()
        assert set(RDL_FAULT_TO_CUE.values()) <= set(cue_dict)
        assert set(GENERIC_POSITIVE_CUE_KEYS) <= set(cue_dict)
        assert "rep_1" in cue_dict
        assert not set(SETUP_CUE_KEYS) & set(cue_dict)
        assert set(SETUP_CUE_KEYS) <= AGENT_SIDE_CUE_KEYS

    def test_hedged_recap_faults_come_from_the_config(self, rdl_profile: RomanianDeadliftProfile):
        config = BiomechanicsConfig()
        assert rdl_profile.get_hedged_recap_faults(config) == frozenset()
        config.faults.rdl.hedged_recap_faults = {"rdl_hip_sink"}
        assert rdl_profile.get_hedged_recap_faults(config) == {"rdl_hip_sink"}


class TestHooks:
    def test_analyser_reference_and_counter_come_from_the_rdl_config(self, rdl_profile: RomanianDeadliftProfile):
        config = BiomechanicsConfig()
        assert isinstance(rdl_profile.create_rep_analyzer(config), RdlRepAnalyzer)
        assert isinstance(rdl_profile.create_session_reference(), HingeSessionReference)
        assert rdl_profile.create_rep_counter_config(config) == config.faults.rdl.counter
        assert rdl_profile.get_readiness_check() is None

    def test_signal_rises_on_the_descent_and_is_nan_without_shoulders(self, rdl_profile: RomanianDeadliftProfile):
        standing = rdl_profile.get_rep_signal(Skeleton3D.from_numpy(STANDING.points()))
        bottom = rdl_profile.get_rep_signal(Skeleton3D.from_numpy(CLEAN_BOTTOM.points()))
        assert standing == pytest.approx(-STANDING_SHOULDER_CM, abs=SHOULDER_TOLERANCE_CM)
        assert bottom == pytest.approx(-BOTTOM_SHOULDER_CM, abs=SHOULDER_TOLERANCE_CM)
        confidences = np.full(NUM_KEYPOINTS, 1.0)
        confidences[CK.RIGHT_SHOULDER] = 0.0
        assert math.isnan(rdl_profile.get_rep_signal(Skeleton3D.from_numpy(STANDING.points(), confidences=confidences)))
