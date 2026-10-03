"""
Squat invariants (deadlift PLAN §5, milestone J0): the squat's fault rules in
cue-priority order with every threshold (code defaults and the live
config/biomechanics.yaml), its counting and IPC config, its cue keys and
priorities, its cache_cues payload and its profile flags, pinned as literals so
that changing any one fails the test that names it.
"""

from __future__ import annotations

import json
from typing import Any, Callable

import pytest

from biomechanics.coaching.cue_cache import (
    DEFAULT_FAULT_CUE_PRIORITY,
    FAULT_CUE_PRIORITY,
    FAULT_TO_CUE_MAP,
    GENERIC_POSITIVE_CUE_KEYS,
    POSITIVE_CUE_KEYS,
    PREEMPT_GAP_RATIO_ORCHESTRATOR,
    PREEMPT_GAP_RATIO_PIPELINE,
    SIDE_CUE_SIDES,
)
from biomechanics.coaching.ipc_bridge import IPCBridge
from biomechanics.config import BiomechanicsConfig, load_pipeline_config
from biomechanics.faults.fault_types import FaultRule
from biomechanics.profiles import get_profile
from biomechanics.profiles.registry import PROFILE_REGISTRY
from biomechanics.profiles.squat import SquatProfile

SQUAT_EXERCISE = "Barbell Back Squat"
THRESHOLD_TOLERANCE = 1e-9

# Code defaults, and what the pipeline subprocess actually loads.
CONFIG_SOURCES: tuple[tuple[str, Callable[[], BiomechanicsConfig]], ...] = (
    ("code_defaults", BiomechanicsConfig),
    ("live_yaml", load_pipeline_config),
)

SQUAT_RULES = (
    ("KneeTrackingRule", "knee_valgus"),
    ("HipShootRule", "hip_shoot"),
    ("HeelRiseRule", "heel_rise"),
    ("BalanceRule", "balance"),
    ("HipShiftRule", "hip_shift"),
    ("BarTiltAsymmetryRule", "bilateral_asymmetry"),
    ("DepthRule", "depth"),
    ("FootPlacementRule", "foot_placement"),
    ("StandingLockoutRule", "lockout"),
    ("DescentControlRule", "tempo"),
    ("DepthDriftRule", "depth_drift"),
    ("VelocityLossRule", "velocity_loss"),
)

# Every public attribute of every squat rule; tiered dicts flattened as "<name>.<tier>".
SQUAT_RULE_THRESHOLDS: dict[str, dict[str, float]] = {
    "knee_valgus": {"mild_threshold": 8.0, "moderate_threshold": 13.0, "severe_threshold": 18.0},
    "hip_shoot": {"mild_threshold": 8.0, "moderate_threshold": 12.0, "severe_threshold": 18.0},
    "heel_rise": {
        "mild_cm": 1.5, "moderate_cm": 3.0, "severe_cm": 5.0, "cooldown_s": 2.0, "min_rise_duration_s": 0.23,
    },
    "balance": {
        "forward.mild": 0.2, "forward.moderate": 0.3, "forward.severe": 0.4,
        "backward.mild": 0.3, "backward.moderate": 0.4, "backward.severe": 0.5,
    },
    "hip_shift": {"mild_threshold": 0.1, "moderate_threshold": 0.15, "severe_threshold": 0.22},
    "bilateral_asymmetry": {
        "bar_length_m": 2.2,
        "mild_deg": 2.0, "moderate_deg": 4.0, "severe_deg": 7.0,
        "mild_cm": 3.0, "moderate_cm": 6.0, "severe_cm": 10.0,
    },
    "depth": {"tolerance_ratio": 0.08, "moderate_deficit_ratio": 0.25, "severe_deficit_ratio": 0.5},
    "foot_placement": {
        "stagger.mild": 0.15, "stagger.moderate": 0.25, "stagger.severe": 0.35,
        "flare.mild": 10.0, "flare.moderate": 15.0, "flare.severe": 22.0,
    },
    "lockout": {"mild_threshold": 0.05, "moderate_threshold": 0.08, "severe_threshold": 0.12},
    "tempo": {"mild_seconds": 0.6, "moderate_seconds": 0.45, "severe_seconds": 0.3},
    "depth_drift": {"mild_threshold": 0.15, "moderate_threshold": 0.25, "severe_threshold": 0.35},
    "velocity_loss": {"mild_pct": 20.0, "moderate_pct": 30.0, "severe_pct": 40.0},
}

SQUAT_DEPTH_TARGETS = {
    "default_target_ratio": 0.0,
    "uncalibrated_target_ratio": 0.2,
    "max_target_ratio": 0.5,
    "tolerance_ratio": 0.08,
    "moderate_deficit_ratio": 0.25,
    "severe_deficit_ratio": 0.5,
}
SQUAT_HIP_COUNTER = {
    "entry_vel_threshold": 3.0,
    "bottom_vel_threshold": 5.0,
    "ascending_vel_threshold": 3.0,
    "min_depth_cm": 10.0,
    "standing_return_cm": 3.0,
    "min_descending_s": 0.1,
    "min_bottom_s": 0.067,
    "min_ascending_s": 0.1,
    "min_rep_duration_s": 0.5,
}
SQUAT_IPC = {"frame_send_interval": 10, "fault_cooldown_seconds": 3.0}
SQUAT_COACHING = {"min_cue_gap_seconds": 1.0, "set_timeout_seconds": 30.0, "cache_cues_before_set": True}

SQUAT_FAULT_TO_CUE_MAP = {
    "knee_valgus": "knees_out",
    "hip_shoot": "chest_up",
    "heel_rise": "heels_down",
    "balance": "whole_foot",
    "hip_shift": "even_it_out",
    "bilateral_asymmetry": "level_bar",
    "depth": "deeper",
    "foot_placement": "square_feet",
    "lockout": "lockout",
    "tempo": "slow_down",
    "depth_drift": "same_depth",
    "velocity_loss": "drive",
    "forward_lean": "chest_up",
    "back_rounding": "chest_up",
}
SQUAT_FAULT_CUE_PRIORITY = {
    "knee_valgus": 0,
    "hip_shoot": 1,
    "heel_rise": 2,
    "balance": 3,
    "hip_shift": 4,
    "bilateral_asymmetry": 5,
    "depth": 6,
    "foot_placement": 7,
    "lockout": 8,
    "tempo": 9,
    "depth_drift": 10,
    "velocity_loss": 11,
}
SQUAT_DEFAULT_FAULT_CUE_PRIORITY = 12

# The squat's cue dict, in the order the cache_cues payload carries it.
SQUAT_CUE_KEYS = (
    "knees_out", "knees_out_left", "knees_out_right",
    "chest_up",
    "heels_down", "heels_down_left", "heels_down_right",
    "whole_foot",
    "even_it_out", "even_it_out_left", "even_it_out_right",
    "level_bar",
    "deeper",
    "square_feet", "square_feet_left", "square_feet_right",
    "lockout", "slow_down", "same_depth", "drive", "brace",
    "stance_explain", "stance_wider", "stance_narrower",
    "toe_out_explain", "toe_out_more", "toe_out_less",
    "adjust_good",
    "good_rep", "great_depth", "strong", "clean", "perfect",
    "rep_1", "rep_2", "rep_3", "rep_4", "rep_5", "rep_6", "rep_7", "rep_8", "rep_9", "rep_10",
    "rep_11", "rep_12", "rep_13", "rep_14", "rep_15", "rep_16", "rep_17", "rep_18", "rep_19", "rep_20",
)

SQUAT_REGISTERED_NAMES = [
    "back_squat", "barbell_back_squat", "barbell_front_squat", "bodyweight_squat",
    "front_squat", "goblet_squat", "squat",
]
SQUAT_EXERCISE_NAMES = (
    "Barbell Back Squat", "Barbell Front Squat", "Back Squat", "Front Squat",
    "Goblet Squat", "Bodyweight Squat", "squat",
)


def _config_source_id(source: tuple[str, Callable[[], BiomechanicsConfig]]) -> str:
    return source[0]


def _squat_rules(config: BiomechanicsConfig) -> list[FaultRule]:
    return SquatProfile().create_fault_rules(config)


def _public_thresholds(rule: FaultRule) -> dict[str, Any]:
    thresholds: dict[str, Any] = {}
    for name, value in vars(rule).items():
        if name.startswith("_"):
            continue
        if isinstance(value, dict):
            thresholds.update({f"{name}.{tier}": tier_value for tier, tier_value in value.items()})
        else:
            thresholds[name] = value
    return thresholds


@pytest.mark.parametrize("config_source", CONFIG_SOURCES, ids=_config_source_id)
class TestSquatRules:
    def test_rules_in_cue_priority_order(self, config_source):
        rules = _squat_rules(config_source[1]())
        assert [(type(rule).__name__, rule.fault_type.value) for rule in rules] == list(SQUAT_RULES)

    @pytest.mark.parametrize("fault_type", list(SQUAT_RULE_THRESHOLDS))
    def test_rule_thresholds(self, config_source, fault_type: str):
        rule = next(rule for rule in _squat_rules(config_source[1]()) if rule.fault_type.value == fault_type)
        assert _public_thresholds(rule) == pytest.approx(SQUAT_RULE_THRESHOLDS[fault_type], abs=THRESHOLD_TOLERANCE)

    def test_depth_targets(self, config_source):
        depth = config_source[1]().faults.depth.model_dump()
        assert depth == pytest.approx(SQUAT_DEPTH_TARGETS, abs=THRESHOLD_TOLERANCE)

    def test_hip_position_counter(self, config_source):
        hip_counter = config_source[1]().hip_counter.model_dump()
        assert hip_counter == pytest.approx(SQUAT_HIP_COUNTER, abs=THRESHOLD_TOLERANCE)

    def test_ipc_and_coaching_timing(self, config_source):
        config = config_source[1]()
        assert config.ipc.model_dump() == pytest.approx(SQUAT_IPC, abs=THRESHOLD_TOLERANCE)
        assert config.coaching.model_dump() == pytest.approx(SQUAT_COACHING, abs=THRESHOLD_TOLERANCE)


class TestSquatCueMaps:
    def test_full_fault_to_cue_map(self):
        assert FAULT_TO_CUE_MAP == SQUAT_FAULT_TO_CUE_MAP
        assert list(FAULT_TO_CUE_MAP) == list(SQUAT_FAULT_TO_CUE_MAP)
        assert SquatProfile().get_fault_to_cue_map() == SQUAT_FAULT_TO_CUE_MAP

    def test_squat_cue_priority_ranks(self):
        # The squat's ranks are exactly as pinned; other exercises only append
        # their own prefixed fault types after the default (the deadlift's 20-30).
        assert {fault: FAULT_CUE_PRIORITY.get(fault) for fault in SQUAT_FAULT_CUE_PRIORITY} == SQUAT_FAULT_CUE_PRIORITY
        assert DEFAULT_FAULT_CUE_PRIORITY == SQUAT_DEFAULT_FAULT_CUE_PRIORITY
        additions = set(FAULT_CUE_PRIORITY) - set(SQUAT_FAULT_CUE_PRIORITY)
        assert all(fault.startswith("deadlift_") for fault in additions)
        assert all(FAULT_CUE_PRIORITY[fault] > DEFAULT_FAULT_CUE_PRIORITY for fault in additions)

    def test_rule_order_follows_cue_priority(self):
        assert [fault_type for _, fault_type in SQUAT_RULES] == sorted(
            SQUAT_FAULT_CUE_PRIORITY, key=SQUAT_FAULT_CUE_PRIORITY.get,
        )

    def test_cue_gap_and_praise_constants(self):
        assert PREEMPT_GAP_RATIO_PIPELINE == pytest.approx(0.0, abs=THRESHOLD_TOLERANCE)
        assert PREEMPT_GAP_RATIO_ORCHESTRATOR == pytest.approx(0.0, abs=THRESHOLD_TOLERANCE)
        assert SIDE_CUE_SIDES == ("left", "right")
        assert POSITIVE_CUE_KEYS == frozenset({"good_rep", "great_depth", "strong", "clean", "perfect"})
        assert GENERIC_POSITIVE_CUE_KEYS == ("good_rep", "strong", "clean", "perfect")

    def test_squat_cue_dict(self):
        cues = SquatProfile().get_cue_dict()
        assert list(cues) == list(SQUAT_CUE_KEYS)
        assert all(cues[key] == key for key in SQUAT_CUE_KEYS)


class TestSquatCacheCuesPayload:
    def test_cache_cues_payload_is_byte_identical(self, mock_ipc_client):
        """New cache_cues fields must be omitted at their defaults (PLAN §4.1)."""
        expected = {
            "type": "cache_cues",
            "exercise_name": SQUAT_EXERCISE,
            "profile": "squat",
            "cues": {key: key for key in SQUAT_CUE_KEYS},
        }
        returned_cues = IPCBridge(mock_ipc_client).prepare_exercise(SQUAT_EXERCISE)

        assert len(mock_ipc_client.messages) == 1
        assert json.dumps(mock_ipc_client.messages[0]) == json.dumps(expected)
        assert returned_cues == expected["cues"]


class TestSquatProfile:
    def test_profile_flags(self):
        profile = SquatProfile()
        assert profile.name == "squat"
        assert profile.movement_pattern == "squat"
        assert profile.uses_diagnosis_engine is True
        assert profile.uses_bilstm_counter is True
        assert profile.coaching_ready is True
        assert profile.display_name == "squats and squat variations (back, front, goblet, bodyweight)"

    def test_registered_names(self):
        assert sorted(name for name, cls in PROFILE_REGISTRY.items() if cls is SquatProfile) == SQUAT_REGISTERED_NAMES

    @pytest.mark.parametrize("exercise_name", SQUAT_EXERCISE_NAMES)
    def test_squat_names_resolve_to_the_squat_profile(self, exercise_name: str):
        assert type(get_profile(exercise_name)) is SquatProfile
