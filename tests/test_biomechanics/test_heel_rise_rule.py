"""Tests for the heel-rise fault rule and its registration in the squat profile."""

from __future__ import annotations

from collections import deque

import pytest

from biomechanics.config import BiomechanicsConfig
from biomechanics.faults.fault_types import FaultType
from biomechanics.faults.rules.heel_rise import HeelRiseRule
from biomechanics.profiles.squat import SquatProfile
from biomechanics.utils.foot_contact import FootState
from biomechanics.utils.types import FaultEvent, FaultSeverity, JointAngles

FRAME_INTERVAL_S = 1.0 / 30.0
PERSISTENCE_FRAMES = 8


def _foot_state(rise_l_cm: float = 0.0, rise_r_cm: float = 0.0, valid: bool = True) -> FootState:
    return FootState(
        valid=valid, planted_l=True, planted_r=True,
        heel_rise_l_cm=rise_l_cm, heel_rise_r_cm=rise_r_cm,
        stance_width_cm=38.0, toe_out_l_deg=15.0, toe_out_r_deg=15.0,
        hip_height_above_floor_cm=90.0, lateral_hip_offset_cm=0.0, floor_roll_deg=0.0,
    )


def _evaluate(
    rule: HeelRiseRule, foot_state: FootState | None, timestamp: float, frame_index: int = 0,
    in_rep: bool = True,
) -> FaultEvent | None:
    rule.set_frame_context(foot_state=foot_state)
    angles = JointAngles(timestamp=timestamp, frame_index=frame_index)
    return rule.evaluate(angles, deque(), in_rep=in_rep, rep_number=1)


def _evaluate_persisted(
    rule: HeelRiseRule, foot_state: FootState | None, start_s: float = 0.0, in_rep: bool = True,
) -> FaultEvent | None:
    fault = None
    for frame in range(PERSISTENCE_FRAMES):
        fault = _evaluate(rule, foot_state, start_s + frame * FRAME_INTERVAL_S, frame, in_rep=in_rep)
    return fault


class TestHeelRiseRuleGating:

    def test_no_foot_state_is_noop(self) -> None:
        assert _evaluate_persisted(HeelRiseRule(), None) is None

    def test_invalid_foot_state_is_noop(self) -> None:
        assert _evaluate_persisted(HeelRiseRule(), _foot_state(rise_r_cm=6.0, valid=False)) is None

    def test_outside_rep_is_noop(self) -> None:
        assert _evaluate_persisted(HeelRiseRule(), _foot_state(rise_r_cm=6.0), in_rep=False) is None

    def test_below_mild_threshold_is_noop(self) -> None:
        assert _evaluate_persisted(HeelRiseRule(), _foot_state(rise_r_cm=1.4)) is None

    def test_nan_rise_is_treated_as_flat(self) -> None:
        assert _evaluate_persisted(HeelRiseRule(), _foot_state(rise_l_cm=float("nan"), rise_r_cm=1.0)) is None

    def test_single_frame_excursion_does_not_fire(self) -> None:
        rule = HeelRiseRule()
        assert _evaluate(rule, _foot_state(rise_r_cm=6.0), 0.0) is None
        assert _evaluate(rule, _foot_state(rise_r_cm=0.0), FRAME_INTERVAL_S) is None
        assert _evaluate(rule, _foot_state(rise_r_cm=6.0), 2 * FRAME_INTERVAL_S) is None

    def test_fires_after_persistence(self) -> None:
        fault = _evaluate_persisted(HeelRiseRule(), _foot_state(rise_r_cm=6.0))
        assert fault is not None
        assert fault.fault_type == FaultType.HEEL_RISE.value


class TestHeelRiseRuleSeverity:

    def test_mild_tier(self) -> None:
        fault = _evaluate_persisted(HeelRiseRule(), _foot_state(rise_r_cm=2.0))
        assert fault is not None
        assert fault.severity == FaultSeverity.MILD
        assert fault.details["affected_side"] == "right"
        assert fault.details["max_heel_rise_cm"] == pytest.approx(2.0)

    def test_moderate_tier(self) -> None:
        fault = _evaluate_persisted(HeelRiseRule(), _foot_state(rise_l_cm=3.5))
        assert fault is not None
        assert fault.severity == FaultSeverity.MODERATE
        assert fault.details["affected_side"] == "left"

    def test_severe_tier(self) -> None:
        fault = _evaluate_persisted(HeelRiseRule(), _foot_state(rise_l_cm=5.5, rise_r_cm=5.2))
        assert fault is not None
        assert fault.severity == FaultSeverity.SEVERE
        assert fault.details["affected_side"] == "both"
        assert fault.severity_score >= 2.5

    def test_custom_thresholds(self) -> None:
        rule = HeelRiseRule(mild_cm=3.0, moderate_cm=4.0, severe_cm=5.0)
        assert _evaluate_persisted(rule, _foot_state(rise_r_cm=2.5)) is None
        fault = _evaluate_persisted(rule, _foot_state(rise_r_cm=4.5))
        assert fault is not None
        assert fault.severity == FaultSeverity.MODERATE

    def test_message_and_event_fields(self) -> None:
        fault = _evaluate_persisted(HeelRiseRule(), _foot_state(rise_r_cm=3.5), start_s=12.0)
        assert fault is not None
        assert "heel" in fault.message.lower()
        assert fault.timestamp == pytest.approx(12.0 + (PERSISTENCE_FRAMES - 1) * FRAME_INTERVAL_S)
        assert fault.rep_number == 1
        assert fault.details["heel_rise_l_cm"] == pytest.approx(0.0)
        assert fault.details["heel_rise_r_cm"] == pytest.approx(3.5)


class TestHeelRiseRuleCooldown:

    def test_cooldown_is_time_based_not_frame_based(self) -> None:
        rule = HeelRiseRule(cooldown_s=2.0)
        rising = _foot_state(rise_r_cm=4.0)
        assert _evaluate_persisted(rule, rising, start_s=0.0) is not None
        # Many frames but little time: still cooling down.
        assert _evaluate(rule, rising, 1.0, frame_index=5000) is None
        # Few frames but enough time: fires again.
        assert _evaluate(rule, rising, 2.5, frame_index=5001) is not None

    def test_reset_clears_cooldown(self) -> None:
        rule = HeelRiseRule()
        rising = _foot_state(rise_r_cm=4.0)
        assert _evaluate_persisted(rule, rising, start_s=0.0) is not None
        assert _evaluate(rule, rising, 0.5) is None
        rule.reset()
        assert _evaluate_persisted(rule, rising, start_s=0.5) is not None


class TestSquatProfileRegistration:

    def test_squat_profile_includes_heel_rise_rule(self) -> None:
        rules = SquatProfile().create_fault_rules(BiomechanicsConfig())
        heel_rules = [rule for rule in rules if isinstance(rule, HeelRiseRule)]
        assert len(heel_rules) == 1
        assert heel_rules[0].fault_type == FaultType.HEEL_RISE
