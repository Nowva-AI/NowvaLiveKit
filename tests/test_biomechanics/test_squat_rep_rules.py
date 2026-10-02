"""Tests for the squat's per-rep fault rules, session reference, observability and engine path."""

from __future__ import annotations

import math

import pytest

from biomechanics.analysis.rep_features import RepFeatures
from biomechanics.config import BiomechanicsConfig
from biomechanics.faults.fault_types import FaultType
from biomechanics.faults.observability import (
    APPROXIMATE,
    NOT_OBSERVABLE,
    OBSERVABLE,
    SINGLE_CAMERA,
    TRIANGULATED,
    fault_observability,
)
from biomechanics.faults.rule_engine import RuleEngine
from biomechanics.faults.rules.balance import BalanceRule
from biomechanics.faults.rules.depth_drift import DepthDriftRule
from biomechanics.faults.rules.descent_control import DescentControlRule
from biomechanics.faults.rules.foot_placement import FootPlacementRule
from biomechanics.faults.rules.hip_shift import HipShiftRule
from biomechanics.faults.rules.hip_shoot import HipShootRule
from biomechanics.faults.rules.knee_tracking import KneeTrackingRule
from biomechanics.faults.rules.standing_lockout import StandingLockoutRule
from biomechanics.faults.rules.velocity_loss import VelocityLossRule
from biomechanics.faults.session_reference import SessionReference
from biomechanics.profiles.squat import SquatProfile
from biomechanics.utils.types import FaultSeverity, JointAngles

CONTRACT_DETAIL_KEYS = {"side", "phase", "is_drift", "value", "unit"}


def _features(**overrides: float) -> RepFeatures:
    return RepFeatures(rep_number=overrides.pop("rep_number", 1), **overrides)


def _angles(timestamp: float = 0.0) -> JointAngles:
    return JointAngles(timestamp=timestamp)


def _squat_engine(capture_mode: str) -> RuleEngine:
    rules = SquatProfile().create_fault_rules(BiomechanicsConfig())
    return RuleEngine(rules=rules, capture_mode=capture_mode)


class TestKneeTrackingRule:
    def test_knees_over_toes_is_clean(self):
        assert KneeTrackingRule().judge_rep(_features(valgus_l=2.0, valgus_r=3.0), SessionReference(), _angles()) is None

    def test_cave_on_the_way_up_is_reported_with_side_and_phase(self):
        fault = KneeTrackingRule().judge_rep(
            _features(valgus_l=3.0, valgus_r=14.0, valgus_peak_phase_r="ascent"), SessionReference(), _angles(),
        )
        assert fault.severity == FaultSeverity.MODERATE
        assert fault.details["side"] == "right"
        assert fault.details["phase"] == "ascent"
        assert CONTRACT_DETAIL_KEYS <= fault.details.keys()

    def test_both_knees_caving_together_reads_both(self):
        fault = KneeTrackingRule().judge_rep(_features(valgus_l=10.0, valgus_r=11.0), SessionReference(), _angles())
        assert fault.details["side"] == "both"

    def test_knees_out_is_never_a_fault(self):
        assert KneeTrackingRule().judge_rep(_features(valgus_l=-20.0, valgus_r=-20.0), SessionReference(), _angles()) is None

    def test_unmeasured_knees_say_nothing(self):
        assert KneeTrackingRule().judge_rep(_features(), SessionReference(), _angles()) is None

    def test_drift_when_the_best_rep_was_clean(self):
        reference = SessionReference()
        reference.update(_features(valgus_l=2.0, valgus_r=2.0))
        fault = KneeTrackingRule().judge_rep(_features(valgus_l=10.0, valgus_r=2.0), reference, _angles())
        assert fault.details["is_drift"] is True

    def test_no_drift_when_it_caved_from_the_first_rep(self):
        reference = SessionReference()
        reference.update(_features(valgus_l=10.0, valgus_r=10.0))
        fault = KneeTrackingRule().judge_rep(_features(valgus_l=10.0, valgus_r=10.0), reference, _angles())
        assert fault.details["is_drift"] is False


class TestHipShootRule:
    def test_chest_dropping_fires(self):
        fault = HipShootRule().judge_rep(_features(hip_shoot_deg=13.0), SessionReference(), _angles())
        assert fault.severity == FaultSeverity.MODERATE
        assert fault.details["phase"] == "ascent"

    def test_chest_rising_with_the_hips_is_clean(self):
        assert HipShootRule().judge_rep(_features(hip_shoot_deg=-4.0), SessionReference(), _angles()) is None


class TestHipShiftRule:
    def test_side_is_the_direction_the_hips_moved(self):
        fault = HipShiftRule().judge_rep(_features(hip_shift_ratio=-0.17), SessionReference(), _angles())
        assert fault.details["side"] == "left"
        assert fault.severity == FaultSeverity.MODERATE

    def test_shift_at_the_single_camera_noise_floor_is_clean(self):
        """Recorded single-camera runs read a median 0.066 shift with no labelled fault."""
        assert HipShiftRule().judge_rep(_features(hip_shift_ratio=0.07), SessionReference(), _angles()) is None

    def test_small_shift_is_clean(self):
        assert HipShiftRule().judge_rep(_features(hip_shift_ratio=0.03), SessionReference(), _angles()) is None


class TestBalanceRule:
    def test_weight_on_the_toes_fires_forward(self):
        fault = BalanceRule().judge_rep(_features(balance_ratio=0.32), SessionReference(), _angles())
        assert fault.details["direction"] == "forward"
        assert fault.severity == FaultSeverity.MODERATE

    def test_heels_get_a_wider_band(self):
        rule = BalanceRule()
        assert rule.judge_rep(_features(balance_ratio=-0.25), SessionReference(), _angles()) is None
        assert rule.judge_rep(_features(balance_ratio=-0.35), SessionReference(), _angles()).details["direction"] == "backward"

    def test_load_source_is_reported(self):
        fault = BalanceRule().judge_rep(_features(balance_ratio=0.3, bar_detected=True), SessionReference(), _angles())
        assert fault.details["load"] == "bar"


class TestDepthDriftRule:
    def test_shallower_than_the_owned_depth_fires(self):
        reference = SessionReference()
        reference.update(_features(depth_ratio=-0.05))
        fault = DepthDriftRule().judge_rep(_features(depth_ratio=0.12), reference, _angles())
        assert fault is not None
        assert fault.details["is_drift"] is True

    def test_one_unusually_deep_rep_does_not_set_the_reference(self):
        reference = SessionReference()
        for depth in (-0.30, 0.0, 0.0, 0.01):
            reference.update(_features(depth_ratio=depth))
        assert reference.depth_reference() == pytest.approx(0.0)
        assert DepthDriftRule().judge_rep(_features(depth_ratio=0.05), reference, _angles()) is None

    def test_a_missed_target_is_the_depth_rules_verdict_not_drift(self):
        reference = SessionReference()
        reference.depth_target_ratio = 0.0
        reference.update(_features(depth_ratio=-0.05))
        assert DepthDriftRule().judge_rep(_features(depth_ratio=0.3), reference, _angles()) is None

    def test_no_best_rep_yet_is_silent(self):
        assert DepthDriftRule().judge_rep(_features(depth_ratio=0.3), SessionReference(), _angles()) is None


class TestStandingLockoutRule:
    def test_not_standing_tall_fires_at_the_top(self):
        fault = StandingLockoutRule().judge_rep(_features(lockout_deficit_ratio=0.09), SessionReference(), _angles())
        assert fault.severity == FaultSeverity.MODERATE
        assert fault.details["phase"] == "top"

    def test_full_lockout_is_clean(self):
        assert StandingLockoutRule().judge_rep(_features(lockout_deficit_ratio=0.01), SessionReference(), _angles()) is None


class TestDescentControlRule:
    def test_dive_bomb_fires(self):
        fault = DescentControlRule().judge_rep(_features(descent_time_s=0.25), SessionReference(), _angles())
        assert fault.severity == FaultSeverity.SEVERE
        assert fault.details["kind"] == "fast_descent"

    def test_controlled_descent_is_clean(self):
        assert DescentControlRule().judge_rep(_features(descent_time_s=1.2), SessionReference(), _angles()) is None


class TestVelocityLossRule:
    def test_loss_against_the_set_best_fires(self):
        fault = VelocityLossRule().judge_rep(_features(velocity_loss_pct=33.0), SessionReference(), _angles())
        assert fault.severity == FaultSeverity.MODERATE

    def test_first_rep_has_no_loss(self):
        assert VelocityLossRule().judge_rep(_features(), SessionReference(), _angles()) is None


class TestFootPlacementRule:
    def test_staggered_foot_names_the_foot_to_move(self):
        fault = FootPlacementRule().judge_rep(_features(stagger_ratio=0.3), SessionReference(), _angles())
        assert fault.details["kind"] == "stagger"
        assert fault.details["side"] == "left"

    def test_uneven_flare_names_the_foot_turned_out_more(self):
        fault = FootPlacementRule().judge_rep(
            _features(stagger_ratio=0.0, toe_out_l_deg=10.0, toe_out_r_deg=28.0), SessionReference(), _angles(),
        )
        assert fault.details["kind"] == "flare"
        assert fault.details["side"] == "right"

    def test_square_feet_are_clean(self):
        assert FootPlacementRule().judge_rep(
            _features(stagger_ratio=0.02, toe_out_l_deg=20.0, toe_out_r_deg=22.0), SessionReference(), _angles(),
        ) is None


class TestSessionReference:
    def test_velocity_loss_is_against_the_fastest_earlier_reps(self):
        reference = SessionReference()
        reference.update(_features(concentric_velocity_mps=0.6))
        slow = _features(concentric_velocity_mps=0.45)
        reference.annotate(slow)
        assert slow.velocity_loss_pct == pytest.approx(25.0)

    def test_one_fast_frame_timed_rep_does_not_inflate_the_reference(self):
        reference = SessionReference()
        for velocity in (0.70, 0.50, 0.50):
            reference.update(_features(concentric_velocity_mps=velocity))
        assert reference.velocity_reference_mps == pytest.approx(0.60)

    def test_velocity_best_resets_each_set(self):
        reference = SessionReference()
        reference.update(_features(concentric_velocity_mps=0.6))
        reference.start_new_set()
        assert math.isnan(reference.set_best_velocity_mps)

    def test_position_bests_survive_a_new_set(self):
        reference = SessionReference()
        reference.update(_features(valgus_l=2.0, valgus_r=1.0))
        reference.start_new_set()
        assert reference.best("valgus") == pytest.approx(2.0)

    def test_no_target_counts_every_descent(self):
        reference = SessionReference()
        assert reference.reaches_depth_target(0.9)
        reference.depth_target_ratio = 0.0
        assert not reference.reaches_depth_target(0.2)


class TestObservability:
    def test_single_camera_sees_knee_tracking_and_vertical_faults(self):
        assert fault_observability("knee_valgus", SINGLE_CAMERA) == OBSERVABLE
        assert fault_observability("depth", SINGLE_CAMERA) == OBSERVABLE
        assert fault_observability("lockout", SINGLE_CAMERA) == OBSERVABLE

    def test_single_camera_hip_shift_is_approximate(self):
        """Backward hip travel leaks into sideways travel through monocular depth error."""
        assert fault_observability("hip_shift", SINGLE_CAMERA) == APPROXIMATE
        assert fault_observability("hip_shift", TRIANGULATED) == OBSERVABLE

    @pytest.mark.parametrize("fault_type", ["hip_shoot", "balance", "velocity_loss"])
    def test_side_view_faults_are_rig_only(self, fault_type: str):
        """Product decision: on one camera the coach says nothing about side-view faults."""
        assert fault_observability(fault_type, SINGLE_CAMERA) == NOT_OBSERVABLE
        assert fault_observability(fault_type, TRIANGULATED) == OBSERVABLE

    def test_heel_rise_needs_the_rig(self):
        assert fault_observability("heel_rise", SINGLE_CAMERA) == NOT_OBSERVABLE
        assert fault_observability("heel_rise", TRIANGULATED) == OBSERVABLE

    def test_other_exercises_faults_count_as_observable(self):
        assert fault_observability("elbow_flare", SINGLE_CAMERA) == OBSERVABLE


class TestRuleEngineRepPath:
    def test_rep_faults_are_tagged_with_observability(self):
        engine = _squat_engine(SINGLE_CAMERA)
        faults = engine.finish_rep(_angles(), 1, _features(valgus_l=14.0, valgus_r=2.0, hip_shift_ratio=0.2))
        tags = {fault.fault_type: fault.details["observability"] for fault in faults}
        assert tags[FaultType.KNEE_VALGUS] == OBSERVABLE
        assert tags[FaultType.HIP_SHIFT] == APPROXIMATE

    def test_single_camera_never_emits_side_view_faults(self):
        engine = _squat_engine(SINGLE_CAMERA)
        engine.finish_rep(_angles(0.0), 1, _features(concentric_velocity_mps=0.6))
        faults = engine.finish_rep(
            _angles(5.0), 2,
            _features(hip_shoot_deg=20.0, balance_ratio=0.5, concentric_velocity_mps=0.3),
        )
        emitted = {fault.fault_type for fault in faults}
        assert not emitted & {FaultType.HIP_SHOOT, FaultType.BALANCE, FaultType.VELOCITY_LOSS}
        triangulated = _squat_engine(TRIANGULATED)
        triangulated.finish_rep(_angles(0.0), 1, _features(concentric_velocity_mps=0.6))
        rig_faults = triangulated.finish_rep(
            _angles(5.0), 2,
            _features(hip_shoot_deg=20.0, balance_ratio=0.5, concentric_velocity_mps=0.3),
        )
        assert {FaultType.HIP_SHOOT, FaultType.BALANCE, FaultType.VELOCITY_LOSS} <= {
            fault.fault_type for fault in rig_faults
        }

    def test_reference_updates_after_judging(self):
        """A rep is judged against the bests as they stood before it."""
        engine = _squat_engine(TRIANGULATED)
        engine.finish_rep(_angles(0.0), 1, _features(valgus_l=12.0, valgus_r=12.0))
        faults = engine.finish_rep(_angles(5.0), 2, _features(valgus_l=12.0, valgus_r=12.0))
        knee = next(fault for fault in faults if fault.fault_type == FaultType.KNEE_VALGUS)
        assert knee.details["is_drift"] is False

    def test_velocity_loss_is_filled_on_the_rep(self):
        engine = _squat_engine(TRIANGULATED)
        engine.finish_rep(_angles(0.0), 1, _features(concentric_velocity_mps=0.6))
        slow = _features(concentric_velocity_mps=0.39)
        faults = engine.finish_rep(_angles(5.0), 2, slow)
        assert slow.velocity_loss_pct == pytest.approx(35.0)
        assert FaultType.VELOCITY_LOSS in {fault.fault_type for fault in faults}

    def test_no_features_means_no_rep_verdicts(self):
        engine = _squat_engine(TRIANGULATED)
        assert engine.finish_rep(_angles(), 1, None) == []
