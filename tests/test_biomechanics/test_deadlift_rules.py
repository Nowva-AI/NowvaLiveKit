"""Deadlift fault rules D1-D10 on hand-built rep features (docs/deadlift/PLAN.md §2.6,
.claude/deadlift/CONTRACT.md §1-2): values, directions, sides, the effective min
tier, the wrist-proxy and body-vertical downgrades, and the drift channel."""

from __future__ import annotations

import math

import pytest

from biomechanics.config import BiomechanicsConfig
from biomechanics.deadlift.session_reference import DeadliftSessionReference
from biomechanics.deadlift.types import (
    BAR_SOURCE_WRIST_PROXY,
    GRAVITY_SOURCE_BODY,
    DeadliftRepFeatures,
)
from biomechanics.faults.rules.deadlift_bar_drift import DeadliftBarDriftRule
from biomechanics.faults.rules.deadlift_bar_position import DeadliftBarPositionRule
from biomechanics.faults.rules.deadlift_bar_tilt import DeadliftBarTiltRule
from biomechanics.faults.rules.deadlift_bent_arms import DeadliftBentArmsRule
from biomechanics.faults.rules.deadlift_hip_shift import DeadliftHipShiftRule
from biomechanics.faults.rules.deadlift_hips_shoot import DeadliftHipsShootRule
from biomechanics.faults.rules.deadlift_lean_back import DeadliftLeanBackRule
from biomechanics.faults.rules.deadlift_lockout import DeadliftLockoutRule
from biomechanics.faults.rules.deadlift_setup_hips import DeadliftSetupHipsRule
from biomechanics.faults.rules.deadlift_shoulders_behind import DeadliftShouldersBehindRule
from biomechanics.faults.rules.deadlift_velocity_loss import DeadliftVelocityLossRule
from biomechanics.utils.types import JointAngles

VALUE_TOLERANCE = 1e-6
FAULTS = BiomechanicsConfig().faults
PROXY_SCALE = FAULTS.deadlift_wrist_proxy_threshold_scale


def _features(**overrides) -> DeadliftRepFeatures:
    return DeadliftRepFeatures(rep_number=1, **overrides)


def _judge(rule, features: DeadliftRepFeatures, reference: DeadliftSessionReference | None = None):
    return rule.judge_rep(features, reference or DeadliftSessionReference(), JointAngles())


class TestSetupRules:
    def test_bar_ahead_of_midfoot_reads_forward(self):
        fault = _judge(DeadliftBarPositionRule(FAULTS.deadlift_bar_position), _features(bar_midfoot_setup_cm=6.0))
        assert fault.fault_type == "deadlift_bar_position"
        assert fault.severity.value == "moderate"
        assert fault.details["direction"] == "forward"
        assert fault.details["value"] == pytest.approx(6.0, abs=VALUE_TOLERANCE)
        assert fault.details["phase"] == "setup"
        assert fault.details["min_tier"] == "mild"

    def test_bar_behind_midfoot_reads_back(self):
        fault = _judge(DeadliftBarPositionRule(FAULTS.deadlift_bar_position), _features(bar_midfoot_setup_cm=-3.5))
        assert fault.details["direction"] == "back"
        assert fault.severity.value == "mild"

    def test_bar_over_midfoot_is_no_fault(self):
        assert _judge(DeadliftBarPositionRule(FAULTS.deadlift_bar_position), _features(bar_midfoot_setup_cm=1.0)) is None

    def test_wrist_proxy_widens_the_bar_thresholds_and_raises_the_min_tier(self):
        rule = DeadliftBarPositionRule(FAULTS.deadlift_bar_position, PROXY_SCALE)
        assert _judge(rule, _features(bar_midfoot_setup_cm=4.0, bar_source=BAR_SOURCE_WRIST_PROXY)) is None
        fault = _judge(rule, _features(bar_midfoot_setup_cm=8.0, bar_source=BAR_SOURCE_WRIST_PROXY))
        assert fault.severity.value == "moderate"
        assert fault.details["min_tier"] == "moderate"
        assert fault.details["bar_source"] == BAR_SOURCE_WRIST_PROXY

    @pytest.mark.parametrize(("height_cm", "direction"), [(50.0, "up"), (70.0, "down")])
    def test_hips_outside_the_band_name_which_way_to_move(self, height_cm: float, direction: str):
        features = _features(setup_hip_height_cm=height_cm, setup_hip_band_low_cm=56.0, setup_hip_band_high_cm=63.0)
        fault = _judge(DeadliftSetupHipsRule(FAULTS.deadlift_setup_hips), features)
        assert fault.details["direction"] == direction
        assert fault.details["value"] == pytest.approx(6.0 if direction == "up" else 7.0, abs=VALUE_TOLERANCE)
        assert fault.details["min_tier"] == "severe"

    def test_hips_inside_the_band_or_without_a_band_are_not_judged(self):
        rule = DeadliftSetupHipsRule(FAULTS.deadlift_setup_hips)
        inside = _features(setup_hip_height_cm=60.0, setup_hip_band_low_cm=56.0, setup_hip_band_high_cm=63.0)
        assert _judge(rule, inside) is None
        assert _judge(rule, _features(setup_hip_height_cm=40.0)) is None

    def test_shoulders_behind_the_bar_fire_and_ahead_do_not(self):
        rule = DeadliftShouldersBehindRule(FAULTS.deadlift_shoulders_behind)
        fault = _judge(rule, _features(shoulder_vs_bar_cm=-4.5))
        assert fault.severity.value == "moderate"
        assert fault.details["value"] == pytest.approx(4.5, abs=VALUE_TOLERANCE)
        assert _judge(rule, _features(shoulder_vs_bar_cm=3.0)) is None


class TestPullAndTopRules:
    def test_a_stall_resumes_only_from_a_hold_d6_would_judge(self):
        """The analyser resumes a pull from a hold at least D6's mild threshold short
        of standing: a lockout D6 calls clean is never a stall."""
        config = BiomechanicsConfig()
        mild_deg = FAULTS.deadlift_lockout.mild
        assert config.deadlift.resume_min_deficit_deg == pytest.approx(mild_deg, abs=VALUE_TOLERANCE)

    @pytest.mark.parametrize(
        ("rep_ratio", "set_ratio"), [(1.2, 1.2), (1.35, math.nan)], ids=["the_set_agrees", "beyond_noise_alone"],
    )
    def test_hips_shoot_fires_when_the_hips_decisively_led(self, rep_ratio: float, set_ratio: float):
        rule = DeadliftHipsShootRule(FAULTS.deadlift_hips_shoot)
        fault = _judge(rule, _features(trunk_change_liftoff_knee_deg=9.0, hip_shoulder_rise_ratio=rep_ratio,
                                       set_rise_ratio=set_ratio, trunk_change_predicted_deg=-16.0))
        assert fault.severity.value == "moderate"
        assert fault.details["rise_ratio"] == pytest.approx(rep_ratio, abs=VALUE_TOLERANCE)
        assert fault.details["model_predicted"] is True
        assert fault.details["min_tier"] == "moderate"

    @pytest.mark.parametrize(
        ("rep_ratio", "set_ratio"),
        [(0.95, 1.2), (1.05, 1.2), (1.2, 1.1), (1.2, math.nan), (math.nan, 1.2)],
        ids=["rep_chest_led", "rep_even", "set_not_decisive", "first_rep_not_decisive", "rep_unmeasured"],
    )
    def test_no_hips_shoot_unless_the_hips_decisively_led(self, rep_ratio: float, set_ratio: float):
        """A held back angle (ratio ~1) reads 6-12 deg against the setup model: the
        rise ratio, not the model, decides whether there is a fault."""
        rule = DeadliftHipsShootRule(FAULTS.deadlift_hips_shoot)
        assert _judge(rule, _features(trunk_change_liftoff_knee_deg=12.0, hip_shoulder_rise_ratio=rep_ratio,
                                      set_rise_ratio=set_ratio, trunk_change_predicted_deg=-16.0)) is None

    def test_body_vertical_raises_the_gravity_rules_to_moderate(self):
        for rule, field in (
            (DeadliftHipsShootRule(FAULTS.deadlift_hips_shoot), "trunk_change_liftoff_knee_deg"),
            (DeadliftBarDriftRule(FAULTS.deadlift_bar_drift), "bar_drift_cm"),
            (DeadliftLeanBackRule(FAULTS.deadlift_lean_back), "lean_back_deg"),
        ):
            fault = _judge(rule, _features(**{
                field: 30.0, "gravity_source": GRAVITY_SOURCE_BODY, "hip_shoulder_rise_ratio": 1.35,
            }))
            assert fault.details["min_tier"] == "moderate"
            assert fault.details["gravity_source"] == GRAVITY_SOURCE_BODY

    def test_bar_drift_is_judged_against_the_session_best(self):
        rule = DeadliftBarDriftRule(FAULTS.deadlift_bar_drift)
        reference = DeadliftSessionReference()
        reference.update(_features(bar_drift_cm=1.0))
        assert _judge(rule, _features(bar_drift_cm=6.0), reference).details["is_drift"] is True
        assert _judge(rule, _features(bar_drift_cm=6.0)).details["is_drift"] is False

    def test_lockout_names_the_worse_joint(self):
        rule = DeadliftLockoutRule(FAULTS.deadlift_lockout)
        fault = _judge(rule, _features(hip_extension_deficit_deg=13.0, knee_extension_deficit_deg=4.0))
        assert fault.details["joint"] == "hip"
        assert fault.details["phase"] == "top"
        assert _judge(rule, _features(hip_extension_deficit_deg=-3.0, knee_extension_deficit_deg=2.0)) is None

    def test_lean_back_past_standing_fires(self):
        fault = _judge(DeadliftLeanBackRule(FAULTS.deadlift_lean_back), _features(lean_back_deg=13.0))
        assert fault.severity.value == "moderate"

    def test_hip_shift_side_is_where_the_hips_went(self):
        rule = DeadliftHipShiftRule(FAULTS.deadlift_hip_shift)
        assert _judge(rule, _features(hip_shift_ratio=0.16)).details["side"] == "right"
        left = _judge(rule, _features(hip_shift_ratio=-0.16, grip="mixed"))
        assert left.details["side"] == "left"
        assert left.details["grip"] == "mixed"

    def test_bar_tilt_names_the_low_end(self):
        fault = _judge(DeadliftBarTiltRule(FAULTS.deadlift_bar_tilt), _features(bar_tilt_cm=5.5, bar_low_side="left"))
        assert fault.details["side"] == "left"
        assert fault.severity.value == "moderate"

    def test_bent_arms_start_cueing_at_mild(self):
        fault = _judge(DeadliftBentArmsRule(FAULTS.deadlift_bent_arms), _features(elbow_flexion_deg=18.0))
        assert fault.severity.value == "mild"
        assert fault.details["min_tier"] == "mild"

    def test_velocity_loss_is_recap_only(self):
        fault = _judge(DeadliftVelocityLossRule(FAULTS.deadlift_velocity_loss),
                       _features(velocity_loss_pct=32.0, concentric_velocity_mps=0.4))
        assert fault.details["min_tier"] == "recap"
        assert fault.details["velocity_mps"] == pytest.approx(0.4, abs=VALUE_TOLERANCE)

    def test_velocity_loss_is_not_emitted_on_the_wrist_proxy(self):
        rule = DeadliftVelocityLossRule(FAULTS.deadlift_velocity_loss)
        assert _judge(rule, _features(velocity_loss_pct=32.0, bar_source=BAR_SOURCE_WRIST_PROXY)) is None


class TestUnmeasured:
    @pytest.mark.parametrize(
        "rule",
        [
            DeadliftBarPositionRule(FAULTS.deadlift_bar_position),
            DeadliftSetupHipsRule(FAULTS.deadlift_setup_hips),
            DeadliftShouldersBehindRule(FAULTS.deadlift_shoulders_behind),
            DeadliftHipsShootRule(FAULTS.deadlift_hips_shoot),
            DeadliftBarDriftRule(FAULTS.deadlift_bar_drift),
            DeadliftLockoutRule(FAULTS.deadlift_lockout),
            DeadliftLeanBackRule(FAULTS.deadlift_lean_back),
            DeadliftHipShiftRule(FAULTS.deadlift_hip_shift),
            DeadliftBarTiltRule(FAULTS.deadlift_bar_tilt),
            DeadliftBentArmsRule(FAULTS.deadlift_bent_arms),
            DeadliftVelocityLossRule(FAULTS.deadlift_velocity_loss),
        ],
        ids=lambda rule: type(rule).__name__,
    )
    def test_a_rep_that_measured_nothing_gets_no_verdict(self, rule):
        assert _judge(rule, _features()) is None

    def test_squat_features_are_never_judged_by_a_deadlift_rule(self):
        from biomechanics.analysis.rep_features import RepFeatures

        rule = DeadliftBarDriftRule(FAULTS.deadlift_bar_drift)
        assert rule.judge_rep(RepFeatures(rep_number=1), DeadliftSessionReference(), JointAngles()) is None


class TestSessionReference:
    def test_velocity_loss_is_against_the_sets_two_fastest_reps(self):
        reference = DeadliftSessionReference()
        for velocity in (0.60, 0.50):
            reference.update(_features(concentric_velocity_mps=velocity))
        slow = _features(concentric_velocity_mps=0.33)
        reference.annotate(slow)
        assert slow.velocity_loss_pct == pytest.approx(40.0, abs=1e-6)

    def test_bests_keep_the_lowest_drift_and_survive_a_new_set(self):
        reference = DeadliftSessionReference()
        reference.update(_features(bar_drift_cm=4.0, hip_shift_ratio=-0.2, trunk_change_liftoff_knee_deg=3.0))
        reference.update(_features(bar_drift_cm=2.0))
        reference.start_new_set()
        assert reference.best("bar_drift_cm") == pytest.approx(2.0, abs=VALUE_TOLERANCE)
        assert reference.best("hip_shift_abs") == pytest.approx(0.2, abs=VALUE_TOLERANCE)
        assert math.isnan(reference.velocity_reference_mps)

    def test_the_depth_target_api_is_intact(self):
        reference = DeadliftSessionReference()
        assert reference.depth_target_ratio is None
        assert reference.reaches_depth_target(0.5)
