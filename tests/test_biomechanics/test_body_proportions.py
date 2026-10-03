"""Tests for fault threshold scaling from body proportions (C8)."""

import pytest

from biomechanics.faults.rule_engine import RuleEngine
from biomechanics.faults.fault_types import FaultType
from biomechanics.utils.segment_lengths import BodyProportions as SegmentBodyProportions

LONG_FEMUR_LEAN_SCALE = 1.22
UPRIGHT_TRUNK_DEG = 180.0
SCALING_REPEATS = 4


def _segment_proportions(forward_lean_scale: float) -> SegmentBodyProportions:
    return SegmentBodyProportions(
        hip_width=0.24,
        femur_length_avg=0.46,
        tibia_length_avg=0.44,
        torso_length_avg=0.50,
        shoulder_width=0.38,
        foot_length_avg=0.26,
        hip_to_femur_ratio=0.52,
        tibia_to_reference_ratio=1.0,
        forward_lean_scale=forward_lean_scale,
    )


def _make_engine_with_squat_rules() -> RuleEngine:
    """Build an engine with the squat-profile rule set (config-driven defaults).

    The proportion-scaling assertions below depend on those config defaults,
    not on the bare-class defaults — the profile is what production uses.
    """
    from biomechanics.config import BiomechanicsConfig
    from biomechanics.profiles.squat import SquatProfile

    profile = SquatProfile()
    rules = profile.create_fault_rules(BiomechanicsConfig())
    return RuleEngine(rules=rules)


def _make_engine_with_forward_lean() -> RuleEngine:
    """An engine holding ForwardLeanRule at the config thresholds.

    The squat no longer registers forward lean (it judges hip shoot instead),
    but the lunge does, so its proportion scaling still has to hold.
    """
    from biomechanics.config import BiomechanicsConfig
    from biomechanics.faults.rules.forward_lean import ForwardLeanRule

    fl = BiomechanicsConfig().faults.forward_lean
    rule = ForwardLeanRule(
        mild_threshold=fl.mild, moderate_threshold=fl.moderate, severe_threshold=fl.severe,
    )
    return RuleEngine(rules=[rule])


class TestRuleEngineProportionScaling:
    """C8: scaling is idempotent (from base thresholds), valgus is not scaled,
    forward lean scales in lean space so a larger scale is MORE lenient."""

    def test_valgus_thresholds_not_scaled(self):
        engine = _make_engine_with_squat_rules()
        valgus_rule = engine.get_rule(FaultType.KNEE_VALGUS)
        before = (valgus_rule.mild_threshold, valgus_rule.moderate_threshold, valgus_rule.severe_threshold)

        engine.apply_body_proportion_scaling(_segment_proportions(LONG_FEMUR_LEAN_SCALE))

        assert (valgus_rule.mild_threshold, valgus_rule.moderate_threshold, valgus_rule.severe_threshold) == before

    def test_forward_lean_thresholds_scaled_in_lean_space(self):
        engine = _make_engine_with_forward_lean()
        fwd_rule = engine.get_rule(FaultType.FORWARD_LEAN)
        base = (fwd_rule.mild_threshold, fwd_rule.moderate_threshold, fwd_rule.severe_threshold)

        engine.apply_body_proportion_scaling(_segment_proportions(1.2))

        for threshold, base_threshold in zip(
            (fwd_rule.mild_threshold, fwd_rule.moderate_threshold, fwd_rule.severe_threshold), base,
        ):
            expected = UPRIGHT_TRUNK_DEG - (UPRIGHT_TRUNK_DEG - base_threshold) * 1.2
            assert threshold == pytest.approx(expected, abs=0.01)

    def test_scaling_is_idempotent(self):
        engine = _make_engine_with_forward_lean()
        fwd_rule = engine.get_rule(FaultType.FORWARD_LEAN)

        engine.apply_body_proportion_scaling(_segment_proportions(LONG_FEMUR_LEAN_SCALE))
        once = (fwd_rule.mild_threshold, fwd_rule.moderate_threshold, fwd_rule.severe_threshold)
        for _ in range(SCALING_REPEATS - 1):
            engine.apply_body_proportion_scaling(_segment_proportions(LONG_FEMUR_LEAN_SCALE))

        assert (fwd_rule.mild_threshold, fwd_rule.moderate_threshold, fwd_rule.severe_threshold) == pytest.approx(once)

    def test_rescaling_with_a_new_scale_starts_from_base(self):
        engine = _make_engine_with_forward_lean()
        fwd_rule = engine.get_rule(FaultType.FORWARD_LEAN)
        base_mild = fwd_rule.mild_threshold

        engine.apply_body_proportion_scaling(_segment_proportions(LONG_FEMUR_LEAN_SCALE))
        engine.apply_body_proportion_scaling(_segment_proportions(1.0))

        assert fwd_rule.mild_threshold == pytest.approx(base_mild)

    def test_long_femurs_make_forward_lean_more_lenient(self):
        from collections import deque
        from biomechanics.utils.types import JointAngles

        engine = _make_engine_with_forward_lean()
        fwd_rule = engine.get_rule(FaultType.FORWARD_LEAN)
        base_mild = fwd_rule.mild_threshold
        # A lean just past the base mild threshold: faults at base scale ...
        trunk_flexion = base_mild - 2.0
        assert fwd_rule.evaluate(
            JointAngles(trunk_flexion=trunk_flexion, timestamp=0.0), deque(), in_rep=True,
        ) is not None

        engine.apply_body_proportion_scaling(_segment_proportions(LONG_FEMUR_LEAN_SCALE))

        # ... but not for a long-femur lifter, whose thresholds moved DOWN (more lean allowed)
        assert fwd_rule.mild_threshold < base_mild
        assert fwd_rule.evaluate(
            JointAngles(trunk_flexion=trunk_flexion, timestamp=100.0), deque(), in_rep=True,
        ) is None


class TestSquatRuleSet:

    def test_squat_judges_hip_shoot_not_absolute_lean(self):
        """Absolute lean is bar position and anatomy; the coachable fault is the chest dropping."""
        engine = _make_engine_with_squat_rules()
        assert engine.get_rule(FaultType.FORWARD_LEAN) is None
        assert engine.get_rule(FaultType.HIP_SHOOT) is not None

    def test_squat_registers_every_contract_fault(self):
        engine = _make_engine_with_squat_rules()
        registered = {rule.fault_type for rule in engine.rules}
        assert registered == {
            FaultType.KNEE_VALGUS, FaultType.HIP_SHOOT, FaultType.HEEL_RISE, FaultType.BALANCE,
            FaultType.HIP_SHIFT, FaultType.BILATERAL_ASYMMETRY, FaultType.DEPTH,
            FaultType.FOOT_PLACEMENT, FaultType.LOCKOUT, FaultType.TEMPO, FaultType.DEPTH_DRIFT,
            FaultType.VELOCITY_LOSS,
        }


class TestForwardLeanBaselineSurvivesScaling:
    def test_baseline_tightening_is_kept_after_proportion_scaling(self):
        from biomechanics.faults.rules.forward_lean import ForwardLeanRule
        from biomechanics.utils.segment_lengths import BodyProportions

        rule = ForwardLeanRule(mild_threshold=145.0, moderate_threshold=135.0, severe_threshold=125.0)
        rule.apply_baseline(130.0)
        assert rule.mild_threshold == pytest.approx(120.0)
        assert rule.moderate_threshold == pytest.approx(115.0)
        assert rule.severe_threshold == pytest.approx(110.0)

        proportions = BodyProportions(
            hip_width=0.25, femur_length_avg=0.45, tibia_length_avg=0.45, torso_length_avg=0.50,
            shoulder_width=0.40, foot_length_avg=0.26, hip_to_femur_ratio=0.556,
            tibia_to_reference_ratio=1.0, forward_lean_scale=1.0,
        )
        rule.scale_for_proportions(proportions)
        rule.scale_for_proportions(proportions)
        assert rule.mild_threshold == pytest.approx(120.0)
        assert rule.moderate_threshold == pytest.approx(115.0)
        assert rule.severe_threshold == pytest.approx(110.0)
