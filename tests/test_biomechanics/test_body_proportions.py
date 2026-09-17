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
        engine = _make_engine_with_squat_rules()
        fwd_rule = engine.get_rule(FaultType.FORWARD_LEAN)
        base = (fwd_rule.mild_threshold, fwd_rule.moderate_threshold, fwd_rule.severe_threshold)

        engine.apply_body_proportion_scaling(_segment_proportions(1.2))

        for threshold, base_threshold in zip(
            (fwd_rule.mild_threshold, fwd_rule.moderate_threshold, fwd_rule.severe_threshold), base,
        ):
            expected = UPRIGHT_TRUNK_DEG - (UPRIGHT_TRUNK_DEG - base_threshold) * 1.2
            assert threshold == pytest.approx(expected, abs=0.01)

    def test_scaling_is_idempotent(self):
        engine = _make_engine_with_squat_rules()
        fwd_rule = engine.get_rule(FaultType.FORWARD_LEAN)

        engine.apply_body_proportion_scaling(_segment_proportions(LONG_FEMUR_LEAN_SCALE))
        once = (fwd_rule.mild_threshold, fwd_rule.moderate_threshold, fwd_rule.severe_threshold)
        for _ in range(SCALING_REPEATS - 1):
            engine.apply_body_proportion_scaling(_segment_proportions(LONG_FEMUR_LEAN_SCALE))

        assert (fwd_rule.mild_threshold, fwd_rule.moderate_threshold, fwd_rule.severe_threshold) == pytest.approx(once)

    def test_rescaling_with_a_new_scale_starts_from_base(self):
        engine = _make_engine_with_squat_rules()
        fwd_rule = engine.get_rule(FaultType.FORWARD_LEAN)
        base_mild = fwd_rule.mild_threshold

        engine.apply_body_proportion_scaling(_segment_proportions(LONG_FEMUR_LEAN_SCALE))
        engine.apply_body_proportion_scaling(_segment_proportions(1.0))

        assert fwd_rule.mild_threshold == pytest.approx(base_mild)

    def test_long_femurs_make_forward_lean_more_lenient(self):
        from collections import deque
        from biomechanics.utils.types import JointAngles

        engine = _make_engine_with_squat_rules()
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


class TestForwardLeanRuleEnabled:

    def test_forward_lean_rule_in_engine(self):
        """ForwardLeanRule should be active in the squat-profile rule set."""
        engine = _make_engine_with_squat_rules()
        fwd_rule = engine.get_rule(FaultType.FORWARD_LEAN)
        assert fwd_rule is not None
        # Squat profile registers depth, symmetry, bar-tilt asymmetry,
        # forward lean, knee valgus, heel rise = 6 rules.
        assert engine.rule_count == 6


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
