"""
Tests for Fault Detection Rules

Tests each fault rule with known angle values.
Verifies severity levels, thresholds, and fault messages.
"""

import math
import pytest
import sys
from pathlib import Path
from collections import deque

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from biomechanics.utils.types import JointAngles, FaultSeverity
from biomechanics.faults.fault_types import FaultType
from biomechanics.faults.rules.depth import DepthRule, DepthCategory, depth_category
from biomechanics.faults.rules.symmetry import SymmetryRule
from biomechanics.faults.rules.forward_lean import ForwardLeanRule
from biomechanics.faults.rules.knee_valgus import KneeValgusRule, KNEE_VALGUS_COOLDOWN_S
from biomechanics.faults.rules.forward_lean import FORWARD_LEAN_COOLDOWN_S
from biomechanics.faults.rules.back_rounding import BackRoundingRule, BACK_ROUNDING_COOLDOWN_S
from biomechanics.faults.rule_engine import RuleEngine, DEDUP_INTERVAL_S

NAN = float("nan")
FPS = 30.0
FEMUR_M = 0.46
# A hip 19.5 cm above the knee: what 90° of knee flexion looks like with a 30° shin.
NINETY_DEGREE_KNEE_DEPTH_RATIO = 0.195 / FEMUR_M
REP_FRAMES = 60
PEAK_KNEE_FLEXION_DEG = 110.0
SYMMETRIC_NOISE_DEG = 2.0
ASYMMETRY_DEG = 12.0


def create_joint_angles(
    frame: int = 0,
    timestamp: float = 0.0,
    hip_flexion_l: float = 0.0,
    hip_flexion_r: float = 0.0,
    knee_flexion_l: float = 0.0,
    knee_flexion_r: float = 0.0,
    hip_adduction_l: float = 0.0,
    hip_adduction_r: float = 0.0,
    ankle_dorsiflexion_l: float = 20.0,
    ankle_dorsiflexion_r: float = 20.0,
    trunk_flexion: float = 0.0,
    knee_valgus_l: float = 0.0,
    knee_valgus_r: float = 0.0,
    foot_confidence_l: float = 0.0,
    foot_confidence_r: float = 0.0,
) -> JointAngles:
    """Create JointAngles with specified values, defaults for rest."""
    return JointAngles(
        hip_flexion_l=hip_flexion_l,
        hip_flexion_r=hip_flexion_r,
        knee_flexion_l=knee_flexion_l,
        knee_flexion_r=knee_flexion_r,
        hip_adduction_l=hip_adduction_l,
        hip_adduction_r=hip_adduction_r,
        ankle_dorsiflexion_l=ankle_dorsiflexion_l,
        ankle_dorsiflexion_r=ankle_dorsiflexion_r,
        knee_valgus_l=knee_valgus_l,
        knee_valgus_r=knee_valgus_r,
        foot_confidence_l=foot_confidence_l,
        foot_confidence_r=foot_confidence_r,
        trunk_flexion=trunk_flexion,
        frame_index=frame,
        timestamp=timestamp,
    )


class TestDepthRule:
    """DepthRule judges hip height against the athlete's target, never the knee angle."""

    @pytest.fixture
    def depth_rule(self):
        return DepthRule()

    @pytest.fixture
    def history(self):
        return deque(maxlen=90)

    def test_depth_category_classification(self):
        assert depth_category(-0.2) == DepthCategory.BELOW_PARALLEL
        assert depth_category(-0.02) == DepthCategory.PARALLEL
        assert depth_category(0.0) == DepthCategory.PARALLEL
        assert depth_category(0.3) == DepthCategory.HALF
        assert depth_category(0.7) == DepthCategory.QUARTER

    def test_ninety_degrees_of_knee_bend_is_not_parallel(self):
        """The old rule called 90° parallel while the hip sat ~20 cm above the knee."""
        assert depth_category(NINETY_DEGREE_KNEE_DEPTH_RATIO) == DepthCategory.HALF

    def test_fault_type(self, depth_rule):
        assert depth_rule.fault_type == FaultType.DEPTH

    def test_within_tolerance_is_no_fault(self, depth_rule):
        fault = depth_rule.judge_shallow_descent(0.05, 0.0, create_joint_angles(), rep_number=1)
        assert fault is None

    def test_short_of_target_is_mild(self, depth_rule):
        fault = depth_rule.judge_shallow_descent(0.15, 0.0, create_joint_angles(), rep_number=1)
        assert fault is not None
        assert fault.severity == FaultSeverity.MILD

    def test_half_squat_is_moderate(self, depth_rule):
        fault = depth_rule.judge_shallow_descent(
            NINETY_DEGREE_KNEE_DEPTH_RATIO, 0.0, create_joint_angles(), rep_number=1,
        )
        assert fault is not None
        assert fault.severity == FaultSeverity.MODERATE
        assert fault.details["category"] == DepthCategory.HALF

    def test_quarter_squat_is_severe(self, depth_rule):
        fault = depth_rule.judge_shallow_descent(0.7, 0.0, create_joint_angles(), rep_number=1)
        assert fault.severity == FaultSeverity.SEVERE

    def test_judged_against_the_athletes_target(self, depth_rule):
        """A target above parallel (limited hips) moves what counts as short."""
        assert depth_rule.judge_shallow_descent(0.3, 0.25, create_joint_angles(), rep_number=1) is None
        assert depth_rule.judge_shallow_descent(0.3, 0.0, create_joint_angles(), rep_number=1) is not None

    def test_shallow_fault_carries_the_contract_details(self, depth_rule):
        fault = depth_rule.judge_shallow_descent(0.4, 0.0, create_joint_angles(), rep_number=4)
        assert fault.rep_number == 4
        assert fault.details["shallow_rep"] is True
        assert fault.details["target_ratio"] == 0.0
        assert fault.details["unit"] == "ratio"

    def test_per_frame_path_never_emits(self, depth_rule, history):
        assert depth_rule.evaluate(create_joint_angles(knee_flexion_l=40.0), history, in_rep=True) is None

    def test_nan_depth_gives_no_fault(self, depth_rule):
        assert depth_rule.judge_shallow_descent(NAN, 0.0, create_joint_angles(), rep_number=1) is None


def _run_symmetry_reps(
    rule: SymmetryRule,
    n_reps: int,
    right_offset_deg: float,
    noise_deg: float,
    seed: int = 0,
) -> list:
    """Feed cosine knee-flexion reps through the rule; returns emitted faults.

    Each rep is REP_FRAMES in-rep frames followed by 10 standing frames.
    """
    rng = np.random.default_rng(seed)
    history: deque = deque(maxlen=90)
    faults = []
    frame = 0
    for rep_index in range(n_reps):
        rep_number = rep_index + 1
        for i in range(REP_FRAMES):
            knee = PEAK_KNEE_FLEXION_DEG * 0.5 * (1.0 - math.cos(2.0 * math.pi * (i + 1) / REP_FRAMES))
            angles = create_joint_angles(
                frame=frame,
                timestamp=frame / FPS,
                knee_flexion_l=knee + rng.normal(0.0, noise_deg),
                knee_flexion_r=knee + right_offset_deg + rng.normal(0.0, noise_deg),
            )
            fault = rule.evaluate(angles, history, in_rep=True, rep_number=rep_number)
            if fault is not None:
                faults.append(fault)
            frame += 1
        # The pipeline judges the rep on the frame the counter completes it.
        fault = rule.finish_rep(angles, rep_number)
        if fault is not None:
            faults.append(fault)
        for _ in range(10):
            angles = create_joint_angles(frame=frame, timestamp=frame / FPS, knee_flexion_l=3.0, knee_flexion_r=3.0)
            fault = rule.evaluate(angles, history, in_rep=False, rep_number=rep_number + 1)
            if fault is not None:
                faults.append(fault)
            frame += 1
    return faults


class TestSymmetryRule:
    """SymmetryRule evaluates one aggregate per rep over the bottom window
    (S16) instead of firing on every noisy frame."""

    @pytest.fixture
    def symmetry_rule(self):
        return SymmetryRule()

    @pytest.fixture
    def history(self):
        return deque(maxlen=90)

    def test_fault_type(self, symmetry_rule):
        """Symmetry rule should report BILATERAL_ASYMMETRY fault type."""
        assert symmetry_rule.fault_type == FaultType.BILATERAL_ASYMMETRY

    def test_symmetric_noisy_reps_produce_no_fault(self, symmetry_rule):
        faults = _run_symmetry_reps(symmetry_rule, n_reps=6, right_offset_deg=0.0, noise_deg=SYMMETRIC_NOISE_DEG)
        assert faults == []

    def test_genuine_asymmetry_fires_once_per_rep(self, symmetry_rule):
        faults = _run_symmetry_reps(symmetry_rule, n_reps=4, right_offset_deg=-ASYMMETRY_DEG, noise_deg=SYMMETRIC_NOISE_DEG)
        assert len(faults) == 4
        assert [fault.rep_number for fault in faults] == [1, 2, 3, 4]
        for fault in faults:
            assert fault.severity == FaultSeverity.MODERATE
            assert fault.details["heavier_side"] == "left"
            assert fault.details["asymmetry"] == pytest.approx(ASYMMETRY_DEG, abs=2.0)

    def test_single_noisy_frame_cannot_fire(self, symmetry_rule, history):
        """A per-frame 20 degree spike used to fire SEVERE immediately."""
        angles = create_joint_angles(frame=0, knee_flexion_l=90.0, knee_flexion_r=70.0)
        assert symmetry_rule.evaluate(angles, history, in_rep=True, rep_number=1) is None

    def test_no_fault_when_not_in_rep(self, symmetry_rule, history):
        angles = create_joint_angles(frame=0, knee_flexion_l=60.0, knee_flexion_r=40.0)
        assert symmetry_rule.evaluate(angles, history, in_rep=False) is None

    def test_severity_bands(self):
        mild = _run_symmetry_reps(SymmetryRule(), n_reps=1, right_offset_deg=-7.0, noise_deg=0.0)
        severe = _run_symmetry_reps(SymmetryRule(), n_reps=1, right_offset_deg=-20.0, noise_deg=0.0)
        assert [fault.severity for fault in mild] == [FaultSeverity.MILD]
        assert [fault.severity for fault in severe] == [FaultSeverity.SEVERE]

    def test_uncounted_rep_is_discarded_without_a_fault(self, symmetry_rule, history):
        """A false start or rejected descent never gets judged."""
        for frame in range(REP_FRAMES):
            angles = create_joint_angles(frame=frame, timestamp=frame / FPS, knee_flexion_l=90.0, knee_flexion_r=70.0)
            assert symmetry_rule.evaluate(angles, history, in_rep=True, rep_number=1) is None
        symmetry_rule.discard_rep()
        after = create_joint_angles(frame=REP_FRAMES, timestamp=REP_FRAMES / FPS, knee_flexion_l=3.0, knee_flexion_r=3.0)
        assert symmetry_rule.evaluate(after, history, in_rep=False, rep_number=2) is None
        assert symmetry_rule.finish_rep(after, 1) is None

    def test_rep_ended_without_verdict_is_dropped_on_next_frame(self, symmetry_rule, history):
        for frame in range(REP_FRAMES):
            angles = create_joint_angles(frame=frame, timestamp=frame / FPS, knee_flexion_l=90.0, knee_flexion_r=70.0)
            symmetry_rule.evaluate(angles, history, in_rep=True, rep_number=1)
        after = create_joint_angles(frame=REP_FRAMES, timestamp=REP_FRAMES / FPS, knee_flexion_l=3.0, knee_flexion_r=3.0)
        assert symmetry_rule.evaluate(after, history, in_rep=False, rep_number=2) is None
        assert symmetry_rule.finish_rep(after, 1) is None

    def test_nan_frames_are_skipped_without_corrupting_the_rep(self, symmetry_rule, history):
        frame = 0
        for i in range(REP_FRAMES):
            knee = PEAK_KNEE_FLEXION_DEG * 0.5 * (1.0 - math.cos(2.0 * math.pi * (i + 1) / REP_FRAMES))
            left = NAN if i % 3 == 0 else knee
            angles = create_joint_angles(frame=frame, timestamp=frame / FPS, knee_flexion_l=left, knee_flexion_r=knee)
            assert symmetry_rule.evaluate(angles, history, in_rep=True, rep_number=1) is None
            frame += 1
        after = create_joint_angles(frame=frame, timestamp=frame / FPS, knee_flexion_l=3.0, knee_flexion_r=3.0)
        assert symmetry_rule.evaluate(after, history, in_rep=False, rep_number=2) is None


class TestForwardLeanRule:
    """Test ForwardLeanRule for excessive trunk flexion."""

    @pytest.fixture
    def forward_lean_rule(self):
        return ForwardLeanRule()

    @pytest.fixture
    def history(self):
        return deque(maxlen=90)

    def test_fault_type(self, forward_lean_rule):
        """Forward lean rule should report FORWARD_LEAN fault type."""
        assert forward_lean_rule.fault_type == FaultType.FORWARD_LEAN

    # trunk_flexion uses the 180-convention: 180° = upright, lower = more lean.
    # Default thresholds: mild < 145°, moderate < 135°, severe < 125°.

    def test_no_fault_when_upright(self, forward_lean_rule, history):
        """No fault when trunk is near-upright (above mild threshold)."""
        angles = create_joint_angles(frame=0, trunk_flexion=170.0)
        fault = forward_lean_rule.evaluate(angles, history, in_rep=True)
        assert fault is None

    def test_no_fault_when_not_in_rep(self, forward_lean_rule, history):
        """No fault when not in rep even with lean."""
        angles = create_joint_angles(frame=0, trunk_flexion=130.0)
        fault = forward_lean_rule.evaluate(angles, history, in_rep=False)
        assert fault is None

    def test_mild_forward_lean(self, forward_lean_rule, history):
        """Mild forward lean (between mild and moderate thresholds)."""
        angles = create_joint_angles(frame=0, trunk_flexion=140.0)
        fault = forward_lean_rule.evaluate(angles, history, in_rep=True)
        assert fault is not None
        assert fault.severity == FaultSeverity.MILD

    def test_moderate_forward_lean(self, forward_lean_rule, history):
        """Moderate forward lean (between moderate and severe thresholds)."""
        angles = create_joint_angles(frame=0, trunk_flexion=130.0)
        fault = forward_lean_rule.evaluate(angles, history, in_rep=True)
        assert fault is not None
        assert fault.severity == FaultSeverity.MODERATE

    def test_severe_forward_lean(self, forward_lean_rule, history):
        """Severe forward lean (below severe threshold)."""
        angles = create_joint_angles(frame=0, trunk_flexion=120.0)
        fault = forward_lean_rule.evaluate(angles, history, in_rep=True)
        assert fault is not None
        assert fault.severity == FaultSeverity.SEVERE

    def test_nan_trunk_flexion_is_skipped(self, forward_lean_rule, history):
        angles = create_joint_angles(frame=0, trunk_flexion=NAN)
        assert forward_lean_rule.evaluate(angles, history, in_rep=True) is None
        # and the cooldown was not consumed
        angles = create_joint_angles(frame=0, timestamp=0.0, trunk_flexion=120.0)
        assert forward_lean_rule.evaluate(angles, history, in_rep=True) is not None

    def test_cooldown_is_time_based_not_frame_based(self, forward_lean_rule, history):
        """Triangulated skeletons carried frame_index 0 forever, so a frame
        cooldown fired once per session (W1)."""
        first = create_joint_angles(frame=0, timestamp=100.0, trunk_flexion=120.0)
        assert forward_lean_rule.evaluate(first, history, in_rep=True) is not None
        within = create_joint_angles(frame=0, timestamp=100.0 + FORWARD_LEAN_COOLDOWN_S / 2, trunk_flexion=120.0)
        assert forward_lean_rule.evaluate(within, history, in_rep=True) is None
        after = create_joint_angles(frame=0, timestamp=100.0 + FORWARD_LEAN_COOLDOWN_S, trunk_flexion=120.0)
        assert forward_lean_rule.evaluate(after, history, in_rep=True) is not None


def _valgus_angles(
    valgus_l: float, valgus_r: float, confidence: float = 0.9, timestamp: float = 0.0,
    hip_adduction: float = 0.0,
) -> JointAngles:
    return create_joint_angles(
        timestamp=timestamp,
        knee_valgus_l=valgus_l,
        knee_valgus_r=valgus_r,
        foot_confidence_l=confidence,
        foot_confidence_r=confidence,
        hip_adduction_l=hip_adduction,
        hip_adduction_r=hip_adduction,
    )


class TestKneeValgusRule:
    """KneeValgusRule (lunge): per-frame knees-over-toes metric, silent without the feet."""

    @pytest.fixture
    def knee_valgus_rule(self):
        rule = KneeValgusRule()
        rule.set_frame_context(phase="bottom")
        return rule

    @pytest.fixture
    def history(self):
        return deque(maxlen=90)

    def test_fault_type(self, knee_valgus_rule):
        assert knee_valgus_rule.fault_type == FaultType.KNEE_VALGUS

    def test_no_fault_when_knees_out(self, knee_valgus_rule, history):
        assert knee_valgus_rule.evaluate(_valgus_angles(-6.0, -5.0), history, in_rep=True) is None

    def test_no_fault_when_not_in_rep(self, knee_valgus_rule, history):
        assert knee_valgus_rule.evaluate(_valgus_angles(20.0, 20.0), history, in_rep=False) is None

    def test_mild_valgus(self, knee_valgus_rule, history):
        fault = knee_valgus_rule.evaluate(_valgus_angles(14.0, 13.0), history, in_rep=True)
        assert fault.severity == FaultSeverity.MILD

    def test_moderate_valgus(self, knee_valgus_rule, history):
        fault = knee_valgus_rule.evaluate(_valgus_angles(19.0, 18.0), history, in_rep=True)
        assert fault.severity == FaultSeverity.MODERATE

    def test_severe_valgus(self, knee_valgus_rule, history):
        fault = knee_valgus_rule.evaluate(_valgus_angles(27.0, 25.0), history, in_rep=True)
        assert fault.severity == FaultSeverity.SEVERE

    def test_identifies_affected_side(self, knee_valgus_rule, history):
        fault = knee_valgus_rule.evaluate(_valgus_angles(6.0, 14.0), history, in_rep=True)
        assert fault.details["affected_side"] == "right"

    def test_no_fault_without_the_feet(self, knee_valgus_rule, history):
        """The hip-adduction fallback read toe-in as valgus; unseen feet now mean silence."""
        angles = _valgus_angles(20.0, 20.0, confidence=0.1, hip_adduction=19.0)
        assert knee_valgus_rule.evaluate(angles, history, in_rep=True) is None

    def test_toe_in_hip_adduction_alone_is_not_valgus(self, knee_valgus_rule, history):
        """Thigh angle against the world axis is foot angle, not knee cave."""
        angles = _valgus_angles(0.0, 0.0, hip_adduction=19.0)
        assert knee_valgus_rule.evaluate(angles, history, in_rep=True) is None

    def test_no_fault_below_threshold(self, knee_valgus_rule, history):
        assert knee_valgus_rule.evaluate(_valgus_angles(2.0, 3.0), history, in_rep=True) is None

    def test_nan_valgus_is_skipped(self, knee_valgus_rule, history):
        assert knee_valgus_rule.evaluate(_valgus_angles(NAN, 30.0), history, in_rep=True) is None

    def test_cooldown_is_time_based_not_frame_based(self, knee_valgus_rule, history):
        assert knee_valgus_rule.evaluate(_valgus_angles(15.0, 15.0, timestamp=50.0), history, in_rep=True) is not None
        within = _valgus_angles(15.0, 15.0, timestamp=50.0 + KNEE_VALGUS_COOLDOWN_S / 2)
        assert knee_valgus_rule.evaluate(within, history, in_rep=True) is None
        after = _valgus_angles(15.0, 15.0, timestamp=50.0 + KNEE_VALGUS_COOLDOWN_S)
        assert knee_valgus_rule.evaluate(after, history, in_rep=True) is not None

    def test_no_proportion_scaling(self, knee_valgus_rule):
        """C8: the valgus metric has no valid hip-width scaling (F5)."""
        class Proportions:
            forward_lean_scale = 1.3
        before = (knee_valgus_rule.mild_threshold, knee_valgus_rule.moderate_threshold, knee_valgus_rule.severe_threshold)
        knee_valgus_rule.scale_for_proportions(Proportions())
        assert (knee_valgus_rule.mild_threshold, knee_valgus_rule.moderate_threshold, knee_valgus_rule.severe_threshold) == before


class TestBackRoundingRule:

    @pytest.fixture
    def history(self):
        return deque(maxlen=90)

    def test_nan_setup_frame_does_not_corrupt_setup(self, history):
        rule = BackRoundingRule()
        assert rule.evaluate(create_joint_angles(frame=0, timestamp=0.0, trunk_flexion=NAN), history, in_rep=True) is None
        assert rule.evaluate(create_joint_angles(frame=1, timestamp=1 / FPS, trunk_flexion=170.0), history, in_rep=True) is None
        assert rule.evaluate(create_joint_angles(frame=2, timestamp=2 / FPS, trunk_flexion=NAN), history, in_rep=True) is None
        fault = rule.evaluate(create_joint_angles(frame=3, timestamp=3 / FPS, trunk_flexion=145.0), history, in_rep=True)
        assert fault is not None
        assert fault.details["setup_trunk_flexion"] == pytest.approx(170.0)

    def test_cooldown_is_time_based_not_frame_based(self, history):
        rule = BackRoundingRule()
        rule.evaluate(create_joint_angles(frame=0, timestamp=10.0, trunk_flexion=170.0), history, in_rep=True)
        assert rule.evaluate(create_joint_angles(frame=0, timestamp=10.1, trunk_flexion=145.0), history, in_rep=True) is not None
        assert rule.evaluate(create_joint_angles(frame=0, timestamp=10.2, trunk_flexion=145.0), history, in_rep=True) is None
        assert rule.evaluate(
            create_joint_angles(frame=0, timestamp=10.1 + BACK_ROUNDING_COOLDOWN_S, trunk_flexion=145.0), history, in_rep=True,
        ) is not None


class TestRuleEngine:
    """Test RuleEngine orchestration."""

    @pytest.fixture
    def engine(self):
        # RuleEngine requires an explicit rule list (profiles are the source of
        # truth in production, but the tests build the set they care about).
        return RuleEngine(
            rules=[
                DepthRule(),
                SymmetryRule(),
                ForwardLeanRule(),
                KneeValgusRule(),
            ],
        )

    def test_initialization(self, engine):
        """Engine should initialize with all rules."""
        assert engine.rule_count == 4  # depth, symmetry, forward_lean, knee_valgus

    def test_history_tracking(self, engine):
        """Engine should maintain angle history."""
        angles = create_joint_angles(frame=0)
        engine.evaluate(angles, in_rep=True)
        assert engine.history_length == 1

        for i in range(1, 10):
            engine.evaluate(create_joint_angles(frame=i), in_rep=True)
        assert engine.history_length == 10

    def test_deduplication(self, engine):
        """Engine should deduplicate consecutive same-fault detections."""
        # Trigger forward lean fault
        angles = create_joint_angles(frame=0, trunk_flexion=50.0)
        faults1 = engine.evaluate(angles, in_rep=True)
        forward_lean_count = sum(1 for f in faults1 if f.fault_type == "forward_lean")

        # Same fault immediately after should be deduplicated
        angles2 = create_joint_angles(frame=1, trunk_flexion=50.0)
        faults2 = engine.evaluate(angles2, in_rep=True)
        forward_lean_count2 = sum(1 for f in faults2 if f.fault_type == "forward_lean")

        # First should fire, second should be deduplicated
        assert forward_lean_count <= 1
        assert forward_lean_count2 == 0

    def test_deduplication_is_time_based(self):
        """Two rules reporting the same fault type: the engine's dedup window
        must be measured in seconds even when frame_index never advances."""
        engine = RuleEngine(rules=[KneeValgusRule(), KneeValgusRule()])
        angles = _valgus_angles(15.0, 15.0, timestamp=20.0)
        assert len(engine.evaluate(angles, in_rep=True, phase="bottom")) == 1
        later = _valgus_angles(15.0, 15.0, timestamp=20.0 + DEDUP_INTERVAL_S + KNEE_VALGUS_COOLDOWN_S)
        assert len(engine.evaluate(later, in_rep=True, phase="bottom")) == 1

    def test_nan_frame_produces_no_faults_and_no_state_change(self, engine):
        angles = create_joint_angles(
            frame=0, timestamp=0.0, trunk_flexion=NAN, knee_flexion_l=NAN, knee_flexion_r=NAN,
            hip_adduction_l=NAN, hip_adduction_r=NAN, knee_valgus_l=NAN, knee_valgus_r=NAN,
            foot_confidence_l=0.0, foot_confidence_r=0.0,
        )
        assert engine.evaluate(angles, in_rep=True, phase="bottom") == []
        assert engine.evaluate_rep_complete(NAN, angles, rep_number=1) == []

    def test_multiple_faults_same_frame(self, engine):
        """Engine should detect multiple fault types in same frame."""
        # Create angles with multiple issues
        angles = create_joint_angles(
            frame=0,
            trunk_flexion=60.0,  # Forward lean
            knee_valgus_l=15.0,  # Knee valgus, feet seen
            knee_valgus_r=15.0,
            foot_confidence_l=0.9,
            foot_confidence_r=0.9,
        )
        faults = engine.evaluate(angles, in_rep=True, phase="bottom")

        fault_types = {f.fault_type for f in faults}
        # Should detect at least forward lean and one other
        assert len(faults) >= 2

    def test_reset(self, engine):
        """Reset should clear history and rule states."""
        for i in range(50):
            engine.evaluate(create_joint_angles(frame=i), in_rep=True)
        assert engine.history_length == 50

        engine.reset()
        assert engine.history_length == 0

    def test_shallow_descent_becomes_a_depth_fault(self, engine):
        engine.set_depth_target(0.0)
        faults = engine.judge_shallow_descent(0.6, create_joint_angles(), rep_number=4)
        assert len(faults) == 1
        assert faults[0].fault_type == "depth"
        assert faults[0].rep_number == 4
        assert faults[0].details["shallow_rep"] is True
        assert faults[0].details["observability"] == "observable"

    def test_no_depth_fault_without_a_target(self, engine):
        """Assessment and calibration count every descent."""
        engine.set_depth_target(None)
        assert engine.judge_shallow_descent(0.6, create_joint_angles(), rep_number=1) == []
        assert engine.reaches_depth_target(0.9)

    def test_target_tolerance_decides_what_counts(self, engine):
        engine.set_depth_target(0.0)
        assert engine.reaches_depth_target(0.05)
        assert not engine.reaches_depth_target(0.2)

    def test_shallow_descent_without_depth_rule(self, engine):
        """Profiles with no depth rule simply produce nothing."""
        engine.remove_rule(FaultType.DEPTH)
        engine.set_depth_target(0.0)
        assert engine.judge_shallow_descent(0.6, create_joint_angles(), rep_number=1) == []

    def test_get_rule(self, engine):
        """Should be able to get specific rule by type."""
        depth_rule = engine.get_rule(FaultType.DEPTH)
        assert depth_rule is not None
        assert isinstance(depth_rule, DepthRule)

    def test_add_remove_rule(self, engine):
        """Should be able to add and remove rules."""
        initial_count = engine.rule_count

        # Remove depth rule
        removed = engine.remove_rule(FaultType.DEPTH)
        assert removed
        assert engine.rule_count == initial_count - 1

        # Add it back
        engine.add_rule(DepthRule())
        assert engine.rule_count == initial_count
