"""Deadlift rep analyser on simulated sets (docs/deadlift/PLAN.md §2.3-2.5, J2/J3):
every rep counted exactly once, events on time, each injected fault detected by
its rule and a clean set fault-free."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pytest

from biomechanics.config import BiomechanicsConfig
from biomechanics.deadlift.analyzer import LIVE_OFFSET_MIN_FRAMES, DeadliftFrameInput, DeadliftRepAnalyzer
from biomechanics.deadlift.rule_base import TIER_RANK
from biomechanics.deadlift.setup_model import MAX_SETUP_KNEE_FLEXION_DEG, MIN_SETUP_KNEE_FLEXION_DEG
from biomechanics.deadlift.session_reference import DeadliftSessionReference
from biomechanics.deadlift.simulator import RepScript, Scenario, SimAthlete, SimFrame, SimulatedSet, simulate
from biomechanics.deadlift.types import (
    BAR_SOURCE_BAR,
    BAR_SOURCE_WRIST_PROXY,
    GRAVITY_SOURCE_BODY,
    GRAVITY_SOURCE_MEASURED,
    BarState3D,
    DeadliftPhase,
    DeadliftRepFeatures,
)
from biomechanics.faults.rule_engine import RuleEngine
from biomechanics.profiles.deadlift import DeadliftProfile
from biomechanics.utils.geometry import WORLD_UP
from biomechanics.utils.types import CocoKeypoints as CK
from biomechanics.utils.types import FaultEvent, JointAngles

# PLAN.md §1 demo gate: event timing median error <= 100 ms; held here per event.
MAX_EVENT_ERROR_S = 0.1
CLEAN_DRIFT_MAX_CM = 1.0
CLEAN_TILT_MAX_CM = 1.0
CLEAN_SHIFT_MAX_RATIO = 0.03
CLEAN_ANGLE_TOLERANCE_DEG = 2.0
# 3 deg of tilted world vertical over the pull fakes this much drift without measured gravity.
TILT_FAKE_DRIFT_MIN_CM = 2.0
# Keypoint noise the platform documents for Kalman-lagged triangulated keypoints
# (utils/segment_lengths.py: 1.6-2.4 cm), and a tracked bar's.
PLATFORM_KEYPOINT_NOISE_M = 0.02
TRACKED_BAR_NOISE_M = 0.003
# The agent arms closed-loop foot guidance beyond this offset (CONTRACT.md §5.5:
# D1's mild threshold), then guides down to the tolerance.
BAR_MIDFOOT_GUIDANCE_ARM_CM = 3.0
BAR_MIDFOOT_GUIDANCE_TOLERANCE_CM = 2.0
# The plates hide the knees from the cameras with the bar this far off the floor.
KNEES_HIDDEN_ABOVE_M = 0.15
# A re-setup at the floor: the lifter shuffles the feet this far toward the bar.
FEET_MOVED_AT_THE_FLOOR_M = 0.05
# Hip shift a stance leaks into D8 beyond the same seed's square stance.
STANCE_LEAK_MAX_RATIO = 0.03
# A staggered stance: the left foot this far ahead of the right, hips square.
STAGGER_M = 0.06
LEFT_FOOT = (CK.LEFT_ANKLE, CK.LEFT_HEEL, CK.LEFT_FOOT_INDEX)
# Keypoints of a stance turned off square to the bar (the hands stay on it).
LOWER_BODY = (
    CK.LEFT_HIP, CK.RIGHT_HIP, CK.LEFT_KNEE, CK.RIGHT_KNEE, CK.LEFT_ANKLE, CK.RIGHT_ANKLE,
    CK.LEFT_HEEL, CK.RIGHT_HEEL, CK.LEFT_FOOT_INDEX, CK.RIGHT_FOOT_INDEX,
)
FAULTS = BiomechanicsConfig().faults
# Kinematic truth vs measurement on noise-free model-free poses.
# One frame (the first after the bar passes the knees) of trunk motion.
TRUNK_CHANGE_TOLERANCE_DEG = 2.0
RISE_RATIO_TOLERANCE = 0.04
HIP_HEIGHT_TOLERANCE_CM = 1.0


FrameMutation = Callable[[int, SimFrame], tuple[np.ndarray, np.ndarray, BarState3D | None]]


def _correlated_noise(sigma_m: float, rho: float, seed: int) -> FrameMutation:
    """AR(1) keypoint noise of stationary std sigma_m: the slow wander a Kalman-
    smoothed triangulation leaves, which a short window reads as motion."""
    rng = np.random.default_rng(seed)
    state: dict[str, np.ndarray] = {}

    def mutate(index: int, frame: SimFrame) -> tuple[np.ndarray, np.ndarray, BarState3D | None]:
        innovation = rng.normal(0.0, sigma_m * math.sqrt(1.0 - rho ** 2), frame.points.shape)
        state["error"] = innovation if "error" not in state else rho * state["error"] + innovation
        return frame.points + state["error"], frame.confidences, frame.bar

    return mutate


def _bar_dropout(probability: float, seed: int) -> FrameMutation:
    rng = np.random.default_rng(seed)

    def mutate(index: int, frame: SimFrame) -> tuple[np.ndarray, np.ndarray, BarState3D | None]:
        return frame.points, frame.confidences, None if rng.random() < probability else frame.bar

    return mutate


# The whole lifter moves toward the bar (-Z) over 0.8 s, starting 0.8 s after
# touchdown, still hinged over it.
def _stepped_closer_at_the_floor(floor_time: float, forward_m: float) -> FrameMutation:
    def mutate(index: int, frame: SimFrame) -> tuple[np.ndarray, np.ndarray, BarState3D | None]:
        fraction = min(1.0, max(0.0, (frame.timestamp - floor_time - 0.8) / 0.8))
        points = frame.points.copy()
        points[:, 2] -= forward_m * fraction
        return points, frame.confidences, frame.bar

    return mutate


# Keypoints turned about the vertical through the ankles, then the noise (if any).
def _turned(
    sim: SimulatedSet, degrees: float, keypoints: tuple[int, ...], noise: FrameMutation | None = None,
) -> FrameMutation:
    pivot = (sim.frames[0].points[CK.LEFT_ANKLE] + sim.frames[0].points[CK.RIGHT_ANKLE]) / 2.0
    angle = math.radians(degrees)
    rotation = np.array([
        [math.cos(angle), 0.0, math.sin(angle)], [0.0, 1.0, 0.0], [-math.sin(angle), 0.0, math.cos(angle)],
    ])
    indices = list(keypoints)

    def mutate(index: int, frame: SimFrame) -> tuple[np.ndarray, np.ndarray, BarState3D | None]:
        points = frame.points.copy()
        points[indices] = (points[indices] - pivot) @ rotation.T + pivot
        turned = frame._replace(points=points)
        return (turned.points, turned.confidences, turned.bar) if noise is None else noise(index, turned)

    return mutate


# The left foot moved forward (-Z), then the noise (if any).
def _staggered(forward_m: float, noise: FrameMutation | None = None) -> FrameMutation:
    def mutate(index: int, frame: SimFrame) -> tuple[np.ndarray, np.ndarray, BarState3D | None]:
        points = frame.points.copy()
        points[list(LEFT_FOOT), 2] -= forward_m
        staggered = frame._replace(points=points)
        return (staggered.points, staggered.confidences, staggered.bar) if noise is None else noise(index, staggered)

    return mutate


# The whole lifter stands turned `degrees` about the ankles, then squares up to
# the bar in place between square_from and square_to, then the noise.
def _turned_then_squared(
    sim: SimulatedSet, degrees: float, square_from: float, square_to: float, noise: FrameMutation,
) -> FrameMutation:
    pivot = (sim.frames[0].points[CK.LEFT_ANKLE] + sim.frames[0].points[CK.RIGHT_ANKLE]) / 2.0

    def mutate(index: int, frame: SimFrame) -> tuple[np.ndarray, np.ndarray, BarState3D | None]:
        share = min(1.0, max(0.0, (square_to - frame.timestamp) / (square_to - square_from)))
        angle = math.radians(degrees * share)
        rotation = np.array([
            [math.cos(angle), 0.0, math.sin(angle)], [0.0, 1.0, 0.0], [-math.sin(angle), 0.0, math.cos(angle)],
        ])
        return noise(index, frame._replace(points=(frame.points - pivot) @ rotation.T + pivot))

    return mutate


def _analyse(
    scenario: Scenario,
    gravity_source: str = GRAVITY_SOURCE_MEASURED,
    meta: dict | None = None,
    mutate: FrameMutation | None = None,
) -> tuple[SimulatedSet, list[DeadliftRepFeatures], DeadliftRepAnalyzer, list[DeadliftPhase]]:
    sim = simulate(scenario)
    analyzer = DeadliftRepAnalyzer(BiomechanicsConfig().deadlift)
    up = sim.gravity_up_world if gravity_source == GRAVITY_SOURCE_MEASURED else np.array(WORLD_UP)
    analyzer.set_gravity(up, gravity_source)
    if meta is not None:
        analyzer.set_session_meta(meta)
    features: list[DeadliftRepFeatures] = []
    phases: list[DeadliftPhase] = []
    for index, frame in enumerate(sim.frames):
        points, confidences, bar = (frame.points, frame.confidences, frame.bar) if mutate is None else mutate(index, frame)
        analyzer.observe(DeadliftFrameInput(frame.timestamp, frame.frame_index, points, confidences, bar))
        if not phases or phases[-1] != analyzer.phase:
            phases.append(analyzer.phase)
        while analyzer.take_completed_rep() is not None:
            features.append(analyzer.finish_rep())
    return sim, features, analyzer, phases


def _faults(features: list[DeadliftRepFeatures]) -> list[list[FaultEvent]]:
    """Each rep's faults, judged as the pipeline does."""
    engine = RuleEngine(
        rules=DeadliftProfile().create_fault_rules(BiomechanicsConfig()),
        reference=DeadliftSessionReference(),
        capture_mode="triangulated",
    )
    return [
        engine.finish_rep(JointAngles(timestamp=rep_features.top_time), rep_features.rep_number, rep_features)
        for rep_features in features
    ]


def _judge(features: list[DeadliftRepFeatures]) -> list[dict[str, str]]:
    """Each rep's faults as {fault_type: severity}."""
    return [{fault.fault_type: fault.severity.value for fault in faults} for faults in _faults(features)]


def _cued(features: list[DeadliftRepFeatures]) -> list[set[str]]:
    """Each rep's faults the voice agent would speak: at or above their min tier."""
    return [
        {fault.fault_type for fault in faults if TIER_RANK[fault.severity.value] >= TIER_RANK[fault.details["min_tier"]]}
        for faults in _faults(features)
    ]


class TestRepCounting:
    def test_clean_set_counts_each_rep_once(self):
        sim, features, analyzer, _ = _analyse(Scenario())
        assert [f.rep_number for f in features] == [1, 2, 3]
        assert analyzer.rep_count == len(sim.reps)
        assert analyzer.failed_reps == 0

    def test_touch_and_go_reps_are_counted_once_each(self):
        scenario = Scenario(reps=[RepScript(floor_hold_s=0.0), RepScript(floor_hold_s=0.0), RepScript()])
        _, features, _, _ = _analyse(scenario)
        assert [f.rep_number for f in features] == [1, 2, 3]
        assert [f.touch_and_go for f in features] == [False, True, True]

    def test_dropped_bar_bounce_is_not_a_rep(self):
        _, features, analyzer, _ = _analyse(Scenario(reps=[RepScript(drop_bar=True), RepScript()]))
        assert analyzer.rep_count == 2
        assert not any(f.touch_and_go for f in features)

    def test_failed_rep_is_an_event_not_a_rep(self):
        scenario = Scenario(reps=[RepScript(), RepScript(fail_rise_m=0.25), RepScript()])
        _, features, analyzer, _ = _analyse(scenario)
        assert analyzer.rep_count == 2
        assert analyzer.failed_reps == 1
        assert len(features) == 2

    def test_quick_repull_judges_the_setup_from_the_moment_before_liftoff(self):
        scenario = Scenario(reps=[RepScript(floor_hold_s=0.3), RepScript(floor_hold_s=0.3), RepScript()])
        _, features, _, _ = _analyse(scenario)
        assert len(features) == 3
        assert all(f.setup_measured for f in features)

    def test_a_repull_too_quick_to_see_leaves_the_setup_unmeasured(self):
        scenario = Scenario(reps=[RepScript(floor_hold_s=0.15), RepScript(floor_hold_s=0.15), RepScript()])
        _, features, _, _ = _analyse(scenario)
        assert len(features) == 3
        assert [f.setup_measured for f in features] == [True, False, False]
        assert math.isnan(features[1].bar_midfoot_setup_cm)

    def test_noisy_keypoints_and_bar_still_count_every_rep(self):
        _, features, _, _ = _analyse(Scenario(keypoint_noise_m=0.006, bar_noise_m=0.003))
        assert len(features) == 3

    def test_wrists_stand_in_for_an_untracked_bar(self):
        _, features, _, _ = _analyse(Scenario(track_bar=False))
        assert len(features) == 3
        assert {f.bar_source for f in features} == {BAR_SOURCE_WRIST_PROXY}


class TestEventTiming:
    @pytest.mark.parametrize(
        "scenario",
        [
            Scenario(),
            Scenario(reps=[RepScript(floor_hold_s=0.0), RepScript()]),
            Scenario(keypoint_noise_m=0.006, bar_noise_m=0.003),
            Scenario(track_bar=False),
        ],
        ids=["dead_stop", "touch_and_go", "noisy", "wrist_proxy"],
    )
    def test_liftoff_knee_pass_top_and_floor_land_within_100_ms(self, scenario: Scenario):
        sim, features, _, _ = _analyse(scenario)
        truth = [rep for rep in sim.reps if rep.counted]
        assert len(features) == len(truth)
        for expected, measured in zip(truth, features):
            assert measured.liftoff_time == pytest.approx(expected.liftoff_time, abs=MAX_EVENT_ERROR_S)
            assert measured.knee_pass_time == pytest.approx(expected.knee_pass_time, abs=MAX_EVENT_ERROR_S)
            assert measured.top_time == pytest.approx(expected.top_time, abs=MAX_EVENT_ERROR_S)
            assert measured.floor_time == pytest.approx(expected.floor_time, abs=MAX_EVENT_ERROR_S)


class TestPhases:
    def test_a_set_walks_through_every_phase_in_order(self):
        _, _, _, phases = _analyse(Scenario(reps=[RepScript()]))
        assert phases[:7] == [
            DeadliftPhase.APPROACH, DeadliftPhase.STANCE, DeadliftPhase.SETUP, DeadliftPhase.PULL,
            DeadliftPhase.TOP, DeadliftPhase.LOWER, DeadliftPhase.FLOOR,
        ]

    def test_standing_up_and_walking_away_returns_to_approach(self):
        _, _, analyzer, phases = _analyse(Scenario(reps=[RepScript()]))
        assert DeadliftPhase.STANCE in phases[7:]
        assert analyzer.phase == DeadliftPhase.APPROACH

    def test_live_bar_offset_is_reported_only_while_standing_at_the_bar(self):
        sim = simulate(Scenario(bar_midfoot_offset_m=0.08))
        analyzer = DeadliftRepAnalyzer()
        analyzer.set_gravity(sim.gravity_up_world, GRAVITY_SOURCE_MEASURED)
        live_by_phase: dict[DeadliftPhase, list[float]] = {}
        before_the_pull: list[float] = []
        for frame in sim.frames:
            analyzer.observe(DeadliftFrameInput(frame.timestamp, frame.frame_index, frame.points, frame.confidences, frame.bar))
            live_by_phase.setdefault(analyzer.status.phase, []).append(analyzer.status.bar_midfoot_live_cm)
            if analyzer.status.phase == DeadliftPhase.STANCE and frame.timestamp < sim.reps[0].liftoff_time:
                before_the_pull.append(analyzer.status.bar_midfoot_live_cm)
        # Standing at the bar it reads, once a median of settled frames is in;
        # hinging down to it (moving) it does not.
        measured = [value for value in before_the_pull if math.isfinite(value)]
        assert all(math.isnan(value) for value in before_the_pull[:LIVE_OFFSET_MIN_FRAMES - 1])
        assert len(measured) >= len(before_the_pull) // 3
        assert np.median(measured) == pytest.approx(8.0, abs=0.5)
        assert all(math.isnan(value) for value in live_by_phase[DeadliftPhase.PULL])

    def test_no_live_offset_after_the_sets_first_rep(self):
        """Standing up and walking off after a set is not a lifter to guide in."""
        sim = simulate(Scenario(reps=[RepScript()]))
        analyzer = DeadliftRepAnalyzer()
        analyzer.set_gravity(sim.gravity_up_world, GRAVITY_SOURCE_MEASURED)
        after_the_rep: list[float] = []
        for frame in sim.frames:
            analyzer.observe(DeadliftFrameInput(frame.timestamp, frame.frame_index, frame.points, frame.confidences, frame.bar))
            if analyzer.rep_count > 0:
                after_the_rep.append(analyzer.status.bar_midfoot_live_cm)
        assert after_the_rep and all(math.isnan(value) for value in after_the_rep)

    def test_rep_signal_is_bar_height_above_its_rest(self):
        sim = simulate(Scenario(reps=[RepScript()]))
        analyzer = DeadliftRepAnalyzer()
        analyzer.set_gravity(sim.gravity_up_world, GRAVITY_SOURCE_MEASURED)
        peak_cm = -math.inf
        for frame in sim.frames:
            analyzer.observe(DeadliftFrameInput(frame.timestamp, frame.frame_index, frame.points, frame.confidences, frame.bar))
            if math.isfinite(analyzer.rep_signal):
                peak_cm = max(peak_cm, analyzer.rep_signal)
        assert analyzer.rep_signal == pytest.approx(0.0, abs=0.5)
        assert 50.0 < peak_cm < 70.0


class TestCleanFeatures:
    def test_a_clean_rep_measures_no_fault(self):
        _, features, _, _ = _analyse(Scenario())
        for rep in features:
            assert rep.bar_drift_cm < CLEAN_DRIFT_MAX_CM
            assert rep.bar_tilt_cm < CLEAN_TILT_MAX_CM
            assert abs(rep.hip_shift_ratio) < CLEAN_SHIFT_MAX_RATIO
            assert rep.trunk_change_liftoff_knee_deg == pytest.approx(0.0, abs=CLEAN_ANGLE_TOLERANCE_DEG)
            assert rep.lean_back_deg == pytest.approx(0.0, abs=CLEAN_ANGLE_TOLERANCE_DEG)
            assert rep.bar_midfoot_setup_cm == pytest.approx(0.0, abs=0.5)

    def test_a_good_setup_sits_inside_the_models_hip_band(self):
        _, features, _, _ = _analyse(Scenario())
        for rep in features:
            assert rep.setup_hip_band_low_cm <= rep.setup_hip_height_cm <= rep.setup_hip_band_high_cm

    def test_stance_bar_offset_belongs_to_the_first_rep_after_the_stance(self):
        _, features, _, _ = _analyse(Scenario())
        assert math.isfinite(features[0].bar_midfoot_stance_cm)
        assert all(math.isnan(rep.bar_midfoot_stance_cm) for rep in features[1:])

    def test_touch_and_go_reps_have_no_setup(self):
        _, features, _, _ = _analyse(Scenario(reps=[RepScript(floor_hold_s=0.0), RepScript()]))
        assert not features[1].setup_measured
        assert math.isnan(features[1].bar_midfoot_setup_cm)

    def test_the_grip_from_the_session_metadata_is_recorded(self):
        _, features, _, _ = _analyse(Scenario(reps=[RepScript()]), meta={"grip": "mixed"})
        assert features[0].grip == "mixed"

    def test_measured_gravity_keeps_a_tilted_world_from_faking_drift(self):
        tilted = Scenario(world_tilt_deg=3.0, reps=[RepScript()])
        _, measured, _, _ = _analyse(tilted, GRAVITY_SOURCE_MEASURED)
        _, body, _, _ = _analyse(tilted, GRAVITY_SOURCE_BODY)
        assert measured[0].bar_drift_cm < CLEAN_DRIFT_MAX_CM
        assert body[0].bar_drift_cm > TILT_FAKE_DRIFT_MIN_CM
        assert body[0].gravity_source == GRAVITY_SOURCE_BODY

    def test_losing_the_bar_mid_set_never_switches_to_the_wrists(self):
        sim = simulate(Scenario(reps=[RepScript(), RepScript()]))
        analyzer = DeadliftRepAnalyzer()
        analyzer.set_gravity(sim.gravity_up_world, GRAVITY_SOURCE_MEASURED)
        sources = set()
        for index, frame in enumerate(sim.frames):
            # The bar vanishes for the second half of the set.
            bar = frame.bar if index < len(sim.frames) // 2 else None
            analyzer.observe(DeadliftFrameInput(frame.timestamp, frame.frame_index, frame.points, frame.confidences, bar))
            if analyzer.phase != DeadliftPhase.APPROACH:
                sources.add(analyzer.status.bar_source)
        assert sources == {BAR_SOURCE_BAR}


class TestFaultScenarios:
    @pytest.mark.parametrize(
        ("scenario", "fault_type"),
        [
            (Scenario(bar_midfoot_offset_m=0.06, reps=[RepScript()] * 2), "deadlift_bar_position"),
            (Scenario(reps=[RepScript(shoulder_ahead_m=-0.05)] * 2), "deadlift_shoulders_behind"),
            (Scenario(reps=[RepScript(shoulder_ahead_m=0.14)] * 2), "deadlift_setup_hips"),
            # Hips first: the trunk tips forward 15 deg off the floor, the hips out-rising
            # the chest beyond what noise reaches (a milder one waits for a second rep).
            (Scenario(reps=[RepScript(setup_trunk_deg=40.0, knee_pass_trunk_deg=55.0)] * 2), "deadlift_hips_shoot"),
            (Scenario(reps=[RepScript(bar_drift_m=0.06)] * 2), "deadlift_bar_drift"),
            (Scenario(reps=[RepScript(lockout_deficit_deg=14.0)] * 2), "deadlift_lockout"),
            (Scenario(reps=[RepScript(lean_back_deg=13.0)] * 2), "deadlift_lean_back"),
            (Scenario(reps=[RepScript(hip_shift_m=0.05)] * 2), "deadlift_hip_shift"),
            (Scenario(reps=[RepScript(bar_tilt_m=0.06)] * 2), "deadlift_bar_tilt"),
            (Scenario(reps=[RepScript(elbow_bend_deg=30.0)] * 2), "deadlift_bent_arms"),
        ],
        ids=["D1", "D7", "D4", "D2", "D3", "D6", "D5", "D8", "D8b", "D9"],
    )
    def test_each_injected_fault_fires_at_moderate_or_worse_on_every_rep(self, scenario: Scenario, fault_type: str):
        _, features, _, _ = _analyse(scenario)
        verdicts = _judge(features)
        assert len(verdicts) == len(scenario.reps)
        for verdict in verdicts:
            assert verdict.get(fault_type) in ("moderate", "severe")

    def test_a_tilted_bar_fires_on_the_wrist_proxy_at_its_wider_thresholds(self):
        _, features, _, _ = _analyse(Scenario(track_bar=False, reps=[RepScript(bar_tilt_m=0.09)] * 2))
        for faults in _faults(features):
            tilt = next(fault for fault in faults if fault.fault_type == "deadlift_bar_tilt")
            assert tilt.severity.value == "moderate"
            assert tilt.details["bar_source"] == BAR_SOURCE_WRIST_PROXY

    def test_velocity_loss_fires_on_the_slow_rep_only(self):
        scenario = Scenario(reps=[RepScript(pull_s=1.0), RepScript(pull_s=1.0), RepScript(pull_s=1.6)])
        _, features, _, _ = _analyse(scenario)
        verdicts = _judge(features)
        assert "deadlift_velocity_loss" not in verdicts[0]
        assert "deadlift_velocity_loss" not in verdicts[1]
        assert verdicts[2]["deadlift_velocity_loss"] in ("moderate", "severe")

    @pytest.mark.parametrize(
        "scenario",
        [Scenario(), Scenario(keypoint_noise_m=0.006, bar_noise_m=0.003), Scenario(world_tilt_deg=3.0)],
        ids=["clean", "noisy", "tilted_world"],
    )
    def test_a_clean_set_is_fault_free(self, scenario: Scenario):
        _, features, _, _ = _analyse(scenario)
        assert _judge(features) == [{}] * len(features)


class TestBodiesAndSetups:
    """The setup model is the analyser's, so these vary what it must measure rather
    than assume: segment proportions, and how far in front of the shins the bar
    sits (shin thickness, how hard the shins press the bar)."""

    @pytest.mark.parametrize(
        "athlete",
        [
            SimAthlete(),
            SimAthlete(torso_m=0.58, upper_arm_m=0.28, forearm_m=0.27),
            SimAthlete(femur_m=0.50, tibia_m=0.45),
            SimAthlete(tibia_m=0.38, femur_m=0.40, torso_m=0.47, upper_arm_m=0.27, forearm_m=0.26),
        ],
        ids=["average", "long_torso_short_arms", "long_femurs", "short"],
    )
    @pytest.mark.parametrize("shin_bar_m", [0.03, 0.05, 0.07])
    def test_a_clean_set_is_fault_free_whatever_the_body_and_shin_contact(self, athlete: SimAthlete, shin_bar_m: float):
        _, features, _, _ = _analyse(Scenario(athlete=athlete, shin_bar_m=shin_bar_m, reps=[RepScript()] * 2))
        assert len(features) == 2
        assert _judge(features) == [{}, {}]
        for rep in features:
            assert rep.trunk_change_liftoff_knee_deg == pytest.approx(0.0, abs=CLEAN_ANGLE_TOLERANCE_DEG)
            assert rep.setup_hip_band_low_cm <= rep.setup_hip_height_cm <= rep.setup_hip_band_high_cm


class TestPlatformNoise:
    """Keypoints as noisy as the platform's own, i.i.d. and correlated (the slow
    wander a Kalman leaves): every rep counted, the stance reached for foot
    guidance, events on time and nothing cued on a clean set."""

    @pytest.mark.parametrize("seed", range(4))
    def test_independent_noise_counts_every_rep_on_time_and_cues_nothing(self, seed: int):
        scenario = Scenario(keypoint_noise_m=PLATFORM_KEYPOINT_NOISE_M, bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed)
        sim, features, _, phases = _analyse(scenario)
        assert len(features) == len(sim.reps)
        assert DeadliftPhase.STANCE in phases
        for expected, measured in zip(sim.reps, features):
            assert measured.liftoff_time == pytest.approx(expected.liftoff_time, abs=MAX_EVENT_ERROR_S)
            assert measured.top_time == pytest.approx(expected.top_time, abs=MAX_EVENT_ERROR_S)
            assert measured.floor_time == pytest.approx(expected.floor_time, abs=MAX_EVENT_ERROR_S)
        assert _cued(features) == [set()] * len(features)

    @pytest.mark.parametrize("seed", range(4))
    def test_correlated_noise_counts_every_rep_and_cues_nothing(self, seed: int):
        scenario = Scenario(bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed)
        sim, features, _, phases = _analyse(scenario, mutate=_correlated_noise(PLATFORM_KEYPOINT_NOISE_M, 0.8, seed))
        assert len(features) == len(sim.reps)
        assert DeadliftPhase.STANCE in phases
        assert _cued(features) == [set()] * len(features)

    def test_noisy_wrist_proxy_counts_every_rep(self):
        sim, features, _, _ = _analyse(Scenario(track_bar=False, keypoint_noise_m=0.01))
        assert len(features) == len(sim.reps)
        assert {f.bar_source for f in features} == {BAR_SOURCE_WRIST_PROXY}


class TestSetupHolds:
    """Grip and rip: the first rep counts however short the setup pause."""

    @pytest.mark.parametrize("hold_s", [0.0, 0.1, 0.2, 0.3])
    @pytest.mark.parametrize("hinge_s", [0.5, 1.0])
    def test_a_pull_without_a_setup_hold_still_counts(self, hold_s: float, hinge_s: float):
        sim, features, _, _ = _analyse(Scenario(setup_hold_s=hold_s, hinge_s=hinge_s))
        assert len(features) == len(sim.reps)
        assert features[0].liftoff_time == pytest.approx(sim.reps[0].liftoff_time, abs=MAX_EVENT_ERROR_S)

    def test_a_grip_and_rip_judges_the_moment_before_liftoff(self):
        _, features, _, _ = _analyse(Scenario(setup_hold_s=0.0, hinge_s=1.0))
        assert features[0].setup_measured
        assert features[0].bar_midfoot_setup_cm == pytest.approx(0.0, abs=1.0)


class TestOcclusionAndDropouts:
    def test_plates_hiding_the_feet_through_the_pull_lose_no_rep(self):
        sim, features, _, _ = _analyse(Scenario(plates_hide_feet_above_m=0.03))
        assert len(features) == len(sim.reps)
        assert _judge(features) == [{}] * len(features)

    def test_feet_hidden_from_the_setup_on_still_count(self):
        def hide_feet_after_the_stance(index: int, frame: SimFrame) -> tuple[np.ndarray, np.ndarray, BarState3D | None]:
            confidences = frame.confidences.copy()
            if frame.timestamp - sim.frames[0].timestamp > 3.5:
                confidences[[CK.LEFT_ANKLE, CK.RIGHT_ANKLE, CK.LEFT_FOOT_INDEX, CK.RIGHT_FOOT_INDEX]] = 0.0
            return frame.points, confidences, frame.bar

        sim = simulate(Scenario())
        _, features, _, _ = _analyse(Scenario(), mutate=hide_feet_after_the_stance)
        assert len(features) == len(sim.reps)

    @pytest.mark.parametrize("probability", [0.2, 0.4])
    def test_random_bar_dropouts_lose_no_rep_and_never_switch_to_the_wrists(self, probability: float):
        sim, features, _, _ = _analyse(Scenario(), mutate=_bar_dropout(probability, seed=2))
        assert len(features) == len(sim.reps)
        assert {f.bar_source for f in features} == {BAR_SOURCE_BAR}

    def test_a_bar_tracked_at_half_the_frame_rate_stays_the_bar(self):
        def every_other(index: int, frame: SimFrame) -> tuple[np.ndarray, np.ndarray, BarState3D | None]:
            return frame.points, frame.confidences, frame.bar if index % 2 == 0 else None

        sim, features, _, _ = _analyse(Scenario(reps=[RepScript(floor_hold_s=0.0)] * 2 + [RepScript()]), mutate=every_other)
        assert len(features) == len(sim.reps)
        assert {f.bar_source for f in features} == {BAR_SOURCE_BAR}


class TestTempo:
    """Events land within 100 ms from a 0.45 s pull to a 5 s grind, on a tracked
    bar's noise: fitted as the change point of the bar's departure from (or
    arrival at) a level, not where it crosses a fixed band."""

    @pytest.mark.parametrize("pull_s", [0.45, 0.8, 3.0, 5.0])
    @pytest.mark.parametrize("seed", range(3))
    def test_events_stay_on_time_from_a_fast_pull_to_a_slow_grind(self, pull_s: float, seed: int):
        scenario = Scenario(
            reps=[RepScript(pull_s=pull_s, lower_s=max(0.4, pull_s * 0.6))] * 2,
            bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed,
        )
        sim, features, _, _ = _analyse(scenario)
        assert len(features) == len(sim.reps)
        for expected, measured in zip(sim.reps, features):
            assert measured.liftoff_time == pytest.approx(expected.liftoff_time, abs=MAX_EVENT_ERROR_S)
            assert measured.knee_pass_time == pytest.approx(expected.knee_pass_time, abs=MAX_EVENT_ERROR_S)
            assert measured.top_time == pytest.approx(expected.top_time, abs=MAX_EVENT_ERROR_S)
            assert measured.floor_time == pytest.approx(expected.floor_time, abs=MAX_EVENT_ERROR_S)


class TestModelFreeKinematics:
    """Poses scripted by trunk angle, not built by the setup model: the analyser's
    measurements against the kinematic truth of the emitted poses, and D2's
    verdicts on pulls the model did not make."""

    @pytest.mark.parametrize(
        "athlete",
        [
            SimAthlete(),
            SimAthlete(torso_m=0.58, upper_arm_m=0.28, forearm_m=0.27),
            SimAthlete(femur_m=0.50, tibia_m=0.45),
            SimAthlete(tibia_m=0.38, femur_m=0.40, torso_m=0.47, upper_arm_m=0.27, forearm_m=0.26),
        ],
        ids=["average", "long_torso_short_arms", "long_femurs", "short"],
    )
    @pytest.mark.parametrize(
        ("setup_deg", "knee_pass_deg"),
        [(65.0, 50.0), (60.0, 60.0), (45.0, 55.0)],
        ids=["chest_rises", "back_angle_held", "hips_first"],
    )
    def test_trunk_change_rise_ratio_and_hip_height_match_the_poses(
        self, athlete: SimAthlete, setup_deg: float, knee_pass_deg: float,
    ):
        script = RepScript(setup_trunk_deg=setup_deg, knee_pass_trunk_deg=knee_pass_deg)
        sim, features, _, _ = _analyse(Scenario(athlete=athlete, reps=[script] * 2))
        assert len(features) == len(sim.reps)
        for truth, rep in zip(sim.reps, features):
            measured_change = rep.trunk_change_liftoff_knee_deg + rep.trunk_change_predicted_deg
            true_change = truth.knee_pass_trunk_deg - truth.liftoff_trunk_deg
            assert measured_change == pytest.approx(true_change, abs=TRUNK_CHANGE_TOLERANCE_DEG)
            assert rep.hip_shoulder_rise_ratio == pytest.approx(truth.hip_rise_m / truth.shoulder_rise_m, abs=RISE_RATIO_TOLERANCE)
            assert rep.setup_hip_height_cm == pytest.approx(truth.setup_hip_height_m * 100.0, abs=HIP_HEIGHT_TOLERANCE_CM)

    @pytest.mark.parametrize("pull_s", [0.6, 1.2, 3.0])
    @pytest.mark.parametrize(
        "athlete",
        [SimAthlete(), SimAthlete(tibia_m=0.38, femur_m=0.40, torso_m=0.47, upper_arm_m=0.27, forearm_m=0.26)],
        ids=["average", "short"],
    )
    @pytest.mark.parametrize(
        ("setup_deg", "knee_pass_deg"),
        [(65.0, 50.0), (60.0, 60.0), (55.0, 65.0), (45.0, 55.0)],
        ids=["chest_rises", "back_angle_held", "tips_forward", "hips_first"],
    )
    def test_rise_ratio_matches_the_poses_at_any_tempo(
        self, athlete: SimAthlete, setup_deg: float, knee_pass_deg: float, pull_s: float,
    ):
        """A fast pull accelerates through the frames before the knee pass: the
        ratio comes from a curve through them, not a line."""
        script = RepScript(setup_trunk_deg=setup_deg, knee_pass_trunk_deg=knee_pass_deg, pull_s=pull_s)
        sim, features, _, _ = _analyse(Scenario(athlete=athlete, reps=[script] * 2))
        assert len(features) == len(sim.reps)
        for truth, rep in zip(sim.reps, features):
            assert rep.hip_shoulder_rise_ratio == pytest.approx(truth.hip_rise_m / truth.shoulder_rise_m, abs=RISE_RATIO_TOLERANCE)

    def test_a_hips_first_pull_is_cued_once_the_set_confirms_it(self):
        """Hips out-rising the chest by ~1.24: within one rep's noise of a held back
        angle, so the first rep waits and the set's second confirms it."""
        script = RepScript(setup_trunk_deg=45.0, knee_pass_trunk_deg=55.0)
        sim, features, _, _ = _analyse(Scenario(reps=[script] * 3))
        assert all(truth.hip_rise_m > truth.shoulder_rise_m for truth in sim.reps)
        assert ["deadlift_hips_shoot" in cued for cued in _cued(features)] == [False, True, True]

    def test_a_held_back_angle_is_no_hips_shoot(self):
        """The model wants the chest to rise off the floor; a lifter holding the
        back angle reads 6-12 deg against it, but the hips never out-rose the
        shoulders, so there is no fault."""
        script = RepScript(setup_trunk_deg=60.0, knee_pass_trunk_deg=60.0)
        sim, features, _, _ = _analyse(Scenario(reps=[script] * 3))
        assert all(truth.hip_rise_m <= truth.shoulder_rise_m for truth in sim.reps)
        assert all("deadlift_hips_shoot" not in verdict for verdict in _judge(features))

    @pytest.mark.parametrize("seed", range(6))
    def test_a_held_back_angle_is_not_cued_at_platform_noise(self, seed: int):
        script = RepScript(setup_trunk_deg=60.0, knee_pass_trunk_deg=60.0)
        scenario = Scenario(reps=[script] * 3, keypoint_noise_m=PLATFORM_KEYPOINT_NOISE_M,
                            bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed)
        _, features, _, _ = _analyse(scenario)
        assert all("deadlift_hips_shoot" not in cued for cued in _cued(features))

    def test_a_held_back_angle_is_rarely_cued_at_correlated_noise(self):
        """Correlated noise scatters one rep's rise ratio by ~0.1 around the held
        angle's 1.0: the set-level check keeps false cues under the VALIDATION.md
        gate of 1 per 10 reps."""
        script = RepScript(setup_trunk_deg=60.0, knee_pass_trunk_deg=60.0)
        cued = reps = 0
        for seed in range(8):
            scenario = Scenario(reps=[script] * 3, bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed)
            _, features, _, _ = _analyse(scenario, mutate=_correlated_noise(PLATFORM_KEYPOINT_NOISE_M, 0.8, seed))
            reps += len(features)
            cued += sum("deadlift_hips_shoot" in rep_cues for rep_cues in _cued(features))
        assert reps == 24
        assert cued <= reps // 10

    def test_a_hips_first_pull_is_cued_on_most_reps_at_platform_noise(self):
        script = RepScript(setup_trunk_deg=45.0, knee_pass_trunk_deg=55.0)
        cued = reps = 0
        for seed in range(4):
            scenario = Scenario(reps=[script] * 3, keypoint_noise_m=PLATFORM_KEYPOINT_NOISE_M,
                                bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed)
            _, features, _, _ = _analyse(scenario)
            reps += len(features)
            cued += sum("deadlift_hips_shoot" in rep_cues for rep_cues in _cued(features))
        assert cued >= reps // 2

    @pytest.mark.parametrize(
        ("first", "then"),
        [((60.0, 60.0), (45.0, 55.0)), ((40.0, 55.0), (65.0, 50.0))],
        ids=["starts_shooting_the_hips", "fixes_the_hips"],
    )
    def test_a_pull_that_changes_mid_set_is_measured_rep_by_rep(
        self, first: tuple[float, float], then: tuple[float, float],
    ):
        reps = [RepScript(setup_trunk_deg=first[0], knee_pass_trunk_deg=first[1])] * 2
        reps += [RepScript(setup_trunk_deg=then[0], knee_pass_trunk_deg=then[1])] * 3
        sim, features, _, _ = _analyse(Scenario(reps=reps))
        assert len(features) == len(sim.reps)
        for truth, rep in zip(sim.reps, features):
            assert rep.hip_shoulder_rise_ratio == pytest.approx(truth.hip_rise_m / truth.shoulder_rise_m, abs=RISE_RATIO_TOLERANCE)

    def test_the_rep_after_the_fix_is_not_cued(self):
        """The demo loop: hips first, the cue, then a pull with the chest rising."""
        reps = [RepScript(setup_trunk_deg=40.0, knee_pass_trunk_deg=55.0)] * 2
        reps += [RepScript(setup_trunk_deg=65.0, knee_pass_trunk_deg=50.0)] * 3
        _, features, _, _ = _analyse(Scenario(reps=reps))
        assert ["deadlift_hips_shoot" in cued for cued in _cued(features)] == [True, True, False, False, False]

    def test_a_pull_whose_chest_rises_is_fault_free(self):
        script = RepScript(setup_trunk_deg=65.0, knee_pass_trunk_deg=50.0)
        _, features, _, _ = _analyse(Scenario(reps=[script] * 2))
        assert all("deadlift_hips_shoot" not in verdict for verdict in _judge(features))

    def test_hips_set_lower_than_the_shins_allow_read_as_low(self):
        script = RepScript(setup_trunk_deg=45.0, knee_pass_trunk_deg=50.0)
        _, features, _, _ = _analyse(Scenario(athlete=SimAthlete(femur_m=0.50, tibia_m=0.45), reps=[script] * 2))
        for rep in features:
            assert rep.setup_hip_height_cm < rep.setup_hip_band_low_cm


class TestSetupModelGeometry:
    """The setup model's solves, checked in the simulator's forward kinematics: a
    pose built from the model's trunk angle must meet the constraints the model
    claims (shins on the bar, a real knee bend), whatever the shin contact."""

    @pytest.mark.parametrize("shin_bar_m", [0.03, 0.05, 0.07])
    def test_a_model_setup_puts_the_shins_on_the_bar_with_bent_knees(self, shin_bar_m: float):
        sim = simulate(Scenario(shin_bar_m=shin_bar_m, reps=[RepScript()]))
        setup = next(frame for frame in sim.frames if frame.timestamp >= sim.reps[0].liftoff_time - 1e-6)
        points = setup.points
        ankle = (points[CK.LEFT_ANKLE] + points[CK.RIGHT_ANKLE]) / 2.0
        knee = (points[CK.LEFT_KNEE] + points[CK.RIGHT_KNEE]) / 2.0
        hip = (points[CK.LEFT_HIP] + points[CK.RIGHT_HIP]) / 2.0
        # The sagittal plane is the world's (Y, Z) here: no tilt, X is lateral.
        shin = (knee - ankle)[1:]
        thigh = (hip - knee)[1:]
        to_bar = (setup.bar.centre - ankle)[1:]
        shin_to_bar_m = abs(shin[0] * to_bar[1] - shin[1] * to_bar[0]) / float(np.linalg.norm(shin))
        knee_flexion_deg = math.degrees(math.acos(
            float(np.dot(shin, thigh)) / (float(np.linalg.norm(shin)) * float(np.linalg.norm(thigh))),
        ))
        assert shin_to_bar_m == pytest.approx(shin_bar_m, abs=0.002)
        assert MIN_SETUP_KNEE_FLEXION_DEG <= knee_flexion_deg <= MAX_SETUP_KNEE_FLEXION_DEG


class TestTouchAndGoUnderNoise:
    """No dead stop between touch-and-go reps: a check that one noisy frame could
    defeat would swallow every rep after it."""

    @pytest.mark.parametrize("seed", range(4))
    def test_noisy_bar_counts_every_touch_and_go_rep(self, seed: int):
        reps = [RepScript(floor_hold_s=0.0, pull_s=1.5, lower_s=0.8)] * 4 + [RepScript(pull_s=1.5, lower_s=0.8)]
        sim, features, _, _ = _analyse(Scenario(reps=reps, bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed))
        assert len(features) == len(sim.reps)
        assert [f.touch_and_go for f in features] == [False, True, True, True, True]

    @pytest.mark.parametrize("seed", range(4))
    def test_a_very_noisy_bar_still_counts_every_rep(self, seed: int):
        """At 5 mm a low point can read as a dead stop: the next rep is then a quick
        re-pull rather than a touch-and-go, but no rep is lost."""
        reps = [RepScript(floor_hold_s=0.0, pull_s=1.5, lower_s=0.8)] * 4 + [RepScript(pull_s=1.5, lower_s=0.8)]
        sim, features, _, _ = _analyse(Scenario(reps=reps, bar_noise_m=0.005, seed=seed))
        assert len(features) == len(sim.reps)

    def test_ten_touch_and_go_reps_at_platform_noise(self):
        reps = [RepScript(floor_hold_s=0.0, pull_s=1.3, lower_s=0.9)] * 9 + [RepScript(pull_s=1.3, lower_s=0.9)]
        scenario = Scenario(reps=reps, bar_noise_m=TRACKED_BAR_NOISE_M, seed=5)
        sim, features, _, _ = _analyse(scenario, mutate=_correlated_noise(PLATFORM_KEYPOINT_NOISE_M, 0.8, 5))
        assert len(features) == len(sim.reps)

    @pytest.mark.parametrize("keypoint_noise_m", [0.01, 0.02])
    def test_noisy_wrist_proxy_counts_every_touch_and_go_rep(self, keypoint_noise_m: float):
        reps = [RepScript(floor_hold_s=0.0)] * 4 + [RepScript()]
        sim, features, _, _ = _analyse(Scenario(track_bar=False, reps=reps, keypoint_noise_m=keypoint_noise_m))
        assert len(features) == len(sim.reps)

    @pytest.mark.parametrize("pull_s", [0.5, 0.6])
    @pytest.mark.parametrize("seed", range(4))
    def test_fast_touch_and_go_on_the_wrist_proxy_under_correlated_noise(self, pull_s: float, seed: int):
        """The wrists leave the knees' height within a few frames of a fast low
        point, and the noise widens the velocity window into the descent."""
        reps = [RepScript(floor_hold_s=0.0, pull_s=pull_s, lower_s=0.4, top_hold_s=0.2)] * 5 + [RepScript(pull_s=pull_s, lower_s=0.4)]
        scenario = Scenario(track_bar=False, reps=reps, seed=seed)
        sim, features, _, _ = _analyse(scenario, mutate=_correlated_noise(PLATFORM_KEYPOINT_NOISE_M, 0.8, seed))
        assert len(features) == len(sim.reps)

    @pytest.mark.parametrize("seed", range(4))
    def test_fast_touch_and_go_on_the_wrist_proxy_under_independent_noise(self, seed: int):
        reps = [RepScript(floor_hold_s=0.0, pull_s=0.5, lower_s=0.4, top_hold_s=0.2)] * 5 + [RepScript(pull_s=0.5, lower_s=0.4)]
        sim, features, _, _ = _analyse(Scenario(track_bar=False, reps=reps, keypoint_noise_m=0.015, seed=seed))
        assert len(features) == len(sim.reps)


class TestWristProxyAtPlatformNoise:
    """The wrist proxy has no bar axis: its left-right is locked per set, and its
    tilt (the hands' noise carried out to the hubs) is cued only when severe."""

    @pytest.mark.parametrize("seed", range(4))
    def test_clean_reps_cue_nothing_under_correlated_noise(self, seed: int):
        athletes = [SimAthlete(), SimAthlete(torso_m=0.58, upper_arm_m=0.28, forearm_m=0.27),
                    SimAthlete(femur_m=0.50, tibia_m=0.45), SimAthlete(tibia_m=0.38, femur_m=0.40, torso_m=0.47)]
        scenario = Scenario(athlete=athletes[seed], track_bar=False, seed=seed, approach_s=3.0)
        sim, features, _, _ = _analyse(scenario, mutate=_correlated_noise(PLATFORM_KEYPOINT_NOISE_M, 0.8, seed + 100))
        assert len(features) == len(sim.reps)
        assert all("deadlift_hip_shift" not in cued for cued in _cued(features))

    def test_a_clean_hip_line_rarely_reads_a_shift(self):
        shifted = reps = 0
        for seed in range(8):
            scenario = Scenario(track_bar=False, seed=seed)
            _, features, _, _ = _analyse(scenario, mutate=_correlated_noise(PLATFORM_KEYPOINT_NOISE_M, 0.8, seed))
            reps += len(features)
            shifted += sum(abs(f.hip_shift_ratio) >= FAULTS.deadlift_hip_shift.moderate for f in features)
        assert reps == 24
        assert shifted <= reps // 10


class TestHipShiftAlongTheLifter:
    """D8 is measured along the lifter's own left-right (hips and ankles, locked
    for the set), not the bar's axis: the hips travel ~45 cm forward in a pull,
    which a stance a few degrees off square to the bar would read as sideways."""

    @pytest.mark.parametrize("track_bar", [True, False], ids=["tracked", "proxy"])
    @pytest.mark.parametrize("degrees", [5.0, 8.0])
    def test_a_stance_off_square_to_the_bar_is_no_hip_shift(self, degrees: float, track_bar: bool):
        scenario = Scenario(track_bar=track_bar)
        sim = simulate(scenario)
        _, features, _, _ = _analyse(scenario, mutate=_turned(sim, degrees, LOWER_BODY))
        assert all(abs(f.hip_shift_ratio) < CLEAN_SHIFT_MAX_RATIO for f in features)

    @pytest.mark.parametrize("seed", range(4))
    def test_an_off_square_stance_reads_as_square_under_correlated_noise(self, seed: int):
        """The same noise on a square stance and one 8 deg off: the same hip shift
        (the noise floor itself is the clean-set tests')."""
        scenario = Scenario(seed=seed, bar_noise_m=TRACKED_BAR_NOISE_M)
        sim = simulate(scenario)
        _, square, _, _ = _analyse(scenario, mutate=_correlated_noise(PLATFORM_KEYPOINT_NOISE_M, 0.8, seed))
        noise = _correlated_noise(PLATFORM_KEYPOINT_NOISE_M, 0.8, seed)
        _, turned, _, _ = _analyse(scenario, mutate=_turned(sim, 8.0, LOWER_BODY, noise))
        assert len(turned) == len(square) == len(sim.reps)
        for square_rep, turned_rep in zip(square, turned):
            assert turned_rep.hip_shift_ratio == pytest.approx(square_rep.hip_shift_ratio, abs=STANCE_LEAK_MAX_RATIO)

    def test_a_whole_lifter_turned_off_square_to_the_bar_is_no_hip_shift(self):
        scenario = Scenario()
        sim = simulate(scenario)
        _, features, _, _ = _analyse(scenario, mutate=_turned(sim, 10.0, tuple(range(len(sim.frames[0].points)))))
        assert all(abs(f.hip_shift_ratio) < CLEAN_SHIFT_MAX_RATIO for f in features)

    def test_a_staggered_stance_is_no_hip_shift(self):
        """The hips stay square to the bar with one foot ahead: the ankle line
        turns 13 deg, the hips' path does not."""
        _, features, _, _ = _analyse(Scenario(), mutate=_staggered(STAGGER_M))
        assert all(abs(f.hip_shift_ratio) < CLEAN_SHIFT_MAX_RATIO for f in features)

    @pytest.mark.parametrize("seed", range(4))
    def test_a_staggered_stance_reads_as_square_under_correlated_noise(self, seed: int):
        scenario = Scenario(seed=seed, bar_noise_m=TRACKED_BAR_NOISE_M)
        _, square, _, _ = _analyse(scenario, mutate=_correlated_noise(PLATFORM_KEYPOINT_NOISE_M, 0.8, seed))
        noise = _correlated_noise(PLATFORM_KEYPOINT_NOISE_M, 0.8, seed)
        _, staggered, _, _ = _analyse(scenario, mutate=_staggered(STAGGER_M, noise))
        assert len(staggered) == len(square)
        for square_rep, staggered_rep in zip(square, staggered):
            assert staggered_rep.hip_shift_ratio == pytest.approx(square_rep.hip_shift_ratio, abs=STANCE_LEAK_MAX_RATIO)

    def test_clean_sets_rarely_cue_hip_shift_under_correlated_noise(self):
        """The noise floor: the hip line's heading from a few seconds of noisy
        frames, carried over ~45 cm of forward travel. Within the VALIDATION.md
        gate of 1 false correction per 10 reps."""
        cued = reps = 0
        for seed in range(8):
            scenario = Scenario(seed=seed, bar_noise_m=TRACKED_BAR_NOISE_M)
            _, features, _, _ = _analyse(scenario, mutate=_correlated_noise(PLATFORM_KEYPOINT_NOISE_M, 0.8, seed))
            reps += len(features)
            cued += sum("deadlift_hip_shift" in rep_cues for rep_cues in _cued(features))
        assert reps == 24
        assert cued <= reps // 10

    @pytest.mark.parametrize("seed", range(4))
    def test_standing_turned_then_squaring_up_is_no_hip_shift(self, seed: int):
        """Turned toward something for 2 s, then squared up in place: a turn in
        place reads settled, but the frames before it no longer lock the axis."""
        scenario = Scenario(stance_s=3.0, seed=seed, bar_noise_m=TRACKED_BAR_NOISE_M)
        sim = simulate(scenario)
        stance_start = scenario.start_time + scenario.approach_s
        noise = _correlated_noise(0.015, 0.8, seed)
        mutate = _turned_then_squared(sim, 20.0, stance_start + 2.0, stance_start + 2.5, noise)
        _, features, _, _ = _analyse(scenario, mutate=mutate)
        assert len(features) == len(sim.reps)
        assert all("deadlift_hip_shift" not in cued for cued in _cued(features))

    def test_a_real_shift_is_still_seen_off_square(self):
        scenario = Scenario(reps=[RepScript(hip_shift_m=0.05)] * 2)
        sim = simulate(scenario)
        _, features, _, _ = _analyse(scenario, mutate=_turned(sim, 8.0, LOWER_BODY))
        assert all(verdict.get("deadlift_hip_shift") in ("moderate", "severe") for verdict in _judge(features))


class TestReSetupAtTheFloor:
    """A lifter who re-sets at the floor may move the feet: the next rep is judged
    against where they now stand."""

    @pytest.mark.parametrize("keypoint_noise_m", [0.0, PLATFORM_KEYPOINT_NOISE_M])
    def test_feet_moved_at_the_floor_are_judged_where_they_now_stand(self, keypoint_noise_m: float):
        scenario = Scenario(
            bar_midfoot_offset_m=0.06, reps=[RepScript(floor_hold_s=3.0), RepScript(), RepScript()],
            keypoint_noise_m=keypoint_noise_m, bar_noise_m=TRACKED_BAR_NOISE_M,
        )
        sim = simulate(scenario)
        mutate = _stepped_closer_at_the_floor(sim.reps[0].floor_time, FEET_MOVED_AT_THE_FLOOR_M)
        _, features, _, _ = _analyse(scenario, mutate=mutate)
        assert len(features) == len(sim.reps)
        assert features[0].bar_midfoot_setup_cm == pytest.approx(6.0, abs=1.0)
        assert "deadlift_bar_position" in _cued(features)[0]
        for later in features[1:]:
            assert later.bar_midfoot_setup_cm == pytest.approx(1.0, abs=1.0)
        assert all("deadlift_bar_position" not in cued for cued in _cued(features)[1:])

    def test_feet_hidden_at_the_floor_keep_the_stance_lock(self):
        def hide_feet_after_the_first_rep(index: int, frame: SimFrame) -> tuple[np.ndarray, np.ndarray, BarState3D | None]:
            confidences = frame.confidences.copy()
            if frame.timestamp >= floor_time:
                confidences[[CK.LEFT_ANKLE, CK.RIGHT_ANKLE, CK.LEFT_FOOT_INDEX, CK.RIGHT_FOOT_INDEX]] = 0.0
            return frame.points, confidences, frame.bar

        scenario = Scenario(bar_midfoot_offset_m=0.06, reps=[RepScript()] * 3)
        floor_time = simulate(scenario).reps[0].floor_time
        _, features, _, _ = _analyse(scenario, mutate=hide_feet_after_the_first_rep)
        assert all(f.bar_midfoot_setup_cm == pytest.approx(6.0, abs=1.0) for f in features)


class TestSetBehaviour:
    @pytest.mark.parametrize("keypoint_noise_m", [0.0, PLATFORM_KEYPOINT_NOISE_M])
    def test_standing_up_between_reps_re_judges_each_setup(self, keypoint_noise_m: float):
        scenario = Scenario(stand_between_reps_s=1.5, keypoint_noise_m=keypoint_noise_m, bar_noise_m=TRACKED_BAR_NOISE_M)
        sim, features, _, _ = _analyse(scenario)
        assert len(features) == len(sim.reps)
        assert all(f.setup_measured for f in features)
        assert _cued(features) == [set()] * len(features)

    def test_a_sticking_point_is_ground_through_not_a_failed_rep(self):
        scenario = Scenario(reps=[RepScript(pull_s=2.0, stall_fraction=0.35, stall_s=0.8)] * 3, bar_noise_m=TRACKED_BAR_NOISE_M)
        sim, features, analyzer, _ = _analyse(scenario)
        assert len(features) == len(sim.reps)
        assert analyzer.failed_reps == 0
        for expected, measured in zip(sim.reps, features):
            assert measured.top_time == pytest.approx(expected.top_time, abs=MAX_EVENT_ERROR_S)

    def test_a_stall_at_the_knees_that_comes_back_down_is_a_failed_rep(self):
        scenario = Scenario(reps=[RepScript(), RepScript(fail_rise_m=0.28), RepScript()])
        _, features, analyzer, _ = _analyse(scenario)
        assert len(features) == 2
        assert analyzer.failed_reps == 1

    @pytest.mark.parametrize("stance_s", [0.5, 1.5], ids=["no_standing_reference", "standing_reference"])
    @pytest.mark.parametrize("seed", range(3))
    def test_a_hitch_near_the_top_is_not_the_top(self, seed: int, stance_s: float):
        """A still bar with the trunk upright enough, a few cm short of the top
        (or anywhere, with no standing height to expect the top at), reads as a
        top until the bar rises on past it."""
        reps = [RepScript(pull_s=3.0, stall_fraction=0.9, stall_s=0.8)] * 3
        scenario = Scenario(reps=reps, approach_s=stance_s, stance_s=stance_s, keypoint_noise_m=0.01,
                            bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed)
        sim, features, analyzer, _ = _analyse(scenario)
        assert len(features) == len(sim.reps)
        assert analyzer.failed_reps == 0
        for expected, measured in zip(sim.reps, features):
            assert measured.top_time == pytest.approx(expected.top_time, abs=MAX_EVENT_ERROR_S)

    @pytest.mark.parametrize("stall_s", [0.4, 0.8])
    @pytest.mark.parametrize("stall_fraction", [0.94, 0.96])
    def test_a_grind_a_few_cm_short_of_lockout_is_not_the_top(self, stall_fraction: float, stall_s: float):
        """Stuck 2-4 cm short of lockout, then finished: the top is the lockout,
        and the stall's frames do not read as a short lockout (D6)."""
        reps = [RepScript(pull_s=2.5, stall_fraction=stall_fraction, stall_s=stall_s)] * 3
        sim, features, analyzer, _ = _analyse(Scenario(reps=reps, bar_noise_m=TRACKED_BAR_NOISE_M))
        assert len(features) == len(sim.reps)
        for expected, measured in zip(sim.reps, features):
            assert measured.top_time == pytest.approx(expected.top_time, abs=MAX_EVENT_ERROR_S)
        assert all("deadlift_lockout" not in cued for cued in _cued(features))

    def test_a_lean_back_from_mid_thigh_with_the_knees_hidden_is_a_failed_rep(self):
        """Knees not measured are not known to be straight: no lockout."""
        scenario = Scenario(reps=[RepScript(), RepScript(fail_rise_m=0.35, lean_back_deg=40.0), RepScript()])

        def hide_knees_off_the_floor(index: int, frame: SimFrame) -> tuple[np.ndarray, np.ndarray, BarState3D | None]:
            confidences = frame.confidences.copy()
            bar_up = -(frame.bar.left_end_m[1] + frame.bar.right_end_m[1]) / 2.0
            if bar_up - floor_bar_up > KNEES_HIDDEN_ABOVE_M:
                confidences[[CK.LEFT_KNEE, CK.RIGHT_KNEE]] = 0.0
            return frame.points, confidences, frame.bar

        first = simulate(scenario).frames[0].bar
        floor_bar_up = -(first.left_end_m[1] + first.right_end_m[1]) / 2.0
        _, features, analyzer, _ = _analyse(scenario, mutate=hide_knees_off_the_floor)
        assert len(features) == 2
        assert analyzer.failed_reps == 1

    def test_a_failed_pull_leaning_back_from_mid_thigh_is_a_failed_rep(self):
        """Leaning back with the knees still bent ~60 deg is no lockout."""
        scenario = Scenario(reps=[RepScript(), RepScript(fail_rise_m=0.35, lean_back_deg=40.0), RepScript()])
        _, features, analyzer, _ = _analyse(scenario)
        assert len(features) == 2
        assert analyzer.failed_reps == 1

    @pytest.mark.parametrize("lean_back_deg", [25.0, 40.0])
    def test_an_over_extended_lockout_is_counted_and_judged(self, lean_back_deg: float):
        _, features, analyzer, _ = _analyse(Scenario(reps=[RepScript(lean_back_deg=lean_back_deg)] * 2))
        assert len(features) == 2
        assert analyzer.failed_reps == 0
        assert all(verdict.get("deadlift_lean_back") == "severe" for verdict in _judge(features))

    def test_feet_hidden_at_setup_still_judge_the_bar_over_midfoot(self):
        def hide_feet_from_the_setup(index: int, frame: SimFrame) -> tuple[np.ndarray, np.ndarray, BarState3D | None]:
            confidences = frame.confidences.copy()
            if frame.timestamp - start_s > 3.5:
                confidences[[CK.LEFT_ANKLE, CK.RIGHT_ANKLE, CK.LEFT_FOOT_INDEX, CK.RIGHT_FOOT_INDEX]] = 0.0
            return frame.points, confidences, frame.bar

        scenario = Scenario(bar_midfoot_offset_m=0.06, reps=[RepScript()] * 2)
        start_s = scenario.start_time
        _, features, _, _ = _analyse(scenario, mutate=hide_feet_from_the_setup)
        assert all(f.bar_midfoot_setup_cm == pytest.approx(6.0, abs=1.0) for f in features)

    def test_the_live_offset_is_a_one_second_median(self):
        """One noisy frame must not arm the foot guidance."""
        sim = simulate(Scenario(keypoint_noise_m=PLATFORM_KEYPOINT_NOISE_M, bar_noise_m=TRACKED_BAR_NOISE_M))
        analyzer = DeadliftRepAnalyzer()
        analyzer.set_gravity(sim.gravity_up_world, GRAVITY_SOURCE_MEASURED)
        live = []
        for frame in sim.frames:
            if frame.timestamp >= sim.reps[0].liftoff_time:
                break
            analyzer.observe(DeadliftFrameInput(frame.timestamp, frame.frame_index, frame.points, frame.confidences, frame.bar))
            if analyzer.status.phase == DeadliftPhase.STANCE and math.isfinite(analyzer.status.bar_midfoot_live_cm):
                live.append(analyzer.status.bar_midfoot_live_cm)
        assert live
        assert max(abs(value) for value in live) <= BAR_MIDFOOT_GUIDANCE_TOLERANCE_CM

    @pytest.mark.parametrize("seed", range(6))
    def test_a_bar_over_midfoot_never_arms_the_guidance_under_correlated_noise(self, seed: int):
        sim = simulate(Scenario(reps=[RepScript()], bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed, stance_s=3.0))
        noise = _correlated_noise(PLATFORM_KEYPOINT_NOISE_M, 0.8, seed)
        analyzer = DeadliftRepAnalyzer()
        analyzer.set_gravity(sim.gravity_up_world, GRAVITY_SOURCE_MEASURED)
        live = []
        for index, frame in enumerate(sim.frames):
            points, confidences, bar = noise(index, frame)
            analyzer.observe(DeadliftFrameInput(frame.timestamp, frame.frame_index, points, confidences, bar))
            if math.isfinite(analyzer.status.bar_midfoot_live_cm):
                live.append(analyzer.status.bar_midfoot_live_cm)
        assert live
        assert max(abs(value) for value in live) <= BAR_MIDFOOT_GUIDANCE_ARM_CM

    def test_each_set_guides_its_own_stance(self):
        """The live offset speaks before each set's first rep, whatever the
        session's rep count."""
        sim = simulate(Scenario(bar_midfoot_offset_m=0.06, reps=[RepScript()]))
        analyzer = DeadliftRepAnalyzer()
        analyzer.set_gravity(sim.gravity_up_world, GRAVITY_SOURCE_MEASURED)
        live_by_set: list[list[float]] = []
        for set_index in range(2):
            analyzer.reset_set()
            offset_s = set_index * (sim.frames[-1].timestamp - sim.frames[0].timestamp + 1.0)
            live = []
            for frame in sim.frames:
                analyzer.observe(DeadliftFrameInput(
                    frame.timestamp + offset_s, frame.frame_index, frame.points, frame.confidences, frame.bar,
                ))
                if math.isfinite(analyzer.status.bar_midfoot_live_cm):
                    live.append(analyzer.status.bar_midfoot_live_cm)
            live_by_set.append(live)
        assert analyzer.rep_count == 2
        assert all(live and np.median(live) == pytest.approx(6.0, abs=0.5) for live in live_by_set)

    @pytest.mark.parametrize("seed", range(4))
    def test_standing_references_survive_platform_noise(self, seed: int):
        scenario = Scenario(keypoint_noise_m=PLATFORM_KEYPOINT_NOISE_M, bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed)
        _, features, _, _ = _analyse(scenario)
        assert all(math.isfinite(f.lean_back_deg) for f in features)

