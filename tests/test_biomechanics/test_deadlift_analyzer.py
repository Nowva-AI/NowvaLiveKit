"""Deadlift rep analyser on simulated sets (docs/deadlift/PLAN.md §2.3-2.5, J2/J3):
every rep counted exactly once, events on time, each injected fault detected by
its rule and a clean set fault-free."""

from __future__ import annotations

import math

import numpy as np
import pytest

from biomechanics.config import BiomechanicsConfig
from biomechanics.deadlift.analyzer import DeadliftFrameInput, DeadliftRepAnalyzer
from biomechanics.deadlift.session_reference import DeadliftSessionReference
from biomechanics.deadlift.simulator import RepScript, Scenario, SimAthlete, SimulatedSet, simulate
from biomechanics.deadlift.types import (
    BAR_SOURCE_BAR,
    BAR_SOURCE_WRIST_PROXY,
    GRAVITY_SOURCE_BODY,
    GRAVITY_SOURCE_MEASURED,
    DeadliftPhase,
    DeadliftRepFeatures,
)
from biomechanics.faults.rule_engine import RuleEngine
from biomechanics.profiles.deadlift import DeadliftProfile
from biomechanics.utils.geometry import WORLD_UP
from biomechanics.utils.types import JointAngles

# PLAN.md §1 demo gate: event timing median error <= 100 ms; held here per event.
MAX_EVENT_ERROR_S = 0.1
CLEAN_DRIFT_MAX_CM = 1.0
CLEAN_TILT_MAX_CM = 1.0
CLEAN_SHIFT_MAX_RATIO = 0.03
CLEAN_ANGLE_TOLERANCE_DEG = 2.0
# 3 deg of tilted world vertical over the pull fakes this much drift without measured gravity.
TILT_FAKE_DRIFT_MIN_CM = 2.0


def _analyse(
    scenario: Scenario,
    gravity_source: str = GRAVITY_SOURCE_MEASURED,
    meta: dict | None = None,
) -> tuple[SimulatedSet, list[DeadliftRepFeatures], DeadliftRepAnalyzer, list[DeadliftPhase]]:
    sim = simulate(scenario)
    analyzer = DeadliftRepAnalyzer(BiomechanicsConfig().deadlift)
    up = sim.gravity_up_world if gravity_source == GRAVITY_SOURCE_MEASURED else np.array(WORLD_UP)
    analyzer.set_gravity(up, gravity_source)
    if meta is not None:
        analyzer.set_session_meta(meta)
    features: list[DeadliftRepFeatures] = []
    phases: list[DeadliftPhase] = []
    for frame in sim.frames:
        analyzer.observe(DeadliftFrameInput(
            frame.timestamp, frame.frame_index, frame.points, frame.confidences, frame.bar,
        ))
        if not phases or phases[-1] != analyzer.phase:
            phases.append(analyzer.phase)
        while analyzer.take_completed_rep() is not None:
            features.append(analyzer.finish_rep())
    return sim, features, analyzer, phases


def _judge(features: list[DeadliftRepFeatures]) -> list[dict[str, str]]:
    """Each rep's faults as {fault_type: severity}, judged as the pipeline does."""
    engine = RuleEngine(
        rules=DeadliftProfile().create_fault_rules(BiomechanicsConfig()),
        reference=DeadliftSessionReference(),
        capture_mode="triangulated",
    )
    verdicts = []
    for rep_features in features:
        faults = engine.finish_rep(JointAngles(timestamp=rep_features.top_time), rep_features.rep_number, rep_features)
        verdicts.append({fault.fault_type: fault.severity.value for fault in faults})
    return verdicts


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
        for frame in sim.frames:
            analyzer.observe(DeadliftFrameInput(frame.timestamp, frame.frame_index, frame.points, frame.confidences, frame.bar))
            live_by_phase.setdefault(analyzer.status.phase, []).append(analyzer.status.bar_midfoot_live_cm)
        assert all(math.isfinite(value) for value in live_by_phase[DeadliftPhase.STANCE])
        assert np.median(live_by_phase[DeadliftPhase.STANCE]) == pytest.approx(8.0, abs=0.5)
        assert all(math.isnan(value) for value in live_by_phase[DeadliftPhase.PULL])

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
            (Scenario(reps=[RepScript(hips_shoot_deg=10.0)] * 2), "deadlift_hips_shoot"),
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
