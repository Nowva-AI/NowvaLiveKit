"""The deadlift through the real BiomechanicsPipeline (docs/deadlift/PLAN.md §3.3, §5.2):
simulated triangulated frames from a fake provider, bar states keyed by capture
time, and the guarantees that keep the squat's state untouched."""

from __future__ import annotations

import math
import types

import numpy as np
import pytest

from biomechanics.config import BiomechanicsConfig
from biomechanics.deadlift.analyzer import DeadliftRepAnalyzer
from biomechanics.deadlift.rule_base import TIER_RANK
from biomechanics.deadlift.session_reference import DeadliftSessionReference
from biomechanics.deadlift.simulator import RepScript, Scenario, SimAthlete, simulate
from biomechanics.deadlift.types import GRAVITY_SOURCE_MEASURED, DeadliftPhase
from biomechanics.faults.session_reference import SessionReference
from biomechanics.profiles.deadlift import DeadliftProfile, GatedDeadliftProfile
from biomechanics.utils.types import CocoKeypoints as CK
from biomechanics.utils.types import PipelineFrame, Skeleton3D

DEADLIFT = "Barbell Conventional Deadlift"
SQUAT = "Barbell Back Squat"
FRAME_DT_S = 1.0 / 30.0
DEPTH_TARGET_RATIO = 0.05
# The triangulator's confidence contract (utils/keypoint_kalman.py): std = 0.02 * sqrt(1/c - 1).
TRIANGULATION_STD_SCALE_M = 0.02
# A top lost from view is dated within this (test_deadlift_analyzer.py's bound).
GAP_TOP_MAX_ERROR_S = 0.15
# VALIDATION.md's gate: 1 false correction per 10 reps.
FALSE_CUE_GATE_RATIO = 0.1
# Correlated triangulation noise: AR(0.8) at the platform's 1.5 cm, a draw where
# one knee hidden across slow pulls' lost tops put their climb in the window.
AR_RHO = 0.8
CORRELATED_NOISE_M = 0.015
ONE_KNEE_NOISE_SEED = 16000
BODIES = (
    SimAthlete(),
    SimAthlete(tibia_m=0.38, femur_m=0.40, torso_m=0.47, upper_arm_m=0.27, forearm_m=0.26,
               hip_half_width_m=0.10, stance_half_width_m=0.11, shoulder_half_width_m=0.17, grip_half_width_m=0.24),
    SimAthlete(tibia_m=0.48, femur_m=0.50, torso_m=0.57, upper_arm_m=0.33, forearm_m=0.31,
               hip_half_width_m=0.14, stance_half_width_m=0.15, shoulder_half_width_m=0.21, grip_half_width_m=0.28),
    SimAthlete(hip_half_width_m=0.095, stance_half_width_m=0.17, grip_half_width_m=0.29),
    SimAthlete(tibia_m=0.40, femur_m=0.52, torso_m=0.48),
)


class _FakeClock:
    def __init__(self, start_s: float) -> None:
        self.now_s = start_s

    def time(self) -> float:
        return self.now_s

    def perf_counter(self) -> float:
        return self.now_s


class _FakeProvider:
    """Stands in for MultiCameraPoseProvider: world-frame skeletons on the fake clock."""

    def __init__(self, clock: _FakeClock) -> None:
        self._clock = clock
        self._queue: list[tuple[np.ndarray, np.ndarray]] = []
        self.sequence = 0
        self.frame = np.zeros((72, 128, 3), dtype=np.uint8)
        self.last_capture_timestamp = math.nan
        self.last_synced_frames = None
        # Uncalibrated: a bar tracker built on this provider reports no bar.
        self.calibration = None

    def push(self, points: np.ndarray, confidences: np.ndarray) -> None:
        self._queue.append((points, confidences))

    def get_pose(self):
        points, confidences = self._queue.pop(0)
        self.sequence += 1
        self.last_capture_timestamp = self._clock.time()
        skeleton = Skeleton3D.from_numpy(
            points, confidences=confidences, timestamp=self._clock.time(), frame_index=self.sequence,
        )
        return self.frame, None, skeleton

    def reset_temporal_state(self) -> None:
        pass

    def lost_cameras(self) -> list[str]:
        return []

    def release(self) -> None:
        pass


def _pipeline(monkeypatch, exercise: str, start_s: float, coaching_ready: bool = True):
    monkeypatch.setenv("NOWVA_MULTI_CAMERA", "true")
    monkeypatch.setattr(DeadliftProfile, "coaching_ready", coaching_ready)
    from biomechanics import pipeline as pipeline_module

    clock = _FakeClock(start_s)
    monkeypatch.setattr(
        pipeline_module, "time", types.SimpleNamespace(time=clock.time, perf_counter=clock.perf_counter),
    )
    pipe = pipeline_module.BiomechanicsPipeline(BiomechanicsConfig(), exercise_name=exercise, defer_capture=True)
    provider = _FakeProvider(clock)
    pipe._multi_camera_provider = provider
    # Bar states come from the simulator; the tracker needs real camera images.
    pipe._bar_tracker_3d = None
    return pipe, provider, clock


def _run_set(
    pipe, provider, clock, scenario: Scenario, measured_gravity: bool = True, keypoint_noise_m: float = 0.0,
    bar_lost: list[tuple[float, float]] | None = None,
) -> list[PipelineFrame]:
    """keypoint_noise_m: triangulation noise added before the pipeline's Kalman, with
    the confidence the triangulator gives that much error. bar_lost: (start, end)
    simulation times no bar state is pushed in."""
    sim = simulate(scenario)
    if measured_gravity:
        pipe.set_gravity(sim.gravity_up_world, GRAVITY_SOURCE_MEASURED)
    rng = np.random.default_rng(scenario.seed)
    noise_confidence = 1.0 / (1.0 + (keypoint_noise_m / TRIANGULATION_STD_SCALE_M) ** 2)
    results = []
    for frame in sim.frames:
        if keypoint_noise_m > 0.0:
            points = frame.points + rng.normal(0.0, keypoint_noise_m, frame.points.shape)
            confidences = np.minimum(frame.confidences, noise_confidence) * (frame.confidences > 0.0)
            provider.push(points, confidences)
        else:
            provider.push(frame.points, frame.confidences)
        lost = any(start <= frame.timestamp < end for start, end in bar_lost or [])
        if frame.bar is not None and not lost:
            pipe.push_bar_state(frame.bar.model_copy(update={"timestamp": clock.time()}))
        results.append(pipe.process_frame())
        clock.now_s += FRAME_DT_S
    return results


def _run_occluded(
    pipe, provider, clock, scenario: Scenario, hidden: list[tuple[tuple[int, ...], list[tuple[float, float]]]],
    bar_lost: list[tuple[float, float]], noise_m: float = 0.0, noise_seed: int = 0,
) -> list[PipelineFrame]:
    """hidden: keypoints at confidence 0 in their windows; bar_lost: simulation times no
    bar state is pushed in; noise_m: AR(0.8) triangulation noise, with the confidence the
    triangulator gives that much error."""
    sim = simulate(scenario)
    pipe.set_gravity(sim.gravity_up_world, GRAVITY_SOURCE_MEASURED)
    rng = np.random.default_rng(noise_seed)
    confidence_cap = 1.0 / (1.0 + (noise_m / TRIANGULATION_STD_SCALE_M) ** 2)
    error = np.zeros_like(sim.frames[0].points)
    results = []
    for frame in sim.frames:
        confidences = frame.confidences.copy()
        for keypoints, windows in hidden:
            if any(start <= frame.timestamp < end for start, end in windows):
                confidences[list(keypoints)] = 0.0
        error = AR_RHO * error + rng.normal(0.0, noise_m * math.sqrt(1.0 - AR_RHO ** 2), error.shape)
        provider.push(frame.points + error, np.minimum(confidences, confidence_cap) * (confidences > 0.0))
        lost = any(start <= frame.timestamp < end for start, end in bar_lost)
        if frame.bar is not None and not lost:
            pipe.push_bar_state(frame.bar.model_copy(update={"timestamp": clock.time()}))
        results.append(pipe.process_frame())
        clock.now_s += FRAME_DT_S
    return results


def _reps(results: list[PipelineFrame]):
    return [result.rep_data for result in results if result.rep_data is not None]


class TestDeadliftThroughThePipeline:
    def test_each_rep_is_counted_once_with_deadlift_features(self, monkeypatch):
        scenario = Scenario()
        pipe, provider, clock = _pipeline(monkeypatch, DEADLIFT, scenario.start_time)
        reps = _reps(_run_set(pipe, provider, clock, scenario))
        assert [rep.rep_number for rep in reps] == [1, 2, 3]
        assert pipe.rep_count == 3
        assert all(rep.features["dl_schema"] == 1 for rep in reps)
        assert all(math.isnan(rep.max_depth_angle) for rep in reps)
        assert all(rep.depth_target_met for rep in reps)

    def test_features_are_dumped_after_the_rep_is_judged(self, monkeypatch):
        scenario = Scenario(reps=[RepScript(pull_s=1.0), RepScript(pull_s=1.0), RepScript(pull_s=1.6)])
        pipe, provider, clock = _pipeline(monkeypatch, DEADLIFT, scenario.start_time)
        reps = _reps(_run_set(pipe, provider, clock, scenario))
        # Velocity loss is annotated by finish_rep: it reaches the dumped features.
        assert reps[2].features["velocity_loss_pct"] > 30.0
        assert "deadlift_velocity_loss" in [fault.fault_type for fault in reps[2].faults]

    def test_injected_faults_reach_the_frame_and_the_rep(self, monkeypatch):
        scenario = Scenario(reps=[RepScript(bar_drift_m=0.06)] * 2)
        pipe, provider, clock = _pipeline(monkeypatch, DEADLIFT, scenario.start_time)
        results = _run_set(pipe, provider, clock, scenario)
        frame_faults = [fault.fault_type for result in results for fault in result.faults]
        assert frame_faults.count("deadlift_bar_drift") == 2
        assert all(fault.startswith("deadlift_") for fault in frame_faults)
        assert all("deadlift_bar_drift" in [f.fault_type for f in rep.faults] for rep in _reps(results))

    def test_the_bar_is_read_at_the_lagged_frames_capture_time(self, monkeypatch):
        scenario = Scenario(reps=[RepScript()])
        pipe, provider, clock = _pipeline(monkeypatch, DEADLIFT, scenario.start_time)
        results = _run_set(pipe, provider, clock, scenario)
        analysed = [result for result in results if result.skeleton_3d is not None and result.bar_state_3d is not None]
        assert analysed
        for result in analysed:
            assert result.bar_state_3d.timestamp == pytest.approx(result.skeleton_3d.timestamp, abs=1e-6)

    def test_frames_carry_the_deadlift_phase(self, monkeypatch):
        scenario = Scenario(reps=[RepScript()])
        pipe, provider, clock = _pipeline(monkeypatch, DEADLIFT, scenario.start_time)
        results = _run_set(pipe, provider, clock, scenario)
        phases = {result.exercise_status.phase for result in results if result.exercise_status is not None}
        assert {DeadliftPhase.STANCE, DeadliftPhase.SETUP, DeadliftPhase.PULL, DeadliftPhase.FLOOR} <= phases

    def test_the_squat_state_is_never_touched(self, monkeypatch):
        scenario = Scenario()
        pipe, provider, clock = _pipeline(monkeypatch, DEADLIFT, scenario.start_time)
        _run_set(pipe, provider, clock, scenario)
        assert len(pipe._rep_trajectory) == 0
        assert pipe._rep_setup is None
        assert math.isnan(pipe._standing_reference_hip_cm)
        assert pipe._bottom_kpts is None
        assert pipe._standing_kpts is not None

    def test_deadlift_frames_never_feed_the_body_measurement(self, monkeypatch):
        scenario = Scenario()
        pipe, provider, clock = _pipeline(monkeypatch, DEADLIFT, scenario.start_time)
        _run_set(pipe, provider, clock, scenario)
        assert pipe.body_calibration.progress[0] == 0

    def test_the_cue_gate_watches_hips_shoulders_and_wrists(self, monkeypatch):
        scenario = Scenario(reps=[RepScript()])
        pipe, provider, clock = _pipeline(monkeypatch, DEADLIFT, scenario.start_time)
        sim = simulate(scenario)
        confidences = sim.frames[0].confidences.copy()
        from biomechanics.utils.types import CocoKeypoints as CK

        confidences[CK.LEFT_ANKLE] = 0.0
        confidences[CK.LEFT_WRIST] = 0.0
        provider.push(sim.frames[0].points, confidences)
        result = pipe.process_frame()
        assert result.missing_keypoints == ["left_wrist"]


class TestRealWorldInput:
    """Through the pipeline's Kalman: keypoints as noisy as the platform documents,
    and plates hiding the feet for the whole pull."""

    @pytest.mark.parametrize("keypoint_noise_m", [0.01, 0.015, 0.02])
    def test_noisy_triangulation_counts_every_rep_and_reaches_the_stance(self, monkeypatch, keypoint_noise_m: float):
        scenario = Scenario(bar_noise_m=0.003)
        pipe, provider, clock = _pipeline(monkeypatch, DEADLIFT, scenario.start_time)
        results = _run_set(pipe, provider, clock, scenario, keypoint_noise_m=keypoint_noise_m)
        assert len(_reps(results)) == 3
        phases = {result.exercise_status.phase for result in results if result.exercise_status is not None}
        assert DeadliftPhase.STANCE in phases

    def test_plates_hiding_the_feet_through_the_pull_lose_no_rep(self, monkeypatch):
        scenario = Scenario(plates_hide_feet_above_m=0.03)
        pipe, provider, clock = _pipeline(monkeypatch, DEADLIFT, scenario.start_time)
        assert len(_reps(_run_set(pipe, provider, clock, scenario))) == 3

    def test_plates_hiding_the_feet_and_a_top_lost_from_view_cue_no_lockout(self, monkeypatch):
        """With the feet hidden the pipeline marks the pull's legs unmeasured: a top
        lost from view is dated by the bar's own rise (dated at the last frame seen,
        0.27 s early, it judged the climb's bent knees: D6 severe on every rep)."""
        scenario = Scenario(plates_hide_feet_above_m=0.03, bar_noise_m=0.003)
        truth = simulate(scenario).reps
        pipe, provider, clock = _pipeline(monkeypatch, DEADLIFT, scenario.start_time)
        lost = [(rep.top_time - 0.2, rep.top_time + 0.8) for rep in truth]
        reps = _reps(_run_set(pipe, provider, clock, scenario, bar_lost=lost))
        assert len(reps) == len(truth)
        for rep, expected in zip(reps, truth):
            assert rep.features["top_time"] == pytest.approx(expected.top_time, abs=GAP_TOP_MAX_ERROR_S)
            cued = [
                fault for fault in rep.faults
                if TIER_RANK[fault.severity.value] >= TIER_RANK[fault.details["min_tier"]]
            ]
            assert all(fault.fault_type != "deadlift_lockout" for fault in cued)


class TestOccludedLegs:
    """One knee untriangulated: the pipeline marks the legs unmeasured (an
    extrapolated knee) while the analyser measures the knee from the other."""

    def test_one_knee_hidden_across_a_lost_top_cues_no_lockout(self, monkeypatch):
        """The bar's fit alone dates slow pulls' lost tops, early through the noise:
        their windows keep no frame of the climb (5 of 45 cued when the window
        tested for any knee measured instead)."""
        reps = cued = 0
        for body_index, body in enumerate(BODIES):
            for seed in range(3):
                scenario = Scenario(athlete=body, reps=[RepScript(pull_s=2.0)] * 3, bar_noise_m=0.003, seed=seed)
                lost = [(rep.top_time - 0.2, rep.top_time + 0.8) for rep in simulate(scenario).reps]
                pipe, provider, clock = _pipeline(monkeypatch, DEADLIFT, scenario.start_time)
                results = _run_occluded(pipe, provider, clock, scenario, [((CK.LEFT_KNEE,), lost)], lost,
                                        CORRELATED_NOISE_M, ONE_KNEE_NOISE_SEED + 10 * body_index + seed)
                got = _reps(results)
                assert len(got) == len(scenario.reps)
                reps += len(got)
                cued += sum(
                    any(fault.fault_type == "deadlift_lockout"
                        and TIER_RANK[fault.severity.value] >= TIER_RANK[fault.details["min_tier"]]
                        for fault in rep.faults)
                    for rep in got
                )
        assert cued <= FALSE_CUE_GATE_RATIO * reps

    @pytest.mark.parametrize("bar", ["tracked", "wrist_proxy"])
    def test_one_knee_hidden_from_the_setup_with_every_top_lost_keeps_the_session(self, monkeypatch, bar: str):
        """No frame of a rep with its legs measured, and its top lost (the bar, or the
        wrists on the proxy): process_frame() raised out of the analyser, and the
        pipeline's process ended the session."""
        scenario = Scenario(reps=[RepScript(pull_s=1.2)] * 3, bar_noise_m=0.003, track_bar=bar == "tracked")
        sim = simulate(scenario)
        knee_hidden = [(sim.reps[0].liftoff_time - 0.5, sim.frames[-1].timestamp + 1.0)]
        across_the_tops = [(rep.top_time - 0.2, rep.top_time + 0.8) for rep in sim.reps]
        hidden = [((CK.LEFT_KNEE,), knee_hidden)]
        if bar == "wrist_proxy":
            hidden.append(((CK.LEFT_WRIST, CK.RIGHT_WRIST), across_the_tops))
        pipe, provider, clock = _pipeline(monkeypatch, DEADLIFT, scenario.start_time)
        results = _run_occluded(pipe, provider, clock, scenario, hidden, across_the_tops if bar == "tracked" else [])
        assert len(_reps(results)) == len(sim.reps)


class TestGate:
    def test_a_gated_deadlift_counts_and_judges_nothing(self, monkeypatch):
        scenario = Scenario()
        pipe, provider, clock = _pipeline(monkeypatch, DEADLIFT, scenario.start_time, coaching_ready=False)
        assert isinstance(pipe.profile, GatedDeadliftProfile)
        results = _run_set(pipe, provider, clock, scenario)
        assert _reps(results) == []
        assert all(not result.faults for result in results)
        assert pipe.body_calibration.progress[0] == 0


class TestSwitching:
    def test_the_squat_depth_target_survives_a_deadlift_round_trip(self, monkeypatch):
        pipe, _, _ = _pipeline(monkeypatch, SQUAT, 0.0)
        pipe.set_depth_target(DEPTH_TARGET_RATIO)
        pipe.set_exercise(DEADLIFT)
        assert pipe._rule_engine.depth_target_ratio is None
        assert isinstance(pipe._rule_engine.reference, DeadliftSessionReference)
        pipe.set_exercise(SQUAT)
        assert pipe._rule_engine.depth_target_ratio == pytest.approx(DEPTH_TARGET_RATIO, abs=1e-9)
        assert type(pipe._rule_engine.reference) is SessionReference
        assert pipe._rep_analyzer is None

    def test_foot_contact_restarts_into_and_out_of_the_deadlift_only(self, monkeypatch):
        pipe, _, _ = _pipeline(monkeypatch, SQUAT, 0.0)
        resets: list[str] = []
        monkeypatch.setattr(pipe._preik, "reset_world_state", lambda: resets.append(pipe.profile.name))
        pipe.set_exercise("Barbell Romanian Deadlift")
        pipe.set_exercise(SQUAT)
        assert resets == []
        pipe.set_exercise(DEADLIFT)
        pipe.set_exercise(SQUAT)
        assert resets == ["deadlift", "squat"]

    def test_measured_gravity_reaches_a_rebuilt_analyser(self, monkeypatch):
        pipe, _, _ = _pipeline(monkeypatch, SQUAT, 0.0)
        up = np.array([0.0, -0.9986, 0.0523])
        pipe.set_gravity(up, GRAVITY_SOURCE_MEASURED)
        pipe.set_exercise(DEADLIFT)
        assert isinstance(pipe._rep_analyzer, DeadliftRepAnalyzer)
        assert pipe._rep_analyzer.gravity_source == GRAVITY_SOURCE_MEASURED
        assert pipe._rep_analyzer._up == pytest.approx(up / np.linalg.norm(up), abs=1e-9)

    def test_a_calibration_installed_later_brings_the_measured_gravity(self, monkeypatch):
        from biomechanics.deadlift import gravity

        up = np.array([0.0, -0.9986, 0.0523])
        monkeypatch.setattr(gravity, "load_world_up_for_provider", lambda provider: (up, GRAVITY_SOURCE_MEASURED))
        pipe, _, _ = _pipeline(monkeypatch, DEADLIFT, 0.0)
        assert pipe._rep_analyzer.gravity_source != GRAVITY_SOURCE_MEASURED
        pipe.on_calibration_changed(world_frame_changed=True)
        assert pipe._rep_analyzer.gravity_source == GRAVITY_SOURCE_MEASURED
        assert pipe._rep_analyzer._up == pytest.approx(up / np.linalg.norm(up), abs=1e-9)

    def test_a_calibration_change_never_touches_the_squats_gravity(self, monkeypatch):
        from biomechanics.deadlift import gravity

        monkeypatch.setattr(
            gravity, "load_world_up_for_provider", lambda provider: pytest.fail("the squat never reads gravity"),
        )
        pipe, _, _ = _pipeline(monkeypatch, SQUAT, 0.0)
        pipe.on_calibration_changed(world_frame_changed=True)
        assert pipe._gravity_up is None

    def test_the_squat_keeps_its_counter_and_leg_tracking(self, monkeypatch):
        pipe, _, _ = _pipeline(monkeypatch, SQUAT, 0.0)
        pipe.set_exercise(DEADLIFT)
        pipe.set_exercise(SQUAT)
        from biomechanics.faults.hip_position_counter import SignalRepCounter

        assert isinstance(pipe._rep_counter, SignalRepCounter)
        assert pipe.profile.tracking_keypoints is None
