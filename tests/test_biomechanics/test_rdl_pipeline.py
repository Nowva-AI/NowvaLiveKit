"""
The RDL end to end through the real BiomechanicsPipeline (docs/exercises/rdl.md
§7 Phase D): synthetic hinge poses in the production frame (Y-down, forward =
-Z) through the readiness gate, pre-IK, IK, RdlRepAnalyzer, the signal counter,
the rules and the recorded-only stamping, on one camera and on the rig.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pytest

from biomechanics.config import BiomechanicsConfig
from biomechanics.faults.hinge_reference import HingeSessionReference
from biomechanics.faults.observability import APPROXIMATE, OBSERVABLE
from biomechanics.profiles.romanian_deadlift import RomanianDeadliftProfile
from biomechanics.utils.types import CocoKeypoints as CK
from biomechanics.utils.types import FaultEvent, FaultSeverity, PipelineFrame, RepData
from conftest import SyntheticSession
from test_rdl_poses import (
    CLEAN_BOTTOM,
    NUM_KEYPOINTS,
    STANDING,
    RdlPose,
    camera_view,
    cosine_crossing_time_s,
    cosine_move,
    dropping_hand,
    hinge_for_travel,
    pose_at_drop,
    rdl_rep,
)

RDL_NAME = "Barbell Romanian Deadlift"
PITCH_ENV = "NOWVA_CAMERA_PITCH_DEG"
RIG_FPS = 30.0
MONO_FPS = 12.0
CONFIDENCE = 0.9
LOW_HIP_CONFIDENCE = 0.3
OCCLUDED_HINGE_DEG = 40.0
MOUNT_PITCH_DEG = 2.0
LENS_HEIGHT_M = 0.95
CLEAN_REPS = 10
VERTICAL_SHIN_BOTTOM = RdlPose(70.0, 33.0, 0.0)
MILD_SQUAT_BOTTOM = RdlPose(70.0, 53.0, 12.0)
MODERATE_SQUAT_BOTTOM = RdlPose(70.0, 60.0, 15.0)
UNRACK_TRAVEL = 0.14
PARTIAL_TRAVEL = 0.35
DIP_S = 0.6
FAST_DESCENT_S = 0.5
CONTROLLED_DESCENT_S = 2.0
# Inside the moderate band (>= 7°) at the bottom window's median.
HAND_DROP_DEG = 7.5
BAR_AHEAD_M = 0.12
LEAN_BACK_DEG = -10.0
SIGMA_XY_M = 0.01
SIGMA_Z_M = 0.03
HIP_DEPTH_BIAS_M = 0.10
PISTON_REPS = 5
TURNAROUND_DROP = 0.2
TURNAROUND_UP_S = 1.0
TURNAROUND_DOWN_S = 1.5
PISTON_DESCENT_TOLERANCE_S = 0.08
LINEAR_CROSSING_FRACTION = 0.9
SETUP_STAND_S = 2.0
KNOCK_DEG = -4.0
KNOCKED_MILD_BOTTOM = RdlPose(70.0, 46.5, 0.0)
RATIO_TOLERANCE = 0.005
HINGE_TOLERANCE_DEG = 1.0
TRIP_DELTA_TOLERANCE_DEG = 0.5


def _mount_config(pitch_deg: float = MOUNT_PITCH_DEG, height_m: float = LENS_HEIGHT_M) -> BiomechanicsConfig:
    config = BiomechanicsConfig()
    config.capture.mount_source = "inclinometer"
    config.capture.mount_pitch_deg = pitch_deg
    config.capture.mount_height_m = height_m
    return config


class _RdlRun:
    """A pipeline coaching the RDL (dev override), fed RdlPose frames."""

    def __init__(
        self, monkeypatch, multi_camera: bool, fps: float, config: BiomechanicsConfig | None = None,
    ) -> None:
        monkeypatch.delenv(PITCH_ENV, raising=False)
        monkeypatch.setattr(RomanianDeadliftProfile, "coaching_ready", True)
        self.session = SyntheticSession(monkeypatch, multi_camera=multi_camera, config=config, exercise_name=RDL_NAME)
        self.fps = fps
        self.view: Callable[[np.ndarray], np.ndarray] | None = None
        self.noise: Callable[[np.ndarray, RdlPose], np.ndarray] | None = None
        self.confidences: Callable[[RdlPose], np.ndarray] | None = None

    @property
    def pipeline(self):
        return self.session.pipeline

    def run(self, poses: list[RdlPose]) -> list[PipelineFrame]:
        results = []
        for pose in poses:
            points = pose.points()
            if self.view is not None:
                points = self.view(points)
            if self.noise is not None:
                points = self.noise(points, pose)
            confidences = self.confidences(pose) if self.confidences is not None else None
            results.append(self.session.step(points, confidences, fps=self.fps))
        return results

    def end_set(self) -> None:
        self.pipeline.reset_readiness_gate()
        self.pipeline.reset_rep_counter()


def _reps(results: list[PipelineFrame]) -> list[RepData]:
    return [result.rep_data for result in results if result.rep_data is not None]


def _faults(results: list[PipelineFrame], fault_type: str | None = None) -> list[FaultEvent]:
    return [
        fault for result in results for fault in result.faults
        if fault_type is None or fault.fault_type == fault_type
    ]


def _cueable(fault: FaultEvent) -> bool:
    return not fault.details.get("recorded_only") and fault.details.get("observability") == OBSERVABLE


def _shallow(results: list[PipelineFrame]) -> list[PipelineFrame]:
    return [result for result in results if result.analyzer_shallow_rep]


def _reps_to(bottom: RdlPose, count: int, fps: float, **rep_kwargs) -> list[RdlPose]:
    poses: list[RdlPose] = []
    for _ in range(count):
        poses += rdl_rep(fps, bottom=bottom, **rep_kwargs)
    return poses + [STANDING] * int(fps)


def _dip(travel: float, fps: float) -> list[RdlPose]:
    bottom = RdlPose(hinge_for_travel(travel, 10.0, 3.0), 10.0, 3.0)
    return rdl_rep(fps, bottom=bottom, stand_s=1.0, down_s=DIP_S, hold_s=0.1, up_s=DIP_S) + [STANDING] * int(fps)


def _partial(fps: float) -> list[RdlPose]:
    bottom = RdlPose(hinge_for_travel(PARTIAL_TRAVEL, 20.0, 5.0), 20.0, 5.0)
    return rdl_rep(fps, bottom=bottom) + [STANDING] * int(fps)


def _anisotropic_noise(seed: int) -> Callable[[np.ndarray, RdlPose], np.ndarray]:
    rng = np.random.default_rng(seed)

    def noise(points: np.ndarray, pose: RdlPose) -> np.ndarray:
        noisy = points + rng.normal(0.0, [SIGMA_XY_M, SIGMA_XY_M, SIGMA_Z_M], points.shape)
        # Monocular depth bias on the hips that grows with the hinge, in the
        # direction that lowers the levelled hip (the worst case for hip sink).
        depth = max(0.0, pose.hinge_deg) / VERTICAL_SHIN_BOTTOM.hinge_deg
        noisy[[CK.LEFT_HIP, CK.RIGHT_HIP], 2] -= HIP_DEPTH_BIAS_M * depth
        return noisy

    return noise


MODES = [
    pytest.param(True, RIG_FPS, id="rig-30fps"),
    pytest.param(False, MONO_FPS, id="mono-12fps"),
]


class TestCleanReps:
    @pytest.mark.parametrize("multi_camera, fps", MODES)
    def test_clean_reps_all_count_with_no_faults(self, monkeypatch, multi_camera: bool, fps: float):
        run = _RdlRun(monkeypatch, multi_camera, fps)
        results = run.run([STANDING] * 10 + _reps_to(CLEAN_BOTTOM, CLEAN_REPS, fps))

        reps = _reps(results)
        assert [rep.rep_number for rep in reps] == list(range(1, CLEAN_REPS + 1))
        assert _faults(results) == []
        assert _shallow(results) == []
        assert all(rep.features["exercise"] == "romanian_deadlift" for rep in reps)
        assert all(rep.features["highlight_score"] < 0.0 for rep in reps)
        assert all(rep.descent_time == pytest.approx(rep.features["descent_time_s"]) for rep in reps)

    def test_the_bilstm_never_runs_on_the_rdl(self, monkeypatch):
        config = BiomechanicsConfig()
        config.bilstm.enabled = True
        run = _RdlRun(monkeypatch, True, RIG_FPS, config)
        assert run.pipeline._bilstm is None
        assert type(run.pipeline.profile) is RomanianDeadliftProfile


class TestHipSinkOnOneCamera:
    @pytest.mark.parametrize("pitch_deg", [8.0, -8.0])
    def test_correct_reps_are_never_cued_at_8_degrees_uncertified(self, monkeypatch, pitch_deg: float):
        run = _RdlRun(monkeypatch, False, MONO_FPS)
        run.view = lambda points: camera_view(points, pitch_deg)
        results = run.run([STANDING] * 10 + _reps_to(VERTICAL_SHIN_BOTTOM, 3, MONO_FPS))

        assert len(_reps(results)) == 3
        assert not any(_cueable(fault) for fault in _faults(results, "rdl_hip_sink"))

    @pytest.mark.parametrize("pitch_deg", [5.0, -5.0])
    def test_correct_reps_read_no_hip_sink_at_5_degrees(self, monkeypatch, pitch_deg: float):
        run = _RdlRun(monkeypatch, False, MONO_FPS)
        run.view = lambda points: camera_view(points, pitch_deg)
        results = run.run([STANDING] * 10 + _reps_to(VERTICAL_SHIN_BOTTOM, 3, MONO_FPS))

        assert len(_reps(results)) == 3
        assert _faults(results, "rdl_hip_sink") == []

    def test_a_certified_levelled_mount_reads_no_hip_sink_on_correct_reps(self, monkeypatch):
        run = _RdlRun(monkeypatch, False, MONO_FPS, _mount_config())
        run.view = lambda points: camera_view(points, MOUNT_PITCH_DEG)
        run.noise = _anisotropic_noise(seed=7)
        results = run.run([STANDING] * 10 + _reps_to(VERTICAL_SHIN_BOTTOM, 5, MONO_FPS))

        assert run.pipeline.pitch_certified
        assert len(_reps(results)) == 5
        assert _faults(results, "rdl_hip_sink") == []

    @pytest.mark.parametrize(
        "bottom, minimum",
        [(MILD_SQUAT_BOTTOM, FaultSeverity.MILD), (MODERATE_SQUAT_BOTTOM, FaultSeverity.MODERATE)],
    )
    def test_squatted_reps_are_cued_on_a_certified_mount(self, monkeypatch, bottom: RdlPose, minimum: FaultSeverity):
        run = _RdlRun(monkeypatch, False, MONO_FPS, _mount_config())
        run.view = lambda points: camera_view(points, MOUNT_PITCH_DEG)
        results = run.run([STANDING] * 10 + _reps_to(bottom, 2, MONO_FPS))

        verdicts = _faults(results, "rdl_hip_sink")
        severity_rank = [FaultSeverity.MILD, FaultSeverity.MODERATE, FaultSeverity.SEVERE]
        assert len(verdicts) == 2
        assert all(severity_rank.index(fault.severity) >= severity_rank.index(minimum) for fault in verdicts)
        assert all(_cueable(fault) for fault in verdicts)

    def test_a_squatted_rep_on_an_unknown_mount_is_recorded_only(self, monkeypatch):
        run = _RdlRun(monkeypatch, False, MONO_FPS)
        results = run.run([STANDING] * 10 + _reps_to(MODERATE_SQUAT_BOTTOM, 1, MONO_FPS))

        verdicts = _faults(results, "rdl_hip_sink")
        assert [fault.severity for fault in verdicts] == [FaultSeverity.MODERATE]
        assert verdicts[0].details["recorded_only"]
        assert _reps(results)[0].is_clean

    def test_hidden_hips_on_one_camera_give_no_verdict_but_two_view_rig_hips_do(self, monkeypatch):
        def occluded_hips(pose: RdlPose) -> np.ndarray:
            confidences = np.full(NUM_KEYPOINTS, CONFIDENCE)
            if pose.hinge_deg > OCCLUDED_HINGE_DEG:
                confidences[[CK.LEFT_HIP, CK.RIGHT_HIP]] = LOW_HIP_CONFIDENCE
            return confidences

        mono = _RdlRun(monkeypatch, False, MONO_FPS, _mount_config(0.0))
        mono.confidences = occluded_hips
        mono_results = mono.run([STANDING] * 10 + _reps_to(MODERATE_SQUAT_BOTTOM, 1, MONO_FPS))
        rig = _RdlRun(monkeypatch, True, RIG_FPS)
        rig.confidences = occluded_hips
        rig_results = rig.run([STANDING] * 10 + _reps_to(MODERATE_SQUAT_BOTTOM, 1, RIG_FPS))

        assert _reps(mono_results)[0].features["hinge_hip_source"] == "setup"
        assert _faults(mono_results, "rdl_hip_sink") == []
        assert _reps(rig_results)[0].features["hinge_hip_source"] == "measured"
        assert [fault.severity for fault in _faults(rig_results, "rdl_hip_sink")] == [FaultSeverity.MODERATE]


class TestRejections:
    def test_unrack_dips_are_ignored_even_after_a_counted_rep(self, monkeypatch):
        run = _RdlRun(monkeypatch, False, MONO_FPS)
        poses = [STANDING] * 10 + _dip(UNRACK_TRAVEL, MONO_FPS) + _reps_to(CLEAN_BOTTOM, 1, MONO_FPS)
        poses += _dip(UNRACK_TRAVEL, MONO_FPS) + _dip(UNRACK_TRAVEL, MONO_FPS)
        results = run.run(poses)

        assert [rep.rep_number for rep in _reps(results)] == [1]
        assert _faults(results) == []
        assert _shallow(results) == []

    def test_partials_send_a_shallow_rep_each_and_cue_from_the_second(self, monkeypatch):
        run = _RdlRun(monkeypatch, False, MONO_FPS)
        results = run.run([STANDING] * 10 + _partial(MONO_FPS) + _partial(MONO_FPS))

        shallow = _shallow(results)
        assert _reps(results) == []
        assert [result.shallow_rep_cue for result in shallow] == [None, "rdl_lower"]
        partials = [fault for result in shallow for fault in result.faults]
        assert [fault.fault_type for fault in partials] == ["rdl_partial", "rdl_partial"]
        assert all(fault.details["shallow_rep"] and fault.details["range_label"] for fault in partials)

    def test_partial_cues_stop_at_two_per_set_and_restart_with_the_set(self, monkeypatch):
        run = _RdlRun(monkeypatch, False, MONO_FPS)
        poses = [STANDING] * 10
        for _ in range(4):
            poses += _partial(MONO_FPS)
        first_set = run.run(poses)
        run.end_set()
        second_set = run.run([STANDING] * 10 + _partial(MONO_FPS))

        assert [result.shallow_rep_cue for result in _shallow(first_set)] == [None, "rdl_lower", "rdl_lower", None]
        assert [result.shallow_rep_cue for result in _shallow(second_set)] == [None]

    @pytest.mark.parametrize("multi_camera, fps", MODES)
    def test_counted_reps_and_a_partial_run_on_the_rdl_session_reference(
        self, monkeypatch, multi_camera: bool, fps: float,
    ):
        run = _RdlRun(monkeypatch, multi_camera, fps)
        results = run.run([STANDING] * 10 + _reps_to(CLEAN_BOTTOM, 3, fps) + _partial(fps) + _reps_to(CLEAN_BOTTOM, 1, fps))

        reference = run.pipeline._rule_engine.reference
        assert isinstance(reference, HingeSessionReference)
        assert [rep.rep_number for rep in _reps(results)] == [1, 2, 3, 4]
        assert len(_shallow(results)) == 1
        assert reference.hinge_reference() == pytest.approx(CLEAN_BOTTOM.hinge_deg, abs=HINGE_TOLERANCE_DEG)
        assert _reps(results)[-1].features["hinge_reference_deg"] == pytest.approx(reference.hinge_reference())

    def test_a_counted_rep_resets_the_partial_streak(self, monkeypatch):
        run = _RdlRun(monkeypatch, False, MONO_FPS)
        poses = [STANDING] * 10 + _partial(MONO_FPS) + _reps_to(CLEAN_BOTTOM, 1, MONO_FPS) + _partial(MONO_FPS)
        results = run.run(poses)

        assert [result.shallow_rep_cue for result in _shallow(results)] == [None, None]


class TestTempoAndTilt:
    @pytest.mark.parametrize("multi_camera, fps", MODES)
    def test_a_dropped_descent_is_cued_and_a_controlled_one_is_not(self, monkeypatch, multi_camera: bool, fps: float):
        run = _RdlRun(monkeypatch, multi_camera, fps)
        fast = run.run([STANDING] * 10 + _reps_to(CLEAN_BOTTOM, 1, fps, down_s=FAST_DESCENT_S))
        controlled = run.run(_reps_to(CLEAN_BOTTOM, 1, fps, down_s=CONTROLLED_DESCENT_S))

        assert [fault.severity for fault in _faults(fast, "rdl_tempo")] == [FaultSeverity.MODERATE]
        assert _faults(controlled, "rdl_tempo") == []

    @pytest.mark.parametrize("multi_camera, fps", MODES)
    def test_one_hand_dropping_reads_moderate_bar_tilt(self, monkeypatch, multi_camera: bool, fps: float):
        run = _RdlRun(monkeypatch, multi_camera, fps)
        run.view = dropping_hand(HAND_DROP_DEG)
        results = run.run([STANDING] * 10 + _reps_to(CLEAN_BOTTOM, 1, fps))

        tilts = _faults(results, "rdl_bar_tilt")
        assert [(fault.severity, fault.details["side"]) for fault in tilts] == [(FaultSeverity.MODERATE, "right")]


class TestRigOnlyFaults:
    def _poses(self, fps: float) -> list[RdlPose]:
        leaning_back = RdlPose(hinge_deg=LEAN_BACK_DEG)
        bottom = RdlPose(75.0, 20.0, 5.0, bar_forward_m=BAR_AHEAD_M)
        return (
            [leaning_back] * int(2 * fps) + cosine_move(leaning_back, bottom, 2.0, fps)
            + cosine_move(bottom, leaning_back, 1.2, fps) + [leaning_back] * int(fps)
        )

    def test_bar_drift_and_lockout_are_silent_on_one_camera(self, monkeypatch):
        run = _RdlRun(monkeypatch, False, MONO_FPS)
        results = run.run(self._poses(MONO_FPS))

        assert len(_reps(results)) == 1
        assert _faults(results, "rdl_bar_drift") == [] and _faults(results, "rdl_lockout") == []

    def test_bar_drift_and_lockout_are_approximate_on_the_rig(self, monkeypatch):
        run = _RdlRun(monkeypatch, True, RIG_FPS)
        results = run.run(self._poses(RIG_FPS))

        drift, lockout = _faults(results, "rdl_bar_drift"), _faults(results, "rdl_lockout")
        assert len(drift) == 1 and len(lockout) == 1
        assert lockout[0].details["direction"] == "back"
        for fault in drift + lockout:
            assert fault.details["observability"] == APPROXIMATE
            # Below the rig start tier (moderate until measured gravity): recorded only.
            if fault.severity == FaultSeverity.MILD:
                assert fault.details["recorded_only"]


def _linear_move(start: RdlPose, end: RdlPose, seconds: float, fps: float) -> list[RdlPose]:
    frames = int(round(seconds * fps))
    return [pose_at_drop(start, end, (i + 1) / frames) for i in range(frames)]


class TestPistonSets:
    @pytest.mark.parametrize("smooth", [True, False], ids=["smooth-turnaround", "sharp-turnaround"])
    def test_constant_tension_reps_count_without_lockout_or_reference_drift(self, monkeypatch, smooth: bool):
        run = _RdlRun(monkeypatch, True, RIG_FPS)
        turnaround = pose_at_drop(STANDING, CLEAN_BOTTOM, TURNAROUND_DROP)
        move = cosine_move if smooth else _linear_move
        poses = [STANDING] * int(SETUP_STAND_S * RIG_FPS) + cosine_move(STANDING, CLEAN_BOTTOM, 2.0, RIG_FPS)
        for _ in range(PISTON_REPS - 1):
            poses += move(CLEAN_BOTTOM, turnaround, TURNAROUND_UP_S, RIG_FPS)
            poses += move(turnaround, CLEAN_BOTTOM, TURNAROUND_DOWN_S, RIG_FPS)
        poses += cosine_move(CLEAN_BOTTOM, STANDING, 1.2, RIG_FPS) + [STANDING] * int(RIG_FPS)
        results = run.run(poses)

        reps = _reps(results)
        assert [rep.rep_number for rep in reps] == list(range(1, PISTON_REPS + 1))
        assert _faults(results, "rdl_lockout") == []
        expected_s = (
            cosine_crossing_time_s(TURNAROUND_DOWN_S) if smooth else LINEAR_CROSSING_FRACTION * TURNAROUND_DOWN_S
        )
        for rep in reps[1:]:
            assert rep.features["started_at_turnaround"]
            assert math.isnan(rep.features["lockout_pitch_deg"])
            assert rep.features["descent_time_s"] == pytest.approx(expected_s, abs=PISTON_DESCENT_TOLERANCE_S)
            assert rep.features["hip_sink_ratio"] == pytest.approx(reps[0].features["hip_sink_ratio"], abs=RATIO_TOLERANCE)
        assert run.pipeline._rep_analyzer.setup_count == 1


class TestCertificationRevokedMidSession:
    def test_a_knocked_camera_revokes_the_mild_tier_for_the_rest_of_the_session(self, monkeypatch):
        config = _mount_config()
        run = _RdlRun(monkeypatch, False, RIG_FPS, config)
        run.view = lambda points: camera_view(points, MOUNT_PITCH_DEG)
        stand = [STANDING] * int(SETUP_STAND_S * RIG_FPS)
        reference_sets = config.faults.rdl.camera.reference_sets
        certified_verdicts: list[FaultEvent] = []
        for _ in range(reference_sets):
            results = run.run(stand + _reps_to(RdlPose(70.0, 45.0, 0.0), 1, RIG_FPS, stand_s=0.0))
            certified_verdicts += _faults(results, "rdl_hip_sink")
            run.end_set()
        assert run.pipeline.pitch_certified

        run.view = lambda points: camera_view(points, MOUNT_PITCH_DEG + KNOCK_DEG)
        knocked = run.run(stand + _reps_to(KNOCKED_MILD_BOTTOM, 1, RIG_FPS, stand_s=0.0))

        assert [fault.severity for fault in certified_verdicts] == [FaultSeverity.MILD] * reference_sets
        assert all(_cueable(fault) for fault in certified_verdicts)
        revoked = [result.pitch_certification_revoked_deg for result in knocked if result.pitch_certification_revoked_deg]
        assert revoked == [pytest.approx(abs(KNOCK_DEG), abs=TRIP_DELTA_TOLERANCE_DEG)]
        assert not run.pipeline.pitch_certified
        after = _faults(knocked, "rdl_hip_sink")
        assert [fault.severity for fault in after] == [FaultSeverity.MILD]
        assert after[0].details["recorded_only"]
