"""
Squat golden masters: fixed synthetic squat sessions replayed through the real
BiomechanicsPipeline, SessionTracker and IPCBridge, snapshotted frame by frame.

They pin the squat path byte for byte while the shared exercise plumbing is
added (docs/exercises/SHARED-PLUMBING.md §8). Each session runs fresh, and
again after a second of overhead-press frames and a switch back. With the
athlete's stored body measurements applied, the switched run must reproduce
the fresh golden exactly: the documented carry-overs (the depth target
survives, SessionReference restarts on every switch) leave the squat
unchanged. A first-time user's body measurement is session-scoped, but press
frames never feed it (feeds_body_calibration False), so that run reproduces
the fresh golden too.

The overhead press is still gated, so that switch only reaches an untracked
stand-in. A second switch goes through a test-only analyser profile that
exercises every hook (readiness, counter baseline, rejection, a counted rep,
recorded-only verdicts, no depth, no camera refine, no body measurement) and
must still reproduce the fresh golden on the way back.

Regenerate only for an intended squat change:
    NOWVA_REGEN_GOLDENS=1 pytest tests/test_biomechanics/test_squat_golden_master.py
"""

from __future__ import annotations

import json
import math
import os
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from pydantic import BaseModel

from biomechanics.config import BiomechanicsConfig
from biomechanics.faults.fault_types import FaultType, RepFaultRule
from biomechanics.faults.session_reference import SessionReference
from biomechanics.profiles import PROFILE_REGISTRY
from biomechanics.profiles.base import ExerciseProfile
from biomechanics.utils.frame_context import FrameContext
from biomechanics.utils.json_safe import nan_to_none
from biomechanics.utils.types import CocoKeypoints as CK
from biomechanics.utils.types import FaultEvent, FaultSeverity, Keypoint2D, Skeleton2D, Skeleton3D
from conftest import SYNTHETIC_FPS, squat_depth_profile, world_squat_points

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "golden"
REGEN_ENV = "NOWVA_REGEN_GOLDENS"
BILSTM_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "bilstm_rep_counter.pt"

FRAME_DT_S = 1.0 / SYNTHETIC_FPS
CLOCK_START_S = 1.7e9
PROVIDER_CONFIDENCE = 0.9
PIXELS_PER_M = 600.0
IMAGE_CENTRE_PX = (640.0, 300.0)
CAPTURE_FRAME = np.zeros((72, 128, 3), dtype=np.uint8)
# A perfectly collinear synthetic leg reads exactly 0 deg; real lifters keep a small bend.
MIN_DEPTH_RATIO = 0.02
READINESS_WARM_UP_FRAMES = 10
VALGUS_SHIFT_M = 0.12
LEAN_EXTRA_DEG = 14.0
SHALLOW_REP_DEPTH = 0.5
FAST_DESCENT_S = 0.3
PISTON_TOP_DEPTH = 0.3
PISTON_TURN_S = 0.5
PISTON_REPS = 3
DROPOUT_FRAMES = 2
# Frames run on the other exercise before switching back: a whole number of
# frame_data intervals, so the bridge's send cadence lines up with a fresh run.
OTHER_EXERCISE_FRAMES = 30
OTHER_EXERCISE_LEAD_S = 10.0
OTHER_EXERCISE = "Barbell Overhead Press"
SQUAT = "Barbell Back Squat"
# The test-only analyser profile, registered per test (never in production code).
FAKE_LIFT = "Fake Lift"
FAKE_LIFT_PROFILE = "fake_lift"
FAKE_LIFT_REPS = 2
# Warm-up, the reps and a rest: a whole number of frame_data intervals that
# ends before the squat's clock starts (OTHER_EXERCISE_LEAD_S).
FAKE_LIFT_FRAMES = 240
FAKE_REJECTION = "partial"
FAKE_SHALLOW_CUE = "fake_lower"
FAKE_DEPTH_DEG = 42.0
# Hip above the ankles standing upright (cm), the analyser's counter baseline.
FAKE_STANDING_SIGNAL_CM = -88.0
ATHLETE_PARAMS = {
    "shoulder_width_m": 0.38,
    "femur_avg_m": 0.45,
    "torso_avg_m": 0.52,
    "hip_width_m": 0.24,
    "tibia_avg_m": 0.43,
    "foot_avg_m": 0.18,
}


class _Clock:
    def __init__(self, start_s: float) -> None:
        self.now_s = start_s

    def advance(self, seconds: float) -> None:
        self.now_s += seconds

    def time(self) -> float:
        return self.now_s

    def perf_counter(self) -> float:
        return self.now_s


class _Provider:
    """MultiCameraPoseProvider stand-in: world-frame skeletons on the fake clock."""

    def __init__(self, clock: _Clock) -> None:
        self._clock = clock
        self._queue: list[np.ndarray | None] = []
        self.sequence = 0
        self.swap_count = 0
        self.frame = np.zeros((72, 128, 3), dtype=np.uint8)
        self.last_capture_timestamp = math.nan

    def push(self, points: np.ndarray | None) -> None:
        self._queue.append(points)

    def get_pose(self):
        points = self._queue.pop(0)
        self.sequence += 1
        self.last_capture_timestamp = self._clock.time()
        if points is None:
            return self.frame, None, None
        skeleton = Skeleton3D.from_numpy(
            points, confidences=np.full(len(points), PROVIDER_CONFIDENCE),
            timestamp=self._clock.time(), frame_index=self.sequence,
        )
        return self.frame, None, skeleton

    def reset_temporal_state(self) -> None:
        pass

    def lost_cameras(self) -> list[str]:
        return []

    def release(self) -> None:
        pass


class _SingleCameraPose:
    """MediaPipe stand-in: hip-centred Y-down 3D plus a pinhole 2D view of it."""

    def __init__(self) -> None:
        self._queue: list[np.ndarray | None] = []

    def push(self, points: np.ndarray | None) -> None:
        self._queue.append(points)

    def estimate_both(self, frame: np.ndarray):
        points = self._queue.pop(0)
        if points is None:
            return None, None
        confidences = np.full(len(points), PROVIDER_CONFIDENCE)
        centred = points - (points[CK.LEFT_HIP] + points[CK.RIGHT_HIP]) / 2.0
        skeleton_3d = Skeleton3D.from_numpy(centred, confidences=confidences)
        keypoints_2d = [
            Keypoint2D(
                x=IMAGE_CENTRE_PX[0] + PIXELS_PER_M * point[0],
                y=IMAGE_CENTRE_PX[1] + PIXELS_PER_M * point[1],
                confidence=float(confidence),
            )
            for point, confidence in zip(centred, confidences)
        ]
        return Skeleton2D(keypoints=keypoints_2d), skeleton_3d

    def reset_tracking(self) -> None:
        pass

    def release(self) -> None:
        pass


class _RecordingClient:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    def send_message(self, message: dict) -> None:
        self.messages.append(message)


class _FakeLiftFeatures(BaseModel):
    exercise: str = FAKE_LIFT_PROFILE
    rep_number: int
    max_depth_angle: float = math.nan


class _FakeLiftAnalyzer:
    """Squat-shaped signal (hip above ankle, cm); rejects its first rep and counts the rest."""

    def __init__(self) -> None:
        self.rep_signal = math.nan
        self.readiness_checks = 0
        self.baseline_seeds = 0
        self.reasons: list[str | None] = []

    def is_ready(self, ctx: FrameContext) -> bool:
        self.readiness_checks += 1
        return ctx.angles is None

    def counter_baseline(self) -> float | None:
        self.baseline_seeds += 1
        return FAKE_STANDING_SIGNAL_CM

    def observe(self, ctx: FrameContext) -> None:
        points = ctx.skeleton_3d.to_numpy()
        hip_y = (points[CK.LEFT_HIP][1] + points[CK.RIGHT_HIP][1]) / 2.0
        ankle_y = (points[CK.LEFT_ANKLE][1] + points[CK.RIGHT_ANKLE][1]) / 2.0
        self.rep_signal = (hip_y - ankle_y) * 100.0

    def on_rep_start(self, piston: bool) -> None:
        pass

    def finish_rep(self, rep_number: int) -> _FakeLiftFeatures:
        return _FakeLiftFeatures(rep_number=rep_number, max_depth_angle=FAKE_DEPTH_DEG)

    def rejection_reason(self, features: _FakeLiftFeatures) -> str | None:
        self.reasons.append(FAKE_REJECTION if not self.reasons else None)
        return self.reasons[-1]

    def discard_rep(self) -> None:
        pass

    def reset_set(self) -> None:
        pass


class _FakeLiftReference(SessionReference):
    def annotate(self, features) -> None:
        pass

    def update(self, features) -> None:
        pass


class _MildVerdictRule(RepFaultRule):
    """A mild verdict on every counted rep, below the fake lift's start tier."""

    @property
    def fault_type(self) -> FaultType:
        return FaultType.PRESS_UNEVEN

    def judge_rep(self, features, reference, angles) -> FaultEvent | None:
        return self._create_fault_event(FaultSeverity.MILD, 1.0, "verdict", angles, features.rep_number)


class _FakeLiftProfile(ExerciseProfile):
    name = FAKE_LIFT_PROFILE
    movement_pattern = None
    reports_depth = False
    allows_camera_refine = False
    feeds_body_calibration = False
    min_cue_tiers = types.MappingProxyType({"press_uneven": FaultSeverity.MODERATE})

    def create_fault_rules(self, config):
        return [_MildVerdictRule()]

    def create_rep_analyzer(self, config) -> _FakeLiftAnalyzer:
        return _FakeLiftAnalyzer()

    def create_session_reference(self, config) -> SessionReference:
        return _FakeLiftReference()

    def get_fault_to_cue_map(self):
        return {"press_uneven": "press_uneven"}

    def on_rep_rejected(self, reason, features, angles, config):
        return [FaultEvent(
            fault_type="rdl_partial", severity=FaultSeverity.MODERATE, severity_score=1.5,
            message="rejected", rep_number=features.rep_number, details={"shallow_rep": True},
        )]

    def shallow_rep_cue(self, fault, streak, cued_this_set, config):
        return FAKE_SHALLOW_CUE


@dataclass
class _FakeLiftRun:
    analyzer: _FakeLiftAnalyzer
    messages: list[dict]
    view_recording: list[bool]


def _cosine_move(start: float, end: float, seconds: float) -> list[float]:
    frames = int(round(seconds * SYNTHETIC_FPS))
    return [start + (end - start) * 0.5 * (1.0 - math.cos(math.pi * (i + 1) / frames)) for i in range(frames)]


def _switch(pipeline, tracker, bridge, exercise_name: str) -> None:
    from biomechanics.pipeline_process import _activate_profile

    _activate_profile(pipeline, tracker, bridge, exercise_name)


def _start(pipeline, tracker, bridge, exercise_name: str) -> None:
    from biomechanics.pipeline_process import _activate_profile

    _activate_profile(pipeline, tracker, bridge, exercise_name, startup=True)


class _Session:
    """Drives frames the way pipeline_process's workout loop does and records every event."""

    def __init__(
        self, monkeypatch, multi_camera: bool, bilstm: bool, athlete_params: bool, clock_start_s: float,
    ) -> None:
        from biomechanics import pipeline as pipeline_module
        from biomechanics.coaching.ipc_bridge import IPCBridge
        from biomechanics.coaching.session_tracker import SessionTracker

        monkeypatch.setenv("NOWVA_MULTI_CAMERA", "true" if multi_camera else "false")
        self.multi_camera = multi_camera
        self.clock = _Clock(clock_start_s)
        monkeypatch.setattr(
            pipeline_module, "time",
            types.SimpleNamespace(time=self.clock.time, perf_counter=self.clock.perf_counter),
        )
        config = BiomechanicsConfig()
        config.bilstm.enabled = bilstm
        config.bilstm.model_path = str(BILSTM_MODEL_PATH)
        self.pipeline = pipeline_module.BiomechanicsPipeline(config, exercise_name=SQUAT, defer_capture=True)
        self.provider = _Provider(self.clock)
        self.pose = _SingleCameraPose()
        if multi_camera:
            self.pipeline._multi_camera_provider = self.provider
        else:
            self.pipeline._pose_estimator = self.pose
        self.client = _RecordingClient()
        self.bridge = IPCBridge(self.client, coaching_config=config.coaching, ipc_config=config.ipc)
        self.tracker = SessionTracker(self.bridge, config=config.coaching)
        if athlete_params:
            self.pipeline.apply_athlete_params(ATHLETE_PARAMS)
            self.tracker.set_athlete_params(ATHLETE_PARAMS, {})
            self.bridge.set_athlete_params(ATHLETE_PARAMS, {})
        self.events: list[dict] = []
        self.phases: list[str] = []
        self.frame_number = 0

    def frames(
        self,
        depths: list[float],
        valgus_m: float = 0.0,
        lean_extra_deg: float = 0.0,
        dropped: frozenset[int] = frozenset(),
    ) -> None:
        for index, depth in enumerate(depths):
            points = None
            if index not in dropped:
                points = world_squat_points(max(depth, MIN_DEPTH_RATIO), valgus_m, lean_extra_deg)
            if self.multi_camera:
                self.provider.push(points)
            else:
                self.pose.push(points)
                self.pipeline._publish_frame(CAPTURE_FRAME, self.clock.time())
            self._deliver(self.pipeline.process_frame())
            self.clock.advance(FRAME_DT_S)

    def rewind(self, clock_start_s: float) -> None:
        """Restart the clock, the capture sequence and the record, for a run comparable to a fresh one."""
        self.clock.now_s = clock_start_s
        self.provider.sequence = 0
        self.pipeline._capture_sequence = 0
        self.pipeline._processed_sequence = 0
        self.events, self.phases, self.frame_number = [], [], 0

    def _deliver(self, result) -> None:
        from biomechanics.pipeline_process import _send_shallow_rep

        pipeline, bridge, tracker = self.pipeline, self.bridge, self.tracker
        bridge.update_tracking_quality(result, active=pipeline.is_ready, in_rep=pipeline.rep_counter.in_rep)
        bridge.update_camera_status(result)
        bridge.send_frame_data(result, rep_phase=pipeline.rep_counter.phase)
        for fault in result.faults:
            if not fault.details.get("shallow_rep"):
                bridge.send_fault(fault)
        if result.shallow_rep_class is not None or result.analyzer_shallow_rep:
            _send_shallow_rep(bridge, result, tracker.current_set_number)
        if result.rep_data:
            bottom_kpts, bottom_angles = pipeline.consume_bottom_frame()
            tracker.on_rep_complete(
                result.rep_data,
                bottom_kpts=bottom_kpts,
                bottom_angles=bottom_angles,
                standing_kpts=pipeline.consume_standing_frame(),
                trajectory_samples=pipeline.consume_rep_trajectory(),
            )
        tracker.check_set_timeout(self.clock.time())

        self.phases.append(("R" if pipeline.is_ready else "-") + pipeline.rep_counter.phase[0])
        if result.faults or result.rep_data is not None or result.shallow_rep_class is not None:
            self.events.append({
                "frame": self.frame_number,
                "timestamp": result.timestamp,
                "rep_count": pipeline.rep_count,
                "faults": [fault.model_dump() for fault in result.faults],
                "rep_data": result.rep_data.model_dump() if result.rep_data is not None else None,
                "shallow_rep_class": result.shallow_rep_class,
            })
        self.frame_number += 1

    def end_set(self) -> None:
        """rest_start, then the gate re-arms and the counter resets at rest end."""
        self.tracker.force_end_set()
        self.pipeline.reset_readiness_gate()
        self.pipeline.reset_rep_counter()
        self.pipeline.reset_bilstm()

    def snapshot(self, ipc_from: int) -> dict:
        return _json_round_trip({
            "phases": "".join(self.phases),
            "events": self.events,
            "ipc": self.client.messages[ipc_from:],
        })


def _json_round_trip(payload: dict) -> dict:
    return json.loads(json.dumps(nan_to_none(payload), sort_keys=True))


def _squat_session(session: _Session) -> None:
    """Two sets: clean, caving-knee, leaning, shallow and fast reps; then pistons and a dropout."""
    warm_up = [0.0] * READINESS_WARM_UP_FRAMES
    session.frames(warm_up)
    for _ in range(2):
        session.frames(squat_depth_profile())
    for _ in range(2):
        session.frames(squat_depth_profile(), valgus_m=VALGUS_SHIFT_M)
    session.frames(squat_depth_profile(), lean_extra_deg=LEAN_EXTRA_DEG)
    session.frames([depth * SHALLOW_REP_DEPTH for depth in squat_depth_profile()])
    session.frames(squat_depth_profile(down_s=FAST_DESCENT_S))
    session.frames([0.0] * int(SYNTHETIC_FPS))
    session.end_set()

    session.frames(warm_up + [0.0] * int(SYNTHETIC_FPS))
    piston = _cosine_move(0.0, 1.0, 1.0)
    for _ in range(PISTON_REPS - 1):
        piston += _cosine_move(1.0, PISTON_TOP_DEPTH, PISTON_TURN_S)
        piston += _cosine_move(PISTON_TOP_DEPTH, 1.0, 1.0)
    piston += _cosine_move(1.0, 0.0, 1.0) + [0.0] * int(SYNTHETIC_FPS)
    session.frames(piston)
    rep = squat_depth_profile()
    dropout_start = len(rep) // 2
    session.frames(rep, dropped=frozenset(range(dropout_start, dropout_start + DROPOUT_FRAMES)))
    session.frames([0.0] * int(SYNTHETIC_FPS))
    session.tracker.force_end_set()


def _fresh_run(monkeypatch, multi_camera: bool, bilstm: bool, athlete_params: bool) -> dict:
    session = _Session(monkeypatch, multi_camera, bilstm, athlete_params, CLOCK_START_S)
    _start(session.pipeline, session.tracker, session.bridge, SQUAT)
    ipc_from = len(session.client.messages)
    _squat_session(session)
    return session.snapshot(ipc_from)


def _after_switch_run(monkeypatch, multi_camera: bool, bilstm: bool, athlete_params: bool) -> dict:
    """Squat -> overhead press for a second of frames -> squat, then the same session.

    The squat part runs on the fresh run's clock and capture sequence, so the
    snapshot is directly comparable."""
    session = _Session(monkeypatch, multi_camera, bilstm, athlete_params, CLOCK_START_S - OTHER_EXERCISE_LEAD_S)
    _start(session.pipeline, session.tracker, session.bridge, SQUAT)
    _switch(session.pipeline, session.tracker, session.bridge, OTHER_EXERCISE)
    session.frames([0.0] * OTHER_EXERCISE_FRAMES)
    _switch(session.pipeline, session.tracker, session.bridge, SQUAT)
    session.rewind(CLOCK_START_S)
    ipc_from = len(session.client.messages)
    _squat_session(session)
    return session.snapshot(ipc_from)


def _after_analyser_switch_run(
    monkeypatch, multi_camera: bool, bilstm: bool, athlete_params: bool,
) -> tuple[dict, _FakeLiftRun]:
    """Squat -> the fake analyser lift (a rejected rep, a counted one) -> squat, then the same session."""
    session = _Session(monkeypatch, multi_camera, bilstm, athlete_params, CLOCK_START_S - OTHER_EXERCISE_LEAD_S)
    view_recording: list[bool] = []
    # The rig keeps buffering views for camera refines only while the exercise allows them.
    session.provider.set_view_recording = view_recording.append
    _start(session.pipeline, session.tracker, session.bridge, SQUAT)
    fake_from = len(session.client.messages)
    _switch(session.pipeline, session.tracker, session.bridge, FAKE_LIFT)
    analyzer = session.pipeline._rep_analyzer
    depths = [0.0] * READINESS_WARM_UP_FRAMES
    for _ in range(FAKE_LIFT_REPS):
        depths += squat_depth_profile()
    session.frames(depths + [0.0] * (FAKE_LIFT_FRAMES - len(depths)))
    fake_run = _FakeLiftRun(analyzer, session.client.messages[fake_from:], view_recording)
    _switch(session.pipeline, session.tracker, session.bridge, SQUAT)
    session.rewind(CLOCK_START_S)
    ipc_from = len(session.client.messages)
    _squat_session(session)
    return session.snapshot(ipc_from), fake_run


def _check_golden(name: str, actual: dict) -> None:
    path = GOLDEN_DIR / f"{name}.json"
    if os.getenv(REGEN_ENV) == "1":
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(actual, sort_keys=True, indent=1) + "\n")
    expected = json.loads(path.read_text())
    assert actual["phases"] == expected["phases"]
    assert len(actual["events"]) == len(expected["events"])
    for actual_event, expected_event in zip(actual["events"], expected["events"]):
        assert actual_event == expected_event
    assert len(actual["ipc"]) == len(expected["ipc"])
    for actual_message, expected_message in zip(actual["ipc"], expected["ipc"]):
        assert actual_message == expected_message


_bilstm_model = pytest.mark.skipif(
    not BILSTM_MODEL_PATH.exists(), reason="BiLSTM model not installed (models/ is not in git)",
)

VARIANTS = [
    pytest.param(True, False, True, "squat_hip_counter_athlete_params", id="rig-hip_counter-athlete_params"),
    pytest.param(True, False, False, "squat_hip_counter_first_time", id="rig-hip_counter-first_time"),
    pytest.param(
        True, True, True, "squat_bilstm_athlete_params", id="rig-bilstm-athlete_params", marks=_bilstm_model,
    ),
    pytest.param(False, False, True, "squat_single_camera_athlete_params", id="mono-hip_counter-athlete_params"),
]


class TestSquatGoldenMaster:
    @pytest.mark.parametrize("multi_camera, bilstm, athlete_params, golden", VARIANTS)
    def test_fresh_session_matches_golden(
        self, monkeypatch, multi_camera: bool, bilstm: bool, athlete_params: bool, golden: str,
    ):
        _check_golden(golden, _fresh_run(monkeypatch, multi_camera, bilstm, athlete_params))

    @pytest.mark.parametrize("multi_camera, bilstm, athlete_params, golden", VARIANTS)
    def test_session_after_switching_away_and_back_matches_the_fresh_golden(
        self, monkeypatch, multi_camera: bool, bilstm: bool, athlete_params: bool, golden: str,
    ):
        if os.getenv(REGEN_ENV) == "1":
            pytest.skip("the fresh run writes this golden")
        _check_golden(golden, _after_switch_run(monkeypatch, multi_camera, bilstm, athlete_params))

    @pytest.mark.parametrize("multi_camera, bilstm, athlete_params, golden", VARIANTS)
    def test_session_after_an_analyser_profile_and_back_matches_the_fresh_golden(
        self, monkeypatch, multi_camera: bool, bilstm: bool, athlete_params: bool, golden: str,
    ):
        if os.getenv(REGEN_ENV) == "1":
            pytest.skip("the fresh run writes this golden")
        monkeypatch.setitem(PROFILE_REGISTRY, FAKE_LIFT_PROFILE, _FakeLiftProfile)

        snapshot, fake_run = _after_analyser_switch_run(monkeypatch, multi_camera, bilstm, athlete_params)

        _check_golden(golden, snapshot)
        # The switch really went through every analyser hook and profile flag.
        assert isinstance(fake_run.analyzer, _FakeLiftAnalyzer)
        assert fake_run.analyzer.readiness_checks > 0
        assert fake_run.analyzer.baseline_seeds > 0
        assert fake_run.analyzer.reasons == [FAKE_REJECTION, None]
        by_type: dict[str, list[dict]] = {}
        for message in fake_run.messages:
            by_type.setdefault(message["type"], []).append(message)
        assert by_type["cache_cues"][0]["reports_depth"] is False
        assert [message["cue"] for message in by_type["shallow_rep"]] == [FAKE_SHALLOW_CUE]
        assert [message["max_depth_angle"] for message in by_type["rep_complete"]] == [None]
        assert [message.get("recorded_only") for message in by_type["fault"]] == [True]
        if multi_camera:
            assert fake_run.view_recording == [True, False, True]

    def test_golden_sessions_exercise_every_squat_outcome(self):
        expected = json.loads((GOLDEN_DIR / "squat_hip_counter_athlete_params.json").read_text())
        message_types = {message["type"] for message in expected["ipc"]}
        assert {"rep_complete", "fault", "shallow_rep", "set_complete", "diagnosis_complete"} <= message_types
        fault_types = {message["fault_type"] for message in expected["ipc"] if message["type"] == "fault"}
        assert "knee_valgus" in fault_types
