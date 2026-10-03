"""
Squat golden masters (deadlift PLAN §5, milestone J0): fixed synthetic squat sets
replayed through the real multi-camera BiomechanicsPipeline, SessionTracker and
IPCBridge, wired as pipeline_process wires them, and snapshotted to JSON. Each
scenario runs fresh, and after switching to another exercise and back.
Regenerate with NOWVA_UPDATE_GOLDENS=1 (whole module) and review the diff.
"""

from __future__ import annotations

import copy
import json
import math
import os
import types
from enum import Enum
from pathlib import Path
from typing import Any, Callable, NamedTuple

import numpy as np
import pytest
from pydantic import BaseModel

from biomechanics import pipeline as pipeline_module
from biomechanics.coaching.ipc_bridge import IPCBridge
from biomechanics.coaching.session_tracker import SessionTracker
from biomechanics.config import BiomechanicsConfig, IPCConfig
from biomechanics.pipeline_process import (
    DEFAULT_ROM_BASELINE,
    _adopt_measured_athlete_params,
    _switch_exercise,
)
from biomechanics.utils.segment_lengths import SegmentLengthEstimator
from biomechanics.utils.types import FaultEvent, PipelineFrame, RepData, Skeleton3D
from conftest import SYNTHETIC_FPS, squat_depth_profile, world_squat_points

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "squat_golden"
AFTER_SWITCH_DIR = GOLDEN_DIR / "after_switch"
UPDATE_GOLDENS_ENV = "NOWVA_UPDATE_GOLDENS"
SQUAT_EXERCISE = "Barbell Back Squat"

FRAME_DT_S = 1.0 / SYNTHETIC_FPS
CLOCK_START_S = 1.7e9
PROVIDER_CONFIDENCE = 0.9
READINESS_WARM_UP_FRAMES = 10
REPS_PER_SET = 3
REP_STAND_S = 0.5
REP_DOWN_S = 1.0
REP_HOLD_S = 0.4
REP_UP_S = 0.8
REP_ASCENT_FRAMES = int(REP_UP_S * SYNTHETIC_FPS)
SET_TAIL_FRAMES = 6
# A perfectly collinear synthetic leg reads exactly 0 deg of knee flexion;
# the squat's standing keeps the small residual bend test_pipeline uses.
MIN_DEPTH_RATIO = 0.02
# The synthetic feet toe out ~9 deg: knees driving straight ahead read as a
# mild cave (9 deg), so a clean rep tracks them out over the toes (1.4 deg).
KNEES_OVER_TOES_M = -0.06
VALGUS_SHIFT_M = 0.12
# Extra trunk pitch at mid-ascent: the chest drops while the hips rise.
HIP_SHOOT_LEAN_DEG = 30.0
# A descent to 60 % of the synthetic bottom stays well above the lenient
# uncalibrated depth target, so the default target rejects it.
SHALLOW_DEPTH = 0.6
SHALLOW_REP_INDEX = 1
DROPOUT_FRAMES = 3
DROPOUT_REP_INDEX = 1

STORED_PARAMS = "stored_params"
FIRST_TIME = "first_time"
VARIANTS = (STORED_PARAMS, FIRST_TIME)
# A returning athlete's stored calibration, which pipeline_process loads at startup.
ATHLETE_PARAMS = {
    "shoulder_width_m": 0.38,
    "femur_avg_m": 0.45,
    "torso_avg_m": 0.52,
    "hip_width_m": 0.24,
    "tibia_avg_m": 0.43,
    "foot_avg_m": 0.18,
}
STORED_BASELINE = {"peakDorsi": 33.0, "peakKneeFlex": 118.0, "peakHipFlex": 115.0}

# The other exercise before the squat: hip hinges with soft knees, locking out
# fully at the top (taller than the squat's soft-kneed stance).
HINGE_REPS = 2
HINGE_KNEE_DEPTH_RATIO = 0.15
HINGE_TRUNK_DEG = 70.0
HINGE_STAND_S = 0.3
HINGE_DOWN_S = 1.0
HINGE_HOLD_S = 0.2
HINGE_UP_S = 1.0
# Harness alignments, not carry-overs. The rest the workout plan puts between
# exercises: the bridge's per-fault-type cooldown (3 s) is session-scoped, so a
# switch with no rest could mute the squat's first fault of a type just sent.
REST_BETWEEN_EXERCISES_S = 30.0
# The bridge's frame_data throttle counter is session-scoped (prepare_exercise
# does not reset it): the other exercise's frames are padded to a multiple of
# the interval so frame_data lands on the same squat frames as when fresh.
FRAME_SEND_INTERVAL = IPCConfig().frame_send_interval

FLOAT_DECIMALS = 6
# One unit in the last stored decimal, plus rounding: forgives a value that
# lands on the other side of a rounding boundary, nothing more.
FLOAT_TOLERANCE = 1.5e-6
MAX_REPORTED_DIFFERENCES = 40


# ---------------------------------------------------------------------------
# Harness: fake clock, fake provider, recording IPC client, wired session
# ---------------------------------------------------------------------------

class _FakeClock:
    """Capture clock on a fixed frame grid. Tick 0 is the first squat frame, so a
    session that runs another exercise first (negative ticks) times its squat
    exactly as a fresh session does."""

    def __init__(self, start_tick: int) -> None:
        self.tick = start_tick

    def advance(self, ticks: int = 1) -> None:
        self.tick += ticks

    def time(self) -> float:
        return CLOCK_START_S + self.tick * FRAME_DT_S

    def perf_counter(self) -> float:
        return self.time()


class _FakeProvider:
    """Stands in for MultiCameraPoseProvider: world-frame skeletons stamped with a
    strictly increasing capture sequence, or None for a dropout."""

    def __init__(self, clock: _FakeClock) -> None:
        self._clock = clock
        self._queue: list[np.ndarray | None] = []
        self.sequence = 0
        self.frame = np.zeros((72, 128, 3), dtype=np.uint8)
        self.last_capture_timestamp = math.nan

    def push(self, points: np.ndarray | None) -> None:
        self._queue.append(points)

    def get_pose(self) -> tuple:
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


class _RecordingIPCClient:
    """Duck-types IPCClient: keeps every message with the squat frame it was sent on."""

    def __init__(self) -> None:
        self.frame = 0
        self.messages: list[tuple[int, dict]] = []

    def send_message(self, message: dict) -> None:
        self.messages.append((self.frame, copy.deepcopy(message)))


class _SquatSession:
    """One pipeline subprocess session, wired as pipeline_process.main wires it."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, variant: str, start_tick: int) -> None:
        monkeypatch.setenv("NOWVA_MULTI_CAMERA", "true")
        self.clock = _FakeClock(start_tick)
        monkeypatch.setattr(
            pipeline_module, "time",
            types.SimpleNamespace(time=self.clock.time, perf_counter=self.clock.perf_counter),
        )
        self.pipeline = pipeline_module.BiomechanicsPipeline(
            BiomechanicsConfig(), exercise_name=SQUAT_EXERCISE, defer_capture=True,
        )
        self.provider = _FakeProvider(self.clock)
        self.pipeline._multi_camera_provider = self.provider
        self.client = _RecordingIPCClient()
        self.bridge = IPCBridge(self.client)
        self.tracker = SessionTracker(self.bridge, config=self.pipeline.config.coaching)
        # Startup (pipeline_process.py:1012-1017).
        self.tracker.diagnosis_enabled = self.pipeline.profile.uses_diagnosis_engine
        self.bridge.prepare_exercise(SQUAT_EXERCISE)
        # A returning athlete's stored calibration (pipeline_process.py:1072-1080).
        self.athlete_params: dict | None = None
        self.baseline: dict = dict(DEFAULT_ROM_BASELINE)
        if variant == STORED_PARAMS:
            self.athlete_params = dict(ATHLETE_PARAMS)
            self.baseline = dict(STORED_BASELINE)
            self.pipeline.apply_athlete_params(self.athlete_params)
            self.tracker.set_athlete_params(self.athlete_params, self.baseline)
            self.bridge.set_athlete_params(self.athlete_params, self.baseline)

    def step(self, points: np.ndarray | None) -> PipelineFrame:
        """One main-loop iteration during a set (pipeline_process.py:1872-1975)."""
        self.provider.push(points)
        result = self.pipeline.process_frame()
        self.bridge.update_tracking_quality(
            result, active=self.pipeline.is_ready, in_rep=self.pipeline.rep_counter.in_rep,
        )
        self.bridge.update_camera_status(result)
        if self.athlete_params is None:
            self.athlete_params = _adopt_measured_athlete_params(
                self.pipeline, self.tracker, self.bridge, self.baseline,
            )
        self.bridge.send_frame_data(result, rep_phase=self.pipeline.rep_counter.phase)
        for fault in result.faults:
            if not fault.details.get("shallow_rep"):
                self.bridge.send_fault(fault)
        if result.shallow_rep_class is not None:
            shallow_fault = next((fault for fault in result.faults if fault.details.get("shallow_rep")), None)
            self.bridge.send_shallow_rep(
                result.shallow_rep_class, fault=shallow_fault,
                set_number=self.tracker.current_set_number,
            )
        if result.rep_data:
            bottom_kpts, bottom_angles = self.pipeline.consume_bottom_frame()
            standing_kpts = self.pipeline.consume_standing_frame()
            self.tracker.on_rep_complete(
                result.rep_data,
                bottom_kpts=bottom_kpts,
                bottom_angles=bottom_angles,
                standing_kpts=standing_kpts,
                trajectory_samples=self.pipeline.consume_rep_trajectory(),
            )
        if self.tracker.check_set_timeout(self.clock.time()):
            self.pipeline.reset_readiness_gate()
        self.clock.advance()
        return result

    def end_set(self) -> None:
        """rest_start reaching the pipeline closes the set (pipeline_process.py:1724-1728)."""
        if self.tracker.set_active:
            self.tracker.force_end_set()

    def switch_exercise(self, exercise_name: str) -> None:
        """set_exercise during the rest, through the real _switch_exercise, then
        the rest-end counter reset (pipeline_process.py:1747, :2009)."""
        _switch_exercise(self.pipeline, self.tracker, self.bridge, exercise_name)
        self.pipeline.rep_counter.reset()


# ---------------------------------------------------------------------------
# Snapshot: normalised and JSON-stable
# ---------------------------------------------------------------------------

# JSON-stable copy: floats rounded, NaN/inf -> None, capture frame indices
# relative to the squat's first capture.
def _normalise(value: Any, frame_offset: int = 0) -> Any:
    if isinstance(value, BaseModel):
        value = value.model_dump()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(value):
            return None
        rounded = round(float(value), FLOAT_DECIMALS)
        return 0.0 if rounded == 0.0 else rounded
    if isinstance(value, np.ndarray):
        return _normalise(value.tolist(), frame_offset)
    if isinstance(value, dict):
        return {
            str(key): (
                item - frame_offset
                if key in ("frame_index", "start_frame", "end_frame") and isinstance(item, int)
                else _normalise(item, frame_offset)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_normalise(item, frame_offset) for item in value]
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"A golden snapshot cannot hold {type(value).__name__}: {value!r}")


def _relative_time(timestamp_s: float) -> float | None:
    return _normalise(timestamp_s - CLOCK_START_S)


def _frame_row(frame_number: int, result: PipelineFrame, session: _SquatSession, frame_offset: int) -> str:
    if result.joint_angles is None:
        analysis = "gated"
    elif result.skeleton_3d_raw is None:
        analysis = "predicted"
    else:
        analysis = "analysed"
    row = (
        f"{frame_number:03d} capture={result.frame_index - frame_offset} {analysis} "
        f"phase={session.pipeline.rep_counter.phase} reps={session.pipeline.rep_count}"
    )
    if result.faults:
        row += " faults=" + ",".join(fault.fault_type for fault in result.faults)
    if result.rep_data is not None:
        row += f" rep_complete={result.rep_data.rep_number}"
    if result.shallow_rep_class is not None:
        row += f" shallow_class={result.shallow_rep_class}"
    return row


def _fault_entry(frame_number: int, fault: FaultEvent, frame_offset: int) -> dict:
    return {
        "frame": frame_number,
        "fault_type": fault.fault_type,
        "severity": fault.severity.value,
        "severity_score": _normalise(fault.severity_score),
        "message": fault.message,
        "rep_number": fault.rep_number,
        "frame_index": fault.frame_index - frame_offset,
        "t_s": _relative_time(fault.timestamp),
        "details": _normalise(fault.details, frame_offset),
    }


def _rep_entry(frame_number: int, rep: RepData, frame_offset: int) -> dict:
    return {
        "frame": frame_number,
        "rep_number": rep.rep_number,
        "start_t_s": _relative_time(rep.start_time),
        "end_t_s": _relative_time(rep.end_time),
        "start_frame": rep.start_frame - frame_offset,
        "end_frame": rep.end_frame - frame_offset,
        "max_depth_angle": _normalise(rep.max_depth_angle),
        "min_depth_angle": _normalise(rep.min_depth_angle),
        "descent_time": _normalise(rep.descent_time),
        "ascent_time": _normalise(rep.ascent_time),
        "depth_target_met": rep.depth_target_met,
        "is_clean": rep.is_clean,
        "fault_types": [fault.fault_type for fault in rep.faults],
        "features": _normalise(rep.features, frame_offset),
    }


# ---------------------------------------------------------------------------
# Squat scenarios
# ---------------------------------------------------------------------------

def _standing_frames(count: int) -> list[np.ndarray | None]:
    return [world_squat_points(MIN_DEPTH_RATIO) for _ in range(count)]


def _squat_rep_depths(depth_scale: float = 1.0) -> list[float]:
    profile = squat_depth_profile(stand_s=REP_STAND_S, down_s=REP_DOWN_S, hold_s=REP_HOLD_S, up_s=REP_UP_S)
    return [depth * depth_scale for depth in profile]


# Warm-up, the reps (lean_fn(frame in rep, rep length) adds trunk pitch), then standing.
def _squat_set_frames(
    rep_depths: list[list[float]],
    knee_shift_m: float = KNEES_OVER_TOES_M,
    lean_fn: Callable[[int, int], float] | None = None,
) -> list[np.ndarray | None]:
    frames = _standing_frames(READINESS_WARM_UP_FRAMES)
    for depths in rep_depths:
        for frame_in_rep, depth in enumerate(depths):
            lean_deg = lean_fn(frame_in_rep, len(depths)) if lean_fn is not None else 0.0
            frames.append(world_squat_points(
                max(depth, MIN_DEPTH_RATIO), valgus_m=knee_shift_m, lean_extra_deg=lean_deg,
            ))
    return frames + _standing_frames(SET_TAIL_FRAMES)


# The trunk pitches forward over the ascent: 0 at the bottom and back to 0 at the top.
def _ascent_lean_deg(frame_in_rep: int, rep_frames: int) -> float:
    ascent_start = rep_frames - REP_ASCENT_FRAMES
    if frame_in_rep < ascent_start:
        return 0.0
    return HIP_SHOOT_LEAN_DEG * math.sin(math.pi * (frame_in_rep - ascent_start) / REP_ASCENT_FRAMES)


def _clean_frames() -> list[np.ndarray | None]:
    return _squat_set_frames([_squat_rep_depths() for _ in range(REPS_PER_SET)])


def _knee_valgus_frames() -> list[np.ndarray | None]:
    return _squat_set_frames([_squat_rep_depths() for _ in range(REPS_PER_SET)], knee_shift_m=VALGUS_SHIFT_M)


def _hip_shoot_frames() -> list[np.ndarray | None]:
    return _squat_set_frames([_squat_rep_depths() for _ in range(REPS_PER_SET)], lean_fn=_ascent_lean_deg)


def _shallow_descent_frames() -> list[np.ndarray | None]:
    reps = [_squat_rep_depths() for _ in range(REPS_PER_SET)]
    reps[SHALLOW_REP_INDEX] = _squat_rep_depths(SHALLOW_DEPTH)
    return _squat_set_frames(reps)


# Captures lost for a few frames halfway down the second rep.
def _dropout_frames() -> list[np.ndarray | None]:
    frames = _clean_frames()
    rep_frames = len(_squat_rep_depths())
    mid_descent = (
        READINESS_WARM_UP_FRAMES + DROPOUT_REP_INDEX * rep_frames
        + int(REP_STAND_S * SYNTHETIC_FPS) + int(REP_DOWN_S * SYNTHETIC_FPS) // 2
    )
    for index in range(mid_descent, mid_descent + DROPOUT_FRAMES):
        frames[index] = None
    return frames


class _Scenario(NamedTuple):
    name: str
    frames_fn: Callable[[], list[np.ndarray | None]]
    # set_depth_target(None) at startup: every descent counts, as test_pipeline's harness does.
    count_every_descent: bool = False


SCENARIOS = (
    _Scenario("clean", _clean_frames),
    _Scenario("knee_valgus", _knee_valgus_frames),
    _Scenario("hip_shoot", _hip_shoot_frames),
    _Scenario("shallow_descent", _shallow_descent_frames),
    _Scenario("shallow_descent_no_depth_target", _shallow_descent_frames, count_every_descent=True),
    _Scenario("dropout", _dropout_frames),
)
SCENARIOS_BY_NAME = {scenario.name: scenario for scenario in SCENARIOS}


# ---------------------------------------------------------------------------
# After a switch: the other exercises and the state they leave behind today
# ---------------------------------------------------------------------------

def _cosine(start: float, end: float, seconds: float) -> list[float]:
    frames = int(round(seconds * SYNTHETIC_FPS))
    return [start + (end - start) * 0.5 * (1.0 - math.cos(math.pi * (i + 1) / frames)) for i in range(frames)]


# An athlete in view, upright, doing a few hip hinges with soft knees.
def _hinge_frames() -> list[np.ndarray | None]:
    hinge: list[float] = [0.0] * READINESS_WARM_UP_FRAMES
    for _ in range(HINGE_REPS):
        hinge += [0.0] * int(HINGE_STAND_S * SYNTHETIC_FPS)
        hinge += _cosine(0.0, 1.0, HINGE_DOWN_S)
        hinge += [1.0] * int(HINGE_HOLD_S * SYNTHETIC_FPS)
        hinge += _cosine(1.0, 0.0, HINGE_UP_S)
    hinge += [0.0] * int(HINGE_STAND_S * SYNTHETIC_FPS)
    return [
        world_squat_points(HINGE_KNEE_DEPTH_RATIO * amount, lean_extra_deg=HINGE_TRUNK_DEG * amount)
        for amount in hinge
    ]


def _body_measurements(pipeline: pipeline_module.BiomechanicsPipeline) -> SegmentLengthEstimator:
    return copy.deepcopy(pipeline.body_calibration)


def _body_measurement_signature(estimator: SegmentLengthEstimator) -> tuple:
    return estimator.progress, estimator.provisional_athlete_params(), estimator.to_athlete_params()


def _transplant_body_measurements(
    pipeline: pipeline_module.BiomechanicsPipeline, estimator: SegmentLengthEstimator,
) -> None:
    pipeline.body_calibration = copy.deepcopy(estimator)
    # What set_exercise does with the measurement on the switch back.
    pipeline._apply_body_proportions()


def _standing_reference(pipeline: pipeline_module.BiomechanicsPipeline) -> float:
    return pipeline._standing_reference_hip_cm


def _transplant_standing_reference(pipeline: pipeline_module.BiomechanicsPipeline, height_cm: float) -> None:
    pipeline._standing_reference_hip_cm = height_cm


class _CarryOver(NamedTuple):
    capture_fn: Callable[[Any], Any]
    signature_fn: Callable[[Any], Any]
    transplant_fn: Callable[[Any, Any], None]


BODY_MEASUREMENTS = "body_measurements"
STANDING_REFERENCE = "standing_reference"

# ALLOW-LIST. Session state another exercise leaves for the squat today
# (PLAN §5.2). For a case that lists some, the expected output is that of a
# fresh session started with exactly this state, captured at the switch back.
# Regenerating proves that live and records it as edits over the fresh golden
# (fixtures/squat_golden/after_switch/), so the review shows what each changes.
CARRY_OVERS: dict[str, _CarryOver] = {
    # pipeline.body_calibration is session-scoped and fed on every analysed
    # frame of every exercise (pipeline.py:749-752, :950). A first-time athlete
    # reaches the squat with the other exercise's samples: after two RDL reps
    # the measurement is complete and already adopted (stance fields in every
    # frame_data, kinematics from the first squat rep); after untracked bench
    # frames the samples sit at rep count 0 and complete on the first squat
    # rep instead of the second. Stored params complete it at startup, so it
    # is never fed. PLAN §5.2: feeds_body_calibration=False for the deadlift.
    BODY_MEASUREMENTS: _CarryOver(_body_measurements, _body_measurement_signature, _transplant_body_measurements),
    # pipeline._standing_reference_hip_cm is reset by neither set_exercise nor
    # reset_readiness_gate, and only rises, in _start_rep_setup, which every
    # rep of a tracked profile runs (pipeline.py:630-641). The RDL's full
    # lockout stands taller than the squat's stance, so the squat's
    # lockout_deficit_ratio is measured against the RDL top. An untracked
    # exercise counts no reps and leaves it alone.
    STANDING_REFERENCE: _CarryOver(_standing_reference, _normalise, _transplant_standing_reference),
}

# ALLOW-LIST. set_depth_target(None) (every descent counts; the assessment and
# calibration phases use it) does not survive a switch: set_exercise keeps
# only a non-None target (pipeline.py:586-587), and the rebuilt squat falls
# back to faults.depth.uncalibrated_target_ratio (pipeline.py:393-397). The
# squat after the switch is the same frames under the uncalibrated target.
AFTER_SWITCH_EXPECTED_SCENARIO = {"shallow_descent_no_depth_target": "shallow_descent"}


class _OtherExercise(NamedTuple):
    exercise_name: str
    frames_fn: Callable[[], list[np.ndarray | None]]
    # CARRY_OVERS keys this exercise leaves for the squat today, per variant.
    carries: dict[str, tuple[str, ...]]


# EXTENSION POINT (J3): add
#     _OtherExercise("Barbell Conventional Deadlift", _deadlift_frames,
#                    {STORED_PARAMS: (), FIRST_TIME: ()}),
# with _deadlift_frames the J2 simulator's synthetic deadlift reps (bar on the
# floor), and patch monkeypatch.setattr(DeadliftProfile, "coaching_ready", True)
# before _after_switch_session so the switch reaches the real profile, not the
# gated one (PLAN §3.1). PLAN §5.2 targets no carry-over at all for it: no body
# measurement (feeds_body_calibration=False) and no standing reference (it
# skips _start_rep_setup). Today the name resolves to the placeholder profile.
OTHER_EXERCISES = (
    _OtherExercise(
        "Barbell Romanian Deadlift", _hinge_frames,
        {STORED_PARAMS: (STANDING_REFERENCE,), FIRST_TIME: (STANDING_REFERENCE, BODY_MEASUREMENTS)},
    ),
    # Untracked: no reps and no faults, but the camera keeps measuring the body.
    _OtherExercise(
        "Barbell Bench Press", _hinge_frames,
        {STORED_PARAMS: (), FIRST_TIME: (BODY_MEASUREMENTS,)},
    ),
)


# ---------------------------------------------------------------------------
# Running sessions
# ---------------------------------------------------------------------------

# The squat set from tick 0, then its close; returns the snapshot.
def _run_squat(session: _SquatSession, scenario: _Scenario) -> dict:
    assert session.clock.tick == 0
    frame_offset = session.provider.sequence
    frames: list[str] = []
    faults: list[dict] = []
    reps: list[dict] = []
    scenario_frames = scenario.frames_fn()
    for frame_number, points in enumerate(scenario_frames):
        session.client.frame = frame_number
        result = session.step(points)
        frames.append(_frame_row(frame_number, result, session, frame_offset))
        faults.extend(_fault_entry(frame_number, fault, frame_offset) for fault in result.faults)
        if result.rep_data is not None:
            reps.append(_rep_entry(frame_number, result.rep_data, frame_offset))
    session.client.frame = len(scenario_frames)
    session.end_set()
    return {
        "frames": frames,
        "faults": faults,
        "reps": reps,
        "ipc": [
            {"frame": frame, "message": _normalise(message, frame_offset)}
            for frame, message in session.client.messages
        ],
    }


def _fresh_session(monkeypatch: pytest.MonkeyPatch, scenario: _Scenario, variant: str) -> _SquatSession:
    session = _SquatSession(monkeypatch, variant, start_tick=0)
    if scenario.count_every_descent:
        session.pipeline.set_depth_target(None)
    return session


# A fresh session given exactly the carried state. Each listed carry-over must
# still be live: different from what a fresh session starts with.
def _transplanted_session(
    monkeypatch: pytest.MonkeyPatch, scenario: _Scenario, variant: str,
    other: _OtherExercise, carried: dict[str, Any],
) -> _SquatSession:
    session = _fresh_session(monkeypatch, scenario, variant)
    for name, state in carried.items():
        carry_over = CARRY_OVERS[name]
        fresh_state = carry_over.capture_fn(session.pipeline)
        if carry_over.signature_fn(state) == carry_over.signature_fn(fresh_state):
            pytest.fail(
                f"{other.exercise_name} no longer leaves '{name}' behind ({variant}): "
                f"remove it from that entry's carries in OTHER_EXERCISES and regenerate"
            )
        carry_over.transplant_fn(session.pipeline, state)
    return session


def _other_exercise_frames(other: _OtherExercise) -> list[np.ndarray | None]:
    frames = other.frames_fn()
    return frames + _standing_frames(-len(frames) % FRAME_SEND_INTERVAL)


# Startup on the squat, the switch to the other exercise, its set, the rest and
# the switch back: ready for the squat set at tick 0, with only the switch
# back's messages recorded.
def _after_switch_session(
    monkeypatch: pytest.MonkeyPatch, scenario: _Scenario, variant: str, other: _OtherExercise,
) -> _SquatSession:
    other_frames = _other_exercise_frames(other)
    rest_ticks = int(round(REST_BETWEEN_EXERCISES_S * SYNTHETIC_FPS))
    session = _SquatSession(monkeypatch, variant, start_tick=-(len(other_frames) + rest_ticks))
    if scenario.count_every_descent:
        session.pipeline.set_depth_target(None)
    session.switch_exercise(other.exercise_name)
    for points in other_frames:
        session.step(points)
    session.end_set()
    session.clock.advance(rest_ticks)
    session.client.messages.clear()
    session.switch_exercise(SQUAT_EXERCISE)
    return session


# ---------------------------------------------------------------------------
# Golden files and differences
# ---------------------------------------------------------------------------

def _golden_path(scenario: _Scenario, variant: str) -> Path:
    return GOLDEN_DIR / f"{scenario.name}__{variant}.json"


# Pretty JSON with sorted keys; lists of scalars stay on one line.
def _format_json(value: Any, indent: int = 0) -> str:
    pad = " " * indent
    if isinstance(value, dict):
        if not value:
            return "{}"
        items = [f"{pad} {json.dumps(key)}: {_format_json(value[key], indent + 1)}" for key in sorted(value)]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(value, list) and any(isinstance(item, (dict, list)) for item in value):
        items = [f"{pad} {_format_json(item, indent + 1)}" for item in value]
        return "[\n" + ",\n".join(items) + "\n" + pad + "]"
    return json.dumps(value)


def _short(value: Any) -> str:
    text = json.dumps(value)
    return text if len(text) <= 160 else text[:157] + "..."


def _differences(expected: Any, actual: Any, path: str = "$") -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        found: list[str] = []
        for key in sorted(set(expected) | set(actual)):
            if key not in actual:
                found.append(f"{path}.{key}: missing (expected {_short(expected[key])})")
            elif key not in expected:
                found.append(f"{path}.{key}: unexpected {_short(actual[key])}")
            else:
                found.extend(_differences(expected[key], actual[key], f"{path}.{key}"))
        return found
    if isinstance(expected, list) and isinstance(actual, list):
        found = []
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            found.extend(_differences(expected_item, actual_item, f"{path}[{index}]"))
        if len(expected) != len(actual):
            found.append(f"{path}: expected {len(expected)} items, got {len(actual)}")
        return found
    both_numbers = all(
        isinstance(value, (int, float)) and not isinstance(value, bool) for value in (expected, actual)
    )
    if both_numbers:
        return [] if abs(expected - actual) <= FLOAT_TOLERANCE else [f"{path}: expected {expected}, got {actual}"]
    return [] if expected == actual else [f"{path}: expected {_short(expected)}, got {_short(actual)}"]


def _report(differences: list[str], title: str) -> str:
    shown = differences[:MAX_REPORTED_DIFFERENCES]
    lines = [title, *shown]
    if len(differences) > len(shown):
        lines.append(f"... and {len(differences) - len(shown)} more")
    return "\n".join(lines)


def _updating_goldens() -> bool:
    return os.environ.get(UPDATE_GOLDENS_ENV) == "1"


def _read_fixture(path: Path) -> Any:
    if not path.exists():
        pytest.fail(f"Missing golden {path.name}: run with {UPDATE_GOLDENS_ENV}=1 to create it")
    return json.loads(path.read_text())


def _write_fixture(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_format_json(value) + "\n")


def _load_golden(scenario: _Scenario, variant: str) -> dict:
    return _read_fixture(_golden_path(scenario, variant))


def _after_switch_patch_path(scenario: _Scenario, variant: str, other: _OtherExercise) -> Path:
    return AFTER_SWITCH_DIR / f"{scenario.name}__{variant}__{_other_exercise_id(other)}.json"


# Leaf edits turning base into target: a value set at a path, or a key deleted.
def _edits(base: Any, target: Any, path: tuple = ()) -> list[dict]:
    if isinstance(base, dict) and isinstance(target, dict):
        found: list[dict] = []
        for key in sorted(set(base) | set(target)):
            if key not in target:
                found.append({"path": [*path, key], "delete": True})
            elif key not in base:
                found.append({"path": [*path, key], "value": target[key]})
            else:
                found.extend(_edits(base[key], target[key], (*path, key)))
        return found
    if isinstance(base, list) and isinstance(target, list) and len(base) == len(target):
        found = []
        for index, (base_item, target_item) in enumerate(zip(base, target)):
            found.extend(_edits(base_item, target_item, (*path, index)))
        return found
    return [] if base == target else [{"path": list(path), "value": target}]


def _apply_edits(base: Any, edits: list[dict]) -> Any:
    patched = copy.deepcopy(base)
    for edit in edits:
        *parents, last = edit["path"]
        container = patched
        for key in parents:
            container = container[key]
        if edit.get("delete"):
            del container[last]
        else:
            container[last] = copy.deepcopy(edit["value"])
    return patched


def _golden_named(name: str, variant: str) -> dict:
    return _load_golden(SCENARIOS_BY_NAME[name], variant)


def _message_types(golden: dict) -> list[str]:
    return [entry["message"]["type"] for entry in golden["ipc"]]


def _rep_numbers(golden: dict) -> list[int]:
    return [rep["rep_number"] for rep in golden["reps"]]


def _reps_with_kinematics(golden: dict) -> list[bool]:
    return [
        "rep_kinematic_summary" in entry["message"]
        for entry in golden["ipc"] if entry["message"]["type"] == "rep_complete"
    ]


def _scenario_id(scenario: _Scenario) -> str:
    return scenario.name


def _other_exercise_id(other: _OtherExercise) -> str:
    return other.exercise_name.lower().replace(" ", "_")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("scenario", SCENARIOS, ids=_scenario_id)
class TestFreshSquatGolden:
    def test_fresh_squat_matches_golden(self, monkeypatch, scenario: _Scenario, variant: str):
        snapshot = _run_squat(_fresh_session(monkeypatch, scenario, variant), scenario)
        path = _golden_path(scenario, variant)
        if _updating_goldens():
            _write_fixture(path, snapshot)
            return
        differences = _differences(_load_golden(scenario, variant), snapshot)
        if differences:
            pytest.fail(_report(
                differences,
                f"The fresh squat differs from {path.name} ({len(differences)} differences). "
                f"If the change is intended, regenerate with {UPDATE_GOLDENS_ENV}=1 and review the diff.",
            ))


@pytest.mark.parametrize("other", OTHER_EXERCISES, ids=_other_exercise_id)
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("scenario", SCENARIOS, ids=_scenario_id)
class TestSquatGoldenAfterSwitch:
    def test_squat_after_switch_matches_fresh_apart_from_carry_overs(
        self, monkeypatch, scenario: _Scenario, variant: str, other: _OtherExercise,
    ):
        session = _after_switch_session(monkeypatch, scenario, variant, other)
        carried = {name: CARRY_OVERS[name].capture_fn(session.pipeline) for name in other.carries[variant]}
        actual = _run_squat(session, scenario)

        expected_scenario = SCENARIOS_BY_NAME[AFTER_SWITCH_EXPECTED_SCENARIO.get(scenario.name, scenario.name)]
        patch_path = _after_switch_patch_path(scenario, variant, other)
        # Built after the switched session has run: each session patches the clock.
        transplanted = _transplanted_session(monkeypatch, expected_scenario, variant, other, carried)
        if _updating_goldens() and carried:
            expected = _run_squat(transplanted, expected_scenario)
            fresh = _run_squat(_fresh_session(monkeypatch, expected_scenario, variant), expected_scenario)
            _write_fixture(patch_path, {"carries": sorted(carried), "edits": _edits(fresh, expected)})
        else:
            if _updating_goldens():
                patch_path.unlink(missing_ok=True)
            expected = _load_golden(expected_scenario, variant)
            if carried:
                patch = _read_fixture(patch_path)
                if patch["carries"] != sorted(carried):
                    pytest.fail(f"{patch_path.name} records {patch['carries']}: regenerate with {UPDATE_GOLDENS_ENV}=1")
                expected = _apply_edits(expected, patch["edits"])

        differences = _differences(expected, actual)
        if differences:
            pytest.fail(_report(
                differences,
                f"The squat after {other.exercise_name} differs from a fresh session beyond the "
                f"documented carry-overs {sorted(carried) or 'none'} ({len(differences)} differences). "
                f"A new carry-over is a squat regression; fix it rather than allow-listing it.",
            ))


@pytest.mark.parametrize("variant", VARIANTS)
class TestGoldenScenariosKeepTheirPoint:
    """A regenerated golden must still exercise what its scenario is for."""

    def test_clean_set_has_three_fault_free_reps(self, variant: str):
        golden = _golden_named("clean", variant)
        assert golden["faults"] == []
        assert _rep_numbers(golden) == [1, 2, 3]
        assert all(rep["is_clean"] for rep in golden["reps"])

    def test_knee_valgus_caves_severely_on_every_rep(self, variant: str):
        golden = _golden_named("knee_valgus", variant)
        faults = [(fault["fault_type"], fault["severity"], fault["rep_number"]) for fault in golden["faults"]]
        assert faults == [("knee_valgus", "severe", rep_number) for rep_number in (1, 2, 3)]

    def test_hip_shoot_fires_on_every_rep(self, variant: str):
        golden = _golden_named("hip_shoot", variant)
        faults = [(fault["fault_type"], fault["rep_number"]) for fault in golden["faults"]]
        assert faults == [("hip_shoot", rep_number) for rep_number in (1, 2, 3)]

    def test_shallow_descent_is_refused_under_the_default_target(self, variant: str):
        golden = _golden_named("shallow_descent", variant)
        assert _rep_numbers(golden) == [1, 2]
        assert [fault["fault_type"] for fault in golden["faults"]] == ["depth"]
        assert _message_types(golden).count("shallow_rep") == 1

    def test_shallow_descent_counts_without_a_depth_target(self, variant: str):
        golden = _golden_named("shallow_descent_no_depth_target", variant)
        assert _rep_numbers(golden) == [1, 2, 3]
        assert "shallow_rep" not in _message_types(golden)

    def test_dropout_is_carried_by_prediction(self, variant: str):
        golden = _golden_named("dropout", variant)
        assert sum(" predicted " in row for row in golden["frames"]) == DROPOUT_FRAMES
        assert _rep_numbers(golden) == [1, 2, 3]

    def test_every_set_opens_with_its_cues_and_closes_with_a_diagnosis(self, variant: str):
        for scenario in SCENARIOS:
            message_types = _message_types(_load_golden(scenario, variant))
            assert message_types[0] == "cache_cues"
            assert message_types[-2:] == ["set_complete", "diagnosis_complete"]


class TestGoldenVariants:
    def test_first_time_athlete_is_measured_during_the_set(self):
        """Without stored params the first rep has no kinematics; the body is measured by the second."""
        assert _reps_with_kinematics(_golden_named("clean", STORED_PARAMS)) == [True, True, True]
        assert _reps_with_kinematics(_golden_named("clean", FIRST_TIME)) == [False, True, True]


class TestGoldenHarness:
    def test_edits_turn_the_base_into_the_target(self):
        base = {"kept": [1, {"changed": "a", "dropped": True}], "grown": [1, 2], "gone": 1}
        target = {"kept": [1, {"changed": "b", "added": None}], "grown": [1, 2, 3], "new": {"x": 1}}
        assert _apply_edits(base, _edits(base, target)) == target

    def test_differences_forgive_one_rounding_step_only(self):
        assert _differences({"value": 1.000001}, {"value": 1.000002}) == []
        assert _differences({"value": 1.0}, {"value": 1.00001}) == ["$.value: expected 1.0, got 1.00001"]

    def test_normalise_rounds_drops_non_finite_and_rebases_frames(self):
        normalised = _normalise({"angle": 1.23456789, "missing": float("nan"), "frame_index": 112}, frame_offset=100)
        assert normalised["angle"] == pytest.approx(1.234568, abs=FLOAT_TOLERANCE)
        assert normalised["missing"] is None
        assert normalised["frame_index"] == 12
