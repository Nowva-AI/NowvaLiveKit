"""
The RDL alongside the squat and the replay tooling (SHARED-PLUMBING.md §8,
rdl.md §7 Phase D): RDL reps, then a switch to the squat, reproduce the fresh
squat golden master; a recorded RDL session replays to the same reps, faults
and shallow reps.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from biomechanics.config import BiomechanicsConfig
from biomechanics.profiles.romanian_deadlift import RomanianDeadliftProfile
from biomechanics.replay.raw_skeleton import replay_recording
from biomechanics.utils.types import CocoKeypoints as CK
from conftest import SYNTHETIC_MIN_DEPTH_RATIO, SyntheticSession, world_squat_points
from test_rdl_poses import CLEAN_BOTTOM, STANDING, RdlPose, hinge_for_travel, rdl_rep
from test_squat_golden_master import (
    CLOCK_START_S,
    FRAME_DT_S,
    SQUAT,
    _check_golden,
    _Session,
    _squat_session,
    _start,
    _switch,
)

RDL_NAME = "Barbell Romanian Deadlift"
RIG_FPS = 30.0
MONO_FPS = 12.0
# A whole number of frame_data intervals, so the bridge's send cadence after
# the switch lines up with a fresh run.
FRAME_DATA_INTERVAL = 10
RDL_LEAD_S = 40.0
SQUATTED_BOTTOM = RdlPose(70.0, 60.0, 15.0)
PARTIAL_TRAVEL = 0.35
REST_FRAMES = 20
FOOT_KEYPOINTS = (
    CK.LEFT_ANKLE, CK.RIGHT_ANKLE, CK.LEFT_FOOT_INDEX, CK.RIGHT_FOOT_INDEX, CK.LEFT_HEEL, CK.RIGHT_HEEL,
)


def _on_squat_feet(points: np.ndarray) -> np.ndarray:
    # The same lifter in the same stance: foot contact anchors are
    # session-scoped, so the squat must find the feet where the RDL left them.
    feet = list(FOOT_KEYPOINTS)
    placed = points.copy()
    placed[feet] = world_squat_points(SYNTHETIC_MIN_DEPTH_RATIO)[feet]
    return placed


def _rdl_frames(fps: float) -> list[RdlPose]:
    partial = RdlPose(hinge_for_travel(PARTIAL_TRAVEL, 20.0, 5.0), 20.0, 5.0)
    poses = [STANDING] * 10 + rdl_rep(fps) + rdl_rep(fps, bottom=SQUATTED_BOTTOM)
    poses += rdl_rep(fps, down_s=0.5) + rdl_rep(fps, bottom=partial) + rdl_rep(fps, bottom=partial)
    return poses + [STANDING] * int(fps)


class TestRdlThenSquat:
    def test_squat_after_rdl_sets_matches_the_fresh_golden(self, monkeypatch):
        monkeypatch.setattr(RomanianDeadliftProfile, "coaching_ready", True)
        session = _Session(monkeypatch, True, False, True, CLOCK_START_S - RDL_LEAD_S)
        _start(session.pipeline, session.tracker, session.bridge, SQUAT)
        _switch(session.pipeline, session.tracker, session.bridge, RDL_NAME)
        assert type(session.pipeline.profile) is RomanianDeadliftProfile

        poses = _rdl_frames(RIG_FPS)
        poses += [STANDING] * (-len(poses) % FRAME_DATA_INTERVAL)
        reps = 0
        for pose in poses:
            session.provider.push(_on_squat_feet(pose.points()))
            result = session.pipeline.process_frame()
            session._deliver(result)
            reps += result.rep_data is not None
            session.clock.advance(FRAME_DT_S)
        assert reps == 3
        session.tracker.force_end_set()

        _switch(session.pipeline, session.tracker, session.bridge, SQUAT)
        session.rewind(CLOCK_START_S)
        ipc_from = len(session.client.messages)
        _squat_session(session)
        _check_golden("squat_hip_counter_athlete_params", session.snapshot(ipc_from))


def _outcome(frames: list) -> dict:
    return {
        "reps": [(frame.timestamp, frame.rep_data.rep_number) for frame in frames if frame.rep_data is not None],
        "faults": [
            (fault.timestamp, fault.fault_type, fault.severity.value, fault.rep_number,
             bool(fault.details.get("recorded_only")))
            for frame in frames for fault in frame.faults
        ],
        "shallow": [(frame.timestamp, frame.shallow_rep_cue) for frame in frames if frame.analyzer_shallow_rep],
    }


class TestRdlReplay:
    @pytest.mark.parametrize("multi_camera, fps", [(False, MONO_FPS), (True, RIG_FPS)], ids=["mono", "rig"])
    def test_a_recorded_rdl_session_replays_exactly(self, monkeypatch, tmp_path: Path, multi_camera: bool, fps: float):
        monkeypatch.setattr(RomanianDeadliftProfile, "coaching_ready", True)
        path = tmp_path / "rdl_raw_skeleton.jsonl"
        session = SyntheticSession(monkeypatch, multi_camera=multi_camera, exercise_name=RDL_NAME)
        pipeline = session.pipeline
        pipeline.enable_raw_recording(path)
        live = [session.step(pose.points(), fps=fps) for pose in _rdl_frames(fps)]
        pipeline.record_event("rest_start", rest_seconds=60)
        pipeline.presence_only = True
        live += [session.step(STANDING.points(), fps=fps) for _ in range(REST_FRAMES)]
        pipeline.presence_only = False
        pipeline.reset_readiness_gate()
        pipeline.reset_rep_counter()
        live += [session.step(pose.points(), fps=fps) for pose in [STANDING] * 10 + rdl_rep(fps, bottom=CLEAN_BOTTOM)]
        pipeline.release()

        replayed = replay_recording(path, BiomechanicsConfig())

        expected = _outcome(live)
        assert expected["faults"] and expected["shallow"]
        assert _outcome(replayed.frames) == expected
        assert replayed.reps == [1, 2, 3, 1]
