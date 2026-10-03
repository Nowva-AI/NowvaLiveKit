"""
Press replay (docs/exercises/ohp.md §7.1 Step 7, SHARED-PLUMBING.md §8): a
single-camera press session — walk-in, unrack, a set with a missed rep and a
re-rack, rest, a second set — recorded raw and replayed through a fresh pipeline
reproduces the latch, rep count, rep timestamps and fault list exactly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from biomechanics.config import BiomechanicsConfig
from biomechanics.replay.raw_skeleton import load_recording, replay_recording
from test_press_poses import (
    FPS_MONO,
    hands_down,
    hold,
    press_rep,
    press_session,
    run_session,
)

REST_FRAMES = 24
MISSED_TOP_M = 0.30
SEVERE_TILT_DEG = 10.0
DIP_M = 0.07


def _config(certified: bool) -> BiomechanicsConfig:
    config = BiomechanicsConfig()
    if certified:
        config.capture.mount_source = "inclinometer"
        config.capture.mount_pitch_deg = 0.0
    return config


def _live_session(monkeypatch, path: Path, config: BiomechanicsConfig) -> list:
    session = press_session(monkeypatch, config=config)
    pipeline = session.pipeline
    pipeline.enable_raw_recording(path)
    set_one = hands_down(1.0, FPS_MONO) + hold(1.5, FPS_MONO) + press_rep(FPS_MONO)
    set_one += press_rep(FPS_MONO, tilt_top_deg=SEVERE_TILT_DEG) + press_rep(FPS_MONO, top_m=MISSED_TOP_M)
    set_one += press_rep(FPS_MONO, dip_m=DIP_M) + hold(1.0, FPS_MONO) + hands_down(2.0, FPS_MONO)
    results = run_session(session, set_one, FPS_MONO)
    pipeline.record_event("rest_start", rest_seconds=60)
    pipeline.presence_only = True
    results += run_session(session, hands_down(REST_FRAMES / FPS_MONO, FPS_MONO), FPS_MONO)
    pipeline.presence_only = False
    pipeline.reset_readiness_gate()
    pipeline.reset_rep_counter()
    set_two = hands_down(1.0, FPS_MONO) + hold(1.5, FPS_MONO) + press_rep(FPS_MONO) + press_rep(FPS_MONO)
    results += run_session(session, set_two + hold(1.0, FPS_MONO), FPS_MONO)
    pipeline.release()
    return results


def _outcome(frames: list) -> dict:
    return {
        "ready": [frame.skeleton_3d is not None for frame in frames],
        "reps": [(frame.timestamp, frame.rep_data.rep_number) for frame in frames if frame.rep_data is not None],
        "faults": [
            (fault.timestamp, fault.fault_type, fault.severity.value, fault.rep_number, bool(fault.details.get("recorded_only")))
            for frame in frames for fault in frame.faults
        ],
    }


class TestPressReplay:
    @pytest.mark.parametrize("certified", [False, True], ids=["uncertified", "level-mount"])
    def test_replay_reproduces_the_live_press_session(self, monkeypatch, tmp_path: Path, certified: bool):
        path = tmp_path / "press_session.jsonl"
        config = _config(certified)
        live = _live_session(monkeypatch, path, config)

        replayed = replay_recording(path, config)

        live_outcome = _outcome(live)
        assert [rep for _, rep in live_outcome["reps"]] == [1, 2, 3, 1, 2]
        fault_types = {fault[1] for fault in live_outcome["faults"]}
        assert {"press_missed", "press_uneven", "press_leg_drive"} <= fault_types
        assert len(replayed.frames) == len(live)
        assert _outcome(replayed.frames) == live_outcome

    def test_recording_starts_before_the_latch_and_keeps_the_control_events(self, monkeypatch, tmp_path: Path):
        path = tmp_path / "press_session.jsonl"
        live = _live_session(monkeypatch, path, _config(False))
        recording = load_recording(path)
        assert len(recording.frames) == len(live)
        assert not live[0].skeleton_3d and recording.frames[0]["skeleton_3d"] is not None
        assert recording.header["exercise_name"] == "Barbell Overhead Press"
        assert [event["event"] for event in recording.events] == [
            "rest_start", "presence_only", "presence_only", "reset_readiness_gate", "reset_rep_counter",
        ]
