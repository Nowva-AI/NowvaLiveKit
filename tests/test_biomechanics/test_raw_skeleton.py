"""
Raw skeleton recorder and replay (SHARED-PLUMBING.md §8): a live session's raw
frames and control events, replayed through a fresh pipeline, reproduce its
latch, reps and faults exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from biomechanics.config import BiomechanicsConfig
from biomechanics.replay.raw_skeleton import load_recording, main, replay_recording
from conftest import SYNTHETIC_FPS, SyntheticSession, squat_depth_profile

READINESS_WARM_UP_FRAMES = 10
REST_FRAMES = 20
VALGUS_SHIFT_M = 0.12
ATHLETE_PARAMS = {
    "shoulder_width_m": 0.38,
    "femur_avg_m": 0.45,
    "torso_avg_m": 0.52,
    "hip_width_m": 0.24,
    "tibia_avg_m": 0.43,
    "foot_avg_m": 0.18,
}


def _reps(count: int) -> list[float]:
    depths = [0.0] * READINESS_WARM_UP_FRAMES
    for _ in range(count):
        depths += squat_depth_profile()
    return depths + [0.0] * int(SYNTHETIC_FPS)


def _live_session(monkeypatch, path: Path, multi_camera: bool) -> list:
    """Two sets with a rest between them, recorded; returns the live frames."""
    session = SyntheticSession(monkeypatch, multi_camera=multi_camera)
    pipeline = session.pipeline
    pipeline.enable_raw_recording(path)
    pipeline.apply_athlete_params(ATHLETE_PARAMS)
    pipeline.set_depth_target(None)
    results = session.run(_reps(2), valgus_m=VALGUS_SHIFT_M)
    pipeline.record_event("rest_start", rest_seconds=60)
    pipeline.presence_only = True
    results += session.run([0.0] * REST_FRAMES)
    pipeline.presence_only = False
    pipeline.reset_readiness_gate()
    pipeline.reset_rep_counter()
    results += session.run(_reps(2))
    pipeline.release()
    return results


def _outcome(frames: list) -> dict:
    return {
        "ready": [frame.skeleton_3d is not None for frame in frames],
        "reps": [(frame.timestamp, frame.rep_data.rep_number) for frame in frames if frame.rep_data is not None],
        "faults": [
            (fault.timestamp, fault.fault_type, fault.severity.value, fault.rep_number)
            for frame in frames for fault in frame.faults
        ],
    }


class TestRawSkeletonReplay:
    @pytest.mark.parametrize("multi_camera", [True, False], ids=["rig", "mono"])
    def test_replay_reproduces_the_live_session(self, monkeypatch, tmp_path: Path, multi_camera: bool):
        path = tmp_path / "raw_skeleton.jsonl"
        live = _live_session(monkeypatch, path, multi_camera)

        replayed = replay_recording(path, BiomechanicsConfig())

        assert len(replayed.frames) == len(live)
        assert _outcome(live)["faults"]
        assert _outcome(replayed.frames) == _outcome(live)
        assert replayed.reps == [1, 2, 1, 2]

    def test_every_processed_frame_is_recorded_from_before_the_latch(self, monkeypatch, tmp_path: Path):
        path = tmp_path / "raw_skeleton.jsonl"
        live = _live_session(monkeypatch, path, multi_camera=True)

        recording = load_recording(path)

        assert len(recording.frames) == len(live)
        assert recording.frames[0]["skeleton_3d"] is not None
        assert not live[0].skeleton_3d
        events = [event["event"] for event in recording.events]
        assert events == [
            "apply_athlete_params", "set_depth_target", "rest_start", "presence_only",
            "presence_only", "reset_readiness_gate", "reset_rep_counter",
        ]
        recorded_reps = [output["rep_number"] for output in recording.outputs if output["rep_number"] is not None]
        assert recorded_reps == [1, 2, 1, 2]
        assert recording.header["capture_mode"] == "triangulated"
        assert recording.header["exercise_name"] == "Barbell Back Squat"

    def test_the_replay_entry_point_checks_the_rep_count(self, monkeypatch, tmp_path: Path, capsys):
        path = tmp_path / "raw_skeleton.jsonl"
        _live_session(monkeypatch, path, multi_camera=True)
        monkeypatch.setattr("biomechanics.replay.raw_skeleton.load_pipeline_config", BiomechanicsConfig)

        assert main([str(path)]) == 0
        assert "replayed reps: [1, 2, 1, 2]" in capsys.readouterr().out

    def test_a_config_change_is_reported(self, monkeypatch, tmp_path: Path, caplog):
        path = tmp_path / "raw_skeleton.jsonl"
        _live_session(monkeypatch, path, multi_camera=True)
        config = BiomechanicsConfig()
        config.faults.knee_valgus.mild += 1.0

        replay_recording(path, config)

        assert "Config differs" in caplog.text

    def test_an_unknown_recording_version_is_refused(self, tmp_path: Path):
        path = tmp_path / "raw_skeleton.jsonl"
        path.write_text(json.dumps({"kind": "header", "version": 99}) + "\n")
        with pytest.raises(ValueError):
            load_recording(path)
