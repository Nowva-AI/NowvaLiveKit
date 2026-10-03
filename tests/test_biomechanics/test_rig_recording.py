"""Rig recordings and their replay (docs/deadlift/PLAN.md §8.2): a replay hands the
pipeline the synced sets the live capture assembled, with their timestamps and dropouts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from biomechanics.triangulation.multi_capture import SyncedFrames
from biomechanics.triangulation.rig_recording import (
    FRAMES_LOG,
    METADATA_FILE,
    RecordedCapture,
    RigRecordingWriter,
    camera_video_path,
)

FRAME_SHAPE = (48, 64, 3)
FPS = 30.0
# Motion JPEG is lossy; a flat frame decodes within a few grey levels.
GREY_TOLERANCE = 3.0
CAMERAS = ["0", "1", "2"]
REPO_ROOT = Path(__file__).resolve().parents[2]


def _frame(grey: int) -> np.ndarray:
    return np.full(FRAME_SHAPE, grey, dtype=np.uint8)


def _record(directory: Path, sets: list[SyncedFrames]) -> None:
    writer = RigRecordingWriter(directory, CAMERAS, FPS)
    for synced in sets:
        writer.write(synced)
    writer.write_metadata({"lift": "deadlift"})
    writer.close()


def _sets() -> list[SyncedFrames]:
    """Three synced sets; camera 2 drops out of the second."""
    return [
        SyncedFrames(frames={"0": _frame(10), "1": _frame(20), "2": _frame(30)}, sequence=4, timestamp=100.0),
        SyncedFrames(frames={"0": _frame(40), "1": _frame(50)}, sequence=5, timestamp=100.033),
        SyncedFrames(frames={"0": _frame(70), "1": _frame(80), "2": _frame(90)}, sequence=6, timestamp=100.067),
    ]


class TestRigRecording:
    def test_replay_returns_the_recorded_sets_in_order(self, tmp_path: Path):
        _record(tmp_path, _sets())
        capture = RecordedCapture(tmp_path)
        capture.start()
        replayed = [capture.get_synced_frames() for _ in range(3)]
        assert [synced.sequence for synced in replayed] == [4, 5, 6]
        assert [synced.timestamp for synced in replayed] == pytest.approx([100.0, 100.033, 100.067], abs=1e-9)
        assert sorted(replayed[1].frames) == ["0", "1"]
        assert float(replayed[2].frames["2"].mean()) == pytest.approx(90.0, abs=GREY_TOLERANCE)
        assert capture.get_synced_frames() is None
        assert capture.finished
        capture.release()

    def test_a_dropout_never_shifts_the_cameras_frames(self, tmp_path: Path):
        _record(tmp_path, _sets())
        capture = RecordedCapture(tmp_path)
        capture.start()
        capture.get_synced_frames()
        capture.get_synced_frames()
        third = capture.get_synced_frames()
        assert float(third.frames["2"].mean()) == pytest.approx(90.0, abs=GREY_TOLERANCE)
        assert capture.actual_resolutions["2"] == (FRAME_SHAPE[1], FRAME_SHAPE[0])

    def test_the_recording_holds_videos_log_and_metadata(self, tmp_path: Path):
        _record(tmp_path, _sets())
        assert all(camera_video_path(tmp_path, camera).exists() for camera in CAMERAS)
        lines = (tmp_path / FRAMES_LOG).read_text().splitlines()
        assert json.loads(lines[1]) == {"sequence": 5, "timestamp": 100.033, "cameras": ["0", "1"]}
        assert json.loads((tmp_path / METADATA_FILE).read_text()) == {"lift": "deadlift"}

    def test_the_provider_replays_when_given_a_recording(self, tmp_path: Path):
        from biomechanics.pose.multi_camera import MultiCameraPoseProvider

        _record(tmp_path, _sets())
        provider = MultiCameraPoseProvider(device_ids=[0, 1, 2], replay_dir=str(tmp_path))
        assert isinstance(provider._open_capture(), RecordedCapture)

    def test_the_tool_copies_the_rigs_calibration_but_writes_nothing_home(self, tmp_path: Path):
        spec = importlib.util.spec_from_file_location("record_rig", REPO_ROOT / "scripts" / "tools" / "record_rig.py")
        record_rig = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(record_rig)
        home = tmp_path / "nowva"
        home.mkdir()
        for name in ("rig_calibration_cams_0_1_2.json", "intrinsics_0.json", "gravity_0.json", "notes.txt"):
            (home / name).write_text("{}")
        copied = record_rig.copy_calibration(home, tmp_path / "recording" / "calibration")
        assert copied == ["rig_calibration_cams_0_1_2.json", "intrinsics_0.json", "gravity_0.json"]
        assert sorted(path.name for path in home.iterdir()) == sorted(
            ["rig_calibration_cams_0_1_2.json", "intrinsics_0.json", "gravity_0.json", "notes.txt"],
        )
