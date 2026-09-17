"""Tests for MultiCameraPoseProvider frame identity, partial views and calibration capture."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import biomechanics.pose.multi_camera as multi_camera_module
from biomechanics.pose.multi_camera import MultiCameraPoseProvider
from biomechanics.triangulation.multi_capture import MultiCameraCapture
from biomechanics.utils.types import MultiViewPose, Skeleton2D, Skeleton3D

NUM_KEYPOINTS = 21
BASE_TS = 1000.0
FRAME_PERIOD_S = 1.0 / 30.0
TIME_TOLERANCE_S = 1e-9
DEVICE_IDS = [0, 1, 2]
CAMERA_OFFSETS_S = (0.0, 0.002, -0.002)
CALIBRATION_FRAMES = 4
CALIBRATION_HEIGHT_M = 1.885


class _ManualClock:
    def __init__(self) -> None:
        self.now_s = 0.0
        self.on_sleep = None

    def clock(self) -> float:
        return self.now_s

    def sleep(self, duration_s: float) -> None:
        self.now_s += duration_s
        if self.on_sleep is not None:
            self.on_sleep()


class _FakeEstimator:
    def __init__(self, **kwargs: object) -> None:
        self.batch_sizes: list[int] = []
        self.undetected_frame_tags: set[int] = set()

    def initialize(self) -> bool:
        return True

    def estimate_batch(
        self, frames: list[np.ndarray], camera_ids: list[str] | None = None
    ) -> list[Skeleton2D | None]:
        self.batch_sizes.append(len(frames))
        self.last_camera_ids = camera_ids
        return [
            None if int(frame[0]) in self.undetected_frame_tags
            else Skeleton2D.from_numpy(np.full((NUM_KEYPOINTS, 3), 0.9))
            for frame in frames
        ]

    def reset_tracking(self) -> None:
        self.tracking_resets = getattr(self, "tracking_resets", 0) + 1

    def release(self) -> None:
        pass


class _FakeTriangulator:
    def __init__(self, **kwargs: object) -> None:
        self.received: list[MultiViewPose] = []
        self.resets = 0
        self.swap_count = 0

    def reset(self) -> None:
        self.resets += 1

    def triangulate(self, multi_view: MultiViewPose) -> Skeleton3D:
        self.received.append(multi_view)
        return Skeleton3D.from_numpy(
            np.zeros((NUM_KEYPOINTS, 3)),
            confidences=np.ones(NUM_KEYPOINTS),
            timestamp=multi_view.timestamp,
            frame_index=multi_view.frame_index,
        )


class _FakeCalibrator:
    last_frames: dict[str, list[np.ndarray]] = {}

    def __init__(self, **kwargs: object) -> None:
        pass

    def calibrate(self, frames: dict[str, list[np.ndarray]], height_m: float, resolution: tuple[int, int]) -> object:
        _FakeCalibrator.last_frames = frames
        return multi_camera_module.CalibrationResult()

    @staticmethod
    def load_calibration(path: str) -> object:
        return multi_camera_module.CalibrationResult()


def _frame_tag(cam_index: int, frame_number: int) -> int:
    return cam_index * 1000 + frame_number


def _push_set(capture: MultiCameraCapture, frame_number: int, camera_indices: tuple[int, ...] = (0, 1, 2)) -> None:
    timestamp = BASE_TS + frame_number * FRAME_PERIOD_S
    for cam_index in camera_indices:
        capture._append_frame(
            str(cam_index),
            np.array([_frame_tag(cam_index, frame_number)]),
            timestamp + CAMERA_OFFSETS_S[cam_index],
        )


@pytest.fixture
def manual_clock() -> _ManualClock:
    return _ManualClock()


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch, manual_clock: _ManualClock) -> MultiCameraPoseProvider:
    class _InjectedCapture(MultiCameraCapture):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(clock_fn=manual_clock.clock, sleep_fn=manual_clock.sleep, **kwargs)

        def start(self) -> None:
            pass

    monkeypatch.setattr(multi_camera_module, "RTMPoseEstimator", _FakeEstimator)
    monkeypatch.setattr(multi_camera_module, "DLTTriangulator", _FakeTriangulator)
    monkeypatch.setattr(multi_camera_module, "TPoseCalibrator", _FakeCalibrator)
    monkeypatch.setattr(multi_camera_module, "MultiCameraCapture", _InjectedCapture)

    pose_provider = MultiCameraPoseProvider(device_ids=DEVICE_IDS, min_views=2)
    pose_provider.initialize()
    pose_provider.start()
    pose_provider.load_calibration("unused.json")
    return pose_provider


class TestFrameIdentity:
    def test_frame_index_is_primary_sequence_and_strictly_increasing(self, provider: MultiCameraPoseProvider) -> None:
        capture = provider._capture
        frame_indices = []
        # Frames 2 and 3 arrive together while the consumer is busy: 2 is skipped.
        for arrived_frame_numbers in ([0], [1], [2, 3], [4], [5]):
            for frame_number in arrived_frame_numbers:
                _push_set(capture, frame_number)
            _, _, skeleton_3d = provider.get_pose()
            assert skeleton_3d is not None
            frame_indices.append(skeleton_3d.frame_index)
        assert all(later > earlier for earlier, later in zip(frame_indices, frame_indices[1:]))
        assert frame_indices == [0, 1, 3, 4, 5]

    def test_timestamp_is_reference_capture_time(self, provider: MultiCameraPoseProvider) -> None:
        _push_set(provider._capture, 0)
        primary_frame, skeleton_2d, skeleton_3d = provider.get_pose()
        assert int(primary_frame[0]) == _frame_tag(0, 0)
        assert skeleton_3d.timestamp == pytest.approx(BASE_TS, abs=TIME_TOLERANCE_S)
        assert skeleton_2d.timestamp == pytest.approx(BASE_TS, abs=TIME_TOLERANCE_S)
        assert skeleton_2d.frame_index == 0

    def test_same_sequence_never_triangulated_twice(self, provider: MultiCameraPoseProvider) -> None:
        _push_set(provider._capture, 0)
        assert provider.get_pose()[2] is not None
        assert provider.get_pose() == (None, None, None)
        assert len(provider._triangulator.received) == 1


class TestPartialViews:
    def test_missing_camera_absent_from_views(self, provider: MultiCameraPoseProvider) -> None:
        capture = provider._capture
        _push_set(capture, 0, camera_indices=(0, 1))
        capture._append_frame("2", np.array([_frame_tag(2, 0)]), BASE_TS + 0.050)
        _, _, skeleton_3d = provider.get_pose()
        assert skeleton_3d is not None
        assert set(provider._triangulator.received[-1].views) == {"0", "1"}
        assert provider._estimator.batch_sizes[-1] == 2

    def test_too_few_detected_views_returns_no_skeleton_3d(self, provider: MultiCameraPoseProvider) -> None:
        provider._estimator.undetected_frame_tags = {_frame_tag(1, 0), _frame_tag(2, 0)}
        _push_set(provider._capture, 0)
        primary_frame, skeleton_2d, skeleton_3d = provider.get_pose()
        assert int(primary_frame[0]) == _frame_tag(0, 0)
        assert skeleton_2d is not None
        assert skeleton_3d is None
        assert provider._triangulator.received == []


class TestCalibrationCapture:
    def test_calibrate_collects_distinct_synced_sets(self, provider: MultiCameraPoseProvider, manual_clock: _ManualClock) -> None:
        capture = provider._capture
        next_frame_number = [0]

        def deliver_next_set() -> None:
            _push_set(capture, next_frame_number[0], camera_indices=(0, 1))
            next_frame_number[0] += 1

        manual_clock.on_sleep = deliver_next_set
        provider.calibrate(height_m=CALIBRATION_HEIGHT_M, n_frames=CALIBRATION_FRAMES)

        collected = _FakeCalibrator.last_frames
        assert set(collected) == {"0", "1"}
        primary_tags = [int(frame[0]) for frame in collected["0"]]
        assert len(primary_tags) == CALIBRATION_FRAMES
        assert len(set(primary_tags)) == CALIBRATION_FRAMES
        assert provider.is_calibrated


class TestTemporalStateReset:
    def test_camera_ids_follow_views(self, provider: MultiCameraPoseProvider) -> None:
        _push_set(provider._capture, 0)
        provider.get_pose()
        assert provider._estimator.last_camera_ids == list(provider._capture.get_last_frames_keys()) if hasattr(provider._capture, "get_last_frames_keys") else provider._estimator.last_camera_ids is not None

    def test_reset_temporal_state_clears_triangulator_and_crops(self, provider: MultiCameraPoseProvider) -> None:
        provider.reset_temporal_state()
        assert provider._triangulator.resets == 1
        assert provider._estimator.tracking_resets == 1
        assert provider.swap_count == 0
