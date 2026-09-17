"""
Tests for MultiCameraPoseProvider frame identity, partial views, calibration capture, the
calibration view buffer and bar-end window, 2D-only operation, calibration install and intrinsics lookup.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import biomechanics.pose.multi_camera as multi_camera_module
import biomechanics.triangulation.calibration as calibration_module
import biomechanics.triangulation.multi_capture as multi_capture_module
from biomechanics.pose.multi_camera import MultiCameraPoseProvider
from biomechanics.triangulation.calibration import save_intrinsics
from biomechanics.triangulation.multi_capture import MultiCameraCapture
from biomechanics.utils.types import BarbellDetection, Keypoint2D, MultiViewPose, Skeleton2D, Skeleton3D

NUM_KEYPOINTS = 21
BASE_TS = 1000.0
FRAME_PERIOD_S = 1.0 / 30.0
TIME_TOLERANCE_S = 1e-9
DEVICE_IDS = [0, 1, 2]
CAMERA_OFFSETS_S = (0.0, 0.002, -0.002)
CALIBRATION_FRAMES = 4
CALIBRATION_HEIGHT_M = 1.885
BUFFER_FRAMES = 5
BAR_DETECTION_STRIDE = 3
REQUESTED_RESOLUTION = (1280, 720)
OPENED_RESOLUTION = (1920, 1080)
FOCAL_LENGTH_FACTOR = 0.8
CALIBRATED_FOCAL_PX = 1111.0
CALIBRATED_DISTORTION = np.array([-0.1, 0.02, 0.0, 0.0, 0.0])
INTRINSICS_TOLERANCE = 1e-9
FAKE_READ_PERIOD_S = 0.001


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


class _FakeBarDetector:
    def __init__(self) -> None:
        self.detected_frame_tags: list[int] = []

    def detect(self, frame: np.ndarray, timestamp: float = 0.0, frame_index: int = 0) -> BarbellDetection:
        self.detected_frame_tags.append(int(frame[0]))
        return BarbellDetection(
            left_end=Keypoint2D(x=100.0, y=200.0, confidence=0.9),
            right_end=Keypoint2D(x=500.0, y=200.0, confidence=0.9),
            timestamp=timestamp,
            frame_index=frame_index,
        )


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
def bar_detector() -> _FakeBarDetector:
    return _FakeBarDetector()


@pytest.fixture
def uncalibrated_provider(
    monkeypatch: pytest.MonkeyPatch, manual_clock: _ManualClock, bar_detector: _FakeBarDetector
) -> MultiCameraPoseProvider:
    class _InjectedCapture(MultiCameraCapture):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(clock_fn=manual_clock.clock, sleep_fn=manual_clock.sleep, **kwargs)

        def start(self) -> None:
            pass

    monkeypatch.setattr(multi_camera_module, "RTMPoseEstimator", _FakeEstimator)
    monkeypatch.setattr(multi_camera_module, "DLTTriangulator", _FakeTriangulator)
    monkeypatch.setattr(multi_camera_module, "TPoseCalibrator", _FakeCalibrator)
    monkeypatch.setattr(multi_camera_module, "MultiCameraCapture", _InjectedCapture)

    pose_provider = MultiCameraPoseProvider(
        device_ids=DEVICE_IDS,
        min_views=2,
        resolution=REQUESTED_RESOLUTION,
        focal_length_factor=FOCAL_LENGTH_FACTOR,
        camera_keys={1: "usb_left"},
        calibration_buffer_frames=BUFFER_FRAMES,
        bar_detection_stride=BAR_DETECTION_STRIDE,
        bar_detector=bar_detector,
    )
    pose_provider.initialize()
    pose_provider.start()
    return pose_provider


@pytest.fixture
def provider(uncalibrated_provider: MultiCameraPoseProvider) -> MultiCameraPoseProvider:
    uncalibrated_provider.load_calibration("unused.json")
    return uncalibrated_provider


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


def _pump(provider: MultiCameraPoseProvider, frame_numbers: range, camera_indices: tuple[int, ...] = (0, 1, 2)) -> None:
    for frame_number in frame_numbers:
        _push_set(provider._capture, frame_number, camera_indices=camera_indices)
        provider.get_pose()


class TestCalibrationViewBuffer:
    def test_buffer_holds_only_frames_seen_by_two_cameras(self, provider: MultiCameraPoseProvider) -> None:
        _pump(provider, range(0, 1))
        provider._estimator.undetected_frame_tags = {_frame_tag(1, 1), _frame_tag(2, 1)}
        _pump(provider, range(1, 2))
        provider._estimator.undetected_frame_tags = {_frame_tag(2, 2)}
        _pump(provider, range(2, 3))

        views = provider.calibration_views()

        assert [set(frame_views) for frame_views in views] == [{"0", "1", "2"}, {"0", "1"}]
        assert [frame_views["0"].frame_index for frame_views in views] == [0, 2]
        assert provider.calibration_frame_count == 2

    def test_buffer_respects_the_size_cap(self, provider: MultiCameraPoseProvider) -> None:
        _pump(provider, range(0, BUFFER_FRAMES + 3))

        views = provider.calibration_views()

        assert len(views) == BUFFER_FRAMES
        assert [frame_views["0"].frame_index for frame_views in views] == list(range(3, BUFFER_FRAMES + 3))
        assert len(provider.calibration_bar_ends()) == BUFFER_FRAMES

    def test_recent_views_are_the_newest_frames(self, provider: MultiCameraPoseProvider) -> None:
        _pump(provider, range(0, 4))

        recent = provider.recent_views(2)

        assert [frame_views["0"].frame_index for frame_views in recent] == [2, 3]
        assert len(provider.recent_views(100)) == 4

    def test_views_are_raw_pixel_skeletons_per_camera(self, provider: MultiCameraPoseProvider) -> None:
        _pump(provider, range(0, 1))

        frame_views = provider.calibration_views()[0]

        assert all(isinstance(skeleton, Skeleton2D) for skeleton in frame_views.values())
        assert all(len(skeleton.keypoints) == NUM_KEYPOINTS for skeleton in frame_views.values())

    def test_opening_a_capture_window_restarts_the_buffer(self, provider: MultiCameraPoseProvider) -> None:
        _pump(provider, range(0, 3))

        provider.begin_calibration_capture()
        _pump(provider, range(3, 5))

        assert [frame_views["0"].frame_index for frame_views in provider.calibration_views()] == [3, 4]


class TestBarDetectionWindow:
    def test_no_bar_detection_outside_a_capture_window(
        self, provider: MultiCameraPoseProvider, bar_detector: _FakeBarDetector
    ) -> None:
        _pump(provider, range(0, 4))

        assert bar_detector.detected_frame_tags == []
        assert provider.calibration_bar_ends() == [{}, {}, {}, {}]

    def test_bar_detection_runs_at_the_stride_on_every_camera(
        self, provider: MultiCameraPoseProvider, bar_detector: _FakeBarDetector
    ) -> None:
        provider.begin_calibration_capture()
        _pump(provider, range(0, 5))

        bar_ends = provider.calibration_bar_ends()

        assert [set(frame_bar_ends) for frame_bar_ends in bar_ends] == [
            {"0", "1", "2"}, set(), set(), {"0", "1", "2"}, set(),
        ]
        assert sorted(bar_detector.detected_frame_tags) == sorted(
            _frame_tag(cam_index, frame_number) for cam_index in (0, 1, 2) for frame_number in (0, 3)
        )
        assert len(bar_ends) == len(provider.calibration_views())

    def test_bar_detection_stops_when_the_window_closes(
        self, provider: MultiCameraPoseProvider, bar_detector: _FakeBarDetector
    ) -> None:
        provider.begin_calibration_capture()
        _pump(provider, range(0, 1))
        provider.end_calibration_capture()
        calls_in_window = len(bar_detector.detected_frame_tags)

        _pump(provider, range(1, 5))

        assert len(bar_detector.detected_frame_tags) == calls_in_window
        assert len(provider.calibration_views()) == 5

    def test_bar_detection_skips_cameras_without_a_skeleton(
        self, provider: MultiCameraPoseProvider, bar_detector: _FakeBarDetector
    ) -> None:
        provider._estimator.undetected_frame_tags = {_frame_tag(2, 0)}
        provider.begin_calibration_capture()
        _pump(provider, range(0, 1))

        assert set(provider.calibration_bar_ends()[0]) == {"0", "1"}

    def test_window_without_a_detector_buffers_views_only(
        self, provider: MultiCameraPoseProvider
    ) -> None:
        provider._bar_detector = None
        provider.begin_calibration_capture()
        _pump(provider, range(0, 2))

        assert not provider.has_bar_detector
        assert provider.calibration_bar_ends() == [{}, {}]


class TestUncalibratedOperation:
    def test_get_pose_is_2d_only_before_a_calibration_exists(
        self, uncalibrated_provider: MultiCameraPoseProvider
    ) -> None:
        _push_set(uncalibrated_provider._capture, 0)

        primary_frame, skeleton_2d, skeleton_3d = uncalibrated_provider.get_pose()

        assert not uncalibrated_provider.is_calibrated
        assert int(primary_frame[0]) == _frame_tag(0, 0)
        assert skeleton_2d is not None
        assert skeleton_2d.frame_index == 0
        assert skeleton_3d is None

    def test_buffer_fills_before_a_calibration_exists(
        self, uncalibrated_provider: MultiCameraPoseProvider
    ) -> None:
        _pump(uncalibrated_provider, range(0, 3))

        assert uncalibrated_provider.calibration_frame_count == 3


class TestInstallCalibration:
    def test_install_swaps_the_triangulator_and_resets_state(
        self, provider: MultiCameraPoseProvider
    ) -> None:
        old_triangulator = provider._triangulator
        new_calibration = multi_camera_module.CalibrationResult(timestamp="refined")

        provider.install_calibration(new_calibration)

        assert provider.calibration is new_calibration
        assert provider._triangulator is not old_triangulator
        assert provider._triangulator.resets == 1
        assert provider._estimator.tracking_resets == 1

    def test_install_on_an_uncalibrated_provider_starts_triangulation(
        self, uncalibrated_provider: MultiCameraPoseProvider
    ) -> None:
        _pump(uncalibrated_provider, range(0, 1))

        uncalibrated_provider.install_calibration(multi_camera_module.CalibrationResult())
        _push_set(uncalibrated_provider._capture, 1)
        _, _, skeleton_3d = uncalibrated_provider.get_pose()

        assert uncalibrated_provider.is_calibrated
        assert skeleton_3d is not None
        assert skeleton_3d.frame_index == 1


def _calibrated_intrinsic_matrix(resolution: tuple[int, int]) -> np.ndarray:
    return np.array([
        [CALIBRATED_FOCAL_PX, 0.0, resolution[0] / 2.0],
        [0.0, CALIBRATED_FOCAL_PX, resolution[1] / 2.0],
        [0.0, 0.0, 1.0],
    ])


class TestIntrinsicsLookup:
    @pytest.fixture
    def intrinsics_provider(
        self, uncalibrated_provider: MultiCameraPoseProvider, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> MultiCameraPoseProvider:
        # Camera 0 refused the requested mode and opened larger; its intrinsics were calibrated there.
        monkeypatch.setattr(calibration_module, "NOWVA_CALIBRATION_DIR", tmp_path)
        uncalibrated_provider._capture._actual_resolutions = {
            "0": OPENED_RESOLUTION, "1": REQUESTED_RESOLUTION, "2": REQUESTED_RESOLUTION,
        }
        save_intrinsics("0", OPENED_RESOLUTION, _calibrated_intrinsic_matrix(OPENED_RESOLUTION), CALIBRATED_DISTORTION, 0.1)
        save_intrinsics(
            "usb_left", REQUESTED_RESOLUTION, _calibrated_intrinsic_matrix(REQUESTED_RESOLUTION), CALIBRATED_DISTORTION, 0.1,
        )
        return uncalibrated_provider

    def test_intrinsics_are_looked_up_at_the_actual_resolution(
        self, intrinsics_provider: MultiCameraPoseProvider
    ) -> None:
        intrinsic_matrix, distortion = intrinsics_provider.rig_intrinsics()["0"]

        assert intrinsic_matrix == pytest.approx(_calibrated_intrinsic_matrix(OPENED_RESOLUTION), abs=INTRINSICS_TOLERANCE)
        assert distortion == pytest.approx(CALIBRATED_DISTORTION, abs=INTRINSICS_TOLERANCE)
        assert intrinsics_provider.capture_resolution == OPENED_RESOLUTION

    def test_camera_keys_name_the_intrinsics_file(self, intrinsics_provider: MultiCameraPoseProvider) -> None:
        intrinsic_matrix, _ = intrinsics_provider.rig_intrinsics()["1"]

        assert intrinsic_matrix[0, 0] == pytest.approx(CALIBRATED_FOCAL_PX, abs=INTRINSICS_TOLERANCE)

    def test_missing_intrinsics_fall_back_to_the_guessed_pinhole(
        self, intrinsics_provider: MultiCameraPoseProvider
    ) -> None:
        intrinsic_matrix, distortion = intrinsics_provider.rig_intrinsics()["2"]

        assert intrinsic_matrix[0, 0] == pytest.approx(
            FOCAL_LENGTH_FACTOR * REQUESTED_RESOLUTION[0], abs=INTRINSICS_TOLERANCE
        )
        assert intrinsic_matrix[:2, 2] == pytest.approx(
            [REQUESTED_RESOLUTION[0] / 2.0, REQUESTED_RESOLUTION[1] / 2.0], abs=INTRINSICS_TOLERANCE
        )
        assert not np.any(distortion)

    def test_missing_intrinsics_warn_once_with_the_command_to_run(
        self, intrinsics_provider: MultiCameraPoseProvider, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=multi_camera_module.__name__):
            intrinsics_provider.rig_intrinsics()
            intrinsics_provider.rig_intrinsics()

        warnings = [record.getMessage() for record in caplog.records if record.name == multi_camera_module.__name__]
        assert len(warnings) == 1
        assert "scripts/tools/calibrate_cameras.py intrinsics --camera 2" in warnings[0]
        assert "--camera 0" not in warnings[0]
        assert "--camera 1" not in warnings[0]

    def test_tpose_calibration_receives_the_actual_resolution_intrinsics(
        self, intrinsics_provider: MultiCameraPoseProvider, manual_clock: _ManualClock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        received: dict[str, object] = {}

        class _RecordingCalibrator(_FakeCalibrator):
            def __init__(self, **kwargs: object) -> None:
                received.update(kwargs)

            def calibrate(self, frames: dict, height_m: float, resolution: tuple[int, int]) -> object:
                received["resolution"] = resolution
                return multi_camera_module.CalibrationResult()

        monkeypatch.setattr(multi_camera_module, "TPoseCalibrator", _RecordingCalibrator)
        next_frame_number = [0]

        def deliver_next_set() -> None:
            _push_set(intrinsics_provider._capture, next_frame_number[0])
            next_frame_number[0] += 1

        manual_clock.on_sleep = deliver_next_set
        intrinsics_provider.calibrate(height_m=CALIBRATION_HEIGHT_M, n_frames=CALIBRATION_FRAMES)

        assert set(received["intrinsics"]) == {"0", "1", "2"}
        assert received["intrinsics"]["0"][0][0, 0] == pytest.approx(CALIBRATED_FOCAL_PX, abs=INTRINSICS_TOLERANCE)
        assert received["resolution"] == OPENED_RESOLUTION


class _FakeVideoCapture:
    """Opens at OPENED_RESOLUTION whatever was requested and never delivers a frame."""

    def __init__(self, device_id: int) -> None:
        self.device_id = device_id

    def set(self, prop: int, value: float) -> bool:
        return False

    def get(self, prop: int) -> float:
        if prop == multi_capture_module.cv2.CAP_PROP_FRAME_WIDTH:
            return float(OPENED_RESOLUTION[0])
        if prop == multi_capture_module.cv2.CAP_PROP_FRAME_HEIGHT:
            return float(OPENED_RESOLUTION[1])
        return 0.0

    def isOpened(self) -> bool:
        return True

    def read(self) -> tuple[bool, None]:
        time.sleep(FAKE_READ_PERIOD_S)
        return False, None

    def release(self) -> None:
        pass


class TestActualResolution:
    def test_capture_reports_the_resolution_each_camera_opened_at(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(multi_capture_module.cv2, "VideoCapture", _FakeVideoCapture)
        capture = MultiCameraCapture(device_ids=[0, 1], resolution=REQUESTED_RESOLUTION)

        assert capture.actual_resolutions == {}
        capture.start()
        try:
            assert capture.actual_resolutions == {"0": OPENED_RESOLUTION, "1": OPENED_RESOLUTION}
        finally:
            capture.release()
