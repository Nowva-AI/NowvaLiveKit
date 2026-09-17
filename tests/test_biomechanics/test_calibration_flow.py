"""
Tests for the multi-camera calibration session flow in pipeline_process: bootstrap from the
lifter's squats (capture window -> threaded person calibration -> save -> install), reload on
the next run, T-pose fallback, board re-anchoring into a "_refined" file, and the between-set
drift monitor. A real MultiCameraPoseProvider is fed by a manual-clock capture replaying the
synthetic 3-camera rig of test_person_calibration; the solver is stubbed except in one
end-to-end test.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import biomechanics.pipeline_process as pipeline_process  # noqa: E402
import biomechanics.pose.multi_camera as multi_camera_module  # noqa: E402
from biomechanics.config import BiomechanicsConfig  # noqa: E402
from biomechanics.pipeline_process import (  # noqa: E402
    CALIBRATION_CAPTURE_HEADLINE,
    CALIBRATION_SOLVING_HEADLINE,
    TPOSE_FALLBACK_HEADLINE,
    WORLD_ANCHOR_HEADLINE,
    CalibrationSolve,
    CameraCalibrationSession,
    SquatExcursionCounter,
    knee_flexion_2d_deg,
    refined_calibration_path,
)
from biomechanics.pose.multi_camera import MultiCameraPoseProvider  # noqa: E402
from biomechanics.triangulation.calibration import (  # noqa: E402
    WORLD_ANCHOR_BOARD,
    WORLD_ANCHOR_PERSON,
    CalibrationResult,
    TPoseCalibrator,
    rig_calibration_path,
)
from biomechanics.triangulation.multi_capture import MultiCameraCapture  # noqa: E402
from biomechanics.triangulation.person_calibration import PersonCalibrationResult  # noqa: E402
from biomechanics.utils.types import BarbellDetection, PipelineFrame, Skeleton2D  # noqa: E402
from biomechanics.utils.types import CocoKeypoints as CK  # noqa: E402
from test_person_calibration import (  # noqa: E402
    BAR_LENGTH_M,
    HEIGHT_M,
    LIFTER_RATIOS,
    RESOLUTION,
    _lifter_sequence,
    _observe,
    _perturbed,
    _rig_errors,
    _true_calibration,
)

DEVICE_IDS = [0, 1, 2]
BASE_TS = 1000.0
FRAME_PERIOD_S = 1.0 / 30.0
USER_HEIGHT_CM = HEIGHT_M * 100.0
DETECTION_NOISE_PX = 1.0
STORED_RMS_PX = 1.0
SOLVED_RMS_PX = 1.2
DRIFT_ROTATION_DEG = 2.0
DRIFT_SHIFT_M = 0.05
MIN_CALIBRATION_FRAMES = 60
CAPTURE_TIMEOUT_S = 4.0
LONG_CAPTURE_TIMEOUT_S = 60.0
BAR_DETECTION_STRIDE = 3
STANDING_FRAMES = 40
THREAD_WAIT_S = 10.0
THREAD_POLL_S = 0.001
FACTORY_MTIME_S = 1_000_000.0
LATER_S = 100.0

STANDING_MAX_FLEXION_DEG = 10.0
SQUAT_MIN_FLEXION_DEG = 80.0
NOISY_CENTRE_TOL_M = 0.05
FEMUR_TOL_M = 0.01
RMS_TOL_PX = 1e-9


class _ManualClock:
    def __init__(self) -> None:
        self.now_s = 0.0

    def clock(self) -> float:
        return self.now_s

    def sleep(self, duration_s: float) -> None:
        self.now_s += duration_s


class _ReplayEstimator:
    """Returns the scripted skeleton of the frame number carried in frame[0] for each camera."""

    def __init__(self, **kwargs: object) -> None:
        self.script: list[dict[str, Skeleton2D]] = []

    def initialize(self) -> bool:
        return True

    def estimate_batch(self, frames: list[np.ndarray], camera_ids: list[str] | None = None) -> list[Skeleton2D | None]:
        return [self.script[int(frame[0])].get(cam_id) for frame, cam_id in zip(frames, camera_ids)]

    def reset_tracking(self) -> None:
        pass

    def release(self) -> None:
        pass


class _ReplayBarDetector:
    """Returns the scripted bar ends of (frame number, camera) carried in the frame array."""

    def __init__(self, script: list[dict[str, BarbellDetection]]) -> None:
        self._script = script

    def detect(self, frame: np.ndarray, timestamp: float = 0.0, frame_index: int = 0) -> BarbellDetection | None:
        return self._script[int(frame[0])].get(str(int(frame[1])))


class _FakePipeline:
    """The slice of BiomechanicsPipeline the session drives; each process_frame delivers the next scripted set."""

    def __init__(self, provider: MultiCameraPoseProvider, clock: _ManualClock, script_length: int) -> None:
        self._multi_camera_provider = provider
        self._clock = clock
        self._script_length = script_length
        self._next_frame_number = 0
        self.presence_only = False
        self.last_frame: np.ndarray | None = None
        self.presence_only_per_frame: list[bool] = []
        self.calibration_changes: list[bool] = []

    @property
    def frames_processed(self) -> int:
        return len(self.presence_only_per_frame)

    def process_frame(self) -> PipelineFrame:
        self.presence_only_per_frame.append(self.presence_only)
        self._clock.now_s += FRAME_PERIOD_S
        if self._next_frame_number < self._script_length:
            timestamp = BASE_TS + self._next_frame_number * FRAME_PERIOD_S
            for device_id in DEVICE_IDS:
                self._multi_camera_provider._capture._append_frame(
                    str(device_id), np.array([self._next_frame_number, device_id]), timestamp,
                )
            self._next_frame_number += 1
        else:
            # Real cameras block until the next frame; without this a waiting loop starves the solver thread.
            time.sleep(THREAD_POLL_S)
        frame, skeleton_2d, _ = self._multi_camera_provider.get_pose()
        self.last_frame = frame
        return PipelineFrame(frame_index=self._next_frame_number, timestamp=self._clock.now_s, skeleton_2d=skeleton_2d)

    def on_calibration_changed(self, world_frame_changed: bool) -> None:
        self.calibration_changes.append(world_frame_changed)


class _Rig:
    """One session's worth of fakes around a real provider."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        factory_path: Path,
        views: list[dict[str, Skeleton2D]],
        bar_ends: list[dict[str, BarbellDetection]] | None = None,
        config: BiomechanicsConfig | None = None,
    ) -> None:
        clock = _ManualClock()

        class _InjectedCapture(MultiCameraCapture):
            def __init__(self, **kwargs: object) -> None:
                super().__init__(clock_fn=clock.clock, sleep_fn=clock.sleep, **kwargs)

            def start(self) -> None:
                pass

        monkeypatch.setattr(multi_camera_module, "RTMPoseEstimator", _ReplayEstimator)
        monkeypatch.setattr(multi_camera_module, "MultiCameraCapture", _InjectedCapture)
        # No intrinsics files: the provider falls back to the guessed pinhole, which is the synthetic rig's K.
        monkeypatch.setattr(multi_camera_module, "load_rig_intrinsics", lambda camera_keys, resolution: {})

        self.config = config or _flow_config()
        self.provider = MultiCameraPoseProvider(
            device_ids=DEVICE_IDS,
            resolution=RESOLUTION,
            focal_length_factor=0.75,
            bar_detection_stride=self.config.camera_calibration.bar_detection_stride,
            bar_detector=_ReplayBarDetector(bar_ends) if bar_ends is not None else None,
        )
        self.provider.initialize()
        self.provider.start()
        self.provider._estimator.script = views
        self.pipeline = _FakePipeline(self.provider, clock, len(views))
        self.status: list[tuple[str, str | None]] = []
        self.session = CameraCalibrationSession(
            self.pipeline,
            self.config,
            factory_path=factory_path,
            user_height_cm=USER_HEIGHT_CM,
            show_status_fn=lambda result, headline, detail: self.status.append((headline, detail)),
            clock_fn=clock.clock,
        )

    def run_frames(self, count: int) -> None:
        for _ in range(count):
            self.pipeline.process_frame()

    @property
    def headlines(self) -> list[str]:
        return [headline for headline, _ in self.status]


def _flow_config() -> BiomechanicsConfig:
    config = BiomechanicsConfig()
    config.camera_calibration.min_calibration_frames = MIN_CALIBRATION_FRAMES
    config.camera_calibration.capture_timeout_s = CAPTURE_TIMEOUT_S
    config.camera_calibration.bar_detection_stride = BAR_DETECTION_STRIDE
    config.barbell_tracking.bar_length_m = BAR_LENGTH_M
    return config


def _solved(calibration: CalibrationResult, rms_px: float = SOLVED_RMS_PX) -> PersonCalibrationResult:
    return PersonCalibrationResult(
        calibration=calibration, rms_reprojection_px=rms_px, scale_source="bar", frames_used=MIN_CALIBRATION_FRAMES,
        bone_lengths_m={}, initial_rms_reprojection_px=9.0, solve_time_s=0.1,
    )


def _save(calibration: CalibrationResult, path: Path, world_anchor: str, rms_px: float) -> bytes:
    calibration.world_anchor = world_anchor
    calibration.rms_reprojection_px = rms_px
    TPoseCalibrator.save_calibration(calibration, str(path))
    os.utime(path, (FACTORY_MTIME_S, FACTORY_MTIME_S))
    return path.read_bytes()


def _wait_for_refine(session: CameraCalibrationSession) -> None:
    deadline_s = time.monotonic() + THREAD_WAIT_S
    while not session._refine.done:
        assert time.monotonic() < deadline_s, "refine thread did not finish"
        time.sleep(THREAD_POLL_S)


@pytest.fixture(scope="module")
def observations() -> tuple[list[dict[str, Skeleton2D]], list[dict[str, BarbellDetection]]]:
    return _observe(_lifter_sequence(LIFTER_RATIOS), _true_calibration(), noise_px=DETECTION_NOISE_PX)


@pytest.fixture(scope="module")
def squat_views(observations: tuple) -> list[dict[str, Skeleton2D]]:
    return observations[0]


@pytest.fixture(scope="module")
def standing_views(squat_views: list[dict[str, Skeleton2D]]) -> list[dict[str, Skeleton2D]]:
    return [squat_views[0]] * STANDING_FRAMES


@pytest.fixture
def factory_path(tmp_path: Path) -> Path:
    return rig_calibration_path(DEVICE_IDS, tmp_path)


@pytest.fixture
def fake_calibrator(monkeypatch: pytest.MonkeyPatch) -> type:
    class _FakePersonCalibrator:
        instances: list["_FakePersonCalibrator"] = []
        outcome: PersonCalibrationResult | Exception = _solved(_true_calibration())
        release: threading.Event | None = None

        def __init__(
            self,
            intrinsics: dict[str, tuple[np.ndarray, np.ndarray]],
            resolution: tuple[int, int],
            bar_length_m: float | None = None,
            height_m: float | None = None,
        ) -> None:
            self.intrinsics = intrinsics
            self.bar_length_m = bar_length_m
            self.height_m = height_m
            self.calls: list[dict[str, object]] = []
            type(self).instances.append(self)

        def calibrate(
            self,
            views: list[dict[str, Skeleton2D]],
            bar_ends: list[dict[str, BarbellDetection]] | None = None,
            initial: CalibrationResult | None = None,
            keep_world_frame: bool = False,
        ) -> PersonCalibrationResult:
            self.calls.append({
                "views": views, "bar_ends": bar_ends, "initial": initial, "keep_world_frame": keep_world_frame,
                "on_main_thread": threading.current_thread() is threading.main_thread(),
            })
            if type(self).release is not None:
                type(self).release.wait(THREAD_WAIT_S)
            if isinstance(type(self).outcome, Exception):
                raise type(self).outcome
            return type(self).outcome

        def refine(
            self,
            calibration: CalibrationResult,
            views: list[dict[str, Skeleton2D]],
            bar_ends: list[dict[str, BarbellDetection]] | None = None,
            keep_world_frame: bool = False,
        ) -> PersonCalibrationResult:
            return self.calibrate(views, bar_ends, initial=calibration, keep_world_frame=keep_world_frame)

    monkeypatch.setattr(pipeline_process, "PersonCalibrator", _FakePersonCalibrator)
    return _FakePersonCalibrator


class TestSquatExcursionProxy:
    def test_standing_reads_small_and_squat_bottom_reads_large(self, squat_views: list[dict[str, Skeleton2D]]) -> None:
        flexions_deg = [knee_flexion_2d_deg(views) for views in squat_views]

        assert flexions_deg[0] < STANDING_MAX_FLEXION_DEG
        assert max(flexions_deg) > SQUAT_MIN_FLEXION_DEG

    def test_no_visible_leg_is_nan(self, squat_views: list[dict[str, Skeleton2D]]) -> None:
        keypoints = squat_views[0]["0"].to_numpy()
        keypoints[[CK.LEFT_KNEE, CK.RIGHT_KNEE], 2] = 0.0

        assert np.isnan(knee_flexion_2d_deg({"0": Skeleton2D.from_numpy(keypoints)}))
        assert np.isnan(knee_flexion_2d_deg({}))

    def test_two_reps_count_as_two_excursions(self, squat_views: list[dict[str, Skeleton2D]]) -> None:
        counter = SquatExcursionCounter()
        for views in squat_views:
            counter.update(knee_flexion_2d_deg(views))

        assert counter.count == 2

    def test_single_frame_spike_is_not_an_excursion(self) -> None:
        counter = SquatExcursionCounter()
        for flexion_deg in (5.0, 95.0, 5.0, float("nan"), 95.0, 95.0, 5.0):
            counter.update(flexion_deg)

        assert counter.count == 0

    def test_excursion_counts_once_back_up(self) -> None:
        counter = SquatExcursionCounter()
        for flexion_deg in (5.0, 70.0, 90.0, 100.0, 70.0, 40.0):
            counter.update(flexion_deg)
        assert counter.count == 0

        counter.update(10.0)
        assert counter.count == 1


class TestCalibrationSolve:
    def test_solve_runs_off_the_main_thread(self) -> None:
        solve = CalibrationSolve(lambda: threading.current_thread() is threading.main_thread())
        solve._thread.join(THREAD_WAIT_S)

        assert solve.done
        assert solve.result is False
        assert solve.error is None

    def test_failed_solve_reports_its_error(self) -> None:
        def fail() -> PersonCalibrationResult:
            raise ValueError("too few frames")

        solve = CalibrationSolve(fail)
        solve._thread.join(THREAD_WAIT_S)

        assert solve.done
        assert solve.result is None
        assert str(solve.error) == "too few frames"


class TestBootstrapFlow:
    def test_no_file_captures_calibrates_saves_and_installs(
        self, monkeypatch: pytest.MonkeyPatch, factory_path: Path, fake_calibrator: type, observations: tuple
    ) -> None:
        views, bar_ends = observations
        rig = _Rig(monkeypatch, factory_path, views, bar_ends)

        rig.session.establish()

        calibrator = fake_calibrator.instances[0]
        call = calibrator.calls[0]
        assert len(fake_calibrator.instances) == 1 and len(calibrator.calls) == 1
        assert call["initial"] is None
        assert not call["on_main_thread"]
        assert len(call["views"]) >= MIN_CALIBRATION_FRAMES
        assert len(call["bar_ends"]) == len(call["views"])
        assert sum(1 for frame_bar_ends in call["bar_ends"] if len(frame_bar_ends) >= 2) >= 10
        assert calibrator.bar_length_m == pytest.approx(BAR_LENGTH_M)
        assert calibrator.height_m == pytest.approx(HEIGHT_M)
        assert set(calibrator.intrinsics) == {"0", "1", "2"}

        assert rig.provider.calibration is fake_calibrator.outcome.calibration
        assert rig.pipeline.calibration_changes == [True]
        assert not rig.provider._capture_window_open
        saved = json.loads(factory_path.read_text())
        assert saved["rms_reprojection_px"] == pytest.approx(SOLVED_RMS_PX, abs=RMS_TOL_PX)
        assert saved["world_anchor"] == WORLD_ANCHOR_PERSON
        assert not refined_calibration_path(factory_path).exists()

    def test_hud_prompts_for_two_slow_squats_and_frames_are_presence_only(
        self, monkeypatch: pytest.MonkeyPatch, factory_path: Path, fake_calibrator: type, squat_views: list
    ) -> None:
        rig = _Rig(monkeypatch, factory_path, squat_views)

        rig.session.establish()

        assert CALIBRATION_CAPTURE_HEADLINE == "CALIBRATING CAMERAS - STEP IN AND DO TWO SLOW SQUATS"
        assert rig.headlines[0] == CALIBRATION_CAPTURE_HEADLINE
        assert set(rig.headlines) <= {CALIBRATION_CAPTURE_HEADLINE, CALIBRATION_SOLVING_HEADLINE}
        assert rig.status[0][1] == f"FRAMES 1/{MIN_CALIBRATION_FRAMES}  SQUATS 0/2"
        assert all(rig.pipeline.presence_only_per_frame)
        assert rig.pipeline.presence_only is False

    def test_capture_waits_for_the_squats_not_just_the_frames(
        self, monkeypatch: pytest.MonkeyPatch, factory_path: Path, fake_calibrator: type, squat_views: list
    ) -> None:
        standing_lead_in = [squat_views[0]] * (2 * MIN_CALIBRATION_FRAMES)
        config = _flow_config()
        config.camera_calibration.capture_timeout_s = LONG_CAPTURE_TIMEOUT_S
        rig = _Rig(monkeypatch, factory_path, standing_lead_in + squat_views, config=config)

        rig.session.establish()

        assert rig.pipeline.frames_processed > len(standing_lead_in)
        assert len(fake_calibrator.instances[0].calls) == 1

    def test_without_a_bar_detector_scale_comes_from_height(
        self, monkeypatch: pytest.MonkeyPatch, factory_path: Path, fake_calibrator: type, squat_views: list
    ) -> None:
        rig = _Rig(monkeypatch, factory_path, squat_views)

        rig.session.establish()

        calibrator = fake_calibrator.instances[0]
        assert calibrator.bar_length_m is None
        assert calibrator.height_m == pytest.approx(HEIGHT_M)
        assert all(frame_bar_ends == {} for frame_bar_ends in calibrator.calls[0]["bar_ends"])

    def test_second_run_loads_the_saved_file_without_capturing(
        self, monkeypatch: pytest.MonkeyPatch, factory_path: Path, fake_calibrator: type, squat_views: list
    ) -> None:
        _Rig(monkeypatch, factory_path, squat_views).session.establish()
        second_run = _Rig(monkeypatch, factory_path, squat_views)

        second_run.session.establish()

        assert len(fake_calibrator.instances) == 1
        assert second_run.pipeline.frames_processed == 0
        assert second_run.status == []
        assert second_run.provider.is_calibrated
        assert second_run.provider.calibration.rms_reprojection_px == pytest.approx(SOLVED_RMS_PX, abs=RMS_TOL_PX)
        assert not second_run.session.needs_world_anchor

    def test_timeout_without_squats_falls_back_to_tpose(
        self, monkeypatch: pytest.MonkeyPatch, factory_path: Path, fake_calibrator: type, standing_views: list
    ) -> None:
        rig = _Rig(monkeypatch, factory_path, standing_views)
        tpose_calls: list[dict[str, object]] = []

        def fake_tpose_calibrate(height_m: float, save_path: str | None = None) -> CalibrationResult:
            tpose_calls.append({"height_m": height_m, "save_path": save_path})
            return CalibrationResult(timestamp="tpose")

        monkeypatch.setattr(rig.provider, "calibrate", fake_tpose_calibrate)

        rig.session.establish()

        assert fake_calibrator.instances == []
        assert tpose_calls == [{"height_m": pytest.approx(HEIGHT_M), "save_path": str(factory_path)}]
        assert TPOSE_FALLBACK_HEADLINE in rig.headlines
        assert rig.status[-1][1] == "T-POSE IN 1"
        assert rig.pipeline.calibration_changes == [True]
        assert rig.pipeline.presence_only is False
        assert not rig.provider._capture_window_open

    def test_failed_person_calibration_falls_back_to_tpose(
        self, monkeypatch: pytest.MonkeyPatch, factory_path: Path, fake_calibrator: type, squat_views: list
    ) -> None:
        fake_calibrator.outcome = ValueError("camera 2 cannot be linked")
        rig = _Rig(monkeypatch, factory_path, squat_views)
        tpose_calls: list[str | None] = []

        def fake_tpose_calibrate(height_m: float, save_path: str | None = None) -> CalibrationResult:
            tpose_calls.append(save_path)
            return CalibrationResult(timestamp="tpose")

        monkeypatch.setattr(rig.provider, "calibrate", fake_tpose_calibrate)

        rig.session.establish()

        assert len(fake_calibrator.instances[0].calls) == 1
        assert tpose_calls == [str(factory_path)]
        assert TPOSE_FALLBACK_HEADLINE in rig.headlines

    def test_end_to_end_person_calibration_on_a_synthetic_rig(
        self, monkeypatch: pytest.MonkeyPatch, factory_path: Path, observations: tuple
    ) -> None:
        views, bar_ends = observations
        rig = _Rig(monkeypatch, factory_path, views, bar_ends)

        rig.session.establish()

        saved = TPoseCalibrator.load_calibration(str(factory_path))
        centre_error_m, _ = _rig_errors(saved, _true_calibration())
        assert centre_error_m < NOISY_CENTRE_TOL_M
        assert saved.world_anchor == WORLD_ANCHOR_PERSON
        assert 0.0 < saved.rms_reprojection_px < 3.0 * DETECTION_NOISE_PX
        assert rig.pipeline.calibration_changes == [True]

        # The installed calibration triangulates: a metric femur from the bar-scaled rig.
        for device_id in DEVICE_IDS:
            rig.provider._capture._append_frame(str(device_id), np.array([0, device_id]), BASE_TS + LATER_S)
        _, _, skeleton_3d = rig.provider.get_pose()
        points = skeleton_3d.to_numpy()
        femur_m = np.linalg.norm(points[CK.LEFT_HIP] - points[CK.LEFT_KNEE])
        assert femur_m == pytest.approx(LIFTER_RATIOS["femur"] * HEIGHT_M, abs=FEMUR_TOL_M)

        second_run = _Rig(monkeypatch, factory_path, views, bar_ends)
        second_run.session.establish()
        assert second_run.pipeline.frames_processed == 0
        assert second_run.provider.is_calibrated


class TestBoardAnchoredFlow:
    def test_board_calibration_is_refined_once_into_the_refined_file(
        self, monkeypatch: pytest.MonkeyPatch, factory_path: Path, fake_calibrator: type, squat_views: list
    ) -> None:
        factory_bytes = _save(_true_calibration(), factory_path, WORLD_ANCHOR_BOARD, STORED_RMS_PX)
        rig = _Rig(monkeypatch, factory_path, squat_views)
        rig.session.establish()
        assert rig.session.needs_world_anchor
        assert rig.pipeline.frames_processed == 0
        loaded = rig.provider.calibration

        rig.run_frames(len(squat_views))
        rig.session.anchor_world_on_lifter()

        calibrator = fake_calibrator.instances[0]
        assert len(fake_calibrator.instances) == 1 and len(calibrator.calls) == 1
        call = calibrator.calls[0]
        assert call["initial"] is loaded
        assert call["keep_world_frame"] is False
        assert not call["on_main_thread"]
        assert len(call["views"]) == len(squat_views)
        # A refine keeps the scale and intrinsics of the calibration it starts from.
        assert calibrator.bar_length_m is None and calibrator.height_m is None
        assert calibrator.intrinsics["1"][0] == pytest.approx(loaded.cameras["1"].intrinsic_matrix)

        refined_path = refined_calibration_path(factory_path)
        assert json.loads(refined_path.read_text())["world_anchor"] == WORLD_ANCHOR_PERSON
        assert factory_path.read_bytes() == factory_bytes
        assert rig.provider.calibration is fake_calibrator.outcome.calibration
        assert rig.pipeline.calibration_changes == [True]
        assert not rig.session.needs_world_anchor

    def test_next_run_prefers_the_refined_file(
        self, monkeypatch: pytest.MonkeyPatch, factory_path: Path, fake_calibrator: type, squat_views: list
    ) -> None:
        _save(_true_calibration(), factory_path, WORLD_ANCHOR_BOARD, STORED_RMS_PX)
        first_run = _Rig(monkeypatch, factory_path, squat_views)
        first_run.session.establish()
        first_run.run_frames(len(squat_views))
        first_run.session.anchor_world_on_lifter()

        second_run = _Rig(monkeypatch, factory_path, squat_views)
        second_run.session.establish()

        assert second_run.provider.calibration.world_anchor == WORLD_ANCHOR_PERSON
        assert not second_run.session.needs_world_anchor

    def test_presence_only_is_restored_after_anchoring(
        self, monkeypatch: pytest.MonkeyPatch, factory_path: Path, fake_calibrator: type, squat_views: list
    ) -> None:
        _save(_true_calibration(), factory_path, WORLD_ANCHOR_BOARD, STORED_RMS_PX)
        fake_calibrator.release = threading.Event()
        rig = _Rig(monkeypatch, factory_path, squat_views)
        rig.session.establish()
        rig.run_frames(len(squat_views))
        frames_before = rig.pipeline.frames_processed
        threading.Timer(0.05, fake_calibrator.release.set).start()

        rig.session.anchor_world_on_lifter()

        waited = rig.pipeline.presence_only_per_frame[frames_before:]
        assert waited and all(waited)
        assert rig.pipeline.presence_only is False
        assert set(rig.headlines) == {WORLD_ANCHOR_HEADLINE}

    def test_without_an_assessment_the_anchor_runs_at_the_first_rest(
        self, monkeypatch: pytest.MonkeyPatch, factory_path: Path, fake_calibrator: type, squat_views: list
    ) -> None:
        _save(_true_calibration(), factory_path, WORLD_ANCHOR_BOARD, STORED_RMS_PX)
        rig = _Rig(monkeypatch, factory_path, squat_views)
        rig.session.establish()
        rig.run_frames(len(squat_views))

        rig.session.on_rest_start()
        _wait_for_refine(rig.session)
        rig.session.poll(can_install=True)

        assert fake_calibrator.instances[0].calls[0]["keep_world_frame"] is False
        assert rig.pipeline.calibration_changes == [True]
        assert refined_calibration_path(factory_path).exists()
        assert not rig.session.needs_world_anchor


class TestDriftMonitor:
    def test_health_within_the_ratio_does_not_refine(
        self, monkeypatch: pytest.MonkeyPatch, factory_path: Path, fake_calibrator: type, squat_views: list
    ) -> None:
        _save(_true_calibration(), factory_path, WORLD_ANCHOR_PERSON, STORED_RMS_PX)
        rig = _Rig(monkeypatch, factory_path, squat_views)
        rig.session.establish()
        rig.run_frames(len(squat_views))

        rig.session.on_rest_start()

        assert not rig.session.is_refining
        assert fake_calibrator.instances == []
        assert rig.pipeline.calibration_changes == []

    def test_drift_refines_in_a_thread_and_installs_only_between_sets(
        self, monkeypatch: pytest.MonkeyPatch, factory_path: Path, fake_calibrator: type, squat_views: list
    ) -> None:
        drifted = _perturbed(_true_calibration(), DRIFT_ROTATION_DEG, DRIFT_SHIFT_M)
        factory_bytes = _save(drifted, factory_path, WORLD_ANCHOR_PERSON, STORED_RMS_PX)
        fake_calibrator.release = threading.Event()
        rig = _Rig(monkeypatch, factory_path, squat_views)
        rig.session.establish()
        loaded = rig.provider.calibration
        rig.run_frames(len(squat_views))

        rig.session.on_rest_start()
        assert rig.session.is_refining

        # Still solving: nothing to install even though we are resting.
        rig.session.poll(can_install=True)
        assert rig.provider.calibration is loaded

        fake_calibrator.release.set()
        _wait_for_refine(rig.session)

        # Solved, but the next set is already under way: never mid-set.
        rig.session.poll(can_install=False)
        assert rig.provider.calibration is loaded
        assert rig.pipeline.calibration_changes == []
        assert rig.session.is_refining

        rig.session.poll(can_install=True)

        call = fake_calibrator.instances[0].calls[0]
        assert call["initial"] is loaded
        assert call["keep_world_frame"] is True
        assert not call["on_main_thread"]
        assert rig.provider.calibration is fake_calibrator.outcome.calibration
        # The world frame was kept: temporal state resets, foot contact and floor survive.
        assert rig.pipeline.calibration_changes == [False]
        assert not rig.session.is_refining
        refined = json.loads(refined_calibration_path(factory_path).read_text())
        assert refined["rms_reprojection_px"] == pytest.approx(SOLVED_RMS_PX, abs=RMS_TOL_PX)
        assert factory_path.read_bytes() == factory_bytes

    def test_refine_finished_after_the_rest_installs_at_the_next_rest(
        self, monkeypatch: pytest.MonkeyPatch, factory_path: Path, fake_calibrator: type, squat_views: list
    ) -> None:
        drifted = _perturbed(_true_calibration(), DRIFT_ROTATION_DEG, DRIFT_SHIFT_M)
        _save(drifted, factory_path, WORLD_ANCHOR_PERSON, STORED_RMS_PX)
        rig = _Rig(monkeypatch, factory_path, squat_views)
        rig.session.establish()
        rig.run_frames(len(squat_views))
        rig.session.on_rest_start()
        _wait_for_refine(rig.session)
        rig.session.poll(can_install=False)
        assert rig.pipeline.calibration_changes == []

        rig.session.on_rest_start()

        assert rig.pipeline.calibration_changes == [False]
        assert rig.provider.calibration is fake_calibrator.outcome.calibration
        assert len(fake_calibrator.instances) == 1

    def test_refine_that_does_not_improve_health_is_discarded(
        self, monkeypatch: pytest.MonkeyPatch, factory_path: Path, fake_calibrator: type, squat_views: list
    ) -> None:
        drifted = _perturbed(_true_calibration(), DRIFT_ROTATION_DEG, DRIFT_SHIFT_M)
        _save(drifted, factory_path, WORLD_ANCHOR_PERSON, STORED_RMS_PX)
        fake_calibrator.outcome = _solved(_perturbed(_true_calibration(), DRIFT_ROTATION_DEG, DRIFT_SHIFT_M))
        rig = _Rig(monkeypatch, factory_path, squat_views)
        rig.session.establish()
        loaded = rig.provider.calibration
        rig.run_frames(len(squat_views))

        rig.session.on_rest_start()
        _wait_for_refine(rig.session)
        rig.session.poll(can_install=True)

        assert rig.provider.calibration is loaded
        assert rig.pipeline.calibration_changes == []
        assert not refined_calibration_path(factory_path).exists()

        # The observed level becomes the baseline, so the next rest does not try again.
        rig.session.on_rest_start()
        assert not rig.session.is_refining
        assert len(fake_calibrator.instances) == 1

    def test_failed_refine_keeps_the_current_calibration(
        self, monkeypatch: pytest.MonkeyPatch, factory_path: Path, fake_calibrator: type, squat_views: list
    ) -> None:
        drifted = _perturbed(_true_calibration(), DRIFT_ROTATION_DEG, DRIFT_SHIFT_M)
        _save(drifted, factory_path, WORLD_ANCHOR_PERSON, STORED_RMS_PX)
        fake_calibrator.outcome = ValueError("too few frames")
        rig = _Rig(monkeypatch, factory_path, squat_views)
        rig.session.establish()
        loaded = rig.provider.calibration
        rig.run_frames(len(squat_views))

        rig.session.on_rest_start()
        _wait_for_refine(rig.session)
        rig.session.poll(can_install=True)

        assert rig.provider.calibration is loaded
        assert rig.pipeline.calibration_changes == []
        assert not rig.session.is_refining

    def test_unmeasured_baseline_is_taken_at_the_first_rest(
        self, monkeypatch: pytest.MonkeyPatch, factory_path: Path, fake_calibrator: type, squat_views: list
    ) -> None:
        drifted = _perturbed(_true_calibration(), DRIFT_ROTATION_DEG, DRIFT_SHIFT_M)
        _save(drifted, factory_path, WORLD_ANCHOR_PERSON, 0.0)
        rig = _Rig(monkeypatch, factory_path, squat_views)
        rig.session.establish()
        rig.run_frames(len(squat_views))

        rig.session.on_rest_start()
        rig.session.on_rest_start()

        assert not rig.session.is_refining
        assert fake_calibrator.instances == []

    def test_rest_without_two_camera_frames_does_nothing(
        self, monkeypatch: pytest.MonkeyPatch, factory_path: Path, fake_calibrator: type, squat_views: list
    ) -> None:
        _save(_true_calibration(), factory_path, WORLD_ANCHOR_PERSON, STORED_RMS_PX)
        rig = _Rig(monkeypatch, factory_path, squat_views)
        rig.session.establish()

        rig.session.on_rest_start()

        assert not rig.session.is_refining
        assert fake_calibrator.instances == []
