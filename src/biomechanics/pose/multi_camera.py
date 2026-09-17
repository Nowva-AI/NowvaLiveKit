"""
Multi-camera pose provider.

Captures synced frames from N cameras, runs RTMPose halpe26 on each,
and triangulates to 3D via DLT. From the pipeline's perspective this
is a black box that produces (frame, Skeleton2D, Skeleton3D). It also keeps
the recent per-camera 2D views (and, inside a capture window, barbell ends)
that person-based camera calibration and the drift monitor consume.
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

from biomechanics.pose.rtmpose import RTMPoseEstimator
from biomechanics.triangulation.calibration import (
    NUM_DISTORTION_COEFFS,
    CalibrationResult,
    TPoseCalibrator,
    load_rig_intrinsics,
)
from biomechanics.triangulation.multi_capture import (
    DEFAULT_MAX_SYNC_DELTA_MS,
    MultiCameraCapture,
)
from biomechanics.triangulation.triangulator import DLTTriangulator
from biomechanics.utils.types import BarbellDetection, MultiViewPose, Skeleton2D, Skeleton3D

logger = logging.getLogger(__name__)

DEFAULT_CALIBRATION_BUFFER_FRAMES = 450
DEFAULT_BAR_DETECTION_STRIDE = 3
# Calibration needs the same instant seen by two cameras, whatever triangulation's min_views is.
MIN_CALIBRATION_VIEWS = 2
INTRINSICS_COMMAND = "venv/bin/python scripts/tools/calibrate_cameras.py intrinsics"


class BarEndDetector(Protocol):
    def detect(
        self, frame: np.ndarray, timestamp: float = 0.0, frame_index: int = 0
    ) -> BarbellDetection | None: ...


def _guessed_intrinsic_matrix(resolution: tuple[int, int], focal_length_factor: float) -> np.ndarray:
    width, height = resolution
    focal_px = focal_length_factor * width
    return np.array(
        [[focal_px, 0.0, width / 2.0], [0.0, focal_px, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


class MultiCameraPoseProvider:
    """
    Encapsulates multi-camera capture, per-view RTMPose estimation, and
    DLT triangulation into a single get_pose() call.
    """

    def __init__(
        self,
        device_ids: list[int],
        confidence_threshold: float = 0.3,
        model_path: str | None = None,
        min_views: int = 2,
        max_reprojection_error: float = 15.0,
        max_sync_delta_ms: float = DEFAULT_MAX_SYNC_DELTA_MS,
        resolution: tuple[int, int] = (1280, 720),
        primary_camera: int = 0,
        focal_length_factor: float = 0.8,
        camera_keys: dict[int, str] | None = None,
        calibration_buffer_frames: int = DEFAULT_CALIBRATION_BUFFER_FRAMES,
        bar_detection_stride: int = DEFAULT_BAR_DETECTION_STRIDE,
        bar_detector: BarEndDetector | None = None,
    ):
        self._device_ids = device_ids
        # device id -> name of its ~/.nowva/intrinsics_<camera_key>.json (default: the device id).
        self._camera_keys = {
            str(dev_id): (camera_keys or {}).get(dev_id, str(dev_id)) for dev_id in device_ids
        }
        self._confidence_threshold = confidence_threshold
        self._model_path = model_path
        self._min_views = min_views
        self._max_reprojection_error = max_reprojection_error
        self._max_sync_delta_ms = max_sync_delta_ms
        self._resolution = resolution
        self._primary_camera = primary_camera
        self._focal_length_factor = focal_length_factor

        self._capture: MultiCameraCapture | None = None
        self._estimator: RTMPoseEstimator | None = None
        self._triangulator: DLTTriangulator | None = None
        self._calibration: CalibrationResult | None = None
        self._intrinsics: dict[str, tuple[np.ndarray, np.ndarray]] | None = None
        self._initialized = False

        # Per frame seen by >= 2 cameras: (cam_id -> raw-pixel skeleton, cam_id -> bar ends).
        self._view_buffer: deque[tuple[dict[str, Skeleton2D], dict[str, BarbellDetection]]] = deque(
            maxlen=calibration_buffer_frames
        )
        self._bar_detector = bar_detector
        self._bar_detection_stride = bar_detection_stride
        self._capture_window_open = False
        self._window_frame_count = 0

    @property
    def is_calibrated(self) -> bool:
        return self._calibration is not None

    @property
    def calibration(self) -> CalibrationResult | None:
        return self._calibration

    @property
    def has_bar_detector(self) -> bool:
        return self._bar_detector is not None

    @property
    def capture_resolution(self) -> tuple[int, int]:
        """The primary camera's actual resolution (the requested one until the cameras are open)."""
        return self._camera_resolution(str(self._primary_camera))

    def initialize(self) -> bool:
        """Create the RTMPose halpe26 estimator."""
        if self._initialized:
            return True

        self._estimator = RTMPoseEstimator(
            confidence_threshold=self._confidence_threshold,
            model_path=self._model_path,
            keypoint_format="halpe26",
            batch_size=len(self._device_ids),
        )
        self._estimator.initialize()
        self._initialized = True
        return True

    def start(self) -> None:
        """Open cameras and start reader threads."""
        if not self._initialized:
            self.initialize()

        self._capture = MultiCameraCapture(
            device_ids=self._device_ids,
            resolution=self._resolution,
            max_sync_delta_ms=self._max_sync_delta_ms,
            min_views=self._min_views,
            primary_device_id=self._primary_camera,
        )
        self._capture.start()

    def load_calibration(self, path: str) -> None:
        """Load a saved calibration and build the triangulator."""
        self._calibration = TPoseCalibrator.load_calibration(path)
        self._triangulator = self._build_triangulator(self._calibration)
        logger.info(
            "Loaded calibration with %d cameras", len(self._calibration.cameras)
        )

    def install_calibration(self, calibration: CalibrationResult) -> None:
        """Swap in a new calibration mid-session: rebuild the triangulator and clear temporal state."""
        self._calibration = calibration
        self._triangulator = self._build_triangulator(calibration)
        self.reset_temporal_state()
        logger.info("Installed calibration with %d cameras", len(calibration.cameras))

    def _build_triangulator(self, calibration: CalibrationResult) -> DLTTriangulator:
        return DLTTriangulator(
            calibration=calibration,
            min_views=self._min_views,
            max_reprojection_error=self._max_reprojection_error,
            min_confidence=self._confidence_threshold,
        )

    def _camera_resolution(self, cam_id: str) -> tuple[int, int]:
        actual_resolutions = self._capture.actual_resolutions if self._capture is not None else {}
        width, height = actual_resolutions.get(cam_id, self._resolution)
        # Some capture backends report 0x0; the request is the best guess then.
        return (width, height) if width > 0 and height > 0 else self._resolution

    def rig_intrinsics(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """
        camera_id -> (K, distortion_coeffs) for every camera, looked up at the resolution the
        camera actually opened at. Cameras without saved intrinsics get the guessed pinhole
        (f = focal_length_factor * width, centred, no distortion) and one warning naming the fix.
        """
        if self._intrinsics is not None:
            return self._intrinsics

        intrinsics: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        missing_commands: list[str] = []
        for dev_id in self._device_ids:
            cam_id = str(dev_id)
            resolution = self._camera_resolution(cam_id)
            camera_key = self._camera_keys[cam_id]
            intrinsics.update(load_rig_intrinsics({cam_id: camera_key}, resolution))
            if cam_id in intrinsics:
                continue
            intrinsics[cam_id] = (
                _guessed_intrinsic_matrix(resolution, self._focal_length_factor),
                np.zeros(NUM_DISTORTION_COEFFS, dtype=np.float64),
            )
            key_argument = "" if camera_key == cam_id else f" --key {camera_key}"
            missing_commands.append(
                f"{INTRINSICS_COMMAND} --camera {dev_id}{key_argument} --resolution {resolution[0]}x{resolution[1]}"
            )
        if missing_commands:
            logger.warning(
                "No calibrated intrinsics for %d of %d cameras: guessing f = %.2f x width with no lens "
                "distortion, which costs 3D accuracy. Calibrate each camera once with the ChArUco board:\n  %s",
                len(missing_commands), len(self._device_ids), self._focal_length_factor,
                "\n  ".join(missing_commands),
            )
        self._intrinsics = intrinsics
        return intrinsics

    def calibrate(
        self,
        height_m: float,
        n_frames: int = 30,
        save_path: str | None = None,
    ) -> CalibrationResult:
        """
        Run T-pose calibration: capture frames from all cameras, detect
        keypoints, solve for camera extrinsics.
        """
        if not self._initialized:
            self.initialize()
        if self._capture is None:
            self.start()

        calibrator = TPoseCalibrator(
            pose_estimator=self._estimator,
            focal_length_factor=self._focal_length_factor,
            intrinsics=self.rig_intrinsics(),
        )

        logger.info("Capturing %d T-pose frames from %d cameras...", n_frames, len(self._device_ids))
        frames_per_camera: dict[str, list[np.ndarray]] = {
            str(dev_id): [] for dev_id in self._device_ids
        }

        collected = 0
        while collected < n_frames:
            synced = self._capture.get_synced_frames()
            if synced is None:
                continue
            for cam_id, frame in synced.frames.items():
                frames_per_camera[cam_id].append(frame)
            collected += 1

        # Partial synced sets can leave a camera without frames; the calibrator
        # handles each camera independently, so pass only cameras that have some.
        self._calibration = calibrator.calibrate(
            frames={
                cam_id: cam_frames
                for cam_id, cam_frames in frames_per_camera.items()
                if cam_frames
            },
            height_m=height_m,
            resolution=self.capture_resolution,
        )
        self._triangulator = self._build_triangulator(self._calibration)

        if save_path is not None:
            TPoseCalibrator.save_calibration(self._calibration, save_path)

        logger.info("Calibration complete: %d cameras", len(self._calibration.cameras))
        return self._calibration

    def get_pose(
        self,
    ) -> tuple[np.ndarray | None, Skeleton2D | None, Skeleton3D | None]:
        """
        Grab the next unprocessed synced set, run per-camera pose estimation, triangulate.

        Returns (primary_frame, primary_skeleton_2d, triangulated_skeleton_3d). Every
        returned skeleton carries the primary camera's capture sequence number as
        frame_index (strictly increasing, never repeated) and the primary capture
        timestamp (wall-aligned seconds). Cameras missing from the synced set are
        simply absent from the triangulation views. Before a calibration exists the
        2D side still runs (and fills the calibration buffer); the 3D skeleton is None.
        """
        if self._capture is None:
            return None, None, None

        synced = self._capture.get_synced_frames()
        if synced is None:
            return None, None, None

        primary_id = str(self._primary_camera)

        views: dict[str, Skeleton2D] = {}
        primary_skeleton_2d: Skeleton2D | None = None

        cam_ids = list(synced.frames.keys())
        skeletons = self._estimator.estimate_batch(
            [synced.frames[cam_id] for cam_id in cam_ids], camera_ids=cam_ids
        )
        for cam_id, skeleton_2d in zip(cam_ids, skeletons):
            if skeleton_2d is not None:
                skeleton_2d.timestamp = synced.timestamp
                skeleton_2d.frame_index = synced.sequence
                views[cam_id] = skeleton_2d
                if cam_id == primary_id:
                    primary_skeleton_2d = skeleton_2d

        self._record_views(synced.frames, views, synced.timestamp, synced.sequence)

        primary_frame = synced.frames[primary_id]
        if self._triangulator is None or len(views) < self._min_views:
            return primary_frame, primary_skeleton_2d, None

        multi_view = MultiViewPose(
            views=views,
            timestamp=synced.timestamp,
            frame_index=synced.sequence,
        )
        skeleton_3d = self._triangulator.triangulate(multi_view)

        return primary_frame, primary_skeleton_2d, skeleton_3d

    def _record_views(
        self,
        frames: dict[str, np.ndarray],
        views: dict[str, Skeleton2D],
        timestamp: float,
        frame_index: int,
    ) -> None:
        if len(views) < MIN_CALIBRATION_VIEWS:
            return
        bar_ends: dict[str, BarbellDetection] = {}
        if self._capture_window_open and self._bar_detector is not None:
            if self._window_frame_count % self._bar_detection_stride == 0:
                # A bar end is only usable in a view whose shoulders label it left/right.
                for cam_id in views:
                    detection = self._bar_detector.detect(
                        frames[cam_id], timestamp=timestamp, frame_index=frame_index
                    )
                    if detection is not None:
                        bar_ends[cam_id] = detection
            self._window_frame_count += 1
        self._view_buffer.append((views, bar_ends))

    def begin_calibration_capture(self) -> None:
        """
        Open a capture window: the view buffer restarts so it holds only window frames, and
        bar-end detection (when a detector is available) runs on every bar_detection_stride-th
        buffered frame until end_calibration_capture(). Never open one during a set.
        """
        self._view_buffer.clear()
        self._window_frame_count = 0
        self._capture_window_open = True

    def end_calibration_capture(self) -> None:
        self._capture_window_open = False

    @property
    def calibration_frame_count(self) -> int:
        return len(self._view_buffer)

    def calibration_views(self) -> list[dict[str, Skeleton2D]]:
        """Buffered frames seen by >= 2 cameras, oldest first: cam_id -> raw-pixel Skeleton2D."""
        return [views for views, _ in self._view_buffer]

    def calibration_bar_ends(self) -> list[dict[str, BarbellDetection]]:
        """Bar-end detections indexed like calibration_views(); empty where none was run or found."""
        return [bar_ends for _, bar_ends in self._view_buffer]

    def recent_views(self, n_frames: int) -> list[dict[str, Skeleton2D]]:
        """The newest n_frames of calibration_views()."""
        return self.calibration_views()[-n_frames:]

    def reset_temporal_state(self) -> None:
        """Clear per-keypoint history in the triangulator and the per-camera crop tracking.

        Call at set boundaries and after long dropouts; cameras and calibration
        are untouched.
        """
        if self._triangulator is not None:
            self._triangulator.reset()
        if self._estimator is not None:
            self._estimator.reset_tracking()

    @property
    def swap_count(self) -> int:
        """Cumulative left/right swap corrections applied by the triangulator."""
        if self._triangulator is None:
            return 0
        return self._triangulator.swap_count

    def release(self) -> None:
        """Stop capture and release resources."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        if self._estimator is not None:
            self._estimator.release()
            self._estimator = None
        self._triangulator = None
        self._calibration = None
        self._intrinsics = None
        self._view_buffer.clear()
        self._capture_window_open = False
        self._initialized = False
