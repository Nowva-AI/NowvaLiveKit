"""
Multi-camera pose provider.

Captures synced frames from N cameras, runs RTMPose halpe26 on each,
and triangulates to 3D via DLT. From the pipeline's perspective this
is a black box that produces (frame, Skeleton2D, Skeleton3D).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from biomechanics.pose.rtmpose import RTMPoseEstimator
from biomechanics.triangulation.calibration import (
    CalibrationResult,
    TPoseCalibrator,
)
from biomechanics.triangulation.multi_capture import (
    DEFAULT_MAX_SYNC_DELTA_MS,
    MultiCameraCapture,
)
from biomechanics.triangulation.triangulator import DLTTriangulator
from biomechanics.utils.types import MultiViewPose, Skeleton2D, Skeleton3D

logger = logging.getLogger(__name__)


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
    ):
        self._device_ids = device_ids
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
        self._initialized = False

    @property
    def is_calibrated(self) -> bool:
        return self._calibration is not None

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
        self._triangulator = DLTTriangulator(
            calibration=self._calibration,
            min_views=self._min_views,
            max_reprojection_error=self._max_reprojection_error,
            min_confidence=self._confidence_threshold,
        )
        logger.info(
            "Loaded calibration with %d cameras", len(self._calibration.cameras)
        )

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
            resolution=self._resolution,
        )

        self._triangulator = DLTTriangulator(
            calibration=self._calibration,
            min_views=self._min_views,
            max_reprojection_error=self._max_reprojection_error,
            min_confidence=self._confidence_threshold,
        )

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
        simply absent from the triangulation views.
        """
        if self._capture is None or self._triangulator is None:
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

        primary_frame = synced.frames[primary_id]
        if len(views) < self._min_views:
            return primary_frame, primary_skeleton_2d, None

        multi_view = MultiViewPose(
            views=views,
            timestamp=synced.timestamp,
            frame_index=synced.sequence,
        )
        skeleton_3d = self._triangulator.triangulate(multi_view)

        return primary_frame, primary_skeleton_2d, skeleton_3d

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
        self._initialized = False
