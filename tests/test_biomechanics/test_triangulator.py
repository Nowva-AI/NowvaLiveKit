"""
Tests for DLTTriangulator: robust view selection, left/right swap test,
metric confidence, world-coordinate output, and recentre_at_hips.

Uses a synthetic 3-camera rig (3.5 m, yaw -40/0/+40, focal 0.8*1280, 1280x720)
and a Y-down squat skeleton (X = subject's left, forward = -Z).
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from biomechanics.triangulation.calibration import CalibrationResult, CameraCalibration
from biomechanics.triangulation.triangulator import (
    NUM_KEYPOINTS,
    PREVIOUS_POINT_MAX_AGE_S,
    TWO_VIEW_CONFIDENCE_CAP,
    DLTTriangulator,
    recentre_at_hips,
)
from biomechanics.utils.types import CocoKeypoints as CK
from biomechanics.utils.types import Keypoint2D, MultiViewPose, Point3D, Skeleton2D, Skeleton3D

IMAGE_WIDTH_PX = 1280
IMAGE_HEIGHT_PX = 720
FOCAL_PX = 0.8 * IMAGE_WIDTH_PX
CAMERA_YAWS_DEG = (-40.0, 0.0, 40.0)
CAMERA_DISTANCE_M = 3.5
FLOOR_Y_M = 0.98
CAMERA_HEIGHT_ABOVE_FLOOR_M = 1.0
AIM_HEIGHT_ABOVE_FLOOR_M = 0.9
FRAME_PERIOD_S = 1.0 / 30.0
FRAMES_PER_REP = 60

CLEAN_NOISE_PX = 1.5
LOW_NOISE_PX = 1.0
EXACT_TOLERANCE_M = 1e-6
CLEAN_MEAN_ERROR_LIMIT_M = 0.01
OUTLIER_ERROR_LIMIT_M = 0.03
SWAP_ERROR_LIMIT_M = 0.02
HIDDEN_HIP_OFFSET_M = np.array([0.15, 0.0, 0.2])
LENS_DISTORTION = np.array([-0.25, 0.08, 0.001, -0.001, 0.0])
UNDISTORTED_ERROR_LIMIT_M = 1e-4
IGNORED_DISTORTION_MIN_ERROR_M = 0.005

LEFT_RIGHT_PAIRS = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18), (19, 20))
LEG_PAIRS = ((11, 12), (13, 14), (15, 16), (17, 18), (19, 20))


def _camera_calibration(camera_id: str, yaw_deg: float) -> CameraCalibration:
    yaw_rad = math.radians(yaw_deg)
    centre = np.array(
        [
            CAMERA_DISTANCE_M * math.sin(yaw_rad),
            FLOOR_Y_M - CAMERA_HEIGHT_ABOVE_FLOOR_M,
            -CAMERA_DISTANCE_M * math.cos(yaw_rad),
        ]
    )
    target = np.array([0.0, FLOOR_Y_M - AIM_HEIGHT_ABOVE_FLOOR_M, 0.0])
    z_axis = (target - centre) / np.linalg.norm(target - centre)
    x_axis = np.cross([0.0, 1.0, 0.0], z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rotation = np.stack([x_axis, y_axis, z_axis])
    translation = -rotation @ centre
    intrinsics = np.array(
        [[FOCAL_PX, 0.0, IMAGE_WIDTH_PX / 2], [0.0, FOCAL_PX, IMAGE_HEIGHT_PX / 2], [0.0, 0.0, 1.0]]
    )
    return CameraCalibration(
        camera_id=camera_id,
        projection_matrix=intrinsics @ np.hstack([rotation, translation[:, None]]),
        intrinsic_matrix=intrinsics,
        rotation_matrix=rotation,
        translation_vector=translation,
        reprojection_error=0.0,
        resolution=(IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX),
    )


def _rig_calibration(yaws_deg: tuple[float, ...] = CAMERA_YAWS_DEG) -> CalibrationResult:
    cameras = {f"cam{i}": _camera_calibration(f"cam{i}", yaw) for i, yaw in enumerate(yaws_deg)}
    return CalibrationResult(cameras=cameras, athlete_height_m=1.885)


def _squat_skeleton(depth_ratio: float, offset_m: np.ndarray | None = None) -> np.ndarray:
    knee_flexion_rad = math.radians(110.0 * depth_ratio)
    dorsiflexion_rad = math.radians(30.0 * depth_ratio)
    trunk_lean_rad = math.radians(35.0 * depth_ratio)
    thigh_rad = knee_flexion_rad - dorsiflexion_rad
    up = np.array([0.0, -1.0, 0.0])
    forward = np.array([0.0, 0.0, -1.0])
    points = np.zeros((NUM_KEYPOINTS, 3))
    sides = (
        (1.0, CK.LEFT_HIP, CK.LEFT_KNEE, CK.LEFT_ANKLE, CK.LEFT_FOOT_INDEX, CK.LEFT_HEEL),
        (-1.0, CK.RIGHT_HIP, CK.RIGHT_KNEE, CK.RIGHT_ANKLE, CK.RIGHT_FOOT_INDEX, CK.RIGHT_HEEL),
    )
    for sign, hip, knee, ankle, toe, heel in sides:
        points[ankle] = [sign * 0.20, FLOOR_Y_M - 0.08, 0.0]
        points[knee] = points[ankle] + 0.44 * (
            up * math.cos(dorsiflexion_rad) + forward * math.sin(dorsiflexion_rad)
        )
        points[knee, 0] = sign * 0.18
        points[hip] = points[knee] + 0.46 * (up * math.cos(thigh_rad) - forward * math.sin(thigh_rad))
        points[hip, 0] = sign * 0.14
        points[toe] = points[ankle] + [sign * 0.04, 0.06, -0.16]
        points[heel] = points[ankle] + [0.0, 0.07, 0.06]
    trunk_direction = up * math.cos(trunk_lean_rad) + forward * math.sin(trunk_lean_rad)
    shoulder_mid = (points[CK.LEFT_HIP] + points[CK.RIGHT_HIP]) / 2 + 0.55 * trunk_direction
    points[CK.LEFT_SHOULDER] = shoulder_mid + [0.20, 0.0, 0.0]
    points[CK.RIGHT_SHOULDER] = shoulder_mid + [-0.20, 0.0, 0.0]
    nose = shoulder_mid + 0.25 * trunk_direction + 0.09 * forward
    points[CK.NOSE] = nose
    points[CK.LEFT_EYE] = nose + [0.03, -0.035, 0.03]
    points[CK.RIGHT_EYE] = nose + [-0.03, -0.035, 0.03]
    points[CK.LEFT_EAR] = nose + [0.075, -0.02, 0.10]
    points[CK.RIGHT_EAR] = nose + [-0.075, -0.02, 0.10]
    for sign, shoulder, elbow, wrist in (
        (1.0, CK.LEFT_SHOULDER, CK.LEFT_ELBOW, CK.LEFT_WRIST),
        (-1.0, CK.RIGHT_SHOULDER, CK.RIGHT_ELBOW, CK.RIGHT_WRIST),
    ):
        points[elbow] = points[shoulder] + [sign * 0.08, 0.15, 0.20]
        points[wrist] = points[elbow] + [-sign * 0.02, -0.20, -0.05]
    if offset_m is not None:
        points = points + offset_m
    return points


def _squat_depth(frame_idx: int) -> float:
    return 0.5 - 0.5 * math.cos(2.0 * math.pi * frame_idx / FRAMES_PER_REP)


def _project(calibration: CalibrationResult, points: np.ndarray) -> np.ndarray:
    pixels = []
    for cam_id in sorted(calibration.cameras):
        projected = np.hstack([points, np.ones((len(points), 1))]) @ calibration.cameras[cam_id].projection_matrix.T
        pixels.append(projected[:, :2] / projected[:, 2:3])
    return np.stack(pixels)


def _project_through_lens(calibration: CalibrationResult, points: np.ndarray) -> np.ndarray:
    pixels = []
    for cam_id in sorted(calibration.cameras):
        camera = calibration.cameras[cam_id]
        projected, _ = cv2.projectPoints(
            points, cv2.Rodrigues(camera.rotation_matrix)[0], camera.translation_vector,
            camera.intrinsic_matrix, LENS_DISTORTION,
        )
        pixels.append(projected.reshape(-1, 2))
    return np.stack(pixels)


def _multi_view(
    pixels: np.ndarray,
    confidences: np.ndarray | None = None,
    timestamp: float = 0.0,
    frame_index: int = 0,
) -> MultiViewPose:
    if confidences is None:
        confidences = np.ones(pixels.shape[:2])
    views = {
        f"cam{view_idx}": Skeleton2D(
            keypoints=[
                Keypoint2D(
                    x=float(pixels[view_idx, k, 0]),
                    y=float(pixels[view_idx, k, 1]),
                    confidence=float(confidences[view_idx, k]),
                )
                for k in range(pixels.shape[1])
            ]
        )
        for view_idx in range(pixels.shape[0])
    }
    return MultiViewPose(views=views, timestamp=timestamp, frame_index=frame_index)


def _skeleton_arrays(skeleton: Skeleton3D) -> tuple[np.ndarray, np.ndarray]:
    positions = np.array([[kp.x, kp.y, kp.z] for kp in skeleton.keypoints])
    confidences = np.array([kp.confidence for kp in skeleton.keypoints])
    return positions, confidences


def _swap_permutation(pairs: tuple[tuple[int, int], ...]) -> np.ndarray:
    permutation = np.arange(NUM_KEYPOINTS)
    for left_idx, right_idx in pairs:
        permutation[left_idx], permutation[right_idx] = right_idx, left_idx
    return permutation


def _make_skeleton_3d(
    positions: np.ndarray, confidences: np.ndarray, timestamp: float = 1.25, frame_index: int = 7
) -> Skeleton3D:
    return Skeleton3D(
        keypoints=[
            Point3D(x=float(p[0]), y=float(p[1]), z=float(p[2]), confidence=float(c))
            for p, c in zip(positions, confidences)
        ],
        timestamp=timestamp,
        frame_index=frame_index,
    )


@pytest.fixture
def rig_calibration() -> CalibrationResult:
    return _rig_calibration()


class TestCleanTriangulation:
    @pytest.mark.parametrize("depth_ratio", [0.0, 0.5, 1.0])
    def test_noiseless_views_reconstruct_exactly(
        self, rig_calibration: CalibrationResult, depth_ratio: float
    ) -> None:
        truth = _squat_skeleton(depth_ratio)
        skeleton = DLTTriangulator(rig_calibration).triangulate(_multi_view(_project(rig_calibration, truth)))
        positions, confidences = _skeleton_arrays(skeleton)
        assert np.abs(positions - truth).max() == pytest.approx(0.0, abs=EXACT_TOLERANCE_M)
        assert (confidences > 0.0).all()

    def test_mean_error_below_1cm_at_1p5px_noise(self, rig_calibration: CalibrationResult) -> None:
        rng = np.random.default_rng(0)
        triangulator = DLTTriangulator(rig_calibration)
        errors = []
        for frame_idx in range(90):
            truth = _squat_skeleton(_squat_depth(frame_idx))
            pixels = _project(rig_calibration, truth) + rng.normal(0.0, CLEAN_NOISE_PX, (3, NUM_KEYPOINTS, 2))
            skeleton = triangulator.triangulate(_multi_view(pixels, timestamp=frame_idx * FRAME_PERIOD_S))
            positions, confidences = _skeleton_arrays(skeleton)
            assert (confidences > 0.0).all()
            errors.append(np.linalg.norm(positions - truth, axis=1))
        assert np.mean(errors) < CLEAN_MEAN_ERROR_LIMIT_M

    def test_output_has_21_keypoints_and_frame_metadata(self, rig_calibration: CalibrationResult) -> None:
        pixels = _project(rig_calibration, _squat_skeleton(0.3))
        skeleton = DLTTriangulator(rig_calibration).triangulate(
            _multi_view(pixels, timestamp=12.5, frame_index=42)
        )
        assert len(skeleton.keypoints) == NUM_KEYPOINTS
        assert skeleton.timestamp == pytest.approx(12.5, abs=EXACT_TOLERANCE_M)
        assert skeleton.frame_index == 42

    def test_19_keypoint_views_leave_heels_untriangulated(self, rig_calibration: CalibrationResult) -> None:
        truth = _squat_skeleton(0.5)
        pixels = _project(rig_calibration, truth)[:, : CK.LEFT_HEEL]
        positions, confidences = _skeleton_arrays(DLTTriangulator(rig_calibration).triangulate(_multi_view(pixels)))
        for heel in (CK.LEFT_HEEL, CK.RIGHT_HEEL):
            assert confidences[heel] == 0.0
            assert positions[heel].tolist() == [0.0, 0.0, 0.0]
        for toe in (CK.LEFT_FOOT_INDEX, CK.RIGHT_FOOT_INDEX):
            assert confidences[toe] > 0.0
            assert np.linalg.norm(positions[toe] - truth[toe]) == pytest.approx(0.0, abs=EXACT_TOLERANCE_M)


class TestWorldCoordinates:
    def test_output_is_not_hip_centred(self, rig_calibration: CalibrationResult) -> None:
        truth = _squat_skeleton(0.4, offset_m=HIDDEN_HIP_OFFSET_M)
        positions, _ = _skeleton_arrays(
            DLTTriangulator(rig_calibration).triangulate(_multi_view(_project(rig_calibration, truth)))
        )
        hip_midpoint = (positions[CK.LEFT_HIP] + positions[CK.RIGHT_HIP]) / 2
        true_hip_midpoint = (truth[CK.LEFT_HIP] + truth[CK.RIGHT_HIP]) / 2
        assert np.linalg.norm(hip_midpoint - true_hip_midpoint) == pytest.approx(0.0, abs=EXACT_TOLERANCE_M)
        assert np.linalg.norm(hip_midpoint) > 0.1
        assert positions[CK.LEFT_ANKLE, 1] == pytest.approx(truth[CK.LEFT_ANKLE, 1], abs=EXACT_TOLERANCE_M)


class TestOutlierRejection:
    @pytest.mark.parametrize("outlier_px", [80.0, 150.0])
    def test_single_view_knee_outlier_error_within_3cm(
        self, rig_calibration: CalibrationResult, outlier_px: float
    ) -> None:
        rng = np.random.default_rng(int(outlier_px))
        triangulator = DLTTriangulator(rig_calibration)
        for frame_idx in range(40):
            truth = _squat_skeleton(_squat_depth(frame_idx))
            pixels = _project(rig_calibration, truth) + rng.normal(0.0, LOW_NOISE_PX, (3, NUM_KEYPOINTS, 2))
            if frame_idx > 0:
                direction_rad = rng.uniform(0.0, 2.0 * math.pi)
                view_idx = int(rng.integers(0, 3))
                pixels[view_idx, CK.LEFT_KNEE] += outlier_px * np.array(
                    [math.cos(direction_rad), math.sin(direction_rad)]
                )
            skeleton = triangulator.triangulate(_multi_view(pixels, timestamp=frame_idx * FRAME_PERIOD_S))
            positions, confidences = _skeleton_arrays(skeleton)
            assert confidences[CK.LEFT_KNEE] > 0.0
            assert np.linalg.norm(positions[CK.LEFT_KNEE] - truth[CK.LEFT_KNEE]) <= OUTLIER_ERROR_LIMIT_M

    def test_vertical_outlier_resolved_without_history(self, rig_calibration: CalibrationResult) -> None:
        truth = _squat_skeleton(0.6)
        pixels = _project(rig_calibration, truth)
        pixels[0, CK.RIGHT_KNEE, 1] += 80.0
        positions, confidences = _skeleton_arrays(DLTTriangulator(rig_calibration).triangulate(_multi_view(pixels)))
        assert confidences[CK.RIGHT_KNEE] > 0.0
        assert np.linalg.norm(positions[CK.RIGHT_KNEE] - truth[CK.RIGHT_KNEE]) <= OUTLIER_ERROR_LIMIT_M

    def test_horizontal_outlier_without_history_is_rejected(self, rig_calibration: CalibrationResult) -> None:
        truth = _squat_skeleton(0.6)
        pixels = _project(rig_calibration, truth)
        pixels[1, CK.LEFT_KNEE, 0] += 150.0
        positions, confidences = _skeleton_arrays(DLTTriangulator(rig_calibration).triangulate(_multi_view(pixels)))
        assert confidences[CK.LEFT_KNEE] == 0.0
        assert positions[CK.LEFT_KNEE].tolist() == [0.0, 0.0, 0.0]
        assert confidences[CK.RIGHT_KNEE] > 0.0

    def test_previous_point_breaks_horizontal_outlier_tie(self, rig_calibration: CalibrationResult) -> None:
        triangulator = DLTTriangulator(rig_calibration)
        triangulator.triangulate(_multi_view(_project(rig_calibration, _squat_skeleton(0.58)), timestamp=0.0))
        truth = _squat_skeleton(0.6)
        pixels = _project(rig_calibration, truth)
        pixels[1, CK.LEFT_KNEE, 0] += 150.0
        positions, confidences = _skeleton_arrays(
            triangulator.triangulate(_multi_view(pixels, timestamp=FRAME_PERIOD_S))
        )
        assert confidences[CK.LEFT_KNEE] > 0.0
        assert np.linalg.norm(positions[CK.LEFT_KNEE] - truth[CK.LEFT_KNEE]) <= OUTLIER_ERROR_LIMIT_M

    def test_reset_clears_previous_points(self, rig_calibration: CalibrationResult) -> None:
        triangulator = DLTTriangulator(rig_calibration)
        triangulator.triangulate(_multi_view(_project(rig_calibration, _squat_skeleton(0.58)), timestamp=0.0))
        triangulator.reset()
        pixels = _project(rig_calibration, _squat_skeleton(0.6))
        pixels[1, CK.LEFT_KNEE, 0] += 150.0
        _, confidences = _skeleton_arrays(triangulator.triangulate(_multi_view(pixels, timestamp=FRAME_PERIOD_S)))
        assert confidences[CK.LEFT_KNEE] == 0.0

    def test_stale_previous_point_is_ignored(self, rig_calibration: CalibrationResult) -> None:
        triangulator = DLTTriangulator(rig_calibration)
        triangulator.triangulate(_multi_view(_project(rig_calibration, _squat_skeleton(0.58)), timestamp=0.0))
        pixels = _project(rig_calibration, _squat_skeleton(0.6))
        pixels[1, CK.LEFT_KNEE, 0] += 150.0
        stale_timestamp = 2.0 * PREVIOUS_POINT_MAX_AGE_S
        _, confidences = _skeleton_arrays(triangulator.triangulate(_multi_view(pixels, timestamp=stale_timestamp)))
        assert confidences[CK.LEFT_KNEE] == 0.0


class TestLeftRightSwap:
    @pytest.mark.parametrize("pairs", [LEFT_RIGHT_PAIRS, LEG_PAIRS], ids=["full", "legs_only"])
    @pytest.mark.parametrize("swapped_view", [0, 1, 2])
    def test_single_view_swap_corrected_within_2cm(
        self,
        rig_calibration: CalibrationResult,
        pairs: tuple[tuple[int, int], ...],
        swapped_view: int,
    ) -> None:
        rng = np.random.default_rng(swapped_view)
        permutation = _swap_permutation(pairs)
        triangulator = DLTTriangulator(rig_calibration)
        num_frames = 10
        for frame_idx in range(num_frames):
            truth = _squat_skeleton(_squat_depth(frame_idx * 3))
            pixels = _project(rig_calibration, truth) + rng.normal(0.0, LOW_NOISE_PX, (3, NUM_KEYPOINTS, 2))
            pixels[swapped_view] = pixels[swapped_view, permutation]
            skeleton = triangulator.triangulate(_multi_view(pixels, timestamp=frame_idx * FRAME_PERIOD_S))
            positions, confidences = _skeleton_arrays(skeleton)
            assert (confidences > 0.0).all()
            assert np.linalg.norm(positions - truth, axis=1).max() <= SWAP_ERROR_LIMIT_M
        assert triangulator.swap_count == num_frames

    def test_clean_and_outlier_frames_never_swap(self, rig_calibration: CalibrationResult) -> None:
        rng = np.random.default_rng(5)
        triangulator = DLTTriangulator(rig_calibration)
        for frame_idx in range(60):
            pixels = _project(rig_calibration, _squat_skeleton(_squat_depth(frame_idx)))
            pixels += rng.normal(0.0, CLEAN_NOISE_PX, pixels.shape)
            if frame_idx % 2 == 1:
                pixels[int(rng.integers(0, 3)), CK.LEFT_KNEE] += rng.uniform(-150.0, 150.0, 2)
            triangulator.triangulate(_multi_view(pixels, timestamp=frame_idx * FRAME_PERIOD_S))
        assert triangulator.swap_count == 0

    def test_swap_count_survives_reset(self, rig_calibration: CalibrationResult) -> None:
        pixels = _project(rig_calibration, _squat_skeleton(0.5))
        pixels[0] = pixels[0, _swap_permutation(LEFT_RIGHT_PAIRS)]
        triangulator = DLTTriangulator(rig_calibration)
        triangulator.triangulate(_multi_view(pixels))
        triangulator.reset()
        assert triangulator.swap_count == 1


class TestTwoViewKeypoints:
    def test_two_view_keypoint_triangulated_with_capped_confidence(self, rig_calibration: CalibrationResult) -> None:
        truth = _squat_skeleton(0.5)
        confidences_2d = np.ones((3, NUM_KEYPOINTS))
        confidences_2d[1, CK.LEFT_ANKLE] = 0.0
        positions, confidences = _skeleton_arrays(
            DLTTriangulator(rig_calibration).triangulate(_multi_view(_project(rig_calibration, truth), confidences_2d))
        )
        assert np.linalg.norm(positions[CK.LEFT_ANKLE] - truth[CK.LEFT_ANKLE]) == pytest.approx(
            0.0, abs=EXACT_TOLERANCE_M
        )
        assert 0.0 < confidences[CK.LEFT_ANKLE] <= TWO_VIEW_CONFIDENCE_CAP
        assert confidences[CK.LEFT_ANKLE] < confidences[CK.RIGHT_ANKLE]

    def test_missing_camera_view_uses_remaining_pair(self, rig_calibration: CalibrationResult) -> None:
        truth = _squat_skeleton(0.5)
        multi_view = _multi_view(_project(rig_calibration, truth))
        del multi_view.views["cam2"]
        positions, confidences = _skeleton_arrays(DLTTriangulator(rig_calibration).triangulate(multi_view))
        assert np.abs(positions - truth).max() == pytest.approx(0.0, abs=EXACT_TOLERANCE_M)
        assert ((confidences > 0.0) & (confidences <= TWO_VIEW_CONFIDENCE_CAP)).all()

    def test_min_views_three_rejects_two_view_keypoint(self, rig_calibration: CalibrationResult) -> None:
        confidences_2d = np.ones((3, NUM_KEYPOINTS))
        confidences_2d[0, CK.LEFT_ANKLE] = 0.0
        pixels = _project(rig_calibration, _squat_skeleton(0.5))
        positions, confidences = _skeleton_arrays(
            DLTTriangulator(rig_calibration, min_views=3).triangulate(_multi_view(pixels, confidences_2d))
        )
        assert confidences[CK.LEFT_ANKLE] == 0.0
        assert positions[CK.LEFT_ANKLE].tolist() == [0.0, 0.0, 0.0]

    def test_min_views_below_two_raises(self, rig_calibration: CalibrationResult) -> None:
        with pytest.raises(ValueError):
            DLTTriangulator(rig_calibration, min_views=1)


class TestInvalidInputs:
    def test_keypoint_invalid_in_all_views_is_zero_at_origin(self, rig_calibration: CalibrationResult) -> None:
        confidences_2d = np.ones((3, NUM_KEYPOINTS))
        confidences_2d[:, CK.LEFT_WRIST] = 0.1
        pixels = _project(rig_calibration, _squat_skeleton(0.5))
        positions, confidences = _skeleton_arrays(
            DLTTriangulator(rig_calibration).triangulate(_multi_view(pixels, confidences_2d))
        )
        assert confidences[CK.LEFT_WRIST] == 0.0
        assert positions[CK.LEFT_WRIST].tolist() == [0.0, 0.0, 0.0]

    def test_all_keypoints_invalid_returns_none(self, rig_calibration: CalibrationResult) -> None:
        pixels = _project(rig_calibration, _squat_skeleton(0.5))
        confidences_2d = np.zeros((3, NUM_KEYPOINTS))
        assert DLTTriangulator(rig_calibration).triangulate(_multi_view(pixels, confidences_2d)) is None

    def test_nan_and_inf_inputs_never_produce_nan(self, rig_calibration: CalibrationResult) -> None:
        truth = _squat_skeleton(0.5)
        pixels = _project(rig_calibration, truth)
        confidences_2d = np.ones((3, NUM_KEYPOINTS))
        pixels[0, CK.LEFT_KNEE, 0] = np.nan
        pixels[1, CK.RIGHT_KNEE, 1] = np.inf
        confidences_2d[2, CK.NOSE] = np.nan
        pixels[:, CK.LEFT_EAR] = np.nan
        positions, confidences = _skeleton_arrays(
            DLTTriangulator(rig_calibration).triangulate(_multi_view(pixels, confidences_2d))
        )
        assert np.isfinite(positions).all()
        assert np.isfinite(confidences).all()
        for keypoint in (CK.LEFT_KNEE, CK.RIGHT_KNEE, CK.NOSE):
            assert 0.0 < confidences[keypoint] <= TWO_VIEW_CONFIDENCE_CAP
            assert np.linalg.norm(positions[keypoint] - truth[keypoint]) == pytest.approx(0.0, abs=EXACT_TOLERANCE_M)
        assert confidences[CK.LEFT_EAR] == 0.0
        assert positions[CK.LEFT_EAR].tolist() == [0.0, 0.0, 0.0]

    def test_coincident_cameras_give_no_confident_points(self, rig_calibration: CalibrationResult) -> None:
        duplicate_camera = copy.deepcopy(rig_calibration.cameras["cam1"])
        degenerate = CalibrationResult(cameras={"cam0": rig_calibration.cameras["cam1"], "cam1": duplicate_camera})
        pixels = _project(rig_calibration, _squat_skeleton(0.5))[[1, 1]]
        assert DLTTriangulator(degenerate).triangulate(_multi_view(pixels)) is None


class TestMetricConfidence:
    def test_confidence_decreases_with_pixel_noise(self, rig_calibration: CalibrationResult) -> None:
        rng = np.random.default_rng(3)
        mean_confidences = []
        for noise_px in (1.5, 5.0, 10.0):
            triangulator = DLTTriangulator(rig_calibration)
            frame_confidences = []
            for frame_idx in range(40):
                pixels = _project(rig_calibration, _squat_skeleton(_squat_depth(frame_idx)))
                pixels += rng.normal(0.0, noise_px, pixels.shape)
                skeleton = triangulator.triangulate(_multi_view(pixels, timestamp=frame_idx * FRAME_PERIOD_S))
                frame_confidences.append(_skeleton_arrays(skeleton)[1])
            mean_confidences.append(float(np.mean(frame_confidences)))
        assert mean_confidences[0] > mean_confidences[1] > mean_confidences[2]

    def test_half_confidence_corresponds_to_centimetre_scale_error(self, rig_calibration: CalibrationResult) -> None:
        rng = np.random.default_rng(4)
        confidences_all = []
        errors_all = []
        for noise_px in (3.0, 5.0, 8.0):
            triangulator = DLTTriangulator(rig_calibration)
            for frame_idx in range(60):
                truth = _squat_skeleton(_squat_depth(frame_idx))
                pixels = _project(rig_calibration, truth) + rng.normal(0.0, noise_px, (3, NUM_KEYPOINTS, 2))
                skeleton = triangulator.triangulate(_multi_view(pixels, timestamp=frame_idx * FRAME_PERIOD_S))
                positions, confidences = _skeleton_arrays(skeleton)
                confidences_all.append(confidences)
                errors_all.append(np.linalg.norm(positions - truth, axis=1))
        confidences_flat = np.concatenate(confidences_all)
        errors_flat = np.concatenate(errors_all)
        near_half = (confidences_flat >= 0.4) & (confidences_flat <= 0.6)
        assert near_half.sum() > 100
        assert 0.01 <= float(np.median(errors_flat[near_half])) <= 0.04

    def test_clean_three_view_confidence_exceeds_two_view_confidence(
        self, rig_calibration: CalibrationResult
    ) -> None:
        pixels = _project(rig_calibration, _squat_skeleton(0.5))
        three_view = _skeleton_arrays(DLTTriangulator(rig_calibration).triangulate(_multi_view(pixels)))[1]
        assert (three_view > TWO_VIEW_CONFIDENCE_CAP).all()
        assert (three_view < 1.0).all()


class TestLensDistortion:
    def test_distorted_views_reconstruct_exactly_when_calibration_has_the_coefficients(
        self, rig_calibration: CalibrationResult
    ) -> None:
        truth = _squat_skeleton(0.6)
        pixels = _project_through_lens(rig_calibration, truth)
        ignored = DLTTriangulator(rig_calibration).triangulate(_multi_view(pixels))
        for camera in rig_calibration.cameras.values():
            camera.distortion_coeffs = LENS_DISTORTION

        undistorted = DLTTriangulator(rig_calibration).triangulate(_multi_view(pixels))

        positions, confidences = _skeleton_arrays(undistorted)
        ignored_positions, _ = _skeleton_arrays(ignored)
        assert np.all(confidences > 0.0)
        assert np.linalg.norm(positions - truth, axis=1).max() < UNDISTORTED_ERROR_LIMIT_M
        assert np.linalg.norm(ignored_positions - truth, axis=1).max() > IGNORED_DISTORTION_MIN_ERROR_M

    def test_invalid_keypoints_in_a_distorted_view_never_produce_nan(self, rig_calibration: CalibrationResult) -> None:
        for camera in rig_calibration.cameras.values():
            camera.distortion_coeffs = LENS_DISTORTION
        truth = _squat_skeleton(0.3)
        pixels = _project_through_lens(rig_calibration, truth)
        pixels[0, CK.LEFT_KNEE] = [np.nan, np.inf]
        confidences = np.ones(pixels.shape[:2])
        confidences[1, CK.RIGHT_ANKLE] = 0.0
        confidences[2] = 0.0

        skeleton = DLTTriangulator(rig_calibration).triangulate(_multi_view(pixels, confidences))

        positions, _ = _skeleton_arrays(skeleton)
        assert np.isfinite(positions).all()
        assert np.linalg.norm(positions[CK.LEFT_HIP] - truth[CK.LEFT_HIP]) < UNDISTORTED_ERROR_LIMIT_M


class TestRecentreAtHips:
    def test_subtracts_hip_midpoint_from_confident_keypoints(self) -> None:
        positions = _squat_skeleton(0.5, offset_m=HIDDEN_HIP_OFFSET_M)
        skeleton = _make_skeleton_3d(positions, np.full(NUM_KEYPOINTS, 0.6))
        centred, _ = _skeleton_arrays(recentre_at_hips(skeleton))
        hip_midpoint = (positions[CK.LEFT_HIP] + positions[CK.RIGHT_HIP]) / 2
        assert np.abs(centred - (positions - hip_midpoint)).max() == pytest.approx(0.0, abs=EXACT_TOLERANCE_M)

    def test_zero_confidence_keypoints_set_to_origin(self) -> None:
        positions = _squat_skeleton(0.5)
        confidences = np.full(NUM_KEYPOINTS, 0.6)
        confidences[CK.LEFT_WRIST] = 0.0
        centred, centred_confidences = _skeleton_arrays(recentre_at_hips(_make_skeleton_3d(positions, confidences)))
        assert centred[CK.LEFT_WRIST].tolist() == [0.0, 0.0, 0.0]
        assert centred_confidences[CK.LEFT_WRIST] == 0.0

    @pytest.mark.parametrize("missing_hip", [CK.LEFT_HIP, CK.RIGHT_HIP])
    def test_missing_hip_returns_none(self, missing_hip: int) -> None:
        confidences = np.full(NUM_KEYPOINTS, 0.6)
        confidences[missing_hip] = 0.0
        assert recentre_at_hips(_make_skeleton_3d(_squat_skeleton(0.5), confidences)) is None

    def test_preserves_confidences_timestamp_and_frame_index(self) -> None:
        confidences = np.linspace(0.1, 0.9, NUM_KEYPOINTS)
        skeleton = _make_skeleton_3d(_squat_skeleton(0.2), confidences, timestamp=3.5, frame_index=11)
        centred_skeleton = recentre_at_hips(skeleton)
        assert centred_skeleton.timestamp == pytest.approx(3.5, abs=EXACT_TOLERANCE_M)
        assert centred_skeleton.frame_index == 11
        assert _skeleton_arrays(centred_skeleton)[1] == pytest.approx(confidences, abs=EXACT_TOLERANCE_M)
