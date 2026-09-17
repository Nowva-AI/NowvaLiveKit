"""
Tests for T-pose calibration: camera pose recovery from synthetic projections of the
T-pose model (world frame X-left, Y-down, +Z = subject's back), per-camera crop
tracking ids, 21-keypoint skeletons, raw-score keypoint gating and T-pose validation.
Also the calibration file schema: lens distortion, keypoint undistortion, saved
per-camera intrinsics and their use by the T-pose solve.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from biomechanics.pose.base import PoseEstimator  # noqa: E402
from biomechanics.triangulation.calibration import (  # noqa: E402
    MIN_MEAN_KEYPOINT_SCORE,
    NUM_DISTORTION_COEFFS,
    SEGMENT_RATIOS,
    WORLD_ANCHOR_BOARD,
    WORLD_ANCHOR_PERSON,
    CalibrationResult,
    CameraCalibration,
    TPoseCalibrator,
    average_detected_keypoints,
    intrinsics_path,
    load_intrinsics,
    load_rig_intrinsics,
    rig_calibration_path,
    save_intrinsics,
    tpose_frame_mask,
    undistort_keypoints,
)
from biomechanics.utils.types import CocoKeypoints as CK  # noqa: E402
from biomechanics.utils.types import Skeleton2D, Skeleton3D  # noqa: E402

HEIGHT_M = 1.885
RESOLUTION = (1280, 720)
FOCAL_LENGTH_FACTOR = 0.8
CAMERA_DISTANCE_M = 4.0
CAMERA_HEIGHT_Y_M = -0.3  # Y-down: camera 0.3 m above the hips
SIDE_CAMERA_YAW_DEG = 40.0
NUM_FRAMES = 10
NUM_SKELETON_KEYPOINTS = 21
DETECTED_SCORE = 0.95
LOW_SCORE = 0.4

# A calibrated lens: focal 10 % below the 0.8 * width guess, off-centre, barrel distortion.
REAL_INTRINSICS = np.array([[921.6, 0.0, 652.0], [0.0, 918.0, 349.0], [0.0, 0.0, 1.0]])
REAL_DISTORTION = np.array([-0.25, 0.08, 0.001, -0.001, 0.0])
INTRINSICS_RMS_PX = 0.21
GUESSED_INTRINSICS_MIN_CENTER_ERROR_M = 0.2
ROUND_TRIP_TOL_PX = 0.1

CAMERA_CENTER_TOL_M = 0.01
ROTATION_TOL = 1e-3
PIXEL_TOL = 1e-6
REPROJECTION_TOL_PX = 0.1


def _look_at_camera(yaw_deg: float) -> tuple[np.ndarray, np.ndarray]:
    # Camera on a circle around the hips; yaw 0 = front camera at -Z looking toward +Z.
    yaw_rad = np.radians(yaw_deg)
    center = np.array([CAMERA_DISTANCE_M * np.sin(yaw_rad), CAMERA_HEIGHT_Y_M, -CAMERA_DISTANCE_M * np.cos(yaw_rad)])
    target = np.array([0.0, CAMERA_HEIGHT_Y_M, 0.0])
    z_axis = (target - center) / np.linalg.norm(target - center)
    x_axis = np.cross(np.array([0.0, 1.0, 0.0]), z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rotation = np.stack([x_axis, y_axis, z_axis])
    return rotation, center


def _intrinsics() -> np.ndarray:
    focal_px = FOCAL_LENGTH_FACTOR * RESOLUTION[0]
    return np.array([[focal_px, 0.0, RESOLUTION[0] / 2.0], [0.0, focal_px, RESOLUTION[1] / 2.0], [0.0, 0.0, 1.0]])


def _project(model_3d: np.ndarray, rotation: np.ndarray, center: np.ndarray) -> np.ndarray:
    camera_points = (model_3d - center) @ rotation.T
    image_points = camera_points @ _intrinsics().T
    return image_points[:, :2] / image_points[:, 2:3]


def _project_through_lens(model_3d: np.ndarray, rotation: np.ndarray, center: np.ndarray) -> np.ndarray:
    image_points, _ = cv2.projectPoints(
        model_3d, cv2.Rodrigues(rotation)[0], -rotation @ center, REAL_INTRINSICS, REAL_DISTORTION
    )
    return image_points.reshape(-1, 2)


def _arms_down_model(height_m: float) -> np.ndarray:
    model = TPoseCalibrator(pose_estimator=None).build_tpose_model(height_m)
    arm_length_m = (SEGMENT_RATIOS["upper_arm"] + SEGMENT_RATIOS["forearm"]) * height_m
    for shoulder, elbow, wrist in [
        (CK.LEFT_SHOULDER, CK.LEFT_ELBOW, CK.LEFT_WRIST),
        (CK.RIGHT_SHOULDER, CK.RIGHT_ELBOW, CK.RIGHT_WRIST),
    ]:
        model[elbow] = model[shoulder] + [0.0, SEGMENT_RATIOS["upper_arm"] * height_m, 0.0]
        model[wrist] = model[shoulder] + [0.0, arm_length_m, 0.0]
    return model


def _skeleton_array(image_points: np.ndarray, scores: np.ndarray) -> np.ndarray:
    # 17 projected keypoints -> 21-entry skeleton array, foot keypoints 17-20 empty.
    keypoints = np.zeros((NUM_SKELETON_KEYPOINTS, 3))
    keypoints[:17, :2] = image_points
    keypoints[:17, 2] = scores
    keypoints[keypoints[:, 2] == 0.0, :2] = 0.0
    return keypoints


class _ScriptedEstimator(PoseEstimator):
    """Returns pre-built skeleton arrays for each (camera_id, frame number) and records the ids."""

    def __init__(self, skeletons_by_camera: dict[str, list[np.ndarray | None]]) -> None:
        super().__init__(confidence_threshold=0.3)
        self._skeletons_by_camera = skeletons_by_camera
        self._next_frame: dict[str, int] = {}
        self.camera_ids_seen: list[str] = []

    def estimate(self, frame: np.ndarray, camera_id: int | str = 0) -> Skeleton2D | None:
        self.camera_ids_seen.append(camera_id)
        frame_number = self._next_frame.get(camera_id, 0)
        self._next_frame[camera_id] = frame_number + 1
        keypoints = self._skeletons_by_camera[camera_id][frame_number]
        return None if keypoints is None else Skeleton2D.from_numpy(keypoints)

    def estimate_3d(self, frame: np.ndarray) -> Skeleton3D | None:
        return None


def _frames() -> list[np.ndarray]:
    return [np.zeros((RESOLUTION[1], RESOLUTION[0], 3), dtype=np.uint8) for _ in range(NUM_FRAMES)]


def _scripted_views(
    model_3d: np.ndarray, yaws_deg: dict[str, float], through_real_lens: bool = False
) -> dict[str, list[np.ndarray | None]]:
    views = {}
    for camera_id, yaw_deg in yaws_deg.items():
        rotation, center = _look_at_camera(yaw_deg)
        project = _project_through_lens if through_real_lens else _project
        image_points = project(model_3d, rotation, center)
        views[camera_id] = [_skeleton_array(image_points, np.full(17, DETECTED_SCORE)) for _ in range(NUM_FRAMES)]
    return views


def _calibrate(
    views: dict[str, list[np.ndarray | None]],
    intrinsics: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[CalibrationResult, _ScriptedEstimator]:
    estimator = _ScriptedEstimator(views)
    calibrator = TPoseCalibrator(
        pose_estimator=estimator, focal_length_factor=FOCAL_LENGTH_FACTOR, intrinsics=intrinsics
    )
    frames = {camera_id: _frames() for camera_id in views}
    return calibrator.calibrate(frames, HEIGHT_M, RESOLUTION), estimator


def _lens_camera(distortion_coeffs: np.ndarray = REAL_DISTORTION) -> CameraCalibration:
    return CameraCalibration(
        camera_id="0",
        projection_matrix=REAL_INTRINSICS @ np.hstack([np.eye(3), np.zeros((3, 1))]),
        intrinsic_matrix=REAL_INTRINSICS,
        rotation_matrix=np.eye(3),
        translation_vector=np.zeros((3, 1)),
        reprojection_error=0.0,
        resolution=RESOLUTION,
        distortion_coeffs=distortion_coeffs,
    )


def _camera_center(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return (-rotation.T @ translation.reshape(3)).reshape(3)


@pytest.fixture
def tpose_model() -> np.ndarray:
    return TPoseCalibrator(pose_estimator=None).build_tpose_model(HEIGHT_M)


class TestTPoseModelAxes:

    def test_left_side_is_positive_x_and_feet_are_positive_y(self, tpose_model: np.ndarray) -> None:
        assert tpose_model[CK.LEFT_WRIST, 0] > 0.0 > tpose_model[CK.RIGHT_WRIST, 0]
        assert tpose_model[CK.LEFT_ANKLE, 1] > 0.0 > tpose_model[CK.NOSE, 1]
        assert np.all(tpose_model[:, 2] == 0.0)


class TestCameraRecovery:

    def test_front_camera_recovered_at_negative_z(self, tpose_model: np.ndarray) -> None:
        result, _ = _calibrate(_scripted_views(tpose_model, {"0": 0.0}))

        camera = result.cameras["0"]
        expected_rotation, expected_center = _look_at_camera(0.0)
        center = _camera_center(camera.rotation_matrix, camera.translation_vector)
        assert center == pytest.approx(expected_center, abs=CAMERA_CENTER_TOL_M)
        assert center[2] < 0.0
        assert camera.rotation_matrix == pytest.approx(expected_rotation, abs=ROTATION_TOL)
        assert camera.reprojection_error == pytest.approx(0.0, abs=REPROJECTION_TOL_PX)

    def test_side_cameras_recovered(self, tpose_model: np.ndarray) -> None:
        yaws = {"0": 0.0, "1": -SIDE_CAMERA_YAW_DEG, "2": SIDE_CAMERA_YAW_DEG}

        result, _ = _calibrate(_scripted_views(tpose_model, yaws))

        for camera_id, yaw_deg in yaws.items():
            camera = result.cameras[camera_id]
            _, expected_center = _look_at_camera(yaw_deg)
            center = _camera_center(camera.rotation_matrix, camera.translation_vector)
            assert center == pytest.approx(expected_center, abs=CAMERA_CENTER_TOL_M)

    def test_each_camera_id_passed_to_estimator(self, tpose_model: np.ndarray) -> None:
        _, estimator = _calibrate(_scripted_views(tpose_model, {"0": 0.0, "1": SIDE_CAMERA_YAW_DEG}))

        assert estimator.camera_ids_seen == ["0"] * NUM_FRAMES + ["1"] * NUM_FRAMES

    def test_missed_keypoints_do_not_pull_average_toward_origin(self, tpose_model: np.ndarray) -> None:
        views = _scripted_views(tpose_model, {"0": 0.0})
        for frame_number in range(0, NUM_FRAMES, 3):
            views["0"][frame_number][CK.LEFT_KNEE] = 0.0
        views["0"][1] = None

        result, _ = _calibrate(views)

        _, expected_center = _look_at_camera(0.0)
        camera = result.cameras["0"]
        center = _camera_center(camera.rotation_matrix, camera.translation_vector)
        assert center == pytest.approx(expected_center, abs=CAMERA_CENTER_TOL_M)
        assert camera.reprojection_error == pytest.approx(0.0, abs=REPROJECTION_TOL_PX)

    def test_camera_without_person_skipped(self, tpose_model: np.ndarray) -> None:
        views = _scripted_views(tpose_model, {"0": 0.0})
        views["1"] = [None] * NUM_FRAMES

        result, _ = _calibrate(views)

        assert set(result.cameras) == {"0"}

    def test_no_camera_with_person_raises_runtime_error(self) -> None:
        with pytest.raises(RuntimeError):
            _calibrate({"0": [None] * NUM_FRAMES})


class TestTPoseValidation:

    def test_arms_down_raises_value_error(self) -> None:
        views = _scripted_views(_arms_down_model(HEIGHT_M), {"0": 0.0})

        with pytest.raises(ValueError, match="T-pose"):
            _calibrate(views)

    def test_tpose_in_under_half_of_frames_raises_value_error(self, tpose_model: np.ndarray) -> None:
        views = _scripted_views(tpose_model, {"0": 0.0})
        arms_down = _scripted_views(_arms_down_model(HEIGHT_M), {"0": 0.0})
        views["0"][: NUM_FRAMES // 2 + 1] = arms_down["0"][: NUM_FRAMES // 2 + 1]

        with pytest.raises(ValueError, match="T-pose"):
            _calibrate(views)

    def test_mask_accepts_side_view_tpose_and_rejects_arms_down(self, tpose_model: np.ndarray) -> None:
        rotation, center = _look_at_camera(SIDE_CAMERA_YAW_DEG)
        tpose_points = _project(tpose_model, rotation, center)
        arms_down_points = _project(_arms_down_model(HEIGHT_M), rotation, center)
        points_xy = np.stack([tpose_points, arms_down_points])
        scores = np.full((2, 17), DETECTED_SCORE)

        mask = tpose_frame_mask(points_xy, scores)

        assert mask.tolist() == [True, False]

    def test_mask_rejects_frame_missing_a_wrist(self, tpose_model: np.ndarray) -> None:
        rotation, center = _look_at_camera(0.0)
        points_xy = _project(tpose_model, rotation, center)[None]
        scores = np.full((1, 17), DETECTED_SCORE)
        scores[0, CK.RIGHT_WRIST] = 0.0

        assert tpose_frame_mask(points_xy, scores).tolist() == [False]


class TestKeypointAveraging:

    def test_average_ignores_missed_frames_and_scores_count_misses(self) -> None:
        points_xy = np.array([[[10.0, 20.0]], [[0.0, 0.0]], [[30.0, 40.0]]])
        scores = np.array([[0.9], [0.0], [0.9]])

        mean_xy, mean_scores = average_detected_keypoints(points_xy, scores)

        assert mean_xy[0] == pytest.approx([20.0, 30.0], abs=PIXEL_TOL)
        assert mean_scores[0] == pytest.approx(0.6, abs=PIXEL_TOL)

    def test_keypoints_detected_in_few_frames_fail_score_gate(self) -> None:
        scores = np.array([[DETECTED_SCORE], [0.0], [0.0]])

        _, mean_scores = average_detected_keypoints(np.zeros((3, 1, 2)), scores)

        assert mean_scores[0] < MIN_MEAN_KEYPOINT_SCORE

    def test_low_raw_scores_fail_score_gate(self) -> None:
        scores = np.full((NUM_FRAMES, 1), LOW_SCORE)

        _, mean_scores = average_detected_keypoints(np.zeros((NUM_FRAMES, 1, 2)), scores)

        assert mean_scores[0] < MIN_MEAN_KEYPOINT_SCORE


class TestRealIntrinsics:

    def test_calibrated_lens_recovers_the_camera_and_is_stored(self, tpose_model: np.ndarray) -> None:
        yaws = {"0": 0.0, "1": SIDE_CAMERA_YAW_DEG}
        views = _scripted_views(tpose_model, yaws, through_real_lens=True)
        intrinsics = {camera_id: (REAL_INTRINSICS, REAL_DISTORTION) for camera_id in yaws}

        result, _ = _calibrate(views, intrinsics)

        for camera_id, yaw_deg in yaws.items():
            camera = result.cameras[camera_id]
            _, expected_center = _look_at_camera(yaw_deg)
            center = _camera_center(camera.rotation_matrix, camera.translation_vector)
            assert center == pytest.approx(expected_center, abs=CAMERA_CENTER_TOL_M)
            assert camera.reprojection_error == pytest.approx(0.0, abs=REPROJECTION_TOL_PX)
            assert camera.intrinsic_matrix == pytest.approx(REAL_INTRINSICS)
            assert camera.distortion_coeffs == pytest.approx(REAL_DISTORTION)
            expected_projection = REAL_INTRINSICS @ np.hstack([camera.rotation_matrix, camera.translation_vector])
            assert camera.projection_matrix == pytest.approx(expected_projection, abs=PIXEL_TOL)

    def test_guessed_intrinsics_misplace_a_real_lens_and_warn(
        self, tpose_model: np.ndarray, caplog: pytest.LogCaptureFixture
    ) -> None:
        views = _scripted_views(tpose_model, {"0": 0.0}, through_real_lens=True)

        with caplog.at_level(logging.WARNING):
            result, _ = _calibrate(views)

        camera = result.cameras["0"]
        _, expected_center = _look_at_camera(0.0)
        center = _camera_center(camera.rotation_matrix, camera.translation_vector)
        assert np.linalg.norm(center - expected_center) > GUESSED_INTRINSICS_MIN_CENTER_ERROR_M
        assert not np.any(camera.distortion_coeffs)
        assert "no calibrated intrinsics" in caplog.text

    def test_only_cameras_without_intrinsics_fall_back(
        self, tpose_model: np.ndarray, caplog: pytest.LogCaptureFixture
    ) -> None:
        views = _scripted_views(tpose_model, {"0": 0.0}, through_real_lens=True)
        views.update(_scripted_views(tpose_model, {"1": SIDE_CAMERA_YAW_DEG}))

        with caplog.at_level(logging.WARNING):
            result, _ = _calibrate(views, {"0": (REAL_INTRINSICS, REAL_DISTORTION)})

        assert result.cameras["0"].intrinsic_matrix == pytest.approx(REAL_INTRINSICS)
        assert result.cameras["1"].intrinsic_matrix == pytest.approx(_intrinsics())
        warned = [record.getMessage() for record in caplog.records if "no calibrated intrinsics" in record.getMessage()]
        assert len(warned) == 1 and "Camera 1" in warned[0]


class TestUndistortKeypoints:

    def test_round_trip_through_a_distorted_lens_below_a_tenth_of_a_pixel(self) -> None:
        rng = np.random.default_rng(0)
        points_3d = np.column_stack([rng.uniform(-1.2, 1.2, 300), rng.uniform(-0.7, 0.7, 300), rng.uniform(1.8, 4.0, 300)])
        distorted, _ = cv2.projectPoints(points_3d, np.zeros(3), np.zeros(3), REAL_INTRINSICS, REAL_DISTORTION)
        pinhole, _ = cv2.projectPoints(points_3d, np.zeros(3), np.zeros(3), REAL_INTRINSICS, np.zeros(NUM_DISTORTION_COEFFS))
        distorted, pinhole = distorted.reshape(-1, 2), pinhole.reshape(-1, 2)

        undistorted = undistort_keypoints(distorted, _lens_camera())

        assert np.abs(distorted - pinhole).max() > 10.0
        assert np.abs(undistorted - pinhole).max() < ROUND_TRIP_TOL_PX

    def test_zero_coefficients_return_an_unchanged_copy(self) -> None:
        points_px = np.array([[10.0, 20.0], [1270.0, 700.0]])

        undistorted = undistort_keypoints(points_px, _lens_camera(np.zeros(NUM_DISTORTION_COEFFS)))

        assert undistorted == pytest.approx(points_px, abs=PIXEL_TOL)
        assert undistorted is not points_px

    def test_no_points_gives_no_points(self) -> None:
        assert undistort_keypoints(np.zeros((0, 2)), _lens_camera()).shape == (0, 2)


class TestCalibrationFileSchema:

    def test_distortion_and_world_anchor_survive_save_and_load(self, tmp_path: Path) -> None:
        result = CalibrationResult(cameras={"0": _lens_camera()}, athlete_height_m=HEIGHT_M, world_anchor=WORLD_ANCHOR_BOARD)
        path = str(tmp_path / "rig.json")

        TPoseCalibrator.save_calibration(result, path)
        loaded = TPoseCalibrator.load_calibration(path)

        assert loaded.world_anchor == WORLD_ANCHOR_BOARD
        assert loaded.cameras["0"].distortion_coeffs == pytest.approx(REAL_DISTORTION)
        assert loaded.cameras["0"].intrinsic_matrix == pytest.approx(REAL_INTRINSICS)

    def test_file_from_before_distortion_loads_with_zero_coefficients(self, tmp_path: Path) -> None:
        path = tmp_path / "old_rig.json"
        TPoseCalibrator.save_calibration(CalibrationResult(cameras={"0": _lens_camera()}), str(path))
        data = json.loads(path.read_text())
        del data["world_anchor"]
        del data["cameras"]["0"]["distortion_coeffs"]
        path.write_text(json.dumps(data))

        loaded = TPoseCalibrator.load_calibration(str(path))

        assert loaded.world_anchor == WORLD_ANCHOR_PERSON
        assert loaded.cameras["0"].distortion_coeffs == pytest.approx(np.zeros(NUM_DISTORTION_COEFFS))

    def test_tpose_result_is_anchored_to_the_person(self, tpose_model: np.ndarray) -> None:
        result, _ = _calibrate(_scripted_views(tpose_model, {"0": 0.0}))

        assert result.world_anchor == WORLD_ANCHOR_PERSON


class TestIntrinsicsStorage:

    def test_saved_intrinsics_load_back_with_the_contract_keys(self, tmp_path: Path) -> None:
        path = save_intrinsics("left_usb", RESOLUTION, REAL_INTRINSICS, REAL_DISTORTION, INTRINSICS_RMS_PX, tmp_path)

        K, distortion_coeffs = load_intrinsics("left_usb", RESOLUTION, tmp_path)

        assert path == tmp_path / "intrinsics_left_usb.json"
        assert set(json.loads(path.read_text())) == {
            "camera_key", "resolution", "intrinsic_matrix", "distortion_coeffs", "rms_reprojection_px", "timestamp",
        }
        assert K == pytest.approx(REAL_INTRINSICS)
        assert distortion_coeffs == pytest.approx(REAL_DISTORTION)

    def test_missing_file_gives_none(self, tmp_path: Path) -> None:
        assert load_intrinsics("never_calibrated", RESOLUTION, tmp_path) is None

    def test_other_capture_resolution_gives_none_and_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        save_intrinsics("0", (1920, 1080), REAL_INTRINSICS, REAL_DISTORTION, INTRINSICS_RMS_PX, tmp_path)

        with caplog.at_level(logging.WARNING):
            loaded = load_intrinsics("0", RESOLUTION, tmp_path)

        assert loaded is None
        assert "calibrated at" in caplog.text

    def test_rig_intrinsics_are_keyed_by_camera_id_and_skip_uncalibrated_cameras(self, tmp_path: Path) -> None:
        save_intrinsics("macbook", RESOLUTION, REAL_INTRINSICS, REAL_DISTORTION, INTRINSICS_RMS_PX, tmp_path)

        intrinsics = load_rig_intrinsics({"0": "macbook", "1": "1"}, RESOLUTION, tmp_path)

        assert set(intrinsics) == {"0"}
        assert intrinsics["0"][0] == pytest.approx(REAL_INTRINSICS)

    def test_paths_follow_the_nowva_naming(self, tmp_path: Path) -> None:
        assert intrinsics_path("2").name == "intrinsics_2.json"
        assert intrinsics_path("2").parent == Path.home() / ".nowva"
        assert rig_calibration_path([0, 1, 2], tmp_path) == tmp_path / "rig_calibration_cams_0-1-2.json"
