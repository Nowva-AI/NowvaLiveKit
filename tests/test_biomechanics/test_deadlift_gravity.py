"""
Tests for the deadlift's measured gravity: the up normal of a ChArUco board lying flat
on the floor in each camera's frame (synthetic distorted views of the factory board),
the per-camera gravity files, and the mapping of every camera's up into one world up
with moved cameras dropped and the body vertical as the fallback.

The synthetic floor is tilted 2.5 deg from world Y (the body-derived vertical can be
off by 2-4 deg, PLAN.md §2.1), so a measured up must differ from WORLD_UP.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from biomechanics.deadlift.gravity import (  # noqa: E402
    CameraGravity,
    board_up_in_camera,
    gravity_path,
    load_camera_gravity,
    load_rig_gravity,
    load_world_up_for_provider,
    measure_camera_gravity,
    save_camera_gravity,
    world_up_from_cameras,
)
from biomechanics.deadlift.types import GRAVITY_SOURCE_BODY, GRAVITY_SOURCE_MEASURED  # noqa: E402
from biomechanics.pose.multi_camera import MultiCameraPoseProvider  # noqa: E402
from biomechanics.triangulation.calibration import CalibrationResult, CameraCalibration  # noqa: E402
from biomechanics.triangulation.charuco import CharucoBoardSpec, CharucoDetection  # noqa: E402
from biomechanics.utils.geometry import WORLD_UP  # noqa: E402

FACTORY_BOARD = CharucoBoardSpec(square_length_m=0.12, marker_length_m=0.09)
FLOOR_Y_M = 0.98
FLOOR_TILT_DEG = 2.5
TRUE_UP = cv2.Rodrigues(np.array([math.radians(FLOOR_TILT_DEG), 0.0, 0.0]))[0] @ WORLD_UP
BOARD_CENTRE_M = np.array([0.0, FLOOR_Y_M, -0.4])

INTRINSICS = np.array([[1005.0, 0.0, 642.0], [0.0, 1001.0, 357.0], [0.0, 0.0, 1.0]])
LENS_DISTORTION = np.array([-0.11, 0.06, 0.0007, -0.0005, 0.0])
# camera id -> (centre, roll about the optical axis in deg): head-on and two 45 deg uprights.
CAMERAS = {
    "0": (np.array([0.0, FLOOR_Y_M - 1.0, -3.0]), 0.0),
    "1": (np.array([1.84, FLOOR_Y_M - 1.4, -1.84]), 3.0),
    "2": (np.array([-1.84, FLOOR_Y_M - 1.4, -1.84]), -2.0),
}
PROVIDER_CAMERA_KEYS = {"0": "front", "1": "1", "2": "2"}
CORNER_NOISE_PX = 0.15
NUM_BOARD_FRAMES = 20

MEASURED_UP_TOL_DEG = 0.1
EXACT_TOL_DEG = 1e-6
GOOD_NORMAL_SIGMA_DEG = 0.3


def _angle_deg(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    cosine = float(vector_a @ vector_b) / float(np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    return math.degrees(math.acos(np.clip(cosine, -1.0, 1.0)))


def _rotation_about(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis) * math.radians(angle_deg))[0]


def _camera_pose(camera_id: str) -> tuple[np.ndarray, np.ndarray]:
    # World -> camera (R, t), looking at the board with image-down along world +Y, then rolled.
    centre, roll_deg = CAMERAS[camera_id]
    z_axis = (BOARD_CENTRE_M - centre) / np.linalg.norm(BOARD_CENTRE_M - centre)
    x_axis = np.cross([0.0, 1.0, 0.0], z_axis)
    x_axis /= np.linalg.norm(x_axis)
    rotation = _rotation_about([0.0, 0.0, 1.0], roll_deg) @ np.stack([x_axis, np.cross(z_axis, x_axis), z_axis])
    return rotation, -rotation @ centre


def _board_pose_on_floor(board_z_up: bool) -> tuple[np.ndarray, np.ndarray]:
    # Board -> world (R, origin) for the board lying on the tilted floor, centred on BOARD_CENTRE_M.
    # A printed ChArUco frame has Z into the board, i.e. down when it lies face up.
    z_axis = TRUE_UP if board_z_up else -TRUE_UP
    x_axis = np.array([1.0, 0.0, 0.0]) - (TRUE_UP @ [1.0, 0.0, 0.0]) * TRUE_UP
    x_axis /= np.linalg.norm(x_axis)
    rotation = np.column_stack([x_axis, np.cross(z_axis, x_axis), z_axis])
    return rotation, BOARD_CENTRE_M - rotation @ FACTORY_BOARD.centre_m()


def _board_detections(
    camera_id: str, rng: np.random.Generator, board_z_up: bool = False, num_frames: int = NUM_BOARD_FRAMES
) -> list[CharucoDetection]:
    camera_rotation, camera_translation = _camera_pose(camera_id)
    board_rotation, board_origin = _board_pose_on_floor(board_z_up)
    corners_m = FACTORY_BOARD.corner_positions_m()
    projected, _ = cv2.projectPoints(
        corners_m,
        cv2.Rodrigues(camera_rotation @ board_rotation)[0],
        camera_rotation @ board_origin + camera_translation,
        INTRINSICS,
        LENS_DISTORTION,
    )
    projected = projected.reshape(-1, 2)
    assert ((projected >= 0.0) & (projected < (1280, 720))).all(), "board corner outside the image"
    corner_ids = np.arange(len(corners_m))
    return [
        CharucoDetection(corners_px=projected + rng.normal(0.0, CORNER_NOISE_PX, projected.shape), corner_ids=corner_ids)
        for _ in range(num_frames)
    ]


def _camera_gravity(camera_key: str, up_camera: np.ndarray) -> CameraGravity:
    return CameraGravity(
        camera_key=camera_key,
        up_camera=tuple(up_camera.tolist()),
        timestamp="2026-10-03T09:00:00",
        board_id="charuco_7x5_120.0mm_90.0mm_DICT_5X5_100",
        residual_px=0.12,
        normal_sigma_deg=0.05,
        num_frames=30,
        num_corners=24,
    )


def _rig_calibration() -> CalibrationResult:
    calibration = CalibrationResult()
    for camera_id in CAMERAS:
        rotation, translation = _camera_pose(camera_id)
        calibration.cameras[camera_id] = CameraCalibration(
            camera_id=camera_id,
            projection_matrix=INTRINSICS @ np.hstack([rotation, translation[:, None]]),
            intrinsic_matrix=INTRINSICS,
            rotation_matrix=rotation,
            translation_vector=translation[:, None],
            reprojection_error=0.0,
            resolution=(1280, 720),
            distortion_coeffs=LENS_DISTORTION,
        )
    return calibration


@pytest.fixture
def camera_rotations() -> dict[str, np.ndarray]:
    return {camera_id: _camera_pose(camera_id)[0] for camera_id in CAMERAS}


@pytest.fixture
def provider() -> MultiCameraPoseProvider:
    # Camera 0's files are named after a key, as with camera_calibration.camera_keys.
    calibrated = MultiCameraPoseProvider(
        device_ids=[0, 1, 2], camera_keys={int(camera_id): key for camera_id, key in PROVIDER_CAMERA_KEYS.items()}
    )
    calibrated.install_calibration(_rig_calibration())
    return calibrated


@pytest.fixture
def exact_camera_ups(camera_rotations: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {camera_id: rotation @ TRUE_UP for camera_id, rotation in camera_rotations.items()}


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(11)


class TestBoardUpInCamera:
    @pytest.mark.parametrize("board_z_up", [True, False])
    def test_normal_points_up_whichever_way_the_board_frame_faces(self, board_z_up):
        camera_rotation, camera_translation = _camera_pose("0")
        board_rotation, board_origin = _board_pose_on_floor(board_z_up)
        up = board_up_in_camera(
            camera_rotation @ board_rotation,
            camera_rotation @ board_origin + camera_translation,
            FACTORY_BOARD.centre_m(),
        )
        assert _angle_deg(up, camera_rotation @ TRUE_UP) == pytest.approx(0.0, abs=EXACT_TOL_DEG)


class TestMeasureCameraGravity:
    @pytest.mark.parametrize("camera_id", sorted(CAMERAS))
    def test_flat_board_gives_up_in_the_camera_frame(self, camera_id, camera_rotations, rng):
        gravity = measure_camera_gravity(_board_detections(camera_id, rng), FACTORY_BOARD, INTRINSICS, LENS_DISTORTION, camera_id)
        assert gravity is not None
        assert _angle_deg(np.array(gravity.up_camera), camera_rotations[camera_id] @ TRUE_UP) <= MEASURED_UP_TOL_DEG
        assert gravity.normal_sigma_deg < GOOD_NORMAL_SIGMA_DEG
        assert gravity.residual_px < 3.0 * CORNER_NOISE_PX
        assert (gravity.num_frames, gravity.num_corners) == (NUM_BOARD_FRAMES, len(FACTORY_BOARD.corner_positions_m()))
        assert gravity.board_id == "charuco_7x5_120.0mm_90.0mm_DICT_5X5_100"

    def test_face_up_board_frame_gives_the_same_up(self, camera_rotations, rng):
        gravity = measure_camera_gravity(
            _board_detections("1", rng, board_z_up=True), FACTORY_BOARD, INTRINSICS, LENS_DISTORTION, "1"
        )
        assert _angle_deg(np.array(gravity.up_camera), camera_rotations["1"] @ TRUE_UP) <= MEASURED_UP_TOL_DEG

    def test_too_few_corners_returns_none(self, rng):
        detections = _board_detections("0", rng, num_frames=1)
        few = [CharucoDetection(corners_px=detections[0].corners_px[:5], corner_ids=detections[0].corner_ids[:5])]
        assert measure_camera_gravity(few, FACTORY_BOARD, INTRINSICS, LENS_DISTORTION, "0") is None

    def test_measured_cameras_agree_on_the_world_up(self, camera_rotations, rng):
        per_camera_up = {
            camera_id: np.array(
                measure_camera_gravity(_board_detections(camera_id, rng), FACTORY_BOARD, INTRINSICS, LENS_DISTORTION, camera_id).up_camera
            )
            for camera_id in CAMERAS
        }
        up_world, used, source = world_up_from_cameras(per_camera_up, camera_rotations)
        assert source == GRAVITY_SOURCE_MEASURED
        assert used == sorted(CAMERAS)
        assert _angle_deg(up_world, TRUE_UP) <= MEASURED_UP_TOL_DEG
        assert _angle_deg(up_world, WORLD_UP) == pytest.approx(FLOOR_TILT_DEG, abs=MEASURED_UP_TOL_DEG)


class TestGravityFiles:
    def test_save_and_load_round_trip(self, tmp_path):
        gravity = _camera_gravity("front_cam", np.array([0.0, -0.96, 0.28]))
        path = save_camera_gravity(gravity, tmp_path)
        assert path == tmp_path / "gravity_front_cam.json" == gravity_path("front_cam", tmp_path)
        assert load_camera_gravity("front_cam", tmp_path) == gravity

    def test_missing_file_loads_none(self, tmp_path):
        assert load_camera_gravity("0", tmp_path) is None

    def test_rig_gravity_is_keyed_by_camera_id(self, tmp_path):
        save_camera_gravity(_camera_gravity("front_cam", np.array([0.0, -1.0, 0.0])), tmp_path)
        per_camera_up = load_rig_gravity({"0": "front_cam", "1": "1"}, tmp_path)
        assert list(per_camera_up) == ["0"]
        assert per_camera_up["0"] == pytest.approx(np.array([0.0, -1.0, 0.0]))

    def test_no_gravity_files_fall_back_to_body(self, tmp_path, camera_rotations):
        per_camera_up = load_rig_gravity({camera_id: camera_id for camera_id in CAMERAS}, tmp_path)
        up_world, used, source = world_up_from_cameras(per_camera_up, camera_rotations)
        assert source == GRAVITY_SOURCE_BODY
        assert used == []
        assert up_world == pytest.approx(WORLD_UP)


class TestWorldUpFromCameras:
    def test_rotated_cameras_map_to_one_world_up(self, exact_camera_ups, camera_rotations):
        up_world, used, source = world_up_from_cameras(exact_camera_ups, camera_rotations)
        assert source == GRAVITY_SOURCE_MEASURED
        assert used == ["0", "1", "2"]
        assert _angle_deg(up_world, TRUE_UP) == pytest.approx(0.0, abs=EXACT_TOL_DEG)
        assert np.linalg.norm(up_world) == pytest.approx(1.0)

    @pytest.mark.parametrize("moved_deg", [2.0, 1.4])
    def test_moved_camera_is_dropped(self, exact_camera_ups, camera_rotations, moved_deg):
        # The camera turned after its gravity file was written: its stored up no longer
        # maps to the floor normal through the current calibration rotation. 1.4 deg is
        # within 1 deg of the plain mean of all three, so a mean consensus would keep it.
        moved = dict(exact_camera_ups)
        moved["2"] = _rotation_about([1.0, 0.0, 0.3], moved_deg) @ exact_camera_ups["2"]
        up_world, used, source = world_up_from_cameras(moved, camera_rotations)
        assert source == GRAVITY_SOURCE_MEASURED
        assert used == ["0", "1"]
        assert _angle_deg(up_world, TRUE_UP) == pytest.approx(0.0, abs=EXACT_TOL_DEG)

    def test_two_disagreeing_cameras_fall_back_to_body(self, exact_camera_ups, camera_rotations):
        pair = {"0": exact_camera_ups["0"], "1": _rotation_about([1.0, 0.0, 0.0], 2.0) @ exact_camera_ups["1"]}
        up_world, used, source = world_up_from_cameras(pair, camera_rotations)
        assert (source, used) == (GRAVITY_SOURCE_BODY, [])
        assert up_world == pytest.approx(WORLD_UP)

    def test_single_camera_is_measured(self, exact_camera_ups, camera_rotations):
        up_world, used, source = world_up_from_cameras({"1": exact_camera_ups["1"]}, camera_rotations)
        assert (source, used) == (GRAVITY_SOURCE_MEASURED, ["1"])
        assert _angle_deg(up_world, TRUE_UP) == pytest.approx(0.0, abs=EXACT_TOL_DEG)

    def test_camera_without_calibration_is_ignored(self, exact_camera_ups, camera_rotations):
        rotations = {camera_id: camera_rotations[camera_id] for camera_id in ("0", "1")}
        up_world, used, source = world_up_from_cameras(exact_camera_ups, rotations)
        assert (source, used) == (GRAVITY_SOURCE_MEASURED, ["0", "1"])

    def test_non_unit_camera_up_is_normalised(self, exact_camera_ups, camera_rotations):
        scaled = {camera_id: 3.0 * up for camera_id, up in exact_camera_ups.items()}
        up_world, _, _ = world_up_from_cameras(scaled, camera_rotations)
        assert np.linalg.norm(up_world) == pytest.approx(1.0)
        assert _angle_deg(up_world, TRUE_UP) == pytest.approx(0.0, abs=EXACT_TOL_DEG)


class TestLoadWorldUpForProvider:
    def test_measured_through_the_provider_camera_keys(self, tmp_path, provider, exact_camera_ups):
        for camera_id, camera_key in PROVIDER_CAMERA_KEYS.items():
            save_camera_gravity(_camera_gravity(camera_key, exact_camera_ups[camera_id]), tmp_path)
        up_world, source = load_world_up_for_provider(provider, tmp_path)
        assert source == GRAVITY_SOURCE_MEASURED
        assert _angle_deg(up_world, TRUE_UP) == pytest.approx(0.0, abs=EXACT_TOL_DEG)

    def test_no_calibration_is_body(self, tmp_path):
        assert load_world_up_for_provider(MultiCameraPoseProvider(device_ids=[0, 1, 2]), tmp_path) == (
            None, GRAVITY_SOURCE_BODY,
        )

    def test_missing_directory_is_body(self, tmp_path, provider):
        assert load_world_up_for_provider(provider, tmp_path / "absent") == (None, GRAVITY_SOURCE_BODY)

    def test_unreadable_file_is_skipped(self, tmp_path, provider, exact_camera_ups):
        for camera_id, camera_key in PROVIDER_CAMERA_KEYS.items():
            save_camera_gravity(_camera_gravity(camera_key, exact_camera_ups[camera_id]), tmp_path)
        gravity_path(PROVIDER_CAMERA_KEYS["2"], tmp_path).write_text("{not json")
        up_world, source = load_world_up_for_provider(provider, tmp_path)
        assert source == GRAVITY_SOURCE_MEASURED
        assert _angle_deg(up_world, TRUE_UP) == pytest.approx(0.0, abs=EXACT_TOL_DEG)

    def test_only_bad_files_is_body(self, tmp_path, provider):
        gravity_path(PROVIDER_CAMERA_KEYS["0"], tmp_path).write_text("[1, 2, 3]")
        gravity_path(PROVIDER_CAMERA_KEYS["1"], tmp_path).write_text(
            _camera_gravity("1", np.array([0.0, -1.0, 0.0])).model_dump_json().replace("-1.0", "-2.0")
        )
        gravity_path(PROVIDER_CAMERA_KEYS["2"], tmp_path).write_text("{not json")
        assert load_world_up_for_provider(provider, tmp_path) == (None, GRAVITY_SOURCE_BODY)

    def test_non_unit_up_vector_is_invalid(self):
        with pytest.raises(ValueError):
            _camera_gravity("0", np.array([0.0, -2.0, 0.0]))
