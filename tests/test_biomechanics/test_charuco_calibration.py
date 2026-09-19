"""
Tests for ChArUco calibration: board rendering, corner detection in the pixel-centre
convention, intrinsics recovery from rendered distorted views, and the factory rig
extrinsics (relative camera geometry, board-anchored world frame, inconsistent placements).

Rendered views map every output pixel back through the true lens model onto the board
texture, so detections can be compared with cv2.projectPoints directly.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from biomechanics.triangulation.calibration import (  # noqa: E402
    WORLD_ANCHOR_BOARD,
    CalibrationResult,
    CameraCalibration,
    TPoseCalibrator,
    load_intrinsics,
    save_intrinsics,
    undistort_keypoints,
)
from biomechanics.triangulation.charuco import (  # noqa: E402
    MIN_CORNERS_PER_VIEW,
    MIN_INTRINSICS_VIEWS,
    CharucoBoardDetector,
    CharucoBoardSpec,
    CharucoDetection,
    IntrinsicsCalibration,
    average_detections,
    calibrate_intrinsics,
    coverage_fraction,
    mean_rotation,
    render_board_image,
    render_printable_board,
    solve_board_pose,
    solve_rig_extrinsics,
    view_novelty_px,
)
from biomechanics.triangulation.triangulator import NUM_KEYPOINTS, DLTTriangulator  # noqa: E402
from biomechanics.utils.types import Keypoint2D, MultiViewPose, Skeleton2D  # noqa: E402

RESOLUTION = (1280, 720)
TEXTURE_SQUARE_PX = 100
BACKGROUND_GRAY = 110
LENS_BLUR_SIGMA_PX = 0.8
SENSOR_NOISE_GRAY = 3.0
UNDISTORT_CRITERIA = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 40, 1e-7)
OPENCV_FAILING_SQUARE_PX = 200  # board.generateImage asserts at this size in OpenCV 4.13

HANDHELD_BOARD = CharucoBoardSpec()
FACTORY_BOARD = CharucoBoardSpec(square_length_m=0.12, marker_length_m=0.09)
A4_MM = (210.0, 297.0)
PRINT_DPI = 300
MM_PER_INCH = 25.4

TRUE_K = np.array([[1012.0, 0.0, 652.0], [0.0, 1008.0, 347.0], [0.0, 0.0, 1.0]])
TRUE_DIST = np.array([-0.26, 0.11, 0.0012, -0.0009, -0.03])
NUM_INTRINSICS_VIEWS = 24

FLOOR_Y_M = 0.98
RIG_DISTANCE_M = 3.5
RIG_YAWS_DEG = {"0": 0.0, "1": -40.0, "2": 40.0}
NUM_RIG_PLACEMENTS = 8
SYNTHETIC_CORNER_NOISE_PX = 0.15

FOCAL_TOL_RATIO = 0.01
PRINCIPAL_POINT_TOL_PX = 3.0
DISTORTION_TOL_RATIO = 0.10
MEAN_FIELD_TOL_PX = 0.5
ROUND_TRIP_TOL_PX = 0.1
DETECTION_BIAS_TOL_PX = 0.1
DETECTION_RMS_TOL_PX = 0.3
CAMERA_CENTRE_TOL_M = 0.005
EXACT_CENTRE_TOL_M = 1e-4
EXACT_TOL = 1e-6
ANGLE_TOL_DEG = 0.1


def _rotation(rx_rad: float, ry_rad: float, rz_rad: float) -> np.ndarray:
    return (
        cv2.Rodrigues(np.array([rx_rad, 0.0, 0.0]))[0]
        @ cv2.Rodrigues(np.array([0.0, ry_rad, 0.0]))[0]
        @ cv2.Rodrigues(np.array([0.0, 0.0, rz_rad]))[0]
    )


def _look_at(centre: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # World -> camera with image-down along world +Y (Y-down world).
    z_axis = (target - centre) / np.linalg.norm(target - centre)
    x_axis = np.cross([0.0, 1.0, 0.0], z_axis)
    x_axis /= np.linalg.norm(x_axis)
    rotation = np.stack([x_axis, np.cross(z_axis, x_axis), z_axis])
    return rotation, -rotation @ centre


def _board_texture(spec: CharucoBoardSpec) -> tuple[np.ndarray, float, int]:
    # Board on a white sheet; returns (texture, texture px per metre, sheet margin px).
    board = render_board_image(spec, TEXTURE_SQUARE_PX)
    square_px = board.shape[1] / spec.squares_x
    margin_px = int(square_px // 2)
    sheet = cv2.copyMakeBorder(board, margin_px, margin_px, margin_px, margin_px, cv2.BORDER_CONSTANT, value=255)
    return sheet, square_px / spec.square_length_m, margin_px


def _render_view(
    texture: tuple[np.ndarray, float, int],
    rotation: np.ndarray,
    translation: np.ndarray,
    K: np.ndarray,
    dist_coeffs: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    sheet, px_per_m, margin_px = texture
    width, height = RESOLUTION
    sheet_size_m = np.array([sheet.shape[1], sheet.shape[0]]) / px_per_m
    edge = np.linspace(0.0, 1.0, 40)
    outline = np.concatenate([
        np.column_stack([edge, np.zeros(40)]), np.column_stack([edge, np.ones(40)]),
        np.column_stack([np.zeros(40), edge]), np.column_stack([np.ones(40), edge]),
    ]) * sheet_size_m - margin_px / px_per_m
    outline_px, _ = cv2.projectPoints(
        np.column_stack([outline, np.zeros(len(outline))]), cv2.Rodrigues(rotation)[0], translation, K, dist_coeffs
    )
    outline_px = outline_px.reshape(-1, 2)
    x0, y0 = np.clip(np.floor(outline_px.min(axis=0)).astype(int) - 3, 0, [width - 1, height - 1])
    x1, y1 = np.clip(np.ceil(outline_px.max(axis=0)).astype(int) + 3, 0, [width - 1, height - 1])

    xs, ys = np.meshgrid(np.arange(x0, x1 + 1, dtype=np.float64), np.arange(y0, y1 + 1, dtype=np.float64))
    pixels = np.stack([xs.ravel(), ys.ravel()], axis=1).reshape(-1, 1, 2)
    rays = cv2.undistortPointsIter(pixels, K, dist_coeffs, None, None, UNDISTORT_CRITERIA).reshape(-1, 2)
    board_to_ray = np.column_stack([rotation[:, 0], rotation[:, 1], translation])
    plane = np.column_stack([rays, np.ones(len(rays))]) @ np.linalg.inv(board_to_ray).T
    board_xy_m = plane[:, :2] / plane[:, 2:3]
    # Texture pixel i covers [i, i + 1) sheet pixels, so its centre is at i + 0.5.
    map_x = (board_xy_m[:, 0] * px_per_m - 0.5 + margin_px).astype(np.float32).reshape(xs.shape)
    map_y = (board_xy_m[:, 1] * px_per_m - 0.5 + margin_px).astype(np.float32).reshape(xs.shape)

    image = np.full((height, width), BACKGROUND_GRAY, dtype=np.uint8)
    image[y0:y1 + 1, x0:x1 + 1] = cv2.remap(
        sheet, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=BACKGROUND_GRAY
    )
    image = cv2.GaussianBlur(image, (0, 0), LENS_BLUR_SIGMA_PX)
    return np.clip(image + rng.normal(0.0, SENSOR_NOISE_GRAY, image.shape), 0, 255).astype(np.uint8)


def _project_corners(
    spec: CharucoBoardSpec, rotation: np.ndarray, translation: np.ndarray, K: np.ndarray, dist_coeffs: np.ndarray
) -> np.ndarray:
    projected, _ = cv2.projectPoints(spec.corner_positions_m(), cv2.Rodrigues(rotation)[0], translation, K, dist_coeffs)
    return projected.reshape(-1, 2)


def _synthetic_detection(
    spec: CharucoBoardSpec,
    rotation: np.ndarray,
    translation: np.ndarray,
    K: np.ndarray,
    dist_coeffs: np.ndarray,
    rng: np.random.Generator | None = None,
) -> CharucoDetection:
    corners_px = _project_corners(spec, rotation, translation, K, dist_coeffs)
    if rng is not None:
        corners_px = corners_px + rng.normal(0.0, SYNTHETIC_CORNER_NOISE_PX, corners_px.shape)
    return CharucoDetection(corners_px=corners_px, corner_ids=np.arange(len(corners_px)))


def _handheld_pose(spec: CharucoBoardSpec, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    ray = np.linalg.inv(TRUE_K) @ np.array([rng.uniform(150.0, 1130.0), rng.uniform(100.0, 620.0), 1.0])
    rotation = _rotation(rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), rng.uniform(-0.4, 0.4))
    return rotation, ray * rng.uniform(0.30, 0.60) - rotation @ spec.centre_m()


def _true_rig(camera_height_m: float = 1.0, distance_m: float = RIG_DISTANCE_M) -> dict[str, tuple]:
    # camera_id -> (world->camera R, t, K, dist); world is Y-down with the origin at the hips.
    rig = {}
    for i, (camera_id, yaw_deg) in enumerate(RIG_YAWS_DEG.items()):
        yaw_rad = math.radians(yaw_deg)
        centre = np.array([
            distance_m * math.sin(yaw_rad), FLOOR_Y_M - camera_height_m - 0.05 * i, -distance_m * math.cos(yaw_rad),
        ])
        rotation, translation = _look_at(centre, np.array([0.0, FLOOR_Y_M - 0.9, 0.0]))
        K = np.array([[1000.0 + 15.0 * i, 0.0, 640.0 + 6.0 * i], [0.0, 1003.0 + 15.0 * i, 360.0 - 5.0 * i], [0.0, 0.0, 1.0]])
        rig[camera_id] = (rotation, translation, K, np.array([-0.12 - 0.04 * i, 0.05, 0.0008, -0.0006, 0.0]))
    return rig


def _rig_intrinsics(rig: dict[str, tuple]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {camera_id: (K, dist_coeffs) for camera_id, (_, _, K, dist_coeffs) in rig.items()}


def _upright_board_pose(spec: CharucoBoardSpec, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    # Board -> world: near the lifter's spot, printed face toward the cameras at -Z.
    centre = np.array([rng.uniform(-0.3, 0.3), rng.uniform(-0.45, 0.25), rng.uniform(-0.3, 0.3)])
    rotation = _rotation(rng.uniform(-0.35, 0.35), rng.uniform(-0.45, 0.45), rng.uniform(-0.2, 0.2))
    return rotation, centre - rotation @ spec.centre_m()


def _flat_board_pose(spec: CharucoBoardSpec) -> tuple[np.ndarray, np.ndarray]:
    # Board lying on the floor, printed face up (board normal Z = world -Y... i.e. up).
    rotation = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    return rotation, np.array([0.0, FLOOR_Y_M, 0.0]) - rotation @ spec.centre_m()


def _synthetic_placements(
    rig: dict[str, tuple],
    board_poses: list[tuple[np.ndarray, np.ndarray]],
    rng: np.random.Generator | None = None,
) -> list[dict[str, CharucoDetection]]:
    return [
        {
            camera_id: _synthetic_detection(FACTORY_BOARD, R @ board_R, R @ board_t + t, K, dist_coeffs, rng)
            for camera_id, (R, t, K, dist_coeffs) in rig.items()
        }
        for board_R, board_t in board_poses
    ]


def _centres_in_reference_frame(poses: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, np.ndarray]:
    # Camera centres expressed in camera "0"'s frame: independent of the world frame choice.
    R_ref, t_ref = poses["0"]
    return {camera_id: R_ref @ (-R.T @ t) + t_ref for camera_id, (R, t) in poses.items()}


def _solved_poses(result: CalibrationResult) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        camera_id: (camera.rotation_matrix, camera.translation_vector.reshape(3))
        for camera_id, camera in result.cameras.items()
    }


def _max_centre_error_m(result: CalibrationResult, rig: dict[str, tuple]) -> float:
    truth = _centres_in_reference_frame({camera_id: (R, t) for camera_id, (R, t, _, _) in rig.items()})
    solved = _centres_in_reference_frame(_solved_poses(result))
    return max(float(np.linalg.norm(solved[camera_id] - truth[camera_id])) for camera_id in truth)


def _world_axes_in_true_world(result: CalibrationResult, rig: dict[str, tuple]) -> np.ndarray:
    # Columns: the solved world X, Y, Z axes expressed in the true world frame.
    return rig["0"][0].T @ result.cameras["0"].rotation_matrix


def _longest_dark_run_px(page: np.ndarray, axis: int) -> int:
    # Extent of the board along the other axis: the longest contiguous run of rows/columns
    # with many dark pixels. The caption's text lines are dark too, but only a pixel or two thick.
    dark_counts = (page < 128).sum(axis=axis)
    is_dark = np.concatenate([[False], dark_counts > page.shape[axis] // 16, [False]])
    edges = np.flatnonzero(np.diff(is_dark.astype(int)))
    return int((edges[1::2] - edges[0::2]).max())


def _angle_deg(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    cos_angle = float(vector_a @ vector_b / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b)))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_angle))))


def _load_cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location("calibrate_cameras", REPO_ROOT / "scripts" / "tools" / "calibrate_cameras.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_cli(monkeypatch: pytest.MonkeyPatch, arguments: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["calibrate_cameras.py"] + arguments)
    _load_cli().main()


@pytest.fixture(scope="module")
def handheld_images() -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    # (image, board rotation, board translation): rendered hand-held views of one camera
    # in which the board is detected.
    rng = np.random.default_rng(1)
    texture = _board_texture(HANDHELD_BOARD)
    detector = CharucoBoardDetector(HANDHELD_BOARD)
    images = []
    while len(images) < NUM_INTRINSICS_VIEWS:
        rotation, translation = _handheld_pose(HANDHELD_BOARD, rng)
        image = _render_view(texture, rotation, translation, TRUE_K, TRUE_DIST, rng)
        if detector.detect(image) is not None:
            images.append((image, rotation, translation))
    return images


@pytest.fixture(scope="module")
def handheld_views(
    handheld_images: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> list[tuple[CharucoDetection, np.ndarray, np.ndarray]]:
    detector = CharucoBoardDetector(HANDHELD_BOARD)
    return [(detector.detect(image), rotation, translation) for image, rotation, translation in handheld_images]


@pytest.fixture(scope="module")
def rig_placement_images() -> list[dict[str, np.ndarray]]:
    # Rendered views of the factory board from every camera of _true_rig(), per placement.
    rig = _true_rig()
    rng = np.random.default_rng(0)
    texture = _board_texture(FACTORY_BOARD)
    placements = []
    for _ in range(NUM_RIG_PLACEMENTS):
        board_R, board_t = _upright_board_pose(FACTORY_BOARD, rng)
        placements.append({
            camera_id: _render_view(texture, R @ board_R, R @ board_t + t, K, dist_coeffs, rng)
            for camera_id, (R, t, K, dist_coeffs) in rig.items()
        })
    return placements


@pytest.fixture(scope="module")
def solved_intrinsics(handheld_views: list[tuple[CharucoDetection, np.ndarray, np.ndarray]]) -> IntrinsicsCalibration:
    return calibrate_intrinsics([view[0] for view in handheld_views], HANDHELD_BOARD, RESOLUTION)


class TestBoardSpec:

    def test_corner_positions_match_the_opencv_board(self) -> None:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        board = cv2.aruco.CharucoBoard((7, 5), 0.035, 0.026, dictionary)

        assert HANDHELD_BOARD.corner_positions_m() == pytest.approx(board.getChessboardCorners(), abs=EXACT_TOL)

    def test_centre_is_half_the_board_size(self) -> None:
        assert HANDHELD_BOARD.centre_m() == pytest.approx([0.1225, 0.0875, 0.0], abs=EXACT_TOL)

    def test_marker_not_smaller_than_square_raises(self) -> None:
        with pytest.raises(ValueError, match="marker"):
            CharucoBoardSpec(square_length_m=0.03, marker_length_m=0.03)

    def test_unknown_dictionary_raises(self) -> None:
        with pytest.raises(ValueError, match="dictionary"):
            CharucoBoardSpec(dictionary_name="DICT_NOT_REAL")


class TestBoardRendering:

    def test_board_image_has_square_cells_at_least_the_requested_size(self) -> None:
        image = render_board_image(HANDHELD_BOARD, 40)

        assert image.shape[1] / HANDHELD_BOARD.squares_x == pytest.approx(image.shape[0] / HANDHELD_BOARD.squares_y)
        assert image.shape[1] // HANDHELD_BOARD.squares_x >= 40

    def test_size_that_opencv_asserts_on_still_renders(self) -> None:
        image = render_board_image(HANDHELD_BOARD, OPENCV_FAILING_SQUARE_PX)

        assert image.shape[1] // HANDHELD_BOARD.squares_x >= OPENCV_FAILING_SQUARE_PX

    def test_printable_page_is_landscape_a4_with_the_board_at_true_scale(self) -> None:
        page = render_printable_board(HANDHELD_BOARD, A4_MM, PRINT_DPI)

        px_per_mm = PRINT_DPI / MM_PER_INCH
        assert page.shape == (round(210.0 * px_per_mm), round(297.0 * px_per_mm))
        width_mm, height_mm = _longest_dark_run_px(page, axis=0) / px_per_mm, _longest_dark_run_px(page, axis=1) / px_per_mm
        assert width_mm == pytest.approx(HANDHELD_BOARD.size_m[0] * 1000.0, abs=0.5)
        assert height_mm == pytest.approx(HANDHELD_BOARD.size_m[1] * 1000.0, abs=0.5)

    def test_printable_page_is_detected_in_full(self) -> None:
        page = render_printable_board(HANDHELD_BOARD, A4_MM, PRINT_DPI)
        small = cv2.resize(page, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

        detection = CharucoBoardDetector(HANDHELD_BOARD).detect(small)

        assert detection is not None
        assert len(detection.corner_ids) == len(HANDHELD_BOARD.corner_positions_m())

    def test_board_larger_than_the_page_raises(self) -> None:
        with pytest.raises(ValueError, match="does not fit"):
            render_printable_board(FACTORY_BOARD, A4_MM, PRINT_DPI)


class TestDetection:

    def test_corners_match_project_points_without_half_pixel_bias(
        self, handheld_views: list[tuple[CharucoDetection, np.ndarray, np.ndarray]]
    ) -> None:
        errors = []
        for detection, rotation, translation in handheld_views:
            truth = _project_corners(HANDHELD_BOARD, rotation, translation, TRUE_K, TRUE_DIST)
            errors.append(detection.corners_px - truth[detection.corner_ids])
        errors = np.concatenate(errors)

        assert errors.mean(axis=0) == pytest.approx([0.0, 0.0], abs=DETECTION_BIAS_TOL_PX)
        assert math.sqrt(float(np.mean(np.sum(errors**2, axis=1)))) < DETECTION_RMS_TOL_PX

    def test_image_without_a_board_returns_none(self) -> None:
        blank = np.full((RESOLUTION[1], RESOLUTION[0]), BACKGROUND_GRAY, dtype=np.uint8)

        assert CharucoBoardDetector(HANDHELD_BOARD).detect(blank) is None

    def test_bgr_image_gives_the_same_corners_as_grayscale(self) -> None:
        rng = np.random.default_rng(3)
        rotation, translation = _rotation(0.3, -0.4, 0.2), np.array([-0.1, -0.08, 0.5])
        gray = _render_view(_board_texture(HANDHELD_BOARD), rotation, translation, TRUE_K, TRUE_DIST, rng)
        detector = CharucoBoardDetector(HANDHELD_BOARD)

        from_gray = detector.detect(gray)
        from_bgr = detector.detect(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))

        assert len(from_gray.corner_ids) >= MIN_CORNERS_PER_VIEW
        assert from_bgr.corners_px == pytest.approx(from_gray.corners_px, abs=EXACT_TOL)


class TestViewSelection:

    def test_novelty_is_infinite_with_nothing_accepted(self) -> None:
        detection = CharucoDetection(corners_px=np.zeros((4, 2)), corner_ids=np.arange(4))

        assert view_novelty_px(detection, []) == math.inf

    def test_novelty_is_the_mean_shift_to_the_closest_view_over_shared_ids(self) -> None:
        base = CharucoDetection(corners_px=np.array([[10.0, 10.0], [20.0, 10.0], [30.0, 10.0]]), corner_ids=np.array([0, 1, 2]))
        near = CharucoDetection(corners_px=np.array([[23.0, 14.0], [33.0, 14.0]]), corner_ids=np.array([1, 2]))
        far = CharucoDetection(corners_px=base.corners_px + 100.0, corner_ids=base.corner_ids)

        assert view_novelty_px(near, [far, base]) == pytest.approx(5.0, abs=EXACT_TOL)

    def test_views_without_shared_ids_are_not_comparable(self) -> None:
        first = CharucoDetection(corners_px=np.zeros((2, 2)), corner_ids=np.array([0, 1]))
        second = CharucoDetection(corners_px=np.zeros((2, 2)), corner_ids=np.array([5, 6]))

        assert view_novelty_px(first, [second]) == math.inf

    def test_average_keeps_ids_seen_in_every_frame(self) -> None:
        first = CharucoDetection(corners_px=np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]]), corner_ids=np.array([2, 0, 1]))
        second = CharucoDetection(corners_px=np.array([[12.0, 2.0], [22.0, 2.0]]), corner_ids=np.array([0, 1]))

        averaged = average_detections([first, second])

        assert averaged.corner_ids.tolist() == [0, 1]
        assert averaged.corners_px == pytest.approx(np.array([[11.0, 1.0], [21.0, 1.0]]), abs=EXACT_TOL)

    def test_coverage_counts_grid_cells_holding_a_corner(self) -> None:
        one_corner = CharucoDetection(corners_px=np.array([[5.0, 5.0]]), corner_ids=np.array([0]))
        xs, ys = np.meshgrid(np.linspace(1.0, RESOLUTION[0] - 1.0, 40), np.linspace(1.0, RESOLUTION[1] - 1.0, 30))
        everywhere = CharucoDetection(corners_px=np.column_stack([xs.ravel(), ys.ravel()]), corner_ids=np.arange(xs.size))

        assert coverage_fraction([one_corner], RESOLUTION) == pytest.approx(1.0 / 48.0, abs=EXACT_TOL)
        assert coverage_fraction([one_corner, everywhere], RESOLUTION) == pytest.approx(1.0, abs=EXACT_TOL)


class TestIntrinsics:

    def test_focal_length_within_one_percent(self, solved_intrinsics: IntrinsicsCalibration) -> None:
        K = solved_intrinsics.intrinsic_matrix

        assert K[0, 0] == pytest.approx(TRUE_K[0, 0], rel=FOCAL_TOL_RATIO)
        assert K[1, 1] == pytest.approx(TRUE_K[1, 1], rel=FOCAL_TOL_RATIO)
        assert K[:2, 2] == pytest.approx(TRUE_K[:2, 2], abs=PRINCIPAL_POINT_TOL_PX)

    def test_distortion_within_ten_percent(self, solved_intrinsics: IntrinsicsCalibration) -> None:
        # k2 and k3 trade off against each other, so beyond the dominant k1 the lens model
        # is judged by what it does: the undistortion shift it applies across the image.
        grid_x, grid_y = np.meshgrid(np.linspace(0.0, RESOLUTION[0] - 1.0, 33), np.linspace(0.0, RESOLUTION[1] - 1.0, 19))
        pixels = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        true_camera = CameraCalibration("true", np.zeros((3, 4)), TRUE_K, np.eye(3), np.zeros(3), 0.0, RESOLUTION, TRUE_DIST)
        solved_camera = CameraCalibration(
            "solved", np.zeros((3, 4)), solved_intrinsics.intrinsic_matrix, np.eye(3), np.zeros(3), 0.0,
            RESOLUTION, solved_intrinsics.distortion_coeffs,
        )
        true_undistorted = undistort_keypoints(pixels, true_camera)
        field_error_px = np.linalg.norm(undistort_keypoints(pixels, solved_camera) - true_undistorted, axis=1)
        max_true_shift_px = float(np.linalg.norm(true_undistorted - pixels, axis=1).max())

        assert solved_intrinsics.distortion_coeffs[0] == pytest.approx(TRUE_DIST[0], rel=DISTORTION_TOL_RATIO)
        assert field_error_px.max() < DISTORTION_TOL_RATIO * max_true_shift_px
        assert field_error_px.mean() < MEAN_FIELD_TOL_PX

    def test_undistortion_round_trip_below_a_tenth_of_a_pixel(self, solved_intrinsics: IntrinsicsCalibration) -> None:
        rng = np.random.default_rng(5)
        points_3d = np.column_stack([rng.uniform(-0.9, 0.9, 200), rng.uniform(-0.5, 0.5, 200), rng.uniform(1.5, 4.0, 200)])
        K, dist_coeffs = solved_intrinsics.intrinsic_matrix, solved_intrinsics.distortion_coeffs
        camera = CameraCalibration("solved", np.zeros((3, 4)), K, np.eye(3), np.zeros(3), 0.0, RESOLUTION, dist_coeffs)
        distorted, _ = cv2.projectPoints(points_3d, np.zeros(3), np.zeros(3), K, dist_coeffs)
        pinhole, _ = cv2.projectPoints(points_3d, np.zeros(3), np.zeros(3), K, np.zeros(5))

        recovered = undistort_keypoints(distorted.reshape(-1, 2), camera)

        assert np.abs(recovered - pinhole.reshape(-1, 2)).max() < ROUND_TRIP_TOL_PX

    def test_reports_rms_views_and_tilts(self, solved_intrinsics: IntrinsicsCalibration) -> None:
        assert solved_intrinsics.rms_reprojection_px < DETECTION_RMS_TOL_PX
        assert solved_intrinsics.views_used == NUM_INTRINSICS_VIEWS
        assert solved_intrinsics.view_tilts_deg.shape == (NUM_INTRINSICS_VIEWS,)
        assert 20.0 < solved_intrinsics.view_tilts_deg.max() < 60.0

    def test_corrupted_view_is_dropped(
        self, handheld_views: list[tuple[CharucoDetection, np.ndarray, np.ndarray]]
    ) -> None:
        detections = [view[0] for view in handheld_views]
        blurred = CharucoDetection(
            corners_px=detections[0].corners_px + np.random.default_rng(7).normal(0.0, 6.0, detections[0].corners_px.shape),
            corner_ids=detections[0].corner_ids,
        )

        result = calibrate_intrinsics([blurred] + detections[1:], HANDHELD_BOARD, RESOLUTION)

        assert result.views_used == NUM_INTRINSICS_VIEWS - 1
        assert result.intrinsic_matrix[0, 0] == pytest.approx(TRUE_K[0, 0], rel=FOCAL_TOL_RATIO)

    def test_too_few_views_raises(self, handheld_views: list[tuple[CharucoDetection, np.ndarray, np.ndarray]]) -> None:
        with pytest.raises(ValueError, match="views"):
            calibrate_intrinsics([view[0] for view in handheld_views[: MIN_INTRINSICS_VIEWS - 1]], HANDHELD_BOARD, RESOLUTION)


class TestBoardPose:

    def test_known_pose_recovered_through_distortion(self) -> None:
        rotation, translation = _rotation(0.4, -0.3, 0.15), np.array([-0.2, -0.1, 2.5])
        detection = _synthetic_detection(FACTORY_BOARD, rotation, translation, TRUE_K, TRUE_DIST)

        solved_rotation, solved_translation = solve_board_pose(
            detection, FACTORY_BOARD.corner_positions_m(), TRUE_K, TRUE_DIST
        )

        assert solved_rotation == pytest.approx(rotation, abs=1e-5)
        assert solved_translation == pytest.approx(translation, abs=1e-5)


class TestMeanRotation:

    def test_symmetric_perturbations_average_to_the_centre_rotation(self) -> None:
        centre = _rotation(0.3, -0.5, 0.8)
        rotations = np.array([centre @ _rotation(0.0, angle_rad, 0.0) for angle_rad in (-0.04, -0.01, 0.01, 0.04)])

        mean = mean_rotation(rotations)

        assert mean == pytest.approx(centre, abs=EXACT_TOL)
        assert np.linalg.det(mean) == pytest.approx(1.0, abs=EXACT_TOL)


class TestRigExtrinsics:

    def test_noiseless_placements_recover_camera_centres_and_metric_scale(self) -> None:
        rig = _true_rig()
        rng = np.random.default_rng(11)
        placements = _synthetic_placements(rig, [_upright_board_pose(FACTORY_BOARD, rng) for _ in range(NUM_RIG_PLACEMENTS)])

        result = solve_rig_extrinsics(placements, FACTORY_BOARD, _rig_intrinsics(rig), RESOLUTION)

        assert _max_centre_error_m(result, rig) < EXACT_CENTRE_TOL_M
        assert all(camera.reprojection_error < 0.01 for camera in result.cameras.values())

    def test_rendered_placements_recover_camera_centres_within_5mm(
        self, rig_placement_images: list[dict[str, np.ndarray]]
    ) -> None:
        rig = _true_rig()
        detector = CharucoBoardDetector(FACTORY_BOARD)
        placements = []
        for images in rig_placement_images:
            views = {camera_id: detector.detect(image) for camera_id, image in images.items()}
            placements.append({camera_id: detection for camera_id, detection in views.items() if detection is not None})

        result = solve_rig_extrinsics(placements, FACTORY_BOARD, _rig_intrinsics(rig), RESOLUTION)

        assert _max_centre_error_m(result, rig) < CAMERA_CENTRE_TOL_M
        assert all(camera.reprojection_error < 0.5 for camera in result.cameras.values())

    def test_result_carries_real_intrinsics_and_the_board_anchor(self) -> None:
        rig = _true_rig()
        rng = np.random.default_rng(12)
        placements = _synthetic_placements(rig, [_upright_board_pose(FACTORY_BOARD, rng) for _ in range(4)])

        result = solve_rig_extrinsics(placements, FACTORY_BOARD, _rig_intrinsics(rig), RESOLUTION)

        assert result.world_anchor == WORLD_ANCHOR_BOARD
        for camera_id, (_, _, K, dist_coeffs) in rig.items():
            camera = result.cameras[camera_id]
            assert camera.intrinsic_matrix == pytest.approx(K)
            assert camera.distortion_coeffs == pytest.approx(dist_coeffs)
            assert camera.resolution == RESOLUTION
            expected_projection = K @ np.hstack([camera.rotation_matrix, camera.translation_vector.reshape(3, 1)])
            assert camera.projection_matrix == pytest.approx(expected_projection, abs=EXACT_TOL)

    def test_world_origin_is_the_mean_board_centre_in_a_right_handed_frame(self) -> None:
        rig = _true_rig()
        rng = np.random.default_rng(13)
        board_poses = [_upright_board_pose(FACTORY_BOARD, rng) for _ in range(6)]
        mean_centre = np.mean([R @ FACTORY_BOARD.centre_m() + t for R, t in board_poses], axis=0)

        result = solve_rig_extrinsics(_synthetic_placements(rig, board_poses), FACTORY_BOARD, _rig_intrinsics(rig), RESOLUTION)

        R_true, t_true, _, _ = rig["0"]
        camera = result.cameras["0"]
        assert camera.translation_vector.reshape(3) == pytest.approx(R_true @ mean_centre + t_true, abs=EXACT_CENTRE_TOL_M)
        axes = _world_axes_in_true_world(result, rig)
        assert np.linalg.det(axes) == pytest.approx(1.0, abs=EXACT_TOL)
        # Level-ish cameras in front of the lifter: X ~ left, Y ~ down, Z ~ back, within a few degrees.
        assert np.diag(axes) == pytest.approx([1.0, 1.0, 1.0], abs=0.01)

    def test_flat_placement_sets_y_down_to_gravity(self) -> None:
        # Rack-style rig: high cameras pitched down ~35 deg, so their image-down axes are far from gravity.
        rig = _true_rig(camera_height_m=2.2, distance_m=2.2)
        rng = np.random.default_rng(14)
        board_poses = [_flat_board_pose(FACTORY_BOARD)] + [_upright_board_pose(FACTORY_BOARD, rng) for _ in range(5)]
        placements = _synthetic_placements(rig, board_poses)
        true_down = np.array([0.0, 1.0, 0.0])

        levelled = solve_rig_extrinsics(placements, FACTORY_BOARD, _rig_intrinsics(rig), RESOLUTION, flat_placement_index=0)
        from_cameras = solve_rig_extrinsics(placements, FACTORY_BOARD, _rig_intrinsics(rig), RESOLUTION)

        assert _angle_deg(_world_axes_in_true_world(levelled, rig)[:, 1], true_down) < ANGLE_TOL_DEG
        assert _angle_deg(_world_axes_in_true_world(from_cameras, rig)[:, 1], true_down) > 10.0
        assert _world_axes_in_true_world(levelled, rig)[2, 2] > 0.99  # +Z still points away from the cameras

    def test_placement_where_the_board_moved_between_cameras_is_dropped(self) -> None:
        rig = _true_rig()
        rng = np.random.default_rng(15)
        board_poses = [_upright_board_pose(FACTORY_BOARD, rng) for _ in range(10)]
        placements = _synthetic_placements(rig, board_poses, rng)
        moved_R = board_poses[3][0] @ _rotation(0.0, math.radians(1.0), 0.0)
        moved_t = board_poses[3][1] + np.array([0.01, 0.0, 0.0])
        R, t, K, dist_coeffs = rig["1"]
        placements[3]["1"] = _synthetic_detection(FACTORY_BOARD, R @ moved_R, R @ moved_t + t, K, dist_coeffs, rng)

        result = solve_rig_extrinsics(placements, FACTORY_BOARD, _rig_intrinsics(rig), RESOLUTION)
        without_it = solve_rig_extrinsics(placements[:3] + placements[4:], FACTORY_BOARD, _rig_intrinsics(rig), RESOLUTION)

        # Kept in the solve, a 1 cm / 1 deg move shifts the camera centres by 4-23 mm (seed
        # dependent), so the result must equal never having seen that placement.
        solved = _centres_in_reference_frame(_solved_poses(result))
        expected = _centres_in_reference_frame(_solved_poses(without_it))
        for camera_id in expected:
            assert solved[camera_id] == pytest.approx(expected[camera_id], abs=EXACT_CENTRE_TOL_M)
        assert _max_centre_error_m(result, rig) < CAMERA_CENTRE_TOL_M

    def test_camera_sharing_too_few_placements_with_the_reference_raises(self) -> None:
        rig = _true_rig()
        rng = np.random.default_rng(16)
        placements = _synthetic_placements(rig, [_upright_board_pose(FACTORY_BOARD, rng) for _ in range(5)])
        for placement in placements[2:]:
            del placement["2"]

        with pytest.raises(ValueError, match="camera 2 shares 2"):
            solve_rig_extrinsics(placements, FACTORY_BOARD, _rig_intrinsics(rig), RESOLUTION)

    def test_flat_placement_seen_by_one_camera_raises(self) -> None:
        rig = _true_rig()
        rng = np.random.default_rng(17)
        placements = _synthetic_placements(rig, [_upright_board_pose(FACTORY_BOARD, rng) for _ in range(5)])
        placements[0] = {"0": placements[0]["0"]}

        with pytest.raises(ValueError, match="flat placement 0"):
            solve_rig_extrinsics(placements, FACTORY_BOARD, _rig_intrinsics(rig), RESOLUTION, flat_placement_index=0)

    def test_triangulation_through_the_solved_rig_is_metric(self) -> None:
        # Distorted projections of known points -> DLTTriangulator on the board calibration.
        # The world frame differs from the true one by a rigid motion, so compare distances.
        rig = _true_rig()
        rng = np.random.default_rng(18)
        placements = _synthetic_placements(rig, [_upright_board_pose(FACTORY_BOARD, rng) for _ in range(NUM_RIG_PLACEMENTS)], rng)
        result = solve_rig_extrinsics(placements, FACTORY_BOARD, _rig_intrinsics(rig), RESOLUTION)
        points = np.column_stack([
            rng.uniform(-0.5, 0.5, NUM_KEYPOINTS), rng.uniform(-0.8, 0.9, NUM_KEYPOINTS), rng.uniform(-0.4, 0.4, NUM_KEYPOINTS),
        ])
        views = {}
        for camera_id, (R, t, K, dist_coeffs) in rig.items():
            pixels, _ = cv2.projectPoints(points, cv2.Rodrigues(R)[0], t, K, dist_coeffs)
            views[camera_id] = Skeleton2D(
                keypoints=[Keypoint2D(x=float(x), y=float(y), confidence=1.0) for x, y in pixels.reshape(-1, 2)]
            )

        skeleton = DLTTriangulator(result).triangulate(MultiViewPose(views=views, timestamp=0.0, frame_index=0))

        solved = np.array([[kp.x, kp.y, kp.z] for kp in skeleton.keypoints])
        true_distances = np.linalg.norm(points[:, None] - points[None], axis=-1)
        solved_distances = np.linalg.norm(solved[:, None] - solved[None], axis=-1)
        assert all(kp.confidence > 0.0 for kp in skeleton.keypoints)
        assert np.abs(solved_distances - true_distances).max() < 0.003


class TestCalibrateCamerasCli:

    def test_board_command_writes_a_png_tagged_with_its_print_resolution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out_path = tmp_path / "board.png"

        _run_cli(monkeypatch, ["board", "--out", str(out_path), "--paper", "A4", "--dpi", "300"])

        with Image.open(out_path) as page:
            assert page.size == (3508, 2480)
            assert page.info["dpi"] == pytest.approx((300.0, 300.0), abs=0.1)

    def test_intrinsics_from_images_saves_a_file_the_loader_reads(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        handheld_images: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> None:
        image_dir = tmp_path / "views"
        image_dir.mkdir()
        for index, (image, _, _) in enumerate(handheld_images):
            cv2.imwrite(str(image_dir / f"view_{index:02d}.png"), image)

        _run_cli(monkeypatch, [
            "intrinsics", "--from-images", str(image_dir), "--key", "bench_cam", "--calibration-dir", str(tmp_path),
        ])

        K, dist_coeffs = load_intrinsics("bench_cam", RESOLUTION, tmp_path)
        assert K[0, 0] == pytest.approx(TRUE_K[0, 0], rel=FOCAL_TOL_RATIO)
        assert dist_coeffs[0] == pytest.approx(TRUE_DIST[0], rel=DISTORTION_TOL_RATIO)

    def test_extrinsics_from_images_saves_the_rig_file_the_pipeline_loads(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rig_placement_images: list[dict[str, np.ndarray]]
    ) -> None:
        rig = _true_rig()
        for camera_id, (_, _, K, dist_coeffs) in rig.items():
            save_intrinsics(camera_id, RESOLUTION, K, dist_coeffs, 0.0, tmp_path)
            (tmp_path / "placements" / camera_id).mkdir(parents=True)
        for index, images in enumerate(rig_placement_images):
            for camera_id, image in images.items():
                cv2.imwrite(str(tmp_path / "placements" / camera_id / f"placement_{index:02d}.png"), image)

        _run_cli(monkeypatch, [
            "extrinsics", "--cameras", "0,1,2", "--from-images", str(tmp_path / "placements"),
            "--square-mm", "120", "--marker-mm", "90", "--calibration-dir", str(tmp_path),
        ])

        result = TPoseCalibrator.load_calibration(str(tmp_path / "rig_calibration_cams_0-1-2.json"))
        assert result.world_anchor == WORLD_ANCHOR_BOARD
        assert _max_centre_error_m(result, rig) < CAMERA_CENTRE_TOL_M
        assert result.cameras["1"].distortion_coeffs == pytest.approx(rig["1"][3])

    def test_extrinsics_without_saved_intrinsics_stops_with_instructions(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rig_placement_images: list[dict[str, np.ndarray]]
    ) -> None:
        for camera_id, image in rig_placement_images[0].items():
            (tmp_path / "placements" / camera_id).mkdir(parents=True)
            cv2.imwrite(str(tmp_path / "placements" / camera_id / "placement_00.png"), image)

        with pytest.raises(SystemExit, match="run 'intrinsics"):
            _run_cli(monkeypatch, [
                "extrinsics", "--cameras", "0,1,2", "--from-images", str(tmp_path / "placements"),
                "--square-mm", "120", "--marker-mm", "90", "--calibration-dir", str(tmp_path),
            ])

    def test_existing_calibration_file_is_kept_as_a_backup(self, tmp_path: Path) -> None:
        target = tmp_path / "intrinsics_0.json"
        target.write_text("previous")

        _load_cli()._backup_existing(target)

        assert not target.exists()
        assert (tmp_path / "intrinsics_0.json.bak").read_text() == "previous"
