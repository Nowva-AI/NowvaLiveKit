"""
ChArUco board calibration: printable board, corner detection, single-camera intrinsics
(real K + lens distortion) and the factory multi-camera rig extrinsics with metric scale
from the board's square size. Pixel coordinates put pixel centres at integers, like
cv2.cornerSubPix, cv2.projectPoints and the pose keypoints.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache

import cv2
import numpy as np
from pydantic import BaseModel, model_validator
from scipy.optimize import least_squares

from biomechanics.triangulation.calibration import (
    WORLD_ANCHOR_BOARD,
    CalibrationResult,
    CameraCalibration,
)

logger = logging.getLogger(__name__)

# cv2.aruco moved to CharucoBoard(...) / CharucoDetector in OpenCV 4.7; older builds use
# CharucoBoard_create + detectMarkers + interpolateCornersCharuco.
HAS_CHARUCO_DETECTOR = hasattr(cv2.aruco, "CharucoDetector")

MIN_CORNERS_PER_VIEW = 8
MIN_INTRINSICS_VIEWS = 8
# A view is dropped from the intrinsics solve when its RMS error exceeds both limits
# (motion blur or a mis-identified marker), then the solve runs once more.
VIEW_OUTLIER_ERROR_FACTOR = 3.0
VIEW_OUTLIER_MIN_ERROR_PX = 1.0
COVERAGE_GRID_CELLS = (8, 6)

# board.generateImage asserts on some sizes (float rounding of the last marker's ROI in
# OpenCV 4.13: 60, 130, 200, 270 px squares); the next square size up always worked.
BOARD_RENDER_SIZE_RETRIES = 3
PIXEL_OFFSET_PROBE_SQUARE_PX = 100
PIXEL_OFFSET_PROBE_BLUR_SIGMA_PX = 1.0
MM_PER_INCH = 25.4
MIN_PAGE_MARGIN_MM = 10.0
PAGE_LABEL_FONT_SCALE_PER_DPI = 1.0 / 150.0

MIN_SHARED_PLACEMENTS = 3
# Placements whose relative camera rotation is further than this from the medoid are
# planar-pose flips or bad detections and are left out of the average.
MAX_RELATIVE_ROTATION_DEVIATION_DEG = 5.0
REFINE_HUBER_SCALE_PX = 1.0
# A placement whose RMS residual after the joint solve exceeds both limits is inconsistent
# between cameras: the cameras are not hardware-synced, so a board that moved 1 cm between
# their grabs shifts the solved camera centres by ~8 mm while the per-camera error barely
# moves. Such placements are dropped and the rig is solved once more. Measured on synthetic
# rigs: clean placements 0.18-0.25 px, a 1 cm move 0.52 px, a 0.5 cm move only 0.32 px
# (still ~5 mm of camera error), so sub-centimetre motion must be prevented at capture time:
# keep the board on a stand and average a steady window (average_detections).
PLACEMENT_OUTLIER_ERROR_FACTOR = 2.0
PLACEMENT_OUTLIER_MIN_ERROR_PX = 0.4


class CharucoBoardSpec(BaseModel):
    """Physical ChArUco board. Lengths are the MEASURED printed sizes; they set the metric scale."""
    squares_x: int = 7
    squares_y: int = 5
    square_length_m: float = 0.035
    marker_length_m: float = 0.026
    dictionary_name: str = "DICT_5X5_100"

    @model_validator(mode="after")
    def _check_geometry(self) -> CharucoBoardSpec:
        if self.squares_x < 3 or self.squares_y < 3:
            raise ValueError("board needs at least 3x3 squares")
        if not 0.0 < self.marker_length_m < self.square_length_m:
            raise ValueError("marker length must be positive and smaller than the square length")
        if not hasattr(cv2.aruco, self.dictionary_name):
            raise ValueError(f"unknown ArUco dictionary {self.dictionary_name}")
        return self

    @property
    def size_m(self) -> tuple[float, float]:
        return self.squares_x * self.square_length_m, self.squares_y * self.square_length_m

    def corner_positions_m(self) -> np.ndarray:
        """(num_corners, 3) inner chessboard corners in the board frame (Z = 0), indexed by ChArUco corner id."""
        columns, rows = np.meshgrid(np.arange(1, self.squares_x), np.arange(1, self.squares_y))
        corners = np.zeros((columns.size, 3), dtype=np.float64)
        corners[:, 0] = columns.ravel() * self.square_length_m
        corners[:, 1] = rows.ravel() * self.square_length_m
        return corners

    def centre_m(self) -> np.ndarray:
        width_m, height_m = self.size_m
        return np.array([width_m / 2.0, height_m / 2.0, 0.0])


@dataclass
class CharucoDetection:
    corners_px: np.ndarray  # (N, 2)
    corner_ids: np.ndarray  # (N,)


@dataclass
class IntrinsicsCalibration:
    intrinsic_matrix: np.ndarray
    distortion_coeffs: np.ndarray  # k1, k2, p1, p2, k3
    rms_reprojection_px: float
    views_used: int
    view_tilts_deg: np.ndarray  # board tilt away from fronto-parallel, per used view


def _build_board(spec: CharucoBoardSpec) -> object:
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec.dictionary_name))
    if HAS_CHARUCO_DETECTOR:
        return cv2.aruco.CharucoBoard(
            (spec.squares_x, spec.squares_y), spec.square_length_m, spec.marker_length_m, dictionary
        )
    return cv2.aruco.CharucoBoard_create(
        spec.squares_x, spec.squares_y, spec.square_length_m, spec.marker_length_m, dictionary
    )


def render_board_image(spec: CharucoBoardSpec, square_px: int) -> np.ndarray:
    """
    Grayscale board with no margin. Squares are at least square_px wide; read the actual
    scale from the returned shape (width / squares_x).
    """
    board = _build_board(spec)
    for size_px in range(square_px, square_px + BOARD_RENDER_SIZE_RETRIES + 1):
        margin_px = size_px
        out_size = ((spec.squares_x + 2) * size_px, (spec.squares_y + 2) * size_px)
        try:
            if HAS_CHARUCO_DETECTOR:
                image = board.generateImage(out_size, marginSize=margin_px, borderBits=1)
            else:
                image = board.draw(out_size, marginSize=margin_px, borderBits=1)
        except cv2.error:
            continue
        return image[margin_px:-margin_px, margin_px:-margin_px]
    raise ValueError(f"OpenCV could not render the board at {square_px} px per square")


def render_printable_board(
    spec: CharucoBoardSpec, paper_size_mm: tuple[float, float], dpi: int
) -> np.ndarray:
    """
    White page (landscape when the board is wider than tall) with the board centred at
    its nominal size and a caption. Printers rescale, so the caption tells the user to
    measure a printed square.
    """
    board_mm = (spec.size_m[0] * 1000.0, spec.size_m[1] * 1000.0)
    short_mm, long_mm = sorted(paper_size_mm)
    page_mm = (long_mm, short_mm) if board_mm[0] >= board_mm[1] else (short_mm, long_mm)
    if any(board_mm[i] + 2.0 * MIN_PAGE_MARGIN_MM > page_mm[i] for i in range(2)):
        raise ValueError(
            f"board {board_mm[0]:.0f}x{board_mm[1]:.0f} mm does not fit on a "
            f"{page_mm[0]:.0f}x{page_mm[1]:.0f} mm page with {MIN_PAGE_MARGIN_MM:.0f} mm margins"
        )
    px_per_mm = dpi / MM_PER_INCH
    page_px = (round(page_mm[0] * px_per_mm), round(page_mm[1] * px_per_mm))
    board_px = (round(board_mm[0] * px_per_mm), round(board_mm[1] * px_per_mm))
    square_px = math.ceil(board_px[0] / spec.squares_x)
    board_image = cv2.resize(render_board_image(spec, square_px), board_px, interpolation=cv2.INTER_AREA)

    page = np.full((page_px[1], page_px[0]), 255, dtype=np.uint8)
    left_px = (page_px[0] - board_px[0]) // 2
    top_px = (page_px[1] - board_px[1]) // 2
    page[top_px:top_px + board_px[1], left_px:left_px + board_px[0]] = board_image
    caption = (
        f"{spec.squares_x}x{spec.squares_y} squares  square {spec.square_length_m * 1000.0:.1f} mm  "
        f"marker {spec.marker_length_m * 1000.0:.1f} mm  {spec.dictionary_name}  "
        f"print at 100%, then MEASURE a square"
    )
    caption_y_px = top_px + board_px[1] + (page_px[1] - top_px - board_px[1]) // 2
    cv2.putText(
        page, caption, (left_px, caption_y_px), cv2.FONT_HERSHEY_SIMPLEX,
        dpi * PAGE_LABEL_FONT_SCALE_PER_DPI, 0, max(1, dpi // 150), cv2.LINE_AA,
    )
    return page


class CharucoBoardDetector:
    """Detects the ChArUco corners of one board spec in grayscale or BGR images."""

    def __init__(self, spec: CharucoBoardSpec) -> None:
        self.spec = spec
        self._board = _build_board(spec)
        if HAS_CHARUCO_DETECTOR:
            self._detector = cv2.aruco.CharucoDetector(self._board)
        else:
            self._dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec.dictionary_name))

    def detect(self, image: np.ndarray) -> CharucoDetection | None:
        """Corners in pixel-centre coordinates, or None below MIN_CORNERS_PER_VIEW corners."""
        detection = self._detect_raw(image)
        if detection is None or len(detection.corner_ids) < MIN_CORNERS_PER_VIEW:
            return None
        detection.corners_px -= _detector_pixel_offset_px()
        return detection

    def _detect_raw(self, image: np.ndarray) -> CharucoDetection | None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        if HAS_CHARUCO_DETECTOR:
            corners, corner_ids, _, _ = self._detector.detectBoard(gray)
        else:
            marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, self._dictionary)
            if marker_ids is None:
                return None
            _, corners, corner_ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, gray, self._board
            )
        if corner_ids is None or len(corner_ids) == 0:
            return None
        return CharucoDetection(
            corners_px=corners.reshape(-1, 2).astype(np.float64),
            corner_ids=corner_ids.ravel().astype(int),
        )


@lru_cache(maxsize=1)
def _detector_pixel_offset_px() -> float:
    # OpenCV >= 4.8 reports ChArUco corners with pixel centres at +0.5; older builds do not.
    # Measured once on a flat render whose corner positions are known exactly, then snapped
    # to a half pixel so detection noise never leaks into real detections.
    spec = CharucoBoardSpec()
    board_image = render_board_image(spec, PIXEL_OFFSET_PROBE_SQUARE_PX)
    square_px = board_image.shape[1] / spec.squares_x
    margin_px = int(square_px)
    image = cv2.copyMakeBorder(
        board_image, margin_px, margin_px, margin_px, margin_px, cv2.BORDER_CONSTANT, value=255
    )
    image = cv2.GaussianBlur(image, (0, 0), PIXEL_OFFSET_PROBE_BLUR_SIGMA_PX)
    detection = CharucoBoardDetector(spec)._detect_raw(image)
    if detection is None:
        raise RuntimeError("ChArUco detector failed on its own reference board render")
    # Square k ends between pixels k*s-1 and k*s, i.e. at k*s - 0.5 in pixel-centre coordinates.
    expected_px = (
        spec.corner_positions_m()[detection.corner_ids, :2] / spec.square_length_m * square_px
        - 0.5 + margin_px
    )
    offset_px = float(np.mean(detection.corners_px - expected_px))
    return round(offset_px * 2.0) / 2.0


def view_novelty_px(detection: CharucoDetection, accepted: list[CharucoDetection]) -> float:
    """
    How far this view's corners are from the closest accepted view: min over views of the
    mean pixel distance between corners with the same id. inf when nothing is comparable.
    """
    novelty_px = math.inf
    for other in accepted:
        _, own_idx, other_idx = np.intersect1d(
            detection.corner_ids, other.corner_ids, return_indices=True
        )
        if len(own_idx) == 0:
            continue
        distances_px = np.linalg.norm(detection.corners_px[own_idx] - other.corners_px[other_idx], axis=1)
        novelty_px = min(novelty_px, float(distances_px.mean()))
    return novelty_px


def average_detections(detections: list[CharucoDetection]) -> CharucoDetection:
    """
    Mean corner positions over frames of a stationary board, keeping the corner ids seen
    in every frame. Averaging a steady window removes detection noise and bounds how far
    the board can have moved between unsynchronised cameras.
    """
    common_ids = detections[0].corner_ids
    for detection in detections[1:]:
        common_ids = np.intersect1d(common_ids, detection.corner_ids)
    rows = []
    for detection in detections:
        order = np.argsort(detection.corner_ids)
        rows.append(detection.corners_px[order][np.searchsorted(detection.corner_ids[order], common_ids)])
    return CharucoDetection(corners_px=np.mean(rows, axis=0), corner_ids=common_ids)


def coverage_fraction(detections: list[CharucoDetection], resolution: tuple[int, int]) -> float:
    """Fraction of a COVERAGE_GRID_CELLS grid over the image holding at least one detected corner."""
    cells_x, cells_y = COVERAGE_GRID_CELLS
    covered = np.zeros((cells_y, cells_x), dtype=bool)
    for detection in detections:
        cell_x = np.clip((detection.corners_px[:, 0] / resolution[0] * cells_x).astype(int), 0, cells_x - 1)
        cell_y = np.clip((detection.corners_px[:, 1] / resolution[1] * cells_y).astype(int), 0, cells_y - 1)
        covered[cell_y, cell_x] = True
    return float(covered.mean())


def _calibrate_views(
    detections: list[CharucoDetection], corner_positions_m: np.ndarray, resolution: tuple[int, int]
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    object_points = [corner_positions_m[view.corner_ids].astype(np.float32) for view in detections]
    image_points = [view.corners_px.astype(np.float32) for view in detections]
    rms_px, K, dist_coeffs, rvecs, _, _, _, view_errors_px = cv2.calibrateCameraExtended(
        object_points, image_points, resolution, None, None
    )
    return float(rms_px), K, dist_coeffs.ravel(), np.array(rvecs).reshape(-1, 3), view_errors_px.ravel()


def calibrate_intrinsics(
    detections: list[CharucoDetection], spec: CharucoBoardSpec, resolution: tuple[int, int]
) -> IntrinsicsCalibration:
    """
    Solve K and (k1, k2, p1, p2, k3) from hand-held board views; same solve as
    cv2.aruco.calibrateCameraCharuco, which newer OpenCV builds no longer ship.
    """
    if len(detections) < MIN_INTRINSICS_VIEWS:
        raise ValueError(f"need at least {MIN_INTRINSICS_VIEWS} board views, got {len(detections)}")
    corner_positions_m = spec.corner_positions_m()
    rms_px, K, dist_coeffs, rvecs, view_errors_px = _calibrate_views(detections, corner_positions_m, resolution)

    error_limit_px = max(VIEW_OUTLIER_MIN_ERROR_PX, VIEW_OUTLIER_ERROR_FACTOR * float(np.median(view_errors_px)))
    kept = [view for view, error_px in zip(detections, view_errors_px) if error_px <= error_limit_px]
    if MIN_INTRINSICS_VIEWS <= len(kept) < len(detections):
        logger.info("Dropping %d outlier views (> %.2f px)", len(detections) - len(kept), error_limit_px)
        rms_px, K, dist_coeffs, rvecs, _ = _calibrate_views(kept, corner_positions_m, resolution)
    else:
        kept = detections

    board_normals_z = np.array([cv2.Rodrigues(rvec)[0][2, 2] for rvec in rvecs])
    return IntrinsicsCalibration(
        intrinsic_matrix=K,
        distortion_coeffs=dist_coeffs,
        rms_reprojection_px=rms_px,
        views_used=len(kept),
        view_tilts_deg=np.degrees(np.arccos(np.clip(np.abs(board_normals_z), 0.0, 1.0))),
    )


def solve_board_pose(
    detection: CharucoDetection,
    corner_positions_m: np.ndarray,
    intrinsic_matrix: np.ndarray,
    distortion_coeffs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Board -> camera pose (R (3,3), t (3,)) from one view, or None when solvePnP fails."""
    object_points = corner_positions_m[detection.corner_ids]
    success, rvec, tvec = cv2.solvePnP(
        object_points, detection.corners_px, intrinsic_matrix, distortion_coeffs, flags=cv2.SOLVEPNP_IPPE
    )
    if not success:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(
        object_points, detection.corners_px, intrinsic_matrix, distortion_coeffs, rvec, tvec
    )
    return cv2.Rodrigues(rvec)[0], tvec.reshape(3)


def mean_rotation(rotations: np.ndarray) -> np.ndarray:
    """Chordal (L2) mean of (N, 3, 3) rotation matrices: the rotation closest to their sum."""
    u, _, vt = np.linalg.svd(np.sum(rotations, axis=0))
    return u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt


def _rotation_angle_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    cos_angle = (np.trace(rotation_a.T @ rotation_b) - 1.0) / 2.0
    return math.degrees(math.acos(float(np.clip(cos_angle, -1.0, 1.0))))


def _average_relative_pose(
    camera_poses: list[tuple[np.ndarray, np.ndarray]], reference_poses: list[tuple[np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray]:
    # Per shared placement: camera <- reference = (board -> camera) o (board -> reference)^-1.
    rotations = np.array([R_cam @ R_ref.T for (R_cam, _), (R_ref, _) in zip(camera_poses, reference_poses)])
    centres = np.array([
        R_ref @ (-R_cam.T @ t_cam) + t_ref
        for (R_cam, t_cam), (R_ref, t_ref) in zip(camera_poses, reference_poses)
    ])
    spread_deg = np.array([
        [_rotation_angle_deg(rotation, other) for other in rotations] for rotation in rotations
    ])
    medoid = int(np.argmin(spread_deg.sum(axis=1)))
    inliers = spread_deg[medoid] <= MAX_RELATIVE_ROTATION_DEVIATION_DEG
    rotation = mean_rotation(rotations[inliers])
    return rotation, -rotation @ centres[inliers].mean(axis=0)


def _observation_residuals_px(
    relative_poses: dict[str, tuple[np.ndarray, np.ndarray]],
    board_poses: list[tuple[np.ndarray, np.ndarray]],
    observations: list[tuple[str, int, CharucoDetection]],
    corner_positions_m: np.ndarray,
    intrinsics: dict[str, tuple[np.ndarray, np.ndarray]],
) -> list[np.ndarray]:
    # One (N, 2) projected-minus-detected array per (camera_id, board slot, detection).
    residuals = []
    for cam_id, slot, detection in observations:
        R_cam, t_cam = relative_poses[cam_id]
        R_board, t_board = board_poses[slot]
        K, dist_coeffs = intrinsics[cam_id]
        projected, _ = cv2.projectPoints(
            corner_positions_m[detection.corner_ids],
            cv2.Rodrigues(R_cam @ R_board)[0], R_cam @ t_board + t_cam, K, dist_coeffs,
        )
        residuals.append(projected.reshape(-1, 2) - detection.corners_px)
    return residuals


def _refine_rig(
    relative_poses: dict[str, tuple[np.ndarray, np.ndarray]],
    board_poses: list[tuple[np.ndarray, np.ndarray]],
    observations: list[tuple[str, int, CharucoDetection]],
    corner_positions_m: np.ndarray,
    intrinsics: dict[str, tuple[np.ndarray, np.ndarray]],
    reference_id: str,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]]:
    # Joint reprojection minimisation over camera <- reference poses and board -> reference
    # poses. Averaged relative poses lever every board-tilt error by the camera distance;
    # this couples all views so that error cancels.
    free_ids = [cam_id for cam_id in relative_poses if cam_id != reference_id]

    def pack(poses: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        return np.concatenate([np.concatenate([cv2.Rodrigues(R)[0].ravel(), t]) for R, t in poses])

    def unpack(params: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(cv2.Rodrigues(row[:3].copy())[0], row[3:]) for row in params.reshape(-1, 6)]

    def residuals(params: np.ndarray) -> np.ndarray:
        poses = unpack(params)
        cameras = dict(zip(free_ids, poses[:len(free_ids)]))
        cameras[reference_id] = (np.eye(3), np.zeros(3))
        errors = _observation_residuals_px(
            cameras, poses[len(free_ids):], observations, corner_positions_m, intrinsics
        )
        return np.concatenate([error.ravel() for error in errors])

    initial = pack([relative_poses[cam_id] for cam_id in free_ids] + board_poses)
    solution = least_squares(residuals, initial, loss="huber", f_scale=REFINE_HUBER_SCALE_PX, x_scale="jac")
    poses = unpack(solution.x)
    refined = dict(zip(free_ids, poses[:len(free_ids)]))
    refined[reference_id] = (np.eye(3), np.zeros(3))
    return refined, poses[len(free_ids):]


def _board_world_axes(
    relative_poses: dict[str, tuple[np.ndarray, np.ndarray]],
    origin: np.ndarray,
    flat_board_pose: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    # Columns = world X (subject's left), Y (down), Z (subject's back) in the reference camera
    # frame. The cameras stand in front of the lifter, so +Z points from them to the origin.
    centres = np.array([-R.T @ t for R, t in relative_poses.values()])
    if flat_board_pose is not None:
        normal = flat_board_pose[0][:, 2]
        faces_cameras = float(normal @ (centres.mean(axis=0) - origin)) > 0.0
        y_down = -normal if faces_cameras else normal
    else:
        y_down = np.mean([R[1] for R, _ in relative_poses.values()], axis=0)
    y_down = y_down / np.linalg.norm(y_down)
    z_back = origin - centres.mean(axis=0)
    z_back = z_back - (z_back @ y_down) * y_down
    z_back = z_back / np.linalg.norm(z_back)
    return np.column_stack([np.cross(y_down, z_back), y_down, z_back])


def solve_rig_extrinsics(
    placements: list[dict[str, CharucoDetection]],
    spec: CharucoBoardSpec,
    intrinsics: dict[str, tuple[np.ndarray, np.ndarray]],
    resolution: tuple[int, int],
    flat_placement_index: int | None = None,
) -> CalibrationResult:
    """
    Factory rig calibration from one board seen by several cameras at once, in several
    placements (camera_id -> detection each). Metric scale comes from spec.square_length_m.

    The reference camera is the first sorted camera id; every other camera must share at
    least MIN_SHARED_PLACEMENTS placements with it. Relative poses are averaged over
    placements, then refined jointly on reprojection error; placements the cameras
    disagree on (board moved between their grabs) are dropped and the solve repeated.

    World frame (world_anchor = "board"): origin at the mean board centre, +Z from the
    cameras toward it, X = Y x Z. Y-down is the mean of the cameras' image-down axes,
    which is only as level as the cameras are mounted; pass flat_placement_index for a
    placement where the board lies flat on the floor to take gravity from its normal
    instead. Either way the lifter's position is unknown here, so
    person_calibration.refine re-anchors the frame to the lifter.
    """
    camera_ids = sorted(intrinsics)
    reference_id = camera_ids[0]
    corner_positions_m = spec.corner_positions_m()

    poses: list[dict[str, tuple[np.ndarray, np.ndarray]]] = []
    for placement in placements:
        placement_poses = {}
        for cam_id in camera_ids:
            if cam_id in placement:
                pose = solve_board_pose(placement[cam_id], corner_positions_m, *intrinsics[cam_id])
                if pose is not None:
                    placement_poses[cam_id] = pose
        poses.append(placement_poses)

    relative_poses = {reference_id: (np.eye(3), np.zeros(3))}
    for cam_id in camera_ids[1:]:
        shared = [seen for seen in poses if cam_id in seen and reference_id in seen]
        if len(shared) < MIN_SHARED_PLACEMENTS:
            raise ValueError(
                f"camera {cam_id} shares {len(shared)} board placements with reference camera "
                f"{reference_id}; need at least {MIN_SHARED_PLACEMENTS}"
            )
        relative_poses[cam_id] = _average_relative_pose(
            [seen[cam_id] for seen in shared], [seen[reference_id] for seen in shared]
        )

    # Board -> reference pose per placement, through whichever camera saw it.
    used_indices = [i for i, seen in enumerate(poses) if len(seen) >= 2]
    board_poses = []
    for i in used_indices:
        cam_id = reference_id if reference_id in poses[i] else sorted(poses[i])[0]
        R_cam, t_cam = relative_poses[cam_id]
        R_board, t_board = poses[i][cam_id]
        board_poses.append((R_cam.T @ R_board, R_cam.T @ (t_board - t_cam)))
    observations = [
        (cam_id, slot, placements[i][cam_id])
        for slot, i in enumerate(used_indices)
        for cam_id in poses[i]
    ]
    relative_poses, board_poses = _refine_rig(
        relative_poses, board_poses, observations, corner_positions_m, intrinsics, reference_id
    )

    residuals_px = _observation_residuals_px(relative_poses, board_poses, observations, corner_positions_m, intrinsics)
    placement_rms_px = np.array([
        math.sqrt(np.mean(np.concatenate(
            [
                np.sum(residual**2, axis=1)
                for residual, (_, observed_slot, _) in zip(residuals_px, observations)
                if observed_slot == slot
            ]
        )))
        for slot in range(len(used_indices))
    ])
    rms_limit_px = max(
        PLACEMENT_OUTLIER_MIN_ERROR_PX, PLACEMENT_OUTLIER_ERROR_FACTOR * float(np.median(placement_rms_px))
    )
    consistent = placement_rms_px <= rms_limit_px
    if not consistent.all():
        logger.warning(
            "Dropping inconsistent board placements %s (RMS %s px > %.2f px): board moved between cameras?",
            [used_indices[slot] for slot in np.flatnonzero(~consistent)],
            placement_rms_px[~consistent].round(2).tolist(), rms_limit_px,
        )
        new_slot = {int(old): new for new, old in enumerate(np.flatnonzero(consistent))}
        observations = [
            (cam_id, new_slot[old_slot], detection)
            for cam_id, old_slot, detection in observations
            if old_slot in new_slot
        ]
        for cam_id in camera_ids:
            if sum(observed_id == cam_id for observed_id, _, _ in observations) < MIN_SHARED_PLACEMENTS:
                raise ValueError(f"camera {cam_id} has too few consistent board placements left; recapture")
        used_indices = [used_indices[old] for old in new_slot]
        board_poses = [board_poses[old] for old in new_slot]
        relative_poses, board_poses = _refine_rig(
            relative_poses, board_poses, observations, corner_positions_m, intrinsics, reference_id
        )

    residuals_px = _observation_residuals_px(relative_poses, board_poses, observations, corner_positions_m, intrinsics)
    origin = np.mean([R @ spec.centre_m() + t for R, t in board_poses], axis=0)
    flat_board_pose = None
    if flat_placement_index is not None:
        if flat_placement_index not in used_indices:
            raise ValueError(
                f"flat placement {flat_placement_index} was not seen consistently by two cameras"
            )
        flat_board_pose = board_poses[used_indices.index(flat_placement_index)]
    world_axes = _board_world_axes(relative_poses, origin, flat_board_pose)

    result = CalibrationResult(timestamp=datetime.now().isoformat(), world_anchor=WORLD_ANCHOR_BOARD)
    for cam_id in camera_ids:
        R_cam, t_cam = relative_poses[cam_id]
        K, dist_coeffs = intrinsics[cam_id]
        rotation = R_cam @ world_axes
        translation = R_cam @ origin + t_cam
        errors_px = [
            np.linalg.norm(residual, axis=1)
            for residual, (observed_id, _, _) in zip(residuals_px, observations)
            if observed_id == cam_id
        ]
        result.cameras[cam_id] = CameraCalibration(
            camera_id=cam_id,
            projection_matrix=K @ np.hstack([rotation, translation[:, None]]),
            intrinsic_matrix=K,
            rotation_matrix=rotation,
            translation_vector=translation[:, None],
            reprojection_error=float(np.mean(np.concatenate(errors_px))),
            resolution=resolution,
            distortion_coeffs=dist_coeffs,
        )
        logger.info("Camera %s: board reprojection error = %.2f px", cam_id, result.cameras[cam_id].reprojection_error)
    return result
