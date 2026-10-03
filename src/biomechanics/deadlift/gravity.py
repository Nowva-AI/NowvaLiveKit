"""Measured gravity for the deadlift (PLAN.md §2.1). A ChArUco board laid flat on the
floor gives "up" as a unit vector in each camera's own frame, saved per camera in
~/.nowva/gravity_<camera_key>.json. At run time each camera's up is mapped into the
current world frame through its calibration rotation; a camera that disagrees with the
others has moved and is dropped. Without a usable camera, up is the body's WORLD_UP.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from pydantic import BaseModel, field_validator

from biomechanics.triangulation.calibration import NOWVA_CALIBRATION_DIR
from biomechanics.triangulation.charuco import (
    MIN_CORNERS_PER_VIEW,
    CharucoBoardSpec,
    CharucoDetection,
    average_detections,
    solve_board_pose,
)
from biomechanics.utils.geometry import WORLD_UP

from .types import GRAVITY_SOURCE_BODY, GRAVITY_SOURCE_MEASURED

if TYPE_CHECKING:
    from biomechanics.pose.multi_camera import MultiCameraPoseProvider

logger = logging.getLogger(__name__)

# PLAN.md §2.1: a camera more than this from the others' consensus has moved.
MAX_CAMERA_DISAGREEMENT_DEG = 1.0
NUM_POSE_PARAMETERS = 6  # rvec, tvec: the first columns of the cv2.projectPoints jacobian
NORMAL_DERIVATIVE_STEP_RAD = 1e-6
UNIT_NORM_TOLERANCE = 1e-3


class CameraGravity(BaseModel):
    """Up (against gravity) in one camera's frame, from the board lying flat.

    residual_px is the RMS corner reprojection error of the board pose and
    normal_sigma_deg the 1-sigma uncertainty of the normal it implies.
    """
    camera_key: str
    up_camera: tuple[float, float, float]
    timestamp: str
    board_id: str
    residual_px: float
    normal_sigma_deg: float
    num_frames: int
    num_corners: int

    @field_validator("up_camera")
    @classmethod
    def _check_unit(cls, up_camera: tuple[float, float, float]) -> tuple[float, float, float]:
        norm = math.sqrt(sum(value * value for value in up_camera))
        if not abs(norm - 1.0) <= UNIT_NORM_TOLERANCE:
            raise ValueError(f"up_camera must be a unit vector, got norm {norm}")
        return up_camera


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def _board_id(spec: CharucoBoardSpec) -> str:
    return (
        f"charuco_{spec.squares_x}x{spec.squares_y}_{spec.square_length_m * 1000.0:.1f}mm_"
        f"{spec.marker_length_m * 1000.0:.1f}mm_{spec.dictionary_name}"
    )


def _normal_sigma_deg(rvec: np.ndarray, pose_covariance: np.ndarray) -> float:
    # First-order spread of the board normal (third column of R(rvec)).
    normal = cv2.Rodrigues(rvec)[0][:, 2]
    derivative = np.zeros((3, 3))
    for i in range(3):
        step = np.zeros(3)
        step[i] = NORMAL_DERIVATIVE_STEP_RAD
        derivative[:, i] = (cv2.Rodrigues(rvec + step)[0][:, 2] - normal) / NORMAL_DERIVATIVE_STEP_RAD
    normal_covariance = derivative @ pose_covariance[:3, :3] @ derivative.T
    return math.degrees(math.sqrt(max(float(np.trace(normal_covariance)), 0.0)))


def gravity_path(camera_key: str, directory: Path | None = None) -> Path:
    return (directory or NOWVA_CALIBRATION_DIR) / f"gravity_{camera_key}.json"


def save_camera_gravity(gravity: CameraGravity, directory: Path | None = None) -> Path:
    path = gravity_path(gravity.camera_key, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(gravity.model_dump(), f, indent=2)
    logger.info("Gravity for camera %s saved to %s", gravity.camera_key, path)
    return path


def load_camera_gravity(camera_key: str, directory: Path | None = None) -> CameraGravity | None:
    path = gravity_path(camera_key, directory)
    if not path.exists():
        return None
    with open(path) as f:
        return CameraGravity(**json.load(f))


def load_rig_gravity(camera_keys: dict[str, str], directory: Path | None = None) -> dict[str, np.ndarray]:
    """camera_id -> up in that camera's frame, for every camera_id -> camera_key with a valid
    gravity file. An unreadable or invalid file is skipped with a warning."""
    per_camera_up = {}
    for camera_id, camera_key in camera_keys.items():
        try:
            gravity = load_camera_gravity(camera_key, directory)
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("Ignoring gravity file %s: %s", gravity_path(camera_key, directory), exc)
            continue
        if gravity is not None:
            per_camera_up[camera_id] = np.array(gravity.up_camera, dtype=np.float64)
    return per_camera_up


def board_up_in_camera(
    board_rotation: np.ndarray, board_translation: np.ndarray, board_centre_m: np.ndarray
) -> np.ndarray:
    """Normal of a board lying on the floor, pointing up, in the camera frame. The camera is
    above the floor, so up is the side of the board that faces the camera."""
    normal = board_rotation[:, 2]
    centre_in_camera = board_rotation @ board_centre_m + np.reshape(board_translation, 3)
    return _unit(normal if float(normal @ -centre_in_camera) > 0.0 else -normal)


def measure_camera_gravity(
    detections: list[CharucoDetection],
    spec: CharucoBoardSpec,
    intrinsic_matrix: np.ndarray,
    distortion_coeffs: np.ndarray,
    camera_key: str,
) -> CameraGravity | None:
    """Up in one camera's frame from detections of the flat board over a steady window
    (corners averaged first). None when too few corners or the pose solve fails."""
    detection = average_detections(detections)
    num_corners = len(detection.corner_ids)
    if num_corners < MIN_CORNERS_PER_VIEW:
        return None
    corner_positions_m = spec.corner_positions_m()
    pose = solve_board_pose(detection, corner_positions_m, intrinsic_matrix, distortion_coeffs)
    if pose is None:
        return None
    rotation, translation = pose
    rvec = cv2.Rodrigues(rotation)[0].ravel()
    projected, jacobian = cv2.projectPoints(
        corner_positions_m[detection.corner_ids], rvec, translation, intrinsic_matrix, distortion_coeffs
    )
    residuals_px = projected.reshape(-1, 2) - detection.corners_px
    residual_px = math.sqrt(float(np.mean(np.sum(residuals_px**2, axis=1))))
    pose_jacobian = jacobian[:, :NUM_POSE_PARAMETERS]
    degrees_of_freedom = max(residuals_px.size - NUM_POSE_PARAMETERS, 1)
    sigma_px_squared = float(np.sum(residuals_px**2)) / degrees_of_freedom
    pose_covariance = sigma_px_squared * np.linalg.pinv(pose_jacobian.T @ pose_jacobian)
    return CameraGravity(
        camera_key=camera_key,
        up_camera=tuple(board_up_in_camera(rotation, translation, spec.centre_m()).tolist()),
        timestamp=datetime.now().isoformat(),
        board_id=_board_id(spec),
        residual_px=residual_px,
        normal_sigma_deg=_normal_sigma_deg(rvec, pose_covariance),
        num_frames=len(detections),
        num_corners=num_corners,
    )


def world_up_from_cameras(
    per_camera_up: dict[str, np.ndarray],
    camera_rotations: dict[str, np.ndarray],
    max_disagreement_deg: float = MAX_CAMERA_DISAGREEMENT_DEG,
) -> tuple[np.ndarray, list[str], str]:
    """(up unit vector in the world frame, cameras used, gravity source).

    per_camera_up is up in each camera's frame; camera_rotations the world -> camera
    rotations of the current calibration (CameraCalibration.rotation_matrix), keyed
    alike. Each camera's up is mapped to the world as R.T @ up. The consensus is the
    mean of the largest group of cameras within max_disagreement_deg of one another;
    cameras further than that from it are dropped. Source is "measured" when a strict
    majority of the mapped cameras agree (or a single camera has a file), otherwise
    (no camera, or no majority) WORLD_UP with source "body".
    """
    camera_keys = sorted(key for key in per_camera_up if key in camera_rotations)
    if not camera_keys:
        return WORLD_UP.copy(), [], GRAVITY_SOURCE_BODY
    ups = np.array([
        _unit(np.asarray(camera_rotations[key], dtype=np.float64).T @ np.asarray(per_camera_up[key], dtype=np.float64))
        for key in camera_keys
    ])
    pairwise_deg = np.degrees(np.arccos(np.clip(ups @ ups.T, -1.0, 1.0)))
    agreeing = pairwise_deg <= max_disagreement_deg
    seed = int(np.lexsort((pairwise_deg.sum(axis=1), -agreeing.sum(axis=1)))[0])
    consensus = _unit(ups[agreeing[seed]].mean(axis=0))
    from_consensus_deg = np.degrees(np.arccos(np.clip(ups @ consensus, -1.0, 1.0)))
    used = from_consensus_deg <= max_disagreement_deg
    for key, angle_deg in zip(camera_keys, from_consensus_deg):
        if angle_deg > max_disagreement_deg:
            logger.warning(
                "Gravity: camera %s is %.2f deg off the other cameras (moved?); re-run measure_gravity.py",
                key, angle_deg,
            )
    if len(camera_keys) > 1 and 2 * int(used.sum()) <= len(camera_keys):
        logger.warning("Gravity: no majority of cameras agree; using the body vertical")
        return WORLD_UP.copy(), [], GRAVITY_SOURCE_BODY
    return (
        _unit(ups[used].mean(axis=0)),
        [key for key, is_used in zip(camera_keys, used) if is_used],
        GRAVITY_SOURCE_MEASURED,
    )


def load_world_up_for_provider(
    provider: MultiCameraPoseProvider, directory: Path | None = None
) -> tuple[np.ndarray | None, str]:
    """Measured up in the provider's current world frame from the gravity files of the cameras
    in its installed calibration: (up_world, "measured"), or (None, "body") without a
    calibration or a usable file. Never raises: deadlift activation falls back to the body."""
    try:
        calibration = provider.calibration
        if calibration is None:
            return None, GRAVITY_SOURCE_BODY
        camera_keys = provider.camera_keys
        per_camera_up = load_rig_gravity(
            {camera_id: camera_keys.get(camera_id, camera_id) for camera_id in calibration.cameras}, directory
        )
        rotations = {camera_id: camera.rotation_matrix for camera_id, camera in calibration.cameras.items()}
        up_world, used, source = world_up_from_cameras(per_camera_up, rotations)
    except Exception as exc:  # activation must never fail over a gravity file
        logger.warning("Gravity: measured gravity unavailable (%s); using the body vertical", exc)
        return None, GRAVITY_SOURCE_BODY
    if source != GRAVITY_SOURCE_MEASURED:
        return None, GRAVITY_SOURCE_BODY
    tilt_deg = math.degrees(math.acos(float(np.clip(up_world @ WORLD_UP, -1.0, 1.0))))
    logger.info("Gravity: measured from cameras %s, %.2f deg off the body vertical", used, tilt_deg)
    return up_world, source
