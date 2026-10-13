"""
Measured gravity for the multi-camera rig.

The rig's world Y is the median standing hip-to-ankle direction, which can sit
2-4° off true gravity. A floor ChArUco measurement (scripts/tools/measure_gravity.py)
stores physical gravity, pointing down, in each camera's own frame:
~/.nowva/gravity_<camera_key>.json = {"gravity": [gx, gy, gz]}. Mapped into
the world through each camera's rotation (OpenCV, world -> camera:
g_world = R^T g_cam), up = -gravity, Y-down world frame.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from biomechanics.triangulation.calibration import CalibrationResult

logger = logging.getLogger(__name__)

GRAVITY_DIR = Path.home() / ".nowva"
# A camera whose gravity disagrees with the others by more than this has moved.
MAX_CAMERA_DISAGREEMENT_DEG = 1.0
MIN_VECTOR_NORM = 1e-9


def gravity_path(camera_key: str, directory: Path | None = None) -> Path:
    return (directory or GRAVITY_DIR) / f"gravity_{camera_key}.json"


def load_camera_gravity(path: Path) -> np.ndarray | None:
    """Unit gravity in the camera's frame, or None when the file is absent or unreadable."""
    if not path.exists():
        return None
    try:
        vector = np.asarray(json.loads(path.read_text())["gravity"], dtype=np.float64)
    except (OSError, ValueError, KeyError, TypeError):
        logger.warning("[GRAVITY] %s is unreadable — ignored", path)
        return None
    norm = float(np.linalg.norm(vector))
    if vector.shape != (3,) or not math.isfinite(norm) or norm < MIN_VECTOR_NORM:
        logger.warning("[GRAVITY] %s holds no usable vector — ignored", path)
        return None
    return vector / norm


def _angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip(np.dot(first, second), -1.0, 1.0))))


def measured_up(
    calibration: CalibrationResult,
    camera_keys: dict[str, str],
    directory: Path | None = None,
) -> np.ndarray | None:
    """World-frame physical up from every calibrated camera with a gravity file,
    or None when none has one. camera_keys maps calibration camera ids to file keys."""
    world_gravities: list[np.ndarray] = []
    for camera_id, camera in calibration.cameras.items():
        gravity_camera = load_camera_gravity(gravity_path(camera_keys.get(camera_id, camera_id), directory))
        if gravity_camera is not None:
            world_gravities.append(np.asarray(camera.rotation_matrix, dtype=np.float64).T @ gravity_camera)
    if not world_gravities:
        return None
    # The largest group of cameras that agree with one another; a moved camera
    # cannot drag a mean consensus away from the others.
    agreeing = max(
        (
            [other for other in world_gravities if _angle_deg(candidate, other) <= MAX_CAMERA_DISAGREEMENT_DEG]
            for candidate in world_gravities
        ),
        key=len,
    )
    if len(agreeing) < len(world_gravities):
        logger.warning(
            "[GRAVITY] %d of %d cameras disagree by more than %.1f° (moved) — dropped",
            len(world_gravities) - len(agreeing), len(world_gravities), MAX_CAMERA_DISAGREEMENT_DEG,
        )
    gravity_world = np.mean(agreeing, axis=0)
    return -gravity_world / np.linalg.norm(gravity_world)
