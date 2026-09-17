"""Three-camera pinhole rig and the two calibration modes fed to the real DLTTriangulator.

perfect: true projection matrices. tpose: the real TPoseCalibrator (solvePnP against the canonical
anthropometric T-pose, guessed f = 0.8 w) run on simulated detections of the TRUE body's T-pose seen by the
true cameras (true f ~= 0.72 w, off-centre principal point), so the systematic calibration error is realistic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from biomechanics.triangulation.calibration import CalibrationResult, CameraCalibration, TPoseCalibrator
from biomechanics.utils.types import Keypoint2D, Skeleton2D

from .body import FLOOR_Y_M, HEIGHT_M, tpose_skeleton

NUM_DETECTED_KPTS = 19


@dataclass
class RigConfig:
    resolution: tuple[int, int] = (1280, 720)
    azimuths_deg: tuple[float, ...] = (0.0, -40.0, 40.0)
    azimuth_jitter_deg: float = 3.0
    distance_m: float = 3.5
    distance_jitter_m: float = 0.25
    height_above_floor_m: float = 1.1
    height_jitter_m: float = 0.1
    aim_y_m: float = 0.05
    aim_jitter_m: float = 0.05
    roll_jitter_deg: float = 1.0
    true_focal_factor: float = 0.72
    focal_jitter_ratio: float = 0.03
    principal_jitter_px: float = 8.0
    calibration_focal_factor: float = 0.8
    tpose_frames: int = 30


@dataclass
class Camera:
    cam_id: str
    intrinsic: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    position: np.ndarray
    resolution: tuple[int, int]
    azimuth_deg: float

    @property
    def projection_matrix(self) -> np.ndarray:
        return self.intrinsic @ np.hstack([self.rotation, self.translation[:, None]])

    def project(self, points_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cam = points_world @ self.rotation.T + self.translation
        depth = cam[..., 2]
        pix = cam @ self.intrinsic.T
        return pix[..., :2] / pix[..., 2:3], depth


def build_rig(seed: int, cfg: RigConfig | None = None) -> list[Camera]:
    cfg = cfg or RigConfig()
    rng = np.random.default_rng([seed, 101])
    width, height = cfg.resolution
    cameras = []
    for idx, azimuth in enumerate(cfg.azimuths_deg):
        az = math.radians(azimuth + rng.uniform(-cfg.azimuth_jitter_deg, cfg.azimuth_jitter_deg))
        dist = cfg.distance_m + rng.uniform(-cfg.distance_jitter_m, cfg.distance_jitter_m)
        cam_y = FLOOR_Y_M - (cfg.height_above_floor_m + rng.uniform(-cfg.height_jitter_m, cfg.height_jitter_m))
        position = np.array([dist * math.sin(az), cam_y, -dist * math.cos(az)])
        target = np.array([0.0, cfg.aim_y_m, 0.0]) + rng.uniform(-cfg.aim_jitter_m, cfg.aim_jitter_m, 3)
        z_axis = target - position
        z_axis /= np.linalg.norm(z_axis)
        x_axis = np.cross(np.array([0.0, 1.0, 0.0]), z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        roll = math.radians(rng.uniform(-cfg.roll_jitter_deg, cfg.roll_jitter_deg))
        x_rolled = math.cos(roll) * x_axis + math.sin(roll) * y_axis
        y_rolled = -math.sin(roll) * x_axis + math.cos(roll) * y_axis
        rotation = np.stack([x_rolled, y_rolled, z_axis])
        focal = cfg.true_focal_factor * width * (1.0 + rng.uniform(-cfg.focal_jitter_ratio, cfg.focal_jitter_ratio))
        principal = np.array([width / 2.0, height / 2.0]) + rng.uniform(-cfg.principal_jitter_px,
                                                                         cfg.principal_jitter_px, 2)
        intrinsic = np.array([[focal, 0.0, principal[0]], [0.0, focal, principal[1]], [0.0, 0.0, 1.0]])
        cameras.append(Camera(cam_id=str(idx), intrinsic=intrinsic, rotation=rotation,
                              translation=-rotation @ position, position=position, resolution=cfg.resolution,
                              azimuth_deg=math.degrees(az)))
    return cameras


def perfect_calibration(cameras: list[Camera]) -> CalibrationResult:
    result = CalibrationResult(athlete_height_m=HEIGHT_M, timestamp="harness-perfect")
    for cam in cameras:
        result.cameras[cam.cam_id] = CameraCalibration(
            camera_id=cam.cam_id, projection_matrix=cam.projection_matrix, intrinsic_matrix=cam.intrinsic,
            rotation_matrix=cam.rotation, translation_vector=cam.translation[:, None], reprojection_error=0.0,
            resolution=cam.resolution)
    return result


@dataclass
class _ReplayEstimator:
    """Duck-typed PoseEstimator: frames are tiny arrays [camera_index, frame_index] into precomputed detections."""

    detections: dict[int, np.ndarray]
    calls: list[int] = field(default_factory=list)

    def estimate(self, frame: np.ndarray, camera_id: int = 0) -> Skeleton2D | None:
        cam_index, frame_index = int(frame[0]), int(frame[1])
        kpts = self.detections[cam_index][frame_index]
        return Skeleton2D(keypoints=[Keypoint2D(x=float(k[0]), y=float(k[1]), confidence=float(k[2]))
                                     for k in kpts], timestamp=0.0, frame_index=frame_index)


def tpose_calibration(cameras: list[Camera], detect_fn, seed: int, cfg: RigConfig | None = None
                      ) -> tuple[CalibrationResult, dict]:
    """Run the production TPoseCalibrator on simulated T-pose detections.

    detect_fn(camera, points_world (F, 21, 3), frame_times (F,), rng) -> (F, 19, 3) decoded detections.
    """
    cfg = cfg or RigConfig()
    rng = np.random.default_rng([seed, 202])
    frames = cfg.tpose_frames
    tpose = tpose_skeleton()
    sway = np.zeros((frames, 1, 3))
    sway[:, 0, 0] = 0.002 * np.sin(np.linspace(0, 2.0, frames))
    sway[:, 0, 2] = 0.003 * np.sin(np.linspace(0, 1.3, frames))
    points = tpose[None] + sway
    times = np.arange(frames) / 30.0
    detections = {idx: detect_fn(cam, points, times, rng) for idx, cam in enumerate(cameras)}
    estimator = _ReplayEstimator(detections)
    calibrator = TPoseCalibrator(pose_estimator=estimator, focal_length_factor=cfg.calibration_focal_factor)
    frame_dict = {cam.cam_id: [np.array([idx, f]) for f in range(frames)] for idx, cam in enumerate(cameras)}
    result = calibrator.calibrate(frames=frame_dict, height_m=HEIGHT_M, resolution=cfg.resolution)
    info = {cam_id: {"pnp_reprojection_px": round(c.reprojection_error, 2)} for cam_id, c in result.cameras.items()}
    for cam in cameras:
        cal = result.cameras.get(cam.cam_id)
        if cal is None:
            continue
        # camera centre error in the calibrated vs true frame
        centre_cal = -cal.rotation_matrix.T @ cal.translation_vector.ravel()
        info[cam.cam_id]["centre_error_m"] = round(float(np.linalg.norm(centre_cal - cam.position)), 3)
        rel = cal.rotation_matrix @ cam.rotation.T
        info[cam.cam_id]["rotation_error_deg"] = round(float(np.degrees(np.arccos(np.clip(
            (np.trace(rel) - 1) / 2, -1, 1)))), 2)
    return result, info
