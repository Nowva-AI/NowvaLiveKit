"""Three-camera pinhole rig and the calibration modes fed to the real DLTTriangulator.

perfect: true projection matrices. tpose: the real TPoseCalibrator (solvePnP against the canonical
anthropometric T-pose, guessed f = 0.8 w) run on simulated detections of the TRUE body's T-pose seen by the
true cameras (true f ~= 0.72 w, off-centre principal point), so the systematic calibration error is realistic.
person_ba*: the real PersonCalibrator (bundle adjustment, no T-pose) on the harness's own noisy detections of a
walk-in + 2 reps, with the true intrinsics or a +-10 % focal error, metric scale from the bar or the height prior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from biomechanics.triangulation.calibration import CalibrationResult, CameraCalibration, TPoseCalibrator
from biomechanics.triangulation.person_calibration import PersonCalibrator
from biomechanics.utils.types import BarbellDetection, Keypoint2D, Skeleton2D

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


@dataclass
class PersonBAConfig:
    focal_scale: float = 1.0  # intrinsics handed to the calibrator = true K with both focal lengths x this
    scale_source: str = "bar"  # "bar": bar_length_m given; "height": height only
    bar_length_m: float = 2.2
    capture_scenario: str = "walkout"  # walk-in (0.5 m + 20 deg turn) then reps
    capture_reps: int = 2
    capture_tail_s: float = 0.5
    capture_seed_offset: int = 1000
    start_from_tpose: bool = False  # refine() the production T-pose calibration instead of the essential-matrix start


PERSON_BA_MODES: dict[str, PersonBAConfig] = {
    "person_ba": PersonBAConfig(),
    "person_ba_from_tpose": PersonBAConfig(start_from_tpose=True),
    "person_ba_height": PersonBAConfig(scale_source="height"),
    "person_ba_k+10": PersonBAConfig(focal_scale=1.1),
    "person_ba_k-10": PersonBAConfig(focal_scale=0.9),
    "person_ba_k+10_height": PersonBAConfig(focal_scale=1.1, scale_source="height"),
    "person_ba_k-10_height": PersonBAConfig(focal_scale=0.9, scale_source="height"),
}


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))))


def triangulate_noise_free(cameras: list[Camera], calibration: CalibrationResult, points_world: np.ndarray
                           ) -> np.ndarray:
    """TRUE noise-free projections of points (..., 3) triangulated (plain DLT) with the calibration under test."""
    rows = []
    for cam in cameras:
        uv, _ = cam.project(points_world)
        mat = calibration.cameras[cam.cam_id].projection_matrix
        rows.append(uv[..., 0:1] * mat[2] - mat[0])
        rows.append(uv[..., 1:2] * mat[2] - mat[1])
    _, _, vt = np.linalg.svd(np.stack(rows, axis=-2))
    return vt[..., -1, :3] / vt[..., -1, 3:4]


def _yaw_gauge(points_calibrated: np.ndarray, points_truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # best yaw-about-vertical + translation with truth ~= rotation @ calibrated + shift (no tilt, no scale)
    a = points_calibrated.reshape(-1, 3)
    b = points_truth.reshape(-1, 3)
    a_c, b_c = a - a.mean(0), b - b.mean(0)
    angle = math.atan2(np.sum(a_c[:, 0] * b_c[:, 2] - a_c[:, 2] * b_c[:, 0]),
                       np.sum(a_c[:, 0] * b_c[:, 0] + a_c[:, 2] * b_c[:, 2]))
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])
    return rotation, b.mean(0) - rotation @ a.mean(0)


def person_ba_calibration(cameras: list[Camera], views: list[dict[str, Skeleton2D]],
                          bar_ends: list[dict[str, BarbellDetection]], capture_truth_world: np.ndarray,
                          cfg: PersonBAConfig, initial: CalibrationResult | None = None
                          ) -> tuple[CalibrationResult, dict]:
    """Run the production PersonCalibrator on a simulated capture and express the result in the harness frame.

    The calibrator anchors its world frame on the lifter (X = the standing subject's left, origin = standing hip
    midpoint, averaged over whichever standing frames it used); the harness truth frame is anchored on the T-pose
    spot. Heading and origin are a gauge choice, not an error, so they are removed: the capture's TRUE joints
    (F, 21, 3) are projected noise-free, triangulated with the solved cameras, and the best yaw-about-vertical +
    translation onto the truth is applied. Tilt of the solved vertical, relative camera poses and scale are NOT
    corrected and stay in every metric.
    """
    intrinsics = {}
    for cam in cameras:
        intrinsic = cam.intrinsic.copy()
        intrinsic[0, 0] *= cfg.focal_scale
        intrinsic[1, 1] *= cfg.focal_scale
        intrinsics[cam.cam_id] = (intrinsic, np.zeros(5))
    use_bar = cfg.scale_source == "bar"
    calibrator = PersonCalibrator(intrinsics, cameras[0].resolution,
                                  bar_length_m=cfg.bar_length_m if use_bar else None, height_m=HEIGHT_M)
    solved = calibrator.calibrate(views, bar_ends=bar_ends if use_bar else None, initial=initial)

    body_points = capture_truth_world[:, 5:NUM_DETECTED_KPTS]
    in_calibrated = triangulate_noise_free(cameras, solved.calibration, body_points)
    gauge_rotation, gauge_shift = _yaw_gauge(in_calibrated, body_points)  # harness = rotation @ calibrated + shift
    # working-volume scale: solved / true distance from each joint to the frame's joint centroid
    spread_solved = np.linalg.norm(in_calibrated - in_calibrated.mean(axis=1, keepdims=True), axis=-1).sum()
    spread_truth = np.linalg.norm(body_points - body_points.mean(axis=1, keepdims=True), axis=-1).sum()

    result = CalibrationResult(athlete_height_m=solved.calibration.athlete_height_m, timestamp="harness-person-ba")
    info: dict = {"rms_reprojection_px": round(solved.rms_reprojection_px, 3),
                  "initial_rms_reprojection_px": round(solved.initial_rms_reprojection_px, 3),
                  "scale_source": solved.scale_source, "frames_used": solved.frames_used,
                  "capture_frames": len(views), "standing_frames": solved.standing_frames,
                  "solve_time_s": round(solved.solve_time_s, 3),
                  "gauge_yaw_deg": round(math.degrees(math.atan2(gauge_rotation[2, 0], gauge_rotation[0, 0])), 2),
                  "scale_error_pct": round(100.0 * (spread_solved / spread_truth - 1.0), 3),
                  "bone_lengths_m": {k: round(v, 4) for k, v in solved.bone_lengths_m.items()}}
    for cam in cameras:
        cal = solved.calibration.cameras[cam.cam_id]
        rotation = cal.rotation_matrix @ gauge_rotation.T
        translation = cal.translation_vector.ravel() - rotation @ gauge_shift
        result.cameras[cam.cam_id] = CameraCalibration(
            camera_id=cam.cam_id, projection_matrix=cal.intrinsic_matrix @ np.hstack([rotation, translation[:, None]]),
            intrinsic_matrix=cal.intrinsic_matrix, rotation_matrix=rotation, translation_vector=translation[:, None],
            reprojection_error=cal.reprojection_error, resolution=cam.resolution)
        centre = -rotation.T @ translation
        info[cam.cam_id] = {"reprojection_rms_px": round(cal.reprojection_error, 2),
                            "centre_error_m": round(float(np.linalg.norm(centre - cam.position)), 4),
                            "rotation_error_deg": round(_rotation_angle_deg(rotation @ cam.rotation.T), 3)}
    # where the TRUE vertical lands in the calibrated frame, seen through each camera (tilt of the solved Y axis)
    down = np.array([0.0, 1.0, 0.0])
    tilts = [np.degrees(np.arccos(np.clip((result.cameras[cam.cam_id].rotation_matrix.T @ cam.rotation @ down)[1],
                                          -1, 1))) for cam in cameras]
    info["vertical_tilt_deg"] = round(float(np.mean(tilts)), 3)
    return result, info
