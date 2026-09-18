"""
T-pose camera calibration: a canonical 3D T-pose scaled to the user's height is matched
to each camera's averaged 2D keypoints with cv2.solvePnP. Height gives absolute scale.

World frame: origin at hip midpoint, meters, right-handed. X = subject's left,
Y = down, +Z = subject's BACK (the front camera sits at -Z; forward is -Z).

Also owns the calibration file schema: per-camera real intrinsics + lens distortion
(~/.nowva/intrinsics_<camera_key>.json) and keypoint undistortion for the triangulator.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from biomechanics.pose.base import PoseEstimator
from biomechanics.utils.types import CocoKeypoints as CK

logger = logging.getLogger(__name__)

# Anthropometric segment-to-height ratios (biomechanics literature averages).
SEGMENT_RATIOS = {
    "head_to_shoulder": 0.130,
    "shoulder_width_half": 0.105,
    "torso": 0.290,
    "hip_width_half": 0.0725,
    "femur": 0.245,
    "tibia": 0.235,
    "upper_arm": 0.175,
    "forearm": 0.150,
}

NUM_TPOSE_KEYPOINTS = 17  # COCO-17; skeleton foot keypoints 17-20 are not in the model

# Per-machine calibration files: intrinsics_<camera_key>.json and rig_calibration_cams_<ids>.json.
NOWVA_CALIBRATION_DIR = Path.home() / ".nowva"
NUM_DISTORTION_COEFFS = 5  # OpenCV order: k1, k2, p1, p2, k3
UNDISTORT_MAX_ITERATIONS = 50
UNDISTORT_CONVERGENCE_PX = 1e-5

# CalibrationResult.world_anchor: "person" = origin at the lifter's hip midpoint (T-pose or
# person calibration); "board" = factory ChArUco rig calibration, origin at the mean board
# centre, which person_calibration.refine re-anchors to the lifter.
WORLD_ANCHOR_PERSON = "person"
WORLD_ANCHOR_BOARD = "board"

# Keypoint gate on the per-camera mean raw RTMPose score (missed frames count as 0).
# Raw scores: no-person images peak at 0.16-0.27; visible keypoints on tracked crops
# average 0.77-0.99 with single-frame minimum 0.61 (data/squats.mov). A keypoint passes
# only if it is detected at visible-keypoint confidence in most frames.
MIN_MEAN_KEYPOINT_SCORE = 0.5
MIN_SOLVE_KEYPOINTS = 6
HIGH_REPROJECTION_ERROR_PX = 10.0

# T-pose check, per frame in 2D. Wrist height offset from its shoulder is normalized
# by torso length (vertical, so camera yaw does not shrink it): 0.5 torso ~ arms
# within ~26 deg of horizontal (0.5 * 0.290 / (0.175 + 0.150) = sin 26 deg).
TPOSE_MAX_WRIST_DROP_TO_TORSO_RATIO = 0.5
# Wrist reach beyond its shoulder along the shoulder line, normalized by shoulder span
# (both shrink equally with yaw): T-pose ~1.55, arms at sides ~0.
TPOSE_MIN_WRIST_REACH_TO_SHOULDER_SPAN_RATIO = 0.75
TPOSE_MIN_FRAME_FRACTION = 0.5
TPOSE_REQUIRED_KEYPOINTS = [
    CK.LEFT_SHOULDER, CK.RIGHT_SHOULDER, CK.LEFT_WRIST, CK.RIGHT_WRIST, CK.LEFT_HIP, CK.RIGHT_HIP,
]
MIN_SHOULDER_SPAN_PX = 1.0


@dataclass
class CameraCalibration:
    """Calibration result for a single camera."""
    camera_id: str
    projection_matrix: np.ndarray
    intrinsic_matrix: np.ndarray
    rotation_matrix: np.ndarray
    translation_vector: np.ndarray
    reprojection_error: float
    resolution: tuple[int, int]
    distortion_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(NUM_DISTORTION_COEFFS))


@dataclass
class CalibrationResult:
    """Complete multi-camera calibration."""
    cameras: dict[str, CameraCalibration] = field(default_factory=dict)
    tpose_model_3d: np.ndarray | None = None
    athlete_height_m: float = 0.0
    timestamp: str = ""
    world_anchor: str = WORLD_ANCHOR_PERSON
    # Robust RMS reprojection residual at solve time, the drift monitor's baseline; 0 = not measured.
    rms_reprojection_px: float = 0.0
    # Field refinements: `timestamp` of the factory / first calibration they descend from ("" otherwise).
    source_timestamp: str = ""


def undistort_keypoints(points_px: np.ndarray, calibration: CameraCalibration) -> np.ndarray:
    """
    Remove lens distortion from (N, 2) pixel coordinates, staying in the same K, so the
    result is what projection_matrix predicts. Identity when every coefficient is zero.
    """
    points_px = np.asarray(points_px, dtype=np.float64)
    if len(points_px) == 0 or not np.any(calibration.distortion_coeffs):
        return points_px.copy()
    criteria = (
        cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS,
        UNDISTORT_MAX_ITERATIONS,
        UNDISTORT_CONVERGENCE_PX,
    )
    undistorted = cv2.undistortPointsIter(
        points_px.reshape(-1, 1, 2),
        calibration.intrinsic_matrix,
        calibration.distortion_coeffs,
        None,
        calibration.intrinsic_matrix,
        criteria,
    )
    return undistorted.reshape(-1, 2)


def intrinsics_path(camera_key: str, directory: Path | None = None) -> Path:
    return (directory or NOWVA_CALIBRATION_DIR) / f"intrinsics_{camera_key}.json"


def rig_calibration_path(device_ids: list[int], directory: Path | None = None) -> Path:
    """The per-rig calibration file pipeline_process loads at session start."""
    camera_ids = "-".join(str(device_id) for device_id in device_ids)
    return (directory or NOWVA_CALIBRATION_DIR) / f"rig_calibration_cams_{camera_ids}.json"


def save_intrinsics(
    camera_key: str,
    resolution: tuple[int, int],
    intrinsic_matrix: np.ndarray,
    distortion_coeffs: np.ndarray,
    rms_reprojection_px: float,
    directory: Path | None = None,
) -> Path:
    path = intrinsics_path(camera_key, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "camera_key": camera_key,
        "resolution": list(resolution),
        "intrinsic_matrix": np.asarray(intrinsic_matrix, dtype=np.float64).tolist(),
        "distortion_coeffs": np.asarray(distortion_coeffs, dtype=np.float64).ravel().tolist(),
        "rms_reprojection_px": float(rms_reprojection_px),
        "timestamp": datetime.now().isoformat(),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Intrinsics for camera %s saved to %s", camera_key, path)
    return path


def load_intrinsics(
    camera_key: str, resolution: tuple[int, int], directory: Path | None = None
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Return (K, distortion_coeffs) calibrated for this camera, or None when there is no
    file or it was calibrated at another resolution (a different sensor mode may crop,
    so K is not rescaled; recalibrate at the capture resolution instead).
    """
    path = intrinsics_path(camera_key, directory)
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    if tuple(data["resolution"]) != tuple(resolution):
        logger.warning(
            "Intrinsics %s were calibrated at %s but capture runs at %s; ignoring them",
            path, tuple(data["resolution"]), tuple(resolution),
        )
        return None
    return (
        np.array(data["intrinsic_matrix"], dtype=np.float64),
        np.array(data["distortion_coeffs"], dtype=np.float64),
    )


def load_rig_intrinsics(
    camera_keys: dict[str, str], resolution: tuple[int, int], directory: Path | None = None
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """camera_id -> (K, distortion_coeffs) for every camera_id -> camera_key with saved intrinsics."""
    intrinsics = {}
    for camera_id, camera_key in camera_keys.items():
        loaded = load_intrinsics(camera_key, resolution, directory)
        if loaded is not None:
            intrinsics[camera_id] = loaded
    return intrinsics


def tpose_frame_mask(points_xy: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """
    Per-frame T-pose check on 2D keypoints (F, 17, 2) with scores (F, 17): both wrists
    near their shoulder's height and reaching well outside the shoulders. Frames missing
    a shoulder, wrist or hip are not T-pose frames.
    """
    all_detected = np.all(scores[:, TPOSE_REQUIRED_KEYPOINTS] > 0.0, axis=1)

    shoulder_l = points_xy[:, CK.LEFT_SHOULDER]
    shoulder_r = points_xy[:, CK.RIGHT_SHOULDER]
    wrist_l = points_xy[:, CK.LEFT_WRIST]
    wrist_r = points_xy[:, CK.RIGHT_WRIST]
    shoulder_mid = (shoulder_l + shoulder_r) / 2.0
    hip_mid = (points_xy[:, CK.LEFT_HIP] + points_xy[:, CK.RIGHT_HIP]) / 2.0

    torso_length_px = np.linalg.norm(hip_mid - shoulder_mid, axis=1)
    shoulder_vec = shoulder_l - shoulder_r
    shoulder_span_px = np.linalg.norm(shoulder_vec, axis=1)
    measurable = all_detected & (shoulder_span_px > MIN_SHOULDER_SPAN_PX) & (torso_length_px > 0.0)

    safe_span = np.where(measurable, shoulder_span_px, 1.0)
    safe_torso = np.where(measurable, torso_length_px, 1.0)
    outward_l = shoulder_vec / safe_span[:, None]

    reach_l_ratio = np.sum((wrist_l - shoulder_l) * outward_l, axis=1) / safe_span
    reach_r_ratio = np.sum((shoulder_r - wrist_r) * outward_l, axis=1) / safe_span
    drop_l_ratio = np.abs(wrist_l[:, 1] - shoulder_l[:, 1]) / safe_torso
    drop_r_ratio = np.abs(wrist_r[:, 1] - shoulder_r[:, 1]) / safe_torso

    arms_level = (drop_l_ratio <= TPOSE_MAX_WRIST_DROP_TO_TORSO_RATIO) & (
        drop_r_ratio <= TPOSE_MAX_WRIST_DROP_TO_TORSO_RATIO
    )
    arms_out = (reach_l_ratio >= TPOSE_MIN_WRIST_REACH_TO_SHOULDER_SPAN_RATIO) & (
        reach_r_ratio >= TPOSE_MIN_WRIST_REACH_TO_SHOULDER_SPAN_RATIO
    )
    return measurable & arms_level & arms_out


def average_detected_keypoints(
    points_xy: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Average (F, 17, 2) keypoints over the frames where each keypoint was detected
    (score > 0); mean scores run over all frames, so misses count as 0.
    """
    detected = scores > 0.0
    detected_counts = detected.sum(axis=0)
    summed_xy = np.sum(points_xy * detected[..., None], axis=0)
    mean_xy = summed_xy / np.maximum(detected_counts, 1)[:, None]
    mean_scores = scores.mean(axis=0)
    return mean_xy, mean_scores


class TPoseCalibrator:
    """
    Calibrates multi-camera extrinsics from a T-pose and known height.

    Flow:
    1. Build a canonical T-pose 3D model scaled to user height.
    2. Detect 2D keypoints in each camera's T-pose frames (crop tracked per camera).
    3. Verify the subject holds a T-pose, then average keypoints per camera.
    4. Use cv2.solvePnP per camera to solve for rotation and translation.
    5. Compute projection matrices P = K @ [R|t].

    intrinsics maps camera_id -> (K, distortion_coeffs) from load_rig_intrinsics; cameras
    without an entry fall back to the guessed pinhole (f = focal_length_factor * width,
    centred, no distortion).
    """

    def __init__(
        self,
        pose_estimator: PoseEstimator,
        focal_length_factor: float = 0.8,
        intrinsics: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> None:
        self._estimator = pose_estimator
        self._focal_length_factor = focal_length_factor
        self._intrinsics = intrinsics or {}

    def build_tpose_model(self, height_m: float) -> np.ndarray:
        """
        Return the (17, 3) canonical T-pose in world coordinates: origin at hip midpoint,
        X = subject's left, Y down, +Z = subject's back; arms horizontal along X, legs
        straight down, every keypoint in the Z = 0 plane.
        """
        r = SEGMENT_RATIOS
        h = height_m

        head_to_shoulder = r["head_to_shoulder"] * h
        shoulder_half = r["shoulder_width_half"] * h
        torso = r["torso"] * h
        hip_half = r["hip_width_half"] * h
        femur = r["femur"] * h
        tibia = r["tibia"] * h
        upper_arm = r["upper_arm"] * h
        forearm = r["forearm"] * h

        # Vertical positions relative to hip midpoint (Y-down: negative = above hip)
        hip_y = 0.0
        shoulder_y = hip_y - torso
        nose_y = shoulder_y - head_to_shoulder
        knee_y = hip_y + femur
        ankle_y = hip_y + femur + tibia

        # Eye and ear positions (slight offsets from nose)
        eye_offset_y = 0.02 * h
        eye_offset_x = 0.015 * h
        ear_offset_x = 0.04 * h

        model = np.zeros((NUM_TPOSE_KEYPOINTS, 3), dtype=np.float64)

        # Head
        model[CK.NOSE] = [0.0, nose_y, 0.0]
        model[CK.LEFT_EYE] = [eye_offset_x, nose_y - eye_offset_y, 0.0]
        model[CK.RIGHT_EYE] = [-eye_offset_x, nose_y - eye_offset_y, 0.0]
        model[CK.LEFT_EAR] = [ear_offset_x, nose_y, 0.0]
        model[CK.RIGHT_EAR] = [-ear_offset_x, nose_y, 0.0]

        # Shoulders (X-left convention: left shoulder has positive X)
        model[CK.LEFT_SHOULDER] = [shoulder_half, shoulder_y, 0.0]
        model[CK.RIGHT_SHOULDER] = [-shoulder_half, shoulder_y, 0.0]

        # Arms in T-pose: extended horizontally along X axis
        model[CK.LEFT_ELBOW] = [shoulder_half + upper_arm, shoulder_y, 0.0]
        model[CK.RIGHT_ELBOW] = [-(shoulder_half + upper_arm), shoulder_y, 0.0]
        model[CK.LEFT_WRIST] = [shoulder_half + upper_arm + forearm, shoulder_y, 0.0]
        model[CK.RIGHT_WRIST] = [-(shoulder_half + upper_arm + forearm), shoulder_y, 0.0]

        # Hips
        model[CK.LEFT_HIP] = [hip_half, hip_y, 0.0]
        model[CK.RIGHT_HIP] = [-hip_half, hip_y, 0.0]

        # Legs straight down
        model[CK.LEFT_KNEE] = [hip_half, knee_y, 0.0]
        model[CK.RIGHT_KNEE] = [-hip_half, knee_y, 0.0]
        model[CK.LEFT_ANKLE] = [hip_half, ankle_y, 0.0]
        model[CK.RIGHT_ANKLE] = [-hip_half, ankle_y, 0.0]

        return model

    def _build_intrinsics(self, resolution: tuple[int, int]) -> np.ndarray:
        w, h = resolution
        f = self._focal_length_factor * w
        return np.array([
            [f, 0.0, w / 2.0],
            [0.0, f, h / 2.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

    def _camera_intrinsics(
        self, camera_id: str, resolution: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        if camera_id in self._intrinsics:
            K, dist_coeffs = self._intrinsics[camera_id]
            return np.asarray(K, dtype=np.float64), np.asarray(dist_coeffs, dtype=np.float64)
        logger.warning(
            "Camera %s: no calibrated intrinsics, guessing f = %.2f * width with no distortion "
            "(run scripts/tools/calibrate_cameras.py intrinsics)",
            camera_id, self._focal_length_factor,
        )
        return self._build_intrinsics(resolution), np.zeros(NUM_DISTORTION_COEFFS, dtype=np.float64)

    def _detect_keypoints(
        self, frames: list[np.ndarray], camera_id: str
    ) -> tuple[np.ndarray, np.ndarray]:
        # (F, 17, 2) positions and (F, 17) scores; frames without a person score 0.
        points_xy = np.zeros((len(frames), NUM_TPOSE_KEYPOINTS, 2), dtype=np.float64)
        scores = np.zeros((len(frames), NUM_TPOSE_KEYPOINTS), dtype=np.float64)

        for i, frame in enumerate(frames):
            skeleton = self._estimator.estimate(frame, camera_id)
            if skeleton is None:
                continue
            keypoints = skeleton.to_numpy()[:NUM_TPOSE_KEYPOINTS]
            points_xy[i] = keypoints[:, :2]
            scores[i] = keypoints[:, 2]

        return points_xy, scores

    def calibrate(
        self,
        frames: dict[str, list[np.ndarray]],
        height_m: float,
        resolution: tuple[int, int],
    ) -> CalibrationResult:
        """
        Run T-pose calibration on collected frames (camera_id -> T-pose frames).

        Cameras with too few detected keypoints are skipped. Raises ValueError if a
        camera sees a person who is not holding a T-pose in at least half the frames,
        and RuntimeError if no camera calibrates.
        """
        model_3d = self.build_tpose_model(height_m)

        result = CalibrationResult(
            tpose_model_3d=model_3d,
            athlete_height_m=height_m,
            timestamp=datetime.now().isoformat(),
        )

        for cam_id, cam_frames in frames.items():
            points_xy, scores = self._detect_keypoints(cam_frames, cam_id)
            avg_kpts, avg_confs = average_detected_keypoints(points_xy, scores)

            valid_indices = np.where(avg_confs > MIN_MEAN_KEYPOINT_SCORE)[0]

            if len(valid_indices) < MIN_SOLVE_KEYPOINTS:
                logger.warning(
                    "Camera %s: only %d valid keypoints (need %d+), skipping",
                    cam_id, len(valid_indices), MIN_SOLVE_KEYPOINTS,
                )
                continue

            tpose_fraction = float(np.mean(tpose_frame_mask(points_xy, scores)))
            if tpose_fraction < TPOSE_MIN_FRAME_FRACTION:
                raise ValueError(
                    f"Camera {cam_id}: subject is not holding a T-pose "
                    f"({tpose_fraction:.0%} of {len(cam_frames)} frames, need "
                    f"{TPOSE_MIN_FRAME_FRACTION:.0%}). Stand facing the front camera with "
                    f"both arms straight out at shoulder height."
                )

            object_points = model_3d[valid_indices].astype(np.float64)
            image_points = avg_kpts[valid_indices].astype(np.float64)
            K, dist_coeffs = self._camera_intrinsics(cam_id, resolution)

            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                K,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

            if not success:
                logger.warning("Camera %s: solvePnP failed", cam_id)
                continue

            # Refine with Levenberg-Marquardt
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points, image_points, K, dist_coeffs, rvec, tvec
            )

            R, _ = cv2.Rodrigues(rvec)
            Rt = np.hstack([R, tvec])
            P = K @ Rt

            # Compute reprojection error
            projected, _ = cv2.projectPoints(
                object_points, rvec, tvec, K, dist_coeffs
            )
            projected = projected.reshape(-1, 2)
            errors = np.linalg.norm(projected - image_points, axis=1)
            reproj_error = float(np.mean(errors))

            if reproj_error > HIGH_REPROJECTION_ERROR_PX:
                logger.warning(
                    "Camera %s: high reprojection error %.1f px", cam_id, reproj_error
                )

            logger.info(
                "Camera %s: reprojection error = %.1f px (%d keypoints)",
                cam_id, reproj_error, len(valid_indices),
            )

            result.cameras[cam_id] = CameraCalibration(
                camera_id=cam_id,
                projection_matrix=P,
                intrinsic_matrix=K,
                rotation_matrix=R,
                translation_vector=tvec,
                reprojection_error=reproj_error,
                resolution=resolution,
                distortion_coeffs=dist_coeffs,
            )

        if not result.cameras:
            raise RuntimeError("Calibration failed: no cameras calibrated successfully")

        return result

    @staticmethod
    def save_calibration(result: CalibrationResult, path: str) -> None:
        """Save calibration result to JSON."""
        data = {
            "athlete_height_m": result.athlete_height_m,
            "timestamp": result.timestamp,
            "world_anchor": result.world_anchor,
            "rms_reprojection_px": result.rms_reprojection_px,
            "source_timestamp": result.source_timestamp,
            "tpose_model_3d": result.tpose_model_3d.tolist() if result.tpose_model_3d is not None else None,
            "cameras": {},
        }

        for cam_id, cam in result.cameras.items():
            data["cameras"][cam_id] = {
                "camera_id": cam.camera_id,
                "projection_matrix": cam.projection_matrix.tolist(),
                "intrinsic_matrix": cam.intrinsic_matrix.tolist(),
                "rotation_matrix": cam.rotation_matrix.tolist(),
                "translation_vector": cam.translation_vector.tolist(),
                "reprojection_error": cam.reprojection_error,
                "resolution": list(cam.resolution),
                "distortion_coeffs": np.asarray(cam.distortion_coeffs).ravel().tolist(),
            }

        # Written next to the target and renamed into place, so a crash mid-write never
        # leaves a truncated calibration where the pipeline expects a valid one.
        temporary_path = f"{path}.tmp"
        try:
            with open(temporary_path, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(temporary_path, path)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)

        logger.info("Calibration saved to %s", path)

    @staticmethod
    def load_calibration(path: str) -> CalibrationResult:
        """Load calibration result from JSON."""
        with open(path) as f:
            data = json.load(f)

        result = CalibrationResult(
            athlete_height_m=data["athlete_height_m"],
            timestamp=data["timestamp"],
            tpose_model_3d=np.array(data["tpose_model_3d"]) if data.get("tpose_model_3d") else None,
            world_anchor=data.get("world_anchor", WORLD_ANCHOR_PERSON),
            rms_reprojection_px=data.get("rms_reprojection_px", 0.0),
            source_timestamp=data.get("source_timestamp", ""),
        )

        for cam_id, cam_data in data["cameras"].items():
            result.cameras[cam_id] = CameraCalibration(
                camera_id=cam_data["camera_id"],
                projection_matrix=np.array(cam_data["projection_matrix"]),
                intrinsic_matrix=np.array(cam_data["intrinsic_matrix"]),
                rotation_matrix=np.array(cam_data["rotation_matrix"]),
                translation_vector=np.array(cam_data["translation_vector"]),
                reprojection_error=cam_data["reprojection_error"],
                resolution=tuple(cam_data["resolution"]),
                distortion_coeffs=np.array(
                    cam_data.get("distortion_coeffs", [0.0] * NUM_DISTORTION_COEFFS), dtype=np.float64
                ),
            )

        logger.info("Calibration loaded from %s (%d cameras)", path, len(result.cameras))
        return result
