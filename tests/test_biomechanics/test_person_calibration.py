"""
Tests for person-based extrinsic calibration on a synthetic 3-camera rig (3.5 m, yaw
-40/0/+40 deg, 1280x720, f = 0.75 w) watching a rigid-segment lifter walk in and squat:
camera recovery, scale sources, initialisation, world frame, robustness, drift monitor.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from biomechanics.pose.base import PoseEstimator  # noqa: E402
from biomechanics.triangulation.calibration import (  # noqa: E402
    SEGMENT_RATIOS,
    CalibrationResult,
    CameraCalibration,
    TPoseCalibrator,
)
from biomechanics.triangulation.person_calibration import (  # noqa: E402
    MAX_FRAMES,
    PersonCalibrationResult,
    PersonCalibrator,
    _camera_landmarks,
    _consensus_rigid_fit,
    _label_bar_ends_by_shoulders,
    reprojection_health_px,
    world_frame_from_standing,
)
from biomechanics.triangulation.triangulator import DLTTriangulator  # noqa: E402
from biomechanics.utils.types import BarbellDetection, Keypoint2D, MultiViewPose, Skeleton2D, Skeleton3D  # noqa: E402
from biomechanics.utils.types import CocoKeypoints as CK  # noqa: E402

RESOLUTION = (1280, 720)
FOCAL_LENGTH_FACTOR = 0.75
CAMERA_DISTANCE_M = 3.5
CAMERA_HEIGHT_Y_M = 0.1  # Y-down: cameras 0.1 m below the standing hips
CAMERA_YAWS_DEG = {"0": 0.0, "1": -40.0, "2": 40.0}
HEIGHT_M = 1.80
BAR_LENGTH_M = 2.2
NUM_KEYPOINTS = 21
BAR_POINTS = (21, 22)  # generator output: 21 keypoints + subject-left / subject-right bar end
DETECTED_SCORE = 0.9
LOW_SCORE = 0.2
WALK_IN_FRAMES = 30
TPOSE_FRAMES = 3
SQUAT_FRAMES = 50
MAX_KNEE_FLEXION_DEG = 110.0

# A body whose proportions differ from the calibration model (like a real lifter)
LIFTER_RATIOS = {"femur": 0.250, "tibia": 0.242, "hip_width_half": 0.068, "torso": 0.300,
                 "shoulder_width_half": 0.110, "upper_arm": 0.182, "forearm": 0.146}
FOOT_LENGTH_M = 0.20

CENTRE_TOL_M = 0.005
REFINE_TOL_M = 0.003
ROTATION_TOL_DEG = 0.1
SCALE_TOL_RATIO = 0.005
BONE_TOL_M = 0.003
NOISY_CENTRE_TOL_M = 0.05
ORIGIN_TOL_M = 0.015
AXIS_TOL_DEG = 1.0
NOISELESS_RMS_TOL_PX = 0.05
NOISY_ROTATION_TOL_DEG = 1.0
KEPT_CENTRE_TOL_M = 0.001
KEPT_ROTATION_TOL_DEG = 0.05
KEPT_JOINT_TOL_M = 0.003
NOISY_KEPT_CENTRE_TOL_M = 0.01
NOISY_KEPT_ROTATION_TOL_DEG = 0.3
REANCHOR_MIN_MOVE_M = 0.5  # the other world is 1.7 m and ~29 deg away, so every camera pose changes by more
TRIANGULATION_FRAME_STEP = 5
MIN_TRIANGULATED_KEYPOINTS = 17
DRIFT_ROTATION_DEG = 2.0
DRIFT_SHIFT_M = 0.05
DRIFTED_MIN_RMS_PX = 5.0


def _intrinsic_matrix() -> np.ndarray:
    focal_px = FOCAL_LENGTH_FACTOR * RESOLUTION[0]
    return np.array([[focal_px, 0.0, RESOLUTION[0] / 2.0], [0.0, focal_px, RESOLUTION[1] / 2.0], [0.0, 0.0, 1.0]])


def _look_at_camera(yaw_deg: float) -> tuple[np.ndarray, np.ndarray]:
    # Camera on a circle around the lifter; yaw 0 = front camera at -Z looking toward +Z.
    yaw_rad = np.radians(yaw_deg)
    centre = np.array([CAMERA_DISTANCE_M * np.sin(yaw_rad), CAMERA_HEIGHT_Y_M, -CAMERA_DISTANCE_M * np.cos(yaw_rad)])
    z_axis = np.array([0.0, 0.3, 0.0]) - centre
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(np.array([0.0, 1.0, 0.0]), z_axis)
    x_axis /= np.linalg.norm(x_axis)
    rotation = np.stack([x_axis, np.cross(z_axis, x_axis), z_axis])
    return rotation, -rotation @ centre


def _true_calibration() -> CalibrationResult:
    result = CalibrationResult(athlete_height_m=HEIGHT_M, timestamp="truth")
    for cam_id, yaw_deg in CAMERA_YAWS_DEG.items():
        rotation, translation = _look_at_camera(yaw_deg)
        result.cameras[cam_id] = CameraCalibration(
            camera_id=cam_id,
            projection_matrix=_intrinsic_matrix() @ np.hstack([rotation, translation[:, None]]),
            intrinsic_matrix=_intrinsic_matrix(),
            rotation_matrix=rotation,
            translation_vector=translation.reshape(3, 1),
            reprojection_error=0.0,
            resolution=RESOLUTION,
        )
    return result


def _lifter_pose(ratios: dict[str, float], depth: float, yaw_deg: float, offset_x_m: float, offset_z_m: float) -> np.ndarray:
    # (23, 3): rigid-segment squat in the sagittal plane, origin at the standing hip midpoint,
    # X = subject's left, Y down, +Z = back; then yawed about the vertical and shifted.
    femur, tibia, torso = (ratios[name] * HEIGHT_M for name in ("femur", "tibia", "torso"))
    hip_half, shoulder_half = ratios["hip_width_half"] * HEIGHT_M, ratios["shoulder_width_half"] * HEIGHT_M
    upper_arm, forearm = ratios["upper_arm"] * HEIGHT_M, ratios["forearm"] * HEIGHT_M
    shank_tilt = np.radians(35.0 * depth)
    thigh_tilt = np.radians(MAX_KNEE_FLEXION_DEG * depth) - shank_tilt
    trunk_lean = np.radians(40.0 * depth)
    arm_flexion = np.radians(10.0 + 60.0 * depth)
    forearm_flexion = arm_flexion + np.radians(20.0)

    ankle = np.array([0.0, femur + tibia, 0.0])
    knee = ankle + tibia * np.array([0.0, -np.cos(shank_tilt), -np.sin(shank_tilt)])
    hip = knee + femur * np.array([0.0, -np.cos(thigh_tilt), np.sin(thigh_tilt)])
    shoulder = hip + torso * np.array([0.0, -np.cos(trunk_lean), -np.sin(trunk_lean)])
    elbow = shoulder + upper_arm * np.array([0.0, np.cos(arm_flexion), -np.sin(arm_flexion)])
    wrist = elbow + forearm * np.array([0.0, np.cos(forearm_flexion), -np.sin(forearm_flexion)])
    nose = shoulder + np.array([0.0, -0.20, -0.09])

    points = np.zeros((NUM_KEYPOINTS + 2, 3))
    points[CK.NOSE] = nose
    for side, eye, ear, shoulder_idx, elbow_idx, wrist_idx, hip_idx, knee_idx, ankle_idx, toe_idx, heel_idx in (
        (1.0, CK.LEFT_EYE, CK.LEFT_EAR, CK.LEFT_SHOULDER, CK.LEFT_ELBOW, CK.LEFT_WRIST, CK.LEFT_HIP,
         CK.LEFT_KNEE, CK.LEFT_ANKLE, CK.LEFT_FOOT_INDEX, CK.LEFT_HEEL),
        (-1.0, CK.RIGHT_EYE, CK.RIGHT_EAR, CK.RIGHT_SHOULDER, CK.RIGHT_ELBOW, CK.RIGHT_WRIST, CK.RIGHT_HIP,
         CK.RIGHT_KNEE, CK.RIGHT_ANKLE, CK.RIGHT_FOOT_INDEX, CK.RIGHT_HEEL),
    ):
        points[eye] = nose + [side * 0.03, -0.03, 0.02]
        points[ear] = nose + [side * 0.075, -0.01, 0.09]
        points[shoulder_idx] = shoulder + [side * shoulder_half, 0.0, 0.0]
        points[elbow_idx] = elbow + [side * shoulder_half, 0.0, 0.0]
        points[wrist_idx] = wrist + [side * shoulder_half, 0.0, 0.0]
        points[hip_idx] = hip + [side * hip_half, 0.0, 0.0]
        points[knee_idx] = knee + [side * hip_half, 0.0, 0.0]
        points[ankle_idx] = ankle + [side * hip_half, 0.0, 0.0]
        points[toe_idx] = points[ankle_idx] + [0.0, 0.0, -FOOT_LENGTH_M]
        points[heel_idx] = points[ankle_idx] + [0.0, 0.04, 0.06]
    points[BAR_POINTS[0]] = shoulder + [BAR_LENGTH_M / 2.0, 0.0, 0.05]
    points[BAR_POINTS[1]] = shoulder + [-BAR_LENGTH_M / 2.0, 0.0, 0.05]

    yaw_rad = np.radians(yaw_deg)
    yaw_rotation = np.array([[np.cos(yaw_rad), 0.0, np.sin(yaw_rad)], [0.0, 1.0, 0.0], [-np.sin(yaw_rad), 0.0, np.cos(yaw_rad)]])
    return points @ yaw_rotation.T + [offset_x_m, 0.0, offset_z_m]


def _lifter_sequence(ratios: dict[str, float], walk_in_frames: int = WALK_IN_FRAMES, squat_frames: int = SQUAT_FRAMES) -> np.ndarray:
    # Walk-in (standing, gliding 0.55 m with a 20 deg turn) then two squat reps on the spot: (F, 23, 3)
    poses = []
    for progress in np.linspace(1.0, 0.0, walk_in_frames, endpoint=False):
        poses.append(_lifter_pose(ratios, 0.0, 20.0 * progress, 0.35 * progress, -0.45 * progress))
    for phase in np.linspace(0.0, 2.0 * np.pi, squat_frames):
        poses.append(_lifter_pose(ratios, float(np.sin(phase) ** 2), 0.0, 0.0, 0.0))
    return np.array(poses)


def _project(points: np.ndarray, camera: CameraCalibration, distortion: np.ndarray | None = None) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(camera.rotation_matrix)
    coeffs = np.zeros(5) if distortion is None else distortion
    projected, _ = cv2.projectPoints(points.reshape(-1, 1, 3), rvec, camera.translation_vector, camera.intrinsic_matrix, coeffs)
    return projected.reshape(points.shape[:-1] + (2,))


def _observe(
    sequence: np.ndarray,
    calibration: CalibrationResult,
    noise_px: float = 0.0,
    seed: int = 0,
    distortion: np.ndarray | None = None,
) -> tuple[list[dict[str, Skeleton2D]], list[dict[str, BarbellDetection]]]:
    rng = np.random.default_rng(seed)
    views: list[dict[str, Skeleton2D]] = [{} for _ in sequence]
    bar_ends: list[dict[str, BarbellDetection]] = [{} for _ in sequence]
    for cam_id, camera in calibration.cameras.items():
        pixels = _project(sequence, camera, distortion)
        pixels = pixels + rng.normal(0.0, noise_px, pixels.shape) if noise_px > 0.0 else pixels
        for frame_idx, frame_pixels in enumerate(pixels):
            keypoints = np.column_stack([frame_pixels[:NUM_KEYPOINTS], np.full(NUM_KEYPOINTS, DETECTED_SCORE)])
            views[frame_idx][cam_id] = Skeleton2D.from_numpy(keypoints, frame_index=frame_idx)
            ends = sorted((frame_pixels[idx] for idx in BAR_POINTS), key=lambda end: end[0])  # detector order: image x
            bar_ends[frame_idx][cam_id] = BarbellDetection(
                left_end=Keypoint2D(x=ends[0][0], y=ends[0][1], confidence=DETECTED_SCORE),
                right_end=Keypoint2D(x=ends[1][0], y=ends[1][1], confidence=DETECTED_SCORE),
                bbox_conf=DETECTED_SCORE,
            )
    return views, bar_ends


def _intrinsics(distortion: np.ndarray | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    coeffs = np.zeros(5) if distortion is None else distortion
    return {cam_id: (_intrinsic_matrix(), coeffs) for cam_id in CAMERA_YAWS_DEG}


def _relative_poses(calibration: CalibrationResult) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    # Gauge-free rig geometry: each camera's rotation and centre expressed in reference camera "0"
    reference = calibration.cameras["0"]
    poses = {}
    for cam_id, camera in calibration.cameras.items():
        if cam_id == "0":
            continue
        rotation = camera.rotation_matrix @ reference.rotation_matrix.T
        translation = camera.translation_vector.ravel() - rotation @ reference.translation_vector.ravel()
        poses[cam_id] = (rotation, -rotation.T @ translation)
    return poses


def _rig_errors(solved: CalibrationResult, truth: CalibrationResult) -> tuple[float, float]:
    # -> largest camera-centre error (m) and rotation error (deg) relative to the reference camera
    solved_poses, true_poses = _relative_poses(solved), _relative_poses(truth)
    centre_errors, rotation_errors = [], []
    for cam_id, (true_rotation, true_centre) in true_poses.items():
        rotation, centre = solved_poses[cam_id]
        centre_errors.append(np.linalg.norm(centre - true_centre))
        cosine = (np.trace(rotation @ true_rotation.T) - 1.0) / 2.0
        rotation_errors.append(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    return float(max(centre_errors)), float(max(rotation_errors))


def _baseline_ratio(solved: CalibrationResult, truth: CalibrationResult) -> float:
    solved_poses, true_poses = _relative_poses(solved), _relative_poses(truth)
    return float(np.mean([np.linalg.norm(solved_poses[c][1]) / np.linalg.norm(true_poses[c][1]) for c in true_poses]))


def _perturbed(
    calibration: CalibrationResult, rotation_deg: float, shift_m: float, cam_ids: tuple[str, ...] | None = None
) -> CalibrationResult:
    # Each listed camera (default: all) turned in place about a different one of its own axes, then its centre moved
    result = CalibrationResult(
        athlete_height_m=calibration.athlete_height_m, timestamp="perturbed", world_anchor=calibration.world_anchor
    )
    for index, (cam_id, camera) in enumerate(calibration.cameras.items()):
        if cam_ids is not None and cam_id not in cam_ids:
            result.cameras[cam_id] = camera
            continue
        axis = np.roll(np.array([1.0, 0.0, 0.0]), index)
        delta, _ = cv2.Rodrigues(axis * np.radians(rotation_deg))
        rotation = delta @ camera.rotation_matrix
        translation = delta @ camera.translation_vector.ravel() + np.roll(np.array([shift_m, 0.0, 0.0]), index + 1)
        result.cameras[cam_id] = CameraCalibration(
            camera_id=cam_id,
            projection_matrix=camera.intrinsic_matrix @ np.hstack([rotation, translation[:, None]]),
            intrinsic_matrix=camera.intrinsic_matrix,
            rotation_matrix=rotation,
            translation_vector=translation.reshape(3, 1),
            reprojection_error=0.0,
            resolution=camera.resolution,
        )
    return result


def _pose_error(camera: CameraCalibration, reference: CameraCalibration) -> tuple[float, float]:
    # World-frame pose difference: camera-centre distance (m) and rotation angle (deg)
    centre = -camera.rotation_matrix.T @ camera.translation_vector.ravel()
    reference_centre = -reference.rotation_matrix.T @ reference.translation_vector.ravel()
    cosine = (np.trace(camera.rotation_matrix @ reference.rotation_matrix.T) - 1.0) / 2.0
    return float(np.linalg.norm(centre - reference_centre)), float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _triangulated_joint_error_m(
    calibration: CalibrationResult, views: list[dict[str, Skeleton2D]], sequence: np.ndarray
) -> float:
    # Largest distance between joints triangulated with the calibration and the true joints, in the truth world
    triangulator = DLTTriangulator(calibration)
    worst_m = 0.0
    for frame_idx in range(0, len(views), TRIANGULATION_FRAME_STEP):
        skeleton = triangulator.triangulate(MultiViewPose(views=views[frame_idx], timestamp=float(frame_idx)))
        accepted = np.array([keypoint.confidence > 0.0 for keypoint in skeleton.keypoints])
        assert accepted.sum() >= MIN_TRIANGULATED_KEYPOINTS
        errors_m = np.linalg.norm(skeleton.to_numpy()[:NUM_KEYPOINTS] - sequence[frame_idx, :NUM_KEYPOINTS], axis=1)
        worst_m = max(worst_m, float(errors_m[accepted].max()))
    return worst_m


def _in_other_world(calibration: CalibrationResult, world_anchor: str) -> CalibrationResult:
    # The same rig expressed in a world frame that is rotated and shifted (e.g. a factory board frame)
    world_rotation, _ = cv2.Rodrigues(np.array([0.2, 0.45, -0.1]))
    world_shift = np.array([0.8, -0.3, 1.5])
    result = CalibrationResult(athlete_height_m=calibration.athlete_height_m, timestamp="other", world_anchor=world_anchor)
    for cam_id, camera in calibration.cameras.items():
        rotation = camera.rotation_matrix @ world_rotation.T
        translation = camera.translation_vector.ravel() - rotation @ world_shift
        result.cameras[cam_id] = CameraCalibration(
            camera_id=cam_id,
            projection_matrix=camera.intrinsic_matrix @ np.hstack([rotation, translation[:, None]]),
            intrinsic_matrix=camera.intrinsic_matrix,
            rotation_matrix=rotation,
            translation_vector=translation.reshape(3, 1),
            reprojection_error=0.0,
            resolution=camera.resolution,
        )
    return result


class _ReplayEstimator(PoseEstimator):
    """Returns the precomputed T-pose skeleton of whichever camera is asked for."""

    def __init__(self, skeletons: dict[str, Skeleton2D]) -> None:
        super().__init__(confidence_threshold=0.3)
        self._skeletons = skeletons

    def estimate(self, frame: np.ndarray, camera_id: int | str = 0) -> Skeleton2D | None:
        return self._skeletons[str(camera_id)]

    def estimate_3d(self, frame: np.ndarray) -> Skeleton3D | None:
        return None


def _lifter_tpose() -> np.ndarray:
    # The lifter (not the calibration model's average body) holding a T-pose at the origin: (21, 3)
    tpose = _lifter_pose(LIFTER_RATIOS, 0.0, 0.0, 0.0, 0.0)[:NUM_KEYPOINTS]
    upper_arm, forearm = LIFTER_RATIOS["upper_arm"] * HEIGHT_M, LIFTER_RATIOS["forearm"] * HEIGHT_M
    for side, shoulder_idx, elbow_idx, wrist_idx in (
        (1.0, CK.LEFT_SHOULDER, CK.LEFT_ELBOW, CK.LEFT_WRIST), (-1.0, CK.RIGHT_SHOULDER, CK.RIGHT_ELBOW, CK.RIGHT_WRIST),
    ):
        tpose[elbow_idx] = tpose[shoulder_idx] + [side * upper_arm, 0.0, 0.0]
        tpose[wrist_idx] = tpose[elbow_idx] + [side * forearm, 0.0, 0.0]
    return tpose


def _tpose_calibration(truth: CalibrationResult) -> CalibrationResult:
    # The production T-pose calibration of the same rig: PnP of the average-body model onto the lifter's T-pose
    tpose = _lifter_tpose()
    skeletons = {
        cam_id: Skeleton2D.from_numpy(np.column_stack([_project(tpose, camera), np.full(NUM_KEYPOINTS, DETECTED_SCORE)]))
        for cam_id, camera in truth.cameras.items()
    }
    calibrator = TPoseCalibrator(pose_estimator=_ReplayEstimator(skeletons), focal_length_factor=FOCAL_LENGTH_FACTOR)
    frames = {cam_id: [np.zeros((2, 2, 3), dtype=np.uint8)] * TPOSE_FRAMES for cam_id in truth.cameras}
    return calibrator.calibrate(frames, HEIGHT_M, RESOLUTION)


@pytest.fixture(scope="module")
def true_calibration() -> CalibrationResult:
    return _true_calibration()


@pytest.fixture(scope="module")
def lifter_sequence() -> np.ndarray:
    return _lifter_sequence(LIFTER_RATIOS)


@pytest.fixture(scope="module")
def noiseless_observations(
    lifter_sequence: np.ndarray, true_calibration: CalibrationResult
) -> tuple[list[dict[str, Skeleton2D]], list[dict[str, BarbellDetection]]]:
    return _observe(lifter_sequence, true_calibration)


@pytest.fixture(scope="module")
def bar_result(noiseless_observations: tuple) -> PersonCalibrationResult:
    views, bar_ends = noiseless_observations
    return PersonCalibrator(_intrinsics(), RESOLUTION, bar_length_m=BAR_LENGTH_M).calibrate(views, bar_ends=bar_ends)


class TestNoiselessRecoveryWithBar:
    def test_camera_centres_within_five_millimetres(self, bar_result: PersonCalibrationResult, true_calibration: CalibrationResult):
        centre_error_m, rotation_error_deg = _rig_errors(bar_result.calibration, true_calibration)
        assert centre_error_m < CENTRE_TOL_M
        assert rotation_error_deg < ROTATION_TOL_DEG

    def test_scale_within_half_percent(self, bar_result: PersonCalibrationResult, true_calibration: CalibrationResult):
        assert bar_result.scale_source == "bar"
        assert _baseline_ratio(bar_result.calibration, true_calibration) == pytest.approx(1.0, abs=SCALE_TOL_RATIO)

    def test_bone_lengths_match_the_lifter(self, bar_result: PersonCalibrationResult):
        expected = {
            "hip_width": 2.0 * LIFTER_RATIOS["hip_width_half"] * HEIGHT_M,
            "femur_l": LIFTER_RATIOS["femur"] * HEIGHT_M,
            "tibia_r": LIFTER_RATIOS["tibia"] * HEIGHT_M,
            "upper_arm_l": LIFTER_RATIOS["upper_arm"] * HEIGHT_M,
            "forearm_r": LIFTER_RATIOS["forearm"] * HEIGHT_M,
            "foot_l": FOOT_LENGTH_M,
        }
        for name, length_m in expected.items():
            assert bar_result.bone_lengths_m[name] == pytest.approx(length_m, abs=BONE_TOL_M)

    def test_reprojection_drops_to_zero(self, bar_result: PersonCalibrationResult):
        assert bar_result.rms_reprojection_px < NOISELESS_RMS_TOL_PX
        assert bar_result.rms_reprojection_px <= bar_result.initial_rms_reprojection_px + NOISELESS_RMS_TOL_PX
        assert bar_result.frames_used == WALK_IN_FRAMES + SQUAT_FRAMES

    def test_projection_matrices_are_k_times_extrinsics(self, bar_result: PersonCalibrationResult):
        for camera in bar_result.calibration.cameras.values():
            extrinsics = np.hstack([camera.rotation_matrix, camera.translation_vector.reshape(3, 1)])
            assert camera.projection_matrix == pytest.approx(camera.intrinsic_matrix @ extrinsics, abs=1e-9)
            assert camera.rotation_matrix @ camera.rotation_matrix.T == pytest.approx(np.eye(3), abs=1e-9)
            assert camera.resolution == RESOLUTION


class TestWorldFrame:
    def test_frame_from_rotated_standing_skeleton(self):
        standing = _lifter_pose(LIFTER_RATIOS, 0.0, 0.0, 0.0, 0.0)[:NUM_KEYPOINTS]
        tilt, _ = cv2.Rodrigues(np.array([0.3, -0.5, 0.2]))
        shift = np.array([0.4, -1.2, 2.5])
        observed = np.stack([standing @ tilt.T + shift] * 3)
        rotation, origin = world_frame_from_standing(observed)
        assert origin == pytest.approx(shift, abs=1e-9)
        assert (observed[0] - origin) @ rotation.T == pytest.approx(standing, abs=1e-9)

    def test_one_bad_frame_does_not_tilt_the_frame(self):
        standing = np.stack([_lifter_pose(LIFTER_RATIOS, 0.0, 0.0, 0.0, 0.0)[:NUM_KEYPOINTS]] * 9)
        standing[4, CK.LEFT_ANKLE] += [0.5, 0.0, 0.5]
        rotation, _ = world_frame_from_standing(standing)
        assert rotation == pytest.approx(np.eye(3), abs=1e-9)

    def test_calibrated_world_is_y_down_x_left_origin_at_standing_hips(self, true_calibration: CalibrationResult):
        # No walk-in: the truth frame already sits at the standing hip midpoint facing -Z
        sequence = _lifter_sequence(LIFTER_RATIOS, walk_in_frames=0, squat_frames=80)
        views, bar_ends = _observe(sequence, true_calibration)
        result = PersonCalibrator(_intrinsics(), RESOLUTION, bar_length_m=BAR_LENGTH_M).calibrate(views, bar_ends=bar_ends)
        for cam_id, camera in result.calibration.cameras.items():
            truth = true_calibration.cameras[cam_id]
            centre = -camera.rotation_matrix.T @ camera.translation_vector.ravel()
            true_centre = -truth.rotation_matrix.T @ truth.translation_vector.ravel()
            assert np.linalg.norm(centre - true_centre) < ORIGIN_TOL_M + CAMERA_DISTANCE_M * np.radians(AXIS_TOL_DEG)
            cosine = (np.trace(camera.rotation_matrix @ truth.rotation_matrix.T) - 1.0) / 2.0
            assert np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))) < AXIS_TOL_DEG
        assert result.standing_frames >= 5


class TestInitialisation:
    def test_tpose_and_essential_starts_converge_to_the_same_rig(self, lifter_sequence: np.ndarray, true_calibration: CalibrationResult):
        views, bar_ends = _observe(lifter_sequence, true_calibration, noise_px=1.0, seed=3)
        calibrator = PersonCalibrator(_intrinsics(), RESOLUTION, bar_length_m=BAR_LENGTH_M)
        from_essential = calibrator.calibrate(views, bar_ends=bar_ends)
        from_tpose = calibrator.calibrate(views, bar_ends=bar_ends, initial=_tpose_calibration(true_calibration))
        centre_error_m, rotation_error_deg = _rig_errors(from_tpose.calibration, from_essential.calibration)
        assert centre_error_m < REFINE_TOL_M
        assert rotation_error_deg < ROTATION_TOL_DEG
        assert from_tpose.rms_reprojection_px == pytest.approx(from_essential.rms_reprojection_px, abs=0.05)

    def test_refine_recovers_a_perturbed_calibration(self, noiseless_observations: tuple, true_calibration: CalibrationResult):
        views, bar_ends = noiseless_observations
        drifted = _perturbed(true_calibration, rotation_deg=DRIFT_ROTATION_DEG, shift_m=DRIFT_SHIFT_M)
        assert _rig_errors(drifted, true_calibration)[0] > DRIFT_SHIFT_M
        assert _rig_errors(drifted, true_calibration)[1] > DRIFT_ROTATION_DEG
        calibrator = PersonCalibrator(_intrinsics(), RESOLUTION, bar_length_m=BAR_LENGTH_M)
        refined = calibrator.refine(drifted, views, bar_ends=bar_ends)
        centre_error_m, rotation_error_deg = _rig_errors(refined.calibration, true_calibration)
        assert centre_error_m < REFINE_TOL_M
        assert rotation_error_deg < ROTATION_TOL_DEG
        assert refined.initial_rms_reprojection_px > DRIFTED_MIN_RMS_PX
        assert refined.rms_reprojection_px < NOISELESS_RMS_TOL_PX

    def test_initial_without_a_camera_raises(self, noiseless_observations: tuple, true_calibration: CalibrationResult):
        views, _ = noiseless_observations
        partial = CalibrationResult(cameras={"0": true_calibration.cameras["0"]})
        with pytest.raises(ValueError, match="initial calibration"):
            PersonCalibrator(_intrinsics(), RESOLUTION).calibrate(views, initial=partial)


class TestKeepWorldFrame:
    @pytest.mark.parametrize("moved_cam_id", ["1", "0"], ids=["side_camera_moved", "reference_camera_moved"])
    def test_one_moved_camera_is_recovered_in_the_original_world(
        self,
        moved_cam_id: str,
        noiseless_observations: tuple,
        lifter_sequence: np.ndarray,
        true_calibration: CalibrationResult,
    ):
        views, bar_ends = noiseless_observations
        drifted = _perturbed(true_calibration, DRIFT_ROTATION_DEG, DRIFT_SHIFT_M, cam_ids=(moved_cam_id,))
        assert _pose_error(drifted.cameras[moved_cam_id], true_calibration.cameras[moved_cam_id])[0] > DRIFT_SHIFT_M / 2.0
        calibrator = PersonCalibrator(_intrinsics(), RESOLUTION, bar_length_m=BAR_LENGTH_M)
        refined = calibrator.refine(drifted, views, bar_ends=bar_ends, keep_world_frame=True)
        for cam_id, camera in refined.calibration.cameras.items():
            if cam_id == moved_cam_id:
                continue
            centre_error_m, rotation_error_deg = _pose_error(camera, drifted.cameras[cam_id])
            assert centre_error_m < KEPT_CENTRE_TOL_M
            assert rotation_error_deg < KEPT_ROTATION_TOL_DEG
        moved_centre_error_m, moved_rotation_error_deg = _pose_error(
            refined.calibration.cameras[moved_cam_id], true_calibration.cameras[moved_cam_id]
        )
        assert moved_centre_error_m < REFINE_TOL_M
        assert moved_rotation_error_deg < ROTATION_TOL_DEG
        assert _triangulated_joint_error_m(refined.calibration, views, lifter_sequence) < KEPT_JOINT_TOL_M
        assert refined.calibration.world_anchor == drifted.world_anchor
        assert refined.standing_frames == 0

    def test_without_the_flag_refine_still_reanchors_on_the_lifter(self, noiseless_observations: tuple, true_calibration: CalibrationResult):
        views, bar_ends = noiseless_observations
        board_world = _in_other_world(true_calibration, world_anchor="board")
        calibrator = PersonCalibrator(_intrinsics(), RESOLUTION, bar_length_m=BAR_LENGTH_M)
        reanchored = calibrator.refine(board_world, views, bar_ends=bar_ends)
        from_scratch = calibrator.calibrate(views, bar_ends=bar_ends)
        assert reanchored.calibration.world_anchor == "person"
        assert reanchored.standing_frames >= 5
        for cam_id, camera in reanchored.calibration.cameras.items():
            centre_error_m, rotation_error_deg = _pose_error(camera, from_scratch.calibration.cameras[cam_id])
            assert centre_error_m < KEPT_CENTRE_TOL_M
            assert rotation_error_deg < KEPT_ROTATION_TOL_DEG
            assert _pose_error(camera, board_world.cameras[cam_id])[0] > REANCHOR_MIN_MOVE_M

    def test_board_world_and_anchor_are_kept(self, noiseless_observations: tuple, true_calibration: CalibrationResult):
        views, bar_ends = noiseless_observations
        board_world = _in_other_world(true_calibration, world_anchor="board")
        calibrator = PersonCalibrator(_intrinsics(), RESOLUTION, bar_length_m=BAR_LENGTH_M)
        kept = calibrator.refine(board_world, views, bar_ends=bar_ends, keep_world_frame=True)
        assert kept.calibration.world_anchor == "board"
        for cam_id, camera in kept.calibration.cameras.items():
            centre_error_m, rotation_error_deg = _pose_error(camera, board_world.cameras[cam_id])
            assert centre_error_m < KEPT_CENTRE_TOL_M
            assert rotation_error_deg < KEPT_ROTATION_TOL_DEG

    def test_noisy_refine_keeps_the_untouched_cameras_close(self, lifter_sequence: np.ndarray, true_calibration: CalibrationResult):
        views, bar_ends = _observe(lifter_sequence, true_calibration, noise_px=2.0, seed=2)
        drifted = _perturbed(true_calibration, DRIFT_ROTATION_DEG, DRIFT_SHIFT_M, cam_ids=("0",))
        calibrator = PersonCalibrator(_intrinsics(), RESOLUTION, bar_length_m=BAR_LENGTH_M)
        refined = calibrator.refine(drifted, views, bar_ends=bar_ends, keep_world_frame=True)
        for cam_id in ("1", "2"):
            centre_error_m, rotation_error_deg = _pose_error(refined.calibration.cameras[cam_id], drifted.cameras[cam_id])
            assert centre_error_m < NOISY_KEPT_CENTRE_TOL_M
            assert rotation_error_deg < NOISY_KEPT_ROTATION_TOL_DEG
        assert _pose_error(refined.calibration.cameras["0"], true_calibration.cameras["0"])[0] < NOISY_KEPT_CENTRE_TOL_M

    def test_two_camera_rig_keeps_the_reference_camera_pose(self, noiseless_observations: tuple, true_calibration: CalibrationResult):
        views, bar_ends = noiseless_observations
        two_camera_views = [{cam_id: frame_views[cam_id] for cam_id in ("0", "2")} for frame_views in views]
        drifted = _perturbed(true_calibration, DRIFT_ROTATION_DEG, DRIFT_SHIFT_M, cam_ids=("2",))
        initial = CalibrationResult(cameras={cam_id: drifted.cameras[cam_id] for cam_id in ("0", "2")})
        intrinsics = {cam_id: _intrinsics()[cam_id] for cam_id in ("0", "2")}
        calibrator = PersonCalibrator(intrinsics, RESOLUTION, bar_length_m=BAR_LENGTH_M)
        refined = calibrator.refine(initial, two_camera_views, bar_ends=bar_ends, keep_world_frame=True)
        assert _pose_error(refined.calibration.cameras["0"], initial.cameras["0"])[0] < KEPT_CENTRE_TOL_M
        assert _pose_error(refined.calibration.cameras["2"], true_calibration.cameras["2"])[0] < REFINE_TOL_M

    def test_flag_is_ignored_without_an_initial_calibration(self, noiseless_observations: tuple, bar_result: PersonCalibrationResult):
        views, bar_ends = noiseless_observations
        calibrator = PersonCalibrator(_intrinsics(), RESOLUTION, bar_length_m=BAR_LENGTH_M)
        result = calibrator.calibrate(views, bar_ends=bar_ends, keep_world_frame=True)
        assert result.calibration.world_anchor == "person"
        assert result.standing_frames == bar_result.standing_frames
        for cam_id, camera in result.calibration.cameras.items():
            assert _pose_error(camera, bar_result.calibration.cameras[cam_id])[0] < KEPT_CENTRE_TOL_M

    def test_consensus_fit_leaves_out_the_one_camera_that_disagrees(self, true_calibration: CalibrationResult):
        cameras = list(true_calibration.cameras.values())
        rotations = np.stack([camera.rotation_matrix for camera in cameras])
        translations = np.stack([camera.translation_vector.ravel() for camera in cameras])
        lever_arms_m = np.full(len(cameras), CAMERA_DISTANCE_M)
        initial_landmarks = _camera_landmarks(rotations, translations, lever_arms_m)
        # The solved rig lives in another frame (world = rotation @ solved + shift) and its camera 2 has moved
        frame_rotation, _ = cv2.Rodrigues(np.array([0.1, -0.7, 0.3]))
        frame_shift = np.array([0.5, 0.2, -1.0])
        solved_landmarks = (initial_landmarks - frame_shift) @ frame_rotation
        solved_landmarks[2] += [0.04, -0.02, 0.03]
        rotation, shift, cameras_used = _consensus_rigid_fit(solved_landmarks, initial_landmarks)
        assert cameras_used.tolist() == [True, True, False]
        assert rotation == pytest.approx(frame_rotation, abs=1e-9)
        assert shift == pytest.approx(frame_shift, abs=1e-9)
        solved_landmarks[2] -= [0.04, -0.02, 0.03]
        assert _consensus_rigid_fit(solved_landmarks, initial_landmarks)[2].all()


class TestScaleSources:
    def test_height_prior_is_exact_for_the_model_body(self, true_calibration: CalibrationResult):
        views, _ = _observe(_lifter_sequence(SEGMENT_RATIOS), true_calibration)
        result = PersonCalibrator(_intrinsics(), RESOLUTION, height_m=HEIGHT_M).calibrate(views)
        assert result.scale_source == "height"
        assert _baseline_ratio(result.calibration, true_calibration) == pytest.approx(1.0, abs=SCALE_TOL_RATIO)
        assert result.calibration.athlete_height_m == pytest.approx(HEIGHT_M)

    def test_height_prior_inherits_the_lifters_proportion_error(self, noiseless_observations: tuple, true_calibration: CalibrationResult):
        views, _ = noiseless_observations
        result = PersonCalibrator(_intrinsics(), RESOLUTION, height_m=HEIGHT_M).calibrate(views)
        model = 2.0 * (SEGMENT_RATIOS["femur"] + SEGMENT_RATIOS["tibia"] + SEGMENT_RATIOS["upper_arm"]
                       + SEGMENT_RATIOS["forearm"] + SEGMENT_RATIOS["hip_width_half"])
        lifter = 2.0 * (LIFTER_RATIOS["femur"] + LIFTER_RATIOS["tibia"] + LIFTER_RATIOS["upper_arm"]
                        + LIFTER_RATIOS["forearm"] + LIFTER_RATIOS["hip_width_half"])
        assert _baseline_ratio(result.calibration, true_calibration) == pytest.approx(model / lifter, abs=SCALE_TOL_RATIO)

    def test_bar_in_too_few_frames_falls_back_to_height(self, noiseless_observations: tuple):
        views, bar_ends = noiseless_observations
        sparse_bar = [frame_bars if frame_idx < 5 else {} for frame_idx, frame_bars in enumerate(bar_ends)]
        calibrator = PersonCalibrator(_intrinsics(), RESOLUTION, bar_length_m=BAR_LENGTH_M, height_m=HEIGHT_M)
        assert calibrator.calibrate(views, bar_ends=sparse_bar).scale_source == "height"

    def test_without_bar_or_height_the_initial_scale_is_kept(self, noiseless_observations: tuple, true_calibration: CalibrationResult):
        views, _ = noiseless_observations
        drifted = _perturbed(true_calibration, rotation_deg=1.0, shift_m=0.0)
        result = PersonCalibrator(_intrinsics(), RESOLUTION).refine(drifted, views)
        assert result.scale_source == "initial"
        first_baseline = np.linalg.norm(_relative_poses(result.calibration)["1"][1])
        assert first_baseline == pytest.approx(np.linalg.norm(_relative_poses(drifted)["1"][1]), rel=SCALE_TOL_RATIO)
        assert _rig_errors(result.calibration, true_calibration)[1] < ROTATION_TOL_DEG

    def test_bar_ends_are_relabelled_by_the_shoulder_line(self):
        # One frame, two views: a front camera (subject's left on image right) and a rear camera
        positions = np.zeros((1, 2, NUM_KEYPOINTS, 2))
        scores = np.full((1, 2, NUM_KEYPOINTS), DETECTED_SCORE)
        positions[0, 0, CK.LEFT_SHOULDER], positions[0, 0, CK.RIGHT_SHOULDER] = (700.0, 300.0), (580.0, 300.0)
        positions[0, 1, CK.LEFT_SHOULDER], positions[0, 1, CK.RIGHT_SHOULDER] = (580.0, 300.0), (700.0, 300.0)
        bar_positions = np.array([[[[300.0, 310.0], [980.0, 305.0]], [[310.0, 300.0], [990.0, 302.0]]]])
        bar_scores = np.full((1, 2, 2), DETECTED_SCORE)
        labelled, labelled_scores = _label_bar_ends_by_shoulders(bar_positions, bar_scores, positions, scores)
        assert labelled[0, 0, 0] == pytest.approx([980.0, 305.0])  # front view: subject-left end is image right
        assert labelled[0, 1, 0] == pytest.approx([310.0, 300.0])  # rear view: subject-left end is image left
        assert (labelled_scores > 0.0).all()


class TestRobustness:
    def test_noisy_detections_stay_close(self, lifter_sequence: np.ndarray, true_calibration: CalibrationResult):
        views, bar_ends = _observe(lifter_sequence, true_calibration, noise_px=2.0, seed=1)
        result = PersonCalibrator(_intrinsics(), RESOLUTION, bar_length_m=BAR_LENGTH_M).calibrate(views, bar_ends=bar_ends)
        centre_error_m, rotation_error_deg = _rig_errors(result.calibration, true_calibration)
        assert centre_error_m < NOISY_CENTRE_TOL_M
        assert rotation_error_deg < NOISY_ROTATION_TOL_DEG
        # residual of a 3-view point: sigma * sqrt(2) * sqrt(3 / 6) = sigma
        assert result.rms_reprojection_px == pytest.approx(2.0, rel=0.25)

    def test_gross_outliers_and_low_scores_are_ignored(self, lifter_sequence: np.ndarray, true_calibration: CalibrationResult):
        views, bar_ends = _observe(lifter_sequence, true_calibration)
        rng = np.random.default_rng(7)
        for frame_views in views:
            for cam_id, skeleton in frame_views.items():
                keypoints = skeleton.to_numpy()
                outliers = rng.random(NUM_KEYPOINTS) < 0.03
                keypoints[outliers, :2] += rng.uniform(40.0, 120.0, (int(outliers.sum()), 2))
                garbage = rng.random(NUM_KEYPOINTS) < 0.05
                keypoints[garbage] = np.column_stack([rng.uniform(0.0, 1280.0, (int(garbage.sum()), 2)),
                                                      np.full(int(garbage.sum()), LOW_SCORE)])
                frame_views[cam_id] = Skeleton2D.from_numpy(keypoints)
        result = PersonCalibrator(_intrinsics(), RESOLUTION, bar_length_m=BAR_LENGTH_M).calibrate(views, bar_ends=bar_ends)
        centre_error_m, rotation_error_deg = _rig_errors(result.calibration, true_calibration)
        assert centre_error_m < 2.0 * CENTRE_TOL_M
        assert rotation_error_deg < 2.0 * ROTATION_TOL_DEG

    def test_distorted_detections_are_undistorted(self, lifter_sequence: np.ndarray, true_calibration: CalibrationResult):
        distortion = np.array([-0.12, 0.05, 0.001, -0.001, 0.0])
        views, bar_ends = _observe(lifter_sequence, true_calibration, distortion=distortion)
        calibrator = PersonCalibrator(_intrinsics(distortion), RESOLUTION, bar_length_m=BAR_LENGTH_M)
        result = calibrator.calibrate(views, bar_ends=bar_ends)
        centre_error_m, _ = _rig_errors(result.calibration, true_calibration)
        assert centre_error_m < CENTRE_TOL_M
        assert result.calibration.cameras["1"].intrinsic_matrix == pytest.approx(_intrinsic_matrix())

    def test_long_captures_are_subsampled(self, true_calibration: CalibrationResult):
        sequence = _lifter_sequence(LIFTER_RATIOS, walk_in_frames=60, squat_frames=140)
        views, bar_ends = _observe(sequence, true_calibration)
        result = PersonCalibrator(_intrinsics(), RESOLUTION, bar_length_m=BAR_LENGTH_M).calibrate(views, bar_ends=bar_ends)
        assert result.frames_used == MAX_FRAMES
        assert _rig_errors(result.calibration, true_calibration)[0] < CENTRE_TOL_M

    def test_two_camera_rig(self, lifter_sequence: np.ndarray, true_calibration: CalibrationResult):
        views, bar_ends = _observe(lifter_sequence, true_calibration)
        two_camera_views = [{cam_id: frame_views[cam_id] for cam_id in ("0", "2")} for frame_views in views]
        intrinsics = {cam_id: _intrinsics()[cam_id] for cam_id in ("0", "2")}
        result = PersonCalibrator(intrinsics, RESOLUTION, bar_length_m=BAR_LENGTH_M).calibrate(two_camera_views, bar_ends=bar_ends)
        solved_centre = _relative_poses(result.calibration)["2"][1]
        assert np.linalg.norm(solved_centre - _relative_poses(true_calibration)["2"][1]) < CENTRE_TOL_M


class TestValidation:
    def test_single_camera_raises(self):
        with pytest.raises(ValueError, match="2 cameras"):
            PersonCalibrator({"0": (_intrinsic_matrix(), np.zeros(5))}, RESOLUTION)

    def test_too_few_frames_raises(self, noiseless_observations: tuple):
        views, _ = noiseless_observations
        with pytest.raises(ValueError, match="frames"):
            PersonCalibrator(_intrinsics(), RESOLUTION).calibrate(views[:4])

    def test_frames_seen_by_one_camera_do_not_count(self, noiseless_observations: tuple):
        views, _ = noiseless_observations
        single_view = [{"0": frame_views["0"]} for frame_views in views]
        with pytest.raises(ValueError, match="two cameras"):
            PersonCalibrator(_intrinsics(), RESOLUTION).calibrate(single_view)


class TestReprojectionHealth:
    def test_true_calibration_reads_zero(self, noiseless_observations: tuple, true_calibration: CalibrationResult):
        views, _ = noiseless_observations
        assert reprojection_health_px(true_calibration, views) < NOISELESS_RMS_TOL_PX

    def test_matches_detection_noise(self, lifter_sequence: np.ndarray, true_calibration: CalibrationResult):
        views, _ = _observe(lifter_sequence, true_calibration, noise_px=3.0, seed=5)
        # three views, 2x3 measurements - 3 unknowns per point: residual RMS = sigma * sqrt(2) * sqrt(3 / 6)
        assert reprojection_health_px(true_calibration, views) == pytest.approx(3.0, rel=0.15)

    def test_rises_when_a_camera_drifts(self, lifter_sequence: np.ndarray, true_calibration: CalibrationResult):
        views, _ = _observe(lifter_sequence, true_calibration, noise_px=3.0, seed=5)
        healthy_px = reprojection_health_px(true_calibration, views)
        drifted_px = reprojection_health_px(_perturbed(true_calibration, rotation_deg=1.0, shift_m=0.0), views)
        assert drifted_px > 2.0 * healthy_px

    def test_gross_outliers_do_not_move_it(self, lifter_sequence: np.ndarray, true_calibration: CalibrationResult):
        views, _ = _observe(lifter_sequence, true_calibration, noise_px=3.0, seed=5)
        healthy_px = reprojection_health_px(true_calibration, views)
        rng = np.random.default_rng(11)
        for frame_views in views:
            keypoints = frame_views["1"].to_numpy()
            outliers = rng.random(NUM_KEYPOINTS) < 0.05
            keypoints[outliers, :2] += 150.0
            frame_views["1"] = Skeleton2D.from_numpy(keypoints)
        assert reprojection_health_px(true_calibration, views) == pytest.approx(healthy_px, rel=0.15)

    def test_nan_when_nothing_is_seen_twice(self, noiseless_observations: tuple, true_calibration: CalibrationResult):
        views, _ = noiseless_observations
        single_view = [{"0": frame_views["0"]} for frame_views in views[:5]]
        assert np.isnan(reprojection_health_px(true_calibration, single_view))
