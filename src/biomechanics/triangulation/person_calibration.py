"""
Person-based extrinsic calibration: bundle adjustment over a lifter's 2D keypoints seen by
cameras with known intrinsics. Solves camera poses, per-frame 3D joints and rigid segment
lengths jointly; metric scale comes from the barbell length, else a height prior, else the
initial calibration. World frame: Y down, X = subject's left, +Z = subject's back, origin at
the standing hip midpoint.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix

from biomechanics.triangulation.calibration import (
    SEGMENT_RATIOS,
    WORLD_ANCHOR_PERSON,
    CalibrationResult,
    CameraCalibration,
    undistort_keypoints,
)
from biomechanics.triangulation.triangulator import _dlt_normal_matrices, _solve_and_measure
from biomechanics.utils.segment_lengths import RIGID_SEGMENTS, SEGMENTS
from biomechanics.utils.types import BarbellDetection, Skeleton2D
from biomechanics.utils.types import CocoKeypoints as CK

logger = logging.getLogger(__name__)

NUM_KEYPOINTS = 21
# Per-frame point layout: 21 body keypoints, then the barbell's subject-left and subject-right ends.
BAR_LEFT_POINT = NUM_KEYPOINTS
BAR_RIGHT_POINT = NUM_KEYPOINTS + 1
NUM_FRAME_POINTS = NUM_KEYPOINTS + 2
FACE_KEYPOINTS = (CK.NOSE, CK.LEFT_EYE, CK.RIGHT_EYE, CK.LEFT_EAR, CK.RIGHT_EAR)
# Face keypoints are not tied to any rigid segment and are the most view-dependent detections.
FACE_KEYPOINT_WEIGHT = 0.5

MIN_KEYPOINT_SCORE = 0.3
MIN_INIT_KEYPOINT_SCORE = 0.5
MIN_VIEWS_PER_POINT = 2
MIN_KEYPOINTS_PER_VIEW = 6
MIN_FRAMES = 10
MAX_FRAMES = 150
MIN_PAIR_CORRESPONDENCES = 30

# Frame subsampling: pose descriptor = body keypoints (shoulders to ankles) in every view,
# median-filtered in time so single-frame detection outliers do not look like diverse poses.
DESCRIPTOR_KEYPOINTS = tuple(range(CK.LEFT_SHOULDER, CK.RIGHT_ANKLE + 1))
DESCRIPTOR_MEDIAN_WINDOW = 5
# Share of the frame budget reserved for frames where the bar is visible in >= 2 views.
BAR_FRAME_BUDGET_RATIO = 1.0 / 3.0

# Rigid segments solved jointly: the pipeline's rigid leg/foot segments plus the arms.
ARM_SEGMENTS: dict[str, tuple[int, int]] = {
    "upper_arm_l": (CK.LEFT_SHOULDER, CK.LEFT_ELBOW),
    "upper_arm_r": (CK.RIGHT_SHOULDER, CK.RIGHT_ELBOW),
    "forearm_l": (CK.LEFT_ELBOW, CK.LEFT_WRIST),
    "forearm_r": (CK.RIGHT_ELBOW, CK.RIGHT_WRIST),
}
BA_SEGMENTS: dict[str, tuple[int, int]] = {
    **{name: (SEGMENTS[name][0], SEGMENTS[name][1]) for name in RIGID_SEGMENTS},
    **ARM_SEGMENTS,
}
BA_SEGMENT_NAMES: tuple[str, ...] = tuple(BA_SEGMENTS)
# Segment length / body height for the height prior (feet have no ratio in the T-pose model).
SEGMENT_HEIGHT_RATIOS: dict[str, float] = {
    "hip_width": 2.0 * SEGMENT_RATIOS["hip_width_half"],
    "femur_l": SEGMENT_RATIOS["femur"],
    "femur_r": SEGMENT_RATIOS["femur"],
    "tibia_l": SEGMENT_RATIOS["tibia"],
    "tibia_r": SEGMENT_RATIOS["tibia"],
    "upper_arm_l": SEGMENT_RATIOS["upper_arm"],
    "upper_arm_r": SEGMENT_RATIOS["upper_arm"],
    "forearm_l": SEGMENT_RATIOS["forearm"],
    "forearm_r": SEGMENT_RATIOS["forearm"],
}
MIN_SEGMENT_FRAMES = 5
# "initial" scale: the refined rig is rescaled so that these segments, triangulated by the
# cameras that did not move, measure what the initial calibration measured with the same
# cameras. Segments seen in fewer frames are skipped; none left -> the baseline hold stays.
SCALE_TARGET_SEGMENTS: tuple[str, ...] = ("hip_width", "femur_l", "femur_r", "tibia_l", "tibia_r")
MIN_SCALE_TARGET_FRAMES = 10

# Residual weights. Reprojection residuals are in pixels (x score / median score); the metric
# terms are converted to pixel-equivalents: 1 cm of bone deviation ~ 3 px, 1 cm of bar-length,
# height-prior or baseline deviation ~ 10 px (strong: these only fix the free scale).
HUBER_DELTA_PX = 6.0
BONE_WEIGHT_PX_PER_M = 300.0
SCALE_WEIGHT_PX_PER_M = 1000.0
MIN_BAR_FRAMES = 10
MIN_BAR_SHOULDER_SPAN_PX = 10.0

# After the first solve, observations beyond this multiple of the median residual (and at
# least the floor) are dropped and the problem is re-solved.
OUTLIER_MEDIAN_FACTOR = 4.0
OUTLIER_FLOOR_PX = 10.0
MAX_SOLVER_EVALUATIONS = 100
MAX_CAMERA_ONLY_EVALUATIONS = 40
APPROACH_LOSS = "soft_l1"
FINAL_LOSS = "huber"
SOLVER_TOLERANCE = 1e-5
RANSAC_THRESHOLD_PX = 6.0
RANSAC_CONFIDENCE = 0.999
RANSAC_SEED = 0
MIN_DEPTH_M = 1e-3

# A frame is standing when every visible leg looks straight in every view (2D knee angle).
STANDING_MAX_KNEE_FLEXION_2D_DEG = 15.0
MIN_STANDING_FRAMES = 5
STANDING_FALLBACK_FRACTION = 0.2

# keep_world_frame: a camera left out of the consensus rejoins the world-frame fit when its
# landmark residual is within this multiple of the consensus cameras' own residual (or the floor).
WORLD_FIT_INLIER_FACTOR = 3.0
WORLD_FIT_INLIER_FLOOR_M = 0.002
MIN_CONSENSUS_CAMERAS = 3

SCALE_SOURCE_BAR = "bar"
SCALE_SOURCE_HEIGHT = "height"
SCALE_SOURCE_INITIAL = "initial"


@dataclass
class PersonCalibrationResult:
    calibration: CalibrationResult
    rms_reprojection_px: float
    scale_source: str
    frames_used: int
    bone_lengths_m: dict[str, float]
    initial_rms_reprojection_px: float = float("nan")
    standing_frames: int = 0
    solve_time_s: float = 0.0
    # Leg-segment lengths triangulated with the solved calibration / with the initial one, by the
    # cameras that did not move (median over the solved frames, same estimator on both sides):
    # 1.0 by construction when the scale comes from the initial calibration, NaN without one.
    scale_change_ratio: float = float("nan")


def _stack_skeletons(views: list[dict[str, Skeleton2D]], cam_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    # -> pixel positions (F, V, 21, 2) and raw scores (F, V, 21); absent cameras / keypoints score 0
    positions = np.zeros((len(views), len(cam_ids), NUM_KEYPOINTS, 2), dtype=np.float64)
    scores = np.zeros((len(views), len(cam_ids), NUM_KEYPOINTS), dtype=np.float64)
    for frame_idx, frame_views in enumerate(views):
        for view_idx, cam_id in enumerate(cam_ids):
            skeleton = frame_views.get(cam_id)
            if skeleton is None or not skeleton.keypoints:
                continue
            keypoints = skeleton.to_numpy()[:NUM_KEYPOINTS]
            positions[frame_idx, view_idx, : len(keypoints)] = keypoints[:, :2]
            scores[frame_idx, view_idx, : len(keypoints)] = keypoints[:, 2]
    finite = np.isfinite(positions).all(axis=-1) & np.isfinite(scores)
    return np.where(finite[..., None], positions, 0.0), np.where(finite, scores, 0.0)


def _stack_bar_ends(
    bar_ends: list[dict[str, BarbellDetection]] | None, cam_ids: list[str], num_frames: int
) -> tuple[np.ndarray, np.ndarray]:
    # -> image-left / image-right end positions (F, V, 2, 2) and confidences (F, V, 2)
    positions = np.zeros((num_frames, len(cam_ids), 2, 2), dtype=np.float64)
    scores = np.zeros((num_frames, len(cam_ids), 2), dtype=np.float64)
    if bar_ends is None:
        return positions, scores
    for frame_idx, frame_bars in enumerate(bar_ends[:num_frames]):
        for view_idx, cam_id in enumerate(cam_ids):
            detection = frame_bars.get(cam_id)
            if detection is None:
                continue
            for end_idx, end in enumerate((detection.left_end, detection.right_end)):
                positions[frame_idx, view_idx, end_idx] = (end.x, end.y)
                scores[frame_idx, view_idx, end_idx] = end.confidence
    finite = np.isfinite(positions).all(axis=-1) & np.isfinite(scores)
    return np.where(finite[..., None], positions, 0.0), np.where(finite, scores, 0.0)


def _undistort_views(positions: np.ndarray, scores: np.ndarray, cameras: list[CameraCalibration]) -> np.ndarray:
    # positions (F, V, N, 2), scores (F, V, N) -> detected points undistorted per camera with
    # calibration.undistort_keypoints, the same mapping the triangulator applies
    undistorted = positions.copy()
    for view_idx, camera in enumerate(cameras):
        detected = scores[:, view_idx] > 0.0
        undistorted[:, view_idx][detected] = undistort_keypoints(positions[:, view_idx][detected], camera)
    return undistorted


def _label_bar_ends_by_shoulders(
    bar_positions: np.ndarray, bar_scores: np.ndarray, positions: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # The detector orders bar ends by image x. Reorder them to (subject-left, subject-right)
    # using the 2D shoulder line, which carries the pose model's anatomical labels; views where
    # the shoulders are missing or seen end-on cannot be labelled and are dropped.
    shoulder_vector = positions[:, :, CK.LEFT_SHOULDER] - positions[:, :, CK.RIGHT_SHOULDER]
    shoulders_seen = (scores[:, :, CK.LEFT_SHOULDER] >= MIN_KEYPOINT_SCORE) & (
        scores[:, :, CK.RIGHT_SHOULDER] >= MIN_KEYPOINT_SCORE
    )
    bar_vector = bar_positions[:, :, 1] - bar_positions[:, :, 0]
    image_right_is_subject_left = np.sum(bar_vector * shoulder_vector, axis=-1) > 0.0
    usable = (
        shoulders_seen
        & (np.linalg.norm(shoulder_vector, axis=-1) >= MIN_BAR_SHOULDER_SPAN_PX)
        & (bar_scores >= MIN_KEYPOINT_SCORE).all(axis=-1)
    )
    flipped = image_right_is_subject_left[..., None, None]
    labelled_positions = np.where(flipped, bar_positions[:, :, ::-1], bar_positions)
    labelled_scores = np.where(flipped[..., 0], bar_scores[:, :, ::-1], bar_scores)
    return labelled_positions, np.where(usable[..., None], labelled_scores, 0.0)


def _knee_flexion_2d_deg(positions: np.ndarray, valid: np.ndarray) -> np.ndarray:
    # positions (F, V, P, 2), valid (F, V, P) -> largest 2D knee flexion over views and legs (F,), -inf if no leg
    flexions = []
    for hip, knee, ankle in ((CK.LEFT_HIP, CK.LEFT_KNEE, CK.LEFT_ANKLE), (CK.RIGHT_HIP, CK.RIGHT_KNEE, CK.RIGHT_ANKLE)):
        thigh = positions[:, :, hip] - positions[:, :, knee]
        shank = positions[:, :, ankle] - positions[:, :, knee]
        lengths = np.linalg.norm(thigh, axis=-1) * np.linalg.norm(shank, axis=-1)
        leg_seen = valid[:, :, hip] & valid[:, :, knee] & valid[:, :, ankle] & (lengths > 0.0)
        cosine = np.sum(thigh * shank, axis=-1) / np.where(leg_seen, lengths, 1.0)
        flexion_deg = 180.0 - np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        flexions.append(np.where(leg_seen, flexion_deg, -np.inf))
    return np.concatenate(flexions, axis=1).max(axis=1)


def _select_diverse_frames(
    descriptors: np.ndarray, present: np.ndarray, candidates: np.ndarray, count: int, selected: list[int]
) -> list[int]:
    # Greedy farthest-point selection among candidate frames; distance = RMS pixel difference
    # over the descriptor entries both frames have. Extends and returns `selected`.
    if count <= 0 or candidates.size == 0:
        return selected
    min_distance = np.full(len(descriptors), -1.0)
    min_distance[candidates] = np.inf

    def absorb(frame_idx: int) -> None:
        shared = present & present[frame_idx]
        squared = np.sum(np.where(shared, descriptors - descriptors[frame_idx], 0.0) ** 2, axis=1)
        distance = np.sqrt(squared / np.maximum(shared.sum(axis=1), 1))
        np.minimum(min_distance, distance, out=min_distance)
        min_distance[frame_idx] = -1.0

    for frame_idx in selected:
        absorb(frame_idx)
    for _ in range(count):
        next_idx = int(np.argmax(min_distance))
        if min_distance[next_idx] < 0.0:
            break
        selected.append(next_idx)
        absorb(next_idx)
    return selected


def _pose_descriptors(positions: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keypoints = list(DESCRIPTOR_KEYPOINTS)
    num_frames = len(positions)
    raw = np.where(valid[:, :, keypoints, None], positions[:, :, keypoints], np.nan).reshape(num_frames, -1)
    half_window = DESCRIPTOR_MEDIAN_WINDOW // 2
    padded = np.pad(raw, ((half_window, half_window), (0, 0)), constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, DESCRIPTOR_MEDIAN_WINDOW, axis=0)
    present = np.isfinite(windows).any(axis=-1)
    filtered = np.nanmedian(np.where(present[..., None], windows, 0.0), axis=-1)
    return np.where(present, filtered, 0.0), present


def _projection_matrices(
    intrinsic_matrices: np.ndarray, rotations: np.ndarray, translations: np.ndarray
) -> np.ndarray:
    return intrinsic_matrices @ np.concatenate([rotations, translations[..., None]], axis=-1)


def _triangulate(projections: np.ndarray, positions: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # projections (V, 3, 4), positions (V, N, 2), valid (V, N) -> points (N, 3), residuals px (V, N)
    # (0 for unused views, inf where the point is behind a camera or unsolvable)
    normals = _dlt_normal_matrices(projections, positions)
    normal_sums = np.einsum("vn,vnij->nij", valid, normals)[None]
    points, residuals = _solve_and_measure(projections, normal_sums, positions, valid[None])
    return points[0], residuals[0]


def _robust_rms_px(residuals_px: np.ndarray) -> float:
    # RMS over inliers; the gate scales with the median so a drifted rig still reads high
    finite = residuals_px[np.isfinite(residuals_px)]
    if finite.size == 0:
        return float("nan")
    gate_px = max(OUTLIER_MEDIAN_FACTOR * float(np.median(finite)), np.finfo(np.float64).tiny)
    inliers = finite[finite <= gate_px]
    return float(np.sqrt(np.mean(inliers**2)))


def _multi_view_residuals_px(
    projections: np.ndarray, positions: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # positions (F, V, P, 2), valid (F, V, P) -> residuals (V, F*P) and the mask of observations
    # belonging to points seen by >= 2 views
    num_frames, num_views, num_points = valid.shape
    flat_positions = positions.transpose(1, 0, 2, 3).reshape(num_views, num_frames * num_points, 2)
    flat_valid = valid.transpose(1, 0, 2).reshape(num_views, num_frames * num_points)
    flat_valid = flat_valid & (flat_valid.sum(axis=0) >= MIN_VIEWS_PER_POINT)
    _, residuals = _triangulate(projections, flat_positions, flat_valid)
    return residuals, flat_valid


def _relative_pose(
    normalized_from: np.ndarray, normalized_to: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    # -> rotation R and unit translation t taking camera-frame points from -> to: x_to = R x_from + t
    identity = np.eye(3)
    cv2.setRNGSeed(RANSAC_SEED)
    essential, inlier_mask = cv2.findEssentialMat(
        normalized_from, normalized_to, identity, method=cv2.RANSAC, prob=RANSAC_CONFIDENCE, threshold=threshold
    )
    if essential is None or essential.shape != (3, 3):
        raise ValueError("essential-matrix initialisation failed: no consistent relative camera pose")
    _, rotation, translation, _ = cv2.recoverPose(essential, normalized_from, normalized_to, identity, mask=inlier_mask)
    return rotation, translation.ravel()


def _translation_scale(
    points_partner: np.ndarray, normalized_new: np.ndarray, rotation: np.ndarray, direction: np.ndarray
) -> float:
    # x_new = R x_partner + scale * direction, observed as normalized image points -> robust 1-D least squares
    rotated = points_partner @ rotation.T
    lhs = np.concatenate([
        normalized_new[:, 0] * direction[2] - direction[0],
        normalized_new[:, 1] * direction[2] - direction[1],
    ])
    rhs = np.concatenate([
        rotated[:, 0] - normalized_new[:, 0] * rotated[:, 2],
        rotated[:, 1] - normalized_new[:, 1] * rotated[:, 2],
    ])
    scale = float(lhs @ rhs / (lhs @ lhs))
    errors = np.abs(lhs * scale - rhs)
    inliers = errors <= OUTLIER_MEDIAN_FACTOR * max(float(np.median(errors)), np.finfo(np.float64).tiny)
    return float(lhs[inliers] @ rhs[inliers] / (lhs[inliers] @ lhs[inliers]))


def _essential_matrix_cameras(
    positions: np.ndarray, scores: np.ndarray, intrinsic_matrices: np.ndarray, cam_ids: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    # Poses relative to camera 0 (identity), first baseline = 1. Each further camera is attached
    # through the registered camera it shares the most confident keypoints with.
    num_views = scores.shape[1]
    confident = (scores >= MIN_INIT_KEYPOINT_SCORE).transpose(1, 0, 2).reshape(num_views, -1)
    pixels = positions.transpose(1, 0, 2, 3).reshape(num_views, -1, 2)
    homogeneous = np.concatenate([pixels, np.ones(pixels.shape[:-1] + (1,))], axis=-1)
    normalized = np.einsum("vij,vnj->vni", np.linalg.inv(intrinsic_matrices), homogeneous)[..., :2]
    mean_focal_px = float(np.mean(intrinsic_matrices[:, 0, 0]))
    threshold = RANSAC_THRESHOLD_PX / mean_focal_px

    rotations = np.tile(np.eye(3), (num_views, 1, 1))
    translations = np.zeros((num_views, 3))
    registered = [0]
    while len(registered) < num_views:
        shared_counts = {
            (new, partner): int(np.sum(confident[new] & confident[partner]))
            for new in range(num_views) if new not in registered for partner in registered
        }
        (new, partner), count = max(shared_counts.items(), key=lambda item: item[1])
        if count < MIN_PAIR_CORRESPONDENCES:
            raise ValueError(
                f"camera {cam_ids[new]} shares only {count} confident keypoints with the other cameras "
                f"(need {MIN_PAIR_CORRESPONDENCES})"
            )
        shared = confident[new] & confident[partner]
        rotation, direction = _relative_pose(normalized[partner][shared], normalized[new][shared], threshold)
        scale = 1.0
        if len(registered) >= 2:
            registered_valid = np.zeros_like(confident)
            registered_valid[registered] = confident[registered]
            known = shared & (registered_valid.sum(axis=0) >= MIN_VIEWS_PER_POINT)
            if known.sum() < MIN_PAIR_CORRESPONDENCES:
                raise ValueError(f"camera {cam_ids[new]} cannot be scaled against the registered cameras")
            normalized_projections = np.concatenate([rotations, translations[..., None]], axis=-1)
            points, _ = _triangulate(normalized_projections, normalized[:, known], registered_valid[:, known])
            points_partner = points @ rotations[partner].T + translations[partner]
            scale = _translation_scale(points_partner, normalized[new][known], rotation, direction)
        rotations[new] = rotation @ rotations[partner]
        translations[new] = rotation @ translations[partner] + scale * direction
        registered.append(new)
    return rotations, translations


def _cameras_relative_to_reference(
    initial: CalibrationResult, cam_ids: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    missing = [cam_id for cam_id in cam_ids if cam_id not in initial.cameras]
    if missing:
        raise ValueError(f"initial calibration has no camera(s) {missing}")
    world_rotations = np.stack([np.asarray(initial.cameras[c].rotation_matrix, dtype=np.float64) for c in cam_ids])
    world_translations = np.stack(
        [np.asarray(initial.cameras[c].translation_vector, dtype=np.float64).ravel() for c in cam_ids]
    )
    reference_inverse = world_rotations[0].T
    rotations = world_rotations @ reference_inverse
    translations = world_translations - np.einsum("vij,j->vi", rotations, world_translations[0])
    return rotations, translations


def _camera_params(rotations: np.ndarray, translations: np.ndarray) -> np.ndarray:
    # [rvec, tvec] of every non-reference camera, flattened
    return np.concatenate([
        np.concatenate([cv2.Rodrigues(rotations[view_idx])[0].ravel(), translations[view_idx]])
        for view_idx in range(1, len(rotations))
    ])


def _cameras_from_params(camera_params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = camera_params.reshape(-1, 6)
    rotations = np.stack([np.eye(3)] + [cv2.Rodrigues(row[:3])[0] for row in rows])
    return rotations, np.vstack([np.zeros(3), rows[:, 3:]])


def _refine_cameras_by_triangulation(
    intrinsic_matrices: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
    positions: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # Camera-only solve: every evaluation re-triangulates all points (DLT) with the candidate
    # cameras, so no point can sit in a wrong place. Converges from the 20-30 degree errors an
    # essential-matrix or T-pose start can have, where the full bundle adjustment can lock onto
    # a wrong minimum. The first baseline is held to keep the scale.
    num_views = len(intrinsic_matrices)
    flat_positions = positions.transpose(1, 0, 2, 3).reshape(num_views, -1, 2)
    flat_weights = weights.transpose(1, 0, 2).reshape(num_views, -1)
    flat_weights = flat_weights * ((flat_weights > 0.0).sum(axis=0) >= MIN_VIEWS_PER_POINT)
    seen = (flat_weights > 0.0).any(axis=0)
    flat_positions, flat_weights = flat_positions[:, seen], flat_weights[:, seen]
    observed = flat_weights > 0.0
    baseline_m = float(np.linalg.norm(translations[1]))

    def residuals(camera_params: np.ndarray) -> np.ndarray:
        candidate_rotations, candidate_translations = _cameras_from_params(camera_params)
        projections = _projection_matrices(intrinsic_matrices, candidate_rotations, candidate_translations)
        points, _ = _triangulate(projections, flat_positions, observed)
        homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
        projected = np.einsum("vij,nj->vni", projections, homogeneous)
        depth = np.maximum(projected[..., 2], MIN_DEPTH_M)
        reprojection = (projected[..., :2] / depth[..., None] - flat_positions) * flat_weights[..., None]
        baseline = SCALE_WEIGHT_PX_PER_M * (np.linalg.norm(candidate_translations[1]) - baseline_m)
        return np.append(reprojection.ravel(), baseline)

    solution = least_squares(
        residuals, _camera_params(rotations, translations), method="trf", loss=APPROACH_LOSS,
        f_scale=HUBER_DELTA_PX, x_scale="jac", max_nfev=MAX_CAMERA_ONLY_EVALUATIONS,
    )
    return _cameras_from_params(solution.x)


def _median_segment_lengths(
    points: np.ndarray, active: np.ndarray, min_frames: int = MIN_SEGMENT_FRAMES
) -> np.ndarray:
    # points (F, P, 3), active (F, P) -> median length per BA segment, NaN when seen in too few frames
    lengths = np.full(len(BA_SEGMENT_NAMES), np.nan)
    for segment_idx, name in enumerate(BA_SEGMENT_NAMES):
        proximal, distal = BA_SEGMENTS[name]
        both = active[:, proximal] & active[:, distal]
        if both.sum() >= min_frames:
            lengths[segment_idx] = np.median(np.linalg.norm(points[both, proximal] - points[both, distal], axis=1))
    return lengths


def _height_prior_sum(segment_lengths: np.ndarray, height_m: float) -> tuple[np.ndarray, float]:
    # -> indices of solved segments with a height ratio, and their expected summed length
    indices = np.array([
        idx for idx, name in enumerate(BA_SEGMENT_NAMES)
        if name in SEGMENT_HEIGHT_RATIOS and np.isfinite(segment_lengths[idx])
    ], dtype=np.int64)
    expected_sum_m = height_m * sum(SEGMENT_HEIGHT_RATIOS[BA_SEGMENT_NAMES[idx]] for idx in indices)
    return indices, float(expected_sum_m)


def _triangulated_segment_lengths(
    projections: np.ndarray, positions: np.ndarray, valid: np.ndarray, cameras_used: np.ndarray
) -> np.ndarray:
    # Median length per BA segment (NaN below MIN_SCALE_TARGET_FRAMES frames) of the points
    # triangulated with projections (V, 3, 4) from the cameras_used (V,) views only
    num_frames, num_views, num_points = valid.shape
    subset_valid = valid & cameras_used[None, :, None]
    subset_valid &= (subset_valid.sum(axis=1) >= MIN_VIEWS_PER_POINT)[:, None, :]
    points, _ = _triangulate(
        projections, positions.transpose(1, 0, 2, 3).reshape(num_views, -1, 2),
        subset_valid.transpose(1, 0, 2).reshape(num_views, -1),
    )
    return _median_segment_lengths(
        points.reshape(num_frames, num_points, 3), subset_valid.any(axis=1), min_frames=MIN_SCALE_TARGET_FRAMES
    )


class _BundleProblem:
    """Residuals and Jacobian sparsity of the joint camera / joint / segment-length solve.

    Parameters: [rvec, tvec per non-reference camera | xyz per active point | segment lengths].
    The reference camera is the identity, so points live in its coordinate frame.
    """

    def __init__(
        self,
        intrinsic_matrices: np.ndarray,
        positions: np.ndarray,
        weights: np.ndarray,
        segment_lengths: np.ndarray,
        scale_source: str,
        bar_length_m: float | None,
        scale_prior: tuple[np.ndarray, float] | None,
        baseline_m: float,
    ) -> None:
        # positions (F, V, NUM_FRAME_POINTS, 2); weights (F, V, NUM_FRAME_POINTS), 0 = unobserved
        num_views = weights.shape[1]
        observed = weights > 0.0
        self.active = observed.sum(axis=1) >= MIN_VIEWS_PER_POINT
        observed &= self.active[:, None, :]
        self.point_index = np.full(self.active.shape, -1, dtype=np.int64)
        self.point_index[self.active] = np.arange(int(self.active.sum()))
        self.num_points = int(self.active.sum())

        frame_idx, view_idx, point_idx = np.nonzero(observed)
        self.obs_view = view_idx
        self.obs_point = self.point_index[frame_idx, point_idx]
        self.obs_xy = positions[frame_idx, view_idx, point_idx]
        self.obs_weight = weights[frame_idx, view_idx, point_idx]
        self.obs_location = (frame_idx, view_idx, point_idx)
        self._focal = intrinsic_matrices[view_idx][:, [0, 1], [0, 1]]
        self._skew = intrinsic_matrices[view_idx][:, 0, 1]
        self._principal = intrinsic_matrices[view_idx][:, :2, 2]

        self.segment_solved = np.isfinite(segment_lengths)
        bone_proximal, bone_distal, bone_segment = [], [], []
        for segment_idx, name in enumerate(BA_SEGMENT_NAMES):
            if not self.segment_solved[segment_idx]:
                continue
            proximal, distal = BA_SEGMENTS[name]
            both = self.active[:, proximal] & self.active[:, distal]
            bone_proximal.append(self.point_index[both, proximal])
            bone_distal.append(self.point_index[both, distal])
            bone_segment.append(np.full(int(both.sum()), segment_idx))
        self.bone_proximal = np.concatenate(bone_proximal) if bone_proximal else np.zeros(0, dtype=np.int64)
        self.bone_distal = np.concatenate(bone_distal) if bone_distal else np.zeros(0, dtype=np.int64)
        self.bone_segment = np.concatenate(bone_segment) if bone_segment else np.zeros(0, dtype=np.int64)

        self.scale_source = scale_source
        self.bar_length_m = bar_length_m
        self.scale_prior = scale_prior
        self.baseline_m = baseline_m
        bar_both = self.active[:, BAR_LEFT_POINT] & self.active[:, BAR_RIGHT_POINT]
        self.bar_left = self.point_index[bar_both, BAR_LEFT_POINT]
        self.bar_right = self.point_index[bar_both, BAR_RIGHT_POINT]

        self.num_camera_params = 6 * (num_views - 1)
        self.points_end = self.num_camera_params + 3 * self.num_points
        self.num_params = self.points_end + len(BA_SEGMENT_NAMES)

    def pack(
        self, rotations: np.ndarray, translations: np.ndarray, points: np.ndarray, lengths: np.ndarray
    ) -> np.ndarray:
        # points (F, NUM_FRAME_POINTS, 3)
        return np.concatenate([
            _camera_params(rotations, translations), points[self.active].ravel(), np.nan_to_num(lengths),
        ])

    def unpack(self, params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rotations, translations = _cameras_from_params(params[: self.num_camera_params])
        points = params[self.num_camera_params: self.points_end].reshape(-1, 3)
        return rotations, translations, points, params[self.points_end:]

    def frame_points(self, points: np.ndarray) -> np.ndarray:
        frame_points = np.zeros(self.active.shape + (3,))
        frame_points[self.active] = points
        return frame_points

    def reprojection_errors_px(self, params: np.ndarray) -> np.ndarray:
        rotations, translations, points, _ = self.unpack(params)
        return np.linalg.norm(self._reprojection(rotations, translations, points), axis=1)

    def _reprojection(self, rotations: np.ndarray, translations: np.ndarray, points: np.ndarray) -> np.ndarray:
        camera_points = np.einsum("mij,mj->mi", rotations[self.obs_view], points[self.obs_point])
        camera_points += translations[self.obs_view]
        depth = np.maximum(camera_points[:, 2], MIN_DEPTH_M)
        x_over_z = camera_points[:, 0] / depth
        y_over_z = camera_points[:, 1] / depth
        projected = np.stack([
            self._focal[:, 0] * x_over_z + self._skew * y_over_z + self._principal[:, 0],
            self._focal[:, 1] * y_over_z + self._principal[:, 1],
        ], axis=1)
        return projected - self.obs_xy

    def residuals(self, params: np.ndarray) -> np.ndarray:
        rotations, translations, points, lengths = self.unpack(params)
        reprojection = self._reprojection(rotations, translations, points) * self.obs_weight[:, None]
        bone_lengths = np.linalg.norm(points[self.bone_proximal] - points[self.bone_distal], axis=1)
        bones = BONE_WEIGHT_PX_PER_M * (bone_lengths - lengths[self.bone_segment])
        return np.concatenate([reprojection.ravel(), bones, self._scale_residuals(translations, points, lengths)])

    def _scale_residuals(self, translations: np.ndarray, points: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        if self.scale_source == SCALE_SOURCE_BAR:
            bar_lengths = np.linalg.norm(points[self.bar_left] - points[self.bar_right], axis=1)
            return SCALE_WEIGHT_PX_PER_M * (bar_lengths - self.bar_length_m)
        if self.scale_prior is not None:
            indices, expected_sum_m = self.scale_prior
            return SCALE_WEIGHT_PX_PER_M * np.array([lengths[indices].sum() - expected_sum_m])
        return SCALE_WEIGHT_PX_PER_M * np.array([np.linalg.norm(translations[1]) - self.baseline_m])

    def jacobian_sparsity(self) -> coo_matrix:
        num_obs = len(self.obs_view)
        point_columns = self.num_camera_params + 3 * self.obs_point[:, None] + np.arange(3)
        rows = [np.repeat(np.arange(2 * num_obs), 3)]
        cols = [np.repeat(point_columns, 2, axis=0).ravel()]
        moving = np.nonzero(self.obs_view > 0)[0]
        camera_columns = 6 * (self.obs_view[moving, None] - 1) + np.arange(6)
        rows.append(np.repeat(np.stack([2 * moving, 2 * moving + 1], axis=1).ravel(), 6))
        cols.append(np.repeat(camera_columns, 2, axis=0).ravel())

        next_row = 2 * num_obs
        bone_rows = next_row + np.arange(len(self.bone_proximal))
        for point_indices in (self.bone_proximal, self.bone_distal):
            rows.append(np.repeat(bone_rows, 3))
            cols.append((self.num_camera_params + 3 * point_indices[:, None] + np.arange(3)).ravel())
        rows.append(bone_rows)
        cols.append(self.points_end + self.bone_segment)
        next_row += len(self.bone_proximal)

        if self.scale_source == SCALE_SOURCE_BAR:
            bar_rows = next_row + np.arange(len(self.bar_left))
            for point_indices in (self.bar_left, self.bar_right):
                rows.append(np.repeat(bar_rows, 3))
                cols.append((self.num_camera_params + 3 * point_indices[:, None] + np.arange(3)).ravel())
            next_row += len(self.bar_left)
        elif self.scale_prior is not None:
            indices = self.scale_prior[0]
            rows.append(np.full(len(indices), next_row))
            cols.append(self.points_end + indices)
            next_row += 1
        else:
            rows.append(np.full(3, next_row))
            cols.append(np.arange(3, 6))
            next_row += 1

        all_rows = np.concatenate(rows)
        all_cols = np.concatenate(cols)
        return coo_matrix(
            (np.ones(len(all_rows), dtype=np.int8), (all_rows, all_cols)), shape=(next_row, self.num_params)
        )


def _camera_landmarks(rotations: np.ndarray, translations: np.ndarray, lever_arms_m: np.ndarray) -> np.ndarray:
    # Per camera (V, 4, 3): its centre and the tips of its three axes, each lever_arm long. With
    # the lever arm = distance to the lifter, a landmark residual is how far that camera's pose
    # disagreement moves the working volume (the optical-axis tip sits on the lifter).
    centres = -np.einsum("vji,vj->vi", rotations, translations)
    axis_tips = centres[:, None, :] + lever_arms_m[:, None, None] * rotations
    return np.concatenate([centres[:, None, :], axis_tips], axis=1)


def _rigid_fit(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Least-squares rotation and shift (no scale) with target ~= rotation @ source + shift; points (N, 3)
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    left, _, right = np.linalg.svd((target - target_mean).T @ (source - source_mean))
    rotation = left @ np.diag([1.0, 1.0, np.sign(np.linalg.det(left @ right))]) @ right
    return rotation, target_mean - rotation @ source_mean


def _similarity_fit(
    source_centres: np.ndarray, source_axes: np.ndarray, target_centres: np.ndarray, target_axes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    # target ~= scale * rotation @ source + shift for camera centres (V, 3), with the rotation taken
    # from the unit camera axes (V, 3, 3) alone: axes do not scale, so the fit stays exact whatever
    # scale the source rig was solved at. One camera cannot fix a scale: it is then 1.
    left, _, right = np.linalg.svd(target_axes.reshape(-1, 3).T @ source_axes.reshape(-1, 3))
    rotation = left @ np.diag([1.0, 1.0, np.sign(np.linalg.det(left @ right))]) @ right
    rotated = source_centres @ rotation.T
    rotated_spread = rotated - rotated.mean(axis=0)
    scale = 1.0
    if len(source_centres) >= MIN_VIEWS_PER_POINT:
        target_spread = target_centres - target_centres.mean(axis=0)
        scale = float(np.sum(rotated_spread * target_spread) / np.sum(rotated_spread**2))
    return rotation, target_centres.mean(axis=0) - scale * rotated.mean(axis=0), scale


def _consensus_rigid_fit(
    solved_landmarks: np.ndarray, initial_landmarks: np.ndarray, with_scale: bool = False
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    # Map of the solved rig onto the initial one that tolerates ONE moved camera: the
    # leave-one-out subset whose cameras agree best defines the fit, and the camera left out
    # rejoins only if it agrees about as well. With two cameras nothing can vote, so the
    # reference camera's pose is kept. Landmarks (V, 4, 3): centre then three axis tips.
    # -> rotation, shift, scale (1 unless with_scale), cameras used (V,).
    num_views = len(solved_landmarks)
    solved_centres, initial_centres = solved_landmarks[:, 0], initial_landmarks[:, 0]
    solved_offsets = solved_landmarks[:, 1:] - solved_centres[:, None]
    initial_offsets = initial_landmarks[:, 1:] - initial_centres[:, None]
    lever_arms_m = np.linalg.norm(solved_offsets, axis=-1)

    def fit(cameras_used: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        if with_scale:
            rotation, shift, scale = _similarity_fit(
                solved_centres[cameras_used], solved_offsets[cameras_used] / lever_arms_m[cameras_used][..., None],
                initial_centres[cameras_used], initial_offsets[cameras_used] / lever_arms_m[cameras_used][..., None],
            )
        else:
            rotation, shift = _rigid_fit(
                solved_landmarks[cameras_used].reshape(-1, 3), initial_landmarks[cameras_used].reshape(-1, 3)
            )
            scale = 1.0
        mapped_centres = scale * solved_centres @ rotation.T + shift
        mapped_tips = mapped_centres[:, None] + solved_offsets @ rotation.T
        mapped = np.concatenate([mapped_centres[:, None], mapped_tips], axis=1)
        residuals_m = np.sqrt(np.mean(np.sum((mapped - initial_landmarks) ** 2, axis=-1), axis=-1))
        return rotation, shift, scale, residuals_m

    if num_views < MIN_CONSENSUS_CAMERAS:
        reference_only = np.arange(num_views) == 0
        rotation, shift, scale, _ = fit(reference_only)
        return rotation, shift, scale, reference_only

    best_left_out, best_rms_m = 0, np.inf
    for left_out in range(num_views):
        cameras_used = np.arange(num_views) != left_out
        residuals_m = fit(cameras_used)[3]
        subset_rms_m = float(np.sqrt(np.mean(residuals_m[cameras_used] ** 2)))
        if subset_rms_m < best_rms_m:
            best_left_out, best_rms_m = left_out, subset_rms_m
    cameras_used = np.arange(num_views) != best_left_out
    rotation, shift, scale, residuals_m = fit(cameras_used)
    if residuals_m[best_left_out] <= max(WORLD_FIT_INLIER_FACTOR * best_rms_m, WORLD_FIT_INLIER_FLOOR_M):
        cameras_used = np.ones(num_views, dtype=bool)
        rotation, shift, scale, _ = fit(cameras_used)
    return rotation, shift, scale, cameras_used


def world_frame_from_standing(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    World axes and origin from standing-frame joints (F, >=17, 3) given in any frame: Y (down)
    is the typical hip-midpoint to ankle-midpoint direction, X the typical right-hip to
    left-hip vector made perpendicular to Y, Z = X x Y (the subject's back); origin = typical
    hip midpoint. "Typical" is the per-component median over frames, so a few frames with a
    mis-detected ankle or hip do not tilt the frame. Returns the rotation whose rows are the
    world axes, and the origin: world = rotation @ (p - origin).
    """
    hip_mid = (points[:, CK.LEFT_HIP] + points[:, CK.RIGHT_HIP]) / 2.0
    ankle_mid = (points[:, CK.LEFT_ANKLE] + points[:, CK.RIGHT_ANKLE]) / 2.0
    down = ankle_mid - hip_mid
    down = np.median(down / np.linalg.norm(down, axis=1, keepdims=True), axis=0)
    y_axis = down / np.linalg.norm(down)
    left = np.median(points[:, CK.LEFT_HIP] - points[:, CK.RIGHT_HIP], axis=0)
    left = left - np.dot(left, y_axis) * y_axis
    x_axis = left / np.linalg.norm(left)
    z_axis = np.cross(x_axis, y_axis)
    return np.stack([x_axis, y_axis, z_axis]), np.median(hip_mid, axis=0)


def reprojection_health_px(calibration: CalibrationResult, views: list[dict[str, Skeleton2D]]) -> float:
    """
    Robust RMS reprojection residual (px) of the frames' keypoints triangulated with the
    calibration: a drift monitor. Gross detection outliers (beyond 4x the median residual)
    are left out, so the value tracks the rig rather than the detector. NaN when nothing
    is seen by two cameras.
    """
    cam_ids = sorted(calibration.cameras)
    cameras = [calibration.cameras[cam_id] for cam_id in cam_ids]
    projections = np.stack([np.asarray(camera.projection_matrix, dtype=np.float64) for camera in cameras])
    positions, scores = _stack_skeletons(views, cam_ids)
    positions = _undistort_views(positions, scores, cameras)
    residuals, used = _multi_view_residuals_px(projections, positions, scores >= MIN_KEYPOINT_SCORE)
    return _robust_rms_px(residuals[used])


class PersonCalibrator:
    """Solves camera extrinsics from a moving person; intrinsics (K, distortion) are given per camera."""

    def __init__(
        self,
        intrinsics: dict[str, tuple[np.ndarray, np.ndarray]],
        resolution: tuple[int, int],
        bar_length_m: float | None = None,
        height_m: float | None = None,
    ) -> None:
        if len(intrinsics) < 2:
            raise ValueError(f"person calibration needs >= 2 cameras, got {len(intrinsics)}")
        self._cam_ids = sorted(intrinsics)
        self._intrinsic_matrices = np.stack(
            [np.asarray(intrinsics[cam_id][0], dtype=np.float64).reshape(3, 3) for cam_id in self._cam_ids]
        )
        self._distortion_coeffs = [
            np.asarray(intrinsics[cam_id][1], dtype=np.float64).ravel() for cam_id in self._cam_ids
        ]
        self._resolution = resolution
        # Intrinsics-only cameras (no pose yet) so detections go through undistort_keypoints
        self._unposed_cameras = [
            CameraCalibration(
                camera_id=cam_id,
                projection_matrix=np.zeros((3, 4)),
                intrinsic_matrix=self._intrinsic_matrices[view_idx],
                rotation_matrix=np.eye(3),
                translation_vector=np.zeros((3, 1)),
                reprojection_error=0.0,
                resolution=resolution,
                distortion_coeffs=self._distortion_coeffs[view_idx],
            )
            for view_idx, cam_id in enumerate(self._cam_ids)
        ]
        self._bar_length_m = bar_length_m
        self._height_m = height_m

    def calibrate(
        self,
        views: list[dict[str, Skeleton2D]],
        bar_ends: list[dict[str, BarbellDetection]] | None = None,
        initial: CalibrationResult | None = None,
        keep_world_frame: bool = False,
    ) -> PersonCalibrationResult:
        """
        Bundle-adjust the rig from per-frame 2D skeletons (cam_id -> Skeleton2D, raw scores)
        and optional per-frame barbell detections. Without `initial`, cameras start from
        pairwise essential matrices. The world frame is re-anchored on the standing lifter,
        unless keep_world_frame is set with an `initial`: then the solved rig is rigidly
        aligned (no scale change) onto the initial cameras, tolerating one moved camera, so
        origin, heading and vertical do not jump. Raises ValueError when too few frames are
        seen by two cameras or a camera cannot be linked to the others.
        """
        start_time = time.perf_counter()
        positions, scores = _stack_skeletons(views, self._cam_ids)
        bar_positions, bar_scores = _stack_bar_ends(bar_ends, self._cam_ids, len(views))
        bar_positions, bar_scores = _label_bar_ends_by_shoulders(bar_positions, bar_scores, positions, scores)
        scores = np.concatenate([scores, bar_scores], axis=2)
        positions = _undistort_views(np.concatenate([positions, bar_positions], axis=2), scores, self._unposed_cameras)
        valid = scores >= MIN_KEYPOINT_SCORE

        use_bar = self._bar_length_m is not None
        frame_indices = self._select_frames(positions, valid, use_bar)
        all_positions, all_valid = positions[:, :, :NUM_KEYPOINTS], valid[:, :, :NUM_KEYPOINTS]
        positions, scores, valid = positions[frame_indices], scores[frame_indices], valid[frame_indices]
        bar_frames = (valid[:, :, BAR_LEFT_POINT:].all(axis=-1).sum(axis=1) >= MIN_VIEWS_PER_POINT).sum()
        use_bar = use_bar and bar_frames >= MIN_BAR_FRAMES
        if not use_bar:
            valid[:, :, BAR_LEFT_POINT:] = False

        if initial is not None:
            rotations, translations = _cameras_relative_to_reference(initial, self._cam_ids)
        else:
            rotations, translations = _essential_matrix_cameras(
                positions[:, :, :NUM_KEYPOINTS], scores[:, :, :NUM_KEYPOINTS], self._intrinsic_matrices, self._cam_ids
            )
        weights = self._observation_weights(scores, valid)
        initial_rms_px = self._rms_reprojection_px(rotations, translations, positions, valid)
        rotations, translations = _refine_cameras_by_triangulation(
            self._intrinsic_matrices, rotations, translations, positions, weights
        )
        points, active = self._triangulate_frames(rotations, translations, positions, weights > 0.0)

        if use_bar:
            scale_source = SCALE_SOURCE_BAR
        elif self._height_m is not None:
            scale_source = SCALE_SOURCE_HEIGHT
        else:
            scale_source = SCALE_SOURCE_INITIAL
            if initial is None:
                logger.warning("[PERSON CALIBRATION] No bar, height or initial calibration: scale is arbitrary")
        if initial is None and scale_source != SCALE_SOURCE_INITIAL:
            initial_scale = self._initial_scale(points, active, scale_source)
            translations = translations * initial_scale
            points = points * initial_scale

        params, problem = self._solve(rotations, translations, points, active, positions, weights, scale_source)
        rotations, translations, solved_points, lengths = problem.unpack(params)
        scale_change_ratio = float("nan")
        if initial is not None:
            scale_change_ratio = self._scale_change_ratio(
                rotations, translations, solved_points, positions[:, :, :NUM_KEYPOINTS], valid[:, :, :NUM_KEYPOINTS],
                initial,
            )
            if scale_source == SCALE_SOURCE_INITIAL and np.isfinite(scale_change_ratio):
                # scale is a free gauge of this solve (baseline hold only): put it back to the initial metric
                translations, solved_points, lengths = (
                    translations / scale_change_ratio, solved_points / scale_change_ratio, lengths / scale_change_ratio
                )
                scale_change_ratio = 1.0
            elif scale_source == SCALE_SOURCE_INITIAL:
                logger.warning(
                    "[PERSON CALIBRATION] Too few frames to triangulate the leg segments; holding the first "
                    "camera baseline instead (metric scale not guaranteed if one of those cameras moved)"
                )

        world_anchor = WORLD_ANCHOR_PERSON
        standing_frames = 0
        if initial is not None and keep_world_frame:
            world_rotation, world_origin = self._initial_world_frame(rotations, translations, solved_points, initial)
            world_anchor = initial.world_anchor
        else:
            standing_points = self._standing_points(rotations, translations, all_positions, all_valid)
            world_rotation, world_origin = world_frame_from_standing(standing_points)
            standing_frames = len(standing_points)
        # world = A (p - o)  ->  camera = R A^T world + (R o + t)
        translations = translations + np.einsum("vij,j->vi", rotations, world_origin)
        rotations = rotations @ world_rotation.T

        rms_px = self._rms_reprojection_px(rotations, translations, positions, valid)
        bone_lengths_m = {
            name: float(lengths[idx]) for idx, name in enumerate(BA_SEGMENT_NAMES) if problem.segment_solved[idx]
        }
        calibration = self._build_calibration(
            rotations, translations, positions, valid, bone_lengths_m, scale_source, initial, world_anchor
        )
        solve_time_s = time.perf_counter() - start_time
        logger.info(
            "[PERSON CALIBRATION] %d frames, scale from %s: reprojection RMS %.2f -> %.2f px in %.1f s",
            len(frame_indices), scale_source, initial_rms_px, rms_px, solve_time_s,
        )
        return PersonCalibrationResult(
            calibration=calibration,
            rms_reprojection_px=rms_px,
            scale_source=scale_source,
            frames_used=len(frame_indices),
            bone_lengths_m=bone_lengths_m,
            initial_rms_reprojection_px=initial_rms_px,
            standing_frames=standing_frames,
            solve_time_s=solve_time_s,
            scale_change_ratio=scale_change_ratio,
        )

    def refine(
        self,
        calibration: CalibrationResult,
        views: list[dict[str, Skeleton2D]],
        bar_ends: list[dict[str, BarbellDetection]] | None = None,
        keep_world_frame: bool = False,
    ) -> PersonCalibrationResult:
        return self.calibrate(views, bar_ends=bar_ends, initial=calibration, keep_world_frame=keep_world_frame)

    def _select_frames(self, positions: np.ndarray, valid: np.ndarray, use_bar: bool) -> np.ndarray:
        body_valid = valid[:, :, :NUM_KEYPOINTS]
        views_seeing = (body_valid.sum(axis=-1) >= MIN_KEYPOINTS_PER_VIEW).sum(axis=1)
        usable = np.nonzero(views_seeing >= MIN_VIEWS_PER_POINT)[0]
        if len(usable) < MIN_FRAMES:
            raise ValueError(
                f"person calibration needs >= {MIN_FRAMES} frames seen by two cameras, got {len(usable)}"
            )
        if len(usable) <= MAX_FRAMES:
            return usable
        descriptors, present = _pose_descriptors(positions[:, :, :NUM_KEYPOINTS], body_valid)
        selected: list[int] = []
        if use_bar:
            bar_views = valid[:, :, BAR_LEFT_POINT:].all(axis=-1).sum(axis=1)
            bar_usable = usable[bar_views[usable] >= MIN_VIEWS_PER_POINT]
            bar_budget = int(MAX_FRAMES * BAR_FRAME_BUDGET_RATIO)
            selected = _select_diverse_frames(descriptors, present, bar_usable, bar_budget, selected)
        remaining = np.setdiff1d(usable, selected)
        selected = _select_diverse_frames(descriptors, present, remaining, MAX_FRAMES - len(selected), selected)
        return np.sort(np.array(selected, dtype=np.int64))

    def _observation_weights(self, scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
        median_score = float(np.median(scores[valid])) if valid.any() else 1.0
        weights = np.where(valid, scores / median_score, 0.0)
        weights[:, :, list(FACE_KEYPOINTS)] *= FACE_KEYPOINT_WEIGHT
        return weights

    def _triangulate_frames(
        self, rotations: np.ndarray, translations: np.ndarray, positions: np.ndarray, observed: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        # -> points (F, NUM_FRAME_POINTS, 3) and the mask of points seen by >= 2 views
        num_frames, num_views, num_points = observed.shape
        projections = _projection_matrices(self._intrinsic_matrices, rotations, translations)
        flat_positions = positions.transpose(1, 0, 2, 3).reshape(num_views, -1, 2)
        flat_observed = observed.transpose(1, 0, 2).reshape(num_views, -1)
        points, _ = _triangulate(projections, flat_positions, flat_observed)
        active = observed.sum(axis=1) >= MIN_VIEWS_PER_POINT
        return points.reshape(num_frames, num_points, 3), active

    def _initial_scale(self, points: np.ndarray, active: np.ndarray, scale_source: str) -> float:
        if scale_source == SCALE_SOURCE_BAR:
            both = active[:, BAR_LEFT_POINT] & active[:, BAR_RIGHT_POINT]
            bar_lengths = np.linalg.norm(points[both, BAR_LEFT_POINT] - points[both, BAR_RIGHT_POINT], axis=1)
            return float(self._bar_length_m / np.median(bar_lengths))
        segment_lengths = _median_segment_lengths(points, active)
        indices, expected_sum_m = _height_prior_sum(segment_lengths, self._height_m)
        if len(indices) == 0:
            raise ValueError("height prior needs at least one rigid segment seen by two cameras")
        return float(expected_sum_m / segment_lengths[indices].sum())

    def _solve(
        self,
        rotations: np.ndarray,
        translations: np.ndarray,
        points: np.ndarray,
        active: np.ndarray,
        positions: np.ndarray,
        weights: np.ndarray,
        scale_source: str,
    ) -> tuple[np.ndarray, _BundleProblem]:
        segment_lengths = _median_segment_lengths(points, active)
        scale_prior = None
        if scale_source == SCALE_SOURCE_HEIGHT:
            scale_prior = _height_prior_sum(segment_lengths, self._height_m)
        baseline_m = float(np.linalg.norm(translations[1]))

        def build(current_weights: np.ndarray) -> _BundleProblem:
            return _BundleProblem(
                self._intrinsic_matrices, positions, current_weights, segment_lengths,
                scale_source, self._bar_length_m, scale_prior, baseline_m,
            )

        def run(problem: _BundleProblem, start: np.ndarray, loss: str) -> np.ndarray:
            return least_squares(
                problem.residuals, start, jac_sparsity=problem.jacobian_sparsity(), method="trf", loss=loss,
                f_scale=HUBER_DELTA_PX, x_scale="jac", ftol=SOLVER_TOLERANCE, xtol=SOLVER_TOLERANCE,
                gtol=SOLVER_TOLERANCE, max_nfev=MAX_SOLVER_EVALUATIONS,
            ).x

        # scipy's Huber zeroes the Jacobian rows of residuals beyond delta, so from a rough start
        # (most residuals beyond delta) it cannot move; soft-L1 keeps a gradient everywhere.
        problem = build(weights)
        params = run(problem, problem.pack(rotations, translations, points, segment_lengths), APPROACH_LOSS)

        errors_px = problem.reprojection_errors_px(params)
        gate_px = max(OUTLIER_MEDIAN_FACTOR * float(np.median(errors_px)), OUTLIER_FLOOR_PX)
        outliers = errors_px > gate_px
        frame_idx, view_idx, point_idx = (location[outliers] for location in problem.obs_location)
        inlier_weights = weights.copy()
        inlier_weights[frame_idx, view_idx, point_idx] = 0.0
        solved_rotations, solved_translations, solved_points, solved_lengths = problem.unpack(params)
        refined_problem = build(inlier_weights)
        refined_start = refined_problem.pack(
            solved_rotations, solved_translations, problem.frame_points(solved_points), solved_lengths
        )
        return run(refined_problem, refined_start, FINAL_LOSS), refined_problem

    def _standing_points(
        self, rotations: np.ndarray, translations: np.ndarray, positions: np.ndarray, valid: np.ndarray
    ) -> np.ndarray:
        # Joints (S, 21, 3) of the standing frames among ALL input frames (not only the solved
        # subset: still frames look alike, so the diversity subsample keeps few of them),
        # triangulated with the solved cameras.
        required = [CK.LEFT_HIP, CK.RIGHT_HIP, CK.LEFT_ANKLE, CK.RIGHT_ANKLE]
        anchored = (valid[:, :, required].sum(axis=1) >= MIN_VIEWS_PER_POINT).all(axis=1)
        if not anchored.any():
            raise ValueError("no frame has both hips and both ankles seen by two cameras")
        flexion_deg = _knee_flexion_2d_deg(positions, valid)
        flexion_deg = np.where(anchored & np.isfinite(flexion_deg), flexion_deg, np.inf)
        standing = flexion_deg <= STANDING_MAX_KNEE_FLEXION_2D_DEG
        if standing.sum() < MIN_STANDING_FRAMES:
            fallback_count = max(int(np.ceil(STANDING_FALLBACK_FRACTION * anchored.sum())), 1)
            logger.warning(
                "[PERSON CALIBRATION] Only %d standing frames; anchoring the world frame on the %d "
                "straightest-leg frames", int(standing.sum()), fallback_count,
            )
            standing = np.zeros_like(anchored)
            standing[np.argsort(flexion_deg)[:fallback_count]] = True
            standing &= anchored
        points, _ = self._triangulate_frames(rotations, translations, positions[standing], valid[standing])
        return points

    def _consensus_with_initial(
        self,
        rotations: np.ndarray,
        translations: np.ndarray,
        solved_points: np.ndarray,
        initial: CalibrationResult,
        with_scale: bool,
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        # Fit of the solved rig onto the initial one over camera landmarks (lever arm = each
        # camera's distance to the lifter) -> rotation, shift, scale, cameras that did not move
        initial_rotations = np.stack(
            [np.asarray(initial.cameras[c].rotation_matrix, dtype=np.float64) for c in self._cam_ids]
        )
        initial_translations = np.stack(
            [np.asarray(initial.cameras[c].translation_vector, dtype=np.float64).ravel() for c in self._cam_ids]
        )
        depths_m = np.einsum("vij,nj->vni", rotations, solved_points)[..., 2] + translations[:, None, 2]
        lever_arms_m = np.median(depths_m, axis=1)
        rotation, shift, scale, cameras_used = _consensus_rigid_fit(
            _camera_landmarks(rotations, translations, lever_arms_m),
            _camera_landmarks(initial_rotations, initial_translations, lever_arms_m),
            with_scale,
        )
        moved = [cam_id for cam_id, used in zip(self._cam_ids, cameras_used) if not used]
        if moved and len(self._cam_ids) >= MIN_CONSENSUS_CAMERAS:
            logger.info("[PERSON CALIBRATION] Camera %s moved relative to the initial calibration", moved)
        return rotation, shift, scale, cameras_used

    def _scale_change_ratio(
        self,
        rotations: np.ndarray,
        translations: np.ndarray,
        solved_points: np.ndarray,
        positions: np.ndarray,
        valid: np.ndarray,
        initial: CalibrationResult,
    ) -> float:
        # Leg segments triangulated by the cameras that did not move, solved / initial. The
        # moved camera is found with a similarity fit (the solve's scale is still a free gauge
        # here); epipolar residuals cannot see a camera sliding along its baseline, but the
        # solved rig can. NaN when the segments are seen in too few frames.
        cameras_used = self._consensus_with_initial(rotations, translations, solved_points, initial, True)[3]
        if cameras_used.sum() < MIN_VIEWS_PER_POINT:
            cameras_used = np.ones(len(self._cam_ids), dtype=bool)
        initial_projections = np.stack(
            [np.asarray(initial.cameras[c].projection_matrix, dtype=np.float64) for c in self._cam_ids]
        )
        solved_projections = _projection_matrices(self._intrinsic_matrices, rotations, translations)
        initial_lengths = _triangulated_segment_lengths(initial_projections, positions, valid, cameras_used)
        solved_lengths = _triangulated_segment_lengths(solved_projections, positions, valid, cameras_used)
        indices = [
            idx for idx, name in enumerate(BA_SEGMENT_NAMES)
            if name in SCALE_TARGET_SEGMENTS and np.isfinite(initial_lengths[idx]) and np.isfinite(solved_lengths[idx])
        ]
        if not indices:
            return float("nan")
        return float(solved_lengths[indices].sum() / initial_lengths[indices].sum())

    def _initial_world_frame(
        self, rotations: np.ndarray, translations: np.ndarray, solved_points: np.ndarray, initial: CalibrationResult
    ) -> tuple[np.ndarray, np.ndarray]:
        # Rotation A and origin o, in the solve's frame, of the INITIAL calibration's world:
        # world = A (p - o), the same form world_frame_from_standing returns. Rigid: no scale.
        rotation, shift, _, _ = self._consensus_with_initial(rotations, translations, solved_points, initial, False)
        # world = rotation @ p + shift = rotation @ (p - o)  with  o = -rotation^T shift
        return rotation, -rotation.T @ shift

    def _rms_reprojection_px(
        self, rotations: np.ndarray, translations: np.ndarray, positions: np.ndarray, valid: np.ndarray
    ) -> float:
        projections = _projection_matrices(self._intrinsic_matrices, rotations, translations)
        residuals, used = _multi_view_residuals_px(
            projections, positions[:, :, :NUM_KEYPOINTS], valid[:, :, :NUM_KEYPOINTS]
        )
        return _robust_rms_px(residuals[used])

    def _build_calibration(
        self,
        rotations: np.ndarray,
        translations: np.ndarray,
        positions: np.ndarray,
        valid: np.ndarray,
        bone_lengths_m: dict[str, float],
        scale_source: str,
        initial: CalibrationResult | None,
        world_anchor: str,
    ) -> CalibrationResult:
        projections = _projection_matrices(self._intrinsic_matrices, rotations, translations)
        residuals, used = _multi_view_residuals_px(
            projections, positions[:, :, :NUM_KEYPOINTS], valid[:, :, :NUM_KEYPOINTS]
        )
        result = CalibrationResult(
            athlete_height_m=self._athlete_height_m(bone_lengths_m, scale_source, initial),
            timestamp=datetime.now().isoformat(),
            world_anchor=world_anchor,
        )
        for view_idx, cam_id in enumerate(self._cam_ids):
            result.cameras[cam_id] = CameraCalibration(
                camera_id=cam_id,
                projection_matrix=projections[view_idx],
                intrinsic_matrix=self._intrinsic_matrices[view_idx],
                rotation_matrix=rotations[view_idx],
                translation_vector=translations[view_idx].reshape(3, 1),
                reprojection_error=_robust_rms_px(residuals[view_idx][used[view_idx]]),
                resolution=self._resolution,
                distortion_coeffs=self._distortion_coeffs[view_idx],
            )
        return result

    def _athlete_height_m(
        self, bone_lengths_m: dict[str, float], scale_source: str, initial: CalibrationResult | None
    ) -> float:
        if self._height_m is not None:
            return self._height_m
        if initial is not None and initial.athlete_height_m > 0.0:
            return initial.athlete_height_m
        ratio_segments = [name for name in bone_lengths_m if name in SEGMENT_HEIGHT_RATIOS]
        if scale_source == SCALE_SOURCE_INITIAL or not ratio_segments:
            return 0.0
        return sum(bone_lengths_m[name] for name in ratio_segments) / sum(
            SEGMENT_HEIGHT_RATIOS[name] for name in ratio_segments
        )
