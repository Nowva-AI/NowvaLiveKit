"""3D barbell tracking for the deadlift from the rack's synced cameras (PLAN.md §3.6,
§6): undistort every 2D bar candidate, associate them across views by reprojection
error and consistency with the hands and feet (this rejects a racked bar), triangulate
both plate hubs, and smooth them with a constant-velocity Kalman filter. Positions are
metres in the Y-down world frame of the provider's calibration.
"""

from __future__ import annotations

import math
from collections import deque
from functools import lru_cache
from typing import TYPE_CHECKING, Protocol

import numpy as np
from pydantic import BaseModel

from biomechanics.config import BarbellTrackingConfig
from biomechanics.triangulation.calibration import (
    CalibrationResult,
    CameraCalibration,
    undistort_keypoints,
)
from biomechanics.utils.geometry import WORLD_UP
from biomechanics.utils.types import CocoKeypoints as CK
from biomechanics.utils.types import Skeleton3D

from .bar_detector_multi import DEFAULT_MAX_CANDIDATES_PER_VIEW, BarCandidate2D, MultiViewBarDetector
from .types import NAN, BarState3D

if TYPE_CHECKING:
    from biomechanics.pose.multi_camera import MultiCameraPoseProvider

# Hub-to-hub distance: the J1 keypoint is the centre of the outer face of the outer plate,
# so the distance is 1.31 m between the inner collars of a 2.2 m Olympic bar plus the plate
# stack on each side; one 20 kg bumper (~6.5 cm) per side gives 1.44 m. The 45 cm plate
# diameter sets the hub height (22.5 cm), not this distance. The tracker replaces it with the
# measured distance as soon as both hubs are triangulated from >= 2 views (a second plate per
# side adds ~13 cm, which would otherwise tilt a single-view hub by several cm).
DEFAULT_HUB_DISTANCE_M = 1.44
# Both hubs triangulated: outside this range the two "ends" belong to different objects.
MIN_HUB_DISTANCE_M = 1.0
MAX_HUB_DISTANCE_M = 2.4
# A deadlift bar tilts a few degrees at most; world Y is within ~4 deg of gravity.
MAX_BAR_TILT_DEG = 25.0
# A single-view hub ray that misses the hub-distance sphere around the other hub by more
# than this does not belong to the same bar.
MAX_SINGLE_VIEW_RAY_MISS_M = 0.10
MIN_TRIANGULATION_VIEWS = 2
# Learned hub distance: running median over the last frames with both hubs from >= 2 views.
HUB_DISTANCE_WINDOW = 90
MIN_HUB_DISTANCE_SAMPLES = 15
# A measurement this far from the predicted hub is an outlier (a wrong association): the
# state is predicted instead; a lasting change re-initialises once the track has expired.
MAX_INNOVATION_M = 0.3
INITIAL_VELOCITY_SIGMA_MPS = 1.0
RELATIVE_DETERMINANT_EPSILON = 1e-12
MIN_DEPTH_M = 1e-6
NUM_ENDS = 2


class BarTracker3DConfig(BaseModel):
    """Initial values, to be re-measured on real captures (J4)."""
    hub_distance_m: float = DEFAULT_HUB_DISTANCE_M
    # Per observation, undistorted pixels (1280x720); the squat triangulator uses 15 px.
    max_reprojection_error_px: float = 12.0
    # A hub keypoint below this confidence is treated as not seen in that view.
    min_end_confidence: float = 0.3
    # Plausibility gates (PLAN.md §3.6): hands on the bar; bar in front of the feet, below the hips.
    max_centre_to_wrists_m: float = 0.30
    max_centre_to_ankles_horizontal_m: float = 0.70
    max_centre_above_ankles_m: float = 1.0
    # Kalman: 3-view hub triangulation noise; white-acceleration spectral density (m^2/s^3)
    # giving <= 0.3 cm lag on a 0.6 m pull in 0.8 s at 30 fps.
    measurement_sigma_m: float = 0.005
    acceleration_noise_psd: float = 2.0
    # Predicted states bridge a 15 Hz bar detector (PLAN.md §9) plus one missed detection.
    max_prediction_s: float = 0.15


DEFAULT_CONFIG = BarTracker3DConfig()


class BarAssociation(BaseModel):
    """The bar found across views on one frame, before smoothing. left/right_views count
    the cameras that saw each hub; 1 means that hub was placed from the hub distance."""
    left_end_m: tuple[float, float, float]
    right_end_m: tuple[float, float, float]
    views: int
    left_views: int
    right_views: int
    residual_px: float
    candidate_indices: dict[str, int]


class BarCandidateDetector(Protocol):
    @property
    def available(self) -> bool: ...

    def detect(self, views: dict[str, np.ndarray]) -> dict[str, list[BarCandidate2D]]: ...


def _projection_matrix(camera: CameraCalibration) -> np.ndarray:
    extrinsics = np.hstack([camera.rotation_matrix, np.reshape(camera.translation_vector, (3, 1))])
    return camera.intrinsic_matrix @ extrinsics


def _camera_ray(camera: CameraCalibration, pixel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # World-frame centre and unit direction of the ray through an undistorted pixel.
    rotation = camera.rotation_matrix
    centre = -rotation.T @ np.reshape(camera.translation_vector, 3)
    direction = rotation.T @ np.linalg.solve(camera.intrinsic_matrix, np.array([pixel[0], pixel[1], 1.0]))
    return centre, direction / np.linalg.norm(direction)


def _view_options(
    candidates: list[BarCandidate2D],
    camera: CameraCalibration,
    projection: np.ndarray,
    min_end_confidence: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Option 0 = view unused; options 2k+1 / 2k+2 = kept candidate k with its ends read as
    # (a, b) / (b, a), which covers a left/right swap between views.
    # -> undistorted pixels (O, 2, 2), valid (O, 2), DLT normal matrices (O, 2, 4, 4),
    #    index of the candidate in the input list (O,), -1 for option 0.
    ranked = sorted(range(len(candidates)), key=lambda i: -candidates[i].score)
    raw_px = np.array([[candidates[i].end_a_px, candidates[i].end_b_px] for i in ranked], dtype=np.float64)
    confidences = np.array([candidates[i].end_confidences for i in ranked], dtype=np.float64)
    end_valid = (confidences >= min_end_confidence) & np.isfinite(raw_px).all(axis=-1)
    kept = end_valid.any(axis=1)
    kept &= np.cumsum(kept) <= DEFAULT_MAX_CANDIDATES_PER_VIEW
    raw_px, end_valid = raw_px[kept], end_valid[kept]
    candidate_index = np.array(ranked)[kept]
    num_kept = len(candidate_index)

    undistorted = undistort_keypoints(
        np.where(np.isfinite(raw_px), raw_px, 0.0).reshape(-1, 2), camera
    ).reshape(num_kept, NUM_ENDS, 2)
    num_options = 2 * num_kept + 1
    pixels = np.zeros((num_options, NUM_ENDS, 2))
    pixels[1::2] = undistorted
    pixels[2::2] = undistorted[:, ::-1]
    valid = np.zeros((num_options, NUM_ENDS), dtype=bool)
    valid[1::2] = end_valid
    valid[2::2] = end_valid[:, ::-1]
    option_candidate = np.concatenate([[-1], np.repeat(candidate_index, 2)])

    rows_u = pixels[..., 0:1] * projection[2] - projection[0]
    rows_v = pixels[..., 1:2] * projection[2] - projection[1]
    normals = rows_u[..., :, None] * rows_u[..., None, :] + rows_v[..., :, None] * rows_v[..., None, :]
    normals *= valid[..., None, None]
    return pixels, valid, normals, option_candidate


@lru_cache(maxsize=64)
def _hypothesis_table(option_counts: tuple[int, ...]) -> np.ndarray:
    # Every per-view option choice with >= 2 views used: (H, V) option indices. Flipping
    # the end order in every view gives the same bar, so the first used view keeps (a, b).
    table = np.indices(option_counts).reshape(len(option_counts), -1).T
    used = table > 0
    first_option = table[np.arange(len(table)), used.argmax(axis=1)]
    table = table[(used.sum(axis=1) >= MIN_TRIANGULATION_VIEWS) & (first_option % 2 == 1)]
    table.setflags(write=False)
    return table


def _solve_symmetric_3x3(matrices: np.ndarray, rhs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Batched solve of positive semi-definite 3x3 systems by the adjugate:
    # (..., 3, 3), (..., 3) -> solution (..., 3), solvable (...). Per-matrix LAPACK calls
    # (np.linalg.solve / eigh) cost several times more on these small batches.
    a, b, c = matrices[..., 0, 0], matrices[..., 0, 1], matrices[..., 0, 2]
    d, e, f = matrices[..., 1, 1], matrices[..., 1, 2], matrices[..., 2, 2]
    adjugate_00, adjugate_01, adjugate_02 = d * f - e * e, c * e - b * f, b * e - c * d
    adjugate_11, adjugate_12, adjugate_22 = a * f - c * c, b * c - a * e, a * d - b * b
    determinant = a * adjugate_00 + b * adjugate_01 + c * adjugate_02
    # Hadamard: det <= a * d * f for a PSD matrix; far below it means rank-deficient (one ray).
    solvable = determinant > RELATIVE_DETERMINANT_EPSILON * a * d * f
    inverse_determinant = 1.0 / np.where(solvable, determinant, 1.0)
    x, y, z = rhs[..., 0], rhs[..., 1], rhs[..., 2]
    solution = np.stack([
        adjugate_00 * x + adjugate_01 * y + adjugate_02 * z,
        adjugate_01 * x + adjugate_11 * y + adjugate_12 * z,
        adjugate_02 * x + adjugate_12 * y + adjugate_22 * z,
    ], axis=-1) * inverse_determinant[..., None]
    return solution, solvable & np.isfinite(solution).all(axis=-1)


def _hub_on_ray(
    anchor: np.ndarray, ray_origin: np.ndarray, ray_direction: np.ndarray, hub_distance_m: float, up: np.ndarray,
) -> np.ndarray | None:
    # Point on the ray at hub_distance_m from the anchor hub. Of the two roots, the bar
    # through the other one is steeply tilted, so the most horizontal bar wins.
    offset = ray_origin - anchor
    half_b = float(ray_direction @ offset)
    discriminant = half_b * half_b - (float(offset @ offset) - hub_distance_m**2)
    if discriminant < 0.0:
        miss_m = math.sqrt(max(float(offset @ offset) - half_b * half_b, 0.0)) - hub_distance_m
        if miss_m > MAX_SINGLE_VIEW_RAY_MISS_M:
            return None
        roots = [-half_b]
    else:
        roots = [-half_b - math.sqrt(discriminant), -half_b + math.sqrt(discriminant)]
    points = [ray_origin + depth * ray_direction for depth in roots if depth > MIN_DEPTH_M]
    if not points:
        return None
    return min(points, key=lambda point: abs(float((point - anchor) @ up)))


def _keypoint_pair(skeleton: Skeleton3D, left_index: int, right_index: int) -> np.ndarray:
    # (2, 3); a keypoint the triangulator rejected (confidence 0, stored at the origin) is NaN.
    points = [skeleton.keypoints[index] for index in (left_index, right_index)]
    return np.array([[point.x, point.y, point.z] if point.confidence > 0.0 else [NAN] * 3 for point in points])


def _finite_mean(points: np.ndarray | None) -> np.ndarray | None:
    if points is None:
        return None
    mean = np.mean(np.reshape(np.asarray(points, dtype=np.float64), (-1, 3)), axis=0)
    return mean if np.isfinite(mean).all() else None


def _plausible(
    ends: np.ndarray,
    wrist_mid: np.ndarray | None,
    ankle_mid: np.ndarray | None,
    config: BarTracker3DConfig,
    up: np.ndarray,
) -> bool:
    span = ends[0] - ends[1]
    tilt_deg = math.degrees(math.asin(min(abs(float(span @ up)) / float(np.linalg.norm(span)), 1.0)))
    if tilt_deg > MAX_BAR_TILT_DEG:
        return False
    centre = ends.mean(axis=0)
    if wrist_mid is not None and np.linalg.norm(centre - wrist_mid) > config.max_centre_to_wrists_m:
        return False
    if ankle_mid is not None:
        offset = centre - ankle_mid
        height_m = float(offset @ up)
        horizontal_m = float(np.linalg.norm(offset - height_m * up))
        if horizontal_m > config.max_centre_to_ankles_horizontal_m or height_m > config.max_centre_above_ankles_m:
            return False
    return True


def wrists_and_ankles_world(skeleton: Skeleton3D | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    """(2, 3) left/right wrists and ankles of a world-frame skeleton, for the bar gates.
    Rejected keypoints are NaN, which skips their gate instead of pulling the mid-point
    to the world origin."""
    if skeleton is None:
        return None, None
    return (
        _keypoint_pair(skeleton, CK.LEFT_WRIST, CK.RIGHT_WRIST),
        _keypoint_pair(skeleton, CK.LEFT_ANKLE, CK.RIGHT_ANKLE),
    )


def associate_and_triangulate(
    candidates_per_view: dict[str, list[BarCandidate2D]],
    cameras: dict[str, CameraCalibration],
    wrists_world: np.ndarray | None = None,
    ankles_world: np.ndarray | None = None,
    config: BarTracker3DConfig = DEFAULT_CONFIG,
    hub_distance_m: float | None = None,
    up: np.ndarray | None = None,
) -> BarAssociation | None:
    """The best cross-view bar on one frame, or None when no association is consistent.

    Every pairing of one candidate (or none) per view, with both end orders, is
    triangulated by DLT. A pairing passes when each hub is seen by >= 1 view, one by
    >= 2, and every observation of a >= 2-view hub reprojects within
    max_reprojection_error_px. Passing pairings are ranked by observation count, then
    RMS residual; the first that is plausible wins. A hub seen by a single view is put
    on that view's ray at hub_distance_m (default config.hub_distance_m) from the other.

    wrists_world / ankles_world are (2, 3) world positions (left, right). Pass wrists
    only while the hands hold the bar and ankles while the feet are planted: the bar
    centre must then be within max_centre_to_wrists_m of the mid-wrist, and within
    max_centre_to_ankles_horizontal_m horizontally and max_centre_above_ankles_m above
    the mid-ankle. Non-finite input skips that gate. up is the world's up (measured
    gravity when known, WORLD_UP otherwise). The left end is the one at larger X: a
    consistent label for the track, which is the subject's left only on a world-
    anchored calibration (the analyser re-labels the ends from the lifter's frame).
    """
    hub_distance_m = config.hub_distance_m if hub_distance_m is None else hub_distance_m
    up = np.asarray(WORLD_UP, dtype=np.float64) if up is None else up
    camera_ids = sorted(cid for cid, candidates in candidates_per_view.items() if candidates and cid in cameras)
    projections = [_projection_matrix(cameras[cid]) for cid in camera_ids]
    view_options = [
        _view_options(candidates_per_view[cid], cameras[cid], projection, config.min_end_confidence)
        for cid, projection in zip(camera_ids, projections)
    ]
    used_views = [i for i, options in enumerate(view_options) if len(options[3]) > 1]
    if len(used_views) < MIN_TRIANGULATION_VIEWS:
        return None
    camera_ids = [camera_ids[i] for i in used_views]
    projections = np.stack([projections[i] for i in used_views])
    option_pixels, option_valid, option_normals, option_candidates = zip(*(view_options[i] for i in used_views))
    table = _hypothesis_table(tuple(len(candidates) for candidates in option_candidates))
    num_views = len(camera_ids)

    # pixels (H, V, 2, 2), valid (H, V, 2), normal sums (H, 2, 4, 4)
    pixels = np.stack([option_pixels[v][table[:, v]] for v in range(num_views)], axis=1)
    valid = np.stack([option_valid[v][table[:, v]] for v in range(num_views)], axis=1)
    normal_sums = sum(option_normals[v][table[:, v]] for v in range(num_views))
    end_views = valid.sum(axis=1)

    # Inhomogeneous DLT: minimise sum |A [X, 1]|^2 over X.
    points, solvable = _solve_symmetric_3x3(normal_sums[..., :3, :3], -normal_sums[..., :3, 3])
    solved = (end_views >= MIN_TRIANGULATION_VIEWS) & solvable
    points = np.where(solved[..., None], points, 0.0)

    points_h = np.concatenate([points, np.ones(points.shape[:-1] + (1,))], axis=-1)
    projected = np.einsum("vij,hej->hvei", projections, points_h)
    depth = projected[..., 2]
    in_front = depth > MIN_DEPTH_M
    residuals = np.linalg.norm(projected[..., :2] / np.where(in_front, depth, 1.0)[..., None] - pixels, axis=-1)
    counted = valid & solved[:, None, :]
    residuals = np.where(counted, np.where(in_front, residuals, np.inf), 0.0)

    num_counted = counted.sum(axis=(1, 2))
    rms_px = np.sqrt(np.sum(residuals**2, axis=(1, 2)) / np.maximum(num_counted, 1))
    feasible = (
        (solved | (end_views == 1)).all(axis=1)
        & solved.any(axis=1)
        & (residuals.max(axis=(1, 2)) <= config.max_reprojection_error_px)
    )
    order = np.lexsort((rms_px, -valid.sum(axis=(1, 2))))
    order = order[feasible[order]]

    wrist_mid = _finite_mean(wrists_world)
    ankle_mid = _finite_mean(ankles_world)
    for h in order:
        ends = points[h].copy()
        if solved[h].all():
            if not MIN_HUB_DISTANCE_M <= float(np.linalg.norm(ends[0] - ends[1])) <= MAX_HUB_DISTANCE_M:
                continue
        else:
            single = int(np.argmin(solved[h]))
            view = int(np.argmax(valid[h, :, single]))
            ray_origin, ray_direction = _camera_ray(cameras[camera_ids[view]], pixels[h, view, single])
            hub = _hub_on_ray(ends[1 - single], ray_origin, ray_direction, hub_distance_m, up)
            if hub is None:
                continue
            ends[single] = hub
        if not _plausible(ends, wrist_mid, ankle_mid, config, up):
            continue
        left, right = (0, 1) if ends[0, 0] >= ends[1, 0] else (1, 0)
        return BarAssociation(
            left_end_m=tuple(ends[left].tolist()),
            right_end_m=tuple(ends[right].tolist()),
            views=int((table[h] > 0).sum()),
            left_views=int(end_views[h, left]),
            right_views=int(end_views[h, right]),
            residual_px=float(rms_px[h]),
            candidate_indices={
                camera_ids[v]: int(option_candidates[v][table[h, v]]) for v in range(num_views) if table[h, v] > 0
            },
        )
    return None


class BarTracker3D:
    """Both plate hubs in the world frame, Kalman-smoothed per capture frame.

    The filter is constant-velocity on the six hub coordinates. They share one model,
    one noise and the same measurement times, so one 2x2 covariance serves all six.
    A frame without an association yields a predicted state (predicted=True) for up
    to max_prediction_s after the last measurement, then None. Without a calibration
    every update returns None.
    """

    def __init__(
        self,
        calibration: CalibrationResult | None,
        detector: BarCandidateDetector | None = None,
        config: BarTracker3DConfig | None = None,
    ) -> None:
        self._detector = detector
        self._config = config or DEFAULT_CONFIG
        self._measurement_variance = self._config.measurement_sigma_m**2
        self._provider: MultiCameraPoseProvider | None = None
        self._calibration = calibration
        self._cameras = dict(calibration.cameras) if calibration is not None else {}
        self._state = np.zeros((2, 3 * NUM_ENDS))  # rows: positions, velocities; cols: left xyz, right xyz
        self._covariance = np.zeros((2, 2))
        self._initialized = False
        self._filter_time = NAN
        self._last_measurement_time = NAN
        self._hub_distances: deque[float] = deque(maxlen=HUB_DISTANCE_WINDOW)
        self._up = np.asarray(WORLD_UP, dtype=np.float64)

    @classmethod
    def from_provider(
        cls,
        provider: MultiCameraPoseProvider,
        barbell_config: BarbellTrackingConfig | None = None,
        config: BarTracker3DConfig | None = None,
        detector: BarCandidateDetector | None = None,
    ) -> BarTracker3D:
        """A tracker that follows provider.calibration: none yet is fine (updates return None
        until one is installed) and a newly installed one resets the track. The detector
        defaults to MultiViewBarDetector on barbell_config's weights, loaded on first use."""
        tracker = cls(
            provider.calibration,
            detector or MultiViewBarDetector.from_config(barbell_config or BarbellTrackingConfig()),
            config,
        )
        tracker._provider = provider
        return tracker

    @property
    def hub_distance_m(self) -> float:
        """Median measured hub distance once enough frames saw both hubs in >= 2 views, else the configured one."""
        if len(self._hub_distances) < MIN_HUB_DISTANCE_SAMPLES:
            return self._config.hub_distance_m
        return float(np.median(self._hub_distances))

    def set_up(self, up_world: np.ndarray) -> None:
        """The world's up for the plausibility gates: measured gravity when known."""
        up = np.asarray(up_world, dtype=np.float64)
        self._up = up / float(np.linalg.norm(up))

    def reset(self) -> None:
        self._state[:] = 0.0
        self._covariance[:] = 0.0
        self._initialized = False
        self._filter_time = NAN
        self._last_measurement_time = NAN
        self._hub_distances.clear()

    def update(
        self,
        views: dict[str, np.ndarray] | None,
        timestamp: float,
        wrists_world: np.ndarray | None = None,
        ankles_world: np.ndarray | None = None,
    ) -> BarState3D | None:
        """Detect on every synced BGR view (camera id -> frame) in one batch, then track. None
        when there are no frames, no calibration, or the detector is unavailable (no weights)."""
        if self._detector is None:
            raise ValueError("BarTracker3D was built without a detector; use update_from_candidates")
        self._follow_provider_calibration()
        if not views or not self._cameras or not self._detector.available:
            return None
        return self.update_from_candidates(self._detector.detect(views), timestamp, wrists_world, ankles_world)

    def update_from_candidates(
        self,
        candidates_per_view: dict[str, list[BarCandidate2D]],
        timestamp: float,
        wrists_world: np.ndarray | None = None,
        ankles_world: np.ndarray | None = None,
    ) -> BarState3D | None:
        self._follow_provider_calibration()
        if not self._cameras:
            return None
        association = associate_and_triangulate(
            candidates_per_view, self._cameras, wrists_world, ankles_world, self._config, self.hub_distance_m,
            self._up,
        )
        if association is None or not self._accept(association, timestamp):
            return self._predicted_state(timestamp)
        if association.left_views >= MIN_TRIANGULATION_VIEWS and association.right_views >= MIN_TRIANGULATION_VIEWS:
            self._hub_distances.append(
                float(np.linalg.norm(np.subtract(association.left_end_m, association.right_end_m)))
            )
        return self._state_at(timestamp, False, association.views, association.residual_px)

    def _follow_provider_calibration(self) -> None:
        if self._provider is None or self._provider.calibration is self._calibration:
            return
        self._calibration = self._provider.calibration
        self._cameras = dict(self._calibration.cameras) if self._calibration is not None else {}
        self.reset()

    def _track_alive(self, timestamp: float) -> bool:
        return self._initialized and timestamp - self._last_measurement_time <= self._config.max_prediction_s

    def _accept(self, association: BarAssociation, timestamp: float) -> bool:
        measurement = np.concatenate([association.left_end_m, association.right_end_m])
        if not self._track_alive(timestamp):
            self._state[0] = measurement
            self._state[1] = 0.0
            self._covariance[:] = np.diag([self._measurement_variance, INITIAL_VELOCITY_SIGMA_MPS**2])
            self._initialized = True
        else:
            self._advance(timestamp)
            innovation = measurement - self._state[0]
            if np.linalg.norm(innovation.reshape(NUM_ENDS, 3), axis=1).max() > MAX_INNOVATION_M:
                return False
            gain = self._covariance[:, 0] / (self._covariance[0, 0] + self._measurement_variance)
            self._state += np.outer(gain, innovation)
            self._covariance -= np.outer(gain, self._covariance[0])
        self._filter_time = timestamp
        self._last_measurement_time = timestamp
        return True

    def _advance(self, timestamp: float) -> None:
        dt = timestamp - self._filter_time
        if not dt > 0.0:
            return
        transition = np.array([[1.0, dt], [0.0, 1.0]])
        # Continuous white-noise acceleration: predicting in steps equals predicting at once.
        process_noise = self._config.acceleration_noise_psd * np.array(
            [[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]]
        )
        self._state = transition @ self._state
        self._covariance = transition @ self._covariance @ transition.T + process_noise
        self._filter_time = timestamp

    def _predicted_state(self, timestamp: float) -> BarState3D | None:
        if not self._track_alive(timestamp):
            self._initialized = False
            return None
        self._advance(timestamp)
        return self._state_at(timestamp, True, 0, NAN)

    def _state_at(self, timestamp: float, predicted: bool, views: int, residual_px: float) -> BarState3D:
        velocity = (self._state[1, :3] + self._state[1, 3:]) / 2.0
        return BarState3D(
            timestamp=timestamp,
            left_end_m=tuple(self._state[0, :3].tolist()),
            right_end_m=tuple(self._state[0, 3:].tolist()),
            velocity_mps=tuple(velocity.tolist()),
            predicted=predicted,
            views=views,
            residual_px=residual_px,
        )
