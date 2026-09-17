"""
Robust multi-view DLT triangulation into world coordinates (Y-down, meters).

Per keypoint: all-view DLT with a fallback to the best consistent camera pair,
a per-view left/right swap test, and a metric confidence derived from the
estimated 3D position uncertainty. Output is NOT hip-centred.
"""

from __future__ import annotations

import itertools
import logging

import numpy as np

from biomechanics.triangulation.calibration import CalibrationResult
from biomechanics.utils.types import CocoKeypoints as CK
from biomechanics.utils.types import MultiViewPose, Point3D, Skeleton3D

logger = logging.getLogger(__name__)

NUM_KEYPOINTS = 21

DEFAULT_MIN_VIEWS = 2
DEFAULT_MAX_REPROJECTION_ERROR_PX = 15.0
DEFAULT_MIN_CONFIDENCE = 0.3

# conf = 1 / (1 + (u / UNCERTAINTY_SCALE_M)^2), u = 3D position std in meters (C3).
UNCERTAINTY_SCALE_M = 0.02
MIN_PIXEL_SIGMA_MULTI_VIEW_PX = 3.0
MIN_PIXEL_SIGMA_TWO_VIEW_PX = 6.0
# Per-keypoint residual sigma is pooled to the frame median unless it exceeds
# this multiple of it (a keypoint whose own residual is a clear outlier).
SIGMA_OUTLIER_FACTOR = 3.0
# Two-view residuals only see the epipolar error component, so their
# uncertainty is under-estimated (audit S0-5: error/u p95 5.2x). Cap their trust.
TWO_VIEW_CONFIDENCE_CAP = 0.3

# A left/right swap hypothesis is applied only if its cost is below this
# fraction of the no-swap cost.
SWAP_MARGIN_RATIO = 0.5
# Per-keypoint RMS residual is truncated at this multiple of the reprojection
# threshold when scoring swap hypotheses, so one gross outlier cannot dominate.
SWAP_RESIDUAL_CAP_FACTOR = 2.0

# Passing camera pairs within this RMS residual of the best pair are ties,
# broken by distance to the previous accepted 3D point (if younger than the max
# age). Without one, ties whose 3D points differ by more than the ambiguity
# distance reject the keypoint.
PAIR_TIE_MARGIN_PX = 4.0
PAIR_AMBIGUITY_DISTANCE_M = 0.05
PREVIOUS_POINT_MAX_AGE_S = 0.5

HOMOGENEOUS_W_EPSILON = 1e-9
MIN_PROJECTIVE_DEPTH = 1e-6
JTJ_RELATIVE_EIGENVALUE_EPSILON = 1e-10
MIN_DEGREES_OF_FREEDOM = 1

LEG_LEFT_RIGHT_PAIRS: tuple[tuple[int, int], ...] = (
    (CK.LEFT_HIP, CK.RIGHT_HIP),
    (CK.LEFT_KNEE, CK.RIGHT_KNEE),
    (CK.LEFT_ANKLE, CK.RIGHT_ANKLE),
    (CK.LEFT_FOOT_INDEX, CK.RIGHT_FOOT_INDEX),
    (CK.LEFT_HEEL, CK.RIGHT_HEEL),
)
LEFT_RIGHT_PAIRS: tuple[tuple[int, int], ...] = (
    (CK.LEFT_EYE, CK.RIGHT_EYE),
    (CK.LEFT_EAR, CK.RIGHT_EAR),
    (CK.LEFT_SHOULDER, CK.RIGHT_SHOULDER),
    (CK.LEFT_ELBOW, CK.RIGHT_ELBOW),
    (CK.LEFT_WRIST, CK.RIGHT_WRIST),
) + LEG_LEFT_RIGHT_PAIRS


def _left_right_permutation(pairs: tuple[tuple[int, int], ...]) -> np.ndarray:
    permutation = np.arange(NUM_KEYPOINTS)
    for left_idx, right_idx in pairs:
        permutation[left_idx], permutation[right_idx] = right_idx, left_idx
    return permutation


def _dlt_normal_matrices(projections: np.ndarray, xy: np.ndarray) -> np.ndarray:
    # projections (V,3,4), xy (V,K,2) -> per-view DLT normal matrices A_v^T A_v, (V,K,4,4)
    rows_u = xy[..., 0:1] * projections[:, None, 2, :] - projections[:, None, 0, :]
    rows_v = xy[..., 1:2] * projections[:, None, 2, :] - projections[:, None, 1, :]
    return rows_u[..., :, None] * rows_u[..., None, :] + rows_v[..., :, None] * rows_v[..., None, :]


def _solve_and_measure(
    projections: np.ndarray,
    normal_sums: np.ndarray,
    xy: np.ndarray,
    used: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # normal_sums (H,K,4,4); xy (V,K,2) or (H,V,K,2); used (H,V,K)
    # -> points (H,K,3) (0 where unsolvable), residuals px (H,V,K): 0 for unused views, inf if invalid
    _, eigenvectors = np.linalg.eigh(normal_sums)
    homogeneous = eigenvectors[..., :, 0]
    w = homogeneous[..., 3]
    solved = np.abs(w) > HOMOGENEOUS_W_EPSILON
    points = homogeneous[..., :3] / np.where(solved, w, 1.0)[..., None]
    solved &= np.isfinite(points).all(axis=-1)
    points = np.where(solved[..., None], points, 0.0)

    points_h = np.concatenate([points, np.ones(points.shape[:-1] + (1,))], axis=-1)
    projected = np.einsum("vij,hkj->hvki", projections, points_h)
    depth = projected[..., 2]
    in_front = depth > MIN_PROJECTIVE_DEPTH
    uv = projected[..., :2] / np.where(in_front, depth, 1.0)[..., None]
    residuals = np.linalg.norm(uv - xy, axis=-1)
    measurable = in_front & solved[:, None, :] & np.isfinite(residuals)
    residuals = np.where(measurable, residuals, np.inf)
    return points, np.where(used, residuals, 0.0)


def _truncated_rms_px(
    residuals: np.ndarray, used: np.ndarray, cap_px: float
) -> tuple[np.ndarray, np.ndarray]:
    # residuals, used (H,V,K) -> truncated per-keypoint RMS (H,K), keypoint has >= 2 views (H,K)
    num_used = used.sum(axis=1)
    rms = np.sqrt(np.sum(residuals**2, axis=1) / np.maximum(num_used, 1))
    return np.minimum(rms, cap_px), num_used >= 2


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.sum(np.where(mask, values, 0.0), axis=-1) / np.maximum(mask.sum(axis=-1), 1)


def recentre_at_hips(skeleton: Skeleton3D) -> Skeleton3D | None:
    """Subtract the hip midpoint from every keypoint with confidence > 0.

    Keypoints with confidence 0 are set to (0, 0, 0). Returns None when
    either hip has confidence 0.
    """
    keypoints = skeleton.keypoints
    if keypoints[CK.LEFT_HIP].confidence <= 0.0 or keypoints[CK.RIGHT_HIP].confidence <= 0.0:
        return None
    positions = np.array([[kp.x, kp.y, kp.z] for kp in keypoints], dtype=np.float64)
    confidences = np.array([kp.confidence for kp in keypoints], dtype=np.float64)
    hip_midpoint = (positions[CK.LEFT_HIP] + positions[CK.RIGHT_HIP]) / 2.0
    centred = np.where(confidences[:, None] > 0.0, positions - hip_midpoint, 0.0)
    return Skeleton3D(
        keypoints=[
            Point3D(
                x=float(centred[i, 0]),
                y=float(centred[i, 1]),
                z=float(centred[i, 2]),
                confidence=float(confidences[i]),
            )
            for i in range(len(keypoints))
        ],
        timestamp=skeleton.timestamp,
        frame_index=skeleton.frame_index,
    )


class DLTTriangulator:
    """Triangulates multi-view 2D detections into world-coordinate 3D keypoints.

    Rejected keypoints get confidence 0.0 and position (0, 0, 0). Holds a small
    per-keypoint state (last accepted point) used only to break ties between
    camera pairs; clear it with reset().
    """

    def __init__(
        self,
        calibration: CalibrationResult,
        min_views: int = DEFAULT_MIN_VIEWS,
        max_reprojection_error: float = DEFAULT_MAX_REPROJECTION_ERROR_PX,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        swap_margin_ratio: float = SWAP_MARGIN_RATIO,
        two_view_confidence_cap: float = TWO_VIEW_CONFIDENCE_CAP,
    ) -> None:
        if min_views < 2:
            raise ValueError(f"min_views must be >= 2 for triangulation, got {min_views}")
        if not calibration.cameras:
            raise ValueError("calibration has no cameras")
        if min_views > len(calibration.cameras):
            raise ValueError(
                f"min_views={min_views} exceeds the {len(calibration.cameras)} calibrated cameras"
            )
        self._min_views = min_views
        self._max_reprojection_error_px = max_reprojection_error
        self._min_confidence = min_confidence
        self._swap_margin_ratio = swap_margin_ratio
        self._two_view_confidence_cap = two_view_confidence_cap

        self._cam_ids = sorted(calibration.cameras.keys())
        self._projections = np.stack(
            [
                np.asarray(calibration.cameras[cam_id].projection_matrix, dtype=np.float64)
                for cam_id in self._cam_ids
            ]
        )
        num_views = len(self._cam_ids)

        pairs = list(itertools.combinations(range(num_views), 2))
        self._pair_masks = np.zeros((len(pairs), num_views), dtype=bool)
        for pair_idx, (view_a, view_b) in enumerate(pairs):
            self._pair_masks[pair_idx, [view_a, view_b]] = True

        self._full_swap_permutation = _left_right_permutation(LEFT_RIGHT_PAIRS)
        self._legs_swap_permutation = _left_right_permutation(LEG_LEFT_RIGHT_PAIRS)
        keypoint_indices = np.arange(NUM_KEYPOINTS)
        self._bilateral_mask = self._full_swap_permutation != keypoint_indices
        self._leg_mask = self._legs_swap_permutation != keypoint_indices
        # _swap_gather[h, v, k]: keypoint index read from view v under "view h fully swapped".
        self._swap_gather = np.tile(keypoint_indices, (num_views, num_views, 1))
        self._swap_gather[np.arange(num_views), np.arange(num_views)] = self._full_swap_permutation

        self._swap_count = 0
        self._previous_points = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float64)
        self._previous_timestamps = np.full(NUM_KEYPOINTS, -np.inf)

    @property
    def swap_count(self) -> int:
        """Cumulative number of views whose left/right labels were swapped (not cleared by reset)."""
        return self._swap_count

    def reset(self) -> None:
        self._previous_points[:] = 0.0
        self._previous_timestamps[:] = -np.inf

    def triangulate(self, multi_view: MultiViewPose) -> Skeleton3D | None:
        """Triangulate one synced frame. Returns None when no keypoint is accepted."""
        xy, valid = self._extract_views(multi_view)
        normals = _dlt_normal_matrices(self._projections, xy)
        points, residuals = self._solve_all_views(normals, xy, valid)

        if self._swap_test_needed(residuals, valid):
            swap = self._find_view_swap(xy, valid, normals, residuals)
            if swap is not None:
                view_idx, permutation = swap
                xy[view_idx] = xy[view_idx, permutation]
                valid[view_idx] = valid[view_idx, permutation]
                normals[view_idx] = normals[view_idx, permutation]
                points, residuals = self._solve_all_views(normals, xy, valid)
                self._swap_count += 1
                logger.debug("Left/right swap applied to camera %s", self._cam_ids[view_idx])

        points, used, residuals, accepted = self._select_views(
            xy, valid, normals, points, residuals, multi_view.timestamp
        )
        confidences = self._metric_confidences(points, used, residuals, accepted)
        kept = confidences > 0.0
        if not kept.any():
            return None
        positions = np.where(kept[:, None], points, 0.0)

        self._previous_points[kept] = positions[kept]
        self._previous_timestamps[kept] = multi_view.timestamp

        return Skeleton3D(
            keypoints=[
                Point3D(
                    x=float(positions[i, 0]),
                    y=float(positions[i, 1]),
                    z=float(positions[i, 2]),
                    confidence=float(confidences[i]),
                )
                for i in range(NUM_KEYPOINTS)
            ],
            timestamp=multi_view.timestamp,
            frame_index=multi_view.frame_index,
        )

    def _extract_views(self, multi_view: MultiViewPose) -> tuple[np.ndarray, np.ndarray]:
        num_views = len(self._cam_ids)
        xy = np.zeros((num_views, NUM_KEYPOINTS, 2), dtype=np.float64)
        view_confidences = np.zeros((num_views, NUM_KEYPOINTS), dtype=np.float64)
        for view_idx, cam_id in enumerate(self._cam_ids):
            skeleton = multi_view.views.get(cam_id)
            if skeleton is None:
                continue
            keypoints = skeleton.keypoints[:NUM_KEYPOINTS]
            if not keypoints:
                continue
            values = np.array([(kp.x, kp.y, kp.confidence) for kp in keypoints], dtype=np.float64)
            xy[view_idx, : len(keypoints)] = values[:, :2]
            view_confidences[view_idx, : len(keypoints)] = values[:, 2]
        valid = (
            np.isfinite(xy).all(axis=-1)
            & np.isfinite(view_confidences)
            & (view_confidences >= self._min_confidence)
        )
        return np.where(valid[..., None], xy, 0.0), valid

    def _solve_all_views(
        self, normals: np.ndarray, xy: np.ndarray, valid: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        normal_sums = np.einsum("vk,vkij->kij", valid, normals)[None]
        points, residuals = _solve_and_measure(self._projections, normal_sums, xy, valid[None])
        return points[0], residuals[0]

    def _swap_test_needed(self, residuals: np.ndarray, valid: np.ndarray) -> bool:
        failing = (
            (valid.sum(axis=0) >= 3)
            & (residuals.max(axis=0) > self._max_reprojection_error_px)
            & self._bilateral_mask
        )
        return bool(failing.any())

    def _find_view_swap(
        self,
        xy: np.ndarray,
        valid: np.ndarray,
        normals: np.ndarray,
        none_residuals: np.ndarray,
    ) -> tuple[int, np.ndarray] | None:
        # Hypotheses per view: full L/R swap and legs-only swap. Both are scored from one
        # solve per view with every bilateral keypoint swapped; the legs-only score keeps
        # the unswapped residuals for upper-body keypoints.
        num_views = xy.shape[0]
        view_rows = np.arange(num_views)[None, :, None]
        swapped_valid = valid[view_rows, self._swap_gather]
        swapped_xy = xy[view_rows, self._swap_gather]
        swapped_normals = normals[view_rows, self._swap_gather]
        normal_sums = np.einsum("hvk,hvkij->hkij", swapped_valid, swapped_normals)
        _, swapped_residuals = _solve_and_measure(
            self._projections, normal_sums, swapped_xy, swapped_valid
        )

        cap_px = SWAP_RESIDUAL_CAP_FACTOR * self._max_reprojection_error_px
        none_error, none_counted = _truncated_rms_px(none_residuals[None], valid[None], cap_px)
        full_error, full_counted = _truncated_rms_px(swapped_residuals, swapped_valid, cap_px)
        legs_error = np.where(self._leg_mask, full_error, none_error)
        legs_counted = np.where(self._leg_mask, full_counted, none_counted)

        none_cost = _masked_mean(none_error, none_counted & self._bilateral_mask)[0]
        hypothesis_costs = _masked_mean(
            np.concatenate([full_error, legs_error]),
            np.concatenate([full_counted, legs_counted]) & self._bilateral_mask,
        )
        best = int(np.argmin(hypothesis_costs))
        if not hypothesis_costs[best] < self._swap_margin_ratio * none_cost:
            return None
        if best < num_views:
            return best, self._full_swap_permutation
        return best - num_views, self._legs_swap_permutation

    def _select_views(
        self,
        xy: np.ndarray,
        valid: np.ndarray,
        normals: np.ndarray,
        points: np.ndarray,
        residuals: np.ndarray,
        timestamp: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        threshold_px = self._max_reprojection_error_px
        num_valid = valid.sum(axis=0)
        accepted = (num_valid >= self._min_views) & (residuals.max(axis=0) <= threshold_px)
        used = valid.copy()

        retry = ~accepted & (num_valid >= 3) & (self._min_views <= 2)
        if not retry.any():
            return points, used, residuals, accepted

        kpt_idx = np.flatnonzero(retry)
        pair_used = self._pair_masks[:, :, None] & valid[None, :, kpt_idx]
        normal_sums = np.einsum("pvn,vnij->pnij", pair_used, normals[:, kpt_idx])
        pair_points, pair_residuals = _solve_and_measure(
            self._projections, normal_sums, xy[:, kpt_idx], pair_used
        )
        passes = (pair_used.sum(axis=1) == 2) & (pair_residuals.max(axis=1) <= threshold_px)
        score = np.where(passes, np.sqrt(np.sum(pair_residuals**2, axis=1) / 2.0), np.inf)
        best_score = score.min(axis=0)
        choice = score.argmin(axis=0)

        # Cameras at similar heights have near-horizontal epipolar lines, so a horizontal
        # outlier leaves several pairs consistent but with 3D points far apart. Break such
        # ties with the previous accepted point; without one the keypoint is ambiguous.
        columns = np.arange(len(kpt_idx))
        ties = passes & (score <= best_score + PAIR_TIE_MARGIN_PX)
        spread_m = np.linalg.norm(pair_points - pair_points[choice, columns], axis=-1)
        ambiguous = (ties & (spread_m > PAIR_AMBIGUITY_DISTANCE_M)).any(axis=0)
        age_s = timestamp - self._previous_timestamps[kpt_idx]
        has_previous = (age_s >= 0.0) & (age_s <= PREVIOUS_POINT_MAX_AGE_S)
        if has_previous.any():
            distance_m = np.linalg.norm(pair_points - self._previous_points[kpt_idx], axis=-1)
            nearest = np.where(ties, distance_m, np.inf).argmin(axis=0)
            choice = np.where(has_previous, nearest, choice)

        points[kpt_idx] = pair_points[choice, columns]
        used[:, kpt_idx] = pair_used[choice, :, columns].T
        residuals[:, kpt_idx] = pair_residuals[choice, :, columns].T
        accepted[kpt_idx] = np.isfinite(best_score) & (has_previous | ~ambiguous)
        return points, used, residuals, accepted

    def _metric_confidences(
        self,
        points: np.ndarray,
        used: np.ndarray,
        residuals: np.ndarray,
        accepted: np.ndarray,
    ) -> np.ndarray:
        # u = max(sigma_hat_px, floor_px) * sqrt(trace((J^T J)^-1)), J = d(pixels)/d(meters) over used views.
        num_used = used.sum(axis=0)
        points_h = np.concatenate([points, np.ones((NUM_KEYPOINTS, 1))], axis=1)
        projected = np.einsum("vij,kj->vki", self._projections, points_h)
        depth = projected[..., 2]
        safe_depth = np.where(depth > MIN_PROJECTIVE_DEPTH, depth, 1.0)
        u_px = projected[..., 0] / safe_depth
        v_px = projected[..., 1] / safe_depth
        depth_row = self._projections[:, None, 2, :3]
        jacobian_u = (self._projections[:, None, 0, :3] - u_px[..., None] * depth_row) / safe_depth[..., None]
        jacobian_v = (self._projections[:, None, 1, :3] - v_px[..., None] * depth_row) / safe_depth[..., None]
        jtj = np.einsum("vk,vki,vkj->kij", used, jacobian_u, jacobian_u) + np.einsum(
            "vk,vki,vkj->kij", used, jacobian_v, jacobian_v
        )
        eigenvalues = np.linalg.eigvalsh(jtj)
        well_conditioned = (eigenvalues[:, 0] > 0.0) & (
            eigenvalues[:, 0] > JTJ_RELATIVE_EIGENVALUE_EPSILON * eigenvalues[:, 2]
        )
        safe_eigenvalues = np.where(well_conditioned[:, None], eigenvalues, 1.0)
        meters_per_px = np.sqrt(np.sum(1.0 / safe_eigenvalues, axis=1))

        degrees_of_freedom = np.maximum(2 * num_used - 3, MIN_DEGREES_OF_FREEDOM)
        sum_squared_px = np.sum(np.where(used, residuals, 0.0) ** 2, axis=0)
        sigma_hat_px = np.sqrt(sum_squared_px / degrees_of_freedom)
        # A per-keypoint sigma from 1-3 residual dof is too noisy to weight the
        # smoother; pool it across the frame, keeping only clear per-keypoint outliers.
        pooled = accepted & (num_used >= 2) & np.isfinite(sigma_hat_px)
        if pooled.any():
            frame_sigma_px = float(np.median(sigma_hat_px[pooled]))
            sigma_hat_px = np.where(
                sigma_hat_px > SIGMA_OUTLIER_FACTOR * frame_sigma_px, sigma_hat_px, frame_sigma_px
            )
        floor_px = np.where(num_used >= 3, MIN_PIXEL_SIGMA_MULTI_VIEW_PX, MIN_PIXEL_SIGMA_TWO_VIEW_PX)
        uncertainty_m = np.maximum(sigma_hat_px, floor_px) * meters_per_px

        confidences = 1.0 / (1.0 + (uncertainty_m / UNCERTAINTY_SCALE_M) ** 2)
        confidences = np.where(
            num_used == 2, np.minimum(confidences, self._two_view_confidence_cap), confidences
        )
        keep = accepted & well_conditioned & np.isfinite(confidences) & np.isfinite(points).all(axis=1)
        return np.where(keep, confidences, 0.0)
