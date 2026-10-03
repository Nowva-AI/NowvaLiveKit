"""
Tests for the deadlift's multi-view 3D bar tracker: cross-view association and
triangulation of the plate hubs (racked-bar rejection, left/right swaps, a hub seen
by a single camera), the constant-velocity Kalman with predicted states, the
time-aligned bar buffer, and the batched multi-view detector (stub model).

Synthetic rig: a head-on camera and two uprights at +-45 deg, 1280x720 with mild lens
distortion, looking at a loaded bar on the floor in front of the lifter (Y-down world,
X = subject's left, forward = -Z, floor 0.98 m below the hip midpoint).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from biomechanics.config import BarbellTrackingConfig  # noqa: E402
from biomechanics.deadlift.bar_buffer import BarStateBuffer  # noqa: E402
from biomechanics.deadlift.bar_detector_multi import BarCandidate2D, MultiViewBarDetector  # noqa: E402
from biomechanics.deadlift.bar_tracker_3d import (  # noqa: E402
    DEFAULT_HUB_DISTANCE_M,
    BarTracker3D,
    BarTracker3DConfig,
    associate_and_triangulate,
    wrists_and_ankles_world,
)
from biomechanics.pose.multi_camera import MultiCameraPoseProvider  # noqa: E402
from biomechanics.triangulation.calibration import CalibrationResult, CameraCalibration  # noqa: E402
from biomechanics.utils.types import CocoKeypoints as CK  # noqa: E402
from biomechanics.utils.types import Skeleton3D  # noqa: E402

RESOLUTION = (1280, 720)
FRAME_PERIOD_S = 1.0 / 30.0
FLOOR_Y_M = 0.98
PLATE_RADIUS_M = 0.225
HUB_Y_M = FLOOR_Y_M - PLATE_RADIUS_M
BAR_CENTRE_M = np.array([0.0, HUB_Y_M, -0.05])
WORLD_LEFT = np.array([1.0, 0.0, 0.0])
ANKLES_M = np.array([[0.15, FLOOR_Y_M - 0.08, 0.0], [-0.15, FLOOR_Y_M - 0.08, 0.0]])
WRISTS_ON_BAR_M = np.array([[0.25, HUB_Y_M - 0.07, -0.05], [-0.25, HUB_Y_M - 0.07, -0.05]])
# Racked at shoulder height behind the lifter, nowhere near the hands.
RACKED_BAR_CENTRE_M = np.array([0.0, FLOOR_Y_M - 1.45, 0.35])
# A spare bar lying on the floor behind the lifter, 0.85 m from the feet.
SPARE_BAR_CENTRE_M = np.array([0.0, HUB_Y_M, 0.85])

LENS_DISTORTION = np.array([-0.11, 0.06, 0.0007, -0.0005, 0.0])
# camera id -> (centre, aim point, fx, fy, cx, cy): head-on, subject's left upright, right upright.
CAMERAS = {
    "0": (np.array([0.0, FLOOR_Y_M - 1.0, -3.0]), np.array([0.0, FLOOR_Y_M - 0.7, 0.0]), 1005.0, 1001.0, 642.0, 357.0),
    "1": (np.array([1.84, FLOOR_Y_M - 1.4, -1.84]), np.array([0.0, FLOOR_Y_M - 0.6, 0.0]), 998.0, 995.0, 636.0, 362.0),
    "2": (np.array([-1.84, FLOOR_Y_M - 1.4, -1.84]), np.array([0.0, FLOOR_Y_M - 0.6, 0.0]), 1010.0, 1007.0, 645.0, 355.0),
}
FRONT, LEFT_UPRIGHT, RIGHT_UPRIGHT = "0", "1", "2"
LEFT, RIGHT = 0, 1

PIXEL_NOISE_PX = 0.7
HIDDEN_CONFIDENCE = 0.05
STATIC_END_TOL_M = 0.01
SINGLE_VIEW_END_TOL_M = 0.015
HUB_DISTANCE_TOL_M = 0.005
PREDICTION_TOL_M = 0.015
EXACT_TOL_M = 1e-6
NUM_STATIC_FRAMES = 30


def _look_at_camera(camera_id: str) -> CameraCalibration:
    centre, target, fx, fy, cx, cy = CAMERAS[camera_id]
    z_axis = (target - centre) / np.linalg.norm(target - centre)
    x_axis = np.cross([0.0, 1.0, 0.0], z_axis)
    x_axis /= np.linalg.norm(x_axis)
    rotation = np.stack([x_axis, np.cross(z_axis, x_axis), z_axis])
    translation = -rotation @ centre
    intrinsics = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    return CameraCalibration(
        camera_id=camera_id,
        projection_matrix=intrinsics @ np.hstack([rotation, translation[:, None]]),
        intrinsic_matrix=intrinsics,
        rotation_matrix=rotation,
        translation_vector=translation[:, None],
        reprojection_error=0.0,
        resolution=RESOLUTION,
        distortion_coeffs=LENS_DISTORTION,
    )


def _project(camera: CameraCalibration, points_m: np.ndarray) -> np.ndarray:
    pixels, _ = cv2.projectPoints(
        np.asarray(points_m, dtype=np.float64).reshape(-1, 3),
        cv2.Rodrigues(camera.rotation_matrix)[0],
        camera.translation_vector,
        camera.intrinsic_matrix,
        camera.distortion_coeffs,
    )
    pixels = pixels.reshape(-1, 2)
    assert ((pixels >= 0.0) & (pixels < RESOLUTION)).all(), "synthetic point outside the image"
    return pixels


def _bar_ends(centre_m: np.ndarray, hub_distance_m: float = DEFAULT_HUB_DISTANCE_M) -> np.ndarray:
    # (2, 3): left (+X) hub, right hub.
    half_span = hub_distance_m / 2.0 * WORLD_LEFT
    return np.stack([centre_m + half_span, centre_m - half_span])


def _bar_candidates(
    calibration: CalibrationResult,
    ends_m: np.ndarray,
    rng: np.random.Generator,
    noise_px: float = PIXEL_NOISE_PX,
    swapped_views: tuple[str, ...] = (),
    hidden_ends: tuple[tuple[str, int], ...] = (),
    score: float = 0.8,
) -> dict[str, BarCandidate2D]:
    # One candidate per camera; swapped views list the right hub first, hidden ends get low confidence.
    candidates = {}
    for camera_id, camera in calibration.cameras.items():
        pixels = _project(camera, ends_m) + rng.normal(0.0, noise_px, (2, 2))
        confidences = [HIDDEN_CONFIDENCE if (camera_id, end) in hidden_ends else 0.9 for end in (LEFT, RIGHT)]
        order = (RIGHT, LEFT) if camera_id in swapped_views else (LEFT, RIGHT)
        xs, ys = pixels[:, 0], pixels[:, 1]
        candidates[camera_id] = BarCandidate2D(
            end_a_px=tuple(pixels[order[0]].tolist()),
            end_b_px=tuple(pixels[order[1]].tolist()),
            end_confidences=(confidences[order[0]], confidences[order[1]]),
            box_xyxy=(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())),
            score=score,
        )
    return candidates


def _per_view(*candidate_sets: dict[str, BarCandidate2D]) -> dict[str, list[BarCandidate2D]]:
    return {
        camera_id: [candidates[camera_id] for candidates in candidate_sets if camera_id in candidates]
        for camera_id in CAMERAS
    }


def _end_errors_m(left_end_m: tuple, right_end_m: tuple, ends_m: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.array([left_end_m, right_end_m]) - ends_m, axis=1)


def _blank_frames() -> dict[str, np.ndarray]:
    return {camera_id: np.zeros((RESOLUTION[1], RESOLUTION[0], 3), dtype=np.uint8) for camera_id in CAMERAS}


class _StubDetector:
    available = True

    def __init__(self, candidates: dict[str, list[BarCandidate2D]]) -> None:
        self.candidates = candidates
        self.seen_views: list[list[str]] = []

    def detect(self, views: dict[str, np.ndarray]) -> dict[str, list[BarCandidate2D]]:
        self.seen_views.append(sorted(views))
        return self.candidates


@pytest.fixture
def rig_calibration() -> CalibrationResult:
    return CalibrationResult(cameras={camera_id: _look_at_camera(camera_id) for camera_id in CAMERAS})


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(7)


class TestAssociateAndTriangulate:
    def test_static_floor_bar_within_one_centimetre(self, rig_calibration, rng):
        ends_m = _bar_ends(BAR_CENTRE_M)
        for _ in range(20):
            association = associate_and_triangulate(
                _per_view(_bar_candidates(rig_calibration, ends_m, rng)), rig_calibration.cameras
            )
            assert association is not None
            assert _end_errors_m(association.left_end_m, association.right_end_m, ends_m).max() <= STATIC_END_TOL_M
            assert association.views == 3
            assert (association.left_views, association.right_views) == (3, 3)
            assert association.residual_px < 3.0 * PIXEL_NOISE_PX

    def test_left_right_swap_between_views_is_handled(self, rig_calibration, rng):
        ends_m = _bar_ends(BAR_CENTRE_M)
        candidates = _bar_candidates(rig_calibration, ends_m, rng, swapped_views=(LEFT_UPRIGHT, FRONT))
        association = associate_and_triangulate(_per_view(candidates), rig_calibration.cameras)
        assert association is not None
        assert association.views == 3
        assert _end_errors_m(association.left_end_m, association.right_end_m, ends_m).max() <= STATIC_END_TOL_M

    def test_exact_projections_triangulate_exactly(self, rig_calibration, rng):
        ends_m = _bar_ends(BAR_CENTRE_M)
        candidates = _bar_candidates(rig_calibration, ends_m, rng, noise_px=0.0)
        association = associate_and_triangulate(_per_view(candidates), rig_calibration.cameras)
        assert _end_errors_m(association.left_end_m, association.right_end_m, ends_m).max() == pytest.approx(
            0.0, abs=EXACT_TOL_M
        )

    def test_racked_bar_rejected_when_hands_are_on_the_floor_bar(self, rig_calibration, rng):
        floor_ends_m = _bar_ends(BAR_CENTRE_M)
        # The racked bar is exact and scores higher: only the hands gate can reject it.
        racked = _bar_candidates(rig_calibration, _bar_ends(RACKED_BAR_CENTRE_M), rng, noise_px=0.0, score=0.95)
        floor = _bar_candidates(rig_calibration, floor_ends_m, rng)
        association = associate_and_triangulate(
            _per_view(racked, floor), rig_calibration.cameras, wrists_world=WRISTS_ON_BAR_M
        )
        assert association is not None
        assert set(association.candidate_indices.values()) == {1}
        assert _end_errors_m(association.left_end_m, association.right_end_m, floor_ends_m).max() <= STATIC_END_TOL_M

    def test_racked_bar_rejected_by_height_above_the_feet(self, rig_calibration, rng):
        floor_ends_m = _bar_ends(BAR_CENTRE_M)
        racked = _bar_candidates(rig_calibration, _bar_ends(RACKED_BAR_CENTRE_M), rng, noise_px=0.0, score=0.95)
        floor = _bar_candidates(rig_calibration, floor_ends_m, rng)
        association = associate_and_triangulate(
            _per_view(racked, floor), rig_calibration.cameras, ankles_world=ANKLES_M
        )
        assert association is not None
        assert _end_errors_m(association.left_end_m, association.right_end_m, floor_ends_m).max() <= STATIC_END_TOL_M

    def test_floor_bar_far_from_the_feet_rejected(self, rig_calibration, rng):
        floor_ends_m = _bar_ends(BAR_CENTRE_M)
        spare = _bar_candidates(rig_calibration, _bar_ends(SPARE_BAR_CENTRE_M), rng, noise_px=0.0, score=0.95)
        floor = _bar_candidates(rig_calibration, floor_ends_m, rng)
        association = associate_and_triangulate(
            _per_view(spare, floor), rig_calibration.cameras, ankles_world=ANKLES_M
        )
        assert association is not None
        assert _end_errors_m(association.left_end_m, association.right_end_m, floor_ends_m).max() <= STATIC_END_TOL_M

    def test_only_racked_bar_with_hands_on_floor_returns_none(self, rig_calibration, rng):
        racked = _bar_candidates(rig_calibration, _bar_ends(RACKED_BAR_CENTRE_M), rng)
        association = associate_and_triangulate(
            _per_view(racked), rig_calibration.cameras, wrists_world=WRISTS_ON_BAR_M
        )
        assert association is None

    def test_non_finite_body_points_skip_the_gates(self, rig_calibration, rng):
        racked = _bar_candidates(rig_calibration, _bar_ends(RACKED_BAR_CENTRE_M), rng)
        association = associate_and_triangulate(
            _per_view(racked), rig_calibration.cameras, wrists_world=np.full((2, 3), np.nan)
        )
        assert association is not None

    def test_single_view_far_hub_placed_at_hub_distance(self, rig_calibration, rng):
        ends_m = _bar_ends(BAR_CENTRE_M)
        # The left hub is seen by its own upright only: hidden in the head-on view and,
        # behind the body and plates, in the opposite upright.
        hidden = ((FRONT, LEFT), (RIGHT_UPRIGHT, LEFT))
        for _ in range(20):
            candidates = _bar_candidates(rig_calibration, ends_m, rng, hidden_ends=hidden)
            association = associate_and_triangulate(_per_view(candidates), rig_calibration.cameras)
            assert association is not None
            assert (association.left_views, association.right_views) == (1, 3)
            span_m = np.linalg.norm(np.subtract(association.left_end_m, association.right_end_m))
            assert span_m == pytest.approx(DEFAULT_HUB_DISTANCE_M, abs=EXACT_TOL_M)
            errors_m = _end_errors_m(association.left_end_m, association.right_end_m, ends_m)
            assert errors_m[LEFT] <= SINGLE_VIEW_END_TOL_M
            assert errors_m[RIGHT] <= STATIC_END_TOL_M

    def test_single_view_hub_uses_the_given_hub_distance(self, rig_calibration, rng):
        ends_m = _bar_ends(BAR_CENTRE_M)
        hidden = ((FRONT, LEFT), (RIGHT_UPRIGHT, LEFT))
        candidates = _bar_candidates(rig_calibration, ends_m, rng, noise_px=0.0, hidden_ends=hidden)
        association = associate_and_triangulate(
            _per_view(candidates), rig_calibration.cameras, hub_distance_m=DEFAULT_HUB_DISTANCE_M + 0.1
        )
        span_m = np.linalg.norm(np.subtract(association.left_end_m, association.right_end_m))
        assert span_m == pytest.approx(DEFAULT_HUB_DISTANCE_M + 0.1, abs=EXACT_TOL_M)

    def test_one_view_returns_none(self, rig_calibration, rng):
        candidates = _bar_candidates(rig_calibration, _bar_ends(BAR_CENTRE_M), rng)
        assert associate_and_triangulate({FRONT: [candidates[FRONT]]}, rig_calibration.cameras) is None

    def test_inconsistent_candidates_return_none(self, rig_calibration, rng):
        candidates = _bar_candidates(rig_calibration, _bar_ends(BAR_CENTRE_M), rng)
        shifted = candidates[LEFT_UPRIGHT].model_copy(update={
            "end_a_px": (candidates[LEFT_UPRIGHT].end_a_px[0], candidates[LEFT_UPRIGHT].end_a_px[1] + 80.0),
            "end_b_px": (candidates[LEFT_UPRIGHT].end_b_px[0], candidates[LEFT_UPRIGHT].end_b_px[1] + 80.0),
        })
        association = associate_and_triangulate(
            {FRONT: [candidates[FRONT]], LEFT_UPRIGHT: [shifted]}, rig_calibration.cameras
        )
        assert association is None

    def test_outlier_view_is_left_out(self, rig_calibration, rng):
        ends_m = _bar_ends(BAR_CENTRE_M)
        candidates = _bar_candidates(rig_calibration, ends_m, rng)
        candidates[RIGHT_UPRIGHT] = candidates[RIGHT_UPRIGHT].model_copy(update={
            "end_a_px": (candidates[RIGHT_UPRIGHT].end_a_px[0] + 60.0, candidates[RIGHT_UPRIGHT].end_a_px[1]),
        })
        association = associate_and_triangulate(_per_view(candidates), rig_calibration.cameras)
        assert association is not None
        assert association.views == 2
        assert RIGHT_UPRIGHT not in association.candidate_indices
        assert _end_errors_m(association.left_end_m, association.right_end_m, ends_m).max() <= STATIC_END_TOL_M


class TestBarTracker3D:
    def test_static_bar_filtered_within_one_centimetre(self, rig_calibration, rng):
        tracker = BarTracker3D(rig_calibration)
        ends_m = _bar_ends(BAR_CENTRE_M)
        for frame in range(NUM_STATIC_FRAMES):
            state = tracker.update_from_candidates(
                _per_view(_bar_candidates(rig_calibration, ends_m, rng)), frame * FRAME_PERIOD_S
            )
        assert state is not None and not state.predicted
        assert state.views == 3
        assert _end_errors_m(state.left_end_m, state.right_end_m, ends_m).max() <= STATIC_END_TOL_M
        assert np.linalg.norm(state.velocity_mps) < 0.1

    def test_missing_detection_is_predicted_then_expires(self, rig_calibration, rng):
        tracker = BarTracker3D(rig_calibration)
        ends_m = _bar_ends(BAR_CENTRE_M)
        for frame in range(10):
            tracker.update_from_candidates(_per_view(_bar_candidates(rig_calibration, ends_m, rng)), frame * FRAME_PERIOD_S)
        last_measured_s = 9 * FRAME_PERIOD_S
        max_prediction_s = BarTracker3DConfig().max_prediction_s

        predicted = tracker.update_from_candidates({}, last_measured_s + FRAME_PERIOD_S)
        assert predicted is not None and predicted.predicted
        assert predicted.views == 0 and math.isnan(predicted.residual_px)
        assert _end_errors_m(predicted.left_end_m, predicted.right_end_m, ends_m).max() <= STATIC_END_TOL_M
        assert tracker.update_from_candidates({}, last_measured_s + max_prediction_s - 1e-3).predicted
        assert tracker.update_from_candidates({}, last_measured_s + max_prediction_s + FRAME_PERIOD_S) is None

        reacquired_s = last_measured_s + 1.0
        state = tracker.update_from_candidates(_per_view(_bar_candidates(rig_calibration, ends_m, rng)), reacquired_s)
        assert state is not None and not state.predicted
        assert state.velocity_mps == (0.0, 0.0, 0.0)

    def test_prediction_follows_a_rising_bar(self, rig_calibration, rng):
        tracker = BarTracker3D(rig_calibration)
        rise_mps = 0.6
        for frame in range(20):
            centre = BAR_CENTRE_M + np.array([0.0, -rise_mps * frame * FRAME_PERIOD_S, 0.0])
            tracker.update_from_candidates(
                _per_view(_bar_candidates(rig_calibration, _bar_ends(centre), rng)), frame * FRAME_PERIOD_S
            )
        gap_frame = 22
        predicted = tracker.update_from_candidates({}, gap_frame * FRAME_PERIOD_S)
        true_ends_m = _bar_ends(BAR_CENTRE_M + np.array([0.0, -rise_mps * gap_frame * FRAME_PERIOD_S, 0.0]))
        assert predicted.predicted
        assert _end_errors_m(predicted.left_end_m, predicted.right_end_m, true_ends_m).max() <= PREDICTION_TOL_M
        assert predicted.velocity_mps[1] == pytest.approx(-rise_mps, abs=0.1)

    def test_outlier_association_is_predicted_not_tracked(self, rig_calibration, rng):
        tracker = BarTracker3D(rig_calibration)
        floor_ends_m = _bar_ends(BAR_CENTRE_M)
        for frame in range(10):
            tracker.update_from_candidates(
                _per_view(_bar_candidates(rig_calibration, floor_ends_m, rng)), frame * FRAME_PERIOD_S
            )
        jump = tracker.update_from_candidates(
            _per_view(_bar_candidates(rig_calibration, _bar_ends(RACKED_BAR_CENTRE_M), rng)), 10 * FRAME_PERIOD_S
        )
        assert jump.predicted
        assert _end_errors_m(jump.left_end_m, jump.right_end_m, floor_ends_m).max() <= STATIC_END_TOL_M

    def test_hub_distance_learned_from_frames_seeing_both_hubs(self, rig_calibration, rng):
        tracker = BarTracker3D(rig_calibration)
        true_hub_distance_m = 1.57  # two bumpers per side
        ends_m = _bar_ends(BAR_CENTRE_M, true_hub_distance_m)
        assert tracker.hub_distance_m == DEFAULT_HUB_DISTANCE_M
        for frame in range(NUM_STATIC_FRAMES):
            tracker.update_from_candidates(_per_view(_bar_candidates(rig_calibration, ends_m, rng)), frame * FRAME_PERIOD_S)
        assert tracker.hub_distance_m == pytest.approx(true_hub_distance_m, abs=HUB_DISTANCE_TOL_M)

        hidden = ((FRONT, LEFT), (RIGHT_UPRIGHT, LEFT))
        state = tracker.update_from_candidates(
            _per_view(_bar_candidates(rig_calibration, ends_m, rng, hidden_ends=hidden)),
            NUM_STATIC_FRAMES * FRAME_PERIOD_S,
        )
        assert _end_errors_m(state.left_end_m, state.right_end_m, ends_m).max() <= STATIC_END_TOL_M

    def test_reset_forgets_track_and_hub_distance(self, rig_calibration, rng):
        tracker = BarTracker3D(rig_calibration)
        ends_m = _bar_ends(BAR_CENTRE_M, 1.57)
        for frame in range(NUM_STATIC_FRAMES):
            tracker.update_from_candidates(_per_view(_bar_candidates(rig_calibration, ends_m, rng)), frame * FRAME_PERIOD_S)
        tracker.reset()
        assert tracker.hub_distance_m == DEFAULT_HUB_DISTANCE_M
        assert tracker.update_from_candidates({}, NUM_STATIC_FRAMES * FRAME_PERIOD_S) is None

    def test_update_detects_on_every_view(self, rig_calibration, rng):
        detector = _StubDetector(_per_view(_bar_candidates(rig_calibration, _bar_ends(BAR_CENTRE_M), rng)))
        tracker = BarTracker3D(rig_calibration, detector=detector)
        state = tracker.update(_blank_frames(), 0.0, wrists_world=WRISTS_ON_BAR_M, ankles_world=ANKLES_M)
        assert state is not None and state.views == 3
        assert detector.seen_views == [sorted(CAMERAS)]

    def test_update_without_detector_raises(self, rig_calibration):
        with pytest.raises(ValueError):
            BarTracker3D(rig_calibration).update({}, 0.0)

    def test_no_calibration_returns_none(self, rig_calibration, rng):
        detector = _StubDetector(_per_view(_bar_candidates(rig_calibration, _bar_ends(BAR_CENTRE_M), rng)))
        tracker = BarTracker3D(None, detector=detector)
        assert tracker.update(_blank_frames(), 0.0) is None
        assert detector.seen_views == []


class TestFromProvider:
    def test_follows_a_calibration_installed_later(self, rig_calibration, rng):
        provider = MultiCameraPoseProvider(device_ids=[0, 1, 2])
        detector = _StubDetector(_per_view(_bar_candidates(rig_calibration, _bar_ends(BAR_CENTRE_M), rng)))
        tracker = BarTracker3D.from_provider(provider, detector=detector)
        assert tracker.update(_blank_frames(), 0.0) is None

        provider.install_calibration(rig_calibration)
        state = tracker.update(_blank_frames(), FRAME_PERIOD_S)
        assert state is not None and state.views == 3

    def test_new_calibration_resets_the_track(self, rig_calibration, rng):
        provider = MultiCameraPoseProvider(device_ids=[0, 1, 2])
        provider.install_calibration(rig_calibration)
        detector = _StubDetector(_per_view(_bar_candidates(rig_calibration, _bar_ends(BAR_CENTRE_M), rng)))
        tracker = BarTracker3D.from_provider(provider, detector=detector)
        assert tracker.update(_blank_frames(), 0.0) is not None

        provider.install_calibration(CalibrationResult(cameras=dict(rig_calibration.cameras)))
        assert tracker.update_from_candidates({}, FRAME_PERIOD_S) is None

    def test_missing_frames_return_none(self, rig_calibration, rng):
        provider = MultiCameraPoseProvider(device_ids=[0, 1, 2])
        provider.install_calibration(rig_calibration)
        detector = _StubDetector(_per_view(_bar_candidates(rig_calibration, _bar_ends(BAR_CENTRE_M), rng)))
        tracker = BarTracker3D.from_provider(provider, detector=detector)
        assert tracker.update(None, 0.0) is None
        assert detector.seen_views == []

    def test_unavailable_detector_returns_none(self, rig_calibration, tmp_path):
        provider = MultiCameraPoseProvider(device_ids=[0, 1, 2])
        provider.install_calibration(rig_calibration)
        barbell_config = BarbellTrackingConfig(model_path=str(tmp_path / "barbell_keypoints.pt"))
        tracker = BarTracker3D.from_provider(provider, barbell_config)
        assert tracker.update(_blank_frames(), 0.0) is None


class TestWristsAndAnklesWorld:
    def test_rejected_keypoints_become_nan(self):
        positions = np.arange(21 * 3, dtype=np.float64).reshape(21, 3)
        confidences = np.ones(21)
        confidences[CK.RIGHT_WRIST] = 0.0
        wrists, ankles = wrists_and_ankles_world(Skeleton3D.from_numpy(positions, confidences))
        assert wrists[0] == pytest.approx(positions[CK.LEFT_WRIST])
        assert np.isnan(wrists[1]).all()
        assert ankles == pytest.approx(positions[[CK.LEFT_ANKLE, CK.RIGHT_ANKLE]])

    def test_no_skeleton_gives_no_body_points(self):
        assert wrists_and_ankles_world(None) == (None, None)


class TestTimeAlignment:
    def test_lagged_skeleton_reads_the_bar_at_its_capture_time(self, rig_calibration, rng):
        # The analysis skeleton lags capture by 2 frames (kalman.lag_frames); bar detection
        # runs at 15 Hz, so every other state is a prediction (PLAN.md §2.2, §9).
        lag_frames = 2
        rise_mps = 0.5
        tracker = BarTracker3D(rig_calibration)
        buffer = BarStateBuffer()
        for frame in range(40):
            capture_s = frame * FRAME_PERIOD_S
            centre = BAR_CENTRE_M + np.array([0.0, -rise_mps * capture_s, 0.0])
            candidates = _per_view(_bar_candidates(rig_calibration, _bar_ends(centre), rng)) if frame % 2 == 0 else {}
            state = tracker.update_from_candidates(candidates, capture_s)
            assert state is not None
            buffer.push(state)
            if frame < 10 + lag_frames:
                continue
            analysis_s = (frame - lag_frames) * FRAME_PERIOD_S
            matched = buffer.at(analysis_s)
            assert matched is not None
            assert matched.timestamp == pytest.approx(analysis_s, abs=1e-9)
            true_height_m = PLATE_RADIUS_M + rise_mps * analysis_s
            matched_height_m = FLOOR_Y_M - matched.centre[1]
            assert matched_height_m == pytest.approx(true_height_m, abs=PREDICTION_TOL_M)
            # Reading the newest state instead would be 2 frames (3.3 cm) ahead of the skeleton.
            newest_height_m = FLOOR_Y_M - state.centre[1]
            assert newest_height_m - matched_height_m == pytest.approx(
                rise_mps * lag_frames * FRAME_PERIOD_S, abs=PREDICTION_TOL_M
            )


def _fake_result(scores: list[float], keypoints: list[list[list[float]]], keypoint_conf: list[list[float]] | None) -> SimpleNamespace:
    boxes = SimpleNamespace(
        conf=np.array(scores, dtype=np.float32),
        xyxy=np.array([[10.0 * i, 20.0, 10.0 * i + 300.0, 80.0] for i in range(len(scores))], dtype=np.float32).reshape(-1, 4),
    )
    keypoint_data = SimpleNamespace(
        xy=np.array(keypoints, dtype=np.float32).reshape(-1, 2, 2),
        conf=None if keypoint_conf is None else np.array(keypoint_conf, dtype=np.float32),
    )
    return SimpleNamespace(boxes=boxes, keypoints=keypoint_data)


class _FakeYolo:
    def __init__(self, results: list[SimpleNamespace]) -> None:
        self.results = results
        self.calls: list[dict] = []

    def predict(self, source: list[np.ndarray], **kwargs: object) -> list[SimpleNamespace]:
        self.calls.append({"batch": len(source), **kwargs})
        return self.results[: len(source)]


class TestMultiViewBarDetector:
    def test_missing_weights_reported_unavailable(self, tmp_path):
        detector = MultiViewBarDetector(tmp_path / "barbell_keypoints.pt")
        assert not detector.available
        assert "barbell_keypoints.pt" in detector.error
        with pytest.raises(RuntimeError):
            detector.detect({FRONT: np.zeros((8, 8, 3), dtype=np.uint8)})

    def test_from_config_uses_the_barbell_tracking_weights(self):
        config = BarbellTrackingConfig(model_path="models/other.pt", conf_threshold=0.4, imgsz=960)
        detector = MultiViewBarDetector.from_config(config)
        assert detector.model_path == Path("models/other.pt")
        assert (detector.conf_threshold, detector.imgsz) == (0.4, 960)

    def test_one_batched_predict_returns_every_candidate(self):
        model = _FakeYolo([
            _fake_result([0.4, 0.9], [[[100, 400], [900, 410]], [[120, 300], [880, 305]]], [[0.8, 0.2], [0.9, 0.95]]),
            _fake_result([], [], None),
            _fake_result([0.7], [[[200, 500], [1000, 520]]], None),
        ])
        detector = MultiViewBarDetector(model=model)
        frames = {camera_id: np.zeros((8, 8, 3), dtype=np.uint8) for camera_id in (FRONT, LEFT_UPRIGHT, RIGHT_UPRIGHT)}
        candidates = detector.detect(frames)

        assert len(model.calls) == 1 and model.calls[0]["batch"] == 3
        assert [candidate.score for candidate in candidates[FRONT]] == pytest.approx([0.9, 0.4])
        assert candidates[FRONT][1].end_confidences == pytest.approx((0.8, 0.2))
        assert candidates[FRONT][0].end_a_px == (120.0, 300.0)
        assert candidates[LEFT_UPRIGHT] == []
        assert candidates[RIGHT_UPRIGHT][0].end_confidences == pytest.approx((0.7, 0.7))

    def test_candidates_capped_per_view(self):
        scores = [0.3, 0.9, 0.5, 0.6, 0.8, 0.7]
        model = _FakeYolo([_fake_result(scores, [[[i, 0], [i + 500, 0]] for i in range(6)], None)])
        detector = MultiViewBarDetector(model=model, max_candidates_per_view=4)
        candidates = detector.detect({FRONT: np.zeros((8, 8, 3), dtype=np.uint8)})[FRONT]
        assert [candidate.score for candidate in candidates] == pytest.approx([0.9, 0.8, 0.7, 0.6])

    def test_no_views_skips_the_model(self):
        model = _FakeYolo([])
        assert MultiViewBarDetector(model=model).detect({}) == {}
        assert model.calls == []
