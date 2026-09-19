"""
Tests for RTMPose: tracked person crop, parabola SimCC decode, raw keypoint scores,
21-keypoint layout and constant-shape batching. Fake ONNX sessions cover the logic;
the real halpe26 model (skipped when absent) checks empty and real frames.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from biomechanics.pose.rtmpose import (  # noqa: E402
    CROP_PADDING,
    DEFAULT_MODEL_DIR,
    DEFAULT_HALPE26_MODEL_NAME,
    HALPE26_LEFT_BIG_TOE,
    HALPE26_LEFT_HEEL,
    HALPE26_RIGHT_BIG_TOE,
    HALPE26_RIGHT_HEEL,
    INPUT_ASPECT_RATIO,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    MIN_CROP_HEIGHT_PX,
    MIN_TRACKED_KEYPOINTS,
    NUM_COCO17_KEYPOINTS,
    NUM_HALPE26_KEYPOINTS,
    NUM_SKELETON_KEYPOINTS,
    ONNXRUNTIME_AVAILABLE,
    RTMPoseEstimator,
    SIMCC_SPLIT_RATIO,
    _crop_box_from_points,
    _full_frame_box,
    _preprocess,
    decode_simcc,
)
from biomechanics.utils.types import CocoKeypoints as CK  # noqa: E402

COORD_TOL_PX = 1e-6
SUBBIN_TOL = 1e-6
SCORE_TOL = 1e-6
ROUND_TRIP_TOL_INPUT_PX = 1.0

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
X_BINS = INPUT_WIDTH * int(SIMCC_SPLIT_RATIO)
Y_BINS = INPUT_HEIGHT * int(SIMCC_SPLIT_RATIO)
BATCH_SHAPE_TAIL = (3, INPUT_HEIGHT, INPUT_WIDTH)

CONFIDENT_LOGIT = 0.9
BACKGROUND_LOGIT = 0.0
NO_PERSON_LOGIT = 0.1
CONFIDENCE_THRESHOLD = 0.3

REAL_MODEL_PATH = DEFAULT_MODEL_DIR / DEFAULT_HALPE26_MODEL_NAME
SQUAT_VIDEO_PATH = Path(__file__).parent.parent.parent / "data" / "squats.mov"
PORTRAIT_CANVAS_WIDTH = 405


def _peaked_logits(peak_bins: np.ndarray, num_bins: int, peak_value: float) -> np.ndarray:
    # (K,) integer peak bins -> (K, num_bins) logits with one symmetric hot bin each.
    logits = np.full((len(peak_bins), num_bins), BACKGROUND_LOGIT, dtype=np.float32)
    logits[np.arange(len(peak_bins)), peak_bins] = peak_value
    return logits


def _spread_peak_bins(num_keypoints: int) -> tuple[np.ndarray, np.ndarray]:
    # Distinct, spread-out peaks inside the model input (a person-like extent).
    x_bins = (60 + 10 * np.arange(num_keypoints)) % (X_BINS - 40) + 20
    y_bins = 40 + 16 * np.arange(num_keypoints)
    return x_bins.astype(int), y_bins.astype(int)


class _FakeSession:
    def __init__(
        self,
        num_keypoints: int = NUM_HALPE26_KEYPOINTS,
        confident_keypoints: int | None = None,
        dead_rows: frozenset[int] = frozenset(),
    ) -> None:
        self.received_batches: list[np.ndarray] = []
        self._num_keypoints = num_keypoints
        self._confident_keypoints = num_keypoints if confident_keypoints is None else confident_keypoints
        self._dead_rows = dead_rows
        self.x_bins, self.y_bins = _spread_peak_bins(num_keypoints)

    @property
    def received_shapes(self) -> list[tuple[int, ...]]:
        return [batch.shape for batch in self.received_batches]

    def run(self, output_names: list[str] | None, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        batch = next(iter(feed.values()))
        self.received_batches.append(batch.copy())
        peak_values = np.full(self._num_keypoints, NO_PERSON_LOGIT, dtype=np.float32)
        peak_values[: self._confident_keypoints] = CONFIDENT_LOGIT
        x_logits = np.stack([_peaked_logits(self.x_bins, X_BINS, 1.0) * peak_values[:, None]] * batch.shape[0])
        y_logits = np.stack([_peaked_logits(self.y_bins, Y_BINS, 1.0) * peak_values[:, None]] * batch.shape[0])
        for row in self._dead_rows:
            if row < batch.shape[0]:
                x_logits[row] = NO_PERSON_LOGIT * 0.5
                y_logits[row] = NO_PERSON_LOGIT * 0.5
        return [x_logits, y_logits]


def _make_estimator(
    batch_size: int = 1,
    keypoint_format: str = "halpe26",
    session: _FakeSession | None = None,
) -> tuple[RTMPoseEstimator, _FakeSession]:
    estimator = RTMPoseEstimator(
        confidence_threshold=CONFIDENCE_THRESHOLD,
        keypoint_format=keypoint_format,
        batch_size=batch_size,
    )
    num_keypoints = NUM_HALPE26_KEYPOINTS if keypoint_format == "halpe26" else NUM_COCO17_KEYPOINTS
    fake = session if session is not None else _FakeSession(num_keypoints=num_keypoints)
    estimator._session = fake
    estimator._input_name = "input"
    estimator._initialized = True
    return estimator, fake


def _make_frame(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)


def _expected_frame_coords(box: np.ndarray, x_bins: np.ndarray, y_bins: np.ndarray) -> np.ndarray:
    frame_x = (x_bins / SIMCC_SPLIT_RATIO / INPUT_WIDTH - 0.5) * box[2] + box[0]
    frame_y = (y_bins / SIMCC_SPLIT_RATIO / INPUT_HEIGHT - 0.5) * box[3] + box[1]
    return np.stack([frame_x, frame_y], axis=-1)


@pytest.mark.skipif(not ONNXRUNTIME_AVAILABLE, reason="onnxruntime not installed")
class TestDecodeSimcc:

    def test_full_frame_box_maps_bins_to_frame_pixels(self) -> None:
        box = np.array([FRAME_WIDTH / 2, FRAME_HEIGHT / 2, FRAME_WIDTH, FRAME_HEIGHT], dtype=np.float64)
        x_logits = _peaked_logits(np.array([40]), X_BINS, CONFIDENT_LOGIT)[None]
        y_logits = _peaked_logits(np.array([100]), Y_BINS, CONFIDENT_LOGIT)[None]

        keypoints = decode_simcc(x_logits, y_logits, box[None])

        assert keypoints.shape == (1, 1, 3)
        assert keypoints[0, 0, 0] == pytest.approx(20.0 * FRAME_WIDTH / INPUT_WIDTH, abs=COORD_TOL_PX)
        assert keypoints[0, 0, 1] == pytest.approx(50.0 * FRAME_HEIGHT / INPUT_HEIGHT, abs=COORD_TOL_PX)

    def test_crop_box_maps_bins_to_frame_pixels(self) -> None:
        box = np.array([600.0, 400.0, 300.0, 400.0])
        x_bins = np.array([0, X_BINS // 2, 300])
        y_bins = np.array([0, Y_BINS // 2, 100])
        x_logits = _peaked_logits(x_bins, X_BINS, CONFIDENT_LOGIT)[None]
        y_logits = _peaked_logits(y_bins, Y_BINS, CONFIDENT_LOGIT)[None]

        keypoints = decode_simcc(x_logits, y_logits, box[None])

        expected = _expected_frame_coords(box, x_bins, y_bins)
        assert keypoints[0, 0, :2] == pytest.approx([450.0, 200.0], abs=COORD_TOL_PX)
        assert keypoints[0, 1, :2] == pytest.approx([600.0, 400.0], abs=COORD_TOL_PX)
        assert keypoints[0, :, :2] == pytest.approx(expected, abs=COORD_TOL_PX)

    def test_decode_uses_each_rows_own_box(self) -> None:
        boxes = np.array([[600.0, 400.0, 300.0, 400.0], [200.0, 300.0, 150.0, 200.0]])
        x_logits = np.stack([_peaked_logits(np.array([100]), X_BINS, CONFIDENT_LOGIT)] * 2)
        y_logits = np.stack([_peaked_logits(np.array([200]), Y_BINS, CONFIDENT_LOGIT)] * 2)

        keypoints = decode_simcc(x_logits, y_logits, boxes)

        for row in range(2):
            expected = _expected_frame_coords(boxes[row], np.array([100]), np.array([200]))
            assert keypoints[row, :, :2] == pytest.approx(expected, abs=COORD_TOL_PX)

    def test_warp_and_decode_round_trip_to_frame_point(self) -> None:
        box = np.array([700.0, 300.0, 384.0, 512.0])
        dot_x, dot_y = 620, 410
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        cv2.circle(frame, (dot_x, dot_y), 2, (255, 255, 255), thickness=-1)

        blob = _preprocess(frame, box)
        brightness = blob[0].sum(axis=0)
        input_y, input_x = np.unravel_index(np.argmax(brightness), brightness.shape)
        x_logits = _peaked_logits(np.array([int(input_x * SIMCC_SPLIT_RATIO)]), X_BINS, CONFIDENT_LOGIT)[None]
        y_logits = _peaked_logits(np.array([int(input_y * SIMCC_SPLIT_RATIO)]), Y_BINS, CONFIDENT_LOGIT)[None]
        keypoints = decode_simcc(x_logits, y_logits, box[None])

        frame_px_per_input_px = box[2] / INPUT_WIDTH
        tolerance_px = ROUND_TRIP_TOL_INPUT_PX * frame_px_per_input_px
        assert keypoints[0, 0, 0] == pytest.approx(dot_x, abs=tolerance_px)
        assert keypoints[0, 0, 1] == pytest.approx(dot_y, abs=tolerance_px)

    def test_parabola_recovers_subbin_peak(self) -> None:
        true_peak_bin = 100.3
        bins = np.arange(X_BINS, dtype=np.float64)
        x_logits = (1.0 - 0.01 * (bins - true_peak_bin) ** 2).astype(np.float64)[None, None]
        y_logits = _peaked_logits(np.array([50]), Y_BINS, CONFIDENT_LOGIT)[None]
        box = np.array([INPUT_WIDTH / 2, INPUT_HEIGHT / 2, INPUT_WIDTH, INPUT_HEIGHT], dtype=np.float64)

        keypoints = decode_simcc(x_logits, y_logits, box[None])

        assert keypoints[0, 0, 0] == pytest.approx(true_peak_bin / SIMCC_SPLIT_RATIO, abs=SUBBIN_TOL)

    def test_peak_on_edge_bin_has_no_subbin_offset(self) -> None:
        x_logits = np.linspace(0.0, 1.0, X_BINS, dtype=np.float32)[None, None]
        y_logits = np.linspace(1.0, 0.0, Y_BINS, dtype=np.float32)[None, None]
        box = np.array([INPUT_WIDTH / 2, INPUT_HEIGHT / 2, INPUT_WIDTH, INPUT_HEIGHT], dtype=np.float64)

        keypoints = decode_simcc(x_logits, y_logits, box[None])

        assert keypoints[0, 0, 0] == pytest.approx((X_BINS - 1) / SIMCC_SPLIT_RATIO, abs=SUBBIN_TOL)
        assert keypoints[0, 0, 1] == pytest.approx(0.0, abs=SUBBIN_TOL)

    def test_score_is_raw_minimum_of_axis_maxima(self) -> None:
        x_logits = _peaked_logits(np.array([100]), X_BINS, 0.7)[None]
        y_logits = _peaked_logits(np.array([100]), Y_BINS, 0.4)[None]
        box = np.array([[640.0, 360.0, 300.0, 400.0]])

        keypoints = decode_simcc(x_logits, y_logits, box)

        assert keypoints[0, 0, 2] == pytest.approx(0.4, abs=SCORE_TOL)

    def test_score_clipped_to_unit_range(self) -> None:
        x_logits = np.stack([_peaked_logits(np.array([100]), X_BINS, 1.3), np.full((1, X_BINS), -0.5, dtype=np.float32)])
        y_logits = np.stack([_peaked_logits(np.array([100]), Y_BINS, 1.2), np.full((1, Y_BINS), -0.5, dtype=np.float32)])
        boxes = np.array([[640.0, 360.0, 300.0, 400.0]] * 2)

        keypoints = decode_simcc(x_logits, y_logits, boxes)

        assert keypoints[0, 0, 2] == pytest.approx(1.0, abs=SCORE_TOL)
        assert keypoints[1, 0, 2] == pytest.approx(0.0, abs=SCORE_TOL)


class TestCropBox:

    def test_tall_points_pad_height_and_match_model_aspect(self) -> None:
        points = np.array([[300.0, 100.0], [400.0, 500.0], [350.0, 300.0]])

        box = _crop_box_from_points(points)

        assert box[:2] == pytest.approx([350.0, 300.0], abs=COORD_TOL_PX)
        assert box[3] == pytest.approx(400.0 * CROP_PADDING, abs=COORD_TOL_PX)
        assert box[2] / box[3] == pytest.approx(INPUT_ASPECT_RATIO, abs=COORD_TOL_PX)

    def test_wide_points_expand_height_to_model_aspect(self) -> None:
        points = np.array([[100.0, 200.0], [700.0, 300.0]])

        box = _crop_box_from_points(points)

        assert box[2] == pytest.approx(600.0 * CROP_PADDING, abs=COORD_TOL_PX)
        assert box[3] == pytest.approx(600.0 * CROP_PADDING / INPUT_ASPECT_RATIO, abs=COORD_TOL_PX)

    def test_degenerate_points_use_minimum_crop_height(self) -> None:
        points = np.array([[500.0, 250.0], [500.0, 250.0]])

        box = _crop_box_from_points(points)

        assert box[3] == pytest.approx(MIN_CROP_HEIGHT_PX * CROP_PADDING, abs=COORD_TOL_PX)
        assert box[2] == pytest.approx(MIN_CROP_HEIGHT_PX * CROP_PADDING * INPUT_ASPECT_RATIO, abs=COORD_TOL_PX)

    def test_full_frame_box_covers_frame(self) -> None:
        box = _full_frame_box(_make_frame())

        assert box == pytest.approx([FRAME_WIDTH / 2, FRAME_HEIGHT / 2, FRAME_WIDTH, FRAME_HEIGHT], abs=COORD_TOL_PX)


@pytest.mark.skipif(not ONNXRUNTIME_AVAILABLE, reason="onnxruntime not installed")
class TestCropTracking:

    def test_first_frame_bootstraps_on_full_frame_then_reruns_on_crop(self) -> None:
        estimator, fake = _make_estimator()
        frame = _make_frame()

        skeleton = estimator.estimate(frame, camera_id=0)

        full_frame_box = _full_frame_box(frame)
        crop_box = _crop_box_from_points(_expected_frame_coords(full_frame_box, fake.x_bins, fake.y_bins))
        assert len(fake.received_batches) == 2
        np.testing.assert_array_equal(fake.received_batches[0], _preprocess(frame, full_frame_box))
        np.testing.assert_array_equal(fake.received_batches[1], _preprocess(frame, crop_box))
        expected_nose = _expected_frame_coords(crop_box, fake.x_bins[:1], fake.y_bins[:1])[0]
        assert [skeleton.keypoints[CK.NOSE].x, skeleton.keypoints[CK.NOSE].y] == pytest.approx(
            expected_nose, abs=COORD_TOL_PX
        )

    def test_tracked_frame_runs_once_on_previous_crop(self) -> None:
        estimator, fake = _make_estimator()
        frame = _make_frame()
        estimator.estimate(frame, camera_id=0)
        previous_box = estimator._crop_boxes[0].copy()

        estimator.estimate(frame, camera_id=0)

        assert len(fake.received_batches) == 3
        np.testing.assert_array_equal(fake.received_batches[2], _preprocess(frame, previous_box))

    def test_too_few_confident_keypoints_falls_back_to_full_frame(self) -> None:
        session = _FakeSession(confident_keypoints=MIN_TRACKED_KEYPOINTS - 1)
        estimator, fake = _make_estimator(session=session)
        frame = _make_frame()

        estimator.estimate(frame, camera_id=0)
        estimator.estimate(frame, camera_id=0)

        assert 0 not in estimator._crop_boxes
        assert len(fake.received_batches) == 2
        for batch in fake.received_batches:
            np.testing.assert_array_equal(batch, _preprocess(frame, _full_frame_box(frame)))

    def test_lost_view_redetects_on_full_frame(self) -> None:
        estimator, fake = _make_estimator()
        frame = _make_frame()
        estimator.estimate(frame, camera_id=0)
        fake._confident_keypoints = MIN_TRACKED_KEYPOINTS - 1

        estimator.estimate(frame, camera_id=0)
        estimator.estimate(frame, camera_id=0)

        np.testing.assert_array_equal(fake.received_batches[-1], _preprocess(frame, _full_frame_box(frame)))

    def test_reset_tracking_clears_crops(self) -> None:
        estimator, _ = _make_estimator()
        estimator.estimate(_make_frame(), camera_id=0)
        assert estimator._crop_boxes

        estimator.reset_tracking()

        assert estimator._crop_boxes == {}

    def test_estimate_keeps_one_crop_per_camera_id(self) -> None:
        estimator, _ = _make_estimator()
        small_frame = np.zeros((FRAME_HEIGHT // 2, FRAME_WIDTH // 2, 3), dtype=np.uint8)

        estimator.estimate(_make_frame(), camera_id="0")
        estimator.estimate(small_frame, camera_id="1")

        assert set(estimator._crop_boxes) == {"0", "1"}
        assert estimator._crop_boxes["0"][3] > estimator._crop_boxes["1"][3]

    def test_batch_keeps_one_crop_per_list_position(self) -> None:
        estimator, _ = _make_estimator(batch_size=3)
        frames = [_make_frame(seed) for seed in range(3)]

        estimator.estimate_batch(frames)

        assert set(estimator._crop_boxes) == {0, 1, 2}

    def test_batch_tracks_crops_by_camera_ids_when_views_are_missing(self) -> None:
        estimator, fake = _make_estimator(batch_size=3)
        frames = [_make_frame(seed) for seed in range(3)]
        estimator.estimate_batch(frames, camera_ids=["0", "1", "2"])
        crop_box_2 = estimator._crop_boxes["2"].copy()

        estimator.estimate_batch([frames[0], frames[2]], camera_ids=["0", "2"])

        assert set(estimator._crop_boxes) == {"0", "1", "2"}
        np.testing.assert_array_equal(fake.received_batches[-1][1], _preprocess(frames[2], crop_box_2)[0])

    def test_batch_rejects_mismatched_or_duplicate_camera_ids(self) -> None:
        estimator, _ = _make_estimator(batch_size=3)
        frames = [_make_frame(seed) for seed in range(2)]

        with pytest.raises(ValueError):
            estimator.estimate_batch(frames, camera_ids=["0"])
        with pytest.raises(ValueError):
            estimator.estimate_batch(frames, camera_ids=["0", "0"])

    def test_batch_shape_constant_through_bootstrap_and_tracking(self) -> None:
        estimator, fake = _make_estimator(batch_size=3)
        frames = [_make_frame(seed) for seed in range(2)]

        estimator.estimate_batch(frames)
        estimator.estimate_batch(frames)
        estimator.estimate(frames[0], camera_id=7)

        assert fake.received_shapes == [(3, *BATCH_SHAPE_TAIL)] * 5

    def test_only_untracked_views_are_rerun(self) -> None:
        estimator, fake = _make_estimator(batch_size=3)
        frames = [_make_frame(seed) for seed in range(3)]
        estimator.estimate_batch(frames[:2])
        rerun_count_before = len(fake.received_batches)

        estimator.estimate_batch(frames)

        bootstrap_crop = _crop_box_from_points(
            _expected_frame_coords(_full_frame_box(frames[2]), fake.x_bins, fake.y_bins)
        )
        assert len(fake.received_batches) == rerun_count_before + 2
        np.testing.assert_array_equal(fake.received_batches[-1][0], _preprocess(frames[2], bootstrap_crop)[0])

    def test_release_clears_crops(self) -> None:
        estimator, _ = _make_estimator()
        estimator.estimate(_make_frame(), camera_id=0)

        estimator.release()

        assert estimator._crop_boxes == {}


@pytest.mark.skipif(not ONNXRUNTIME_AVAILABLE, reason="onnxruntime not installed")
class TestSkeletonLayout:

    def test_halpe26_skeleton_has_21_keypoints_with_toes_and_heels(self) -> None:
        estimator, fake = _make_estimator()
        frame = _make_frame()

        skeleton = estimator.estimate(frame)

        crop_box = _crop_box_from_points(_expected_frame_coords(_full_frame_box(frame), fake.x_bins, fake.y_bins))
        raw_coords = _expected_frame_coords(crop_box, fake.x_bins, fake.y_bins)
        assert len(skeleton.keypoints) == NUM_SKELETON_KEYPOINTS
        for skeleton_index, raw_index in [
            (CK.LEFT_KNEE, CK.LEFT_KNEE),
            (CK.LEFT_FOOT_INDEX, HALPE26_LEFT_BIG_TOE),
            (CK.RIGHT_FOOT_INDEX, HALPE26_RIGHT_BIG_TOE),
            (CK.LEFT_HEEL, HALPE26_LEFT_HEEL),
            (CK.RIGHT_HEEL, HALPE26_RIGHT_HEEL),
        ]:
            keypoint = skeleton.keypoints[skeleton_index]
            assert [keypoint.x, keypoint.y] == pytest.approx(raw_coords[raw_index], abs=COORD_TOL_PX)
            assert keypoint.confidence == pytest.approx(CONFIDENT_LOGIT, abs=SCORE_TOL)

    def test_coco17_pads_foot_keypoints_with_zero_confidence(self) -> None:
        estimator, _ = _make_estimator(keypoint_format="coco17")

        skeleton = estimator.estimate(_make_frame())

        assert len(skeleton.keypoints) == NUM_SKELETON_KEYPOINTS
        for index in range(NUM_COCO17_KEYPOINTS, NUM_SKELETON_KEYPOINTS):
            assert skeleton.keypoints[index].confidence == 0.0
        assert skeleton.keypoints[CK.RIGHT_ANKLE].confidence == pytest.approx(CONFIDENT_LOGIT, abs=SCORE_TOL)

    def test_below_threshold_keypoints_are_zeroed(self) -> None:
        session = _FakeSession(confident_keypoints=CK.LEFT_KNEE)
        estimator, _ = _make_estimator(session=session)

        skeleton = estimator.estimate(_make_frame())

        assert skeleton.keypoints[CK.LEFT_HIP].confidence == pytest.approx(CONFIDENT_LOGIT, abs=SCORE_TOL)
        knee = skeleton.keypoints[CK.LEFT_KNEE]
        assert (knee.x, knee.y, knee.confidence) == (0.0, 0.0, 0.0)

    def test_batch_returns_none_for_view_without_person(self) -> None:
        # The fake kills a batch row, so the dead view goes last: the re-run of views 0-1 keeps rows 0-1.
        session = _FakeSession(dead_rows=frozenset({2}))
        estimator, _ = _make_estimator(batch_size=3, session=session)

        skeletons = estimator.estimate_batch([_make_frame(seed) for seed in range(3)])

        assert skeletons[0] is not None
        assert skeletons[1] is not None
        assert skeletons[2] is None
        assert set(estimator._crop_boxes) == {0, 1}


@pytest.mark.skipif(not ONNXRUNTIME_AVAILABLE, reason="onnxruntime not installed")
class TestEstimateBatch:

    def test_too_many_frames_raises(self) -> None:
        estimator, _ = _make_estimator(batch_size=2)

        with pytest.raises(ValueError):
            estimator.estimate_batch([_make_frame() for _ in range(3)])

    def test_empty_batch_returns_empty(self) -> None:
        estimator, fake = _make_estimator(batch_size=3)

        assert estimator.estimate_batch([]) == []
        assert fake.received_batches == []

    def test_batch_shares_frame_index(self) -> None:
        estimator, _ = _make_estimator(batch_size=3)
        frames = [_make_frame(seed) for seed in range(3)]

        first = estimator.estimate_batch(frames)
        second = estimator.estimate_batch(frames)

        assert len({s.frame_index for s in first}) == 1
        assert second[0].frame_index == first[0].frame_index + 1

    def test_invalid_batch_size_raises(self) -> None:
        with pytest.raises(ValueError):
            RTMPoseEstimator(batch_size=0)


@pytest.fixture(scope="module")
def real_halpe26_estimator() -> RTMPoseEstimator:
    if not ONNXRUNTIME_AVAILABLE or not REAL_MODEL_PATH.exists():
        pytest.skip("halpe26 ONNX model or onnxruntime not available")
    import onnxruntime as ort

    estimator = RTMPoseEstimator(
        confidence_threshold=CONFIDENCE_THRESHOLD,
        model_path=str(REAL_MODEL_PATH),
        keypoint_format="halpe26",
    )
    estimator._session = ort.InferenceSession(str(REAL_MODEL_PATH), providers=["CPUExecutionProvider"])
    estimator._input_name = estimator._session.get_inputs()[0].name
    estimator._initialized = True
    return estimator


def _squat_canvas() -> np.ndarray:
    capture = cv2.VideoCapture(str(SQUAT_VIDEO_PATH))
    ok, portrait = capture.read()
    capture.release()
    if not ok:
        pytest.skip("data/squats.mov could not be read")
    small = cv2.resize(portrait, (PORTRAIT_CANVAS_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
    left = (FRAME_WIDTH - PORTRAIT_CANVAS_WIDTH) // 2
    right = FRAME_WIDTH - PORTRAIT_CANVAS_WIDTH - left
    return cv2.copyMakeBorder(small, 0, 0, left, right, cv2.BORDER_REPLICATE)


class TestRealModel:

    def test_black_image_scores_below_threshold(self, real_halpe26_estimator: RTMPoseEstimator) -> None:
        estimator = real_halpe26_estimator
        estimator.reset_tracking()
        black = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

        raw_keypoints = estimator._estimate_views([black], [0])[0]

        assert raw_keypoints.shape == (NUM_HALPE26_KEYPOINTS, 3)
        assert np.all(raw_keypoints[:, 2] < CONFIDENCE_THRESHOLD)
        assert estimator.estimate(black, camera_id=0) is None
        assert estimator._crop_boxes == {}

    def test_noise_image_returns_none(self, real_halpe26_estimator: RTMPoseEstimator) -> None:
        estimator = real_halpe26_estimator
        estimator.reset_tracking()

        assert estimator.estimate(_make_frame(), camera_id=0) is None

    def test_person_frame_tracks_with_unit_range_scores(self, real_halpe26_estimator: RTMPoseEstimator) -> None:
        if not SQUAT_VIDEO_PATH.exists():
            pytest.skip("data/squats.mov not available")
        estimator = real_halpe26_estimator
        estimator.reset_tracking()
        canvas = _squat_canvas()

        skeleton = estimator.estimate(canvas, camera_id=0)
        tracked = estimator.estimate(canvas, camera_id=0)

        assert skeleton is not None and tracked is not None
        keypoints = tracked.to_numpy()
        assert keypoints.shape == (NUM_SKELETON_KEYPOINTS, 3)
        assert np.all((keypoints[:, 2] >= 0.0) & (keypoints[:, 2] <= 1.0))
        assert np.all(keypoints[:, 2] >= CONFIDENCE_THRESHOLD)
        crop_box = estimator._crop_boxes[0]
        assert crop_box[2] / crop_box[3] == pytest.approx(INPUT_ASPECT_RATIO, abs=COORD_TOL_PX)
        person_x = keypoints[:, 0]
        assert np.all((person_x > (FRAME_WIDTH - PORTRAIT_CANVAS_WIDTH) / 2) & (person_x < (FRAME_WIDTH + PORTRAIT_CANVAS_WIDTH) / 2))
