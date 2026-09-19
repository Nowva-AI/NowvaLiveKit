"""
RTMPose-m (256x192) 2D pose estimation on ONNX Runtime with a tracked person crop.

Each camera view keeps a crop box built from its previous frame's confident keypoints;
views without a box run on the full frame and are re-run on their new crop in the same
call. Keypoints are decoded with sub-pixel parabola refinement and scored with the raw
SimCC maximum clipped to [0, 1]. 3D comes from the triangulation layer.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import onnxruntime as ort

    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False

from biomechanics.pose.base import PoseEstimator
from biomechanics.utils.types import CocoKeypoints as CK
from biomechanics.utils.types import Keypoint2D, Skeleton2D, Skeleton3D

logger = logging.getLogger(__name__)

# Default model location
DEFAULT_MODEL_DIR = Path(__file__).parent / "models"
DEFAULT_MODEL_NAME = "rtmpose-m-256x192.onnx"
DEFAULT_HALPE26_MODEL_NAME = "rtmpose-m-halpe26-256x192.onnx"

NUM_COCO17_KEYPOINTS = 17
NUM_HALPE26_KEYPOINTS = 26
NUM_SKELETON_KEYPOINTS = 21  # COCO-17 + big toes + heels (C1 layout)

# Halpe26 indices 0-16 are identical to COCO-17; the foot keypoints map to 17-20.
HALPE26_LEFT_BIG_TOE = 20
HALPE26_RIGHT_BIG_TOE = 21
HALPE26_LEFT_HEEL = 24
HALPE26_RIGHT_HEEL = 25

HALPE26_TO_SKELETON_INDEX = np.arange(NUM_SKELETON_KEYPOINTS)
HALPE26_TO_SKELETON_INDEX[CK.LEFT_FOOT_INDEX] = HALPE26_LEFT_BIG_TOE
HALPE26_TO_SKELETON_INDEX[CK.RIGHT_FOOT_INDEX] = HALPE26_RIGHT_BIG_TOE
HALPE26_TO_SKELETON_INDEX[CK.LEFT_HEEL] = HALPE26_LEFT_HEEL
HALPE26_TO_SKELETON_INDEX[CK.RIGHT_HEEL] = HALPE26_RIGHT_HEEL

# Model input resolution
INPUT_WIDTH = 192
INPUT_HEIGHT = 256
INPUT_ASPECT_RATIO = INPUT_WIDTH / INPUT_HEIGHT

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Fused normalization: (x/255 - mean)/std == x * SCALE - OFFSET.
# Shaped (3, 1, 1) so the broadcast runs over contiguous CHW channel
# planes — broadcasting over the trailing HWC axis is ~7x slower.
IMAGENET_SCALE_CHW = (1.0 / (255.0 * IMAGENET_STD)).astype(np.float32).reshape(3, 1, 1)
IMAGENET_OFFSET_CHW = (IMAGENET_MEAN / IMAGENET_STD).astype(np.float32).reshape(3, 1, 1)

# SimCC split ratio (standard for RTMPose): SimCC bins per input pixel
SIMCC_SPLIT_RATIO = 2.0

# Raw SimCC maxima exceed 1.0 on clear keypoints; scores are clipped to this.
MAX_KEYPOINT_SCORE = 1.0
# A parabola peak needs strictly negative curvature to be a maximum.
MIN_PEAK_CURVATURE = 1e-6
MAX_SUBPIXEL_OFFSET_BINS = 0.5

# Person crop (mmpose/rtmlib top-down convention: bbox x 1.25, model aspect).
CROP_PADDING = 1.25
# Views with fewer confident raw keypoints re-detect on the full frame next frame.
MIN_TRACKED_KEYPOINTS = 8
# Floor on the crop height so a degenerate keypoint cluster cannot collapse the box.
MIN_CROP_HEIGHT_PX = 64.0
# Crop box layout: [center_x, center_y, width, height] in frame pixels.
BOX_SIZE = 4


def _full_frame_box(frame: np.ndarray) -> np.ndarray:
    frame_height, frame_width = frame.shape[:2]
    return np.array(
        [frame_width / 2.0, frame_height / 2.0, float(frame_width), float(frame_height)]
    )


def _crop_box_from_points(points_xy: np.ndarray) -> np.ndarray:
    x_min, y_min = points_xy.min(axis=0)
    x_max, y_max = points_xy.max(axis=0)
    box_height = max(y_max - y_min, (x_max - x_min) / INPUT_ASPECT_RATIO, MIN_CROP_HEIGHT_PX)
    box_height *= CROP_PADDING
    return np.array(
        [(x_min + x_max) / 2.0, (y_min + y_max) / 2.0, box_height * INPUT_ASPECT_RATIO, box_height]
    )


def _warp_matrix(box: np.ndarray) -> np.ndarray:
    scale_x = INPUT_WIDTH / box[2]
    scale_y = INPUT_HEIGHT / box[3]
    return np.array(
        [
            [scale_x, 0.0, INPUT_WIDTH / 2.0 - box[0] * scale_x],
            [0.0, scale_y, INPUT_HEIGHT / 2.0 - box[1] * scale_y],
        ],
        dtype=np.float64,
    )


def _preprocess(frame: np.ndarray, box: np.ndarray) -> np.ndarray:
    # Axis-aligned affine warp of the box to 192x256; outside the image is zero.
    # BGR->RGB is folded into the CHW transpose while still uint8.
    warped = cv2.warpAffine(
        frame,
        _warp_matrix(box),
        (INPUT_WIDTH, INPUT_HEIGHT),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    chw = np.ascontiguousarray(warped.transpose(2, 0, 1)[::-1])
    blob = chw.astype(np.float32)
    blob *= IMAGENET_SCALE_CHW
    blob -= IMAGENET_OFFSET_CHW
    return blob[np.newaxis]


def _parabola_peak(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # (..., num_bins) -> sub-bin peak location and peak value, both (...).
    peak_index = np.argmax(logits, axis=-1)
    num_bins = logits.shape[-1]
    inner_index = np.clip(peak_index, 1, num_bins - 2)
    center = np.take_along_axis(logits, inner_index[..., None], axis=-1)[..., 0].astype(np.float64)
    left = np.take_along_axis(logits, inner_index[..., None] - 1, axis=-1)[..., 0].astype(np.float64)
    right = np.take_along_axis(logits, inner_index[..., None] + 1, axis=-1)[..., 0].astype(np.float64)

    curvature = left - 2.0 * center + right
    is_peak = (curvature < -MIN_PEAK_CURVATURE) & (inner_index == peak_index)
    safe_curvature = np.where(is_peak, curvature, -1.0)
    offset = np.where(is_peak, 0.5 * (left - right) / safe_curvature, 0.0)
    offset = np.clip(offset, -MAX_SUBPIXEL_OFFSET_BINS, MAX_SUBPIXEL_OFFSET_BINS)

    peak_value = np.take_along_axis(logits, peak_index[..., None], axis=-1)[..., 0]
    return peak_index + offset, peak_value.astype(np.float64)


def decode_simcc(x_logits: np.ndarray, y_logits: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    Decode SimCC logits (B, K, bins) of crops taken at boxes (B, 4) to full-frame
    keypoints (B, K, 3) of [x_px, y_px, score], score = raw min(max_x, max_y) in [0, 1].
    """
    x_bins, x_peak = _parabola_peak(x_logits)
    y_bins, y_peak = _parabola_peak(y_logits)

    input_x = x_bins / SIMCC_SPLIT_RATIO
    input_y = y_bins / SIMCC_SPLIT_RATIO
    frame_x = (input_x / INPUT_WIDTH - 0.5) * boxes[:, 2:3] + boxes[:, 0:1]
    frame_y = (input_y / INPUT_HEIGHT - 0.5) * boxes[:, 3:4] + boxes[:, 1:2]

    scores = np.clip(np.minimum(x_peak, y_peak), 0.0, MAX_KEYPOINT_SCORE)
    return np.stack([frame_x, frame_y, scores], axis=-1)


class RTMPoseEstimator(PoseEstimator):
    """
    RTMPose 2D estimator (CoreML / CUDA / CPU execution providers).

    Skeleton2D output has 21 keypoints (COCO-17, big toes 17/18, heels 19/20); coco17
    models pad 17-20 with zero confidence. estimate() tracks crops per camera_id,
    estimate_batch() per camera_ids entry or list position. estimate_3d() returns None.
    """

    INPUT_WIDTH = INPUT_WIDTH
    INPUT_HEIGHT = INPUT_HEIGHT
    NUM_KEYPOINTS = NUM_SKELETON_KEYPOINTS

    def __init__(
        self,
        confidence_threshold: float = 0.3,
        model_path: str | None = None,
        keypoint_format: str = "coco17",
        batch_size: int = 1,
    ) -> None:
        super().__init__(confidence_threshold)

        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        if not ONNXRUNTIME_AVAILABLE:
            raise ImportError(
                "onnxruntime is not installed. Run: pip install onnxruntime"
            )

        self._keypoint_format = keypoint_format
        if keypoint_format == "halpe26":
            self._num_raw_keypoints = NUM_HALPE26_KEYPOINTS
            default_name = DEFAULT_HALPE26_MODEL_NAME
        else:
            self._num_raw_keypoints = NUM_COCO17_KEYPOINTS
            default_name = DEFAULT_MODEL_NAME

        if model_path is not None:
            self._model_path = Path(model_path)
        else:
            self._model_path = DEFAULT_MODEL_DIR / default_name

        self._batch_size = batch_size
        self._session: ort.InferenceSession | None = None
        self._input_name: str | None = None
        self._frame_index = 0
        self._crop_boxes: dict[int | str, np.ndarray] = {}

    def initialize(self) -> bool:
        """Load ONNX model and create inference session."""
        if self._initialized:
            return True

        if not self._model_path.exists():
            raise FileNotFoundError(
                f"RTMPose model not found at {self._model_path}. "
                f"Run: python scripts/download_models.py"
            )

        # Select execution providers: prefer CUDA > CoreML > CPU
        providers = []
        available = ort.get_available_providers()

        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        if "CoreMLExecutionProvider" in available:
            # MLProgram format runs ~30% faster than the legacy
            # NeuralNetwork format and matches CPU decode results exactly.
            providers.append((
                "CoreMLExecutionProvider",
                {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"},
            ))
        providers.append("CPUExecutionProvider")

        logger.info("Loading RTMPose from %s", self._model_path)
        logger.info("ONNX Runtime providers: %s", providers)

        self._session = ort.InferenceSession(
            str(self._model_path),
            providers=providers,
        )

        # Cache input name
        self._input_name = self._session.get_inputs()[0].name

        # Verify expected output count
        outputs = self._session.get_outputs()
        if len(outputs) < 2:
            raise RuntimeError(
                f"Expected at least 2 outputs (simcc_x, simcc_y), got {len(outputs)}"
            )

        active_providers = self._session.get_providers()
        logger.info("Active ONNX Runtime providers: %s", active_providers)

        self._initialized = True
        return True

    def release(self) -> None:
        """Release ONNX session and tracking state."""
        self._session = None
        self._input_name = None
        self._initialized = False
        self.reset_tracking()

    def reset_tracking(self) -> None:
        """Forget every view's crop box; the next frame of each view runs on the full frame."""
        self._crop_boxes.clear()

    def _run_padded(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Pad to a constant batch shape: CoreML compiles the model for the
        # first input shape it sees and fails on any other shape afterward.
        num_real = batch.shape[0]
        if num_real < self._batch_size:
            pad = np.repeat(batch[-1:], self._batch_size - num_real, axis=0)
            batch = np.concatenate([batch, pad], axis=0)

        outputs = self._session.run(None, {self._input_name: batch})
        x_logits, y_logits = outputs[0], outputs[1]
        return x_logits[:num_real], y_logits[:num_real]

    def _infer(self, frames: list[np.ndarray], boxes: np.ndarray) -> np.ndarray:
        blob = np.concatenate([_preprocess(frame, box) for frame, box in zip(frames, boxes)])
        x_logits, y_logits = self._run_padded(blob)
        return decode_simcc(x_logits, y_logits, boxes)

    def _update_crop_box(self, view_key: int | str, raw_keypoints: np.ndarray) -> None:
        confident = raw_keypoints[:, 2] >= self.confidence_threshold
        if np.count_nonzero(confident) < MIN_TRACKED_KEYPOINTS:
            self._crop_boxes.pop(view_key, None)
            return
        self._crop_boxes[view_key] = _crop_box_from_points(raw_keypoints[confident, :2])

    def _estimate_views(
        self, frames: list[np.ndarray], view_keys: list[int | str]
    ) -> np.ndarray:
        # One padded inference over every view; views that had no crop box and found a
        # person are re-run on their new crop so returned keypoints always come from a crop.
        boxes = np.empty((len(frames), BOX_SIZE), dtype=np.float64)
        bootstrapped = []
        for i, (frame, view_key) in enumerate(zip(frames, view_keys)):
            tracked_box = self._crop_boxes.get(view_key)
            if tracked_box is None:
                bootstrapped.append(i)
                boxes[i] = _full_frame_box(frame)
            else:
                boxes[i] = tracked_box

        raw_keypoints = self._infer(frames, boxes)
        for i, view_key in enumerate(view_keys):
            self._update_crop_box(view_key, raw_keypoints[i])

        refine = [i for i in bootstrapped if view_keys[i] in self._crop_boxes]
        if refine:
            refine_boxes = np.array([self._crop_boxes[view_keys[i]] for i in refine])
            refined = self._infer([frames[i] for i in refine], refine_boxes)
            for row, i in enumerate(refine):
                raw_keypoints[i] = refined[row]
                self._update_crop_box(view_keys[i], refined[row])

        return raw_keypoints

    def _kpts_to_skeleton2d(
        self, kpts: np.ndarray, timestamp: float
    ) -> Skeleton2D | None:
        # Raw (K, 3) keypoints -> 21-entry Skeleton2D; None if nothing passes the threshold.
        if self._keypoint_format == "halpe26":
            mapped = kpts[HALPE26_TO_SKELETON_INDEX]
        else:
            mapped = np.zeros((NUM_SKELETON_KEYPOINTS, 3), dtype=np.float64)
            mapped[:NUM_COCO17_KEYPOINTS] = kpts[:NUM_COCO17_KEYPOINTS]

        below_threshold = mapped[:, 2] < self.confidence_threshold
        if below_threshold.all():
            return None
        mapped[below_threshold] = 0.0

        keypoints = [
            Keypoint2D(x=x, y=y, confidence=confidence)
            for x, y, confidence in mapped.tolist()
        ]
        return Skeleton2D(
            keypoints=keypoints,
            timestamp=timestamp,
            frame_index=self._frame_index,
        )

    def estimate(self, frame: np.ndarray, camera_id: int | str = 0) -> Skeleton2D | None:
        """
        Estimate the 2D pose in one frame, tracking the crop under camera_id.
        Returns None if no keypoint passes the confidence threshold.
        """
        if not self._initialized:
            if not self.initialize():
                return None

        kpts = self._estimate_views([frame], [camera_id])[0]

        timestamp = time.time()
        self._frame_index += 1

        return self._kpts_to_skeleton2d(kpts, timestamp)

    def estimate_batch(
        self,
        frames: list[np.ndarray],
        camera_ids: list[int | str] | None = None,
    ) -> list[Skeleton2D | None]:
        """
        Estimate 2D poses for several camera views in one inference call.

        len(frames) must not exceed batch_size; the batch is padded to batch_size so
        CoreML sees a constant shape. Crops are tracked per camera_ids entry, or by list
        position when camera_ids is None (then each camera must keep its position).
        All skeletons share timestamp and frame_index.
        """
        if len(frames) > self._batch_size:
            raise ValueError(
                f"Got {len(frames)} frames but batch_size is {self._batch_size}. "
                f"Construct RTMPoseEstimator with batch_size >= the number of cameras."
            )
        if camera_ids is None:
            camera_ids = list(range(len(frames)))
        elif len(camera_ids) != len(frames) or len(set(camera_ids)) != len(camera_ids):
            raise ValueError(
                f"camera_ids must be {len(frames)} unique ids, one per frame; got {camera_ids}"
            )
        if not frames:
            return []

        if not self._initialized:
            if not self.initialize():
                return [None] * len(frames)

        raw_keypoints = self._estimate_views(frames, camera_ids)

        timestamp = time.time()
        skeletons = [self._kpts_to_skeleton2d(kpts, timestamp) for kpts in raw_keypoints]
        self._frame_index += 1

        return skeletons

    def estimate_3d(self, frame: np.ndarray) -> Skeleton3D | None:
        """RTMPose is 2D-only; 3D comes from the triangulation layer."""
        return None

    def estimate_both(
        self, frame: np.ndarray
    ) -> tuple[Skeleton2D | None, Skeleton3D | None]:
        """Estimate the 2D pose (camera slot 0); 3D is always None for RTMPose."""
        skeleton_2d = self.estimate(frame)
        return skeleton_2d, None
