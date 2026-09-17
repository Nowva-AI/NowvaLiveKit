"""
Session-scoped body measurements from analysis keypoints. Streams rigid segment
lengths (hip width, femurs, tibias, ankle to big-toe feet) plus reported shoulder
width and torso sides into fixed 1 mm histograms and locks a noise-corrected
robust median once settled. Measures only: nothing here moves keypoints.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from pydantic import BaseModel

from biomechanics.utils.types import CocoKeypoints as CK

logger = logging.getLogger(__name__)

# C3 metric confidence: conf 0.4 ~ 2.4 cm position std on triangulated keypoints.
# 0.6 (1.6 cm) only passed 4-31 % of Kalman-lagged frames; 0.4 passes 64-94 %.
MIN_ENDPOINT_CONFIDENCE = 0.4

HISTOGRAM_BIN_M = 0.001
HISTOGRAM_MAX_LENGTH_M = 0.80
HISTOGRAM_BIN_COUNT = int(round(HISTOGRAM_MAX_LENGTH_M / HISTOGRAM_BIN_M))

# Strict completion
MIN_SAMPLES_PER_SEGMENT = 150
MIN_DISTINCT_REPS = 2
MAX_CI95_HALF_WIDTH_M = 0.005
MAX_BILATERAL_DIFFERENCE_M = 0.025
# Reported (non-rigid) segments only need enough samples for a usable median
MIN_ESTIMATE_SAMPLES = 30

# Fallback completion when the strict checks never all pass
FALLBACK_MIN_ACCEPTED_FRAMES = 600
FALLBACK_MIN_DISTINCT_REPS = 3

MAD_TO_SIGMA = 1.4826
MEDIAN_STANDARD_ERROR_FACTOR = math.sqrt(math.pi / 2.0)
Z_SCORE_95 = 1.96

REFERENCE_TIBIA_LENGTH_M = 0.45
# Keypoint-proportion femur/torso-side ratio (calibration SEGMENT_RATIOS 0.245 / 0.290)
REFERENCE_FEMUR_TO_TORSO_RATIO = 0.84
FORWARD_LEAN_SCALE_MIN = 0.8
FORWARD_LEAN_SCALE_MAX = 1.3
# Only used when no big-toe keypoint was ever measured (17-keypoint layouts)
DEFAULT_FOOT_LENGTH_M = 0.26

# name -> (proximal keypoint, distal keypoint, min plausible m, max plausible m)
SEGMENTS: dict[str, tuple[int, int, float, float]] = {
    "hip_width": (CK.LEFT_HIP, CK.RIGHT_HIP, 0.08, 0.45),
    "femur_l": (CK.LEFT_HIP, CK.LEFT_KNEE, 0.25, 0.65),
    "femur_r": (CK.RIGHT_HIP, CK.RIGHT_KNEE, 0.25, 0.65),
    "tibia_l": (CK.LEFT_KNEE, CK.LEFT_ANKLE, 0.22, 0.62),
    "tibia_r": (CK.RIGHT_KNEE, CK.RIGHT_ANKLE, 0.22, 0.62),
    "foot_l": (CK.LEFT_ANKLE, CK.LEFT_FOOT_INDEX, 0.08, 0.32),
    "foot_r": (CK.RIGHT_ANKLE, CK.RIGHT_FOOT_INDEX, 0.08, 0.32),
    "shoulder_width": (CK.LEFT_SHOULDER, CK.RIGHT_SHOULDER, 0.18, 0.60),
    "torso_l": (CK.LEFT_SHOULDER, CK.LEFT_HIP, 0.30, 0.78),
    "torso_r": (CK.RIGHT_SHOULDER, CK.RIGHT_HIP, 0.30, 0.78),
}
SEGMENT_NAMES: tuple[str, ...] = tuple(SEGMENTS)
RIGID_SEGMENTS: tuple[str, ...] = (
    "hip_width", "femur_l", "femur_r", "tibia_l", "tibia_r", "foot_l", "foot_r",
)
REPORTED_SEGMENTS: tuple[str, ...] = ("shoulder_width", "torso_l", "torso_r")
BILATERAL_PAIRS: tuple[tuple[str, str], ...] = (("femur_l", "femur_r"), ("tibia_l", "tibia_r"))
# Athlete-param quantities -> the segments averaged into each
PARAM_SEGMENTS: dict[str, tuple[str, ...]] = {
    "shoulder_width_m": ("shoulder_width",),
    "femur_avg_m": ("femur_l", "femur_r"),
    "torso_avg_m": ("torso_l", "torso_r"),
    "hip_width_m": ("hip_width",),
    "tibia_avg_m": ("tibia_l", "tibia_r"),
    "foot_avg_m": ("foot_l", "foot_r"),
}

_SEGMENT_INDEX = {name: i for i, name in enumerate(SEGMENT_NAMES)}
_PROXIMAL = np.array([SEGMENTS[name][0] for name in SEGMENT_NAMES])
_DISTAL = np.array([SEGMENTS[name][1] for name in SEGMENT_NAMES])
_MIN_LENGTH_M = np.array([SEGMENTS[name][2] for name in SEGMENT_NAMES])
_MAX_LENGTH_M = np.array([SEGMENTS[name][3] for name in SEGMENT_NAMES])
_SEGMENT_ROWS = np.arange(len(SEGMENT_NAMES))
_RIGID_ROWS = np.array([_SEGMENT_INDEX[name] for name in RIGID_SEGMENTS])
_RIGID_ROWS_WITHOUT_FEET = np.array([_SEGMENT_INDEX[name] for name in RIGID_SEGMENTS if not name.startswith("foot")])
_REPORTED_ROWS = np.array([_SEGMENT_INDEX[name] for name in REPORTED_SEGMENTS])
_BILATERAL_ROWS = np.array([[_SEGMENT_INDEX[left], _SEGMENT_INDEX[right]] for left, right in BILATERAL_PAIRS])
_BIN_INDICES = np.arange(HISTOGRAM_BIN_COUNT)


def _robust_lengths(histograms: np.ndarray, sample_counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = np.arange(len(histograms))
    half_counts = sample_counts / 2.0
    cumulative = np.cumsum(histograms, axis=1)
    median_bins = np.argmax(cumulative >= half_counts[:, None], axis=1)
    counts_below = np.where(median_bins > 0, cumulative[rows, median_bins - 1], 0)
    medians_m = (median_bins + (half_counts - counts_below) / histograms[rows, median_bins]) * HISTOGRAM_BIN_M

    # MAD: fold each histogram around its median bin, then take the median distance
    bin_distances = np.abs(_BIN_INDICES[None, :] - median_bins[:, None]) + (rows * HISTOGRAM_BIN_COUNT)[:, None]
    deviation_histograms = np.bincount(
        bin_distances.ravel(), weights=histograms.ravel(), minlength=histograms.size,
    ).reshape(histograms.shape)
    deviation_cumulative = np.cumsum(deviation_histograms, axis=1)
    mads_m = np.argmax(deviation_cumulative >= half_counts[:, None], axis=1) * HISTOGRAM_BIN_M
    sigmas_m = MAD_TO_SIGMA * mads_m

    # Keypoint noise perpendicular to a segment lengthens it by ~sigma^2 / L
    # (sigma = along-segment spread); invert m = L + sigma^2 / L.
    lengths_m = (medians_m + np.sqrt(np.maximum(medians_m ** 2 - 4.0 * sigmas_m ** 2, 0.0))) / 2.0
    ci95_half_widths_m = Z_SCORE_95 * MEDIAN_STANDARD_ERROR_FACTOR * sigmas_m / np.sqrt(sample_counts)
    return lengths_m, ci95_half_widths_m


def _params_from_lengths(lengths_m: np.ndarray) -> dict[str, float] | None:
    params: dict[str, float] = {}
    for key, segment_names in PARAM_SEGMENTS.items():
        side_lengths_m = lengths_m[[_SEGMENT_INDEX[name] for name in segment_names]]
        measured_m = side_lengths_m[np.isfinite(side_lengths_m)]
        if measured_m.size > 0:
            params[key] = float(measured_m.mean())
        elif key == "foot_avg_m":
            logger.warning("[BODY MEASUREMENT] No foot length measured; using default %.2f m", DEFAULT_FOOT_LENGTH_M)
            params[key] = DEFAULT_FOOT_LENGTH_M
        else:
            return None
    return params


def _proportions_from_params(params: dict[str, float]) -> BodyProportions:
    femur_m = params["femur_avg_m"]
    torso_m = params["torso_avg_m"]
    forward_lean_scale = float(np.clip(
        (femur_m / torso_m) / REFERENCE_FEMUR_TO_TORSO_RATIO,
        FORWARD_LEAN_SCALE_MIN, FORWARD_LEAN_SCALE_MAX,
    ))
    return BodyProportions(
        hip_width=params["hip_width_m"],
        femur_length_avg=femur_m,
        tibia_length_avg=params["tibia_avg_m"],
        torso_length_avg=torso_m,
        shoulder_width=params["shoulder_width_m"],
        foot_length_avg=params["foot_avg_m"],
        hip_to_femur_ratio=params["hip_width_m"] / femur_m,
        tibia_to_reference_ratio=params["tibia_avg_m"] / REFERENCE_TIBIA_LENGTH_M,
        forward_lean_scale=forward_lean_scale,
    )


class BodyProportions(BaseModel):
    hip_width: float
    femur_length_avg: float
    tibia_length_avg: float
    torso_length_avg: float
    shoulder_width: float
    foot_length_avg: float
    hip_to_femur_ratio: float
    tibia_to_reference_ratio: float
    # Longer femur relative to torso -> more trunk lean needed -> scale > 1
    forward_lean_scale: float


class SegmentLengthEstimator:
    """Robust once-per-session segment length measurement.

    Complete when every rigid segment in the keypoint layout has
    MIN_SAMPLES_PER_SEGMENT samples with a 95 % CI half-width under
    MAX_CI95_HALF_WIDTH_M, reported segments have MIN_ESTIMATE_SAMPLES, samples
    span MIN_DISTINCT_REPS rep counts, and left/right femur and tibia agree within
    MAX_BILATERAL_DIFFERENCE_M. If that never happens, completes after
    FALLBACK_MIN_ACCEPTED_FRAMES accepted frames over FALLBACK_MIN_DISTINCT_REPS
    rep counts with whatever segments have estimates, and logs a warning.
    The CI assumes independent samples.
    """

    def __init__(self, min_endpoint_confidence: float = MIN_ENDPOINT_CONFIDENCE) -> None:
        self._min_endpoint_confidence = min_endpoint_confidence
        self._histograms = np.zeros((len(SEGMENT_NAMES), HISTOGRAM_BIN_COUNT), dtype=np.int64)
        self._sample_counts = np.zeros(len(SEGMENT_NAMES), dtype=np.int64)
        self._accepted_frames = 0
        self._distinct_reps: set[int] = set()
        self._layout_keypoint_count = 0
        self._layout_rows = _SEGMENT_ROWS
        self._rigid_rows = _RIGID_ROWS
        self._athlete_params: dict[str, float] | None = None
        self._body_proportions: BodyProportions | None = None

    @property
    def is_complete(self) -> bool:
        return self._athlete_params is not None

    @property
    def progress(self) -> tuple[int, int]:
        if self.is_complete:
            return (MIN_SAMPLES_PER_SEGMENT, MIN_SAMPLES_PER_SEGMENT)
        fewest_samples = int(self._sample_counts[self._rigid_rows].min())
        return (min(fewest_samples, MIN_SAMPLES_PER_SEGMENT), MIN_SAMPLES_PER_SEGMENT)

    @property
    def body_proportions(self) -> BodyProportions | None:
        return self._body_proportions

    def record(self, points: np.ndarray, confidences: np.ndarray, rep_count: int) -> None:
        if self._athlete_params is not None:
            return
        if len(points) != self._layout_keypoint_count:
            self._set_layout(len(points))
        rows = self._layout_rows
        proximal = _PROXIMAL[rows]
        distal = _DISTAL[rows]

        lengths_m = np.linalg.norm(points[distal] - points[proximal], axis=1)
        accepted = (
            (np.minimum(confidences[proximal], confidences[distal]) >= self._min_endpoint_confidence)
            & (lengths_m >= _MIN_LENGTH_M[rows])
            & (lengths_m <= _MAX_LENGTH_M[rows])
        )
        if not accepted.any():
            return

        accepted_rows = rows[accepted]
        bins = np.minimum((lengths_m[accepted] / HISTOGRAM_BIN_M).astype(np.int64), HISTOGRAM_BIN_COUNT - 1)
        self._histograms[accepted_rows, bins] += 1
        self._sample_counts[accepted_rows] += 1
        self._accepted_frames += 1
        self._distinct_reps.add(int(rep_count))
        self._try_complete()

    def to_athlete_params(self) -> dict | None:
        if self._athlete_params is None:
            return None
        return dict(self._athlete_params)

    @classmethod
    def from_athlete_params(cls, params: dict) -> SegmentLengthEstimator:
        lengths: dict[str, float] = {}
        for key in PARAM_SEGMENTS:
            value = params.get(key)
            if value is None and key == "foot_avg_m":
                # Rows written before feet were measured carry no foot length.
                value = DEFAULT_FOOT_LENGTH_M
            if value is None or not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"athlete_params[{key!r}] must be a positive length, got {value!r}")
            lengths[key] = float(value)
        estimator = cls()
        estimator._finish(lengths)
        return estimator

    def _set_layout(self, keypoint_count: int) -> None:
        in_layout = np.maximum(_PROXIMAL, _DISTAL) < keypoint_count
        self._layout_keypoint_count = keypoint_count
        self._layout_rows = _SEGMENT_ROWS[in_layout]
        has_feet = in_layout[_SEGMENT_INDEX["foot_l"]] and in_layout[_SEGMENT_INDEX["foot_r"]]
        self._rigid_rows = _RIGID_ROWS if has_feet else _RIGID_ROWS_WITHOUT_FEET

    def _try_complete(self) -> None:
        counts = self._sample_counts
        reps_seen = len(self._distinct_reps)
        strict_ready = (
            counts[self._rigid_rows].min() >= MIN_SAMPLES_PER_SEGMENT
            and counts[_REPORTED_ROWS].min() >= MIN_ESTIMATE_SAMPLES
            and reps_seen >= MIN_DISTINCT_REPS
        )
        fallback_due = self._accepted_frames >= FALLBACK_MIN_ACCEPTED_FRAMES and reps_seen >= FALLBACK_MIN_DISTINCT_REPS
        if not (strict_ready or fallback_due):
            return

        estimated = counts >= MIN_ESTIMATE_SAMPLES
        lengths_m = np.full(len(SEGMENT_NAMES), np.nan)
        ci95_half_widths_m = np.full(len(SEGMENT_NAMES), np.inf)
        lengths_m[estimated], ci95_half_widths_m[estimated] = _robust_lengths(
            self._histograms[estimated], counts[estimated],
        )
        bilateral_differences_m = np.abs(lengths_m[_BILATERAL_ROWS[:, 0]] - lengths_m[_BILATERAL_ROWS[:, 1]])
        if (
            strict_ready
            and (ci95_half_widths_m[self._rigid_rows] < MAX_CI95_HALF_WIDTH_M).all()
            and (bilateral_differences_m < MAX_BILATERAL_DIFFERENCE_M).all()
        ):
            self._finish(_params_from_lengths(lengths_m))
            return
        if not fallback_due:
            return

        params = _params_from_lengths(lengths_m)
        if params is None:
            return
        logger.warning(
            "[BODY MEASUREMENT] Strict checks never passed; completing on fallback after %d frames / %d reps. "
            "samples=%s ci95_mm=%s femur/tibia L-R diff_cm=%s",
            self._accepted_frames, reps_seen,
            dict(zip(SEGMENT_NAMES, counts.tolist())),
            dict(zip(SEGMENT_NAMES, np.round(ci95_half_widths_m * 1000.0, 1).tolist())),
            np.round(bilateral_differences_m * 100.0, 1).tolist(),
        )
        self._finish(params)

    def _finish(self, params: dict[str, float]) -> None:
        self._athlete_params = params
        self._body_proportions = _proportions_from_params(params)
        logger.info(
            "[BODY MEASUREMENT] Complete: hip_w=%.3fm femur=%.3fm tibia=%.3fm torso=%.3fm "
            "shoulder_w=%.3fm foot=%.3fm fwd_lean_scale=%.2f",
            params["hip_width_m"], params["femur_avg_m"], params["tibia_avg_m"], params["torso_avg_m"],
            params["shoulder_width_m"], params["foot_avg_m"], self._body_proportions.forward_lean_scale,
        )
