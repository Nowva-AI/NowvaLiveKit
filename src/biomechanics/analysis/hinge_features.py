"""
Barbell Romanian deadlift rep analyser (docs/exercises/rdl.md §3.2-§4.2).

Every judged feature is a height ratio or a timing along the athlete's own
vertical: hip sink, hand travel toward the knees, torso foreshortening (hinge),
the 5-95% shoulder-drop time and the hand-height tilt. Heights are cm above the
ankle midpoint along `up`; a certified single-camera mount is levelled by its
measured pitch first. Range is only ever judged against the athlete's own setups.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel

from biomechanics.analysis.rep_features import (
    CONCENTRIC_END_FRACTION,
    MIN_FOOT_CONFIDENCE,
    MIN_KEYPOINT_CONFIDENCE,
)
from biomechanics.config import RdlCameraConfig, RdlConfig, ReadinessGateConfig
from biomechanics.diagnosis.lean_model import MIDFOOT_FRACTION_OF_ANKLE_TO_TOE
from biomechanics.faults.observability import TRIANGULATED
from biomechanics.utils.camera_mount import PitchCertification, PitchDeltaTripwire
from biomechanics.utils.frame_context import GRAVITY_CAMERA_PITCH, MOUNT_RACK, FrameContext
from biomechanics.utils.geometry import WORLD_UP, joint_angle_3_points, level_camera_frame, sagittal_forward
from biomechanics.utils.types import CocoKeypoints as CK
from biomechanics.utils.types import FaultSeverity, Skeleton3D

EXERCISE_NAME = "romanian_deadlift"
HIP_SOURCE_MEASURED = "measured"
HIP_SOURCE_SETUP = "setup"
HIGHLIGHT_MEANING = "their most hips-back clean rep"
REJECT_PARTIAL = "partial"
REJECT_IGNORE = "ignore"

# Bottom window: shoulder samples within this of the lowest (~3° of hinge at depth).
BOTTOM_WINDOW_CM = 3.0
# Single camera: raw MediaPipe hip visibility needed to trust a bottom hip height.
SINGLE_CAMERA_MIN_HIP_CONFIDENCE = 0.5
PRE_ROLL_S = 0.5
DESCENT_START_FRACTION = 0.05
DESCENT_END_FRACTION = 0.95
CROSSING_MEDIAN_SAMPLES = 3
# A setup is the idle frames within this of the interval's highest hip.
SETUP_HIP_BAND_CM = 1.0
MAX_SETUPS = 5
LOWER_HALF_FRACTION = 0.5
MIN_WINDOW_FRAMES = 2
START_TILT_SAMPLES = 3
MIN_WRIST_SEPARATION_CM = 20.0
MIN_SEPARATION_FRACTION = 0.6
MAX_BAR_OFFSET_RATIO = 1.5
OFFSET_PERCENTILE = 80.0
MIN_FOOT_LENGTH_M = 0.05
CM_PER_M = 100.0
# The rep signal (like the squat's) is unknown when any of these is untracked.
MIN_SIGNAL_CONFIDENCE = 0.0
SIGNAL_KEYPOINTS = (CK.LEFT_SHOULDER, CK.RIGHT_SHOULDER, CK.LEFT_ANKLE, CK.RIGHT_ANKLE)
# Timestamps carry rounding error; a sample this close to the pre-roll edge is in it.
TIME_TOLERANCE_S = 1e-6
# Buffered samples are bounded (~20 s at 30 fps): a lifter who stands or holds
# a rep cannot grow them on an edge device.
MAX_IDLE_SAMPLES = 600
MAX_REP_SAMPLES = 600
# Bottom-window bar height over standing knee height, highest first; the bins
# absorb the bar sitting ~0.2 knee heights below the wrists.
RANGE_LABEL_BINS = (
    (1.35, "above the knee"),
    (1.05, "at the knee"),
    (0.8, "just below the knee"),
    (0.55, "mid-shin"),
)
LOWEST_RANGE_LABEL = "low on the shin"
PHASE_DESCENT = "descent"
PHASE_BOTTOM = "bottom"
PHASE_ASCENT = "ascent"
SIDE_LEFT = "left"
SIDE_RIGHT = "right"
DEGREES_UPRIGHT = 180.0


@dataclass(slots=True)
class HingeFrameSample:
    """One analysed frame; heights in cm above the ankle midpoint along `up`."""
    timestamp: float
    shoulder_cm: float
    hip_cm: float
    knee_cm: float
    wrist_l_cm: float
    wrist_r_cm: float
    wrist_lateral_sep_cm: float
    # Positive: the right hand is lower.
    bar_tilt_signed_deg: float
    # Wrist midpoint ahead of midfoot, in ankle-to-toe lengths.
    bar_offset_ratio: float
    # Hip -> shoulder tilt from up; positive leaning forward.
    trunk_pitch_signed_deg: float
    hip_conf: float
    knee_flexion_mean: float
    trunk_pitch: float
    # Camera-frame ankle -> hip tilt (single-camera delta tripwire); positive
    # hips further from the camera, the sign of the standing-pitch estimate.
    ankle_hip_tilt_deg: float
    # An idle frame the readiness gate's knee and trunk limits would pass.
    standing: bool

    @property
    def wrist_mid_cm(self) -> float:
        return (self.wrist_l_cm + self.wrist_r_cm) / 2.0


class HingeRepFeatures(BaseModel):
    """Per-rep RDL features (rdl.md §4.2). The thresholds the rejection path
    needs are stamped in, so the rejected-rep faults read the features alone."""
    exercise: str = EXERCISE_NAME
    rep_number: int
    max_depth_angle: float = math.nan
    capture_mode: str = ""
    gravity_source: str = ""
    started_at_turnaround: bool = False
    hinge_hip_source: str = HIP_SOURCE_SETUP
    hip_sink_ratio: float = math.nan
    bar_travel_ratio: float = math.nan
    hinge_deg: float = math.nan
    bar_height_ratio: float = math.nan
    range_label: str | None = None
    bar_tilt_deg: float = math.nan
    bar_tilt_side: str | None = None
    bar_offset_ratio_p80: float = math.nan
    bar_offset_peak_phase: str | None = None
    lockout_pitch_deg: float = math.nan
    shoulder_drop_cm: float = math.nan
    descent_time_s: float = math.nan
    ascent_time_s: float = math.nan
    duration_s: float = math.nan
    # RepData timings (the eccentric is the 5-95% descent).
    eccentric_time_s: float = math.nan
    concentric_time_s: float = math.nan
    concentric_velocity_mps: float = math.nan
    knee_flexion_bottom_deg: float = math.nan
    trunk_pitch_bottom_3d: float = math.nan
    # Filled by HingeSessionReference.annotate before the rules judge.
    hinge_reference_deg: float = math.nan
    highlight_score: float = math.nan
    highlight_meaning: str = HIGHLIGHT_MEANING
    # Uncertified single camera reading beyond the gross pitch tripwire this session.
    camera_pitch_untrusted: bool = False
    # Start tier of rdl_hip_sink at the rep's last frame (FaultSeverity value).
    hip_sink_start_tier: str = FaultSeverity.MILD.value
    hip_sink_bands: tuple[float, float, float]
    count_floor_travel: float
    partial_min_travel: float
    partial_min_duration_s: float
    fallback_floor_deg: float
    partial_bands: tuple[float, float]


@dataclass(frozen=True)
class _Setup:
    hip_cm: float
    knee_cm: float
    wrist_cm: float
    shoulder_cm: float
    torso_cm: float
    trunk_pitch_signed_deg: float


def _nanmedian(values: Iterable[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.median(finite)) if finite else math.nan


def _ratio(numerator: float, denominator: float) -> float:
    if not (math.isfinite(numerator) and math.isfinite(denominator)) or denominator <= 0.0:
        return math.nan
    return numerator / denominator


def _unit(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        return None
    return vector / norm


def _horizontal(vector: np.ndarray, up: np.ndarray) -> np.ndarray:
    return vector - float(np.dot(vector, up)) * up


def _running_median(values: Sequence[float], window: int) -> list[float]:
    half = window // 2
    return [_nanmedian(values[max(0, i - half):i + half + 1]) for i in range(len(values))]


def _confidences(ctx: FrameContext) -> np.ndarray:
    if ctx.raw_confidences is not None:
        return ctx.raw_confidences
    return np.array([kp.confidence for kp in ctx.skeleton_3d.keypoints], dtype=np.float64)


def _vertical_frame(ctx: FrameContext) -> tuple[np.ndarray, np.ndarray]:
    """(points, unit up). The rig uses ctx.up as measured; a certified single
    camera is levelled by the measured mount angle; an uncertified one keeps
    its own vertical, never the unvalidated pitch-corrected estimate."""
    points = ctx.skeleton_3d.to_numpy()
    if ctx.capture_mode == TRIANGULATED:
        return points, np.asarray(ctx.up, dtype=np.float64) / float(np.linalg.norm(ctx.up))
    if ctx.pitch_certified and math.isfinite(ctx.mount_pitch_deg):
        return level_camera_frame(points, ctx.mount_pitch_deg), WORLD_UP
    if ctx.gravity_source == GRAVITY_CAMERA_PITCH:
        return points, WORLD_UP
    return points, np.asarray(ctx.up, dtype=np.float64) / float(np.linalg.norm(ctx.up))


def _frontal_knee_flexion_deg(points: np.ndarray) -> float:
    frontal = points.copy()
    frontal[:, 2] = 0.0
    return max(
        DEGREES_UPRIGHT - joint_angle_3_points(frontal[CK.LEFT_HIP], frontal[CK.LEFT_KNEE], frontal[CK.LEFT_ANKLE]),
        DEGREES_UPRIGHT - joint_angle_3_points(frontal[CK.RIGHT_HIP], frontal[CK.RIGHT_KNEE], frontal[CK.RIGHT_ANKLE]),
    )


def _camera_tilt_deg(vector: np.ndarray) -> float:
    # Y-down camera frame: up the body is -y, away from the lens +z.
    return math.degrees(math.atan2(float(vector[2]), float(-vector[1])))


def _feet(
    points: np.ndarray, confidences: np.ndarray, up: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    """(left foot, right foot, midfoot, ankle-to-toe length) on the ground plane."""
    if len(points) <= CK.RIGHT_FOOT_INDEX:
        return None
    if min(confidences[CK.LEFT_FOOT_INDEX], confidences[CK.RIGHT_FOOT_INDEX]) < MIN_FOOT_CONFIDENCE:
        return None
    foot_l = _horizontal(points[CK.LEFT_FOOT_INDEX] - points[CK.LEFT_ANKLE], up)
    foot_r = _horizontal(points[CK.RIGHT_FOOT_INDEX] - points[CK.RIGHT_ANKLE], up)
    foot_len = (float(np.linalg.norm(foot_l)) + float(np.linalg.norm(foot_r))) / 2.0
    if foot_len < MIN_FOOT_LENGTH_M:
        return None
    midfoot = (
        points[CK.LEFT_ANKLE] + MIDFOOT_FRACTION_OF_ANKLE_TO_TOE * foot_l
        + points[CK.RIGHT_ANKLE] + MIDFOOT_FRACTION_OF_ANKLE_TO_TOE * foot_r
    ) / 2.0
    return foot_l, foot_r, midfoot, foot_len


def shoulder_signal_cm(points: np.ndarray, confidences: np.ndarray, up: np.ndarray = WORLD_UP) -> float:
    """The RDL rep signal: minus the shoulder height above the ankles, in cm, so
    it rises through the descent. NaN when a shoulder or ankle is untracked."""
    if any(confidences[index] <= MIN_SIGNAL_CONFIDENCE for index in SIGNAL_KEYPOINTS):
        return math.nan
    shoulder_mid = (points[CK.LEFT_SHOULDER] + points[CK.RIGHT_SHOULDER]) / 2.0
    ankle_mid = (points[CK.LEFT_ANKLE] + points[CK.RIGHT_ANKLE]) / 2.0
    return -CM_PER_M * float(np.dot(shoulder_mid - ankle_mid, up))


def skeleton_signal_cm(skeleton_3d: Skeleton3D) -> float:
    """shoulder_signal_cm of a skeleton with its own confidences, in WORLD_UP."""
    confidences = np.array([kp.confidence for kp in skeleton_3d.keypoints], dtype=np.float64)
    return shoulder_signal_cm(skeleton_3d.to_numpy(), confidences)


def lens_checks_pass(ctx: FrameContext, camera: RdlCameraConfig) -> bool:
    """Lens 0.8-1.1 m high and at most 10 cm below this lifter's hips (once
    body proportions exist); an unknown height fails (rdl.md §3.2)."""
    height_m = ctx.mount_height_m
    if not math.isfinite(height_m) or not camera.lens_min_height_m <= height_m <= camera.lens_max_height_m:
        return False
    proportions = ctx.body_proportions
    if proportions is None:
        return True
    hip_floor_m = proportions.femur_length_avg + proportions.tibia_length_avg + camera.ankle_height_m
    return height_m >= hip_floor_m - camera.lens_max_below_hip_m


def hip_sink_start_tier(ctx: FrameContext, camera: RdlCameraConfig) -> FaultSeverity:
    """rdl_hip_sink start tier (rdl.md §3.2, §5.1). Single camera: mild on a
    certified mount (rack, or inclinometer within the certified pitch) whose lens
    checks pass, moderate when only the pitch is uncertified (or revoked),
    severe when the lens checks fail. The rig's vertical budget keeps mild."""
    if ctx.capture_mode == TRIANGULATED:
        return FaultSeverity.MILD
    if not lens_checks_pass(ctx, camera):
        return FaultSeverity.SEVERE
    certified = ctx.pitch_certified and (
        ctx.mount_source == MOUNT_RACK or abs(ctx.mount_pitch_deg) <= camera.certified_max_pitch_deg
    )
    return FaultSeverity.MILD if certified else FaultSeverity.MODERATE


def build_hinge_sample(ctx: FrameContext, readiness: ReadinessGateConfig, in_rep: bool) -> HingeFrameSample:
    """One frame's heights, hand line, bar offset and trunk pitch (rdl.md §4.1)."""
    points, up = _vertical_frame(ctx)
    confidences = _confidences(ctx)
    skeleton_confidences = np.array([kp.confidence for kp in ctx.skeleton_3d.keypoints], dtype=np.float64)
    ankle_mid = (points[CK.LEFT_ANKLE] + points[CK.RIGHT_ANKLE]) / 2.0
    hip_mid = (points[CK.LEFT_HIP] + points[CK.RIGHT_HIP]) / 2.0
    shoulder_mid = (points[CK.LEFT_SHOULDER] + points[CK.RIGHT_SHOULDER]) / 2.0
    knee_mid = (points[CK.LEFT_KNEE] + points[CK.RIGHT_KNEE]) / 2.0

    def height_cm(point: np.ndarray) -> float:
        return CM_PER_M * float(np.dot(point - ankle_mid, up))

    wrist_min_confidence = MIN_KEYPOINT_CONFIDENCE if ctx.capture_mode == TRIANGULATED else readiness.min_confidence
    wrists_seen = min(confidences[CK.LEFT_WRIST], confidences[CK.RIGHT_WRIST]) >= wrist_min_confidence
    wrist_l_cm = height_cm(points[CK.LEFT_WRIST]) if wrists_seen else math.nan
    wrist_r_cm = height_cm(points[CK.RIGHT_WRIST]) if wrists_seen else math.nan

    lateral = _unit(_horizontal(points[CK.RIGHT_HIP] - points[CK.LEFT_HIP], up))
    feet = _feet(points, confidences, up)
    forward = None
    if lateral is not None:
        forward = sagittal_forward(up, lateral)
        if feet is not None and float(np.dot(forward, feet[0] + feet[1])) < 0.0:
            forward = -forward

    separation_cm = tilt_deg = offset_ratio = math.nan
    if wrists_seen and lateral is not None:
        wrist_line = _horizontal(points[CK.RIGHT_WRIST] - points[CK.LEFT_WRIST], up)
        separation_cm = abs(float(np.dot(wrist_line, lateral))) * CM_PER_M
        tilt_deg = math.degrees(math.atan2(wrist_l_cm - wrist_r_cm, separation_cm))
    if wrists_seen and forward is not None and feet is not None:
        wrist_mid = (points[CK.LEFT_WRIST] + points[CK.RIGHT_WRIST]) / 2.0
        offset_ratio = float(np.dot(_horizontal(wrist_mid - feet[2], up), forward)) / feet[3]
        if abs(offset_ratio) > MAX_BAR_OFFSET_RATIO:
            offset_ratio = math.nan

    trunk = shoulder_mid - hip_mid
    trunk_pitch_signed = math.nan
    trunk_from_up_deg = math.nan
    trunk_unit = _unit(trunk)
    if forward is not None:
        trunk_pitch_signed = math.degrees(math.atan2(float(np.dot(trunk, forward)), float(np.dot(trunk, up))))
    if trunk_unit is not None:
        trunk_from_up_deg = math.degrees(math.acos(float(np.clip(np.dot(trunk_unit, up), -1.0, 1.0))))

    raw_points = ctx.skeleton_3d.to_numpy()
    raw_ankle_mid = (raw_points[CK.LEFT_ANKLE] + raw_points[CK.RIGHT_ANKLE]) / 2.0
    raw_hip_mid = (raw_points[CK.LEFT_HIP] + raw_points[CK.RIGHT_HIP]) / 2.0
    standing = (
        not in_rep
        and _frontal_knee_flexion_deg(raw_points) <= readiness.max_knee_flexion_deg
        and trunk_from_up_deg <= readiness.max_trunk_flexion_deg
    )

    angles = ctx.angles
    signal = shoulder_signal_cm(points, skeleton_confidences, up)
    return HingeFrameSample(
        timestamp=ctx.timestamp,
        shoulder_cm=-signal,
        hip_cm=height_cm(hip_mid),
        knee_cm=height_cm(knee_mid),
        wrist_l_cm=wrist_l_cm,
        wrist_r_cm=wrist_r_cm,
        wrist_lateral_sep_cm=separation_cm,
        bar_tilt_signed_deg=tilt_deg,
        bar_offset_ratio=offset_ratio,
        trunk_pitch_signed_deg=trunk_pitch_signed,
        hip_conf=float(min(confidences[CK.LEFT_HIP], confidences[CK.RIGHT_HIP])),
        knee_flexion_mean=angles.avg_knee_flexion if angles is not None else math.nan,
        trunk_pitch=DEGREES_UPRIGHT - angles.trunk_flexion if angles is not None else math.nan,
        ankle_hip_tilt_deg=_camera_tilt_deg(raw_hip_mid - raw_ankle_mid),
        standing=standing,
    )


def _close_setup(samples: Iterable[HingeFrameSample]) -> _Setup | None:
    samples = list(samples)
    hips = [sample.hip_cm for sample in samples if math.isfinite(sample.hip_cm)]
    if not hips:
        return None
    peak = max(hips)
    frames = [sample for sample in samples if sample.hip_cm >= peak - SETUP_HIP_BAND_CM]
    return _Setup(
        hip_cm=_nanmedian(sample.hip_cm for sample in frames),
        knee_cm=_nanmedian(sample.knee_cm for sample in frames),
        wrist_cm=_nanmedian(sample.wrist_mid_cm for sample in frames),
        shoulder_cm=_nanmedian(sample.shoulder_cm for sample in frames),
        torso_cm=_nanmedian(sample.shoulder_cm - sample.hip_cm for sample in frames),
        trunk_pitch_signed_deg=_nanmedian(sample.trunk_pitch_signed_deg for sample in frames),
    )


def _references(setups: Sequence[_Setup]) -> _Setup:
    return _Setup(
        hip_cm=_nanmedian(setup.hip_cm for setup in setups),
        knee_cm=_nanmedian(setup.knee_cm for setup in setups),
        wrist_cm=_nanmedian(setup.wrist_cm for setup in setups),
        shoulder_cm=_nanmedian(setup.shoulder_cm for setup in setups),
        torso_cm=_nanmedian(setup.torso_cm for setup in setups),
        trunk_pitch_signed_deg=_nanmedian(setup.trunk_pitch_signed_deg for setup in setups),
    )


def _lowest_index(smoothed: Sequence[float], start: int) -> int | None:
    # The lowest point is read on the 3-sample median: a single-sample minimum
    # is biased low by the noise, which moves the 95% crossing.
    finite = [i for i in range(start, len(smoothed)) if math.isfinite(smoothed[i])]
    if not finite:
        return None
    return min(finite, key=lambda i: smoothed[i])


def _interpolated_time(times: Sequence[float], index: int, before: float, after: float, level: float) -> float:
    fraction = (before - level) / (before - after)
    return times[index - 1] + fraction * (times[index] - times[index - 1])


def descent_crossings(
    times: Sequence[float], smoothed: Sequence[float], i_top: int, i_low: int, top_cm: float, drop_cm: float,
) -> tuple[float, float]:
    """Times of the downward 5% and 95% shoulder-drop crossings, searched only
    in (i_top, i_low] and linearly interpolated; NaN when not crossed."""
    if not (math.isfinite(top_cm) and drop_cm > 0.0):
        return math.nan, math.nan
    start_level = top_cm - DESCENT_START_FRACTION * drop_cm
    end_level = top_cm - DESCENT_END_FRACTION * drop_cm
    start_index = None
    start_time = math.nan
    for i in range(i_top + 1, i_low + 1):
        before, after = smoothed[i - 1], smoothed[i]
        if before >= start_level > after:
            start_index, start_time = i, _interpolated_time(times, i, before, after, start_level)
            break
    if start_index is None:
        return math.nan, math.nan
    for i in range(start_index, i_low + 1):
        before, after = smoothed[i - 1], smoothed[i]
        if before > end_level >= after:
            return start_time, _interpolated_time(times, i, before, after, end_level)
    return start_time, math.nan


def _range_label(bar_height_ratio: float) -> str | None:
    if not math.isfinite(bar_height_ratio):
        return None
    for lower_bound, label in RANGE_LABEL_BINS:
        if bar_height_ratio > lower_bound:
            return label
    return LOWEST_RANGE_LABEL


def _bar_tilt(
    samples: Sequence[HingeFrameSample], rep_start: int, bottom: Sequence[int],
) -> tuple[float, str | None]:
    # The standing pre-roll before the descent; a hand that drops gradually has
    # already dropped in the rep's first samples. A rep started at a turnaround
    # has no standing pre-roll: its first samples are the start.
    start = [sample for sample in samples[:rep_start] if sample.standing]
    if not start:
        start = samples[rep_start:rep_start + START_TILT_SAMPLES]
    start_tilt = _nanmedian(sample.bar_tilt_signed_deg for sample in start)
    start_separation = _nanmedian(sample.wrist_lateral_sep_cm for sample in start)
    bottom_tilt = _nanmedian(samples[i].bar_tilt_signed_deg for i in bottom)
    bottom_separation = _nanmedian(samples[i].wrist_lateral_sep_cm for i in bottom)
    if not all(math.isfinite(value) for value in (start_tilt, start_separation, bottom_tilt, bottom_separation)):
        return math.nan, None
    # Hands turned edge-on to the camera (yaw) or lost: the height difference means nothing.
    if bottom_separation < MIN_SEPARATION_FRACTION * start_separation or bottom_separation < MIN_WRIST_SEPARATION_CM:
        return math.nan, None
    delta = bottom_tilt - start_tilt
    return abs(delta), SIDE_RIGHT if delta > 0.0 else SIDE_LEFT


def _bar_offset(
    samples: Sequence[HingeFrameSample], lower_half: Sequence[int], bottom: Sequence[int], i_low: int,
) -> tuple[float, str | None]:
    offsets = [(samples[i].bar_offset_ratio, i) for i in lower_half if math.isfinite(samples[i].bar_offset_ratio)]
    if not offsets:
        return math.nan, None
    p80 = float(np.percentile([offset for offset, _ in offsets], OFFSET_PERCENTILE))
    peak_index = max(offsets)[1]
    if peak_index in bottom:
        return p80, PHASE_BOTTOM
    return p80, PHASE_DESCENT if peak_index < i_low else PHASE_ASCENT


def _concentric_velocity_mps(
    shoulders: Sequence[float], times: Sequence[float], i_low: int, bottom_cm: float,
) -> float:
    # Same method as rep_features' squat velocity, on the shoulder rise. The
    # counter ends a rep within standing_return of the top, which can be short
    # of the end height: then the rep's last sample ends the rise.
    before = [value for value in shoulders[:i_low] if math.isfinite(value)]
    after = [i for i in range(i_low + 1, len(shoulders)) if math.isfinite(shoulders[i])]
    if not before or not after:
        return math.nan
    top_cm = max(before)
    end_cm = top_cm - (top_cm - bottom_cm) * CONCENTRIC_END_FRACTION
    end_index = next((i for i in after if shoulders[i] >= end_cm), after[-1])
    elapsed_s = times[end_index] - times[i_low]
    return (shoulders[end_index] - bottom_cm) / CM_PER_M / elapsed_s if elapsed_s > 0.0 else math.nan


def _hinge_deg(shoulder_cm: float, hip_cm: float, torso_cm: float) -> float:
    cosine = _ratio(shoulder_cm - hip_cm, torso_cm)
    if not math.isfinite(cosine):
        return math.nan
    return math.degrees(math.acos(min(1.0, max(-1.0, cosine))))


class RdlRepAnalyzer:
    """RepAnalyzer for the barbell RDL (SHARED-PLUMBING.md §2, rdl.md §4.1).

    Survives reset_set: the standing setups (references for every rep) and the
    certified-mount tripwire's reference. A fresh analyser (set_exercise)
    starts the setups empty; the tripwire, once bound to the session's
    (bind_pitch_certification), keeps its reference and the sets pooled into
    it across switches. A rep that starts at the previous rep's turnaround has
    no setup of its own: its pre-roll is the previous rep's samples after its
    lowest point (plus any idle frames at the turnaround), and its lockout is
    not judged. That covers the counter's touch-and-go starts (piston) and a
    smooth turnaround the counter ended as settled at the top, when the
    shoulders never came back within the counter's standing return of the
    standing setups.
    """

    def __init__(self, rdl: RdlConfig, readiness: ReadinessGateConfig) -> None:
        self._rdl = rdl
        self._readiness = readiness
        self.rep_signal = math.nan
        self._setups: deque[_Setup] = deque(maxlen=MAX_SETUPS)
        self._idle: deque[HingeFrameSample] = deque(maxlen=MAX_IDLE_SAMPLES)
        self._rep: deque[HingeFrameSample] = deque(maxlen=MAX_REP_SAMPLES)
        self._pre_roll: list[HingeFrameSample] = []
        self._tail: list[HingeFrameSample] = []
        self._rep_setup: _Setup | None = None
        self._in_rep = False
        self._piston = False
        self._capture_mode = ""
        self._gravity_source = ""
        self._hip_sink_tier = FaultSeverity.MILD
        self._camera_pitch_untrusted = False
        # Certified single camera: the set's standing ankle->hip tilt against a
        # reference pooled over the first sets.
        self._tripwire = PitchDeltaTripwire(rdl.camera.delta_tripwire_deg)
        self._set_tilt_sum = 0.0
        self._set_tilt_count = 0
        self._set_checked = False
        self._pending_trip_deg: float | None = None

    @property
    def setup_count(self) -> int:
        return len(self._setups)

    @property
    def camera_pitch_untrusted(self) -> bool:
        """An uncertified single camera read beyond the gross pitch tripwire this session."""
        return self._camera_pitch_untrusted

    def is_ready(self, ctx: FrameContext) -> bool:
        """Both wrists seen and below the hips: the wrist features will exist.
        A visibility check only (arms at the sides pass too). Stateless."""
        points, up = _vertical_frame(ctx)
        confidences = _confidences(ctx)
        hip_mid = (points[CK.LEFT_HIP] + points[CK.RIGHT_HIP]) / 2.0
        for wrist in (CK.LEFT_WRIST, CK.RIGHT_WRIST):
            if confidences[wrist] < self._readiness.min_confidence:
                return False
            if float(np.dot(points[wrist] - hip_mid, up)) >= -self._rdl.readiness_wrist_below_hip_m:
                return False
        return True

    def observe(self, ctx: FrameContext) -> None:
        sample = build_hinge_sample(ctx, self._readiness, in_rep=self._in_rep)
        self.rep_signal = -sample.shoulder_cm
        self._capture_mode = ctx.capture_mode
        self._gravity_source = ctx.gravity_source
        self._hip_sink_tier = hip_sink_start_tier(ctx, self._rdl.camera)
        self._watch_camera_pitch(ctx, sample)
        if self._in_rep:
            self._rep.append(sample)
        else:
            self._idle.append(sample)

    def bind_pitch_certification(self, certification: PitchCertification, profile_name: str) -> None:
        self._tripwire = certification.tripwire(profile_name, self._rdl.camera.delta_tripwire_deg)

    def pitch_tripwire_delta(self) -> float | None:
        """The set-averaged ankle->hip tilt change that tripped the certified
        mount's delta tripwire, once; None otherwise."""
        delta_deg, self._pending_trip_deg = self._pending_trip_deg, None
        return delta_deg

    def on_rep_start(self, piston: bool) -> None:
        turnaround = piston or self._never_stood()
        if turnaround:
            self._pre_roll = self._tail + list(self._idle)
            self._rep_setup = None
        else:
            self._rep_setup = _close_setup(self._idle)
            last_time = self._idle[-1].timestamp if self._idle else math.nan
            self._pre_roll = [
                sample for sample in self._idle if sample.timestamp >= last_time - PRE_ROLL_S - TIME_TOLERANCE_S
            ]
        self._tail = []
        self._rep.clear()
        self._piston = turnaround
        self._in_rep = True

    def finish_rep(self, rep_number: int) -> HingeRepFeatures:
        setups = list(self._setups)
        if not self._piston and self._rep_setup is not None:
            setups.append(self._rep_setup)
        samples = self._pre_roll + list(self._rep)
        features = self._features(rep_number, samples, len(self._pre_roll), setups)
        self._tail = self._after_lowest(list(self._rep))
        if not self._piston and self._rep_setup is not None:
            self._setups.append(self._rep_setup)
        self._idle.clear()
        self._end_rep()
        return features

    def rejection_reason(self, features: HingeRepFeatures) -> str | None:
        """None for a counted rep, "partial" for a real attempt stopped above
        the hand-travel floor, "ignore" for the rest (unrack, re-rack, peeking).
        Pure: the pipeline owns the streak."""
        travel = features.bar_travel_ratio
        if math.isfinite(travel):
            if travel >= features.count_floor_travel:
                return None
            if travel >= features.partial_min_travel and features.duration_s >= features.partial_min_duration_s:
                return REJECT_PARTIAL
            return REJECT_IGNORE
        # Wrists unseen at the bottom: the hinge floor decides.
        if features.hinge_deg >= features.fallback_floor_deg:
            return None
        return REJECT_IGNORE

    def discard_rep(self) -> None:
        if not self._in_rep:
            return
        rep = list(self._rep)
        self._tail = self._after_lowest(rep)
        # A descent that was no rep (false start, abort) is idle time.
        self._idle.extend(rep)
        self._end_rep()

    def reset_set(self) -> None:
        camera = self._rdl.camera
        if not math.isfinite(self._tripwire.reference_deg) and self._set_tilt_count >= camera.delta_min_frames:
            self._tripwire.pool_reference(self._set_tilt_sum / self._set_tilt_count, camera.reference_sets)
        self._set_tilt_sum = 0.0
        self._set_tilt_count = 0
        self._set_checked = False
        self._idle.clear()
        self._tail = []
        self._end_rep()

    def _end_rep(self) -> None:
        self._rep.clear()
        self._pre_roll = []
        self._rep_setup = None
        self._in_rep = False
        self._piston = False

    def _standing_shoulder_floor_cm(self) -> float:
        """Shoulder height below which the lifter has not stood back up (NaN
        before the first setup): the counter's standing return under the setups."""
        if not self._setups:
            return math.nan
        return _references(self._setups).shoulder_cm - self._rdl.counter.standing_return_cm

    def _never_stood(self) -> bool:
        floor_cm = self._standing_shoulder_floor_cm()
        if not self._idle or not math.isfinite(floor_cm):
            return False
        peak_cm = max(
            (value for value in _running_median([sample.shoulder_cm for sample in self._idle], CROSSING_MEDIAN_SAMPLES)
             if math.isfinite(value)),
            default=math.nan,
        )
        return peak_cm < floor_cm

    @staticmethod
    def _after_lowest(rep: list[HingeFrameSample]) -> list[HingeFrameSample]:
        smoothed = _running_median([sample.shoulder_cm for sample in rep], CROSSING_MEDIAN_SAMPLES)
        i_low = _lowest_index(smoothed, 0)
        return [] if i_low is None else rep[i_low + 1:]

    def _watch_camera_pitch(self, ctx: FrameContext, sample: HingeFrameSample) -> None:
        camera = self._rdl.camera
        if (
            ctx.capture_mode != TRIANGULATED
            and not ctx.pitch_certified
            and math.isfinite(ctx.camera_pitch_deg)
            and abs(ctx.camera_pitch_deg - camera.posture_offset_deg) > camera.tripwire_deg
        ):
            self._camera_pitch_untrusted = True
        if not (ctx.pitch_certified and sample.standing and math.isfinite(sample.ankle_hip_tilt_deg)):
            return
        # Idle tops of a constant-tension set are not standing frames.
        if sample.shoulder_cm < self._standing_shoulder_floor_cm():
            return
        self._set_tilt_sum += sample.ankle_hip_tilt_deg
        self._set_tilt_count += 1
        if (
            self._set_checked
            or not math.isfinite(self._tripwire.reference_deg)
            or self._set_tilt_count < camera.delta_min_frames
        ):
            return
        # One look per set, once its average rests on enough standing frames.
        self._set_checked = True
        delta_deg = self._tripwire.check(self._set_tilt_sum / self._set_tilt_count)
        if delta_deg is not None:
            self._pending_trip_deg = delta_deg

    def _top(self, shoulders: Sequence[float], smoothed: Sequence[float], rep_start: int) -> tuple[int, float]:
        if rep_start == 0:
            return 0, smoothed[0]
        finite = [i for i in range(rep_start) if math.isfinite(smoothed[i])]
        if not finite:
            return rep_start - 1, math.nan
        i_top = max(finite, key=lambda i: smoothed[i])
        if self._piston:
            return i_top, smoothed[i_top]
        return i_top, _nanmedian(shoulders[:rep_start])

    def _features(
        self, rep_number: int, samples: list[HingeFrameSample], rep_start: int, setups: list[_Setup],
    ) -> HingeRepFeatures:
        rdl = self._rdl
        features = HingeRepFeatures(
            rep_number=rep_number,
            capture_mode=self._capture_mode,
            gravity_source=self._gravity_source,
            started_at_turnaround=self._piston,
            camera_pitch_untrusted=self._camera_pitch_untrusted,
            hip_sink_start_tier=self._hip_sink_tier.value,
            hip_sink_bands=(rdl.hip_sink.mild, rdl.hip_sink.moderate, rdl.hip_sink.severe),
            count_floor_travel=rdl.rom.count_floor_travel,
            partial_min_travel=rdl.rom.partial_min_travel,
            partial_min_duration_s=rdl.rom.partial_min_duration_s,
            fallback_floor_deg=rdl.rom.fallback_floor_deg,
            partial_bands=(rdl.partial.moderate, rdl.partial.severe),
        )
        shoulders = [sample.shoulder_cm for sample in samples]
        times = [sample.timestamp for sample in samples]
        smoothed = _running_median(shoulders, CROSSING_MEDIAN_SAMPLES)
        i_low = _lowest_index(smoothed, rep_start)
        if i_low is None:
            return features
        low_cm = smoothed[i_low]
        i_top, top_cm = self._top(shoulders, smoothed, rep_start)
        drop_cm = top_cm - low_cm

        rep_indices = [i for i in range(rep_start, len(samples)) if math.isfinite(shoulders[i])]
        bottom = [i for i in rep_indices if shoulders[i] <= low_cm + BOTTOM_WINDOW_CM]
        lower_half = [i for i in rep_indices if top_cm - shoulders[i] >= LOWER_HALF_FRACTION * drop_cm]
        if self._capture_mode == TRIANGULATED:
            # Any analysed rig frame has hips triangulated from two or more views.
            hips_seen = bottom
        else:
            hips_seen = [i for i in bottom if samples[i].hip_conf >= SINGLE_CAMERA_MIN_HIP_CONFIDENCE]
        hips_measured = len(hips_seen) >= MIN_WINDOW_FRAMES

        references = _references(setups)
        hip_sink_ratio = math.nan
        if hips_measured:
            bottom_hip_cm = _nanmedian(samples[i].hip_cm for i in hips_seen)
            hip_sink_ratio = _ratio(references.hip_cm - bottom_hip_cm, references.hip_cm)
            hinge_deg = _nanmedian(
                _hinge_deg(shoulders[i], samples[i].hip_cm, references.torso_cm) for i in hips_seen
            )
        else:
            hinge_deg = _nanmedian(_hinge_deg(shoulders[i], references.hip_cm, references.torso_cm) for i in bottom)

        wrist_bottom = [samples[i].wrist_mid_cm for i in bottom if math.isfinite(samples[i].wrist_mid_cm)]
        wrist_bottom_cm = _nanmedian(wrist_bottom) if len(wrist_bottom) >= MIN_WINDOW_FRAMES else math.nan
        bar_height_ratio = _ratio(wrist_bottom_cm, references.knee_cm)
        bar_tilt_deg, bar_tilt_side = _bar_tilt(samples, rep_start, bottom)
        offset_p80, offset_phase = _bar_offset(samples, lower_half, bottom, i_low)

        start_time, end_time = descent_crossings(times, smoothed, i_top, i_low, top_cm, drop_cm)
        descent_s = end_time - start_time
        ascent_s = times[-1] - times[i_low]
        rep_setup = None if self._piston else self._rep_setup

        features.max_depth_angle = hinge_deg
        features.hinge_hip_source = HIP_SOURCE_MEASURED if hips_measured else HIP_SOURCE_SETUP
        features.hip_sink_ratio = hip_sink_ratio
        features.bar_travel_ratio = _ratio(references.wrist_cm - wrist_bottom_cm, references.wrist_cm - references.knee_cm)
        features.hinge_deg = hinge_deg
        features.bar_height_ratio = bar_height_ratio
        features.range_label = _range_label(bar_height_ratio)
        features.bar_tilt_deg = bar_tilt_deg
        features.bar_tilt_side = bar_tilt_side
        features.bar_offset_ratio_p80 = offset_p80
        features.bar_offset_peak_phase = offset_phase
        features.lockout_pitch_deg = rep_setup.trunk_pitch_signed_deg if rep_setup is not None else math.nan
        features.shoulder_drop_cm = drop_cm
        features.descent_time_s = descent_s
        features.ascent_time_s = ascent_s
        features.duration_s = times[-1] - start_time
        features.eccentric_time_s = descent_s
        features.concentric_time_s = ascent_s
        features.concentric_velocity_mps = _concentric_velocity_mps(shoulders, times, i_low, low_cm)
        features.knee_flexion_bottom_deg = _nanmedian(samples[i].knee_flexion_mean for i in bottom)
        features.trunk_pitch_bottom_3d = _nanmedian(samples[i].trunk_pitch for i in bottom)
        return features
