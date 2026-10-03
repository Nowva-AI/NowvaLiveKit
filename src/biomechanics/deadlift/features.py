"""Per-rep deadlift features (PLAN.md §2.5) from the frames the analyser kept.

Pure functions over a RepTrack (liftoff through the top to the floor) and the
setup frames before liftoff. The setup model (§2.7) turns the measured setup into
the D4 hip band and the trunk change D2 expects for this body.
"""

from __future__ import annotations

import logging
import math
from typing import NamedTuple

import numpy as np

from biomechanics.config import DeadliftConfig

from .frame import SagittalFrame, forward_m, height_m, lateral_m
from .measure import FrameMeasure
from .setup_model import AthleteSegments, SetupPrediction, predict_setup
from .types import BAR_SOURCE_BAR, BAR_SOURCE_WRIST_PROXY, NAN, DeadliftRepFeatures

logger = logging.getLogger(__name__)

# Percentiles of the robust per-rep statistics (rep_features.py convention).
WORST_PERCENTILE = 90.0
# The hips' start position: the first fifth of the pull, at least this many frames.
SHIFT_START_FRAMES = 5
SHIFT_START_FRACTION = 0.2
# Ankle separation below this cannot normalise a sideways hip shift.
MIN_ANKLE_SEPARATION_M = 0.05
# Plate-hub span of a standard 20 kg bar: D8b is the height difference of its ends.
PROXY_HUB_SPAN_M = 1.70
# Hands closer than this cannot carry a tilt out to the hubs.
MIN_GRIP_WIDTH_M = 0.25
# Tilt is read on a running median over this many frames.
TILT_SMOOTHING_FRAMES = 5
# D2's poses: the median over this long before liftoff, a curve fitted over this
# long up to the knee pass: a parabola with this many frames (a fast pull
# accelerates through the window, which a line evaluated at its end reads ~0.1
# low in rise ratio), a line with fewer, a median below that.
COORDINATION_WINDOW_S = 0.2
COORDINATION_FIT_FRAMES = 4
COORDINATION_CURVE_FRAMES = 6
# Hip/shoulder rise ratio needs this much shoulder rise to be a ratio at all.
MIN_SHOULDER_RISE_M = 0.01
# Plausible segment lengths for the setup model (m); outside them the keypoints are wrong.
MIN_SEGMENT_M = 0.15
MAX_SEGMENT_M = 0.90
# Event timing: leaving or reaching a level with constant acceleration, the
# distance from the level is c * (t - t0)^2 on one side of the event t0 and zero
# on the other. Fitted on the frames within this distance of the level, by a grid
# over t0 (least squares in height, where the noise is uniform).
EVENT_FIT_SPAN_M = 0.04
EVENT_FIT_MIN_FRAMES = 3
EVENT_GRID_STEP_S = 0.002
# The fitted event stays within this of the frame that detected it (the
# liftoff look-back).
EVENT_MAX_SHIFT_S = 0.5


class AthleteState(NamedTuple):
    """What the analyser knows about the athlete when a rep's features are taken."""
    standing_refs: dict[str, float]
    seeded_segments: dict[str, float]
    grip_offset_m: float


class RepTrack:
    """The rep in progress: from liftoff through the top to the floor."""

    def __init__(self, liftoff: FrameMeasure, touch_and_go: bool, setup: dict[str, float]) -> None:
        self.liftoff = liftoff
        self.touch_and_go = touch_and_go
        self.setup = setup
        self.frames: list[FrameMeasure] = [liftoff]
        self.velocities: list[float] = [0.0]
        self.knee_pass: FrameMeasure | None = None
        self.top_frames: list[FrameMeasure] = []
        self.top_time = NAN
        self.top_up = NAN
        self.lower_start = NAN
        self.peak_up = liftoff.bar_up
        self.peak_index = 0
        self.floor_time = NAN
        # History frames just before liftoff, for the liftoff's timing fit.
        self.lead_in: list[FrameMeasure] = []
        # The rep ended in a touch-and-go: its floor is the low point itself.
        self.touch_and_go_out = False

    def append(self, measure: FrameMeasure, velocity: float) -> None:
        self.frames.append(measure)
        self.velocities.append(velocity)
        if measure.bar_up > self.peak_up:
            self.peak_up = measure.bar_up
            self.peak_index = len(self.frames) - 1


def _median(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.median(finite)) if finite else NAN


def _percentile(values: list[float], percentile: float) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.percentile(finite, percentile)) if finite else NAN


# When the bar left (leaving) or reached level_up, between earliest and latest:
# the change point of a parabola that is flat on the level's side. NaN when too
# few frames clear of the band show the motion (a fast bar crosses it in one or
# two frames, and the detecting frame is then already the event).
def _event_time(
    frames: list[FrameMeasure], level_up: float, band_m: float, earliest: float, latest: float, leaving: bool,
) -> float:
    times = []
    distances = []
    for frame in frames:
        distance = abs(frame.bar_up - level_up)
        if math.isfinite(distance) and distance <= EVENT_FIT_SPAN_M:
            times.append(frame.t)
            distances.append(distance)
    moving = sum(1 for distance in distances if distance >= band_m)
    if moving < EVENT_FIT_MIN_FRAMES or latest <= earliest:
        return NAN
    time_array = np.asarray(times)
    distance_array = np.asarray(distances)
    candidates = np.arange(earliest, latest + 1e-9, EVENT_GRID_STEP_S)
    # (candidates, frames): squared time since leaving (or until reaching) the level.
    offsets = time_array[None, :] - candidates[:, None]
    squared = np.maximum(0.0, offsets if leaving else -offsets) ** 2
    weight = np.sum(squared ** 2, axis=1)
    usable = weight > 0.0
    if not usable.any():
        return NAN
    curvature = np.where(usable, squared @ distance_array / np.where(usable, weight, 1.0), 0.0)
    residual = np.sum((distance_array[None, :] - curvature[:, None] * squared) ** 2, axis=1)
    residual = np.where(usable, residual, np.inf)
    return float(candidates[int(np.argmin(residual))])


def liftoff_time(rep: RepTrack, rest_up: float, band_m: float) -> float:
    """The bar's departure from its rest. The back-dated liftoff frame is the last
    one inside the rest band, so a slow, smooth start left the floor earlier."""
    first_clear = next(
        (frame.t for frame in rep.frames if frame.bar_up - rest_up > band_m), rep.liftoff.t,
    )
    frames = rep.lead_in + [frame for frame in rep.frames if frame.t <= rep.top_time]
    estimate = _event_time(frames, rest_up, band_m, rep.liftoff.t - EVENT_MAX_SHIFT_S, first_clear, leaving=True)
    return estimate if math.isfinite(estimate) else rep.liftoff.t


def top_time(rep: RepTrack, band_m: float) -> float:
    """The bar's arrival at its top: the level it held there (or its peak, for a
    top that never held), reached after the first frame inside the band."""
    level = _median([frame.bar_up for frame in rep.top_frames]) if rep.top_frames else rep.top_up
    if not math.isfinite(level):
        return rep.top_time
    arrival = next(
        (frame.t for frame in rep.frames if frame.bar_up >= level - band_m), rep.top_time,
    )
    leaving = rep.lower_start if math.isfinite(rep.lower_start) else rep.frames[-1].t
    frames = [frame for frame in rep.frames if frame.t <= leaving]
    estimate = _event_time(frames, level, band_m, arrival, min(arrival + EVENT_MAX_SHIFT_S, leaving), leaving=False)
    return estimate if math.isfinite(estimate) else arrival


def floor_time(rep: RepTrack, rest_up: float, band_m: float) -> float:
    """The bar's touchdown: after the first frame inside the rest band, from the
    frames of the lowering and the rest after it."""
    if rep.touch_and_go_out or math.isnan(rep.lower_start) or math.isnan(rep.floor_time):
        return rep.floor_time
    frames = [frame for frame in rep.frames if frame.t >= rep.lower_start]
    latest = min(rep.floor_time + EVENT_MAX_SHIFT_S, rep.frames[-1].t)
    estimate = _event_time(frames, rest_up, band_m, rep.floor_time, latest, leaving=False)
    return estimate if math.isfinite(estimate) else rep.floor_time


def _segment_lengths(
    frames: list[FrameMeasure], athlete: AthleteState,
) -> AthleteSegments | None:
    def projected(lower: np.ndarray | None, upper: np.ndarray | None, frame: SagittalFrame) -> float:
        if lower is None or upper is None:
            return NAN
        return math.hypot(forward_m(frame, upper, lower), height_m(frame, upper, lower))

    tibia = _median([projected(f.ankle_mid, f.knee_mid, f.frame) for f in frames])
    femur = _median([projected(f.knee_mid, f.hip_mid, f.frame) for f in frames])
    torso = _median([projected(f.hip_mid, f.shoulder_mid, f.frame) for f in frames])
    arm = athlete.standing_refs.get("arm_m", NAN)
    if not math.isfinite(arm):
        arm = _median([f.arm_m for f in frames])
    if not math.isfinite(torso):
        torso = athlete.seeded_segments.get("torso_avg_m", NAN)
    lengths = (tibia, femur, torso, arm)
    if not all(math.isfinite(length) and MIN_SEGMENT_M <= length <= MAX_SEGMENT_M for length in lengths):
        return None
    return AthleteSegments(
        tibia_m=tibia, femur_m=femur, torso_m=torso, arm_m=arm, grip_offset_m=athlete.grip_offset_m,
    )


# How far in front of this lifter's shin line the bar sits at setup.
#
# Shin thickness and how hard the shins press the bar vary by person; with an
# assumed distance, 2 cm of difference moved the model's hip band ~7 cm and its
# trunk prediction ~7 deg (simulator), faking D4 and D2 on a clean setup. Measured
# on the setup frames and clipped to what a shin allows.
def _shin_bar_distance_m(frames: list[FrameMeasure], config: DeadliftConfig) -> float:
    distances = []
    for frame in frames:
        if frame.knee_mid is None or frame.bar_centre is None:
            continue
        shin_forward = forward_m(frame.frame, frame.knee_mid, frame.ankle_mid)
        shin_up = height_m(frame.frame, frame.knee_mid, frame.ankle_mid)
        bar_forward = forward_m(frame.frame, frame.bar_centre, frame.ankle_mid)
        bar_up = height_m(frame.frame, frame.bar_centre, frame.ankle_mid)
        shin_length = math.hypot(shin_forward, shin_up)
        if shin_length > 0.0:
            distances.append((bar_forward * shin_up - bar_up * shin_forward) / shin_length)
    measured = _median(distances)
    if not math.isfinite(measured):
        return config.shin_bar_distance_m
    return min(max(measured, config.min_shin_bar_distance_m), config.max_shin_bar_distance_m)


def _predict_setup(
    frames: list[FrameMeasure], shoulder_vs_bar_cm: float, athlete: AthleteState, config: DeadliftConfig,
) -> SetupPrediction | None:
    segments = _segment_lengths(frames, athlete)
    if segments is None:
        return None
    prediction = predict_setup(
        segments,
        bar_forward_m=_median([forward_m(f.frame, f.bar_centre, f.ankle_mid) for f in frames]),
        bar_height_m=_median([height_m(f.frame, f.bar_centre, f.ankle_mid) for f in frames]),
        shoulder_band_m=(config.shoulder_band_low_m, config.shoulder_band_high_m),
        shoulder_ahead_m=shoulder_vs_bar_cm / 100.0,
        shin_bar_m=_shin_bar_distance_m(frames, config),
    )
    if prediction is None:
        logger.info(
            "[DEADLIFT] No setup fits these segments (tibia %.2f, femur %.2f, torso %.2f, arm %.2f m): "
            "setup hips and shoulders are not judged for this athlete",
            segments.tibia_m, segments.femur_m, segments.torso_m, segments.arm_m,
        )
    return prediction


def measure_setup(
    setup_frames: list[FrameMeasure],
    liftoff_t: float,
    window_s: float,
    locked_midfoot: np.ndarray | None,
    athlete: AthleteState,
    config: DeadliftConfig,
) -> dict[str, float]:
    """The setup over window_s before liftoff: hip height, trunk, shoulders vs bar,
    bar vs midfoot and the model's band and trunk change. "grip_offset_m" is the
    wrist-to-bar offset measured on the tracked bar, for the athlete.

    The leg measures need frames whose legs were measured, not carried by the
    Kalman. The bar measures need only the bar, the shoulders and the midfoot
    locked at the stance, so feet hidden by the plates at setup still leave D1
    and D7 judged. Missing keys mean too few frames."""
    window = [frame for frame in setup_frames if liftoff_t - window_s - 1e-6 <= frame.t <= liftoff_t]
    setup: dict[str, float] = {}
    legs = [f for f in window if f.legs_measured]
    if len(legs) >= config.min_setup_frames:
        setup["hip_height_cm"] = _median([height_m(f.frame, f.hip_mid, f.ankle_mid) * 100.0 for f in legs])
        setup["trunk_deg"] = _median([f.trunk_deg for f in legs])
    # A bar carried over a tracking gap or predicted by the tracker is not a
    # setup measurement.
    bar_frames = [f for f in window if f.bar_centre is not None and f.bar_measured]
    if len(bar_frames) < config.min_setup_frames:
        return setup
    setup["shoulder_vs_bar_cm"] = _median([
        forward_m(f.frame, f.shoulder_mid, f.bar_centre) * 100.0 for f in bar_frames
    ])
    if locked_midfoot is not None:
        setup["bar_midfoot_cm"] = _median([
            forward_m(f.frame, f.bar_centre, locked_midfoot) * 100.0 for f in bar_frames
        ])
    tracked = [f for f in bar_frames if f.bar_source == BAR_SOURCE_BAR and f.wrist_mid is not None]
    if tracked:
        offset = _median([height_m(f.frame, f.wrist_mid, f.bar_centre) for f in tracked])
        if math.isfinite(offset):
            setup["grip_offset_m"] = min(
                max(offset, config.min_wrist_to_bar_offset_m), config.max_wrist_to_bar_offset_m,
            )
            athlete = athlete._replace(grip_offset_m=setup["grip_offset_m"])
    model_frames = [f for f in bar_frames if f.legs_measured]
    if len(model_frames) < config.min_setup_frames:
        return setup
    prediction = _predict_setup(model_frames, setup["shoulder_vs_bar_cm"], athlete, config)
    if prediction is not None:
        setup["band_low_cm"] = prediction.hip_band_low_m * 100.0
        setup["band_high_cm"] = prediction.hip_band_high_m * 100.0
        setup["predicted_trunk_change_deg"] = prediction.trunk_change_deg
    return setup


# (trunk deg, hip height, shoulder height) at time t, from a curve fitted through
# the frames leading to it (their median when too few): a single frame carries
# the whole keypoint noise into D2.
def _pose_at(frames: list[FrameMeasure], t: float, up: np.ndarray) -> tuple[float, float, float]:
    times = np.array([frame.t for frame in frames]) - t
    values = np.array([
        (frame.trunk_deg, float(np.dot(frame.hip_mid, up)), float(np.dot(frame.shoulder_mid, up)))
        for frame in frames
    ])
    if len(frames) < COORDINATION_FIT_FRAMES or np.ptp(times) <= 0.0:
        return tuple(float(value) for value in np.median(values, axis=0))
    degree = 2 if len(frames) >= COORDINATION_CURVE_FRAMES else 1
    return tuple(float(value) for value in np.polyfit(times, values, degree)[-1])


def _coordination_features(rep: RepTrack, features: DeadliftRepFeatures) -> None:
    knee_pass = rep.knee_pass
    liftoff = rep.liftoff
    if knee_pass is None or math.isnan(rep.top_time) or knee_pass.t > rep.top_time:
        return
    features.knee_pass_time = knee_pass.t
    up = knee_pass.frame.up
    # Before liftoff the lifter is set (a median); up to the knee pass, moving (a
    # line through the frames leading to it: past the knees the trunk changes course).
    before = [frame for frame in rep.lead_in if liftoff.t - frame.t <= COORDINATION_WINDOW_S] + [liftoff]
    trunk_lo, hip_lo, shoulder_lo = (float(value) for value in np.median([
        (frame.trunk_deg, float(np.dot(frame.hip_mid, up)), float(np.dot(frame.shoulder_mid, up)))
        for frame in before
    ], axis=0))
    leading = [frame for frame in rep.frames if 0.0 <= knee_pass.t - frame.t <= COORDINATION_WINDOW_S]
    trunk_kp, hip_kp, shoulder_kp = _pose_at(leading, knee_pass.t, up)
    predicted = features.trunk_change_predicted_deg
    change = trunk_kp - trunk_lo
    # Without the setup model the raw change is judged: the trunk must still
    # not tip further forward off the floor.
    features.trunk_change_liftoff_knee_deg = change - predicted if math.isfinite(predicted) else change
    hip_rise = hip_kp - hip_lo
    shoulder_rise = shoulder_kp - shoulder_lo
    if shoulder_rise > MIN_SHOULDER_RISE_M:
        features.hip_shoulder_rise_ratio = hip_rise / shoulder_rise


def _running_median(values: list[float], frames: int) -> list[float]:
    half = frames // 2
    return [float(np.median(values[max(0, i - half):i + half + 1])) for i in range(len(values))]


# Left end above the right end (cm) at the plate hubs: measured there on the
# tracked bar; without it, the hands' height difference carried out to the hubs.
def _tilt_cm(frame: FrameMeasure) -> float:
    if frame.bar_left is not None and frame.bar_right is not None:
        return height_m(frame.frame, frame.bar_left, frame.bar_right) * 100.0
    if frame.bar_source == BAR_SOURCE_WRIST_PROXY and frame.l_wrist is not None and frame.r_wrist is not None:
        grip_m = abs(lateral_m(frame.frame, frame.r_wrist, frame.l_wrist))
        if grip_m >= MIN_GRIP_WIDTH_M:
            return height_m(frame.frame, frame.l_wrist, frame.r_wrist) * PROXY_HUB_SPAN_M / grip_m * 100.0
    return NAN


def _bar_path_features(pull: list[FrameMeasure], hold: list[FrameMeasure], features: DeadliftRepFeatures) -> None:
    bar_frames = [f for f in pull if f.bar_centre is not None]
    if len(bar_frames) < 2:
        return
    start = bar_frames[0]
    drift = [
        max(0.0, forward_m(f.frame, f.bar_centre, start.bar_centre)) * 100.0 for f in bar_frames
    ]
    features.bar_drift_cm = _percentile(drift, WORST_PERCENTILE)
    if bar_frames[0].bar_source == BAR_SOURCE_WRIST_PROXY:
        # The hands' keypoint noise, carried ~3x out to the hubs, swamps any one
        # frame: on the proxy the tilt is the level held from mid-pull through the top.
        held = bar_frames[len(bar_frames) // 2:] + [f for f in hold if f.bar_centre is not None]
        tilts = [tilt for tilt in (_tilt_cm(frame) for frame in held) if math.isfinite(tilt)]
        if not tilts:
            return
        features.bar_tilt_cm = abs(_median(tilts))
    else:
        tilts = [tilt for tilt in (_tilt_cm(frame) for frame in bar_frames) if math.isfinite(tilt)]
        if not tilts:
            return
        if len(tilts) >= TILT_SMOOTHING_FRAMES:
            tilts = _running_median(tilts, TILT_SMOOTHING_FRAMES)
        features.bar_tilt_cm = _percentile([abs(tilt) for tilt in tilts], WORST_PERCENTILE)
    features.bar_low_side = "left" if _median(tilts) < 0.0 else "right"


def _top_features(rep: RepTrack, refs: dict[str, float], features: DeadliftRepFeatures) -> None:
    top = [f for f in rep.top_frames if f.legs_measured] or rep.top_frames
    if not top or not refs:
        return
    features.hip_extension_deficit_deg = _median([f.hip_flex_deg for f in top]) - refs.get("hip_flex_deg", NAN)
    features.knee_extension_deficit_deg = _median([f.knee_flex_deg for f in top]) - refs.get("knee_flex_deg", NAN)
    features.lean_back_deg = refs.get("trunk_deg", NAN) - _median([f.trunk_deg for f in top])


# The hips' sideways position over the second half of the pull against where
# they started, as a fraction of ankle separation, along the lifter's own
# left-right (PLAN.md §2.5: the ankle axis; the bar's axis leaks forward travel
# into sideways on a stance a few degrees off square). Medians over many frames:
# a hip keypoint jitters by more than a real shift's first centimetres.
def _hip_shift_feature(
    pull: list[FrameMeasure],
    locked_midfoot: np.ndarray | None,
    body_lateral: np.ndarray,
    features: DeadliftRepFeatures,
) -> None:
    ratios: list[float] = []
    for frame in pull:
        if frame.l_ankle is None or frame.r_ankle is None:
            continue
        separation = abs(float(np.dot(frame.r_ankle - frame.l_ankle, body_lateral)))
        if separation < MIN_ANKLE_SEPARATION_M:
            continue
        reference = locked_midfoot if locked_midfoot is not None else frame.ankle_mid
        ratios.append(float(np.dot(frame.hip_mid - reference, body_lateral)) / separation)
    if len(ratios) < 2 * SHIFT_START_FRAMES:
        return
    start_frames = max(SHIFT_START_FRAMES, int(len(ratios) * SHIFT_START_FRACTION))
    start = float(np.median(ratios[:start_frames]))
    features.hip_shift_ratio = float(np.median(ratios[len(ratios) // 2:])) - start


def rep_features(
    rep: RepTrack,
    rep_number: int,
    rest_up: float,
    event_band_m: float,
    locked_midfoot: np.ndarray | None,
    body_lateral: np.ndarray,
    athlete: AthleteState,
    gravity_source: str,
    grip: str,
) -> DeadliftRepFeatures:
    """body_lateral: the lifter's horizontal left-to-right unit vector, locked for
    the set (hips and ankles), that sideways hip shift is measured along."""
    pull = [frame for frame in rep.frames if frame.t <= rep.top_time]
    liftoff = rep.liftoff
    setup = rep.setup
    refs = athlete.standing_refs
    bar_source = BAR_SOURCE_WRIST_PROXY if any(f.bar_source == BAR_SOURCE_WRIST_PROXY for f in pull) else BAR_SOURCE_BAR
    start_t = liftoff.t if rep.touch_and_go else liftoff_time(rep, rest_up, event_band_m)
    end_t = top_time(rep, event_band_m)

    features = DeadliftRepFeatures(
        rep_number=rep_number,
        sample_count=len(rep.frames),
        touch_and_go=rep.touch_and_go,
        setup_measured="hip_height_cm" in setup,
        bar_midfoot_stance_cm=setup.get("stance_bar_midfoot_cm", NAN),
        bar_midfoot_setup_cm=setup.get("bar_midfoot_cm", NAN),
        setup_hip_height_cm=setup.get("hip_height_cm", NAN),
        setup_hip_band_low_cm=setup.get("band_low_cm", NAN),
        setup_hip_band_high_cm=setup.get("band_high_cm", NAN),
        shoulder_vs_bar_cm=setup.get("shoulder_vs_bar_cm", NAN),
        setup_trunk_deg=setup.get("trunk_deg", NAN),
        trunk_change_predicted_deg=setup.get("predicted_trunk_change_deg", NAN),
        bar_source=bar_source,
        gravity_source=gravity_source,
        grip=grip,
        liftoff_time=start_t,
        top_time=end_t,
        floor_time=floor_time(rep, rest_up, event_band_m),
    )
    rise_m = rep.top_up - liftoff.bar_up
    pull_time = end_t - start_t
    features.bar_rise_cm = rise_m * 100.0
    features.pull_time_s = pull_time
    if pull_time > 0.0:
        features.concentric_velocity_mps = rise_m / pull_time
    if math.isfinite(rep.lower_start) and math.isfinite(features.floor_time):
        features.lower_time_s = features.floor_time - rep.lower_start

    _coordination_features(rep, features)
    _bar_path_features(pull, rep.top_frames, features)
    _top_features(rep, refs, features)
    _hip_shift_feature(pull, locked_midfoot, body_lateral, features)
    if refs and math.isfinite(refs.get("elbow_flex_deg", NAN)):
        features.elbow_flexion_deg = _percentile(
            [f.elbow_flex_deg - refs["elbow_flex_deg"] for f in pull], WORST_PERCENTILE,
        )
    return features
