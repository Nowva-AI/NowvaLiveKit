"""The deadlift rep analyser: a bar-driven state machine and the per-rep features
its fault rules judge (PLAN.md §2.2–2.7).

It owns all deadlift per-frame state, so the profile stays stateless and the squat
pipeline state (setup snapshot, standing reference, trajectory) is never touched.
Input is the lagged world-frame skeleton (Y-down metres) with the bar state matched
to its capture time. Everything is measured in the sagittal frame (up = measured
gravity), as differences: bar vs its resting height, bar vs midfoot, joints vs
their standing reference.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import NamedTuple

import numpy as np

from biomechanics.config import DeadliftConfig
from biomechanics.diagnosis.lean_model import MIDFOOT_FRACTION_OF_ANKLE_TO_TOE
from biomechanics.utils.types import CocoKeypoints as CK

from .frame import (
    SagittalFrame,
    build_sagittal_frame,
    default_up,
    forward_m,
    height_m,
    lateral_m,
    segment_angle_deg,
)
from .setup_model import AthleteSegments, SetupPrediction, predict_setup
from .types import (
    BAR_SOURCE_BAR,
    BAR_SOURCE_WRIST_PROXY,
    GRAVITY_SOURCE_BODY,
    GRAVITY_SOURCE_MEASURED,
    NAN,
    BarState3D,
    DeadliftFrameStatus,
    DeadliftPhase,
    DeadliftRepFeatures,
)

logger = logging.getLogger(__name__)

# Keypoints below this confidence are missing (the IK solver's floor).
MIN_KEYPOINT_CONFIDENCE = 0.1
# Frames of history kept: longer than the liftoff look-back and any setup window.
HISTORY_S = 3.0
# Body speed is the least-squares slope over this window: a finite difference
# over a short window turns keypoint jitter into apparent motion.
SPEED_WINDOW_S = 0.3
# The bar counts as still when its height's slope over this longer window is small.
STILL_BAR_WINDOW_S = 0.25
# Percentiles of the robust per-rep statistics (rep_features.py convention).
WORST_PERCENTILE = 90.0
SHIFT_SUSTAINED_PERCENTILE = 80.0
SHIFT_START_FRAMES = 5
# Ankle separation below this cannot normalise a sideways hip shift.
MIN_ANKLE_SEPARATION_M = 0.05
# Hip/shoulder rise ratio needs this much shoulder rise to be a ratio at all.
MIN_SHOULDER_RISE_M = 0.01
# The top event is the bar's arrival this close to its peak height.
TOP_ARRIVAL_BAND_M = 0.005
# Frames either side of the peak that stand in for a top that never held still.
TOP_PEAK_WINDOW_S = 0.1
# Rest height is the median of this many recent still frames of the bar.
REST_FRAMES = 10
# Frame-to-frame dip still counted as "rising" off a touch-and-go low point.
TOUCH_GO_JITTER_M = 0.003
# A rep never lasts longer than this; a stuck PULL is abandoned.
MAX_PULL_S = 15.0
# Plausible segment lengths for the setup model (m); outside them the keypoints are wrong.
MIN_SEGMENT_M = 0.15
MAX_SEGMENT_M = 0.90


class DeadliftFrameInput(NamedTuple):
    """One lagged analysis frame. points: (N, 3) world metres, Y-down.
    legs_measured: the legs were triangulated, not carried by the Kalman."""
    timestamp: float
    frame_index: int
    points: np.ndarray
    confidences: np.ndarray
    bar: BarState3D | None
    legs_measured: bool = True


class _Measure(NamedTuple):
    t: float
    frame_index: int
    legs_measured: bool
    frame: SagittalFrame
    ankle_mid: np.ndarray
    midfoot: np.ndarray | None
    hip_mid: np.ndarray
    shoulder_mid: np.ndarray
    knee_mid: np.ndarray | None
    wrist_mid: np.ndarray | None
    l_wrist: np.ndarray | None
    r_wrist: np.ndarray | None
    l_ankle: np.ndarray | None
    r_ankle: np.ndarray | None
    bar_centre: np.ndarray | None
    bar_left: np.ndarray | None
    bar_right: np.ndarray | None
    bar_source: str
    bar_up: float
    trunk_deg: float
    hip_flex_deg: float
    knee_flex_deg: float
    elbow_flex_deg: float
    arm_m: float
    hands_on_bar: bool
    standing: bool
    foot_forward: np.ndarray | None
    points: np.ndarray


class _Rep:
    """The rep in progress: from liftoff through the top to the floor."""

    def __init__(self, liftoff: _Measure, touch_and_go: bool, setup: dict[str, float]) -> None:
        self.liftoff = liftoff
        self.touch_and_go = touch_and_go
        self.setup = setup
        self.frames: list[_Measure] = [liftoff]
        self.velocities: list[float] = [0.0]
        self.knee_pass: _Measure | None = None
        self.top_frames: list[_Measure] = []
        self.top_time = NAN
        self.top_up = NAN
        self.lower_start = NAN
        self.peak_up = liftoff.bar_up
        self.peak_index = 0
        self.floor_time = NAN

    def append(self, measure: _Measure, velocity: float) -> None:
        self.frames.append(measure)
        self.velocities.append(velocity)
        if measure.bar_up > self.peak_up:
            self.peak_up = measure.bar_up
            self.peak_index = len(self.frames) - 1


class _CompletedRep(NamedTuple):
    features: DeadliftRepFeatures
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int


def _point(points: np.ndarray, confidences: np.ndarray, index: int) -> np.ndarray | None:
    if index >= len(points) or confidences[index] < MIN_KEYPOINT_CONFIDENCE:
        return None
    return points[index]


def _midpoint(first: np.ndarray | None, second: np.ndarray | None) -> np.ndarray | None:
    if first is None or second is None:
        return first if second is None else second
    return (first + second) / 2.0


def _strict_midpoint(first: np.ndarray | None, second: np.ndarray | None) -> np.ndarray | None:
    if first is None or second is None:
        return None
    return (first + second) / 2.0


def _elbow_flexion_deg(
    frame: SagittalFrame, shoulder: np.ndarray | None, elbow: np.ndarray | None, wrist: np.ndarray | None,
) -> float:
    """Elbow bend seen from the side. In 3D the angle also changes with grip width
    (hands out on the bar vs hanging), which is not a bent arm."""
    if shoulder is None or elbow is None or wrist is None:
        return NAN
    upper_deg = segment_angle_deg(frame, elbow, shoulder)
    fore_deg = segment_angle_deg(frame, wrist, elbow)
    return abs(upper_deg - fore_deg)


def _arm_length_m(shoulder: np.ndarray | None, elbow: np.ndarray | None, wrist: np.ndarray | None) -> float:
    if shoulder is None or elbow is None or wrist is None:
        return NAN
    return float(np.linalg.norm(elbow - shoulder) + np.linalg.norm(wrist - elbow))


def _nanmean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else NAN


def _median(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.median(finite)) if finite else NAN


def _percentile(values: list[float], percentile: float) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.percentile(finite, percentile)) if finite else NAN


def _slope(times: list[float], values: list[float]) -> float:
    if len(times) < 2:
        return 0.0
    time_array = np.asarray(times) - times[-1]
    value_array = np.asarray(values)
    time_centred = time_array - time_array.mean()
    variance = float(np.dot(time_centred, time_centred))
    if variance <= 0.0:
        return 0.0
    return float(np.dot(time_centred, value_array - value_array.mean()) / variance)


class DeadliftRepAnalyzer:
    """Feed observe() once per analysis frame; completed reps queue up for the
    counter (rep data) and the pipeline (finish_rep features)."""

    def __init__(self, config: DeadliftConfig | None = None) -> None:
        self.config = config or DeadliftConfig()
        self._up = default_up()
        self.gravity_source = GRAVITY_SOURCE_BODY
        self._seeded_segments: dict[str, float] = {}
        self._learned_grip_offset_m = NAN
        self._rep_counter = None
        self._grip = ""
        self.reset()

    # ------------------------------------------------------------------
    # Session and set lifecycle
    # ------------------------------------------------------------------

    def set_gravity(self, up_world: np.ndarray, source: str) -> None:
        """Measured gravity in the current world frame, or the body vertical (source "body")."""
        up = np.asarray(up_world, dtype=np.float64)
        self._up = up / float(np.linalg.norm(up))
        self.gravity_source = source

    def set_session_meta(self, meta: dict) -> None:
        """Session metadata from the agent; the grip type is recorded with every rep
        (a mixed grip rotates the trunk by design, so it explains, never excuses)."""
        grip = meta.get("grip") or meta.get("grip_type") or ""
        self._grip = str(grip)

    def seed_segments(self, athlete_params: dict[str, float] | None) -> None:
        """Read-only lengths from the session's body measurement, used only where
        the deadlift's own setup frames did not measure a segment."""
        self._seeded_segments = dict(athlete_params or {})

    def reset(self) -> None:
        """Forget everything but the athlete (grip offset, seeded lengths)."""
        self.reset_set()
        self.rep_count = 0
        self.failed_reps = 0
        self._standing_refs: dict[str, float] = {}
        self._standing_points: list[list[float]] | None = None

    def reset_set(self) -> None:
        """Per-set state; standing references and the athlete survive."""
        self.phase = DeadliftPhase.APPROACH
        self._phase_since = NAN
        self._history: deque[_Measure] = deque()
        self._velocities: deque[float] = deque()
        self._held_since: dict[str, float] = {}
        self._rest_heights: deque[float] = deque(maxlen=REST_FRAMES)
        self._rest_up = NAN
        self._rest_source: str | None = None
        self._rest_axis: np.ndarray | None = None
        self._stance_midfoots: deque[np.ndarray] = deque(maxlen=self.config.midfoot_lock_frames)
        self._stance_bar_offsets: deque[tuple[float, float]] = deque()
        self._locked_midfoot: np.ndarray | None = None
        self._setup_frames: deque[_Measure] = deque()
        self._standing_run: list[_Measure] = []
        self._rep: _Rep | None = None
        self._completed: deque[_CompletedRep] = deque()
        self._awaiting_finish: deque[DeadliftRepFeatures] = deque()
        self._set_top_heights: list[float] = []
        self._stance_bar_midfoot_cm = NAN
        self._top_still_frames = 0
        self._dead_stop_frames = 0
        self._bar_still_velocity = 0.0
        self._last_measure: _Measure | None = None
        self.rep_started = False
        self._status = DeadliftFrameStatus(phase=self.phase, gravity_source=self.gravity_source)

    # ------------------------------------------------------------------
    # Public read-outs
    # ------------------------------------------------------------------

    @property
    def in_rep(self) -> bool:
        return self.phase in (DeadliftPhase.PULL, DeadliftPhase.TOP, DeadliftPhase.LOWER)

    @property
    def rep_counter(self):
        """The pipeline's rep counter view of this analyser (one per analyser)."""
        if self._rep_counter is None:
            from .rep_counter import DeadliftRepCounter

            self._rep_counter = DeadliftRepCounter(self)
        return self._rep_counter

    @property
    def status(self) -> DeadliftFrameStatus:
        return self._status

    @property
    def rep_signal(self) -> float:
        """Bar height above its resting height (cm); NaN until the rest is known."""
        return self._status.bar_height_cm

    @property
    def standing_points(self) -> list[list[float]] | None:
        """The last standing skeleton (world frame), for replays."""
        return self._standing_points

    def take_completed_rep(self) -> _CompletedRep | None:
        """Hand the next counted rep to the counter, once; its features then wait
        for the pipeline's finish_rep()."""
        if not self._completed:
            return None
        completed = self._completed.popleft()
        self._awaiting_finish.append(completed.features)
        return completed

    def finish_rep(self) -> DeadliftRepFeatures | None:
        """The features of the rep the counter just reported, for the fault rules."""
        return self._awaiting_finish.popleft() if self._awaiting_finish else None

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def observe(self, frame_input: DeadliftFrameInput) -> None:
        self.rep_started = False
        if self._rest_source is not None and self.phase == DeadliftPhase.APPROACH:
            if (frame_input.bar is not None) != (self._rest_source == BAR_SOURCE_BAR):
                # Away from the bar the other source (bar found or lost) may
                # define the rest afresh.
                self._clear_rest()
        measure = self._measure(frame_input)
        if measure is None:
            return
        if self._history and measure.t <= self._history[-1].t:
            return
        self._history.append(measure)
        while self._history and measure.t - self._history[0].t > HISTORY_S:
            self._history.popleft()
        velocity = self._bar_velocity(self.config.velocity_window_s)
        self._velocities.append(velocity)
        while len(self._velocities) > len(self._history):
            self._velocities.popleft()
        self._bar_still_velocity = self._bar_velocity(STILL_BAR_WINDOW_S)
        if math.isnan(self._phase_since):
            self._phase_since = measure.t
        self._last_measure = measure

        self._update_rest(measure)
        self._update_standing_reference(measure)
        self._step_state_machine(measure, velocity)
        self._status = self._frame_status(measure)

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def _measure(self, frame_input: DeadliftFrameInput) -> _Measure | None:
        points = np.asarray(frame_input.points, dtype=np.float64)
        confidences = np.asarray(frame_input.confidences, dtype=np.float64)
        get = lambda index: _point(points, confidences, index)  # noqa: E731

        l_hip, r_hip = get(CK.LEFT_HIP), get(CK.RIGHT_HIP)
        l_shoulder, r_shoulder = get(CK.LEFT_SHOULDER), get(CK.RIGHT_SHOULDER)
        l_ankle, r_ankle = get(CK.LEFT_ANKLE), get(CK.RIGHT_ANKLE)
        hip_mid = _strict_midpoint(l_hip, r_hip)
        shoulder_mid = _midpoint(l_shoulder, r_shoulder)
        ankle_mid = _midpoint(l_ankle, r_ankle)
        if hip_mid is None or shoulder_mid is None or ankle_mid is None:
            return None
        l_knee, r_knee = get(CK.LEFT_KNEE), get(CK.RIGHT_KNEE)
        knee_mid = _midpoint(l_knee, r_knee)
        l_elbow, r_elbow = get(CK.LEFT_ELBOW), get(CK.RIGHT_ELBOW)
        l_wrist, r_wrist = get(CK.LEFT_WRIST), get(CK.RIGHT_WRIST)
        wrist_mid = _midpoint(l_wrist, r_wrist)

        toe_vectors = []
        midfoot_points = []
        for ankle, toe_index in ((l_ankle, CK.LEFT_FOOT_INDEX), (r_ankle, CK.RIGHT_FOOT_INDEX)):
            toe = get(toe_index)
            if ankle is None or toe is None:
                continue
            foot = toe - ankle
            foot = foot - float(np.dot(foot, self._up)) * self._up
            toe_vectors.append(foot)
            midfoot_points.append(ankle + MIDFOOT_FRACTION_OF_ANKLE_TO_TOE * foot)
        toe_direction = np.sum(toe_vectors, axis=0) if toe_vectors else None
        midfoot = np.mean(midfoot_points, axis=0) if len(midfoot_points) == 2 else None

        bar = frame_input.bar
        lateral_hint = r_hip - l_hip
        if bar is not None:
            bar_axis = -bar.axis
            if float(np.dot(bar_axis, lateral_hint)) < 0.0:
                bar_axis = -bar_axis
            lateral_hint = bar_axis
        elif self._rest_axis is not None:
            lateral_hint = self._rest_axis
        frame = build_sagittal_frame(self._up, lateral_hint, toe_direction)
        if frame is None:
            return None

        foot_forward = None
        if toe_direction is not None:
            norm = float(np.linalg.norm(toe_direction))
            if norm > 1e-6:
                foot_forward = toe_direction / norm

        trunk_deg = segment_angle_deg(frame, hip_mid, shoulder_mid)
        hip_flex = NAN
        knee_flex = NAN
        if knee_mid is not None:
            thigh_deg = segment_angle_deg(frame, knee_mid, hip_mid)
            shin_deg = segment_angle_deg(frame, ankle_mid, knee_mid)
            hip_flex = trunk_deg - thigh_deg
            knee_flex = shin_deg - thigh_deg

        elbow_flex = _nanmean([
            _elbow_flexion_deg(frame, l_shoulder, l_elbow, l_wrist),
            _elbow_flexion_deg(frame, r_shoulder, r_elbow, r_wrist),
        ])
        arm_m = _nanmean([
            _arm_length_m(l_shoulder, l_elbow, l_wrist),
            _arm_length_m(r_shoulder, r_elbow, r_wrist),
        ])

        cfg = self.config
        # Heights are only comparable with a rest measured from the same source:
        # once a set runs on the wrists it stays on them, and a set on the tracked
        # bar has no bar on a frame the bar is lost rather than a jump to the wrists.
        use_bar = bar is not None and self._rest_source != BAR_SOURCE_WRIST_PROXY
        bar_centre = bar_left = bar_right = None
        bar_source = BAR_SOURCE_BAR
        if use_bar:
            bar_centre = bar.centre
            bar_left = np.asarray(bar.left_end_m, dtype=np.float64)
            bar_right = np.asarray(bar.right_end_m, dtype=np.float64)
        elif self._rest_source != BAR_SOURCE_BAR and wrist_mid is not None:
            bar_source = BAR_SOURCE_WRIST_PROXY
            bar_centre = wrist_mid - self._grip_offset_m() * frame.up
        bar_up = float(np.dot(bar_centre, frame.up)) if bar_centre is not None else NAN

        standing = (
            math.isfinite(knee_flex)
            and abs(knee_flex) <= cfg.standing_max_knee_deg
            and abs(trunk_deg) <= cfg.standing_max_trunk_deg
        )
        hands_on_bar = self._hands_on_bar(
            frame, bar_source, bar_centre, l_wrist, r_wrist, knee_mid, trunk_deg,
        )
        return _Measure(
            t=frame_input.timestamp,
            frame_index=frame_input.frame_index,
            legs_measured=frame_input.legs_measured,
            frame=frame,
            ankle_mid=ankle_mid,
            midfoot=midfoot,
            hip_mid=hip_mid,
            shoulder_mid=shoulder_mid,
            knee_mid=knee_mid,
            wrist_mid=wrist_mid,
            l_wrist=l_wrist,
            r_wrist=r_wrist,
            l_ankle=l_ankle,
            r_ankle=r_ankle,
            bar_centre=bar_centre,
            bar_left=bar_left,
            bar_right=bar_right,
            bar_source=bar_source,
            bar_up=bar_up,
            trunk_deg=trunk_deg,
            hip_flex_deg=hip_flex,
            knee_flex_deg=knee_flex,
            elbow_flex_deg=elbow_flex,
            arm_m=arm_m,
            hands_on_bar=hands_on_bar,
            standing=standing,
            foot_forward=foot_forward,
            points=points,
        )

    def _hands_on_bar(
        self,
        frame: SagittalFrame,
        bar_source: str,
        bar_centre: np.ndarray | None,
        l_wrist: np.ndarray | None,
        r_wrist: np.ndarray | None,
        knee_mid: np.ndarray | None,
        trunk_deg: float,
    ) -> bool:
        cfg = self.config
        wrists = [wrist for wrist in (l_wrist, r_wrist) if wrist is not None]
        if bar_centre is None or not wrists:
            return False
        if bar_source == BAR_SOURCE_WRIST_PROXY:
            # The proxy bar is the wrists, so "hands on the bar" is a hinge with
            # the hands below the knees.
            if knee_mid is None or trunk_deg < cfg.wrist_proxy_min_hinge_deg:
                return False
            return all(height_m(frame, wrist, knee_mid) < 0.0 for wrist in wrists)
        for wrist in wrists:
            above = height_m(frame, wrist, bar_centre)
            if not -cfg.hands_max_below_bar_m <= above <= cfg.hands_max_above_bar_m:
                return False
            if abs(forward_m(frame, wrist, bar_centre)) > cfg.hands_max_forward_m:
                return False
        return True

    def _grip_offset_m(self) -> float:
        if math.isfinite(self._learned_grip_offset_m):
            return self._learned_grip_offset_m
        return self.config.wrist_to_bar_offset_m

    def _bar_velocity(self, window_s: float) -> float:
        latest = self._history[-1]
        times: list[float] = []
        heights: list[float] = []
        for measure in reversed(self._history):
            if latest.t - measure.t > window_s + 1e-6:
                break
            if math.isfinite(measure.bar_up):
                times.append(measure.t)
                heights.append(measure.bar_up)
        if len(times) < 2:
            return 0.0
        times.reverse()
        heights.reverse()
        return _slope(times, heights)

    def _body_speed_mps(self) -> float:
        latest = self._history[-1]
        window = [m for m in self._history if latest.t - m.t <= SPEED_WINDOW_S + 1e-6]
        if len(window) < 3:
            return 0.0
        times = np.array([m.t for m in window]) - latest.t
        centred = times - times.mean()
        variance = float(np.dot(centred, centred))
        if variance <= 0.0:
            return 0.0
        speeds = []
        for points in (np.array([m.hip_mid for m in window]), np.array([m.shoulder_mid for m in window])):
            slope = centred @ (points - points.mean(axis=0)) / variance
            speeds.append(float(np.linalg.norm(slope)))
        return max(speeds)

    # ------------------------------------------------------------------
    # References
    # ------------------------------------------------------------------

    def _update_rest(self, measure: _Measure) -> None:
        """The bar's resting height: still and not in a rep. Without bar tracking it
        is the wrists' height while set up on the bar."""
        if self.in_rep or not math.isfinite(measure.bar_up):
            return
        if abs(self._bar_still_velocity) > self.config.liftoff_rest_speed_mps:
            return
        if measure.bar_source == BAR_SOURCE_WRIST_PROXY:
            if self.phase not in (DeadliftPhase.SETUP, DeadliftPhase.FLOOR) or not measure.hands_on_bar:
                return
        elif measure.hands_on_bar and self.phase not in (DeadliftPhase.SETUP, DeadliftPhase.FLOOR):
            # Someone holding the bar off the floor outside a set-up is not a rest.
            return
        self._rest_heights.append(measure.bar_up)
        self._rest_up = float(np.median(self._rest_heights))
        self._rest_source = measure.bar_source
        if measure.bar_left is not None and measure.bar_right is not None:
            axis = measure.bar_right - measure.bar_left
            self._rest_axis = axis / float(np.linalg.norm(axis))

    def _clear_rest(self) -> None:
        self._rest_heights.clear()
        self._rest_up = NAN
        self._rest_source = None

    def _update_standing_reference(self, measure: _Measure) -> None:
        """Standing still with the arms hanging: the angles lockout, lean-back and
        bent arms are measured against, and the height the bar reaches at the top."""
        if self.phase not in (DeadliftPhase.APPROACH, DeadliftPhase.STANCE):
            self._standing_run = []
            return
        still = self._body_speed_mps() <= self.config.still_speed_mps
        if not (measure.standing and still and measure.legs_measured):
            self._standing_run = []
            return
        self._standing_run.append(measure)
        run = self._standing_run
        if run[-1].t - run[0].t < self.config.standing_still_s:
            return
        window = [m for m in run if run[-1].t - m.t <= self.config.standing_still_s]
        self._standing_refs = {
            "trunk_deg": _median([m.trunk_deg for m in window]),
            "hip_flex_deg": _median([m.hip_flex_deg for m in window]),
            "knee_flex_deg": _median([m.knee_flex_deg for m in window]),
            "elbow_flex_deg": _median([m.elbow_flex_deg for m in window]),
            "arm_m": _median([m.arm_m for m in window]),
            "wrist_up": _median([
                float(np.dot(m.wrist_mid, m.frame.up)) for m in window if m.wrist_mid is not None
            ]),
        }
        self._standing_points = (measure.points - measure.hip_mid).tolist()

    def _expected_top_up(self) -> float:
        wrist_up = self._standing_refs.get("wrist_up", NAN)
        if math.isfinite(wrist_up):
            return wrist_up - self._grip_offset_m()
        if self._set_top_heights:
            return float(np.median(self._set_top_heights))
        return NAN

    # ------------------------------------------------------------------
    # State machine (PLAN.md §2.3)
    # ------------------------------------------------------------------

    def _held(self, name: str, condition: bool, t: float, duration_s: float) -> bool:
        if not condition:
            self._held_since.pop(name, None)
            return False
        since = self._held_since.setdefault(name, t)
        return t - since >= duration_s - 1e-6

    def _enter(self, phase: DeadliftPhase, t: float) -> None:
        if phase != self.phase:
            logger.debug("[DEADLIFT] %s -> %s at %.3f", self.phase.value, phase.value, t)
        self.phase = phase
        self._phase_since = t
        self._held_since.clear()
        self._top_still_frames = 0
        self._dead_stop_frames = 0

    def _near_and_facing(self, measure: _Measure) -> tuple[bool, bool]:
        cfg = self.config
        if measure.bar_source == BAR_SOURCE_WRIST_PROXY or measure.bar_centre is None:
            # No bar to measure against: being there and standing still is all we know.
            return True, True
        reference = measure.midfoot if measure.midfoot is not None else measure.ankle_mid
        ahead = forward_m(measure.frame, measure.bar_centre, reference)
        sideways = lateral_m(measure.frame, measure.bar_centre, reference)
        near = -cfg.max_bar_behind_m <= ahead <= cfg.stance_max_bar_ahead_m and abs(sideways) <= cfg.stance_max_bar_lateral_m
        facing = True
        if measure.foot_forward is not None and self._rest_axis is not None:
            # Facing the bar: the feet point across it, not along it.
            along = abs(float(np.dot(measure.foot_forward, self._rest_axis)))
            facing = along <= math.cos(math.radians(90.0 - cfg.facing_max_deg))
        return near, facing

    def _step_state_machine(self, measure: _Measure, velocity: float) -> None:
        phase = self.phase
        if phase == DeadliftPhase.APPROACH:
            self._step_approach(measure)
        elif phase == DeadliftPhase.STANCE:
            self._step_stance(measure)
        elif phase == DeadliftPhase.SETUP:
            self._step_setup(measure, velocity)
        elif phase == DeadliftPhase.PULL:
            self._step_pull(measure, velocity)
        elif phase == DeadliftPhase.TOP:
            self._step_top(measure, velocity)
        elif phase == DeadliftPhase.LOWER:
            self._step_lower(measure, velocity)
        elif phase == DeadliftPhase.FLOOR:
            self._step_floor(measure, velocity)

    def _step_approach(self, measure: _Measure) -> None:
        cfg = self.config
        still = self._body_speed_mps() <= cfg.still_speed_mps
        near, facing = self._near_and_facing(measure)
        if self._held("stance", measure.standing and still and near and facing, measure.t, cfg.stance_still_s):
            self._enter(DeadliftPhase.STANCE, measure.t)
            self._stance_midfoots.clear()
            self._stance_bar_offsets.clear()
            return
        # A lifter who walks in and sets straight up skips a visible stance.
        if self._held("setup", measure.hands_on_bar and still and near, measure.t, cfg.setup_still_s):
            self._stance_bar_midfoot_cm = NAN
            self._begin_setup(measure)

    def _step_stance(self, measure: _Measure) -> None:
        cfg = self.config
        still = self._body_speed_mps() <= cfg.still_speed_mps
        if measure.midfoot is not None and still and measure.legs_measured:
            self._stance_midfoots.append(measure.midfoot)
        if measure.midfoot is not None and measure.bar_centre is not None:
            offset_cm = forward_m(measure.frame, measure.bar_centre, measure.midfoot) * 100.0
            self._stance_bar_offsets.append((measure.t, offset_cm))
            while measure.t - self._stance_bar_offsets[0][0] > cfg.setup_window_s:
                self._stance_bar_offsets.popleft()
        near, facing = self._near_and_facing(measure)
        if self._held("away", not (near and facing), measure.t, cfg.walk_away_s):
            self._enter(DeadliftPhase.APPROACH, measure.t)
            return
        if self._held("setup", measure.hands_on_bar and still, measure.t, cfg.setup_still_s):
            self._stance_bar_midfoot_cm = _median([offset for _, offset in self._stance_bar_offsets])
            self._begin_setup(measure)

    def _begin_setup(self, measure: _Measure) -> None:
        if self._stance_midfoots:
            self._locked_midfoot = np.mean(list(self._stance_midfoots), axis=0)
        elif measure.midfoot is not None:
            self._locked_midfoot = measure.midfoot
        # The still hold that qualified this setup is already part of it.
        held_since = self._held_since.get("setup", measure.t)
        self._setup_frames.clear()
        self._enter(DeadliftPhase.SETUP, measure.t)
        for frame in self._history:
            if frame.t >= held_since and frame.hands_on_bar:
                self._record_setup_frame(frame)

    def _record_setup_frame(self, measure: _Measure) -> None:
        self._setup_frames.append(measure)
        while measure.t - self._setup_frames[0].t > self.config.setup_window_s + self.config.liftoff_lookback_s:
            self._setup_frames.popleft()

    def _lifted_off(self, measure: _Measure, velocity: float) -> bool:
        if math.isnan(self._rest_up) or not math.isfinite(measure.bar_up):
            return False
        cfg = self.config
        return measure.bar_up - self._rest_up > cfg.liftoff_rise_m and velocity > cfg.liftoff_velocity_mps

    def _stood_up(self, measure: _Measure) -> bool:
        return self._held("stood", measure.standing and not measure.hands_on_bar, measure.t, self.config.setup_still_s)

    def _leave_floor_area(self, measure: _Measure) -> None:
        near, facing = self._near_and_facing(measure)
        self._enter(DeadliftPhase.STANCE if near and facing else DeadliftPhase.APPROACH, measure.t)
        self._stance_midfoots.clear()
        self._stance_bar_offsets.clear()
        self._locked_midfoot = None

    def _step_setup(self, measure: _Measure, velocity: float) -> None:
        if measure.hands_on_bar:
            self._record_setup_frame(measure)
        if self._lifted_off(measure, velocity):
            self._start_rep(measure, touch_and_go=False, window_s=self.config.setup_window_s)
            return
        if self._stood_up(measure):
            self._leave_floor_area(measure)

    def _step_floor(self, measure: _Measure, velocity: float) -> None:
        cfg = self.config
        if measure.hands_on_bar:
            self._record_setup_frame(measure)
        else:
            self._setup_frames.clear()
        if self._lifted_off(measure, velocity):
            # Quick re-pull: the setup is the last moment before liftoff.
            self._start_rep(measure, touch_and_go=False, window_s=cfg.quick_pull_window_s)
            return
        bar_still = abs(self._bar_still_velocity) <= cfg.liftoff_rest_speed_mps
        if self._held("resetup", measure.hands_on_bar and bar_still, measure.t, cfg.resetup_hold_s):
            setup_frames = list(self._setup_frames)
            self._enter(DeadliftPhase.SETUP, measure.t)
            self._setup_frames = deque(setup_frames)
            return
        if self._stood_up(measure):
            self._leave_floor_area(measure)

    def _back_dated_liftoff(self) -> int:
        """Index into history of motion onset: the last frame in the look-back
        where the bar sat on its rest and was not moving."""
        cfg = self.config
        latest_t = self._history[-1].t
        history = list(self._history)
        lowest = len(history) - 1
        for index in range(len(history) - 1, -1, -1):
            measure = history[index]
            if latest_t - measure.t > cfg.liftoff_lookback_s:
                break
            if not math.isfinite(measure.bar_up):
                continue
            # Heights, not the windowed velocity: on a noisy bar the velocity test
            # reaches back too far, while the bar's height leaves its rest at onset.
            if measure.bar_up - self._rest_up <= cfg.liftoff_rest_band_m:
                return index
            if measure.bar_up < history[lowest].bar_up:
                lowest = index
        return lowest

    def _start_rep(self, measure: _Measure, touch_and_go: bool, window_s: float, liftoff_index: int | None = None) -> None:
        history = list(self._history)
        velocities = list(self._velocities)
        if liftoff_index is None:
            liftoff_index = self._back_dated_liftoff()
        liftoff = history[liftoff_index]
        setup = {} if touch_and_go else self._measure_setup(liftoff.t, window_s)
        if not touch_and_go:
            setup["stance_bar_midfoot_cm"] = self._stance_bar_midfoot_cm
            self._stance_bar_midfoot_cm = NAN
        rep = _Rep(liftoff, touch_and_go, setup)
        for index in range(liftoff_index + 1, len(history)):
            rep.append(history[index], velocities[index])
        self._rep = rep
        self._enter(DeadliftPhase.PULL, measure.t)
        self.rep_started = True
        self._setup_frames.clear()

    def _top_reached(self, rep: _Rep, measure: _Measure) -> bool:
        cfg = self.config
        rise = rep.peak_up - rep.liftoff.bar_up
        if rise < cfg.failed_rep_min_rise_m:
            return False
        expected = self._expected_top_up()
        if math.isfinite(expected) and rep.peak_up < expected - cfg.top_margin_m:
            return False
        return abs(measure.trunk_deg) <= cfg.top_max_trunk_deg

    def _step_pull(self, measure: _Measure, velocity: float) -> None:
        cfg = self.config
        rep = self._rep
        rep.append(measure, velocity)
        if rep.knee_pass is None and measure.knee_mid is not None:
            if math.isfinite(measure.bar_up) and measure.bar_up >= float(np.dot(measure.knee_mid, measure.frame.up)):
                rep.knee_pass = measure
        if abs(velocity) < cfg.top_still_speed_mps:
            self._top_still_frames += 1
        else:
            self._top_still_frames = 0
        if self._top_still_frames >= cfg.top_still_frames and self._top_reached(rep, measure):
            rep.top_time = self._top_arrival(rep)
            rep.top_up = rep.peak_up
            rep.top_frames = [frame for frame in rep.frames if frame.t >= rep.top_time]
            self._enter(DeadliftPhase.TOP, measure.t)
            return
        if velocity < -cfg.lower_velocity_mps and self._top_reached(rep, rep.frames[rep.peak_index]):
            # A top that never held still (a quick touch-and-go set): the peak was the top.
            peak = rep.frames[rep.peak_index]
            rep.top_time = self._top_arrival(rep)
            rep.top_up = rep.peak_up
            rep.top_frames = [frame for frame in rep.frames if abs(frame.t - peak.t) <= TOP_PEAK_WINDOW_S]
            rep.lower_start = measure.t
            self._enter(DeadliftPhase.LOWER, measure.t)
            return
        if self._back_on_floor(measure, velocity):
            rise = rep.peak_up - rep.liftoff.bar_up
            if rise >= cfg.failed_rep_min_rise_m:
                self.failed_reps += 1
                logger.info("[DEADLIFT] Failed rep: the bar rose %.0f cm and came back down", rise * 100.0)
            self._rep = None
            self._enter(DeadliftPhase.FLOOR, measure.t)
            return
        if measure.t - rep.liftoff.t > MAX_PULL_S:
            self._rep = None
            self._enter(DeadliftPhase.FLOOR, measure.t)

    @staticmethod
    def _top_arrival(rep: _Rep) -> float:
        """The top event: the bar's first arrival within TOP_ARRIVAL_BAND_M of its peak."""
        for frame in rep.frames:
            if frame.bar_up >= rep.peak_up - TOP_ARRIVAL_BAND_M:
                return frame.t
        return rep.frames[rep.peak_index].t

    def _step_top(self, measure: _Measure, velocity: float) -> None:
        rep = self._rep
        rep.append(measure, velocity)
        if abs(velocity) < self.config.top_still_speed_mps:
            rep.top_frames.append(measure)
        if velocity < -self.config.lower_velocity_mps:
            rep.lower_start = measure.t
            self._enter(DeadliftPhase.LOWER, measure.t)

    def _back_on_floor(self, measure: _Measure, velocity: float) -> bool:
        """Dead stop: on the rest and not moving for a few frames, or on the rest long enough."""
        cfg = self.config
        if math.isnan(self._rest_up) or not math.isfinite(measure.bar_up):
            return False
        on_rest = measure.bar_up - self._rest_up <= cfg.floor_band_m
        if on_rest and abs(self._bar_still_velocity) < cfg.dead_stop_speed_mps:
            self._dead_stop_frames += 1
        else:
            self._dead_stop_frames = 0
        held = self._held("on_rest", on_rest, measure.t, cfg.dead_stop_hold_s)
        return self._dead_stop_frames >= cfg.dead_stop_frames or held

    def _step_lower(self, measure: _Measure, velocity: float) -> None:
        cfg = self.config
        rep = self._rep
        rep.append(measure, velocity)
        if math.isnan(rep.floor_time) and measure.bar_up - self._rest_up <= cfg.liftoff_rest_band_m:
            # Touchdown: back on the rest itself, not just inside the dead-stop band.
            rep.floor_time = measure.t
        if self._touch_and_go(rep, measure):
            self._finish_touch_and_go(rep, measure)
            return
        if self._back_on_floor(measure, velocity):
            if math.isnan(rep.floor_time):
                rep.floor_time = measure.t
            self._complete_rep(rep, measure)
            self._rep = None
            self._enter(DeadliftPhase.FLOOR, measure.t)

    def _lowest_since_lower(self, rep: _Rep) -> int:
        start = next(
            (index for index, frame in enumerate(rep.frames) if frame.t >= rep.lower_start),
            len(rep.frames) - 1,
        )
        lowest = start
        for index in range(start, len(rep.frames)):
            if rep.frames[index].bar_up < rep.frames[lowest].bar_up:
                lowest = index
        return lowest

    def _touch_and_go(self, rep: _Rep, measure: _Measure) -> bool:
        cfg = self.config
        if not measure.hands_on_bar:
            return False
        low_index = self._lowest_since_lower(rep)
        low = rep.frames[low_index]
        if low.bar_up - self._rest_up > cfg.touch_go_band_m:
            return False
        if measure.bar_up - low.bar_up <= cfg.touch_go_rise_m:
            return False
        # Upward the whole way since the low point, for long enough. Positions, not
        # the windowed velocity, which still reads the descent for a frame or two.
        after = rep.frames[low_index:]
        rising = all(
            later.bar_up >= earlier.bar_up - TOUCH_GO_JITTER_M for earlier, later in zip(after, after[1:])
        )
        return rising and measure.t - low.t >= cfg.touch_go_sustain_s - 1e-6

    def _finish_touch_and_go(self, rep: _Rep, measure: _Measure) -> None:
        low_index = self._lowest_since_lower(rep)
        low = rep.frames[low_index]
        if math.isnan(rep.floor_time) or rep.floor_time > low.t:
            rep.floor_time = low.t
        next_frames = rep.frames[low_index:]
        next_velocities = rep.velocities[low_index:]
        rep.frames = rep.frames[:low_index + 1]
        rep.velocities = rep.velocities[:low_index + 1]
        self._complete_rep(rep, low)
        next_rep = _Rep(next_frames[0], touch_and_go=True, setup={})
        for frame, velocity in zip(next_frames[1:], next_velocities[1:]):
            next_rep.append(frame, velocity)
        self._rep = next_rep
        self._enter(DeadliftPhase.PULL, measure.t)
        self.rep_started = True

    # ------------------------------------------------------------------
    # Features (PLAN.md §2.5)
    # ------------------------------------------------------------------

    def _measure_setup(self, liftoff_t: float, window_s: float) -> dict[str, float]:
        frames = [
            frame for frame in self._setup_frames
            if liftoff_t - window_s - 1e-6 <= frame.t <= liftoff_t and frame.legs_measured
        ]
        if len(frames) < self.config.min_setup_frames:
            return {}
        midfoot = self._locked_midfoot
        setup: dict[str, float] = {
            "hip_height_cm": _median([height_m(f.frame, f.hip_mid, f.ankle_mid) * 100.0 for f in frames]),
            "trunk_deg": _median([f.trunk_deg for f in frames]),
        }
        bar_frames = [f for f in frames if f.bar_centre is not None]
        if bar_frames:
            setup["shoulder_vs_bar_cm"] = _median([
                forward_m(f.frame, f.shoulder_mid, f.bar_centre) * 100.0 for f in bar_frames
            ])
            if midfoot is not None:
                setup["bar_midfoot_cm"] = _median([
                    forward_m(f.frame, f.bar_centre, midfoot) * 100.0 for f in bar_frames
                ])
            measured_bar = [f for f in bar_frames if f.bar_source == BAR_SOURCE_BAR and f.wrist_mid is not None]
            if measured_bar:
                offset = _median([height_m(f.frame, f.wrist_mid, f.bar_centre) for f in measured_bar])
                cfg = self.config
                if math.isfinite(offset):
                    self._learned_grip_offset_m = min(
                        max(offset, cfg.min_wrist_to_bar_offset_m), cfg.max_wrist_to_bar_offset_m,
                    )
            prediction = self._predict_setup(bar_frames, setup.get("shoulder_vs_bar_cm", NAN))
            if prediction is not None:
                setup["band_low_cm"] = prediction.hip_band_low_m * 100.0
                setup["band_high_cm"] = prediction.hip_band_high_m * 100.0
                setup["predicted_trunk_change_deg"] = prediction.trunk_change_deg
        return setup

    def _segment_lengths(self, frames: list[_Measure]) -> AthleteSegments | None:
        def projected(lower: np.ndarray | None, upper: np.ndarray | None, frame: SagittalFrame) -> float:
            if lower is None or upper is None:
                return NAN
            return math.hypot(forward_m(frame, upper, lower), height_m(frame, upper, lower))

        tibia = _median([projected(f.ankle_mid, f.knee_mid, f.frame) for f in frames])
        femur = _median([projected(f.knee_mid, f.hip_mid, f.frame) for f in frames])
        torso = _median([projected(f.hip_mid, f.shoulder_mid, f.frame) for f in frames])
        arm = self._standing_refs.get("arm_m", NAN)
        if not math.isfinite(arm):
            arm = _median([f.arm_m for f in frames])
        seeded = self._seeded_segments
        if not math.isfinite(torso):
            torso = seeded.get("torso_avg_m", NAN)
        lengths = (tibia, femur, torso, arm)
        if not all(math.isfinite(length) and MIN_SEGMENT_M <= length <= MAX_SEGMENT_M for length in lengths):
            return None
        return AthleteSegments(
            tibia_m=tibia, femur_m=femur, torso_m=torso, arm_m=arm, grip_offset_m=self._grip_offset_m(),
        )

    def _shin_bar_distance_m(self, frames: list[_Measure]) -> float:
        """How far in front of this lifter's shin line the bar sits at setup.

        Shin thickness and how hard the shins press the bar vary by person; with
        an assumed distance, 2 cm of difference moved the model's hip band ~7 cm
        and its trunk prediction ~7 deg (simulator), faking D4 and D2 on a clean
        setup. Measured on the setup frames and clipped to what a shin allows.
        """
        cfg = self.config
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
            return cfg.shin_bar_distance_m
        return min(max(measured, cfg.min_shin_bar_distance_m), cfg.max_shin_bar_distance_m)

    def _predict_setup(self, frames: list[_Measure], shoulder_vs_bar_cm: float) -> SetupPrediction | None:
        segments = self._segment_lengths(frames)
        if segments is None:
            return None
        cfg = self.config
        bar_forward = _median([forward_m(f.frame, f.bar_centre, f.ankle_mid) for f in frames])
        bar_height = _median([height_m(f.frame, f.bar_centre, f.ankle_mid) for f in frames])
        prediction = predict_setup(
            segments,
            bar_forward_m=bar_forward,
            bar_height_m=bar_height,
            shoulder_band_m=(cfg.shoulder_band_low_m, cfg.shoulder_band_high_m),
            shoulder_ahead_m=shoulder_vs_bar_cm / 100.0,
            shin_bar_m=self._shin_bar_distance_m(frames),
        )
        if prediction is None:
            logger.info(
                "[DEADLIFT] No setup fits these segments (tibia %.2f, femur %.2f, torso %.2f, arm %.2f m): "
                "setup hips and shoulders are not judged for this athlete",
                segments.tibia_m, segments.femur_m, segments.torso_m, segments.arm_m,
            )
        return prediction

    def _complete_rep(self, rep: _Rep, end: _Measure) -> None:
        if math.isnan(rep.top_time):
            return
        self.rep_count += 1
        self._set_top_heights.append(rep.top_up)
        features = self._rep_features(rep)
        self._completed.append(_CompletedRep(
            features=features,
            start_time=rep.liftoff.t,
            end_time=end.t,
            start_frame=rep.liftoff.frame_index,
            end_frame=end.frame_index,
        ))
        logger.info(
            "[DEADLIFT] Rep %d counted%s: %.0f cm in %.2f s",
            self.rep_count, " (touch-and-go)" if rep.touch_and_go else "",
            features.bar_rise_cm, features.pull_time_s,
        )

    def _rep_features(self, rep: _Rep) -> DeadliftRepFeatures:
        pull = [frame for frame in rep.frames if frame.t <= rep.top_time]
        liftoff = rep.liftoff
        setup = rep.setup
        refs = self._standing_refs
        bar_source = BAR_SOURCE_WRIST_PROXY if any(f.bar_source == BAR_SOURCE_WRIST_PROXY for f in pull) else BAR_SOURCE_BAR

        features = DeadliftRepFeatures(
            rep_number=self.rep_count,
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
            gravity_source=self.gravity_source,
            grip=self._grip,
            liftoff_time=liftoff.t,
            top_time=rep.top_time,
            floor_time=rep.floor_time,
        )
        rise_m = rep.top_up - liftoff.bar_up
        pull_time = rep.top_time - liftoff.t
        features.bar_rise_cm = rise_m * 100.0
        features.pull_time_s = pull_time
        if pull_time > 0.0:
            features.concentric_velocity_mps = rise_m / pull_time
        if math.isfinite(rep.lower_start) and math.isfinite(rep.floor_time):
            features.lower_time_s = rep.floor_time - rep.lower_start

        self._coordination_features(rep, features)
        self._bar_path_features(pull, features)
        self._top_features(rep, refs, features)
        self._hip_shift_feature(pull, features)
        if refs and math.isfinite(refs.get("elbow_flex_deg", NAN)):
            features.elbow_flexion_deg = _percentile(
                [f.elbow_flex_deg - refs["elbow_flex_deg"] for f in pull], WORST_PERCENTILE,
            )
        return features

    def _coordination_features(self, rep: _Rep, features: DeadliftRepFeatures) -> None:
        knee_pass = rep.knee_pass
        liftoff = rep.liftoff
        if knee_pass is None or math.isnan(rep.top_time) or knee_pass.t > rep.top_time:
            return
        features.knee_pass_time = knee_pass.t
        change = knee_pass.trunk_deg - liftoff.trunk_deg
        predicted = features.trunk_change_predicted_deg
        # Without the setup model the raw change is judged: the trunk must still
        # not tip further forward off the floor.
        features.trunk_change_liftoff_knee_deg = change - predicted if math.isfinite(predicted) else change
        frame = knee_pass.frame
        hip_rise = height_m(frame, knee_pass.hip_mid, liftoff.hip_mid)
        shoulder_rise = height_m(frame, knee_pass.shoulder_mid, liftoff.shoulder_mid)
        if shoulder_rise > MIN_SHOULDER_RISE_M:
            features.hip_shoulder_rise_ratio = hip_rise / shoulder_rise

    def _bar_path_features(self, pull: list[_Measure], features: DeadliftRepFeatures) -> None:
        bar_frames = [f for f in pull if f.bar_centre is not None]
        if len(bar_frames) < 2:
            return
        start = bar_frames[0]
        drift = [
            max(0.0, forward_m(f.frame, f.bar_centre, start.bar_centre)) * 100.0 for f in bar_frames
        ]
        features.bar_drift_cm = _percentile(drift, WORST_PERCENTILE)
        # Left end above the right end (cm): the plate hubs, or the hands without bar tracking.
        tilts: list[float] = []
        for frame in bar_frames:
            if frame.bar_left is not None and frame.bar_right is not None:
                tilts.append(height_m(frame.frame, frame.bar_left, frame.bar_right) * 100.0)
            elif frame.bar_source == BAR_SOURCE_WRIST_PROXY and frame.l_wrist is not None and frame.r_wrist is not None:
                tilts.append(height_m(frame.frame, frame.l_wrist, frame.r_wrist) * 100.0)
        if tilts:
            features.bar_tilt_cm = _percentile([abs(tilt) for tilt in tilts], WORST_PERCENTILE)
            features.bar_low_side = "left" if _median(tilts) < 0.0 else "right"

    def _top_features(self, rep: _Rep, refs: dict[str, float], features: DeadliftRepFeatures) -> None:
        top = [f for f in rep.top_frames if f.legs_measured] or rep.top_frames
        if not top or not refs:
            return
        features.hip_extension_deficit_deg = _median([f.hip_flex_deg for f in top]) - refs.get("hip_flex_deg", NAN)
        features.knee_extension_deficit_deg = _median([f.knee_flex_deg for f in top]) - refs.get("knee_flex_deg", NAN)
        features.lean_back_deg = refs.get("trunk_deg", NAN) - _median([f.trunk_deg for f in top])

    def _hip_shift_feature(self, pull: list[_Measure], features: DeadliftRepFeatures) -> None:
        midfoot = self._locked_midfoot
        ratios: list[float] = []
        for frame in pull:
            if frame.l_ankle is None or frame.r_ankle is None:
                continue
            separation = abs(lateral_m(frame.frame, frame.r_ankle, frame.l_ankle))
            if separation < MIN_ANKLE_SEPARATION_M:
                continue
            reference = midfoot if midfoot is not None else frame.ankle_mid
            ratios.append(lateral_m(frame.frame, frame.hip_mid, reference) / separation)
        if len(ratios) < SHIFT_START_FRAMES:
            return
        start = float(np.median(ratios[:SHIFT_START_FRAMES]))
        deviations = np.asarray(ratios) - start
        cutoff = np.percentile(np.abs(deviations), SHIFT_SUSTAINED_PERCENTILE)
        sustained = deviations[np.abs(deviations) >= cutoff]
        features.hip_shift_ratio = float(np.median(sustained))

    # ------------------------------------------------------------------
    # Frame status
    # ------------------------------------------------------------------

    def _frame_status(self, measure: _Measure) -> DeadliftFrameStatus:
        live_cm = NAN
        if self.phase == DeadliftPhase.STANCE and measure.midfoot is not None and measure.bar_centre is not None:
            live_cm = forward_m(measure.frame, measure.bar_centre, measure.midfoot) * 100.0
        height_cm = NAN
        if math.isfinite(self._rest_up) and math.isfinite(measure.bar_up):
            height_cm = (measure.bar_up - self._rest_up) * 100.0
        return DeadliftFrameStatus(
            phase=self.phase,
            bar_midfoot_live_cm=live_cm,
            bar_source=measure.bar_source,
            gravity_source=self.gravity_source,
            bar_height_cm=height_cm,
        )
