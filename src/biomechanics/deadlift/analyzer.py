"""The deadlift rep analyser: a bar-driven state machine (PLAN.md §2.3) over the
per-frame measurements of measure.py, handing each counted rep's features
(features.py) to the fault rules.

It owns all deadlift per-frame state, so the profile stays stateless and the squat
pipeline state (setup snapshot, standing reference, trajectory) is never touched.
Input is the lagged world-frame skeleton (Y-down metres) with the bar state matched
to its capture time. Everything is measured in the sagittal frame (up = measured
gravity), as differences: bar vs its resting height, bar vs midfoot, joints vs
their standing reference.

Gates are built for keypoints as noisy as the platform's (1.5-2.5 cm, correlated
by the Kalman): stillness is a displacement over half a second, holds survive
short dropouts, and a pull counts from any phase where the hands are on the bar.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from biomechanics.config import DeadliftConfig
from biomechanics.utils.types import CocoKeypoints as CK

from .features import AthleteState, RepTrack, measure_setup, rep_features
from .frame import default_up, forward_m, lateral_m
from .measure import (
    FOOT_KEYPOINTS,
    DeadliftFrameInput,
    FrameMeasure,
    MeasureContext,
    feet_measured,
    measure_frame,
)
from .types import (
    BAR_SOURCE_BAR,
    BAR_SOURCE_WRIST_PROXY,
    GRAVITY_SOURCE_BODY,
    NAN,
    BarState3D,
    DeadliftFrameStatus,
    DeadliftPhase,
    DeadliftRepFeatures,
)

if TYPE_CHECKING:
    from .rep_counter import DeadliftRepCounter

__all__ = ["DeadliftFrameInput", "DeadliftRepAnalyzer"]

logger = logging.getLogger(__name__)

# Frames of history kept: longer than the liftoff look-back and any setup window.
HISTORY_S = 3.0
# Body speed needs this many frames in its window; fewer reads as moving.
MIN_SPEED_FRAMES = 6
# The bar counts as still when its height's slope over this window is small.
STILL_BAR_WINDOW_S = 0.3
# The bar velocity's noise the window is widened for, and the widest window.
VELOCITY_NOISE_MPS = 0.03
MAX_VELOCITY_WINDOW_S = 0.4
# Frames needed for a curvature-free noise estimate of the bar's height.
MIN_NOISE_FRAMES = 5
# The top event is the bar's arrival this close to its peak height.
TOP_ARRIVAL_BAND_M = 0.005
# Frames either side of the peak that stand in for a top that never held still.
TOP_PEAK_WINDOW_S = 0.1
# The top hold: frames in TOP with the bar this close to its top height (or two
# noise bands, on a noisier bar).
TOP_HOLD_BAND_M = 0.02
# A bar this far above the top it held (or the hold band, if wider) was stalled
# at a hitch, not at the top: the pull goes on.
TOP_RESUME_RISE_M = 0.05
# The live bar-over-midfoot offset the foot guidance speaks from is the median
# over this long, once it holds this many settled stance frames.
LIVE_OFFSET_WINDOW_S = 1.0
LIVE_OFFSET_MIN_FRAMES = 10
# D2's set-level evidence: the median rise ratio over this many recent reps.
SET_RISE_RATIO_REPS = 3
# The lifter's left-right is locked from up to this many frames at the bar (8 s:
# stance, setup, and the floor between reps), or from this much history when
# there are none. The hips travel ~45 cm forward in a pull, so each degree off
# reads ~0.03 of hip shift: under correlated keypoint noise the hip and ankle
# lines need seconds of frames (1.2 deg from 4 s, 2 deg from 1 s).
LATERAL_LOCK_FRAMES = 240
LATERAL_LOCK_WINDOW_S = 0.5
# Rest height is the median of this many recent still frames of the bar (1 s):
# the first, still-slow frames of a grind never move it.
REST_FRAMES = 30
# Rest frames needed before their scatter estimates the bar's height noise.
MIN_NOISE_REST_FRAMES = 5
# MAD to standard deviation, for normal noise.
MAD_TO_STD = 1.4826
# Event bands (liftoff, touchdown, top arrival) are this many noise deviations
# wide, never under liftoff_rest_band_m: a noisy bar crosses a fixed band at random.
EVENT_NOISE_BANDS = 3.0
# A rep never lasts longer than this; a stuck PULL is abandoned.
MAX_PULL_S = 15.0
_NOT_IN_REP = (DeadliftPhase.APPROACH, DeadliftPhase.STANCE, DeadliftPhase.SETUP, DeadliftPhase.FLOOR)


class _CompletedRep(NamedTuple):
    features: DeadliftRepFeatures
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int


# The lifter's left-to-right on one frame, horizontal: hips and ankles, plus the
# wrists when asked (the widest baseline, the hands on the bar, then weighs most).
def _lateral_line(measure: FrameMeasure, with_wrists: bool) -> np.ndarray:
    line = measure.points[CK.RIGHT_HIP] - measure.points[CK.LEFT_HIP]
    pairs = [(measure.l_ankle, measure.r_ankle)]
    if with_wrists:
        pairs.append((measure.l_wrist, measure.r_wrist))
    for left, right in pairs:
        if left is not None and right is not None:
            line = line + (right - left)
    up = measure.frame.up
    return line - float(np.dot(line, up)) * up


def _locked_axis(frames: list[FrameMeasure], with_wrists: bool) -> np.ndarray:
    line = np.sum([_lateral_line(frame, with_wrists) for frame in frames], axis=0)
    return line / float(np.linalg.norm(line))


def _median(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.median(finite)) if finite else NAN


# Least-squares slope and its standard error. The noise behind the error is the
# scatter about a parabola, so a bar turning around does not read as noise.
def _slope_and_error(times: list[float], values: list[float]) -> tuple[float, float]:
    if len(times) < 2:
        return 0.0, 0.0
    time_array = np.asarray(times) - times[-1]
    value_array = np.asarray(values)
    centred = time_array - time_array.mean()
    variance = float(np.dot(centred, centred))
    if variance <= 0.0:
        return 0.0, 0.0
    slope = float(np.dot(centred, value_array - value_array.mean()) / variance)
    if len(times) < MIN_NOISE_FRAMES:
        return slope, 0.0
    residuals = value_array - np.polyval(np.polyfit(time_array, value_array, 2), time_array)
    noise = math.sqrt(float(np.dot(residuals, residuals)) / (len(times) - 3))
    return slope, noise / math.sqrt(variance)


class DeadliftRepAnalyzer:
    """Feed observe() once per analysis frame; completed reps queue up for the
    counter (rep data) and the pipeline (finish_rep features)."""

    def __init__(self, config: DeadliftConfig | None = None) -> None:
        self.config = config or DeadliftConfig()
        self._up = default_up()
        self.gravity_source = GRAVITY_SOURCE_BODY
        self._seeded_segments: dict[str, float] = {}
        self._learned_grip_offset_m = NAN
        self._rep_counter: DeadliftRepCounter | None = None
        self._grip = ""
        self._bar_frames = 0
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
        self._log_set_health()
        self.phase = DeadliftPhase.APPROACH
        self._phase_since = NAN
        self._history: deque[FrameMeasure] = deque()
        self._velocities: deque[float] = deque()
        self._held_since: dict[str, float] = {}
        self._held_seen: dict[str, float] = {}
        self._rest_heights: deque[float] = deque(maxlen=REST_FRAMES)
        self._rest_up = NAN
        self._rest_noise_m = 0.0
        self._rest_source: str | None = None
        self._rest_axis: np.ndarray | None = None
        self._last_bar: BarState3D | None = None
        self._last_bar_t = -math.inf
        self._planted_feet: dict[int, np.ndarray] = {}
        self._stance_midfoots: deque[np.ndarray] = deque(maxlen=self.config.midfoot_lock_frames)
        # Frames at the bar the lifter's left-right is locked from.
        self._lateral_frames: deque[FrameMeasure] = deque(maxlen=LATERAL_LOCK_FRAMES)
        # Hips, ankles and wrists: the sagittal frame's left-right without a bar axis.
        self._locked_lateral: np.ndarray | None = None
        # Hips and ankles: the axis sideways hip shift (D8) is measured along, on
        # either bar source (the bar's own axis turns forward travel into
        # sideways when the stance is a few degrees off square to it).
        self._locked_body_lateral: np.ndarray | None = None
        # Midfoots of feet measured since the last dead stop; None outside a
        # re-setup at the floor.
        self._floor_midfoots: deque[np.ndarray] | None = None
        self._stance_bar_offsets: deque[tuple[float, float]] = deque()
        self._locked_midfoot: np.ndarray | None = None
        self._setup_frames: deque[FrameMeasure] = deque()
        self._standing_run: list[FrameMeasure] = []
        self._rep: RepTrack | None = None
        self._completed: deque[_CompletedRep] = deque()
        self._awaiting_finish: deque[DeadliftRepFeatures] = deque()
        self._set_top_heights: list[float] = []
        self._set_predicted_change_deg = NAN
        self._set_rise_ratios: list[float] = []
        self._stance_bar_midfoot_cm = NAN
        self._top_still_frames = 0
        self._dead_stop_frames = 0
        self._bar_slope_mps = 0.0
        self._bar_slope_error_mps = 0.0
        self._bar_residuals_px: list[float] = []
        self._bar_frames = 0
        self._bar_predicted_frames = 0
        self._last_measure: FrameMeasure | None = None
        self._is_settled = False
        self.rep_started = False
        self._status = DeadliftFrameStatus(phase=self.phase, gravity_source=self.gravity_source)

    # ------------------------------------------------------------------
    # Public read-outs
    # ------------------------------------------------------------------

    @property
    def in_rep(self) -> bool:
        return self.phase in (DeadliftPhase.PULL, DeadliftPhase.TOP, DeadliftPhase.LOWER)

    @property
    def rep_counter(self) -> DeadliftRepCounter:
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
        t = frame_input.timestamp
        bar = frame_input.bar
        if bar is not None:
            self._last_bar = bar
            self._last_bar_t = t
            self._record_bar_health(bar)
        # A bar missing for a frame or two is a tracking gap, not a lost bar.
        bar_recent = t - self._last_bar_t <= self.config.max_bar_gap_s
        if self._rest_source is not None and self.phase == DeadliftPhase.APPROACH:
            if bar_recent != (self._rest_source == BAR_SOURCE_BAR):
                # Away from the bar the other source (bar found or lost) may
                # define the rest afresh.
                self._clear_rest()
        context = MeasureContext(
            up=self._up,
            rest_source=self._rest_source,
            # Without a bar axis, the lifter's left-right is locked per set: rebuilt
            # from the noisy hip line every frame, it turned forward travel into
            # sideways hip shift.
            rest_axis=self._rest_axis if self._rest_axis is not None else self._locked_lateral,
            grip_offset_m=self._grip_offset_m(),
            carried_bar=self._last_bar if bar is None and bar_recent else None,
            # Feet stay planted once the lifter is at the bar: the plates hiding
            # them through the pull hide nothing that moved.
            planted_feet=self._planted_feet if self.phase != DeadliftPhase.APPROACH else None,
        )
        measure = measure_frame(frame_input, context, self.config)
        if measure is None:
            return
        if self._history and measure.t <= self._history[-1].t:
            return
        self._plant_feet(frame_input)
        self._history.append(measure)
        while self._history and measure.t - self._history[0].t > HISTORY_S:
            self._history.popleft()
        velocity = self._bar_velocity(self._velocity_window_s())
        self._velocities.append(velocity)
        while len(self._velocities) > len(self._history):
            self._velocities.popleft()
        self._bar_slope_mps, self._bar_slope_error_mps = self._bar_slope(STILL_BAR_WINDOW_S)
        if math.isnan(self._phase_since):
            self._phase_since = measure.t
        self._last_measure = measure
        self._is_settled = (
            self.phase in (DeadliftPhase.APPROACH, DeadliftPhase.STANCE)
            and self._body_speed_mps() <= self.config.settled_speed_mps
        )

        self._update_rest(measure)
        self._update_standing_reference(measure)
        self._step_state_machine(measure, velocity)
        self._status = self._frame_status(measure)

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def _plant_feet(self, frame_input: DeadliftFrameInput) -> None:
        measured = feet_measured(np.asarray(frame_input.confidences))
        points = np.asarray(frame_input.points, dtype=np.float64)
        for index in FOOT_KEYPOINTS:
            if measured[index]:
                self._planted_feet[index] = points[index].copy()

    def _grip_offset_m(self) -> float:
        if math.isfinite(self._learned_grip_offset_m):
            return self._learned_grip_offset_m
        return self.config.wrist_to_bar_offset_m

    def _recent_bar_heights(self, window_s: float) -> tuple[list[float], list[float]]:
        latest = self._history[-1]
        times: list[float] = []
        heights: list[float] = []
        for measure in reversed(self._history):
            if latest.t - measure.t > window_s + 1e-6:
                break
            if math.isfinite(measure.bar_up):
                times.append(measure.t)
                heights.append(measure.bar_up)
        times.reverse()
        heights.reverse()
        return times, heights

    # The configured window, widened on a noisy bar (the wrist proxy) until the
    # slope's noise falls to VELOCITY_NOISE_MPS: a least-squares slope over n
    # frames has standard error noise / (dt * sqrt(n (n^2 - 1) / 12)).
    def _velocity_window_s(self) -> float:
        base_s = self.config.velocity_window_s
        if self._rest_noise_m <= 0.0 or len(self._history) < 2:
            return base_s
        frame_s = (self._history[-1].t - self._history[0].t) / (len(self._history) - 1)
        if frame_s <= 0.0:
            return base_s
        frames = (12.0 * (self._rest_noise_m / (VELOCITY_NOISE_MPS * frame_s)) ** 2) ** (1.0 / 3.0)
        return min(max(base_s, frames * frame_s), MAX_VELOCITY_WINDOW_S)

    def _bar_velocity(self, window_s: float) -> float:
        times, heights = self._recent_bar_heights(window_s)
        return _slope_and_error(times, heights)[0]

    def _bar_slope(self, window_s: float) -> tuple[float, float]:
        return _slope_and_error(*self._recent_bar_heights(window_s))

    # The bar's slope is under speed_mps, or under what its own noise allows: the
    # wrists as a proxy bar jitter far more than a tracked bar.
    def _bar_still(self, speed_mps: float) -> bool:
        allowed = max(speed_mps, self.config.still_noise_factor * self._bar_slope_error_mps)
        return abs(self._bar_slope_mps) <= allowed

    # Hips and shoulders: displacement of the median position between the first
    # and last thirds of the window, over the time between them. Medians of
    # positions, not a slope of noisy keypoints, so standing reads standing.
    def _body_speed_mps(self) -> float:
        latest = self._history[-1]
        window = [m for m in self._history if latest.t - m.t <= self.config.body_speed_window_s + 1e-6]
        if len(window) < MIN_SPEED_FRAMES:
            return math.inf
        part = len(window) // 3
        first, last = window[:part], window[-part:]
        elapsed = float(np.median([m.t for m in last]) - np.median([m.t for m in first]))
        if elapsed <= 0.0:
            return math.inf
        speeds = []
        for attribute in ("hip_mid", "shoulder_mid"):
            before = np.median([getattr(m, attribute) for m in first], axis=0)
            after = np.median([getattr(m, attribute) for m in last], axis=0)
            speeds.append(float(np.linalg.norm(after - before)) / elapsed)
        return max(speeds)

    # ------------------------------------------------------------------
    # References
    # ------------------------------------------------------------------

    # The bar's resting height: still and not in a rep. Without bar tracking it is
    # the wrists' height while hinged over the bar.
    def _update_rest(self, measure: FrameMeasure) -> None:
        if self.in_rep or not math.isfinite(measure.bar_up):
            return
        if not self._bar_still(self.config.liftoff_rest_speed_mps):
            return
        if measure.bar_source == BAR_SOURCE_WRIST_PROXY:
            if self.phase == DeadliftPhase.APPROACH or not measure.hands_on_bar:
                return
        elif measure.hands_on_bar and self.phase not in (DeadliftPhase.SETUP, DeadliftPhase.FLOOR):
            # Someone holding the bar off the floor outside a set-up is not a rest.
            return
        self._rest_heights.append(measure.bar_up)
        self._rest_up = float(np.median(self._rest_heights))
        if len(self._rest_heights) >= MIN_NOISE_REST_FRAMES:
            deviations = np.abs(np.asarray(self._rest_heights) - self._rest_up)
            self._rest_noise_m = MAD_TO_STD * float(np.median(deviations))
        self._rest_source = measure.bar_source
        if measure.bar_left is not None and measure.bar_right is not None:
            axis = measure.bar_right - measure.bar_left
            self._rest_axis = axis / float(np.linalg.norm(axis))

    def _clear_rest(self) -> None:
        self._rest_heights.clear()
        self._rest_up = NAN
        self._rest_noise_m = 0.0
        self._rest_source = None

    # Standing settled with the arms hanging: the angles lockout, lean-back and
    # bent arms are measured against, and the height the bar reaches at the top.
    def _update_standing_reference(self, measure: FrameMeasure) -> None:
        if self.phase not in (DeadliftPhase.APPROACH, DeadliftPhase.STANCE):
            self._standing_run = []
            return
        if not (measure.standing and measure.legs_measured and self._is_settled):
            # A noisy knee angle on one frame is not the lifter moving.
            if self._standing_run and measure.t - self._standing_run[-1].t > self.config.hold_grace_s:
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

    def _event_band_m(self) -> float:
        return max(self.config.liftoff_rest_band_m, EVENT_NOISE_BANDS * self._rest_noise_m)

    def _athlete(self) -> AthleteState:
        return AthleteState(
            standing_refs=self._standing_refs,
            seeded_segments=self._seeded_segments,
            grip_offset_m=self._grip_offset_m(),
        )

    # ------------------------------------------------------------------
    # Set health (PLAN.md §3.6)
    # ------------------------------------------------------------------

    def _record_bar_health(self, bar: BarState3D) -> None:
        self._bar_frames += 1
        if bar.predicted:
            self._bar_predicted_frames += 1
        elif math.isfinite(bar.residual_px):
            self._bar_residuals_px.append(bar.residual_px)

    # One line per set on how well the bar was tracked: the share of predicted
    # states and the reprojection residual of the measured ones.
    def _log_set_health(self) -> None:
        if not self._bar_frames:
            return
        residuals = self._bar_residuals_px
        logger.info(
            "[DEADLIFT] Set bar health: %d states, %.0f%% predicted, residual median %.1f px, p95 %.1f px",
            self._bar_frames,
            100.0 * self._bar_predicted_frames / self._bar_frames,
            float(np.median(residuals)) if residuals else NAN,
            float(np.percentile(residuals, 95.0)) if residuals else NAN,
        )

    # ------------------------------------------------------------------
    # State machine (PLAN.md §2.3)
    # ------------------------------------------------------------------

    # condition true for duration_s; a lapse shorter than the grace (one noisy
    # frame, a dropped keypoint) does not restart the clock.
    def _held(self, name: str, condition: bool, t: float, duration_s: float) -> bool:
        if condition:
            since = self._held_since.setdefault(name, t)
            self._held_seen[name] = t
            return t - since >= duration_s - 1e-6
        last_seen = self._held_seen.get(name)
        if last_seen is None or t - last_seen > self.config.hold_grace_s:
            self._held_since.pop(name, None)
            self._held_seen.pop(name, None)
        return False

    def _enter(self, phase: DeadliftPhase, t: float) -> None:
        if phase != self.phase:
            logger.debug("[DEADLIFT] %s -> %s at %.3f", self.phase.value, phase.value, t)
        if phase == DeadliftPhase.APPROACH:
            self._planted_feet = {}
            self._stance_midfoots.clear()
            self._lateral_frames.clear()
            self._stance_bar_offsets.clear()
            self._locked_lateral = None
            self._locked_body_lateral = None
        if phase == DeadliftPhase.FLOOR:
            # A re-setup at the floor may move the feet: the next liftoff locks
            # the midfoot afresh from what follows.
            self._floor_midfoots = deque(maxlen=self.config.midfoot_lock_frames)
        self.phase = phase
        self._phase_since = t
        self._held_since.clear()
        self._held_seen.clear()
        self._top_still_frames = 0
        self._dead_stop_frames = 0

    def _near_and_facing(self, measure: FrameMeasure) -> tuple[bool, bool]:
        cfg = self.config
        if measure.bar_source == BAR_SOURCE_WRIST_PROXY or measure.bar_centre is None:
            # No bar to measure against: being there and standing is all we know.
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

    def _step_state_machine(self, measure: FrameMeasure, velocity: float) -> None:
        phase = self.phase
        if phase in _NOT_IN_REP and measure.hands_on_bar:
            self._record_setup_frame(measure)
        if phase == DeadliftPhase.APPROACH:
            self._step_approach(measure, velocity)
        elif phase == DeadliftPhase.STANCE:
            self._step_stance(measure, velocity)
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

    # Grip and rip: the bar left its rest in the lifter's hands before any visible
    # setup hold. The setup is then the last moment before liftoff.
    def _pulled_without_setup(self, measure: FrameMeasure, velocity: float) -> bool:
        if not (measure.hands_on_bar and self._lifted_off(measure, velocity)):
            return False
        self._lock_midfoot(measure)
        self._start_rep(measure, touch_and_go=False, window_s=self.config.quick_pull_window_s)
        return True

    def _step_approach(self, measure: FrameMeasure, velocity: float) -> None:
        cfg = self.config
        near, facing = self._near_and_facing(measure)
        if near and self._pulled_without_setup(measure, velocity):
            return
        if self._held("stance", measure.standing and near and facing and self._is_settled, measure.t, cfg.stance_still_s):
            self._enter(DeadliftPhase.STANCE, measure.t)
            self._stance_midfoots.clear()
            self._stance_bar_offsets.clear()
            return
        # A lifter who walks in and sets straight up skips a visible stance.
        if self._held("setup", measure.hands_on_bar and near, measure.t, cfg.setup_hold_s):
            self._begin_setup(measure)

    def _step_stance(self, measure: FrameMeasure, velocity: float) -> None:
        cfg = self.config
        if measure.midfoot is not None and measure.feet_measured and measure.legs_measured and self._is_settled:
            self._stance_midfoots.append(measure.midfoot)
        if measure.legs_measured and self._is_settled:
            self._lateral_frames.append(measure)
        if (
            measure.midfoot is not None and measure.bar_centre is not None
            and measure.bar_source == BAR_SOURCE_BAR and self._is_settled
        ):
            offset_cm = forward_m(measure.frame, measure.bar_centre, measure.midfoot) * 100.0
            self._stance_bar_offsets.append((measure.t, offset_cm))
            while measure.t - self._stance_bar_offsets[0][0] > LIVE_OFFSET_WINDOW_S:
                self._stance_bar_offsets.popleft()
        if self._pulled_without_setup(measure, velocity):
            return
        near, facing = self._near_and_facing(measure)
        if self._held("away", not (near and facing), measure.t, cfg.walk_away_s):
            self._enter(DeadliftPhase.APPROACH, measure.t)
            return
        if self._held("setup", measure.hands_on_bar, measure.t, cfg.setup_hold_s):
            self._begin_setup(measure)

    # The midfoot every setup offset is measured from, and the stance's own
    # bar-over-midfoot reading (D1 at stance).
    def _lock_midfoot(self, measure: FrameMeasure) -> None:
        if self._stance_midfoots:
            self._locked_midfoot = np.mean(list(self._stance_midfoots), axis=0)
        elif measure.midfoot is not None:
            self._locked_midfoot = measure.midfoot
        # The last settled stance frames, before the hinge to the bar began.
        self._stance_bar_midfoot_cm = _median([offset for _, offset in self._stance_bar_offsets])
        self._lock_lateral(measure)

    # The lifter's left-right from the frames at the bar, or the last half
    # second's: for D8 always, and for the sagittal frame when no bar axis gives it.
    def _lock_lateral(self, measure: FrameMeasure) -> None:
        frames = list(self._lateral_frames) or [
            frame for frame in self._history if measure.t - frame.t <= LATERAL_LOCK_WINDOW_S
        ]
        self._locked_body_lateral = _locked_axis(frames, with_wrists=False)
        if self._rest_axis is None:
            self._locked_lateral = _locked_axis(frames, with_wrists=True)

    # A re-setup at the floor: the midfoot of the feet measured since the dead
    # stop, if any, and the lifter's left-right with the floor frames added.
    def _relock_at_floor(self, measure: FrameMeasure) -> None:
        if self._floor_midfoots:
            self._locked_midfoot = np.mean(list(self._floor_midfoots), axis=0)
        self._floor_midfoots = None
        self._lock_lateral(measure)

    def _begin_setup(self, measure: FrameMeasure) -> None:
        self._lock_midfoot(measure)
        self._floor_midfoots = None
        self._enter(DeadliftPhase.SETUP, measure.t)

    def _record_setup_frame(self, measure: FrameMeasure) -> None:
        self._setup_frames.append(measure)
        while measure.t - self._setup_frames[0].t > self.config.setup_window_s + self.config.liftoff_lookback_s:
            self._setup_frames.popleft()

    def _lifted_off(self, measure: FrameMeasure, velocity: float) -> bool:
        if math.isnan(self._rest_up) or not math.isfinite(measure.bar_up):
            return False
        cfg = self.config
        return measure.bar_up - self._rest_up > cfg.liftoff_rise_m and velocity > cfg.liftoff_velocity_mps

    def _stood_up(self, measure: FrameMeasure) -> bool:
        return self._held("stood", measure.standing and not measure.hands_on_bar, measure.t, self.config.setup_hold_s)

    def _leave_floor_area(self, measure: FrameMeasure) -> None:
        near, facing = self._near_and_facing(measure)
        self._enter(DeadliftPhase.STANCE if near and facing else DeadliftPhase.APPROACH, measure.t)
        self._stance_midfoots.clear()
        self._lateral_frames.clear()
        self._stance_bar_offsets.clear()
        self._locked_midfoot = None
        self._locked_lateral = None
        self._floor_midfoots = None

    def _record_at_bar(self, measure: FrameMeasure) -> None:
        if measure.hands_on_bar and measure.legs_measured:
            self._lateral_frames.append(measure)
        if self._floor_midfoots is not None and measure.feet_measured and measure.midfoot is not None:
            self._floor_midfoots.append(measure.midfoot)

    def _step_setup(self, measure: FrameMeasure, velocity: float) -> None:
        self._record_at_bar(measure)
        if self._lifted_off(measure, velocity):
            if self._floor_midfoots is None:
                self._lock_lateral(measure)
            else:
                self._relock_at_floor(measure)
            self._start_rep(measure, touch_and_go=False, window_s=self.config.setup_window_s)
            return
        if self._stood_up(measure):
            self._leave_floor_area(measure)

    def _step_floor(self, measure: FrameMeasure, velocity: float) -> None:
        cfg = self.config
        self._record_at_bar(measure)
        if self._lifted_off(measure, velocity):
            # Quick re-pull: the setup is the last moment before liftoff.
            self._relock_at_floor(measure)
            self._start_rep(measure, touch_and_go=False, window_s=cfg.quick_pull_window_s)
            return
        if self._held("resetup", measure.hands_on_bar and self._bar_still(cfg.liftoff_rest_speed_mps), measure.t, cfg.resetup_hold_s):
            self._enter(DeadliftPhase.SETUP, measure.t)
            return
        if self._stood_up(measure):
            self._leave_floor_area(measure)

    # Index into history of motion onset: the last frame in the look-back where
    # the bar sat on its rest.
    def _back_dated_liftoff(self) -> int:
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
            if measure.bar_up - self._rest_up <= self._event_band_m():
                return index
            if not math.isfinite(history[lowest].bar_up) or measure.bar_up < history[lowest].bar_up:
                lowest = index
        return lowest

    def _start_rep(self, measure: FrameMeasure, touch_and_go: bool, window_s: float) -> None:
        history = list(self._history)
        velocities = list(self._velocities)
        liftoff_index = self._back_dated_liftoff()
        liftoff = history[liftoff_index]
        setup: dict[str, float] = {}
        if not touch_and_go:
            setup = measure_setup(
                list(self._setup_frames), liftoff.t, window_s, self._locked_midfoot, self._athlete(), self.config,
            )
            if "grip_offset_m" in setup:
                self._learned_grip_offset_m = setup["grip_offset_m"]
            if "predicted_trunk_change_deg" in setup:
                self._set_predicted_change_deg = setup["predicted_trunk_change_deg"]
            setup["stance_bar_midfoot_cm"] = self._stance_bar_midfoot_cm
            self._stance_bar_midfoot_cm = NAN
        rep = RepTrack(liftoff, touch_and_go, setup)
        rep.lead_in = [frame for frame in history[:liftoff_index] if liftoff.t - frame.t <= self.config.liftoff_lookback_s]
        for index in range(liftoff_index + 1, len(history)):
            rep.append(history[index], velocities[index])
        self._rep = rep
        self._enter(DeadliftPhase.PULL, measure.t)
        self.rep_started = True
        self._setup_frames.clear()

    def _top_reached(self, rep: RepTrack, measure: FrameMeasure) -> bool:
        cfg = self.config
        rise = rep.peak_up - rep.liftoff.bar_up
        if rise < cfg.failed_rep_min_rise_m:
            return False
        expected = self._expected_top_up()
        if math.isfinite(expected) and rep.peak_up < expected - cfg.top_margin_m:
            # Leaning back past vertical the hips went through, legs straight: a
            # lockout (D5 judges it), however low the bar hangs from shoulders
            # pulled back.
            return (
                measure.trunk_deg <= -cfg.overextended_top_deg
                and not measure.knee_flex_deg > cfg.overextended_max_knee_deg
            )
        return abs(measure.trunk_deg) <= cfg.top_max_trunk_deg

    def _step_pull(self, measure: FrameMeasure, velocity: float) -> None:
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
        if self._back_on_floor(measure):
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

    # The top event: the bar's first arrival within TOP_ARRIVAL_BAND_M of its peak.
    @staticmethod
    def _top_arrival(rep: RepTrack) -> float:
        for frame in rep.frames:
            if frame.bar_up >= rep.peak_up - TOP_ARRIVAL_BAND_M:
                return frame.t
        return rep.frames[rep.peak_index].t

    def _hold_band_m(self) -> float:
        return max(TOP_HOLD_BAND_M, 2.0 * self._event_band_m())

    # A bar well above the top the rep held was stalled at a hitch (without a
    # standing reference, a still bar and an upright-enough trunk read as a
    # top): the pull resumes, and its top comes later.
    def _resume_pull(self, rep: RepTrack, measure: FrameMeasure) -> bool:
        if not measure.bar_up > rep.top_up + max(TOP_RESUME_RISE_M, self._hold_band_m()):
            return False
        rep.top_time = NAN
        rep.top_up = NAN
        rep.top_frames = []
        rep.lower_start = NAN
        rep.floor_time = NAN
        self._enter(DeadliftPhase.PULL, measure.t)
        return True

    def _step_top(self, measure: FrameMeasure, velocity: float) -> None:
        rep = self._rep
        rep.append(measure, velocity)
        if self._resume_pull(rep, measure):
            return
        # The hold is where the bar sits, not where its velocity reads still: a
        # noisy bar (the wrist proxy) rarely reads still frame by frame.
        if math.isfinite(measure.bar_up) and measure.bar_up >= rep.top_up - self._hold_band_m():
            rep.top_frames.append(measure)
        if velocity < -self.config.lower_velocity_mps:
            rep.lower_start = measure.t
            self._enter(DeadliftPhase.LOWER, measure.t)

    # Dead stop: on the rest and not moving for a few frames, or on the rest long
    # enough.
    def _back_on_floor(self, measure: FrameMeasure) -> bool:
        cfg = self.config
        if math.isnan(self._rest_up) or not math.isfinite(measure.bar_up):
            return False
        on_rest = measure.bar_up - self._rest_up <= cfg.floor_band_m
        if on_rest and self._bar_still(cfg.dead_stop_speed_mps):
            self._dead_stop_frames += 1
        else:
            self._dead_stop_frames = 0
        held = self._held("on_rest", on_rest, measure.t, cfg.dead_stop_hold_s)
        return self._dead_stop_frames >= cfg.dead_stop_frames or held

    def _step_lower(self, measure: FrameMeasure, velocity: float) -> None:
        rep = self._rep
        rep.append(measure, velocity)
        if self._resume_pull(rep, measure):
            return
        if math.isnan(rep.floor_time) and measure.bar_up - self._rest_up <= self._event_band_m():
            # Touchdown: back on the rest itself, not just inside the dead-stop band.
            rep.floor_time = measure.t
        if self._touch_and_go(rep, measure):
            self._finish_touch_and_go(rep, measure)
            return
        if self._back_on_floor(measure):
            if math.isnan(rep.floor_time):
                rep.floor_time = measure.t
            self._complete_rep(rep, measure)
            self._rep = None
            self._enter(DeadliftPhase.FLOOR, measure.t)

    @staticmethod
    def _lowest_since_lower(rep: RepTrack) -> int:
        start = next(
            (index for index, frame in enumerate(rep.frames) if frame.t >= rep.lower_start),
            len(rep.frames) - 1,
        )
        lowest = start
        for index in range(start, len(rep.frames)):
            height = rep.frames[index].bar_up
            if math.isfinite(height) and not height >= rep.frames[lowest].bar_up:
                lowest = index
        return lowest

    # Back up off a low point near the rest with the hands on the bar: risen clear
    # of the noise band, at least touch_go_sustain_s after the low point, and
    # rising over the frames since it. Their slope, not every step: one noisy
    # frame must not hide a rep (and with no dead stop, every rep after it); and
    # not a window reaching back into the descent, which a fast rep's rise never
    # outweighs.
    def _touch_and_go(self, rep: RepTrack, measure: FrameMeasure) -> bool:
        cfg = self.config
        if not math.isfinite(measure.bar_up):
            return False
        low = rep.frames[self._lowest_since_lower(rep)]
        # On the proxy "hands on the bar" is the wrists below the knees, which a
        # fast pull leaves within a few frames: the hands count at the low point.
        hands_frame = measure if measure.bar_source == BAR_SOURCE_BAR else low
        if not hands_frame.hands_on_bar:
            return False
        if not low.bar_up - self._rest_up <= cfg.touch_go_band_m:
            return False
        if measure.bar_up - low.bar_up <= cfg.touch_go_rise_m + self._event_band_m():
            return False
        if measure.t - low.t < cfg.touch_go_sustain_s - 1e-6:
            return False
        return self._bar_velocity(measure.t - low.t) > cfg.liftoff_velocity_mps

    def _finish_touch_and_go(self, rep: RepTrack, measure: FrameMeasure) -> None:
        low_index = self._lowest_since_lower(rep)
        low = rep.frames[low_index]
        if math.isnan(rep.floor_time) or rep.floor_time > low.t:
            rep.floor_time = low.t
        rep.touch_and_go_out = True
        next_frames = rep.frames[low_index:]
        next_velocities = rep.velocities[low_index:]
        rep.frames = rep.frames[:low_index + 1]
        rep.velocities = rep.velocities[:low_index + 1]
        self._complete_rep(rep, low)
        next_rep = RepTrack(next_frames[0], touch_and_go=True, setup={})
        for frame, velocity in zip(next_frames[1:], next_velocities[1:]):
            next_rep.append(frame, velocity)
        self._rep = next_rep
        self._enter(DeadliftPhase.PULL, measure.t)
        self.rep_started = True

    def _complete_rep(self, rep: RepTrack, end: FrameMeasure) -> None:
        if math.isnan(rep.top_time):
            return
        self.rep_count += 1
        self._set_top_heights.append(rep.top_up)
        if rep.touch_and_go and math.isfinite(self._set_predicted_change_deg):
            # Same lifter, same bar, same feet: the set's setup model still applies.
            rep.setup = {"predicted_trunk_change_deg": self._set_predicted_change_deg}
        features = rep_features(
            rep, self.rep_count, self._rest_up, self._event_band_m(), self._locked_midfoot,
            self._locked_body_lateral, self._athlete(), self.gravity_source, self._grip,
        )
        if math.isfinite(features.hip_shoulder_rise_ratio):
            self._set_rise_ratios.append(features.hip_shoulder_rise_ratio)
            recent = self._set_rise_ratios[-SET_RISE_RATIO_REPS:]
            if len(recent) >= 2:
                features.set_rise_ratio = float(np.median(recent))
        self._completed.append(_CompletedRep(
            features=features,
            start_time=features.liftoff_time,
            end_time=end.t,
            start_frame=rep.liftoff.frame_index,
            end_frame=end.frame_index,
        ))
        logger.info(
            "[DEADLIFT] Rep %d counted%s: %.0f cm in %.2f s",
            self.rep_count, " (touch-and-go)" if rep.touch_and_go else "",
            features.bar_rise_cm, features.pull_time_s,
        )

    # ------------------------------------------------------------------
    # Frame status
    # ------------------------------------------------------------------

    def _frame_status(self, measure: FrameMeasure) -> DeadliftFrameStatus:
        live_cm = NAN
        # Closed-loop foot guidance needs the bar itself: the hanging wrists of
        # the proxy are nowhere near where the bar sits on the floor. It speaks to
        # a lifter standing at the bar before the set's first rep (none counted
        # since reset_set), not walking off after it, from a median over a second:
        # a few noisy frames must not start it.
        guiding = self.phase == DeadliftPhase.STANCE and not self._set_top_heights and self._is_settled
        if guiding and measure.bar_source == BAR_SOURCE_BAR and len(self._stance_bar_offsets) >= LIVE_OFFSET_MIN_FRAMES:
            live_cm = _median([offset for _, offset in self._stance_bar_offsets])
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
