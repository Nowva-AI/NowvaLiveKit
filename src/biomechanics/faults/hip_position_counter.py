"""
Signal-Based Rep Counter — Causal Real-Time 4-State Machine

Counts reps using a configurable signal and its causal velocity in a
4-state machine that operates frame-by-frame without any look-ahead.

The signal is exercise-specific and provided by the active ExerciseProfile:
    - Squats:    hip vertical position (cm, relative to ankle)
    - Deadlifts: trunk flexion angle
    - Curls:     elbow flexion angle
    - OHP:       wrist Y position relative to shoulder

State machine:
    IDLE  →  DESCENDING  →  BOTTOM  →  ASCENDING  →  IDLE

Velocity is a least-squares slope over the last few samples (no lagging
filter), BOTTOM is entered on a sign-aware velocity test so no-pause reps
are never merged, and a false start falls back to IDLE once the signal is
back at the standing baseline.
"""

from __future__ import annotations

import math
import time
from collections import deque
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

from biomechanics.config import HipPositionCounterConfig
from biomechanics.utils.filters import ExponentialMovingAverage
from biomechanics.utils.types import FaultEvent, JointAngles, RepData

# Least-squares velocity window (samples). Five samples at 30 Hz is ~2 frames of lag.
VELOCITY_WINDOW_FRAMES = 5
# Position used by the state machine = mean of the last N samples.
POSITION_WINDOW_FRAMES = 2
# A rep may only start once the signal has moved this fraction of min_depth
# away from the standing baseline (0.1 x 10 cm = 1 cm for squats).
ENTRY_GATE_DEPTH_FRACTION = 0.1
# BOTTOM requires the descent to have peaked at this multiple of the bottom
# velocity threshold first, so the slow first frames of a descent never count
# as the bottom.
BOTTOM_ARM_VELOCITY_RATIO = 2.0
# A gap longer than this re-initialises the velocity window.
MAX_GAP_S = 0.5
BASELINE_EMA_ALPHA = 0.15


class SignalRepState(str, Enum):
    """State machine states."""
    IDLE = "idle"
    DESCENDING = "descending"
    BOTTOM = "bottom"
    ASCENDING = "ascending"

# Backward compat alias
HipPositionState = SignalRepState


class SignalRepCounter:
    """
    Real-time rep counter driven by a configurable signal.

    The signal is provided per-frame by the active ExerciseProfile's
    get_rep_signal() method. Depth and asymmetry metrics are extracted
    via pluggable callables from the profile.

    Exposes the same public interface (in_rep, phase, rep_count, update,
    snapshot_rep_metrics, etc.) so the pipeline, dashboard, and BiLSTM
    enrichment code work without changes.
    """

    def __init__(
        self,
        config: Optional[HipPositionCounterConfig] = None,
        depth_metric_fn: Optional[Callable[[JointAngles], float]] = None,
        asymmetry_fn: Optional[Callable[[JointAngles], Dict[str, float]]] = None,
    ):
        self.config = config or HipPositionCounterConfig()
        self.state = SignalRepState.IDLE
        self.rep_count = 0

        # Pluggable metric extractors (default: squat knee-based)
        self._depth_metric_fn = depth_metric_fn or (lambda a: a.avg_knee_flexion)
        self._asymmetry_fn = asymmetry_fn or (
            lambda a: {"knee": a.knee_asymmetry, "hip": a.hip_asymmetry}
        )

        # ---- signal window (raw samples) ----
        self._signal_times: deque[float] = deque(maxlen=VELOCITY_WINDOW_FRAMES)
        self._signal_values: deque[float] = deque(maxlen=VELOCITY_WINDOW_FRAMES)

        # ---- standing baseline ----
        self._standing_baseline: Optional[float] = None
        self._baseline_ema = ExponentialMovingAverage(alpha=BASELINE_EMA_ALPHA)

        # ---- state dwell tracking ----
        self._frames_in_state: int = 0

        # ---- current-rep signal tracking ----
        self._max_position_in_rep: float = 0.0  # peak signal = rep bottom
        self._peak_descent_velocity: float = 0.0

        # ---- current-rep metric tracking (from JointAngles) ----
        self._rep_start_time: float = 0.0
        self._rep_start_frame: int = 0
        self._frames_in_rep: int = 0
        self._max_depth_angle: float = math.nan
        self._min_depth_angle: float = math.nan
        self._bottom_time: float = 0.0
        self._current_faults: List[FaultEvent] = []
        self._asymmetry_sums: Dict[str, float] = {}
        self._angle_samples: int = 0

        # Max depth metric of the last rep rejected for insufficient depth, so
        # the pipeline can still report a depth fault for it.
        self.rejected_rep_max_depth_angle: float = math.nan

        # ---- assessment mode ----
        self._assessment_mode: bool = False

    # ------------------------------------------------------------------
    # Public properties (same interface as RepCounter)
    # ------------------------------------------------------------------

    @property
    def in_rep(self) -> bool:
        return self.state in (
            SignalRepState.DESCENDING,
            SignalRepState.BOTTOM,
            SignalRepState.ASCENDING,
        )

    @property
    def phase(self) -> str:
        return self.state.value

    @property
    def entry_gate_cm(self) -> float:
        """Signal displacement from the standing baseline required to start a rep."""
        return ENTRY_GATE_DEPTH_FRACTION * self.config.min_depth_cm

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def set_assessment_mode(self, enabled: bool) -> None:
        """Bypass the min-depth validation so any completed rep counts."""
        self._assessment_mode = enabled

    def reset(self) -> None:
        self.state = SignalRepState.IDLE
        self.rep_count = 0
        self._reset_rep_tracking()
        self._frames_in_state = 0
        self._signal_times.clear()
        self._signal_values.clear()
        self._baseline_ema.reset()
        self._standing_baseline = None
        self.rejected_rep_max_depth_angle = math.nan

    def add_fault(self, fault: FaultEvent) -> None:
        self._current_faults.append(fault)

    def clear_current_faults(self) -> None:
        self._current_faults.clear()

    def snapshot_rep_metrics(self, now: float | None = None) -> dict:
        avg_asymmetry = {}
        for key, total in self._asymmetry_sums.items():
            avg_asymmetry[key] = total / self._angle_samples if self._angle_samples > 0 else 0.0
        if now is None:
            now = time.time()
        return {
            "max_depth_angle": self._max_depth_angle,
            "min_depth_angle": self._min_depth_angle,
            "descent_time": (self._bottom_time - self._rep_start_time)
                if self._bottom_time > 0 else 0.0,
            "ascent_time": (now - self._bottom_time)
                if self._bottom_time > 0 else 0.0,
            "faults": self._current_faults.copy(),
            "asymmetry": avg_asymmetry,
            # Backward compat
            "avg_knee_asymmetry": avg_asymmetry.get("knee", 0.0),
            "avg_hip_asymmetry": avg_asymmetry.get("hip", 0.0),
            "in_rep": self.in_rep,
        }

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(
        self,
        signal_value: float = None,
        timestamp: float = None,
        angles: Optional[JointAngles] = None,
        faults: Optional[List[FaultEvent]] = None,
        *,
        hip_position_cm: float = None,
    ) -> Tuple[Optional[RepData], Optional[str]]:
        """
        Process one frame.

        Args:
            signal_value: Rep counting signal from the exercise profile.
                For squats this is (hip_mid_y - ankle_mid_y) * 100.
                Other exercises feed different signals (trunk flexion, etc.).
            timestamp: Wall-clock time for this frame.
            angles: JointAngles for metric tracking (knee depth, asymmetry).
            faults: Faults detected this frame.
            hip_position_cm: Deprecated alias for signal_value (backward compat).

        Returns:
            (RepData, None) when a rep completes.
            (None, "go_deeper") when a rep ends but depth was insufficient.
            (None, None) otherwise, including frames with a NaN signal or a
            timestamp that has not advanced (duplicates), which are ignored.
        """
        # Backward compatibility: accept hip_position_cm as alias
        if signal_value is None and hip_position_cm is not None:
            signal_value = hip_position_cm
        elif signal_value is None:
            raise ValueError("signal_value (or hip_position_cm) is required")
        if faults:
            self._current_faults.extend(faults)

        if math.isnan(signal_value):
            return None, None
        if self._signal_times:
            dt = timestamp - self._signal_times[-1]
            if dt <= 0.0:
                return None, None
            if dt > MAX_GAP_S:
                self._signal_times.clear()
                self._signal_values.clear()

        # ---- causal position & velocity from the raw sample window ----
        self._signal_times.append(timestamp)
        self._signal_values.append(signal_value)
        position = self._window_position()
        velocity = self._window_velocity()

        # ---- initialise standing baseline from first frames ----
        if self._standing_baseline is None:
            self._standing_baseline = position
            self._baseline_ema.value = position

        # ---- track state dwell ----
        self._frames_in_state += 1

        # ---- track rep frame count and angle metrics ----
        if self.in_rep:
            self._frames_in_rep += 1
            if angles is not None:
                self._track_angles(angles)

        # ---- state machine ----
        completed_rep: Optional[RepData] = None
        feedback: Optional[str] = None
        cfg = self.config

        if self.state == SignalRepState.IDLE:
            # Update the standing baseline only while actually standing still,
            # so a slow descent cannot drag the baseline down with it.
            if abs(velocity) < cfg.entry_vel_threshold:
                self._standing_baseline = self._baseline_ema.filter(position)

            if velocity > cfg.entry_vel_threshold and position > self._standing_baseline + self.entry_gate_cm:
                self._change_state(SignalRepState.DESCENDING)
                self._start_rep(position, timestamp, angles)
                self._peak_descent_velocity = velocity

        elif self.state == SignalRepState.DESCENDING:
            self._track_bottom(position, timestamp)
            self._peak_descent_velocity = max(self._peak_descent_velocity, velocity)

            if self._is_false_start(position):
                self._abort_rep()
            elif (
                self._frames_in_state >= cfg.min_frames_descending
                and self._peak_descent_velocity >= BOTTOM_ARM_VELOCITY_RATIO * cfg.bottom_vel_threshold
                and velocity < cfg.bottom_vel_threshold
            ):
                self._change_state(SignalRepState.BOTTOM)

        elif self.state == SignalRepState.BOTTOM:
            # Still track depth in case bottom drifts deeper
            self._track_bottom(position, timestamp)

            if self._is_false_start(position):
                self._abort_rep()
            elif self._frames_in_state >= cfg.min_frames_bottom:
                if velocity < -cfg.ascending_vel_threshold:
                    self._change_state(SignalRepState.ASCENDING)

        elif self.state == SignalRepState.ASCENDING:
            if self._frames_in_state >= cfg.min_frames_ascending:
                returned = position < self._standing_baseline + cfg.standing_return_cm
                if returned:
                    # Validate rep
                    depth = self._max_position_in_rep - self._standing_baseline
                    depth_ok = self._assessment_mode or depth >= cfg.min_depth_cm
                    if (
                        self._frames_in_rep >= cfg.min_rep_duration_frames
                        and depth_ok
                    ):
                        self.rep_count += 1
                        completed_rep = self._create_rep_data(timestamp, angles)
                    elif not depth_ok:
                        feedback = "go_deeper"
                        self.rejected_rep_max_depth_angle = self._max_depth_angle

                    self._change_state(SignalRepState.IDLE)
                    self._reset_rep_tracking()

        return completed_rep, feedback

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _window_position(self) -> float:
        recent = list(self._signal_values)[-POSITION_WINDOW_FRAMES:]
        return sum(recent) / len(recent)

    def _window_velocity(self) -> float:
        n = len(self._signal_times)
        if n < 2:
            return 0.0
        time_mean = sum(self._signal_times) / n
        value_mean = sum(self._signal_values) / n
        covariance = 0.0
        variance = 0.0
        for sample_time, sample_value in zip(self._signal_times, self._signal_values):
            time_offset = sample_time - time_mean
            covariance += time_offset * (sample_value - value_mean)
            variance += time_offset * time_offset
        if variance <= 0.0:
            return 0.0
        return covariance / variance

    def _track_bottom(self, position: float, timestamp: float) -> None:
        if position > self._max_position_in_rep:
            self._max_position_in_rep = position
            self._bottom_time = timestamp

    def _is_false_start(self, position: float) -> bool:
        # The signal is back at the standing baseline without ever ascending:
        # noise, a shuffle, or an aborted descent — not a rep.
        return (
            self._frames_in_rep > self.config.min_frames_descending
            and position < self._standing_baseline + self.entry_gate_cm
        )

    def _abort_rep(self) -> None:
        self._change_state(SignalRepState.IDLE)
        self._reset_rep_tracking()

    def _change_state(self, new_state: HipPositionState) -> None:
        self.state = new_state
        self._frames_in_state = 0

    def _start_rep(
        self,
        position: float,
        timestamp: float,
        angles: Optional[JointAngles],
    ) -> None:
        self._rep_start_time = timestamp
        self._rep_start_frame = angles.frame_index if angles else 0
        self._frames_in_rep = 1
        self._max_position_in_rep = position
        self._peak_descent_velocity = 0.0
        self._max_depth_angle = math.nan
        self._min_depth_angle = math.nan
        self._bottom_time = 0.0
        self._current_faults = []
        self._asymmetry_sums = {}
        self._angle_samples = 0
        self.rejected_rep_max_depth_angle = math.nan

        if angles is not None:
            self._track_angles(angles)

    def _track_angles(self, angles: JointAngles) -> None:
        depth = self._depth_metric_fn(angles)

        if math.isnan(depth):
            return

        if math.isnan(self._max_depth_angle) or depth > self._max_depth_angle:
            self._max_depth_angle = depth

        if math.isnan(self._min_depth_angle) or depth < self._min_depth_angle:
            self._min_depth_angle = depth

        asym = self._asymmetry_fn(angles)
        if any(math.isnan(val) for val in asym.values()):
            return
        for key, val in asym.items():
            self._asymmetry_sums[key] = self._asymmetry_sums.get(key, 0.0) + val
        self._angle_samples += 1

    def _reset_rep_tracking(self) -> None:
        self._rep_start_time = 0.0
        self._rep_start_frame = 0
        self._frames_in_rep = 0
        self._max_position_in_rep = 0.0
        self._peak_descent_velocity = 0.0
        self._max_depth_angle = math.nan
        self._min_depth_angle = math.nan
        self._bottom_time = 0.0
        self._current_faults = []
        self._asymmetry_sums = {}
        self._angle_samples = 0

    def _create_rep_data(
        self, end_time: float, angles: Optional[JointAngles]
    ) -> RepData:
        descent_time = (
            (self._bottom_time - self._rep_start_time)
            if self._bottom_time > 0 else 0.0
        )
        ascent_time = (
            (end_time - self._bottom_time)
            if self._bottom_time > 0 else 0.0
        )
        avg_asymmetry = {}
        for key, total in self._asymmetry_sums.items():
            avg_asymmetry[key] = total / self._angle_samples if self._angle_samples > 0 else 0.0

        return RepData(
            rep_number=self.rep_count,
            start_time=self._rep_start_time,
            end_time=end_time,
            start_frame=self._rep_start_frame,
            end_frame=angles.frame_index if angles else 0,
            max_depth_angle=self._max_depth_angle,
            min_depth_angle=self._min_depth_angle,
            descent_time=descent_time,
            ascent_time=ascent_time,
            faults=self._current_faults.copy(),
            asymmetry=avg_asymmetry,
            # Backward compat
            avg_knee_asymmetry=avg_asymmetry.get("knee", 0.0),
            avg_hip_asymmetry=avg_asymmetry.get("hip", 0.0),
        )


# Backward compat alias — pipeline and other consumers may still import this name
HipPositionRepCounter = SignalRepCounter
