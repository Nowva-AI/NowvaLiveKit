"""
BiLSTM-based Rep Counter (5-class depth classification)

Converts per-frame depth class probabilities into rep counts using:
  1. Exponential moving average for smoothing (vector EMA over all classes)
  2. Two-state finite state machine (UP / DOWN) with configurable min_depth_class

Descents that never reach min_depth_class are reported as shallow reps so the
coaching layer can cue depth instead of silently dropping the movement.
"""

from enum import Enum
from typing import Optional, Tuple
import time as _time

import numpy as np
from pydantic import BaseModel

from biomechanics.utils.types import RepData, DEPTH_CLASS_NAMES

# Depth class required to enter DOWN during assessment (any descent counts).
ASSESSMENT_MIN_DEPTH_CLASS = 1

# Depth class meaning "standing" — the resting point of every rep.
STANDING_DEPTH_CLASS = 0


class BiLSTMCounterConfig(BaseModel):
    """Configuration for the 5-class BiLSTM-based rep counter."""
    min_depth_class: int = 3       # minimum depth class to enter DOWN (3=parallel)
    min_rep_frames: int = 12       # minimum frames in DOWN before counting
    ema_alpha: float = 0.2         # EMA smoothing factor
    num_classes: int = 5


class BiLSTMCounterState(str, Enum):
    UP = "up"
    DOWN = "down"


class BiLSTMRepCounter:
    """
    Two-state FSM that counts reps from 5-class depth probabilities.

    Transitions:
        UP → DOWN   when argmax(smoothed_probs) >= min_depth_class
        DOWN → UP   when argmax(smoothed_probs) == 0 (standing) AND
                    frames_in_down >= min_rep_frames
                    → rep counted, RepData returned with depth info

    While in UP, a sustained descent that never reaches min_depth_class is
    tracked separately and reported as a shallow rep when the lifter returns
    to standing. Shallow reps are not counted — they exist so the coaching
    layer can tell the lifter to go deeper.
    """

    def __init__(self, config: Optional[BiLSTMCounterConfig] = None):
        self.config = config or BiLSTMCounterConfig()
        self.state = BiLSTMCounterState.UP
        self.rep_count = 0

        self._smoothed_probs = np.zeros(self.config.num_classes)
        self._first_update = True

        # Per-rep tracking
        self._frames_in_down: int = 0
        self._max_depth_seen: int = 0
        self._rep_start_time: float = 0.0
        self._rep_start_frame: int = 0
        self._assessment_mode: bool = False

        # Shallow-descent tracking (UP state only)
        self._shallow_frames: int = 0
        self._shallow_max_class: int = 0

    def set_assessment_mode(self, enabled: bool) -> None:
        """Lower the DOWN-entry threshold so any descent counts as a rep."""
        self._assessment_mode = enabled

    def set_min_depth_class(self, depth_class: int) -> None:
        """Change the class a descent must reach to open a rep."""
        self.config.min_depth_class = depth_class

    def reject_last_rep(self) -> None:
        """Un-count the rep just returned: a caller judged it too shallow."""
        self.rep_count = max(0, self.rep_count - 1)

    @property
    def in_rep(self) -> bool:
        return self.state == BiLSTMCounterState.DOWN

    @property
    def predicted_depth_class(self) -> int:
        return int(np.argmax(self._smoothed_probs))

    @property
    def smoothed_probabilities(self) -> np.ndarray:
        return self._smoothed_probs.copy()

    @property
    def in_rep_probability(self) -> float:
        """Sum of smoothed probabilities for classes >= min_depth_class."""
        return float(self._smoothed_probs[self.config.min_depth_class:].sum())

    @property
    def smoothed_probability(self) -> float:
        """Backward compat — returns in_rep_probability."""
        return self.in_rep_probability

    def update(
        self,
        raw_probs: np.ndarray,
        timestamp: float = 0.0,
        frame_index: int = 0,
    ) -> Tuple[Optional[RepData], Optional[int]]:
        """
        Feed one frame's depth class probabilities (shape (num_classes,)).

        Returns (rep_data, shallow_depth_class):
            (RepData, None) when a rep completes (DOWN → UP after
                sufficient frames).
            (None, max_class) when a sustained descent returned to standing
                without ever reaching min_depth_class — a shallow rep.
            (None, None) otherwise.
        """
        # Vector EMA smoothing
        if self._first_update:
            self._smoothed_probs = raw_probs.copy()
            self._first_update = False
        else:
            alpha = self.config.ema_alpha
            self._smoothed_probs = alpha * raw_probs + (1 - alpha) * self._smoothed_probs

        predicted_class = int(np.argmax(self._smoothed_probs))

        if self.state == BiLSTMCounterState.UP:
            entry_threshold = ASSESSMENT_MIN_DEPTH_CLASS if self._assessment_mode else self.config.min_depth_class
            if predicted_class >= entry_threshold:
                self.state = BiLSTMCounterState.DOWN
                self._frames_in_down = 0
                self._max_depth_seen = predicted_class
                self._rep_start_time = timestamp or _time.time()
                self._rep_start_frame = frame_index
                self._reset_shallow_tracking()
            elif predicted_class > STANDING_DEPTH_CLASS:
                # Descending, but not deep enough to open a rep
                self._shallow_frames += 1
                self._shallow_max_class = max(self._shallow_max_class, predicted_class)
            else:
                # Back to standing — a long enough descent was a shallow rep
                shallow_class = (
                    self._shallow_max_class
                    if self._shallow_frames >= self.config.min_rep_frames
                    else None
                )
                self._reset_shallow_tracking()
                if shallow_class is not None:
                    return None, shallow_class

        elif self.state == BiLSTMCounterState.DOWN:
            self._frames_in_down += 1
            self._max_depth_seen = max(self._max_depth_seen, predicted_class)

            if (
                predicted_class == STANDING_DEPTH_CLASS
                and self._frames_in_down >= self.config.min_rep_frames
            ):
                self.rep_count += 1
                self.state = BiLSTMCounterState.UP

                end_time = timestamp or _time.time()
                depth_name = DEPTH_CLASS_NAMES.get(self._max_depth_seen, "Unknown")
                return RepData(
                    rep_number=self.rep_count,
                    start_time=self._rep_start_time,
                    end_time=end_time,
                    start_frame=self._rep_start_frame,
                    end_frame=frame_index,
                    depth_class=self._max_depth_seen,
                    depth_class_name=depth_name,
                    max_depth_class=self._max_depth_seen,
                ), None

        return None, None

    def _reset_shallow_tracking(self) -> None:
        self._shallow_frames = 0
        self._shallow_max_class = 0

    def reset(self) -> None:
        """Reset state for a new set/session."""
        self.state = BiLSTMCounterState.UP
        self.rep_count = 0
        self._smoothed_probs = np.zeros(self.config.num_classes)
        self._first_update = True
        self._frames_in_down = 0
        self._max_depth_seen = 0
        self._reset_shallow_tracking()
