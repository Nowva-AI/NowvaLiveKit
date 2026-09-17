"""
Temporal Smoothing Filters for Pose Data

Provides low-pass filters to smooth noisy joint angle estimates
from pose detection. Essential for stable rep counting and fault detection.

One Euro Filter: Adaptive filter that balances smoothness and responsiveness.
EMA: Simple exponential moving average for baseline smoothing.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Optional
import time


class LowPassFilter:
    """Simple first-order low-pass filter."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.y: Optional[float] = None
        self.initialized = False

    def filter(self, x: float) -> float:
        if not self.initialized:
            self.y = x
            self.initialized = True
        else:
            self.y = self.alpha * x + (1 - self.alpha) * self.y
        return self.y

    def reset(self):
        self.y = None
        self.initialized = False


class OneEuroFilter:
    """
    One Euro Filter for real-time signal smoothing.

    The One Euro filter is an adaptive low-pass filter that adjusts its
    cutoff frequency based on the speed of the input signal:
    - When the signal is changing slowly, use more smoothing (low cutoff)
    - When the signal is changing quickly, use less smoothing (high cutoff)

    This gives smooth output for slow movements while preserving fast movements.

    Reference: Casiez et al., "1€ Filter: A Simple Speed-based Low-pass Filter
    for Noisy Input in Interactive Systems", CHI 2012.

    Usage:
        filter = OneEuroFilter(min_cutoff=1.0, beta=0.007)
        smoothed_value = filter.filter(noisy_value, timestamp)
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ):
        """
        Initialize One Euro Filter.

        Args:
            min_cutoff: Minimum cutoff frequency in Hz. Lower = smoother.
                        Good starting value: 1.0 for joint angles.
            beta: Speed coefficient. Higher = more responsive to fast movements.
                  Good starting value: 0.007 for joint angles.
            d_cutoff: Cutoff frequency for the derivative filter.
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()

        self.last_time: Optional[float] = None
        self.last_value: Optional[float] = None

    def _compute_alpha(self, cutoff: float, dt: float) -> float:
        """Compute smoothing factor from cutoff frequency and time delta."""
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: float, t: Optional[float] = None) -> float:
        """
        Filter a value.

        Args:
            x: Input value to filter
            t: Timestamp in seconds. If None, uses current time.

        Returns:
            Filtered value
        """
        if t is None:
            t = time.time()

        if self.last_time is None:
            # First value - initialize
            self.last_time = t
            self.last_value = x
            self.x_filter.y = x
            self.x_filter.initialized = True
            self.dx_filter.y = 0.0
            self.dx_filter.initialized = True
            return x

        # Compute time delta
        dt = t - self.last_time
        if dt <= 0:
            dt = 1e-6  # Prevent division by zero

        # Estimate derivative (rate of change)
        dx = (x - self.last_value) / dt

        # Filter the derivative
        alpha_d = self._compute_alpha(self.d_cutoff, dt)
        self.dx_filter.alpha = alpha_d
        dx_hat = self.dx_filter.filter(dx)

        # Compute adaptive cutoff based on speed
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)

        # Filter the value with adaptive cutoff
        alpha = self._compute_alpha(cutoff, dt)
        self.x_filter.alpha = alpha
        x_hat = self.x_filter.filter(x)

        # Store for next iteration
        self.last_time = t
        self.last_value = x

        return x_hat

    def reset(self):
        """Reset filter state."""
        self.x_filter.reset()
        self.dx_filter.reset()
        self.last_time = None
        self.last_value = None


class ExponentialMovingAverage:
    """
    Simple Exponential Moving Average filter.

    EMA_t = alpha * x_t + (1 - alpha) * EMA_{t-1}

    Simpler than One Euro but doesn't adapt to movement speed.
    Good for consistent smoothing.
    """

    def __init__(self, alpha: float = 0.3):
        """
        Initialize EMA filter.

        Args:
            alpha: Smoothing factor (0-1). Higher = less smoothing.
                   0.3 is a good starting point for pose data at 30fps.
        """
        self.alpha = alpha
        self.value: Optional[float] = None

    def filter(self, x: float) -> float:
        """Filter a value."""
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value

    def reset(self):
        """Reset filter state."""
        self.value = None


@dataclass
class JointAngleFilter:
    """
    Applies One Euro Filter to all joint angles.

    Maintains separate filters for each joint angle field,
    providing smooth, stable angle estimates for rep counting.
    """

    min_cutoff: float = 1.0
    beta: float = 0.007

    # Filters for each angle (created lazily)
    _filters: Dict[str, OneEuroFilter] = field(default_factory=dict)

    # Phase -> (min_cutoff, beta) mapping
    PHASE_PARAMS = {
        "idle":       (0.3, 0.003),   # Heavy smoothing — kill standing jitter
        "descending": (1.0, 0.007),   # Standard — responsive to fast descent
        "bottom":     (0.8, 0.005),   # Moderate — stable at bottom
        "ascending":  (1.0, 0.007),   # Standard — responsive to ascent
    }

    DEFAULT_PARAMS = (1.0, 0.007)

    def update_phase(self, phase: str) -> None:
        """
        Adjust filter parameters based on current rep phase.

        Called each frame from the pipeline after rep counter updates.
        Changes apply to the NEXT filter call, not retroactively.

        Args:
            phase: One of "idle", "descending", "bottom", "ascending"
        """
        min_cutoff, beta = self.PHASE_PARAMS.get(phase, self.DEFAULT_PARAMS)

        # Only update if parameters actually changed
        if min_cutoff != self.min_cutoff or beta != self.beta:
            self.min_cutoff = min_cutoff
            self.beta = beta

            # Update existing filter instances
            for filt in self._filters.values():
                filt.min_cutoff = min_cutoff
                filt.beta = beta

    def _get_filter(self, name: str) -> OneEuroFilter:
        """Get or create filter for a joint angle."""
        if name not in self._filters:
            self._filters[name] = OneEuroFilter(
                min_cutoff=self.min_cutoff,
                beta=self.beta,
            )
        return self._filters[name]

    def filter_angles(self, angles, timestamp: Optional[float] = None):
        """
        Filter all joint angles in a JointAngles object.

        Args:
            angles: JointAngles object with raw angle values
            timestamp: Timestamp for filtering (uses current time if None)

        Returns:
            New JointAngles object with smoothed values
        """
        from biomechanics.utils.types import JointAngles

        if timestamp is None:
            timestamp = angles.timestamp if angles.timestamp else time.time()

        # Filter each angle field
        filtered_data = {
            'hip_flexion_l': self._get_filter('hip_flexion_l').filter(angles.hip_flexion_l, timestamp),
            'hip_flexion_r': self._get_filter('hip_flexion_r').filter(angles.hip_flexion_r, timestamp),
            'hip_adduction_l': self._get_filter('hip_adduction_l').filter(angles.hip_adduction_l, timestamp),
            'hip_adduction_r': self._get_filter('hip_adduction_r').filter(angles.hip_adduction_r, timestamp),
            'hip_rotation_l': self._get_filter('hip_rotation_l').filter(angles.hip_rotation_l, timestamp),
            'hip_rotation_r': self._get_filter('hip_rotation_r').filter(angles.hip_rotation_r, timestamp),
            'knee_flexion_l': self._get_filter('knee_flexion_l').filter(angles.knee_flexion_l, timestamp),
            'knee_flexion_r': self._get_filter('knee_flexion_r').filter(angles.knee_flexion_r, timestamp),
            'ankle_dorsiflexion_l': self._get_filter('ankle_dorsiflexion_l').filter(angles.ankle_dorsiflexion_l, timestamp),
            'ankle_dorsiflexion_r': self._get_filter('ankle_dorsiflexion_r').filter(angles.ankle_dorsiflexion_r, timestamp),
            'knee_valgus_l': self._get_filter('knee_valgus_l').filter(angles.knee_valgus_l, timestamp),
            'knee_valgus_r': self._get_filter('knee_valgus_r').filter(angles.knee_valgus_r, timestamp),
            'foot_confidence_l': angles.foot_confidence_l,
            'foot_confidence_r': angles.foot_confidence_r,
            'knee_ankle_sep_ratio': angles.knee_ankle_sep_ratio,
            'shoulder_flexion_l': self._get_filter('shoulder_flexion_l').filter(angles.shoulder_flexion_l, timestamp),
            'shoulder_flexion_r': self._get_filter('shoulder_flexion_r').filter(angles.shoulder_flexion_r, timestamp),
            'shoulder_abduction_l': self._get_filter('shoulder_abduction_l').filter(angles.shoulder_abduction_l, timestamp),
            'shoulder_abduction_r': self._get_filter('shoulder_abduction_r').filter(angles.shoulder_abduction_r, timestamp),
            'elbow_flexion_l': self._get_filter('elbow_flexion_l').filter(angles.elbow_flexion_l, timestamp),
            'elbow_flexion_r': self._get_filter('elbow_flexion_r').filter(angles.elbow_flexion_r, timestamp),
            'wrist_y_l': self._get_filter('wrist_y_l').filter(angles.wrist_y_l, timestamp),
            'wrist_y_r': self._get_filter('wrist_y_r').filter(angles.wrist_y_r, timestamp),
            'wrist_x_l': self._get_filter('wrist_x_l').filter(angles.wrist_x_l, timestamp),
            'wrist_x_r': self._get_filter('wrist_x_r').filter(angles.wrist_x_r, timestamp),
            'trunk_flexion': self._get_filter('trunk_flexion').filter(angles.trunk_flexion, timestamp),
            'trunk_lateral_flexion': self._get_filter('trunk_lateral_flexion').filter(angles.trunk_lateral_flexion, timestamp),
            'trunk_rotation': self._get_filter('trunk_rotation').filter(angles.trunk_rotation, timestamp),
            'pelvis_tilt': self._get_filter('pelvis_tilt').filter(angles.pelvis_tilt, timestamp),
            'pelvis_list': self._get_filter('pelvis_list').filter(angles.pelvis_list, timestamp),
            'pelvis_rotation': self._get_filter('pelvis_rotation').filter(angles.pelvis_rotation, timestamp),
            'timestamp': angles.timestamp,
            'frame_index': angles.frame_index,
        }

        return JointAngles(**filtered_data)

    def reset(self):
        """Reset all filters."""
        self._filters.clear()
