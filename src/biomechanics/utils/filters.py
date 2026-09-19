"""
Temporal Smoothing Filters for Pose Data

Provides low-pass filters to smooth noisy joint angle estimates
from pose detection. Essential for stable rep counting and fault detection.

One Euro Filter: Adaptive filter that balances smoothness and responsiveness.
EMA: Simple exponential moving average for baseline smoothing.
RunningMedian: short-window median that rejects single-frame transients.
"""

import math
from collections import deque
from statistics import median
from typing import Optional
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

        # A non-advancing clock (duplicate frame) updates nothing and
        # re-emits the last output.
        dt = t - self.last_time
        if dt <= 0:
            return self.x_filter.y

        # Estimate derivative from the previous FILTERED value (Casiez's
        # reference implementation); the raw previous sample doubles the lag.
        dx = (x - self.x_filter.y) / dt

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


class RunningMedian:
    """Median of the last `window_frames` finite values.

    A single-frame transient cannot move the output, so per-rep extrema
    taken over this statistic ignore one-frame spikes. NaN inputs are
    skipped; the output is NaN until the first finite value arrives.
    """

    def __init__(self, window_frames: int = 3):
        if window_frames < 1:
            raise ValueError("window_frames must be >= 1")
        self._window: deque = deque(maxlen=window_frames)

    def update(self, x: float) -> float:
        if not math.isnan(x):
            self._window.append(x)
        if not self._window:
            return math.nan
        return float(median(self._window))

    def reset(self):
        self._window.clear()
