"""Bar states keyed by capture time, so the lagged analysis skeleton reads the
bar as it was when that skeleton was captured (PLAN.md §2.2). The skeleton
lags capture by the Kalman's lag_frames; without this pairing, liftoff timing,
hips-shoot and shoulders-vs-bar would mix samples ~67 ms apart.
"""

from __future__ import annotations

from collections import deque

from .types import BarState3D

# Several more captures than the Kalman lag, so a re-initialised lag still pairs.
BAR_BUFFER_FRAMES = 16
# Nearest state within half a frame at 30 fps.
DEFAULT_MATCH_TOLERANCE_S = 1.0 / 60.0


class BarStateBuffer:

    def __init__(self, capacity: int = BAR_BUFFER_FRAMES) -> None:
        self._states: deque[BarState3D] = deque(maxlen=capacity)

    def push(self, state: BarState3D) -> None:
        self._states.append(state)

    def clear(self) -> None:
        self._states.clear()

    def at(self, timestamp: float, tolerance_s: float = DEFAULT_MATCH_TOLERANCE_S) -> BarState3D | None:
        """The state captured nearest to timestamp, or None when none is within tolerance."""
        best: BarState3D | None = None
        best_gap_s = tolerance_s
        for state in self._states:
            gap_s = abs(state.timestamp - timestamp)
            if gap_s <= best_gap_s:
                best = state
                best_gap_s = gap_s
        return best

    def __len__(self) -> int:
        return len(self._states)
