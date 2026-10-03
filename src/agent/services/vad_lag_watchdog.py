"""VAD lag watchdog: while Silero runs behind real time, Nova waits longer before taking a turn.

On an overloaded machine Silero has fallen 6-13 s behind the audio; its stale speech
events then commit turns on fragments and Nova answers half-sentences. LiveKit reports
the backlog in its "inference is slower than realtime" warning (extra field `delay`),
which is the only signal it exposes, so this handler listens for it.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Silero (livekit.plugins.silero) and LiveKit's own VAD (livekit.agents) both log this.
LIVEKIT_LOGGER_NAME = "livekit"
SLOW_INFERENCE_MESSAGE = "inference is slower than realtime"
LAG_THRESHOLD_S = 1.0
# The warning only fires on a slow window; once they stop, the backlog drains in seconds.
RECOVERY_QUIET_S = 5.0
CHECK_INTERVAL_S = 1.0
LAGGED_MIN_ENDPOINTING_DELAY_S = 2.0


class VadLagWatchdog(logging.Handler):
    def __init__(
        self,
        session: Any,
        normal_min_delay_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(level=logging.WARNING)
        self._session = session
        self._normal_min_delay_s = normal_min_delay_s
        self._clock = clock
        self._last_lag_at = 0.0
        self.lagging = False

    def emit(self, record: logging.LogRecord) -> None:
        if record.getMessage() != SLOW_INFERENCE_MESSAGE:
            return
        delay_s = float(getattr(record, "delay", 0.0) or 0.0)
        if delay_s < LAG_THRESHOLD_S:
            return
        self._last_lag_at = self._clock()
        if self.lagging:
            return
        self.lagging = True
        logger.warning(
            f"[TURN] VAD is {delay_s:.1f}s behind real time — waiting "
            f"{LAGGED_MIN_ENDPOINTING_DELAY_S:.1f}s before taking a turn until it recovers"
        )
        self._session.update_options(
            endpointing_opts={"min_delay": max(LAGGED_MIN_ENDPOINTING_DELAY_S, self._normal_min_delay_s)}
        )

    def check_recovery(self) -> None:
        if not self.lagging or self._clock() - self._last_lag_at < RECOVERY_QUIET_S:
            return
        self.lagging = False
        logger.info("[TURN] VAD caught up with real time — normal turn-taking restored")
        self._session.update_options(endpointing_opts={"min_delay": self._normal_min_delay_s})

    async def run(self) -> None:
        """Attach to LiveKit's loggers and restore normal turn-taking once the lag clears."""
        livekit_logger = logging.getLogger(LIVEKIT_LOGGER_NAME)
        livekit_logger.addHandler(self)
        try:
            while True:
                await asyncio.sleep(CHECK_INTERVAL_S)
                self.check_recovery()
        finally:
            livekit_logger.removeHandler(self)
