"""Tests for the VAD lag watchdog: hold turns longer while Silero runs behind real time, then restore."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.services.vad_lag_watchdog import (
    LAGGED_MIN_ENDPOINTING_DELAY_S,
    RECOVERY_QUIET_S,
    SLOW_INFERENCE_MESSAGE,
    VadLagWatchdog,
)

NORMAL_MIN_DELAY_S = 0.2
SILERO_LOGGER = logging.getLogger("livekit.plugins.silero")


class _FakeSession:
    def __init__(self) -> None:
        self.min_delays: list[float] = []

    def update_options(self, *, endpointing_opts: dict) -> None:
        self.min_delays.append(endpointing_opts["min_delay"])


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def watched() -> tuple[VadLagWatchdog, _FakeSession, _Clock]:
    session = _FakeSession()
    clock = _Clock()
    watchdog = VadLagWatchdog(session, NORMAL_MIN_DELAY_S, clock=clock)
    SILERO_LOGGER.addHandler(watchdog)
    yield watchdog, session, clock
    SILERO_LOGGER.removeHandler(watchdog)


def _lag_warning(delay_s: float) -> None:
    SILERO_LOGGER.warning(SLOW_INFERENCE_MESSAGE, extra={"delay": delay_s})


class TestVadLagWatchdog:
    def test_small_lag_changes_nothing(self, watched) -> None:
        watchdog, session, _ = watched
        _lag_warning(0.4)
        assert not watchdog.lagging
        assert session.min_delays == []

    def test_large_lag_holds_turns_longer_once(self, watched) -> None:
        watchdog, session, _ = watched
        _lag_warning(6.5)
        _lag_warning(8.0)
        assert watchdog.lagging
        assert session.min_delays == [pytest.approx(LAGGED_MIN_ENDPOINTING_DELAY_S)]

    def test_restores_after_quiet_period(self, watched) -> None:
        watchdog, session, clock = watched
        _lag_warning(6.5)
        clock.now += RECOVERY_QUIET_S / 2
        watchdog.check_recovery()
        assert watchdog.lagging
        clock.now += RECOVERY_QUIET_S
        watchdog.check_recovery()
        assert not watchdog.lagging
        assert session.min_delays[-1] == pytest.approx(NORMAL_MIN_DELAY_S)

    def test_unrelated_warnings_ignored(self, watched) -> None:
        watchdog, session, _ = watched
        SILERO_LOGGER.warning("max_buffered_speech reached", extra={"delay": 9.0})
        assert not watchdog.lagging
