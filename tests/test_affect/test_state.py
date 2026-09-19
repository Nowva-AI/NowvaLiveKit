"""Tests for athlete state hysteresis counted in utterances, rep-based effort, and the prompt line."""

from __future__ import annotations

import numpy as np

from affect.config import StateConfig
from affect.state import AthleteState, AthleteStateTracker, classify_affect


def _tracker(**overrides) -> AthleteStateTracker:
    return AthleteStateTracker(StateConfig(**overrides))


def _z(arousal: float = 0.0, dominance: float = 0.0, valence: float = 0.0) -> np.ndarray:
    return np.array([arousal, dominance, valence])


class TestClassification:
    def test_frustrated_is_low_valence_high_arousal(self) -> None:
        assert classify_affect(_z(arousal=0.5, valence=-2.0), 1.5)[0] == "frustrated"

    def test_strained_is_high_arousal_non_positive_valence(self) -> None:
        assert classify_affect(_z(arousal=2.0, valence=-0.5), 1.5)[0] == "strained"

    def test_engaged_is_high_arousal_positive_valence(self) -> None:
        assert classify_affect(_z(arousal=2.0, valence=0.5), 1.5)[0] == "engaged"

    def test_flat_otherwise(self) -> None:
        assert classify_affect(_z(arousal=-2.0, valence=-2.0), 1.5)[0] == "flat"
        assert classify_affect(_z(), 1.5)[0] == "flat"


class TestHysteresis:
    def test_one_moderate_utterance_does_not_enter(self) -> None:
        tracker = _tracker()
        state = tracker.observe_utterance(_z(arousal=1.8, valence=-0.2), now=0.0)
        assert state.affect == "flat"

    def test_two_consecutive_enter(self) -> None:
        tracker = _tracker()
        tracker.observe_utterance(_z(arousal=1.8, valence=-0.2), now=0.0)
        state = tracker.observe_utterance(_z(arousal=1.9, valence=-0.3), now=5.0)
        assert state.affect == "strained"

    def test_single_strong_utterance_enters(self) -> None:
        tracker = _tracker()
        state = tracker.observe_utterance(_z(arousal=0.5, valence=-3.0), now=0.0)
        assert state.affect == "frustrated"

    def test_dwell_prevents_immediate_exit(self) -> None:
        tracker = _tracker(window_utterances=1)
        tracker.observe_utterance(_z(arousal=0.5, valence=-3.0), now=0.0)
        state = tracker.observe_utterance(_z(), now=10.0)
        assert state.affect == "frustrated"
        state = tracker.observe_utterance(_z(), now=20.0)
        assert state.affect == "flat"

    def test_window_median_smooths_outlier(self) -> None:
        tracker = _tracker()
        tracker.observe_utterance(_z(), now=0.0)
        tracker.observe_utterance(_z(), now=1.0)
        state = tracker.observe_utterance(_z(arousal=0.5, valence=-3.0), now=2.0)
        assert state.affect == "flat"

    def test_window_expires_after_seconds(self) -> None:
        tracker = _tracker(window_seconds=10.0, window_utterances=3)
        tracker.observe_utterance(_z(arousal=1.8, valence=-0.2), now=0.0)
        state = tracker.observe_utterance(_z(arousal=1.8, valence=-0.2), now=100.0)
        assert state.affect == "flat"

    def test_strained_nudges_effort_up_only(self) -> None:
        tracker = _tracker()
        tracker.observe_utterance(_z(arousal=3.0, valence=-0.2), now=0.0)
        assert tracker.snapshot().effort == "working"


class TestEffort:
    def test_ratio_thresholds(self) -> None:
        tracker = _tracker(effort_working_ratio=1.15, effort_near_limit_ratio=1.35)
        assert tracker.observe_rep(1.0).effort == "fresh"
        assert tracker.observe_rep(1.1).effort == "fresh"
        assert tracker.observe_rep(1.2).effort == "working"
        assert tracker.observe_rep(1.5).effort == "near_limit"

    def test_faster_rep_resets_best(self) -> None:
        tracker = _tracker()
        tracker.observe_rep(1.5)
        tracker.observe_rep(1.0)
        assert tracker.best_ascent_s == 1.0
        assert tracker.snapshot().effort == "fresh"

    def test_set_reset_clears(self) -> None:
        tracker = _tracker()
        tracker.observe_rep(1.0)
        tracker.observe_rep(1.6)
        tracker.on_set_reset()
        assert tracker.best_ascent_s is None
        assert tracker.snapshot().effort == "fresh"

    def test_zero_ascent_ignored(self) -> None:
        tracker = _tracker()
        assert tracker.observe_rep(0.0).effort == "fresh"


class TestPromptLine:
    def test_none_when_not_confident(self) -> None:
        assert AthleteState(effort="working", affect="strained", confident=False).to_prompt_line() is None

    def test_none_when_default(self) -> None:
        assert AthleteState(confident=True).to_prompt_line() is None

    def test_two_fields(self) -> None:
        line = AthleteState(effort="near_limit", affect="frustrated", confident=True).to_prompt_line()
        assert line == "[athlete: effort=near_limit affect=frustrated]"

    def test_display_payload(self) -> None:
        payload = AthleteState(effort="working", affect="engaged", confident=True, arousal_z=1.234).to_display()
        assert payload["type"] == "affect"
        assert payload["arousal_z"] == 1.23
