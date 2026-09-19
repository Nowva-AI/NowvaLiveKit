"""Athlete state: z-scores → hysteresis counted in utterances → {effort, affect, confident} for the agent."""

from __future__ import annotations

import time
from collections import deque
from typing import Literal

import numpy as np
from pydantic import BaseModel

from affect.config import StateConfig

EffortState = Literal["fresh", "working", "near_limit"]
AffectStateName = Literal["flat", "engaged", "strained", "frustrated"]

EFFORT_ORDER: dict[str, int] = {"fresh": 0, "working": 1, "near_limit": 2}
AFFECT_DEFAULT: AffectStateName = "flat"
EFFORT_DEFAULT: EffortState = "fresh"
AROUSAL, DOMINANCE, VALENCE = 0, 1, 2
STATE_ITEM_PREFIX = "[athlete:"


class AthleteState(BaseModel):
    effort: EffortState = EFFORT_DEFAULT
    affect: AffectStateName = AFFECT_DEFAULT
    confident: bool = False
    fresh: bool = False
    updated_at: float = 0.0
    arousal_z: float = 0.0
    dominance_z: float = 0.0
    valence_z: float = 0.0
    source: str = "none"

    def is_default(self) -> bool:
        return self.effort == EFFORT_DEFAULT and self.affect == AFFECT_DEFAULT

    def to_prompt_line(self) -> str | None:
        """Two fields at most, nothing when unsure or when both fields are at their defaults."""
        if not self.confident or self.is_default():
            return None
        return f"{STATE_ITEM_PREFIX} effort={self.effort} affect={self.affect}]"

    def to_display(self) -> dict:
        return {
            "type": "affect",
            "effort": self.effort,
            "affect": self.affect,
            "confident": self.confident,
            "fresh": self.fresh,
            "arousal_z": round(self.arousal_z, 2),
            "valence_z": round(self.valence_z, 2),
            "dominance_z": round(self.dominance_z, 2),
        }


def classify_affect(z: np.ndarray, enter_z: float) -> tuple[AffectStateName, float]:
    """Candidate affect for a z-score triple and the magnitude that drives it."""
    arousal, valence = float(z[AROUSAL]), float(z[VALENCE])
    if valence <= -enter_z and arousal > 0.0:
        return "frustrated", abs(valence)
    if arousal >= enter_z and valence <= 0.0:
        return "strained", abs(arousal)
    if arousal >= enter_z and valence > 0.0:
        return "engaged", abs(arousal)
    return AFFECT_DEFAULT, max(abs(arousal), abs(valence))


def driving_magnitude(state: AffectStateName, z: np.ndarray) -> float:
    if state == "frustrated":
        return abs(float(z[VALENCE]))
    if state in ("strained", "engaged"):
        return abs(float(z[AROUSAL]))
    return 0.0


class AthleteStateTracker:
    """Turns per-utterance z-scores and per-rep tempo into a stable athlete state."""

    def __init__(self, config: StateConfig) -> None:
        self.config = config
        self._window: deque[tuple[float, np.ndarray]] = deque()
        self._affect: AffectStateName = AFFECT_DEFAULT
        self._effort: EffortState = EFFORT_DEFAULT
        self._candidate: AffectStateName | None = None
        self._candidate_count = 0
        self._candidate_time = 0.0
        self._dwell = 0
        self._confident = False
        self._best_ascent_s: float | None = None
        self._last_z = np.zeros(3)
        self._updated_at = 0.0
        self._fresh = False

    # -- utterances --------------------------------------------------------------------

    def _windowed_median(self, now: float) -> np.ndarray:
        while self._window and now - self._window[0][0] > self.config.window_seconds:
            self._window.popleft()
        while len(self._window) > self.config.window_utterances:
            self._window.popleft()
        if not self._window:
            return np.zeros(3)
        return np.median(np.stack([z for _, z in self._window]), axis=0)

    def observe_utterance(self, z: np.ndarray, now: float | None = None, confident: bool = True) -> AthleteState:
        now = time.time() if now is None else now
        z = np.asarray(z, dtype=np.float64)
        self._window.append((now, z))
        median = self._windowed_median(now)
        self._last_z = median
        self._confident = confident
        self._updated_at = now
        self._fresh = True
        candidate, magnitude = classify_affect(median, self.config.enter_z)
        self._step_affect(candidate, magnitude, median, now)
        return self.snapshot()

    def _step_affect(self, candidate: AffectStateName, magnitude: float, median: np.ndarray, now: float) -> None:
        cfg = self.config
        if self._candidate is not None and now - self._candidate_time > cfg.window_seconds:
            # Consecutive means within the window: a streak older than the window is stale.
            self._candidate = None
            self._candidate_count = 0
        if self._affect != AFFECT_DEFAULT:
            self._dwell += 1
            current_drive = driving_magnitude(self._affect, median)
            if candidate == self._affect:
                self._candidate = None
                self._candidate_count = 0
                return
            if self._dwell >= cfg.min_dwell_utterances and current_drive < cfg.exit_z:
                self._affect = AFFECT_DEFAULT
                self._dwell = 0
            elif self._dwell < cfg.min_dwell_utterances:
                return
        if candidate == AFFECT_DEFAULT:
            self._candidate = None
            self._candidate_count = 0
            return
        if candidate == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = candidate
            self._candidate_count = 1
        self._candidate_time = now
        enters = magnitude >= cfg.enter_z_single or self._candidate_count >= 2
        if enters and self._affect == AFFECT_DEFAULT:
            self._affect = candidate
            self._dwell = 0
            self._candidate = None
            self._candidate_count = 0
            if self._effort == EFFORT_DEFAULT and candidate == "strained":
                self._effort = "working"

    # -- reps --------------------------------------------------------------------------

    def observe_rep(self, ascent_time_s: float, now: float | None = None) -> AthleteState:
        if ascent_time_s <= 0.0:
            return self.snapshot()
        if self._best_ascent_s is None or ascent_time_s < self._best_ascent_s:
            self._best_ascent_s = ascent_time_s
        ratio = ascent_time_s / self._best_ascent_s
        if ratio >= self.config.effort_near_limit_ratio:
            effort: EffortState = "near_limit"
        elif ratio >= self.config.effort_working_ratio:
            effort = "working"
        else:
            effort = EFFORT_DEFAULT
        # Rep tempo is the stronger signal: it may lower effort when a faster rep resets the best.
        # Only the voice nudge (in _step_affect) is restricted to raising effort.
        self._effort = effort
        self._updated_at = time.time() if now is None else now
        return self.snapshot()

    def on_set_reset(self) -> None:
        self._best_ascent_s = None
        self._effort = EFFORT_DEFAULT

    def mark_stale(self) -> None:
        self._fresh = False

    # -- output ------------------------------------------------------------------------

    @property
    def best_ascent_s(self) -> float | None:
        return self._best_ascent_s

    def snapshot(self) -> AthleteState:
        return AthleteState(
            effort=self._effort,
            affect=self._affect,
            confident=self._confident,
            fresh=self._fresh,
            updated_at=self._updated_at,
            arousal_z=float(self._last_z[AROUSAL]),
            dominance_z=float(self._last_z[DOMINANCE]),
            valence_z=float(self._last_z[VALENCE]),
            source="tracker",
        )
