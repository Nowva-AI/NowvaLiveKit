"""
The press rep counter (docs/exercises/ohp.md §3.4): faults.press.counter on the
bar-above-shoulders signal, seeded at the rack, with the 0.40 s velocity window.
Noise-injected at 12 and 30 fps: no phantom rep over a minute in the rack, and
exact counts over ten-rep sets.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from biomechanics.config import BiomechanicsConfig
from biomechanics.faults.hip_position_counter import SignalRepCounter

COUNTER = BiomechanicsConfig().faults.press.counter
RACK_CM = 4.0
TOP_CM = 62.0
RACK_HOLD_S = 60.0
REPS_PER_SET = 10
HOLD_S = 1.0
PRESS_S = 0.8
LOCK_S = 0.4
LOWER_S = 1.0
SEED = 20261003
NOISE_CASES = [
    pytest.param(12.0, 1.0, id="12fps-sigma1.0cm"),
    pytest.param(30.0, 0.5, id="30fps-sigma0.5cm"),
    pytest.param(30.0, 1.0, id="30fps-sigma1.0cm"),
]


def _cosine(start: float, end: float, seconds: float, fps: float) -> list[float]:
    frames = int(round(seconds * fps))
    return [start + (end - start) * 0.5 * (1.0 - math.cos(math.pi * (i + 1) / frames)) for i in range(frames)]


def _rep(fps: float) -> list[float]:
    return (
        [RACK_CM] * int(round(HOLD_S * fps))
        + _cosine(RACK_CM, TOP_CM, PRESS_S, fps)
        + [TOP_CM] * int(round(LOCK_S * fps))
        + _cosine(TOP_CM, RACK_CM, LOWER_S, fps)
    )


def _count(signal: list[float], fps: float, sigma_cm: float = 0.0, seed: int = SEED) -> int:
    rng = np.random.default_rng(seed)
    counter = SignalRepCounter(config=COUNTER)
    counter.seed_baseline(RACK_CM)
    for i, value in enumerate(signal):
        counter.update(signal_value=value + rng.normal(0.0, sigma_cm), timestamp=i / fps)
    return counter.rep_count


class TestPressCounterConfig:
    def test_counter_config_is_the_plans(self):
        assert COUNTER.velocity_window_s == pytest.approx(0.40)
        assert COUNTER.entry_vel_threshold == pytest.approx(12.0)
        assert COUNTER.min_depth_cm == pytest.approx(20.0)
        assert COUNTER.min_rep_duration_s == pytest.approx(0.8)

    @pytest.mark.parametrize(("fps", "samples"), [(12.0, 5), (30.0, 12)])
    def test_velocity_window_holds_the_planned_samples(self, fps: float, samples: int):
        counter = SignalRepCounter(config=COUNTER)
        for i in range(40):
            counter.update(signal_value=RACK_CM, timestamp=i / fps)
        assert len(counter._signal_times) == samples

    def test_seeded_baseline_is_the_rack(self):
        counter = SignalRepCounter(config=COUNTER)
        counter.seed_baseline(RACK_CM)
        # A first sample already rising would otherwise become the baseline; seeded,
        # it only nudges the baseline's slow average and stays inside the entry gate.
        counter.update(signal_value=RACK_CM + 10.0, timestamp=0.0)
        assert counter._standing_baseline < RACK_CM + counter.entry_gate_cm


class TestPressCounterNoise:
    @pytest.mark.parametrize(("fps", "sigma_cm"), NOISE_CASES)
    def test_no_phantom_rep_over_a_minute_in_the_rack(self, fps: float, sigma_cm: float):
        assert _count([RACK_CM] * int(RACK_HOLD_S * fps), fps, sigma_cm) == 0

    @pytest.mark.parametrize(("fps", "sigma_cm"), NOISE_CASES)
    def test_ten_rep_set_counts_exactly(self, fps: float, sigma_cm: float):
        signal = []
        for _ in range(REPS_PER_SET):
            signal += _rep(fps)
        signal += [RACK_CM] * int(round(HOLD_S * fps))
        assert _count(signal, fps) == REPS_PER_SET
        assert _count(signal, fps, sigma_cm) == REPS_PER_SET

    @pytest.mark.parametrize("fps", [12.0, 30.0])
    def test_regrip_and_hook_clearance_are_not_reps(self, fps: float):
        small = [RACK_CM] * int(fps) + _cosine(RACK_CM, RACK_CM + 5.0, 0.4, fps) + _cosine(RACK_CM + 5.0, RACK_CM, 0.4, fps)
        assert _count(small * 3 + [RACK_CM] * int(fps), fps) == 0
