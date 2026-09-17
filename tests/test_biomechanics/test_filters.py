"""Tests for the temporal filter utilities in biomechanics.utils.filters."""

from __future__ import annotations

import math

import pytest

from biomechanics.utils.filters import OneEuroFilter, RunningMedian

ANGLE_TOL_DEG = 1e-9
SPIKE_DEG = 15.0
BASE_DEG = 100.0


class TestRunningMedian:

    def test_single_frame_spike_does_not_move_output(self):
        running = RunningMedian(window_frames=3)
        running.update(BASE_DEG)
        running.update(BASE_DEG)
        spiked = running.update(BASE_DEG + SPIKE_DEG)

        assert spiked == pytest.approx(BASE_DEG, abs=ANGLE_TOL_DEG)

    def test_sustained_change_passes_after_majority(self):
        running = RunningMedian(window_frames=3)
        running.update(BASE_DEG)
        running.update(BASE_DEG + SPIKE_DEG)
        assert running.update(BASE_DEG + SPIKE_DEG) == pytest.approx(BASE_DEG + SPIKE_DEG, abs=ANGLE_TOL_DEG)

    def test_nan_inputs_are_skipped(self):
        running = RunningMedian(window_frames=3)
        assert math.isnan(running.update(math.nan))
        running.update(BASE_DEG)
        assert running.update(math.nan) == pytest.approx(BASE_DEG, abs=ANGLE_TOL_DEG)

    def test_reset_forgets_history(self):
        running = RunningMedian(window_frames=3)
        running.update(BASE_DEG)
        running.reset()
        assert math.isnan(running.update(math.nan))

    def test_rejects_empty_window(self):
        with pytest.raises(ValueError):
            RunningMedian(window_frames=0)


class TestOneEuroFilter:

    def test_non_advancing_clock_re_emits_last_output(self):
        one_euro = OneEuroFilter(min_cutoff=1.0, beta=0.0)
        one_euro.filter(0.0, 0.0)
        smoothed = one_euro.filter(10.0, 1.0 / 30.0)

        assert one_euro.filter(50.0, 1.0 / 30.0) == pytest.approx(smoothed)

    def test_derivative_uses_previous_filtered_value(self):
        """Casiez's reference: the speed estimate comes from the filtered
        history, so two filters that agree on their filtered state agree on
        their next output regardless of the raw sample that preceded it."""
        left = OneEuroFilter(min_cutoff=1.0, beta=1.0)
        right = OneEuroFilter(min_cutoff=1.0, beta=1.0)
        left.filter(0.0, 0.0)
        right.filter(0.0, 0.0)
        # Same filtered state, different last raw samples.
        left.last_value = 5.0
        right.last_value = -5.0

        assert left.filter(1.0, 0.1) == pytest.approx(right.filter(1.0, 0.1))
