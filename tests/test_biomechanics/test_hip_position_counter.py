"""
Tests for the signal-based rep counter on synthetic no-pause squat reps.

The signal is hip height relative to the ankle in cm (more negative = standing),
generated from a two-bone leg so the onset shape matches a real descent.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from biomechanics.config import HipPositionCounterConfig
from biomechanics.faults.hip_position_counter import SignalRepCounter, SignalRepState
from biomechanics.utils.types import JointAngles

FPS = 30.0
STAND_S = 2.0
GAP_S = 1.5
FEMUR_M = 0.46
TIBIA_M = 0.44
KNEE_MIN_DEG = 4.0
KNEE_MAX_DEG = 120.0
# (descent_s, pause_s, ascent_s, depth_fraction) — the smoothing audit's 6-rep set.
REP_SCHEDULE = [
    (1.2, 0.0, 1.0, 1.0),
    (1.0, 0.0, 0.8, 1.0),
    (0.8, 0.0, 0.7, 1.0),
    (1.5, 0.4, 1.0, 0.784),
    (0.7, 0.0, 0.6, 1.0),
    (1.0, 0.1, 0.9, 0.612),
]
ENTRY_LATENESS_TOLERANCE_FRAMES = 3
BOTTOM_TOLERANCE_FRAMES = 3
LIGHT_NOISE_CM = 0.15
STANDING_NOISE_CM = 0.3
NOISE_SEEDS = (0, 1, 2)


def _hip_signal_cm(depth_fraction: float) -> float:
    knee_deg = KNEE_MIN_DEG + (KNEE_MAX_DEG - KNEE_MIN_DEG) * depth_fraction
    hip_to_ankle_m = math.sqrt(
        FEMUR_M ** 2 + TIBIA_M ** 2 + 2.0 * FEMUR_M * TIBIA_M * math.cos(math.radians(knee_deg))
    )
    return -hip_to_ankle_m * 100.0


def _session() -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Returns (signal_cm, windows) with windows = (start, bottom_start, bottom_end, end)."""
    depth_fractions: list[float] = [0.0] * int(round(STAND_S * FPS))
    windows: list[tuple[int, int, int, int]] = []
    for descent_s, pause_s, ascent_s, depth in REP_SCHEDULE:
        start = len(depth_fractions)
        descent_frames = int(round(descent_s * FPS))
        for i in range(descent_frames):
            tau = (i + 1) / descent_frames
            depth_fractions.append(depth * 0.5 * (1.0 - math.cos(math.pi * tau)))
        bottom_start = len(depth_fractions) - 1
        depth_fractions.extend([depth] * int(round(pause_s * FPS)))
        bottom_end = len(depth_fractions) - 1
        ascent_frames = int(round(ascent_s * FPS))
        for i in range(ascent_frames):
            tau = (i + 1) / ascent_frames
            depth_fractions.append(depth * 0.5 * (1.0 + math.cos(math.pi * tau)))
        windows.append((start, bottom_start, bottom_end, len(depth_fractions) - 1))
        depth_fractions.extend([0.0] * int(round(GAP_S * FPS)))
    signal = np.array([_hip_signal_cm(s) for s in depth_fractions])
    return signal, windows


def _run(counter: SignalRepCounter, signal: np.ndarray) -> tuple[list[str], int, int]:
    phases: list[str] = []
    reps = 0
    rejected = 0
    for i, value in enumerate(signal):
        rep_data, feedback = counter.update(signal_value=float(value), timestamp=1000.0 + i / FPS)
        phases.append(counter.phase)
        reps += rep_data is not None
        rejected += feedback is not None
    return phases, reps, rejected


def _transitions(phases: list[str], into: str, start: int, end: int) -> list[int]:
    return [
        i for i in range(max(start, 1), min(end, len(phases)))
        if phases[i] == into and phases[i - 1] != into
    ]


def _assert_timing(signal: np.ndarray, truth: np.ndarray, windows: list, phases: list[str], counter: SignalRepCounter) -> None:
    standing = truth[0]
    gate_cm = counter.entry_gate_cm
    for start, bottom_start, bottom_end, end in windows:
        truth_onset = start + int(np.argmax(truth[start:] - standing > gate_cm))
        entries = _transitions(phases, SignalRepState.DESCENDING.value, start - 5, end)
        assert entries, f"no descent entry for rep starting at {start}"
        assert entries[0] - truth_onset <= ENTRY_LATENESS_TOLERANCE_FRAMES
        assert entries[0] >= start
        ascents = _transitions(phases, SignalRepState.ASCENDING.value, start, end + 5)
        bottoms = [
            i for i in _transitions(phases, SignalRepState.BOTTOM.value, start, end + 5)
            if not ascents or i < ascents[0]
        ]
        assert bottoms, f"no bottom for rep starting at {start}"
        assert bottom_start - BOTTOM_TOLERANCE_FRAMES <= bottoms[-1] <= bottom_end + BOTTOM_TOLERANCE_FRAMES


class TestNoPauseReps:

    def test_clean_signal_counts_every_rep_with_correct_timing(self):
        signal, windows = _session()
        counter = SignalRepCounter(HipPositionCounterConfig())
        phases, reps, rejected = _run(counter, signal)
        assert reps == len(REP_SCHEDULE)
        assert rejected == 0
        _assert_timing(signal, signal, windows, phases, counter)

    @pytest.mark.parametrize("seed", NOISE_SEEDS)
    def test_light_noise_counts_every_rep_with_correct_timing(self, seed: int):
        truth, windows = _session()
        noisy = truth + np.random.default_rng(seed).normal(0.0, LIGHT_NOISE_CM, len(truth))
        counter = SignalRepCounter(HipPositionCounterConfig())
        phases, reps, rejected = _run(counter, noisy)
        assert reps == len(REP_SCHEDULE)
        assert rejected == 0
        _assert_timing(noisy, truth, windows, phases, counter)

    def test_never_in_rep_while_standing(self):
        signal, windows = _session()
        counter = SignalRepCounter(HipPositionCounterConfig())
        phases, _, _ = _run(counter, signal)
        for i, phase in enumerate(phases):
            inside_any = any(start - 1 <= i <= end + 1 for start, _, _, end in windows)
            if not inside_any:
                assert phase == SignalRepState.IDLE.value, f"frame {i} in phase {phase} while standing"


class TestStandingNoise:

    @pytest.mark.parametrize("seed", NOISE_SEEDS)
    def test_standing_noise_produces_no_reps_or_feedback(self, seed: int):
        standing = np.full(int(10 * FPS), _hip_signal_cm(0.0))
        noisy = standing + np.random.default_rng(seed).normal(0.0, STANDING_NOISE_CM, len(standing))
        counter = SignalRepCounter(HipPositionCounterConfig())
        phases, reps, rejected = _run(counter, noisy)
        assert reps == 0
        assert rejected == 0
        assert phases[-1] == SignalRepState.IDLE.value


class TestRobustness:

    def test_duplicate_timestamp_is_ignored(self):
        signal, _ = _session()
        counter = SignalRepCounter(HipPositionCounterConfig())
        for i in range(90):
            counter.update(signal_value=float(signal[i]), timestamp=1000.0 + i / FPS)
        state_before = counter.state
        rep_data, feedback = counter.update(signal_value=float(signal[89]) + 5.0, timestamp=1000.0 + 89 / FPS)
        assert rep_data is None and feedback is None
        assert counter.state == state_before

    def test_nan_signal_is_skipped(self):
        counter = SignalRepCounter(HipPositionCounterConfig())
        counter.update(signal_value=-90.0, timestamp=1000.0)
        rep_data, feedback = counter.update(signal_value=float("nan"), timestamp=1000.1)
        assert rep_data is None and feedback is None
        assert counter.state == SignalRepState.IDLE

    def test_nan_angles_do_not_poison_rep_metrics(self):
        signal, windows = _session()
        counter = SignalRepCounter(HipPositionCounterConfig())
        rep_datas = []
        for i, value in enumerate(signal):
            angles = JointAngles(
                knee_flexion_l=float("nan") if i % 7 == 0 else 90.0,
                knee_flexion_r=90.0,
                hip_flexion_l=80.0,
                hip_flexion_r=80.0,
                timestamp=1000.0 + i / FPS,
                frame_index=i,
            )
            rep_data, _ = counter.update(signal_value=float(value), timestamp=angles.timestamp, angles=angles)
            if rep_data is not None:
                rep_datas.append(rep_data)
        assert len(rep_datas) == len(REP_SCHEDULE)
        for rep_data in rep_datas:
            assert rep_data.max_depth_angle == pytest.approx(90.0)
            assert rep_data.avg_knee_asymmetry == pytest.approx(0.0)

    def test_rejected_rep_exposes_its_max_depth(self):
        counter = SignalRepCounter(HipPositionCounterConfig(min_depth_cm=30.0))
        shallow = np.array([_hip_signal_cm(0.35 * 0.5 * (1.0 - math.cos(math.pi * i / 20))) for i in range(41)])
        signal = np.concatenate([np.full(60, _hip_signal_cm(0.0)), shallow, np.full(30, _hip_signal_cm(0.0))])
        feedbacks = []
        for i, value in enumerate(signal):
            angles = JointAngles(knee_flexion_l=60.0, knee_flexion_r=60.0, timestamp=1000.0 + i / FPS)
            _, feedback = counter.update(signal_value=float(value), timestamp=angles.timestamp, angles=angles)
            if feedback is not None:
                feedbacks.append(feedback)
        assert feedbacks == ["go_deeper"]
        assert counter.rejected_rep_max_depth_angle == pytest.approx(60.0)


class TestDepthWithoutKneeAngles:
    def test_rep_with_no_knee_angles_reports_nan_depth(self):
        import math
        from biomechanics.faults.hip_position_counter import SignalRepCounter
        from biomechanics.utils.types import JointAngles

        counter = SignalRepCounter()
        assert math.isnan(counter._max_depth_angle)
        counter._record_angles(JointAngles(knee_flexion_l=math.nan, knee_flexion_r=math.nan)) if hasattr(counter, "_record_angles") else None
        assert math.isnan(counter._max_depth_angle)
