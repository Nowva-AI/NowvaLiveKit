"""Tests for MultiCameraCapture clock alignment, sequence numbers and sync policy."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import biomechanics.triangulation.multi_capture as multi_capture_module
from biomechanics.triangulation.multi_capture import (
    MAX_WAIT_S,
    POLL_INTERVAL_S,
    STALE_CAMERA_S,
    MultiCameraCapture,
    SyncedFrames,
)

WALL_CLOCK_TOLERANCE_S = 1.0
FAKE_READ_LATENCY_S = 0.005
FIRST_FRAME_TIMEOUT_S = 1.0
TIME_TOLERANCE_S = 1e-9
BASE_TS = 1000.0
FRAME_PERIOD_S = 1.0 / 30.0
SYNC_DELTA_MS = 20.0

SIM_CAMERA_FPS = (30.00, 29.97, 30.03)
SIM_READ_LATENCY_MEAN_S = 0.008
SIM_READ_LATENCY_STD_S = 0.002
SIM_READ_LATENCY_MIN_S = 0.001
SIM_DURATION_S = 20.0
SIM_WARMUP_S = 0.5
SIM_PROCESSING_S = 0.015
SIM_CONSUMER_PERIOD_S = 1.0 / 30.0
SIM_MIN_PROCESSED_HZ = 24.0
SIM_MIN_FULL_SET_FRACTION = 0.9
SIM_SEED = 7


class _FakeVideoCapture:
    def __init__(self, device_id: int) -> None:
        self._device_id = device_id

    def set(self, prop: int, value: float) -> bool:
        return True

    def get(self, prop: int) -> float:
        return 30.0

    def isOpened(self) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray]:
        time.sleep(FAKE_READ_LATENCY_S)
        return True, np.zeros((2, 2, 3), dtype=np.uint8)

    def release(self) -> None:
        pass


class _ManualClock:
    def __init__(self, start_s: float = 0.0) -> None:
        self.now_s = start_s
        self.sleep_calls = 0
        self.on_sleep = None

    def clock(self) -> float:
        return self.now_s

    def sleep(self, duration_s: float) -> None:
        self.now_s += duration_s
        self.sleep_calls += 1
        if self.on_sleep is not None:
            self.on_sleep()


def _frame(tag: int) -> np.ndarray:
    return np.array([tag])


def _make_capture(
    clock: _ManualClock,
    device_ids: list[int] | None = None,
    min_views: int = 2,
    primary_device_id: int | None = None,
) -> MultiCameraCapture:
    return MultiCameraCapture(
        device_ids=device_ids if device_ids is not None else [0, 1, 2],
        max_sync_delta_ms=SYNC_DELTA_MS,
        min_views=min_views,
        primary_device_id=primary_device_id,
        clock_fn=clock.clock,
        sleep_fn=clock.sleep,
    )


def _push(capture: MultiCameraCapture, cam_id: str, timestamp: float, tag: int = 0) -> None:
    capture._append_frame(cam_id, _frame(tag), timestamp)


def _push_set(capture: MultiCameraCapture, timestamp: float, offsets_s: tuple[float, ...] = (0.0, 0.002, -0.002)) -> None:
    for cam_index, offset_s in enumerate(offsets_s):
        _push(capture, str(cam_index), timestamp + offset_s, tag=cam_index)


class TestCaptureClock:
    def test_timestamps_comparable_with_wall_clock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(multi_capture_module.cv2, "VideoCapture", _FakeVideoCapture)
        capture = MultiCameraCapture(device_ids=[0, 1])
        capture.start()
        try:
            synced = None
            deadline_s = time.perf_counter() + FIRST_FRAME_TIMEOUT_S
            while synced is None and time.perf_counter() < deadline_s:
                synced = capture.get_synced_frames()
            assert synced is not None
            assert synced.timestamp == pytest.approx(time.time(), abs=WALL_CLOCK_TOLERANCE_S)
        finally:
            capture.release()

    def test_reader_threads_assign_increasing_sequences(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(multi_capture_module.cv2, "VideoCapture", _FakeVideoCapture)
        capture = MultiCameraCapture(device_ids=[0, 1])
        capture.start()
        returned_sequences = []
        try:
            deadline_s = time.perf_counter() + FIRST_FRAME_TIMEOUT_S
            while len(returned_sequences) < 3 and time.perf_counter() < deadline_s:
                synced = capture.get_synced_frames()
                if synced is not None:
                    returned_sequences.append(synced.sequence)
        finally:
            capture.release()
        assert len(returned_sequences) == 3
        assert all(later > earlier for earlier, later in zip(returned_sequences, returned_sequences[1:]))


class TestSequenceNumbers:
    def test_sequences_count_up_per_camera(self) -> None:
        capture = _make_capture(_ManualClock())
        for frame_number in range(3):
            _push_set(capture, BASE_TS + frame_number * FRAME_PERIOD_S)
        _push(capture, "1", BASE_TS + 3 * FRAME_PERIOD_S)
        primary_sequences = [entry[0] for entry in capture._buffers["0"]]
        secondary_sequences = [entry[0] for entry in capture._buffers["1"]]
        assert primary_sequences == [0, 1, 2]
        assert secondary_sequences == [0, 1, 2, 3]


class TestSyncPolicy:
    def test_complete_set_returns_sequence_and_reference_timestamp(self) -> None:
        capture = _make_capture(_ManualClock())
        _push_set(capture, BASE_TS)
        synced = capture.get_synced_frames()
        assert isinstance(synced, SyncedFrames)
        assert set(synced.frames) == {"0", "1", "2"}
        assert synced.sequence == 0
        assert synced.timestamp == pytest.approx(BASE_TS, abs=TIME_TOLERANCE_S)

    def test_same_primary_sequence_never_returned_twice(self) -> None:
        clock = _ManualClock()
        capture = _make_capture(clock)
        _push_set(capture, BASE_TS)
        assert capture.get_synced_frames() is not None
        assert capture.get_synced_frames() is None

    def test_returns_none_after_bounded_wait(self) -> None:
        clock = _ManualClock()
        capture = _make_capture(clock)
        _push_set(capture, BASE_TS)
        capture.get_synced_frames()
        start_s = clock.now_s
        assert capture.get_synced_frames() is None
        waited_s = clock.now_s - start_s
        assert waited_s >= MAX_WAIT_S
        assert waited_s <= MAX_WAIT_S + 2 * POLL_INTERVAL_S

    def test_polls_until_new_set_arrives(self) -> None:
        clock = _ManualClock()
        capture = _make_capture(clock)
        _push_set(capture, BASE_TS)
        capture.get_synced_frames()

        def deliver_on_third_poll() -> None:
            if clock.sleep_calls == 3:
                _push_set(capture, BASE_TS + FRAME_PERIOD_S)

        clock.on_sleep = deliver_on_third_poll
        synced = capture.get_synced_frames()
        assert synced is not None
        assert synced.sequence == 1
        assert clock.sleep_calls == 3

    def test_older_unreturned_primary_skipped_after_newer_returned(self) -> None:
        capture = _make_capture(_ManualClock())
        for frame_number in range(3):
            _push_set(capture, BASE_TS + frame_number * FRAME_PERIOD_S)
        synced = capture.get_synced_frames()
        assert synced is not None
        assert synced.sequence == 2
        assert capture.get_synced_frames() is None

    def test_reference_waits_for_lagging_camera_to_catch_up(self) -> None:
        capture = _make_capture(_ManualClock())
        _push_set(capture, BASE_TS)
        _push(capture, "0", BASE_TS + FRAME_PERIOD_S)
        _push(capture, "1", BASE_TS + FRAME_PERIOD_S + 0.001)

        synced = capture.get_synced_frames()
        assert synced is not None
        assert synced.sequence == 0
        assert set(synced.frames) == {"0", "1", "2"}

        _push(capture, "2", BASE_TS + FRAME_PERIOD_S + 0.002)
        synced = capture.get_synced_frames()
        assert synced is not None
        assert synced.sequence == 1
        assert set(synced.frames) == {"0", "1", "2"}

    def test_nearest_frame_selected_per_camera(self) -> None:
        capture = _make_capture(_ManualClock())
        _push(capture, "0", BASE_TS)
        _push(capture, "1", BASE_TS - 0.015, tag=10)
        _push(capture, "1", BASE_TS + 0.004, tag=11)
        _push(capture, "1", BASE_TS + 0.037, tag=12)
        _push(capture, "2", BASE_TS + 0.001, tag=20)
        synced = capture.get_synced_frames()
        assert synced is not None
        assert int(synced.frames["1"][0]) == 11
        assert int(synced.frames["2"][0]) == 20

    def test_partial_set_accepted_with_min_views(self) -> None:
        capture = _make_capture(_ManualClock(), min_views=2)
        _push(capture, "0", BASE_TS)
        _push(capture, "1", BASE_TS + 0.003)
        _push(capture, "2", BASE_TS - 0.030)
        _push(capture, "2", BASE_TS + 0.030)
        synced = capture.get_synced_frames()
        assert synced is not None
        assert set(synced.frames) == {"0", "1"}

    def test_set_below_min_views_rejected(self) -> None:
        capture = _make_capture(_ManualClock(), min_views=3)
        _push(capture, "0", BASE_TS)
        _push(capture, "1", BASE_TS + 0.003)
        _push(capture, "2", BASE_TS - 0.030)
        _push(capture, "2", BASE_TS + 0.030)
        assert capture.get_synced_frames() is None

    def test_stalled_camera_does_not_block_sets(self) -> None:
        capture = _make_capture(_ManualClock())
        _push(capture, "2", BASE_TS)
        stalled_gap_s = STALE_CAMERA_S + FRAME_PERIOD_S
        _push(capture, "0", BASE_TS + stalled_gap_s)
        _push(capture, "1", BASE_TS + stalled_gap_s + 0.002)
        synced = capture.get_synced_frames()
        assert synced is not None
        assert set(synced.frames) == {"0", "1"}

    def test_primary_device_id_selects_reference_camera(self) -> None:
        capture = _make_capture(_ManualClock(), primary_device_id=2)
        _push(capture, "2", BASE_TS - FRAME_PERIOD_S)
        _push_set(capture, BASE_TS)
        synced = capture.get_synced_frames()
        assert synced is not None
        assert synced.sequence == 1
        assert synced.timestamp == pytest.approx(BASE_TS - 0.002, abs=TIME_TOLERANCE_S)

    def test_primary_not_in_device_ids_raises(self) -> None:
        with pytest.raises(ValueError):
            _make_capture(_ManualClock(), device_ids=[0, 1], primary_device_id=5)


class _SimulatedRig:
    def __init__(self, capture: MultiCameraCapture, clock: _ManualClock, arrivals_s: list[np.ndarray]) -> None:
        self._capture = capture
        self._clock = clock
        self._arrivals_s = arrivals_s
        self._next_frame = [0] * len(arrivals_s)

    def deliver(self) -> None:
        for cam_index, arrivals_s in enumerate(self._arrivals_s):
            frame_number = self._next_frame[cam_index]
            while frame_number < len(arrivals_s) and arrivals_s[frame_number] <= self._clock.now_s:
                self._capture._append_frame(str(cam_index), _frame(frame_number), float(arrivals_s[frame_number]))
                frame_number += 1
            self._next_frame[cam_index] = frame_number

    def advance(self, duration_s: float) -> None:
        self._clock.now_s += duration_s
        self.deliver()


def _simulated_arrivals(rng: np.random.Generator) -> list[np.ndarray]:
    arrivals_s = []
    for fps in SIM_CAMERA_FPS:
        frame_count = int((SIM_DURATION_S + 1.0) * fps)
        phase_s = rng.uniform(0.0, 1.0 / fps)
        capture_times_s = phase_s + np.arange(frame_count) / fps
        latencies_s = np.maximum(
            rng.normal(SIM_READ_LATENCY_MEAN_S, SIM_READ_LATENCY_STD_S, frame_count),
            SIM_READ_LATENCY_MIN_S,
        )
        # Sequential reads: a frame is never delivered before the previous one.
        arrivals_s.append(np.maximum.accumulate(capture_times_s + latencies_s))
    return arrivals_s


class TestSyncSimulation:
    def test_three_drifting_cameras_thirty_hz_consumer(self) -> None:
        rng = np.random.default_rng(SIM_SEED)
        clock = _ManualClock()
        capture = _make_capture(clock)
        rig = _SimulatedRig(capture, clock, _simulated_arrivals(rng))
        clock.on_sleep = rig.deliver
        rig.advance(SIM_WARMUP_S)

        returned_sequences = []
        full_sets = 0
        while clock.now_s < SIM_DURATION_S:
            loop_start_s = clock.now_s
            synced = capture.get_synced_frames()
            if synced is not None:
                returned_sequences.append(synced.sequence)
                full_sets += len(synced.frames) == len(SIM_CAMERA_FPS)
                rig.advance(SIM_PROCESSING_S)
            elapsed_s = clock.now_s - loop_start_s
            if elapsed_s < SIM_CONSUMER_PERIOD_S:
                rig.advance(SIM_CONSUMER_PERIOD_S - elapsed_s)

        processed_hz = len(returned_sequences) / (SIM_DURATION_S - SIM_WARMUP_S)
        duplicate_count = len(returned_sequences) - len(set(returned_sequences))
        assert duplicate_count == 0
        assert all(later > earlier for earlier, later in zip(returned_sequences, returned_sequences[1:]))
        assert processed_hz >= SIM_MIN_PROCESSED_HZ
        assert full_sets / len(returned_sequences) >= SIM_MIN_FULL_SET_FRACTION
