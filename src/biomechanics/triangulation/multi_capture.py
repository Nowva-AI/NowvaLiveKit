"""
Synchronized multi-camera capture.

Opens N cameras by device ID, reads frames in parallel threads stamped on one
wall-aligned clock with per-camera sequence numbers, and assembles synced sets
around the primary camera's frames.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from typing import Callable, NamedTuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MAX_SYNC_DELTA_MS = 20.0
DEFAULT_BUFFER_SIZE = 5
DEFAULT_MIN_VIEWS = 2
POLL_INTERVAL_S = 0.002
MAX_WAIT_S = 0.040
# A secondary camera has "caught up" to a reference once its newest frame is no
# earlier than the reference minus this margin (no later frame can match better).
CATCH_UP_MARGIN_S = 0.005
# A secondary whose newest frame trails the primary's newest by more than this is
# treated as stalled: sets stop waiting for it (it is still used if in tolerance).
STALE_CAMERA_S = 0.150
MIN_FPS_ELAPSED_S = 0.01
READER_JOIN_TIMEOUT_S = 2.0
DEFAULT_FPS = 30.0
MS_PER_S = 1000.0


class SyncedFrames(NamedTuple):
    frames: dict[str, np.ndarray]  # camera_id -> frame; primary always present
    sequence: int  # primary camera capture sequence number
    timestamp: float  # primary capture time, wall-aligned seconds


class MultiCameraCapture:
    """
    Synchronized capture from multiple USB cameras.

    Each camera gets its own reader thread that stamps frames with
    perf_counter() + offset (offset = time.time() - perf_counter() fixed at
    start), so timestamps are comparable with time.time(). A ring buffer per
    camera holds (sequence, timestamp, frame) entries.
    """

    def __init__(
        self,
        device_ids: list[int],
        resolution: tuple[int, int] = (1280, 720),
        max_sync_delta_ms: float = DEFAULT_MAX_SYNC_DELTA_MS,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        min_views: int = DEFAULT_MIN_VIEWS,
        primary_device_id: int | None = None,
        clock_fn: Callable[[], float] = time.perf_counter,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        if primary_device_id is None:
            primary_device_id = device_ids[0]
        if primary_device_id not in device_ids:
            raise ValueError(
                f"Primary camera {primary_device_id} is not in device_ids {device_ids}"
            )

        self._device_ids = device_ids
        self._resolution = resolution
        self._max_sync_delta_s = max_sync_delta_ms / MS_PER_S
        self._min_views = min_views
        self._clock_fn = clock_fn
        self._sleep_fn = sleep_fn

        self._primary_id = str(primary_device_id)
        self._secondary_ids = [
            str(dev_id) for dev_id in device_ids if dev_id != primary_device_id
        ]
        camera_ids = [str(dev_id) for dev_id in device_ids]

        self._caps: dict[str, cv2.VideoCapture] = {}
        self._actual_resolutions: dict[str, tuple[int, int]] = {}
        self._buffers: dict[str, deque[tuple[int, float, np.ndarray]]] = {
            cam_id: deque(maxlen=buffer_size) for cam_id in camera_ids
        }
        self._locks: dict[str, threading.Lock] = {
            cam_id: threading.Lock() for cam_id in camera_ids
        }
        self._next_sequence: dict[str, int] = {cam_id: 0 for cam_id in camera_ids}
        self._threads: dict[str, threading.Thread] = {}
        self._running = False

        self._last_returned_sequence = -1
        self._clock_offset_s = 0.0
        self._start_time: float = 0.0

    def start(self) -> None:
        """Open all cameras and start reader threads."""
        self._running = True
        self._clock_offset_s = time.time() - self._clock_fn()
        self._start_time = self._clock_fn()

        for dev_id in self._device_ids:
            cam_id = str(dev_id)
            cap = cv2.VideoCapture(dev_id)

            width, height = self._resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                raise RuntimeError(f"Could not open camera device {dev_id}")

            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
            logger.info(
                "Camera %s: %dx%d @ %.1f fps", cam_id, actual_width, actual_height, actual_fps
            )
            self._actual_resolutions[cam_id] = (actual_width, actual_height)

            self._caps[cam_id] = cap
            thread = threading.Thread(
                target=self._reader_loop, args=(cam_id,), daemon=True
            )
            self._threads[cam_id] = thread
            thread.start()

    @property
    def actual_resolutions(self) -> dict[str, tuple[int, int]]:
        """camera_id -> (width, height) each camera really opened at, which may differ from the request."""
        return dict(self._actual_resolutions)

    def _reader_loop(self, cam_id: str) -> None:
        cap = self._caps[cam_id]
        while self._running:
            ret, frame = cap.read()
            if ret and frame is not None:
                self._append_frame(cam_id, frame, self._clock_fn() + self._clock_offset_s)

    def _append_frame(self, cam_id: str, frame: np.ndarray, timestamp: float) -> None:
        with self._locks[cam_id]:
            sequence = self._next_sequence[cam_id]
            self._next_sequence[cam_id] = sequence + 1
            self._buffers[cam_id].append((sequence, timestamp, frame))

    def _select_synced_set(self) -> SyncedFrames | None:
        snapshots: dict[str, list[tuple[int, float, np.ndarray]]] = {}
        for cam_id, buffer in self._buffers.items():
            with self._locks[cam_id]:
                snapshots[cam_id] = list(buffer)

        primary_entries = snapshots[self._primary_id]
        if not primary_entries:
            return None
        primary_newest_ts = primary_entries[-1][1]

        horizon_ts = math.inf
        for cam_id in self._secondary_ids:
            entries = snapshots[cam_id]
            if not entries:
                continue
            newest_ts = entries[-1][1]
            if primary_newest_ts - newest_ts > STALE_CAMERA_S:
                continue
            horizon_ts = min(horizon_ts, newest_ts + CATCH_UP_MARGIN_S)

        for sequence, reference_ts, primary_frame in reversed(primary_entries):
            if sequence <= self._last_returned_sequence:
                return None
            if reference_ts > horizon_ts:
                continue

            frames = {self._primary_id: primary_frame}
            for cam_id in self._secondary_ids:
                entries = snapshots[cam_id]
                if not entries:
                    continue
                _, match_ts, match_frame = min(
                    entries, key=lambda entry: abs(entry[1] - reference_ts)
                )
                if abs(match_ts - reference_ts) <= self._max_sync_delta_s:
                    frames[cam_id] = match_frame

            if len(frames) >= self._min_views:
                return SyncedFrames(frames, sequence, reference_ts)
        return None

    def get_synced_frames(self) -> SyncedFrames | None:
        """
        Return the newest synced set whose primary frame has not been returned before.

        Reference = newest unreturned primary frame that every live secondary camera
        has caught up to. Each secondary contributes its nearest-timestamp frame if
        within max_sync_delta_ms, otherwise it is omitted; the set needs min_views
        cameras. Polls briefly for a new set before returning None. Sequence numbers
        returned are strictly increasing.
        """
        deadline_s = self._clock_fn() + MAX_WAIT_S
        while True:
            synced = self._select_synced_set()
            if synced is not None:
                self._last_returned_sequence = synced.sequence
                return synced
            if self._clock_fn() >= deadline_s:
                return None
            self._sleep_fn(POLL_INTERVAL_S)

    def get_fps_stats(self) -> dict[str, float]:
        """Return achieved FPS per camera since start."""
        elapsed_s = self._clock_fn() - self._start_time
        if elapsed_s < MIN_FPS_ELAPSED_S:
            return {cam_id: 0.0 for cam_id in self._next_sequence}
        return {
            cam_id: frame_count / elapsed_s
            for cam_id, frame_count in self._next_sequence.items()
        }

    def release(self) -> None:
        """Stop threads and release all VideoCapture handles."""
        self._running = False
        for thread in self._threads.values():
            thread.join(timeout=READER_JOIN_TIMEOUT_S)
        for cap in self._caps.values():
            cap.release()
        self._caps.clear()
        for buffer in self._buffers.values():
            buffer.clear()
        self._threads.clear()

    @staticmethod
    def detect_cameras(max_id: int = 10) -> list[int]:
        """
        Probe /dev/video0..N for valid capture devices.

        Linux creates two device nodes per USB camera (capture + metadata).
        This method only returns IDs that can actually capture frames.
        """
        valid = []
        for dev_id in range(max_id):
            cap = cv2.VideoCapture(dev_id)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    valid.append(dev_id)
            cap.release()
        return valid
