"""Camera clocks + pipeline loop pacing + MultiCameraCapture.get_synced_frames() semantics.

Cameras free-run at ~30 Hz with their own phase/frequency; each frame is stamped with a perf_counter-like
clock on arrival (exposure + USB/decode latency + scheduler jitter) into a 5-deep ring buffer. The pipeline
loop independently calls get_synced_frames(): latest primary frame (possibly the one already processed ->
duplicate with identical timestamp) paired with the nearest secondary frames, None if any delta > 15 ms.
Loop period = max(1000/30, latency_ms) + untracked overhead (display, IPC) + sleep overshoot, like
pipeline_process.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DeliveryConfig:
    camera_fps: float = 30.0
    camera_freq_jitter_ratio: float = 0.0001
    secondary_phase_ms: float = 10.0
    exposure_jitter_ms: float = 0.3
    capture_latency_ms: float = 45.0
    capture_latency_spread_ms: float = 2.0
    stamp_jitter_ms: float = 1.5
    buffer_size: int = 5
    max_sync_delta_ms: float = 15.0
    target_fps: float = 30.0
    pose_latency_ms: float = 18.0
    pose_latency_sd_ms: float = 3.0
    overhead_ms: float = 1.5
    overhead_sd_ms: float = 1.0
    sleep_overshoot_ms: float = 0.4
    perf_counter_offset_s: float = 12345.678
    wall_clock_offset_s: float = 1.79e9
    mix_clocks: bool = True  # production dropout hold stamps held skeletons with time.time()


@dataclass
class CameraStream:
    exposure_s: np.ndarray
    stamp_s: np.ndarray
    latency_s: float


@dataclass
class Tick:
    index: int
    t_call_s: float
    frames: dict[str, int] | None
    ref_stamp_s: float | None


def simulate_cameras(cam_ids: list[str], duration_s: float, seed: int, cfg: DeliveryConfig
                     ) -> dict[str, CameraStream]:
    rng = np.random.default_rng([seed, 303])
    streams = {}
    primary_phase = rng.uniform(0.0, 1.0 / cfg.camera_fps)
    for idx, cam_id in enumerate(cam_ids):
        period = (1.0 / cfg.camera_fps) * (1.0 + rng.uniform(-cfg.camera_freq_jitter_ratio,
                                                              cfg.camera_freq_jitter_ratio))
        phase = primary_phase if idx == 0 else primary_phase + rng.uniform(
            -cfg.secondary_phase_ms, cfg.secondary_phase_ms) / 1000.0
        count = int((duration_s + 1.5) / period)
        exposure = -1.0 + phase + np.arange(count) * period + rng.normal(0, cfg.exposure_jitter_ms / 1000.0, count)
        latency = (cfg.capture_latency_ms + rng.uniform(-cfg.capture_latency_spread_ms,
                                                        cfg.capture_latency_spread_ms)) / 1000.0
        stamp = exposure + latency + np.abs(rng.normal(0, cfg.stamp_jitter_ms / 1000.0, count))
        stamp = np.maximum.accumulate(stamp)
        streams[cam_id] = CameraStream(exposure_s=exposure, stamp_s=stamp, latency_s=latency)
    return streams


def get_synced_frames(streams: dict[str, CameraStream], cam_ids: list[str], t_call_s: float, cfg: DeliveryConfig
                      ) -> tuple[dict[str, int] | None, float | None]:
    primary = streams[cam_ids[0]]
    last = int(np.searchsorted(primary.stamp_s, t_call_s, side="right")) - 1
    if last < 0:
        return None, None
    ref_stamp = float(primary.stamp_s[last])
    result = {cam_ids[0]: last}
    for cam_id in cam_ids[1:]:
        stream = streams[cam_id]
        newest = int(np.searchsorted(stream.stamp_s, t_call_s, side="right")) - 1
        if newest < 0:
            return None, None
        candidates = np.arange(max(0, newest - cfg.buffer_size + 1), newest + 1)
        deltas = np.abs(stream.stamp_s[candidates] - ref_stamp)
        best = int(np.argmin(deltas))
        if deltas[best] > cfg.max_sync_delta_ms / 1000.0:
            return None, None
        result[cam_id] = int(candidates[best])
    return result, ref_stamp


def simulate_loop(streams: dict[str, CameraStream], cam_ids: list[str], duration_s: float, seed: int,
                  cfg: DeliveryConfig) -> list[Tick]:
    rng = np.random.default_rng([seed, 404])
    ticks = []
    t_call = float(rng.uniform(0.0, 0.02))
    target_s = 1.0 / cfg.target_fps
    index = 0
    while t_call < duration_s:
        frames, ref_stamp = get_synced_frames(streams, cam_ids, t_call, cfg)
        ticks.append(Tick(index=index, t_call_s=t_call, frames=frames, ref_stamp_s=ref_stamp))
        latency = max(rng.normal(cfg.pose_latency_ms, cfg.pose_latency_sd_ms), 5.0) / 1000.0
        overhead = max(rng.normal(cfg.overhead_ms, cfg.overhead_sd_ms), 0.2) / 1000.0
        overshoot = rng.exponential(cfg.sleep_overshoot_ms) / 1000.0 if latency < target_s else 0.0
        t_call += max(target_s, latency) + overhead + overshoot
        index += 1
    return ticks
