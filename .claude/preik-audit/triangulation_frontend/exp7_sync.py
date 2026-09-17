"""Exp 7: MultiCameraCapture.get_synced_frames behaviour (simulated clocks, same algorithm), duplicate frames,
residual sync offset -> triangulation error, and perf_counter/time.time clock mixing on real DerivativeTracker.
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

import synth

sys.path.insert(0, "/Users/naiahoard/NowvaLiveKit/src")
from biomechanics.utils.derivatives import DerivativeTracker  # noqa: E402
from biomechanics.utils.types import JointAngles  # noqa: E402

OUT = Path(__file__).parent
RNG = np.random.default_rng(21)


def get_synced(stamps: list[np.ndarray], now: float, max_delta_s: float, buffer_size: int = 5):
    # identical selection logic to multi_capture.get_synced_frames, restricted to frames that have arrived at `now`
    # (stamps are sorted per camera; the stamp is the arrival time, as production stamps perf_counter at read()).
    n0 = int(np.searchsorted(stamps[0], now, side="right"))
    if n0 == 0:
        return None
    ref_ts = float(stamps[0][n0 - 1])
    out = [n0 - 1]
    deltas = []
    for st in stamps[1:]:
        n = int(np.searchsorted(st, now, side="right"))
        if n == 0:
            return None
        window = st[max(0, n - buffer_size):n]
        j = int(np.argmin(np.abs(window - ref_ts)))
        d = float(window[j] - ref_ts)
        if abs(d) > max_delta_s:
            return None
        out.append(max(0, n - buffer_size) + j)
        deltas.append(d)
    return ref_ts, out, deltas


def simulate(duration_s: float, fps_actual: list[float], process_ms: float, process_jitter_ms: float,
             max_delta_ms: float = 15.0, latency_ms: float = 25.0, latency_jitter_ms: float = 3.0,
             target_fps: float = 30.0) -> dict:
    n_cams = len(fps_actual)
    phases = RNG.uniform(0, 1 / 30, n_cams)
    frames = []
    for c in range(n_cams):
        n = int(duration_s * fps_actual[c])
        cap_t = phases[c] + np.arange(n) / fps_actual[c]
        # production stamps perf_counter when read() returns: capture + transfer/decode latency
        stamp = cap_t + (latency_ms + RNG.normal(0, latency_jitter_ms, n)) / 1000
        frames.append(np.sort(stamp))
    buffers = frames
    t = 0.2
    last_ref = None
    calls = none = dup = 0
    fail_run = max_fail_run = 0
    processed_ids = []
    abs_deltas = []
    period = 1 / target_fps
    while t < duration_s - 0.2:
        calls += 1
        r = get_synced(buffers, t, max_delta_ms / 1000)
        proc = max(process_ms + RNG.normal(0, process_jitter_ms), 1.0) / 1000
        if r is None:
            none += 1
            fail_run += 1
            max_fail_run = max(max_fail_run, fail_run)
        else:
            fail_run = 0
            ref_ts, ids, deltas = r
            if last_ref is not None and ref_ts == last_ref:
                dup += 1
            else:
                processed_ids.append(ids[0])
            last_ref = ref_ts
            abs_deltas += [abs(d) * 1000 for d in deltas]
        # pipeline_process paces to target_fps by sleeping the remainder of the period
        t += max(period, proc)
    ids = np.array(processed_ids)
    skipped = int(np.sum(np.diff(ids) - 1)) if len(ids) > 1 else 0
    return {"calls": calls, "sync_fail_frac": none / calls, "max_consecutive_fail_s": max_fail_run * period,
            "duplicate_frac_of_successful": dup / max(calls - none, 1), "skipped_primary_frames_frac": skipped / max(ids[-1] - ids[0], 1) if len(ids) > 1 else None,
            "accepted_abs_offset_ms_p50_p95_max": np.percentile(abs_deltas, [50, 95, 100]).round(1).tolist() if abs_deltas else None}


def sync_offset_error() -> dict:
    P = synth.rig()
    fps = 30.0
    period_s = 2.4
    res = {}
    t_axis = np.arange(0, period_s, 1 / fps)
    phase = lambda t: 0.5 - 0.5 * math.cos(2 * math.pi * t / period_s)  # noqa: E731
    hip_speed = max(np.linalg.norm(np.diff([synth.squat_skeleton(phase(t))[11] for t in t_axis], axis=0), axis=1) * fps)
    res["peak_hip_speed_m_s"] = float(hip_speed)
    knee_speed = max(np.linalg.norm(np.diff([synth.squat_skeleton(phase(t))[13] for t in t_axis], axis=0), axis=1) * fps)
    res["peak_knee_speed_m_s"] = float(knee_speed)
    for delta_ms in (5.0, 10.0, 15.0, 16.7, 33.3):
        errs, kf = [], []
        for t in t_axis:
            gt = synth.squat_skeleton(phase(t))
            late = synth.squat_skeleton(phase(t + delta_ms / 1000))
            uv, _ = synth.project(P, gt)
            uv_late, _ = synth.project(P, late)
            uv[2] = uv_late[2]  # one side camera is delta late
            X = synth.dlt(P, uv)
            errs.append(np.linalg.norm(X - gt, axis=1).max() * 1000)
            u, v = X[11] - X[13], X[15] - X[13]
            ug, vg = gt[11] - gt[13], gt[15] - gt[13]
            a = np.degrees(np.arccos(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)))
            ag = np.degrees(np.arccos(np.dot(ug, vg) / np.linalg.norm(ug) / np.linalg.norm(vg)))
            kf.append(abs(a - ag))
        res[f"offset_{delta_ms:g}ms"] = {"max_kpt_err_mm_peak": float(np.max(errs)), "max_kpt_err_mm_mean": float(np.mean(errs)),
                                          "knee_flex_err_deg_max": float(np.max(kf))}
    return res


def clock_mixing() -> dict:
    tracker = DerivativeTracker()
    base_perf = time.perf_counter()
    out = []
    knee = 60.0
    stamps = []
    for i in range(10):
        stamps.append(("perf", base_perf + i / 30))
    for i in range(2):
        stamps.append(("wall(dropout hold)", time.time() + i / 30))
    for i in range(10, 14):
        stamps.append(("perf", base_perf + i / 30))
    for i, (kind, ts) in enumerate(stamps):
        knee += 2.0  # steady 60 deg/s descent
        ang = JointAngles(knee_flexion_l=knee, knee_flexion_r=knee, hip_flexion_l=knee, hip_flexion_r=knee, timestamp=ts, frame_index=0)
        d = tracker.update(ang)
        out.append((kind, round(float(d.knee_velocity_l), 1)))
    # ipc_bridge.send_fault cooldown arithmetic
    held_fault_t = time.time()
    later_real_fault_t = time.perf_counter() + 60.0
    return {"knee_velocity_deg_s_sequence(true=60)": out,
            "fault_cooldown: now - last_send after a fault on a held frame (s)": later_real_fault_t - held_fault_t}


def main() -> None:
    res = {
        "mac_14ms_process_cams_30.00_29.97_30.03": simulate(600, [30.0, 29.97, 30.03], 14, 2),
        "mac_14ms_process_cams_30.000_30.001_29.999": simulate(600, [30.0, 30.001, 29.999], 14, 2),
        "jetson_45ms_process_cams_30.00_29.97_30.03": simulate(600, [30.0, 29.97, 30.03], 45, 5),
        "mac_14ms_no_latency_jitter": simulate(600, [30.0, 29.97, 30.03], 14, 2, latency_jitter_ms=0.0),
        "sync_offset_triangulation": sync_offset_error(),
        "clock_mixing": clock_mixing(),
    }
    (OUT / "exp7_results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
