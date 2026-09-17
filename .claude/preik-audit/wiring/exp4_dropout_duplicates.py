"""E4: dropout-hold clock mixing, stale held skeleton after rest, duplicate frames -> derivative/predictive artefacts."""
import numpy as np
from harness import *

def trace(ts_clock: str, drop_frames: set[int], dup_frames: set[int], label: str):
    clock = FakeClock()
    pipe, prov = build_pipeline(clock)
    prov.frame_index_mode = "incrementing"
    prov.ts_clock = ts_clock
    depths = [0.0] * 40 + rep_depth_profile(stand_s=0.0)
    rows = []
    for i, d in enumerate(depths):
        pts = None if i in drop_frames else squat_points(d)
        res = step(pipe, prov, clock, pts, duplicate=(i in dup_frames))
        if res.joint_angles is None:
            continue
        truth_knee = 120.0 * d  # approx (synthetic knee flexion ~ 121*d)
        rows.append((i, d, res.joint_angles.knee_flexion_l,
                     pipe._derivative_tracker._prev_velocities.knee_velocity_l if pipe._derivative_tracker._prev_velocities else 0.0,
                     pipe._derivative_tracker._prev_velocities.knee_acceleration_l if pipe._derivative_tracker._prev_velocities else 0.0))
    return rows

def summarize(rows, window, label):
    print(f"-- {label}")
    for i, d, knee, vel, acc in rows:
        if i in window:
            print(f"   frame {i:3d} depth={d:.3f} knee_filt={knee:7.2f} knee_vel={vel:12.1f} knee_acc={acc:14.1f}")

base = trace("perf", set(), set(), "baseline")
drop_perf = trace("perf", {55, 56}, set(), "drop perf")
drop_time = trace("time", {55, 56}, set(), "drop time")
win = set(range(53, 61))
summarize(base, win, "baseline (no dropout), perf_counter timestamps")
summarize(drop_perf, win, "2-frame dropout, triangulated perf_counter timestamps (held frames stamped time.time())")
summarize(drop_time, win, "2-frame dropout, time.time() timestamps (MediaPipe-like clocks agree)")

# duplicates: every 3rd frame during the descent repeats the previous capture (camera slower than loop)
dups = {i for i in range(41, 70) if i % 3 == 0}
dup_rows = trace("perf", set(), dups, "dups")
b = {r[0]: r for r in base}; dd = {r[0]: r for r in dup_rows}
knee_err = [abs(dd[i][2] - b[i][2]) for i in range(41, 70)]
vel_ratio = [dd[i][3] / b[i][3] for i in range(45, 70) if abs(b[i][3]) > 1]
acc = [abs(dd[i][4]) for i in range(41, 70)]
acc_base = [abs(b[i][4]) for i in range(41, 70)]
print("-- duplicates every 3rd frame during descent (identical timestamp)")
print(f"   max |knee_filt dup - base| = {max(knee_err):.2f} deg")
print(f"   knee velocity dup/base ratio min={min(vel_ratio):.2f} max={max(vel_ratio):.2f}")
print(f"   max |knee accel| dup = {max(acc):.3g} deg/s^2 vs base {max(acc_base):.3g}")

# predictive estimator eval angle on duplicates vs base (0.2 s horizon, +-15 deg clip)
from biomechanics.utils.predictive_state import PredictiveStateEstimator
