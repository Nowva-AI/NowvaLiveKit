"""WS4: cost per frame of FixedLagKeypointSmoother (N=21) and closed-form check against fixedlag_fast.FixedLagCV."""
from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "/Users/naiahoard/NowvaLiveKit/src")
sys.path.insert(0, "/Users/naiahoard/NowvaLiveKit/.claude/preik-audit/smoothing")

from biomechanics.utils.keypoint_kalman import FixedLagKeypointSmoother  # noqa: E402
from fixedlag_fast import FixedLagCV  # noqa: E402

KEYPOINTS, FRAMES, FPS = 21, 3000, 30.0
CONF_FOR_1CM = 1.0 / (1.0 + (0.01 / 0.02) ** 2)  # u = 1 cm -> conf 0.8

rng = np.random.default_rng(0)
base = rng.normal(0.0, 0.3, (KEYPOINTS, 3))
ts = 1000.0 + np.arange(FRAMES) / FPS
points = base + 0.2 * np.sin(ts * 3.0)[:, None, None] + rng.normal(0.0, 0.01, (FRAMES, KEYPOINTS, 3))
conf = np.full((FRAMES, KEYPOINTS), CONF_FOR_1CM)
for lag in (0, 1, 2):
    timings = []
    for _ in range(5):
        smoother = FixedLagKeypointSmoother(lag_frames=lag)
        smoother.update(points[0], conf[0], ts[0])
        t0 = time.perf_counter()
        for i in range(1, FRAMES):
            smoother.update(points[i], conf[i], ts[i])
        timings.append((time.perf_counter() - t0) / (FRAMES - 1) * 1e6)
    reference = FixedLagCV(q=10.0, r_m=0.01, lag=lag, n=KEYPOINTS)
    ungated = FixedLagKeypointSmoother(lag_frames=lag, gate_sigma=1e9, gate_min_radius_m=1e9)
    max_diff = 0.0
    for i in range(FRAMES):
        a = reference.step(points[i], ts[i])
        b = ungated.update(points[i], conf[i], ts[i]).lagged_points
        if i > 5:
            max_diff = max(max_diff, float(np.abs(a - b).max()))
    print(f"lag {lag}: {min(timings):.1f} us/frame (best of 5, N={KEYPOINTS}), max |FixedLagCV - ours| {max_diff:.1e} m")
