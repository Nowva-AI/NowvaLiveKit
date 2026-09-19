"""WS4 B1 follow-up: burst-length sweep for the re-acquire rules (same setup as harness/repro_kalman_reacquire.py)."""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/Users/naiahoard/NowvaLiveKit/src")

from biomechanics.utils.keypoint_kalman import FixedLagKeypointSmoother  # noqa: E402

FPS, NUM_KPTS, KNEE, NOISE_M, GOOD_CONF, OFFSET_M, WARMUP = 30.0, 21, 13, 0.01, 0.55, 0.60, 60


def run(burst_frames: int, burst_conf: float, seed: int = 0) -> tuple[float, int, int, int]:
    rng = np.random.default_rng(seed)
    base = np.zeros((NUM_KPTS, 3))
    base[:, 1] = np.linspace(-0.6, 0.95, NUM_KPTS)
    smoother = FixedLagKeypointSmoother()
    worst, excursion, rejections, first_follow = 0.0, 0, 0, -1
    for frame in range(WARMUP + burst_frames + 14):
        pts = base + rng.normal(0.0, NOISE_M, base.shape)
        conf = np.full(NUM_KPTS, GOOD_CONF)
        if WARMUP <= frame < WARMUP + burst_frames:
            pts[KNEE, 0] += OFFSET_M
            conf[KNEE] = burst_conf
        out = smoother.update(pts, conf, frame / FPS)
        if frame >= WARMUP:
            err = float(np.linalg.norm(out.lagged_points[KNEE] - base[KNEE]))
            worst = max(worst, err)
            excursion += err > 0.10
            rejections += int(out.gate_rejected[KNEE])
            if first_follow < 0 and np.linalg.norm(out.current_points[KNEE] - base[KNEE]) > 0.30:
                first_follow = frame - WARMUP + 1
    return worst, excursion, rejections, first_follow


for conf in (0.20, 0.30, 0.55):
    for burst in range(1, 9):
        worst, excursion, rejections, first_follow = run(burst, conf)
        print(f"conf {conf:.2f} burst {burst}: worst lagged {worst * 100:5.1f} cm, lagged frames > 10 cm {excursion:2d}, "
              f"knee rejections {rejections:2d}, first frame following the burst {first_follow if first_follow > 0 else '-'}")
