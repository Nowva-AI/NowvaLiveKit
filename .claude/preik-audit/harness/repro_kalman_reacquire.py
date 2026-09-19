"""Minimal repro: FixedLagKeypointSmoother re-acquires onto a 2-frame consistent outlier burst.

Two consecutive gate-rejected measurements that agree within REACQUIRE_AGREEMENT_M (10 cm) trigger a snap
(contract C4). Real outlier bursts are consistent (RTMPose outliers run 1.6 frames mean, p90 3, constant offset;
L/R swaps last 2-6 frames), so a 2-frame burst 60 cm away with confidence 0.2 snaps the state to the outlier and
the lagged analysis stream carries the excursion; the good measurements that follow are then rejected for two more
frames and snap back. Run: /Users/naiahoard/NowvaLiveKit/venv/bin/python repro_kalman_reacquire.py
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/Users/naiahoard/NowvaLiveKit/src")

from biomechanics.utils.keypoint_kalman import FixedLagKeypointSmoother  # noqa: E402

FPS = 30.0
NUM_KPTS = 21
KNEE = 13
NOISE_M = 0.01
GOOD_CONF = 0.55
BURST_CONF = 0.2
BURST_OFFSET_M = 0.60
WARMUP_FRAMES = 60


def run(burst_frames: int, burst_conf: float, lag_frames: int = 2, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    base = np.zeros((NUM_KPTS, 3))
    base[:, 1] = np.linspace(-0.6, 0.95, NUM_KPTS)
    smoother = FixedLagKeypointSmoother(lag_frames=lag_frames)
    worst_lagged = 0.0
    worst_current = 0.0
    excursion_frames = 0
    for frame in range(WARMUP_FRAMES + burst_frames + 12):
        pts = base + rng.normal(0.0, NOISE_M, base.shape)
        conf = np.full(NUM_KPTS, GOOD_CONF)
        in_burst = WARMUP_FRAMES <= frame < WARMUP_FRAMES + burst_frames
        if in_burst:
            pts[KNEE, 0] += BURST_OFFSET_M  # consistent offset for the whole burst (same as a swap or an outlier run)
            conf[KNEE] = burst_conf
        out = smoother.update(pts, conf, frame / FPS)
        if frame >= WARMUP_FRAMES:
            err_lagged = float(np.linalg.norm(out.lagged_points[KNEE] - base[KNEE]))
            err_current = float(np.linalg.norm(out.current_points[KNEE] - base[KNEE]))
            worst_lagged = max(worst_lagged, err_lagged)
            worst_current = max(worst_current, err_current)
            excursion_frames += err_lagged > 0.10
    print(f"burst {burst_frames} frames at conf {burst_conf:.2f}, lag {lag_frames}: worst lagged knee error "
          f"{worst_lagged * 100:5.1f} cm over {excursion_frames} lagged frames > 10 cm; worst current (display) error "
          f"{worst_current * 100:5.1f} cm")


if __name__ == "__main__":
    for burst in (1, 2, 3):
        run(burst, BURST_CONF)
    run(2, GOOD_CONF)
    run(2, BURST_CONF, lag_frames=0)
