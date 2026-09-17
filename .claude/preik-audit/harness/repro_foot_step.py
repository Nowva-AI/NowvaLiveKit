"""Minimal repro: FootContactModel keeps a foot planted at its old anchor through a normal walking step.

A planted foot that moves faster than ~8 cm per residual window (a 25 cm step in 0.5 s at 30 fps) fails the
per-frame outlier gate (max(OUTLIER_MIN_M = 8 cm, 6 sigma) from the anchor), so its observations are rejected
instead of counted as movement; the "moved" test never sees 3 inliers and the foot is only released by the
DISPLACED_RELEASE_FRAMES (45 frame, 1.5 s) displacement timeout. Compares a 25 cm step and a 6 cm step.
Run: /Users/naiahoard/NowvaLiveKit/venv/bin/python repro_foot_step.py
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/Users/naiahoard/NowvaLiveKit/src")

from biomechanics.utils.foot_contact import FootContactModel  # noqa: E402
from biomechanics.utils.types import CocoKeypoints as CK  # noqa: E402

FPS = 30.0
NOISE_M = 0.01
PLANT_FRAMES = 90
STEP_FRAMES = 15
AFTER_FRAMES = 90
STEP_LIFT_M = 0.07
NUM_KPTS = 21


def _standing(rng: np.random.Generator) -> np.ndarray:
    pts = np.zeros((NUM_KPTS, 3))
    pts[CK.LEFT_HIP], pts[CK.RIGHT_HIP] = (0.13, 0.0, 0.0), (-0.13, 0.0, 0.0)
    pts[CK.LEFT_SHOULDER], pts[CK.RIGHT_SHOULDER] = (0.19, -0.5, 0.0), (-0.19, -0.5, 0.0)
    pts[CK.LEFT_KNEE], pts[CK.RIGHT_KNEE] = (0.17, 0.45, -0.02), (-0.17, 0.45, -0.02)
    pts[CK.LEFT_ANKLE], pts[CK.RIGHT_ANKLE] = (0.20, 0.90, 0.0), (-0.20, 0.90, 0.0)
    pts[CK.LEFT_FOOT_INDEX], pts[CK.RIGHT_FOOT_INDEX] = (0.25, 0.95, -0.18), (-0.25, 0.95, -0.18)
    pts[CK.LEFT_HEEL], pts[CK.RIGHT_HEEL] = (0.20, 0.95, 0.05), (-0.20, 0.95, 0.05)
    return pts


def _min_jerk(tau: np.ndarray) -> np.ndarray:
    tau = np.clip(tau, 0.0, 1.0)
    return tau ** 3 * (10 - 15 * tau + 6 * tau ** 2)


def run(step_m: float, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    base = _standing(rng)
    model = FootContactModel()
    total = PLANT_FRAMES + STEP_FRAMES + AFTER_FRAMES
    conf = np.full(NUM_KPTS, 0.6)
    release_frame = None
    replant_frame = None
    worst_err = 0.0
    for frame in range(total):
        pts = base.copy()
        if frame >= PLANT_FRAMES:
            tau = (frame - PLANT_FRAMES) / STEP_FRAMES
            progress = _min_jerk(np.array([tau]))[0]
            lift = STEP_LIFT_M * np.sin(np.pi * min(tau, 1.0))
            for idx in (CK.LEFT_ANKLE, CK.LEFT_FOOT_INDEX, CK.LEFT_HEEL):
                pts[idx, 2] -= step_m * progress
                pts[idx, 1] -= lift
        noisy = pts + rng.normal(0.0, NOISE_M, pts.shape)
        out, state = model.update(noisy, conf, frame / FPS)
        if frame >= PLANT_FRAMES:
            err = float(np.linalg.norm(out[CK.LEFT_ANKLE] - pts[CK.LEFT_ANKLE]))
            worst_err = max(worst_err, err)
            if release_frame is None and not state.planted_l:
                release_frame = frame
            if release_frame is not None and replant_frame is None and state.planted_l:
                replant_frame = frame
    step_end = PLANT_FRAMES + STEP_FRAMES
    print(f"step {step_m * 100:.0f} cm in {STEP_FRAMES / FPS:.2f} s: released at frame "
          f"{release_frame} ({(release_frame - PLANT_FRAMES) / FPS if release_frame is not None else float('nan'):.2f} s "
          f"after step start; step ends at {STEP_FRAMES / FPS:.2f} s), re-planted at frame {replant_frame} "
          f"({(replant_frame - step_end) / FPS if replant_frame is not None else float('nan'):.2f} s after step end); "
          f"worst output-ankle error during/after step {worst_err * 100:.1f} cm")


if __name__ == "__main__":
    for step in (0.06, 0.12, 0.25):
        run(step)
