"""Simulated barbell end-point detections for the person-based calibration mode.

The bar rides on the upper back: centre a little behind and above the shoulder midpoint, axis along the
shoulder line. Each camera reports the two ends ordered by image x (as barbell_tracking/detector.py does) with
white + slow AR(1) pixel noise and rare gross outliers; an end outside the image means no detection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from biomechanics.utils.types import BarbellDetection, Keypoint2D

from .cameras import Camera

L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12


@dataclass
class BarNoise:
    white_px: float = 3.0
    slow_px: float = 3.0
    slow_tau_s: float = 1.0
    outlier_rate: float = 0.005
    outlier_px: tuple[float, float] = (30.0, 150.0)
    conf_mean: float = 0.8
    conf_sd: float = 0.05
    behind_m: float = 0.05
    above_m: float = 0.02


def bar_end_points(points_world: np.ndarray, bar_length_m: float, noise: BarNoise | None = None) -> np.ndarray:
    """points_world (F, 21, 3) -> bar ends (F, 2, 3): subject-left end, subject-right end."""
    noise = noise or BarNoise()
    left, right = points_world[:, L_SHOULDER], points_world[:, R_SHOULDER]
    axis = (left - right) / np.linalg.norm(left - right, axis=1, keepdims=True)
    shoulder_mid = 0.5 * (left + right)
    trunk_up = shoulder_mid - 0.5 * (points_world[:, L_HIP] + points_world[:, R_HIP])
    trunk_up /= np.linalg.norm(trunk_up, axis=1, keepdims=True)
    back = np.cross(axis, -trunk_up)  # X x Y(down) = subject's back
    centre = shoulder_mid + noise.behind_m * back + noise.above_m * trunk_up
    return np.stack([centre + 0.5 * bar_length_m * axis, centre - 0.5 * bar_length_m * axis], axis=1)


def detect_bar(camera: Camera, bar_world: np.ndarray, frame_times_s: np.ndarray, rng: np.random.Generator,
               noise: BarNoise | None = None, noiseless: bool = False) -> list[BarbellDetection | None]:
    noise = noise or BarNoise()
    frames = len(bar_world)
    uv, depth = camera.project(bar_world)
    if not noiseless:
        dt = float(np.median(np.diff(frame_times_s))) if frames > 1 else 1.0 / 30.0
        rho = math.exp(-dt / noise.slow_tau_s)
        slow = np.zeros((frames, 2, 2))
        slow[0] = rng.normal(0.0, 1.0, (2, 2))
        for k in range(1, frames):
            slow[k] = rho * slow[k - 1] + math.sqrt(1.0 - rho * rho) * rng.normal(0.0, 1.0, (2, 2))
        uv = uv + noise.white_px * rng.normal(0.0, 1.0, uv.shape) + noise.slow_px * slow
        outlier = rng.random((frames, 2)) < noise.outlier_rate
        magnitude = rng.uniform(noise.outlier_px[0], noise.outlier_px[1], (frames, 2))
        angle = rng.uniform(0.0, 2 * math.pi, (frames, 2))
        uv = uv + (outlier * magnitude)[..., None] * np.stack([np.cos(angle), np.sin(angle)], -1)
    conf = np.full((frames, 2), noise.conf_mean) if noiseless else np.clip(
        rng.normal(noise.conf_mean, noise.conf_sd, (frames, 2)), 0.5, 0.95)
    width, height = camera.resolution
    inside = ((uv[..., 0] >= 0) & (uv[..., 0] < width) & (uv[..., 1] >= 0) & (uv[..., 1] < height)
              & (depth > 0.1)).all(axis=1)
    detections: list[BarbellDetection | None] = []
    for k in range(frames):
        if not inside[k]:
            detections.append(None)
            continue
        order = np.argsort(uv[k, :, 0])  # detector convention: left_end = smaller image x
        ends = [Keypoint2D(x=float(uv[k, i, 0]), y=float(uv[k, i, 1]), confidence=float(conf[k, i])) for i in order]
        detections.append(BarbellDetection(left_end=ends[0], right_end=ends[1], bbox_conf=float(conf[k].min()),
                                           timestamp=float(frame_times_s[k]), frame_index=k))
    return detections
