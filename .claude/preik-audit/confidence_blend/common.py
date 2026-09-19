"""Shared helpers for the confidence_blend audit: realistic triangulated inputs + production blender runner."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/Users/naiahoard/NowvaLiveKit/src")

import synth_snapshot as synth  # noqa: E402
from biomechanics.utils.confidence_blend import ConfidenceBlender  # noqa: E402
from biomechanics.utils.types import CocoKeypoints as CK  # noqa: E402
from biomechanics.utils.types import Skeleton3D  # noqa: E402
from biomechanics.utils.velocity_clamp import VelocityClamp  # noqa: E402

STEMS = ["squat_20260530_012118", "squat_20260515_145040", "squat_20260512_134408"]


def real_conf_traces() -> list[np.ndarray]:
    return [np.load(HERE / f"{s}_rtm.npy")[:, :, 2] for s in STEMS]


def view_conf(T: int, rng: np.random.Generator, n_views: int = 3) -> np.ndarray:
    """(T,19,V) per-view RTMPose sigmoid confidences resampled from the real E0 traces (temporal structure kept)."""
    traces = real_conf_traces()
    out = np.empty((T, 19, n_views))
    for v in range(n_views):
        tr = traces[v % len(traces)]
        start = int(rng.integers(0, tr.shape[0]))
        idx = (start + np.arange(T)) % tr.shape[0]
        out[:, :, v] = tr[idx]
    return out


def triangulate(world: np.ndarray, rng: np.random.Generator, sigma_px: float = 3.0, outlier_p: float = 0.0,
                outlier_px: float = 40.0, P: np.ndarray | None = None, vconf: np.ndarray | None = None):
    """Production-like: project 3 cams, 2D noise + SimCC quantisation, unweighted DLT, triangulator.py confidence, hip recentre.
    Returns recentred (T,19,3), conf (T,19), reproj (T,19), abs world positions (T,19,3)."""
    P = synth.cameras() if P is None else P
    uv = synth.project(P, world)
    uv_noisy = uv + rng.normal(0.0, sigma_px, uv.shape)
    if outlier_p > 0:
        mask = rng.random(uv.shape[:-1]) < outlier_p
        uv_noisy += mask[..., None] * rng.normal(0.0, outlier_px, uv.shape)
    uv_noisy[..., 0] = np.round(uv_noisy[..., 0] / synth.QUANT_X_PX) * synth.QUANT_X_PX
    uv_noisy[..., 1] = np.round(uv_noisy[..., 1] / synth.QUANT_Y_PX) * synth.QUANT_Y_PX
    X = synth.dlt(P, uv_noisy)
    reproj = np.linalg.norm(synth.project(P, X) - uv_noisy, axis=-1).mean(axis=-1)
    vc = view_conf(world.shape[0], rng, P.shape[0]) if vconf is None else vconf
    base = vc.min(axis=-1)
    scale = np.where(reproj < 15.0, 1.0 - reproj / 15.0, 0.1)
    conf = np.clip(base * scale, 0.0, 1.0)
    return synth.recenter(X), conf, reproj, X


def run_blender(pts: np.ndarray, conf: np.ndarray, ts: np.ndarray | None = None,
                min_c: float = 0.1, max_c: float = 0.9) -> np.ndarray:
    """Runs the PRODUCTION ConfidenceBlender class frame by frame."""
    b = ConfidenceBlender(min_confidence=min_c, max_confidence=max_c)
    out = np.empty_like(pts)
    for t in range(pts.shape[0]):
        sk = Skeleton3D.from_numpy(pts[t], confidences=conf[t], timestamp=0.0 if ts is None else float(ts[t]))
        out[t] = b.blend(sk).to_numpy()
    return out


def run_blend_clamp(pts: np.ndarray, conf: np.ndarray, ts: np.ndarray, use_blend: bool = True,
                    use_clamp: bool = True, vmax: float = 2.5) -> np.ndarray:
    """PRODUCTION chain as preik_chain.py: blend -> velocity clamp."""
    b = ConfidenceBlender()
    c = VelocityClamp(max_velocity_m_per_s=vmax, target_fps=30)
    out = np.empty_like(pts)
    for t in range(pts.shape[0]):
        sk = Skeleton3D.from_numpy(pts[t], confidences=conf[t], timestamp=float(ts[t]))
        if use_blend:
            sk = b.blend(sk)
        if use_clamp:
            sk = c.clamp(sk)
        out[t] = sk.to_numpy()
    return out


def blend_weights(conf: np.ndarray, min_c: float = 0.1, max_c: float = 0.9) -> np.ndarray:
    return np.clip((conf - min_c) / (max_c - min_c), 0.0, 1.0)
