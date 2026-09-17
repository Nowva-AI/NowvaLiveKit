"""World-frame (un-centred) replica of DLTTriangulator for the world-gate variant; recentre helper."""
from __future__ import annotations
import numpy as np
from synth_vc import *


def triangulate_world(mv, cams, min_conf=0.3, max_reproj=15.0):
    ids = sorted(cams)
    uv = np.stack([mv.views[c].to_numpy()[:, :2] for c in ids]); cf = np.stack([mv.views[c].to_numpy()[:, 2] for c in ids])
    P = np.stack([cams[c]["P"] for c in ids])
    pts = np.zeros((N_KPTS, 3)); conf = np.zeros(N_KPTS)
    for k in range(N_KPTS):
        vs = [v for v in range(len(ids)) if cf[v, k] >= min_conf]
        if len(vs) < 2:
            continue
        A = []
        for v in vs:
            A.append(uv[v, k, 0] * P[v][2] - P[v][0]); A.append(uv[v, k, 1] * P[v][2] - P[v][1])
        X = np.linalg.svd(np.array(A))[2][-1]
        X = X[:3] / X[3]
        errs = []
        for v in vs:
            h = P[v] @ np.append(X, 1.0); errs.append(np.linalg.norm(h[:2] / h[2] - uv[v, k]))
        e = float(np.mean(errs))
        conf[k] = min(cf[vs, k]) * ((1 - e / max_reproj) if e < max_reproj else 0.1)
        pts[k] = X
    return pts, conf


def world_sequence(times, frames_world, cams, seed=0, sigma_px=2.0, corrupt_at=None):
    rng = np.random.default_rng(seed)
    out = []
    for i, (ts, pts) in enumerate(zip(times, frames_world)):
        corrupt = corrupt_at.get(i) if corrupt_at else None
        mv = views_for_frame(pts, cams, rng, sigma_px=sigma_px, corrupt=corrupt, timestamp=float(ts))
        out.append(triangulate_world(mv, cams))
    return out


def recentre(pts, conf):
    p = pts.copy()
    if conf[CK.LEFT_HIP] > 0 and conf[CK.RIGHT_HIP] > 0:
        centre = (p[CK.LEFT_HIP] + p[CK.RIGHT_HIP]) / 2.0
        p[conf > 0] -= centre
    return p
