"""Bone-length correction variants: current production class + prototypes (a)-(e). All operate on (19,3) arrays."""
from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, "/Users/naiahoard/NowvaLiveKit/src")

from biomechanics.utils.bone_constraints import BONE_PAIRS, BoneLengthConstraints, _pairs_present  # noqa: E402
from biomechanics.utils.types import CocoKeypoints as CK, Skeleton3D  # noqa: E402

PAIRS19 = _pairs_present(19)

# (a) pelvis-rooted tree: parent -> child
TREE_A = [
    (CK.LEFT_HIP, CK.LEFT_KNEE), (CK.LEFT_KNEE, CK.LEFT_ANKLE), (CK.LEFT_ANKLE, CK.LEFT_FOOT_INDEX),
    (CK.RIGHT_HIP, CK.RIGHT_KNEE), (CK.RIGHT_KNEE, CK.RIGHT_ANKLE), (CK.RIGHT_ANKLE, CK.RIGHT_FOOT_INDEX),
    (CK.LEFT_HIP, CK.LEFT_SHOULDER), (CK.LEFT_SHOULDER, CK.LEFT_ELBOW), (CK.LEFT_ELBOW, CK.LEFT_WRIST),
    (CK.RIGHT_HIP, CK.RIGHT_SHOULDER), (CK.RIGHT_SHOULDER, CK.RIGHT_ELBOW), (CK.RIGHT_ELBOW, CK.RIGHT_WRIST),
]

LIMBS_B = [
    (CK.LEFT_HIP, CK.LEFT_KNEE, CK.LEFT_ANKLE), (CK.RIGHT_HIP, CK.RIGHT_KNEE, CK.RIGHT_ANKLE),
    (CK.LEFT_SHOULDER, CK.LEFT_ELBOW, CK.LEFT_WRIST), (CK.RIGHT_SHOULDER, CK.RIGHT_ELBOW, CK.RIGHT_WRIST),
]


def lengths_from(pts: np.ndarray, pairs=PAIRS19) -> dict[tuple[int, int], float]:
    return {p: float(np.linalg.norm(pts[p[1]] - pts[p[0]])) for p in pairs}


def _length(lengths: dict, a: int, b: int) -> float | None:
    if (a, b) in lengths:
        return lengths[(a, b)]
    if (b, a) in lengths:
        return lengths[(b, a)]
    return None


def current_oracle(lengths: dict, tolerance: float = 0.0) -> BoneLengthConstraints:
    bc = BoneLengthConstraints(calibration_frames=1, tolerance=tolerance)
    bc._calibrated_lengths = dict(lengths)
    bc._calibrated = True
    return bc


def run_current(bc: BoneLengthConstraints, pts: np.ndarray, conf: np.ndarray | None = None) -> np.ndarray:
    conf = np.full(len(pts), 0.9) if conf is None else conf
    skel = Skeleton3D.from_numpy(pts, confidences=list(conf), timestamp=0.0, frame_index=0)
    return bc.enforce(skel).to_numpy()


def tree_a(pts: np.ndarray, lengths: dict) -> np.ndarray:
    out = pts.copy()
    for parent, child in TREE_A:
        target = _length(lengths, parent, child)
        vec = out[child] - out[parent]
        n = np.linalg.norm(vec)
        if target is None or n < 1e-9:
            continue
        out[child] = out[parent] + vec * (target / n)
    return out


def two_bone_b(pts: np.ndarray, lengths: dict, limbs=LIMBS_B) -> np.ndarray:
    """Keep root (hip/shoulder) and end (ankle/wrist) fixed; place the middle joint by law of cosines in its current plane."""
    out = pts.copy()
    for root, mid, end in limbs:
        a = _length(lengths, root, mid)
        b = _length(lengths, mid, end)
        if a is None or b is None:
            continue
        d_vec = out[end] - out[root]
        dist = float(np.linalg.norm(d_vec))
        if dist < 1e-6:
            continue
        u = d_vec / dist
        dist_c = min(max(dist, abs(a - b) + 1e-6), a + b - 1e-6)
        x = (a * a - b * b + dist_c * dist_c) / (2 * dist_c)
        r = math.sqrt(max(a * a - x * x, 0.0))
        perp = (out[mid] - out[root]) - np.dot(out[mid] - out[root], u) * u
        pn = np.linalg.norm(perp)
        if pn < 1e-9:
            out[mid] = out[root] + u * a * (dist / (a + b))
            continue
        out[mid] = out[root] + x * u + r * perp / pn
    return out


def symmetric_c(pts: np.ndarray, lengths: dict, conf: np.ndarray | None = None, pairs=PAIRS19) -> np.ndarray:
    """Single pass; each bone's radial error split between both ends in proportion to the other end's confidence."""
    conf = np.full(len(pts), 0.9) if conf is None else conf
    out = pts.copy()
    for p, d in pairs:
        target = lengths.get((p, d))
        if target is None:
            continue
        vec = out[d] - out[p]
        n = np.linalg.norm(vec)
        if n < 1e-9:
            continue
        excess = n - target
        w_p = conf[d] / (conf[p] + conf[d])  # low-confidence end moves more
        corr = excess * vec / n
        out[p] += w_p * corr
        out[d] -= (1.0 - w_p) * corr
    return out


def soft_d(pts: np.ndarray, lengths: dict, tol_m: float, conf: np.ndarray | None = None,
           pairs=PAIRS19, iters: int = 1) -> np.ndarray:
    """Deadband: only length error beyond tol_m is corrected (split symmetrically)."""
    conf = np.full(len(pts), 0.9) if conf is None else conf
    out = pts.copy()
    for _ in range(iters):
        for p, d in pairs:
            target = lengths.get((p, d))
            if target is None:
                continue
            vec = out[d] - out[p]
            n = np.linalg.norm(vec)
            if n < 1e-9:
                continue
            excess = n - target
            if abs(excess) <= tol_m:
                continue
            excess -= math.copysign(tol_m, excess)
            w_p = conf[d] / (conf[p] + conf[d])
            corr = excess * vec / n
            out[p] += w_p * corr
            out[d] -= (1.0 - w_p) * corr
    return out


def pbd_e(pts: np.ndarray, lengths: dict, iters: int = 5, conf: np.ndarray | None = None, pairs=PAIRS19,
          stiffness: float = 1.0) -> np.ndarray:
    conf = np.full(len(pts), 0.9) if conf is None else conf
    inv_mass = 1.0 / np.maximum(conf, 1e-3)
    out = pts.copy()
    for _ in range(iters):
        for p, d in pairs:
            target = lengths.get((p, d))
            if target is None:
                continue
            vec = out[d] - out[p]
            n = np.linalg.norm(vec)
            if n < 1e-9:
                continue
            c = (n - target) * stiffness
            wsum = inv_mass[p] + inv_mass[d]
            corr = c * vec / n
            out[p] += (inv_mass[p] / wsum) * corr
            out[d] -= (inv_mass[d] / wsum) * corr
    return out
