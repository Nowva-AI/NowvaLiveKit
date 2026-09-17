"""Prototype (f): rigid-bone residual outlier gate with attribution + constant-velocity repair (causal, O(bones))."""
from __future__ import annotations
import numpy as np
from biomechanics.utils.types import CocoKeypoints as CK

RIGID = [
    (CK.LEFT_HIP, CK.RIGHT_HIP),
    (CK.LEFT_HIP, CK.LEFT_KNEE), (CK.LEFT_KNEE, CK.LEFT_ANKLE), (CK.LEFT_ANKLE, CK.LEFT_FOOT_INDEX),
    (CK.RIGHT_HIP, CK.RIGHT_KNEE), (CK.RIGHT_KNEE, CK.RIGHT_ANKLE), (CK.RIGHT_ANKLE, CK.RIGHT_FOOT_INDEX),
]


class BoneResidualGate:
    def __init__(self, lengths: dict, sigma_len: dict, k_sigma: float = 3.0, max_hold: int = 5):
        self.lengths = lengths
        self.sigma = sigma_len
        self.k = k_sigma
        self.max_hold = max_hold
        self.last = None
        self.vel = None
        self.hold = np.zeros(19, dtype=int)

    def step(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        viol = {}
        for p, d in RIGID:
            r = abs(np.linalg.norm(pts[d] - pts[p]) - self.lengths[(p, d)])
            viol[(p, d)] = r > self.k * self.sigma[(p, d)]
        inc: dict[int, list] = {}
        for bone in RIGID:
            inc.setdefault(bone[0], []).append(bone)
            inc.setdefault(bone[1], []).append(bone)
        flagged = np.zeros(len(pts), dtype=bool)
        for j, bones in inc.items():
            nv = sum(viol[b] for b in bones)
            if len(bones) >= 2 and nv == len(bones):
                flagged[j] = True
            elif len(bones) == 1 and nv == 1:
                other = bones[0][0] if bones[0][1] == j else bones[0][1]
                if sum(viol[b] for b in inc[other]) == 1:
                    flagged[j] = True
        out = pts.copy()
        if self.last is not None:
            for j in np.nonzero(flagged)[0]:
                if self.hold[j] < self.max_hold:
                    out[j] = self.last[j] + self.vel[j]
                    self.hold[j] += 1
        self.hold[~flagged] = 0
        if self.last is None:
            self.vel = np.zeros_like(pts)
        else:
            self.vel = 0.5 * self.vel + 0.5 * (out - self.last)
        self.last = out
        return out, flagged
