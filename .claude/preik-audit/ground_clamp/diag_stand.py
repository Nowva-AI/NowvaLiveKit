"""Stage-by-stage diagnosis of the fake heel-rise signal created by bone1 -> GroundClamp -> bone2 at squat bottom."""
from __future__ import annotations
import numpy as np
PHASE = "stand"
from sim import CK, FPS, NoiseModel, generate, measure
from run import SCENARIOS, _gate
from biomechanics.utils.bone_constraints import BoneLengthConstraints
from biomechanics.utils.ground_clamp import GroundClamp
from biomechanics.utils.types import Skeleton3D

gen = generate(SCENARIOS["clean"])
acc = {k: [] for k in ("raw", "bone1", "gc", "bone2")}
moves = {k: [] for k in ("gc_ankle_dy", "gc_ankle_dxz", "bone2_ankle_dy", "gc_tibia_change", "bone2_toe_dy")}
for seed in range(8):
    meas = measure(gen, NoiseModel(0.015, 0.006), seed)
    gate = _gate(); bones = BoneLengthConstraints(30, 0.0, gate)
    ground = GroundClamp(30, 0.02, 0.01, 0.75, gate)
    conf = np.full(19, 0.9)
    for i in range(len(gen["s"])):
        rel = meas["rel"][i, :19]
        sk = Skeleton3D.from_numpy(rel, confidences=conf, timestamp=i / FPS, frame_index=i)
        gate.check(sk)
        if not gate.is_ready:
            continue
        b1 = bones.enforce(sk).to_numpy()
        g = ground.clamp(Skeleton3D.from_numpy(b1, confidences=conf)).to_numpy()
        b2 = bones.enforce(Skeleton3D.from_numpy(g, confidences=conf)).to_numpy()
        if gen["phase"][i] == PHASE and gen["s"][i] == 0 and ground.is_calibrated and i > 40:
            for name, p in (("raw", rel), ("bone1", b1), ("gc", g), ("bone2", b2)):
                acc[name].append([p[CK.LEFT_FOOT_INDEX, 1] - p[CK.LEFT_ANKLE, 1], p[CK.RIGHT_FOOT_INDEX, 1] - p[CK.RIGHT_ANKLE, 1]])
            for side, (a, kn) in enumerate(((CK.LEFT_ANKLE, CK.LEFT_KNEE), (CK.RIGHT_ANKLE, CK.RIGHT_KNEE))):
                moves["gc_ankle_dy"].append(g[a, 1] - b1[a, 1])
                moves["gc_ankle_dxz"].append(np.linalg.norm((g[a] - b1[a])[[0, 2]]))
                moves["bone2_ankle_dy"].append(b2[a, 1] - g[a, 1])
                moves["gc_tibia_change"].append(np.linalg.norm(g[a] - g[kn]) - np.linalg.norm(b1[a] - b1[kn]))
            moves["bone2_toe_dy"].append(b2[CK.LEFT_FOOT_INDEX, 1] - g[CK.LEFT_FOOT_INDEX, 1])
ideal_at = gen["world"][gen["phase"] == PHASE]
print(f"true ankle-above-toe at {PHASE} (cm):", 100 * np.mean(ideal_at[:, CK.LEFT_FOOT_INDEX, 1] - ideal_at[:, CK.LEFT_ANKLE, 1]))
for k, v in acc.items():
    print(f"{k:6s} ankle-above-toe mean (cm) {100*np.mean(v):6.2f}")
for k, v in moves.items():
    v = np.array(v)
    print(f"{k:16s} mean {100*v.mean():6.2f} cm  mean|.| {100*np.abs(v).mean():5.2f} cm")
