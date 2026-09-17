"""E8: per-frame cost of the chain as wired (Skeleton3D <-> numpy round-trip per stage) vs array-only stage math."""
import time, sys
import numpy as np
from harness import *
from biomechanics.utils.types import Skeleton3D
from biomechanics.utils.confidence_blend import ConfidenceBlender
from biomechanics.utils.velocity_clamp import VelocityClamp
from biomechanics.utils.bone_constraints import BoneLengthConstraints
from biomechanics.utils.ground_clamp import GroundClamp
from biomechanics.utils.position_filter import KeypointPositionSmoother

N = 600
frames = [squat_points(d) for d in (rep_depth_profile() * 3)[:N]]
skels = [Skeleton3D.from_numpy(p, confidences=[0.9] * 19, timestamp=i / 30.0, frame_index=i) for i, p in enumerate(frames)]

def bench(fn, label):
    t0 = time.perf_counter()
    for s in skels:
        fn(s)
    ms = (time.perf_counter() - t0) * 1000.0 / len(skels)
    print(f"{label:60s} {ms:7.3f} ms/frame")

b, v = ConfidenceBlender(), VelocityClamp()
bench(lambda s: v.clamp(b.blend(s)), "production chain (blend+clamp)")
b2, v2, bc, g, sm = ConfidenceBlender(), VelocityClamp(), BoneLengthConstraints(calibration_frames=5), GroundClamp(calibration_frames=5), KeypointPositionSmoother()
def full(s):
    s = v2.clamp(b2.blend(s)); s = bc.enforce(s); s = g.clamp(s); s = sm.smooth(s); return bc.enforce(s)
bench(full, "full 6-stage chain (as preik_chain comments)")
bench(lambda s: Skeleton3D.from_numpy(s.to_numpy(), confidences=[kp.confidence for kp in s.keypoints], timestamp=s.timestamp, frame_index=s.frame_index), "one Skeleton3D to_numpy+from_numpy round trip")
arr = np.stack(frames); prev = arr[0].copy()
t0 = time.perf_counter()
for x in arr:
    d = x - prev; n = np.linalg.norm(d, axis=1); m = n > 0.083
    out = np.where(m[:, None], prev + d / np.maximum(n, 1e-9)[:, None] * 0.083, x); prev = out
print(f"{'array-only velocity clamp math':60s} {(time.perf_counter()-t0)*1000/len(arr):7.3f} ms/frame")
