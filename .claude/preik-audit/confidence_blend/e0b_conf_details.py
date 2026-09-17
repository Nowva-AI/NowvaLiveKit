"""E0b: raw-logit view of RTMPose confidence, static jitter, and conf vs squat phase."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent
names = ["nose", "leye", "reye", "lear", "rear", "lsho", "rsho", "lelb", "relb", "lwri", "rwri",
         "lhip", "rhip", "lknee", "rknee", "lank", "rank", "ltoe", "rtoe"]
LEG = [11, 12, 13, 14, 15, 16, 17, 18]

for stem in ["squat_20260530_012118", "squat_20260515_145040", "squat_20260512_134408"]:
    arr = np.load(OUT / f"{stem}_rtm.npy")
    conf = arr[:, :, 2]
    raw = np.log(conf / (1 - conf))
    print(f"== {stem}: sigmoid conf overall min {conf.min():.3f} p1 {np.percentile(conf,1):.3f} "
          f"p50 {np.median(conf):.3f} p99 {np.percentile(conf,99):.3f} max {conf.max():.3f}")
    print(f"   raw max-logit: min {raw.min():.3f} p50 {np.median(raw):.3f} max {raw.max():.3f}")
    # squat phase from image hip y (pixels, y down => larger = lower)
    hip_y = arr[:, [11, 12], 1].mean(axis=1)
    lo, hi = np.percentile(hip_y, 5), np.percentile(hip_y, 95)
    depth = (hip_y - lo) / max(hi - lo, 1e-6)
    vel = np.abs(np.gradient(hip_y))
    standing = depth < 0.15
    bottom = depth > 0.85
    moving = (depth > 0.3) & (depth < 0.7)
    for label, m in [("standing", standing), ("mid-motion", moving), ("bottom", bottom)]:
        print(f"   {label:10s} n={m.sum():4d} leg conf mean {conf[m][:, LEG].mean():.3f} "
              f"min-over-leg mean {conf[m][:, LEG].min(axis=1).mean():.3f}")
    # static jitter: frames where hip speed is in lowest 30% and depth < 0.15
    slow = standing & (vel < np.percentile(vel, 30))
    if slow.sum() > 10:
        d = np.diff(arr[:, :, :2], axis=0)
        slow_pairs = slow[1:] & slow[:-1]
        fd = np.abs(d[slow_pairs])  # (n,19,2)
        print(f"   static frame-to-frame |dx| px median {np.median(fd[:,LEG,0]):.2f} p95 {np.percentile(fd[:,LEG,0],95):.2f};"
              f" |dy| median {np.median(fd[:,LEG,1]):.2f} p95 {np.percentile(fd[:,LEG,1],95):.2f} (n={slow_pairs.sum()})")
        print(f"   fraction of static frames with dx==0: {(fd[:,LEG,0]==0).mean():.2f}, dy==0: {(fd[:,LEG,1]==0).mean():.2f}")
