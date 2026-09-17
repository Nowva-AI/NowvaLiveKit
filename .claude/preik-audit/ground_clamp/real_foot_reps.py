"""Per-rep planted-foot keypoint deviation (bottom vs the standing frames just before that rep), real RTMPose halpe26.

Avoids walk-in / re-positioning contamination: each rep is referenced to its own preceding standing pose.
Scale: that standing pose's knee->ankle pixel length = 0.443 m (approximate; frontal shank ~ vertical at standing).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks

OUT = Path(__file__).parent
KPTS = {"ankle_l": 15, "ankle_r": 16, "bigtoe_l": 20, "bigtoe_r": 21, "heel_l": 24, "heel_r": 25}
SHANK_M = 0.443
fps_of = {"squat_20260421_124127": 15, "squat_20260427_143958": 30, "squat_20260512_134408": 30,
          "squat_20260515_145040": 30, "squat_20260530_012118": 15}
summary = {}
for stem, fps in fps_of.items():
    a = np.load(OUT / f"{stem}_h26.npy")
    hy = (a[:, 11, 1] + a[:, 12, 1]) / 2
    hy_s = np.convolve(hy, np.ones(3) / 3, mode="same")
    span = np.percentile(hy_s, 95) - np.percentile(hy_s, 5)
    bottoms, _ = find_peaks(hy_s, prominence=0.5 * span, distance=int(fps))
    reps = []
    for b in bottoms:
        lo = max(0, b - int(2.5 * fps))
        s = lo + int(np.argmin(hy_s[lo:b]))
        if hy_s[b] - hy_s[s] < 0.5 * span:
            continue
        ref_idx = np.arange(max(0, s - 2), min(len(a), s + 3))
        bot_idx = np.arange(max(0, b - 1), min(len(a), b + 2))
        shank = np.median(np.linalg.norm(a[ref_idx][:, [13, 14], :2] - a[ref_idx][:, [15, 16], :2], axis=2))
        cm_px = 100 * SHANK_M / shank
        rep = {"stand_frame": int(s), "bottom_frame": int(b), "hip_drop_cm": round(float((hy_s[b] - hy_s[s]) * cm_px), 1)}
        for name, k in KPTS.items():
            ref = np.median(a[ref_idx, k, :2], axis=0)
            dev = (np.median(a[bot_idx, k, :2], axis=0) - ref) * cm_px
            rep[name] = [round(float(dev[0]), 1), round(float(dev[1]), 1)]
        reps.append(rep)
    dist = {name: [float(np.hypot(*r[name])) for r in reps] for name in KPTS}
    summary[stem] = {
        "n_reps": len(reps),
        "median_dist_cm": {n: round(float(np.median(v)), 1) for n, v in dist.items()},
        "max_dist_cm": {n: round(float(np.max(v)), 1) for n, v in dist.items()},
        "reps_gt10cm_pct": {n: round(100 * float(np.mean(np.array(v) > 10)), 0) for n, v in dist.items()},
        "reps": reps,
    }
    print(stem, "reps", len(reps), "hip drops", [r["hip_drop_cm"] for r in reps])
    print("   median |dev| cm", summary[stem]["median_dist_cm"])
    print("   reps >10cm %   ", summary[stem]["reps_gt10cm_pct"])
all_d = {n: [] for n in KPTS}
for stem in fps_of:
    for r in summary[stem]["reps"]:
        for n in KPTS:
            all_d[n].append(float(np.hypot(*r[n])))
pooled = {n: {"median": round(float(np.median(v)), 1), "p75": round(float(np.percentile(v, 75)), 1),
              "gt10_pct": round(100 * float(np.mean(np.array(v) > 10)), 0), "n": len(v)} for n, v in all_d.items()}
summary["pooled"] = pooled
print("pooled", pooled)
json.dump(summary, open(OUT / "real_foot_reps.json", "w"), indent=1)
