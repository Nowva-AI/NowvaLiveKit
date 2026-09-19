"""Rep-segmented statistics of planted-foot keypoints on real single-camera RTMPose-m halpe26 output.

Uses cached *_h26.npy from real_foot_drift.py. Active windows chosen from real_traces.png (walk-in excluded).
For each foot keypoint: standing jitter, deviation at squat bottom from the standing median (systematic drift and
gross failures > 10 cm), and whether confidence flags the failures.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent
WINDOWS = {
    "squat_20260421_124127": (60, 440),
    "squat_20260427_143958": (240, 503),
    "squat_20260512_134408": (60, 250),
    "squat_20260515_145040": (25, 259),
    "squat_20260530_012118": (110, 667),
}
SHANK_M = 0.443
GROSS_CM = 10.0
KPTS = {"ankle_l": 15, "ankle_r": 16, "bigtoe_l": 20, "bigtoe_r": 21, "heel_l": 24, "heel_r": 25}


def main() -> None:
    summary: dict = {}
    pooled = {"good_conf": [], "gross_conf": []}
    for stem, (start, end) in WINDOWS.items():
        arr = np.load(OUT / f"{stem}_h26.npy")[start:end]
        hip_y = (arr[:, 11, 1] + arr[:, 12, 1]) / 2
        top, low = np.percentile(hip_y, 5), np.percentile(hip_y, 95)
        span = low - top
        standing = hip_y < top + 0.15 * span
        bottom = hip_y > low - 0.15 * span
        # scale from standing knee->ankle using robust median of the less-corrupted standing frames
        shank = np.linalg.norm(arr[standing][:, [13, 14], :2] - arr[standing][:, [15, 16], :2], axis=2)
        cm_per_px = 100 * SHANK_M / float(np.median(shank))
        res: dict = {"frames": int(end - start), "n_standing": int(standing.sum()), "n_bottom": int(bottom.sum()),
                     "cm_per_px": round(cm_per_px, 3), "hip_drop_cm": round(float(span * cm_per_px), 1)}
        for name, k in KPTS.items():
            xy = arr[:, k, :2]
            conf = arr[:, k, 2]
            ref = np.median(xy[standing], axis=0)
            dev_stand = (xy[standing] - ref) * cm_per_px
            dev_bottom = (xy[bottom] - ref) * cm_per_px
            dist_bottom = np.linalg.norm(dev_bottom, axis=1)
            dist_stand = np.linalg.norm(dev_stand, axis=1)
            d1 = np.diff(xy[standing], axis=0) * cm_per_px
            mad_jitter = 1.4826 * np.median(np.abs(d1 - np.median(d1, axis=0)), axis=0) / np.sqrt(2)
            gross_b = dist_bottom > GROSS_CM
            res[name] = {
                "jitter_standing_robust_cm_xy": [round(float(v), 2) for v in mad_jitter],
                "standing_gross_pct": round(100 * float(np.mean(dist_stand > GROSS_CM)), 1),
                "bottom_median_dev_cm_xy": [round(float(v), 2) for v in np.median(dev_bottom, axis=0)],
                "bottom_median_dist_cm": round(float(np.median(dist_bottom)), 2),
                "bottom_p90_dist_cm": round(float(np.percentile(dist_bottom, 90)), 2),
                "bottom_gross_pct": round(100 * float(np.mean(gross_b)), 1),
                "conf_bottom_gross_median": round(float(np.median(conf[bottom][gross_b])), 3) if gross_b.any() else None,
                "conf_bottom_good_median": round(float(np.median(conf[bottom][~gross_b])), 3) if (~gross_b).any() else None,
            }
            pooled["good_conf"].extend(conf[bottom][~gross_b].tolist())
            pooled["gross_conf"].extend(conf[bottom][gross_b].tolist())
        summary[stem] = res
    summary["pooled_conf"] = {
        "good_median": float(np.median(pooled["good_conf"])), "good_p10": float(np.percentile(pooled["good_conf"], 10)),
        "gross_median": float(np.median(pooled["gross_conf"])), "gross_p90": float(np.percentile(pooled["gross_conf"], 90)),
        "n_good": len(pooled["good_conf"]), "n_gross": len(pooled["gross_conf"]),
    }
    with open(OUT / "real_foot_stats.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    for stem, res in summary.items():
        if stem == "pooled_conf":
            print("pooled_conf", res)
            continue
        print(f"\n{stem} standing={res['n_standing']} bottom={res['n_bottom']} cm/px={res['cm_per_px']} hip_drop={res['hip_drop_cm']}")
        for name in KPTS:
            r = res[name]
            print(f"  {name:9s} jitter {r['jitter_standing_robust_cm_xy']} stand_gross {r['standing_gross_pct']:5.1f}% | bottom med dev {r['bottom_median_dev_cm_xy']} "
                  f"med {r['bottom_median_dist_cm']:5.1f} p90 {r['bottom_p90_dist_cm']:5.1f} gross {r['bottom_gross_pct']:5.1f}% conf gross/good {r['conf_bottom_gross_median']}/{r['conf_bottom_good_median']}")


if __name__ == "__main__":
    main()
