"""WS4 diagnostic: why WS3 per-keypoint confidence scores worse than frame-median R in the J harness."""
from __future__ import annotations

import numpy as np

import harness as H
import synth
from ws4_kalman_eval import LAG, SEEDS, c3_conf, run_production, ws3_noise

LEGS = [11, 12, 13, 14, 15, 16]


def ws3_parts(kind, seed, sigma):
    """Same as ws3_noise but also returns sigma_hat and meters_per_px separately."""
    d = ws3_noise(kind, seed, sigma)
    return d


for sigma in (2.0, 4.0, 8.0):
    js = {}
    stats = []
    for kind in ("smooth", "hard"):
        for seed in SEEDS:
            d = H.make_data(seed, sigma, kind=kind)
            w = ws3_noise(kind, seed, sigma)
            u = w["u"]
            r_frame = np.clip(0.005 * np.median(d["reproj"], axis=1), 0.003, 0.08)
            err = np.linalg.norm(synth.recenter(w["world"]) - d["truth"], axis=-1)[:, LEGS]
            err_axis = np.sqrt(np.mean((w["world"] - H._SESSION_CACHE[kind + "None"][0]) ** 2, axis=(0, 2)))
            stats.append((np.median(r_frame), np.median(u[:, LEGS]), np.percentile(u[:, LEGS], 10),
                          np.percentile(u[:, LEGS], 90), float(np.mean(err_axis[LEGS]))))
            # D: pooled u -> frame median of WS3 u per frame (keeps per-frame level, removes per-keypoint scatter)
            u_pool = np.repeat(np.median(u, axis=1, keepdims=True), u.shape[1], axis=1)
            # E: pooled but outlier-preserving: per-keypoint u only where it exceeds 3x the frame median
            u_keep = np.where(u > 3.0 * u_pool, u, u_pool)
            for name, conf in (("B ws3 per-kpt", w["conf"]), ("D ws3 frame-median u", c3_conf(u_pool)),
                               ("E ws3 median u, keep >3x outliers", c3_conf(u_keep)),
                               ("F ws3 per-kpt u / sqrt(3) (per-axis)", c3_conf(u / np.sqrt(3)))):
                out, _ = run_production(d["noisy"], conf, d["ts"])
                js.setdefault(name, {}).setdefault(kind, []).append(H.fast_eval(out, d, LAG)["J"])
    s = np.mean(np.array(stats), axis=0)
    print(f"sigma {sigma:g}px: median baseline r {s[0]*100:.2f} cm | WS3 legs u median {s[1]*100:.2f} cm "
          f"(p10 {s[2]*100:.2f}, p90 {s[3]*100:.2f}) | true world per-axis err std legs {s[4]*100:.2f} cm")
    for name, per in js.items():
        print(f"    {name:40s} J_mean {(np.mean(per['smooth']) + np.mean(per['hard'])) / 2:.2f}")
