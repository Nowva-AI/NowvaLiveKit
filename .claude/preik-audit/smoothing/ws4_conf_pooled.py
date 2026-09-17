"""WS4: recommended WS3 tweak — pool sigma_hat_px per frame (keep per-keypoint value when > 3x the frame median)."""
from __future__ import annotations

import numpy as np

import harness as H
from ws4_kalman_eval import LAG, SEEDS, c3_conf, run_production, ws3_noise

OUTLIER_RATIO = 3.0
for sigma in (2.0, 4.0, 8.0):
    js = {}
    rej = {}
    for kind in ("smooth", "hard"):
        for seed in SEEDS:
            d = H.make_data(seed, sigma, kind=kind)
            w = ws3_noise(kind, seed, sigma)
            sh = w["sigma_hat"]
            pooled = np.median(sh, axis=1, keepdims=True)
            sh_g = np.where(sh > OUTLIER_RATIO * pooled, sh, pooled)
            conf_g = c3_conf(np.maximum(sh_g, 3.0) * w["meters_per_px"])
            for name, (pts, conf, world) in (("B WS3 as implemented (hip-centred)", (d["noisy"], w["conf"], False)),
                                             ("G pooled sigma_hat (hip-centred)", (d["noisy"], conf_g, False)),
                                             ("G pooled sigma_hat (WORLD, recentre after)", (w["world"], conf_g, True))):
                out, n = run_production(pts, conf, d["ts"], world)
                js.setdefault(name, {}).setdefault(kind, []).append(H.fast_eval(out, d, LAG)["J"])
                rej[name] = rej.get(name, 0) + n
    for name, per in js.items():
        print(f"sigma {sigma:g}px  {name:45s} J_mean {(np.mean(per['smooth']) + np.mean(per['hard'])) / 2:.2f}  rejections {rej[name]}")
