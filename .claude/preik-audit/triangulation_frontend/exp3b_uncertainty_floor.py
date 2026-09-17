"""Exp 3b: uncertainty with a prior 2D-noise floor; compare rank correlation + calibration, and production conf on
the SAME (best-subset) estimates. Same noise model as exp3."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
import synth
from exp3_confidence_semantics import FOCAL, geometric_sigma

OUT = Path(__file__).parent
RNG = np.random.default_rng(11)
P = synth.rig((-40.0, 0.0, 40.0))
seq, _ = synth.squat_sequence()
rec = {k: [] for k in ("err_prod", "conf_prod", "err_b", "n", "rms", "g", "uv_sigma_true")}
for trial in range(3):
    for f in range(0, len(seq), 2):
        gt = seq[f]
        uv, depth = synth.project(P, gt)
        sig = RNG.uniform(0.5, 8.0, uv.shape[:2])
        uv = uv + RNG.normal(size=uv.shape) * sig[..., None]
        outl = RNG.random(uv.shape[:2]) < 0.10
        mag = RNG.uniform(10, 150, uv.shape[:2]); ang = RNG.uniform(0, 2 * np.pi, uv.shape[:2])
        uv[..., 0] += outl * mag * np.cos(ang); uv[..., 1] += outl * mag * np.sin(ang)
        conf = np.clip(0.95 - 0.02 * sig + RNG.normal(0, 0.05, sig.shape), 0.3, 1.0)
        Xp = synth.dlt(P, uv)
        me = synth.reproj_err(P, Xp, uv).mean(0)
        rec["conf_prod"] += list(np.clip(conf.min(0) * np.where(me < 15, 1 - me / 15, 0.1), 0, 1))
        rec["err_prod"] += list(np.linalg.norm(Xp - gt, axis=1))
        Xb, n_used, rms = synth.best_subset(P, uv, 10.0)
        used = np.ones((3, gt.shape[0]), bool)
        for k in np.nonzero(n_used == 2)[0]:
            ek = synth.reproj_err(P, Xb[k:k + 1], uv[:, k:k + 1])[:, 0]
            used[np.argmax(ek), k] = False
        rec["err_b"] += list(np.linalg.norm(Xb - gt, axis=1))
        rec["n"] += list(n_used); rec["rms"] += list(rms); rec["g"] += list(geometric_sigma(P, Xb, used))
a = {k: np.array(v) for k, v in rec.items() if v}
res = {}
dof = np.where(a["n"] == 3, 3.0, 1.0)
sig_hat = a["rms"] * np.sqrt(2 * a["n"] / dof)
for floor in (1.0, 3.0, 5.0):
    u = np.maximum(sig_hat, floor) * a["g"]
    u2 = np.where(a["n"] == 2, np.maximum(u, floor * 2 * a["g"]), u)
    for name, uu in (("u", u), ("u_2view_doubled_floor", u2)):
        r3 = a["err_b"][a["n"] == 3] / uu[a["n"] == 3]
        r2 = a["err_b"][a["n"] == 2] / uu[a["n"] == 2]
        res[f"floor{floor:g}px_{name}"] = {"spearman": float(spearmanr(uu, a["err_b"]).correlation),
                                           "err/u pct25,50,75,95 3views": np.percentile(r3, [25, 50, 75, 95]).round(2).tolist(),
                                           "err/u pct25,50,75,95 2views": np.percentile(r2, [25, 50, 75, 95]).round(2).tolist()}
res["prod_conf_vs_prod_err_spearman(-conf)"] = float(spearmanr(-a["conf_prod"], a["err_prod"]).correlation)
res["prod_points_conf_lt_0.1_frac"] = float((a["conf_prod"] < 0.1).mean())
res["prod_points_conf_lt_0.1_median_err_mm"] = float(np.median(a["err_prod"][a["conf_prod"] < 0.1]) * 1000)
res["err_mm_mean prod vs bestsubset"] = [float(a["err_prod"].mean() * 1000), float(a["err_b"].mean() * 1000)]
res["err_mm_median prod vs bestsubset"] = [float(np.median(a["err_prod"]) * 1000), float(np.median(a["err_b"]) * 1000)]
(OUT / "exp3b_results.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
