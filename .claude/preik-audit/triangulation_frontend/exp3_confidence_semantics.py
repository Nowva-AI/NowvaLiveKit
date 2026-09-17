"""Exp 3: is triangulator confidence monotone with 3D error? Compare with proposed metric uncertainties.

Per keypoint, per view: heteroscedastic Gaussian noise sigma ~ U(0.5, 8) px, 10% chance of an outlier (10-150 px).
2D confidences are only weakly informative (as measured in exp1): conf = clip(0.95 - 0.02*sigma + N(0, .05)).
Also exp2b: best-subset heavy tail and a temporal tie-break.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import synth

OUT = Path(__file__).parent
RNG = np.random.default_rng(11)
FOCAL = 0.8 * synth.IMG_W
SIGMA_REF_M = 0.02


def geometric_sigma(P: np.ndarray, X: np.ndarray, views_mask: np.ndarray) -> np.ndarray:
    # sqrt(trace((J^T J)^-1)) per keypoint: 3D std (m) per 1 px of 2D noise, using the views actually used.
    K = X.shape[0]
    Xh = np.hstack([X, np.ones((K, 1))])
    out = np.zeros(K)
    for k in range(K):
        rows = []
        for v in np.nonzero(views_mask[:, k])[0]:
            p = P[v] @ Xh[k]
            for r in range(2):
                rows.append((P[v, r, :3] * p[2] - P[v, 2, :3] * p[r]) / p[2] ** 2)
        J = np.array(rows)
        out[k] = np.sqrt(np.trace(np.linalg.inv(J.T @ J)))
    return out


def main() -> None:
    P = synth.rig((-40.0, 0.0, 40.0))
    seq, _ = synth.squat_sequence()
    rows = []
    for trial in range(3):
        for f in range(0, len(seq), 2):
            gt = seq[f]
            uv, depth = synth.project(P, gt)
            sig = RNG.uniform(0.5, 8.0, uv.shape[:2])
            uv = uv + RNG.normal(size=uv.shape) * sig[..., None]
            outl = RNG.random(uv.shape[:2]) < 0.10
            mag = RNG.uniform(10, 150, uv.shape[:2])
            ang = RNG.uniform(0, 2 * np.pi, uv.shape[:2])
            uv[..., 0] += outl * mag * np.cos(ang)
            uv[..., 1] += outl * mag * np.sin(ang)
            conf = np.clip(0.95 - 0.02 * sig + RNG.normal(0, 0.05, sig.shape), 0.3, 1.0)
            # production
            Xp = synth.dlt(P, uv)
            e = synth.reproj_err(P, Xp, uv)
            mean_err = e.mean(0)
            prod_c = np.clip(conf.min(0) * np.where(mean_err < 15, 1 - mean_err / 15, 0.1), 0, 1)
            err_prod = np.linalg.norm(Xp - gt, axis=1)
            # best subset + uncertainty
            Xb, n_used, rms = synth.best_subset(P, uv, 10.0)
            err_b = np.linalg.norm(Xb - gt, axis=1)
            used = np.ones((3, gt.shape[0]), bool)
            if (n_used == 2).any():
                for k in np.nonzero(n_used == 2)[0]:
                    ek = synth.reproj_err(P, Xb[k:k + 1], uv[:, k:k + 1])[:, 0]
                    used[np.argmax(ek), k] = False
            zmean = np.array([depth[used[:, k], k].mean() for k in range(gt.shape[0])])
            u_simple = np.maximum(rms, 1.0) * zmean / FOCAL
            dof = np.where(n_used == 3, 3.0, 1.0)  # 2V - 3
            sigma_px_hat = np.maximum(rms * np.sqrt(2 * n_used / dof), 1.0)
            u_geom = sigma_px_hat * geometric_sigma(P, Xb, used)
            for k in range(gt.shape[0]):
                rows.append((prod_c[k], err_prod[k], err_b[k], u_simple[k], u_geom[k], n_used[k], rms[k]))
    a = np.array(rows)
    prod_c, err_prod, err_b, u_simple, u_geom, n_used, rms = a.T
    res = {
        "n_samples": int(len(a)),
        "spearman_prodconf_vs_prod_err": float(spearmanr(-prod_c, err_prod).correlation),
        "spearman_rms_vs_prod_err": float(spearmanr(rms, err_prod).correlation),
        "spearman_u_simple_vs_bestsubset_err": float(spearmanr(u_simple, err_b).correlation),
        "spearman_u_geom_vs_bestsubset_err": float(spearmanr(u_geom, err_b).correlation),
        "prod_err_mm_by_prodconf_bin": {},
        "bestsubset_err_mm_by_u_geom_bin": {},
        "err_over_u_geom_ratio_pct_3views": np.percentile(err_b[n_used == 3] / u_geom[n_used == 3], [25, 50, 75, 95]).round(2).tolist(),
        "err_over_u_geom_ratio_pct_2views": np.percentile(err_b[n_used == 2] / u_geom[n_used == 2], [25, 50, 75, 95]).round(2).tolist(),
        "bestsubset_err_mm_mean": float(err_b.mean() * 1000), "prod_err_mm_mean": float(err_prod.mean() * 1000),
        "bestsubset_err_mm_p95": float(np.percentile(err_b, 95) * 1000), "prod_err_mm_p95": float(np.percentile(err_prod, 95) * 1000),
        "frac_2view": float((n_used == 2).mean()),
    }
    for lo, hi in [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]:
        m = (prod_c >= lo) & (prod_c < hi)
        if m.any():
            res["prod_err_mm_by_prodconf_bin"][f"{lo}-{hi}"] = {"n": int(m.sum()), "median": float(np.median(err_prod[m]) * 1000), "p90": float(np.percentile(err_prod[m], 90) * 1000)}
    for lo, hi in [(0, 0.01), (0.01, 0.02), (0.02, 0.04), (0.04, 0.08), (0.08, 10)]:
        m = (u_geom >= lo) & (u_geom < hi)
        if m.any():
            res["bestsubset_err_mm_by_u_geom_bin"][f"{lo}-{hi}m"] = {"n": int(m.sum()), "median": float(np.median(err_b[m]) * 1000), "p90": float(np.percentile(err_b[m], 90) * 1000)}

    # ---------------------------------------------------------------- exp2b: heavy tail + temporal tie-break
    tail = {}
    for mag in (40, 80, 150):
        errs_b, errs_t = [], []
        for trial in range(3):
            prev = None
            for f in range(len(seq)):
                gt = seq[f]
                uv, _ = synth.project(P, gt)
                uv = uv + RNG.normal(0, 1.5, uv.shape)
                view = int(RNG.integers(0, 3))
                ang = RNG.uniform(0, 2 * np.pi)
                uv[view, 13] += mag * np.array([np.cos(ang), np.sin(ang)])
                Xb, n_used, _ = synth.best_subset(P, uv, 10.0)
                errs_b.append(np.linalg.norm(Xb[13] - gt[13]))
                # temporal: among triple + 3 pairs with own rms <= 4 px, choose closest to previous estimate
                cands = []
                X3 = synth.dlt(P, uv[:, 13:14])
                e3 = synth.reproj_err(P, X3, uv[:, 13:14])[:, 0]
                if e3.max() <= 10.0:
                    cands.append((0.0, X3[0]))
                else:
                    for drop in range(3):
                        keep = [v for v in range(3) if v != drop]
                        Xp = synth.dlt(P[keep], uv[keep, 13:14])
                        rp = np.sqrt((synth.reproj_err(P[keep], Xp, uv[keep, 13:14]) ** 2).mean())
                        cands.append((rp, Xp[0]))
                    cands.sort(key=lambda c: c[0])
                    good = [c for c in cands if c[0] <= 4.0]
                    if prev is not None and len(good) > 1:
                        cands = sorted(good, key=lambda c: np.linalg.norm(c[1] - prev))
                chosen = cands[0][1]
                prev = chosen
                errs_t.append(np.linalg.norm(chosen - gt[13]))
        eb, et = np.array(errs_b) * 1000, np.array(errs_t) * 1000
        tail[f"outlier_{mag}px"] = {"best_subset_mean_mm": float(eb.mean()), "best_subset_frac_gt50mm": float((eb > 50).mean()),
                                    "temporal_tiebreak_mean_mm": float(et.mean()), "temporal_frac_gt50mm": float((et > 50).mean())}
    res["exp2b_heavy_tail"] = tail
    (OUT / "exp3_results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
