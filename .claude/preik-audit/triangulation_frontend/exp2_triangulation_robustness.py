"""Exp 2: synthetic 3-camera triangulation robustness.

Real DLT (parity-checked against DLTTriangulator: max diff 0.0) vs weighted / normalized / IRLS / best-subset /
L-R swap check, under Gaussian noise, single-view outliers and one-view L/R swaps.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import synth

OUT = Path(__file__).parent
RNG = np.random.default_rng(7)
MAX_REPROJ_PX = 15.0
BASE_CONF = 0.9


def knee_flex_deg(X: np.ndarray, side: str = "l") -> float:
    h, k, a = (11, 13, 15) if side == "l" else (12, 14, 16)
    u, v = X[h] - X[k], X[a] - X[k]
    c = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return 180.0 - np.degrees(np.arccos(np.clip(c, -1, 1)))


def prod_conf(P: np.ndarray, X: np.ndarray, uv: np.ndarray, conf: np.ndarray) -> np.ndarray:
    err = synth.reproj_err(P, X, uv).mean(0)
    scale = np.where(err < MAX_REPROJ_PX, 1.0 - err / MAX_REPROJ_PX, 0.1)
    return np.clip(conf.min(0) * scale, 0, 1)


def make_views(P: np.ndarray, gt: np.ndarray, sigma: float, scenario: str) -> tuple[np.ndarray, np.ndarray, dict]:
    uv, _ = synth.project(P, gt)  # (V,K,2)
    uv = uv + RNG.normal(0, sigma, uv.shape)
    conf = np.full(uv.shape[:2], BASE_CONF)
    info: dict = {"affected": [13]}
    if scenario.startswith("outlier"):
        mag = float(scenario.split("_")[1])
        view = int(RNG.integers(0, 3))
        ang = RNG.uniform(0, 2 * np.pi)
        uv[view, 13] += mag * np.array([np.cos(ang), np.sin(ang)])
        if scenario.endswith("lowconf"):
            conf[view, 13] = 0.5
        info["view"] = view
    elif scenario == "swap_full_view0":
        uv[0] = uv[0][synth.lr_perm(synth.L_PAIRS)]
        info["affected"] = [11, 12, 13, 14, 15, 16]
    elif scenario == "swap_legs_view2":
        uv[2] = uv[2][synth.lr_perm(synth.LEG_PAIRS)]
        info["affected"] = [11, 12, 13, 14, 15, 16]
    return uv, conf, info


def methods(P: np.ndarray, Ks: np.ndarray, tau: float) -> dict:
    return {
        "dlt_unweighted(prod)": lambda uv, cf: synth.dlt(P, uv),
        "dlt_conf_weighted": lambda uv, cf: synth.dlt(P, uv, cf),
        "dlt_row_normalized": lambda uv, cf: synth.dlt(P, uv, row_normalize=True),
        "dlt_hartley_Kinv": lambda uv, cf: synth.dlt_normalized(P, Ks, uv),
        "dlt_depth_irls2": lambda uv, cf: synth.dlt_depth_irls(P, uv),
        f"best_subset_tau{tau:g}": lambda uv, cf: synth.best_subset(P, uv, tau)[0],
        f"swapcheck+best_subset_tau{tau:g}": lambda uv, cf: synth.best_subset(P, synth.swap_check(P, uv)[0], tau)[0],
    }


def run(yaws: tuple[float, ...], tau: float = 10.0, n_trials: int = 2) -> dict:
    P = synth.rig(yaws)
    Ks = np.stack([synth.look_at_camera(y)[1] for y in yaws])
    seq, _ = synth.squat_sequence()
    scenarios = [("clean", 1.5), ("clean", 5.0), ("outlier_40", 1.5), ("outlier_80", 1.5), ("outlier_150", 1.5),
                 ("outlier_80_lowconf", 1.5), ("swap_full_view0", 1.5), ("swap_legs_view2", 1.5)]
    res: dict = {}
    for scen, sigma in scenarios:
        key = f"{scen}_sigma{sigma:g}"
        acc: dict = {}
        conf_aff = []
        for _ in range(n_trials):
            for f in range(0, len(seq), 2):
                gt = seq[f]
                uv, cf, info = make_views(P, gt, sigma, scen)
                Xp = synth.dlt(P, uv)
                conf_aff.append(prod_conf(P, Xp, uv, cf)[info["affected"]].mean())
                for name, fn in methods(P, Ks, tau).items():
                    X = fn(uv, cf)
                    err = np.linalg.norm(X - gt, axis=1) * 1000
                    a = acc.setdefault(name, {"aff": [], "all": [], "kflex": [], "knee_sep": []})
                    a["aff"].append(err[info["affected"]].mean())
                    a["all"].append(err.mean())
                    a["kflex"].append(abs(knee_flex_deg(X) - knee_flex_deg(gt)))
                    a["knee_sep"].append(abs(abs(X[13, 0] - X[14, 0]) - abs(gt[13, 0] - gt[14, 0])) * 1000)
        res[key] = {name: {"affected_err_mm_mean": float(np.mean(a["aff"])), "affected_err_mm_p95": float(np.percentile(a["aff"], 95)),
                           "all_err_mm_mean": float(np.mean(a["all"])), "knee_flex_err_deg_mean": float(np.mean(a["kflex"])),
                           "knee_flex_err_deg_p95": float(np.percentile(a["kflex"], 95)), "knee_sep_err_mm_mean": float(np.mean(a["knee_sep"]))}
                    for name, a in acc.items()}
        res[key]["_prod_conf_affected_mean"] = float(np.mean(conf_aff))
    return res


def timing(yaws: tuple[float, ...] = (-40.0, 0.0, 40.0), tau: float = 10.0) -> dict:
    P = synth.rig(yaws)
    Ks = np.stack([synth.look_at_camera(y)[1] for y in yaws])
    seq, _ = synth.squat_sequence()
    uv, cf, _ = make_views(P, seq[40], 1.5, "outlier_80")
    out = {}
    for name, fn in methods(P, Ks, tau).items():
        fn(uv, cf)
        t0 = time.perf_counter()
        n = 300
        for _ in range(n):
            fn(uv, cf)
        out[name] = (time.perf_counter() - t0) / n * 1000
    return out


def main() -> None:
    results = {
        "rig_-40_0_40": run((-40.0, 0.0, 40.0)),
        "rig_-70_0_70": run((-70.0, 0.0, 70.0)),
        "timing_ms_per_frame_21kpts_3views": timing(),
    }
    (OUT / "exp2_results.json").write_text(json.dumps(results, indent=1))
    for rig_name in ("rig_-40_0_40", "rig_-70_0_70"):
        print(f"\n=== {rig_name}")
        for scen, rows in results[rig_name].items():
            print(f"-- {scen}  prod conf on affected kpts: {rows['_prod_conf_affected_mean']:.3f}")
            for name, r in rows.items():
                if name.startswith("_"):
                    continue
                print(f"   {name:34s} aff {r['affected_err_mm_mean']:7.1f} mm (p95 {r['affected_err_mm_p95']:7.1f}) all {r['all_err_mm_mean']:6.1f} mm"
                      f"  kneeflex {r['knee_flex_err_deg_mean']:5.2f} deg (p95 {r['knee_flex_err_deg_p95']:5.2f})  kneesep {r['knee_sep_err_mm_mean']:6.1f} mm")
    print("\ntiming ms/frame:", json.dumps(results["timing_ms_per_frame_21kpts_3views"], indent=1))


if __name__ == "__main__":
    main()
