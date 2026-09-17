"""E3b: rigid subset (pelvis+legs+feet) projections, soft PBD, and residual-gate outlier repair; outlier-focused metrics."""
from __future__ import annotations
import json
from multiprocessing import Pool
import numpy as np
from synth import sequence, add_correlated_noise, apply_semantic_drift
from metrics import summarize, seq_metrics, ta_joint_errors
from methods import PAIRS19, lengths_from, symmetric_c, pbd_e, soft_d
from detect import RIGID, BoneResidualGate
from e3_sequences import prod_calibration, CASES, FAULT_METRIC

N_FRAMES = 420
NOISES = {
    "real_2cm": dict(sigma_m=0.02, z_scale=1.5, rho=0.6, outlier_p=0.01, outlier_burst=3),
    "real_2cm_drift": dict(sigma_m=0.02, z_scale=1.5, rho=0.6, outlier_p=0.01, outlier_burst=3, drift=True),
    "outlier_heavy": dict(sigma_m=0.015, z_scale=1.5, rho=0.6, outlier_p=0.03, outlier_burst=2),
}


def job(args):
    noise_name, case_name, seed = args
    cfg = dict(NOISES[noise_name]); drift = cfg.pop("drift", False)
    rng = np.random.default_rng(seed)
    fault = dict(CASES[case_name])
    if drift:
        fault["spine_shorten_m"] = 0.04
    truth, s = sequence(N_FRAMES, **fault)
    measured = apply_semantic_drift(truth, s, hip_fwd_m=0.03, hip_down_m=0.02, knee_fwd_m=0.02) if drift else truth
    noisy, mask = add_correlated_noise(measured, rng, **cfg)
    bc, used = prod_calibration(noisy)
    L = dict(bc._calibrated_lengths)
    obs = np.array([[np.linalg.norm(noisy[i][d] - noisy[i][p]) for p, d in RIGID] for i in used])
    rstd = 1.4826 * np.median(np.abs(obs - np.median(obs, axis=0)), axis=0)
    sig = {b: max(sd, 0.01) for b, sd in zip(RIGID, rstd)}

    def run_gate(k):
        g = BoneResidualGate(L, sig, k_sigma=k)
        outs, flags = [], []
        for p in noisy:
            o, f = g.step(p); outs.append(o); flags.append(f)
        return np.stack(outs), np.stack(flags)

    def soft_pbd(p, k=1.5, iters=3):
        out = p.copy()
        for _ in range(iters):
            for b in RIGID:
                out = soft_d(out, L, k * sig[b], pairs=[b])
        return out

    ests = {
        "none": noisy,
        "current_cal": np.stack([bc.enforce(__import__("synth").to_skel(p)).to_numpy() for p in noisy]),
        "e_pbd5_all": np.stack([pbd_e(p, L, 5) for p in noisy]),
        "c_symm_rigid": np.stack([symmetric_c(p, L, pairs=RIGID) for p in noisy]),
        "e_pbd5_rigid": np.stack([pbd_e(p, L, 5, pairs=RIGID) for p in noisy]),
        "d_softpbd_rigid1.5sd": np.stack([soft_pbd(p) for p in noisy]),
    }
    g3, flags3 = run_gate(3.0)
    ests["f_gate3sd"] = g3
    ests["f_gate3sd+e_rigid"] = np.stack([pbd_e(p, L, 5, pairs=RIGID) for p in g3])
    truth_m = seq_metrics(truth)
    eval_frames = np.arange(60, N_FRAMES)
    bottom = np.nonzero(s > 0.9)[0]
    res = {}
    legs = [11, 12, 13, 14, 15, 16, 17, 18]
    om = mask[:, legs]
    tp = (flags3[:, legs] & om).sum(); fp = (flags3[:, legs] & ~om).sum()
    res["gate_stats"] = dict(outlier_entries=int(om.sum()), detected=int(tp), false_flags=int(fp),
                             total_entries=int(om.size))
    for name, est in ests.items():
        err = ta_joint_errors(est, truth)[:, legs]
        res[name] = {"all": summarize(est, truth, truth_m, eval_frames), "bottom": summarize(est, truth, truth_m, bottom),
                     "outlier_joint_err_cm": float(100 * err[om].mean()) if om.any() else None,
                     "clean_joint_err_cm": float(100 * err[~om].mean())}
    return (noise_name, case_name, seed, res)


if __name__ == "__main__":
    jobs = [(n, c, sd) for n in NOISES for c in CASES for sd in (1, 2, 3)]
    with Pool(8) as pool:
        out = pool.map(job, jobs)
    json.dump(out, open("e3b_raw.json", "w"))
    METHODS = list(out[0][3].keys()); METHODS.remove("gate_stats")
    COLS = ["mpjpe_cm", "hip_cm", "knee_cm", "ankle_cm", "knee_flex_l_rmse", "valgus_l_rmse", "hip_shift_cm_rmse",
            "heel_rise_cm_l_rmse", "depth_cm_rmse", "pelvis_list_rmse", "hip_asym_cm_rmse", "trunk_flex_rmse"]
    for noise in NOISES:
        rows = [o for o in out if o[0] == noise]
        gs = [o[3]["gate_stats"] for o in rows]
        print(f"\n-- {noise}: gate(3sd) detected {sum(g['detected'] for g in gs)}/{sum(g['outlier_entries'] for g in gs)} leg outlier entries, "
              f"false flags {sum(g['false_flags'] for g in gs)}/{sum(g['total_entries'] for g in gs)}")
        print(f"{'method':22s}{'outl_err':>9s}{'clean_err':>10s} | " + "".join(f"{c.replace('_rmse','')[:10]:>11s}" for c in COLS) + "   (all frames)")
        for m in METHODS:
            oe = np.mean([o[3][m]["outlier_joint_err_cm"] for o in rows if o[3][m]["outlier_joint_err_cm"] is not None])
            ce = np.mean([o[3][m]["clean_joint_err_cm"] for o in rows])
            vals = [np.mean([o[3][m]["all"][c] for o in rows]) for c in COLS]
            print(f"{m:22s}{oe:9.2f}{ce:10.2f} | " + "".join(f"{v:11.2f}" for v in vals))
        print("fault bias at bottom (est-truth):")
        for case, keys in FAULT_METRIC.items():
            for k in keys:
                print(f"  {case:10s}{k:15s}" + "".join(f" {m[:12]:>12s} {np.mean([o[3][m]['bottom'][k + '_bias'] for o in rows if o[1] == case]):+6.2f}" for m in METHODS))
