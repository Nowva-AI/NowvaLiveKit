"""E3: full squat sequences with triangulation-like noise; none vs current (prod calib / oracle) vs prototypes a-e."""
from __future__ import annotations
import json
import sys
from multiprocessing import Pool
import numpy as np
from synth import sequence, add_correlated_noise, apply_semantic_drift, to_skel
from metrics import summarize, seq_metrics
from methods import (PAIRS19, lengths_from, current_oracle, run_current, tree_a, two_bone_b, symmetric_c, soft_d, pbd_e,
                     LIMBS_B)
from biomechanics.utils.bone_constraints import BoneLengthConstraints
from biomechanics.utils.standing_gate import StandingPoseGate
from biomechanics.config import StandingGateConfig

N_FRAMES = 420
NOISES = {
    "iid_1cm": dict(sigma_m=0.01),
    "iid_2cm": dict(sigma_m=0.02),
    "iid_3cm": dict(sigma_m=0.03),
    "real_2cm": dict(sigma_m=0.02, z_scale=1.5, rho=0.6, outlier_p=0.01, outlier_burst=3),
    "real_2cm_drift": dict(sigma_m=0.02, z_scale=1.5, rho=0.6, outlier_p=0.01, outlier_burst=3, drift=True),
}
CASES = {
    "clean": {}, "valgus_l": dict(valgus_deg_l=25.0), "hip_shift": dict(hip_shift_m=0.06),
    "list": dict(list_m=0.03), "heel_rise": dict(heel_rise_m=0.04),
}
FAULT_METRIC = {"valgus_l": ["valgus_l", "knee_dev_cm_l"], "hip_shift": ["hip_shift_cm"], "list": ["pelvis_list", "hip_asym_cm"],
                "heel_rise": ["heel_rise_cm_l"], "clean": ["knee_flex_l", "depth_cm"]}
LEGS = LIMBS_B[:2]


def prod_calibration(noisy: np.ndarray):
    g = StandingGateConfig()
    gate = StandingPoseGate(min_confidence=g.min_confidence, max_knee_flexion_deg=g.max_knee_flexion_deg,
                            max_trunk_flexion_deg=g.max_trunk_flexion_deg, min_torso_length_m=g.min_torso_length_m,
                            max_torso_length_m=g.max_torso_length_m, min_leg_extension_ratio=g.min_leg_extension_ratio,
                            required_consecutive_frames=g.required_consecutive_frames)
    bc = BoneLengthConstraints(calibration_frames=30, tolerance=0.0, standing_gate=gate)
    used = []
    for i, p in enumerate(noisy):
        sk = to_skel(p)
        gate.check(sk)
        if gate.is_ready and not bc.is_calibrated:
            bc.enforce(sk)
            used.append(i)
        if bc.is_calibrated:
            break
    return bc, used


def job(args):
    noise_name, case_name, seed = args
    cfg = dict(NOISES[noise_name])
    drift = cfg.pop("drift", False)
    rng = np.random.default_rng(seed)
    fault = dict(CASES[case_name])
    if drift:
        fault["spine_shorten_m"] = 0.04
    truth, s = sequence(N_FRAMES, **fault)
    measured = apply_semantic_drift(truth, s, hip_fwd_m=0.03, hip_down_m=0.02, knee_fwd_m=0.02) if drift else truth
    noisy, _ = add_correlated_noise(measured, rng, **cfg)
    bc, used = prod_calibration(noisy)
    if not bc.is_calibrated:
        return (noise_name, case_name, seed, None)
    L_cal = dict(bc._calibrated_lengths)
    L_true = lengths_from(truth[0])
    obs = np.array([[np.linalg.norm(noisy[i][d] - noisy[i][p]) for p, d in PAIRS19] for i in used])
    rstd = 1.4826 * np.median(np.abs(obs - np.median(obs, axis=0)), axis=0)
    tol = {pair: max(2.5 * sd, 0.01) for pair, sd in zip(PAIRS19, rstd)}
    bc_oracle = current_oracle(L_true)

    def soft(p):
        out = p.copy()
        for pair in PAIRS19:
            out = soft_d(out, L_cal, tol[pair], pairs=[pair])
        return out

    methods = {
        "none": lambda p: p,
        "current_cal": lambda p: run_current(bc, p),
        "current_oracle": lambda p: run_current(bc_oracle, p),
        "a_tree": lambda p: tree_a(p, L_cal),
        "b_twobone_legs": lambda p: two_bone_b(p, L_cal, limbs=LEGS),
        "c_symm": lambda p: symmetric_c(p, L_cal),
        "d_soft2.5sd": soft,
        "e_pbd5": lambda p: pbd_e(p, L_cal, iters=5),
    }
    truth_m = seq_metrics(truth)
    eval_frames = np.arange(60, N_FRAMES)
    bottom = np.nonzero(s > 0.9)[0]
    res = {"calib_err_mm": {f"{p}-{d}": round(1000 * (L_cal[(p, d)] - L_true[(p, d)]), 1) for p, d in PAIRS19}}
    for name, fn in methods.items():
        est = np.stack([fn(p) for p in noisy])
        allm = summarize(est, truth, truth_m, eval_frames)
        botm = summarize(est, truth, truth_m, bottom)
        res[name] = {"all": allm, "bottom": botm}
    return (noise_name, case_name, seed, res)


if __name__ == "__main__":
    seeds = [1, 2, 3]
    jobs = [(n, c, sd) for n in NOISES for c in CASES for sd in seeds]
    with Pool(8) as pool:
        out = pool.map(job, jobs)
    json.dump([o for o in out], open("e3_raw.json", "w"))
    print("done", sum(o[3] is None for o in out), "failed calibrations")
