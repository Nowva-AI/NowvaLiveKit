"""E4: calibration accuracy — standing 30-frame median vs running robust estimate; outlier bursts; sensitivity to length error."""
from __future__ import annotations
import json
from multiprocessing import Pool
import numpy as np
from synth import sequence, add_correlated_noise, apply_semantic_drift, to_skel, body, pose, recenter
from methods import PAIRS19, lengths_from, current_oracle, run_current, pbd_e
from metrics import summarize, seq_metrics
from detect import RIGID
from biomechanics.utils.bone_constraints import BoneLengthConstraints
from biomechanics.utils.types import CocoKeypoints as CK
from e3_sequences import prod_calibration

KEYS = {"femur_l": (CK.LEFT_HIP, CK.LEFT_KNEE), "tibia_l": (CK.LEFT_KNEE, CK.LEFT_ANKLE), "hip_w": (CK.LEFT_HIP, CK.RIGHT_HIP),
        "torso_l": (CK.LEFT_SHOULDER, CK.LEFT_HIP), "sho_w": (CK.LEFT_SHOULDER, CK.RIGHT_SHOULDER), "foot_l": (CK.LEFT_ANKLE, CK.LEFT_FOOT_INDEX)}
N = 600  # 2 s standing + ~6 reps: a 20 s assessment


def lens(seq, pair):
    return np.linalg.norm(seq[:, pair[1]] - seq[:, pair[0]], axis=1)


def e4a(args):
    sigma, rho, seed, drift = args
    rng = np.random.default_rng(seed)
    truth, s = sequence(N)
    measured = apply_semantic_drift(truth, s, hip_fwd_m=0.03, hip_down_m=0.02, knee_fwd_m=0.02) if drift else truth
    noisy, _ = add_correlated_noise(measured, rng, sigma, rho=rho, z_scale=1.5)
    bc, used = prod_calibration(noisy)
    L_true = lengths_from(truth[0])
    out = {}
    for k, pair in KEYS.items():
        L_obs = lens(noisy, pair)
        running_all = np.median(L_obs[60:])            # every non-standing-window frame of the assessment
        med_used = bc._calibrated_lengths[pair]
        # robust running estimate restricted to frames whose knee flexion (from the noisy skeleton) < 90 deg
        out[k] = dict(stand30=1000 * (med_used - L_true[pair]), run_all=1000 * (np.median(L_obs) - L_true[pair]))
    return (sigma, rho, drift, out)


if __name__ == "__main__":
    report = {}
    jobs = [(sg, rh, sd, dr) for sg in (0.01, 0.02, 0.03) for rh in (0.0, 0.6, 0.9, 0.97) for sd in range(40) for dr in (False,)]
    jobs += [(0.02, rh, sd, True) for rh in (0.6, 0.9) for sd in range(40)]
    with Pool(8) as pool:
        res = pool.map(e4a, jobs)
    print("E4a calibration error (mm): mean-abs / bias [standing 30-frame median (prod) | median over whole 20 s assessment]")
    for sg in (0.01, 0.02, 0.03):
        for rh in (0.0, 0.6, 0.9, 0.97):
            rows = [r[3] for r in res if r[0] == sg and r[1] == rh and not r[2]]
            line = f"sigma {100*sg:.0f}cm rho {rh:4.2f}: "
            for k in KEYS:
                a = np.array([r[k]["stand30"] for r in rows]); b = np.array([r[k]["run_all"] for r in rows])
                line += f"{k} {np.abs(a).mean():4.1f}/{a.mean():+4.1f} | {np.abs(b).mean():4.1f}/{b.mean():+4.1f};  "
                report[f"sigma{sg}_rho{rh}_{k}"] = dict(stand30_mae=round(float(np.abs(a).mean()), 1), stand30_p95=round(float(np.percentile(np.abs(a), 95)), 1),
                                                         run_mae=round(float(np.abs(b).mean()), 1), run_p95=round(float(np.percentile(np.abs(b), 95)), 1))
            print(line)
    for rh in (0.6, 0.9):
        rows = [r[3] for r in res if r[2] and r[1] == rh]
        line = f"DRIFT sigma 2cm rho {rh}: "
        for k in KEYS:
            a = np.array([r[k]["stand30"] for r in rows]); b = np.array([r[k]["run_all"] for r in rows])
            line += f"{k} {np.abs(a).mean():4.1f}/{a.mean():+4.1f} | {np.abs(b).mean():4.1f}/{b.mean():+4.1f};  "
        print(line)
    print("\np95 abs error (mm) femur_l / hip_w / torso_l, stand30 vs run_all:")
    for sg in (0.01, 0.02, 0.03):
        for rh in (0.0, 0.6, 0.9, 0.97):
            r = report
            print(f"  sigma {100*sg:.0f} rho {rh}: femur {r[f'sigma{sg}_rho{rh}_femur_l']['stand30_p95']}/{r[f'sigma{sg}_rho{rh}_femur_l']['run_p95']}  "
                  f"hip_w {r[f'sigma{sg}_rho{rh}_hip_w']['stand30_p95']}/{r[f'sigma{sg}_rho{rh}_hip_w']['run_p95']}  "
                  f"torso {r[f'sigma{sg}_rho{rh}_torso_l']['stand30_p95']}/{r[f'sigma{sg}_rho{rh}_torso_l']['run_p95']}")
    json.dump(report, open("e4a_calibration.json", "w"), indent=1)

    # E4b: outlier burst on L knee during the calibration window
    print("\nE4b: L knee +15 cm lateral burst of B frames starting at the first calibration frame (sigma 1.5 cm iid)")
    for burst in (0, 5, 10, 15, 16, 20):
        errs = []
        for seed in range(20):
            rng = np.random.default_rng(100 + seed)
            truth, s = sequence(N)
            noisy, _ = add_correlated_noise(truth, rng, 0.015)
            # find calibration start: gate latch frame (5 consecutive standing frames) -> frame 4
            start = 4
            noisy[start:start + burst, CK.LEFT_KNEE] += np.array([0.15, 0.0, 0.0])
            bc, used = prod_calibration(noisy)
            L_true = lengths_from(truth[0])
            errs.append([1000 * (bc._calibrated_lengths[KEYS[k]] - L_true[KEYS[k]]) for k in ("femur_l", "tibia_l")])
        errs = np.array(errs)
        print(f"  burst {burst:2d} frames: femur err {errs[:,0].mean():+6.1f} mm (max {np.abs(errs[:,0]).max():5.1f}), tibia err {errs[:,1].mean():+6.1f} mm; used frames start {used[0]}")

    # E4c: sensitivity of enforcement to calibration error (sigma 1 cm iid, rigid PBD and current)
    print("\nE4c: femur+tibia calibration error delta -> bottom-of-squat bias (sigma 1 cm iid, clean squat, 3 seeds)")
    truth, s = sequence(420)
    truth_m = seq_metrics(truth)
    bottom = np.nonzero(s > 0.9)[0]
    L_true = lengths_from(truth[0])
    for delta_cm in (-2.0, -1.0, 0.0, 1.0, 2.0):
        L = dict(L_true)
        for pair in ((CK.LEFT_HIP, CK.LEFT_KNEE), (CK.LEFT_KNEE, CK.LEFT_ANKLE), (CK.RIGHT_HIP, CK.RIGHT_KNEE), (CK.RIGHT_KNEE, CK.RIGHT_ANKLE)):
            L[pair] += delta_cm / 100.0
        rows = {"none": [], "current": [], "pbd_rigid": []}
        for seed in range(3):
            rng = np.random.default_rng(200 + seed)
            noisy, _ = add_correlated_noise(truth, rng, 0.01)
            bc = current_oracle(L)
            ests = {"none": noisy, "current": np.stack([run_current(bc, p) for p in noisy]),
                    "pbd_rigid": np.stack([pbd_e(p, L, 5, pairs=RIGID) for p in noisy])}
            for m, e in ests.items():
                rows[m].append(summarize(e, truth, truth_m, bottom))
        line = f"  delta {delta_cm:+.0f} cm: "
        for m, rr in rows.items():
            line += (f"{m}: knee_flex bias {np.mean([r['knee_flex_l_bias'] for r in rr]):+5.2f} rmse {np.mean([r['knee_flex_l_rmse'] for r in rr]):4.2f}, "
                     f"depth bias {np.mean([r['depth_cm_bias'] for r in rr]):+5.2f}, heel {np.mean([r['heel_rise_cm_l_bias'] for r in rr]):+5.2f}, "
                     f"dorsi {np.mean([r['dorsi_l_bias'] for r in rr]):+5.2f} | ")
        print(line)
