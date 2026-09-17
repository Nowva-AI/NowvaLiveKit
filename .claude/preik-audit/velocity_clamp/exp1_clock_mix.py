"""Dropout-hold clock mixing: held skeletons stamped time.time() vs triangulated perf_counter."""
import sys, json
sys.path.insert(0, ".")
import numpy as np
from chain_sim import *

calib, cams = make_rig()
tempo = "bw_normal"
t, fr = sequence_world(tempo, fps=30, n_reps=1, stand_s=0.6)
tri = triangulate_sequence(t + PERF_BASE, fr, calib, cams, seed=3, sigma_px=3.0)
truth_hc = hip_center(fr)
# find mid-descent frame
s = depth_profile(t, tempo, 0.6, 1)
mid = int(np.argmin(np.abs(s[: len(s)//2] - 0.5)))
report = {}
for label, start, n_drop in (("mid_descent_3f", mid - 1, 3), ("mid_descent_5f", mid - 2, 5), ("mid_descent_7f", mid - 3, 7), ("standing_3f", 5, 3)):
    sk = list(tri)
    for k in range(start, start + n_drop):
        sk[k] = None
    ref_nodrop = run(list(tri), t)
    mixed = run(sk, t, hold_clock="wall")
    consistent = run(sk, t, hold_clock="perf")
    rows = []
    lo, hi = start - 1, start + n_drop + 10
    for k in range(lo, hi):
        m, c, r = mixed[k], consistent[k], ref_nodrop[k]
        if m is None:
            rows.append(dict(k=k, dropped_no_hold=True)); continue
        knee_err_m = float(np.linalg.norm(m["pos"][CK.LEFT_KNEE] - truth_hc[k][CK.LEFT_KNEE]) * 100)
        ank_err_m = float(np.linalg.norm(m["pos"][CK.LEFT_ANKLE] - truth_hc[k][CK.LEFT_ANKLE]) * 100)
        ank_err_c = float(np.linalg.norm(c["pos"][CK.LEFT_ANKLE] - truth_hc[k][CK.LEFT_ANKLE]) * 100)
        rows.append(dict(k=k, held=m["held"], clamp_raw_dt=m["clamp_raw_dt"], clamp_dt_used=m["clamp_dt"], clamped_n=m["clamped_n"],
                         clamped_n_consistent=c["clamped_n"], oneeuro_dt=m["oneeuro_dt"], deriv_dt=m["deriv_dt"],
                         ankle_err_cm_mixed=round(ank_err_m, 1), ankle_err_cm_consistent=round(ank_err_c, 1),
                         knee_true=round(r["raw_knee"], 1), filt_knee_mixed=round(m["filt_knee"], 1), filt_knee_consistent=round(c["filt_knee"], 1), filt_knee_nodrop=round(r["filt_knee"], 1),
                         vel_mixed=round(m["knee_vel"], 1), vel_consistent=round(c["knee_vel"], 1), vel_nodrop=round(r["knee_vel"], 1),
                         acc_mixed=float(f"{m['knee_acc']:.3g}"), acc_consistent=float(f"{c['knee_acc']:.3g}"),
                         pred_mixed=round(m["pred_knee"], 1), pred_consistent=round(c["pred_knee"], 1), pred_nodrop=round(r["pred_knee"], 1)))
    report[label] = rows
    print(f"\n=== {label} (drop frames {start}..{start+n_drop-1}; 5-frame hold max)")
    print(" k  held  clamp_raw_dt      dt_used  nclamp(mix/cons)  1e_dt          deriv_dt   ankErr mix/cons  kneeFilt mix/cons/nodrop  vel mix/cons/nodrop  acc mix / cons   pred mix/cons/nodrop")
    for rw in rows:
        if rw.get("dropped_no_hold"):
            print(f"{rw['k']:3d}  NONE (hold exhausted -> early return, filters skip)"); continue
        f = lambda v: "None" if v is None else f"{v:.4g}"
        print(f"{rw['k']:3d} {str(rw['held'])[0]:>4}  {f(rw['clamp_raw_dt']):>12} {f(rw['clamp_dt_used']):>10}  {rw['clamped_n']:>3}/{rw['clamped_n_consistent']:<3}  {f(rw['oneeuro_dt']):>12} {f(rw['deriv_dt']):>12}  {rw['ankle_err_cm_mixed']:5.1f}/{rw['ankle_err_cm_consistent']:<5.1f}  {rw['filt_knee_mixed']:6.1f}/{rw['filt_knee_consistent']:6.1f}/{rw['filt_knee_nodrop']:6.1f}  {rw['vel_mixed']:7.1f}/{rw['vel_consistent']:6.1f}/{rw['vel_nodrop']:6.1f}  {rw['acc_mixed']:10.3g}/{rw['acc_consistent']:9.3g}  {rw['pred_mixed']:6.1f}/{rw['pred_consistent']:6.1f}/{rw['pred_nodrop']:6.1f}")
json.dump(report, open("exp1_clock_mix.json", "w"), indent=1, default=float)
