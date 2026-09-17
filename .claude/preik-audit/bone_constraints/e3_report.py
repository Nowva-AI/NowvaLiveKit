"""Aggregate e3_raw.json into tables."""
from __future__ import annotations
import json
import numpy as np
raw = json.load(open("e3_raw.json"))
from e3_sequences import NOISES, CASES, FAULT_METRIC
METHODS = ["none", "current_cal", "current_oracle", "a_tree", "b_twobone_legs", "c_symm", "d_soft2.5sd", "e_pbd5"]
COLS = ["mpjpe_cm", "hip_cm", "knee_cm", "ankle_cm", "toe_cm", "knee_flex_l_rmse", "hip_flex_l_rmse", "valgus_l_rmse", "knee_dev_cm_l_rmse",
        "hip_shift_cm_rmse", "heel_rise_cm_l_rmse", "depth_cm_rmse", "pelvis_list_rmse", "hip_asym_cm_rmse", "trunk_flex_rmse", "dorsi_l_rmse"]
agg = {}
for noise, case, seed, res in raw:
    if res is None:
        continue
    for m in METHODS:
        for scope in ("all", "bottom"):
            agg.setdefault((noise, m, scope), []).append(res[m][scope])
summary = {}
for scope in ("all", "bottom"):
    print(f"\n######## scope={scope} (mean over 5 fault cases x 3 seeds; frames >=60 for 'all', s>0.9 for 'bottom')")
    for noise in NOISES:
        print(f"\n-- {noise}")
        print(f"{'method':16s}" + "".join(f"{c.replace('_rmse','').replace('_cm','')[:9]:>10s}" for c in COLS))
        for m in METHODS:
            rows = agg[(noise, m, scope)]
            vals = [np.mean([r[c] for r in rows]) for c in COLS]
            summary[f"{scope}|{noise}|{m}"] = dict(zip(COLS, [round(v, 3) for v in vals]))
            print(f"{m:16s}" + "".join(f"{v:10.2f}" for v in vals))
# fault preservation: bias of fault metric at bottom
print("\n######## fault-metric bias at bottom (est - truth), mean over seeds")
for noise in NOISES:
    print(f"\n-- {noise}")
    for case, keys in FAULT_METRIC.items():
        for k in keys:
            line = f"{case:10s} {k:16s}"
            for m in METHODS:
                vals = [res[m]["bottom"][f"{k}_bias"] for n, c, sd, res in raw if n == noise and c == case]
                rm = [res[m]["bottom"][f"{k}_rmse"] for n, c, sd, res in raw if n == noise and c == case]
                line += f" {m[:8]:>8s} {np.mean(vals):+6.2f}/{np.mean(rm):5.2f}"
            print(line)
# calibration errors
print("\n######## production 30-frame median calibration error (mm): mean |err| and max |err| per bone")
for noise in NOISES:
    errs = {}
    for n, c, sd, res in raw:
        if n != noise: continue
        for k, v in res["calib_err_mm"].items():
            errs.setdefault(k, []).append(v)
    print(noise, " ".join(f"{k}:{np.mean(np.abs(v)):.0f}/{np.max(np.abs(v)):.0f}" for k, v in errs.items()))
json.dump(summary, open("e3_summary.json", "w"), indent=1)
