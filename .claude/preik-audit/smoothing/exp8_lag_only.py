"""E8: pure-lag (noise-free positions, realistic 4 px confidences/reprojection) end-to-end depth/valgus/bottom-frame errors."""
from __future__ import annotations

import json

import numpy as np

import harness as H
import smoothers as sm
from exp3_stack import CONFIGS, run  # noqa: E402

SEL = ["B_production_now(blend+vclamp,JAF,pred)", "C_designed_full(+posOE)", "D_blend+vclamp_only",
       "E_posOE_current_only", "F_JAF_only(phase-aware)", "P1_OE_tuned(1.2,16,2,paper)_only", "P4_proposed_lag2"]
K = ["knee_lag_ms", "mid_descent_err_deg", "depth_err_mean_deg", "depth_err_worst_deg", "pred_depth_err_mean_deg",
     "valgus_peak_err_mean_deg", "rep_signal_peak_err_mean_cm", "thigh_min_minus_true_cm"]
out = {}
for kind in ("smooth", "hard"):
    d = H.make_data(0, 4.0, kind=kind)
    d["noisy"] = d["truth"].copy()
    print(f"\n## {kind} (noise-free positions, 4 px confidences): " + " | ".join(K))
    for name in SEL:
        m, fs = run(CONFIGS[name], d)
        out[f"{kind}/{name}"] = {k: m[k] for k in K} | {"depth_per_rep": m["depth_err_per_rep_deg"],
                                                        "valgus_per_rep": m["valgus_peak_err_deg"],
                                                        "bottoms": fs["bottoms"]}
        print(f"  {name[:38]:38s}" + "".join(f"{m[k]:10.2f}" for k in K), "| depth/rep", m["depth_err_per_rep_deg"],
              "| valgus/rep", m["valgus_peak_err_deg"])
json.dump(out, open("exp8_results.json", "w"), indent=1)
