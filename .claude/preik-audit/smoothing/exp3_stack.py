"""E3: end-to-end smoothing stack (pre-IK -> IK -> JointAngleFilter -> DerivativeTracker -> Predictive -> rules).

Every configuration runs through the same pipeline replay (harness.run_stack) on 5 noise seeds.
Reports live-angle accuracy, predicted-angle (eval) accuracy, fault outcomes and bottom-frame selection.
Usage: exp3_stack.py [sigma_px] [phase_source]
"""
from __future__ import annotations

import json
import sys

import numpy as np

import harness as H
import smoothers as sm

SIGMA = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
PHASE = sys.argv[2] if len(sys.argv) > 2 else "oracle"
SEEDS = [0, 1, 2, 3, 4]

# Proposal parameters (from exp4 grid / exp6, see REPORT.md)
from fixedlag_fast import FixedLagCV, ProposedSmoother  # noqa: E402

OE_TUNED = dict(min_cutoff=1.2, beta=16.0, d_cutoff=2.0)

CONFIGS = {
    "A_raw_no_filters": dict(jaf=False, predictive=False),
    "B_production_now(blend+vclamp,JAF,pred)": dict(blend=True, vclamp=True, jaf=True, predictive=True),
    "C_designed_full(+posOE)": dict(blend=True, vclamp=True, pos=lambda: sm.OneEuroVec(0.8, 4.0, 1.0, "raw"),
                                    jaf=True, predictive=True),
    "D_blend+vclamp_only": dict(blend=True, vclamp=True, jaf=False, predictive=False),
    "E_posOE_current_only": dict(pos=lambda: sm.OneEuroVec(0.8, 4.0, 1.0, "raw"), jaf=False, predictive=False),
    "F_JAF_only(phase-aware)": dict(jaf=True, predictive=False, phase_aware=True),
    "G_predictive_only": dict(jaf=False, predictive=True, phase_aware=False),
    "H_prod_without_predictive": dict(blend=True, vclamp=True, jaf=True, predictive=False, phase_aware=True),
    "P1_OE_tuned(1.2,16,2,paper)_only": dict(pos=lambda: sm.OneEuroVec(**OE_TUNED, deriv_from="filtered"), jaf=False,
                                             predictive=False),
    "P2_KF_CV_causal_lag0(q10,r.02)": dict(pos=lambda: FixedLagCV(10.0, 0.02, 0), jaf=False, predictive=False),
    "P3_proposed_lag1": dict(pos=lambda: ProposedSmoother(lag=1), jaf=False, predictive=False, lag=1),
    "P4_proposed_lag2": dict(pos=lambda: ProposedSmoother(lag=2), jaf=False, predictive=False, lag=2),
    "P5_proposed_lag2+predictive": dict(pos=lambda: ProposedSmoother(lag=2), jaf=False, predictive=True,
                                        phase_aware=False, lag=2),
    "P6_proposed_lag2+JAF": dict(pos=lambda: ProposedSmoother(lag=2), jaf=True, predictive=False, phase_aware=True,
                                 lag=2),
    "P7_blend+proposed_lag2": dict(blend=True, pos=lambda: ProposedSmoother(lag=2), jaf=False, predictive=False,
                                   lag=2),
}


def _shift(arr, lag):
    if not lag:
        return arr
    return np.concatenate([arr[lag:], np.repeat(arr[-1:], lag, axis=0)])


def run(cfg, d):
    kw = dict(cfg)
    pos = kw.pop("pos", None)
    lag = kw.pop("lag", 0)
    r = H.run_stack(d, pos_filter=pos() if pos else None, phase_source=PHASE, **kw)
    live = {k: _shift(v, lag) for k, v in r["angles"].items()}
    ev = {k: _shift(v, lag) for k, v in r["eval"].items()}
    m = H.angle_metrics(live, d)
    m.update(H.angle_metrics(ev, d, "pred_"))
    m.update(H.position_metrics(_shift(r["skel"], lag), d))
    fs = H.fault_summary(r, d)
    return m, fs


def fault_counts(fs_list, kind):
    valgus_true = {1, 3, 4} if kind == "smooth" else {1, 2, 3}
    depth_true = {5: "mild"} if kind == "smooth" else {}
    n = len(fs_list)
    out = dict(valgus_tp_rate=0.0, valgus_fp_reps=0.0, symmetry_events=0.0, depth_fp_or_wrong=0.0,
               depth_tp_rate=0.0, reps_counted=0.0, bottom_abs_frame_offset=0.0, bottom_true_knee_err=0.0,
               bottom_stored_knee_err=0.0)
    for fs in fs_list:
        f = fs["faults"]
        v = {int(r) for r in f.get("knee_valgus", {}) if int(r) >= 0}
        out["valgus_tp_rate"] += len(v & valgus_true) / len(valgus_true) / n
        out["valgus_fp_reps"] += len(v - valgus_true) / n
        out["symmetry_events"] += sum(len(x) for x in f.get("bilateral_asymmetry", {}).values()) / n
        dep = f.get("depth", {})
        for r, sevs in dep.items():
            r = int(r)
            if r in depth_true and depth_true[r] in sevs:
                out["depth_tp_rate"] += 1 / max(len(depth_true), 1) / n
            if r not in depth_true or any(s != depth_true.get(r) for s in sevs):
                out["depth_fp_or_wrong"] += 1 / n
        out["reps_counted"] += fs["n_reps_counted"] / n
        b = fs["bottoms"]
        if b:
            out["bottom_abs_frame_offset"] += np.mean([abs(x["frame_offset_from_true_bottom"]) for x in b]) / n
            out["bottom_true_knee_err"] += np.mean([x["true_knee_at_frame_minus_true_max"] for x in b]) / n
            out["bottom_stored_knee_err"] += np.mean([x["stored_skel_knee_minus_true_max"] for x in b]) / n
    return {k: round(float(v), 2) for k, v in out.items()}


if __name__ == "__main__":
    results = {}
    for kind in ("smooth", "hard"):
        results[kind] = {}
        for name, cfg in CONFIGS.items():
            ms, fss = [], []
            for sd in SEEDS:
                d = H.make_data(sd, SIGMA, kind=kind)
                m, fs = run(cfg, d)
                ms.append(m); fss.append(fs)
            summ = H.summarize(ms)
            summ["depth_err_per_rep_deg"] = np.round(np.mean([m["depth_err_per_rep_deg"] for m in ms], 0), 1).tolist()
            summ["pred_depth_err_per_rep_deg"] = np.round(np.mean([m["pred_depth_err_per_rep_deg"] for m in ms], 0), 1).tolist()
            summ["valgus_peak_err_deg"] = np.round(np.mean([m["valgus_peak_err_deg"] for m in ms], 0), 1).tolist()
            summ.update(fault_counts(fss, kind))
            results[kind][name] = summ
            print(kind, name, "done", flush=True)

    with open(f"exp3_results_sigma{SIGMA:g}_{PHASE}.json", "w") as fh:
        json.dump(results, fh, indent=1)

    KEYS = ["knee_rms_moving_deg", "knee_lag_ms", "mid_descent_err_deg", "depth_err_mean_deg", "depth_err_worst_deg",
            "valgus_peak_err_mean_deg", "knee_std_standing_deg", "pred_knee_rms_moving_deg", "pred_depth_err_mean_deg",
            "pos_rms_moving_cm", "standing_jitter_mm_per_frame", "rep_signal_peak_err_mean_cm", "thigh_min_minus_true_cm",
            "valgus_tp_rate", "valgus_fp_reps", "symmetry_events", "depth_tp_rate", "depth_fp_or_wrong", "reps_counted",
            "bottom_abs_frame_offset", "bottom_true_knee_err", "bottom_stored_knee_err"]
    for kind, block in results.items():
        print(f"\n### {kind} session, sigma {SIGMA:g}px, phase={PHASE}")
        print("config".ljust(42) + "".join(k.replace("_deg", "").replace("_mean", "")[-11:].rjust(12) for k in KEYS))
        for name, m in block.items():
            print(name[:41].ljust(42) + "".join(f"{m.get(k, float('nan')):12.2f}" for k in KEYS))
        for name, m in block.items():
            print("  ", name, "| depth/rep", m["depth_err_per_rep_deg"], "| pred depth/rep", m["pred_depth_err_per_rep_deg"],
                  "| valgus/rep", m["valgus_peak_err_deg"])
