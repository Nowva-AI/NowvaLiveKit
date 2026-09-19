"""Evaluate the NEW pre-IK stack (wave-1 components) against the old chains on the ground-truth harness.

Inputs come from the real rewritten DLTTriangulator (world frame). Old chains (raw, current, full_original) get
hip-centred input via recentre_at_hips and keep the old tail (IK with 0.0 for missing -> JointAngleFilter), so they
are a fresh baseline on the new triangulator. `proposed_nofoot` / `proposed` use FixedLagKeypointSmoother (lag 2)
[-> FootContactModel] -> recentre_at_hips with the new tail (NaN IK, no angle filter), aligned to truth by the lag.
The old-triangulator `raw` row is read from baseline_results.json.
Writes proposed_results.json and proposed_summary.md (generated tables + proposed_findings.md appended) next to this file.
Run: /Users/naiahoard/NowvaLiveKit/venv/bin/python run_proposed.py [--workers N] [--seeds 3] [--no-harsh]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from preik_harness import SCENARIOS, evaluate_chains, production_chain, proposed_chain

HERE = Path(__file__).parent
CALIBRATIONS = ["perfect", "tpose"]
OLD_TRI_LABEL = "raw (old triangulator)"
OVERALL_COLUMNS = [
    ("mpjpe_mm", "MPJPE mm", 1), ("mpjpe_lower_mm", "lower mm", 1), ("p95_err_mm", "p95 mm", 1),
    ("jitter_still_p50_mm", "still jitter p50 mm", 1), ("lag_knee_flex_ik_ms", "knee lag IK ms", 0),
    ("knee_flex_tb_err_deg", "knee@true-bottom err deg", 1), ("knee_flex_ik_moving_mae_deg", "knee MAE moving deg", 1),
    ("depth_err_ik_deg", "depth err IK deg", 1), ("depth_err_final_deg", "depth err final deg", 1),
    ("bottom_sel_delay_ms", "bottom-frame delay ms (aligned)", 0),
    ("bottom_sel_delay_wall_ms", "bottom-frame delay ms (wall)", 0), ("analysis_latency_ms", "analysis latency ms", 0),
    ("ik_knee_invalid_frac", "knee NaN/0 frac", 3), ("stance_width_mae_mm", "stance MAE mm", 1),
    ("chain_us_mean", "chain us", 0), ("frame_us_mean", "frame us", 0),
]
DELTA_METRICS = ["mpjpe_mm", "mpjpe_lower_mm", "p95_err_mm", "jitter_still_p50_mm", "lag_knee_flex_ik_ms",
                 "knee_flex_tb_err_deg", "knee_flex_ik_moving_mae_deg", "depth_err_ik_deg", "depth_err_final_deg",
                 "valgus_tb_err_deg", "stance_width_mae_mm", "ik_knee_invalid_frac"]
PER_SCENARIO_METRICS = [
    ("mpjpe_mm", "MPJPE mm", 1), ("knee_flex_tb_err_deg", "knee@true-bottom err deg", 1),
    ("depth_err_final_deg", "depth err final deg", 1), ("ik_knee_invalid_frac", "knee NaN/0 frac", 3),
]
FAULT_ROWS = [
    ("valgus", "valgus_amp_ratio_tb", "valgus amplitude ratio @true bottom (same estimator on truth; 1 = preserved)", 2),
    ("valgus", "valgus_amp_ratio_sel", "valgus amplitude ratio @pipeline bottom frame", 2),
    ("valgus", "valgus_tb_err_deg", "valgus @true bottom err deg", 2),
    ("valgus", "valgus_sel_err_deg", "valgus @pipeline bottom frame err deg", 2),
    ("heel_rise", "heel_rise_amp_ratio_tb", "heel rise (ankle over toe, positions) amplitude ratio @true bottom", 2),
    ("heel_rise", "heel_rise_amp_ratio_sel", "heel rise amplitude ratio @pipeline bottom frame", 2),
    ("heel_rise", "foot_state_heel_rise_amp_ratio_tb", "FootState heel_rise_cm / true ankle rise @true bottom", 2),
    ("heel_rise", "heel_rise_peak_ratio_tb", "heel rise PEAK per rep (positions) / true peak ankle rise", 2),
    ("heel_rise", "foot_state_heel_rise_peak_ratio_tb", "FootState heel rise PEAK per rep / true peak ankle rise", 2),
    ("heel_rise", "foot_state_heel_rise_peak_cm_tb", "FootState heel rise PEAK per rep cm", 2),
    ("heel_rise", "true_ankle_rise_peak_cm_tb", "true peak ankle rise per rep cm (harness GT)", 2),
    ("hip_shift", "hip_shift_amp_ratio_tb", "lateral hip shift amplitude ratio @true bottom", 2),
    ("hip_shift", "hip_shift_amp_ratio_sel", "lateral hip shift amplitude ratio @pipeline bottom frame", 2),
    ("hip_shift", "pelvis_list_amp_ratio_tb", "pelvic list amplitude ratio @true bottom", 2),
    ("hip_shift", "pelvis_list_amp_ratio_sel", "pelvic list amplitude ratio @pipeline bottom frame", 2),
    ("asymmetric_depth", "knee_asym_tb_err_deg", "L-R knee asymmetry err @true bottom deg (truth ~ -8)", 2),
    ("asymmetric_depth", "knee_asym_sel_err_deg", "L-R knee asymmetry err @pipeline bottom frame deg", 2),
    ("clean", "heel_rise_amp_err_tb_mm", "clean: false heel-rise amplitude mm (positions) @true bottom", 1),
    ("clean", "foot_state_false_heel_rise_tb_cm", "clean: FootState false heel rise cm @true bottom", 2),
    ("clean", "foot_state_heel_rise_peak_cm_tb", "clean: FootState false heel rise PEAK per rep cm", 2),
    ("clean", "true_ankle_rise_peak_cm_tb", "clean: true peak ankle rise per rep cm (should be ~0)", 2),
    ("clean", "hip_shift_amp_err_tb_mm", "clean: false hip-shift amplitude mm @true bottom", 1),
    ("clean", "depth_err_final_deg", "clean: depth err final deg", 2),
    ("stance_change", "stance_change_ratio", "stance widening ratio (1 = preserved)", 2),
    ("stance_change", "stance_width_mae_mm", "stance width MAE mm", 1),
    ("walkout", "mpjpe_mm", "walkout MPJPE mm", 1),
    ("walkout", "processed_frames", "walkout processed frames (readiness opened?)", 0),
    ("walkout", "depth_err_final_deg", "walkout depth err final deg", 2),
    ("walkout", "foot_state_valid_frac", "walkout FootState valid frac", 2),
    ("fast_reps", "knee_flex_tb_err_deg", "fast reps knee @true bottom err deg", 2),
    ("fast_reps", "depth_err_ik_deg", "fast reps depth err IK deg", 2),
    ("fast_reps", "depth_err_final_deg", "fast reps depth err final deg", 2),
    ("fast_reps", "lag_knee_flex_ik_ms", "fast reps knee lag IK ms", 1),
]
HARSH_COLUMNS = [
    ("mpjpe_mm", "MPJPE mm", 1), ("p95_err_mm", "p95 mm", 1), ("jitter_still_p50_mm", "still jitter p50 mm", 1),
    ("knee_flex_tb_err_deg", "knee@true-bottom err deg", 1), ("knee_flex_ik_moving_mae_deg", "knee MAE moving deg", 1),
    ("depth_err_ik_deg", "depth err IK deg", 1), ("depth_err_final_deg", "depth err final deg", 1),
    ("valgus_tb_err_deg", "valgus@true-bottom err deg", 1), ("ik_knee_invalid_frac", "knee NaN/0 frac", 3),
    ("predicted_frac", "predicted frac", 3),
]
STATS_KEYS = ["loop_hz", "sync_fail_frac", "triangulation_none_frac", "mean_reprojection_px",
              "tri_lower_conf_p50", "tri_lower_conf_below_0.1_frac", "tri_lower_conf_zero_frac", "tri_swap_count"]


def chain_factories() -> dict:
    return {
        "raw": production_chain([]),
        "current": production_chain(["blend", "vclamp"]),
        "full_original": production_chain(["blend", "vclamp", "bone", "ground", "smooth", "bone"]),
        "proposed_nofoot": proposed_chain(foot_contact=False),
        "proposed": proposed_chain(foot_contact=True),
    }


def _json_safe(obj: object) -> object:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def _fmt(value: float | None, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    return f"{value:.{digits}f}"


def _overall(summary: dict, key: str) -> float | None:
    return summary["overall_mean_of_scenarios"].get(key)


def _scenario_stat(summary: dict, scenario: str, key: str) -> dict:
    return summary["per_scenario"].get(scenario, {}).get(key, {"mean": None, "std": None, "n": 0})


def _paired_deltas(results: dict, calibration: str, chain: str, base: str) -> dict[str, tuple[float, float]]:
    base_runs = {(r["scenario"], r["seed"]): r["metrics"] for r in results[base]["runs"]
                 if r["calibration"] == calibration}
    deltas: dict[str, list[float]] = {k: [] for k in DELTA_METRICS}
    for run in results[chain]["runs"]:
        if run["calibration"] != calibration:
            continue
        ref = base_runs[(run["scenario"], run["seed"])]
        for key in DELTA_METRICS:
            a, b = run["metrics"].get(key), ref.get(key)
            if a is not None and b is not None:
                deltas[key].append(a - b)
    return {k: (float(np.mean(v)), float(np.std(v))) if v else (float("nan"), float("nan"))
            for k, v in deltas.items()}


def _summaries(results: dict, old_tri: dict, cal: str) -> list[tuple[str, dict]]:
    rows = [(OLD_TRI_LABEL, old_tri["results"]["raw"]["summary"][cal])]
    rows += [(chain, results[chain]["summary"][cal]) for chain in results]
    return rows


def _summary_markdown(results: dict, harsh: dict | None, old_tri: dict, meta: dict) -> str:
    lines = ["# New pre-IK stack vs old chains — triangulated 3-camera harness", "",
             f"Generated {meta['generated']}; wall {meta['wall_s']:.0f} s; seeds {meta['seeds']}; noise `realistic`; "
             f"scenarios: {', '.join(SCENARIOS)}; workers {meta['workers']}.", "",
             "- Inputs: real rewritten `DLTTriangulator` (world frame, metric confidence, 21 keypoints; heels never "
             "detected by the simulated detector -> conf 0). Delivery model unchanged (old `get_synced_frames` "
             "semantics: sync failures are skipped ticks). Detector noise model unchanged (WS2 crop fix NOT simulated).",
             "- Old chains (`raw`, `current`, `full_original`): `recentre_at_hips` -> chain -> IK (NaN->0.0 as the old "
             "IK did) -> `JointAngleFilter` (legacy copy), production dropout hold. Fresh baseline on the new triangulator.",
             f"- `{OLD_TRI_LABEL}`: numbers copied from `baseline_results.json` (old triangulator, same harness).",
             "- `proposed_nofoot`: `FixedLagKeypointSmoother` (lag 2) -> `recentre_at_hips`; `proposed`: + "
             "`FootContactModel.update` on the lagged world stream before recentring. New tail: `AnalyticalIKSolver` "
             "(NaN for missing), NO angle filter (final == IK), `predict_missing` on frames without a skeleton. "
             "Outputs are aligned to the truth of the tick they belong to (2-frame lag removed) — `analysis latency` "
             "and `bottom-frame delay (wall)` add it back.",
             "- Valgus: truth is the SAME `TriangulatedValgusEstimator` on the noiseless GT skeleton, so the WS7 metric "
             "change cancels in ratios and errors.", ""]
    for cal in CALIBRATIONS:
        rows = _summaries(results, old_tri, cal)
        lines += [f"## Overall (mean over 8 scenarios x 3 seeds) — calibration `{cal}`", "",
                  "| chain | " + " | ".join(c[1] for c in OVERALL_COLUMNS) + " |",
                  "|---|" + "---:|" * len(OVERALL_COLUMNS)]
        for label, summary in rows:
            lines.append(f"| {label} | " + " | ".join(_fmt(_overall(summary, key), digits)
                                                      for key, _, digits in OVERALL_COLUMNS) + " |")
        lines.append("")
        lines += [f"### Paired delta vs `raw` (new triangulator), mean ± std over 8 scenarios x seeds — `{cal}`", "",
                  "| chain | " + " | ".join(DELTA_METRICS) + " |", "|---|" + "---:|" * len(DELTA_METRICS)]
        for chain in results:
            if chain == "raw":
                continue
            deltas = _paired_deltas(results, cal, chain, "raw")
            lines.append(f"| {chain} | " + " | ".join(
                f"{_fmt(deltas[k][0], 3 if 'frac' in k else 1)} ± {_fmt(deltas[k][1], 3 if 'frac' in k else 1)}"
                for k in DELTA_METRICS) + " |")
        lines.append("")
        lines += [f"### Fault-signal preservation and scenario-specific metrics — `{cal}` (mean ± std over seeds)", "",
                  "| scenario / metric | " + " | ".join(label for label, _ in rows) + " |",
                  "|---|" + "---:|" * len(rows)]
        for scenario, key, label, digits in FAULT_ROWS:
            cells = []
            for _, summary in rows:
                stat = _scenario_stat(summary, scenario, key)
                cells.append(f"{_fmt(stat['mean'], digits)} ± {_fmt(stat['std'], digits)}"
                             if stat["mean"] is not None else "-")
            lines.append(f"| {scenario}: {label} | " + " | ".join(cells) + " |")
        lines.append("")
        for key, label, digits in PER_SCENARIO_METRICS:
            lines += [f"### Per scenario: {label} — `{cal}` (mean ± std over seeds)", "",
                      "| scenario | " + " | ".join(row_label for row_label, _ in rows) + " |",
                      "|---|" + "---:|" * len(rows)]
            for scenario in SCENARIOS:
                cells = []
                for _, summary in rows:
                    stat = _scenario_stat(summary, scenario, key)
                    cells.append(f"{_fmt(stat['mean'], digits)} ± {_fmt(stat['std'], digits)}"
                                 if stat["mean"] is not None else "-")
                lines.append(f"| {scenario} | " + " | ".join(cells) + " |")
            lines.append("")
        lines += [f"### Gates / foot model / compute (mean over scenarios) — `{cal}`", "",
                  "| chain | readiness gate s | FootState valid s | FootState valid frac | predicted frac | "
                  "chain us | kalman us | foot us | recentre+Skeleton3D us | frame us (chain+IK+valgus[+filter]) |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for label, summary in rows:
            lines.append(f"| {label} | {_fmt(_overall(summary, 'readiness_gate_s'), 2)} | "
                         f"{_fmt(_overall(summary, 'chain_calibrated_s'), 2)} | "
                         f"{_fmt(_overall(summary, 'foot_state_valid_frac'), 2)} | "
                         f"{_fmt(_overall(summary, 'predicted_frac'), 3)} | {_fmt(_overall(summary, 'chain_us_mean'), 0)} | "
                         f"{_fmt(_overall(summary, 'stage_kalman_us_mean'), 0)} | "
                         f"{_fmt(_overall(summary, 'stage_foot_us_mean'), 0)} | "
                         f"{_fmt(_overall(summary, 'stage_recentre_us_mean'), 0)} | "
                         f"{_fmt(_overall(summary, 'frame_us_mean'), 0)} |")
        lines.append("")

    lines += ["## Input stats on the new triangulator (per scenario, mean over seeds)", "",
              "| calibration | scenario | " + " | ".join(STATS_KEYS) + " |", "|---|---|" + "---:|" * len(STATS_KEYS)]
    for cal in CALIBRATIONS:
        runs = [r for r in results["raw"]["runs"] if r["calibration"] == cal]
        for scenario in SCENARIOS:
            stat_rows = [r["delivery"] for r in runs if r["scenario"] == scenario]
            cells = []
            for key in STATS_KEYS:
                values = [row[key] for row in stat_rows if row.get(key) is not None]
                cells.append(_fmt(float(np.mean(values)), 4 if "frac" in key or "p50" in key else 2) if values else "-")
            lines.append(f"| {cal} | {scenario} | " + " | ".join(cells) + " |")
    lines.append("")

    if harsh:
        lines += ["## Noise profile `harsh` (calibration `perfect`, mean over 8 scenarios x seeds)", "",
                  "| chain | " + " | ".join(c[1] for c in HARSH_COLUMNS) + " |", "|---|" + "---:|" * len(HARSH_COLUMNS)]
        harsh_rows = [(OLD_TRI_LABEL, old_tri["noise_sensitivity"]["harsh"]["raw"])]
        harsh_rows += [(chain, res["summary"]) for chain, res in harsh.items()]
        for label, summary in harsh_rows:
            lines.append(f"| {label} | " + " | ".join(_fmt(_overall(summary, key), digits)
                                                      for key, _, digits in HARSH_COLUMNS) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--no-harsh", action="store_true")
    args = parser.parse_args()

    t_start = time.perf_counter()
    factories = chain_factories()
    results = evaluate_chains(factories, scenarios=SCENARIOS, calibration=CALIBRATIONS, n_seeds=args.seeds,
                              noise="realistic", workers=args.workers)
    t_main = time.perf_counter() - t_start
    harsh = None
    if not args.no_harsh:
        harsh = evaluate_chains({k: factories[k] for k in ("raw", "proposed")}, scenarios=SCENARIOS,
                                calibration="perfect", n_seeds=args.seeds, noise="harsh", workers=args.workers)
    wall = time.perf_counter() - t_start
    old_tri = json.loads((HERE / "baseline_results.json").read_text())
    meta = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "wall_s": wall, "main_wall_s": t_main,
            "seeds": list(range(args.seeds)), "workers": args.workers,
            "n_runs": sum(len(r["runs"]) for r in results.values()),
            "old_triangulator_baseline_generated": old_tri["meta"]["generated"]}
    payload = {
        "meta": meta,
        "results": {chain: {"summary": res["summary"], "runs": res["runs"]} for chain, res in results.items()},
        "old_triangulator_raw": {"summary": old_tri["results"]["raw"]["summary"],
                                 "harsh_summary": old_tri["noise_sensitivity"]["harsh"]["raw"]},
        "harsh": {chain: {"summary": res["summary"], "runs": res["runs"]} for chain, res in harsh.items()}
        if harsh else {},
    }
    (HERE / "proposed_results.json").write_text(json.dumps(_json_safe(payload), indent=1, allow_nan=False))
    findings = HERE / "proposed_findings.md"
    (HERE / "proposed_summary.md").write_text(_summary_markdown(results, harsh, old_tri, meta)
                                              + (findings.read_text() if findings.exists() else ""))
    print(f"done in {wall:.1f} s (main {t_main:.1f} s, workers {args.workers})")


if __name__ == "__main__":
    main()
