"""Baseline evaluation of the production pre-IK chain configurations on triangulated 3-camera input.

Writes baseline_results.json and baseline_summary.md next to this file.
Run: /Users/naiahoard/NowvaLiveKit/venv/bin/python run_baseline.py [--workers N] [--seeds 3] [--quick]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from preik_harness import SCENARIOS, DeliveryConfig, DiagnosticConfidenceFloor, baseline_factories, evaluate_chains

HERE = Path(__file__).parent
CALIBRATIONS = ["perfect", "tpose"]
OVERALL_COLUMNS = [
    ("mpjpe_mm", "MPJPE mm"), ("mpjpe_lower_mm", "lower mm"), ("p95_err_mm", "p95 mm"),
    ("bone_len_std_mm", "bone std mm"), ("jitter_still_p50_mm", "still jitter p50 mm"),
    ("lag_knee_y_ms", "lag knee-y ms"), ("lag_knee_flex_ik_ms", "lag knee IK ms"),
    ("lag_knee_flex_final_ms", "lag knee final ms"), ("knee_flex_tb_err_deg", "knee@true-bottom err IK deg"),
    ("depth_err_ik_deg", "depth err IK deg"), ("depth_err_final_deg", "depth err final deg"),
    ("bottom_sel_delay_ms", "bottom-frame delay ms"), ("knee_flex_ik_moving_mae_deg", "knee MAE moving IK deg"),
    ("valgus_tb_err_deg", "valgus@true-bottom err IK deg"), ("stance_width_mae_mm", "stance MAE mm"),
    ("ik_knee_zero_frac", "IK knee=0 frac"), ("chain_us_mean", "chain us"),
]
DELTA_METRICS = ["mpjpe_mm", "mpjpe_lower_mm", "p95_err_mm", "bone_len_std_mm", "jitter_still_p50_mm",
                 "lag_knee_flex_ik_ms", "knee_flex_tb_err_deg", "depth_err_ik_deg", "depth_err_final_deg",
                 "knee_flex_ik_moving_mae_deg", "valgus_tb_err_deg", "stance_width_mae_mm"]
FAULT_ROWS = [
    ("valgus", "valgus_amp_ratio_tb", "valgus amplitude ratio @true bottom, IK (1 = preserved)"),
    ("valgus", "valgus_amp_ratio_sel", "valgus amplitude ratio @pipeline bottom frame, final"),
    ("valgus", "valgus_tb_err_deg", "valgus @true bottom err deg (IK)"),
    ("heel_rise", "heel_rise_amp_ratio_tb", "heel rise (ankle over toe) amplitude ratio @true bottom"),
    ("heel_rise", "heel_rise_amp_ratio_sel", "heel rise amplitude ratio @pipeline bottom frame"),
    ("hip_shift", "hip_shift_amp_ratio_tb", "lateral hip shift amplitude ratio @true bottom"),
    ("hip_shift", "hip_shift_amp_ratio_sel", "lateral hip shift amplitude ratio @pipeline bottom frame"),
    ("hip_shift", "pelvis_list_amp_ratio_tb", "pelvic list amplitude ratio @true bottom, IK"),
    ("hip_shift", "pelvis_list_amp_ratio_sel", "pelvic list amplitude ratio @pipeline bottom frame, final"),
    ("asymmetric_depth", "knee_asym_tb_err_deg", "L-R knee asymmetry err @true bottom, IK deg (truth ~ -8)"),
    ("asymmetric_depth", "knee_asym_sel_err_deg", "L-R knee asymmetry err @pipeline bottom frame deg"),
    ("clean", "heel_rise_amp_err_tb_mm", "clean: false heel-rise amplitude mm @true bottom"),
    ("clean", "hip_shift_amp_err_tb_mm", "clean: false hip-shift amplitude mm @true bottom"),
    ("clean", "depth_err_final_deg", "clean depth err final deg"),
    ("stance_change", "stance_change_ratio", "stance widening ratio (1 = preserved)"),
    ("stance_change", "stance_width_mae_mm", "stance width MAE mm"),
    ("walkout", "mpjpe_mm", "walkout MPJPE mm"),
    ("walkout", "processed_frames", "walkout processed frames (readiness opened?)"),
    ("walkout", "depth_err_final_deg", "walkout depth err final deg"),
    ("fast_reps", "knee_flex_tb_err_deg", "fast reps knee @true bottom err IK deg"),
    ("fast_reps", "depth_err_ik_deg", "fast reps depth err IK deg"),
    ("fast_reps", "depth_err_final_deg", "fast reps depth err final deg"),
    ("fast_reps", "lag_knee_flex_ik_ms", "fast reps knee lag IK ms"),
]


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


def _paired_deltas(results: dict, calibration: str, chain: str, base: str = "raw") -> dict[str, tuple[float, float]]:
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


def _summary_markdown(results: dict, floor: dict, sensitivity: dict, meta: dict) -> str:
    lines = ["# Pre-IK baseline — triangulated 3-camera harness", "",
             f"Generated {meta['generated']}; {meta['n_runs']} chain runs; wall {meta['wall_s']:.0f} s; "
             f"seeds {meta['seeds']}; noise profile `realistic`; scenarios: {', '.join(SCENARIOS)}.",
             "All chains see identical inputs per (scenario, seed, calibration): comparisons are paired.",
             "'IK' = AnalyticalIKSolver on the chain output; 'final' = after JointAngleFilter (what rules/bottom "
             "buffer consume). Depth err = measured max knee flexion per rep - true max (negative = undershoot).", ""]
    chains = list(results)
    for cal in CALIBRATIONS:
        lines += [f"## Overall (mean over 8 scenarios) — calibration `{cal}`", "",
                  "| chain | " + " | ".join(c[1] for c in OVERALL_COLUMNS) + " |",
                  "|---|" + "---:|" * len(OVERALL_COLUMNS)]
        for chain in chains:
            overall = results[chain]["summary"][cal]["overall_mean_of_scenarios"]
            lines.append(f"| {chain} | " + " | ".join(_fmt(overall.get(c[0]), 2 if "frac" in c[0] else 1)
                                                      for c in OVERALL_COLUMNS) + " |")
        lines.append("")
        lines += [f"### Paired delta vs raw (mean ± std over 8 scenarios x seeds) — `{cal}`", "",
                  "| chain | " + " | ".join(DELTA_METRICS) + " |", "|---|" + "---:|" * len(DELTA_METRICS)]
        for chain in chains:
            if chain == "raw":
                continue
            deltas = _paired_deltas(results, cal, chain)
            lines.append(f"| {chain} | " + " | ".join(f"{_fmt(deltas[k][0])} ± {_fmt(deltas[k][1])}"
                                                      for k in DELTA_METRICS) + " |")
        lines.append("")
        lines += [f"### Fault-signal preservation and scenario-specific metrics — `{cal}` (mean ± std over seeds)",
                  "", "| scenario / metric | " + " | ".join(chains) + " |", "|---|" + "---:|" * len(chains)]
        for scenario, key, label in FAULT_ROWS:
            cells = []
            for chain in chains:
                stat = results[chain]["summary"][cal]["per_scenario"][scenario][key]
                cells.append(f"{_fmt(stat['mean'], 2)} ± {_fmt(stat['std'], 2)}" if stat["mean"] is not None else "-")
            lines.append(f"| {scenario}: {label} | " + " | ".join(cells) + " |")
        lines.append("")
        lines += [f"### Gates / calibration timing (s from loop start, mean over scenarios) — `{cal}`", "",
                  "| chain | standing gate | readiness gate | chain calibrated | frame us (chain+IK+valgus+angle filter) |",
                  "|---|---:|---:|---:|---:|"]
        for chain in chains:
            o = results[chain]["summary"][cal]["overall_mean_of_scenarios"]
            lines.append(f"| {chain} | {_fmt(o['standing_gate_s'], 2)} | {_fmt(o['readiness_gate_s'], 2)} | "
                         f"{_fmt(o['chain_calibrated_s'], 2)} | {_fmt(o['frame_us_mean'], 0)} |")
        lines.append("")

    lines += ["## Input realism / delivery stats (per scenario, mean over seeds, calibration `tpose`)", "",
              "| scenario | loop Hz | sync fail | skipped primary | duplicates | reproj px | tri lower conf<0.1 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    runs = [r for r in results["raw"]["runs"] if r["calibration"] == "tpose"]
    for scenario in SCENARIOS:
        rows = [r["delivery"] for r in runs if r["scenario"] == scenario]
        avg = {k: float(np.mean([row[k] for row in rows])) for k in rows[0]}
        lines.append(f"| {scenario} | {avg['loop_hz']:.1f} | {avg['sync_fail_frac']:.3f} | "
                     f"{avg['skipped_primary_frac']:.3f} | {avg['duplicate_frac']:.3f} | "
                     f"{avg['mean_reprojection_px']:.2f} | {avg['tri_lower_conf_below_0.1_frac']:.3f} |")
    calib_rows = [r["calibration_info"] for r in runs if r["scenario"] == "clean"]
    lines += ["", "T-pose calibration error (clean scenario, per seed): " + "; ".join(
        json.dumps(c) for c in calib_rows), ""]

    lines += ["## Calibration floor: raw chain with noise profile `none` (no 2D noise/quantization)", "",
              "| calibration | MPJPE mm | lower mm | bone std mm | depth err IK deg | depth err final deg | "
              "lag knee final ms | valgus@true-bottom err deg | trunk MAE deg |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for cal in CALIBRATIONS:
        o = floor[cal]["overall_mean_of_scenarios"]
        lines.append(f"| {cal} | {_fmt(o['mpjpe_mm'])} | {_fmt(o['mpjpe_lower_mm'])} | {_fmt(o['bone_len_std_mm'])} | "
                     f"{_fmt(o['depth_err_ik_deg'])} | {_fmt(o['depth_err_final_deg'])} | "
                     f"{_fmt(o['lag_knee_flex_final_ms'])} | {_fmt(o['valgus_tb_err_deg'])} | "
                     f"{_fmt(o['trunk_flex_final_mae_deg'])} |")
    lines.append("")
    if sensitivity:
        lines += ["## Noise sensitivity (calibration `perfect`, overall mean over scenarios)", "",
                  "| noise | chain | MPJPE mm | p95 mm | depth err IK | depth err final | knee MAE moving IK | "
                  "valgus@true-bottom err | IK knee=0 frac |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for noise_name, chain_results in sensitivity.items():
            for chain, res in chain_results.items():
                o = res["summary"]["overall_mean_of_scenarios"]
                lines.append(f"| {noise_name} | {chain} | {_fmt(o['mpjpe_mm'])} | {_fmt(o['p95_err_mm'])} | "
                             f"{_fmt(o['depth_err_ik_deg'])} | {_fmt(o['depth_err_final_deg'])} | "
                             f"{_fmt(o['knee_flex_ik_moving_mae_deg'])} | {_fmt(o['valgus_tb_err_deg'])} | "
                             f"{_fmt(o['ik_knee_zero_frac'], 3)} |")
    return "\n".join(lines) + "\n"


def _extra_markdown(diagnostics: dict, low_fps: dict) -> str:
    keys = [("mpjpe_mm", "MPJPE mm"), ("ik_knee_zero_frac", "IK knee=0 frac"),
            ("knee_flex_ik_moving_mae_deg", "knee MAE moving IK"), ("depth_err_ik_deg", "depth err IK"),
            ("depth_err_final_deg", "depth err final"), ("bottom_sel_delay_ms", "bottom delay ms"),
            ("valgus_sel_err_deg", "valgus@pipeline-bottom err")]
    lines = ["", "## Diagnostic: how much error is the IK dropping joints with confidence < 0.1?",
             "`diag_conf_floor` keeps raw positions but floors confidences at 0.11 (NOT a proposed filter).", "",
             "| calibration | chain | " + " | ".join(k[1] for k in keys) + " |", "|---|---|" + "---:|" * len(keys)]
    for cal in CALIBRATIONS:
        for chain, res in diagnostics.items():
            o = res["summary"][cal]["overall_mean_of_scenarios"]
            lines.append(f"| {cal} | {chain} | " + " | ".join(_fmt(o.get(k[0]), 2 if "frac" in k[0] else 1)
                                                            for k in keys) + " |")
    keys2 = [("duplicate_frac", "dup frac"), ("mpjpe_mm", "MPJPE mm"), ("lag_knee_flex_ik_ms", "lag knee IK ms"),
             ("knee_flex_tb_err_deg", "knee@true-bottom err IK"), ("depth_err_final_deg", "depth err final"),
             ("jitter_still_p50_mm", "still jitter p50 mm")]
    lines += ["", "## Delivery sensitivity: cameras at 24 fps (low light) vs loop ~28 Hz -> duplicated frames "
              "(calibration `perfect`)", "", "| chain | " + " | ".join(k[1] for k in keys2) + " |",
              "|---|" + "---:|" * len(keys2)]
    for chain, res in low_fps.items():
        o = res["summary"]["overall_mean_of_scenarios"]
        dup = float(np.mean([r["delivery"]["duplicate_frac"] for r in res["runs"]]))
        cells = [f"{dup:.3f}"] + [_fmt(o.get(k[0])) for k in keys2[1:]]
        lines.append(f"| {chain} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--no-sensitivity", action="store_true")
    args = parser.parse_args()

    t_start = time.perf_counter()
    factories = baseline_factories()
    results = evaluate_chains(factories, scenarios=SCENARIOS, calibration=CALIBRATIONS, n_seeds=args.seeds,
                              noise="realistic", workers=args.workers)
    t_main = time.perf_counter() - t_start
    floor = evaluate_chains({"raw": factories["raw"]}, scenarios=SCENARIOS, calibration=CALIBRATIONS,
                            n_seeds=args.seeds, noise="none", workers=args.workers)["raw"]["summary"]
    sensitivity = {}
    if not args.no_sensitivity:
        subset = {k: factories[k] for k in ("raw", "current", "full_original")}
        for noise_name in ("mild", "harsh"):
            sensitivity[noise_name] = evaluate_chains(subset, scenarios=SCENARIOS, calibration="perfect",
                                                      n_seeds=args.seeds, noise=noise_name, workers=args.workers)
    diagnostics = evaluate_chains({"raw": factories["raw"], "diag_conf_floor": DiagnosticConfidenceFloor},
                                  scenarios=SCENARIOS, calibration=CALIBRATIONS, n_seeds=args.seeds,
                                  noise="realistic", workers=args.workers)
    low_fps = evaluate_chains({k: factories[k] for k in ("raw", "current", "full_original", "smooth_only")},
                              scenarios=SCENARIOS, calibration="perfect", n_seeds=args.seeds, noise="realistic",
                              workers=args.workers, delivery_config=DeliveryConfig(camera_fps=24.0))
    wall = time.perf_counter() - t_start
    meta = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "wall_s": wall, "main_wall_s": t_main,
            "seeds": list(range(args.seeds)), "workers": args.workers,
            "n_runs": sum(len(r["runs"]) for r in results.values())}
    noise_calib = json.loads((HERE / "noise_calib" / "noise_params.json").read_text())
    payload = {
        "meta": meta,
        "measured_noise": {"groups": noise_calib["groups"], "outliers": noise_calib["outliers"],
                           "slow_component": noise_calib["slow_component"]},
        "results": {chain: {"summary": res["summary"], "runs": res["runs"]} for chain, res in results.items()},
        "calibration_floor_noise_none": floor,
        "noise_sensitivity": {noise_name: {chain: res["summary"] for chain, res in chain_results.items()}
                              for noise_name, chain_results in sensitivity.items()},
        "diagnostic_conf_floor": {chain: res["summary"] for chain, res in diagnostics.items()},
        "camera_fps_24": {chain: res["summary"] for chain, res in low_fps.items()},
    }
    (HERE / "baseline_results.json").write_text(json.dumps(_json_safe(payload), indent=1, allow_nan=False))
    (HERE / "baseline_summary.md").write_text(_summary_markdown(results, floor, sensitivity, meta)
                                              + _extra_markdown(diagnostics, low_fps))
    print(f"done in {wall:.1f} s (main {t_main:.1f} s, workers {args.workers})")


if __name__ == "__main__":
    main()
