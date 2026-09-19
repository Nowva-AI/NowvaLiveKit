"""Evaluate person-based extrinsic calibration (contract K2, `PersonCalibrator`) on the ground-truth harness.

Each `person_ba*` mode calibrates the rig from the harness's own noisy detections of a walk-in + 2 reps (one capture
per seed, reused by every scenario of that seed) and then the `proposed` chain runs on the real DLTTriangulator with
that calibration, exactly like `perfect` and `tpose`. Modes: true K + bar scale, true K + height scale, true K
started from the T-pose calibration (refine), K with +-10 % focal error (bar and height scale).
Writes person_ba_results.json and person_ba_summary.md next to this file.
Run: /Users/naiahoard/NowvaLiveKit/venv/bin/python run_person_ba.py [--workers N] [--seeds 3]
"""

from __future__ import annotations

import os

# numpy uses Accelerate on macOS; its threaded LAPACK deadlocks in fork()ed workers (the calibrator's dense
# camera-only solve). Must be set before numpy loads.
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from preik_harness import PERSON_BA_MODES, SCENARIOS, evaluate_chains, production_chain, proposed_chain  # noqa: E402
from preik_harness import runner  # noqa: E402

HERE = Path(__file__).parent
BASELINES = ["perfect", "tpose"]
CALIBRATIONS = BASELINES + list(PERSON_BA_MODES)
MODE_NOTES = {
    "perfect": "true projection matrices",
    "tpose": "production T-pose PnP, guessed f = 0.8 w",
    "person_ba": "true K, bar scale, essential-matrix start",
    "person_ba_from_tpose": "true K, bar scale, refine() from the T-pose calibration",
    "person_ba_height": "true K, height-prior scale",
    "person_ba_k+10": "focal +10 %, bar scale",
    "person_ba_k-10": "focal -10 %, bar scale",
    "person_ba_k+10_height": "focal +10 %, height-prior scale",
    "person_ba_k-10_height": "focal -10 %, height-prior scale",
}
COLUMNS = [
    ("mpjpe_mm", "MPJPE mm", 1), ("mpjpe_lower_mm", "lower mm", 1), ("p95_err_mm", "p95 mm", 1),
    ("stance_width_mae_mm", "stance MAE mm", 1), ("bone_len_mae_mm", "bone len MAE mm", 1),
    ("knee_flex_tb_err_deg", "knee@true-bottom err deg", 2), ("knee_flex_ik_moving_mae_deg", "knee MAE moving deg", 2),
    ("depth_err_final_deg", "depth err deg", 2), ("valgus_tb_err_deg", "valgus@true-bottom err deg", 2),
    ("ik_knee_invalid_frac", "knee NaN frac", 3),
]
TARGET_MM = 3.0


def _warm_calibration(job: tuple[int, str]) -> dict:
    runner.person_ba_for_seed(*job)
    return runner._PERSON_BA_CACHE


def _fmt(value: float | None, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    return f"{value:.{digits}f}"


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


def _stat_mean(runs: list[dict], key: str) -> float | None:
    values = [r["delivery"].get(key) for r in runs if r["delivery"].get(key) is not None]
    return float(np.mean(values)) if values else None


def _calibration_rows(runs: list[dict], seeds: list[int]) -> list[dict]:
    rows = []
    for seed in seeds:
        info = next(r["calibration_info"] for r in runs if r["seed"] == seed)
        cams = [v for v in info.values() if isinstance(v, dict) and "centre_error_m" in v]
        rows.append({
            "seed": seed,
            "centre_error_mm": 1000.0 * float(np.mean([c["centre_error_m"] for c in cams])),
            "rotation_error_deg": float(np.mean([c["rotation_error_deg"] for c in cams])),
            "vertical_tilt_deg": info.get("vertical_tilt_deg"), "scale_error_pct": info.get("scale_error_pct"),
            "rms_px": info.get("rms_reprojection_px"), "initial_rms_px": info.get("initial_rms_reprojection_px"),
            "solve_time_s": info.get("solve_time_s"), "frames_used": info.get("frames_used"),
            "capture_frames": info.get("capture_frames"), "standing_frames": info.get("standing_frames"),
        })
    return rows


def _mean_of(rows: list[dict], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def _summary_markdown(results: dict, timing: dict, meta: dict) -> str:
    seeds = meta["seeds"]
    lines = ["# Person-based extrinsic calibration (K2) — triangulated 3-camera harness", "",
             f"Generated {meta['generated']}; wall {meta['wall_s']:.0f} s; seeds {seeds}; noise `realistic`; "
             f"scenarios: {', '.join(SCENARIOS)}; workers {meta['workers']}.", "",
             "- `person_ba*`: the production `PersonCalibrator` on the harness's own noisy detections of ONE capture per "
             "seed (the `walkout` generator with a separate seed: stand, walk 0.5 m with a 20 deg turn, stand, 2 reps; "
             "same detector noise, occlusion, L/R swaps and camera-delivery model as every run; bar ends simulated with "
             "3 px white + 3 px slow noise, ordered by image x like the YOLO detector). Every scenario of that seed is "
             "then triangulated with that calibration, as `perfect`/`tpose` are.",
             "- Gauge: the calibrator anchors its world frame on the lifter (heading + origin from the standing frames); "
             "the harness truth is anchored on the T-pose spot. Heading (yaw about the true vertical) and origin are "
             "removed with the capture's true joints; **tilt of the solved vertical, relative camera poses and scale are "
             "not corrected** and are inside every number below. Camera centre / rotation errors are after that "
             "yaw + origin alignment.",
             "- `floor mm`: calibration-only error — noise-free true projections triangulated with the calibration, "
             "hip-centred MPJPE vs truth (0 for `perfect`).",
             "- Chains: `proposed` (FixedLagKeypointSmoother + FootContactModel, new tail) and `raw`.", ""]
    for chain in results:
        lines += [f"## Chain `{chain}` — mean over {len(SCENARIOS)} scenarios x {len(seeds)} seeds", "",
                  "| calibration | " + " | ".join(c[1] for c in COLUMNS) + " | floor mm | reproj px | setup |",
                  "|---|" + "---:|" * (len(COLUMNS) + 2) + "---|"]
        for cal in CALIBRATIONS:
            overall = results[chain]["summary"][cal]["overall_mean_of_scenarios"]
            runs = [r for r in results[chain]["runs"] if r["calibration"] == cal]
            lines.append(f"| {cal} | " + " | ".join(_fmt(overall.get(key), digits) for key, _, digits in COLUMNS)
                         + f" | {_fmt(_stat_mean(runs, 'calibration_floor_mm'))} | "
                         f"{_fmt(_stat_mean(runs, 'mean_reprojection_px'), 2)} | {MODE_NOTES[cal]} |")
        lines.append("")
    reference = results["proposed"]["summary"]["perfect"]["overall_mean_of_scenarios"]
    lines += [f"## Targets (chain `proposed`): within {TARGET_MM:.0f} mm of `perfect`", "",
              "| calibration | MPJPE - perfect mm | stance MAE - perfect mm | met |", "|---|---:|---:|---|"]
    for cal in CALIBRATIONS[1:]:
        overall = results["proposed"]["summary"][cal]["overall_mean_of_scenarios"]
        d_mpjpe = overall["mpjpe_mm"] - reference["mpjpe_mm"]
        d_stance = overall["stance_width_mae_mm"] - reference["stance_width_mae_mm"]
        met = "yes" if d_mpjpe <= TARGET_MM and d_stance <= TARGET_MM else "no"
        lines.append(f"| {cal} | {d_mpjpe:+.1f} | {d_stance:+.1f} | {met} |")
    lines += ["", "## Calibration quality (mean over seeds; per-seed values in person_ba_results.json)", "",
              "| calibration | cam centre err mm | cam rotation err deg | vertical tilt deg | working-volume scale err % | "
              "BA reproj RMS px (start -> end) | frames used / captured | standing frames | calibrate() s (inside the parallel pool) |",
              "|---|---:|---:|---:|---:|---|---|---:|---:|"]
    for cal in CALIBRATIONS[1:]:
        runs = [r for r in results["proposed"]["runs"] if r["calibration"] == cal]
        rows = _calibration_rows(runs, seeds)
        lines.append(
            f"| {cal} | {_fmt(_mean_of(rows, 'centre_error_mm'))} | {_fmt(_mean_of(rows, 'rotation_error_deg'), 2)} | "
            f"{_fmt(_mean_of(rows, 'vertical_tilt_deg'), 2)} | {_fmt(_mean_of(rows, 'scale_error_pct'), 2)} | "
            f"{_fmt(_mean_of(rows, 'initial_rms_px'), 2)} -> {_fmt(_mean_of(rows, 'rms_px'), 2)} | "
            f"{_fmt(_mean_of(rows, 'frames_used'), 0)} / {_fmt(_mean_of(rows, 'capture_frames'), 0)} | "
            f"{_fmt(_mean_of(rows, 'standing_frames'), 0)} | {_fmt(_mean_of(rows, 'solve_time_s'), 2)} |")
    lines += ["", "`tpose` rows: centre / rotation errors as reported by the T-pose harness mode (no alignment needed: "
              "the T-pose defines the truth frame).", "",
              "## Per scenario, chain `proposed`: MPJPE mm / stance MAE mm (mean over seeds)", "",
              "| scenario | " + " | ".join(CALIBRATIONS) + " |", "|---|" + "---:|" * len(CALIBRATIONS)]
    for scenario in SCENARIOS:
        cells = []
        for cal in CALIBRATIONS:
            per = results["proposed"]["summary"][cal]["per_scenario"][scenario]
            cells.append(f"{_fmt(per['mpjpe_mm']['mean'])} / {_fmt(per['stance_width_mae_mm']['mean'])}")
        lines.append(f"| {scenario} | " + " | ".join(cells) + " |")
    lines += ["", "## `calibrate()` timing on this machine (Apple M2, single process, 150 frames, 3 cameras, 19 keypoints + bar; single-threaded BLAS)",
              "", "| mode | runs | mean s | min s | max s |", "|---|---:|---:|---:|---:|"]
    for mode, values in timing.items():
        lines.append(f"| {mode} | {len(values)} | {np.mean(values):.2f} | {np.min(values):.2f} | {np.max(values):.2f} |")
    lines.append("")
    return "\n".join(lines) + "\n"


def _time_calibrations(seeds: list[int]) -> dict[str, list[float]]:
    # re-solve in this (otherwise idle) process: the pool's own timings are inflated by the parallel workers
    timing: dict[str, list[float]] = {}
    for mode in ("person_ba", "person_ba_from_tpose", "person_ba_height"):
        for seed in seeds:
            runner._PERSON_BA_CACHE.clear()
            _, info = runner.person_ba_for_seed(seed, mode)
            timing.setdefault(mode, []).append(info["solve_time_s"])
    return timing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    seeds = list(range(args.seeds))

    t_start = time.perf_counter()
    jobs = [(seed, mode) for seed in seeds for mode in PERSON_BA_MODES]
    with mp.get_context("fork").Pool(args.workers) as pool:
        for solved in pool.map(_warm_calibration, jobs, chunksize=1):
            runner._PERSON_BA_CACHE.update(solved)  # inherited by the evaluation workers: one capture per seed
    t_calibrated = time.perf_counter() - t_start
    factories = {"proposed": proposed_chain(foot_contact=True), "raw": production_chain([])}
    results = evaluate_chains(factories, scenarios=SCENARIOS, calibration=CALIBRATIONS, n_seeds=args.seeds,
                              noise="realistic", workers=args.workers)
    timing = _time_calibrations(seeds)
    wall = time.perf_counter() - t_start
    meta = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "wall_s": wall, "calibration_wall_s": t_calibrated,
            "seeds": seeds, "workers": args.workers}
    payload = {"meta": meta, "timing_s": timing, "results": {
        chain: {"summary": res["summary"],
                "runs": [{k: run[k] for k in ("scenario", "seed", "calibration", "metrics", "delivery",
                                              "calibration_info")} for run in res["runs"]]}
        for chain, res in results.items()}}
    (HERE / "person_ba_results.json").write_text(json.dumps(_json_safe(payload), indent=1, allow_nan=False))
    (HERE / "person_ba_summary.md").write_text(_summary_markdown(results, timing, meta))
    print(f"done in {wall:.1f} s (calibrations {t_calibrated:.1f} s, workers {args.workers})")


if __name__ == "__main__":
    main()
