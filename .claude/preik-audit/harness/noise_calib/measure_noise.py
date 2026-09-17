"""Measure realistic RTMPose-m (production decode) 2D noise statistics from real squat video.

Inputs: squats_mov_rtm.npz (this folder, data/squats.mov embedded in a 720p canvas) and the colleague's
real 1280x720 recordings decoded with the same production code (confidence_blend/*_rtm.npy).
Outputs noise_params.json: per keypoint-group white jitter sigma (fitted through the SimCC quantizer),
slow-component bound, confidence distribution, outlier / swap rates. Numbers feed the harness defaults.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import savgol_filter

HERE = Path(__file__).parent
AUDIT = HERE.parent.parent
QUANT_X_PX = 0.5 * 1280 / 192
QUANT_Y_PX = 0.5 * 720 / 256
OUTLIER_PX = 30.0
NAMES = ["nose", "leye", "reye", "lear", "rear", "lsho", "rsho", "lelb", "relb", "lwri", "rwri",
         "lhip", "rhip", "lknee", "rknee", "lank", "rank", "ltoe", "rtoe"]
GROUPS = {
    "head": [0, 1, 2, 3, 4], "shoulder": [5, 6], "elbow": [7, 8], "wrist": [9, 10],
    "hip": [11, 12], "knee": [13, 14], "ankle": [15, 16], "toe": [17, 18],
}
COCO19_FROM_HALPE = list(range(17)) + [20, 21]


def _load_sequences() -> dict[str, tuple[np.ndarray, float]]:
    sequences = {}
    mov = np.load(HERE / "squats_mov_rtm.npz")
    sequences["squats_mov(canvas)"] = (mov["kpts"][:, COCO19_FROM_HALPE], float(mov["fps"]))
    for stem, fps in [("squat_20260515_145040", 30.0), ("squat_20260512_134408", 30.0),
                      ("squat_20260530_012118", 15.0)]:
        sequences[stem] = (np.load(AUDIT / "confidence_blend" / f"{stem}_rtm.npy"), fps)
    return sequences


def _still_standing_mask(kpts: np.ndarray, fps: float) -> np.ndarray:
    legs = kpts[:, 11:17, :2].mean(axis=1)
    smooth = uniform_filter1d(median_filter(legs, size=(5, 1)), 5, axis=0)
    speed_px_per_s = np.linalg.norm(np.gradient(smooth, axis=0), axis=1) * fps
    hip_y = kpts[:, 11:13, 1].mean(axis=1)
    low, high = np.percentile(hip_y, 5), np.percentile(hip_y, 95)
    standing = hip_y < low + 0.2 * (high - low)
    still = standing & (speed_px_per_s < 25.0)
    # keep runs of >= 10 frames
    mask = np.zeros_like(still)
    run_start = None
    for i, flag in enumerate(np.append(still, False)):
        if flag and run_start is None:
            run_start = i
        elif not flag and run_start is not None:
            if i - run_start >= 10:
                mask[run_start:i] = True
            run_start = None
    return mask


def _simulated_quantized_diff_std(sigma: float, quant: float, rng: np.random.Generator) -> tuple[float, float]:
    offsets = rng.uniform(0, quant, size=4000)
    a = np.round((offsets + rng.normal(0, sigma, 4000)) / quant) * quant
    b = np.round((offsets + rng.normal(0, sigma, 4000)) / quant) * quant
    d = a - b
    return float(d.std()), float((d == 0).mean())


def _fit_white_sigma(observed_diff_std: float, quant: float) -> float:
    rng = np.random.default_rng(0)
    grid = np.linspace(0.05, 15.0, 300)
    stds = np.array([_simulated_quantized_diff_std(s, quant, rng)[0] for s in grid])
    return float(grid[np.argmin(np.abs(stds - observed_diff_std))])


def _run_lengths(flags: np.ndarray) -> list[int]:
    lengths, count = [], 0
    for flag in np.append(flags, False):
        if flag:
            count += 1
        elif count:
            lengths.append(count)
            count = 0
    return lengths


def main() -> None:
    sequences = _load_sequences()
    report: dict = {"quant_px": {"x": QUANT_X_PX, "y": QUANT_Y_PX}, "sequences": {}}
    pooled_diffs = {g: {"x": [], "y": []} for g in GROUPS}
    pooled_conf = {g: [] for g in GROUPS}
    pooled_conf_still = {g: [] for g in GROUPS}
    pooled_outlier = {g: [] for g in GROUPS}
    pooled_outlier_runs = []
    pooled_outlier_conf, pooled_normal_conf = [], []
    pooled_len_slow = {"femur": [], "tibia": []}
    pooled_conf_ac1 = []
    swap_frames = 0
    total_frames = 0

    for name, (kpts, fps) in sequences.items():
        still = _still_standing_mask(kpts, fps)
        pairs = still[1:] & still[:-1]
        diffs = np.diff(kpts[:, :, :2], axis=0)[pairs]  # (n, 19, 2)
        conf = kpts[:, :, 2]
        seq = {"fps": fps, "frames": int(len(kpts)), "still_frames": int(still.sum()), "groups": {}}

        robust = savgol_filter(median_filter(kpts[:, :, :2], size=(7, 1, 1)), 7, 2, axis=0)
        resid = np.linalg.norm(kpts[:, :, :2] - robust, axis=2)
        outlier = resid > OUTLIER_PX

        for group, idx in GROUPS.items():
            dx = diffs[:, idx, 0].ravel()
            dy = diffs[:, idx, 1].ravel()
            dx = dx[np.abs(dx) < 20]
            dy = dy[np.abs(dy) < 20]
            if fps >= 25:
                pooled_diffs[group]["x"].append(dx)
                pooled_diffs[group]["y"].append(dy)
            pooled_conf[group].append(conf[:, idx].ravel())
            pooled_conf_still[group].append(conf[still][:, idx].ravel())
            pooled_outlier[group].append(outlier[:, idx].ravel())
            seq["groups"][group] = {
                "diff_std_x_px": round(float(dx.std()), 2) if dx.size else None,
                "diff_std_y_px": round(float(dy.std()), 2) if dy.size else None,
                "frac_dx_zero": round(float((dx == 0).mean()), 2) if dx.size else None,
                "conf_p50": round(float(np.median(conf[:, idx])), 3),
                "outlier_rate": round(float(outlier[:, idx].mean()), 4),
            }
        for k in range(11, 19):
            pooled_outlier_runs += _run_lengths(outlier[:, k])
        pooled_outlier_conf.append(conf[:, 11:19][outlier[:, 11:19]])
        pooled_normal_conf.append(conf[:, 11:19][~outlier[:, 11:19]])

        # slow component bound: 2D segment length variation inside still runs (sway cancels)
        for label, (a_idx, b_idx) in {"femur": (11, 13), "tibia": (13, 15)}.items():
            for side in (0, 1):
                length = np.linalg.norm(kpts[:, a_idx + side, :2] - kpts[:, b_idx + side, :2], axis=1)
                smooth_len = uniform_filter1d(length, 7)
                for run_start, run_len in _still_runs(still):
                    if run_len >= 20:
                        seg = smooth_len[run_start + 3: run_start + run_len - 3]
                        pooled_len_slow[label].append(float(seg.std()))
        for k in range(11, 19):
            c = conf[:, k] - uniform_filter1d(conf[:, k], 15)
            pooled_conf_ac1.append(float(np.corrcoef(c[1:], c[:-1])[0, 1]))

        if name != "squats_mov(canvas)" or True:
            swapped = (kpts[:, 13, 0] < kpts[:, 14, 0] - 5) | (kpts[:, 15, 0] < kpts[:, 16, 0] - 5)
            swap_frames += int(swapped.sum())
            total_frames += len(kpts)
            seq["lr_swap_frames_frontal"] = int(swapped.sum())
        report["sequences"][name] = seq

    groups_out = {}
    for group in GROUPS:
        dx = np.concatenate(pooled_diffs[group]["x"])
        dy = np.concatenate(pooled_diffs[group]["y"])
        c_all = np.concatenate(pooled_conf[group])
        c_still = np.concatenate(pooled_conf_still[group])
        groups_out[group] = {
            "n_diff_pairs": int(dx.size),
            "diff_std_x_px": round(float(dx.std()), 3),
            "diff_std_y_px": round(float(dy.std()), 3),
            "frac_dx_zero": round(float((dx == 0).mean()), 3),
            "frac_dy_zero": round(float((dy == 0).mean()), 3),
            "white_sigma_x_px": round(_fit_white_sigma(float(dx.std()), QUANT_X_PX), 2),
            "white_sigma_y_px": round(_fit_white_sigma(float(dy.std()), QUANT_Y_PX), 2),
            "conf_p5": round(float(np.percentile(c_all, 5)), 3),
            "conf_p50": round(float(np.median(c_all)), 3),
            "conf_p95": round(float(np.percentile(c_all, 95)), 3),
            "conf_min": round(float(c_all.min()), 3),
            "conf_still_p50": round(float(np.median(c_still)), 3),
            "conf_still_std": round(float(c_still.std()), 4),
            "outlier_rate_gt30px": round(float(np.concatenate(pooled_outlier[group]).mean()), 4),
        }
    report["groups"] = groups_out
    runs = np.array(pooled_outlier_runs) if pooled_outlier_runs else np.array([0])
    report["outliers"] = {
        "threshold_px": OUTLIER_PX,
        "run_length_frames_p50": float(np.median(runs)),
        "run_length_frames_mean": round(float(runs.mean()), 2),
        "run_length_frames_p90": float(np.percentile(runs, 90)),
        "conf_during_outlier_p50": round(float(np.median(np.concatenate(pooled_outlier_conf))), 3),
        "conf_normal_p50": round(float(np.median(np.concatenate(pooled_normal_conf))), 3),
    }
    report["slow_component"] = {
        "femur_2d_length_std_px_in_still_runs_median": round(float(np.median(pooled_len_slow["femur"])), 2)
        if pooled_len_slow["femur"] else None,
        "tibia_2d_length_std_px_in_still_runs_median": round(float(np.median(pooled_len_slow["tibia"])), 2)
        if pooled_len_slow["tibia"] else None,
        "note": "7-frame-averaged 2D bone length std within still-standing runs; ~sqrt(2)*per-keypoint slow sigma "
                "(upper bound: includes residual white noise/sqrt(7) and tiny real posture changes)",
    }
    report["confidence_lag1_autocorr_detrended_median"] = round(float(np.nanmedian(pooled_conf_ac1)), 3)
    report["lr_swap_frame_rate_frontal"] = round(swap_frames / max(total_frames, 1), 4)
    (HERE / "noise_params.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({"groups": groups_out, "outliers": report["outliers"],
                      "slow": report["slow_component"],
                      "conf_ac1": report["confidence_lag1_autocorr_detrended_median"],
                      "swap": report["lr_swap_frame_rate_frontal"],
                      "still": {k: v["still_frames"] for k, v in report["sequences"].items()}}, indent=1))


def _still_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs, start = [], None
    for i, flag in enumerate(np.append(mask, False)):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i - start))
            start = None
    return runs


if __name__ == "__main__":
    main()
