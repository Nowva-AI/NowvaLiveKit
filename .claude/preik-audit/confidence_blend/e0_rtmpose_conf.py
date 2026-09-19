"""E0: real RTMPose-m halpe26 confidence distribution + 2D jitter on real squat videos.

Runs the production RTMPoseEstimator (full-frame squash, argmax SimCC, sigmoid conf) on
1280x720 squat recordings. Records raw (pre-threshold) decode so we see the full conf range.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/Users/naiahoard/NowvaLiveKit/src")
from biomechanics.pose.rtmpose import RTMPoseEstimator  # noqa: E402

OUT = Path(__file__).parent
VIDEOS = [
    "/Users/naiahoard/NowvaLiveKit/recordings/squat_20260530_012118.mp4",
    "/Users/naiahoard/NowvaLiveKit/recordings/squat_20260515_145040.mp4",
    "/Users/naiahoard/NowvaLiveKit/recordings/squat_20260512_134408.mp4",
]
IDX = list(range(17)) + [20, 21]  # coco17 + big toes (what production keeps)

est = RTMPoseEstimator(keypoint_format="halpe26", batch_size=1)
est.initialize()

all_data = {}
for video in VIDEOS:
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    rows = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        blob, sx, sy = est._preprocess(frame)
        xl, yl = est._run_padded(blob)
        kpts = est._decode_simcc(xl, yl, sx, sy)  # (26,3)
        rows.append(kpts[IDX])
    arr = np.array(rows)  # (T,19,3)
    np.save(OUT / (Path(video).stem + "_rtm.npy"), arr)
    all_data[Path(video).stem] = (arr, fps)
    print(video, arr.shape, fps)

names = ["nose", "leye", "reye", "lear", "rear", "lsho", "rsho", "lelb", "relb", "lwri", "rwri",
         "lhip", "rhip", "lknee", "rknee", "lank", "rank", "ltoe", "rtoe"]
summary = {}
for stem, (arr, fps) in all_data.items():
    conf = arr[:, :, 2]
    # Jitter: residual vs centered 5-frame quadratic (Savitzky-Golay) fit
    from scipy.signal import savgol_filter
    xy = arr[:, :, :2]
    smooth = savgol_filter(xy, 7, 2, axis=0)
    resid = np.linalg.norm(xy - smooth, axis=2)  # (T,19)
    per = {}
    for k, n in enumerate(names):
        c = conf[:, k]
        dc = np.abs(np.diff(c))
        r = resid[:, k]
        corr = float(np.corrcoef(c, r)[0, 1]) if r.std() > 0 else float("nan")
        per[n] = {
            "conf_p5": round(float(np.percentile(c, 5)), 3),
            "conf_p50": round(float(np.percentile(c, 50)), 3),
            "conf_p95": round(float(np.percentile(c, 95)), 3),
            "frac_below_0.3": round(float((c < 0.3).mean()), 3),
            "abs_dconf_p50": round(float(np.median(dc)), 3),
            "abs_dconf_p95": round(float(np.percentile(dc, 95)), 3),
            "resid_px_rms": round(float(np.sqrt((r ** 2).mean())), 2),
            "corr_conf_vs_resid": round(corr, 3),
        }
    # Pooled: does low conf predict large residual? bin by conf
    c_all = conf[:, 5:].ravel()
    r_all = resid[:, 5:].ravel()
    bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    binned = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (c_all >= lo) & (c_all < hi)
        if m.sum() > 10:
            binned[f"{lo:.1f}-{hi:.2f}"] = {"n": int(m.sum()),
                                           "resid_rms_px": round(float(np.sqrt((r_all[m] ** 2).mean())), 2),
                                           "resid_p95_px": round(float(np.percentile(r_all[m], 95)), 2)}
    summary[stem] = {"fps": fps, "frames": int(arr.shape[0]), "per_keypoint": per, "resid_by_conf_bin": binned,
                     "min_conf_overall": round(float(conf.min()), 3)}

(OUT / "e0_summary.json").write_text(json.dumps(summary, indent=1))
for stem, s in summary.items():
    print("==", stem, s["fps"], s["frames"], "min conf", s["min_conf_overall"])
    for n in ["lsho", "lhip", "rhip", "lknee", "rknee", "lank", "rank", "ltoe", "rtoe", "lwri"]:
        print(n, s["per_keypoint"][n])
    print("resid by conf bin", s["resid_by_conf_bin"])
