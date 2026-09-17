"""Real-video check of the planted-foot assumption with the production RTMPose-m halpe26 decode.

Fixed single camera, 1280x720, real squats. For ankle / big toe / heel (halpe26 15,16,20,21,24,25) measures:
jitter while standing, systematic drift of the planted keypoint from standing to squat bottom (would bias a
zero-velocity anchor), heel confidence. Pixels -> cm via standing shank length (knee->ankle) = 0.443 m.
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
    "/Users/naiahoard/NowvaLiveKit/recordings/squat_20260427_143958.mp4",
    "/Users/naiahoard/NowvaLiveKit/recordings/squat_20260421_124127.mp4",
]
SHANK_M = 0.443
H26 = {"hip_l": 11, "hip_r": 12, "knee_l": 13, "knee_r": 14, "ankle_l": 15, "ankle_r": 16,
       "bigtoe_l": 20, "bigtoe_r": 21, "smalltoe_l": 22, "smalltoe_r": 23, "heel_l": 24, "heel_r": 25}
FOOT_NAMES = ["ankle_l", "ankle_r", "bigtoe_l", "bigtoe_r", "heel_l", "heel_r"]


def _extract(video: str, est: RTMPoseEstimator) -> np.ndarray:
    cache = OUT / (Path(video).stem + "_h26.npy")
    if cache.exists():
        return np.load(cache)
    cap = cv2.VideoCapture(video)
    rows = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        blob, sx, sy = est._preprocess(frame)
        xl, yl = est._run_padded(blob)
        rows.append(est._decode_simcc(xl, yl, sx, sy))
    arr = np.array(rows)
    np.save(cache, arr)
    return arr


def main() -> None:
    est = RTMPoseEstimator(keypoint_format="halpe26", batch_size=1)
    est.initialize()
    summary = {}
    for video in VIDEOS:
        arr = _extract(video, est)
        fps = cv2.VideoCapture(video).get(cv2.CAP_PROP_FPS)
        hip_y = (arr[:, H26["hip_l"], 1] + arr[:, H26["hip_r"], 1]) / 2
        lo, hi = np.percentile(hip_y, 5), np.percentile(hip_y, 95)
        span = hi - lo
        standing = hip_y < lo + 0.10 * span
        bottom = hip_y > hi - 0.10 * span
        shank_px = np.median(np.concatenate([
            np.linalg.norm(arr[standing, H26["knee_l"], :2] - arr[standing, H26["ankle_l"], :2], axis=1),
            np.linalg.norm(arr[standing, H26["knee_r"], :2] - arr[standing, H26["ankle_r"], :2], axis=1)]))
        cm_per_px = 100 * SHANK_M / shank_px
        res = {"frames": int(len(arr)), "fps": fps, "squat_span_px": float(span),
               "hip_drop_cm": float(span * cm_per_px), "cm_per_px": float(cm_per_px),
               "n_standing": int(standing.sum()), "n_bottom": int(bottom.sum())}
        for name in FOOT_NAMES:
            k = H26[name]
            xy = arr[:, k, :2]
            conf = arr[:, k, 2]
            stand_med = np.median(xy[standing], axis=0)
            bottom_med = np.median(xy[bottom], axis=0)
            drift = (bottom_med - stand_med) * cm_per_px
            diffs = np.diff(xy[standing], axis=0)
            jitter = np.std(diffs, axis=0) / np.sqrt(2) * cm_per_px
            whole = np.std(xy, axis=0) * cm_per_px
            res[name] = {
                "drift_bottom_minus_stand_cm_xy": [round(float(v), 2) for v in drift],
                "jitter_standing_cm_xy": [round(float(v), 2) for v in jitter],
                "std_whole_video_cm_xy": [round(float(v), 2) for v in whole],
                "conf_p5_p50": [round(float(np.percentile(conf, 5)), 3), round(float(np.median(conf)), 3)],
            }
        # heel below ankle in image (y-down px): foot geometry sanity
        res["heel_minus_ankle_y_cm_standing"] = float(np.median(
            arr[standing, H26["heel_l"], 1] - arr[standing, H26["ankle_l"], 1]) * cm_per_px)
        summary[Path(video).stem] = res
        print(Path(video).stem, json.dumps(res, indent=None)[:1500])
    with open(OUT / "real_foot_drift.json", "w") as fh:
        json.dump(summary, fh, indent=1)


if __name__ == "__main__":
    main()
