"""Run the production RTMPose-m halpe26 decode on data/squats.mov and save raw keypoints.

The portrait 1080x1920 clip is embedded in a 1280x720 landscape canvas (scaled to 405x720,
edge-replicated sides) so the full-frame 192x256 squash matches the 720p production cameras.
Output: squats_mov_rtm.npz with kpts (T, 26, 3) [x_px, y_px, sigmoid_conf] pre-threshold, fps.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/Users/naiahoard/NowvaLiveKit/src")
from biomechanics.pose.rtmpose import RTMPoseEstimator  # noqa: E402

OUT_DIR = Path(__file__).parent
REPO = Path("/Users/naiahoard/NowvaLiveKit")
CANVAS_W, CANVAS_H = 1280, 720


def _make_canvas(frame_portrait: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame_portrait, (405, 720), interpolation=cv2.INTER_AREA)
    left = (CANVAS_W - 405) // 2
    right = CANVAS_W - 405 - left
    return cv2.copyMakeBorder(small, 0, 0, left, right, cv2.BORDER_REPLICATE)


def main() -> None:
    estimator = RTMPoseEstimator(keypoint_format="halpe26", batch_size=1)
    estimator.initialize()
    capture = cv2.VideoCapture(str(REPO / "data/squats.mov"))
    fps = capture.get(cv2.CAP_PROP_FPS)
    rows = []
    t_start = time.perf_counter()
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        canvas = _make_canvas(frame)
        blob, scale_x, scale_y = estimator._preprocess(canvas)
        x_logits, y_logits = estimator._run_padded(blob)
        rows.append(estimator._decode_simcc(x_logits, y_logits, scale_x, scale_y))
    kpts = np.array(rows)
    np.savez_compressed(OUT_DIR / "squats_mov_rtm.npz", kpts=kpts, fps=fps)
    print("frames", kpts.shape, "fps", fps, "sec", round(time.perf_counter() - t_start, 1))


if __name__ == "__main__":
    main()
