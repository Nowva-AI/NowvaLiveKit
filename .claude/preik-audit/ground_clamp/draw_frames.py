"""Draw halpe26 foot/leg keypoints on real frames (standing vs bottom) to verify foot keypoint failures visually."""
from __future__ import annotations
import sys
from pathlib import Path
import cv2
import numpy as np

OUT = Path(__file__).parent
video, frames = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]
arr = np.load(OUT / f"{Path(video).stem}_h26.npy")
cap = cv2.VideoCapture(video)
tiles = []
for idx in frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    k = arr[idx]
    for a, b in ((11, 13), (13, 15), (12, 14), (14, 16), (15, 20), (15, 24), (16, 21), (16, 25)):
        cv2.line(frame, tuple(int(v) for v in k[a, :2]), tuple(int(v) for v in k[b, :2]), (0, 255, 0), 2)
    for j, col in ((15, (0, 0, 255)), (16, (255, 0, 0)), (20, (0, 255, 255)), (21, (255, 255, 0)), (24, (255, 0, 255)), (25, (0, 128, 255))):
        cv2.circle(frame, tuple(int(v) for v in k[j, :2]), 6, col, -1)
    cv2.putText(frame, f"frame {idx}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    tiles.append(cv2.resize(frame[180:720, 280:1000], (540, 405)))
cv2.imwrite(str(OUT / f"{Path(video).stem}_frames.jpg"), np.hstack(tiles))
