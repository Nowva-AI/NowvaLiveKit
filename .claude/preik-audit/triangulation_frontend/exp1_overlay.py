"""Exp 1c: overlay full-frame-squash (red) vs person-crop (green) keypoints, DARK decode, zoomed."""
from __future__ import annotations
import sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from exp1_analyze import SIGMA_X, SIGMA_Y, dec_parabola, to_canvas_crop, to_canvas_ff  # noqa: E402
from exp1_run_detector import REPO, make_canvas  # noqa: E402

OUT = Path(__file__).parent
EDGES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
         (15, 24), (15, 20), (16, 25), (16, 21), (20, 22), (21, 23)]
d = np.load(OUT / "exp1_logits.npz")
kff = to_canvas_ff(dec_parabola(d["ff_x"]), dec_parabola(d["ff_y"]), d["ff_scale"])
kcr = to_canvas_crop(dec_parabola(d["cr_x"]), dec_parabola(d["cr_y"]), d["centers"], d["scales"])
want = [40, 75, 120, 280]
cap = cv2.VideoCapture(str(REPO / "data/squats.mov"))
tiles = []
i = 0
while True:
    ok, fr = cap.read()
    if not ok:
        break
    if i in want:
        img = make_canvas(fr)
        for kp, col in ((kff[i], (0, 0, 255)), (kcr[i], (0, 255, 0))):
            for a, b in EDGES:
                cv2.line(img, tuple(np.int32(kp[a])), tuple(np.int32(kp[b])), col, 2)
            for p in kp:
                cv2.circle(img, tuple(np.int32(p)), 3, col, -1)
        tiles.append(cv2.resize(img[:, 400:880], (480, 720)))
    i += 1
cv2.imwrite(str(OUT / "overlay_ff_red_crop_green.jpg"), np.hstack(tiles))
