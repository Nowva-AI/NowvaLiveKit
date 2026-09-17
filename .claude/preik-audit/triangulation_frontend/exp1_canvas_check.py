"""Exp 1d: is the full-frame error from the 2.37x squash or from the edge-replicated canvas?
Compare FF-vs-crop distance for: replicate-padded 1280x720, gray-padded 1280x720, native portrait 1080x1920
squash (1.33x the other way), and native portrait letterboxed to 3:4 (no distortion, no crop)."""
from __future__ import annotations
import sys
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
sys.path.insert(0, str(Path(__file__).parent))
from exp1_analyze import BODY, SIGMA_X, SIGMA_Y, dec_parabola  # noqa: E402
from exp1_run_detector import IN_H, IN_W, REPO, crop_blob, crop_params_from_kpts, make_canvas, normalize  # noqa: E402

OUT = Path(__file__).parent
sess = ort.InferenceSession(str(REPO / "src/biomechanics/pose/models/rtmpose-m-halpe26-256x192.onnx"), providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0].name
d = np.load(OUT / "exp1_logits.npz")


def run_affine(img, center, scale):
    xl, yl = sess.run(None, {inp: normalize(crop_blob(img, center, scale))})
    bx, by = dec_parabola(xl[0]), dec_parabola(yl[0])
    return np.stack([(bx / 2 - IN_W / 2) * scale[0] / IN_W + center[0], (by / 2 - IN_H / 2) * scale[1] / IN_H + center[1]], -1), np.minimum(xl[0].max(-1), yl[0].max(-1))


def run_squash(img):
    h, w = img.shape[:2]
    return run_affine(img, np.array([w / 2, h / 2]), np.array([w, h], float))


def gray_canvas(fr):
    small = cv2.resize(fr, (405, 720), interpolation=cv2.INTER_AREA)
    c = np.full((720, 1280, 3), 127, np.uint8)
    c[:, 437:842] = small
    return c


variants = {"replicate_1280x720": [], "gray_1280x720": [], "portrait_native_squash": [], "portrait_letterbox_3x4": []}
cap = cv2.VideoCapture(str(REPO / "data/squats.mov"))
i = 0
while True:
    ok, fr = cap.read()
    if not ok:
        break
    if i % 9 == 0:
        c_center, c_scale = d["centers"][i], d["scales"][i]
        for name in variants:
            if name == "replicate_1280x720":
                img = make_canvas(fr); off = 0.0; s = 1.0
            elif name == "gray_1280x720":
                img = gray_canvas(fr); off = 0.0; s = 1.0
            else:
                img = fr; s = 1920 / 720; off = 437.0  # canvas x = x_portrait/s + 437
            if name == "portrait_letterbox_3x4":
                ff, _ = run_affine(img, np.array([540.0, 960.0]), np.array([1440.0, 1920.0]))
            else:
                ff, _ = run_squash(img)
            ff_canvas = ff / s + np.array([off if s != 1.0 else 0.0, 0.0])
            # crop in the same image space
            cc = c_center if s == 1.0 else (c_center - np.array([437.0, 0])) * s
            cs = c_scale * s
            cr, _ = run_affine(img, cc, cs)
            cr_canvas = cr / s + np.array([off if s != 1.0 else 0.0, 0.0])
            variants[name].append(np.linalg.norm(ff_canvas - cr_canvas, axis=-1)[BODY])
    i += 1
for name, v in variants.items():
    v = np.array(v)
    print(f"{name:28s} FF-vs-crop body dist (canvas px): median {np.median(v):6.2f}  mean {v.mean():6.2f}  p95 {np.percentile(v, 95):6.2f}")
