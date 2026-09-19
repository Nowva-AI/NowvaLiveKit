"""Exp 1a: run RTMPose halpe26 on data/squats.mov under full-frame squash (production) and
person-crop (bbox tracked from the previous crop pose), save raw SimCC logits + mapping params.

The portrait 1080x1920 video is embedded into a 1280x720 landscape canvas (scaled to 405x720,
edge-replicated sides) so the full-frame squash has the same geometry as the 720p production cams.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, "/Users/naiahoard/NowvaLiveKit/src")
from biomechanics.pose.rtmpose import RTMPoseEstimator  # noqa: E402

OUT = Path(__file__).parent
REPO = Path("/Users/naiahoard/NowvaLiveKit")
CANVAS_W, CANVAS_H = 1280, 720
IN_W, IN_H = 192, 256
PADDING = 1.25


def make_canvas(frame_portrait: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame_portrait, (405, 720), interpolation=cv2.INTER_AREA)
    left = (CANVAS_W - 405) // 2
    right = CANVAS_W - 405 - left
    return cv2.copyMakeBorder(small, 0, 0, left, right, cv2.BORDER_REPLICATE)


def crop_params_from_kpts(kpts_xy: np.ndarray, conf: np.ndarray, thr: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    good = conf >= thr
    pts = kpts_xy[good]
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    w, h = (x2 - x1), (y2 - y1)
    aspect = IN_W / IN_H
    if w > h * aspect:
        h = w / aspect
    else:
        w = h * aspect
    scale = np.array([w, h]) * PADDING
    return center, scale


def crop_blob(img: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    sx = IN_W / scale[0]
    sy = IN_H / scale[1]
    M = np.array([[sx, 0, IN_W / 2 - center[0] * sx], [0, sy, IN_H / 2 - center[1] * sy]], dtype=np.float64)
    warped = cv2.warpAffine(img, M, (IN_W, IN_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return warped


def normalize(resized_bgr: np.ndarray) -> np.ndarray:
    from biomechanics.pose.rtmpose import IMAGENET_OFFSET_CHW, IMAGENET_SCALE_CHW
    chw = np.ascontiguousarray(resized_bgr.transpose(2, 0, 1)[::-1]).astype(np.float32)
    chw *= IMAGENET_SCALE_CHW
    chw -= IMAGENET_OFFSET_CHW
    return chw[np.newaxis]


def main() -> None:
    est = RTMPoseEstimator(keypoint_format="halpe26", batch_size=1,
                           model_path=str(REPO / "src/biomechanics/pose/models/rtmpose-m-halpe26-256x192.onnx"))
    sess = ort.InferenceSession(str(est._model_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name

    cap = cv2.VideoCapture(str(REPO / "data/squats.mov"))
    ff_x, ff_y, cr_x, cr_y, centers, scales = [], [], [], [], [], []
    prev_center = prev_scale = None
    t_inf = []
    n = 0
    first_canvas = None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        canvas = make_canvas(fr)
        if first_canvas is None:
            first_canvas = canvas.copy()
            cv2.imwrite(str(OUT / "canvas_frame0.jpg"), canvas)
        blob, scx, scy = est._preprocess(canvas)
        t0 = time.perf_counter()
        xl, yl = sess.run(None, {inp: blob})
        t_inf.append(time.perf_counter() - t0)
        ff_x.append(xl[0]); ff_y.append(yl[0])
        kff = est._decode_simcc(xl, yl, scx, scy)
        if prev_center is None:
            prev_center, prev_scale = crop_params_from_kpts(kff[:, :2], kff[:, 2])
        cblob = normalize(crop_blob(canvas, prev_center, prev_scale))
        cxl, cyl = sess.run(None, {inp: cblob})
        cr_x.append(cxl[0]); cr_y.append(cyl[0])
        centers.append(prev_center.copy()); scales.append(prev_scale.copy())
        # decode crop (argmax) to update bbox for next frame
        u = np.argmax(cxl[0], axis=1) / 2.0
        v = np.argmax(cyl[0], axis=1) / 2.0
        kx = (u - IN_W / 2) * prev_scale[0] / IN_W + prev_center[0]
        ky = (v - IN_H / 2) * prev_scale[1] / IN_H + prev_center[1]
        conf = 1 / (1 + np.exp(-np.minimum(cxl[0].max(1), cyl[0].max(1))))
        prev_center, prev_scale = crop_params_from_kpts(np.stack([kx, ky], 1), conf)
        n += 1
    np.savez_compressed(OUT / "exp1_logits.npz", ff_x=np.array(ff_x), ff_y=np.array(ff_y), cr_x=np.array(cr_x),
                        cr_y=np.array(cr_y), centers=np.array(centers), scales=np.array(scales),
                        ff_scale=np.array([CANVAS_W / IN_W, CANVAS_H / IN_H]))
    print("frames", n, "cpu inference ms median", 1000 * np.median(t_inf))


if __name__ == "__main__":
    main()
