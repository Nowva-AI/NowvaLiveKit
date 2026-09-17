"""Exp 1b: decode variants, jitter, quantization, confidence semantics, crop vs full-frame.

Uses exp1_logits.npz from exp1_run_detector.py plus extra model runs (sub-pixel shift test,
occlusion test, no-person images, flip-consistency accuracy proxy).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from scipy.signal import savgol_filter

sys.path.insert(0, "/Users/naiahoard/NowvaLiveKit/src")
sys.path.insert(0, str(Path(__file__).parent))
from exp1_run_detector import (IN_H, IN_W, REPO, crop_blob, crop_params_from_kpts, make_canvas,  # noqa: E402
                               normalize)
from biomechanics.pose.rtmpose import RTMPoseEstimator  # noqa: E402

OUT = Path(__file__).parent
BODY = list(range(5, 17))          # shoulders..ankles
FEET = list(range(17, 26))          # head(17) neck(18) hip(19) big toes 20,21 small toes 22,23 heels 24,25
LEGS = [11, 12, 13, 14, 15, 16, 20, 21, 24, 25]
STILL_WINDOWS = [(66, 85), (137, 152), (211, 234)]
HALPE_FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 17, 18, 19, 21, 20, 23, 22, 25, 24]
RESULTS: dict = {}


# ---------------------------------------------------------------- decoders (return bin coords)
def dec_argmax(lg: np.ndarray) -> np.ndarray:
    return np.argmax(lg, axis=-1).astype(np.float64)


def dec_parabola(lg: np.ndarray) -> np.ndarray:
    idx = np.argmax(lg, axis=-1)
    n = lg.shape[-1]
    i0 = np.clip(idx, 1, n - 2)
    c = np.take_along_axis(lg, i0[..., None], -1)[..., 0]
    left = np.take_along_axis(lg, (i0 - 1)[..., None], -1)[..., 0]
    right = np.take_along_axis(lg, (i0 + 1)[..., None], -1)[..., 0]
    den = left - 2 * c + right
    off = np.where(den < -1e-9, 0.5 * (left - right) / np.where(den < -1e-9, den, -1), 0.0)
    return i0 + np.clip(off, -0.5, 0.5)


def dec_softargmax(lg: np.ndarray, beta: float = 10.0, half_window: int = 8) -> np.ndarray:
    idx = np.argmax(lg, axis=-1)
    n = lg.shape[-1]
    offs = np.arange(-half_window, half_window + 1)
    pos = np.clip(idx[..., None] + offs, 0, n - 1)
    vals = np.take_along_axis(lg, pos, -1)
    w = np.exp(beta * (vals - vals.max(-1, keepdims=True)))
    return (w * pos).sum(-1) / w.sum(-1)


def dec_dark(lg: np.ndarray, sigma: float) -> np.ndarray:
    # mmpose refine_simcc_dark replica, lg shape (..., K, W)
    blur = int((sigma * 20 - 7) // 3)
    blur -= int(blur % 2 == 0)
    shp = lg.shape
    flat = lg.reshape(-1, shp[-1]).astype(np.float32).copy()
    border = (blur - 1) // 2
    for r in range(flat.shape[0]):
        om = flat[r].max()
        dr = np.zeros((1, shp[-1] + 2 * border), np.float32)
        dr[0, border:-border] = flat[r]
        dr = cv2.GaussianBlur(dr, (blur, 1), 0)
        flat[r] = dr[0, border:-border] * (om / dr[0, border:-border].max())
    flat = np.log(np.clip(flat, 1e-3, 50.0))
    flat = np.pad(flat, ((0, 0), (2, 2)), mode="edge")
    idx = np.argmax(lg.reshape(-1, shp[-1]), -1)
    px = idx + 2
    g = lambda o: flat[np.arange(len(px)), px + o]  # noqa: E731
    dx = 0.5 * (g(1) - g(-1))
    dxx = 1e-9 + 0.25 * (g(2) - 2 * g(0) + g(-2))
    return (idx - dx / dxx).reshape(shp[:-1])


DECODERS = {
    "argmax": (lambda lg, s: dec_argmax(lg)),
    "parabola": (lambda lg, s: dec_parabola(lg)),
    "softargmax_b10_w8": (lambda lg, s: dec_softargmax(lg)),
    "dark": (lambda lg, s: dec_dark(lg, s)),
}
SIGMA_X, SIGMA_Y = 4.9, 5.66


def to_canvas_ff(bx: np.ndarray, by: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.stack([bx / 2 * scale[0], by / 2 * scale[1]], -1)


def to_canvas_crop(bx: np.ndarray, by: np.ndarray, centers: np.ndarray, scales: np.ndarray) -> np.ndarray:
    u, v = bx / 2, by / 2
    x = (u - IN_W / 2) * scales[:, 0:1] / IN_W + centers[:, 0:1]
    y = (v - IN_H / 2) * scales[:, 1:2] / IN_H + centers[:, 1:2]
    return np.stack([x, y], -1)


def jitter_still(kp: np.ndarray, joints: list[int]) -> float:
    # RMS residual (px, per axis combined) after quadratic detrend inside each still window
    res = []
    for a, b in STILL_WINDOWS:
        seg = kp[a:b, joints]  # (T, J, 2)
        t = np.arange(b - a)
        for j in range(seg.shape[1]):
            for ax in range(2):
                coef = np.polyfit(t, seg[:, j, ax], 2)
                res.append(seg[:, j, ax] - np.polyval(coef, t))
    return float(np.sqrt(np.mean(np.concatenate(res) ** 2)))


def hf_noise(kp: np.ndarray, joints: list[int]) -> float:
    sm = savgol_filter(kp[:, joints], 9, 3, axis=0)
    return float(np.sqrt(np.mean((kp[:, joints] - sm) ** 2)))


def frame_to_frame_zero_fraction(kp: np.ndarray, joints: list[int]) -> float:
    d = np.abs(np.diff(kp[:, joints], axis=0))
    return float(np.mean(d < 1e-6))


def main() -> None:
    d = np.load(OUT / "exp1_logits.npz")
    ff_x, ff_y, cr_x, cr_y = d["ff_x"], d["ff_y"], d["cr_x"], d["cr_y"]
    centers, scales, ff_scale = d["centers"], d["scales"], d["ff_scale"]
    decoded = {}
    table = {}
    for name, fn in DECODERS.items():
        kff = to_canvas_ff(fn(ff_x, SIGMA_X), fn(ff_y, SIGMA_Y), ff_scale)
        kcr = to_canvas_crop(fn(cr_x, SIGMA_X), fn(cr_y, SIGMA_Y), centers, scales)
        decoded[name] = (kff, kcr)
        table[name] = {
            "ff_still_jitter_px_body": jitter_still(kff, BODY),
            "crop_still_jitter_px_body": jitter_still(kcr, BODY),
            "ff_still_jitter_px_legs": jitter_still(kff, LEGS),
            "crop_still_jitter_px_legs": jitter_still(kcr, LEGS),
            "ff_hf_noise_px_body": hf_noise(kff, BODY),
            "crop_hf_noise_px_body": hf_noise(kcr, BODY),
            "ff_hf_noise_px_legs": hf_noise(kff, LEGS),
            "crop_hf_noise_px_legs": hf_noise(kcr, LEGS),
            "ff_zero_motion_fraction_body": frame_to_frame_zero_fraction(kff, BODY),
            "crop_zero_motion_fraction_body": frame_to_frame_zero_fraction(kcr, BODY),
        }
    RESULTS["decoders"] = table
    RESULTS["quant_step_px"] = {
        "ff_x": float(ff_scale[0] / 2), "ff_y": float(ff_scale[1] / 2),
        "crop_x_median": float(np.median(scales[:, 0]) / IN_W / 2), "crop_y_median": float(np.median(scales[:, 1]) / IN_H / 2),
        "crop_bbox_h_median_px": float(np.median(scales[:, 1])),
    }
    axis_tab = {}
    for name in ("argmax", "parabola", "dark", "softargmax_b10_w8"):
        kff, kcr = decoded[name]
        row = {}
        for cond, kp in (("ff", kff), ("crop", kcr)):
            for ax, axn in ((0, "x"), (1, "y")):
                res = []
                for a, b in STILL_WINDOWS:
                    t = np.arange(b - a)
                    for j in BODY:
                        s = kp[a:b, j, ax]
                        res.append(s - np.polyval(np.polyfit(t, s, 2), t))
                row[f"{cond}_{axn}"] = float(np.sqrt(np.mean(np.concatenate(res) ** 2)))
        axis_tab[name] = row
    RESULTS["still_jitter_axis"] = axis_tab

    # agreement full-frame vs crop (dark decode), per joint group
    kff, kcr = decoded["dark"]
    dist = np.linalg.norm(kff - kcr, axis=-1)
    RESULTS["ff_vs_crop_dist_px"] = {
        "body_median": float(np.median(dist[:, BODY])), "body_p95": float(np.percentile(dist[:, BODY], 95)),
        "feet_median": float(np.median(dist[:, [20, 21, 24, 25]])), "feet_p95": float(np.percentile(dist[:, [20, 21, 24, 25]], 95)),
        "hip_width_ff_median": float(np.median(np.abs(kff[:, 11, 0] - kff[:, 12, 0]))),
        "hip_width_crop_median": float(np.median(np.abs(kcr[:, 11, 0] - kcr[:, 12, 0]))),
        "knee_sep_ff_median": float(np.median(np.abs(kff[:, 13, 0] - kff[:, 14, 0]))),
        "knee_sep_crop_median": float(np.median(np.abs(kcr[:, 13, 0] - kcr[:, 14, 0]))),
    }
    # confidences
    raw_ff = np.minimum(ff_x.max(-1), ff_y.max(-1))
    raw_cr = np.minimum(cr_x.max(-1), cr_y.max(-1))
    pct = [0, 5, 50, 95, 100]
    RESULTS["conf_raw_ff_pct"] = np.percentile(raw_ff, pct).round(3).tolist()
    RESULTS["conf_raw_crop_pct"] = np.percentile(raw_cr, pct).round(3).tolist()
    RESULTS["conf_sigmoid_ff_pct"] = np.percentile(1 / (1 + np.exp(-raw_ff)), pct).round(3).tolist()
    RESULTS["conf_raw_ff_per_kpt_median"] = np.median(raw_ff, 0).round(3).tolist()
    RESULTS["conf_raw_crop_per_kpt_median"] = np.median(raw_cr, 0).round(3).tolist()

    # ------------------------------------------------------------ extra model runs
    est = RTMPoseEstimator(keypoint_format="halpe26", batch_size=1,
                           model_path=str(REPO / "src/biomechanics/pose/models/rtmpose-m-halpe26-256x192.onnx"))
    sess = ort.InferenceSession(str(est._model_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name

    def run_ff(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
        blob, sx, sy = est._preprocess(img)
        xl, yl = sess.run(None, {inp: blob})
        return xl[0], yl[0], sx, sy

    def run_crop(img: np.ndarray, c: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xl, yl = sess.run(None, {inp: normalize(crop_blob(img, c, s))})
        return xl[0], yl[0]

    cap = cv2.VideoCapture(str(REPO / "data/squats.mov"))
    frames = {}
    wanted = {0, 40, 75, 120, 160, 220, 280}
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i in wanted:
            frames[i] = make_canvas(fr)
        i += 1

    # sub-pixel shift linearity test on frame 75 (standing) and 40 (bottom)
    shift_tab = {}
    shifts = np.arange(0.0, 8.01, 0.25)
    for fidx in (75, 40):
        base = frames[fidx]
        c, s = centers[fidx], scales[fidx]
        for axis in (0, 1):
            preds = {cond: {n: [] for n in DECODERS} for cond in ("ff", "crop")}
            for sh in shifts:
                M = np.float64([[1, 0, sh if axis == 0 else 0], [0, 1, sh if axis == 1 else 0]])
                img = cv2.warpAffine(base, M, (base.shape[1], base.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                xl, yl, sx, sy = run_ff(img)
                cxl, cyl = run_crop(img, c, s)
                for n, fn in DECODERS.items():
                    lg = xl if axis == 0 else yl
                    sig = SIGMA_X if axis == 0 else SIGMA_Y
                    b = fn(lg[None], sig)[0]
                    preds["ff"][n].append(b / 2 * (sx if axis == 0 else sy))
                    clg = cxl if axis == 0 else cyl
                    cb = fn(clg[None], sig)[0]
                    size = IN_W if axis == 0 else IN_H
                    preds["crop"][n].append((cb / 2 - size / 2) * s[axis] / size + c[axis])
            for cond in preds:
                for n in DECODERS:
                    p = np.array(preds[cond][n])[:, BODY]  # (S, J)
                    disp = p - p[0]
                    err = disp - shifts[:, None]
                    # remove per-joint constant bias (model is not exactly equivariant) -> staircase residual
                    err_c = err - err.mean(0, keepdims=True)
                    slope = np.mean([np.polyfit(shifts, disp[:, j], 1)[0] for j in range(disp.shape[1])])
                    shift_tab[f"f{fidx}_{'x' if axis == 0 else 'y'}_{cond}_{n}"] = {
                        "rms_err_px": float(np.sqrt(np.mean(err ** 2))),
                        "rms_staircase_px": float(np.sqrt(np.mean(err_c ** 2))),
                        "mean_slope": float(slope),
                    }
    RESULTS["shift_test"] = shift_tab

    # flip-consistency (accuracy proxy): mirror image, predict, unmirror, compare with direct prediction
    flip_tab = {"ff": [], "crop": []}
    for fidx, img in frames.items():
        W = img.shape[1]
        flipped = img[:, ::-1].copy()
        xl, yl, sx, sy = run_ff(img)
        fxl, fyl, _, _ = run_ff(flipped)
        p = np.stack([dec_dark(xl[None], SIGMA_X)[0] / 2 * sx, dec_dark(yl[None], SIGMA_Y)[0] / 2 * sy], -1)
        pf = np.stack([dec_dark(fxl[None], SIGMA_X)[0] / 2 * sx, dec_dark(fyl[None], SIGMA_Y)[0] / 2 * sy], -1)
        pf[:, 0] = (W - 1) - pf[:, 0]
        pf = pf[HALPE_FLIP]
        flip_tab["ff"].append(np.linalg.norm(p - pf, axis=-1))
        c, s = centers[fidx], scales[fidx]
        cf = np.array([(W - 1) - c[0], c[1]])
        cxl, cyl = run_crop(img, c, s)
        fcxl, fcyl = run_crop(flipped, cf, s)
        def back(bx, by, cc):
            return np.stack([(bx / 2 - IN_W / 2) * s[0] / IN_W + cc[0], (by / 2 - IN_H / 2) * s[1] / IN_H + cc[1]], -1)
        q = back(dec_dark(cxl[None], SIGMA_X)[0], dec_dark(cyl[None], SIGMA_Y)[0], c)
        qf = back(dec_dark(fcxl[None], SIGMA_X)[0], dec_dark(fcyl[None], SIGMA_Y)[0], cf)
        qf[:, 0] = (W - 1) - qf[:, 0]
        qf = qf[HALPE_FLIP]
        flip_tab["crop"].append(np.linalg.norm(q - qf, axis=-1))
    RESULTS["flip_disagreement_px"] = {
        cond: {
            "body_median": float(np.median(np.array(v)[:, BODY])),
            "body_mean": float(np.mean(np.array(v)[:, BODY])),
            "feet_median": float(np.median(np.array(v)[:, [20, 21, 24, 25]])),
            "knees_ankles_mean": float(np.mean(np.array(v)[:, [13, 14, 15, 16]])),
        } for cond, v in flip_tab.items()
    }

    # occlusion test: grey box over left knee (halpe 13) on standing + bottom frames
    occ_tab = []
    for fidx in (75, 40, 160, 220):
        img = frames[fidx].copy()
        c, s = centers[fidx], scales[fidx]
        cxl, cyl = run_crop(img, c, s)
        q = np.stack([(dec_dark(cxl[None], SIGMA_X)[0] / 2 - IN_W / 2) * s[0] / IN_W + c[0],
                      (dec_dark(cyl[None], SIGMA_Y)[0] / 2 - IN_H / 2) * s[1] / IN_H + c[1]], -1)
        conf0 = np.minimum(cxl.max(-1), cyl.max(-1))
        kx, ky = q[13]
        half = 35
        occl = img.copy()
        occl[int(ky - half):int(ky + half), int(kx - half):int(kx + half)] = 127
        oxl, oyl = run_crop(occl, c, s)
        qo = np.stack([(dec_dark(oxl[None], SIGMA_X)[0] / 2 - IN_W / 2) * s[0] / IN_W + c[0],
                       (dec_dark(oyl[None], SIGMA_Y)[0] / 2 - IN_H / 2) * s[1] / IN_H + c[1]], -1)
        conf1 = np.minimum(oxl.max(-1), oyl.max(-1))
        blob, sx, sy = est._preprocess(occl)
        fxl, fyl = sess.run(None, {inp: blob})
        conf_ff1 = np.minimum(fxl[0].max(-1), fyl[0].max(-1))
        blob, sx, sy = est._preprocess(img)
        gxl, gyl = sess.run(None, {inp: blob})
        conf_ff0 = np.minimum(gxl[0].max(-1), gyl[0].max(-1))
        occ_tab.append({
            "frame": fidx,
            "crop_raw_conf_knee_before": float(conf0[13]), "crop_raw_conf_knee_occluded": float(conf1[13]),
            "crop_knee_shift_px": float(np.linalg.norm(qo[13] - q[13])),
            "crop_other_body_raw_conf_median": float(np.median(conf1[[5, 6, 11, 12, 14, 15, 16]])),
            "ff_raw_conf_knee_before": float(conf_ff0[13]), "ff_raw_conf_knee_occluded": float(conf_ff1[13]),
            "ff_sigmoid_knee_occluded": float(1 / (1 + np.exp(-conf_ff1[13]))),
            "occluder_px": 2 * half,
        })
        if fidx == 40:
            cv2.imwrite(str(OUT / "occlusion_frame40.jpg"), occl)
    RESULTS["occlusion_test"] = occ_tab

    # no-person images
    nop = {}
    bg = frames[0].copy()
    bg_patch = np.ascontiguousarray(bg[:, :400])  # left 400 px of canvas: replicated background only
    for name, img in {
        "black": np.zeros((720, 1280, 3), np.uint8),
        "noise": np.random.default_rng(0).integers(0, 255, (720, 1280, 3)).astype(np.uint8),
        "background_strip": cv2.resize(bg_patch, (1280, 720)),
    }.items():
        xl, yl, _, _ = run_ff(img)
        raw = np.minimum(xl.max(-1), yl.max(-1))
        nop[name] = {"raw_median": float(np.median(raw)), "raw_max": float(raw.max()),
                     "sigmoid_median": float(np.median(1 / (1 + np.exp(-raw)))), "sigmoid_max": float(1 / (1 + np.exp(-raw.max()))),
                     "n_kpts_pass_sigmoid_0.3": int(np.sum(1 / (1 + np.exp(-raw)) >= 0.3)),
                     "n_kpts_pass_raw_0.3": int(np.sum(raw >= 0.3))}
    RESULTS["no_person"] = nop

    # decode cost
    import time
    t0 = time.perf_counter()
    for _ in range(200):
        dec_argmax(ff_x[:3]); dec_argmax(ff_y[:3])
    t_arg = (time.perf_counter() - t0) / 200
    t0 = time.perf_counter()
    for _ in range(200):
        dec_parabola(ff_x[:3]); dec_parabola(ff_y[:3])
    t_par = (time.perf_counter() - t0) / 200
    t0 = time.perf_counter()
    for _ in range(200):
        dec_softargmax(ff_x[:3]); dec_softargmax(ff_y[:3])
    t_soft = (time.perf_counter() - t0) / 200
    t0 = time.perf_counter()
    for _ in range(20):
        dec_dark(ff_x[:3], SIGMA_X); dec_dark(ff_y[:3], SIGMA_Y)
    t_dark = (time.perf_counter() - t0) / 20
    RESULTS["decode_cost_ms_per_3views"] = {"argmax": t_arg * 1e3, "parabola": t_par * 1e3, "softargmax": t_soft * 1e3, "dark_python_loop": t_dark * 1e3}

    # crop warp cost vs resize
    img = frames[75]
    t0 = time.perf_counter()
    for _ in range(200):
        est._preprocess(img)
    t_ff = (time.perf_counter() - t0) / 200
    t0 = time.perf_counter()
    for _ in range(200):
        normalize(crop_blob(img, centers[75], scales[75]))
    t_cr = (time.perf_counter() - t0) / 200
    RESULTS["preprocess_cost_ms_per_view"] = {"ff_resize": t_ff * 1e3, "crop_warpaffine": t_cr * 1e3}

    (OUT / "exp1_results.json").write_text(json.dumps(RESULTS, indent=1))
    print(json.dumps(RESULTS, indent=1))


if __name__ == "__main__":
    main()
