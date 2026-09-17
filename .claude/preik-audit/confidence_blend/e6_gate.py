"""E6: (a) reprojection-threshold detection table, (b) prototype KeypointGate vs production blender (+clamp) on a squat
session with 2D outliers, 3-view and 2-view, 30 and 11.6 fps, (c) duplicate frames / NaN / seeding behaviour of the gate,
(d) per-frame cost.
"""
from __future__ import annotations

import json
import time

import numpy as np

import common as C
import kin_snapshot as K
from keypoint_gate_proto import KeypointGate

CK = C.CK
LEG = [CK.LEFT_HIP, CK.RIGHT_HIP, CK.LEFT_KNEE, CK.RIGHT_KNEE, CK.LEFT_ANKLE, CK.RIGHT_ANKLE, 17, 18]
res: dict = {}


def tri(world, rng, P, sigma, op):
    uv = C.synth.project(P, world)
    uvn = uv + rng.normal(0, sigma, uv.shape)
    m = rng.random(uv.shape[:-1]) < op
    uvn += m[..., None] * rng.normal(0, 40.0, uv.shape)
    uvn[..., 0] = np.round(uvn[..., 0] / C.synth.QUANT_X_PX) * C.synth.QUANT_X_PX
    uvn[..., 1] = np.round(uvn[..., 1] / C.synth.QUANT_Y_PX) * C.synth.QUANT_Y_PX
    X = C.synth.dlt(P, uvn)
    reproj = np.linalg.norm(C.synth.project(P, X) - uvn, axis=-1).mean(-1)
    base = C.view_conf(world.shape[0], rng, P.shape[0]).min(-1)
    conf = np.clip(base * np.where(reproj < 15, 1 - reproj / 15, 0.1), 0, 1)
    return C.synth.recenter(X), conf, reproj, X


P3 = C.synth.cameras()

def run_gate(pts, conf, rep, ts, thr):
    g = KeypointGate(19, max_reproj_px=thr)
    out = np.empty_like(pts)
    for t in range(len(pts)):
        out[t], _ = g.update(pts[t], conf[t], rep[t], float(ts[t]))
    return out

def m(out, true, windows, fps):
    el = 100 * np.linalg.norm(out[:, LEG] - true[:, LEG], axis=-1)
    kt = np.stack([K.knee_flexion(true, "l"), K.knee_flexion(true, "r")]); ko = np.stack([K.knee_flexion(out, "l"), K.knee_flexion(out, "r")])
    ke = ko - kt
    kvel = np.gradient(kt, axis=1) * fps
    peaks = [round(float(ko[0, st:en + 1].max() - kt[0, st:en + 1].max()), 1) for st, b0, b1, en in windows]
    return dict(leg_err_rms_cm=round(float(np.sqrt((el ** 2).mean())), 2), leg_err_p99_cm=round(float(np.percentile(el, 99)), 1),
                knee_rmse=round(float(np.sqrt((ke ** 2).mean())), 2), knee_fast_abs=round(float(np.abs(ke[np.abs(kvel) > 100]).mean()), 2),
                frames_knee_err_gt_10deg=int((np.abs(ke) > 10).sum()), knee_err_p99=round(float(np.percentile(np.abs(ke), 99)), 1),
                peak_knee_l_err=peaks)


def main() -> None:
    # ---------------------------------------------------------------- (a) threshold table
    rng = np.random.default_rng(31)
    world, s, phase, windows = C.synth.session(fps=30.0)
    world4 = np.concatenate([world] * 3)
    table = {}
    for label, P in [("3view", P3), ("2view", P3[[0, 2]])]:
        for sigma in [3.0, 6.0]:
            _, conf, rep, X = tri(world4, rng, P, sigma, 0.03)
            e = 100 * np.linalg.norm(X - world4, axis=-1)[:, LEG].ravel(); r = rep[:, LEG].ravel()
            rows = []
            for thr in [4, 6, 8, 10, 12, 15]:
                flag = r > thr
                rows.append(dict(thr_px=thr, flagged=round(float(flag.mean()), 3), caught_err_gt_5cm=round(float(flag[e > 5].mean()), 3),
                                 caught_err_gt_10cm=round(float(flag[e > 10].mean()), 3), false_flag_err_lt_2cm=round(float(flag[e < 2].mean()), 3)))
            table[f"{label}_sigma{sigma:g}"] = dict(frac_err_gt_5cm=round(float((e > 5).mean()), 3), rows=rows)
    res["a_reproj_threshold_table"] = table


    # ---------------------------------------------------------------- (b) session comparison
    cmp = {}
    for fps in [30.0, 11.6]:
        world, s, phase, windows = C.synth.session(fps=fps)
        ts = np.arange(world.shape[0]) / fps
        true = C.synth.recenter(world)
        for label, P in [("3view", P3), ("2view", P3[[0, 2]])]:
            rng = np.random.default_rng(41)
            pts, conf, rep, _ = tri(world, rng, P, 3.0, 0.03)
            variants = {
                "raw": pts,
                "blender(prod)": C.run_blender(pts, conf, ts),
                "blend+clamp(prod)": C.run_blend_clamp(pts, conf, ts),
                "gate_thr8px": run_gate(pts, conf, rep, ts, 8.0),
                "gate_thr8px+clamp": C.run_blend_clamp(run_gate(pts, conf, rep, ts, 8.0), np.full(conf.shape, 0.9), ts, use_blend=False),
            }
            cmp[f"fps{fps:g}_{label}"] = {k: m(v, true, windows, fps) for k, v in variants.items()}
    res["b_session_with_3pct_40px_outliers"] = cmp

    # ---------------------------------------------------------------- (c) gate edge cases
    std = C.synth.recenter(C.synth.pose(0.0, C.synth.body())); half = C.synth.recenter(C.synth.pose(0.5, C.synth.body()))
    g = KeypointGate(19)
    p0 = std.copy(); c0 = np.full(19, 0.5); c0[17] = 0.0; p0[17] = 0.0
    o, oc = g.update(p0, c0, np.zeros(19), 0.0)
    first = dict(toe_out=o[17].round(3).tolist(), toe_conf=float(oc[17]))
    o, oc = g.update(std, np.full(19, 0.5), np.zeros(19), 1 / 30)
    first["toe_err_cm_next_frame"] = round(float(100 * np.linalg.norm(o[17] - std[17])), 2)
    g = KeypointGate(19)
    g.update(std, np.full(19, 0.5), np.zeros(19), 0.0)
    pn = std.copy(); pn[CK.LEFT_KNEE] = np.nan
    o, oc = g.update(pn, np.full(19, 0.5), np.zeros(19), 1 / 30)
    o2, _ = g.update(std, np.full(19, 0.5), np.zeros(19), 2 / 30)
    nan_case = dict(out_finite_on_nan_frame=bool(np.isfinite(o).all()), conf_on_nan_frame=float(oc[CK.LEFT_KNEE]),
                    out_after=bool(np.allclose(o2, std)))
    g = KeypointGate(19)
    g.update(std, np.full(19, 0.5), np.zeros(19), 0.0)
    o_dup, _ = g.update(std, np.full(19, 0.5), np.zeros(19), 0.0)
    o_ret, oc_ret = g.update(half, np.full(19, 0.5), np.zeros(19), 10.0)
    res["c_gate_edge_cases"] = dict(first_frame_untriangulated_toe=first, nan_frame=nan_case,
                                    duplicate_frame_identity=bool(np.allclose(o_dup, std)),
                                    return_after_10s_err_cm=round(float(100 * np.abs(o_ret - half).max()), 3))

    # ---------------------------------------------------------------- (d) cost
    g = KeypointGate(19); pts1 = std + 0.0; conf1 = np.full(19, 0.5); rep1 = np.full(19, 2.0)
    t0 = time.perf_counter()
    for i in range(5000):
        g.update(pts1, conf1, rep1, i / 30)
    gate_us = (time.perf_counter() - t0) / 5000 * 1e6
    from biomechanics.utils.confidence_blend import ConfidenceBlender  # noqa: E402
    from biomechanics.utils.types import Skeleton3D  # noqa: E402
    b = ConfidenceBlender(); sk = Skeleton3D.from_numpy(std, confidences=conf1)
    t0 = time.perf_counter()
    for i in range(2000):
        b.blend(sk)
    blend_us = (time.perf_counter() - t0) / 2000 * 1e6
    res["d_cost_us_per_frame_mac"] = dict(gate_proto_numpy=round(gate_us, 1), production_blender_incl_pydantic=round(blend_us, 1))

    (C.HERE / "e6_results.json").write_text(json.dumps(res, indent=1))
    for k, v in res["a_reproj_threshold_table"].items():
        print("==", k, "frac>5cm", v["frac_err_gt_5cm"])
        for r in v["rows"]:
            print("   ", r)
    for k, v in cmp.items():
        print("==", k)
        for name, mm in v.items():
            print(f"   {name:20s} {mm}")
    print(json.dumps(res["c_gate_edge_cases"], indent=1)); print(res["d_cost_us_per_frame_mac"])


if __name__ == "__main__":
    main()
