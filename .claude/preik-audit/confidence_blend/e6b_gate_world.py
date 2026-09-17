"""E6b: gate in the triangulator's world frame BEFORE hip re-centering (+ optional equal-noise One Euro), vs production
blender; plus ankle-lost-1s scenario with world-frame hold."""
from __future__ import annotations

import json

import numpy as np

import common as C
import filters_alt as F
import kin_snapshot as K
from e6_gate import P3, m, run_gate, tri
from keypoint_gate_proto import KeypointGate

CK = C.CK
res: dict = {}
OE = {30.0: (4.0, 8.0), 11.6: (1.0, 8.0)}  # equal static-noise One Euro params from E2
for fps in [30.0, 11.6]:
    world, s, phase, windows = C.synth.session(fps=fps)
    ts = np.arange(world.shape[0]) / fps
    true = C.synth.recenter(world)
    for label, P, op in [("3view_outliers", P3, 0.03), ("3view_clean", P3, 0.0), ("2view_outliers", P3[[0, 2]], 0.03)]:
        rng = np.random.default_rng(41)
        pts, conf, rep, X = tri(world, rng, P, 3.0, op)
        gated_world = run_gate(X, conf, rep, ts, 8.0)
        gw = C.synth.recenter(gated_world)
        mc, beta = OE[fps]
        variants = {
            "raw": pts,
            "blend+clamp(prod)": C.run_blend_clamp(pts, conf, ts),
            "gate_after_recentre": run_gate(pts, conf, rep, ts, 8.0),
            "gate_world_then_recentre": gw,
            "gate_world+one_euro_equal_noise": F.one_euro(gw, ts, mc, beta),
        }
        res[f"fps{fps:g}_{label}"] = {k: m(v, true, windows, fps) for k, v in variants.items()}

# ankle lost 1 s during descent (reproj >= 15 px -> invalid), world-frame gate vs blender
fps = 30.0
world, s, phase, windows = C.synth.session(fps=fps)
ts = np.arange(world.shape[0]) / fps
rng = np.random.default_rng(5)
pts, conf, rep, X = tri(world, rng, P3, 3.0, 0.0)
true = C.synth.recenter(world)
st = windows[0][0]
lost = slice(st + 2, st + 32)
Xb = X.copy(); Xb[lost, CK.LEFT_ANKLE] += rng.normal(0, 0.10, (30, 3))
repb = rep.copy(); repb[lost, CK.LEFT_ANKLE] = 18.0
confb = conf.copy(); confb[lost, CK.LEFT_ANKLE] = 0.06
ptsb = C.synth.recenter(Xb)
g = KeypointGate(19, 8.0, hold_max_s=0.1)
gw = np.empty_like(Xb); gc = np.empty(confb.shape)
for t in range(len(Xb)):
    gw[t], gc[t] = g.update(Xb[t], confb[t], repb[t], float(ts[t]))
g2 = KeypointGate(19, 8.0, hold_max_s=1.0)
gw2 = np.empty_like(Xb)
for t in range(len(Xb)):
    gw2[t], _ = g2.update(Xb[t], confb[t], repb[t], float(ts[t]))
bl = C.run_blend_clamp(ptsb, confb, ts)
def err(o: np.ndarray, recentre: bool = True) -> np.ndarray:
    oo = C.synth.recenter(o) if recentre else o
    return 100 * np.linalg.norm(oo[:, CK.LEFT_ANKLE] - true[:, CK.LEFT_ANKLE], axis=-1)

ret = lost.stop
res["ankle_lost_1s"] = dict(
    during_mean_err_cm=dict(blend_clamp_prod=round(float(err(bl, False)[lost].mean()), 1), gate_world_hold0p1=round(float(err(gw)[lost].mean()), 1),
                            gate_world_hold1p0=round(float(err(gw2)[lost].mean()), 1)),
    gate_hold0p1_conf_reported_during=[float(x) for x in gc[lost, CK.LEFT_ANKLE][[0, 2, 3, 4, 29]]],
    after_return_err_cm=dict(blend_clamp_prod=[round(float(x), 1) for x in err(bl, False)[ret:ret + 6]],
                             gate_world=[round(float(x), 1) for x in err(gw)[ret:ret + 6]]),
)
(C.HERE / "e6b_results.json").write_text(json.dumps(res, indent=1))
for k, v in res.items():
    print("==", k)
    if isinstance(v, dict) and "raw" in v:
        for name, mm in v.items():
            print(f"   {name:34s} {mm}")
    else:
        print("  ", v)
