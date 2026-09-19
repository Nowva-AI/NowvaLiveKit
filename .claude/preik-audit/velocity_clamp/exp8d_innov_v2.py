"""Clean innovation stats of the v2 alpha-beta predictor (0.85/0.5), world vs hip-centred, 30 Hz vs 15 Hz; gate cost."""
import sys, json, time
sys.path.insert(0, ".")
import numpy as np
from synth_vc import *
from tri_world import world_sequence
from gate_proto2 import InnovationGate2

calib, cams = make_rig()
res = {}
LOWER = [CK.LEFT_KNEE, CK.RIGHT_KNEE, CK.LEFT_ANKLE, CK.RIGHT_ANKLE, CK.LEFT_FOOT_INDEX, CK.RIGHT_FOOT_INDEX]
for frame_step, label in ((1, "30Hz"), (2, "15Hz")):
    for sigma in (3.0, 5.0):
        inn_w, inn_h = [], []
        for arms, tempo in (("bar", "bw_normal"), ("bar", "bw_fast"), ("bar", "bw_explosive"), ("bw_arms", "bw_fast"), ("bar", "loaded_heavy")):
            t, fr = sequence_world(tempo, fps=30, n_reps=3, arms=arms, stand_s=0.6)
            idx = np.arange(0, len(t), frame_step)
            w = world_sequence(t, fr, cams, seed=41, sigma_px=sigma)
            h = triangulate_sequence(t, fr, calib, cams, seed=41, sigma_px=sigma)
            for data, store in ((np.array([w[i][0] for i in idx]), inn_w), (np.array([h[i].to_numpy() for i in idx]), inn_h)):
                x = data[0].copy(); v = np.zeros_like(x)
                for k in range(1, len(idx)):
                    dt = t[idx[k]] - t[idx[k - 1]]
                    pred = x + v * dt
                    r = data[k] - pred
                    store.append(np.linalg.norm(r, axis=1))
                    x = pred + 0.85 * r; v = v + 0.5 * r / dt
        for name, arr in (("world", np.array(inn_w) * 100), ("hipcentred", np.array(inn_h) * 100)):
            m = dict(all_p999=round(float(np.percentile(arr, 99.9)), 2), all_max=round(float(arr.max()), 2),
                     ankles_toes_p999=round(float(np.percentile(arr[:, LOWER[2:]], 99.9)), 2), knees_p999=round(float(np.percentile(arr[:, LOWER[:2]], 99.9)), 2),
                     wrists_p999=round(float(np.percentile(arr[:, [CK.LEFT_WRIST, CK.RIGHT_WRIST]], 99.9)), 2))
            res[f"{label}/s{sigma}/{name}"] = m
            print(f"{label} sigma{sigma} {name:10s} innovation cm: {m}")
g = InnovationGate2()
z = np.random.rand(19, 3); c = np.full(19, 0.7)
t0 = time.perf_counter()
for i in range(3000):
    g.update(z + 0.001 * i, c, i / 30)
res["gate_v2_numpy_ms_per_frame"] = round((time.perf_counter() - t0) / 3000 * 1000, 4)
print("gate v2 numpy cost ms/frame:", res["gate_v2_numpy_ms_per_frame"])
json.dump(res, open("exp8d_innov_v2.json", "w"), indent=1)
