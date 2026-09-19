"""Peak accelerations (hip-centred) and clean-data innovation statistics vs a constant-velocity prediction."""
import sys, json
sys.path.insert(0, ".")
import numpy as np
from synth_vc import *

res = {}
for arms in ("bar", "bw_arms"):
    for tempo in TEMPOS:
        t, fr = sequence_world(tempo, fps=600, arms=arms)
        hc = hip_center(fr)
        acc = np.linalg.norm(np.gradient(np.gradient(hc, t, axis=0), t, axis=0), axis=2)
        pk = {n: round(float(acc[:, i].max()), 1) for n, i in (("knee", CK.LEFT_KNEE), ("ankle", CK.LEFT_ANKLE), ("wrist", CK.LEFT_WRIST), ("nose", CK.NOSE))}
        res[f"acc/{arms}/{tempo}"] = pk
        print(f"peak |acc| m/s^2 hip-centred {arms:8s} {tempo:18s} {pk}")

calib, cams = make_rig()
print("\nclean innovation |z - (p + v dt)| cm, v = EMA(beta) of accepted differences (99.9 pct / max), all 19 kpts")
for beta in (1.0, 0.5, 0.3):
    for sigma in (3.0, 5.0):
        allv = []
        for arms, tempo in (("bar", "bw_normal"), ("bar", "bw_fast"), ("bar", "bw_explosive"), ("bw_arms", "bw_fast")):
            t, fr = sequence_world(tempo, fps=30, n_reps=3, arms=arms, stand_s=0.6)
            sks = triangulate_sequence(t, fr, calib, cams, seed=21, sigma_px=sigma)
            z = np.array([s.to_numpy() for s in sks])
            p = z[0].copy(); v = np.zeros_like(p)
            inn = []
            for k in range(1, len(z)):
                dt = t[k] - t[k - 1]
                pred = p + v * dt
                inn.append(np.linalg.norm(z[k] - pred, axis=1))
                v = beta * (z[k] - p) / dt + (1 - beta) * v
                p = z[k]
            inn = np.array(inn) * 100
            res[f"innov/beta{beta}/s{sigma}/{arms}/{tempo}"] = dict(p999=round(float(np.percentile(inn, 99.9)), 2), max=round(float(inn.max()), 2))
            allv.append(inn)
            print(f"  beta={beta} sigma={sigma} {arms:8s} {tempo:14s} p99.9={np.percentile(inn, 99.9):5.2f} max={inn.max():5.2f}")
json.dump(res, open("exp8a_accel_innov.json", "w"), indent=1)
