"""Evaluate gate v2 (hip-centred drop-in and world-frame-before-recentring) vs production blend+clamp."""
import sys, json, time
sys.path.insert(0, ".")
import numpy as np
from synth_vc import *
from tri_world import world_sequence, recentre
from gate_proto2 import InnovationGate2
from biomechanics.utils.confidence_blend import ConfidenceBlender
from biomechanics.utils.velocity_clamp import VelocityClamp
from biomechanics.utils.geometry import joint_angle_3_points

calib, cams = make_rig()
KNEE = CK.LEFT_KNEE
GRID = [(0.06, 40.0), (0.08, 40.0), (0.08, 80.0), (0.10, 80.0)]


def kang(a):
    return 180 - joint_angle_3_points(a[CK.LEFT_HIP], a[KNEE], a[CK.LEFT_ANKLE])


def run_hc(sks, variant, r0=0.08, amax=40.0):
    bl, cl = ConfidenceBlender(0.1, 0.9), VelocityClamp(2.5, 30)
    g = InnovationGate2(r_noise_m=r0, a_max_m_s2=amax, hip_centred_common_mode=True)
    outs, confs_ = [], []
    for sk in sks:
        if variant in ("blend", "blend+clamp"):
            sk = bl.blend(sk)
        if variant == "blend+clamp":
            sk = cl.clamp(sk)
        z = sk.to_numpy(); c = confs(sk)
        if variant == "gateHC":
            z, c, _ = g.update(z, c, sk.timestamp)
        outs.append(z); confs_.append(c)
    return np.array(outs), np.array(confs_), g.stats


def run_world(wseq, times, r0=0.08, amax=40.0):
    g = InnovationGate2(r_noise_m=r0, a_max_m_s2=amax)
    outs, cs = [], []
    for (pts, c), ts in zip(wseq, times):
        o, oc, _ = g.update(pts, c, float(ts))
        outs.append(recentre(o, oc)); cs.append(oc)
    return np.array(outs), np.array(cs), g.stats


def clean_metrics(arr, cf, truth, st):
    e = np.linalg.norm(arr - truth, axis=2) * 100
    ka = np.array([kang(a) for a in arr]); kt = np.array([kang(a) for a in truth])
    return dict(rms=round(float(np.sqrt((e ** 2).mean())), 2), p99=round(float(np.percentile(e, 99)), 1), wrist_max=round(float(e[:, CK.LEFT_WRIST].max()), 1),
                knee_rms=round(float(np.sqrt(((ka - kt) ** 2).mean())), 2), knee_max=round(float(np.abs(ka - kt).max()), 1),
                rej=(st or {}).get("rejected"), cm=(st or {}).get("common_mode"))


res = {}
print("=== CLEAN (3 reps) vs ground truth: rms/p99 cm all kpts | wrist max cm | knee angle rms/max deg | rejections")
for arms, tempo in (("bar", "loaded_heavy"), ("bar", "bw_normal"), ("bar", "bw_fast"), ("bar", "bw_explosive"), ("bw_arms", "bw_fast"), ("bw_arms", "bw_explosive")):
    for sigma in (3.0, 5.0):
        t, fr = sequence_world(tempo, fps=30, n_reps=3, arms=arms, stand_s=0.6)
        truth = hip_center(fr)
        sks = triangulate_sequence(t, fr, calib, cams, seed=31, sigma_px=sigma)
        wseq = world_sequence(t, fr, cams, seed=31, sigma_px=sigma)
        rows = {}
        for v in ("none", "blend", "blend+clamp"):
            a, c, _ = run_hc(sks, v); rows[v] = clean_metrics(a, c, truth, None)
        for r0, amax in GRID:
            a, c, st = run_hc(sks, "gateHC", r0, amax); rows[f"gateHC_{r0}_{amax}"] = clean_metrics(a, c, truth, st)
            a, c, st = run_world(wseq, t, r0, amax); rows[f"gateW_{r0}_{amax}"] = clean_metrics(a, c, truth, st)
        for k, m in rows.items():
            res[f"clean/{arms}/{tempo}/s{sigma}/{k}"] = m
        print(f"\n{arms}/{tempo}/sigma{sigma}")
        for k, m in rows.items():
            print(f"   {k:20s} rms {m['rms']:6.2f} p99 {m['p99']:6.1f} wrist {m['wrist_max']:6.1f} knee {m['knee_rms']:5.2f}/{m['knee_max']:5.1f}" + (f" rej {m['rej']} cm {m['cm']}" if m['rej'] is not None else ""))
json.dump(res, open("exp8b_clean.json", "w"), indent=1)
