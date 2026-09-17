"""Evaluate prototype InnovationGate vs production blend+clamp on clean data and outlier scenarios."""
import sys, json
sys.path.insert(0, ".")
import numpy as np
from synth_vc import *
from gate_proto import InnovationGate
from biomechanics.utils.confidence_blend import ConfidenceBlender
from biomechanics.utils.velocity_clamp import VelocityClamp
from biomechanics.utils.geometry import joint_angle_3_points

calib, cams = make_rig()
KNEE = CK.LEFT_KNEE
VARS = ("none", "blend", "blend+clamp", "gate6", "gate8")


def run(sks, variant):
    bl, cl = ConfidenceBlender(0.1, 0.9), VelocityClamp(2.5, 30)
    g = InnovationGate(gate_noise_m=0.06 if variant == "gate6" else 0.08)
    out = []
    for sk in sks:
        if variant in ("blend", "blend+clamp"):
            sk = bl.blend(sk)
        if variant == "blend+clamp":
            sk = cl.clamp(sk)
        if variant.startswith("gate"):
            sk, _ = g.update(sk)
        out.append(sk)
    return out, (g.stats if variant.startswith("gate") else None)


def kang(a):
    return 180 - joint_angle_3_points(a[CK.LEFT_HIP], a[KNEE], a[CK.LEFT_ANKLE])


def rel_metrics(out, ref, lo, hi):
    e = np.array([np.linalg.norm(out[k].to_numpy()[KNEE] - ref[k].to_numpy()[KNEE]) * 100 for k in range(lo, hi)])
    ae = np.array([abs(kang(out[k].to_numpy()) - kang(ref[k].to_numpy())) for k in range(lo, hi)])
    mc = np.array([min(kp.confidence for kp in (out[k].keypoints[CK.LEFT_HIP], out[k].keypoints[KNEE], out[k].keypoints[CK.LEFT_ANKLE])) for k in range(lo, hi)])
    return dict(max_cm=round(float(e.max()), 1), f5cm=int((e > 5).sum()), max_deg=round(float(ae.max()), 1), f5deg=int((ae > 5).sum()),
                ik_zeroed_frames=int((mc < 0.1).sum()), trace=[round(float(x), 1) for x in e[:16]])


res = {}
print("=== CLEAN DATA (3 reps): error vs ground truth, all kpts / knee angle")
for arms, tempo in (("bar", "loaded_heavy"), ("bar", "bw_normal"), ("bar", "bw_fast"), ("bar", "bw_explosive"), ("bw_arms", "bw_fast")):
    for sigma in (3.0, 5.0):
        t, fr = sequence_world(tempo, fps=30, n_reps=3, arms=arms, stand_s=0.6)
        truth = hip_center(fr)
        sks = triangulate_sequence(t, fr, calib, cams, seed=31, sigma_px=sigma)
        line = f"{arms:7s} {tempo:13s} s{sigma}: "
        for v in VARS:
            out, st = run(sks, v)
            arr = np.array([o.to_numpy() for o in out])
            e = np.linalg.norm(arr - truth, axis=2) * 100
            ka = np.array([kang(a) for a in arr]); kt = np.array([kang(a) for a in truth])
            wr = e[:, CK.LEFT_WRIST]
            r = dict(rms_cm=round(float(np.sqrt((e ** 2).mean())), 2), p99_cm=round(float(np.percentile(e, 99)), 2), wrist_max_cm=round(float(wr.max()), 1),
                     knee_rms_deg=round(float(np.sqrt(((ka - kt) ** 2).mean())), 2), knee_max_deg=round(float(np.abs(ka - kt).max()), 1),
                     rejected=(st or {}).get("rejected"))
            res[f"clean/{arms}/{tempo}/s{sigma}/{v}"] = r
            line += f"| {v}: rms {r['rms_cm']:.2f} p99 {r['p99_cm']:.1f} wristMax {r['wrist_max_cm']:.1f} kneeRMS {r['knee_rms_deg']:.2f} max {r['knee_max_deg']:.1f}" + (f" rej {r['rejected']}" if st else "") + " "
        print(line)

print("\n=== OUTLIER SCENARIOS (bw_normal, sigma 3 px): knee error vs uncorrupted triangulation")
t, fr = sequence_world("bw_normal", fps=30, stand_s=0.6)
base = triangulate_sequence(t, fr, calib, cams, seed=5, sigma_px=3.0)
s = depth_profile(t, "bw_normal", 0.6, 1)
mid = int(np.argmin(np.abs(s[: len(s) // 2] - 0.5)))


def rep3d(sk, arr):
    return Skeleton3D.from_numpy(arr, confidences=confs(sk), timestamp=sk.timestamp, frame_index=0)


legs = [(CK.LEFT_KNEE, CK.RIGHT_KNEE), (CK.LEFT_ANKLE, CK.RIGHT_ANKLE), (CK.LEFT_FOOT_INDEX, CK.RIGHT_FOOT_INDEX)]
def swap(uv, c):
    for l, r in legs:
        uv[[l, r]] = uv[[r, l]]
    return uv, c
def occl0(uv, c):
    c[KNEE] = 0.0; return uv, c

for where, k0 in (("standing", 8), ("mid_descent", mid)):
    lo, hi = k0 - 1, k0 + 22
    scen = {}
    for d in (20, 45, 60):
        sk = list(base); a = sk[k0].to_numpy(); a[KNEE] += [d / 100, 0, 0]; sk[k0] = rep3d(sk[k0], a); scen[f"S1_single3d_{d}cm"] = (sk, base)
    sk = list(base)
    for k in range(k0, k0 + 10):
        a = sk[k].to_numpy(); a[KNEE] += [0.30, 0, 0]; sk[k] = rep3d(sk[k], a)
    scen["S1b_persist10_30cm"] = (sk, base)
    sk = list(base); a = sk[k0].to_numpy(); a[CK.LEFT_HIP] += [0, 0, 0.30]; a -= [0, 0, 0.15]; sk[k0] = rep3d(sk[k0], a); scen["S5_hip_30cm"] = (sk, base)
    for px in (80,):
        def bad(uv, c, px=px):
            uv[KNEE] += [px, 0.0]; return uv, c
        scen[f"S2_1view_{px}px"] = (triangulate_sequence(t, fr, calib, cams, seed=5, sigma_px=3.0, corrupt_at={k0: {"1": bad}}), base)
        base2 = triangulate_sequence(t, fr, calib, cams, seed=5, sigma_px=3.0, corrupt_at={k: {"0": occl0} for k in range(len(t))})
        cor = {k: {"0": occl0} for k in range(len(t))}; cor[k0] = {"0": occl0, "1": bad}
        scen[f"S2b_2viewonly_{px}px"] = (triangulate_sequence(t, fr, calib, cams, seed=5, sigma_px=3.0, corrupt_at=cor), base2)
    scen["S3_legswap_1view_10f"] = (triangulate_sequence(t, fr, calib, cams, seed=5, sigma_px=3.0, corrupt_at={k: {"1": swap} for k in range(k0, k0 + 10)}), base)
    for name, (sk, ref) in scen.items():
        print(f"\n{where}/{name}")
        for v in VARS:
            out, st = run(sk, v)
            m = rel_metrics(out, ref, lo, hi)
            res[f"outlier/{where}/{name}/{v}"] = m
            print(f"   {v:12s} max {m['max_cm']:5.1f} cm >5cm {m['f5cm']:2d}f | knee max {m['max_deg']:5.1f} >5deg {m['f5deg']:2d}f | IK-zeroed {m['ik_zeroed_frames']:2d}f | {m['trace'][:14]}")

# first frame conf-0 seed
t2, fr2 = sequence_world("bw_normal", fps=30, stand_s=1.5)
def occl_tw(uv, c):
    c[[CK.LEFT_FOOT_INDEX, CK.LEFT_WRIST]] = 0.0; return uv, c
sk = triangulate_sequence(t2, fr2, calib, cams, seed=4, sigma_px=3.0, corrupt_at={5: {"0": occl_tw, "1": occl_tw, "2": occl_tw}})[5:30]
ref = triangulate_sequence(t2, fr2, calib, cams, seed=4, sigma_px=3.0)[5:30]
print("\nfirst frame after reset, toe conf 0 at origin: toe error trace cm")
for v in VARS:
    out, _ = run(sk, v)
    e = [round(float(np.linalg.norm(out[k].to_numpy()[CK.LEFT_FOOT_INDEX] - ref[k].to_numpy()[CK.LEFT_FOOT_INDEX]) * 100), 1) for k in range(14)]
    res[f"firstframe/{v}"] = e
    print(f"   {v:12s} {e}")

# cost
import time
g = InnovationGate(); sks = base * 20
t0 = time.perf_counter()
for i, sk_ in enumerate(sks):
    g.update(Skeleton3D(keypoints=sk_.keypoints, timestamp=i / 30, frame_index=0))
res["cost_ms_per_frame_incl_pydantic"] = round((time.perf_counter() - t0) * 1000 / len(sks), 3)
print("\ngate cost ms/frame (python+pydantic, M-series):", res["cost_ms_per_frame_incl_pydantic"])
json.dump(res, open("exp8_gate_eval.json", "w"), indent=1)
