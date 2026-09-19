"""Outlier scenarios for gate v2 (world and hip-centred) vs production. Errors vs GROUND TRUTH (hip-centred)."""
import sys, json
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
tempo = "bw_normal"
t, fr = sequence_world(tempo, fps=30, stand_s=0.6)
truth = hip_center(fr)
s = depth_profile(t, tempo, 0.6, 1)
mid = int(np.argmin(np.abs(s[: len(s) // 2] - 0.5)))
IK_MIN = 0.1


def kang(a):
    return 180 - joint_angle_3_points(a[CK.LEFT_HIP], a[KNEE], a[CK.LEFT_ANKLE])


def prod(sks, variant):
    bl, cl = ConfidenceBlender(0.1, 0.9), VelocityClamp(2.5, 30)
    o, c = [], []
    for sk in sks:
        if variant != "none":
            sk = bl.blend(sk)
        if variant == "blend+clamp":
            sk = cl.clamp(sk)
        o.append(sk.to_numpy()); c.append(confs(sk))
    return np.array(o), np.array(c)


def gate_hc(sks):
    g = InnovationGate2(r_noise_m=0.10, a_max_m_s2=80.0, hip_centred_common_mode=True)
    o, c = [], []
    for sk in sks:
        z, cc, _ = g.update(sk.to_numpy(), confs(sk), sk.timestamp); o.append(z); c.append(cc)
    return np.array(o), np.array(c)


def gate_w(wseq):
    g = InnovationGate2(r_noise_m=0.08, a_max_m_s2=80.0)
    o, c = [], []
    for (pts, cf), ts in zip(wseq, t):
        z, cc, _ = g.update(pts, cf, float(ts)); o.append(recentre(z, cc)); c.append(cc)
    return np.array(o), np.array(c)


def metrics(arr, cf, lo, hi):
    e = np.linalg.norm(arr[lo:hi, KNEE] - truth[lo:hi, KNEE], axis=1) * 100
    ae = np.abs(np.array([kang(arr[k]) - kang(truth[k]) for k in range(lo, hi)]))
    ikz = int((cf[lo:hi, [CK.LEFT_HIP, KNEE, CK.LEFT_ANKLE]].min(axis=1) < IK_MIN).sum())
    return dict(max_cm=round(float(e.max()), 1), f5cm=int((e > 5).sum()), max_deg=round(float(ae.max()), 1), f8deg=int((ae > 8).sum()), ik_zeroed=ikz,
                trace=[round(float(x), 1) for x in e[:15]])


legs = [(CK.LEFT_KNEE, CK.RIGHT_KNEE), (CK.LEFT_ANKLE, CK.RIGHT_ANKLE), (CK.LEFT_FOOT_INDEX, CK.RIGHT_FOOT_INDEX)]
def swap(uv, c):
    for l, r in legs:
        uv[[l, r]] = uv[[r, l]]
    return uv, c
def occl0(uv, c):
    c[KNEE] = 0.0; return uv, c

res = {}
for where, k0 in (("standing", 8), ("mid_descent", mid)):
    lo, hi = k0 - 1, k0 + 20
    scen = {}
    base_hc = triangulate_sequence(t, fr, calib, cams, seed=5, sigma_px=3.0)
    base_w = world_sequence(t, fr, cams, seed=5, sigma_px=3.0)

    def inj_hc(mod_fn, frames):
        sk = list(base_hc)
        for k in frames:
            a = sk[k].to_numpy(); mod_fn(a, hip_centred=True)
            sk[k] = Skeleton3D.from_numpy(a, confidences=confs(sk[k]), timestamp=sk[k].timestamp, frame_index=0)
        return sk

    def inj_w(mod_fn, frames):
        w = list(base_w)
        for k in frames:
            p, c = w[k]; p = p.copy(); mod_fn(p, hip_centred=False); w[k] = (p, c)
        return w

    for d in (20, 60):
        def f(a, hip_centred, d=d): a[KNEE] += [d / 100, 0, 0]
        scen[f"S1_single3d_{d}cm"] = (inj_hc(f, [k0]), inj_w(f, [k0]))
    def f30(a, hip_centred): a[KNEE] += [0.30, 0, 0]
    scen["S1b_persist10_30cm"] = (inj_hc(f30, range(k0, k0 + 10)), inj_w(f30, range(k0, k0 + 10)))
    def fhip(a, hip_centred):
        a[CK.LEFT_HIP] += [0, 0, 0.30]
        if hip_centred:
            a -= [0, 0, 0.15]
    scen["S5_hip_30cm"] = (inj_hc(fhip, [k0]), inj_w(fhip, [k0]))
    def bad(uv, c):
        uv[KNEE] += [80, 0.0]; return uv, c
    scen["S2_1view_80px"] = (triangulate_sequence(t, fr, calib, cams, seed=5, sigma_px=3.0, corrupt_at={k0: {"1": bad}}),
                             world_sequence(t, fr, cams, seed=5, sigma_px=3.0, corrupt_at={k0: {"1": bad}}))
    cor = {k: {"0": occl0} for k in range(len(t))}; cor[k0] = {"0": occl0, "1": bad}
    scen["S2b_2viewonly_80px"] = (triangulate_sequence(t, fr, calib, cams, seed=5, sigma_px=3.0, corrupt_at=cor),
                                  world_sequence(t, fr, cams, seed=5, sigma_px=3.0, corrupt_at=cor))
    for nf in (1, 10):
        cs = {k: {"1": swap} for k in range(k0, k0 + nf)}
        scen[f"S3_legswap_1view_{nf}f"] = (triangulate_sequence(t, fr, calib, cams, seed=5, sigma_px=3.0, corrupt_at=cs),
                                            world_sequence(t, fr, cams, seed=5, sigma_px=3.0, corrupt_at=cs))
        cs2 = {k: {"1": swap, "2": swap} for k in range(k0, k0 + nf)}
        scen[f"S3b_legswap_2views_{nf}f"] = (triangulate_sequence(t, fr, calib, cams, seed=5, sigma_px=3.0, corrupt_at=cs2),
                                              world_sequence(t, fr, cams, seed=5, sigma_px=3.0, corrupt_at=cs2))
    for name, (sk_hc, wseq) in scen.items():
        print(f"\n{where}/{name}")
        rows = {"none": prod(sk_hc, "none"), "blend": prod(sk_hc, "blend"), "blend+clamp": prod(sk_hc, "blend+clamp"),
                "gateHC(10cm,80)": gate_hc(sk_hc), "gateW(8cm,80)": gate_w(wseq)}
        for v, (arr, cf) in rows.items():
            m = metrics(arr, cf, lo, hi)
            res[f"{where}/{name}/{v}"] = m
            print(f"   {v:16s} knee max {m['max_cm']:5.1f} cm >5cm {m['f5cm']:2d}f | angle max {m['max_deg']:5.1f} >8deg {m['f8deg']:2d}f | IKzero {m['ik_zeroed']:2d} | {m['trace'][:13]}")
json.dump(res, open("exp8c_outliers.json", "w"), indent=1)
