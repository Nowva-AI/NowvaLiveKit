"""What the VelocityClamp does to realistic triangulation failures (real DLTTriangulator + real Blender/Clamp/IK)."""
import sys, json
sys.path.insert(0, ".")
import numpy as np
from synth_vc import *
from biomechanics.utils.confidence_blend import ConfidenceBlender
from biomechanics.utils.velocity_clamp import VelocityClamp
from biomechanics.kinematics.analytical_ik import AnalyticalIKSolver

IK = AnalyticalIKSolver()
calib, cams = make_rig()
KNEE = CK.LEFT_KNEE


def run_variant(sks, variant):
    bl, cl = ConfidenceBlender(0.1, 0.9), VelocityClamp(2.5, 30)
    out = []
    for sk in sks:
        if variant in ("blend", "blend+clamp"):
            sk = bl.blend(sk)
        if variant in ("clamp", "blend+clamp"):
            sk = cl.clamp(sk)
        out.append(sk)
    return out


from biomechanics.utils.geometry import joint_angle_3_points


def knee_angle(sk):
    a = sk.to_numpy()
    return 180.0 - joint_angle_3_points(a[CK.LEFT_HIP], a[KNEE], a[CK.LEFT_ANKLE])


def ik_knee(sk):
    return IK.solve(sk).knee_flexion_l


def metrics(out, truth_sks, kpt, lo, hi):
    err = np.array([np.linalg.norm(out[k].to_numpy()[kpt] - truth_sks[k].to_numpy()[kpt]) * 100 for k in range(lo, hi)])
    aerr = np.array([abs(knee_angle(out[k]) - knee_angle(truth_sks[k])) for k in range(lo, hi)])
    ikerr = np.array([abs(ik_knee(out[k]) - ik_knee(truth_sks[k])) for k in range(lo, hi)])
    return dict(max_err_cm=round(float(err.max()), 1), frames_err_gt5cm=int((err > 5).sum()), err_trace=[round(float(e), 1) for e in err[:20]],
                max_knee_deg=round(float(aerr.max()), 1), frames_knee_gt5deg=int((aerr > 5).sum()),
                ik_knee_max_deg=round(float(ikerr.max()), 1))


def replace(sk, arr=None, conf=None):
    a = sk.to_numpy() if arr is None else arr
    c = confs(sk) if conf is None else conf
    return Skeleton3D.from_numpy(a, confidences=c, timestamp=sk.timestamp, frame_index=0)


results = {}
VARIANTS = ("none", "clamp", "blend", "blend+clamp")
for tempo in ("bw_normal", "loaded_heavy"):
    t, fr = sequence_world(tempo, fps=30, stand_s=0.6)
    base = triangulate_sequence(t, fr, calib, cams, seed=5, sigma_px=3.0)
    s = depth_profile(t, tempo, 0.6, 1)
    mid = int(np.argmin(np.abs(s[: len(s) // 2] - 0.5)))
    for where, k0 in (("standing", 8), ("mid_descent", mid)):
        lo, hi = k0 - 1, k0 + 25
        # --- S1: injected 3D single-frame outlier on knee, conf unchanged
        for d_cm in (10, 20, 30, 45, 60):
            sks = list(base)
            a = sks[k0].to_numpy(); a[KNEE] += np.array([d_cm / 100, 0, 0])  # lateral (valgus-like direction)
            sks[k0] = replace(sks[k0], a)
            for v in VARIANTS:
                results[f"{tempo}/{where}/S1_3d_single_{d_cm}cm/{v}"] = metrics(run_variant(sks, v), base, KNEE, lo, hi)
        # --- S1b: persistent wrong solution 30 cm for 10 frames then correct
        sks = list(base)
        for k in range(k0, k0 + 10):
            a = sks[k].to_numpy(); a[KNEE] += np.array([0.30, 0, 0]); sks[k] = replace(sks[k], a)
        for v in VARIANTS:
            results[f"{tempo}/{where}/S1b_3d_persist10_30cm/{v}"] = metrics(run_variant(sks, v), base, KNEE, lo, hi)
        # --- S5: single-frame hip outlier 30 cm (propagates via re-centring)
        sks = list(base)
        a = sks[k0].to_numpy(); a[CK.LEFT_HIP] += np.array([0.0, 0.0, 0.30]); a = a - np.array([0.0, 0.0, 0.15])
        sks[k0] = replace(sks[k0], a)
        for v in VARIANTS:
            o = run_variant(sks, v)
            m = metrics(o, base, KNEE, lo, hi)
            moved = [int((np.linalg.norm(o[k0].to_numpy() - base[k0].to_numpy(), axis=1) > 0.05).sum())]
            m["kpts_moved_gt5cm_at_outlier"] = moved[0]
            results[f"{tempo}/{where}/S5_hip_single_30cm/{v}"] = m
        # --- S2: 2D single-view knee outlier (side cam) through the real DLT
        for px in (15, 25, 40, 80, 150):
            def corrupt(uv, c, px=px):
                uv[KNEE] += np.array([px, 0.0]); return uv, c
            sks = triangulate_sequence(t, fr, calib, cams, seed=5, sigma_px=3.0, corrupt_at={k0: {"1": corrupt}})
            conf_at = float(confs(sks[k0])[KNEE])
            for v in VARIANTS:
                m = metrics(run_variant(sks, v), base, KNEE, lo, hi); m["tri_conf"] = round(conf_at, 3)
                m["raw3d_jump_cm"] = round(float(np.linalg.norm(sks[k0].to_numpy()[KNEE] - base[k0].to_numpy()[KNEE]) * 100), 1)
                results[f"{tempo}/{where}/S2_2d_view1_{px}px/{v}"] = m
        # --- S3: L/R leg swap in one side view, 1 frame and 10 frames
        legs = [(CK.LEFT_HIP, CK.RIGHT_HIP), (CK.LEFT_KNEE, CK.RIGHT_KNEE), (CK.LEFT_ANKLE, CK.RIGHT_ANKLE), (CK.LEFT_FOOT_INDEX, CK.RIGHT_FOOT_INDEX)]
        def swap(uv, c):
            for l, r in legs[1:]:
                uv[[l, r]] = uv[[r, l]]
            return uv, c
        for n in (1, 10):
            sks = triangulate_sequence(t, fr, calib, cams, seed=5, sigma_px=3.0, corrupt_at={k: {"1": swap} for k in range(k0, k0 + n)})
            conf_at = float(confs(sks[k0])[KNEE])
            for v in VARIANTS:
                m = metrics(run_variant(sks, v), base, KNEE, lo, hi); m["tri_conf"] = round(conf_at, 3)
                m["raw3d_jump_cm"] = round(float(np.linalg.norm(sks[k0].to_numpy()[KNEE] - base[k0].to_numpy()[KNEE]) * 100), 1)
                results[f"{tempo}/{where}/S3_legswap_view1_{n}f/{v}"] = m

json.dump(results, open("exp4_outliers.json", "w"), indent=1)
# compact print
last = None
for key, m in results.items():
    scen = key.rsplit("/", 1)[0]
    if scen != last:
        extra = f" tri_conf={m.get('tri_conf')} raw3d_jump={m.get('raw3d_jump_cm')}cm" if "tri_conf" in m else ""
        print(f"\n{scen}{extra}"); last = scen
    print(f"   {key.rsplit('/',1)[1]:12s} kneeErr max {m['max_err_cm']:5.1f} cm, >5cm {m['frames_err_gt5cm']:2d}f | geomKnee max {m['max_knee_deg']:5.1f} deg, >5deg {m['frames_knee_gt5deg']:2d}f | IKknee max {m['ik_knee_max_deg']:5.1f} | trace {m['err_trace'][:16]}"
          + (f" | kpts moved>5cm {m['kpts_moved_gt5cm_at_outlier']}" if 'kpts_moved_gt5cm_at_outlier' in m else ""))
