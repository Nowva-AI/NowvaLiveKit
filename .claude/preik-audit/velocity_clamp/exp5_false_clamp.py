"""False clamping of real motion + noise, bone-length distortion and angle bias caused by VelocityClamp (no outliers)."""
import sys, json
sys.path.insert(0, ".")
import numpy as np
from synth_vc import *
from biomechanics.utils.confidence_blend import ConfidenceBlender
from biomechanics.utils.velocity_clamp import VelocityClamp
from biomechanics.utils.geometry import joint_angle_3_points

calib, cams = make_rig()
BONES = {"femur_l": (CK.LEFT_HIP, CK.LEFT_KNEE), "tibia_l": (CK.LEFT_KNEE, CK.LEFT_ANKLE),
         "uarm_l": (CK.LEFT_SHOULDER, CK.LEFT_ELBOW), "farm_l": (CK.LEFT_ELBOW, CK.LEFT_WRIST)}


def knee(a, side="l"):
    h, k, an = (CK.LEFT_HIP, CK.LEFT_KNEE, CK.LEFT_ANKLE) if side == "l" else (CK.RIGHT_HIP, CK.RIGHT_KNEE, CK.RIGHT_ANKLE)
    return 180 - joint_angle_3_points(a[h], a[k], a[an])


res = {}
print(f"{'case':38s} {'%frm clamp':>10s} {'kpts':>28s} {'maxLag cm':>9s} {'femur dL max/p99 cm':>20s} {'forearm dL max':>14s} {'knee dAng max':>13s} {'bottom depth d':>14s} {'kneesep d cm':>12s}")
for arms in ("bar", "bw_arms"):
    for tempo in ("bw_normal", "loaded_light_fast", "bw_fast", "bw_explosive"):
        if arms == "bw_arms" and tempo.startswith("loaded"):
            continue
        for sigma in (2.0, 3.0, 5.0):
            t, fr = sequence_world(tempo, fps=30, n_reps=3, arms=arms, stand_s=0.6)
            sks = triangulate_sequence(t, fr, calib, cams, seed=11, sigma_px=sigma)
            for chain in ("clamp_only", "blend+clamp"):
                bl0, bl1, cl = ConfidenceBlender(0.1, 0.9), ConfidenceBlender(0.1, 0.9), VelocityClamp(2.5, 30)
                ins, outs = [], []
                for sk in sks:
                    x = bl0.blend(sk) if chain == "blend+clamp" else sk
                    ins.append(x.to_numpy())
                    outs.append(cl.clamp(x).to_numpy())
                ins, outs = np.array(ins), np.array(outs)
                diff = np.linalg.norm(outs - ins, axis=2)
                clamped = diff > 1e-9
                frac = clamped.any(axis=1).mean() * 100
                per_k = clamped.sum(0)
                top = {n: int(per_k[i]) for n, i in (("knee", CK.LEFT_KNEE), ("ankle", CK.LEFT_ANKLE), ("wrist", CK.LEFT_WRIST), ("elbow", CK.LEFT_ELBOW), ("nose", CK.NOSE))}
                dL = {}
                for b, (i, j) in BONES.items():
                    dL[b] = np.abs(np.linalg.norm(outs[:, i] - outs[:, j], axis=1) - np.linalg.norm(ins[:, i] - ins[:, j], axis=1)) * 100
                dknee = np.abs(np.array([knee(o) - knee(a) for o, a in zip(outs, ins)]))
                depth_in = np.array([knee(a) for a in ins]); depth_out = np.array([knee(o) for o in outs])
                sep = lambda arr: np.abs(arr[:, CK.LEFT_KNEE, 0] - arr[:, CK.RIGHT_KNEE, 0])
                dsep = np.abs(sep(outs) - sep(ins)) * 100
                key = f"{arms}/{tempo}/s{sigma}/{chain}"
                res[key] = dict(pct_frames_clamped=round(float(frac), 1), clamp_counts=top, max_lag_cm=round(float(diff.max() * 100), 1),
                                femur_dL_max=round(float(dL["femur_l"].max()), 1), femur_dL_p99=round(float(np.percentile(dL["femur_l"], 99)), 2),
                                tibia_dL_max=round(float(dL["tibia_l"].max()), 1), forearm_dL_max=round(float(dL["farm_l"].max()), 1),
                                knee_dang_max=round(float(dknee.max()), 1), bottom_depth_delta=round(float(depth_out.max() - depth_in.max()), 1),
                                kneesep_d_max_cm=round(float(dsep.max()), 1))
                r = res[key]
                print(f"{key:38s} {r['pct_frames_clamped']:10.1f} {str(top):>28s} {r['max_lag_cm']:9.1f} {r['femur_dL_max']:9.1f}/{r['femur_dL_p99']:<9.2f} {r['forearm_dL_max']:14.1f} {r['knee_dang_max']:13.1f} {r['bottom_depth_delta']:14.1f} {r['kneesep_d_max_cm']:12.1f}")
json.dump(res, open("exp5_false_clamp.json", "w"), indent=1)
