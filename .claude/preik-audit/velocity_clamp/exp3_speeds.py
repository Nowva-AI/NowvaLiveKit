"""Peak keypoint speeds (world vs hip-centred) for squat tempos, and per-frame displacement noise from 3-cam DLT."""
import sys, json
sys.path.insert(0, ".")
import numpy as np
from synth_vc import *

NAMES = {CK.NOSE: "nose", CK.LEFT_SHOULDER: "shoulder_l", CK.LEFT_ELBOW: "elbow_l", CK.LEFT_WRIST: "wrist_l",
         CK.LEFT_HIP: "hip_l", CK.LEFT_KNEE: "knee_l", CK.LEFT_ANKLE: "ankle_l", CK.LEFT_FOOT_INDEX: "toe_l"}
res = {}
print("peak speed m/s  (world | hip-centred)   and max 30Hz per-frame displacement cm (hip-centred)")
for arms in ("bar", "bw_arms"):
    for tempo in TEMPOS:
        if arms == "bw_arms" and tempo.startswith("loaded"):
            continue
        t, fr = sequence_world(tempo, fps=600, arms=arms)
        hc = hip_center(fr)
        vw = np.linalg.norm(np.gradient(fr, t, axis=0), axis=2).max(0)
        vh = np.linalg.norm(np.gradient(hc, t, axis=0), axis=2).max(0)
        t30, fr30 = sequence_world(tempo, fps=30, arms=arms)
        d30 = np.linalg.norm(np.diff(hip_center(fr30), axis=0), axis=2).max(0) * 100
        hipmid_v = np.abs(np.gradient((fr[:, CK.LEFT_HIP, 1] + fr[:, CK.RIGHT_HIP, 1]) / 2, t)).max()
        row = {NAMES[k]: (round(float(vw[k]), 2), round(float(vh[k]), 2), round(float(d30[k]), 1)) for k in NAMES}
        res[f"{arms}/{tempo}"] = dict(hip_vertical_peak=round(float(hipmid_v), 2), kpts=row)
        print(f"{arms:8s} {tempo:18s} hipmid_vert_peak={hipmid_v:.2f}  " + "  ".join(f"{n}:{v[0]:.2f}|{v[1]:.2f}|{v[2]:.1f}" for n, v in row.items()))

# noise: frame-to-frame displacement of a static standing pose, hip-centred (real triangulator) vs world (same DLT, no recentre)
calib, cams = make_rig()
rng = np.random.default_rng(1)
b = body()
static = pose_world(0.0, b)
tri = DLTTriangulator(calib, min_views=2, max_reprojection_error=15.0, min_confidence=0.3)

def dlt_world(mv):
    P = [cams[c]["P"] for c in sorted(cams)]
    uv = np.stack([mv.views[c].to_numpy()[:, :2] for c in sorted(cams)])
    A = np.empty((N_KPTS, 6, 4))
    for v in range(3):
        A[:, 2*v] = uv[v, :, 0, None] * P[v][2] - P[v][0]
        A[:, 2*v+1] = uv[v, :, 1, None] * P[v][2] - P[v][1]
    X = np.linalg.svd(A)[2][:, -1, :]
    return X[:, :3] / X[:, 3:4]

for sigma in (2.0, 3.0, 5.0):
    hc_list, w_list = [], []
    for i in range(400):
        mv = views_for_frame(static, cams, rng, sigma_px=sigma)
        hc_list.append(to_arr(tri.triangulate(mv)))
        w_list.append(dlt_world(mv))
    hc_a, w_a = np.array(hc_list), np.array(w_list)
    dh = np.linalg.norm(np.diff(hc_a, axis=0), axis=2)
    dw = np.linalg.norm(np.diff(w_a, axis=0), axis=2)
    ph = np.std(hc_a, axis=0).mean(1) * 100
    pw = np.std(w_a, axis=0).mean(1) * 100
    print(f"sigma={sigma}px per-axis pos std cm world[knee,ankle,wrist,hip]={pw[[13,15,9,11]].round(2)} hipcentred={ph[[13,15,9,11]].round(2)}; "
          f"frame-diff p99 cm world={np.percentile(dw,99)*100:.2f} hipcentred={np.percentile(dh,99)*100:.2f} max hc={dh.max()*100:.2f}")
    res[f"noise_sigma{sigma}"] = dict(world_std_cm=pw.round(2).tolist(), hc_std_cm=ph.round(2).tolist(),
                                      diff_p99_world_cm=float(np.percentile(dw,99)*100), diff_p99_hc_cm=float(np.percentile(dh,99)*100))
json.dump(res, open("exp3_speeds.json", "w"), indent=1)
