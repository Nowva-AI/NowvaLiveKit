import numpy as np, sys
sys.path.insert(0, ".")
from synth_vc import *
from biomechanics.kinematics.analytical_ik import AnalyticalIKSolver
t, fr = sequence_world("bw_fast", fps=30)
b = body()
fem = np.linalg.norm(fr[:, CK.LEFT_HIP]-fr[:, CK.LEFT_KNEE], axis=1)
tib = np.linalg.norm(fr[:, CK.LEFT_KNEE]-fr[:, CK.LEFT_ANKLE], axis=1)
femr = np.linalg.norm(fr[:, CK.RIGHT_HIP]-fr[:, CK.RIGHT_KNEE], axis=1)
print("femur", fem.min(), fem.max(), b["femur"], "tibia", tib.min(), tib.max(), "femR", femr.min(), femr.max())
hipy = (fr[:, CK.LEFT_HIP,1]+fr[:, CK.RIGHT_HIP,1])/2
print("hip y range", hipy.min(), hipy.max(), "ankle y", fr[0, CK.LEFT_ANKLE])
ik = AnalyticalIKSolver()
hc = hip_center(fr)
kf = [ik.solve(skel(hc[i], t[i], 0.9)).knee_flexion_l for i in range(len(t))]
hf = [ik.solve(skel(hc[i], t[i], 0.9)).hip_flexion_l for i in range(len(t))]
tf = [ik.solve(skel(hc[i], t[i], 0.9)).trunk_flexion for i in range(len(t))]
print("knee flex min/max", min(kf), max(kf), "hip flex", min(hf), max(hf), "trunk", min(tf), max(tf))
calib, cams = make_rig()
sks = triangulate_sequence(t, fr, calib, cams, sigma_px=2.0)
err = np.array([np.linalg.norm(to_arr(s) - hc[i], axis=1) for i, s in enumerate(sks)])
print("tri err (hip-centred) median cm", np.median(err)*100, "p95", np.percentile(err,95)*100)
print("conf sample", confs(sks[0]).round(2))
for cid, cam in cams.items():
    uv = project(cam["P"], fr[0]); print(cid, uv.min(0).round(0), uv.max(0).round(0))
