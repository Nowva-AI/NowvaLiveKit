"""Exp 4: hip-midpoint re-centering in DLTTriangulator (real class).
(a) conf-0 keypoints stay at (0,0,0); (b) a missing hip disables centering (frame jumps to world coords);
(c) centering injects hip-mid noise coherently into every keypoint; (d) a hip outlier moves the whole skeleton."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import synth
sys.path.insert(0, "/Users/naiahoard/NowvaLiveKit/src")
from biomechanics.triangulation.calibration import CalibrationResult, CameraCalibration  # noqa: E402
from biomechanics.triangulation.triangulator import DLTTriangulator  # noqa: E402
from biomechanics.utils.types import Keypoint2D, MultiViewPose, Skeleton2D  # noqa: E402

OUT = Path(__file__).parent
RNG = np.random.default_rng(3)
P = synth.rig()
cal = CalibrationResult()
for v in range(3):
    cal.cameras[str(v)] = CameraCalibration(str(v), P[v], np.eye(3), np.eye(3), np.zeros(3), 0.0, (1280, 720))
tri = DLTTriangulator(cal)
seq, phase = synth.squat_sequence()


def views(uv, conf):
    return MultiViewPose(views={str(v): Skeleton2D(keypoints=[Keypoint2D(x=float(a), y=float(b), confidence=float(c))
                                                            for (a, b), c in zip(uv[v], conf[v])]) for v in range(3)})


res = {}
f = 54  # near bottom
uv, _ = synth.project(P, seq[f, :19])
conf = np.full((3, 19), 0.9)
conf[:, 15] = 0.1  # left ankle below threshold in all views -> untriangulated
sk = tri.triangulate(views(uv, conf))
res["a_untriangulated_left_ankle_xyz_conf"] = sk.to_numpy()[15].tolist() + [sk.keypoints[15].confidence]
res["a_true_left_ankle_hipcentered"] = (seq[f, 15] - (seq[f, 11] + seq[f, 12]) / 2).round(3).tolist()
conf2 = np.full((3, 19), 0.9)
conf2[:2, 11] = 0.1  # left hip seen by only one view
sk2 = tri.triangulate(views(uv, conf2))
sk_ok = tri.triangulate(views(uv, np.full((3, 19), 0.9)))
res["b_hip_missing_nose_y_vs_centered_nose_y_m"] = [float(sk2.to_numpy()[0, 1]), float(sk_ok.to_numpy()[0, 1])]
res["b_hip_missing_whole_skeleton_offset_m"] = float(np.linalg.norm(sk2.to_numpy()[5] - sk_ok.to_numpy()[5]))
# (c) noise injection at standing
wr, ce = [], []
for _ in range(400):
    uvn = synth.project(P, seq[0, :19])[0] + RNG.normal(0, 1.5, (3, 19, 2))
    X = synth.dlt(P, uvn)
    wr.append(X[[15, 16, 5, 6, 13, 14]])
    ce.append(X[[15, 16, 5, 6, 13, 14]] - (X[11] + X[12]) / 2)
wr, ce = np.array(wr), np.array(ce)
res["c_std_mm_world_ankles_shoulders_knees"] = (wr.std(0).mean(1) * 1000).round(2).tolist()
res["c_std_mm_hipcentered_same"] = (ce.std(0).mean(1) * 1000).round(2).tolist()
# common-mode: correlation of left and right ankle x noise
res["c_corr_Lankle_Rankle_x_world"] = float(np.corrcoef(wr[:, 0, 0], wr[:, 1, 0])[0, 1])
res["c_corr_Lankle_Rankle_x_centered"] = float(np.corrcoef(ce[:, 0, 0], ce[:, 1, 0])[0, 1])
# (d) 80 px outlier on left hip in one view: displacement of feet after centering
uvo = uv.copy(); uvo[0, 11] += np.array([80.0, 0.0])
Xo = synth.dlt(P, uvo); Xc = synth.dlt(P, uv)
res["d_hip_outlier_80px_feet_shift_mm_after_centering"] = float(np.linalg.norm((Xo[15] - (Xo[11] + Xo[12]) / 2) - (Xc[15] - (Xc[11] + Xc[12]) / 2)) * 1000)
# (e) ankle apparent speed in hip-centered frame during descent (feet static in world)
cen = seq - (seq[:, 11:12] + seq[:, 12:13]) / 2
v = np.linalg.norm(np.diff(cen[:, 15], axis=0), axis=1) * 30
res["e_ankle_speed_hipcentered_m_s_max"] = float(v.max())
res["e_hip_drop_m"] = float(seq[:, 11, 1].max() - seq[:, 11, 1].min())
(OUT / "exp4_results.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
