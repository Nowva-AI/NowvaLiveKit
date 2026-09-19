"""Production-equivalent diagnosis metrics on (T, 19, 3) skeleton sequences (AnalyticalIKSolver + TriangulatedValgusEstimator)."""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/Users/naiahoard/NowvaLiveKit/src")

from biomechanics.kinematics.analytical_ik import AnalyticalIKSolver  # noqa: E402
from biomechanics.kinematics.valgus import TriangulatedValgusEstimator  # noqa: E402
from biomechanics.utils.types import CocoKeypoints as CK, Skeleton3D  # noqa: E402

_IK = AnalyticalIKSolver()
_VAL = TriangulatedValgusEstimator()

ANGLE_KEYS = ["knee_flex_l", "knee_flex_r", "hip_flex_l", "hip_flex_r", "trunk_flex", "valgus_l", "valgus_r",
              "hip_add_l", "pelvis_list", "dorsi_l"]
POS_KEYS = ["depth_cm", "hip_shift_cm", "heel_rise_cm_l", "stance_cm", "hip_asym_cm", "knee_asym_deg", "knee_dev_cm_l"]


def frame_metrics(pts: np.ndarray, conf: np.ndarray | None = None) -> dict[str, float]:
    if conf is None:
        conf = np.full(len(pts), 0.9)
    skel = Skeleton3D.from_numpy(pts, confidences=list(conf), timestamp=0.0, frame_index=0)
    ang = _IK.solve(skel)
    vr = _VAL.estimate(None, skel)
    hip_mid = (pts[CK.LEFT_HIP] + pts[CK.RIGHT_HIP]) / 2.0
    ank_mid = (pts[CK.LEFT_ANKLE] + pts[CK.RIGHT_ANKLE]) / 2.0
    return {
        "knee_flex_l": ang.knee_flexion_l, "knee_flex_r": ang.knee_flexion_r,
        "hip_flex_l": ang.hip_flexion_l, "hip_flex_r": ang.hip_flexion_r,
        "trunk_flex": ang.trunk_flexion, "valgus_l": vr.valgus_l, "valgus_r": vr.valgus_r,
        "hip_add_l": ang.hip_adduction_l, "pelvis_list": ang.pelvis_list, "dorsi_l": ang.ankle_dorsiflexion_l,
        "depth_cm": (hip_mid[1] - ank_mid[1]) * 100.0,
        "hip_shift_cm": (hip_mid[0] - ank_mid[0]) * 100.0,
        "heel_rise_cm_l": (pts[CK.LEFT_FOOT_INDEX, 1] - pts[CK.LEFT_ANKLE, 1]) * 100.0,
        "stance_cm": abs(pts[CK.LEFT_ANKLE, 0] - pts[CK.RIGHT_ANKLE, 0]) * 100.0,
        "hip_asym_cm": (pts[CK.LEFT_HIP, 1] - pts[CK.RIGHT_HIP, 1]) * 100.0,
        "knee_asym_deg": ang.knee_flexion_l - ang.knee_flexion_r,
        "knee_dev_cm_l": _knee_medial_dev_cm(pts),
    }


def _knee_medial_dev_cm(pts: np.ndarray) -> float:
    hip, knee, ankle = pts[CK.LEFT_HIP], pts[CK.LEFT_KNEE], pts[CK.LEFT_ANKLE]
    line = ankle - hip
    proj = hip + np.dot(knee - hip, line) / max(float(np.dot(line, line)), 1e-9) * line
    return float(-(knee - proj)[0] * 100.0)


def seq_metrics(seq: np.ndarray) -> dict[str, np.ndarray]:
    rows = [frame_metrics(p) for p in seq]
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def ta_joint_errors(est: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Translation-aligned per-joint error (T, J) in metres: per-frame mean offset over body joints removed."""
    diff = est - truth
    offset = diff.mean(axis=1, keepdims=True)
    return np.linalg.norm(diff - offset, axis=-1)


def hip_aligned_joint_errors(est: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-joint error after aligning hip midpoints (the IK root)."""
    e = est - ((est[:, CK.LEFT_HIP] + est[:, CK.RIGHT_HIP]) / 2.0)[:, None]
    t = truth - ((truth[:, CK.LEFT_HIP] + truth[:, CK.RIGHT_HIP]) / 2.0)[:, None]
    return np.linalg.norm(e - t, axis=-1)


def summarize(est: np.ndarray, truth: np.ndarray, truth_m: dict[str, np.ndarray] | None = None,
              frames: np.ndarray | None = None) -> dict[str, float]:
    truth_m = truth_m or seq_metrics(truth)
    est_m = seq_metrics(est)
    idx = np.arange(len(est)) if frames is None else frames
    out: dict[str, float] = {}
    err = ta_joint_errors(est[idx], truth[idx])
    out["mpjpe_cm"] = float(err.mean() * 100)
    for name, j in (("hip", CK.LEFT_HIP), ("knee", CK.LEFT_KNEE), ("ankle", CK.LEFT_ANKLE), ("toe", CK.LEFT_FOOT_INDEX),
                    ("shoulder", CK.LEFT_SHOULDER), ("wrist", CK.LEFT_WRIST)):
        out[f"{name}_cm"] = float(np.sqrt((err[:, j] ** 2).mean()) * 100)
    for k in ANGLE_KEYS + POS_KEYS:
        d = est_m[k][idx] - truth_m[k][idx]
        out[f"{k}_rmse"] = float(np.sqrt((d ** 2).mean()))
        out[f"{k}_bias"] = float(d.mean())
    return out
