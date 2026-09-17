"""E5a: pre-rest skeleton held through dropout after gate re-arm counts toward readiness gate.
E5b: set start seeds blender/clamp with a conf-0 keypoint parked at the origin (triangulator output) -> crawl."""
import numpy as np
from harness import *
from biomechanics.utils.types import CocoKeypoints as CK

# ---- E5a
clock = FakeClock()
pipe, prov = build_pipeline(clock)
prov.frame_index_mode = "incrementing"
for d in [0.0] * 20:
    step(pipe, prov, clock, squat_points(d))
print("E5a set1 ready:", pipe.is_ready)
pipe.presence_only = True              # rest_start (pipeline_process.py:1076)
for _ in range(90 * 30):                # 90 s rest, user visible
    step(pipe, prov, clock, squat_points(0.0))
pipe.presence_only = False              # gate pre-arm (1340-1341)
pipe.reset_readiness_gate()
progress = []
for _ in range(5):                      # user steps out of view: pose None
    res = step(pipe, prov, clock, None)
    progress.append((pipe._readiness_gate.progress[0], None if res.skeleton_3d is None else round(res.skeleton_3d.keypoints[CK.LEFT_KNEE].confidence, 2)))
print("E5a readiness progress during 5 None frames right after re-arm (progress, held knee conf):", progress)
print("    held skeleton timestamp age at re-arm = %.1f s" % (clock.time() - 1.7e9 - 20/30.0))

# ---- E5b
def first_frames(conf0_first: bool):
    clock = FakeClock()
    pipe, prov = build_pipeline(clock)
    prov.frame_index_mode = "incrementing"
    errs = []
    for i in range(40):
        pts = squat_points(0.0)
        conf = np.full(19, 0.9)
        if conf0_first and i < 6:
            # untriangulated toe: triangulator leaves (0,0,0) with conf 0 (triangulator.py:90,136-137)
            pts[CK.LEFT_FOOT_INDEX] = 0.0
            conf[CK.LEFT_FOOT_INDEX] = 0.0
        res = step(pipe, prov, clock, pts, conf)
        if res.joint_angles is not None:
            truth = squat_points(0.0)[CK.LEFT_FOOT_INDEX]
            out = res.skeleton_3d.to_numpy()[CK.LEFT_FOOT_INDEX]
            errs.append((i, round(float(np.linalg.norm(out - truth)) * 100, 1),
                         round(res.joint_angles.hip_rotation_l, 1), round(res.skeleton_3d.keypoints[CK.LEFT_FOOT_INDEX].confidence, 2)))
    return errs

print("E5b toe error (cm) / hip_rotation_l / conf per ready frame, toe conf 0 at origin for first 6 frames:")
for row in first_frames(True):
    print("   ", row)
print("E5b control (toe visible from start):", first_frames(False)[:3])
