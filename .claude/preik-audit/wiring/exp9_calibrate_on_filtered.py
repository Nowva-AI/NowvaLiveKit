"""E9: bone calibration (driven from pipeline_process wait loop) records the FILTERED skeleton in the frames right after
the RAW-evaluated gates latch -> converging filter state is baked into athlete_params (foot_avg_m)."""
import numpy as np
from harness import *
from biomechanics.pipeline_process import _extract_athlete_params
from biomechanics.utils.types import CocoKeypoints as CK

truth_foot = float(np.linalg.norm(squat_points(0.0)[CK.LEFT_FOOT_INDEX] - squat_points(0.0)[CK.LEFT_ANKLE]))
for hidden in (0, 3, 6, 12):
    clock = FakeClock()
    pipe, prov = build_pipeline(clock)
    prov.frame_index_mode = "incrementing"
    i = 0
    while not (pipe.is_ready and pipe._bone_constraints.is_calibrated):
        pts = squat_points(0.0); conf = np.full(19, 0.9)
        if i < hidden:
            pts[CK.LEFT_FOOT_INDEX] = 0.0; conf[CK.LEFT_FOOT_INDEX] = 0.0; pts[CK.RIGHT_FOOT_INDEX] = 0.0; conf[CK.RIGHT_FOOT_INDEX] = 0.0
        res = step(pipe, prov, clock, pts, conf)
        if pipe.is_ready and res.skeleton_3d is not None:
            pipe._bone_constraints.enforce(res.skeleton_3d)
        i += 1
    p = _extract_athlete_params(pipe)
    print(f"toes untriangulated for first {hidden:2d} frames -> foot_avg_m={p['foot_avg_m']:.3f} (truth {truth_foot:.3f}), frames={i}")
