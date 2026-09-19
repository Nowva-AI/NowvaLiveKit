"""E6: inspector hooks record stale stage data on gated/rest/None frames.
E7: visualize_triangulated.py stabilization loop attribute access; debug_filters.py import/call compatibility."""
import numpy as np
from harness import *
from biomechanics.viz.pipeline_inspector import STAGE_NAMES

clock = FakeClock()
pipe, prov = build_pipeline(clock)
prov.frame_index_mode = "incrementing"
pipe.enable_inspect()

def snapshot(tag, res):
    inter = pipe._inspect_intermediates
    raw = pipe._inspect_raw_kpts
    print(f"   {tag:38s} ready={pipe.is_ready!s:5s} skel3d={'yes' if res.skeleton_3d is not None else 'no ':3s} "
          f"joint_angles={'yes' if res.joint_angles is not None else 'no ':3s} "
          f"intermediates={'None' if inter is None else sorted(inter)} raw_angles={'set' if pipe._inspect_raw_angles is not None else 'None'} "
          f"raw_kpts_left_knee_y={None if raw is None else round(float(raw[13][1]), 3)}")

for d in [0.0] * 8:
    res = step(pipe, prov, clock, squat_points(d))
snapshot("ready standing frame", res)
res = step(pipe, prov, clock, squat_points(0.6))
snapshot("ready mid-squat frame (depth 0.6)", res)
pipe.presence_only = True
res = step(pipe, prov, clock, squat_points(0.0))
snapshot("presence_only (rest) frame, standing", res)
pipe.presence_only = False
pipe.reset_readiness_gate()
res = step(pipe, prov, clock, squat_points(0.0))
snapshot("gated frame after reset, standing", res)
for _ in range(6):
    res = step(pipe, prov, clock, None)
snapshot("pose None beyond hold window", res)
print("   inspector STAGE_NAMES:", STAGE_NAMES, "-> 4 of 7 stages always None in production")

# E7: visualize_triangulated.py line 470 attribute
from biomechanics.utils.bone_constraints import BoneLengthConstraints
bc = BoneLengthConstraints(calibration_frames=30, tolerance=0.15)
try:
    status = f"calibrating [{bc._frame_count}/{bc._calibration_frames}]"
    print("E7 visualize_triangulated line 470 OK:", status)
except AttributeError as exc:
    print("E7 visualize_triangulated.py:470 raises:", repr(exc))

# debug_filters.py: imports + RepCounter.update signature
import inspect
try:
    from biomechanics.faults import RuleEngine, RepCounter, RepCounterConfig
    print("E7 debug_filters imports OK; RepCounter.update signature:", inspect.signature(RepCounter.update))
except Exception as exc:
    print("E7 debug_filters.py import fails:", repr(exc))
from biomechanics.faults.rep_counter import RepCounter as RC
print("E7 visualize_triangulated RepCounter.update signature:", inspect.signature(RC.update))
