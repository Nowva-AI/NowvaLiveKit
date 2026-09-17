"""E3: replicate pipeline_process.py calibration-mode sequence; athlete_params lost after line 827 reset.
Also ENABLE_PREIK_FILTERS=false crashes the calibration-mode wait loop."""
from harness import *
from biomechanics.pipeline_process import _extract_athlete_params

clock = FakeClock()
pipe, prov = build_pipeline(clock)
prov.frame_index_mode = "incrementing"

# pipeline_process.py:502-505 wait loop
frames = 0
while not (pipe.is_ready and pipe._bone_constraints.is_calibrated):
    res = step(pipe, prov, clock, squat_points(0.0))
    if pipe.is_ready and res.skeleton_3d is not None:
        pipe._bone_constraints.enforce(res.skeleton_3d)
    frames += 1
params_543 = _extract_athlete_params(pipe)
print(f"wait loop exited after {frames} frames; line 543 athlete_params = {params_543}")
print("proportions applied to rule engine/IK on next frame?", end=" ")
step(pipe, prov, clock, squat_points(0.0))
print(pipe._proportions_applied, "pelvis_tilt_coupling =", pipe._ik_solver._pelvis_tilt_coupling)

# assessment reps (line 653-719)
for d in rep_depth_profile():
    step(pipe, prov, clock, squat_points(d))
# line 827: reset before calibration phase
pipe.reset_readiness_gate()
for d in [0.0] * 30 + rep_depth_profile() * 5:
    step(pipe, prov, clock, squat_points(d))
params_909 = _extract_athlete_params(pipe)
print(f"line 909 cal_athlete_params after 5 calibration reps = {params_909}")
print("bones calibrated:", pipe._bone_constraints.is_calibrated, "progress:", pipe._bone_constraints.progress,
      "proportions_applied:", pipe._proportions_applied, "pelvis_tilt_coupling:", pipe._ik_solver._pelvis_tilt_coupling)

# ENABLE_PREIK_FILTERS=false
clock2 = FakeClock()
pipe2, prov2 = build_pipeline(clock2, preik=False)
try:
    pipe2.is_ready and pipe2._bone_constraints.is_calibrated
    step(pipe2, prov2, clock2, squat_points(0.0))
    for _ in range(10):
        step(pipe2, prov2, clock2, squat_points(0.0))
    print("preik=false: is_ready", pipe2.is_ready)
    while not (pipe2.is_ready and pipe2._bone_constraints.is_calibrated):
        step(pipe2, prov2, clock2, squat_points(0.0))
except AttributeError as exc:
    print("ENABLE_PREIK_FILTERS=false -> wait loop condition raises:", repr(exc))
try:
    _extract_athlete_params(pipe2)
except AttributeError as exc:
    print("ENABLE_PREIK_FILTERS=false -> _extract_athlete_params raises:", repr(exc))
print("preik=false standing gate ever reset by reset_readiness_gate?", end=" ")
pipe2.reset_readiness_gate()
print("standing_gate.is_ready after reset =", pipe2._standing_gate.is_ready)
