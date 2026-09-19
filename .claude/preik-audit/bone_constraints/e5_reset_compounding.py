"""E5: drive the REAL BiomechanicsPipeline (fake 3D pose provider) through the production calibration flow and set resets."""
from __future__ import annotations
import logging
import os
import sys
import numpy as np
logging.disable(logging.WARNING)
sys.path.insert(0, "/Users/naiahoard/NowvaLiveKit/src")
from synth import sequence, add_correlated_noise, to_skel  # noqa: E402
import biomechanics.pipeline as pl  # noqa: E402
import biomechanics.pipeline_process as pp  # noqa: E402
from biomechanics.pipeline import BiomechanicsPipeline  # noqa: E402
from biomechanics.config import load_pipeline_config  # noqa: E402
from biomechanics.faults.rule_engine import RuleEngine  # noqa: E402
from biomechanics.faults.fault_types import FaultType  # noqa: E402
from biomechanics.kinematics.valgus import TriangulatedValgusEstimator  # noqa: E402
from biomechanics.calibration import apply_calibration_to_rule_engine, build_calibration_profile  # noqa: E402

rng = np.random.default_rng(7)
truth, s = sequence(900)
noisy, _ = add_correlated_noise(truth, rng, 0.015, rho=0.6, z_scale=1.5)
DUMMY = np.zeros((4, 4, 3), dtype=np.uint8)


class FakeProvider:
    def __init__(self):
        self.i = 0

    def get_pose(self):
        p = noisy[self.i % len(noisy)]
        self.i += 1
        return DUMMY, None, to_skel(p, t=self.i / 30.0, i=self.i)


def make_pipeline():
    cfg = load_pipeline_config()
    p = BiomechanicsPipeline(config=cfg, defer_capture=True)
    os.environ["NOWVA_MULTI_CAMERA"] = "true"
    p._rule_engine = RuleEngine(rules=p._profile.create_fault_rules(cfg))
    p._rule_engine._profile = p._profile
    os.environ["NOWVA_MULTI_CAMERA"] = "false"
    p._multi_camera = True
    p._multi_camera_provider = FakeProvider()
    p._valgus_estimator = TriangulatedValgusEstimator()
    return p


def thr(p):
    v = p._rule_engine.get_rule(FaultType.KNEE_VALGUS)
    f = p._rule_engine.get_rule(FaultType.FORWARD_LEAN)
    return f"valgus mild/mod/sev {v.mild_threshold:.2f}/{v.moderate_threshold:.2f}/{v.severe_threshold:.2f}  fwd_lean mild/mod/sev {f.mild_threshold:.1f}/{f.moderate_threshold:.1f}/{f.severe_threshold:.1f}"


def wait_loop(p, max_frames=300):
    n = 0
    while not (p.is_ready and p._bone_constraints.is_calibrated) and n < max_frames:
        r = p.process_frame()
        if p.is_ready and r.skeleton_3d is not None:
            p._bone_constraints.enforce(r.skeleton_3d)
        n += 1
    return n


def run(p, n):
    for _ in range(n):
        p.process_frame()


print("=== Scenario A: production today (chain disabled) — calibration_mode flow")
p = make_pipeline()
print("initial:", thr(p))
n = wait_loop(p)
print(f"wait loop exited after {n} frames; bones calibrated={p._bone_constraints.is_calibrated}")
bp = p._bone_constraints.body_proportions
print(f"body proportions: hip_w {bp.hip_width:.3f} femur {bp.femur_length_avg:.3f} torso {bp.torso_length_avg:.3f} -> valgus_scale {bp.valgus_scale:.3f} fwd_lean_scale {bp.forward_lean_scale:.3f} pelvis_coupling {bp.pelvis_tilt_coupling:.3f}")
print("athlete params after wait loop:", {k: round(v, 3) for k, v in pp._extract_athlete_params(p).items()})
run(p, 3)
print("after next frames (scaling applied):", thr(p), "| IK coupling", p._ik_solver._pelvis_tilt_coupling)
run(p, 200)
p.reset_readiness_gate()   # pipeline_process.py:827 after assessment
print(f"after reset_readiness_gate (line 827): bones calibrated={p._bone_constraints.is_calibrated} athlete_params={pp._extract_athlete_params(p)} proportions_applied={p._proportions_applied}")
run(p, 400)                # phase-2 calibration loop never calls enforce()
print(f"after 400 phase-2 frames: bones calibrated={p._bone_constraints.is_calibrated} athlete_params (line 907) = {pp._extract_athlete_params(p)}")
print("thresholds before apply_calibration_to_rule_engine:", thr(p))
peaks = dict(trunk_flexion=150.0, hip_adduction=5.0, asymmetry=3.0, avg_depth=110.0)
apply_calibration_to_rule_engine(p._rule_engine, build_calibration_profile(peaks, p.config))
print("after apply_calibration_to_rule_engine (line 906):", thr(p), "-> proportion scaling erased")

print("\n=== Scenario B: bone enforce re-enabled in the chain (as the commented lines in preik_chain.py)")
orig = pl.apply_preik_filters


def chain_with_bones(skeleton, *, confidence_blender, velocity_clamp, bone_constraints, ground_clamp, position_smoother):
    skeleton = confidence_blender.blend(skeleton)
    skeleton = velocity_clamp.clamp(skeleton)
    skeleton = bone_constraints.enforce(skeleton)
    return bone_constraints.enforce(skeleton)


pl.apply_preik_filters = chain_with_bones
p = make_pipeline()
print("initial:", thr(p))
for set_no in range(1, 5):
    run(p, 220)
    print(f"end of set {set_no}: bones calibrated={p._bone_constraints.is_calibrated} applied={p._proportions_applied} | {thr(p)}")
    p.reset_readiness_gate()
pl.apply_preik_filters = orig
