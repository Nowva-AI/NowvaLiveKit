"""E2: body-proportion scaling re-applied every set when the chain calls bone constraints -> thresholds compound.
Also shows forward-lean scaling direction is inverted for the 180-convention threshold."""
from collections import Counter
import harness
from harness import *
from biomechanics import pipeline as pipeline_module
from biomechanics.utils.bone_constraints import BoneLengthConstraints

def full_chain(skeleton, *, confidence_blender, velocity_clamp, bone_constraints, ground_clamp, position_smoother):
    # the commented-out production order in preik_chain.py
    skeleton = confidence_blender.blend(skeleton)
    skeleton = velocity_clamp.clamp(skeleton)
    skeleton = bone_constraints.enforce(skeleton)
    skeleton = ground_clamp.clamp(skeleton)
    skeleton = position_smoother.smooth(skeleton)
    return bone_constraints.enforce(skeleton)

def thresholds(pipe):
    out = {}
    for r in pipe._rule_engine.rules:
        n = type(r).__name__
        if n in ("ForwardLeanRule", "KneeValgusRule"):
            out[n] = (round(r.mild_threshold, 1), round(r.moderate_threshold, 1), round(r.severe_threshold, 1))
    return out

def run(label: str, femur: float, torso: float, chain_enabled: bool, sets: int = 3):
    harness.FEMUR_M, harness.TORSO_M = femur, torso
    clock = FakeClock()
    pipe, prov = build_pipeline(clock)
    prov.frame_index_mode = "incrementing"
    if chain_enabled:
        pipeline_module.apply_preik_filters = full_chain
    print(f"\n== {label}: femur={femur} torso={torso} chain_enabled={chain_enabled}")
    print("   initial", thresholds(pipe))
    for s in range(1, sets + 1):
        faults = Counter()
        for d in [0.0] * 60 + rep_depth_profile() * 2:
            # upright squat: only 20 deg trunk lean at the bottom (trunk_flexion 160 > default mild 145)
            res = step(pipe, prov, clock, squat_points(d, valgus_m=0.0, lean_extra_deg=-20.0 * d))
            for f in res.faults:
                faults[f.fault_type] += 1
        bc = pipe._bone_constraints
        p = bc.body_proportions
        scales = (round(p.valgus_scale, 3), round(p.forward_lean_scale, 3)) if p else None
        print(f"   set {s}: bones_calibrated={bc.is_calibrated} (valgus_scale, fwd_lean_scale)={scales} "
              f"thresholds={thresholds(pipe)} faults={dict(faults)} reps={pipe.rep_count}")
        pipe.reset_readiness_gate()
    import importlib
    from biomechanics.utils import preik_chain
    pipeline_module.apply_preik_filters = preik_chain.apply_preik_filters

run("production (chain stages commented out)", 0.50, 0.45, chain_enabled=False)
run("chain re-enabled, long femur", 0.50, 0.45, chain_enabled=True)
run("chain re-enabled, short femur", 0.40, 0.58, chain_enabled=True)
