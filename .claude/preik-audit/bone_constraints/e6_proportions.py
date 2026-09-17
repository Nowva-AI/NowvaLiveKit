"""E6: BodyProportions reference ratios/clamps for the 1.885 m user; hip-width sweep; does GS valgus depend on hip width;
forward-lean scaling direction."""
from __future__ import annotations
import logging
import numpy as np
logging.disable(logging.WARNING)
from synth import body, pose, recenter, sequence
from metrics import frame_metrics, seq_metrics
from biomechanics.utils.bone_constraints import BoneLengthConstraints, REFERENCE_HIP_TO_FEMUR_RATIO, REFERENCE_FEMUR_TO_TORSO
from biomechanics.utils.types import CocoKeypoints as CK
from methods import lengths_from


def proportions_for(pts):
    bc = BoneLengthConstraints(calibration_frames=1)
    bc.enforce(__import__("synth").to_skel(pts))
    return bc.body_proportions, bc._calibrated_lengths


b = body()
p0 = recenter(pose(0.0, b))
bp, L = proportions_for(p0)
print(f"1.885 m user, SEGMENT_RATIOS: hip_w {bp.hip_width:.3f} femur {bp.femur_length_avg:.3f} tibia {bp.tibia_length_avg:.3f} torso(side) {bp.torso_length_avg:.3f}")
print(f"  hip/femur {bp.hip_to_femur_ratio:.3f} (ref {REFERENCE_HIP_TO_FEMUR_RATIO}) -> valgus_scale {bp.valgus_scale:.3f}; femur/torso {bp.femur_length_avg/bp.torso_length_avg:.3f} (ref {REFERENCE_FEMUR_TO_TORSO}) -> fwd_lean_scale {bp.forward_lean_scale:.3f}; coupling {bp.pelvis_tilt_coupling:.3f}; tibia/0.45 {bp.tibia_to_reference_ratio:.3f}")

print("\nHip-width sweep (keypoint hip-joint distance), same 1.885 m body:")
for hw in (0.16, 0.18, 0.20, 0.22, 0.25, 0.273, 0.30):
    bb = body(hip_half_m=hw / 2)
    bpp, _ = proportions_for(recenter(pose(0.0, bb)))
    print(f"  hip_w {hw:.3f}: valgus_scale {bpp.valgus_scale:.3f} -> 3D thresholds {12*bpp.valgus_scale:.1f}/{17*bpp.valgus_scale:.1f}/{24*bpp.valgus_scale:.1f}  coupling {bpp.pelvis_tilt_coupling:.3f}")
print("  unclamped scale hits 0.7 clamp at hip_w <=", round(0.7 * REFERENCE_HIP_TO_FEMUR_RATIO * bp.femur_length_avg, 3), "m for this femur")
for h in (1.60, 1.75, 1.885, 2.0):
    bb = body(height_m=h, hip_half_m=0.10)
    bpp, _ = proportions_for(recenter(pose(0.0, bb)))
    print(f"  height {h} m with hip_w 0.20: femur {bpp.femur_length_avg:.3f} valgus_scale {bpp.valgus_scale:.3f}")

print("\nDoes the triangulated GS valgus metric depend on hip width? (bottom of squat, same stance 0.40 m, same knee swivel)")
for sw in (0.0, 25.0, 35.0):
    row = f"  knee swivel {sw:4.0f}: "
    for hw in (0.18, 0.22, 0.273, 0.30):
        bb = body(hip_half_m=hw / 2)
        m = frame_metrics(recenter(pose(1.0, bb, valgus_deg_l=sw)))
        row += f"hip_w {hw:.3f} -> GS {m['valgus_l']:+6.1f} hip_add {m['hip_add_l']:+6.1f} knee_dev {m['knee_dev_cm_l']:+5.1f}cm | "
    print(row)

print("\nForward-lean rule: yaml thresholds 145/135/125 (180-convention: 180 = upright), multiplied by forward_lean_scale")
truth, s = sequence(420)
tf = seq_metrics(truth)["trunk_flex"]
in_rep = s > 0.05
for sc in (0.8, 0.933, 1.0, 1.1, 1.2, 1.3):
    mild = 145.0 * sc
    lean_allowed = 180.0 - mild
    frac = float((tf[in_rep] < mild).mean())
    intended_mild = 180.0 - 35.0 * sc  # lean-space scaling: longer femur -> MORE lean allowed
    frac_int = float((tf[in_rep] < intended_mild).mean())
    print(f"  scale {sc:5.3f}: mild -> {mild:6.1f} (lean allowed {lean_allowed:+6.1f} deg) fires on {100*frac:5.1f}% of in-rep frames | lean-space intent {intended_mild:.1f} -> {100*frac_int:5.1f}%")
print(f"  (synthetic squat trunk_flex range {tf.min():.1f}..{tf.max():.1f})")
