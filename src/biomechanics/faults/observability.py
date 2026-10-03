"""Which faults the current camera setup can actually see.

A coach who cannot see something says nothing about it. A single frontal
camera measures heights, timing and knee tracking well, because they lie in the
image plane. Anything along the camera axis — trunk pitch, front-to-back
balance, foot direction — comes from monocular depth regression and is only
approximate, and so is sideways hip travel, which that depth error leaks into.
Heel rise needs the multi-camera foot contact model. The intra-set side-view
faults (hip shoot, balance, velocity loss) are rig-only by product decision:
on one ~12 fps camera they are noise-dominated, so they are never emitted.
"""

from __future__ import annotations

import os

SINGLE_CAMERA = "single_camera"
TRIANGULATED = "triangulated"

OBSERVABLE = "observable"
APPROXIMATE = "approximate"
NOT_OBSERVABLE = "not_observable"

_MEASUREMENT_LEVELS: dict[str, dict[str, str]] = {
    "vertical": {SINGLE_CAMERA: OBSERVABLE, TRIANGULATED: OBSERVABLE},
    "frontal": {SINGLE_CAMERA: OBSERVABLE, TRIANGULATED: OBSERVABLE},
    # Sideways travel of the pelvis over the whole rep. The hips also travel
    # 25-30 cm backwards, and a few degrees of error in the monocular depth
    # axis leaks that into a consistent fake shift (seen on recorded runs:
    # ~half of reps, same sign within a set) — approximate until triangulated.
    "lateral_travel": {SINGLE_CAMERA: APPROXIMATE, TRIANGULATED: OBSERVABLE},
    "sagittal": {SINGLE_CAMERA: APPROXIMATE, TRIANGULATED: OBSERVABLE},
    # Side-view rep verdicts: pitch at a third of the ascent, load over midfoot,
    # bar speed. Only the triangulated rig at >= 30 fps resolves them.
    "side_view": {SINGLE_CAMERA: NOT_OBSERVABLE, TRIANGULATED: OBSERVABLE},
    "foot_contact": {SINGLE_CAMERA: NOT_OBSERVABLE, TRIANGULATED: OBSERVABLE},
    "bar": {SINGLE_CAMERA: OBSERVABLE, TRIANGULATED: OBSERVABLE},
    # Depth-axis metrics no camera can resolve, emitted on the rig but only
    # hedged until the rig validation promotes them (shared across exercises).
    "rig_unvalidated": {SINGLE_CAMERA: NOT_OBSERVABLE, TRIANGULATED: APPROXIMATE},
    # The overhead press keeps its own classes so each promotion is a one-line
    # edit that cannot move another exercise (docs/exercises/ohp.md §5.9).
    "press_vertical": {SINGLE_CAMERA: OBSERVABLE, TRIANGULATED: OBSERVABLE},
    "press_frontal": {SINGLE_CAMERA: OBSERVABLE, TRIANGULATED: OBSERVABLE},
    "press_elbow": {SINGLE_CAMERA: APPROXIMATE, TRIANGULATED: APPROXIMATE},
    "press_side_view": {SINGLE_CAMERA: NOT_OBSERVABLE, TRIANGULATED: APPROXIMATE},
}

# What each fault is measured from.
FAULT_MEASUREMENT: dict[str, str] = {
    # Squat
    "depth": "vertical",
    "depth_drift": "vertical",
    "lockout": "vertical",
    "tempo": "vertical",
    "velocity_loss": "side_view",
    "knee_valgus": "frontal",
    "hip_shift": "lateral_travel",
    "hip_shoot": "side_view",
    "balance": "side_view",
    "foot_placement": "sagittal",
    "heel_rise": "foot_contact",
    "bilateral_asymmetry": "bar",
    # Standing barbell overhead press (ohp.md §5.9)
    "press_leg_drive": "press_vertical",
    "press_missed": "press_vertical",
    "press_uneven": "press_frontal",
    "press_grip": "press_frontal",
    "press_lockout": "press_elbow",
    "press_layback": "press_side_view",
    "press_bar_forward": "press_side_view",
    "press_velocity_loss": "press_side_view",
    # Barbell Romanian deadlift (rdl.md §5)
    "rdl_hip_sink": "vertical",
    "rdl_range_drift": "vertical",
    "rdl_partial": "vertical",
    "rdl_tempo": "vertical",
    "rdl_bar_tilt": "frontal",
    "rdl_bar_drift": "rig_unvalidated",
    "rdl_lockout": "rig_unvalidated",
}


def capture_mode_from_env() -> str:
    multi = os.getenv("NOWVA_MULTI_CAMERA", "false").lower() == "true"
    return TRIANGULATED if multi else SINGLE_CAMERA


def measurement_observability(measurement: str, capture_mode: str) -> str:
    levels = _MEASUREMENT_LEVELS.get(measurement)
    if levels is None:
        return OBSERVABLE
    return levels.get(capture_mode, OBSERVABLE)


def fault_observability(fault_type: str, capture_mode: str) -> str:
    """Faults with no declared measurement (other exercises) count as observable."""
    measurement = FAULT_MEASUREMENT.get(fault_type)
    if measurement is None:
        return OBSERVABLE
    return measurement_observability(measurement, capture_mode)
