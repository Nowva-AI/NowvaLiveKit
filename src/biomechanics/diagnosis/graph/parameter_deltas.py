"""Parameter-delta functions for tier-1 (cue-correctable) causes.

Each delta function returns a dict. The KeypointCorrector uses the cause_id
to decide what geometric correction to apply; these deltas serve as
structured metadata about what the engine thinks should change.

Each delta function is paired with a magnitude function that renders its
delta as a short spoken phrase, so the key names stay defined and consumed
in one file. Magnitude functions return None when no phrase is derivable.

Shared anthropometry-driven target functions also live here so the geometric
corrections and the spoken explanation templates derive from the same
computation and cannot disagree.
"""

from __future__ import annotations

import math

from ..types import RepKinematicSummary


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _unit(value: int, unit: str) -> str:
    return unit if value == 1 else f"{unit}s"


def dorsi_driven_targets(
    dorsiflexion_capacity: float, anthro: dict
) -> tuple[float, float]:
    """Compute target stance ratio and toe-out from dorsiflexion + femur proportions.

    Returns (target_stance_ratio_x_shoulder_width, target_toe_out_degrees).
    """
    femur_torso_ratio = anthro.get("femur_torso_ratio", 1.0)
    dorsi_clamped = max(10.0, min(45.0, dorsiflexion_capacity))

    dorsi_factor = _clamp((45.0 - dorsi_clamped) / 25.0)
    femur_factor = _clamp((femur_torso_ratio - 0.70) / 0.30)

    target_stance = (
        1.1
        + femur_factor * 0.5
        + dorsi_factor * 0.5
        + femur_factor * dorsi_factor * 0.3
    )
    target_stance = min(target_stance, 2.0)

    target_toe_out = (
        15.0
        + femur_factor * 10.0
        + dorsi_factor * 8.0
        + femur_factor * dorsi_factor * 5.0
    )

    return (max(1.1, target_stance), max(15.0, min(40.0, target_toe_out)))


def stance_target_ratio(current_ratio: float, anthro: dict, rom: dict) -> float:
    """Single source of truth for the stance-width target — used by both the
    geometric correction and the spoken explanation.

    The target is whatever the athlete's dorsiflexion and femur proportions
    imply. It used to be max(..., current_ratio + 0.15), which recommended
    widening by at least 0.15 shoulder-widths no matter what was measured —
    including for athletes already standing wider than their own target,
    where the evidence test that justified the cue returns zero.
    """
    dorsi_capacity = rom.get("peak_dorsiflexion", 35.0)
    dorsi_target_ratio, _ = dorsi_driven_targets(dorsi_capacity, anthro)
    return min(dorsi_target_ratio, 2.5)


def foot_angle_target_deg(anthro: dict, rom: dict) -> float:
    """Single source of truth for the toe-out target — used by both the
    geometric correction and the spoken explanation.

    Returns the personalized target from dorsi_driven_targets as computed.
    A max(30.0, ...) floor used to discard the bottom half of that range;
    since natural toe-out is 5-15°, that made the cue fire for nearly
    everyone and prescribed 30-40° of forced external rotation.
    """
    dorsi_capacity = rom.get("peak_dorsiflexion", 35.0)
    _, dorsi_target_angle = dorsi_driven_targets(dorsi_capacity, anthro)
    return dorsi_target_angle


def delta_widen_stance(
    features: RepKinematicSummary, anthro: dict, rom: dict
) -> dict[str, float]:
    current_ratio = features.stance_width_ratio
    shoulder_width = anthro.get("shoulder_width", 0.40)

    target_ratio = stance_target_ratio(current_ratio, anthro, rom)
    width_increase_per_side = max(
        0.0, (target_ratio - current_ratio) * shoulder_width / 2.0
    )

    return {
        "__foot_target_delta": [
            0.0, 0.0, -width_increase_per_side,
            0.0, 0.0, width_increase_per_side,
        ],
        "__target_stance_ratio": target_ratio,
        "__width_increase_per_side_m": width_increase_per_side,
    }


def magnitude_widen_stance(parameter_delta: dict) -> str | None:
    # Athletes think in how far to move each foot, not in stance ratios.
    per_side_m = parameter_delta.get("__width_increase_per_side_m")
    if per_side_m is None:
        return None
    per_side_cm = round(float(per_side_m) * 100.0)
    if per_side_cm <= 0:
        return None
    return f"each foot about {per_side_cm} {_unit(per_side_cm, 'centimeter')} wider"


def delta_widen_foot_angle(
    features: RepKinematicSummary, anthro: dict, rom: dict
) -> dict[str, float]:
    avg_current = (
        features.foot_direction_angle_l + features.foot_direction_angle_r
    ) / 2.0
    if not math.isfinite(avg_current):
        avg_current = 0.0

    target_angle = foot_angle_target_deg(anthro, rom)
    delta_degrees = max(0.0, min(target_angle - avg_current, 50.0))
    delta_radians = math.radians(delta_degrees)

    return {
        "L_ankle.ry": delta_radians,
        "R_ankle.ry": -delta_radians,
    }


def magnitude_widen_foot_angle(parameter_delta: dict) -> str | None:
    radians = parameter_delta.get("L_ankle.ry")
    if radians is None:
        return None
    degrees = round(math.degrees(abs(float(radians))))
    if degrees <= 0:
        return None
    return f"toes turned out about {degrees} more {_unit(degrees, 'degree')}"


def delta_brace_trunk(
    features: RepKinematicSummary, anthro: dict, rom: dict
) -> dict[str, float]:
    # Only the lean the athlete's own build and ankles do not explain is
    # correctable by bracing (diagnosis.lean_model).
    expected_lean = features.expected_pitch_with_ankles
    if not math.isfinite(expected_lean):
        expected_lean = features.expected_pitch_athlete
    excess_lean = features.trunk_pitch_at_bottom - expected_lean
    if not math.isfinite(excess_lean):
        excess_lean = 0.0
    correction_degrees = _clamp(excess_lean * 0.4, 0.0, 8.0)

    return {
        "trunk.rx": -math.radians(correction_degrees),
    }


def delta_knees_out(
    features: RepKinematicSummary, anthro: dict, rom: dict
) -> dict[str, float]:
    # The worse knee that was measured; a NaN side must not decide the max.
    measured = [v for v in (features.knee_valgus_l, features.knee_valgus_r) if math.isfinite(v)]
    max_valgus = max(measured) if measured else 0.0
    # Proportional to the measured valgus only. A 4° floor used to push the
    # knees out by 4° (about 3 cm per side) for a rep with 4.1° of valgus —
    # a correction as large as the fault it was correcting.
    correction_degrees = _clamp(max_valgus * 0.5, 0.0, 8.0)
    correction_radians = math.radians(correction_degrees)

    return {
        "L_hip.ry": -correction_radians,
        "R_hip.ry": correction_radians,
    }


def magnitude_knees_out(parameter_delta: dict) -> str | None:
    radians = parameter_delta.get("R_hip.ry")
    if radians is None:
        return None
    degrees = round(math.degrees(abs(float(radians))))
    if degrees <= 0:
        return None
    return f"knees pushed out about {degrees} {_unit(degrees, 'degree')} more"


def delta_center_weight(
    features: RepKinematicSummary, anthro: dict, rom: dict
) -> dict[str, float]:
    # Undo the measured hip shift (fraction of ankle separation), toward the
    # midline. Ankle separation ~ stance ratio x biacromial width.
    shift_ratio = features.hip_shift_ratio if math.isfinite(features.hip_shift_ratio) else 0.0
    stance_m = features.stance_width_ratio * anthro.get("shoulder_width", 0.40) / 0.80
    shift_meters = -shift_ratio * stance_m
    shift_meters = max(-0.04, min(0.04, shift_meters))

    return {
        "pelvis.tx": shift_meters,
    }


def magnitude_center_weight(parameter_delta: dict) -> str | None:
    shift_m = parameter_delta.get("pelvis.tx")
    if shift_m is None:
        return None
    shift_cm = round(abs(float(shift_m)) * 100.0)
    if shift_cm <= 0:
        return None
    return f"weight centered by about {shift_cm} {_unit(shift_cm, 'centimeter')}"


def delta_increase_depth(
    features: RepKinematicSummary, anthro: dict, rom: dict
) -> dict:
    return {
        "depth_target": "parallel",
    }
