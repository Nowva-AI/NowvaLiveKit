"""Parameter deltas for the deadlift's tier-1 causes. Each returns exactly one
number: the change to make, in its key's unit (_cm, _deg, _pct), positive in the
direction its key names (only feet_toward_bar_cm and bar_toward_shins_cm can be
negative). deadlift.diagnosis fills the explanations from these same functions,
so what is said and what is recorded agree."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from biomechanics.deadlift.diagnosis import DeadliftRepSummary

DELTA_DECIMALS = 1
PERCENT = 100.0


def _measured_or_zero(value: float) -> float:
    return round(value, DELTA_DECIMALS) if math.isfinite(value) else 0.0


def _positive_part(value: float) -> float:
    return max(0.0, _measured_or_zero(value))


def delta_feet_position(
    features: DeadliftRepSummary, anthro: dict, rom: dict
) -> dict[str, float]:
    # The feet set the bar's offset while standing at it; a bar ahead of the
    # midfoot (+) needs the feet that much closer.
    offset = features.bar_midfoot_stance_cm
    if not math.isfinite(offset):
        offset = features.bar_midfoot_setup_cm
    return {"feet_toward_bar_cm": _measured_or_zero(offset)}


def delta_setup_habit(
    features: DeadliftRepSummary, anthro: dict, rom: dict
) -> dict[str, float]:
    # How far the bar rolled away (+) or was dragged in (-) while getting set.
    roll = features.bar_midfoot_setup_cm - features.bar_midfoot_stance_cm
    return {"bar_toward_shins_cm": _measured_or_zero(roll)}


def delta_hips_up(
    features: DeadliftRepSummary, anthro: dict, rom: dict
) -> dict[str, float]:
    below = features.setup_hip_band_low_cm - features.setup_hip_height_cm
    return {"hips_up_cm": _positive_part(below)}


def delta_hips_down(
    features: DeadliftRepSummary, anthro: dict, rom: dict
) -> dict[str, float]:
    above = features.setup_hip_height_cm - features.setup_hip_band_high_cm
    return {"hips_down_cm": _positive_part(above)}


def delta_shoulders_forward(
    features: DeadliftRepSummary, anthro: dict, rom: dict
) -> dict[str, float]:
    return {"shoulders_forward_cm": _positive_part(-features.shoulder_vs_bar_cm)}


def delta_chest_up(
    features: DeadliftRepSummary, anthro: dict, rom: dict
) -> dict[str, float]:
    # Trunk tipped forward off the floor beyond what the setup model predicts.
    return {"chest_up_deg": _positive_part(features.trunk_change_liftoff_knee_deg)}


def delta_bar_start_closer(
    features: DeadliftRepSummary, anthro: dict, rom: dict
) -> dict[str, float]:
    return {"bar_start_closer_cm": _positive_part(features.bar_midfoot_setup_cm)}


def delta_bar_closer(
    features: DeadliftRepSummary, anthro: dict, rom: dict
) -> dict[str, float]:
    return {"bar_closer_cm": _positive_part(features.bar_drift_cm)}


def delta_stand_taller(
    features: DeadliftRepSummary, anthro: dict, rom: dict
) -> dict[str, float]:
    deficits = [
        value
        for value in (features.hip_extension_deficit_deg, features.knee_extension_deficit_deg)
        if math.isfinite(value)
    ]
    return {"stand_taller_deg": _positive_part(max(deficits) if deficits else math.nan)}


def delta_less_lean_back(
    features: DeadliftRepSummary, anthro: dict, rom: dict
) -> dict[str, float]:
    return {"lean_back_less_deg": _positive_part(features.lean_back_deg)}


def delta_hips_to_centre(
    features: DeadliftRepSummary, anthro: dict, rom: dict
) -> dict[str, float]:
    # The hips' sideways slide, as a percentage of the ankle separation.
    return {"hips_to_centre_pct": _positive_part(abs(features.hip_shift_ratio) * PERCENT)}


def delta_level_bar(
    features: DeadliftRepSummary, anthro: dict, rom: dict
) -> dict[str, float]:
    # The tilt to take out. A grip shift in cm would need the grip's offset on
    # the bar, which DeadliftRepFeatures does not measure yet.
    return {"bar_level_cm": _positive_part(abs(features.bar_tilt_cm))}
