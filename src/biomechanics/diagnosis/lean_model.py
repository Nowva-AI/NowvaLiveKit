"""Sagittal balance model: the trunk pitch a squat needs to keep the shoulders over midfoot.

Trunk angle is not a free choice in a squat — it is whatever keeps the bar over
midfoot, given the athlete's femur, tibia and torso lengths, how far the knees
travel forward (ankle dorsiflexion) and how deep they sit. The load point is
the shoulders, where a back-squat bar sits; this reproduces published back-squat
trunk angles (~30-45° at parallel). A bodyweight squatter's arms counterbalance,
so they can sit somewhat more upright — which makes this an upper bound, and
lean beyond it is never explained by anatomy.
"""

from __future__ import annotations

import math

from biomechanics.utils.segment_lengths import REFERENCE_FEMUR_TO_TORSO_RATIO

# Midfoot sits this fraction of the ankle→toe-tip distance ahead of the ankle
# (half of a ~26 cm heel-to-toe foot whose ankle sits ~6 cm in front of the heel).
MIDFOOT_FRACTION_OF_ANKLE_TO_TOE = 0.35

# Shank tilt (deg from vertical) of an unrestricted ankle at the bottom of a squat.
UNRESTRICTED_SHANK_DEG = 30.0
# The reference lifter's tibia is as long as their femur.
REFERENCE_TIBIA_TO_FEMUR_RATIO = 1.0
MAX_PITCH_DEG = 80.0


def balanced_trunk_pitch_deg(
    femur_m: float,
    tibia_m: float,
    torso_m: float,
    shank_deg: float,
    depth_ratio: float,
    foot_len_m: float,
) -> float:
    """Trunk pitch from vertical (deg) that puts the shoulders over midfoot.

    ``depth_ratio`` is hip height above the knee in femur lengths (0 = parallel);
    ``foot_len_m`` is the ankle-to-toe length. Ankle at x = 0, forward positive.
    """
    knee_x = tibia_m * math.sin(math.radians(shank_deg))
    thigh_above = math.asin(max(-1.0, min(1.0, depth_ratio)))
    hip_x = knee_x - femur_m * math.cos(thigh_above)
    midfoot_x = MIDFOOT_FRACTION_OF_ANKLE_TO_TOE * foot_len_m
    reach = (midfoot_x - hip_x) / torso_m
    return math.degrees(math.asin(max(0.0, min(math.sin(math.radians(MAX_PITCH_DEG)), reach))))


def expected_pitches(anthro: dict, depth_ratio: float, shank_deg: float) -> tuple[float, float, float]:
    """(reference, athlete, with_ankles) balanced trunk pitch at this rep's depth.

    reference: population proportions, unrestricted ankles.
    athlete: this athlete's segments, unrestricted ankles.
    with_ankles: this athlete's segments and their measured shank tilt.
    The steps between them attribute lean to anatomy and to the ankles.
    """
    nan = float("nan")
    torso = anthro.get("torso_length")
    femur = anthro.get("femur_length_avg")
    tibia = anthro.get("tibia_length_avg")
    foot = anthro.get("foot_length", 0.20)
    if not torso or not femur or not tibia or not math.isfinite(depth_ratio):
        return nan, nan, nan

    reference_femur = REFERENCE_FEMUR_TO_TORSO_RATIO * torso
    reference = balanced_trunk_pitch_deg(
        reference_femur, reference_femur * REFERENCE_TIBIA_TO_FEMUR_RATIO, torso,
        UNRESTRICTED_SHANK_DEG, depth_ratio, foot,
    )
    athlete = balanced_trunk_pitch_deg(femur, tibia, torso, UNRESTRICTED_SHANK_DEG, depth_ratio, foot)
    with_ankles = nan
    if math.isfinite(shank_deg):
        with_ankles = balanced_trunk_pitch_deg(femur, tibia, torso, shank_deg, depth_ratio, foot)
    return reference, athlete, with_ankles
