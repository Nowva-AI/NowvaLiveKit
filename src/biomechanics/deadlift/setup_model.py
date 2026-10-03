"""The setup model: the right deadlift start position for this body (PLAN.md §2.7).

A planar (sagittal) linkage, ankle at the origin, forward and up in metres. The
constraints of a good conventional setup — bar over midfoot, shins touching the
bar, arms vertical, shoulder joint 0-6 cm ahead of the bar — leave the hip to the
athlete's own segment lengths. Solve 1 places the hip at setup; solve 2 places it
with the bar at knee height, which predicts how much the trunk should rise off the
floor. Hip heights the shoulder band allows form the D4 band; solve 2 minus solve 1
is the trunk change D2 compares against.
"""

from __future__ import annotations

import math
from typing import NamedTuple

from pydantic import BaseModel

# Knee flexion range of a real conventional setup; outside it the solution is the
# linkage's mirror (or a squat), not a deadlift start.
MIN_SETUP_KNEE_FLEXION_DEG = 40.0
MAX_SETUP_KNEE_FLEXION_DEG = 120.0
# Shin angle (forward of vertical) a lifter can actually reach at setup.
MIN_SHIN_ANGLE_DEG = -10.0
MAX_SHIN_ANGLE_DEG = 45.0


class AthleteSegments(BaseModel):
    """Sagittal segment lengths (m). grip_offset_m: wrist joint above bar centre."""
    tibia_m: float
    femur_m: float
    torso_m: float
    arm_m: float
    grip_offset_m: float


class SetupSolution(NamedTuple):
    hip_forward_m: float
    hip_height_m: float
    knee_forward_m: float
    knee_height_m: float
    trunk_deg: float
    knee_flexion_deg: float


class SetupPrediction(NamedTuple):
    hip_band_low_m: float
    hip_band_high_m: float
    trunk_setup_deg: float
    trunk_knee_pass_deg: float

    @property
    def trunk_change_deg(self) -> float:
        return self.trunk_knee_pass_deg - self.trunk_setup_deg


def _circle_intersections(
    centre_a: tuple[float, float], radius_a: float, centre_b: tuple[float, float], radius_b: float,
) -> list[tuple[float, float]]:
    dx = centre_b[0] - centre_a[0]
    dy = centre_b[1] - centre_a[1]
    distance = math.hypot(dx, dy)
    if distance < 1e-9 or distance > radius_a + radius_b or distance < abs(radius_a - radius_b):
        return []
    along = (radius_a ** 2 - radius_b ** 2 + distance ** 2) / (2.0 * distance)
    across = math.sqrt(max(0.0, radius_a ** 2 - along ** 2))
    base_x = centre_a[0] + along * dx / distance
    base_y = centre_a[1] + along * dy / distance
    return [
        (base_x + across * dy / distance, base_y - across * dx / distance),
        (base_x - across * dy / distance, base_y + across * dx / distance),
    ]


def _knee_flexion_deg(knee: tuple[float, float], hip: tuple[float, float]) -> float:
    shin = (knee[0], knee[1])
    thigh = (hip[0] - knee[0], hip[1] - knee[1])
    dot = shin[0] * thigh[0] + shin[1] * thigh[1]
    norms = math.hypot(*shin) * math.hypot(*thigh)
    if norms < 1e-12:
        return math.nan
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / norms))))


def _shin_angles_for_contact(
    bar_forward_m: float, bar_height_m: float, shin_bar_m: float,
) -> list[float]:
    """Shin angles (rad, forward of vertical) whose line through the ankle passes
    shin_bar_m behind the bar centre."""
    reach = math.hypot(bar_forward_m, bar_height_m)
    if reach < shin_bar_m or reach < 1e-9:
        return []
    bearing = math.atan2(bar_height_m, bar_forward_m)
    spread = math.acos(shin_bar_m / reach)
    return [spread - bearing, -spread - bearing]


def _hip_for(
    knee: tuple[float, float], shoulder: tuple[float, float], segments: AthleteSegments,
) -> tuple[float, float] | None:
    candidates = _circle_intersections(knee, segments.femur_m, shoulder, segments.torso_m)
    # The hip sits behind the knee-to-shoulder line, never in front of it.
    line_forward = shoulder[0] - knee[0]
    line_up = shoulder[1] - knee[1]
    behind = [
        hip for hip in candidates
        if line_forward * (hip[1] - knee[1]) - line_up * (hip[0] - knee[0]) > 0.0
    ]
    return behind[0] if behind else None


def _shoulder_over_grip(
    segments: AthleteSegments, bar_forward_m: float, bar_height_m: float, shoulder_ahead_m: float,
) -> tuple[float, float]:
    grip_height = bar_height_m + segments.grip_offset_m
    rise = math.sqrt(max(0.0, segments.arm_m ** 2 - shoulder_ahead_m ** 2))
    return (bar_forward_m + shoulder_ahead_m, grip_height + rise)


def _trunk_deg(hip: tuple[float, float], shoulder: tuple[float, float]) -> float:
    return math.degrees(math.atan2(shoulder[0] - hip[0], shoulder[1] - hip[1]))


def solve_setup(
    segments: AthleteSegments,
    bar_forward_m: float,
    bar_height_m: float,
    shoulder_ahead_m: float,
    shin_bar_m: float,
) -> SetupSolution | None:
    """Solve 1: the hip with shins on the bar, straight arms from the grip and the
    shoulder joint shoulder_ahead_m in front of the bar. None when no real setup
    fits this body."""
    shoulder = _shoulder_over_grip(segments, bar_forward_m, bar_height_m, shoulder_ahead_m)
    best: SetupSolution | None = None
    for shin_rad in _shin_angles_for_contact(bar_forward_m, bar_height_m, shin_bar_m):
        shin_deg = math.degrees(shin_rad)
        if not MIN_SHIN_ANGLE_DEG <= shin_deg <= MAX_SHIN_ANGLE_DEG:
            continue
        knee = (segments.tibia_m * math.sin(shin_rad), segments.tibia_m * math.cos(shin_rad))
        hip = _hip_for(knee, shoulder, segments)
        if hip is None:
            continue
        flexion = _knee_flexion_deg(knee, hip)
        if not MIN_SETUP_KNEE_FLEXION_DEG <= flexion <= MAX_SETUP_KNEE_FLEXION_DEG:
            continue
        solution = SetupSolution(
            hip_forward_m=hip[0],
            hip_height_m=hip[1],
            knee_forward_m=knee[0],
            knee_height_m=knee[1],
            trunk_deg=_trunk_deg(hip, shoulder),
            knee_flexion_deg=flexion,
        )
        # Of two contact solutions, the knee in front of the ankle is the deadlift.
        if best is None or solution.knee_forward_m > best.knee_forward_m:
            best = solution
    return best


def solve_knee_pass(
    segments: AthleteSegments,
    bar_forward_m: float,
    shoulder_ahead_m: float,
    shin_bar_m: float,
) -> SetupSolution | None:
    """Solve 2: the bar rising vertically has reached knee height, shins as far
    forward as the bar allows (near vertical)."""
    knee_forward = min(bar_forward_m - shin_bar_m, segments.tibia_m)
    shin_rad = math.asin(max(-1.0, min(1.0, knee_forward / segments.tibia_m)))
    knee = (segments.tibia_m * math.sin(shin_rad), segments.tibia_m * math.cos(shin_rad))
    shoulder = _shoulder_over_grip(segments, bar_forward_m, knee[1], shoulder_ahead_m)
    hip = _hip_for(knee, shoulder, segments)
    if hip is None:
        return None
    return SetupSolution(
        hip_forward_m=hip[0],
        hip_height_m=hip[1],
        knee_forward_m=knee[0],
        knee_height_m=knee[1],
        trunk_deg=_trunk_deg(hip, shoulder),
        knee_flexion_deg=_knee_flexion_deg(knee, hip),
    )


def predict_setup(
    segments: AthleteSegments,
    bar_forward_m: float,
    bar_height_m: float,
    shoulder_band_m: tuple[float, float],
    shoulder_ahead_m: float,
    shin_bar_m: float,
) -> SetupPrediction | None:
    """The D4 hip band (the hip heights the shoulder band allows) and the trunk
    change D2 expects, at the athlete's own shoulder offset clipped into the band."""
    edges = [
        solve_setup(segments, bar_forward_m, bar_height_m, shoulder_edge, shin_bar_m)
        for shoulder_edge in shoulder_band_m
    ]
    if any(edge is None for edge in edges):
        return None
    clipped_shoulder = min(max(shoulder_ahead_m, shoulder_band_m[0]), shoulder_band_m[1])
    if not math.isfinite(clipped_shoulder):
        clipped_shoulder = (shoulder_band_m[0] + shoulder_band_m[1]) / 2.0
    setup = solve_setup(segments, bar_forward_m, bar_height_m, clipped_shoulder, shin_bar_m)
    knee_pass = solve_knee_pass(segments, bar_forward_m, clipped_shoulder, shin_bar_m)
    if setup is None or knee_pass is None:
        return None
    heights = [edge.hip_height_m for edge in edges]
    return SetupPrediction(
        hip_band_low_m=min(heights),
        hip_band_high_m=max(heights),
        trunk_setup_deg=setup.trunk_deg,
        trunk_knee_pass_deg=knee_pass.trunk_deg,
    )
