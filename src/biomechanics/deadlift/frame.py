"""The deadlift's sagittal frame (PLAN.md §2.1).

up = measured gravity (or the body vertical as a fallback); lateral = the resting
bar's axis, or the hip line, made horizontal and pointing from the subject's left
to their right; forward = heel to toe. Every deadlift metric is a difference in
this frame, so none depends on where the floor is. Unlike the squat's unsigned
trunk pitch, angles here are signed: forward of vertical is positive.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from biomechanics.utils.geometry import WORLD_UP

MIN_AXIS_NORM = 1e-6


class SagittalFrame(NamedTuple):
    up: np.ndarray
    lateral: np.ndarray
    forward: np.ndarray


def _horizontal_unit(vector: np.ndarray, up: np.ndarray) -> np.ndarray | None:
    flat = vector - float(np.dot(vector, up)) * up
    norm = float(np.linalg.norm(flat))
    if norm < MIN_AXIS_NORM:
        return None
    return flat / norm


def build_sagittal_frame(
    up: np.ndarray,
    lateral_hint: np.ndarray,
    toe_direction: np.ndarray | None = None,
) -> SagittalFrame | None:
    """lateral_hint points from the subject's left to their right; toe_direction
    (heel or ankle to toe) fixes which way is forward. None when the hint is vertical."""
    up_unit = np.asarray(up, dtype=np.float64)
    up_unit = up_unit / float(np.linalg.norm(up_unit))
    lateral = _horizontal_unit(np.asarray(lateral_hint, dtype=np.float64), up_unit)
    if lateral is None:
        return None
    forward = np.cross(up_unit, lateral)
    if toe_direction is not None and float(np.dot(forward, toe_direction)) < 0.0:
        forward = -forward
        lateral = -lateral
    return SagittalFrame(up=up_unit, lateral=lateral, forward=forward)


def default_up() -> np.ndarray:
    return np.array(WORLD_UP, dtype=np.float64)


def height_m(frame: SagittalFrame, point: np.ndarray, reference: np.ndarray) -> float:
    """Signed height of point above reference along up."""
    return float(np.dot(point - reference, frame.up))


def forward_m(frame: SagittalFrame, point: np.ndarray, reference: np.ndarray) -> float:
    """Signed horizontal offset of point ahead of reference (toward the toes)."""
    return float(np.dot(point - reference, frame.forward))


def lateral_m(frame: SagittalFrame, point: np.ndarray, reference: np.ndarray) -> float:
    """Signed horizontal offset of point toward the subject's right of reference."""
    return float(np.dot(point - reference, frame.lateral))


def segment_angle_deg(frame: SagittalFrame, lower: np.ndarray, upper: np.ndarray) -> float:
    """Signed sagittal angle of the segment lower→upper from up: positive when the
    upper end is ahead of the lower one (trunk leaning forward, shin tilted forward)."""
    segment = upper - lower
    return math.degrees(math.atan2(float(np.dot(segment, frame.forward)), float(np.dot(segment, frame.up))))
