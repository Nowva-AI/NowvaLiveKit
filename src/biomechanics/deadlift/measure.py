"""One analysis frame of a deadlift, measured in the sagittal frame (PLAN.md §2.4).

The analyser's per-frame view of the lifter and the bar: joint midpoints, the
trunk, hip, knee and elbow angles, the bar (tracked, carried over a short gap, or
the wrist proxy) and whether the hands are on it. Pure: everything the
measurement depends on from the analyser's state comes in a MeasureContext.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from biomechanics.config import DeadliftConfig
from biomechanics.diagnosis.lean_model import MIDFOOT_FRACTION_OF_ANKLE_TO_TOE
from biomechanics.utils.types import CocoKeypoints as CK

from .frame import SagittalFrame, build_sagittal_frame, forward_m, height_m, lateral_m, segment_angle_deg
from .types import BAR_SOURCE_BAR, BAR_SOURCE_WRIST_PROXY, NAN, BarState3D

# Keypoints below this confidence are missing (the IK solver's floor).
MIN_KEYPOINT_CONFIDENCE = 0.1
# The feet the plates hide during a pull; planted at the bar, so the last
# measured position stands in for them.
FOOT_KEYPOINTS = (CK.LEFT_ANKLE, CK.RIGHT_ANKLE, CK.LEFT_FOOT_INDEX, CK.RIGHT_FOOT_INDEX)


class DeadliftFrameInput(NamedTuple):
    """One lagged analysis frame. points: (N, 3) world metres, Y-down.
    legs_measured: the legs were triangulated, not carried by the Kalman."""
    timestamp: float
    frame_index: int
    points: np.ndarray
    confidences: np.ndarray
    bar: BarState3D | None
    legs_measured: bool = True


class MeasureContext(NamedTuple):
    """The analyser state a measurement depends on."""
    up: np.ndarray
    # Source of the bar's resting height: a set never mixes tracked-bar and wrist heights.
    rest_source: str | None
    rest_axis: np.ndarray | None
    grip_offset_m: float
    # The last tracked bar, for a frame inside a short tracking gap (geometry only).
    carried_bar: BarState3D | None
    # Last measured feet, standing in for feet the plates hide (None: no substitution).
    planted_feet: dict[int, np.ndarray] | None


class FrameMeasure(NamedTuple):
    t: float
    frame_index: int
    legs_measured: bool
    feet_measured: bool
    frame: SagittalFrame
    ankle_mid: np.ndarray
    midfoot: np.ndarray | None
    hip_mid: np.ndarray
    shoulder_mid: np.ndarray
    knee_mid: np.ndarray | None
    wrist_mid: np.ndarray | None
    l_wrist: np.ndarray | None
    r_wrist: np.ndarray | None
    l_ankle: np.ndarray | None
    r_ankle: np.ndarray | None
    bar_centre: np.ndarray | None
    bar_left: np.ndarray | None
    bar_right: np.ndarray | None
    bar_source: str
    # The bar's height was measured on this frame (not carried over a gap, not a
    # tracker prediction): only these frames enter the setup features.
    bar_measured: bool
    bar_up: float
    trunk_deg: float
    hip_flex_deg: float
    knee_flex_deg: float
    elbow_flex_deg: float
    arm_m: float
    hands_on_bar: bool
    standing: bool
    foot_forward: np.ndarray | None
    points: np.ndarray


def _point(points: np.ndarray, confidences: np.ndarray, index: int) -> np.ndarray | None:
    if index >= len(points) or confidences[index] < MIN_KEYPOINT_CONFIDENCE:
        return None
    return points[index]


def _midpoint(first: np.ndarray | None, second: np.ndarray | None) -> np.ndarray | None:
    if first is None or second is None:
        return first if second is None else second
    return (first + second) / 2.0


def _strict_midpoint(first: np.ndarray | None, second: np.ndarray | None) -> np.ndarray | None:
    if first is None or second is None:
        return None
    return (first + second) / 2.0


# Elbow bend seen from the side. In 3D the angle also changes with grip width
# (hands out on the bar vs hanging), which is not a bent arm.
def _elbow_flexion_deg(
    frame: SagittalFrame, shoulder: np.ndarray | None, elbow: np.ndarray | None, wrist: np.ndarray | None,
) -> float:
    if shoulder is None or elbow is None or wrist is None:
        return NAN
    upper_deg = segment_angle_deg(frame, elbow, shoulder)
    fore_deg = segment_angle_deg(frame, wrist, elbow)
    return abs(upper_deg - fore_deg)


def _arm_length_m(shoulder: np.ndarray | None, elbow: np.ndarray | None, wrist: np.ndarray | None) -> float:
    if shoulder is None or elbow is None or wrist is None:
        return NAN
    return float(np.linalg.norm(elbow - shoulder) + np.linalg.norm(wrist - elbow))


def _nanmean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else NAN


def feet_measured(confidences: np.ndarray) -> dict[int, bool]:
    return {
        index: index < len(confidences) and confidences[index] >= MIN_KEYPOINT_CONFIDENCE
        for index in FOOT_KEYPOINTS
    }


def hands_on_bar(
    frame: SagittalFrame,
    bar_source: str,
    bar_centre: np.ndarray | None,
    wrists: list[np.ndarray],
    knee_mid: np.ndarray | None,
    trunk_deg: float,
    config: DeadliftConfig,
) -> bool:
    if bar_centre is None or not wrists:
        return False
    if bar_source == BAR_SOURCE_WRIST_PROXY:
        # The proxy bar is the wrists, so "hands on the bar" is a hinge with
        # the hands below the knees.
        if knee_mid is None or trunk_deg < config.wrist_proxy_min_hinge_deg:
            return False
        return all(height_m(frame, wrist, knee_mid) < 0.0 for wrist in wrists)
    for wrist in wrists:
        above = height_m(frame, wrist, bar_centre)
        if not -config.hands_max_below_bar_m <= above <= config.hands_max_above_bar_m:
            return False
        if abs(forward_m(frame, wrist, bar_centre)) > config.hands_max_forward_m:
            return False
    return True


def measure_frame(
    frame_input: DeadliftFrameInput, context: MeasureContext, config: DeadliftConfig,
) -> FrameMeasure | None:
    """None when the hips, shoulders or (unplanted) feet are missing."""
    points = np.asarray(frame_input.points, dtype=np.float64)
    confidences = np.asarray(frame_input.confidences, dtype=np.float64)
    get = lambda index: _point(points, confidences, index)  # noqa: E731
    up = context.up

    measured_feet = feet_measured(confidences)
    foot_points = {index: get(index) for index in FOOT_KEYPOINTS}
    if context.planted_feet is not None:
        for index, planted in context.planted_feet.items():
            if foot_points[index] is None:
                foot_points[index] = planted

    l_hip, r_hip = get(CK.LEFT_HIP), get(CK.RIGHT_HIP)
    l_shoulder, r_shoulder = get(CK.LEFT_SHOULDER), get(CK.RIGHT_SHOULDER)
    l_ankle, r_ankle = foot_points[CK.LEFT_ANKLE], foot_points[CK.RIGHT_ANKLE]
    hip_mid = _strict_midpoint(l_hip, r_hip)
    shoulder_mid = _midpoint(l_shoulder, r_shoulder)
    ankle_mid = _midpoint(l_ankle, r_ankle)
    if hip_mid is None or shoulder_mid is None or ankle_mid is None:
        return None
    l_knee, r_knee = get(CK.LEFT_KNEE), get(CK.RIGHT_KNEE)
    knee_mid = _midpoint(l_knee, r_knee)
    l_elbow, r_elbow = get(CK.LEFT_ELBOW), get(CK.RIGHT_ELBOW)
    l_wrist, r_wrist = get(CK.LEFT_WRIST), get(CK.RIGHT_WRIST)
    wrist_mid = _midpoint(l_wrist, r_wrist)

    toe_vectors = []
    midfoot_points = []
    for ankle, toe_index in ((l_ankle, CK.LEFT_FOOT_INDEX), (r_ankle, CK.RIGHT_FOOT_INDEX)):
        toe = foot_points[toe_index]
        if ankle is None or toe is None:
            continue
        foot = toe - ankle
        foot = foot - float(np.dot(foot, up)) * up
        toe_vectors.append(foot)
        midfoot_points.append(ankle + MIDFOOT_FRACTION_OF_ANKLE_TO_TOE * foot)
    toe_direction = np.sum(toe_vectors, axis=0) if toe_vectors else None
    midfoot = np.mean(midfoot_points, axis=0) if len(midfoot_points) == 2 else None

    bar = frame_input.bar
    carried = bar is None and context.carried_bar is not None
    if carried:
        bar = context.carried_bar
    lateral_hint = r_hip - l_hip
    if bar is not None:
        bar_axis = -bar.axis
        if float(np.dot(bar_axis, lateral_hint)) < 0.0:
            bar_axis = -bar_axis
        lateral_hint = bar_axis
    elif context.rest_axis is not None:
        lateral_hint = context.rest_axis
    frame = build_sagittal_frame(up, lateral_hint, toe_direction)
    if frame is None:
        return None

    foot_forward = None
    if toe_direction is not None:
        norm = float(np.linalg.norm(toe_direction))
        if norm > 1e-6:
            foot_forward = toe_direction / norm

    trunk_deg = segment_angle_deg(frame, hip_mid, shoulder_mid)
    hip_flex = NAN
    knee_flex = NAN
    if knee_mid is not None:
        thigh_deg = segment_angle_deg(frame, knee_mid, hip_mid)
        shin_deg = segment_angle_deg(frame, ankle_mid, knee_mid)
        hip_flex = trunk_deg - thigh_deg
        knee_flex = shin_deg - thigh_deg

    elbow_flex = _nanmean([
        _elbow_flexion_deg(frame, l_shoulder, l_elbow, l_wrist),
        _elbow_flexion_deg(frame, r_shoulder, r_elbow, r_wrist),
    ])
    arm_m = _nanmean([
        _arm_length_m(l_shoulder, l_elbow, l_wrist),
        _arm_length_m(r_shoulder, r_elbow, r_wrist),
    ])

    # Heights are only comparable with a rest measured from the same source:
    # once a set runs on the wrists it stays on them, and a set on the tracked
    # bar has no bar on a frame the bar is lost rather than a jump to the wrists.
    use_bar = bar is not None and context.rest_source != BAR_SOURCE_WRIST_PROXY
    bar_centre = bar_left = bar_right = None
    bar_source = BAR_SOURCE_BAR
    bar_measured = False
    bar_up = NAN
    if use_bar:
        bar_centre = bar.centre
        bar_left = np.asarray(bar.left_end_m, dtype=np.float64)
        bar_right = np.asarray(bar.right_end_m, dtype=np.float64)
        # The subject's left end, whatever the tracker called it: the world's X
        # is only the subject's left on a world-anchored calibration.
        if lateral_m(frame, bar_left, bar_right) > 0.0:
            bar_left, bar_right = bar_right, bar_left
        # A prediction (the tracker carrying the last velocity on) keeps the bar's
        # geometry but gives no height: coasting into a top, it overshoots a bar
        # that stops.
        bar_measured = not carried and not bar.predicted
        if bar_measured:
            bar_up = float(np.dot(bar_centre, frame.up))
    elif context.rest_source != BAR_SOURCE_BAR and wrist_mid is not None:
        bar_source = BAR_SOURCE_WRIST_PROXY
        bar_centre = wrist_mid - context.grip_offset_m * frame.up
        bar_measured = True
        bar_up = float(np.dot(bar_centre, frame.up))

    standing = (
        math.isfinite(knee_flex)
        and abs(knee_flex) <= config.standing_max_knee_deg
        and abs(trunk_deg) <= config.standing_max_trunk_deg
    )
    wrists = [wrist for wrist in (l_wrist, r_wrist) if wrist is not None]
    return FrameMeasure(
        t=frame_input.timestamp,
        frame_index=frame_input.frame_index,
        legs_measured=frame_input.legs_measured,
        feet_measured=all(measured_feet.values()),
        frame=frame,
        ankle_mid=ankle_mid,
        midfoot=midfoot,
        hip_mid=hip_mid,
        shoulder_mid=shoulder_mid,
        knee_mid=knee_mid,
        wrist_mid=wrist_mid,
        l_wrist=l_wrist,
        r_wrist=r_wrist,
        l_ankle=l_ankle,
        r_ankle=r_ankle,
        bar_centre=bar_centre,
        bar_left=bar_left,
        bar_right=bar_right,
        bar_source=bar_source,
        bar_measured=bar_measured,
        bar_up=bar_up,
        trunk_deg=trunk_deg,
        hip_flex_deg=hip_flex,
        knee_flex_deg=knee_flex,
        elbow_flex_deg=elbow_flex,
        arm_m=arm_m,
        hands_on_bar=hands_on_bar(frame, bar_source, bar_centre, wrists, knee_mid, trunk_deg, config),
        standing=standing,
        foot_forward=foot_forward,
        points=points,
    )
