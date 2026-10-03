"""
Synthetic barbell RDL poses in the production frame (docs/exercises/rdl.md §4.4)
and checks that they reproduce the plan's geometry: the hip-sink table across
knee flexion and shin angle, and the shoulder drop. Other RDL tests import the
generators from here.

Frame: Y-down metres, X = the lifter's left, forward = -Z, ankles on y = 0.
Winter proportions; the bar stays on the legs (wrists over the shin front)
unless pushed forward, and the arms reach it from the shoulders.
"""

from __future__ import annotations

import math
from dataclasses import astuple, dataclass
from typing import Callable

import numpy as np
import pytest

from biomechanics.utils.geometry import level_camera_frame
from biomechanics.utils.types import CocoKeypoints as CK

TALL_HEIGHT_M = 1.885
SHORT_HEIGHT_M = 1.60
NUM_KEYPOINTS = 21
TIBIA_RATIO = 0.246
FEMUR_RATIO = 0.245
TORSO_RATIO = 0.288
ARM_RATIO = 0.332
HIP_HALF_WIDTH_RATIO = 0.064
SHOULDER_HALF_WIDTH_RATIO = 0.10
GRIP_HALF_WIDTH_RATIO = 0.13
TOE_FORWARD_RATIO = 0.10
HEEL_BACK_RATIO = 0.035
HEAD_RATIO = 0.12
# The bar touching the shin front sits 0.2 ankle-to-toe lengths ahead of the
# ankle, i.e. -0.15 from midfoot (rdl.md §5.2).
BAR_ON_SHIN_FOOT_FRACTION = 0.2
STANDING_KNEE_FLEX_DEG = 3.0
STANDING_HINGE_DEG = 0.0

# The plan's hip-sink table (rdl.md §4.4), percent, vs a 3° standing knee.
HIP_SINK_TABLE = {
    # (knee flexion, shin inclination): hip sink %
    (33.0, 0.0): 8.0,
    (45.0, 0.0): 14.6,
    (60.0, 0.0): 24.9,
    (40.0, 10.0): 7.4,
    (60.0, 15.0): 16.3,
    (40.0, 20.0): 6.0,
    (60.0, 30.0): 13.3,
}
HIP_SINK_TABLE_TOLERANCE_PCT = 0.15
SHOULDER_DROP_TOLERANCE_CM = 2.0


def _unit_up() -> np.ndarray:
    return np.array([0.0, -1.0, 0.0])


def _unit_forward() -> np.ndarray:
    return np.array([0.0, 0.0, -1.0])


def world_rdl_points(
    hinge_deg: float,
    knee_flex_deg: float = STANDING_KNEE_FLEX_DEG,
    shank_deg: float = 0.0,
    tilt_deg: float = 0.0,
    bar_forward_m: float = 0.0,
    height_m: float = TALL_HEIGHT_M,
    yaw_deg: float = 0.0,
) -> np.ndarray:
    """(21, 3) RDL pose. hinge_deg tilts the trunk forward from vertical
    (negative leans back), knee_flex_deg bends the knee, shank_deg inclines the
    shin forward, tilt_deg drops the right hand, bar_forward_m moves the bar off
    the legs, yaw_deg turns the lifter about the vertical through the ankles."""
    up, forward = _unit_up(), _unit_forward()
    tibia, femur, torso = TIBIA_RATIO * height_m, FEMUR_RATIO * height_m, TORSO_RATIO * height_m
    arm = ARM_RATIO * height_m
    shank = math.radians(shank_deg)
    thigh = math.radians(knee_flex_deg - shank_deg)
    trunk = math.radians(hinge_deg)
    knee = tibia * (math.cos(shank) * up + math.sin(shank) * forward)
    hip = knee + femur * (math.cos(thigh) * up - math.sin(thigh) * forward)
    shoulder = hip + torso * (math.cos(trunk) * up + math.sin(trunk) * forward)
    toe_forward = TOE_FORWARD_RATIO * height_m
    bar_forward = BAR_ON_SHIN_FOOT_FRACTION * toe_forward + bar_forward_m

    points = np.zeros((NUM_KEYPOINTS, 3))
    sides = (
        (1.0, CK.LEFT_HIP, CK.LEFT_KNEE, CK.LEFT_ANKLE, CK.LEFT_SHOULDER, CK.LEFT_ELBOW,
         CK.LEFT_WRIST, CK.LEFT_FOOT_INDEX, CK.LEFT_HEEL),
        (-1.0, CK.RIGHT_HIP, CK.RIGHT_KNEE, CK.RIGHT_ANKLE, CK.RIGHT_SHOULDER, CK.RIGHT_ELBOW,
         CK.RIGHT_WRIST, CK.RIGHT_FOOT_INDEX, CK.RIGHT_HEEL),
    )
    lateral_reach = (GRIP_HALF_WIDTH_RATIO - SHOULDER_HALF_WIDTH_RATIO) * height_m
    shoulder_forward = float(np.dot(shoulder, forward))
    forward_reach = shoulder_forward - bar_forward
    arm_drop = math.sqrt(max(arm ** 2 - lateral_reach ** 2 - forward_reach ** 2, 0.0))
    for sign, hip_i, knee_i, ankle_i, shoulder_i, elbow_i, wrist_i, toe_i, heel_i in sides:
        lateral = np.array([sign * HIP_HALF_WIDTH_RATIO * height_m, 0.0, 0.0])
        points[ankle_i] = lateral
        points[knee_i] = lateral + knee
        points[hip_i] = lateral + hip
        points[shoulder_i] = shoulder + [sign * SHOULDER_HALF_WIDTH_RATIO * height_m, 0.0, 0.0]
        tilt_rise = sign * GRIP_HALF_WIDTH_RATIO * height_m * math.sin(math.radians(tilt_deg))
        wrist_height = float(np.dot(shoulder, up)) - arm_drop + tilt_rise
        points[wrist_i] = [sign * GRIP_HALF_WIDTH_RATIO * height_m, -wrist_height, -bar_forward]
        points[elbow_i] = (points[shoulder_i] + points[wrist_i]) / 2.0 + [sign * 0.04, 0.0, 0.03]
        points[toe_i] = lateral + [sign * 0.02, 0.0, -toe_forward]
        points[heel_i] = lateral + [0.0, 0.0, HEEL_BACK_RATIO * height_m]
    head = shoulder + HEAD_RATIO * height_m * (math.cos(trunk) * up + math.sin(trunk) * forward)
    points[CK.NOSE] = head + 0.03 * forward
    points[CK.LEFT_EYE] = head + [0.03, -0.02, -0.02]
    points[CK.RIGHT_EYE] = head + [-0.03, -0.02, -0.02]
    points[CK.LEFT_EAR] = head + [0.07, 0.0, 0.03]
    points[CK.RIGHT_EAR] = head + [-0.07, 0.0, 0.03]
    if yaw_deg:
        yaw = math.radians(yaw_deg)
        rotation = np.array([
            [math.cos(yaw), 0.0, math.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-math.sin(yaw), 0.0, math.cos(yaw)],
        ])
        points = points @ rotation.T
    return points


def dropping_hand(degrees: float, height_m: float = TALL_HEIGHT_M) -> Callable[..., np.ndarray]:
    """A view of a pose whose right hand (negative: left) drops as the
    shoulders go down, by `degrees` of bar tilt at the clean bottom; with
    constant=True the drop is there from the start (a grip asymmetry)."""
    standing_cm = height_cm(STANDING.points(height_m), CK.LEFT_SHOULDER, CK.RIGHT_SHOULDER)
    bottom_cm = height_cm(CLEAN_BOTTOM.points(height_m), CK.LEFT_SHOULDER, CK.RIGHT_SHOULDER)

    def view(points: np.ndarray, constant: bool = False) -> np.ndarray:
        shoulder_cm = height_cm(points, CK.LEFT_SHOULDER, CK.RIGHT_SHOULDER)
        depth = 1.0 if constant else min(1.0, max(0.0, (standing_cm - shoulder_cm) / (standing_cm - bottom_cm)))
        half_grip_m = abs(points[CK.LEFT_WRIST][0] - points[CK.RIGHT_WRIST][0]) / 2.0
        rise_m = half_grip_m * math.sin(math.radians(degrees * depth ** 2))
        dropped = points.copy()
        dropped[CK.LEFT_WRIST, 1] -= rise_m
        dropped[CK.RIGHT_WRIST, 1] += rise_m
        return dropped

    return view


def camera_view(points: np.ndarray, pitch_deg: float) -> np.ndarray:
    """The world pose as a camera pitched up by pitch_deg sees it (inverse of levelling)."""
    return level_camera_frame(points, -pitch_deg)


def height_cm(points: np.ndarray, index_l: int, index_r: int) -> float:
    """Height of a keypoint pair's midpoint above the ankle midpoint, Y-down."""
    ankle_y = (points[CK.LEFT_ANKLE][1] + points[CK.RIGHT_ANKLE][1]) / 2.0
    return (ankle_y - (points[index_l][1] + points[index_r][1]) / 2.0) * 100.0


def hip_sink_pct(knee_flex_deg: float, shank_deg: float, height_m: float = TALL_HEIGHT_M) -> float:
    standing = world_rdl_points(STANDING_HINGE_DEG, height_m=height_m)
    bottom = world_rdl_points(60.0, knee_flex_deg, shank_deg, height_m=height_m)
    standing_cm = height_cm(standing, CK.LEFT_HIP, CK.RIGHT_HIP)
    return (standing_cm - height_cm(bottom, CK.LEFT_HIP, CK.RIGHT_HIP)) / standing_cm * 100.0


def bar_travel(hinge_deg: float, knee_flex_deg: float, shank_deg: float, height_m: float = TALL_HEIGHT_M) -> float:
    """Hand travel toward the knees of a noise-free pose (0 at hang, 1 at the knee)."""
    standing = world_rdl_points(STANDING_HINGE_DEG, height_m=height_m)
    bottom = world_rdl_points(hinge_deg, knee_flex_deg, shank_deg, height_m=height_m)
    wrist_ref = height_cm(standing, CK.LEFT_WRIST, CK.RIGHT_WRIST)
    knee_ref = height_cm(standing, CK.LEFT_KNEE, CK.RIGHT_KNEE)
    return (wrist_ref - height_cm(bottom, CK.LEFT_WRIST, CK.RIGHT_WRIST)) / (wrist_ref - knee_ref)


def hinge_for_travel(
    target_travel: float, knee_flex_deg: float, shank_deg: float, height_m: float = TALL_HEIGHT_M,
) -> float:
    """The hinge whose noise-free hand travel is target_travel (bisection)."""
    low_deg, high_deg = 0.0, 90.0
    for _ in range(60):
        middle = (low_deg + high_deg) / 2.0
        if bar_travel(middle, knee_flex_deg, shank_deg, height_m) < target_travel:
            low_deg = middle
        else:
            high_deg = middle
    return (low_deg + high_deg) / 2.0


def cosine_fractions(seconds: float, fps: float) -> list[float]:
    """0 -> 1 on a cosine profile, one value per frame, ending at 1."""
    frames = max(1, int(round(seconds * fps)))
    return [0.5 * (1.0 - math.cos(math.pi * (i + 1) / frames)) for i in range(frames)]


def cosine_crossing_time_s(seconds: float) -> float:
    """5-95% time of a cosine move lasting `seconds`."""
    start = math.acos(1.0 - 2.0 * 0.05) / math.pi
    end = math.acos(1.0 - 2.0 * 0.95) / math.pi
    return (end - start) * seconds


@dataclass(frozen=True)
class RdlPose:
    """The generator's arguments for one pose."""
    hinge_deg: float = STANDING_HINGE_DEG
    knee_flex_deg: float = STANDING_KNEE_FLEX_DEG
    shank_deg: float = 0.0
    tilt_deg: float = 0.0
    bar_forward_m: float = 0.0

    def blend(self, other: RdlPose, fraction: float) -> RdlPose:
        return RdlPose(*(mine + (theirs - mine) * fraction for mine, theirs in zip(astuple(self), astuple(other))))

    def points(self, height_m: float = TALL_HEIGHT_M, yaw_deg: float = 0.0) -> np.ndarray:
        return world_rdl_points(*astuple(self), height_m=height_m, yaw_deg=yaw_deg)


STANDING = RdlPose()
CLEAN_BOTTOM = RdlPose(75.0, 20.0, 5.0)


def _shoulder_cm(pose: RdlPose, height_m: float) -> float:
    return height_cm(pose.points(height_m), CK.LEFT_SHOULDER, CK.RIGHT_SHOULDER)


def pose_at_drop(start: RdlPose, end: RdlPose, drop_fraction: float, height_m: float = TALL_HEIGHT_M) -> RdlPose:
    """The blend of start and end whose shoulders have dropped drop_fraction of the way."""
    start_cm, end_cm = _shoulder_cm(start, height_m), _shoulder_cm(end, height_m)
    target_cm = start_cm + (end_cm - start_cm) * drop_fraction
    low, high = 0.0, 1.0
    for _ in range(40):
        middle = (low + high) / 2.0
        if (_shoulder_cm(start.blend(end, middle), height_m) - target_cm) * (end_cm - start_cm) < 0.0:
            low = middle
        else:
            high = middle
    return start.blend(end, (low + high) / 2.0)


def cosine_move(
    start: RdlPose, end: RdlPose, seconds: float, fps: float, height_m: float = TALL_HEIGHT_M,
) -> list[RdlPose]:
    """Poses whose shoulder height follows a cosine from start to end."""
    return [pose_at_drop(start, end, fraction, height_m) for fraction in cosine_fractions(seconds, fps)]


def rdl_rep(
    fps: float,
    bottom: RdlPose = CLEAN_BOTTOM,
    stand_s: float = 1.5,
    down_s: float = 2.0,
    hold_s: float = 0.3,
    up_s: float = 1.2,
    height_m: float = TALL_HEIGHT_M,
) -> list[RdlPose]:
    """Stand, a cosine shoulder descent, a hold, a cosine ascent."""
    poses = [STANDING] * int(round(stand_s * fps))
    poses += cosine_move(STANDING, bottom, down_s, fps, height_m)
    poses += [bottom] * int(round(hold_s * fps))
    poses += cosine_move(bottom, STANDING, up_s, fps, height_m)
    return poses


class TestHipSinkGeometry:
    @pytest.mark.parametrize("pose, expected_pct", list(HIP_SINK_TABLE.items()))
    def test_hip_sink_matches_the_plan_table(self, pose: tuple[float, float], expected_pct: float):
        assert hip_sink_pct(*pose) == pytest.approx(expected_pct, abs=HIP_SINK_TABLE_TOLERANCE_PCT)

    @pytest.mark.parametrize("height_m", [TALL_HEIGHT_M, SHORT_HEIGHT_M])
    def test_hip_sink_is_scale_free(self, height_m: float):
        assert hip_sink_pct(33.0, 0.0, height_m) == pytest.approx(hip_sink_pct(33.0, 0.0), abs=1e-9)

    def test_inclining_the_shin_reduces_hip_sink_at_a_fixed_knee_angle(self):
        assert hip_sink_pct(40.0, 0.0) > hip_sink_pct(40.0, 10.0) > hip_sink_pct(40.0, 20.0)


class TestShoulderDrop:
    def test_standing_and_bottom_shoulder_heights_match_the_plan(self):
        standing = world_rdl_points(STANDING_HINGE_DEG)
        bottom = world_rdl_points(75.0, 20.0, 5.0)
        assert height_cm(standing, CK.LEFT_SHOULDER, CK.RIGHT_SHOULDER) == pytest.approx(
            147.0, abs=SHOULDER_DROP_TOLERANCE_CM,
        )
        assert height_cm(bottom, CK.LEFT_SHOULDER, CK.RIGHT_SHOULDER) == pytest.approx(
            105.0, abs=SHOULDER_DROP_TOLERANCE_CM,
        )

    def test_hand_travel_grows_with_the_hinge(self):
        travels = [bar_travel(hinge, 20.0, 5.0) for hinge in (25.0, 45.0, 60.0, 75.0)]
        assert travels == sorted(travels)
        assert bar_travel(75.0, 20.0, 5.0) > 0.45

    def test_the_travel_solver_inverts_bar_travel(self):
        hinge = hinge_for_travel(0.35, 20.0, 5.0)
        assert bar_travel(hinge, 20.0, 5.0) == pytest.approx(0.35, abs=1e-6)


class TestRepGenerator:
    def test_shoulder_height_follows_a_cosine_descent(self):
        fps = 30.0
        poses = cosine_move(STANDING, CLEAN_BOTTOM, 2.0, fps)
        top_cm, low_cm = _shoulder_cm(STANDING, TALL_HEIGHT_M), _shoulder_cm(CLEAN_BOTTOM, TALL_HEIGHT_M)
        for fraction, pose in zip(cosine_fractions(2.0, fps), poses):
            assert _shoulder_cm(pose, TALL_HEIGHT_M) == pytest.approx(top_cm + (low_cm - top_cm) * fraction, abs=1e-3)

    def test_cosine_crossing_time_is_the_known_ratio(self):
        assert cosine_crossing_time_s(2.0) == pytest.approx(1.4257, abs=1e-3)
