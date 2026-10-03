"""Synthetic conventional deadlifts with ground truth (PLAN.md §10, J2).

Generates the lagged world-frame skeleton (21 keypoints, Y-down metres, floor at
y = 0, X = subject's left, forward = -Z) and the 3D bar state a perfect tracker
would report, frame by frame, for scripted sets: dead stops, touch-and-go,
quick re-pulls, failed and dropped reps, and every v1 fault injected on demand.
The pull is built bar-first: the bar path and the trunk angle are scripted, the
shoulders hang the arms from the bar, and the legs are solved to reach the hip,
so each injected fault has a known size.

By default the setup and knee-pass trunk angles come from the analyser's own
setup model (setup_model.py), so a clean set is clean by that model's standard.
RepScript.setup_trunk_deg / knee_pass_trunk_deg script them instead: the poses
then owe nothing to the model, and RepTruth carries the kinematic ground truth
(trunk angles, setup hip height, knee flexion at the knee pass) read off the
emitted poses, for tests that must not grade the model against itself. Hips
shooting (D2) is injected that way: a knee-pass trunk well forward of the setup's.

Each rep is lowered into the next rep's setup, so consecutive reps with
different scripts change the lifter's shape over the lowering, never in a frame.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from pydantic import BaseModel, Field

from biomechanics.utils.types import CocoKeypoints as CK

from .setup_model import AthleteSegments, solve_knee_pass, solve_setup
from .types import BarState3D

NUM_KEYPOINTS = 21
DEFAULT_FPS = 30.0
PLATE_RADIUS_M = 0.225
# Plate-hub centres either side of the bar centre.
HUB_HALF_SPAN_M = 0.85
# Legs at the top are this fraction of fully straight (a real lockout is never
# perfectly collinear, and the knee solve needs a non-degenerate triangle).
LOCKOUT_REACH_FRACTION = 0.998
STANDING_REACH_FRACTION = 0.995


class SimAthlete(BaseModel):
    tibia_m: float = 0.43
    femur_m: float = 0.45
    torso_m: float = 0.52
    upper_arm_m: float = 0.30
    forearm_m: float = 0.29
    grip_offset_m: float = 0.075
    ankle_height_m: float = 0.08
    foot_m: float = 0.18
    heel_m: float = 0.06
    stance_half_width_m: float = 0.13
    hip_half_width_m: float = 0.12
    shoulder_half_width_m: float = 0.19
    grip_half_width_m: float = 0.26

    @property
    def arm_m(self) -> float:
        return self.upper_arm_m + self.forearm_m

    @property
    def midfoot_forward_m(self) -> float:
        return 0.35 * self.foot_m

    def segments(self) -> AthleteSegments:
        return AthleteSegments(
            tibia_m=self.tibia_m, femur_m=self.femur_m, torso_m=self.torso_m,
            arm_m=self.arm_m, grip_offset_m=self.grip_offset_m,
        )


class RepScript(BaseModel):
    """One rep. Fault sizes are the injected truth (cm / deg / m)."""
    shoulder_ahead_m: float = 0.03
    bar_drift_m: float = 0.0
    lockout_deficit_deg: float = 0.0
    lean_back_deg: float = 0.0
    hip_shift_m: float = 0.0
    bar_tilt_m: float = 0.0
    elbow_bend_deg: float = 0.0
    pull_s: float = 1.2
    top_hold_s: float = 0.6
    lower_s: float = 1.0
    # Hands stay on the resting bar this long after the dead stop; 0 = touch-and-go
    # into the next rep.
    floor_hold_s: float = 1.2
    drop_bar: bool = False
    # A failed rep: the bar rises this far and comes back down without a top.
    fail_rise_m: float | None = None
    # Model-free poses: the trunk angle (deg forward of vertical) at setup and when
    # the bar reaches the knees, instead of the setup model's solves. A trunk the
    # legs cannot reach is brought upright until they can (RepTruth has the result).
    setup_trunk_deg: float | None = None
    knee_pass_trunk_deg: float | None = None
    # A sticking point: the bar stops this fraction of the way up for stall_s,
    # then the lifter grinds through it.
    stall_fraction: float | None = None
    stall_s: float = 0.0


class Scenario(BaseModel):
    athlete: SimAthlete = Field(default_factory=SimAthlete)
    reps: list[RepScript] = Field(default_factory=lambda: [RepScript(), RepScript(), RepScript()])
    # Bar centre ahead of the midfoot when set up (D1 truth, m).
    bar_midfoot_offset_m: float = 0.0
    fps: float = DEFAULT_FPS
    approach_m: float = 0.8
    approach_s: float = 1.5
    stance_s: float = 1.5
    hinge_s: float = 1.0
    setup_hold_s: float = 1.0
    leave_s: float = 1.5
    keypoint_noise_m: float = 0.0
    bar_noise_m: float = 0.0
    # The world's Y axis tilted from true gravity by this much (about the lateral axis).
    world_tilt_deg: float = 0.0
    track_bar: bool = True
    # Bar axis to the shin line at contact; the analyser's setup model assumes
    # 5 cm, real lifters vary, so a different value tests that assumption.
    shin_bar_m: float = 0.05
    # The plates hide the feet (confidence 0) whenever the bar is this far off its
    # rest; None: never hidden.
    plates_hide_feet_above_m: float | None = None
    # Between dead-stop reps the lifter stands up without the bar for this long,
    # then sets up again (a re-setup); None: stays on the bar.
    stand_between_reps_s: float | None = None
    seed: int = 7
    start_time: float = 100.0


class SimFrame(NamedTuple):
    timestamp: float
    frame_index: int
    points: np.ndarray
    confidences: np.ndarray
    bar: BarState3D | None


class RepTruth(NamedTuple):
    liftoff_time: float
    knee_pass_time: float
    top_time: float
    floor_time: float
    counted: bool
    touch_and_go: bool
    # Kinematic truth read off the emitted poses (sagittal, ankle-relative).
    liftoff_trunk_deg: float = math.nan
    knee_pass_trunk_deg: float = math.nan
    knee_pass_knee_flexion_deg: float = math.nan
    setup_hip_height_m: float = math.nan
    hip_rise_m: float = math.nan
    shoulder_rise_m: float = math.nan


class SimulatedSet(NamedTuple):
    frames: list[SimFrame]
    reps: list[RepTruth]
    gravity_up_world: np.ndarray


class _Pose(NamedTuple):
    """Sagittal (forward, up) positions relative to the ankle, plus lateral offsets."""
    bar: tuple[float, float]
    shoulder: tuple[float, float]
    elbow: tuple[float, float]
    wrist: tuple[float, float]
    hip: tuple[float, float]
    knee: tuple[float, float]
    hip_shift_m: float
    bar_tilt_m: float
    hands_on_bar: bool


def _smooth(fraction: float) -> float:
    fraction = min(1.0, max(0.0, fraction))
    return 0.5 - 0.5 * math.cos(math.pi * fraction)


# Two-link leg from the ankle to the hip, knee in front. Returns (knee, hip),
# pulling an out-of-reach hip back along the ankle-hip line.
def _knee_for(hip: tuple[float, float], athlete: SimAthlete) -> tuple[tuple[float, float], tuple[float, float]]:
    reach = math.hypot(*hip)
    max_reach = (athlete.tibia_m + athlete.femur_m) * LOCKOUT_REACH_FRACTION
    if reach > max_reach:
        hip = (hip[0] * max_reach / reach, hip[1] * max_reach / reach)
        reach = max_reach
    tibia, femur = athlete.tibia_m, athlete.femur_m
    along = (tibia ** 2 - femur ** 2 + reach ** 2) / (2.0 * reach)
    across = math.sqrt(max(0.0, tibia ** 2 - along ** 2))
    unit = (hip[0] / reach, hip[1] / reach)
    # Perpendicular pointing forward-ish (the knee bends forward).
    normal = (unit[1], -unit[0])
    knee = (unit[0] * along + normal[0] * across, unit[1] * along + normal[1] * across)
    return knee, hip


# The elbow on the shoulder-wrist line, pushed back by the bend.
def _elbow_between(
    shoulder: tuple[float, float], wrist: tuple[float, float], bend_deg: float, athlete: SimAthlete,
) -> tuple[float, float]:
    fraction = athlete.upper_arm_m / (athlete.upper_arm_m + athlete.forearm_m)
    along = (shoulder[0] + (wrist[0] - shoulder[0]) * fraction, shoulder[1] + (wrist[1] - shoulder[1]) * fraction)
    back = athlete.upper_arm_m * math.sin(math.radians(bend_deg) / 2.0)
    return (along[0] - back, along[1])


# The trunk angle closest to trunk_deg (never more forward) whose hip the legs
# reach.
def _reachable_trunk_deg(shoulder: tuple[float, float], trunk_deg: float, athlete: SimAthlete) -> float:
    reach = (athlete.tibia_m + athlete.femur_m) * LOCKOUT_REACH_FRACTION

    def hip_reach(angle_deg: float) -> float:
        angle = math.radians(angle_deg)
        return math.hypot(shoulder[0] - athlete.torso_m * math.sin(angle), shoulder[1] - athlete.torso_m * math.cos(angle))

    if hip_reach(trunk_deg) <= reach:
        return trunk_deg
    # With the shoulders hung from the bar, a more upright trunk brings the hip
    # closer to the ankle: the most forward angle the legs still reach.
    reachable, unreachable = -30.0, trunk_deg
    if hip_reach(reachable) > reach:
        return trunk_deg
    for _ in range(40):
        middle = (reachable + unreachable) / 2.0
        if hip_reach(middle) > reach:
            unreachable = middle
        else:
            reachable = middle
    return reachable


# Hands on the bar, straight-ish arms angled from the shoulder to the grip.
def _pose_from_bar(
    bar: tuple[float, float],
    trunk_deg: float,
    shoulder_ahead_m: float,
    elbow_bend_deg: float,
    hip_shift_m: float,
    bar_tilt_m: float,
    athlete: SimAthlete,
) -> _Pose:
    half = math.radians(elbow_bend_deg) / 2.0
    arm_span = (athlete.upper_arm_m + athlete.forearm_m) * math.cos(half)
    wrist = (bar[0], bar[1] + athlete.grip_offset_m)
    rise = math.sqrt(max(0.0, arm_span ** 2 - shoulder_ahead_m ** 2))
    shoulder = (bar[0] + shoulder_ahead_m, wrist[1] + rise)
    trunk = math.radians(_reachable_trunk_deg(shoulder, trunk_deg, athlete))
    hip = (shoulder[0] - athlete.torso_m * math.sin(trunk), shoulder[1] - athlete.torso_m * math.cos(trunk))
    knee, solved_hip = _knee_for(hip, athlete)
    if solved_hip != hip:
        # Still out of reach (the bar is above the lockout height): the body and
        # the bar ride down with the hip.
        shift = (solved_hip[0] - hip[0], solved_hip[1] - hip[1])
        hip = solved_hip
        shoulder = (shoulder[0] + shift[0], shoulder[1] + shift[1])
        wrist = (wrist[0] + shift[0], wrist[1] + shift[1])
        bar = (bar[0] + shift[0], bar[1] + shift[1])
    elbow = _elbow_between(shoulder, wrist, elbow_bend_deg, athlete)
    return _Pose(
        bar=bar, shoulder=shoulder, elbow=elbow, wrist=wrist, hip=hip, knee=knee,
        hip_shift_m=hip_shift_m, bar_tilt_m=bar_tilt_m, hands_on_bar=True,
    )


def _trunk_deg(pose: _Pose) -> float:
    return math.degrees(math.atan2(pose.shoulder[0] - pose.hip[0], pose.shoulder[1] - pose.hip[1]))


def _knee_flexion_deg(pose: _Pose) -> float:
    shin = pose.knee
    thigh = (pose.hip[0] - pose.knee[0], pose.hip[1] - pose.knee[1])
    cosine = (shin[0] * thigh[0] + shin[1] * thigh[1]) / (math.hypot(*shin) * math.hypot(*thigh))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _standing_pose(athlete: SimAthlete, bar_rest: tuple[float, float], back_m: float) -> _Pose:
    leg = (athlete.tibia_m + athlete.femur_m) * STANDING_REACH_FRACTION
    hip = (0.0, leg)
    knee, hip = _knee_for(hip, athlete)
    shoulder = (hip[0], hip[1] + athlete.torso_m)
    elbow = (shoulder[0], shoulder[1] - athlete.upper_arm_m)
    wrist = (shoulder[0], elbow[1] - athlete.forearm_m)
    # Walking in: the body is behind its final stance, the bar stays put.
    bar = (bar_rest[0] + back_m, bar_rest[1])
    return _Pose(
        bar=bar, shoulder=shoulder, elbow=elbow, wrist=wrist, hip=hip, knee=knee,
        hip_shift_m=0.0, bar_tilt_m=0.0, hands_on_bar=False,
    )


def _lerp_pose(first: _Pose, second: _Pose, fraction: float) -> _Pose:
    def mix(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
        return (a[0] + (b[0] - a[0]) * fraction, a[1] + (b[1] - a[1]) * fraction)

    return _Pose(
        bar=second.bar if second.hands_on_bar else first.bar,
        shoulder=mix(first.shoulder, second.shoulder),
        elbow=mix(first.elbow, second.elbow),
        wrist=mix(first.wrist, second.wrist),
        hip=mix(first.hip, second.hip),
        knee=mix(first.knee, second.knee),
        hip_shift_m=first.hip_shift_m + (second.hip_shift_m - first.hip_shift_m) * fraction,
        bar_tilt_m=first.bar_tilt_m + (second.bar_tilt_m - first.bar_tilt_m) * fraction,
        hands_on_bar=second.hands_on_bar and fraction >= 1.0,
    )


class _Builder:
    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        self.athlete = scenario.athlete
        self.dt = 1.0 / scenario.fps
        self.t = scenario.start_time
        self.frames: list[SimFrame] = []
        self.reps: list[RepTruth] = []
        self.rng = np.random.default_rng(scenario.seed)
        tilt = math.radians(scenario.world_tilt_deg)
        # Rotation about the lateral (X) axis: true up seen in the tilted world.
        self.rotation = np.array([
            [1.0, 0.0, 0.0],
            [0.0, math.cos(tilt), -math.sin(tilt)],
            [0.0, math.sin(tilt), math.cos(tilt)],
        ])
        athlete = self.athlete
        self.bar_rest = (
            athlete.midfoot_forward_m + scenario.bar_midfoot_offset_m,
            PLATE_RADIUS_M - athlete.ankle_height_m,
        )

    # -- world mapping -------------------------------------------------

    def _world(self, forward: float, up: float, lateral_left: float) -> np.ndarray:
        local = np.array([lateral_left, -(up + self.athlete.ankle_height_m), -forward])
        return self.rotation @ local

    def _emit(self, pose: _Pose, back_m: float = 0.0, bar_up_override: float | None = None) -> None:
        athlete = self.athlete
        points = np.zeros((NUM_KEYPOINTS, 3))
        shift = pose.hip_shift_m
        offset = -back_m
        for sign, hip_i, knee_i, ankle_i, shoulder_i, elbow_i, wrist_i, toe_i, heel_i in (
            (1.0, CK.LEFT_HIP, CK.LEFT_KNEE, CK.LEFT_ANKLE, CK.LEFT_SHOULDER, CK.LEFT_ELBOW,
             CK.LEFT_WRIST, CK.LEFT_FOOT_INDEX, CK.LEFT_HEEL),
            (-1.0, CK.RIGHT_HIP, CK.RIGHT_KNEE, CK.RIGHT_ANKLE, CK.RIGHT_SHOULDER, CK.RIGHT_ELBOW,
             CK.RIGHT_WRIST, CK.RIGHT_FOOT_INDEX, CK.RIGHT_HEEL),
        ):
            # Hip shift > 0 moves the hips toward the subject's right (-X).
            points[ankle_i] = self._world(offset, 0.0, sign * athlete.stance_half_width_m)
            points[toe_i] = self._world(offset + athlete.foot_m, -0.06, sign * (athlete.stance_half_width_m + 0.03))
            points[heel_i] = self._world(offset - athlete.heel_m, -0.06, sign * athlete.stance_half_width_m)
            points[knee_i] = self._world(offset + pose.knee[0], pose.knee[1], sign * athlete.stance_half_width_m - shift * 0.5)
            points[hip_i] = self._world(offset + pose.hip[0], pose.hip[1], sign * athlete.hip_half_width_m - shift)
            points[shoulder_i] = self._world(
                offset + pose.shoulder[0], pose.shoulder[1], sign * athlete.shoulder_half_width_m - shift,
            )
            arm_x = athlete.grip_half_width_m if pose.hands_on_bar else athlete.shoulder_half_width_m + 0.03
            # The hands ride the tilted bar: the left end low by half the tilt.
            hand_drop = sign * (pose.bar_tilt_m / 2.0) * (arm_x / HUB_HALF_SPAN_M) if pose.hands_on_bar else 0.0
            points[elbow_i] = self._world(offset + pose.elbow[0], pose.elbow[1] - hand_drop, sign * arm_x)
            points[wrist_i] = self._world(offset + pose.wrist[0], pose.wrist[1] - hand_drop, sign * arm_x)
        head = (
            offset + pose.shoulder[0] + 0.03, pose.shoulder[1] + 0.22,
        )
        points[CK.NOSE] = self._world(head[0] + 0.08, head[1], 0.0)
        points[CK.LEFT_EYE] = self._world(head[0] + 0.07, head[1] + 0.03, 0.03)
        points[CK.RIGHT_EYE] = self._world(head[0] + 0.07, head[1] + 0.03, -0.03)
        points[CK.LEFT_EAR] = self._world(head[0], head[1] + 0.01, 0.07)
        points[CK.RIGHT_EAR] = self._world(head[0], head[1] + 0.01, -0.07)

        scenario = self.scenario
        if scenario.keypoint_noise_m > 0.0:
            points = points + self.rng.normal(0.0, scenario.keypoint_noise_m, points.shape)
        confidences = np.full(NUM_KEYPOINTS, 0.9)
        hide_above = scenario.plates_hide_feet_above_m
        if hide_above is not None and pose.bar[1] - self.bar_rest[1] > hide_above:
            confidences[[CK.LEFT_ANKLE, CK.RIGHT_ANKLE, CK.LEFT_FOOT_INDEX, CK.RIGHT_FOOT_INDEX,
                         CK.LEFT_HEEL, CK.RIGHT_HEEL]] = 0.0

        bar_state = None
        if scenario.track_bar:
            bar_forward, bar_up = pose.bar
            if bar_up_override is not None:
                bar_up = bar_up_override
            tilt = pose.bar_tilt_m
            left = self._world(bar_forward, bar_up - tilt / 2.0, HUB_HALF_SPAN_M)
            right = self._world(bar_forward, bar_up + tilt / 2.0, -HUB_HALF_SPAN_M)
            if scenario.bar_noise_m > 0.0:
                left = left + self.rng.normal(0.0, scenario.bar_noise_m, 3)
                right = right + self.rng.normal(0.0, scenario.bar_noise_m, 3)
            bar_state = BarState3D(
                timestamp=self.t,
                left_end_m=tuple(float(v) for v in left),
                right_end_m=tuple(float(v) for v in right),
                views=3,
            )
        self.frames.append(SimFrame(
            timestamp=self.t, frame_index=len(self.frames) + 1,
            points=points, confidences=confidences, bar=bar_state,
        ))
        self.t += self.dt

    def _hold(self, pose: _Pose, seconds: float, back_m: float = 0.0) -> None:
        for _ in range(int(round(seconds * self.scenario.fps))):
            self._emit(pose, back_m)

    def _blend(self, first: _Pose, second: _Pose, seconds: float, back_from: float = 0.0, back_to: float = 0.0) -> None:
        steps = max(1, int(round(seconds * self.scenario.fps)))
        for step in range(1, steps + 1):
            fraction = _smooth(step / steps)
            self._emit(_lerp_pose(first, second, fraction), back_from + (back_to - back_from) * fraction)

    # -- rep geometry --------------------------------------------------

    def _setup_trunk_deg(self, script: RepScript) -> float:
        if script.setup_trunk_deg is not None:
            return script.setup_trunk_deg
        athlete = self.athlete
        solution = solve_setup(
            athlete.segments(), self.bar_rest[0], self.bar_rest[1], script.shoulder_ahead_m, self.scenario.shin_bar_m,
        )
        return solution.trunk_deg if solution is not None else 60.0

    def _setup_pose(self, script: RepScript) -> _Pose:
        return _pose_from_bar(
            self.bar_rest, self._setup_trunk_deg(script), script.shoulder_ahead_m, 0.0, 0.0, 0.0, self.athlete,
        )

    # Bar height at the top and the trunk angle there.
    def _top_geometry(self, script: RepScript) -> tuple[float, float]:
        athlete = self.athlete
        trunk_top = script.lockout_deficit_deg - script.lean_back_deg
        reach = (athlete.tibia_m + athlete.femur_m) * LOCKOUT_REACH_FRACTION
        shoulder_forward = self.bar_rest[0]
        hip_forward = shoulder_forward - athlete.torso_m * math.sin(math.radians(trunk_top))
        hip_up = math.sqrt(max(0.0, reach ** 2 - hip_forward ** 2))
        shoulder_up = hip_up + athlete.torso_m * math.cos(math.radians(trunk_top))
        bar_up = shoulder_up - athlete.grip_offset_m - athlete.arm_m
        return bar_up, trunk_top

    # The trunk angle when the bar reaches the knees, and that bar height.
    def _knee_pass_geometry(self, script: RepScript) -> tuple[float, float]:
        athlete = self.athlete
        if script.knee_pass_trunk_deg is not None:
            # Model-free: the trunk reaches its scripted angle with the bar at the
            # height of a vertical shin's knee.
            return script.knee_pass_trunk_deg, athlete.tibia_m
        knee_pass = solve_knee_pass(
            athlete.segments(), self.bar_rest[0], script.shoulder_ahead_m, self.scenario.shin_bar_m,
        )
        if knee_pass is None:
            return 45.0, athlete.tibia_m
        return knee_pass.trunk_deg, knee_pass.knee_height_m

    def _pull_pose(self, script: RepScript, fraction: float, top_up: float, trunk_top: float,
                   trunk_setup: float, trunk_knee: float, knee_up: float) -> _Pose:
        rest_forward, rest_up = self.bar_rest
        bar_up = rest_up + (top_up - rest_up) * fraction
        drift = script.bar_drift_m * math.sin(math.pi * min(1.0, fraction / 0.8)) if fraction < 0.8 else 0.0
        if bar_up <= knee_up:
            share = (bar_up - rest_up) / max(1e-6, knee_up - rest_up)
            trunk = trunk_setup + (trunk_knee - trunk_setup) * share
        else:
            share = (bar_up - knee_up) / max(1e-6, top_up - knee_up)
            trunk = trunk_knee + (trunk_top - trunk_knee) * _smooth(share)
        shoulder_ahead = script.shoulder_ahead_m * (1.0 - fraction)
        bend = script.elbow_bend_deg * math.sin(math.pi * fraction)
        shift = script.hip_shift_m * min(1.0, fraction / 0.5)
        tilt = script.bar_tilt_m * math.sin(math.pi * min(1.0, fraction / 0.5) / 2.0)
        return _pose_from_bar(
            (rest_forward + drift, bar_up), trunk, shoulder_ahead, bend, shift, tilt, self.athlete,
        )

    # end: the rep whose setup this one is lowered into (the next rep's, or its
    # own when the lifter then stands up or the set ends).
    def _rep(self, script: RepScript, touch_and_go_in: bool, next_is_touch_and_go: bool, end: RepScript) -> None:
        athlete = self.athlete
        trunk_setup = self._setup_trunk_deg(script)
        trunk_knee, knee_up = self._knee_pass_geometry(script)
        top_up, trunk_top = self._top_geometry(script)

        # The bar leaves its rest (or its touch-and-go low point) on the last frame
        # before it moves: the eased pull starts there with zero velocity.
        liftoff_time = self.t - self.dt
        if script.fail_rise_m is not None:
            fail_up = self.bar_rest[1] + script.fail_rise_m
            top_fraction = (fail_up - self.bar_rest[1]) / (top_up - self.bar_rest[1])
            steps = int(round(script.pull_s * self.scenario.fps))
            for step in range(1, steps + 1):
                self._emit(self._pull_pose(script, top_fraction * _smooth(step / steps), top_up, trunk_top,
                                           trunk_setup, trunk_knee, knee_up))
            for step in range(1, steps + 1):
                self._emit(self._pull_pose(script, top_fraction * (1.0 - _smooth(step / steps)), top_up,
                                           trunk_top, trunk_setup, trunk_knee, knee_up))
            if script.floor_hold_s > 0.0:
                self._blend(self._setup_pose(script), self._setup_pose(end), script.floor_hold_s)
            self.reps.append(RepTruth(liftoff_time, math.nan, math.nan, math.nan, False, touch_and_go_in))
            return

        start_pose = self._pull_pose(script, 0.0, top_up, trunk_top, trunk_setup, trunk_knee, knee_up)
        previous_pose = start_pose
        knee_pass: tuple[float, float, _Pose] | None = None
        for fraction in self._pull_fractions(script):
            pose = self._pull_pose(script, fraction, top_up, trunk_top, trunk_setup, trunk_knee, knee_up)
            if knee_pass is None and pose.bar[1] >= pose.knee[1]:
                # The bar crosses the knee's height between the previous frame and this one.
                before = previous_pose.bar[1] - previous_pose.knee[1]
                after = pose.bar[1] - pose.knee[1]
                part = -before / (after - before) if after != before else 1.0
                knee_pass = (self.t - self.dt + part * self.dt, part, previous_pose)
                knee_pass_pose = pose
            self._emit(pose)
            previous_pose = pose
        top_time = self.t - self.dt
        knee_pass_time = math.nan
        truth_angles: dict[str, float] = {}
        if knee_pass is not None:
            knee_pass_time, part, before_pose = knee_pass
            truth_angles = {
                "knee_pass_trunk_deg": _trunk_deg(before_pose) + part * (_trunk_deg(knee_pass_pose) - _trunk_deg(before_pose)),
                "knee_pass_knee_flexion_deg": _knee_flexion_deg(knee_pass_pose),
                "hip_rise_m": knee_pass_pose.hip[1] - start_pose.hip[1],
                "shoulder_rise_m": knee_pass_pose.shoulder[1] - start_pose.shoulder[1],
            }
        top_pose = self._pull_pose(script, 1.0, top_up, trunk_top, trunk_setup, trunk_knee, knee_up)
        self._hold(top_pose, script.top_hold_s)

        if script.drop_bar:
            floor_time = self._drop(top_pose, top_up)
        else:
            lower_steps = int(round(script.lower_s * self.scenario.fps))
            # Down from this rep's top into the end rep's setup.
            lowering = script.model_copy(update={
                "bar_drift_m": 0.0, "elbow_bend_deg": 0.0, "shoulder_ahead_m": end.shoulder_ahead_m,
            })
            end_trunk_knee, end_knee_up = self._knee_pass_geometry(end)
            end_trunk_setup = self._setup_trunk_deg(end)
            for step in range(1, lower_steps + 1):
                fraction = 1.0 - _smooth(step / lower_steps)
                self._emit(self._pull_pose(
                    lowering, fraction, top_up, trunk_top, end_trunk_setup, end_trunk_knee, end_knee_up,
                ))
            floor_time = self.t - self.dt
        self.reps.append(RepTruth(
            liftoff_time, knee_pass_time, top_time, floor_time, True, touch_and_go_in,
            liftoff_trunk_deg=_trunk_deg(start_pose),
            setup_hip_height_m=start_pose.hip[1],
            **truth_angles,
        ))
        if script.drop_bar:
            # The lifter let go at the top and stays standing over the dropped bar.
            self._hold(_standing_pose(athlete, self.bar_rest, 0.0), script.floor_hold_s)
        elif not next_is_touch_and_go:
            self._hold(self._setup_pose(end), script.floor_hold_s)

    # The bar's share of the way up on each pull frame: one eased move, or two
    # either side of a sticking point the bar holds at.
    def _pull_fractions(self, script: RepScript) -> list[float]:
        fps = self.scenario.fps
        if script.stall_fraction is None:
            steps = int(round(script.pull_s * fps))
            return [_smooth(step / steps) for step in range(1, steps + 1)]
        stall = script.stall_fraction
        before = max(1, int(round(script.pull_s * stall * fps)))
        after = max(1, int(round(script.pull_s * (1.0 - stall) * fps)))
        return (
            [stall * _smooth(step / before) for step in range(1, before + 1)]
            + [stall] * int(round(script.stall_s * fps))
            + [stall + (1.0 - stall) * _smooth(step / after) for step in range(1, after + 1)]
        )

    # Hands open at the top: the bar falls, bounces on bumpers, settles. The
    # lifter stays standing.
    def _drop(self, top_pose: _Pose, top_up: float) -> float:
        athlete = self.athlete
        rest_up = self.bar_rest[1]
        standing = _standing_pose(athlete, self.bar_rest, 0.0)
        bar_up = top_up
        velocity = 0.0
        floor_time = math.nan
        bounce_velocity = 0.0
        for _ in range(int(round(1.5 * self.scenario.fps))):
            velocity -= 9.81 * self.dt
            bar_up += velocity * self.dt
            if bar_up <= rest_up:
                bar_up = rest_up
                if math.isnan(floor_time):
                    floor_time = self.t
                    # A bumper returns a small bounce (~2 cm).
                    bounce_velocity = math.sqrt(2.0 * 9.81 * 0.02)
                    velocity = bounce_velocity
                else:
                    velocity = 0.0
            pose = standing._replace(bar=(self.bar_rest[0], bar_up))
            self._emit(pose)
        return floor_time

    def build(self) -> SimulatedSet:
        scenario = self.scenario
        athlete = self.athlete
        reps = scenario.reps
        standing = _standing_pose(athlete, self.bar_rest, 0.0)
        self._blend(standing, standing, scenario.approach_s, back_from=scenario.approach_m, back_to=0.0)
        self._hold(standing, scenario.stance_s)
        setup = self._setup_pose(reps[0])
        self._blend(standing, setup, scenario.hinge_s)
        self._hold(setup, scenario.setup_hold_s)
        touch_and_go_in = False
        for index, script in enumerate(reps):
            next_tng = script.floor_hold_s <= 0.0 and index + 1 < len(reps) and not script.drop_bar
            stays_at_bar = index + 1 < len(reps) and (next_tng or scenario.stand_between_reps_s is None)
            self._rep(script, touch_and_go_in, next_tng, reps[index + 1] if stays_at_bar else script)
            if script.drop_bar and index + 1 < len(reps):
                # Back down to the bar for the next rep.
                self._blend(standing, self._setup_pose(reps[index + 1]), scenario.hinge_s)
                self._hold(self._setup_pose(reps[index + 1]), scenario.setup_hold_s)
            elif scenario.stand_between_reps_s is not None and index + 1 < len(reps) and not next_tng:
                # Stands up off the bar, then sets up again.
                self._blend(self._setup_pose(script), standing, scenario.hinge_s)
                self._hold(standing, scenario.stand_between_reps_s)
                self._blend(standing, self._setup_pose(reps[index + 1]), scenario.hinge_s)
                self._hold(self._setup_pose(reps[index + 1]), scenario.setup_hold_s)
            touch_and_go_in = next_tng
        last_setup = self._setup_pose(reps[-1])
        if not reps[-1].drop_bar:
            self._blend(last_setup, standing, scenario.hinge_s)
        self._blend(standing, standing, scenario.leave_s, back_from=0.0, back_to=scenario.approach_m)
        gravity_up = self.rotation @ np.array([0.0, -1.0, 0.0])
        return SimulatedSet(frames=self.frames, reps=self.reps, gravity_up_world=gravity_up)


def simulate(scenario: Scenario | None = None) -> SimulatedSet:
    return _Builder(scenario or Scenario()).build()
