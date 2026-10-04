"""Deadlift rep analyser on simulated sets (docs/deadlift/PLAN.md §2.3-2.5, J2/J3):
every rep counted exactly once, events on time, each injected fault detected by
its rule and a clean set fault-free."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pytest

from biomechanics.config import BiomechanicsConfig
from biomechanics.deadlift.analyzer import LIVE_OFFSET_MIN_FRAMES, DeadliftFrameInput, DeadliftRepAnalyzer
from biomechanics.deadlift.bar_tracker_3d import DEFAULT_CONFIG as TRACKER_CONFIG
from biomechanics.deadlift.rule_base import TIER_RANK
from biomechanics.deadlift.setup_model import MAX_SETUP_KNEE_FLEXION_DEG, MIN_SETUP_KNEE_FLEXION_DEG
from biomechanics.deadlift.session_reference import DeadliftSessionReference
from biomechanics.deadlift.simulator import RepScript, Scenario, SimAthlete, SimFrame, SimulatedSet, simulate
from biomechanics.deadlift.types import (
    BAR_SOURCE_BAR,
    BAR_SOURCE_WRIST_PROXY,
    GRAVITY_SOURCE_BODY,
    GRAVITY_SOURCE_MEASURED,
    BarState3D,
    DeadliftPhase,
    DeadliftRepFeatures,
)
from biomechanics.faults.rule_engine import RuleEngine
from biomechanics.profiles.deadlift import DeadliftProfile
from biomechanics.utils.geometry import WORLD_UP
from biomechanics.utils.types import CocoKeypoints as CK
from biomechanics.utils.types import FaultEvent, JointAngles

# PLAN.md §1 demo gate: event timing median error <= 100 ms; held here per event.
MAX_EVENT_ERROR_S = 0.1
CLEAN_DRIFT_MAX_CM = 1.0
CLEAN_TILT_MAX_CM = 1.0
CLEAN_SHIFT_MAX_RATIO = 0.03
CLEAN_ANGLE_TOLERANCE_DEG = 2.0
# 3 deg of tilted world vertical over the pull fakes this much drift without measured gravity.
TILT_FAKE_DRIFT_MIN_CM = 2.0
# Keypoint noise the platform documents for Kalman-lagged triangulated keypoints
# (utils/segment_lengths.py: 1.6-2.4 cm), and a tracked bar's.
PLATFORM_KEYPOINT_NOISE_M = 0.02
TRACKED_BAR_NOISE_M = 0.003
# The agent arms closed-loop foot guidance beyond this offset (CONTRACT.md §5.5:
# D1's mild threshold), then guides down to the tolerance.
BAR_MIDFOOT_GUIDANCE_ARM_CM = 3.0
BAR_MIDFOOT_GUIDANCE_TOLERANCE_CM = 2.0
# The plates hide the knees from the cameras with the bar this far off the floor.
KNEES_HIDDEN_ABOVE_M = 0.15
KNEES = (CK.LEFT_KNEE, CK.RIGHT_KNEE)
# A re-setup at the floor: the lifter shuffles the feet this far toward the bar.
FEET_MOVED_AT_THE_FLOOR_M = 0.05
# Hip shift a stance leaks into D8 beyond the same seed's square stance.
STANCE_LEAK_MAX_RATIO = 0.03
# A 0.20-separation hip shift reads this noise-free (the first fifth's median
# takes ~0.015 off it), and within this of it whatever the pelvis does.
SQUARE_PELVIS_SHIFT_RATIO = 0.185
SHIFT_SIZE_TOLERANCE_RATIO = 0.03
# A 2.5 cm shrug at the top moves the top event no more than this.
SHRUG_TOP_SHIFT_MAX_S = 0.15
# A lockout settling upward inside the hold band (the shoulders drawn back on
# straight legs): SETTLE_RISE_M from SETTLE_AFTER_TOP_S after the top, over
# SETTLE_RAMP_S, held until the lowering.
SETTLE_RISE_M = 0.01
SETTLE_AFTER_TOP_S = 0.3
SETTLE_RAMP_S = 0.3
# A bar lost low on its last climb and seen again at the top: its arrival read
# from its last seen speed, within this.
GAP_TOP_MAX_ERROR_S = 0.15
# A sag after the lockout: from SAG_START_S after the top, down and back up over
# SAG_RAMP_S each, back up from SAG_UP_S.
SAG_START_S = 0.1
SAG_RAMP_S = 0.2
SAG_UP_S = 0.6
# Touch-and-go with no pause at the top, lowered faster than pulled (pull_s, lower_s).
FAST_LOWERING_TEMPOS = ((1.2, 0.5), (1.5, 0.5), (2.0, 0.6))
# A soft lockout cued moderate or worse on every rep, noise-free.
SOFT_LOCKOUT_DEG = 15.0
# On the wrist proxy with the wrists hidden through the hold, at 1.5 cm of
# correlated noise: a soft lockout is still cued on this share of reps (38-40 of
# 45 on four draws; 43-45 with the wrists seen).
HIDDEN_SOFT_LOCKOUT_RECALL_RATIO = 0.8
# The wrist proxy's tops beyond MAX_PROXY_TOP_ERROR_S: at most this share.
LATE_PROXY_TOP_RATIO = 0.1
# A noise-free wrist-proxy lockout reads within this of the tracked bar's.
NOISE_FREE_PROXY_LOCKOUT_TOLERANCE_DEG = 1.0
# A noise-free 0.6 s pull with no pause at the top reads its lockout within this
# of standing (judged 0.1 s either side of the top, it read 7.5 deg short).
FAST_TOP_MAX_DEFICIT_DEG = 4.0
# A grind stalled 93 % of the way up for 0.6 s, finishing in 0.4 s.
GRIND_93 = RepScript(pull_s=1.6, stall_fraction=0.93, stall_s=0.6, finish_s=0.4)
# The tracker's last measured centres its coasting velocity is fitted to.
TRACKER_VELOCITY_FRAMES = 4
# Holds lost from view at platform noise: the share of tops dated beyond twice
# GAP_TOP_MAX_ERROR_S.
LOST_TOP_FAR_RATIO = 0.05
# A draw (2 cm AR(0.8), 1.5 s pulls) whose hips and knees wander through the
# hold: against their least, the fit was vetoed and one top read +0.47 s.
WANDERING_JOINTS_NOISE_SEED = 3600
# A 2 cm AR(0.8) draw over long unseen holds whose hips and knees' 5-frame medians
# reached the resume deficit on both: frame by frame, 8 soft lockouts and 3 holds
# without a standing reference resumed and read 0.33-1.04 s late.
UNSEEN_HOLD_NOISE_SEED = 2800
# The faulted lockouts a top lost from view is dated at (deg).
FAULTED_LOCKOUT_DEG = 15.0
# Grinds stalled 3 % short and lost through their finish and hold, 1.5 cm AR(0.8):
# the share dated beyond 0.3 s (the hips finish such a stall by little more than
# the resume deficit; requiring it of both joints, 19 of 27 on this draw).
NEAR_LOCKOUT_GRIND_NOISE_SEED = 4100
NEAR_LOCKOUT_GRIND_FAR_RATIO = 0.5
# The bar dropped for SHORT_GAP_S every SHORT_GAP_EVERY_S through a hold, at the
# top of the platform's noise (2.5 cm AR(0.8)): a draw where, without a minimum
# gap, holds resumed on each gap's few frames.
SHORT_GAP_S = 0.1
SHORT_GAP_EVERY_S = 0.25
WORST_KEYPOINT_NOISE_M = 0.025
SHORT_GAPS_NOISE_SEED = 1400
# A 1.5 cm AR(0.8) draw whose expected top heights put the bar's fit alone early on
# 2 s pulls with the knees hidden: the climb's bent knees, judged, cued D6 on 7 of 45.
EARLY_FIT_NOISE_SEED = 4900
# Failed reps this far short of the full rise (past the top margin), the bar and the
# knees hidden this long either side of their peak, at 1.5 cm AR(0.8).
FAILED_SHORT_OF_TOP_M = (0.10, 0.12)
FAILED_HIDDEN_HALF_S = {1.2: 0.3, 2.0: 0.4}
FAILED_HIDDEN_NOISE_SEED = 5900
# Lockouts leaned back this far (deg), past the over-extended top's 10 deg.
OVER_EXTENDED_LEANS_DEG = (25.0, 40.0)
# The share of a lost top's gap, from its start, the knees are hidden over.
PARTLY_HIDDEN_SHARE = 0.55
# A 1.5 cm AR(0.8) draw for standing's and the over-extended lockout's recall with
# the bar unseen.
STANDING_RECALL_NOISE_SEED = 6900
# A no-pause top lost from view reads its lockout within this of straight,
# noise-free (seen: within 3 deg), short of D6's mild threshold.
LOST_TOP_MAX_DEFICIT_DEG = 6.0
# VALIDATION.md's gate: 1 false correction per 10 reps.
FALSE_CUE_GATE_RATIO = 0.1
# The wrist proxy's top event: bound by the wrists' noise, not the 100 ms gate.
MAX_PROXY_TOP_ERROR_S = 0.3
# A slow proxy pull's joints creep their last few degrees to the lockout over
# ~0.25 s; its top is dated early by up to this.
MAX_PROXY_SLOW_TOP_ERROR_S = 0.5
PROXY_KEYPOINT_NOISE_M = 0.015
# A noisier tracked bar than TRACKED_BAR_NOISE_M.
NOISY_TRACKED_BAR_NOISE_M = 0.005
# Correlation of the platform's frame-to-frame keypoint error (AR(1)).
PLATFORM_NOISE_RHO = 0.8
# A soft lockout's deficit under noise reads within this of its noise-free reading.
MAX_LOCKOUT_BIAS_DEG = 1.0
# A grind finishing its last 2.5 cm over 1 s: the last ~0.17 s move the bar
# under 2 mm, inside a tracked bar's noise.
SLOW_FINISH_MAX_MEDIAN_ERROR_S = 0.15
# Keypoints a shrug at the top lifts with the bar.
UPPER_BODY = (
    CK.LEFT_SHOULDER, CK.RIGHT_SHOULDER, CK.LEFT_ELBOW, CK.RIGHT_ELBOW, CK.LEFT_WRIST, CK.RIGHT_WRIST,
    CK.NOSE, CK.LEFT_EYE, CK.RIGHT_EYE, CK.LEFT_EAR, CK.RIGHT_EAR,
)
# A staggered stance: the left foot this far ahead of the right, hips square.
STAGGER_M = 0.06
LEFT_FOOT = (CK.LEFT_ANKLE, CK.LEFT_HEEL, CK.LEFT_FOOT_INDEX)
# Keypoints of a stance turned off square to the bar (the hands stay on it).
LOWER_BODY = (
    CK.LEFT_HIP, CK.RIGHT_HIP, CK.LEFT_KNEE, CK.RIGHT_KNEE, CK.LEFT_ANKLE, CK.RIGHT_ANKLE,
    CK.LEFT_HEEL, CK.RIGHT_HEEL, CK.LEFT_FOOT_INDEX, CK.RIGHT_FOOT_INDEX,
)
FAULTS = BiomechanicsConfig().faults
# Five bodies: default, short, tall, narrow hips on a wide stance, long femurs.
BODIES = (
    SimAthlete(),
    SimAthlete(tibia_m=0.38, femur_m=0.40, torso_m=0.47, upper_arm_m=0.27, forearm_m=0.26,
               hip_half_width_m=0.10, stance_half_width_m=0.11, shoulder_half_width_m=0.17, grip_half_width_m=0.24),
    SimAthlete(tibia_m=0.48, femur_m=0.50, torso_m=0.57, upper_arm_m=0.33, forearm_m=0.31,
               hip_half_width_m=0.14, stance_half_width_m=0.15, shoulder_half_width_m=0.21, grip_half_width_m=0.28),
    SimAthlete(hip_half_width_m=0.095, stance_half_width_m=0.17, grip_half_width_m=0.29),
    SimAthlete(tibia_m=0.40, femur_m=0.52, torso_m=0.48),
)
# A grind finishing its last centimetres in a few frames: the hold must not pull
# its fitted top late.
QUICK_FINISH_MAX_ERROR_S = 0.2
# Kinematic truth vs measurement on noise-free model-free poses.
# One frame (the first after the bar passes the knees) of trunk motion.
TRUNK_CHANGE_TOLERANCE_DEG = 2.0
RISE_RATIO_TOLERANCE = 0.04
HIP_HEIGHT_TOLERANCE_CM = 1.0


FrameMutation = Callable[[int, SimFrame], tuple[np.ndarray, np.ndarray, BarState3D | None]]


def _correlated_noise(sigma_m: float, rho: float, seed: int) -> FrameMutation:
    """AR(1) keypoint noise of stationary std sigma_m: the slow wander a Kalman-
    smoothed triangulation leaves, which a short window reads as motion."""
    rng = np.random.default_rng(seed)
    state: dict[str, np.ndarray] = {}

    def mutate(index: int, frame: SimFrame) -> tuple[np.ndarray, np.ndarray, BarState3D | None]:
        innovation = rng.normal(0.0, sigma_m * math.sqrt(1.0 - rho ** 2), frame.points.shape)
        state["error"] = innovation if "error" not in state else rho * state["error"] + innovation
        return frame.points + state["error"], frame.confidences, frame.bar

    return mutate


def _bar_dropout(probability: float, seed: int) -> FrameMutation:
    rng = np.random.default_rng(seed)

    def mutate(index: int, frame: SimFrame) -> tuple[np.ndarray, np.ndarray, BarState3D | None]:
        return frame.points, frame.confidences, None if rng.random() < probability else frame.bar

    return mutate


# The whole lifter moves toward the bar (-Z) over 0.8 s, starting 0.8 s after
# touchdown, still hinged over it.
def _stepped_closer_at_the_floor(floor_time: float, forward_m: float) -> FrameMutation:
    def mutate(index: int, frame: SimFrame) -> tuple[np.ndarray, np.ndarray, BarState3D | None]:
        fraction = min(1.0, max(0.0, (frame.timestamp - floor_time - 0.8) / 0.8))
        points = frame.points.copy()
        points[:, 2] -= forward_m * fraction
        return points, frame.confidences, frame.bar

    return mutate


# Keypoints turned about the vertical through the ankles, then the noise (if any).
def _turned(
    sim: SimulatedSet, degrees: float, keypoints: tuple[int, ...], noise: FrameMutation | None = None,
) -> FrameMutation:
    pivot = (sim.frames[0].points[CK.LEFT_ANKLE] + sim.frames[0].points[CK.RIGHT_ANKLE]) / 2.0
    angle = math.radians(degrees)
    rotation = np.array([
        [math.cos(angle), 0.0, math.sin(angle)], [0.0, 1.0, 0.0], [-math.sin(angle), 0.0, math.cos(angle)],
    ])
    indices = list(keypoints)

    def mutate(index: int, frame: SimFrame) -> tuple[np.ndarray, np.ndarray, BarState3D | None]:
        points = frame.points.copy()
        points[indices] = (points[indices] - pivot) @ rotation.T + pivot
        turned = frame._replace(points=points)
        return (turned.points, turned.confidences, turned.bar) if noise is None else noise(index, turned)

    return mutate


# The left foot moved forward (-Z), then the noise (if any).
def _staggered(forward_m: float, noise: FrameMutation | None = None) -> FrameMutation:
    def mutate(index: int, frame: SimFrame) -> tuple[np.ndarray, np.ndarray, BarState3D | None]:
        points = frame.points.copy()
        points[list(LEFT_FOOT), 2] -= forward_m
        staggered = frame._replace(points=points)
        return (staggered.points, staggered.confidences, staggered.bar) if noise is None else noise(index, staggered)

    return mutate


# The whole lifter stands turned `degrees` about the ankles, then squares up to
# the bar in place between square_from and square_to, then the noise.
def _turned_then_squared(
    sim: SimulatedSet, degrees: float, square_from: float, square_to: float, noise: FrameMutation,
) -> FrameMutation:
    pivot = (sim.frames[0].points[CK.LEFT_ANKLE] + sim.frames[0].points[CK.RIGHT_ANKLE]) / 2.0

    def mutate(index: int, frame: SimFrame) -> tuple[np.ndarray, np.ndarray, BarState3D | None]:
        share = min(1.0, max(0.0, (square_to - frame.timestamp) / (square_to - square_from)))
        angle = math.radians(degrees * share)
        rotation = np.array([
            [math.cos(angle), 0.0, math.sin(angle)], [0.0, 1.0, 0.0], [-math.sin(angle), 0.0, math.cos(angle)],
        ])
        return noise(index, frame._replace(points=(frame.points - pivot) @ rotation.T + pivot))

    return mutate


def _yaw(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    return np.array([[math.cos(angle), 0.0, math.sin(angle)], [0.0, 1.0, 0.0], [-math.sin(angle), 0.0, math.cos(angle)]])


# The two hip keypoints turned about their midpoint (the midpoint stays put).
def _hips_yawed(points: np.ndarray, degrees: float) -> np.ndarray:
    points = points.copy()
    middle = (points[CK.LEFT_HIP] + points[CK.RIGHT_HIP]) / 2.0
    for index in (CK.LEFT_HIP, CK.RIGHT_HIP):
        points[index] = (points[index] - middle) @ _yaw(degrees).T + middle
    return points


# The hip line turned against legs and travel that stay square to the bar, by a
# fixed angle (pelvic rotation, or a front-back asymmetry of the hip keypoints)
# or, with shift_m, in proportion to how far the hips have moved sideways.
def _hip_line_turned(degrees: float, shift_m: float | None = None) -> FrameMutation:
    def mutate(index: int, frame: SimFrame) -> tuple[np.ndarray, np.ndarray, BarState3D | None]:
        share = 1.0
        if shift_m is not None:
            share = -(frame.points[CK.LEFT_HIP][0] + frame.points[CK.RIGHT_HIP][0]) / 2.0 / shift_m
        return _hips_yawed(frame.points, degrees * share), frame.confidences, frame.bar

    return mutate


# A pelvis twist with no sideways travel: in over the first half of each pull,
# held through the top, out over the lowering.
def _pelvis_twisting(sim: SimulatedSet, degrees: float) -> FrameMutation:
    windows = [(rep.liftoff_time, rep.top_time, rep.floor_time) for rep in sim.reps if rep.counted]

    def mutate(index: int, frame: SimFrame) -> tuple[np.ndarray, np.ndarray, BarState3D | None]:
        share = 0.0
        for liftoff, top, floor in windows:
            if liftoff <= frame.timestamp <= top:
                share = min(1.0, (frame.timestamp - liftoff) / (0.5 * (top - liftoff)))
            elif top < frame.timestamp <= floor:
                share = 1.0 - (frame.timestamp - top) / (floor - top)
        return _hips_yawed(frame.points, degrees * share), frame.confidences, frame.bar

    return mutate


# A shrug of rise_m at the top of each of the reps: shoulders, arms and bar up,
# then down. Then the next mutation (noise, occlusion), if any.
def _shrugged(
    sim: SimulatedSet, rise_m: float, rep_indices: tuple[int, ...], then: FrameMutation | None = None,
) -> FrameMutation:
    starts = [sim.reps[rep_index].top_time + 0.25 for rep_index in rep_indices]

    def lift_m(t: float) -> float:
        for start in starts:
            up_end, hold_end, down_end = start + 0.25, start + 0.55, start + 0.8
            if start <= t < up_end:
                return rise_m * (t - start) / (up_end - start)
            if up_end <= t < hold_end:
                return rise_m
            if hold_end <= t < down_end:
                return rise_m * (1.0 - (t - hold_end) / (down_end - hold_end))
        return 0.0

    def mutate(index: int, frame: SimFrame) -> tuple[np.ndarray, np.ndarray, BarState3D | None]:
        frame = _upper_body_lifted(frame, lift_m(frame.timestamp))
        return (frame.points, frame.confidences, frame.bar) if then is None else then(index, frame)

    return mutate


# The shoulders, arms, head and bar lifted by lift_m, lowered when negative
# (Y-down: up is -y).
def _upper_body_lifted(frame: SimFrame, lift_m: float) -> SimFrame:
    if lift_m == 0.0:
        return frame
    points = frame.points.copy()
    points[list(UPPER_BODY), 1] -= lift_m
    left, right = list(frame.bar.left_end_m), list(frame.bar.right_end_m)
    left[1] -= lift_m
    right[1] -= lift_m
    bar = frame.bar.model_copy(update={"left_end_m": tuple(left), "right_end_m": tuple(right)})
    return frame._replace(points=points, bar=bar)


# The bar lost in the windows as BarTracker3D reports it: predicted states
# carried on at the last measured velocity for its max_prediction_s, then None.
def _tracker_coasting(windows: list[tuple[float, float]]) -> FrameMutation:
    history: list[tuple[float, BarState3D]] = []

    def mutate(index: int, frame: SimFrame) -> tuple[np.ndarray, np.ndarray, BarState3D | None]:
        if not any(start <= frame.timestamp < end for start, end in windows):
            history.append((frame.timestamp, frame.bar))
            del history[:-TRACKER_VELOCITY_FRAMES]
            return frame.points, frame.confidences, frame.bar
        last_t, last = history[-1]
        if frame.timestamp - last_t > TRACKER_CONFIG.max_prediction_s + 1e-9:
            return frame.points, frame.confidences, None
        times = np.array([t for t, _ in history]) - last_t
        velocity = np.polyfit(times, np.array([bar.centre for _, bar in history]), 1)[0]
        shift = velocity * (frame.timestamp - last_t)
        bar = last.model_copy(update={
            "timestamp": frame.timestamp,
            "left_end_m": tuple((np.asarray(last.left_end_m) + shift).tolist()),
            "right_end_m": tuple((np.asarray(last.right_end_m) + shift).tolist()),
            "predicted": True,
        })
        return frame.points, frame.confidences, bar

    return mutate


# The bar gone (None) in the windows: the detector's frames dropped outright.
def _bar_dropped(windows: list[tuple[float, float]]) -> FrameMutation:
    def mutate(index: int, frame: SimFrame) -> tuple[np.ndarray, np.ndarray, BarState3D | None]:
        lost = any(start <= frame.timestamp < end for start, end in windows)
        return frame.points, frame.confidences, None if lost else frame.bar

    return mutate


LOSSES = {"dropped": _bar_dropped, "coasting": _tracker_coasting}


# One mutation, then the next.
def _then(first: FrameMutation, second: FrameMutation) -> FrameMutation:
    def mutate(index: int, frame: SimFrame) -> tuple[np.ndarray, np.ndarray, BarState3D | None]:
        points, confidences, bar = first(index, frame)
        return second(index, frame._replace(points=points, confidences=confidences, bar=bar))

    return mutate


# The windows the simulator's plates hide the feet in: the pipeline marks those
# frames' legs unmeasured.
def _feet_hidden_windows(sim: SimulatedSet) -> list[tuple[float, float]]:
    windows: list[tuple[float, float]] = []
    for frame, after in zip(sim.frames, sim.frames[1:]):
        if frame.confidences[[CK.LEFT_ANKLE, CK.RIGHT_ANKLE]].max() > 0.0:
            continue
        if windows and abs(windows[-1][1] - frame.timestamp) < 1e-9:
            windows[-1] = (windows[-1][0], after.timestamp)
        else:
            windows.append((frame.timestamp, after.timestamp))
    return windows


# The bar's full rise for a body: its highest point above the rest, seen.
def _full_rise_m(body: SimAthlete) -> float:
    sim = simulate(Scenario(athlete=body, reps=[RepScript()]))
    heights = [-frame.bar.centre[1] for frame in sim.frames]
    return max(heights) - heights[0]


# The given keypoints unseen (confidence 0) in the windows.
def _hidden_keypoints(windows: list[tuple[float, float]], keypoints: tuple[int, ...]) -> FrameMutation:
    def mutate(index: int, frame: SimFrame) -> tuple[np.ndarray, np.ndarray, BarState3D | None]:
        if not any(start <= frame.timestamp < end for start, end in windows):
            return frame.points, frame.confidences, frame.bar
        confidences = frame.confidences.copy()
        confidences[list(keypoints)] = 0.0
        return frame.points, confidences, frame.bar

    return mutate


# The wrists unseen (confidence 0) in the windows, the wrist proxy's bar lost
# with them. Then the next mutation (noise), if any.
def _wrists_hidden(windows: list[tuple[float, float]], then: FrameMutation | None = None) -> FrameMutation:
    def mutate(index: int, frame: SimFrame) -> tuple[np.ndarray, np.ndarray, BarState3D | None]:
        if any(start <= frame.timestamp < end for start, end in windows):
            confidences = frame.confidences.copy()
            confidences[[CK.LEFT_WRIST, CK.RIGHT_WRIST]] = 0.0
            frame = frame._replace(confidences=confidences)
        return (frame.points, frame.confidences, frame.bar) if then is None else then(index, frame)

    return mutate


# A lockout whose shoulders relax after the top, lowering the bar sag_m, and
# tighten again before the lowering. Then the next mutation (noise), if any.
def _sagged(
    sim: SimulatedSet, sag_m: float, rep_indices: tuple[int, ...], then: FrameMutation | None = None,
) -> FrameMutation:
    tops = [sim.reps[rep_index].top_time for rep_index in rep_indices]

    def sag_at(t: float) -> float:
        for top in tops:
            down, held, up, back = top + SAG_START_S, top + SAG_START_S + SAG_RAMP_S, top + SAG_UP_S, top + SAG_UP_S + SAG_RAMP_S
            if down <= t < held:
                return sag_m * (t - down) / SAG_RAMP_S
            if held <= t < up:
                return sag_m
            if up <= t < back:
                return sag_m * (1.0 - (t - up) / SAG_RAMP_S)
        return 0.0

    def mutate(index: int, frame: SimFrame) -> tuple[np.ndarray, np.ndarray, BarState3D | None]:
        frame = _upper_body_lifted(frame, -sag_at(frame.timestamp))
        return (frame.points, frame.confidences, frame.bar) if then is None else then(index, frame)

    return mutate


# Touch-and-go reps with no pause at the top.
def _no_pause_touch_and_go(pull_s: float, lower_s: float) -> list[RepScript]:
    script = RepScript(top_hold_s=0.0, pull_s=pull_s, lower_s=lower_s)
    return [script.model_copy(update={"floor_hold_s": 0.0})] * 4 + [script]


# A lockout that settles upward by rise_m on each of the reps and stays there
# until the lowering, then comes back down with the bar.
def _settled_up(sim: SimulatedSet, rise_m: float, rep_indices: tuple[int, ...], hold_s: float, lower_s: float) -> FrameMutation:
    tops = [sim.reps[rep_index].top_time for rep_index in rep_indices]

    def lift_m(t: float) -> float:
        for top in tops:
            start, lowering = top + SETTLE_AFTER_TOP_S, top + hold_s
            if start <= t < start + SETTLE_RAMP_S:
                return rise_m * (t - start) / SETTLE_RAMP_S
            if start + SETTLE_RAMP_S <= t < lowering:
                return rise_m
            if lowering <= t < lowering + lower_s:
                return rise_m * (1.0 - (t - lowering) / lower_s)
        return 0.0

    def mutate(index: int, frame: SimFrame) -> tuple[np.ndarray, np.ndarray, BarState3D | None]:
        frame = _upper_body_lifted(frame, lift_m(frame.timestamp))
        return frame.points, frame.confidences, frame.bar

    return mutate


# The knees at confidence 0 while the bar is more than KNEES_HIDDEN_ABOVE_M off
# the floor: the plates hide them from the side views.
def _knees_hidden_off_the_floor(sim: SimulatedSet) -> FrameMutation:
    first = sim.frames[0].bar
    floor_bar_up = -(first.left_end_m[1] + first.right_end_m[1]) / 2.0

    def mutate(index: int, frame: SimFrame) -> tuple[np.ndarray, np.ndarray, BarState3D | None]:
        confidences = frame.confidences
        if -(frame.bar.left_end_m[1] + frame.bar.right_end_m[1]) / 2.0 - floor_bar_up > KNEES_HIDDEN_ABOVE_M:
            confidences = confidences.copy()
            confidences[list(KNEES)] = 0.0
        return frame.points, confidences, frame.bar

    return mutate


def _analyse(
    scenario: Scenario,
    gravity_source: str = GRAVITY_SOURCE_MEASURED,
    meta: dict | None = None,
    mutate: FrameMutation | None = None,
    legs_unmeasured: list[tuple[float, float]] | None = None,
) -> tuple[SimulatedSet, list[DeadliftRepFeatures], DeadliftRepAnalyzer, list[DeadliftPhase]]:
    sim = simulate(scenario)
    analyzer = DeadliftRepAnalyzer(BiomechanicsConfig().deadlift)
    up = sim.gravity_up_world if gravity_source == GRAVITY_SOURCE_MEASURED else np.array(WORLD_UP)
    analyzer.set_gravity(up, gravity_source)
    if meta is not None:
        analyzer.set_session_meta(meta)
    features: list[DeadliftRepFeatures] = []
    phases: list[DeadliftPhase] = []
    for index, frame in enumerate(sim.frames):
        points, confidences, bar = (frame.points, frame.confidences, frame.bar) if mutate is None else mutate(index, frame)
        carried = any(start <= frame.timestamp < end for start, end in legs_unmeasured or [])
        analyzer.observe(DeadliftFrameInput(frame.timestamp, frame.frame_index, points, confidences, bar,
                                            legs_measured=not carried))
        if not phases or phases[-1] != analyzer.phase:
            phases.append(analyzer.phase)
        while analyzer.take_completed_rep() is not None:
            features.append(analyzer.finish_rep())
    return sim, features, analyzer, phases


def _faults(features: list[DeadliftRepFeatures]) -> list[list[FaultEvent]]:
    """Each rep's faults, judged as the pipeline does."""
    engine = RuleEngine(
        rules=DeadliftProfile().create_fault_rules(BiomechanicsConfig()),
        reference=DeadliftSessionReference(),
        capture_mode="triangulated",
    )
    return [
        engine.finish_rep(JointAngles(timestamp=rep_features.top_time), rep_features.rep_number, rep_features)
        for rep_features in features
    ]


def _judge(features: list[DeadliftRepFeatures]) -> list[dict[str, str]]:
    """Each rep's faults as {fault_type: severity}."""
    return [{fault.fault_type: fault.severity.value for fault in faults} for faults in _faults(features)]


def _cued(features: list[DeadliftRepFeatures]) -> list[set[str]]:
    """Each rep's faults the voice agent would speak: at or above their min tier."""
    return [
        {fault.fault_type for fault in faults if TIER_RANK[fault.severity.value] >= TIER_RANK[fault.details["min_tier"]]}
        for faults in _faults(features)
    ]


class TestRepCounting:
    def test_clean_set_counts_each_rep_once(self):
        sim, features, analyzer, _ = _analyse(Scenario())
        assert [f.rep_number for f in features] == [1, 2, 3]
        assert analyzer.rep_count == len(sim.reps)
        assert analyzer.failed_reps == 0

    def test_touch_and_go_reps_are_counted_once_each(self):
        scenario = Scenario(reps=[RepScript(floor_hold_s=0.0), RepScript(floor_hold_s=0.0), RepScript()])
        _, features, _, _ = _analyse(scenario)
        assert [f.rep_number for f in features] == [1, 2, 3]
        assert [f.touch_and_go for f in features] == [False, True, True]

    def test_dropped_bar_bounce_is_not_a_rep(self):
        _, features, analyzer, _ = _analyse(Scenario(reps=[RepScript(drop_bar=True), RepScript()]))
        assert analyzer.rep_count == 2
        assert not any(f.touch_and_go for f in features)

    def test_failed_rep_is_an_event_not_a_rep(self):
        scenario = Scenario(reps=[RepScript(), RepScript(fail_rise_m=0.25), RepScript()])
        _, features, analyzer, _ = _analyse(scenario)
        assert analyzer.rep_count == 2
        assert analyzer.failed_reps == 1
        assert len(features) == 2

    def test_quick_repull_judges_the_setup_from_the_moment_before_liftoff(self):
        scenario = Scenario(reps=[RepScript(floor_hold_s=0.3), RepScript(floor_hold_s=0.3), RepScript()])
        _, features, _, _ = _analyse(scenario)
        assert len(features) == 3
        assert all(f.setup_measured for f in features)

    def test_a_repull_too_quick_to_see_leaves_the_setup_unmeasured(self):
        scenario = Scenario(reps=[RepScript(floor_hold_s=0.15), RepScript(floor_hold_s=0.15), RepScript()])
        _, features, _, _ = _analyse(scenario)
        assert len(features) == 3
        assert [f.setup_measured for f in features] == [True, False, False]
        assert math.isnan(features[1].bar_midfoot_setup_cm)

    def test_noisy_keypoints_and_bar_still_count_every_rep(self):
        _, features, _, _ = _analyse(Scenario(keypoint_noise_m=0.006, bar_noise_m=0.003))
        assert len(features) == 3

    def test_wrists_stand_in_for_an_untracked_bar(self):
        _, features, _, _ = _analyse(Scenario(track_bar=False))
        assert len(features) == 3
        assert {f.bar_source for f in features} == {BAR_SOURCE_WRIST_PROXY}


class TestEventTiming:
    @pytest.mark.parametrize(
        "scenario",
        [
            Scenario(),
            Scenario(reps=[RepScript(floor_hold_s=0.0), RepScript()]),
            Scenario(keypoint_noise_m=0.006, bar_noise_m=0.003),
            Scenario(track_bar=False),
        ],
        ids=["dead_stop", "touch_and_go", "noisy", "wrist_proxy"],
    )
    def test_liftoff_knee_pass_top_and_floor_land_within_100_ms(self, scenario: Scenario):
        sim, features, _, _ = _analyse(scenario)
        truth = [rep for rep in sim.reps if rep.counted]
        assert len(features) == len(truth)
        for expected, measured in zip(truth, features):
            assert measured.liftoff_time == pytest.approx(expected.liftoff_time, abs=MAX_EVENT_ERROR_S)
            assert measured.knee_pass_time == pytest.approx(expected.knee_pass_time, abs=MAX_EVENT_ERROR_S)
            assert measured.top_time == pytest.approx(expected.top_time, abs=MAX_EVENT_ERROR_S)
            assert measured.floor_time == pytest.approx(expected.floor_time, abs=MAX_EVENT_ERROR_S)


class TestPhases:
    def test_a_set_walks_through_every_phase_in_order(self):
        _, _, _, phases = _analyse(Scenario(reps=[RepScript()]))
        assert phases[:7] == [
            DeadliftPhase.APPROACH, DeadliftPhase.STANCE, DeadliftPhase.SETUP, DeadliftPhase.PULL,
            DeadliftPhase.TOP, DeadliftPhase.LOWER, DeadliftPhase.FLOOR,
        ]

    def test_standing_up_and_walking_away_returns_to_approach(self):
        _, _, analyzer, phases = _analyse(Scenario(reps=[RepScript()]))
        assert DeadliftPhase.STANCE in phases[7:]
        assert analyzer.phase == DeadliftPhase.APPROACH

    def test_live_bar_offset_is_reported_only_while_standing_at_the_bar(self):
        sim = simulate(Scenario(bar_midfoot_offset_m=0.08))
        analyzer = DeadliftRepAnalyzer()
        analyzer.set_gravity(sim.gravity_up_world, GRAVITY_SOURCE_MEASURED)
        live_by_phase: dict[DeadliftPhase, list[float]] = {}
        before_the_pull: list[float] = []
        for frame in sim.frames:
            analyzer.observe(DeadliftFrameInput(frame.timestamp, frame.frame_index, frame.points, frame.confidences, frame.bar))
            live_by_phase.setdefault(analyzer.status.phase, []).append(analyzer.status.bar_midfoot_live_cm)
            if analyzer.status.phase == DeadliftPhase.STANCE and frame.timestamp < sim.reps[0].liftoff_time:
                before_the_pull.append(analyzer.status.bar_midfoot_live_cm)
        # Standing at the bar it reads, once a median of settled frames is in;
        # hinging down to it (moving) it does not.
        measured = [value for value in before_the_pull if math.isfinite(value)]
        assert all(math.isnan(value) for value in before_the_pull[:LIVE_OFFSET_MIN_FRAMES - 1])
        assert len(measured) >= len(before_the_pull) // 3
        assert np.median(measured) == pytest.approx(8.0, abs=0.5)
        assert all(math.isnan(value) for value in live_by_phase[DeadliftPhase.PULL])

    def test_no_live_offset_after_a_rep_whose_top_went_unseen(self):
        """Every top lost from view, with re-setups between reps: a counted rep ends
        the guidance whether or not its top's height was seen."""
        scenario = Scenario(reps=[RepScript(pull_s=1.2)] * 3, bar_noise_m=TRACKED_BAR_NOISE_M, seed=1,
                            stand_between_reps_s=2.0)
        sim = simulate(scenario)
        mutate = _tracker_coasting([(rep.top_time - 0.2, rep.top_time + 0.8) for rep in sim.reps])
        analyzer = DeadliftRepAnalyzer()
        analyzer.set_gravity(sim.gravity_up_world, GRAVITY_SOURCE_MEASURED)
        after_a_rep: list[float] = []
        for index, frame in enumerate(sim.frames):
            points, confidences, bar = mutate(index, frame)
            analyzer.observe(DeadliftFrameInput(frame.timestamp, frame.frame_index, points, confidences, bar))
            while analyzer.take_completed_rep() is not None:
                analyzer.finish_rep()
            if analyzer.rep_count > 0:
                after_a_rep.append(analyzer.status.bar_midfoot_live_cm)
        assert analyzer.rep_count == len(sim.reps)
        assert after_a_rep and all(math.isnan(value) for value in after_a_rep)

    def test_no_live_offset_after_the_sets_first_rep(self):
        """Standing up and walking off after a set is not a lifter to guide in."""
        sim = simulate(Scenario(reps=[RepScript()]))
        analyzer = DeadliftRepAnalyzer()
        analyzer.set_gravity(sim.gravity_up_world, GRAVITY_SOURCE_MEASURED)
        after_the_rep: list[float] = []
        for frame in sim.frames:
            analyzer.observe(DeadliftFrameInput(frame.timestamp, frame.frame_index, frame.points, frame.confidences, frame.bar))
            if analyzer.rep_count > 0:
                after_the_rep.append(analyzer.status.bar_midfoot_live_cm)
        assert after_the_rep and all(math.isnan(value) for value in after_the_rep)

    def test_rep_signal_is_bar_height_above_its_rest(self):
        sim = simulate(Scenario(reps=[RepScript()]))
        analyzer = DeadliftRepAnalyzer()
        analyzer.set_gravity(sim.gravity_up_world, GRAVITY_SOURCE_MEASURED)
        peak_cm = -math.inf
        for frame in sim.frames:
            analyzer.observe(DeadliftFrameInput(frame.timestamp, frame.frame_index, frame.points, frame.confidences, frame.bar))
            if math.isfinite(analyzer.rep_signal):
                peak_cm = max(peak_cm, analyzer.rep_signal)
        assert analyzer.rep_signal == pytest.approx(0.0, abs=0.5)
        assert 50.0 < peak_cm < 70.0


class TestCleanFeatures:
    def test_a_clean_rep_measures_no_fault(self):
        _, features, _, _ = _analyse(Scenario())
        for rep in features:
            assert rep.bar_drift_cm < CLEAN_DRIFT_MAX_CM
            assert rep.bar_tilt_cm < CLEAN_TILT_MAX_CM
            assert abs(rep.hip_shift_ratio) < CLEAN_SHIFT_MAX_RATIO
            assert rep.trunk_change_liftoff_knee_deg == pytest.approx(0.0, abs=CLEAN_ANGLE_TOLERANCE_DEG)
            assert rep.lean_back_deg == pytest.approx(0.0, abs=CLEAN_ANGLE_TOLERANCE_DEG)
            assert rep.bar_midfoot_setup_cm == pytest.approx(0.0, abs=0.5)

    def test_a_good_setup_sits_inside_the_models_hip_band(self):
        _, features, _, _ = _analyse(Scenario())
        for rep in features:
            assert rep.setup_hip_band_low_cm <= rep.setup_hip_height_cm <= rep.setup_hip_band_high_cm

    def test_stance_bar_offset_belongs_to_the_first_rep_after_the_stance(self):
        _, features, _, _ = _analyse(Scenario())
        assert math.isfinite(features[0].bar_midfoot_stance_cm)
        assert all(math.isnan(rep.bar_midfoot_stance_cm) for rep in features[1:])

    def test_touch_and_go_reps_have_no_setup(self):
        _, features, _, _ = _analyse(Scenario(reps=[RepScript(floor_hold_s=0.0), RepScript()]))
        assert not features[1].setup_measured
        assert math.isnan(features[1].bar_midfoot_setup_cm)

    def test_the_grip_from_the_session_metadata_is_recorded(self):
        _, features, _, _ = _analyse(Scenario(reps=[RepScript()]), meta={"grip": "mixed"})
        assert features[0].grip == "mixed"

    def test_measured_gravity_keeps_a_tilted_world_from_faking_drift(self):
        tilted = Scenario(world_tilt_deg=3.0, reps=[RepScript()])
        _, measured, _, _ = _analyse(tilted, GRAVITY_SOURCE_MEASURED)
        _, body, _, _ = _analyse(tilted, GRAVITY_SOURCE_BODY)
        assert measured[0].bar_drift_cm < CLEAN_DRIFT_MAX_CM
        assert body[0].bar_drift_cm > TILT_FAKE_DRIFT_MIN_CM
        assert body[0].gravity_source == GRAVITY_SOURCE_BODY

    def test_losing_the_bar_mid_set_never_switches_to_the_wrists(self):
        sim = simulate(Scenario(reps=[RepScript(), RepScript()]))
        analyzer = DeadliftRepAnalyzer()
        analyzer.set_gravity(sim.gravity_up_world, GRAVITY_SOURCE_MEASURED)
        sources = set()
        for index, frame in enumerate(sim.frames):
            # The bar vanishes for the second half of the set.
            bar = frame.bar if index < len(sim.frames) // 2 else None
            analyzer.observe(DeadliftFrameInput(frame.timestamp, frame.frame_index, frame.points, frame.confidences, bar))
            if analyzer.phase != DeadliftPhase.APPROACH:
                sources.add(analyzer.status.bar_source)
        assert sources == {BAR_SOURCE_BAR}


class TestFaultScenarios:
    @pytest.mark.parametrize(
        ("scenario", "fault_type"),
        [
            (Scenario(bar_midfoot_offset_m=0.06, reps=[RepScript()] * 2), "deadlift_bar_position"),
            (Scenario(reps=[RepScript(shoulder_ahead_m=-0.05)] * 2), "deadlift_shoulders_behind"),
            (Scenario(reps=[RepScript(shoulder_ahead_m=0.14)] * 2), "deadlift_setup_hips"),
            # Hips first: the trunk tips forward 15 deg off the floor, the hips out-rising
            # the chest beyond what noise reaches (a milder one waits for a second rep).
            (Scenario(reps=[RepScript(setup_trunk_deg=40.0, knee_pass_trunk_deg=55.0)] * 2), "deadlift_hips_shoot"),
            (Scenario(reps=[RepScript(bar_drift_m=0.06)] * 2), "deadlift_bar_drift"),
            (Scenario(reps=[RepScript(lockout_deficit_deg=14.0)] * 2), "deadlift_lockout"),
            (Scenario(reps=[RepScript(lean_back_deg=13.0)] * 2), "deadlift_lean_back"),
            (Scenario(reps=[RepScript(hip_shift_m=0.05)] * 2), "deadlift_hip_shift"),
            (Scenario(reps=[RepScript(bar_tilt_m=0.06)] * 2), "deadlift_bar_tilt"),
            (Scenario(reps=[RepScript(elbow_bend_deg=30.0)] * 2), "deadlift_bent_arms"),
        ],
        ids=["D1", "D7", "D4", "D2", "D3", "D6", "D5", "D8", "D8b", "D9"],
    )
    def test_each_injected_fault_fires_at_moderate_or_worse_on_every_rep(self, scenario: Scenario, fault_type: str):
        _, features, _, _ = _analyse(scenario)
        verdicts = _judge(features)
        assert len(verdicts) == len(scenario.reps)
        for verdict in verdicts:
            assert verdict.get(fault_type) in ("moderate", "severe")

    def test_a_tilted_bar_fires_on_the_wrist_proxy_at_its_wider_thresholds_from_the_second_rep(self):
        _, features, _, _ = _analyse(Scenario(track_bar=False, reps=[RepScript(bar_tilt_m=0.09)] * 3))
        verdicts = _faults(features)
        assert all(fault.fault_type != "deadlift_bar_tilt" for fault in verdicts[0])
        for faults in verdicts[1:]:
            tilt = next(fault for fault in faults if fault.fault_type == "deadlift_bar_tilt")
            assert tilt.severity.value == "moderate"
            assert tilt.details["bar_source"] == BAR_SOURCE_WRIST_PROXY

    def test_velocity_loss_fires_on_the_slow_rep_only(self):
        scenario = Scenario(reps=[RepScript(pull_s=1.0), RepScript(pull_s=1.0), RepScript(pull_s=1.6)])
        _, features, _, _ = _analyse(scenario)
        verdicts = _judge(features)
        assert "deadlift_velocity_loss" not in verdicts[0]
        assert "deadlift_velocity_loss" not in verdicts[1]
        assert verdicts[2]["deadlift_velocity_loss"] in ("moderate", "severe")

    @pytest.mark.parametrize(
        "scenario",
        [Scenario(), Scenario(keypoint_noise_m=0.006, bar_noise_m=0.003), Scenario(world_tilt_deg=3.0)],
        ids=["clean", "noisy", "tilted_world"],
    )
    def test_a_clean_set_is_fault_free(self, scenario: Scenario):
        _, features, _, _ = _analyse(scenario)
        assert _judge(features) == [{}] * len(features)


class TestBodiesAndSetups:
    """The setup model is the analyser's, so these vary what it must measure rather
    than assume: segment proportions, and how far in front of the shins the bar
    sits (shin thickness, how hard the shins press the bar)."""

    @pytest.mark.parametrize(
        "athlete",
        [
            SimAthlete(),
            SimAthlete(torso_m=0.58, upper_arm_m=0.28, forearm_m=0.27),
            SimAthlete(femur_m=0.50, tibia_m=0.45),
            SimAthlete(tibia_m=0.38, femur_m=0.40, torso_m=0.47, upper_arm_m=0.27, forearm_m=0.26),
        ],
        ids=["average", "long_torso_short_arms", "long_femurs", "short"],
    )
    @pytest.mark.parametrize("shin_bar_m", [0.03, 0.05, 0.07])
    def test_a_clean_set_is_fault_free_whatever_the_body_and_shin_contact(self, athlete: SimAthlete, shin_bar_m: float):
        _, features, _, _ = _analyse(Scenario(athlete=athlete, shin_bar_m=shin_bar_m, reps=[RepScript()] * 2))
        assert len(features) == 2
        assert _judge(features) == [{}, {}]
        for rep in features:
            assert rep.trunk_change_liftoff_knee_deg == pytest.approx(0.0, abs=CLEAN_ANGLE_TOLERANCE_DEG)
            assert rep.setup_hip_band_low_cm <= rep.setup_hip_height_cm <= rep.setup_hip_band_high_cm


class TestPlatformNoise:
    """Keypoints as noisy as the platform's own, i.i.d. and correlated (the slow
    wander a Kalman leaves): every rep counted, the stance reached for foot
    guidance, events on time and nothing cued on a clean set."""

    @pytest.mark.parametrize("seed", range(4))
    def test_independent_noise_counts_every_rep_on_time_and_cues_nothing(self, seed: int):
        scenario = Scenario(keypoint_noise_m=PLATFORM_KEYPOINT_NOISE_M, bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed)
        sim, features, _, phases = _analyse(scenario)
        assert len(features) == len(sim.reps)
        assert DeadliftPhase.STANCE in phases
        for expected, measured in zip(sim.reps, features):
            assert measured.liftoff_time == pytest.approx(expected.liftoff_time, abs=MAX_EVENT_ERROR_S)
            assert measured.top_time == pytest.approx(expected.top_time, abs=MAX_EVENT_ERROR_S)
            assert measured.floor_time == pytest.approx(expected.floor_time, abs=MAX_EVENT_ERROR_S)
        assert _cued(features) == [set()] * len(features)

    @pytest.mark.parametrize("seed", range(4))
    def test_correlated_noise_counts_every_rep_and_cues_nothing(self, seed: int):
        scenario = Scenario(bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed)
        sim, features, _, phases = _analyse(scenario, mutate=_correlated_noise(PLATFORM_KEYPOINT_NOISE_M, 0.8, seed))
        assert len(features) == len(sim.reps)
        assert DeadliftPhase.STANCE in phases
        assert _cued(features) == [set()] * len(features)

    def test_noisy_wrist_proxy_counts_every_rep(self):
        sim, features, _, _ = _analyse(Scenario(track_bar=False, keypoint_noise_m=0.01))
        assert len(features) == len(sim.reps)
        assert {f.bar_source for f in features} == {BAR_SOURCE_WRIST_PROXY}


class TestSetupHolds:
    """Grip and rip: the first rep counts however short the setup pause."""

    @pytest.mark.parametrize("hold_s", [0.0, 0.1, 0.2, 0.3])
    @pytest.mark.parametrize("hinge_s", [0.5, 1.0])
    def test_a_pull_without_a_setup_hold_still_counts(self, hold_s: float, hinge_s: float):
        sim, features, _, _ = _analyse(Scenario(setup_hold_s=hold_s, hinge_s=hinge_s))
        assert len(features) == len(sim.reps)
        assert features[0].liftoff_time == pytest.approx(sim.reps[0].liftoff_time, abs=MAX_EVENT_ERROR_S)

    def test_a_grip_and_rip_judges_the_moment_before_liftoff(self):
        _, features, _, _ = _analyse(Scenario(setup_hold_s=0.0, hinge_s=1.0))
        assert features[0].setup_measured
        assert features[0].bar_midfoot_setup_cm == pytest.approx(0.0, abs=1.0)


class TestOcclusionAndDropouts:
    def test_plates_hiding_the_feet_through_the_pull_lose_no_rep(self):
        sim, features, _, _ = _analyse(Scenario(plates_hide_feet_above_m=0.03))
        assert len(features) == len(sim.reps)
        assert _judge(features) == [{}] * len(features)

    def test_feet_hidden_from_the_setup_on_still_count(self):
        def hide_feet_after_the_stance(index: int, frame: SimFrame) -> tuple[np.ndarray, np.ndarray, BarState3D | None]:
            confidences = frame.confidences.copy()
            if frame.timestamp - sim.frames[0].timestamp > 3.5:
                confidences[[CK.LEFT_ANKLE, CK.RIGHT_ANKLE, CK.LEFT_FOOT_INDEX, CK.RIGHT_FOOT_INDEX]] = 0.0
            return frame.points, confidences, frame.bar

        sim = simulate(Scenario())
        _, features, _, _ = _analyse(Scenario(), mutate=hide_feet_after_the_stance)
        assert len(features) == len(sim.reps)

    @pytest.mark.parametrize("probability", [0.2, 0.4])
    def test_random_bar_dropouts_lose_no_rep_and_never_switch_to_the_wrists(self, probability: float):
        sim, features, _, _ = _analyse(Scenario(), mutate=_bar_dropout(probability, seed=2))
        assert len(features) == len(sim.reps)
        assert {f.bar_source for f in features} == {BAR_SOURCE_BAR}

    @pytest.mark.parametrize("probability", [0.1, 0.2])
    def test_random_bar_dropouts_do_not_move_the_top(self, probability: float):
        """A dropped bar frame is no height: not a flat stretch of the climb."""
        errors = []
        for body in BODIES[:3]:
            for seed in range(4):
                scenario = Scenario(athlete=body, reps=[RepScript(pull_s=2.5)] * 3, bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed)
                sim, features, _, _ = _analyse(scenario, mutate=_bar_dropout(probability, seed=50 + seed))
                assert len(features) == len(sim.reps)
                errors += [abs(measured.top_time - expected.top_time) for measured, expected in zip(features, sim.reps)]
        assert max(errors) <= MAX_EVENT_ERROR_S

    def test_a_bar_lost_as_it_arrives_at_the_top_dates_the_top_in_the_gap(self):
        """The fit may date the arrival from the gap's first frame, not only from
        the first frame the bar is seen again."""
        errors = []
        for seed in range(3):
            scenario = Scenario(reps=[RepScript(pull_s=1.5)] * 3, bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed)
            sim = simulate(scenario)
            gaps = [(rep.top_time - 0.1, rep.top_time + 0.3) for rep in sim.reps]

            def lose_the_bar(index: int, frame: SimFrame) -> tuple[np.ndarray, np.ndarray, BarState3D | None]:
                lost = any(start <= frame.timestamp < end for start, end in gaps)
                return frame.points, frame.confidences, None if lost else frame.bar

            _, features, _, _ = _analyse(scenario, mutate=lose_the_bar)
            assert len(features) == len(sim.reps)
            errors += [abs(measured.top_time - expected.top_time) for measured, expected in zip(features, sim.reps)]
        assert max(errors) <= MAX_EVENT_ERROR_S

    @pytest.mark.parametrize("lost_before_top_s", [0.5, 0.9])
    def test_a_bar_lost_low_on_its_climb_and_found_at_the_top_is_dated_near_the_top(self, lost_before_top_s: float):
        """Lost with most of the climb to go, the bar's last seen height and speed
        say when it could have arrived (slowing evenly into the top): not the
        gap's middle, which reads the rep fast and the others as a velocity loss."""
        errors = []
        slowed = 0
        for body in BODIES[:3]:
            for seed in range(4):
                scenario = Scenario(athlete=body, reps=[RepScript(pull_s=1.2)] * 4, bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed)
                top = simulate(scenario).reps[1].top_time

                def lose_the_bar(index: int, frame: SimFrame) -> tuple[np.ndarray, np.ndarray, BarState3D | None]:
                    lost = top - lost_before_top_s <= frame.timestamp < top + 0.1
                    return frame.points, frame.confidences, None if lost else frame.bar

                sim, features, _, _ = _analyse(scenario, mutate=lose_the_bar)
                assert len(features) == len(sim.reps)
                errors.append(features[1].top_time - top)
                slowed += any("deadlift_velocity_loss" in judged for judged in _judge(features))
        assert max(map(abs, errors)) <= GAP_TOP_MAX_ERROR_S, errors
        assert slowed == 0

    @pytest.mark.parametrize(("lost_before_top_s", "lost_s"), [(0.7, 1.0), (0.3, 0.6)])
    def test_a_bar_lost_through_a_grinds_finish_is_dated_by_the_joints(self, lost_before_top_s: float, lost_s: float):
        """Lost still in the stall, or speeding up out of it, the bar's last speed
        says nothing of when it finished: the hips and knees, still seen, reaching
        the lockout date it (the gap's middle read up to 0.18 s early, the frame
        it was seen again 0.33 s late)."""
        errors = []
        for body in BODIES[:3]:
            for seed in range(3):
                scenario = Scenario(athlete=body, reps=[GRIND_93] * 3, bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed)
                gaps = [(rep.top_time - lost_before_top_s, rep.top_time - lost_before_top_s + lost_s)
                        for rep in simulate(scenario).reps]

                def lose_the_bar(index: int, frame: SimFrame) -> tuple[np.ndarray, np.ndarray, BarState3D | None]:
                    lost = any(start <= frame.timestamp < end for start, end in gaps)
                    return frame.points, frame.confidences, None if lost else frame.bar

                sim, features, _, _ = _analyse(scenario, mutate=lose_the_bar)
                assert len(features) == len(sim.reps)
                errors += [measured.top_time - expected.top_time for measured, expected in zip(features, sim.reps)]
        assert max(map(abs, errors)) <= GAP_TOP_MAX_ERROR_S, errors

    @pytest.mark.parametrize("lost_after_top_s", [0.1, 0.8], ids=["seen_holding", "lost_until_the_lowering"])
    @pytest.mark.parametrize("loss", LOSSES)
    @pytest.mark.parametrize(("stall_s", "finish_s", "lost_before_stall_s"), [(0.4, 0.3, 0.1), (0.6, 0.4, 0.2)])
    def test_a_bar_lost_on_its_way_into_a_stall_is_dated_at_the_top(
        self, stall_s: float, finish_s: float, lost_before_stall_s: float, loss: str, lost_after_top_s: float,
    ):
        """Lost still moving, its last speed would put the arrival inside the
        stall; the hips and knees, still bent like a stall there, say it had not
        arrived (the speed alone read 0.3-0.8 s early; lost until the lowering, the
        bar's fitted rise, -0.78 s)."""
        grind = RepScript(pull_s=1.6, stall_fraction=0.93, stall_s=stall_s, finish_s=finish_s)
        errors = []
        for body in BODIES[:3]:
            for seed in range(3):
                scenario = Scenario(athlete=body, reps=[RepScript(pull_s=1.6), grind, RepScript(pull_s=1.6)],
                                    bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed)
                top = simulate(scenario).reps[1].top_time
                lose_the_bar = LOSSES[loss]([(top - finish_s - stall_s - lost_before_stall_s, top + lost_after_top_s)])
                sim, features, _, _ = _analyse(scenario, mutate=lose_the_bar)
                assert len(features) == len(sim.reps)
                errors.append(features[1].top_time - top)
        assert max(map(abs, errors)) <= GAP_TOP_MAX_ERROR_S, errors

    def test_a_bar_the_tracker_coasts_over_at_the_top_neither_moves_the_top_nor_fakes_a_soft_lockout(self):
        """The tracker carries a lost bar on at its last velocity: past the frame
        between two detections, the prediction overshoots a bar that stops, and
        it is no height (as a prediction the top read 0.13-0.17 s early, with a
        false D6 on 4 of 12)."""
        errors = []
        cued = 0
        for body in BODIES[:3] + BODIES[4:]:
            for seed in range(3):
                scenario = Scenario(athlete=body, reps=[RepScript(pull_s=1.2)] * 4, bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed)
                top = simulate(scenario).reps[1].top_time
                sim, features, _, _ = _analyse(scenario, mutate=_tracker_coasting([(top - 0.3, top + 0.1)]))
                assert len(features) == len(sim.reps)
                errors.append(features[1].top_time - top)
                cued += "deadlift_lockout" in _cued(features)[1]
        assert max(map(abs, errors)) <= MAX_EVENT_ERROR_S, errors
        assert cued == 0

    @pytest.mark.parametrize("loss", LOSSES)
    def test_a_bar_lost_across_a_top_that_never_held_is_judged_at_the_top(self, loss: str):
        """The highest frame seen is on the climb: the lockout is judged where the
        hips and knees were most extended while the bar was unseen, and a gap is
        no hold (judged on the climb, 30 of 30 reps were cued D6)."""
        cued = reps = 0
        errors = []
        for body in BODIES:
            for seed in range(2):
                scenario = Scenario(athlete=body, reps=_no_pause_touch_and_go(1.2, 1.0), bar_noise_m=TRACKED_BAR_NOISE_M,
                                    seed=seed)
                lost = simulate(scenario).reps[1:4]
                mutate = LOSSES[loss]([(rep.top_time - 0.2, rep.top_time + 0.2) for rep in lost])
                sim, features, _, _ = _analyse(scenario, mutate=mutate)
                assert len(features) == len(sim.reps)
                reps += len(lost)
                cued += sum("deadlift_lockout" in rep_cues for rep_cues in _cued(features)[1:4])
                errors += [measured.top_time - expected.top_time for measured, expected in zip(features[1:4], lost)]
        assert max(map(abs, errors)) <= MAX_EVENT_ERROR_S, errors
        assert cued <= FALSE_CUE_GATE_RATIO * reps, f"D6 on {cued} of {reps}"

    @pytest.mark.parametrize("loss", LOSSES)
    def test_a_bar_lost_across_a_fast_top_that_never_held_loses_no_rep(self, loss: str):
        """The highest frame seen sits short of the expected top: the lifter seen
        standing while the bar was unseen makes it a top (three reps merged)."""
        for body in BODIES[:3]:
            scenario = Scenario(athlete=body, reps=_no_pause_touch_and_go(0.9, 0.8), bar_noise_m=TRACKED_BAR_NOISE_M)
            mutate = LOSSES[loss]([(rep.top_time - 0.2, rep.top_time + 0.2) for rep in simulate(scenario).reps[1:4]])
            sim, features, _, _ = _analyse(scenario, mutate=mutate)
            assert len(features) == len(sim.reps)

    @pytest.mark.parametrize(
        ("pull_s", "lost_from_s", "lost_to_s"),
        [(1.2, -0.1, 0.7), (1.2, 0.03, 0.75), (1.2, -0.3, 0.8), (1.5, -0.3, 0.8), (2.0, -0.4, 0.8), (2.5, -0.4, 0.8),
         (0.6, -0.3, 0.8), (0.6, -0.2, 0.8)],
        ids=["before_the_top", "after_arriving", "lost_earlier", "slower_pull", "two_second_pull", "heavy_pull",
             "quick_pull_from_its_middle", "quick_pull_from_a_third"],
    )
    def test_a_hold_lost_from_view_until_the_lowering_is_dated_at_the_top(
        self, pull_s: float, lost_from_s: float, lost_to_s: float,
    ):
        """The hold is never seen, and the hips and knees are flat through it: the
        top is where the bar's last rise, slowing into the expected top height,
        arrives (dated at the joints' most extended frame, anywhere in the hold, it
        read up to 0.37 s late and the recap a velocity loss in every set; by a
        parabola's free vertex, up to 0.56 s late on 2 s pulls). Seen again coming
        down higher than it was last seen going up, the same. A quick pull's last
        0.3 s seen is not yet slowing into the top: the fit overshoots it (up to
        +0.59 s), and the hips and knees reaching their plateau date it."""
        errors = []
        slowed = 0
        for body in BODIES:
            for seed in range(3):
                scenario = Scenario(athlete=body, reps=[RepScript(pull_s=pull_s)] * 4, bar_noise_m=TRACKED_BAR_NOISE_M,
                                    seed=seed)
                lost = simulate(scenario).reps[1:3]
                mutate = _tracker_coasting([(rep.top_time + lost_from_s, rep.top_time + lost_to_s) for rep in lost])
                sim, features, _, _ = _analyse(scenario, mutate=mutate)
                assert len(features) == len(sim.reps)
                errors += [measured.top_time - expected.top_time for measured, expected in zip(features[1:3], lost)]
                slowed += any("deadlift_velocity_loss" in judged for judged in _judge(features))
        assert max(map(abs, errors)) <= GAP_TOP_MAX_ERROR_S, errors
        assert slowed == 0

    def test_a_top_dated_in_a_gap_has_no_bar_speed(self):
        """An estimate: no concentric velocity for the recap's velocity loss to read;
        the reps whose tops were seen keep theirs."""
        scenario = Scenario(reps=[RepScript(pull_s=2.0)] * 3, bar_noise_m=TRACKED_BAR_NOISE_M)
        lost = simulate(scenario).reps[1]
        sim, features, _, _ = _analyse(scenario, mutate=_tracker_coasting([(lost.top_time - 0.4, lost.top_time + 0.8)]))
        assert len(features) == len(sim.reps)
        assert math.isnan(features[1].concentric_velocity_mps)
        assert math.isfinite(features[0].concentric_velocity_mps) and math.isfinite(features[2].concentric_velocity_mps)

    @pytest.mark.parametrize(("pull_s", "lost_from_s"), [(1.2, -0.4), (1.5, -0.3), (2.0, -0.4), (2.5, -0.3)])
    def test_holds_lost_from_view_under_keypoint_noise_are_rarely_dated_far_off(self, pull_s: float, lost_from_s: float):
        """At 1.5 cm of correlated keypoint noise the fit's expected top height (the
        standing pose's) and the joints' veto carry the noise: a top is dated beyond
        0.3 s on at most 1 rep in 20, and none reads a velocity loss (an estimate has
        no speed)."""
        errors = []
        slowed = 0
        for body_index, body in enumerate(BODIES):
            for seed in range(2):
                scenario = Scenario(athlete=body, reps=[RepScript(pull_s=pull_s)] * 3, bar_noise_m=TRACKED_BAR_NOISE_M,
                                    seed=seed)
                lost = _tracker_coasting([(rep.top_time + lost_from_s, rep.top_time + 0.8) for rep in simulate(scenario).reps])
                noise = _correlated_noise(PROXY_KEYPOINT_NOISE_M, PLATFORM_NOISE_RHO, 2600 + 10 * body_index + seed)
                sim, features, _, _ = _analyse(scenario, mutate=_then(lost, noise))
                assert len(features) == len(sim.reps)
                errors += [measured.top_time - expected.top_time for measured, expected in zip(features, sim.reps)]
                slowed += any("deadlift_velocity_loss" in judged for judged in _judge(features))
        far = [error for error in errors if abs(error) > 2.0 * GAP_TOP_MAX_ERROR_S]
        assert len(far) <= LOST_TOP_FAR_RATIO * len(errors), errors
        assert slowed == 0

    def test_joints_wandering_through_a_lost_hold_do_not_veto_its_fitted_top(self):
        """Through a hold the keypoint noise wanders the hips and knees' angles by
        ~15 deg: their least is a dip, and against it a slow pull's last extension
        read as a stall, so the fit was vetoed and the joints dated the top late.
        The veto reads them against their plateau in the gap."""
        errors = []
        for body_index, body in enumerate(BODIES):
            for seed in range(2):
                scenario = Scenario(athlete=body, reps=[RepScript(pull_s=1.5)] * 3, bar_noise_m=TRACKED_BAR_NOISE_M,
                                    seed=seed)
                lost = _tracker_coasting([(rep.top_time - 0.3, rep.top_time + 0.8) for rep in simulate(scenario).reps])
                noise = _correlated_noise(PLATFORM_KEYPOINT_NOISE_M, PLATFORM_NOISE_RHO,
                                          WANDERING_JOINTS_NOISE_SEED + 10 * body_index + seed)
                sim, features, _, _ = _analyse(scenario, mutate=_then(lost, noise))
                assert len(features) == len(sim.reps)
                errors += [measured.top_time - expected.top_time for measured, expected in zip(features, sim.reps)]
        assert max(map(abs, errors)) <= 2.0 * GAP_TOP_MAX_ERROR_S, errors

    @pytest.mark.parametrize("lost_from_s", [-0.2, 0.1], ids=["lost_before_the_top", "seen_arriving"])
    @pytest.mark.parametrize("fault", ["lean_back_deg", "lockout_deficit_deg"], ids=["leaned_back", "soft"])
    def test_a_faulted_lockout_lost_from_view_is_dated_at_its_top(self, fault: str, lost_from_s: float):
        """A leaned-back or soft lockout stops short of the expected top height the
        bar's fit runs into. Seen arriving, the hips and knees are at their plateau
        when the bar was last seen: the top seen (fitted, up to 0.6 s late). Lost
        before, they reach it well before the fit: they date it."""
        errors = []
        for body in BODIES:
            for seed in range(3):
                script = RepScript(pull_s=1.2, **{fault: FAULTED_LOCKOUT_DEG})
                scenario = Scenario(athlete=body, reps=[script] * 3, bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed)
                lost = [(rep.top_time + lost_from_s, rep.top_time + 0.8) for rep in simulate(scenario).reps]
                sim, features, _, _ = _analyse(scenario, mutate=_tracker_coasting(lost))
                assert len(features) == len(sim.reps)
                errors += [measured.top_time - expected.top_time for measured, expected in zip(features, sim.reps)]
        assert max(map(abs, errors)) <= GAP_TOP_MAX_ERROR_S, errors

    def test_every_top_lost_without_a_standing_reference_is_dated_in_the_gap(self):
        """No standing pose, no expected top height to fit to: the hips and knees
        reaching their plateau date it (the bar's last frame seen read 0.33 s early)."""
        errors = []
        for body in BODIES:
            for seed in range(3):
                scenario = Scenario(athlete=body, reps=[RepScript(pull_s=1.5)] * 3, bar_noise_m=TRACKED_BAR_NOISE_M,
                                    seed=seed, approach_s=0.5, stance_s=0.5)
                lost = [(rep.top_time - 0.3, rep.top_time + 0.8) for rep in simulate(scenario).reps]
                sim, features, _, _ = _analyse(scenario, mutate=_tracker_coasting(lost))
                assert len(features) == len(sim.reps)
                errors += [measured.top_time - expected.top_time for measured, expected in zip(features, sim.reps)]
        assert max(map(abs, errors)) <= GAP_TOP_MAX_ERROR_S, errors

    @pytest.mark.parametrize(
        ("deficit_deg", "stance_s"), [(FAULTED_LOCKOUT_DEG, 1.5), (0.0, 0.5)],
        ids=["soft_lockout", "no_standing_reference"],
    )
    def test_a_long_hold_lost_from_view_does_not_resume_on_keypoint_noise(self, deficit_deg: float, stance_s: float):
        """A 1.5 s hold seen for 0.15 s, then lost until 0.2 s into the lowering:
        nothing stalled, the hips and knees only wander with the noise. A resume
        on them is decided once, over the whole gap."""
        script = RepScript(pull_s=1.2, top_hold_s=1.5, lockout_deficit_deg=deficit_deg)
        errors = []
        for body_index, body in enumerate(BODIES):
            for seed in range(3):
                scenario = Scenario(athlete=body, reps=[script] * 3, bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed,
                                    approach_s=stance_s, stance_s=stance_s)
                lost = _tracker_coasting([(rep.top_time + 0.15, rep.top_time + 1.7) for rep in simulate(scenario).reps])
                noise = _correlated_noise(PLATFORM_KEYPOINT_NOISE_M, PLATFORM_NOISE_RHO,
                                          UNSEEN_HOLD_NOISE_SEED + 10 * body_index + seed)
                sim, features, _, _ = _analyse(scenario, mutate=_then(lost, noise))
                assert len(features) == len(sim.reps)
                errors += [measured.top_time - expected.top_time for measured, expected in zip(features, sim.reps)]
        assert max(map(abs, errors)) <= 2.0 * GAP_TOP_MAX_ERROR_S, errors

    def test_a_grind_stalled_near_lockout_and_lost_through_its_finish_mostly_resumes_under_noise(self):
        """A stall 3 % short is finished by the hips by little more than the resume
        deficit: the hips and knees together extending by twice it over the gap
        resume it more often than not at 1.5 cm (a documented limit: the rest are
        dated at the stall)."""
        errors = []
        for body_index, body in enumerate(BODIES[:3]):
            for seed in range(3):
                grind = RepScript(pull_s=1.6, stall_fraction=0.97, stall_s=0.6, finish_s=0.4)
                scenario = Scenario(athlete=body, reps=[grind] * 3, bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed)
                lost = _tracker_coasting([(rep.top_time - 0.2, rep.top_time + 0.8) for rep in simulate(scenario).reps])
                noise = _correlated_noise(PROXY_KEYPOINT_NOISE_M, PLATFORM_NOISE_RHO,
                                          NEAR_LOCKOUT_GRIND_NOISE_SEED + 10 * body_index + seed)
                sim, features, _, _ = _analyse(scenario, mutate=_then(lost, noise))
                assert len(features) == len(sim.reps)
                errors += [measured.top_time - expected.top_time for measured, expected in zip(features, sim.reps)]
        far = [error for error in errors if abs(error) > 2.0 * GAP_TOP_MAX_ERROR_S]
        assert len(far) <= NEAR_LOCKOUT_GRIND_FAR_RATIO * len(errors), errors

    @pytest.mark.parametrize("unmeasured", ["carried", "knees_hidden"])
    def test_a_hold_lost_with_the_legs_unmeasured_is_dated_by_the_bar(self, unmeasured: str):
        """The hips and knees unmeasured through the gap (carried by the pose
        tracker, or the knees hidden): the bar's fit alone dates the top (their few
        frames' plateau put it at the last frame seen, 0.2 s early, and judged the
        climb's bent knees: D6 severe on every rep)."""
        errors = []
        cued = 0
        for pull_s in (1.2, 2.0):
            for body in BODIES:
                scenario = Scenario(athlete=body, reps=[RepScript(pull_s=pull_s)] * 3,
                                    bar_noise_m=TRACKED_BAR_NOISE_M)
                lost = [(rep.top_time - 0.2, rep.top_time + 0.8) for rep in simulate(scenario).reps]
                if unmeasured == "carried":
                    sim, features, _, _ = _analyse(scenario, mutate=_tracker_coasting(lost), legs_unmeasured=lost)
                else:
                    hide = _hidden_keypoints(lost, KNEES)
                    sim, features, _, _ = _analyse(scenario, mutate=_then(_tracker_coasting(lost), hide))
                assert len(features) == len(sim.reps)
                errors += [measured.top_time - expected.top_time for measured, expected in zip(features, sim.reps)]
                cued += sum("deadlift_lockout" in rep for rep in _cued(features))
        assert cued == 0
        assert max(map(abs, errors)) <= GAP_TOP_MAX_ERROR_S, errors

    @pytest.mark.parametrize("unmeasured", ["knees_hidden", "legs_carried", "one_knee_hidden"])
    def test_a_top_dated_early_in_a_gap_does_not_judge_the_climb(self, unmeasured: str):
        """With the hips and knees too little measured in the gap (the knees
        hidden; the legs carried by the pose tracker, one knee hidden or not), the
        bar's fit alone dates the top, and on a slow pull the expected top height's
        noise can put it early: its window keeps no frame from before the gap, the
        climb seen with the knees bent (the window had tested for any knee measured
        instead, and kept the climb when one was: 7 of 45 cued)."""
        cued = 0
        for body_index, body in enumerate(BODIES):
            for seed in range(3):
                scenario = Scenario(athlete=body, reps=[RepScript(pull_s=2.0)] * 3, bar_noise_m=TRACKED_BAR_NOISE_M,
                                    seed=seed)
                lost = [(rep.top_time - 0.2, rep.top_time + 0.8) for rep in simulate(scenario).reps]
                hidden_knees = {"knees_hidden": KNEES, "legs_carried": (), "one_knee_hidden": (CK.LEFT_KNEE,)}
                hidden = _then(_tracker_coasting(lost), _hidden_keypoints(lost, hidden_knees[unmeasured]))
                noise = _correlated_noise(PROXY_KEYPOINT_NOISE_M, PLATFORM_NOISE_RHO,
                                          EARLY_FIT_NOISE_SEED + 10 * body_index + seed)
                carried = lost if unmeasured != "knees_hidden" else None
                sim, features, _, _ = _analyse(scenario, mutate=_then(hidden, noise), legs_unmeasured=carried)
                assert len(features) == len(sim.reps)
                cued += sum("deadlift_lockout" in rep for rep in _cued(features))
        assert cued == 0

    @pytest.mark.parametrize("lockout_deficit_deg", [0.0, SOFT_LOCKOUT_DEG], ids=["clean", "soft"])
    def test_a_grind_lost_into_its_stall_with_the_knees_hidden_through_it_is_dated_at_the_lockout(
        self, lockout_deficit_deg: float,
    ):
        """The bar lost on its way into a 93 % grind's stall until after the top, the
        knees hidden through the stall and seen for the finish: the fit lands in the
        stall, no joint frame lies within 0.15 s of it, and their first frame after
        it (the finish's, bent) still vetoes it (unvetoed, the top read 0.75-0.79 s
        early and a soft lockout went uncued)."""
        errors = []
        cued = 0
        grind = RepScript(pull_s=1.6, stall_fraction=0.93, stall_s=0.6, finish_s=0.4,
                          lockout_deficit_deg=lockout_deficit_deg)
        for body in BODIES:
            for seed in range(3):
                scenario = Scenario(athlete=body, reps=[grind] * 3, bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed)
                truth = simulate(scenario).reps
                lost = [(rep.top_time - 1.2, rep.top_time + 0.8) for rep in truth]
                through_the_stall = [(start, rep.top_time - 0.4) for (start, _), rep in zip(lost, truth)]
                mutate = _then(_tracker_coasting(lost), _hidden_keypoints(through_the_stall, KNEES))
                sim, features, _, _ = _analyse(scenario, mutate=mutate)
                assert len(features) == len(sim.reps)
                errors += [measured.top_time - expected.top_time for measured, expected in zip(features, sim.reps)]
                cued += sum("deadlift_lockout" in rep for rep in _cued(features))
        assert max(map(abs, errors)) <= GAP_TOP_MAX_ERROR_S, errors
        assert cued == (len(errors) if lockout_deficit_deg else 0)

    def test_legs_unmeasured_through_the_set_with_every_top_lost_lose_no_rep(self):
        """No frame of a rep with its legs measured (the pipeline's extrapolated
        knee, from before the setup): the joints' track is empty, not an error out
        of observe() (it ended the pipeline's session)."""
        scenario = Scenario(reps=[RepScript(pull_s=1.2)] * 3, bar_noise_m=TRACKED_BAR_NOISE_M)
        sim = simulate(scenario)
        whole_set = [(sim.frames[0].timestamp, sim.frames[-1].timestamp + 1.0)]
        lost = [(rep.top_time - 0.2, rep.top_time + 0.8) for rep in sim.reps]
        sim, features, _, _ = _analyse(scenario, mutate=_tracker_coasting(lost), legs_unmeasured=whole_set)
        assert len(features) == len(sim.reps)

    @pytest.mark.parametrize("feet", ["seen", "under_the_plates"])
    def test_a_no_pause_top_lost_just_after_it_is_judged_on_both_sides(self, feet: str):
        """The bar lost from 0.03 s after a no-pause top (0.8 / 0.6 s): dated at the
        last frame seen, the top keeps the climb's side of its window (judged on the
        lowering alone, D6 was cued on 12 of 75 clean reps). With the plates hiding
        the feet, the hips and knees still measure the gap."""
        cued = 0
        for body in BODIES:
            for seed in range(3):
                scenario = Scenario(athlete=body, reps=_no_pause_touch_and_go(0.8, 0.6), bar_noise_m=TRACKED_BAR_NOISE_M,
                                    seed=seed, plates_hide_feet_above_m=0.03 if feet == "under_the_plates" else None)
                sim = simulate(scenario)
                lost = [(rep.top_time + 0.03, rep.top_time + 0.4) for rep in sim.reps]
                unmeasured = _feet_hidden_windows(sim) if feet == "under_the_plates" else None
                sim, features, _, _ = _analyse(scenario, mutate=_tracker_coasting(lost), legs_unmeasured=unmeasured)
                assert len(features) == len(sim.reps)
                cued += sum("deadlift_lockout" in rep for rep in _cued(features))
        assert cued == 0

    def test_plates_hiding_the_feet_leave_the_hips_and_knees_to_date_a_lost_top(self):
        """The pipeline marks the legs unmeasured when the plates hide the feet, but
        the hips and knees are seen: they check the bar's fit, which overshoots a
        quick pull (alone, +0.45 to +0.59 s on 0.6 s pulls lost from 0.3 s before)."""
        errors = []
        for body in BODIES:
            for seed in range(3):
                scenario = Scenario(athlete=body, reps=[RepScript(pull_s=0.6)] * 3, bar_noise_m=TRACKED_BAR_NOISE_M,
                                    seed=seed, plates_hide_feet_above_m=0.03)
                sim = simulate(scenario)
                lost = [(rep.top_time - 0.3, rep.top_time + 0.8) for rep in sim.reps]
                sim, features, _, _ = _analyse(scenario, mutate=_tracker_coasting(lost),
                                               legs_unmeasured=_feet_hidden_windows(sim))
                assert len(features) == len(sim.reps)
                errors += [measured.top_time - expected.top_time for measured, expected in zip(features, sim.reps)]
        assert max(map(abs, errors)) <= GAP_TOP_MAX_ERROR_S, errors

    def test_a_failed_rep_with_the_knees_and_bar_hidden_is_not_a_rep(self):
        """With the knees hidden, standing is read on running medians of the trunk
        and the legs' length, within 3 cm of standing's: one noisy frame within the
        over-extended lockout's 6 cm (~40 deg of knee bend) made a failed rep 10-12 cm
        short a rep (16 of 60)."""
        counted = 0
        for body_index, body in enumerate(BODIES):
            full_rise_m = _full_rise_m(body)
            for pull_s, half_s in FAILED_HIDDEN_HALF_S.items():
                for short_m in FAILED_SHORT_OF_TOP_M:
                    failed = RepScript(pull_s=pull_s, fail_rise_m=full_rise_m - short_m)
                    for seed in range(3):
                        scenario = Scenario(athlete=body, reps=[RepScript(), failed, RepScript()],
                                            bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed)
                        peak_t = simulate(scenario).reps[1].liftoff_time + pull_s
                        hidden = [(peak_t - half_s, peak_t + half_s)]
                        noise = _correlated_noise(PROXY_KEYPOINT_NOISE_M, PLATFORM_NOISE_RHO,
                                                  FAILED_HIDDEN_NOISE_SEED + 100 * body_index + 10 * seed
                                                  + int(100 * pull_s) + int(1000 * short_m))
                        mutate = _then(_then(_tracker_coasting(hidden), _hidden_keypoints(hidden, KNEES)), noise)
                        _, features, _, _ = _analyse(scenario, mutate=mutate)
                        counted += len(features) - 2
        assert counted == 0

    @pytest.mark.parametrize("lean_back_deg", OVER_EXTENDED_LEANS_DEG)
    @pytest.mark.parametrize("noise_m", [0.0, PROXY_KEYPOINT_NOISE_M])
    def test_an_over_extended_lockout_lost_from_view_is_counted_and_judged(self, lean_back_deg: float, noise_m: float):
        """Leaned back past the over-extended top, the bar hangs well short of the
        expected top: lost from view before it, the lifter seen leaned back on
        straight legs while it was unseen makes it a top (0 of 30 counted),
        noise-free and at 1.5 cm AR(0.8)."""
        for body_index, body in enumerate(BODIES):
            for seed in range(2):
                scenario = Scenario(athlete=body, reps=[RepScript(pull_s=1.2, lean_back_deg=lean_back_deg)] * 3,
                                    bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed)
                lost = [(rep.top_time - 0.2, rep.top_time + 0.8) for rep in simulate(scenario).reps]
                mutate = _tracker_coasting(lost)
                if noise_m:
                    mutate = _then(mutate, _correlated_noise(noise_m, PLATFORM_NOISE_RHO,
                                                             STANDING_RECALL_NOISE_SEED + 10 * body_index + seed))
                sim, features, _, _ = _analyse(scenario, mutate=mutate)
                assert len(features) == len(sim.reps)
                assert all("deadlift_lean_back" in rep for rep in _cued(features))

    def test_knees_seen_late_in_the_gap_do_not_veto_the_fit(self):
        """The knees hidden over the gap's first 55 %: the stall check reads the
        joints only within 0.15 s of the fitted top (the nearest frame, the climb's,
        vetoed it and dated the top +0.3 s late)."""
        errors = []
        for body in BODIES:
            for seed in range(3):
                scenario = Scenario(athlete=body, reps=[RepScript(pull_s=1.2)] * 3, bar_noise_m=TRACKED_BAR_NOISE_M,
                                    seed=seed)
                lost = [(rep.top_time - 0.2, rep.top_time + 0.8) for rep in simulate(scenario).reps]
                hidden = [(start, start + PARTLY_HIDDEN_SHARE * (end - start)) for start, end in lost]
                sim, features, _, _ = _analyse(scenario, mutate=_then(_tracker_coasting(lost),
                                                                       _hidden_keypoints(hidden, KNEES)))
                assert len(features) == len(sim.reps)
                errors += [measured.top_time - expected.top_time for measured, expected in zip(features, sim.reps)]
        assert max(map(abs, errors)) <= GAP_TOP_MAX_ERROR_S, errors

    @pytest.mark.parametrize("noise_m", [0.0, PROXY_KEYPOINT_NOISE_M])
    def test_knees_and_bar_hidden_across_the_top_lose_no_rep(self, noise_m: float):
        """Seen standing is the lifter's top when the bar is lost across it; with the
        knees hidden too, upright within the standing shortening of the legs, on
        running medians (each rep had been a failed rep on a tall lifter), noise-free
        and at 1.5 cm AR(0.8)."""
        for body_index, body in enumerate(BODIES):
            for seed in range(3):
                scenario = Scenario(athlete=body, reps=[RepScript(pull_s=1.2)] * 3, bar_noise_m=TRACKED_BAR_NOISE_M,
                                    seed=seed)
                lost = [(rep.top_time - 0.3, rep.top_time + 0.8) for rep in simulate(scenario).reps]
                mutate = _then(_tracker_coasting(lost), _hidden_keypoints(lost, KNEES))
                if noise_m:
                    mutate = _then(mutate, _correlated_noise(noise_m, PLATFORM_NOISE_RHO,
                                                             STANDING_RECALL_NOISE_SEED + 10 * body_index + seed))
                sim, features, _, _ = _analyse(scenario, mutate=mutate)
                assert len(features) == len(sim.reps)

    def test_short_gaps_through_a_hold_do_not_resume_it(self):
        """Without a standing reference, the bar dropped for 3 frames every 0.25 s
        through a 1.5 s hold at 2.5 cm AR(0.8): a resume on the joints needs a gap of
        UNSEEN_RESUME_MIN_S (on each short gap's few frames of noise, holds resumed
        and read up to 1.4 s late)."""
        script = RepScript(pull_s=1.2, top_hold_s=1.5)
        errors = []
        for body_index, body in enumerate(BODIES):
            for seed in range(3):
                scenario = Scenario(athlete=body, reps=[script] * 3, bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed,
                                    approach_s=0.5, stance_s=0.5)
                drops = []
                for rep in simulate(scenario).reps:
                    start = rep.top_time + 0.1
                    while start + SHORT_GAP_S < rep.top_time + script.top_hold_s:
                        drops.append((start, start + SHORT_GAP_S))
                        start += SHORT_GAP_EVERY_S
                noise = _correlated_noise(WORST_KEYPOINT_NOISE_M, PLATFORM_NOISE_RHO,
                                          SHORT_GAPS_NOISE_SEED + 10 * body_index + seed)
                sim, features, _, _ = _analyse(scenario, mutate=_then(_bar_dropped(drops), noise))
                assert len(features) == len(sim.reps)
                errors += [measured.top_time - expected.top_time for measured, expected in zip(features, sim.reps)]
        assert max(map(abs, errors)) <= 2.0 * GAP_TOP_MAX_ERROR_S, errors

    @pytest.mark.parametrize("stall_fraction", [0.93, 0.95, 0.97])
    def test_a_grind_lost_through_its_finish_and_hold_is_dated_and_judged_at_the_lockout(self, stall_fraction: float):
        """The stall is seen and held (PULL -> TOP), the finish and the hold are not:
        the hips and knees extending out of it while the bar is unseen resume the
        pull (it stayed at the stall, its top dated about 1 s early and D6 cued on
        every rep that locked out)."""
        errors = []
        cued = 0
        for body in BODIES[:3]:
            for seed in range(3):
                grind = RepScript(pull_s=1.6, stall_fraction=stall_fraction, stall_s=0.6, finish_s=0.4)
                scenario = Scenario(athlete=body, reps=[RepScript(pull_s=1.6), grind, RepScript(pull_s=1.6)],
                                    bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed)
                top = simulate(scenario).reps[1].top_time
                sim, features, _, _ = _analyse(scenario, mutate=_tracker_coasting([(top - 0.2, top + 0.8)]))
                assert len(features) == len(sim.reps)
                errors.append(features[1].top_time - top)
                cued += "deadlift_lockout" in _cued(features)[1]
        assert cued == 0
        assert max(map(abs, errors)) <= GAP_TOP_MAX_ERROR_S, errors

    def test_a_no_pause_top_lost_from_view_is_judged_either_side_of_it(self):
        """Seen again well into the lowering, the bar's last frame at its peak is the
        lowering's start: the window still keeps its frames after the top (judged
        on the climb side alone, the lockout read 10 deg short)."""
        deficits = []
        for body in BODIES[:3]:
            scenario = Scenario(athlete=body, reps=_no_pause_touch_and_go(1.2, 0.5), bar_noise_m=TRACKED_BAR_NOISE_M)
            lost = simulate(scenario).reps[1:4]
            mutate = _tracker_coasting([(rep.top_time - 0.2, rep.top_time + 0.3) for rep in lost])
            sim, features, _, _ = _analyse(scenario, mutate=mutate)
            assert len(features) == len(sim.reps)
            deficits += [max(f.hip_extension_deficit_deg, f.knee_extension_deficit_deg) for f in features[1:4]]
        assert max(deficits) <= LOST_TOP_MAX_DEFICIT_DEG, deficits

    def test_a_hold_lost_from_view_under_keypoint_noise_reads_no_velocity_loss(self):
        """1 s losses from 0.2 s before every top, at 1.5 cm of correlated keypoint
        noise: the joints alone wander through a hold, the bar's last rise does not."""
        errors = []
        slowed = 0
        for body_index, body in enumerate(BODIES[:3]):
            for seed in range(3):
                scenario = Scenario(athlete=body, reps=[RepScript(pull_s=1.5)] * 3, bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed)
                lost = _tracker_coasting([(rep.top_time - 0.2, rep.top_time + 0.8) for rep in simulate(scenario).reps])
                noise = _correlated_noise(PROXY_KEYPOINT_NOISE_M, PLATFORM_NOISE_RHO, 1200 + 10 * body_index + seed)

                def lose_then_noise(index: int, frame: SimFrame) -> tuple[np.ndarray, np.ndarray, BarState3D | None]:
                    points, confidences, bar = lost(index, frame)
                    return noise(index, frame._replace(points=points, confidences=confidences, bar=bar))

                sim, features, _, _ = _analyse(scenario, mutate=lose_then_noise)
                assert len(features) == len(sim.reps)
                errors += [measured.top_time - expected.top_time for measured, expected in zip(features, sim.reps)]
                slowed += any("deadlift_velocity_loss" in judged for judged in _judge(features))
        assert max(map(abs, errors)) <= GAP_TOP_MAX_ERROR_S, errors
        assert slowed == 0

    def test_a_bar_tracked_at_half_the_frame_rate_is_timed_at_its_tops(self):
        """One unseen frame after the highest is a skipped detection, not a top lost
        from view; and a 0.6 s hold is still read at half the frame rate."""
        def every_other(index: int, frame: SimFrame) -> tuple[np.ndarray, np.ndarray, BarState3D | None]:
            return frame.points, frame.confidences, frame.bar if frame.frame_index % 2 == 0 else None

        errors = []
        for seed in range(4):
            scenario = Scenario(reps=[RepScript(pull_s=0.6)] * 3, bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed)
            sim, features, _, _ = _analyse(scenario, mutate=every_other)
            assert len(features) == len(sim.reps)
            errors += [measured.top_time - expected.top_time for measured, expected in zip(features, sim.reps)]
        assert max(map(abs, errors)) <= MAX_EVENT_ERROR_S, errors

    def test_a_bar_tracked_at_half_the_frame_rate_stays_the_bar(self):
        def every_other(index: int, frame: SimFrame) -> tuple[np.ndarray, np.ndarray, BarState3D | None]:
            return frame.points, frame.confidences, frame.bar if index % 2 == 0 else None

        sim, features, _, _ = _analyse(Scenario(reps=[RepScript(floor_hold_s=0.0)] * 2 + [RepScript()]), mutate=every_other)
        assert len(features) == len(sim.reps)
        assert {f.bar_source for f in features} == {BAR_SOURCE_BAR}


class TestTempo:
    """Events land within 100 ms from a 0.45 s pull to a 5 s grind, on a tracked
    bar's noise: fitted as the change point of the bar's departure from (or
    arrival at) a level, not where it crosses a fixed band."""

    @pytest.mark.parametrize("pull_s", [0.45, 0.8, 3.0, 5.0])
    @pytest.mark.parametrize("seed", range(3))
    def test_events_stay_on_time_from_a_fast_pull_to_a_slow_grind(self, pull_s: float, seed: int):
        scenario = Scenario(
            reps=[RepScript(pull_s=pull_s, lower_s=max(0.4, pull_s * 0.6))] * 2,
            bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed,
        )
        sim, features, _, _ = _analyse(scenario)
        assert len(features) == len(sim.reps)
        for expected, measured in zip(sim.reps, features):
            assert measured.liftoff_time == pytest.approx(expected.liftoff_time, abs=MAX_EVENT_ERROR_S)
            assert measured.knee_pass_time == pytest.approx(expected.knee_pass_time, abs=MAX_EVENT_ERROR_S)
            assert measured.top_time == pytest.approx(expected.top_time, abs=MAX_EVENT_ERROR_S)
            assert measured.floor_time == pytest.approx(expected.floor_time, abs=MAX_EVENT_ERROR_S)

    def test_a_slow_pull_under_platform_noise_meets_the_event_gate(self):
        errors = []
        for seed in range(6):
            scenario = Scenario(reps=[RepScript(pull_s=5.0)] * 3, bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed)
            noise = _correlated_noise(PLATFORM_KEYPOINT_NOISE_M, PLATFORM_NOISE_RHO, 70 + seed)
            sim, features, _, _ = _analyse(scenario, mutate=noise)
            assert len(features) == len(sim.reps)
            errors += [measured.top_time - expected.top_time for measured, expected in zip(features, sim.reps)]
        assert max(abs(error) for error in errors) <= MAX_EVENT_ERROR_S

    def test_a_grind_that_finishes_quickly_is_not_dated_late(self):
        """Its climb's upper half is a few frames against the hold: the fit keeps
        0.2 s of the hold, and reads a frame above the level as arrived."""
        errors = []
        for seed in range(8):
            reps = [RepScript(pull_s=1.6, stall_fraction=0.95, stall_s=0.8, finish_s=0.3)] * 3
            sim, features, _, _ = _analyse(Scenario(reps=reps, bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed))
            assert len(features) == len(sim.reps)
            errors += [abs(measured.top_time - expected.top_time) for measured, expected in zip(features, sim.reps)]
        assert max(errors) <= QUICK_FINISH_MAX_ERROR_S

    def test_a_grind_that_finishes_slowly_is_timed_at_its_top(self):
        """The top is fitted on the last climb out of the stall, over its upper half."""
        errors = []
        for seed in range(4):
            reps = [RepScript(pull_s=2.0, stall_fraction=0.95, stall_s=0.8, finish_s=1.0)] * 3
            sim, features, _, _ = _analyse(Scenario(reps=reps, bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed))
            assert len(features) == len(sim.reps)
            errors += [measured.top_time - expected.top_time for measured, expected in zip(features, sim.reps)]
        assert float(np.median(np.abs(errors))) <= SLOW_FINISH_MAX_MEDIAN_ERROR_S


class TestModelFreeKinematics:
    """Poses scripted by trunk angle, not built by the setup model: the analyser's
    measurements against the kinematic truth of the emitted poses, and D2's
    verdicts on pulls the model did not make."""

    @pytest.mark.parametrize(
        "athlete",
        [
            SimAthlete(),
            SimAthlete(torso_m=0.58, upper_arm_m=0.28, forearm_m=0.27),
            SimAthlete(femur_m=0.50, tibia_m=0.45),
            SimAthlete(tibia_m=0.38, femur_m=0.40, torso_m=0.47, upper_arm_m=0.27, forearm_m=0.26),
        ],
        ids=["average", "long_torso_short_arms", "long_femurs", "short"],
    )
    @pytest.mark.parametrize(
        ("setup_deg", "knee_pass_deg"),
        [(65.0, 50.0), (60.0, 60.0), (45.0, 55.0)],
        ids=["chest_rises", "back_angle_held", "hips_first"],
    )
    def test_trunk_change_rise_ratio_and_hip_height_match_the_poses(
        self, athlete: SimAthlete, setup_deg: float, knee_pass_deg: float,
    ):
        script = RepScript(setup_trunk_deg=setup_deg, knee_pass_trunk_deg=knee_pass_deg)
        sim, features, _, _ = _analyse(Scenario(athlete=athlete, reps=[script] * 2))
        assert len(features) == len(sim.reps)
        for truth, rep in zip(sim.reps, features):
            measured_change = rep.trunk_change_liftoff_knee_deg + rep.trunk_change_predicted_deg
            true_change = truth.knee_pass_trunk_deg - truth.liftoff_trunk_deg
            assert measured_change == pytest.approx(true_change, abs=TRUNK_CHANGE_TOLERANCE_DEG)
            assert rep.hip_shoulder_rise_ratio == pytest.approx(truth.hip_rise_m / truth.shoulder_rise_m, abs=RISE_RATIO_TOLERANCE)
            assert rep.setup_hip_height_cm == pytest.approx(truth.setup_hip_height_m * 100.0, abs=HIP_HEIGHT_TOLERANCE_CM)

    @pytest.mark.parametrize("pull_s", [0.6, 1.2, 3.0])
    @pytest.mark.parametrize(
        "athlete",
        [SimAthlete(), SimAthlete(tibia_m=0.38, femur_m=0.40, torso_m=0.47, upper_arm_m=0.27, forearm_m=0.26)],
        ids=["average", "short"],
    )
    @pytest.mark.parametrize(
        ("setup_deg", "knee_pass_deg"),
        [(65.0, 50.0), (60.0, 60.0), (55.0, 65.0), (45.0, 55.0)],
        ids=["chest_rises", "back_angle_held", "tips_forward", "hips_first"],
    )
    def test_rise_ratio_matches_the_poses_at_any_tempo(
        self, athlete: SimAthlete, setup_deg: float, knee_pass_deg: float, pull_s: float,
    ):
        """A fast pull accelerates through the frames before the knee pass: the
        ratio comes from a curve through them, not a line."""
        script = RepScript(setup_trunk_deg=setup_deg, knee_pass_trunk_deg=knee_pass_deg, pull_s=pull_s)
        sim, features, _, _ = _analyse(Scenario(athlete=athlete, reps=[script] * 2))
        assert len(features) == len(sim.reps)
        for truth, rep in zip(sim.reps, features):
            assert rep.hip_shoulder_rise_ratio == pytest.approx(truth.hip_rise_m / truth.shoulder_rise_m, abs=RISE_RATIO_TOLERANCE)

    def test_a_hips_first_pull_is_cued_once_the_set_confirms_it(self):
        """Hips out-rising the chest by ~1.24: within one rep's noise of a held back
        angle, so the first rep waits and the set's second confirms it."""
        script = RepScript(setup_trunk_deg=45.0, knee_pass_trunk_deg=55.0)
        sim, features, _, _ = _analyse(Scenario(reps=[script] * 3))
        assert all(truth.hip_rise_m > truth.shoulder_rise_m for truth in sim.reps)
        assert ["deadlift_hips_shoot" in cued for cued in _cued(features)] == [False, True, True]

    def test_a_held_back_angle_is_no_hips_shoot(self):
        """The model wants the chest to rise off the floor; a lifter holding the
        back angle reads 6-12 deg against it, but the hips never out-rose the
        shoulders, so there is no fault."""
        script = RepScript(setup_trunk_deg=60.0, knee_pass_trunk_deg=60.0)
        sim, features, _, _ = _analyse(Scenario(reps=[script] * 3))
        assert all(truth.hip_rise_m <= truth.shoulder_rise_m for truth in sim.reps)
        assert all("deadlift_hips_shoot" not in verdict for verdict in _judge(features))

    @pytest.mark.parametrize("seed", range(6))
    def test_a_held_back_angle_is_not_cued_at_platform_noise(self, seed: int):
        script = RepScript(setup_trunk_deg=60.0, knee_pass_trunk_deg=60.0)
        scenario = Scenario(reps=[script] * 3, keypoint_noise_m=PLATFORM_KEYPOINT_NOISE_M,
                            bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed)
        _, features, _, _ = _analyse(scenario)
        assert all("deadlift_hips_shoot" not in cued for cued in _cued(features))

    def test_a_held_back_angle_is_rarely_cued_at_correlated_noise(self):
        """Correlated noise scatters one rep's rise ratio by ~0.1 around the held
        angle's 1.0: the set-level check keeps false cues under the VALIDATION.md
        gate of 1 per 10 reps."""
        script = RepScript(setup_trunk_deg=60.0, knee_pass_trunk_deg=60.0)
        cued = reps = 0
        for seed in range(8):
            scenario = Scenario(reps=[script] * 3, bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed)
            _, features, _, _ = _analyse(scenario, mutate=_correlated_noise(PLATFORM_KEYPOINT_NOISE_M, 0.8, seed))
            reps += len(features)
            cued += sum("deadlift_hips_shoot" in rep_cues for rep_cues in _cued(features))
        assert reps == 24
        assert cued <= reps // 10

    def test_a_hips_first_pull_is_cued_on_most_reps_at_platform_noise(self):
        script = RepScript(setup_trunk_deg=45.0, knee_pass_trunk_deg=55.0)
        cued = reps = 0
        for seed in range(4):
            scenario = Scenario(reps=[script] * 3, keypoint_noise_m=PLATFORM_KEYPOINT_NOISE_M,
                                bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed)
            _, features, _, _ = _analyse(scenario)
            reps += len(features)
            cued += sum("deadlift_hips_shoot" in rep_cues for rep_cues in _cued(features))
        assert cued >= reps // 2

    @pytest.mark.parametrize(
        ("first", "then"),
        [((60.0, 60.0), (45.0, 55.0)), ((40.0, 55.0), (65.0, 50.0))],
        ids=["starts_shooting_the_hips", "fixes_the_hips"],
    )
    def test_a_pull_that_changes_mid_set_is_measured_rep_by_rep(
        self, first: tuple[float, float], then: tuple[float, float],
    ):
        reps = [RepScript(setup_trunk_deg=first[0], knee_pass_trunk_deg=first[1])] * 2
        reps += [RepScript(setup_trunk_deg=then[0], knee_pass_trunk_deg=then[1])] * 3
        sim, features, _, _ = _analyse(Scenario(reps=reps))
        assert len(features) == len(sim.reps)
        for truth, rep in zip(sim.reps, features):
            assert rep.hip_shoulder_rise_ratio == pytest.approx(truth.hip_rise_m / truth.shoulder_rise_m, abs=RISE_RATIO_TOLERANCE)

    def test_the_rep_after_the_fix_is_not_cued(self):
        """The demo loop: hips first, the cue, then a pull with the chest rising."""
        reps = [RepScript(setup_trunk_deg=40.0, knee_pass_trunk_deg=55.0)] * 2
        reps += [RepScript(setup_trunk_deg=65.0, knee_pass_trunk_deg=50.0)] * 3
        _, features, _, _ = _analyse(Scenario(reps=reps))
        assert ["deadlift_hips_shoot" in cued for cued in _cued(features)] == [True, True, False, False, False]

    def test_a_pull_whose_chest_rises_is_fault_free(self):
        script = RepScript(setup_trunk_deg=65.0, knee_pass_trunk_deg=50.0)
        _, features, _, _ = _analyse(Scenario(reps=[script] * 2))
        assert all("deadlift_hips_shoot" not in verdict for verdict in _judge(features))

    def test_hips_set_lower_than_the_shins_allow_read_as_low(self):
        script = RepScript(setup_trunk_deg=45.0, knee_pass_trunk_deg=50.0)
        _, features, _, _ = _analyse(Scenario(athlete=SimAthlete(femur_m=0.50, tibia_m=0.45), reps=[script] * 2))
        for rep in features:
            assert rep.setup_hip_height_cm < rep.setup_hip_band_low_cm


class TestSetupModelGeometry:
    """The setup model's solves, checked in the simulator's forward kinematics: a
    pose built from the model's trunk angle must meet the constraints the model
    claims (shins on the bar, a real knee bend), whatever the shin contact."""

    @pytest.mark.parametrize("shin_bar_m", [0.03, 0.05, 0.07])
    def test_a_model_setup_puts_the_shins_on_the_bar_with_bent_knees(self, shin_bar_m: float):
        sim = simulate(Scenario(shin_bar_m=shin_bar_m, reps=[RepScript()]))
        setup = next(frame for frame in sim.frames if frame.timestamp >= sim.reps[0].liftoff_time - 1e-6)
        points = setup.points
        ankle = (points[CK.LEFT_ANKLE] + points[CK.RIGHT_ANKLE]) / 2.0
        knee = (points[CK.LEFT_KNEE] + points[CK.RIGHT_KNEE]) / 2.0
        hip = (points[CK.LEFT_HIP] + points[CK.RIGHT_HIP]) / 2.0
        # The sagittal plane is the world's (Y, Z) here: no tilt, X is lateral.
        shin = (knee - ankle)[1:]
        thigh = (hip - knee)[1:]
        to_bar = (setup.bar.centre - ankle)[1:]
        shin_to_bar_m = abs(shin[0] * to_bar[1] - shin[1] * to_bar[0]) / float(np.linalg.norm(shin))
        knee_flexion_deg = math.degrees(math.acos(
            float(np.dot(shin, thigh)) / (float(np.linalg.norm(shin)) * float(np.linalg.norm(thigh))),
        ))
        assert shin_to_bar_m == pytest.approx(shin_bar_m, abs=0.002)
        assert MIN_SETUP_KNEE_FLEXION_DEG <= knee_flexion_deg <= MAX_SETUP_KNEE_FLEXION_DEG


class TestTouchAndGoUnderNoise:
    """No dead stop between touch-and-go reps: a check that one noisy frame could
    defeat would swallow every rep after it."""

    @pytest.mark.parametrize("seed", range(4))
    def test_noisy_bar_counts_every_touch_and_go_rep(self, seed: int):
        reps = [RepScript(floor_hold_s=0.0, pull_s=1.5, lower_s=0.8)] * 4 + [RepScript(pull_s=1.5, lower_s=0.8)]
        sim, features, _, _ = _analyse(Scenario(reps=reps, bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed))
        assert len(features) == len(sim.reps)
        assert [f.touch_and_go for f in features] == [False, True, True, True, True]

    @pytest.mark.parametrize("seed", range(4))
    def test_a_very_noisy_bar_still_counts_every_rep(self, seed: int):
        """At 5 mm a low point can read as a dead stop: the next rep is then a quick
        re-pull rather than a touch-and-go, but no rep is lost."""
        reps = [RepScript(floor_hold_s=0.0, pull_s=1.5, lower_s=0.8)] * 4 + [RepScript(pull_s=1.5, lower_s=0.8)]
        sim, features, _, _ = _analyse(Scenario(reps=reps, bar_noise_m=0.005, seed=seed))
        assert len(features) == len(sim.reps)

    def test_ten_touch_and_go_reps_at_platform_noise(self):
        reps = [RepScript(floor_hold_s=0.0, pull_s=1.3, lower_s=0.9)] * 9 + [RepScript(pull_s=1.3, lower_s=0.9)]
        scenario = Scenario(reps=reps, bar_noise_m=TRACKED_BAR_NOISE_M, seed=5)
        sim, features, _, _ = _analyse(scenario, mutate=_correlated_noise(PLATFORM_KEYPOINT_NOISE_M, 0.8, 5))
        assert len(features) == len(sim.reps)

    @pytest.mark.parametrize("keypoint_noise_m", [0.01, 0.02])
    def test_noisy_wrist_proxy_counts_every_touch_and_go_rep(self, keypoint_noise_m: float):
        reps = [RepScript(floor_hold_s=0.0)] * 4 + [RepScript()]
        sim, features, _, _ = _analyse(Scenario(track_bar=False, reps=reps, keypoint_noise_m=keypoint_noise_m))
        assert len(features) == len(sim.reps)

    @pytest.mark.parametrize("pull_s", [0.5, 0.6])
    @pytest.mark.parametrize("seed", range(4))
    def test_fast_touch_and_go_on_the_wrist_proxy_under_correlated_noise(self, pull_s: float, seed: int):
        """The wrists leave the knees' height within a few frames of a fast low
        point, and the noise widens the velocity window into the descent."""
        reps = [RepScript(floor_hold_s=0.0, pull_s=pull_s, lower_s=0.4, top_hold_s=0.2)] * 5 + [RepScript(pull_s=pull_s, lower_s=0.4)]
        scenario = Scenario(track_bar=False, reps=reps, seed=seed)
        sim, features, _, _ = _analyse(scenario, mutate=_correlated_noise(PLATFORM_KEYPOINT_NOISE_M, 0.8, seed))
        assert len(features) == len(sim.reps)

    @pytest.mark.parametrize("seed", range(4))
    def test_fast_touch_and_go_on_the_wrist_proxy_under_independent_noise(self, seed: int):
        reps = [RepScript(floor_hold_s=0.0, pull_s=0.5, lower_s=0.4, top_hold_s=0.2)] * 5 + [RepScript(pull_s=0.5, lower_s=0.4)]
        sim, features, _, _ = _analyse(Scenario(track_bar=False, reps=reps, keypoint_noise_m=0.015, seed=seed))
        assert len(features) == len(sim.reps)

    @pytest.mark.parametrize("track_bar", [True, False], ids=["tracked", "proxy"])
    def test_a_fast_top_that_never_held_reads_its_lockout(self, track_bar: bool):
        """Noise-free. On a 0.6 s pull the knees still bend 0.1 s either side of
        the top: a quicker pull is judged over a proportionally narrower window."""
        scripts = [RepScript(top_hold_s=0.0, floor_hold_s=0.0, pull_s=0.6, lower_s=0.5)] * 4 + [
            RepScript(top_hold_s=0.0, pull_s=0.6, lower_s=0.5)]
        for body in BODIES[:3]:
            sim, features, _, _ = _analyse(Scenario(athlete=body, reps=scripts, track_bar=track_bar))
            assert len(features) == len(sim.reps)
            deficits = [max(f.hip_extension_deficit_deg, f.knee_extension_deficit_deg) for f in features]
            assert max(deficits) <= FAST_TOP_MAX_DEFICIT_DEG, deficits


class TestWristProxyAtPlatformNoise:
    """The wrist proxy has no bar axis: its left-right is locked per set, and its
    tilt (the hands' noise carried out to the hubs) is cued only when severe."""

    @pytest.mark.parametrize("seed", range(4))
    def test_clean_reps_cue_nothing_under_correlated_noise(self, seed: int):
        athletes = [SimAthlete(), SimAthlete(torso_m=0.58, upper_arm_m=0.28, forearm_m=0.27),
                    SimAthlete(femur_m=0.50, tibia_m=0.45), SimAthlete(tibia_m=0.38, femur_m=0.40, torso_m=0.47)]
        scenario = Scenario(athlete=athletes[seed], track_bar=False, seed=seed, approach_s=3.0)
        sim, features, _, _ = _analyse(scenario, mutate=_correlated_noise(PLATFORM_KEYPOINT_NOISE_M, 0.8, seed + 100))
        assert len(features) == len(sim.reps)
        assert all("deadlift_hip_shift" not in cued for cued in _cued(features))

    def test_a_clean_hip_line_rarely_reads_a_shift(self):
        shifted = reps = 0
        for seed in range(8):
            scenario = Scenario(track_bar=False, seed=seed)
            _, features, _, _ = _analyse(scenario, mutate=_correlated_noise(PLATFORM_KEYPOINT_NOISE_M, 0.8, seed))
            reps += len(features)
            shifted += sum(abs(f.hip_shift_ratio) >= FAULTS.deadlift_hip_shift.moderate for f in features)
        assert reps == 24
        assert shifted <= reps // 10

    def test_the_bar_speed_is_not_measured(self):
        """Its top is the joints', its rise and liftoff the wrists': no speed for the
        recap, the spoken summary or the diagnosis (D10 is off here too)."""
        sim, features, _, _ = _analyse(Scenario(track_bar=False))
        assert len(features) == len(sim.reps)
        assert all(math.isnan(f.concentric_velocity_mps) for f in features)

    def test_a_top_with_no_pause_draws_no_more_lockout_cues_than_on_the_tracked_bar(self):
        """A touch-and-go top never holds: it is judged around its peak, which on
        the wrist proxy is the vertex of the wrists' height, not their highest
        noisy frame (up to ~0.15 s early, the knees still bent)."""
        cued = {True: 0, False: 0}
        severe = reps = 0
        for track_bar in (True, False):
            for body_index, body in enumerate(BODIES):
                for seed in range(4):
                    scripts = [RepScript(top_hold_s=0.0, floor_hold_s=0.0)] * 4 + [RepScript(top_hold_s=0.0)]
                    scenario = Scenario(athlete=body, reps=scripts, track_bar=track_bar,
                                        bar_noise_m=TRACKED_BAR_NOISE_M if track_bar else 0.0, seed=seed)
                    noise = _correlated_noise(PLATFORM_KEYPOINT_NOISE_M, PLATFORM_NOISE_RHO, 5000 + 100 * body_index + seed)
                    sim, features, _, _ = _analyse(scenario, mutate=noise)
                    assert len(features) == len(sim.reps)
                    reps += len(features) if track_bar else 0
                    for faults in _faults(features):
                        lockout = [f for f in faults if f.fault_type == "deadlift_lockout"
                                   and TIER_RANK[f.severity.value] >= TIER_RANK[f.details["min_tier"]]]
                        cued[track_bar] += bool(lockout)
                        severe += any(f.severity.value == "severe" for f in lockout) and not track_bar
        assert reps == 100
        assert cued[False] <= cued[True] + reps // 20
        assert severe == 0

    @pytest.mark.parametrize(("pull_s", "lower_s"), FAST_LOWERING_TEMPOS)
    def test_a_lowering_faster_than_the_pull_reads_the_lockout_the_tracked_bar_reads(self, pull_s: float, lower_s: float):
        """Noise-free. Let down faster than it went up, the peak is lopsided: a
        symmetric parabola's vertex lands in the end of the rise, the knees still
        bending. Each side is fitted with its own curvature."""
        readings = {}
        for track_bar in (True, False):
            scripts = [RepScript(top_hold_s=0.0, floor_hold_s=0.0, pull_s=pull_s, lower_s=lower_s)] * 4 + [
                RepScript(top_hold_s=0.0, pull_s=pull_s, lower_s=lower_s)]
            sim, features, _, _ = _analyse(Scenario(reps=scripts, track_bar=track_bar))
            assert len(features) == len(sim.reps)
            readings[track_bar] = float(np.median([max(f.hip_extension_deficit_deg, f.knee_extension_deficit_deg) for f in features]))
        assert readings[False] <= readings[True] + NOISE_FREE_PROXY_LOCKOUT_TOLERANCE_DEG, readings

    def test_a_fast_lowering_draws_no_more_lockout_cues_than_on_the_tracked_bar(self):
        cued = {True: 0, False: 0}
        severe = reps = 0
        for pull_s, lower_s in FAST_LOWERING_TEMPOS[:2]:
            for track_bar in (True, False):
                for body_index, body in enumerate(BODIES):
                    for seed in range(4):
                        scripts = [RepScript(top_hold_s=0.0, floor_hold_s=0.0, pull_s=pull_s, lower_s=lower_s)] * 4 + [
                            RepScript(top_hold_s=0.0, pull_s=pull_s, lower_s=lower_s)]
                        scenario = Scenario(athlete=body, reps=scripts, track_bar=track_bar,
                                            bar_noise_m=TRACKED_BAR_NOISE_M if track_bar else 0.0, seed=seed)
                        noise = _correlated_noise(PLATFORM_KEYPOINT_NOISE_M, PLATFORM_NOISE_RHO, 6000 + 100 * body_index + seed)
                        sim, features, _, _ = _analyse(scenario, mutate=noise)
                        assert len(features) == len(sim.reps)
                        reps += len(features) if track_bar else 0
                        for faults in _faults(features):
                            lockout = [f for f in faults if f.fault_type == "deadlift_lockout"
                                       and TIER_RANK[f.severity.value] >= TIER_RANK[f.details["min_tier"]]]
                            cued[track_bar] += bool(lockout)
                            severe += any(f.severity.value == "severe" for f in lockout) and not track_bar
        assert reps == 200
        assert cued[False] <= cued[True] + reps // 20, cued
        assert severe == 0

    def test_the_top_is_the_hips_and_knees_reaching_the_lockout(self):
        """The wrists' height wanders through the hold; the joints finishing their
        extension date the top."""
        errors = []
        for seed in range(6):
            scenario = Scenario(track_bar=False, seed=seed)
            noise = _correlated_noise(PROXY_KEYPOINT_NOISE_M, PLATFORM_NOISE_RHO, 500 + seed)
            sim, features, _, _ = _analyse(scenario, mutate=noise)
            assert len(features) == len(sim.reps)
            errors += [abs(measured.top_time - expected.top_time) for measured, expected in zip(features, sim.reps)]
        assert float(np.median(errors)) <= MAX_EVENT_ERROR_S
        assert max(errors) <= MAX_PROXY_TOP_ERROR_S

    @pytest.mark.parametrize("lockout_deficit_deg", [0.0, SOFT_LOCKOUT_DEG])
    def test_wrists_hidden_through_the_hold_still_judge_the_lockout(self, lockout_deficit_deg: float):
        """Hidden from 0.2 s before the top until 0.2 s into the lowering: the
        wrists are first seen leaving the top 0.8 s after it, and the 0.5 s before
        that held none of the top's frames (the lockout went unjudged). Judged on
        the few frames around the top instead of the hold's last 0.5 s, a soft
        lockout is cued a little less often than with the wrists seen."""
        judged = cued = reps = 0
        for body_index, body in enumerate(BODIES):
            for seed in range(3):
                scripts = [RepScript(pull_s=1.2, lockout_deficit_deg=lockout_deficit_deg)] * 3
                scenario = Scenario(athlete=body, reps=scripts, track_bar=False, seed=seed)
                hidden = [(rep.top_time - 0.2, rep.top_time + 0.8) for rep in simulate(scenario).reps]
                noise = _correlated_noise(PROXY_KEYPOINT_NOISE_M, PLATFORM_NOISE_RHO, 1300 + 10 * body_index + seed)
                sim, features, _, _ = _analyse(scenario, mutate=_wrists_hidden(hidden, then=noise))
                assert len(features) == len(sim.reps)
                reps += len(features)
                judged += sum(math.isfinite(f.hip_extension_deficit_deg) for f in features)
                cued += sum("deadlift_lockout" in rep_cues for rep_cues in _cued(features))
        assert judged == reps
        if lockout_deficit_deg:
            assert cued >= HIDDEN_SOFT_LOCKOUT_RECALL_RATIO * reps, f"D6 on {cued} of {reps}"
        else:
            assert cued <= FALSE_CUE_GATE_RATIO * reps, f"D6 on {cued} of {reps}"

    def test_wrists_hidden_across_a_top_with_no_pause_read_no_soft_lockout(self):
        """Hidden 0.2 s either side of each top: judged where the hips and knees were
        most extended while the wrists were unseen, not around the last height seen
        on the climb (at 1.5 cm, D6 on 21 of 75 reps)."""
        cued = reps = 0
        for body_index, body in enumerate(BODIES):
            for seed in range(3):
                scenario = Scenario(athlete=body, reps=_no_pause_touch_and_go(1.2, 1.0), track_bar=False, seed=seed)
                hidden = [(rep.top_time - 0.2, rep.top_time + 0.2) for rep in simulate(scenario).reps]
                noise = _correlated_noise(PROXY_KEYPOINT_NOISE_M, PLATFORM_NOISE_RHO, 1500 + 10 * body_index + seed)
                sim, features, _, _ = _analyse(scenario, mutate=_wrists_hidden(hidden, then=noise))
                assert len(features) == len(sim.reps)
                reps += len(features)
                cued += sum("deadlift_lockout" in rep_cues for rep_cues in _cued(features))
        assert cued <= FALSE_CUE_GATE_RATIO * reps, f"D6 on {cued} of {reps}"

    def test_wrists_hidden_as_they_arrive_date_the_top_by_the_joints(self):
        """Hidden from 0.1 s before the top to 0.3 s after: the hips and knees, still
        seen, reach the lockout in the gap (dated from where the wrists were seen
        again, 13-16 of 45 tops read beyond 0.3 s, up to 0.43 s late)."""
        errors = []
        for body_index, body in enumerate(BODIES):
            for seed in range(3):
                scenario = Scenario(athlete=body, reps=[RepScript(pull_s=1.2)] * 3, track_bar=False, seed=seed)
                hidden = [(rep.top_time - 0.1, rep.top_time + 0.3) for rep in simulate(scenario).reps]
                noise = _correlated_noise(PROXY_KEYPOINT_NOISE_M, PLATFORM_NOISE_RHO, 1400 + 10 * body_index + seed)
                sim, features, _, _ = _analyse(scenario, mutate=_wrists_hidden(hidden, then=noise))
                assert len(features) == len(sim.reps)
                errors += [measured.top_time - expected.top_time for measured, expected in zip(features, sim.reps)]
        late = sum(abs(error) > MAX_PROXY_TOP_ERROR_S for error in errors)
        assert late <= LATE_PROXY_TOP_RATIO * len(errors), errors

    def test_a_slow_pull_is_dated_early_by_the_joints_last_creep(self):
        errors = []
        for seed in range(6):
            scenario = Scenario(reps=[RepScript(pull_s=5.0)] * 3, track_bar=False, seed=seed)
            noise = _correlated_noise(PROXY_KEYPOINT_NOISE_M, PLATFORM_NOISE_RHO, 500 + seed)
            sim, features, _, _ = _analyse(scenario, mutate=noise)
            assert len(features) == len(sim.reps)
            errors += [abs(measured.top_time - expected.top_time) for measured, expected in zip(features, sim.reps)]
        assert float(np.median(errors)) <= MAX_PROXY_TOP_ERROR_S
        assert max(errors) <= MAX_PROXY_SLOW_TOP_ERROR_S


class TestHipShiftAlongTheLifter:
    """D8 is measured along the lifter's own left-right (hips and ankles, locked
    for the set), not the bar's axis: the hips travel ~45 cm forward in a pull,
    which a stance a few degrees off square to the bar would read as sideways."""

    @pytest.mark.parametrize("track_bar", [True, False], ids=["tracked", "proxy"])
    @pytest.mark.parametrize("degrees", [5.0, 8.0])
    def test_a_stance_off_square_to_the_bar_is_no_hip_shift(self, degrees: float, track_bar: bool):
        scenario = Scenario(track_bar=track_bar)
        sim = simulate(scenario)
        _, features, _, _ = _analyse(scenario, mutate=_turned(sim, degrees, LOWER_BODY))
        assert all(abs(f.hip_shift_ratio) < CLEAN_SHIFT_MAX_RATIO for f in features)

    def test_an_off_square_stance_rarely_cues_hip_shift_under_correlated_noise(self):
        """8 deg off square: the bar's axis leaks the hips' travel into sideways,
        the hip line does not, and D8 needs both to agree."""
        cued = reps = 0
        for seed in range(8):
            scenario = Scenario(seed=seed, bar_noise_m=TRACKED_BAR_NOISE_M)
            sim = simulate(scenario)
            noise = _correlated_noise(PLATFORM_KEYPOINT_NOISE_M, 0.8, seed)
            _, features, _, _ = _analyse(scenario, mutate=_turned(sim, 8.0, LOWER_BODY, noise))
            reps += len(features)
            cued += sum("deadlift_hip_shift" in rep_cues for rep_cues in _cued(features))
        assert reps == 24
        assert cued <= reps // 10

    def test_a_whole_lifter_turned_off_square_to_the_bar_is_no_hip_shift(self):
        scenario = Scenario()
        sim = simulate(scenario)
        _, features, _, _ = _analyse(scenario, mutate=_turned(sim, 10.0, tuple(range(len(sim.frames[0].points)))))
        assert all(abs(f.hip_shift_ratio) < CLEAN_SHIFT_MAX_RATIO for f in features)

    def test_a_staggered_stance_is_no_hip_shift(self):
        """The hips stay square to the bar with one foot ahead: the ankle line
        turns 13 deg, the hips' path does not."""
        _, features, _, _ = _analyse(Scenario(), mutate=_staggered(STAGGER_M))
        assert all(abs(f.hip_shift_ratio) < CLEAN_SHIFT_MAX_RATIO for f in features)

    @pytest.mark.parametrize("seed", range(4))
    def test_a_staggered_stance_reads_as_square_under_correlated_noise(self, seed: int):
        scenario = Scenario(seed=seed, bar_noise_m=TRACKED_BAR_NOISE_M)
        _, square, _, _ = _analyse(scenario, mutate=_correlated_noise(PLATFORM_KEYPOINT_NOISE_M, 0.8, seed))
        noise = _correlated_noise(PLATFORM_KEYPOINT_NOISE_M, 0.8, seed)
        _, staggered, _, _ = _analyse(scenario, mutate=_staggered(STAGGER_M, noise))
        assert len(staggered) == len(square)
        for square_rep, staggered_rep in zip(square, staggered):
            assert staggered_rep.hip_shift_ratio == pytest.approx(square_rep.hip_shift_ratio, abs=STANCE_LEAK_MAX_RATIO)

    def test_clean_sets_rarely_cue_hip_shift_under_correlated_noise(self):
        """The noise floor: the hip line's heading from a few seconds of noisy
        frames, carried over ~45 cm of forward travel. Within the VALIDATION.md
        gate of 1 false correction per 10 reps."""
        cued = reps = 0
        for seed in range(8):
            scenario = Scenario(seed=seed, bar_noise_m=TRACKED_BAR_NOISE_M)
            _, features, _, _ = _analyse(scenario, mutate=_correlated_noise(PLATFORM_KEYPOINT_NOISE_M, 0.8, seed))
            reps += len(features)
            cued += sum("deadlift_hip_shift" in rep_cues for rep_cues in _cued(features))
        assert reps == 24
        assert cued <= reps // 10

    @pytest.mark.parametrize("seed", range(4))
    def test_standing_turned_then_squaring_up_is_no_hip_shift(self, seed: int):
        """Turned toward something for 2 s, then squared up in place: a turn in
        place reads settled, but the frames before it no longer lock the axis."""
        scenario = Scenario(stance_s=3.0, seed=seed, bar_noise_m=TRACKED_BAR_NOISE_M)
        sim = simulate(scenario)
        stance_start = scenario.start_time + scenario.approach_s
        noise = _correlated_noise(0.015, 0.8, seed)
        mutate = _turned_then_squared(sim, 20.0, stance_start + 2.0, stance_start + 2.5, noise)
        _, features, _, _ = _analyse(scenario, mutate=mutate)
        assert len(features) == len(sim.reps)
        assert all("deadlift_hip_shift" not in cued for cued in _cued(features))

    def test_a_hip_line_turned_against_the_legs_is_no_hip_shift(self):
        """The hip keypoints' line 4 deg off legs and travel square to the bar (a
        habitual pelvic rotation, or 1.5 cm of front-back asymmetry between the
        two hip keypoints): the hip line leaks, the bar's axis does not."""
        _, features, _, _ = _analyse(Scenario(), mutate=_hip_line_turned(4.0))
        assert all(abs(f.hip_shift_ratio) < CLEAN_SHIFT_MAX_RATIO for f in features)

    def test_a_hip_line_turned_against_the_legs_with_one_wrist_hidden_is_no_hip_shift(self):
        """On the wrist proxy with one wrist never seen, the hands give no bar line:
        the feet's line stands in for it, not the hip line a second time."""
        hip_line_turned = _hip_line_turned(4.0)

        def one_wrist_hidden(index: int, frame: SimFrame) -> tuple[np.ndarray, np.ndarray, BarState3D | None]:
            points, confidences, bar = hip_line_turned(index, frame)
            confidences = confidences.copy()
            confidences[CK.RIGHT_WRIST] = 0.0
            return points, confidences, bar

        sim, features, _, _ = _analyse(Scenario(track_bar=False), mutate=one_wrist_hidden)
        assert len(features) == len(sim.reps)
        assert all(abs(f.hip_shift_ratio) < CLEAN_SHIFT_MAX_RATIO for f in features)

    def test_a_pelvis_twist_without_sideways_travel_is_no_hip_shift(self):
        sim = simulate(Scenario())
        _, features, _, _ = _analyse(Scenario(), mutate=_pelvis_twisting(sim, 15.0))
        assert all(abs(f.hip_shift_ratio) < CLEAN_SHIFT_MAX_RATIO for f in features)

    @pytest.mark.parametrize("track_bar", [True, False], ids=["tracked", "proxy"])
    @pytest.mark.parametrize("degrees", [10.0, -10.0])
    def test_a_real_shift_reads_its_size_when_the_pelvis_turns_with_it(self, degrees: float, track_bar: bool):
        """The hip axis is taken before the rep: a pelvis turning during the pull
        neither absorbs the shift nor inflates it."""
        shift_m = 0.20 * 2.0 * SimAthlete().stance_half_width_m
        scenario = Scenario(reps=[RepScript(hip_shift_m=shift_m)] * 3, track_bar=track_bar)
        _, features, _, _ = _analyse(scenario, mutate=_hip_line_turned(degrees, shift_m))
        assert all(
            f.hip_shift_ratio == pytest.approx(SQUARE_PELVIS_SHIFT_RATIO, abs=SHIFT_SIZE_TOLERANCE_RATIO) for f in features
        )

    def test_a_real_shift_is_still_seen_off_square(self):
        scenario = Scenario(reps=[RepScript(hip_shift_m=0.05)] * 2)
        sim = simulate(scenario)
        _, features, _, _ = _analyse(scenario, mutate=_turned(sim, 8.0, LOWER_BODY))
        assert all(verdict.get("deadlift_hip_shift") in ("moderate", "severe") for verdict in _judge(features))


class TestReSetupAtTheFloor:
    """A lifter who re-sets at the floor may move the feet: the next rep is judged
    against where they now stand."""

    @pytest.mark.parametrize("keypoint_noise_m", [0.0, PLATFORM_KEYPOINT_NOISE_M])
    def test_feet_moved_at_the_floor_are_judged_where_they_now_stand(self, keypoint_noise_m: float):
        scenario = Scenario(
            bar_midfoot_offset_m=0.06, reps=[RepScript(floor_hold_s=3.0), RepScript(), RepScript()],
            keypoint_noise_m=keypoint_noise_m, bar_noise_m=TRACKED_BAR_NOISE_M,
        )
        sim = simulate(scenario)
        mutate = _stepped_closer_at_the_floor(sim.reps[0].floor_time, FEET_MOVED_AT_THE_FLOOR_M)
        _, features, _, _ = _analyse(scenario, mutate=mutate)
        assert len(features) == len(sim.reps)
        assert features[0].bar_midfoot_setup_cm == pytest.approx(6.0, abs=1.0)
        assert "deadlift_bar_position" in _cued(features)[0]
        for later in features[1:]:
            assert later.bar_midfoot_setup_cm == pytest.approx(1.0, abs=1.0)
        assert all("deadlift_bar_position" not in cued for cued in _cued(features)[1:])

    def test_feet_hidden_at_the_floor_keep_the_stance_lock(self):
        def hide_feet_after_the_first_rep(index: int, frame: SimFrame) -> tuple[np.ndarray, np.ndarray, BarState3D | None]:
            confidences = frame.confidences.copy()
            if frame.timestamp >= floor_time:
                confidences[[CK.LEFT_ANKLE, CK.RIGHT_ANKLE, CK.LEFT_FOOT_INDEX, CK.RIGHT_FOOT_INDEX]] = 0.0
            return frame.points, confidences, frame.bar

        scenario = Scenario(bar_midfoot_offset_m=0.06, reps=[RepScript()] * 3)
        floor_time = simulate(scenario).reps[0].floor_time
        _, features, _, _ = _analyse(scenario, mutate=hide_feet_after_the_first_rep)
        assert all(f.bar_midfoot_setup_cm == pytest.approx(6.0, abs=1.0) for f in features)


class TestSetBehaviour:
    @pytest.mark.parametrize("keypoint_noise_m", [0.0, PLATFORM_KEYPOINT_NOISE_M])
    def test_standing_up_between_reps_re_judges_each_setup(self, keypoint_noise_m: float):
        scenario = Scenario(stand_between_reps_s=1.5, keypoint_noise_m=keypoint_noise_m, bar_noise_m=TRACKED_BAR_NOISE_M)
        sim, features, _, _ = _analyse(scenario)
        assert len(features) == len(sim.reps)
        assert all(f.setup_measured for f in features)
        assert _cued(features) == [set()] * len(features)

    def test_a_sticking_point_is_ground_through_not_a_failed_rep(self):
        scenario = Scenario(reps=[RepScript(pull_s=2.0, stall_fraction=0.35, stall_s=0.8)] * 3, bar_noise_m=TRACKED_BAR_NOISE_M)
        sim, features, analyzer, _ = _analyse(scenario)
        assert len(features) == len(sim.reps)
        assert analyzer.failed_reps == 0
        for expected, measured in zip(sim.reps, features):
            assert measured.top_time == pytest.approx(expected.top_time, abs=MAX_EVENT_ERROR_S)

    def test_a_stall_at_the_knees_that_comes_back_down_is_a_failed_rep(self):
        scenario = Scenario(reps=[RepScript(), RepScript(fail_rise_m=0.28), RepScript()])
        _, features, analyzer, _ = _analyse(scenario)
        assert len(features) == 2
        assert analyzer.failed_reps == 1

    @pytest.mark.parametrize("stance_s", [0.5, 1.5], ids=["no_standing_reference", "standing_reference"])
    @pytest.mark.parametrize("seed", range(3))
    def test_a_hitch_near_the_top_is_not_the_top(self, seed: int, stance_s: float):
        """A still bar with the trunk upright enough, a few cm short of the top
        (or anywhere, with no standing height to expect the top at), reads as a
        top until the bar rises on past it."""
        reps = [RepScript(pull_s=3.0, stall_fraction=0.9, stall_s=0.8)] * 3
        scenario = Scenario(reps=reps, approach_s=stance_s, stance_s=stance_s, keypoint_noise_m=0.01,
                            bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed)
        sim, features, analyzer, _ = _analyse(scenario)
        assert len(features) == len(sim.reps)
        assert analyzer.failed_reps == 0
        for expected, measured in zip(sim.reps, features):
            assert measured.top_time == pytest.approx(expected.top_time, abs=MAX_EVENT_ERROR_S)

    @pytest.mark.parametrize("stall_s", [0.4, 0.8])
    @pytest.mark.parametrize("stall_fraction", [0.94, 0.96])
    def test_a_grind_a_few_cm_short_of_lockout_is_not_the_top(self, stall_fraction: float, stall_s: float):
        """Stuck 2-4 cm short of lockout, then finished: the top is the lockout,
        and the stall's frames do not read as a short lockout (D6)."""
        reps = [RepScript(pull_s=2.5, stall_fraction=stall_fraction, stall_s=stall_s)] * 3
        sim, features, analyzer, _ = _analyse(Scenario(reps=reps, bar_noise_m=TRACKED_BAR_NOISE_M))
        assert len(features) == len(sim.reps)
        for expected, measured in zip(sim.reps, features):
            assert measured.top_time == pytest.approx(expected.top_time, abs=MAX_EVENT_ERROR_S)
        assert all("deadlift_lockout" not in cued for cued in _cued(features))

    def test_a_grind_short_of_lockout_on_the_wrist_proxy_is_not_judged_as_the_lockout(self):
        """A stall ~3 cm short sits inside the proxy's noise bands: the lockout is
        judged on the hold's last 0.5 s, and the top is the hips and knees reaching
        it, not the bar reaching the stall."""
        lockout_cues = off_tops = reps = 0
        for seed in range(6):
            scenario = Scenario(reps=[RepScript(pull_s=2.5, stall_fraction=0.95, stall_s=0.8)] * 3, track_bar=False, seed=seed)
            noise = _correlated_noise(PROXY_KEYPOINT_NOISE_M, PLATFORM_NOISE_RHO, 40 + seed)
            sim, features, _, _ = _analyse(scenario, mutate=noise)
            assert len(features) == len(sim.reps)
            reps += len(features)
            off_tops += sum(abs(f.top_time - truth.top_time) > MAX_PROXY_TOP_ERROR_S for f, truth in zip(features, sim.reps))
            lockout_cues += sum("deadlift_lockout" in cued for cued in _cued(features))
        assert lockout_cues <= reps // 10
        assert off_tops <= reps // 10

    @pytest.mark.parametrize("stance_s", [0.5, 1.5], ids=["no_standing_reference", "standing_reference"])
    def test_a_shrug_at_the_top_is_not_a_slower_rep(self, stance_s: float):
        """A shrug lifts the bar off a lockout on straight legs: no stall to resume
        from (held short of standing or, with no standing reference, the hips and
        knees extending as the bar rises), so the top stays near where the bar
        arrived and no velocity loss is read."""
        scenario = Scenario(reps=[RepScript(top_hold_s=1.2)] * 4, approach_s=stance_s, stance_s=stance_s,
                            bar_noise_m=TRACKED_BAR_NOISE_M)
        sim = simulate(scenario)
        _, features, _, _ = _analyse(scenario, mutate=_shrugged(sim, 0.025, rep_indices=(3,)))
        assert len(features) == len(sim.reps)
        assert features[3].top_time == pytest.approx(sim.reps[3].top_time, abs=SHRUG_TOP_SHIFT_MAX_S)
        assert "deadlift_velocity_loss" not in _judge(features)[3]

    def test_a_moderate_soft_lockout_is_cued_under_platform_noise(self):
        """The lockout is the hold's last 0.5 s, chosen by time: a window chosen by
        the angles it judges selects the noise and reads the lockout straighter."""
        cued = reps = 0
        for seed in range(6):
            scenario = Scenario(reps=[RepScript(top_hold_s=2.0, lockout_deficit_deg=14.0)] * 3,
                                bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed)
            noise = _correlated_noise(PLATFORM_KEYPOINT_NOISE_M, PLATFORM_NOISE_RHO, 700 + seed)
            sim, features, _, _ = _analyse(scenario, mutate=noise)
            assert len(features) == len(sim.reps)
            reps += len(features)
            cued += sum("deadlift_lockout" in rep_cues for rep_cues in _cued(features))
        assert cued >= reps - reps // 10

    def test_the_lockout_deficit_is_not_read_low_under_noise(self):
        noisy, noise_free = [], []
        for seed in range(6):
            scenario = Scenario(reps=[RepScript(top_hold_s=1.0, lockout_deficit_deg=12.0)] * 3,
                                bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed)
            noise = _correlated_noise(PLATFORM_KEYPOINT_NOISE_M, PLATFORM_NOISE_RHO, 700 + seed)
            _, features, _, _ = _analyse(scenario, mutate=noise)
            noisy += [f.hip_extension_deficit_deg for f in features]
            _, features, _, _ = _analyse(scenario)
            noise_free += [f.hip_extension_deficit_deg for f in features]
        assert float(np.median(noisy)) == pytest.approx(float(np.median(noise_free)), abs=MAX_LOCKOUT_BIAS_DEG)

    @pytest.mark.parametrize(
        ("track_bar", "bar_noise_m", "keypoint_noise_m"),
        [(False, 0.0, PROXY_KEYPOINT_NOISE_M), (True, NOISY_TRACKED_BAR_NOISE_M, PLATFORM_KEYPOINT_NOISE_M)],
        ids=["wrist_proxy", "noisy_tracked_bar"],
    )
    def test_noise_does_not_resume_a_soft_lockout(self, track_bar: bool, bar_noise_m: float, keypoint_noise_m: float):
        """A soft lockout opens the resume's deficit gate; the bar must still clear
        the hold band, frame by frame, so the noise does not re-date its top later
        in the hold."""
        errors = []
        for seed in range(6):
            scenario = Scenario(reps=[RepScript(top_hold_s=2.0, lockout_deficit_deg=12.0)] * 3,
                                track_bar=track_bar, bar_noise_m=bar_noise_m, seed=seed)
            noise = _correlated_noise(keypoint_noise_m, PLATFORM_NOISE_RHO, 900 + seed)
            sim, features, _, _ = _analyse(scenario, mutate=noise)
            assert len(features) == len(sim.reps)
            errors += [abs(measured.top_time - expected.top_time) for measured, expected in zip(features, sim.reps)]
        assert float(np.median(errors)) <= MAX_EVENT_ERROR_S
        assert max(errors) <= MAX_PROXY_TOP_ERROR_S

    def test_the_lowering_is_dated_where_the_bar_leaves_the_hold_on_the_wrist_proxy(self):
        """An event band of the wrists' noise reaches well into the lowering, and
        the lockout window would judge its frames."""
        errors = []
        for seed in range(6):
            sim, features, _, _ = _analyse(
                Scenario(track_bar=False, seed=seed),
                mutate=_correlated_noise(PLATFORM_KEYPOINT_NOISE_M, PLATFORM_NOISE_RHO, 900 + seed),
            )
            assert len(features) == len(sim.reps)
            hold_s = RepScript().top_hold_s
            errors += [
                (measured.floor_time - measured.lower_time_s) - (expected.top_time + hold_s)
                for measured, expected in zip(features, sim.reps)
            ]
        assert float(np.median(errors)) <= MAX_EVENT_ERROR_S

    def test_a_mild_soft_lockout_is_cued_no_more_often_on_the_wrist_proxy(self):
        """A 10 deg lockout reads 9.7 deg noise-free: mild, never cued. D6 is not
        a bar-measured rule, so the wrist proxy must not cue it more."""
        cued = {True: 0, False: 0}
        reps = 0
        for track_bar in (True, False):
            for body in BODIES:
                for seed in range(4):
                    scenario = Scenario(athlete=body, reps=[RepScript(lockout_deficit_deg=10.0)] * 3, track_bar=track_bar,
                                        bar_noise_m=TRACKED_BAR_NOISE_M if track_bar else 0.0, seed=seed)
                    noise = _correlated_noise(PLATFORM_KEYPOINT_NOISE_M, PLATFORM_NOISE_RHO, 900 + seed)
                    _, features, _, _ = _analyse(scenario, mutate=noise)
                    reps += len(features) if track_bar else 0
                    cued[track_bar] += sum("deadlift_lockout" in rep_cues for rep_cues in _cued(features))
        assert reps == 60
        assert cued[False] <= cued[True] + reps // 10

    def test_a_shrug_inside_the_hold_band_does_not_move_the_top(self):
        """A 1.5 cm shrug stays in the hold, and the lockout window catches its way
        down: the lockout's level is the window's lowest plateau, not its median."""
        errors = []
        for body in BODIES[:3]:
            for seed in range(4):
                scenario = Scenario(athlete=body, reps=[RepScript(top_hold_s=1.2)] * 3, bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed)
                sim = simulate(scenario)
                _, features, _, _ = _analyse(scenario, mutate=_shrugged(sim, 0.015, rep_indices=(0, 1, 2)))
                assert len(features) == len(sim.reps)
                errors += [abs(measured.top_time - expected.top_time) for measured, expected in zip(features, sim.reps)]
        assert max(errors) <= SHRUG_TOP_SHIFT_MAX_S

    @pytest.mark.parametrize("rep_indices", [(1,), (0, 1, 2, 3)], ids=["one_rep", "every_rep"])
    def test_a_lockout_that_settles_upward_does_not_move_the_top_or_read_slower(self, rep_indices: tuple[int, ...]):
        """Settled 1 cm up inside the hold band until the lowering, the lockout
        window sits on the raised plateau: on straight legs that is the same
        lockout, so its level is the first plateau after the climb, not a stall
        the bar climbed out of."""
        errors = []
        slowed = 0
        for body in BODIES:
            for seed in range(4):
                scenario = Scenario(athlete=body, reps=[RepScript(top_hold_s=1.2, lower_s=1.0)] * 4,
                                    bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed)
                sim = simulate(scenario)
                mutate = _settled_up(sim, SETTLE_RISE_M, rep_indices, hold_s=1.2, lower_s=1.0)
                _, features, _, _ = _analyse(scenario, mutate=mutate)
                assert len(features) == len(sim.reps)
                errors += [features[index].top_time - sim.reps[index].top_time for index in rep_indices]
                slowed += any("deadlift_velocity_loss" in judged for judged in _judge(features))
        assert max(map(abs, errors)) <= MAX_EVENT_ERROR_S, errors
        assert slowed == 0

    @pytest.mark.parametrize("sag_m", [0.012, 0.02])
    def test_a_lockout_that_sags_and_is_tightened_again_keeps_its_top(self, sag_m: float):
        """A plateau after the bar reached the lockout's level is a sag it came back
        up from, not a lockout settling upward (read as one, the top was dated
        0.13-0.17 s early)."""
        errors = []
        for body in BODIES:
            for seed in range(2):
                scenario = Scenario(athlete=body, reps=[RepScript(top_hold_s=1.2)] * 4, bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed)
                sim = simulate(scenario)
                _, features, _, _ = _analyse(scenario, mutate=_sagged(sim, sag_m, rep_indices=(0, 1, 2, 3)))
                assert len(features) == len(sim.reps)
                errors += [measured.top_time - expected.top_time for measured, expected in zip(features, sim.reps)]
        assert max(map(abs, errors)) <= MAX_EVENT_ERROR_S, errors

    def test_keypoint_noise_does_not_make_a_sag_a_stall(self):
        """Through 2 cm of correlated noise the joints on a sag's plateau can read a
        stall's 12 deg: the bar's first arrival at the level ends its climbs, so the
        tightening back up is no last climb (read as one, 9 of 64 tops were 0.7 s
        late and the recap read a velocity loss in 6 of 16 sets)."""
        late = slowed = 0
        for body_index, body in enumerate(BODIES[:4]):
            for seed in range(4):
                scenario = Scenario(athlete=body, reps=[RepScript(top_hold_s=1.2)] * 4, bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed)
                sim = simulate(scenario)
                noise = _correlated_noise(PLATFORM_KEYPOINT_NOISE_M, PLATFORM_NOISE_RHO, 7000 + 10 * body_index + seed)
                _, features, _, _ = _analyse(scenario, mutate=_sagged(sim, 0.015, rep_indices=(0, 1, 2, 3), then=noise))
                assert len(features) == len(sim.reps)
                late += sum(abs(measured.top_time - expected.top_time) > SHRUG_TOP_SHIFT_MAX_S
                            for measured, expected in zip(features, sim.reps))
                slowed += any("deadlift_velocity_loss" in judged for judged in _judge(features))
        assert late <= 1
        assert slowed == 0

    def test_a_shrug_with_the_knees_hidden_is_not_a_slower_rep(self):
        """With the knees hidden nothing tells a stall from a shrug: the pull does
        not resume."""
        scenario = Scenario(reps=[RepScript(top_hold_s=1.2)] * 4, bar_noise_m=TRACKED_BAR_NOISE_M)
        sim = simulate(scenario)
        mutate = _shrugged(sim, 0.025, rep_indices=(3,), then=_knees_hidden_off_the_floor(sim))
        _, features, _, _ = _analyse(scenario, mutate=mutate)
        assert len(features) == len(sim.reps)
        assert features[3].top_time == pytest.approx(sim.reps[3].top_time, abs=SHRUG_TOP_SHIFT_MAX_S)
        assert "deadlift_velocity_loss" not in _judge(features)[3]

    def test_keypoint_noise_does_not_resume_a_shrug_without_a_standing_reference(self):
        """Without a standing reference the hips and the knees must both extend as
        the bar rises: noise moves one joint at a time."""
        late = {0.0: 0, 0.025: 0}
        for noise_m in late:
            for body in BODIES[:3]:
                for seed in range(4):
                    scenario = Scenario(athlete=body, reps=[RepScript(top_hold_s=1.2)] * 3, approach_s=0.5, stance_s=0.5,
                                        bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed)
                    sim = simulate(scenario)
                    noise = _correlated_noise(noise_m, PLATFORM_NOISE_RHO, 60 + seed) if noise_m else None
                    _, features, _, _ = _analyse(scenario, mutate=_shrugged(sim, 0.04, rep_indices=(0, 1, 2), then=noise))
                    assert len(features) == len(sim.reps)
                    late[noise_m] += sum(
                        measured.top_time - expected.top_time > MAX_PROXY_TOP_ERROR_S
                        for measured, expected in zip(features, sim.reps)
                    )
        assert late[0.025] <= late[0.0] + 2

    @pytest.mark.parametrize("lean_back_deg", [25.0, 40.0])
    def test_an_over_extended_lockout_with_the_knees_hidden_is_counted(self, lean_back_deg: float):
        """The knees hidden at the top, the hips as far from the ankles as standing
        say the legs are straight."""
        scenario = Scenario(reps=[RepScript(lean_back_deg=lean_back_deg)] * 3, bar_noise_m=TRACKED_BAR_NOISE_M)
        first = simulate(scenario).frames[0].bar
        floor_bar_up = -(first.left_end_m[1] + first.right_end_m[1]) / 2.0

        def hide_knees_off_the_floor(index: int, frame: SimFrame) -> tuple[np.ndarray, np.ndarray, BarState3D | None]:
            confidences = frame.confidences.copy()
            if -(frame.bar.left_end_m[1] + frame.bar.right_end_m[1]) / 2.0 - floor_bar_up > KNEES_HIDDEN_ABOVE_M:
                confidences[[CK.LEFT_KNEE, CK.RIGHT_KNEE]] = 0.0
            return frame.points, confidences, frame.bar

        sim, features, analyzer, _ = _analyse(scenario, mutate=hide_knees_off_the_floor)
        assert len(features) == len(sim.reps)
        assert analyzer.failed_reps == 0

    def test_a_lean_back_from_mid_thigh_with_the_knees_hidden_is_a_failed_rep(self):
        """The knees hidden, the hips ~12 cm closer to the ankles than standing say
        they are bent."""
        scenario = Scenario(reps=[RepScript(), RepScript(fail_rise_m=0.35, lean_back_deg=40.0), RepScript()])

        def hide_knees_off_the_floor(index: int, frame: SimFrame) -> tuple[np.ndarray, np.ndarray, BarState3D | None]:
            confidences = frame.confidences.copy()
            bar_up = -(frame.bar.left_end_m[1] + frame.bar.right_end_m[1]) / 2.0
            if bar_up - floor_bar_up > KNEES_HIDDEN_ABOVE_M:
                confidences[[CK.LEFT_KNEE, CK.RIGHT_KNEE]] = 0.0
            return frame.points, confidences, frame.bar

        first = simulate(scenario).frames[0].bar
        floor_bar_up = -(first.left_end_m[1] + first.right_end_m[1]) / 2.0
        _, features, analyzer, _ = _analyse(scenario, mutate=hide_knees_off_the_floor)
        assert len(features) == 2
        assert analyzer.failed_reps == 1

    def test_a_failed_pull_leaning_back_from_mid_thigh_is_a_failed_rep(self):
        """Leaning back with the knees still bent ~60 deg is no lockout."""
        scenario = Scenario(reps=[RepScript(), RepScript(fail_rise_m=0.35, lean_back_deg=40.0), RepScript()])
        _, features, analyzer, _ = _analyse(scenario)
        assert len(features) == 2
        assert analyzer.failed_reps == 1

    @pytest.mark.parametrize("lean_back_deg", [25.0, 40.0])
    def test_an_over_extended_lockout_is_counted_and_judged(self, lean_back_deg: float):
        _, features, analyzer, _ = _analyse(Scenario(reps=[RepScript(lean_back_deg=lean_back_deg)] * 2))
        assert len(features) == 2
        assert analyzer.failed_reps == 0
        assert all(verdict.get("deadlift_lean_back") == "severe" for verdict in _judge(features))

    @pytest.mark.parametrize("lean_back_deg", [25.0, 40.0])
    def test_an_over_extended_lockout_with_no_top_to_expect_is_counted(self, lean_back_deg: float):
        """No standing reference and no earlier top: the legs straight still say
        a trunk leaning back is a lockout."""
        scenario = Scenario(reps=[RepScript(lean_back_deg=lean_back_deg)] * 2, approach_s=0.5, stance_s=0.5)
        sim, features, analyzer, _ = _analyse(scenario)
        assert len(features) == len(sim.reps)
        assert analyzer.failed_reps == 0

    def test_feet_hidden_at_setup_still_judge_the_bar_over_midfoot(self):
        def hide_feet_from_the_setup(index: int, frame: SimFrame) -> tuple[np.ndarray, np.ndarray, BarState3D | None]:
            confidences = frame.confidences.copy()
            if frame.timestamp - start_s > 3.5:
                confidences[[CK.LEFT_ANKLE, CK.RIGHT_ANKLE, CK.LEFT_FOOT_INDEX, CK.RIGHT_FOOT_INDEX]] = 0.0
            return frame.points, confidences, frame.bar

        scenario = Scenario(bar_midfoot_offset_m=0.06, reps=[RepScript()] * 2)
        start_s = scenario.start_time
        _, features, _, _ = _analyse(scenario, mutate=hide_feet_from_the_setup)
        assert all(f.bar_midfoot_setup_cm == pytest.approx(6.0, abs=1.0) for f in features)

    def test_the_live_offset_is_a_one_second_median(self):
        """One noisy frame must not arm the foot guidance."""
        sim = simulate(Scenario(keypoint_noise_m=PLATFORM_KEYPOINT_NOISE_M, bar_noise_m=TRACKED_BAR_NOISE_M))
        analyzer = DeadliftRepAnalyzer()
        analyzer.set_gravity(sim.gravity_up_world, GRAVITY_SOURCE_MEASURED)
        live = []
        for frame in sim.frames:
            if frame.timestamp >= sim.reps[0].liftoff_time:
                break
            analyzer.observe(DeadliftFrameInput(frame.timestamp, frame.frame_index, frame.points, frame.confidences, frame.bar))
            if analyzer.status.phase == DeadliftPhase.STANCE and math.isfinite(analyzer.status.bar_midfoot_live_cm):
                live.append(analyzer.status.bar_midfoot_live_cm)
        assert live
        assert max(abs(value) for value in live) <= BAR_MIDFOOT_GUIDANCE_TOLERANCE_CM

    @pytest.mark.parametrize("seed", range(6))
    def test_a_bar_over_midfoot_never_arms_the_guidance_under_correlated_noise(self, seed: int):
        sim = simulate(Scenario(reps=[RepScript()], bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed, stance_s=3.0))
        noise = _correlated_noise(PLATFORM_KEYPOINT_NOISE_M, 0.8, seed)
        analyzer = DeadliftRepAnalyzer()
        analyzer.set_gravity(sim.gravity_up_world, GRAVITY_SOURCE_MEASURED)
        live = []
        for index, frame in enumerate(sim.frames):
            points, confidences, bar = noise(index, frame)
            analyzer.observe(DeadliftFrameInput(frame.timestamp, frame.frame_index, points, confidences, bar))
            if math.isfinite(analyzer.status.bar_midfoot_live_cm):
                live.append(analyzer.status.bar_midfoot_live_cm)
        assert live
        assert max(abs(value) for value in live) <= BAR_MIDFOOT_GUIDANCE_ARM_CM

    def test_each_set_guides_its_own_stance(self):
        """The live offset speaks before each set's first rep, whatever the
        session's rep count."""
        sim = simulate(Scenario(bar_midfoot_offset_m=0.06, reps=[RepScript()]))
        analyzer = DeadliftRepAnalyzer()
        analyzer.set_gravity(sim.gravity_up_world, GRAVITY_SOURCE_MEASURED)
        live_by_set: list[list[float]] = []
        for set_index in range(2):
            analyzer.reset_set()
            offset_s = set_index * (sim.frames[-1].timestamp - sim.frames[0].timestamp + 1.0)
            live = []
            for frame in sim.frames:
                analyzer.observe(DeadliftFrameInput(
                    frame.timestamp + offset_s, frame.frame_index, frame.points, frame.confidences, frame.bar,
                ))
                if math.isfinite(analyzer.status.bar_midfoot_live_cm):
                    live.append(analyzer.status.bar_midfoot_live_cm)
            live_by_set.append(live)
        assert analyzer.rep_count == 2
        assert all(live and np.median(live) == pytest.approx(6.0, abs=0.5) for live in live_by_set)

    @pytest.mark.parametrize("seed", range(4))
    def test_standing_references_survive_platform_noise(self, seed: int):
        scenario = Scenario(keypoint_noise_m=PLATFORM_KEYPOINT_NOISE_M, bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed)
        _, features, _, _ = _analyse(scenario)
        assert all(math.isfinite(f.lean_back_deg) for f in features)

