#!/usr/bin/env python3
"""Re-measure the deadlift's simulated envelope (docs/deadlift/IMPLEMENTATION.md, KNOWLEDGE.md §7):
`PYTHONPATH=src python scripts/tools/deadlift_envelope.py <sweep>... [--salt N]`. Keypoint noise is
drawn independently for every sweep, row (noise level included), body and seed, so a row of n sets
is n draws; --salt N draws them all afresh. Each row prints its bodies and seeds.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import zlib
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from biomechanics.config import BiomechanicsConfig  # noqa: E402
from biomechanics.deadlift.analyzer import DeadliftFrameInput, DeadliftRepAnalyzer  # noqa: E402
from biomechanics.deadlift.bar_tracker_3d import DEFAULT_CONFIG as TRACKER_CONFIG  # noqa: E402
from biomechanics.deadlift.rule_base import TIER_RANK  # noqa: E402
from biomechanics.deadlift.session_reference import DeadliftSessionReference  # noqa: E402
from biomechanics.deadlift.simulator import (  # noqa: E402
    DEFAULT_FPS, RepScript, Scenario, SimAthlete, SimFrame, SimulatedSet, simulate,
)
from biomechanics.deadlift.types import GRAVITY_SOURCE_MEASURED, BarState3D, DeadliftRepFeatures  # noqa: E402
from biomechanics.faults.rule_engine import RuleEngine  # noqa: E402
from biomechanics.profiles.deadlift import DeadliftProfile  # noqa: E402
from biomechanics.utils.types import CocoKeypoints as CK  # noqa: E402
from biomechanics.utils.types import FaultEvent, JointAngles  # noqa: E402

BODIES = {
    "default": SimAthlete(),
    "short": SimAthlete(tibia_m=0.38, femur_m=0.40, torso_m=0.47, upper_arm_m=0.27, forearm_m=0.26,
                        hip_half_width_m=0.10, stance_half_width_m=0.11, shoulder_half_width_m=0.17,
                        grip_half_width_m=0.24),
    "tall": SimAthlete(tibia_m=0.48, femur_m=0.50, torso_m=0.57, upper_arm_m=0.33, forearm_m=0.31,
                       hip_half_width_m=0.14, stance_half_width_m=0.15, shoulder_half_width_m=0.21,
                       grip_half_width_m=0.28),
    "narrow_hips_wide_stance": SimAthlete(hip_half_width_m=0.095, stance_half_width_m=0.17, grip_half_width_m=0.29),
    "long_femurs": SimAthlete(tibia_m=0.40, femur_m=0.52, torso_m=0.48),
}
TRACKED_BAR_NOISE_M = 0.003
AR_RHO = 0.8
PULLS_S = (0.6, 1.2, 3.0)
TOP_LATE_S = 0.3
SHRUG_SHIFT_S = 0.18
KNEES_HIDDEN_ABOVE_M = 0.15
UPPER_BODY = (CK.LEFT_SHOULDER, CK.RIGHT_SHOULDER, CK.LEFT_ELBOW, CK.RIGHT_ELBOW, CK.LEFT_WRIST, CK.RIGHT_WRIST,
              CK.NOSE, CK.LEFT_EYE, CK.RIGHT_EYE, CK.LEFT_EAR, CK.RIGHT_EAR)
LOWER_BODY = (CK.LEFT_HIP, CK.RIGHT_HIP, CK.LEFT_KNEE, CK.RIGHT_KNEE, CK.LEFT_ANKLE, CK.RIGHT_ANKLE,
              CK.LEFT_HEEL, CK.RIGHT_HEEL, CK.LEFT_FOOT_INDEX, CK.RIGHT_FOOT_INDEX)

# The tracker's last measured centres its coasting velocity is fitted to.
TRACKER_VELOCITY_FRAMES = 4
# A lockout sagging after the top: from SAG_START_S, down and back up over
# SAG_RAMP_S each, back up from SAG_UP_S.
SAG_START_S = 0.1
SAG_RAMP_S = 0.2
SAG_UP_S = 0.6
# The holds a settle or a sag happens in, and the lowering after them.
HELD_TOP = RepScript(top_hold_s=1.2)
# A soft lockout cued moderate or worse on every rep, noise-free.
SOFT_LOCKOUT_DEG = 15.0

Mutation = Callable[[SimFrame], SimFrame]


# ---------------------------------------------------------------- running a set

# --salt: a fresh draw of every row's noise (0, the default, is the docs' draw).
SALT = {"value": 0}


def _rng(*key: object) -> np.random.Generator:
    salted = (SALT["value"], *key) if SALT["value"] else key
    return np.random.default_rng([zlib.crc32(str(part).encode()) for part in salted])


def _noise(model: str, sigma_m: float, *key: object) -> Mutation | None:
    if sigma_m <= 0.0:
        return None
    rng = _rng(*key)
    state: dict[str, np.ndarray] = {}

    def mutate(frame: SimFrame) -> SimFrame:
        if model == "iid":
            return frame._replace(points=frame.points + rng.normal(0.0, sigma_m, frame.points.shape))
        innovation = rng.normal(0.0, sigma_m * math.sqrt(1.0 - AR_RHO ** 2), frame.points.shape)
        state["error"] = innovation if "error" not in state else AR_RHO * state["error"] + innovation
        return frame._replace(points=frame.points + state["error"])

    return mutate


def _analyse(sim: SimulatedSet, *mutations: Mutation | None) -> list[DeadliftRepFeatures]:
    analyzer = DeadliftRepAnalyzer(BiomechanicsConfig().deadlift)
    analyzer.set_gravity(sim.gravity_up_world, GRAVITY_SOURCE_MEASURED)
    features = []
    for frame in sim.frames:
        for mutate in mutations:
            if mutate is not None:
                frame = mutate(frame)
        analyzer.observe(DeadliftFrameInput(frame.timestamp, frame.frame_index, frame.points, frame.confidences, frame.bar))
        while analyzer.take_completed_rep() is not None:
            features.append(analyzer.finish_rep())
    return features


def _faults(features: list[DeadliftRepFeatures]) -> list[list[FaultEvent]]:
    engine = RuleEngine(rules=DeadliftProfile().create_fault_rules(BiomechanicsConfig()),
                        reference=DeadliftSessionReference(), capture_mode="triangulated")
    return [engine.finish_rep(JointAngles(timestamp=f.top_time), f.rep_number, f) for f in features]


def _cued(features: list[DeadliftRepFeatures]) -> list[set[str]]:
    return [
        {fault.fault_type for fault in faults if TIER_RANK[fault.severity.value] >= TIER_RANK[fault.details["min_tier"]]}
        for faults in _faults(features)
    ]


def _judged(features: list[DeadliftRepFeatures]) -> list[set[str]]:
    return [{fault.fault_type for fault in faults} for faults in _faults(features)]


def _top_errors(sim: SimulatedSet, features: list[DeadliftRepFeatures]) -> list[float]:
    truth = [rep for rep in sim.reps if rep.counted]
    if len(features) != len(truth):
        return []
    return [f.top_time - t.top_time for f, t in zip(features, truth)]


def _summary(errors: list[float]) -> str:
    if not errors:
        return "no rep counted right"
    absolute = np.abs(errors)
    return (f"median {np.median(absolute):.3f} s, p90 {np.percentile(absolute, 90):.3f}, max {absolute.max():.3f} "
            f"(signed {min(errors):+.3f}..{max(errors):+.3f}); >0.1 s {int((absolute > 0.1).sum())}, "
            f">0.3 s {int((absolute > TOP_LATE_S).sum())}, >0.5 s {int((absolute > 0.5).sum())} of {len(errors)}")


def _cue_summary(cues: list[set[str]]) -> str:
    counts = Counter(fault for rep in cues for fault in rep)
    return f"{sum(1 for rep in cues if rep)} of {len(cues)} reps cued {dict(counts) or ''}"


def _not_counted(expected: int, cues: list[set[str]]) -> str:
    return f"; {expected - len(cues)} of {expected} reps not counted" if len(cues) != expected else ""


# --------------------------------------------------------------- frame mutations

def _shrugged(sim: SimulatedSet, rise_m: float) -> Mutation:
    starts = [rep.top_time + 0.25 for rep in sim.reps]

    def lift_m(t: float) -> float:
        for start in starts:
            if start <= t < start + 0.25:
                return rise_m * (t - start) / 0.25
            if start + 0.25 <= t < start + 0.55:
                return rise_m
            if start + 0.55 <= t < start + 0.8:
                return rise_m * (1.0 - (t - start - 0.55) / 0.25)
        return 0.0

    def mutate(frame: SimFrame) -> SimFrame:
        lift = lift_m(frame.timestamp)
        if lift <= 0.0:
            return frame
        points = frame.points.copy()
        points[list(UPPER_BODY), 1] -= lift
        bar = frame.bar
        if bar is not None:
            left, right = list(bar.left_end_m), list(bar.right_end_m)
            left[1] -= lift
            right[1] -= lift
            bar = bar.model_copy(update={"left_end_m": tuple(left), "right_end_m": tuple(right)})
        return frame._replace(points=points, bar=bar)

    return mutate


def _knees_hidden(sim: SimulatedSet) -> Mutation:
    first = sim.frames[0].bar
    floor_up = -(first.left_end_m[1] + first.right_end_m[1]) / 2.0

    def mutate(frame: SimFrame) -> SimFrame:
        if frame.bar is None or -(frame.bar.left_end_m[1] + frame.bar.right_end_m[1]) / 2.0 - floor_up <= KNEES_HIDDEN_ABOVE_M:
            return frame
        confidences = frame.confidences.copy()
        confidences[[CK.LEFT_KNEE, CK.RIGHT_KNEE]] = 0.0
        return frame._replace(confidences=confidences)

    return mutate


def _yaw(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    return np.array([[math.cos(angle), 0.0, math.sin(angle)], [0.0, 1.0, 0.0], [-math.sin(angle), 0.0, math.cos(angle)]])


def _hips_yawed(points: np.ndarray, degrees: float) -> np.ndarray:
    points = points.copy()
    middle = (points[CK.LEFT_HIP] + points[CK.RIGHT_HIP]) / 2.0
    for index in (CK.LEFT_HIP, CK.RIGHT_HIP):
        points[index] = (points[index] - middle) @ _yaw(degrees).T + middle
    return points


def _stance(sim: SimulatedSet, kind: str, amount: float, shift_m: float) -> Mutation:
    pivot = (sim.frames[0].points[CK.LEFT_ANKLE] + sim.frames[0].points[CK.RIGHT_ANKLE]) / 2.0
    hip_x0 = (sim.frames[0].points[CK.LEFT_HIP][0] + sim.frames[0].points[CK.RIGHT_HIP][0]) / 2.0
    stance_start = sim.frames[0].timestamp + 1.5
    floor_time = sim.reps[0].floor_time
    windows = [(rep.liftoff_time, rep.top_time, rep.floor_time) for rep in sim.reps if rep.counted]

    def mutate(frame: SimFrame) -> SimFrame:
        points = frame.points.copy()
        t = frame.timestamp
        if kind == "stagger":
            points[[CK.LEFT_ANKLE, CK.LEFT_HEEL, CK.LEFT_FOOT_INDEX], 2] -= amount
        elif kind == "off_square":
            points[list(LOWER_BODY)] = (points[list(LOWER_BODY)] - pivot) @ _yaw(amount).T + pivot
        elif kind == "both_axes_leak":
            points[list(LOWER_BODY)] = (points[list(LOWER_BODY)] - pivot) @ _yaw(amount).T + pivot
            points = _hips_yawed(points, -amount)
        elif kind == "turn_then_square":
            share = 1.0 if t <= stance_start + 2.0 else max(0.0, 1.0 - (t - stance_start - 2.0) / 0.5)
            points = (points - pivot) @ _yaw(amount * share).T + pivot
        elif kind == "floor_turn":
            share = min(1.0, max(0.0, (t - floor_time - 0.3) / 0.8))
            points = (points - pivot) @ _yaw(amount * share).T + pivot
        elif kind == "hip_yaw":
            points = _hips_yawed(points, amount)
        elif kind == "pelvis_with_shift":
            hip_x = (points[CK.LEFT_HIP][0] + points[CK.RIGHT_HIP][0]) / 2.0 - hip_x0
            points = _hips_yawed(points, amount * -hip_x / shift_m)
        elif kind == "twist":
            share = 0.0
            for liftoff, top, floor in windows:
                if liftoff <= t <= top:
                    share = min(1.0, (t - liftoff) / (0.5 * (top - liftoff)))
                elif top < t <= floor:
                    share = 1.0 - (t - top) / (floor - top)
            points = _hips_yawed(points, amount * share)
        return frame._replace(points=points)

    return mutate


def _settled_up(sim: SimulatedSet, rise_m: float, start_after_s: float, ramp_s: float, rep_indices: tuple[int, ...]) -> Mutation:
    hold_s, lower_s = HELD_TOP.top_hold_s, HELD_TOP.lower_s
    windows = [(sim.reps[index].top_time + start_after_s, sim.reps[index].top_time + hold_s) for index in rep_indices]

    def lift_m(t: float) -> float:
        for start, lowering in windows:
            if start <= t < start + ramp_s:
                return rise_m * (t - start) / ramp_s
            if start + ramp_s <= t < lowering:
                return rise_m
            if lowering <= t < lowering + lower_s:
                return rise_m * (1.0 - (t - lowering) / lower_s)
        return 0.0

    def mutate(frame: SimFrame) -> SimFrame:
        lift = lift_m(frame.timestamp)
        if lift <= 0.0:
            return frame
        points = frame.points.copy()
        points[list(UPPER_BODY), 1] -= lift
        bar = frame.bar
        if bar is not None:
            left, right = list(bar.left_end_m), list(bar.right_end_m)
            left[1] -= lift
            right[1] -= lift
            bar = bar.model_copy(update={"left_end_m": tuple(left), "right_end_m": tuple(right)})
        return frame._replace(points=points, bar=bar)

    return mutate


# The bar lowered by sag_m (shoulders relaxed) after the top of each of the reps,
# then tightened back up before the lowering.
def _sagged(sim: SimulatedSet, sag_m: float, rep_indices: tuple[int, ...]) -> Mutation:
    tops = [sim.reps[index].top_time for index in rep_indices]

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

    def mutate(frame: SimFrame) -> SimFrame:
        sag = sag_at(frame.timestamp)
        if sag <= 0.0:
            return frame
        points = frame.points.copy()
        points[list(UPPER_BODY), 1] += sag
        left, right = list(frame.bar.left_end_m), list(frame.bar.right_end_m)
        left[1] += sag
        right[1] += sag
        bar = frame.bar.model_copy(update={"left_end_m": tuple(left), "right_end_m": tuple(right)})
        return frame._replace(points=points, bar=bar)

    return mutate


# The bar lost on the frames lost(t) picks, as BarTracker3D reports it: a predicted
# state carried on at the last measured velocity for its max_prediction_s, then None.
def _coasting(lost: Callable[[float], bool]) -> Mutation:
    history: list[tuple[float, BarState3D]] = []

    def mutate(frame: SimFrame) -> SimFrame:
        if not lost(frame.timestamp):
            history.append((frame.timestamp, frame.bar))
            del history[:-TRACKER_VELOCITY_FRAMES]
            return frame
        if len(history) < 2 or frame.timestamp - history[-1][0] > TRACKER_CONFIG.max_prediction_s + 1e-9:
            return frame._replace(bar=None)
        last_t, last = history[-1]
        times = np.array([t for t, _ in history]) - last_t
        velocity = np.polyfit(times, np.array([bar.centre for _, bar in history]), 1)[0]
        shift = velocity * (frame.timestamp - last_t)
        return frame._replace(bar=last.model_copy(update={
            "timestamp": frame.timestamp,
            "left_end_m": tuple((np.asarray(last.left_end_m) + shift).tolist()),
            "right_end_m": tuple((np.asarray(last.right_end_m) + shift).tolist()),
            "predicted": True,
        }))

    return mutate


def _bar_lost(windows: list[tuple[float, float]]) -> Mutation:
    return _coasting(lambda t: any(start <= t < end for start, end in windows))


def _dropouts(probability: float, *key: object) -> Mutation:
    rng = _rng(*key)
    return _coasting(lambda t: bool(rng.random() < probability))


# A 15 Hz detector under a 30 Hz pose: every other frame the tracker's prediction.
def _half_rate() -> Mutation:
    return _coasting(lambda t: round(t * DEFAULT_FPS) % 2 == 1)


# The wrists unseen (confidence 0) in the windows: the wrist proxy's bar lost.
def _wrists_hidden(windows: list[tuple[float, float]]) -> Mutation:
    def mutate(frame: SimFrame) -> SimFrame:
        if not any(start <= frame.timestamp < end for start, end in windows):
            return frame
        confidences = frame.confidences.copy()
        confidences[[CK.LEFT_WRIST, CK.RIGHT_WRIST]] = 0.0
        return frame._replace(confidences=confidences)

    return mutate


def _knees_hidden_in(windows: list[tuple[float, float]]) -> Mutation:
    def mutate(frame: SimFrame) -> SimFrame:
        if not any(start <= frame.timestamp < end for start, end in windows):
            return frame
        confidences = frame.confidences.copy()
        confidences[[CK.LEFT_KNEE, CK.RIGHT_KNEE]] = 0.0
        return frame._replace(confidences=confidences)

    return mutate


# Touch-and-go reps with no pause at the top, the last a dead stop.
def _no_pause_touch_and_go(pull_s: float, lower_s: float) -> list[RepScript]:
    script = RepScript(top_hold_s=0.0, pull_s=pull_s, lower_s=lower_s)
    return [script.model_copy(update={"floor_hold_s": 0.0})] * 4 + [script]


# ------------------------------------------------------------------------ sweeps

def sweep_events() -> None:
    """Counting, event timing and false cues on clean sets, default body."""
    def table(name: str, track_bar: bool, touch_and_go: bool, rows: tuple[tuple[str, str, float], ...]) -> None:
        print(f"\n{name} (default body, pulls {PULLS_S} s x seeds 0-3)")
        for label, model, sigma_m in rows:
            exact = sets = 0
            errors: list[float] = []
            cues: list[set[str]] = []
            for pull_s in PULLS_S:
                for seed in range(4):
                    reps = ([RepScript(pull_s=pull_s, floor_hold_s=0.0)] * 4 + [RepScript(pull_s=pull_s)]
                            if touch_and_go else [RepScript(pull_s=pull_s)] * 3)
                    sim = simulate(Scenario(reps=reps, track_bar=track_bar,
                                            bar_noise_m=TRACKED_BAR_NOISE_M if track_bar else 0.0, seed=seed))
                    features = _analyse(sim, _noise(model, sigma_m, name, label, pull_s, seed))
                    truth = [rep for rep in sim.reps if rep.counted]
                    sets += 1
                    if len(features) != len(truth):
                        continue
                    exact += 1
                    cues += _cued(features)
                    for f, t in zip(features, truth):
                        for got, want in ((f.liftoff_time, t.liftoff_time), (f.knee_pass_time, t.knee_pass_time),
                                          (f.top_time, t.top_time), (f.floor_time, t.floor_time)):
                            if math.isfinite(got) and math.isfinite(want):
                                errors.append(abs(got - want) * 1000.0)
            print(f"| {label} | {exact}/{sets} | median {np.median(errors):.0f} ms, p95 {np.percentile(errors, 95):.0f} ms "
                  f"(max {max(errors):.0f}) | {_cue_summary(cues)} |")

    rows = (("none", "iid", 0.0), ("2.0 cm i.i.d.", "iid", 0.02), ("2.0 cm AR(0.8)", "ar", 0.02),
            ("2.5 cm i.i.d.", "iid", 0.025), ("2.5 cm AR(0.8)", "ar", 0.025))
    table("Tracked bar, dead stop", True, False, rows)
    table("Tracked bar, touch-and-go", True, True, (rows[0], rows[2]))
    proxy_rows = (("none", "iid", 0.0), ("1.0 cm i.i.d.", "iid", 0.01), ("1.0 cm AR(0.8)", "ar", 0.01),
                  ("2.0 cm i.i.d.", "iid", 0.02), ("2.0 cm AR(0.8)", "ar", 0.02))
    table("Wrist proxy, dead stop", False, False, proxy_rows)
    table("Wrist proxy, touch-and-go", False, True, (proxy_rows[0], proxy_rows[1], proxy_rows[4]))


def sweep_clean() -> None:
    """False cues on clean sets over five bodies."""
    for sigma_m in (0.02, 0.025):
        cues: list[set[str]] = []
        expected = 0
        for body_name, body in BODIES.items():
            for pull_s in PULLS_S:
                for seed in range(4):
                    sim = simulate(Scenario(athlete=body, reps=[RepScript(pull_s=pull_s)] * 3,
                                            bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed))
                    cues += _cued(_analyse(sim, _noise("ar", sigma_m, "clean_bar", sigma_m, body_name, pull_s, seed)))
                    expected += len(sim.reps)
        print(f"tracked, 5 bodies x pulls {PULLS_S} x seeds 0-3, {sigma_m * 100:.1f} cm AR: {_cue_summary(cues)}"
              f"{_not_counted(expected, cues)}")
    for sigma_m in (0.015, 0.02, 0.025):
        cues = []
        expected = 0
        for body_name, body in BODIES.items():
            for seed in range(6):
                sim = simulate(Scenario(athlete=body, reps=[RepScript()] * 3, track_bar=False, seed=seed))
                cues += _cued(_analyse(sim, _noise("ar", sigma_m, "clean_proxy", sigma_m, body_name, seed)))
                expected += len(sim.reps)
        print(f"proxy, 5 bodies x seeds 0-5, 0.6 s holds, {sigma_m * 100:.1f} cm AR: {_cue_summary(cues)}"
              f"{_not_counted(expected, cues)}")


def sweep_no_pause() -> None:
    """Clean reps with no pause at the top (0 and 0.1 s holds), five bodies x seeds 0-7."""
    # (label, touch-and-go, hold, pull, lowering)
    rows = (("dead stop", False, 0.0, 1.2, 1.0), ("dead stop", False, 0.1, 1.2, 1.0),
            ("touch-and-go", True, 0.0, 1.2, 1.0), ("touch-and-go", True, 0.0, 0.9, 0.8),
            ("touch-and-go", True, 0.0, 1.2, 0.5), ("touch-and-go", True, 0.0, 0.6, 0.5),
            ("touch-and-go", True, 0.1, 0.9, 0.8))
    for track_bar in (False, True):
        for label, touch_and_go, hold_s, pull_s, lower_s in rows:
            for sigma_m in (0.02, 0.024):
                cues: list[set[str]] = []
                expected = 0
                for body_name, body in BODIES.items():
                    for seed in range(8):
                        script = RepScript(top_hold_s=hold_s, pull_s=pull_s, lower_s=lower_s)
                        reps = [script.model_copy(update={"floor_hold_s": 0.0})] * 4 + [script] if touch_and_go else [script] * 3
                        sim = simulate(Scenario(athlete=body, reps=reps, track_bar=track_bar,
                                                bar_noise_m=TRACKED_BAR_NOISE_M if track_bar else 0.0, seed=seed))
                        features = _analyse(sim, _noise("ar", sigma_m, "no_pause", track_bar, label, hold_s, pull_s, lower_s,
                                                        sigma_m, body_name, seed))
                        cues += _cued(features)
                        expected += len(sim.reps)
                source = "tracked" if track_bar else "proxy"
                print(f"{source} {label} hold {hold_s} s, pull {pull_s} s / lowering {lower_s} s, {sigma_m * 100:.1f} cm AR: "
                      f"{_cue_summary(cues)}{_not_counted(expected, cues)}")


def sweep_lockout() -> None:
    """D6 under noise: recall of a moderate soft lockout, the reading's bias, a mild one."""
    cued = reps = 0
    for body_name, body in BODIES.items():
        for seed in range(6):
            sim = simulate(Scenario(athlete=body, reps=[RepScript(top_hold_s=2.0, lockout_deficit_deg=14.0)] * 3,
                                    bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed))
            features = _analyse(sim, _noise("ar", 0.02, "lockout14", body_name, seed))
            reps += len(features)
            cued += sum("deadlift_lockout" in rep for rep in _cued(features))
    print(f"14 deg soft lockout, tracked, 5 bodies x seeds 0-5, 2 cm AR: cued {cued} of {reps}")
    noisy: list[float] = []
    clean: list[float] = []
    for body_name, body in BODIES.items():
        for seed in range(6):
            sim = simulate(Scenario(athlete=body, reps=[RepScript(top_hold_s=1.0, lockout_deficit_deg=12.0)] * 3,
                                    bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed))
            noisy += [f.hip_extension_deficit_deg for f in _analyse(sim, _noise("ar", 0.02, "lockout12", body_name, seed))]
            clean += [f.hip_extension_deficit_deg for f in _analyse(sim)]
    print(f"12 deg soft lockout, tracked, 5 bodies x seeds 0-5: median reading {np.nanmedian(noisy):.1f} deg at 2 cm AR, "
          f"{np.nanmedian(clean):.1f} noise-free")
    for track_bar in (True, False):
        cued = reps = 0
        for body_name, body in BODIES.items():
            for seed in range(4):
                sim = simulate(Scenario(athlete=body, reps=[RepScript(lockout_deficit_deg=10.0)] * 3, track_bar=track_bar,
                                        bar_noise_m=TRACKED_BAR_NOISE_M if track_bar else 0.0, seed=seed))
                features = _analyse(sim, _noise("ar", 0.02, "lockout10", track_bar, body_name, seed))
                reps += len(features)
                cued += sum("deadlift_lockout" in rep for rep in _cued(features))
        print(f"10 deg (mild) soft lockout, {'tracked' if track_bar else 'proxy'}, 5 bodies x seeds 0-3, 2 cm AR: "
              f"cued {cued} of {reps}")


def sweep_tops() -> None:
    """The top event: tracked grinds and slow pulls, and the wrist proxy's tops."""
    def run(label: str, scripts: list[RepScript], track_bar: bool, bar_noise_m: float, sigma_m: float, seeds: int,
            bodies: tuple[str, ...]) -> None:
        errors: list[float] = []
        for body_name in bodies:
            for seed in range(seeds):
                sim = simulate(Scenario(athlete=BODIES[body_name], reps=scripts, track_bar=track_bar,
                                        bar_noise_m=bar_noise_m, seed=seed))
                errors += _top_errors(sim, _analyse(sim, _noise("ar", sigma_m, "tops", label, body_name, seed)))
        print(f"{label} ({len(bodies)} bodies x seeds 0-{seeds - 1}): {_summary(errors)}")

    one = ("default",)
    for fraction in (0.94, 0.96):
        for stall_s in (0.4, 0.8):
            run(f"tracked grind {fraction:.0%} stall {stall_s} s", [RepScript(pull_s=2.5, stall_fraction=fraction, stall_s=stall_s)] * 3,
                True, TRACKED_BAR_NOISE_M, 0.0, 1, one)
    run("tracked grind 95%, 0.3 s finish", [RepScript(pull_s=1.6, stall_fraction=0.95, stall_s=0.8, finish_s=0.3)] * 3,
        True, TRACKED_BAR_NOISE_M, 0.0, 8, one)
    run("tracked grind 95%, 1 s finish", [RepScript(pull_s=2.0, stall_fraction=0.95, stall_s=0.8, finish_s=1.0)] * 3,
        True, TRACKED_BAR_NOISE_M, 0.0, 4, one)
    for fraction in (0.96, 0.97):
        run(f"tracked grind {fraction:.0%}, 1 s finish",
            [RepScript(pull_s=2.0, stall_fraction=fraction, stall_s=0.8, finish_s=1.0)] * 2,
            True, TRACKED_BAR_NOISE_M, 0.0, 2, tuple(BODIES))
    run("tracked 5 s pull, 2 cm AR", [RepScript(pull_s=5.0)] * 3, True, TRACKED_BAR_NOISE_M, 0.02, 6, tuple(BODIES))
    run("tracked soft lockout 12 deg held 2 s, 5 mm bar, 2 cm AR",
        [RepScript(top_hold_s=2.0, lockout_deficit_deg=12.0)] * 3, True, 0.005, 0.02, 6, tuple(BODIES))
    for sigma_m in (0.015, 0.02):
        for label, script in (("plain", RepScript()), ("5 s pull", RepScript(pull_s=5.0)),
                              ("grind 95% 0.8 s", RepScript(pull_s=2.5, stall_fraction=0.95, stall_s=0.8)),
                              ("soft lockout 12 deg held 2 s", RepScript(top_hold_s=2.0, lockout_deficit_deg=12.0))):
            run(f"proxy {label}, {sigma_m * 100:.1f} cm AR", [script] * 3, False, 0.0, sigma_m, 6, tuple(BODIES))


def sweep_shrug() -> None:
    """Shrugs at the top of every rep (1.2 s holds): the top's shift and the recap's D10."""
    for stance_s in (1.5, 0.5):
        for knees_hidden in (False, True):
            for rise_m in (0.015, 0.025, 0.04):
                for sigma_m in (0.0, 0.025):
                    errors: list[float] = []
                    slowed = sets = 0
                    for body_name in ("default", "short", "tall", "long_femurs"):
                        for seed in range(4):
                            sim = simulate(Scenario(athlete=BODIES[body_name], reps=[RepScript(top_hold_s=1.2)] * 3,
                                                    approach_s=stance_s, stance_s=stance_s,
                                                    bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed))
                            features = _analyse(sim, _shrugged(sim, rise_m), _knees_hidden(sim) if knees_hidden else None,
                                                _noise("ar", sigma_m, "shrug", stance_s, knees_hidden, rise_m, body_name, seed))
                            errors += _top_errors(sim, features)
                            sets += 1
                            slowed += any("deadlift_velocity_loss" in rep for rep in _judged(features))
                    late = sum(error > SHRUG_SHIFT_S for error in errors)
                    print(f"shrug {rise_m * 100:.1f} cm, {'standing reference' if stance_s > 1.0 else 'no standing reference'}, "
                          f"knees {'hidden' if knees_hidden else 'seen'}, {sigma_m * 100:.1f} cm AR (4 bodies x seeds 0-3): "
                          f"tops > {SHRUG_SHIFT_S} s late {late} of {len(errors)} (max {max(errors):+.3f}), sets with D10 {slowed} of {sets}")
    # A lockout that settles upward and stays up (shoulders drawn back, legs straight),
    # on every rep or only the second.
    for rise_m, start_after_s, ramp_s in ((0.008, 0.2, 0.4), (0.01, 0.3, 0.3), (0.015, 0.2, 0.4)):
        for every_rep in (False, True):
            for sigma_m in (0.0, 0.02):
                errors = []
                slowed = sets = 0
                for body_name in ("default", "short", "tall", "long_femurs"):
                    for seed in range(4):
                        sim = simulate(Scenario(athlete=BODIES[body_name], reps=[HELD_TOP] * 4,
                                                bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed))
                        indices = tuple(range(4)) if every_rep else (1,)
                        features = _analyse(sim, _settled_up(sim, rise_m, start_after_s, ramp_s, indices),
                                            _noise("ar", sigma_m, "settle", rise_m, every_rep, sigma_m, body_name, seed))
                        rep_errors = _top_errors(sim, features)
                        errors += rep_errors if every_rep else rep_errors[1:2]
                        sets += 1
                        slowed += any("deadlift_velocity_loss" in rep for rep in _judged(features))
                late = sum(error > SHRUG_SHIFT_S for error in errors)
                print(f"lockout settling {rise_m * 100:.1f} cm up from +{start_after_s} s, {'every rep' if every_rep else 'rep 2'}, "
                      f"{sigma_m * 100:.1f} cm AR (4 bodies x seeds 0-3): tops > {SHRUG_SHIFT_S} s late {late} of {len(errors)} "
                      f"(max {max(errors):+.3f}), sets with D10 {slowed} of {sets}")
    # A lockout that sags (shoulders relaxed) and is tightened again before the lowering, every rep.
    for sag_m in (0.012, 0.015, 0.02):
        for sigma_m in (0.0, 0.02):
            errors = []
            slowed = 0
            for body_name in ("default", "short", "tall", "long_femurs"):
                for seed in range(4):
                    sim = simulate(Scenario(athlete=BODIES[body_name], reps=[HELD_TOP] * 4, bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed))
                    features = _analyse(sim, _sagged(sim, sag_m, tuple(range(4))),
                                        _noise("ar", sigma_m, "sag", sag_m, sigma_m, body_name, seed))
                    errors += _top_errors(sim, features)
                    slowed += any("deadlift_velocity_loss" in rep for rep in _judged(features))
            print(f"lockout sagging {sag_m * 100:.1f} cm and tightened again, every rep, {sigma_m * 100:.1f} cm AR "
                  f"(4 bodies x seeds 0-3): {_summary(errors)}; sets with D10 {slowed} of 16")


def sweep_hip_shift() -> None:
    """D8: false cues by stance and pelvis, and recall of a real shift (30 sets: 5 bodies x seeds 0-5)."""
    rows = (("clean", "none", 0.0, 0.02, 0.0), ("clean 2.5 cm", "none", 0.0, 0.025, 0.0),
            ("left foot 6 cm ahead", "stagger", 0.06, 0.02, 0.0), ("lower body 8 deg off the bar", "off_square", 8.0, 0.02, 0.0),
            ("hip line 4 deg off the legs", "hip_yaw", 4.0, 0.02, 0.0), ("15 deg pelvis twist", "twist", 15.0, 0.02, 0.0),
            ("both axes 6 deg off the travel", "both_axes_leak", 6.0, 0.02, 0.0),
            ("turned 20 deg then squared up, 1.5 cm", "turn_then_square", 20.0, 0.015, 0.0),
            ("turned 10 deg at the floor, 1.5 cm", "floor_turn", 10.0, 0.015, 0.0),
            ("real shift 0.15", "none", 0.0, 0.02, 0.15), ("real shift 0.20", "none", 0.0, 0.02, 0.20),
            ("real shift 0.20, pelvis +10 deg with it", "pelvis_with_shift", 10.0, 0.02, 0.20),
            ("real shift 0.20, pelvis -10 deg with it", "pelvis_with_shift", -10.0, 0.02, 0.20),
            ("real shift 0.25", "none", 0.0, 0.02, 0.25))
    for track_bar in (True, False):
        for label, kind, amount, sigma_m, shift_ratio in rows:
            cued = reps = 0
            readings: list[float] = []
            for body_name, body in BODIES.items():
                for seed in range(6):
                    shift_m = shift_ratio * 2.0 * body.stance_half_width_m
                    first_floor = kind == "floor_turn"
                    reps_scripts = ([RepScript(floor_hold_s=2.0)] + [RepScript()] * 3 if first_floor
                                    else [RepScript(hip_shift_m=shift_m)] * 3)
                    sim = simulate(Scenario(athlete=body, reps=reps_scripts, track_bar=track_bar,
                                            bar_noise_m=TRACKED_BAR_NOISE_M if track_bar else 0.0, seed=seed,
                                            stance_s=3.0 if kind == "turn_then_square" else 1.5))
                    mutate = _stance(sim, kind, amount, max(shift_m, 1e-6)) if kind != "none" else None
                    features = _analyse(sim, mutate, _noise("ar", sigma_m, "hip_shift", track_bar, label, body_name, seed))
                    cues = _cued(features)
                    if first_floor:
                        # The turn comes after the first rep.
                        features, cues = features[1:], cues[1:]
                    reps += len(features)
                    cued += sum("deadlift_hip_shift" in rep for rep in cues)
                    readings += [f.hip_shift_ratio for f in features if math.isfinite(f.hip_shift_ratio)]
            print(f"{'tracked' if track_bar else 'proxy'} {label}: D8 cued {cued} of {reps} (median reading "
                  f"{np.median(readings):+.3f})")


def sweep_gaps() -> None:
    """Bar dropouts and a bar lost around the top as the tracker reports it (predicted, then none), a 15 Hz
    detector, and the wrist proxy's wrists hidden around the top; 1.5 cm AR."""
    for probability in (0.1, 0.2):
        errors: list[float] = []
        for body_name in ("default", "short", "tall"):
            for seed in range(4):
                sim = simulate(Scenario(athlete=BODIES[body_name], reps=[RepScript(pull_s=2.5)] * 3,
                                        bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed))
                errors += _top_errors(sim, _analyse(sim, _dropouts(probability, "dropouts", probability, body_name, seed),
                                                    _noise("ar", 0.015, "dropout_keypoints", probability, body_name, seed)))
        print(f"{probability:.0%} random bar dropouts, 2.5 s pulls (3 bodies x seeds 0-3): {_summary(errors)}")
    for kind, scripts in (("plain", [RepScript(pull_s=1.5)] * 3),
                          ("grind 93%", [RepScript(pull_s=1.6, stall_fraction=0.93, stall_s=0.6, finish_s=0.4)] * 3)):
        for offset_s, gap_s in ((-0.9, 1.0), (-0.7, 0.8), (-0.5, 0.6), (-0.7, 1.0), (-0.3, 0.2), (-0.3, 0.4), (-0.3, 0.6),
                                (-0.1, 0.2), (-0.1, 0.4), (-0.1, 0.6), (0.0, 0.4), (0.2, 0.4)):
            errors = []
            for body_name in ("default", "short", "tall"):
                for seed in range(3):
                    sim = simulate(Scenario(athlete=BODIES[body_name], reps=scripts, bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed))
                    lost = _bar_lost([(rep.top_time + offset_s, rep.top_time + offset_s + gap_s) for rep in sim.reps])
                    errors += _top_errors(sim, _analyse(sim, lost, _noise("ar", 0.015, "gaps", kind, offset_s, gap_s,
                                                                            body_name, seed)))
            print(f"{kind}: bar lost {gap_s} s from {offset_s:+.1f} s of the top (3 bodies x seeds 0-2): {_summary(errors)}")
    # Lost on its way into a grind's stall, until just after the top.
    for stall_s, finish_s, before_s in ((0.4, 0.3, 0.1), (0.6, 0.4, 0.2)):
        grind = RepScript(pull_s=1.6, stall_fraction=0.93, stall_s=stall_s, finish_s=finish_s)
        errors = []
        for body_name in ("default", "short", "tall"):
            for seed in range(3):
                sim = simulate(Scenario(athlete=BODIES[body_name], reps=[grind] * 3, bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed))
                lost = _bar_lost([(rep.top_time - finish_s - stall_s - before_s, rep.top_time + 0.1) for rep in sim.reps])
                errors += _top_errors(sim, _analyse(sim, lost, _noise("ar", 0.015, "gaps_into_stall", stall_s, body_name, seed)))
        print(f"grind 93% (stall {stall_s} s, finish {finish_s} s): bar lost from {before_s} s before the stall to 0.1 s "
              f"after the top (3 bodies x seeds 0-2): {_summary(errors)}")
    # A top that never held, the bar lost across it on reps 2-4.
    for pull_s, lower_s in ((1.2, 1.0), (0.9, 0.8)):
        script = RepScript(top_hold_s=0.0, pull_s=pull_s, lower_s=lower_s)
        reps = [script.model_copy(update={"floor_hold_s": 0.0})] * 4 + [script]
        errors = []
        cues: list[set[str]] = []
        exact = sets = 0
        for body_name, body in BODIES.items():
            for seed in range(4):
                sim = simulate(Scenario(athlete=body, reps=reps, bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed))
                lost = _bar_lost([(rep.top_time - 0.2, rep.top_time + 0.2) for rep in sim.reps[1:4]])
                features = _analyse(sim, lost, _noise("ar", 0.015, "gaps_no_pause", pull_s, body_name, seed))
                rep_errors = _top_errors(sim, features)
                sets += 1
                if rep_errors:
                    exact += 1
                    errors += rep_errors[1:4]
                    cues += _cued(features)[1:4]
        lockout = sum("deadlift_lockout" in rep for rep in cues)
        print(f"no-pause touch-and-go {pull_s} / {lower_s} s, bar lost 0.2 s either side of reps 2-4's tops (5 bodies x "
              f"seeds 0-3): {exact}/{sets} sets exact; {_summary(errors)}; D6 cued on {lockout} of {len(cues)}")
    # A 0.6 s hold never seen: the bar lost until the lowering, by pull duration and
    # when the loss began; then, on 1.2 s pulls, lost from just before the top or
    # just after it arrived.
    holds = [(pull_s, start_s, 0.8) for pull_s in (0.6, 1.2, 1.5, 2.0, 2.5, 5.0) for start_s in (-0.4, -0.3, -0.2)]
    for pull_s, start_s, end_s in holds + [(1.2, -0.1, 0.7), (1.2, 0.03, 0.75)]:
        errors = []
        cues = []
        slowed = sets = 0
        for body_name, body in BODIES.items():
            for seed in range(3):
                sim = simulate(Scenario(athlete=body, reps=[RepScript(pull_s=pull_s)] * 3,
                                        bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed))
                lost = _bar_lost([(rep.top_time + start_s, rep.top_time + end_s) for rep in sim.reps])
                features = _analyse(sim, lost, _noise("ar", 0.015, "gaps_hold", pull_s, start_s, body_name, seed))
                errors += _top_errors(sim, features)
                cues += _cued(features)
                sets += 1
                slowed += any("deadlift_velocity_loss" in rep for rep in _judged(features))
        lockout = sum("deadlift_lockout" in rep for rep in cues)
        print(f"{pull_s} s pulls held 0.6 s, bar lost from {start_s:+.2f} s to {end_s:+.2f} s of the top (5 bodies x "
              f"seeds 0-2): {_summary(errors)}; D6 cued on {lockout} of {len(cues)}; D10 judged in {slowed} of {sets} sets")
    # A grind lost through its finish and hold, as the hold rows lose it.
    for fraction in (0.93, 0.95, 0.97):
        grind = RepScript(pull_s=1.6, stall_fraction=fraction, stall_s=0.6, finish_s=0.4)
        errors = []
        cues = []
        for body_name in ("default", "short", "tall"):
            for seed in range(3):
                sim = simulate(Scenario(athlete=BODIES[body_name], reps=[grind] * 3, bar_noise_m=TRACKED_BAR_NOISE_M,
                                        seed=seed))
                lost = _bar_lost([(rep.top_time - 0.2, rep.top_time + 0.8) for rep in sim.reps])
                features = _analyse(sim, lost, _noise("ar", 0.015, "gaps_grind_hold", fraction, body_name, seed))
                errors += _top_errors(sim, features)
                cues += _cued(features)
        lockout = sum("deadlift_lockout" in rep for rep in cues)
        print(f"grind {fraction:.0%} (stall 0.6 s, finish 0.4 s), bar lost from 0.2 s before the top to 0.8 s after "
              f"(3 bodies x seeds 0-2): {_summary(errors)}; D6 cued on {lockout} of {len(cues)}")
    # Faulted lockouts (15 deg leaned back, 15 deg soft) on 1.2 s pulls held 0.6 s, the bar lost
    # until the lowering from before the top, or seen arriving first: dated, and cued.
    faulted = (("15 deg lean-back", RepScript(pull_s=1.2, lean_back_deg=15.0), "deadlift_lean_back"),
               ("15 deg soft lockout", RepScript(pull_s=1.2, lockout_deficit_deg=15.0), "deadlift_lockout"))
    for label, script, fault in faulted:
        for start_s in (-0.2, 0.05, 0.1):
            errors = []
            cues = []
            for body_name, body in BODIES.items():
                for seed in range(3):
                    sim = simulate(Scenario(athlete=body, reps=[script] * 3, bar_noise_m=TRACKED_BAR_NOISE_M,
                                            seed=seed))
                    lost = _bar_lost([(rep.top_time + start_s, rep.top_time + 0.8) for rep in sim.reps])
                    features = _analyse(sim, lost, _noise("ar", 0.015, "gaps_faulted", label, start_s, body_name, seed))
                    errors += _top_errors(sim, features)
                    cues += _cued(features)
            cued = sum(fault in rep for rep in cues)
            print(f"{label} held 0.6 s, bar lost from {start_s:+.2f} s to +0.80 s of the top (5 bodies x seeds 0-2): "
                  f"{_summary(errors)}; {fault} cued on {cued} of {len(cues)}")
    # A 1.5 s hold seen for 0.15 s (PULL -> TOP), then lost until 0.2 s into the lowering: nothing
    # stalled, so nothing resumes (clean, a 15 deg soft lockout, no standing reference).
    long_holds = (("clean", RepScript(pull_s=1.2, top_hold_s=1.5), 1.5),
                  ("15 deg soft lockout", RepScript(pull_s=1.2, top_hold_s=1.5, lockout_deficit_deg=15.0), 1.5),
                  ("no standing reference", RepScript(pull_s=1.2, top_hold_s=1.5), 0.5))
    for sigma_m in (0.015, 0.02):
        for label, script, stance_s in long_holds:
            errors = []
            for body_name, body in BODIES.items():
                for seed in range(3):
                    sim = simulate(Scenario(athlete=body, reps=[script] * 3, bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed,
                                            approach_s=stance_s, stance_s=stance_s))
                    lost = _bar_lost([(rep.top_time + 0.15, rep.top_time + 1.7) for rep in sim.reps])
                    errors += _top_errors(sim, _analyse(sim, lost, _noise("ar", sigma_m, "gaps_seen_hold", sigma_m,
                                                                           label, body_name, seed)))
            print(f"{label}, 1.5 s hold seen 0.15 s then lost until 0.2 s into the lowering, {sigma_m * 100:.1f} cm AR "
                  f"(5 bodies x seeds 0-2): {_summary(errors)}")
    # No standing reference, every top lost (no expected top height to fit to).
    errors = []
    for body_name, body in BODIES.items():
        for seed in range(3):
            sim = simulate(Scenario(athlete=body, reps=[RepScript(pull_s=1.5)] * 3, bar_noise_m=TRACKED_BAR_NOISE_M,
                                    seed=seed, approach_s=0.5, stance_s=0.5))
            lost = _bar_lost([(rep.top_time - 0.3, rep.top_time + 0.8) for rep in sim.reps])
            errors += _top_errors(sim, _analyse(sim, lost, _noise("ar", 0.015, "gaps_no_reference", body_name, seed)))
    print(f"no standing reference, 1.5 s pulls held 0.6 s, bar lost from -0.30 s to +0.80 s of every top "
          f"(5 bodies x seeds 0-2): {_summary(errors)}")
    # The knees hidden with the bar, from before the top until the lowering: no plateau of
    # the hips and knees to read in the gap; standing seen by the legs' length.
    for pull_s in (1.2, 2.0):
        for start_s in (-0.3, -0.2):
            errors = []
            cues = []
            expected = 0
            for body_name, body in BODIES.items():
                for seed in range(3):
                    sim = simulate(Scenario(athlete=body, reps=[RepScript(pull_s=pull_s)] * 3,
                                            bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed))
                    windows = [(rep.top_time + start_s, rep.top_time + 0.8) for rep in sim.reps]
                    features = _analyse(sim, _bar_lost(windows), _knees_hidden_in(windows),
                                        _noise("ar", 0.015, "gaps_knees_hidden", pull_s, start_s, body_name, seed))
                    errors += _top_errors(sim, features)
                    cues += _cued(features)
                    expected += len(sim.reps)
            print(f"{pull_s} s pulls held 0.6 s, knees and bar hidden from {start_s:+.2f} s to +0.80 s of the top "
                  f"(5 bodies x seeds 0-2): {_summary(errors)}; {_cue_summary(cues)}{_not_counted(expected, cues)}")
    # Short gaps through a hold: no standing reference, the bar dropped for 0.1 s every
    # 0.25 s through a 1.5 s hold, at 2.5 cm (nothing stalled, nothing resumes).
    script = RepScript(pull_s=1.2, top_hold_s=1.5)
    errors = []
    for body_name, body in BODIES.items():
        for seed in range(3):
            sim = simulate(Scenario(athlete=body, reps=[script] * 3, bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed,
                                    approach_s=0.5, stance_s=0.5))
            drops = [(rep.top_time + 0.1 + 0.25 * k, rep.top_time + 0.2 + 0.25 * k) for rep in sim.reps
                     for k in range(5)]
            errors += _top_errors(sim, _analyse(sim, _bar_lost(drops), _noise("ar", 0.025, "gaps_short", body_name,
                                                                              seed)))
    print(f"no standing reference, 1.5 s hold, bar lost 0.1 s every 0.25 s through it, 2.5 cm AR "
          f"(5 bodies x seeds 0-2): {_summary(errors)}")
    # A 15 Hz detector through the tracker.
    for label, scripts in (("0.6 s pulls", [RepScript(pull_s=0.6)] * 3), ("1.2 s pulls", [RepScript(pull_s=1.2)] * 3),
                           ("5 s pulls", [RepScript(pull_s=5.0)] * 3),
                           ("no-pause touch-and-go 1.2 / 1.0 s", _no_pause_touch_and_go(1.2, 1.0))):
        errors = []
        cues = []
        expected = 0
        for body_name, body in BODIES.items():
            for seed in range(3):
                sim = simulate(Scenario(athlete=body, reps=scripts, bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed))
                features = _analyse(sim, _half_rate(), _noise("ar", 0.015, "half_rate", label, body_name, seed))
                errors += _top_errors(sim, features)
                cues += _cued(features)
                expected += len(sim.reps)
        print(f"15 Hz detector, every other frame predicted, {label} (5 bodies x seeds 0-2): {_summary(errors)}; "
              f"{_cue_summary(cues)}{_not_counted(expected, cues)}")
    # The wrist proxy with the wrists hidden around the top; a soft lockout's recall, hidden and seen.
    held = [RepScript(pull_s=1.2)] * 3
    soft = [RepScript(pull_s=1.2, lockout_deficit_deg=SOFT_LOCKOUT_DEG)] * 3
    for label, scripts, start_s, end_s in (("held 0.6 s", held, -0.1, 0.3), ("held 0.6 s", held, -0.2, 0.8),
                                           ("no-pause touch-and-go 1.2 / 1.0 s", _no_pause_touch_and_go(1.2, 1.0), -0.2, 0.2),
                                           (f"{SOFT_LOCKOUT_DEG:.0f} deg soft lockout held 0.6 s", soft, -0.2, 0.8),
                                           (f"{SOFT_LOCKOUT_DEG:.0f} deg soft lockout held 0.6 s", soft, 0.0, 0.0)):
        errors = []
        cues = []
        expected = 0
        for body_name, body in BODIES.items():
            for seed in range(3):
                sim = simulate(Scenario(athlete=body, reps=scripts, track_bar=False, seed=seed))
                hidden = _wrists_hidden([(rep.top_time + start_s, rep.top_time + end_s) for rep in sim.reps])
                features = _analyse(sim, hidden, _noise("ar", 0.015, "wrists_hidden", label, start_s, body_name, seed))
                errors += _top_errors(sim, features)
                cues += _cued(features)
                expected += len(sim.reps)
        where = f"wrists hidden from {start_s:+.1f} s to {end_s:+.1f} s of the top" if end_s > start_s else "wrists seen"
        print(f"proxy {label}, {where} (5 bodies x seeds 0-2): {_summary(errors)}; {_cue_summary(cues)}"
              f"{_not_counted(expected, cues)}")


def sweep_counting() -> None:
    """Rep counting: touch-and-go under bar noise, long sets, fast touch-and-go on the wrist proxy."""
    for bar_noise_m in (0.002, 0.003, 0.004, 0.005):
        exact = 0
        for seed in range(25):
            rng = _rng("counting", bar_noise_m, seed)
            pull_s, lower_s = float(rng.uniform(0.6, 2.0)), float(rng.uniform(0.5, 1.5))
            sim = simulate(Scenario(reps=[RepScript(pull_s=pull_s, lower_s=lower_s, floor_hold_s=0.0)] * 4
                                    + [RepScript(pull_s=pull_s, lower_s=lower_s)], bar_noise_m=bar_noise_m, seed=seed))
            exact += len(_analyse(sim)) == len(sim.reps)
        print(f"5-rep touch-and-go sets, {bar_noise_m * 1000:.0f} mm bar noise, 25 random tempos (pull 0.6-2 s): {exact}/25 exact")
    exact = 0
    for seed in range(10):
        sim = simulate(Scenario(reps=[RepScript(floor_hold_s=0.0)] * 9 + [RepScript()], bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed))
        exact += len(_analyse(sim, _noise("ar", 0.02, "counting10", seed))) == len(sim.reps)
    print(f"10-rep touch-and-go sets, 3 mm bar, 2 cm AR, seeds 0-9: {exact}/10 exact")
    for pull_s in (0.5, 0.6, 0.8):
        for model, sigma_m in (("ar", 0.02), ("iid", 0.015)):
            exact = lost = 0
            for seed in range(10):
                reps = [RepScript(floor_hold_s=0.0, pull_s=pull_s, lower_s=0.4, top_hold_s=0.2)] * 5 + [RepScript(pull_s=pull_s, lower_s=0.4)]
                sim = simulate(Scenario(track_bar=False, reps=reps, seed=seed))
                counted = len(_analyse(sim, _noise(model, sigma_m, "fast_proxy", pull_s, model, seed)))
                exact += counted == len(sim.reps)
                lost += len(sim.reps) - counted
            print(f"proxy fast touch-and-go, {pull_s} s pulls, {sigma_m * 100:.1f} cm {model}, seeds 0-9: {exact}/10 sets exact, "
                  f"{lost} of 60 reps lost")


def sweep_d2() -> None:
    """D2 (hips shoot) against a held back angle and a scripted hips-first pull, seeds 0-29."""
    for name, setup_deg, knee_pass_deg in (("held back angle 60->60", 60.0, 60.0), ("hips first 45->55", 45.0, 55.0)):
        for model in ("iid", "ar"):
            cued = reps = 0
            for seed in range(30):
                sim = simulate(Scenario(reps=[RepScript(setup_trunk_deg=setup_deg, knee_pass_trunk_deg=knee_pass_deg)] * 3,
                                        bar_noise_m=TRACKED_BAR_NOISE_M, seed=seed))
                features = _analyse(sim, _noise(model, 0.02, "d2", name, seed))
                reps += len(features)
                cued += sum("deadlift_hips_shoot" in rep for rep in _cued(features))
            print(f"D2, {name}, 2 cm {model}: cued {cued} of {reps}")


def sweep_compute() -> None:
    """Analyser time per frame (this machine), 1.5 cm i.i.d. keypoint noise; the first set warms numpy up."""
    for label, scripts, track_bar in (("default pulls", [RepScript()] * 5, True), ("default pulls", [RepScript()] * 5, False),
                                      ("5 s pulls, 2 s holds", [RepScript(pull_s=5.0, top_hold_s=2.0)] * 3, True),
                                      ("5 s pulls, 2 s holds", [RepScript(pull_s=5.0, top_hold_s=2.0)] * 3, False)):
        sim = simulate(Scenario(reps=scripts, track_bar=track_bar, bar_noise_m=TRACKED_BAR_NOISE_M if track_bar else 0.0))
        noise = _noise("iid", 0.015, "compute", label, track_bar)
        frames = [noise(frame) for frame in sim.frames]
        per_frame: list[float] = []
        completing: list[float] = []
        for run in range(3):
            analyzer = DeadliftRepAnalyzer(BiomechanicsConfig().deadlift)
            analyzer.set_gravity(sim.gravity_up_world, GRAVITY_SOURCE_MEASURED)
            for frame in frames:
                start = time.perf_counter()
                analyzer.observe(DeadliftFrameInput(frame.timestamp, frame.frame_index, frame.points, frame.confidences, frame.bar))
                completed = False
                while analyzer.take_completed_rep() is not None:
                    analyzer.finish_rep()
                    completed = True
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                if run > 0:
                    (completing if completed else per_frame).append(elapsed_ms)
        print(f"{label}, {'tracked' if track_bar else 'proxy'}: median {np.median(per_frame):.2f} ms, p95 "
              f"{np.percentile(per_frame, 95):.2f} ms per frame; frames completing a rep {np.median(completing):.1f} ms "
              f"median, {max(completing):.1f} ms max")


SWEEPS = {
    "events": sweep_events, "counting": sweep_counting, "clean": sweep_clean, "no_pause": sweep_no_pause,
    "lockout": sweep_lockout, "tops": sweep_tops, "shrug": sweep_shrug, "hip_shift": sweep_hip_shift, "d2": sweep_d2,
    "gaps": sweep_gaps, "compute": sweep_compute,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweeps", nargs="+", choices=[*SWEEPS, "all"])
    parser.add_argument("--salt", type=int, default=0, help="draw every row's noise afresh")
    args = parser.parse_args()
    SALT["value"] = args.salt
    for name in SWEEPS if "all" in args.sweeps else args.sweeps:
        print(f"\n== {name}: {SWEEPS[name].__doc__}", flush=True)
        SWEEPS[name]()


if __name__ == "__main__":
    main()
