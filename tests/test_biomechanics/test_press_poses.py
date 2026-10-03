"""
Synthetic standing-press poses in the production frame (docs/exercises/ohp.md §7.2),
shared by the press tests, plus a pipeline-free analyser harness. The builder's
own geometry is checked here.

Frame: Y-down metres, X = subject's left, forward = -Z, ankles on y = 0.
Proportions of the 188.5 cm user: torso 0.543, femur 0.462, tibia 0.464,
upper arm 0.35, forearm 0.275, shoulder width 0.38.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from biomechanics.analysis.press_features import PressRepAnalyzer, PressSessionReference
from biomechanics.config import BiomechanicsConfig
from biomechanics.faults.hip_position_counter import SignalRepCounter
from biomechanics.faults.observability import SINGLE_CAMERA, TRIANGULATED
from biomechanics.faults.rule_engine import RuleEngine
from biomechanics.profiles.overhead_press import OverheadPressProfile
from biomechanics.utils.frame_context import GRAVITY_BODY, FrameContext
from biomechanics.utils.geometry import WORLD_UP, height_above, level_camera_frame
from biomechanics.utils.segment_lengths import BodyProportions
from biomechanics.utils.types import CocoKeypoints as CK
from biomechanics.utils.types import FaultSeverity, JointAngles, Skeleton3D

TORSO_M = 0.543
FEMUR_M = 0.462
TIBIA_M = 0.464
UPPER_ARM_M = 0.35
FOREARM_M = 0.275
SHOULDER_HALF_WIDTH_M = 0.19
HIP_HALF_WIDTH_M = 0.12
GRIP_HALF_WIDTH_M = 0.275
TOE_FORWARD_M = 0.17
HEEL_BACK_M = 0.06
MIDFOOT_Z_M = -0.35 * TOE_FORWARD_M
# Hips and trunk stand over mid-foot, so a clean lockout has the bar over both.
BODY_FORWARD_M = -MIDFOOT_Z_M
NUM_KEYPOINTS = 21
LEG_M = FEMUR_M + TIBIA_M
ARM_REF_M = 1.15 * TORSO_M
# Front rack: wrists just above the shoulder keypoints, bar on the front delts.
RACK_BAR_HEIGHT_M = 0.04
RACK_BAR_AHEAD_OF_SHOULDER_M = 0.09
HANGING_WRIST_DROP_M = 0.60
CONFIDENCE = 0.9
MIN_PROGRESS_SPAN_M = 0.05

FPS_MONO = 12.0
FPS_RIG = 30.0
LATCH_FRAMES = 5
SEVERITY_ORDER = (FaultSeverity.NONE, FaultSeverity.MILD, FaultSeverity.MODERATE, FaultSeverity.SEVERE)


def _rotation_x(angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    return np.array([[1.0, 0.0, 0.0], [0.0, math.cos(angle), -math.sin(angle)], [0.0, math.sin(angle), math.cos(angle)]])


def _rotation_y(angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    return np.array([[math.cos(angle), 0.0, math.sin(angle)], [0.0, 1.0, 0.0], [-math.sin(angle), 0.0, math.cos(angle)]])


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def lockout_bar_height(elbow_flex_deg: float = 0.0) -> float:
    """Bar height above the shoulder keypoints at lockout with this elbow bend."""
    chord_sq = UPPER_ARM_M ** 2 + FOREARM_M ** 2 + 2.0 * UPPER_ARM_M * FOREARM_M * math.cos(math.radians(elbow_flex_deg))
    lateral = GRIP_HALF_WIDTH_M - SHOULDER_HALF_WIDTH_M
    return math.sqrt(chord_sq - lateral ** 2)


def _triangle_elbow(shoulder: np.ndarray, wrist: np.ndarray, flex_deg: float, hint: np.ndarray) -> np.ndarray:
    chord = wrist - shoulder
    length = float(np.linalg.norm(chord))
    interior = math.radians(180.0 - flex_deg)
    scale = length / math.sqrt(UPPER_ARM_M ** 2 + FOREARM_M ** 2 - 2.0 * UPPER_ARM_M * FOREARM_M * math.cos(interior))
    upper, fore = scale * UPPER_ARM_M, scale * FOREARM_M
    cos_beta = float(np.clip((upper ** 2 + length ** 2 - fore ** 2) / (2.0 * upper * length), -1.0, 1.0))
    along = chord / length
    perpendicular = _unit(hint - np.dot(hint, along) * along)
    return shoulder + upper * (cos_beta * along + math.sqrt(1.0 - cos_beta ** 2) * perpendicular)


def _head(points: np.ndarray) -> None:
    head = (points[CK.LEFT_SHOULDER] + points[CK.RIGHT_SHOULDER]) / 2.0 + np.array([0.0, -0.22, -0.03])
    points[CK.NOSE] = head
    points[CK.LEFT_EYE] = head + [0.03, -0.02, 0.0]
    points[CK.RIGHT_EYE] = head + [-0.03, -0.02, 0.0]
    points[CK.LEFT_EAR] = head + [0.07, 0.0, 0.05]
    points[CK.RIGHT_EAR] = head + [-0.07, 0.0, 0.05]


def _lower_body(points: np.ndarray, hip_drop_m: float, hip_forward_m: float) -> np.ndarray:
    # Hip drop d = L(1 - cos α): shank forward and thigh back by α, hip over the ankle.
    alpha = math.acos(1.0 - hip_drop_m / LEG_M)
    forward_m = hip_forward_m + BODY_FORWARD_M
    knee = np.array([0.0, -TIBIA_M * math.cos(alpha), -TIBIA_M * math.sin(alpha)])
    hip = knee + np.array([0.0, -FEMUR_M * math.cos(alpha), FEMUR_M * math.sin(alpha) - forward_m])
    knee[2] -= forward_m / 2.0
    for sign, hip_i, knee_i, ankle_i, toe_i, heel_i in (
        (1.0, CK.LEFT_HIP, CK.LEFT_KNEE, CK.LEFT_ANKLE, CK.LEFT_FOOT_INDEX, CK.LEFT_HEEL),
        (-1.0, CK.RIGHT_HIP, CK.RIGHT_KNEE, CK.RIGHT_ANKLE, CK.RIGHT_FOOT_INDEX, CK.RIGHT_HEEL),
    ):
        x = sign * HIP_HALF_WIDTH_M
        points[ankle_i] = [x, 0.0, 0.0]
        points[knee_i] = [x, knee[1], knee[2]]
        points[hip_i] = [x, hip[1], hip[2]]
        points[toe_i] = [x + sign * 0.03, 0.0, -TOE_FORWARD_M]
        points[heel_i] = [x, 0.0, HEEL_BACK_M]
    return np.array([0.0, hip[1], hip[2]])


def _whole_body(points: np.ndarray, yaw_deg: float, gravity_tilt_deg: float) -> np.ndarray:
    hip_mid = (points[CK.LEFT_HIP] + points[CK.RIGHT_HIP]) / 2.0
    points = (points - hip_mid) @ _rotation_y(yaw_deg).T + hip_mid
    # The world frame sits gravity_tilt_deg off true gravity (the rig's body-axis up).
    return points @ _rotation_x(gravity_tilt_deg).T


def world_press_points(
    bar_height_m: float = RACK_BAR_HEIGHT_M,
    elbow_flex_deg: float = 0.0,
    layback_deg: float = 0.0,
    hip_drop_m: float = 0.0,
    tilt_deg: float = 0.0,
    grip_tilt_deg: float = 0.0,
    bar_forward_m: float | None = None,
    yaw_deg: float = 0.0,
    shoulder_lift_m: float = 0.0,
    gravity_tilt_deg: float = 0.0,
    hip_forward_m: float = 0.0,
    bar_yaw_m: float = 0.0,
) -> np.ndarray:
    """(21, 3) standing press pose.

    bar_height_m: wrist midpoint above the shoulder keypoints (before shoulder_lift_m
    raises the shoulder keypoints alone). elbow_flex_deg: elbow bend at the top.
    tilt_deg > 0: left hand higher. grip_tilt_deg > 0: wrists outside the elbows in
    the rack. bar_forward_m: bar ahead of mid-foot (default: from in front of the
    shoulders in the rack to over mid-foot at the top). bar_yaw_m: left hand this much
    further forward than the right. gravity_tilt_deg: the world frame's tilt from gravity.
    """
    points = np.zeros((NUM_KEYPOINTS, 3))
    hip_mid = _lower_body(points, hip_drop_m, hip_forward_m)
    layback = math.radians(layback_deg)
    shoulder_mid = hip_mid + TORSO_M * np.array([0.0, -math.cos(layback), math.sin(layback)])
    points[CK.LEFT_SHOULDER] = shoulder_mid + [SHOULDER_HALF_WIDTH_M, 0.0, 0.0]
    points[CK.RIGHT_SHOULDER] = shoulder_mid - [SHOULDER_HALF_WIDTH_M, 0.0, 0.0]

    top_m = lockout_bar_height(elbow_flex_deg)
    progress = float(np.clip((bar_height_m - RACK_BAR_HEIGHT_M) / max(top_m - RACK_BAR_HEIGHT_M, MIN_PROGRESS_SPAN_M), 0.0, 1.0))
    if bar_forward_m is None:
        rack_forward_m = MIDFOOT_Z_M - (shoulder_mid[2] - RACK_BAR_AHEAD_OF_SHOULDER_M)
        bar_forward_m = (1.0 - progress) * rack_forward_m
    wrist_mid = np.array([0.0, shoulder_mid[1] - bar_height_m, MIDFOOT_Z_M - bar_forward_m])
    lift = GRIP_HALF_WIDTH_M * math.tan(math.radians(tilt_deg))
    for sign, shoulder_i, elbow_i, wrist_i in (
        (1.0, CK.LEFT_SHOULDER, CK.LEFT_ELBOW, CK.LEFT_WRIST),
        (-1.0, CK.RIGHT_SHOULDER, CK.RIGHT_ELBOW, CK.RIGHT_WRIST),
    ):
        wrist = wrist_mid + [sign * GRIP_HALF_WIDTH_M, -sign * lift, -sign * bar_yaw_m / 2.0]
        outward = np.array([sign, 0.0, 0.0])
        grip = math.radians(grip_tilt_deg)
        rack_elbow = wrist - FOREARM_M * (math.sin(grip) * outward + math.cos(grip) * WORLD_UP)
        top_elbow = _triangle_elbow(points[shoulder_i], wrist, elbow_flex_deg, outward + np.array([0.0, 0.0, -0.3]))
        points[wrist_i] = wrist
        points[elbow_i] = (1.0 - progress) * rack_elbow + progress * top_elbow
    points[CK.LEFT_SHOULDER, 1] -= shoulder_lift_m
    points[CK.RIGHT_SHOULDER, 1] -= shoulder_lift_m
    _head(points)
    return _whole_body(points, yaw_deg, gravity_tilt_deg)


def world_hands_down_points(yaw_deg: float = 0.0) -> np.ndarray:
    """Standing, arms hanging at the sides (no bar): the rep signal is below the rack floor."""
    points = np.zeros((NUM_KEYPOINTS, 3))
    hip_mid = _lower_body(points, 0.0, 0.0)
    shoulder_mid = hip_mid + np.array([0.0, -TORSO_M, 0.0])
    for sign, shoulder_i, elbow_i, wrist_i in (
        (1.0, CK.LEFT_SHOULDER, CK.LEFT_ELBOW, CK.LEFT_WRIST),
        (-1.0, CK.RIGHT_SHOULDER, CK.RIGHT_ELBOW, CK.RIGHT_WRIST),
    ):
        shoulder = shoulder_mid + [sign * SHOULDER_HALF_WIDTH_M, 0.0, 0.0]
        points[shoulder_i] = shoulder
        points[elbow_i] = shoulder + [sign * 0.03, UPPER_ARM_M, 0.0]
        points[wrist_i] = shoulder + [sign * 0.04, HANGING_WRIST_DROP_M, 0.0]
    _head(points)
    return _whole_body(points, yaw_deg, 0.0)


def camera_pitched(points: np.ndarray, pitch_deg: float) -> np.ndarray:
    """What a camera pitched by pitch_deg (+ = lens up) sees; level_camera_frame undoes it."""
    return np.asarray(points) @ _rotation_x(pitch_deg)


def tilted_up(gravity_tilt_deg: float) -> np.ndarray:
    """True physical up in a world frame tilted gravity_tilt_deg from gravity."""
    return _rotation_x(gravity_tilt_deg) @ WORLD_UP


# ----------------------------------------------------------------------------
# Rep plans: per-frame builder arguments
# ----------------------------------------------------------------------------

HANDS_DOWN = "hands_down"
REACH = "reach"


def hold(seconds: float, fps: float, **pose) -> list[dict]:
    return [dict(pose) for _ in range(int(round(seconds * fps)))]


def hands_down(seconds: float, fps: float) -> list[dict]:
    return [{"pose": HANDS_DOWN} for _ in range(int(round(seconds * fps)))]


def press_rep(
    fps: float,
    hold_s: float = 1.0,
    press_s: float = 0.8,
    lock_s: float = 0.4,
    lower_s: float = 1.0,
    top_m: float | None = None,
    end_progress: float = 0.0,
    start_progress: float = 0.0,
    dip_m: float = 0.0,
    dip_s: float = 0.3,
    lockout_flex_deg: float = 0.0,
    tilt_top_deg: float = 0.0,
    layback_top_deg: float = 0.0,
    bar_forward_top_m: float | None = None,
    bar_yaw_top_m: float = 0.0,
    shoulder_lift_top_m: float = 0.0,
    hip_forward_m: float = 0.0,
    **constant,
) -> list[dict]:
    """One rep: rack hold, cosine press, lockout hold, cosine lower to end_progress.
    A leg-drive dip lowers the hips over the hold's last dip_s and drives them back
    up over the press's first dip_s; *_top faults ramp in with the bar's progress,
    and any other builder argument (constant) holds for the whole rep."""
    top = lockout_bar_height(lockout_flex_deg) if top_m is None else top_m
    hold_n, press_n = int(round(hold_s * fps)), int(round(press_s * fps))
    lock_n, lower_n = int(round(lock_s * fps)), int(round(lower_s * fps))
    dip_n = int(round(dip_s * fps))
    frames: list[dict] = []

    def frame(progress: float, hip_drop_m: float) -> dict:
        pose = dict(
            bar_height_m=RACK_BAR_HEIGHT_M + progress * (top - RACK_BAR_HEIGHT_M),
            elbow_flex_deg=lockout_flex_deg * progress,
            tilt_deg=tilt_top_deg * progress,
            layback_deg=layback_top_deg * progress,
            bar_yaw_m=bar_yaw_top_m * progress,
            shoulder_lift_m=shoulder_lift_top_m * progress,
            hip_drop_m=hip_drop_m,
            hip_forward_m=hip_forward_m * min(1.0, progress / 0.25),
        )
        # Constant pose arguments hold for the whole rep, overriding the ramps.
        pose.update(constant)
        if bar_forward_top_m is not None:
            pose["bar_forward_m"] = bar_forward_top_m * progress + (1.0 - progress) * (
                MIDFOOT_Z_M + RACK_BAR_AHEAD_OF_SHOULDER_M
            )
        return pose

    for i in range(hold_n):
        into_dip = i - (hold_n - dip_n)
        drop = dip_m * 0.5 * (1.0 - math.cos(math.pi * (into_dip + 1) / dip_n)) if into_dip >= 0 and dip_n else 0.0
        frames.append(frame(start_progress, drop))
    for i in range(press_n):
        progress = start_progress + (1.0 - start_progress) * 0.5 * (1.0 - math.cos(math.pi * (i + 1) / press_n))
        drop = dip_m * 0.5 * (1.0 + math.cos(math.pi * (i + 1) / dip_n)) if dip_n and i < dip_n else 0.0
        frames.append(frame(progress, drop))
    frames += [frame(1.0, 0.0) for _ in range(lock_n)]
    for i in range(lower_n):
        progress = 1.0 - (1.0 - end_progress) * 0.5 * (1.0 - math.cos(math.pi * (i + 1) / lower_n))
        frames.append(frame(progress, 0.0))
    return frames


def points_for(pose: dict) -> np.ndarray:
    pose = dict(pose)
    kind = pose.pop("pose", None)
    if kind == HANDS_DOWN:
        return world_hands_down_points(yaw_deg=pose.get("yaw_deg", 0.0))
    if kind == REACH:
        progress = pose["progress"]
        return (1.0 - progress) * world_hands_down_points() + progress * world_press_points(lockout_bar_height())
    return world_press_points(**pose)


def wrists_clipped_above(height_m: float):
    """Confidence function: both wrists unseen while the bar is above height_m (frame top)."""

    def confidence(points: np.ndarray) -> np.ndarray:
        confidences = np.full(len(points), CONFIDENCE)
        shoulder_mid = (points[CK.LEFT_SHOULDER] + points[CK.RIGHT_SHOULDER]) / 2.0
        wrist_mid = (points[CK.LEFT_WRIST] + points[CK.RIGHT_WRIST]) / 2.0
        if height_above(wrist_mid, shoulder_mid) > height_m:
            confidences[[CK.LEFT_WRIST, CK.RIGHT_WRIST]] = 0.0
        return confidences

    return confidence


def bare_hand_reach(fps: float, seconds: float = 0.6, top_s: float = 0.3) -> list[dict]:
    """Arms swing from the sides straight overhead and back, no bar and no rack hold."""
    frames = int(round(seconds * fps))
    up = [{"pose": REACH, "progress": (i + 1) / frames} for i in range(frames)]
    return up + [{"pose": REACH, "progress": 1.0}] * int(round(top_s * fps)) + up[::-1]


def hip_centred(points: np.ndarray) -> np.ndarray:
    return points - (points[CK.LEFT_HIP] + points[CK.RIGHT_HIP]) / 2.0


# ----------------------------------------------------------------------------
# Pipeline-free harness: analyser + counter + rules in the pipeline's order
# ----------------------------------------------------------------------------


class AnalyzerHarness:
    """Drives PressRepAnalyzer like BiomechanicsPipeline does, without pre-IK/IK:
    rack predicate until LATCH_FRAMES consecutive passes, then observe -> counter
    -> finish/reject/discard -> on_rep_start, rules and recorded-only stamping."""

    def __init__(
        self,
        config: BiomechanicsConfig | None = None,
        capture_mode: str = SINGLE_CAMERA,
        pitch_certified: bool = False,
        mount_pitch_deg: float = math.nan,
        up: np.ndarray = WORLD_UP,
        gravity_source: str = GRAVITY_BODY,
        fps: float = FPS_MONO,
        body_proportions: BodyProportions | None = None,
    ) -> None:
        self.config = config or BiomechanicsConfig()
        self.profile = OverheadPressProfile()
        self.engine = RuleEngine(
            config=self.config, rules=self.profile.create_fault_rules(self.config),
            capture_mode=capture_mode, reference=PressSessionReference(),
        )
        self.analyzer: PressRepAnalyzer = self.profile.create_rep_analyzer(self.config)
        self.counter = SignalRepCounter(config=self.config.faults.press.counter)
        self.capture_mode = capture_mode
        self.pitch_certified = pitch_certified
        self.mount_pitch_deg = mount_pitch_deg
        self.up = np.asarray(up, dtype=np.float64)
        self.gravity_source = gravity_source
        self.body_proportions = body_proportions
        self.dt = 1.0 / fps
        self.time = 0.0
        self.latched = False
        self._passes = 0
        self.features: list = []
        self.counted: list = []
        self.rejections: list[tuple[str, object]] = []
        self.faults: list = []
        self.rejected_faults: list = []
        self.latch_time = math.nan
        self.starts: list[bool] = []

    def context(self, points: np.ndarray, confidences: np.ndarray | None = None, raw: bool = False) -> FrameContext:
        confidences = np.full(len(points), CONFIDENCE) if confidences is None else confidences
        skeleton = Skeleton3D.from_numpy(hip_centred(points), confidences=confidences, timestamp=self.time)
        return FrameContext(
            skeleton_3d=skeleton,
            angles=None if raw else JointAngles(timestamp=self.time),
            timestamp=self.time,
            phase=self.counter.phase,
            in_rep=self.counter.in_rep,
            capture_mode=self.capture_mode,
            up=self.up,
            gravity_source=self.gravity_source,
            pitch_certified=self.pitch_certified,
            mount_source="inclinometer" if self.pitch_certified else "none",
            mount_pitch_deg=self.mount_pitch_deg,
            raw_confidences=confidences,
            body_proportions=self.body_proportions,
            skeleton_world=Skeleton3D.from_numpy(points, confidences=confidences) if self.capture_mode == TRIANGULATED else None,
        )

    def step(self, points: np.ndarray, confidences: np.ndarray | None = None) -> None:
        if not self.latched:
            ready = self.analyzer.is_ready(self.context(points, confidences, raw=True))
            self._passes = self._passes + 1 if ready else 0
            if self._passes >= LATCH_FRAMES:
                self.latched = True
                self.latch_time = self.time
                self.counter.seed_baseline(self.analyzer.counter_baseline())
        if self.latched:
            self._analyse(points, confidences)
        self.time += self.dt

    def _analyse(self, points: np.ndarray, confidences: np.ndarray | None) -> None:
        ctx = self.context(points, confidences)
        self.analyzer.observe(ctx)
        was_in_rep = self.counter.in_rep
        angles = JointAngles(timestamp=self.time)
        rep_data, feedback = self.counter.update(signal_value=self.analyzer.rep_signal, timestamp=self.time)
        if rep_data is not None:
            features = self.analyzer.finish_rep(rep_data.rep_number)
            self.features.append(features)
            reason = self.analyzer.rejection_reason(features)
            if reason is None:
                faults = self.engine.finish_rep(angles, rep_data.rep_number, features)
                self._stamp(faults, ctx)
                self.counted.append(features)
                self.faults.extend(faults)
            else:
                self.counter.reject_last_rep()
                self.engine.discard_rep()
                self.analyzer.discard_rep()
                self.rejections.append((reason, features))
                rejected = [f for f in self.profile.on_rep_rejected(reason, features, angles) if self.engine.tag_observability(f)]
                self.rejected_faults.extend(rejected)
        elif feedback == "go_deeper":
            self.engine.discard_rep()
            self.analyzer.discard_rep()
        elif was_in_rep and (not self.counter.in_rep or self.counter.rep_started):
            self.analyzer.discard_rep()
        if self.counter.rep_started:
            self.starts.append(was_in_rep)
            self.analyzer.on_rep_start(piston=was_in_rep)

    def _stamp(self, faults: list, ctx: FrameContext) -> None:
        for fault in faults:
            start_tier = self.profile.cue_start_tier(fault.fault_type, ctx)
            if SEVERITY_ORDER.index(fault.severity) < SEVERITY_ORDER.index(start_tier):
                fault.details["recorded_only"] = True

    def run(self, poses: list[dict], transform=None, noise=None, confidence=None) -> AnalyzerHarness:
        """transform(points) -> points, noise(shape) -> offsets, confidence(points) -> per-keypoint confidences."""
        for pose in poses:
            points = points_for(pose)
            if transform is not None:
                points = transform(points)
            if noise is not None:
                points = points + noise(points.shape)
            self.step(points, confidence(points) if confidence is not None else None)
        return self

    def reset_set(self) -> None:
        """The pipeline's set boundary: the gate re-arms, the counter resets."""
        self.latched = False
        self._passes = 0
        self.analyzer.reset_set()
        self.engine.reset()
        self.counter.reset()


# ----------------------------------------------------------------------------
# Real-pipeline sessions (conftest.SyntheticSession) on the press profile
# ----------------------------------------------------------------------------


def press_session(monkeypatch, multi_camera: bool = False, config: BiomechanicsConfig | None = None):
    """A real BiomechanicsPipeline on the press, coaching-ready for the test."""
    from conftest import SyntheticSession

    monkeypatch.setattr(OverheadPressProfile, "coaching_ready", True)
    return SyntheticSession(monkeypatch, multi_camera=multi_camera, config=config, exercise_name="Barbell Overhead Press")


def run_session(session, poses: list[dict], fps: float, confidence=None, transform=None) -> list:
    results = []
    for pose in poses:
        points = points_for(pose)
        if transform is not None:
            points = transform(points)
        results.append(session.step(points, confidence(points) if confidence is not None else None, fps=fps))
    return results


def counted_reps(results: list) -> list:
    return [result.rep_data for result in results if result.rep_data is not None]


def faults_of(results: list, fault_type: str | None = None) -> list:
    return [fault for result in results for fault in result.faults if fault_type in (None, fault.fault_type)]


class TestBuilderGeometry:
    def test_rack_pose_sits_in_the_rack_band(self):
        points = world_press_points()
        shoulder_mid = (points[CK.LEFT_SHOULDER] + points[CK.RIGHT_SHOULDER]) / 2.0
        wrist_mid = (points[CK.LEFT_WRIST] + points[CK.RIGHT_WRIST]) / 2.0
        assert height_above(wrist_mid, shoulder_mid) == pytest.approx(RACK_BAR_HEIGHT_M, abs=1e-9)
        assert height_above(points[CK.LEFT_WRIST], points[CK.LEFT_ELBOW]) == pytest.approx(FOREARM_M, abs=1e-9)

    def test_lockout_reaches_with_true_segment_lengths(self):
        points = world_press_points(bar_height_m=lockout_bar_height(0.0))
        upper = np.linalg.norm(points[CK.LEFT_ELBOW] - points[CK.LEFT_SHOULDER])
        fore = np.linalg.norm(points[CK.LEFT_WRIST] - points[CK.LEFT_ELBOW])
        assert upper == pytest.approx(UPPER_ARM_M, abs=1e-6)
        assert fore == pytest.approx(FOREARM_M, abs=1e-6)

    def test_bent_lockout_has_the_requested_elbow_bend(self):
        points = world_press_points(bar_height_m=lockout_bar_height(30.0), elbow_flex_deg=30.0)
        upper = points[CK.LEFT_SHOULDER] - points[CK.LEFT_ELBOW]
        fore = points[CK.LEFT_WRIST] - points[CK.LEFT_ELBOW]
        interior = math.degrees(math.acos(np.dot(upper, fore) / np.linalg.norm(upper) / np.linalg.norm(fore)))
        assert 180.0 - interior == pytest.approx(30.0, abs=1e-6)

    def test_hip_drop_and_layback_are_exact(self):
        upright = world_press_points()
        dipped = world_press_points(hip_drop_m=0.05, layback_deg=10.0)
        hip_up = (upright[CK.LEFT_HIP] + upright[CK.RIGHT_HIP]) / 2.0
        hip_down = (dipped[CK.LEFT_HIP] + dipped[CK.RIGHT_HIP]) / 2.0
        assert height_above(hip_up, hip_down) == pytest.approx(0.05, abs=1e-9)
        trunk = (dipped[CK.LEFT_SHOULDER] + dipped[CK.RIGHT_SHOULDER]) / 2.0 - hip_down
        assert math.degrees(math.atan2(trunk[2], -trunk[1])) == pytest.approx(10.0, abs=1e-9)

    def test_camera_pitch_is_undone_by_levelling(self):
        points = world_press_points(bar_height_m=0.3)
        assert np.allclose(level_camera_frame(camera_pitched(points, 15.0), 15.0), points)

    def test_tilted_world_keeps_true_up_on_the_upright_trunk(self):
        points = world_press_points(gravity_tilt_deg=3.0)
        hip_mid = (points[CK.LEFT_HIP] + points[CK.RIGHT_HIP]) / 2.0
        shoulder_mid = (points[CK.LEFT_SHOULDER] + points[CK.RIGHT_SHOULDER]) / 2.0
        trunk_axis = (shoulder_mid - hip_mid) / np.linalg.norm(shoulder_mid - hip_mid)
        assert float(np.dot(trunk_axis, tilted_up(3.0))) == pytest.approx(1.0, abs=1e-9)
        assert float(np.dot(trunk_axis, WORLD_UP)) == pytest.approx(math.cos(math.radians(3.0)), abs=1e-9)

    def test_rep_plan_frame_counts(self):
        frames = press_rep(FPS_MONO, hold_s=1.0, press_s=0.75, lock_s=0.5, lower_s=1.0)
        assert len(frames) == 12 + 9 + 6 + 12
        assert frames[0]["bar_height_m"] == pytest.approx(RACK_BAR_HEIGHT_M)
        assert frames[-1]["bar_height_m"] == pytest.approx(RACK_BAR_HEIGHT_M)
