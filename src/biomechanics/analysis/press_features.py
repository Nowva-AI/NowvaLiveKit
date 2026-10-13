"""
Standing barbell overhead press: per-frame samples, per-rep features, the rep
analyser that owns them and the press session reference (docs/exercises/ohp.md §3-§4).

Frame: Y-down metres (WORLD_UP = (0, -1, 0)), X = subject's left, forward = -Z.
h(p, q) = height of p above q = q_y - p_y. A certified single camera is levelled
first (geometry.level_camera_frame); rig sagittal metrics use FrameContext.up as is.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel

from biomechanics.config import PressCameraConfig, PressConfig
from biomechanics.diagnosis.lean_model import MIDFOOT_FRACTION_OF_ANKLE_TO_TOE
from biomechanics.faults.fault_types import FAULT_MESSAGES, FaultType
from biomechanics.faults.observability import APPROXIMATE, SINGLE_CAMERA, TRIANGULATED
from biomechanics.faults.session_reference import SessionReference
from biomechanics.utils.camera_mount import PitchCertification, PitchDeltaTripwire
from biomechanics.utils.frame_context import GRAVITY_BODY, GRAVITY_MEASURED, FrameContext
from biomechanics.utils.geometry import WORLD_UP, forward_offset, lean_back_deg, level_camera_frame, sagittal_forward
from biomechanics.utils.types import CocoKeypoints as CK
from biomechanics.utils.types import FaultEvent, FaultSeverity, JointAngles

NAN = float("nan")
CM_PER_M = 100.0
EXERCISE_NAME = "overhead_press"

# Single camera: image-plane X is the lateral axis; the subject's right is -X.
MONO_LATERAL_RIGHT = np.array([-1.0, 0.0, 0.0])
MIN_SEGMENT_M = 1e-6

# Mount pitch bands (ohp.md §4.1).
PITCH_BAND_LEVEL = "level"
PITCH_BAND_TILTED = "tilted"
PITCH_BAND_UNKNOWN = "unknown"

# Press phases as fault details name them; the counter's wire states are unchanged.
PHASE_SETUP = "setup"
PHASE_PRESS = "press"
PHASE_LOCKOUT = "lockout"

# Rejection reasons (ohp.md §3.5).
REJECT_NO_RACK_START = "no_rack_start"
REJECT_NO_LOCKOUT = "no_lockout"

# Session-reference drift metrics (lower is the better rep).
DRIFT_LAYBACK = "press_layback_deg"
DRIFT_BAR_FORWARD = "press_bar_forward"
DRIFT_LOCKOUT = "press_lockout_deg"
DRIFT_LEG_DRIVE = "press_leg_drive"
DRIFT_TILT = "press_tilt_abs"

# Rep windows (ohp.md §4.3), fractions of the rep's range of motion.
START_FRACTION_OF_ROM = 0.05
LOCKOUT_FRACTION_OF_ROM = 0.05
TILT_MIN_FRACTION_OF_ROM = 0.2
FACING_MAX_FRACTION_OF_ROM = 0.3
# Idle frames kept before a rep (the setup window), and what counts as nearly still.
SETUP_RING_S = 1.0
SETUP_STILL_CM_PER_FRAME = 2.0
# Dip window: from this long before the press start to this fraction of the press after it.
DIP_LEAD_S = 0.5
DIP_PRESS_FRACTION = 0.5
# Fewer samples than these in a window make its features NaN.
MIN_PRESS_SAMPLES = 4
MIN_LOCKOUT_SAMPLES = 2
MIN_SETUP_SAMPLES = 4
TOP_PERCENTILE = 95.0
WORST_PERCENTILE = 90.0
# A rep that never ends is bounded like the pipeline's own trajectory (~20 s at 30 fps).
MAX_REP_SAMPLES = 600
# Facing guard on the first set: biacromial width ~0.70 torso (Drillis & Contini);
# a reference below cos 30° of it means the lifter stood rotated all set.
SHOULDER_WIDTH_TO_TORSO_RATIO = 0.70
FIRST_SET_FACING_GUARD_RATIO = 0.87
# A gap this long between measured frames inside a rep: the wrists left the frame
# (3+ frames at 12 fps; a single dropped frame never reads as clipping).
CLIPPED_GAP_S = 0.3
# Two lockouts split by a dip of this fraction of the range: a turnaround above the chin.
PARTIAL_BOTTOM_DIP_FRACTION = 0.25


def _finite(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def _median(values) -> float:
    finite = _finite(values)
    return float(np.median(finite)) if finite.size else NAN


def _percentile(values, percentile: float) -> float:
    finite = _finite(values)
    return float(np.percentile(finite, percentile)) if finite.size else NAN


def _med3(values: np.ndarray) -> np.ndarray:
    """Centred 3-sample median over finite neighbours; NaN where none is finite."""
    padded = np.concatenate(([NAN], values, [NAN]))
    windows = np.stack([padded[:-2], padded[1:-1], padded[2:]])
    smoothed = np.full(len(values), NAN)
    finite = np.isfinite(windows)
    for i in range(len(values)):
        column = windows[finite[:, i], i]
        if column.size:
            smoothed[i] = float(np.median(column))
    return smoothed


def _mid(points: np.ndarray, left: int, right: int) -> np.ndarray:
    return (points[left] + points[right]) / 2.0


def _height(point: np.ndarray, reference: np.ndarray) -> float:
    return float(reference[1] - point[1])


def _flexion_deg(proximal: np.ndarray, joint: np.ndarray, distal: np.ndarray) -> float:
    first = proximal - joint
    second = distal - joint
    norms = float(np.linalg.norm(first) * np.linalg.norm(second))
    if norms < MIN_SEGMENT_M:
        return NAN
    cosine = float(np.clip(np.dot(first, second) / norms, -1.0, 1.0))
    return 180.0 - math.degrees(math.acos(cosine))


def _frontal(point: np.ndarray) -> np.ndarray:
    return np.array([point[0], point[1], 0.0])


def _standing_pitch_deg(points: np.ndarray) -> float:
    # The pipeline's ankle->shoulder tilt from the camera vertical; only ever used
    # as a delta from its own reading at certification (the tripwire).
    body_axis = _mid(points, CK.LEFT_SHOULDER, CK.RIGHT_SHOULDER) - _mid(points, CK.LEFT_ANKLE, CK.RIGHT_ANKLE)
    return math.degrees(math.atan2(float(body_axis[2]), float(-body_axis[1])))


def _lateral_right(points: np.ndarray, capture_mode: str) -> np.ndarray:
    if capture_mode == SINGLE_CAMERA:
        return MONO_LATERAL_RIGHT
    across = points[CK.RIGHT_HIP] - points[CK.LEFT_HIP]
    horizontal = np.array([across[0], 0.0, across[2]])
    norm = float(np.linalg.norm(horizontal))
    return horizontal / norm if norm > MIN_SEGMENT_M else MONO_LATERAL_RIGHT


def pitch_band(pitch_certified: bool, mount_pitch_deg: float, camera: PressCameraConfig) -> str:
    """level: certified within the level band; tilted: certified within the tilted
    band; unknown: uncertified, or pitched beyond the tilted band."""
    if not pitch_certified or not math.isfinite(mount_pitch_deg):
        return PITCH_BAND_UNKNOWN
    if abs(mount_pitch_deg) <= camera.level_band_deg:
        return PITCH_BAND_LEVEL
    if abs(mount_pitch_deg) <= camera.tilted_band_deg:
        return PITCH_BAND_TILTED
    return PITCH_BAND_UNKNOWN


def mono_band_observability_cap(features: PressRepFeatures, tilted_band_validated: bool) -> str | None:
    """Single-camera vertical/frontal verdicts outside a validated band are capped at
    approximate (recap-only, hedged); None leaves the observability table in charge."""
    if features.capture_mode != SINGLE_CAMERA or features.pitch_band == PITCH_BAND_LEVEL:
        return None
    if features.pitch_band == PITCH_BAND_TILTED and tilted_band_validated:
        return None
    return APPROXIMATE


@dataclass(frozen=True)
class PressFrameSample:
    """One measured frame. Heights in cm, angles in degrees, NaN when unmeasured."""
    timestamp: float
    bar_cm: float
    bar_above_hip_cm: float
    shoulder_above_hip_cm: float
    tilt_deg: float
    elbow_l_deg: float
    elbow_r_deg: float
    elbow2d_l_deg: float
    elbow2d_r_deg: float
    hip_cm: float
    knee_flex_deg: float
    layback_deg: float
    bar_fwd_shoulder_m: float
    grip_tilt_l_deg: float
    grip_tilt_r_deg: float
    shoulder_width_x: float
    torso_m: float
    leg_m: float
    trunk_lateral_deg: float
    standing_pitch_deg: float
    # Rig: the bar and mid-foot in the world frame, with this frame's up and forward.
    wrist_mid: np.ndarray
    midfoot: np.ndarray
    up: np.ndarray
    forward: np.ndarray


class PressRepFeatures(BaseModel):
    """Everything judged about one press. NaN = not measurable this rep.

    Signs: bar_tilt_deg > 0 means the left hand finished higher (the right lags);
    grip_tilt_deg > 0 means the wrists sat outside the elbows (grip too wide);
    layback_deg > 0 means the trunk leaned back; bar offsets > 0 mean ahead.
    """
    exercise: str = EXERCISE_NAME
    rep_number: int
    max_depth_angle: float = NAN
    capture_mode: str = SINGLE_CAMERA
    pitch_band: str = PITCH_BAND_UNKNOWN
    gravity_source: str = GRAVITY_BODY
    arm_ref_m: float = NAN
    leg_ref_m: float = NAN
    sample_count: int = 0
    rom_cm: float = NAN
    top_height_ratio: float = NAN
    rack_height_ratio: float = NAN
    lockout_elbow_l_deg: float = NAN
    lockout_elbow_r_deg: float = NAN
    leg_drive_ratio: float = NAN
    leg_drive_phase: str = ""
    knee_flexion_change_deg: float = NAN
    layback_deg: float = NAN
    layback_change_deg: float = NAN
    bar_over_midfoot_ratio: float = NAN
    bar_forward_shoulder_ratio: float = NAN
    bar_tilt_deg: float = NAN
    bar_tilt_phase: str = ""
    rack_tilt_deg: float = NAN
    grip_tilt_deg: float = NAN
    press_time_s: float = NAN
    lower_time_s: float = NAN
    concentric_time_s: float = NAN
    eccentric_time_s: float = NAN
    concentric_velocity_mps: float = NAN
    velocity_loss_pct: float = NAN
    highlight_score: float = NAN
    trunk_lateral_deg: float = NAN
    width_ratio: float = NAN
    facing_ok: bool = False
    wrists_clipped: bool = False
    clip_height_ratio: float = NAN
    partial_bottom: bool = False
    press_start_observed: bool = False
    rack_start: bool = False
    first_rep_after_latch: bool = False
    piston: bool = False

    def drift_metrics(self) -> dict[str, float]:
        bar_forward = (
            self.bar_over_midfoot_ratio if self.gravity_source == GRAVITY_MEASURED else self.bar_forward_shoulder_ratio
        )
        lockout = _finite([self.lockout_elbow_l_deg, self.lockout_elbow_r_deg])
        return {
            DRIFT_LAYBACK: self.layback_deg,
            DRIFT_BAR_FORWARD: bar_forward,
            DRIFT_LOCKOUT: float(lockout.max()) if lockout.size else NAN,
            DRIFT_LEG_DRIVE: self.leg_drive_ratio,
            DRIFT_TILT: abs(self.bar_tilt_deg),
        }


class PressSessionReference(SessionReference):
    """The athlete's best press per drift metric this session, and the set's
    velocities for velocity loss (annotate is inherited)."""

    def update(self, features: PressRepFeatures) -> None:
        for metric, value in features.drift_metrics().items():
            if math.isfinite(value) and (metric not in self._best or value < self._best[metric]):
                self._best[metric] = value
        velocity = features.concentric_velocity_mps
        if math.isfinite(velocity):
            self._set_velocities.append(velocity)
            if math.isnan(self._set_best_velocity_mps) or velocity > self._set_best_velocity_mps:
                self._set_best_velocity_mps = velocity


def missed_rep_event(features: PressRepFeatures, angles: JointAngles, min_travel_cm: float) -> FaultEvent | None:
    """press_missed for a press that travelled at least min_travel_cm and stalled short of lockout."""
    if not features.rom_cm >= min_travel_cm:
        return None
    severity = FaultSeverity.MODERATE
    return FaultEvent(
        fault_type=FaultType.PRESS_MISSED.value,
        severity=severity,
        severity_score=2.0,
        message=FAULT_MESSAGES[FaultType.PRESS_MISSED][severity.value],
        timestamp=angles.timestamp,
        frame_index=angles.frame_index,
        rep_number=features.rep_number,
        details={
            "side": None,
            "phase": PHASE_PRESS,
            "is_drift": False,
            "value": features.top_height_ratio,
            "unit": "arm_ratio",
            "rom_cm": features.rom_cm,
        },
    )


@dataclass
class _RackReference:
    """Medians of the frames the readiness gate latched on (ohp.md §3.2)."""
    samples: list[PressFrameSample]
    timestamp: float
    bar_cm: float
    hip_cm: float
    shoulder_above_hip_cm: float
    tilt_deg: float
    shoulder_width_x: float
    midfoot: np.ndarray


@dataclass
class _Commit:
    """List lengths before a rep's setup frames were pooled, to undo a rejected rep."""
    tilt_pool: int
    set_widths: int
    set_torsos: int
    set_legs: int


class PressRepAnalyzer:
    """Rack readiness, the rep signal and every per-rep press feature.

    Session state (survives reset_set): the facing reference of past sets and the
    certified-mount tripwire, which is the session's once bound
    (bind_pitch_certification), so its reference reading also survives
    exercise switches. Everything else is per set, including the set's
    rack-hold pitch readings: their mean is checked against the tripwire once
    the set has enough of them, and the first sets' means pool its reference.
    """

    def __init__(self, config: PressConfig) -> None:
        self.cfg = config
        self.rep_signal = NAN
        self._set_width_medians: list[float] = []
        self._tripwire = PitchDeltaTripwire(config.camera.tripwire_deg)
        self._tripped_delta_deg: float | None = None
        self._capture_mode = SINGLE_CAMERA
        self._pitch_band = PITCH_BAND_UNKNOWN
        self._pitch_certified = False
        self._gravity_source = GRAVITY_BODY
        self._body_torso_m = NAN
        self._body_leg_m = NAN
        self._reset_set_state()

    # ------------------------------------------------------------------
    # RepAnalyzer protocol
    # ------------------------------------------------------------------

    def is_ready(self, ctx: FrameContext) -> bool:
        """The front-rack predicate on a pre-latch frame; keeps the frames that pass."""
        # Called only before the latch: any reference frozen earlier was premature.
        self._rack = None
        self._note_context(ctx)
        points, raw_points, confidences = self._points(ctx)
        frame_torso_m = float(np.linalg.norm(
            _mid(points, CK.LEFT_SHOULDER, CK.RIGHT_SHOULDER) - _mid(points, CK.LEFT_HIP, CK.RIGHT_HIP)
        ))
        sample = self._sample(ctx, points, raw_points, confidences, frame_torso_m, None)
        if not self._rack_pose(points, confidences, frame_torso_m):
            self._ready_frames.clear()
            return False
        self._ready_frames.append(sample)
        while sample.timestamp - self._ready_frames[0].timestamp > self.cfg.rack.reference_window_s:
            self._ready_frames.popleft()
        return True

    def counter_baseline(self) -> float | None:
        if self._rack is None and self._ready_frames:
            self._freeze_rack()
        return self._rack.bar_cm if self._rack is not None else None

    def bind_pitch_certification(self, certification: PitchCertification, profile_name: str) -> None:
        self._tripwire = certification.tripwire(profile_name, self.cfg.camera.tripwire_deg)

    def pitch_tripwire_delta(self) -> float | None:
        delta_deg, self._tripped_delta_deg = self._tripped_delta_deg, None
        return delta_deg

    def observe(self, ctx: FrameContext) -> None:
        self._note_context(ctx)
        if self._rack is None and self._ready_frames:
            self._freeze_rack()
        points, raw_points, confidences = self._points(ctx)
        sample = self._sample(ctx, points, raw_points, confidences, self._torso_ref_m(), ctx.angles)
        self.rep_signal = sample.bar_cm
        self._track_re_rack(sample)
        self._ring.append(sample)
        while sample.timestamp - self._ring[0].timestamp > SETUP_RING_S:
            self._ring.popleft()
        if self._rep_open:
            self._rep_samples.append(sample)

    def on_rep_start(self, piston: bool) -> None:
        self._pending_commit = None
        self._preroll = list(self._ring)
        self._rep_samples = deque(maxlen=MAX_REP_SAMPLES)
        self._rep_open = True
        # The counter often ends a touch-and-go at the turnaround (the bar stops in
        # the lockout zone) and opens the next rep a frame later: a press that starts
        # within the setup window of a counted rep's end had no rack hold either.
        now = self._ring[-1].timestamp if self._ring else NAN
        quick_restart = self._last_rep_counted and now - self._last_finish_time <= self.cfg.acceptance.setup_window_s
        self._rep_piston = piston or quick_restart
        self._rep_continues_counted = self._rep_piston and self._last_rep_counted
        self._rep_first = self._first_rep_pending and not self._rep_piston
        self._first_rep_pending = False
        self._last_rep_counted = False

    def finish_rep(self, rep_number: int) -> PressRepFeatures:
        trajectory = self._preroll + list(self._rep_samples)
        features = self._features(rep_number, trajectory)
        self._last_finish_time = trajectory[-1].timestamp if trajectory else NAN
        self._rep_open = False
        self._last_rep_counted = True
        return features

    def rejection_reason(self, features: PressRepFeatures) -> str | None:
        acceptance = self.cfg.acceptance
        if not features.rack_start:
            return REJECT_NO_RACK_START
        reached = features.top_height_ratio >= acceptance.top_height_min_ratio
        waived = features.wrists_clipped and features.clip_height_ratio >= acceptance.clipped_wrists_min_ratio
        if not (reached or waived):
            return REJECT_NO_LOCKOUT
        return None

    def discard_rep(self) -> None:
        if self._rep_open:
            self._rep_open = False
            self._rep_samples = deque(maxlen=MAX_REP_SAMPLES)
            self._preroll = []
            if self._rep_first:
                self._first_rep_pending = True
        elif self._pending_commit is not None:
            commit = self._pending_commit
            del self._tilt_pool[commit.tilt_pool:]
            del self._set_widths[commit.set_widths:]
            del self._set_torsos[commit.set_torsos:]
            del self._set_legs[commit.set_legs:]
            self._pending_commit = None
        self._last_rep_counted = False

    def reset_set(self) -> None:
        set_width = _median(self._set_widths)
        if math.isfinite(set_width):
            self._set_width_medians.append(set_width)
        camera = self.cfg.camera
        if not math.isfinite(self._tripwire.reference_deg) and len(self._set_pitch_readings) >= camera.tripwire_min_frames:
            self._tripwire.pool_reference(float(np.mean(self._set_pitch_readings)), camera.tripwire_reference_sets)
        self._reset_set_state()

    # ------------------------------------------------------------------
    # Per-frame
    # ------------------------------------------------------------------

    def _reset_set_state(self) -> None:
        self.rep_signal = NAN
        self._ready_frames: deque[PressFrameSample] = deque(maxlen=self.cfg.rack.reference_frames)
        self._rack: _RackReference | None = None
        self._ring: deque[PressFrameSample] = deque()
        self._preroll: list[PressFrameSample] = []
        self._rep_samples: deque[PressFrameSample] = deque(maxlen=MAX_REP_SAMPLES)
        self._rep_open = False
        self._rep_piston = False
        self._rep_first = False
        self._rep_continues_counted = False
        self._first_rep_pending = False
        self._last_rep_counted = False
        self._tilt_pool: list[float] = []
        self._set_widths: list[float] = []
        self._set_torsos: list[float] = []
        self._set_legs: list[float] = []
        self._pending_commit: _Commit | None = None
        self._last_signal_time = NAN
        self._last_finish_time = NAN
        self._set_pitch_readings: list[float] = []
        self._set_pitch_checked = False

    def _note_context(self, ctx: FrameContext) -> None:
        self._capture_mode = ctx.capture_mode
        self._pitch_certified = ctx.pitch_certified
        self._pitch_band = pitch_band(ctx.pitch_certified, ctx.mount_pitch_deg, self.cfg.camera)
        self._gravity_source = ctx.gravity_source
        proportions = ctx.body_proportions
        if proportions is not None:
            self._body_torso_m = proportions.torso_length_avg
            self._body_leg_m = proportions.femur_length_avg + proportions.tibia_length_avg

    def _points(self, ctx: FrameContext) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raw_points = ctx.skeleton_3d.to_numpy()
        confidences = ctx.raw_confidences
        if confidences is None:
            confidences = np.array([keypoint.confidence for keypoint in ctx.skeleton_3d.keypoints])
        points = raw_points
        if ctx.capture_mode == SINGLE_CAMERA and ctx.pitch_certified and math.isfinite(ctx.mount_pitch_deg):
            points = level_camera_frame(raw_points, ctx.mount_pitch_deg)
        return points, raw_points, confidences

    def _sample(
        self,
        ctx: FrameContext,
        points: np.ndarray,
        raw_points: np.ndarray,
        confidences: np.ndarray,
        torso_ref_m: float,
        angles: JointAngles | None,
    ) -> PressFrameSample:
        floor = self.cfg.rack.min_signal_confidence

        def seen(*indices: int) -> bool:
            return all(confidences[index] >= floor for index in indices)

        shoulder_mid = _mid(points, CK.LEFT_SHOULDER, CK.RIGHT_SHOULDER)
        hip_mid = _mid(points, CK.LEFT_HIP, CK.RIGHT_HIP)
        ankle_mid = _mid(points, CK.LEFT_ANKLE, CK.RIGHT_ANKLE)
        wrist_l, wrist_r = points[CK.LEFT_WRIST], points[CK.RIGHT_WRIST]
        shoulder_l, shoulder_r = points[CK.LEFT_SHOULDER], points[CK.RIGHT_SHOULDER]
        wrist_mid = (wrist_l + wrist_r) / 2.0
        lateral_right = _lateral_right(points, ctx.capture_mode)
        forward = sagittal_forward(ctx.up, lateral_right)

        shoulders = seen(CK.LEFT_SHOULDER, CK.RIGHT_SHOULDER)
        hips = seen(CK.LEFT_HIP, CK.RIGHT_HIP)
        wrists = seen(CK.LEFT_WRIST, CK.RIGHT_WRIST)
        torso_m = float(np.linalg.norm(shoulder_mid - hip_mid)) if shoulders and hips else NAN

        bar_cm = NAN
        if wrists and shoulders:
            bar_cm = CM_PER_M * _height(wrist_mid, shoulder_mid)
            floor_torso_m = torso_ref_m if math.isfinite(torso_ref_m) else torso_m
            if bar_cm < -self.cfg.rack.rack_floor_torso_ratio * CM_PER_M * floor_torso_m:
                bar_cm = NAN

        tilt_deg = NAN
        if wrists:
            spread = abs(float(np.dot(wrist_l - wrist_r, lateral_right)))
            tilt_deg = math.degrees(math.atan2(_height(wrist_l, wrist_r), spread))

        def arm(side_shoulder: int, side_elbow: int, side_wrist: int) -> tuple[float, float, float]:
            if not seen(side_shoulder, side_elbow, side_wrist):
                return NAN, NAN, NAN
            shoulder, elbow, wrist = points[side_shoulder], points[side_elbow], points[side_wrist]
            outward = -lateral_right if side_wrist == CK.LEFT_WRIST else lateral_right
            forearm = wrist - elbow
            grip = math.degrees(math.atan2(float(np.dot(forearm, outward)), float(np.dot(forearm, WORLD_UP))))
            return (
                _flexion_deg(shoulder, elbow, wrist),
                _flexion_deg(_frontal(shoulder), _frontal(elbow), _frontal(wrist)),
                grip,
            )

        elbow_l_deg, elbow2d_l_deg, grip_l_deg = arm(CK.LEFT_SHOULDER, CK.LEFT_ELBOW, CK.LEFT_WRIST)
        elbow_r_deg, elbow2d_r_deg, grip_r_deg = arm(CK.RIGHT_SHOULDER, CK.RIGHT_ELBOW, CK.RIGHT_WRIST)

        legs = seen(CK.LEFT_HIP, CK.RIGHT_HIP, CK.LEFT_KNEE, CK.RIGHT_KNEE, CK.LEFT_ANKLE, CK.RIGHT_ANKLE)
        knee_flex_deg = leg_m = NAN
        if legs:
            knee_flex_deg = (
                _flexion_deg(points[CK.LEFT_HIP], points[CK.LEFT_KNEE], points[CK.LEFT_ANKLE])
                + _flexion_deg(points[CK.RIGHT_HIP], points[CK.RIGHT_KNEE], points[CK.RIGHT_ANKLE])
            ) / 2.0
            leg_m = float(
                np.linalg.norm(points[CK.LEFT_HIP] - points[CK.LEFT_KNEE])
                + np.linalg.norm(points[CK.LEFT_KNEE] - points[CK.LEFT_ANKLE])
                + np.linalg.norm(points[CK.RIGHT_HIP] - points[CK.RIGHT_KNEE])
                + np.linalg.norm(points[CK.RIGHT_KNEE] - points[CK.RIGHT_ANKLE])
            ) / 2.0

        # Rig: the bar against a mid-foot locked per rep needs the world frame.
        world = ctx.skeleton_world.to_numpy() if ctx.skeleton_world is not None else points
        feet = seen(CK.LEFT_ANKLE, CK.RIGHT_ANKLE, CK.LEFT_FOOT_INDEX, CK.RIGHT_FOOT_INDEX)
        midfoot = np.full(3, NAN)
        if feet:
            ankle_world = _mid(world, CK.LEFT_ANKLE, CK.RIGHT_ANKLE)
            toe_world = _mid(world, CK.LEFT_FOOT_INDEX, CK.RIGHT_FOOT_INDEX)
            midfoot = ankle_world + MIDFOOT_FRACTION_OF_ANKLE_TO_TOE * (toe_world - ankle_world)
        trunk = shoulder_mid - hip_mid

        return PressFrameSample(
            timestamp=ctx.timestamp,
            bar_cm=bar_cm,
            bar_above_hip_cm=CM_PER_M * _height(wrist_mid, hip_mid) if wrists and hips else NAN,
            shoulder_above_hip_cm=CM_PER_M * _height(shoulder_mid, hip_mid) if shoulders and hips else NAN,
            tilt_deg=tilt_deg,
            elbow_l_deg=elbow_l_deg,
            elbow_r_deg=elbow_r_deg,
            elbow2d_l_deg=elbow2d_l_deg,
            elbow2d_r_deg=elbow2d_r_deg,
            hip_cm=CM_PER_M * _height(hip_mid, ankle_mid) if legs else NAN,
            knee_flex_deg=knee_flex_deg,
            layback_deg=lean_back_deg(trunk, ctx.up, forward) if shoulders and hips else NAN,
            bar_fwd_shoulder_m=forward_offset(wrist_mid - shoulder_mid, ctx.up, forward) if wrists and shoulders else NAN,
            grip_tilt_l_deg=grip_l_deg,
            grip_tilt_r_deg=grip_r_deg,
            shoulder_width_x=abs(float(shoulder_l[0] - shoulder_r[0])) if shoulders else NAN,
            torso_m=torso_m,
            leg_m=leg_m,
            trunk_lateral_deg=angles.trunk_lateral_flexion if angles is not None else NAN,
            standing_pitch_deg=_standing_pitch_deg(raw_points) if shoulders and legs else NAN,
            wrist_mid=(world[CK.LEFT_WRIST] + world[CK.RIGHT_WRIST]) / 2.0 if wrists else np.full(3, NAN),
            midfoot=midfoot,
            up=np.asarray(ctx.up, dtype=np.float64),
            forward=forward,
        )

    def _rack_pose(self, points: np.ndarray, confidences: np.ndarray, torso_m: float) -> bool:
        rack = self.cfg.rack
        arm_indices = (CK.LEFT_WRIST, CK.RIGHT_WRIST, CK.LEFT_ELBOW, CK.RIGHT_ELBOW)
        if any(confidences[index] < rack.min_arm_confidence for index in arm_indices):
            return False
        arm_est_m = self.cfg.arm_to_torso_ratio * torso_m
        if not arm_est_m > MIN_SEGMENT_M:
            return False
        wrist_l, wrist_r = points[CK.LEFT_WRIST], points[CK.RIGHT_WRIST]
        shoulder_mid = _mid(points, CK.LEFT_SHOULDER, CK.RIGHT_SHOULDER)
        bar_ratio = _height((wrist_l + wrist_r) / 2.0, shoulder_mid) / arm_est_m
        if not rack.bar_height_min_ratio <= bar_ratio <= rack.bar_height_max_ratio:
            return False
        if _height(wrist_l, points[CK.LEFT_ELBOW]) < rack.min_forearm_rise_m:
            return False
        if _height(wrist_r, points[CK.RIGHT_ELBOW]) < rack.min_forearm_rise_m:
            return False
        if abs(_height(wrist_l, wrist_r)) > rack.max_hand_level_ratio * arm_est_m:
            return False
        lateral_right = _lateral_right(points, self._capture_mode)
        grip_width = abs(float(np.dot(wrist_l - wrist_r, lateral_right)))
        shoulder_width = abs(float(np.dot(points[CK.LEFT_SHOULDER] - points[CK.RIGHT_SHOULDER], lateral_right)))
        return grip_width >= rack.min_grip_to_shoulder_width_ratio * shoulder_width

    def _track_re_rack(self, sample: PressFrameSample) -> None:
        if not math.isfinite(sample.bar_cm):
            return
        lost_s = sample.timestamp - self._last_signal_time
        if not self._rep_open and lost_s > self.cfg.rack.re_rack_reset_s:
            # Hands off the bar long enough to re-rack: grip and camera may differ now.
            self._tilt_pool.clear()
        self._last_signal_time = sample.timestamp

    def _freeze_rack(self) -> None:
        samples = list(self._ready_frames)
        midfeet = np.array([sample.midfoot for sample in samples])
        finite_midfeet = midfeet[np.all(np.isfinite(midfeet), axis=1)]
        self._rack = _RackReference(
            samples=samples,
            timestamp=samples[-1].timestamp,
            bar_cm=_median([sample.bar_cm for sample in samples]),
            hip_cm=_median([sample.hip_cm for sample in samples]),
            shoulder_above_hip_cm=_median([sample.shoulder_above_hip_cm for sample in samples]),
            tilt_deg=_median([sample.tilt_deg for sample in samples]),
            shoulder_width_x=_median([sample.shoulder_width_x for sample in samples]),
            midfoot=np.median(finite_midfeet, axis=0) if len(finite_midfeet) else np.full(3, NAN),
        )
        # The latch frames open this set's pools (frozen once per set, so rebuilt here).
        self._tilt_pool = list(_finite([sample.tilt_deg for sample in samples]))
        self._set_widths = list(_finite([sample.shoulder_width_x for sample in samples]))
        self._set_torsos = list(_finite([sample.torso_m for sample in samples]))
        self._set_legs = list(_finite([sample.leg_m for sample in samples]))
        self._first_rep_pending = True
        self._note_rack_pitch(samples)

    # Pools rack-hold standing pitch readings into the set's; True when the
    # set's one tripwire check, due once enough are in, finds the camera moved.
    def _note_rack_pitch(self, samples: list[PressFrameSample]) -> bool:
        if self._capture_mode != SINGLE_CAMERA or not self._pitch_certified:
            return False
        self._set_pitch_readings.extend(_finite([sample.standing_pitch_deg for sample in samples]))
        camera = self.cfg.camera
        if (
            self._set_pitch_checked
            or not math.isfinite(self._tripwire.reference_deg)
            or len(self._set_pitch_readings) < camera.tripwire_min_frames
        ):
            return False
        self._set_pitch_checked = True
        delta_deg = self._tripwire.check(float(np.mean(self._set_pitch_readings)))
        if delta_deg is None:
            return False
        self._tripped_delta_deg = delta_deg
        return True

    # ------------------------------------------------------------------
    # Normalizers
    # ------------------------------------------------------------------

    def _torso_ref_m(self) -> float:
        if math.isfinite(self._body_torso_m):
            return self._body_torso_m
        return _median(self._set_torsos)

    def _arm_ref_m(self) -> float:
        return self.cfg.arm_to_torso_ratio * self._torso_ref_m()

    def _leg_ref_m(self) -> float:
        if math.isfinite(self._body_leg_m):
            return self._body_leg_m
        return _median(self._set_legs)

    @staticmethod
    def _grip_of(samples: list[PressFrameSample]) -> float:
        left = _median([sample.grip_tilt_l_deg for sample in samples])
        right = _median([sample.grip_tilt_r_deg for sample in samples])
        return (left + right) / 2.0

    # ------------------------------------------------------------------
    # Per-rep features
    # ------------------------------------------------------------------

    def _in_band(self, bar_cm: float, arm_ref_m: float) -> bool:
        ratio = bar_cm / (CM_PER_M * arm_ref_m)
        return self.cfg.rack.bar_height_min_ratio <= ratio <= self.cfg.rack.bar_height_max_ratio

    def _setup_candidates(self, valid: list[PressFrameSample], preroll_count: int, arm_ref_m: float) -> list[int]:
        candidates = []
        for i in range(preroll_count):
            neighbour = i - 1 if i > 0 else i + 1
            if neighbour >= len(valid):
                continue
            still = abs(valid[i].bar_cm - valid[neighbour].bar_cm) < SETUP_STILL_CM_PER_FRAME
            if still and self._in_band(valid[i].bar_cm, arm_ref_m):
                candidates.append(i)
        return candidates

    @staticmethod
    def _windows(bar: np.ndarray, base: int, rack_h: float) -> tuple[float, float, int, int | None]:
        top = _percentile(bar[base:], TOP_PERCENTILE)
        rom = top - rack_h
        if not rom > 0.0:
            return top, rom, base, None
        lock_level = top - LOCKOUT_FRACTION_OF_ROM * rom
        lock = next(i for i in range(base, len(bar)) if bar[i] >= lock_level)
        start_level = rack_h + START_FRACTION_OF_ROM * rom
        start = next((i for i in range(lock - 1, base - 1, -1) if bar[i] <= start_level), None)
        return top, rom, lock, start

    def _features(self, rep_number: int, trajectory: list[PressFrameSample]) -> PressRepFeatures:
        cfg = self.cfg
        arm_ref_m = self._arm_ref_m()
        leg_ref_m = self._leg_ref_m()
        features = PressRepFeatures(
            rep_number=rep_number,
            capture_mode=self._capture_mode,
            pitch_band=self._pitch_band,
            gravity_source=self._gravity_source,
            arm_ref_m=arm_ref_m,
            leg_ref_m=leg_ref_m,
            first_rep_after_latch=self._rep_first,
            piston=self._rep_piston,
        )
        preroll_ids = {id(sample) for sample in self._preroll}
        valid = [sample for sample in trajectory if math.isfinite(sample.bar_cm)]
        preroll_count = sum(1 for sample in valid if id(sample) in preroll_ids)
        features.sample_count = len(valid)
        if len(valid) < MIN_PRESS_SAMPLES or not math.isfinite(arm_ref_m):
            return features

        times = np.array([sample.timestamp for sample in valid])
        bar = _med3(np.array([sample.bar_cm for sample in valid]))
        base = int(np.argmin(bar[:preroll_count])) if preroll_count else 0
        rack = self._rack

        # Setup: the rack hold before the press — the idle ring's still in-band
        # frames, plus the latch frames on rep 1 (whose ring may be empty). A
        # touch-and-go has none: its turnaround only times the rep.
        candidates = [] if self._rep_piston else self._setup_candidates(valid, preroll_count, arm_ref_m)
        latch_setup = rack.samples if self._rep_first and rack is not None else []
        latch_bars = [sample.bar_cm for sample in latch_setup]
        if self._rep_piston:
            rack_h = float(bar[base])
        elif len(candidates) + len(latch_setup) >= MIN_SETUP_SAMPLES:
            rack_h = _median([valid[i].bar_cm for i in candidates] + latch_bars)
        else:
            rack_h = rack.bar_cm if rack is not None else float(bar[base])
        top, rom, lock, start = self._windows(bar, base, rack_h)
        setup_ids = candidates
        if candidates and start is not None:
            setup_ids = [i for i in candidates if times[i] <= times[start]]
            if len(setup_ids) + len(latch_setup) >= MIN_SETUP_SAMPLES:
                rack_h = _median([valid[i].bar_cm for i in setup_ids] + latch_bars)
                top, rom, lock, start = self._windows(bar, base, rack_h)
        if not rom > 0.0:
            return features
        features.rom_cm = float(rom)

        # new_setup: this rep's own frames, which join the set's pools.
        new_setup = [valid[i] for i in setup_ids]
        setup = latch_setup + new_setup
        if len(setup) < MIN_SETUP_SAMPLES:
            setup, new_setup = [], []
        t_start = float(times[start] if start is not None else times[base])
        press_time_s = float(times[lock] - times[start]) if start is not None else NAN

        # Criterion 1: started from the rack.
        if self._rep_first:
            features.rack_start = True
        elif self._rep_piston:
            features.rack_start = self._rep_continues_counted
        else:
            window_start = t_start - cfg.acceptance.setup_window_s
            held = sum(1 for i in candidates if window_start <= times[i] <= t_start)
            features.rack_start = held >= cfg.acceptance.setup_min_samples

        # Windows.
        lock_level = top - LOCKOUT_FRACTION_OF_ROM * rom
        lock_end = lock
        while lock_end < len(bar) and bar[lock_end] >= lock_level:
            lock_end += 1
        lockout_ids = list(range(lock, lock_end))
        if len(lockout_ids) < MIN_LOCKOUT_SAMPLES:
            lockout_ids = sorted(base + int(i) for i in np.argsort(bar[base:])[-MIN_LOCKOUT_SAMPLES:])
        press_ids = list(range(start if start is not None else base, lock))
        rack_level = rack_h + START_FRACTION_OF_ROM * rom
        lower_end = next((i for i in range(lock_end, len(bar)) if bar[i] <= rack_level), len(bar) - 1)
        lower_from = max(lock_end - 1, lock)

        gaps = [i for i in range(base + 1, len(times)) if times[i] - times[i - 1] > CLIPPED_GAP_S]
        shoulder_above_hip_cm = (
            rack.shoulder_above_hip_cm if rack is not None
            else _median([sample.shoulder_above_hip_cm for sample in setup or valid[:preroll_count] or valid])
        )
        height_ratio = (
            _med3(np.array([sample.bar_above_hip_cm for sample in valid])) - shoulder_above_hip_cm
        ) / (CM_PER_M * arm_ref_m)
        features.top_height_ratio = _percentile(height_ratio[base:], TOP_PERCENTILE)
        if gaps:
            features.wrists_clipped = True
            features.clip_height_ratio = float(height_ratio[gaps[0] - 1])
        clipped = features.wrists_clipped

        features.press_start_observed = start is not None and start > 0
        features.press_time_s = press_time_s
        features.lower_time_s = float(times[lower_end] - times[lower_from]) if lower_end > lower_from else NAN
        features.concentric_time_s = press_time_s
        features.eccentric_time_s = features.lower_time_s
        # A touch-and-go starts mid-range: its mean velocity is not comparable.
        if features.press_start_observed and not clipped and not self._rep_piston and press_time_s > 0.0:
            features.concentric_velocity_mps = float(bar[lock] - bar[start]) / press_time_s / CM_PER_M
        if self._capture_mode == TRIANGULATED:
            features.highlight_score = features.concentric_velocity_mps

        if setup:
            features.rack_height_ratio = _median([sample.bar_cm for sample in setup]) / (CM_PER_M * arm_ref_m)

        # Facing (single camera): shoulder width on setup and press-start frames
        # against the session's widest per-set median.
        commit = _Commit(len(self._tilt_pool), len(self._set_widths), len(self._set_torsos), len(self._set_legs))
        self._set_widths.extend(_finite([sample.shoulder_width_x for sample in new_setup]))
        self._set_torsos.extend(_finite([sample.torso_m for sample in new_setup]))
        self._set_legs.extend(_finite([sample.leg_m for sample in new_setup]))
        if self._capture_mode == SINGLE_CAMERA:
            facing_level = rack_h + FACING_MAX_FRACTION_OF_ROM * rom
            facing_widths = [sample.shoulder_width_x for sample in setup] + [
                valid[i].shoulder_width_x for i in press_ids if bar[i] < facing_level
            ]
            set_medians = _finite([*self._set_width_medians, _median(self._set_widths)])
            width_ref = float(set_medians.max()) if set_medians.size else NAN
            features.width_ratio = _median(facing_widths) / width_ref
            rotated_all_set = width_ref < (
                FIRST_SET_FACING_GUARD_RATIO * SHOULDER_WIDTH_TO_TORSO_RATIO * self._torso_ref_m()
            )
            features.facing_ok = features.width_ratio >= cfg.facing_min_width_ratio and not rotated_all_set
        else:
            features.facing_ok = True
        frontal_ok = features.facing_ok

        if frontal_ok and new_setup:
            self._tilt_pool.extend(_finite([sample.tilt_deg for sample in new_setup]))
        self._pending_commit = commit
        # A rep with no rack start held no rack posture worth a pitch reading.
        if new_setup and features.rack_start and self._note_rack_pitch(new_setup):
            # This rep's rack hold completed the set's reading and showed the camera moved.
            self._pitch_band = features.pitch_band = PITCH_BAND_UNKNOWN

        # Leg drive: the hip's lowest point around the press start against the setup.
        hip = _med3(np.array([sample.hip_cm for sample in valid]))
        if self._rep_piston or not setup:
            hip_ref = rack.hip_cm if rack is not None else NAN
        else:
            hip_ref = _median([sample.hip_cm for sample in setup])
        dip_end = t_start + DIP_PRESS_FRACTION * (press_time_s if math.isfinite(press_time_s) else 0.0)
        dip_ids = [i for i in range(len(valid)) if t_start - DIP_LEAD_S <= times[i] <= dip_end and math.isfinite(hip[i])]
        if dip_ids and math.isfinite(hip_ref):
            lowest = min(dip_ids, key=lambda i: hip[i])
            features.leg_drive_ratio = float(hip_ref - hip[lowest]) / (CM_PER_M * leg_ref_m)
            features.leg_drive_phase = PHASE_SETUP if times[lowest] < t_start else PHASE_PRESS
            knee_peak = _percentile([valid[i].knee_flex_deg for i in dip_ids], 100.0)
            features.knee_flexion_change_deg = knee_peak - _median([sample.knee_flex_deg for sample in setup])

        # Layback (rig): the trunk's lean back over the press.
        if len(press_ids) >= MIN_PRESS_SAMPLES:
            layback = _med3(np.array([sample.layback_deg for sample in valid]))
            features.layback_deg = _percentile(layback[press_ids], WORST_PERCENTILE)
            features.layback_change_deg = features.layback_deg - _median([sample.layback_deg for sample in setup])
            features.trunk_lateral_deg = _median([valid[i].trunk_lateral_deg for i in press_ids])

        # Lockout: elbows and where the bar finished (rig sagittal metrics).
        lockout = [valid[i] for i in lockout_ids]
        if len(lockout) >= MIN_LOCKOUT_SAMPLES and not clipped:
            if self._capture_mode == SINGLE_CAMERA:
                if frontal_ok:
                    features.lockout_elbow_l_deg = _median([sample.elbow2d_l_deg for sample in lockout])
                    features.lockout_elbow_r_deg = _median([sample.elbow2d_r_deg for sample in lockout])
            else:
                features.lockout_elbow_l_deg = _median([sample.elbow_l_deg for sample in lockout])
                features.lockout_elbow_r_deg = _median([sample.elbow_r_deg for sample in lockout])
            features.bar_forward_shoulder_ratio = _median([sample.bar_fwd_shoulder_m for sample in lockout]) / arm_ref_m
            midfoot = self._locked_midfoot(setup)
            features.bar_over_midfoot_ratio = _median([
                forward_offset(sample.wrist_mid - midfoot, sample.up, sample.forward) for sample in lockout
            ]) / arm_ref_m
        lockout_elbows = _finite([features.lockout_elbow_l_deg, features.lockout_elbow_r_deg])
        if lockout_elbows.size:
            features.max_depth_angle = 180.0 - float(np.mean(lockout_elbows))

        # Uneven: bar tilt above the rack against the set's pooled rack tilt.
        features.rack_tilt_deg = _median(self._tilt_pool)
        tilt_level = rack_h + TILT_MIN_FRACTION_OF_ROM * rom
        tilt_ids = [i for i in range(press_ids[0] if press_ids else lock, lock_end) if bar[i] >= tilt_level]
        tilt = _med3(np.array([sample.tilt_deg for sample in valid])) - features.rack_tilt_deg
        tilt_ids = [i for i in tilt_ids if math.isfinite(tilt[i])]
        if frontal_ok and len(tilt_ids) >= MIN_PRESS_SAMPLES:
            deviations = tilt[tilt_ids]
            sign = 1.0 if np.median(deviations) >= 0.0 else -1.0
            features.bar_tilt_deg = sign * _percentile(np.abs(deviations), WORST_PERCENTILE)
            peak = tilt_ids[int(np.argmax(np.abs(deviations)))]
            features.bar_tilt_phase = PHASE_LOCKOUT if peak >= lock else PHASE_PRESS

        # Grip: forearm tilt in the rack (setup-dependent: never on a touch-and-go).
        if frontal_ok and not self._rep_piston and setup:
            features.grip_tilt_deg = self._grip_of(setup)

        features.partial_bottom = self._partial_bottom(bar, base, lock_level, top - PARTIAL_BOTTOM_DIP_FRACTION * rom)
        return features

    def _locked_midfoot(self, setup: list[PressFrameSample]) -> np.ndarray:
        midfeet = np.array([sample.midfoot for sample in setup]) if setup else np.empty((0, 3))
        finite = midfeet[np.all(np.isfinite(midfeet), axis=1)] if len(midfeet) else midfeet
        if len(finite):
            return np.median(finite, axis=0)
        return self._rack.midfoot if self._rack is not None else np.full(3, NAN)

    @staticmethod
    def _partial_bottom(bar: np.ndarray, base: int, lock_level: float, dip_level: float) -> bool:
        locked_once = dipped = False
        for value in bar[base:]:
            if value >= lock_level:
                if locked_once and dipped:
                    return True
                locked_once = True
            elif locked_once and value < dip_level:
                dipped = True
        return False
