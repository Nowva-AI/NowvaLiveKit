"""
Biomechanics Pipeline

Wires all processing layers into a single pipeline:
  Capture → Pose estimation → Pre-IK chain (Kalman, foot contact, re-centring)
  → IK solve → Fault detection → Rep counting

Returns a PipelineFrame per iteration with per-layer timing. The analysis
skeleton lags the capture by `kalman.lag_frames`; the display skeleton is
undelayed.
"""

from __future__ import annotations

import math

import logging
import os
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from biomechanics.analysis.rep_features import (
    MIN_KEYPOINT_CONFIDENCE,
    RepFeatures,
    SetupSnapshot,
    build_frame_sample,
    build_setup_snapshot,
    compute_rep_features,
)
from biomechanics.config import BiomechanicsConfig
from biomechanics.faults.rules.depth import DepthCategory, DepthRule, depth_category
from biomechanics.ml.bilstm_counter import ASSESSMENT_MIN_DEPTH_CLASS
from biomechanics.pose.mediapipe_fallback import MediaPipePoseEstimator
from biomechanics.kinematics.analytical_ik import AnalyticalIKSolver
from biomechanics.kinematics.valgus import build_valgus_estimator
from biomechanics.faults import RuleEngine
from biomechanics.profiles import get_profile
from biomechanics.triangulation.triangulator import recentre_at_hips
from biomechanics.utils.types import (
    PipelineFrame,
    Skeleton2D,
    Skeleton3D,
    JointAngles,
    FaultEvent,
    CocoKeypoints,
    DEPTH_CLASS_NAMES,
    BarbellDetection,
    BarTrackState,
)
from biomechanics.utils.filters import RunningMedian
from biomechanics.utils.position_filter import Skeleton2DSmoother
from biomechanics.utils.preik_chain import PreIKResult, build_preik_chain
from biomechanics.utils.segment_lengths import SegmentLengthEstimator, MIN_ENDPOINT_CONFIDENCE
from biomechanics.utils.standing_gate import StandingPoseGate

logger = logging.getLogger(__name__)

# Roughly 20 seconds at 30 fps — long enough for any real rep, short enough
# that a rep the counter never closes cannot grow the buffer indefinitely.
MAX_REP_TRAJECTORY_FRAMES = 600

# Below the two-view confidence cap (0.3) so 2-camera rigs can measure the body.
TWO_CAMERA_MEASUREMENT_CONFIDENCE = 0.25

# Single camera: process_frame waits this long for a frame it has not processed.
NEW_FRAME_WAIT_S = 0.1
# A failed camera read is retried after this pause, never in a busy loop, and
# logged on the first failure and then once per this many failures.
READ_FAILURE_BACKOFF_S = 0.05
READ_FAILURES_PER_LOG = 100
# Repeated pose-estimation failures are logged at most this often.
POSE_FAILURE_LOG_INTERVAL_S = 5.0
# Captures kept to pair the lagged analysis frame with what was measured when
# it was captured; more than the Kalman lag so a re-initialised lag still matches.
RECENT_CAPTURES = 8
# Single camera: the world frame is the camera's, so a tilted camera tilts every
# sagittal angle and the depth reading with it. Standing upright beyond this
# angle from the camera's vertical means the camera needs levelling.
MAX_CAMERA_PITCH_DEG = 8.0

# A stored bottom/standing frame is only useful with the whole lower body present.
_LEG_KEYPOINTS = (
    CocoKeypoints.LEFT_HIP, CocoKeypoints.RIGHT_HIP,
    CocoKeypoints.LEFT_KNEE, CocoKeypoints.RIGHT_KNEE,
    CocoKeypoints.LEFT_ANKLE, CocoKeypoints.RIGHT_ANKLE,
)
_LEG_KEYPOINT_NAMES = {
    CocoKeypoints.LEFT_HIP: "left_hip", CocoKeypoints.RIGHT_HIP: "right_hip",
    CocoKeypoints.LEFT_KNEE: "left_knee", CocoKeypoints.RIGHT_KNEE: "right_knee",
    CocoKeypoints.LEFT_ANKLE: "left_ankle", CocoKeypoints.RIGHT_ANKLE: "right_ankle",
}

# Depth category → the 5-class depth vocabulary the shallow-rep message speaks.
_DEPTH_CATEGORY_CLASS = {
    DepthCategory.QUARTER: 1,
    DepthCategory.HALF: 2,
    DepthCategory.PARALLEL: 3,
    DepthCategory.BELOW_PARALLEL: 4,
}

# With no post-IK angle filter, a per-rep max over raw frames is a
# single-frame statistic that one transient can over-read by 7–10°. Every
# rep extremum (bottom frame, depth class, BiLSTM window) is taken over a
# running median of this many frames instead.
KNEE_FLEXION_MEDIAN_FRAMES = 3


def _standing_pitch_deg(skeleton: Skeleton3D) -> float:
    """Forward tilt of the ankle-to-shoulder line from the camera's vertical, in degrees.
    Positive = shoulders further from the camera than the ankles (camera pitched up)."""
    points = skeleton.to_numpy()
    shoulder_mid = (points[CocoKeypoints.LEFT_SHOULDER] + points[CocoKeypoints.RIGHT_SHOULDER]) / 2.0
    ankle_mid = (points[CocoKeypoints.LEFT_ANKLE] + points[CocoKeypoints.RIGHT_ANKLE]) / 2.0
    body_axis = shoulder_mid - ankle_mid
    # Y-down: up the body is -y.
    return math.degrees(math.atan2(float(body_axis[2]), float(-body_axis[1])))


class BiomechanicsPipeline:
    """
    Full biomechanics processing pipeline.

    Captures video, estimates pose, computes joint angles, detects faults,
    and counts reps. Each call to process_frame() runs one iteration and
    returns a PipelineFrame with all results and per-layer timing.
    """

    def __init__(
        self,
        config: BiomechanicsConfig | None = None,
        exercise_name: str = "Barbell Back Squat",
        defer_capture: bool = False,
    ):
        self.config = config or BiomechanicsConfig()
        # Single-camera: counts process_frame calls. Multi-camera: the
        # primary camera's capture sequence of the last skeleton received.
        self._frame_index = 0

        # Load exercise profile (bundles fault rules, rep signal, cues)
        self._profile = get_profile(exercise_name)

        # Layer 1: Capture (threaded — always holds the latest frame, its
        # capture time and a sequence number so no frame is processed twice)
        self._cap = None
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._frame_ready = threading.Condition(self._frame_lock)
        self._capture_sequence = 0
        self._latest_capture_time = math.nan
        self._processed_sequence = 0
        self._capture_running = False
        self._capture_thread = None
        self._pose_failures = 0
        self._last_pose_failure_log_s = -math.inf
        # (capture time, raw 2D pose, unmeasured leg keypoints) per recent capture.
        self._recent_captures: deque[tuple[float, Skeleton2D | None, list[str]]] = deque(
            maxlen=RECENT_CAPTURES
        )

        # Multi-camera mode (env-driven)
        self._multi_camera = os.getenv("NOWVA_MULTI_CAMERA", "false").lower() == "true"
        self._valgus_estimator = build_valgus_estimator(self._multi_camera)
        self._multi_camera_provider = None

        # One barbell detector serves Layer 3b tracking and, inside a camera-calibration
        # capture window, the provider (the bar as a metric ruler).
        bar_detector = self._build_bar_detector()

        if self._multi_camera:
            from biomechanics.pose.multi_camera import MultiCameraPoseProvider

            tri = self.config.triangulation
            camera_calibration = self.config.camera_calibration
            self._multi_camera_provider = MultiCameraPoseProvider(
                device_ids=tri.device_ids,
                confidence_threshold=self.config.pose.confidence_threshold,
                model_path=self.config.pose.model_path,
                min_views=tri.min_views,
                max_reprojection_error=tri.max_reprojection_error,
                max_sync_delta_ms=tri.max_sync_delta_ms,
                resolution=self.config.capture.resolution,
                primary_camera=tri.primary_camera,
                focal_length_factor=tri.focal_length_factor,
                camera_keys=camera_calibration.camera_keys,
                calibration_buffer_frames=camera_calibration.calibration_buffer_frames,
                bar_detection_stride=camera_calibration.bar_detection_stride,
                bar_detector=bar_detector,
            )
            # The session flow loads calibrations (and writes this path when it
            # does not exist yet); loading here serves standalone tools only.
            if tri.calibration_file and Path(tri.calibration_file).exists():
                self._multi_camera_provider.load_calibration(tri.calibration_file)
        elif not defer_capture:
            self._open_capture()

        # Layer 2: Pose estimation (single-camera only; multi-camera owns its estimator)
        if self.config.pose.backend == "rtmpose":
            from biomechanics.pose.rtmpose import RTMPoseEstimator

            self._pose_estimator = RTMPoseEstimator(
                confidence_threshold=self.config.pose.confidence_threshold,
                model_path=self.config.pose.model_path,
                keypoint_format=self.config.pose.keypoint_format,
            )
        else:
            model_complexity = self.config.pose.model_complexity
            self._pose_estimator = MediaPipePoseEstimator(
                confidence_threshold=self.config.pose.confidence_threshold,
                model_complexity=model_complexity,
            )

        # Layer 2b: Pre-IK chain (temporal state reset per set) and the
        # session-scoped body measurement that feeds proportion scaling.
        self._preik = build_preik_chain(self.config, self._multi_camera)
        # Two-view triangulation caps confidence at TWO_VIEW_CONFIDENCE_CAP, so
        # a 2-camera rig needs a lower endpoint gate or it never completes.
        measurement_gate = MIN_ENDPOINT_CONFIDENCE
        if self._multi_camera and len(self.config.triangulation.device_ids) == 2:
            measurement_gate = TWO_CAMERA_MEASUREMENT_CONFIDENCE
        self.body_calibration = SegmentLengthEstimator(min_endpoint_confidence=measurement_gate)

        # Layer 3: Inverse kinematics
        self._ik_solver = AnalyticalIKSolver()

        # Standing pose gate — runs unconditionally to validate user is
        # in frame and standing before any calibration starts.
        sg = self.config.standing_gate
        self._standing_gate = StandingPoseGate(
            min_confidence=sg.min_confidence,
            max_knee_flexion_deg=sg.max_knee_flexion_deg,
            max_trunk_flexion_deg=sg.max_trunk_flexion_deg,
            min_torso_length_m=sg.min_torso_length_m,
            max_torso_length_m=sg.max_torso_length_m,
            min_leg_extension_ratio=sg.min_leg_extension_ratio,
            required_consecutive_frames=sg.required_consecutive_frames,
        )

        # Readiness gate — per-set gate that ensures the user is fully
        # detected and standing before data collection begins. Resets
        # between sets so each set starts with clean data.
        rg = self.config.readiness_gate
        self._readiness_gate = StandingPoseGate(
            min_confidence=rg.min_confidence,
            max_knee_flexion_deg=rg.max_knee_flexion_deg,
            max_trunk_flexion_deg=rg.max_trunk_flexion_deg,
            min_torso_length_m=rg.min_torso_length_m,
            max_torso_length_m=rg.max_torso_length_m,
            min_leg_extension_ratio=rg.min_leg_extension_ratio,
            required_consecutive_frames=rg.required_consecutive_frames,
        )

        # Single camera: the standing body's tilt from the camera vertical,
        # measured each time the readiness gate latches.
        self.camera_pitch_deg: float = math.nan
        self._standing_pitches: deque[float] = deque(maxlen=rg.required_consecutive_frames)
        self._camera_pitch_warned = False

        # Presence-only mode (rest periods / workout complete): pose
        # estimation keeps running so the user stays detected, but gates,
        # IK, fault detection, rep counting, and data collection are
        # all skipped until the flag is cleared.
        self.presence_only: bool = False

        # Inspector state: populated by the chain's tap when --inspect is on.
        self._inspecting: bool = False
        self._inspect_raw_kpts: np.ndarray | None = None
        self._inspect_intermediates: dict[str, np.ndarray] = {}
        self._inspect_raw_angles: JointAngles | None = None

        # Display-only 2D skeleton smoothing (does not affect analysis pipeline)
        self._display_smoother: Skeleton2DSmoother | None = None
        if self.config.display_filter.enabled:
            df = self.config.display_filter
            self._display_smoother = Skeleton2DSmoother(
                min_cutoff=df.min_cutoff,
                beta=df.beta,
                d_cutoff=df.d_cutoff,
            )

        # Layer 3b (optional): Barbell detection + Kalman-smoothed tracking.
        # Runs on the raw frame independently of pose; feeds BarPathRule
        # and BarTiltAsymmetryRule via bar_detection kwarg.
        self._barbell_detector = None
        self._bar_tracker = None
        if self.config.barbell_tracking.enabled:
            from biomechanics.barbell_tracking import BarPathTracker

            bt = self.config.barbell_tracking
            self._barbell_detector = bar_detector
            self._bar_tracker = BarPathTracker(
                bar_length_m=bt.bar_length_m,
                kalman_q=bt.kalman_q,
                kalman_r=bt.kalman_r,
                path_history_len=bt.path_history_len,
            )

        # Layer 4: Fault detection (rules provided by exercise profile)
        profile_rules = self._profile.create_fault_rules(self.config)
        self._rule_engine = RuleEngine(rules=profile_rules)

        # Layer 5: Rep counting (strategy determined by exercise profile)
        self._rep_counter = self._profile.create_rep_counter(self.config)

        # Layer 6 (optional): BiLSTM rep counting
        self._bilstm = None
        if self.config.bilstm.enabled:
            from biomechanics.ml.inference import BiLSTMInference
            from biomechanics.ml.bilstm_counter import BiLSTMCounterConfig

            bilstm_counter_cfg = BiLSTMCounterConfig(
                min_depth_class=self.config.bilstm.min_depth_class,
                min_rep_frames=self.config.bilstm.min_rep_frames,
                ema_alpha=self.config.bilstm.ema_alpha,
                num_classes=self.config.bilstm.num_classes,
            )
            self._bilstm = BiLSTMInference(
                model_path=self.config.bilstm.model_path,
                device=self.config.bilstm.device,
                config=bilstm_counter_cfg,
            )

        # Track max knee flexion independently for BiLSTM rep windows.
        # The hip position counter's snapshot may be desync'd from the
        # BiLSTM's rep boundaries, so we track angle peaks here.
        self._bilstm_max_knee_flex: float = 0.0
        self._bilstm_min_knee_flex: float = 180.0

        # Buffer faults from the hip counter for the BiLSTM to consume.
        # The hip counter resets _current_faults on rep completion, which
        # happens before the BiLSTM fires — so we stash them here.
        self._pending_bilstm_faults: list[FaultEvent] = []
        self._pending_bilstm_features: RepFeatures | None = None
        # Deepest hip position (femur lengths above parallel) over the
        # BiLSTM's rep window, for its depth-target gate.
        self._bilstm_min_depth_ratio: float = math.nan

        # A profile with a depth rule (squat) counts reps against the
        # athlete's own geometric depth target, which the learned depth
        # classes cannot express. The BiLSTM then only segments movement —
        # any descent opens a rep — and the target decides whether it counts.
        self._depth_gated = any(isinstance(rule, DepthRule) for rule in self._rule_engine.rules)
        if self._depth_gated:
            # Until calibration measures the athlete, reps count at a lenient target.
            self._rule_engine.set_depth_target(self.config.faults.depth.uncalibrated_target_ratio)
            if self._bilstm is not None:
                self._bilstm.set_min_depth_class(ASSESSMENT_MIN_DEPTH_CLASS)

        # Robust per-frame knee flexion: running median feeding every rep
        # extremum below. Per-set temporal state.
        self._knee_flexion_median = RunningMedian(window_frames=KNEE_FLEXION_MEDIAN_FRAMES)
        # Max of the median over the current rep window; what the depth rule
        # and RepData report as the rep's depth.
        self._rep_max_knee_flex: float = math.nan

        # Bottom-of-rep buffer for diagnosis engine: the frame with the
        # lowest hip relative to the knees (the geometric bottom).
        self._bottom_min_depth_ratio: float = math.inf
        self._rep_min_depth_ratio: float = math.nan
        self._bottom_kpts: list[list[float]] | None = None
        self._bottom_angles: dict | None = None

        # Standing-frame buffer: last skeleton before in_rep becomes True.
        self._standing_kpts: list[list[float]] | None = None
        self._standing_captured: bool = False

        # The most upright frame between reps — the athlete's setup for the
        # next rep (foot placement, lockout) — and the tallest they have
        # stood this session, which lockout is measured against.
        self._idle_peak_hip_cm: float = -math.inf
        self._idle_peak_skeleton: Skeleton3D | None = None
        self._rep_setup: SetupSnapshot | None = None
        self._standing_reference_hip_cm: float = math.nan

        # Per-rep trajectory buffer for whole-rep scoring. Holds a handful of
        # scalars per frame rather than full skeletons — a stuck rep must not
        # grow this without bound on an edge device.
        self._rep_trajectory: deque[dict] = deque(maxlen=MAX_REP_TRAJECTORY_FRAMES)

        # Store last raw frame for dashboard access
        self.last_frame: np.ndarray | None = None

    def _build_bar_detector(self):
        bt = self.config.barbell_tracking
        calibration_ruler = self._multi_camera and self.config.camera_calibration.use_bar_scale
        if not bt.enabled and not calibration_ruler:
            return None
        if not bt.enabled and not Path(bt.model_path).exists():
            logger.warning(
                "[PIPELINE] camera_calibration.use_bar_scale is on but %s is missing; "
                "camera calibration takes its scale from the user's height instead",
                bt.model_path,
            )
            return None

        from biomechanics.barbell_tracking import BarbellDetector

        return BarbellDetector(
            model_path=bt.model_path,
            conf_threshold=bt.conf_threshold,
            imgsz=bt.imgsz,
            device=bt.device,
        )

    def _open_capture(self) -> None:
        self._cap = cv2.VideoCapture(self.config.capture.device_id)
        w, h = self.config.capture.resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open camera device {self.config.capture.device_id}"
            )

        self._capture_running = True
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True
        )
        self._capture_thread.start()

    def start_capture(self) -> None:
        """Open camera and start capture thread (phase 2 for deferred init)."""
        if self._multi_camera:
            if self._multi_camera_provider is not None:
                self._multi_camera_provider.start()
            return
        if self._cap is not None:
            return
        self._open_capture()

    def preload_pose_model(self) -> None:
        """Eagerly load the pose estimation model instead of waiting for the first frame."""
        if self._multi_camera and self._multi_camera_provider is not None:
            self._multi_camera_provider.initialize()
        else:
            self._pose_estimator.initialize()

    @property
    def rep_counter(self):
        """Expose rep counter for external access (e.g. dashboard)."""
        return self._rep_counter

    @property
    def rep_count(self) -> int:
        """Reps that were actually logged.

        The BiLSTM is the authority whenever it is enabled — it emits the
        RepData that reaches the session tracker, so anything displayed to
        the lifter has to come from the same counter or the screen and the
        workout disagree.
        """
        if self._bilstm is not None:
            return self._bilstm.rep_count
        return self._rep_counter.rep_count

    @property
    def is_ready(self) -> bool:
        """Whether the readiness gate has passed and data is being collected."""
        return self._readiness_gate.is_ready

    @property
    def preik_stage_names(self) -> tuple[str, ...]:
        return self._preik.stage_names

    def reset_readiness_gate(self) -> None:
        """Reset the per-set state.

        Call this at set boundaries (shortly before rest ends, set
        timeout) so the next set requires the user to be fully detected
        before collection resumes. Resets the temporal filter state so
        set 2+ doesn't smooth against stale positions. Body measurements
        and foot contact anchors are session-scoped and untouched.
        """
        self._readiness_gate.reset()
        self._preik.reset()
        # Per-rep rule state (asymmetry samples, cooldowns) must not cross sets;
        # baseline calibration is kept by the engine's reset.
        self._rule_engine.reset()
        self._knee_flexion_median.reset()
        self._rep_max_knee_flex = math.nan
        self._bilstm_max_knee_flex = 0.0
        self._bilstm_min_knee_flex = 180.0
        self._bilstm_min_depth_ratio = math.nan
        self._pending_bilstm_faults.clear()
        self._pending_bilstm_features = None
        self._bottom_min_depth_ratio = math.inf
        self._rep_min_depth_ratio = math.nan
        self._bottom_kpts = None
        self._bottom_angles = None
        self._standing_kpts = None
        self._standing_captured = False
        self._idle_peak_hip_cm = -math.inf
        self._idle_peak_skeleton = None
        self._rep_setup = None
        self._rep_trajectory.clear()
        self._recent_captures.clear()

        if self._multi_camera_provider is not None:
            logger.info(
                "[PIPELINE] Triangulator L/R swap corrections this session: %d",
                getattr(self._multi_camera_provider, "swap_count", 0),
            )
            reset_provider = getattr(self._multi_camera_provider, "reset_temporal_state", None)
            if callable(reset_provider):
                reset_provider()
        else:
            reset_tracking = getattr(self._pose_estimator, "reset_tracking", None)
            if callable(reset_tracking):
                reset_tracking()

        if self._display_smoother is not None:
            self._display_smoother.reset()

    def on_calibration_changed(self, world_frame_changed: bool) -> None:
        """
        A new camera calibration was installed in the provider. Temporal state, foot
        contact anchors and the floor restart: even a refine that keeps the world frame
        shifts it by ~1 cm, which the planted-foot model would carry as a heel-rise bias.
        Body measurements are lengths, so they survive. world_frame_changed says whether
        the origin, heading or vertical moved (first calibration, board -> person re-anchor).
        """
        self._preik.reset_world_state()
        if self._multi_camera_provider is not None:
            self._multi_camera_provider.reset_temporal_state()
        logger.info(
            "[PIPELINE] Camera calibration installed (%s): temporal and foot contact state reset",
            "world frame moved" if world_frame_changed else "world frame kept",
        )

    def apply_athlete_params(self, params: dict) -> None:
        """Adopt a returning user's stored body measurements and scale thresholds once."""
        self.body_calibration = SegmentLengthEstimator.from_athlete_params(params)
        self._apply_body_proportions()

    def _apply_body_proportions(self) -> None:
        proportions = self.body_calibration.body_proportions
        if proportions is None:
            return
        self._rule_engine.apply_body_proportion_scaling(proportions)
        # The single-camera knee metric needs a femur length the weak depth
        # axis cannot shorten; a standing measurement is exactly that.
        set_femur_length = getattr(self._valgus_estimator, "set_femur_length", None)
        if set_femur_length is not None:
            set_femur_length(proportions.femur_length_avg)
        logger.info(
            "[PIPELINE] Body proportions applied: femur=%.3fm torso=%.3fm lean_scale=%.2f",
            proportions.femur_length_avg, proportions.torso_length_avg, proportions.forward_lean_scale,
        )

    def set_depth_target(self, target_ratio: float | None) -> None:
        """The athlete's depth target (femur lengths above parallel); None counts every descent."""
        self._rule_engine.set_depth_target(target_ratio)

    def consume_bottom_frame(self) -> tuple[list[list[float]] | None, dict | None]:
        """Return and reset the bottom-of-rep keypoints and angles.

        Returns (bottom_kpts, bottom_angles) captured at the rep's lowest hip
        position relative to the knees, then clears the buffer for the next rep.
        """
        kpts = self._bottom_kpts
        angles = self._bottom_angles
        self._bottom_min_depth_ratio = math.inf
        self._bottom_kpts = None
        self._bottom_angles = None
        return kpts, angles

    def consume_standing_frame(self) -> list[list[float]] | None:
        """Return the standing keypoints captured just before the rep started."""
        kpts = self._standing_kpts
        self._standing_captured = False
        return kpts

    def _femur_length_m(self) -> float | None:
        proportions = self.body_calibration.body_proportions
        return proportions.femur_length_avg if proportions is not None else None

    def _leg_length_m(self) -> float | None:
        proportions = self.body_calibration.body_proportions
        if proportions is None:
            return None
        return proportions.femur_length_avg + proportions.tibia_length_avg

    def _track_setup(self, skeleton_3d: Skeleton3D, sample) -> None:
        """Keep the most upright frame between reps."""
        height = sample.hip_height_cm
        if math.isfinite(height) and height > self._idle_peak_hip_cm:
            self._idle_peak_hip_cm = height
            self._idle_peak_skeleton = skeleton_3d

    def _start_rep_setup(self) -> None:
        """A rep began: freeze the setup it started from and open a fresh trajectory."""
        self._rep_trajectory.clear()
        self._rep_min_depth_ratio = math.nan
        self._rep_setup = None
        if self._idle_peak_skeleton is not None:
            self._rep_setup = build_setup_snapshot(self._idle_peak_skeleton)
            height = self._rep_setup.hip_height_cm
            if math.isfinite(height) and not height <= self._standing_reference_hip_cm:
                self._standing_reference_hip_cm = height
        self._idle_peak_hip_cm = -math.inf
        self._idle_peak_skeleton = None

    def _rep_features(self, rep_number: int) -> RepFeatures:
        return compute_rep_features(
            list(self._rep_trajectory),
            rep_number,
            femur_m=self._femur_length_m(),
            setup=self._rep_setup,
            standing_reference_hip_cm=self._standing_reference_hip_cm,
            leg_length_m=self._leg_length_m(),
        )

    def _shallow_descent(
        self, depth_ratio: float, angles: JointAngles, rep_number: int
    ) -> tuple[list[FaultEvent], int | None]:
        """Depth fault + depth class for a descent rejected for missing the target."""
        femur_m = self._femur_length_m()
        depth_cm = depth_ratio * femur_m * 100.0 if femur_m and math.isfinite(depth_ratio) else math.nan
        faults = self._rule_engine.judge_shallow_descent(depth_ratio, angles, rep_number, depth_cm)
        if not faults or math.isnan(depth_ratio):
            return faults, None
        return faults, _DEPTH_CATEGORY_CLASS[depth_category(depth_ratio)]

    @staticmethod
    def _legs_present(skeleton_3d: Skeleton3D) -> bool:
        # The IK solver's own floor: anything less yields NaN angles.
        return all(
            skeleton_3d.keypoints[idx].confidence >= MIN_KEYPOINT_CONFIDENCE for idx in _LEG_KEYPOINTS
        )

    def consume_rep_trajectory(self) -> list[dict]:
        """Return and reset the per-frame samples buffered during the last rep."""
        samples = list(self._rep_trajectory)
        self._rep_trajectory.clear()
        return samples

    def enable_inspect(self) -> None:
        self._inspecting = True
        self._preik.set_tap(self._record_inspect_stage)

    def _record_inspect_stage(self, stage: str, points: np.ndarray, confidences: np.ndarray) -> None:
        snapshot = np.array(points, dtype=np.float64, copy=True)
        self._inspect_intermediates[stage] = snapshot
        if stage == self._preik.stage_names[0]:
            self._inspect_raw_kpts = snapshot

    def _publish_frame(self, frame: np.ndarray, capture_time: float) -> None:
        with self._frame_ready:
            self._latest_frame = frame
            self._latest_capture_time = capture_time
            self._capture_sequence += 1
            self._frame_ready.notify_all()

    def _capture_loop(self) -> None:
        """Continuously read frames from the camera in a background thread."""
        read_failures = 0
        while self._capture_running:
            ret, frame = self._cap.read()
            if ret and frame is not None:
                read_failures = 0
                # read() returns when the frame arrives: the closest clock to its capture.
                self._publish_frame(frame, time.time())
                continue
            if read_failures % READ_FAILURES_PER_LOG == 0:
                logger.warning(
                    "[PIPELINE] Camera %s read failed (%d in a row) — retrying every %.0f ms",
                    self.config.capture.device_id, read_failures + 1, READ_FAILURE_BACKOFF_S * 1000.0,
                )
            read_failures += 1
            time.sleep(READ_FAILURE_BACKOFF_S)

    def _track_camera_pitch(self, raw_centred: Skeleton3D) -> None:
        self._standing_pitches.append(_standing_pitch_deg(raw_centred))
        if not self._readiness_gate.is_ready:
            return
        # The gate just latched on consecutive standing frames: those are the window.
        self.camera_pitch_deg = float(np.median(self._standing_pitches))
        self._standing_pitches.clear()
        if abs(self.camera_pitch_deg) > MAX_CAMERA_PITCH_DEG and not self._camera_pitch_warned:
            self._camera_pitch_warned = True
            logger.warning(
                "[PIPELINE] Standing upright reads %.0f° from the camera's vertical (limit %.0f°): "
                "level the camera — depth and every side-view angle tilt with it",
                self.camera_pitch_deg, MAX_CAMERA_PITCH_DEG,
            )

    def _log_pose_failure(self) -> None:
        self._pose_failures += 1
        clock_s = time.perf_counter()
        if clock_s - self._last_pose_failure_log_s >= POSE_FAILURE_LOG_INTERVAL_S:
            self._last_pose_failure_log_s = clock_s
            logger.exception("[PIPELINE] Pose estimation failed (%d failures so far)", self._pose_failures)

    @staticmethod
    def _unmeasured_leg_keypoints(skeleton_3d: Skeleton3D | None) -> list[str]:
        if skeleton_3d is None:
            return list(_LEG_KEYPOINT_NAMES.values())
        return [
            name for idx, name in _LEG_KEYPOINT_NAMES.items()
            if skeleton_3d.keypoints[idx].confidence < MIN_KEYPOINT_CONFIDENCE
        ]

    def _capture_at(self, timestamp: float) -> tuple[float, Skeleton2D | None, list[str]] | None:
        for capture in self._recent_captures:
            if capture[0] == timestamp:
                return capture
        return None

    def _record_body_measurements(self, result: PreIKResult) -> None:
        source = result.analysis_world if result.analysis_world is not None else result.analysis
        confidences = np.array([kp.confidence for kp in source.keypoints], dtype=np.float64)
        self.body_calibration.record(source.to_numpy(), confidences, self.rep_count)
        if self.body_calibration.is_complete:
            self._apply_body_proportions()

    def process_frame(self) -> PipelineFrame:
        """
        Run one full pipeline iteration.

        Returns:
            PipelineFrame with all layer outputs and per-layer timing.
        """
        latency_ms: dict[str, float] = {}
        now = time.time()

        if self._inspecting:
            self._inspect_raw_kpts = None
            self._inspect_intermediates = {}
            self._inspect_raw_angles = None

        # --- Capture + Pose estimation ---
        skeleton_2d: Skeleton2D | None = None
        raw_3d: Skeleton3D | None = None
        bar_detection: BarbellDetection | None = None
        bar_track: BarTrackState | None = None

        lost_cameras: list[str] | None = None
        if self._multi_camera and self._multi_camera_provider is not None:
            t0 = time.perf_counter()
            frame, skeleton_2d, raw_3d = self._multi_camera_provider.get_pose()
            latency_ms["capture"] = 0.0
            latency_ms["pose"] = (time.perf_counter() - t0) * 1000.0
            lost_cameras = self._multi_camera_provider.lost_cameras()
            if frame is not None:
                # The synced set's primary capture time: every skeleton of the
                # set carries it, and so does a dropout predicted in its place.
                now = self._multi_camera_provider.last_capture_timestamp
            if raw_3d is not None:
                # The skeleton carries the primary camera's capture sequence
                # and timestamp; never overwrite them.
                self._frame_index = raw_3d.frame_index
        else:
            t0 = time.perf_counter()
            with self._frame_ready:
                if self._capture_sequence == self._processed_sequence:
                    self._frame_ready.wait(timeout=NEW_FRAME_WAIT_S)
                frame = self._latest_frame
                sequence = self._capture_sequence
                capture_time = self._latest_capture_time
            latency_ms["capture"] = (time.perf_counter() - t0) * 1000.0
            if sequence == self._processed_sequence:
                # The loop outran the camera: never run the same frame twice.
                frame = None

            if frame is not None:
                self._processed_sequence = sequence
                self._frame_index = sequence
                now = capture_time
                t0 = time.perf_counter()
                try:
                    skeleton_2d, raw_3d = self._pose_estimator.estimate_both(frame)
                except Exception:
                    self._log_pose_failure()
                # Skeletons describe the moment of capture, not when inference finished.
                for skeleton in (skeleton_2d, raw_3d):
                    if skeleton is not None:
                        skeleton.timestamp = capture_time
                        skeleton.frame_index = sequence
                latency_ms["pose"] = (time.perf_counter() - t0) * 1000.0

        frame_index = self._frame_index

        if frame is None:
            return PipelineFrame(
                frame_index=frame_index,
                timestamp=now,
                latency_ms=latency_ms,
                lost_cameras=lost_cameras,
            )

        self.last_frame = frame
        # What this capture measured, kept so the lagged analysis frame can be
        # paired with its own 2D pose and leg measurements.
        missing_keypoints = self._unmeasured_leg_keypoints(raw_3d)
        self._recent_captures.append((now, skeleton_2d, missing_keypoints))

        # Hip-centred view of the measured skeleton for the gates and the
        # BiLSTM. Triangulated skeletons arrive in the world frame; MediaPipe
        # ones are hip-centred already. Missing hips = no skeleton this tick.
        raw_centred: Skeleton3D | None = None
        if raw_3d is not None:
            raw_centred = recentre_at_hips(raw_3d) if self._multi_camera else raw_3d

        # Presence-only mode: return after pose estimation so rest
        # periods keep detecting the user without advancing gates,
        # tracking state, or collecting any data.
        if self.presence_only:
            return PipelineFrame(
                frame_index=frame_index,
                timestamp=now,
                skeleton_2d=skeleton_2d,
                skeleton_3d_raw=raw_centred,
                latency_ms=latency_ms,
                missing_keypoints=missing_keypoints,
                lost_cameras=lost_cameras,
            )

        # The display smoother must never leak into diagnosis: the
        # single-camera valgus estimator reads the raw 2D kept above.
        if skeleton_2d is not None and self._display_smoother is not None:
            skeleton_2d = self._display_smoother.smooth(skeleton_2d)

        # --- Barbell detection + tracking (independent of pose) ---
        if self._barbell_detector is not None:
            t0 = time.perf_counter()
            try:
                bar_detection = self._barbell_detector.detect(
                    frame, timestamp=now, frame_index=frame_index
                )
            except Exception:
                bar_detection = None
            if self._bar_tracker is not None:
                bar_track = self._bar_tracker.update(bar_detection, timestamp=now)
            latency_ms["barbell"] = (time.perf_counter() - t0) * 1000.0

        # --- BiLSTM rep counting + gates: measured skeletons only, never
        # predicted ones ---
        bilstm_rep_data = None
        bilstm_shallow_class = None
        bilstm_prob = None
        bilstm_depth_class = None
        bilstm_depth_class_name = None
        bilstm_class_probs = None
        if raw_centred is not None:
            if self._bilstm is not None:
                t0 = time.perf_counter()
                bilstm_rep_data, bilstm_shallow_class = self._bilstm.process_skeleton(raw_centred)
                bilstm_prob = self._bilstm.current_probability
                bilstm_depth_class = self._bilstm.current_depth_class
                bilstm_depth_class_name = DEPTH_CLASS_NAMES.get(bilstm_depth_class, "Unknown")
                bilstm_class_probs = self._bilstm.current_class_probabilities.tolist()
                latency_ms["bilstm"] = (time.perf_counter() - t0) * 1000.0

            # Standing pose gate runs every frame, unconditionally; the
            # readiness gate is per-set and resets between sets.
            was_ready = self._readiness_gate.is_ready
            self._standing_gate.check(raw_centred)
            self._readiness_gate.check(raw_centred)
            if not self._multi_camera and not was_ready:
                self._track_camera_pitch(raw_centred)

        if not self._readiness_gate.is_ready:
            return PipelineFrame(
                frame_index=frame_index,
                timestamp=now,
                skeleton_2d=skeleton_2d,
                skeleton_3d_raw=raw_centred,
                bar_detection=bar_detection,
                bar_track=bar_track,
                latency_ms=latency_ms,
                missing_keypoints=missing_keypoints,
                lost_cameras=lost_cameras,
            )

        # --- Pre-IK chain: analysis continues through short dropouts ---
        t0 = time.perf_counter()
        if raw_centred is not None:
            result = self._preik.run(raw_3d)
        else:
            # On the capture clock: a processing-time stamp would run ahead of
            # the next capture, and the Kalman drops that frame as out of order.
            result = self._preik.predict_missing(now)
        latency_ms["pre_ik"] = (time.perf_counter() - t0) * 1000.0

        if result is None:
            return PipelineFrame(
                frame_index=frame_index,
                timestamp=now,
                skeleton_2d=skeleton_2d,
                skeleton_3d_raw=raw_centred,
                bar_detection=bar_detection,
                bar_track=bar_track,
                latency_ms=latency_ms,
                missing_keypoints=missing_keypoints,
                lost_cameras=lost_cameras,
            )

        analysis = result.analysis
        predicted_frame = raw_centred is None
        # The analysis frame lags the capture: pair it with what was measured
        # when it was captured. Leg keypoints the Kalman carried rather than
        # measured make the frame a guess, which never enters the rep features.
        analysed_capture = self._capture_at(analysis.timestamp)
        if analysed_capture is None:
            analysis_2d, extrapolated_keypoints = None, list(_LEG_KEYPOINT_NAMES.values())
        else:
            _, analysis_2d, extrapolated_keypoints = analysed_capture

        if not self.body_calibration.is_complete and not predicted_frame:
            self._record_body_measurements(result)

        # --- IK solve ---
        t0 = time.perf_counter()
        angles = self._ik_solver.solve(analysis)

        # Mode-aware valgus estimation (the analysed frame's raw 2D pose, or 3D)
        vr = self._valgus_estimator.estimate(
            None if self._multi_camera else analysis_2d, analysis,
        )
        angles.knee_valgus_l = vr.valgus_l
        angles.knee_valgus_r = vr.valgus_r
        angles.foot_confidence_l = vr.foot_confidence_l
        angles.foot_confidence_r = vr.foot_confidence_r
        angles.knee_ankle_sep_ratio = vr.kasr
        angles.hip_rotation_l = vr.hip_rotation_l
        angles.hip_rotation_r = vr.hip_rotation_r

        if self._inspecting:
            self._inspect_raw_angles = angles
        knee_flexion = self._knee_flexion_median.update(angles.avg_knee_flexion)
        latency_ms["ik"] = (time.perf_counter() - t0) * 1000.0

        # --- Compute rep signal (exercise-specific, from profile) ---
        rep_signal = self._profile.get_rep_signal(analysis, angles)

        # --- One measured sample per frame (analysis.rep_features) ---
        foot_state = result.foot_state
        sample = build_frame_sample(
            analysis,
            angles,
            phase=self._rep_counter.phase,
            femur_m=self._femur_length_m(),
            heel_rise_l_cm=foot_state.heel_rise_l_cm if foot_state is not None and foot_state.valid else math.nan,
            heel_rise_r_cm=foot_state.heel_rise_r_cm if foot_state is not None and foot_state.valid else math.nan,
            bar_detected=bar_detection is not None,
        )

        # --- Buffer standing frame: last skeleton before rep starts ---
        # Stored frames (standing, setup, bottom) need legs measured, not carried.
        legs_present = self._legs_present(analysis) and not extrapolated_keypoints
        if not self._rep_counter.in_rep:
            if legs_present:
                self._standing_kpts = analysis.to_numpy().tolist()
                self._track_setup(analysis, sample)
            self._standing_captured = False
        elif not self._standing_captured:
            self._standing_captured = True
            self._start_rep_setup()

        # --- Buffer bottom-of-rep frame for diagnosis engine ---
        if self._rep_counter.in_rep:
            if math.isnan(self._rep_max_knee_flex) or knee_flexion > self._rep_max_knee_flex:
                self._rep_max_knee_flex = knee_flexion
            depth_ratio = sample.depth_ratio
            if math.isfinite(depth_ratio):
                # Like the trajectory, the depth gate reads measured legs only:
                # a Kalman-carried descent overshoots the real bottom.
                if not extrapolated_keypoints and not depth_ratio >= self._rep_min_depth_ratio:
                    self._rep_min_depth_ratio = depth_ratio
                if depth_ratio < self._bottom_min_depth_ratio and legs_present:
                    self._bottom_min_depth_ratio = depth_ratio
                    self._bottom_kpts = analysis.to_numpy().tolist()
                    self._bottom_angles = angles.as_dict()

            if not extrapolated_keypoints:
                self._rep_trajectory.append(sample)
        else:
            self._rep_max_knee_flex = math.nan

        # --- Fault detection + rep counting ---
        t0 = time.perf_counter()

        # Extrapolated frames keep the rep counter alive but never justify a cue.
        faults: list[FaultEvent] = []
        if not predicted_frame:
            faults = self._rule_engine.evaluate(
                angles,
                in_rep=self._rep_counter.in_rep,
                rep_number=self._rep_counter.rep_count + 1,
                bar_detection=bar_detection,
                derivatives=None,
                phase=self._rep_counter.phase,
                foot_state=result.foot_state,
            )

        # Rep counter uses profile-provided signal for state, angles for
        # metrics, and the analysis clock so velocities match the capture.
        was_in_rep = self._rep_counter.in_rep
        rep_data, feedback = self._rep_counter.update(
            signal_value=rep_signal,
            timestamp=analysis.timestamp,
            angles=angles,
            faults=faults,
        )
        # A rep that ended short of lockout and turned straight back down: the
        # next rep began on this frame, and its turnaround is its setup. The
        # ended rep may have been too short to return anything, so ask the counter.
        next_rep_started = was_in_rep and self._rep_counter.rep_started

        shallow_rep_class: int | None = None

        # A completed movement whose hip never reached the athlete's depth
        # target is not a rep — and the depth fault says why.
        if (
            rep_data is not None
            and self._bilstm is None
            and self._depth_gated
            and not self._rule_engine.reaches_depth_target(self._rep_min_depth_ratio)
        ):
            self._rep_counter.reject_last_rep()
            self._rule_engine.discard_rep()
            depth_faults, shallow_rep_class = self._shallow_descent(
                self._rep_min_depth_ratio, angles, rep_data.rep_number,
            )
            faults.extend(depth_faults)
            rep_data = None

        if rep_data is not None:
            # The rep's depth is the robust statistic, not the counter's
            # single-frame max.
            # NaN when no frame of the rep had both knees; the depth rule skips it.
            rep_data.max_depth_angle = self._rep_max_knee_flex
            # Whole-rep features: the same numbers drive this rep's verdicts
            # and, through RepData, the set diagnosis.
            features = self._rep_features(rep_data.rep_number)
            rep_data.features = features.model_dump()
            rep_data.depth_target_met = self._rule_engine.reaches_depth_target(features.depth_ratio)
            rep_faults = self._rule_engine.finish_rep(angles, rep_data.rep_number, features)
            faults.extend(rep_faults)
            rep_data.faults.extend(rep_faults)
            # Profiles without a depth target judge depth from the knee angle.
            if self._bilstm is None and not self._depth_gated:
                depth_faults = self._rule_engine.evaluate_rep_complete(
                    rep_data.max_depth_angle, angles, rep_data.rep_number
                )
                faults.extend(depth_faults)

            # When BiLSTM is active the hip counter's rep_data is suppressed
            # (line below), but it has the correct faults.  Stash them so the
            # BiLSTM path can pick them up via _pending_bilstm_faults.
            if self._bilstm is not None:
                self._pending_bilstm_faults.extend(rep_data.faults)
                self._pending_bilstm_features = features
        elif feedback == "go_deeper" and self._bilstm is None:
            self._rule_engine.discard_rep()
            # A descent rejected for depth produces no rep; the depth fault
            # is the only thing telling the lifter why nothing was counted.
            if self._depth_gated:
                depth_faults, shallow_rep_class = self._shallow_descent(
                    self._rep_min_depth_ratio, angles, self._rep_counter.rep_count + 1,
                )
                faults.extend(depth_faults)
            else:
                rejected_depth = (
                    self._rep_max_knee_flex
                    if not math.isnan(self._rep_max_knee_flex)
                    else self._rep_counter.rejected_rep_max_depth_angle
                )
                faults.extend(
                    self._rule_engine.evaluate_rep_complete(
                        rejected_depth, angles, self._rep_counter.rep_count + 1,
                    )
                )

        if next_rep_started:
            if legs_present:
                self._track_setup(analysis, sample)
            self._start_rep_setup()
            self._rep_max_knee_flex = math.nan
            if not extrapolated_keypoints:
                self._rep_trajectory.append(sample)

        latency_ms["faults"] = (time.perf_counter() - t0) * 1000.0

        # Track max knee flexion during BiLSTM rep windows independently
        # of the hip counter, which may be at a different phase. Tracking
        # starts as soon as the lifter leaves standing so shallow reps —
        # which never open a rep window — still get a depth angle.
        if self._bilstm is not None and (
            self._bilstm.in_rep or self._bilstm.current_depth_class > 0
        ):
            if knee_flexion > self._bilstm_max_knee_flex:
                self._bilstm_max_knee_flex = knee_flexion
            if knee_flexion < self._bilstm_min_knee_flex:
                self._bilstm_min_knee_flex = knee_flexion
            if (
                not extrapolated_keypoints
                and math.isfinite(sample.depth_ratio)
                and not sample.depth_ratio >= self._bilstm_min_depth_ratio
            ):
                self._bilstm_min_depth_ratio = sample.depth_ratio

        # The BiLSTM segments the movement; the athlete's depth target
        # decides whether it was a rep.
        if (
            bilstm_rep_data is not None
            and self._depth_gated
            and not self._rule_engine.reaches_depth_target(self._bilstm_min_depth_ratio)
        ):
            self._bilstm.reject_last_rep()
            depth_faults, shallow_rep_class = self._shallow_descent(
                self._bilstm_min_depth_ratio, angles, bilstm_rep_data.rep_number,
            )
            faults.extend(depth_faults)
            self._pending_bilstm_faults.clear()
            self._pending_bilstm_features = None
            self._rep_counter.clear_current_faults()
            self._bilstm_max_knee_flex = 0.0
            self._bilstm_min_knee_flex = 180.0
            self._bilstm_min_depth_ratio = math.nan
            bilstm_rep_data = None

        # Use BiLSTM rep data as primary when enabled and available.
        # When BiLSTM is active, suppress rule-based rep events to prevent
        # double-counting when the two counters fire on different frames.
        final_rep_data = rep_data if self._bilstm is None else None
        if self._bilstm is not None and bilstm_rep_data is not None:
            # Enrich BiLSTM RepData with rule-based metrics so downstream
            # consumers (IPC bridge, coaching LLM, set reports) get real
            # angle data, faults, timing, and asymmetry values.
            metrics = self._rep_counter.snapshot_rep_metrics(now=analysis.timestamp)

            # Use our independently-tracked knee flexion for depth, since
            # the hip counter's snapshot may be desync'd from the BiLSTM's
            # rep boundaries (causing false "quarter" depth classifications).
            bilstm_rep_data.max_depth_angle = self._bilstm_max_knee_flex
            bilstm_rep_data.min_depth_angle = self._bilstm_min_knee_flex
            bilstm_rep_data.depth_target_met = self._rule_engine.reaches_depth_target(
                self._bilstm_min_depth_ratio
            )
            self._bilstm_max_knee_flex = 0.0
            self._bilstm_min_knee_flex = 180.0
            self._bilstm_min_depth_ratio = math.nan
            if self._pending_bilstm_features is not None:
                bilstm_rep_data.features = self._pending_bilstm_features.model_dump()
                self._pending_bilstm_features = None

            bilstm_rep_data.descent_time = metrics["descent_time"]
            bilstm_rep_data.ascent_time = metrics["ascent_time"]
            # Combine buffered faults (from hip counter rep completions) with
            # any still in the counter's accumulator.  The hip counter clears
            # _current_faults on rep completion, so metrics["faults"] is often
            # empty — the buffered list has the real data.
            bilstm_rep_data.faults = self._pending_bilstm_faults + metrics["faults"]
            self._pending_bilstm_faults.clear()
            bilstm_rep_data.avg_knee_asymmetry = metrics["avg_knee_asymmetry"]
            bilstm_rep_data.avg_hip_asymmetry = metrics["avg_hip_asymmetry"]
            self._rep_counter.clear_current_faults()

            # Profiles without a depth target judge depth from the knee angle.
            if not self._depth_gated:
                depth_faults = self._rule_engine.evaluate_rep_complete(
                    bilstm_rep_data.max_depth_angle, angles, bilstm_rep_data.rep_number
                )
                faults.extend(depth_faults)
                bilstm_rep_data.faults.extend(depth_faults)

            final_rep_data = bilstm_rep_data

        # A descent the BiLSTM never opened a rep for. It produces no rep, so
        # the depth fault is the only thing telling the lifter why nothing
        # was counted.
        if bilstm_shallow_class is not None:
            if self._depth_gated:
                depth_faults, shallow_rep_class = self._shallow_descent(
                    self._bilstm_min_depth_ratio, angles, self._bilstm.rep_count + 1,
                )
                faults.extend(depth_faults)
            else:
                shallow_rep_class = bilstm_shallow_class
            self._bilstm_max_knee_flex = 0.0
            self._bilstm_min_knee_flex = 180.0
            self._bilstm_min_depth_ratio = math.nan

        return PipelineFrame(
            frame_index=frame_index,
            timestamp=now,
            skeleton_2d=skeleton_2d,
            skeleton_3d=analysis,
            skeleton_3d_raw=raw_centred,
            skeleton_3d_display=result.display,
            foot_state=result.foot_state,
            joint_angles=angles,
            faults=faults,
            rep_data=final_rep_data,
            bilstm_probability=bilstm_prob,
            bilstm_rep_data=bilstm_rep_data,
            bilstm_depth_class=bilstm_depth_class,
            bilstm_depth_class_name=bilstm_depth_class_name,
            bilstm_class_probabilities=bilstm_class_probs,
            shallow_rep_class=shallow_rep_class,
            bar_detection=bar_detection,
            bar_track=bar_track,
            latency_ms=latency_ms,
            missing_keypoints=missing_keypoints,
            extrapolated_keypoints=extrapolated_keypoints,
            lost_cameras=lost_cameras,
        )

    def release(self):
        """Release all resources."""
        if self._multi_camera and self._multi_camera_provider is not None:
            self._multi_camera_provider.release()
        else:
            self._capture_running = False
            if self._capture_thread is not None:
                self._capture_thread.join(timeout=2.0)
            if self._cap is not None:
                self._cap.release()
        if self._pose_estimator is not None:
            self._pose_estimator.release()
