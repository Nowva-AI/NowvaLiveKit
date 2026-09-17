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

from biomechanics.config import BiomechanicsConfig
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

# A stored bottom/standing frame is only useful with the whole lower body present.
_LEG_KEYPOINTS = (
    CocoKeypoints.LEFT_HIP, CocoKeypoints.RIGHT_HIP,
    CocoKeypoints.LEFT_KNEE, CocoKeypoints.RIGHT_KNEE,
    CocoKeypoints.LEFT_ANKLE, CocoKeypoints.RIGHT_ANKLE,
)

# With no post-IK angle filter, a per-rep max over raw frames is a
# single-frame statistic that one transient can over-read by 7–10°. Every
# rep extremum (bottom frame, depth class, BiLSTM window) is taken over a
# running median of this many frames instead.
KNEE_FLEXION_MEDIAN_FRAMES = 3


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

        # Layer 1: Capture (threaded — always holds the latest frame)
        self._cap = None
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._capture_running = False
        self._capture_thread = None

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
            if tri.calibration_file:
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
        self._rule_engine.set_profile(self._profile)

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

        # Robust per-frame knee flexion: running median feeding every rep
        # extremum below. Per-set temporal state.
        self._knee_flexion_median = RunningMedian(window_frames=KNEE_FLEXION_MEDIAN_FRAMES)
        # Max of the median over the current rep window; what the depth rule
        # and RepData report as the rep's depth.
        self._rep_max_knee_flex: float = math.nan

        # Bottom-of-rep buffer for diagnosis engine.
        # Tracks the frame with max median knee flexion during each rep.
        self._bottom_max_knee_flex: float = 0.0
        self._bottom_kpts: list[list[float]] | None = None
        self._bottom_angles: dict | None = None

        # Standing-frame buffer: last skeleton before in_rep becomes True.
        self._standing_kpts: list[list[float]] | None = None
        self._standing_captured: bool = False

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
        self._pending_bilstm_faults.clear()
        self._bottom_max_knee_flex = 0.0
        self._bottom_kpts = None
        self._bottom_angles = None
        self._standing_kpts = None
        self._standing_captured = False
        self._rep_trajectory.clear()

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
        A new camera calibration was installed in the provider. Temporal state always
        restarts; when the world frame itself moved (first calibration, board -> person
        re-anchor) the foot contact anchors and floor go too. Body measurements are
        lengths, so they survive either way.
        """
        if world_frame_changed:
            self._preik.reset_world_state()
        else:
            self._preik.reset()
        if self._multi_camera_provider is not None:
            self._multi_camera_provider.reset_temporal_state()

    def apply_athlete_params(self, params: dict) -> None:
        """Adopt a returning user's stored body measurements and scale thresholds once."""
        self.body_calibration = SegmentLengthEstimator.from_athlete_params(params)
        self._apply_body_proportions()

    def _apply_body_proportions(self) -> None:
        proportions = self.body_calibration.body_proportions
        if proportions is None:
            return
        self._rule_engine.apply_body_proportion_scaling(proportions)
        logger.info(
            "[PIPELINE] Body proportions applied: femur=%.3fm torso=%.3fm lean_scale=%.2f",
            proportions.femur_length_avg, proportions.torso_length_avg, proportions.forward_lean_scale,
        )

    def consume_bottom_frame(self) -> tuple[list[list[float]] | None, dict | None]:
        """Return and reset the bottom-of-rep keypoints and angles.

        Returns (bottom_kpts, bottom_angles) captured at max knee flexion
        during the most recent rep, then clears the buffer for the next rep.
        """
        kpts = self._bottom_kpts
        angles = self._bottom_angles
        self._bottom_max_knee_flex = 0.0
        self._bottom_kpts = None
        self._bottom_angles = None
        return kpts, angles

    def consume_standing_frame(self) -> list[list[float]] | None:
        """Return the standing keypoints captured just before the rep started."""
        kpts = self._standing_kpts
        self._standing_captured = False
        return kpts

    @staticmethod
    def _build_trajectory_sample(
        skeleton_3d: Skeleton3D, angles: JointAngles
    ) -> dict:
        """One frame of scoring input, taken as measured — no re-grounding. NaN angles pass through."""
        kpts = skeleton_3d.to_numpy()
        keypoints = skeleton_3d.keypoints

        def _height_cm(idx: int) -> float:
            if keypoints[idx].confidence <= 0.0:
                return float("nan")
            return float(kpts[idx][1]) * 100.0

        return {
            "trunk_pitch": 180.0 - angles.trunk_flexion,
            "knee_valgus_l": angles.knee_valgus_l,
            "knee_valgus_r": angles.knee_valgus_r,
            "hip_y_l": _height_cm(CocoKeypoints.LEFT_HIP),
            "hip_y_r": _height_cm(CocoKeypoints.RIGHT_HIP),
            "knee_y_l": _height_cm(CocoKeypoints.LEFT_KNEE),
            "knee_y_r": _height_cm(CocoKeypoints.RIGHT_KNEE),
        }

    @staticmethod
    def _legs_present(skeleton_3d: Skeleton3D) -> bool:
        return all(
            skeleton_3d.keypoints[idx].confidence > 0.0 for idx in _LEG_KEYPOINTS
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

    def _capture_loop(self) -> None:
        """Continuously read frames from the camera in a background thread."""
        while self._capture_running:
            ret, frame = self._cap.read()
            if ret and frame is not None:
                with self._frame_lock:
                    self._latest_frame = frame

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

        if self._multi_camera and self._multi_camera_provider is not None:
            t0 = time.perf_counter()
            frame, skeleton_2d, raw_3d = self._multi_camera_provider.get_pose()
            latency_ms["capture"] = 0.0
            latency_ms["pose"] = (time.perf_counter() - t0) * 1000.0
            if raw_3d is not None:
                # The skeleton carries the primary camera's capture sequence
                # and timestamp; never overwrite them.
                self._frame_index = raw_3d.frame_index
        else:
            self._frame_index += 1
            t0 = time.perf_counter()
            with self._frame_lock:
                frame = self._latest_frame
            latency_ms["capture"] = (time.perf_counter() - t0) * 1000.0

            if frame is not None:
                t0 = time.perf_counter()
                try:
                    skeleton_2d, raw_3d = self._pose_estimator.estimate_both(frame)
                except Exception:
                    pass
                latency_ms["pose"] = (time.perf_counter() - t0) * 1000.0

        frame_index = self._frame_index

        if frame is None:
            return PipelineFrame(
                frame_index=frame_index,
                timestamp=now,
                latency_ms=latency_ms,
            )

        self.last_frame = frame

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
            )

        # The single-camera valgus estimator reads the raw 2D pose; the
        # display smoother must never leak into diagnosis.
        skeleton_2d_raw = skeleton_2d
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
            self._standing_gate.check(raw_centred)
            self._readiness_gate.check(raw_centred)

        if not self._readiness_gate.is_ready:
            return PipelineFrame(
                frame_index=frame_index,
                timestamp=now,
                skeleton_2d=skeleton_2d,
                skeleton_3d_raw=raw_centred,
                bar_detection=bar_detection,
                bar_track=bar_track,
                latency_ms=latency_ms,
            )

        # --- Pre-IK chain: analysis continues through short dropouts ---
        t0 = time.perf_counter()
        if raw_centred is not None:
            result = self._preik.run(raw_3d)
        else:
            result = self._preik.predict_missing(raw_3d.timestamp if raw_3d is not None else time.time())
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
            )

        analysis = result.analysis
        predicted_frame = raw_centred is None

        if not self.body_calibration.is_complete and not predicted_frame:
            self._record_body_measurements(result)

        # --- IK solve ---
        t0 = time.perf_counter()
        angles = self._ik_solver.solve(analysis)

        # Mode-aware valgus estimation (2D FPPA on the raw 2D pose, or 3D)
        vr = self._valgus_estimator.estimate(
            None if self._multi_camera else skeleton_2d_raw, analysis,
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

        # --- Buffer standing frame: last skeleton before rep starts ---
        legs_present = self._legs_present(analysis)
        if not self._rep_counter.in_rep:
            if legs_present:
                self._standing_kpts = analysis.to_numpy().tolist()
            self._standing_captured = False
        elif not self._standing_captured:
            self._standing_captured = True

        # --- Buffer bottom-of-rep frame for diagnosis engine ---
        if self._rep_counter.in_rep:
            if math.isnan(self._rep_max_knee_flex) or knee_flexion > self._rep_max_knee_flex:
                self._rep_max_knee_flex = knee_flexion
            if knee_flexion > self._bottom_max_knee_flex and legs_present:
                self._bottom_max_knee_flex = knee_flexion
                self._bottom_kpts = analysis.to_numpy().tolist()
                self._bottom_angles = angles.as_dict()

            self._rep_trajectory.append(
                self._build_trajectory_sample(analysis, angles)
            )
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

        if not self._rule_engine.calibrated and self._rep_counter.in_rep:
            self._rule_engine.record_frame_for_calibration(angles)

        # Rep counter uses profile-provided signal for state, angles for
        # metrics, and the analysis clock so velocities match the capture.
        rep_data, feedback = self._rep_counter.update(
            signal_value=rep_signal,
            timestamp=analysis.timestamp,
            angles=angles,
            faults=faults,
        )

        # If rep completed, check depth faults and advance calibration
        if rep_data is not None:
            # The rep's depth is the robust statistic, not the counter's
            # single-frame max.
            # NaN when no frame of the rep had both knees; the depth rule skips it.
            rep_data.max_depth_angle = self._rep_max_knee_flex
            # Per-rep verdicts (bilateral asymmetry) belong to this rep.
            rep_faults = self._rule_engine.finish_rep(angles, rep_data.rep_number)
            faults.extend(rep_faults)
            rep_data.faults.extend(rep_faults)
            # Only evaluate depth here when BiLSTM is NOT active.
            # When BiLSTM is active, depth evaluation happens in the BiLSTM
            # path below to avoid double-counting.
            if self._bilstm is None:
                depth_faults = self._rule_engine.evaluate_rep_complete(
                    rep_data.max_depth_angle, angles, rep_data.rep_number
                )
                faults.extend(depth_faults)
            self._rule_engine.on_rep_complete_calibration(is_clean=rep_data.is_clean)

            # When BiLSTM is active the hip counter's rep_data is suppressed
            # (line below), but it has the correct faults.  Stash them so the
            # BiLSTM path can pick them up via _pending_bilstm_faults.
            if self._bilstm is not None:
                self._pending_bilstm_faults.extend(rep_data.faults)
        elif feedback == "go_deeper" and self._bilstm is None:
            self._rule_engine.discard_rep()
            # A descent rejected for depth produces no rep; the depth fault
            # is the only thing telling the lifter why nothing was counted.
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
            self._bilstm_max_knee_flex = 0.0
            self._bilstm_min_knee_flex = 180.0

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

            # Evaluate depth faults for BiLSTM reps (rule-based only runs
            # this when its own counter fires, which may not align)
            depth_faults = self._rule_engine.evaluate_rep_complete(
                bilstm_rep_data.max_depth_angle, angles, bilstm_rep_data.rep_number
            )
            faults.extend(depth_faults)
            bilstm_rep_data.faults.extend(depth_faults)
            self._rule_engine.on_rep_complete_calibration(is_clean=bilstm_rep_data.is_clean)

            final_rep_data = bilstm_rep_data

        # A descent the counter rejected for depth. It produces no rep, so
        # the depth fault is the only thing telling the lifter why nothing
        # was counted — emit it from the same depth class that rejected it.
        if bilstm_shallow_class is not None:
            faults.extend(
                self._rule_engine.evaluate_shallow_rep(
                    max_depth_class=bilstm_shallow_class,
                    angles=angles,
                    rep_number=self._bilstm.rep_count + 1,
                    max_knee_flexion=self._bilstm_max_knee_flex,
                )
            )
            self._bilstm_max_knee_flex = 0.0
            self._bilstm_min_knee_flex = 180.0

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
            shallow_rep_class=bilstm_shallow_class,
            bar_detection=bar_detection,
            bar_track=bar_track,
            latency_ms=latency_ms,
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
