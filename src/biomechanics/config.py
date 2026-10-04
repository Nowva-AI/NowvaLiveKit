"""
Pipeline Configuration for Biomechanics System

Loads configuration from YAML files and provides typed access to settings.
"""

import os
from typing import Any, List, Optional, Tuple
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


# =============================================================================
# SUB-CONFIGURATIONS
# =============================================================================

class PipelineConfig(BaseModel):
    """Top-level pipeline configuration."""
    target_fps: int = 30
    log_level: str = "INFO"


class CaptureConfig(BaseModel):
    """Camera capture configuration."""
    source: str = "webcam"
    device_id: int = 0
    resolution: Tuple[int, int] = (1280, 720)


class PoseConfig(BaseModel):
    """Pose estimation configuration."""
    backend: str = "mediapipe"  # mediapipe | rtmpose
    confidence_threshold: float = 0.3
    model_complexity: int = 2  # 0=lite, 1=full, 2=heavy (mediapipe only)
    model_path: Optional[str] = None  # Path to ONNX model (rtmpose only)
    keypoint_format: str = "coco17"  # coco17 | halpe26


class TriangulationConfig(BaseModel):
    """Stereo triangulation configuration (multi-camera mode is NOWVA_MULTI_CAMERA)."""
    device_ids: List[int] = [0, 1, 2]
    primary_camera: int = 0
    max_sync_delta_ms: float = 20.0
    focal_length_factor: float = 0.8
    min_views: int = 2
    max_reprojection_error: float = 15.0
    calibration_file: Optional[str] = None
    tpose_capture_frames: int = 30


class CameraCalibrationConfig(BaseModel):
    """Person-based camera calibration: capture buffer, bootstrap window and drift monitor (multi-camera only)."""
    # Device id -> name of its ~/.nowva/intrinsics_<key>.json (default: the device id).
    camera_keys: dict[int, str] = Field(default_factory=dict)
    # Ring buffer of per-frame 2D views (>= 2 cameras) kept by the provider.
    calibration_buffer_frames: int = 450
    # Barbell as a metric ruler during a capture window, even with barbell_tracking off.
    use_bar_scale: bool = False
    bar_detection_stride: int = 3
    # Bootstrap (no calibration file): capture until both are met, or the timeout.
    min_calibration_frames: int = 240
    min_squat_excursions: int = 2
    capture_timeout_s: float = 90.0
    # Drift monitor: refine between sets when health exceeds drift_ratio x the stored RMS.
    drift_ratio: float = 1.5
    drift_check_frames: int = 150


class DepthFaultConfig(BaseModel):
    """Squat depth as hip height above the knee, in femur lengths (0 = parallel, + = above).

    A rep counts when it reaches the athlete's target within the tolerance.
    The target is the athlete's assessed capacity, never shallower than
    max_target_ratio and never asked deeper than default_target_ratio
    (parallel); uncalibrated_target_ratio applies until calibration runs.
    """
    default_target_ratio: float = 0.0
    # Before calibration measures the athlete, a lenient start: refusing to
    # count reps of someone not yet measured is worse than counting a high one.
    uncalibrated_target_ratio: float = 0.2
    max_target_ratio: float = 0.5
    tolerance_ratio: float = 0.08
    moderate_deficit_ratio: float = 0.25
    severe_deficit_ratio: float = 0.5


class BilateralAsymmetryConfig(BaseModel):
    """Bilateral asymmetry fault thresholds."""
    mild: float = 8.0
    moderate: float = 13.0
    severe: float = 18.0


class ForwardLeanConfig(BaseModel):
    """Forward lean fault thresholds (180-convention: lower = more lean)."""
    mild: float = 135.0
    moderate: float = 125.0
    severe: float = 115.0


class KneeValgusConfig(BaseModel):
    """Knees caving inside the toe line: femur deviation (deg) from the knees-over-toes
    plane. One scale in both capture modes — 8° is ~6 cm of inward knee travel."""
    mild: float = 8.0
    moderate: float = 13.0
    severe: float = 18.0


class HeelRiseConfig(BaseModel):
    """Heel-rise fault thresholds (ankle rise above its planted floor reference; multi-camera only)."""
    mild_cm: float = 1.5
    moderate_cm: float = 3.0
    severe_cm: float = 5.0
    cooldown_s: float = 2.0
    min_rise_duration_s: float = 0.23


class HipShootConfig(BaseModel):
    """Chest dropping (deg of extra trunk pitch) while the hip recovers its first third."""
    mild: float = 8.0
    moderate: float = 12.0
    severe: float = 18.0


class HipShiftConfig(BaseModel):
    """Sideways pelvis travel during a rep, as a fraction of ankle separation
    (0.10 ≈ 3.5 cm on a 35 cm stance). Mild sits above the recorded
    single-camera noise floor (median 0.066)."""
    mild: float = 0.10
    moderate: float = 0.15
    severe: float = 0.22


class BalanceConfig(BaseModel):
    """Load offset from midfoot at the bottom, as a fraction of ankle-to-toe length."""
    forward_mild: float = 0.20
    forward_moderate: float = 0.30
    forward_severe: float = 0.40
    backward_mild: float = 0.30
    backward_moderate: float = 0.40
    backward_severe: float = 0.50


class DepthDriftConfig(BaseModel):
    """Rep shallower than the session's reference depth (median of its 3 deepest
    reps), in femur lengths. Within-set depth noise is ~0.1 on one camera."""
    mild: float = 0.15
    moderate: float = 0.25
    severe: float = 0.35


class LockoutConfig(BaseModel):
    """Top-of-rep hip height short of full standing, as a fraction of leg length."""
    mild: float = 0.05
    moderate: float = 0.08
    severe: float = 0.12


class DescentControlConfig(BaseModel):
    """Descents shorter than these (seconds) are dive-bombed."""
    mild_seconds: float = 0.6
    moderate_seconds: float = 0.45
    severe_seconds: float = 0.3


class VelocityLossConfig(BaseModel):
    """Concentric velocity loss (%) against the set's fastest rep."""
    mild_pct: float = 20.0
    moderate_pct: float = 30.0
    severe_pct: float = 40.0


class FootPlacementConfig(BaseModel):
    """Setup asymmetry: front-back stagger (fraction of ankle-to-toe length) and toe-out difference (deg)."""
    stagger_mild: float = 0.15
    stagger_moderate: float = 0.25
    stagger_severe: float = 0.35
    flare_mild_deg: float = 10.0
    flare_moderate_deg: float = 15.0
    flare_severe_deg: float = 22.0


class DeadliftFaultConfig(BaseModel):
    """One deadlift fault's mild / moderate / severe thresholds and the lowest tier
    that may be cued (PLAN.md §2.9: a tier is cued only once its measurement error
    is proven below a third of its threshold). min_tier "recap" is never cued."""
    mild: float
    moderate: float
    severe: float
    min_tier: str = "moderate"


def _deadlift_fault(mild: float, moderate: float, severe: float, min_tier: str) -> Any:
    return Field(default_factory=lambda: DeadliftFaultConfig(
        mild=mild, moderate=moderate, severe=severe, min_tier=min_tier,
    ))


class FaultsConfig(BaseModel):
    """Fault detection configuration."""
    depth: DepthFaultConfig = Field(default_factory=DepthFaultConfig)
    bilateral_asymmetry: BilateralAsymmetryConfig = Field(default_factory=BilateralAsymmetryConfig)
    forward_lean: ForwardLeanConfig = Field(default_factory=ForwardLeanConfig)
    knee_valgus: KneeValgusConfig = Field(default_factory=KneeValgusConfig)
    heel_rise: HeelRiseConfig = Field(default_factory=HeelRiseConfig)
    hip_shoot: HipShootConfig = Field(default_factory=HipShootConfig)
    hip_shift: HipShiftConfig = Field(default_factory=HipShiftConfig)
    balance: BalanceConfig = Field(default_factory=BalanceConfig)
    depth_drift: DepthDriftConfig = Field(default_factory=DepthDriftConfig)
    lockout: LockoutConfig = Field(default_factory=LockoutConfig)
    descent_control: DescentControlConfig = Field(default_factory=DescentControlConfig)
    velocity_loss: VelocityLossConfig = Field(default_factory=VelocityLossConfig)
    foot_placement: FootPlacementConfig = Field(default_factory=FootPlacementConfig)
    # Conventional deadlift (PLAN.md §2.6). Initial values; J6 re-sets thresholds
    # and min tiers from real data. Units: cm (D1, D4, D7, D3, D8b), deg (D2, D6,
    # D5, D9), ratio of ankle separation (D8), percent (D10).
    deadlift_bar_position: DeadliftFaultConfig = _deadlift_fault(3.0, 5.0, 8.0, "mild")
    # Outside the setup model's hip band. Severe-only until the model's P95 error
    # (model + hip-keypoint bias) is measured (§2.7).
    deadlift_setup_hips: DeadliftFaultConfig = _deadlift_fault(4.0, 7.0, 10.0, "severe")
    deadlift_shoulders_behind: DeadliftFaultConfig = _deadlift_fault(2.0, 4.0, 6.0, "moderate")
    # Lower than the plan's first 10/15/20: with the bar at the knees, straight legs
    # cap the excess at ~8-12 deg for typical bodies (simulator, J2), so 10/15/20
    # could almost never reach moderate. Mild stays uncued (error budget §2.9). The
    # sizes come from the unvalidated setup model, so the rule fires only when the
    # hips decisively out-rose the shoulders (faults/rules/deadlift_hips_shoot.py).
    deadlift_hips_shoot: DeadliftFaultConfig = _deadlift_fault(5.0, 8.0, 11.0, "moderate")
    deadlift_bar_drift: DeadliftFaultConfig = _deadlift_fault(3.0, 5.0, 8.0, "moderate")
    deadlift_lockout: DeadliftFaultConfig = _deadlift_fault(8.0, 12.0, 20.0, "moderate")
    deadlift_lean_back: DeadliftFaultConfig = _deadlift_fault(8.0, 12.0, 18.0, "moderate")
    deadlift_hip_shift: DeadliftFaultConfig = _deadlift_fault(0.10, 0.15, 0.22, "moderate")
    deadlift_bar_tilt: DeadliftFaultConfig = _deadlift_fault(3.0, 5.0, 7.0, "moderate")
    deadlift_bent_arms: DeadliftFaultConfig = _deadlift_fault(15.0, 25.0, 35.0, "mild")
    # Sánchez-Medina 2011. Fatigue is load advice, never a mid-set technique cue.
    deadlift_velocity_loss: DeadliftFaultConfig = _deadlift_fault(20.0, 30.0, 40.0, "recap")
    # Bar measured from the wrists (no bar tracking): D1, D3 and D8b thresholds
    # widen by this factor (§6).
    deadlift_wrist_proxy_threshold_scale: float = 1.5


class BiLSTMConfig(BaseModel):
    """BiLSTM rep counting configuration (5-class depth classification)."""
    enabled: bool = False
    model_path: str = "models/bilstm_rep_counter.pt"
    device: str = "cpu"  # cpu | cuda | mps
    num_classes: int = 5
    min_depth_class: int = 3  # 0=standing, 1=quarter, 2=half, 3=parallel, 4=deep
    min_rep_frames: int = 12
    ema_alpha: float = 0.2


class BarbellTrackingConfig(BaseModel):
    """
    Real-time barbell detection & tracking configuration.

    Uses a YOLO11n-pose model with 2 keypoints (left end, right end of the bar)
    plus a bounding box. Smoothed with a constant-velocity Kalman filter per
    endpoint so position/velocity stay stable through detection dropouts.
    """
    enabled: bool = False
    model_path: str = "models/barbell_keypoints.pt"
    conf_threshold: float = 0.25
    imgsz: int = 640
    device: str = "auto"             # "auto" | "cpu" | "cuda" | "mps"

    # Real-world calibration: length of a standard Olympic barbell
    bar_length_m: float = 2.2

    # Kalman noise parameters (constant-velocity model, px units)
    kalman_q: float = 1e-2            # process noise
    kalman_r: float = 1.0             # measurement noise

    path_history_len: int = 120       # ~4 s at 30 FPS

    # Tilt color coding for overlay
    tilt_warn_deg: float = 2.0
    tilt_error_deg: float = 5.0

    # Tilt-based bilateral asymmetry fault thresholds
    tilt_asym_mild_deg: float = 2.0
    tilt_asym_moderate_deg: float = 4.0
    tilt_asym_severe_deg: float = 7.0
    tilt_asym_mild_cm: float = 3.0
    tilt_asym_moderate_cm: float = 6.0
    tilt_asym_severe_cm: float = 10.0


class HipPositionCounterConfig(BaseModel):
    """Hip-position-based rep counter thresholds.

    Uses the same signal as the post-hoc rep segmenter (hip_mid_y - ankle_mid_y)
    but in a causal 4-state machine suitable for real-time counting.
    """
    # Velocity thresholds (cm/s)
    entry_vel_threshold: float = 3.0        # velocity > this → STANDING→DESCENDING
    bottom_vel_threshold: float = 5.0       # abs(vel) < this → at bottom
    ascending_vel_threshold: float = 3.0    # vel < -this → BOTTOM→ASCENDING

    # Position thresholds (cm)
    min_depth_cm: float = 10.0              # minimum displacement for valid rep (prominence)
    standing_return_cm: float = 3.0         # must return within this of baseline

    # Minimum time in each state (prevents noise flipping): 3, 2 and 3 frames
    # at 30 fps. Seconds, not frames, so 12 fps capture keeps the same timing.
    min_descending_s: float = 0.1
    min_bottom_s: float = 0.067
    min_ascending_s: float = 0.1

    # Rep validation
    min_rep_duration_s: float = 0.5         # 15 frames at 30 fps


class DeadliftConfig(BaseModel):
    """Deadlift rep state machine and analyser (PLAN.md §2.3–2.4, §2.7).

    Initial values, re-set from real lifts at J6. Distances in metres, speeds in
    m/s, durations in seconds; heights are along measured gravity.
    """
    # APPROACH -> STANCE: standing settled, facing the bar, close enough to guide.
    stance_still_s: float = 0.5
    stance_max_bar_ahead_m: float = 0.40
    stance_max_bar_lateral_m: float = 0.30
    facing_max_deg: float = 60.0
    # The bar behind the midfoot by more than this is behind the lifter.
    max_bar_behind_m: float = 0.10
    # STANCE -> SETUP: hands within this band around the bar for setup_hold_s. The
    # wrist joint sits ~7.5 cm above the bar centre; the band leaves room for
    # 2 cm keypoint noise either side.
    hands_max_above_bar_m: float = 0.16
    hands_max_below_bar_m: float = 0.06
    hands_max_forward_m: float = 0.15
    setup_hold_s: float = 0.3
    # A held condition survives lapses this short (a noisy frame, a dropped keypoint).
    hold_grace_s: float = 0.15
    # Without bar tracking, a setup is a hinge with the wrists below the knees.
    wrist_proxy_min_hinge_deg: float = 30.0
    # SETUP / FLOOR -> PULL and the back-dated liftoff.
    liftoff_rise_m: float = 0.03
    liftoff_velocity_mps: float = 0.10
    liftoff_rest_band_m: float = 0.005
    liftoff_rest_speed_mps: float = 0.02
    liftoff_lookback_s: float = 0.5
    # PULL -> TOP.
    top_margin_m: float = 0.08
    top_still_speed_mps: float = 0.05
    top_still_frames: int = 3
    top_max_trunk_deg: float = 35.0
    # A trunk this far behind vertical is an over-extended lockout, below the
    # expected top height or not, with the legs straight: the knees bent no more
    # than this (a failed pull leaning back from mid-thigh keeps them bent ~60
    # deg) or, the knees hidden, the hips no more than this closer to the ankles
    # than standing (~45 deg of knee bend; 60 deg shortens the leg ~12 cm).
    overextended_top_deg: float = 10.0
    overextended_max_knee_deg: float = 40.0
    overextended_max_leg_shortening_m: float = 0.06
    # TOP -> PULL: the bar rose past the top it held, which was a stall short of
    # lockout only if the hips or knees held this far short of standing (D6's
    # mild threshold); a shrug or a settling lockout rises too.
    resume_min_deficit_deg: float = 8.0
    # A pull that rose this far but never reached the top is a failed rep.
    failed_rep_min_rise_m: float = 0.10
    # TOP -> LOWER.
    lower_velocity_mps: float = 0.10
    # LOWER -> FLOOR (dead stop).
    floor_band_m: float = 0.02
    dead_stop_speed_mps: float = 0.02
    dead_stop_frames: int = 3
    dead_stop_hold_s: float = 0.3
    # LOWER -> PULL (touch-and-go). A bumper bounce rises no more than touch_go_rise_m.
    touch_go_band_m: float = 0.05
    touch_go_rise_m: float = 0.03
    touch_go_sustain_s: float = 0.1
    # FLOOR -> SETUP, and the setup windows judged before liftoff.
    resetup_hold_s: float = 0.3
    setup_window_s: float = 0.3
    quick_pull_window_s: float = 0.2
    min_setup_frames: int = 3
    # STANCE -> APPROACH.
    walk_away_s: float = 0.5
    # Standing references (lockout, lean-back, bent arms). "Settled" is the hips'
    # and shoulders' median displacement over body_speed_window_s: below walking,
    # well above what 2 cm (and Kalman-correlated) keypoint noise reads as.
    standing_still_s: float = 1.0
    standing_max_knee_deg: float = 20.0
    standing_max_trunk_deg: float = 20.0
    settled_speed_mps: float = 0.20
    body_speed_window_s: float = 0.5
    # The bar is still when its slope is under the speed threshold, or under this
    # many standard errors of its own noise (the wrist proxy jitters).
    still_noise_factor: float = 2.5
    # Bar velocity is the slope of its height over this window.
    velocity_window_s: float = 0.1
    # The midfoot is locked from the last still frames of STANCE.
    midfoot_lock_frames: int = 15
    # Wrist joint above the bar centre; learned per athlete when the bar is tracked.
    wrist_to_bar_offset_m: float = 0.075
    min_wrist_to_bar_offset_m: float = 0.04
    max_wrist_to_bar_offset_m: float = 0.12
    # Setup model (§2.7): bar axis to the shin line at contact (measured per lifter
    # on the setup frames, this default when it cannot be), and the shoulder band.
    shin_bar_distance_m: float = 0.05
    min_shin_bar_distance_m: float = 0.02
    max_shin_bar_distance_m: float = 0.10
    shoulder_band_low_m: float = 0.0
    shoulder_band_high_m: float = 0.06
    # A frame up to this long after the last bar state keeps that bar for its
    # geometry (hands on bar, facing), with no height: a tracking gap, not a lost
    # bar. Carried and predicted states never enter the setup features.
    max_bar_gap_s: float = 0.2


class CoachingConfig(BaseModel):
    """Coaching integration configuration."""
    min_cue_gap_seconds: float = 1.0
    set_timeout_seconds: float = 30.0
    cache_cues_before_set: bool = True


class IPCConfig(BaseModel):
    """IPC communication configuration."""
    frame_send_interval: int = 10
    fault_cooldown_seconds: float = 3.0


class KalmanSmootherConfig(BaseModel):
    """Fixed-lag constant-velocity Kalman smoother — the only temporal filter before IK."""
    lag_frames: int = 2
    process_noise: float = 10.0
    # Measurement std from triangulator confidence, clipped to [floor, ceiling].
    measurement_std_floor_m: float = 0.003
    # MediaPipe confidences carry no metric meaning, so single-camera mode uses a larger floor.
    single_camera_measurement_std_floor_m: float = 0.01
    measurement_std_ceiling_m: float = 0.08
    gate_sigma: float = 4.0
    gate_min_radius_m: float = 0.08
    # Analysis carries an unmeasured keypoint this long (5 frames at 30 fps).
    max_prediction_s: float = 0.17
    min_output_confidence: float = 0.15


class FootContactConfig(BaseModel):
    """World-frame foot contact model (multi-camera only)."""
    enabled: bool = True


class DisplayFilterConfig(BaseModel):
    """One Euro Filter for 2D skeleton overlay smoothing (display-only, pixel units)."""
    enabled: bool = True
    min_cutoff: float = 1.0
    beta: float = 0.02
    d_cutoff: float = 1.0


class StandingGateConfig(BaseModel):
    """Standing pose gate configuration."""
    min_confidence: float = 0.25
    max_knee_flexion_deg: float = 25.0
    max_trunk_flexion_deg: float = 25.0
    min_torso_length_m: float = 0.25
    max_torso_length_m: float = 0.80
    min_leg_extension_ratio: float = 0.6
    required_consecutive_frames: int = 5


class ReadinessGateConfig(BaseModel):
    """Per-set readiness gate configuration.

    Same checks as StandingGateConfig but resets between sets and
    uses more lenient thresholds — the goal is just to confirm the user
    is standing in frame, not to calibrate bone lengths.
    """
    min_confidence: float = 0.3
    max_knee_flexion_deg: float = 35.0
    max_trunk_flexion_deg: float = 35.0
    min_torso_length_m: float = 0.15
    max_torso_length_m: float = 1.00
    min_leg_extension_ratio: float = 0.5
    required_consecutive_frames: int = 5


# =============================================================================
# FULL CONFIGURATION
# =============================================================================

class BiomechanicsConfig(BaseModel):
    """Complete biomechanics pipeline configuration."""
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    pose: PoseConfig = Field(default_factory=PoseConfig)
    triangulation: TriangulationConfig = Field(default_factory=TriangulationConfig)
    camera_calibration: CameraCalibrationConfig = Field(default_factory=CameraCalibrationConfig)
    faults: FaultsConfig = Field(default_factory=FaultsConfig)
    coaching: CoachingConfig = Field(default_factory=CoachingConfig)
    ipc: IPCConfig = Field(default_factory=IPCConfig)
    bilstm: BiLSTMConfig = Field(default_factory=BiLSTMConfig)
    barbell_tracking: BarbellTrackingConfig = Field(default_factory=BarbellTrackingConfig)
    kalman: KalmanSmootherConfig = Field(default_factory=KalmanSmootherConfig)
    foot_contact: FootContactConfig = Field(default_factory=FootContactConfig)
    display_filter: DisplayFilterConfig = Field(default_factory=DisplayFilterConfig)
    standing_gate: StandingGateConfig = Field(default_factory=StandingGateConfig)
    readiness_gate: ReadinessGateConfig = Field(default_factory=ReadinessGateConfig)
    hip_counter: HipPositionCounterConfig = Field(default_factory=HipPositionCounterConfig)
    deadlift: DeadliftConfig = Field(default_factory=DeadliftConfig)

    # Convenience properties
    @property
    def target_fps(self) -> int:
        return self.pipeline.target_fps

    @property
    def frame_time_ms(self) -> float:
        return 1000.0 / self.pipeline.target_fps


# =============================================================================
# LOADING FUNCTIONS
# =============================================================================

def _get_default_config_path() -> Path:
    """Get the default configuration file path."""
    # Try relative to this file first
    module_dir = Path(__file__).parent
    config_path = module_dir.parent.parent / "config" / "biomechanics.yaml"

    if config_path.exists():
        return config_path

    # Try relative to current working directory
    cwd_config = Path.cwd() / "config" / "biomechanics.yaml"
    if cwd_config.exists():
        return cwd_config

    return config_path


def load_pipeline_config(path: Optional[str] = None) -> BiomechanicsConfig:
    """
    Load pipeline configuration from a YAML file.

    Args:
        path: Path to config file. If None, uses default location.

    Returns:
        BiomechanicsConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    if path is None:
        config_path = _get_default_config_path()
    else:
        config_path = Path(path)

    if not config_path.exists():
        print(f"Warning: Config file not found at {config_path}, using defaults")
        return BiomechanicsConfig()

    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)

    if raw_config is None:
        return BiomechanicsConfig()

    # Parse nested configs
    config_dict = {}

    if "pipeline" in raw_config:
        config_dict["pipeline"] = PipelineConfig(**raw_config["pipeline"])

    if "capture" in raw_config:
        capture_data = raw_config["capture"]
        # Handle resolution as list -> tuple
        if "resolution" in capture_data and isinstance(capture_data["resolution"], list):
            capture_data["resolution"] = tuple(capture_data["resolution"])
        config_dict["capture"] = CaptureConfig(**capture_data)

    if "pose" in raw_config:
        config_dict["pose"] = PoseConfig(**raw_config["pose"])

    if "triangulation" in raw_config:
        config_dict["triangulation"] = TriangulationConfig(**raw_config["triangulation"])

    if "camera_calibration" in raw_config:
        config_dict["camera_calibration"] = CameraCalibrationConfig(**raw_config["camera_calibration"])

    if "faults" in raw_config:
        # Validated as one nested model, so a new fault's YAML block can never
        # be silently dropped by a hand-written field list.
        config_dict["faults"] = FaultsConfig(**raw_config["faults"])

    if "coaching" in raw_config:
        config_dict["coaching"] = CoachingConfig(**raw_config["coaching"])

    if "ipc" in raw_config:
        config_dict["ipc"] = IPCConfig(**raw_config["ipc"])

    if "bilstm" in raw_config:
        config_dict["bilstm"] = BiLSTMConfig(**raw_config["bilstm"])

    if "barbell_tracking" in raw_config:
        config_dict["barbell_tracking"] = BarbellTrackingConfig(**raw_config["barbell_tracking"])

    if "kalman" in raw_config:
        config_dict["kalman"] = KalmanSmootherConfig(**raw_config["kalman"])

    if "foot_contact" in raw_config:
        config_dict["foot_contact"] = FootContactConfig(**raw_config["foot_contact"])

    if "display_filter" in raw_config:
        config_dict["display_filter"] = DisplayFilterConfig(**raw_config["display_filter"])

    if "standing_gate" in raw_config:
        config_dict["standing_gate"] = StandingGateConfig(**raw_config["standing_gate"])

    if "readiness_gate" in raw_config:
        config_dict["readiness_gate"] = ReadinessGateConfig(**raw_config["readiness_gate"])

    if "hip_counter" in raw_config:
        config_dict["hip_counter"] = HipPositionCounterConfig(**raw_config["hip_counter"])

    if "deadlift" in raw_config:
        config_dict["deadlift"] = DeadliftConfig(**raw_config["deadlift"])

    return BiomechanicsConfig(**config_dict)


# Global config instance (lazy loaded)
_global_config: Optional[BiomechanicsConfig] = None


def get_config() -> BiomechanicsConfig:
    """Get the global configuration instance (loads on first call)."""
    global _global_config
    if _global_config is None:
        _global_config = load_pipeline_config()
    return _global_config


def reload_config(path: Optional[str] = None) -> BiomechanicsConfig:
    """Reload the global configuration from disk."""
    global _global_config
    _global_config = load_pipeline_config(path)
    return _global_config
