"""
Pipeline Configuration for Biomechanics System

Loads configuration from YAML files and provides typed access to settings.
"""

import os
from typing import List, Optional, Tuple
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
    single_camera_mode: bool = True


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
    """Stereo triangulation configuration."""
    enabled: bool = False
    device_ids: List[int] = [0, 1, 2]
    primary_camera: int = 0
    max_sync_delta_ms: float = 20.0
    focal_length_factor: float = 0.8
    min_views: int = 2
    max_reprojection_error: float = 15.0
    calibration_file: Optional[str] = None
    tpose_capture_frames: int = 30


class KinematicsConfig(BaseModel):
    """Inverse kinematics configuration."""
    backend: str = "analytical"  # analytical | opensim


class DepthFaultConfig(BaseModel):
    """Squat depth fault thresholds."""
    parallel: float = 90.0
    below_parallel: float = 100.0


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
    """Knee valgus fault thresholds (3D triangulated and 2D FPPA)."""
    mild: float = 12.0
    moderate: float = 17.0
    severe: float = 24.0
    mild_2d: float = 6.0
    moderate_2d: float = 10.0
    severe_2d: float = 16.0


class HeelRiseConfig(BaseModel):
    """Heel-rise fault thresholds (ankle rise above its planted floor reference; multi-camera only)."""
    mild_cm: float = 1.5
    moderate_cm: float = 3.0
    severe_cm: float = 5.0
    cooldown_s: float = 2.0
    min_rise_duration_s: float = 0.23


class FaultsConfig(BaseModel):
    """Fault detection configuration."""
    depth: DepthFaultConfig = Field(default_factory=DepthFaultConfig)
    bilateral_asymmetry: BilateralAsymmetryConfig = Field(default_factory=BilateralAsymmetryConfig)
    forward_lean: ForwardLeanConfig = Field(default_factory=ForwardLeanConfig)
    knee_valgus: KneeValgusConfig = Field(default_factory=KneeValgusConfig)
    heel_rise: HeelRiseConfig = Field(default_factory=HeelRiseConfig)


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


class RepDetectionConfig(BaseModel):
    """Rep detection configuration."""
    entry_threshold: float = 30.0
    min_rep_duration_frames: int = 20


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

    # Minimum frames in each state (prevents noise flipping)
    min_frames_descending: int = 3
    min_frames_bottom: int = 2
    min_frames_ascending: int = 3

    # Rep validation
    min_rep_duration_frames: int = 15       # ~0.5s at 30fps


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
    max_predicted_frames: int = 5
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
    kinematics: KinematicsConfig = Field(default_factory=KinematicsConfig)
    faults: FaultsConfig = Field(default_factory=FaultsConfig)
    rep_detection: RepDetectionConfig = Field(default_factory=RepDetectionConfig)
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

    if "kinematics" in raw_config:
        config_dict["kinematics"] = KinematicsConfig(**raw_config["kinematics"])

    if "faults" in raw_config:
        faults_data = raw_config["faults"]
        faults_config = FaultsConfig(
            depth=DepthFaultConfig(**faults_data.get("depth", {})),
            bilateral_asymmetry=BilateralAsymmetryConfig(**faults_data.get("bilateral_asymmetry", {})),
            forward_lean=ForwardLeanConfig(**faults_data.get("forward_lean", {})),
            knee_valgus=KneeValgusConfig(**faults_data.get("knee_valgus", {})),
            heel_rise=HeelRiseConfig(**faults_data.get("heel_rise", {})),
        )
        config_dict["faults"] = faults_config

    if "rep_detection" in raw_config:
        config_dict["rep_detection"] = RepDetectionConfig(**raw_config["rep_detection"])

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
