from __future__ import annotations

import math

from pydantic import BaseModel


class RepTrajectorySample(BaseModel):
    """One frame of a rep, as measured. Heights in cm, angles in degrees.

    Heights are Y-up (larger = higher), measured above the ankle midpoint of
    the same frame. Every metric derived from them is a within-frame
    difference or an angle, so the scorer never re-grounds them. Fields past
    the first seven default to NaN for samples recorded before they existed.
    """
    trunk_pitch: float
    knee_valgus_l: float
    knee_valgus_r: float
    hip_y_l: float
    hip_y_r: float
    knee_y_l: float
    knee_y_r: float
    timestamp: float = 0.0
    phase: str = ""
    hip_height_cm: float = math.nan
    shoulder_height_cm: float = math.nan
    depth_ratio: float = math.nan
    hip_lateral_ratio: float = math.nan
    balance_ratio: float = math.nan
    heel_rise_l_cm: float = math.nan
    heel_rise_r_cm: float = math.nan
    knee_flexion_l: float = math.nan
    knee_flexion_r: float = math.nan
    hip_flexion_l: float = math.nan
    hip_flexion_r: float = math.nan
    dorsiflexion_l: float = math.nan
    dorsiflexion_r: float = math.nan
    neck_flexion_deg: float = math.nan
    bar_detected: bool = False


class RepTrajectory(BaseModel):
    samples: list[RepTrajectorySample]


class RepKinematicSummary(BaseModel):
    rep_number: int
    trunk_pitch_at_bottom: float
    knee_valgus_l: float
    knee_valgus_r: float
    ankle_df_l_max: float
    ankle_df_r_max: float
    hip_y_l_at_bottom: float
    hip_y_r_at_bottom: float
    knee_y_l_at_bottom: float
    knee_y_r_at_bottom: float
    stance_width_ratio: float
    foot_direction_angle_l: float
    foot_direction_angle_r: float
    depth_class_int: int
    hip_y_l_at_top: float = 0.0
    hip_y_r_at_top: float = 0.0
    knee_y_l_at_top: float = 0.0
    knee_y_r_at_top: float = 0.0
    descent_time_s: float = 0.0
    ascent_time_s: float = 0.0
    # Whole-rep features (analysis.rep_features) — the same numbers the
    # intra-set fault rules judged, so cue and recap cannot disagree.
    depth_ratio: float = math.nan
    depth_cm_above_parallel: float = math.nan
    hip_shoot_deg: float = math.nan
    hip_shift_ratio: float = math.nan
    balance_ratio: float = math.nan
    heel_rise_max_cm: float = math.nan
    concentric_velocity_mps: float = math.nan
    velocity_loss_pct: float = math.nan
    initiation_ratio: float = math.nan
    neck_flexion_deg: float = math.nan
    lockout_deficit_ratio: float = math.nan
    stagger_ratio: float = math.nan
    hip_flexion_l_max: float = math.nan
    hip_flexion_r_max: float = math.nan
    bar_detected: bool = False
    # Trunk pitch (deg from vertical) that puts the load over midfoot at this
    # rep's depth, from the sagittal balance model (diagnosis.lean_model):
    # reference proportions with unrestricted ankles; the athlete's own
    # segments with unrestricted ankles; the athlete's segments and ankles.
    expected_pitch_reference: float = math.nan
    expected_pitch_athlete: float = math.nan
    expected_pitch_with_ankles: float = math.nan


class SetFeatures(BaseModel):
    user_id: int
    set_id: str
    rep_count: int
    per_rep_kinematics: list[RepKinematicSummary]
    anthropometry: dict[str, float]
    rom: dict[str, float]
    # "single_camera" or "triangulated" — decides which symptoms are
    # observable (diagnosis.observability).
    capture_mode: str = "single_camera"


class DetectedSymptom(BaseModel):
    symptom_id: str
    severity: float
    contributing_reps: list[int]
    observability: str = "observable"


class HypothesizedCause(BaseModel):
    cause_id: str
    tier: int
    score: float
    evidence_score: float
    prior: float
    implicated_by: list[str]
    parameter_delta: dict | None
    explanation: str
    observability: str = "observable"


class RepScore(BaseModel):
    rep_number: int
    depth_score: float
    trunk_control_score: float
    knee_tracking_score: float
    symmetry_score: float
    tempo_score: float
    composite_score: float


class SetScoreSummary(BaseModel):
    mean_score: float
    best_rep_number: int
    worst_rep_number: int
    trend_slope: float
    per_rep_scores: list[RepScore]


class DiagnosisResult(BaseModel):
    set_id: str
    detected_symptoms: list[DetectedSymptom]
    immediate_causes: list[HypothesizedCause]
    session_causes: list[HypothesizedCause]
    longterm_causes: list[HypothesizedCause]
    contextual_notes: list[HypothesizedCause]
    combined_perturbation: dict
    confidence: float
