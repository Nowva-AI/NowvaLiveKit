"""Bridge between frame data formats and the diagnosis engine.

Maps frame_data dicts (from visualize_video_squats.py or the live pipeline)
+ athlete_params into the RepKinematicSummary / SetFeatures that
HypothesisEngine expects.
"""

from __future__ import annotations

import math

import numpy as np

from biomechanics.faults.rules.depth import PARALLEL_LABEL_TOLERANCE_RATIO

from .lean_model import expected_pitches
from .types import (
    RepKinematicSummary,
    RepTrajectory,
    RepTrajectorySample,
    SetFeatures,
)

# Shoulder keypoints are joint centres; coaching's "shoulder width" is the
# biacromial breadth, about this much wider.
KEYPOINT_TO_BIACROMIAL_RATIO = 0.80
DEFAULT_FEMUR_M = 0.42
# Ankle-to-toe-tip length assumed when the feet were never measured — the
# span the balance model places midfoot on.
DEFAULT_ANKLE_TO_TOE_M = 0.20
# Geometric depth class (hip height above the knee, femur lengths), in the same
# 1-4 vocabulary as the BiLSTM depth classes; 0 = unmeasurable.
_QUARTER_SQUAT_RATIO = 0.5
_BELOW_PARALLEL_RATIO = -0.05

# RepFeatures keys copied onto the kinematic summary under the same name.
_SUMMARY_FEATURE_KEYS = (
    "depth_ratio",
    "depth_cm_above_parallel",
    "hip_shoot_deg",
    "hip_shift_ratio",
    "balance_ratio",
    "concentric_velocity_mps",
    "velocity_loss_pct",
    "initiation_ratio",
    "lockout_deficit_ratio",
    "stagger_ratio",
    "bar_detected",
)


def mediapipe_to_viewer_coords(kpts: list[list[float]]) -> list[list[float]]:
    """Transform MediaPipe world coords → visualizer coords.

    MediaPipe: X=subject's left, Y=down, Z=toward camera.
    Visualizer: vis_x=mp_z, vis_y=-mp_y, vis_z=-mp_x.

    compute_foot_direction_angle measures against forward=[-1,0] in this
    frame, so any caller of it must transform first.
    """
    return [[pt[2], -pt[1], -pt[0]] for pt in kpts]


def _ground_and_center(kpts: list[list[float]]) -> list[list[float]]:
    """Ground the lowest foot keypoint Y to 0 and center hip midpoint in XZ.

    MediaPipe world coords are hip-centered, so hip heights are meaningless
    across frames until each frame is re-grounded to its own foot level.
    Matches visualize_video_squats ground_and_center preprocessing.
    """
    arr = np.asarray(kpts, dtype=np.float64)
    # Index 15 onward is every foot keypoint — ankles, toes, and heels.
    # Slicing rather than listing indices keeps this correct for both the
    # 19-keypoint poses stored before heels were tracked and 21-keypoint ones.
    arr[:, 1] -= arr[15:, 1].min()
    arr[:, 0] -= (arr[11, 0] + arr[12, 0]) / 2.0
    arr[:, 2] -= (arr[11, 2] + arr[12, 2]) / 2.0
    return arr.tolist()


def build_frame_from_live_pipeline(
    bottom_kpts: list[list[float]],
    bottom_angles: dict[str, float],
    standing_kpts: list[list[float]] | None = None,
) -> dict:
    """Convert live pipeline data to the frame dict build_rep_kinematic_summary expects."""
    kpts_vis = _ground_and_center(mediapipe_to_viewer_coords(bottom_kpts))

    angles = {
        "trunk_flexion": bottom_angles["trunk_flexion"],
        "knee_valgus_l": bottom_angles["knee_valgus_l"],
        "knee_valgus_r": bottom_angles["knee_valgus_r"],
        "dorsi_l": bottom_angles["ankle_dorsiflexion_l"],
        "dorsi_r": bottom_angles["ankle_dorsiflexion_r"],
        "knee_flex": max(
            bottom_angles["knee_flexion_l"],
            bottom_angles["knee_flexion_r"],
        ),
    }

    result: dict = {"angles": angles, "kpts": kpts_vis}
    if standing_kpts is not None:
        result["standing_kpts"] = _ground_and_center(
            mediapipe_to_viewer_coords(standing_kpts)
        )
    return result


def _depth_ratio_from_kpts(kpts: list[list[float]], femur_m: float) -> float:
    """Hip height above the knee in femur lengths, from viewer (Y-up) keypoints."""
    hip_y = (kpts[11][1] + kpts[12][1]) / 2.0
    knee_y = (kpts[13][1] + kpts[14][1]) / 2.0
    if femur_m < 1e-6:
        return math.nan
    return max(-1.0, min(1.0, (hip_y - knee_y) / femur_m))


def find_bottom_frame(rep_frames: list[dict], femur_m: float = DEFAULT_FEMUR_M) -> dict | None:
    """Return the frame where the hip sits lowest relative to the knees.

    Returns None if the rep contains no valid frames.
    """
    best_frame = None
    best_ratio = math.inf
    for frame in rep_frames:
        if frame is None:
            continue
        ratio = _depth_ratio_from_kpts(frame["kpts"], femur_m)
        if math.isnan(ratio):
            continue
        if ratio < best_ratio:
            best_ratio = ratio
            best_frame = frame
    return best_frame


def compute_foot_direction_angle(kpts: list[list[float]], ankle_idx: int, foot_idx: int) -> float:
    """Signed toe-out of one foot in degrees, from viewer-coords keypoints.

    Measured on the ground plane (XZ) against the athlete's own forward axis
    (perpendicular to the hip line), so a turned body is not read as toe-out.
    Positive = toes turned away from the midline, negative = toed in; NaN when
    the foot or hips are not measurable. Left side is ankle index 15.
    """
    ankle = kpts[ankle_idx]
    foot = kpts[foot_idx]
    vec_xz = np.array([foot[0] - ankle[0], foot[2] - ankle[2]])
    lateral = np.array([kpts[12][0] - kpts[11][0], kpts[12][2] - kpts[11][2]])
    foot_norm = float(np.linalg.norm(vec_xz))
    lateral_norm = float(np.linalg.norm(lateral))
    if foot_norm < 1e-6 or lateral_norm < 1e-6:
        return math.nan
    lateral /= lateral_norm
    forward = np.array([-lateral[1], lateral[0]])
    if np.dot(forward, vec_xz) < 0.0:
        forward = -forward
    outward = -lateral if ankle_idx == 15 else lateral
    return float(np.degrees(np.arctan2(np.dot(vec_xz, outward), np.dot(vec_xz, forward))))


def classify_depth(depth_ratio: float) -> int:
    """Depth class from hip height above the knee (femur lengths).

    Returns 1=quarter, 2=half, 3=parallel, 4=below parallel — the BiLSTM's
    vocabulary — and 0 when depth was not measured. Never the deepest class
    for a missing value.
    """
    if math.isnan(depth_ratio):
        return 0
    if depth_ratio < _BELOW_PARALLEL_RATIO:
        return 4
    if depth_ratio <= PARALLEL_LABEL_TOLERANCE_RATIO:
        return 3
    if depth_ratio <= _QUARTER_SQUAT_RATIO:
        return 2
    return 1


def compute_stance_width_ratio(kpts: list[list[float]], shoulder_width: float) -> float:
    """Ankle separation over biacromial shoulder width (coaching's "shoulder width").

    ``shoulder_width`` is the keypoint (joint-centre) distance; it is widened to
    biacromial breadth so a ratio of 1.0 means feet under the shoulders.
    """
    l_ankle = kpts[15]
    r_ankle = kpts[16]
    dx = l_ankle[0] - r_ankle[0]
    dz = l_ankle[2] - r_ankle[2]
    ankle_xz_dist = math.sqrt(dx * dx + dz * dz)
    if shoulder_width < 1e-6:
        return 1.0
    return ankle_xz_dist / (shoulder_width / KEYPOINT_TO_BIACROMIAL_RATIO)


def _nanmax(first: float, second: float) -> float:
    finite = [value for value in (first, second) if math.isfinite(value)]
    return max(finite) if finite else math.nan


def _finite_or(value: float | None, fallback: float) -> float:
    return value if value is not None and math.isfinite(value) else fallback


def build_rep_trajectory(
    samples: list[RepTrajectorySample | dict] | None,
) -> RepTrajectory | None:
    """Wrap per-frame samples for the scorer, unmodified.

    The live pipeline hands over RepTrajectorySample objects; recorded
    sessions store them as dicts.
    """
    if not samples:
        return None
    return RepTrajectory(
        samples=[
            sample if isinstance(sample, RepTrajectorySample) else RepTrajectorySample(**sample)
            for sample in samples
        ]
    )


def build_rep_kinematic_summary(
    frame: dict,
    athlete_params: dict,
    rep_number: int,
    descent_time_s: float = 0.0,
    ascent_time_s: float = 0.0,
    features: dict | None = None,
) -> RepKinematicSummary:
    """Map a rep's bottom frame + its whole-rep features + athlete params to engine input.

    ``features`` (analysis.rep_features.RepFeatures as a dict) are the numbers
    the intra-set fault rules judged; they take precedence over single-frame
    values so the set diagnosis reads the same rep the cues did.
    """
    angles = frame["angles"]
    kpts = frame["kpts"]
    features = features or {}

    trunk_pitch = _finite_or(features.get("trunk_pitch_bottom"), 180.0 - angles["trunk_flexion"])

    foot_angle_l = compute_foot_direction_angle(kpts, ankle_idx=15, foot_idx=17)
    foot_angle_r = compute_foot_direction_angle(kpts, ankle_idx=16, foot_idx=18)

    femur_m = athlete_params.get("femur_avg_m", DEFAULT_FEMUR_M)
    depth_ratio = _finite_or(features.get("depth_ratio"), _depth_ratio_from_kpts(kpts, femur_m))
    depth_class = classify_depth(depth_ratio)

    dorsi_l = _finite_or(features.get("dorsiflexion_max_l"), angles.get("dorsi_l", math.nan))
    dorsi_r = _finite_or(features.get("dorsiflexion_max_r"), angles.get("dorsi_r", math.nan))
    shank_deg = float(np.nanmean([dorsi_l, dorsi_r])) if math.isfinite(_nanmax(dorsi_l, dorsi_r)) else math.nan
    reference_pitch, athlete_pitch, ankle_pitch = expected_pitches(
        build_anthro_dict(athlete_params), depth_ratio, shank_deg,
    )

    feature_kwargs = {
        key: features[key]
        for key in _SUMMARY_FEATURE_KEYS
        if key in features and features[key] is not None
    }
    feature_kwargs["depth_ratio"] = depth_ratio
    feature_kwargs["heel_rise_max_cm"] = _nanmax(
        _finite_or(features.get("heel_rise_max_cm_l"), math.nan),
        _finite_or(features.get("heel_rise_max_cm_r"), math.nan),
    )
    feature_kwargs["neck_flexion_deg"] = _finite_or(features.get("neck_flexion_bottom"), math.nan)
    feature_kwargs["hip_flexion_l_max"] = _finite_or(features.get("hip_flexion_max_l"), math.nan)
    feature_kwargs["hip_flexion_r_max"] = _finite_or(features.get("hip_flexion_max_r"), math.nan)
    if descent_time_s <= 0.0:
        descent_time_s = _finite_or(features.get("descent_time_s"), 0.0)
    if ascent_time_s <= 0.0:
        ascent_time_s = _finite_or(features.get("ascent_time_s"), 0.0)

    shoulder_width = athlete_params.get("shoulder_width_m", 0.40)
    stance_ratio = compute_stance_width_ratio(kpts, shoulder_width)

    standing_kpts = frame.get("standing_kpts")
    top_kwargs: dict = {}
    if standing_kpts is not None:
        top_kwargs = {
            "hip_y_l_at_top": standing_kpts[11][1] * 100.0,
            "hip_y_r_at_top": standing_kpts[12][1] * 100.0,
            "knee_y_l_at_top": standing_kpts[13][1] * 100.0,
            "knee_y_r_at_top": standing_kpts[14][1] * 100.0,
        }

    return RepKinematicSummary(
        rep_number=rep_number,
        trunk_pitch_at_bottom=trunk_pitch,
        knee_valgus_l=_finite_or(features.get("valgus_l"), angles.get("knee_valgus_l", math.nan)),
        knee_valgus_r=_finite_or(features.get("valgus_r"), angles.get("knee_valgus_r", math.nan)),
        ankle_df_l_max=dorsi_l,
        ankle_df_r_max=dorsi_r,
        hip_y_l_at_bottom=kpts[11][1] * 100.0,
        hip_y_r_at_bottom=kpts[12][1] * 100.0,
        knee_y_l_at_bottom=kpts[13][1] * 100.0,
        knee_y_r_at_bottom=kpts[14][1] * 100.0,
        stance_width_ratio=stance_ratio,
        foot_direction_angle_l=foot_angle_l,
        foot_direction_angle_r=foot_angle_r,
        depth_class_int=depth_class,
        descent_time_s=descent_time_s,
        ascent_time_s=ascent_time_s,
        expected_pitch_reference=reference_pitch,
        expected_pitch_athlete=athlete_pitch,
        expected_pitch_with_ankles=ankle_pitch,
        **feature_kwargs,
        **top_kwargs,
    )


def build_anthro_dict(athlete_params: dict) -> dict[str, float]:
    """Extract anthropometry dict expected by evidence tests."""
    femur_avg = athlete_params.get("femur_avg_m", 0.42)
    torso_avg = athlete_params.get("torso_avg_m", 0.45)
    return {
        "femur_torso_ratio": femur_avg / max(torso_avg, 0.01),
        "hip_width": athlete_params.get("hip_width_m", 0.30),
        "shoulder_width": athlete_params.get("shoulder_width_m", 0.40),
        "femur_length_avg": femur_avg,
        "tibia_length_avg": athlete_params.get("tibia_avg_m", 0.43),
        "torso_length": torso_avg,
        "foot_length": athlete_params.get("foot_avg_m", DEFAULT_ANKLE_TO_TOE_M),
    }


def build_rom_dict(athlete_params: dict, baseline: dict) -> dict[str, float]:
    """Build ROM dict from the athlete's calibrated capacities.

    peak_dorsiflexion and peak_hip_flexion are the median calibration rep's
    95th percentile at the bottom; avg_depth is knee flexion, kept for display
    only — it is not a hip measure.
    """
    rom = {
        "peak_dorsiflexion": baseline.get("peakDorsi", 35.0),
        "peak_hip_flexion": baseline.get("peakHipFlex", 120.0),
        "avg_depth": baseline.get("peakKneeFlex", 120.0),
    }
    for rom_key, baseline_key in (
        ("depth_capacity_ratio", "depthCapacityRatio"),
        ("depth_target_ratio", "depthTargetRatio"),
    ):
        if baseline.get(baseline_key) is not None:
            rom[rom_key] = baseline[baseline_key]
    return rom


def build_set_features(
    replay_reps: list[list[dict]],
    athlete_params: dict,
    baseline: dict,
) -> SetFeatures:
    """Build SetFeatures from session data for the diagnosis engine."""
    anthro = build_anthro_dict(athlete_params)
    rom = build_rom_dict(athlete_params, baseline)

    per_rep_kinematics = []
    for rep_idx, rep_frames in enumerate(replay_reps):
        bottom_frame = find_bottom_frame(rep_frames, athlete_params.get("femur_avg_m", DEFAULT_FEMUR_M))
        if bottom_frame is None:
            continue
        first_frame = next(f for f in rep_frames if f is not None)
        frame = {**bottom_frame, "standing_kpts": first_frame["kpts"]}
        rep_number = rep_idx + 2
        summary = build_rep_kinematic_summary(frame, athlete_params, rep_number)
        per_rep_kinematics.append(summary)

    return SetFeatures(
        user_id=0,
        set_id="session",
        rep_count=len(replay_reps),
        per_rep_kinematics=per_rep_kinematics,
        anthropometry=anthro,
        rom=rom,
    )
