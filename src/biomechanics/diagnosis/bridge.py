"""Bridge between frame data formats and the diagnosis engine.

Maps frame_data dicts (from visualize_video_squats.py or the live pipeline)
+ athlete_params into the RepKinematicSummary / SetFeatures that
HypothesisEngine expects.
"""

from __future__ import annotations

import math

import numpy as np

from .types import (
    RepKinematicSummary,
    RepTrajectory,
    RepTrajectorySample,
    SetFeatures,
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


def find_bottom_frame(rep_frames: list[dict]) -> dict | None:
    """Return the frame with maximum knee flexion (deepest squat position).

    Returns None if the rep contains no valid frames.
    """
    best_frame = None
    best_knee_flex = -1.0
    for frame in rep_frames:
        if frame is None:
            continue
        knee_flex = frame["angles"].get("knee_flex", 0.0)
        if math.isnan(knee_flex):
            continue
        if knee_flex > best_knee_flex:
            best_knee_flex = knee_flex
            best_frame = frame
    return best_frame


def compute_foot_direction_angle(kpts: list[list[float]], ankle_idx: int, foot_idx: int) -> float:
    """Compute foot direction angle in degrees from keypoints.

    Uses the ankle→foot_index vector projected onto the ground plane (XZ).
    Forward direction is [-1, 0] in viewer coords (vis_x = mp_z, pointing
    away from camera = backward, so forward = -x).
    Returns degrees of external rotation from straight ahead.
    """
    ankle = kpts[ankle_idx]
    foot = kpts[foot_idx]
    vec_xz = np.array([foot[0] - ankle[0], foot[2] - ankle[2]])
    norm = np.linalg.norm(vec_xz)
    if norm < 1e-6:
        return 0.0
    forward = np.array([-1.0, 0.0])
    cos_angle = np.clip(np.dot(vec_xz, forward) / norm, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def classify_depth(knee_flex_degrees: float) -> int:
    """Classify squat depth from knee flexion angle.

    Returns: 0=quarter (<45), 1=half (45-70), 2=above-parallel (70-90),
             3=parallel (90-105), 4=below-parallel (>105)
    """
    if knee_flex_degrees < 45.0:
        return 0
    elif knee_flex_degrees < 70.0:
        return 1
    elif knee_flex_degrees < 90.0:
        return 2
    elif knee_flex_degrees < 105.0:
        return 3
    else:
        return 4


def compute_stance_width_ratio(kpts: list[list[float]], shoulder_width: float) -> float:
    """Compute stance width as ankle XZ distance normalized by shoulder width."""
    l_ankle = kpts[15]
    r_ankle = kpts[16]
    dx = l_ankle[0] - r_ankle[0]
    dz = l_ankle[2] - r_ankle[2]
    ankle_xz_dist = math.sqrt(dx * dx + dz * dz)
    if shoulder_width < 1e-6:
        return 1.0
    return ankle_xz_dist / shoulder_width


def build_rep_trajectory(samples: list[dict] | None) -> RepTrajectory | None:
    """Wrap raw per-frame pipeline samples for the scorer, unmodified."""
    if not samples:
        return None
    return RepTrajectory(
        samples=[RepTrajectorySample(**sample) for sample in samples]
    )


def build_rep_kinematic_summary(
    frame: dict,
    athlete_params: dict,
    rep_number: int,
    descent_time_s: float = 0.0,
    ascent_time_s: float = 0.0,
) -> RepKinematicSummary:
    """Map a single bottom-of-rep frame + athlete params to engine input."""
    angles = frame["angles"]
    kpts = frame["kpts"]

    trunk_pitch = 180.0 - angles["trunk_flexion"]

    foot_angle_l = compute_foot_direction_angle(kpts, ankle_idx=15, foot_idx=17)
    foot_angle_r = compute_foot_direction_angle(kpts, ankle_idx=16, foot_idx=18)

    knee_flex = angles.get("knee_flex", 0.0)
    depth_class = classify_depth(knee_flex)

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
        knee_valgus_l=angles.get("knee_valgus_l", 0.0),
        knee_valgus_r=angles.get("knee_valgus_r", 0.0),
        ankle_df_l_max=angles.get("dorsi_l", 0.0),
        ankle_df_r_max=angles.get("dorsi_r", 0.0),
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
        "foot_length": athlete_params.get("foot_avg_m", 0.26),
    }


def build_rom_dict(athlete_params: dict, baseline: dict) -> dict[str, float]:
    """Build ROM dict from baseline peak values."""
    return {
        "peak_dorsiflexion": baseline.get("peakDorsi", 35.0),
        "avg_depth": baseline.get("peakKneeFlex", 120.0),
    }


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
        bottom_frame = find_bottom_frame(rep_frames)
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
