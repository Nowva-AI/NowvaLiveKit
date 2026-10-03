"""Tests for live pipeline → diagnosis bridge."""

from __future__ import annotations

import math

import numpy as np
import pytest

from biomechanics.diagnosis.bridge import (
    KEYPOINT_TO_BIACROMIAL_RATIO,
    build_anthro_dict,
    build_frame_from_live_pipeline,
    build_rep_kinematic_summary,
    build_rom_dict,
    build_set_features,
    classify_depth,
    compute_foot_direction_angle,
    compute_stance_width_ratio,
    find_bottom_frame,
    mediapipe_to_viewer_coords,
)
from biomechanics.diagnosis.lean_model import expected_pitches
from biomechanics.diagnosis.rep_scoring import (
    DEPTH_DECAY_RATIO,
    DEPTH_TARGET_TOLERANCE_RATIO,
    score_depth,
)

ANGLE_TOLERANCE_DEG = 1e-6
RATIO_TOLERANCE = 1e-6
FOOT_LENGTH_M = 0.20
# The bottom fixture's toes sit 2 cm medial and 6 cm forward of each ankle.
FIXTURE_TOE_IN_DEG = math.degrees(math.atan2(0.02, 0.06))


def _standing_kpts_mediapipe() -> list[list[float]]:
    """Realistic 19-keypoint skeleton in MediaPipe world coords (meters).

    MediaPipe convention: X=subject's left, Y=down, Z=toward camera.
    Re-centered at hip midpoint (hips average to ~0,0,0).
    Represents a standing pose with slight toe-in (toes 2 cm toward the
    midline of each ankle).
    """
    return [
        [0.00, -0.55, 0.02],   # 0  nose
        [0.02, -0.57, 0.01],   # 1  left_eye
        [-0.02, -0.57, 0.01],  # 2  right_eye
        [0.05, -0.55, -0.02],  # 3  left_ear
        [-0.05, -0.55, -0.02], # 4  right_ear
        [0.18, -0.40, 0.00],   # 5  left_shoulder
        [-0.18, -0.40, 0.00],  # 6  right_shoulder
        [0.20, -0.20, 0.05],   # 7  left_elbow
        [-0.20, -0.20, 0.05],  # 8  right_elbow
        [0.18, -0.05, 0.02],   # 9  left_wrist
        [-0.18, -0.05, 0.02],  # 10 right_wrist
        [0.10, 0.00, 0.00],    # 11 left_hip
        [-0.10, 0.00, 0.00],   # 12 right_hip
        [0.10, 0.40, 0.01],    # 13 left_knee
        [-0.10, 0.40, 0.01],   # 14 right_knee
        [0.12, 0.82, 0.00],    # 15 left_ankle
        [-0.12, 0.82, 0.00],   # 16 right_ankle
        [0.10, 0.84, 0.06],    # 17 left_foot_index (slightly forward)
        [-0.10, 0.84, 0.06],   # 18 right_foot_index
    ]


def _squat_bottom_kpts_mediapipe() -> list[list[float]]:
    """19-keypoint skeleton at squat bottom in MediaPipe world coords.

    Hips dropped close to knee height, trunk leaned forward,
    knees tracking over toes.
    """
    return [
        [0.00, -0.25, 0.20],   # 0  nose
        [0.02, -0.27, 0.19],   # 1  left_eye
        [-0.02, -0.27, 0.19],  # 2  right_eye
        [0.05, -0.25, 0.16],   # 3  left_ear
        [-0.05, -0.25, 0.16],  # 4  right_ear
        [0.18, -0.15, 0.12],   # 5  left_shoulder
        [-0.18, -0.15, 0.12],  # 6  right_shoulder
        [0.22, 0.05, 0.18],    # 7  left_elbow
        [-0.22, 0.05, 0.18],   # 8  right_elbow
        [0.20, 0.10, 0.22],    # 9  left_wrist
        [-0.20, 0.10, 0.22],   # 10 right_wrist
        [0.12, 0.00, 0.00],    # 11 left_hip
        [-0.12, 0.00, 0.00],   # 12 right_hip
        [0.12, 0.02, 0.30],    # 13 left_knee
        [-0.12, 0.02, 0.30],   # 14 right_knee
        [0.14, 0.38, 0.10],    # 15 left_ankle
        [-0.14, 0.38, 0.10],   # 16 right_ankle
        [0.12, 0.40, 0.16],    # 17 left_foot_index
        [-0.12, 0.40, 0.16],   # 18 right_foot_index
    ]


def _squat_bottom_angles() -> dict[str, float]:
    """Angles dict matching JointAngles.as_dict() at squat bottom."""
    return {
        "hip_flexion_l": 95.0,
        "hip_flexion_r": 93.0,
        "hip_adduction_l": 2.0,
        "hip_adduction_r": 1.5,
        "hip_rotation_l": 5.0,
        "hip_rotation_r": 4.0,
        "knee_flexion_l": 110.0,
        "knee_flexion_r": 108.0,
        "ankle_dorsiflexion_l": 28.0,
        "ankle_dorsiflexion_r": 26.0,
        "knee_valgus_l": 3.5,
        "knee_valgus_r": 2.0,
        "foot_confidence_l": 0.8,
        "foot_confidence_r": 0.7,
        "shoulder_flexion_l": 40.0,
        "shoulder_flexion_r": 38.0,
        "shoulder_abduction_l": 15.0,
        "shoulder_abduction_r": 14.0,
        "elbow_flexion_l": 90.0,
        "elbow_flexion_r": 88.0,
        "wrist_y_l": -10.0,
        "wrist_y_r": -10.0,
        "wrist_x_l": 5.0,
        "wrist_x_r": 5.0,
        "trunk_flexion": 145.0,
        "trunk_lateral_flexion": 1.0,
        "trunk_rotation": 2.0,
        "pelvis_tilt": 12.0,
        "pelvis_list": 0.5,
        "pelvis_rotation": 1.0,
    }


def _default_athlete_params() -> dict:
    return {
        "shoulder_width_m": 0.40,
        "hip_width_m": 0.30,
        "femur_avg_m": 0.42,
        "torso_avg_m": 0.45,
        "tibia_avg_m": 0.43,
        "foot_avg_m": 0.26,
    }


def _feet_kpts_mediapipe(
    toe_out_l_deg: float, toe_out_r_deg: float, body_yaw_deg: float = 0.0,
) -> list[list[float]]:
    """19 MediaPipe keypoints with only hips, ankles and toes placed.

    The athlete faces the camera (+Z) before ``body_yaw_deg`` turns the whole
    body about the vertical. Toe-out is positive away from the midline; the
    athlete's left is +X.
    """
    kpts = [[0.0, 0.0, 0.0] for _ in range(19)]
    kpts[11] = [0.12, 0.0, 0.0]
    kpts[12] = [-0.12, 0.0, 0.0]
    kpts[15] = [0.14, 0.80, 0.0]
    kpts[16] = [-0.14, 0.80, 0.0]
    for ankle_idx, toe_idx, outward_x, toe_out_deg in (
        (15, 17, 1.0, toe_out_l_deg),
        (16, 18, -1.0, toe_out_r_deg),
    ):
        toe_out_rad = math.radians(toe_out_deg)
        ankle = kpts[ankle_idx]
        kpts[toe_idx] = [
            ankle[0] + outward_x * FOOT_LENGTH_M * math.sin(toe_out_rad),
            ankle[1] + 0.02,
            ankle[2] + FOOT_LENGTH_M * math.cos(toe_out_rad),
        ]
    yaw = math.radians(body_yaw_deg)
    return [
        [
            x * math.cos(yaw) + z * math.sin(yaw),
            y,
            -x * math.sin(yaw) + z * math.cos(yaw),
        ]
        for x, y, z in kpts
    ]


def _foot_angles(kpts_mediapipe: list[list[float]]) -> tuple[float, float]:
    kpts_vis = mediapipe_to_viewer_coords(kpts_mediapipe)
    return (
        compute_foot_direction_angle(kpts_vis, ankle_idx=15, foot_idx=17),
        compute_foot_direction_angle(kpts_vis, ankle_idx=16, foot_idx=18),
    )


def _height_frame(hip_height_m: float, knee_height_m: float) -> dict:
    """A viewer-coords frame with only hip and knee heights (Y-up) set."""
    kpts = [[0.0, 0.0, 0.0] for _ in range(19)]
    for hip_idx in (11, 12):
        kpts[hip_idx][1] = hip_height_m
    for knee_idx in (13, 14):
        kpts[knee_idx][1] = knee_height_m
    return {"kpts": kpts, "angles": {}}


class TestBuildFrameFromLivePipeline:

    def test_angle_key_mapping(self):
        angles = _squat_bottom_angles()
        kpts = _squat_bottom_kpts_mediapipe()
        frame = build_frame_from_live_pipeline(kpts, angles)

        assert frame["angles"]["dorsi_l"] == 28.0
        assert frame["angles"]["dorsi_r"] == 26.0
        assert frame["angles"]["trunk_flexion"] == 145.0
        assert frame["angles"]["knee_valgus_l"] == 3.5
        assert frame["angles"]["knee_valgus_r"] == 2.0

    def test_knee_flex_takes_max(self):
        angles = _squat_bottom_angles()
        kpts = _squat_bottom_kpts_mediapipe()
        frame = build_frame_from_live_pipeline(kpts, angles)

        assert frame["angles"]["knee_flex"] == 110.0

    def test_coordinate_transform(self):
        """Axis swap preserves relative geometry; translation is grounding."""
        kpts_mp = _squat_bottom_kpts_mediapipe()
        angles = _squat_bottom_angles()
        frame = build_frame_from_live_pipeline(kpts_mp, angles)
        kpts_vis = np.array(frame["kpts"])

        assert kpts_vis.shape == (19, 3)

        # vis_x = mp_z, vis_y = -mp_y, vis_z = -mp_x (up to a uniform shift)
        swapped = np.array([[p[2], -p[1], -p[0]] for p in kpts_mp])
        relative_vis = kpts_vis - kpts_vis[11]
        relative_swapped = swapped - swapped[11]
        np.testing.assert_allclose(relative_vis, relative_swapped, atol=1e-9)

    def test_hip_y_grounded_above_ankles(self):
        """Hip vis_y is floor-relative height, not hip-centered zero.

        MediaPipe world coords put the origin at the hip midpoint; the
        bridge must re-ground so hip height is measured from the floor —
        the lowest foot keypoint, not the ankle joint.
        """
        kpts_mp = _squat_bottom_kpts_mediapipe()
        angles = _squat_bottom_angles()
        frame = build_frame_from_live_pipeline(kpts_mp, angles)

        # mp hips at y=0, toes at y=0.40 → grounded hip height = 0.40 m
        assert frame["kpts"][11][1] == pytest.approx(0.40, abs=1e-9)
        assert frame["kpts"][12][1] == pytest.approx(0.40, abs=1e-9)
        min_foot_y = min(frame["kpts"][i][1] for i in (15, 16, 17, 18))
        assert min_foot_y == pytest.approx(0.0, abs=1e-9)
        # Ankle joints keep their anatomical height above the floor
        assert frame["kpts"][15][1] == pytest.approx(0.02, abs=1e-9)


class TestEndToEndBridgeToSummary:

    def test_produces_valid_summary(self):
        kpts = _squat_bottom_kpts_mediapipe()
        angles = _squat_bottom_angles()
        frame = build_frame_from_live_pipeline(kpts, angles)
        params = _default_athlete_params()

        summary = build_rep_kinematic_summary(frame, params, rep_number=1)

        assert summary.rep_number == 1

    def test_trunk_pitch_reasonable(self):
        """trunk_pitch = 180 - trunk_flexion. At 145° flexion → 35° pitch."""
        kpts = _squat_bottom_kpts_mediapipe()
        angles = _squat_bottom_angles()
        frame = build_frame_from_live_pipeline(kpts, angles)
        params = _default_athlete_params()

        summary = build_rep_kinematic_summary(frame, params, rep_number=1)

        assert summary.trunk_pitch_at_bottom == pytest.approx(35.0, abs=0.1)

    def test_dorsiflexion_passthrough(self):
        kpts = _squat_bottom_kpts_mediapipe()
        angles = _squat_bottom_angles()
        frame = build_frame_from_live_pipeline(kpts, angles)
        params = _default_athlete_params()

        summary = build_rep_kinematic_summary(frame, params, rep_number=1)

        assert summary.ankle_df_l_max == pytest.approx(28.0, abs=0.1)
        assert summary.ankle_df_r_max == pytest.approx(26.0, abs=0.1)

    def test_knee_valgus_passthrough(self):
        kpts = _squat_bottom_kpts_mediapipe()
        angles = _squat_bottom_angles()
        frame = build_frame_from_live_pipeline(kpts, angles)
        params = _default_athlete_params()

        summary = build_rep_kinematic_summary(frame, params, rep_number=1)

        assert summary.knee_valgus_l == pytest.approx(3.5, abs=0.1)
        assert summary.knee_valgus_r == pytest.approx(2.0, abs=0.1)

    def test_knee_flexion_does_not_define_depth(self):
        """Regression: 110° of knee flexion used to read as depth class 4
        ("below parallel"). The hip here is 2 cm above the knee — within the
        parallel counting tolerance, so parallel (3), whatever the knee says."""
        kpts = _squat_bottom_kpts_mediapipe()
        angles = _squat_bottom_angles()
        frame = build_frame_from_live_pipeline(kpts, angles)
        params = _default_athlete_params()

        summary = build_rep_kinematic_summary(frame, params, rep_number=1)

        assert summary.depth_ratio == pytest.approx(0.02 / params["femur_avg_m"], abs=RATIO_TOLERANCE)
        assert summary.depth_class_int == 3

    def test_hip_below_knee_is_below_parallel(self):
        kpts = _squat_bottom_kpts_mediapipe()
        # MediaPipe Y is down: drop both hips 6 cm, 4 cm below the knees.
        kpts[11][1] = 0.06
        kpts[12][1] = 0.06
        frame = build_frame_from_live_pipeline(kpts, _squat_bottom_angles())

        summary = build_rep_kinematic_summary(frame, _default_athlete_params(), rep_number=1)

        assert summary.depth_ratio < 0.0
        assert summary.depth_class_int == 4

    def test_expected_pitches_come_from_the_balance_model(self):
        frame = build_frame_from_live_pipeline(
            _squat_bottom_kpts_mediapipe(), _squat_bottom_angles(),
        )
        params = _default_athlete_params()

        summary = build_rep_kinematic_summary(frame, params, rep_number=1)

        shank_deg = (28.0 + 26.0) / 2.0
        expected = expected_pitches(build_anthro_dict(params), summary.depth_ratio, shank_deg)
        assert (
            summary.expected_pitch_reference,
            summary.expected_pitch_athlete,
            summary.expected_pitch_with_ankles,
        ) == pytest.approx(expected, abs=ANGLE_TOLERANCE_DEG)

    def test_stance_width_ratio_positive(self):
        kpts = _squat_bottom_kpts_mediapipe()
        angles = _squat_bottom_angles()
        frame = build_frame_from_live_pipeline(kpts, angles)
        params = _default_athlete_params()

        summary = build_rep_kinematic_summary(frame, params, rep_number=1)

        assert summary.stance_width_ratio > 0.0

    def test_toed_in_fixture_reads_negative_on_both_feet(self):
        """Regression: the angle was unsigned, so these toed-in feet read as
        ~18° of toe-out."""
        kpts = _squat_bottom_kpts_mediapipe()
        angles = _squat_bottom_angles()
        frame = build_frame_from_live_pipeline(kpts, angles)
        params = _default_athlete_params()

        summary = build_rep_kinematic_summary(frame, params, rep_number=1)

        assert summary.foot_direction_angle_l == pytest.approx(-FIXTURE_TOE_IN_DEG, abs=ANGLE_TOLERANCE_DEG)
        assert summary.foot_direction_angle_r == pytest.approx(-FIXTURE_TOE_IN_DEG, abs=ANGLE_TOLERANCE_DEG)


class TestLiveFrameGrounding:

    def test_standing_frame_grounded_independently(self):
        """Each frame grounds to its own foot level, so hip heights compare across frames."""
        frame = build_frame_from_live_pipeline(
            _squat_bottom_kpts_mediapipe(),
            _squat_bottom_angles(),
            standing_kpts=_standing_kpts_mediapipe(),
        )
        standing_vis = frame["standing_kpts"]

        min_foot_y = min(standing_vis[i][1] for i in (15, 16, 17, 18))
        assert min_foot_y == pytest.approx(0.0, abs=1e-9)
        # mp standing: hips at y=0, toes at y=0.84 → hip height 0.84 m
        assert standing_vis[11][1] == pytest.approx(0.84, abs=1e-9)

    def test_hip_mid_centered_in_xz(self):
        """A global XZ offset in camera coords must not survive the transform."""
        kpts_offset = [
            [x + 0.3, y, z + 0.5] for x, y, z in _squat_bottom_kpts_mediapipe()
        ]
        frame = build_frame_from_live_pipeline(kpts_offset, _squat_bottom_angles())
        kpts_vis = frame["kpts"]

        hip_mid_x = (kpts_vis[11][0] + kpts_vis[12][0]) / 2.0
        hip_mid_z = (kpts_vis[11][2] + kpts_vis[12][2]) / 2.0
        assert hip_mid_x == pytest.approx(0.0, abs=1e-9)
        assert hip_mid_z == pytest.approx(0.0, abs=1e-9)

    def test_depth_score_meaningful_from_live_frames(self):
        """Depth is hip height above the knee in femur lengths, within the
        bottom frame: 2 cm above the knee on a 42 cm femur is within the
        parallel tolerance, so it scores full depth."""
        frame = build_frame_from_live_pipeline(
            _squat_bottom_kpts_mediapipe(),
            _squat_bottom_angles(),
            standing_kpts=_standing_kpts_mediapipe(),
        )
        summary = build_rep_kinematic_summary(
            frame, _default_athlete_params(), rep_number=1,
        )

        assert score_depth(summary, {}, {}) == pytest.approx(1.0, abs=RATIO_TOLERANCE)

    def test_shallow_live_frame_loses_depth_score(self):
        kpts = _squat_bottom_kpts_mediapipe()
        # MediaPipe Y is down: knees 20 cm below the hips.
        kpts[13][1] = 0.20
        kpts[14][1] = 0.20
        frame = build_frame_from_live_pipeline(
            kpts, _squat_bottom_angles(), standing_kpts=_standing_kpts_mediapipe(),
        )
        params = _default_athlete_params()
        summary = build_rep_kinematic_summary(frame, params, rep_number=1)

        depth_ratio = 0.20 / params["femur_avg_m"]
        expected = 1.0 - (depth_ratio - DEPTH_TARGET_TOLERANCE_RATIO) / DEPTH_DECAY_RATIO
        assert score_depth(summary, {}, {}) == pytest.approx(expected, abs=RATIO_TOLERANCE)


def _valid_frame() -> dict:
    return build_frame_from_live_pipeline(
        _squat_bottom_kpts_mediapipe(), _squat_bottom_angles(),
    )


class TestFindBottomFrame:

    def test_skips_none_frames(self):
        valid = _valid_frame()
        assert find_bottom_frame([None, valid, None]) is valid

    def test_all_none_returns_none(self):
        assert find_bottom_frame([None, None]) is None

    def test_empty_returns_none(self):
        assert find_bottom_frame([]) is None

    def test_picks_the_hip_lowest_relative_to_the_knee(self):
        # The first frame has the lower hip in absolute terms, but the second
        # sits deeper relative to its own knees.
        lower_hip_frame = _height_frame(hip_height_m=0.42, knee_height_m=0.36)
        deeper_frame = _height_frame(hip_height_m=0.45, knee_height_m=0.47)
        assert find_bottom_frame([lower_hip_frame, deeper_frame]) is deeper_frame


class TestBuildSetFeaturesEdgeCases:

    def test_degenerate_reps_skipped(self):
        replay_reps = [[], [None, None], [_valid_frame()]]
        features = build_set_features(
            replay_reps, _default_athlete_params(),
            baseline={"peakDorsi": 35.0, "peakKneeFlex": 120.0},
        )
        assert len(features.per_rep_kinematics) == 1

    def test_does_not_mutate_caller_frames(self):
        frame = _valid_frame()
        build_set_features(
            [[frame]], _default_athlete_params(),
            baseline={"peakDorsi": 35.0, "peakKneeFlex": 120.0},
        )
        assert "standing_kpts" not in frame


class TestFootDirectionAngle:
    """Toe-out is signed: positive = toes away from the midline, negative =
    toed in. It used to come from arccos, so 20° of toe-in read as 20° of
    toe-out."""

    def test_toe_out_is_positive_on_both_feet(self):
        angle_l, angle_r = _foot_angles(_feet_kpts_mediapipe(20.0, 20.0))
        assert angle_l == pytest.approx(20.0, abs=ANGLE_TOLERANCE_DEG)
        assert angle_r == pytest.approx(20.0, abs=ANGLE_TOLERANCE_DEG)

    def test_toe_in_is_negative_on_both_feet(self):
        angle_l, angle_r = _foot_angles(_feet_kpts_mediapipe(-15.0, -15.0))
        assert angle_l == pytest.approx(-15.0, abs=ANGLE_TOLERANCE_DEG)
        assert angle_r == pytest.approx(-15.0, abs=ANGLE_TOLERANCE_DEG)

    def test_toe_in_and_toe_out_are_not_confused(self):
        angle_l, angle_r = _foot_angles(_feet_kpts_mediapipe(20.0, -20.0))
        assert angle_l == pytest.approx(20.0, abs=ANGLE_TOLERANCE_DEG)
        assert angle_r == pytest.approx(-20.0, abs=ANGLE_TOLERANCE_DEG)

    def test_straight_feet_read_zero(self):
        angle_l, angle_r = _foot_angles(_feet_kpts_mediapipe(0.0, 0.0))
        assert angle_l == pytest.approx(0.0, abs=ANGLE_TOLERANCE_DEG)
        assert angle_r == pytest.approx(0.0, abs=ANGLE_TOLERANCE_DEG)

    @pytest.mark.parametrize("body_yaw_deg", [35.0, -50.0, 180.0])
    def test_turned_body_is_not_read_as_toe_out(self, body_yaw_deg: float):
        # Measured against the athlete's own forward axis, not the camera's.
        angle_l, angle_r = _foot_angles(
            _feet_kpts_mediapipe(12.0, -6.0, body_yaw_deg=body_yaw_deg)
        )
        assert angle_l == pytest.approx(12.0, abs=ANGLE_TOLERANCE_DEG)
        assert angle_r == pytest.approx(-6.0, abs=ANGLE_TOLERANCE_DEG)

    def test_zero_length_foot_is_unmeasurable(self):
        kpts = _feet_kpts_mediapipe(20.0, 20.0)
        kpts[17] = list(kpts[15])
        angle_l, angle_r = _foot_angles(kpts)
        assert math.isnan(angle_l)
        assert angle_r == pytest.approx(20.0, abs=ANGLE_TOLERANCE_DEG)

    def test_coincident_hips_are_unmeasurable(self):
        kpts = _feet_kpts_mediapipe(20.0, 20.0)
        kpts[12] = list(kpts[11])
        angle_l, angle_r = _foot_angles(kpts)
        assert math.isnan(angle_l)
        assert math.isnan(angle_r)


class TestClassifyDepth:
    """Depth class is geometric — hip height above the knee in femur lengths —
    in the BiLSTM's 1-4 vocabulary, with 0 for unmeasured."""

    def test_unmeasured_depth_is_class_zero(self):
        # Regression: NaN fell through every comparison to class 4, the best.
        assert classify_depth(math.nan) == 0

    @pytest.mark.parametrize(
        ("depth_ratio", "expected_class"),
        [
            (-0.30, 4),
            (-0.06, 4),
            (-0.02, 3),
            (0.0, 3),
            (0.05, 3),
            (0.08, 3),
            (0.09, 2),
            (0.50, 2),
            (0.51, 1),
            (1.0, 1),
        ],
    )
    def test_class_boundaries(self, depth_ratio: float, expected_class: int):
        assert classify_depth(depth_ratio) == expected_class


class TestStanceWidthRatio:
    """Stance is measured against biacromial width — coaching's "shoulder
    width" — not the narrower shoulder joint-centre distance."""

    def test_feet_under_the_shoulders_read_one(self):
        keypoint_shoulder_width_m = 0.32
        ankle_separation_m = keypoint_shoulder_width_m / KEYPOINT_TO_BIACROMIAL_RATIO
        kpts = [[0.0, 0.0, 0.0] for _ in range(19)]
        kpts[15] = [0.0, 0.0, -ankle_separation_m / 2.0]
        kpts[16] = [0.0, 0.0, ankle_separation_m / 2.0]
        ratio = compute_stance_width_ratio(kpts, keypoint_shoulder_width_m)
        assert ratio == pytest.approx(1.0, abs=RATIO_TOLERANCE)


class TestWholeRepFeatures:
    """Whole-rep features (the numbers the intra-set fault rules judged) take
    precedence over single bottom-frame values, so cue and recap agree."""

    def _features(self) -> dict:
        return {
            "trunk_pitch_bottom": 41.0,
            "depth_ratio": -0.08,
            "valgus_l": 7.0,
            "valgus_r": 3.0,
            "dorsiflexion_max_l": 31.0,
            "dorsiflexion_max_r": 29.0,
            "hip_shift_ratio": 0.06,
            "hip_shoot_deg": None,
            "heel_rise_max_cm_l": 1.0,
            "heel_rise_max_cm_r": 2.0,
            "neck_flexion_bottom": -12.0,
            "hip_flexion_max_l": 112.0,
            "hip_flexion_max_r": 118.0,
            "descent_time_s": 1.4,
            "ascent_time_s": 0.9,
        }

    def _summary(self, features: dict | None):
        return build_rep_kinematic_summary(
            _valid_frame(), _default_athlete_params(), rep_number=3, features=features,
        )

    def test_features_override_the_bottom_frame(self):
        summary = self._summary(self._features())
        assert summary.trunk_pitch_at_bottom == pytest.approx(41.0)
        assert summary.depth_ratio == pytest.approx(-0.08)
        assert summary.depth_class_int == 4
        assert summary.knee_valgus_l == pytest.approx(7.0)
        assert summary.knee_valgus_r == pytest.approx(3.0)
        assert summary.ankle_df_l_max == pytest.approx(31.0)
        assert summary.ankle_df_r_max == pytest.approx(29.0)

    def test_feature_only_measures_are_carried(self):
        summary = self._summary(self._features())
        assert summary.hip_shift_ratio == pytest.approx(0.06)
        assert summary.heel_rise_max_cm == pytest.approx(2.0)
        assert summary.neck_flexion_deg == pytest.approx(-12.0)
        assert summary.hip_flexion_l_max == pytest.approx(112.0)
        assert summary.hip_flexion_r_max == pytest.approx(118.0)
        assert summary.descent_time_s == pytest.approx(1.4)
        assert summary.ascent_time_s == pytest.approx(0.9)
        assert math.isnan(summary.hip_shoot_deg)

    def test_expected_pitches_use_the_feature_depth_and_ankles(self):
        params = _default_athlete_params()
        summary = self._summary(self._features())
        expected = expected_pitches(build_anthro_dict(params), -0.08, 30.0)
        assert (
            summary.expected_pitch_reference,
            summary.expected_pitch_athlete,
            summary.expected_pitch_with_ankles,
        ) == pytest.approx(expected, abs=ANGLE_TOLERANCE_DEG)

    def test_missing_features_fall_back_to_the_bottom_frame(self):
        features = self._features()
        features["trunk_pitch_bottom"] = math.nan
        features["depth_ratio"] = math.nan
        summary = self._summary(features)
        assert summary.trunk_pitch_at_bottom == pytest.approx(35.0)
        assert summary.depth_ratio == pytest.approx(0.02 / 0.42, abs=RATIO_TOLERANCE)


class TestBuildRomDict:

    def test_depth_capacity_and_target_are_passed_through(self):
        rom = build_rom_dict(
            _default_athlete_params(),
            {"peakDorsi": 28.0, "peakHipFlex": 110.0, "peakKneeFlex": 125.0,
             "depthCapacityRatio": -0.1, "depthTargetRatio": 0.05},
        )
        assert rom["peak_dorsiflexion"] == pytest.approx(28.0)
        assert rom["peak_hip_flexion"] == pytest.approx(110.0)
        assert rom["avg_depth"] == pytest.approx(125.0)
        assert rom["depth_capacity_ratio"] == pytest.approx(-0.1)
        assert rom["depth_target_ratio"] == pytest.approx(0.05)

    def test_uncalibrated_depth_keys_are_absent(self):
        rom = build_rom_dict(_default_athlete_params(), {"peakDorsi": 35.0})
        assert "depth_capacity_ratio" not in rom
        assert "depth_target_ratio" not in rom
