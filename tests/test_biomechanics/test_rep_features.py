"""Tests for analysis.rep_features: per-frame squat samples and the per-rep features."""

from __future__ import annotations

import math

import numpy as np
import pytest

from biomechanics.analysis.rep_features import (
    SetupSnapshot,
    build_frame_sample,
    build_setup_snapshot,
    compute_rep_features,
)
from biomechanics.diagnosis.types import RepTrajectorySample
from biomechanics.utils.types import CocoKeypoints as CK
from biomechanics.utils.types import JointAngles, Skeleton3D

from conftest import SYNTHETIC_FEMUR_M, world_squat_points

RATIO_TOLERANCE = 0.01
DEG_TOLERANCE = 0.5
FPS = 30.0
STANDING_HIP_CM = 90.0
BOTTOM_HIP_CM = 45.0
SHOULDER_ABOVE_HIP_CM = 50.0


def _skeleton(points: np.ndarray) -> Skeleton3D:
    return Skeleton3D.from_numpy(points)


def _sample(
    timestamp: float,
    depth_ratio: float,
    hip_height_cm: float,
    trunk_pitch: float = 30.0,
    valgus: tuple[float, float] = (0.0, 0.0),
    hip_lateral_ratio: float = 0.0,
    balance_ratio: float = 0.0,
    knee_flexion: float = 60.0,
    hip_flexion: float = 60.0,
    shoulder_height_cm: float | None = None,
) -> RepTrajectorySample:
    shoulder = hip_height_cm + SHOULDER_ABOVE_HIP_CM if shoulder_height_cm is None else shoulder_height_cm
    return RepTrajectorySample(
        trunk_pitch=trunk_pitch,
        knee_valgus_l=valgus[0],
        knee_valgus_r=valgus[1],
        hip_y_l=hip_height_cm,
        hip_y_r=hip_height_cm,
        knee_y_l=50.0,
        knee_y_r=50.0,
        timestamp=timestamp,
        hip_height_cm=hip_height_cm,
        shoulder_height_cm=shoulder,
        depth_ratio=depth_ratio,
        hip_lateral_ratio=hip_lateral_ratio,
        balance_ratio=balance_ratio,
        knee_flexion_l=knee_flexion,
        knee_flexion_r=knee_flexion,
        hip_flexion_l=hip_flexion,
        hip_flexion_r=hip_flexion,
        dorsiflexion_l=25.0,
        dorsiflexion_r=25.0,
    )


def _rep(
    frames_down: int = 30,
    frames_up: int = 30,
    pitch_rise_deg: float = 0.0,
    ascent_valgus: float = 0.0,
    bottom_valgus: float = 0.0,
    hip_shift_ratio: float = 0.0,
) -> list[RepTrajectorySample]:
    """A synthetic rep: standing → bottom → standing, with optional faults."""
    samples = []
    total = frames_down + frames_up
    for i in range(total + 1):
        if i <= frames_down:
            progress = i / frames_down
        else:
            progress = 1.0 - (i - frames_down) / frames_up
        depth = 1.0 - progress
        hip = STANDING_HIP_CM - (STANDING_HIP_CM - BOTTOM_HIP_CM) * progress
        ascending = i > frames_down
        # The chest drops while the hip rises through its first third.
        early_ascent = ascending and progress > 0.66
        pitch = 30.0 + 10.0 * progress + (pitch_rise_deg if early_ascent else 0.0)
        valgus = bottom_valgus if progress > 0.9 else (ascent_valgus if ascending and progress > 0.3 else 0.0)
        samples.append(
            _sample(
                timestamp=i / FPS,
                depth_ratio=depth,
                hip_height_cm=hip,
                trunk_pitch=pitch,
                valgus=(valgus, 0.0),
                hip_lateral_ratio=hip_shift_ratio if progress > 0.3 else 0.0,
                knee_flexion=20.0 + 90.0 * progress,
                hip_flexion=20.0 + 90.0 * progress,
            )
        )
    return samples


class TestBuildFrameSample:
    def test_heights_are_y_up_above_the_ankles(self):
        """A1 regression: Y-down heights once made the scorer read standing frames as the bottom."""
        sample = build_frame_sample(_skeleton(world_squat_points(0.0)), JointAngles(trunk_flexion=180.0))
        assert sample.hip_height_cm > sample.knee_y_l > 0.0
        assert sample.shoulder_height_cm > sample.hip_height_cm

    def test_depth_ratio_is_positive_standing_and_near_zero_at_the_synthetic_bottom(self):
        standing = build_frame_sample(_skeleton(world_squat_points(0.0)), JointAngles(), femur_m=SYNTHETIC_FEMUR_M)
        bottom = build_frame_sample(_skeleton(world_squat_points(1.0)), JointAngles(), femur_m=SYNTHETIC_FEMUR_M)
        assert standing.depth_ratio == pytest.approx(1.0, abs=RATIO_TOLERANCE)
        # The synthetic thigh stops 5° above horizontal.
        assert bottom.depth_ratio == pytest.approx(math.sin(math.radians(5.0)), abs=RATIO_TOLERANCE)

    def test_centred_hips_have_no_lateral_offset(self):
        sample = build_frame_sample(_skeleton(world_squat_points(0.5)), JointAngles())
        assert sample.hip_lateral_ratio == pytest.approx(0.0, abs=RATIO_TOLERANCE)

    def test_hips_toward_the_right_foot_read_positive(self):
        points = world_squat_points(0.5)
        points[CK.LEFT_HIP][0] -= 0.05  # X is the subject's left; move hips right
        points[CK.RIGHT_HIP][0] -= 0.05
        sample = build_frame_sample(_skeleton(points), JointAngles())
        assert sample.hip_lateral_ratio > 0.0

    def test_trunk_pitch_is_degrees_from_vertical(self):
        sample = build_frame_sample(_skeleton(world_squat_points(0.0)), JointAngles(trunk_flexion=140.0))
        assert sample.trunk_pitch == pytest.approx(40.0)

    def test_bar_moves_the_load_point_to_the_shoulders(self):
        points = world_squat_points(1.0)
        body = build_frame_sample(_skeleton(points), JointAngles(), bar_detected=False)
        loaded = build_frame_sample(_skeleton(points), JointAngles(), bar_detected=True)
        assert math.isfinite(body.balance_ratio) and math.isfinite(loaded.balance_ratio)
        assert loaded.bar_detected and not body.bar_detected
        assert loaded.balance_ratio != pytest.approx(body.balance_ratio)

    def test_knee_reading_needs_the_feet_seen(self):
        """A coach who cannot see the feet says nothing about the knees."""
        seen = JointAngles(knee_valgus_l=12.0, knee_valgus_r=12.0, foot_confidence_l=0.9, foot_confidence_r=0.1)
        sample = build_frame_sample(_skeleton(world_squat_points(1.0)), seen)
        assert sample.knee_valgus_l == pytest.approx(12.0)
        assert math.isnan(sample.knee_valgus_r)

    def test_missing_toes_make_balance_unmeasurable(self):
        points = world_squat_points(1.0)
        skeleton = _skeleton(points)
        skeleton.keypoints[CK.LEFT_FOOT_INDEX].confidence = 0.0
        assert math.isnan(build_frame_sample(skeleton, JointAngles()).balance_ratio)


class TestBuildSetupSnapshot:
    def test_square_feet_read_no_stagger(self):
        snapshot = build_setup_snapshot(_skeleton(world_squat_points(0.0)))
        assert snapshot.stagger_ratio == pytest.approx(0.0, abs=RATIO_TOLERANCE)

    def test_left_foot_ahead_reads_positive_stagger(self):
        points = world_squat_points(0.0)
        for index in (CK.LEFT_ANKLE, CK.LEFT_FOOT_INDEX, CK.LEFT_HEEL):
            points[index][2] -= 0.06  # forward is -Z
        assert build_setup_snapshot(_skeleton(points)).stagger_ratio > 0.0

    def test_toe_out_is_signed_and_positive_away_from_the_midline(self):
        snapshot = build_setup_snapshot(_skeleton(world_squat_points(0.0)))
        # The synthetic toes sit 3 cm outside the ankles, 18 cm forward.
        expected = math.degrees(math.atan2(0.03, 0.18))
        assert snapshot.toe_out_l_deg == pytest.approx(expected, abs=DEG_TOLERANCE)
        assert snapshot.toe_out_r_deg == pytest.approx(expected, abs=DEG_TOLERANCE)

    def test_toe_in_reads_negative(self):
        points = world_squat_points(0.0)
        points[CK.LEFT_FOOT_INDEX][0] -= 0.08
        assert build_setup_snapshot(_skeleton(points)).toe_out_l_deg < 0.0


class TestComputeRepFeatures:
    def test_empty_rep_is_all_unmeasured(self):
        features = compute_rep_features([], rep_number=3)
        assert features.rep_number == 3
        assert math.isnan(features.depth_ratio)

    def test_depth_is_the_deep_end_of_the_rep(self):
        features = compute_rep_features(_rep(), rep_number=1, femur_m=0.45)
        assert features.depth_ratio == pytest.approx(0.0, abs=0.1)
        assert features.depth_cm_above_parallel == pytest.approx(features.depth_ratio * 45.0)

    def test_valgus_on_the_way_up_is_caught_and_phased(self):
        """Knees caving out of the hole is the pattern a coach watches for."""
        features = compute_rep_features(_rep(ascent_valgus=12.0), rep_number=1)
        assert features.valgus_l == pytest.approx(12.0)
        assert features.valgus_peak_phase_l == "ascent"

    def test_valgus_at_the_bottom_is_phased_bottom(self):
        features = compute_rep_features(_rep(bottom_valgus=12.0), rep_number=1)
        assert features.valgus_peak_phase_l == "bottom"

    def test_chest_dropping_out_of_the_hole_reads_as_hip_shoot(self):
        clean = compute_rep_features(_rep(), rep_number=1)
        shoot = compute_rep_features(_rep(pitch_rise_deg=15.0), rep_number=1)
        assert shoot.hip_shoot_deg > clean.hip_shoot_deg + 10.0

    def test_hip_shift_is_measured_from_where_the_rep_started(self):
        features = compute_rep_features(_rep(hip_shift_ratio=0.12), rep_number=1)
        assert features.hip_shift_ratio == pytest.approx(0.12, abs=RATIO_TOLERANCE)

    def test_one_jittery_frame_is_not_a_hip_shift(self):
        """The max over frames once made noise a shift on 60% of real reps."""
        samples = _rep()
        samples[len(samples) // 2].hip_lateral_ratio = 0.3
        features = compute_rep_features(samples, rep_number=1)
        assert abs(features.hip_shift_ratio) < RATIO_TOLERANCE

    def test_concentric_velocity_is_shoulder_rise_over_time(self):
        features = compute_rep_features(_rep(frames_up=30), rep_number=1)
        # Shoulders rise ~45 cm in about a second.
        assert features.concentric_velocity_mps == pytest.approx(0.45, abs=0.05)

    def test_descent_and_ascent_times(self):
        features = compute_rep_features(_rep(frames_down=24, frames_up=36), rep_number=1)
        assert features.descent_time_s == pytest.approx(24 / FPS)
        assert features.ascent_time_s == pytest.approx(36 / FPS)

    def test_lockout_deficit_against_the_tallest_standing(self):
        setup = SetupSnapshot(hip_height_cm=STANDING_HIP_CM - 5.0)
        features = compute_rep_features(
            _rep(), rep_number=2, setup=setup,
            standing_reference_hip_cm=STANDING_HIP_CM, leg_length_m=0.9,
        )
        assert features.lockout_deficit_ratio == pytest.approx(5.0 / 90.0, abs=RATIO_TOLERANCE)

    def test_initiation_ratio_is_one_when_hips_and_knees_break_together(self):
        features = compute_rep_features(_rep(), rep_number=1)
        assert features.initiation_ratio == pytest.approx(1.0, abs=RATIO_TOLERANCE)
