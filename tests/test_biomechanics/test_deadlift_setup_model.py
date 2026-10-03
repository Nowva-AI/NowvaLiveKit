"""The deadlift setup model (docs/deadlift/PLAN.md §2.7) and its sagittal frame (§2.1)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from biomechanics.deadlift.frame import (
    build_sagittal_frame,
    forward_m,
    height_m,
    lateral_m,
    segment_angle_deg,
)
from biomechanics.deadlift.setup_model import (
    AthleteSegments,
    predict_setup,
    solve_knee_pass,
    solve_setup,
)
from biomechanics.utils.geometry import WORLD_UP

ANGLE_TOLERANCE_DEG = 1e-6
LENGTH_TOLERANCE_M = 1e-6
# Bar over midfoot (0.35 x an 18 cm foot) with 45 cm plates and the ankle 8 cm up.
BAR_FORWARD_M = 0.063
BAR_HEIGHT_M = 0.145
SHIN_BAR_M = 0.05
SHOULDER_BAND_M = (0.0, 0.06)


def _segments(**overrides) -> AthleteSegments:
    values = {"tibia_m": 0.43, "femur_m": 0.45, "torso_m": 0.52, "arm_m": 0.59, "grip_offset_m": 0.075}
    values.update(overrides)
    return AthleteSegments(**values)


class TestSetupSolve:
    def test_a_typical_body_sets_up_with_knees_bent_and_trunk_inclined(self):
        solution = solve_setup(_segments(), BAR_FORWARD_M, BAR_HEIGHT_M, 0.03, SHIN_BAR_M)
        assert 40.0 <= solution.knee_flexion_deg <= 120.0
        assert 45.0 < solution.trunk_deg < 80.0
        assert solution.knee_forward_m > 0.0
        assert solution.hip_forward_m < 0.0

    def test_the_shin_line_passes_the_required_distance_behind_the_bar(self):
        solution = solve_setup(_segments(), BAR_FORWARD_M, BAR_HEIGHT_M, 0.03, SHIN_BAR_M)
        shin = np.array([solution.knee_forward_m, solution.knee_height_m])
        shin = shin / np.linalg.norm(shin)
        bar = np.array([BAR_FORWARD_M, BAR_HEIGHT_M])
        distance = float(bar[0] * shin[1] - bar[1] * shin[0])
        assert distance == pytest.approx(SHIN_BAR_M, abs=LENGTH_TOLERANCE_M)

    def test_shoulders_further_ahead_put_the_hips_higher(self):
        low = solve_setup(_segments(), BAR_FORWARD_M, BAR_HEIGHT_M, 0.0, SHIN_BAR_M)
        high = solve_setup(_segments(), BAR_FORWARD_M, BAR_HEIGHT_M, 0.06, SHIN_BAR_M)
        assert high.hip_height_m > low.hip_height_m

    def test_no_setup_fits_an_impossible_body(self):
        assert solve_setup(_segments(torso_m=0.05), BAR_FORWARD_M, BAR_HEIGHT_M, 0.03, SHIN_BAR_M) is None


class TestPrediction:
    def test_the_trunk_rises_between_the_floor_and_the_knees(self):
        prediction = predict_setup(_segments(), BAR_FORWARD_M, BAR_HEIGHT_M, SHOULDER_BAND_M, 0.03, SHIN_BAR_M)
        assert prediction.trunk_change_deg < 0.0
        knee_pass = solve_knee_pass(_segments(), BAR_FORWARD_M, 0.03, SHIN_BAR_M)
        assert prediction.trunk_knee_pass_deg == pytest.approx(knee_pass.trunk_deg, abs=ANGLE_TOLERANCE_DEG)

    def test_the_hip_band_spans_the_shoulder_band(self):
        prediction = predict_setup(_segments(), BAR_FORWARD_M, BAR_HEIGHT_M, SHOULDER_BAND_M, 0.03, SHIN_BAR_M)
        low = solve_setup(_segments(), BAR_FORWARD_M, BAR_HEIGHT_M, 0.0, SHIN_BAR_M)
        high = solve_setup(_segments(), BAR_FORWARD_M, BAR_HEIGHT_M, 0.06, SHIN_BAR_M)
        assert prediction.hip_band_low_m == pytest.approx(low.hip_height_m, abs=LENGTH_TOLERANCE_M)
        assert prediction.hip_band_high_m == pytest.approx(high.hip_height_m, abs=LENGTH_TOLERANCE_M)

    def test_a_shoulder_offset_outside_the_band_is_clipped_into_it(self):
        clipped = predict_setup(_segments(), BAR_FORWARD_M, BAR_HEIGHT_M, SHOULDER_BAND_M, 0.20, SHIN_BAR_M)
        edge = predict_setup(_segments(), BAR_FORWARD_M, BAR_HEIGHT_M, SHOULDER_BAND_M, 0.06, SHIN_BAR_M)
        assert clipped.trunk_setup_deg == pytest.approx(edge.trunk_setup_deg, abs=ANGLE_TOLERANCE_DEG)
        unmeasured = predict_setup(_segments(), BAR_FORWARD_M, BAR_HEIGHT_M, SHOULDER_BAND_M, math.nan, SHIN_BAR_M)
        middle = predict_setup(_segments(), BAR_FORWARD_M, BAR_HEIGHT_M, SHOULDER_BAND_M, 0.03, SHIN_BAR_M)
        assert unmeasured.trunk_setup_deg == pytest.approx(middle.trunk_setup_deg, abs=ANGLE_TOLERANCE_DEG)


class TestSagittalFrame:
    def test_world_axes_map_to_forward_up_and_the_subjects_right(self):
        # Y-down world: subject's left is +X, forward is -Z.
        frame = build_sagittal_frame(np.array(WORLD_UP), np.array([-1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0]))
        origin = np.zeros(3)
        assert forward_m(frame, np.array([0.0, 0.0, -0.2]), origin) == pytest.approx(0.2, abs=LENGTH_TOLERANCE_M)
        assert height_m(frame, np.array([0.0, -0.3, 0.0]), origin) == pytest.approx(0.3, abs=LENGTH_TOLERANCE_M)
        assert lateral_m(frame, np.array([-0.1, 0.0, 0.0]), origin) == pytest.approx(0.1, abs=LENGTH_TOLERANCE_M)

    def test_toes_decide_forward_even_from_a_reversed_hint(self):
        frame = build_sagittal_frame(np.array(WORLD_UP), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0]))
        assert frame.forward == pytest.approx(np.array([0.0, 0.0, -1.0]), abs=LENGTH_TOLERANCE_M)
        assert frame.lateral == pytest.approx(np.array([-1.0, 0.0, 0.0]), abs=LENGTH_TOLERANCE_M)

    def test_segment_angles_are_signed_forward_positive(self):
        frame = build_sagittal_frame(np.array(WORLD_UP), np.array([-1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0]))
        hip = np.zeros(3)
        leaning_forward = np.array([0.0, -0.5, -0.5])
        leaning_back = np.array([0.0, -0.5, 0.2])
        assert segment_angle_deg(frame, hip, leaning_forward) == pytest.approx(45.0, abs=ANGLE_TOLERANCE_DEG)
        assert segment_angle_deg(frame, hip, leaning_back) < 0.0

    def test_a_vertical_lateral_hint_has_no_frame(self):
        assert build_sagittal_frame(np.array(WORLD_UP), np.array([0.0, 1.0, 0.0])) is None
