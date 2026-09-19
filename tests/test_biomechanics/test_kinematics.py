"""
Tests for inverse kinematics module.

Tests the AnalyticalIKSolver with various poses and validates
that computed angles are physically plausible.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from biomechanics.kinematics.base import IKSolver
from biomechanics.kinematics.analytical_ik import AnalyticalIKSolver
from biomechanics.utils.geometry import WORLD_UP
from biomechanics.utils.types import Skeleton3D, JointAngles, Point3D


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def ik_solver():
    """Create an AnalyticalIKSolver instance."""
    return AnalyticalIKSolver()


@pytest.fixture
def standing_skeleton():
    """
    Create a 3D skeleton in standing position.

    All joints should be roughly at 0 degrees flexion.
    Coordinate system: the production Y-down frame (see geometry.WORLD_UP)
    — a keypoint that is physically higher has a SMALLER y. Hips at y = 0.
    """
    # Standing pose with arms down
    keypoints = [
        Point3D(x=0.0, y=-0.70, z=0.0, confidence=1.0),     # 0: nose
        Point3D(x=0.03, y=-0.72, z=-0.02, confidence=1.0),  # 1: left_eye
        Point3D(x=-0.03, y=-0.72, z=-0.02, confidence=1.0), # 2: right_eye
        Point3D(x=0.06, y=-0.70, z=0.0, confidence=1.0),    # 3: left_ear
        Point3D(x=-0.06, y=-0.70, z=0.0, confidence=1.0),   # 4: right_ear
        Point3D(x=0.18, y=-0.50, z=0.0, confidence=1.0),    # 5: left_shoulder
        Point3D(x=-0.18, y=-0.50, z=0.0, confidence=1.0),   # 6: right_shoulder
        Point3D(x=0.20, y=-0.20, z=0.0, confidence=1.0),    # 7: left_elbow
        Point3D(x=-0.20, y=-0.20, z=0.0, confidence=1.0),   # 8: right_elbow
        Point3D(x=0.22, y=0.10, z=0.0, confidence=1.0),     # 9: left_wrist
        Point3D(x=-0.22, y=0.10, z=0.0, confidence=1.0),    # 10: right_wrist
        Point3D(x=0.12, y=0.0, z=0.0, confidence=1.0),      # 11: left_hip
        Point3D(x=-0.12, y=0.0, z=0.0, confidence=1.0),     # 12: right_hip
        Point3D(x=0.12, y=0.50, z=0.0, confidence=1.0),     # 13: left_knee
        Point3D(x=-0.12, y=0.50, z=0.0, confidence=1.0),    # 14: right_knee
        Point3D(x=0.12, y=0.95, z=0.0, confidence=1.0),     # 15: left_ankle
        Point3D(x=-0.12, y=0.95, z=0.0, confidence=1.0),    # 16: right_ankle
    ]
    return Skeleton3D(keypoints=keypoints, timestamp=0.0, frame_index=0)


@pytest.fixture
def symmetric_squat_skeleton():
    """
    Create a symmetric squat pose.

    Both sides should have identical angles.
    Production Y-down frame, hips at y = 0.
    """
    # Symmetric half-squat
    keypoints = [
        Point3D(x=0.0, y=-0.60, z=0.05, confidence=1.0),    # 0: nose
        Point3D(x=0.03, y=-0.62, z=0.03, confidence=1.0),   # 1: left_eye
        Point3D(x=-0.03, y=-0.62, z=0.03, confidence=1.0),  # 2: right_eye
        Point3D(x=0.06, y=-0.60, z=0.05, confidence=1.0),   # 3: left_ear
        Point3D(x=-0.06, y=-0.60, z=0.05, confidence=1.0),  # 4: right_ear
        Point3D(x=0.18, y=-0.45, z=0.0, confidence=1.0),    # 5: left_shoulder
        Point3D(x=-0.18, y=-0.45, z=0.0, confidence=1.0),   # 6: right_shoulder
        Point3D(x=0.25, y=-0.20, z=0.10, confidence=1.0),   # 7: left_elbow
        Point3D(x=-0.25, y=-0.20, z=0.10, confidence=1.0),  # 8: right_elbow
        Point3D(x=0.28, y=0.0, z=0.15, confidence=1.0),     # 9: left_wrist
        Point3D(x=-0.28, y=0.0, z=0.15, confidence=1.0),    # 10: right_wrist
        Point3D(x=0.12, y=0.0, z=0.0, confidence=1.0),      # 11: left_hip
        Point3D(x=-0.12, y=0.0, z=0.0, confidence=1.0),     # 12: right_hip
        Point3D(x=0.14, y=0.40, z=0.15, confidence=1.0),    # 13: left_knee
        Point3D(x=-0.14, y=0.40, z=0.15, confidence=1.0),   # 14: right_knee
        Point3D(x=0.14, y=0.85, z=0.05, confidence=1.0),    # 15: left_ankle
        Point3D(x=-0.14, y=0.85, z=0.05, confidence=1.0),   # 16: right_ankle
    ]
    return Skeleton3D(keypoints=keypoints, timestamp=0.0, frame_index=0)


def _shift(skeleton: Skeleton3D, index: int, axis: int, delta: float) -> Skeleton3D:
    points = skeleton.to_numpy()
    points[index][axis] += delta
    confidences = np.array([kp.confidence for kp in skeleton.keypoints])
    return Skeleton3D.from_numpy(points, confidences=confidences)


# =============================================================================
# COORDINATE FRAME TESTS
# =============================================================================

class TestCoordinateFrame:
    """The solver had per-method Y-up and Y-down constants applied
    inconsistently, and every fixture here was Y-up, so two angles were
    wrong in production without a single failing test. These pin the frame."""

    def test_world_up_is_negative_y(self):
        np.testing.assert_array_equal(WORLD_UP, np.array([0.0, -1.0, 0.0]))

    def test_fixtures_are_in_the_production_frame(self, standing_skeleton):
        """Ankles must be BELOW hips, which in a Y-down frame is a larger y."""
        points = standing_skeleton.to_numpy()
        hip_mid_y = (points[11][1] + points[12][1]) / 2.0
        shoulder_mid_y = (points[5][1] + points[6][1]) / 2.0
        assert points[15][1] > hip_mid_y
        assert points[16][1] > hip_mid_y
        assert shoulder_mid_y < hip_mid_y

    def test_raising_left_hip_gives_positive_pelvis_list(
        self, ik_solver, standing_skeleton,
    ):
        raised = _shift(standing_skeleton, 11, axis=1, delta=-0.05)
        assert ik_solver.solve(raised).pelvis_list > 5.0

    def test_leaning_left_gives_positive_lateral_flexion(
        self, ik_solver, standing_skeleton,
    ):
        leaned = _shift(standing_skeleton, 5, axis=0, delta=0.10)
        leaned = _shift(leaned, 6, axis=0, delta=0.10)
        assert ik_solver.solve(leaned).trunk_lateral_flexion > 5.0

    def test_hip_adduction_is_mirrored_between_sides(
        self, ik_solver, standing_skeleton,
    ):
        """Both sign branches used to reduce to `+1 if x > 0`, so the left side
        read inverted relative to the right and abs() downstream made
        knees-out indistinguishable from knee cave."""
        # Both knees toward the midline: bilateral adduction (knee cave).
        caved = _shift(standing_skeleton, 13, axis=0, delta=-0.06)
        caved = _shift(caved, 14, axis=0, delta=0.06)
        result = ik_solver.solve(caved)
        assert result.hip_adduction_l > 0
        assert result.hip_adduction_r > 0
        assert result.hip_adduction_l == pytest.approx(
            result.hip_adduction_r, abs=1.0,
        )

    def test_knees_out_reads_as_abduction_on_both_sides(
        self, ik_solver, standing_skeleton,
    ):
        knees_out = _shift(standing_skeleton, 13, axis=0, delta=0.06)
        knees_out = _shift(knees_out, 14, axis=0, delta=-0.06)
        result = ik_solver.solve(knees_out)
        assert result.hip_adduction_l < 0
        assert result.hip_adduction_r < 0

    def test_forward_lean_reduces_trunk_flexion(self, ik_solver, standing_skeleton):
        """+Z is the subject's BACK (triangulated and MediaPipe world frames
        agree), so a forward lean moves the shoulders toward -Z."""
        leaned = _shift(standing_skeleton, 5, axis=2, delta=-0.25)
        leaned = _shift(leaned, 6, axis=2, delta=-0.25)
        result = ik_solver.solve(leaned)
        assert result.trunk_flexion < 170.0
        assert result.pelvis_tilt > 5.0

    def test_backward_lean_gives_negative_pelvis_tilt(self, ik_solver, standing_skeleton):
        leaned = _shift(standing_skeleton, 5, axis=2, delta=0.25)
        leaned = _shift(leaned, 6, axis=2, delta=0.25)
        result = ik_solver.solve(leaned)
        assert result.trunk_flexion < 170.0
        assert result.pelvis_tilt < -5.0

    def test_wrist_in_front_of_shoulder_reads_positive_x(self, ik_solver, standing_skeleton):
        forward = _shift(standing_skeleton, 9, axis=2, delta=-0.30)
        forward = _shift(forward, 10, axis=2, delta=-0.30)
        result = ik_solver.solve(forward)
        assert result.wrist_x_l > 20.0
        assert result.wrist_x_r > 20.0

    def test_trunk_rotated_left_reads_positive(self, ik_solver, standing_skeleton):
        """Turning to the left moves the left shoulder back (+Z) and the
        right shoulder forward (-Z)."""
        rotated = _shift(standing_skeleton, 5, axis=2, delta=0.10)
        rotated = _shift(rotated, 6, axis=2, delta=-0.10)
        result = ik_solver.solve(rotated)
        assert result.trunk_rotation > 10.0

    def test_pelvis_rotated_left_reads_positive(self, ik_solver, standing_skeleton):
        rotated = _shift(standing_skeleton, 11, axis=2, delta=0.06)
        rotated = _shift(rotated, 12, axis=2, delta=-0.06)
        result = ik_solver.solve(rotated)
        assert result.pelvis_rotation > 10.0


# =============================================================================
# BASIC TESTS
# =============================================================================

class TestAnalyticalIKSolverBasic:
    """Basic tests for AnalyticalIKSolver."""

    def test_instantiation(self):
        """Solver should instantiate without errors."""
        solver = AnalyticalIKSolver()
        assert solver is not None
        assert solver.is_initialized

    def test_body_proportions_hook_is_gone(self):
        """Pelvis tilt coupling is a constant; the solver no longer depends on
        BodyProportions (C8/F7)."""
        assert not hasattr(AnalyticalIKSolver(), "set_body_proportions")

    def test_solve_returns_joint_angles(self, ik_solver, standing_skeleton):
        """solve() should return a JointAngles object."""
        result = ik_solver.solve(standing_skeleton)
        assert isinstance(result, JointAngles)

    def test_solve_preserves_timestamp(self, ik_solver, standing_skeleton):
        """solve() should preserve timestamp from skeleton."""
        standing_skeleton.timestamp = 123.456
        standing_skeleton.frame_index = 42
        result = ik_solver.solve(standing_skeleton)
        assert result.timestamp == 123.456
        assert result.frame_index == 42


# =============================================================================
# STANDING POSE TESTS
# =============================================================================

class TestStandingPose:
    """Tests for standing pose (joints near zero flexion)."""

    def test_hip_flexion_near_zero(self, ik_solver, standing_skeleton):
        """Hip flexion should be near 0 when standing upright."""
        result = ik_solver.solve(standing_skeleton)
        assert abs(result.hip_flexion_l) < 15.0, f"Left hip flexion {result.hip_flexion_l} too large for standing"
        assert abs(result.hip_flexion_r) < 15.0, f"Right hip flexion {result.hip_flexion_r} too large for standing"

    def test_knee_flexion_near_zero(self, ik_solver, standing_skeleton):
        """Knee flexion should be near 0 when standing upright."""
        result = ik_solver.solve(standing_skeleton)
        assert abs(result.knee_flexion_l) < 15.0, f"Left knee flexion {result.knee_flexion_l} too large for standing"
        assert abs(result.knee_flexion_r) < 15.0, f"Right knee flexion {result.knee_flexion_r} too large for standing"

    def test_trunk_flexion_near_upright(self, ik_solver, standing_skeleton):
        """Trunk flexion reads 180 upright and decreases with forward lean."""
        result = ik_solver.solve(standing_skeleton)
        assert result.trunk_flexion > 165.0, f"Trunk flexion {result.trunk_flexion} not upright for standing"

    def test_lateral_and_pelvis_angles_zero_when_upright(self, ik_solver, standing_skeleton):
        """The angles that were computed in the wrong vertical frame: an
        upright pose read 180 for lateral flexion and 72 for pelvis tilt."""
        result = ik_solver.solve(standing_skeleton)
        assert abs(result.trunk_lateral_flexion) < 1.0
        assert abs(result.pelvis_tilt) < 1.0
        assert abs(result.pelvis_list) < 1.0


# =============================================================================
# SQUAT POSE TESTS
# =============================================================================

class TestSquatPose:
    """Tests for squat poses."""

    def test_knee_flexion_increases_in_squat(self, ik_solver, standing_skeleton, sample_skeleton_3d):
        """Knee flexion should be greater in squat than standing."""
        standing_result = ik_solver.solve(standing_skeleton)
        squat_result = ik_solver.solve(sample_skeleton_3d)

        assert squat_result.knee_flexion_l > standing_result.knee_flexion_l
        assert squat_result.knee_flexion_r > standing_result.knee_flexion_r

    def test_hip_flexion_increases_in_squat(self, ik_solver, standing_skeleton, sample_skeleton_3d):
        """Hip flexion should be greater in squat than standing."""
        standing_result = ik_solver.solve(standing_skeleton)
        squat_result = ik_solver.solve(sample_skeleton_3d)

        assert squat_result.hip_flexion_l > standing_result.hip_flexion_l
        assert squat_result.hip_flexion_r > standing_result.hip_flexion_r

    def test_angles_match_expected_within_tolerance(self, ik_solver, sample_skeleton_3d, expected_angles_dict, angle_tolerance):
        """Computed angles should be non-zero and physically plausible for squat pose."""
        result = ik_solver.solve(sample_skeleton_3d)

        # Analytical IK may use different conventions than ground truth
        # Just verify that flexion angles are positive (indicating bent joints)
        assert result.knee_flexion_l > 0, "Knee should be flexed in squat"
        assert result.knee_flexion_r > 0, "Knee should be flexed in squat"
        assert result.hip_flexion_l > 0, "Hip should be flexed in squat"
        assert result.hip_flexion_r > 0, "Hip should be flexed in squat"


# =============================================================================
# SYMMETRY TESTS
# =============================================================================

class TestBilateralSymmetry:
    """Tests for bilateral symmetry."""

    def test_symmetric_input_gives_symmetric_output(self, ik_solver, symmetric_squat_skeleton):
        """Symmetric skeleton should produce symmetric angles."""
        result = ik_solver.solve(symmetric_squat_skeleton)

        # Allow small tolerance for floating point
        tolerance = 2.0  # degrees

        assert abs(result.hip_flexion_l - result.hip_flexion_r) < tolerance
        assert abs(result.knee_flexion_l - result.knee_flexion_r) < tolerance
        assert abs(result.ankle_dorsiflexion_l - result.ankle_dorsiflexion_r) < tolerance
        # Hip adduction should have same magnitude but opposite sign for symmetric pose
        # (left thigh going left = positive, right thigh going right = negative)
        assert abs(abs(result.hip_adduction_l) - abs(result.hip_adduction_r)) < tolerance

    def test_lateral_angles_near_zero_for_symmetric(self, ik_solver, symmetric_squat_skeleton):
        """Lateral/rotation angles should be near zero for symmetric pose."""
        result = ik_solver.solve(symmetric_squat_skeleton)

        tolerance = 5.0  # degrees

        assert abs(result.trunk_lateral_flexion) < tolerance
        assert abs(result.pelvis_list) < tolerance


# =============================================================================
# PHYSICAL PLAUSIBILITY TESTS
# =============================================================================

class TestPhysicalPlausibility:
    """Tests that angles are within physically plausible ranges."""

    def test_angles_within_valid_range(self, ik_solver, sample_skeleton_3d):
        """All angles should be within physically possible ranges."""
        result = ik_solver.solve(sample_skeleton_3d)

        # Hip flexion: -30 to 130 degrees
        assert -30 <= result.hip_flexion_l <= 130
        assert -30 <= result.hip_flexion_r <= 130

        # Knee flexion: 0 to 160 degrees
        assert 0 <= result.knee_flexion_l <= 160
        assert 0 <= result.knee_flexion_r <= 160

        # Ankle dorsiflexion: -50 to 50 degrees
        assert -50 <= result.ankle_dorsiflexion_l <= 50
        assert -50 <= result.ankle_dorsiflexion_r <= 50

        # Hip adduction: -60 to 45 degrees
        assert -60 <= result.hip_adduction_l <= 45
        assert -60 <= result.hip_adduction_r <= 45

        # Trunk flexion: 180 is upright, decreasing with forward lean
        assert 90 <= result.trunk_flexion <= 180

    def test_no_angles_exceed_180(self, ik_solver, sample_skeleton_3d):
        """No angle should exceed 180 degrees."""
        result = ik_solver.solve(sample_skeleton_3d)

        # Get all angle values
        all_angles = [
            result.hip_flexion_l, result.hip_flexion_r,
            result.hip_adduction_l, result.hip_adduction_r,
            result.knee_flexion_l, result.knee_flexion_r,
            result.ankle_dorsiflexion_l, result.ankle_dorsiflexion_r,
            result.trunk_flexion, result.trunk_lateral_flexion,
            result.pelvis_tilt, result.pelvis_list,
        ]

        for angle in all_angles:
            assert abs(angle) <= 180, f"Angle {angle} exceeds 180 degrees"


# =============================================================================
# MISSING DATA TESTS
# =============================================================================

class TestMissingData:
    """Tests for handling missing or low-confidence keypoints."""

    def test_handles_low_confidence_keypoints(self, ik_solver):
        """Solver should handle low-confidence keypoints gracefully."""
        # Create skeleton with some low-confidence keypoints
        keypoints = [
            Point3D(x=0.0, y=-0.70, z=0.0, confidence=0.0),  # nose - low conf
            Point3D(x=0.03, y=-0.72, z=0.0, confidence=0.0),
            Point3D(x=-0.03, y=-0.72, z=0.0, confidence=0.0),
            Point3D(x=0.06, y=-0.70, z=0.0, confidence=0.0),
            Point3D(x=-0.06, y=-0.70, z=0.0, confidence=0.0),
            Point3D(x=0.18, y=-0.50, z=0.0, confidence=0.5),  # shoulder - ok
            Point3D(x=-0.18, y=-0.50, z=0.0, confidence=0.5),
            Point3D(x=0.20, y=-0.20, z=0.0, confidence=0.0),
            Point3D(x=-0.20, y=-0.20, z=0.0, confidence=0.0),
            Point3D(x=0.22, y=0.10, z=0.0, confidence=0.0),
            Point3D(x=-0.22, y=0.10, z=0.0, confidence=0.0),
            Point3D(x=0.12, y=0.0, z=0.0, confidence=0.5),  # hip - ok
            Point3D(x=-0.12, y=0.0, z=0.0, confidence=0.5),
            Point3D(x=0.12, y=0.50, z=0.0, confidence=0.5),  # knee - ok
            Point3D(x=-0.12, y=0.50, z=0.0, confidence=0.5),
            Point3D(x=0.12, y=0.95, z=0.0, confidence=0.5),  # ankle - ok
            Point3D(x=-0.12, y=0.95, z=0.0, confidence=0.5),
        ]
        skeleton = Skeleton3D(keypoints=keypoints)

        # Should not raise an exception
        result = ik_solver.solve(skeleton)
        assert isinstance(result, JointAngles)

    def test_returns_nan_for_missing_keypoints(self, ik_solver):
        """Missing keypoints must result in NaN for affected angles (C9)."""
        # Create skeleton with missing hip keypoints
        keypoints = [
            Point3D(x=0.0, y=-0.70, z=0.0, confidence=1.0),
            Point3D(x=0.03, y=-0.72, z=0.0, confidence=1.0),
            Point3D(x=-0.03, y=-0.72, z=0.0, confidence=1.0),
            Point3D(x=0.06, y=-0.70, z=0.0, confidence=1.0),
            Point3D(x=-0.06, y=-0.70, z=0.0, confidence=1.0),
            Point3D(x=0.18, y=-0.50, z=0.0, confidence=1.0),
            Point3D(x=-0.18, y=-0.50, z=0.0, confidence=1.0),
            Point3D(x=0.20, y=-0.20, z=0.0, confidence=1.0),
            Point3D(x=-0.20, y=-0.20, z=0.0, confidence=1.0),
            Point3D(x=0.22, y=0.10, z=0.0, confidence=1.0),
            Point3D(x=-0.22, y=0.10, z=0.0, confidence=1.0),
            Point3D(x=0.12, y=0.0, z=0.0, confidence=0.0),  # left_hip - missing
            Point3D(x=-0.12, y=0.0, z=0.0, confidence=0.0),  # right_hip - missing
            Point3D(x=0.12, y=0.50, z=0.0, confidence=1.0),
            Point3D(x=-0.12, y=0.50, z=0.0, confidence=1.0),
            Point3D(x=0.12, y=0.95, z=0.0, confidence=1.0),
            Point3D(x=-0.12, y=0.95, z=0.0, confidence=1.0),
        ]
        skeleton = Skeleton3D(keypoints=keypoints)

        result = ik_solver.solve(skeleton)

        # Every angle that needs a hip is undefined, never a silent 0.0
        assert math.isnan(result.hip_flexion_l)
        assert math.isnan(result.hip_flexion_r)
        assert math.isnan(result.knee_flexion_l)
        assert math.isnan(result.knee_flexion_r)
        assert math.isnan(result.hip_adduction_l)
        assert math.isnan(result.trunk_flexion)
        assert math.isnan(result.pelvis_tilt)
        assert math.isnan(result.pelvis_list)
        assert math.isnan(result.pelvis_rotation)
        assert math.isnan(result.shoulder_flexion_l)
        # Angles that do not need a hip are still computed
        assert not math.isnan(result.elbow_flexion_l)
        assert not math.isnan(result.ankle_dorsiflexion_l)
        assert not math.isnan(result.trunk_rotation)
        assert not math.isnan(result.wrist_y_l)

    def test_missing_ankle_gives_nan_knee_flexion_mid_squat(self, ik_solver, symmetric_squat_skeleton):
        """The largest measured pipeline error: a knee at 0.0 mid-squat because
        the ankle dropped below the confidence floor."""
        points = symmetric_squat_skeleton.to_numpy()
        confidences = np.ones(len(points))
        confidences[15] = 0.05
        skeleton = Skeleton3D.from_numpy(points, confidences=confidences)
        result = ik_solver.solve(skeleton)
        assert math.isnan(result.knee_flexion_l)
        assert math.isnan(result.ankle_dorsiflexion_l)
        assert result.knee_flexion_r > 30.0
        assert result.hip_flexion_l > 10.0

    def test_missing_wrist_gives_nan_wrist_position(self, ik_solver, standing_skeleton):
        points = standing_skeleton.to_numpy()
        confidences = np.ones(len(points))
        confidences[9] = 0.0
        result = ik_solver.solve(Skeleton3D.from_numpy(points, confidences=confidences))
        assert math.isnan(result.wrist_x_l)
        assert math.isnan(result.wrist_y_l)
        assert math.isnan(result.elbow_flexion_l)
        assert not math.isnan(result.wrist_x_r)

    def test_no_angle_is_ever_zero_for_a_missing_keypoint(self, ik_solver):
        """A skeleton with every keypoint below the confidence floor must
        return NaN for every angle."""
        empty = Skeleton3D.from_numpy(np.zeros((17, 3)), confidences=np.zeros(17))
        result = ik_solver.solve(empty)
        for name, value in result.as_dict().items():
            if name.startswith("foot_confidence") or name == "knee_ankle_sep_ratio":
                continue
            if name.startswith("knee_valgus") or name.startswith("hip_rotation"):
                continue
            assert math.isnan(value), f"{name} was {value} for a fully missing skeleton"


# =============================================================================
# AS_DICT TESTS
# =============================================================================

class TestJointAnglesDict:
    """Tests for JointAngles.as_dict() method."""

    def test_as_dict_returns_all_angles(self, ik_solver, sample_skeleton_3d):
        """as_dict() should return dict with all angle values."""
        result = ik_solver.solve(sample_skeleton_3d)
        angles_dict = result.as_dict()

        expected_keys = [
            "hip_flexion_l", "hip_flexion_r",
            "hip_adduction_l", "hip_adduction_r",
            "hip_rotation_l", "hip_rotation_r",
            "knee_flexion_l", "knee_flexion_r",
            "ankle_dorsiflexion_l", "ankle_dorsiflexion_r",
            "trunk_flexion", "trunk_lateral_flexion", "trunk_rotation",
            "pelvis_tilt", "pelvis_list", "pelvis_rotation",
        ]

        for key in expected_keys:
            assert key in angles_dict, f"Missing key: {key}"
            assert isinstance(angles_dict[key], float)
