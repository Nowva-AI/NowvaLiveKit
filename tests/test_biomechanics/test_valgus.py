"""
Tests for mode-aware knee valgus estimation module.

Validates SingleCameraValgusEstimator (2D FPPA) and
TriangulatedValgusEstimator (3D abduction + hip-IR decomposition)
against synthetic skeletons with known geometry.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from biomechanics.kinematics.valgus import (
    SingleCameraValgusEstimator,
    TriangulatedValgusEstimator,
    ValgusResult,
    build_valgus_estimator,
)
from biomechanics.utils.types import Keypoint2D, Point3D, Skeleton2D, Skeleton3D


ANGLE_TOLERANCE = 2.0
KASR_TOLERANCE = 0.05
TOE_OUT_ANGLES_DEG = (0.0, 15.0, 30.0)
KNEE_SWIVEL_DEG = 10.0
KNEE_FLEXION_DEG = 100.0
FEMUR_M = 0.46
TIBIA_M = 0.44
HIP_HALF_WIDTH_M = 0.14
STANCE_HALF_WIDTH_M = 0.20
FOOT_LENGTH_M = 0.20


def _make_skeleton_2d(overrides: dict[int, tuple[float, float]] | None = None) -> Skeleton2D:
    """Frontal-view standing skeleton (17 COCO keypoints).

    Hips at y=500, knees at y=700, ankles at y=900, shoulders at y=350.
    Symmetric — knees directly above ankles.
    """
    base = {
        0: (400, 200),   # nose
        1: (410, 190),   # left_eye
        2: (390, 190),   # right_eye
        3: (420, 200),   # left_ear
        4: (380, 200),   # right_ear
        5: (450, 350),   # left_shoulder
        6: (350, 350),   # right_shoulder
        7: (470, 500),   # left_elbow
        8: (330, 500),   # right_elbow
        9: (480, 650),   # left_wrist
        10: (320, 650),  # right_wrist
        11: (440, 500),  # left_hip
        12: (360, 500),  # right_hip
        13: (440, 700),  # left_knee
        14: (360, 700),  # right_knee
        15: (440, 900),  # left_ankle
        16: (360, 900),  # right_ankle
    }
    if overrides:
        base.update(overrides)
    kpts = [Keypoint2D(x=base[i][0], y=base[i][1], confidence=0.9) for i in range(17)]
    return Skeleton2D(keypoints=kpts)


def _make_skeleton_3d(
    overrides: dict[int, tuple[float, float, float]] | None = None,
    n_keypoints: int = 19,
) -> Skeleton3D:
    """Standing skeleton in 3D (Y-down, X = subject's left, +Z = subject's back).

    Hip width 0.24m, knees directly below hips, ankles below knees.
    Includes foot_index keypoints (17, 18) pointing forward (-Z).
    """
    base = {
        0: (0.0, -1.70, 0.0),
        1: (0.03, -1.72, -0.02),
        2: (-0.03, -1.72, -0.02),
        3: (0.06, -1.70, 0.0),
        4: (-0.06, -1.70, 0.0),
        5: (0.18, -1.50, 0.0),
        6: (-0.18, -1.50, 0.0),
        7: (0.20, -1.20, 0.0),
        8: (-0.20, -1.20, 0.0),
        9: (0.22, -0.90, 0.0),
        10: (-0.22, -0.90, 0.0),
        11: (0.12, -1.00, 0.0),    # left_hip
        12: (-0.12, -1.00, 0.0),   # right_hip
        13: (0.12, -0.50, 0.0),    # left_knee
        14: (-0.12, -0.50, 0.0),   # right_knee
        15: (0.12, -0.05, 0.0),    # left_ankle
        16: (-0.12, -0.05, 0.0),   # right_ankle
        17: (0.12, -0.02, -0.15),   # left_foot_index
        18: (-0.12, -0.02, -0.15),  # right_foot_index
    }
    if overrides:
        base.update(overrides)
    kpts = [
        Point3D(x=base[i][0], y=base[i][1], z=base[i][2], confidence=0.9)
        for i in range(n_keypoints)
    ]
    return Skeleton3D(keypoints=kpts, timestamp=0.0, frame_index=0)


class TestSingleCameraValgusEstimator:
    """Tests for 2D FPPA-based valgus estimation."""

    def test_neutral_stance_near_zero(self):
        skel = _make_skeleton_2d()
        est = SingleCameraValgusEstimator()
        result = est.estimate(skel)
        assert abs(result.valgus_l) < ANGLE_TOLERANCE
        assert abs(result.valgus_r) < ANGLE_TOLERANCE

    def test_valgus_positive_when_knees_cave(self):
        """Knees moved medially (toward midline) should produce positive valgus."""
        skel = _make_skeleton_2d({
            13: (420, 700),  # left knee shifted right (medial)
            14: (380, 700),  # right knee shifted left (medial)
        })
        est = SingleCameraValgusEstimator()
        result = est.estimate(skel)
        assert result.valgus_l > 1.0
        assert result.valgus_r > 1.0

    def test_varus_negative_when_knees_bow(self):
        """Knees moved laterally (away from midline) should produce negative valgus."""
        skel = _make_skeleton_2d({
            13: (470, 700),  # left knee shifted further left (lateral)
            14: (330, 700),  # right knee shifted further right (lateral)
        })
        est = SingleCameraValgusEstimator()
        result = est.estimate(skel)
        assert result.valgus_l < -1.0
        assert result.valgus_r < -1.0

    def test_kasr_below_one_when_knees_cave(self):
        skel = _make_skeleton_2d({
            13: (420, 700),
            14: (380, 700),
        })
        est = SingleCameraValgusEstimator()
        result = est.estimate(skel)
        assert result.kasr < 1.0

    def test_kasr_above_one_when_knees_bow(self):
        skel = _make_skeleton_2d({
            13: (470, 700),
            14: (330, 700),
        })
        est = SingleCameraValgusEstimator()
        result = est.estimate(skel)
        assert result.kasr > 1.0

    def test_mistracked_ankle_at_hip_height_returns_nan(self):
        """A confidently-wrong ankle near hip height must not spike the angle.

        The vertical span sits in the arctan2 denominator, so a degenerate
        span would turn ordinary knee-to-ankle x offsets into ~90 degree
        valgus readings. The measurement is unusable, so it is NaN.
        """
        skel = _make_skeleton_2d({
            15: (440, 505),  # left ankle mistracked to just below the hip
        })
        est = SingleCameraValgusEstimator()
        result = est.estimate(skel)
        assert math.isnan(result.valgus_l)
        assert abs(result.valgus_r) < ANGLE_TOLERANCE

    def test_facing_confidence_drops_when_rotated(self):
        """Collapsed hip separation (side view) lowers confidence."""
        skel = _make_skeleton_2d({
            11: (402, 500),  # hips nearly overlapping horizontally
            12: (398, 500),
        })
        est = SingleCameraValgusEstimator()
        result = est.estimate(skel)
        assert result.foot_confidence_l < 0.5
        assert result.foot_confidence_r < 0.5


class TestTriangulatedValgusEstimator:
    """Tests for 3D abduction-based valgus estimation."""

    def test_neutral_stance_near_zero(self):
        skel = _make_skeleton_3d()
        est = TriangulatedValgusEstimator()
        result = est.estimate(None, skel)
        assert abs(result.valgus_l) < ANGLE_TOLERANCE
        assert abs(result.valgus_r) < ANGLE_TOLERANCE

    def test_valgus_positive_when_knees_cave(self):
        """Knees shifted medially in 3D produce positive valgus."""
        skel = _make_skeleton_3d({
            13: (0.06, -0.50, 0.0),   # left knee shifted right (medial)
            14: (-0.06, -0.50, 0.0),  # right knee shifted left (medial)
        })
        est = TriangulatedValgusEstimator()
        result = est.estimate(None, skel)
        assert result.valgus_l > 1.0
        assert result.valgus_r > 1.0

    def test_pure_hip_rotation_reads_near_zero_valgus(self):
        """The moat test: rotate feet inward (hip IR) but keep knees over ankles.

        A single-camera system would see this as valgus because the 2D projection
        of the knee shifts medially. The 3D Grood-Suntay abduction should remain
        near zero because the knee truly is over the ankle — the apparent medial
        shift is purely a rotation artifact.
        """
        skel = _make_skeleton_3d({
            17: (0.06, -0.02, -0.14),   # left foot rotated inward
            18: (-0.06, -0.02, -0.14),  # right foot rotated inward
        })
        est = TriangulatedValgusEstimator()
        result = est.estimate(None, skel)
        assert abs(result.valgus_l) < ANGLE_TOLERANCE
        assert abs(result.valgus_r) < ANGLE_TOLERANCE
        assert result.hip_rotation_l > 5.0
        assert result.hip_rotation_r > 5.0

    def test_feet_rotated_outward_read_negative_hip_rotation(self):
        skel = _make_skeleton_3d({
            17: (0.18, -0.02, -0.14),   # left foot rotated outward
            18: (-0.18, -0.02, -0.14),  # right foot rotated outward
        })
        result = TriangulatedValgusEstimator().estimate(None, skel)
        assert result.hip_rotation_l < -5.0
        assert result.hip_rotation_r < -5.0

    def test_neutral_feet_read_near_zero_hip_rotation(self):
        result = TriangulatedValgusEstimator().estimate(None, _make_skeleton_3d())
        assert abs(result.hip_rotation_l) < ANGLE_TOLERANCE
        assert abs(result.hip_rotation_r) < ANGLE_TOLERANCE

    def test_3d_kasr_below_one_when_knees_cave(self):
        skel = _make_skeleton_3d({
            13: (0.06, -0.50, 0.0),
            14: (-0.06, -0.50, 0.0),
        })
        est = TriangulatedValgusEstimator()
        result = est.estimate(None, skel)
        assert result.kasr < 1.0

    def test_none_skeleton_returns_nan(self):
        est = TriangulatedValgusEstimator()
        result = est.estimate(None, None)
        assert math.isnan(result.valgus_l)
        assert math.isnan(result.valgus_r)
        assert math.isnan(result.kasr)
        assert math.isnan(result.hip_rotation_l)
        assert result.foot_confidence_l == 0.0

    def test_missing_knee_gives_nan_for_that_side_only(self):
        skel = _make_skeleton_3d()
        skel.keypoints[13].confidence = 0.0
        result = TriangulatedValgusEstimator().estimate(None, skel)
        assert math.isnan(result.valgus_l)
        assert math.isnan(result.hip_rotation_l)
        assert math.isnan(result.kasr)
        assert result.foot_confidence_l == 0.0
        assert not math.isnan(result.valgus_r)

    def test_missing_toe_gives_nan_valgus_and_zero_confidence(self):
        skel = _make_skeleton_3d(n_keypoints=17)
        result = TriangulatedValgusEstimator().estimate(None, skel)
        assert math.isnan(result.valgus_l)
        assert result.foot_confidence_l == 0.0


def _two_bone_knee(
    hip: np.ndarray,
    ankle: np.ndarray,
    pole: np.ndarray,
    swivel_deg: float,
    medial: np.ndarray,
) -> np.ndarray:
    """Knee position for a femur/tibia pair bending toward ``pole`` and
    swivelled about the hip-ankle axis by ``swivel_deg`` toward ``medial``."""
    hip_to_ankle = ankle - hip
    dist = float(np.linalg.norm(hip_to_ankle))
    axis = hip_to_ankle / dist
    along = (FEMUR_M ** 2 - TIBIA_M ** 2 + dist ** 2) / (2.0 * dist)
    radius = math.sqrt(max(FEMUR_M ** 2 - along ** 2, 0.0))
    pole_perp = pole - np.dot(pole, axis) * axis
    pole_perp /= np.linalg.norm(pole_perp)
    medial_perp = np.cross(axis, pole_perp)
    if np.dot(medial_perp, medial) < 0:
        medial_perp = -medial_perp
    swivel = math.radians(swivel_deg)
    return hip + along * axis + radius * (math.cos(swivel) * pole_perp + math.sin(swivel) * medial_perp)


def _squat_skeleton_3d(toe_out_deg: float, swivel_deg: float, knee_flexion_deg: float = KNEE_FLEXION_DEG) -> Skeleton3D:
    """Bottom-of-squat legs with symmetric toe-out; both knees swivelled
    medially by ``swivel_deg`` (positive = knee cave) from the plane the knee
    tracks in when it follows the toes. Y-down, X = left, forward = -Z."""
    toe_out = math.radians(toe_out_deg)
    hip_to_ankle = math.sqrt(
        FEMUR_M ** 2 + TIBIA_M ** 2 - 2.0 * FEMUR_M * TIBIA_M * math.cos(math.pi - math.radians(knee_flexion_deg))
    )
    hip_back_m = 0.15
    points = np.zeros((19, 3))
    for hip_index, knee_index, ankle_index, toe_index, side in (
        (11, 13, 15, 17, 1.0), (12, 14, 16, 18, -1.0),
    ):
        foot_dir = np.array([side * math.sin(toe_out), 0.0, -math.cos(toe_out)])
        ankle = np.array([side * STANCE_HALF_WIDTH_M, 0.0, 0.0])
        dx = side * HIP_HALF_WIDTH_M - ankle[0]
        dy = -math.sqrt(max(hip_to_ankle ** 2 - dx ** 2 - hip_back_m ** 2, 1e-6))
        hip = np.array([side * HIP_HALF_WIDTH_M, dy, hip_back_m])
        medial = np.array([-side, 0.0, 0.0])
        points[hip_index] = hip
        points[ankle_index] = ankle
        points[toe_index] = ankle + FOOT_LENGTH_M * foot_dir
        points[knee_index] = _two_bone_knee(hip, ankle, foot_dir, swivel_deg, medial)
    points[5] = points[11] + [0.06, -0.45, 0.0]
    points[6] = points[12] + [-0.06, -0.45, 0.0]
    return Skeleton3D.from_numpy(points, confidences=np.full(19, 0.9))


class TestTriangulatedToeOutInvariance:
    """Magnitude used to come from the pelvis-ML Grood-Suntay axis and sign from
    the hip-ankle line: with 15 degrees of toe-out a 10 degree knee cave read
    -4.7 (varus). Sign and magnitude must come from one geometry, and knees
    tracking over the toes must read neutral at any toe-out angle."""

    @pytest.mark.parametrize("toe_out_deg", TOE_OUT_ANGLES_DEG)
    def test_knees_over_toes_read_neutral(self, toe_out_deg: float):
        result = TriangulatedValgusEstimator().estimate(None, _squat_skeleton_3d(toe_out_deg, 0.0))
        assert result.valgus_l == pytest.approx(0.0, abs=ANGLE_TOLERANCE)
        assert result.valgus_r == pytest.approx(0.0, abs=ANGLE_TOLERANCE)

    @pytest.mark.parametrize("toe_out_deg", TOE_OUT_ANGLES_DEG)
    def test_knee_cave_reads_positive(self, toe_out_deg: float):
        result = TriangulatedValgusEstimator().estimate(None, _squat_skeleton_3d(toe_out_deg, KNEE_SWIVEL_DEG))
        assert result.valgus_l > 5.0
        assert result.valgus_r > 5.0

    @pytest.mark.parametrize("toe_out_deg", TOE_OUT_ANGLES_DEG)
    def test_varus_reads_negative(self, toe_out_deg: float):
        result = TriangulatedValgusEstimator().estimate(None, _squat_skeleton_3d(toe_out_deg, -KNEE_SWIVEL_DEG))
        assert result.valgus_l < -5.0
        assert result.valgus_r < -5.0

    @pytest.mark.parametrize("toe_out_deg", TOE_OUT_ANGLES_DEG)
    def test_same_cave_reads_the_same_at_every_toe_out(self, toe_out_deg: float):
        reference = TriangulatedValgusEstimator().estimate(None, _squat_skeleton_3d(0.0, KNEE_SWIVEL_DEG))
        result = TriangulatedValgusEstimator().estimate(None, _squat_skeleton_3d(toe_out_deg, KNEE_SWIVEL_DEG))
        assert result.valgus_l == pytest.approx(reference.valgus_l, abs=ANGLE_TOLERANCE)

    def test_cave_and_varus_are_antisymmetric(self):
        cave = TriangulatedValgusEstimator().estimate(None, _squat_skeleton_3d(15.0, KNEE_SWIVEL_DEG))
        varus = TriangulatedValgusEstimator().estimate(None, _squat_skeleton_3d(15.0, -KNEE_SWIVEL_DEG))
        assert cave.valgus_l == pytest.approx(-varus.valgus_l, abs=0.5)


class TestSingleCameraMissingInputs:

    def test_none_skeleton_returns_nan(self):
        result = SingleCameraValgusEstimator().estimate(None)
        assert math.isnan(result.valgus_l)
        assert math.isnan(result.kasr)
        assert result.foot_confidence_l == 0.0

    def test_missing_knee_gives_nan_for_that_side(self):
        skel = _make_skeleton_2d()
        skel.keypoints[13].confidence = 0.0
        result = SingleCameraValgusEstimator().estimate(skel)
        assert math.isnan(result.valgus_l)
        assert math.isnan(result.kasr)
        assert not math.isnan(result.valgus_r)
        assert result.foot_confidence_l == 0.0


class TestBuildValgusEstimator:
    """Tests for the factory function."""

    def test_single_camera_mode(self):
        est = build_valgus_estimator(multi_camera=False)
        assert isinstance(est, SingleCameraValgusEstimator)

    def test_multi_camera_mode(self):
        est = build_valgus_estimator(multi_camera=True)
        assert isinstance(est, TriangulatedValgusEstimator)

    def test_result_is_named_tuple(self):
        est = build_valgus_estimator(multi_camera=False)
        result = est.estimate(None)
        assert isinstance(result, ValgusResult)


class TestSignConvention:
    """Verify the non-negotiable sign convention: positive = valgus (medial)."""

    def test_single_camera_sign(self):
        skel = _make_skeleton_2d({
            13: (415, 700),  # left knee slightly medial
            14: (385, 700),  # right knee slightly medial
        })
        est = SingleCameraValgusEstimator()
        result = est.estimate(skel)
        assert result.valgus_l > 0, "Left knee medial shift must be positive"
        assert result.valgus_r > 0, "Right knee medial shift must be positive"

    def test_triangulated_sign(self):
        skel = _make_skeleton_3d({
            13: (0.08, -0.50, 0.0),   # left knee medial
            14: (-0.08, -0.50, 0.0),  # right knee medial
        })
        est = TriangulatedValgusEstimator()
        result = est.estimate(None, skel)
        assert result.valgus_l > 0, "Left knee medial shift must be positive"
        assert result.valgus_r > 0, "Right knee medial shift must be positive"


class TestFPPADepthInvariance:
    """FPPA was normalized by the live hip-to-ankle vertical span, which
    collapses as the athlete descends — identical knee cave read ~1.4x larger
    at parallel and ~1.8x at the bottom. The valgus rule samples only at the
    bottom, so it always measured at maximum inflation. Pelvis width is
    frontal and holds across depth."""

    CAVE_PX = 24.0

    def _rep_at_depth(self, hip_y: float) -> Skeleton2D:
        """Same physical knee cave, hips descending toward the ankles."""
        knee_y = (hip_y + 900) / 2.0
        return _make_skeleton_2d({
            11: (440, hip_y), 12: (360, hip_y),
            13: (440 - self.CAVE_PX, knee_y), 14: (360 + self.CAVE_PX, knee_y),
        })

    def test_valgus_is_constant_across_depth(self):
        estimator = SingleCameraValgusEstimator()
        standing = estimator.estimate(self._rep_at_depth(500))
        parallel = estimator.estimate(self._rep_at_depth(700))
        bottom = estimator.estimate(self._rep_at_depth(800))

        assert standing.valgus_l == pytest.approx(parallel.valgus_l, abs=0.5)
        assert standing.valgus_l == pytest.approx(bottom.valgus_l, abs=0.5)
        assert standing.valgus_r == pytest.approx(bottom.valgus_r, abs=0.5)

    def test_medial_knee_is_positive(self):
        result = SingleCameraValgusEstimator().estimate(self._rep_at_depth(700))
        assert result.valgus_l > 0
        assert result.valgus_r > 0

    def test_no_cave_reads_zero(self):
        estimator = SingleCameraValgusEstimator()
        result = estimator.estimate(_make_skeleton_2d())
        assert result.valgus_l == pytest.approx(0.0, abs=ANGLE_TOLERANCE)
        assert result.valgus_r == pytest.approx(0.0, abs=ANGLE_TOLERANCE)

    def test_scale_invariant_to_camera_distance(self):
        """Both the deviation and pelvis width are in pixels, so standing
        twice as far away must not change the reading."""
        estimator = SingleCameraValgusEstimator()
        near = estimator.estimate(self._rep_at_depth(700))
        far = _make_skeleton_2d({
            11: (420, 600), 12: (380, 600),
            13: (420 - self.CAVE_PX / 2, 700), 14: (380 + self.CAVE_PX / 2, 700),
            15: (420, 800), 16: (380, 800),
        })
        assert estimator.estimate(far).valgus_l == pytest.approx(
            near.valgus_l, abs=0.5,
        )
