"""
Tests for mode-aware knee valgus estimation module.

Validates SingleCameraValgusEstimator (2D knee deviation from the
knees-over-toes expectation of the monocular 3D pose, as femur tilt) and
TriangulatedValgusEstimator (3D deviation from the knees-over-toes plane +
hip-IR) against synthetic skeletons with known geometry.
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
from biomechanics.utils.types import (
    CocoKeypoints as CK,
    Keypoint2D,
    Point3D,
    Skeleton2D,
    Skeleton3D,
)


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

# Single-camera synthetic capture: an orthographic frontal view of the 3D pose.
IMAGE_PX_PER_M = 500.0
IMAGE_CENTER_X_PX = 640.0
IMAGE_CENTER_Y_PX = 360.0
# Monocular depth error: the Z axis comes out this fraction of its true length.
MONOCULAR_DEPTH_SCALE = 0.6
MONO_TOE_OUT_ANGLES_DEG = (0.0, 10.0, 20.0, 30.0, 45.0)
MONO_KNEE_FLEXIONS_DEG = (60.0, 100.0, 120.0)
LARGE_SWIVEL_DEG = 20.0
# Knees tracking the toes must read neutral, not just "below mild".
MONO_NEUTRAL_TOLERANCE_DEG = 0.25
# Single-camera and triangulated readings share one scale (femur tilt).
MONO_TRI_AGREEMENT_DEG = 0.5
# Readings that must not move at all (camera distance, image mirroring).
INVARIANCE_TOLERANCE_DEG = 1e-3
# Depth compression may under-read a cave but must never inflate it.
COMPRESSION_OVER_READ_TOLERANCE_DEG = 0.05
# A per-frame thigh shortened by depth compression inflates the reading by more than this.
COMPRESSED_THIGH_OVER_READ_DEG = 1.0
MIN_SWIVEL_READING_DEG = 5.0
CONFIDENCE_TOLERANCE = 1e-6
SYNTHETIC_CONFIDENCE = 0.9
LOW_TOE_CONFIDENCE = 0.25


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
    return Skeleton3D.from_numpy(points, confidences=np.full(19, SYNTHETIC_CONFIDENCE))


def _project_frontal(
    skeleton_3d: Skeleton3D, px_per_m: float = IMAGE_PX_PER_M, mirror: bool = False,
) -> Skeleton2D:
    """Orthographic image of the pose from a camera in front of the subject.

    Image y follows world Y (down); image x follows world X, or -X when mirrored.
    """
    x_sign = -1.0 if mirror else 1.0
    return Skeleton2D(keypoints=[
        Keypoint2D(
            x=IMAGE_CENTER_X_PX + x_sign * px_per_m * point.x,
            y=IMAGE_CENTER_Y_PX + px_per_m * point.y,
            confidence=point.confidence,
        )
        for point in skeleton_3d.keypoints
    ])


def _compress_depth(skeleton_3d: Skeleton3D, depth_scale: float = MONOCULAR_DEPTH_SCALE) -> Skeleton3D:
    """The monocular 3D pose with its weak depth axis shrunk; X and Y stay right."""
    return Skeleton3D(keypoints=[
        Point3D(x=point.x, y=point.y, z=point.z * depth_scale, confidence=point.confidence)
        for point in skeleton_3d.keypoints
    ])


def _mono_squat(
    toe_out_deg: float, swivel_deg: float, knee_flexion_deg: float = KNEE_FLEXION_DEG,
) -> tuple[Skeleton2D, Skeleton3D]:
    """Matching 2D image + monocular 3D pose of the same bottom-of-squat legs."""
    skeleton_3d = _squat_skeleton_3d(toe_out_deg, swivel_deg, knee_flexion_deg)
    return _project_frontal(skeleton_3d), skeleton_3d


class TestSingleCameraValgusEstimator:
    """Knee deviation from where the knee would project if it tracked the toes."""

    @pytest.mark.parametrize("knee_flexion_deg", MONO_KNEE_FLEXIONS_DEG)
    @pytest.mark.parametrize("toe_out_deg", MONO_TOE_OUT_ANGLES_DEG)
    def test_knees_over_toes_read_neutral_at_any_toe_out(
        self, toe_out_deg: float, knee_flexion_deg: float,
    ):
        """The fixed toe-out bias: raw FPPA read knees tracking toes turned out
        20 degrees as ~35 degrees lateral, so a real cave could never fire."""
        skeleton_2d, skeleton_3d = _mono_squat(toe_out_deg, 0.0, knee_flexion_deg)
        result = SingleCameraValgusEstimator().estimate(skeleton_2d, skeleton_3d)
        assert result.valgus_l == pytest.approx(0.0, abs=MONO_NEUTRAL_TOLERANCE_DEG)
        assert result.valgus_r == pytest.approx(0.0, abs=MONO_NEUTRAL_TOLERANCE_DEG)

    def test_toed_out_knees_sit_wide_in_the_image_yet_read_neutral(self):
        """Non-trivial neutral: in the image the knees track well outside the
        ankles, which the old raw projection angle read as strong varus."""
        skeleton_2d, skeleton_3d = _mono_squat(30.0, 0.0)
        result = SingleCameraValgusEstimator().estimate(skeleton_2d, skeleton_3d)
        assert result.kasr > 1.3
        assert result.valgus_l == pytest.approx(0.0, abs=MONO_NEUTRAL_TOLERANCE_DEG)
        assert result.valgus_r == pytest.approx(0.0, abs=MONO_NEUTRAL_TOLERANCE_DEG)

    @pytest.mark.parametrize("toe_out_deg", TOE_OUT_ANGLES_DEG)
    def test_knee_cave_reads_positive_on_both_legs(self, toe_out_deg: float):
        result = SingleCameraValgusEstimator().estimate(*_mono_squat(toe_out_deg, KNEE_SWIVEL_DEG))
        assert result.valgus_l > MIN_SWIVEL_READING_DEG
        assert result.valgus_r > MIN_SWIVEL_READING_DEG

    @pytest.mark.parametrize("toe_out_deg", TOE_OUT_ANGLES_DEG)
    def test_varus_reads_negative_on_both_legs(self, toe_out_deg: float):
        result = SingleCameraValgusEstimator().estimate(*_mono_squat(toe_out_deg, -KNEE_SWIVEL_DEG))
        assert result.valgus_l < -MIN_SWIVEL_READING_DEG
        assert result.valgus_r < -MIN_SWIVEL_READING_DEG

    def test_cave_and_varus_are_antisymmetric(self):
        cave = SingleCameraValgusEstimator().estimate(*_mono_squat(15.0, KNEE_SWIVEL_DEG))
        varus = SingleCameraValgusEstimator().estimate(*_mono_squat(15.0, -KNEE_SWIVEL_DEG))
        assert cave.valgus_l == pytest.approx(-varus.valgus_l, abs=MONO_TRI_AGREEMENT_DEG)
        assert cave.valgus_r == pytest.approx(-varus.valgus_r, abs=MONO_TRI_AGREEMENT_DEG)

    def test_one_sided_cave_reads_on_that_leg_only(self):
        _, caved = _mono_squat(20.0, KNEE_SWIVEL_DEG)
        _, skeleton_3d = _mono_squat(20.0, 0.0)
        skeleton_3d.keypoints[CK.LEFT_KNEE] = caved.keypoints[CK.LEFT_KNEE]
        result = SingleCameraValgusEstimator().estimate(_project_frontal(skeleton_3d), skeleton_3d)
        assert result.valgus_l > MIN_SWIVEL_READING_DEG
        assert result.valgus_r == pytest.approx(0.0, abs=MONO_NEUTRAL_TOLERANCE_DEG)

    def test_mirrored_image_reads_the_same(self):
        """A front camera shows the subject's left on the image right; the
        medial direction comes from the feet, so mirroring must not flip sign."""
        skeleton_3d = _squat_skeleton_3d(20.0, KNEE_SWIVEL_DEG)
        direct = SingleCameraValgusEstimator().estimate(_project_frontal(skeleton_3d), skeleton_3d)
        mirrored = SingleCameraValgusEstimator().estimate(
            _project_frontal(skeleton_3d, mirror=True), skeleton_3d,
        )
        assert mirrored.valgus_l == pytest.approx(direct.valgus_l, abs=INVARIANCE_TOLERANCE_DEG)
        assert mirrored.valgus_r == pytest.approx(direct.valgus_r, abs=INVARIANCE_TOLERANCE_DEG)

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

        The hip-ankle line at the knee's height is undefined when the vertical
        span collapses, so the measurement is unusable and is NaN.
        """
        skeleton_2d, skeleton_3d = _mono_squat(15.0, 0.0)
        left_hip = skeleton_2d.keypoints[CK.LEFT_HIP]
        skeleton_2d.keypoints[CK.LEFT_ANKLE].y = left_hip.y + 5.0
        result = SingleCameraValgusEstimator().estimate(skeleton_2d, skeleton_3d)
        assert math.isnan(result.valgus_l)
        assert result.valgus_r == pytest.approx(0.0, abs=MONO_NEUTRAL_TOLERANCE_DEG)

    def test_facing_confidence_drops_when_rotated(self):
        """Collapsed hip separation (side view) lowers confidence."""
        skeleton_2d, skeleton_3d = _mono_squat(15.0, 0.0)
        skeleton_2d.keypoints[CK.LEFT_HIP].x = IMAGE_CENTER_X_PX + 2.0
        skeleton_2d.keypoints[CK.RIGHT_HIP].x = IMAGE_CENTER_X_PX - 2.0
        result = SingleCameraValgusEstimator().estimate(skeleton_2d, skeleton_3d)
        assert result.foot_confidence_l < 0.5
        assert result.foot_confidence_r < 0.5

    def test_frontal_view_keeps_full_keypoint_confidence(self):
        result = SingleCameraValgusEstimator().estimate(*_mono_squat(15.0, 0.0))
        assert result.foot_confidence_l == pytest.approx(SYNTHETIC_CONFIDENCE, abs=CONFIDENCE_TOLERANCE)
        assert result.foot_confidence_r == pytest.approx(SYNTHETIC_CONFIDENCE, abs=CONFIDENCE_TOLERANCE)

    def test_foot_confidence_includes_the_toe(self):
        skeleton_2d, skeleton_3d = _mono_squat(15.0, 0.0)
        skeleton_2d.keypoints[CK.LEFT_FOOT_INDEX].confidence = LOW_TOE_CONFIDENCE
        result = SingleCameraValgusEstimator().estimate(skeleton_2d, skeleton_3d)
        assert result.foot_confidence_l == pytest.approx(LOW_TOE_CONFIDENCE, abs=CONFIDENCE_TOLERANCE)
        assert result.foot_confidence_r == pytest.approx(SYNTHETIC_CONFIDENCE, abs=CONFIDENCE_TOLERANCE)

    def test_image_without_toes_has_zero_foot_confidence(self):
        skeleton_2d, skeleton_3d = _mono_squat(15.0, 0.0)
        no_toes = Skeleton2D(keypoints=skeleton_2d.keypoints[:CK.LEFT_FOOT_INDEX])
        result = SingleCameraValgusEstimator().estimate(no_toes, skeleton_3d)
        assert result.foot_confidence_l == 0.0
        assert result.foot_confidence_r == 0.0


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
    """A coach who can't see the feet says nothing: no 3D pose or no toes is NaN."""

    def test_none_skeleton_returns_nan(self):
        result = SingleCameraValgusEstimator().estimate(None)
        assert math.isnan(result.valgus_l)
        assert math.isnan(result.kasr)
        assert result.foot_confidence_l == 0.0

    def test_without_3d_skeleton_valgus_is_nan(self):
        skeleton_2d, _ = _mono_squat(15.0, KNEE_SWIVEL_DEG)
        result = SingleCameraValgusEstimator().estimate(skeleton_2d)
        assert math.isnan(result.valgus_l)
        assert math.isnan(result.valgus_r)

    def test_3d_skeleton_without_toes_gives_nan(self):
        skeleton_2d, skeleton_3d = _mono_squat(15.0, KNEE_SWIVEL_DEG)
        no_toes = Skeleton3D(keypoints=skeleton_3d.keypoints[:CK.LEFT_FOOT_INDEX])
        result = SingleCameraValgusEstimator().estimate(skeleton_2d, no_toes)
        assert math.isnan(result.valgus_l)
        assert math.isnan(result.valgus_r)

    def test_missing_3d_toe_gives_nan_for_that_side_only(self):
        skeleton_2d, skeleton_3d = _mono_squat(15.0, KNEE_SWIVEL_DEG)
        skeleton_3d.keypoints[CK.LEFT_FOOT_INDEX].confidence = 0.0
        result = SingleCameraValgusEstimator().estimate(skeleton_2d, skeleton_3d)
        assert math.isnan(result.valgus_l)
        assert result.valgus_r > MIN_SWIVEL_READING_DEG

    def test_missing_knee_gives_nan_for_that_side(self):
        skeleton_2d, skeleton_3d = _mono_squat(15.0, 0.0)
        skeleton_2d.keypoints[CK.LEFT_KNEE].confidence = 0.0
        result = SingleCameraValgusEstimator().estimate(skeleton_2d, skeleton_3d)
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
        result = SingleCameraValgusEstimator().estimate(*_mono_squat(15.0, KNEE_SWIVEL_DEG / 2.0))
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


class TestSingleCameraDepthAndScale:
    """FPPA was once normalized by the live hip-to-ankle vertical span, which
    collapses on the way down — the same cave read ~1.8x larger at the bottom.
    The reading is now femur tilt out of the knees-over-toes plane: the
    triangulated definition, which does not change with squat depth or with
    how far the camera stands."""

    @pytest.mark.parametrize("knee_flexion_deg", MONO_KNEE_FLEXIONS_DEG)
    def test_reads_the_triangulated_femur_tilt_at_every_depth(self, knee_flexion_deg: float):
        skeleton_2d, skeleton_3d = _mono_squat(0.0, LARGE_SWIVEL_DEG, knee_flexion_deg)
        mono = SingleCameraValgusEstimator().estimate(skeleton_2d, skeleton_3d)
        tri = TriangulatedValgusEstimator().estimate(None, skeleton_3d)
        assert mono.valgus_l == pytest.approx(tri.valgus_l, abs=MONO_TRI_AGREEMENT_DEG)
        assert mono.valgus_r == pytest.approx(tri.valgus_r, abs=MONO_TRI_AGREEMENT_DEG)

    @pytest.mark.parametrize("toe_out_deg", TOE_OUT_ANGLES_DEG)
    def test_scale_invariant_to_camera_distance(self, toe_out_deg: float):
        """Only the image shrinks when the camera backs off; the 3D pose is the same."""
        skeleton_3d = _squat_skeleton_3d(toe_out_deg, KNEE_SWIVEL_DEG)
        estimator = SingleCameraValgusEstimator()
        near = estimator.estimate(_project_frontal(skeleton_3d), skeleton_3d)
        far = estimator.estimate(_project_frontal(skeleton_3d, IMAGE_PX_PER_M / 2.0), skeleton_3d)
        assert far.valgus_l == pytest.approx(near.valgus_l, abs=INVARIANCE_TOLERANCE_DEG)
        assert far.valgus_r == pytest.approx(near.valgus_r, abs=INVARIANCE_TOLERANCE_DEG)


class TestSingleCameraDepthCompression:
    """Monocular depth is the weak axis. It may only set the knees-over-toes
    expectation; with a measured femur it must never invent or inflate a cave."""

    @pytest.mark.parametrize("toe_out_deg", MONO_TOE_OUT_ANGLES_DEG)
    def test_measured_femur_keeps_neutral_under_depth_compression(self, toe_out_deg: float):
        skeleton_3d = _squat_skeleton_3d(toe_out_deg, 0.0)
        estimator = SingleCameraValgusEstimator()
        estimator.set_femur_length(FEMUR_M)
        result = estimator.estimate(_project_frontal(skeleton_3d), _compress_depth(skeleton_3d))
        assert result.valgus_l == pytest.approx(0.0, abs=MONO_NEUTRAL_TOLERANCE_DEG)
        assert result.valgus_r == pytest.approx(0.0, abs=MONO_NEUTRAL_TOLERANCE_DEG)

    @pytest.mark.parametrize("toe_out_deg", MONO_TOE_OUT_ANGLES_DEG)
    def test_measured_femur_never_over_reads_cave_under_depth_compression(self, toe_out_deg: float):
        skeleton_3d = _squat_skeleton_3d(toe_out_deg, LARGE_SWIVEL_DEG)
        skeleton_2d = _project_frontal(skeleton_3d)
        estimator = SingleCameraValgusEstimator()
        estimator.set_femur_length(FEMUR_M)
        true_depth = estimator.estimate(skeleton_2d, skeleton_3d)
        compressed = estimator.estimate(skeleton_2d, _compress_depth(skeleton_3d))
        for side_true, side_compressed in (
            (true_depth.valgus_l, compressed.valgus_l), (true_depth.valgus_r, compressed.valgus_r),
        ):
            assert side_compressed > 0.0
            assert side_compressed <= side_true + COMPRESSION_OVER_READ_TOLERANCE_DEG

    def test_measured_femur_undoes_over_read_from_compressed_thigh(self):
        """With depth compressed the per-frame 3D thigh comes out short, which
        inflates the tilt; the standing femur length restores the true reading."""
        skeleton_3d = _squat_skeleton_3d(0.0, KNEE_SWIVEL_DEG)
        skeleton_2d = _project_frontal(skeleton_3d)
        compressed_3d = _compress_depth(skeleton_3d)
        true_depth = SingleCameraValgusEstimator().estimate(skeleton_2d, skeleton_3d)
        per_frame_thigh = SingleCameraValgusEstimator().estimate(skeleton_2d, compressed_3d)
        measured = SingleCameraValgusEstimator()
        measured.set_femur_length(FEMUR_M)
        measured_femur = measured.estimate(skeleton_2d, compressed_3d)
        assert per_frame_thigh.valgus_l > true_depth.valgus_l + COMPRESSED_THIGH_OVER_READ_DEG
        assert measured_femur.valgus_l == pytest.approx(true_depth.valgus_l, abs=COMPRESSION_OVER_READ_TOLERANCE_DEG)
        assert measured_femur.valgus_r == pytest.approx(true_depth.valgus_r, abs=COMPRESSION_OVER_READ_TOLERANCE_DEG)
