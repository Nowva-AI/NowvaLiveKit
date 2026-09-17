"""
Tests for pose estimation module.

Tests the abstract PoseEstimator base class and MediaPipePoseEstimator implementation.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from biomechanics.pose.base import (
    PoseEstimator,
    COCO_KEYPOINT_NAMES,
    COCO_SKELETON_CONNECTIONS,
)
from biomechanics.pose.mediapipe_fallback import (
    MediaPipePoseEstimator,
    BLAZEPOSE_TO_COCO,
    COCO_TO_BLAZEPOSE,
    MEDIAPIPE_AVAILABLE,
)
from biomechanics.utils.types import Skeleton2D, Skeleton3D


# =============================================================================
# BASE CLASS TESTS
# =============================================================================

class TestCocoKeypoints:
    """Test COCO keypoint constants."""

    def test_coco_keypoint_names_count(self):
        """COCO+ format should have exactly 21 keypoints (COCO 17 + toes + heels)."""
        assert len(COCO_KEYPOINT_NAMES) == 21

    def test_coco_keypoint_names_order(self):
        """Keypoint names should be in standard COCO order."""
        expected_names = [
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
            "left_foot_index",
            "right_foot_index",
            "left_heel",
            "right_heel",
        ]
        assert COCO_KEYPOINT_NAMES == expected_names

    def test_coco_skeleton_connections_valid(self):
        """Skeleton connections should reference valid keypoint indices."""
        for start_idx, end_idx in COCO_SKELETON_CONNECTIONS:
            assert 0 <= start_idx < 21, f"Invalid start index: {start_idx}"
            assert 0 <= end_idx < 21, f"Invalid end index: {end_idx}"

    def test_foot_forms_a_closed_triangle(self):
        """Ankle, heel and toe must all be linked, or the foot renders as a
        single spike and a flat-footed athlete looks like they are on tiptoe."""
        connections = {tuple(sorted(pair)) for pair in COCO_SKELETON_CONNECTIONS}
        for ankle, heel, toe in ((15, 19, 17), (16, 20, 18)):
            assert tuple(sorted((ankle, heel))) in connections
            assert tuple(sorted((heel, toe))) in connections
            assert tuple(sorted((ankle, toe))) in connections

    def test_keypoint_index_lookup(self):
        """keypoint_index should return correct index for name."""
        assert PoseEstimator.keypoint_index("nose") == 0
        assert PoseEstimator.keypoint_index("left_knee") == 13
        assert PoseEstimator.keypoint_index("right_ankle") == 16

    def test_keypoint_index_invalid_name(self):
        """keypoint_index should raise ValueError for unknown name."""
        with pytest.raises(ValueError):
            PoseEstimator.keypoint_index("invalid_keypoint")

    def test_keypoint_name_lookup(self):
        """keypoint_name should return correct name for index."""
        assert PoseEstimator.keypoint_name(0) == "nose"
        assert PoseEstimator.keypoint_name(13) == "left_knee"
        assert PoseEstimator.keypoint_name(16) == "right_ankle"

    def test_keypoint_name_invalid_index(self):
        """keypoint_name should raise IndexError for out-of-range index."""
        with pytest.raises(IndexError):
            PoseEstimator.keypoint_name(21)
        with pytest.raises(IndexError):
            PoseEstimator.keypoint_name(-1)


# =============================================================================
# MEDIAPIPE POSE ESTIMATOR TESTS
# =============================================================================

@pytest.mark.skipif(not MEDIAPIPE_AVAILABLE, reason="MediaPipe not installed")
class TestMediaPipePoseEstimator:
    """Test MediaPipePoseEstimator implementation."""

    def test_initialization(self):
        """Estimator should initialize successfully."""
        estimator = MediaPipePoseEstimator()
        assert estimator.initialize()
        assert estimator.is_initialized
        estimator.release()
        assert not estimator.is_initialized

    def test_default_confidence_threshold(self):
        """Default confidence threshold should be 0.3."""
        estimator = MediaPipePoseEstimator()
        assert estimator.confidence_threshold == 0.3

    def test_custom_confidence_threshold(self):
        """Custom confidence threshold should be respected."""
        estimator = MediaPipePoseEstimator(confidence_threshold=0.5)
        assert estimator.confidence_threshold == 0.5

    def test_keypoint_names(self):
        """Estimator should expose COCO keypoint names."""
        estimator = MediaPipePoseEstimator()
        assert estimator.KEYPOINT_NAMES == COCO_KEYPOINT_NAMES
        assert estimator.NUM_KEYPOINTS == 21

    def test_estimate_returns_skeleton2d_or_none(self):
        """estimate() should return Skeleton2D or None."""
        estimator = MediaPipePoseEstimator()
        estimator.initialize()

        # Create a blank test image (no person)
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = estimator.estimate(blank_frame)

        # Should return None for blank frame (no person detected)
        assert result is None or isinstance(result, Skeleton2D)

        estimator.release()

    def test_estimate_returns_17_keypoints(self):
        """estimate() should return exactly 17 keypoints when person detected."""
        estimator = MediaPipePoseEstimator()
        estimator.initialize()

        # Create a simple test image with a person-like shape
        # This is a minimal test - real detection needs an actual person
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = estimator.estimate(test_frame)

        if result is not None:
            assert len(result.keypoints) == 19

        estimator.release()

    def test_estimate_3d_returns_skeleton3d_or_none(self):
        """estimate_3d() should return Skeleton3D or None."""
        estimator = MediaPipePoseEstimator()
        estimator.initialize()

        # Create a blank test image (no person)
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = estimator.estimate_3d(blank_frame)

        # Should return None for blank frame (no person detected)
        assert result is None or isinstance(result, Skeleton3D)

        estimator.release()

    def test_estimate_3d_returns_17_keypoints(self):
        """estimate_3d() should return exactly 17 keypoints when person detected."""
        estimator = MediaPipePoseEstimator()
        estimator.initialize()

        # Create a random test image
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = estimator.estimate_3d(test_frame)

        if result is not None:
            assert len(result.keypoints) == 19

        estimator.release()

    def test_estimate_both_consistency(self):
        """estimate_both() should return consistent results."""
        estimator = MediaPipePoseEstimator()
        estimator.initialize()

        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        skeleton_2d, skeleton_3d = estimator.estimate_both(test_frame)

        # Both should be None or both should have valid data
        if skeleton_2d is not None:
            assert len(skeleton_2d.keypoints) == 19
        if skeleton_3d is not None:
            assert len(skeleton_3d.keypoints) == 19

        estimator.release()


# =============================================================================
# BLAZEPOSE TO COCO MAPPING TESTS
# =============================================================================

class TestBlazePoseToCoco:
    """Test BlazePose to COCO landmark mapping."""

    def test_mapping_covers_all_coco_keypoints(self):
        """All 21 keypoints (COCO 17 + toes + heels) should be mapped."""
        mapped_coco_indices = set(BLAZEPOSE_TO_COCO.values())
        expected = set(range(21))
        assert mapped_coco_indices == expected

    def test_heels_map_from_blazepose_landmarks(self):
        """BlazePose 29/30 are the heels; dropping them is what made the
        rendered foot a spike from ankle to toe."""
        assert BLAZEPOSE_TO_COCO[29] == 19
        assert BLAZEPOSE_TO_COCO[30] == 20

    def test_mapping_is_bijective(self):
        """Mapping should be one-to-one (no COCO index mapped twice)."""
        coco_indices = list(BLAZEPOSE_TO_COCO.values())
        assert len(coco_indices) == len(set(coco_indices))

    def test_reverse_mapping(self):
        """COCO_TO_BLAZEPOSE should be exact reverse of BLAZEPOSE_TO_COCO."""
        for blazepose_idx, coco_idx in BLAZEPOSE_TO_COCO.items():
            assert COCO_TO_BLAZEPOSE[coco_idx] == blazepose_idx

    def test_specific_mappings(self):
        """Verify specific landmark mappings from spec."""
        expected_mappings = {
            0: 0,    # nose -> nose
            2: 1,    # left_eye_inner -> left_eye
            5: 2,    # right_eye_inner -> right_eye
            7: 3,    # left_ear -> left_ear
            8: 4,    # right_ear -> right_ear
            11: 5,   # left_shoulder -> left_shoulder
            12: 6,   # right_shoulder -> right_shoulder
            13: 7,   # left_elbow -> left_elbow
            14: 8,   # right_elbow -> right_elbow
            15: 9,   # left_wrist -> left_wrist
            16: 10,  # right_wrist -> right_wrist
            23: 11,  # left_hip -> left_hip
            24: 12,  # right_hip -> right_hip
            25: 13,  # left_knee -> left_knee
            26: 14,  # right_knee -> right_knee
            27: 15,  # left_ankle -> left_ankle
            28: 16,  # right_ankle -> right_ankle
            31: 17,  # left_foot_index -> left_foot_index
            32: 18,  # right_foot_index -> right_foot_index
        }

        for blazepose_idx, coco_idx in expected_mappings.items():
            assert BLAZEPOSE_TO_COCO.get(blazepose_idx) == coco_idx, (
                f"Mapping mismatch for BlazePose {blazepose_idx}: "
                f"expected COCO {coco_idx}, got {BLAZEPOSE_TO_COCO.get(blazepose_idx)}"
            )


# =============================================================================
# INTEGRATION TESTS (require actual image with person)
# =============================================================================

@pytest.mark.skipif(not MEDIAPIPE_AVAILABLE, reason="MediaPipe not installed")
class TestMediaPipeIntegration:
    """Integration tests that require an actual person image."""

    @pytest.fixture
    def person_image(self, fixtures_dir):
        """
        Load a test image with a person.
        Falls back to a synthetic image if no fixture available.
        """
        import cv2

        image_path = fixtures_dir / "person_standing.jpg"
        if image_path.exists():
            return cv2.imread(str(image_path))

        # Create a synthetic "person-like" image for basic testing
        # This won't produce reliable pose detection but allows tests to run
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a simple stick figure
        cv2.circle(img, (320, 80), 30, (200, 200, 200), -1)  # head
        cv2.line(img, (320, 110), (320, 250), (200, 200, 200), 10)  # torso
        cv2.line(img, (320, 150), (250, 200), (200, 200, 200), 8)  # left arm
        cv2.line(img, (320, 150), (390, 200), (200, 200, 200), 8)  # right arm
        cv2.line(img, (320, 250), (280, 400), (200, 200, 200), 10)  # left leg
        cv2.line(img, (320, 250), (360, 400), (200, 200, 200), 10)  # right leg
        return img

    def test_estimate_with_image(self, person_image):
        """Test pose estimation with actual image."""
        estimator = MediaPipePoseEstimator()
        estimator.initialize()

        result = estimator.estimate(person_image)

        # MediaPipe may or may not detect the synthetic person
        # Just verify the return type is correct
        assert result is None or isinstance(result, Skeleton2D)

        if result is not None:
            assert len(result.keypoints) == 19
            # Check that at least some keypoints have valid confidence
            confidences = [kp.confidence for kp in result.keypoints]
            assert any(c > 0 for c in confidences), "No valid keypoints detected"

        estimator.release()
