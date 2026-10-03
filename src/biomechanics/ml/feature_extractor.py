"""
Landmark Feature Extractor

Extracts a fixed-length, body-size-normalized feature vector from a Skeleton3D.
Features: joint angles + normalized bone lengths + normalized vertical displacements.

The normalization factor (torso or thigh length) removes scale variance from
camera distance and body size differences. Features are in the training
convention: Y-up (the synthetic training skeletons). Live skeletons are Y-down
(geometry.WORLD_UP), so the live extractor flips their vertical differences.
"""

import numpy as np

from biomechanics.utils.types import Skeleton3D, CocoKeypoints as CK
from biomechanics.utils.geometry import joint_angle_3_points, distance_3d, midpoint


# Pairs for normalization factor computation (tried in order, first positive wins)
_NORM_PAIRS = [
    (CK.LEFT_SHOULDER, CK.LEFT_HIP),
    (CK.RIGHT_SHOULDER, CK.RIGHT_HIP),
    (CK.LEFT_HIP, CK.LEFT_KNEE),
    (CK.RIGHT_HIP, CK.RIGHT_KNEE),
]

_NORM_FALLBACK = 0.5

# Bone-length pairs for normalized distance features
_BONE_PAIRS = [
    (CK.LEFT_HIP, CK.LEFT_KNEE),
    (CK.RIGHT_HIP, CK.RIGHT_KNEE),
    (CK.LEFT_KNEE, CK.LEFT_ANKLE),
    (CK.RIGHT_KNEE, CK.RIGHT_ANKLE),
    (CK.LEFT_SHOULDER, CK.RIGHT_SHOULDER),
    (CK.LEFT_HIP, CK.RIGHT_HIP),
]


class LandmarkFeatureExtractor:
    """
    Extracts a fixed-length feature vector from a Skeleton3D.

    Feature vector layout (14 floats):
        [0:4]   - Joint angles: left_knee, right_knee, left_hip, right_hip (degrees)
        [4:10]  - Normalized bone lengths (6 pairs)
        [10:14] - Normalized y-differences (hip_mid, knee_mid, ankle_mid, shoulder_mid
                  relative to hip_mid)
    """

    FEATURE_DIM = 14

    def __init__(self, input_y_down: bool = False):
        self._vertical_sign = -1.0 if input_y_down else 1.0

    def extract(self, skeleton: Skeleton3D) -> np.ndarray:
        """
        Extract feature vector from a single frame's skeleton.

        Args:
            skeleton: Skeleton3D with COCO 17 keypoints in world coordinates.

        Returns:
            numpy array of shape (FEATURE_DIM,)
        """
        return self.extract_points(skeleton.to_numpy())

    def extract_points(self, kpts: np.ndarray) -> np.ndarray:
        """Feature vector from an (N, 3) keypoint array."""
        norm = self._compute_normalization_factor(kpts)
        angles = self._compute_angles(kpts)
        distances = self._compute_normalized_distances(kpts, norm)
        y_diffs = self._compute_normalized_y_diffs(kpts, norm)

        return np.concatenate([angles, distances, y_diffs])

    def _compute_normalization_factor(self, kpts: np.ndarray) -> float:
        """First positive distance among torso/thigh pairs, fallback 0.5."""
        for i, j in _NORM_PAIRS:
            d = distance_3d(kpts[i], kpts[j])
            if d > 1e-6:
                return d
        return _NORM_FALLBACK

    def _compute_angles(self, kpts: np.ndarray) -> np.ndarray:
        """Compute 4 joint angles: left/right knee, left/right hip."""
        left_knee = joint_angle_3_points(
            kpts[CK.LEFT_HIP], kpts[CK.LEFT_KNEE], kpts[CK.LEFT_ANKLE]
        )
        right_knee = joint_angle_3_points(
            kpts[CK.RIGHT_HIP], kpts[CK.RIGHT_KNEE], kpts[CK.RIGHT_ANKLE]
        )
        left_hip = joint_angle_3_points(
            kpts[CK.LEFT_SHOULDER], kpts[CK.LEFT_HIP], kpts[CK.LEFT_KNEE]
        )
        right_hip = joint_angle_3_points(
            kpts[CK.RIGHT_SHOULDER], kpts[CK.RIGHT_HIP], kpts[CK.RIGHT_KNEE]
        )
        return np.array([left_knee, right_knee, left_hip, right_hip])

    def _compute_normalized_distances(self, kpts: np.ndarray, norm: float) -> np.ndarray:
        """Compute normalized bone lengths for 6 bone pairs."""
        return np.array([
            distance_3d(kpts[i], kpts[j]) / norm for i, j in _BONE_PAIRS
        ])

    def _compute_normalized_y_diffs(self, kpts: np.ndarray, norm: float) -> np.ndarray:
        """
        Compute normalized vertical displacements relative to hip midpoint.

        Returns 4 values: hip_mid_y, knee_mid_y, ankle_mid_y, shoulder_mid_y
        all relative to hip_mid_y and divided by normalization factor.
        """
        hip_mid = midpoint(kpts[CK.LEFT_HIP], kpts[CK.RIGHT_HIP])
        knee_mid = midpoint(kpts[CK.LEFT_KNEE], kpts[CK.RIGHT_KNEE])
        ankle_mid = midpoint(kpts[CK.LEFT_ANKLE], kpts[CK.RIGHT_ANKLE])
        shoulder_mid = midpoint(kpts[CK.LEFT_SHOULDER], kpts[CK.RIGHT_SHOULDER])

        ref_y = hip_mid[1]
        return self._vertical_sign * np.array([
            (hip_mid[1] - ref_y) / norm,       # always 0, but keeps vector fixed-length
            (knee_mid[1] - ref_y) / norm,
            (ankle_mid[1] - ref_y) / norm,
            (shoulder_mid[1] - ref_y) / norm,
        ])
