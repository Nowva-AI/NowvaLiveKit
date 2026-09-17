"""
Abstract base class for pose estimation backends.

All pose estimators (MediaPipe, RTMPose, etc.) must inherit from this class
and implement the estimate() and estimate_3d() methods.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np

from biomechanics.utils.types import Skeleton2D, Skeleton3D


# COCO 17 keypoint names in order
COCO_KEYPOINT_NAMES = [
    "nose",           # 0
    "left_eye",       # 1
    "right_eye",      # 2
    "left_ear",       # 3
    "right_ear",      # 4
    "left_shoulder",  # 5
    "right_shoulder", # 6
    "left_elbow",     # 7
    "right_elbow",    # 8
    "left_wrist",     # 9
    "right_wrist",    # 10
    "left_hip",       # 11
    "right_hip",      # 12
    "left_knee",      # 13
    "right_knee",     # 14
    "left_ankle",      # 15
    "right_ankle",     # 16
    "left_foot_index", # 17
    "right_foot_index",# 18
    "left_heel",       # 19
    "right_heel",      # 20
]

# COCO skeleton connections (pairs of keypoint indices)
COCO_SKELETON_CONNECTIONS = [
    (0, 1),    # nose - left_eye
    (0, 2),    # nose - right_eye
    (1, 3),    # left_eye - left_ear
    (2, 4),    # right_eye - right_ear
    (0, 5),    # nose - left_shoulder
    (0, 6),    # nose - right_shoulder
    (5, 6),    # left_shoulder - right_shoulder
    (5, 7),    # left_shoulder - left_elbow
    (7, 9),    # left_elbow - left_wrist
    (6, 8),    # right_shoulder - right_elbow
    (8, 10),   # right_elbow - right_wrist
    (5, 11),   # left_shoulder - left_hip
    (6, 12),   # right_shoulder - right_hip
    (11, 12),  # left_hip - right_hip
    (11, 13),  # left_hip - left_knee
    (13, 15),  # left_knee - left_ankle
    (12, 14),  # right_hip - right_knee
    (14, 16),  # right_knee - right_ankle
    (15, 17),  # left_ankle - left_foot_index
    (16, 18),  # right_ankle - right_foot_index
    (15, 19),  # left_ankle - left_heel
    (19, 17),  # left_heel - left_foot_index
    (16, 20),  # right_ankle - right_heel
    (20, 18),  # right_heel - right_foot_index
]


class PoseEstimator(ABC):
    """
    Abstract base class for pose estimation.

    All pose estimation backends must implement:
    - estimate(): Returns 2D pose in pixel coordinates
    - estimate_3d(): Returns 3D pose (if supported by backend)

    Keypoints are always returned in COCO 17 format.
    """

    # Class-level keypoint names
    KEYPOINT_NAMES = COCO_KEYPOINT_NAMES
    NUM_KEYPOINTS = 21

    def __init__(self, confidence_threshold: float = 0.3):
        """
        Initialize pose estimator.

        Args:
            confidence_threshold: Minimum confidence to consider keypoint valid
        """
        self.confidence_threshold = confidence_threshold
        self._initialized = False

    @abstractmethod
    def estimate(self, frame: np.ndarray, camera_id: int = 0) -> Optional[Skeleton2D]:
        """
        Estimate 2D pose from a single frame.

        Args:
            frame: BGR image as numpy array (H, W, 3)
            camera_id: Camera identifier (for multi-view tracking)

        Returns:
            Skeleton2D with 17 keypoints in pixel coordinates, or None if no person detected
        """
        pass

    @abstractmethod
    def estimate_3d(self, frame: np.ndarray) -> Optional[Skeleton3D]:
        """
        Estimate 3D pose from a single frame.

        For single-camera backends like MediaPipe, this uses the backend's
        built-in 3D estimation. For stereo setups, use the triangulation layer.

        Args:
            frame: BGR image as numpy array (H, W, 3)

        Returns:
            Skeleton3D with 17 keypoints in world coordinates (meters), or None if no person detected
        """
        pass

    def estimate_both(self, frame: np.ndarray) -> Tuple[Optional[Skeleton2D], Optional[Skeleton3D]]:
        """
        Estimate both 2D and 3D pose from a single frame.

        Default implementation calls estimate() and estimate_3d() separately.
        Subclasses may override for efficiency (e.g., MediaPipe single-pass).

        Args:
            frame: BGR image as numpy array (H, W, 3)

        Returns:
            Tuple of (Skeleton2D, Skeleton3D), either may be None
        """
        skeleton_2d = self.estimate(frame)
        skeleton_3d = self.estimate_3d(frame)
        return skeleton_2d, skeleton_3d

    def initialize(self) -> bool:
        """
        Initialize the pose estimator (load models, etc.).

        Returns:
            True if initialization successful
        """
        self._initialized = True
        return True

    def release(self) -> None:
        """Release any resources held by the estimator."""
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if estimator is initialized."""
        return self._initialized

    @staticmethod
    def keypoint_index(name: str) -> int:
        """
        Get index for a keypoint by name.

        Args:
            name: Keypoint name (e.g., "left_knee")

        Returns:
            Index in COCO 17 format

        Raises:
            ValueError: If keypoint name not found
        """
        try:
            return COCO_KEYPOINT_NAMES.index(name)
        except ValueError:
            raise ValueError(f"Unknown keypoint name: {name}")

    @staticmethod
    def keypoint_name(index: int) -> str:
        """
        Get name for a keypoint by index.

        Args:
            index: Keypoint index (0-16)

        Returns:
            Keypoint name

        Raises:
            IndexError: If index out of range
        """
        if 0 <= index < len(COCO_KEYPOINT_NAMES):
            return COCO_KEYPOINT_NAMES[index]
        raise IndexError(f"Keypoint index out of range: {index}")
