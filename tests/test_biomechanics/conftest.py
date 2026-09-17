"""
Shared Pytest Fixtures for Biomechanics Tests

Provides common test data and utilities used across test modules.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from biomechanics.utils.types import (
    Keypoint2D,
    Skeleton2D,
    Point3D,
    Skeleton3D,
    JointAngles,
    FaultEvent,
    FaultSeverity,
    RepData,
    PipelineFrame,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture
def sample_keypoints_2d() -> np.ndarray:
    """
    Load sample 2D keypoints (COCO 17 format) for a half-squat pose.

    Returns:
        numpy array of shape (17, 3) where columns are [x, y, confidence]
    """
    with open(FIXTURES_DIR / "sample_keypoints.json") as f:
        data = json.load(f)
    return np.array(data["keypoints"])


@pytest.fixture
def sample_skeleton_2d(sample_keypoints_2d) -> Skeleton2D:
    """Sample 2D skeleton from fixture keypoints."""
    return Skeleton2D.from_numpy(sample_keypoints_2d)


@pytest.fixture
def sample_3d_points() -> np.ndarray:
    """
    Load sample 3D points (same pose in meters).

    Returns:
        numpy array of shape (17, 3) where columns are [x, y, z]
    """
    with open(FIXTURES_DIR / "sample_3d_points.json") as f:
        data = json.load(f)
    return np.array(data["points"])


@pytest.fixture
def sample_skeleton_3d(sample_3d_points) -> Skeleton3D:
    """Sample 3D skeleton from fixture points."""
    return Skeleton3D.from_numpy(sample_3d_points)


@pytest.fixture
def sample_joint_angles() -> JointAngles:
    """
    Load expected joint angles for the sample pose.

    These are the "ground truth" angles for the half-squat pose.
    """
    with open(FIXTURES_DIR / "sample_angles.json") as f:
        data = json.load(f)
    return JointAngles(**data["angles"])


@pytest.fixture
def expected_angles_dict() -> dict:
    """Expected angles as a dict for comparison."""
    with open(FIXTURES_DIR / "sample_angles.json") as f:
        data = json.load(f)
    return data["angles"]


@pytest.fixture
def sample_fault_event() -> FaultEvent:
    """Sample fault event for testing."""
    return FaultEvent(
        fault_type="knee_valgus",
        severity=FaultSeverity.MODERATE,
        severity_score=2.0,
        message="Knees tracking inward",
        rep_number=1,
    )


@pytest.fixture
def sample_rep_data(sample_fault_event) -> RepData:
    """Sample rep data for testing."""
    return RepData(
        rep_number=1,
        start_time=0.0,
        end_time=2.5,
        start_frame=0,
        end_frame=75,
        max_depth_angle=85.0,
        min_depth_angle=15.0,
        descent_time=1.2,
        ascent_time=1.0,
        faults=[sample_fault_event],
    )


@pytest.fixture
def sample_pipeline_frame(
    sample_skeleton_2d, sample_skeleton_3d, sample_joint_angles
) -> PipelineFrame:
    """Sample complete pipeline frame for testing."""
    return PipelineFrame(
        frame_index=100,
        timestamp=3.33,
        skeleton_2d=sample_skeleton_2d,
        skeleton_3d=sample_skeleton_3d,
        joint_angles=sample_joint_angles,
        faults=[],
        latency_ms={"pose": 8.0, "ik": 4.0, "faults": 1.0},
    )


# =============================================================================
# HELPER FIXTURES
# =============================================================================

@pytest.fixture
def angle_tolerance() -> float:
    """Tolerance for angle comparisons (degrees)."""
    return 5.0


@pytest.fixture
def position_tolerance() -> float:
    """Tolerance for position comparisons (meters)."""
    return 0.01


# =============================================================================
# MOCK FIXTURES
# =============================================================================

@pytest.fixture
def mock_ipc_client():
    """Mock IPC client for testing coaching integration."""
    class MockIPCClient:
        def __init__(self):
            self.messages = []
            self.connected = True

        def send_message(self, message):
            self.messages.append(message)

        def connect(self, timeout=10):
            return True

        def disconnect(self):
            self.connected = False

    return MockIPCClient()


# =============================================================================
# SYNTHETIC WORLD-FRAME SQUAT (shared by the pre-IK chain and pipeline tests)
# =============================================================================
# Triangulated frame: Y-down metres, X = subject's left, forward = -Z, floor
# at y = 0. 21 keypoints (COCO-17 + toes + heels).

SYNTHETIC_FPS = 30.0
SYNTHETIC_FEMUR_M = 0.45
SYNTHETIC_TIBIA_M = 0.43
SYNTHETIC_TORSO_M = 0.52
SYNTHETIC_HIP_HALF_WIDTH_M = 0.12
SYNTHETIC_SHOULDER_HALF_WIDTH_M = 0.19
SYNTHETIC_TOE_FORWARD_M = 0.18
SYNTHETIC_HEEL_BACK_M = 0.06
SYNTHETIC_NUM_KEYPOINTS = 21
SYNTHETIC_BOTTOM_SHANK_DEG = 35.0
SYNTHETIC_BOTTOM_THIGH_DEG = 85.0
SYNTHETIC_BOTTOM_TRUNK_DEG = 40.0


def world_squat_points(
    depth_ratio: float, valgus_m: float = 0.0, lean_extra_deg: float = 0.0,
) -> np.ndarray:
    """(21, 3) world-frame squat pose. depth_ratio 0 = standing, 1 = bottom;
    valgus_m shifts both knees medially at the bottom."""
    import math

    from biomechanics.utils.types import CocoKeypoints as CK

    shank = math.radians(SYNTHETIC_BOTTOM_SHANK_DEG * depth_ratio)
    thigh = math.radians(SYNTHETIC_BOTTOM_THIGH_DEG * depth_ratio)
    trunk = math.radians(SYNTHETIC_BOTTOM_TRUNK_DEG * depth_ratio + lean_extra_deg)
    knee = np.array([0.0, -SYNTHETIC_TIBIA_M * math.cos(shank), -SYNTHETIC_TIBIA_M * math.sin(shank)])
    hip = knee + np.array([0.0, -SYNTHETIC_FEMUR_M * math.cos(thigh), SYNTHETIC_FEMUR_M * math.sin(thigh)])
    shoulder = hip + np.array([0.0, -SYNTHETIC_TORSO_M * math.cos(trunk), -SYNTHETIC_TORSO_M * math.sin(trunk)])

    points = np.zeros((SYNTHETIC_NUM_KEYPOINTS, 3))
    sides = (
        (1.0, CK.LEFT_HIP, CK.LEFT_KNEE, CK.LEFT_ANKLE, CK.LEFT_SHOULDER, CK.LEFT_ELBOW,
         CK.LEFT_WRIST, CK.LEFT_FOOT_INDEX, CK.LEFT_HEEL),
        (-1.0, CK.RIGHT_HIP, CK.RIGHT_KNEE, CK.RIGHT_ANKLE, CK.RIGHT_SHOULDER, CK.RIGHT_ELBOW,
         CK.RIGHT_WRIST, CK.RIGHT_FOOT_INDEX, CK.RIGHT_HEEL),
    )
    for sign, hip_i, knee_i, ankle_i, shoulder_i, elbow_i, wrist_i, toe_i, heel_i in sides:
        x = sign * SYNTHETIC_HIP_HALF_WIDTH_M
        points[ankle_i] = [x, 0.0, 0.0]
        points[knee_i] = [x - sign * valgus_m * depth_ratio, knee[1], knee[2]]
        points[hip_i] = [x, hip[1], hip[2]]
        points[shoulder_i] = [sign * SYNTHETIC_SHOULDER_HALF_WIDTH_M, shoulder[1], shoulder[2]]
        points[elbow_i] = [sign * 0.24, shoulder[1] + 0.25, shoulder[2] + 0.05]
        points[wrist_i] = [sign * 0.24, shoulder[1] + 0.48, shoulder[2] + 0.10]
        points[toe_i] = [x + sign * 0.03, 0.0, -SYNTHETIC_TOE_FORWARD_M]
        points[heel_i] = [x, 0.0, SYNTHETIC_HEEL_BACK_M]
    head = (points[CK.LEFT_SHOULDER] + points[CK.RIGHT_SHOULDER]) / 2.0 + np.array([0.0, -0.22, -0.03])
    points[CK.NOSE] = head
    points[CK.LEFT_EYE] = head + [0.03, -0.02, 0.0]
    points[CK.RIGHT_EYE] = head + [-0.03, -0.02, 0.0]
    points[CK.LEFT_EAR] = head + [0.07, 0.0, 0.05]
    points[CK.RIGHT_EAR] = head + [-0.07, 0.0, 0.05]
    return points


def squat_depth_profile(
    stand_s: float = 1.0, down_s: float = 1.0, hold_s: float = 0.4, up_s: float = 1.0,
) -> list[float]:
    """Per-frame depth ratios for one rep: stand, cosine descent, hold, cosine ascent."""
    import math

    frames: list[float] = [0.0] * int(stand_s * SYNTHETIC_FPS)
    down_frames = int(down_s * SYNTHETIC_FPS)
    frames += [0.5 - 0.5 * math.cos(math.pi * i / down_frames) for i in range(down_frames)]
    frames += [1.0] * int(hold_s * SYNTHETIC_FPS)
    up_frames = int(up_s * SYNTHETIC_FPS)
    frames += [0.5 + 0.5 * math.cos(math.pi * i / up_frames) for i in range(up_frames)]
    return frames
