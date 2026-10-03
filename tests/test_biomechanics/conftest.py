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


# =============================================================================
# SYNTHETIC PIPELINE DRIVERS (fake clock, triangulated provider, mono pose)
# =============================================================================

SYNTHETIC_CLOCK_START_S = 1.7e9
SYNTHETIC_CONFIDENCE = 0.9
SYNTHETIC_PIXELS_PER_M = 600.0
SYNTHETIC_IMAGE_CENTRE_PX = (640.0, 300.0)
SYNTHETIC_CAPTURE_FRAME = np.zeros((72, 128, 3), dtype=np.uint8)
# A perfectly collinear synthetic leg reads exactly 0 deg; real lifters keep a small bend.
SYNTHETIC_MIN_DEPTH_RATIO = 0.02


class SyntheticClock:
    """One clock for capture timestamps, time.time() and perf_counter()."""

    def __init__(self, start_s: float = SYNTHETIC_CLOCK_START_S) -> None:
        self.now_s = start_s

    def advance(self, seconds: float) -> None:
        self.now_s += seconds

    def time(self) -> float:
        return self.now_s

    def perf_counter(self) -> float:
        return self.now_s


class SyntheticProvider:
    """MultiCameraPoseProvider stand-in: world-frame skeletons on the fake clock (None = dropout)."""

    def __init__(self, clock: SyntheticClock) -> None:
        self._clock = clock
        self._queue: list[tuple[np.ndarray | None, np.ndarray | None]] = []
        self.sequence = 0
        self.swap_count = 0
        self.frame = np.zeros((72, 128, 3), dtype=np.uint8)
        self.last_capture_timestamp = float("nan")

    def push(self, points: np.ndarray | None, confidences: np.ndarray | None = None) -> None:
        self._queue.append((points, confidences))

    def get_pose(self):
        points, confidences = self._queue.pop(0)
        self.sequence += 1
        self.last_capture_timestamp = self._clock.time()
        if points is None:
            return self.frame, None, None
        if confidences is None:
            confidences = np.full(len(points), SYNTHETIC_CONFIDENCE)
        skeleton = Skeleton3D.from_numpy(
            points, confidences=confidences, timestamp=self._clock.time(), frame_index=self.sequence,
        )
        return self.frame, None, skeleton

    def reset_temporal_state(self) -> None:
        pass

    def lost_cameras(self) -> list[str]:
        return []

    def release(self) -> None:
        pass


class SyntheticMonoPose:
    """MediaPipe stand-in: hip-centred Y-down 3D plus a pinhole 2D view of it."""

    def __init__(self) -> None:
        self._queue: list[tuple[np.ndarray | None, np.ndarray | None]] = []

    def push(self, points: np.ndarray | None, confidences: np.ndarray | None = None) -> None:
        self._queue.append((points, confidences))

    def estimate_both(self, frame: np.ndarray):
        from biomechanics.utils.types import CocoKeypoints as CK

        points, confidences = self._queue.pop(0)
        if points is None:
            return None, None
        if confidences is None:
            confidences = np.full(len(points), SYNTHETIC_CONFIDENCE)
        centred = points - (points[CK.LEFT_HIP] + points[CK.RIGHT_HIP]) / 2.0
        skeleton_3d = Skeleton3D.from_numpy(centred, confidences=confidences)
        keypoints_2d = [
            Keypoint2D(
                x=SYNTHETIC_IMAGE_CENTRE_PX[0] + SYNTHETIC_PIXELS_PER_M * point[0],
                y=SYNTHETIC_IMAGE_CENTRE_PX[1] + SYNTHETIC_PIXELS_PER_M * point[1],
                confidence=float(confidence),
            )
            for point, confidence in zip(centred, confidences)
        ]
        return Skeleton2D(keypoints=keypoints_2d), skeleton_3d

    def reset_tracking(self) -> None:
        pass

    def release(self) -> None:
        pass


class SyntheticSession:
    """A real BiomechanicsPipeline fed synthetic world-frame squat poses on a fake clock."""

    def __init__(self, monkeypatch, multi_camera: bool = True, config=None, exercise_name: str = "Barbell Back Squat"):
        import types as _types

        from biomechanics import pipeline as pipeline_module
        from biomechanics.config import BiomechanicsConfig

        monkeypatch.setenv("NOWVA_MULTI_CAMERA", "true" if multi_camera else "false")
        self.multi_camera = multi_camera
        self.clock = SyntheticClock()
        monkeypatch.setattr(
            pipeline_module, "time",
            _types.SimpleNamespace(time=self.clock.time, perf_counter=self.clock.perf_counter),
        )
        self.pipeline = pipeline_module.BiomechanicsPipeline(
            config or BiomechanicsConfig(), exercise_name=exercise_name, defer_capture=True,
        )
        self.provider = SyntheticProvider(self.clock)
        self.pose = SyntheticMonoPose()
        if multi_camera:
            self.pipeline._multi_camera_provider = self.provider
        else:
            self.pipeline._pose_estimator = self.pose

    def step(self, points: np.ndarray | None, confidences: np.ndarray | None = None, fps: float = SYNTHETIC_FPS):
        if self.multi_camera:
            self.provider.push(points, confidences)
        else:
            self.pose.push(points, confidences)
            self.pipeline._publish_frame(SYNTHETIC_CAPTURE_FRAME, self.clock.time())
        result = self.pipeline.process_frame()
        self.clock.advance(1.0 / fps)
        return result

    def run(self, depths: list[float], fps: float = SYNTHETIC_FPS, dropped: frozenset[int] = frozenset(), **pose):
        return [
            self.step(
                None if index in dropped else world_squat_points(max(depth, SYNTHETIC_MIN_DEPTH_RATIO), **pose),
                fps=fps,
            )
            for index, depth in enumerate(depths)
        ]
