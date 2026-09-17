"""Synthetic data generators for reproducible benchmarks."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

FIXTURES_DIR = Path(__file__).parent.parent.parent / "tests" / "test_biomechanics" / "fixtures"
DATA_DIR = Path(__file__).parent.parent.parent / "data"

# Standing pose (arms down, legs straight) — 21 keypoints (COCO 17 + toes + heels)
NUM_KEYPOINTS = 21
_STANDING_3D = np.array([
    [0.0, 1.75, 0.0],     # 0  nose
    [0.03, 1.78, -0.02],   # 1  left_eye
    [-0.03, 1.78, -0.02],  # 2  right_eye
    [0.06, 1.76, 0.0],     # 3  left_ear
    [-0.06, 1.76, 0.0],    # 4  right_ear
    [0.18, 1.50, -0.02],   # 5  left_shoulder
    [-0.18, 1.50, -0.02],  # 6  right_shoulder
    [0.22, 1.25, 0.0],     # 7  left_elbow
    [-0.22, 1.25, 0.0],    # 8  right_elbow
    [0.22, 1.00, 0.0],     # 9  left_wrist
    [-0.22, 1.00, 0.0],    # 10 right_wrist
    [0.10, 0.95, 0.0],     # 11 left_hip
    [-0.10, 0.95, 0.0],    # 12 right_hip
    [0.10, 0.50, 0.0],     # 13 left_knee
    [-0.10, 0.50, 0.0],    # 14 right_knee
    [0.10, 0.05, 0.0],     # 15 left_ankle
    [-0.10, 0.05, 0.0],    # 16 right_ankle
    [0.12, 0.0, 0.10],     # 17 left_foot_index
    [-0.12, 0.0, 0.10],    # 18 right_foot_index
    [0.10, 0.0, -0.05],    # 19 left_heel
    [-0.10, 0.0, -0.05],   # 20 right_heel
])


def _load_json(name: str) -> dict:
    with open(FIXTURES_DIR / name) as f:
        return json.load(f)


def _ensure_21_keypoints(pts: np.ndarray) -> np.ndarray:
    """Extend a 17-keypoint array to 21 by adding toe and heel keypoints."""
    if len(pts) >= NUM_KEYPOINTS:
        return pts
    left_ankle = pts[15]
    right_ankle = pts[16]
    left_foot = left_ankle + np.array([0.02, -0.05, 0.10])
    right_foot = right_ankle + np.array([-0.02, -0.05, 0.10])
    left_heel = left_ankle + np.array([0.0, -0.05, -0.05])
    right_heel = right_ankle + np.array([0.0, -0.05, -0.05])
    return np.vstack([pts[:17], [left_foot], [right_foot], [left_heel], [right_heel]])


def load_fixture_skeleton_3d():
    """Load the half-squat 3D skeleton fixture (21 keypoints)."""
    from biomechanics.utils.types import Skeleton3D
    data = _load_json("sample_3d_points.json")
    pts = _ensure_21_keypoints(np.array(data["points"]))
    return Skeleton3D.from_numpy(pts)


def load_fixture_skeleton_2d():
    """Load the half-squat 2D skeleton fixture."""
    from biomechanics.utils.types import Skeleton2D
    data = _load_json("sample_keypoints.json")
    return Skeleton2D.from_numpy(np.array(data["keypoints"]))


def load_fixture_angles():
    """Load the half-squat joint angles fixture."""
    from biomechanics.utils.types import JointAngles
    data = _load_json("sample_angles.json")
    return JointAngles(**data["angles"])


def generate_squat_sequence(n_frames: int = 90) -> list:
    """
    Generate a synthetic squat rep sequence: standing -> bottom -> standing.
    Returns list of Skeleton3D objects (one full rep over n_frames).
    """
    from biomechanics.utils.types import Skeleton3D

    data = _load_json("sample_3d_points.json")
    squat_bottom = _ensure_21_keypoints(np.array(data["points"]))

    frames = []
    for i in range(n_frames):
        phase = i / n_frames
        if phase < 0.33:
            t = phase / 0.33
        elif phase < 0.66:
            t = 1.0 - (phase - 0.33) / 0.33
        else:
            t = 0.0

        pts = _STANDING_3D * (1 - t) + squat_bottom * t
        frames.append(Skeleton3D.from_numpy(
            pts,
            timestamp=i / 30.0,
            frame_index=i,
        ))
    return frames


def generate_synthetic_image(width: int = 1280, height: int = 720) -> np.ndarray:
    """Generate a simple synthetic BGR image for pose estimation benchmarks."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :, 0] = 40
    img[:, :, 1] = 40
    img[:, :, 2] = 60
    cv_x, cv_y = width // 2, height // 2
    img[cv_y - 200:cv_y + 200, cv_x - 50:cv_x + 50] = [180, 160, 140]
    return img


def load_video_frames(n_frames: int = 100) -> list[np.ndarray]:
    """Load frames from data/squats.mov, falling back to synthetic images."""
    video_path = DATA_DIR / "squats.mov"
    if video_path.exists():
        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            frames = []
            for _ in range(n_frames):
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        break
                if frame.shape[:2] != (720, 1280):
                    frame = cv2.resize(frame, (1280, 720))
                frames.append(frame)
            cap.release()
            if frames:
                return frames
        except ImportError:
            pass
    return [generate_synthetic_image() for _ in range(n_frames)]


def generate_ipc_message() -> dict:
    """Create a sample IPC message matching the coaching protocol."""
    return {
        "type": "skeleton_update",
        "timestamp": time.time(),
        "frame_index": 42,
        "faults": [{"fault_type": "knee_valgus", "severity": "MODERATE", "message": "Knees tracking inward"}],
        "rep_data": {"rep_number": 3, "max_depth_angle": 85.0, "descent_time": 1.2},
        "angles": {"knee_flexion_l": 80.0, "knee_flexion_r": 78.0, "trunk_flexion": 30.0},
    }


def generate_agent_state_dict() -> dict:
    """Create a realistic agent state dict for save/load benchmarks."""
    return {
        "mode": "workout",
        "user": {
            "id": "usr_bench_001",
            "name": "Benchmark User",
            "email": "bench@test.com",
        },
        "workout": {
            "session_id": "ws_bench_001",
            "exercise_name": "barbell_back_squat",
            "current_set": 3,
            "total_sets": 5,
            "target_reps": 8,
            "completed_reps": 6,
            "rest_timer": 120,
        },
        "program_creation": {
            "fitness_level": "intermediate",
            "goals": ["strength", "hypertrophy"],
            "days_per_week": 4,
        },
    }


def generate_compaction_events(n_events: int = 20) -> list[dict]:
    """Create sample compaction event buffer entries."""
    events = []
    roles = ["user", "assistant"]
    for i in range(n_events):
        events.append({
            "role": roles[i % 2],
            "content": f"Sample conversation turn {i} with enough text to be realistic for compaction benchmarking.",
            "timestamp": time.time() + i * 2.0,
        })
    return events
