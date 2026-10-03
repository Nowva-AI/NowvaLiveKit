"""
Tests for bottom-of-rep frame buffer (Part 1: Pipeline Enrichment).

Verifies that the pipeline's bottom-frame buffer captures the same frame
that the visualizer's find_bottom_frame() would select, and that the
underlying data is equivalent across both paths.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from biomechanics.coaching.ipc_bridge import IPCBridge
from biomechanics.coaching.session_tracker import SessionTracker
from biomechanics.config import CoachingConfig, IPCConfig
from biomechanics.diagnosis.bridge import find_bottom_frame
from biomechanics.utils.types import (
    JointAngles,
    Point3D,
    RepData,
    Skeleton3D,
)


# =============================================================================
# HELPERS
# =============================================================================

_STANDING_KNEE_FLEX = 15.0
_HALF_SQUAT_KNEE_FLEX = 55.0
_DEEP_SQUAT_KNEE_FLEX = 110.0

# 19 keypoints: COCO 17 + 2 foot_index
_NUM_KEYPOINTS = 19


def _make_skeleton_3d(knee_flex_deg: float) -> Skeleton3D:
    """Build a synthetic 19-keypoint Skeleton3D in the production frame (Y down).

    Varies hip Y position with knee_flex_deg so deeper squats have
    lower hips (larger y), making the skeleton physically plausible. Ankle
    and foot positions stay fixed.
    """
    hip_y = 0.0 + (knee_flex_deg / 180.0) * 0.4
    knee_y = 0.45
    ankle_y = 0.85
    shoulder_y = hip_y - 0.45

    coords = np.zeros((_NUM_KEYPOINTS, 3))
    # Shoulders (5, 6)
    coords[5] = [-0.20, shoulder_y, 0.0]
    coords[6] = [0.20, shoulder_y, 0.0]
    # Hips (11, 12)
    coords[11] = [-0.15, hip_y, 0.0]
    coords[12] = [0.15, hip_y, 0.0]
    # Knees (13, 14)
    coords[13] = [-0.15, knee_y, 0.10]
    coords[14] = [0.15, knee_y, 0.10]
    # Ankles (15, 16)
    coords[15] = [-0.15, ankle_y, 0.0]
    coords[16] = [0.15, ankle_y, 0.0]
    # Foot index (17, 18)
    coords[17] = [-0.15, ankle_y, -0.10]
    coords[18] = [0.15, ankle_y, -0.10]

    keypoints = [
        Point3D(x=float(c[0]), y=float(c[1]), z=float(c[2]), confidence=0.9)
        for c in coords
    ]
    return Skeleton3D(keypoints=keypoints)


def _make_angles(knee_flex_deg: float) -> JointAngles:
    """Build JointAngles with specified knee flexion (symmetric L/R)."""
    return JointAngles(
        knee_flexion_l=knee_flex_deg,
        knee_flexion_r=knee_flex_deg,
        hip_flexion_l=knee_flex_deg * 0.8,
        hip_flexion_r=knee_flex_deg * 0.8,
        trunk_flexion=knee_flex_deg * 0.3,
        ankle_dorsiflexion_l=knee_flex_deg * 0.15,
        ankle_dorsiflexion_r=knee_flex_deg * 0.15,
        knee_valgus_l=3.0,
        knee_valgus_r=2.5,
    )


def _extract_frame_data(skeleton_3d: Skeleton3D, angles: JointAngles, frame_idx: int) -> dict:
    """Replicate the visualizer's extract_frame_data() coordinate transform."""
    kpts_mp = skeleton_3d.to_numpy()[:19]
    kpts_vis = np.zeros_like(kpts_mp)
    kpts_vis[:, 0] = kpts_mp[:, 2]   # vis_x = mp_z
    kpts_vis[:, 1] = -kpts_mp[:, 1]  # vis_y = -mp_y
    kpts_vis[:, 2] = -kpts_mp[:, 0]  # vis_z = -mp_x
    kpts_vis[17][1] = kpts_vis[15][1]
    kpts_vis[18][1] = kpts_vis[16][1]
    return {
        "kpts": kpts_vis.tolist(),
        "angles": {
            "knee_flex": angles.avg_knee_flexion,
            "knee_flex_l": angles.knee_flexion_l,
            "knee_flex_r": angles.knee_flexion_r,
            "trunk_flexion": angles.trunk_flexion,
            "knee_valgus_l": angles.knee_valgus_l,
            "knee_valgus_r": angles.knee_valgus_r,
            "dorsi_l": angles.ankle_dorsiflexion_l,
            "dorsi_r": angles.ankle_dorsiflexion_r,
            "hip_flex_l": angles.hip_flexion_l,
            "hip_flex_r": angles.hip_flexion_r,
            "shoulder_flex_l": angles.shoulder_flexion_l,
            "shoulder_flex_r": angles.shoulder_flexion_r,
            "elbow_flex_l": angles.elbow_flexion_l,
            "elbow_flex_r": angles.elbow_flexion_r,
        },
        "frame": frame_idx,
    }


def _apply_visualizer_transform(raw_kpts: list[list[float]]) -> list[list[float]]:
    """Apply the visualizer's coordinate swap to raw pipeline keypoints."""
    kpts_mp = np.array(raw_kpts[:19])
    kpts_vis = np.zeros_like(kpts_mp)
    kpts_vis[:, 0] = kpts_mp[:, 2]
    kpts_vis[:, 1] = -kpts_mp[:, 1]
    kpts_vis[:, 2] = -kpts_mp[:, 0]
    kpts_vis[17][1] = kpts_vis[15][1]
    kpts_vis[18][1] = kpts_vis[16][1]
    return kpts_vis.tolist()


# Angle key mapping: pipeline as_dict() name → visualizer name
_ANGLE_KEY_MAP = {
    "knee_flexion_l": "knee_flex_l",
    "knee_flexion_r": "knee_flex_r",
    "trunk_flexion": "trunk_flexion",
    "knee_valgus_l": "knee_valgus_l",
    "knee_valgus_r": "knee_valgus_r",
    "ankle_dorsiflexion_l": "dorsi_l",
    "ankle_dorsiflexion_r": "dorsi_r",
    "hip_flexion_l": "hip_flex_l",
    "hip_flexion_r": "hip_flex_r",
    "shoulder_flexion_l": "shoulder_flex_l",
    "shoulder_flexion_r": "shoulder_flex_r",
    "elbow_flexion_l": "elbow_flex_l",
    "elbow_flexion_r": "elbow_flex_r",
}


def _depth_ratio(skeleton: Skeleton3D) -> float:
    """Hip height above the knee in thigh lengths (Y down) — the pipeline's bottom criterion."""
    kpts = skeleton.to_numpy()
    hip_y = (kpts[11][1] + kpts[12][1]) / 2.0
    knee_y = (kpts[13][1] + kpts[14][1]) / 2.0
    thigh = (np.linalg.norm(kpts[11] - kpts[13]) + np.linalg.norm(kpts[12] - kpts[14])) / 2.0
    return (knee_y - hip_y) / thigh


def _simulate_rep_sequence() -> list[float]:
    """Knee flexion values for a realistic squat rep: descend → bottom → ascend."""
    return [15.0, 30.0, 55.0, 80.0, 100.0, 110.0, 105.0, 85.0, 50.0, 20.0]


# =============================================================================
# TESTS
# =============================================================================


class TestBottomFrameBuffer:
    """Verify the pipeline buffer captures the same bottom frame as the visualizer."""

    def test_buffer_selects_lowest_hip_frame(self):
        """Buffer picks the frame where the hip sits lowest relative to the knees."""
        flex_sequence = _simulate_rep_sequence()
        expected_bottom_idx = int(np.argmax(flex_sequence))

        bottom_min = float("inf")
        bottom_kpts = None
        bottom_angles = None
        captured_idx = -1

        for i, flex in enumerate(flex_sequence):
            angles = _make_angles(flex)
            skeleton = _make_skeleton_3d(flex)
            depth = _depth_ratio(skeleton)
            if depth < bottom_min:
                bottom_min = depth
                bottom_kpts = skeleton.to_numpy().tolist()
                bottom_angles = angles.as_dict()
                captured_idx = i

        assert captured_idx == expected_bottom_idx
        assert bottom_kpts is not None
        assert bottom_angles is not None

    def test_buffer_matches_find_bottom_frame_selection(self):
        """Buffer and find_bottom_frame() select the same frame."""
        flex_sequence = _simulate_rep_sequence()

        # Path A: pipeline buffer logic
        bottom_min = float("inf")
        buffer_kpts = None
        buffer_angles_dict = None
        buffer_idx = -1

        # Path B: visualizer frame list for find_bottom_frame()
        vis_frames = []

        for i, flex in enumerate(flex_sequence):
            angles = _make_angles(flex)
            skeleton = _make_skeleton_3d(flex)

            # Buffer path
            depth = _depth_ratio(skeleton)
            if depth < bottom_min:
                bottom_min = depth
                buffer_kpts = skeleton.to_numpy().tolist()
                buffer_angles_dict = angles.as_dict()
                buffer_idx = i

            # Visualizer path
            vis_frames.append(_extract_frame_data(skeleton, angles, i))

        vis_bottom = find_bottom_frame(vis_frames)

        assert buffer_idx == vis_bottom["frame"]
        assert buffer_angles_dict["knee_flexion_l"] == vis_bottom["angles"]["knee_flex_l"]

    def test_buffer_keypoints_match_visualizer_after_transform(self):
        """Raw pipeline kpts, when transformed, equal the visualizer's kpts."""
        flex_sequence = _simulate_rep_sequence()

        bottom_min = float("inf")
        buffer_kpts = None
        vis_frames = []

        for i, flex in enumerate(flex_sequence):
            angles = _make_angles(flex)
            skeleton = _make_skeleton_3d(flex)

            depth = _depth_ratio(skeleton)
            if depth < bottom_min:
                bottom_min = depth
                buffer_kpts = skeleton.to_numpy().tolist()

            vis_frames.append(_extract_frame_data(skeleton, angles, i))

        vis_bottom = find_bottom_frame(vis_frames)
        transformed_buffer = _apply_visualizer_transform(buffer_kpts)

        np.testing.assert_allclose(
            transformed_buffer,
            vis_bottom["kpts"],
            atol=1e-10,
        )

    def test_buffer_angles_match_visualizer_values(self):
        """Angle values in the buffer match the visualizer's angles for mapped keys."""
        flex_sequence = _simulate_rep_sequence()

        bottom_min = float("inf")
        buffer_angles_dict = None
        vis_frames = []

        for i, flex in enumerate(flex_sequence):
            angles = _make_angles(flex)
            skeleton = _make_skeleton_3d(flex)

            depth = _depth_ratio(skeleton)
            if depth < bottom_min:
                bottom_min = depth
                buffer_angles_dict = angles.as_dict()

            vis_frames.append(_extract_frame_data(skeleton, angles, i))

        vis_bottom = find_bottom_frame(vis_frames)
        vis_angles = vis_bottom["angles"]

        for pipeline_key, vis_key in _ANGLE_KEY_MAP.items():
            assert buffer_angles_dict[pipeline_key] == pytest.approx(
                vis_angles[vis_key], abs=1e-10,
            ), f"Mismatch: {pipeline_key}={buffer_angles_dict[pipeline_key]} vs {vis_key}={vis_angles[vis_key]}"

        avg_knee = (buffer_angles_dict["knee_flexion_l"] + buffer_angles_dict["knee_flexion_r"]) / 2
        assert avg_knee == pytest.approx(vis_angles["knee_flex"], abs=1e-10)

    def test_buffer_angles_superset_of_visualizer(self):
        """The pipeline buffer stores all angles the visualizer has, plus more."""
        angles = _make_angles(_DEEP_SQUAT_KNEE_FLEX)
        buffer_dict = angles.as_dict()
        vis_frame = _extract_frame_data(_make_skeleton_3d(_DEEP_SQUAT_KNEE_FLEX), angles, 0)
        vis_angles = vis_frame["angles"]

        for vis_key in vis_angles:
            if vis_key == "knee_flex":
                continue
            pipeline_key = next(
                (pk for pk, vk in _ANGLE_KEY_MAP.items() if vk == vis_key), None
            )
            assert pipeline_key is not None, f"Visualizer key '{vis_key}' has no pipeline mapping"
            assert pipeline_key in buffer_dict, f"Pipeline key '{pipeline_key}' not in as_dict()"

    def test_consume_resets_buffer(self):
        """consume_bottom_frame() returns data once, then returns None."""
        from biomechanics.pipeline import BiomechanicsPipeline

        pipe = BiomechanicsPipeline.__new__(BiomechanicsPipeline)
        pipe._bottom_min_depth_ratio = -0.1
        pipe._bottom_kpts = [[0.0, 0.0, 0.0]] * 19
        pipe._bottom_angles = {"knee_flexion_l": 110.0}

        kpts, angles = pipe.consume_bottom_frame()
        assert kpts is not None
        assert angles is not None
        assert len(kpts) == 19
        assert pipe._bottom_min_depth_ratio == float("inf")

        kpts2, angles2 = pipe.consume_bottom_frame()
        assert kpts2 is None
        assert angles2 is None


class TestBottomFrameIPC:
    """Verify bottom-frame data flows through IPC bridge and session tracker."""

    @pytest.fixture
    def mock_ipc_client(self):
        class MockIPCClient:
            def __init__(self):
                self.messages = []
            def send_message(self, message):
                self.messages.append(message)
        return MockIPCClient()

    @pytest.fixture
    def bridge(self, mock_ipc_client):
        return IPCBridge(mock_ipc_client, CoachingConfig(), IPCConfig())

    @pytest.fixture
    def tracker(self, bridge):
        return SessionTracker(bridge, CoachingConfig())

    def _make_rep(self, rep_number: int) -> RepData:
        return RepData(
            rep_number=rep_number,
            start_time=0.0,
            end_time=2.5,
            start_frame=0,
            end_frame=75,
            max_depth_angle=110.0,
        )

    def test_bridge_includes_bottom_data_when_present(self, bridge, mock_ipc_client):
        """send_rep_complete includes bottom_kpts and bottom_angles in the message."""
        kpts = [[float(i), 0.0, 0.0] for i in range(19)]
        angles = {"knee_flexion_l": 110.0, "knee_flexion_r": 108.0}

        bridge.send_rep_complete(self._make_rep(1), bottom_kpts=kpts, bottom_angles=angles)

        rep_msg = next(m for m in mock_ipc_client.messages if m["type"] == "rep_complete")
        assert rep_msg["bottom_kpts"] == kpts
        assert rep_msg["bottom_angles"] == angles

    def test_bridge_omits_bottom_data_when_absent(self, bridge, mock_ipc_client):
        """send_rep_complete without bottom data doesn't include those keys."""
        bridge.send_rep_complete(self._make_rep(1))

        rep_msg = next(m for m in mock_ipc_client.messages if m["type"] == "rep_complete")
        assert "bottom_kpts" not in rep_msg
        assert "bottom_angles" not in rep_msg

    def test_tracker_forwards_bottom_data_to_bridge(self, tracker, mock_ipc_client):
        """SessionTracker.on_rep_complete passes bottom data through to bridge."""
        kpts = [[float(i), 0.0, 0.0] for i in range(19)]
        angles = {"knee_flexion_l": 110.0}

        tracker.on_rep_complete(self._make_rep(1), bottom_kpts=kpts, bottom_angles=angles)

        rep_msg = next(m for m in mock_ipc_client.messages if m["type"] == "rep_complete")
        assert rep_msg["bottom_kpts"] == kpts
        assert rep_msg["bottom_angles"] == angles

    def test_tracker_backward_compatible_without_bottom_data(self, tracker, mock_ipc_client):
        """Existing callers that don't pass bottom data still work."""
        tracker.on_rep_complete(self._make_rep(1))

        rep_msg = next(m for m in mock_ipc_client.messages if m["type"] == "rep_complete")
        assert rep_msg["type"] == "rep_complete"
        assert rep_msg["rep_number"] == 1
        assert "bottom_kpts" not in rep_msg


class TestBottomFrameEdgeCases:
    """Edge cases for the buffering logic."""

    def test_plateau_keeps_first_lowest(self):
        """When multiple frames tie for the lowest hip, the first one wins (strict <)."""
        flex_sequence = [15.0, 110.0, 110.0, 110.0, 50.0]

        bottom_min = float("inf")
        captured_idx = -1

        for i, flex in enumerate(flex_sequence):
            depth = _depth_ratio(_make_skeleton_3d(flex))
            if depth < bottom_min:
                bottom_min = depth
                captured_idx = i

        assert captured_idx == 1

    def test_monotonic_descent_captures_last(self):
        """If the hip only drops (no ascent phase), the last frame is the bottom."""
        flex_sequence = [15.0, 40.0, 70.0, 95.0, 115.0]

        bottom_min = float("inf")
        captured_idx = -1

        for i, flex in enumerate(flex_sequence):
            depth = _depth_ratio(_make_skeleton_3d(flex))
            if depth < bottom_min:
                bottom_min = depth
                captured_idx = i

        assert captured_idx == 4

    def test_find_bottom_frame_ignores_knee_angle(self):
        """The bottom is where the hip is lowest, even if another frame reports more knee bend."""
        deep = _extract_frame_data(_make_skeleton_3d(110.0), _make_angles(90.0), 0)
        shallow = _extract_frame_data(_make_skeleton_3d(60.0), _make_angles(120.0), 1)

        assert find_bottom_frame([shallow, deep])["frame"] == 0
