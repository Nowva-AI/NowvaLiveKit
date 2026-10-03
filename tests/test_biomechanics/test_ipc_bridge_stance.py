"""Tests for the stance metrics IPCBridge attaches to frame_data.

These drive the intra-set stance/toe-out coaching loop, so the values must
be in the same convention the diagnosis engine's targets are expressed in.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from biomechanics.coaching.ipc_bridge import IPCBridge
from biomechanics.config import IPCConfig
from biomechanics.diagnosis.bridge import KEYPOINT_TO_BIACROMIAL_RATIO
from biomechanics.utils.types import JointAngles, PipelineFrame, Skeleton3D

ANGLE_TOLERANCE_DEG = 0.5
RATIO_TOLERANCE = 0.01

SHOULDER_WIDTH_M = 0.40
ANKLE_HALF_SPACING_M = 0.15
HIP_HALF_WIDTH_M = 0.10
FOOT_LENGTH_M = 0.20


class _RecordingClient:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    def send_message(self, message: dict) -> None:
        self.messages.append(message)


def _standing_kpts(toe_out_deg: float = 0.0) -> np.ndarray:
    """Build MediaPipe world keypoints for a standing athlete.

    MediaPipe world coords: X = subject's left, Y = down, Z = away from
    camera. The athlete faces the camera, so their forward is -Z.
    """
    kpts = np.zeros((21, 3))
    # Hips define the athlete's forward axis for toe-out.
    kpts[11] = [HIP_HALF_WIDTH_M, 0.0, 0.0]
    kpts[12] = [-HIP_HALF_WIDTH_M, 0.0, 0.0]
    kpts[15] = [ANKLE_HALF_SPACING_M, 0.9, 0.0]
    kpts[16] = [-ANKLE_HALF_SPACING_M, 0.9, 0.0]

    angle_rad = math.radians(toe_out_deg)
    lateral = FOOT_LENGTH_M * math.sin(angle_rad)
    forward = -FOOT_LENGTH_M * math.cos(angle_rad)
    # Toe-out is symmetric: each foot rotates away from the midline.
    kpts[17] = [ANKLE_HALF_SPACING_M + lateral, 0.9, forward]
    kpts[18] = [-ANKLE_HALF_SPACING_M - lateral, 0.9, forward]
    return kpts


def _make_frame(kpts: np.ndarray) -> PipelineFrame:
    skeleton = Skeleton3D.from_numpy(kpts)
    return PipelineFrame(
        frame_index=1,
        timestamp=0.0,
        joint_angles=JointAngles(),
        skeleton_3d=skeleton,
    )


def _send_one_frame(bridge: IPCBridge, client: _RecordingClient, kpts: np.ndarray) -> dict:
    """Drive the bridge past its frame throttle and return the frame_data msg."""
    for _ in range(bridge.frame_send_interval):
        bridge.send_frame_data(_make_frame(kpts), rep_phase="idle")
    frames = [m for m in client.messages if m["type"] == "frame_data"]
    assert frames, "no frame_data emitted"
    return frames[-1]


@pytest.fixture
def bridge_and_client() -> tuple[IPCBridge, _RecordingClient]:
    client = _RecordingClient()
    bridge = IPCBridge(client, ipc_config=IPCConfig())
    bridge.shoulder_width_m = SHOULDER_WIDTH_M
    return bridge, client


class TestStanceMetrics:
    def test_feet_straight_ahead_report_zero_toe_out(self, bridge_and_client):
        bridge, client = bridge_and_client
        msg = _send_one_frame(bridge, client, _standing_kpts(toe_out_deg=0.0))
        assert msg["foot_direction_angle_l"] == pytest.approx(0.0, abs=ANGLE_TOLERANCE_DEG)
        assert msg["foot_direction_angle_r"] == pytest.approx(0.0, abs=ANGLE_TOLERANCE_DEG)

    def test_toe_out_matches_the_angle_the_athlete_actually_has(self, bridge_and_client):
        """Guards the MediaPipe→viewer transform: without it this reads ~115°."""
        bridge, client = bridge_and_client
        msg = _send_one_frame(bridge, client, _standing_kpts(toe_out_deg=25.0))
        assert msg["foot_direction_angle_l"] == pytest.approx(25.0, abs=ANGLE_TOLERANCE_DEG)
        assert msg["foot_direction_angle_r"] == pytest.approx(25.0, abs=ANGLE_TOLERANCE_DEG)

    def test_toe_in_reads_negative(self, bridge_and_client):
        """Toe-in used to read exactly like toe-out (unsigned arccos)."""
        bridge, client = bridge_and_client
        msg = _send_one_frame(bridge, client, _standing_kpts(toe_out_deg=-10.0))
        assert msg["foot_direction_angle_l"] == pytest.approx(-10.0, abs=ANGLE_TOLERANCE_DEG)
        assert msg["foot_direction_angle_r"] == pytest.approx(-10.0, abs=ANGLE_TOLERANCE_DEG)

    def test_toe_out_is_comparable_to_engine_targets(self, bridge_and_client):
        """Engine targets live in 15-40°; a neutral stance must fall below them."""
        bridge, client = bridge_and_client
        msg = _send_one_frame(bridge, client, _standing_kpts(toe_out_deg=5.0))
        assert 0.0 <= msg["foot_direction_angle_l"] < 15.0

    def test_stance_width_ratio_is_ankle_spread_over_biacromial_width(self, bridge_and_client):
        """Shoulder keypoints are joint centres; coaching's shoulder width is wider."""
        bridge, client = bridge_and_client
        msg = _send_one_frame(bridge, client, _standing_kpts())
        expected = (2 * ANKLE_HALF_SPACING_M) / (SHOULDER_WIDTH_M / KEYPOINT_TO_BIACROMIAL_RATIO)
        assert msg["stance_width_ratio"] == pytest.approx(expected, abs=RATIO_TOLERANCE)

    def test_targets_absent_until_athlete_params_are_set(self, bridge_and_client):
        bridge, client = bridge_and_client
        msg = _send_one_frame(bridge, client, _standing_kpts())
        assert "target_stance_ratio" not in msg

    def test_targets_included_once_calibrated(self, bridge_and_client):
        bridge, client = bridge_and_client
        bridge.set_athlete_params(
            {"shoulder_width_m": SHOULDER_WIDTH_M, "femur_avg_m": 0.45, "tibia_avg_m": 0.40},
            {"peakDorsi": 25.0},
        )
        msg = _send_one_frame(bridge, client, _standing_kpts())
        assert msg["target_stance_ratio"] >= 1.1
        assert 15.0 <= msg["target_toe_out_deg"] <= 40.0

    def test_stance_metrics_omitted_without_shoulder_width(self):
        client = _RecordingClient()
        bridge = IPCBridge(client, ipc_config=IPCConfig())
        msg = _send_one_frame(bridge, client, _standing_kpts())
        assert "stance_width_ratio" not in msg
        assert msg["rep_phase"] == "idle"
