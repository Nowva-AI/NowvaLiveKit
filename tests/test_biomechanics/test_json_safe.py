"""
Tests for NaN/inf sanitization at the JSON/IPC boundary.
"""

from __future__ import annotations

import math

import numpy as np

from biomechanics.coaching.ipc_bridge import IPCBridge
from biomechanics.utils.json_safe import nan_to_none
from biomechanics.utils.types import FaultEvent, FaultSeverity, JointAngles, PipelineFrame, RepData


class _RecordingClient:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    def send_message(self, message: dict) -> None:
        self.messages.append(message)


class TestNanToNone:

    def test_nan_and_inf_become_none(self):
        assert nan_to_none(float("nan")) is None
        assert nan_to_none(float("inf")) is None
        assert nan_to_none(float("-inf")) is None
        assert nan_to_none(np.float64("nan")) is None

    def test_finite_values_and_other_types_pass_through(self):
        assert nan_to_none(1.5) == 1.5
        assert nan_to_none(np.float32(2.0)) == 2.0
        assert nan_to_none(3) == 3
        assert nan_to_none(True) is True
        assert nan_to_none("x") == "x"
        assert nan_to_none(None) is None

    def test_nested_containers_are_sanitized(self):
        payload = {"a": [1.0, float("nan"), {"b": (float("inf"), 2.0)}], "c": {"d": float("nan")}}
        assert nan_to_none(payload) == {"a": [1.0, None, {"b": [None, 2.0]}], "c": {"d": None}}


class TestIPCBridgeSanitizes:

    def test_frame_data_joint_angles_nan_become_none(self):
        client = _RecordingClient()
        bridge = IPCBridge(client)
        bridge.frame_send_interval = 1
        frame = PipelineFrame(
            frame_index=1,
            timestamp=0.0,
            joint_angles=JointAngles(knee_flexion_l=float("nan"), knee_flexion_r=95.0),
        )
        bridge.send_frame_data(frame)
        angles = client.messages[-1]["joint_angles"]
        assert angles["knee_flexion_l"] is None
        assert angles["knee_flexion_r"] == 95.0

    def test_rep_complete_details_and_bottom_angles_nan_become_none(self):
        client = _RecordingClient()
        bridge = IPCBridge(client)
        fault = FaultEvent(
            fault_type="knee_valgus",
            severity=FaultSeverity.MILD,
            severity_score=1.0,
            message="m",
            details={"hip_adduction_l": float("nan"), "max_valgus": 13.0},
        )
        rep = RepData(
            rep_number=1, start_time=0.0, end_time=2.0, start_frame=0, end_frame=60,
            max_depth_angle=100.0, min_depth_angle=5.0, descent_time=1.0, ascent_time=1.0,
            faults=[fault],
        )
        bridge.send_rep_complete(rep, bottom_angles={"trunk_flexion": float("nan"), "knee_flexion_l": 100.0})
        msg = client.messages[0]
        assert msg["type"] == "rep_complete"
        assert msg["faults_detailed"][0]["details"] == {"hip_adduction_l": None, "max_valgus": 13.0}
        assert msg["bottom_angles"] == {"trunk_flexion": None, "knee_flexion_l": 100.0}
        for message in client.messages:
            assert not _contains_nan(message)


def _contains_nan(value: object) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_contains_nan(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_nan(v) for v in value)
    return False
