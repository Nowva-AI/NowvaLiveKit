"""Tests for the tracking_quality messages IPCBridge sends the voice agent.

Mid-set the coach must know when it can no longer see the legs (and when it can
again), so a silent stretch is explained instead of looking like a clean rep —
but an athlete racking the bar and walking off after the last rep is not lost.
"""

from __future__ import annotations

from biomechanics.coaching.ipc_bridge import (
    TRACKING_ARMED_AFTER_REP_S,
    TRACKING_LOST_AFTER_S,
    TRACKING_RECOVERED_AFTER_S,
    IPCBridge,
)
from biomechanics.utils.types import PipelineFrame

FRAME_DT_S = 1.0 / 12.0
START_S = 1000.0
LEFT_KNEE = ["left_knee"]
REP_S = 2.0
WALK_OFF_S = 2.0


class _RecordingClient:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    def send_message(self, message: dict) -> None:
        self.messages.append(message)


def _frame(index: int, missing: list[str] | None, lost_cameras: list[str] | None = None) -> PipelineFrame:
    return PipelineFrame(
        frame_index=index,
        timestamp=START_S + index * FRAME_DT_S,
        missing_keypoints=missing,
        lost_cameras=lost_cameras,
    )


def _tracking_messages(client: _RecordingClient) -> list[dict]:
    return [message for message in client.messages if message["type"] == "tracking_quality"]


def _frames_for(seconds: float) -> int:
    return int(seconds / FRAME_DT_S) + 1


class TestTrackingQuality:
    def test_brief_dropout_says_nothing(self):
        """The Kalman carries a keypoint through a short gap; only a sustained loss is news."""
        client = _RecordingClient()
        bridge = IPCBridge(client)
        for index in range(_frames_for(TRACKING_LOST_AFTER_S) - 2):
            bridge.update_tracking_quality(_frame(index, LEFT_KNEE), active=True, in_rep=True)
        bridge.update_tracking_quality(_frame(100, []), active=True, in_rep=True)

        assert _tracking_messages(client) == []

    def test_sustained_loss_then_recovery(self):
        client = _RecordingClient()
        bridge = IPCBridge(client)
        lost_frames = _frames_for(TRACKING_LOST_AFTER_S) + 3
        for index in range(lost_frames):
            bridge.update_tracking_quality(_frame(index, LEFT_KNEE), active=True, in_rep=True)
        for index in range(lost_frames, lost_frames + _frames_for(TRACKING_RECOVERED_AFTER_S) + 1):
            bridge.update_tracking_quality(_frame(index, []), active=True, in_rep=True)

        assert _tracking_messages(client) == [
            {"type": "tracking_quality", "status": "lost", "reason": "keypoints", "missing": LEFT_KNEE},
            {"type": "tracking_quality", "status": "recovered", "reason": "keypoints", "missing": []},
        ]

    def test_frames_without_a_new_capture_carry_no_information(self):
        client = _RecordingClient()
        bridge = IPCBridge(client)
        lost_frames = _frames_for(TRACKING_LOST_AFTER_S) + 1
        for index in range(lost_frames):
            bridge.update_tracking_quality(_frame(index, LEFT_KNEE), active=True, in_rep=True)
        bridge.update_tracking_quality(_frame(lost_frames, None), active=True, in_rep=True)

        assert [message["status"] for message in _tracking_messages(client)] == ["lost"]

    def test_outside_a_set_nothing_is_reported(self):
        """Walking away after the last rep, or resting, is not lost tracking."""
        client = _RecordingClient()
        bridge = IPCBridge(client)
        for index in range(_frames_for(TRACKING_LOST_AFTER_S) * 3):
            bridge.update_tracking_quality(_frame(index, LEFT_KNEE), active=False, in_rep=False)

        assert _tracking_messages(client) == []

    def test_a_new_set_starts_clean(self):
        client = _RecordingClient()
        bridge = IPCBridge(client)
        lost_frames = _frames_for(TRACKING_LOST_AFTER_S) + 1
        for index in range(lost_frames):
            bridge.update_tracking_quality(_frame(index, LEFT_KNEE), active=True, in_rep=True)
        bridge.update_tracking_quality(_frame(lost_frames, LEFT_KNEE), active=False, in_rep=False)
        bridge.update_tracking_quality(_frame(lost_frames + 1, []), active=True, in_rep=False)

        assert [message["status"] for message in _tracking_messages(client)] == ["lost"]

    def test_walking_off_after_the_last_rep_is_not_lost(self):
        """A short set ends in silence: the athlete racks and walks away, and the
        agent only closes the set after its idle timeout."""
        client = _RecordingClient()
        bridge = IPCBridge(client)
        index = 0
        for _ in range(_frames_for(REP_S)):
            bridge.update_tracking_quality(_frame(index, []), active=True, in_rep=True)
            index += 1
        for _ in range(_frames_for(TRACKING_ARMED_AFTER_REP_S)):
            bridge.update_tracking_quality(_frame(index, []), active=True, in_rep=False)
            index += 1
        for _ in range(_frames_for(WALK_OFF_S)):
            bridge.update_tracking_quality(_frame(index, LEFT_KNEE), active=True, in_rep=False)
            index += 1

        assert _tracking_messages(client) == []

    def test_loss_right_after_a_rep_is_reported(self):
        client = _RecordingClient()
        bridge = IPCBridge(client)
        index = 0
        for _ in range(_frames_for(REP_S)):
            bridge.update_tracking_quality(_frame(index, []), active=True, in_rep=True)
            index += 1
        for _ in range(_frames_for(TRACKING_LOST_AFTER_S) + 1):
            bridge.update_tracking_quality(_frame(index, LEFT_KNEE), active=True, in_rep=False)
            index += 1

        assert [message["status"] for message in _tracking_messages(client)] == ["lost"]

    def test_flickering_keypoint_does_not_toggle_lost_and_recovered(self):
        client = _RecordingClient()
        bridge = IPCBridge(client)
        index = 0
        for _ in range(_frames_for(TRACKING_LOST_AFTER_S) + 1):
            bridge.update_tracking_quality(_frame(index, LEFT_KNEE), active=True, in_rep=True)
            index += 1
        for flicker in range(_frames_for(WALK_OFF_S)):
            missing = [] if flicker % 2 == 0 else LEFT_KNEE
            bridge.update_tracking_quality(_frame(index, missing), active=True, in_rep=True)
            index += 1
        assert [message["status"] for message in _tracking_messages(client)] == ["lost"]

        for _ in range(_frames_for(TRACKING_RECOVERED_AFTER_S) + 1):
            bridge.update_tracking_quality(_frame(index, []), active=True, in_rep=True)
            index += 1
        assert [message["status"] for message in _tracking_messages(client)] == ["lost", "recovered"]


class TestCameraStatus:
    def test_lost_and_recovered_cameras_are_reported_once_each(self):
        client = _RecordingClient()
        bridge = IPCBridge(client)
        bridge.update_camera_status(_frame(0, [], lost_cameras=[]))
        bridge.update_camera_status(_frame(1, [], lost_cameras=["1"]))
        bridge.update_camera_status(_frame(2, [], lost_cameras=["1"]))
        bridge.update_camera_status(_frame(3, [], lost_cameras=[]))

        assert _tracking_messages(client) == [
            {"type": "tracking_quality", "status": "lost", "reason": "camera", "missing": [], "cameras": ["1"]},
            {"type": "tracking_quality", "status": "recovered", "reason": "camera", "missing": [], "cameras": []},
        ]

    def test_single_camera_frames_carry_no_camera_status(self):
        client = _RecordingClient()
        bridge = IPCBridge(client)
        bridge.update_camera_status(_frame(0, [], lost_cameras=None))

        assert _tracking_messages(client) == []
