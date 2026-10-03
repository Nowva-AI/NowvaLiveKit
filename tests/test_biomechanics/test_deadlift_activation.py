"""Pipeline-side delivery of the deadlift (docs/deadlift/PLAN.md §3.4, §3.6, §4.1;
.claude/deadlift/CONTRACT.md §3): one activation path, camera-refine gates, the
cache_cues / frame_data / fault fields, and the session tracker's deadlift path.
The squat's messages must not change."""

from __future__ import annotations

import math

import numpy as np
import pytest

from biomechanics.coaching.ipc_bridge import IPCBridge
from biomechanics.coaching.session_tracker import SessionTracker
from biomechanics.config import BiomechanicsConfig
from biomechanics.deadlift.types import DeadliftFrameStatus, DeadliftPhase, DeadliftRepFeatures
from biomechanics.profiles.deadlift import DeadliftProfile
from biomechanics.utils.types import FaultEvent, FaultSeverity, JointAngles, PipelineFrame, RepData

DEADLIFT = "Barbell Conventional Deadlift"
SQUAT = "Barbell Back Squat"
SQUAT_CACHE_CUES_KEYS = {"type", "exercise_name", "profile", "cues"}


class _RecordingClient:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    def send_message(self, message: dict) -> None:
        self.messages.append(message)

    def of_type(self, message_type: str) -> list[dict]:
        return [message for message in self.messages if message["type"] == message_type]


class _RecordingProvider:
    def __init__(self) -> None:
        self.view_recording: list[bool] = []

    def set_view_recording(self, enabled: bool) -> None:
        self.view_recording.append(enabled)


class _Pipeline:
    """What _activate_profile touches on the pipeline."""

    def __init__(self, profile, provider=None) -> None:
        self.profile = profile
        self._multi_camera_provider = provider
        self.config = BiomechanicsConfig()
        self.meta: dict | None = None
        self.gravity: tuple | None = None

    def set_session_meta(self, meta: dict) -> None:
        self.meta = meta

    def resolve_gravity(self) -> None:
        if self.profile.needs_bar_3d and self._multi_camera_provider is not None:
            self.gravity = (np.array([0.0, -1.0, 0.0]), "measured")


class _CalibrationSession:
    def __init__(self, needs_world_anchor: bool = False) -> None:
        self.needs_world_anchor = needs_world_anchor
        self.calls: list[str] = []

    def on_rest_start(self) -> None:
        self.calls.append("refine")

    def anchor_world_on_lifter(self) -> None:
        self.calls.append("anchor")


def _rep(rep_number: int, velocity_mps: float, faults: list[FaultEvent] | None = None) -> RepData:
    features = DeadliftRepFeatures(rep_number=rep_number, concentric_velocity_mps=velocity_mps).model_dump()
    return RepData(
        rep_number=rep_number, start_time=float(rep_number), end_time=float(rep_number) + 2.0,
        start_frame=0, end_frame=1, max_depth_angle=math.nan, features=features, faults=faults or [],
    )


def _fault(fault_type: str, severity: FaultSeverity, min_tier: str, **details) -> FaultEvent:
    return FaultEvent(
        fault_type=fault_type, severity=severity, severity_score=1.0, message="", timestamp=100.0,
        rep_number=1, details={"min_tier": min_tier, **details},
    )


@pytest.fixture
def deadlift_ready(monkeypatch) -> None:
    monkeypatch.setattr(DeadliftProfile, "coaching_ready", True)


class TestActivation:
    def test_startup_and_switch_install_the_deadlift_the_same_way(self, monkeypatch, deadlift_ready):
        from biomechanics import pipeline_process
        from biomechanics.profiles import get_profile

        for switching in (False, True):
            client = _RecordingClient()
            bridge = IPCBridge(client)
            tracker = SessionTracker(bridge)
            provider = _RecordingProvider()
            pipe = _Pipeline(get_profile(DEADLIFT), provider)
            pipeline_process._activate_profile(pipe, tracker, bridge, DEADLIFT, {"grip": "hook"}, switching=switching)
            assert tracker.set_diagnosis is not None
            assert not tracker.diagnosis_enabled
            assert pipe.meta == {"grip": "hook"}
            assert pipe.gravity[1] == "measured"
            assert provider.view_recording == [False]
            assert client.of_type("cache_cues")[0]["profile"] == "deadlift"

    def test_the_squat_activation_sets_what_it_always_set(self):
        from biomechanics import pipeline_process
        from biomechanics.profiles import get_profile

        client = _RecordingClient()
        bridge = IPCBridge(client)
        tracker = SessionTracker(bridge)
        provider = _RecordingProvider()
        pipe = _Pipeline(get_profile(SQUAT), provider)
        pipeline_process._activate_profile(pipe, tracker, bridge, SQUAT, None, switching=False)
        assert tracker.diagnosis_enabled
        assert tracker.set_diagnosis is None
        assert pipe.gravity is None
        assert provider.view_recording == [True]
        assert set(client.of_type("cache_cues")[0]) == SQUAT_CACHE_CUES_KEYS

    def test_start_capture_hands_back_its_message(self, monkeypatch):
        from biomechanics import pipeline_process

        messages = iter([{"type": "status"}, {"type": "start_capture", "exercise_meta": {"grip": "mixed"}}])
        monkeypatch.setattr(pipeline_process, "_recv_framed", lambda sock: next(messages))
        client = type("Client", (), {"client_socket": object()})()
        assert pipeline_process._wait_for_start_capture(client)["exercise_meta"] == {"grip": "mixed"}

    def test_a_disconnect_while_waiting_is_none(self, monkeypatch):
        from biomechanics import pipeline_process

        monkeypatch.setattr(pipeline_process, "_recv_framed", lambda sock: None)
        client = type("Client", (), {"client_socket": object()})()
        assert pipeline_process._wait_for_start_capture(client) is None


class TestCameraRefineGate:
    def test_deadlift_sets_never_refine_or_re_anchor_the_rig(self, deadlift_ready):
        from biomechanics.pipeline_process import _anchor_world_if_pending, _refine_at_rest
        from biomechanics.profiles import get_profile

        session = _CalibrationSession(needs_world_anchor=True)
        _refine_at_rest(session, get_profile(DEADLIFT))
        _anchor_world_if_pending(session, get_profile(DEADLIFT))
        _refine_at_rest(session, get_profile("Barbell Sumo Deadlift"))
        assert session.calls == []

    def test_squat_sets_still_refine_and_re_anchor(self):
        from biomechanics.pipeline_process import _anchor_world_if_pending, _refine_at_rest
        from biomechanics.profiles import get_profile

        session = _CalibrationSession(needs_world_anchor=True)
        _refine_at_rest(session, get_profile(SQUAT))
        _anchor_world_if_pending(session, get_profile(SQUAT))
        assert session.calls == ["refine", "anchor"]

    def test_the_provider_stops_buffering_views_outside_a_capture_window(self):
        from biomechanics.pose.multi_camera import MultiCameraPoseProvider

        provider = MultiCameraPoseProvider.__new__(MultiCameraPoseProvider)
        provider._view_buffer = []
        provider._capture_window_open = False
        provider._bar_detector = None
        provider._view_recording = True
        views = {"0": object(), "1": object()}
        provider.set_view_recording(False)
        provider._record_views({}, views, 0.0, 1)
        assert provider._view_buffer == []
        provider._capture_window_open = True
        provider._window_frame_count = 0
        provider._record_views({}, views, 0.0, 2)
        assert len(provider._view_buffer) == 1


class TestBridgeMessages:
    def test_deadlift_cache_cues_carry_the_delivery_fields(self, deadlift_ready):
        client = _RecordingClient()
        IPCBridge(client).prepare_exercise(DEADLIFT, BiomechanicsConfig())
        message = client.of_type("cache_cues")[0]
        assert message["fault_to_cue"]["deadlift_bar_position"] == "deadlift_bar_midfoot"
        assert message["min_cue_tiers"]["deadlift_velocity_loss"] == "recap"
        assert message["set_idle_timeout_s"] == pytest.approx(30.0, abs=1e-9)
        assert message["waits_for_diagnosis"] is True
        assert message["closed_loop"] == "deadlift_bar_midfoot"

    def test_squat_cache_cues_are_unchanged(self):
        client = _RecordingClient()
        IPCBridge(client).prepare_exercise(SQUAT, BiomechanicsConfig())
        assert set(client.of_type("cache_cues")[0]) == SQUAT_CACHE_CUES_KEYS

    def test_frame_data_carries_the_phase_and_live_bar_offset(self):
        client = _RecordingClient()
        bridge = IPCBridge(client)
        bridge.frame_send_interval = 1
        status = DeadliftFrameStatus(phase=DeadliftPhase.STANCE, bar_midfoot_live_cm=7.5)
        bridge.send_frame_data(
            PipelineFrame(frame_index=1, timestamp=0.0, joint_angles=JointAngles(), exercise_status=status),
            rep_phase="stance",
        )
        message = client.of_type("frame_data")[0]
        assert message["deadlift_phase"] == "stance"
        assert message["bar_midfoot_live_cm"] == pytest.approx(7.5, abs=1e-9)
        assert message["bar_source"] == "bar"
        assert message["rep_phase"] == "stance"

    def test_squat_frame_data_has_no_deadlift_fields(self):
        client = _RecordingClient()
        bridge = IPCBridge(client)
        bridge.frame_send_interval = 1
        bridge.send_frame_data(PipelineFrame(frame_index=1, timestamp=0.0, joint_angles=JointAngles()), rep_phase="idle")
        assert "deadlift_phase" not in client.of_type("frame_data")[0]

    def test_setup_hips_cue_picks_its_direction_variant(self, deadlift_ready):
        client = _RecordingClient()
        bridge = IPCBridge(client)
        bridge.prepare_exercise(DEADLIFT, BiomechanicsConfig())
        bridge.send_fault(_fault("deadlift_setup_hips", FaultSeverity.SEVERE, "severe", direction="up"))
        assert client.of_type("fault")[0]["cue"] == "deadlift_hips_up"

    def test_a_deadlift_rep_has_no_depth_category(self):
        client = _RecordingClient()
        IPCBridge(client).send_rep_complete(_rep(1, 0.5))
        message = client.of_type("rep_complete")[0]
        assert message["depth_category"] == "n/a"
        assert message["max_depth_angle"] is None


class TestSessionTracker:
    def test_the_fastest_rep_without_a_correctable_fault_is_the_best(self):
        tracker = SessionTracker(IPCBridge(_RecordingClient()))
        below_min_tier = _fault("deadlift_bent_arms", FaultSeverity.MILD, "moderate")
        tracker.on_rep_complete(_rep(1, 0.50))
        tracker.on_rep_complete(_rep(2, 0.55, [below_min_tier]))
        tracker.on_rep_complete(_rep(3, 0.60, [_fault("deadlift_bar_drift", FaultSeverity.MODERATE, "moderate")]))
        highlights = [m["highlights"] for m in tracker.ipc_bridge.ipc_client.of_type("rep_complete")]
        assert highlights == [["clean"], ["best_rep_so_far"], []]

    def test_the_set_diagnosis_runs_per_rep_and_at_set_end(self, deadlift_ready):
        client = _RecordingClient()
        tracker = SessionTracker(IPCBridge(client))
        tracker.set_diagnosis = DeadliftProfile().create_set_diagnosis("triangulated")
        for rep_number in (1, 2, 3):
            features = DeadliftRepFeatures(
                rep_number=rep_number, bar_midfoot_setup_cm=7.0, bar_drift_cm=1.0, concentric_velocity_mps=0.5,
                setup_measured=True,
            ).model_dump()
            tracker.on_rep_complete(RepData(
                rep_number=rep_number, start_time=rep_number * 5.0, end_time=rep_number * 5.0 + 2.0,
                start_frame=0, end_frame=1, max_depth_angle=math.nan, features=features,
            ))
        tracker.force_end_set()
        assert client.of_type("rep_diagnosis")
        complete = client.of_type("diagnosis_complete")
        assert len(complete) == 1
        assert complete[0]["set_number"] == 1
        symptoms = [symptom["symptom_id"] for symptom in complete[0]["diagnosis"]["detected_symptoms"]]
        assert symptoms
