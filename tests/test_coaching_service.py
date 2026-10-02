"""Tests for CoachingService shutdown behavior."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from unittest.mock import AsyncMock

from agent.services.assessment_logger import AssessmentLogger
from agent.services.coaching_orchestrator import CoachingOrchestrator
from agent.services.coaching_service import CoachingService


class TestStopFinalizesAssessment:
    def test_stop_finalizes_active_assessment_log(self, tmp_path):
        service = CoachingService(session=None, state=None)
        service._started = True
        service._assessment_logger = AssessmentLogger(
            session_dir=tmp_path,
            session_id="test_session",
            user_height_cm=180.0,
        )

        asyncio.run(service.stop())

        log = json.loads((tmp_path / "assessment" / "assessment_log.json").read_text())
        assert log["completed_at"] is not None
        assert log["passed"] is False
        assert service._assessment_logger is None

    def test_stop_without_assessment_is_clean(self, tmp_path):
        service = CoachingService(session=None, state=None)
        service._started = True

        asyncio.run(service.stop())

        assert service._assessment_logger is None
        assert not (tmp_path / "assessment").exists()


def _wait_for(condition_fn, timeout_s: float = 5.0) -> bool:
    import time
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if condition_fn():
            return True
        time.sleep(0.02)
    return False


def _serve_one_client(server) -> threading.Thread:
    def _serve():
        try:
            server.accept_client()
            server.listen()
        except OSError:
            pass
    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()
    return thread


class TestListenerReconnect:
    def test_listener_reconnects_after_server_restart(self, monkeypatch):
        """main.py stops the coaching socket between workout passes — the
        listener must reconnect when the server rebinds, not die silently."""
        import tempfile
        import agent.services.coaching_service as cs
        from agent.core.ipc_communication import IPCServer

        socket_path = tempfile.mkdtemp() + "/coaching.sock"
        monkeypatch.setattr(cs, "COACHING_SOCKET_PATH", socket_path)
        monkeypatch.setattr(cs, "LISTENER_RECONNECT_POLL_S", 0.05)

        service = CoachingService(session=None, state=None)

        server = IPCServer(socket_path=socket_path)
        server.bind()
        _serve_one_client(server)

        asyncio.run(service._start_ipc_listener())
        assert _wait_for(lambda: service._listener_running)

        # Workout ends: main.py stops the coaching server
        server.stop()
        assert _wait_for(lambda: not service._listener_running)

        # Next workout: main.py rebinds — the listener must come back
        server = IPCServer(socket_path=socket_path)
        server.bind()
        _serve_one_client(server)
        assert _wait_for(lambda: service._listener_running)

        service._stop_ipc_listener()
        server.stop()

    def test_start_assessment_resets_demo_latch(self, tmp_path):
        service = CoachingService(session=None, state=None)
        service._assessment_demo_played = True

        service.start_assessment(
            session_dir=tmp_path, session_id="s", user_height_cm=180.0,
        )

        assert service._assessment_demo_played is False


class _StubState:
    def __init__(self, user_id: str) -> None:
        self.values: dict = {"user.id": user_id}

    def get(self, key: str, default: object = None) -> object:
        return self.values.get(key, default)

    def set(self, key: str, value: object) -> None:
        self.values[key] = value

    def save_state(self) -> None:
        pass


class TestCalibrationCompletePersistence:
    def test_calibration_without_athlete_params_keeps_stored_params(self, monkeypatch):
        """A calibration_complete that lost its body measurements must not
        erase the stored ones — returning users would lose diagnosis."""
        import types
        import uuid
        from db.models import UserCalibration

        stored_params = {
            "shoulder_width_m": 0.39, "femur_avg_m": 0.46, "torso_avg_m": 0.55,
            "hip_width_m": 0.27, "tibia_avg_m": 0.44, "foot_avg_m": 0.21,
        }
        user_id = uuid.uuid4()
        row = UserCalibration(
            user_id=user_id, movement_pattern="squat", peaks={}, thresholds={},
            calibration_reps=5, athlete_params=dict(stored_params),
            baseline={"peakDorsi": 33.0, "peakKneeFlex": 118.0},
        )

        class _Query:
            def filter(self, *conditions):
                return self

            def first(self):
                return row

        class _Session:
            def query(self, model):
                return _Query()

            def add(self, new_row):
                raise AssertionError("existing row must be updated, not re-added")

            def commit(self):
                pass

            def close(self):
                pass

        monkeypatch.setitem(
            sys.modules, "db.database", types.SimpleNamespace(SessionLocal=_Session),
        )
        service = CoachingService(session=None, state=_StubState(str(user_id)))

        async def _no_reply(instructions):
            return None

        monkeypatch.setattr(service, "_coaching_llm_reply", _no_reply)

        asyncio.run(service._on_calibration_complete({
            "type": "calibration_complete",
            "movement_pattern": "squat",
            "peaks": {"trunk_flexion": 41.0},
            "thresholds": {"knee_valgus": {"mild": 12.5}},
        }))

        assert row.athlete_params == stored_params
        assert row.baseline == {"peakDorsi": 33.0, "peakKneeFlex": 118.0}
        assert row.thresholds == {"knee_valgus": {"mild": 12.5}}


class _FakeRecorder:
    def __init__(self) -> None:
        self.faults: list[dict] = []
        self.reps: list[dict] = []
        self.sets: list[dict] = []

    def record_fault(self, message: dict) -> None:
        self.faults.append(message)

    def record_rep(self, message: dict) -> None:
        self.reps.append(message)

    def record_set(self, message: dict) -> None:
        self.sets.append(message)


def _wired_service() -> tuple[CoachingService, CoachingOrchestrator, _FakeRecorder]:
    service = CoachingService(session=None, state=None)
    orchestrator = CoachingOrchestrator(
        play_cached_audio_fn=AsyncMock(),
        generate_llm_reply_fn=AsyncMock(),
        get_cue_audio_fn=lambda key: bool(key),
    )
    orchestrator.reset_set(target_reps=10)
    recorder = _FakeRecorder()
    service._coaching_orchestrator = orchestrator
    service._biomech_recorder = recorder
    service._workout_active_flag = True
    return service, orchestrator, recorder


class TestPipelineMessageWiring:
    def test_approximate_fault_is_recorded_but_not_cued(self):
        service, orchestrator, recorder = _wired_service()
        message = {
            "type": "fault", "fault_type": "hip_shoot", "severity": "moderate",
            "cue": None, "rep_number": 2, "side": None, "observability": "approximate",
        }
        asyncio.run(service._handle_message(message))
        assert recorder.faults == [message]
        assert orchestrator._queue.empty()

    def test_side_cue_reaches_the_orchestrator_as_given(self):
        service, orchestrator, _ = _wired_service()
        asyncio.run(service._handle_message({
            "type": "fault", "fault_type": "knee_valgus", "severity": "moderate",
            "cue": "knees_out_left", "rep_number": 2, "side": "left",
            "observability": "observable",
        }))
        assert orchestrator._queue.get_nowait().cue_key == "knees_out_left"

    def test_rep_complete_passes_highlights_and_set_number(self):
        service, orchestrator, _ = _wired_service()
        orchestrator.positive_cue_keys = ["strong"]
        asyncio.run(service._handle_message({
            "type": "rep_complete", "rep_number": 1, "is_clean": True,
            "faults_in_rep": [], "set_number": 3, "highlights": ["best_rep_so_far"],
        }))
        assert orchestrator._build_set_summary()["diagnosis_set_number"] == 3
        assert orchestrator._queue.get_nowait().cue_key == "strong"

    def test_diagnosis_complete_passes_its_set_number(self):
        service, orchestrator, _ = _wired_service()
        asyncio.run(service._handle_message({
            "type": "diagnosis_complete", "set_number": 3,
            "diagnosis": {
                "confidence": 0.8,
                "immediate_causes": [{"cause_id": "narrow_stance", "explanation": "x"}],
            },
            "scoring": {"mean_score": 0.7},
        }))
        assert orchestrator._pending_diagnosis_set_number == 3
        assert service.get_top_cause_id() == "narrow_stance"


class _FakeCoachingIPC:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    def send_message(self, message: dict) -> None:
        self.messages.append(message)


class _FakeAudioCueService:
    session = object()
    rep_track_ready = False

    async def cache_cues(self, cues: dict) -> None:
        pass


def _two_exercise_session_dict(first_sets: int = 1) -> dict:
    """A squat then a press, as main_menu_agent stores a scheduled workout."""
    from agent.core.workout_session import WorkoutSession

    squat = WorkoutSession.create_quick_session("u1", "Barbell Back Squat", sets=first_sets, reps=5, rest_seconds=90)
    press = WorkoutSession.create_quick_session("u1", "Barbell Overhead Press", sets=3, reps=8, weight=95.0)
    session = squat.to_dict()
    second = press.to_dict()["exercises"][0]
    second["order_number"] = 2
    session["exercises"].append(second)
    return session


def _service_on_last_squat_set() -> tuple[CoachingService, CoachingOrchestrator, _FakeCoachingIPC, _StubState]:
    state = _StubState("u1")
    state.set("workout.current_session", _two_exercise_session_dict())
    state.set("workout.exercise_name", "Barbell Back Squat")
    service, orchestrator, _ = _wired_service()
    service._state = state
    ipc = _FakeCoachingIPC()
    service._coaching_ipc = ipc
    return service, orchestrator, ipc, state


class TestExerciseSwitch:
    def test_finishing_an_exercise_closes_its_set_before_switching(self):
        """rest_start must reach the pipeline first so the squat's last set
        and diagnosis are finalized under the squat profile."""
        service, orchestrator, ipc, state = _service_on_last_squat_set()

        new_target = asyncio.run(service._advance_workout_set())

        assert new_target == 8
        assert [m["type"] for m in ipc.messages] == ["rest_start", "set_exercise"]
        switch = ipc.messages[1]
        assert switch["exercise_name"] == "Barbell Overhead Press"
        assert switch["total_sets"] == 3
        assert switch["target_reps"] == 8
        assert switch["weight_lbs"] == 95.0
        assert state.get("workout.exercise_name") == "Barbell Overhead Press"
        assert orchestrator.next_exercise_pending

    def test_next_set_of_the_same_exercise_does_not_switch(self):
        service, orchestrator, ipc, state = _service_on_last_squat_set()
        state.set("workout.current_session", _two_exercise_session_dict(first_sets=2))

        asyncio.run(service._advance_workout_set())

        assert [m["type"] for m in ipc.messages] == ["rest_start"]
        assert not orchestrator.next_exercise_pending

    def test_skipping_to_the_next_exercise_starts_it_now(self):
        from agent.core.workout_session import WorkoutSession

        service, orchestrator, ipc, state = _service_on_last_squat_set()
        session = WorkoutSession.from_dict(state.get("workout.current_session"))
        session.skip_current_exercise()
        state.set("workout.current_session", session.to_dict())

        service.start_current_exercise()

        assert [m["type"] for m in ipc.messages] == ["set_exercise"]
        assert not orchestrator.next_exercise_pending
        assert orchestrator._total_sets == 3
        assert orchestrator._set_target_reps == 8

    def test_cues_for_a_new_exercise_open_a_new_db_session(self, monkeypatch):
        service, orchestrator, _, _ = _service_on_last_squat_set()
        service._audio_cue_service = _FakeAudioCueService()
        service._biomech_recorder.exercise = "squat"
        calls: list[str] = []

        async def _close() -> None:
            calls.append("close")

        monkeypatch.setattr(service, "_close_biomech_recording", _close)
        monkeypatch.setattr(service, "_start_biomech_recording", lambda: calls.append("start"))

        asyncio.run(service._on_cache_cues({
            "type": "cache_cues", "exercise_name": "Barbell Overhead Press",
            "profile": "overhead_press", "cues": {"press_lockout": "press_lockout", "good_rep": "good_rep"},
        }))

        assert calls == ["close", "start"]
        assert orchestrator._exercise_name == "Barbell Overhead Press"
        assert not orchestrator._is_squat

    def test_db_session_is_tagged_with_the_exercise_profile(self, monkeypatch):
        import db.biomechanics_persistence as persistence

        created: list[dict] = []

        class _Recorder:
            session_id = "s1"

            def __init__(self, **kwargs) -> None:
                created.append(kwargs)

            def start(self) -> None:
                pass

        monkeypatch.setattr(persistence, "BiomechanicsRecorder", _Recorder)
        state = _StubState("u1")
        state.set("workout.exercise_name", "Barbell Romanian Deadlift")
        service = CoachingService(session=None, state=state)

        async def _no_baseline(user_id):
            return None

        monkeypatch.setattr(service, "_fetch_progress_baseline", _no_baseline)

        async def _run() -> None:
            service._start_biomech_recording()

        asyncio.run(_run())
        assert created[0]["exercise"] == "romanian_deadlift"


class TestCalibrationNeedsAPattern:
    def test_calibration_without_movement_pattern_is_not_saved(self, monkeypatch):
        """It used to default to "squat" and overwrite the athlete's squat calibration."""
        import types

        class _Session:
            def query(self, model):
                raise AssertionError("nothing may be written without a movement pattern")

            def close(self):
                pass

        monkeypatch.setitem(sys.modules, "db.database", types.SimpleNamespace(SessionLocal=_Session))
        service = CoachingService(session=None, state=_StubState("u1"))

        async def _no_reply(instructions):
            return None

        monkeypatch.setattr(service, "_coaching_llm_reply", _no_reply)
        asyncio.run(service._on_calibration_complete({
            "type": "calibration_complete", "peaks": {}, "thresholds": {},
        }))
