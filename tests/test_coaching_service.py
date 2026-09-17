"""Tests for CoachingService shutdown behavior."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.services.assessment_logger import AssessmentLogger
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
