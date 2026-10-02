"""Tests for AgentState deep merge, path anchoring, validation, and file locking."""

from __future__ import annotations

import json
import logging
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.core.agent_state import AgentState

USER_ID = "test-user"
# Captured before the autouse fixture stubs it out.
REAL_LOAD_USER_FROM_DATABASE = AgentState._load_user_from_database
SAVES_PER_THREAD = 25


@pytest.fixture(autouse=True)
def no_database(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        AgentState, "_load_user_from_database", lambda self, user_id: None
    )


def _state_file(tmp_path: Path) -> Path:
    return tmp_path / f".agent_state_{USER_ID}.json"


def _make_state(tmp_path: Path) -> AgentState:
    return AgentState(user_id=USER_ID, state_dir=tmp_path)


class TestDeepMerge:
    def test_partial_workout_dict_preserves_default_siblings(self, tmp_path: Path) -> None:
        _state_file(tmp_path).write_text(json.dumps({"workout": {"active": True}}))

        state = _make_state(tmp_path)

        workout = state.to_dict()["workout"]
        assert workout["active"] is True
        assert workout["reps"] == 0
        assert "current_session" in workout
        assert "exercise" in workout

    def test_loaded_values_override_defaults(self, tmp_path: Path) -> None:
        _state_file(tmp_path).write_text(
            json.dumps({"mode": "workout", "user": {"name": "Marwan"}})
        )

        state = _make_state(tmp_path)

        assert state.get_mode() == "workout"
        assert state.get("user.name") == "Marwan"
        assert state.get("user.first_time_main_menu") is True


class TestStateFilePath:
    def test_save_writes_into_state_dir(self, tmp_path: Path) -> None:
        state = _make_state(tmp_path)
        state.save_state()

        saved = json.loads(_state_file(tmp_path).read_text())
        assert saved["user"]["id"] == USER_ID


class TestModeValidation:
    def test_all_known_modes_accepted(self, tmp_path: Path) -> None:
        state = _make_state(tmp_path)
        for mode in ("main_menu", "workout", "program_creation", "schedule", "onboarding"):
            state.switch_mode(mode)
            assert state.get_mode() == mode

    def test_unknown_mode_rejected(self, tmp_path: Path) -> None:
        state = _make_state(tmp_path)

        with pytest.raises(ValueError):
            state.switch_mode("disco_mode")

        assert state.get_mode() == "onboarding"


class TestSetValidation:
    def test_known_top_level_keys_accepted(self, tmp_path: Path) -> None:
        state = _make_state(tmp_path)

        state.set("workout.active", True)
        state.set("shutdown_requested", True)
        state.set("calibration.active", True)
        state.set("program_update.selected_program_id", 7)

        assert state.get("workout.active") is True
        assert state.get("shutdown_requested") is True
        assert state.get("calibration.active") is True
        assert state.get("program_update.selected_program_id") == 7

    def test_unknown_top_level_key_rejected(self, tmp_path: Path) -> None:
        state = _make_state(tmp_path)

        with pytest.raises(ValueError):
            state.set("wrokout.active", True)

        assert "wrokout" not in state.to_dict()


class TestCorruptedJsonLogging:
    def test_corrupted_state_file_logs_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        _state_file(tmp_path).write_text("{not valid json")

        with caplog.at_level(logging.WARNING, logger="agent.core.agent_state"):
            state = _make_state(tmp_path)

        assert "Corrupted state file" in caplog.text
        assert state.get_mode() == "onboarding"


class TestConcurrentSaves:
    def test_interleaved_saves_produce_valid_json(self, tmp_path: Path) -> None:
        state_a = _make_state(tmp_path)
        state_b = _make_state(tmp_path)
        state_a.set("workout.reps", 111)
        state_b.set("workout.reps", 222)

        def hammer(instance: AgentState) -> None:
            for _ in range(SAVES_PER_THREAD):
                instance.save_state()

        threads = [
            threading.Thread(target=hammer, args=(instance,))
            for instance in (state_a, state_b)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        saved = json.loads(_state_file(tmp_path).read_text())
        assert saved["workout"]["reps"] in (111, 222)
        assert saved["mode"] == "onboarding"


class TestNoPersonalDataInOutput:
    def test_update_user_does_not_print_values(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        state = _make_state(tmp_path)
        state.update_user(name="Sam", email="sam@example.com", username="sam42")
        output = capsys.readouterr().out
        for value in ("Sam", "sam@example.com", "sam42"):
            assert value not in output

    def test_database_load_does_not_print_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        import db.database as database

        user_row = SimpleNamespace(
            id=USER_ID, username="sam42", name="Sam", email="sam@example.com",
            height_cm=None, created_at=None,
        )

        class _FakeQuery:
            def filter(self, *args):
                return self

            def first(self):
                return user_row

        class _FakeDb:
            def query(self, model):
                return _FakeQuery()

            def close(self) -> None:
                return None

        monkeypatch.setattr(database, "SessionLocal", _FakeDb)
        state = AgentState(state_dir=tmp_path)
        capsys.readouterr()

        REAL_LOAD_USER_FROM_DATABASE(state, USER_ID)

        output = capsys.readouterr().out
        assert state.get("user.name") == "Sam"
        assert "Sam" not in output
        assert "sam42" not in output


class TestSessionScopedFlags:
    def test_a_new_session_forgets_the_last_sessions_mode_switch(self, tmp_path: Path) -> None:
        """Main menu re-entry lines name where the user came from — this session only."""
        first = _make_state(tmp_path)
        first.switch_mode("workout")
        first.switch_mode("main_menu")
        first.save_state()
        assert first.get("session.last_mode_switch") is not None

        second = _make_state(tmp_path)
        assert second.get("session.last_mode_switch") is None
