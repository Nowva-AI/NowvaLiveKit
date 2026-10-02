"""Tests for onboarding account creation: a failed save must never reach the main menu."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import agent.agents.onboarding_agent as onboarding_module
from agent.agents.onboarding_agent import CollectOnboardingDataTask
from agent.agents.shared.userdata import UserData
from agent.core.agent_state import AgentState


@pytest.fixture
def state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AgentState:
    monkeypatch.setattr(AgentState, "_load_user_from_database", lambda self, user_id: None)
    return AgentState(state_dir=tmp_path)


def _confirm_email(state: AgentState) -> tuple[object, CollectOnboardingDataTask]:
    async def _run():
        userdata = UserData(state=state)
        userdata.temp_first_name = "Sam"
        userdata.temp_email = "sam@example.com"
        task = CollectOnboardingDataTask(state=state, userdata=userdata)
        audio_toggles: list[bool] = []
        task._activity = SimpleNamespace(
            session=SimpleNamespace(input=SimpleNamespace(set_audio_enabled=audio_toggles.append))
        )
        result = await task.confirm_email_correct(None)
        return result, task

    return asyncio.run(_run())


class TestAccountCreationFailure:
    def test_failed_save_tells_user_and_stays_in_onboarding(
        self, state: AgentState, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        def _fail(first_name: str, email: str):
            raise RuntimeError("database unreachable")

        monkeypatch.setattr(onboarding_module, "create_user_account", _fail)

        result, task = _confirm_email(state)

        assert result is not None
        assert task.done() is False
        assert state.get_mode() == "onboarding"
        assert state.get("user.id") is None
        assert "ONBOARDING_COMPLETE" not in capsys.readouterr().out

    def test_successful_save_completes_with_user_id(
        self, state: AgentState, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            onboarding_module, "create_user_account",
            lambda first_name, email: (SimpleNamespace(id="user-42"), "sam42"),
        )

        _, task = _confirm_email(state)

        assert task.done() is True
        assert state.get("user.id") == "user-42"
        assert state.get_mode() == "main_menu"


class TestOnboardingOutputPrivacy:
    def test_username_not_printed(
        self, state: AgentState, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        monkeypatch.setattr(
            onboarding_module, "create_user_account",
            lambda first_name, email: (SimpleNamespace(id="user-42"), "sam42"),
        )

        _confirm_email(state)

        assert "sam42" not in capsys.readouterr().out
