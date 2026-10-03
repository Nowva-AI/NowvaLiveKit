"""Tests for MainMenuAgent tools: shutdown confirmation and quick-exercise routing."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.agents.main_menu_agent import MainMenuAgent
from agent.agents.shared.base_agent import BaseNovaAgent
from agent.core.agent_state import AgentState


@pytest.fixture
def main_menu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MainMenuAgent:
    monkeypatch.setattr(AgentState, "_load_user_from_database", lambda self, user_id: None)
    monkeypatch.setattr(BaseNovaAgent, "_log_function_call", lambda self, *args, **kwargs: None)
    state = AgentState(state_dir=tmp_path)
    state.state["user"]["id"] = "athlete-1"
    userdata = SimpleNamespace(state=state, compaction_service=None, visual_bridge=None, affect_service=None)
    return MainMenuAgent(state=state, userdata=userdata)


def _user_turn(agent: MainMenuAgent, text: str = "hi") -> None:
    """A user message in the agent's chat, as LiveKit adds it for spoken and typed turns."""
    ctx = agent.chat_ctx.copy()
    ctx.add_message(role="user", content=text)
    asyncio.run(agent.update_chat_ctx(ctx))


def _shutdown(agent: MainMenuAgent, confirmed: bool = False):
    return asyncio.run(agent.shutdown(None, confirmed=confirmed))


class TestShutdownConfirmation:
    def test_first_request_only_asks_to_confirm(self, main_menu: MainMenuAgent) -> None:
        _user_turn(main_menu)
        _shutdown(main_menu)
        assert not main_menu.state.get("shutdown_requested")

    def test_confirmed_flag_without_prior_question_still_asks(self, main_menu: MainMenuAgent) -> None:
        _user_turn(main_menu)
        _shutdown(main_menu, confirmed=True)
        assert not main_menu.state.get("shutdown_requested")

    def test_yes_on_next_turn_shuts_down(self, main_menu: MainMenuAgent) -> None:
        _user_turn(main_menu)
        _shutdown(main_menu)
        _user_turn(main_menu)
        _shutdown(main_menu, confirmed=True)
        assert main_menu.state.get("shutdown_requested") is True

    def test_stale_confirmation_question_does_not_count(self, main_menu: MainMenuAgent) -> None:
        _user_turn(main_menu)
        _shutdown(main_menu)
        _user_turn(main_menu)
        _user_turn(main_menu)
        _shutdown(main_menu, confirmed=True)
        assert not main_menu.state.get("shutdown_requested")
