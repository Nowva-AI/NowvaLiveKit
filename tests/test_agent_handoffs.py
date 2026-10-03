"""Tests for context pruning, conversation carry-over across agent handoffs, and entry speech."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from livekit.agents import llm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.agents.main_menu_agent import MainMenuAgent
from agent.agents.schedule_agent import ScheduleMaintenanceAgent
from agent.agents.shared.base_agent import SUMMARY_PREFIX, BaseNovaAgent
from agent.core.agent_state import AgentState
from agent.services.athlete_facts import set_preference

PRUNE_ROUNDS = 5


class _FakeCompaction:
    def __init__(self) -> None:
        self.calls = 0

    def get_summary(self) -> str:
        self.calls += 1
        return f"summary number {self.calls}"


class _FakeSpeechHandle:
    async def wait_for_playout(self) -> None:
        return None


class _FakeSession:
    def __init__(self) -> None:
        self.replies: list[dict] = []
        self.input = SimpleNamespace(set_audio_enabled=lambda enabled: None)

    def generate_reply(self, **kwargs) -> _FakeSpeechHandle:
        self.replies.append(kwargs)
        return _FakeSpeechHandle()


@pytest.fixture
def state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AgentState:
    monkeypatch.setattr(AgentState, "_load_user_from_database", lambda self, user_id: None)
    monkeypatch.setattr(BaseNovaAgent, "_log_function_call", lambda self, *args, **kwargs: None)
    agent_state = AgentState(state_dir=tmp_path)
    agent_state.state["user"]["id"] = "athlete-1"
    return agent_state


def _userdata(state: AgentState, compaction: object | None = None) -> SimpleNamespace:
    return SimpleNamespace(state=state, compaction_service=compaction, visual_bridge=None, affect_service=None)


def _add_turns(agent: BaseNovaAgent, count: int, tag: str) -> None:
    for i in range(count):
        agent._chat_ctx.add_message(role="user", content=f"{tag} user {i}")
        agent._chat_ctx.add_message(role="assistant", content=f"{tag} nova {i}")


def _summary_items(ctx: llm.ChatContext) -> list:
    return [
        item for item in ctx.items
        if getattr(item, "role", None) == "system" and (item.text_content or "").startswith(SUMMARY_PREFIX)
    ]


def _attach_session(agent: BaseNovaAgent, session: _FakeSession) -> None:
    agent._activity = SimpleNamespace(session=session)


class TestSummaryPrune:
    def test_repeated_prunes_keep_exactly_one_summary(self, state: AgentState) -> None:
        agent = BaseNovaAgent(state=state, userdata=_userdata(state, _FakeCompaction()), instructions="test")

        async def _run() -> None:
            for round_number in range(PRUNE_ROUNDS):
                _add_turns(agent, 5, f"round{round_number}")
                await agent._truncate_context_for_handoff(max_items=4)

        asyncio.run(_run())

        summaries = _summary_items(agent.chat_ctx)
        assert len(summaries) == 1
        assert f"summary number {PRUNE_ROUNDS}" in summaries[0].text_content

    def test_prune_returns_context_for_next_agent(self, state: AgentState) -> None:
        agent = BaseNovaAgent(state=state, userdata=_userdata(state, _FakeCompaction()), instructions="test")
        _add_turns(agent, 5, "talk")

        handed_over = asyncio.run(agent._truncate_context_for_handoff(max_items=4))

        texts = [item.text_content for item in handed_over.items if item.type == "message"]
        assert any(text.startswith(SUMMARY_PREFIX) for text in texts)
        assert "talk nova 4" in texts

    def test_short_context_is_handed_over_unchanged(self, state: AgentState) -> None:
        agent = BaseNovaAgent(state=state, userdata=_userdata(state), instructions="test")
        _add_turns(agent, 1, "short")

        handed_over = asyncio.run(agent._truncate_context_for_handoff())

        texts = [item.text_content for item in handed_over.items if item.type == "message"]
        assert texts == ["short user 0", "short nova 0"]


class TestHandoffCarriesConversation:
    def test_schedule_back_to_main_menu_keeps_conversation(self, state: AgentState) -> None:
        schedule = ScheduleMaintenanceAgent(state=state, userdata=_userdata(state))
        schedule._chat_ctx.add_message(role="user", content="skip leg day")
        schedule._chat_ctx.add_message(role="assistant", content="Done, leg day is skipped.")

        main_menu = asyncio.run(schedule.back_to_main_menu(None))

        assert isinstance(main_menu, MainMenuAgent)
        texts = [item.text_content for item in main_menu.chat_ctx.items if item.type == "message"]
        assert "skip leg day" in texts
        assert "Done, leg day is skipped." in texts

    def test_main_menu_manage_schedule_keeps_conversation(self, state: AgentState) -> None:
        main_menu = MainMenuAgent(state=state, userdata=_userdata(state))
        main_menu._chat_ctx.add_message(role="user", content="move my workout to friday")

        schedule = asyncio.run(main_menu.manage_schedule(None, "move my workout to friday"))

        assert isinstance(schedule, ScheduleMaintenanceAgent)
        texts = [item.text_content for item in schedule.chat_ctx.items if item.type == "message"]
        assert "move my workout to friday" in texts


class TestEntrySpeech:
    def test_schedule_agent_speaks_on_entry(self, state: AgentState) -> None:
        state.set("schedule.precaptured_request", "skip today")
        schedule = ScheduleMaintenanceAgent(state=state, userdata=_userdata(state))
        session = _FakeSession()
        _attach_session(schedule, session)

        asyncio.run(schedule.on_enter())

        assert len(session.replies) == 1

    def test_main_menu_reentry_after_workout_speaks_what_next(self, state: AgentState) -> None:
        state.set("session.main_menu_greeted", True)
        state.state["user"]["first_time_main_menu"] = False
        state.switch_mode("workout")
        state.switch_mode("main_menu")
        main_menu = MainMenuAgent(state=state, userdata=_userdata(state))
        spoken: list[str] = []

        async def _fake_say(instructions: str, wait: bool = True, restore: bool = True) -> None:
            spoken.append(instructions)

        main_menu._say = _fake_say
        asyncio.run(main_menu.on_enter())

        assert len(spoken) == 1
        assert "workout" in spoken[0]
        assert "goodbye" in spoken[0].lower()


class TestAthleteFactsInInstructions:
    def test_preferences_reach_agent_instructions(self, state: AgentState) -> None:
        set_preference(state, "humor", "none, user disliked jokes")
        main_menu = MainMenuAgent(state=state, userdata=_userdata(state))
        assert "humor: none, user disliked jokes" in main_menu.instructions

    def test_no_facts_line_without_facts(self, state: AgentState) -> None:
        main_menu = MainMenuAgent(state=state, userdata=_userdata(state))
        assert "ATHLETE FACTS" not in main_menu.instructions

    def test_refresh_picks_up_a_fact_added_mid_session(self, state: AgentState) -> None:
        main_menu = MainMenuAgent(state=state, userdata=_userdata(state))
        set_preference(state, "cue_style", "short cues only")

        asyncio.run(main_menu._refresh_athlete_facts())

        assert "cue_style: short cues only" in main_menu.instructions
        assert "# Main Menu" in main_menu.instructions
