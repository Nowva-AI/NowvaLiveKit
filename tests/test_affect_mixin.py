"""Tests for AffectNodesMixin: idempotent state injection, untouched originals, user_prefix mode, TTS styling."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from livekit.agents import Agent, llm  # noqa: E402

from affect.config import AffectConfig  # noqa: E402
from affect.state import AthleteState  # noqa: E402
from affect.tts_adapters import CartesiaInlineTagAdapter  # noqa: E402
from affect.voice_style import VoiceStyle  # noqa: E402
from agent.agents.shared.affect_mixin import STATE_ITEM_ID, AffectNodesMixin, inject_state  # noqa: E402


class _FakeService:
    def __init__(self, state: AthleteState, inject_as: str = "system") -> None:
        self.enabled = True
        self.config = AffectConfig()
        self.config.trigger.inject_as = inject_as
        self._state = state
        self.style_adapter = CartesiaInlineTagAdapter(self.config.style)
        self.prepared: list = []
        self.waits: list[tuple[float, bool]] = []

    def speech_kind(self) -> str:
        return "conversation"

    def wait_budget_s(self, kind: str) -> float:
        return 0.0

    async def snapshot(self, max_wait_s: float = 0.0) -> AthleteState:
        return self._state

    def record_llm_wait(self, wait_ms: float, fresh: bool) -> None:
        self.waits.append((wait_ms, fresh))

    def current_style(self, kind: str) -> VoiceStyle:
        return VoiceStyle(energy=-1, pace=-1, warmth=1)


class _Agent(AffectNodesMixin, Agent):
    def __init__(self, service) -> None:
        super().__init__(instructions="base")
        self.userdata = SimpleNamespace(affect_service=service)


def _ctx() -> llm.ChatContext:
    ctx = llm.ChatContext.empty()
    ctx.add_message(role="system", content="You are Nova.")
    ctx.add_message(role="user", content="How did that set look?")
    return ctx


def _run_llm(agent: _Agent, ctx: llm.ChatContext, captured: list, monkeypatch) -> None:
    async def _fake_llm_node(agent_, chat_ctx, tools, model_settings):
        captured.append(chat_ctx)
        if False:
            yield None

    monkeypatch.setattr(Agent.default, "llm_node", staticmethod(_fake_llm_node))

    async def _drive() -> None:
        async for _ in agent.llm_node(ctx, [], None):
            pass

    asyncio.run(_drive())


class TestInjection:
    def test_one_state_item_even_when_called_twice(self, monkeypatch) -> None:
        state = AthleteState(effort="working", affect="strained", confident=True)
        agent = _Agent(_FakeService(state))
        ctx = _ctx()
        captured: list = []
        _run_llm(agent, ctx, captured, monkeypatch)
        first = captured[0]
        state_items = [i for i in first.items if getattr(i, "id", None) == STATE_ITEM_ID]
        assert len(state_items) == 1
        assert state_items[0].text_content == "[athlete: effort=working affect=strained]"
        _run_llm(agent, first, captured, monkeypatch)
        second = captured[1]
        assert len([i for i in second.items if getattr(i, "id", None) == STATE_ITEM_ID]) == 1

    def test_original_context_untouched(self, monkeypatch) -> None:
        state = AthleteState(effort="working", affect="strained", confident=True)
        agent = _Agent(_FakeService(state))
        ctx = _ctx()
        before = [(i.id, i.text_content) for i in ctx.items]
        captured: list = []
        _run_llm(agent, ctx, captured, monkeypatch)
        assert [(i.id, i.text_content) for i in ctx.items] == before
        assert captured[0] is not ctx

    def test_no_line_when_not_confident_or_default(self, monkeypatch) -> None:
        for state in (AthleteState(effort="working", affect="strained", confident=False), AthleteState(confident=True)):
            agent = _Agent(_FakeService(state))
            captured: list = []
            _run_llm(agent, _ctx(), captured, monkeypatch)
            assert not [i for i in captured[0].items if getattr(i, "id", None) == STATE_ITEM_ID]
            assert len(captured[0].items) == 2

    def test_user_prefix_mode(self, monkeypatch) -> None:
        state = AthleteState(effort="near_limit", affect="frustrated", confident=True)
        agent = _Agent(_FakeService(state, inject_as="user_prefix"))
        ctx = _ctx()
        captured: list = []
        _run_llm(agent, ctx, captured, monkeypatch)
        injected = captured[0]
        user_items = [i for i in injected.items if getattr(i, "role", None) == "user"]
        assert len(user_items) == 1
        assert user_items[0].text_content.startswith("[athlete: effort=near_limit affect=frustrated]\nHow did")
        assert ctx.items[-1].text_content == "How did that set look?"
        _run_llm(agent, injected, captured, monkeypatch)
        again = [i for i in captured[1].items if getattr(i, "role", None) == "user"][0]
        assert again.text_content.count("[athlete:") == 1

    def test_inject_state_helper_removes_stale_item(self) -> None:
        ctx = _ctx()
        ctx.items.append(llm.ChatMessage(id=STATE_ITEM_ID, role="system", content=["[athlete: effort=fresh affect=engaged]"]))
        out = inject_state(ctx, AthleteState(confident=False), "system")
        assert not [i for i in out.items if getattr(i, "id", None) == STATE_ITEM_ID]
        assert len(ctx.items) == 3

    def test_record_wait_called(self, monkeypatch) -> None:
        service = _FakeService(AthleteState(confident=True))
        agent = _Agent(service)
        _run_llm(agent, _ctx(), [], monkeypatch)
        assert len(service.waits) == 1


class TestTTS:
    def test_style_prefix_and_tag_stripping(self, monkeypatch) -> None:
        service = _FakeService(AthleteState(affect="frustrated", confident=True))
        agent = _Agent(service)
        collected: list[str] = []

        async def _fake_tts_node(agent_, text, model_settings):
            async for chunk in text:
                collected.append(chunk)
            if False:
                yield None

        monkeypatch.setattr(Agent.default, "tts_node", staticmethod(_fake_tts_node))

        async def _text():
            yield '<emotion value="angry"/>Rack it. '
            yield "You are done."

        async def _drive() -> None:
            async for _ in agent.tts_node(_text(), None):
                pass

        asyncio.run(_drive())
        assert collected[0].startswith('<emotion value="sympathetic"/><speed ratio="0.9"/>')
        assert "angry" not in "".join(collected)
        assert collected[-1] == "You are done."

    def test_no_service_falls_back_to_normalizer(self, monkeypatch) -> None:
        agent = _Agent(None)
        collected: list[str] = []

        async def _fake_tts_node(agent_, text, model_settings):
            async for chunk in text:
                collected.append(chunk)
            if False:
                yield None

        monkeypatch.setattr(Agent.default, "tts_node", staticmethod(_fake_tts_node))

        async def _text():
            yield "toward 22° today"

        async def _drive() -> None:
            async for _ in agent.tts_node(_text(), None):
                pass

        asyncio.run(_drive())
        assert collected == ["toward 22 degrees today"]


class TestHowDoISoundTool:
    def test_tool_registered_on_agents(self) -> None:
        agent = _Agent(None)
        names = [getattr(getattr(t, "info", None), "name", None) for t in agent.tools]
        assert "how_do_i_sound" in names

    def test_tool_without_service(self) -> None:
        agent = _Agent(None)
        text = asyncio.run(agent.how_do_i_sound(None))
        assert "off" in text

    def test_tool_delegates_to_service(self) -> None:
        class _Service(_FakeService):
            def describe_for_llm(self) -> str:
                return "Arousal (energy) low, about their usual."

        agent = _Agent(_Service(AthleteState(confident=True)))
        assert "Arousal" in asyncio.run(agent.how_do_i_sound(None))
