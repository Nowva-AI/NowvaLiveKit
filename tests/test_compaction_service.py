"""Tests for CompactionService session folder creation and final flush on stop."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.core.agent_state import AgentState
from agent.services.athlete_facts import athlete_facts_line
from agent.services.compaction_service import HOT_WINDOW_SECONDS, CompactionService


class _FakeSession:
    def __init__(self) -> None:
        self.handlers: dict[str, object] = {}

    def on(self, event_name: str, handler) -> None:
        self.handlers[event_name] = handler

    def off(self, event_name: str, handler) -> None:
        self.handlers.pop(event_name, None)


class _FakeState:
    def get_mode(self) -> str:
        return "main_menu"


class _FakeOpenAIClient:
    """Returns a fixed compression result and records calls."""

    def __init__(self, response_text: str = "COMPRESSED_SUMMARY") -> None:
        self.calls: list[dict] = []
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=response_text))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )

        async def _create(**kwargs):
            self.calls.append(kwargs)
            return response

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=_create))


def _make_service(tmp_path: Path, monkeypatch) -> tuple[CompactionService, _FakeOpenAIClient]:
    monkeypatch.setenv("NOWVA_SESSION_OUTPUT_DIR", str(tmp_path))
    client = _FakeOpenAIClient()
    service = CompactionService(
        session=_FakeSession(),
        state=_FakeState(),
        user_id="test_user",
        openai_client=client,
    )
    return service, client


def _conversation_event(role: str, text: str, transcript_confidence: float | None = None):
    # Real user turns carry an STT confidence; assistant turns and synthetic
    # generate_reply(user_input=...) prompts carry None.
    if role == "user" and transcript_confidence is None:
        transcript_confidence = 0.9
    return SimpleNamespace(
        item=SimpleNamespace(role=role, text_content=text, transcript_confidence=transcript_confidence)
    )


def _synthetic_user_event(text: str):
    return SimpleNamespace(item=SimpleNamespace(role="user", text_content=text, transcript_confidence=None))


def _make_service_with_state(
    tmp_path: Path, monkeypatch, response_text: str
) -> tuple[CompactionService, _FakeOpenAIClient, AgentState]:
    monkeypatch.setenv("NOWVA_SESSION_OUTPUT_DIR", str(tmp_path / "run"))
    monkeypatch.setattr(AgentState, "_load_user_from_database", lambda self, user_id: None)
    state = AgentState(state_dir=tmp_path)
    state.state["user"]["id"] = "athlete-1"
    client = _FakeOpenAIClient(response_text)
    service = CompactionService(
        session=_FakeSession(), state=state, user_id="athlete-1", openai_client=client,
    )
    return service, client, state


def _age_out_buffer(service: CompactionService) -> None:
    for event in service._event_buffer:
        event["timestamp"] -= HOT_WINDOW_SECONDS + 1


class TestSessionFolder:
    def test_creates_compaction_dir_in_session_output(self, tmp_path, monkeypatch):
        service, _ = _make_service(tmp_path, monkeypatch)

        async def _run():
            await service.start()
            await service.stop()

        asyncio.run(_run())

        compaction_dir = tmp_path / "compaction"
        assert compaction_dir.is_dir()
        assert (compaction_dir / "memory.md").exists()
        assert (compaction_dir / "session_meta.json").exists()

    def test_final_metadata_written_on_stop(self, tmp_path, monkeypatch):
        service, _ = _make_service(tmp_path, monkeypatch)

        async def _run():
            await service.start()
            await service.stop()

        asyncio.run(_run())

        meta = json.loads((tmp_path / "compaction" / "session_meta.json").read_text())
        assert meta["user_id"] == "test_user"
        assert "session_end" in meta


class TestFinalFlush:
    def test_stop_flushes_buffered_events_to_memory(self, tmp_path, monkeypatch):
        service, client = _make_service(tmp_path, monkeypatch)

        async def _run():
            await service.start()
            handler = service._session.handlers["conversation_item_added"]
            handler(_conversation_event("user", "I squatted 100 kg for 5 reps"))
            handler(_conversation_event("assistant", "Great depth on that set"))
            await service.stop()

        asyncio.run(_run())

        memory = (tmp_path / "compaction" / "memory.md").read_text()
        assert "Session end flush" in memory
        assert "COMPRESSED_SUMMARY" in memory
        assert len(client.calls) == 1

    def test_stop_flushes_raw_when_llm_fails(self, tmp_path, monkeypatch):
        service, client = _make_service(tmp_path, monkeypatch)

        async def _failing_create(**kwargs):
            raise RuntimeError("API down")

        client.chat.completions.create = _failing_create

        async def _run():
            await service.start()
            handler = service._session.handlers["conversation_item_added"]
            handler(_conversation_event("user", "my knee hurts on rep three"))
            await service.stop()

        asyncio.run(_run())

        memory = (tmp_path / "compaction" / "memory.md").read_text()
        assert "Session end flush" in memory
        assert "my knee hurts on rep three" in memory

    def test_stop_without_events_writes_no_flush_section(self, tmp_path, monkeypatch):
        service, _ = _make_service(tmp_path, monkeypatch)

        async def _run():
            await service.start()
            await service.stop()

        asyncio.run(_run())

        memory = (tmp_path / "compaction" / "memory.md").read_text()
        assert "Session end flush" not in memory


class TestSyntheticPromptsFiltered:
    def test_synthetic_user_input_not_buffered(self, tmp_path, monkeypatch):
        service, _ = _make_service(tmp_path, monkeypatch)
        service._handle_conversation_item(_synthetic_user_event("Toes are turned out too far"))
        assert service.get_event_buffer_size() == 0

    def test_real_user_speech_and_nova_buffered(self, tmp_path, monkeypatch):
        service, _ = _make_service(tmp_path, monkeypatch)
        service._handle_conversation_item(_conversation_event("user", "that felt heavy"))
        service._handle_conversation_item(_conversation_event("assistant", "Good depth though"))
        assert service.get_event_buffer_size() == 2


class TestDurablePreferences:
    def test_stated_preference_reaches_athlete_facts(self, tmp_path, monkeypatch):
        service, _, state = _make_service_with_state(
            tmp_path, monkeypatch, "User asked for no jokes.\nPREFERENCE: humor = none, dislikes jokes",
        )

        async def _run():
            service._handle_conversation_item(_conversation_event("user", "please stop with the jokes"))
            _age_out_buffer(service)
            await service._run_compaction_cycle()

        asyncio.run(_run())

        assert "humor: none, dislikes jokes" in athlete_facts_line(state)

    def test_preference_said_at_session_end_is_kept(self, tmp_path, monkeypatch):
        service, _, state = _make_service_with_state(
            tmp_path, monkeypatch, "PREFERENCE: talk_amount = fewer words between sets",
        )

        async def _run():
            await service.start()
            service._handle_conversation_item(_conversation_event("user", "talk less between sets"))
            await service.stop()

        asyncio.run(_run())

        assert "talk_amount: fewer words between sets" in athlete_facts_line(state)

    def test_unknown_topic_is_ignored(self, tmp_path, monkeypatch):
        service, _, state = _make_service_with_state(
            tmp_path, monkeypatch, "PREFERENCE: favourite_colour = blue",
        )

        async def _run():
            service._handle_conversation_item(_conversation_event("user", "I like blue"))
            _age_out_buffer(service)
            await service._run_compaction_cycle()

        asyncio.run(_run())

        assert athlete_facts_line(state) is None


class TestColdFlushStaysReadable:
    def test_summary_carries_flushed_memory_not_a_file_path(self, tmp_path, monkeypatch):
        service, _ = _make_service(tmp_path, monkeypatch)
        service._client = _FakeOpenAIClient("Squatted 3x5 at 100 kg; left knee ached on set two.")

        async def _run():
            await service.start()
            service._cold = "older facts " * 400
            await service._flush_cold_to_memory()
            summary = service.get_summary()
            await service.stop()
            return summary

        summary = asyncio.run(_run())

        assert "left knee ached on set two" in summary
        assert "memory.md" not in summary
        assert str(tmp_path) not in summary
