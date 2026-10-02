"""Tests for WorkoutAgent: wake word gating, pain and effort tools, live state, form checks."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from livekit.agents import StopResponse, llm
from livekit.agents.voice.events import SpeechCreatedEvent
from livekit.agents.voice.speech_handle import SpeechHandle

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.agents import workout_agent as workout_agent_module
from agent.agents.prompts import get_workout_prompt
from agent.agents.workout_agent import WORKOUT_STATE_ITEM_ID, WorkoutAgent, _effort_to_rpe
from agent.core.agent_state import AgentState
from agent.services.athlete_facts import active_pain_flags, add_pain_flag

WAKE_WORD_FRAME_SAMPLES = 1280
WAKE_WORD_FRAMES_TO_DETECT = 30
SAVED_MIN_DELAY_S = 0.25


class _FakeSession:
    def __init__(self) -> None:
        self.handlers: dict[str, set] = {}
        self.replies: list[dict] = []
        self.agent_state = "listening"
        self.user_state = "listening"
        self.current_speech = None
        self.input = SimpleNamespace(set_audio_enabled=lambda enabled: None)
        self.options = SimpleNamespace(turn_handling={
            "endpointing": {"min_delay": SAVED_MIN_DELAY_S},
            "preemptive_generation": {"enabled": False},
        })

    def on(self, event: str, handler) -> None:
        self.handlers.setdefault(event, set()).add(handler)

    def off(self, event: str, handler) -> None:
        self.handlers.get(event, set()).discard(handler)

    def clear_user_turn(self) -> None:
        pass

    async def generate_reply(self, **kwargs) -> None:
        self.replies.append(kwargs)


@pytest.fixture
def state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AgentState:
    monkeypatch.setattr(AgentState, "_load_user_from_database", lambda self, user_id: None)
    agent_state = AgentState(state_dir=tmp_path)
    agent_state.state["user"]["id"] = "athlete-1"
    return agent_state


def _make_agent(state: AgentState, coaching=None) -> tuple[WorkoutAgent, _FakeSession]:
    room = SimpleNamespace(remote_participants={}, on=lambda *args: None, off=lambda *args: None)
    userdata = SimpleNamespace(
        state=state,
        coaching_service=coaching,
        visual_bridge=None,
        affect_service=None,
        compaction_service=None,
        room=room,
        wakeword_model=object(),
    )
    agent = WorkoutAgent(state=state, userdata=userdata)
    session = _FakeSession()
    agent._activity = SimpleNamespace(session=session)
    agent._ww_session = session
    return agent, session


def _make_dormant_agent(state: AgentState, coaching=None) -> tuple[WorkoutAgent, _FakeSession]:
    agent, session = _make_agent(state, coaching)
    agent._wake_word_active = True
    agent._wake_word_listening = False
    return agent, session


def _speech_created(source: str) -> tuple[SpeechHandle, SpeechCreatedEvent]:
    handle = SpeechHandle.create(allow_interruptions=False)
    return handle, SpeechCreatedEvent(speech_handle=handle, user_initiated=True, source=source)


def _user_message(text: str = "how many sets are left") -> llm.ChatMessage:
    return llm.ChatMessage(role="user", content=[text])


def _coaching(**overrides) -> MagicMock:
    coaching = MagicMock()
    coaching.is_coaching_speaking = False
    coaching.workout_state_line.return_value = "WORKOUT — squat set 2 of 3, rep 4 of 8"
    coaching.get_current_form_snapshot.return_value = None
    coaching.last_rep_verdict.return_value = None
    for name, value in overrides.items():
        setattr(coaching, name, value)
    return coaching


def _snapshot(rep_phase: str = "descent") -> dict:
    return {
        # 60 degrees of lean from vertical: the old fixed 35-degree check flagged this.
        "angles": {"rep_phase": rep_phase, "trunk_flexion": 120.0, "knee_flexion_l": 90.0, "knee_flexion_r": 120.0},
        "data_age_ms": 100.0,
        "last_cue": None,
    }


def _verdict(faults: list[dict]) -> dict:
    return {"rep_number": 4, "faults": faults}


async def _silent_frames(count: int):
    for _ in range(count):
        yield np.zeros(WAKE_WORD_FRAME_SAMPLES, dtype=np.int16)


class _AlwaysHeyNova:
    def predict(self, window: np.ndarray) -> dict[str, float]:
        return {"hey_nova": 0.95}


class TestConsoleWakeWordMic:
    def test_console_job_uses_local_mic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("WAKE_WORD_LOCAL_MIC", raising=False)
        console_job = SimpleNamespace(is_fake_job=lambda: True)
        monkeypatch.setattr(workout_agent_module, "get_job_context", lambda required: console_job)
        assert workout_agent_module._wake_word_uses_local_mic() is True

    def test_room_job_uses_room_track(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("WAKE_WORD_LOCAL_MIC", raising=False)
        room_job = SimpleNamespace(is_fake_job=lambda: False)
        monkeypatch.setattr(workout_agent_module, "get_job_context", lambda required: room_job)
        assert workout_agent_module._wake_word_uses_local_mic() is False

    def test_no_job_context_uses_room_track(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("WAKE_WORD_LOCAL_MIC", raising=False)
        monkeypatch.setattr(workout_agent_module, "get_job_context", lambda required: None)
        assert workout_agent_module._wake_word_uses_local_mic() is False

    def test_env_override_beats_console_detection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        console_job = SimpleNamespace(is_fake_job=lambda: True)
        monkeypatch.setattr(workout_agent_module, "get_job_context", lambda required: console_job)
        monkeypatch.setenv("WAKE_WORD_LOCAL_MIC", "0")
        assert workout_agent_module._wake_word_uses_local_mic() is False
        monkeypatch.setattr(workout_agent_module, "get_job_context", lambda required: None)
        monkeypatch.setenv("WAKE_WORD_LOCAL_MIC", "1")
        assert workout_agent_module._wake_word_uses_local_mic() is True


class TestDormantModeFilter:
    def test_cached_cue_plays_in_dormant_mode(self, state: AgentState, caplog: pytest.LogCaptureFixture) -> None:
        async def _run() -> None:
            agent, _ = _make_dormant_agent(state)
            cue, event = _speech_created("say")
            agent._on_speech_created_for_wake_word(event)
            assert not cue.interrupted

        asyncio.run(_run())
        assert not [record for record in caplog.records if record.levelno >= logging.ERROR]

    def test_coaching_llm_speech_plays_in_dormant_mode(self, state: AgentState, caplog: pytest.LogCaptureFixture) -> None:
        async def _run() -> None:
            agent, _ = _make_dormant_agent(state)
            recap, event = _speech_created("generate_reply")
            agent._on_speech_created_for_wake_word(event)
            assert not recap.interrupted

        asyncio.run(_run())
        assert not [record for record in caplog.records if record.levelno >= logging.ERROR]

    def test_turn_reply_suppressed_in_dormant_mode(self, state: AgentState) -> None:
        agent, _ = _make_dormant_agent(state)
        with pytest.raises(StopResponse):
            asyncio.run(agent.on_user_turn_completed(llm.ChatContext.empty(), new_message=_user_message()))

    def test_turn_reply_allowed_while_listening(self, state: AgentState) -> None:
        agent, _ = _make_dormant_agent(state)
        agent._wake_word_listening = True
        asyncio.run(agent.on_user_turn_completed(llm.ChatContext.empty(), new_message=_user_message()))


class TestWakeWordInterruptsCoaching:
    def test_wake_word_interrupts_llm_coaching_but_not_cues(self, state: AgentState) -> None:
        async def _run() -> None:
            agent, _ = _make_dormant_agent(state)
            recap, recap_event = _speech_created("generate_reply")
            cue, cue_event = _speech_created("say")
            agent._on_speech_created_for_wake_word(recap_event)
            agent._on_speech_created_for_wake_word(cue_event)
            # LiveKit marks an interrupted speech done once its playout stops.
            asyncio.get_running_loop().call_soon(recap._mark_done)

            await agent._interrupt_llm_speech()

            assert recap.interrupted
            assert not cue.interrupted

        asyncio.run(_run())

    def test_wake_word_listens_even_while_coaching_speaks(self, state: AgentState) -> None:
        async def _run() -> None:
            agent, session = _make_dormant_agent(state, _coaching(is_coaching_speaking=True))
            agent._restart_wake_word_timeout = MagicMock()
            await agent._activate_listening_mode()
            assert agent._wake_word_listening is True
            assert len(session.replies) == 1

        asyncio.run(_run())

    def test_detection_runs_while_agent_is_speaking(self, state: AgentState) -> None:
        async def _run() -> None:
            agent, session = _make_dormant_agent(state, _coaching(is_coaching_speaking=True))
            session.agent_state = "speaking"
            agent._ww_model = _AlwaysHeyNova()
            agent._activate_listening_mode = AsyncMock()

            await agent._wake_word_detection_loop(_silent_frames(WAKE_WORD_FRAMES_TO_DETECT))
            await asyncio.sleep(0)

            agent._activate_listening_mode.assert_awaited_once()

        asyncio.run(_run())


class TestWakeWordSystemHygiene:
    def test_stop_unregisters_every_session_handler(self, state: AgentState, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WAKE_WORD_LOCAL_MIC", "0")

        async def _run() -> None:
            agent, session = _make_agent(state)
            await agent._start_wake_word_system()
            assert any(session.handlers.values())
            await agent._stop_wake_word_system()
            assert not any(session.handlers.values())

        asyncio.run(_run())

    def test_stop_restores_turn_options_in_effect_before(self, state: AgentState, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WAKE_WORD_LOCAL_MIC", "0")

        async def _run() -> None:
            agent, session = _make_agent(state)
            await agent._start_wake_word_system()
            await agent._stop_wake_word_system()
            assert session.options.turn_handling["preemptive_generation"]["enabled"] is False

        asyncio.run(_run())

    def test_stop_cancels_a_pending_acknowledgment(self, state: AgentState, monkeypatch: pytest.MonkeyPatch) -> None:
        """A "Hey Nova" during the final recap must not speak after the hand-off."""
        monkeypatch.setenv("WAKE_WORD_LOCAL_MIC", "0")

        async def _run() -> bool:
            agent, _ = _make_agent(state)
            await agent._start_wake_word_system()
            pending = asyncio.Event()

            async def _ack() -> None:
                await pending.wait()

            agent._spawn(_ack())
            task = next(iter(agent._background_tasks))
            await agent._stop_wake_word_system()
            await asyncio.sleep(0)
            return task.cancelled()

        assert asyncio.run(_run()) is True


class TestPainTools:
    def test_flag_pain_records_flag_and_gives_safety_guidance(self, state: AgentState) -> None:
        agent, _ = _make_agent(state)
        _, instruction = asyncio.run(agent.flag_pain("left knee", "sharp at the bottom"))
        assert [flag["body_part"] for flag in active_pain_flags(state)] == ["left knee"]
        assert "professional" in instruction
        assert "numbness" in instruction

    def test_clear_pain_resolves_matching_flag(self, state: AgentState) -> None:
        agent, _ = _make_agent(state)
        add_pain_flag(state, "left knee", "sharp")
        asyncio.run(agent.clear_pain("knee"))
        assert active_pain_flags(state) == []

    def test_clear_pain_without_flag_keeps_others(self, state: AgentState) -> None:
        agent, _ = _make_agent(state)
        add_pain_flag(state, "lower back", "tight")
        asyncio.run(agent.clear_pain("shoulder"))
        assert [flag["body_part"] for flag in active_pain_flags(state)] == ["lower back"]



class TestEffortParsing:
    @pytest.mark.parametrize("said,rpe", [
        ("8", 8.0),
        ("eight", 8.0),
        ("ten out of ten", 10.0),
        ("8 out of 10", 8.0),
        ("maybe seven, max eight", 8.0),
        ("about a nine", 9.0),
        ("pretty hard", 8.5),
        ("very hard", 9.0),
        ("easy", 6.0),
    ])
    def test_spoken_ratings(self, said: str, rpe: float) -> None:
        assert _effort_to_rpe(said) == pytest.approx(rpe)

    @pytest.mark.parametrize("said", ["fine", "eleven", "0", ""])
    def test_unclear_answers_get_none(self, said: str) -> None:
        assert _effort_to_rpe(said) is None


class TestWorkoutStateInjection:
    def _workout_items(self, ctx: llm.ChatContext) -> list:
        return [item for item in ctx.items if item.id == WORKOUT_STATE_ITEM_ID]

    def test_turn_gets_one_item_with_state_and_facts(self, state: AgentState) -> None:
        add_pain_flag(state, "left knee", "sharp")
        agent, _ = _make_dormant_agent(state, _coaching())
        agent._wake_word_listening = True
        ctx = llm.ChatContext.empty()
        asyncio.run(agent.on_user_turn_completed(ctx, new_message=_user_message()))

        items = self._workout_items(ctx)
        assert len(items) == 1
        assert items[0].role == "system"
        assert "set 2 of 3" in items[0].text_content
        assert "left knee" in items[0].text_content

    def test_item_is_replaced_in_place_not_appended(self, state: AgentState) -> None:
        coaching = _coaching()
        agent, _ = _make_agent(state, coaching)
        ctx = llm.ChatContext.empty()
        asyncio.run(agent.on_user_turn_completed(ctx, new_message=_user_message()))
        ctx.add_message(role="user", content="and after that?")
        coaching.workout_state_line.return_value = "WORKOUT — squat set 3 of 3"
        asyncio.run(agent.on_user_turn_completed(ctx, new_message=_user_message()))

        items = self._workout_items(ctx)
        assert len(items) == 1
        assert "set 3 of 3" in items[0].text_content
        assert ctx.items.index(items[0]) == 0

    def test_no_item_without_state_or_facts(self, state: AgentState) -> None:
        agent, _ = _make_agent(state, coaching=None)
        ctx = llm.ChatContext.empty()
        asyncio.run(agent.on_user_turn_completed(ctx, new_message=_user_message()))
        assert self._workout_items(ctx) == []


class TestSetEffort:
    @pytest.mark.parametrize(
        ("said", "rpe"),
        [("8", 8.0), ("RPE 7.5", 7.5), ("easy", 6.0), ("medium", 7.0), ("pretty hard", 8.5),
         ("very hard", 9.0), ("max", 10.0), ("all out", 10.0)],
    )
    def test_effort_is_recorded_as_rpe(self, state: AgentState, said: str, rpe: float) -> None:
        coaching = _coaching()
        agent, _ = _make_agent(state, coaching)
        asyncio.run(agent.log_set_effort(said))
        coaching.record_set_rpe.assert_called_once_with(pytest.approx(rpe))

    @pytest.mark.parametrize("said", ["banana", "12", "0"])
    def test_unclear_effort_asks_for_a_rating(self, state: AgentState, said: str) -> None:
        coaching = _coaching()
        agent, _ = _make_agent(state, coaching)
        _, instruction = asyncio.run(agent.log_set_effort(said))
        coaching.record_set_rpe.assert_not_called()
        assert "one to ten" in instruction


class TestCheckMyForm:
    def test_lean_is_never_judged_from_a_single_frame(self, state: AgentState) -> None:
        coaching = _coaching()
        coaching.get_current_form_snapshot.return_value = _snapshot()
        coaching.last_rep_verdict.return_value = _verdict([])
        agent, _ = _make_agent(state, coaching)
        _, instruction = asyncio.run(agent.check_my_form())
        assert "chest" not in instruction
        assert "one side is bending" not in instruction
        assert "nothing to correct" in instruction

    def test_reports_observable_faults_and_skips_approximate_ones(self, state: AgentState) -> None:
        coaching = _coaching()
        coaching.get_current_form_snapshot.return_value = _snapshot()
        coaching.last_rep_verdict.return_value = _verdict([
            {"fault_type": "hip_shoot", "side": None, "observability": "approximate"},
            {"fault_type": "knee_valgus", "side": "left", "observability": "observable"},
        ])
        agent, _ = _make_agent(state, coaching)
        _, instruction = asyncio.run(agent.check_my_form())
        assert "knees caving in" in instruction
        assert "left" in instruction
        assert "hips rising" not in instruction

    def test_standing_setup_and_last_rep_are_both_reported(self, state: AgentState) -> None:
        coaching = _coaching()
        snapshot = _snapshot(rep_phase="idle")
        snapshot["angles"].update({"stance_width_ratio": 1.4, "target_stance_ratio": 1.4})
        coaching.get_current_form_snapshot.return_value = snapshot
        coaching.last_rep_verdict.return_value = _verdict([])
        agent, _ = _make_agent(state, coaching)
        _, instruction = asyncio.run(agent.check_my_form())
        assert "stance width is right" in instruction
        assert "nothing to correct" in instruction

    def test_no_rep_yet_asks_for_one(self, state: AgentState) -> None:
        coaching = _coaching()
        coaching.get_current_form_snapshot.return_value = _snapshot()
        agent, _ = _make_agent(state, coaching)
        _, instruction = asyncio.run(agent.check_my_form())
        assert "rep" in instruction
        assert "no verdict" in instruction


class TestEndWorkout:
    def test_end_workout_says_goodbye_before_main_menu(self, state: AgentState) -> None:
        from agent.agents.main_menu_agent import MainMenuAgent

        agent, _ = _make_agent(state)
        agent._cleanup_workout = AsyncMock()
        agent._suppress_turn_detection = AsyncMock()
        agent._truncate_context_for_handoff = AsyncMock(return_value=llm.ChatContext.empty())
        next_agent, instruction = asyncio.run(agent.end_workout(None))
        assert isinstance(next_agent, MainMenuAgent)
        assert "goodbye" in instruction.lower()


class TestWorkoutPrompt:
    def test_register_is_calm_direct(self) -> None:
        prompt = get_workout_prompt()
        assert "HIGH energy" not in prompt
        assert "Calm and direct" in prompt

    def test_no_verbatim_spoken_example_lines(self) -> None:
        prompt = get_workout_prompt()
        for parroted in ("Nice set!", "Niiice", "Done with the set or done for today?", "safety first"):
            assert parroted not in prompt

    def test_pain_is_flagged_not_routed_to_skip(self) -> None:
        prompt = get_workout_prompt()
        skip_section = prompt.split("## skip_exercise")[1].split("## get_next_exercise")[0]
        assert "hurts\" -> use skip_exercise" not in skip_section
        assert "flag_pain" in skip_section
        assert "medical professional" in prompt
        assert "numbness" in prompt

    def test_never_claims_what_cameras_cannot_see(self) -> None:
        prompt = get_workout_prompt()
        assert "explain_squat" in prompt
        assert "butt wink" in prompt

    def test_wake_word_ack_has_no_scripted_lines(self, state: AgentState) -> None:
        async def _run() -> None:
            agent, session = _make_dormant_agent(state)
            agent._restart_wake_word_timeout = MagicMock()
            await agent._activate_listening_mode()
            assert "Yeah?" not in session.replies[0]["instructions"]

        asyncio.run(_run())


class TestEndSetEarly:
    @pytest.mark.parametrize("status", ["advanced", "workout_complete"])
    def test_queued_recap_speaks_so_the_tool_stays_silent(self, state: AgentState, status: str) -> None:
        coaching = _coaching(is_resting=False)
        coaching.force_end_current_set = AsyncMock(return_value={"status": status, "recap_queued": True})
        agent, _ = _make_agent(state, coaching)
        assert asyncio.run(agent.end_set_early(5)) is None

    def test_set_without_recap_is_confirmed(self, state: AgentState) -> None:
        coaching = _coaching(is_resting=False)
        coaching.force_end_current_set = AsyncMock(return_value={"status": "no_session"})
        agent, _ = _make_agent(state, coaching)
        _, instruction = asyncio.run(agent.end_set_early(5))
        assert "end_workout" not in instruction
