"""Agent-side deadlift session: the coaching-ready gate, session metadata on the quick and
scheduled paths, the first-session briefing, explain_deadlift, the deadlift form check,
the start_capture / set_exercise / display messages and the DB records
(docs/deadlift/PLAN.md §3.4, §4.4-§4.6; .claude/deadlift/CONTRACT.md).
"""

from __future__ import annotations

import ast
import asyncio
import json
import math
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Callable
from unittest.mock import MagicMock

import pytest
from livekit.agents.llm.utils import build_legacy_openai_schema

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import agent.agents.main_menu_agent as main_menu_module
import agent.agents.quickExerciseAgent as quick_exercise_module
import agent.agents.workout_agent as workout_agent_module
from agent.agents.deadlift_setup_task import DeadliftSetupTask
from agent.agents.main_menu_agent import MainMenuAgent
from agent.agents.prompts import get_main_menu_prompt, get_workout_prompt
from agent.agents.prompts.main_menu_prompt import coachable_exercises_text
from agent.agents.prompts.workout_prompt import DEADLIFT_SECTION
from agent.agents.quickExerciseAgent import (
    CollectDeadliftInfoTask,
    CollectExerciseInfoTask,
    build_task_instructions,
)
from agent.agents.shared.base_agent import BaseNovaAgent
from agent.agents.shared.deadlift_card import (
    DEADLIFT_CARD,
    NEVER_VISIBLE,
    OBSERVABILITY_TOPIC,
    TOPIC_KEYWORDS,
    explain_deadlift,
    explain_deadlift_topic,
)
from agent.agents.shared.deadlift_session import (
    DEFAULT_PLATE_DIAMETER_CM,
    EXERCISE_META_STATE_KEY,
    SETUP_QUESTION,
    SETUP_STEPS,
    DeadliftSessionMeta,
    deadlift_form_findings,
    exercise_meta_for,
    is_coached_deadlift,
    session_includes_coached_deadlift,
    store_exercise_meta,
)
from agent.agents.workout_agent import WorkoutAgent
from agent.core.agent_state import AgentState
from agent.core.workout_session import WorkoutSession
from biomechanics.faults.observability import SINGLE_CAMERA, TRIANGULATED
from biomechanics.profiles.deadlift import DeadliftProfile
from db.biomechanics_persistence import END_OF_REP_FAULT_TYPES, BiomechanicsRecorder, _build_rep_row

DEADLIFT = "Barbell Deadlift"
PROGRAM_DEADLIFT = "Barbell Conventional Deadlift"
SQUAT = "Barbell Back Squat"
USER_ID = "8d3c5a2e-31a4-4a8f-9a47-0d5f7b2c1e90"
GOLDEN_PATH = Path(__file__).parent / "fixtures" / "squat_agent_golden.json"
MAIN_PY = Path(__file__).parent.parent / "src" / "main.py"
CARD_TOKEN_BUDGET = 1500
CHARS_PER_TOKEN = 4

# Every deadlift fault type in .claude/deadlift/CONTRACT.md §1.
CONTRACT_FAULT_TYPES = (
    "deadlift_bar_position",
    "deadlift_setup_hips",
    "deadlift_shoulders_behind",
    "deadlift_hips_shoot",
    "deadlift_bar_drift",
    "deadlift_lockout",
    "deadlift_lean_back",
    "deadlift_hip_shift",
    "deadlift_bar_tilt",
    "deadlift_bent_arms",
    "deadlift_velocity_loss",
)


def _schemas(agent: object) -> list[dict]:
    return sorted(
        (build_legacy_openai_schema(tool)["function"] for tool in agent.tools),
        key=lambda schema: schema["name"],
    )


def _tool_names(agent: object) -> list[str]:
    return sorted(tool.info.name for tool in agent.tools)


def _workout_data(exercise_names: list[str]) -> dict:
    return {
        "schedule_id": 7,
        "workout_id": 3,
        "workout_name": "Pull day",
        "exercises": [
            {
                "workout_exercise_id": order,
                "exercise_id": order,
                "exercise_name": name,
                "order_number": order,
                "sets": [{"set_id": order * 10 + number, "set_number": number, "reps": 5, "weight": 60.0}
                         for number in (1, 2, 3)],
            }
            for order, name in enumerate(exercise_names, start=1)
        ],
    }


def _start_session(state: AgentState, exercise_names: list[str], quick: bool = False) -> None:
    if quick:
        session = WorkoutSession.create_quick_session(
            user_id=USER_ID, exercise_name=exercise_names[0], sets=3, reps=5, weight=60.0,
            rest_seconds=120, weight_unit="kg",
        )
    else:
        session = WorkoutSession(user_id=USER_ID, schedule_id=7, workout_data=_workout_data(exercise_names))
    state.set("workout.current_session", session.to_dict())
    state.set("workout.exercise_name", exercise_names[0])


def _userdata(state: AgentState, coaching: object = None) -> SimpleNamespace:
    return SimpleNamespace(
        state=state, coaching_service=coaching, visual_bridge=None, affect_service=None,
        compaction_service=None, room=None, wakeword_model=object(), audio_cue_service=None,
    )


def _fake_session() -> SimpleNamespace:
    return SimpleNamespace(input=SimpleNamespace(set_audio_enabled=lambda enabled: None))


def _features(**overrides: float) -> dict:
    features = {
        "rep_number": 2,
        "bar_midfoot_setup_cm": 1.2,
        "bar_drift_cm": 1.4,
        "concentric_velocity_mps": 0.62,
        "lean_back_deg": math.nan,
        "bar_source": "bar",
        "gravity_source": "measured",
        "dl_schema": 1,
    }
    features.update(overrides)
    return features


def _fault(fault_type: str, severity: str = "moderate", **details: str) -> dict:
    return {
        "fault_type": fault_type,
        "severity": severity,
        "severity_score": 0.5,
        "message": "",
        "details": {"observability": "observable", "min_tier": "mild", **details},
    }


def _rep_message(faults: list[dict] | None = None, rep_number: int = 2, **feature_overrides: float) -> dict:
    return {
        "type": "rep_complete",
        "rep_number": rep_number,
        "set_number": 1,
        "max_depth_angle": None,
        "depth_category": "n/a",
        "depth_target_met": True,
        "is_clean": not faults,
        "features": _features(rep_number=rep_number, **feature_overrides),
        "faults_detailed": faults or [],
        "faults_in_rep": [fault["fault_type"] for fault in faults or []],
    }


class _FakeCoachingService:
    def __init__(self, **kwargs: object) -> None:
        self._workout_active = False

    async def start(self) -> None:
        pass

    async def wait_progress_context(self) -> tuple[None, None]:
        return None, None


class _Recorder:
    """Records the order of what on_enter says and asks."""

    def __init__(self, state: AgentState, meta: DeadliftSessionMeta) -> None:
        self.state = state
        self.meta = meta
        self.events: list[tuple[str, object]] = []

    def setup_task_class(self) -> type:
        recorder = self

        class _FakeSetupTask:
            def __init__(self, **kwargs: object) -> None:
                pass

            def __await__(self):
                async def _answer() -> DeadliftSessionMeta:
                    recorder.events.append(("setup_task", recorder.state.get("workout.greeting_done")))
                    return recorder.meta

                return _answer().__await__()

        return _FakeSetupTask

    async def say(self, instructions: str, wait: bool = True, restore: bool = True) -> None:
        self.events.append(("say", instructions))


def _enter_workout(
    state: AgentState,
    monkeypatch: pytest.MonkeyPatch,
    last_session: dict | None = None,
    meta: DeadliftSessionMeta | None = None,
) -> tuple[_Recorder, list[dict]]:
    """Runs WorkoutAgent.on_enter with fakes; returns what it said and the DB lookups."""
    recorder = _Recorder(state, meta or DeadliftSessionMeta(grip="mixed", belt=True, shoes="flat"))
    lookups: list[dict] = []

    def _fake_last_session(db: object, user_id: str, exercise: str = "squat") -> dict | None:
        lookups.append({"user_id": user_id, "exercise": exercise})
        return last_session

    import agent.services.coaching_service as coaching_service_module
    import db.biomechanics_persistence as persistence_module

    monkeypatch.setattr(coaching_service_module, "CoachingService", _FakeCoachingService)
    monkeypatch.setattr(persistence_module, "get_last_completed_session", _fake_last_session)
    monkeypatch.setattr(workout_agent_module, "SessionLocal", lambda: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(workout_agent_module, "DeadliftSetupTask", recorder.setup_task_class())

    agent = WorkoutAgent(state=state, userdata=_userdata(state))
    agent._activity = SimpleNamespace(session=_fake_session())
    monkeypatch.setattr(agent, "_say", recorder.say)

    async def _no_wake_word() -> None:
        recorder.events.append(("greeting_done", state.get("workout.greeting_done")))

    monkeypatch.setattr(agent, "_start_wake_word_system", _no_wake_word)
    asyncio.run(agent.on_enter())
    return recorder, lookups


@pytest.fixture
def state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AgentState:
    monkeypatch.setattr(AgentState, "_load_user_from_database", lambda self, user_id: None)
    agent_state = AgentState(state_dir=tmp_path)
    agent_state.state["user"]["id"] = USER_ID
    return agent_state


@pytest.fixture
def deadlift_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    """The dev override NOWVA_DEV_COACHING_READY=deadlift, as the profile exposes it."""
    monkeypatch.setattr(DeadliftProfile, "coaching_ready", True)


@pytest.fixture
def deadlift_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(DeadliftProfile, "coaching_ready", False)


@pytest.fixture
def golden() -> dict:
    return json.loads(GOLDEN_PATH.read_text())


class TestCoachingReadyGate:
    def test_gated_deadlift_is_not_coached(self, deadlift_gated: None) -> None:
        assert not is_coached_deadlift(DEADLIFT)
        assert not is_coached_deadlift(PROGRAM_DEADLIFT)

    def test_dev_override_coaches_the_conventional_deadlift(self, deadlift_ready: None) -> None:
        for name in ("deadlift", DEADLIFT, PROGRAM_DEADLIFT):
            assert is_coached_deadlift(name)

    def test_other_exercises_are_never_the_deadlift(self, deadlift_ready: None) -> None:
        for name in (SQUAT, "Romanian Deadlift", "Barbell Bench Press", "", None):
            assert not is_coached_deadlift(name)

    def test_session_with_a_later_deadlift_includes_it(self, state: AgentState, deadlift_ready: None) -> None:
        _start_session(state, [SQUAT, PROGRAM_DEADLIFT])
        assert session_includes_coached_deadlift(state)

    def test_squat_session_does_not_include_it(self, state: AgentState, deadlift_ready: None) -> None:
        _start_session(state, [SQUAT])
        assert not session_includes_coached_deadlift(state)


class TestMenuGate:
    def test_menu_offers_the_deadlift_only_with_the_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(DeadliftProfile, "coaching_ready", False)
        assert "deadlift" not in get_main_menu_prompt().lower()
        monkeypatch.setattr(DeadliftProfile, "coaching_ready", True)
        assert "deadlift" in coachable_exercises_text()
        assert "deadlift" in get_main_menu_prompt().lower()

    def test_quick_deadlift_is_refused_while_gated(
        self, state: AgentState, deadlift_gated: None, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(BaseNovaAgent, "_log_function_call", lambda self, *args, **kwargs: None)
        menu = MainMenuAgent(state=state, userdata=_userdata(state))
        result = asyncio.run(menu.start_quick_exercise(exercise_name="deadlift", context=None))
        assert isinstance(result, tuple)
        assert "can't coach" in result[1]

    def test_quick_deadlift_gets_the_deadlift_task_with_the_override(
        self, state: AgentState, deadlift_ready: None,
    ) -> None:
        menu = MainMenuAgent(state=state, userdata=_userdata(state))

        async def _run() -> object:
            task = await menu.start_quick_exercise(exercise_name="deadlift", context=None)
            task.calibration_task.cancel()
            return task

        assert isinstance(asyncio.run(_run()), CollectDeadliftInfoTask)

    def test_quick_squat_keeps_the_plain_task(self, state: AgentState, deadlift_ready: None) -> None:
        menu = MainMenuAgent(state=state, userdata=_userdata(state))

        async def _run() -> object:
            task = await menu.start_quick_exercise(exercise_name="back squat", context=None)
            task.calibration_task.cancel()
            return task

        assert type(asyncio.run(_run())) is CollectExerciseInfoTask


def _start_scheduled_workout(
    state: AgentState, monkeypatch: pytest.MonkeyPatch, exercise_names: list[str],
) -> object:
    import db.schedule_utils as schedule_utils_module

    monkeypatch.setattr(schedule_utils_module, "get_todays_workout", lambda db, user_id: _workout_data(exercise_names))
    monkeypatch.setattr(main_menu_module, "SessionLocal", lambda: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(BaseNovaAgent, "_log_function_call", lambda self, *args, **kwargs: None)
    menu = MainMenuAgent(state=state, userdata=_userdata(state))
    return asyncio.run(menu.start_workout(context=None))


class TestScheduledWorkoutGate:
    """O4: a program's deadlift must not get a deadlift briefing, setup questions or tool
    while the deadlift is gated; the pipeline runs it untracked."""

    def test_gated_program_deadlift_runs_as_a_plain_workout(
        self, state: AgentState, deadlift_gated: None, monkeypatch: pytest.MonkeyPatch, golden: dict,
    ) -> None:
        agent = _start_scheduled_workout(state, monkeypatch, [PROGRAM_DEADLIFT])
        assert isinstance(agent, WorkoutAgent)
        assert not agent._includes_deadlift
        assert _schemas(agent) == golden["workout_agent_tools"]
        assert not state.get("calibration.active")

    def test_gated_program_deadlift_says_nothing_deadlift_specific(
        self, state: AgentState, deadlift_gated: None, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _start_session(state, [PROGRAM_DEADLIFT])
        recorder, lookups = _enter_workout(state, monkeypatch)
        assert [event for event, _ in recorder.events] == ["say", "greeting_done"]
        assert lookups == []
        assert state.get(EXERCISE_META_STATE_KEY) is None

    def test_ready_program_deadlift_skips_the_squat_assessment(
        self, state: AgentState, deadlift_ready: None, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state.set(EXERCISE_META_STATE_KEY, {"grip": "hook"})
        agent = _start_scheduled_workout(state, monkeypatch, [PROGRAM_DEADLIFT])
        assert agent._includes_deadlift
        assert not state.get("calibration.active")
        assert state.get(EXERCISE_META_STATE_KEY) is None, "a stale setup must be asked afresh"


class TestQuickPathMetadata:
    def test_setup_is_asked_even_when_every_set_parameter_is_known(self) -> None:
        instructions = build_task_instructions(DEADLIFT, 3, 5, 60.0, None, "kg", SETUP_QUESTION)
        assert "IMMEDIATELY" not in instructions
        assert SETUP_QUESTION in instructions
        assert "Do NOT ask for these again" in instructions

    def test_setup_is_asked_with_the_missing_parameters(self) -> None:
        instructions = build_task_instructions(DEADLIFT, None, None, None, None, None, SETUP_QUESTION)
        assert "Ask ONLY for how many sets and reps" in instructions
        assert instructions.index("Ask ONLY for") < instructions.index(SETUP_QUESTION)

    def test_deadlift_start_workout_takes_the_setup(self, state: AgentState, deadlift_ready: None) -> None:
        async def _schema() -> dict:
            task = CollectDeadliftInfoTask(exercise_name=DEADLIFT, user_id=USER_ID, state=state, userdata=object())
            task.calibration_task.cancel()
            assert not task.all_params_known
            return {schema["name"]: schema for schema in _schemas(task)}["start_workout"]

        properties = asyncio.run(_schema())["parameters"]["properties"]
        assert {"grip", "belt", "shoes", "plate_diameter_cm"} <= set(properties)
        assert {"sets", "reps", "weight", "weight_unit", "rest_seconds"} <= set(properties)

    def test_start_workout_stores_the_setup_and_skips_calibration(
        self, state: AgentState, deadlift_ready: None, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        handed_to: list[str] = []

        class _StubAgent:
            def __init__(self, state: AgentState, userdata: object) -> None:
                handed_to.append(type(self).__name__)

            async def update_chat_ctx(self, chat_ctx: object) -> None:
                pass

        class _StubWorkoutAgent(_StubAgent):
            pass

        class _StubTeachingAgent(_StubAgent):
            pass

        monkeypatch.setattr(quick_exercise_module, "WorkoutAgent", _StubWorkoutAgent)
        monkeypatch.setattr(quick_exercise_module, "TeachingAgent", _StubTeachingAgent)
        state.set("calibration.active", True)

        async def _run() -> None:
            task = CollectDeadliftInfoTask(exercise_name=DEADLIFT, user_id=USER_ID, state=state, userdata=object())
            await task.start_workout(sets=3, reps=5, weight=60.0, weight_unit="kg", grip="mixed", belt=True, shoes="flat")

        asyncio.run(_run())
        assert handed_to == ["_StubWorkoutAgent"]
        assert not state.get("calibration.active")
        assert state.get(EXERCISE_META_STATE_KEY) == {
            "grip": "mixed", "plate_diameter_cm": DEFAULT_PLATE_DIAMETER_CM, "belt": True, "shoes": "flat",
        }
        assert state.get("workout.exercise_name") == DEADLIFT

    def test_skipped_setup_still_records_the_default_plates(self, state: AgentState) -> None:
        store_exercise_meta(state, DeadliftSessionMeta())
        assert state.get(EXERCISE_META_STATE_KEY) == {"plate_diameter_cm": DEFAULT_PLATE_DIAMETER_CM}


class TestScheduledPathMetadata:
    def test_setup_is_asked_before_greeting_done(
        self, state: AgentState, deadlift_ready: None, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _start_session(state, [PROGRAM_DEADLIFT])
        recorder, _ = _enter_workout(state, monkeypatch, last_session={"total_reps": 15})
        assert [event for event, _ in recorder.events] == ["say", "setup_task", "greeting_done"]
        assert dict(recorder.events)["setup_task"] is not True
        assert state.get(EXERCISE_META_STATE_KEY) == {
            "grip": "mixed", "plate_diameter_cm": DEFAULT_PLATE_DIAMETER_CM, "belt": True, "shoes": "flat",
        }
        assert dict(recorder.events)["greeting_done"] is True

    def test_quick_path_setup_is_not_asked_twice(
        self, state: AgentState, deadlift_ready: None, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _start_session(state, [DEADLIFT], quick=True)
        store_exercise_meta(state, DeadliftSessionMeta(grip="hook"))
        recorder, _ = _enter_workout(state, monkeypatch, last_session={"total_reps": 15})
        assert [event for event, _ in recorder.events] == ["say", "greeting_done"]
        assert state.get(EXERCISE_META_STATE_KEY)["grip"] == "hook"

    def test_later_deadlift_is_set_up_but_not_briefed_at_the_start(
        self, state: AgentState, deadlift_ready: None, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _start_session(state, [SQUAT, PROGRAM_DEADLIFT])
        recorder, lookups = _enter_workout(state, monkeypatch, last_session=None)
        assert [event for event, _ in recorder.events] == ["say", "setup_task", "greeting_done"]
        assert lookups == []

    def test_squat_session_greets_exactly_as_before(
        self, state: AgentState, deadlift_ready: None, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _start_session(state, [SQUAT])
        recorder, lookups = _enter_workout(state, monkeypatch)
        assert [event for event, _ in recorder.events] == ["say", "greeting_done"]
        assert lookups == []
        assert state.get(EXERCISE_META_STATE_KEY) is None


class TestDeadliftSetupTask:
    def test_records_what_they_said_with_default_plates(
        self, state: AgentState, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        completed: list[DeadliftSessionMeta] = []

        async def _run() -> None:
            task = DeadliftSetupTask(state=state, userdata=_userdata(state))
            monkeypatch.setattr(task, "complete", completed.append)
            await task.record_deadlift_setup(grip="double", belt=False)

        asyncio.run(_run())
        assert completed == [DeadliftSessionMeta(grip="double", belt=False)]
        assert completed[0].plate_diameter_cm == pytest.approx(DEFAULT_PLATE_DIAMETER_CM, abs=1e-9)

    def test_asks_grip_belt_shoes_and_assumes_standard_plates(self, state: AgentState) -> None:
        async def _schema() -> dict:
            task = DeadliftSetupTask(state=state, userdata=_userdata(state))
            assert SETUP_QUESTION in task.instructions
            return {schema["name"]: schema for schema in _schemas(task)}["record_deadlift_setup"]

        properties = asyncio.run(_schema())["parameters"]["properties"]
        assert set(properties) == {"grip", "belt", "shoes", "plate_diameter_cm"}
        assert "45 centimetre" in SETUP_QUESTION


class TestFirstDeadliftSession:
    def test_first_session_gets_the_five_step_briefing_and_empty_bar(
        self, state: AgentState, deadlift_ready: None, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _start_session(state, [DEADLIFT], quick=True)
        store_exercise_meta(state, DeadliftSessionMeta())
        recorder, lookups = _enter_workout(state, monkeypatch, last_session=None)
        said = [payload for event, payload in recorder.events if event == "say"]
        assert len(said) == 2
        briefing = said[1]
        positions = [briefing.index(step) for step in SETUP_STEPS]
        assert positions == sorted(positions)
        assert "empty bar" in briefing
        assert lookups == [{"user_id": USER_ID, "exercise": "deadlift"}]
        assert [event for event, _ in recorder.events][-1] == "greeting_done"

    def test_returning_deadlifter_is_not_briefed(
        self, state: AgentState, deadlift_ready: None, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _start_session(state, [DEADLIFT], quick=True)
        store_exercise_meta(state, DeadliftSessionMeta())
        recorder, _ = _enter_workout(state, monkeypatch, last_session={"total_reps": 15})
        assert len([event for event, _ in recorder.events if event == "say"]) == 1

    def test_briefing_names_the_five_steps(self) -> None:
        assert SETUP_STEPS == (
            "feet hip-width with the bar over the middle of the foot",
            "grip just outside the legs",
            "shins to the bar",
            "shoulders over the bar",
            "pull the slack out, then push the floor away",
        )


class TestExplainDeadliftExposure:
    def test_deadlift_session_adds_only_explain_deadlift(
        self, state: AgentState, deadlift_ready: None, golden: dict,
    ) -> None:
        _start_session(state, [DEADLIFT], quick=True)
        agent = WorkoutAgent(state=state, userdata=_userdata(state))
        schemas = _schemas(agent)
        assert [schema for schema in schemas if schema["name"] != "explain_deadlift"] == golden["workout_agent_tools"]
        assert "explain_deadlift" in _tool_names(agent)

    def test_deadlift_session_prompt_adds_the_deadlift_section(self, state: AgentState, deadlift_ready: None) -> None:
        _start_session(state, [SQUAT, PROGRAM_DEADLIFT])
        agent = WorkoutAgent(state=state, userdata=_userdata(state))
        assert get_workout_prompt(includes_deadlift=True) == get_workout_prompt() + DEADLIFT_SECTION
        assert DEADLIFT_SECTION in agent.instructions
        assert "explain_deadlift" in _tool_names(agent)

    def test_gated_deadlift_session_keeps_the_squat_tools_and_prompt(
        self, state: AgentState, deadlift_gated: None, golden: dict,
    ) -> None:
        _start_session(state, [PROGRAM_DEADLIFT])
        agent = WorkoutAgent(state=state, userdata=_userdata(state))
        assert _schemas(agent) == golden["workout_agent_tools"]
        assert agent.instructions == golden["workout_agent_instructions"]

    def test_tool_takes_one_topic(self) -> None:
        schema = build_legacy_openai_schema(explain_deadlift)["function"]
        assert schema["name"] == "explain_deadlift"
        assert set(schema["parameters"]["properties"]) == {"topic"}

    def test_tool_answers_from_the_card(self) -> None:
        answer = asyncio.run(explain_deadlift(topic="mixed grip"))
        assert DEADLIFT_CARD["grip"] in answer
        assert "Answer from these notes only" in answer


class TestDeadliftCard:
    def test_card_stays_under_token_budget(self) -> None:
        card_chars = sum(len(topic) + len(section) for topic, section in DEADLIFT_CARD.items())
        assert card_chars / CHARS_PER_TOKEN <= CARD_TOKEN_BUDGET

    def test_back_rounding_is_never_claimed(self) -> None:
        assert "never say the back rounded or stayed flat" in DEADLIFT_CARD["back_rounding"]
        for invisible in NEVER_VISIBLE:
            assert invisible in DEADLIFT_CARD[OBSERVABILITY_TOPIC]

    def test_every_topic_has_keywords(self) -> None:
        assert set(TOPIC_KEYWORDS) == set(DEADLIFT_CARD)

    def test_setup_section_lists_the_briefing_steps(self) -> None:
        for step in SETUP_STEPS:
            assert step in DEADLIFT_CARD["setup"]

    @pytest.mark.parametrize(
        ("question", "topic"),
        [
            ("how close should I stand to the bar?", "bar_over_midfoot"),
            ("was my back rounded?", "back_rounding"),
            ("should I use a mixed grip?", "grip"),
            ("why do my hips shoot up first?", "hips_shoot"),
            ("the bar drifts away from my legs", "bar_path"),
            ("do touch and go reps count?", "touch_and_go"),
            ("should I wear a belt?", "belt_and_shoes"),
            ("can you see my back?", OBSERVABILITY_TOPIC),
            ("grip", "grip"),
        ],
    )
    def test_question_finds_its_section(self, question: str, topic: str) -> None:
        assert DEADLIFT_CARD[topic] in explain_deadlift_topic(question, TRIANGULATED)

    def test_unknown_question_lists_topics_and_what_is_seen(self) -> None:
        answer = explain_deadlift_topic("what about my nutrition", TRIANGULATED)
        assert "Topics with notes" in answer
        assert DEADLIFT_CARD[OBSERVABILITY_TOPIC] in answer

    def test_single_camera_says_the_bar_is_rough(self) -> None:
        assert "rough" in explain_deadlift_topic("bar path", SINGLE_CAMERA)
        assert "three-camera rig" in explain_deadlift_topic("bar path", TRIANGULATED)


def _check_form(state: AgentState, rep_message: dict | None) -> str:
    coaching = MagicMock()
    coaching._last_rep_message = rep_message
    agent = WorkoutAgent(state=state, userdata=_userdata(state, coaching))
    _, instructions = asyncio.run(agent.check_my_form())
    return instructions


class TestDeadliftFormCheck:
    def test_no_deadlift_rep_yet_asks_for_one(self, state: AgentState, deadlift_ready: None) -> None:
        _start_session(state, [DEADLIFT], quick=True)
        assert "haven't seen a deadlift rep" in _check_form(state, None)

    def test_squat_rep_left_over_is_not_a_deadlift_rep(self, state: AgentState, deadlift_ready: None) -> None:
        _start_session(state, [DEADLIFT], quick=True)
        squat_rep = {"rep_number": 4, "features": {"depth_ratio": 0.1}, "faults_detailed": []}
        assert "haven't seen a deadlift rep" in _check_form(state, squat_rep)

    def test_answers_from_the_last_rep_features(self, state: AgentState, deadlift_ready: None) -> None:
        _start_session(state, [DEADLIFT], quick=True)
        answer = _check_form(state, _rep_message([_fault("deadlift_hips_shoot", min_tier="moderate")]))
        assert "the bar started over the middle of their foot" in answer
        assert "the bar stayed close to their legs" in answer
        assert "0.6 metres per second" in answer
        assert "hips rising before the chest off the floor" in answer
        assert answer.index("the bar started") < answer.index("hips rising")

    def test_never_claims_to_see_the_back(self, state: AgentState, deadlift_ready: None) -> None:
        _start_session(state, [DEADLIFT], quick=True)
        answer = _check_form(state, _rep_message())
        assert "Never say their back rounded or stayed flat" in answer

    def test_direction_tells_which_way_to_fix_the_bar(self) -> None:
        findings = deadlift_form_findings(_rep_message([_fault("deadlift_bar_position", direction="forward")]))
        assert any("step closer" in finding for finding in findings)
        assert not any("started over the middle" in finding for finding in findings)

    def test_fault_below_its_min_tier_is_neither_relayed_nor_praised(self) -> None:
        wrist_proxy_drift = _fault("deadlift_bar_drift", severity="mild", min_tier="moderate", bar_source="wrist_proxy")
        findings = deadlift_form_findings(_rep_message([wrist_proxy_drift]))
        assert not any("drifting" in finding for finding in findings)
        assert not any("stayed close" in finding for finding in findings)

    def test_recap_only_velocity_loss_is_not_a_form_verdict(self) -> None:
        findings = deadlift_form_findings(_rep_message([_fault("deadlift_velocity_loss", min_tier="recap")]))
        assert not any("slowing" in finding for finding in findings)

    def test_unobservable_fault_is_left_out(self) -> None:
        fault = _fault("deadlift_lockout", observability="not_observable")
        findings = deadlift_form_findings(_rep_message([fault]))
        assert not any("standing tall" in finding for finding in findings)

    def test_nothing_measured_still_answers_from_the_rep(self) -> None:
        findings = deadlift_form_findings(
            _rep_message(bar_midfoot_setup_cm=math.nan, bar_drift_cm=math.nan, concentric_velocity_mps=math.nan)
        )
        assert findings == ["their last rep had nothing to correct in what the cameras measure"]

    def test_squat_form_check_is_unchanged_in_a_squat_session(self, state: AgentState, deadlift_ready: None) -> None:
        _start_session(state, [SQUAT], quick=True)
        coaching = MagicMock()
        coaching.get_current_form_snapshot.return_value = None
        coaching.last_rep_verdict.return_value = None
        agent = WorkoutAgent(state=state, userdata=_userdata(state, coaching))
        _, instructions = asyncio.run(agent.check_my_form())
        assert "face the camera and do one" in instructions


class TestPipelineMessages:
    def test_deadlift_start_capture_carries_the_session_setup(
        self, state: AgentState, deadlift_ready: None, main_function: Callable,
    ) -> None:
        start_capture_message = main_function("_start_capture_message")
        store_exercise_meta(state, DeadliftSessionMeta(grip="hook", belt=False, shoes="barefoot"))
        assert start_capture_message(exercise_meta_for(state, DEADLIFT)) == {
            "type": "start_capture",
            "exercise_meta": {"grip": "hook", "plate_diameter_cm": DEFAULT_PLATE_DIAMETER_CM, "belt": False, "shoes": "barefoot"},
        }

    def test_deadlift_without_a_setup_sends_an_empty_dict(
        self, state: AgentState, deadlift_ready: None, main_function: Callable,
    ) -> None:
        start_capture_message = main_function("_start_capture_message")
        assert start_capture_message(exercise_meta_for(state, DEADLIFT)) == {"type": "start_capture", "exercise_meta": {}}

    def test_gated_deadlift_sends_no_metadata(self, state: AgentState, deadlift_gated: None) -> None:
        store_exercise_meta(state, DeadliftSessionMeta(grip="mixed"))
        assert exercise_meta_for(state, DEADLIFT) is None

    def test_switch_to_the_deadlift_carries_meta(
        self, state: AgentState, deadlift_ready: None, main_function: Callable,
    ) -> None:
        with_switch_meta = main_function("_with_switch_meta")
        store_exercise_meta(state, DeadliftSessionMeta(grip="mixed"))
        message = {"type": "set_exercise", "exercise_name": PROGRAM_DEADLIFT, "total_sets": 3}
        relayed = with_switch_meta(message, exercise_meta_for(state, PROGRAM_DEADLIFT))
        assert relayed == {**message, "meta": {"grip": "mixed", "plate_diameter_cm": DEFAULT_PLATE_DIAMETER_CM}}
        assert "meta" not in message

    def test_main_sends_these_messages(self) -> None:
        source = MAIN_PY.read_text()
        assert "send_message(_start_capture_message(exercise_meta))" in source
        assert source.count('{"type": "start_capture"}') == 1, "only _start_capture_message builds it"
        assert "_with_switch_meta(message, self._exercise_meta(exercise_name))" in source

    def test_workout_cleanup_forgets_the_setup(self, state: AgentState, deadlift_ready: None) -> None:
        _start_session(state, [DEADLIFT], quick=True)
        store_exercise_meta(state, DeadliftSessionMeta(grip="mixed"))
        agent = WorkoutAgent(state=state, userdata=_userdata(state))
        agent._activity = SimpleNamespace(session=_fake_session())

        async def _no_wake_word() -> None:
            pass

        agent._stop_wake_word_system = _no_wake_word
        asyncio.run(agent._cleanup_workout())
        assert state.get(EXERCISE_META_STATE_KEY) is None


def _main_constant_set(name: str) -> set[str]:
    tree = ast.parse(MAIN_PY.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return set(ast.literal_eval(node.value.args[0]))
    raise AssertionError(f"{name} not found in main.py")


class TestDisplayEvents:
    def test_depth_follows_the_profile_the_pipeline_announced(self) -> None:
        """The agent saves the next exercise right after rest_start, before the pipeline's
        last set_complete of the old one; the pipeline's own cache_cues is in stream order."""
        source = MAIN_PY.read_text()
        assert _main_constant_set("DEPTHLESS_PROFILES") == {"deadlift"}
        assert 'self._pipeline_profile = message.get("profile")' in source
        assert source.count("has_depth = self._pipeline_profile not in DEPTHLESS_PROFILES") == 2

    def test_deadlift_set_summary_sends_null_depth(self, main_function: Callable) -> None:
        set_summary_display_event = main_function("_set_summary_display_event")
        message = {"set_number": 1, "total_reps": 5, "clean_reps": 3, "avg_depth": math.nan, "depth_consistency": math.nan}
        event = set_summary_display_event(message, False)
        assert event["avg_depth"] is None
        assert event["depth_consistency"] is None
        assert json.loads(json.dumps(event))["avg_depth"] is None

    def test_deadlift_rep_receipt_has_no_depth_class(self, main_function: Callable) -> None:
        rep_display_event = main_function("_rep_display_event")
        assert rep_display_event(_rep_message(), False)["depth_class_name"] == ""

    def test_deadlift_workout_start_names_its_profile(self, main_function: Callable) -> None:
        with_display_profile = main_function("_with_display_profile")
        event = {"type": "workout", "action": "start", "exercise": DEADLIFT}
        assert with_display_profile(event, True) == {**event, "profile": "deadlift"}


class TestDeadliftPersistence:
    def test_deadlift_faults_are_not_end_of_rep(self) -> None:
        assert not set(CONTRACT_FAULT_TYPES) & END_OF_REP_FAULT_TYPES

    def test_one_session_per_exercise_is_tagged_deadlift(self) -> None:
        recorder = BiomechanicsRecorder(user_id=USER_ID, exercise="deadlift")
        assert recorder._ops_session_start()[0][1]["exercise"] == "deadlift"

    def test_deadlift_cue_links_to_its_own_rep_and_is_judged_by_the_next(self) -> None:
        recorder = BiomechanicsRecorder(user_id=USER_ID, exercise="deadlift")
        recorder._ops_fault({"fault_type": "deadlift_bar_drift", "rep_number": 2, "severity_score": 0.6,
                             "severity": "moderate", "cue": "deadlift_bar_close"})
        rep_ops = recorder._ops_rep(_rep_message([_fault("deadlift_bar_drift")], rep_number=2))
        assert [kind for kind, _ in rep_ops] == ["insert_rep", "link_cues"]
        next_ops = recorder._ops_rep(_rep_message(rep_number=3))
        outcomes = [payload for kind, payload in next_ops if kind == "cue_outcome"]
        assert outcomes and outcomes[0]["effective"] is True

    def test_deadlift_features_are_stored_with_their_schema(self) -> None:
        row = _build_rep_row(_rep_message())
        assert row["kinematics"]["dl_schema"] == 1
        assert row["kinematics"]["bar_drift_cm"] == pytest.approx(1.4, abs=1e-9)
        assert row["kinematics"]["lean_back_deg"] is None, "NaN is stored as null: JSONB has no NaN"

    def test_squat_rep_keeps_its_kinematic_summary(self) -> None:
        message = {"rep_number": 1, "features": {"depth_ratio": 0.1}, "rep_kinematic_summary": {"rep_number": 1}}
        assert _build_rep_row(message)["kinematics"] == {"rep_number": 1}

    def test_recorder_session_id_is_a_uuid(self) -> None:
        assert isinstance(BiomechanicsRecorder(user_id=USER_ID, exercise="deadlift").session_id, uuid.UUID)
