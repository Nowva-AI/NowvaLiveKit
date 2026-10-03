"""Squat sessions must not change while the deadlift lands: prompts, instructions and tool
schemas are pinned against a golden captured before the deadlift work, and the squat's
start_capture / set_exercise / display messages against today's payloads.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pytest
from livekit.agents.llm.utils import build_legacy_openai_schema

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.agents.main_menu_agent import MainMenuAgent
from agent.agents.prompts import get_main_menu_prompt, get_workout_prompt
from agent.agents.quickExerciseAgent import CollectExerciseInfoTask, build_task_instructions
from agent.agents.workout_agent import WorkoutAgent
from agent.core.agent_state import AgentState
from agent.core.workout_session import WorkoutSession
from biomechanics.profiles.deadlift import DeadliftProfile

GOLDEN_PATH = Path(__file__).parent / "fixtures" / "squat_agent_golden.json"
SQUAT_EXERCISE = "Barbell Back Squat"


def _schemas(agent: object) -> list[dict]:
    return sorted(
        (build_legacy_openai_schema(tool)["function"] for tool in agent.tools),
        key=lambda schema: schema["name"],
    )


def _squat_session_state(tmp_path: Path) -> AgentState:
    state = AgentState(state_dir=tmp_path)
    state.state["user"]["id"] = "athlete-1"
    session = WorkoutSession.create_quick_session(
        user_id="athlete-1", exercise_name=SQUAT_EXERCISE, sets=3, reps=5, weight=60.0,
        rest_seconds=120, weight_unit="kg",
    )
    state.set("workout.current_session", session.to_dict())
    state.set("workout.exercise_name", SQUAT_EXERCISE)
    return state


def _userdata(state: AgentState) -> SimpleNamespace:
    return SimpleNamespace(
        state=state, coaching_service=None, visual_bridge=None, affect_service=None,
        compaction_service=None, room=None, wakeword_model=object(),
    )


@pytest.fixture(autouse=True)
def deadlift_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Normal users: the deadlift is not coaching-ready (no NOWVA_DEV_COACHING_READY)."""
    monkeypatch.setattr(DeadliftProfile, "coaching_ready", False)


@pytest.fixture
def golden() -> dict:
    return json.loads(GOLDEN_PATH.read_text())


@pytest.fixture
def squat_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AgentState:
    monkeypatch.setattr(AgentState, "_load_user_from_database", lambda self, user_id: None)
    return _squat_session_state(tmp_path)


class TestSquatPromptsUnchanged:
    def test_workout_prompt_is_the_golden(self, golden: dict) -> None:
        assert get_workout_prompt() == golden["workout_prompt"]

    def test_main_menu_prompt_is_the_golden(self, golden: dict) -> None:
        assert get_main_menu_prompt() == golden["main_menu_prompt"]

    def test_workout_agent_instructions_are_the_golden(self, golden: dict, squat_state: AgentState) -> None:
        agent = WorkoutAgent(state=squat_state, userdata=_userdata(squat_state))
        assert agent.instructions == golden["workout_agent_instructions"]

    @pytest.mark.parametrize(
        ("case", "args"),
        [
            ("bodyweight_nothing_known", ("Bodyweight Squat", None, None, None, None)),
            ("loaded_nothing_known", ("Barbell Back Squat", None, None, None, None)),
            ("partial_prefill", ("Bodyweight Squat", 2, None, None, 30)),
            ("loaded_all_known", ("Barbell Back Squat", 3, 5, 60.0, None, "kg")),
        ],
    )
    def test_quick_task_instructions_are_the_golden(self, golden: dict, case: str, args: tuple) -> None:
        assert build_task_instructions(*args) == golden["quick_task_instructions"][case]


class TestSquatToolListsUnchanged:
    def test_workout_agent_tools_are_the_golden(self, golden: dict, squat_state: AgentState) -> None:
        agent = WorkoutAgent(state=squat_state, userdata=_userdata(squat_state))
        assert _schemas(agent) == golden["workout_agent_tools"]

    def test_dev_override_leaves_a_squat_session_unchanged(
        self, golden: dict, squat_state: AgentState, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(DeadliftProfile, "coaching_ready", True)
        agent = WorkoutAgent(state=squat_state, userdata=_userdata(squat_state))
        assert _schemas(agent) == golden["workout_agent_tools"]
        assert agent.instructions == golden["workout_agent_instructions"]

    def test_quick_task_tools_are_the_golden(self, golden: dict, squat_state: AgentState) -> None:
        async def _tools() -> list[dict]:
            task = CollectExerciseInfoTask(
                exercise_name=SQUAT_EXERCISE, user_id="athlete-1", state=squat_state,
                userdata=_userdata(squat_state),
            )
            task.calibration_task.cancel()
            return _schemas(task)

        assert asyncio.run(_tools()) == golden["quick_task_tools"]

    def test_main_menu_tools_are_the_golden(self, golden: dict, squat_state: AgentState) -> None:
        menu = MainMenuAgent(state=squat_state, userdata=_userdata(squat_state))
        assert _schemas(menu) == golden["main_menu_tools"]


class TestSquatMessagesUnchanged:
    def test_start_capture_is_byte_identical(self, main_function: Callable) -> None:
        start_capture_message = main_function("_start_capture_message")
        assert json.dumps(start_capture_message(None)) == json.dumps({"type": "start_capture"})

    def test_squat_has_no_exercise_meta(self, squat_state: AgentState) -> None:
        from agent.agents.shared.deadlift_session import exercise_meta_for

        assert exercise_meta_for(squat_state, SQUAT_EXERCISE) is None

    def test_squat_set_exercise_is_relayed_untouched(self, main_function: Callable) -> None:
        with_switch_meta = main_function("_with_switch_meta")
        message = {"type": "set_exercise", "exercise_name": SQUAT_EXERCISE, "total_sets": 3}
        assert with_switch_meta(message, None) is message

    def test_squat_set_exercise_meta_from_the_agent_is_kept(self, main_function: Callable) -> None:
        with_switch_meta = main_function("_with_switch_meta")
        message = {"type": "set_exercise", "exercise_name": SQUAT_EXERCISE, "meta": {"note": "x"}}
        assert with_switch_meta(message, {"grip": "mixed"})["meta"] == {"note": "x"}

    def test_squat_workout_start_display_event_has_no_profile(self, main_function: Callable) -> None:
        with_display_profile = main_function("_with_display_profile")
        event = {"type": "workout", "action": "start", "exercise": SQUAT_EXERCISE}
        assert with_display_profile(event, False) is event

    def test_squat_set_summary_display_event_is_todays(self, main_function: Callable) -> None:
        set_summary_display_event = main_function("_set_summary_display_event")
        message = {
            "type": "set_complete", "set_number": 2, "total_reps": 5, "clean_reps": 4,
            "avg_depth": 101.3, "depth_consistency": 2.1, "fault_summary": {"knee_valgus": {"count": 1}},
        }
        assert set_summary_display_event(message, True) == {
            "type": "set_summary",
            "set_number": 2,
            "total_reps": 5,
            "clean_reps": 4,
            "avg_depth": 101.3,
            "depth_consistency": 2.1,
            "fault_summary": {"knee_valgus": {"count": 1}},
        }

    def test_squat_rep_display_event_is_todays(self, main_function: Callable) -> None:
        rep_display_event = main_function("_rep_display_event")
        message = {
            "type": "rep_complete", "rep_number": 3, "set_number": 1, "depth_category": "parallel",
            "is_clean": False, "faults_in_rep": ["knee_valgus"],
        }
        assert rep_display_event(message, True) == {
            "type": "rep",
            "rep_number": 3,
            "set_number": 1,
            "depth_class_name": "parallel",
            "is_clean": False,
            "faults": ["knee_valgus"],
        }
