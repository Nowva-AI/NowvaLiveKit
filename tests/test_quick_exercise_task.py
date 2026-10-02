"""
Tests for quick-exercise instruction building with prefilled parameters.
"""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import agent.agents.quickExerciseAgent as quick_exercise_module
from agent.agents.quickExerciseAgent import CollectExerciseInfoTask, build_task_instructions
from agent.core.agent_state import AgentState


class _StubAgent:
    def __init__(self, state=None, userdata=None):
        self.state = state
        self.userdata = userdata
        self.handed_chat_ctx = None

    async def update_chat_ctx(self, chat_ctx) -> None:
        self.handed_chat_ctx = chat_ctx


class _StubWorkoutAgent(_StubAgent):
    pass


class _StubTeachingAgent(_StubAgent):
    pass


def _run_start_workout(state: AgentState, calibration_profile: dict | None):
    async def _run():
        task = CollectExerciseInfoTask(
            exercise_name="squat",
            user_id="test-user",
            state=state,
            userdata=object(),
        )
        return await task.start_workout(sets=2, reps=4, weight=0.0, rest_seconds=30)

    async def _fake_check_calibration(user_id: str, exercise_name: str):
        return calibration_profile

    return _fake_check_calibration, _run


@pytest.fixture
def state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AgentState:
    monkeypatch.setattr(
        AgentState, "_load_user_from_database", lambda self, user_id: None
    )
    return AgentState(state_dir=tmp_path)


class TestStartWorkoutCalibrationFlag:
    def test_stale_calibration_flag_cleared_when_profile_found(
        self, state: AgentState, monkeypatch: pytest.MonkeyPatch
    ):
        # Regression: a session killed mid-calibration leaves calibration.active=True
        # in the persisted state. When the next quick exercise finds a calibration
        # profile in the DB, the stale flag must be cleared — otherwise main.py
        # launches the pipeline in assessment mode while the voice side runs a
        # normal workout set.
        state.set("calibration.active", True)
        state.set("calibration.pending_workout", {"type": "quick_exercise"})

        monkeypatch.setattr(quick_exercise_module, "WorkoutAgent", _StubWorkoutAgent)
        monkeypatch.setattr(quick_exercise_module, "TeachingAgent", _StubTeachingAgent)
        fake_check, run = _run_start_workout(state, calibration_profile={"depth": {}})
        monkeypatch.setattr(quick_exercise_module, "check_calibration", fake_check)

        result = asyncio.run(run())

        assert isinstance(result, _StubWorkoutAgent)
        assert not state.get("calibration.active")
        assert state.get("workout.calibration_profile") == {"depth": {}}
        assert state.get_mode() == "workout"

    def test_no_profile_enters_calibration_mode(
        self, state: AgentState, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(quick_exercise_module, "WorkoutAgent", _StubWorkoutAgent)
        monkeypatch.setattr(quick_exercise_module, "TeachingAgent", _StubTeachingAgent)
        fake_check, run = _run_start_workout(state, calibration_profile=None)
        monkeypatch.setattr(quick_exercise_module, "check_calibration", fake_check)

        result = asyncio.run(run())

        assert isinstance(result, _StubTeachingAgent)
        assert state.get("calibration.active") is True
        assert state.get_mode() == "workout"


class TestBuildTaskInstructions:
    def test_bodyweight_asks_only_sets_and_reps(self):
        instructions = build_task_instructions("Bodyweight Squat", None, None, None, None)
        assert "sets and reps" in instructions
        assert "weight on the bar" not in instructions
        assert "Never ask about rest" in instructions
        assert "ALREADY provided" not in instructions

    def test_loaded_lift_also_asks_weight_with_unit(self):
        instructions = build_task_instructions("Barbell Back Squat", None, None, None, None)
        assert "sets and reps" in instructions
        assert "weight on the bar" in instructions
        assert "kilograms or pounds" in instructions

    def test_partial_prefill_only_asks_for_missing(self):
        instructions = build_task_instructions("Bodyweight Squat", 2, None, None, 30)
        assert "sets = 2" in instructions
        assert "rest = 30 seconds" in instructions
        assert "Do NOT ask for these again" in instructions
        assert "Ask ONLY for" in instructions

    def test_sets_and_reps_known_starts_immediately_for_bodyweight(self):
        instructions = build_task_instructions("Bodyweight Squat", 2, 3, None, None)
        assert "Call the start_workout tool IMMEDIATELY" in instructions
        assert "Ask ONLY for" not in instructions

    def test_loaded_weight_known_with_unit(self):
        instructions = build_task_instructions("Barbell Back Squat", 3, 5, 60.0, None, "kg")
        assert "weight = 60 kg" in instructions
        assert "Call the start_workout tool IMMEDIATELY" in instructions

    def test_bodyweight_zero_counts_as_provided(self):
        instructions = build_task_instructions("Barbell Back Squat", 3, 5, 0.0, None)
        assert "weight = bodyweight" in instructions
        assert "Call the start_workout tool IMMEDIATELY" in instructions


class TestStartWorkoutParameters:
    def test_weight_unit_and_default_rest_reach_session(
        self, state: AgentState, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(quick_exercise_module, "WorkoutAgent", _StubWorkoutAgent)
        monkeypatch.setattr(quick_exercise_module, "TeachingAgent", _StubTeachingAgent)

        async def _fake_check(user_id: str, exercise_name: str):
            return {"depth": {}}

        monkeypatch.setattr(quick_exercise_module, "check_calibration", _fake_check)

        async def _run():
            task = CollectExerciseInfoTask(
                exercise_name="Barbell Back Squat", user_id="test-user", state=state, userdata=object(),
            )
            return await task.start_workout(sets=3, reps=5, weight=60.0, weight_unit="kg")

        asyncio.run(_run())

        first_set = state.get("workout.current_session")["exercises"][0]["sets"][0]
        assert first_set["target_weight"] == pytest.approx(60.0)
        assert first_set["weight_unit"] == "kg"
        assert first_set["rest_seconds"] == quick_exercise_module.DEFAULT_REST_SECONDS_LOADED

    def test_values_given_to_main_menu_survive_a_bare_tool_call(
        self, state: AgentState, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(quick_exercise_module, "WorkoutAgent", _StubWorkoutAgent)
        monkeypatch.setattr(quick_exercise_module, "TeachingAgent", _StubTeachingAgent)

        async def _fake_check(user_id: str, exercise_name: str):
            return {"depth": {}}

        monkeypatch.setattr(quick_exercise_module, "check_calibration", _fake_check)

        async def _run():
            task = CollectExerciseInfoTask(
                exercise_name="Barbell Back Squat", user_id="test-user", state=state, userdata=object(),
                weight=135.0, weight_unit="lb", rest_seconds=45,
            )
            return await task.start_workout(sets=3, reps=5)

        asyncio.run(_run())

        first_set = state.get("workout.current_session")["exercises"][0]["sets"][0]
        assert first_set["target_weight"] == pytest.approx(135.0)
        assert first_set["weight_unit"] == "lb"
        assert first_set["rest_seconds"] == 45


class TestCancel:
    def test_cancel_returns_to_main_menu(self, state: AgentState, monkeypatch: pytest.MonkeyPatch):
        from agent.agents.main_menu_agent import MainMenuAgent

        async def _fake_check(user_id: str, exercise_name: str):
            return None

        monkeypatch.setattr(quick_exercise_module, "check_calibration", _fake_check)

        async def _run():
            task = CollectExerciseInfoTask(
                exercise_name="Bodyweight Squat", user_id="test-user", state=state, userdata=object(),
            )
            return await task.cancel_exercise(shut_down=True)

        result = asyncio.run(_run())

        assert isinstance(result, MainMenuAgent)
        assert result._ask_shutdown_on_entry is True
        assert state.get_mode() != "workout"
