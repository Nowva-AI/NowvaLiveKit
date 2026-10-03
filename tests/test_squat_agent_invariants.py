"""
Squat invariants on the voice-agent side (deadlift PLAN §5, milestone J0): the
squat cue text, fix praise and report labels, the orchestrator's squat-defined
constants and the workout agent's tool list for a squat-only session, pinned as
literals so that changing one fails the test that names it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.core.workout_session import WorkoutSession
from agent.services import coaching_orchestrator as orchestrator_module
from agent.services.coaching_constants import CUE_DISPLAY_LABELS, CUE_TEXT_MAP, FIXED_CUE_TEXT, TRACKING_LOST_CUE
from agent.services.coaching_orchestrator import CoachingOrchestrator
from agent.services.coaching_service import DEFAULT_EXERCISE_NAME, POSITIVE_CUE_KEYS, _profile_name
from biomechanics.profiles.squat import SquatProfile

SQUAT_EXERCISE = "Barbell Back Squat"
TIMING_TOLERANCE_S = 1e-9

SQUAT_CUE_TEXT = {
    "knees_out": "Knees out!",
    "knees_out_left": "Left knee out!",
    "knees_out_right": "Right knee out!",
    "chest_up": "Chest up!",
    "heels_down": "Heels down!",
    "heels_down_left": "Left heel down!",
    "heels_down_right": "Right heel down!",
    "whole_foot": "Whole foot!",
    "even_it_out": "Even it out!",
    "even_it_out_left": "Drifting left, stay centered!",
    "even_it_out_right": "Drifting right, stay centered!",
    "level_bar": "Keep the bar level!",
    "deeper": "Get deeper!",
    "square_feet": "Square your feet!",
    "square_feet_left": "Left foot even!",
    "square_feet_right": "Right foot even!",
    "lockout": "Stand tall up top!",
    "slow_down": "Control the way down!",
    "same_depth": "Match your first rep!",
    "drive": "Drive up hard!",
    "brace": "Brace your core!",
    "stance_explain": "That's coming from your stance — step your feet out wider.",
    "stance_wider": "A little wider.",
    "stance_narrower": "Bring it in a touch.",
    "toe_out_explain": "That's coming from your feet — turn your toes out more.",
    "toe_out_more": "More toe-out.",
    "toe_out_less": "Ease them back in.",
    "adjust_good": "Right there — hold that.",
    "good_rep": "Good rep!",
    "great_depth": "Great depth!",
    "strong": "Strong!",
    "clean": "Clean!",
    "perfect": "Perfect!",
    "tracking_lost": "I can't see you fully, step back into view.",
    "rep_1": "One!", "rep_2": "Two!", "rep_3": "Three!", "rep_4": "Four!", "rep_5": "Five!",
    "rep_6": "Six!", "rep_7": "Seven!", "rep_8": "Eight!", "rep_9": "Nine!", "rep_10": "Ten!",
    "rep_11": "Eleven!", "rep_12": "Twelve!", "rep_13": "Thirteen!", "rep_14": "Fourteen!",
    "rep_15": "Fifteen!", "rep_16": "Sixteen!", "rep_17": "Seventeen!", "rep_18": "Eighteen!",
    "rep_19": "Nineteen!", "rep_20": "Twenty!",
}
SQUAT_FIXED_CUE_TEXT = {
    "knees_out_fixed": "That's it, spreading the floor.",
    "chest_up_fixed": "Better, bar and hips rose together.",
    "heels_down_fixed": "Good, whole foot stayed down.",
    "whole_foot_fixed": "Better, balanced over mid-foot.",
    "even_it_out_fixed": "Good, staying centered.",
    "level_bar_fixed": "Good, bar stayed level.",
    "deeper_fixed": "There's your depth.",
    "square_feet_fixed": "Good, feet are even.",
    "lockout_fixed": "Good, all the way up.",
    "slow_down_fixed": "Better, controlled on the way down.",
    "same_depth_fixed": "Right back to your depth.",
    "drive_fixed": "That one moved, good drive.",
}
SQUAT_CUE_DISPLAY_LABELS = {
    "knees_out": "Knees out!",
    "knees_out_left": "Left knee out!",
    "knees_out_right": "Right knee out!",
    "chest_up": "Chest up!",
    "heels_down": "Heels down!",
    "heels_down_left": "Left heel down!",
    "heels_down_right": "Right heel down!",
    "whole_foot": "Whole foot!",
    "even_it_out": "Even it out!",
    "even_it_out_left": "Drifting left",
    "even_it_out_right": "Drifting right",
    "level_bar": "Level the bar",
    "deeper": "Go deeper!",
    "square_feet": "Square feet",
    "square_feet_left": "Left foot even",
    "square_feet_right": "Right foot even",
    "lockout": "Stand tall",
    "slow_down": "Control the descent",
    "same_depth": "Same depth",
    "drive": "Drive up!",
    "brace": "Brace core!",
    "stance_explain": "Stance is the cause",
    "stance_wider": "Wider",
    "stance_narrower": "Narrower",
    "toe_out_explain": "Foot angle is the cause",
    "toe_out_more": "More toe-out",
    "toe_out_less": "Less toe-out",
    "adjust_good": "On target",
    "good_rep": "Good rep!",
    "great_depth": "Great depth!",
    "strong": "Strong!",
    "clean": "Clean!",
    "perfect": "Perfect!",
    "tracking_lost": "Out of view",
    **{key: "Fixed" for key in SQUAT_FIXED_CUE_TEXT},
}

# The workout agent's tools in a squat-only session. PLAN §4.4: deadlift tools
# appear only when the plan holds a deadlift, so this list must not change.
SQUAT_WORKOUT_TOOLS = [
    "check_my_form",
    "clear_pain",
    "end_set_early",
    "end_workout",
    "explain_squat",
    "flag_pain",
    "get_next_exercise",
    "get_workout_progress",
    "how_do_i_sound",
    "log_set_effort",
    "show_me",
    "skip_exercise",
]


def _workout_agent_class() -> type:
    try:
        from agent.agents.workout_agent import WorkoutAgent
    except ValueError as error:
        # db.database refuses to import without DATABASE_URL; nothing connects on import.
        pytest.skip(f"WorkoutAgent imports the database module; set DATABASE_URL (any URL) to run: {error}")
    return WorkoutAgent


def _tool_name(tool: object) -> str:
    return tool.info.name


class TestSquatCueText:
    def test_squat_cue_text(self):
        assert {key: CUE_TEXT_MAP.get(key) for key in SQUAT_CUE_TEXT} == SQUAT_CUE_TEXT
        assert TRACKING_LOST_CUE == "tracking_lost"

    def test_squat_fix_praise_text(self):
        assert FIXED_CUE_TEXT == SQUAT_FIXED_CUE_TEXT
        assert {key: CUE_TEXT_MAP.get(key) for key in SQUAT_FIXED_CUE_TEXT} == SQUAT_FIXED_CUE_TEXT

    def test_squat_report_labels(self):
        assert {key: CUE_DISPLAY_LABELS.get(key) for key in SQUAT_CUE_DISPLAY_LABELS} == SQUAT_CUE_DISPLAY_LABELS

    def test_every_cached_squat_cue_has_text(self):
        assert [key for key in SquatProfile().get_cue_dict() if key not in CUE_TEXT_MAP] == []

    def test_praise_keys_for_the_cue_banner(self):
        expected = {"good_rep", "great_depth", "strong", "clean", "perfect", "adjust_good", *SQUAT_FIXED_CUE_TEXT}
        assert POSITIVE_CUE_KEYS == frozenset(expected)


class TestSquatOrchestratorConstants:
    """Squat-defined today; PLAN §4.3 item 7 selects them per exercise, squat by default."""

    def test_safety_and_side_view_sets(self):
        assert orchestrator_module.SAFETY_FAULT_TYPE == "knee_valgus"
        assert orchestrator_module.SAFETY_SEVERITY == "severe"
        assert orchestrator_module.SIDE_VIEW_FAULTS == frozenset({"forward_lean", "hip_shoot", "velocity_loss", "balance"})
        assert orchestrator_module.SIDE_VIEW_SYMPTOMS == frozenset(
            {"excessive_trunk_lean", "hip_shoot", "velocity_loss", "balance_forward"}
        )
        assert orchestrator_module.STANCE_CAUSE_IDS == frozenset(
            {"narrow_stance", "stance_toe_mismatch", "narrow_foot_angle"}
        )

    def test_symptom_to_fault_map(self):
        assert orchestrator_module.SYMPTOM_FAULT_TYPES == {
            "knee_not_tracking_toes": "knee_valgus",
            "hip_shoot": "hip_shoot",
            "heel_rise": "heel_rise",
            "balance_forward": "balance",
            "hip_shift": "hip_shift",
            "depth_limit": "depth",
            "uneven_setup": "foot_placement",
            "incomplete_lockout": "lockout",
            "fast_descent": "tempo",
            "velocity_loss": "velocity_loss",
        }

    def test_cue_bandwidth(self):
        assert orchestrator_module.CUE_SEVERITIES == frozenset({"moderate", "severe"})
        assert orchestrator_module.MILD_REPEAT_WINDOW_REPS == 3
        assert orchestrator_module.MILD_REPEAT_MIN_REPS == 2
        assert orchestrator_module.MAX_CUES_PER_FAULT_PER_SET == 2
        assert orchestrator_module.FIX_HOLD_REPS == 2
        assert orchestrator_module.POSITIVE_CUE_MIN_REP_GAP == 3
        assert orchestrator_module.BEST_REP_CUE_KEY == "strong"

    def test_timing(self):
        orchestrator = CoachingOrchestrator(
            play_cached_audio_fn=None, generate_llm_reply_fn=None, get_cue_audio_fn=None,
        )
        assert orchestrator._min_fault_cue_gap == pytest.approx(8.0, abs=TIMING_TOLERANCE_S)
        assert orchestrator._set_idle_timeout_s == pytest.approx(15.0, abs=TIMING_TOLERANCE_S)
        assert orchestrator._diagnosis_wait_s == pytest.approx(2.5, abs=TIMING_TOLERANCE_S)

    def test_the_default_exercise_is_the_squat(self):
        assert DEFAULT_EXERCISE_NAME == SQUAT_EXERCISE
        assert _profile_name(SQUAT_EXERCISE) == "squat"


class TestSquatWorkoutAgentTools:
    def test_squat_session_tool_list(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        workout_agent_class = _workout_agent_class()
        from agent.core.agent_state import AgentState

        monkeypatch.setattr(AgentState, "_load_user_from_database", lambda self, user_id: None)
        state = AgentState(state_dir=tmp_path)
        state.state["user"]["id"] = "athlete-1"
        session = WorkoutSession.create_quick_session("athlete-1", SQUAT_EXERCISE, sets=3, reps=5)
        state.set("workout.current_session", session.to_dict())
        state.set("workout.exercise_name", SQUAT_EXERCISE)
        room = SimpleNamespace(remote_participants={}, on=lambda *args: None, off=lambda *args: None)
        userdata = SimpleNamespace(
            state=state, coaching_service=None, visual_bridge=None, affect_service=None,
            compaction_service=None, room=room, wakeword_model=object(),
        )

        agent = workout_agent_class(state=state, userdata=userdata)

        assert sorted(_tool_name(tool) for tool in agent.tools) == SQUAT_WORKOUT_TOOLS
