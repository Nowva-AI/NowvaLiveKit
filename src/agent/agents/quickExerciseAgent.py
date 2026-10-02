"""
CollectExerciseInfoTask - Collects quick-exercise parameters then hands off to calibration or workout.
"""

import asyncio
import logging
from typing import Literal

from livekit.agents import function_tool, AgentTask

from agent.agents.shared.base_agent import build_agent_instructions
from agent.agents.shared.helpers import check_calibration, start_calibration_mode
from agent.agents.teaching_agent import TeachingAgent
from agent.agents.workout_agent import WorkoutAgent
from agent.core.workout_session import WorkoutSession
from agent.agents.shared.affect_mixin import AffectNodesMixin


logger = logging.getLogger(__name__)

# Every recorded quick session was bodyweight; a load is only asked about for a loaded lift.
BODYWEIGHT_EXERCISES = ("Bodyweight Squat",)
DEFAULT_REST_SECONDS_BODYWEIGHT = 60
DEFAULT_REST_SECONDS_LOADED = 120
# What create_quick_session assumed before units were tracked.
DEFAULT_WEIGHT_UNIT = "lb"
LB_PER_KG = 2.20462


def _is_loaded(exercise_name: str, weight: float | None) -> bool:
    return exercise_name not in BODYWEIGHT_EXERCISES or bool(weight)


def default_rest_seconds(exercise_name: str, weight: float | None) -> int:
    return DEFAULT_REST_SECONDS_LOADED if _is_loaded(exercise_name, weight) else DEFAULT_REST_SECONDS_BODYWEIGHT


def _display_params(sets: int | None, reps: int | None, weight: float | None, weight_unit: str | None,
                    rest_seconds: int | None) -> dict:
    # The display page renders weight_lbs in pounds.
    weight_lbs = weight * LB_PER_KG if weight and weight_unit == "kg" else weight
    return {"sets": sets, "reps": reps, "weight_lbs": weight_lbs, "rest_seconds": rest_seconds}


def build_task_instructions(
    exercise_name: str,
    sets: int | None,
    reps: int | None,
    weight: float | None,
    rest_seconds: int | None,
    weight_unit: str | None = None,
) -> str:
    """Collection instructions: sets and reps, plus the weight only for a loaded lift; never rest."""
    loaded = _is_loaded(exercise_name, weight)
    known = []
    if sets is not None:
        known.append(f"sets = {sets}")
    if reps is not None:
        known.append(f"reps per set = {reps}")
    if weight is not None:
        known.append(f"weight = {weight:g} {weight_unit or DEFAULT_WEIGHT_UNIT}" if weight else "weight = bodyweight")
    if rest_seconds is not None:
        known.append(f"rest = {rest_seconds} seconds")

    lines = [f"The user wants to do a quick exercise: {exercise_name}."]
    if known:
        lines.append(
            f"The user ALREADY provided: {', '.join(known)}. "
            f"Do NOT ask for these again — asking again is a bad experience."
        )

    missing = []
    if sets is None or reps is None:
        missing.append("how many sets and reps")
    if weight is None and loaded:
        missing.append("the weight on the bar, in kilograms or pounds as they prefer")
    if not missing:
        lines.append(
            "Every parameter needed is known. Call the start_workout tool IMMEDIATELY "
            "with the values above. Do not ask the user anything."
        )
        return "\n".join(lines)

    lines.append(
        f"Ask ONLY for {' and '.join(missing)}, in one short question. If they're unsure, "
        f"suggest three sets of eight. Never ask about rest: it defaults to "
        f"{default_rest_seconds(exercise_name, weight)} seconds unless they bring it up."
    )
    if not loaded:
        lines.append(
            "It's bodyweight unless they mention a bar or a weight. If they give a weight, "
            "pass it with its unit, and ask the unit if they only give a number."
        )
    lines.append("Once you have them, call start_workout.")
    return "\n".join(lines)


class CollectExerciseInfoTask(AffectNodesMixin, AgentTask):
    def __init__(
        self,
        exercise_name: str,
        user_id: str,
        state,
        userdata,
        chat_ctx=None,
        sets: int | None = None,
        reps: int | None = None,
        weight: float | None = None,
        rest_seconds: int | None = None,
        weight_unit: str | None = None,
    ):
        task_instructions = build_task_instructions(
            exercise_name, sets, reps, weight, rest_seconds, weight_unit
        )
        super().__init__(
            instructions=build_agent_instructions(state, task_instructions),
            chat_ctx=chat_ctx,
        )

        self.user_id = user_id
        self.state = state
        self.userdata = userdata
        self.exercise_name = exercise_name
        self.initial_params = {
            "sets": sets,
            "reps": reps,
            "weight": weight,
            "weight_unit": weight_unit,
            "rest_seconds": rest_seconds,
        }
        self.all_params_known = (
            sets is not None
            and reps is not None
            and (weight is not None or not _is_loaded(exercise_name, weight))
        )

        self.calibration_task = asyncio.create_task(
            check_calibration(user_id, exercise_name)
        )

    def _publish_visual(self, event: dict) -> None:
        bridge = getattr(self.userdata, "visual_bridge", None)
        if bridge is not None:
            bridge.send(event)

    async def on_enter(self):
        self._publish_visual({
            "type": "setup",
            "action": "show",
            "exercise": self.exercise_name,
            "params": _display_params(**self.initial_params),
        })
        if self.all_params_known:
            await self.session.generate_reply(
                instructions=(
                    "All exercise details were already provided. Confirm the plan back "
                    "to the user in one short sentence and call start_workout immediately."
                )
            )
        else:
            await self.session.generate_reply(
                instructions=(
                    "Acknowledge the switch in a few words, then ask for the "
                    "missing details in one natural question. One or two "
                    "sentences total."
                )
            )

    @function_tool
    async def start_workout(
        self,
        sets: int,
        reps: int,
        weight: float | None = None,
        weight_unit: Literal["kg", "lb"] | None = None,
        rest_seconds: int | None = None,
    ):
        """
        Call this once you know the sets and reps, and the weight for a loaded lift.

        Args:
            sets: Number of sets to perform
            reps: Target reps per set
            weight: Load in the unit the user said. 0 for bodyweight.
            weight_unit: "kg" or "lb", as the user said it
            rest_seconds: Rest between sets in seconds, only if the user asked for one
        """
        exercise_name = self.exercise_name
        if weight is None:
            weight = self.initial_params["weight"] or 0.0
        weight_unit = weight_unit or self.initial_params["weight_unit"] or DEFAULT_WEIGHT_UNIT
        if rest_seconds is None:
            rest_seconds = self.initial_params["rest_seconds"] or default_rest_seconds(exercise_name, weight)
        logger.info(
            f"[QUICK EXERCISE] Collected: {sets}x{reps}, "
            f"weight={weight}{weight_unit}, rest={rest_seconds}s, exercise={exercise_name}"
        )
        self._publish_visual({
            "type": "setup",
            "action": "complete",
            "exercise": exercise_name,
            "params": _display_params(sets, reps, weight, weight_unit, rest_seconds),
        })

        calibration_profile = await self.calibration_task

        if calibration_profile:
            self.state.set("workout.calibration_profile", calibration_profile)
            # Explicitly disarm calibration mode — a stale flag from a dead
            # session would make main.py launch the pipeline in assessment mode
            self.state.set("calibration.active", None)
            logger.info(f"[CALIBRATION] Found existing calibration for {exercise_name}")
        else:
            start_calibration_mode(self.state, exercise_name, {
                "type": "quick_exercise",
            })
            logger.info(f"[CALIBRATION] No calibration for {exercise_name} — entering calibration mode")

        session = WorkoutSession.create_quick_session(
            user_id=self.user_id,
            exercise_name=exercise_name,
            sets=sets,
            reps=reps,
            weight=weight,
            rest_seconds=rest_seconds,
            weight_unit=weight_unit,
        )

        self.state.set("workout.current_session", session.to_dict())
        self.state.set("workout.exercise_name", exercise_name)
        self.state.set("workout.active", True)
        self.state.switch_mode("workout")
        self.state.save_state()

        logger.info("[STATE] Switched to workout mode — main.py will detect and start pose estimation")

        next_agent = (
            WorkoutAgent(state=self.state, userdata=self.userdata)
            if calibration_profile
            else TeachingAgent(state=self.state, userdata=self.userdata)
        )
        # LiveKit starts a new agent with an empty context; hand this conversation over.
        await next_agent.update_chat_ctx(self.chat_ctx.copy())
        return next_agent

    @function_tool
    async def cancel_exercise(self, shut_down: bool = False):
        """
        Call this when the user no longer wants this exercise: they want to go back, do
        something else, or stop for the day.

        Args:
            shut_down: True if they asked to shut down or turn Nova off
        """
        logger.info(f"[QUICK EXERCISE] Cancelled {self.exercise_name} (shut_down={shut_down})")
        self._publish_visual({"type": "menu", "action": "show"})
        from agent.agents.main_menu_agent import MainMenuAgent

        main_menu = MainMenuAgent(
            state=self.state, userdata=self.userdata, ask_shutdown_confirmation=shut_down,
        )
        await main_menu.update_chat_ctx(self.chat_ctx.copy())
        return main_menu
