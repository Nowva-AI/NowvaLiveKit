"""
CollectExerciseInfoTask - Collects quick-exercise parameters then hands off to calibration or workout.
"""

import asyncio
import logging

from livekit.agents import function_tool, AgentTask

from agent.agents.prompts import BASE_PROMPT
from agent.agents.shared.helpers import check_calibration, start_calibration_mode
from agent.agents.teaching_agent import TeachingAgent
from agent.agents.workout_agent import WorkoutAgent
from agent.core.workout_session import WorkoutSession
from agent.agents.shared.affect_mixin import AffectNodesMixin


logger = logging.getLogger(__name__)

PARAM_DESCRIPTIONS = {
    "sets": "number of sets",
    "reps": "reps per set",
    "weight": "weight in lbs (0 for bodyweight)",
    "rest_seconds": "rest between sets in seconds",
}


def build_task_instructions(
    exercise_name: str,
    sets: int | None,
    reps: int | None,
    weight: float | None,
    rest_seconds: int | None,
) -> str:
    """Build collection instructions that only ask for parameters the user has not given yet."""
    provided = {
        "sets": sets,
        "reps": reps,
        "weight": weight,
        "rest_seconds": rest_seconds,
    }
    known = {name: value for name, value in provided.items() if value is not None}
    missing = [name for name, value in provided.items() if value is None]

    lines = [f"The user wants to do a quick exercise: {exercise_name}."]

    if known:
        known_text = ", ".join(
            f"{PARAM_DESCRIPTIONS[name]} = {value}" for name, value in known.items()
        )
        lines.append(
            f"The user ALREADY provided: {known_text}. "
            f"Do NOT ask for these again — asking again is a bad experience."
        )

    if missing:
        missing_text = ", ".join(PARAM_DESCRIPTIONS[name] for name in missing)
        lines.append(
            f"Conversationally collect ONLY the remaining details: {missing_text}. "
            f"Ask for them together in one natural question, not one at a time. "
            f"If the user is unsure, suggest defaults: 3-5 sets, 5-10 reps, "
            f"bodyweight or a light weight, 90-120 seconds rest."
        )
        lines.append(
            "Once every value is known, call the start_workout tool with all of them."
        )
    else:
        lines.append(
            "Every parameter is already known. Call the start_workout tool IMMEDIATELY "
            "with the values above. Do not ask the user anything."
        )

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
    ):
        task_instructions = build_task_instructions(
            exercise_name, sets, reps, weight, rest_seconds
        )
        super().__init__(
            instructions=f"{BASE_PROMPT}\n\n{task_instructions}",
            chat_ctx=chat_ctx,
        )

        self.user_id = user_id
        self.state = state
        self.userdata = userdata
        self.exercise_name = exercise_name
        self.initial_params = {
            "sets": sets,
            "reps": reps,
            "weight_lbs": weight,
            "rest_seconds": rest_seconds,
        }
        self.all_params_known = all(
            value is not None for value in (sets, reps, weight, rest_seconds)
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
            "params": self.initial_params,
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
        weight: float = 0.0,
        rest_seconds: int = 120,
    ):
        """
        Call this once the user has provided sets, reps, weight, and rest time.

        Args:
            sets: Number of sets to perform
            reps: Target reps per set
            weight: Weight in lbs. Use 0 for bodyweight exercises.
            rest_seconds: Rest between sets in seconds (default 120)
        """
        exercise_name = self.exercise_name
        logger.info(
            f"[QUICK EXERCISE] Collected: {sets}x{reps}, "
            f"weight={weight}, rest={rest_seconds}s, exercise={exercise_name}"
        )
        self._publish_visual({
            "type": "setup",
            "action": "complete",
            "exercise": exercise_name,
            "params": {
                "sets": sets,
                "reps": reps,
                "weight_lbs": weight,
                "rest_seconds": rest_seconds,
            },
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
        )

        self.state.set("workout.current_session", session.to_dict())
        self.state.set("workout.exercise_name", exercise_name)
        self.state.set("workout.active", True)
        self.state.switch_mode("workout")
        self.state.save_state()

        logger.info("[STATE] Switched to workout mode — main.py will detect and start pose estimation")

        if calibration_profile:
            return WorkoutAgent(state=self.state, userdata=self.userdata)
        else:
            return TeachingAgent(state=self.state, userdata=self.userdata)
