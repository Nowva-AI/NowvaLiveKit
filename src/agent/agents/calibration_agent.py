"""
CalibrationAgent - Handles the calibration phase before a workout begins.
"""

from __future__ import annotations

import logging

from agent.agents.shared.base_agent import BaseNovaAgent

logger = logging.getLogger(__name__)

CALIBRATION_PROMPT = """
# Calibration Mode
You are calibrating the user's movement — a short, conversational phase before the real workout starts.

# How Calibration Works
The system runs two phases automatically:

1. **Form Assessment** (1 rep): The user does 1 bodyweight rep. The system
   analyzes form and either asks for corrections or moves on.
2. **Calibration** (5 reps): The user does 5 deep bodyweight reps that measure
   their range and set their personal depth target.

# Your Job
- Be encouraging and patient
- The coaching orchestrator sends you generation instructions with analysis
  data — follow them
- Do NOT say "assessment" or "calibration" to the user — keep it natural
- Once calibration completes, the orchestrator will tell you to announce it

# Safety
- If the user reports pain, stop immediately. Ask what hurts.
- Never push through discomfort during calibration.

"""


class CalibrationAgent(BaseNovaAgent):
    """Runs the calibration phase, then hands off to WorkoutAgent."""

    def __init__(self, state, userdata) -> None:
        super().__init__(state=state, userdata=userdata, instructions=CALIBRATION_PROMPT)

    async def on_enter(self):
        existing = getattr(self.userdata, "coaching_service", None)
        if existing is not None:
            existing.set_calibration_complete_callback(self._on_calibration_complete)
            logger.info("[CALIBRATION] Reusing CoachingService from TeachingAgent")
        else:
            from agent.services.coaching_service import CoachingService

            coaching_service = CoachingService(
                session=self.session,
                state=self.state,
                room=self.userdata.room,
                on_calibration_complete=self._on_calibration_complete,
                audio_cue_service=self.userdata.audio_cue_service,
            )
            await coaching_service.start()
            self.userdata.coaching_service = coaching_service

            exercise_name = self.state.get("workout.exercise_name", "this exercise")

            await self._say(
                f"Conversationally tell the user that you haven't seen them do {exercise_name} before, "
                "and that you want to take a quick look at their form first. They have to do one bodyweight squat, "
                "hands out in front, and go as deep as they can. You will correct them until they achieve acceptable, safe form.",
                restore=True,
            )

            self.state.set("workout.greeting_done", True)
            self.state.save_state()
            logger.info("[CALIBRATION] Greeting done — signalled main.py to start pose estimation")

    async def _on_calibration_complete(self):
        logger.info("[CALIBRATION] Calibration complete — handing off to WorkoutAgent")
        from agent.agents.workout_agent import WorkoutAgent

        new_agent = await self._carry_context_to(WorkoutAgent(
            state=self.state,
            userdata=self.userdata,
            from_calibration=True,
        ))
        self.session.update_agent(new_agent)
