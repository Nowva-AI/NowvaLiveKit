"""
TeachingAgent — Assesses a beginner's squat form before calibration.

The pipeline drives the assessment loop (collects reps, runs diagnosis,
decides pass/fail). This agent owns the speech: setup cues, correction
feedback, choreographed demo on the first failure, and the handoff to
CalibrationAgent once the user passes.

No live fault cues fire during assessment — those start in calibration.

Flow:
  on_enter → setup speech → pipeline assessment loop →
    (fail round 1 → demo + encouragement) |
    (fail round N → spoken corrections) |
    (pass → handoff to CalibrationAgent)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from agent.agents.shared.base_agent import BaseNovaAgent
from agent.services.assessment_logger import RecommendationRecord
from agent.services.coaching_service import assessment_focus_line
from agent.services.squat_card import ExplainSquatMixin
from biomechanics.diagnosis.demo_builder import summarize_cue_magnitude

logger = logging.getLogger(__name__)

_TALL_THRESHOLD_CM = 185


def _get_teaching_prompt(exercise: str) -> str:
    return (
        f"# Teaching Mode\n"
        f"You are assessing a beginner's {exercise} form before their workout. "
        f"Be brief, warm, and clear.\n"
        f"For any how or why question about squat technique, call explain_squat and "
        f"answer from it. Never claim to see what the cameras can't: lower-back tuck, "
        f"upper-back rounding, bracing and foot arch are invisible on every setup."
    )


class TeachingAgent(ExplainSquatMixin, BaseNovaAgent):
    """Assesses a beginner's squat form, then hands off to CalibrationAgent."""

    def __init__(self, state, userdata, exercise: str = "squat") -> None:
        self.exercise = exercise
        self._handed_off: bool = False
        self._squat_cued: bool = False

        user = state.get_user() or {}
        height = user.get("height_cm")
        self._user_height_cm: float = float(height) if height is not None else 0.0
        self.user_is_tall: bool = self._user_height_cm >= _TALL_THRESHOLD_CM

        super().__init__(
            state=state,
            userdata=userdata,
            instructions=_get_teaching_prompt(exercise),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def on_enter(self):
        existing = getattr(self.userdata, "coaching_service", None)
        if existing is not None:
            existing.set_assessment_result_callback(self._on_assessment_result)
            existing.set_assessment_ready_callback(self._on_assessment_ready)
            logger.info("[TEACHING] Reusing existing CoachingService")
        else:
            from agent.services.coaching_service import CoachingService

            coaching_service = CoachingService(
                session=self.session,
                state=self.state,
                room=self.userdata.room,
                on_assessment_result=self._on_assessment_result,
                audio_cue_service=self.userdata.audio_cue_service,
            )
            coaching_service.set_assessment_ready_callback(self._on_assessment_ready)
            await coaching_service.start()
            self.userdata.coaching_service = coaching_service

        self._start_assessment_logging()

        await self._say(
            f"Say something natural and brief: before their workout you want "
            f"to watch how they move their {self.exercise}. One sentence, no "
            f"filler, in your own words."
        )
        await self._run_setup()

    async def on_exit(self):
        if not self._handed_off:
            self._stop_assessment_logging(passed=False)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    async def _run_setup(self):
        context_block = (
            f"[CONTEXT] exercise={self.exercise}, is_tall={self.user_is_tall}\n\n"
        )
        instruction = (
            context_block
            + "Walk the user through the starting position for the squat. "
            "Cover in this order, naturally: "
            "(1) Stand back far enough that the camera can see them head to toe. "
            "(2) Feet shoulder-width apart, toes slightly out. "
            "If is_tall is true, mention they can go a touch wider "
            "if it feels more comfortable — keep this casual, not prescriptive. "
            "(3) Keep gaze forward throughout. "
            "(4) Ask them to stand tall and hold still for a moment while you "
            "lock onto their movement. Do NOT tell them to squat yet — you will "
            "cue the squat separately once tracking is ready. "
            "Sound like a real coach, not a manual. Brief, warm, clear. "
            "Do not number the points. Do not say 'step one'. "
            "This should flow as natural speech."
        )
        await self._say(instruction)

        self.state.set("workout.greeting_done", True)
        self.state.save_state()
        logger.info("[TEACHING] Setup complete — pipeline will start assessment")

    # ------------------------------------------------------------------
    # Assessment ready callback
    # ------------------------------------------------------------------

    async def _on_assessment_ready(self) -> None:
        if self._handed_off or self._squat_cued:
            return
        self._squat_cued = True
        logger.info("[TEACHING] Pipeline ready — cueing first squat")
        await self._say(
            f"[CONTEXT] exercise={self.exercise}\n\n"
            "Tracking just locked onto the user — they are set up correctly "
            "and you can see them fully. Now cue the movement: take a big "
            "breath, brace the core like they're about to take a little punch, "
            "and squat down when ready. One or two sentences, natural, no filler."
        )

    # ------------------------------------------------------------------
    # Assessment result callback
    # ------------------------------------------------------------------

    async def _on_assessment_result(self, message: dict, demo_was_played: bool) -> None:
        passed = message.get("passed", False)
        round_num = message.get("round", 1)
        diagnosis = message.get("diagnosis", {})
        scoring = message.get("scoring", {})

        self._log_assessment_recommendations(diagnosis, round_num)

        if passed:
            await self._handoff(diagnosis, scoring, round_num)
            return

        if message.get("proceed_anyway"):
            coaching = getattr(self.userdata, "coaching_service", None)
            immediate = diagnosis.get("immediate_causes") or []
            top_cause = coaching.top_cause(diagnosis) if coaching else (immediate[0] if immediate else None)
            await self._handoff(
                diagnosis, scoring, round_num,
                carried_focus=assessment_focus_line(top_cause, message.get("body_measurement")),
            )
            return

        if demo_was_played:
            await self._say(
                f"[CONTEXT] exercise={self.exercise}, round={round_num}\n\n"
                "The user just watched a visual demo showing their form issue "
                "and how to correct it. Briefly acknowledge that they've now seen "
                "the difference, then tell them to try again with that adjustment. "
                "One sentence, warm and direct, in your own words."
            )
        else:
            await self._say(self._build_correction_prompt(diagnosis, scoring, round_num))

    # ------------------------------------------------------------------
    # Correction prompt
    # ------------------------------------------------------------------

    def _build_correction_prompt(
        self, diagnosis: dict, scoring: dict, round_num: int,
    ) -> str:
        immediate = diagnosis.get("immediate_causes", [])
        top_issue = immediate[0] if immediate else {}

        context_parts = [f"Assessment round {round_num}."]

        mean_pct = round(scoring.get("mean_score", 0) * 100)
        context_parts.append(f"Form score: {mean_pct}/100.")

        if top_issue:
            context_parts.append(
                f"Main issue: {top_issue.get('explanation', 'form issue detected')}."
            )
            delta = top_issue.get("parameter_delta")
            if delta:
                magnitude = summarize_cue_magnitude(top_issue.get("cause_id", ""), delta)
                context_parts.append(f"Recommended adjustment: {magnitude}.")

        if len(immediate) > 1:
            context_parts.append(
                f"Secondary issue: {immediate[1].get('explanation', '')}."
            )

        context_str = " ".join(context_parts)

        return (
            f"[CONTEXT] exercise={self.exercise}\n"
            f"{context_str}\n\n"
            f"The user's form needs adjustment before you can move on. "
            f"Tell them the specific issue and the one concrete change to make, "
            f"then ask to see it again with that change. "
            f"Keep it encouraging and brief (2-3 sentences), in your own words. "
            f"Do NOT say assessment."
        )

    # ------------------------------------------------------------------
    # Handoff to CalibrationAgent
    # ------------------------------------------------------------------

    async def _handoff(
        self, diagnosis: dict, scoring: dict, round_num: int,
        carried_focus: str | None = None,
    ) -> None:
        self._handed_off = True
        logger.info("[TEACHING] Assessment done — handing off to CalibrationAgent")

        self._stop_assessment_logging(passed=carried_focus is None)

        session_causes = diagnosis.get("session_causes", [])
        contextual = diagnosis.get("contextual_notes", [])
        mean_pct = round(scoring.get("mean_score", 0) * 100)

        notes = []
        for cause in session_causes + contextual:
            explanation = cause.get("explanation", "")
            if explanation:
                notes.append(explanation)

        context = f"Form score: {mean_pct}/100. Round {round_num}."
        if notes:
            context += f" Things to watch under load: {'; '.join(notes[:2])}."

        if carried_focus:
            verdict = (
                f"After {round_num} tries one thing is still off, and that's fine: it becomes "
                f"the focus of the workout instead of another retry. {carried_focus} "
                f"Tell them the one thing to keep working on, without asking them to try again. "
            )
        else:
            verdict = "The user's squat form passed the assessment. Praise their form briefly. "
        await self._say(
            f"[CONTEXT] exercise={self.exercise}, {context}\n\n"
            f"{verdict}"
            f"Then transition: tell them you need a few deep bodyweight squats "
            f"to learn their movement pattern so you can coach them properly "
            f"during the workout. "
            f"Keep it natural and brief (2-3 sentences)."
        )

        coaching = getattr(self.userdata, "coaching_service", None)
        if coaching is not None:
            coaching.set_assessment_result_callback(None)
            coaching.set_assessment_ready_callback(None)

        from agent.agents.calibration_agent import CalibrationAgent
        new_agent = await self._carry_context_to(CalibrationAgent(state=self.state, userdata=self.userdata))
        self.session.update_agent(new_agent)

    # ------------------------------------------------------------------
    # Assessment logging
    # ------------------------------------------------------------------

    def _start_assessment_logging(self) -> None:
        session_dir = os.environ.get("NOWVA_SESSION_OUTPUT_DIR")
        if not session_dir:
            return
        coaching = getattr(self.userdata, "coaching_service", None)
        if coaching is None:
            return
        session_id = Path(session_dir).name
        coaching.start_assessment(
            session_dir=session_dir,
            session_id=session_id,
            user_height_cm=self._user_height_cm,
        )
        logger.info("[TEACHING] Assessment logging started")

    def _stop_assessment_logging(self, passed: bool) -> None:
        coaching = getattr(self.userdata, "coaching_service", None)
        if coaching is not None:
            coaching.stop_assessment(passed)

    def _log_assessment_recommendations(
        self, diagnosis: dict, round_num: int,
    ) -> None:
        coaching = getattr(self.userdata, "coaching_service", None)
        if coaching is None:
            return
        al = coaching.assessment_logger
        if al is None:
            return
        immediate = diagnosis.get("immediate_causes", [])
        recommendations = [
            RecommendationRecord(
                fault_type=cause.get("cause_id", "unknown"),
                recommendation=cause.get("explanation", ""),
            )
            for cause in immediate
        ]
        if recommendations:
            al.set_outgoing_recommendations_latest(recommendations)
