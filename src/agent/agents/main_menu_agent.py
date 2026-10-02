"""
MainMenuAgent - Primary interaction hub with schedule management, workout start, and program creation
"""

import asyncio
import logging
import os
import re
from typing import Literal

from livekit.agents import RunContext
from livekit.agents.llm import function_tool

from agent.agents.quickExerciseAgent import CollectExerciseInfoTask
from agent.agents.prompts import get_main_menu_prompt
from agent.agents.shared.base_agent import BaseNovaAgent
from agent.agents.shared.helpers import check_calibration, normalize_exercise_name, start_calibration_mode
from db.database import SessionLocal

logger = logging.getLogger(__name__)


def _reentry_instructions(last_mode_switch: dict | None) -> str:
    came_from = (last_mode_switch or {}).get("from")
    if came_from in (None, "main_menu", "onboarding"):
        where = "The user is back at the main menu."
    else:
        where = f"The user just came back to the main menu from {came_from.replace('_', ' ')}."
    return (
        f"{where} In one short line, acknowledge what just happened if the conversation "
        "shows it, then ask what they want to do next — or, if their last request is still "
        "undone, ask about that. This is not a goodbye: never say goodbye or sign off. "
        "Vary the wording every time."
    )


class MainMenuAgent(BaseNovaAgent):
    """Primary interaction hub: schedule management, workout start, program creation."""

    def __init__(self, state, userdata, ask_shutdown_confirmation: bool = False) -> None:
        super().__init__(state=state, userdata=userdata, instructions=get_main_menu_prompt()),
        self.calibration_profile = None
        self._ask_shutdown_on_entry = ask_shutdown_confirmation
        # User-message count when "shut down?" was asked, so only the next reply confirms.
        self._shutdown_asked_on_turn: int | None = None

    def _user_message_count(self) -> int:
        # Spoken and typed turns both land in the chat before a tool runs;
        # on_user_turn_completed never sees typed ones.
        return sum(1 for item in self.chat_ctx.items if getattr(item, "role", None) == "user")

    async def on_enter(self):
        """Greet on first visit or first login; on every later return, one short "what next" line."""
        self._publish_visual({"type": "menu", "action": "show"})
        if self._ask_shutdown_on_entry:
            self._shutdown_asked_on_turn = self._user_message_count()
            await self._say(
                "The user asked to shut down. Ask in a few words to confirm they want you to "
                "turn off. Don't say goodbye yet. Vary the wording."
            )
        elif self.state.is_first_time_main_menu():
            self.state.mark_main_menu_visited()
            self.state.save_state()
            await self._say(
                "Welcome the user for the first time. Mention just two things they can do — "
                "start a workout, or set up a program — and that they can just ask for anything else. "
                "One or two sentences."
            )
        elif not self.state.get("session.main_menu_greeted", False):
            self.state.set("session.main_menu_greeted", True)
            self.state.save_state()
            await self._say(
                "The user is back at the main menu. Greet them back with one short, natural line "
                "and an open question about what they want to do — different phrasing every session, "
                "never a menu of options."
            )
        else:
            await self._say(_reentry_instructions(self.state.get("session.last_mode_switch")))

    # ===== WORKOUT START TOOLS =====

    @function_tool
    async def start_workout(self, context: RunContext):
        """
        Call this when the user wants to start a workout.
        User might say: "start workout", "let's train", "I'm ready", "begin workout"
        """
        logger.info("[MAIN MENU] User requested to start workout")
        self._publish_visual({"type": "menu", "action": "select", "choice": "workout"})

        from db.schedule_utils import get_todays_workout
        from agent.core.workout_session import WorkoutSession

        db = SessionLocal()
        try:
            user_id = self.user_id
            workout = get_todays_workout(db, user_id)

            if not workout:
                logger.info("[WORKOUT] No workout scheduled for today")
                self._publish_visual({"type": "menu", "action": "show"})
                return None, "Tell the user nothing is scheduled today and offer to check their upcoming schedule or do a quick exercise instead. One or two sentences, vary the phrasing."

            # Initialize workout session
            session = WorkoutSession(
                user_id=user_id,
                schedule_id=workout["schedule_id"],
                workout_data=workout
            )

            # Store session in state
            self.state.set("workout.current_session", session.to_dict())

            # Set exercise name for main.py to pass to pose estimation
            first_exercise = session.get_current_exercise()
            exercise_name = first_exercise.exercise_name if first_exercise else "Barbell Back Squat"
            self.state.set("workout.exercise_name", exercise_name)

            # Check calibration for the first exercise
            self.calibration_profile = await check_calibration(user_id, exercise_name)

            if self.calibration_profile:
                self.state.set("workout.calibration_profile", self.calibration_profile)
                # Explicitly disarm calibration mode — a stale flag from a dead
                # session would make main.py launch the pipeline in assessment mode
                self.state.set("calibration.active", None)
                logger.info(f"[CALIBRATION] Found existing calibration for {exercise_name}")
            else:
                start_calibration_mode(self.state, exercise_name, {
                    "type": "scheduled_workout",
                })
                logger.info(f"[CALIBRATION] No calibration found for {exercise_name} — entering calibration mode")

            self.state.switch_mode("workout")
            self.state.set("workout.active", True)
            self.state.save_state()

            logger.info("[STATE] Switched to workout mode - main.py will detect and start pose estimation")

            self._log_function_call("start_workout", {}, "handoff to WorkoutAgent")
            self._publish_visual({
                "type": "menu", "action": "prepare",
                "label": f"Loading {exercise_name}",
            })

            # Handoff to WorkoutAgent
            await self._suppress_turn_detection()
            from agent.agents.workout_agent import WorkoutAgent
            return await self._carry_context_to(WorkoutAgent(state=self.state, userdata=self.userdata))

        except Exception as e:
            logger.exception("[WORKOUT ERROR] Failed to load workout")
            self._publish_visual({"type": "menu", "action": "show"})
            result = (None, "Tell the user you're having trouble loading their workout right now and to give it a moment. Keep it reassuring, one sentence, vary the phrasing.")
            self._log_function_call("start_workout", {}, result)
            return result
        finally:
            db.close()

    @function_tool
    async def start_quick_exercise(
        self,
        exercise_name: str,
        context: RunContext,
        sets: int | None = None,
        reps: int | None = None,
        weight: float | None = None,
        weight_unit: Literal["kg", "lb"] | None = None,
        rest_seconds: int | None = None,
    ):
        """
        Call this when the user wants to do a single exercise without a scheduled workout.
        User might say: "I want to squat", "let me do some bench press",
        "I just want to deadlift", "can I just do squats?"
        Or the user might not mention a particular exercise but just say "I want to start a quick exercise"

        IMPORTANT: Extract EVERY detail the user already mentioned and pass it as a parameter.
        Only omit parameters the user did not mention.
        Example: "I wanna squat, two sets of three, thirty seconds rest, bodyweight"
        -> exercise_name="squat", sets=2, reps=3, rest_seconds=30, weight=0

        Args:
            exercise_name: The exercise the user wants to do (e.g., "squat", "bench press", "deadlift")
            sets: Number of sets, if the user mentioned it
            reps: Reps per set, if the user mentioned it
            weight: Load in the unit the user said, if they mentioned one. Use 0 for bodyweight.
            weight_unit: "kg" or "lb", as the user said it, whenever weight is non-zero
            rest_seconds: Rest between sets in seconds, if the user mentioned it
        """
        exercise_name = normalize_exercise_name(exercise_name) or exercise_name
        logger.info(
            f"[MAIN MENU] User wants quick exercise: {exercise_name} "
            f"(sets={sets}, reps={reps}, weight={weight}{weight_unit or ''}, rest={rest_seconds})"
        )
        self._publish_visual({"type": "menu", "action": "select", "choice": "quick_exercise"})
        return await self._carry_context_to(CollectExerciseInfoTask(
            exercise_name=exercise_name,
            user_id=self.user_id,
            state=self.state,
            userdata=self.userdata,
            sets=sets,
            reps=reps,
            weight=weight,
            rest_seconds=rest_seconds,
            weight_unit=weight_unit,
        ))

    # ===== PROGRAM TOOLS =====

    def _extract_program_params_from_request(self, user_request: str) -> dict:
        """Extract program parameters from natural language request."""
        from datetime import datetime

        extracted = {}
        request_lower = user_request.lower()

        # Extract GOAL/CATEGORY
        hypertrophy_keywords = ['muscle', 'bigger', 'size', 'mass', 'hypertrophy', 'bulk', 'grow', 'butt', 'glutes', 'chest', 'arms', 'legs', 'aesthetic', 'look good', 'shredded', 'toned']
        strength_keywords = ['stronger', 'strength', 'powerlifting', 'max', '1rm', 'heavy', 'strong']
        power_keywords = ['explosive', 'power', 'jump', 'vertical', 'sprint', 'athletics', 'athleticism', 'athlete', 'speed', 'fast', 'quick']

        hypertrophy_score = sum(1 for kw in hypertrophy_keywords if kw in request_lower)
        strength_score = sum(1 for kw in strength_keywords if kw in request_lower)
        power_score = sum(1 for kw in power_keywords if kw in request_lower)

        if max(hypertrophy_score, strength_score, power_score) > 0:
            if hypertrophy_score > strength_score and hypertrophy_score > power_score:
                extracted['goal'] = 'hypertrophy'
            elif strength_score > hypertrophy_score and strength_score > power_score:
                extracted['goal'] = 'strength'
            elif power_score > hypertrophy_score and power_score > strength_score:
                extracted['goal'] = 'power'

        # Extract DURATION (weeks)
        week_match = re.search(r'(\d+)\s*(?:weeks?|wks?)', request_lower)
        if week_match:
            extracted['duration'] = int(week_match.group(1))
        else:
            month_match = re.search(r'(\d+)\s*months?', request_lower)
            if month_match:
                extracted['duration'] = int(month_match.group(1)) * 4

        if 'christmas' in request_lower and 'duration' not in extracted:
            today = datetime.now()
            christmas = datetime(today.year if today.month < 12 else today.year + 1, 12, 25)
            weeks_until = max(1, int((christmas - today).days / 7))
            if weeks_until <= 52:
                extracted['duration'] = weeks_until

        # Extract TRAINING FREQUENCY
        freq_match = re.search(r'(\d+)\s*(?:days?|times?|x)\s*(?:a|per)?\s*week', request_lower)
        if freq_match:
            extracted['frequency'] = int(freq_match.group(1))

        # Extract USER NOTES (specific preferences)
        notes_parts = []
        if 'glute' in request_lower or 'butt' in request_lower:
            notes_parts.append("glute emphasis")
        if 'chest' in request_lower:
            notes_parts.append("chest emphasis")
        if 'leg' in request_lower:
            notes_parts.append("leg emphasis")
        if 'arm' in request_lower:
            notes_parts.append("arm emphasis")
        if 'back' in request_lower:
            notes_parts.append("back emphasis")
        if 'vertical' in request_lower and 'jump' in request_lower:
            notes_parts.append("vertical jump focus")
        if 'sprint' in request_lower:
            notes_parts.append("sprint speed focus")
        if notes_parts:
            extracted['notes'] = ", ".join(notes_parts)

        # Extract SPORT
        sports = ['basketball', 'football', 'soccer', 'volleyball', 'track', 'baseball', 'powerlifting', 'weightlifting', 'crossfit']
        for sport in sports:
            if sport in request_lower:
                extracted['sport'] = sport
                break

        # Extract INJURIES
        injury_keywords = ['injury', 'injured', 'hurt', 'pain', 'bad knee', 'bad shoulder', 'back pain']
        for kw in injury_keywords:
            if kw in request_lower:
                if 'knee' in request_lower:
                    extracted['injuries'] = "knee issues mentioned"
                elif 'shoulder' in request_lower:
                    extracted['injuries'] = "shoulder issues mentioned"
                elif 'back' in request_lower:
                    extracted['injuries'] = "back issues mentioned"
                else:
                    extracted['injuries'] = "injury mentioned - needs clarification"
                break

        # Extract SESSION DURATION
        duration_match = re.search(r'(\d+)\s*(?:minute|min)\s*(?:workout|session)', request_lower)
        if duration_match:
            extracted['session_duration'] = int(duration_match.group(1))
        else:
            hour_match = re.search(r'(\d+)\s*hour\s*(?:workout|session)', request_lower)
            if hour_match:
                extracted['session_duration'] = int(hour_match.group(1)) * 60

        return extracted

    async def _enter_program_creation_mode(self, db, user_id: str, extracted_params: dict, user_request: str = "") -> str:
        """Shared logic for entering program creation mode."""
        from db.models import User

        db_user = db.query(User).filter(User.id == user_id).first()
        existing_data = {}
        if db_user:
            existing_data = {
                "height_cm": float(db_user.height_cm) if db_user.height_cm else None,
                "weight_kg": float(db_user.weight_kg) if db_user.weight_kg else None,
                "age": int(db_user.age) if db_user.age else None,
                "sex": db_user.sex
            }
            logger.info(f"[PROGRAM] Cached existing user data fields: {[k for k, v in existing_data.items() if v is not None]}")

        self.state.set("program_creation.existing_data", existing_data)

        # Store extracted parameters with precaptured_ prefix
        param_mappings = {
            'goal': 'precaptured_goal',
            'duration': 'precaptured_duration',
            'frequency': 'precaptured_frequency',
            'notes': 'precaptured_notes',
            'sport': 'precaptured_sport',
            'injuries': 'precaptured_injuries',
            'session_duration': 'precaptured_session_duration',
        }

        for key, state_key in param_mappings.items():
            if key in extracted_params:
                self.state.set(f"program_creation.{state_key}", extracted_params[key])
                logger.info(f"[PROGRAM] Pre-captured {key}: {extracted_params[key]}")

        if 'goal' in extracted_params and user_request:
            self.state.set("program_creation.precaptured_goal_raw", user_request)

        self.state.switch_mode("program_creation")
        self.state.save_state()

        logger.info("[PROGRAM] Entering program creation mode")

    @function_tool
    async def create_program(self, context: RunContext, user_request: str = ""):
        """
        Call this IMMEDIATELY when the user wants to create a program.

        IMPORTANT: Pass the user's FULL original message as user_request to enable intelligent parameter extraction.

        Args:
            user_request: The user's full original request (enables smart parameter extraction)
        """
        self._publish_visual({"type": "menu", "action": "select", "choice": "program"})
        logger.info("="*80)
        logger.info("[MAIN MENU] create_program() CALLED")
        logger.debug(f"[MAIN MENU] User request: {user_request}")
        logger.info("="*80)

        user_id = self.user_id
        # Extract program parameters from user request
        extracted_params = {}
        if user_request:
            extracted_params = self._extract_program_params_from_request(user_request)
            if extracted_params:
                logger.info(f"[PROGRAM] Extracted parameters: {extracted_params}")

        db = SessionLocal()
        try:
            # Enter program creation mode (caches user data, stores params, switches mode)
            await self._enter_program_creation_mode(db, user_id, extracted_params, user_request)

            # Handoff to ProgramCreationAgent
            await self._suppress_turn_detection()
            from agent.agents.program_creation_agent import ProgramCreationAgent
            return await self._carry_context_to(ProgramCreationAgent(state=self.state, userdata=self.userdata))

        except Exception as e:
            logger.error(f"[ERROR] Failed to enter program creation: {e}")
            return None, "Program setup failed to start. Tell the user briefly that you couldn't get it going and offer to try again — apologetic, one sentence, vary the wording."
        finally:
            db.close()

    @function_tool
    async def update_program(self, context: RunContext):
        """
        Call this when the user wants to update or modify an existing program.
        User might say: "update my program", "modify my program", "change my program"
        """
        logger.info("[MAIN MENU] User requested to update program")
        self._publish_visual({"type": "menu", "action": "select", "choice": "program"})

        user_id = self.user_id
        db = SessionLocal()
        try:
            from db.program_utils import get_program_summary_list

            programs = get_program_summary_list(db, user_id)

            if len(programs) == 0:
                return None, "Tell the user they don't have any programs yet and offer to create their first one. One or two sentences, vary the phrasing."
            elif len(programs) == 1:
                program = programs[0]
                self.state.set("program_update.selected_program_id", program["id"])
                self.state.set("program_update.selected_program_name", program["name"])
                self.state.save_state()

                logger.info(f"[PROGRAM UPDATE] User has 1 program: {program['name']} (ID: {program['id']})")

                # Handoff to ProgramCreationAgent (which handles updates too)
                await self._suppress_turn_detection()
                from agent.agents.program_creation_agent import ProgramCreationAgent
                return await self._carry_context_to(ProgramCreationAgent(state=self.state, userdata=self.userdata))
            else:
                # Multiple programs - store list and handoff to let ProgramCreationAgent handle selection
                self.state.set("program_update.available_programs", programs)
                self.state.save_state()

                logger.info(f"[PROGRAM UPDATE] User has {len(programs)} programs")

                await self._suppress_turn_detection()
                from agent.agents.program_creation_agent import ProgramCreationAgent
                return await self._carry_context_to(ProgramCreationAgent(state=self.state, userdata=self.userdata))

        except Exception as e:
            logger.error(f"[ERROR] Failed to list programs: {e}")
            return None, "Tell the user you're having trouble accessing their programs right now and offer to try again in a moment. One or two sentences, vary the phrasing."
        finally:
            db.close()

    # ===== SCHEDULE & INFO TOOLS =====

    @function_tool
    async def view_schedule(self, days_ahead: int = 7, context: RunContext = None):
        """
        Call this when the user wants to see their upcoming workout schedule.

        Args:
            days_ahead: Number of days to look ahead (default 7)
        """
        logger.info(f"[MAIN MENU] User requested schedule (next {days_ahead} days)")
        self._publish_visual({"type": "menu", "action": "select", "choice": "schedule"})

        from db.schedule_utils import get_upcoming_workouts
        from datetime import datetime

        db = SessionLocal()
        try:
            user_id = self.user_id
            workouts = get_upcoming_workouts(db, user_id, days_ahead)

            if not workouts:
                return None, f"The user has no workouts scheduled in the next {days_ahead} days. Suggest creating a new program or offer to help with other options."

            schedule_list = []
            for w in workouts:
                date_obj = datetime.fromisoformat(w['scheduled_date'])
                date_display = date_obj.strftime("%A, %B %d")
                status_suffix = ", already done" if w['completed'] else ""
                schedule_list.append(f"{date_display}: {w['workout_name']}{status_suffix}")

            schedule_text = "\n".join(schedule_list)

            return None, (
                f"The user wants to see their schedule. They have {len(workouts)} workouts in the next {days_ahead} days:\n"
                f"{schedule_text}\n\n"
                "Summarize this in two to three sentences: lead with the next workout by name and day, "
                "then how many more they have this week. Workouts marked already done are behind them. "
                "Do not read the full list or any status labels aloud unless the user asks."
            )

        except Exception as e:
            logger.exception("[SCHEDULE ERROR] Failed to load schedule")
            return None, "There was an error loading the schedule. Apologize and suggest trying again."
        finally:
            db.close()

    @function_tool
    async def view_workout_exercises(self, context: RunContext, date_text: str = "today"):
        """
        Call this when the user wants to see the exercises in their workout for a specific day.

        Args:
            date_text: The day to view (e.g., "today", "tomorrow", "monday", "next friday")
        """
        logger.info(f"[MAIN MENU] User requested exercises for: {date_text}")
        self._publish_visual({"type": "menu", "action": "select", "choice": "schedule"})

        from db.schedule_utils import get_upcoming_workouts
        from datetime import datetime, date, timedelta
        from utils.date_parser import parse_natural_date, DateParseError

        db = SessionLocal()
        try:
            user_id = self.user_id
            try:
                target_date = parse_natural_date(date_text)
            except DateParseError as e:
                return None, f"I had trouble understanding '{date_text}' as a date. Please call this function again with a clearer date like 'today', 'tomorrow', or a day of the week."

            from db.models import Schedule, Workout, WorkoutExercise, Exercise, Set
            from sqlalchemy import and_
            from sqlalchemy.orm import joinedload

            schedule = db.query(Schedule).options(
                joinedload(Schedule.workout).joinedload(Workout.workout_exercises).joinedload(WorkoutExercise.exercise),
                joinedload(Schedule.workout).joinedload(Workout.workout_exercises).joinedload(WorkoutExercise.sets)
            ).filter(
                and_(
                    Schedule.user_id == user_id,
                    Schedule.scheduled_date == target_date
                )
            ).first()

            if not schedule:
                date_str = target_date.strftime("%A, %B %d")
                return None, f"No workout scheduled for {date_str}. Suggest viewing their schedule or creating a program."

            workout = schedule.workout

            exercise_list = []
            for we in sorted(workout.workout_exercises, key=lambda x: x.order_number):
                ex_name = we.exercise.name if we.exercise else "Unknown Exercise"
                sets_count = len(we.sets)

                if we.sets:
                    reps = [s.reps for s in we.sets if s.reps]
                    if reps:
                        if min(reps) == max(reps):
                            rep_info = f"{reps[0]} reps"
                        else:
                            rep_info = f"{min(reps)}-{max(reps)} reps"
                    else:
                        rep_info = "reps not specified"
                else:
                    rep_info = "no sets"

                exercise_info = f"{ex_name}: {sets_count} sets of {rep_info}"
                if we.notes:
                    exercise_info += f" ({we.notes})"
                exercise_list.append(exercise_info)

            exercises_text = "\n".join(exercise_list)
            date_str = target_date.strftime("%A, %B %d")

            return None, (
                f"Workout for {date_str} - {workout.name}:\n\n{exercises_text}\n\n"
                "Name the exercises in flowing conversational speech with their sets and reps — "
                "never as a numbered list read aloud, never reciting the lines verbatim."
            )

        except Exception as e:
            logger.exception("[WORKOUT VIEW ERROR] Failed to load exercises")
            return None, "There was an error loading the workout details. Apologize and suggest trying again."
        finally:
            db.close()

    # ===== SCHEDULE MANAGEMENT ROUTING =====

    @function_tool
    async def manage_schedule(self, context: RunContext, user_request: str):
        """
        Call this when the user wants to modify their schedule. This includes:
        moving workouts, swapping workouts/weeks, skipping workouts, adding rest days,
        repeating workouts, applying deload weeks, vacation mode, pushing workouts forward,
        undoing changes, viewing change history, analyzing recovery, or checking training load.

        IMPORTANT: Pass the user's FULL original message as user_request.

        Args:
            user_request: The user's complete original request about schedule changes
        """
        logger.debug(f"[MAIN MENU] Schedule management requested: {user_request}")
        self._publish_visual({"type": "menu", "action": "select", "choice": "schedule"})

        intent = self._classify_schedule_intent(user_request)
        logger.info(f"[MAIN MENU] Classified schedule intent: {intent}")

        self.state.set("schedule.precaptured_intent", intent)
        self.state.set("schedule.precaptured_request", user_request)
        self.state.switch_mode("schedule")
        self.state.save_state()

        await self._suppress_turn_detection()

        self._log_function_call("manage_schedule", {"user_request": user_request, "intent": intent}, "handoff to ScheduleMaintenanceAgent")

        from agent.agents.schedule_agent import ScheduleMaintenanceAgent
        return await self._carry_context_to(ScheduleMaintenanceAgent(state=self.state, userdata=self.userdata))

    def _classify_schedule_intent(self, request: str) -> str:
        """Lightweight intent classification — fallback is 'general' (schedule agent LLM resolves)."""
        r = request.lower()
        if any(w in r for w in ['undo', 'revert', 'go back', 'nevermind']):
            return 'undo'
        if any(w in r for w in ['swap', 'switch']) and 'week' in r:
            return 'swap_weeks'
        if any(w in r for w in ['swap', 'switch']):
            return 'swap_workouts'
        if any(w in r for w in ['move', 'reschedule']) and 'remaining' not in r and 'push' not in r:
            return 'move_workout'
        if any(w in r for w in ['skip', "can't do today"]):
            return 'skip_workout'
        if any(w in r for w in ['rest day', 'add rest', 'need rest']):
            return 'add_rest_day'
        if any(w in r for w in ['repeat', 'duplicate', 'again']):
            return 'repeat_workout'
        if any(w in r for w in ['vacation', 'holiday', 'clear schedule', 'time off']):
            return 'vacation'
        if any(w in r for w in ['push', 'shift forward', 'push remaining']):
            return 'push_week'
        if any(w in r for w in ['change history', 'recent changes', 'what did i change']):
            return 'view_changes'
        if any(w in r for w in ['analyze', 'recovery analysis', 'suggest rest', 'check recovery']):
            return 'analyze_recovery'
        if any(w in r for w in ['apply rest', 'add those rest', 'apply recommendation']):
            return 'apply_rest_days'
        if any(w in r for w in ['need deload', 'should i deload', 'overtrained', 'check deload']):
            return 'check_deload'
        if any(w in r for w in ['apply deload', 'add deload']):
            return 'apply_deload'
        if any(w in r for w in ['deload', 'reduce intensity', 'lighter week']):
            return 'deload_week'
        if any(w in r for w in ['training load', 'fatigue', 'training history']):
            return 'view_training_load'
        return 'general'

    @function_tool
    async def view_progress(self, context: RunContext):
        """
        Call this when the user wants to view their progress, stats, or history.
        """
        logger.info("[MAIN MENU] User requested to view progress")
        self._publish_visual({"type": "menu", "action": "select", "choice": "progress"})
        user_id = self.state.get("user.id")
        if not user_id:
            return None, (
                "No user profile is loaded, so there is no progress data yet. "
                "Let them know their progress tracking starts once they log in "
                "and complete a workout."
            )

        def _fetch():
            from db.biomechanics_persistence import (
                get_progress_baseline,
                get_score_progress,
            )
            db = SessionLocal()
            try:
                return (
                    get_progress_baseline(db, user_id),
                    get_score_progress(db, user_id),
                )
            finally:
                db.close()

        try:
            baseline, score_rows = await asyncio.to_thread(_fetch)
        except Exception:
            logger.exception("[MAIN MENU] Progress fetch failed")
            return None, (
                "Progress data could not be loaded right now. Apologize briefly "
                "and suggest trying again in a moment."
            )

        from agent.services.progress_context import build_progress_report
        report = build_progress_report(score_rows, baseline)
        return None, (
            "Here is the user's real squat progress data:\n"
            f"{report}\n"
            "Present this conversationally in 2-4 sentences. Lead with the trend "
            "if there is one, mention their biggest opportunity, and keep it "
            "encouraging. Do not read every number."
        )

    @function_tool
    async def update_profile(self, context: RunContext):
        """
        Call this when the user wants to update their profile or settings.
        """
        logger.info("[MAIN MENU] User requested to update profile")
        self._publish_visual({"type": "menu", "action": "select", "choice": "progress"})
        return None, "The user wants to update their profile. Tell them profile updates are coming soon and offer to note down any specific changes they want in the meantime. One or two sentences, vary the phrasing."

    @function_tool
    async def shutdown(self, context: RunContext, confirmed: bool = False):
        """
        Call this when the user wants to shut down, exit, turn off, or say goodbye.
        User might say: "shut down", "turn off", "exit", "goodbye", "I'm done",
        "quit", "close", "power off", "see you later"
        The first call only asks the user to confirm. Call again with confirmed=true
        only when their very next reply clearly says yes.

        Args:
            confirmed: True only when the user just said yes to your shutdown question
        """
        # A misheard fragment ("We're gone.") once fired this tool; never shut down
        # on one utterance. The yes must come in the turn right after the question.
        user_turns = self._user_message_count()
        if not (confirmed and self._shutdown_asked_on_turn == user_turns - 1):
            self._shutdown_asked_on_turn = user_turns
            logger.info("[MAIN MENU] Shutdown requested — asking the user to confirm")
            return None, (
                "Ask the user in a few words to confirm they want you to shut down. "
                "Don't shut down or say goodbye yet. Vary the wording."
            )

        logger.info("[MAIN MENU] User confirmed shutdown")
        self._publish_visual({"type": "menu", "action": "hide"})

        # Signal main.py to initiate graceful shutdown
        self.state.set("shutdown_requested", True)
        self.state.save_state()

        self._log_function_call("shutdown", {}, "shutdown_requested")

        return None, (
            "The user wants to shut down. Say one warm, brief goodbye — "
            "one or two sentences max, different phrasing every session."
        )
