"""
ScheduleMaintenanceAgent - Dedicated agent for schedule modifications, recovery analysis, and training load.
"""

import logging

from livekit.agents import RunContext
from livekit.agents.llm import function_tool

from agent.agents.prompts.schedule_prompt import get_schedule_prompt
from agent.agents.shared.base_agent import BaseNovaAgent
from db.database import SessionLocal

logger = logging.getLogger(__name__)


class ScheduleMaintenanceAgent(BaseNovaAgent):
    """Handles all schedule modification operations: move, swap, skip, rest days, deload, etc."""

    def __init__(self, state, userdata) -> None:
        precaptured_intent = state.get("schedule.precaptured_intent")
        precaptured_request = state.get("schedule.precaptured_request")
        instructions = get_schedule_prompt(precaptured_intent, precaptured_request)
        super().__init__(state=state, userdata=userdata, instructions=instructions)

    async def on_enter(self):
        """Speak on entry: act on the request the main menu captured, or ask what to change."""
        handle = self.session.generate_reply(
            instructions=(
                "You just took over the user's schedule. If your instructions carry their "
                "request, act on it now: a brief natural preamble, then the right tool. "
                "Otherwise ask in one short question what they want to change. Vary the wording."
            )
        )
        await handle.wait_for_playout()
        self._restore_turn_detection()

    # ===== NAVIGATION =====

    @function_tool
    async def back_to_main_menu(self, context: RunContext):
        """
        Call this when the user wants to go back to the main menu, start a workout,
        create a program, or do anything outside of schedule management.
        """
        logger.info("[SCHEDULE] Returning to main menu")

        self.state.set("schedule.precaptured_intent", None)
        self.state.set("schedule.precaptured_request", None)
        self.state.switch_mode("main_menu")
        self.state.save_state()

        await self._suppress_turn_detection()

        self._log_function_call("back_to_main_menu", {}, "handoff to MainMenuAgent")

        from agent.agents.main_menu_agent import MainMenuAgent
        return await self._carry_context_to(MainMenuAgent(state=self.state, userdata=self.userdata))

    # ===== SCHEDULE MODIFICATION TOOLS =====

    @function_tool
    async def move_workout_to_date(self, context: RunContext, workout_description: str, target_date_text: str):
        """
        Move a specific workout to a new date (NO cascading).

        Args:
            workout_description: Description of workout to move (e.g., "leg day", "tuesday's workout")
            target_date_text: Natural language target date (e.g., "tomorrow", "next friday")
        """
        from db.schedule_utils import find_schedule_by_criteria, move_workout
        from utils.date_parser import parse_natural_date, get_date_description, DateParseError
        from datetime import date, timedelta

        user_id = self.user_id
        db = SessionLocal()
        try:
            try:
                target_date = parse_natural_date(target_date_text)
            except DateParseError:
                return None, f"I couldn't understand the date '{target_date_text}'. Could you say it differently?"

            source_date = None
            workout_name_hint = None

            if "today" in workout_description.lower():
                source_date = date.today()
            elif "tomorrow" in workout_description.lower():
                source_date = date.today() + timedelta(days=1)

            for day_name in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']:
                if day_name in workout_description.lower():
                    source_date = parse_natural_date(day_name)
                    break

            for hint in ['leg', 'upper', 'lower', 'push', 'pull', 'chest', 'back', 'shoulder']:
                if hint in workout_description.lower():
                    workout_name_hint = hint
                    break

            matches = find_schedule_by_criteria(db, user_id, target_date=source_date, workout_name_fragment=workout_name_hint)

            if len(matches) == 0:
                return None, f"I couldn't find a workout matching '{workout_description}'. Could you be more specific?"
            if len(matches) > 1:
                workout_list = ", ".join([f"{w.workout.name} on {w.scheduled_date}" for w in matches[:3]])
                return None, f"I found multiple workouts: {workout_list}. Which one did you mean?"

            schedule = matches[0]
            success, error_msg = move_workout(db, schedule.id, target_date)

            if not success:
                return None, f"I couldn't move that workout. {error_msg}"

            target_desc = get_date_description(target_date)
            return None, f"Done! I moved '{schedule.workout.name}' to {target_desc}. Your schedule is updated."

        except Exception as e:
            logger.exception("[ERROR] move_workout_to_date failed")
            return None, f"Sorry I ran into an issue moving that workout. Let's try again."
        finally:
            db.close()

    @function_tool
    async def swap_two_workouts(self, context: RunContext, workout1_description: str, workout2_description: str):
        """
        Swap two individual workouts by exchanging their dates.

        Args:
            workout1_description: First workout (e.g., "tuesday's workout", "leg day")
            workout2_description: Second workout (e.g., "thursday's workout", "push day")
        """
        from db.schedule_utils import find_schedule_by_criteria, swap_workouts
        from utils.date_parser import parse_natural_date
        from datetime import date, timedelta

        user_id = self.user_id
        db = SessionLocal()
        try:
            def find_workout(description: str):
                source_date = None
                workout_name_hint = None

                if "today" in description.lower():
                    source_date = date.today()
                elif "tomorrow" in description.lower():
                    source_date = date.today() + timedelta(days=1)

                for day_name in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']:
                    if day_name in description.lower():
                        source_date = parse_natural_date(day_name)
                        break

                for hint in ['leg', 'upper', 'lower', 'push', 'pull', 'chest', 'back', 'shoulder']:
                    if hint in description.lower():
                        workout_name_hint = hint
                        break

                return find_schedule_by_criteria(db, user_id, target_date=source_date, workout_name_fragment=workout_name_hint)

            matches1 = find_workout(workout1_description)
            matches2 = find_workout(workout2_description)

            if len(matches1) == 0 or len(matches2) == 0:
                return None, f"I couldn't find both workouts. Could you be more specific about which workouts to swap?"

            if len(matches1) > 1 or len(matches2) > 1:
                return None, f"I found multiple matches. Could you specify the exact dates?"

            success, error_msg = swap_workouts(db, matches1[0].id, matches2[0].id)

            if not success:
                return None, f"I couldn't swap those workouts. {error_msg}"

            return None, f"Done! I swapped '{matches1[0].workout.name}' and '{matches2[0].workout.name}'. Your schedule is updated."

        except Exception as e:
            logger.exception("[ERROR] swap_two_workouts failed")
            return None, f"Sorry I ran into an issue swapping those workouts. Let's try again."
        finally:
            db.close()

    @function_tool
    async def swap_entire_weeks(self, context: RunContext, week1_description: str, week2_description: str):
        """
        Swap ALL workouts between two weeks.

        Args:
            week1_description: First week (e.g., "this week", "next week")
            week2_description: Second week (e.g., "the week after", "next week")
        """
        from db.schedule_utils import swap_weeks
        from utils.date_parser import parse_week_range, DateParseError

        user_id = self.user_id
        db = SessionLocal()
        try:
            try:
                week1_start, week1_end = parse_week_range(week1_description)
                week2_start, week2_end = parse_week_range(week2_description)
            except DateParseError:
                return None, f"I couldn't understand those week descriptions. Could you say it like 'this week and next week'?"

            success, error_msg, swapped_pairs = swap_weeks(db, user_id, week1_start, week2_start)

            if not success:
                return None, f"I couldn't swap those weeks. {error_msg}"

            return None, f"Done! I swapped all workouts between {week1_description} and {week2_description}. {len(swapped_pairs)} workouts were moved."

        except Exception as e:
            logger.exception("[ERROR] swap_entire_weeks failed")
            return None, f"Sorry I ran into an issue swapping those weeks. Let's try again."
        finally:
            db.close()

    @function_tool
    async def skip_workout_today(self, context: RunContext, reason: str = None):
        """
        Skip today's workout (does NOT reschedule automatically).

        Args:
            reason: Optional reason for skipping (e.g., "tired", "injured", "travel")
        """
        from db.schedule_utils import get_todays_workout, skip_workout

        user_id = self.user_id
        db = SessionLocal()
        try:
            workout = get_todays_workout(db, user_id)

            if not workout:
                return None, f"you don't have a workout scheduled today. Would you like to see your upcoming schedule?"

            success, error_msg = skip_workout(db, workout["schedule_id"], reason=reason)

            if not success:
                return None, f"I couldn't skip that workout. {error_msg}"

            return None, f"No problem. I've marked today's workout as skipped. Rest up and we'll get back to it next time!"

        except Exception as e:
            logger.exception("[ERROR] skip_workout_today failed")
            return None, f"Sorry I ran into an issue. Let's try again."
        finally:
            db.close()

    @function_tool
    async def add_rest_day_and_shift(self, context: RunContext, rest_date_text: str):
        """
        Add a rest day and shift future workouts forward.

        Args:
            rest_date_text: Natural language date for rest day (e.g., "tomorrow", "friday")
        """
        from db.schedule_utils import add_rest_day
        from utils.date_parser import parse_natural_date, get_date_description, DateParseError

        user_id = self.user_id
        db = SessionLocal()
        try:
            try:
                rest_date = parse_natural_date(rest_date_text)
            except DateParseError:
                return None, f"I couldn't understand '{rest_date_text}'. Could you say it differently?"

            success, error_msg, shifted_count = add_rest_day(db, user_id, rest_date, shift_future_workouts=True)

            if not success:
                return None, f"I couldn't add a rest day. {error_msg}"

            rest_desc = get_date_description(rest_date)
            return None, f"Done! I added a rest day on {rest_desc} and pushed {shifted_count} future workouts forward by one day."

        except Exception as e:
            logger.exception("[ERROR] add_rest_day_and_shift failed")
            return None, f"Sorry I ran into an issue. Let's try again."
        finally:
            db.close()

    @function_tool
    async def repeat_workout_on_date(self, context: RunContext, workout_description: str, repeat_date_text: str):
        """
        Duplicate a workout to another date.

        Args:
            workout_description: Workout to repeat (e.g., "today's workout", "leg day")
            repeat_date_text: Date to repeat on (e.g., "friday", "next monday")
        """
        from db.schedule_utils import find_schedule_by_criteria, repeat_workout
        from utils.date_parser import parse_natural_date, get_date_description, DateParseError
        from datetime import date

        user_id = self.user_id
        db = SessionLocal()
        try:
            try:
                repeat_date = parse_natural_date(repeat_date_text)
            except DateParseError:
                return None, f"I couldn't understand '{repeat_date_text}'. Could you say it differently?"

            source_date = None
            workout_name_hint = None

            if "today" in workout_description.lower():
                source_date = date.today()

            for hint in ['leg', 'upper', 'lower', 'push', 'pull']:
                if hint in workout_description.lower():
                    workout_name_hint = hint
                    break

            matches = find_schedule_by_criteria(db, user_id, target_date=source_date, workout_name_fragment=workout_name_hint)

            if len(matches) == 0:
                return None, f"I couldn't find a workout matching '{workout_description}'."
            if len(matches) > 1:
                return None, f"I found multiple matches. Could you be more specific?"

            success, error_msg, new_schedule_id = repeat_workout(db, matches[0].id, repeat_date)

            if not success:
                return None, f"I couldn't repeat that workout. {error_msg}"

            repeat_desc = get_date_description(repeat_date)
            return None, f"Done! I added '{matches[0].workout.name}' on {repeat_desc}."

        except Exception as e:
            logger.exception("[ERROR] repeat_workout_on_date failed")
            return None, f"Sorry I ran into an issue. Let's try again."
        finally:
            db.close()

    @function_tool
    async def apply_deload_to_week(self, context: RunContext, week_description: str, intensity_percentage: int = 70):
        """
        Apply deload to a week (reduce intensity for recovery).

        Args:
            week_description: Week to deload (e.g., "this week", "next week")
            intensity_percentage: Target intensity as percentage (default 70%)
        """
        from db.schedule_utils import apply_deload_week
        from utils.date_parser import parse_week_range, DateParseError

        user_id = self.user_id
        db = SessionLocal()
        try:
            try:
                week_start, week_end = parse_week_range(week_description)
            except DateParseError:
                return None, f"I couldn't understand '{week_description}'. Could you say 'this week' or 'next week'?"

            if not (30 <= intensity_percentage <= 100):
                return None, f"Intensity should be between 30% and 100%. Did you mean {intensity_percentage}%?"

            intensity_modifier = intensity_percentage / 100.0
            success, error_msg, modified_count = apply_deload_week(db, user_id, week_start, intensity_modifier)

            if not success:
                return None, f"I couldn't apply deload. {error_msg}"

            return None, f"Done! I set {week_description} as a deload week at {intensity_percentage}% intensity. {modified_count} workouts were modified."

        except Exception as e:
            logger.exception("[ERROR] apply_deload_to_week failed")
            return None, f"Sorry I ran into an issue. Let's try again."
        finally:
            db.close()

    @function_tool
    async def clear_schedule_for_vacation(self, context: RunContext, start_date_text: str, end_date_text: str):
        """
        Clear workouts in a date range (vacation mode).

        Args:
            start_date_text: Vacation start date
            end_date_text: Vacation end date
        """
        from db.schedule_utils import clear_date_range
        from utils.date_parser import parse_natural_date, DateParseError

        user_id = self.user_id
        db = SessionLocal()
        try:
            try:
                start_date = parse_natural_date(start_date_text)
                end_date = parse_natural_date(end_date_text)
            except DateParseError:
                return None, f"I couldn't understand those dates. Could you say them differently?"

            success, error_msg, cleared_count = clear_date_range(db, user_id, start_date, end_date, preserve_completed=True)

            if not success:
                return None, f"I couldn't clear that range. {error_msg}"

            return None, f"Done! I cleared {cleared_count} workouts from {start_date} to {end_date}. Enjoy your break!"

        except Exception as e:
            logger.exception("[ERROR] clear_schedule_for_vacation failed")
            return None, f"Sorry I ran into an issue. Let's try again."
        finally:
            db.close()

    @function_tool
    async def push_remaining_week_forward(self, context: RunContext, days: int = 1):
        """
        Push remaining workouts this week forward by N days.

        Args:
            days: Number of days to shift forward (default: 1)
        """
        from db.schedule_utils import reschedule_remaining_week

        user_id = self.user_id
        db = SessionLocal()
        try:
            if not (1 <= days <= 7):
                return None, f"I can only shift workouts by 1-7 days. Did you mean {days} days?"

            success, error_msg, rescheduled_count = reschedule_remaining_week(db, user_id, days_offset=days)

            if not success:
                return None, f"I couldn't reschedule those workouts. {error_msg}"

            return None, f"Done! I pushed {rescheduled_count} remaining workouts forward by {days} day{'s' if days > 1 else ''}."

        except Exception as e:
            logger.exception("[ERROR] push_remaining_week_forward failed")
            return None, f"Sorry I ran into an issue. Let's try again."
        finally:
            db.close()

    @function_tool
    async def undo_last_schedule_change(self, context: RunContext):
        """
        Undo the last schedule change made by the user.
        """
        from db.schedule_history import undo_last_change

        user_id = self.user_id
        db = SessionLocal()
        try:
            success, error_msg = undo_last_change(db, user_id)

            if not success:
                return None, f"{error_msg}"

            return None, f"Done! I've undone your last change."

        except Exception as e:
            logger.exception("[ERROR] undo_last_schedule_change failed")
            return None, f"Sorry I ran into an issue undoing that change. Let's try again."
        finally:
            db.close()

    @function_tool
    async def view_schedule_change_history(self, context: RunContext, limit: int = 5):
        """
        View recent schedule changes.

        Args:
            limit: Number of recent changes to show (default: 5, max: 10)
        """
        from db.schedule_history import get_recent_changes, format_change_for_display

        user_id = self.user_id
        db = SessionLocal()
        try:
            if not (1 <= limit <= 10):
                limit = 5

            changes = get_recent_changes(db, user_id, limit=limit)

            if not changes:
                return None, "You haven't made any schedule changes yet."

            response = f"The user's {len(changes)} most recent schedule changes, newest first:\n"
            for change in changes:
                formatted = format_change_for_display(change)
                response += f"{formatted}\n"
            response += (
                "Present this conversationally in 2-3 sentences — lead with the "
                "most recent change, group similar ones together, and never read "
                "the list aloud item by item."
            )

            return None, response.strip()

        except Exception as e:
            logger.exception("[ERROR] view_schedule_change_history failed")
            return None, f"Sorry I ran into an issue retrieving your change history."
        finally:
            db.close()

    @function_tool
    async def analyze_schedule_for_recovery(self, context: RunContext):
        """
        Analyze the user's schedule for recovery issues and suggest rest days.
        """
        from db.recovery_analysis import analyze_schedule_recovery, format_recommendation_for_display

        user_id = self.user_id
        db = SessionLocal()
        try:
            analysis = analyze_schedule_recovery(db, user_id)

            response = f"Recovery analysis of the user's schedule:\n{analysis['analysis_summary']}\n"

            if analysis["recommendations"]:
                for rec in analysis["recommendations"][:3]:
                    formatted = format_recommendation_for_display(rec)
                    response += f"{formatted}\n"
                response += (
                    "Present this conversationally in 2-3 sentences — lead with "
                    "the overall recovery picture, mention at most two specific "
                    "concerns, and never read the list or labels aloud. Then ask "
                    "if they want the recommended rest days added to their schedule."
                )
            else:
                response += (
                    "No rest days are needed. Tell them in one short sentence that "
                    "their schedule looks good for recovery."
                )

            return None, response.strip()

        except Exception as e:
            logger.exception("[ERROR] analyze_schedule_for_recovery failed")
            return None, f"Sorry I ran into an issue analyzing your schedule."
        finally:
            db.close()

    @function_tool
    async def apply_recommended_rest_days(self, context: RunContext, shift_future_workouts: bool = True):
        """
        Apply the recommended rest days from schedule analysis.

        Args:
            shift_future_workouts: Whether to push future workouts forward (default: True)
        """
        from db.recovery_analysis import apply_all_recommended_rest_days

        user_id = self.user_id
        db = SessionLocal()
        try:
            success, error, added_count = apply_all_recommended_rest_days(
                db, user_id, max_rest_days=3, shift_future_workouts=shift_future_workouts
            )

            if not success:
                return None, f"{error}"

            shift_msg = " and shifted future workouts" if shift_future_workouts else ""
            return None, f"Done! I added {added_count} rest day{'s' if added_count != 1 else ''}{shift_msg}. Your recovery should be much better now!"

        except Exception as e:
            logger.exception("[ERROR] apply_recommended_rest_days failed")
            return None, f"Sorry I ran into an issue adding those rest days."
        finally:
            db.close()

    @function_tool
    async def check_if_deload_needed(self, context: RunContext):
        """
        Check if the user needs a deload week based on training load analysis.
        """
        from db.training_load import check_deload_recommendation

        user_id = self.user_id
        db = SessionLocal()
        try:
            needs_deload, recommendation, reason = check_deload_recommendation(db, user_id)

            if not needs_deload:
                return None, f"Good news! You don't need a deload right now. {reason}"

            response = "The user's training load analysis recommends a deload week:\n"
            response += f"Fatigue score {recommendation.get('fatigue_score', 'unknown')} out of 100.\n"
            response += (
                f"Recommended week {recommendation['week_start'].strftime('%b %d')} "
                f"to {recommendation['week_end'].strftime('%b %d')} at "
                f"{int(recommendation['intensity_modifier'] * 100)} percent of normal intensity.\n"
            )
            for r in recommendation['trigger_reasons']:
                response += f"{r}\n"
            response += (
                "Present this conversationally in 2-3 sentences — lead with the "
                "deload recommendation, mention at most two or three numbers, and "
                "never read the list or labels aloud. Then ask if they want the "
                "deload week applied to their schedule."
            )

            return None, response.strip()

        except Exception as e:
            logger.exception("[ERROR] check_if_deload_needed failed")
            return None, f"Sorry I ran into an issue checking your training load."
        finally:
            db.close()

    @function_tool
    async def apply_deload_week_recommendation(self, context: RunContext):
        """
        Apply the recommended deload week from training load analysis.
        """
        from db.training_load import check_deload_recommendation, apply_deload_recommendation

        user_id = self.user_id
        db = SessionLocal()
        try:
            needs_deload, recommendation, reason = check_deload_recommendation(db, user_id)

            if not needs_deload:
                return None, f"There's no deload recommendation right now. {reason}"

            success, error = apply_deload_recommendation(db, user_id, recommendation)

            if not success:
                return None, f"I couldn't apply the deload week. {error}"

            intensity_pct = int(recommendation['intensity_modifier'] * 100)
            week_str = recommendation['week_start'].strftime('%b %d')

            return None, f"Done! I've applied a {intensity_pct}% deload week starting {week_str}. Focus on recovery and lighter training this week!"

        except Exception as e:
            logger.exception("[ERROR] apply_deload_week_recommendation failed")
            return None, f"Sorry I ran into an issue applying the deload week."
        finally:
            db.close()

    @function_tool
    async def view_training_load_history(self, context: RunContext, weeks: int = 4):
        """
        View recent training load metrics and fatigue trends.

        Args:
            weeks: Number of weeks to show (default: 4)
        """
        from db.models import TrainingLoadMetrics
        from sqlalchemy import desc

        user_id = self.user_id
        db = SessionLocal()
        try:
            if not (1 <= weeks <= 12):
                weeks = 4

            metrics = db.query(TrainingLoadMetrics).filter(
                TrainingLoadMetrics.user_id == user_id
            ).order_by(desc(TrainingLoadMetrics.week_start_date)).limit(weeks).all()

            if not metrics:
                return None, "I don't have any training load data for you yet. Complete some workouts and I'll start tracking!"

            response = f"The user's training load for the past {len(metrics)} weeks, most recent first:\n"

            for m in metrics:
                week_str = m.week_start_date.strftime('%b %d')
                line = (
                    f"Week of {week_str} — {m.workouts_completed} workouts, "
                    f"{m.total_sets} total sets, {float(m.total_volume_kg):.0f} kg volume"
                )
                if m.avg_rpe:
                    line += f", average RPE {float(m.avg_rpe):.1f} out of 10"
                if m.fatigue_score:
                    line += f", fatigue score {float(m.fatigue_score):.1f} out of 100"
                if m.velocity_decline_percent:
                    line += f", velocity decline {float(m.velocity_decline_percent):.1f} percent"
                response += line + "\n"

            response += (
                "Present this conversationally in 2-3 sentences — lead with the "
                "overall trend, mention at most two or three numbers, and never "
                "read the list or labels aloud."
            )

            return None, response.strip()

        except Exception as e:
            logger.exception("[ERROR] view_training_load_history failed")
            return None, f"Sorry I ran into an issue retrieving your training load."
        finally:
            db.close()
