"""
ProgramCreationAgent - Handles program creation (10-question capture flow) and program updates
"""

import asyncio
import logging
import os
import time
from pathlib import Path

from livekit.agents import RunContext
from livekit.agents.llm import function_tool

from agent.agents.prompts import get_program_creation_prompt
from agent.agents.shared.base_agent import BaseNovaAgent, is_summary_item
from agent.agents.shared.unit_conversion import normalize_height_to_cm, normalize_weight_to_kg, categorize_goal
from agent.agents.shared.helpers import (
    build_program_generation_payload,
    get_recommended_duration,
    normalize_fitness_level,
    normalize_sex,
    service_headers,
    should_enable_vbt,
)
from db.database import SessionLocal

logger = logging.getLogger(__name__)

# Context summarization constants
MAX_CONTEXT_TOKENS = 28672
SUMMARY_TRIGGER_RATIO = 0.70
SUMMARY_TRIGGER_TOKENS = int(MAX_CONTEXT_TOKENS * SUMMARY_TRIGGER_RATIO)
KEEP_LAST_TURNS = 4
SUMMARY_MODEL = os.getenv("CONTEXT_SUMMARY_MODEL", "gpt-4o-mini")

# Seconds of build wait before speaking a single mid-wait filler line
MID_BUILD_FILLER_DELAY_S = 45.0


class ProgramCreationAgent(BaseNovaAgent):
    """Handles program creation (10-question capture) and program updates."""

    def __init__(self, state, userdata) -> None:
        instructions = self._build_instructions(state)
        super().__init__(state=state, userdata=userdata, instructions=instructions)

    def _build_instructions(self, state) -> str:
        """Build instructions based on whether we're creating or updating a program."""
        # Check if we're in update mode
        if state.get("program_update.selected_program_id"):
            program_name = state.get("program_update.selected_program_name", "your program")
            name = state.get_user().get("name", "there")
            return (
                f"You are helping the user update their '{program_name}' program. "
                f"Ask what they want to change about the program, then call capture_program_change_request() "
                f"with their description. Be conversational and helpful."
            )

        # Creation mode - use the full prompt
        return self._get_program_creation_instructions(state)

    def _get_program_creation_instructions(self, state) -> str:
        """Build program creation instructions with existing data and pre-captured params."""
        user_id = state.get_user().get("id")
        name = state.get_user().get("name", "there")

        # Try to get cached existing data from state first
        existing_data = state.get("program_creation.existing_data")

        if existing_data is None:
            logger.info("[PROGRAM] No cached user data found, querying database")
            existing_data = {}
            db = SessionLocal()
            try:
                from db.models import User
                db_user = db.query(User).filter(User.id == user_id).first()
                if db_user:
                    existing_data = {
                        "height_cm": db_user.height_cm,
                        "weight_kg": db_user.weight_kg,
                        "age": db_user.age,
                        "sex": db_user.sex
                    }
            except Exception as e:
                logger.info(f"[PROGRAM] Error checking existing user data: {e}")
            finally:
                db.close()

        # Gather pre-captured parameters from state
        precaptured_params = {}
        if state.get("program_creation.precaptured_goal"):
            precaptured_params["goal"] = state.get("program_creation.precaptured_goal")
            precaptured_params["goal_raw"] = state.get("program_creation.precaptured_goal_raw", "")
        if state.get("program_creation.precaptured_duration"):
            precaptured_params["duration"] = state.get("program_creation.precaptured_duration")
        if state.get("program_creation.precaptured_frequency"):
            precaptured_params["frequency"] = state.get("program_creation.precaptured_frequency")
        if state.get("program_creation.precaptured_notes"):
            precaptured_params["notes"] = state.get("program_creation.precaptured_notes")
        if state.get("program_creation.precaptured_sport"):
            precaptured_params["sport"] = state.get("program_creation.precaptured_sport")
        if state.get("program_creation.precaptured_injuries"):
            precaptured_params["injuries"] = state.get("program_creation.precaptured_injuries")
        if state.get("program_creation.precaptured_session_duration"):
            precaptured_params["session_duration"] = state.get("program_creation.precaptured_session_duration")

        return get_program_creation_prompt(existing_data, precaptured_params,
                                           user_name=state.get("user.name") or "")

    async def on_enter(self):
        """Generate greeting based on creation vs update mode."""
        if self.state.get("program_update.selected_program_id"):
            program_name = self.state.get("program_update.selected_program_name", "your program")
            await self._say(
                f"You're helping the user update their '{program_name}' program. Ask what they'd like to change about it. Be conversational."
            )
        else:
            await self._say(
                f"You're helping the user create a new workout program. Start by asking the first question based on what data you already have. Follow the program creation flow."
            )

    # ===== CONTEXT SUMMARIZATION =====

    def _items_to_text(self, items: list) -> str:
        """Convert chat items to readable text format for summarization."""
        lines = []
        for item in items:
            role = item.role.upper() if hasattr(item, 'role') else 'UNKNOWN'

            text = ""
            if hasattr(item, 'content'):
                if isinstance(item.content, str):
                    text = item.content
                elif isinstance(item.content, list):
                    for content_item in item.content:
                        if isinstance(content_item, dict):
                            text += content_item.get('text', '') or content_item.get('transcript', '')
                        elif hasattr(content_item, 'text'):
                            text += content_item.text or ''
            elif hasattr(item, 'text'):
                text = item.text or ''

            if text.strip():
                lines.append(f"{role}: {text.strip()}")

        return "\n".join(lines)

    def _build_fallback_summary(self) -> str:
        """Build a simple summary from agent state if LLM call fails."""
        try:
            mode = self.state.get_mode()
            parts = []

            if mode == "program_creation":
                height = self.state.get("program_creation.height_cm")
                weight = self.state.get("program_creation.weight_kg")
                age = self.state.get("program_creation.age")
                sex = self.state.get("program_creation.sex")
                goal = self.state.get("program_creation.goal_raw")
                experience = self.state.get("program_creation.experience_level")
                equipment = self.state.get("program_creation.equipment_access")
                schedule = self.state.get("program_creation.days_per_week")
                injuries = self.state.get("program_creation.injuries_limitations")

                if height or weight:
                    parts.append(f"User is {height}cm tall, weighing {weight}kg" if height and weight else f"Height/weight: {height or weight}")
                if age or sex:
                    parts.append(f"{age} year old {sex}" if age and sex else f"{age or sex}")
                if goal:
                    parts.append(f"Goal: {goal}")
                if experience:
                    parts.append(f"Experience: {experience}")
                if equipment:
                    parts.append(f"Equipment: {equipment}")
                if schedule:
                    parts.append(f"Training {schedule} days per week")
                if injuries:
                    parts.append(f"Injuries/limitations: {injuries}")

            if parts:
                return "Collected data: " + ". ".join(parts) + "."
            else:
                return "Conversation in progress. Some user information has been collected."

        except Exception as e:
            logger.info(f"[SUMMARY] Fallback summary generation failed: {e}")
            return "Fitness consultation in progress."

    async def _generate_conversation_summary(self, items: list) -> str | None:
        """Call gpt-4o-mini to generate a 2-3 sentence summary of conversation items."""
        try:
            conversation_text = self._items_to_text(items)

            if not conversation_text.strip():
                logger.info("[SUMMARY] No conversation text to summarize")
                return None

            logger.info(f"[SUMMARY] Generating summary for {len(conversation_text)} characters of conversation...")

            if self.userdata.openai_client is None:
                from openai import OpenAI
                self.userdata.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            client = self.userdata.openai_client

            response = await asyncio.to_thread(
                lambda: client.chat.completions.create(
                    model=SUMMARY_MODEL,
                    temperature=0.3,
                    max_tokens=200,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a summarization assistant for a fitness consultation. "
                                "Summarize the conversation in 2-3 concise sentences. "
                                "Focus on data collected: height, weight, age, sex, fitness goals, "
                                "experience level, equipment access, training schedule, and injuries/limitations. "
                                "Be specific with numbers and measurements. "
                                "Do not include pleasantries, greetings, or filler words."
                            )
                        },
                        {
                            "role": "user",
                            "content": f"Summarize this fitness consultation:\n\n{conversation_text}"
                        }
                    ]
                )
            )

            summary = response.choices[0].message.content.strip()
            logger.info(f"[SUMMARY] Generated summary ({len(summary)} chars): {summary[:100]}...")

            from agent.core.session_logger import SessionLogger
            session_logger = SessionLogger.get_instance()

            usage = response.usage
            session_logger.log_llm_call(
                component="context_summarization",
                model=SUMMARY_MODEL,
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                details={
                    "summary_length": len(summary),
                    "items_summarized": len(items),
                    "summary_number": self.userdata.summary_count
                }
            )

            return summary

        except Exception as e:
            logger.info(f"[SUMMARY] LLM summary generation failed: {e}")
            return None

    async def _update_context_with_summary(
        self, agent, summary_text: str, system_items: list, recent_items: list, old_items: list
    ) -> None:
        """Update the agent's chat context with the summary."""
        from livekit.agents import llm

        new_ctx = llm.ChatContext.empty()

        for item in system_items:
            new_ctx.items.append(item)

        summary_message = llm.ChatMessage(
            role="system",
            content=[f"[CONVERSATION SUMMARY] {summary_text}"]
        )
        new_ctx.items.append(summary_message)

        for item in recent_items:
            new_ctx.items.append(item)

        await agent.update_chat_ctx(new_ctx)

        logger.info(f"[SUMMARY] Context updated: {len(system_items)} system + 1 summary + {len(recent_items)} recent items")
        logger.info(f"[SUMMARY] Removed {len(old_items)} old items")

    async def _summarize_and_prune_context(self) -> None:
        """Async summarization: compress old conversation turns into a summary."""
        if self.userdata.is_summarizing:
            logger.info("[SUMMARY] Summarization already in progress, skipping")
            return

        self.userdata.is_summarizing = True
        logger.info(f"[SUMMARY] Starting async summarization...")

        try:
            agent = self

            if not agent or not hasattr(agent, 'chat_ctx'):
                logger.info("[SUMMARY] No agent or chat context available for summarization")
                return

            chat_ctx = agent.chat_ctx
            items = list(chat_ctx.items)

            if len(items) <= KEEP_LAST_TURNS + 1:
                logger.info(f"[SUMMARY] Not enough items to summarize ({len(items)} items)")
                return

            system_items = [
                item for item in items
                if hasattr(item, 'role') and item.role == "system" and not is_summary_item(item)
            ]
            non_system_items = [item for item in items if not hasattr(item, 'role') or item.role != "system"]

            if len(non_system_items) <= KEEP_LAST_TURNS:
                logger.info(f"[SUMMARY] Not enough non-system items to summarize ({len(non_system_items)} items)")
                return

            old_items = non_system_items[:-KEEP_LAST_TURNS]
            recent_items = non_system_items[-KEEP_LAST_TURNS:]

            logger.info(f"[SUMMARY] Splitting context: {len(old_items)} old items to summarize, {len(recent_items)} recent items to keep")

            summary_text = await self._generate_conversation_summary(old_items)

            if not summary_text:
                logger.info("[SUMMARY] Failed to generate LLM summary, using fallback")
                summary_text = self._build_fallback_summary()

            await self._update_context_with_summary(
                agent=agent,
                summary_text=summary_text,
                system_items=system_items,
                recent_items=recent_items,
                old_items=old_items
            )

            self.userdata.summary_count += 1
            logger.info(f"[SUMMARY] Summarization complete (summary #{self.userdata.summary_count})")

        except Exception as e:
            logger.exception("[SUMMARY] Summarization failed")
        finally:
            self.userdata.is_summarizing = False

    async def check_and_summarize_if_needed(self, context: RunContext = None) -> None:
        """Check if summarization is needed based on message count and trigger if necessary."""
        try:
            if not hasattr(self, 'chat_ctx'):
                return

            chat_ctx = self.chat_ctx
            items = list(chat_ctx.items) if hasattr(chat_ctx, 'items') else []

            MESSAGE_COUNT_THRESHOLD = 15

            if len(items) > MESSAGE_COUNT_THRESHOLD and not self.userdata.is_summarizing:
                logger.info(f"[SUMMARY] Message count ({len(items)}) exceeds threshold ({MESSAGE_COUNT_THRESHOLD})")
                self._summary_task = asyncio.create_task(self._summarize_and_prune_context())

        except Exception as e:
            logger.info(f"[SUMMARY] Error checking for summarization: {e}")

    async def _truncate_conversation_history(self, context: RunContext, max_items: int = 10):
        """Truncate conversation history to prevent context window exhaustion."""
        try:
            from livekit.agents import llm

            agent = context.session.current_agent
            current_ctx = agent.chat_ctx
            items = current_ctx.items if hasattr(current_ctx, 'items') else []
            messages_before = len(items)

            truncated_items = items[-max_items:] if len(items) > max_items else items

            new_ctx = llm.ChatContext.empty()
            for item in truncated_items:
                new_ctx.insert(item)

            await agent.update_chat_ctx(new_ctx)

            messages_removed = messages_before - len(truncated_items)
            if messages_removed > 0:
                logger.info(f"[CONTEXT] Truncated: removed {messages_removed} messages, kept last {len(truncated_items)}")
            else:
                logger.info(f"[CONTEXT] No truncation needed: {messages_before} messages <= {max_items} limit")

        except Exception as e:
            logger.info(f"[CONTEXT] Warning: Failed to truncate conversation: {e}")

    # ===== PROGRAM CREATION TOOLS =====

    @function_tool
    async def capture_height_weight(self, context: RunContext, height_value: str = None, weight_value: str = None):
        """
        Call this when the user provides both height and weight together.
        Can also be called without arguments to use existing DB values.

        Args:
            height_value: The height as spoken by the user (e.g., "5'10\"", "175 cm"), or None to use DB value
            weight_value: The weight as spoken by the user (e.g., "185 pounds", "80 kg"), or None to use DB value
        """
        user_id = self.user_id
        db = SessionLocal()
        try:
            from db.models import User
            db_user = db.query(User).filter(User.id == user_id).first()

            if height_value is None and weight_value is None and db_user:
                if db_user.height_cm and db_user.weight_kg:
                    logger.info(f"[PROGRAM] Using existing DB values: height={db_user.height_cm} cm, weight={db_user.weight_kg} kg")
                    self.state.set("program_creation.height_cm", float(db_user.height_cm))
                    self.state.set("program_creation.weight_kg", float(db_user.weight_kg))
                    return None, "Height and weight loaded. Now call capture_age_sex() with no arguments, then ask about their fitness goal."
                else:
                    logger.info(f"[PROGRAM] No complete height/weight data in DB - need to ask user")
                    return None, "No height and weight on file. Ask: 'What's your height and weight?'"

            if height_value:
                logger.info(f"[PROGRAM] Capturing new height and weight: {height_value}, {weight_value}")

                height_cm = normalize_height_to_cm(height_value)
                if height_cm is None or height_cm < 50 or height_cm > 300:
                    return None, f"That height doesn't seem right. Say: 'Hmm, that height doesn't sound quite right. Can you tell me your height again? For example, five foot nine, or 175 centimeters.' Keep it friendly."

                weight_kg = normalize_weight_to_kg(weight_value)
                if weight_kg is None or weight_kg < 30 or weight_kg > 300:
                    return None, f"That weight doesn't seem right. Say: 'Hmm, that weight doesn't sound quite right. Can you tell me your weight again? For example, 185 pounds or 80 kilograms.' Keep it friendly."

                if db_user:
                    db_user.height_cm = height_cm
                    db_user.weight_kg = weight_kg
                    db.commit()
                    logger.info(f"[PROGRAM] Saved to database: height={height_cm} cm, weight={weight_kg} kg")

                self.state.set("program_creation.height_cm", height_cm)
                self.state.set("program_creation.weight_kg", weight_kg)

                logger.info(f"[PROGRAM] Height: {height_cm} cm, Weight: {weight_kg} kg")

                return None, "Captured. Immediately ask the next question."

        except Exception as e:
            logger.error(f"[ERROR] Failed to handle height/weight: {e}")
            db.rollback()
        finally:
            db.close()

        return None, f"Error: Could not capture height and weight. Please provide them again."

    @function_tool
    async def capture_age_sex(self, context: RunContext, age: int = None, sex: str = None):
        """
        Call this when the user provides both age and sex together.
        Can also be called without arguments to use existing DB values.

        Args:
            age: User's age in years, or None to use DB value
            sex: "male", "female", "M", "F", etc., or None to use DB value
        """
        user_id = self.user_id
        db = SessionLocal()
        try:
            from db.models import User
            db_user = db.query(User).filter(User.id == user_id).first()

            if age is None and sex is None and db_user:
                if db_user.age and db_user.sex:
                    logger.info("[PROGRAM] Using existing DB values for age and sex")
                    self.state.set("program_creation.age", int(db_user.age))
                    self.state.set("program_creation.sex", db_user.sex)
                    return None, "Age and sex loaded. Stats confirmation complete. Immediately ask about their fitness goal."
                else:
                    logger.info(f"[PROGRAM] No complete age/sex data in DB - need to ask user")
                    return None, "No age and sex on file. Ask: 'How old are you, and are you male or female?'"

            if age is not None:
                logger.info("[PROGRAM] Capturing new age and sex")

                if age < 13 or age > 100:
                    return None, f"That age seems unusual. Say: 'Hmm, that age doesn't seem right. How old are you?' Keep it friendly."

                if sex is None:
                    return None, "I got your age but I still need your sex. Say: 'Are you male or female?' Keep it simple."

                sex_normalized = normalize_sex(sex)
                if sex_normalized is None:
                    return None, f"I didn't catch the sex. Say: 'Sorry, are you male or female?' Keep it simple."

                if db_user:
                    db_user.age = age
                    db_user.sex = sex_normalized
                    db.commit()
                    logger.info("[PROGRAM] Saved age and sex to database")

                self.state.set("program_creation.age", age)
                self.state.set("program_creation.sex", sex_normalized)

                logger.info("[PROGRAM] Age and sex captured")

                await self.check_and_summarize_if_needed(context)

                return None, "Captured. Immediately ask the next question."

        except Exception as e:
            logger.error(f"[ERROR] Failed to handle age/sex: {e}")
            db.rollback()
        finally:
            db.close()

        return None, f"Error: Could not capture age and sex. Please provide them again."

    @function_tool
    async def capture_age(self, context: RunContext, age: int):
        """
        Call this when the user provides their age.

        Args:
            age: User's age in years
        """
        logger.info(f"[PROGRAM] Capturing age: {age}")

        if age < 13 or age > 100:
            return None, f"That age seems unusual. Say: 'Hmm, that doesn't seem right. How old are you?' Keep it friendly."

        self.state.set("program_creation.age", age)
        logger.info(f"[PROGRAM] Age set to: {age}")

        return None, "Captured. Immediately ask the next question."

    @function_tool
    async def capture_sex(self, context: RunContext, sex: str):
        """
        Call this when the user provides their biological sex.

        Args:
            sex: "male", "female", "M", "F", etc.
        """
        logger.info("[PROGRAM] Capturing sex")

        sex_normalized = normalize_sex(sex)
        if sex_normalized is None:
            return None, f"I didn't catch that. Say: 'Sorry, are you male or female?' Keep it simple."

        self.state.set("program_creation.sex", sex_normalized)
        logger.info("[PROGRAM] Sex captured")

        return None, "Captured. Immediately ask the next question."

    @function_tool
    async def capture_goal(self, context: RunContext, goal_description: str):
        """
        Call this when the user describes their fitness goal.
        Accepts free-form input and categorizes it into power, strength, or hypertrophy focus.

        Args:
            goal_description: The user's goal as they described it
        """
        logger.info(f"[PROGRAM] Capturing goal: {goal_description}")

        height_cm = self.state.get("program_creation.height_cm")
        weight_kg = self.state.get("program_creation.weight_kg")
        age = self.state.get("program_creation.age")
        sex = self.state.get("program_creation.sex")

        existing_data = self.state.get("program_creation.existing_data", {})
        has_existing_stats = (existing_data.get("height_cm") and existing_data.get("weight_kg") and
                             existing_data.get("age") and existing_data.get("sex"))

        if not (height_cm and weight_kg and age and sex) and not has_existing_stats:
            logger.error(f"[ERROR] Goal asked before prerequisites!")
            return None, f"ERROR: You MUST ask for height/weight (Question 1) and age/sex (Question 2) BEFORE asking about goals (Question 3). Go back and ask Questions 1 and 2 first!"

        goal_category = categorize_goal(goal_description)

        self.state.set("program_creation.goal_raw", goal_description)
        self.state.set("program_creation.goal_category", goal_category)

        logger.info(f"[PROGRAM] Goal categorized as: {goal_category}")

        if goal_category == "power":
            confirmation = "explosiveness and athletic performance"
        elif goal_category == "strength":
            confirmation = "building maximum strength"
        else:
            confirmation = "building muscle and aesthetics"

        self.state.set("program_creation.goal_confirmation", confirmation)
        self.state.set("program_creation.recommended_duration", get_recommended_duration(goal_category))

        return None, "Goal captured. Immediately ask the next question based on what's missing in state."

    @function_tool
    async def capture_program_duration(self, context: RunContext, duration_weeks: int):
        """
        Call this when the user specifies how long they want their program to be.

        Args:
            duration_weeks: Number of weeks for the program (e.g., 8, 12, 16)
        """
        logger.info(f"[PROGRAM] Capturing program duration: {duration_weeks} weeks")

        if duration_weeks < 2 or duration_weeks > 52:
            return None, f"Invalid duration. Say something like: 'Hmm, {duration_weeks} weeks seems a bit off. Most programs work best between 4 and 16 weeks. How long would you like your program to be?' Keep it helpful."

        self.state.set("program_creation.duration_weeks", duration_weeks)
        logger.info(f"[PROGRAM] Duration set to: {duration_weeks} weeks")

        return None, "Captured. Immediately ask the next question."

    @function_tool
    async def capture_training_frequency(self, context: RunContext, days_per_week: int):
        """
        Call this when the user specifies how many days per week they can train.

        Args:
            days_per_week: Number of training days per week (e.g., 3, 4, 5)
        """
        logger.info(f"[PROGRAM] Capturing training frequency: {days_per_week} days/week")

        if days_per_week < 1 or days_per_week > 7:
            return None, f"Invalid frequency. Say something like: 'That doesn't sound quite right. How many days per week can you realistically train? Something between 2 and 6 days works best for most people.' Keep it supportive."

        self.state.set("program_creation.days_per_week", days_per_week)
        logger.info(f"[PROGRAM] Frequency set to: {days_per_week} days/week")

        await self.check_and_summarize_if_needed(context)

        return None, "Training frequency captured. Immediately ask the next question."

    @function_tool
    async def capture_session_duration(self, context: RunContext, duration_minutes: int):
        """
        Call this when the user specifies session duration.

        Args:
            duration_minutes: Session duration in minutes (e.g., 60, 90, 45)
        """
        logger.info(f"[PROGRAM] Capturing session duration: {duration_minutes} minutes")

        if duration_minutes < 20 or duration_minutes > 180:
            return None, f"That seems unusual. Say: 'Hmm, {duration_minutes} minutes seems a bit off. Most sessions are between 30 and 120 minutes. How much time do you realistically have?' Keep it supportive."

        self.state.set("program_creation.session_duration", duration_minutes)
        logger.info(f"[PROGRAM] Session duration set to: {duration_minutes} minutes")

        return None, "Captured. Immediately ask the next question."

    @function_tool
    async def capture_injury_history(self, context: RunContext, injury_description: str):
        """
        Call this when the user describes injury history.

        Args:
            injury_description: Description of injuries or "none"
        """
        logger.info(f"[PROGRAM] Capturing injury history: {injury_description}")

        self.state.set("program_creation.injury_history", injury_description)
        logger.info(f"[PROGRAM] Injury history saved")

        return None, "Captured. Immediately ask the next question."

    @function_tool
    async def capture_specific_sport(self, context: RunContext, sport_name: str):
        """
        Call this when the user specifies a sport they're training for.

        Args:
            sport_name: Name of sport (e.g., "basketball", "powerlifting") or "none"
        """
        logger.info(f"[PROGRAM] Capturing specific sport: {sport_name}")

        sport_normalized = sport_name.lower().strip()
        if sport_normalized in ["no", "nothing", "general", "general fitness", "just fitness"]:
            sport_normalized = "none"

        self.state.set("program_creation.specific_sport", sport_normalized)
        logger.info(f"[PROGRAM] Specific sport set to: {sport_normalized}")

        return None, "Captured. Immediately ask the next question."

    @function_tool
    async def capture_training_season(self, context: RunContext, training_season: str):
        """
        Call this when the user specifies their current training season.
        Only ask this if the user plays a sport — skip for general fitness.

        Args:
            training_season: "off_season", "pre_season", "in_season", "post_season", or "none"
        """
        logger.info(f"[PROGRAM] Capturing training season: {training_season}")

        season = training_season.lower().strip()
        valid_seasons = {"off_season", "pre_season", "in_season", "post_season"}
        if season not in valid_seasons:
            season = None  # Will use off_season behavior by default

        self.state.set("program_creation.training_season", season)
        logger.info(f"[PROGRAM] Training season set to: {season}")

        return None, "Captured. Immediately ask the next question."

    @function_tool
    async def capture_games_per_week(self, context: RunContext, games_per_week: int):
        """
        Call this when the user specifies how many games or competitions they have per week.
        Only ask this if the user is in-season. Skip otherwise.

        Args:
            games_per_week: Number of games/competitions per week (0-7)
        """
        logger.info(f"[PROGRAM] Capturing games per week: {games_per_week}")

        games = max(0, min(7, games_per_week))
        self.state.set("program_creation.games_per_week", games)
        logger.info(f"[PROGRAM] Games per week set to: {games}")

        return None, "Captured. Immediately ask the next question."

    @function_tool
    async def capture_user_notes(self, context: RunContext, notes: str):
        """
        Call this when the user provides additional notes or preferences.

        Args:
            notes: User's additional notes/preferences or "none"
        """
        logger.info(f"[PROGRAM] Capturing user notes: {notes}")

        self.state.set("program_creation.user_notes", notes)
        logger.info(f"[PROGRAM] User notes saved")

        await self.check_and_summarize_if_needed(context)

        return None, "Captured. Immediately ask the next question."

    @function_tool
    async def capture_fitness_level(self, context: RunContext, fitness_level: str):
        """
        Call this when the user describes their fitness level.

        Args:
            fitness_level: The user's fitness level (e.g., "beginner", "intermediate", "I've been lifting for 2 years")
        """
        logger.info(f"[PROGRAM] Capturing fitness level: {fitness_level}")

        normalized_level = normalize_fitness_level(fitness_level)

        self.state.set("program_creation.fitness_level", normalized_level)

        logger.info(f"[PROGRAM] Fitness level normalized to: {normalized_level}")

        height_cm = self.state.get("program_creation.height_cm")
        weight_kg = self.state.get("program_creation.weight_kg")
        goal_category = self.state.get("program_creation.goal_category")
        goal_raw = self.state.get("program_creation.goal_raw")
        duration_weeks = self.state.get("program_creation.duration_weeks")
        days_per_week = self.state.get("program_creation.days_per_week")

        logger.info("="*60)
        logger.info("[PROGRAM CREATION] All parameters collected:")
        logger.info(f"  User: the user (ID: {self.user_id})")
        logger.info(f"  Height: {height_cm} cm, Weight: {weight_kg} kg")
        logger.info(f"  Goal: {goal_category} (\"{goal_raw}\")")
        logger.info(f"  Duration: {duration_weeks} weeks, Frequency: {days_per_week} days/week")
        logger.info(f"  Fitness Level: {normalized_level}")
        logger.info("="*60 + "\n")

        vbt_enabled = should_enable_vbt(
            fitness_level=normalized_level,
            goal_category=goal_category,
            specific_sport=self.state.get("program_creation.specific_sport", "none")
        )

        logger.info(f"[PROGRAM] VBT Decision: {'ENABLED' if vbt_enabled else 'DISABLED'}")

        self.state.set("program_creation.vbt_enabled", vbt_enabled)
        self.state.set("program_creation.all_params_collected", True)

        return None, "All parameters collected. Summarize their program, call set_vbt_capability, then generate_workout_program."

    @function_tool
    async def set_vbt_capability(self, context: RunContext, enabled: bool):
        """
        Automatically enable or disable VBT based on training parameters.
        DO NOT call this manually - it's automatically called after capture_fitness_level().

        Args:
            enabled: True to enable VBT programming, False to disable
        """
        logger.info(f"[PROGRAM] Setting VBT capability: {enabled}")

        self.state.set("program_creation.has_vbt_capability", enabled)

        if enabled:
            logger.info("[PROGRAM] VBT ENABLED - Program will include velocity-based training")
        else:
            logger.info("[PROGRAM] VBT DISABLED - Program will use traditional percentage-based loading")

        return None, "Captured. Immediately ask the next question."

    @function_tool
    async def generate_workout_program(self, context: RunContext):
        """
        Call this to generate a complete workout program via FastAPI backend.
        This tool handles the entire generation process including polling — do NOT call any status-check tool after this.
        """
        import httpx

        logger.info("="*80)
        logger.info("[PROGRAM] generate_workout_program() CALLED (FastAPI mode)")
        logger.info("="*80)

        user_id = self.user_id
        saved_program_id = self.state.get("program_creation.saved_program_id")
        if saved_program_id:
            return None, f"Program already generated. Now call finish_program_creation() to complete."

        height_cm = self.state.get("program_creation.height_cm")
        weight_kg = self.state.get("program_creation.weight_kg")
        age = self.state.get("program_creation.age")
        sex = self.state.get("program_creation.sex")
        goal_category = self.state.get("program_creation.goal_category")
        goal_raw = self.state.get("program_creation.goal_raw")
        duration_weeks = self.state.get("program_creation.duration_weeks")
        days_per_week = self.state.get("program_creation.days_per_week")
        fitness_level = self.state.get("program_creation.fitness_level")

        missing = []
        if not height_cm: missing.append("height_cm")
        if not weight_kg: missing.append("weight_kg")
        if not age: missing.append("age")
        if not sex: missing.append("sex")
        if not goal_category: missing.append("goal_category")
        if not goal_raw: missing.append("goal_raw")
        if not duration_weeks: missing.append("duration_weeks")
        if not days_per_week: missing.append("days_per_week")
        if not fitness_level: missing.append("fitness_level")

        if missing:
            logger.info(f"[PROGRAM] ERROR: Missing required parameters: {', '.join(missing)}")
            return None, f"ERROR: Cannot generate program - missing required parameters: {', '.join(missing)}. Go back and ask the missing questions."

        # Speak hold message and suppress VAD so user speech doesn't interrupt polling
        await self._say(
            "Tell the user you've got everything you need and you're building "
            "their custom program right now — it takes about thirty seconds. "
            "One or two sentences, your own words, vary it between sessions.",
            wait=True,
            restore=False,
        )

        try:
            user_info = self.state.get_user()
            params = build_program_generation_payload(
                self.state.get("program_creation", {}),
                user_id=str(user_id),
                name=user_info.get("name", "Unknown"),
                email=user_info.get("email", "unknown@nowva.ai"),
                send_email=False,
            )

            fastapi_url = os.getenv("FASTAPI_URL", "http://localhost:8000")
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{fastapi_url}/api/programs/generate",
                    json=params,
                    headers=service_headers(),
                    timeout=10.0
                )
                data = response.json()
                if response.status_code != 202:
                    raise Exception(f"FastAPI error {response.status_code}: {data}")
                job_id = data["job_id"]

            self.state.set("program_creation.job_id", job_id)
            logger.info(f"[PROGRAM] Started generation job: {job_id}")
            self._log_function_call("generate_workout_program", params, {"job_id": job_id})

            # Poll for completion internally — never ask the LLM to poll
            final_status = None
            poll_started = time.monotonic()
            filler_spoken = False
            for attempt in range(1, 25):
                await asyncio.sleep(5.0)
                if not filler_spoken and time.monotonic() - poll_started >= MID_BUILD_FILLER_DELAY_S:
                    filler_spoken = True
                    self.session.generate_reply(
                        instructions=(
                            "One short reassuring line, your own words: the "
                            "program build is still running and almost done."
                        ),
                        tool_choice="none",
                    )
                try:
                    async with httpx.AsyncClient() as client:
                        status_resp = await client.get(
                            f"{fastapi_url}/api/programs/status/{job_id}",
                            headers=service_headers(),
                            timeout=5.0
                        )
                        status_data = status_resp.json()

                    status = status_data["status"]
                    progress = status_data.get("progress", 0)
                    logger.info(f"[PROGRAM] Poll attempt {attempt}: {status} ({progress}%)")

                    if status == "completed":
                        program_id = status_data["program_id"]
                        self.state.set("program_creation.saved_program_id", program_id)
                        logger.info(f"[PROGRAM] Program generation complete! ID: {program_id}")
                        final_status = "completed"
                        break
                    elif status == "failed":
                        error = status_data.get("error_message", "Unknown error")
                        logger.info(f"[PROGRAM] Generation failed: {error}")
                        final_status = "failed"
                        break
                except Exception as poll_err:
                    logger.info(f"[PROGRAM] Poll error on attempt {attempt}: {poll_err}")

            self._restore_turn_detection()

            if final_status == "completed":
                return None, (
                    "Program is ready! Say something like: 'Great news! Your custom program is ready. "
                    "I've saved it to your account.' Then call finish_program_creation(). Be enthusiastic!"
                )
            elif final_status == "failed":
                return None, (
                    "Generation failed. Say something like: 'Hmmm, seems like I had trouble creating "
                    "your program. Let me try again.' Keep it apologetic."
                )
            else:
                return None, (
                    "The program is still generating but it's taking longer than usual. Say something like: "
                    "'Your program is taking a bit longer than expected — it should be ready soon. "
                    "Check back in a minute and I'll have it for you.' Keep it reassuring."
                )

        except Exception as e:
            logger.exception("[PROGRAM] ERROR")
            self._restore_turn_detection()

            result = (None, f"Error starting generation. Say something like: 'I had trouble starting your program. Let me try again.' Keep it apologetic.")
            self._log_function_call("generate_workout_program", {}, result)

            return result

    @function_tool
    async def finish_program_creation(self, context: RunContext):
        """
        Call this to complete the program creation process and return to main menu.
        """
        logger.info("[PROGRAM] Finishing program creation, returning to main menu...")

        self.state.set("program_creation", None)
        self.state.switch_mode("main_menu")
        self.state.save_state()

        logger.info("[STATE] Returned to main_menu mode")

        # Handoff to MainMenuAgent
        await self._suppress_turn_detection()
        from agent.agents.main_menu_agent import MainMenuAgent
        return await self._carry_context_to(MainMenuAgent(state=self.state, userdata=self.userdata))

    # ===== PROGRAM UPDATE TOOLS =====

    @function_tool
    async def select_program_for_update(self, context: RunContext, program_name: str):
        """
        Call this when the user selects which program they want to update (when they have multiple programs).

        Args:
            program_name: The name of the program the user wants to update
        """
        programs = self.state.get("program_update.available_programs", [])

        if not programs:
            return None, f"Error: No programs available for selection. Try calling update_program() first."

        selected_program = None
        program_name_lower = program_name.lower()

        for program in programs:
            if program_name_lower in program["name"].lower():
                selected_program = program
                break

        if not selected_program:
            program_list = ", ".join([f"'{p['name']}'" for p in programs])
            return None, f"I didn't find a program called '{program_name}'. Your programs are: {program_list}. Which one would you like to update?"

        self.state.set("program_update.selected_program_id", selected_program["id"])
        self.state.set("program_update.selected_program_name", selected_program["name"])

        logger.info(f"[PROGRAM UPDATE] Selected program: {selected_program['name']} (ID: {selected_program['id']})")

        return None, f"Great! I'll update your {selected_program['name']} program. What would you like to change about it?"

    @function_tool
    async def capture_program_change_request(self, context: RunContext, change_request: str):
        """
        Call this when the user describes what they want to change about their program.

        Args:
            change_request: The user's description of what they want to change
        """
        user_id = self.user_id
        program_id = self.state.get("program_update.selected_program_id")
        program_name = self.state.get("program_update.selected_program_name")

        if not program_id:
            return None, f"Error: No program selected for update. Call update_program() first."

        self.state.set("program_update.change_request", change_request)

        logger.info(f"[PROGRAM UPDATE] Change request: {change_request}")

        db = SessionLocal()
        try:
            from api.services.simple_program_updates import detect_simple_update, handle_simple_update

            update_type, params = detect_simple_update(change_request)

            if update_type != "requires_llm":
                logger.info(f"[PROGRAM UPDATE] Detected safe simple update: {update_type}")
                success, message = handle_simple_update(db, program_id, change_request)

                if success:
                    self.state.set("program_update", None)
                    # Handoff back to MainMenuAgent after simple update
                    self.state.switch_mode("main_menu")
                    self.state.save_state()
                    await self._suppress_turn_detection()
                    from agent.agents.main_menu_agent import MainMenuAgent
                    return await self._carry_context_to(MainMenuAgent(state=self.state, userdata=self.userdata))
                else:
                    return None, f"Say something like: 'I had trouble updating that. {message}' Keep it apologetic."

            logger.info(f"[PROGRAM UPDATE] Training change detected, validating with LLM...")

            from db.models import User
            db_user = db.query(User).filter(User.id == user_id).first()

            if not db_user:
                return None, f"Error: User not found in database."

            if not db_user.age or not db_user.sex or not db_user.height_cm or not db_user.weight_kg:
                return None, f"Say something like: 'I need some more information about you first. Let me ask you a few quick questions.' Then ask for missing: age, sex, height, weight."

            user_profile = {
                "age": int(db_user.age),
                "sex": db_user.sex,
                "height_cm": float(db_user.height_cm),
                "weight_kg": float(db_user.weight_kg),
                "fitness_level": "intermediate"
            }

            from api.services.program_updater import _get_current_program_as_json, validate_program_change_with_llm
            current_program = _get_current_program_as_json(db, program_id)
            if not current_program:
                return None, f"Error: Could not load program for validation."

            validation_result = await validate_program_change_with_llm(
                current_program=current_program,
                change_request=change_request,
                user_profile=user_profile
            )

            is_risky = validation_result.get("is_risky", False)

            if not is_risky:
                logger.info(f"[PROGRAM UPDATE] Validation passed, proceeding with update")
                self.state.set("program_update.user_profile", user_profile)
                return None, f"Got it! I'll update your {program_name} program: '{change_request}'. Now call apply_program_update() to apply the changes."

            else:
                warning = validation_result.get("warning", "This change may not be ideal for your goals.")
                alternative = validation_result.get("alternative", "")

                self.state.set("program_update.validation_result", validation_result)
                self.state.set("program_update.user_profile", user_profile)
                self.state.set("program_update.awaiting_choice", True)

                if alternative:
                    return None, f"Say something like: 'I noticed something about this change. {warning} {alternative} Would you prefer that alternative, or do you still want to go with your original request?' Keep it conversational."
                else:
                    return None, f"Say something like: 'I need to mention something. {warning} Are you sure you want to make this change?' Keep it concerned but supportive."

        except Exception as e:
            logger.exception("[ERROR] Failed to process change request")
            return None, f"Error processing change request. Try again."
        finally:
            db.close()

    @function_tool
    async def apply_program_update(self, context: RunContext, user_response: str = ""):
        """
        Call this to apply a program update. Handles the entire update process including polling.
        Do NOT call any status-check tool after this — it handles everything internally.

        Args:
            user_response: Optional - user's response if they were presented with validation choices
        """
        import httpx

        program_id = self.state.get("program_update.selected_program_id")
        program_name = self.state.get("program_update.selected_program_name")
        change_request = self.state.get("program_update.change_request")
        user_profile = self.state.get("program_update.user_profile")
        awaiting_choice = self.state.get("program_update.awaiting_choice", False)
        validation_result = self.state.get("program_update.validation_result")

        if not program_id or not change_request or not user_profile:
            return None, f"Error: Missing required data. Ensure capture_program_change_request() was called first."

        # Resolve user choice if validation presented alternatives
        if awaiting_choice and validation_result:
            logger.info(f"[PROGRAM UPDATE] Processing user choice: {user_response}")

            alternative = validation_result.get("alternative", "")
            user_response_lower = user_response.lower()

            wants_alternative = any(phrase in user_response_lower for phrase in [
                "alternative", "better", "that sounds good", "yes", "yeah", "sure",
                "front squat", "safety bar", "2 days", "3 days"
            ])

            wants_original = any(phrase in user_response_lower for phrase in [
                "original", "no", "still want", "barbell curl", "1 day", "stick with"
            ])

            cancel = any(phrase in user_response_lower for phrase in [
                "cancel", "never mind", "forget it", "don't"
            ])

            if cancel:
                self.state.set("program_update", None)
                self.state.switch_mode("main_menu")
                self.state.save_state()
                await self._suppress_turn_detection()
                from agent.agents.main_menu_agent import MainMenuAgent
                return await self._carry_context_to(MainMenuAgent(state=self.state, userdata=self.userdata))

            elif wants_alternative and alternative:
                logger.info(f"[PROGRAM UPDATE] User chose alternative: {alternative}")
                change_request = alternative
            elif wants_original:
                logger.info(f"[PROGRAM UPDATE] User insists on original request")
            else:
                if alternative:
                    change_request = alternative

            self.state.set("program_update.awaiting_choice", False)
            self.state.set("program_update.validation_result", None)

        # Speak hold message and suppress VAD so user speech doesn't interrupt polling
        await self._say(
            f"Tell the user you're updating their {program_name} program now — "
            f"it takes about a minute. One or two sentences, your own words, "
            f"vary it between sessions.",
            wait=True,
            restore=False,
        )

        try:
            fastapi_url = os.getenv("FASTAPI_URL", "http://localhost:8000")
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{fastapi_url}/api/programs/{program_id}/update",
                    json={
                        "change_request": change_request,
                        "age": user_profile["age"],
                        "sex": user_profile["sex"],
                        "height_cm": user_profile["height_cm"],
                        "weight_kg": user_profile["weight_kg"],
                        "fitness_level": user_profile["fitness_level"]
                    },
                    headers=service_headers(),
                    timeout=10.0
                )
                data = response.json()
                job_id = data["job_id"]

            self.state.set("program_update.job_id", job_id)
            logger.info(f"[PROGRAM UPDATE] Started update job: {job_id}")

            # Poll for completion internally — never ask the LLM to poll
            final_status = None
            final_data = None
            for attempt in range(1, 25):
                await asyncio.sleep(5.0)
                try:
                    async with httpx.AsyncClient() as client:
                        status_resp = await client.get(
                            f"{fastapi_url}/api/programs/update-status/{job_id}",
                            headers=service_headers(),
                            timeout=5.0
                        )
                        status_data = status_resp.json()

                    status = status_data["status"]
                    progress = status_data.get("progress", 0)
                    logger.info(f"[PROGRAM UPDATE] Poll attempt {attempt}: {status} ({progress}%)")

                    if status == "completed":
                        final_status = "completed"
                        final_data = status_data
                        logger.info(f"[PROGRAM UPDATE] Update complete!")
                        break
                    elif status == "failed":
                        error = status_data.get("error_message", "Unknown error")
                        logger.info(f"[PROGRAM UPDATE] Update failed: {error}")
                        final_status = "failed"
                        break
                except Exception as poll_err:
                    logger.info(f"[PROGRAM UPDATE] Poll error on attempt {attempt}: {poll_err}")

            self._restore_turn_detection()

            if final_status == "completed":
                self.state.set("program_update", None)
                self.state.switch_mode("main_menu")
                self.state.save_state()
                await self._suppress_turn_detection()
                from agent.agents.main_menu_agent import MainMenuAgent
                return await self._carry_context_to(MainMenuAgent(state=self.state, userdata=self.userdata))

            elif final_status == "failed":
                return None, (
                    "Update failed. Say something like: 'I had trouble updating your program. "
                    "Let's try again.' Keep it apologetic."
                )
            else:
                return None, (
                    "The update is still processing but it's taking longer than usual. Say something like: "
                    "'Your program update is taking a bit longer than expected — it should be ready soon. "
                    "Check back in a minute and I'll have it for you.' Keep it reassuring."
                )

        except Exception as e:
            logger.exception("[PROGRAM UPDATE] ERROR")
            self._restore_turn_detection()
            return None, f"Error starting update. Say something like: 'I had trouble starting the update. Let me try again.'"
