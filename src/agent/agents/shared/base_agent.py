"""
Base agent class with shared properties and helpers for all Nova agents.
"""

import logging
from livekit.agents import Agent, llm
from agent.core.agent_state import AgentState
from agent.agents.prompts.base_prompt import BASE_PROMPT
from agent.agents.shared.affect_mixin import AffectNodesMixin
from agent.services.athlete_facts import athlete_facts_line

logger = logging.getLogger(__name__)

SUMMARY_PREFIX = "[CONVERSATION SUMMARY]"


def is_summary_item(item: object) -> bool:
    """Whether a chat item is a compaction summary injected by a prune."""
    return getattr(item, "role", None) == "system" and (
        getattr(item, "text_content", None) or ""
    ).startswith(SUMMARY_PREFIX)


def build_agent_instructions(state: AgentState, instructions: str) -> str:
    """Base prompt + agent prompt + the athlete's durable facts (pain, preferences) when any."""
    parts = [BASE_PROMPT, instructions]
    facts = athlete_facts_line(state)
    if facts:
        parts.append(
            "# What You Know About This Athlete\n"
            f"{facts}\n"
            "Honor these in every reply without reading them out."
        )
    return "\n\n".join(parts)


class BaseNovaAgent(AffectNodesMixin, Agent):
    """Base class for all Nova voice agents with shared state access and utilities.

    AffectNodesMixin supplies tts_node (TTS text normalization + voice style) and
    llm_node (athlete-state injection).
    """

    def __init__(self, state: AgentState, userdata, instructions: str) -> None:
        self.state = state
        self.userdata = userdata
        self._agent_instructions = instructions
        super().__init__(instructions=build_agent_instructions(state, instructions))

    @property
    def user_id(self) -> str:
        """Get current user ID from state"""
        return self.state.get_user().get("id")

    def _publish_visual(self, event: dict) -> None:
        """Best-effort publish to the display page; no-op without a bridge."""
        bridge = getattr(self.userdata, "visual_bridge", None)
        if bridge is not None:
            bridge.send(event)

    @property
    def user_name(self) -> str:
        """Get current user name from state, defaults to 'there'"""
        return self.state.get_user().get("name", "there")

    async def _suppress_turn_detection(self):
        """Disable audio input so agent speech isn't interrupted.

        In cascade mode this is a local operation — no server round-trip needed.
        """
        try:
            self.session.input.set_audio_enabled(False)
            logger.info("[TURN DETECTION] Audio input disabled (suppressed)")
        except Exception as e:
            logger.warning(f"[TURN DETECTION] Failed to suppress: {e}")

    def _restore_turn_detection(self):
        """Re-enable audio input for normal conversation."""
        try:
            self.session.input.set_audio_enabled(True)
            logger.info("[TURN DETECTION] Audio input restored")
        except Exception as e:
            logger.warning(f"[TURN DETECTION] Failed to restore: {e}")

    async def _say(self, instructions: str, wait: bool = True, restore: bool = True):
        """Generate a greeting that won't be cut off by turn detection.

        Suppresses auto-responses/interruptions before speaking, waits for
        full playout, then optionally restores normal turn detection.

        Args:
            instructions: The instruction for the LLM to generate speech from.
            wait: If True, block until speech finishes playing.
            restore: If True, restore normal turn detection after playout.
        """
        await self._suppress_turn_detection()
        handle = self.session.generate_reply(
            instructions=instructions,
            tool_choice="none",
        )
        if wait:
            await handle.wait_for_playout()
        if restore:
            self._restore_turn_detection()
        return handle

    async def _truncate_context_for_handoff(self, max_items: int = 6) -> llm.ChatContext:
        """Shrink the context to one compaction summary plus recent turns, and return it.

        LiveKit starts an agent built without chat_ctx with an EMPTY context, so every
        handoff must pass the returned context on (see _carry_context_to) or the conversation is
        lost. Falls back to simple truncation without a summary.
        """
        try:
            ctx = self.chat_ctx
            if len(ctx.items) <= max_items:
                return ctx.copy()

            old_count = len(ctx.items)

            # Try to get compaction summary
            summary_text = ""
            compaction = getattr(self.userdata, 'compaction_service', None)
            if compaction:
                summary_text = compaction.get_summary()

            if not summary_text:
                # Fallback: simple truncation (original behavior)
                new_ctx = ctx.copy()
                new_ctx.truncate(max_items=max_items)
                await self.update_chat_ctx(new_ctx)
                logger.info(
                    f"[HANDOFF] Truncated context: {old_count} → {len(new_ctx.items)} items (no summary)"
                )
                return self.chat_ctx.copy()

            # Summary-aware truncation: system items + ONE summary (an earlier one is
            # replaced, never stacked) + recent items
            items = list(ctx.items)
            system_items = [
                i for i in items
                if hasattr(i, 'role') and i.role in ("system", "developer") and not is_summary_item(i)
            ]
            non_system = [i for i in items if not (hasattr(i, 'role') and i.role in ("system", "developer"))]
            recent_items = non_system[-max_items:] if len(non_system) > max_items else non_system

            new_ctx = llm.ChatContext.empty()
            for item in system_items:
                new_ctx.items.append(item)

            summary_message = llm.ChatMessage(
                role="system",
                content=[f"{SUMMARY_PREFIX}\n{summary_text}"],
            )
            new_ctx.items.append(summary_message)

            for item in recent_items:
                new_ctx.items.append(item)

            await self.update_chat_ctx(new_ctx)
            logger.info(
                f"[HANDOFF] Context with summary: {old_count} → {len(new_ctx.items)} items "
                f"({len(system_items)} system + 1 summary + {len(recent_items)} recent)"
            )
            logger.info("[COMPACTION:SWAP] Summary injected into context on handoff")
            return self.chat_ctx.copy()
        except Exception as e:
            logger.warning(f"[HANDOFF] Context truncation failed: {e}")
            return self.chat_ctx.copy()

    async def _refresh_athlete_facts(self) -> None:
        """Rebuild the instructions with the athlete's current facts, e.g. right after a pain report."""
        try:
            await self.update_instructions(build_agent_instructions(self.state, self._agent_instructions))
        except Exception as e:
            # The fact is already stored; a failed refresh must not break the tool reply.
            logger.warning(f"[FACTS] Instruction refresh failed: {e}")

    async def _carry_context_to(self, next_agent: Agent) -> Agent:
        """Give the agent we switch to this conversation, pruned; LiveKit would start it empty."""
        await next_agent.update_chat_ctx(await self._truncate_context_for_handoff())
        return next_agent

    def _log_function_call(self, function_name: str, parameters: dict, result):
        """Helper method to log function tool calls"""
        from agent.core.session_logger import SessionLogger
        from agent.core.token_estimator import estimate_function_call_tokens

        session_logger = SessionLogger.get_instance()
        estimated_tokens = estimate_function_call_tokens(function_name, parameters, result)

        session_logger.log_function_call(
            function_name=function_name,
            parameters=parameters,
            result=result,
            estimated_tokens=estimated_tokens
        )
