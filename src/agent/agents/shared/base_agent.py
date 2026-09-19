"""
Base agent class with shared properties and helpers for all Nova agents.
"""

import logging
from livekit.agents import Agent
from agent.core.agent_state import AgentState
from agent.agents.prompts.base_prompt import BASE_PROMPT
from agent.agents.shared.affect_mixin import AffectNodesMixin

logger = logging.getLogger(__name__)


class BaseNovaAgent(AffectNodesMixin, Agent):
    """Base class for all Nova voice agents with shared state access and utilities.

    AffectNodesMixin supplies tts_node (TTS text normalization + voice style) and
    llm_node (athlete-state injection).
    """

    def __init__(self, state: AgentState, userdata, instructions: str) -> None:
        self.state = state
        self.userdata = userdata
        super().__init__(instructions=f"{BASE_PROMPT}\n\n{instructions}")

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

    async def _truncate_context_for_handoff(self, max_items: int = 6):
        """Replace old context with compaction summary before agent handoff.

        If a CompactionService is active, injects its pre-built summary as
        a system message so the next agent retains conversation history.
        Falls back to simple truncation if no summary is available.
        """
        try:
            ctx = self.chat_ctx
            if len(ctx.items) <= max_items:
                return

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
                return

            # Summary-aware truncation: system items + summary + recent items
            from livekit.agents import llm

            items = list(ctx.items)
            system_items = [i for i in items if hasattr(i, 'role') and i.role in ("system", "developer")]
            non_system = [i for i in items if not (hasattr(i, 'role') and i.role in ("system", "developer"))]
            recent_items = non_system[-max_items:] if len(non_system) > max_items else non_system

            new_ctx = llm.ChatContext.empty()
            for item in system_items:
                new_ctx.items.append(item)

            summary_message = llm.ChatMessage(
                role="system",
                content=[f"[CONVERSATION SUMMARY]\n{summary_text}"],
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
        except Exception as e:
            logger.warning(f"[HANDOFF] Context truncation failed: {e}")

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
