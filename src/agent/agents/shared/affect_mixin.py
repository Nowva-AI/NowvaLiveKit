"""AffectNodesMixin: injects the athlete state line into llm_node and the voice style into tts_node."""

from __future__ import annotations

import logging
import time
from typing import Any

from livekit.agents import Agent, RunContext, function_tool, llm

from affect.state import STATE_ITEM_PREFIX, AthleteState
from agent.services.tts_normalizer import normalize_stream

logger = logging.getLogger(__name__)

STATE_ITEM_ID = "nowva_athlete_state"


def _strip_prefix_line(text: str) -> str:
    if text.startswith(STATE_ITEM_PREFIX):
        first_newline = text.find("\n")
        return text[first_newline + 1 :] if first_newline >= 0 else ""
    return text


def inject_state(chat_ctx: llm.ChatContext, state: AthleteState, inject_as: str) -> llm.ChatContext:
    """Return a copy of chat_ctx carrying exactly one athlete line (or none). The original is never mutated."""
    ctx = chat_ctx.copy()
    items = [item for item in ctx.items if getattr(item, "id", None) != STATE_ITEM_ID]
    line = state.to_prompt_line()
    if inject_as == "user_prefix":
        for index in range(len(items) - 1, -1, -1):
            item = items[index]
            if getattr(item, "type", None) == "message" and getattr(item, "role", None) == "user":
                original = _strip_prefix_line(item.text_content or "")
                content = f"{line}\n{original}" if line else original
                if content != (item.text_content or ""):
                    items[index] = llm.ChatMessage(id=item.id, role="user", content=[content], created_at=item.created_at)
                break
        ctx.items = items
        return ctx
    ctx.items = items
    if line:
        ctx.items.append(llm.ChatMessage(id=STATE_ITEM_ID, role="system", content=[line]))
    return ctx


class AffectNodesMixin:
    """Mix into Agent / AgentTask subclasses before Agent in the MRO."""

    @function_tool
    async def how_do_i_sound(self, context: RunContext) -> str:
        """Report what you hear in the athlete's voice right now: energy, positivity, assertiveness,
        effort and how confident the reading is.

        Call this ONLY when the user asks how they sound, what tone or emotion you pick up, or
        whether they seem tired, stressed or frustrated. Never call it unprompted. Answer in one
        or two plain sentences and hedge when confidence is low.
        """
        service = self._affect_service()
        if service is None:
            return "Voice perception is off in this session, so go only by their words."
        try:
            return service.describe_for_llm()
        except Exception:  # noqa: BLE001
            logger.exception("[AFFECT] how_do_i_sound failed")
            return "The voice reading is unavailable right now."

    def _affect_service(self) -> Any:
        userdata = getattr(self, "userdata", None)
        if userdata is None:
            session = getattr(self, "session", None)
            userdata = getattr(session, "userdata", None) if session is not None else None
        return getattr(userdata, "affect_service", None)

    async def llm_node(self, chat_ctx: llm.ChatContext, tools: list, model_settings: Any):
        service = self._affect_service()
        ctx = chat_ctx
        if service is not None and getattr(service, "enabled", False):
            try:
                kind = service.speech_kind()
                t0 = time.perf_counter()
                state = await service.snapshot(max_wait_s=service.wait_budget_s(kind))
                service.record_llm_wait((time.perf_counter() - t0) * 1000.0, state.fresh)
                ctx = inject_state(chat_ctx, state, service.config.trigger.inject_as)
            except Exception:  # noqa: BLE001 — affect must never break a reply
                logger.exception("[AFFECT] llm_node injection failed; continuing without state")
                ctx = chat_ctx
        async for chunk in Agent.default.llm_node(self, ctx, tools, model_settings):
            yield chunk

    def tts_node(self, text, model_settings: Any):
        stream = normalize_stream(text)
        service = self._affect_service()
        if service is not None and getattr(service, "enabled", False):
            try:
                style = service.current_style(service.speech_kind())
                adapter = service.style_adapter
                activity = getattr(self, "_activity", None)
                tts = getattr(activity, "tts", None) if activity is not None else None
                if tts is not None:
                    adapter.prepare_tts(tts, style)
                stream = adapter.wrap_text(stream, style)
            except Exception:  # noqa: BLE001
                logger.exception("[AFFECT] tts_node styling failed; continuing unstyled")
        return Agent.default.tts_node(self, stream, model_settings)
