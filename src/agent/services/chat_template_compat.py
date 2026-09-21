"""Reshape a chat context for models whose chat template is stricter than OpenAI's.

Open-weight models served behind an OpenAI-compatible API still run their own
chat template, and Qwen's (like Gemma's) rejects what Nova sends all the time:

  - only ONE system message is allowed, and it must come first
  - at least one user message must exist

Nova breaks both by design. generate_reply(instructions=...) appends its
direction as a trailing system message (every greeting and scripted line), the
athlete-state line is a late system message, and a handoff inserts a
conversation summary as a second leading one. OpenAI and Groq's gpt-oss accept
all of that; Cerebras answers 400 and Nova goes silent.

The reshaped copy is only ever handed to the LLM call. The session's own
history is never touched, so nothing here can leak into later turns.
"""

from __future__ import annotations

import os

from livekit.agents import llm

STRICT_TEMPLATE_PROVIDERS = ("cerebras",)
SYSTEM_ROLES = ("system", "developer")

# Marks a late system message once it has to travel as a user turn, so the model
# follows it as direction instead of answering it as something the user said.
DIRECTION_MARKER = "[Direction for Nova — not something the user said] "
NO_SPEECH_YET = "(The session just opened. The user has not spoken yet.)"


def requires_strict_template() -> bool:
    return os.getenv("LLM_PROVIDER", "").strip().lower() in STRICT_TEMPLATE_PROVIDERS


def _is_system_message(item: object) -> bool:
    return getattr(item, "type", None) == "message" and getattr(item, "role", None) in SYSTEM_ROLES


def _as_direction(item: llm.ChatMessage) -> llm.ChatMessage:
    return llm.ChatMessage(
        id=item.id,
        role="user",
        content=[DIRECTION_MARKER + (item.text_content or "")],
        created_at=item.created_at,
    )


def normalize_for_strict_template(chat_ctx: llm.ChatContext) -> llm.ChatContext:
    """Return a copy with one leading system message, late ones re-roled, and a user turn guaranteed."""
    ctx = chat_ctx.copy()
    # Before anyone has spoken, a system message after the prompt can only be a direction
    # for this reply (a greeting). Once there is history, system messages stacked at the
    # top are standing context — the prompt plus a handoff summary — and belong together.
    conversation_started = any(not _is_system_message(item) for item in ctx.items)

    leading: list[str] = []
    rest: list = []
    for item in ctx.items:
        if not _is_system_message(item):
            rest.append(item)
        elif not leading or (conversation_started and not rest):
            if item.text_content:
                leading.append(item.text_content)
        else:
            rest.append(_as_direction(item))

    has_user_turn = any(
        getattr(item, "type", None) == "message" and getattr(item, "role", None) == "user" for item in rest
    )
    if not has_user_turn:
        rest.append(llm.ChatMessage(role="user", content=[NO_SPEECH_YET]))

    items: list = []
    if leading:
        items.append(llm.ChatMessage(role="system", content=["\n\n".join(leading)]))
    ctx.items = items + rest
    return ctx
