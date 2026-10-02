"""Tests for reshaping a chat context to fit single-system-message chat templates."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from livekit.agents import llm

from agent.services.chat_template_compat import (
    DIRECTION_MARKER,
    NO_SPEECH_YET,
    normalize_for_strict_template,
    requires_strict_template,
)

PROMPT = "You are Nova."
GREET_DIRECTION = "Greet the user back in one short line."


def _ctx(*turns: tuple[str, str]) -> llm.ChatContext:
    ctx = llm.ChatContext.empty()
    for role, text in turns:
        ctx.add_message(role=role, content=text)
    return ctx


def _shape(ctx: llm.ChatContext) -> list[tuple[str, str]]:
    return [(item.role, item.text_content) for item in ctx.items if item.type == "message"]


class TestGate:
    def test_cerebras_requires_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "cerebras")
        assert requires_strict_template() is True

    @pytest.mark.parametrize("provider", ["openai", "groq", ""])
    def test_tolerant_providers_are_left_alone(self, monkeypatch: pytest.MonkeyPatch, provider: str) -> None:
        monkeypatch.setenv("LLM_PROVIDER", provider)
        assert requires_strict_template() is False


class TestGreetingBeforeTheUserSpeaks:
    """generate_reply(instructions=...) on a fresh session: a prompt, a direction, no user turn."""

    def test_direction_travels_as_a_marked_user_turn(self) -> None:
        shaped = _shape(normalize_for_strict_template(_ctx(("system", PROMPT), ("system", GREET_DIRECTION))))
        assert shaped == [("system", PROMPT), ("user", DIRECTION_MARKER + GREET_DIRECTION)]

    def test_exactly_one_system_message_and_it_is_first(self) -> None:
        ctx = normalize_for_strict_template(_ctx(("system", PROMPT), ("system", GREET_DIRECTION)))
        roles = [role for role, _ in _shape(ctx)]
        assert roles[0] == "system"
        assert roles.count("system") == 1


class TestMidConversation:
    def test_trailing_direction_is_re_roled_in_place(self) -> None:
        ctx = _ctx(
            ("system", PROMPT),
            ("assistant", "Hold still."),
            ("user", "Okay."),
            ("system", "Tell them calibration finished."),
        )
        shaped = _shape(normalize_for_strict_template(ctx))
        assert shaped[-1] == ("user", DIRECTION_MARKER + "Tell them calibration finished.")
        assert shaped[1:3] == [("assistant", "Hold still."), ("user", "Okay.")]

    def test_handoff_summary_merges_into_the_single_system_message(self) -> None:
        ctx = _ctx(("system", PROMPT), ("system", "[CONVERSATION SUMMARY]\nCalibrated earlier."), ("user", "What's next?"))
        shaped = _shape(normalize_for_strict_template(ctx))
        assert shaped == [
            ("system", f"{PROMPT}\n\n[CONVERSATION SUMMARY]\nCalibrated earlier."),
            ("user", "What's next?"),
        ]

    def test_developer_role_counts_as_system(self) -> None:
        shaped = _shape(normalize_for_strict_template(_ctx(("system", PROMPT), ("user", "Hi."), ("developer", "Be brief."))))
        assert shaped[-1] == ("user", DIRECTION_MARKER + "Be brief.")


class TestGuarantees:
    def test_a_user_turn_always_exists(self) -> None:
        shaped = _shape(normalize_for_strict_template(_ctx(("system", PROMPT))))
        assert shaped == [("system", PROMPT), ("user", NO_SPEECH_YET)]

    def test_no_placeholder_when_the_user_has_spoken(self) -> None:
        shaped = _shape(normalize_for_strict_template(_ctx(("system", PROMPT), ("user", "Hey."))))
        assert shaped == [("system", PROMPT), ("user", "Hey.")]

    def test_the_session_history_is_never_mutated(self) -> None:
        original = _ctx(("system", PROMPT), ("system", GREET_DIRECTION))
        normalize_for_strict_template(original)
        assert _shape(original) == [("system", PROMPT), ("system", GREET_DIRECTION)]

    def test_already_valid_context_is_unchanged(self) -> None:
        turns = [("system", PROMPT), ("assistant", "Hey."), ("user", "Let's train.")]
        assert _shape(normalize_for_strict_template(_ctx(*turns))) == turns
