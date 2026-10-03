"""Durable facts about the athlete that the coach must never forget.

Pain reports and stated preferences live in the athlete's local state file, so
they survive context pruning, agent handoffs and sessions without any cloud
call. Prompts get them as one short line.
"""

from __future__ import annotations

from datetime import date, datetime

from agent.core.agent_state import AgentState

FACTS_KEY = "athlete"
MAX_PAIN_FLAGS = 5
MAX_PREFERENCES = 8
MAX_TEXT_CHARS = 80


def _clean(text: str) -> str:
    return " ".join(str(text).split())[:MAX_TEXT_CHARS]


def _load(state: AgentState) -> dict:
    facts = state.get(FACTS_KEY) or {}
    return {
        "pain_flags": list(facts.get("pain_flags") or []),
        "preferences": dict(facts.get("preferences") or {}),
    }


def _store(state: AgentState, facts: dict) -> None:
    state.set(FACTS_KEY, facts)
    state.save_state()


def _when(reported_at: str, today: date) -> str:
    reported_day = datetime.fromisoformat(reported_at).date()
    if reported_day == today:
        return "today"
    return f"since {reported_day.strftime('%b')} {reported_day.day}"


def add_pain_flag(state: AgentState, body_part: str, note: str) -> dict:
    """Record (or refresh) a pain report for one body part."""
    facts = _load(state)
    part = _clean(body_part).lower()
    flag = {
        "body_part": part,
        "note": _clean(note),
        "reported_at": datetime.now().isoformat(timespec="seconds"),
    }
    flags = [f for f in facts["pain_flags"] if f.get("body_part") != part]
    flags.append(flag)
    facts["pain_flags"] = flags[-MAX_PAIN_FLAGS:]
    _store(state, facts)
    return flag


def resolve_pain_flag(state: AgentState, body_part: str) -> bool:
    facts = _load(state)
    part = _clean(body_part).lower()
    remaining = [f for f in facts["pain_flags"] if f.get("body_part") != part]
    if len(remaining) == len(facts["pain_flags"]):
        return False
    facts["pain_flags"] = remaining
    _store(state, facts)
    return True


def active_pain_flags(state: AgentState) -> list[dict]:
    return _load(state)["pain_flags"]


def set_preference(state: AgentState, topic: str, value: str) -> None:
    """Remember a coaching preference, e.g. topic='humor', value='none'."""
    facts = _load(state)
    preferences = facts["preferences"]
    key = _clean(topic).lower()
    preferences.pop(key, None)
    preferences[key] = _clean(value)
    while len(preferences) > MAX_PREFERENCES:
        preferences.pop(next(iter(preferences)))
    facts["preferences"] = preferences
    _store(state, facts)


def athlete_facts_line(state: AgentState, today: date | None = None) -> str | None:
    """One line for any prompt: active pain reports and preferences, or None."""
    facts = _load(state)
    today = today or date.today()
    parts = []
    if facts["pain_flags"]:
        pains = "; ".join(
            f"{f['body_part']} ({f['note']}, {_when(f['reported_at'], today)})"
            if f.get("note")
            else f"{f['body_part']} ({_when(f['reported_at'], today)})"
            for f in facts["pain_flags"]
        )
        parts.append(f"pain reported: {pains}")
    if facts["preferences"]:
        prefs = ", ".join(f"{topic}: {value}" for topic, value in facts["preferences"].items())
        parts.append(f"preferences: {prefs}")
    if not parts:
        return None
    return "ATHLETE FACTS — " + " | ".join(parts)


def safety_line(state: AgentState) -> str | None:
    """Instruction for coaching prompts while any pain report is open, else None."""
    flags = _load(state)["pain_flags"]
    if not flags:
        return None
    parts = ", ".join(f["body_part"] for f in flags)
    return (
        f"SAFETY: the athlete reported pain ({parts}). No hype; never encourage pushing "
        "through pain, and tell them to stop if it hurts."
    )
