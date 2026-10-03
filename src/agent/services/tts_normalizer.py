"""
Deterministic backstop that strips written-text artifacts from agent speech
before it reaches TTS. Prompt rules are the primary defense; this catches
stragglers (emoji, markdown, unit symbols, ISO dates) that would otherwise be
read aloud. Inline tags like <break time="400ms"/> and [laughter] pass through.
Streamed text is released only at word boundaries, so a pattern the LLM splits
across tokens ("90", "-", "120") converts as if it had arrived whole.
"""

from __future__ import annotations

import re
from datetime import date
from typing import AsyncIterable

# The voice reads the brand as spelled ("Now-va"); it is pronounced No-va.
_BRAND_RE = re.compile(r"\bNowva\b")

MONTH_NAMES = (
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
)
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")

# Unit abbreviations after a number. Seconds only up to three digits, so "1990s" stays a decade.
_UNIT_WORDS = {
    "kg": "kilogram", "kgs": "kilogram",
    "lb": "pound", "lbs": "pound",
    "ms": "millisecond", "s": "second",
}
_WEIGHT_UNIT_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s?(kgs?|lbs?)\b", re.IGNORECASE)
_TIME_UNIT_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:\.\d+)?)\s?(ms|s)\b")
_APPROX_RE = re.compile(r"~\s*(?=\d)")
_REP_MAX_RE = re.compile(r"(?<!\w)(\d+)\s?RM\b")
_AMRAP_RE = re.compile(r"\bAMRAP\b")
# Tags are passed through verbatim: "400ms" inside <break time="400ms"/> is not speech.
_ANGLE_TAG_RE = re.compile(r"(<[^<>]*>)")

# Emoji, pictographs, dingbats, flags, and joiners. Conservative ranges —
# arrows and math operators are handled by targeted replacements instead.
_EMOJI_RE = re.compile(
    "["
    "\U0001f000-\U0001faff"  # emoticons, pictographs, transport, supplemental
    "☀-➿"          # misc symbols and dingbats
    "⬀-⯿"          # misc symbols and arrows (stars, circles)
    "︀-️"          # variation selectors
    "‍"                 # zero-width joiner
    "]+"
)

# Text emoticons like :) ;-) :D at a word boundary.
_EMOTICON_RE = re.compile(r"(?:^|(?<=\s))[:;]-?[)(DPpOo](?=\s|$|[.,!?])")

_MARKDOWN_HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_MARKDOWN_BULLET_RE = re.compile(r"^\s*[-*•]\s+", re.MULTILINE)
# "^" also matches at the start of a streamed segment, which is mid-line unless the
# previous one ended a line; this sentinel shields that position.
_MID_LINE_SENTINEL = "\x00"

_DIGIT_RANGE_RE = re.compile(r"(?<=\d)\s*[–—-]\s*(?=\d)")
_DIGIT_TIMES_RE = re.compile(r"(?<=\d)\s*[×]\s*")
_DIGIT_X_DIGIT_RE = re.compile(r"(?<=\d)x(?=\d)")
# Only score-style slashes ("84/100", "7/10") — never dates like 04/20/2023.
_SCORE_RE = re.compile(r"(?<=\d)/(?=100\b|10\b)")
_ARROW_RE = re.compile(r"\s*(?:->|→|⇒)\s*")

# Models occasionally emit asides like "[no further response]" or "(pauses)".
# Anything bracketed that is not an allowed inline tag would be spoken aloud.
_ALLOWED_TAGS = ("laughter", "laughs", "sigh", "breath")
MAX_HELD_MARKUP_CHARS = 80
_BRACKET_ASIDE_RE = re.compile(
    r"\[(?!\s*(?:" + "|".join(_ALLOWED_TAGS) + r")\s*\])[^\]]{0,80}\]",
    re.IGNORECASE,
)


def _spoken_date(match: re.Match) -> str:
    try:
        day = date(int(match[1]), int(match[2]), int(match[3]))
    except ValueError:
        return match[0]
    return f"{MONTH_NAMES[day.month - 1]} {day.day}, {day.year}"


def _spoken_unit(match: re.Match) -> str:
    number, unit = match[1], match[2].lower()
    word = _UNIT_WORDS[unit]
    return f"{number} {word}" if number == "1" else f"{number} {word}s"


def _speak_numbers(text: str) -> str:
    text = _ISO_DATE_RE.sub(_spoken_date, text)
    text = _APPROX_RE.sub("about ", text)
    text = _REP_MAX_RE.sub(r"\1-rep max", text)
    text = _AMRAP_RE.sub("as many reps as possible", text)
    text = _WEIGHT_UNIT_RE.sub(_spoken_unit, text)
    return _TIME_UNIT_RE.sub(_spoken_unit, text)


def normalize_tts_text(text: str, at_line_start: bool = True) -> str:
    text = _BRAND_RE.sub("Nova", text)
    text = "".join(
        part if index % 2 else _speak_numbers(part)
        for index, part in enumerate(_ANGLE_TAG_RE.split(text))
    )

    # Symbol expansions first, while the surrounding digits are intact.
    text = text.replace("°", " degrees")
    text = text.replace("%", " percent")
    text = _DIGIT_TIMES_RE.sub(" times ", text)
    text = _DIGIT_X_DIGIT_RE.sub(" by ", text)
    text = _DIGIT_RANGE_RE.sub(" to ", text)
    text = _SCORE_RE.sub(" out of ", text)
    text = _ARROW_RE.sub(" to ", text)

    # Written-text markup that TTS would read aloud.
    if not at_line_start:
        text = _MID_LINE_SENTINEL + text
    text = _MARKDOWN_HEADER_RE.sub("", text)
    text = _MARKDOWN_BULLET_RE.sub("", text)
    text = text.removeprefix(_MID_LINE_SENTINEL)
    text = text.replace("**", "").replace("*", "").replace("`", "")
    text = re.sub(r"(?<=\w)_(?=\w)", " ", text)

    text = _BRACKET_ASIDE_RE.sub("", text)
    text = _EMOJI_RE.sub("", text)
    text = _EMOTICON_RE.sub("", text)

    # Speech has no paragraphs. Collapse newlines and space runs within the
    # chunk only — never trim chunk edges, or streamed words would join.
    text = text.replace("\n", " ")
    text = re.sub(r"  +", " ", text)
    return text


# Text may be released after whitespace that follows a finished word (a letter or
# closing punctuation). After a number or a symbol it may not: "90" + "-120" is a
# range and "100" + " kg" a unit, so those wait for the next chunk.
_WORD_BOUNDARY_RE = re.compile(r"(?:(?<=[^\W\d_])|(?<=[.,!?;:'\")\]>]))\s+")


def _unclosed_markup_index(text: str) -> int | None:
    for opener, closer in (("[", "]"), ("<", ">")):
        index = text.rfind(opener)
        if index != -1 and closer not in text[index:]:
            return index
    return None


def _release_index(held: str) -> int:
    cut = 0
    for match in _WORD_BOUNDARY_RE.finditer(held):
        cut = match.end()
    # Never split a tag or a bracketed aside: "[no further response]" streams as
    # "[", "no", " further", ... and must reach the aside pattern whole.
    while (opener := _unclosed_markup_index(held[:cut])) is not None:
        cut = opener
    if cut == 0 and len(held) > MAX_HELD_MARKUP_CHARS:
        return len(held)
    return cut


async def normalize_stream(text: AsyncIterable[str]) -> AsyncIterable[str]:
    # Hold back the trailing partial token (at most MAX_HELD_MARKUP_CHARS of it) so
    # every pattern is matched on whole words; tags like [laughter] or <break .../>
    # also reach the voice adapters in one piece.
    held = ""
    at_line_start = True
    async for chunk in text:
        held += chunk
        cut = _release_index(held)
        if cut == 0:
            continue
        ready, held = held[:cut], held[cut:]
        cleaned = normalize_tts_text(ready, at_line_start)
        at_line_start = ready.rstrip(" \t").endswith("\n")
        if cleaned:
            yield cleaned
    cleaned = normalize_tts_text(held, at_line_start)
    if cleaned:
        yield cleaned
