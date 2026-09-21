"""Tests for the pre-TTS text normalizer backstop."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.services.tts_normalizer import normalize_stream, normalize_tts_text


class TestSymbolExpansion:
    def test_degree_symbol_becomes_word(self) -> None:
        assert normalize_tts_text("toward 22°.") == "toward 22 degrees."

    def test_percent_becomes_word(self) -> None:
        assert normalize_tts_text("80% clean") == "80 percent clean"

    def test_multiplication_sign_becomes_times(self) -> None:
        assert normalize_tts_text("1.5× shoulder width") == "1.5 times shoulder width"

    def test_sets_by_reps_shorthand(self) -> None:
        assert normalize_tts_text("3x8 at 135") == "3 by 8 at 135"

    def test_score_slash_becomes_out_of(self) -> None:
        assert normalize_tts_text("scored 84/100 today") == "scored 84 out of 100 today"

    def test_date_slashes_preserved(self) -> None:
        assert normalize_tts_text("on 04/20/2023") == "on 04/20/2023"

    def test_numeric_range_becomes_to(self) -> None:
        assert normalize_tts_text("rest 90–120 seconds") == "rest 90 to 120 seconds"

    def test_arrow_becomes_to(self) -> None:
        assert normalize_tts_text("overall 72 -> 78") == "overall 72 to 78"


class TestMarkupStripping:
    def test_bold_asterisks_removed(self) -> None:
        assert normalize_tts_text("did you mean **20 seconds**?") == "did you mean 20 seconds?"

    def test_heading_marker_removed(self) -> None:
        assert normalize_tts_text("# Recap\nGood set.") == "Recap Good set."

    def test_bullet_markers_removed(self) -> None:
        assert normalize_tts_text("- Workouts: 3\n- Sets: 12") == "Workouts: 3 Sets: 12"

    def test_snake_case_becomes_spaces(self) -> None:
        assert normalize_tts_text("knee_valgus showed up") == "knee valgus showed up"

    def test_newlines_collapse_to_spaces(self) -> None:
        assert normalize_tts_text("Nice set.\n\nReady?") == "Nice set. Ready?"


class TestEmojiStripping:
    def test_emoji_removed(self) -> None:
        assert normalize_tts_text("Let's go! 💪") == "Let's go! "

    def test_smiley_emoji_removed(self) -> None:
        assert normalize_tts_text("Glad it landed! 😄") == "Glad it landed! "

    def test_text_emoticon_removed(self) -> None:
        assert normalize_tts_text("Great job :) keep going") == "Great job keep going"


class TestBracketedAsides:
    def test_model_aside_removed(self) -> None:
        assert (
            normalize_tts_text("Are you sore?[no further response]") == "Are you sore?"
        )

    def test_stage_direction_removed(self) -> None:
        assert normalize_tts_text("Nice. [pauses] Reset.") == "Nice. Reset."

    def test_unclosed_bracket_left_alone(self) -> None:
        assert normalize_tts_text("Set 1 [of 3") == "Set 1 [of 3"


def _stream_through_normalizer(chunks: list[str]) -> list[str]:
    async def source():
        for chunk in chunks:
            yield chunk

    async def collect() -> list[str]:
        return [chunk async for chunk in normalize_stream(source())]

    return asyncio.run(collect())


class TestMarkupSplitAcrossChunks:
    """LLM tokens split markup, so no single chunk ever holds a whole tag."""

    def test_tokenized_aside_removed(self) -> None:
        chunks = ["Are", " you", " sore", "?", "[", "no", " further", " response", "]"]
        assert "".join(_stream_through_normalizer(chunks)) == "Are you sore?"

    def test_tokenized_laughter_arrives_whole(self) -> None:
        out = _stream_through_normalizer(["Ha", " [", "laughter", "]", " fair", "."])
        assert "[laughter]" in out
        assert "".join(out) == "Ha [laughter] fair."

    def test_tokenized_break_tag_arrives_whole(self) -> None:
        out = _stream_through_normalizer(["Ready?", "<break", " time=", '"400ms"', "/>", " Go."])
        assert '<break time="400ms"/>' in out

    def test_plain_text_is_not_held_back(self) -> None:
        assert _stream_through_normalizer(["Chest", " up", "."]) == ["Chest", " up", "."]

    def test_unclosed_bracket_is_flushed_at_stream_end(self) -> None:
        assert "".join(_stream_through_normalizer(["Set 1 ", "[of 3"])) == "Set 1 [of 3"

    def test_runaway_bracket_stops_being_held(self) -> None:
        long_tail = "x" * 100
        out = _stream_through_normalizer(["Go [", long_tail, " still talking"])
        assert "".join(out) == f"Go [{long_tail} still talking"
        assert len(out) > 1


class TestInlineTagsPreserved:
    def test_laughter_tag_preserved(self) -> None:
        assert normalize_tts_text("[laughter] Noted.") == "[laughter] Noted."

    def test_laughs_tag_preserved(self) -> None:
        assert normalize_tts_text("Okay [laughs] fine.") == "Okay [laughs] fine."

    def test_break_tag_preserved(self) -> None:
        text = 'Ready?<break time="400ms"/> Go.'
        assert normalize_tts_text(text) == text


class TestStreamBehavior:
    def test_chunk_edges_not_trimmed(self) -> None:
        async def chunks():
            yield "hello "
            yield "world"

        async def collect() -> str:
            return "".join([c async for c in normalize_stream(chunks())])

        assert asyncio.run(collect()) == "hello world"

    def test_pure_emoji_chunk_dropped(self) -> None:
        async def chunks():
            yield "💪"
            yield "go"

        async def collect() -> list[str]:
            return [c async for c in normalize_stream(chunks())]

        assert asyncio.run(collect()) == ["go"]
