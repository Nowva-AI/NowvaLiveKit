"""Tests for the pure parts of the cue audio pipeline scripts (no network).

draft_cue_text.py drafts the spoken lines into cues.json; generate_cue_audio.py
speaks them into the WAV clips AudioCueService loads.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import wave
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from agent.services import audio_cue_service  # noqa: E402
from biomechanics.coaching.cue_cache import SQUAT_CUES  # noqa: E402


def _load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


draft = _load_script("draft_cue_text")
generate = _load_script("generate_cue_audio")


class TestDraftScenarios:
    def test_every_squat_cue_gets_lines(self):
        for cue_key in SQUAT_CUES:
            if cue_key.startswith("rep_"):
                assert len(draft.rep_count_lines(cue_key)) == draft.VARIANTS_PER_CUE
            else:
                assert draft.scenario_for(cue_key)

    def test_side_scenario_names_the_side(self):
        scenario = draft.scenario_for("knees_out_left")
        assert scenario.startswith(draft.CUE_SCENARIOS["knees_out"])
        assert "left knee" in scenario

    def test_rep_count_lines_are_just_the_number(self):
        assert draft.rep_count_lines("rep_3") == ["Three!"] * draft.VARIANTS_PER_CUE

    def test_scenarios_avoid_jargon(self):
        for cue_key, scenario in draft.CUE_SCENARIOS.items():
            assert "valgus" not in scenario.lower(), cue_key


class TestAgentSideCues:
    def test_every_playable_cue_gets_drafted(self):
        from agent.services.coaching_constants import CUE_TEXT_MAP
        from biomechanics.profiles import PROFILE_REGISTRY

        # Other exercises' cue keys are voiced once those exercises are coached.
        other_exercise_cues = {
            key
            for profile_class in set(PROFILE_REGISTRY.values())
            if profile_class.name != "squat"
            for key in profile_class().get_cue_dict()
        } - set(SQUAT_CUES)
        agent_side = [
            key for key in CUE_TEXT_MAP
            if key not in SQUAT_CUES and not key.startswith("rep_") and key not in other_exercise_cues
        ]
        assert agent_side
        for cue_key in agent_side:
            assert cue_key in draft.cue_keys_to_draft(), cue_key
            assert draft.scenario_for(cue_key), cue_key

    def test_fix_confirmation_builds_on_its_fault(self):
        scenario = draft.scenario_for("knees_out_fixed")
        assert scenario.startswith(draft.CUE_SCENARIOS["knees_out"])
        assert "held" in scenario

    def test_agent_side_keys_follow_the_squat_cues(self):
        keys = draft.cue_keys_to_draft()
        assert keys[: len(SQUAT_CUES)] == list(SQUAT_CUES)
        assert len(keys) == len(set(keys))


class TestDraftPrompt:
    def test_prompt_asks_for_three_short_lines_as_json(self):
        prompt = draft.build_cue_prompt("drive")
        assert draft.CUE_SCENARIOS["drive"] in prompt
        assert "exactly 3" in prompt
        assert f"at most {draft.MAX_CUE_WORDS} words" in prompt
        assert '{"lines"' in prompt

    def test_explain_cues_get_a_full_sentence(self):
        prompt = draft.build_cue_prompt("stance_explain")
        assert f"10 to {draft.MAX_EXPLAIN_WORDS} words" in prompt

    def test_model_defaults_to_astra(self):
        assert draft.DEFAULT_MODEL == "gpt-6-astra"


class TestParseVariants:
    def test_valid_response_is_cleaned(self):
        lines = draft.parse_variants('{"lines": [" Knees out ", "Spread  the floor", "Push them out"]}')
        assert lines == ["Knees out", "Spread the floor", "Push them out"]

    @pytest.mark.parametrize("response_text", [
        '{"lines": ["one", "two"]}',
        '{"lines": "Knees out"}',
        '["a", "b", "c"]',
        '{"lines": ["a", "", "c"]}',
    ])
    def test_wrong_shape_is_rejected(self, response_text):
        with pytest.raises(ValueError):
            draft.parse_variants(response_text)


class TestLineLength:
    def test_dash_does_not_count_as_a_word(self):
        assert draft.spoken_word_count("Drifting left — stay centered") == 4

    def test_over_length_lines_are_flagged(self):
        lines = ["Knees out", "Push your knees out wide now"]
        assert draft.over_length_lines("knees_out", lines) == ["Push your knees out wide now"]

    def test_explain_cues_allow_a_sentence(self):
        line = "That's coming from your stance, so step your feet out wider."
        assert draft.over_length_lines("stance_explain", [line]) == []


class TestMergeCueLines:
    def test_drafted_lines_replace_and_reviewed_edits_survive(self):
        existing = {"drive": ["Drive up"] * 3, "knees_out": ["Edited by hand"] * 3}
        drafted = {"drive": ["Drive hard", "Push up fast", "Explode up"]}
        merged = draft.merge_cue_lines(existing, drafted)
        assert merged["drive"] == ["Drive hard", "Push up fast", "Explode up"]
        assert merged["knees_out"] == ["Edited by hand"] * 3

    def test_keys_follow_squat_cue_order_with_extras_last(self):
        merged = draft.merge_cue_lines({"custom_cue": ["x"] * 3, "drive": ["y"] * 3}, {"knees_out": ["z"] * 3})
        assert list(merged) == ["knees_out", "drive", "custom_cue"]

    def test_review_page_escapes_lines(self):
        page = draft.build_review_page({"drive": ["<b>Drive</b>", "Up", "Go"]})
        assert "&lt;b&gt;Drive&lt;/b&gt;" in page
        assert "<b>Drive</b>" not in page


class TestClipNaming:
    @pytest.mark.parametrize("cue_key", ["knees_out", "knees_out_left", "rep_12", "square_feet_right"])
    def test_filename_round_trips_through_the_loader_rule(self, cue_key):
        filename = generate.wav_filename(cue_key, 2)
        assert filename == f"{cue_key}_2.wav"
        assert generate.parse_wav_filename(filename) == (cue_key, 2)

    def test_non_clip_files_are_not_parsed(self):
        assert generate.parse_wav_filename("review.html") is None
        assert generate.parse_wav_filename("knees_out_final.wav") is None

    def test_stale_variants_are_only_this_keys_extra_takes(self):
        filenames = [
            "knees_out_0.wav", "knees_out_2.wav", "knees_out_3.wav", "knees_out_9.wav",
            "knees_out_left_5.wav",
        ]
        assert generate.stale_variant_files("knees_out", 3, filenames) == [
            "knees_out_3.wav", "knees_out_9.wav",
        ]


class TestSynthesisDecisions:
    def test_new_clip_needs_synthesis(self):
        assert generate.needs_synthesis(None, "Knees out", "voice", force=False)

    def test_unchanged_clip_is_skipped(self):
        entry = {"text": "Knees out", "voice_id": "voice", "model": generate.TTS_MODEL}
        assert not generate.needs_synthesis(entry, "Knees out", "voice", force=False)

    @pytest.mark.parametrize("text,voice_id", [("Knees wide", "voice"), ("Knees out", "other")])
    def test_edited_line_or_new_voice_is_respoken(self, text, voice_id):
        entry = {"text": "Knees out", "voice_id": "voice", "model": generate.TTS_MODEL}
        assert generate.needs_synthesis(entry, text, voice_id, force=False)

    def test_force_respeaks_everything(self):
        entry = {"text": "Knees out", "voice_id": "voice", "model": generate.TTS_MODEL}
        assert generate.needs_synthesis(entry, "Knees out", "voice", force=True)


class TestAudioFormat:
    def test_payload_requests_flash_with_the_live_voice_settings(self):
        payload = generate.build_tts_payload("Knees out")
        assert payload["model_id"] == "eleven_flash_v2_5"
        assert payload["text"] == "Knees out"
        assert payload["voice_settings"] == {
            "stability": generate.ELEVENLABS_DEFAULT_STABILITY,
            "similarity_boost": generate.ELEVENLABS_DEFAULT_SIMILARITY,
            "speed": generate.ELEVENLABS_DEFAULT_SPEED,
        }

    def test_output_format_is_raw_pcm_at_the_loader_rate(self):
        assert generate.OUTPUT_FORMAT == f"pcm_{audio_cue_service.SAMPLE_RATE}"

    def test_wav_matches_what_the_loader_assumes(self):
        wav_bytes = generate.pcm_to_wav_bytes(b"\x00\x01" * 240)
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            assert wav_file.getframerate() == audio_cue_service.SAMPLE_RATE
            assert wav_file.getnchannels() == audio_cue_service.NUM_CHANNELS
            assert wav_file.getsampwidth() == 2
            assert wav_file.getnframes() == 240

    def test_clips_go_where_the_loader_reads(self):
        assert generate.CUES_WAV_DIR.resolve() == audio_cue_service.CUES_WAV_DIR.resolve()


class TestLoadCueLines:
    def test_valid_file_loads(self, tmp_path):
        path = tmp_path / "cues.json"
        path.write_text(json.dumps({"drive": ["Drive hard", "Push up", "Go"]}))
        assert generate.load_cue_lines(path) == {"drive": ["Drive hard", "Push up", "Go"]}

    @pytest.mark.parametrize("content", [
        ["Drive"],
        {"drive": "Drive hard"},
        {"drive": []},
        {"drive": ["Drive", "  "]},
    ])
    def test_malformed_file_is_rejected(self, tmp_path, content):
        path = tmp_path / "cues.json"
        path.write_text(json.dumps(content))
        with pytest.raises(ValueError):
            generate.load_cue_lines(path)

    def test_review_page_marks_missing_clips(self):
        page = generate.build_review_page(
            {"drive": ["Drive hard", "Push up", "Go"]}, 3, "voice", {"drive_0.wav"},
        )
        assert "wav/drive_0.wav" in page
        assert page.count('<span class="missing">') == 2
