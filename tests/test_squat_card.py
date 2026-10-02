"""Tests for the squat knowledge card the voice coach answers technique questions from."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.services.squat_card import (
    GRAPH_DIR,
    NEVER_VISIBLE,
    OBSERVABILITY_TOPIC,
    SQUAT_CARD,
    ExplainSquatMixin,
    build_squat_card,
    explain_squat_topic,
)
from biomechanics.faults.observability import SINGLE_CAMERA, TRIANGULATED

CARD_TOKEN_BUDGET = 1500
CHARS_PER_TOKEN = 4


def _symptoms() -> dict:
    with open(GRAPH_DIR / "symptoms.yaml") as graph_file:
        return yaml.safe_load(graph_file)


def _causes() -> dict:
    with open(GRAPH_DIR / "causes.yaml") as graph_file:
        return yaml.safe_load(graph_file)


class TestCardBuild:
    def test_every_graph_symptom_has_a_section(self) -> None:
        card = build_squat_card()
        for symptom_id in _symptoms():
            assert symptom_id in card

    def test_symptom_section_is_generated_from_the_yaml(self) -> None:
        symptoms = _symptoms()
        causes = _causes()
        heel_rise = symptoms["heel_rise"]
        top_cause = max(heel_rise["candidate_causes"], key=lambda entry: entry["prior"])
        assert heel_rise["description"] in SQUAT_CARD["heel_rise"]
        assert causes[top_cause["cause_id"]]["description"] in SQUAT_CARD["heel_rise"]

    def test_card_stays_under_token_budget(self) -> None:
        card_chars = sum(len(topic) + len(section) for topic, section in SQUAT_CARD.items())
        assert card_chars / CHARS_PER_TOKEN <= CARD_TOKEN_BUDGET

    def test_card_lists_what_no_setup_can_see(self) -> None:
        section = SQUAT_CARD[OBSERVABILITY_TOPIC].lower()
        for invisible in ("butt wink", "upper-back rounding", "bracing", "foot arch"):
            assert invisible in section
        assert len(NEVER_VISIBLE) == 4

    def test_side_view_symptoms_are_marked_rig_only(self) -> None:
        assert "three-camera rig only" in SQUAT_CARD["hip_shoot"]
        assert "three-camera rig only" in SQUAT_CARD["balance_forward"]
        assert "three-camera rig only" not in SQUAT_CARD["knee_not_tracking_toes"]


class TestExplainSquatTopic:
    @pytest.mark.parametrize(
        ("question", "topic"),
        [
            ("why do my heels come up?", "heel_rise"),
            ("is butt wink bad?", "butt_wink"),
            ("should I go heavier?", "go_heavier"),
            ("my knees cave in", "knee_not_tracking_toes"),
            ("is it fine if my knees go past my toes", "knees_past_toes"),
            ("can you see my back?", OBSERVABILITY_TOPIC),
            ("heel_rise", "heel_rise"),
        ],
    )
    def test_question_returns_matching_section(self, question: str, topic: str) -> None:
        assert SQUAT_CARD[topic] in explain_squat_topic(question, capture_mode=SINGLE_CAMERA)

    def test_unknown_topic_lists_topics_and_what_can_be_seen(self) -> None:
        answer = explain_squat_topic("what about my grip", capture_mode=SINGLE_CAMERA)
        assert "Topics with notes" in answer
        assert SQUAT_CARD[OBSERVABILITY_TOPIC] in answer

    def test_answer_names_the_camera_setup(self) -> None:
        assert "one camera" in explain_squat_topic("depth", capture_mode=SINGLE_CAMERA)
        assert "three-camera rig" in explain_squat_topic("depth", capture_mode=TRIANGULATED)

    def test_explain_squat_tool_returns_card_section(self) -> None:
        answer = asyncio.run(ExplainSquatMixin().explain_squat("is butt wink bad?"))
        assert SQUAT_CARD["butt_wink"] in answer
