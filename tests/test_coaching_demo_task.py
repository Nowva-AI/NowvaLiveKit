"""Tests for the instructions the choreographed demo task answers mid-demo questions with."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.agents.coaching_demo_task import CoachingDemoTask
from agent.agents.prompts.base_prompt import BASE_PROMPT


def _demo_task() -> CoachingDemoTask:
    async def _build() -> CoachingDemoTask:
        cues = [{"cue_index": 0, "cause_id": "knee_track_cue", "explanation": "Knees out", "magnitude_text": "a little"}]
        return CoachingDemoTask(cues=cues, lines_task=None, send_to_pipeline_fn=lambda message: None)

    return asyncio.run(_build())


class TestDemoInstructions:
    def test_demo_inherits_the_base_prompt_like_every_agent(self) -> None:
        assert _demo_task().instructions.startswith(BASE_PROMPT)

    def test_demo_instructions_carry_the_cue_summary(self) -> None:
        instructions = _demo_task().instructions
        assert "1. Knees out (a little)" in instructions
        assert "NEVER advance to the next correction" in instructions
