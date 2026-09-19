"""AgentTask that delivers the choreographed coaching demo after a failed assessment."""

from __future__ import annotations

import asyncio
import logging
from typing import Callable

from livekit.agents import AgentTask

from agent.services.coaching_constants import COACHING_PERSONA
from agent.services.demo_narration import (
    NARRATION_TIMEOUT_SECONDS,
    build_fallback_script,
)
from agent.agents.shared.affect_mixin import AffectNodesMixin

logger = logging.getLogger(__name__)

MORPH_IN_WAIT_SECONDS = 0.9
STARTED_ACK_TIMEOUT_SECONDS = 4.0
INTER_CUE_PAUSE_SECONDS = 0.4
MAX_LINE_REPLAYS = 2


class DemoStartAck:
    """Signal that the viewer actually began the morph-in, with the ack time."""

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self.acked_at: float | None = None

    def set(self) -> None:
        if self.acked_at is None:
            self.acked_at = asyncio.get_running_loop().time()
        self._event.set()

    async def wait(self, timeout: float) -> bool:
        try:
            await asyncio.wait_for(self._event.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

_DEMO_INSTRUCTIONS_TEMPLATE = (
    f"{COACHING_PERSONA} You are mid-demonstration of squat-form "
    "corrections. The screen shows the user's own skeleton animating each correction. "
    "The demo script is driven for you — NEVER advance to the next correction or end "
    "the demo yourself. If the user asks a question, answer briefly (2 sentences max) "
    "referencing what is on screen, then stop talking so the demo can resume.\n"
    "The corrections being shown, in order:\n{cue_summary}"
)


class CoachingDemoTask(AffectNodesMixin, AgentTask):
    def __init__(
        self,
        cues: list[dict],
        lines_task: asyncio.Task | None,
        send_to_pipeline_fn: Callable[[dict], None],
        started_ack: DemoStartAck | None = None,
        chat_ctx=None,
    ) -> None:
        cue_summary = "\n".join(
            f"{cue['cue_index'] + 1}. {cue.get('explanation', cue['cause_id'])} "
            f"({cue.get('magnitude_text', '')})"
            for cue in cues
        )
        super().__init__(
            instructions=_DEMO_INSTRUCTIONS_TEMPLATE.format(cue_summary=cue_summary),
            chat_ctx=chat_ctx,
        )
        self._cues = cues
        self._lines_task = lines_task
        self._send = send_to_pipeline_fn
        self._started_ack = started_ack

    async def on_enter(self) -> None:
        try:
            await self._run_demo()
        except Exception:
            logger.exception("[DEMO TASK] Demo failed mid-run")
        finally:
            if not self.done():
                self.complete(None)

    async def _run_demo(self) -> None:
        loop = asyncio.get_running_loop()
        start_time = loop.time()

        # Morph-in starts immediately; it visually covers script generation.
        self._send({"type": "demo_start"})

        script = None
        if self._lines_task is not None:
            try:
                script = await asyncio.wait_for(self._lines_task, NARRATION_TIMEOUT_SECONDS)
            except Exception:
                logger.exception("[DEMO TASK] Script generation failed — using fallback")
        if script is None:
            script = build_fallback_script(self._cues)

        # Hold the intro until the viewer confirms the morph-in actually began;
        # fall back to dead-reckoning from demo_start if no ack arrives.
        morph_started_at = start_time
        if self._started_ack is not None:
            ack_remaining = STARTED_ACK_TIMEOUT_SECONDS - (loop.time() - start_time)
            if await self._started_ack.wait(max(0.0, ack_remaining)):
                morph_started_at = self._started_ack.acked_at
            else:
                logger.warning("[DEMO TASK] No viewer started-ack — dead-reckoning intro")

        morph_remaining = MORPH_IN_WAIT_SECONDS - (loop.time() - morph_started_at)
        if morph_remaining > 0:
            await asyncio.sleep(morph_remaining)

        await self._say_line(script.intro)

        for cue in self._cues:
            cue_index = cue["cue_index"]
            self._send({"type": "demo_cue", "cue_index": cue_index})
            await self._say_line(script.cue_lines[cue_index])
            await asyncio.sleep(INTER_CUE_PAUSE_SECONDS)

        # Morph-out and readiness reset run underneath the outro speech.
        self._send({"type": "demo_end"})
        await self._say_line(script.outro)

    async def _say_line(self, line: str) -> None:
        for _ in range(MAX_LINE_REPLAYS + 1):
            handle = self.session.say(line, allow_interruptions=True)
            await handle.wait_for_playout()
            if not handle.interrupted:
                return
            # User spoke: the session answers with this task's LLM; the
            # replayed line schedules after that reply.
            logger.info("[DEMO TASK] Cue line interrupted — replaying after answer")
