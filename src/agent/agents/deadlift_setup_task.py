"""DeadliftSetupTask - asks a scheduled deadlift session's setup (grip, plates, belt, shoes)
before the camera starts. The quick path asks the same in CollectDeadliftInfoTask.
"""

from __future__ import annotations

import logging

from livekit.agents import AgentTask, function_tool, llm

from agent.agents.shared.affect_mixin import AffectNodesMixin
from agent.agents.shared.base_agent import build_agent_instructions
from agent.agents.shared.deadlift_session import (
    DEFAULT_PLATE_DIAMETER_CM,
    SETUP_QUESTION,
    DeadliftSessionMeta,
    GripType,
    ShoeType,
)
from agent.agents.shared.userdata import UserData
from agent.core.agent_state import AgentState

logger = logging.getLogger(__name__)

TASK_INSTRUCTIONS = (
    "# Deadlift Setup\n"
    "Today's workout includes the conventional deadlift. Before the first set, find out their "
    "setup. " + SETUP_QUESTION + " Once they answer, or skip it, call record_deadlift_setup "
    "with what they said. Don't coach or brief yet."
)


class DeadliftSetupTask(AffectNodesMixin, AgentTask[DeadliftSessionMeta]):
    def __init__(
        self, state: AgentState, userdata: UserData, chat_ctx: llm.ChatContext | None = None,
    ) -> None:
        super().__init__(
            instructions=build_agent_instructions(state, TASK_INSTRUCTIONS),
            chat_ctx=chat_ctx,
        )
        self.state = state
        self.userdata = userdata

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions=(
                "Before the first set: they have deadlifts today. " + SETUP_QUESTION
                + " Vary the wording."
            )
        )

    @function_tool
    async def record_deadlift_setup(
        self,
        grip: GripType | None = None,
        belt: bool | None = None,
        shoes: ShoeType | None = None,
        plate_diameter_cm: float | None = None,
    ) -> None:
        """
        Call this once the user has told you their deadlift setup, or wants to skip it.
        Leave out anything they didn't say.

        Args:
            grip: "double" for double overhand, "mixed" for one hand over and one under, "hook" for hook grip
            belt: True if they wear a lifting belt, False if not
            shoes: "flat" for flat thin soles, "barefoot" for socks or barefoot, "heeled" for raised-heel lifting shoes, "cushioned" for running shoes or trainers
            plate_diameter_cm: Only if they use plates smaller than standard 45 centimetre plates
        """
        meta = DeadliftSessionMeta(
            grip=grip,
            belt=belt,
            shoes=shoes,
            plate_diameter_cm=plate_diameter_cm or DEFAULT_PLATE_DIAMETER_CM,
        )
        logger.info(f"[DEADLIFT] Session setup recorded: {meta.model_dump(exclude_none=True)}")
        self.complete(meta)
