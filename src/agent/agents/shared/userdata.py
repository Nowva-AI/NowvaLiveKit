"""
Shared UserData dataclass for passing ephemeral session state across agent handoffs.
"""

from dataclasses import dataclass, field
from typing import Optional, Any
from agent.core.agent_state import AgentState


@dataclass
class UserData:
    """Shared state passed via session.userdata across all agent handoffs.

    AgentState handles persistent (file-based) state.
    UserData holds ephemeral session-level state that must survive
    agent handoffs but should not be persisted to disk.
    """
    state: AgentState
    room: Any = None  # rtc.Room reference

    # Onboarding-specific tracking (transient within session)
    temp_first_name: Optional[str] = None
    temp_email: Optional[str] = None
    first_name_confirmed: bool = False
    email_confirmed: bool = False
    first_name_retry_count: int = 0
    email_retry_count: int = 0
    max_retries: int = 3

    # Coaching service (live object reference, set by WorkoutAgent)
    coaching_service: Any = None

    # Prewarmed AudioCueService (set by prewarm, used by CoachingService)
    audio_cue_service: Any = None

    # Prewarmed WakeWordModel (set by prewarm, used by WorkoutAgent)
    wakeword_model: Any = None

    # VisualBridge to the display page (set by voice_agent entrypoint)
    visual_bridge: Any = None

    # Compaction service (live object reference, set by voice_agent entrypoint)
    compaction_service: Any = None

    # AffectService (speech emotion / effort perception, set by voice_agent entrypoint)
    affect_service: Any = None

    # Context summarization state (used by ProgramCreationAgent)
    current_token_count: int = 0
    is_summarizing: bool = False
    summary_count: int = 0
    openai_client: Any = None  # Lazy-initialized OpenAI client for summarization
