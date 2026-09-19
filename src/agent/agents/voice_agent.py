"""
Nova Voice Agent - Multi-Agent Entrypoint
Routes to the appropriate agent based on persisted mode.
"""

import asyncio
import glob
import logging
import os
import signal
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add parent directory (src/) to path when running as subprocess
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

load_dotenv()

from livekit import agents
from livekit.agents import AgentSession, TurnHandlingOptions, AgentStateChangedEvent, MetricsCollectedEvent, inference, metrics
from openai.types import Reasoning
from livekit.agents.voice.room_io import RoomInputOptions
from livekit.plugins import deepgram, openai, silero, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from agent.core.agent_state import AgentState, set_state_notify_fd, PROJECT_ROOT
from agent.agents.onboarding_agent import OnboardingAgent
from agent.agents.main_menu_agent import MainMenuAgent
from agent.agents.workout_agent import WorkoutAgent
from agent.agents.program_creation_agent import ProgramCreationAgent
from agent.agents.schedule_agent import ScheduleMaintenanceAgent
from agent.agents.shared.userdata import UserData
from agent.core.latency_tracker import LatencyTracker
from agent.services.compaction_service import CompactionService
from agent.services.context_viewer import ContextViewer
from profiler.collector import SessionProfiler

logger = logging.getLogger(__name__)


async def entrypoint(ctx: agents.JobContext):
    """Main entry point for Nova voice agent"""

    logger.info("[NOVA] Entrypoint function called")

    usage_collector = metrics.UsageCollector()

    async def log_usage(reason: str):
        summary = usage_collector.get_summary()
        logger.info("Usage summary: %s", summary)

    ctx.add_shutdown_callback(log_usage)


    notify_fd_str = os.environ.get("NOWVA_STATE_NOTIFY_FD")
    if notify_fd_str is not None:
        set_state_notify_fd(int(notify_fd_str))
        logger.info(f"[NOVA] State notify pipe fd={notify_fd_str}")

    # Discover user_id from room metadata or most recent state file
    logger.info("[NOVA] Checking for user_id in room metadata...")
    user_id = ctx.room.metadata.get('user_id') if ctx.room.metadata else None
    logger.info(f"[NOVA] user_id from metadata: {user_id}")

    fresh_onboarding = os.environ.get("NOWVA_FRESH_ONBOARDING") == "1"
    if not user_id and not fresh_onboarding:
        logger.info("[NOVA] Searching for state files...")
        # Anchor to the project root — AgentState writes there regardless of CWD
        state_files = glob.glob(str(PROJECT_ROOT / '.agent_state_*.json'))
        logger.info(f"[NOVA] Found {len(state_files)} state files")
        if state_files:
            latest_state = max(state_files, key=os.path.getmtime)
            user_id = Path(latest_state).name.replace('.agent_state_', '').replace('.json', '')
            logger.info(f"[NOVA] Found recent state file for user: {user_id}")
    elif fresh_onboarding:
        logger.info("[NOVA] Fresh onboarding — skipping state file scan")

    # Initialize state
    logger.info(f"[NOVA] Creating AgentState with user_id: {user_id}...")
    state = AgentState(user_id=user_id)
    logger.info(f"[NOVA] Starting with mode: {state.get_mode()}")
    if user_id:
        logger.info(f"[NOVA] Loaded existing user: {user_id}")

    # Create shared userdata
    userdata = UserData(state=state, room=ctx.room)

    # Data-flush safety net: the session "close" event does not always fire
    # when the process is terminated during shutdown, but shutdown callbacks
    # run during LiveKit's graceful drain. All stop() methods are idempotent,
    # so running both paths is safe.
    async def flush_session_services(reason: str):
        for name in ("context_viewer", "compaction_service", "coaching_service", "affect_service"):
            obj = getattr(userdata, name, None)
            if obj:
                try:
                    await obj.stop()
                    setattr(userdata, name, None)
                    logger.info(f"[SHUTDOWN] {name} stopped via shutdown callback")
                except Exception as e:
                    logger.error(f"[SHUTDOWN] Failed to stop {name}: {e}")

    ctx.add_shutdown_callback(flush_session_services)

    # Retrieve prewarmed AudioCueService (if available)
    prewarmed_cue_svc = ctx.proc.userdata.get("audio_cue_service")
    if prewarmed_cue_svc is not None:
        userdata.audio_cue_service = prewarmed_cue_svc
        logger.info("[NOVA] Prewarmed AudioCueService attached to userdata")

    # Retrieve prewarmed WakeWordModel (if available)
    prewarmed_ww = ctx.proc.userdata.get("wakeword_model")
    if prewarmed_ww is not None:
        userdata.wakeword_model = prewarmed_ww
        logger.info("[NOVA] Prewarmed WakeWordModel attached to userdata")

    # Best-effort bridge to the display page (idle orb / wake word indicator)
    from agent.services.visual_bridge import VisualBridge
    visual_bridge = VisualBridge()
    visual_bridge.start()
    userdata.visual_bridge = visual_bridge

    async def close_visual_bridge():
        await visual_bridge.aclose()

    ctx.add_shutdown_callback(close_visual_bridge)

    # Initialize cascade pipeline components (STT + LLM + TTS)
    logger.info("[NOVA] Initializing cascade pipeline...")
    stt = deepgram.STT(
        model="nova-3",
        language="en",
        keyterm=[
            "Barbell Back Squat", "Romanian Deadlift", "Barbell Bench Press",
            "Barbell Overhead Press", "Barbell Front Squat", "Goblet Squat",
            "Sumo Deadlift", "Barbell Deadlift",
            "reps", "sets", "RPE", "deload", "hypertrophy",
        ],
    )

    llm_model = os.getenv("LLM_MODEL", "gpt-5.4-mini")
    if llm_model.startswith(("gpt-5.5", "gpt-5.6")):
        # gpt-5.5+ rejects reasoning_effort + function tools on /v1/chat/completions;
        # OpenAI requires the Responses API for this combination.
        llm = openai.responses.LLM(
            model=llm_model,
            reasoning=Reasoning(effort="low"),
        )
    else:
        llm = openai.LLM(
            model=llm_model,
            reasoning_effort="low",
        )

    # Cartesia via LiveKit Inference — billed to LiveKit Cloud credits,
    # authenticated with LIVEKIT_API_KEY/SECRET (no Cartesia key needed).
    tts = inference.TTS(
        model="cartesia/sonic-3",
        voice=os.getenv("CARTESIA_VOICE_ID", "3e39e9a5-585c-4f5f-bac6-5e4905c51095"),
        language="en",
    )
    logger.info("[NOVA] Cascade pipeline initialized")

    # Retrieve prewarmed VAD or load fresh as fallback
    vad = ctx.proc.userdata.get("vad")
    if vad is None:
        logger.warning("[NOVA] No prewarmed VAD found, loading fresh (~100-500ms)...")
        vad = silero.VAD.load()
    else:
        logger.info("[NOVA] Using prewarmed Silero VAD")

    # Speech affect / effort perception: wrap the VAD so every user utterance
    # reaches the AffectService without touching STT, turn detection or agents.
    from affect.config import load_affect_config
    from agent.services.affect_service import AffectService
    from agent.services.affect_vad_tap import TappedVAD

    affect_config = load_affect_config()
    affect_service = AffectService(
        affect_config,
        ctx.proc.userdata.get("affect_engine"),
        state=state,
        profiler=SessionProfiler.get_instance(),
        visual_bridge=visual_bridge,
        coaching_speaking_fn=lambda: bool(
            getattr(getattr(userdata, "coaching_service", None), "is_coaching_speaking", False)
        ),
        user_id=user_id,
    )
    userdata.affect_service = affect_service
    if affect_service.enabled:
        vad = TappedVAD(vad, affect_service)
        logger.info(
            f"[NOVA] Affect perception enabled: model={affect_service.engine.manifest.version} "
            f"provider={affect_service.engine.provider} adapter={affect_service.style_adapter.name}"
        )
    else:
        logger.info("[NOVA] Affect perception disabled (AFFECT_ENABLED=0 or no model)")

    # Create agent session with cascade pipeline
    logger.info("[NOVA] Creating agent session...")
    session = AgentSession(
        stt=stt,
        llm=llm,
        tts=tts,
        vad=vad,
        turn_handling=TurnHandlingOptions(
            turn_detection=MultilingualModel(),
            endpointing={"min_delay": 0.3},
            preemptive_generation={"enabled": True, "preemptive_tts": True},
        ),
        userdata=userdata,
    )
    logger.info("[NOVA] Agent session created")

    # Initialize compaction service for rolling context summarization
    compaction_service = CompactionService(
        session=session,
        state=state,
        user_id=user_id or "guest",
    )
    userdata.compaction_service = compaction_service

    # Select agent based on persisted mode
    mode = state.get_mode()
    agent_map = {
        "onboarding": OnboardingAgent,
        "main_menu": MainMenuAgent,
        "workout": WorkoutAgent,
        "program_creation": ProgramCreationAgent,
        "schedule": ScheduleMaintenanceAgent,
    }
    AgentClass = agent_map.get(mode, OnboardingAgent)
    if os.environ.get("NOWVA_TEST_ASSESS") == "1" and mode == "workout":
        from agent.agents.teaching_agent import TeachingAgent
        AgentClass = TeachingAgent
        logger.info("[NOVA] Test-assess fast path — starting TeachingAgent directly")
    agent = AgentClass(state=state, userdata=userdata)

    logger.info(f"[NOVA] Starting {AgentClass.__name__} (mode={mode})")

    # Initialize latency tracker
    latency_tracker = LatencyTracker.get_instance()
    latency_tracker.reset()

    # Initialize session profiler (no-op when NOWVA_PROFILE != "1")
    profiler = SessionProfiler.get_instance()
    profiler.start()
    profiler.record("agent", "mode_change", from_mode=None, to_mode=mode)

    # Flush profiler on SIGTERM so data is written even if LiveKit's
    # session-close event never fires (e.g. process killed during shutdown).
    try:
        _prev_sigterm = signal.getsignal(signal.SIGTERM)

        def _sigterm_handler(signum, frame):
            profiler.stop()
            if callable(_prev_sigterm) and _prev_sigterm not in (signal.SIG_DFL, signal.SIG_IGN):
                _prev_sigterm(signum, frame)
            else:
                raise SystemExit(0)

        signal.signal(signal.SIGTERM, _sigterm_handler)
    except ValueError:
        pass

    # --- Session event tracking ---
    @session.on("agent_state_changed")
    def _on_agent_state(ev):
        logger.info(f"[SESSION] Agent state: {ev.old_state} → {ev.new_state}")
        profiler.record("agent", "state_change", old=str(ev.old_state), new=str(ev.new_state))
        visual_bridge.send_agent_state(str(ev.new_state))

    @session.on("user_state_changed")
    def _on_user_state(ev):
        logger.info(f"[SESSION] User state: {ev.old_state} → {ev.new_state}")
        profiler.record("turn", "user_state", old=str(ev.old_state), new=str(ev.new_state))
        old = str(ev.old_state).lower()
        new = str(ev.new_state).lower()
        if "idle" in old and "speaking" in new:
            profiler.begin_turn()
            profiler.record_user_speech_start()
        elif "speaking" in old and "idle" in new:
            profiler.record_user_speech_end()

    @session.on("user_input_transcribed")
    def _on_user_input(ev):
        if not ev.is_final and not ev.transcript.strip():
            return  # skip empty partials (VAD speech-start with no text yet)
        final_tag = "FINAL" if ev.is_final else "partial"
        logger.info(f"[SESSION] User speech [{final_tag}]: {ev.transcript}")
        if ev.is_final:
            profiler.record_transcript(ev.transcript)
            profiler.record("turn", "transcript", text=ev.transcript, is_final=True)

    @session.on("conversation_item_added")
    def _on_conversation_item(ev):
        item = ev.item
        role = getattr(item, "role", "unknown")
        text = getattr(item, "text_content", None)
        if callable(text):
            text = text()
        if text:
            logger.info(f"[SESSION] Conversation item ({role}): {text[:200]}")
        else:
            logger.info(f"[SESSION] Conversation item ({role}): [non-text content]")

    @session.on("speech_created")
    def _on_speech_created(ev):
        logger.info(
            f"[SESSION] Speech created — source={ev.source}, "
            f"user_initiated={ev.user_initiated}, "
            f"speech_id={ev.speech_handle.id}"
        )
        profiler.record("turn", "speech_created", source=str(ev.source), user_initiated=ev.user_initiated)
        profiler.record_speech_created()

    # TODO(v2.0): migrate to ChatMessage.metrics for per-turn latency
    _lk_logger = logging.getLogger("livekit.agents.voice.agent_session")
    _prev_level = _lk_logger.level
    _lk_logger.setLevel(logging.ERROR)
            
    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

        m = ev.metrics
        if hasattr(m, "ttft"):
            logger.info(
                f"[METRICS] {m.type} — ttft={m.ttft:.3f}s, duration={m.duration:.3f}s, "
                f"cancelled={m.cancelled}, input_tokens={getattr(m, 'input_tokens', 'N/A')}, "
                f"output_tokens={getattr(m, 'output_tokens', 'N/A')}, "
                f"tps={getattr(m, 'tokens_per_second', 'N/A')}"
            )
            latency_tracker.record_ttft(m.ttft)
            profiler.record_llm_metrics(
                ttft=m.ttft,
                duration=m.duration,
                input_tokens=getattr(m, "input_tokens", None),
                output_tokens=getattr(m, "output_tokens", None),
                tokens_per_second=getattr(m, "tokens_per_second", None),
                cancelled=m.cancelled,
            )
        elif hasattr(m, "audio_duration"):
            logger.info(
                f"[METRICS] {m.type} — duration={m.duration:.3f}s, "
                f"audio_duration={m.audio_duration:.3f}s"
            )
            profiler.record_tts_metrics(duration=m.duration, audio_duration=m.audio_duration)
            profiler.finalize_turn()
        else:
            logger.info(f"[METRICS] {m.type}")

    _lk_logger.setLevel(_prev_level)

    # --- Session usage tracking (feeds SessionLogger) ---
    from agent.core.session_logger import SessionLogger
    _session_logger = SessionLogger.get_instance()
    _prev_llm_usage: dict[str, int] = {}

    @session.on("session_usage_updated")
    def _on_usage(ev):
        for mu in ev.usage.model_usage:
            if mu.type != "llm_usage":
                continue
            key = f"{mu.provider}/{mu.model}"
            prev_input = _prev_llm_usage.get(f"{key}/in", 0)
            prev_output = _prev_llm_usage.get(f"{key}/out", 0)
            prev_cached = _prev_llm_usage.get(f"{key}/cached", 0)
            delta_input = mu.input_tokens - prev_input
            delta_output = mu.output_tokens - prev_output
            delta_cached = max(0, mu.input_cached_tokens - prev_cached)
            _prev_llm_usage[f"{key}/in"] = mu.input_tokens
            _prev_llm_usage[f"{key}/out"] = mu.output_tokens
            _prev_llm_usage[f"{key}/cached"] = mu.input_cached_tokens
            if delta_input > 0 or delta_output > 0:
                _session_logger.log_llm_call(
                    component="cascade_pipeline",
                    model=mu.model,
                    input_tokens=delta_input,
                    output_tokens=delta_output,
                    cached_tokens=delta_cached,
                )
                logger.info(
                    f"[USAGE] {mu.model} — "
                    f"Δin={delta_input:,} Δout={delta_output:,} "
                    f"(cumulative: in={mu.input_tokens:,} out={mu.output_tokens:,})"
                )

    @session.on("function_tools_executed")
    def _on_tools_executed(ev):
        for call, output in ev.zipped():
            result_str = str(output.output)[:150] if output else "None"
            logger.info(f"[SESSION] Tool executed: {call.name}({call.arguments}) → {result_str}")
            profiler.record("tool", "executed", name=call.name, args=str(call.arguments)[:200])

    @session.on("error")
    def _on_error(ev):
        logger.error(f"[SESSION] Error from {ev.source}: {ev.error}")
        profiler.record("error", "session_error", source=str(ev.source), error=str(ev.error))

    @session.on("close")
    def _on_close(ev):
        logger.info(f"[SESSION] Session closed — reason={ev.reason.value}, error={ev.error}")
        latency_tracker.log_summary()
        profiler.stop()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning("[SESSION] No running event loop — skipping async cleanup")
            if state.get_mode() == "workout":
                state.save_state()
            return

        async def _async_cleanup():
            for name, obj in [
                ("context_viewer", getattr(userdata, 'context_viewer', None)),
                ("compaction_service", getattr(userdata, 'compaction_service', None)),
                ("coaching_service", getattr(userdata, 'coaching_service', None) if state.get_mode() == "workout" else None),
                ("affect_service", getattr(userdata, 'affect_service', None)),
            ]:
                if obj:
                    try:
                        await obj.stop()
                        setattr(userdata, name, None)
                        logger.info(f"[SESSION] {name} stopped on close")
                    except Exception as e:
                        logger.error(f"[SESSION] Failed to stop {name}: {e}")
            if state.get_mode() == "workout":
                state.save_state()
                logger.info("[SESSION] Workout state saved on close")

        asyncio.run_coroutine_threadsafe(_async_cleanup(), loop)

    await ctx.connect()

    await session.start(
        room=ctx.room,
        agent=agent,
        record=True,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
            pre_connect_audio=True,
            pre_connect_audio_timeout=5.0,
        ),
    )

    # Start compaction service after session is live
    if userdata.compaction_service:
        await userdata.compaction_service.start()
        logger.info("[NOVA] Compaction service started")

    # Start context viewer debug dashboard (http://localhost:8899)
    context_viewer = ContextViewer(
        session=session,
        compaction_service=userdata.compaction_service,
        state=state,
    )
    userdata.context_viewer = context_viewer
    await context_viewer.start()

    logger.info(f"Nova voice agent started in room: {ctx.room.name}")


def prewarm(proc: agents.JobProcess):
    """Pre-load heavy resources before any room connection.

    Runs silero VAD model loading and audio cue disk I/O concurrently
    so neither blocks the other.  Results are stored on proc.userdata
    for retrieval in entrypoint().
    """
    import concurrent.futures
    import time

    logger.info("[PREWARM] Starting parallel pre-load (VAD + audio cues + wake word + affect)...")
    start = time.monotonic()

    def _load_vad():
        return silero.VAD.load()

    def _load_affect():
        from affect.config import load_affect_config
        from affect.engine import AffectEngine

        config = load_affect_config()
        if not config.enabled:
            return None
        model_dir = config.resolve_model_dir()
        if not (model_dir / "model.onnx").exists():
            logger.warning(f"[PREWARM] No affect model at {model_dir} — affect perception disabled")
            return None
        engine = AffectEngine(model_dir, config.engine)
        engine.initialize()
        return engine

    def _load_audio_cues():
        from agent.services.audio_cue_service import AudioCueService
        return AudioCueService(session=None)

    def _load_wakeword():
        model_path = os.environ.get("WAKE_WORD_MODEL_PATH", "models/hey_nova.onnx")
        if Path(model_path).exists():
            from livekit.wakeword import WakeWordModel
            return WakeWordModel(models=[model_path])
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        vad_future = executor.submit(_load_vad)
        cue_future = executor.submit(_load_audio_cues)
        ww_future = executor.submit(_load_wakeword)
        affect_future = executor.submit(_load_affect)

        try:
            proc.userdata["vad"] = vad_future.result(timeout=10)
            logger.info("[PREWARM] Silero VAD pre-loaded")
        except Exception as e:
            logger.warning(f"[PREWARM] VAD pre-load failed (will load in entrypoint): {e}")

        try:
            cue_svc = cue_future.result(timeout=30)
            proc.userdata["audio_cue_service"] = cue_svc
            logger.info(
                f"[PREWARM] Audio cues pre-loaded: {len(cue_svc._memory_cache)} cue keys, "
                f"{sum(len(v) for v in cue_svc._memory_cache.values())} variants"
            )
        except Exception as e:
            logger.warning(f"[PREWARM] Audio cue pre-load failed (will retry at session start): {e}")

        try:
            ww_model = ww_future.result(timeout=10)
            if ww_model is not None:
                proc.userdata["wakeword_model"] = ww_model
                logger.info("[PREWARM] WakeWordModel pre-loaded")
            else:
                logger.info("[PREWARM] No wake word model found (will run without wake word detection)")
        except Exception as e:
            logger.warning(f"[PREWARM] WakeWordModel pre-load failed: {e}")

        try:
            affect_engine = affect_future.result(timeout=90)
            if affect_engine is not None:
                proc.userdata["affect_engine"] = affect_engine
                logger.info(
                    f"[PREWARM] Affect engine pre-loaded: {affect_engine.manifest.version} "
                    f"provider={affect_engine.provider}"
                )
            else:
                logger.info("[PREWARM] Affect engine not loaded (disabled or no model)")
        except Exception as e:
            logger.warning(f"[PREWARM] Affect engine pre-load failed: {e}")

    elapsed = time.monotonic() - start
    logger.info(f"[PREWARM] Parallel pre-load complete in {elapsed:.3f}s")


if __name__ == "__main__":
    import sys

    try:
        agents.cli.run_app(
            agents.WorkerOptions(
                entrypoint_fnc=entrypoint,
                prewarm_fnc=prewarm,
            )
        )
    except KeyboardInterrupt:
        logger.info("[SHUTDOWN] Agent stopped by user")
        sys.exit(0)
    except Exception as e:
        if "termios" not in str(e).lower():
            logger.error(f"[ERROR] {e}")
        sys.exit(0)
