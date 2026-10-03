"""
WorkoutAgent - Handles active workout sessions with wake word system and coaching
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import re
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from livekit import rtc
from livekit.agents import RunContext, StopResponse, get_job_context, llm
from livekit.agents.llm import function_tool

from agent.agents.deadlift_setup_task import DeadliftSetupTask
from agent.agents.prompts import get_workout_prompt
from agent.agents.shared.base_agent import BaseNovaAgent
from agent.agents.shared.deadlift_card import explain_deadlift
from agent.agents.shared.deadlift_session import (
    DEADLIFT_PROFILE_NAME,
    EXERCISE_META_STATE_KEY,
    deadlift_form_findings,
    first_session_briefing_instructions,
    is_coached_deadlift,
    session_includes_coached_deadlift,
    store_exercise_meta,
)
from agent.services.athlete_facts import (
    active_pain_flags,
    add_pain_flag,
    athlete_facts_line,
    resolve_pain_flag,
)
from agent.services.squat_card import ExplainSquatMixin
from biomechanics.faults.observability import OBSERVABLE
from db.database import SessionLocal

logger = logging.getLogger(__name__)

# Live workout state + athlete facts, one system item per conversational turn.
WORKOUT_STATE_ITEM_ID = "nowva_workout_state"

# A live frame older than this can't describe how the athlete is standing now.
FORM_SNAPSHOT_MAX_AGE_MS = 3000.0

RPE_MIN = 1.0
RPE_MAX = 10.0
RPE_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
RPE_NUMBER_PATTERN = re.compile(r"\b(\d+(?:\.\d+)?|" + "|".join(RPE_NUMBER_WORDS) + r")\b")
RPE_SCALE_PATTERN = re.compile(r"\bout of (?:10|ten)\b")
# Effort words -> RPE, checked in order ("very hard" before "hard"). Roughly reps left:
# easy ~4, medium ~3, hard ~1-2, very hard ~1, max 0.
EFFORT_WORD_RPE: tuple[tuple[str, float], ...] = (
    ("very hard", 9.0),
    ("brutal", 9.0),
    ("all out", 10.0),
    ("max", 10.0),
    ("failure", 10.0),
    ("hard", 8.5),
    ("tough", 8.5),
    ("heavy", 8.5),
    ("medium", 7.0),
    ("moderate", 7.0),
    ("solid", 7.0),
    ("easy", 6.0),
    ("light", 6.0),
)

# Wake word ONNX detection parameters (matching livekit-wakeword internals)
_WW_SAMPLE_RATE = 16_000
_WW_STRIDE_SAMPLES = 1280         # 80 ms between predictions, also local mic blocksize
_WW_CHUNK_SECONDS = 2.0
_WW_CHUNK_SAMPLES = int(_WW_CHUNK_SECONDS * _WW_SAMPLE_RATE)
# A genuine phrase ramps and holds a high score across strides; false positives
# tend to be single-stride spikes. Require the previous stride to clear this too.
_WW_CONFIRM_SCORE = 0.5


def _assess_standing_setup(angles: dict) -> list[str]:
    """Judge the setup the lifter can actually change while standing."""
    from agent.services.coaching_orchestrator import (
        STANCE_TOLERANCE,
        TOE_OUT_TOLERANCE_DEG,
    )

    findings: list[str] = []

    stance = angles.get("stance_width_ratio")
    target_stance = angles.get("target_stance_ratio", 0.0)
    if stance is not None and target_stance > 0:
        delta = target_stance - stance
        if abs(delta) <= STANCE_TOLERANCE:
            findings.append("their stance width is right where you want it")
        elif delta > 0:
            findings.append("their stance is still narrower than their target")
        else:
            findings.append("their stance is wider than their target")

    toe_l = angles.get("foot_direction_angle_l")
    toe_r = angles.get("foot_direction_angle_r")
    target_toe = angles.get("target_toe_out_deg", 0.0)
    if toe_l is not None and toe_r is not None and target_toe > 0:
        delta = target_toe - (toe_l + toe_r) / 2.0
        if abs(delta) <= TOE_OUT_TOLERANCE_DEG:
            findings.append("their toe angle is on target")
        elif delta > 0:
            findings.append("their toes need to turn out a bit more")
        else:
            findings.append("their toes are turned out further than needed")

    return findings


def _assess_last_rep(verdict: dict) -> list[str]:
    """Relay the engine's verdict on the last rep. Faults this camera setup can only
    roughly read are left out: a coach who can't see something says nothing about it."""
    from agent.services.progress_context import fault_label

    findings = [
        f"their last rep showed {fault_label(fault['fault_type'])}"
        + (f" on the {fault['side']} side" if fault.get("side") in ("left", "right") else "")
        for fault in verdict.get("faults", [])
        if fault.get("observability", OBSERVABLE) == OBSERVABLE
    ]
    if not findings:
        findings.append("their last rep had nothing to correct in what this camera can see")
    return findings


def _deadlift_form_reply(rep_message: dict | None) -> str:
    """The check_my_form answer for a deadlift, from its last rep. Behaviour only: no camera
    sees the spine, so the back is never judged either way."""
    findings = deadlift_form_findings(rep_message)
    if not findings:
        return (
            "Tell the user you haven't seen a deadlift rep from them yet — ask them to pull "
            "one and ask again right after. One short sentence, no verdict on their form."
        )
    return (
        f"The user asked how their deadlift looks. What you measured on their last rep: "
        f"{'; '.join(findings)}. Relay this in 1-2 short sentences as a coach — plain words, "
        f"no jargon, no number except the bar speed. Lead with whatever is already right, "
        f"then the one thing to change. Never say their back rounded or stayed flat: no "
        f"camera sees the spine."
    )


def _wake_word_uses_local_mic() -> bool:
    """Console jobs have no room audio track to tap, so they listen on the local mic.
    WAKE_WORD_LOCAL_MIC=1 or 0 overrides the detection."""
    override = os.environ.get("WAKE_WORD_LOCAL_MIC")
    if override in ("0", "1"):
        return override == "1"
    job_ctx = get_job_context(required=False)
    return job_ctx is not None and job_ctx.is_fake_job()


def _effort_to_rpe(effort: str) -> float | None:
    # Speech-to-text spells numbers out ("eight"); "out of ten" is the scale, not a rating.
    text = RPE_SCALE_PATTERN.sub("", effort.lower())
    numbers = [float(RPE_NUMBER_WORDS.get(n, n)) for n in RPE_NUMBER_PATTERN.findall(text)]
    if numbers:
        # "seven, maybe eight": the top of the range they gave
        rpe = max(numbers)
        return rpe if RPE_MIN <= rpe <= RPE_MAX else None
    for word, rpe in EFFORT_WORD_RPE:
        if re.search(rf"\b{word}\b", text):
            return rpe
    return None


class WorkoutAgent(ExplainSquatMixin, BaseNovaAgent):
    """Handles active workout sessions with wake word detection and coaching integration."""

    def __init__(self, state, userdata, from_calibration: bool = False) -> None:
        self._from_calibration = from_calibration

        # Wake word system state (agent-local)
        self._wake_word_active: bool = False
        self._wake_word_listening: bool = False
        self._wake_word_timeout_task: Optional[asyncio.Task] = None
        # Must outlast the turn detector's max endpointing delay (3.0s) so a
        # pending user turn can't commit after the window closes.
        self._wake_word_timeout_seconds: float = 3.0
        # LLM speech (recaps, motivation, rest lines) still playing or queued —
        # what "Hey Nova" cuts off. Cached cues and rep counts are never in here.
        self._llm_speech_handles: set = set()
        # Fire-and-forget tasks, held so the loop can't garbage-collect them mid-run
        self._background_tasks: set[asyncio.Task] = set()
        # Session turn options in effect before the workout, restored after it.
        self._saved_preemptive_enabled: bool | None = None

        # Wake word detection (ONNX default, Porcupine via WAKE_WORD_ENGINE=porcupine)
        self._ww_model = None
        self._porcupine = None
        self._ww_session = None  # session ref captured at start; Agent.session raises after shutdown
        self._ww_audio_stream: rtc.AudioStream | None = None
        self._ww_detection_task: asyncio.Task | None = None
        self._ww_executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._ww_threshold: float = float(os.environ.get("WAKE_WORD_THRESHOLD", "0.7"))
        self._ww_debounce: float = float(os.environ.get("WAKE_WORD_DEBOUNCE", "2.0"))
        self._ww_last_detection: float = 0.0
        self._ww_last_near_miss: float = 0.0

        # Only a coaching-ready deadlift adds the deadlift prompt section and explain_deadlift;
        # every other session keeps the squat's prompt and tool list.
        self._includes_deadlift = session_includes_coached_deadlift(state)
        super().__init__(
            state=state,
            userdata=userdata,
            instructions=get_workout_prompt(includes_deadlift=self._includes_deadlift),
            tools=[explain_deadlift] if self._includes_deadlift else None,
        )

    async def on_enter(self):
        """Start coaching service, generate greeting, then activate wake word system."""
        if self._from_calibration:
            # CalibrationAgent already started CoachingService and showed greeting.
            # Wire up the workout-complete callback and start wake word.
            coaching = self.userdata.coaching_service
            if coaching:
                coaching.set_workout_complete_callback(self._on_workout_complete_signal)
            await self._start_wake_word_system()
            return

        # Normal path — no calibration, create CoachingService fresh
        from agent.services.coaching_service import CoachingService
        coaching_service = CoachingService(
            session=self.session,
            state=self.state,
            room=self.userdata.room,
            on_workout_complete=self._on_workout_complete_signal,
            audio_cue_service=self.userdata.audio_cue_service,
        )
        await coaching_service.start()
        coaching_service._workout_active = True
        self.userdata.coaching_service = coaching_service

        # Last-session progress context and multi-session trends for the greeting
        from agent.services.progress_context import build_detailed_greeting_context
        baseline, fault_trends = await coaching_service.wait_progress_context()
        progress_line = build_detailed_greeting_context(baseline, fault_trends)
        progress_block = f"\n{progress_line}" if progress_line else ""

        # Generate context-aware greeting BEFORE starting wake word system.
        # _say() suppresses turn detection so the greeting can't be interrupted.
        # Don't restore — wake word system sets its own turn detection next.
        from agent.core.workout_session import WorkoutSession
        session_data = self.state.get("workout.current_session")
        if session_data:
            session = WorkoutSession.from_dict(session_data)
            first_desc = session.get_current_exercise_description()
            is_quick = session.is_quick_exercise
            if is_quick:
                await self._say(
                    f"[CONTEXT] quick exercise session just started, "
                    f"first exercise: {first_desc}{progress_block}\n\n"
                    "Greet the user into the workout, calm and direct, name "
                    "the first exercise, and get them moving. Two sentences "
                    "max. Vary your opener between sessions.",
                    restore=False,
                )
            else:
                workout_name = session_data.get("workout_name", "today's workout")
                await self._say(
                    f"[CONTEXT] workout just started: {workout_name}, "
                    f"first exercise: {first_desc}{progress_block}\n\n"
                    "Greet the user into the workout, calm and direct, name "
                    "the first exercise, and mention you're watching their "
                    "form. Two sentences max. Vary your opener between "
                    "sessions.",
                    restore=False,
                )
        else:
            await self._say(
                f"[CONTEXT] workout mode just started, no session details "
                f"available{progress_block}\n\n"
                "Greet the user into the workout, calm and direct, and get "
                "them moving. Two sentences max. Vary your opener between "
                "sessions.",
                restore=False,
            )

        if self._includes_deadlift:
            await self._prepare_deadlift()

        self.state.set("workout.greeting_done", True)
        self.state.save_state()
        logger.info("[WORKOUT] Greeting done — signalled main.py to start pose estimation")

        await self._start_wake_word_system()

    async def _prepare_deadlift(self) -> None:
        """Before the camera starts: the deadlift setup questions when the quick path
        hasn't asked them (scheduled path), then, when the workout opens with the deadlift
        and it is the athlete's first deadlift session, the setup briefing. No squat
        assessment runs for the deadlift."""
        opens_with_deadlift = is_coached_deadlift(self.state.get("workout.exercise_name"))
        first_session = (
            asyncio.create_task(self._is_first_deadlift_session()) if opens_with_deadlift else None
        )
        if self.state.get(EXERCISE_META_STATE_KEY) is None:
            self._restore_turn_detection()
            meta = await DeadliftSetupTask(
                state=self.state, userdata=self.userdata, chat_ctx=self.chat_ctx.copy(),
            )
            store_exercise_meta(self.state, meta)
            self.state.save_state()
        if first_session is not None and await first_session:
            await self._say(first_session_briefing_instructions(), restore=False)

    async def _is_first_deadlift_session(self) -> bool:
        from db.biomechanics_persistence import get_last_completed_session

        def _query() -> bool:
            db = SessionLocal()
            try:
                return get_last_completed_session(db, self.user_id, exercise=DEADLIFT_PROFILE_NAME) is None
            finally:
                db.close()

        try:
            return await asyncio.to_thread(_query)
        except Exception:
            # Unknown history: a returning lifter hearing the setup again beats a new one missing it.
            logger.exception("[DEADLIFT] Last-session lookup failed — briefing as a first session")
            return True

    # ===== COACHING SERVICE CALLBACKS =====

    async def _on_workout_complete_signal(self, data: dict):
        """Called by CoachingService when the entire workout is done."""
        logger.info("[COACHING] Workout complete signal received — scheduling cleanup")
        self._spawn(self._handle_workout_complete())

    def _spawn(self, coroutine) -> None:
        task = asyncio.create_task(coroutine)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _handle_workout_complete(self):
        """Run cleanup and agent handoff outside the orchestrator's task."""
        await self._cleanup_workout()
        # Brief pause to let the LLM finish processing pending conversation
        # events from the exercise recap before truncation.
        await asyncio.sleep(0.5)
        from agent.agents.main_menu_agent import MainMenuAgent
        self.session.update_agent(
            await self._carry_context_to(MainMenuAgent(state=self.state, userdata=self.userdata))
        )

    async def _cleanup_workout(self):
        """Shared cleanup for ending a workout (DB logging, state clearing)."""
        from db.schedule_utils import mark_workout_completed
        from db.progress_utils import log_completed_set
        from agent.core.workout_session import WorkoutSession

        session_data = self.state.get("workout.current_session")
        if session_data:
            try:
                session = WorkoutSession.from_dict(session_data)
                session.end_session()

                if not session.is_quick_exercise:
                    db = SessionLocal()
                    try:
                        for set_data in session.get_completed_sets_for_logging():
                            if set_data["performed_reps"] > 0:
                                log_completed_set(
                                    db=db,
                                    user_id=session.user_id,
                                    set_id=set_data["set_id"],
                                    performed_reps=set_data["performed_reps"],
                                    performed_weight=set_data.get("performed_weight"),
                                    rpe=set_data.get("rpe"),
                                    measured_velocity=set_data.get("measured_velocity")
                                )
                                logger.info(f"[WORKOUT] Logged set {set_data['set_id']}")

                        mark_workout_completed(db, session.schedule_id)
                        logger.info(f"[WORKOUT] Marked schedule {session.schedule_id} as completed")
                    except Exception:
                        logger.exception("[WORKOUT ERROR] Failed to save workout data")
                    finally:
                        db.close()
                else:
                    logger.info("[QUICK EXERCISE] Skipping DB logging for ad-hoc session")

            except Exception:
                logger.exception("[WORKOUT ERROR] Failed to process session")

        # Clear workout session, quick exercise, and calibration state
        self.state.set("workout.current_session", None)
        self.state.set("workout.exercise_name", None)
        self.state.set("workout.calibration_profile", None)
        self.state.set("quick_exercise.exercise_name", None)
        self.state.set("quick_exercise.gathering_params", False)
        self.state.set("calibration.active", None)
        self.state.set("calibration.movement_pattern", None)
        self.state.set("calibration.pending_workout", None)
        self.state.set("workout.greeting_done", False)
        self.state.set(EXERCISE_META_STATE_KEY, None)

        await self._stop_wake_word_system()

        if self.userdata.coaching_service:
            await self.userdata.coaching_service.stop()
            self.userdata.coaching_service = None

        self.state.switch_mode("main_menu")
        self.state.set("workout.active", False)
        self.state.save_state()

        logger.info("[STATE] Workout cleanup complete — switched to main_menu")

    # ===== WAKE WORD SYSTEM =====

    def _set_workout_turn_detection(self):
        """Workout mode: disable auto-responses, only respond to wake word."""
        try:
            self.session.input.set_audio_enabled(False)
            logger.info("[WAKE WORD] Turn detection disabled (workout mode)")
        except Exception as e:
            logger.error(f"[WAKE WORD] FAILED to set workout turn detection: {e}", exc_info=True)

    def _set_conversational_turn_detection(self):
        """Restore normal conversational turn detection."""
        try:
            self.session.input.set_audio_enabled(True)
            logger.info("[WAKE WORD] Turn detection restored (conversational mode)")
        except Exception as e:
            logger.error(f"[WAKE WORD] FAILED to set conversational turn detection: {e}", exc_info=True)

    def _set_active_listening_turn_detection(self):
        """Temporarily enable responses after wake word detection."""
        try:
            self.session.input.set_audio_enabled(True)
            logger.info("[WAKE WORD] Turn detection enabled (active listening)")
        except Exception as e:
            logger.error(f"[WAKE WORD] FAILED to set active listening turn detection: {e}", exc_info=True)

    def _on_speech_created_for_wake_word(self, ev):
        """Track LLM speech so "Hey Nova" can cut it off. Nothing is cancelled here:
        cached cues and rep counts arrive as say() speech and always play, and late
        turn replies are dropped in on_user_turn_completed before they exist."""
        logger.info(
            f"[WAKE WORD] speech_created event: user_initiated={ev.user_initiated} "
            f"source={getattr(ev, 'source', 'unknown')} "
            f"active={self._wake_word_active} listening={self._wake_word_listening}"
        )
        if ev.source != "generate_reply":
            return
        self._llm_speech_handles.add(ev.speech_handle)
        ev.speech_handle.add_done_callback(self._llm_speech_handles.discard)

    async def on_user_turn_completed(self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage) -> None:
        if self._wake_word_active and not self._wake_word_listening:
            # Dormant: a turn that committed after the listening window closed.
            logger.info("[WAKE WORD] ✗ Dropped user turn (dormant mode)")
            raise StopResponse()
        self._inject_workout_state(turn_ctx)
        await super().on_user_turn_completed(turn_ctx, new_message)

    def _inject_workout_state(self, turn_ctx: llm.ChatContext) -> None:
        """One fixed-id system item with the live workout state and athlete facts.

        turn_ctx is this reply's own copy, so the item never piles up in the
        conversation history; an existing one is replaced where it sits."""
        coaching = self.userdata.coaching_service
        lines = [
            coaching.workout_state_line() if coaching else None,
            athlete_facts_line(self.state),
        ]
        text = "\n".join(line for line in lines if line)
        item = llm.ChatMessage(id=WORKOUT_STATE_ITEM_ID, role="system", content=[text])
        for index, existing in enumerate(turn_ctx.items):
            if existing.id == WORKOUT_STATE_ITEM_ID:
                if text:
                    turn_ctx.items[index] = item
                else:
                    del turn_ctx.items[index]
                return
        if text:
            turn_ctx.items.append(item)

    async def _interrupt_llm_speech(self):
        """Cut off LLM coaching speech, playing or queued, and wait until it stops."""
        pending = [handle for handle in self._llm_speech_handles if not handle.done()]
        if not pending:
            return
        logger.info(f"[WAKE WORD] Interrupting {len(pending)} coaching LLM speech(es)")
        for handle in pending:
            handle.interrupt(force=True)
        await asyncio.gather(*(handle.wait_for_playout() for handle in pending))

    async def _activate_listening_mode(self):
        """Activate listening mode after wake word detection."""
        await self._interrupt_llm_speech()

        logger.info("[WAKE WORD] Activating listening mode")
        self._wake_word_listening = True
        if self.userdata.visual_bridge:
            self.userdata.visual_bridge.send_wake_event("detected")
        self._set_active_listening_turn_detection()

        await self.session.generate_reply(
            instructions=(
                "The user just said the wake word mid-workout to talk to you. Let them "
                "know you're listening in two to five words, a short acknowledgment that "
                "invites their question. Vary it every time, then wait."
            )
        )

        self._restart_wake_word_timeout()

    async def _deactivate_listening_mode(self):
        """Revert from active listening back to wake word detection mode."""
        logger.info("[WAKE WORD] Deactivating listening mode → back to wake word detection")
        self._wake_word_listening = False
        if self._wake_word_timeout_task:
            self._wake_word_timeout_task.cancel()
            self._wake_word_timeout_task = None
        if self.userdata.visual_bridge:
            self.userdata.visual_bridge.send_wake_event("dormant")
        try:
            self._ww_session.clear_user_turn()
        except Exception as e:
            logger.error(f"[WAKE WORD] Failed to clear pending user turn: {e}", exc_info=True)
        self._set_workout_turn_detection()
        logger.info("[WAKE WORD] Reverted to wake word mode")

    def _conversation_in_progress(self) -> bool:
        return (
            self._ww_session.agent_state in ("thinking", "speaking")
            or self._ww_session.user_state == "speaking"
        )

    async def _wake_word_timeout(self):
        """Revert to wake word mode once the conversation goes idle."""
        try:
            while True:
                await asyncio.sleep(self._wake_word_timeout_seconds)
                if not (self._wake_word_active and self._wake_word_listening):
                    return
                # Session already shut down (Ctrl+C race) — nothing to revert.
                if not getattr(self._ww_session, "_started", False):
                    return
                if self._conversation_in_progress():
                    continue
                logger.info("[WAKE WORD] Conversation idle — reverting to wake word mode")
                await self._deactivate_listening_mode()
                return
        except asyncio.CancelledError:
            pass

    def _restart_wake_word_timeout(self):
        if self._wake_word_timeout_task:
            self._wake_word_timeout_task.cancel()
        self._wake_word_timeout_task = asyncio.create_task(self._wake_word_timeout())

    def _on_agent_state_changed_for_wake_word(self, ev):
        """Auto-revert after wake-word-triggered responses complete."""
        if not self._wake_word_active or not self._wake_word_listening:
            return

        if ev.new_state in ("listening", "idle") and ev.old_state == "speaking":
            self._restart_wake_word_timeout()

    def _on_user_state_changed_for_wake_word(self, ev):
        """Keep the listening window open while the user is talking."""
        if not self._wake_word_active or not self._wake_word_listening:
            return
        self._restart_wake_word_timeout()

    def _on_user_transcript_for_wake_word(self, ev):
        """Keep the listening window open while a user turn is still endpointing."""
        if not self._wake_word_active or not self._wake_word_listening:
            return
        self._restart_wake_word_timeout()

    def _find_microphone_track(self, room) -> rtc.RemoteAudioTrack | None:
        """Find the first subscribed microphone track from any remote participant."""
        for participant in room.remote_participants.values():
            for pub in participant.track_publications.values():
                if (pub.source == rtc.TrackSource.SOURCE_MICROPHONE
                        and pub.track is not None
                        and pub.subscribed):
                    return pub.track
        return None

    def _on_track_subscribed_for_wakeword(self, track, publication, participant):
        """Start detection loop when a microphone track becomes available."""
        if not self._wake_word_active:
            return
        if publication.source != rtc.TrackSource.SOURCE_MICROPHONE:
            return
        if self._ww_detection_task is not None and not self._ww_detection_task.done():
            return
        logger.info(f"[WAKE WORD] Microphone track subscribed from {participant.identity}")
        self._start_detection_on_track(track)

    def _start_detection_on_track(self, track):
        """Create AudioStream from track and launch the detection loop."""
        self._ww_audio_stream = rtc.AudioStream(
            track,
            sample_rate=_WW_SAMPLE_RATE,
            num_channels=1,
        )
        self._ww_detection_task = asyncio.create_task(
            self._make_detection_loop(self._room_audio_frames())
        )
        logger.info("[WAKE WORD] Detection loop started on audio track")

    def _make_detection_loop(self, frames):
        if self._porcupine is not None:
            return self._porcupine_detection_loop(frames)
        return self._wake_word_detection_loop(frames)

    async def _room_audio_frames(self):
        async for event in self._ww_audio_stream:
            yield np.frombuffer(event.frame.data, dtype=np.int16)

    async def _local_mic_frames(self):
        import sounddevice as sd

        loop = asyncio.get_event_loop()
        frame_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=64)

        def _enqueue(data):
            try:
                frame_queue.put_nowait(data)
            except asyncio.QueueFull:
                pass

        def _on_audio(indata, frames, time_info, status):
            data = indata[:, 0].copy()
            try:
                loop.call_soon_threadsafe(_enqueue, data)
            except RuntimeError:
                pass

        stream = sd.InputStream(
            samplerate=_WW_SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=_WW_STRIDE_SAMPLES,
            callback=_on_audio,
        )
        stream.start()
        try:
            while True:
                yield await frame_queue.get()
        finally:
            stream.stop()
            stream.close()

    async def _wake_word_detection_loop(self, frames):
        """Buffer 16 kHz mono int16 audio and run ONNX wake word detection.

        Source frames can be any size (room tracks deliver ~10 ms frames,
        the local mic 80 ms) — audio is accumulated by sample count into a
        rolling 2-second window, with one prediction per 80 ms stride.
        """
        loop = asyncio.get_event_loop()
        window = np.zeros(_WW_CHUNK_SAMPLES, dtype=np.int16)
        samples_filled = 0
        samples_since_predict = 0
        prev_scores: dict[str, float] = {}

        try:
            async for frame_data in frames:
                if not self._wake_word_active:
                    break

                # Keeps running while Nova speaks, so "Hey Nova" can cut off a recap.
                if self._wake_word_listening:
                    samples_filled = 0
                    samples_since_predict = 0
                    prev_scores.clear()
                    continue

                n_samples = len(frame_data)
                if n_samples >= _WW_CHUNK_SAMPLES:
                    window[:] = frame_data[-_WW_CHUNK_SAMPLES:]
                else:
                    window[:-n_samples] = window[n_samples:]
                    window[-n_samples:] = frame_data
                samples_filled = min(samples_filled + n_samples, _WW_CHUNK_SAMPLES)
                samples_since_predict += n_samples

                if (samples_filled < _WW_CHUNK_SAMPLES
                        or samples_since_predict < _WW_STRIDE_SAMPLES):
                    continue
                samples_since_predict = 0

                try:
                    scores = await loop.run_in_executor(
                        self._ww_executor,
                        self._ww_model.predict,
                        window.copy(),
                    )
                except Exception as e:
                    logger.error(f"[WAKE WORD] Inference error: {e}")
                    continue

                now = time.monotonic()
                for name, score in scores.items():
                    prev_score = prev_scores.get(name, 0.0)
                    prev_scores[name] = score
                    if 0.4 <= score < self._ww_threshold:
                        if now - self._ww_last_near_miss >= 1.0:
                            self._ww_last_near_miss = now
                            logger.info(
                                f"[WAKE WORD] near miss '{name}' "
                                f"(confidence={score:.3f} < threshold={self._ww_threshold})"
                            )
                    if score >= self._ww_threshold:
                        if prev_score < _WW_CONFIRM_SCORE:
                            logger.info(
                                f"[WAKE WORD] Spike rejected '{name}' "
                                f"(confidence={score:.3f}, prev={prev_score:.3f} < {_WW_CONFIRM_SCORE})"
                            )
                            continue
                        if now - self._ww_last_detection >= self._ww_debounce:
                            self._ww_last_detection = now
                            samples_filled = 0
                            prev_scores.clear()
                            logger.info(
                                f"[WAKE WORD] ★ DETECTED '{name}' "
                                f"(confidence={score:.3f})"
                            )
                            self._spawn(self._activate_listening_mode())
                            break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[WAKE WORD] Detection loop crashed: {e}", exc_info=True)
        finally:
            await frames.aclose()
            if self._ww_audio_stream:
                await self._ww_audio_stream.aclose()
                self._ww_audio_stream = None
            logger.info("[WAKE WORD] Detection loop ended")

    async def _porcupine_detection_loop(self, frames):
        """Feed fixed-size frames to Porcupine's stateful streaming detector.

        Porcupine consumes exactly frame_length samples (512 @ 16 kHz) per
        call and returns a keyword index >= 0 on detection — no windowing,
        scoring, or spike rejection needed.
        """
        loop = asyncio.get_event_loop()
        frame_length = self._porcupine.frame_length
        buffer = np.zeros(0, dtype=np.int16)

        try:
            async for frame_data in frames:
                if not self._wake_word_active:
                    break

                if self._wake_word_listening:
                    buffer = np.zeros(0, dtype=np.int16)
                    continue

                buffer = np.concatenate([buffer, frame_data])
                while len(buffer) >= frame_length:
                    chunk = buffer[:frame_length]
                    buffer = buffer[frame_length:]
                    try:
                        keyword_index = await loop.run_in_executor(
                            self._ww_executor,
                            self._porcupine.process,
                            chunk,
                        )
                    except Exception as e:
                        logger.error(f"[WAKE WORD] Porcupine inference error: {e}")
                        continue

                    if keyword_index < 0:
                        continue
                    now = time.monotonic()
                    if now - self._ww_last_detection < self._ww_debounce:
                        continue
                    self._ww_last_detection = now
                    buffer = np.zeros(0, dtype=np.int16)
                    logger.info("[WAKE WORD] ★ DETECTED 'hey_nova' (porcupine)")
                    self._spawn(self._activate_listening_mode())
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[WAKE WORD] Detection loop crashed: {e}", exc_info=True)
        finally:
            await frames.aclose()
            if self._ww_audio_stream:
                await self._ww_audio_stream.aclose()
                self._ww_audio_stream = None
            logger.info("[WAKE WORD] Detection loop ended")

    def _create_porcupine(self):
        """Build a Porcupine handle from env config, or None to fall back to ONNX."""
        try:
            import pvporcupine
        except ImportError:
            logger.error("[WAKE WORD] pvporcupine not installed — falling back to ONNX")
            return None

        access_key = os.environ.get("PORCUPINE_ACCESS_KEY", "")
        if not access_key:
            logger.error("[WAKE WORD] PORCUPINE_ACCESS_KEY not set — falling back to ONNX")
            return None

        sensitivity = float(os.environ.get("PORCUPINE_SENSITIVITY", "0.5"))
        keyword_path = os.environ.get("PORCUPINE_KEYWORD_PATH", "")
        try:
            if keyword_path:
                handle = pvporcupine.create(
                    access_key=access_key,
                    keyword_paths=[keyword_path],
                    sensitivities=[sensitivity],
                )
                logger.info(f"[WAKE WORD] Porcupine ready (custom keyword: {keyword_path})")
            else:
                keyword = os.environ.get("PORCUPINE_KEYWORD", "porcupine")
                handle = pvporcupine.create(
                    access_key=access_key,
                    keywords=[keyword],
                    sensitivities=[sensitivity],
                )
                logger.info(f"[WAKE WORD] Porcupine ready (built-in keyword: '{keyword}')")
            return handle
        except Exception as e:
            logger.error(f"[WAKE WORD] Porcupine init failed: {e} — falling back to ONNX")
            return None

    def _wake_word_session_handlers(self) -> list[tuple[str, Callable]]:
        return [
            ("agent_state_changed", self._on_agent_state_changed_for_wake_word),
            ("speech_created", self._on_speech_created_for_wake_word),
            ("user_state_changed", self._on_user_state_changed_for_wake_word),
            ("user_input_transcribed", self._on_user_transcript_for_wake_word),
        ]

    async def _start_wake_word_system(self):
        """Activate wake word detection for workout mode."""
        if self._wake_word_active:
            await self._stop_wake_word_system()

        logger.info("[WAKE WORD] === Starting wake word system ===")

        if os.environ.get("WAKE_WORD_ENGINE", "onnx") == "porcupine":
            self._porcupine = self._create_porcupine()

        # ONNX path (default, and fallback if Porcupine init failed)
        if self._porcupine is None:
            # Load model (prefer prewarmed, fall back to disk)
            self._ww_model = getattr(self.userdata, "wakeword_model", None)
            if self._ww_model is None:
                from livekit.wakeword import WakeWordModel
                model_path = os.environ.get("WAKE_WORD_MODEL_PATH", "models/hey_nova.onnx")
                if not Path(model_path).exists():
                    logger.error(f"[WAKE WORD] Model not found at {model_path} — wake word disabled")
                    self._wake_word_active = True
                    self._set_workout_turn_detection()
                    return
                self._ww_model = WakeWordModel(models=[model_path])
                logger.info(f"[WAKE WORD] Loaded model from {model_path}")

        self._ww_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="wakeword",
        )
        self._wake_word_active = True
        self._wake_word_listening = False
        self._ww_session = self.session

        for event, handler in self._wake_word_session_handlers():
            try:
                self.session.on(event, handler)
            except Exception as e:
                logger.error(f"[WAKE WORD] FAILED to register {event}: {e}", exc_info=True)

        try:
            preemptive = self.session.options.turn_handling["preemptive_generation"]
            self._saved_preemptive_enabled = preemptive["enabled"]
            preemptive["enabled"] = False
        except Exception:
            pass

        # Attach to microphone track (or wait for subscription). Console mode
        # has no LiveKit room track to tap, so it captures the local mic.
        if _wake_word_uses_local_mic():
            self._ww_detection_task = asyncio.create_task(
                self._make_detection_loop(self._local_mic_frames())
            )
            logger.info("[WAKE WORD] Detection loop started on local microphone")
        else:
            room = self.userdata.room
            track = self._find_microphone_track(room)
            if track:
                self._start_detection_on_track(track)
            else:
                room.on("track_subscribed", self._on_track_subscribed_for_wakeword)
                logger.info("[WAKE WORD] Waiting for participant audio track...")

        self._set_workout_turn_detection()
        logger.info("[WAKE WORD] === ONNX wake word system ACTIVE ===")

    async def _stop_wake_word_system(self):
        """Deactivate wake word detection when leaving workout mode."""
        self._wake_word_active = False
        self._wake_word_listening = False
        if self.userdata.visual_bridge:
            self.userdata.visual_bridge.send_wake_event("dormant")

        if self._wake_word_timeout_task:
            self._wake_word_timeout_task.cancel()
            self._wake_word_timeout_task = None

        # A "Hey Nova" acknowledgment still pending must not speak after the
        # hand-off; the workout-complete task calling this keeps running.
        current = asyncio.current_task()
        for task in list(self._background_tasks):
            if task is not current:
                task.cancel()

        if self._ww_detection_task:
            self._ww_detection_task.cancel()
            try:
                await self._ww_detection_task
            except asyncio.CancelledError:
                pass
            self._ww_detection_task = None

        if self._ww_audio_stream:
            await self._ww_audio_stream.aclose()
            self._ww_audio_stream = None

        if self._porcupine is not None:
            try:
                self._porcupine.delete()
            except Exception:
                pass
            self._porcupine = None

        if self._ww_executor:
            self._ww_executor.shutdown(wait=False)
            self._ww_executor = None

        try:
            self.userdata.room.off("track_subscribed", self._on_track_subscribed_for_wakeword)
        except Exception:
            pass

        if self._ww_session is not None:
            for event, handler in self._wake_word_session_handlers():
                try:
                    self._ww_session.off(event, handler)
                except Exception:
                    pass

        if self._saved_preemptive_enabled is not None:
            try:
                self.session.options.turn_handling["preemptive_generation"]["enabled"] = (
                    self._saved_preemptive_enabled
                )
            except Exception:
                pass
            self._saved_preemptive_enabled = None

        self._set_conversational_turn_detection()
        logger.info("[WAKE WORD] System deactivated")

    # ===== WORKOUT FUNCTION TOOLS =====

    @function_tool
    async def end_workout(self, context: RunContext):
        """
        Call this when the user wants to end/stop their workout. It closes the workout
        out with a goodbye, so don't say goodbye before calling it.
        User might say: "stop workout", "I'm done", "end session", "finish"
        """
        logger.info("[WORKOUT] User requested to end workout")
        await self._cleanup_workout()

        # Handoff to MainMenuAgent; the closing line plays before it takes over.
        await self._suppress_turn_detection()
        from agent.agents.main_menu_agent import MainMenuAgent
        return await self._carry_context_to(MainMenuAgent(state=self.state, userdata=self.userdata)), (
            "The workout is over. Say one short goodbye for this workout: name one specific "
            "thing they did well if you know it, otherwise keep it plain. No question about "
            "what to do next. Vary the wording every time."
        )

    @function_tool
    async def end_set_early(self, reps_completed: int, context: RunContext = None):
        """
        Call this when the user wants to stop the current set before the target reps.
        User might say: "I'm done, that was 5", "stop, I got 3", "rack it".
        Do NOT call this when sets complete normally — the coaching system handles that automatically.

        Args:
            reps_completed: Number of reps the user completed before stopping
        """
        logger.info(f"[WORKOUT] User ending set early: {reps_completed} reps")

        coaching = self.userdata.coaching_service
        if not coaching:
            return None, (
                "Confirm the set is logged, in a few words. Vary the phrasing."
            )

        # Check if the orchestrator already auto-completed this set
        if coaching.is_resting:
            return None, (
                "The set was already tracked automatically. "
                "Let the user know their set is recorded and to rest up."
            )

        try:
            result = await coaching.force_end_current_set(reps=reps_completed)

            # The set's recap (rest and what's next), or after the last set the
            # exercise recap and the workout's close, follows on its own; a
            # reply here would talk over it.
            if result.get("recap_queued"):
                return None
            return None, (
                f"Confirm you logged {reps_completed} reps, in a few words. Vary the phrasing."
            )

        except Exception as e:
            logger.exception("[WORKOUT ERROR] Failed to end set early")
            return None, (
                "Confirm the set is logged and tell them to rest up. One brief sentence."
            )

    @function_tool
    async def skip_exercise(self, reason: Optional[str] = None, context: RunContext = None):
        """
        Call this when the user wants to skip the current exercise.
        User might say: "skip this", "I can't do this one", "next exercise", "equipment not available"

        Args:
            reason: Optional reason for skipping (e.g., "injury", "no equipment")
        """
        logger.info("[WORKOUT] User wants to skip exercise")
        logger.debug(f"[WORKOUT] Skip reason: {reason}")

        from agent.core.workout_session import WorkoutSession

        session_data = self.state.get("workout.current_session")
        if not session_data:
            return None, "Tell the user there's no active workout to skip and offer to start one. One helpful sentence."

        try:
            session = WorkoutSession.from_dict(session_data)

            current_exercise = session.get_current_exercise()
            if not current_exercise:
                return None, "Tell the user the workout is already finished — nothing left to skip. One positive sentence."

            exercise_name = current_exercise.exercise_name

            moved_on = session.skip_current_exercise(reason=reason)

            self.state.set("workout.current_session", session.to_dict())
            self.state.save_state()

            # The pipeline and the orchestrator follow the plan to the next exercise
            coaching = self.userdata.coaching_service
            if moved_on and coaching:
                coaching.start_current_exercise()

            next_exercise = session.get_current_exercise()
            if next_exercise:
                next_desc = session.get_current_exercise_description()
                return None, (
                    f"Confirm you're skipping {exercise_name} and introduce {next_desc}. "
                    f"Supportive and matter-of-fact, one or two sentences, vary the phrasing."
                )
            else:
                return None, (
                    f"Confirm you skipped {exercise_name} — that was the last exercise. "
                    f"Congratulate them on what they did today and ask if they're ready "
                    f"to wrap up. One or two encouraging sentences."
                )

        except Exception as e:
            logger.exception("[WORKOUT ERROR] Failed to skip exercise")
            return None, "Tell the user you're moving on to the next exercise. One brief sentence."

    @function_tool
    async def get_next_exercise(self, context: RunContext = None):
        """
        Call this when the user asks what's next or wants to preview upcoming exercises.
        User might say: "what's next", "what exercise is coming up", "show me next"
        """
        logger.info("[WORKOUT] User wants to see next exercise")

        from agent.core.workout_session import WorkoutSession

        session_data = self.state.get("workout.current_session")
        if not session_data:
            return None, "Tell the user there's no active workout and offer to start one. One helpful sentence."

        try:
            session = WorkoutSession.from_dict(session_data)

            next_exercise = session.get_next_exercise()

            if next_exercise:
                set_count = len(next_exercise.sets)
                return None, (
                    f"Preview what's next — {next_exercise.exercise_name}, {set_count} sets — "
                    f"then steer them back to finishing the current exercise. "
                    f"One motivating sentence."
                )
            else:
                current = session.get_current_exercise()
                if current:
                    return None, (
                        f"Tell them {current.exercise_name} is the last exercise and to "
                        f"finish strong. One sentence, vary the phrasing."
                    )
                else:
                    return None, "Tell them the workout is done and congratulate them. One sentence."

        except Exception as e:
            logger.exception("[WORKOUT ERROR] Failed to get next exercise")
            return None, "Redirect them to the current exercise. One brief sentence."

    @function_tool
    async def get_workout_progress(self, context: RunContext = None):
        """
        Call this when the user asks about their progress or where they are in the workout.
        User might say: "how much left", "where am I", "progress", "how many sets left"
        """
        logger.info("[WORKOUT] User wants to see workout progress")

        from agent.core.workout_session import WorkoutSession

        session_data = self.state.get("workout.current_session")
        if not session_data:
            return None, "Tell the user there's no active workout and offer to start one. One helpful sentence."

        try:
            session = WorkoutSession.from_dict(session_data)

            summary = session.get_progress_summary()

            return None, (
                f"Give them their progress: {summary['completed_sets']} of "
                f"{summary['total_sets']} sets done ({summary['percent_complete']} percent), "
                f"currently on {summary['current_exercise_name']}. Encouraging, one or two "
                f"sentences, say the numbers naturally, vary the phrasing."
            )

        except Exception as e:
            logger.exception("[WORKOUT ERROR] Failed to get progress")
            return None, "Encourage them to keep going. One brief sentence."

    @function_tool
    async def check_my_form(self, context: RunContext = None):
        """
        Call this when the user asks about their current form or positioning.
        User might say: "like this?", "is this right?", "how's my form?",
        "am I doing it right?", "is this good?", "how does this look?"
        """
        logger.info("[WORKOUT] User asking about current form")

        coaching = self.userdata.coaching_service
        if not coaching:
            return None, (
                "Tell the user you can't check their form right now. "
                "Keep it brief."
            )

        if is_coached_deadlift(self.state.get("workout.exercise_name")):
            # Judged from the last rep's features; the squat's standing checks below read
            # stance and toe-angle targets the deadlift doesn't have.
            return None, _deadlift_form_reply(getattr(coaching, "_last_rep_message", None))

        snapshot = coaching.get_current_form_snapshot()
        fresh = snapshot is not None and snapshot["data_age_ms"] <= FORM_SNAPSHOT_MAX_AGE_MS
        angles = snapshot["angles"] if fresh else {}

        # Standing still, the setup they can change right now is judged here.
        # The movement itself is only ever judged from the engine's verdict on
        # their last rep, never from one frame, so this answer agrees with the
        # cues and the recap and stays silent on what the camera can't see.
        # Whether the last cue took is in the workout state line of this turn.
        findings = _assess_standing_setup(angles) if angles.get("rep_phase") == "idle" else []
        verdict = coaching.last_rep_verdict()
        if verdict is not None:
            findings += _assess_last_rep(verdict)

        # No findings means nothing was measured yet, not that everything is
        # correct — claiming their form looks fine would be a verdict with no evidence.
        if not findings:
            return None, (
                "Tell the user you haven't seen a rep from them yet — ask them "
                "to face the camera and do one so you can watch the movement. "
                "One short sentence, no verdict on their form."
            )

        return None, (
            f"The user asked how their form looks. What you can see: "
            f"{'; '.join(findings)}. Relay this in 1-2 short sentences as a "
            f"coach — natural language, no numbers, no jargon. Lead with "
            f"whatever is already correct, then the one thing to change."
        )

    @function_tool
    async def show_me(self, what: str = "correction", context: RunContext = None):
        """
        Call this when the user wants to see a visual demonstration of a correction
        or their last rep. User might say: "show me that", "show me what you mean",
        "show me my last rep", "what should it look like?", "can I see that?"

        Args:
            what: Either "correction" to show the recommended fix, or "last_rep" to replay the last rep
        """
        logger.info(f"[WORKOUT] User requested visual demo: {what}")

        coaching = self.userdata.coaching_service
        if not coaching:
            return None, "Tell the user you can't show visuals right now. Keep it brief."

        if what == "last_rep":
            result = await coaching.request_last_rep_replay()
            if not result or result.get("error"):
                return None, (
                    "Tell them you don't have a rep saved yet — they should do "
                    "a few reps and ask again. One encouraging sentence."
                )
            from agent.services.progress_context import fault_label

            rep_data = result.get("rep_data", {})
            faults = rep_data.get("faults", [])
            fault_desc = (
                ", ".join(fault_label(f["fault_type"]) for f in faults)
                if faults
                else "clean form"
            )
            return None, (
                f"Tell the user their last rep is up on the screen and to "
                f"take a look. The rep had: {fault_desc}. Briefly describe "
                f"what you see in plain coach language, never technical "
                f"fault names. Keep it to 1-2 sentences."
            )

        cause_id = coaching.get_top_cause_id()

        result = await coaching.request_on_demand_demo(cause_id=cause_id)
        if not result or result.get("error") or result.get("status") == "unavailable":
            return None, (
                "Tell them you need to see a few more reps before you can "
                "show a demo. One encouraging sentence."
            )
        return None, (
            "Point them to the screen — you're about to show what the "
            "correction looks like. One brief sentence, then let the "
            "visual do the talking."
        )

    @function_tool
    async def flag_pain(self, body_part: str, note: str, context: RunContext = None):
        """
        Call this as soon as the user mentions pain, an injury, or that something feels off,
        before anything else. It is remembered for the rest of the workout and future sessions.

        Args:
            body_part: Where it hurts, in plain words, e.g. "left knee"
            note: What they said about it, in a few words
        """
        logger.info("[WORKOUT] Pain flagged")
        logger.debug(f"[WORKOUT] Pain flagged: {body_part} ({note})")
        add_pain_flag(self.state, body_part, note)
        await self._refresh_athlete_facts()
        return None, (
            f"Pain noted for the {body_part}. Acknowledge it plainly in one short sentence: "
            f"no hype, no diagnosis, never suggest pushing through. If they described sharp or "
            f"worsening pain, numbness, tingling, dizziness or chest pain, tell them to stop this "
            f"exercise now and get it checked by a medical professional. Otherwise offer to adjust "
            f"the movement, skip this exercise, or stop for today, and let them choose."
        )

    @function_tool
    async def clear_pain(self, body_part: str, context: RunContext = None):
        """
        Call this only when the user says a pain they reported earlier is gone or feels fine now.

        Args:
            body_part: The body part they say feels fine now
        """
        part = body_part.lower().strip()
        flagged = [
            flag["body_part"] for flag in active_pain_flags(self.state)
            if part in flag["body_part"] or flag["body_part"] in part
        ]
        for flagged_part in flagged:
            resolve_pain_flag(self.state, flagged_part)
        await self._refresh_athlete_facts()
        logger.debug(f"[WORKOUT] Pain cleared for '{body_part}': {flagged}")
        if not flagged:
            return None, "There was no open pain report for that. Acknowledge briefly, in a few words."
        return None, (
            "The pain report is cleared. Acknowledge in a few words and ask them to tell "
            "you if it comes back. Vary the phrasing."
        )

    @function_tool
    async def log_set_effort(self, effort: str, context: RunContext = None):
        """
        Call this when the user tells you how hard their last set felt.

        Args:
            effort: Their rating as they said it: a number from one to ten, or a word such as easy, medium, hard or max
        """
        rpe = _effort_to_rpe(effort)
        if rpe is None:
            return None, (
                "Ask them to rate that set from one to ten, ten meaning nothing left in the tank. "
                "One short question."
            )
        logger.info(f"[WORKOUT] Set effort -> RPE {rpe}")
        coaching = self.userdata.coaching_service
        if coaching:
            coaching.record_set_rpe(rpe)
        return None, "Acknowledge their rating in a few words. Vary the phrasing."
