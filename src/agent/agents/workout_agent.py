"""
WorkoutAgent - Handles active workout sessions with wake word system and coaching
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
from livekit import rtc
from livekit.agents import RunContext
from livekit.agents.llm import function_tool

from agent.agents.prompts import get_workout_prompt
from agent.agents.shared.base_agent import BaseNovaAgent
from db.database import SessionLocal

logger = logging.getLogger(__name__)

# Trunk flexion uses the 180-convention: 180 = upright, lower = more lean.
UPRIGHT_TRUNK_DEG = 180.0
# ForwardLeanRule's mild threshold, expressed as lean from vertical.
NOTABLE_LEAN_DEG = 35.0

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


def _assess_in_rep(angles: dict) -> list[str]:
    """Judge what is visible mid-rep, in plain lean-from-vertical terms."""
    findings: list[str] = []

    trunk = angles.get("trunk_flexion")
    if trunk is not None:
        lean = UPRIGHT_TRUNK_DEG - trunk
        if lean > NOTABLE_LEAN_DEG:
            findings.append("their chest is dropping forward more than it should")
        else:
            findings.append("their torso angle looks good")

    knee_l = angles.get("knee_flexion_l")
    knee_r = angles.get("knee_flexion_r")
    if knee_l is not None and knee_r is not None:
        if abs(knee_l - knee_r) > 10.0:
            findings.append("one side is bending more than the other")

    return findings


class WorkoutAgent(BaseNovaAgent):
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

        super().__init__(state=state, userdata=userdata, instructions=get_workout_prompt())

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
                    "Greet the user into the workout with real energy, name "
                    "the first exercise, and get them moving. Two sentences "
                    "max. Vary your opener between sessions.",
                    restore=False,
                )
            else:
                workout_name = session_data.get("workout_name", "today's workout")
                await self._say(
                    f"[CONTEXT] workout just started: {workout_name}, "
                    f"first exercise: {first_desc}{progress_block}\n\n"
                    "Greet the user into the workout with real energy, name "
                    "the first exercise, and mention you're watching their "
                    "form. Two sentences max. Vary your opener between "
                    "sessions.",
                    restore=False,
                )
        else:
            await self._say(
                f"[CONTEXT] workout mode just started, no session details "
                f"available{progress_block}\n\n"
                "Greet the user into the workout with real energy and get "
                "them moving. Two sentences max. Vary your opener between "
                "sessions.",
                restore=False,
            )

        self.state.set("workout.greeting_done", True)
        self.state.save_state()
        logger.info("[WORKOUT] Greeting done — signalled main.py to start pose estimation")

        await self._start_wake_word_system()

    # ===== COACHING SERVICE CALLBACKS =====

    async def _on_workout_complete_signal(self, data: dict):
        """Called by CoachingService when the entire workout is done."""
        logger.info("[COACHING] Workout complete signal received — scheduling cleanup")
        asyncio.create_task(self._handle_workout_complete())

    async def _handle_workout_complete(self):
        """Run cleanup and agent handoff outside the orchestrator's task."""
        await self._cleanup_workout()
        # Brief pause to let the LLM finish processing pending conversation
        # events from the exercise recap before truncation.
        await asyncio.sleep(0.5)
        await self._truncate_context_for_handoff()
        from agent.agents.main_menu_agent import MainMenuAgent
        new_agent = MainMenuAgent(state=self.state, userdata=self.userdata)
        self.session.update_agent(new_agent)

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
        """Cancel auto-generated responses in workout mode.
        Allows coaching LLM speech and programmatic calls through.
        """
        logger.info(
            f"[WAKE WORD] speech_created event: user_initiated={ev.user_initiated} "
            f"source={getattr(ev, 'source', 'unknown')} "
            f"active={self._wake_word_active} listening={self._wake_word_listening}"
        )

        if not self._wake_word_active or self._wake_word_listening:
            logger.info("[WAKE WORD] Allowing speech (wake word inactive or in listening mode)")
            return

        # Never cancel speech from the coaching service
        coaching = self.userdata.coaching_service
        if coaching and coaching.is_coaching_speaking:
            logger.info("[WAKE WORD] Allowing speech (coaching LLM in progress)")
            return

        # Dormant mode: cancel everything else. Turn replies that commit late
        # (endpointing delay) arrive with user_initiated=True, so that flag
        # can't distinguish them from programmatic calls.
        try:
            ev.speech_handle.cancel()
            logger.info("[WAKE WORD] ✗ CANCELLED speech (dormant mode)")
        except Exception as e:
            logger.error(f"[WAKE WORD] Failed to cancel speech: {e}", exc_info=True)

    async def _activate_listening_mode(self):
        """Activate listening mode after wake word detection."""
        coaching = self.userdata.coaching_service
        if coaching and coaching.is_coaching_speaking:
            logger.info("[WAKE WORD] Coaching LLM in progress — deferring wake word response")
            return

        logger.info("[WAKE WORD] Activating listening mode")
        self._wake_word_listening = True
        if self.userdata.visual_bridge:
            self.userdata.visual_bridge.send_wake_event("detected")
        self._set_active_listening_turn_detection()

        await self.session.generate_reply(
            instructions="The user just said 'Hey Nova' during a workout. Respond very briefly (2-5 words max) like 'Yeah?', 'What's up?', or 'I'm here!' — then wait for their question."
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

                if self._wake_word_listening or self._ww_session.agent_state == "speaking":
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
                            coaching = self.userdata.coaching_service
                            if coaching and coaching.is_coaching_speaking:
                                logger.info("[WAKE WORD] Coaching in progress — ignoring detection")
                                break
                            asyncio.create_task(self._activate_listening_mode())
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

                if self._wake_word_listening or self._ww_session.agent_state == "speaking":
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
                    coaching = self.userdata.coaching_service
                    if coaching and coaching.is_coaching_speaking:
                        logger.info("[WAKE WORD] Coaching in progress — ignoring detection")
                        break
                    asyncio.create_task(self._activate_listening_mode())
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

        try:
            self.session.on("agent_state_changed", self._on_agent_state_changed_for_wake_word)
        except Exception as e:
            logger.error(f"[WAKE WORD] FAILED to register agent_state_changed: {e}", exc_info=True)

        try:
            self.session.on("speech_created", self._on_speech_created_for_wake_word)
        except Exception as e:
            logger.error(f"[WAKE WORD] FAILED to register speech_created: {e}", exc_info=True)

        try:
            self.session.on("user_state_changed", self._on_user_state_changed_for_wake_word)
        except Exception as e:
            logger.error(f"[WAKE WORD] FAILED to register user_state_changed: {e}", exc_info=True)

        try:
            self.session.on("user_input_transcribed", self._on_user_transcript_for_wake_word)
        except Exception as e:
            logger.error(f"[WAKE WORD] FAILED to register user_input_transcribed: {e}", exc_info=True)

        try:
            self.session.options.turn_handling["preemptive_generation"]["enabled"] = False
        except Exception:
            pass

        try:
            self.session.options.turn_handling["endpointing"]["min_delay"] = 0.2
        except Exception:
            pass

        # Attach to microphone track (or wait for subscription).
        # WAKE_WORD_LOCAL_MIC=1 captures the local mic instead — console mode
        # has no LiveKit room track to tap.
        if os.environ.get("WAKE_WORD_LOCAL_MIC") == "1":
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

        try:
            self.session.off("agent_state_changed", self._on_agent_state_changed_for_wake_word)
            self.session.off("speech_created", self._on_speech_created_for_wake_word)
        except Exception:
            pass

        try:
            self.session.options.turn_handling["preemptive_generation"]["enabled"] = True
        except Exception:
            pass

        try:
            self.session.options.turn_handling["endpointing"]["min_delay"] = 0.3
        except Exception:
            pass

        self._set_conversational_turn_detection()
        logger.info("[WAKE WORD] System deactivated")

    # ===== WORKOUT FUNCTION TOOLS =====

    @function_tool
    async def end_workout(self, context: RunContext):
        """
        Call this when the user wants to end/stop their workout.
        User might say: "stop workout", "I'm done", "end session", "finish"
        """
        logger.info("[WORKOUT] User requested to end workout")
        await self._cleanup_workout()

        # Handoff to MainMenuAgent
        await self._suppress_turn_detection()
        await self._truncate_context_for_handoff()
        from agent.agents.main_menu_agent import MainMenuAgent
        return MainMenuAgent(state=self.state, userdata=self.userdata)

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

            if result["status"] == "advanced":
                rest_sec = result.get("rest_seconds", 60)
                next_desc = result.get("next_set_description", "the next set")
                rest_display = f"{rest_sec // 60}:{rest_sec % 60:02d}" if rest_sec >= 60 else f"{rest_sec} seconds"
                return None, (
                    f"Confirm you logged {reps_completed} reps, tell them to rest "
                    f"for {rest_display}, and name what's next: {next_desc}. "
                    f"One or two brief sentences, vary the phrasing."
                )
            elif result["status"] == "workout_complete":
                return None, (
                    f"Celebrate the finish in your own words — "
                    f"{reps_completed} reps on the final set — one or two "
                    f"sentences, then call end_workout."
                )
            else:
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
        logger.info(f"[WORKOUT] User wants to skip exercise. Reason: {reason}")

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

            session.skip_current_exercise(reason=reason)

            self.state.set("workout.current_session", session.to_dict())
            self.state.save_state()

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

        snapshot = coaching.get_current_form_snapshot()
        if not snapshot:
            return None, (
                "Ask them to hold their position for a second so you can "
                "get a read on them. One natural sentence."
            )

        if snapshot["data_age_ms"] > 3000:
            return None, (
                "Ask them to hold the position so you can get a fresh look. "
                "One brief, encouraging sentence."
            )

        angles = snapshot["angles"]
        last_cue = snapshot.get("last_cue") or {}
        standing = angles.get("rep_phase") == "idle"

        # Judge here rather than handing the LLM raw angles: the codebase's
        # trunk convention is inverted (180 = upright) and valgus has no
        # fixed scale, so a model reading bare numbers guesses backwards.
        findings = _assess_standing_setup(angles) if standing else _assess_in_rep(angles)

        # No findings means the metrics were absent, not that everything is
        # correct — an uncalibrated athlete gets no targets at all, and
        # claiming their setup looks fine would be a verdict with no evidence.
        if not findings:
            return None, (
                "Tell the user you can't get a clean read from where they "
                "are — ask them to face the camera and run a rep so you can "
                "watch the movement. One short sentence, no verdict on their "
                "form."
            )

        cue_context = ""
        if last_cue.get("cue_key"):
            cue_context = (
                f" They are asking because your last cue was "
                f"'{last_cue['cue_key']}'."
            )

        return None, (
            f"The user asked how their form looks.{cue_context} "
            f"Here is what you can see right now: {'; '.join(findings)}. "
            f"Relay this in 1-2 short sentences as a coach — natural "
            f"language, no numbers, no jargon. Lead with whatever is "
            f"already correct, then the one thing to change."
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
