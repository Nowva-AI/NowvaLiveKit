"""
Standalone Coaching Service

Handles all real-time biomechanics coaching during workout sets.
Owns: IPC listener, CoachingOrchestrator, AudioCueService.
The voice agent is passive during sets — this service fires
generate_reply() calls directly for all coaching speech.
"""

import asyncio
import logging
import os
import threading
from collections import deque
from pathlib import Path
from typing import Optional, Callable, Dict, Any

from agent.services.assessment_logger import AssessmentLogger
from agent.services.coaching_constants import (
    ADJUSTMENT_CUES,
    ADJUSTMENT_ON_TARGET_CUE,
    ADJUSTMENT_PARAM_LABELS,
    ADJUSTMENT_SYSTEM_PROMPT,
    COACHING_PERSONA,
    CUE_TEXT_MAP,
)
from agent.services.visual_bridge import VisualBridge

logger = logging.getLogger(__name__)

COACHING_SOCKET_PATH = "/tmp/nowva_coaching.sock"
LISTENER_RECONNECT_POLL_S = 1.0

# Cue keys whose on-screen banner should read as praise, not correction
POSITIVE_CUE_KEYS = frozenset(
    {"good_rep", "great_depth", "strong", "clean", "perfect", "adjust_good"}
)


class CoachingService:
    """
    Standalone coaching service that runs in the same process as the voice agent.
    Holds references to AgentSession and AgentState, handles all IPC,
    orchestration, audio playback, and LLM calls independently.
    """

    def __init__(
        self,
        session,          # AgentSession — for generate_reply, output.audio
        state,            # AgentState — for reading/writing WorkoutSession
        room=None,        # rtc.Room — for publishing separate audio tracks
        on_workout_complete: Optional[Callable] = None,
        on_calibration_complete: Optional[Callable] = None,
        on_assessment_result: Callable | None = None,
        audio_cue_service=None,  # Prewarmed AudioCueService (optional)
    ):
        self._session = session
        self._state = state
        self._room = room
        self._on_workout_complete_callback = on_workout_complete
        self._on_calibration_complete_callback = on_calibration_complete
        self._on_assessment_result_callback = on_assessment_result
        self._on_assessment_ready_callback: Callable | None = None

        # Owned components
        self._coaching_ipc = None
        self._coaching_orchestrator = None
        self._audio_cue_service = audio_cue_service
        self._visual_bridge: VisualBridge | None = None

        # Internal state
        self._listener_running = False
        self._listener_thread: threading.Thread | None = None
        self._listener_stop = threading.Event()
        self._event_loop = None
        self._started = False

        # Flag for wake word system to detect active coaching speech
        self.is_coaching_speaking: bool = False

        # Last few spoken coaching lines — injected into every coaching prompt
        # so the LLM never reuses phrasing (ephemeral context stripping means
        # it otherwise has no memory of what it said last set).
        self._spoken_coaching_lines: deque[str] = deque(maxlen=4)

        # Choreographed assessment demo fires once per session
        self._assessment_demo_played: bool = False
        self._demo_start_ack = None

        # Orchestrator is dormant until assessment+calibration finish
        self._workout_active_flag: bool = False

        # Assessment data persistence
        self._assessment_logger: AssessmentLogger | None = None

        # Workout data persistence (sessions/sets/reps/cue outcomes → PostgreSQL)
        self._biomech_recorder = None

        # Last-session baseline and multi-session trends (fetched async at workout start)
        self._progress_baseline: dict | None = None
        self._fault_trends: dict | None = None
        self._baseline_task: Optional[asyncio.Task] = None

        # Pending on-demand demo response (request_last_rep / request_demo)
        self._pending_demo_response: dict | None = None

    @property
    def _workout_active(self) -> bool:
        return self._workout_active_flag

    @_workout_active.setter
    def _workout_active(self, value: bool) -> None:
        # Starts DB recording on the False→True transition, covering both
        # activation paths: calibration completion and the returning-user
        # path where workout_agent sets this attribute directly.
        if value and not self._workout_active_flag:
            self._start_biomech_recording()
        self._workout_active_flag = value

    def _start_biomech_recording(self) -> None:
        if self._biomech_recorder is not None:
            return
        user_id = self._state.get("user.id")
        if not user_id:
            logger.warning("[BIOMECH DB] No user_id in state — workout persistence disabled")
            return
        from db.biomechanics_persistence import BiomechanicsRecorder
        try:
            self._biomech_recorder = BiomechanicsRecorder(
                user_id=user_id,
                calibration_snapshot=self._state.get("workout.calibration_profile"),
            )
            self._biomech_recorder.start()
            if self._coaching_orchestrator:
                self._coaching_orchestrator.on_fault_cue_delivered = (
                    self._biomech_recorder.record_cue_delivered
                )
            logger.info(
                f"[BIOMECH DB] Recording workout session {self._biomech_recorder.session_id} "
                f"for user={user_id}"
            )
        except Exception as e:
            self._biomech_recorder = None
            logger.error(f"[BIOMECH DB] Failed to start recorder: {e}", exc_info=True)
        self._baseline_task = asyncio.create_task(
            self._fetch_progress_baseline(user_id)
        )

    async def _fetch_progress_baseline(self, user_id) -> dict | None:
        """Load last-session baseline and multi-session fault trends."""
        def _fetch():
            from db.database import SessionLocal
            from db.biomechanics_persistence import (
                get_progress_baseline,
                get_multi_session_fault_trends,
            )
            db = SessionLocal()
            try:
                baseline = get_progress_baseline(db, user_id)
                fault_trends = get_multi_session_fault_trends(db, user_id)
                return baseline, fault_trends
            finally:
                db.close()

        try:
            baseline, fault_trends = await asyncio.to_thread(_fetch)
        except Exception:
            logger.exception("[BIOMECH DB] Progress baseline fetch failed")
            return None
        self._progress_baseline = baseline
        self._fault_trends = fault_trends
        if baseline:
            if self._coaching_orchestrator:
                self._coaching_orchestrator.progress_baseline = baseline
            logger.info(
                f"[BIOMECH DB] Progress baseline loaded: last session "
                f"{baseline.get('days_ago')}d ago, score={baseline.get('mean_score')}"
            )
        else:
            logger.info("[BIOMECH DB] No previous session — first-workout baseline")
        if fault_trends:
            if self._coaching_orchestrator:
                self._coaching_orchestrator.fault_trends = fault_trends
            chronic = fault_trends.get("chronic_faults", [])
            logger.info(
                f"[BIOMECH DB] Fault trends loaded: {fault_trends.get('sessions_analyzed', 0)} sessions, "
                f"{len(fault_trends.get('fault_profile', []))} fault types, "
                f"chronic={chronic}"
            )
        return baseline

    async def wait_progress_baseline(self, timeout_s: float = 2.5) -> dict | None:
        """Baseline dict once the fetch finishes, or None on timeout/first workout."""
        if self._baseline_task is None:
            return self._progress_baseline
        try:
            await asyncio.wait_for(
                asyncio.shield(self._baseline_task), timeout=timeout_s
            )
        except asyncio.TimeoutError:
            pass
        return self._progress_baseline

    async def wait_progress_context(
        self, timeout_s: float = 2.5
    ) -> tuple[dict | None, dict | None]:
        """Return (baseline, fault_trends) once the background fetch finishes."""
        if self._baseline_task is None:
            return self._progress_baseline, self._fault_trends
        try:
            await asyncio.wait_for(
                asyncio.shield(self._baseline_task), timeout=timeout_s
            )
        except asyncio.TimeoutError:
            pass
        return self._progress_baseline, self._fault_trends

    async def _close_biomech_recording(self) -> None:
        if self._biomech_recorder is None:
            return
        recorder = self._biomech_recorder
        self._biomech_recorder = None
        if self._coaching_orchestrator:
            self._coaching_orchestrator.on_fault_cue_delivered = None
        flushed = await asyncio.to_thread(recorder.close)
        if flushed:
            logger.info("[BIOMECH DB] Workout session persisted and finalized")
        else:
            logger.warning(
                "[BIOMECH DB] Workout session close timed out — data may be incomplete"
            )

    def set_workout_complete_callback(self, callback: Callable) -> None:
        self._on_workout_complete_callback = callback

    def set_calibration_complete_callback(self, callback: Callable) -> None:
        self._on_calibration_complete_callback = callback

    def set_assessment_result_callback(self, callback: Callable | None) -> None:
        self._on_assessment_result_callback = callback

    def set_assessment_ready_callback(self, callback: Callable | None) -> None:
        self._on_assessment_ready_callback = callback

    # ------------------------------------------------------------------
    # Lifecycle API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start all coaching subsystems."""
        if self._started:
            return

        self._event_loop = asyncio.get_running_loop()
        self._init_orchestrator()
        asyncio.create_task(self._start_ipc_listener())

        # Best-effort link to the display page for on-screen cue banners
        self._visual_bridge = VisualBridge()
        self._visual_bridge.start()

        self._started = True
        logger.info("[COACHING SERVICE] Started")

    async def stop(self) -> None:
        """Stop all coaching subsystems and clean up."""
        if not self._started:
            return

        # Mark an interrupted assessment as complete so the on-disk log
        # has a completed_at timestamp even on abnormal shutdown.
        if self._assessment_logger is not None:
            self._assessment_logger.finalize(passed=False)
            self._assessment_logger = None
            logger.info("[COACHING SERVICE] Active assessment finalized on stop")

        # Flush any unfinalized workout data (abnormal shutdown backstop)
        await self._close_biomech_recording()
        if self._baseline_task is not None and not self._baseline_task.done():
            self._baseline_task.cancel()
        self._baseline_task = None

        if self._coaching_orchestrator:
            self._coaching_orchestrator.stop()
            self._coaching_orchestrator = None

        self._stop_ipc_listener()
        self._audio_cue_service = None
        if self._visual_bridge is not None:
            await self._visual_bridge.aclose()
            self._visual_bridge = None
        self._started = False
        logger.info("[COACHING SERVICE] Stopped")

    @property
    def is_resting(self) -> bool:
        """Whether the service is in rest-between-sets mode."""
        if self._coaching_orchestrator:
            return self._coaching_orchestrator.resting
        return False

    def get_last_cue_cause_id(self) -> str | None:
        orch = self._coaching_orchestrator
        if orch and orch.last_cue_context:
            return orch.last_cue_context.get("fault_type")
        return None

    # ------------------------------------------------------------------
    # Real-time form queries
    # ------------------------------------------------------------------

    def get_current_form_snapshot(self) -> dict | None:
        """Return a snapshot of the user's current biomechanics state.

        Used by the check_my_form tool to answer "like this?" queries.
        Returns None if no orchestrator is active or no data is available.
        """
        orch = self._coaching_orchestrator
        if not orch or not orch.latest_angles:
            return None

        import time as _time
        data_age_ms = (_time.time() - orch.latest_angles_time) * 1000

        snapshot: dict = {
            "angles": orch.latest_angles,
            "data_age_ms": round(data_age_ms, 0),
            "last_cue": orch.last_cue_context,
        }

        if orch._pending_diagnosis:
            snapshot["diagnosis"] = orch._pending_diagnosis

        return snapshot

    # ------------------------------------------------------------------
    # On-demand demo / last-rep replay
    # ------------------------------------------------------------------

    async def _request_from_pipeline(
        self, message: dict, timeout: float,
    ) -> dict | None:
        """Send a request and await its matching reply.

        Only one demo request is in flight at a time — a second concurrent
        "show me" would otherwise overwrite the first one's slot and leave
        it waiting forever.
        """
        import uuid

        if self._pending_demo_response is not None:
            logger.info("[COACHING SERVICE] Demo request already in flight — ignoring")
            return None

        request_id = str(uuid.uuid4())
        pending = {"event": asyncio.Event(), "request_id": request_id, "data": None}
        self._pending_demo_response = pending
        self._send_to_pipeline({**message, "request_id": request_id})
        try:
            await asyncio.wait_for(pending["event"].wait(), timeout=timeout)
            return pending["data"]
        except asyncio.TimeoutError:
            logger.warning(
                f"[COACHING SERVICE] {message.get('type')} timed out after {timeout}s"
            )
            return None
        finally:
            if self._pending_demo_response is pending:
                self._pending_demo_response = None

    async def request_last_rep_replay(self) -> dict | None:
        return await self._request_from_pipeline({"type": "request_last_rep"}, timeout=5.0)

    async def request_on_demand_demo(self, cause_id: str | None = None) -> dict | None:
        data = await self._request_from_pipeline(
            {"type": "request_demo", "cause_id": cause_id}, timeout=5.0,
        )
        if data and data.get("status") == "available" and data.get("cues"):
            asyncio.create_task(self._run_on_demand_demo(data["cues"]))
        return data

    async def _run_on_demand_demo(self, cues: list) -> None:
        """Launch the choreographed demo task for an on-demand "show me" request."""
        try:
            await self._run_assessment_demo(cues)
        except Exception:
            logger.exception("[COACHING SERVICE] On-demand demo task failed")

    # ------------------------------------------------------------------
    # IPC Listener
    # ------------------------------------------------------------------

    async def _start_ipc_listener(self):
        """Connect to the coaching IPC server and listen for biomechanics messages.

        The listener thread is resilient: main.py tears the coaching socket
        down when a workout ends and rebinds it when the next one starts,
        so on any disconnect the thread waits for the socket file to
        reappear and reconnects. A second pass through the workout flow in
        the same session therefore gets coaching data again.
        """
        if self._listener_thread is not None and self._listener_thread.is_alive():
            logger.info("[COACHING SERVICE] Listener already running")
            return

        from agent.core.ipc_communication import IPCClient

        self._listener_stop.clear()

        def on_message(message: dict):
            if self._event_loop and self._event_loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self._handle_message(message),
                    self._event_loop,
                )

        def _listen_thread():
            while not self._listener_stop.is_set():
                if not os.path.exists(COACHING_SOCKET_PATH):
                    # Server not up (e.g. between workout passes) — poll quietly
                    self._listener_stop.wait(LISTENER_RECONNECT_POLL_S)
                    continue
                if self._coaching_ipc is not None:
                    self._coaching_ipc.disconnect()
                self._coaching_ipc = IPCClient(socket_path=COACHING_SOCKET_PATH)
                if not self._coaching_ipc.connect(timeout=5):
                    self._listener_stop.wait(LISTENER_RECONNECT_POLL_S)
                    continue
                logger.info("[COACHING SERVICE] Connected to coaching IPC server")
                self._listener_running = True
                try:
                    self._coaching_ipc.listen(message_callback=on_message)
                except Exception as e:
                    logger.error(f"[COACHING SERVICE] Listener error: {e}")
                finally:
                    self._listener_running = False
                if not self._listener_stop.is_set():
                    logger.info(
                        "[COACHING SERVICE] Coaching IPC disconnected — waiting for server"
                    )

        self._listener_thread = threading.Thread(target=_listen_thread, daemon=True)
        self._listener_thread.start()
        logger.info("[COACHING SERVICE] Listener thread started")

    def _stop_ipc_listener(self):
        """Disconnect from the coaching IPC server and stop reconnecting."""
        self._listener_stop.set()
        if self._coaching_ipc:
            self._coaching_ipc.disconnect()
            self._coaching_ipc = None
        self._listener_running = False
        self._listener_thread = None
        logger.info("[COACHING SERVICE] Listener stopped")

    # ------------------------------------------------------------------
    # Message Dispatch
    # ------------------------------------------------------------------

    async def _handle_message(self, message: dict):
        """Dispatch incoming coaching message through the orchestrator."""
        msg_type = message.get("type")
        logger.info(f"[COACHING SERVICE] ← IPC message received: type={msg_type} | keys={list(message.keys())}")

        from profiler.collector import SessionProfiler
        _profiler = SessionProfiler.get_instance()
        _profiler.record("coaching", msg_type or "unknown", **{
            k: v for k, v in message.items()
            if k != "type" and not isinstance(v, (bytes, bytearray)) and k != "joint_angles"
        })

        try:
            if msg_type == "cache_cues":
                await self._on_cache_cues(message)
            elif msg_type == "fault":
                cue_key = message.get("cue")
                fault_type = message.get("fault_type", "")
                severity = message.get("severity", "")
                fault_msg = message.get("message", "")
                logger.info(f"[COACHING SERVICE] FAULT received: type={fault_type} severity={severity} cue={cue_key} msg='{fault_msg}'")
                if self._workout_active and self._biomech_recorder:
                    self._biomech_recorder.record_fault(message)
                if not self._workout_active:
                    logger.debug("[COACHING SERVICE] Fault ignored — workout not active yet")
                elif self._coaching_orchestrator:
                    await self._coaching_orchestrator.on_fault(
                        cue_key=cue_key,
                        fault_type=fault_type,
                        severity=severity,
                        message=fault_msg,
                    )
                else:
                    logger.warning("[COACHING SERVICE] No orchestrator — fault dropped")
            elif msg_type == "rep_complete":
                rep = message.get("rep_number")
                depth = message.get("depth_category", "")
                is_clean = message.get("is_clean", False)
                faults = message.get("faults_in_rep", [])
                max_depth_angle = message.get("max_depth_angle", 0.0)
                rep_duration_ms = message.get("rep_duration_ms", 0)
                ascent_time_s = float(message.get("ascent_time_s", 0.0) or 0.0)
                logger.info(
                    f"[COACHING SERVICE] REP COMPLETE received: rep={rep} depth={depth} "
                    f"is_clean={is_clean} faults={faults}"
                )
                if self._assessment_logger is not None:
                    self._assessment_logger.on_rep_complete(message)
                if self._workout_active and self._biomech_recorder:
                    self._biomech_recorder.record_rep(message)
                if not self._workout_active:
                    logger.debug("[COACHING SERVICE] rep_complete ignored — workout not active yet")
                elif self._coaching_orchestrator:
                    await self._coaching_orchestrator.on_rep_complete(
                        rep_number=rep or 0,
                        depth=depth,
                        is_clean=is_clean,
                        faults=faults,
                        max_depth_angle=max_depth_angle,
                        rep_duration_ms=rep_duration_ms,
                        ascent_time_s=ascent_time_s,
                    )
                else:
                    logger.warning("[COACHING SERVICE] No orchestrator — rep_complete dropped")
            elif msg_type == "shallow_rep":
                depth_name = message.get("depth_class_name", "")
                logger.info(
                    f"[COACHING SERVICE] SHALLOW REP received: depth={depth_name} "
                    f"knee={message.get('max_knee_flexion')}°"
                )
                if self._workout_active and self._biomech_recorder and message.get("fault_type"):
                    self._biomech_recorder.record_fault(message)
                if not self._workout_active:
                    logger.debug("[COACHING SERVICE] shallow_rep ignored — workout not active yet")
                elif self._coaching_orchestrator:
                    await self._coaching_orchestrator.on_shallow_rep(
                        cue_key=message.get("cue"),
                        depth_class_name=depth_name,
                    )
                else:
                    logger.warning("[COACHING SERVICE] No orchestrator — shallow_rep dropped")
            elif msg_type == "rep_diagnosis":
                if self._assessment_logger is not None:
                    self._assessment_logger.on_rep_diagnosis(
                        message.get("rep_number", 0),
                        message.get("diagnosis", {}),
                        rep_score=message.get("rep_score"),
                    )
                if self._workout_active and self._biomech_recorder:
                    self._biomech_recorder.record_rep_diagnosis(message)
            elif msg_type == "frame_data":
                if self._workout_active and self._coaching_orchestrator:
                    angles_data = message.get("joint_angles", {})
                    for extra_key in (
                        "rep_phase",
                        "stance_width_ratio",
                        "foot_direction_angle_l",
                        "foot_direction_angle_r",
                        "target_stance_ratio",
                        "target_toe_out_deg",
                    ):
                        if extra_key in message:
                            angles_data[extra_key] = message[extra_key]
                    self._coaching_orchestrator.record_angle_sample(angles_data)
            elif msg_type == "diagnosis_complete":
                diagnosis = message.get("diagnosis", {})
                scoring = message.get("scoring", {})
                logger.info(
                    f"[COACHING SERVICE] DIAGNOSIS COMPLETE: "
                    f"confidence={diagnosis.get('confidence', 0):.2f} "
                    f"score={scoring.get('mean_score', 0):.3f}"
                )
                if self._workout_active and self._biomech_recorder:
                    self._biomech_recorder.record_set(message)
                if self._coaching_orchestrator:
                    self._coaching_orchestrator.set_diagnosis_data(diagnosis, scoring)
                else:
                    logger.warning("[COACHING SERVICE] No orchestrator — diagnosis_complete dropped")
            elif msg_type == "set_complete":
                logger.info("[COACHING SERVICE] set_complete from pipeline (ignored — orchestrator handles via rep count)")
            elif msg_type == "rest_complete":
                logger.info("[COACHING SERVICE] REST COMPLETE — firing LLM prompt for next set")

                # Get set numbers from workout session
                completed_set, next_set, total_sets = self._get_set_numbers()

                instructions = (
                    f"[REST COMPLETE] Rest is over. "
                    f"Set {completed_set} of {total_sets} is done. Set {next_set} of {total_sets} starts now. "
                    f"Announce it in one energetic sentence, in your own words — "
                    f"never the same announcement twice in a session. "
                    f"Do NOT ask if they are ready. Do NOT wait for confirmation."
                )
                logger.info("[COACHING SERVICE] → Calling coaching LLM for rest_complete")
                await self._coaching_llm_reply(instructions)
                logger.info("[COACHING SERVICE] ✓ Coaching LLM returned for rest_complete")

                # Resume rep/fault processing AFTER the announcement finishes.
                # Keeping _resting=True during speech prevents the orchestrator
                # from dispatching cached cues that collide with the announcement.
                if self._coaching_orchestrator:
                    self._coaching_orchestrator.on_rest_complete()
            elif msg_type == "assessment_rep":
                rep = message.get("rep_number", 0)
                total = message.get("total_required", 2)
                round_num = message.get("round", 1)
                logger.info(f"[COACHING SERVICE] ASSESSMENT REP {rep}/{total} (round {round_num})")
                cue_key = f"rep_{rep}"
                await self._play_cached_cue_audio(cue_key)
            elif msg_type == "assessment_ready":
                logger.info(
                    "[COACHING SERVICE] ASSESSMENT READY — tracking calibrated, cueing first squat"
                )
                if self._on_assessment_ready_callback is not None:
                    try:
                        ret = self._on_assessment_ready_callback()
                        if asyncio.iscoroutine(ret):
                            await ret
                    except Exception as e:
                        logger.error(
                            f"[ASSESSMENT] on_assessment_ready callback error: {e}",
                            exc_info=True,
                        )
            elif msg_type == "assessment_result":
                passed = message.get("passed", False)
                round_num = message.get("round", 1)
                logger.info(
                    f"[COACHING SERVICE] ASSESSMENT RESULT: passed={passed} round={round_num}"
                )
                await self._on_assessment_result(message)
            elif msg_type == "demo_abort":
                # Pipeline has no demo data — the demo task keeps speaking audio-only.
                logger.warning(
                    f"[COACHING SERVICE] Demo visuals aborted: {message.get('reason')}"
                )
            elif msg_type == "demo_started":
                if self._demo_start_ack is not None:
                    self._demo_start_ack.set()
            elif msg_type == "calibration_rep":
                rep = message.get("rep_number", 0)
                total = message.get("total_required", 5)
                depth = message.get("depth_angle", 0)
                logger.info(f"[COACHING SERVICE] CALIBRATION REP {rep}/{total} depth={depth}°")
                cue_key = f"rep_{rep}"
                await self._play_cached_cue_audio(cue_key)
            elif msg_type == "calibration_complete":
                logger.info("[COACHING SERVICE] CALIBRATION COMPLETE — saving to DB")
                await self._on_calibration_complete(message)
            elif msg_type in ("last_rep_snapshot", "demo_data_ready"):
                pending = self._pending_demo_response
                if pending and message.get("request_id") == pending["request_id"]:
                    pending["data"] = message
                    pending["event"].set()
            elif msg_type == "play_cue":
                logger.debug("[COACHING SERVICE] play_cue ignored (orchestrator handles dispatch)")
            else:
                logger.warning(f"[COACHING SERVICE] Unknown message type: {msg_type}")
        except Exception as e:
            logger.error(f"[COACHING SERVICE] Error handling {msg_type}: {e}", exc_info=True)

    async def ensure_rep_track(self) -> bool:
        """Publish the dedicated rep-sound track if it isn't up yet.

        The rep beep must land on every counted rep regardless of what the
        coach is saying, which means its own LiveKit track — on the shared
        speech track it queues behind cues and recaps. Safe to call repeatedly.
        """
        service = self._audio_cue_service
        if service is None or not self._room:
            return False
        if service.rep_track_ready:
            return True
        await service.setup_rep_track(self._room)
        if not service.rep_track_ready:
            logger.warning(
                "[COACHING SERVICE] Rep sound track unavailable — rep beeps will "
                "share the speech track and may be delayed behind coaching audio"
            )
        return service.rep_track_ready

    async def _on_cache_cues(self, message: dict):
        """Pre-generate TTS audio for all cues in the message."""
        from agent.services.audio_cue_service import AudioCueService

        if self._audio_cue_service is None:
            self._audio_cue_service = AudioCueService(session=self._session)
        elif self._audio_cue_service.session is None:
            # Prewarmed service — attach the live session on first use
            self._audio_cue_service.attach_session(self._session)
            logger.info("[COACHING SERVICE] Attached session to prewarmed AudioCueService")
        await self.ensure_rep_track()

        cues = message.get("cues", {})
        exercise = message.get("exercise_name", "unknown")
        logger.info(f"[COACHING SERVICE] Pre-caching {len(cues)} cues for {exercise}")

        try:
            await self._audio_cue_service.cache_cues(cues)
            if self._coaching_orchestrator:
                from biomechanics.coaching.cue_cache import POSITIVE_CUE_KEYS
                available = [k for k in cues if k in POSITIVE_CUE_KEYS]
                self._coaching_orchestrator.positive_cue_keys = available
        except Exception as e:
            logger.error(f"[COACHING SERVICE] TTS cache generation failed: {e}")

    async def _on_assessment_result(self, message: dict) -> None:
        """Handle assessment result — choreographed demo on first failure, else spoken feedback."""
        passed = message.get("passed", False)
        round_num = message.get("round", 1)
        diagnosis = message.get("diagnosis", {})
        scoring = message.get("scoring", {})
        demo = message.get("demo", {})

        demo_was_played = False
        if not passed and demo.get("available") and not self._assessment_demo_played:
            self._assessment_demo_played = True
            if await self._run_assessment_demo(demo.get("cues", [])):
                demo_was_played = True

        if self._on_assessment_result_callback is not None:
            try:
                ret = self._on_assessment_result_callback(message, demo_was_played)
                if asyncio.iscoroutine(ret):
                    await ret
            except Exception as e:
                logger.error(f"[ASSESSMENT] on_assessment_result callback error: {e}", exc_info=True)
            return

        if not passed and not demo_was_played:
            instructions = self._build_assessment_correction_prompt(
                diagnosis, scoring, round_num,
            )
        elif passed:
            instructions = self._build_assessment_pass_prompt(
                diagnosis, scoring, round_num,
            )
        else:
            return

        logger.info(f"[ASSESSMENT] LLM prompt: {instructions[:150]}...")
        await self._coaching_llm_reply(instructions)

    async def _run_assessment_demo(self, demo_cues: list) -> bool:
        """Run the choreographed demo task. Returns False if it failed to run."""
        from agent.agents.coaching_demo_task import CoachingDemoTask, DemoStartAck
        from agent.services.demo_narration import generate_demo_lines
        from agent.services.inline_task_runner import start_inline_agent_task

        if not demo_cues or self._session.llm is None:
            return False

        # Script generation overlaps everything that follows (demo_start, morph-in).
        lines_task = asyncio.create_task(
            generate_demo_lines(self._session.llm, demo_cues)
        )
        self.is_coaching_speaking = True
        self._demo_start_ack = DemoStartAck()
        try:
            demo_task = CoachingDemoTask(
                cues=demo_cues,
                lines_task=lines_task,
                send_to_pipeline_fn=self._send_to_pipeline,
                started_ack=self._demo_start_ack,
            )
            runner = start_inline_agent_task(self._session, demo_task)
            await runner
            logger.info("[COACHING SERVICE] Choreographed demo complete")
            return True
        except Exception:
            logger.exception("[COACHING SERVICE] Demo task failed")
            lines_task.cancel()
            return False
        finally:
            self.is_coaching_speaking = False
            self._demo_start_ack = None

    def _send_to_pipeline(self, message: dict) -> None:
        if not self._coaching_ipc:
            return
        try:
            self._coaching_ipc.send_message(message)
        except Exception as e:
            logger.error(f"[COACHING SERVICE] Failed to send {message.get('type')}: {e}")

    # ------------------------------------------------------------------
    # Assessment logging
    # ------------------------------------------------------------------

    def start_assessment(
        self,
        session_dir: str | Path,
        session_id: str,
        user_height_cm: float,
        target_reps: int = 1,
    ) -> None:
        self._assessment_logger = AssessmentLogger(
            session_dir=Path(session_dir),
            session_id=session_id,
            user_height_cm=user_height_cm,
            target_reps=target_reps,
        )
        # Each pipeline process allows one choreographed demo per assessment,
        # so a new assessment (second pass through the flow) gets its demo.
        self._assessment_demo_played = False
        self._send_to_pipeline({"type": "assessment_mode", "enabled": True})
        logger.info("[COACHING SERVICE] Assessment logging started")

    def stop_assessment(self, passed: bool) -> None:
        if self._assessment_logger is not None:
            self._assessment_logger.finalize(passed)
            self._send_to_pipeline({"type": "assessment_mode", "enabled": False})
            logger.info(f"[COACHING SERVICE] Assessment logging finalized (passed={passed})")
            self._assessment_logger = None

    @property
    def assessment_logger(self) -> AssessmentLogger | None:
        return self._assessment_logger

    def _build_assessment_correction_prompt(
        self, diagnosis: dict, scoring: dict, round_num: int,
    ) -> str:
        immediate = diagnosis.get("immediate_causes", [])
        top_issue = immediate[0] if immediate else {}

        context_parts = []
        context_parts.append(f"Assessment round {round_num}: their squat form was just analyzed.")

        mean_pct = round(scoring.get("mean_score", 0) * 100)
        context_parts.append(f"Form score: {mean_pct} out of 100.")

        if top_issue:
            context_parts.append(
                f"Main issue: {top_issue.get('explanation', 'form issue detected')}."
            )
            delta = top_issue.get("parameter_delta")
            if delta:
                from biomechanics.diagnosis.demo_builder import summarize_cue_magnitude
                context_parts.append(
                    f"Recommended adjustment: "
                    f"{summarize_cue_magnitude(top_issue.get('cause_id', ''), delta)}."
                )

        if len(immediate) > 1:
            other = immediate[1]
            context_parts.append(
                f"Secondary issue: {other.get('explanation', '')}."
            )

        context_str = " ".join(context_parts)

        return (
            f"[PRE-WORKOUT ASSESSMENT] You are assessing the user's squat form before their workout. "
            f"This is a form check — NOT a workout set. "
            f"{context_str} "
            f"Tell the user the specific issue you found and what to adjust. "
            f"Be specific and actionable (e.g., 'widen your stance a bit' or 'push your knees out more'). "
            f"Then tell them to try it again with that adjustment. "
            f"Keep it encouraging and brief (2-3 sentences). "
            f"Do NOT say 'assessment' — frame it as wanting to see the fix, in your own words."
        )

    def _build_assessment_pass_prompt(
        self, diagnosis: dict, scoring: dict, round_num: int,
    ) -> str:
        session_causes = diagnosis.get("session_causes", [])
        contextual = diagnosis.get("contextual_notes", [])
        mean_pct = round(scoring.get("mean_score", 0) * 100)

        context_parts = []
        context_parts.append(
            f"Assessment complete after {round_num} round(s). Form score: {mean_pct} out of 100."
        )

        other_notes = []
        for cause in session_causes:
            explanation = cause.get("explanation", "")
            if explanation:
                other_notes.append(explanation)
        for note in contextual:
            explanation = note.get("explanation", "")
            if explanation:
                other_notes.append(explanation)

        if other_notes:
            context_parts.append(
                f"Things that may show up during heavier sets: {'; '.join(other_notes[:2])}."
            )

        context_str = " ".join(context_parts)

        if other_notes:
            return (
                f"[PRE-WORKOUT ASSESSMENT PASSED] The user's squat form looks good — no immediate fixes needed. "
                f"{context_str} "
                f"Praise their form briefly, then mention 1-2 things they might notice during heavier sets "
                f"(these are informational, not something to fix right now). "
                f"Then transition to calibration: tell them you need 5 deep bodyweight squats "
                f"to learn their movement pattern so you can coach them during the workout. "
                f"Keep it natural and brief (3-4 sentences max)."
            )
        else:
            return (
                f"[PRE-WORKOUT ASSESSMENT PASSED] The user's squat form looks great — no issues at all! "
                f"{context_str} "
                f"Praise their form enthusiastically. "
                f"Then transition to calibration: tell them you need 5 deep bodyweight squats "
                f"to fine-tune your understanding of how they move. "
                f"Keep it brief and hype (2-3 sentences)."
            )

    async def _on_calibration_complete(self, message: dict):
        """Handle calibration_complete IPC message — save to DB and notify user."""
        from db.database import SessionLocal
        from db.calibration_utils import save_user_calibration

        movement_pattern = message.get("movement_pattern", "squat")
        peaks = message.get("peaks", {})
        thresholds = message.get("thresholds", {})
        athlete_params = message.get("athlete_params")
        baseline = message.get("baseline")
        if athlete_params is None:
            # None tells the upsert to keep stored body measurements
            logger.warning(
                "[CALIBRATION] calibration_complete has no athlete_params — "
                "body measurements unfinished; stored athlete params are kept"
            )

        # Save calibration to database
        user_id = self._state.get("user.id")
        if user_id:
            db = SessionLocal()
            try:
                save_user_calibration(
                    db=db,
                    user_id=user_id,
                    movement_pattern=movement_pattern,
                    peaks=peaks,
                    thresholds=thresholds,
                    athlete_params=athlete_params,
                    baseline=baseline,
                )
                logger.info(f"[CALIBRATION] Saved calibration to DB for user={user_id} pattern={movement_pattern}")
            except Exception as e:
                logger.error(f"[CALIBRATION] Failed to save to DB: {e}", exc_info=True)
            finally:
                db.close()
        else:
            logger.warning("[CALIBRATION] No user_id in state — cannot save calibration to DB")

        # Store profile in state for future use
        self._state.set("workout.calibration_profile", thresholds)
        self._state.set("calibration.active", None)
        self._state.set("calibration.movement_pattern", None)
        self._state.save_state()

        # Announce calibration completion via LLM
        instructions = (
            "[CALIBRATION COMPLETE] Calibration is done — you now know exactly how they move. "
            "Tell them you're dialed in and the workout starts now, in one short hyped sentence "
            "of your own. Don't say 'calibration'."
        )
        await self._coaching_llm_reply(instructions)

        # Reset orchestrator for the actual workout sets
        if self._coaching_orchestrator:
            target_reps = self._get_current_target_reps()
            total_sets = self._get_total_sets()
            self._coaching_orchestrator.reset_set(target_reps=target_reps, total_sets=total_sets)
            self._workout_active = True
            logger.info("[CALIBRATION] Orchestrator reset for workout phase — workout_active=True")

        # Notify voice agent so it can start the wake word system now that
        # calibration speech is finished and the user is ready to work out.
        if self._on_calibration_complete_callback:
            try:
                ret = self._on_calibration_complete_callback()
                if asyncio.iscoroutine(ret):
                    await ret
            except Exception as e:
                logger.error(f"[CALIBRATION] on_calibration_complete callback error: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # Orchestrator Setup
    # ------------------------------------------------------------------

    def _init_orchestrator(self):
        """Initialize the coaching orchestrator with internal callbacks."""
        from agent.services.coaching_orchestrator import CoachingOrchestrator

        self._coaching_orchestrator = CoachingOrchestrator(
            play_cached_audio_fn=self._play_cached_cue_audio,
            generate_llm_reply_fn=self._coaching_llm_reply,
            get_cue_audio_fn=self._get_cached_audio,
            advance_set_fn=self._advance_workout_set,
            on_workout_complete_fn=self._on_workout_complete,
            prune_context_fn=self._prune_conversation_context,
            affect_service=getattr(getattr(self._session, "userdata", None), "affect_service", None),
        )
        self._coaching_orchestrator._generate_tts_fn = self._generate_tts_audio
        self._coaching_orchestrator._play_raw_frames_fn = self._play_raw_audio
        self._coaching_orchestrator._speak_adjustment_fn = self._speak_adjustment_feedback

        target_reps = self._get_current_target_reps()
        total_sets = self._get_total_sets()
        self._coaching_orchestrator.reset_set(target_reps=target_reps, total_sets=total_sets)
        self._coaching_orchestrator.start()

    def _get_workout_session(self):
        """Deserialize WorkoutSession from state. Single entry point to avoid repeated parsing."""
        session_data = self._state.get("workout.current_session")
        if not session_data:
            return None
        from agent.core.workout_session import WorkoutSession
        try:
            return WorkoutSession.from_dict(session_data)
        except Exception:
            return None

    def _get_current_target_reps(self) -> Optional[int]:
        """Get target reps for the current set from WorkoutSession."""
        session = self._get_workout_session()
        if session:
            current_set = session.get_current_set()
            if current_set:
                return current_set.target_reps
        return None

    def _get_total_sets(self) -> Optional[int]:
        """Get total number of sets for the current exercise."""
        session = self._get_workout_session()
        if session:
            exercise = session.get_current_exercise()
            if exercise:
                return len(exercise.sets)
        return None

    def _get_set_numbers(self) -> tuple:
        """Get (completed_set, next_set, total_sets) from WorkoutSession."""
        session = self._get_workout_session()
        if session:
            exercise = session.get_current_exercise()
            if exercise:
                next_set = exercise.current_set_index + 1
                completed_set = next_set - 1
                total_sets = len(exercise.sets)
                return completed_set, next_set, total_sets
        return 1, 2, 3  # Safe fallback

    # ------------------------------------------------------------------
    # Set Management
    # ------------------------------------------------------------------

    async def _advance_workout_set(self) -> Optional[int]:
        """Advance WorkoutSession to the next set. Returns new target_reps or None."""
        session = self._get_workout_session()
        if not session:
            logger.warning("[COACHING SERVICE] No active session to advance")
            return None

        try:

            completed_set = session.get_current_set()
            rest_seconds = completed_set.rest_seconds if completed_set else 30
            if self._coaching_orchestrator:
                self._coaching_orchestrator.rest_seconds = rest_seconds

            rep_count = (
                self._coaching_orchestrator.set_rep_count
                if self._coaching_orchestrator
                else 0
            )
            session.mark_set_complete(performed_reps=rep_count)

            has_next = session.advance_to_next_set()

            self._state.set("workout.current_session", session.to_dict())
            self._state.save_state()

            if has_next:
                next_set = session.get_current_set()
                new_target = next_set.target_reps if next_set else None
                logger.info(f"[COACHING SERVICE] Advanced to next set — target_reps={new_target}")

                if self._coaching_ipc and rest_seconds > 0:
                    try:
                        self._coaching_ipc.send_message({
                            "type": "rest_start",
                            "rest_seconds": rest_seconds,
                        })
                        logger.info(f"[COACHING SERVICE] Sent rest_start ({rest_seconds}s)")
                    except Exception as e:
                        logger.error(f"[COACHING SERVICE] Failed to send rest_start: {e}")

                return new_target
            else:
                logger.info("[COACHING SERVICE] Workout complete — no more sets")

                # Signal pipeline to stop counting reps
                if self._coaching_ipc:
                    try:
                        self._coaching_ipc.send_message({"type": "workout_complete"})
                        logger.info("[COACHING SERVICE] Sent workout_complete to pipeline")
                    except Exception as e:
                        logger.error(f"[COACHING SERVICE] Failed to send workout_complete: {e}")

                return None

        except Exception:
            logger.exception("[COACHING SERVICE] Failed to advance workout set")
            return None

    async def force_end_current_set(self, reps: int) -> dict:
        """Force-end the current set early (called by workout agent when user verbally stops).

        Puts the orchestrator into rest mode, overrides the rep count,
        advances the WorkoutSession, and resets for the next set if any.
        """
        from agent.core.workout_session import WorkoutSession

        session_data = self._state.get("workout.current_session")
        if not session_data:
            return {"status": "no_session"}

        session = WorkoutSession.from_dict(session_data)
        completed_set = session.get_current_set()
        rest_seconds = completed_set.rest_seconds if completed_set else 60

        if self._coaching_orchestrator:
            self._coaching_orchestrator.resting = True
            self._coaching_orchestrator.set_rep_count = reps

        new_target = await self._advance_workout_set()

        if new_target is not None:
            # There is a next set — reset orchestrator
            if self._coaching_orchestrator:
                self._coaching_orchestrator.reset_set(target_reps=new_target)

            session_data = self._state.get("workout.current_session")
            next_desc = ""
            if session_data:
                s = WorkoutSession.from_dict(session_data)
                next_desc = s.get_current_exercise_description()

            return {
                "status": "advanced",
                "rest_seconds": rest_seconds,
                "next_set_description": next_desc,
            }
        else:
            return {"status": "workout_complete"}

    async def _on_workout_complete(self):
        """Called by orchestrator after exercise recap is spoken."""
        logger.info("[COACHING SERVICE] Workout complete — notifying voice agent")
        await self._close_biomech_recording()
        if self._on_workout_complete_callback:
            await self._on_workout_complete_callback({"workout_complete": True})

    # ------------------------------------------------------------------
    # Audio Playback
    # ------------------------------------------------------------------

    async def _play_cached_cue_audio(self, cue_key: str):
        """Play a cached cue through LiveKit's session.say()."""
        logger.info(f"[COACHING SERVICE] → Playing cached cue: {cue_key}")
        self._publish_cue_banner(cue_key)
        if self._audio_cue_service:
            await self._audio_cue_service.play_cue(cue_key)
            logger.info(f"[COACHING SERVICE] ✓ Cached cue played: {cue_key}")
        else:
            logger.warning(f"[COACHING SERVICE] No audio_cue_service — cannot play: {cue_key}")

    def _publish_cue_banner(self, cue_key: str) -> None:
        """Mirror a spoken cue on the display page. Rep-count callouts are
        skipped — the on-screen rep counter already ticks for those."""
        if self._visual_bridge is None or cue_key.startswith("rep_"):
            return
        text = CUE_TEXT_MAP.get(cue_key)
        if not text:
            return
        self._visual_bridge.send({
            "type": "cue",
            "key": cue_key,
            "text": text,
            "kind": "positive" if cue_key in POSITIVE_CUE_KEYS else "correction",
        })

    def _get_cached_audio(self, cue_key: str) -> bool:
        """Check if cached audio exists for a cue key."""
        if self._audio_cue_service:
            return self._audio_cue_service.has_cue(cue_key)
        return False

    async def _generate_tts_audio(self, text: str) -> list | None:
        """Generate TTS audio frames for arbitrary text (preemptive speech)."""
        if not self._audio_cue_service:
            return None
        return await self._audio_cue_service.generate_tts(text)

    async def _play_raw_audio(self, frames: list) -> None:
        """Play raw AudioFrame list through the audio cue service."""
        if self._audio_cue_service:
            await self._audio_cue_service.play_frames(frames)

    # ------------------------------------------------------------------
    # Adjustment feedback (Feature 2: after-cue coaching loop)
    # ------------------------------------------------------------------

    @staticmethod
    def _adjustment_cue_key(param: str, delta: float, hit_target: bool) -> str | None:
        if hit_target:
            return ADJUSTMENT_ON_TARGET_CUE
        directions = ADJUSTMENT_CUES.get(param)
        if not directions:
            return None
        # delta = target - current, so positive means "give me more".
        return directions["more"] if delta > 0 else directions["less"]

    async def _speak_adjustment_feedback(self, param: str, delta: float, hit_target: bool) -> None:
        # Pre-cached audio first — this fires every 1.5s between reps, and an
        # LLM round-trip lands after the lifter has already moved.
        cue_key = self._adjustment_cue_key(param, delta, hit_target)
        if cue_key and self._get_cached_audio(cue_key):
            self.is_coaching_speaking = True
            try:
                await self._play_cached_cue_audio(cue_key)
            except Exception as e:
                logger.error(f"[COACHING SERVICE] Adjustment cue failed: {e}", exc_info=True)
            finally:
                self.is_coaching_speaking = False
            return

        param_label = ADJUSTMENT_PARAM_LABELS.get(param, param)
        self.is_coaching_speaking = True
        try:
            if hit_target:
                user_input = f"{param_label} is right on target"
            elif param == "stance_width":
                if delta > 0:
                    user_input = f"Stance is too narrow, go wider"
                else:
                    user_input = f"Stance is too wide, bring it in"
            elif param == "toe_out":
                if delta > 0:
                    user_input = f"Toes need about {abs(delta):.0f} degrees more turn-out"
                else:
                    user_input = f"Toes are turned out too far by about {abs(delta):.0f} degrees"
            else:
                user_input = f"Adjust your {param_label}"

            # These fire every 1.5s, so they are stripped from the chat
            # context afterwards rather than left to bloat TTFT.
            # generate_reply has no add_to_chat_ctx parameter — that belongs
            # to session.say() — so the removal is manual. It is done by item
            # id, not by truncating to a saved length: a set recap or
            # motivation reply can finish mid-playout here, and a positional
            # trim would either strand this turn or delete theirs.
            from livekit.agents import llm

            agent = self._session.current_agent
            user_message = llm.ChatMessage(role="user", content=[user_input])

            handle = self._session.generate_reply(
                instructions=ADJUSTMENT_SYSTEM_PROMPT,
                user_input=user_message,
                tool_choice="none",
                allow_interruptions=False,
            )
            await handle.wait_for_playout()

            if agent:
                ephemeral = {user_message.id} | {item.id for item in handle.chat_items}
                new_ctx = llm.ChatContext.empty()
                for item in agent.chat_ctx.items:
                    if item.id not in ephemeral:
                        new_ctx.items.append(item)
                await agent.update_chat_ctx(new_ctx)
        except Exception as e:
            logger.error(f"[COACHING SERVICE] Adjustment feedback failed: {e}", exc_info=True)
        finally:
            self.is_coaching_speaking = False

    # ------------------------------------------------------------------
    # LLM & Audio Ducking
    # ------------------------------------------------------------------

    async def _prune_conversation_context(self, max_items: int = 20):
        """Prune old conversation items, injecting compaction summary.

        If a CompactionService is active, injects its pre-built summary
        so historical context is preserved. Falls back to simple truncation.
        """
        try:
            agent = self._session.current_agent
            if agent is None:
                return
            ctx = agent.chat_ctx
            if len(ctx.items) <= max_items:
                return

            old_count = len(ctx.items)

            # Try to get compaction summary via agent's userdata
            summary_text = ""
            userdata = getattr(agent, 'userdata', None)
            if userdata:
                compaction = getattr(userdata, 'compaction_service', None)
                if compaction:
                    summary_text = compaction.get_summary()

            if not summary_text:
                # Fallback: simple truncation (original behavior)
                new_ctx = ctx.copy()
                new_ctx.truncate(max_items=max_items)
                await agent.update_chat_ctx(new_ctx)
                logger.info(
                    f"[COACHING SERVICE] Pruned context: "
                    f"{old_count} → {len(new_ctx.items)} items (no summary)"
                )
                return

            # Summary-aware pruning: system items + summary + recent items
            from livekit.agents import llm

            items = list(ctx.items)
            system_items = [i for i in items if hasattr(i, 'role') and i.role in ("system", "developer")]
            non_system = [i for i in items if not (hasattr(i, 'role') and i.role in ("system", "developer"))]
            recent_items = non_system[-max_items:] if len(non_system) > max_items else non_system

            new_ctx = llm.ChatContext.empty()
            for item in system_items:
                new_ctx.items.append(item)

            summary_message = llm.ChatMessage(
                role="system",
                content=[f"[CONVERSATION SUMMARY]\n{summary_text}"],
            )
            new_ctx.items.append(summary_message)

            for item in recent_items:
                new_ctx.items.append(item)

            await agent.update_chat_ctx(new_ctx)
            logger.info(
                f"[COACHING SERVICE] Pruned context with summary: "
                f"{old_count} → {len(new_ctx.items)} items "
                f"({len(system_items)} system + 1 summary + {len(recent_items)} recent)"
            )
            logger.info("[COMPACTION:SWAP] Summary injected into context on prune")
        except Exception as e:
            logger.warning(f"[COACHING SERVICE] Context pruning failed: {e}")

    async def _coaching_llm_reply(self, instructions: str):
        """Generate coaching speech via generate_reply with SpeechHandle tracking.

        Sends the coaching persona as system instructions and the coaching
        task as user_input so the LLM has a clear user-role message to
        respond to (Llama models may generate near-empty responses when
        only system instructions are provided with no user turn).

        Strips the user_input + assistant response from the chat context
        after playout so ephemeral coaching calls don't bloat TTFT.
        """
        logger.info(f"[COACHING SERVICE] → Coaching LLM | instructions[:80]={instructions[:80]}...")
        self.is_coaching_speaking = True
        try:
            agent = self._session.current_agent
            ctx_len_before = len(agent.chat_ctx.items) if agent else 0

            if self._spoken_coaching_lines:
                recent = " | ".join(self._spoken_coaching_lines)
                instructions = (
                    f"{instructions}\n\nYou already said these lines this session — "
                    f"do not reuse their phrasing or openers: {recent}"
                )

            handle = self._session.generate_reply(
                instructions=COACHING_PERSONA,
                user_input=instructions,
                tool_choice="none",
                allow_interruptions=False,
            )
            await handle.wait_for_playout()
            logger.info("[COACHING SERVICE] ✓ Coaching LLM playout complete")

            if agent and len(agent.chat_ctx.items) > ctx_len_before:
                added = len(agent.chat_ctx.items) - ctx_len_before
                for item in list(agent.chat_ctx.items)[ctx_len_before:]:
                    if getattr(item, "role", None) == "assistant":
                        spoken = getattr(item, "text_content", None)
                        if spoken:
                            self._spoken_coaching_lines.append(spoken.strip())
                from livekit.agents import llm
                new_ctx = llm.ChatContext.empty()
                for item in list(agent.chat_ctx.items)[:ctx_len_before]:
                    new_ctx.items.append(item)
                await agent.update_chat_ctx(new_ctx)
                logger.debug(
                    f"[COACHING SERVICE] Stripped {added} ephemeral coaching items from context"
                )

            return handle
        except Exception as e:
            logger.error(f"[COACHING SERVICE] Coaching LLM reply failed: {e}", exc_info=True)
            return None
        finally:
            self.is_coaching_speaking = False

