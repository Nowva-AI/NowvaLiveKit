#!/usr/bin/env python3
"""
Nowva Main Application
Orchestrates voice agent and pose estimation with IPC communication
"""

from __future__ import annotations

import asyncio
import atexit
import os
import select
import sys
import signal
import threading
import subprocess
import logging
import time
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from agent.core.session_manager import SessionManager
from agent.core.ipc_communication import IPCServer
from agent.core.session_logger import SessionLogger
from db import init_db
from agent.agents.console_launcher import run_console_voice_onboarding, terminate_process_group
from auth.user_management import create_user_account
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from profiler.collector import SessionProfiler

# Suppress SQLAlchemy INFO logs
logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)

DEFAULT_EXERCISE_NAME = "Barbell Back Squat"
COACHING_SOCKET_PATH = "/tmp/nowva_coaching.sock"

# Played via afplay when the display window opens (browser autoplay policies
# block page-side audio without a user gesture). Any afplay-supported format
# works (wav/mp3/aiff/m4a); silently skipped if the file doesn't exist.
BOOT_SOUND_PATH = Path(
    os.environ.get("NOWVA_BOOT_SOUND", str(Path(__file__).parent / "assets" / "boot_sound.wav"))
)

# Session params seeded by --test_assess. The assessment itself is bodyweight
# with a pipeline-defined rep count; these only matter if the run continues
# past calibration into the workout.
TEST_ASSESS_SETS = 3
TEST_ASSESS_REPS = 5
TEST_ASSESS_WEIGHT_LBS = 45.0
TEST_ASSESS_REST_SECONDS = 120


class _TeeStream:
    def __init__(self, original, log_file):
        self._original = original
        self._log_file = log_file

    def write(self, data):
        self._original.write(data)
        self._log_file.write(data)

    def flush(self):
        self._original.flush()
        self._log_file.flush()

    def fileno(self):
        return self._original.fileno()

    def isatty(self):
        return self._original.isatty()


# Set NOWVA_LOG_CONSOLE=1 to mirror all console output to session_logs/console_<timestamp>.log
_tee_file = None
if os.environ.get("NOWVA_LOG_CONSOLE", "").lower() == "true":
    from datetime import datetime as _dt
    _log_dir = Path("session_logs")
    _log_dir.mkdir(exist_ok=True)
    _tee_file = open(_log_dir / f"console_{_dt.now().strftime('%Y%m%d_%H%M%S')}.log", "w")
    _original_stdout = sys.stdout
    _original_stderr = sys.stderr

    def _close_tee():
        # Restore the real streams BEFORE closing the log file — CPython's
        # final flush of sys.stdout would otherwise hit the closed tee file
        # and fail interpreter finalization (exit code 120)
        sys.stdout = _original_stdout
        sys.stderr = _original_stderr
        _tee_file.close()

    atexit.register(_close_tee)
    sys.stdout = _TeeStream(sys.stdout, _tee_file)
    sys.stderr = _TeeStream(sys.stderr, _tee_file)


class NowvaApp:
    """Main Nowva application orchestrator"""

    def __init__(self):
        self.session_manager = SessionManager()
        self.session_logger = SessionLogger.get_instance()
        self.ipc_server = None
        self.coaching_ipc = None  # Second IPC server for forwarding to voice agent
        self.pose_process = None
        self.fastapi_process = None
        self.current_user = None
        self.state = None  # Track state for cleanup
        self._recording_process = None
        self._recording_log = None
        self._session_dir: Path | None = None
        self._fastapi_log = None
        self._cal_file: str | None = None
        self.test_assess_mode = False
        self.display_server = None
        self._boot_progress = 0.0
        self._boot_sound_last_played = float("-inf")

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Trigger graceful shutdown. State reset happens in _cleanup(), which
        run() guarantees via try/finally — mutating state here could interrupt
        the main thread mid-reload and corrupt it."""
        print("\n[SIGNAL] Received shutdown signal - shutting down...")
        raise KeyboardInterrupt

    def _start_fastapi_server(self):
        """Start the FastAPI backend server as a subprocess."""
        project_root = str(Path(__file__).parent.parent)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).parent)
        env["DYLD_FALLBACK_LIBRARY_PATH"] = f"/opt/homebrew/lib:{env.get('DYLD_FALLBACK_LIBRARY_PATH', '')}"

        # Log to a file instead of PIPE — undrained pipes fill the OS buffer
        # (~16KB) and deadlock uvicorn once it has logged enough.
        log_dir = Path("session_logs")
        log_dir.mkdir(exist_ok=True)
        self._fastapi_log = open(log_dir / "fastapi.log", "w")
        self.fastapi_process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "api.main:app", "--port", "8000"],
            cwd=project_root,
            env=env,
            stdout=self._fastapi_log,
            stderr=subprocess.STDOUT,
        )
        print(f"[FASTAPI] Server started (PID: {self.fastapi_process.pid}, logs: {log_dir / 'fastapi.log'})")

    # ------------------------------------------------------------------
    #  Screen + audio recording (NOWVA_RECORD_SESSION=true)
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_avfoundation_devices() -> dict[str, int | None]:
        """Parse ffmpeg avfoundation device list to find screen, mic, and BlackHole indices."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                capture_output=True, text=True, timeout=5,
            )
            output = result.stderr
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return {"screen": None, "mic": None, "blackhole": None}

        screen_idx: int | None = None
        mic_idx: int | None = None
        blackhole_idx: int | None = None
        in_audio = False

        for line in output.splitlines():
            if "AVFoundation video devices" in line:
                in_audio = False
            elif "AVFoundation audio devices" in line:
                in_audio = True
                continue

            bracket = line.rfind("[")
            close = line.find("]", bracket + 1)
            if bracket == -1 or close == -1:
                continue
            try:
                idx = int(line[bracket + 1:close])
            except ValueError:
                continue

            name = line[close + 1:].strip()

            if not in_audio:
                if "screen" in name.lower() or "capture screen" in name.lower():
                    if screen_idx is None:
                        screen_idx = idx
            else:
                if "blackhole" in name.lower():
                    blackhole_idx = idx
                elif "microphone" in name.lower() or "built-in" in name.lower():
                    if mic_idx is None:
                        mic_idx = idx

        return {"screen": screen_idx, "mic": mic_idx, "blackhole": blackhole_idx}

    async def _start_screen_recording(self, session_dir: Path) -> None:
        """Launch ffmpeg to record screen + audio into session_dir."""
        devices = self._detect_avfoundation_devices()

        if devices["screen"] is None:
            print("[RECORDING] No screen capture device found — skipping recording")
            return

        screen_idx = devices["screen"]
        mic_idx = devices["mic"]
        blackhole_idx = devices["blackhole"]

        if mic_idx is not None and blackhole_idx is not None:
            cmd = [
                "ffmpeg", "-y",
                "-f", "avfoundation", "-framerate", "30",
                "-capture_cursor", "1",
                "-i", f"{screen_idx}:{mic_idx}",
                "-f", "avfoundation",
                "-i", f":{blackhole_idx}",
                "-map", "0:v", "-map", "0:a", "-map", "1:a",
                "-metadata:s:a:0", "title=Microphone",
                "-metadata:s:a:1", "title=System Audio",
                "-c:v", "h264_videotoolbox", "-b:v", "5M",
                "-c:a", "aac", "-ar", "48000", "-b:a", "128k",
                str(session_dir / "screen_recording.mp4"),
            ]
            print(f"[RECORDING] Screen + mic + system audio (BlackHole, separate tracks)")
        elif mic_idx is not None:
            cmd = [
                "ffmpeg", "-y",
                "-f", "avfoundation", "-framerate", "30",
                "-capture_cursor", "1",
                "-i", f"{screen_idx}:{mic_idx}",
                "-c:v", "h264_videotoolbox", "-b:v", "5M",
                "-c:a", "aac", "-ar", "48000", "-b:a", "128k",
                str(session_dir / "screen_recording.mp4"),
            ]
            print("[RECORDING] Screen + mic (no BlackHole detected for system audio)")
        else:
            cmd = [
                "ffmpeg", "-y",
                "-f", "avfoundation", "-framerate", "30",
                "-capture_cursor", "1",
                "-i", f"{screen_idx}:none",
                "-c:v", "h264_videotoolbox", "-b:v", "5M",
                str(session_dir / "screen_recording.mp4"),
            ]
            print("[RECORDING] Screen only (no audio devices detected)")

        self._recording_log = open(session_dir / "ffmpeg_recording.log", "w")
        self._recording_process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self._recording_log,
        )
        print(f"[RECORDING] Started (PID: {self._recording_process.pid})")

        await asyncio.sleep(1)
        if self._recording_process.poll() is not None:
            self._recording_log.close()
            log_content = (session_dir / "ffmpeg_recording.log").read_text()
            print(f"[RECORDING] ffmpeg exited immediately! stderr:\n{log_content}")

    def _stop_screen_recording(self) -> None:
        """Gracefully stop ffmpeg recording."""
        proc = getattr(self, "_recording_process", None)
        if proc is None or proc.poll() is not None:
            self._close_recording_log()
            return

        print("[RECORDING] Stopping screen recording...")
        try:
            proc.stdin.write(b"q")
            proc.stdin.flush()
            proc.wait(timeout=10)
            print("[RECORDING] Recording saved")
        except (BrokenPipeError, OSError):
            proc.terminate()
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait(timeout=5)
        self._close_recording_log()

    def _close_recording_log(self) -> None:
        log = getattr(self, "_recording_log", None)
        if log and not log.closed:
            log.close()

    def _stop_pose_process(self) -> None:
        """SIGTERM the pose process and wait for its data-saving finally block.

        The pose subprocess writes set plots and the session dashboard on
        shutdown — give it time to finish before force-killing.
        """
        if not self.pose_process:
            return
        self.pose_process.terminate()
        try:
            self.pose_process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            print("[POSE] Did not exit within 30s — force killing")
            self.pose_process.kill()
            self.pose_process.wait()
        self.pose_process = None

    def check_session(self):
        """Check if user has an existing session"""
        return self.session_manager.session_exists()

    def load_user_from_session(self):
        """Load user from existing session"""
        session = self.session_manager.load_session()
        if not session:
            return None

        user_id = session.get('user_id')
        username = session.get('username')

        return {
            'user_id': user_id,
            'username': username,
            'email': session.get('email')
        }

    def _seed_test_assessment_state(self) -> None:
        """Mirror CollectExerciseInfoTask.start_workout() so the app boots
        straight into the pre-workout form assessment (--test_assess)."""
        from agent.agents.shared.helpers import start_calibration_mode
        from agent.core.workout_session import WorkoutSession

        start_calibration_mode(self.state, DEFAULT_EXERCISE_NAME, {"type": "quick_exercise"})
        session = WorkoutSession.create_quick_session(
            user_id=self.current_user["user_id"],
            exercise_name=DEFAULT_EXERCISE_NAME,
            sets=TEST_ASSESS_SETS,
            reps=TEST_ASSESS_REPS,
            weight=TEST_ASSESS_WEIGHT_LBS,
            rest_seconds=TEST_ASSESS_REST_SECONDS,
        )
        self.state.set("workout.calibration_profile", None)  # drop stale profile from a prior run
        self.state.set("workout.current_session", session.to_dict())
        self.state.set("workout.exercise_name", DEFAULT_EXERCISE_NAME)
        self.state.set("workout.active", True)
        self.state.switch_mode("workout")
        self.state.save_state()
        print("[TEST ASSESS] State seeded — booting directly into form assessment")

    def create_user(self, first_name: str, email: str):
        """
        Create new user in database using auth system

        Args:
            first_name: User's first name
            email: User's email

        Returns:
            tuple of (user_id, username) if successful, (None, None) otherwise
        """
        try:
            # Use the auth system's create_user_account which handles:
            # - Duplicate checking
            # - Username generation
            # - Password creation
            user, username = create_user_account(first_name, email)

            if user:
                print(f"User account ready: {username} (ID: {user.id})")
                return user.id, username

            return None, None

        except Exception as e:
            print(f"Error creating user: {e}")
            return None, None

    def _load_athlete_calibration(self, exercise_name: str) -> dict:
        """Body proportions + ROM baseline for a returning user, if stored.

        Returns {} when unavailable — the pipeline degrades to no diagnosis
        rather than failing, and logs which case it hit.
        """
        try:
            from agent.agents.shared.helpers import normalize_exercise_name
            from biomechanics.calibration import get_movement_pattern
            from db.database import SessionLocal
            from db.calibration_utils import get_user_calibration_full

            user_id = self.state.get("user.id")
            # Same normalization check_calibration uses to find this row —
            # get_movement_pattern only knows canonical names, so a stored
            # "squat" resolves to None unless it is normalized first.
            canonical = normalize_exercise_name(exercise_name) or exercise_name
            pattern = get_movement_pattern(canonical)
            if not user_id or not pattern:
                print(
                    f"[CALIBRATION] No athlete params lookup: "
                    f"user_id={'set' if user_id else 'MISSING'} "
                    f"exercise={exercise_name!r} pattern={pattern!r}"
                )
                return {}

            db = SessionLocal()
            try:
                full = get_user_calibration_full(db, user_id, pattern)
            finally:
                db.close()

            if not full or not full.get("athlete_params"):
                print(
                    "[CALIBRATION] Stored calibration has no athlete_params — "
                    "diagnosis and stance coaching will be unavailable this session"
                )
                return {}
            return {
                "athlete_params": full["athlete_params"],
                "baseline": full.get("baseline") or {},
            }
        except Exception as e:
            print(f"[CALIBRATION] Could not load athlete params: {e}")
            return {}

    def start_pose_estimation(self, cam0_id: int = 0, cam1_id: int = 1,
                              exercise_name: str = DEFAULT_EXERCISE_NAME,
                              calibration_file: str = None,
                              calibration_mode: bool = False,
                              preload: bool = False):
        """
        Start pose estimation process

        Args:
            cam0_id: First camera ID
            cam1_id: Second camera ID
            exercise_name: Name of the exercise for coaching cues
            calibration_file: Path to existing calibration JSON (if returning user)
            calibration_mode: If True, run calibration phase before workout
            preload: If True, pass --preload so subprocess loads the model
                     without opening the camera, then waits for start_capture.
        """
        print("\nStarting pose estimation process...")

        # Start pose estimation as subprocess
        pose_script = Path(__file__).parent / 'biomechanics' / 'pipeline_process.py'

        cmd = [sys.executable, "-u", str(pose_script), str(cam0_id), str(cam1_id), exercise_name]
        if calibration_file:
            cmd.extend(["--calibration-file", calibration_file])
        if calibration_mode:
            cmd.append("--calibration-mode")
        if preload:
            cmd.append("--preload")
        # Multi-camera T-pose calibration scales every 3D length by the
        # user's height; without it the subprocess falls back to an env default.
        user_height_cm = self.state.get("user.height_cm")
        if user_height_cm:
            cmd.extend(["--user-height-cm", str(float(user_height_cm))])

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

        self.pose_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        # Print pose process output
        def print_output():
            for line in self.pose_process.stdout:
                print(f"[Pose] {line.strip()}")

        output_thread = threading.Thread(target=print_output, daemon=True)
        output_thread.start()

        print("Pose estimation process started")

    def _start_display(self):
        """Start the display hub and open the browser on the idle screen.

        The boot sound fires when the page actually connects (websocket
        subscribe), not when the browser is asked to open — so audio and
        the page's fade-in land together.
        """
        from visual import DISPLAY_PORT, DisplayServer

        self.display_server = DisplayServer(on_subscriber=self._play_boot_sound)
        if not self.display_server.start():
            print("[DISPLAY] Server failed to start — continuing without display")
            self.display_server = None
            return
        print(f"[DISPLAY] Live at http://localhost:{DISPLAY_PORT}")

        def _open_browser():
            import webbrowser
            webbrowser.open(f"http://localhost:{DISPLAY_PORT}")

        threading.Thread(target=_open_browser, daemon=True).start()

    def _play_boot_sound(self):
        """Fire the display-open sound without blocking; no-op if absent.

        Debounced: page refreshes replay it, rapid reconnects don't.
        """
        now = time.monotonic()
        # Debounce past the clip length so a quick refresh can't overlap
        # two full-volume instances of the sound.
        if now - self._boot_sound_last_played < 6.0:
            return
        self._boot_sound_last_played = now
        if not BOOT_SOUND_PATH.exists():
            print(f"[DISPLAY] No boot sound at {BOOT_SOUND_PATH} — skipping")
            return
        try:
            subprocess.Popen(
                ["afplay", str(BOOT_SOUND_PATH)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as e:
            print(f"[DISPLAY] Boot sound failed: {e}")

    def _boot_step(self, label: str, progress: float):
        """Publish a boot milestone to the display page (monotonic)."""
        if not self.display_server or progress <= self._boot_progress:
            return
        self._boot_progress = progress
        self.display_server.publish({
            "type": "boot", "label": label, "progress": round(progress, 3),
        })

    def _publish_display(self, event: dict) -> None:
        """Best-effort publish to the display page; no-op without a display."""
        if self.display_server:
            self.display_server.publish(event)

    def _publish_workout_start(self) -> None:
        """Push the workout config to the display so the HUD knows targets."""
        exercise_name = self.state.get("workout.exercise_name") or DEFAULT_EXERCISE_NAME
        total_sets, target_reps, weight = 0, 0, 0.0
        session_data = self.state.get("workout.current_session") or {}
        exercises = session_data.get("exercises") or []
        if exercises:
            sets = exercises[0].get("sets") or []
            total_sets = len(sets)
            if sets:
                target_reps = sets[0].get("target_reps") or 0
                weight = sets[0].get("target_weight") or 0.0
        self._publish_display({
            "type": "workout",
            "action": "start",
            "exercise": exercise_name,
            "total_sets": total_sets,
            "target_reps": target_reps,
            "weight_lbs": weight,
        })

    # Markers in the voice agent's stdout mapped to boot milestones.
    # Matched by substring in _pump_agent_output as the agent initializes.
    _AGENT_BOOT_MARKERS = (
        ("[PREWARM] Starting parallel pre-load", "Spinning up neural cores", 0.34),
        ("[PREWARM] Silero VAD", "Calibrating voice activity sensors", 0.46),
        ("[PREWARM] Audio cues pre-loaded", "Loading coaching audio matrix", 0.55),
        ("[PREWARM] WakeWordModel pre-loaded", "Arming wake-word sentinel", 0.63),
        ("[PREWARM] Affect engine pre-loaded", "Tuning affect perception", 0.66),
        ("[NOVA] Initializing cascade pipeline", "Synthesizing speech cortex", 0.74),
        ("[NOVA] Agent session created", "Linking conversational reasoning engine", 0.88),
        ("Nova voice agent started in room", "All systems nominal — Nova online", 1.0),
    )

    def _scan_agent_boot_markers(self, line: str):
        if self._boot_progress >= 1.0:
            return
        for marker, label, progress in self._AGENT_BOOT_MARKERS:
            if marker in line:
                self._boot_step(label, progress)
                return

    def _start_coaching_ipc(self):
        """Bind the coaching IPC server and serve it from a daemon thread.

        Called at startup and again whenever workout mode restarts — the
        server is stopped when a workout ends, so a second pass through
        the flow needs a fresh bind for the voice agent to reconnect to.
        """
        def _coaching_message_handler(message: dict):
            """Handle messages FROM voice agent (reverse direction)."""
            msg_type = message.get("type")
            if msg_type not in ("rest_start", "workout_complete", "demo_start",
                                "demo_cue", "demo_end", "assessment_mode",
                                "request_last_rep", "request_demo"):
                return
            if msg_type == "rest_start":
                rest_sec = message.get("rest_seconds", 30)
                print(f"[COACHING IPC] Received rest_start ({rest_sec}s) from voice agent")
                self._publish_display({
                    "type": "rest", "action": "start", "seconds": rest_sec,
                })
            elif msg_type == "workout_complete":
                print("[COACHING IPC] Received workout_complete from voice agent")
                self._publish_display({"type": "workout", "action": "complete"})
            # Snapshot the reference — this runs on the coaching IPC thread
            # while the main loop can nil self.ipc_server during shutdown
            pose_ipc = self.ipc_server
            if pose_ipc and pose_ipc.client_socket:
                try:
                    pose_ipc.send_message(message)
                    print(f"[COACHING IPC] Forwarded {msg_type} to pose process")
                except Exception as e:
                    print(f"[COACHING IPC] Failed to forward {msg_type}: {e}")

        self.coaching_ipc = IPCServer(socket_path=COACHING_SOCKET_PATH)
        self.coaching_ipc.bind(message_callback=_coaching_message_handler)

        def _run_coaching_server():
            # Snapshot — the main loop nils self.coaching_ipc during shutdown
            coaching = self.coaching_ipc
            try:
                coaching.accept_client()
                coaching.listen()
            except OSError:
                # Expected when stop() closes the socket during shutdown
                if coaching.running:
                    raise

        threading.Thread(target=_run_coaching_server, daemon=True).start()
        print("[COACHING IPC] Server bound — waiting for voice agent to connect")

    async def run_onboarding(self, state_notify_fd: int | None = None):
        """
        Run voice-based onboarding flow

        Returns:
            Tuple of (success: bool, agent_process: subprocess.Popen or None)
        """
        print("\n" + "="*50)
        print("VOICE ONBOARDING")
        print("="*50)
        print("\nStarting voice-based onboarding...")
        print("The voice agent will:")
        print("1. Welcome you to Nowva")
        print("2. Explain the product")
        print("3. Ask for your name")
        print("4. Ask for your email")
        print("5. Confirm your information")
        print("6. Transition to main menu\n")

        # Run voice onboarding - returns (first_name, email, process)
        result = await run_console_voice_onboarding(state_notify_fd=state_notify_fd)

        if not result:
            print("\nVoice onboarding failed. Exiting.")
            return (False, None)

        first_name, email, agent_process = result

        # Note: User is already created by voice_agent
        # We just need to retrieve the user from the database and save the session
        # The first_name from onboarding is the user's first name
        # We need to get the actual username from the database
        try:
            from db import get_db
            from db.models import User

            db = next(get_db())
            try:
                user = db.query(User).filter(User.email == email).first()

                if not user:
                    print("Failed to retrieve user after onboarding")
                    return (False, agent_process)

                user_id = str(user.id)
                actual_username = user.username
            finally:
                db.close()

        except Exception as e:
            print(f"Error retrieving user: {e}")
            return (False, agent_process)

        # Save session
        if self.session_manager.save_session(user_id, actual_username, email):
            self.current_user = {
                'user_id': user_id,
                'username': actual_username,
                'email': email
            }
            print(f"\n✓ Onboarding complete! Agent continuing in main menu mode...")
            return (True, agent_process)

        return (False, agent_process)


    async def run(self):
        """Main application loop with voice agent coordination"""
        record_session = os.environ.get("NOWVA_RECORD_SESSION", "").lower() == "true"

        if record_session or self.test_assess_mode:
            from datetime import datetime as _dt
            session_id = _dt.now().strftime("%Y-%m-%d_%H-%M-%S")
            self._session_dir = Path("user_test_runs") / session_id
            self._session_dir.mkdir(parents=True, exist_ok=True)
            os.environ["NOWVA_SESSION_OUTPUT_DIR"] = str(self._session_dir)
            self.session_logger.start_session(log_dir=str(self._session_dir))
            if record_session:
                await self._start_screen_recording(self._session_dir)
        else:
            self.session_logger.start_session()

        self.session_logger.log_system_event("app_started")

        # Start profiler if enabled
        self._profiler = SessionProfiler.get_instance()
        self._profiler.start()

        self._voice_agent_process = None
        self._state_pipe_r: int | None = None

        try:
            await self._run_inner()
        finally:
            await self._cleanup()

    async def _run_inner(self):
        """Core application logic — always wrapped by run()'s try/finally."""
        print("\n" + "="*60)
        print("NOWVA - AI-Powered Smart Squat Rack")
        print("="*60)

        # Open the persistent display first — boot progress renders there
        self._start_display()
        self._boot_step("Initializing neural substrate", 0.04)

        # Start FastAPI backend server
        print("\nStarting FastAPI backend...")
        self._boot_step("Establishing secure data link", 0.10)
        self._start_fastapi_server()

        # Initialize database (skips if tables already exist)
        self._boot_step("Mounting athlete memory core", 0.18)
        init_db()

        # Check for existing session
        if self.test_assess_mode and not self.check_session():
            print("[TEST ASSESS] No existing user session found.")
            print("[TEST ASSESS] Run 'python src/main.py' once to complete onboarding, then retry.")
            return

        if self.check_session():
            self.current_user = self.load_user_from_session()
            print("\n" + "="*50)
            print("RETURNING USER")
            print("="*50)
            print(f"\nWelcome back, {self.current_user['username']}!")
            print("\nStarting voice agent in main menu mode...\n")

            # Load existing user's state
            from agent.core.agent_state import AgentState
            self.state = AgentState(user_id=self.current_user['user_id'])

            # ALWAYS reset to main_menu mode on startup for safety
            # (prevents "ready to squat" if app crashed during workout)
            current_mode = self.state.get_mode()
            print(f"[STATE] Previous mode was '{current_mode}' - resetting to main_menu for safety")
            self.state.switch_mode("main_menu")
            self.state.set("workout.active", False)
            self.state.set("workout.greeting_done", False)
            self.state.set("shutdown_requested", False)
            # A session killed mid-calibration leaves these armed; a stale
            # calibration.active would force the next pose launch into
            # assessment mode even when the workout flow chose otherwise
            self.state.set("calibration.active", None)
            self.state.set("calibration.movement_pattern", None)
            self.state.set("calibration.pending_workout", None)
            self.state.save_state()

            if self.test_assess_mode:
                self._seed_test_assessment_state()

            # Small delay to ensure state file is written before voice agent loads it
            await asyncio.sleep(0.5)

            # Create notification pipe so voice agent can signal state changes
            self._state_pipe_r, state_pipe_w = os.pipe()
            os.set_blocking(self._state_pipe_r, False)

            # Start voice agent for returning user
            self._boot_step("Identity verified — waking conversational cortex", 0.26)
            from agent.agents.console_launcher import run_console_voice_agent
            try:
                self._voice_agent_process = await run_console_voice_agent(
                    user_id=self.current_user['user_id'],
                    state_notify_fd=state_pipe_w,
                )
            finally:
                # Always close our copy of the write end — a leaked fd would
                # keep the read end from ever seeing EOF
                os.close(state_pipe_w)

        else:
            # Create notification pipe before onboarding subprocess launches
            self._state_pipe_r, state_pipe_w = os.pipe()
            os.set_blocking(self._state_pipe_r, False)

            # Run onboarding - returns (success, agent_process)
            self._boot_step("Waking conversational cortex", 0.26)
            try:
                success, voice_agent_process = await self.run_onboarding(state_notify_fd=state_pipe_w)
                self._voice_agent_process = voice_agent_process
            finally:
                os.close(state_pipe_w)

            if not success:
                print("Onboarding failed. Exiting.")
                return

            # Load state for new user
            from agent.core.agent_state import AgentState
            self.state = AgentState(user_id=self.current_user['user_id'])

        if not self._voice_agent_process:
            print("Error: Voice agent failed to start. Exiting.")
            return

        # Bind coaching IPC server immediately so the socket is ready
        # before WorkoutAgent.on_enter() tries to connect. The server
        # just idles until a client connects — zero overhead until then.
        self._start_coaching_ipc()

        print("\n" + "="*50)
        print("SYSTEM READY")
        print("="*50)
        print("\nVoice agent is running and listening...")
        print("Speak to Nova to interact!")
        print("\nMonitoring for state changes...")
        print("Press Ctrl+C to exit.\n")

        # Main monitoring loop - watches state and controls pose estimation
        # Synchronously monitors voice agent output on main thread
        pose_running = False
        last_mode = self.state.get_mode()

        # Set stdout to line-buffered mode for immediate output
        sys.stdout.flush()

        # Pump agent stdout from a dedicated thread. select()+readline() on a
        # buffered stream hides lines the buffer already holds, so a crashed
        # agent's traceback would sit invisible until new output arrived.
        def _pump_agent_output():
            for line in iter(self._voice_agent_process.stdout.readline, ''):
                print(line, end='', flush=True)
                self._scan_agent_boot_markers(line)

        agent_output_thread = threading.Thread(target=_pump_agent_output, daemon=True)
        agent_output_thread.start()

        try:
            while True:
                # Check if voice agent is still running
                if self._voice_agent_process.poll() is not None:
                    # Let the pump thread finish printing the agent's dying traceback
                    agent_output_thread.join(timeout=2.0)
                    print(f"\n[SYSTEM] Voice agent terminated (exit code {self._voice_agent_process.returncode})")
                    break

                # Monitor state notification pipe
                state_notified = False
                try:
                    if self._state_pipe_r is not None:
                        ready, _, _ = select.select([self._state_pipe_r], [], [], 0.05)
                        if ready:
                            try:
                                os.read(self._state_pipe_r, 1024)
                            except BlockingIOError:
                                pass
                            state_notified = True
                except (OSError, ValueError):
                    # Expected when an fd closes mid-select during shutdown.
                    # Anything else is a real bug — let it propagate instead
                    # of silently hanging the app.
                    pass

                if state_notified:
                    self.state.reload_state()
                current_mode = self.state.get_mode()

                # Check for graceful shutdown request from voice agent
                if self.state.get("shutdown_requested", False):
                    print("\n[SYSTEM] Shutdown requested by user via voice agent")
                    self.session_logger.log_system_event("shutdown_requested")
                    # Wait for goodbye speech to finish playing
                    await asyncio.sleep(5)
                    break

                # Detect mode changes
                if current_mode != last_mode:
                    print(f"\n[STATE CHANGE] {last_mode} → {current_mode}")
                    self.session_logger.log_system_event("mode_change", {
                        "from_mode": last_mode,
                        "to_mode": current_mode
                    })
                    self._profiler.record("agent", "mode_change", from_mode=last_mode, to_mode=current_mode)
                    last_mode = current_mode

                # Handle workout mode — only start if workout session is fully configured
                # (workout.current_session is set by confirm_quick_exercise/start_workout)
                has_workout_session = self.state.get("workout.current_session") is not None
                if current_mode == "workout" and not pose_running and has_workout_session:
                    # Coaching IPC is stopped when a workout ends — rebind
                    # so a second workout in the same session reconnects.
                    if not self.coaching_ipc:
                        self._start_coaching_ipc()

                    # Start IPC server for pose estimation communication
                    if not self.ipc_server:
                        print("[IPC] Starting IPC server...")

                        def ipc_message_handler(message: dict, raw_bytes: bytes):
                            """Handle messages from pose estimation / biomechanics pipeline.

                            Receives both the parsed dict and the raw wire bytes so
                            frame_data can be forwarded without re-serialization.
                            """
                            msg_type = message.get('type')

                            # --- New biomechanics message types ---
                            if msg_type == 'rep_complete':
                                rep_num = message.get('rep_number')
                                depth = message.get('depth_category', '')
                                print(f"[BIOMECH] Rep {rep_num} complete — {depth}")
                            elif msg_type == 'fault':
                                print(f"[BIOMECH] Fault: {message.get('fault_type')} ({message.get('severity')})")
                            elif msg_type == 'set_complete':
                                print(f"[BIOMECH] Set {message.get('set_number')} complete — {message.get('total_reps')} reps")
                            elif msg_type == 'cache_cues':
                                cues = message.get('cues', {})
                                print(f"[BIOMECH] Caching {len(cues)} audio cues for {message.get('exercise_name')}")
                            elif msg_type == 'play_cue':
                                print(f"[BIOMECH] Play cue: {message.get('cue')}")
                            elif msg_type == 'rest_complete':
                                print("[BIOMECH] Rest timer expired")
                            elif msg_type == 'calibration_rep':
                                rep = message.get('rep_number', 0)
                                total = message.get('total_required', 5)
                                print(f"[BIOMECH] Calibration rep {rep}/{total}")
                            elif msg_type == 'calibration_complete':
                                print(f"[BIOMECH] Calibration complete for {message.get('movement_pattern')}")
                                self._publish_workout_start()
                            elif msg_type == 'diagnosis_complete':
                                print(f"[BIOMECH] Diagnosis complete — confidence={message.get('diagnosis', {}).get('confidence', 0):.2f}")
                            elif msg_type == 'assessment_rep':
                                rep = message.get('rep_number', 0)
                                total = message.get('total_required', 2)
                                print(f"[BIOMECH] Assessment rep {rep}/{total} (round {message.get('round', 1)})")
                            elif msg_type == 'assessment_result':
                                passed = message.get('passed', False)
                                print(f"[BIOMECH] Assessment result: {'PASSED' if passed else 'NEEDS CORRECTION'} (round {message.get('round', 1)})")
                                if passed:
                                    self._publish_workout_start()
                            elif msg_type == 'pipeline_status':
                                print(f"[BIOMECH] Pipeline: {message.get('status')}")
                            # --- Legacy / backward-compatible types ---
                            elif msg_type == 'rep_count':
                                value = message.get('value')
                                print(f"[IPC] Rep count: {value}")
                            elif msg_type == 'feedback':
                                value = message.get('value')
                                print(f"[IPC] Form feedback: {value}")
                            elif msg_type == 'status':
                                value = message.get('value')
                                print(f"[IPC] Status: {value}")
                            elif msg_type == 'error':
                                value = message.get('value')
                                print(f"[IPC] Error: {value}")

                            # --- Display page events (best-effort) ---
                            if msg_type == 'rep_complete':
                                self._publish_display({
                                    "type": "rep",
                                    "rep_number": message.get("rep_number"),
                                    "set_number": message.get("set_number"),
                                    "depth_class_name": message.get("depth_class_name")
                                        or message.get("depth_category", ""),
                                    "is_clean": message.get("is_clean", True),
                                    "faults": message.get("faults_in_rep", []),
                                })
                            elif msg_type == 'shallow_rep':
                                self._publish_display({
                                    "type": "shallow_rep",
                                    "depth_class_name": message.get("depth_class_name", ""),
                                })
                            elif msg_type == 'set_complete':
                                self._publish_display({
                                    "type": "set_summary",
                                    "set_number": message.get("set_number"),
                                    "total_reps": message.get("total_reps"),
                                    "clean_reps": message.get("clean_reps"),
                                    "avg_depth": message.get("avg_depth"),
                                    "depth_consistency": message.get("depth_consistency"),
                                    "fault_summary": message.get("fault_summary", {}),
                                })
                            elif msg_type == 'diagnosis_complete':
                                scoring = message.get("scoring") or {}
                                self._publish_display({
                                    "type": "set_scores",
                                    "set_number": message.get("set_number"),
                                    "mean_score": scoring.get("mean_score"),
                                    "per_dimension": scoring.get("per_dimension", {}),
                                    "best_rep": scoring.get("best_rep"),
                                    "worst_rep": scoring.get("worst_rep"),
                                })
                            elif msg_type == 'rest_complete':
                                self._publish_display({"type": "rest", "action": "end"})

                            # Record IPC messages for profiler
                            if msg_type and msg_type != 'frame_data':
                                self._profiler.record("ipc", msg_type, direction="pose_to_main")

                            # Forward coaching-relevant messages to voice agent.
                            # Snapshot the reference — this runs on the pose IPC
                            # thread while the main loop can nil self.coaching_ipc
                            if msg_type in ('cache_cues', 'fault', 'rep_complete', 'rest_complete', 'frame_data', 'calibration_rep', 'calibration_complete', 'diagnosis_complete', 'assessment_result', 'assessment_rep', 'demo_abort', 'demo_started', 'last_rep_snapshot', 'demo_data_ready'):
                                coaching = self.coaching_ipc
                                if coaching and coaching.client_socket:
                                    try:
                                        if msg_type == 'frame_data':
                                            coaching.send_raw_message(raw_bytes)
                                        else:
                                            coaching.send_message(message)
                                    except Exception as e:
                                        print(f"[COACHING IPC] Forward failed: {e}")
                                else:
                                    print(f"[COACHING IPC] No voice agent connected — dropping {msg_type}")

                        self.ipc_server = IPCServer()
                        self.ipc_server.bind(raw_message_callback=ipc_message_handler)
                        print("[IPC] Server ready")

                        def run_server(pose_ipc=self.ipc_server):
                            # Bound as default arg — the main loop can nil
                            # self.ipc_server while this thread is running
                            pose_ipc.accept_client()
                            pose_ipc.listen()

                        ipc_thread = threading.Thread(target=run_server, daemon=True)
                        ipc_thread.start()

                    # Preload pose model during greeting speech.
                    # Launch the subprocess with --preload as soon as IPC
                    # is bound so the model loads while the greeting plays.
                    if not self.pose_process and not getattr(self, 'simulate_mode', False):
                        exercise_name = self.state.get("workout.exercise_name", DEFAULT_EXERCISE_NAME)
                        if not exercise_name:
                            session_data = self.state.get("workout.current_session")
                            if session_data and session_data.get("exercises"):
                                exercise_name = session_data["exercises"][0].get("exercise_name", DEFAULT_EXERCISE_NAME)
                            else:
                                exercise_name = DEFAULT_EXERCISE_NAME

                        cal_mode = bool(self.state.get("calibration.active"))
                        cal_file = None
                        cal_profile = self.state.get("workout.calibration_profile")
                        if cal_profile and not cal_mode:
                            import json, tempfile
                            cal_file = os.path.join(tempfile.gettempdir(), f"nowva_cal_{id(self)}.json")
                            # Returning users skip the assessment/calibration
                            # phase, which is where body proportions are
                            # normally derived. Without them the pipeline has
                            # no shoulder width, so stance metrics and the
                            # whole diagnosis engine stay dark for the session.
                            payload = {"thresholds": cal_profile}
                            payload.update(self._load_athlete_calibration(exercise_name))
                            with open(cal_file, "w") as f:
                                json.dump(payload, f)
                            self._cal_file = cal_file
                            have_params = "athlete_params" in payload
                            print(
                                f"[CALIBRATION] Wrote calibration profile to {cal_file} "
                                f"(athlete_params={'yes' if have_params else 'MISSING'})"
                            )

                        self.start_pose_estimation(
                            exercise_name=exercise_name,
                            calibration_file=cal_file,
                            calibration_mode=cal_mode,
                            preload=True,
                        )
                        print("[PRELOAD] Pose subprocess launched — model loading during greeting")

                    # Wait for greeting to finish before telling subprocess
                    # to open camera (window popup can cancel TTS on macOS)
                    if not self.state.get("workout.greeting_done", False):
                        continue

                    print("\n" + "="*50)
                    print("STARTING WORKOUT SESSION")
                    print("="*50)

                    if getattr(self, 'simulate_mode', False):
                        print("[SIMULATE] Skipping real pose estimation — use simulate_squat_workout.py")
                    else:
                        # Signal the preloaded subprocess to open camera
                        if self.ipc_server and self.ipc_server.client_socket:
                            try:
                                self.ipc_server.send_message({"type": "start_capture"})
                                print("[PRELOAD] Sent start_capture — camera opening")
                            except Exception as e:
                                print(f"[PRELOAD] Failed to send start_capture: {e}")

                    pose_running = True
                    # In calibration/assessment mode the working sets start
                    # later — the HUD is armed on calibration_complete /
                    # assessment_result instead of here.
                    if not self.state.get("calibration.active"):
                        self._publish_workout_start()
                    print("[POSE] Pose estimation started" if not getattr(self, 'simulate_mode', False) else "[SIMULATE] IPC servers ready — waiting for simulator")

                elif current_mode != "workout" and pose_running:
                    print("\n" + "="*50)
                    print("ENDING WORKOUT SESSION")
                    print("="*50)
                    self._publish_display({"type": "workout", "action": "end"})
                    self._stop_pose_process()
                    if self.coaching_ipc:
                        self.coaching_ipc.stop()
                        self.coaching_ipc = None
                        print("[COACHING IPC] Stopped")
                    if self.ipc_server:
                        self.ipc_server.stop()
                        self.ipc_server = None
                        print("[IPC] Server stopped")
                    pose_running = False
                    print("[POSE] Pose estimation stopped")

                if state_notified:
                    continue
                # Nothing happened — long fallback poll for safety
                await asyncio.sleep(2.0)
                self.state.reload_state()

        except KeyboardInterrupt:
            print("\n\n" + "="*50)
            print("SHUTTING DOWN")
            print("="*50)

    async def _cleanup(self):
        """Cleanup that runs on ANY exit path — normal, early return, or Ctrl+C."""
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)

        print("\nCleaning up...")

        if self._state_pipe_r is not None:
            try:
                os.close(self._state_pipe_r)
            except OSError:
                pass
            self._state_pipe_r = None

        if self.state:
            print("Resetting state to main_menu...")
            self.state.switch_mode("main_menu")
            self.state.set("workout.active", False)
            self.state.set("shutdown_requested", False)
            self.state.save_state()

        if self._voice_agent_process:
            print("Stopping voice agent...")
            terminate_process_group(self._voice_agent_process)

        if self.pose_process:
            print("Stopping pose estimation...")
            self._stop_pose_process()

        if self.coaching_ipc:
            print("Stopping coaching IPC server...")
            self.coaching_ipc.stop()

        if self.fastapi_process:
            print("Stopping FastAPI server...")
            self.fastapi_process.terminate()
            self.fastapi_process.wait()
        if self._fastapi_log:
            self._fastapi_log.close()
            self._fastapi_log = None

        if self._cal_file:
            try:
                os.unlink(self._cal_file)
            except OSError:
                pass
            self._cal_file = None

        if self.ipc_server:
            print("Stopping IPC server...")
            self.ipc_server.stop()

        if self.display_server:
            self.display_server.stop()
            self.display_server = None

        self._stop_screen_recording()

        # End session and generate summary
        self.session_logger.log_system_event("app_shutdown")
        summary = self.session_logger.end_session()
        print("\n" + summary)
        print(f"\nSession log saved to: {self.session_logger.get_log_path()}")

        # Generate profiler report if enabled
        if getattr(self, 'profile_mode', False):
            print("\nGenerating session profile report...")
            try:
                self._profiler.stop()
                from profiler.report import generate_profile_report
                from profiler.collector import PROFILE_OUTPUT_DIR, AGENT_JSON_FILENAME
                agent_json = PROFILE_OUTPUT_DIR / AGENT_JSON_FILENAME
                for _ in range(50):
                    if agent_json.exists():
                        break
                    await asyncio.sleep(0.1)
                profiler_out = self._session_dir / "profiler_results" if self._session_dir else PROFILE_OUTPUT_DIR
                profiler_out.mkdir(parents=True, exist_ok=True)
                report_path = generate_profile_report(
                    agent_json_path=agent_json if agent_json.exists() else None,
                    main_profiler=self._profiler,
                    output_dir=profiler_out,
                )
                print(f"Profile report saved to: {report_path}")
            except Exception as e:
                print(f"Failed to generate profile report: {e}")

        print("\nGoodbye!")


def _check_python_environment():
    try:
        from livekit.agents import TurnHandlingOptions  # noqa: F401
    except ImportError:
        import importlib.metadata
        version = importlib.metadata.version("livekit-agents")
        print("ERROR: livekit-agents in this Python environment is too old for the voice agent.")
        print(f"  Interpreter: {sys.executable}")
        print(f"  livekit-agents: {version} (requirements.txt needs >=1.5.1)")
        print("  Run the app from the project venv instead:")
        print("    source venv/bin/activate && python src/main.py")
        sys.exit(1)


async def main():
    """Entry point"""
    _check_python_environment()

    import argparse
    parser = argparse.ArgumentParser(description="Nowva Main Application")
    parser.add_argument("--simulate", action="store_true",
                        help="Skip real pose estimation (use simulate_squat_workout.py instead)")
    parser.add_argument("--profile", action="store_true",
                        help="Enable live session profiling (writes HTML report on exit)")
    parser.add_argument("--test_assess", action="store_true",
                        help="Dev fast path: boot straight into the pre-workout form assessment")
    parser.add_argument("--valgus", action="store_true",
                        help="Record a knee-valgus debug session (preview video, per-frame "
                             "knee metrics, faults, IPC traffic) and write an HTML report")
    parser.add_argument("--inspect", action="store_true",
                        help="Record all pipeline stages (raw vs filtered skeletons, "
                             "joint angles) and generate a scrubbable HTML debug viewer")
    parser.add_argument("--user", type=str, default=None,
                        help="Switch to a saved user profile before launch. "
                             "Use 'new' to save current session and start fresh onboarding. "
                             "Use 'list' to show saved profiles and exit.")
    args = parser.parse_args()

    if args.profile:
        os.environ["NOWVA_PROFILE"] = "1"
    if args.test_assess:
        os.environ["NOWVA_TEST_ASSESS"] = "1"
    if args.valgus:
        os.environ["NOWVA_VALGUS_DEBUG"] = "1"
    if args.inspect:
        os.environ["NOWVA_PIPELINE_INSPECT"] = "1"

    # ── User profile switching ───────────────────────────────────
    if args.user is not None:
        sm = SessionManager()
        if args.user == "list":
            profiles = sm.list_profiles()
            active = sm.load_session()
            print("\n  Saved profiles:")
            for p in profiles:
                print(f"    • {p['name']}  ({p['username']} / {p['email']})")
            if not profiles:
                print("    (none)")
            if active:
                print(f"\n  Active session: {active['username']} ({active['email']})")
            else:
                print("\n  Active session: none (next launch triggers onboarding)")
            return
        elif args.user == "new":
            if sm.session_exists():
                session = sm.load_session()
                if session:
                    name = session["username"]
                    sm.save_profile(name)
                    print(f"Current session saved as profile '{name}'.")
                sm.clear_session()
            print("Session cleared. Next launch will start onboarding for a new user.")
            return
        else:
            if sm.session_exists():
                session = sm.load_session()
                if session:
                    sm.save_profile(session["username"])
            if sm.load_profile(args.user):
                print(f"Profile '{args.user}' loaded. Launching...")
            else:
                available = [p["name"] for p in sm.list_profiles()]
                print(f"Available profiles: {', '.join(available) if available else '(none)'}")
                return

    app = NowvaApp()
    app.simulate_mode = args.simulate
    app.profile_mode = args.profile
    app.test_assess_mode = args.test_assess
    await app.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nApplication interrupted by user. Exiting.")
