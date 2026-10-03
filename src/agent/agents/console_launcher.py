"""
Console Voice Onboarding
Voice-based onboarding that runs directly in the terminal (no browser)
"""

import asyncio
import os
import signal
import subprocess
import sys
import time
import threading
from typing import Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from livekit.agents import AgentSession, Agent, llm
from livekit.plugins import openai


def _find_physical_output_device() -> Optional[str]:
    """Return the device ID of a physical speaker if the default output is BlackHole.

    When BlackHole is the macOS system output (e.g. for screen recording), the
    LiveKit console would send all TTS audio there — inaudible. This detects
    that case and returns the first non-virtual output device so we can pass
    --output-device to the voice agent subprocess.

    Returns None if the default output is already a physical device.
    """
    try:
        import sounddevice as sd
    except ImportError:
        return None

    try:
        _, default_output = sd.default.device
        default_info = sd.query_devices(default_output, kind="output")
        if "blackhole" not in default_info["name"].lower():
            return None

        for idx, dev in enumerate(sd.query_devices()):
            if dev["max_output_channels"] <= 0:
                continue
            if idx == default_output:
                continue
            if "blackhole" in dev["name"].lower():
                continue
            print(f"[AUDIO] Default output is BlackHole — using '{dev['name']}' instead")
            return str(idx)

    except Exception:
        pass
    return None


def _build_subprocess_env(state_notify_fd: Optional[int] = None) -> dict:
    env = os.environ.copy()
    if state_notify_fd is not None:
        env["NOWVA_STATE_NOTIFY_FD"] = str(state_notify_fd)
    from agent.core.session_logger import SessionLogger
    sl = SessionLogger.get_instance()
    if sl.log_file_path:
        env["NOWVA_SESSION_LOG"] = str(sl.log_file_path)
    return env


def terminate_process_group(process, timeout: float = 20):
    """Kill a process and all its children by sending SIGTERM to the process group.

    LiveKit's graceful shutdown sequence (session.aclose → session_end →
    OTel flush → shutdown callbacks) needs time to run before SIGKILL.
    """
    try:
        pgid = os.getpgid(process.pid)
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, OSError):
        pass
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            pgid = os.getpgid(process.pid)
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass


async def run_console_voice_agent(
    user_id: Optional[str] = None,
    state_notify_fd: Optional[int] = None,
):
    """
    Run voice agent in console mode - monitors output synchronously on main thread

    Args:
        user_id: Optional user ID for existing users, None for new onboarding
        state_notify_fd: Write end of a pipe; the child writes a byte after each save_state()

    Returns:
        subprocess.Popen: The running agent process
    """
    print("\n" + "="*60)
    print("NOVA VOICE AGENT")
    print("="*60)
    print("\nStarting Nova voice agent in console mode...")
    print("You'll be able to speak with Nova directly via your microphone.")
    print("\nAgent will continue running. Press Ctrl+C in main.py to exit.\n")

    try:
        # Path to voice agent (same directory now)
        voice_agent_path = Path(__file__).parent / 'voice_agent.py'

        env = _build_subprocess_env(state_notify_fd)
        pass_fds: tuple = ()
        if state_notify_fd is not None:
            pass_fds = (state_notify_fd,)

        cmd = [sys.executable, str(voice_agent_path), 'console']
        physical_output = _find_physical_output_device()
        if physical_output is not None:
            cmd.extend(['--output-device', physical_output])

        # Run the voice agent in its own process group so we can kill all children
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid,
            env=env,
            pass_fds=pass_fds,
        )

        return process

    except Exception as e:
        print(f"\nError starting voice agent: {e}")
        import traceback
        traceback.print_exc()
        return None


async def run_console_voice_onboarding(
    state_notify_fd: Optional[int] = None,
) -> Optional[Tuple[str, str]]:
    """
    Run voice onboarding in console by launching the voice agent
    DEPRECATED: This is kept for backwards compatibility but now just monitors for completion

    Returns:
        Tuple of (first_name, email) if successful, None otherwise
    """
    print("\n" + "="*60)
    print("CONSOLE VOICE ONBOARDING")
    print("="*60)
    print("\nStarting voice agent in onboarding mode...")
    print("You'll be able to speak with Nova directly via your microphone.")
    print("\nPress Ctrl+C anytime to cancel and use text onboarding instead.\n")

    try:
        # Path to voice agent (same directory now)
        voice_agent_path = Path(__file__).parent / 'voice_agent.py'

        env = _build_subprocess_env(state_notify_fd)
        env["NOWVA_FRESH_ONBOARDING"] = "1"
        pass_fds: tuple = ()
        if state_notify_fd is not None:
            pass_fds = (state_notify_fd,)

        cmd = [sys.executable, str(voice_agent_path), 'console']
        physical_output = _find_physical_output_device()
        if physical_output is not None:
            cmd.extend(['--output-device', physical_output])

        # Run the voice agent in its own process group so we can kill all children
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid,
            env=env,
            pass_fds=pass_fds,
        )

        # Monitor output for onboarding completion markers
        first_name = None
        email = None
        onboarding_complete = False

        try:
            for line in iter(process.stdout.readline, ''):
                if not line:
                    break

                # Name and email markers are for this parser only: echoing them would put
                # personal data in the console log (NOWVA_LOG_CONSOLE tees it to disk).
                if 'ONBOARDING_FIRST_NAME:' in line:
                    first_name = line.split('ONBOARDING_FIRST_NAME:')[1].strip()
                    continue
                if 'ONBOARDING_EMAIL:' in line:
                    email = line.split('ONBOARDING_EMAIL:')[1].strip()
                    continue

                print(line, end='')

                if 'ONBOARDING_COMPLETE' in line:
                    # Data captured - wait for welcome message then return process handle
                    onboarding_complete = True
                    print("[WRAPPER] Onboarding data captured...")
                    print("[WRAPPER] Agent will continue running in main menu mode")
                    time.sleep(3)
                    break

        except KeyboardInterrupt:
            print("\n\nOnboarding cancelled by user")
            terminate_process_group(process)
            return None

        # Check if we got the data
        if first_name and email:
            # Return the data but DON'T terminate process - it continues to main menu
            return (first_name, email, process)  # Return process handle too

        return None

    except Exception as e:
        print(f"\nError during voice onboarding: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Test the console onboarding
    async def test():
        result = await run_console_voice_onboarding()
        if result:
            username, email, process = result
            print(f"\n✓ Onboarding successful!")
            print(f"  Username: {username}")
            print(f"  Email: {email}")
        else:
            print("\n✗ Onboarding failed")

    asyncio.run(test())
