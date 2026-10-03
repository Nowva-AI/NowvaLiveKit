"""Tests for the console onboarding launcher: it must not echo the user's name or email."""

from __future__ import annotations

import asyncio
import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import agent.agents.console_launcher as console_launcher

AGENT_OUTPUT = (
    "[NOVA] Entrypoint function called\n"
    "ONBOARDING_USER_ID: user-42\n"
    "ONBOARDING_FIRST_NAME: Sam\n"
    "ONBOARDING_EMAIL: sam@example.com\n"
    "ONBOARDING_COMPLETE\n"
)


class _FakeProcess:
    def __init__(self, *args, **kwargs) -> None:
        self.stdout = io.StringIO(AGENT_OUTPUT)


class TestOnboardingMarkers:
    def test_markers_parsed_but_not_echoed(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        monkeypatch.setattr(console_launcher.subprocess, "Popen", _FakeProcess)
        monkeypatch.setattr(console_launcher, "_find_physical_output_device", lambda: None)
        monkeypatch.setattr(console_launcher.time, "sleep", lambda seconds: None)

        first_name, email, _ = asyncio.run(console_launcher.run_console_voice_onboarding())

        output = capsys.readouterr().out
        assert (first_name, email) == ("Sam", "sam@example.com")
        assert "sam@example.com" not in output
        assert "ONBOARDING_FIRST_NAME" not in output
        assert "Entrypoint function called" in output
