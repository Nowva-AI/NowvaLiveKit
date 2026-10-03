"""Tests that the session CSV never stores the athlete's identity or medical history."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.core.session_logger import SessionLogger


class TestFunctionCallRedaction:
    def test_program_params_redacted_in_csv(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        session_logger = SessionLogger.get_instance()
        csv_path = tmp_path / "session_test.csv"
        monkeypatch.setattr(session_logger, "log_file_path", csv_path)

        session_logger.log_function_call(
            function_name="generate_workout_program",
            parameters={
                "name": "Sam",
                "email": "sam@example.com",
                "injury_history": "torn ACL in 2022",
                "days_per_week": 4,
            },
            result={"job_id": "job-1"},
        )

        written = csv_path.read_text()
        assert "sam@example.com" not in written
        assert "torn ACL" not in written
        assert "Sam" not in written
        assert "days_per_week" in written
