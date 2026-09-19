"""
Tests for the user calibration upsert: a calibration that arrives without
body measurements must never erase the athlete params already stored.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from db.calibration_utils import save_user_calibration
from db.models import UserCalibration

USER_ID = uuid.uuid4()

STORED_ATHLETE_PARAMS = {
    "shoulder_width_m": 0.39,
    "femur_avg_m": 0.46,
    "torso_avg_m": 0.55,
    "hip_width_m": 0.27,
    "tibia_avg_m": 0.44,
    "foot_avg_m": 0.21,
}
STORED_BASELINE = {"peakDorsi": 33.0, "peakKneeFlex": 118.0}


class _FakeQuery:
    def __init__(self, row: UserCalibration | None) -> None:
        self._row = row

    def filter(self, *conditions: object) -> "_FakeQuery":
        return self

    def first(self) -> UserCalibration | None:
        return self._row


class _FakeSession:
    """Single-row stand-in for a SQLAlchemy session (no database needed)."""

    def __init__(self, row: UserCalibration | None = None) -> None:
        self.row = row
        self.commits = 0

    def query(self, model: type) -> _FakeQuery:
        return _FakeQuery(self.row)

    def add(self, row: UserCalibration) -> None:
        self.row = row

    def commit(self) -> None:
        self.commits += 1


def _stored_row() -> UserCalibration:
    return UserCalibration(
        user_id=USER_ID,
        movement_pattern="squat",
        peaks={"trunk_flexion": 40.0},
        thresholds={"knee_valgus": {"mild": 12.0}},
        calibration_reps=5,
        athlete_params=dict(STORED_ATHLETE_PARAMS),
        baseline=dict(STORED_BASELINE),
    )


class TestSaveUserCalibrationGuard:
    def test_missing_athlete_params_keep_stored_values(self) -> None:
        session = _FakeSession(_stored_row())

        save_user_calibration(
            db=session, user_id=USER_ID, movement_pattern="squat",
            peaks={"trunk_flexion": 45.0}, thresholds={"knee_valgus": {"mild": 13.0}},
            athlete_params=None, baseline=None,
        )

        assert session.row.athlete_params == STORED_ATHLETE_PARAMS
        assert session.row.baseline == STORED_BASELINE
        assert session.commits == 1

    def test_thresholds_still_update_when_athlete_params_missing(self) -> None:
        session = _FakeSession(_stored_row())
        new_thresholds = {"knee_valgus": {"mild": 13.0}}

        save_user_calibration(
            db=session, user_id=USER_ID, movement_pattern="squat",
            peaks={"trunk_flexion": 45.0}, thresholds=new_thresholds,
        )

        assert session.row.thresholds == new_thresholds
        assert session.row.peaks == {"trunk_flexion": 45.0}

    def test_new_athlete_params_replace_stored_values(self) -> None:
        session = _FakeSession(_stored_row())
        new_params = dict(STORED_ATHLETE_PARAMS, femur_avg_m=0.47)
        new_baseline = {"peakDorsi": 36.0, "peakKneeFlex": 121.0}

        save_user_calibration(
            db=session, user_id=USER_ID, movement_pattern="squat",
            peaks={}, thresholds={}, athlete_params=new_params, baseline=new_baseline,
        )

        assert session.row.athlete_params == new_params
        assert session.row.baseline == new_baseline

    def test_first_calibration_without_params_creates_row(self) -> None:
        session = _FakeSession(None)

        save_user_calibration(
            db=session, user_id=USER_ID, movement_pattern="squat",
            peaks={}, thresholds={"knee_valgus": {"mild": 12.0}},
        )

        assert session.row is not None
        assert session.row.athlete_params is None
        assert session.row.thresholds == {"knee_valgus": {"mild": 12.0}}
