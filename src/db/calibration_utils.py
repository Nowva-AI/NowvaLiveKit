"""
Database utilities for user biomechanical calibration data.
"""

import logging
from typing import Optional
from sqlalchemy.orm import Session

from db.models import UserCalibration

logger = logging.getLogger(__name__)


def get_user_calibration(db: Session, user_id, movement_pattern: str) -> Optional[dict]:
    """Load calibration thresholds from DB.

    Returns the thresholds dict if calibration exists, None otherwise.
    Body proportions live on the same row but are NOT returned here — use
    get_user_calibration_full when the diagnosis engine needs them.
    """
    row = (
        db.query(UserCalibration)
        .filter(
            UserCalibration.user_id == user_id,
            UserCalibration.movement_pattern == movement_pattern,
        )
        .first()
    )
    if row is None:
        return None

    return row.thresholds


def get_user_calibration_full(db: Session, user_id, movement_pattern: str) -> Optional[dict]:
    """Load full calibration data (peaks + thresholds) from DB.

    Returns dict with 'peaks' and 'thresholds' keys, or None.
    """
    row = (
        db.query(UserCalibration)
        .filter(
            UserCalibration.user_id == user_id,
            UserCalibration.movement_pattern == movement_pattern,
        )
        .first()
    )
    if row is None:
        return None

    return {
        "peaks": row.peaks,
        "thresholds": row.thresholds,
        "calibration_reps": row.calibration_reps,
        "athlete_params": row.athlete_params,
        "baseline": row.baseline,
    }


def save_user_calibration(
    db: Session,
    user_id,
    movement_pattern: str,
    peaks: dict,
    thresholds: dict,
    calibration_reps: int = 5,
    athlete_params: dict | None = None,
    baseline: dict | None = None,
) -> None:
    """Upsert calibration profile to DB.

    If a row for (user_id, movement_pattern) already exists, it is updated.
    athlete_params / baseline of None mean "not measured this time": stored
    values are kept, so a calibration that lost its body measurements can't
    leave a returning user without diagnosis.
    """
    row = (
        db.query(UserCalibration)
        .filter(
            UserCalibration.user_id == user_id,
            UserCalibration.movement_pattern == movement_pattern,
        )
        .first()
    )

    if row:
        row.peaks = peaks
        row.thresholds = thresholds
        row.calibration_reps = calibration_reps
        if athlete_params is not None:
            row.athlete_params = athlete_params
        elif row.athlete_params is not None:
            logger.warning(
                f"[CALIBRATION] No athlete_params in new calibration for user={user_id} "
                f"pattern={movement_pattern} — keeping stored body measurements"
            )
        if baseline is not None:
            row.baseline = baseline
        logger.info(
            f"[CALIBRATION] Updated calibration for user={user_id} pattern={movement_pattern}"
        )
    else:
        row = UserCalibration(
            user_id=user_id,
            movement_pattern=movement_pattern,
            peaks=peaks,
            thresholds=thresholds,
            calibration_reps=calibration_reps,
            athlete_params=athlete_params,
            baseline=baseline,
        )
        db.add(row)
        logger.info(
            f"[CALIBRATION] Created calibration for user={user_id} pattern={movement_pattern}"
        )

    db.commit()
