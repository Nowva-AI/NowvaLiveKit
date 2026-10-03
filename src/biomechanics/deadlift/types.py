"""Data shapes shared by the deadlift analyser, its fault rules, the bar tracker
and the delivery contract (.claude/deadlift/CONTRACT.md). World positions are
metres in the Y-down world frame; heights and offsets are cm in the deadlift's
sagittal frame (up = measured gravity, forward = heel to toe).
"""

from __future__ import annotations

import math
from enum import Enum

import numpy as np
from pydantic import BaseModel

NAN = float("nan")

# Schema version of DeadliftRepFeatures as stored with every rep (DB, IPC).
DEADLIFT_FEATURES_SCHEMA = 1

BAR_SOURCE_BAR = "bar"
BAR_SOURCE_WRIST_PROXY = "wrist_proxy"
GRAVITY_SOURCE_MEASURED = "measured"
GRAVITY_SOURCE_BODY = "body"


class DeadliftPhase(str, Enum):
    """Where the lifter is in the deadlift cycle (PLAN.md §2.3)."""
    APPROACH = "approach"
    STANCE = "stance"
    SETUP = "setup"
    PULL = "pull"
    TOP = "top"
    LOWER = "lower"
    FLOOR = "floor"


IN_REP_PHASES = frozenset({DeadliftPhase.PULL, DeadliftPhase.TOP, DeadliftPhase.LOWER})


class BarState3D(BaseModel):
    """The barbell in the world frame at one capture time.

    left_end_m / right_end_m are the plate-hub centres on the subject's left
    (+X) and right sides. predicted marks a Kalman prediction on a frame with no
    detection; views is how many cameras supported the measurement.
    """
    timestamp: float
    left_end_m: tuple[float, float, float]
    right_end_m: tuple[float, float, float]
    velocity_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    predicted: bool = False
    views: int = 0
    residual_px: float = NAN

    @property
    def centre(self) -> np.ndarray:
        return (np.asarray(self.left_end_m) + np.asarray(self.right_end_m)) / 2.0

    @property
    def axis(self) -> np.ndarray:
        """Unit vector from the right end to the left end."""
        span = np.asarray(self.left_end_m) - np.asarray(self.right_end_m)
        norm = float(np.linalg.norm(span))
        return span / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0])


class DeadliftRepFeatures(BaseModel):
    """Everything a coach reads off one deadlift rep (PLAN.md §2.5).

    NaN means "not measured this rep", never 0. Offsets are cm in the sagittal
    frame: bar_midfoot_* > 0 means the bar sits ahead of (toes side of) the
    midfoot; shoulder_vs_bar_cm > 0 means the shoulders are ahead of the bar;
    bar_drift_cm is the forward drift away from the legs; hip_shift_ratio > 0
    means the hips moved toward the right foot; lean_back_deg > 0 means the trunk
    finished behind its standing angle.
    """
    rep_number: int
    sample_count: int = 0
    touch_and_go: bool = False
    setup_measured: bool = False
    bar_midfoot_stance_cm: float = NAN
    bar_midfoot_setup_cm: float = NAN
    setup_hip_height_cm: float = NAN
    setup_hip_band_low_cm: float = NAN
    setup_hip_band_high_cm: float = NAN
    shoulder_vs_bar_cm: float = NAN
    setup_trunk_deg: float = NAN
    trunk_change_liftoff_knee_deg: float = NAN
    trunk_change_predicted_deg: float = NAN
    hip_shoulder_rise_ratio: float = NAN
    bar_drift_cm: float = NAN
    hip_extension_deficit_deg: float = NAN
    knee_extension_deficit_deg: float = NAN
    lean_back_deg: float = NAN
    hip_shift_ratio: float = NAN
    bar_tilt_cm: float = NAN
    # "left" / "right": the end that sat lower; "" when not measured.
    bar_low_side: str = ""
    elbow_flexion_deg: float = NAN
    concentric_velocity_mps: float = NAN
    velocity_loss_pct: float = NAN
    bar_rise_cm: float = NAN
    pull_time_s: float = NAN
    lower_time_s: float = NAN
    liftoff_time: float = NAN
    knee_pass_time: float = NAN
    top_time: float = NAN
    floor_time: float = NAN
    bar_source: str = BAR_SOURCE_BAR
    gravity_source: str = GRAVITY_SOURCE_MEASURED
    dl_schema: int = DEADLIFT_FEATURES_SCHEMA

    @property
    def hip_shift_abs(self) -> float:
        return abs(self.hip_shift_ratio) if math.isfinite(self.hip_shift_ratio) else NAN


class DeadliftFrameStatus(BaseModel):
    """What the deadlift analyser saw on one frame, for the voice agent's
    closed-loop foot guidance and the display."""
    phase: DeadliftPhase
    bar_midfoot_live_cm: float = NAN
    bar_source: str = BAR_SOURCE_BAR
    gravity_source: str = GRAVITY_SOURCE_MEASURED
    bar_height_cm: float = NAN
