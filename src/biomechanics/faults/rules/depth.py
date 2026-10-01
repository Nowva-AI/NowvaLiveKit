"""Depth rule: a squat judged against the athlete's own depth target, by hip height.

Depth is the hip joint's height above the knee joint in femur lengths (0 =
parallel, > 0 = above). Knee angle cannot define depth: the same hip height
reads ~25° apart at different shin angles, which is how 90° of knee flexion was
once taken for parallel while the hip sat ~20 cm above the knee.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional

from biomechanics.faults.fault_types import FAULT_MESSAGES, FaultRule, FaultType
from biomechanics.utils.types import FaultEvent, FaultSeverity, JointAngles

PARALLEL_RATIO = 0.0
# Within this of parallel a rep counts at a parallel target, so it is labelled
# parallel too — matches the default DepthFaultConfig.tolerance_ratio.
PARALLEL_LABEL_TOLERANCE_RATIO = 0.08
# Hip this far below the knee (femur lengths) is clearly below parallel.
BELOW_PARALLEL_MARGIN_RATIO = 0.05
# Thigh 30° above horizontal — shallower than this is a quarter squat.
HALF_SQUAT_RATIO = 0.5


class DepthCategory:
    """Depth category constants."""
    QUARTER = "quarter"
    HALF = "half"
    PARALLEL = "parallel"
    BELOW_PARALLEL = "below_parallel"


def depth_category(depth_ratio: float) -> str:
    if depth_ratio < PARALLEL_RATIO - BELOW_PARALLEL_MARGIN_RATIO:
        return DepthCategory.BELOW_PARALLEL
    if depth_ratio <= PARALLEL_RATIO + PARALLEL_LABEL_TOLERANCE_RATIO:
        return DepthCategory.PARALLEL
    if depth_ratio <= HALF_SQUAT_RATIO:
        return DepthCategory.HALF
    return DepthCategory.QUARTER


class DepthRule(FaultRule):
    """A descent that misses the athlete's depth target is not counted, and this
    rule says why. Counted reps met the target by definition, so the rule has no
    per-frame or per-rep verdict of its own — only ``judge_shallow_descent``.
    """

    def __init__(
        self,
        tolerance_ratio: float = 0.08,
        moderate_deficit_ratio: float = 0.25,
        severe_deficit_ratio: float = 0.5,
    ) -> None:
        self.tolerance_ratio = tolerance_ratio
        self.moderate_deficit_ratio = moderate_deficit_ratio
        self.severe_deficit_ratio = severe_deficit_ratio

    @property
    def fault_type(self) -> FaultType:
        return FaultType.DEPTH

    def evaluate(
        self,
        angles: JointAngles,
        history: deque,
        in_rep: bool = False,
        rep_number: int = 0,
    ) -> Optional[FaultEvent]:
        return None

    def judge_shallow_descent(
        self,
        depth_ratio: float,
        target_ratio: float,
        angles: JointAngles,
        rep_number: int,
        depth_cm_above_parallel: float = math.nan,
    ) -> Optional[FaultEvent]:
        """The fault for a descent that stopped short of the target. NaN depth → None."""
        if math.isnan(depth_ratio):
            return None
        deficit = depth_ratio - target_ratio
        if deficit <= self.tolerance_ratio:
            return None
        severity, score = self._get_severity(
            deficit,
            {
                "mild": self.tolerance_ratio,
                "moderate": self.moderate_deficit_ratio,
                "severe": self.severe_deficit_ratio,
            },
        )
        if severity == FaultSeverity.NONE:
            return None
        return self._create_fault_event(
            severity=severity,
            severity_score=score,
            message=FAULT_MESSAGES[FaultType.DEPTH][severity.value],
            angles=angles,
            rep_number=rep_number,
            details={
                "side": None,
                "phase": "bottom",
                "is_drift": False,
                "value": deficit,
                "unit": "ratio",
                "depth_ratio": depth_ratio,
                "target_ratio": target_ratio,
                "depth_cm_above_parallel": depth_cm_above_parallel,
                "category": depth_category(depth_ratio),
                "shallow_rep": True,
            },
        )
