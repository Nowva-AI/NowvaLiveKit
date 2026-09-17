"""
Fault Detection Types and Base Classes

Defines the FaultType enum, default thresholds, and abstract FaultRule base class
for implementing individual fault detection rules.
"""

from abc import ABC, abstractmethod
from collections import deque
from enum import Enum
from typing import Optional, Dict, Any

from biomechanics.utils.types import JointAngles, FaultEvent, FaultSeverity


class FaultType(str, Enum):
    """Enumeration of all detectable form faults."""
    DEPTH = "depth"
    RANGE_OF_MOTION = "range_of_motion"
    BILATERAL_ASYMMETRY = "bilateral_asymmetry"
    FORWARD_LEAN = "forward_lean"
    KNEE_VALGUS = "knee_valgus"
    BACK_ROUNDING = "back_rounding"
    LOCKOUT = "lockout"
    ELBOW_FLARE = "elbow_flare"
    BAR_PATH = "bar_path"
    SHOULDER_STABILITY = "shoulder_stability"
    TRUNK_STABILITY = "trunk_stability"
    TEMPO = "tempo"  # Tempo-related faults (too fast, stalling)
    HEEL_RISE = "heel_rise"


# Default thresholds from config (degrees unless specified)
DEFAULT_THRESHOLDS: Dict[FaultType, Dict[str, float]] = {
    FaultType.DEPTH: {
        "quarter": 60.0,       # < 60° knee flexion
        "half": 90.0,          # 60-90° knee flexion
        "parallel": 100.0,     # 90-100° knee flexion
        "below_parallel": 100.0,  # > 100° knee flexion
    },
    FaultType.BILATERAL_ASYMMETRY: {
        "mild": 5.0,           # 5-10° difference
        "moderate": 10.0,      # 10-15° difference
        "severe": 15.0,        # > 15° difference
    },
    FaultType.FORWARD_LEAN: {
        "mild": 35.0,          # 35-45° trunk flexion
        "moderate": 45.0,      # 45-55° trunk flexion
        "severe": 55.0,        # > 55° trunk flexion
    },
    FaultType.KNEE_VALGUS: {
        "mild": 12.0,
        "moderate": 17.0,
        "severe": 24.0,
    },
    FaultType.BACK_ROUNDING: {
        "mild": 35.0,
        "moderate": 45.0,
        "severe": 55.0,
    },
    FaultType.TEMPO: {
        "eccentric_too_fast": 100.0,  # deg/sec - uncontrolled descent
        "stalling_velocity": 15.0,    # deg/sec - considered stalling
        "stall_frames": 10,           # frames before triggering stall warning
    },
}


# Human-readable fault messages
FAULT_MESSAGES: Dict[FaultType, Dict[str, str]] = {
    FaultType.DEPTH: {
        "quarter": "Only quarter squat — try to go deeper",
        "half": "Half squat depth — push a bit lower",
        "parallel": "Good depth at parallel",
        "below_parallel": "Great depth below parallel!",
    },
    FaultType.BILATERAL_ASYMMETRY: {
        "mild": "Slight imbalance between sides",
        "moderate": "Noticeable weight shift — even it out",
        "severe": "Significant asymmetry — focus on balance",
    },
    FaultType.FORWARD_LEAN: {
        "mild": "Slight forward lean",
        "moderate": "Too much forward lean — chest up",
        "severe": "Excessive forward lean — stay upright",
    },
    FaultType.KNEE_VALGUS: {
        "mild": "Slight knee cave",
        "moderate": "Knees caving in — push knees out",
        "severe": "Significant knee valgus — knees out!",
    },
    FaultType.BACK_ROUNDING: {
        "mild": "Slight back rounding",
        "moderate": "Back rounding — keep spine neutral",
        "severe": "Excessive back rounding — brace core",
    },
    FaultType.TEMPO: {
        "eccentric_fast": "Slow down! Control your descent",
        "stalling": "Grinding — push through!",
        "good": "Good tempo",
    },
}


class FaultRule(ABC):
    """
    Abstract base class for fault detection rules.

    Each rule evaluates joint angles and history to detect a specific
    type of form fault. Rules are stateless — all state is passed
    in via the history parameter.

    Some rules (bar_path, bar_tilt_asymmetry) also need per-frame context
    beyond joint angles — notably a real barbell detection. Rather than
    threading optional kwargs through every rule signature, the engine
    calls ``set_frame_context(...)`` before each ``evaluate(...)`` to set
    attributes on the rule; rules that care read them, others ignore.
    """

    # Per-frame context set by RuleEngine before each evaluate() call.
    # Rules that don't need it simply never read it.
    _bar_detection = None  # type: Optional["BarbellDetection"]
    _derivatives = None    # type: Optional["AngleDerivatives"]
    _phase = None          # type: Optional[str]
    _foot_state = None     # type: Optional["FootState"]

    def set_frame_context(self, bar_detection=None, derivatives=None, phase=None, foot_state=None) -> None:
        """Update per-frame auxiliary context for this rule.

        Called by the engine once per frame, before ``evaluate()``. Subclasses
        may override to react to context changes, but the default is just to
        stash references as attributes.
        """
        self._bar_detection = bar_detection
        self._derivatives = derivatives
        self._phase = phase
        self._foot_state = foot_state

    @property
    @abstractmethod
    def fault_type(self) -> FaultType:
        """Return the type of fault this rule detects."""
        pass

    @abstractmethod
    def evaluate(
        self,
        angles: JointAngles,
        history: deque,  # deque[JointAngles]
        in_rep: bool = False,
        rep_number: int = 0,
    ) -> Optional[FaultEvent]:
        """
        Evaluate joint angles for this fault type.

        Args:
            angles: Current frame's joint angles
            history: Rolling history of recent joint angles
            in_rep: Whether currently in the middle of a rep
            rep_number: Current rep number

        Returns:
            FaultEvent if fault detected, None otherwise
        """
        pass

    def _create_fault_event(
        self,
        severity: FaultSeverity,
        severity_score: float,
        message: str,
        angles: JointAngles,
        rep_number: int = 0,
        details: Optional[Dict[str, Any]] = None,
    ) -> FaultEvent:
        """Helper to create a FaultEvent with common fields."""
        return FaultEvent(
            fault_type=self.fault_type.value,
            severity=severity,
            severity_score=severity_score,
            message=message,
            timestamp=angles.timestamp,
            frame_index=angles.frame_index,
            rep_number=rep_number,
            details=details or {},
        )

    def scale_for_proportions(self, proportions) -> None:
        """Adjust thresholds based on user body proportions. Default: no-op.

        Subclasses (e.g. ForwardLeanRule) override to rescale their
        thresholds from the base values stored at construction, so repeated
        calls never compound (C8).
        """
        pass

    def _get_severity(self, value: float, thresholds: Dict[str, float]) -> tuple[FaultSeverity, float]:
        """
        Get severity level and score based on value and thresholds.

        Args:
            value: Measured value to evaluate
            thresholds: Dict with 'mild', 'moderate', 'severe' keys

        Returns:
            Tuple of (FaultSeverity, severity_score 0-3)
        """
        mild = thresholds.get("mild", 5.0)
        moderate = thresholds.get("moderate", 10.0)
        severe = thresholds.get("severe", 15.0)

        if value >= severe:
            # Map to 2.5-3.0 range
            score = min(3.0, 2.5 + 0.5 * (value - severe) / severe)
            return FaultSeverity.SEVERE, score
        elif value >= moderate:
            # Map to 1.5-2.5 range
            score = 1.5 + (value - moderate) / (severe - moderate)
            return FaultSeverity.MODERATE, score
        elif value >= mild:
            # Map to 0.5-1.5 range
            score = 0.5 + (value - mild) / (moderate - mild)
            return FaultSeverity.MILD, score
        else:
            return FaultSeverity.NONE, 0.0
