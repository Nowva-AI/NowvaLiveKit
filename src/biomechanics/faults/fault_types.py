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
    HIP_SHOOT = "hip_shoot"
    HIP_SHIFT = "hip_shift"
    BALANCE = "balance"
    DEPTH_DRIFT = "depth_drift"
    VELOCITY_LOSS = "velocity_loss"
    FOOT_PLACEMENT = "foot_placement"
    # Conventional deadlift (docs/deadlift/PLAN.md §2.6). Prefixed so none shares
    # the squat's priority, observability or cue entries.
    DEADLIFT_BAR_POSITION = "deadlift_bar_position"
    DEADLIFT_SETUP_HIPS = "deadlift_setup_hips"
    DEADLIFT_SHOULDERS_BEHIND = "deadlift_shoulders_behind"
    DEADLIFT_HIPS_SHOOT = "deadlift_hips_shoot"
    DEADLIFT_BAR_DRIFT = "deadlift_bar_drift"
    DEADLIFT_LOCKOUT = "deadlift_lockout"
    DEADLIFT_LEAN_BACK = "deadlift_lean_back"
    DEADLIFT_HIP_SHIFT = "deadlift_hip_shift"
    DEADLIFT_BAR_TILT = "deadlift_bar_tilt"
    DEADLIFT_BENT_ARMS = "deadlift_bent_arms"
    DEADLIFT_VELOCITY_LOSS = "deadlift_velocity_loss"


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


# Human-readable fault messages. Plain words and external-focus cues — the
# athlete never hears biomechanics jargon.
FAULT_MESSAGES: Dict[FaultType, Dict[str, str]] = {
    FaultType.DEPTH: {
        "quarter": "Only a quarter squat — sit lower",
        "half": "Half squat — sit a little lower",
        "parallel": "Good depth at parallel",
        "below_parallel": "Great depth below parallel",
        "mild": "Just short of your depth — sit a little lower",
        "moderate": "Short of your depth — sit lower",
        "severe": "Well short of your depth — sit down between your heels",
    },
    FaultType.BILATERAL_ASYMMETRY: {
        "mild": "Bar tipping slightly — keep it level",
        "moderate": "Bar tilting — push evenly with both feet",
        "severe": "Bar tilting a lot — even out both sides",
    },
    FaultType.FORWARD_LEAN: {
        "mild": "Slight forward lean",
        "moderate": "Too much forward lean — chest up",
        "severe": "Excessive forward lean — stay upright",
    },
    FaultType.KNEE_VALGUS: {
        "mild": "Knees drifting in — spread the floor",
        "moderate": "Knees caving — push them out over your little toes",
        "severe": "Knees collapsing in — knees out, spread the floor",
    },
    FaultType.BACK_ROUNDING: {
        "mild": "Slight back rounding",
        "moderate": "Back rounding — keep spine neutral",
        "severe": "Excessive back rounding — brace core",
    },
    FaultType.TEMPO: {
        "eccentric_fast": "Slow down — control the way down",
        "stalling": "Keep driving up",
        "good": "Good tempo",
        "mild": "A bit fast on the way down — control it",
        "moderate": "Dropping fast — control the way down",
        "severe": "Dropping into the bottom — slow the descent",
    },
    FaultType.HIP_SHOOT: {
        "mild": "Hips rising first — chest and hips up together",
        "moderate": "Hips shooting up — drive your back into the bar",
        "severe": "Turning it into a good morning — lead with your chest",
    },
    FaultType.HIP_SHIFT: {
        "mild": "Hips drifting to one side — stay centered",
        "moderate": "Hips shifting — push evenly through both feet",
        "severe": "Big hip shift — center up between your feet",
    },
    FaultType.BALANCE: {
        "mild": "Weight drifting — stay over the whole foot",
        "moderate": "Off balance — feel your heel and big toe",
        "severe": "Way off balance — whole foot on the floor",
    },
    FaultType.DEPTH_DRIFT: {
        "mild": "Getting a little shallower — same depth as your first rep",
        "moderate": "Cutting depth — match your first rep",
        "severe": "Much shallower than earlier — match your first rep",
    },
    FaultType.LOCKOUT: {
        "mild": "Stand all the way up",
        "moderate": "Not standing tall — finish each rep",
        "severe": "Stand fully between reps",
    },
    FaultType.VELOCITY_LOSS: {
        "mild": "Rep slowing down — drive hard",
        "moderate": "That one slowed a lot — drive out of the bottom",
        "severe": "Big slowdown — that's close to your limit",
    },
    FaultType.FOOT_PLACEMENT: {
        "mild": "Feet a little uneven — square them up",
        "moderate": "Feet uneven — line them up",
        "severe": "Feet set up crooked — reset your stance",
    },
    # Deadlift: behavioural words only, nothing medical, never a "flat back" claim.
    FaultType.DEADLIFT_BAR_POSITION: {
        "mild": "Bar a little off midfoot at the start",
        "moderate": "Bar not over midfoot — set it over the middle of your foot",
        "severe": "Bar far from midfoot — reset your feet to the bar",
    },
    FaultType.DEADLIFT_SETUP_HIPS: {
        "mild": "Hips slightly off at the start",
        "moderate": "Hip height off at the start — reset your hips",
        "severe": "Hips well off at the start — reset before you pull",
    },
    FaultType.DEADLIFT_SHOULDERS_BEHIND: {
        "mild": "Shoulders slightly behind the bar",
        "moderate": "Shoulders behind the bar — bring them over it",
        "severe": "Shoulders well behind the bar — hips up, shoulders over the bar",
    },
    FaultType.DEADLIFT_HIPS_SHOOT: {
        "mild": "Hips rising a little first — chest and hips together",
        "moderate": "Hips shooting up — lift your chest with your hips",
        "severe": "Hips shot up first — push the floor, chest and hips together",
    },
    FaultType.DEADLIFT_BAR_DRIFT: {
        "mild": "Bar drifting off your legs",
        "moderate": "Bar drifting forward — keep it close",
        "severe": "Bar swinging away — drag it up your legs",
    },
    FaultType.DEADLIFT_LOCKOUT: {
        "mild": "Not quite standing tall at the top",
        "moderate": "Finish the rep — stand all the way up",
        "severe": "Rep left unfinished — stand tall at the top",
    },
    FaultType.DEADLIFT_LEAN_BACK: {
        "mild": "Leaning back a little at the top",
        "moderate": "Leaning back at the top — just stand tall",
        "severe": "Leaning way back at the top — finish tall, no lean",
    },
    FaultType.DEADLIFT_HIP_SHIFT: {
        "mild": "Hips drifting to one side",
        "moderate": "Hips shifting — push evenly through both feet",
        "severe": "Big hip shift — even out both sides",
    },
    FaultType.DEADLIFT_BAR_TILT: {
        "mild": "Bar tipping slightly",
        "moderate": "Bar tilting — keep it level",
        "severe": "Bar tilting a lot — pull evenly with both hands",
    },
    FaultType.DEADLIFT_BENT_ARMS: {
        "mild": "Arms bending a little",
        "moderate": "Arms bending — keep them long",
        "severe": "Pulling with the arms — long arms, push the floor",
    },
    FaultType.DEADLIFT_VELOCITY_LOSS: {
        "mild": "That rep slowed down",
        "moderate": "Bar slowing a lot",
        "severe": "Big slowdown — that's close to your limit",
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


class RepFaultRule(FaultRule):
    """A rule judged once per rep from the rep's features, never frame by frame.

    The engine calls ``judge_rep`` when a rep completes. Judging the whole rep
    keeps one noisy frame from producing a fault and puts each verdict on the
    rep it belongs to.
    """

    def evaluate(
        self,
        angles: JointAngles,
        history: deque,
        in_rep: bool = False,
        rep_number: int = 0,
    ) -> Optional[FaultEvent]:
        return None

    @abstractmethod
    def judge_rep(self, features, reference, angles: JointAngles) -> Optional[FaultEvent]:
        """Return this rep's fault, or None. ``reference`` is the SessionReference."""

    def _rep_fault(
        self,
        value: float,
        thresholds: Dict[str, float],
        angles: JointAngles,
        rep_number: int,
        unit: str,
        side: Optional[str] = None,
        phase: Optional[str] = None,
        is_drift: bool = False,
        **extra: Any,
    ) -> Optional[FaultEvent]:
        severity, score = self._get_severity(value, thresholds)
        if severity == FaultSeverity.NONE:
            return None
        message = FAULT_MESSAGES.get(self.fault_type, {}).get(severity.value, self.fault_type.value)
        details: Dict[str, Any] = {
            "side": side,
            "phase": phase,
            "is_drift": is_drift,
            "value": value,
            "unit": unit,
        }
        details.update(extra)
        return self._create_fault_event(
            severity=severity,
            severity_score=score,
            message=message,
            angles=angles,
            rep_number=rep_number,
            details=details,
        )
