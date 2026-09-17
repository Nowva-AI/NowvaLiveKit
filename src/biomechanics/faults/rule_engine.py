"""
Fault Detection Rule Engine

Orchestrates all fault detection rules, maintains angle history,
and deduplicates same-fault reports that arrive within DEDUP_INTERVAL_S.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING, List, Optional, Dict

from biomechanics.utils.types import JointAngles, FaultEvent, BarbellDetection
from biomechanics.utils.derivatives import AngleDerivatives
from biomechanics.faults.fault_types import FaultRule, FaultType
from biomechanics.config import BiomechanicsConfig, get_config

if TYPE_CHECKING:
    from biomechanics.utils.segment_lengths import BodyProportions

logger = logging.getLogger(__name__)

# Minimum time between two reports of the same fault type (was 15 frames at 30 fps).
DEDUP_INTERVAL_S = 0.5


class RuleEngine:
    """
    Orchestrates all fault detection rules.

    Responsibilities:
    - Maintains rolling history of joint angles (maxlen=90 frames)
    - Runs all enabled fault rules on each frame
    - Deduplicates consecutive same-fault detections
    - Provides centralized fault reporting

    Usage:
        engine = RuleEngine()
        faults = engine.evaluate(angles, in_rep=True, rep_number=1)
    """

    def __init__(
        self,
        config: Optional[BiomechanicsConfig] = None,
        history_maxlen: int = 90,
        rules: Optional[List[FaultRule]] = None,
    ):
        """
        Initialize the rule engine.

        Args:
            config: Pipeline configuration (uses global if not provided)
            history_maxlen: Maximum frames to keep in history (default 90 = ~3s at 30fps)
            rules: List of fault rules from the exercise profile. Required —
                   the profile is the single source of truth for which rules
                   apply to each exercise.
        """
        self.config = config or get_config()
        self.history: deque = deque(maxlen=history_maxlen)

        if rules is None:
            raise ValueError(
                "RuleEngine requires rules from the exercise profile. "
                "Pass rules=profile.create_fault_rules(config)."
            )
        self.rules: List[FaultRule] = rules

        # Deduplication tracking: fault_type -> timestamp of the last report
        self._last_fault_times: Dict[str, float] = {}

        # Baseline calibration — after first clean rep, adjust thresholds
        # to the user's natural movement pattern. The profile owns the
        # exercise-specific tracking logic; RuleEngine just holds the state
        # dict and calls the profile hooks.
        self._calibrated: bool = False
        self._calibration_reps: int = 0
        self._calibration_target: int = 1  # Calibrate after 1 clean rep
        self._calibration_state: Dict = {}  # Profile-owned state bag
        self._profile = None  # Set by pipeline after construction

    def apply_body_proportion_scaling(self, proportions: BodyProportions) -> None:
        """Scale fault thresholds based on the user's body proportions.

        Idempotent: each rule rescales from the base thresholds it stored at
        construction, so calling this again (or with new proportions) never
        compounds (C8). Rules that don't need scaling inherit the no-op default.
        """
        for rule in self.rules:
            rule.scale_for_proportions(proportions)
            logger.info(
                "[RULE ENGINE] Proportion scaling applied to %s",
                rule.fault_type.value if hasattr(rule.fault_type, 'value') else rule.fault_type,
            )

    def reset(self) -> None:
        """Reset engine state (clear history and rule states)."""
        self.history.clear()
        self._last_fault_times.clear()

        # Reset stateful rules
        for rule in self.rules:
            if hasattr(rule, "reset"):
                rule.reset()

    def add_rule(self, rule: FaultRule) -> None:
        """Add a custom rule to the engine."""
        self.rules.append(rule)

    def remove_rule(self, fault_type: FaultType) -> bool:
        """Remove a rule by fault type. Returns True if removed."""
        for i, rule in enumerate(self.rules):
            if rule.fault_type == fault_type:
                self.rules.pop(i)
                return True
        return False

    def evaluate(
        self,
        angles: JointAngles,
        in_rep: bool = False,
        rep_number: int = 0,
        bar_detection: Optional[BarbellDetection] = None,
        derivatives: Optional[AngleDerivatives] = None,
        phase: Optional[str] = None,
        foot_state=None,
    ) -> List[FaultEvent]:
        """
        Evaluate all rules for the current frame.

        Args:
            angles: Current frame's joint angles
            in_rep: Whether currently in a rep
            rep_number: Current rep number
            bar_detection: Optional real barbell detection for this frame.
                Rules that care (BarPathRule, BarTiltAsymmetryRule) pick it up
                via ``set_frame_context``; others ignore it.
            derivatives: Optional velocity/acceleration data for tempo rules.
            phase: Current rep phase (descending, bottom, ascending, idle).
            foot_state: Optional FootState from the foot contact model (multi-camera only).

        Returns:
            List of detected faults (deduplicated)
        """
        # Add to history
        self.history.append(angles)

        faults: List[FaultEvent] = []

        for rule in self.rules:
            rule.set_frame_context(bar_detection=bar_detection, derivatives=derivatives, phase=phase, foot_state=foot_state)
            fault = rule.evaluate(
                angles=angles,
                history=self.history,
                in_rep=in_rep,
                rep_number=rep_number,
            )

            if fault is not None:
                # Deduplicate same-fault reports within the dedup interval
                if self._should_report_fault(fault, angles.timestamp):
                    faults.append(fault)
                    self._last_fault_times[fault.fault_type] = angles.timestamp

        return faults

    def _should_report_fault(self, fault: FaultEvent, timestamp: float) -> bool:
        """Check if fault should be reported (time-based deduplication)."""
        last_time = self._last_fault_times.get(fault.fault_type, float("-inf"))
        return timestamp - last_time >= DEDUP_INTERVAL_S

    # ------------------------------------------------------------------
    # Baseline calibration (delegates to profile)
    # ------------------------------------------------------------------

    def set_profile(self, profile) -> None:
        """Set the exercise profile for calibration delegation."""
        self._profile = profile

    def record_frame_for_calibration(self, angles: JointAngles) -> None:
        """Track peak angle values during reps for baseline calibration."""
        if self._calibrated:
            return
        if self._profile is not None:
            self._profile.record_calibration_frame(angles, self._calibration_state)

    def on_rep_complete_calibration(self, is_clean: bool) -> None:
        """Called after a rep completes to advance calibration."""
        if self._calibrated:
            return

        if is_clean:
            self._calibration_reps += 1

        if self._calibration_reps >= self._calibration_target:
            self._calibrated = True
            if self._profile is not None:
                self._profile.apply_baseline(self.rules, self._calibration_state)
            logger.info("[RULE ENGINE] Baseline calibration complete")

    @property
    def calibrated(self) -> bool:
        """Whether baseline calibration is complete."""
        return self._calibrated

    def finish_rep(self, angles: JointAngles, rep_number: int) -> List[FaultEvent]:
        """Give per-rep rules their verdict on the rep that just completed."""
        faults: List[FaultEvent] = []
        for rule in self.rules:
            finish = getattr(rule, "finish_rep", None)
            if finish is None:
                continue
            fault = finish(angles, rep_number)
            if fault is not None and self._should_report_fault(fault, angles.timestamp):
                faults.append(fault)
                self._last_fault_times[fault.fault_type] = angles.timestamp
        return faults

    def discard_rep(self) -> None:
        """Drop per-rep rule state for a descent that was not counted."""
        for rule in self.rules:
            discard = getattr(rule, "discard_rep", None)
            if discard is not None:
                discard()

    def evaluate_rep_complete(
        self,
        max_depth_angle: float,
        angles: JointAngles,
        rep_number: int,
    ) -> List[FaultEvent]:
        """
        Evaluate rules that fire at rep completion.

        Currently only depth rule fires at rep end.

        Args:
            max_depth_angle: Maximum knee flexion achieved in rep
            angles: Current (end of rep) joint angles
            rep_number: Completed rep number

        Returns:
            List of rep-completion faults
        """
        faults: List[FaultEvent] = []

        # Find and evaluate depth rule (use fault_type check so any
        # profile's depth rule works, not just the squat DepthRule class)
        for rule in self.rules:
            if rule.fault_type == FaultType.DEPTH:
                if hasattr(rule, "evaluate_max_depth"):
                    fault = rule.evaluate_max_depth(
                        max_knee_flexion=max_depth_angle,
                        angles=angles,
                        rep_number=rep_number,
                    )
                    if fault is not None:
                        faults.append(fault)
                break

        return faults

    def evaluate_shallow_rep(
        self,
        max_depth_class: int,
        angles: JointAngles,
        rep_number: int,
        max_knee_flexion: float = 0.0,
    ) -> List[FaultEvent]:
        """
        Evaluate a rep that was rejected for insufficient depth.

        Mirrors evaluate_rep_complete, but the depth class from the rep
        counter is the input rather than the measured knee angle — the rep
        was rejected on that class, so the fault must follow it.
        """
        faults: List[FaultEvent] = []

        for rule in self.rules:
            if rule.fault_type == FaultType.DEPTH:
                if hasattr(rule, "evaluate_depth_class"):
                    fault = rule.evaluate_depth_class(
                        max_depth_class=max_depth_class,
                        angles=angles,
                        rep_number=rep_number,
                        max_knee_flexion=max_knee_flexion,
                    )
                    if fault is not None:
                        faults.append(fault)
                break

        return faults

    def get_rule(self, fault_type: FaultType) -> Optional[FaultRule]:
        """Get a specific rule by fault type."""
        for rule in self.rules:
            if rule.fault_type == fault_type:
                return rule
        return None

    @property
    def rule_count(self) -> int:
        """Number of active rules."""
        return len(self.rules)

    @property
    def history_length(self) -> int:
        """Current history length."""
        return len(self.history)
