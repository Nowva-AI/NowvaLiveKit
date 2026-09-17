"""
Tempo-Based Fault Detection Rules

Detects tempo-related faults based on velocity analysis:
- EccentricTempoRule: Detects uncontrolled descent (too fast)
- StallingRule: Detects stalling/grinding during concentric phase
"""

from collections import deque
from typing import Optional

from biomechanics.faults.fault_types import FaultRule, FaultType, DEFAULT_THRESHOLDS, FAULT_MESSAGES
from biomechanics.utils.types import JointAngles, FaultEvent, FaultSeverity

# Minimum time between two eccentric-tempo reports (was 30 frames at 30 fps)
ECCENTRIC_TEMPO_COOLDOWN_S = 1.0


class EccentricTempoRule(FaultRule):
    """
    Detect uncontrolled descent (eccentric phase too fast).

    Triggers when knee flexion velocity exceeds threshold during descent.
    Based on research: optimal eccentric is 2-4 seconds, < 1 second is too fast.
    For ~120° ROM, 100 deg/sec = ~1.2 seconds for full descent.
    """

    def __init__(self, max_velocity: float = 100.0):
        """
        Initialize eccentric tempo rule.

        Args:
            max_velocity: Maximum allowed velocity (deg/sec) during descent.
                          Default 100 deg/sec based on research.
        """
        self.max_velocity = max_velocity
        self._last_fault_time_s: float = float("-inf")

    @property
    def fault_type(self) -> FaultType:
        return FaultType.TEMPO

    def evaluate(
        self,
        angles: JointAngles,
        history: deque,
        in_rep: bool = False,
        rep_number: int = 0,
    ) -> Optional[FaultEvent]:
        """Evaluate for too-fast eccentric movement."""
        # Cooldown check
        if angles.timestamp - self._last_fault_time_s < ECCENTRIC_TEMPO_COOLDOWN_S:
            return None

        # Only check during descending phase
        if not in_rep or self._phase != "descending":
            return None

        # Need derivatives (set via set_frame_context)
        derivatives = self._derivatives
        if derivatives is None:
            return None

        velocity = derivatives.avg_knee_velocity

        # During descent, velocity should be negative (angle increasing)
        # We check if the magnitude is too high
        if velocity < -self.max_velocity:
            self._last_fault_time_s = angles.timestamp

            # Calculate severity based on how fast
            excess = abs(velocity) - self.max_velocity
            if excess > 50:  # Very fast
                severity = FaultSeverity.SEVERE
                score = 2.5 + min(0.5, excess / 100)
            elif excess > 25:  # Moderately fast
                severity = FaultSeverity.MODERATE
                score = 1.5 + excess / 50
            else:  # Slightly fast
                severity = FaultSeverity.MILD
                score = 0.5 + excess / 50

            return self._create_fault_event(
                severity=severity,
                severity_score=score,
                message=FAULT_MESSAGES[FaultType.TEMPO]["eccentric_fast"],
                angles=angles,
                rep_number=rep_number,
                details={
                    "velocity": abs(velocity),
                    "threshold": self.max_velocity,
                    "excess": excess,
                },
            )

        return None

    def reset(self):
        """Reset cooldown."""
        self._last_fault_time_s = float("-inf")


class StallingRule(FaultRule):
    """
    Detect stalling/grinding during concentric phase.

    Triggers when velocity stays below threshold for extended time
    while in the ascending phase of a rep.
    """

    def __init__(self, min_velocity: float = 15.0, stall_frames: int = 10):
        """
        Initialize stalling rule.

        Args:
            min_velocity: Minimum expected velocity (deg/sec) during ascent.
            stall_frames: Number of consecutive slow frames before triggering.
        """
        self.min_velocity = min_velocity
        self.stall_frames = stall_frames
        self._slow_frame_count = 0
        self._triggered = False  # Only trigger once per stall

    @property
    def fault_type(self) -> FaultType:
        return FaultType.TEMPO

    def evaluate(
        self,
        angles: JointAngles,
        history: deque,
        in_rep: bool = False,
        rep_number: int = 0,
    ) -> Optional[FaultEvent]:
        """Evaluate for stalling during ascent."""
        # Only check during ascending phase
        if not in_rep or self._phase != "ascending":
            self._slow_frame_count = 0
            self._triggered = False
            return None

        # Need derivatives (set via set_frame_context)
        derivatives = self._derivatives
        if derivatives is None:
            return None

        velocity = derivatives.avg_knee_velocity

        # During ascent, velocity should be positive (angle decreasing = knee extending)
        # We check if it's too slow
        if velocity < self.min_velocity:
            self._slow_frame_count += 1

            if self._slow_frame_count >= self.stall_frames and not self._triggered:
                self._triggered = True

                # Mild severity for stalling
                return self._create_fault_event(
                    severity=FaultSeverity.MILD,
                    severity_score=1.0,
                    message=FAULT_MESSAGES[FaultType.TEMPO]["stalling"],
                    angles=angles,
                    rep_number=rep_number,
                    details={
                        "velocity": velocity,
                        "threshold": self.min_velocity,
                        "stall_duration_frames": self._slow_frame_count,
                    },
                )
        else:
            self._slow_frame_count = 0
            self._triggered = False

        return None

    def reset(self):
        """Reset stall tracking."""
        self._slow_frame_count = 0
        self._triggered = False
