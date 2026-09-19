"""Fixed-lag constant-velocity Kalman smoother for (N, 3) keypoint arrays.

The single temporal filter of the pre-IK chain: measurement noise from triangulator confidence, a per-keypoint
innovation gate with re-acquisition, bounded prediction for missing keypoints, a closed-form fixed-lag (RTS)
estimate for analysis and the undelayed filtered estimate for display. Frame-agnostic (world or hip-centred).
"""
from __future__ import annotations

from collections import deque
from typing import NamedTuple

import numpy as np

MAX_GAP_S = 0.5
MAX_REJECTION_S = 0.2
REACQUIRE_AGREEMENT_M = 0.10
# Real RTMPose outlier runs last 1.6 frames on average (p90 3) with a constant offset and L/R swaps 2-6 frames,
# so a snap needs 4 agreeing rejections; the 0.2 s backstop needs the last 2 to agree.
REACQUIRE_AGREEING_FRAMES = 4
TIMEOUT_AGREEING_FRAMES = 2
# Two-view keypoints are capped at this confidence (triangulator): such measurements can never re-seed a tracked
# keypoint, neither by snapping nor through the gate widened by predict-only uncertainty.
MIN_SNAP_CONFIDENCE = 0.3
INITIAL_VELOCITY_VARIANCE = 1.0
MIN_CONFIDENCE_FOR_VARIANCE = 1e-6
TIME_EPSILON_S = 1e-6
SPATIAL_AXES = 3


class KeypointKalmanOutput(NamedTuple):
    lagged_points: np.ndarray
    lagged_confidences: np.ndarray
    lagged_timestamp: float
    current_points: np.ndarray
    current_confidences: np.ndarray
    velocities: np.ndarray
    current_timestamp: float
    gate_rejected: np.ndarray


class _FilterStep(NamedTuple):
    timestamp: float
    positions: np.ndarray
    velocities: np.ndarray
    predicted_positions: np.ndarray
    predicted_velocities: np.ndarray
    smoother_gains: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None
    confidences: np.ndarray


class FixedLagKeypointSmoother:
    """Per keypoint, per axis constant-velocity Kalman filter with a `lag_frames` fixed-lag smoothed output.

    Confidence c maps to measurement std u = scale * sqrt(1/c - 1) (triangulator contract); posterior std maps
    back the same way, so output confidences share the input's metric meaning. `gate_rejected` flags valid
    measurements that failed the innovation gate on the current tick (including ones that then re-acquired).

    A tracked keypoint snaps to rejected measurements only after REACQUIRE_AGREEING_FRAMES agreeing rejections
    (or after MAX_REJECTION_S with the last TIMEOUT_AGREEING_FRAMES agreeing), and only at confidence above
    MIN_SNAP_CONFIDENCE. A keypoint that exceeds max_predicted_frames misses is dropped (confidence 0) and
    re-seeds from its next valid measurement at any positive confidence.
    """

    def __init__(self, lag_frames: int = 2, process_noise: float = 10.0,
                 uncertainty_scale_m: float = 0.02, measurement_std_floor_m: float = 0.003,
                 measurement_std_ceiling_m: float = 0.08, gate_sigma: float = 4.0,
                 gate_min_radius_m: float = 0.08, max_predicted_frames: int = 5,
                 min_output_confidence: float = 0.15) -> None:
        if lag_frames < 0 or max_predicted_frames < 0:
            raise ValueError("lag_frames and max_predicted_frames must be >= 0")
        if process_noise <= 0.0 or uncertainty_scale_m <= 0.0:
            raise ValueError("process_noise and uncertainty_scale_m must be > 0")
        if not 0.0 < measurement_std_floor_m <= measurement_std_ceiling_m:
            raise ValueError("need 0 < measurement_std_floor_m <= measurement_std_ceiling_m")
        self._lag_frames = lag_frames
        self._process_noise = process_noise
        self._scale_sq = uncertainty_scale_m ** 2
        self._variance_floor = measurement_std_floor_m ** 2
        self._variance_ceiling = measurement_std_ceiling_m ** 2
        # 3D innovation distance has RMS sqrt(3) * per-axis sigma, so gate_sigma is applied to that RMS.
        self._gate_sigma_sq = gate_sigma ** 2 * SPATIAL_AXES
        self._gate_min_radius_sq = gate_min_radius_m ** 2
        self._max_predicted_frames = max_predicted_frames
        self._min_output_confidence = min_output_confidence
        self._reacquire_agreement_sq = REACQUIRE_AGREEMENT_M ** 2
        self._history: deque[_FilterStep] = deque(maxlen=lag_frames + 1)
        self.reset()

    def reset(self) -> None:
        self._timestamp: float | None = None
        self._last_output: KeypointKalmanOutput | None = None
        self._history.clear()

    def update(self, points: np.ndarray, confidences: np.ndarray,
               timestamp: float) -> KeypointKalmanOutput | None:
        if self._timestamp is not None:
            dt = timestamp - self._timestamp
            if dt <= 0.0:
                return self._last_output
            if dt > MAX_GAP_S:
                self.reset()
        if self._timestamp is None:
            return self._initialise(points, confidences, timestamp)
        return self._step(points, confidences, timestamp, dt)

    def predict_missing(self, timestamp: float) -> KeypointKalmanOutput | None:
        if self._timestamp is None:
            return None
        return self.update(self._no_points, self._no_confidences, timestamp)

    def _measurement_variance(self, confidences: np.ndarray) -> np.ndarray:
        variance = self._scale_sq / np.fmax(confidences, MIN_CONFIDENCE_FOR_VARIANCE) - self._scale_sq
        return np.minimum(np.maximum(variance, self._variance_floor), self._variance_ceiling)[:, None]

    def _initialise(self, points: np.ndarray, confidences: np.ndarray,
                    timestamp: float) -> KeypointKalmanOutput | None:
        valid = (confidences > 0.0) & np.isfinite(points).all(axis=1)
        if not valid.any():
            return None
        keypoint_count = len(points)
        self._no_points = np.zeros((keypoint_count, SPATIAL_AXES))
        self._no_confidences = np.zeros(keypoint_count)
        self._no_rejections = np.zeros(keypoint_count, dtype=bool)
        self._no_rejections.flags.writeable = False
        self._positions = np.where(valid[:, None], points, 0.0)
        self._velocities = np.zeros((keypoint_count, SPATIAL_AXES))
        self._p00 = np.where(valid[:, None], self._measurement_variance(confidences), self._variance_floor)
        self._p01 = np.zeros((keypoint_count, 1))
        self._p11 = np.full((keypoint_count, 1), INITIAL_VELOCITY_VARIANCE)
        self._tracked = valid
        self._all_tracked = bool(valid.all())
        self._missed_frames = np.zeros(keypoint_count, dtype=np.int64)
        self._any_missed = False
        self._last_accepted_timestamps = np.full(keypoint_count, timestamp)
        self._rejection_streak = np.zeros(keypoint_count, dtype=np.int64)
        self._any_rejection_streak = False
        self._candidate_positions = np.zeros((keypoint_count, SPATIAL_AXES))
        self._candidate_timestamps = np.full(keypoint_count, timestamp)
        self._timestamp = timestamp
        confidences_out = np.where(valid, self._scale_sq / (self._p00[:, 0] + self._scale_sq), 0.0)
        confidences_out.flags.writeable = False
        self._history.append(_FilterStep(timestamp, self._positions, self._velocities, self._positions,
                                         self._velocities, None, confidences_out))
        return self._emit(self._no_rejections)

    def _step(self, points: np.ndarray, confidences: np.ndarray, timestamp: float,
              dt: float) -> KeypointKalmanOutput:
        q = self._process_noise
        dt_sq = dt * dt
        p00, p01, p11 = self._p00, self._p01, self._p11
        predicted_velocities = self._velocities
        predicted_positions = self._positions + dt * predicted_velocities
        pp00 = p00 + (2.0 * dt) * p01 + dt_sq * p11 + q * dt_sq * dt / 3.0
        pp01 = p01 + dt * p11 + q * dt_sq / 2.0
        pp11 = p11 + q * dt

        variance = self._measurement_variance(confidences)
        innovation_variance = pp00 + variance
        innovation = points - predicted_positions
        distance_sq = np.einsum("ij,ij->i", innovation, innovation)
        # Prediction uncertainty widens the gate only for measurements trusted enough to re-seed the keypoint and
        # only while no rejection streak is running: uncertainty accumulated by rejecting is not evidence for
        # the rejected hypothesis (a burst would otherwise walk in through the gate on its third frame).
        trusted = confidences > MIN_SNAP_CONFIDENCE
        if self._any_rejection_streak:
            trusted &= self._rejection_streak == 0
        gate_variance = variance[:, 0] + pp00[:, 0] * trusted
        gate_sq = np.maximum(self._gate_sigma_sq * gate_variance, self._gate_min_radius_sq)
        accepted = (distance_sq <= gate_sq) & (confidences > 0.0)
        if not self._all_tracked:
            accepted &= self._tracked
        all_accepted = bool(accepted.all())

        gain_position = pp00 / innovation_variance
        gain_velocity = pp01 / innovation_variance
        if not all_accepted:
            accepted_column = accepted[:, None]
            gain_position = gain_position * accepted_column
            gain_velocity = gain_velocity * accepted_column
            innovation = np.where(accepted_column, innovation, 0.0)
        positions = predicted_positions + gain_position * innovation
        velocities = predicted_velocities + gain_velocity * innovation
        new_p00 = pp00 - gain_position * pp00
        new_p01 = pp01 - gain_position * pp01
        new_p11 = pp11 - gain_velocity * pp01

        smoother_gains = None
        if self._lag_frames > 0:
            # C = P_filtered F^T inv(P_predicted), closed form for the 2x2 [position, velocity] covariance.
            inverse_determinant = 1.0 / (pp00 * pp11 - pp01 * pp01)
            cross_position = p00 + dt * p01
            cross_velocity = p01 + dt * p11
            smoother_gains = ((cross_position * pp11 - p01 * pp01) * inverse_determinant,
                              (p01 * pp00 - cross_position * pp01) * inverse_determinant,
                              (cross_velocity * pp11 - p11 * pp01) * inverse_determinant,
                              (p11 * pp00 - cross_velocity * pp01) * inverse_determinant)

        if all_accepted:
            gate_rejected = self._no_rejections
            if self._any_rejection_streak:
                self._rejection_streak.fill(0)
                self._any_rejection_streak = False
            if self._any_missed:
                self._missed_frames.fill(0)
                self._any_missed = False
            self._last_accepted_timestamps.fill(timestamp)
            confidences_out = self._scale_sq / (new_p00[:, 0] + self._scale_sq)
        else:
            gate_rejected, smoother_gains, confidences_out = self._handle_exceptions(
                points, confidences, timestamp, distance_sq, accepted, variance, predicted_velocities,
                positions, velocities, new_p00, new_p01, new_p11, smoother_gains)

        confidences_out.flags.writeable = False
        self._positions, self._velocities = positions, velocities
        self._p00, self._p01, self._p11 = new_p00, new_p01, new_p11
        self._timestamp = timestamp
        self._history.append(_FilterStep(timestamp, positions, velocities, predicted_positions,
                                         predicted_velocities, smoother_gains, confidences_out))
        return self._emit(gate_rejected)

    def _handle_exceptions(self, points: np.ndarray, confidences: np.ndarray, timestamp: float,
                           distance_sq: np.ndarray, accepted: np.ndarray, variance: np.ndarray,
                           predicted_velocities: np.ndarray, positions: np.ndarray, velocities: np.ndarray,
                           new_p00: np.ndarray, new_p01: np.ndarray, new_p11: np.ndarray,
                           smoother_gains: tuple[np.ndarray, ...] | None,
                           ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None, np.ndarray]:
        tracked_before = self._tracked
        valid = (confidences > 0.0) & np.isfinite(distance_sq)
        rejected = valid & tracked_before & ~accepted
        fresh = valid & ~tracked_before
        keep_velocity = None
        streak = self._rejection_streak
        if rejected.any():
            # Agreeing streak: rejected measurements each within REACQUIRE_AGREEMENT_M of the previous rejected
            # one (velocity-compensated); survives missing frames, a disagreeing rejection restarts it at 1.
            elapsed = timestamp - self._candidate_timestamps
            drift = points - self._candidate_positions - predicted_velocities * elapsed[:, None]
            agrees = (streak > 0) & (np.einsum("ij,ij->i", drift, drift) <= self._reacquire_agreement_sq)
            streak = np.where(rejected, np.where(agrees, streak + 1, 1), streak)
            can_snap = rejected & (confidences > MIN_SNAP_CONFIDENCE)
            reacquired = can_snap & (streak >= REACQUIRE_AGREEING_FRAMES)
            timed_out = (can_snap & ~reacquired & (streak >= TIMEOUT_AGREEING_FRAMES)
                         & (timestamp - self._last_accepted_timestamps >= MAX_REJECTION_S - TIME_EPSILON_S))
            self._candidate_positions = np.where(rejected[:, None], points, self._candidate_positions)
            self._candidate_timestamps = np.where(rejected, timestamp, self._candidate_timestamps)
            keep_velocity = reacquired
            fresh = fresh | reacquired | timed_out

        received = accepted | fresh
        tracked = tracked_before | fresh
        if fresh.any():
            positions[fresh] = points[fresh]
            velocities[fresh if keep_velocity is None else fresh & ~keep_velocity] = 0.0
            new_p00[fresh] = variance[fresh]
            new_p01[fresh] = 0.0
            new_p11[fresh] = INITIAL_VELOCITY_VARIANCE

        self._missed_frames = np.where(received, 0, self._missed_frames + 1)
        self._any_missed = True
        self._last_accepted_timestamps = np.where(received, timestamp, self._last_accepted_timestamps)
        lost = tracked & (self._missed_frames > self._max_predicted_frames)
        if lost.any():
            tracked = tracked & ~lost
            velocities[lost] = 0.0
        untracked = ~tracked
        if untracked.any():
            new_p00[untracked] = self._variance_floor
            new_p01[untracked] = 0.0
            new_p11[untracked] = INITIAL_VELOCITY_VARIANCE
        self._tracked = tracked
        self._all_tracked = bool(tracked.all())
        self._rejection_streak = np.where(received | ~tracked, 0, streak)
        self._any_rejection_streak = bool(self._rejection_streak.any())

        continuous = tracked_before & tracked & ~fresh
        if smoother_gains is not None and not continuous.all():
            continuous_column = continuous[:, None]
            smoother_gains = tuple(gain * continuous_column for gain in smoother_gains)

        confidences_out = self._scale_sq / (new_p00[:, 0] + self._scale_sq)
        confidences_out = np.where(received, confidences_out,
                                   np.maximum(confidences_out, self._min_output_confidence))
        confidences_out = np.where(tracked, confidences_out, 0.0)
        return rejected, smoother_gains, confidences_out

    def _emit(self, gate_rejected: np.ndarray) -> KeypointKalmanOutput:
        history = self._history
        newest = history[-1]
        smoothed_positions, smoothed_velocities = newest.positions, newest.velocities
        for index in range(len(history) - 1, 0, -1):
            later, earlier = history[index], history[index - 1]
            gain_pp, gain_pv, gain_vp, gain_vv = later.smoother_gains
            position_residual = smoothed_positions - later.predicted_positions
            velocity_residual = smoothed_velocities - later.predicted_velocities
            smoothed_positions = earlier.positions + gain_pp * position_residual + gain_pv * velocity_residual
            smoothed_velocities = earlier.velocities + gain_vp * position_residual + gain_vv * velocity_residual
        if len(history) == 1:
            smoothed_positions, smoothed_velocities = smoothed_positions.copy(), smoothed_velocities.copy()
        oldest = history[0]
        self._last_output = KeypointKalmanOutput(
            lagged_points=smoothed_positions,
            lagged_confidences=oldest.confidences,
            lagged_timestamp=oldest.timestamp,
            current_points=newest.positions.copy(),
            current_confidences=newest.confidences,
            velocities=smoothed_velocities,
            current_timestamp=newest.timestamp,
            gate_rejected=gate_rejected,
        )
        return self._last_output
