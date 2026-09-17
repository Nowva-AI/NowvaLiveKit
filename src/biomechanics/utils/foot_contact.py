"""World-frame foot contact model: planted-foot anchors, heel rise, and floor-referenced stance metrics.

Runs on WORLD keypoints (Y-down, meters) before hip re-centring. Each planted foot keypoint is held at a
running-mean anchor, pose-model foot failures are ignored, a heel rise releases the ankle and heel while the
toe stays held, and a foot that moves is released and re-planted. The two feet are never coupled and knees
and hips are never moved. State is session-scoped.
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel

from biomechanics.utils.geometry import WORLD_UP
from biomechanics.utils.types import CocoKeypoints as CK

# Row = foot (left, right); column = keypoint within the foot (ankle, big toe, heel).
FOOT_KEYPOINT_INDICES = np.array([
    [CK.LEFT_ANKLE, CK.LEFT_FOOT_INDEX, CK.LEFT_HEEL],
    [CK.RIGHT_ANKLE, CK.RIGHT_FOOT_INDEX, CK.RIGHT_HEEL],
])
LEFT_FOOT = 0
RIGHT_FOOT = 1
ANKLE = 0
TOE = 1
HEEL = 2
NUM_FEET = 2
KEYPOINTS_PER_FOOT = 3
REAR_KEYPOINT_MASK = np.array([True, False, True])  # ankle and heel lift during a heel rise; the toe stays down
VERTICAL_AXIS = 1                                   # Y-down: rising means y decreases

WINDOW_FRAMES = 6                           # 0.2 s at 30 fps; planted decisions use window-mean residuals
RAW_HISTORY_FRAMES = 2 * WINDOW_FRAMES      # raw observation ring; holds STILL_SPAN_S at 30 fps
MIN_WINDOW_INLIERS = WINDOW_FRAMES // 2
STILL_SPAN_S = 0.35                         # a released foot is still when the medians of the two halves of this
STILL_CONFIRM_S = 0.03                      # span agree, and stays still this long (one 30 fps tick) before re-planting
MIN_STILL_SAMPLES = 2                       # accepted observations needed in each half of the span

OUTLIER_MIN_M = 0.08                        # per-frame jump beyond max(this, OUTLIER_SIGMAS * sigma) = pose failure
OUTLIER_SIGMAS = 6.0
MOVE_MIN_M = 0.03                           # window displacement that counts as the keypoint moving
MOVE_SIGMAS = 4.0
RISE_MIN_M = 0.02                           # window rise that counts as the rear foot lifting
RISE_SIGMAS = 3.0
UNCERTAINTY_SCALE_M = 0.02                  # triangulator confidence contract: conf = 1 / (1 + (u / scale)^2)
STEP_MAX_UNCERTAINTY_M = 0.025              # keypoints less certain than this (e.g. the smoother's predicted output
                                            # while it rejects a jump) cannot testify that the foot stepped
STEP_EVIDENCE_FRAMES = MIN_WINDOW_INLIERS   # a step displaces ankle and toe on every one of the last observed frames
STEP_MAX_SPEED_M_S = 3.0                    # (a brisk step peaks ~1.6 m/s; a smoothed pose failure slides to the
                                            # knee faster), in the same direction (below), keeping the foot rigid:
STEP_COHERENCE_MIN_COS = 0.5                # ankle and toe displacements pointing the same way...
STEP_LENGTH_MIN_M = 0.03                    # ...and ankle-toe / ankle-heel spacing kept within
STEP_LENGTH_SIGMAS = 2.0                    # max(STEP_LENGTH_MIN_M, STEP_LENGTH_SIGMAS * noise), on the median of
                                            # those frames and on the current frame: a pose failure that drags the
                                            # foot to the knee collapses the foot
ANCHOR_UPDATE_SIGMAS = 1.5                  # only average observations into an anchor while the window sits within
ANCHOR_UPDATE_MIN_M = 0.04                  # max(this, ANCHOR_UPDATE_SIGMAS * sigma): a poorly planted anchor must
                                            # still be able to converge
ANCHOR_CONFIRM_FRAMES = RAW_HISTORY_FRAMES  # ...and only once the foot has stayed flat this long afterwards: the frames
                                            # just before a heel rise is detected are already lifting and would creep
                                            # the floor reference upward
ANCHOR_CONVERGED_COUNT = 30                 # heel rise is only judged against an ankle anchor with this many samples
                                            # (~1 s after planting): a fresh anchor off by the rise threshold would
                                            # otherwise read as a permanent heel rise and never be corrected
ANCHOR_MAX_COUNT = 300                      # running mean becomes a ~10 s EMA: follows slow triangulation drift
SIGMA_INIT_M = 0.015
SIGMA_EMA = 0.02
SIGMA_UPDATE_SIGMAS = 4.0                   # robust noise learning: larger residuals don't update sigma
REACQUIRE_REJECTED_S = 1.0                  # a released keypoint rejected this long is re-acquired where it is: a
                                            # released foot follows real motion through its re-seeded gate, so a
                                            # persistent teleport is a pose failure and must outlast a squat bottom
DISPLACED_RELEASE_S = 1.5                   # a planted keypoint displaced or rejected this long releases its foot
REPLANT_SAME_PLACE_M = 0.05                 # ankle re-planted within this horizontal distance keeps the floor reference
REPLANT_MERGE_M = 0.03                      # per keypoint: re-plant within this distance keeps the old anchor
WARMUP_PLANTED_S = 0.5                      # both feet planted this long before FootState is valid
MAX_FRAME_GAP_S = 0.5                       # longer gaps release both feet (anchors kept)
M_TO_CM = 100.0
NAN = float("nan")
# Below this an axis has no direction (both ankle anchors on the same vertical).
DEGENERATE_LENGTH_M = 1e-6

_WORLD_UP = tuple(WORLD_UP.tolist())
# A sample's squared deviation from the mean of the WINDOW_FRAMES samples it belongs to, per axis, back to sigma^2.
_DEVIATION_VARIANCE_SCALE = WINDOW_FRAMES / (WINDOW_FRAMES - 1) / 3.0
# Noise of a per-axis median over STEP_EVIDENCE_FRAMES Gaussian samples, relative to sigma.
_MEDIAN_NOISE_FACTOR = 1.2 / math.sqrt(STEP_EVIDENCE_FRAMES)
_STEP_MIN_CONFIDENCE = 1.0 / (1.0 + (STEP_MAX_UNCERTAINTY_M / UNCERTAINTY_SCALE_M) ** 2)
_TIME_EPSILON_S = 1e-6


def _dot(vector_a: tuple[float, float, float], vector_b: tuple[float, float, float]) -> float:
    return vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1] + vector_a[2] * vector_b[2]


def _scaled_difference(
    vector_a: tuple[float, float, float], vector_b: tuple[float, float, float], scale_b: float,
) -> tuple[float, float, float]:
    return (vector_a[0] - scale_b * vector_b[0], vector_a[1] - scale_b * vector_b[1],
            vector_a[2] - scale_b * vector_b[2])


def _unit(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(_dot(vector, vector))
    if length < DEGENERATE_LENGTH_M:
        return (NAN, NAN, NAN)
    return (vector[0] / length, vector[1] / length, vector[2] / length)


def _floor_geometry(
    anchor: np.ndarray, has_anchor: np.ndarray, hip_midpoint: tuple[float, float, float] | None,
) -> tuple[float, float, float, float, float, float]:
    has = has_anchor.tolist()
    if not (has[LEFT_FOOT][ANKLE] and has[RIGHT_FOOT][ANKLE]):
        return NAN, NAN, NAN, NAN, NAN, NAN
    left, right = anchor.tolist()
    # Slope from the same keypoint type on both feet, so keypoint heights above the floor cancel.
    paired = [keypoint for keypoint in range(KEYPOINTS_PER_FOOT)
              if has[LEFT_FOOT][keypoint] and has[RIGHT_FOOT][keypoint]]
    height_difference_m = sum(left[keypoint][VERTICAL_AXIS] - right[keypoint][VERTICAL_AXIS] for keypoint in paired)
    horizontal_distance_m = sum(
        math.hypot(left[keypoint][0] - right[keypoint][0], left[keypoint][2] - right[keypoint][2])
        for keypoint in paired)
    floor_roll_rad = math.atan2(height_difference_m, horizontal_distance_m)
    ankle_offset = (left[ANKLE][0] - right[ANKLE][0], left[ANKLE][1] - right[ANKLE][1],
                    left[ANKLE][2] - right[ANKLE][2])
    ankle_horizontal_m = math.hypot(ankle_offset[0], ankle_offset[2])
    lateral_axis = _unit((ankle_offset[0], ankle_horizontal_m * math.tan(floor_roll_rad), ankle_offset[2]))
    floor_normal = _unit(_scaled_difference(_WORLD_UP, lateral_axis, _dot(_WORLD_UP, lateral_axis)))
    stance_width_m = math.sqrt(max(_dot(ankle_offset, ankle_offset) - _dot(ankle_offset, floor_normal) ** 2, 0.0))

    toe_vectors = [(foot[TOE][0] - foot[ANKLE][0], foot[TOE][1] - foot[ANKLE][1], foot[TOE][2] - foot[ANKLE][2])
                   for foot in (left, right)]
    has_toe = [has[foot][TOE] for foot in (LEFT_FOOT, RIGHT_FOOT)]
    toe_out_deg = [NAN, NAN]
    if any(has_toe):
        forward_axis = (lateral_axis[1] * floor_normal[2] - lateral_axis[2] * floor_normal[1],
                        lateral_axis[2] * floor_normal[0] - lateral_axis[0] * floor_normal[2],
                        lateral_axis[0] * floor_normal[1] - lateral_axis[1] * floor_normal[0])
        # Orient forward toward the toes so the result doesn't depend on the world frame's handedness.
        if sum(_dot(forward_axis, toe_vectors[foot]) for foot in (LEFT_FOOT, RIGHT_FOOT) if has_toe[foot]) < 0.0:
            forward_axis = (-forward_axis[0], -forward_axis[1], -forward_axis[2])
        for foot, outward_sign in ((LEFT_FOOT, 1.0), (RIGHT_FOOT, -1.0)):
            if has_toe[foot]:
                toe_out_deg[foot] = math.degrees(math.atan2(
                    outward_sign * _dot(toe_vectors[foot], lateral_axis), _dot(toe_vectors[foot], forward_axis)))

    hip_height_cm = NAN
    lateral_offset_cm = NAN
    if hip_midpoint is not None:
        # Floor under each foot = its lowest anchored keypoint (largest y).
        floor_y = [max(foot_points[keypoint][VERTICAL_AXIS] for keypoint in range(KEYPOINTS_PER_FOOT)
                       if has[foot][keypoint]) for foot, foot_points in ((LEFT_FOOT, left), (RIGHT_FOOT, right))]
        hip_offset = (hip_midpoint[0] - (left[ANKLE][0] + right[ANKLE][0]) / 2.0,
                      hip_midpoint[1] - (floor_y[LEFT_FOOT] + floor_y[RIGHT_FOOT]) / 2.0,
                      hip_midpoint[2] - (left[ANKLE][2] + right[ANKLE][2]) / 2.0)
        hip_height_cm = _dot(hip_offset, floor_normal) * M_TO_CM
        lateral_offset_cm = _dot(hip_offset, lateral_axis) * M_TO_CM

    return (stance_width_m * M_TO_CM, toe_out_deg[LEFT_FOOT], toe_out_deg[RIGHT_FOOT],
            hip_height_cm, lateral_offset_cm, math.degrees(floor_roll_rad))


class FootState(BaseModel):
    """Per-frame foot summary. NaN when not measurable.

    heel_rise_*_cm: ankle height above its planted floor reference while the rear foot is detected lifted
    (above the noise-scaled rise threshold with the toe held, or the foot released); 0.0 while flat.
    floor_roll_deg: positive when the floor under the left foot is lower. lateral_hip_offset_cm: positive
    toward the left foot. toe_out_*_deg: positive when the toe points away from the midline. Stance width
    and toe-out come from planted anchors.
    """

    valid: bool
    planted_l: bool
    planted_r: bool
    heel_rise_l_cm: float
    heel_rise_r_cm: float
    stance_width_cm: float
    toe_out_l_deg: float
    toe_out_r_deg: float
    hip_height_above_floor_cm: float
    lateral_hip_offset_cm: float
    floor_roll_deg: float


class FootContactModel:
    """Session-scoped per-foot contact model over world-frame keypoints (see module docstring)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        keypoint_shape = (NUM_FEET, KEYPOINTS_PER_FOOT)
        self._anchor = np.zeros(keypoint_shape + (3,))
        self._has_anchor = np.zeros(keypoint_shape, dtype=bool)
        self._anchor_count = np.zeros(keypoint_shape)
        self._sigma = np.full(keypoint_shape, SIGMA_INIT_M)
        self._residuals = np.zeros(keypoint_shape + (WINDOW_FRAMES, 3))
        self._residual_ok = np.zeros(keypoint_shape + (WINDOW_FRAMES,), dtype=bool)    # gate-accepted inliers
        self._residual_seen = np.zeros(keypoint_shape + (WINDOW_FRAMES,), dtype=bool)  # every observation
        self._residual_head = 0
        self._raw = np.zeros(keypoint_shape + (RAW_HISTORY_FRAMES, 3))
        self._raw_ok = np.zeros(keypoint_shape + (RAW_HISTORY_FRAMES,), dtype=bool)
        self._raw_seen = np.zeros(keypoint_shape + (RAW_HISTORY_FRAMES,), dtype=bool)
        self._raw_time = np.full(RAW_HISTORY_FRAMES, -np.inf)
        self._flat_ok = np.zeros(keypoint_shape + (ANCHOR_CONFIRM_FRAMES,), dtype=bool)
        self._raw_head = 0
        self._last_out = np.zeros(keypoint_shape + (3,))
        self._has_out = np.zeros(keypoint_shape, dtype=bool)
        self._rejected_since = np.full(keypoint_shape, NAN)
        self._displaced_since = np.full(keypoint_shape, NAN)
        self._planted = np.zeros(NUM_FEET, dtype=bool)
        self._heel_mode = np.zeros(NUM_FEET, dtype=bool)
        self._still_since: list[float | None] = [None, None]
        self._planted_time_s = [0.0, 0.0]
        self._valid = False
        self._last_timestamp: float | None = None
        self._last_points: np.ndarray | None = None
        self._last_state: FootState | None = None

    def update(
        self, points: np.ndarray, confidences: np.ndarray, timestamp: float,
    ) -> tuple[np.ndarray, FootState]:
        """Stabilize foot keypoints (conf > 0 only) and summarize contact; all other keypoints pass through."""
        dt_s = 0.0
        if self._last_timestamp is not None and self._last_state is not None:
            dt_s = timestamp - self._last_timestamp
            if dt_s <= 0.0:
                return self._last_points.copy(), self._last_state
            if dt_s > MAX_FRAME_GAP_S:
                self._release_after_gap()
                dt_s = 0.0
        self._last_timestamp = timestamp

        observations = points[FOOT_KEYPOINT_INDICES]
        foot_confidences = confidences[FOOT_KEYPOINT_INDICES]
        observed = foot_confidences > 0.0

        # Gate: planted keypoints against their anchor, released or heel-rising ones against their last output.
        rear_released = self._heel_mode[:, None] & REAR_KEYPOINT_MASK
        anchor_gated = self._planted[:, None] & ~rear_released & self._has_anchor
        reference = np.where(anchor_gated[..., None], self._anchor, self._last_out)
        jump = observations - reference
        jump_m = np.sqrt(np.einsum("fki,fki->fk", jump, jump))
        within_gate = jump_m < np.maximum(OUTLIER_MIN_M, OUTLIER_SIGMAS * self._sigma)
        reacquire = ~anchor_gated & (timestamp - self._rejected_since >= REACQUIRE_REJECTED_S)
        accepted = observed & (within_gate | reacquire | ~(anchor_gated | self._has_out))
        # Only accepted observations can plant a foot: a smoother-smeared pose failure must not become an anchor.
        # The slot being overwritten holds the observation from ANCHOR_CONFIRM_FRAMES ago, the anchor candidate.
        slot = self._raw_head
        previous_slot = (slot - 1) % RAW_HISTORY_FRAMES
        candidate = self._raw[:, :, slot].copy()
        candidate_flat = self._flat_ok[:, :, slot].copy()
        travel = observations - self._raw[:, :, previous_slot]
        travel_m = np.sqrt(np.einsum("fki,fki->fk", travel, travel))
        too_fast = (self._raw_seen[:, :, previous_slot] & (travel_m > STEP_MAX_SPEED_M_S * max(
            timestamp - self._raw_time[previous_slot], _TIME_EPSILON_S))).tolist()
        self._raw[:, :, slot] = observations
        self._raw_ok[:, :, slot] = accepted
        self._raw_seen[:, :, slot] = observed
        self._raw_time[slot] = timestamp
        self._raw_head = (slot + 1) % RAW_HISTORY_FRAMES

        planted_anchored = self._planted[:, None] & self._has_anchor
        self._residuals[:, :, self._residual_head] = observations - self._anchor
        self._residual_ok[:, :, self._residual_head] = accepted & planted_anchored
        self._residual_seen[:, :, self._residual_head] = observed & planted_anchored
        self._residual_head = (self._residual_head + 1) % WINDOW_FRAMES
        inlier_count = self._residual_ok.sum(axis=2)
        window_mean = (np.einsum("fkwi,fkw->fki", self._residuals, self._residual_ok)
                       / np.maximum(inlier_count, 1)[..., None])
        window_ready = inlier_count >= MIN_WINDOW_INLIERS
        window_shift_m = np.sqrt(np.einsum("fki,fki->fk", window_mean, window_mean))
        # Noise is learned from deviations about the window mean, so slow motion (a lifting heel, triangulation
        # drift) doesn't inflate sigma and with it the very thresholds that should detect the motion.
        deviation = observations - self._anchor - window_mean
        deviation_m = np.sqrt(np.einsum("fki,fki->fk", deviation, deviation))
        noise_sample = (anchor_gated & accepted & (inlier_count >= WINDOW_FRAMES)
                        & (deviation_m < SIGMA_UPDATE_SIGMAS * self._sigma))
        self._sigma = np.where(noise_sample, np.sqrt(
            (1.0 - SIGMA_EMA) * self._sigma ** 2 + SIGMA_EMA * deviation_m ** 2 * _DEVIATION_VARIANCE_SCALE),
            self._sigma)
        window_noise_m = self._sigma / math.sqrt(WINDOW_FRAMES)
        move_threshold_m = np.maximum(MOVE_MIN_M, MOVE_SIGMAS * window_noise_m)
        rise_threshold_m = np.maximum(RISE_MIN_M, RISE_SIGMAS * window_noise_m)
        moved = window_ready & (window_shift_m > move_threshold_m)
        rise_ratio = np.where(window_ready, -window_mean[:, :, VERTICAL_AXIS] / rise_threshold_m, 0.0)
        # Step detection sees every observation, gate-rejected ones included: a fast step leaves the gate at once.
        seen_count = self._residual_seen.sum(axis=2)
        seen_mean = (np.einsum("fkwi,fkw->fki", self._residuals, self._residual_seen)
                     / np.maximum(seen_count, 1)[..., None])
        seen_moved = ((seen_count >= MIN_WINDOW_INLIERS)
                      & (np.sqrt(np.einsum("fki,fki->fk", seen_mean, seen_mean)) > move_threshold_m)).tolist()

        changed = [False, False]
        for foot in (LEFT_FOOT, RIGHT_FOOT):
            if self._planted[foot]:
                stepped = self._coherent_step(
                    foot, observations, foot_confidences, seen_moved, too_fast, move_threshold_m)
                changed[foot] = self._planted_step(foot, moved[foot], rise_ratio[foot], stepped)
            else:
                changed[foot] = self._released_step(foot, move_threshold_m[foot], rise_threshold_m[foot], timestamp)
        stable = self._planted & ~np.array(changed)

        # Persistent displacement (toe pivot, heel swing, long pose failure) re-plants the foot.
        rear_released = self._heel_mode[:, None] & REAR_KEYPOINT_MASK
        counting = stable[:, None] & ~rear_released & self._has_anchor
        displaced = counting & (moved | (observed & ~accepted))
        self._displaced_since = np.where(displaced & np.isnan(self._displaced_since), timestamp,
                                         np.where(counting & observed & ~displaced, NAN, self._displaced_since))
        displaced_for_s = np.where(np.isnan(self._displaced_since), -np.inf, timestamp - self._displaced_since)
        for foot in np.flatnonzero(stable & (displaced_for_s.max(axis=1) > DISPLACED_RELEASE_S)):
            self._release(foot)
            stable[foot] = False

        # A foot released this frame is where its observations say: they re-seed the gate reference and the
        # stillness history instead of being judged against the anchor it just left.
        released_now = np.array(changed) & ~self._planted
        accepted |= released_now[:, None] & observed
        self._raw_ok[:, :, slot] = accepted
        rejected = observed & ~accepted
        self._rejected_since = np.where(rejected & np.isnan(self._rejected_since), timestamp,
                                        np.where(accepted, NAN, self._rejected_since))

        held_flat = stable[:, None] & ~rear_released & self._has_anchor
        update_anchor = candidate_flat & held_flat
        self._anchor_count = np.where(update_anchor, np.minimum(self._anchor_count + 1.0, ANCHOR_MAX_COUNT),
                                      self._anchor_count)
        self._anchor = np.where(update_anchor[..., None], self._anchor + (candidate - self._anchor)
                                / np.maximum(self._anchor_count, 1.0)[..., None], self._anchor)
        self._flat_ok[:, :, slot] = (held_flat & accepted & window_ready
                                     & (window_shift_m < np.maximum(ANCHOR_UPDATE_MIN_M,
                                                                    ANCHOR_UPDATE_SIGMAS * self._sigma)))

        hold = self._planted[:, None] & ~(self._heel_mode[:, None] & REAR_KEYPOINT_MASK) & self._has_anchor
        fresh = np.where(accepted[..., None], observations, self._last_out)
        foot_out = np.where(hold[..., None], self._anchor, fresh)
        self._last_out = np.where(observed[..., None], foot_out, self._last_out)
        self._has_out |= observed
        output_points = points.copy()
        output_points[FOOT_KEYPOINT_INDICES[observed]] = foot_out[observed]

        for foot in (LEFT_FOOT, RIGHT_FOOT):
            if self._planted[foot]:
                self._planted_time_s[foot] += dt_s
        self._valid = self._valid or min(self._planted_time_s) >= WARMUP_PLANTED_S

        state = self._build_state(points, confidences, window_mean, window_ready, stable)
        self._last_points = output_points.copy()
        self._last_state = state
        return output_points, state

    def _coherent_step(
        self, foot: int, observations: np.ndarray, foot_confidences: np.ndarray, seen_moved: list[list[bool]],
        too_fast: list[list[bool]], move_threshold_m: np.ndarray,
    ) -> bool:
        # Cheap pre-check on window means; the evidence below is only computed when something moved.
        if not (seen_moved[foot][ANKLE] and seen_moved[foot][TOE]):
            return False
        if too_fast[foot][ANKLE] or too_fast[foot][TOE]:
            return False
        head = self._residual_head
        recent = [(head - 1 - back) % WINDOW_FRAMES for back in range(STEP_EVIDENCE_FRAMES)]
        usable = (self._has_anchor[foot] & (foot_confidences[foot] >= _STEP_MIN_CONFIDENCE)
                  & self._residual_seen[foot][:, recent].all(axis=1)).tolist()
        if not (usable[ANKLE] and usable[TOE]):
            return False
        residuals = self._residuals[foot][:, recent]                       # (keypoint, frame, xyz)
        displacement_m = np.sqrt(np.einsum("kfi,kfi->kf", residuals, residuals)).min(axis=1)
        if displacement_m[ANKLE] <= move_threshold_m[foot, ANKLE] or displacement_m[TOE] <= move_threshold_m[foot, TOE]:
            return False
        shifts = np.median(residuals, axis=1)                             # per-axis median, a real sample of 3
        ankle_norm_m = float(np.linalg.norm(shifts[ANKLE]))
        toe_norm_m = float(np.linalg.norm(shifts[TOE]))
        if float(shifts[ANKLE] @ shifts[TOE]) < STEP_COHERENCE_MIN_COS * ankle_norm_m * toe_norm_m:
            return False
        anchor = self._anchor[foot]
        sigma = self._sigma[foot]
        for keypoint in (TOE, HEEL):
            if not usable[keypoint]:
                continue
            anchor_length_m = float(np.linalg.norm(anchor[keypoint] - anchor[ANKLE]))
            pair_sigma_m = math.hypot(sigma[keypoint], sigma[ANKLE])
            window_length_m = float(np.linalg.norm(
                (anchor[keypoint] + shifts[keypoint]) - (anchor[ANKLE] + shifts[ANKLE])))
            if abs(window_length_m - anchor_length_m) > max(
                    STEP_LENGTH_MIN_M, STEP_LENGTH_SIGMAS * pair_sigma_m * _MEDIAN_NOISE_FACTOR):
                return False
            frame_length_m = float(np.linalg.norm(observations[foot, keypoint] - observations[foot, ANKLE]))
            if abs(frame_length_m - anchor_length_m) > max(STEP_LENGTH_MIN_M, STEP_LENGTH_SIGMAS * pair_sigma_m):
                return False
        return True

    def _planted_step(self, foot: int, moved: np.ndarray, rise_ratio: np.ndarray, stepped: bool) -> bool:
        # Only a coherent step releases a planted foot: inlier-mean movement tests are fooled by pose failures,
        # whose rejected frames shrink the window to the few (noisy, already displaced) frames before them.
        if stepped:
            self._release(foot)
            return True
        toe_moved = bool(moved[TOE])
        ankle_rising = (float(rise_ratio[ANKLE]) > 1.0
                        and self._anchor_count[foot, ANKLE] >= ANCHOR_CONVERGED_COUNT)
        heel_mode = ankle_rising and not toe_moved
        if heel_mode and not self._heel_mode[foot]:
            self._flat_ok[foot, REAR_KEYPOINT_MASK] = False
        self._heel_mode[foot] = heel_mode
        if not heel_mode:
            self._anchor_missing_keypoints(foot)
        return False

    def _released_step(
        self, foot: int, move_threshold_m: np.ndarray, rise_threshold_m: np.ndarray, timestamp: float,
    ) -> bool:
        age_s = timestamp - self._raw_time
        in_span = age_s <= STILL_SPAN_S
        recent = in_span & (age_s < STILL_SPAN_S / 2.0)
        previous = in_span & ~recent
        complete = self._raw_ok[foot][:, in_span].all(axis=1)
        if (not (complete[ANKLE] and complete[TOE]) or recent.sum() < MIN_STILL_SAMPLES
                or previous.sum() < MIN_STILL_SAMPLES):
            self._still_since[foot] = None
            return False
        keypoints = [ANKLE, TOE, HEEL] if complete[HEEL] else [ANKLE, TOE]
        history = self._raw[foot][keypoints]
        previous_median = np.median(history[:, previous], axis=1)
        recent_median = np.median(history[:, recent], axis=1)
        still = bool(np.all(np.linalg.norm(recent_median - previous_median, axis=1) <= move_threshold_m[keypoints]))
        if not still:
            self._still_since[foot] = None
            return False
        if self._still_since[foot] is None:
            self._still_since[foot] = timestamp
        if timestamp - self._still_since[foot] < STILL_CONFIRM_S:
            return False
        self._plant(foot, keypoints, np.median(history[:, in_span], axis=1), int(in_span.sum()), rise_threshold_m)
        return True

    def _plant(
        self, foot: int, keypoints: list[int], window_positions: np.ndarray, sample_count: int,
        rise_threshold_m: np.ndarray,
    ) -> None:
        positions = dict(zip(keypoints, window_positions))
        ankle_shift = positions[ANKLE] - self._anchor[foot, ANKLE]
        same_place = bool(self._has_anchor[foot, ANKLE]) and math.hypot(
            ankle_shift[0], ankle_shift[2]) < REPLANT_SAME_PLACE_M
        # Re-planted with the rear foot still up: keep the floor anchors and resume in heel-rise mode.
        rear_raised = same_place and any(
            self._has_anchor[foot, keypoint]
            and self._anchor[foot, keypoint, VERTICAL_AXIS] - positions[keypoint][VERTICAL_AXIS]
            > rise_threshold_m[keypoint]
            for keypoint in keypoints if REAR_KEYPOINT_MASK[keypoint])
        for keypoint, position in positions.items():
            if REAR_KEYPOINT_MASK[keypoint] and rear_raised:
                continue
            keep = same_place and bool(self._has_anchor[foot, keypoint]) and float(
                np.linalg.norm(position - self._anchor[foot, keypoint])) < REPLANT_MERGE_M
            if not keep:
                self._anchor[foot, keypoint] = position
                self._anchor_count[foot, keypoint] = sample_count
                self._has_anchor[foot, keypoint] = True
        self._planted[foot] = True
        self._heel_mode[foot] = rear_raised
        self._still_since[foot] = None
        self._residual_ok[foot] = False
        self._residual_seen[foot] = False
        self._flat_ok[foot] = False
        self._displaced_since[foot] = NAN

    def _anchor_missing_keypoints(self, foot: int) -> None:
        # A keypoint first seen after its foot planted (typically a heel) gets its anchor once it has settled.
        for keypoint, has_anchor in enumerate(self._has_anchor[foot].tolist()):
            if not has_anchor and self._raw_ok[foot, keypoint].all():
                self._anchor[foot, keypoint] = np.median(self._raw[foot, keypoint], axis=0)
                self._anchor_count[foot, keypoint] = RAW_HISTORY_FRAMES
                self._has_anchor[foot, keypoint] = True

    def _release(self, foot: int) -> None:
        self._planted[foot] = False
        self._heel_mode[foot] = False
        self._still_since[foot] = None
        self._residual_ok[foot] = False
        self._residual_seen[foot] = False
        self._flat_ok[foot] = False
        self._displaced_since[foot] = NAN
        self._rejected_since[foot] = NAN

    def _release_after_gap(self) -> None:
        for foot in (LEFT_FOOT, RIGHT_FOOT):
            self._release(foot)
        self._raw_ok[:] = False
        self._raw_seen[:] = False
        self._raw_time[:] = -np.inf
        self._has_out[:] = False

    def _build_state(
        self, points: np.ndarray, confidences: np.ndarray, window_mean: np.ndarray,
        window_ready: np.ndarray, stable: np.ndarray,
    ) -> FootState:
        heel_rise_cm = [NAN, NAN]
        for foot in (LEFT_FOOT, RIGHT_FOOT):
            if not self._has_anchor[foot, ANKLE]:
                continue
            if self._planted[foot] and not self._heel_mode[foot]:
                heel_rise_cm[foot] = 0.0
                continue
            if not self._planted[foot] and not self._heel_mode[foot]:
                # A foot that stepped away has no floor reference until it re-plants.
                continue
            anchor_y = self._anchor[foot, ANKLE, VERTICAL_AXIS]
            if stable[foot] and window_ready[foot, ANKLE]:
                current_y = anchor_y + window_mean[foot, ANKLE, VERTICAL_AXIS]
            elif self._has_out[foot, ANKLE]:
                current_y = self._last_out[foot, ANKLE, VERTICAL_AXIS]
            else:
                continue
            heel_rise_cm[foot] = float(anchor_y - current_y) * M_TO_CM

        hip_midpoint = None
        if confidences[CK.LEFT_HIP] > 0.0 and confidences[CK.RIGHT_HIP] > 0.0:
            hip_l, hip_r = points[CK.LEFT_HIP].tolist(), points[CK.RIGHT_HIP].tolist()
            hip_midpoint = ((hip_l[0] + hip_r[0]) / 2.0, (hip_l[1] + hip_r[1]) / 2.0, (hip_l[2] + hip_r[2]) / 2.0)
        (stance_width_cm, toe_out_l_deg, toe_out_r_deg, hip_height_cm, lateral_offset_cm,
         floor_roll_deg) = _floor_geometry(self._anchor, self._has_anchor, hip_midpoint)

        return FootState(
            valid=self._valid,
            planted_l=bool(self._planted[LEFT_FOOT]),
            planted_r=bool(self._planted[RIGHT_FOOT]),
            heel_rise_l_cm=heel_rise_cm[LEFT_FOOT],
            heel_rise_r_cm=heel_rise_cm[RIGHT_FOOT],
            stance_width_cm=stance_width_cm,
            toe_out_l_deg=toe_out_l_deg,
            toe_out_r_deg=toe_out_r_deg,
            hip_height_above_floor_cm=hip_height_cm,
            lateral_hip_offset_cm=lateral_offset_cm,
            floor_roll_deg=floor_roll_deg,
        )
