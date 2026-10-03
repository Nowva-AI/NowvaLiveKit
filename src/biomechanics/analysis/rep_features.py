"""Per-frame squat samples and the per-rep features both fault rules and diagnosis read.

One module measures what a rep looked like, so the intra-set cue and the post-set
recap can never disagree about it. Input frame: X = subject's left, Y = down,
+Z = subject's back. Sample heights are Y-up cm above the ankle midpoint.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from pydantic import BaseModel

from biomechanics.diagnosis.keypoint_corrector import compute_com_ground
from biomechanics.diagnosis.lean_model import MIDFOOT_FRACTION_OF_ANKLE_TO_TOE
from biomechanics.diagnosis.types import RepTrajectorySample
from biomechanics.utils.types import CocoKeypoints as CK
from biomechanics.utils.types import JointAngles, Skeleton3D

NAN = float("nan")

# Keypoints below this confidence are treated as missing — the IK solver's floor.
MIN_KEYPOINT_CONFIDENCE = 0.1
# Knee tracking is only read where the feet were seen this well (the valgus
# estimators' confidence includes the toes); unseen feet mean no reading.
MIN_FOOT_CONFIDENCE = 0.3
# Thigh shorter than this (m) cannot normalize depth.
MIN_FEMUR_M = 0.20
MIN_FOOT_M = 0.05
MIN_ANKLE_SEPARATION_M = 0.05

# Frames within this many femur lengths of the deepest hip position are "loaded":
# the bottom, where position faults are judged.
LOADED_WINDOW_RATIO = 0.25
# Ascent frames this close to the bottom still count for knee tracking; nearer
# standing the knees-over-toes plane degenerates.
TRACKING_WINDOW_RATIO = 0.6
DEEPEST_PERCENTILE = 10.0
WORST_PERCENTILE = 90.0
# Hip shoot is read where the hip has recovered this fraction of its bottom-to-top rise.
HIP_SHOOT_RECOVERY_FRACTION = 1.0 / 3.0
# Descent initiation is read over this fraction of the hip's drop.
INITIATION_DESCENT_FRACTION = 0.2
MIN_INITIATION_KNEE_CHANGE_DEG = 2.0
# Concentric phase ends when the shoulders are back within this fraction of their range.
CONCENTRIC_END_FRACTION = 0.05
# Hip shift is read from the median of the rep's first frames (not one noisy
# frame) to a sustained level — the median of the frames at or past this
# percentile of sideways travel — so a single jittery frame cannot make a shift.
SHIFT_START_FRAMES = 5
SHIFT_SUSTAINED_PERCENTILE = 80.0
# The load cannot sit further than this many ankle-to-toe lengths off midfoot
# while the athlete stays on their feet; beyond it the reading is depth noise.
MAX_PLAUSIBLE_BALANCE_RATIO = 1.0
# A squat setup is never staggered by more than a foot length.
MAX_PLAUSIBLE_STAGGER_RATIO = 1.0


class SetupSnapshot(BaseModel):
    """The most upright frame between reps: the top the athlete set up from."""
    hip_height_cm: float = NAN
    stagger_ratio: float = NAN
    toe_out_l_deg: float = NAN
    toe_out_r_deg: float = NAN


class RepFeatures(BaseModel):
    """Everything a coach would read off one rep. NaN = not measurable this rep.

    Signs: depth_ratio > 0 means the hip stayed above the knee (0 = parallel);
    hip_shoot_deg > 0 means the chest dropped out of the hole; hip_shift_ratio > 0
    means the hips moved toward the right foot; balance_ratio > 0 means the load
    sat ahead of midfoot; stagger_ratio > 0 means the left foot was ahead;
    toe-out > 0 means the toes point away from the midline.
    """
    rep_number: int
    sample_count: int = 0
    depth_ratio: float = NAN
    depth_cm_above_parallel: float = NAN
    knee_flexion_max: float = NAN
    trunk_pitch_bottom: float = NAN
    valgus_l: float = NAN
    valgus_r: float = NAN
    valgus_peak_phase_l: str = ""
    valgus_peak_phase_r: str = ""
    hip_shoot_deg: float = NAN
    hip_shift_ratio: float = NAN
    hip_shift_peak_phase: str = ""
    balance_ratio: float = NAN
    heel_rise_max_cm_l: float = NAN
    heel_rise_max_cm_r: float = NAN
    descent_time_s: float = NAN
    ascent_time_s: float = NAN
    concentric_velocity_mps: float = NAN
    velocity_loss_pct: float = NAN
    knee_flexion_asym_bottom: float = NAN
    initiation_ratio: float = NAN
    neck_flexion_bottom: float = NAN
    dorsiflexion_max_l: float = NAN
    dorsiflexion_max_r: float = NAN
    hip_flexion_max_l: float = NAN
    hip_flexion_max_r: float = NAN
    bar_detected: bool = False
    lockout_deficit_ratio: float = NAN
    stagger_ratio: float = NAN
    toe_out_l_deg: float = NAN
    toe_out_r_deg: float = NAN


def _point(kpts: np.ndarray, confidences: np.ndarray, index: int) -> np.ndarray | None:
    if index >= len(kpts) or confidences[index] < MIN_KEYPOINT_CONFIDENCE:
        return None
    return kpts[index]


def _midpoint(first: np.ndarray | None, second: np.ndarray | None) -> np.ndarray | None:
    if first is None or second is None:
        return None
    return (first + second) / 2.0


def _horizontal(vector: np.ndarray) -> np.ndarray:
    return np.array([vector[0], 0.0, vector[2]])


def _unit(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return None
    return vector / norm


def _height_cm(point: np.ndarray | None, ankle_mid: np.ndarray | None) -> float:
    if point is None or ankle_mid is None:
        return NAN
    return float(ankle_mid[1] - point[1]) * 100.0


def _foot_frame(
    kpts: np.ndarray, confidences: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    """(lateral unit L→R, forward unit, midfoot point, ankle→toe length) on the ground plane.

    The athlete's axes come from the hip line, which a staggered or uneven
    stance does not rotate; the ankle line is the fallback without hips.
    """
    l_ankle = _point(kpts, confidences, CK.LEFT_ANKLE)
    r_ankle = _point(kpts, confidences, CK.RIGHT_ANKLE)
    l_toe = _point(kpts, confidences, CK.LEFT_FOOT_INDEX)
    r_toe = _point(kpts, confidences, CK.RIGHT_FOOT_INDEX)
    if l_ankle is None or r_ankle is None or l_toe is None or r_toe is None:
        return None

    l_hip = _point(kpts, confidences, CK.LEFT_HIP)
    r_hip = _point(kpts, confidences, CK.RIGHT_HIP)
    lateral = None
    if l_hip is not None and r_hip is not None:
        lateral = _unit(_horizontal(r_hip - l_hip))
    if lateral is None:
        lateral = _unit(_horizontal(r_ankle - l_ankle))
    foot_l = _horizontal(l_toe - l_ankle)
    foot_r = _horizontal(r_toe - r_ankle)
    foot_len = (float(np.linalg.norm(foot_l)) + float(np.linalg.norm(foot_r))) / 2.0
    if lateral is None or foot_len < MIN_FOOT_M:
        return None

    forward = np.cross(lateral, np.array([0.0, -1.0, 0.0]))
    if float(np.dot(forward, foot_l + foot_r)) < 0.0:
        forward = -forward

    midfoot = (
        l_ankle + MIDFOOT_FRACTION_OF_ANKLE_TO_TOE * foot_l
        + r_ankle + MIDFOOT_FRACTION_OF_ANKLE_TO_TOE * foot_r
    ) / 2.0
    return lateral, forward, midfoot, foot_len


def _signed_toe_out_deg(foot: np.ndarray, forward: np.ndarray, outward: np.ndarray) -> float:
    angle = math.degrees(math.atan2(float(np.dot(foot, outward)), float(np.dot(foot, forward))))
    return angle


def _neck_flexion_deg(
    kpts: np.ndarray, confidences: np.ndarray, forward: np.ndarray | None
) -> float:
    shoulder_mid = _midpoint(
        _point(kpts, confidences, CK.LEFT_SHOULDER), _point(kpts, confidences, CK.RIGHT_SHOULDER)
    )
    hip_mid = _midpoint(_point(kpts, confidences, CK.LEFT_HIP), _point(kpts, confidences, CK.RIGHT_HIP))
    ear_mid = _midpoint(_point(kpts, confidences, CK.LEFT_EAR), _point(kpts, confidences, CK.RIGHT_EAR))
    if shoulder_mid is None or hip_mid is None or ear_mid is None or forward is None:
        return NAN

    up = np.array([0.0, -1.0, 0.0])
    trunk = shoulder_mid - hip_mid
    neck = ear_mid - shoulder_mid
    # Signed angle in the sagittal plane: positive when the head tips forward
    # of the trunk line (chin toward chest), negative when it cranks back.
    trunk_angle = math.atan2(float(np.dot(trunk, forward)), float(np.dot(trunk, up)))
    neck_angle = math.atan2(float(np.dot(neck, forward)), float(np.dot(neck, up)))
    return math.degrees(neck_angle - trunk_angle)


def build_frame_sample(
    skeleton_3d: Skeleton3D,
    angles: JointAngles,
    phase: str = "",
    femur_m: float | None = None,
    heel_rise_l_cm: float = NAN,
    heel_rise_r_cm: float = NAN,
    bar_detected: bool = False,
) -> RepTrajectorySample:
    """Measure one frame. Missing keypoints make their measurements NaN, never 0."""
    kpts = skeleton_3d.to_numpy()
    confidences = np.array([point.confidence for point in skeleton_3d.keypoints])

    l_hip = _point(kpts, confidences, CK.LEFT_HIP)
    r_hip = _point(kpts, confidences, CK.RIGHT_HIP)
    l_knee = _point(kpts, confidences, CK.LEFT_KNEE)
    r_knee = _point(kpts, confidences, CK.RIGHT_KNEE)
    l_ankle = _point(kpts, confidences, CK.LEFT_ANKLE)
    r_ankle = _point(kpts, confidences, CK.RIGHT_ANKLE)
    ankle_mid = _midpoint(l_ankle, r_ankle)
    hip_mid = _midpoint(l_hip, r_hip)
    knee_mid = _midpoint(l_knee, r_knee)
    shoulder_mid = _midpoint(
        _point(kpts, confidences, CK.LEFT_SHOULDER), _point(kpts, confidences, CK.RIGHT_SHOULDER)
    )

    depth_ratio = NAN
    if hip_mid is not None and knee_mid is not None:
        thigh_m = femur_m
        if thigh_m is None or thigh_m < MIN_FEMUR_M:
            thigh_m = float(np.linalg.norm(l_hip - l_knee) + np.linalg.norm(r_hip - r_knee)) / 2.0
        if thigh_m >= MIN_FEMUR_M:
            # Y-down: the hip is above the knee when its y is smaller.
            depth_ratio = float(np.clip((knee_mid[1] - hip_mid[1]) / thigh_m, -1.0, 1.0))

    hip_lateral_ratio = NAN
    if l_ankle is not None and r_ankle is not None and hip_mid is not None and ankle_mid is not None:
        ankle_axis = _horizontal(r_ankle - l_ankle)
        separation = float(np.linalg.norm(ankle_axis))
        if separation >= MIN_ANKLE_SEPARATION_M:
            hip_lateral_ratio = float(np.dot(_horizontal(hip_mid - ankle_mid), ankle_axis / separation)) / separation

    balance_ratio = NAN
    forward = None
    foot_frame = _foot_frame(kpts, confidences)
    if foot_frame is not None:
        _, forward, midfoot, foot_len = foot_frame
        load_point = None
        if bar_detected and shoulder_mid is not None:
            load_point = shoulder_mid
        elif confidences[: CK.RIGHT_ANKLE + 1].min() >= MIN_KEYPOINT_CONFIDENCE:
            com_xz = compute_com_ground(kpts, foot_len)
            load_point = np.array([com_xz[0], 0.0, com_xz[1]])
        if load_point is not None:
            balance_ratio = float(np.dot(_horizontal(load_point - midfoot), forward)) / foot_len
            if abs(balance_ratio) > MAX_PLAUSIBLE_BALANCE_RATIO:
                balance_ratio = NAN

    valgus_l = angles.knee_valgus_l if angles.foot_confidence_l >= MIN_FOOT_CONFIDENCE else NAN
    valgus_r = angles.knee_valgus_r if angles.foot_confidence_r >= MIN_FOOT_CONFIDENCE else NAN

    return RepTrajectorySample(
        timestamp=angles.timestamp,
        phase=phase,
        trunk_pitch=180.0 - angles.trunk_flexion,
        knee_valgus_l=valgus_l,
        knee_valgus_r=valgus_r,
        hip_y_l=_height_cm(l_hip, ankle_mid),
        hip_y_r=_height_cm(r_hip, ankle_mid),
        knee_y_l=_height_cm(l_knee, ankle_mid),
        knee_y_r=_height_cm(r_knee, ankle_mid),
        hip_height_cm=_height_cm(hip_mid, ankle_mid),
        shoulder_height_cm=_height_cm(shoulder_mid, ankle_mid),
        depth_ratio=depth_ratio,
        hip_lateral_ratio=hip_lateral_ratio,
        balance_ratio=balance_ratio,
        heel_rise_l_cm=heel_rise_l_cm,
        heel_rise_r_cm=heel_rise_r_cm,
        knee_flexion_l=angles.knee_flexion_l,
        knee_flexion_r=angles.knee_flexion_r,
        hip_flexion_l=angles.hip_flexion_l,
        hip_flexion_r=angles.hip_flexion_r,
        dorsiflexion_l=angles.ankle_dorsiflexion_l,
        dorsiflexion_r=angles.ankle_dorsiflexion_r,
        neck_flexion_deg=_neck_flexion_deg(kpts, confidences, forward),
        bar_detected=bar_detected,
    )


def build_setup_snapshot(skeleton_3d: Skeleton3D) -> SetupSnapshot:
    """Foot placement and standing height from an upright frame between reps."""
    kpts = skeleton_3d.to_numpy()
    confidences = np.array([point.confidence for point in skeleton_3d.keypoints])
    l_ankle = _point(kpts, confidences, CK.LEFT_ANKLE)
    r_ankle = _point(kpts, confidences, CK.RIGHT_ANKLE)
    ankle_mid = _midpoint(l_ankle, r_ankle)
    hip_mid = _midpoint(_point(kpts, confidences, CK.LEFT_HIP), _point(kpts, confidences, CK.RIGHT_HIP))

    snapshot = SetupSnapshot(hip_height_cm=_height_cm(hip_mid, ankle_mid))
    foot_frame = _foot_frame(kpts, confidences)
    if foot_frame is None:
        return snapshot

    lateral, forward, _, foot_len = foot_frame
    l_toe = kpts[CK.LEFT_FOOT_INDEX]
    r_toe = kpts[CK.RIGHT_FOOT_INDEX]
    stagger = float(np.dot(_horizontal(l_ankle - r_ankle), forward)) / foot_len
    snapshot.stagger_ratio = stagger if abs(stagger) <= MAX_PLAUSIBLE_STAGGER_RATIO else NAN
    # Outward is toward the athlete's left for the left foot, right for the right.
    snapshot.toe_out_l_deg = _signed_toe_out_deg(_horizontal(l_toe - l_ankle), forward, -lateral)
    snapshot.toe_out_r_deg = _signed_toe_out_deg(_horizontal(r_toe - r_ankle), forward, lateral)
    return snapshot


def _finite(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def _percentile(values: Sequence[float], percentile: float) -> float:
    finite = _finite(values)
    if finite.size == 0:
        return NAN
    return float(np.percentile(finite, percentile))


def _median(values: Sequence[float]) -> float:
    finite = _finite(values)
    if finite.size == 0:
        return NAN
    return float(np.median(finite))


def _peak_phase(index: int, deepest_index: int, loaded: set[int]) -> str:
    if index in loaded:
        return "bottom"
    return "ascent" if index > deepest_index else "descent"


def _valgus_side(
    samples: list[RepTrajectorySample], window: list[int], deepest_index: int, loaded: set[int], side: str
) -> tuple[float, str]:
    values = [getattr(samples[i], f"knee_valgus_{side}") for i in window]
    p90 = _percentile(values, WORST_PERCENTILE)
    if math.isnan(p90):
        return NAN, ""
    finite = [(value, i) for value, i in zip(values, window) if math.isfinite(value)]
    _, peak_index = max(finite)
    return p90, _peak_phase(peak_index, deepest_index, loaded)


def _hip_shoot_deg(samples: list[RepTrajectorySample], deepest_index: int) -> float:
    heights = [sample.hip_height_cm for sample in samples]
    bottom_height = heights[deepest_index]
    start_heights = _finite(heights[:deepest_index])
    if math.isnan(bottom_height) or start_heights.size == 0:
        return NAN
    target = bottom_height + (float(start_heights.max()) - bottom_height) * HIP_SHOOT_RECOVERY_FRACTION
    for i in range(deepest_index + 1, len(samples)):
        if math.isfinite(heights[i]) and heights[i] >= target:
            bottom_pitch = _median([s.trunk_pitch for s in samples[max(0, deepest_index - 1):deepest_index + 2]])
            return samples[i].trunk_pitch - bottom_pitch
    return NAN


def _hip_shift(
    samples: list[RepTrajectorySample], deepest_index: int, loaded: set[int]
) -> tuple[float, str]:
    lateral = [sample.hip_lateral_ratio for sample in samples]
    start_values = _finite([value for value in lateral if math.isfinite(value)][:SHIFT_START_FRAMES])
    if start_values.size == 0:
        return NAN, ""
    start = float(np.median(start_values))
    window = [
        (value - start, i)
        for i, value in enumerate(lateral)
        if math.isfinite(value) and (i in loaded or i > deepest_index)
    ]
    if not window:
        return NAN, ""
    deviations = np.array([deviation for deviation, _ in window])
    cutoff = np.percentile(np.abs(deviations), SHIFT_SUSTAINED_PERCENTILE)
    sustained = deviations[np.abs(deviations) >= cutoff]
    _, peak_index = max(window, key=lambda entry: abs(entry[0]))
    return float(np.median(sustained)), _peak_phase(peak_index, deepest_index, loaded)


def _concentric_velocity_mps(samples: list[RepTrajectorySample], deepest_index: int) -> float:
    shoulder = [sample.shoulder_height_cm for sample in samples]
    bottom = shoulder[deepest_index]
    start = _finite(shoulder[:deepest_index])
    if math.isnan(bottom) or start.size == 0:
        return NAN
    top = float(start.max())
    end_height = top - (top - bottom) * CONCENTRIC_END_FRACTION
    end_index = None
    for i in range(deepest_index + 1, len(samples)):
        if math.isfinite(shoulder[i]) and shoulder[i] >= end_height:
            end_index = i
            break
    if end_index is None:
        return NAN
    elapsed = samples[end_index].timestamp - samples[deepest_index].timestamp
    if elapsed <= 0.0:
        return NAN
    return (shoulder[end_index] - bottom) / 100.0 / elapsed


def _initiation_ratio(samples: list[RepTrajectorySample], deepest_index: int) -> float:
    heights = [sample.hip_height_cm for sample in samples]
    if deepest_index < 1 or not math.isfinite(heights[0]) or not math.isfinite(heights[deepest_index]):
        return NAN
    drop_target = heights[0] - (heights[0] - heights[deepest_index]) * INITIATION_DESCENT_FRACTION
    end_index = next(
        (i for i in range(1, deepest_index + 1) if math.isfinite(heights[i]) and heights[i] <= drop_target),
        None,
    )
    if end_index is None:
        return NAN

    def _mean_side(sample: RepTrajectorySample, joint: str) -> float:
        return (getattr(sample, f"{joint}_l") + getattr(sample, f"{joint}_r")) / 2.0

    knee_change = _mean_side(samples[end_index], "knee_flexion") - _mean_side(samples[0], "knee_flexion")
    hip_change = _mean_side(samples[end_index], "hip_flexion") - _mean_side(samples[0], "hip_flexion")
    if not (math.isfinite(knee_change) and math.isfinite(hip_change)):
        return NAN
    if abs(knee_change) < MIN_INITIATION_KNEE_CHANGE_DEG:
        return NAN
    return hip_change / knee_change


def compute_rep_features(
    samples: Sequence[RepTrajectorySample],
    rep_number: int,
    femur_m: float | None = None,
    setup: SetupSnapshot | None = None,
    standing_reference_hip_cm: float = NAN,
    leg_length_m: float | None = None,
) -> RepFeatures:
    """Reduce one rep's frames to the coaching features.

    ``standing_reference_hip_cm`` is the athlete's fully locked-out hip height
    this session; the lockout deficit is measured against it.
    """
    ordered = sorted(samples, key=lambda sample: sample.timestamp)
    features = RepFeatures(rep_number=rep_number, sample_count=len(ordered))
    depth = [sample.depth_ratio for sample in ordered]
    if not ordered or _finite(depth).size == 0:
        return features

    deepest_index = int(np.nanargmin(np.asarray(depth, dtype=np.float64)))
    deepest = depth[deepest_index]
    loaded = {i for i, value in enumerate(depth) if math.isfinite(value) and value <= deepest + LOADED_WINDOW_RATIO}
    tracking = sorted(
        loaded
        | {
            i for i, value in enumerate(depth)
            if i > deepest_index and math.isfinite(value) and value <= deepest + TRACKING_WINDOW_RATIO
        }
    )
    loaded_samples = [ordered[i] for i in sorted(loaded)]

    features.depth_ratio = _percentile(depth, DEEPEST_PERCENTILE)
    femur_cm = (femur_m if femur_m and femur_m >= MIN_FEMUR_M else 0.0) * 100.0
    if femur_cm > 0.0:
        features.depth_cm_above_parallel = features.depth_ratio * femur_cm
    features.knee_flexion_max = _percentile(
        [max(s.knee_flexion_l, s.knee_flexion_r) for s in ordered], WORST_PERCENTILE
    )
    features.trunk_pitch_bottom = _median([s.trunk_pitch for s in loaded_samples])
    features.valgus_l, features.valgus_peak_phase_l = _valgus_side(ordered, tracking, deepest_index, loaded, "l")
    features.valgus_r, features.valgus_peak_phase_r = _valgus_side(ordered, tracking, deepest_index, loaded, "r")
    features.hip_shoot_deg = _hip_shoot_deg(ordered, deepest_index)
    features.hip_shift_ratio, features.hip_shift_peak_phase = _hip_shift(ordered, deepest_index, loaded)
    features.balance_ratio = _median([s.balance_ratio for s in loaded_samples])
    features.heel_rise_max_cm_l = _percentile([s.heel_rise_l_cm for s in ordered], 100.0)
    features.heel_rise_max_cm_r = _percentile([s.heel_rise_r_cm for s in ordered], 100.0)
    features.descent_time_s = ordered[deepest_index].timestamp - ordered[0].timestamp
    features.ascent_time_s = ordered[-1].timestamp - ordered[deepest_index].timestamp
    features.concentric_velocity_mps = _concentric_velocity_mps(ordered, deepest_index)
    features.knee_flexion_asym_bottom = _median(
        [s.knee_flexion_l - s.knee_flexion_r for s in loaded_samples]
    )
    features.initiation_ratio = _initiation_ratio(ordered, deepest_index)
    features.neck_flexion_bottom = _median([s.neck_flexion_deg for s in loaded_samples])
    features.dorsiflexion_max_l = _percentile([s.dorsiflexion_l for s in loaded_samples], 95.0)
    features.dorsiflexion_max_r = _percentile([s.dorsiflexion_r for s in loaded_samples], 95.0)
    features.hip_flexion_max_l = _percentile([s.hip_flexion_l for s in loaded_samples], 95.0)
    features.hip_flexion_max_r = _percentile([s.hip_flexion_r for s in loaded_samples], 95.0)
    features.bar_detected = any(s.bar_detected for s in ordered)

    if setup is not None:
        features.stagger_ratio = setup.stagger_ratio
        features.toe_out_l_deg = setup.toe_out_l_deg
        features.toe_out_r_deg = setup.toe_out_r_deg
        if leg_length_m and math.isfinite(standing_reference_hip_cm) and math.isfinite(setup.hip_height_cm):
            deficit_cm = max(0.0, standing_reference_hip_cm - setup.hip_height_cm)
            features.lockout_deficit_ratio = deficit_cm / (leg_length_m * 100.0)
    return features
