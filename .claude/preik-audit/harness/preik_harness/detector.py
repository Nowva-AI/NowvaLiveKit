"""RTMPose-like 2D detector simulator (production decode semantics).

Per camera, per keypoint: white pixel noise + slow AR(1) per-view bias, gross outliers (Markov episodes),
leg left/right swaps (side cameras at squat bottoms), far-side occlusion (more noise, lower confidence,
rare dropouts), SimCC argmax quantization of the full-frame 192x256 squash (0.5 input px bins, SIMCC_SPLIT_RATIO
2.0 in rtmpose.py) including the half-pixel resize offset, sigmoid-range confidences (measured 0.53-0.72),
and the rtmpose.py rule: confidence < 0.3 -> (0, 0, 0).
Defaults come from noise_calib/noise_params.json (see REPORT.md for the measured vs assumed split).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from biomechanics.pose.rtmpose import SIMCC_SPLIT_RATIO, RTMPoseEstimator

GROUP_OF_KPT = ["head"] * 5 + ["shoulder"] * 2 + ["elbow"] * 2 + ["wrist"] * 2 + ["hip"] * 2 + ["knee"] * 2 + \
               ["ankle"] * 2 + ["toe"] * 2
SIDE_OF_KPT = np.array([0, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1], dtype=np.float64)
OCCLUSION_WEIGHT = {"head": 0.0, "shoulder": 0.3, "elbow": 0.4, "wrist": 0.4, "hip": 0.8, "knee": 0.5,
                    "ankle": 0.7, "toe": 0.9}
LEG_SWAP_PAIRS = [(13, 14), (15, 16), (17, 18)]  # knees, ankles, big toes (hips stay disambiguated by the torso)
CONFIDENCE_THRESHOLD = 0.3


def _group_array(values: dict[str, float]) -> np.ndarray:
    return np.array([values[g] for g in GROUP_OF_KPT], dtype=np.float64)


@dataclass
class NoiseProfile:
    white_px: dict[str, float] = field(default_factory=lambda: {
        "head": 2.0, "shoulder": 2.0, "elbow": 2.0, "wrist": 2.5, "hip": 2.0, "knee": 2.0, "ankle": 2.5,
        "toe": 3.5})
    slow_px: dict[str, float] = field(default_factory=lambda: {
        "head": 2.5, "shoulder": 2.5, "elbow": 3.0, "wrist": 3.5, "hip": 3.5, "knee": 3.0, "ankle": 3.0,
        "toe": 4.0})
    slow_tau_s: float = 1.0
    outlier_rate: dict[str, float] = field(default_factory=lambda: {
        "head": 0.002, "shoulder": 0.002, "elbow": 0.004, "wrist": 0.01, "hip": 0.005, "knee": 0.01,
        "ankle": 0.03, "toe": 0.05})
    outlier_px: tuple[float, float] = (30.0, 150.0)
    outlier_mean_frames: float = 1.6
    conf_base: dict[str, float] = field(default_factory=lambda: {
        "head": 0.66, "shoulder": 0.64, "elbow": 0.645, "wrist": 0.635, "hip": 0.63, "knee": 0.64,
        "ankle": 0.635, "toe": 0.62})
    conf_slow_sd: float = 0.02
    conf_fast_sd: float = 0.012
    conf_fast_ar: float = 0.6
    conf_outlier_penalty: float = 0.045
    conf_error_coef: float = 0.01
    conf_clip: tuple[float, float] = (0.50, 0.76)
    occlusion_noise_gain: float = 1.0
    occlusion_outlier_gain: float = 1.5
    occlusion_conf_penalty: float = 0.05
    occlusion_dropout_rate: float = 0.004
    swap_prob_side_per_rep: float = 0.15
    swap_prob_front_per_rep: float = 0.02
    swap_frames: tuple[int, int] = (2, 6)
    swap_background_rate: float = 0.0003
    quantize: bool = True
    decode_half_pixel_offset: bool = True


PROFILES: dict[str, NoiseProfile] = {
    "realistic": NoiseProfile(),
    "harsh": NoiseProfile(
        white_px={"head": 3.0, "shoulder": 3.0, "elbow": 3.0, "wrist": 3.5, "hip": 3.0, "knee": 3.0, "ankle": 4.0,
                  "toe": 5.0},
        slow_px={"head": 4.0, "shoulder": 4.0, "elbow": 4.5, "wrist": 5.0, "hip": 5.0, "knee": 4.5, "ankle": 4.5,
                 "toe": 6.0},
        outlier_rate={"head": 0.004, "shoulder": 0.004, "elbow": 0.008, "wrist": 0.04, "hip": 0.01, "knee": 0.02,
                      "ankle": 0.10, "toe": 0.12},
        outlier_px=(30.0, 230.0), swap_prob_side_per_rep=0.6, swap_prob_front_per_rep=0.1),
    "mild": NoiseProfile(
        slow_px={"head": 1.5, "shoulder": 1.5, "elbow": 2.0, "wrist": 2.0, "hip": 2.0, "knee": 1.5, "ankle": 2.0,
                 "toe": 2.5},
        outlier_rate={"head": 0.001, "shoulder": 0.001, "elbow": 0.002, "wrist": 0.005, "hip": 0.002, "knee": 0.004,
                      "ankle": 0.01, "toe": 0.02},
        occlusion_noise_gain=0.5, occlusion_outlier_gain=0.5, swap_prob_side_per_rep=0.1,
        swap_prob_front_per_rep=0.0),
    "none": NoiseProfile(
        white_px={g: 0.0 for g in OCCLUSION_WEIGHT}, slow_px={g: 0.0 for g in OCCLUSION_WEIGHT},
        outlier_rate={g: 0.0 for g in OCCLUSION_WEIGHT}, conf_slow_sd=0.0, conf_fast_sd=0.0,
        conf_outlier_penalty=0.0, conf_error_coef=0.0, occlusion_noise_gain=0.0, occlusion_outlier_gain=0.0,
        occlusion_conf_penalty=0.0, occlusion_dropout_rate=0.0, swap_prob_side_per_rep=0.0,
        swap_prob_front_per_rep=0.0, swap_background_rate=0.0, quantize=False, decode_half_pixel_offset=False),
}


def get_profile(profile: str | NoiseProfile) -> NoiseProfile:
    if isinstance(profile, NoiseProfile):
        return profile
    if profile not in PROFILES:
        raise ValueError(f"unknown noise profile {profile}; choose from {list(PROFILES)}")
    return replace(PROFILES[profile])


def _ar1(rng: np.random.Generator, frames: int, shape: tuple[int, ...], sd: np.ndarray | float, rho: float
         ) -> np.ndarray:
    out = np.zeros((frames,) + shape)
    innovation = math.sqrt(max(1.0 - rho * rho, 0.0))
    out[0] = rng.normal(0.0, 1.0, shape)
    for k in range(1, frames):
        out[k] = rho * out[k - 1] + innovation * rng.normal(0.0, 1.0, shape)
    return out * sd


def detect(uv_true: np.ndarray, occlusion: np.ndarray, frame_times_s: np.ndarray, swap_windows: list[tuple[float, float]],
           profile: NoiseProfile, rng: np.random.Generator, resolution: tuple[int, int]) -> np.ndarray:
    """uv_true (F, 19, 2) px, occlusion (F, 19) in [0, 1] -> decoded detections (F, 19, 3) [x, y, conf]."""
    frames, n_kpts, _ = uv_true.shape
    dt = float(np.median(np.diff(frame_times_s))) if frames > 1 else 1.0 / 30.0
    white = _group_array(profile.white_px)
    slow = _group_array(profile.slow_px)
    noise_gain = 1.0 + profile.occlusion_noise_gain * occlusion  # (F, K)

    white_noise = rng.normal(0.0, 1.0, (frames, n_kpts, 2)) * white[None, :, None]
    slow_noise = _ar1(rng, frames, (n_kpts, 2), slow[None, :, None], math.exp(-dt / profile.slow_tau_s))
    uv = uv_true + (white_noise + slow_noise) * noise_gain[:, :, None]

    # gross outliers: Markov episodes with constant displacement
    rate = _group_array(profile.outlier_rate)[None, :] * (1.0 + profile.occlusion_outlier_gain * occlusion)
    mean_frames = max(profile.outlier_mean_frames, 1.0)
    start_prob = np.clip(rate / mean_frames, 0.0, 1.0)
    end_prob = 1.0 / mean_frames
    in_outlier = np.zeros((frames, n_kpts), dtype=bool)
    offset = np.zeros((n_kpts, 2))
    active = np.zeros(n_kpts, dtype=bool)
    starts = rng.random((frames, n_kpts))
    ends = rng.random((frames, n_kpts))
    magnitudes = rng.uniform(profile.outlier_px[0], profile.outlier_px[1], (frames, n_kpts))
    angles = rng.uniform(0.0, 2 * math.pi, (frames, n_kpts))
    for k in range(frames):
        stop = active & (ends[k] < end_prob)
        active &= ~stop
        begin = ~active & (starts[k] < start_prob[k])
        offset[begin, 0] = magnitudes[k, begin] * np.cos(angles[k, begin])
        offset[begin, 1] = magnitudes[k, begin] * np.sin(angles[k, begin])
        active |= begin
        in_outlier[k] = active
        uv[k, active] += offset[active]

    # left/right leg swaps inside the given windows (per camera)
    swapped = np.zeros(frames, dtype=bool)
    for start_s, end_s in swap_windows:
        swapped |= (frame_times_s >= start_s) & (frame_times_s <= end_s)
    background = rng.random(frames) < profile.swap_background_rate
    for k in np.nonzero(background)[0]:
        swapped[k:k + int(rng.integers(1, 4))] = True

    # confidences
    conf = _group_array(profile.conf_base)[None, :].repeat(frames, 0)
    conf = conf + _ar1(rng, frames, (n_kpts,), profile.conf_slow_sd, math.exp(-dt / 1.0))
    conf = conf + _ar1(rng, frames, (n_kpts,), profile.conf_fast_sd, profile.conf_fast_ar)
    with np.errstate(invalid="ignore", divide="ignore"):
        normalized_err = np.where(white[None, :] > 0,
                                  np.linalg.norm(white_noise, axis=2) / np.maximum(white[None, :], 1e-9), 0.0)
    conf -= profile.conf_error_coef * (normalized_err - 1.25)
    conf -= in_outlier * profile.conf_outlier_penalty * rng.uniform(0.5, 1.5, (frames, n_kpts))
    conf -= profile.occlusion_conf_penalty * occlusion
    conf = np.clip(conf, profile.conf_clip[0], profile.conf_clip[1])
    dropout = rng.random((frames, n_kpts)) < profile.occlusion_dropout_rate * occlusion
    conf[dropout] = rng.uniform(0.2, 0.3, dropout.sum())

    for left, right in LEG_SWAP_PAIRS:
        uv[swapped, left], uv[swapped, right] = uv[swapped, right].copy(), uv[swapped, left].copy()
        conf[swapped, left], conf[swapped, right] = conf[swapped, right].copy(), conf[swapped, left].copy()

    width, height = resolution
    input_w, input_h = RTMPoseEstimator.INPUT_WIDTH, RTMPoseEstimator.INPUT_HEIGHT
    scale_x, scale_y = width / input_w, height / input_h
    if profile.quantize:
        half = 0.5 if profile.decode_half_pixel_offset else 0.0
        # model sees resized content: input coord u = (x + 0.5) / scale - 0.5; argmax bin = round(u * split)
        u_in = (uv[..., 0] + half) / scale_x - half
        v_in = (uv[..., 1] + half) / scale_y - half
        bins_x = np.clip(np.round(u_in * SIMCC_SPLIT_RATIO), 0, input_w * SIMCC_SPLIT_RATIO - 1)
        bins_y = np.clip(np.round(v_in * SIMCC_SPLIT_RATIO), 0, input_h * SIMCC_SPLIT_RATIO - 1)
        uv = np.stack([bins_x / SIMCC_SPLIT_RATIO * scale_x, bins_y / SIMCC_SPLIT_RATIO * scale_y], -1)

    out = np.concatenate([uv, conf[..., None]], axis=-1)
    below = out[..., 2] < CONFIDENCE_THRESHOLD
    out[below] = 0.0
    return out


def occlusion_levels(camera_position: np.ndarray, pelvis_world: np.ndarray, body_yaw_deg: np.ndarray,
                     squat_phase: np.ndarray) -> np.ndarray:
    """(F, 19) far-side occlusion in [0, 1]: grows with squat depth for side cameras."""
    yaw = np.radians(body_yaw_deg)
    to_cam = camera_position[None, :] - pelvis_world
    to_cam /= np.linalg.norm(to_cam, axis=1, keepdims=True)
    # body-left axis in world for yaw psi (rot_y applied to +X)
    left_axis = np.stack([np.cos(yaw), np.zeros_like(yaw), -np.sin(yaw)], -1)
    lateral_cos = np.sum(to_cam * left_axis, axis=1)
    weights = np.array([OCCLUSION_WEIGHT[g] for g in GROUP_OF_KPT])
    far = np.maximum(0.0, -SIDE_OF_KPT[None, :] * lateral_cos[:, None])
    return np.clip(weights[None, :] * far * squat_phase[:, None], 0.0, 1.0)
