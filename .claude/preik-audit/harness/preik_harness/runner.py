"""Input preparation (GT -> cameras -> detections -> delivery -> real DLTTriangulator, WORLD frame) and a replica of
BiomechanicsPipeline.process_frame for the parts that shape what the pre-IK chain sees and what it feeds.

Two chain families are supported, selected by attributes on the chain object (all optional):
  input_frame     "hip" (default): the runner recentres the world skeleton at the hips (recentre_at_hips) before the
                  gates and the chain, exactly as the baseline saw hip-centred triangulator output; "world": the chain
                  receives the world skeleton on EVERY tick with a skeleton (also before the readiness gate opens) and
                  must return a hip-centred analysis skeleton or None.
  handles_missing False (default): production dropout hold (5 held frames, decayed confidences); True: the runner
                  calls chain.process_missing(context) instead when triangulation returned None or a hip is missing.
  lagged_output   False (default): output belongs to the current tick; True: output.frame_index names the tick it
                  belongs to (fixed-lag smoother), and records are stored at that index so metrics align to truth.
  tail            "old" (default): AnalyticalIKSolver -> NaN->0.0 (the pre-overhaul IK/valgus returned 0.0 for missing
                  keypoints) -> JointAngleFilter (legacy copy) with phase from SignalRepCounter; "new": AnalyticalIKSolver
                  (NaN for missing), no angle filter, rep counter fed with the lagged timestamp.
Optional chain attributes read after each frame: last_foot_state (FootState), last_stage_us (dict), calibrated().
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from biomechanics.config import BiomechanicsConfig, load_pipeline_config
from biomechanics.kinematics.analytical_ik import AnalyticalIKSolver
from biomechanics.kinematics.valgus import build_valgus_estimator
from biomechanics.profiles import get_profile
from biomechanics.triangulation.calibration import CalibrationResult
from biomechanics.triangulation.triangulator import DLTTriangulator, recentre_at_hips
from biomechanics.utils.standing_gate import StandingPoseGate
from biomechanics.utils.types import JointAngles, Keypoint2D, MultiViewPose, Skeleton2D, Skeleton3D

from . import body
from .barbell import bar_end_points, detect_bar
from .cameras import (PERSON_BA_MODES, Camera, PersonBAConfig, RigConfig, build_rig, perfect_calibration,
                      person_ba_calibration, tpose_calibration, triangulate_noise_free)
from .delivery import DeliveryConfig, Tick, simulate_cameras, simulate_loop
from .detector import NoiseProfile, detect, get_profile as get_noise_profile, occlusion_levels
from .legacy.filters import JointAngleFilter

MAX_DROPOUT_HOLD_FRAMES = 5
DENSE_DT_S = 0.002
NUM_METRIC_KPTS = 19
ANGLE_FIELDS = ["knee_flexion_l", "knee_flexion_r", "hip_flexion_l", "hip_flexion_r", "trunk_flexion",
                "knee_valgus_l", "knee_valgus_r", "pelvis_list", "ankle_dorsiflexion_l", "ankle_dorsiflexion_r"]
_JOINT_ANGLE_FLOAT_FIELDS = [name for name, info in JointAngles.model_fields.items()
                             if info.annotation is float and name not in ("timestamp",)]

_CONFIG: BiomechanicsConfig | None = None


def pipeline_config() -> BiomechanicsConfig:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = load_pipeline_config()
    return _CONFIG


class PreIKChain(Protocol):
    def process(self, skeleton: Skeleton3D, context: "FrameContext") -> Skeleton3D | None: ...

    def reset(self) -> None: ...


@dataclass
class FrameContext:
    """Per-frame info a chain may use. Only things production process_frame also has."""

    tick_index: int
    loop_time_s: float  # time.time()-like clock (pipeline `now`)
    is_held: bool  # dropout-hold skeleton (stale raw skeleton, decayed confidences, wall-clock timestamp)
    standing_gate: StandingPoseGate
    readiness_gate: StandingPoseGate
    multi_view: MultiViewPose | None
    calibration: CalibrationResult


@dataclass
class PreparedRun:
    scenario: body.Scenario
    seed: int
    calibration_mode: str
    noise: str
    cameras: list[Camera]
    calibration: CalibrationResult
    calibration_info: dict
    ticks: list[Tick]
    raw_skeletons: list[Skeleton3D | None]  # WORLD frame, 21 keypoints (new triangulator)
    multi_views: list[MultiViewPose | None]
    truth_time_s: np.ndarray
    truth_world: np.ndarray  # (T, 21, 3) true world keypoints at truth_time
    truth_hc: np.ndarray  # (T, 21, 3) hip-centred true keypoints at truth_time
    truth_angles: dict[str, np.ndarray]
    dense_t_s: np.ndarray
    dense_hc: np.ndarray  # (D, 21, 3)
    stats: dict = field(default_factory=dict)


def _numpy_to_skeleton2d(kpts: np.ndarray, timestamp: float) -> Skeleton2D:
    return Skeleton2D(keypoints=[Keypoint2D(x=float(k[0]), y=float(k[1]), confidence=float(k[2])) for k in kpts],
                      timestamp=timestamp, frame_index=0)


def _swap_windows(scenario: body.Scenario, camera: Camera, profile: NoiseProfile, rng: np.random.Generator
                  ) -> list[tuple[float, float]]:
    is_side = abs(camera.azimuth_deg) > 15.0
    prob = profile.swap_prob_side_per_rep if is_side else profile.swap_prob_front_per_rep
    windows = []
    for rep in scenario.reps:
        if rng.random() < prob:
            frames = int(rng.integers(profile.swap_frames[0], profile.swap_frames[1] + 1))
            centre = 0.5 * (rep.bottom_start_s + rep.bottom_end_s) + rng.uniform(-0.2, 0.2)
            windows.append((centre - frames / 60.0, centre + frames / 60.0))
    return windows


def _truth_angles(points_hc: np.ndarray) -> dict[str, np.ndarray]:
    ik = AnalyticalIKSolver()
    valgus = build_valgus_estimator(True)
    out = {name: np.zeros(len(points_hc)) for name in ANGLE_FIELDS}
    for i, pts in enumerate(points_hc):
        skel = Skeleton3D.from_numpy(pts[:NUM_METRIC_KPTS], confidences=np.ones(NUM_METRIC_KPTS), timestamp=1.0,
                                     frame_index=0)
        angles = ik.solve(skel)
        result = valgus.estimate(None, skel)
        angles.knee_valgus_l, angles.knee_valgus_r = result.valgus_l, result.valgus_r
        for name in ANGLE_FIELDS:
            out[name][i] = getattr(angles, name)
    return out


def hip_centre(points: np.ndarray) -> np.ndarray:
    return points - 0.5 * (points[..., body.L_HIP:body.L_HIP + 1, :] + points[..., body.R_HIP:body.R_HIP + 1, :])


def _reprojection_stats(detections: dict[str, np.ndarray], frame_sets: list[dict[str, int]], cameras: list[Camera],
                        calibration: CalibrationResult) -> float:
    errors = []
    proj = {c.cam_id: calibration.cameras[c.cam_id].projection_matrix for c in cameras}
    for frames in frame_sets[::3]:
        views = [(proj[cid], detections[cid][k]) for cid, k in frames.items()]
        valid = np.all([v[1][:, 2] >= 0.3 for v in views], axis=0)
        for kpt in np.nonzero(valid)[0]:
            rows = []
            for mat, det in views:
                rows.append(det[kpt, 0] * mat[2] - mat[0])
                rows.append(det[kpt, 1] * mat[2] - mat[1])
            _, _, vt = np.linalg.svd(np.array(rows))
            point = vt[-1] / vt[-1, 3]
            for mat, det in views:
                pix = mat @ point
                errors.append(np.linalg.norm(pix[:2] / pix[2] - det[kpt, :2]))
    return float(np.mean(errors)) if errors else float("nan")


_PERSON_BA_CACHE: dict[tuple, tuple[CalibrationResult, dict]] = {}


def person_ba_capture(cameras: list[Camera], profile: NoiseProfile, seed: int, delivery_config: DeliveryConfig,
                      cfg: PersonBAConfig) -> tuple[list[dict[str, Skeleton2D]], list[dict], np.ndarray]:
    """Simulate the calibration capture (walk-in + reps with the bar on the back) through the same detector and
    camera-delivery models as a normal run. Returns per-synced-frame views, bar detections, and the TRUE joints of
    the capture at 30 Hz (for the harness gauge alignment and diagnostics only)."""
    capture_seed = seed + cfg.capture_seed_offset
    scenario = body.build_scenario(cfg.capture_scenario, capture_seed)
    duration_s = scenario.reps[cfg.capture_reps - 1].end_s + cfg.capture_tail_s
    cam_ids = [c.cam_id for c in cameras]
    streams = simulate_cameras(cam_ids, duration_s, capture_seed, delivery_config)
    ticks = simulate_loop(streams, cam_ids, duration_s, capture_seed, delivery_config)
    noiseless = all(v == 0.0 for v in profile.white_px.values())
    detections: dict[str, np.ndarray] = {}
    bars: dict[str, list] = {}
    for cam_index, camera in enumerate(cameras):
        rng = np.random.default_rng([seed, 606, cam_index])
        times = np.clip(streams[camera.cam_id].exposure_s, 0.0, None)
        pts = scenario.pose(times)
        params = scenario.sample(times)
        pelvis = 0.5 * (pts[:, body.L_HIP] + pts[:, body.R_HIP])
        occ = occlusion_levels(camera.position, pelvis, params["body_yaw"], params["s"])
        uv, _ = camera.project(pts[:, :NUM_METRIC_KPTS])
        detections[camera.cam_id] = detect(uv, occ, streams[camera.cam_id].exposure_s,
                                           _swap_windows(scenario, camera, profile, rng), profile, rng,
                                           camera.resolution)
        bars[camera.cam_id] = detect_bar(camera, bar_end_points(pts, cfg.bar_length_m),
                                         streams[camera.cam_id].exposure_s, rng, noiseless=noiseless)
    views, bar_ends, seen = [], [], set()
    for tick in ticks:
        if tick.frames is None:
            continue
        key = tuple(tick.frames[c] for c in cam_ids)
        if key in seen:
            continue
        seen.add(key)
        views.append({cid: _numpy_to_skeleton2d(detections[cid][tick.frames[cid]], tick.t_call_s) for cid in cam_ids})
        bar_ends.append({cid: bars[cid][tick.frames[cid]] for cid in cam_ids
                         if bars[cid][tick.frames[cid]] is not None})
    return views, bar_ends, scenario.pose(np.arange(0.0, duration_s, 1.0 / 30.0))


def _make_detect_fn(profile: NoiseProfile):
    def detect_fn(camera: Camera, points_world: np.ndarray, times: np.ndarray, rng: np.random.Generator,
                  occlusion: np.ndarray | None = None, swaps: list[tuple[float, float]] | None = None) -> np.ndarray:
        # detector simulates the 19 COCO+toe keypoints only (heels 19/20 never detected -> triangulator conf 0)
        uv, _ = camera.project(points_world[:, :NUM_METRIC_KPTS])
        occ = np.zeros(uv.shape[:2]) if occlusion is None else occlusion
        return detect(uv, occ, times, swaps or [], profile, rng, camera.resolution)
    return detect_fn


def person_ba_for_seed(seed: int, mode: str, noise: str | NoiseProfile = "realistic",
                       rig_config: RigConfig | None = None, delivery_config: DeliveryConfig | None = None
                       ) -> tuple[CalibrationResult, dict]:
    """One calibration per (seed, mode, noise, rig, delivery): every scenario of that seed reuses it, like a real
    session. Cached in-process; api.evaluate_chains fills the cache in the parent before forking workers (the
    calibrator's dense solve can deadlock inside fork()ed children on macOS/Accelerate)."""
    rig_config = rig_config or RigConfig()
    delivery_config = delivery_config or DeliveryConfig()
    key = (seed, mode, repr(noise), repr(rig_config), repr(delivery_config))
    if key not in _PERSON_BA_CACHE:
        cfg = PERSON_BA_MODES[mode]
        profile = get_noise_profile(noise)
        cameras = build_rig(seed, rig_config)
        views, bar_ends, capture_truth = person_ba_capture(cameras, profile, seed, delivery_config, cfg)
        initial = None
        if cfg.start_from_tpose:
            initial, _ = tpose_calibration(cameras, _make_detect_fn(profile), seed, rig_config)
        _PERSON_BA_CACHE[key] = person_ba_calibration(cameras, views, bar_ends, capture_truth, cfg, initial)
    return _PERSON_BA_CACHE[key]


def _calibration_floor_mm(scenario: body.Scenario, cameras: list[Camera], calibration: CalibrationResult) -> float:
    # calibration-only error of the hip-centred skeleton: no detection noise, plain DLT
    points = scenario.pose(np.arange(0.0, scenario.duration_s, 0.1))[:, :NUM_METRIC_KPTS]
    solved = triangulate_noise_free(cameras, calibration, points)
    return float(np.mean(np.linalg.norm(hip_centre(solved) - hip_centre(points), axis=-1)) * 1000.0)


def prepare_run(scenario_name: str, seed: int, calibration: str = "perfect", noise: str | NoiseProfile = "realistic",
                rig_config: RigConfig | None = None, delivery_config: DeliveryConfig | None = None) -> PreparedRun:
    if calibration not in ("perfect", "tpose") and calibration not in PERSON_BA_MODES:
        raise ValueError(f"calibration must be 'perfect', 'tpose' or one of {list(PERSON_BA_MODES)}")
    rig_config = rig_config or RigConfig()
    delivery_config = delivery_config or DeliveryConfig()
    profile = get_noise_profile(noise)
    noise_name = noise if isinstance(noise, str) else "custom"
    scenario = body.build_scenario(scenario_name, seed)
    cameras = build_rig(seed, rig_config)
    cam_ids = [c.cam_id for c in cameras]

    detect_fn = _make_detect_fn(profile)

    if calibration == "perfect":
        calib = perfect_calibration(cameras)
        calib_info = {}
    elif calibration == "tpose":
        calib, calib_info = tpose_calibration(cameras, detect_fn, seed, rig_config)
    else:
        calib, calib_info = person_ba_for_seed(seed, calibration, noise, rig_config, delivery_config)

    streams = simulate_cameras(cam_ids, scenario.duration_s, seed, delivery_config)
    ticks = simulate_loop(streams, cam_ids, scenario.duration_s, seed, delivery_config)

    detections: dict[str, np.ndarray] = {}
    for cam_index, camera in enumerate(cameras):
        rng = np.random.default_rng([seed, 505, cam_index, body.SCENARIOS.index(scenario_name)])
        times = streams[camera.cam_id].exposure_s
        pts = scenario.pose(np.clip(times, 0.0, None))
        params = scenario.sample(np.clip(times, 0.0, None))
        pelvis = 0.5 * (pts[:, body.L_HIP] + pts[:, body.R_HIP])
        occ = occlusion_levels(camera.position, pelvis, params["body_yaw"], params["s"])
        swaps = _swap_windows(scenario, camera, profile, rng)
        detections[camera.cam_id] = detect_fn(camera, pts, times, rng, occ, swaps)

    triangulator = DLTTriangulator(calibration=calib, min_views=2, max_reprojection_error=15.0, min_confidence=0.3)
    cache: dict[tuple, tuple[Skeleton3D | None, MultiViewPose]] = {}
    raw_skeletons: list[Skeleton3D | None] = []
    multi_views: list[MultiViewPose | None] = []
    truth_time = np.zeros(len(ticks))
    primary = cam_ids[0]
    nominal_latency = streams[primary].latency_s
    perf_offset = delivery_config.perf_counter_offset_s
    wall_offset = delivery_config.wall_clock_offset_s
    for i, tick in enumerate(ticks):
        if tick.frames is None:
            raw_skeletons.append(None)
            multi_views.append(None)
            truth_time[i] = tick.t_call_s - nominal_latency
            continue
        key = tuple(tick.frames[c] for c in cam_ids)
        if key not in cache:
            views = {cid: _numpy_to_skeleton2d(detections[cid][tick.frames[cid]], tick.t_call_s + wall_offset)
                     for cid in cam_ids}
            multi_view = MultiViewPose(views=views, timestamp=tick.ref_stamp_s + perf_offset, frame_index=0)
            cache[key] = (triangulator.triangulate(multi_view), multi_view)
        skeleton, multi_view = cache[key]
        raw_skeletons.append(skeleton)
        multi_views.append(multi_view)
        truth_time[i] = streams[primary].exposure_s[tick.frames[primary]]

    truth_world = scenario.pose(np.clip(truth_time, 0.0, None))
    truth_hc = hip_centre(truth_world)
    dense_t = np.arange(0.0, scenario.duration_s, DENSE_DT_S)
    dense_hc = hip_centre(scenario.pose(dense_t))

    latest_primary = np.searchsorted(streams[primary].stamp_s, [t.t_call_s for t in ticks], side="right") - 1
    primary_frames_in_span = int(latest_primary[-1] - latest_primary[0] + 1)
    skipped_primary_frac = 1.0 - len(set(latest_primary.tolist())) / max(primary_frames_in_span, 1)
    duplicates = 0
    prev = None
    for t in ticks:
        current = None if t.frames is None else t.frames[primary]
        if current is not None and current == prev:
            duplicates += 1
        prev = current if current is not None else prev
    lower_conf = np.array([[kp.confidence for kp in sk.keypoints[11:NUM_METRIC_KPTS]]
                           for sk in raw_skeletons if sk is not None])
    stats = {
        "ticks": len(ticks),
        "loop_hz": round(len(ticks) / (ticks[-1].t_call_s - ticks[0].t_call_s), 2),
        "duplicate_frac": round(duplicates / len(ticks), 4),
        "skipped_primary_frac": round(skipped_primary_frac, 4),
        "sync_fail_frac": round(sum(t.frames is None for t in ticks) / len(ticks), 4),
        "triangulation_none_frac": round(sum(t.frames is not None and s is None
                                             for t, s in zip(ticks, raw_skeletons)) / len(ticks), 4),
        "mean_reprojection_px": round(_reprojection_stats(detections, [t.frames for t in ticks if t.frames],
                                                          cameras, calib), 2),
        "detected_zero_conf_frac": round(float(np.mean([np.mean(d[:, :, 2] == 0) for d in detections.values()])), 5),
        "tri_lower_conf_below_0.1_frac": round(float(np.mean(lower_conf < 0.1)), 4) if lower_conf.size else None,
        "tri_lower_conf_zero_frac": round(float(np.mean(lower_conf == 0.0)), 4) if lower_conf.size else None,
        "tri_lower_conf_p50": round(float(np.median(lower_conf)), 4) if lower_conf.size else None,
        "tri_swap_count": int(triangulator.swap_count),
        "calibration_floor_mm": round(_calibration_floor_mm(scenario, cameras, calib), 2),
    }
    return PreparedRun(scenario=scenario, seed=seed, calibration_mode=calibration, noise=noise_name, cameras=cameras,
                       calibration=calib, calibration_info=calib_info, ticks=ticks, raw_skeletons=raw_skeletons,
                       multi_views=multi_views, truth_time_s=truth_time, truth_world=truth_world, truth_hc=truth_hc,
                       truth_angles=_truth_angles(truth_hc), dense_t_s=dense_t, dense_hc=dense_hc, stats=stats)


@dataclass
class RunRecords:
    status: np.ndarray  # 0 no skeleton, 1 gated by readiness, 2 processed
    held: np.ndarray
    predicted: np.ndarray  # output came from chain.process_missing (no triangulated skeleton at that tick)
    out_xyz: np.ndarray  # (T, 19, 3) NaN unless processed
    out_conf: np.ndarray
    ik: dict[str, np.ndarray]
    final: dict[str, np.ndarray]
    phase: list[str]
    chain_us: np.ndarray
    total_us: np.ndarray
    latency_ms: np.ndarray  # loop time when the output was produced minus loop time of the tick it belongs to
    stage_us: dict[str, np.ndarray]
    foot_heel_rise_cm: np.ndarray  # (T, 2) from FootState when the chain exposes one, else NaN
    foot_planted: np.ndarray  # (T, 2)
    foot_valid: np.ndarray
    standing_ready: np.ndarray
    readiness_ready: np.ndarray
    chain_calibrated: np.ndarray  # -1 unknown, 0/1


def _copy_skeleton(skeleton: Skeleton3D, frame_index: int) -> Skeleton3D:
    return Skeleton3D.from_numpy(skeleton.to_numpy(), confidences=[kp.confidence for kp in skeleton.keypoints],
                                 timestamp=skeleton.timestamp, frame_index=frame_index)


def _chain_calibrated(chain: Any) -> int:
    probe = getattr(chain, "calibrated", None)
    if probe is None:
        return -1
    value = probe() if callable(probe) else probe
    return -1 if value is None else int(bool(value))


def _nan_angles_to_zero(angles: JointAngles) -> None:
    # The pre-overhaul AnalyticalIKSolver / TriangulatedValgusEstimator returned 0.0 for any angle whose keypoints
    # fell below the confidence floor; the old tail reproduces that so its numbers stay comparable to the baseline.
    for name in _JOINT_ANGLE_FLOAT_FIELDS:
        value = getattr(angles, name)
        if isinstance(value, float) and math.isnan(value):
            setattr(angles, name, 0.0)


def _build_gate(cfg: Any) -> StandingPoseGate:
    return StandingPoseGate(
        min_confidence=cfg.min_confidence, max_knee_flexion_deg=cfg.max_knee_flexion_deg,
        max_trunk_flexion_deg=cfg.max_trunk_flexion_deg, min_torso_length_m=cfg.min_torso_length_m,
        max_torso_length_m=cfg.max_torso_length_m, min_leg_extension_ratio=cfg.min_leg_extension_ratio,
        required_consecutive_frames=cfg.required_consecutive_frames)


def run_chain(prepared: PreparedRun, chain_factory, delivery_config: DeliveryConfig | None = None) -> RunRecords:
    delivery_config = delivery_config or DeliveryConfig()
    cfg = pipeline_config()
    standing_gate = _build_gate(cfg.standing_gate)
    readiness_gate = _build_gate(cfg.readiness_gate)
    ik_solver = AnalyticalIKSolver()
    valgus_estimator = build_valgus_estimator(True)
    angle_filter = JointAngleFilter(min_cutoff=1.0, beta=0.007)
    profile = get_profile("Barbell Back Squat")
    rep_counter = profile.create_rep_counter(cfg)
    chain = chain_factory()
    chain.reset()
    world_input = getattr(chain, "input_frame", "hip") == "world"
    handles_missing = bool(getattr(chain, "handles_missing", False))
    lagged_output = bool(getattr(chain, "lagged_output", False))
    old_tail = getattr(chain, "tail", "old") == "old"

    n = len(prepared.ticks)
    rec = RunRecords(status=np.zeros(n, dtype=np.int8), held=np.zeros(n, dtype=bool),
                     predicted=np.zeros(n, dtype=bool),
                     out_xyz=np.full((n, NUM_METRIC_KPTS, 3), np.nan), out_conf=np.zeros((n, NUM_METRIC_KPTS)),
                     ik={k: np.full(n, np.nan) for k in ANGLE_FIELDS},
                     final={k: np.full(n, np.nan) for k in ANGLE_FIELDS}, phase=["-"] * n,
                     chain_us=np.full(n, np.nan), total_us=np.full(n, np.nan), latency_ms=np.full(n, np.nan),
                     stage_us={}, foot_heel_rise_cm=np.full((n, 2), np.nan), foot_planted=np.zeros((n, 2), dtype=bool),
                     foot_valid=np.zeros(n, dtype=bool),
                     standing_ready=np.zeros(n, dtype=bool), readiness_ready=np.zeros(n, dtype=bool),
                     chain_calibrated=np.full(n, -1, dtype=np.int8))
    last_valid: Skeleton3D | None = None
    dropout_frames = 0
    for i, tick in enumerate(prepared.ticks):
        now = tick.t_call_s + delivery_config.wall_clock_offset_s
        if tick.frames is None:
            # get_synced_frames() -> None makes get_pose() return frame=None, so process_frame returns before
            # the dropout hold (pipeline.py `if frame is None`): nothing reaches gates, chain or IK.
            continue
        world = prepared.raw_skeletons[i]
        # C3: hip-centred view of the world skeleton; None when a hip is missing (= production hip-dropout path)
        skeleton = recentre_at_hips(world) if world is not None else None
        held = False
        predicted = False
        context = FrameContext(tick_index=i, loop_time_s=now, is_held=False, standing_gate=standing_gate,
                               readiness_gate=readiness_gate, multi_view=prepared.multi_views[i],
                               calibration=prepared.calibration)
        if skeleton is None and handles_missing:
            t_start = time.perf_counter()
            filtered = chain.process_missing(context)
            t_chain = time.perf_counter()
            predicted = True
        else:
            if skeleton is None:
                if last_valid is not None and dropout_frames < MAX_DROPOUT_HOLD_FRAMES:
                    dropout_frames += 1
                    decay = 1.0 - dropout_frames / MAX_DROPOUT_HOLD_FRAMES
                    held_ts = now if delivery_config.mix_clocks else last_valid.timestamp + (
                        tick.t_call_s - prepared.ticks[i - dropout_frames].t_call_s)
                    skeleton = Skeleton3D.from_numpy(last_valid.to_numpy(),
                                                     confidences=[kp.confidence * decay for kp in last_valid.keypoints],
                                                     timestamp=held_ts, frame_index=i)
                    held = True
                else:
                    continue
            else:
                dropout_frames = 0
                last_valid = skeleton
            rec.held[i] = held
            context.is_held = held

            standing_gate.check(skeleton)
            readiness_gate.check(skeleton)
            rec.standing_ready[i] = standing_gate.is_ready
            rec.readiness_ready[i] = readiness_gate.is_ready
            if not world_input and not readiness_gate.is_ready:
                rec.status[i] = 1
                continue
            chain_input = world if world_input and not held else skeleton
            t_start = time.perf_counter()
            filtered = chain.process(_copy_skeleton(chain_input, i), context)
            t_chain = time.perf_counter()
        if filtered is None:
            continue
        j = int(filtered.frame_index) if lagged_output else i
        if not readiness_gate.is_ready:
            rec.status[j] = max(rec.status[j], 1)
            continue

        raw_angles = ik_solver.solve(filtered)
        valgus = valgus_estimator.estimate(None, filtered)
        raw_angles.knee_valgus_l = valgus.valgus_l
        raw_angles.knee_valgus_r = valgus.valgus_r
        raw_angles.foot_confidence_l = valgus.foot_confidence_l
        raw_angles.foot_confidence_r = valgus.foot_confidence_r
        raw_angles.knee_ankle_sep_ratio = valgus.kasr
        raw_angles.hip_rotation_l = valgus.hip_rotation_l
        raw_angles.hip_rotation_r = valgus.hip_rotation_r
        if old_tail:
            _nan_angles_to_zero(raw_angles)
            angle_filter.update_phase(rep_counter.phase)
            angles = angle_filter.filter_angles(raw_angles)
            counter_timestamp = now
        else:
            angles = raw_angles
            counter_timestamp = filtered.timestamp
        t_end = time.perf_counter()
        rep_counter.update(signal_value=profile.get_rep_signal(filtered, angles), timestamp=counter_timestamp,
                           angles=angles, faults=[])

        rec.status[j] = 2
        rec.held[j] = held
        rec.predicted[j] = predicted
        rec.out_xyz[j] = filtered.to_numpy()[:NUM_METRIC_KPTS]
        rec.out_conf[j] = [kp.confidence for kp in filtered.keypoints[:NUM_METRIC_KPTS]]
        for name in ANGLE_FIELDS:
            rec.ik[name][j] = getattr(raw_angles, name)
            rec.final[name][j] = getattr(angles, name)
        rec.phase[j] = rep_counter.phase
        rec.chain_us[j] = (t_chain - t_start) * 1e6
        rec.total_us[j] = (t_end - t_start) * 1e6
        rec.latency_ms[j] = (tick.t_call_s - prepared.ticks[j].t_call_s) * 1000.0
        rec.chain_calibrated[j] = _chain_calibrated(chain)
        stage_us = getattr(chain, "last_stage_us", None)
        if stage_us:
            for name, value in stage_us.items():
                rec.stage_us.setdefault(name, np.full(n, np.nan))[j] = value
        foot_state = getattr(chain, "last_foot_state", None)
        if foot_state is not None:
            rec.foot_heel_rise_cm[j] = (foot_state.heel_rise_l_cm, foot_state.heel_rise_r_cm)
            rec.foot_planted[j] = (foot_state.planted_l, foot_state.planted_r)
            rec.foot_valid[j] = foot_state.valid
    return rec
