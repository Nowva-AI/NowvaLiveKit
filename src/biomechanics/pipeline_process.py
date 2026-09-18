"""
Biomechanics Pipeline Process

Replaces the old pose_estimation_process.py with the full layered pipeline.
Launched as a subprocess by main.py when workout mode starts.
Communicates with the voice agent via the existing IPC system.
"""

from __future__ import annotations

import json
import logging
import math
import select
import signal
import sys
import os
import threading
import time
import webbrowser
from pathlib import Path
from typing import Callable

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from agent.core.ipc_communication import IPCClient, _recv_framed
from biomechanics.pipeline import BiomechanicsPipeline
from biomechanics.calibration import (
    CalibrationTracker,
    build_calibration_profile,
    apply_calibration_to_rule_engine,
    extract_thresholds_from_rule_engine,
    get_movement_pattern,
)
from biomechanics.coaching.ipc_bridge import IPCBridge
from biomechanics.coaching.session_tracker import SessionTracker
from biomechanics.config import BiomechanicsConfig, load_pipeline_config
from biomechanics.diagnosis.bridge import build_anthro_dict, build_rom_dict
from biomechanics.diagnosis.demo_builder import build_demo_data
from biomechanics.diagnosis.engine import HypothesisEngine
from biomechanics.diagnosis.rep_scoring import score_set
from biomechanics.utils.json_safe import nan_to_none
from biomechanics.diagnosis.types import SetFeatures
from biomechanics.triangulation.calibration import (
    WORLD_ANCHOR_BOARD,
    CalibrationResult,
    TPoseCalibrator,
    rig_calibration_path,
)
from biomechanics.triangulation.person_calibration import (
    PersonCalibrationResult,
    PersonCalibrator,
    reprojection_health_px,
)
from biomechanics.triangulation.triangulator import DLTTriangulator
from biomechanics.utils.types import CocoKeypoints as CK
from biomechanics.utils.types import MultiViewPose, PipelineFrame, Skeleton2D
from biomechanics.viz import draw_skeleton, draw_fps, FPSCounter
from biomechanics.viz.live_stream import get_display_sink
from biomechanics.viz.demo_ws_bridge import (
    DemoWSBridge,
    FAULT_HIGHLIGHT_JOINTS,
    mediapipe_to_viewer_coords,
)
from biomechanics.viz.set_plots import plot_hip_position, plot_hip_velocity, make_output_dir
from biomechanics.viz.valgus_debug import build_recorder
from biomechanics.viz.pipeline_inspector import build_inspector
from biomechanics.analysis.set_finalizer import SetDataCollector, finalize_set
from biomechanics.viz.html_dashboard import generate_session_dashboard


# Ensure gate diagnostics are visible on stderr
logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")

DEMO_START_TIMEOUT_S = 10.0
DEMO_INACTIVITY_TIMEOUT_S = 45.0
# Covers settle + final hold + morph-out (~2.2s) plus browser latency.
DEMO_FINISH_TIMEOUT_S = 6.0

# Re-arm the readiness gate this long before rest ends, so the gate is
# already latched when the timer expires and the user can start at once.
GATE_PREARM_SECONDS = 3.0

# ROM baseline used until calibration reps measure the real peaks.
DEFAULT_ROM_BASELINE = {"peakDorsi": 35.0, "peakKneeFlex": 120.0}

# Body measurement accumulates during the assessment reps. Reps finished
# before it completes are held (their bottom frames are kept) so they still
# get kinematics; past this wait after the last rep they go through without.
ASSESSMENT_MEASUREMENT_WAIT_S = 5.0

# Camera extrinsics are a property of the rig, so they persist across sessions: one
# file per camera set (calibration.rig_calibration_path), reused until deleted. Refines
# made in the field go to a sibling file so the factory / first calibration is never overwritten.
REFINED_CALIBRATION_SUFFIX = "_refined"
FALLBACK_USER_HEIGHT_M = 1.885

# No calibration file: the lifter is the calibration target. Hershey fonts are ASCII-only.
CALIBRATION_CAPTURE_HEADLINE = "CALIBRATING CAMERAS - STEP IN AND DO TWO SLOW SQUATS"
CALIBRATION_SOLVING_HEADLINE = "CALIBRATING CAMERAS - SOLVING, ONE MOMENT"
WORLD_ANCHOR_HEADLINE = "ALIGNING CAMERAS TO YOU - ONE MOMENT"
TPOSE_FALLBACK_HEADLINE = "HOLD A T-POSE FACING THE FRONT CAMERA"
TPOSE_COUNTDOWN_S = 5.0
TPOSE_MAX_ATTEMPTS = 3

# A refine may not change the metric scale: the session's body measurements are lengths in
# the installed calibration's scale. Reprojection health cannot see a scale change (finding:
# a camera moved along the cam0-cam1 baseline reads as -2 % / +4 %), leg lengths can; a
# noise-only refine reads within 0.6 %.
MAX_REFINE_SCALE_CHANGE_RATIO = 0.01
LEG_SEGMENTS = (
    (CK.LEFT_HIP, CK.LEFT_KNEE), (CK.RIGHT_HIP, CK.RIGHT_KNEE),
    (CK.LEFT_KNEE, CK.LEFT_ANKLE), (CK.RIGHT_KNEE, CK.RIGHT_ANKLE),
)

# Squat-like excursion from the 2D knee-flexion proxy: down for a few frames, then back up.
SQUAT_DOWN_KNEE_FLEXION_2D_DEG = 60.0
SQUAT_UP_KNEE_FLEXION_2D_DEG = 25.0
SQUAT_MIN_DOWN_FRAMES = 3
MIN_LEG_KEYPOINT_SCORE = 0.5
LEG_KEYPOINT_TRIPLES = (
    (CK.LEFT_HIP, CK.LEFT_KNEE, CK.LEFT_ANKLE),
    (CK.RIGHT_HIP, CK.RIGHT_KNEE, CK.RIGHT_ANKLE),
)

# --- HUD styling (BGR brand palette) ---
HUD_CYAN = (255, 229, 0)
HUD_VIOLET = (246, 92, 139)
HUD_INK = (245, 240, 235)
HUD_PANEL = (16, 5, 5)
HUD_PANEL_ALPHA = 0.62
HUD_FONT = cv2.FONT_HERSHEY_DUPLEX

# On-screen hints for why the standing gate is rejecting frames.
GATE_FAILURE_HINTS = {
    "visibility": "STEP BACK - FULL BODY IN VIEW",
    "knee_extension": "STAND TALL - STRAIGHTEN KNEES",
    "torso_upright": "STAND UPRIGHT",
    "leg_extension": "TRACKING LOST - CHECK CAMERA ANGLE",
    "tracking_lost": "TRACKING LOST - CHECK CAMERA ANGLE",
    "distance": "ADJUST DISTANCE FROM CAMERA",
}


def _draw_rounded_rect(img, pt1, pt2, color, radius, thickness):
    x1, y1 = pt1
    x2, y2 = pt2
    if thickness < 0:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        for cx, cy in ((x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                       (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)):
            cv2.circle(img, (cx, cy), radius, color, -1, cv2.LINE_AA)
    else:
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)


def _draw_hud_pill(frame, text, align="right", accent=HUD_CYAN, font_scale=0.55, y=16):
    """Glass HUD pill: dark alpha-blended rounded rect, accent dot, uppercase text."""
    label = text.upper()
    text_thickness = 2 if font_scale > 0.8 else 1
    (tw, th), _ = cv2.getTextSize(label, HUD_FONT, font_scale, text_thickness)
    pad_x = max(14, int(th * 0.9))
    pad_y = max(10, int(th * 0.55))
    dot_gap = pad_x - 2
    pill_h = th + pad_y * 2
    pill_w = tw + pad_x * 2 + dot_gap
    frame_h, frame_w = frame.shape[:2]
    x = frame_w - pill_w - 16 if align == "right" else 16
    x2, y2 = x + pill_w, y + pill_h
    if x < 0 or y < 0 or x2 > frame_w or y2 > frame_h:
        return frame
    radius = pill_h // 2

    # Alpha-blend the panel over the frame (ROI-only — cheap)
    roi = frame[y:y2, x:x2]
    panel = roi.copy()
    _draw_rounded_rect(panel, (0, 0), (pill_w - 1, pill_h - 1), HUD_PANEL, radius, -1)
    cv2.addWeighted(panel, HUD_PANEL_ALPHA, roi, 1 - HUD_PANEL_ALPHA, 0, dst=roi)

    border = tuple(int(c * 0.5) for c in accent)
    _draw_rounded_rect(frame, (x, y), (x2 - 1, y2 - 1), border, radius, 1)

    cy = y + pill_h // 2
    cv2.circle(frame, (x + pad_x, cy), 3, accent, -1, cv2.LINE_AA)
    cv2.putText(frame, label, (x + pad_x + dot_gap, y + pad_y + th),
                HUD_FONT, font_scale, HUD_INK, text_thickness, cv2.LINE_AA)
    return frame


def _draw_wordmark(frame):
    frame_h = frame.shape[0]
    cv2.putText(frame, "N O W V A", (18, frame_h - 18),
                HUD_FONT, 0.5, (150, 138, 112), 1, cv2.LINE_AA)


def _show_status_frame(
    pipeline: BiomechanicsPipeline,
    result: PipelineFrame,
    display_sink,
    fps_counter: FPSCounter,
    headline: str,
    detail: str | None,
) -> None:
    frame = pipeline.last_frame
    if frame is None:
        return
    display = frame.copy()
    if result.skeleton_2d is not None:
        draw_skeleton(display, result.skeleton_2d)
    fps_counter.update()
    draw_fps(display, fps_counter.fps)
    _draw_hud_pill(display, headline, accent=HUD_VIOLET)
    if detail is not None:
        _draw_hud_pill(display, detail, align="left", accent=HUD_CYAN, font_scale=1.1, y=48)
    _draw_wordmark(display)
    display_sink.show(display)


def _save_calibration_report(peaks: dict, profile: dict, cal_reps: int, out_dir: str):
    """Save calibration_profile.json and calibration_profile.md to output dir."""
    # JSON
    cal_json = {
        "calibration_reps": cal_reps,
        "peaks": peaks,
        "profile": {k: v for k, v in profile.items() if k != "defaults"},
        "defaults": profile.get("defaults", {}),
    }
    json_path = str(Path(out_dir) / "calibration_profile.json")
    with open(json_path, "w") as f:
        json.dump(cal_json, f, indent=2)
    print(f"  Saved: {json_path}")

    # Markdown
    defaults = profile.get("defaults", {})
    kv = profile["knee_valgus"]
    fl = profile["forward_lean"]
    ba = profile["bilateral_asymmetry"]
    dp = profile.get("depth", {})
    d_kv = defaults.get("knee_valgus", {})
    d_fl = defaults.get("forward_lean", {})
    d_ba = defaults.get("bilateral_asymmetry", {})
    d_dp = defaults.get("depth", {})

    md_lines = [
        f"# Calibration Profile ({cal_reps} reps)",
        "",
        "## Observed Peaks",
        "",
        "| Signal | Value |",
        "|--------|-------|",
        f"| Trunk Flexion (peak) | {peaks['trunk_flexion']:.1f}° |",
        *[f"| Hip Adduction Rep {i+1} | {p:.1f}° |" for i, p in enumerate(peaks.get('hip_adduction_per_rep', []))],
        f"| **Hip Adduction (avg → band)** | **±{peaks['hip_adduction']:.1f}°** |",
        f"| Bilateral Asymmetry (peak) | {peaks['asymmetry']:.1f}° |",
        f"| Peak Dorsiflexion | {peaks['peak_dorsiflexion']:.1f}° |",
        f"| **Avg Squat Depth** | **{peaks.get('avg_depth', 0):.1f}°** |",
        *[f"| Squat Depth Rep {i+1} | {d:.1f}° |" for i, d in enumerate(peaks.get('depth_per_rep', []))],
        "",
        "## Calibrated Thresholds",
        "",
        "### Fault Thresholds (Mild / Moderate / Severe)",
        "",
        "| Fault | Calibrated | Default |",
        "|-------|-----------|---------|",
        f"| Knee Valgus | {kv['mild']:.1f}° / {kv['moderate']:.1f}° / {kv['severe']:.1f}° | {d_kv.get('mild', '-')}° / {d_kv.get('moderate', '-')}° / {d_kv.get('severe', '-')}° |",
        f"| Forward Lean | {fl['mild']:.1f}° / {fl['moderate']:.1f}° / {fl['severe']:.1f}° | {d_fl.get('mild', '-')}° / {d_fl.get('moderate', '-')}° / {d_fl.get('severe', '-')}° |",
        f"| Bilateral Asymmetry | {ba['mild']:.1f}° / {ba['moderate']:.1f}° / {ba['severe']:.1f}° | {d_ba.get('mild', '-')}° / {d_ba.get('moderate', '-')}° / {d_ba.get('severe', '-')}° |",
        "",
        "### Depth Thresholds",
        "",
        "| Category | Calibrated | Default |",
        "|----------|-----------|---------|",
        f"| Deep Squat (below parallel) | {dp.get('parallel_threshold', '-')}° | {d_dp.get('parallel_threshold', '-')}° |",
        f"| Parallel | {dp.get('half_threshold', '-')}° | {d_dp.get('half_threshold', '-')}° |",
        f"| Half Squat | {dp.get('quarter_threshold', '-')}° | {d_dp.get('quarter_threshold', '-')}° |",
        "",
    ]
    md_path = str(Path(out_dir) / "calibration_profile.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"  Saved: {md_path}")


def _throttle(pipeline, result, target_fps: float) -> None:
    # Multi-camera get_pose already waits for the next synced frame; sleeping on
    # top of it skips frames because the next call returns the newest set.
    if pipeline._multi_camera:
        return
    total_ms = sum(result.latency_ms.values())
    target_ms = 1000.0 / target_fps
    if total_ms < target_ms:
        time.sleep((target_ms - total_ms) / 1000.0)


def _extract_athlete_params(pipeline) -> dict | None:
    return pipeline.body_calibration.to_athlete_params()


def _adopt_measured_athlete_params(pipeline, session_tracker, bridge, baseline: dict) -> dict | None:
    athlete_params = _extract_athlete_params(pipeline)
    if athlete_params is not None:
        session_tracker.set_athlete_params(athlete_params, baseline)
        bridge.set_athlete_params(athlete_params, baseline)
        print(
            f"  [DIAGNOSIS] Body measured: "
            f"shoulder={athlete_params['shoulder_width_m']:.3f}m "
            f"femur={athlete_params['femur_avg_m']:.3f}m "
            f"tibia={athlete_params['tibia_avg_m']:.3f}m"
        )
    return athlete_params


def _build_calibration_complete_message(
    movement_pattern: str,
    peaks: dict,
    cal_profile: dict,
    athlete_params: dict | None,
    baseline: dict,
) -> dict:
    message = {
        "type": "calibration_complete",
        "movement_pattern": movement_pattern,
        "peaks": peaks,
        "thresholds": {k: v for k, v in cal_profile.items() if k != "defaults"},
    }
    # Omitted, never None, when the body was not measured: the voice agent
    # persists whatever arrives here.
    if athlete_params is not None:
        message["athlete_params"] = athlete_params
        message["baseline"] = baseline
    return message


def _resolve_user_height_m(user_height_cm: float | None) -> float:
    if user_height_cm:
        return user_height_cm / 100.0
    height_m = float(os.getenv("NOWVA_USER_HEIGHT_M", str(FALLBACK_USER_HEIGHT_M)))
    print(
        f"[MULTI-CAM] WARNING: no user height in profile — camera calibration "
        f"assumes {height_m} m (NOWVA_USER_HEIGHT_M); without a barbell every 3D length scales with it"
    )
    return height_m


def refined_calibration_path(factory_path: Path) -> Path:
    return factory_path.with_name(f"{factory_path.stem}{REFINED_CALIBRATION_SUFFIX}{factory_path.suffix}")


def _file_timestamps(path: Path) -> tuple[str, str]:
    with open(path) as file:
        data = json.load(file)
    return data.get("timestamp", ""), data.get("source_timestamp", "")


def select_calibration_file(factory_path: Path) -> Path | None:
    """
    Load order: the refined calibration when it descends from the factory / previous file
    (its source_timestamp is that file's timestamp), else that file, else None. A refined
    file left over from an earlier factory calibration is ignored.
    """
    if not factory_path.exists():
        return None
    refined_path = refined_calibration_path(factory_path)
    if refined_path.exists():
        factory_timestamp, _ = _file_timestamps(factory_path)
        _, source_timestamp = _file_timestamps(refined_path)
        if factory_timestamp and source_timestamp == factory_timestamp:
            return refined_path
    return factory_path


def leg_length_ratio(
    calibration: CalibrationResult, views: list[dict[str, Skeleton2D]], athlete_params: dict
) -> float:
    """
    Median femur + tibia length triangulated with `calibration` over the frames, divided by
    the session's known femur + tibia (athlete_params femur_avg_m + tibia_avg_m). 1.0 means
    the calibration reproduces the scale the body was measured in. NaN without leg data.
    """
    triangulator = DLTTriangulator(calibration)
    femur_lengths_m: list[float] = []
    tibia_lengths_m: list[float] = []
    for frame_idx, frame_views in enumerate(views):
        skeleton = triangulator.triangulate(MultiViewPose(views=frame_views, timestamp=float(frame_idx)))
        if skeleton is None:
            continue
        points = skeleton.to_numpy()
        confidences = np.array([keypoint.confidence for keypoint in skeleton.keypoints])
        for proximal, distal in LEG_SEGMENTS:
            if confidences[proximal] <= 0.0 or confidences[distal] <= 0.0:
                continue
            length_m = float(np.linalg.norm(points[proximal] - points[distal]))
            (femur_lengths_m if proximal in (CK.LEFT_HIP, CK.RIGHT_HIP) else tibia_lengths_m).append(length_m)
    if not femur_lengths_m or not tibia_lengths_m:
        return math.nan
    measured_leg_m = float(np.median(femur_lengths_m) + np.median(tibia_lengths_m))
    return measured_leg_m / (athlete_params["femur_avg_m"] + athlete_params["tibia_avg_m"])


def knee_flexion_2d_deg(views: dict[str, Skeleton2D]) -> float:
    """
    Largest 2D knee flexion over cameras and legs, in degrees; NaN when no camera sees a whole
    leg. Foreshortening only ever lowers the 2D angle, so the most side-on camera reads closest.
    """
    flexions_deg = []
    for skeleton in views.values():
        keypoints = skeleton.to_numpy()
        for hip, knee, ankle in LEG_KEYPOINT_TRIPLES:
            if keypoints[[hip, knee, ankle], 2].min() < MIN_LEG_KEYPOINT_SCORE:
                continue
            thigh = keypoints[hip, :2] - keypoints[knee, :2]
            shank = keypoints[ankle, :2] - keypoints[knee, :2]
            length_product = float(np.linalg.norm(thigh) * np.linalg.norm(shank))
            if length_product <= 0.0:
                continue
            cosine = float(np.clip(np.dot(thigh, shank) / length_product, -1.0, 1.0))
            flexions_deg.append(180.0 - math.degrees(math.acos(cosine)))
    return max(flexions_deg) if flexions_deg else math.nan


class SquatExcursionCounter:
    """Counts squat-like excursions of the 2D knee-flexion proxy: held down a few frames, then back up."""

    def __init__(self) -> None:
        self.count = 0
        self._down_frames = 0
        self._is_down = False

    def update(self, knee_flexion_deg: float) -> None:
        if math.isnan(knee_flexion_deg):
            return
        if knee_flexion_deg >= SQUAT_DOWN_KNEE_FLEXION_2D_DEG:
            self._down_frames += 1
            if self._down_frames >= SQUAT_MIN_DOWN_FRAMES:
                self._is_down = True
            return
        self._down_frames = 0
        if self._is_down and knee_flexion_deg <= SQUAT_UP_KNEE_FLEXION_2D_DEG:
            self._is_down = False
            self.count += 1


class CalibrationSolve:
    """
    One person-calibration solve on a worker thread, so frames keep flowing. Never a forked
    process: on macOS Accelerate deadlocks in forked children.
    """

    def __init__(self, solve_fn: Callable[[], PersonCalibrationResult]) -> None:
        self.result: PersonCalibrationResult | None = None
        self.error: Exception | None = None
        self._solve_fn = solve_fn
        self._thread = threading.Thread(target=self._run, name="camera-calibration", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        # Any failed solve has a fallback (T-pose, or keeping the current calibration),
        # so it is reported to the session instead of killing the thread silently.
        try:
            self.result = self._solve_fn()
        except Exception as error:
            self.error = error

    @property
    def done(self) -> bool:
        return not self._thread.is_alive()


ShowStatusFn = Callable[[PipelineFrame, str, str | None], None]


class CameraCalibrationSession:
    """
    Owns the rig's camera calibration for one multi-camera session: loads it or bootstraps
    it from the lifter's own squats, re-anchors a board (factory) calibration on the lifter,
    and refines it between sets when the reprojection health drifts.
    """

    def __init__(
        self,
        pipeline: BiomechanicsPipeline,
        config: BiomechanicsConfig,
        factory_path: Path,
        user_height_cm: float | None,
        show_status_fn: ShowStatusFn,
        clock_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._pipeline = pipeline
        self._provider = pipeline._multi_camera_provider
        self._settings = config.camera_calibration
        self._bar_length_m = config.barbell_tracking.bar_length_m
        self._factory_path = factory_path
        self._refined_path = refined_calibration_path(factory_path)
        self._user_height_cm = user_height_cm
        self._show_status_fn = show_status_fn
        self._clock_fn = clock_fn

        # Drift baseline: the RMS stored with the calibration; 0 = measure it at the first set boundary.
        self._baseline_rms_px = 0.0
        self._world_anchor_pending = False
        # `timestamp` of the factory / first calibration every refined file is linked back to.
        self._source_timestamp = ""
        # Background refine between sets: the solve, the health that triggered it, and the
        # held-out frames (never solved on) that judge it.
        self._refine: CalibrationSolve | None = None
        self._refine_keeps_world_frame = True
        self._refine_health_before_px = math.nan
        self._refine_check_views: list[dict[str, Skeleton2D]] = []

    @property
    def needs_world_anchor(self) -> bool:
        """A board-anchored (factory) calibration is loaded and has not been re-anchored on the lifter yet."""
        return self._world_anchor_pending

    @property
    def is_refining(self) -> bool:
        """A between-set refine is running or waiting for its install window."""
        return self._refine is not None

    def establish(self) -> None:
        """
        Refined file (if derived from it) -> factory / previous file -> bootstrap from the
        lifter -> T-pose. A file for other cameras or another capture resolution is ignored.
        """
        selected_path = select_calibration_file(self._factory_path)
        if selected_path is None:
            self._bootstrap()
            return
        calibration = TPoseCalibrator.load_calibration(str(selected_path))
        mismatches = self._provider.calibration_mismatches(calibration)
        if mismatches:
            print(
                f"[MULTI-CAM] WARNING: rig calibration {selected_path} does not describe this rig "
                f"({'; '.join(mismatches)}) — recalibrating"
            )
            self._bootstrap()
            return
        self._provider.install_calibration(calibration)
        self._adopt(calibration)
        self._source_timestamp = calibration.source_timestamp or calibration.timestamp
        print(
            f"[MULTI-CAM] Loaded rig calibration {selected_path} "
            f"(delete rig_calibration_cams_* in {selected_path.parent} to recalibrate after moving cameras)"
        )
        if self._world_anchor_pending:
            print("[MULTI-CAM] Board-anchored calibration: the world frame moves onto the lifter after the first reps")

    def anchor_world_on_lifter(self) -> None:
        """
        One refine of a board-anchored calibration on the buffered views (the reps just done),
        re-anchoring the world frame on the lifter. Frames keep flowing while it solves; the
        factory file is never overwritten. Call between sets only.
        """
        refine = self._start_refine(self._provider.calibration_views(), keep_world_frame=False)
        was_presence_only = self._pipeline.presence_only
        self._pipeline.presence_only = True
        self._show_until_done(refine, WORLD_ANCHOR_HEADLINE)
        self._pipeline.presence_only = was_presence_only
        self._finish_refine()

    def on_rest_start(self) -> None:
        """
        A set just ended: install a refine that finished too late for the previous rest, else
        check the reprojection health of the set's last frames and start a refine if it drifted.
        """
        if self._refine is not None:
            if self._refine.done:
                self._finish_refine()
            return
        views = self._provider.calibration_views()
        if self._world_anchor_pending:
            self._start_refine(views, keep_world_frame=False)
            return

        check_frames = self._settings.drift_check_frames
        check_views = views[-check_frames:]
        health_px = reprojection_health_px(self._provider.calibration, check_views)
        if not math.isfinite(health_px):
            return
        if not self._baseline_rms_px > 0.0:
            self._baseline_rms_px = health_px
            print(f"[MULTI-CAM] Reprojection baseline measured: {health_px:.2f} px")
            return
        limit_px = self._settings.drift_ratio * self._baseline_rms_px
        if health_px <= limit_px:
            print(f"[MULTI-CAM] Reprojection health {health_px:.2f} px (limit {limit_px:.2f} px) — calibration holds")
            return
        # The refine solves on the frames before the check window, so the health it is judged
        # on is out of sample: a solve that only chases this set's noise cannot pass.
        solve_views = views[:-check_frames]
        if not solve_views:
            print(
                f"[MULTI-CAM] Reprojection health {health_px:.2f} px exceeds {limit_px:.2f} px but only "
                f"{len(views)} frames are buffered — refine skipped until a longer set"
            )
            return
        print(
            f"[MULTI-CAM] Reprojection health {health_px:.2f} px exceeds {limit_px:.2f} px "
            f"({self._settings.drift_ratio:.1f} x {self._baseline_rms_px:.2f}) — refining cameras during rest"
        )
        self._refine_health_before_px = health_px
        self._start_refine(solve_views, keep_world_frame=True)

    def poll(self, can_install: bool) -> None:
        """Every frame. can_install is True only between sets (resting before the gate re-arms): never mid-set."""
        if self._refine is not None and self._refine.done and can_install:
            self._finish_refine()

    def _adopt(self, calibration: CalibrationResult) -> None:
        stored_rms_px = calibration.rms_reprojection_px
        self._baseline_rms_px = stored_rms_px if math.isfinite(stored_rms_px) else 0.0
        self._world_anchor_pending = calibration.world_anchor == WORLD_ANCHOR_BOARD

    def _install(self, calibration: CalibrationResult, world_frame_changed: bool) -> None:
        self._provider.install_calibration(calibration)
        self._pipeline.on_calibration_changed(world_frame_changed)
        self._adopt(calibration)

    def _show_until_done(self, solve: CalibrationSolve, headline: str) -> None:
        while not solve.done:
            self._show_status_fn(self._pipeline.process_frame(), headline, None)

    def _bootstrap(self) -> None:
        settings = self._settings
        provider = self._provider
        height_m = _resolve_user_height_m(self._user_height_cm)
        print(
            f"[MULTI-CAM] No rig calibration at {self._factory_path} — calibrating from the lifter: "
            f"step in and do two slow squats"
        )

        self._pipeline.presence_only = True
        provider.begin_calibration_capture()
        excursions = SquatExcursionCounter()
        newest_views: dict[str, Skeleton2D] | None = None
        deadline_s = self._clock_fn() + settings.capture_timeout_s
        captured = False
        while self._clock_fn() < deadline_s:
            result = self._pipeline.process_frame()
            recent = provider.recent_views(1)
            if recent and recent[0] is not newest_views:
                newest_views = recent[0]
                excursions.update(knee_flexion_2d_deg(newest_views))
            frame_count = provider.calibration_frame_count
            if frame_count >= settings.min_calibration_frames and excursions.count >= settings.min_squat_excursions:
                captured = True
                break
            self._show_status_fn(
                result,
                CALIBRATION_CAPTURE_HEADLINE,
                f"FRAMES {min(frame_count, settings.min_calibration_frames)}/{settings.min_calibration_frames}"
                f"  SQUATS {min(excursions.count, settings.min_squat_excursions)}/{settings.min_squat_excursions}",
            )
        provider.end_calibration_capture()

        failure = f"no two slow squats seen by two cameras within {settings.capture_timeout_s:.0f} s"
        if captured:
            views = provider.calibration_views()
            bar_ends = provider.calibration_bar_ends()
            calibrator = PersonCalibrator(
                provider.rig_intrinsics(),
                provider.capture_resolution,
                bar_length_m=self._bar_length_m if provider.has_bar_detector else None,
                height_m=height_m,
            )
            solve = CalibrationSolve(lambda: calibrator.calibrate(views, bar_ends, initial=None))
            self._show_until_done(solve, CALIBRATION_SOLVING_HEADLINE)
            if solve.result is not None:
                self._save_and_install(solve.result, self._factory_path, world_frame_changed=True)
                self._pipeline.presence_only = False
                return
            failure = f"person-based calibration failed ({solve.error})"

        if self._pipeline.last_frame is None:
            raise RuntimeError(
                f"[MULTI-CAM] {failure}, and the cameras delivered no frame — check the camera connections"
            )
        self._tpose_fallback(failure, height_m)
        self._pipeline.presence_only = False

    def _tpose_fallback(self, reason: str, height_m: float) -> None:
        """
        The T-pose calibration, retried a bounded number of times when the subject is not
        holding the pose (ValueError). Stalled cameras or no solvable camera (RuntimeError)
        fail the session at once.
        """
        print(
            f"[MULTI-CAM] {reason} — falling back to the T-pose calibration (height={height_m} m): "
            f"face the front camera with both arms straight out at shoulder height"
        )
        self._factory_path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, TPOSE_MAX_ATTEMPTS + 1):
            countdown_end_s = self._clock_fn() + TPOSE_COUNTDOWN_S
            while self._clock_fn() < countdown_end_s:
                remaining_s = math.ceil(countdown_end_s - self._clock_fn())
                self._show_status_fn(
                    self._pipeline.process_frame(), TPOSE_FALLBACK_HEADLINE, f"T-POSE IN {remaining_s}",
                )
            try:
                calibration = self._provider.calibrate(height_m=height_m, save_path=str(self._factory_path))
            except ValueError as error:
                print(f"[MULTI-CAM] T-pose attempt {attempt}/{TPOSE_MAX_ATTEMPTS}: {error}")
                continue
            self._pipeline.on_calibration_changed(world_frame_changed=True)
            self._adopt(calibration)
            self._source_timestamp = calibration.timestamp
            print(f"[MULTI-CAM] T-pose calibration complete — saved to {self._factory_path}")
            return
        raise RuntimeError(
            f"[MULTI-CAM] T-pose calibration failed {TPOSE_MAX_ATTEMPTS} times — no camera calibration; "
            f"restart with the lifter fully visible to every camera"
        )

    def _save_and_install(
        self, result: PersonCalibrationResult, save_path: Path, world_frame_changed: bool
    ) -> None:
        calibration = result.calibration
        calibration.rms_reprojection_px = result.rms_reprojection_px
        if save_path == self._factory_path:
            self._source_timestamp = calibration.timestamp
        else:
            calibration.source_timestamp = self._source_timestamp
        save_path.parent.mkdir(parents=True, exist_ok=True)
        TPoseCalibrator.save_calibration(calibration, str(save_path))
        self._install(calibration, world_frame_changed)
        print(
            f"[MULTI-CAM] Camera calibration from {result.frames_used} frames (scale: {result.scale_source}): "
            f"reprojection {result.initial_rms_reprojection_px:.2f} -> {result.rms_reprojection_px:.2f} px "
            f"in {result.solve_time_s:.1f} s — saved to {save_path}"
        )

    def _start_refine(self, views: list[dict[str, Skeleton2D]], keep_world_frame: bool) -> CalibrationSolve:
        calibration = self._provider.calibration
        self._refine_check_views = self._provider.recent_views(self._settings.drift_check_frames)
        # A refine keeps the intrinsics and the metric scale of the calibration it starts from
        # (no bar or height prior): segment lengths measured earlier in the session stay valid.
        calibrator = PersonCalibrator(
            {
                cam_id: (camera.intrinsic_matrix, camera.distortion_coeffs)
                for cam_id, camera in calibration.cameras.items()
            },
            self._provider.capture_resolution,
        )
        self._refine_keeps_world_frame = keep_world_frame
        self._refine = CalibrationSolve(
            lambda: calibrator.refine(calibration, views, keep_world_frame=keep_world_frame)
        )
        return self._refine

    def _finish_refine(self) -> None:
        refine = self._refine
        keeps_world_frame = self._refine_keeps_world_frame
        self._refine = None
        check_views = self._refine_check_views
        self._refine_check_views = []
        if refine.result is None:
            print(f"[MULTI-CAM] Camera refine failed ({refine.error}) — keeping the current calibration")
            return
        refined = refine.result.calibration
        if keeps_world_frame:
            health_after_px = reprojection_health_px(refined, check_views)
            health_before_px = self._refine_health_before_px
            limit_px = self._settings.drift_ratio * self._baseline_rms_px
            healthy = health_after_px <= limit_px or health_after_px < health_before_px / self._settings.drift_ratio
            if not healthy:
                # Judged on frames it never solved on, the refine explains nothing the current
                # rig does not: this is how well these cameras fit these frames. New baseline.
                self._baseline_rms_px = health_before_px
                print(
                    f"[MULTI-CAM] Drift refine did not help on held-out frames ({health_before_px:.2f} -> "
                    f"{health_after_px:.2f} px, limit {limit_px:.2f}) — keeping the current calibration"
                )
                return
            print(f"[MULTI-CAM] Drift refine: held-out reprojection health {health_before_px:.2f} -> {health_after_px:.2f} px")
        if not self._scale_is_kept(refined, check_views):
            return
        self._save_and_install(refine.result, self._refined_path, world_frame_changed=not keeps_world_frame)

    def _scale_is_kept(self, refined: CalibrationResult, check_views: list[dict[str, Skeleton2D]]) -> bool:
        athlete_params = self._pipeline.body_calibration.to_athlete_params()
        if athlete_params is None:
            print("[MULTI-CAM] Body not measured yet — refine scale unchecked")
            return True
        ratio = leg_length_ratio(refined, check_views, athlete_params)
        if not math.isfinite(ratio):
            print("[MULTI-CAM] No leg seen by two cameras in the check frames — refine scale unchecked")
            return True
        if abs(ratio - 1.0) <= MAX_REFINE_SCALE_CHANGE_RATIO:
            return True
        print(
            f"[MULTI-CAM] Refine rejected: it changes the leg length by {100.0 * (ratio - 1.0):+.2f} % "
            f"(limit {100.0 * MAX_REFINE_SCALE_CHANGE_RATIO:.0f} %) — a camera moved along the baseline; "
            f"recalibrate by deleting {self._factory_path}"
        )
        return False


def _serialize_diagnosis(diagnosis_result, score_summary) -> tuple[dict, dict]:
    """Convert diagnosis engine output into JSON-serializable dicts for IPC."""
    per_rep = score_summary.per_rep_scores
    n = len(per_rep)
    diagnosis_dict = {
        "confidence": diagnosis_result.confidence,
        "detected_symptoms": [
            {"symptom_id": s.symptom_id, "severity": s.severity, "contributing_reps": s.contributing_reps}
            for s in diagnosis_result.detected_symptoms
        ],
        "immediate_causes": [
            {"cause_id": c.cause_id, "score": c.score, "explanation": c.explanation, "parameter_delta": c.parameter_delta}
            for c in diagnosis_result.immediate_causes
        ],
        "session_causes": [
            {"cause_id": c.cause_id, "score": c.score, "explanation": c.explanation}
            for c in diagnosis_result.session_causes
        ],
        "contextual_notes": [
            {"cause_id": c.cause_id, "score": c.score, "explanation": c.explanation}
            for c in diagnosis_result.contextual_notes
        ],
        "combined_perturbation": diagnosis_result.combined_perturbation,
    }
    scoring_dict = {
        "mean_score": score_summary.mean_score,
        "per_dimension": {
            "depth": round(sum(r.depth_score for r in per_rep) / n, 3) if n else 0,
            "trunk_control": round(sum(r.trunk_control_score for r in per_rep) / n, 3) if n else 0,
            "knee_tracking": round(sum(r.knee_tracking_score for r in per_rep) / n, 3) if n else 0,
            "symmetry": round(sum(r.symmetry_score for r in per_rep) / n, 3) if n else 0,
            "tempo": round(sum(r.tempo_score for r in per_rep) / n, 3) if n else 0,
        },
        "best_rep": score_summary.best_rep_number,
        "worst_rep": score_summary.worst_rep_number,
        "trend_slope": score_summary.trend_slope,
    }
    return diagnosis_dict, scoring_dict


def _poll_incoming_message(ipc_client: IPCClient):
    """Non-blocking check for a message from main.py."""
    try:
        sock = ipc_client.client_socket
        if sock is None:
            return None
        ready, _, _ = select.select([sock], [], [], 0)
        if ready:
            return _recv_framed(sock)
    except Exception:
        pass
    return None


def _skeleton_pixels(skeleton_2d) -> np.ndarray | None:
    keypoints = skeleton_2d.keypoints
    if len(keypoints) < 17:
        return None
    if any(kp.confidence <= 0 for kp in keypoints[:17]):
        return None
    return np.array([[kp.x, kp.y] for kp in keypoints[:17]], dtype=np.float32)


def _restart_demo_bridge(existing: DemoWSBridge | None) -> DemoWSBridge:
    """Replace any live viewer bridge with a fresh one and show it on the display.

    Stopping first matters: the bridge binds a fixed port, so starting a
    second one on top of a live one leaves the new server thread dead.
    The display announce runs off-thread so the frame loop never waits on
    the bind delay, and the bridge replays its init payload to whichever
    client connects, so nothing is lost by announcing late.
    """
    if existing is not None:
        existing.stop()
    bridge = DemoWSBridge()
    bridge.start()

    def _announce_viewer() -> None:
        time.sleep(0.3)  # let the server finish binding before the display iframes it
        get_display_sink().send_event({"type": "demo", "action": "start"})

    threading.Thread(target=_announce_viewer, daemon=True).start()
    return bridge


def _wait_for_start_capture(ipc_client: IPCClient) -> bool:
    """Block until a start_capture message arrives over IPC. Returns False on disconnect."""
    sock = ipc_client.client_socket
    if sock is None:
        return False
    while True:
        try:
            msg = _recv_framed(sock)
            if msg is None:
                return False
            if msg.get("type") == "start_capture":
                return True
        except Exception:
            return False


def run_biomechanics_pipeline(
    cam0_id: int = 0,
    cam1_id: int = 1,
    config_path: str = None,
    exercise_name: str = "Barbell Back Squat",
    calibration_file: str = None,
    calibration_mode: bool = False,
    calibration_reps: int = 5,
    preload: bool = False,
    user_height_cm: float | None = None,
):
    """
    Run the full biomechanics pipeline as a subprocess.

    This is called by main.py when workout mode starts.
    Connects to the existing IPC server and sends real-time
    coaching data to the voice agent.

    When preload=True the subprocess loads config + pose model eagerly
    (without opening the camera) and waits for a start_capture IPC
    message before opening the camera and entering the frame loop.
    This lets main.py overlap model loading with the voice greeting.
    """
    print("\n=== Biomechanics Pipeline Starting ===")

    # Handle SIGTERM gracefully so finally blocks run when main.py terminates us
    def _sigterm_handler(signum, frame):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, _sigterm_handler)

    # Connect to IPC server (started by main.py)
    ipc_client = IPCClient()
    if not ipc_client.connect(timeout=10):
        print("Failed to connect to IPC server. Exiting.")
        return

    # Load config
    config = load_pipeline_config(config_path)
    # Override camera device from CLI arg
    config.capture.device_id = cam0_id

    fps_counter = FPSCounter()
    display_sink = get_display_sink()
    camera_calibration: CameraCalibrationSession | None = None

    # Initialize pipeline
    try:
        pipeline = BiomechanicsPipeline(
            config, exercise_name=exercise_name, defer_capture=preload,
        )

        if preload:
            pipeline.preload_pose_model()
            print("[PRELOAD] Pose model loaded — waiting for start_capture signal")
            ipc_client.send_message({"type": "pipeline_status", "status": "preloaded"})

            if not _wait_for_start_capture(ipc_client):
                print("[PRELOAD] IPC disconnected while waiting for start_capture")
                ipc_client.disconnect()
                return

            pipeline.start_capture()
            print("[PRELOAD] Camera opened — entering frame loop")

        # Multi-camera calibration. The factory / previous file is config
        # triangulation.calibration_file when set, else this rig's file in ~/.nowva;
        # a newer "_refined" sibling wins, and with no file the lifter's first
        # squats calibrate the cameras (T-pose as the last resort).
        if pipeline._multi_camera and pipeline._multi_camera_provider is not None:
            camera_calibration = CameraCalibrationSession(
                pipeline,
                config,
                factory_path=(
                    Path(config.triangulation.calibration_file)
                    if config.triangulation.calibration_file
                    else rig_calibration_path(config.triangulation.device_ids)
                ),
                user_height_cm=user_height_cm,
                show_status_fn=lambda result, headline, detail: _show_status_frame(
                    pipeline, result, display_sink, fps_counter, headline, detail,
                ),
            )
            camera_calibration.establish()

        bridge = IPCBridge(ipc_client)
        session_tracker = SessionTracker(bridge, config=config.coaching)

        # Pre-cache coaching cues for the exercise
        bridge.prepare_exercise(exercise_name)

        ipc_client.send_message({"type": "status", "value": "initialized"})
        bridge.send_pipeline_status("running", {})

        print(f"Pipeline initialized for: {exercise_name}")
        print(f"Running at target {config.target_fps} FPS")
        print("Press Ctrl+C to stop\n")

    except Exception as e:
        print(f"Pipeline initialization failed: {e}")
        ipc_client.send_message({"type": "error", "value": str(e)})
        ipc_client.disconnect()
        return

    session_output = os.environ.get("NOWVA_SESSION_OUTPUT_DIR")
    out_dir = make_output_dir(base=os.path.join(session_output, "output") if session_output else "output")

    # --valgus debug recording. The tap wraps the IPC client so every message
    # sent from here on is captured; boot messages sent before this point are
    # not knee data. Both the local name and the bridge's reference are
    # swapped, since the bridge duck-types on send_message.
    valgus_recorder = build_recorder(out_dir, config.target_fps, pipeline._multi_camera)
    if valgus_recorder is not None:
        ipc_client = valgus_recorder.tap(ipc_client)
        bridge.ipc_client = ipc_client

    pipeline_inspector = build_inspector(
        out_dir, config.target_fps, pipeline._multi_camera, pipeline.preik_stage_names,
    )
    if pipeline_inspector is not None:
        pipeline.enable_inspect()

    # Session-scoped body measurements feeding diagnosis: restored from the
    # stored calibration, or adopted once the pipeline finishes measuring.
    athlete_params: dict | None = None
    athlete_baseline: dict = dict(DEFAULT_ROM_BASELINE)

    # --- Apply existing calibration if provided ---
    if calibration_file and os.path.exists(calibration_file):
        try:
            with open(calibration_file, "r") as f:
                stored = json.load(f)
            # Newer files nest thresholds alongside the athlete's body
            # proportions; older flat files are just the thresholds.
            cal_profile = stored.get("thresholds", stored)
            apply_calibration_to_rule_engine(pipeline._rule_engine, cal_profile)
            print(f"[CALIBRATION] Loaded calibration from {calibration_file}")

            # This is the only place a returning user's proportions get
            # wired — they skip the phase that would otherwise derive them.
            stored_params = stored.get("athlete_params")
            if stored_params:
                stored_baseline = stored.get("baseline") or {}
                pipeline.apply_athlete_params(stored_params)
                athlete_params = stored_params
                athlete_baseline = stored_baseline
                session_tracker.set_athlete_params(stored_params, stored_baseline)
                bridge.set_athlete_params(stored_params, stored_baseline)
                print(
                    f"  [DIAGNOSIS] Athlete params restored: "
                    f"shoulder={stored_params.get('shoulder_width_m', 0):.3f}m "
                    f"femur={stored_params.get('femur_avg_m', 0):.3f}m"
                )
            elif not calibration_mode:
                print(
                    "  [DIAGNOSIS] No stored athlete params — diagnosis and "
                    "stance coaching unavailable this session"
                )
        except Exception as e:
            print(f"[CALIBRATION] Failed to load calibration file: {e}")

    # --- Assessment + Calibration phase (if no existing calibration) ---
    if calibration_mode:
        movement_pattern = get_movement_pattern(exercise_name) or "squat"

        # ============================================================
        #  PHASE 1: PRE-WORKOUT FORM ASSESSMENT (2-rep loop)
        # ============================================================
        ASSESSMENT_TARGET_REPS = 1

        print(f"\n{'='*60}")
        print(f"  ASSESSMENT PHASE ({ASSESSMENT_TARGET_REPS} bodyweight squats)")
        print(f"{'='*60}\n")

        # Wait for the readiness gate only. Body measurements accumulate inside
        # the pipeline on every ready frame during the assessment reps.
        print("  [ASSESSMENT] Waiting for readiness gate...")
        try:
            while not pipeline.is_ready:
                result = pipeline.process_frame()

                frame = pipeline.last_frame
                if frame is not None:
                    display = frame.copy()
                    if result.skeleton_2d is not None:
                        draw_skeleton(display, result.skeleton_2d)
                    fps_counter.update()
                    draw_fps(display, fps_counter.fps)

                    readiness_gate = pipeline._readiness_gate
                    current, required = readiness_gate.progress
                    hint = GATE_FAILURE_HINTS.get(readiness_gate.last_failure, "HOLD STILL")
                    _draw_hud_pill(display, f"{hint}  {current}/{required}", accent=HUD_VIOLET)
                    _draw_wordmark(display)

                    display_sink.show(display)

                _throttle(pipeline, result, config.target_fps)
        except KeyboardInterrupt:
            print("\nAssessment stopped by user")
            pipeline.release()
            ipc_client.disconnect()
            return

        # Tracking is locked in — tell the agent it can cue the first squat
        ipc_client.send_message({"type": "assessment_ready"})
        print("  [ASSESSMENT] Gate passed — sent assessment_ready")

        def _run_demo_phase(demo_data) -> bool:
            """Drive the choreographed demo from agent messages. Returns False if user quit."""
            bridge = DemoWSBridge()
            bridge.start()
            time.sleep(0.3)
            display_sink.send_event({"type": "demo", "action": "start"})
            time.sleep(1.0)
            bridge.send_init(demo_data)

            phase_start = time.time()
            last_message_time = phase_start
            finishing = False
            finish_deadline = 0.0
            demo_started = False
            started_ack_sent = False

            try:
                while True:
                    result = pipeline.process_frame()

                    live_pose = result.skeleton_3d_display or result.skeleton_3d_raw
                    if live_pose is not None:
                        bridge.send_live_pose(live_pose.to_numpy())

                    if not started_ack_sent and bridge.wait_started(timeout=0):
                        ipc_client.send_message({"type": "demo_started"})
                        started_ack_sent = True

                    incoming = _poll_incoming_message(ipc_client)
                    if incoming is not None:
                        last_message_time = time.time()
                        msg_type = incoming.get("type")
                        if msg_type == "demo_start":
                            bridge.send_event({"type": "demo_start"})
                            demo_started = True
                        elif msg_type == "demo_cue":
                            bridge.send_event({
                                "type": "demo_cue",
                                "cue_index": int(incoming.get("cue_index", 0)),
                            })
                        elif msg_type == "demo_end":
                            bridge.send_event({"type": "demo_end"})
                            finishing = True
                            finish_deadline = time.time() + DEMO_FINISH_TIMEOUT_S

                    now = time.time()
                    if finishing:
                        if bridge.wait_done(timeout=0):
                            return True
                        if now > finish_deadline:
                            print("  [DEMO] Viewer never confirmed done — ending demo")
                            return True
                    if not demo_started and now - phase_start > DEMO_START_TIMEOUT_S:
                        print("  [DEMO] No demo_start received — skipping demo")
                        return True
                    if demo_started and not finishing and now - last_message_time > DEMO_INACTIVITY_TIMEOUT_S:
                        print("  [DEMO] Agent went silent mid-demo — ending demo")
                        bridge.send_event({"type": "demo_end"})
                        finishing = True
                        finish_deadline = now + DEMO_FINISH_TIMEOUT_S

                    frame = pipeline.last_frame
                    if frame is not None:
                        display = frame.copy()
                        if result.skeleton_2d is not None:
                            draw_skeleton(display, result.skeleton_2d)
                        fps_counter.update()
                        display_sink.show(display)

                    _throttle(pipeline, result, config.target_fps)
            finally:
                bridge.stop()
                display_sink.send_event({"type": "demo", "action": "end"})

        # Assessment loop: collect reps, diagnose, repeat until no immediate causes
        assessment_passed = False
        assessment_round = 0
        demo_played = False

        try:
            while not assessment_passed:
                assessment_round += 1
                assessment_reps_done = 0
                pipeline.rep_counter.reset()
                pipeline.rep_counter.set_assessment_mode(True)
                if pipeline._bilstm is not None:
                    pipeline._bilstm.set_assessment_mode(True)
                session_tracker.set_assessment_mode(True)
                session_tracker.reset_rep_buffers()
                session_tracker.current_set_reps = []
                session_tracker.set_active = False

                print(f"\n  [ASSESSMENT] Round {assessment_round} — collecting {ASSESSMENT_TARGET_REPS} reps")

                # Reps finished before the body is measured wait here with their
                # bottom frames, so they still get kinematics once it is.
                pending_reps: list[tuple] = []
                last_rep_time = 0.0

                while assessment_reps_done < ASSESSMENT_TARGET_REPS or pending_reps:
                    result = pipeline.process_frame()

                    if athlete_params is None:
                        athlete_params = _adopt_measured_athlete_params(
                            pipeline, session_tracker, bridge, athlete_baseline,
                        )

                    if pipeline.is_ready and result.skeleton_3d is not None:
                        bridge.send_frame_data(result, rep_phase=pipeline.rep_counter.phase)
                        for fault in result.faults:
                            bridge.send_fault(fault)

                        if result.rep_data is not None and assessment_reps_done < ASSESSMENT_TARGET_REPS:
                            bottom_kpts = None
                            bottom_angles = None
                            standing_kpts = None
                            if hasattr(pipeline, 'consume_bottom_frame'):
                                bottom_kpts, bottom_angles = pipeline.consume_bottom_frame()
                            if hasattr(pipeline, 'consume_standing_frame'):
                                standing_kpts = pipeline.consume_standing_frame()
                            trajectory_samples = None
                            if hasattr(pipeline, 'consume_rep_trajectory'):
                                trajectory_samples = pipeline.consume_rep_trajectory()

                            pending_reps.append(
                                (result.rep_data, bottom_kpts, bottom_angles, standing_kpts, trajectory_samples)
                            )
                            last_rep_time = time.time()
                            assessment_reps_done += 1

                            depth = result.rep_data.max_depth_angle
                            print(f"  [ASSESSMENT REP {assessment_reps_done}/{ASSESSMENT_TARGET_REPS}] depth={depth:.1f}°")

                            ipc_client.send_message(nan_to_none({
                                "type": "assessment_rep",
                                "rep_number": assessment_reps_done,
                                "total_required": ASSESSMENT_TARGET_REPS,
                                "round": assessment_round,
                                "depth_angle": round(depth, 1),
                            }))

                    measurement_overdue = (
                        assessment_reps_done >= ASSESSMENT_TARGET_REPS
                        and time.time() - last_rep_time > ASSESSMENT_MEASUREMENT_WAIT_S
                    )
                    if pending_reps and (athlete_params is not None or measurement_overdue):
                        if athlete_params is None:
                            print("  [ASSESSMENT] WARNING: body measurement unfinished — reps pass without kinematics")
                        for rep_data, bottom_kpts, bottom_angles, standing_kpts, trajectory_samples in pending_reps:
                            session_tracker.on_rep_complete(
                                rep_data,
                                bottom_kpts=bottom_kpts,
                                bottom_angles=bottom_angles,
                                standing_kpts=standing_kpts,
                                trajectory_samples=trajectory_samples,
                            )
                        pending_reps.clear()

                    # Display
                    frame = pipeline.last_frame
                    if frame is not None:
                        display = frame.copy()
                        if result.skeleton_2d is not None:
                            draw_skeleton(display, result.skeleton_2d)
                        fps_counter.update()
                        draw_fps(display, fps_counter.fps)

                        if pipeline.body_calibration.is_complete:
                            status_text = f"ASSESSMENT  ROUND {assessment_round}"
                        else:
                            measured, required = pipeline.body_calibration.progress
                            status_text = f"MEASURING BODY  {measured}/{required}"
                        _draw_hud_pill(display, status_text, accent=HUD_CYAN)
                        _draw_hud_pill(
                            display,
                            f"REP {assessment_reps_done}/{ASSESSMENT_TARGET_REPS}",
                            align="left", accent=HUD_CYAN, font_scale=1.1, y=48,
                        )
                        _draw_wordmark(display)

                        display_sink.show(display)

                    _throttle(pipeline, result, config.target_fps)

                # --- Run diagnosis on assessment reps ---
                kinematic_buffer = list(session_tracker._rep_kinematic_buffer)
                trajectory_buffer = list(session_tracker._rep_trajectory_buffer)

                if kinematic_buffer and athlete_params is not None:
                    anthro = build_anthro_dict(athlete_params)
                    rom = build_rom_dict(athlete_params, athlete_baseline)
                    set_features = SetFeatures(
                        user_id=0,
                        set_id=f"assessment_{assessment_round}",
                        rep_count=len(kinematic_buffer),
                        per_rep_kinematics=kinematic_buffer,
                        anthropometry=anthro,
                        rom=rom,
                    )
                    diagnosis_result = HypothesisEngine().diagnose(set_features)
                    score_summary = score_set(
                        kinematic_buffer, anthro, rom, trajectory_buffer,
                    )

                    has_immediate = len(diagnosis_result.immediate_causes) > 0
                    diagnosis_dict, scoring_dict = _serialize_diagnosis(diagnosis_result, score_summary)

                    print(f"\n  [ASSESSMENT] Diagnosis: confidence={diagnosis_result.confidence:.2f}")
                    print(f"  [ASSESSMENT] Immediate causes: {len(diagnosis_result.immediate_causes)}")
                    print(f"  [ASSESSMENT] Session causes: {len(diagnosis_result.session_causes)}")
                    print(f"  [ASSESSMENT] Form score: {round(score_summary.mean_score * 100)}/100")

                    # Build demo data before announcing the result so the
                    # choreography is ready the moment the agent reacts.
                    pending_demo = None
                    if has_immediate and not demo_played:
                        observed_kpts = session_tracker.bottom_frame_for_rep(
                            score_summary.worst_rep_number
                        )
                        if observed_kpts is not None:
                            pending_demo = build_demo_data(
                                observed_kpts, diagnosis_result, anthro=anthro, rom=rom,
                            )
                    if pending_demo is not None:
                        print(f"  [DEMO] Pose stack ready: {len(pending_demo.cues)} cue(s)")

                    ipc_client.send_message(nan_to_none({
                        "type": "assessment_result",
                        "round": assessment_round,
                        "passed": not has_immediate,
                        "diagnosis": diagnosis_dict,
                        "scoring": scoring_dict,
                        "demo": {
                            "available": pending_demo is not None,
                            "cues": [cue.model_dump() for cue in pending_demo.cues]
                            if pending_demo is not None else [],
                        },
                    }))

                    if not has_immediate:
                        assessment_passed = True
                        print(f"\n  [ASSESSMENT] PASSED after {assessment_round} round(s)")
                    else:
                        print(f"  [ASSESSMENT] Issues found — user needs to correct and retry")
                        if pending_demo is not None:
                            demo_played = True
                            if not _run_demo_phase(pending_demo):
                                pipeline.release()
                                ipc_client.disconnect()
                                return
                        pipeline.reset_readiness_gate()
                        # Wait for readiness gate again before next round
                        while not pipeline.is_ready:
                            result = pipeline.process_frame()
                            frame = pipeline.last_frame
                            if frame is not None:
                                display = frame.copy()
                                if result.skeleton_2d is not None:
                                    draw_skeleton(display, result.skeleton_2d)
                                fps_counter.update()
                                draw_fps(display, fps_counter.fps)

                                current, required = pipeline._readiness_gate.progress
                                _draw_hud_pill(display, f"WAITING {current}/{required}", accent=HUD_VIOLET)
                                _draw_wordmark(display)

                                display_sink.show(display)
                            _throttle(pipeline, result, config.target_fps)
                else:
                    # No kinematic data (athlete_params missing) — pass through
                    print("  [ASSESSMENT] No kinematic data — skipping diagnosis, passing assessment")
                    ipc_client.send_message({
                        "type": "assessment_result",
                        "round": assessment_round,
                        "passed": True,
                        "diagnosis": {},
                        "scoring": {},
                    })
                    assessment_passed = True

            # A factory (board-anchored) camera calibration moves its world frame
            # onto the lifter using the assessment reps just buffered.
            if camera_calibration is not None and camera_calibration.needs_world_anchor:
                camera_calibration.anchor_world_on_lifter()

        except KeyboardInterrupt:
            print("\nAssessment stopped by user")
            pipeline.release()
            ipc_client.disconnect()
            return

        # Reset pipeline state for calibration
        pipeline.reset_readiness_gate()
        pipeline.rep_counter.reset()
        pipeline.rep_counter.set_assessment_mode(False)
        if pipeline._bilstm is not None:
            pipeline._bilstm.reset()
            pipeline._bilstm.set_assessment_mode(False)
        session_tracker.set_assessment_mode(False)
        session_tracker.reset_rep_buffers()
        session_tracker.current_set_reps = []
        session_tracker.set_active = False
        session_tracker.current_set_number = 0

        # ============================================================
        #  PHASE 2: CALIBRATION (5-rep threshold calibration)
        # ============================================================
        print(f"\n{'='*60}")
        print(f"  CALIBRATION PHASE ({calibration_reps} bodyweight squats)")
        print(f"{'='*60}\n")

        tracker = CalibrationTracker(target_reps=calibration_reps)
        cal_set_collector = SetDataCollector()

        try:
            while not tracker.is_complete:
                result = pipeline.process_frame()

                if pipeline.is_ready and result.skeleton_3d is not None:
                    cal_set_collector.record_frame(result, result.skeleton_3d)

                    if result.joint_angles is not None:
                        tracker.record_frame(result.joint_angles, in_rep=pipeline.rep_counter.in_rep)

                    # Count reps but do NOT report faults during calibration
                    if result.rep_data is not None:
                        depth = result.rep_data.max_depth_angle
                        tracker.on_rep_complete(depth)
                        print(f"  [CAL REP {tracker.reps_completed}/{calibration_reps}] depth={depth:.1f}°")

                        # Notify voice agent of calibration rep
                        ipc_client.send_message(nan_to_none({
                            "type": "calibration_rep",
                            "rep_number": tracker.reps_completed,
                            "total_required": calibration_reps,
                            "depth_angle": round(depth, 1),
                        }))

                # Display
                frame = pipeline.last_frame
                if frame is not None:
                    display = frame.copy()
                    if result.skeleton_2d is not None:
                        draw_skeleton(display, result.skeleton_2d)
                    fps_counter.update()
                    draw_fps(display, fps_counter.fps)

                    if pipeline.is_ready:
                        _draw_hud_pill(display, "CALIBRATING", accent=HUD_CYAN)
                    else:
                        current, required = pipeline._readiness_gate.progress
                        _draw_hud_pill(display, f"WAITING {current}/{required}", accent=HUD_VIOLET)

                    _draw_hud_pill(
                        display,
                        f"REP {tracker.reps_completed}/{calibration_reps}",
                        align="left", accent=HUD_CYAN, font_scale=1.1, y=48,
                    )
                    _draw_wordmark(display)

                    display_sink.show(display)

                # Throttle
                _throttle(pipeline, result, config.target_fps)

            # --- Calibration complete ---
            peaks = tracker.get_peaks()
            cal_profile = build_calibration_profile(peaks, config)
            apply_calibration_to_rule_engine(pipeline._rule_engine, cal_profile)

            # Body measurements are session-scoped (per-set resets never touch
            # them); the params adopted during assessment back that up.
            cal_athlete_params = _extract_athlete_params(pipeline) or athlete_params
            # Build real baseline from calibration peaks
            cal_baseline = {
                "peakDorsi": peaks["peak_dorsiflexion"],
                "peakKneeFlex": peaks["avg_depth"],
            }
            athlete_baseline = cal_baseline

            # Send calibration results + athlete data to voice agent
            ipc_client.send_message(nan_to_none(_build_calibration_complete_message(
                movement_pattern, peaks, cal_profile, cal_athlete_params, cal_baseline,
            )))

            print(f"\n{'='*60}")
            print(f"  CALIBRATION COMPLETE ({tracker.reps_completed} reps)")
            print(f"  Peak trunk flexion: {peaks['trunk_flexion']:.1f}°")
            print(f"  Avg hip adduction:  {peaks['hip_adduction']:.1f}°")
            print(f"  Peak asymmetry:     {peaks['asymmetry']:.1f}°")
            print(f"  Peak dorsiflexion:  {peaks['peak_dorsiflexion']:.1f}°")
            print(f"  Avg squat depth:    {peaks.get('avg_depth', 0):.1f}°")
            print(f"{'='*60}\n")

            # Save calibration report
            _save_calibration_report(peaks, cal_profile, calibration_reps, out_dir)

            # Wire athlete params for diagnosis engine
            if cal_athlete_params is not None:
                athlete_params = cal_athlete_params
                session_tracker.set_athlete_params(cal_athlete_params, cal_baseline)
                bridge.set_athlete_params(cal_athlete_params, cal_baseline)
                print(
                    f"  [DIAGNOSIS] Athlete params set: "
                    f"shoulder={cal_athlete_params['shoulder_width_m']:.3f}m "
                    f"femur={cal_athlete_params['femur_avg_m']:.3f}m "
                    f"tibia={cal_athlete_params['tibia_avg_m']:.3f}m"
                )
            else:
                print("  [DIAGNOSIS] Body measurement not finished — diagnosis starts once it completes")

            # Reset pipeline for workout phase
            pipeline.reset_readiness_gate()
            gate_knee = tracker.standing_knee_flexion + 10.0
            pipeline._readiness_gate.max_knee_flexion_deg = gate_knee
            print(f"  [GATE] Standing knee threshold raised to {gate_knee:.1f}°")
            pipeline.rep_counter.reset()
            if pipeline._bilstm is not None:
                pipeline._bilstm.reset()

        except KeyboardInterrupt:
            print("\nCalibration stopped by user")
            pipeline.release()
            ipc_client.disconnect()
            return

    # Main processing loop
    # Rest timer state
    resting = False
    rest_end_time = 0.0
    rest_total_seconds = 0.0
    gate_prearmed = False
    workout_finished = False

    def _check_incoming_message():
        """Non-blocking check for messages from main.py."""
        return _poll_incoming_message(ipc_client)

    def _draw_rest_timer(frame, remaining_seconds, total_seconds=0.0):
        """Draw a centered rest countdown with a remaining-time progress arc."""
        h, w = frame.shape[:2]
        mins = int(remaining_seconds) // 60
        secs = int(remaining_seconds) % 60
        timer_text = f"{mins}:{secs:02d}" if mins > 0 else f"{secs}"

        # Semi-transparent dark navy overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), HUD_PANEL, -1)
        cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)

        cx, cy = w // 2, h // 2
        ring_radius = min(160, h // 3)

        # Track ring + remaining-time arc (clockwise from 12 o'clock)
        cv2.circle(frame, (cx, cy), ring_radius, (70, 55, 45), 2, cv2.LINE_AA)
        if total_seconds > 0:
            frac = max(0.0, min(1.0, remaining_seconds / total_seconds))
            cv2.ellipse(
                frame, (cx, cy), (ring_radius, ring_radius),
                -90, 0, 360 * frac, HUD_CYAN, 4, cv2.LINE_AA,
            )

        label = " ".join("REST")
        (label_w, _), _ = cv2.getTextSize(label, HUD_FONT, 0.9, 1)
        cv2.putText(
            frame, label, (cx - label_w // 2, cy - ring_radius // 3),
            HUD_FONT, 0.9, (205, 195, 175), 1, cv2.LINE_AA,
        )

        (timer_w, timer_h), _ = cv2.getTextSize(timer_text, HUD_FONT, 3.0, 4)
        cv2.putText(
            frame, timer_text, (cx - timer_w // 2, cy + timer_h // 2 + ring_radius // 6),
            HUD_FONT, 3.0, HUD_INK, 4, cv2.LINE_AA,
        )

    def _draw_readiness_indicator(frame, is_ready, progress):
        """Draw a readiness gate status pill in the top-right corner."""
        if is_ready:
            _draw_hud_pill(frame, "COLLECTING", accent=HUD_CYAN)
        else:
            current, required = progress
            _draw_hud_pill(frame, f"WAITING {current}/{required}", accent=HUD_VIOLET)

    def _format_set_summary(summary):
        """Format a set summary dict into a context_str-style log line."""
        parts = [
            f"Set {summary['set_number']} complete: {summary['total_reps']} reps.",
            f"{summary['clean_reps']}/{summary['total_reps']} clean reps.",
        ]
        if summary["avg_depth"]:
            parts.append(f"Average depth: {summary['avg_depth']}°.")
        if summary["depth_consistency"]:
            parts.append(f"Depth consistency: {summary['depth_consistency']}°.")
        if summary["fault_summary"]:
            fault_lines = []
            for fault_type, stats in summary["fault_summary"].items():
                fault_lines.append(
                    f"{fault_type}: {stats['count']}x (avg severity {stats['avg_severity']})"
                )
            parts.append(f"Faults: {'; '.join(fault_lines)}.")
        return " ".join(parts)

    # --- Data collection for post-session plots ---
    plot_timestamps = []
    plot_hip_l = []
    plot_hip_r = []
    plot_vel_l = []
    plot_vel_r = []
    plot_rep_events = []

    # --- Per-set rich data collection ---
    set_collector = SetDataCollector()
    set_collector.thresholds = extract_thresholds_from_rule_engine(pipeline._rule_engine)
    set_plot_data = []
    completed_sets = 0

    # --- On-demand demo bridge (active during "show me" flow) ---
    on_demand_bridge: DemoWSBridge | None = None
    on_demand_started_ack_sent = False
    on_demand_replay = False

    try:
        while True:
            # Check for incoming messages (rest_start, etc.)
            incoming = _check_incoming_message()
            if incoming:
                if incoming.get("type") == "rest_start":
                    rest_seconds = incoming.get("rest_seconds", 30)
                    rest_end_time = time.time() + rest_seconds
                    rest_total_seconds = float(rest_seconds)
                    resting = True
                    gate_prearmed = False
                    # Presence detection only during rest — the gate is
                    # re-armed GATE_PREARM_SECONDS before the timer expires
                    # so it can latch while the user gets into position.
                    pipeline.presence_only = True

                    # Finalize the just-completed set
                    if session_tracker.set_active:
                        session_tracker.force_end_set()
                        if session_tracker.last_set_summary:
                            print(f"[SET] {_format_set_summary(session_tracker.last_set_summary)}")
                    completed_sets += 1
                    if set_collector.has_enough_data():
                        print(f"\n  Generating set {completed_sets} plots...")
                        plot_result = finalize_set(set_collector, completed_sets, out_dir)
                        if plot_result is not None:
                            set_plot_data.append(plot_result)
                    else:
                        set_collector.reset()

                    if camera_calibration is not None:
                        camera_calibration.on_rest_start()

                    print(f"[REST] Starting {rest_seconds}s rest timer")
                elif incoming.get("type") == "assessment_mode":
                    session_tracker.set_assessment_mode(incoming.get("enabled", False))
                    print(f"[PIPELINE] Assessment mode {'enabled' if incoming.get('enabled') else 'disabled'}")
                elif incoming.get("type") == "request_last_rep":
                    snapshot = session_tracker.get_last_rep_snapshot()
                    request_id = incoming.get("request_id", "")
                    if snapshot:
                        bottom_kpts = snapshot.get("bottom_kpts")
                        rep_data = snapshot.get("rep_data", {})
                        if hasattr(rep_data, "model_dump"):
                            rep_data = rep_data.model_dump()

                        faults = rep_data.get("faults", [])
                        highlight = set()
                        for fault in faults:
                            ft = fault.get("fault_type", "") if isinstance(fault, dict) else ""
                            joints = FAULT_HIGHLIGHT_JOINTS.get(ft)
                            if joints:
                                highlight.update(joints)

                        # None when the rep could not be scored — the viewer
                        # hides the badge rather than showing a real-looking 0.
                        rep_score = snapshot.get("rep_score")

                        if bottom_kpts is not None:
                            kpts_arr = np.array(bottom_kpts, dtype=np.float64)
                            viewer_kpts = mediapipe_to_viewer_coords(kpts_arr)

                            on_demand_bridge = _restart_demo_bridge(on_demand_bridge)
                            on_demand_replay = True
                            on_demand_started_ack_sent = True
                            on_demand_bridge.send_replay_init(
                                viewer_kpts,
                                highlight_joints=sorted(highlight),
                                rep_number=rep_data.get("rep_number", 0),
                                rep_score=rep_score,
                                faults=faults,
                            )

                        resp: dict = {"type": "last_rep_snapshot", "request_id": request_id}
                        for key in ("bottom_kpts", "standing_kpts"):
                            kpts = snapshot.get(key)
                            if kpts is not None:
                                resp[key] = kpts.tolist() if isinstance(kpts, np.ndarray) else kpts
                        if snapshot.get("kinematic_summary"):
                            resp["kinematic_summary"] = snapshot["kinematic_summary"]
                        resp["rep_data"] = rep_data
                        ipc_client.send_message(nan_to_none(resp))
                    else:
                        ipc_client.send_message({
                            "type": "last_rep_snapshot",
                            "request_id": request_id,
                            "error": "no_data",
                        })
                elif incoming.get("type") == "request_demo":
                    request_id = incoming.get("request_id", "")
                    demo_data = session_tracker.build_on_demand_demo()
                    if demo_data is not None:
                        on_demand_bridge = _restart_demo_bridge(on_demand_bridge)
                        on_demand_replay = False
                        on_demand_started_ack_sent = False
                        on_demand_bridge.send_init(demo_data)
                        ipc_client.send_message({
                            "type": "demo_data_ready",
                            "request_id": request_id,
                            "status": "available",
                            "cues": [cue.model_dump() for cue in demo_data.cues],
                        })
                        print(f"  [ON-DEMAND DEMO] Started with {len(demo_data.cues)} cue(s)")
                    else:
                        ipc_client.send_message({
                            "type": "demo_data_ready",
                            "request_id": request_id,
                            "status": "unavailable",
                            "error": "no_diagnosis_or_rep_data",
                        })
                elif incoming.get("type") == "demo_start" and on_demand_bridge is not None:
                    on_demand_bridge.send_event({"type": "demo_start"})
                elif incoming.get("type") == "demo_cue" and on_demand_bridge is not None:
                    on_demand_bridge.send_event({
                        "type": "demo_cue",
                        "cue_index": int(incoming.get("cue_index", 0)),
                    })
                elif incoming.get("type") == "demo_end" and on_demand_bridge is not None:
                    on_demand_bridge.send_event({"type": "demo_end"})
                elif incoming.get("type") == "workout_complete":
                    if on_demand_bridge is not None:
                        on_demand_bridge.stop()
                        on_demand_bridge = None
                        on_demand_replay = False
                        display_sink.send_event({"type": "demo", "action": "end"})
                    workout_finished = True
                    pipeline.presence_only = True
                    print("[PIPELINE] Workout complete — stopping rep counting")

                    # Finalize the last set (no rest_start is sent for it)
                    if session_tracker.set_active:
                        session_tracker.force_end_set()
                        if session_tracker.last_set_summary:
                            print(f"[SET] {_format_set_summary(session_tracker.last_set_summary)}")
                    completed_sets += 1
                    if set_collector.has_enough_data():
                        print(f"\n  Generating set {completed_sets} plots...")
                        plot_result = finalize_set(set_collector, completed_sets, out_dir)
                        if plot_result is not None:
                            set_plot_data.append(plot_result)
                    else:
                        set_collector.reset()

                    # Generate session dashboard now while we can
                    if set_plot_data:
                        generate_session_dashboard(
                            set_plot_data, {"exercise": exercise_name},
                            out_dir, auto_open=False,
                        )

            # A camera refine that finished installs between sets only: while
            # resting before the gate re-arms, or once the workout is over.
            if camera_calibration is not None:
                camera_calibration.poll(
                    can_install=(resting and not gate_prearmed) or workout_finished,
                )

            result = pipeline.process_frame()

            # Returning users without stored params get measured during sets
            if athlete_params is None:
                athlete_params = _adopt_measured_athlete_params(
                    pipeline, session_tracker, bridge, athlete_baseline,
                )

            if on_demand_bridge is not None:
                # Replay shows a frozen past rep — streaming the live pose
                # would repaint over it every frame.
                live_pose = result.skeleton_3d_display or result.skeleton_3d_raw
                if not on_demand_replay and live_pose is not None:
                    on_demand_bridge.send_live_pose(live_pose.to_numpy())
                if not on_demand_started_ack_sent and on_demand_bridge.wait_started(timeout=0):
                    ipc_client.send_message({"type": "demo_started"})
                    on_demand_started_ack_sent = True
                if on_demand_bridge.wait_done(timeout=0):
                    on_demand_bridge.stop()
                    on_demand_bridge = None
                    on_demand_replay = False
                    display_sink.send_event({"type": "demo", "action": "end"})
                    print("  [ON-DEMAND DEMO] Viewer closed")

            # Collect data for post-session plots (not during rest — the
            # gate pre-arm window produces angles in the last few seconds)
            if result.joint_angles is not None and not resting:
                t = result.timestamp
                plot_timestamps.append(t)
                plot_hip_l.append(result.joint_angles.hip_flexion_l)
                plot_hip_r.append(result.joint_angles.hip_flexion_r)
                if len(plot_timestamps) >= 2:
                    dt = plot_timestamps[-1] - plot_timestamps[-2]
                    if dt > 0:
                        plot_vel_l.append((plot_hip_l[-1] - plot_hip_l[-2]) / dt)
                        plot_vel_r.append((plot_hip_r[-1] - plot_hip_r[-2]) / dt)
                    else:
                        plot_vel_l.append(0.0)
                        plot_vel_r.append(0.0)
                else:
                    plot_vel_l.append(0.0)
                    plot_vel_r.append(0.0)
                if result.rep_data is not None:
                    plot_rep_events.append((t, result.rep_data.rep_number))

            # Collect per-set rich data
            if pipeline.is_ready and result.skeleton_3d is not None and not resting and not workout_finished:
                set_collector.record_frame(result, result.skeleton_3d)

            if not resting and not workout_finished:
                # Normal mode: send biomechanics data
                bridge.send_frame_data(result, rep_phase=pipeline.rep_counter.phase)

                for fault in result.faults:
                    # Shallow-rep depth faults are delivered by the
                    # shallow_rep message below, which is not rate-limited.
                    if not fault.details.get("shallow_rep"):
                        bridge.send_fault(fault)
                    set_collector.record_fault(fault)

                if result.shallow_rep_class is not None:
                    shallow_fault = next(
                        (f for f in result.faults if f.details.get("shallow_rep")),
                        None,
                    )
                    bridge.send_shallow_rep(
                        result.shallow_rep_class,
                        fault=shallow_fault,
                        set_number=session_tracker.current_set_number,
                    )
                    print(
                        f"[SHALLOW REP] depth_class={result.shallow_rep_class} — not counted"
                    )

                if result.rep_data:
                    bottom_kpts, bottom_angles = pipeline.consume_bottom_frame()
                    standing_kpts = pipeline.consume_standing_frame()
                    session_tracker.on_rep_complete(
                        result.rep_data,
                        bottom_kpts=bottom_kpts,
                        bottom_angles=bottom_angles,
                        standing_kpts=standing_kpts,
                        trajectory_samples=pipeline.consume_rep_trajectory(),
                    )

                if session_tracker.check_set_timeout(time.time()):
                    pipeline.reset_readiness_gate()
                    if session_tracker.last_set_summary:
                        print(f"[SET] {_format_set_summary(session_tracker.last_set_summary)}")

                    # Finalize the timed-out set
                    completed_sets += 1
                    if set_collector.has_enough_data():
                        print(f"\n  Generating set {completed_sets} plots...")
                        plot_result = finalize_set(set_collector, completed_sets, out_dir)
                        if plot_result is not None:
                            set_plot_data.append(plot_result)
                    else:
                        set_collector.reset()

            # Display video feed
            frame = pipeline.last_frame
            if frame is not None:
                display = frame.copy()
                if result.skeleton_2d is not None and not resting:
                    draw_skeleton(display, result.skeleton_2d)
                fps_counter.update()
                draw_fps(display, fps_counter.fps)

                # Readiness gate indicator (hidden while resting/finished —
                # the rest timer is the status during rest)
                if not resting and not workout_finished:
                    _draw_readiness_indicator(
                        display, pipeline.is_ready, pipeline._readiness_gate.progress,
                    )

                if resting:
                    remaining = max(0.0, rest_end_time - time.time())
                    if remaining <= 0:
                        resting = False
                        if not gate_prearmed:
                            # Fallback: the loop never reached the pre-arm
                            # window — arm the gate now (old behavior).
                            pipeline.presence_only = False
                            pipeline.reset_readiness_gate()
                        # Rep counters reset at rest end so anything done
                        # while settling in doesn't count into the new set.
                        pipeline.rep_counter.reset()
                        if pipeline._bilstm is not None:
                            pipeline._bilstm.reset()
                        ipc_client.send_message({"type": "rest_complete"})
                        print("[REST] Timer expired — sent rest_complete")
                    else:
                        if not gate_prearmed and remaining <= GATE_PREARM_SECONDS:
                            # Re-arm early so the gate can latch while the
                            # user gets set; collection stays off until the
                            # timer expires (resting still True).
                            gate_prearmed = True
                            pipeline.presence_only = False
                            pipeline.reset_readiness_gate()
                            print(f"[REST] Gate pre-armed {remaining:.1f}s before rest end")
                        _draw_rest_timer(display, remaining, rest_total_seconds)
                else:
                    # Show rep count overlay — the logged count, so the
                    # screen never claims reps the workout didn't record
                    rep_count = pipeline.rep_count
                    _draw_hud_pill(
                        display, f"REPS {rep_count}",
                        align="left", accent=HUD_CYAN, font_scale=1.1, y=48,
                    )
                _draw_wordmark(display)

                if valgus_recorder is not None:
                    valgus_recorder.record_frame(result, display, pipeline, resting)
                if pipeline_inspector is not None:
                    pipeline_inspector.record_frame(result, display, pipeline, resting)

                display_sink.show(display)

            # Throttle to target FPS
            _throttle(pipeline, result, config.target_fps)

    except KeyboardInterrupt:
        print("\nPipeline stopped by user")
    except SystemExit:
        print("\nPipeline terminated by parent process")
    except Exception as e:
        print(f"\nPipeline error: {e}")
        ipc_client.send_message({"type": "error", "value": str(e)})
    finally:
        # Block further shutdown signals so the data saving below cannot be
        # interrupted by main.py's terminate() arriving mid-save.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        display_sink.close()
        pipeline.release()
        try:
            bridge.send_pipeline_status("stopped", {})
            ipc_client.disconnect()
        except Exception:
            pass
        print("Biomechanics pipeline stopped")

        # Finalize any in-progress set
        if set_collector.has_enough_data() and not resting:
            if session_tracker.set_active:
                session_tracker.force_end_set()
                if session_tracker.last_set_summary:
                    print(f"[SET] {_format_set_summary(session_tracker.last_set_summary)}")
            completed_sets += 1
            print(f"\n  Generating set {completed_sets} plots...")
            plot_result = finalize_set(set_collector, completed_sets, out_dir)
            if plot_result is not None:
                set_plot_data.append(plot_result)
        else:
            # End any active set and print its summary
            if session_tracker.set_active:
                session_tracker.force_end_set()
                if session_tracker.last_set_summary:
                    print(f"[SET] {_format_set_summary(session_tracker.last_set_summary)}")

        # Generate session dashboard
        if set_plot_data:
            generate_session_dashboard(
                set_plot_data, {"exercise": exercise_name},
                out_dir, auto_open=False,
            )

        # Generate post-session plots
        if len(plot_timestamps) > 10:
            print(f"\nGenerating plots from {len(plot_timestamps)} frames...")
            plot_data = {
                "timestamps": np.array(plot_timestamps),
                "hip_flexion_l": np.array(plot_hip_l),
                "hip_flexion_r": np.array(plot_hip_r),
                "hip_velocity_l": np.array(plot_vel_l),
                "hip_velocity_r": np.array(plot_vel_r),
                "rep_events": plot_rep_events,
            }
            print(f"Plots will be saved to: {out_dir}\n")
            plot_hip_position(plot_data, out_dir, show=False)
            plot_hip_velocity(plot_data, out_dir, show=False)
            print(f"\nAll plots saved in: {out_dir}")
        else:
            print("Not enough data for plots (need >10 frames with joint angles).")

        if valgus_recorder is not None:
            try:
                valgus_recorder.finalize(exercise_name)
            except Exception as e:
                print(f"[VALGUS] Report generation failed: {e}")

        if pipeline_inspector is not None:
            try:
                report_path = pipeline_inspector.finalize(exercise_name)
                if report_path is not None:
                    webbrowser.open(f"file://{report_path}")
            except Exception as e:
                print(f"[INSPECT] Report generation failed: {e}")

        # Remove the workout dir if this run never produced any output
        # (e.g. workout started then abandoned before any reps)
        try:
            os.rmdir(out_dir)
            print(f"Removed empty workout output dir: {out_dir}")
        except OSError:
            pass


if __name__ == "__main__":
    cam0 = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cam1 = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    exercise = sys.argv[3] if len(sys.argv) > 3 else "Barbell Back Squat"

    # Parse optional args
    cal_file = None
    cal_mode = False
    cal_reps = 5
    preload_flag = False
    height_cm = None
    i = 4
    while i < len(sys.argv):
        if sys.argv[i] == "--calibration-file" and i + 1 < len(sys.argv):
            cal_file = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--calibration-mode":
            cal_mode = True
            i += 1
        elif sys.argv[i] == "--calibration-reps" and i + 1 < len(sys.argv):
            cal_reps = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--preload":
            preload_flag = True
            i += 1
        elif sys.argv[i] == "--user-height-cm" and i + 1 < len(sys.argv):
            height_cm = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--valgus":
            os.environ["NOWVA_VALGUS_DEBUG"] = "1"
            i += 1
        elif sys.argv[i] == "--inspect":
            os.environ["NOWVA_PIPELINE_INSPECT"] = "1"
            i += 1
        else:
            i += 1

    run_biomechanics_pipeline(
        cam0_id=cam0,
        cam1_id=cam1,
        exercise_name=exercise,
        calibration_file=cal_file,
        calibration_mode=cal_mode,
        calibration_reps=cal_reps,
        preload=preload_flag,
        user_height_cm=height_cm,
    )
