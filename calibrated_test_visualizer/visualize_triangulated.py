#!/usr/bin/env python3
"""
Multi-camera triangulated squat visualizer.

Runs the production BiomechanicsPipeline in multi-camera mode (synchronized
capture, per-view RTMPose, DLT triangulation, the shared pre-IK chain, IK,
rep counting), records the primary camera, and generates the same interactive
HTML dashboard as the single-camera visualizer.

Designed for headless operation (no cv2.imshow) — all feedback via terminal.

Usage:
    python calibrated_test_visualizer/visualize_triangulated.py --height 188.5
    python calibrated_test_visualizer/visualize_triangulated.py --calibration outputs/calibration.json
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src/ and the single-camera visualizer (shared HTML builder) to the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "demos"))

import cv2

from biomechanics.config import BiomechanicsConfig, load_pipeline_config
from biomechanics.pipeline import BiomechanicsPipeline
from biomechanics.pose.multi_camera import MultiCameraPoseProvider
from biomechanics.triangulation import MultiCameraCapture

from visualize_video_squats import build_html, extract_frame_data, process_captured_reps

TARGET_REPS = 5
MIN_CAMERAS = 2
MAX_CAMERAS = 3
MAX_PROBE_DEVICE_ID = 10
DEFAULT_HEIGHT_CM = 188.5
VIDEO_FPS = 30.0
COUNTDOWN_SECONDS = 3
CAPTURE_WARMUP_S = 0.5
PROGRESS_EVERY_FRAMES = 30
MIN_REPS_FOR_REPLAY = 2


# ---------------------------------------------------------------------------
# Phase 0: Camera discovery
# ---------------------------------------------------------------------------

def discover_cameras(requested_ids: list[int] | None) -> list[int]:
    """Detect available cameras or validate requested IDs."""
    if requested_ids:
        print(f"Using requested camera IDs: {requested_ids}")
        return requested_ids

    print("Detecting cameras...")
    found = MultiCameraCapture.detect_cameras(max_id=MAX_PROBE_DEVICE_ID)
    print(f"  Found {len(found)} camera(s): {found}")

    if len(found) < MIN_CAMERAS:
        print(f"ERROR: Need at least {MIN_CAMERAS} cameras for triangulation.")
        sys.exit(1)

    if len(found) > MAX_CAMERAS:
        found = found[:MAX_CAMERAS]
        print(f"  Using first {MAX_CAMERAS}: {found}")

    return found


# ---------------------------------------------------------------------------
# Phase 1: T-Pose calibration
# ---------------------------------------------------------------------------

def run_tpose_calibration(
    config: BiomechanicsConfig, camera_ids: list[int], height_cm: float, calibration_path: Path,
) -> None:
    """Capture T-pose frames through the production provider and save the rig calibration."""
    tri = config.triangulation
    n_frames = tri.tpose_capture_frames

    print(f"\n{'='*50}")
    print("  T-POSE CALIBRATION")
    print(f"{'='*50}")
    print(f"  Height: {height_cm:.1f} cm ({height_cm / 100:.3f} m)")
    print(f"  Cameras: {camera_ids}")
    print(f"  Frames to capture: {n_frames}")
    print(f"{'='*50}")

    provider = MultiCameraPoseProvider(
        device_ids=camera_ids,
        confidence_threshold=config.pose.confidence_threshold,
        model_path=config.pose.model_path,
        min_views=tri.min_views,
        max_reprojection_error=tri.max_reprojection_error,
        max_sync_delta_ms=tri.max_sync_delta_ms,
        resolution=config.capture.resolution,
        primary_camera=camera_ids[0],
        focal_length_factor=tri.focal_length_factor,
    )
    provider.initialize()

    print("\nStand in T-pose (arms extended horizontally, feet shoulder-width).")
    print("Stay still...")
    for i in range(COUNTDOWN_SECONDS, 0, -1):
        print(f"  Capturing in {i}...")
        time.sleep(1)
    print("  Capturing...")

    provider.start()
    time.sleep(CAPTURE_WARMUP_S)
    try:
        result = provider.calibrate(
            height_m=height_cm / 100.0, n_frames=n_frames, save_path=str(calibration_path),
        )
    finally:
        provider.release()

    print("\nCalibration results:")
    for cam_id, cam_cal in result.cameras.items():
        print(f"  Camera {cam_id}: reprojection error = {cam_cal.reprojection_error:.1f} px")
    print(f"Calibration saved: {calibration_path}")


# ---------------------------------------------------------------------------
# Phase 2 & 3: Stabilization + Recording (production pipeline)
# ---------------------------------------------------------------------------

def run_triangulated_capture(
    config: BiomechanicsConfig, output_dir: Path, target_reps: int,
) -> tuple[list, list, list, float, dict | None, Path]:
    """Run the multi-camera pipeline until target_reps reps; record the primary camera."""
    print(f"\n{'='*50}")
    print("  TRIANGULATED CAPTURE")
    print(f"{'='*50}")

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = output_dir / f"squat_tri_{timestamp_str}.mp4"

    pipeline = BiomechanicsPipeline(config, exercise_name="squat")
    pipeline.preload_pose_model()
    pipeline.start_capture()
    time.sleep(CAPTURE_WARMUP_S)

    # --- Stabilization: the pipeline's readiness gate is the only criterion ---
    print("\n=== STABILIZATION ===")
    print("Stand still with full body visible...")

    frame_count = 0
    while not pipeline.is_ready:
        pipeline.process_frame()
        frame_count += 1
        if frame_count % PROGRESS_EVERY_FRAMES == 0:
            passes, required = pipeline._readiness_gate.progress
            failure = pipeline._readiness_gate.last_failure or "passing"
            print(f"  Readiness gate: {passes}/{required} ({failure})")

    print("Stabilization complete.\n")

    # --- Recording phase ---
    print("=== RECORDING ===")
    print(f"Perform {target_reps} squats.\n")

    video_writer: cv2.VideoWriter | None = None
    frames_data: list = []
    reps: list = []
    rep_boundaries: list = []
    current_rep_start = None
    prev_in_rep = False
    rec_frame_idx = 0
    last_written_frame = None
    rec_start_time = time.time()

    while len(reps) < target_reps:
        result = pipeline.process_frame()
        frame = pipeline.last_frame
        if frame is None or frame is last_written_frame:
            continue

        if video_writer is None:
            frame_height, frame_width = frame.shape[:2]
            video_writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, (frame_width, frame_height),
            )
        video_writer.write(frame)
        last_written_frame = frame

        angles = result.joint_angles
        if result.skeleton_3d is None or angles is None:
            frames_data.append(None)
            rec_frame_idx += 1
            continue

        in_rep = pipeline.rep_counter.in_rep
        if in_rep and not prev_in_rep:
            current_rep_start = rec_frame_idx
        if not in_rep and prev_in_rep and current_rep_start is not None:
            rep_boundaries.append((current_rep_start, rec_frame_idx))
            current_rep_start = None
        prev_in_rep = in_rep

        if result.rep_data is not None:
            reps.append(result.rep_data)
            elapsed = time.time() - rec_start_time
            print(f"  Rep {result.rep_data.rep_number}: depth={result.rep_data.max_depth_angle:.1f}°  ({elapsed:.1f}s)")

        frames_data.append(extract_frame_data(result.skeleton_3d, angles, rec_frame_idx))
        rec_frame_idx += 1

    print(f"\n{target_reps} reps captured!")

    if video_writer is not None:
        video_writer.release()
    measured = pipeline.body_calibration.to_athlete_params()
    pipeline.release()

    print("\nRecording stats:")
    print(f"  Total frames: {rec_frame_idx}")
    print(f"  Body measured: {'yes' if measured is not None else 'no (sandbox sliders will not pre-fill)'}")
    print(f"  Video saved: {video_path}")

    return frames_data, reps, rep_boundaries, VIDEO_FPS, measured, video_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-camera triangulated squat capture")
    parser.add_argument("--height", type=float, default=DEFAULT_HEIGHT_CM,
                        help=f"Athlete height in cm (default: {DEFAULT_HEIGHT_CM})")
    parser.add_argument("--camera-ids", type=str, default=None,
                        help="Comma-separated camera device IDs (default: auto-detect)")
    parser.add_argument("--calibration", type=str, default=None,
                        help="Path to existing calibration JSON (skip T-pose)")
    parser.add_argument("--reps", type=int, default=TARGET_REPS,
                        help=f"Number of reps to capture (default: {TARGET_REPS})")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).parent / "outputs"),
                        help="Output directory")
    parser.add_argument("--no-open", action="store_true",
                        help="Don't auto-open HTML in browser")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  MULTI-CAMERA TRIANGULATED SQUAT CAPTURE")
    print("=" * 60)

    # The pipeline reads this at construction to pick the multi-camera provider.
    os.environ["NOWVA_MULTI_CAMERA"] = "true"
    config = load_pipeline_config()

    # Phase 0: Camera discovery
    requested_ids = None
    if args.camera_ids:
        requested_ids = [int(x.strip()) for x in args.camera_ids.split(",")]
    camera_ids = discover_cameras(requested_ids)

    tri = config.triangulation
    tri.device_ids = camera_ids
    tri.primary_camera = camera_ids[0]

    # Phase 1: T-Pose calibration (or load existing)
    if args.calibration:
        calibration_path = Path(args.calibration)
        print(f"\nUsing existing calibration: {calibration_path}")
    else:
        calibration_path = output_dir / f"calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        run_tpose_calibration(config, camera_ids, args.height, calibration_path)
    tri.calibration_file = str(calibration_path)

    # Phases 2-3: Stabilization + Recording
    frames_data, reps, rep_boundaries, fps, measured, video_path = run_triangulated_capture(
        config, output_dir, target_reps=args.reps,
    )

    if len(reps) < MIN_REPS_FOR_REPLAY:
        print(f"ERROR: Need at least {MIN_REPS_FOR_REPLAY} reps, got {len(reps)}.")
        sys.exit(1)

    # Phase 4: Generate output
    print(f"\n{'='*50}")
    print("  GENERATING OUTPUT")
    print(f"{'='*50}")
    print(f"Processing {len(reps)} reps...")
    print(f"  Using rep 1 as baseline, replaying reps 2-{len(reps)}")

    baseline, replay_reps, athlete_params = process_captured_reps(frames_data, rep_boundaries, measured)
    print(f"  Baseline trunk offset: {baseline['peakTrunkOffset']}°")
    print(f"  Lean thresholds: {baseline['leanThresholds']}")
    print(f"  Valgus thresholds: {baseline['valgusThresholds']}")

    if athlete_params:
        print("  Athlete params:")
        print(f"    Stance width: {athlete_params['stanceWidth']}x  Toe-out: {athlete_params['toeOut']}°")
        print(f"    Dorsi ratio: {athlete_params['dorsiRatio']}  Body scale: {athlete_params['bodyScale']}")

    html = build_html(baseline, replay_reps, fps, athlete_params)
    html_path = video_path.with_suffix(".html")
    html_path.write_text(html)

    print(f"\nVideo saved: {video_path}")
    print(f"HTML saved:  {html_path}")

    if not args.no_open:
        import webbrowser
        webbrowser.open(f"file://{html_path.resolve()}")

    print("\nDone!")


if __name__ == "__main__":
    main()
