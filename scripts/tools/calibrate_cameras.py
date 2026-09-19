#!/usr/bin/env python3
"""
Camera calibration with a ChArUco board: per-camera intrinsics (hand-held board) and the
factory rig extrinsics (one board seen by all cameras). Thin CLI over
biomechanics.triangulation.charuco; files land in ~/.nowva where the pipeline finds them.

    python scripts/tools/calibrate_cameras.py board --out board.png
    python scripts/tools/calibrate_cameras.py intrinsics --camera 0
    python scripts/tools/calibrate_cameras.py extrinsics --cameras 0,1,2
    python scripts/tools/calibrate_cameras.py check --cameras 0,1,2

Print board.png at 100 % scale, MEASURE a printed square and marker, and pass the measured
--square-mm / --marker-mm to every command: they set the metric scale of the rig.
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from biomechanics.triangulation.calibration import (  # noqa: E402
    NOWVA_CALIBRATION_DIR,
    TPoseCalibrator,
    intrinsics_path,
    load_rig_intrinsics,
    rig_calibration_path,
    save_intrinsics,
)
from biomechanics.triangulation.charuco import (  # noqa: E402
    CharucoBoardDetector,
    CharucoBoardSpec,
    CharucoDetection,
    average_detections,
    calibrate_intrinsics,
    coverage_fraction,
    render_printable_board,
    solve_rig_extrinsics,
    view_novelty_px,
)

PAPER_SIZES_MM = {"A4": (210.0, 297.0), "A3": (297.0, 420.0), "A2": (420.0, 594.0), "A1": (594.0, 841.0), "A0": (841.0, 1189.0)}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
REFERENCE_WIDTH_PX = 1280.0  # pixel thresholds below are for this width and scale with it

# Hand-held intrinsics views: the board must be briefly still (motion blur, rolling shutter)
# and clearly different from every view already taken.
HANDHELD_STEADY_FRAMES = 3
HANDHELD_MAX_MOTION_PX = 1.0
MIN_VIEW_NOVELTY_PX = 40.0
MIN_GOOD_COVERAGE = 0.7
MIN_GOOD_TILT_DEG = 20.0
# Rig placements: the cameras are not hardware-synced, and a board that moves 5 mm between
# their grabs costs ~5 mm of camera position, so it must sit still on a stand for the whole
# window, whose frames are then averaged.
RIG_STEADY_FRAMES = 10
RIG_MAX_MOTION_PX = 0.2
MIN_PLACEMENT_NOVELTY_PX = 60.0
PREVIEW_WIDTH_PX = 640
CHECK_FRAMES = 30
CHECK_GOOD_RMS_PX = 3.0
WINDOW_NAME = "calibrate_cameras"
KEY_QUIT = (ord("q"), 27)
KEY_FORCE = ord(" ")


def _board_spec(args: argparse.Namespace) -> CharucoBoardSpec:
    squares_x, squares_y = (int(value) for value in args.squares.lower().split("x"))
    return CharucoBoardSpec(
        squares_x=squares_x,
        squares_y=squares_y,
        square_length_m=args.square_mm / 1000.0,
        marker_length_m=args.marker_mm / 1000.0,
        dictionary_name=args.dict,
    )


def _parse_resolution(text: str) -> tuple[int, int]:
    width, height = (int(value) for value in text.lower().split("x"))
    return width, height


def _open_camera(device_id: int, resolution: tuple[int, int]) -> cv2.VideoCapture:
    # Same open sequence as MultiCameraCapture, so intrinsics match the pipeline's sensor mode.
    capture = cv2.VideoCapture(device_id)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not capture.isOpened():
        raise SystemExit(f"Could not open camera device {device_id}")
    return capture


def _require_resolution(frame: np.ndarray, resolution: tuple[int, int], device_id: int) -> None:
    actual = (frame.shape[1], frame.shape[0])
    if actual != resolution:
        raise SystemExit(
            f"Camera {device_id} delivers {actual[0]}x{actual[1]}, not the requested "
            f"{resolution[0]}x{resolution[1]}; rerun with --resolution {actual[0]}x{actual[1]} "
            f"and run the pipeline at that resolution too"
        )


def _is_steady(window: deque, max_motion_px: float) -> bool:
    if len(window) < window.maxlen:
        return False
    frames = list(window)
    return all(view_novelty_px(b, [a]) <= max_motion_px for a, b in zip(frames, frames[1:]))


def _draw_preview(frame: np.ndarray, detection: CharucoDetection | None, lines: list[str]) -> np.ndarray:
    preview = frame.copy()
    if detection is not None:
        for x, y in detection.corners_px:
            cv2.circle(preview, (int(round(x)), int(round(y))), 5, (0, 255, 0), 2)
    scale = PREVIEW_WIDTH_PX / preview.shape[1]
    preview = cv2.resize(preview, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    for row, line in enumerate(lines):
        cv2.putText(preview, line, (10, 24 + 22 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(preview, line, (10, 24 + 22 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
    return preview


def _backup_existing(path: Path) -> None:
    if path.exists():
        backup = path.with_suffix(path.suffix + ".bak")
        path.replace(backup)
        print(f"Previous {path.name} kept as {backup.name}")


def _image_files(directory: Path) -> list[Path]:
    return sorted(entry for entry in directory.iterdir() if entry.suffix.lower() in IMAGE_SUFFIXES)


def run_board(args: argparse.Namespace) -> None:
    spec = _board_spec(args)
    page = render_printable_board(spec, PAPER_SIZES_MM[args.paper], args.dpi)
    Image.fromarray(page).save(args.out, dpi=(args.dpi, args.dpi))
    width_mm, height_mm = spec.size_m[0] * 1000.0, spec.size_m[1] * 1000.0
    print(f"Wrote {args.out}: {args.paper} at {args.dpi} dpi, board {width_mm:.0f} x {height_mm:.0f} mm")
    print("Print at 100 % (no 'fit to page'), glue it to something rigid and flat, then measure a")
    print("square and a marker with a ruler and pass those values as --square-mm / --marker-mm.")


def _intrinsics_views_from_images(directory: Path, detector: CharucoBoardDetector) -> tuple[list[CharucoDetection], tuple[int, int]]:
    detections, resolution = [], None
    for image_path in _image_files(directory):
        image = cv2.imread(str(image_path))
        resolution = (image.shape[1], image.shape[0])
        detection = detector.detect(image)
        if detection is not None:
            detections.append(detection)
    if resolution is None:
        raise SystemExit(f"No images found in {directory}")
    return detections, resolution


def _intrinsics_views_from_camera(args: argparse.Namespace, detector: CharucoBoardDetector) -> tuple[list[CharucoDetection], tuple[int, int]]:
    resolution = _parse_resolution(args.resolution)
    pixel_scale = resolution[0] / REFERENCE_WIDTH_PX
    capture = _open_camera(args.camera, resolution)
    window: deque = deque(maxlen=HANDHELD_STEADY_FRAMES)
    accepted: list[CharucoDetection] = []
    print(f"Show the board to camera {args.camera}: fill the frame, reach every corner of the image,")
    print("tilt it 20-45 degrees in different directions, vary the distance. Hold still to capture. q = done.")
    while len(accepted) < args.views:
        ok, frame = capture.read()
        if not ok:
            continue
        _require_resolution(frame, resolution, args.camera)
        detection = detector.detect(frame)
        if detection is None:
            window.clear()
        else:
            window.append(detection)
            steady = _is_steady(window, HANDHELD_MAX_MOTION_PX * pixel_scale)
            if steady and view_novelty_px(detection, accepted) >= MIN_VIEW_NOVELTY_PX * pixel_scale:
                accepted.append(detection)
                window.clear()
                print(f"  view {len(accepted)}/{args.views}  coverage {coverage_fraction(accepted, resolution):.0%}")
        status = [f"views {len(accepted)}/{args.views}", f"coverage {coverage_fraction(accepted, resolution):.0%}"]
        cv2.imshow(WINDOW_NAME, _draw_preview(frame, detection, status))
        if cv2.waitKey(1) & 0xFF in KEY_QUIT:
            break
    capture.release()
    cv2.destroyAllWindows()
    return accepted, resolution


def run_intrinsics(args: argparse.Namespace) -> None:
    spec = _board_spec(args)
    detector = CharucoBoardDetector(spec)
    if args.from_images:
        detections, resolution = _intrinsics_views_from_images(Path(args.from_images), detector)
    else:
        detections, resolution = _intrinsics_views_from_camera(args, detector)
    result = calibrate_intrinsics(detections, spec, resolution)

    camera_key = args.key or str(args.camera)
    coverage = coverage_fraction(detections, resolution)
    K = result.intrinsic_matrix
    print(f"\nCamera '{camera_key}' at {resolution[0]}x{resolution[1]}: {result.views_used}/{len(detections)} views used")
    print(f"  RMS reprojection  {result.rms_reprojection_px:.3f} px   (good: < 0.5)")
    print(f"  image coverage    {coverage:.0%}   max board tilt {result.view_tilts_deg.max():.0f} deg")
    print(f"  fx {K[0, 0]:.1f}  fy {K[1, 1]:.1f}  cx {K[0, 2]:.1f}  cy {K[1, 2]:.1f}   (f / width = {K[0, 0] / resolution[0]:.3f})")
    print(f"  distortion k1 k2 p1 p2 k3 = {np.round(result.distortion_coeffs, 4).tolist()}")
    if coverage < MIN_GOOD_COVERAGE:
        print(f"  WARNING: coverage below {MIN_GOOD_COVERAGE:.0%}; distortion is extrapolated where the board never went. Redo and reach the image corners.")
    if result.view_tilts_deg.max() < MIN_GOOD_TILT_DEG:
        print(f"  WARNING: board never tilted past {MIN_GOOD_TILT_DEG:.0f} deg; focal length is poorly constrained. Redo with tilted views.")
    if not args.dry_run:
        directory = Path(args.calibration_dir)
        _backup_existing(intrinsics_path(camera_key, directory))
        path = save_intrinsics(camera_key, resolution, K, result.distortion_coeffs, result.rms_reprojection_px, directory)
        print(f"Saved {path}")


def _placements_from_images(directory: Path, camera_ids: list[str], detector: CharucoBoardDetector) -> tuple[list[dict[str, CharucoDetection]], tuple[int, int]]:
    # <dir>/<camera_id>/<placement>.png: files with the same name are the same board placement.
    by_name: dict[str, dict[str, CharucoDetection]] = {}
    resolution = None
    for camera_id in camera_ids:
        for image_path in _image_files(directory / camera_id):
            image = cv2.imread(str(image_path))
            resolution = (image.shape[1], image.shape[0])
            detection = detector.detect(image)
            if detection is not None:
                by_name.setdefault(image_path.stem, {})[camera_id] = detection
    if resolution is None:
        raise SystemExit(f"No images found under {directory}/<camera_id>/")
    return [by_name[name] for name in sorted(by_name)], resolution


def _placements_from_cameras(args: argparse.Namespace, device_ids: list[int], detector: CharucoBoardDetector) -> tuple[list[dict[str, CharucoDetection]], tuple[int, int]]:
    resolution = _parse_resolution(args.resolution)
    pixel_scale = resolution[0] / REFERENCE_WIDTH_PX
    captures = {str(device_id): _open_camera(device_id, resolution) for device_id in device_ids}
    windows = {camera_id: deque(maxlen=RIG_STEADY_FRAMES) for camera_id in captures}
    placements: list[dict[str, CharucoDetection]] = []
    print("Put the board on a stand where the lifter stands, facing the cameras; do NOT hand-hold it.")
    print("Each placement is captured once the board is still in every camera that sees it (at least 2).")
    print("Move and tilt it between placements. SPACE = capture now, q = done.")
    while len(placements) < args.placements:
        for capture in captures.values():  # grab everything first: smallest skew between cameras
            capture.grab()
        frames = {camera_id: capture.retrieve()[1] for camera_id, capture in captures.items()}
        if any(frame is None for frame in frames.values()):
            continue
        detections = {}
        for camera_id, frame in frames.items():
            _require_resolution(frame, resolution, int(camera_id))
            detections[camera_id] = detector.detect(frame)
            if detections[camera_id] is None:
                windows[camera_id].clear()
            else:
                windows[camera_id].append(detections[camera_id])

        seeing = [camera_id for camera_id, detection in detections.items() if detection is not None]
        all_steady = all(_is_steady(windows[camera_id], RIG_MAX_MOTION_PX * pixel_scale) for camera_id in seeing)
        novelty_px = max(
            (
                view_novelty_px(detections[camera_id], [taken[camera_id] for taken in placements if camera_id in taken])
                for camera_id in seeing
            ),
            default=0.0,
        )
        key = cv2.waitKey(1) & 0xFF
        ready = len(seeing) >= 2 and all_steady and novelty_px >= MIN_PLACEMENT_NOVELTY_PX * pixel_scale
        if ready or (key == KEY_FORCE and len(seeing) >= 2):
            placements.append({camera_id: average_detections(list(windows[camera_id])) for camera_id in seeing})
            for window in windows.values():
                window.clear()
            print(f"  placement {len(placements)}/{args.placements} seen by cameras {', '.join(seeing)}")
        previews = [
            _draw_preview(
                frames[camera_id], detections[camera_id],
                [
                    f"cam {camera_id}", f"placements {len(placements)}/{args.placements}",
                    "STEADY" if _is_steady(windows[camera_id], RIG_MAX_MOTION_PX * pixel_scale) else "",
                ],
            )
            for camera_id in captures
        ]
        cv2.imshow(WINDOW_NAME, np.hstack(previews))
        if key in KEY_QUIT:
            break
    for capture in captures.values():
        capture.release()
    cv2.destroyAllWindows()
    return placements, resolution


def run_extrinsics(args: argparse.Namespace) -> None:
    spec = _board_spec(args)
    detector = CharucoBoardDetector(spec)
    device_ids = [int(value) for value in args.cameras.split(",")]
    camera_ids = [str(device_id) for device_id in device_ids]
    keys = args.keys.split(",") if args.keys else camera_ids
    if len(keys) != len(camera_ids):
        raise SystemExit("--keys needs one name per camera in --cameras")

    if args.from_images:
        placements, resolution = _placements_from_images(Path(args.from_images), camera_ids, detector)
    else:
        placements, resolution = _placements_from_cameras(args, device_ids, detector)
    directory = Path(args.calibration_dir)
    intrinsics = load_rig_intrinsics(dict(zip(camera_ids, keys)), resolution, directory)
    missing = [camera_id for camera_id in camera_ids if camera_id not in intrinsics]
    if missing:
        raise SystemExit(
            f"No intrinsics at {resolution[0]}x{resolution[1]} for camera(s) {', '.join(missing)}: "
            f"run 'intrinsics --camera <id>' for each first"
        )

    flat_index = None if args.flat_placement is None else args.flat_placement - 1
    result = solve_rig_extrinsics(placements, spec, intrinsics, resolution, flat_placement_index=flat_index)
    print(f"\nRig solved from {len(placements)} placements (world anchor: {result.world_anchor})")
    centres = {
        camera_id: -camera.rotation_matrix.T @ camera.translation_vector.reshape(3)
        for camera_id, camera in result.cameras.items()
    }
    for camera_id, camera in result.cameras.items():
        x, y, z = centres[camera_id]
        print(f"  camera {camera_id}: reprojection {camera.reprojection_error:.2f} px   centre x {x:+.3f}  y {y:+.3f}  z {z:+.3f} m")
    for first in camera_ids:
        for second in camera_ids:
            if first < second:
                print(f"  distance {first}-{second}: {np.linalg.norm(centres[first] - centres[second]):.3f} m   <- check against a tape measure")
    if flat_index is None:
        print("  Y-down is the mean of the cameras' image-down axes (approximate). person_calibration.refine re-anchors the frame to the lifter.")
    if not args.dry_run:
        path = rig_calibration_path(device_ids, directory)
        path.parent.mkdir(parents=True, exist_ok=True)
        _backup_existing(path)
        TPoseCalibrator.save_calibration(result, str(path))
        print(f"Saved {path}")


def run_check(args: argparse.Namespace) -> None:
    try:
        from biomechanics.triangulation.person_calibration import reprojection_health_px
    except ImportError:
        raise SystemExit("biomechanics.triangulation.person_calibration is not available yet (WS-B)")
    from biomechanics.pose.rtmpose import RTMPoseEstimator
    from biomechanics.triangulation.multi_capture import MultiCameraCapture

    device_ids = [int(value) for value in args.cameras.split(",")]
    path = rig_calibration_path(device_ids, Path(args.calibration_dir))
    if not path.exists():
        raise SystemExit(f"No rig calibration at {path}")
    calibration = TPoseCalibrator.load_calibration(str(path))
    estimator = RTMPoseEstimator(keypoint_format="halpe26", batch_size=len(device_ids))
    estimator.initialize()
    capture = MultiCameraCapture(device_ids=device_ids, resolution=_parse_resolution(args.resolution))
    capture.start()
    print(f"Stand where you lift, fully visible; collecting {CHECK_FRAMES} frames...")
    views = []
    while len(views) < CHECK_FRAMES:
        synced = capture.get_synced_frames()
        if synced is None:
            continue
        camera_ids = list(synced.frames)
        skeletons = estimator.estimate_batch(
            [synced.frames[camera_id] for camera_id in camera_ids], camera_ids=camera_ids
        )
        frame_views = {
            camera_id: skeleton for camera_id, skeleton in zip(camera_ids, skeletons) if skeleton is not None
        }
        if len(frame_views) >= 2:
            views.append(frame_views)
    capture.release()
    estimator.release()
    rms_px = reprojection_health_px(calibration, views)
    verdict = "OK" if rms_px <= CHECK_GOOD_RMS_PX else "DRIFTED? recalibrate or run person refinement"
    print(f"Reprojection health: {rms_px:.2f} px over {len(views)} frames ({verdict}; pose keypoint noise alone is ~2-3 px)")


def _add_directory_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--calibration-dir", default=str(NOWVA_CALIBRATION_DIR), help="where calibration files live (the pipeline reads ~/.nowva)")


def _add_board_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--squares", default="7x5", help="squares across x down (default 7x5)")
    parser.add_argument("--square-mm", type=float, default=35.0, help="MEASURED square side in mm")
    parser.add_argument("--marker-mm", type=float, default=26.0, help="MEASURED marker side in mm")
    parser.add_argument("--dict", default="DICT_5X5_100", help="cv2.aruco dictionary name")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    board = commands.add_parser("board", help="write a printable board image")
    _add_board_arguments(board)
    board.add_argument("--out", default="board.png")
    board.add_argument("--paper", choices=sorted(PAPER_SIZES_MM), default="A4")
    board.add_argument("--dpi", type=int, default=300)
    board.set_defaults(run=run_board)

    intrinsics = commands.add_parser("intrinsics", help="calibrate one camera's K and lens distortion")
    _add_board_arguments(intrinsics)
    intrinsics.add_argument("--camera", type=int, default=0, help="OpenCV device id")
    intrinsics.add_argument("--key", help="name for intrinsics_<key>.json (default: the device id, which the pipeline looks up)")
    intrinsics.add_argument("--resolution", default="1280x720", help="must equal the pipeline capture resolution")
    intrinsics.add_argument("--views", type=int, default=30)
    intrinsics.add_argument("--from-images", help="directory of board photos instead of a live camera")
    intrinsics.add_argument("--dry-run", action="store_true", help="solve and print, do not save")
    _add_directory_argument(intrinsics)
    intrinsics.set_defaults(run=run_intrinsics)

    extrinsics = commands.add_parser("extrinsics", help="factory rig calibration: one board seen by all cameras")
    _add_board_arguments(extrinsics)
    extrinsics.add_argument("--cameras", default="0,1,2", help="comma-separated OpenCV device ids")
    extrinsics.add_argument("--keys", help="comma-separated intrinsics keys, one per camera (default: the device ids)")
    extrinsics.add_argument("--resolution", default="1280x720")
    extrinsics.add_argument("--placements", type=int, default=10)
    extrinsics.add_argument("--flat-placement", type=int, help="1-based placement where the board lies flat on the floor: sets Y-down to gravity")
    extrinsics.add_argument("--from-images", help="directory with <camera_id>/<placement>.png files")
    extrinsics.add_argument("--dry-run", action="store_true", help="solve and print, do not save")
    _add_directory_argument(extrinsics)
    extrinsics.set_defaults(run=run_extrinsics)

    check = commands.add_parser("check", help="reprojection health of the saved rig calibration on live frames")
    check.add_argument("--cameras", default="0,1,2")
    check.add_argument("--resolution", default="1280x720")
    _add_directory_argument(check)
    check.set_defaults(run=run_check)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
