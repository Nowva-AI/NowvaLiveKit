#!/usr/bin/env python3
"""
Measure gravity per camera for the deadlift (docs/deadlift/PLAN.md §2.1). Lay the ChArUco
board FLAT on the floor where the bar sits, in view of the cameras, then run

    python scripts/tools/measure_gravity.py --cameras 0,1,2

Each camera solves the board pose from a steady window of frames; the board normal pointing
up, in that camera's own frame, is saved to ~/.nowva/gravity_<camera_key>.json. The tool
writes nothing else. Pass the same board arguments as calibrate_cameras.py and use the large
factory board: a small board on the floor 3 m away gives a poor normal (see normal sigma).
Offline: --from-images <dir> with <dir>/<camera_id>/*.png of the still board.
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from biomechanics.deadlift.gravity import (  # noqa: E402
    CameraGravity,
    gravity_path,
    measure_camera_gravity,
    save_camera_gravity,
    world_up_from_cameras,
)
from biomechanics.triangulation.calibration import (  # noqa: E402
    NOWVA_CALIBRATION_DIR,
    TPoseCalibrator,
    load_rig_intrinsics,
    rig_calibration_path,
)
from biomechanics.triangulation.charuco import (  # noqa: E402
    CharucoBoardDetector,
    CharucoBoardSpec,
    CharucoDetection,
    view_novelty_px,
)
from biomechanics.utils.geometry import WORLD_UP  # noqa: E402

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
REFERENCE_WIDTH_PX = 1280.0  # pixel thresholds below are for this width and scale with it
# The board must lie still for the whole window, whose frames are then averaged.
MAX_MOTION_PX = 0.2
GOOD_NORMAL_SIGMA_DEG = 0.3  # PLAN.md §8.4 gravity target
PREVIEW_WIDTH_PX = 480
WINDOW_NAME = "measure_gravity"
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


def _is_steady(window: deque, max_motion_px: float) -> bool:
    if len(window) < window.maxlen:
        return False
    frames = list(window)
    return all(view_novelty_px(b, [a]) <= max_motion_px for a, b in zip(frames, frames[1:]))


def _windows_from_images(
    directory: Path, camera_ids: list[str], detector: CharucoBoardDetector
) -> tuple[dict[str, list[CharucoDetection]], tuple[int, int]]:
    windows: dict[str, list[CharucoDetection]] = {}
    resolution = None
    for camera_id in camera_ids:
        camera_dir = directory / camera_id
        image_paths = sorted(p for p in camera_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES) if camera_dir.is_dir() else []
        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            resolution = (image.shape[1], image.shape[0])
            detection = detector.detect(image)
            if detection is not None:
                windows.setdefault(camera_id, []).append(detection)
    if resolution is None:
        raise SystemExit(f"No images found under {directory}/<camera_id>/")
    return windows, resolution


def _windows_from_cameras(
    args: argparse.Namespace, device_ids: list[int], detector: CharucoBoardDetector
) -> tuple[dict[str, list[CharucoDetection]], tuple[int, int]]:
    resolution = _parse_resolution(args.resolution)
    max_motion_px = MAX_MOTION_PX * resolution[0] / REFERENCE_WIDTH_PX
    captures = {}
    for device_id in device_ids:
        capture = cv2.VideoCapture(device_id)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not capture.isOpened():
            raise SystemExit(f"Could not open camera device {device_id}")
        captures[str(device_id)] = capture
    windows = {camera_id: deque(maxlen=args.frames) for camera_id in captures}
    print("Board flat on the floor, nobody touching it. Captures once every camera holds a steady")
    print(f"window of {args.frames} frames. SPACE = use the cameras that are steady now, q = abort.")
    steady: list[str] = []
    while True:
        for capture in captures.values():
            capture.grab()
        frames = {camera_id: capture.retrieve()[1] for camera_id, capture in captures.items()}
        if any(frame is None for frame in frames.values()):
            continue
        for camera_id, frame in frames.items():
            if (frame.shape[1], frame.shape[0]) != resolution:
                raise SystemExit(f"Camera {camera_id} delivers {frame.shape[1]}x{frame.shape[0]}, not {args.resolution}")
            detection = detector.detect(frame)
            if detection is None:
                windows[camera_id].clear()
            else:
                windows[camera_id].append(detection)
        steady = [camera_id for camera_id in captures if _is_steady(windows[camera_id], max_motion_px)]
        previews = []
        for camera_id, frame in frames.items():
            scale = PREVIEW_WIDTH_PX / frame.shape[1]
            preview = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            status = f"cam {camera_id}: {len(windows[camera_id])}/{args.frames} {'STEADY' if camera_id in steady else ''}"
            cv2.putText(preview, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            previews.append(preview)
        cv2.imshow(WINDOW_NAME, np.hstack(previews))
        key = cv2.waitKey(1) & 0xFF
        if key in KEY_QUIT:
            steady = []
            break
        if len(steady) == len(captures) or (key == KEY_FORCE and steady):
            break
    for capture in captures.values():
        capture.release()
    cv2.destroyAllWindows()
    return {camera_id: list(windows[camera_id]) for camera_id in steady}, resolution


def _print_world_check(device_ids: list[int], measured: dict[str, CameraGravity], directory: Path) -> None:
    # Read-only cross-check against the saved rig calibration, when there is one.
    path = rig_calibration_path(device_ids, directory)
    if not path.exists() or len(measured) < 2:
        return
    calibration = TPoseCalibrator.load_calibration(str(path))
    rotations = {camera_id: camera.rotation_matrix for camera_id, camera in calibration.cameras.items()}
    per_camera_up = {camera_id: np.array(gravity.up_camera) for camera_id, gravity in measured.items()}
    up_world, used, source = world_up_from_cameras(per_camera_up, rotations)
    print(f"\nWorld frame of {path.name}: gravity source '{source}', cameras used {used or '-'}")
    for camera_id, up_camera in per_camera_up.items():
        if camera_id not in rotations:
            continue
        mapped = rotations[camera_id].T @ up_camera
        off_deg = np.degrees(np.arccos(np.clip(mapped @ up_world, -1.0, 1.0)))
        print(f"  camera {camera_id}: {off_deg:.2f} deg from the consensus")
    tilt_deg = np.degrees(np.arccos(np.clip(up_world @ WORLD_UP, -1.0, 1.0)))
    print(f"  world Y is {tilt_deg:.2f} deg off measured gravity")


def run(args: argparse.Namespace) -> None:
    spec = _board_spec(args)
    detector = CharucoBoardDetector(spec)
    device_ids = [int(value) for value in args.cameras.split(",")]
    camera_ids = [str(device_id) for device_id in device_ids]
    keys = args.keys.split(",") if args.keys else camera_ids
    if len(keys) != len(camera_ids):
        raise SystemExit("--keys needs one name per camera in --cameras")
    camera_keys = dict(zip(camera_ids, keys))

    if args.from_images:
        windows, resolution = _windows_from_images(Path(args.from_images), camera_ids, detector)
    else:
        windows, resolution = _windows_from_cameras(args, device_ids, detector)
    directory = Path(args.calibration_dir)
    intrinsics = load_rig_intrinsics(camera_keys, resolution, directory)

    measured: dict[str, CameraGravity] = {}
    for camera_id in camera_ids:
        if camera_id not in windows:
            print(f"camera {camera_id}: board not seen steadily; nothing saved")
            continue
        if camera_id not in intrinsics:
            print(f"camera {camera_id}: no intrinsics at {resolution[0]}x{resolution[1]}; run calibrate_cameras.py intrinsics first")
            continue
        gravity = measure_camera_gravity(windows[camera_id], spec, *intrinsics[camera_id], camera_keys[camera_id])
        if gravity is None:
            print(f"camera {camera_id}: board pose failed; nothing saved")
            continue
        measured[camera_id] = gravity
        up = np.array(gravity.up_camera)
        # Image up is -y in the camera frame: how far the camera's vertical axis leans from gravity.
        lean_deg = np.degrees(np.arccos(np.clip(-up[1], -1.0, 1.0)))
        quality = "OK" if gravity.normal_sigma_deg <= GOOD_NORMAL_SIGMA_DEG else "POOR: use a bigger board or bring it closer"
        print(
            f"camera {camera_id} ('{gravity.camera_key}'): up {np.round(up, 4).tolist()}  image-vertical lean "
            f"{lean_deg:.1f} deg  residual {gravity.residual_px:.2f} px  normal sigma {gravity.normal_sigma_deg:.2f} deg "
            f"({quality})  {gravity.num_corners} corners x {gravity.num_frames} frames"
        )
        if not args.dry_run:
            print(f"  saved {save_camera_gravity(gravity, directory)}")
        else:
            print(f"  dry run: would save {gravity_path(gravity.camera_key, directory)}")
    _print_world_check(device_ids, measured, directory)
    if not measured:
        raise SystemExit("No gravity measured")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cameras", default="0,1,2", help="comma-separated OpenCV device ids")
    parser.add_argument("--keys", help="comma-separated intrinsics keys, one per camera (default: the device ids)")
    parser.add_argument("--resolution", default="1280x720", help="must equal the pipeline capture resolution")
    parser.add_argument("--frames", type=int, default=30, help="steady frames averaged per camera")
    parser.add_argument("--from-images", help="directory with <camera_id>/*.png of the still, flat board")
    parser.add_argument("--squares", default="7x5", help="squares across x down (default 7x5)")
    parser.add_argument("--square-mm", type=float, default=35.0, help="MEASURED square side in mm")
    parser.add_argument("--marker-mm", type=float, default=26.0, help="MEASURED marker side in mm")
    parser.add_argument("--dict", default="DICT_5X5_100", help="cv2.aruco dictionary name")
    parser.add_argument("--calibration-dir", default=str(NOWVA_CALIBRATION_DIR), help="where calibration files live (the pipeline reads ~/.nowva)")
    parser.add_argument("--dry-run", action="store_true", help="measure and print, do not save")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
