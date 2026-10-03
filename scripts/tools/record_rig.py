#!/usr/bin/env python3
"""
Record the rack's cameras raw, for offline replay through the real pipeline
(docs/deadlift/PLAN.md §8.2: round-1 capture, detector data, validation sets).

    python scripts/tools/record_rig.py --cameras 0,1,2 --seconds 120 --out recordings/lifter01_set1 \
        --meta lifter=01 lift=deadlift load_kg=60 scripted_fault=bar_drift

Writes one video per camera, the synced-set log, a copy of the rig's calibration,
intrinsics and gravity files, and metadata.json into --out. It writes nothing to
~/.nowva. Replay with NOWVA_MULTI_CAMERA=true NOWVA_REPLAY_DIR=<out>, pointing
triangulation.calibration_file at the copied rig file.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from biomechanics.triangulation.calibration import NOWVA_CALIBRATION_DIR  # noqa: E402
from biomechanics.triangulation.multi_capture import MultiCameraCapture  # noqa: E402
from biomechanics.triangulation.rig_recording import RigRecordingWriter  # noqa: E402

DEFAULT_FPS = 30.0
CALIBRATION_PATTERNS = ("rig_calibration_*.json", "intrinsics_*.json", "gravity_*.json")
PROGRESS_EVERY_S = 5.0


def _parse_meta(pairs: list[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for pair in pairs:
        key, separator, value = pair.partition("=")
        if not separator:
            raise SystemExit(f"--meta expects key=value, got '{pair}'")
        metadata[key] = value
    return metadata


def copy_calibration(source: Path, destination: Path) -> list[str]:
    """Copy the rig's calibration files so the recording replays on the same cameras."""
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for pattern in CALIBRATION_PATTERNS:
        for path in sorted(source.glob(pattern)):
            shutil.copy2(path, destination / path.name)
            copied.append(path.name)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cameras", default="0,1,2", help="comma-separated device ids; the first is the primary")
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--resolution", default="1280x720")
    parser.add_argument("--meta", nargs="*", default=[], help="key=value pairs stored in metadata.json")
    args = parser.parse_args()

    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} is not empty: pick a new directory per recording")
    device_ids = [int(camera) for camera in args.cameras.split(",")]
    width, height = (int(value) for value in args.resolution.split("x"))

    capture = MultiCameraCapture(device_ids=device_ids, resolution=(width, height), primary_device_id=device_ids[0])
    capture.start()
    writer = RigRecordingWriter(args.out, [str(device_id) for device_id in device_ids], args.fps)
    started_s = time.monotonic()
    next_progress_s = started_s + PROGRESS_EVERY_S
    try:
        while time.monotonic() - started_s < args.seconds:
            synced = capture.get_synced_frames()
            if synced is not None:
                writer.write(synced)
            if time.monotonic() >= next_progress_s:
                print(f"  {writer.sets_written} synced sets, lost cameras: {capture.lost_cameras() or 'none'}")
                next_progress_s += PROGRESS_EVERY_S
    finally:
        fps_stats = capture.get_fps_stats()
        capture.release()
        writer.write_metadata({
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "cameras": device_ids,
            "resolution": capture.actual_resolutions,
            "fps": fps_stats,
            "synced_sets": writer.sets_written,
            "calibration_files": copy_calibration(NOWVA_CALIBRATION_DIR, args.out / "calibration"),
            **_parse_meta(args.meta),
        })
        writer.close()
    print(f"Recorded {writer.sets_written} synced sets to {args.out}")


if __name__ == "__main__":
    main()
