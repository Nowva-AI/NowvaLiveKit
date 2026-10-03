"""Raw rig recordings and their replay (docs/deadlift/PLAN.md §8.2).

A recording is one video per camera plus a log of the synced sets as the live
capture assembled them (primary sequence, primary timestamp, which cameras were
in the set), so a replay hands the real pipeline exactly the sets it saw live:
same timestamps, same dropouts. RecordedCapture has MultiCameraCapture's
interface; the pose provider swaps it in when a replay directory is given.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .multi_capture import SyncedFrames

FRAMES_LOG = "frames.jsonl"
METADATA_FILE = "metadata.json"
# Motion JPEG: every frame a keyframe, so a camera missing from a set never
# shifts the others, and it decodes everywhere OpenCV does.
VIDEO_FOURCC = "MJPG"
VIDEO_SUFFIX = ".avi"
JPEG_QUALITY = 95


def camera_video_path(directory: Path, camera_id: str) -> Path:
    return directory / f"cam_{camera_id}{VIDEO_SUFFIX}"


class RigRecordingWriter:
    """Writes synced sets as they come from MultiCameraCapture.get_synced_frames()."""

    def __init__(self, directory: Path, camera_ids: list[str], fps: float) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._camera_ids = list(camera_ids)
        self._fps = fps
        self._writers: dict[str, cv2.VideoWriter] = {}
        self._log = open(self.directory / FRAMES_LOG, "w")
        self.sets_written = 0

    def write(self, synced: SyncedFrames) -> None:
        for camera_id, frame in synced.frames.items():
            writer = self._writers.get(camera_id)
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(camera_video_path(self.directory, camera_id)),
                    cv2.VideoWriter_fourcc(*VIDEO_FOURCC), self._fps, (width, height),
                )
                writer.set(cv2.VIDEOWRITER_PROP_QUALITY, JPEG_QUALITY)
                self._writers[camera_id] = writer
            writer.write(frame)
        self._log.write(json.dumps({
            "sequence": synced.sequence,
            "timestamp": synced.timestamp,
            "cameras": sorted(synced.frames),
        }) + "\n")
        self.sets_written += 1

    def write_metadata(self, metadata: dict) -> None:
        (self.directory / METADATA_FILE).write_text(json.dumps(metadata, indent=2, sort_keys=True))

    def close(self) -> None:
        for writer in self._writers.values():
            writer.release()
        self._writers.clear()
        self._log.close()


class RecordedCapture:
    """Replays a recording through MultiCameraCapture's interface: each call to
    get_synced_frames() returns the next recorded set, then None at the end."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        with open(self.directory / FRAMES_LOG) as log:
            self._sets = [json.loads(line) for line in log if line.strip()]
        self._readers: dict[str, cv2.VideoCapture] = {}
        self._next_set = 0
        self._actual_resolutions: dict[str, tuple[int, int]] = {}

    def start(self) -> None:
        camera_ids = sorted({camera_id for entry in self._sets for camera_id in entry["cameras"]})
        for camera_id in camera_ids:
            reader = cv2.VideoCapture(str(camera_video_path(self.directory, camera_id)))
            if not reader.isOpened():
                raise RuntimeError(f"Recording {self.directory} has no readable video for camera {camera_id}")
            self._readers[camera_id] = reader
            self._actual_resolutions[camera_id] = (
                int(reader.get(cv2.CAP_PROP_FRAME_WIDTH)), int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )

    @property
    def actual_resolutions(self) -> dict[str, tuple[int, int]]:
        return dict(self._actual_resolutions)

    @property
    def finished(self) -> bool:
        return self._next_set >= len(self._sets)

    def get_synced_frames(self) -> SyncedFrames | None:
        if self.finished:
            return None
        entry = self._sets[self._next_set]
        self._next_set += 1
        frames: dict[str, np.ndarray] = {}
        for camera_id in entry["cameras"]:
            ok, frame = self._readers[camera_id].read()
            if not ok:
                raise RuntimeError(f"Recording {self.directory}: camera {camera_id} video ends early")
            frames[camera_id] = frame
        return SyncedFrames(frames=frames, sequence=entry["sequence"], timestamp=entry["timestamp"])

    def lost_cameras(self) -> list[str]:
        return []

    def get_fps_stats(self) -> dict[str, float]:
        """Recorded frames per second of each camera over the whole recording."""
        elapsed_s = self._sets[-1]["timestamp"] - self._sets[0]["timestamp"] if self._sets else 0.0
        if elapsed_s <= 0.0:
            return {}
        return {
            camera_id: sum(camera_id in entry["cameras"] for entry in self._sets) / elapsed_s
            for camera_id in self._readers
        }

    def release(self) -> None:
        for reader in self._readers.values():
            reader.release()
        self._readers.clear()
