"""
Raw skeleton recorder and replay (SHARED-PLUMBING.md §8).

The recorder writes one JSON line per processed frame, latch or not: the raw
3D skeleton with confidences, the raw 2D skeleton, the capture timestamp and
lost cameras, plus every control event between frames (set_exercise,
reset_readiness_gate, presence_only, reset_rep_counter, rest_start...) and
the live outputs (latch, reps, faults). Replay feeds the recorded skeletons
back as the pipeline's pose source and re-applies the events before the
frame they preceded, so the real pipeline runs again from the readiness gate
through pre-IK, IK, the analyser or counter and the rules.

The header keeps the single camera's mount (config fields and the
NOWVA_CAMERA_PITCH_DEG override), which replay re-applies so pitch
certification comes out the same.

Not recorded: barbell detections (no images are kept) and frames processed
before recording starts (a multi-camera calibration bootstrap).
Usage: python -m biomechanics.replay.raw_skeleton <recording.jsonl>
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

import numpy as np

from biomechanics.config import BiomechanicsConfig, load_pipeline_config
from biomechanics.faults.observability import SINGLE_CAMERA, TRIANGULATED
from biomechanics.utils.camera_mount import PITCH_ENV
from biomechanics.utils.types import Keypoint2D, PipelineFrame, Skeleton2D, Skeleton3D

logger = logging.getLogger(__name__)

RECORDING_VERSION = 1
MULTI_CAMERA_ENV = "NOWVA_MULTI_CAMERA"
# Replayed frames carry no image; analysis never reads one.
PLACEHOLDER_FRAME = np.zeros((1, 1, 3), dtype=np.uint8)
GIT_TIMEOUT_S = 2.0
# CaptureConfig fields describing the single camera's physical mount.
MOUNT_FIELDS = ("mount_source", "mount_pitch_deg", "mount_height_m")


def config_hash(config: BiomechanicsConfig) -> str:
    return hashlib.sha256(config.model_dump_json().encode()).hexdigest()


def _git_sha() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            timeout=GIT_TIMEOUT_S, cwd=Path(__file__).parent,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _skeleton_3d_rows(skeleton: Skeleton3D | None) -> list[list[float]] | None:
    if skeleton is None:
        return None
    return [[kp.x, kp.y, kp.z, kp.confidence] for kp in skeleton.keypoints]


def _skeleton_2d_rows(skeleton: Skeleton2D | None) -> list[list[float]] | None:
    if skeleton is None:
        return None
    return [[kp.x, kp.y, kp.confidence] for kp in skeleton.keypoints]


class RawSkeletonRecorder:
    """Appends a session's raw frames, control events and outputs to a JSONL file."""

    def __init__(
        self, path: Path, config: BiomechanicsConfig, exercise_name: str, multi_camera: bool,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file: IO[str] | None = open(self.path, "w", buffering=1)
        self.frames_recorded = 0
        self._latched = False
        self._write({
            "kind": "header",
            "version": RECORDING_VERSION,
            "exercise_name": exercise_name,
            "capture_mode": TRIANGULATED if multi_camera else SINGLE_CAMERA,
            "config_hash": config_hash(config),
            "git_sha": _git_sha(),
            "camera_mount": {field: getattr(config.capture, field) for field in MOUNT_FIELDS},
            "camera_pitch_env": os.getenv(PITCH_ENV),
        })

    def _write(self, record: dict[str, Any]) -> None:
        if self._file is not None:
            self._file.write(json.dumps(record) + "\n")

    def record_frame(
        self,
        timestamp: float,
        frame_index: int,
        frame_shape: tuple[int, ...],
        skeleton_2d: Skeleton2D | None,
        skeleton_3d: Skeleton3D | None,
        lost_cameras: list[str] | None,
    ) -> None:
        self._write({
            "kind": "frame",
            "index": self.frames_recorded,
            "timestamp": timestamp,
            "frame_index": frame_index,
            "frame_shape": list(frame_shape[:2]),
            "skeleton_3d": _skeleton_3d_rows(skeleton_3d),
            "skeleton_3d_timestamp": skeleton_3d.timestamp if skeleton_3d is not None else None,
            "skeleton_2d": _skeleton_2d_rows(skeleton_2d),
            "lost_cameras": lost_cameras,
        })
        self.frames_recorded += 1

    def record_event(self, kind: str, **data: Any) -> None:
        self._write({"kind": "event", "before_frame": self.frames_recorded, "event": kind, "data": data})

    def record_outputs(self, result: PipelineFrame, latched: bool) -> None:
        """Latch changes, reps and faults of the frame just processed."""
        latch_changed = latched != self._latched
        self._latched = latched
        if not (latch_changed or result.faults or result.rep_data is not None):
            return
        self._write({
            "kind": "output",
            "frame": self.frames_recorded - 1,
            "timestamp": result.timestamp,
            "latched": latched,
            "rep_number": result.rep_data.rep_number if result.rep_data is not None else None,
            "faults": [
                {"fault_type": fault.fault_type, "severity": fault.severity.value, "rep_number": fault.rep_number}
                for fault in result.faults
            ],
        })

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


@dataclass
class Recording:
    header: dict[str, Any]
    frames: list[dict[str, Any]]
    events: list[dict[str, Any]]
    outputs: list[dict[str, Any]]


def load_recording(path: Path) -> Recording:
    header: dict[str, Any] = {}
    frames: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        kind = record["kind"]
        if kind == "header":
            header = record
        elif kind == "frame":
            frames.append(record)
        elif kind == "event":
            events.append(record)
        elif kind == "output":
            outputs.append(record)
    if header.get("version") != RECORDING_VERSION:
        raise ValueError(f"{path}: unsupported recording version {header.get('version')!r}")
    return Recording(header=header, frames=frames, events=events, outputs=outputs)


class ReplayPoseSource:
    """Serves recorded raw skeletons to BiomechanicsPipeline(pose_source=...)."""

    def __init__(self, frames: list[dict[str, Any]]) -> None:
        self._frames = frames
        self._next = 0
        self.last_capture_timestamp = math.nan
        self.swap_count = 0
        self._lost_cameras: list[str] | None = None

    def get_pose(self) -> tuple[np.ndarray, Skeleton2D | None, Skeleton3D | None]:
        record = self._frames[self._next]
        self._next += 1
        self.last_capture_timestamp = record["timestamp"]
        self._lost_cameras = record["lost_cameras"]
        skeleton_3d = None
        if record["skeleton_3d"] is not None:
            rows = np.asarray(record["skeleton_3d"], dtype=np.float64)
            skeleton_3d = Skeleton3D.from_numpy(
                rows[:, :3], confidences=rows[:, 3],
                timestamp=record["skeleton_3d_timestamp"], frame_index=record["frame_index"],
            )
        skeleton_2d = None
        if record["skeleton_2d"] is not None:
            skeleton_2d = Skeleton2D(
                keypoints=[Keypoint2D(x=x, y=y, confidence=c) for x, y, c in record["skeleton_2d"]],
                timestamp=record["timestamp"], frame_index=record["frame_index"],
            )
        return PLACEHOLDER_FRAME, skeleton_2d, skeleton_3d

    def lost_cameras(self) -> list[str] | None:
        return self._lost_cameras

    def reset_temporal_state(self) -> None:
        pass


def _apply_event(pipeline, event: str, data: dict[str, Any]) -> None:
    if event == "set_exercise":
        pipeline.set_exercise(data["exercise_name"])
    elif event == "reset_readiness_gate":
        pipeline.reset_readiness_gate()
    elif event == "reset_rep_counter":
        pipeline.reset_rep_counter()
    elif event == "reset_bilstm":
        pipeline.reset_bilstm()
    elif event == "presence_only":
        pipeline.presence_only = data["value"]
    elif event == "apply_athlete_params":
        pipeline.apply_athlete_params(data["params"])
    elif event == "set_depth_target":
        pipeline.set_depth_target(data["target_ratio"])
    elif event == "set_assessment_mode":
        pipeline.set_assessment_mode(data["enabled"])
    elif event == "apply_calibration":
        pipeline.apply_calibration(data["calibration_profile"])
    elif event == "set_readiness_knee_limit":
        pipeline.set_readiness_knee_limit(data["max_knee_flexion_deg"])
    # Annotations (rest_start, revoke_pitch_certification...) are outcomes or
    # context, not inputs: the replayed pipeline recomputes them.


@dataclass
class ReplayResult:
    header: dict[str, Any]
    frames: list[PipelineFrame]

    @property
    def reps(self) -> list[int]:
        return [frame.rep_data.rep_number for frame in self.frames if frame.rep_data is not None]


def _set_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def replay_recording(path: Path, config: BiomechanicsConfig | None = None) -> ReplayResult:
    """Run a recording through a fresh pipeline built for its capture mode,
    exercise and camera mount."""
    from biomechanics.pipeline import BiomechanicsPipeline

    recording = load_recording(path)
    config = config or load_pipeline_config()
    if config_hash(config) != recording.header.get("config_hash"):
        logger.warning("[REPLAY] Config differs from the recorded session's — outputs may differ")
    events_before: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in recording.events:
        events_before[event["before_frame"]].append(event)

    # A recording made before the mount was recorded keeps the replay's own.
    config = config.model_copy(update={
        "capture": config.capture.model_copy(update=recording.header.get("camera_mount", {})),
    })

    previous_mode = os.environ.get(MULTI_CAMERA_ENV)
    previous_pitch = os.environ.get(PITCH_ENV)
    os.environ[MULTI_CAMERA_ENV] = "true" if recording.header["capture_mode"] == TRIANGULATED else "false"
    _set_env(PITCH_ENV, recording.header.get("camera_pitch_env", previous_pitch))
    try:
        pipeline = BiomechanicsPipeline(
            config,
            exercise_name=recording.header["exercise_name"],
            defer_capture=True,
            pose_source=ReplayPoseSource(recording.frames),
        )
        results: list[PipelineFrame] = []
        for index in range(len(recording.frames)):
            for event in events_before.get(index, []):
                _apply_event(pipeline, event["event"], event["data"])
            results.append(pipeline.process_frame())
    finally:
        _set_env(MULTI_CAMERA_ENV, previous_mode)
        _set_env(PITCH_ENV, previous_pitch)
    return ReplayResult(header=recording.header, frames=results)


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python -m biomechanics.replay.raw_skeleton <recording.jsonl>")
        return 2
    path = Path(argv[0])
    recorded = [output["rep_number"] for output in load_recording(path).outputs if output["rep_number"] is not None]
    replayed = replay_recording(path).reps
    print(f"recorded reps: {recorded}\nreplayed reps: {replayed}")
    return 0 if recorded == replayed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
