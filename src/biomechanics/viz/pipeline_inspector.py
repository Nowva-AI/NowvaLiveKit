"""Pipeline inspection recorder.

Records the preview video, the 3D skeleton after every pre-IK chain stage
(stage names come from the chain itself), joint angles, and pipeline
metadata for every frame, then builds a self-contained HTML report
with synchronized video + skeleton + angle-trace scrubbing.
Enabled by the --inspect flag; inert otherwise.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from biomechanics.utils.json_safe import nan_to_none
from biomechanics.utils.types import PipelineFrame

TEMPLATE_PATH = Path(__file__).with_name("pipeline_inspector_template.html")

MAX_VIDEO_WIDTH = 960
VIDEO_CRF = "26"
MIN_RETIME_FPS = 1.0
MAX_RETIME_FPS = 120.0

ANGLE_KEYS = [
    "knee_flexion_l", "knee_flexion_r",
    "hip_flexion_l", "hip_flexion_r",
    "trunk_flexion",
    "ankle_dorsiflexion_l", "ankle_dorsiflexion_r",
    "knee_valgus_l", "knee_valgus_r",
    "hip_adduction_l", "hip_adduction_r",
    "hip_rotation_l", "hip_rotation_r",
    "pelvis_tilt",
]


def _round(value: float | None, digits: int = 2) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _stage_label(stage_name: str) -> str:
    return stage_name.replace("_", " ").title()


def _flatten_kpts(kpts: np.ndarray | None) -> list[float] | None:
    """(N, 3+) numpy array to flat [x0, y0, z0, x1, y1, z1, ...] list."""
    if kpts is None:
        return None
    arr = np.asarray(kpts, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[:, :3]
    return [round(float(v), 4) for v in arr.ravel()]


class PipelineInspector:
    """Records every pipeline stage per frame and writes an HTML debug report."""

    def __init__(
        self,
        out_dir: str,
        nominal_fps: float,
        multi_camera: bool,
        stage_names: tuple[str, ...],
    ) -> None:
        self._dir = Path(out_dir) / "pipeline_inspect"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._nominal_fps = max(1.0, float(nominal_fps))
        self._multi_camera = multi_camera
        self._stage_names = tuple(stage_names)
        self._started_at = datetime.now()
        self._clock_start = time.perf_counter()

        self._series: dict[str, list] = {
            "t": [], "fi": [], "phase": [], "rep": [],
            "ready": [], "resting": [], "standing_gate": [],
        }

        self._skeletons: dict[str, list] = {name: [] for name in self._stage_names}
        self._angles_raw: dict[str, list] = {key: [] for key in ANGLE_KEYS}
        self._angles_filtered: dict[str, list] = {key: [] for key in ANGLE_KEYS}

        self._faults: list[dict] = []
        self._rep_events: list[dict] = []

        self._encoder: subprocess.Popen | None = None
        self._writer: cv2.VideoWriter | None = None
        self._video_path = self._dir / "video.mp4"
        self._size: tuple[int, int] | None = None
        self._encode_failed = False

    @property
    def frame_count(self) -> int:
        return len(self._series["t"])

    @property
    def path(self) -> Path:
        return self._dir

    def record_frame(
        self,
        result: PipelineFrame,
        display: np.ndarray,
        pipeline: Any,
        resting: bool,
    ) -> None:
        index = self.frame_count
        self._write_video_frame(display)

        counter = pipeline.rep_counter
        phase = getattr(counter.phase, "value", str(counter.phase))

        self._series["t"].append(round(time.perf_counter() - self._clock_start, 3))
        self._series["fi"].append(int(result.frame_index))
        self._series["phase"].append(phase)
        self._series["rep"].append(int(counter.rep_count))
        self._series["ready"].append(bool(pipeline.is_ready))
        self._series["resting"].append(bool(resting))
        self._series["standing_gate"].append(bool(pipeline._standing_gate.is_ready))

        intermediates = getattr(pipeline, "_inspect_intermediates", None) or {}
        for stage in self._stage_names:
            self._skeletons[stage].append(_flatten_kpts(intermediates.get(stage)))

        raw_angles = getattr(pipeline, "_inspect_raw_angles", None)
        filtered_angles = result.joint_angles

        for key in ANGLE_KEYS:
            self._angles_raw[key].append(
                _round(getattr(raw_angles, key, None)) if raw_angles else None
            )
            self._angles_filtered[key].append(
                _round(getattr(filtered_angles, key, None)) if filtered_angles else None
            )

        for fault in result.faults:
            self._faults.append({
                "i": index,
                "type": fault.fault_type,
                "severity": getattr(fault.severity, "value", str(fault.severity)),
                "message": fault.message,
            })

        if result.rep_data is not None:
            self._rep_events.append({
                "i": index,
                "rep": int(result.rep_data.rep_number),
                "depth": round(result.rep_data.max_depth_angle, 1),
            })

    # ------------------------------------------------------------------
    # Video encoding (mirrors valgus_debug.py)
    # ------------------------------------------------------------------

    def _write_video_frame(self, display: np.ndarray) -> None:
        if self._encode_failed:
            return
        frame = self._scale(display)
        if self._encoder is None and self._writer is None:
            self._open_encoder(frame.shape[1], frame.shape[0])
        try:
            if self._encoder is not None and self._encoder.stdin is not None:
                self._encoder.stdin.write(frame.tobytes())
            elif self._writer is not None:
                self._writer.write(frame)
        except (BrokenPipeError, OSError) as exc:
            print(f"[INSPECT] Video encoding stopped: {exc}")
            self._encode_failed = True

    def _scale(self, display: np.ndarray) -> np.ndarray:
        if self._size is None:
            height, width = display.shape[:2]
            if width > MAX_VIDEO_WIDTH:
                scale = MAX_VIDEO_WIDTH / width
                width, height = int(width * scale), int(height * scale)
            self._size = (width - width % 2, height - height % 2)
        if display.shape[1::-1] != self._size:
            return cv2.resize(display, self._size, interpolation=cv2.INTER_AREA)
        return display

    def _open_encoder(self, width: int, height: int) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is not None:
            self._encoder = subprocess.Popen(
                [
                    ffmpeg, "-y", "-loglevel", "error",
                    "-f", "rawvideo", "-pix_fmt", "bgr24",
                    "-s", f"{width}x{height}", "-r", f"{self._nominal_fps:.3f}",
                    "-i", "-", "-an",
                    "-c:v", "libx264", "-preset", "veryfast",
                    "-pix_fmt", "yuv420p", "-crf", VIDEO_CRF,
                    "-movflags", "+faststart", str(self._video_path),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
        self._writer = cv2.VideoWriter(
            str(self._video_path),
            cv2.VideoWriter_fourcc(*"avc1"),
            self._nominal_fps,
            (width, height),
        )
        if not self._writer.isOpened():
            print("[INSPECT] No ffmpeg and no avc1 writer — video will be omitted")
            self._writer = None
            self._encode_failed = True

    def _close_encoder(self) -> None:
        if self._encoder is not None:
            try:
                if self._encoder.stdin is not None:
                    self._encoder.stdin.close()
                self._encoder.wait(timeout=60)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                self._encoder.kill()
            self._encoder = None
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def _measured_fps(self) -> float:
        times = self._series["t"]
        if len(times) < 2 or times[-1] <= times[0]:
            return self._nominal_fps
        return (len(times) - 1) / (times[-1] - times[0])

    def _retime_video(self, measured_fps: float) -> float:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None or not self._video_path.exists():
            return self._nominal_fps
        if not MIN_RETIME_FPS <= measured_fps <= MAX_RETIME_FPS:
            return self._nominal_fps
        if abs(measured_fps - self._nominal_fps) / self._nominal_fps < 0.05:
            return self._nominal_fps
        retimed = self._dir / "video_retimed.mp4"
        try:
            subprocess.run(
                [
                    ffmpeg, "-y", "-loglevel", "error",
                    "-r", f"{measured_fps:.3f}", "-i", str(self._video_path),
                    "-c", "copy", str(retimed),
                ],
                check=True,
                capture_output=True,
                timeout=120,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            retimed.unlink(missing_ok=True)
            return self._nominal_fps
        retimed.replace(self._video_path)
        return measured_fps

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def finalize(self, exercise_name: str) -> Path | None:
        self._close_encoder()
        if self.frame_count == 0:
            print("[INSPECT] No frames recorded — skipping report")
            return None

        measured_fps = self._measured_fps()
        video_fps = self._retime_video(measured_fps)

        data = {
            "session": self._started_at.strftime("%Y%m%d_%H%M%S"),
            "recorded_at": self._started_at.isoformat(timespec="seconds"),
            "exercise": exercise_name,
            "multi_camera": self._multi_camera,
            "n_frames": self.frame_count,
            "video_fps": round(video_fps, 3),
            "measured_fps": round(measured_fps, 2),
            "duration_s": round(self._series["t"][-1], 2),
            "has_video": self._video_path.exists() and not self._encode_failed,
            "stage_names": list(self._stage_names),
            "stage_labels": [_stage_label(name) for name in self._stage_names],
            "series": self._series,
            "skeletons": self._skeletons,
            "angle_keys": ANGLE_KEYS,
            "angles_raw": self._angles_raw,
            "angles_filtered": self._angles_filtered,
            "faults": self._faults,
            "reps": self._rep_events,
        }
        data = nan_to_none(data)
        (self._dir / "data.json").write_text(json.dumps(data))

        video_b64 = ""
        if data["has_video"]:
            video_b64 = base64.b64encode(
                self._video_path.read_bytes()
            ).decode("ascii")

        html = TEMPLATE_PATH.read_text()
        html = html.replace("__TITLE__", f"Pipeline Inspector — {data['session']}")
        html = html.replace("__VIDEO_B64__", video_b64)
        html = html.replace("__DATA_JSON__", json.dumps(data, separators=(",", ":")))
        report_path = self._dir / "pipeline_inspector.html"
        report_path.write_text(html)

        n_skel = sum(1 for v in self._skeletons[self._stage_names[0]] if v is not None)
        print(
            f"\n[INSPECT] {self.frame_count} frames, "
            f"{n_skel} with skeleton data, "
            f"{len(self._faults)} faults, "
            f"{len(self._rep_events)} reps"
        )
        print(f"[INSPECT] Report: {report_path}")
        return report_path


def build_inspector(
    out_dir: str, nominal_fps: float, multi_camera: bool, stage_names: tuple[str, ...],
) -> PipelineInspector | None:
    """Create an inspector when --inspect / NOWVA_PIPELINE_INSPECT is active."""
    if os.getenv("NOWVA_PIPELINE_INSPECT", "").lower() not in ("1", "true", "yes"):
        return None
    inspector = PipelineInspector(out_dir, nominal_fps, multi_camera, stage_names)
    print(f"[INSPECT] Pipeline inspection enabled → {inspector.path}")
    return inspector
