"""Knee-valgus debug recorder.

Records the annotated preview window, every per-frame knee metric, all outgoing
IPC traffic, and the live fault thresholds during a run, then builds a
self-contained HTML report with a scrubbable video synced to the data.
Enabled only by the --valgus flag; inert otherwise.
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

from biomechanics.faults.rules.knee_valgus import (
    FOOT_CONFIDENCE_THRESHOLD,
    KneeValgusRule,
)
from biomechanics.utils.json_safe import nan_to_none
from biomechanics.utils.types import FaultEvent, PipelineFrame

TEMPLATE_PATH = Path(__file__).with_name("valgus_debug_template.html")

# Preview frames are downscaled to this width before encoding — the video is
# base64-embedded in the report, so it has to stay small enough to parse.
MAX_VIDEO_WIDTH = 960
VIDEO_CRF = "26"
# Anthropometric scaling rewrites thresholds mid-run, so a single snapshot at
# the end would misrepresent earlier reps. Re-read this often and store changes.
THRESHOLD_SAMPLE_INTERVAL = 30
# Backstop so a very long session cannot grow the report past what a browser
# can comfortably parse. Overflow is reported in the page header.
MAX_IPC_MESSAGES = 20000
# Plausible bounds for a real-time capture rate. A measured rate outside this
# means the loop was not running in real time, so the video keeps its declared
# rate rather than being retimed to something a browser cannot play.
MIN_RETIME_FPS = 1.0
MAX_RETIME_FPS = 120.0
# Substrings that mark an IPC message as carrying knee data.
KNEE_TOKENS = ("valgus", "knee", "adduction")
# The rule reports at most one fault per this many frames (KneeValgusRule).
FAULT_COOLDOWN_FRAMES = 30

_SERIES_KEYS = (
    "t", "fi", "vl", "vr", "al", "ar", "fcl", "fcr",
    "kasr", "kfl", "kfr", "hrl", "hrr", "phase", "rep", "ready", "resting",
)


def _read_valgus_thresholds(rule_engine: Any) -> dict[str, dict[str, float]] | None:
    """Live primary + fallback valgus thresholds, post anthropometric scaling."""
    for rule in rule_engine.rules:
        if isinstance(rule, KneeValgusRule):
            return {
                "primary": {
                    "mild": round(float(rule.mild_threshold), 3),
                    "moderate": round(float(rule.moderate_threshold), 3),
                    "severe": round(float(rule.severe_threshold), 3),
                },
                "fallback": {
                    "mild": round(float(rule.fallback_mild_threshold), 3),
                    "moderate": round(float(rule.fallback_moderate_threshold), 3),
                    "severe": round(float(rule.fallback_severe_threshold), 3),
                },
            }
    return None


def _is_knee_related(message: dict[str, Any]) -> bool:
    try:
        blob = json.dumps(message, default=str).lower()
    except (TypeError, ValueError):
        return False
    return any(token in blob for token in KNEE_TOKENS)


def _round(value: float | None, digits: int = 2) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


class _IPCTap:
    """Transparent proxy that records every message sent through an IPC client."""

    def __init__(self, client: Any, recorder: ValgusDebugRecorder) -> None:
        self._client = client
        self._recorder = recorder

    def send_message(self, message: dict[str, Any]) -> None:
        self._recorder.record_ipc(message)
        self._client.send_message(message)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class ValgusDebugRecorder:
    """Collects knee data + preview video during a run and writes an HTML report."""

    def __init__(self, out_dir: str, nominal_fps: float, multi_camera: bool) -> None:
        self._dir = Path(out_dir) / "valgus_debug"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._nominal_fps = max(1.0, float(nominal_fps))
        self._multi_camera = multi_camera
        self._started_at = datetime.now()
        self._clock_start = time.perf_counter()

        self._series: dict[str, list] = {key: [] for key in _SERIES_KEYS}
        self._faults: list[dict] = []
        self._ipc: list[dict] = []
        self._ipc_overflow = 0
        self._threshold_history: list[dict] = []
        self._rep_events: list[dict] = []
        self._last_thresholds: dict | None = None

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

    def tap(self, ipc_client: Any) -> Any:
        return _IPCTap(ipc_client, self)

    def record_ipc(self, message: dict[str, Any]) -> None:
        if len(self._ipc) >= MAX_IPC_MESSAGES:
            self._ipc_overflow += 1
            return
        self._ipc.append({
            "i": self.frame_count,
            "t": round(time.perf_counter() - self._clock_start, 3),
            "type": str(message.get("type", "?")),
            "knee": _is_knee_related(message),
            "msg": message,
        })

    def record_frame(
        self,
        result: PipelineFrame,
        display: np.ndarray,
        pipeline: Any,
        resting: bool,
    ) -> None:
        index = self.frame_count
        self._write_video_frame(display)

        angles = result.joint_angles
        counter = pipeline.rep_counter
        phase = getattr(counter.phase, "value", str(counter.phase))

        self._series["t"].append(round(time.perf_counter() - self._clock_start, 3))
        self._series["fi"].append(int(result.frame_index))
        self._series["phase"].append(phase)
        self._series["rep"].append(int(counter.rep_count))
        self._series["ready"].append(bool(pipeline.is_ready))
        self._series["resting"].append(bool(resting))

        pairs = (
            ("vl", "knee_valgus_l"), ("vr", "knee_valgus_r"),
            ("al", "hip_adduction_l"), ("ar", "hip_adduction_r"),
            ("fcl", "foot_confidence_l"), ("fcr", "foot_confidence_r"),
            ("kasr", "knee_ankle_sep_ratio"),
            ("kfl", "knee_flexion_l"), ("kfr", "knee_flexion_r"),
            ("hrl", "hip_rotation_l"), ("hrr", "hip_rotation_r"),
        )
        for key, field in pairs:
            value = getattr(angles, field, None) if angles is not None else None
            self._series[key].append(_round(value))

        for fault in result.faults:
            self._record_fault(fault, index)

        if result.rep_data is not None:
            self._rep_events.append({
                "i": index,
                "rep": int(result.rep_data.rep_number),
            })

        if index % THRESHOLD_SAMPLE_INTERVAL == 0:
            self._sample_thresholds(pipeline, index)

    def _record_fault(self, fault: FaultEvent, index: int) -> None:
        self._faults.append({
            "i": index,
            "fault_type": fault.fault_type,
            "knee": "valgus" in fault.fault_type or "knee" in fault.fault_type,
            "severity": getattr(fault.severity, "value", str(fault.severity)),
            "severity_score": _round(fault.severity_score),
            "message": fault.message,
            "rep": int(fault.rep_number),
            "frame_index": int(fault.frame_index),
            "details": {
                key: (_round(value) if isinstance(value, (int, float)) else value)
                for key, value in fault.details.items()
            },
        })

    def _sample_thresholds(self, pipeline: Any, index: int) -> None:
        engine = getattr(pipeline, "_rule_engine", None)
        if engine is None:
            return
        thresholds = _read_valgus_thresholds(engine)
        if thresholds is None or thresholds == self._last_thresholds:
            return
        self._last_thresholds = thresholds
        self._threshold_history.append({"i": index, **thresholds})

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
            print(f"[VALGUS] Video encoding stopped: {exc}")
            self._encode_failed = True

    def _scale(self, display: np.ndarray) -> np.ndarray:
        if self._size is None:
            height, width = display.shape[:2]
            if width > MAX_VIDEO_WIDTH:
                scale = MAX_VIDEO_WIDTH / width
                width, height = int(width * scale), int(height * scale)
            # libx264 with yuv420p needs even dimensions.
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
            print("[VALGUS] No ffmpeg and no avc1 writer — video will be omitted")
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
        """Stream-copy the video to the measured rate so playback runs at real speed."""
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None or not self._video_path.exists():
            return self._nominal_fps
        # Outside a plausible capture rate the measurement is not real-time
        # (an unthrottled or stalled loop); the declared rate plays better.
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

    def finalize(self, exercise_name: str) -> Path | None:
        self._close_encoder()
        if self.frame_count == 0:
            print("[VALGUS] No frames recorded — skipping report")
            return None

        measured_fps = self._measured_fps()
        video_fps = self._retime_video(measured_fps)

        data = {
            "session": self._started_at.strftime("%Y%m%d_%H%M%S"),
            "recorded_at": self._started_at.isoformat(timespec="seconds"),
            "exercise": exercise_name,
            "mode": "3D abduction (triangulated)" if self._multi_camera
                    else "2D FPPA (single camera)",
            "multi_camera": self._multi_camera,
            "n_frames": self.frame_count,
            "video_fps": round(video_fps, 3),
            "measured_fps": round(measured_fps, 2),
            "duration_s": round(self._series["t"][-1], 2),
            "has_video": self._video_path.exists() and not self._encode_failed,
            "foot_confidence_threshold": FOOT_CONFIDENCE_THRESHOLD,
            "fault_cooldown_frames": FAULT_COOLDOWN_FRAMES,
            "series": self._series,
            "faults": self._faults,
            "reps": self._rep_events,
            "thresholds": self._threshold_history,
            "ipc": self._ipc,
            "ipc_overflow": self._ipc_overflow,
        }
        data = nan_to_none(data)
        (self._dir / "data.json").write_text(json.dumps(data))

        video_b64 = ""
        if data["has_video"]:
            video_b64 = base64.b64encode(self._video_path.read_bytes()).decode("ascii")

        html = TEMPLATE_PATH.read_text()
        html = html.replace("__TITLE__", f"Knee Valgus Debug — {data['session']}")
        html = html.replace("__VIDEO_B64__", video_b64)
        html = html.replace("__DATA_JSON__", json.dumps(data, separators=(",", ":")))
        report_path = self._dir / "valgus_debug.html"
        report_path.write_text(html)

        knee_faults = sum(1 for fault in self._faults if fault["knee"])
        print(f"\n[VALGUS] {self.frame_count} frames, {knee_faults} knee faults, "
              f"{len(self._ipc)} IPC messages")
        print(f"[VALGUS] Report: {report_path}")
        return report_path


def build_recorder(out_dir: str, nominal_fps: float, multi_camera: bool) -> ValgusDebugRecorder | None:
    """Create a recorder when --valgus / NOWVA_VALGUS_DEBUG is active."""
    if os.getenv("NOWVA_VALGUS_DEBUG", "").lower() not in ("1", "true", "yes"):
        return None
    recorder = ValgusDebugRecorder(out_dir, nominal_fps, multi_camera)
    print(f"[VALGUS] Debug recording enabled → {recorder.path}")
    return recorder
