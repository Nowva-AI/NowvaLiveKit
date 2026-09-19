"""SessionProfiler — thread-safe event and resource collection for live sessions."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from profiler.resource_sampler import ResourceSampler

PROFILE_OUTPUT_DIR = Path(__file__).parent.parent.parent / "profiler_results"
AGENT_JSON_FILENAME = "session_agent.json"


def _is_enabled() -> bool:
    return os.getenv("NOWVA_PROFILE") == "1"


@dataclass
class TurnBuilder:
    """Accumulates metrics for a single conversation turn."""
    turn_number: int = 0
    start_s: float = 0.0
    user_speech_start_s: float | None = None
    user_speech_end_s: float | None = None
    transcript: str = ""
    llm_ttft_s: float | None = None
    llm_duration_s: float | None = None
    llm_input_tokens: int | None = None
    llm_output_tokens: int | None = None
    llm_tokens_per_second: float | None = None
    tts_duration_s: float | None = None
    tts_audio_duration_s: float | None = None
    ttfa_s: float | None = None
    e2e_s: float | None = None
    affect_wait_ms: float | None = None
    affect_fresh: bool | None = None
    affect_infer_ms: float | None = None
    cancelled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class SessionProfiler:
    """Singleton that collects timestamped events during a live Nowva session.

    All public methods are no-ops when NOWVA_PROFILE != "1".
    """

    _instance: SessionProfiler | None = None
    _lock_cls = threading.Lock()

    def __new__(cls) -> SessionProfiler:
        if cls._instance is None:
            with cls._lock_cls:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self._lock = threading.Lock()
        self._session_start: float = 0.0
        self._events: list[dict[str, Any]] = []
        self._resource_samples: list[dict[str, Any]] = []
        self._turns: list[dict[str, Any]] = []
        self._current_turn: TurnBuilder | None = None
        self._turn_counter = 0
        self._user_speech_end_ts: float | None = None
        self._sampler: ResourceSampler | None = None
        self._started = False

    @classmethod
    def get_instance(cls) -> SessionProfiler:
        return cls()

    def start(self) -> None:
        if not _is_enabled() or self._started:
            return
        self._session_start = time.time()
        self._started = True
        self._sampler = ResourceSampler(callback=self._on_resource_sample, interval_s=1.0)
        self._sampler.start()

    def _elapsed(self) -> float:
        return time.time() - self._session_start if self._session_start else 0.0

    # -- Generic event recording --

    def record(self, category: str, event_type: str, **data: Any) -> None:
        if not _is_enabled() or not self._started:
            return
        event = {
            "timestamp": time.time(),
            "elapsed_s": round(self._elapsed(), 3),
            "category": category,
            "event_type": event_type,
            **data,
        }
        with self._lock:
            self._events.append(event)

    # -- Turn tracking --

    def begin_turn(self) -> None:
        """Start a new conversation turn."""
        if not _is_enabled() or not self._started:
            return
        with self._lock:
            if self._current_turn:
                self._turns.append(self._current_turn.to_dict())
            self._turn_counter += 1
            self._current_turn = TurnBuilder(
                turn_number=self._turn_counter,
                start_s=round(self._elapsed(), 3),
            )

    def record_user_speech_start(self) -> None:
        if not _is_enabled() or not self._started:
            return
        with self._lock:
            if self._current_turn:
                self._current_turn.user_speech_start_s = round(self._elapsed(), 3)

    def record_user_speech_end(self) -> None:
        if not _is_enabled() or not self._started:
            return
        self._user_speech_end_ts = time.time()
        with self._lock:
            if self._current_turn:
                self._current_turn.user_speech_end_s = round(self._elapsed(), 3)

    def record_transcript(self, text: str) -> None:
        if not _is_enabled() or not self._started:
            return
        with self._lock:
            if self._current_turn:
                self._current_turn.transcript = text

    def record_speech_created(self) -> None:
        """Record when TTS audio is first scheduled (for TTFA computation)."""
        if not _is_enabled() or not self._started:
            return
        now = time.time()
        with self._lock:
            if self._current_turn and self._user_speech_end_ts:
                self._current_turn.ttfa_s = round(now - self._user_speech_end_ts, 3)

    def record_llm_metrics(
        self,
        ttft: float,
        duration: float,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        tokens_per_second: float | None = None,
        cancelled: bool = False,
    ) -> None:
        if not _is_enabled() or not self._started:
            return
        with self._lock:
            if not self._current_turn:
                self._turn_counter += 1
                self._current_turn = TurnBuilder(
                    turn_number=self._turn_counter,
                    start_s=round(self._elapsed(), 3),
                )
            self._current_turn.llm_ttft_s = round(ttft, 4)
            self._current_turn.llm_duration_s = round(duration, 4)
            self._current_turn.llm_input_tokens = input_tokens
            self._current_turn.llm_output_tokens = output_tokens
            self._current_turn.llm_tokens_per_second = (
                round(tokens_per_second, 1) if tokens_per_second else None
            )
            self._current_turn.cancelled = cancelled

            if self._current_turn.user_speech_end_s is not None:
                self._current_turn.e2e_s = round(
                    self._elapsed() - self._current_turn.user_speech_end_s, 3
                )

    def record_tts_metrics(self, duration: float, audio_duration: float) -> None:
        if not _is_enabled() or not self._started:
            return
        with self._lock:
            if self._current_turn:
                self._current_turn.tts_duration_s = round(duration, 4)
                self._current_turn.tts_audio_duration_s = round(audio_duration, 4)

    def record_affect_wait(self, wait_ms: float, fresh: bool) -> None:
        if not _is_enabled() or not self._started:
            return
        with self._lock:
            if self._current_turn:
                self._current_turn.affect_wait_ms = round(wait_ms, 2)
                self._current_turn.affect_fresh = bool(fresh)

    def record_affect_infer(self, infer_ms: float) -> None:
        if not _is_enabled() or not self._started:
            return
        with self._lock:
            if self._current_turn:
                self._current_turn.affect_infer_ms = round(infer_ms, 2)

    def finalize_turn(self) -> None:
        """Flush current turn to the turns list."""
        if not _is_enabled() or not self._started:
            return
        with self._lock:
            if self._current_turn:
                self._turns.append(self._current_turn.to_dict())
                self._current_turn = None
                self._user_speech_end_ts = None

    # -- Resource samples --

    def _on_resource_sample(self, stats: dict[str, Any]) -> None:
        sample = {"elapsed_s": round(self._elapsed(), 1), **stats}
        with self._lock:
            self._resource_samples.append(sample)

    # -- System info --

    @staticmethod
    def _system_info() -> dict[str, Any]:
        info: dict[str, Any] = {
            "platform": sys.platform,
            "python": platform.python_version(),
        }
        try:
            info["cpu"] = platform.processor() or platform.machine()
        except Exception:
            info["cpu"] = "unknown"
        try:
            import psutil
            info["ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 1)
        except Exception:
            info["ram_gb"] = None
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=5,
            )
            info["git_commit"] = result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            info["git_commit"] = None
        return info

    # -- Flush / stop --

    def _build_output(self) -> dict[str, Any]:
        with self._lock:
            if self._current_turn:
                self._turns.append(self._current_turn.to_dict())
                self._current_turn = None

            events = list(self._events)
            turns = list(self._turns)
            samples = list(self._resource_samples)

        now = time.time()
        duration = now - self._session_start if self._session_start else 0.0

        mode_transitions = [e for e in events if e.get("category") == "agent" and e.get("event_type") == "mode_change"]
        errors = [e for e in events if e.get("category") == "error"]
        coaching = [e for e in events if e.get("category") == "coaching"]
        ipc = [e for e in events if e.get("category") == "ipc"]
        tools = [e for e in events if e.get("category") == "tool"]
        compaction = [e for e in events if e.get("category") == "compaction"]
        affect = [e for e in events if e.get("category") == "affect"]

        affect_fresh_flags = [t["affect_fresh"] for t in turns if t.get("affect_fresh") is not None]
        affect_waits = [t["affect_wait_ms"] for t in turns if t.get("affect_wait_ms") is not None]

        ttfts = [t["llm_ttft_s"] for t in turns if t.get("llm_ttft_s")]
        ttfas = [t["ttfa_s"] for t in turns if t.get("ttfa_s")]
        e2es = [t["e2e_s"] for t in turns if t.get("e2e_s")]

        total_reps = sum(1 for e in coaching if e.get("event_type") == "rep_complete")
        total_faults = sum(1 for e in coaching if e.get("event_type") == "fault")

        return {
            "session_start": datetime.fromtimestamp(self._session_start).isoformat() if self._session_start else None,
            "session_end": datetime.fromtimestamp(now).isoformat(),
            "duration_s": round(duration, 1),
            "system": self._system_info(),
            "summary": {
                "total_turns": len(turns),
                "total_errors": len(errors),
                "avg_ttft_s": round(sum(ttfts) / len(ttfts), 3) if ttfts else None,
                "avg_ttfa_s": round(sum(ttfas) / len(ttfas), 3) if ttfas else None,
                "avg_e2e_s": round(sum(e2es) / len(e2es), 3) if e2es else None,
                "modes_visited": list(dict.fromkeys(
                    e.get("to_mode", "") for e in mode_transitions
                )),
                "total_reps": total_reps,
                "total_faults": total_faults,
                "affect_utterances": len(affect),
                "affect_fresh_rate": round(sum(affect_fresh_flags) / len(affect_fresh_flags), 3) if affect_fresh_flags else None,
                "affect_avg_wait_ms": round(sum(affect_waits) / len(affect_waits), 2) if affect_waits else None,
            },
            "turns": turns,
            "mode_transitions": [
                {"elapsed_s": e["elapsed_s"], "from": e.get("from_mode"), "to": e.get("to_mode")}
                for e in mode_transitions
            ],
            "errors": [
                {"elapsed_s": e["elapsed_s"], "source": e.get("source", ""), "error": e.get("error", "")}
                for e in errors
            ],
            "coaching_events": [
                {k: v for k, v in e.items() if k not in ("timestamp", "category")}
                for e in coaching
            ],
            "ipc_messages": [
                {k: v for k, v in e.items() if k not in ("timestamp", "category")}
                for e in ipc
            ],
            "tool_calls": [
                {k: v for k, v in e.items() if k not in ("timestamp", "category")}
                for e in tools
            ],
            "compaction_cycles": [
                {k: v for k, v in e.items() if k not in ("timestamp", "category")}
                for e in compaction
            ],
            "affect_events": [
                {k: v for k, v in e.items() if k not in ("timestamp", "category")}
                for e in affect
            ],
            "resource_samples": samples,
        }

    def flush(self, path: Path | None = None) -> Path | None:
        if not _is_enabled() or not self._started:
            return None
        if path is None:
            PROFILE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            path = PROFILE_OUTPUT_DIR / AGENT_JSON_FILENAME
        else:
            path.parent.mkdir(parents=True, exist_ok=True)

        data = self._build_output()
        path.write_text(json.dumps(data, indent=2))
        return path

    def stop(self) -> Path | None:
        if not _is_enabled() or not self._started:
            return None
        if self._sampler:
            self._sampler.stop()
        path = self.flush()
        self._started = False
        return path

    def reset(self) -> None:
        """Reset all state for a fresh session."""
        with self._lock:
            self._events.clear()
            self._resource_samples.clear()
            self._turns.clear()
            self._current_turn = None
            self._turn_counter = 0
            self._user_speech_end_ts = None
        self._session_start = 0.0
        self._started = False
