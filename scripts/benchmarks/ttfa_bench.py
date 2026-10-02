#!/usr/bin/env python3
"""Measure time-to-first-audio (TTFA) for the Nova cascade voice pipeline.

Drives the real AgentSession built by agent.core.pipeline_factory with a
pre-recorded user utterance fed through a synthetic AudioInput at real-time
pace, and timestamps the first audio frame the pipeline pushes to its
AudioOutput.

TTFA is measured from the instant the last sample of user speech would have
reached the microphone to the instant the first agent audio frame is handed to
the output sink. It excludes WebRTC transport and the client jitter buffer
(see --transport-ms to fold in a fixed estimate).

Usage:
    python scripts/benchmarks/ttfa_bench.py --trials 5
    python scripts/benchmarks/ttfa_bench.py --utterances start_workout,progress
    ENDPOINTING_MIN_DELAY=0.15 python scripts/benchmarks/ttfa_bench.py --label fast-endpoint
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import statistics
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

load_dotenv(PROJECT_ROOT / ".env")

from livekit import rtc  # noqa: E402
from livekit.agents import AgentSession  # noqa: E402
from livekit.agents.utils import http_context  # noqa: E402
from livekit.agents.voice.io import AudioInput, AudioOutput, AudioOutputCapabilities  # noqa: E402

FRAME_MS = 20
BENCH_SAMPLE_RATE = 24000
LEAD_SILENCE_S = 0.6
TAIL_SILENCE_S = 6.0
DEFAULT_TRIALS = 5
DEFAULT_TRANSPORT_MS = 0.0
TARGET_TTFA_MS = 500.0
BENCH_USER_ID = "00000000-0000-4000-8000-0000000000bc"

FIXTURES_DIR = PROJECT_ROOT / "benchmarks" / "fixtures" / "voice"
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results" / "ttfa"

logger = logging.getLogger("ttfa_bench")


# --- in-process EOU inference (production runs this over IPC in a subprocess) ---


class _LocalInferenceExecutor:
    def __init__(self) -> None:
        self._runners: dict[str, Any] = {}

    def warmup(self, method: str) -> None:
        self._get(method)

    def _get(self, method: str):
        runner = self._runners.get(method)
        if runner is None:
            from livekit.agents.inference_runner import _InferenceRunner

            runner_cls = _InferenceRunner.registered_runners[method]
            runner = runner_cls()
            runner.initialize()
            self._runners[method] = runner
        return runner

    async def do_inference(self, method: str, data: bytes) -> bytes | None:
        runner = self._get(method)
        return await asyncio.get_running_loop().run_in_executor(None, runner.run, data)


class _StubJobContext:
    def __init__(self, executor: _LocalInferenceExecutor) -> None:
        self.inference_executor = executor
        self.job = None
        self._primary_agent_session = None

    def init_recording(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def _install_stub_job_context(executor: _LocalInferenceExecutor) -> None:
    """Let the turn-detector plugin construct outside a LiveKit job."""
    import livekit.plugins.turn_detector.base as td_base

    stub = _StubJobContext(executor)
    td_base.get_job_context = lambda *a, **k: stub  # type: ignore[assignment]


# --- synthetic audio I/O ---


@dataclass
class _Utterance:
    name: str
    path: Path
    samples: np.ndarray
    sample_rate: int

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.sample_rate


class BenchAudioInput(AudioInput):
    """Streams silence, then one utterance at real-time pace, then silence."""

    def __init__(self, sample_rate: int = BENCH_SAMPLE_RATE) -> None:
        super().__init__(label="BenchAudioInput")
        self._sample_rate = sample_rate
        self._samples_per_frame = sample_rate * FRAME_MS // 1000
        self._queue: asyncio.Queue[rtc.AudioFrame] = asyncio.Queue()
        self._pending: np.ndarray = np.zeros(0, dtype=np.int16)
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None
        self._emitted_samples = 0
        self._t0: float | None = None
        self._last_emit_ts: float = 0.0
        self.max_drift_s: float = 0.0

    def start(self) -> None:
        self._task = asyncio.create_task(self._pump())

    async def aclose(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    def schedule(self, samples: np.ndarray) -> int:
        """Queue samples after a one-frame lead; return their last absolute index."""
        lead = self._samples_per_frame
        start = self._emitted_samples + len(self._pending) + lead
        self._pending = np.concatenate(
            [self._pending, np.zeros(lead, dtype=np.int16), samples]
        )
        return start + len(samples)

    async def wall_time_of(self, sample_index: int) -> float:
        """Wall clock at which the frame carrying `sample_index` actually left the pump.

        The pump paces itself but the event loop is shared with STT, the LLM,
        TTS and EOU inference, so emission drifts behind the ideal schedule.
        Deriving end-of-speech from the schedule instead of the real emission
        inflates every downstream measurement by the accumulated drift.
        """
        while self._emitted_samples < sample_index:
            await asyncio.sleep(0.002)
        return self._last_emit_ts

    def reset_drift(self) -> None:
        self.max_drift_s = 0.0

    async def _pump(self) -> None:
        """Emit exactly one frame every FRAME_MS of wall clock."""
        self._t0 = time.monotonic()
        next_deadline = self._t0
        while True:
            if len(self._pending) >= self._samples_per_frame:
                chunk = self._pending[: self._samples_per_frame]
                self._pending = self._pending[self._samples_per_frame :]
            else:
                chunk = np.zeros(self._samples_per_frame, dtype=np.int16)
            frame = rtc.AudioFrame(
                data=chunk.tobytes(),
                sample_rate=self._sample_rate,
                num_channels=1,
                samples_per_channel=self._samples_per_frame,
            )
            await self._queue.put(frame)
            now = time.monotonic()
            self._emitted_samples += self._samples_per_frame
            self._last_emit_ts = now
            next_deadline += FRAME_MS / 1000
            sleep_for = next_deadline - now
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            else:
                # Behind schedule: catch up rather than silently resetting the
                # clock, and remember how far behind we fell.
                self.max_drift_s = max(self.max_drift_s, -sleep_for)

    async def __anext__(self) -> rtc.AudioFrame:
        return await self._queue.get()


class BenchAudioOutput(AudioOutput):
    """Accepts agent audio, timestamps the first frame of each segment."""

    def __init__(self, sample_rate: int = BENCH_SAMPLE_RATE) -> None:
        super().__init__(
            label="BenchAudioOutput",
            capabilities=AudioOutputCapabilities(pause=False),
            sample_rate=sample_rate,
        )
        self.first_frame_ts: float | None = None
        self.last_frame_ts: float = 0.0
        self.segment_starts: list[float] = []
        self.first_frame_event = asyncio.Event()
        self._pushed_s = 0.0
        self._capturing = False
        self.captured_audio_s = 0.0

    def arm(self) -> None:
        self.first_frame_ts = None
        self.first_frame_event = asyncio.Event()
        self.segment_starts = []
        self.captured_audio_s = 0.0

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        await super().capture_frame(frame)
        self.last_frame_ts = time.monotonic()
        if self.first_frame_ts is None:
            self.first_frame_ts = self.last_frame_ts
            self.first_frame_event.set()
        if not self._capturing:
            self._capturing = True
            self._pushed_s = 0.0
            self.segment_starts.append(self.last_frame_ts)
        self._pushed_s += frame.duration
        self.captured_audio_s += frame.duration

    def flush(self) -> None:
        super().flush()
        self._capturing = False
        # Report playout immediately: the bench never plays audio back, and the
        # session must not stall waiting on a speaker that does not exist.
        self.on_playback_finished(playback_position=self._pushed_s, interrupted=False)
        self._pushed_s = 0.0

    def clear_buffer(self) -> None:
        self._capturing = False
        self._pushed_s = 0.0


# --- fixtures ---


def _load_wav(path: Path) -> _Utterance:
    with wave.open(str(path), "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        raw = wav.readframes(wav.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)
    if sample_rate != BENCH_SAMPLE_RATE:
        target_len = int(len(samples) * BENCH_SAMPLE_RATE / sample_rate)
        samples = np.interp(
            np.linspace(0, len(samples) - 1, target_len),
            np.arange(len(samples)),
            samples.astype(np.float32),
        ).astype(np.int16)
    return _Utterance(name=path.stem, path=path, samples=samples, sample_rate=BENCH_SAMPLE_RATE)


def _trim_trailing_silence(utterance: _Utterance, threshold: int = 400) -> _Utterance:
    """Cut trailing digital silence so speech-end is the real end of speech."""
    magnitude = np.abs(utterance.samples)
    voiced = np.nonzero(magnitude > threshold)[0]
    if len(voiced) == 0:
        return utterance
    end = min(len(utterance.samples), voiced[-1] + int(0.05 * utterance.sample_rate))
    return _Utterance(
        name=utterance.name,
        path=utterance.path,
        samples=utterance.samples[:end],
        sample_rate=utterance.sample_rate,
    )


def load_utterances(names: list[str] | None, include_tools: bool = False) -> list[_Utterance]:
    if not FIXTURES_DIR.exists():
        raise SystemExit(
            f"No fixtures at {FIXTURES_DIR}. Run scripts/benchmarks/make_ttfa_fixtures.py first."
        )
    paths = sorted(FIXTURES_DIR.glob("*.wav"))
    if names:
        wanted = {n.strip() for n in names}
        paths = [p for p in paths if p.stem in wanted]
    elif not include_tools:
        # tool turns spend most of their latency in the tool, not the pipeline
        paths = [p for p in paths if not p.stem.startswith("tool_")]
    if not paths:
        raise SystemExit(f"No matching fixtures in {FIXTURES_DIR}")
    return [_trim_trailing_silence(_load_wav(p)) for p in paths]


# --- trial bookkeeping ---


@dataclass
class TrialResult:
    utterance: str
    trial: int
    ttfa_ms: float | None = None
    stt_first_final_ms: float | None = None
    stt_last_final_ms: float | None = None
    stt_finals: int = 0
    llm_ttft_ms: float | None = None
    llm_first_token_ms: float | None = None
    tts_start_ms: float | None = None
    tts_first_text_ms: float | None = None
    tts_first_audio_ms: float | None = None
    llm_done_ms: float | None = None
    llm_first_chunk_last_ms: float | None = None
    tts_first_audio_last_ms: float | None = None
    llm_node_runs: int = 0
    tts_node_runs: int = 0
    false_cutoff: bool = False
    pump_drift_ms: float = 0.0
    transcript: str = ""
    reply: str = ""
    error: str | None = None


@dataclass
class BenchRun:
    label: str
    config: dict[str, Any]
    trials: list[TrialResult] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        ok = [t for t in self.trials if t.ttfa_ms is not None and not t.false_cutoff]
        cutoffs = sum(1 for t in self.trials if t.false_cutoff)
        values = sorted(t.ttfa_ms for t in ok)  # type: ignore[misc]

        def pct(p: float) -> float | None:
            if not values:
                return None
            return round(values[min(int(len(values) * p), len(values) - 1)], 1)

        def mean_of(attr: str) -> float | None:
            vals = [getattr(t, attr) for t in ok if getattr(t, attr) is not None]
            return round(statistics.mean(vals), 1) if vals else None

        return {
            "label": self.label,
            "config": self.config,
            "n": len(values),
            "failed": len(self.trials) - len(values),
            "ttfa_ms": {
                "p50": pct(0.5),
                "p90": pct(0.9),
                "max": round(values[-1], 1) if values else None,
                "min": round(values[0], 1) if values else None,
                "mean": round(statistics.mean(values), 1) if values else None,
            },
            "stage_means_ms": {
                "stt_first_final": mean_of("stt_first_final_ms"),
                "stt_last_final": mean_of("stt_last_final_ms"),
                "stt_finals_per_turn": mean_of("stt_finals"),
                "tts_node_entered": mean_of("tts_start_ms"),
                "tts_first_text": mean_of("tts_first_text_ms"),
                "llm_done": mean_of("llm_done_ms"),
                "tts_first_audio": mean_of("tts_first_audio_ms"),
                "llm_first_chunk_last": mean_of("llm_first_chunk_last_ms"),
                "tts_first_audio_last": mean_of("tts_first_audio_last_ms"),
                "llm_node_runs": mean_of("llm_node_runs"),
                "tts_node_runs": mean_of("tts_node_runs"),
                "pump_drift": mean_of("pump_drift_ms"),
                "llm_first_token": mean_of("llm_first_token_ms"),
                "llm_ttft_reported": mean_of("llm_ttft_ms"),
            },
            "under_target": sum(1 for v in values if v <= TARGET_TTFA_MS),
            "target_ms": TARGET_TTFA_MS,
            "false_cutoffs": cutoffs,
            "false_cutoff_rate": round(cutoffs / len(self.trials), 3) if self.trials else None,
        }


# --- stage instrumentation ---


class _StageRecorder:
    def __init__(self) -> None:
        self.marks: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def arm(self) -> None:
        self.marks = {}
        self.counts = {}

    def mark(self, name: str) -> None:
        self.marks.setdefault(name, time.monotonic())
        self.counts[name] = self.counts.get(name, 0) + 1

    def mark_last(self, name: str) -> None:
        """Overwrite — for stages that legitimately run more than once a turn."""
        self.marks[name] = time.monotonic()
        self.counts[name] = self.counts.get(name, 0) + 1

    def since(self, name: str, origin: float) -> float | None:
        ts = self.marks.get(name)
        return None if ts is None else round((ts - origin) * 1000, 1)


def _timed_agent_class(base_cls: type, recorder: _StageRecorder) -> type:
    """Subclass the production agent purely to timestamp node entry/exit."""

    class _TimedAgent(base_cls):  # type: ignore[misc, valid-type]
        async def llm_node(self, chat_ctx, tools, model_settings):  # type: ignore[no-untyped-def]
            recorder.mark("llm_start")
            recorder.mark_last("llm_start_last")
            first = True
            async for chunk in super().llm_node(chat_ctx, tools, model_settings):
                if first:
                    recorder.mark("llm_first_chunk")
                    recorder.mark_last("llm_first_chunk_last")
                    first = False
                yield chunk
            recorder.mark("llm_done")

        def tts_node(self, text, model_settings):  # type: ignore[no-untyped-def]
            recorder.mark("tts_start")
            outer = super()

            async def _timed_text(source):
                first = True
                async for chunk in source:
                    if first:
                        recorder.mark("tts_first_text")
                        first = False
                    yield chunk
                recorder.mark("tts_text_done")

            async def _timed_audio():
                first = True
                async for frame in outer.tts_node(_timed_text(text), model_settings):
                    if first:
                        recorder.mark("tts_first_audio")
                        recorder.mark_last("tts_first_audio_last")
                        first = False
                    yield frame

            return _timed_audio()

    _TimedAgent.__name__ = f"Timed{base_cls.__name__}"
    return _TimedAgent


def _build_state(tmp_dir: Path):
    from agent.core.agent_state import AgentState

    state = AgentState(user_id=BENCH_USER_ID, state_dir=tmp_dir)
    state.state["mode"] = "main_menu"
    state.state["user"]["first_time_main_menu"] = False
    state.state["user"]["name"] = "Alex"
    state.set("session.main_menu_greeted", True)
    return state


async def run_bench(args: argparse.Namespace) -> BenchRun:
    from agent.agents.main_menu_agent import MainMenuAgent
    from agent.agents.shared.userdata import UserData
    from agent.core import pipeline_factory as pf

    executor = _LocalInferenceExecutor()
    _install_stub_job_context(executor)

    detector = os.getenv("TURN_DETECTOR", pf.DEFAULT_TURN_DETECTOR).strip().lower()
    if detector not in ("vad", "none", "off"):
        method = (
            "lk_end_of_utterance_en" if detector == "english"
            else "lk_end_of_utterance_multilingual"
        )
        t0 = time.monotonic()
        await asyncio.get_running_loop().run_in_executor(None, executor.warmup, method)
        await executor.do_inference(
            method, json.dumps({"chat_ctx": [{"role": "user", "content": "warm up"}]}).encode()
        )
        logger.info("EOU model warm in %.2fs", time.monotonic() - t0)

    tmp_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "ttfa_bench_state"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    utterances = load_utterances(
        args.utterances.split(",") if args.utterances else None, args.include_tools
    )
    logger.info("Loaded %d fixtures: %s", len(utterances), [u.name for u in utterances])

    config = {
        "llm_model": os.getenv("LLM_MODEL", pf.DEFAULT_LLM_MODEL),
        "llm_reasoning_effort": os.getenv("LLM_REASONING_EFFORT", "(default)"),
        "stt_backend": os.getenv("STT_BACKEND", pf.DEFAULT_STT_BACKEND),
        "stt_model": os.getenv("STT_MODEL", pf.DEFAULT_STT_MODEL),
        "eager_eot": os.getenv("EAGER_EOT_THRESHOLD", "(default)"),
        "tts_backend": os.getenv("TTS_BACKEND", pf.DEFAULT_TTS_BACKEND),
        "tts_model": os.getenv("TTS_MODEL", pf.DEFAULT_TTS_MODEL),
        "llm_provider": os.getenv("LLM_PROVIDER", pf.DEFAULT_LLM_PROVIDER),
        "turn_detector": os.getenv("TURN_DETECTOR", pf.DEFAULT_TURN_DETECTOR),
        "vad_backend": os.getenv("VAD_BACKEND", pf.DEFAULT_VAD),
        "vad_min_silence": os.getenv("VAD_MIN_SILENCE", "(default)"),
        "endpointing_min_delay": os.getenv(
            "ENDPOINTING_MIN_DELAY", str(pf.DEFAULT_ENDPOINTING_MIN_DELAY_S)
        ),
        "preemptive_generation": os.getenv("PREEMPTIVE_GENERATION", "1"),
        "preemptive_tts": os.getenv("PREEMPTIVE_TTS", "1"),
        "affect_enabled": os.getenv("AFFECT_ENABLED", "(default)"),
        "transport_ms": args.transport_ms,
    }
    run = BenchRun(label=args.label, config=config)

    vad = pf.build_vad()
    recorder = _StageRecorder()

    audio_in = BenchAudioInput()
    audio_out = BenchAudioOutput()

    session = AgentSession(
        stt=pf.build_stt(),
        llm=pf.build_llm(),
        tts=pf.build_tts(),
        vad=vad,
        turn_handling=pf.build_turn_handling(),
        userdata=None,
    )

    state = _build_state(tmp_dir)
    userdata = UserData(state=state, room=None)
    session.userdata = userdata

    stt_final_ts: list[float] = []
    transcripts: list[str] = []
    replies: list[str] = []

    @session.on("user_input_transcribed")
    def _on_transcript(ev) -> None:  # type: ignore[no-untyped-def]
        if ev.is_final and ev.transcript.strip():
            stt_final_ts.append(time.monotonic())
            transcripts.append(ev.transcript)

    @session.on("conversation_item_added")
    def _on_item(ev) -> None:  # type: ignore[no-untyped-def]
        if getattr(ev.item, "role", None) == "assistant":
            text = getattr(ev.item, "text_content", None)
            if callable(text):
                text = text()
            if text:
                replies.append(text)

    llm_ttfts: list[float] = []

    @session.on("metrics_collected")
    def _on_metrics(ev) -> None:  # type: ignore[no-untyped-def]
        if hasattr(ev.metrics, "ttft") and not getattr(ev.metrics, "cancelled", False):
            llm_ttfts.append(ev.metrics.ttft)

    agent_cls = _timed_agent_class(MainMenuAgent, recorder)
    agent = agent_cls(state=state, userdata=userdata)

    session.input.audio = audio_in
    session.output.audio = audio_out
    audio_in.start()

    await session.start(agent=agent, record=False)
    await asyncio.sleep(LEAD_SILENCE_S)

    try:
        for warm in range(args.warmup):
            for utterance in utterances[:1]:
                res = await _run_trial(
                    session, audio_in, audio_out, recorder, utterance, -1,
                    stt_final_ts, transcripts, replies, llm_ttfts, args,
                )
                logger.info("  [warmup %d] TTFA=%s (discarded)", warm, res.ttfa_ms)
                await asyncio.sleep(args.gap_s)

        for trial in range(args.trials):
            for utterance in utterances:
                result = await _run_trial(
                    session, audio_in, audio_out, recorder, utterance, trial,
                    stt_final_ts, transcripts, replies, llm_ttfts, args,
                )
                run.trials.append(result)
                status = (
                    "CUTOFF" if result.false_cutoff
                    else "FAIL" if result.error
                    else f"{result.ttfa_ms:.0f}ms"
                )
                logger.info(
                    "  [%s t%d] TTFA=%s  stt_last_final=%s  llm_first=%s  tts_audio=%s  %r",
                    utterance.name, trial, status, result.stt_last_final_ms,
                    result.llm_first_token_ms, result.tts_first_audio_ms,
                    result.reply[:60],
                )
                await asyncio.sleep(args.gap_s)
    finally:
        with contextlib.suppress(Exception):
            await session.aclose()
        await audio_in.aclose()

    return run


async def _wait_until_settled(session, audio_out, quiet_s: float, timeout_s: float) -> None:
    """Block until the agent has stopped speaking and the pipeline is idle.

    Without this the next trial arms its output sink while the previous reply is
    still streaming, and the "first" frame it sees belongs to the old turn.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        speaking = session.current_speech is not None and not session.current_speech.done()
        quiet_for = time.monotonic() - audio_out.last_frame_ts
        if not speaking and quiet_for >= quiet_s:
            return
        await asyncio.sleep(0.05)
    logger.warning("  session never settled within %.1fs — trial may be noisy", timeout_s)


async def _run_trial(
    session, audio_in, audio_out, recorder, utterance, trial,
    stt_final_ts, transcripts, replies, llm_ttfts, args,
) -> TrialResult:
    result = TrialResult(utterance=utterance.name, trial=trial)

    await _wait_until_settled(session, audio_out, args.quiet_s, args.timeout_s)

    stt_final_ts.clear()
    transcripts.clear()
    replies.clear()
    llm_ttfts.clear()
    recorder.arm()
    audio_out.arm()

    audio_in.reset_drift()
    end_index = audio_in.schedule(utterance.samples)
    speech_end = await audio_in.wall_time_of(end_index)
    result.pump_drift_ms = round(audio_in.max_drift_s * 1000, 1)

    deadline = time.monotonic() + args.timeout_s
    first_after: float | None = None
    while time.monotonic() < deadline:
        after = [t for t in audio_out.segment_starts if t >= speech_end]
        if after:
            first_after = after[0]
            break
        await asyncio.sleep(0.01)

    result.false_cutoff = any(t < speech_end for t in audio_out.segment_starts)
    if first_after is None:
        result.error = "false cutoff — agent spoke over the user and never replied after"
        if not result.false_cutoff:
            result.error = "timeout waiting for first audio frame"
        await _wait_until_settled(session, audio_out, args.quiet_s, args.timeout_s)
        return result

    result.ttfa_ms = round((first_after - speech_end) * 1000 + args.transport_ms, 1)
    before_audio = [t for t in stt_final_ts if t <= first_after]
    result.stt_finals = len(before_audio)
    if before_audio:
        result.stt_first_final_ms = round((before_audio[0] - speech_end) * 1000, 1)
        result.stt_last_final_ms = round((before_audio[-1] - speech_end) * 1000, 1)
    result.llm_first_token_ms = recorder.since("llm_first_chunk", speech_end)
    result.tts_start_ms = recorder.since("tts_start", speech_end)
    result.tts_first_text_ms = recorder.since("tts_first_text", speech_end)
    result.tts_first_audio_ms = recorder.since("tts_first_audio", speech_end)
    result.llm_done_ms = recorder.since("llm_done", speech_end)
    result.llm_first_chunk_last_ms = recorder.since("llm_first_chunk_last", speech_end)
    result.tts_first_audio_last_ms = recorder.since("tts_first_audio_last", speech_end)
    result.llm_node_runs = recorder.counts.get("llm_start_last", 0)
    result.tts_node_runs = recorder.counts.get("tts_start", 0)
    result.transcript = transcripts[0] if transcripts else ""

    await _wait_until_settled(session, audio_out, args.quiet_s, args.timeout_s)
    result.reply = replies[0] if replies else ""
    result.llm_ttft_ms = round(llm_ttfts[0] * 1000, 1) if llm_ttfts else None
    return result


def _print_summary(run: BenchRun) -> None:
    s = run.summary()
    t = s["ttfa_ms"]
    print("\n" + "=" * 74)
    print(f"  TTFA BENCHMARK — {s['label']}")
    print("=" * 74)
    for key, value in s["config"].items():
        print(f"    {key:24s} {value}")
    print("-" * 74)
    print(f"    trials                   {s['n']} clean, {s['failed']} excluded")
    print(
        f"    false cutoffs            {s['false_cutoffs']} "
        f"({(s['false_cutoff_rate'] or 0) * 100:.0f}% of turns — agent talked over the user)"
    )
    print(
        f"    TTFA                     p50={t['p50']}ms  p90={t['p90']}ms  "
        f"min={t['min']}ms  max={t['max']}ms"
    )
    print(f"    under {int(s['target_ms'])}ms              {s['under_target']}/{s['n']}")
    print("-" * 74)
    stages = s["stage_means_ms"]
    print("    mean offsets from end of user speech (negative = before it):")
    print(f"      stt first final        {stages['stt_first_final']} ms")
    print(
        f"      stt last final         {stages['stt_last_final']} ms  "
        f"({stages['stt_finals_per_turn']} finals/turn — the reply is built from the last)"
    )
    print(f"      llm first chunk        {stages['llm_first_token']} ms")
    print(f"      llm first chunk (rep)  {stages['llm_ttft_reported']} ms  (provider ttft)")
    print(f"      tts node entered       {stages['tts_node_entered']} ms")
    print(f"      tts first text chunk   {stages['tts_first_text']} ms")
    print(f"      llm stream done        {stages['llm_done']} ms")
    print(f"      tts first audio frame  {stages['tts_first_audio']} ms")
    print(f"    last generation of the turn (preemptive retries land here):")
    print(f"      llm first chunk        {stages['llm_first_chunk_last']} ms")
    print(f"      tts first audio frame  {stages['tts_first_audio_last']} ms")
    print(f"      llm_node runs/turn     {stages['llm_node_runs']}")
    print(f"      tts_node runs/turn     {stages['tts_node_runs']}")
    print(f"      input pump drift       {stages['pump_drift']} ms  (>30ms means the host was too busy to trust)")
    print("=" * 74 + "\n")


def _save(run: BenchRun) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"{stamp}_{run.label}.json"
    path.write_text(
        json.dumps(
            {"summary": run.summary(), "trials": [t.__dict__ for t in run.trials]}, indent=2
        )
    )
    return path


async def _main_async(args: argparse.Namespace) -> BenchRun:
    async with http_context.open():
        return await run_bench(args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--warmup", type=int, default=1, help="discarded warmup turns")
    parser.add_argument("--utterances", type=str, default=None)
    parser.add_argument("--include-tools", action="store_true",
                        help="also replay turns that trigger a function tool")
    parser.add_argument("--label", type=str, default="baseline")
    parser.add_argument("--gap-s", type=float, default=1.0)
    parser.add_argument("--quiet-s", type=float, default=0.7,
                        help="silence required before a trial counts the session idle")
    parser.add_argument("--timeout-s", type=float, default=20.0)
    parser.add_argument(
        "--transport-ms",
        type=float,
        default=DEFAULT_TRANSPORT_MS,
        help="fixed ms added for WebRTC transport + jitter buffer",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    if not args.verbose:
        for noisy in ("livekit", "livekit.agents", "openai", "httpx", "httpcore", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.ERROR)

    run = asyncio.run(_main_async(args))
    _print_summary(run)
    path = _save(run)
    print(f"  saved → {path.relative_to(PROJECT_ROOT)}\n")


if __name__ == "__main__":
    main()
