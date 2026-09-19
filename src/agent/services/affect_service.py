"""AffectService: assembles utterances from the VAD tap, runs the encoder early, keeps the athlete state current."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np
from livekit import rtc
from livekit.agents import vad as agents_vad

from affect.audio import (
    StreamingResampler,
    crop_last_seconds,
    int16_to_float32,
    trim_to_speech,
    voiced_seconds,
)
from affect.baseline import SpeakerBaseline
from affect.config import AffectConfig
from affect.engine import AffectEngine, AffectResult
from affect.prosody import ProsodyFeatures, prosody_features
from affect.recorder import UtteranceRecorder
from affect.state import AthleteState, AthleteStateTracker
from affect.tts_adapters import TTSStyleAdapter, build_adapter
from affect.voice_style import SpeechKind, VoiceStyle, style_for

logger = logging.getLogger(__name__)

PRE_ROLL_SECONDS = 0.3
VAD_WINDOW_SECONDS = 0.032
WORKOUT_MODES = ("workout",)
PROFILE_SUFFIX = ".npz"
LEVEL_LOW = 0.4
LEVEL_HIGH = 0.6
RELATIVE_Z = 1.0


class AffectService:
    """Owns the per-session affect state. Every method runs on the event loop unless noted."""

    def __init__(
        self,
        config: AffectConfig,
        engine: AffectEngine | None,
        state: Any = None,
        profiler: Any = None,
        visual_bridge: Any = None,
        coaching_speaking_fn: Callable[[], bool] | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.config = config
        self.engine = engine
        self._state = state
        self._profiler = profiler
        self._visual_bridge = visual_bridge
        self._coaching_speaking_fn = coaching_speaking_fn or (lambda: False)
        self._clock = clock
        self._user_id = user_id
        self.session_id = session_id or time.strftime("%Y%m%d_%H%M%S")
        self.enabled = bool(config.enabled and engine is not None)

        manifest = engine.manifest if engine is not None and engine.manifest is not None else None
        population_mean = np.array(manifest.population_avd_mean if manifest else [0.5, 0.5, 0.5])
        population_std = np.array(manifest.population_avd_std if manifest else [0.15, 0.15, 0.15])
        self._population_mean = population_mean
        self._population_std = population_std
        self.baseline = self._load_or_create_baseline(user_id, population_mean, population_std)
        self.tracker = AthleteStateTracker(config.state)
        self.style_adapter: TTSStyleAdapter = build_adapter(config.style)
        self.recorder = UtteranceRecorder(
            config.recorder, config.resolve_path(config.recorder.output_dir), user_id, self.session_id
        )

        self._sample_rate = config.audio.model_sample_rate
        self._resampler: StreamingResampler | None = None
        self._input_rate: int | None = None
        self._pre_roll: deque[np.ndarray] = deque()
        self._pre_roll_samples = 0
        self._segment: list[np.ndarray] | None = None
        self._segment_pre_roll = 0
        self._probabilities: list[float] = []
        self._speaking = False
        self._early_task: asyncio.Task | None = None
        self._early_key: int | None = None
        self._final_task: asyncio.Task | None = None
        self._segment_seq = 0
        self._state_seq = 0
        self._consumed_seq = 0
        self._last_state = AthleteState()
        self._last_result: AffectResult | None = None
        self._last_prosody: ProsodyFeatures | None = None
        self._stats = {"utterances": 0, "skipped_short": 0, "early_reused": 0, "final_runs": 0, "fresh_hits": 0, "snapshots": 0}

    # -- baseline persistence -----------------------------------------------------------

    def _profile_path(self, user_id: str) -> Any:
        return self.config.resolve_path(self.config.baseline.profile_dir) / f"{user_id}{PROFILE_SUFFIX}"

    def _load_or_create_baseline(self, user_id: str | None, mean: np.ndarray, std: np.ndarray) -> SpeakerBaseline:
        if user_id:
            path = self._profile_path(user_id)
            if path.exists():
                try:
                    baseline = SpeakerBaseline.load(path, self.config.baseline, mean, std)
                    logger.info("[AFFECT] Loaded baseline profile %s (%s)", path, baseline.summary())
                    return baseline
                except Exception:  # noqa: BLE001
                    logger.exception("[AFFECT] Baseline profile unreadable, starting fresh: %s", path)
        return SpeakerBaseline(self.config.baseline, mean, std, user_id=user_id)

    def set_user_id(self, user_id: str | None) -> None:
        if not user_id or user_id == self._user_id:
            return
        self._user_id = user_id
        self.baseline.user_id = user_id
        self.recorder = UtteranceRecorder(
            self.config.recorder, self.config.resolve_path(self.config.recorder.output_dir), user_id, self.session_id
        )

    def _resolve_user_id(self) -> str | None:
        if self._user_id:
            return self._user_id
        try:
            user_id = self._state.get_user().get("id") if self._state is not None else None
        except Exception:  # noqa: BLE001
            user_id = None
        if user_id:
            self.set_user_id(str(user_id))
        return self._user_id

    def save_profile(self) -> None:
        user_id = self._resolve_user_id()
        if not user_id or self.baseline.sample_count == 0:
            return
        try:
            path = self.baseline.save(self._profile_path(user_id))
            logger.info("[AFFECT] Baseline profile saved: %s", path)
        except Exception:  # noqa: BLE001
            logger.exception("[AFFECT] Failed to save baseline profile")

    # -- context ----------------------------------------------------------------------------

    def current_mode(self) -> str:
        try:
            return str(self._state.get_mode()) if self._state is not None else "unknown"
        except Exception:  # noqa: BLE001
            return "unknown"

    def is_coaching_speaking(self) -> bool:
        try:
            return bool(self._coaching_speaking_fn())
        except Exception:  # noqa: BLE001
            return False

    def speech_kind(self) -> SpeechKind:
        return "coaching" if self.is_coaching_speaking() else "conversation"

    def neutral_context(self) -> bool:
        return self.current_mode() not in WORKOUT_MODES and not self.is_coaching_speaking()

    def wait_budget_s(self, kind: SpeechKind) -> float:
        ms = self.config.trigger.coaching_wait_ms if kind == "coaching" else self.config.trigger.llm_wait_ms
        return max(0.0, ms / 1000.0)

    # -- audio intake (UtteranceSink) --------------------------------------------------------

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if not self.enabled:
            return
        if self._resampler is None or self._input_rate != frame.sample_rate:
            self._input_rate = frame.sample_rate
            self._resampler = StreamingResampler(frame.sample_rate, self._sample_rate)
        pcm = np.frombuffer(frame.data, dtype=np.int16)
        if frame.num_channels > 1:
            pcm = pcm.reshape(-1, frame.num_channels)[:, 0]
        chunk = self._resampler.push(int16_to_float32(pcm))
        if chunk.shape[0] == 0:
            return
        if self._segment is not None:
            self._segment.append(chunk)
            return
        self._pre_roll.append(chunk)
        self._pre_roll_samples += chunk.shape[0]
        limit = int(PRE_ROLL_SECONDS * self._sample_rate)
        while self._pre_roll_samples > limit and len(self._pre_roll) > 1:
            dropped = self._pre_roll.popleft()
            self._pre_roll_samples -= dropped.shape[0]

    def on_speech_start(self, timestamp: float) -> None:
        if not self.enabled:
            return
        self._cancel_early()
        self._segment = list(self._pre_roll)
        self._segment_pre_roll = self._pre_roll_samples
        self._pre_roll.clear()
        self._pre_roll_samples = 0
        self._probabilities = []
        self._speaking = True
        self._segment_seq += 1

    def on_speech_window(self, event: agents_vad.VADEvent) -> None:
        if not self.enabled or self._segment is None:
            return
        self._probabilities.append(float(event.probability))
        cfg = self.config.trigger
        window = event.frames[0].duration if event.frames else VAD_WINDOW_SECONDS
        voiced = voiced_seconds(np.asarray(self._probabilities), window, self.config.audio.speech_prob_threshold)
        if self._early_key == self._segment_seq:
            if event.raw_accumulated_silence < window:
                # Speech resumed after the early trigger: that result is stale.
                self._cancel_early()
            return
        if event.speaking and event.raw_accumulated_silence >= cfg.early_silence_seconds and voiced >= cfg.early_min_voiced_seconds:
            self._early_key = self._segment_seq
            self._early_task = self._schedule_utterance(reason="early")

    def on_speech_end(self, event: agents_vad.VADEvent) -> None:
        if not self.enabled or self._segment is None:
            return
        self._speaking = False
        if self._early_key == self._segment_seq and self._early_task is not None:
            self._stats["early_reused"] += 1
            self._final_task = self._early_task
        else:
            self._cancel_early()
            self._stats["final_runs"] += 1
            self._final_task = self._schedule_utterance(reason="final")
        tail = self._segment[-4:] if self._segment else []
        self._segment = None
        self._probabilities = []
        self._pre_roll = deque(tail)
        self._pre_roll_samples = int(sum(c.shape[0] for c in tail))
        self._early_task = None
        self._early_key = None

    def _cancel_early(self) -> None:
        if self._early_task is not None and not self._early_task.done():
            self._early_task.cancel()
        self._early_task = None
        self._early_key = None

    # -- inference pipeline ---------------------------------------------------------------------

    def _schedule_utterance(self, reason: str) -> asyncio.Task | None:
        if self._segment is None:
            return None
        audio = np.concatenate(self._segment) if self._segment else np.zeros(0, dtype=np.float32)
        probabilities = np.asarray(self._probabilities, dtype=np.float32)
        segment_seq = self._segment_seq
        pre_roll = self._segment_pre_roll
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return None
        return loop.create_task(
            self._run_utterance(audio, probabilities, pre_roll, segment_seq, reason), name=f"affect-{reason}"
        )

    def _prepare_audio(self, audio: np.ndarray, probabilities: np.ndarray, pre_roll: int) -> tuple[np.ndarray, float]:
        cfg = self.config.audio
        # Probabilities are aligned to the audio after the pre-roll captured before speech start.
        probs_audio = audio[pre_roll:] if audio.shape[0] > pre_roll else audio
        trimmed = trim_to_speech(probs_audio, self._sample_rate, probabilities, VAD_WINDOW_SECONDS, cfg.speech_prob_threshold, cfg.trim_pad_seconds)
        trimmed = crop_last_seconds(trimmed, self._sample_rate, cfg.max_seconds)
        voiced = voiced_seconds(probabilities, VAD_WINDOW_SECONDS, cfg.speech_prob_threshold)
        return trimmed.astype(np.float32, copy=False), voiced

    async def _run_utterance(
        self, audio: np.ndarray, probabilities: np.ndarray, pre_roll: int, segment_seq: int, reason: str
    ) -> AthleteState | None:
        wave, voiced = self._prepare_audio(audio, probabilities, pre_roll)
        if voiced < self.config.audio.min_voiced_seconds or wave.shape[0] < int(self.config.audio.min_voiced_seconds * self._sample_rate):
            self._stats["skipped_short"] += 1
            self.tracker.mark_stale()
            return None
        assert self.engine is not None
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        result = await self.engine.infer_async(wave, loop)
        prosody = await self.engine.run_in_executor(loop, prosody_features, wave, self._sample_rate, probabilities, self.config.audio.speech_prob_threshold)
        total_ms = (time.perf_counter() - t0) * 1000.0
        return self._integrate(result, prosody, wave, voiced, segment_seq, reason, total_ms)

    def _integrate(
        self,
        result: AffectResult,
        prosody: ProsodyFeatures,
        wave: np.ndarray,
        voiced: float,
        segment_seq: int,
        reason: str,
        total_ms: float,
    ) -> AthleteState:
        utterance_id = f"{self.session_id}-{segment_seq}-{uuid.uuid4().hex[:6]}"
        avd = result.avd_vector()
        neutral = self.neutral_context()
        accepted = self.baseline.observe(avd, voiced, neutral, utterance_id=utterance_id, embedding=result.embedding)
        z = self.baseline.z_scores(avd, exclude_ids={utterance_id} if accepted else frozenset())
        confident = self.baseline.ready or not self.config.baseline.require_ready_for_confidence
        now = self._clock()
        state = self.tracker.observe_utterance(z, now=now, confident=confident)
        self._last_state = state
        self._last_result = result
        self._last_prosody = prosody
        self._state_seq += 1
        self._stats["utterances"] += 1
        logger.info(
            "[AFFECT] %s utt=%s voiced=%.1fs avd=(%.2f,%.2f,%.2f) z=(%.2f,%.2f,%.2f) → effort=%s affect=%s conf=%s infer=%.0fms total=%.0fms enrol=%.0fs%s",
            reason, segment_seq, voiced, result.arousal, result.dominance, result.valence,
            z[0], z[1], z[2], state.effort, state.affect, confident, result.infer_ms, total_ms,
            self.baseline.enrollment_seconds, " (neutral sample)" if accepted else "",
        )
        if self._profiler is not None:
            self._profiler.record(
                "affect", "utterance", reason=reason, voiced_s=round(voiced, 2), infer_ms=round(result.infer_ms, 1),
                total_ms=round(total_ms, 1), arousal=round(result.arousal, 3), dominance=round(result.dominance, 3),
                valence=round(result.valence, 3), arousal_z=round(float(z[0]), 2), valence_z=round(float(z[2]), 2),
                effort=state.effort, affect=state.affect, confident=confident, provider=result.provider,
            )
            record_infer = getattr(self._profiler, "record_affect_infer", None)
            if callable(record_infer):
                record_infer(result.infer_ms)
        self._publish(state)
        self.recorder.record(
            wave, self._sample_rate,
            {
                "utterance_id": utterance_id, "reason": reason, "mode": self.current_mode(),
                "coaching_speaking": self.is_coaching_speaking(), "voiced_s": round(voiced, 3),
                "avd": [result.arousal, result.dominance, result.valence], "z": [float(v) for v in z],
                "effort": state.effort, "affect": state.affect, "confident": confident,
                "prosody": prosody.model_dump(), "model": self.engine.manifest.version if self.engine and self.engine.manifest else "",
                "best_ascent_s": self.tracker.best_ascent_s, "neutral_sample": accepted,
            },
        )
        return state

    def _publish(self, state: AthleteState) -> None:
        if self._visual_bridge is None:
            return
        try:
            self._visual_bridge.send(state.to_display())
        except Exception:  # noqa: BLE001
            logger.debug("[AFFECT] display publish failed", exc_info=True)

    # -- consumers ------------------------------------------------------------------------------

    async def snapshot(self, max_wait_s: float = 0.0) -> AthleteState:
        """Current state, waiting at most max_wait_s for an in-flight utterance; never blocks longer."""
        task = self._final_task if self._final_task is not None and not self._final_task.done() else self._early_task
        if task is not None and not task.done() and max_wait_s > 0.0:
            await asyncio.wait({task}, timeout=max_wait_s)
        self._stats["snapshots"] += 1
        fresh = self._state_seq > self._consumed_seq
        self._consumed_seq = self._state_seq
        if fresh:
            self._stats["fresh_hits"] += 1
        return self._last_state.model_copy(update={"fresh": fresh})

    @property
    def last_state(self) -> AthleteState:
        return self._last_state

    @property
    def last_result(self) -> AffectResult | None:
        return self._last_result

    def current_style(self, kind: SpeechKind | None = None) -> VoiceStyle:
        return style_for(self._last_state, kind or self.speech_kind())

    def record_llm_wait(self, wait_ms: float, fresh: bool) -> None:
        if self._profiler is None:
            return
        record = getattr(self._profiler, "record_affect_wait", None)
        if callable(record):
            record(wait_ms, fresh)

    # -- biomechanics hooks ---------------------------------------------------------------------

    def on_rep_effort(self, ascent_time_s: float) -> AthleteState:
        state = self.tracker.observe_rep(ascent_time_s, now=self._clock())
        self._last_state = state
        self._state_seq += 1
        self._publish(state)
        return state

    def on_set_reset(self) -> None:
        self.tracker.on_set_reset()
        self._last_state = self.tracker.snapshot()
        self._state_seq += 1

    # -- introspection ----------------------------------------------------------------------------

    @staticmethod
    def _level(value: float) -> str:
        if value < LEVEL_LOW:
            return "low"
        if value > LEVEL_HIGH:
            return "high"
        return "medium"

    @staticmethod
    def _relative(z: float) -> str:
        if z > RELATIVE_Z:
            return "above their usual"
        if z < -RELATIVE_Z:
            return "below their usual"
        return "about their usual"

    def describe_for_llm(self) -> str:
        """Plain-language summary of the latest voice reading, for the how_do_i_sound tool."""
        if not self.enabled:
            return "Voice perception is off in this session, so go only by their words."
        result = self._last_result
        state = self._last_state
        if result is None:
            return (
                "No voice reading yet: the recent turns were too short or too quiet to read "
                "(a reading needs about a second of speech)."
            )
        age_s = max(0.0, self._clock() - state.updated_at) if state.updated_at else 0.0
        lines = [
            f"Last voice reading was {age_s:.0f} seconds ago.",
            f"Arousal (energy) {self._level(result.arousal)}, {self._relative(state.arousal_z)}. "
            f"Valence (positivity) {self._level(result.valence)}, {self._relative(state.valence_z)}. "
            f"Dominance (assertiveness) {self._level(result.dominance)}, {self._relative(state.dominance_z)}.",
            f"Overall state: effort {state.effort.replace('_', ' ')}, affect {state.affect}.",
        ]
        if self.baseline.ready:
            lines.append("Confidence: high; the baseline for this user is enrolled.")
        else:
            lines.append(
                f"Confidence: low; only {self.baseline.enrollment_seconds:.0f} of "
                f"{self.config.baseline.enrollment_floor_seconds:.0f} seconds of relaxed speech are enrolled, "
                "so comparisons are against population norms and are rough."
            )
        if self._last_prosody is not None and self._last_prosody.f0_median_hz > 0:
            lines.append(f"Pitch median {self._last_prosody.f0_median_hz:.0f} hertz, pause ratio {self._last_prosody.pause_ratio:.2f}.")
        return " ".join(lines)

    def snapshot_dict(self) -> dict[str, Any]:
        result = self._last_result
        return {
            "enabled": self.enabled,
            "model": self.engine.manifest.to_summary() if self.engine and self.engine.manifest else None,
            "provider": self.engine.provider if self.engine else None,
            "state": self._last_state.model_dump(),
            "prompt_line": self._last_state.to_prompt_line(),
            "style": self.current_style().model_dump(),
            "baseline": self.baseline.summary(),
            "last_avd": [result.arousal, result.dominance, result.valence] if result else None,
            "last_infer_ms": round(result.infer_ms, 1) if result else None,
            "prosody": self._last_prosody.model_dump() if self._last_prosody else None,
            "stats": dict(self._stats),
        }

    async def stop(self) -> None:
        self._cancel_early()
        if self._final_task is not None and not self._final_task.done():
            self._final_task.cancel()
        self.baseline.end_session()
        self.save_profile()
        if self.engine is not None:
            self.engine.release()
        logger.info("[AFFECT] Service stopped: %s", self._stats)
