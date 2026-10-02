"""Builds the cascade voice pipeline (STT, LLM, TTS, turn handling).

Shared by the production entrypoint and the TTFA benchmark so that measured
latency always reflects the pipeline that ships. Every knob reads an env var
with the production value as its default, which lets the benchmark sweep
configurations without forking the pipeline definition.
"""

from __future__ import annotations

import os

from livekit.agents import TurnHandlingOptions, inference
from livekit.plugins import cartesia, deepgram, elevenlabs, google, openai, silero
from livekit.plugins.turn_detector.english import EnglishModel
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from openai.types import Reasoning

STT_KEYTERMS = [
    "Barbell Back Squat", "Romanian Deadlift", "Barbell Bench Press",
    "Barbell Overhead Press", "Barbell Front Squat", "Goblet Squat",
    "Sumo Deadlift", "Barbell Deadlift",
    "reps", "sets", "RPE", "deload", "hypertrophy",
]

DEFAULT_STT_BACKEND = "nova"
DEFAULT_STT_MODEL = "nova-3"
DEFAULT_FLUX_MODEL = "flux-general-en"
# Deepgram's default endpoint is US-hosted (~120 ms round trip from Europe, against ~25 ms
# for the EU one). Every audio packet and transcript pays that, and STT is the first link
# of the serial chain: measured from France, the EU endpoint returns the final transcript
# ~100 ms sooner. Same API key. The EU endpoint is the default; set STT_BASE_URL to the
# region nearest the rack.
DEEPGRAM_US_URL = "https://api.deepgram.com/v1/listen"
DEEPGRAM_EU_URL = "https://api.eu.deepgram.com/v1/listen"
DEFAULT_EAGER_EOT_THRESHOLD = 0.5
DEFAULT_EOT_THRESHOLD = 0.7
DEFAULT_LLM_MODEL = "gpt-5.4-mini"
# Nova's product voice is ElevenLabs Flash v2.5: both Cartesia paths (LiveKit Inference
# gateway and the direct account) are out of credits, and Flash measured ~172 ms TTFB.
DEFAULT_TTS_BACKEND = "elevenlabs"
DEFAULT_TTS_MODEL = "eleven_flash_v2_5"
DEFAULT_ELEVENLABS_VOICE_ID = "1SM7GgM6IMuvQlz2BwM3"
INFERENCE_TTS_MODEL = "cartesia/sonic-3"
CARTESIA_TTS_MODEL = "sonic-3"
DEFAULT_VOICE_ID = "3e39e9a5-585c-4f5f-bac6-5e4905c51095"
DEFAULT_SILERO_MIN_SILENCE_S = 0.35
DEFAULT_ENDPOINTING_MIN_DELAY_S = 0.2
# What the shipped agent ran with: it never set max_delay, and LiveKit's default for a
# transcript-based turn detector is 3.0. This is how long Nova waits when the detector
# thinks the user is mid-thought, so lowering it trades patience for speed — measure
# false cutoffs on real hesitant speech before changing it.
DEFAULT_ENDPOINTING_MAX_DELAY_S = 3.0
DEFAULT_TURN_DETECTOR = "multilingual"
DEFAULT_VAD = "silero"
DEFAULT_LLM_PROVIDER = "openai"

RESPONSES_API_PREFIXES = ("gpt-5.5", "gpt-5.6", "gpt-6")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None or raw == "" else float(raw)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    return default if raw is None or raw == "" else raw.strip().lower() in ("1", "true", "yes", "on")


def build_stt():
    """Nova-3, or Flux which fuses transcription with end-of-turn detection.

    Flux emits preflight transcripts before the turn is final, which is what
    lets preemptive generation start the LLM while the user is still talking.
    """
    if os.getenv("STT_BACKEND", DEFAULT_STT_BACKEND).strip().lower() == "flux":
        return deepgram.STTv2(
            model=os.getenv("STT_MODEL", DEFAULT_FLUX_MODEL),
            eager_eot_threshold=_env_float("EAGER_EOT_THRESHOLD", DEFAULT_EAGER_EOT_THRESHOLD),
            eot_threshold=_env_float("EOT_THRESHOLD", DEFAULT_EOT_THRESHOLD),
            keyterm=STT_KEYTERMS,
        )
    return deepgram.STT(
        model=os.getenv("STT_MODEL", DEFAULT_STT_MODEL),
        language="en",
        keyterm=STT_KEYTERMS,
        base_url=os.getenv("STT_BASE_URL", DEEPGRAM_EU_URL),
    )


def build_vad():
    """Silero (0.35s silence floor) or the LiveKit native VAD (0.25s)."""
    if os.getenv("VAD_BACKEND", DEFAULT_VAD).strip().lower() == "inference":
        return inference.VAD(
            min_silence_duration=_env_float("VAD_MIN_SILENCE", 0.25),
        )
    return silero.VAD.load(
        min_silence_duration=_env_float("VAD_MIN_SILENCE", DEFAULT_SILERO_MIN_SILENCE_S),
    )


def build_llm():
    provider = os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER).strip().lower()
    if provider == "inference":
        return inference.LLM(model=os.environ["LLM_MODEL"])
    if provider == "google":
        return google.LLM(model=os.environ["LLM_MODEL"])
    if provider == "cerebras":
        # Cerebras runs Qwen with reasoning effort "high" unless told otherwise, which
        # would put seconds of thinking in front of every spoken reply.
        return openai.LLM(
            model=os.environ["LLM_MODEL"],
            api_key=os.environ["CEREBRAS_API_KEY"],
            base_url=CEREBRAS_BASE_URL,
            reasoning_effort=os.getenv("LLM_REASONING_EFFORT", "none"),
        )
    if provider == "groq":
        # Groq is OpenAI-compatible; the openai plugin talks to it directly.
        return openai.LLM(
            model=os.environ["LLM_MODEL"],
            api_key=os.environ["GROQ_API_KEY"],
            base_url=GROQ_BASE_URL,
        )

    model = os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)
    # LLM_API=responses opts an older model into the Responses API, where reasoning can be
    # switched fully off: gpt-5.4-mini measured ~620 ms to first token that way against
    # ~940 ms on chat completions, whose lowest setting for it is "low".
    use_responses = os.getenv("LLM_API", "").strip().lower() == "responses"
    if use_responses or model.startswith(RESPONSES_API_PREFIXES):
        # gpt-5.5+ rejects reasoning_effort + function tools on /v1/chat/completions;
        # OpenAI requires the Responses API for this combination.
        return openai.responses.LLM(
            model=model,
            reasoning=Reasoning(effort=os.getenv("LLM_REASONING_EFFORT", "none")),
        )
    return openai.LLM(
        model=model,
        reasoning_effort=os.getenv("LLM_REASONING_EFFORT", "low"),
    )


def build_tts():
    """ElevenLabs (the default), or Cartesia over LiveKit Inference or direct.

    TTS_BACKEND=inference uses LiveKit credits, cartesia the Cartesia key.
    """
    voice = os.getenv("TTS_VOICE_ID") or os.getenv("CARTESIA_VOICE_ID", DEFAULT_VOICE_ID)
    backend = os.getenv("TTS_BACKEND", DEFAULT_TTS_BACKEND).strip().lower()
    if backend == "elevenlabs":
        return elevenlabs.TTS(
            model=os.getenv("TTS_MODEL", DEFAULT_TTS_MODEL),
            voice_id=os.getenv("ELEVENLABS_VOICE_ID") or DEFAULT_ELEVENLABS_VOICE_ID,
        )
    if backend == "cartesia":
        return cartesia.TTS(
            model=os.getenv("TTS_MODEL", CARTESIA_TTS_MODEL),
            voice=voice,
            language="en",
        )
    return inference.TTS(
        model=os.getenv("TTS_MODEL", INFERENCE_TTS_MODEL),
        voice=voice,
        language="en",
    )


def build_turn_detector():
    name = os.getenv("TURN_DETECTOR", DEFAULT_TURN_DETECTOR).strip().lower()
    if name in ("vad", "none", "off"):
        return "vad"
    if name == "stt":
        return "stt"
    if name == "english":
        return EnglishModel()
    if name in ("inference", "streaming"):
        version = os.getenv("TURN_DETECTOR_VERSION")
        return inference.TurnDetector(version=version) if version else inference.TurnDetector()
    return MultilingualModel()


LIVEKIT_BOUNDARY_END_S = 1.0
DEFAULT_REPLY_START_VAD_WINDOW_S = 0.0


def _interruption_options() -> dict:
    """REPLY_START_VAD_WINDOW: seconds at the start of each reply during which raw VAD may interrupt.

    LiveKit keeps plain VAD interruption armed for the first 1.0 s of every reply
    (backchannel_boundary) and only then hands over to its adaptive detector, the one that
    can tell a person from noise. In that second the rule is simply "VAD speech_duration
    >= 0.5 s", and speech_duration still counts the user's own utterance if their VAD
    segment has not closed, as well as any of Nova's voice the echo canceller lets through.
    A fast reply lands inside that window and she cuts herself off; a slow one does not.
    0 (the default, confirmed live 2026-09-21) hands the whole reply to the adaptive
    detector: barge-in still works, no latency added. 1.0 restores LiveKit's default.
    The adaptive detector only runs in console/dev mode (or with LIVEKIT_REMOTE_EOT_URL);
    under `start` this window is a no-op and raw VAD stays armed for the whole reply.
    """
    window = _env_float("REPLY_START_VAD_WINDOW", DEFAULT_REPLY_START_VAD_WINDOW_S)
    return {"interruption": {"backchannel_boundary": (window, LIVEKIT_BOUNDARY_END_S)}}


def build_turn_handling() -> TurnHandlingOptions:
    return TurnHandlingOptions(
        turn_detection=build_turn_detector(),
        endpointing={
            "min_delay": _env_float("ENDPOINTING_MIN_DELAY", DEFAULT_ENDPOINTING_MIN_DELAY_S),
            "max_delay": _env_float("ENDPOINTING_MAX_DELAY", DEFAULT_ENDPOINTING_MAX_DELAY_S),
        },
        preemptive_generation={
            "enabled": _env_flag("PREEMPTIVE_GENERATION", True),
            "preemptive_tts": _env_flag("PREEMPTIVE_TTS", True),
        },
        **_interruption_options(),
    )
