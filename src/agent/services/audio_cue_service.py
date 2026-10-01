"""
Audio Cue Service — Loads pre-generated TTS audio for real-time coaching.

Audio files are generated once via scripts/tools/generate_cue_audio.py, which
speaks the reviewed lines in src/assets/cue_text/cues.json with Cartesia
Sonic-3 (same voice as the voice agent). Multiple variants per cue are stored
on disk as WAV files; this service indexes the paths and picks a random
variant each time for natural-sounding playback.

Playback routes through LiveKit's session.say() so audio reaches the user
via the WebRTC track.

Fallback: if no pre-generated files exist for a cue, generates audio on
the fly via the OpenAI TTS API and plays it through session.say().
"""

import asyncio
import logging
import os
import random
import time
import wave
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Tuple

from livekit import rtc
from livekit.agents.utils.audio import audio_frames_from_file
from openai import AsyncOpenAI

from agent.services.coaching_constants import CUE_TEXT_MAP

logger = logging.getLogger(__name__)

# Directory where pre-generated cue files live
CUES_DIR = Path(__file__).parent.parent.parent / "assets" / "cues"
CUES_WAV_DIR = CUES_DIR / "wav"
REP_SOUND_PATH = CUES_DIR / "validation_sound.wav"

# Fallback TTS configuration (only used if pre-generated files are missing)
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = os.getenv("REALTIME_VOICE", "cedar")
TTS_SPEED = 1.1
TTS_FORMAT = "pcm"  # 24kHz 16-bit mono PCM
TTS_INSTRUCTIONS = (
    "You are an energetic gym coach giving real-time cues during a workout. "
    "Speak with high energy, urgency, and motivation — like you're right next "
    "to the lifter on the gym floor. Short, punchy, and commanding."
)

# Audio format constants
SAMPLE_RATE = 24000
NUM_CHANNELS = 1
SAMPLES_PER_CHUNK = 2048  # ~85ms per chunk at 24kHz


class AudioCueService:
    """
    Indexes pre-generated WAV cue files from disk with random variant selection.
    Plays cues through LiveKit's session.say() for WebRTC delivery.

    Falls back to runtime TTS generation for any cues missing on disk.
    """

    def __init__(self, session) -> None:
        self._session = session  # AgentSession — for session.say()
        # cue_key → list of WAV file paths (one per variant)
        self._disk_cache: Dict[str, List[Path]] = {}
        # cue_key → list of variants, each variant is a list of AudioFrame chunks (in-memory)
        self._memory_cache: Dict[str, List[List[rtc.AudioFrame]]] = {}
        # Runtime-generated fallback cache: cue_key → raw PCM bytes
        self._fallback_cache: Dict[str, bytes] = {}
        self._client: Optional[AsyncOpenAI] = None
        self._disk_loaded: bool = False
        # Single rep sound (replaces per-number TTS variants)
        self._rep_sound_frames: Optional[List[rtc.AudioFrame]] = None
        # Separate audio track for rep sounds (plays independently of agent speech)
        self._rep_audio_source: Optional[rtc.AudioSource] = None
        self._rep_track_ready: bool = False
        # The rep sound's own format — it is a real recording, not TTS output,
        # so it does not necessarily match the 24kHz mono the cues use.
        self._rep_sample_rate: int = SAMPLE_RATE
        self._rep_channels: int = NUM_CHANNELS

        # Eagerly load and pre-read all WAV cues into memory
        self._load_rep_sound()
        self._load_from_disk()
        self._validate_expected_cues()

    @property
    def session(self):
        """The bound AgentSession, or None before attach_session()."""
        return self._session

    @property
    def rep_track_ready(self) -> bool:
        """Whether the dedicated rep sound track has been published."""
        return self._rep_track_ready

    def attach_session(self, session) -> None:
        """Bind a live AgentSession for audio playback (after prewarm)."""
        self._session = session

    def _load_rep_sound(self) -> None:
        """Load the single rep validation sound into memory."""
        if not REP_SOUND_PATH.exists():
            logger.warning(f"[AUDIO CUE] Rep sound not found: {REP_SOUND_PATH}")
            return
        try:
            with wave.open(str(REP_SOUND_PATH), "rb") as wf:
                sample_rate = wf.getframerate()
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                pcm_data = wf.readframes(wf.getnframes())

            if sample_width != 2:
                logger.error(
                    f"[AUDIO CUE] Rep sound must be 16-bit PCM, got "
                    f"{sample_width * 8}-bit — rep sounds disabled"
                )
                return

            # Describe the frames with the file's real format. Declaring a
            # 44.1kHz stereo recording as 24kHz mono makes LiveKit read
            # interleaved channel bytes as consecutive samples: the beep
            # comes out as noise at the wrong speed, or not at all.
            self._rep_sample_rate = sample_rate
            self._rep_channels = channels

            chunk_bytes = SAMPLES_PER_CHUNK * channels * sample_width
            frames = []
            for offset in range(0, len(pcm_data), chunk_bytes):
                chunk = pcm_data[offset:offset + chunk_bytes]
                samples = len(chunk) // (channels * sample_width)
                if samples == 0:
                    continue
                frames.append(rtc.AudioFrame(
                    data=chunk,
                    sample_rate=sample_rate,
                    num_channels=channels,
                    samples_per_channel=samples,
                ))
            self._rep_sound_frames = frames
            duration = len(pcm_data) / (sample_rate * channels * sample_width)
            logger.info(
                f"[AUDIO CUE] Loaded rep sound from {REP_SOUND_PATH} "
                f"({len(frames)} chunks, {sample_rate}Hz x{channels}, {duration:.2f}s)"
            )
        except Exception as e:
            logger.error(f"[AUDIO CUE] Failed to load rep sound: {e}")

    async def setup_rep_track(self, room) -> None:
        """Publish a dedicated audio track for rep validation sounds.

        This track is independent of the agent's main speech track so rep
        sounds can play concurrently without blocking or being blocked.
        """
        if self._rep_track_ready or self._rep_sound_frames is None:
            return
        try:
            # Must match the frames being captured, not the TTS cue format.
            self._rep_audio_source = rtc.AudioSource(
                sample_rate=self._rep_sample_rate,
                num_channels=self._rep_channels,
            )
            track = rtc.LocalAudioTrack.create_audio_track(
                "rep_sound", self._rep_audio_source,
            )
            await room.local_participant.publish_track(track)
            self._rep_track_ready = True
            logger.info("[AUDIO CUE] Published separate rep sound track")
        except Exception as e:
            logger.error(f"[AUDIO CUE] Failed to publish rep sound track: {e}")
            self._rep_audio_source = None

    async def play_rep_sound(self) -> None:
        """Play the rep validation sound on the dedicated track (non-blocking
        relative to the main agent audio)."""
        if not self._rep_track_ready or not self._rep_audio_source or not self._rep_sound_frames:
            # Fall back to session.say() if the separate track isn't set up
            if self._rep_sound_frames:
                handle = self._session.say(
                    "",
                    audio=self._frames_to_async_gen(self._rep_sound_frames),
                    allow_interruptions=False,
                    add_to_chat_ctx=False,
                )
                await handle.wait_for_playout()
            return
        try:
            for frame in self._rep_sound_frames:
                await self._rep_audio_source.capture_frame(frame)
        except Exception as e:
            logger.error(f"[AUDIO CUE] Rep sound playback failed: {e}")

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI()
        return self._client

    def _load_from_disk(self) -> None:
        """Index all pre-generated WAV files from the cues directory."""
        if self._disk_loaded:
            return

        if not CUES_WAV_DIR.exists():
            logger.warning(f"[AUDIO CUE] WAV cues directory not found: {CUES_WAV_DIR}")
            self._disk_loaded = True
            return

        count = 0
        for wav_file in CUES_WAV_DIR.glob("*.wav"):
            # Filename format: {cue_key}_{variant}.wav
            name = wav_file.stem
            # Split on last underscore to get cue_key and variant number
            last_underscore = name.rfind("_")
            if last_underscore == -1:
                continue
            cue_key = name[:last_underscore]
            try:
                int(name[last_underscore + 1:])  # Validate variant is a number
            except ValueError:
                continue

            if cue_key not in self._disk_cache:
                self._disk_cache[cue_key] = []
            self._disk_cache[cue_key].append(wav_file)
            count += 1

        self._disk_loaded = True
        logger.info(
            f"[AUDIO CUE] Indexed {count} pre-generated WAV files "
            f"for {len(self._disk_cache)} cue keys from {CUES_WAV_DIR}"
        )

        # Pre-read all WAV files into memory as AudioFrame lists
        # (audio_frames_from_file is async, so we read WAV files directly here)
        mem_count = 0
        bytes_per_sample = 2  # int16
        chunk_bytes = SAMPLES_PER_CHUNK * NUM_CHANNELS * bytes_per_sample
        for cue_key, paths in self._disk_cache.items():
            self._memory_cache[cue_key] = []
            for wav_path in paths:
                try:
                    with wave.open(str(wav_path), "rb") as wf:
                        pcm_data = wf.readframes(wf.getnframes())
                    frames = []
                    for offset in range(0, len(pcm_data), chunk_bytes):
                        chunk = pcm_data[offset:offset + chunk_bytes]
                        samples = len(chunk) // (NUM_CHANNELS * bytes_per_sample)
                        frames.append(rtc.AudioFrame(
                            data=chunk,
                            sample_rate=SAMPLE_RATE,
                            num_channels=NUM_CHANNELS,
                            samples_per_channel=samples,
                        ))
                    self._memory_cache[cue_key].append(frames)
                    mem_count += 1
                except Exception as e:
                    logger.warning(f"[AUDIO CUE] Failed to pre-load {wav_path}: {e}")
        logger.info(f"[AUDIO CUE] Pre-loaded {mem_count} WAV variants into memory")

    def _validate_expected_cues(self) -> None:
        """Log warnings for expected cue keys missing from disk (will use slow TTS fallback)."""
        missing = [
            k for k in CUE_TEXT_MAP
            if not k.startswith("rep_")
            and k not in self._memory_cache
            and k not in self._disk_cache
        ]
        if missing:
            logger.warning(
                f"[AUDIO CUE] {len(missing)} cues missing from disk — "
                f"will use TTS fallback (~500ms each): {missing}"
            )

    async def cache_cues(self, cues: Dict[str, str]) -> None:
        """
        Ensure cues are available for playback.

        Indexes pre-generated WAV files from disk. For any requested cue keys
        that don't have pre-generated files, falls back to runtime TTS.
        """
        self._load_from_disk()

        # Find cues that need runtime generation (no disk files)
        # Skip rep cues — they use the single validation sound
        keys_to_generate = [
            k for k in cues
            if k in CUE_TEXT_MAP
            and not k.startswith("rep_")
            and k not in self._disk_cache
            and k not in self._fallback_cache
        ]

        if not keys_to_generate:
            logger.info(f"[AUDIO CUE] All {len(cues)} cues available (pre-generated)")
            return

        logger.info(
            f"[AUDIO CUE] {len(keys_to_generate)} cues missing from disk, "
            f"generating via TTS fallback..."
        )

        start = time.monotonic()
        tasks = [
            self._generate_single_cue(key, CUE_TEXT_MAP[key])
            for key in keys_to_generate
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        errors = 0
        for result in results:
            if isinstance(result, Exception):
                errors += 1
                logger.error(f"[AUDIO CUE] Fallback generation failed: {result}")
            else:
                key, audio_bytes = result
                self._fallback_cache[key] = audio_bytes

        elapsed = time.monotonic() - start
        logger.info(
            f"[AUDIO CUE] Fallback generated {len(keys_to_generate) - errors} cues "
            f"in {elapsed:.2f}s ({errors} errors)"
        )

    def has_cue(self, cue_key: str) -> bool:
        """Check whether audio is available for a cue key."""
        self._load_from_disk()
        if cue_key.startswith("rep_") and self._rep_sound_frames:
            return True
        return cue_key in self._memory_cache or cue_key in self._disk_cache or cue_key in self._fallback_cache

    async def play_cue(self, cue_key: str) -> None:
        """Play a cue through LiveKit's session.say()."""
        self._load_from_disk()

        # Rep cues use the single validation sound on a separate track
        if cue_key.startswith("rep_") and self._rep_sound_frames:
            await self.play_rep_sound()
            return

        # Prefer in-memory pre-loaded frames (zero disk I/O)
        mem_variants = self._memory_cache.get(cue_key)
        if mem_variants:
            frames = random.choice(mem_variants)
            handle = self._session.say(
                "",
                audio=self._frames_to_async_gen(frames),
                allow_interruptions=False,
                add_to_chat_ctx=False,
            )
            await handle.wait_for_playout()
            return

        # Fall back to disk-based WAV variants (if memory pre-load failed)
        disk_variants = self._disk_cache.get(cue_key)
        if disk_variants:
            wav_path = random.choice(disk_variants)
            audio_source = audio_frames_from_file(
                str(wav_path), sample_rate=SAMPLE_RATE, num_channels=NUM_CHANNELS,
            )
            handle = self._session.say(
                "",
                audio=audio_source,
                allow_interruptions=False,
                add_to_chat_ctx=False,
            )
            await handle.wait_for_playout()
            return

        # Fall back to runtime-generated PCM cache
        pcm_bytes = self._fallback_cache.get(cue_key)
        if pcm_bytes:
            audio_source = self._pcm_to_audio_frames(pcm_bytes)
            handle = self._session.say(
                "",
                audio=audio_source,
                allow_interruptions=False,
                add_to_chat_ctx=False,
            )
            await handle.wait_for_playout()
            return

        logger.warning(f"[AUDIO CUE] No audio available for cue: {cue_key}")

    async def generate_tts(self, text: str) -> List[rtc.AudioFrame]:
        """Generate TTS audio for arbitrary text and return as AudioFrame list."""
        client = self._get_client()
        response = await client.audio.speech.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=text,
            instructions=TTS_INSTRUCTIONS,
            response_format=TTS_FORMAT,
            speed=TTS_SPEED,
        )
        pcm_bytes = response.read()
        bytes_per_sample = 2
        chunk_bytes = SAMPLES_PER_CHUNK * NUM_CHANNELS * bytes_per_sample
        frames: List[rtc.AudioFrame] = []
        for offset in range(0, len(pcm_bytes), chunk_bytes):
            chunk = pcm_bytes[offset:offset + chunk_bytes]
            samples = len(chunk) // (NUM_CHANNELS * bytes_per_sample)
            frames.append(rtc.AudioFrame(
                data=chunk,
                sample_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
                samples_per_channel=samples,
            ))
        return frames

    async def play_frames(self, frames: List[rtc.AudioFrame]) -> None:
        """Play raw AudioFrame list through session.say()."""
        if not frames:
            return
        handle = self._session.say(
            "",
            audio=self._frames_to_async_gen(frames),
            allow_interruptions=False,
            add_to_chat_ctx=False,
        )
        await handle.wait_for_playout()

    @staticmethod
    async def _frames_to_async_gen(
        frames: List[rtc.AudioFrame],
    ) -> AsyncGenerator[rtc.AudioFrame, None]:
        """Yield pre-loaded AudioFrame chunks as an async generator."""
        for frame in frames:
            yield frame

    def is_cache_valid(self) -> bool:
        """Check whether any cue audio is available."""
        self._load_from_disk()
        return bool(self._disk_cache) or bool(self._fallback_cache)

    @staticmethod
    def get_cue_text(cue_key: str) -> Optional[str]:
        """Return the spoken text for a cue key (used by fallback)."""
        return CUE_TEXT_MAP.get(cue_key)

    async def _generate_single_cue(self, key: str, text: str) -> Tuple[str, bytes]:
        """Generate TTS audio for a single cue (fallback path)."""
        client = self._get_client()
        response = await client.audio.speech.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=text,
            instructions=TTS_INSTRUCTIONS,
            response_format=TTS_FORMAT,
            speed=TTS_SPEED,
        )
        audio_bytes = response.read()
        return key, audio_bytes

    @staticmethod
    async def _pcm_to_audio_frames(
        pcm_bytes: bytes,
    ) -> AsyncGenerator[rtc.AudioFrame, None]:
        """Convert raw PCM bytes (24kHz, 16-bit mono) into AudioFrame chunks."""
        bytes_per_sample = 2  # int16
        chunk_bytes = SAMPLES_PER_CHUNK * NUM_CHANNELS * bytes_per_sample
        for offset in range(0, len(pcm_bytes), chunk_bytes):
            chunk = pcm_bytes[offset:offset + chunk_bytes]
            samples = len(chunk) // (NUM_CHANNELS * bytes_per_sample)
            yield rtc.AudioFrame(
                data=chunk,
                sample_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
                samples_per_channel=samples,
            )
