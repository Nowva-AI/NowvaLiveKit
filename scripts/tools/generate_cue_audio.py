"""Speak the reviewed coaching cue lines with Cartesia Sonic-3, in Nova's live voice.

Reads src/assets/cue_text/cues.json (drafted by draft_cue_text.py, then reviewed by a
person) and writes src/assets/cues/wav/{cue_key}_{variant}.wav: 24 kHz mono 16-bit PCM,
the format AudioCueService loads. A manifest beside the clips records the line and voice
each clip was spoken from, so re-runs only re-speak lines that changed. Finishes with a
review page (src/assets/cues/review.html) to listen to every clip.

Usage:
    python scripts/tools/generate_cue_audio.py                 # new or edited lines only
    python scripts/tools/generate_cue_audio.py --force         # re-speak everything
    python scripts/tools/generate_cue_audio.py --cue knees_out_left
    python scripts/tools/generate_cue_audio.py --variants 3
"""

from __future__ import annotations

import argparse
import asyncio
import html
import io
import json
import os
import sys
import wave
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(REPO_ROOT / ".env")

CUES_JSON_PATH = REPO_ROOT / "src" / "assets" / "cue_text" / "cues.json"
CUES_DIR = REPO_ROOT / "src" / "assets" / "cues"
CUES_WAV_DIR = CUES_DIR / "wav"
MANIFEST_PATH = CUES_DIR / "manifest.json"
REVIEW_PAGE_PATH = CUES_DIR / "review.html"

CARTESIA_URL = "https://api.cartesia.ai/tts/bytes"
CARTESIA_API_VERSION = "2025-04-16"
TTS_MODEL = "sonic-3"
# Same default as the live agent (voice_agent.py), so cached cues and live
# speech are one voice.
DEFAULT_VOICE_ID = "3e39e9a5-585c-4f5f-bac6-5e4905c51095"
DEFAULT_VARIANTS = 3
REQUEST_TIMEOUT_S = 60

# Must match agent.services.audio_cue_service: it reads the clips with this
# format hardcoded, so anything else plays as noise.
SAMPLE_RATE = 24000
NUM_CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2


async def _synthesize(session: aiohttp.ClientSession, api_key: str, payload: dict) -> bytes:
    async with session.post(
        CARTESIA_URL,
        headers={"X-API-Key": api_key, "Cartesia-Version": CARTESIA_API_VERSION},
        json=payload,
        timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S),
    ) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {(await response.text())[:300]}")
        return await response.read()


def _load_manifest() -> dict[str, dict]:
    if not MANIFEST_PATH.exists():
        return {}
    return json.loads(MANIFEST_PATH.read_text())


def load_cue_lines(path: Path) -> dict[str, list[str]]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must map cue keys to lists of lines")
    for cue_key, lines in data.items():
        if not isinstance(lines, list) or not lines or not all(
            isinstance(line, str) and line.strip() for line in lines
        ):
            raise ValueError(f"{path}: '{cue_key}' needs a non-empty list of non-empty lines")
    return data


def wav_filename(cue_key: str, variant: int) -> str:
    return f"{cue_key}_{variant}.wav"


def parse_wav_filename(filename: str) -> tuple[str, int] | None:
    """Mirror of AudioCueService._load_from_disk: key, then the number after the last underscore."""
    if not filename.endswith(".wav"):
        return None
    stem = filename[:-len(".wav")]
    cue_key, separator, variant = stem.rpartition("_")
    if not separator or not cue_key or not variant.isdigit():
        return None
    return cue_key, int(variant)


def stale_variant_files(cue_key: str, variant_count: int, filenames: list[str]) -> list[str]:
    """Clips of this key beyond the current variant count — e.g. old voice takes
    that would otherwise keep playing at random alongside the new ones."""
    stale = []
    for filename in filenames:
        parsed = parse_wav_filename(filename)
        if parsed is not None and parsed[0] == cue_key and parsed[1] >= variant_count:
            stale.append(filename)
    return sorted(stale)


def needs_synthesis(manifest_entry: dict | None, text: str, voice_id: str, force: bool) -> bool:
    if force or manifest_entry is None:
        return True
    return (
        manifest_entry.get("text") != text
        or manifest_entry.get("voice_id") != voice_id
        or manifest_entry.get("model") != TTS_MODEL
    )


def build_tts_payload(text: str, voice_id: str) -> dict:
    return {
        "model_id": TTS_MODEL,
        "transcript": text,
        "voice": {"mode": "id", "id": voice_id},
        "output_format": {
            "container": "raw",
            "encoding": "pcm_s16le",
            "sample_rate": SAMPLE_RATE,
        },
        "language": "en",
    }


def pcm_to_wav_bytes(pcm_bytes: bytes) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(NUM_CHANNELS)
        wav_file.setsampwidth(SAMPLE_WIDTH_BYTES)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm_bytes)
    return buffer.getvalue()


def build_review_page(
    cue_lines: dict[str, list[str]], variant_count: int, voice_id: str, existing_files: set[str],
) -> str:
    rows = []
    for cue_key, lines in cue_lines.items():
        cells = []
        for variant, line in enumerate(lines[:variant_count]):
            filename = wav_filename(cue_key, variant)
            if filename in existing_files:
                player = (
                    f'<button onclick="playClip(\'wav/{html.escape(filename)}\', this)">'
                    f'&#9654;</button>'
                )
            else:
                player = '<span class="missing">missing</span>'
            cells.append(f"<td>{player} {html.escape(line)}</td>")
        rows.append(f'<tr><td class="key">{html.escape(cue_key)}</td>{"".join(cells)}</tr>')
    headers = "".join(f"<th>Take {variant + 1}</th>" for variant in range(variant_count))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cue Audio Review</title>
<style>
  :root {{ --bg: #fafafa; --fg: #1a1a1a; --muted: #666; --line: #ddd; --accent: #1e7a46; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg: #0a0a0a; --fg: #e0e0e0; --muted: #888; --line: #222; --accent: #4ade80; }}
  }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0;
         padding: 16px; background: var(--bg); color: var(--fg); }}
  p {{ color: var(--muted); }}
  .wrap {{ overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ padding: 8px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
  .key {{ font-weight: 600; white-space: nowrap; }}
  .missing {{ color: var(--muted); }}
  button {{ background: none; color: var(--accent); border: 1px solid var(--accent); border-radius: 4px;
            padding: 2px 8px; cursor: pointer; }}
  button.playing {{ background: var(--accent); color: var(--bg); }}
</style>
</head>
<body>
<h1>Cue Audio Review</h1>
<p>Cartesia {TTS_MODEL} · voice {html.escape(voice_id)} · {len(cue_lines)} cues · lines from
src/assets/cue_text/cues.json</p>
<div class="wrap">
<table>
<thead><tr><th>Cue</th>{headers}</tr></thead>
<tbody>
{chr(10).join(rows)}
</tbody>
</table>
</div>
<script>
let currentAudio = null;
let currentButton = null;
function playClip(path, button) {{
  if (currentAudio) {{ currentAudio.pause(); }}
  if (currentButton) {{ currentButton.classList.remove("playing"); }}
  currentAudio = new Audio(path);
  currentButton = button;
  button.classList.add("playing");
  currentAudio.onended = () => button.classList.remove("playing");
  currentAudio.play();
}}
</script>
</body>
</html>
"""


async def generate(
    cue_lines: dict[str, list[str]], cue_keys: list[str], variant_count: int,
    voice_id: str, api_key: str, force: bool,
) -> int:
    CUES_WAV_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest()
    existing = [path.name for path in CUES_WAV_DIR.glob("*.wav")]
    written = skipped = failed = removed = 0
    print(f"Cartesia {TTS_MODEL} · voice {voice_id} · {SAMPLE_RATE} Hz mono → {CUES_WAV_DIR}\n")

    try:
        async with aiohttp.ClientSession() as session:
            for cue_key in cue_keys:
                for filename in stale_variant_files(cue_key, variant_count, existing):
                    (CUES_WAV_DIR / filename).unlink()
                    manifest.pop(filename, None)
                    removed += 1
                for variant, text in enumerate(cue_lines[cue_key][:variant_count]):
                    filename = wav_filename(cue_key, variant)
                    path = CUES_WAV_DIR / filename
                    if path.exists() and not needs_synthesis(manifest.get(filename), text, voice_id, force):
                        skipped += 1
                        continue
                    try:
                        pcm_bytes = await _synthesize(session, api_key, build_tts_payload(text, voice_id))
                    except Exception as error:
                        print(f"  {filename:30} FAILED — {error}")
                        failed += 1
                        continue
                    path.write_bytes(pcm_to_wav_bytes(pcm_bytes))
                    manifest[filename] = {"text": text, "voice_id": voice_id, "model": TTS_MODEL}
                    duration_s = len(pcm_bytes) / (SAMPLE_RATE * NUM_CHANNELS * SAMPLE_WIDTH_BYTES)
                    print(f"  {filename:30} {duration_s:4.2f}s  \"{text}\"")
                    written += 1
    finally:
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    print(f"\n{written} written, {skipped} unchanged, {removed} stale removed, {failed} failed")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", type=int, default=DEFAULT_VARIANTS,
                        help=f"clips per cue (default {DEFAULT_VARIANTS})")
    parser.add_argument("--force", action="store_true", help="re-speak lines that are unchanged")
    parser.add_argument("--cue", action="append", dest="cues", help="only this cue key (repeatable)")
    args = parser.parse_args()

    if not CUES_JSON_PATH.exists():
        print(f"{CUES_JSON_PATH} not found — run scripts/tools/draft_cue_text.py and review it first")
        return 1
    cue_lines = load_cue_lines(CUES_JSON_PATH)
    cue_keys = args.cues or list(cue_lines)
    unknown = [key for key in cue_keys if key not in cue_lines]
    if unknown:
        print(f"Not in {CUES_JSON_PATH.name}: {', '.join(unknown)}")
        return 1

    api_key = os.getenv("CARTESIA_API_KEY")
    if not api_key:
        print("CARTESIA_API_KEY is not set — add it to .env")
        return 1
    voice_id = os.getenv("CARTESIA_VOICE_ID", DEFAULT_VOICE_ID)

    status = asyncio.run(generate(cue_lines, cue_keys, args.variants, voice_id, api_key, args.force))

    existing_files = {path.name for path in CUES_WAV_DIR.glob("*.wav")}
    REVIEW_PAGE_PATH.write_text(build_review_page(cue_lines, args.variants, voice_id, existing_files))
    print(f"Review page: file://{REVIEW_PAGE_PATH}")
    return status


if __name__ == "__main__":
    sys.exit(main())
