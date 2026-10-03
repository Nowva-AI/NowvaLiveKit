"""Draft the spoken lines for every squat coaching cue with an LLM, for human review.

Writes src/assets/cue_text/cues.json ({cue_key: [line, line, line]}) and a review
page next to it. A person reviews and edits cues.json; generate_cue_audio.py then
speaks exactly those lines. Re-runs only draft keys missing from cues.json, so
reviewed edits survive; --cue re-drafts specific keys.

Usage:
    python scripts/tools/draft_cue_text.py
    python scripts/tools/draft_cue_text.py --cue knees_out_left --cue drive
    python scripts/tools/draft_cue_text.py --model gpt-6-astra
"""

from __future__ import annotations

import argparse
import asyncio
import html
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
load_dotenv(REPO_ROOT / ".env")

from agent.agents.prompts.base_prompt import NOVA_IDENTITY  # noqa: E402
from agent.services.coaching_constants import CUE_TEXT_MAP, TRACKING_LOST_CUE  # noqa: E402
from agent.services.coaching_orchestrator import FIXED_CUE_SUFFIX  # noqa: E402
from biomechanics.coaching.cue_cache import SQUAT_CUES, base_cue_key  # noqa: E402

CUE_TEXT_DIR = REPO_ROOT / "src" / "assets" / "cue_text"
CUES_JSON_PATH = CUE_TEXT_DIR / "cues.json"
REVIEW_PAGE_PATH = CUE_TEXT_DIR / "review.html"

DEFAULT_MODEL = "gpt-6-astra"
VARIANTS_PER_CUE = 3
MAX_CUE_WORDS = 4
MAX_EXPLAIN_WORDS = 16
MAX_CONCURRENT_REQUESTS = 8

# Spoken once when the stance monitor arms — they carry the why and the fix,
# so they get a full sentence instead of a 4-word call.
EXPLAIN_CUE_KEYS = frozenset({"stance_explain", "toe_out_explain"})

SYSTEM_PROMPT = (
    f"{NOVA_IDENTITY} "
    "You are writing the short lines you call out to your athlete during a set of squats. "
    "Calm, direct coach voice — warm, never hype, never cheesy. "
    "External focus: point at the floor, the bar, the feet or a direction, not at muscles. "
    "No jargon a beginner wouldn't know instantly (never 'valgus', 'eccentric', "
    "'concentric', 'dorsiflexion' or 'hinge'). "
    "A voice engine speaks the lines: no emojis, no ALL CAPS, no stage directions. "
    "The athlete may be squatting with no bar at all, so never mention the bar unless the "
    "scenario is about the bar itself."
)

# Base cue key -> the coaching moment the lines must fit. Side variants add
# SIDE_INSTRUCTIONS on top of their base scenario.
CUE_SCENARIOS: dict[str, str] = {
    "knees_out": (
        "The athlete's knees cave inward on the way up out of the squat. Push them back out "
        "over the little toes, like spreading the floor apart with the feet."
    ),
    "chest_up": (
        "The athlete's hips rise faster than their chest coming out of the bottom. Chest and "
        "hips should rise together."
    ),
    "heels_down": (
        "One of the athlete's heels lifts off the floor. Keep the whole foot planted, weight "
        "through the heel and the big toe."
    ),
    "whole_foot": (
        "The athlete's weight is drifting onto their toes. Get them over the whole foot, "
        "feeling the heel."
    ),
    "even_it_out": (
        "The athlete's hips slide to one side during the squat. Stay centered and push the "
        "floor away evenly with both feet."
    ),
    "level_bar": "The bar tilts to one side during the rep. Keep the bar level and push evenly.",
    "deeper": "The athlete isn't reaching their own target depth. Sit a little lower.",
    "square_feet": (
        "The athlete set their feet up uneven — one foot staggered forward or turned out more "
        "than the other. Square the feet up before the next rep."
    ),
    "lockout": "The athlete isn't standing all the way up between reps. Stand tall at the top.",
    "slow_down": "The athlete drops into the bottom too fast. Control the way down.",
    "same_depth": (
        "Later reps are getting shallower than the first ones. Hit the same depth as the "
        "first rep."
    ),
    "drive": "The last rep slowed down a lot. Drive hard out of the bottom.",
    "brace": (
        "The athlete's midsection is going soft under the bar. Big breath and get tight "
        "before the next rep."
    ),
    "stance_explain": (
        "You just corrected your athlete mid-set, and the real cause is their stance being "
        "too narrow. Tell them it's coming from their stance and to step their feet out wider."
    ),
    "stance_wider": (
        "The athlete is standing between reps adjusting their stance, and their feet are "
        "still a bit too close together. Nudge them to widen slightly."
    ),
    "stance_narrower": (
        "The athlete is standing between reps adjusting their stance and has overshot — their "
        "feet are now too wide. Nudge them to bring it in a bit."
    ),
    "toe_out_explain": (
        "You just corrected your athlete mid-set, and the real cause is their toes pointing "
        "too straight ahead. Tell them it's coming from their feet and to turn their toes "
        "out more."
    ),
    "toe_out_more": (
        "The athlete is standing between reps adjusting their feet, and their toes still need "
        "to point out a bit further. Nudge them to turn out more."
    ),
    "toe_out_less": (
        "The athlete is standing between reps adjusting their feet and has overshot — their "
        "toes are turned out too far. Nudge them to ease back."
    ),
    "adjust_good": (
        "The athlete just moved their feet into exactly the position you asked for. Confirm "
        "it and tell them to hold that."
    ),
    "good_rep": "The athlete just finished a solid rep with good form. Let them know that was a good one.",
    "great_depth": "The athlete just hit their depth with control. Acknowledge the depth.",
    "strong": "The athlete just moved their best rep so far — it looked strong. Acknowledge it.",
    "clean": "The athlete just did a rep with nothing to fix. Tell them how clean it was.",
    "perfect": (
        "The athlete just did a flawless rep — depth, position and speed all right. "
        "Acknowledge it."
    ),
}

# Cues the voice agent plays on its own (the pipeline never lists them):
# fix confirmations, built from their fault's scenario, plus these.
AGENT_CUE_SCENARIOS: dict[str, str] = {
    TRACKING_LOST_CUE: (
        "Mid-set, the camera has lost sight of the athlete's legs. Ask them to step back "
        "into view so you can see all of them."
    ),
}

SIDE_INSTRUCTIONS: dict[str, str] = {
    "knees_out": "Only the {side} knee caves. Every line must name the {side} knee.",
    "heels_down": "Only the {side} heel lifts. Every line must name the {side} heel.",
    "even_it_out": (
        "The hips drift to the athlete's {side}. Every line must say they're drifting {side} "
        "and to stay centered."
    ),
    "square_feet": (
        "The {side} foot is the one to move. Every line must name the {side} foot and bring it "
        "even with the other."
    ),
}


def _is_rep_count_key(cue_key: str) -> bool:
    return cue_key.startswith("rep_")


def _load_existing(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


async def _draft_one(
    client: AsyncOpenAI, model: str, cue_key: str, semaphore: asyncio.Semaphore,
) -> list[str]:
    async with semaphore:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_cue_prompt(cue_key)},
            ],
            response_format={"type": "json_object"},
        )
    return parse_variants(response.choices[0].message.content or "")


async def _draft_with_llm(model: str, cue_keys: list[str]) -> dict[str, list[str]]:
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    results = await asyncio.gather(
        *(_draft_one(client, model, key, semaphore) for key in cue_keys),
        return_exceptions=True,
    )
    drafted: dict[str, list[str]] = {}
    for cue_key, result in zip(cue_keys, results):
        if isinstance(result, Exception):
            print(f"  {cue_key:22} FAILED — {result}")
            continue
        drafted[cue_key] = result
        print(f"  {cue_key:22} {' | '.join(result)}")
    return drafted


def cue_keys_to_draft() -> list[str]:
    fix_confirmations = [key for key in CUE_TEXT_MAP if key.endswith(FIXED_CUE_SUFFIX)]
    agent_keys = [key for key in fix_confirmations + list(AGENT_CUE_SCENARIOS) if key not in SQUAT_CUES]
    return list(SQUAT_CUES) + agent_keys


def scenario_for(cue_key: str) -> str:
    if cue_key in AGENT_CUE_SCENARIOS:
        return AGENT_CUE_SCENARIOS[cue_key]
    if cue_key.endswith(FIXED_CUE_SUFFIX):
        fault_scenario = CUE_SCENARIOS[cue_key[:-len(FIXED_CUE_SUFFIX)]]
        return (
            f"{fault_scenario} You cued this earlier in the set; the athlete fixed it and has "
            "held it for two reps. Confirm it briefly and name what they did right — praise, "
            "not another correction."
        )
    base_key = base_cue_key(cue_key)
    scenario = CUE_SCENARIOS[base_key]
    if base_key == cue_key:
        return scenario
    side = cue_key[len(base_key) + 1:]
    return f"{scenario} {SIDE_INSTRUCTIONS[base_key].format(side=side)}"


def build_cue_prompt(cue_key: str) -> str:
    if cue_key in EXPLAIN_CUE_KEYS:
        length_rule = (
            f"each one natural sentence of 10 to {MAX_EXPLAIN_WORDS} words that names the cause "
            "and the fix"
        )
    else:
        length_rule = f"each at most {MAX_CUE_WORDS} words"
    return (
        f"Scenario: {scenario_for(cue_key)}\n"
        f"Write exactly {VARIANTS_PER_CUE} different lines you could call out for this, "
        f"{length_rule}. They should mean the same thing in different words.\n"
        'Reply with JSON only: {"lines": ["...", "...", "..."]}'
    )


def rep_count_lines(cue_key: str) -> list[str]:
    """The rep counts are just the number, so they skip the LLM."""
    return [CUE_TEXT_MAP[cue_key]] * VARIANTS_PER_CUE


def parse_variants(response_text: str) -> list[str]:
    data = json.loads(response_text)
    lines = data.get("lines") if isinstance(data, dict) else None
    if not isinstance(lines, list):
        raise ValueError(f"No 'lines' list in response: {response_text[:200]}")
    cleaned = [" ".join(str(line).split()) for line in lines]
    cleaned = [line for line in cleaned if line]
    if len(cleaned) != VARIANTS_PER_CUE:
        raise ValueError(f"Expected {VARIANTS_PER_CUE} lines, got {len(cleaned)}: {cleaned}")
    return cleaned


def spoken_word_count(line: str) -> int:
    return sum(1 for token in line.split() if any(char.isalnum() for char in token))


def over_length_lines(cue_key: str, lines: list[str]) -> list[str]:
    if _is_rep_count_key(cue_key):
        return []
    limit = MAX_EXPLAIN_WORDS if cue_key in EXPLAIN_CUE_KEYS else MAX_CUE_WORDS
    return [line for line in lines if spoken_word_count(line) > limit]


def merge_cue_lines(
    existing: dict[str, list[str]], drafted: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Drafted lines win; keys keep SQUAT_CUES order, extra hand-added keys go last."""
    merged = {**existing, **drafted}
    ordered = [key for key in cue_keys_to_draft() if key in merged]
    ordered += [key for key in merged if key not in ordered]
    return {key: merged[key] for key in ordered}


def build_review_page(cue_lines: dict[str, list[str]]) -> str:
    rows = []
    for cue_key, lines in cue_lines.items():
        too_long = set(over_length_lines(cue_key, lines))
        scenario = "" if _is_rep_count_key(cue_key) else scenario_for(cue_key)
        cells = "".join(
            f'<td class="{"long" if line in too_long else ""}">{html.escape(line)}'
            f'<span class="count">{spoken_word_count(line)}w</span></td>'
            for line in lines
        )
        rows.append(
            f'<tr><td class="key">{html.escape(cue_key)}</td>'
            f'<td class="scenario">{html.escape(scenario)}</td>{cells}</tr>'
        )
    line_headers = "".join(f"<th>Line {index + 1}</th>" for index in range(VARIANTS_PER_CUE))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cue Text Review</title>
<style>
  :root {{ --bg: #fafafa; --fg: #1a1a1a; --muted: #666; --line: #ddd; --warn: #b42318; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg: #111; --fg: #eee; --muted: #999; --line: #333; --warn: #f97066; }}
  }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0;
         padding: 16px; background: var(--bg); color: var(--fg); }}
  p {{ color: var(--muted); }}
  .wrap {{ overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ padding: 8px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
  .key {{ font-weight: 600; white-space: nowrap; }}
  .scenario {{ color: var(--muted); font-size: 0.85rem; max-width: 420px; }}
  .count {{ color: var(--muted); font-size: 0.75rem; margin-left: 6px; }}
  .long {{ color: var(--warn); }}
</style>
</head>
<body>
<h1>Cue Text Review</h1>
<p>Edit src/assets/cue_text/cues.json, then run generate_cue_audio.py. Red lines are over the
word limit ({MAX_CUE_WORDS} words; {MAX_EXPLAIN_WORDS} for the stance and toe-out explanations).</p>
<div class="wrap">
<table>
<thead><tr><th>Cue</th><th>Scenario</th>{line_headers}</tr></thead>
<tbody>
{chr(10).join(rows)}
</tbody>
</table>
</div>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI chat model (default {DEFAULT_MODEL})")
    parser.add_argument("--cue", action="append", dest="cues", help="re-draft this cue key (repeatable)")
    args = parser.parse_args()

    existing = _load_existing(CUES_JSON_PATH)
    all_keys = cue_keys_to_draft()
    if args.cues:
        unknown = [key for key in args.cues if key not in all_keys]
        if unknown:
            print(f"Unknown cue key(s): {', '.join(unknown)}")
            return 1
        keys = args.cues
    else:
        keys = [key for key in all_keys if key not in existing]
    if not keys:
        print(f"Every cue already has lines in {CUES_JSON_PATH} — pass --cue to re-draft one.")
        return 0

    drafted = {key: rep_count_lines(key) for key in keys if _is_rep_count_key(key)}
    llm_keys = [key for key in keys if not _is_rep_count_key(key)]
    if llm_keys:
        if not os.getenv("OPENAI_API_KEY"):
            print("OPENAI_API_KEY is not set — add it to .env")
            return 1
        print(f"Drafting {len(llm_keys)} cue(s) with {args.model}\n")
        drafted.update(asyncio.run(_draft_with_llm(args.model, llm_keys)))

    cue_lines = merge_cue_lines(existing, drafted)
    CUE_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    CUES_JSON_PATH.write_text(json.dumps(cue_lines, indent=2, ensure_ascii=False) + "\n")
    REVIEW_PAGE_PATH.write_text(build_review_page(cue_lines))

    failed = len(llm_keys) - sum(1 for key in llm_keys if key in drafted)
    long_lines = sum(len(over_length_lines(key, lines)) for key, lines in cue_lines.items())
    print(f"\nWrote {CUES_JSON_PATH} ({len(cue_lines)} cues)")
    print(f"Review page: file://{REVIEW_PAGE_PATH}")
    if long_lines:
        print(f"{long_lines} line(s) over the word limit — edit them before generating audio.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
