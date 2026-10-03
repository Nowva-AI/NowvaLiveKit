#!/usr/bin/env python3
"""Score the agent replies captured by ttfa_bench against Nova's spoken-output contract.

Latency work is only a win if the replies still sound like Nova. The
deterministic rules below encode the hard parts of BASE_PROMPT's TTS contract —
the things that are audibly wrong when a model breaks them. The optional LLM
judge covers persona and tone, which rules cannot see.

Usage:
    python scripts/benchmarks/reply_quality.py benchmarks/results/ttfa/*.json
    python scripts/benchmarks/reply_quality.py --judge <result.json>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
load_dotenv(PROJECT_ROOT / ".env")

MAX_SPOKEN_WORDS = 60

_MARKDOWN_RE = re.compile(r"\*\*|`|^\s{0,3}#{1,6}\s|^\s*[-*•]\s", re.MULTILINE)
_EMOJI_RE = re.compile("[\U0001f000-\U0001faff☀-➿]")
_UNIT_SYMBOL_RE = re.compile(r"[°%×]|(?<=\d)\s*/\s*(?=100\b|10\b)|(?<=\d)x(?=\d)|->|→")
_ALL_CAPS_RE = re.compile(r"\b[A-Z]{3,}\b")
_STAGE_DIRECTION_RE = re.compile(r"[\(\[][^)\]]*[)\]]")
_ALLOWED_CAPS = {"RPE", "AMRAP", "PR", "RIR", "OK"}
_ALLOWED_BRACKETS = {"[laughter]", "[laughs]"}


def check_reply(reply: str, user_name: str = "Alex") -> list[str]:
    violations: list[str] = []
    if not reply.strip():
        return ["empty reply"]

    if _MARKDOWN_RE.search(reply):
        violations.append("markdown")
    if _EMOJI_RE.search(reply):
        violations.append("emoji")
    if _UNIT_SYMBOL_RE.search(reply):
        violations.append("unit symbol or written shorthand")

    caps = [w for w in _ALL_CAPS_RE.findall(reply) if w not in _ALLOWED_CAPS]
    if caps:
        violations.append(f"all-caps word(s): {caps}")

    brackets = [
        m for m in _STAGE_DIRECTION_RE.findall(reply)
        if m.lower() not in _ALLOWED_BRACKETS
    ]
    if brackets:
        violations.append(f"stage direction: {brackets}")

    words = reply.split()
    if len(words) > MAX_SPOKEN_WORDS:
        violations.append(f"too long ({len(words)} words > {MAX_SPOKEN_WORDS})")

    if user_name and reply.lower().count(user_name.lower()) > 1:
        violations.append("repeats the user's name")

    return violations


JUDGE_SYSTEM = """You rate single replies from Nova, the voice coach built into a smart squat rack.

Nova's contract: calm, direct, warm. Never performs enthusiasm. 1-2 short sentences, one
question at a time. Contractions. Everything is spoken aloud, so no markdown, no lists, no
symbols, no stage directions. Almost never uses the user's name. Humor is dry and rare.
Safety overrides everything — no jokes near pain, and pain is never pushed through.

Score each reply 1-5 on:
  persona   — does this sound like that coach, not a generic assistant?
  brevity   — is it as short as the moment allows?
  spoken    — would it sound right read aloud, with nothing unreadable in it?
  useful    — does it actually answer or move the conversation forward?

Return JSON only: {"persona":n,"brevity":n,"spoken":n,"useful":n,"note":"<=12 words"}"""


def judge_reply(client, model: str, prompt: str, reply: str) -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": f"User said: {prompt!r}\nNova replied: {reply!r}"},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def score_file(path: Path, judge: bool, judge_model: str) -> dict:
    data = json.loads(path.read_text())
    label = data["summary"]["label"]
    trials = [t for t in data["trials"] if t.get("reply")]

    rule_hits: list[tuple[str, list[str]]] = []
    for t in trials:
        violations = check_reply(t["reply"])
        if violations:
            rule_hits.append((t["reply"], violations))

    result = {
        "label": label,
        "replies": len(trials),
        "clean": len(trials) - len(rule_hits),
        "violations": rule_hits,
        "empty_replies": sum(1 for t in data["trials"] if not t.get("reply")),
        "false_cutoffs": data["summary"].get("false_cutoffs", 0),
    }

    if judge and trials:
        from openai import OpenAI

        client = OpenAI()
        scores = []
        for t in trials:
            try:
                scores.append(judge_reply(client, judge_model, t.get("transcript", ""), t["reply"]))
            except Exception as exc:  # noqa: BLE001 — a judge failure must not sink the run
                print(f"    judge failed: {exc}", file=sys.stderr)
        if scores:
            result["judge"] = {
                key: round(sum(s[key] for s in scores) / len(scores), 2)
                for key in ("persona", "brevity", "spoken", "useful")
            }
            result["judge_notes"] = [s.get("note", "") for s in scores]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--judge", action="store_true", help="also run the LLM persona judge")
    parser.add_argument("--judge-model", default="gpt-5.4-mini")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"\n  {'arm':30s} {'replies':>8s} {'clean':>7s} {'empty':>7s} {'cutoff':>7s}   judge (persona/brevity/spoken/useful)")
    print("  " + "-" * 104)
    for path in args.results:
        r = score_file(path, args.judge, args.judge_model)
        j = r.get("judge")
        judge_str = (
            f"{j['persona']:.1f} / {j['brevity']:.1f} / {j['spoken']:.1f} / {j['useful']:.1f}"
            if j else "-"
        )
        print(
            f"  {r['label']:30s} {r['replies']:8d} {r['clean']:7d} {r['empty_replies']:7d} "
            f"{r['false_cutoffs']:7d}   {judge_str}"
        )
        for reply, violations in r["violations"]:
            print(f"      ! {', '.join(violations)}")
            print(f"        {reply[:100]!r}")
        if args.verbose and r.get("judge_notes"):
            for note in r["judge_notes"]:
                print(f"      · {note}")
    print()


if __name__ == "__main__":
    main()
