#!/usr/bin/env python3
"""Score candidate LLMs for Nova's voice on the three axes that matter together.

A conversational model for the rack has to be fast, do what it says, and sound
like Nova. Models that win one axis routinely lose another, so every candidate
runs the same cases against the real MainMenuAgent prompt and tools:

  speed    time to first token (text or tool call) with the full prompt + tools
  tools    right tool, right arguments, and no "phantom actions" — saying
           "starting that now" while calling nothing
  persona  banter, teasing, small talk and a pain report, scored blind by two
           judges from different model families

Text only: no STT or TTS is involved, so a run costs LLM tokens and nothing else.

Usage:
    python scripts/benchmarks/llm_voice_eval.py --list
    python scripts/benchmarks/llm_voice_eval.py --quick
    python scripts/benchmarks/llm_voice_eval.py --candidates luna,groq-120b,haiku-4.5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
load_dotenv(PROJECT_ROOT / ".env")

from livekit.agents import llm as lkllm  # noqa: E402
from livekit.agents.utils import http_context  # noqa: E402
from livekit.plugins import google as lk_google  # noqa: E402
from livekit.plugins import openai as lk_openai  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402
from openai.types import Reasoning  # noqa: E402

from agent.services.chat_template_compat import normalize_for_strict_template  # noqa: E402
from reply_quality import check_reply  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results" / "llm_eval"
BENCH_USER_ID = "00000000-0000-4000-8000-0000000000bc"
GROQ_URL = "https://api.groq.com/openai/v1"
ANTHROPIC_COMPAT_URL = "https://api.anthropic.com/v1/"
CEREBRAS_URL = "https://api.cerebras.ai/v1"
CALL_TIMEOUT_S = 45.0

GREETING = "Hey again, what would you like to work on today?"
# Nova speaks ~25 scripted lines through generate_reply(instructions=...), which sends the
# direction as a trailing system message with no user turn. Strict chat templates 400 on it.
SCRIPTED_DIRECTION = (
    "The user is back at the main menu. Greet them back with one short, natural line and an "
    "open question about what they want to do."
)

# (user line, expected tool or None, arguments that must match)
TOOL_CASES: list[tuple[str, str | None, dict[str, Any]]] = [
    ("Let's train.", "start_workout", {}),
    ("Let's start my workout.", "start_workout", {}),
    (
        "I wanna do a quick exercise. Three sets of five squats with one thirty five.",
        "start_quick_exercise",
        {"sets": 3, "reps": 5, "weight": 135},
    ),
    ("I just wanna squat.", "start_quick_exercise", {}),
    ("How have I been doing lately?", "view_progress", {}),
    ("Can you show me what's on my schedule this week?", "view_schedule", {}),
    ("Build me a six week program to get stronger legs.", "create_program", {}),
    ("How do I sound right now?", "how_do_i_sound", {}),
    ("Should I be going all the way down on my squats?", None, {}),
    ("How long should I rest between sets?", None, {}),
]
QUICK_TOOL_CASES = [TOOL_CASES[0], TOOL_CASES[2], TOOL_CASES[4], TOOL_CASES[8]]

_JOKE = "Why did the squat go to therapy? It couldn't handle the pressure and needed a good support."
# name -> conversation after the greeting; the last line is the user's.
PERSONA_PROBES: dict[str, list[tuple[str, str]]] = {
    "banter": [
        ("user", "Can you tell me a quick joke?"),
        ("assistant", _JOKE),
        ("user", "That was a terrible ass joke, man."),
    ],
    "teasing": [("user", "Honestly, I think you just like watching me suffer.")],
    "small_talk": [("user", "Man, my legs are still wrecked from Tuesday.")],
    "pain": [("user", "My left knee's been hurting every time I squat lately.")],
}
QUICK_PROBES = ("banter", "pain")

PHANTOM_RE = re.compile(
    r"\b(starting|getting (that|it|you) (started|going|set)|setting (that|it|you) up|kicking (that|it) off|"
    r"pulling (that|it|up)|let me (check|pull|grab|look|get|start|set)|one sec|firing (that|it) up|on it)\b",
    re.IGNORECASE,
)

JUDGE_MODELS = (
    ("claude-sonnet-5", ANTHROPIC_COMPAT_URL, "ANTHROPIC_API_KEY"),
    ("gpt-5.4", None, "OPENAI_API_KEY"),
)
JUDGE_SYSTEM = """You score single spoken replies from Nova, the voice coach built into a smart squat rack.

Nova is an experienced strength coach: calm, direct, warm, never performs enthusiasm. She is a
person, not an assistant. Dry humor that comes from THIS conversation; if the user banters or
teases, she plays along in one specific line — she does not apologize for a joke, and she does
not steer back to a menu of options in the same breath. 1-2 short spoken sentences, at most one
question. If the user mentions pain: no jokes at all, calm, asks what's wrong, never pushes
through it.

Score 1-5:
  persona  1 = generic customer-service assistant ("Got it. What would you like to work on?")
           3 = pleasant but interchangeable   5 = unmistakably that coach, specific to the moment
  fit      did she respond to what was actually said, with the right register for it?
           (a joke after a pain report, or a menu redirect after banter, is a 1-2)

Return JSON only: {"persona": n, "fit": n}"""


@dataclass
class Candidate:
    name: str
    build: Callable[[], Any]
    requires: str
    note: str = ""


def _openai_chat(model: str, **kwargs: Any) -> Callable[[], Any]:
    return lambda: lk_openai.LLM(model=model, **kwargs)


def _openai_responses(model: str, **kwargs: Any) -> Callable[[], Any]:
    return lambda: lk_openai.responses.LLM(model=model, reasoning=Reasoning(effort="none"), **kwargs)


def _compatible(model: str, base_url: str, key_env: str, **kwargs: Any) -> Callable[[], Any]:
    return lambda: lk_openai.LLM(model=model, api_key=os.environ[key_env], base_url=base_url, **kwargs)


def _gemini(model: str, thinking: dict[str, Any]) -> Callable[[], Any]:
    return lambda: lk_google.LLM(model=model, thinking_config=thinking)


CANDIDATES: list[Candidate] = [
    # controls — already measured by hand
    Candidate("luna", _openai_responses("gpt-5.6-luna"), "OPENAI_API_KEY", "shipped"),
    Candidate("groq-120b", _compatible("openai/gpt-oss-120b", GROQ_URL, "GROQ_API_KEY"), "GROQ_API_KEY", "tuned"),
    Candidate("groq-qwen3.8", _compatible("qwen/qwen3.8-27b", GROQ_URL, "GROQ_API_KEY", reasoning_effort="none"), "GROQ_API_KEY"),
    # anthropic, through its OpenAI-compatible endpoint
    Candidate("haiku-4.5", _compatible("claude-haiku-4-5-20251001", ANTHROPIC_COMPAT_URL, "ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY"),
    Candidate("sonnet-5", _compatible("claude-sonnet-5", ANTHROPIC_COMPAT_URL, "ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY"),
    # openai non-reasoning chat models
    Candidate("gpt-4.1", _openai_chat("gpt-4.1"), "OPENAI_API_KEY"),
    Candidate("gpt-4.1-mini", _openai_chat("gpt-4.1-mini"), "OPENAI_API_KEY"),
    Candidate("gpt-5.3-chat", _openai_chat("gpt-5.3-chat-latest"), "OPENAI_API_KEY"),
    Candidate("gpt-5.2-chat", _openai_chat("gpt-5.2-chat-latest"), "OPENAI_API_KEY"),
    # luna's siblings
    Candidate("gpt-5.6-sol", _openai_responses("gpt-5.6-sol"), "OPENAI_API_KEY"),
    Candidate("gpt-5.6-terra", _openai_responses("gpt-5.6-terra"), "OPENAI_API_KEY"),
    Candidate("gpt-6-astra", _openai_responses("gpt-6-astra"), "OPENAI_API_KEY"),
    # same persona models, bought faster: OpenAI's priority tier trades price for latency
    Candidate("luna-priority", _openai_responses("gpt-5.6-luna", service_tier="priority"), "OPENAI_API_KEY"),
    Candidate("terra-priority", _openai_responses("gpt-5.6-terra", service_tier="priority"), "OPENAI_API_KEY"),
    Candidate("gpt-4.1-priority", _openai_chat("gpt-4.1", service_tier="priority"), "OPENAI_API_KEY"),
    Candidate("gpt-5.4-mini", _openai_responses("gpt-5.4-mini"), "OPENAI_API_KEY"),
    Candidate("gpt-5.4", _openai_responses("gpt-5.4"), "OPENAI_API_KEY"),
    # gemini flash line
    Candidate("gemini-2.5-flash", _gemini("gemini-2.5-flash", {"thinking_budget": 0}), "GOOGLE_API_KEY"),
    Candidate("gemini-3.5-flash", _gemini("gemini-3.5-flash", {"thinking_level": "minimal"}), "GOOGLE_API_KEY"),
    Candidate("gemini-3.5-flash-lite", _gemini("gemini-3.5-flash-lite", {"thinking_level": "minimal"}), "GOOGLE_API_KEY"),
    Candidate("gemini-3.8-flash", _gemini("gemini-3.8-flash", {"thinking_level": "low"}), "GOOGLE_API_KEY"),
    # needs a key you do not have yet; listed so a new key is a one-word test
    Candidate("cerebras-120b", _compatible("gpt-oss-120b", CEREBRAS_URL, "CEREBRAS_API_KEY"), "CEREBRAS_API_KEY"),
    # Research lead: Qwen's phantom actions may be Groq's preview hosting, not the model —
    # independent voice evals score it 98% on tool turns elsewhere.
    Candidate(
        "cerebras-qwen3.8",
        _compatible("qwen-3.8-27b", CEREBRAS_URL, "CEREBRAS_API_KEY", reasoning_effort="none"),
        "CEREBRAS_API_KEY",
    ),
]


@dataclass
class CandidateResult:
    name: str
    note: str = ""
    error: str | None = None
    ttfts_ms: list[float] = field(default_factory=list)
    tool_total: int = 0
    tool_correct: int = 0
    phantom_actions: int = 0
    unwanted_tool_calls: int = 0
    tool_failures: list[str] = field(default_factory=list)
    contract_violations: int = 0
    replies: dict[str, list[str]] = field(default_factory=dict)
    persona: float | None = None
    fit: float | None = None
    scripted: str = "-"

    @property
    def ttft_p50(self) -> float | None:
        return statistics.median(self.ttfts_ms) if self.ttfts_ms else None

    @property
    def ttft_max(self) -> float | None:
        return max(self.ttfts_ms) if self.ttfts_ms else None


def _build_agent() -> Any:
    from agent.agents.main_menu_agent import MainMenuAgent
    from agent.agents.shared.userdata import UserData
    from agent.core.agent_state import AgentState

    state = AgentState(user_id=BENCH_USER_ID, state_dir=tempfile.mkdtemp())
    state.state["mode"] = "main_menu"
    state.state["user"]["first_time_main_menu"] = False
    return MainMenuAgent(state=state, userdata=UserData(state=state, room=None))


_min_interval_s = 0.0
_last_call_at = 0.0


def _conversation(agent: Any, turns: list[tuple[str, str]]) -> lkllm.ChatContext:
    ctx = lkllm.ChatContext.empty()
    ctx.add_message(role="system", content=agent.instructions)
    ctx.add_message(role="assistant", content=GREETING)
    for role, text in turns:
        ctx.add_message(role=role, content=text)
    return ctx


async def _ask(
    model: Any, agent: Any, turns: list[tuple[str, str]], ctx: lkllm.ChatContext | None = None
) -> tuple[str, list[tuple[str, dict]], float | None]:
    global _last_call_at
    wait = _last_call_at + _min_interval_s - time.monotonic()
    if wait > 0:
        await asyncio.sleep(wait)
    _last_call_at = time.monotonic()
    print(".", end="", flush=True)

    if ctx is None:
        ctx = _conversation(agent, turns)

    parts: list[str] = []
    calls: list[tuple[str, dict]] = []
    first_ms: float | None = None
    started = time.monotonic()
    async with model.chat(chat_ctx=ctx, tools=agent.tools) as stream:
        async for chunk in stream:
            delta = chunk.delta
            if delta is None:
                continue
            if first_ms is None and (delta.content or delta.tool_calls):
                first_ms = (time.monotonic() - started) * 1000
            if delta.content:
                parts.append(delta.content)
            for call in delta.tool_calls or []:
                try:
                    arguments = json.loads(call.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {"_unparseable": call.arguments}
                calls.append((call.name, arguments))
    return "".join(parts).strip(), calls, first_ms


async def _judge(client: AsyncOpenAI, model: str, user_line: str, reply: str) -> dict[str, float] | None:
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": f"User said: {user_line!r}\nNova replied: {reply!r}"},
            ],
            max_completion_tokens=200,
        )
        match = re.search(r"\{.*\}", response.choices[0].message.content or "", re.DOTALL)
        scores = json.loads(match.group(0)) if match else None
        return {"persona": float(scores["persona"]), "fit": float(scores["fit"])} if scores else None
    except Exception as exc:  # noqa: BLE001 — one judge failing must not sink a run
        logging.getLogger("llm_eval").debug("judge %s failed: %s", model, exc)
        return None


async def _check_scripted_speech(model: Any, agent: Any) -> str:
    """Can it speak a scripted line as LiveKit sends one? 'ok', 'reshape' (needs chat_template_compat) or 'FAIL'."""
    raw = lkllm.ChatContext.empty()
    raw.add_message(role="system", content=agent.instructions)
    raw.add_message(role="system", content=SCRIPTED_DIRECTION)
    for label, ctx in (("ok", raw), ("reshape", normalize_for_strict_template(raw))):
        try:
            text, _, _ = await asyncio.wait_for(_ask(model, agent, [], ctx=ctx), CALL_TIMEOUT_S)
            if text:
                return label
        except Exception:  # noqa: BLE001 — a 400 here is the finding, not a crash
            continue
    return "FAIL"


async def evaluate(candidate: Candidate, agent: Any, judges: list[tuple[AsyncOpenAI, str]], args: argparse.Namespace) -> CandidateResult:
    result = CandidateResult(name=candidate.name, note=candidate.note)
    if not os.getenv(candidate.requires):
        result.error = f"no {candidate.requires}"
        return result
    try:
        model = candidate.build()
    except Exception as exc:  # noqa: BLE001
        result.error = f"build failed: {str(exc)[:80]}"
        return result

    tool_cases = QUICK_TOOL_CASES if args.quick else TOOL_CASES
    probes = {k: v for k, v in PERSONA_PROBES.items() if not args.quick or k in QUICK_PROBES}
    repeats = 1 if args.quick else args.repeats

    try:
        # One throwaway call so connection setup and prompt-cache priming never count.
        await asyncio.wait_for(_ask(model, agent, [("user", "Hey.")]), CALL_TIMEOUT_S)
        result.scripted = await _check_scripted_speech(model, agent)

        for user_line, expected, required_args in tool_cases:
            for _ in range(repeats):
                text, calls, ttft = await asyncio.wait_for(_ask(model, agent, [("user", user_line)]), CALL_TIMEOUT_S)
                if ttft is not None:
                    result.ttfts_ms.append(ttft)
                called = [name for name, _ in calls]
                if expected is None:
                    result.unwanted_tool_calls += bool(calls)
                    if text:
                        result.contract_violations += bool(check_reply(text))
                    continue
                result.tool_total += 1
                arguments = next((a for name, a in calls if name == expected), {})
                correct = expected in called and all(arguments.get(k) == v for k, v in required_args.items())
                result.tool_correct += correct
                if not correct:
                    phantom = not calls and bool(PHANTOM_RE.search(text))
                    result.phantom_actions += phantom
                    kind = "PHANTOM" if phantom else ("wrong args" if expected in called else f"called {called or 'nothing'}")
                    result.tool_failures.append(f"{user_line[:38]!r}: {kind} — said {text[:60]!r}")

        judged: list[dict[str, float]] = []
        for probe, turns in probes.items():
            for _ in range(repeats):
                text, calls, ttft = await asyncio.wait_for(_ask(model, agent, turns), CALL_TIMEOUT_S)
                if ttft is not None:
                    result.ttfts_ms.append(ttft)
                result.replies.setdefault(probe, []).append(text or f"[tool: {[c[0] for c in calls]}]")
                if not text:
                    continue
                result.contract_violations += bool(check_reply(text))
                scores = await asyncio.gather(*(_judge(c, m, turns[-1][1], text) for c, m in judges))
                judged += [s for s in scores if s]
        if judged:
            result.persona = round(statistics.mean(s["persona"] for s in judged), 2)
            result.fit = round(statistics.mean(s["fit"] for s in judged), 2)
    except asyncio.TimeoutError:
        result.error = f"timed out (> {CALL_TIMEOUT_S:.0f}s on one call)"
    except Exception as exc:  # noqa: BLE001
        result.error = str(exc).replace("\n", " ")[:110]
    finally:
        try:
            await model.aclose()
        except Exception:  # noqa: BLE001
            pass
    return result


def _print_table(results: list[CandidateResult]) -> None:
    def rank_key(r: CandidateResult) -> tuple:
        return (r.error is not None, -(r.persona or 0))

    print(
        f"\n  {'candidate':24s}{'ttft p50':>9s}{'max':>7s}{'tools':>8s}{'phantom':>8s}"
        f"{'persona':>8s}{'fit':>6s}{'contract':>9s}{'scripted':>10s}"
    )
    print("  " + "-" * 89)
    for r in sorted(results, key=rank_key):
        if r.error and not r.ttfts_ms:
            print(f"  {r.name:24s}  — {r.error}")
            continue
        tools = f"{r.tool_correct}/{r.tool_total}" if r.tool_total else "-"
        persona = f"{r.persona:.2f}" if r.persona is not None else "-"
        fit = f"{r.fit:.2f}" if r.fit is not None else "-"
        tail = f"   ! {r.error}" if r.error else (f"   ({r.note})" if r.note else "")
        print(
            f"  {r.name:24s}{r.ttft_p50:8.0f}ms{r.ttft_max:6.0f}{tools:>8s}{r.phantom_actions:>8d}"
            f"{persona:>8s}{fit:>6s}{r.contract_violations:>9d}{r.scripted:>10s}{tail}"
        )


def _print_samples(results: list[CandidateResult]) -> None:
    print("\n  what each one actually said")
    for probe in PERSONA_PROBES:
        shown = [r for r in results if r.replies.get(probe)]
        if not shown:
            continue
        print(f"\n  [{probe}]  user: {PERSONA_PROBES[probe][-1][1]!r}")
        for r in shown:
            print(f"    {r.name:22s} {r.replies[probe][0][:150]!r}")
    failures = [(r.name, f) for r in results for f in r.tool_failures]
    if failures:
        print("\n  tool failures")
        for name, failure in failures:
            print(f"    {name:22s} {failure}")


async def main_async(args: argparse.Namespace) -> None:
    wanted = {n.strip() for n in args.candidates.split(",")} if args.candidates else None
    chosen = [c for c in CANDIDATES if wanted is None or c.name in wanted]
    if wanted and (missing := wanted - {c.name for c in chosen}):
        raise SystemExit(f"unknown candidates: {sorted(missing)} — see --list")

    judges = [
        (AsyncOpenAI(api_key=os.environ[key_env], base_url=base_url), model)
        for model, base_url, key_env in JUDGE_MODELS
        if os.getenv(key_env)
    ]
    async with http_context.open():
        agent = _build_agent()
        print(f"  real prompt ~{len(agent.instructions) // 4} tok, {len(agent.tools)} tools, judges: {[m for _, m in judges]}")
        results = []
        for candidate in chosen:
            print(f"  … {candidate.name} ", end="", flush=True)
            results.append(await evaluate(candidate, agent, judges, args))
            print(flush=True)

    _print_table(results)
    _print_samples(results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{'quick' if args.quick else 'full'}.json"
    path.write_text(json.dumps([{**asdict(r), "ttft_p50": r.ttft_p50, "ttft_max": r.ttft_max} for r in results], indent=2))
    print(f"\n  saved → {path.relative_to(PROJECT_ROOT)}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", help="comma-separated names (default: all with a key)")
    parser.add_argument("--quick", action="store_true", help="screening pass: fewer cases, one repeat")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--rpm", type=float, default=0, help="cap requests/min (Cerebras free tier: 5)")
    args = parser.parse_args()

    global _min_interval_s
    _min_interval_s = 60.0 / args.rpm if args.rpm > 0 else 0.0

    if args.list:
        for c in CANDIDATES:
            print(f"  {c.name:24s} {'ready' if os.getenv(c.requires) else 'needs ' + c.requires}  {c.note}")
        return

    logging.basicConfig(level=logging.ERROR)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
