#!/usr/bin/env python3
"""Run ttfa_bench.py across several pipeline configurations and rank them.

Each configuration runs in its own subprocess so plugin state, connection pools
and model handles never leak between arms.

Usage:
    python scripts/benchmarks/ttfa_sweep.py --arms arms.json --trials 2
    python scripts/benchmarks/ttfa_sweep.py --preset endpointing
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BENCH = PROJECT_ROOT / "scripts" / "benchmarks" / "ttfa_bench.py"
PYTHON = PROJECT_ROOT / "venv" / "bin" / "python"
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results" / "ttfa"

PRESETS: dict[str, list[dict]] = {
    "preemptive": [
        {"label": "preemptive-tts-on", "env": {"PREEMPTIVE_TTS": "1"}},
        {"label": "preemptive-tts-off", "env": {"PREEMPTIVE_TTS": "0"}},
        {"label": "preemptive-all-off", "env": {"PREEMPTIVE_GENERATION": "0", "PREEMPTIVE_TTS": "0"}},
    ],
    "turntaking": [
        {"label": "silero-multilingual", "env": {}},
        {"label": "inference-vad-only", "env": {"VAD_BACKEND": "inference"}},
        {"label": "inference-vad-td", "env": {"VAD_BACKEND": "inference", "TURN_DETECTOR": "inference"}},
    ],
    "llm": [
        {"label": "llm-luna", "env": {}},
        {"label": "llm-gemma4-31b", "env": {"LLM_PROVIDER": "inference", "LLM_MODEL": "google/gemma-4-31b-it"}},
    ],
}


def run_arm(arm: dict, args: argparse.Namespace) -> dict | None:
    env = {**os.environ, **arm.get("env", {})}
    cmd = [
        str(PYTHON), str(BENCH),
        "--trials", str(args.trials),
        "--label", arm["label"],
        "--timeout-s", str(args.timeout_s),
        "--warmup", str(args.warmup),
    ]
    if args.utterances:
        cmd += ["--utterances", args.utterances]

    print(f"\n{'=' * 74}\n  ARM: {arm['label']}   {arm.get('env', {})}\n{'=' * 74}", flush=True)
    before = set(RESULTS_DIR.glob("*.json")) if RESULTS_DIR.exists() else set()
    proc = subprocess.run(cmd, env=env, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  FAILED (exit {proc.returncode})")
        print("  " + "\n  ".join(proc.stderr.strip().splitlines()[-15:]))
        return None

    for line in proc.stdout.splitlines():
        if line.startswith("  [") or "TTFA " in line or "under " in line or "  tts " in line \
           or "  llm " in line or "  stt " in line:
            print(line, flush=True)

    after = set(RESULTS_DIR.glob("*.json")) - before
    if not after:
        return None
    data = json.loads(max(after, key=lambda p: p.stat().st_mtime).read_text())
    return data["summary"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS))
    parser.add_argument("--arms", type=str, help="path to a JSON list of {label, env}")
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--utterances", type=str, default=None)
    parser.add_argument("--timeout-s", type=float, default=25.0)
    parser.add_argument("--cooldown-s", type=float, default=30.0,
                        help="pause between arms so provider rate limits reset")
    args = parser.parse_args()

    if args.arms:
        arms = json.loads(Path(args.arms).read_text())
    elif args.preset:
        arms = PRESETS[args.preset]
    else:
        parser.error("pass --preset or --arms")

    summaries = []
    for index, arm in enumerate(arms):
        if index:
            time.sleep(args.cooldown_s)
        summary = run_arm(arm, args)
        if summary:
            summaries.append(summary)

    print(f"\n{'=' * 74}\n  SWEEP RANKING (by TTFA p50)\n{'=' * 74}")
    print(f"  {'arm':28s} {'p50':>7s} {'p90':>7s} {'max':>7s} {'<=500ms':>9s} {'cutoffs':>9s}")
    for s in sorted(summaries, key=lambda s: s["ttfa_ms"]["p50"] or 1e9):
        t = s["ttfa_ms"]
        print(
            f"  {s['label']:28s} {t['p50'] or 0:7.0f} {t['p90'] or 0:7.0f} "
            f"{t['max'] or 0:7.0f} {s['under_target']:5d}/{s['n']:<3d} "
            f"{s.get('false_cutoffs', 0):5d}    "
        )
    print("=" * 74 + "\n")


if __name__ == "__main__":
    main()
