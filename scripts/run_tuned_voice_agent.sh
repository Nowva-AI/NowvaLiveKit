#!/usr/bin/env bash
# Talk to Nova on the shipped voice pipeline with a choice of LLM (branch: voice-agent-latency-overhaul).
#
# The tuned stack is now the factory default in src/agent/core/pipeline_factory.py, so
# main.py ships it too: ElevenLabs Flash v2.5 for the voice (both Cartesia accounts are out
# of credits), Deepgram nova-3 on the EU endpoint, VAD_MIN_SILENCE=0.35,
# ENDPOINTING_MIN_DELAY=0.2 and REPLY_START_VAD_WINDOW=0 (the echo fix: LiveKit leaves raw
# VAD interruption armed for the first second of every reply, a fast reply lands inside it
# and Nova cuts herself off; 0 gives the whole reply to the adaptive detector, barge-in
# intact, no latency added, confirmed live 2026-09-21). Any of those env vars still overrides.
# This script only picks the LLM.
#
# Measured: TTFA p50 1453ms -> 1025ms, p90 3142ms -> 1607ms (scripts/benchmarks/ttfa_bench.py).
#
#   ./scripts/run_tuned_voice_agent.sh              # persona: gpt-5.4-mini (sounds like Nova)
#   LLM=fast ./scripts/run_tuned_voice_agent.sh     # Qwen 3.8 27B on Cerebras (needs CEREBRAS_API_KEY)
#   LLM=luna ./scripts/run_tuned_voice_agent.sh     # gpt-5.6-luna
#   LLM=groq ./scripts/run_tuned_voice_agent.sh     # Groq gpt-oss-120b: quickest, generic voice
#   BASELINE=1 ./scripts/run_tuned_voice_agent.sh   # exactly what main.py ships (LLM from .env)
# (NOVA_LLM= works too.)
#
# Measured with the real prompt + 11 tools (scripts/benchmarks/llm_voice_eval.py):
#   gpt-5.4-mini   ~620 ms to first token, 16/16 commands, persona 4.1/5  <- default here
#   gpt-5.6-luna   700-1190 ms,            16/16,          persona 3.9/5  (LLM_MODEL in .env)
#   gpt-oss-120b   ~350 ms,                16/16,          persona 1.6/5
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ "${BASELINE:-0}" == "1" ]]; then
  echo "→ shipped pipeline (factory defaults: ElevenLabs Flash v2.5 + Deepgram nova-3 EU; LLM from .env)"
else
  choice="${NOVA_LLM:-${LLM:-persona}}"
  if [[ "$choice" == "fast" ]]; then
    if [[ -z "${CEREBRAS_API_KEY:-}" ]] && ! grep -qE '^CEREBRAS_API_KEY=.+' .env 2>/dev/null; then
      echo "LLM=fast needs a Cerebras key:" >&2
      echo "  1. create one at https://cloud.cerebras.ai  (API Keys, left sidebar)" >&2
      echo "  2. add a line to .env:  CEREBRAS_API_KEY=<your key>" >&2
      echo "  The free tier is capped at 5 requests/min — too few for live talk; use the Developer tier." >&2
      exit 1
    fi
    echo "→ shipped pipeline, Qwen 3.8 27B on Cerebras"
    export LLM_PROVIDER=cerebras
    export LLM_MODEL=qwen-3.8-27b
  elif [[ "$choice" == "luna" ]]; then
    # The factory sends gpt-5.6 through the Responses API with reasoning off, exactly as
    # production does.
    echo "→ shipped pipeline, gpt-5.6-luna"
    export LLM_PROVIDER=openai
    export LLM_MODEL=gpt-5.6-luna
  elif [[ "$choice" == "groq" ]]; then
    echo "→ shipped pipeline, Groq gpt-oss-120b"
    export LLM_PROVIDER=groq
    export LLM_MODEL=openai/gpt-oss-120b
  else
    echo "→ shipped pipeline, PERSONA llm (gpt-5.4-mini)"
    export LLM_PROVIDER=openai
    export LLM_MODEL=gpt-5.4-mini
    export LLM_API=responses
  fi
fi

# Per-turn latency lands in the log as [METRICS] llm_metrics / tts_metrics / eou_metrics.
export NOWVA_PROFILE="${NOWVA_PROFILE:-1}"

# venv only: the system python on this machine carries livekit-agents 1.2.15 and
# the agent dies on import.
exec ./venv/bin/python src/agent/agents/voice_agent.py console "$@"
