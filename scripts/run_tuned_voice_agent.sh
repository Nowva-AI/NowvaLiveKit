#!/usr/bin/env bash
# Talk to Nova with the tuned low-latency pipeline (branch: voice-agent-latency-overhaul).
#
# Measured against the shipped config: TTFA p50 1591ms -> 1025ms, p90 3329ms -> 1607ms,
# with reply quality unchanged. See scripts/benchmarks/ttfa_bench.py.
#
# The providers here are NOT the shipped ones. LiveKit Inference (which production
# uses for TTS) is out of gateway credits and returns 429 on every model, and the
# direct Cartesia account is out of credits too, so this uses ElevenLabs for voice
# and Groq for the LLM. That means Nova speaks in a different voice than usual.
#
#   ./scripts/run_tuned_voice_agent.sh              # persona: gpt-5.4-mini (sounds like Nova)
#   LLM=fast ./scripts/run_tuned_voice_agent.sh     # Qwen 3.8 27B on Cerebras (needs CEREBRAS_API_KEY)
#   LLM=luna ./scripts/run_tuned_voice_agent.sh     # gpt-5.6-luna on the tuned pipeline
#   LLM=groq ./scripts/run_tuned_voice_agent.sh     # Groq gpt-oss-120b: quickest, generic voice
#   BASELINE=1 ./scripts/run_tuned_voice_agent.sh   # shipped pipeline, for comparison
# (NOVA_LLM= works too.)
#
# Measured with the real prompt + 11 tools (scripts/benchmarks/llm_voice_eval.py):
#   gpt-5.4-mini   ~620 ms to first token, 16/16 commands, persona 4.1/5  <- default here
#   gpt-5.6-luna   700-1190 ms,            16/16,          persona 3.9/5  (shipped)
#   gpt-oss-120b   ~350 ms,                16/16,          persona 1.6/5
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ "${BASELINE:-0}" == "1" ]]; then
  echo "→ shipped pipeline (gpt-5.6-luna + Cartesia via LiveKit Inference)"
  echo "  heads up: LiveKit Inference TTS is currently 429ing, so Nova may not speak."
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
    echo "→ tuned pipeline, Qwen 3.8 27B on Cerebras (+ ElevenLabs Flash v2.5 + Deepgram nova-3 EU)"
    export LLM_PROVIDER=cerebras
    export LLM_MODEL=qwen-3.8-27b
  elif [[ "$choice" == "luna" ]]; then
    # The shipped model on the tuned pipeline: the factory sends gpt-5.6 through the
    # Responses API with reasoning off, exactly as production does.
    echo "→ tuned pipeline, gpt-5.6-luna (+ ElevenLabs Flash v2.5 + Deepgram nova-3 EU)"
    export LLM_PROVIDER=openai
    export LLM_MODEL=gpt-5.6-luna
  elif [[ "$choice" == "groq" ]]; then
    echo "→ tuned pipeline, Groq gpt-oss-120b (+ ElevenLabs Flash v2.5 + Deepgram nova-3 EU)"
    export LLM_PROVIDER=groq
    export LLM_MODEL=openai/gpt-oss-120b
  else
    echo "→ tuned pipeline, PERSONA llm (gpt-5.4-mini + ElevenLabs Flash v2.5 + Deepgram nova-3 EU)"
    export LLM_PROVIDER=openai
    export LLM_MODEL=gpt-5.4-mini
    export LLM_API=responses
  fi
  export TTS_BACKEND=elevenlabs
  export TTS_MODEL=eleven_flash_v2_5
  # EU Deepgram endpoint: final transcript arrives ~100 ms sooner from Europe (measured).
  export STT_BASE_URL=https://api.eu.deepgram.com/v1/listen
  export VAD_MIN_SILENCE="${VAD_MIN_SILENCE:-0.35}"
  export ENDPOINTING_MIN_DELAY="${ENDPOINTING_MIN_DELAY:-0.2}"
  # LiveKit leaves raw VAD interruption armed for the first second of every reply; a fast
  # reply lands inside it and Nova cuts herself off. 0 gives the whole reply to the adaptive
  # detector — barge-in intact, no latency added. Confirmed live 2026-09-21.
  export REPLY_START_VAD_WINDOW="${REPLY_START_VAD_WINDOW:-0}"
fi

# Per-turn latency lands in the log as [METRICS] llm_metrics / tts_metrics / eou_metrics.
export NOWVA_PROFILE="${NOWVA_PROFILE:-1}"

# venv only: the system python on this machine carries livekit-agents 1.2.15 and
# the agent dies on import.
exec ./venv/bin/python src/agent/agents/voice_agent.py console "$@"
