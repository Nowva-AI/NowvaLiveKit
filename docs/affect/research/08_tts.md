# Expressive local TTS on Jetson (agent report, 2026-09-16)

## Top 3
1. Qwen3-TTS-12Hz-0.6B (CustomVoice/Base), Apache-2.0. Only expressive model with published reproducible Orin Nano numbers: TTFA ~0.7 s / RTF 0.65 int4-AWQ on 235 MB TRT engine (jetson-voice-engine). Control = natural-language instruct= string. True text-in streaming. On AGX Orin 240 ms TTFA at chunk_size=1 after CUDA-graph + StaticCache (faster-qwen3-tts), so <300 ms on Orin Nano plausible not demonstrated. ~1 GB resident.
2. NVIDIA Magpie-TTS-Multilingual 357M. Elo 1061, genuinely expressive; 32-79 ms TTFA on datacenter GPUs. Blockers: Riva embedded needs 8.5 GB + Jetson Thor → must export HF checkpoint to TRT yourself; open checkpoint = 1 public voice + 4 proprietary; emotion subvoices may not be in open weights. High ceiling, integration risk.
3. Chatterbox Turbo 350M, MIT. Cleanest control: continuous exaggeration scalar + paralinguistic tags. Proven on Orin Nano 8 GB Super (JetPack 6.2.2 fp16, 2.4 GB VRAM; fp32 OOM) but ~4 s/utterance warm in unoptimised PyTorch; needs TRT.
Fallback: Kokoro 82M — RTF 0.54 Orin Nano TRT verified, <1 GB, no emotion range.

## Co-residency
Orin Nano Super 8 GB unified @102 GB/s; 4B Q8 LLM capped ~19 tok/s (bandwidth-bound). Budget: JetPack ~1.5 + 4B LLM Q4 2.5-3 + pose TRT 1-1.5 + ASR 0.5 + 0.6B TTS 1 ≈ 6.5-7.5 GB. Fits but LLM decode, TTS decode, 120 FPS pose contend for same 102 GB/s. RECOMMEND TWO UNITS: Node A cameras/pose/diagnosis; Node B ASR+LLM+TTS. One unit only realistic with 2B LLM and ~600-900 ms TTFA under load.

## Evidence (selected)
| Qwen3-TTS 12Hz | 0.6B/1.7B | instruct= NL text; VoiceDesign persona | streaming text-in+audio-out | 0.7 s TTFA RTF 0.65 (Orin Nano int4); 556→240 ms (AGX Orin optimised) | 235 MB int4 / 526 MB fp16 | Qwen-Audio-3.0-TTS-Plus #4 arena |
| Magpie-TTS Multilingual | 357M | emotion subvoices + text-context conditioning; ref audio | yes | 32 ms B200 / 47 H100 / 79 A100 | Riva embedded 8.5 GB Thor-only | Elo 1061 #5 open |
| Chatterbox Turbo | 350M | exaggeration 0-1, cfg_weight, paralinguistic tags | yes | 75 ms/6× RT vendor; ~4 s warm Orin Nano PyTorch | 2.4 GB fp16 Orin | Elo 1020 base |
| Chatterbox Nano | 110M | tags; 1-step decoder | n/d | 3× RT on 8 CPU cores | small | below Turbo |
| Kokoro | 82M | voice ID + speed only | chunked | RTF 0.54 Orin Nano TRT; 28 ms TTFA 5090 | <1 GB | Elo 1061 |
| CosyVoice 3 / Fun-CosyVoice3 | 0.5B | instruct text (emotion, speed, volume) | bidirectional | 150 ms first packet datacenter | ~60% less than v2 | strong |
| IndexTTS-2.5 | 0.8B | 8-D emotion vector + emo_alpha, text desc, ref audio | via Faster-IndexTTS2 TRT | RTF 0.21 4090 bf16 | ~6 GB VRAM | best explicit control API (arXiv 2607.21042) |
| Orpheus | 3B | inline tags | yes | ~100-200 ms RTX/H100 | ~8 GB | ageing |
| Higgs Audio v3 | ~4B | emotion/style/prosody tags | sub-sentence | 617 ms mean total H100 | large | Elo 1038 |
| Fish Audio S2-Pro | 4B+400M | 15k bracket tags | yes | n/a | large | Elo 1121 #2 open |
| Breeze TTS 2 | 3B | Voice Direction text | yes | <40 ms TTFA H100 vendor | n/a | Elo 1202 #1 open (non-commercial) |
| Maya1 | 3B | NL description + 20 emotion tags | vLLM+SNAC | sub-100 ms claimed | ≥16 GB | Elo 1041 |
Excluded on Orin: VibeVoice, Zonos2 8B MoE, LLaSA-3B, Spark-TTS-0.5B (no emotion API), Piper/F5/XTTS/Dia/CSM (no control or superseded).

## Mapping (energy, pace, warmth) ∈ [0,1]³ → backend
- Qwen3-TTS: compose instruct from 3 word-banks quantised to 3 levels → 27 deterministic strings, pre-tokenise + prefix-cache (recovers TTFA). Persona lives in CustomVoice speaker.
- Magpie: nearest emotion subvoice on energy×warmth grid; pace via context sentence; pre-render 3×3 voice matrix.
- Chatterbox Turbo: exaggeration = 0.25 + 0.55·energy; cfg_weight = 0.7 − 0.4·pace; warmth = 2-3 reference clips; ±10% resample for residual pace.
- Kokoro: warmth→voice ID, pace→speed, energy unavailable (log degraded).
Do NOT let the 2-4B LLM write TTS tags directly (tag drift; violates simple-output guardrail). Perception layer emits only 3 floats.

## Caveats
RTF conventions conflict across sources. No public TTFA for any expressive TTS on Orin Nano Super under concurrent load (expect 1.5-2.5× degradation). 240 ms AGX result runs RTF 0.75 (falls behind). Breeze/Chatterbox TTFA vendor-stated. Verify Magpie open weights expose emotion variants. Qwen3-TTS instruct over cloned voices not shipped (discussion #218). jetson-voice-engine is community (35 local TRT patches).
Next step: bench Qwen3-TTS-0.6B int4 + Chatterbox Turbo fp16 on one Orin Nano Super WITH pose pipeline running, TTFA at chunk 1/2/4 → decides one vs two units.
