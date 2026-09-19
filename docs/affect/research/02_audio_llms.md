# Audio-native LLM vs dedicated SER (agent report, 2026-09-15)

## Verdict: Partially — and not for models that fit on Orin Nano Super
Three independent 2025-26 studies decoupling words from prosody: current audio LLMs mostly READ THE TRANSCRIPT.
- LISTEN (Oct 2025, arxiv 2510.10444): neutral transcripts → 6 audio LLMs ~96% text-only but 25-35% audio-only; best paralinguistic 22.7% vs 10-12.5% chance.
- Incongruent speech (arxiv 2510.25054): 4 SLMs 25-38% on acoustic emotion (chance 25) while predicting lexical emotion 66-100%; plain audio-only SER baseline 46-53%.
- VoxParadox (May 2026, arxiv 2605.27772, 12 models): GPT-4o Audio 8.6% on lexical-acoustic contradictions, agrees with misleading transcript 81.6%; emotion acc 0% (GPT-4o, AF2), 14% (Qwen2-Audio). Reversed-audio control: accuracy RISES when words removed. PCLM+DPO fine-tune lifted Qwen2-Audio 14.9→72.3%.
Positive: RAVDESS (lexically neutral) Qwen2.5-Omni-7B 75.4% > emotion2vec 70.1 > WavLM 61.9 (arxiv 2609.04236). Fine-tuned 7B audio LLMs beat EmoBox supervised in-domain (VoxEmo). But 7B class only.
Edge class (Gemma 3n/4 E2B/E4B): ZERO published emotion/paralinguistic evals. Gemma 4 report only ASR/translation (arxiv 2607.02770). PARSA-Bench: Gemma-E2B collapses to chance on speaker gender (arxiv 2603.14456).

## Evidence
| Model | Size | Benchmark | Value vs dedicated | Source |
| Qwen2-Audio / AF3 zero-shot | 7B | IEMOCAP 4-class macro-F1 | 57.3 / 62.6 vs EmoBox supervised 73.5 | VoxEmo arxiv 2603.08936 |
| Qwen2-Audio SFT | 7B | IEMOCAP macro-F1 | 82.7 vs 73.5 (needs fine-tune) | same |
| Qwen2.5-Omni / AF3 zero-shot | 7B | IEMOCAP UAR | 0.541 / 0.754; +0.04/+0.02 with eGeMAPS cue token in prompt | arxiv 2606.07309 |
| Qwen2.5-Omni | 7B | TWIN-SER tone-word conflict acc | 34.7 vs WavLM 34.2, emotion2vec 31.5, DAS 59.4 | arxiv 2609.04236 |
| best of 6 audio LLMs | 7-35B | LISTEN audio-only | 34.9 (text-only 96) | arxiv 2510.10444 |
| GPT-4o Audio zero-shot | closed | MELD W-F1 | 51.3 vs fine-tuned audio 52.0 | OmniVox arxiv 2503.21480 |
| Kimi-Audio / Qwen2-Audio / Qwen2.5-Omni | 7B | MELD acc | 59.1 / 51.2 / 49.8 | arxiv 2504.18425 |
| Qwen2.5-Omni-3B vs 7B | | MELD acc | 0.558 / 0.570 | HF card |
| Step-Audio 2 mini | 8B | StepEval-Paralinguistic emotion | 82 (Step-Audio 2: 86, GPT-4o 82) | github stepfun-ai/Step-Audio2 |
| WavLM-large SER head | 0.3B | MSP-Podcast 8-class macro-F1 | 0.311 (Odyssey 2024 baseline) | HF 3loi |
| emotion2vec | 90M/300M | IEMOCAP WA | 71.8 | ACL 2024 |

## Edge feasibility (Orin Nano Super 8GB)
- Gemma 4 E2B (5.1B incl. embeddings, 305M conformer audio encoder, 30s max audio): 3.4-3.6 GB at QAT Q4 (Ollama); 25.7 tok/s decode, 67 s cold load (julien.cloud). llama.cpp conformer support merged June 2026 (PR 21421) but needs BF16 mmproj; Jetson AI Lab: "audio issue with E2B under llama.cpp... use vLLM"; NVIDIA's own Orin Nano Super Gemma 4 demo used Parakeet STT, not native audio. No TTFT/audio numbers published.
- Gemma 3n E2B/E4B: E2B 16.55 tok/s decode (Ollama, NVIDIA forum); E4B crashed llama runner; no llama.cpp audio path; superseded by Gemma 4.
- Qwen2.5-Omni-3B: 18.4 GB BF16 for 15 s audio via transformers (infeasible); GGUF Q4 est ~4-5 GB (unverified); 7B Q8 audio works on AGX Orin 64GB after bug fix (llama.cpp #15923).
- Step-Audio 2 mini 8B, Kimi-Audio 7B, Qwen3-Omni 30B-A3B, MiniCPM-o 4.5 9B, Ultravox, Moshi: don't fit next to pose pipeline.

## Recommended: HYBRID — dedicated SER encoder → short symbolic tone token → text-only Gemma 4 E2B/E4B
1. Dedicated encoders beat zero-shot audio LLMs on acoustic emotion in every controlled study.
2. Symbolic cue injection validated (+0.04 UAR; LM integrates rather than blindly follows).
3. Latency: 90-300M encoder on 2-5 s clip = tens of ms on Orin (estimate), parallel with STT; keeps LLM on mature text path.
4. SER model fine-tunable on own gym audio (grunts/breathing/effort).
Revisit native audio when Gemma 4 E2B audio is stable in llama.cpp on JetPack 6 AND a lexical-neutral emotion eval exists.

## Caveats
No measured audio TTFT for any audio LLM on Orin Nano Super. MELD is lexically dominated. StepEval is LLM-judged, 50 items/dim. Gemma 4 audio encoder prosody preservation unknown. SER edge latency figures are estimates.
