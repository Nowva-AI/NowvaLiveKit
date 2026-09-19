# Jetson Orin Nano Super SER latency (agent report, 2026-09-16)

## Verdict
- Affordable on ONE Orin Nano Super next to LLM: ~95M encoder (WavLM-Base+/HuBERT-Base/emotion2vec+ base) TRT FP16: est 30-60 ms per 4 s utterance, ~250-350 MB resident.
- WavLM-Large / emotion2vec+ large (300M+): affordable in TIME (est 80-150 ms) but NOT in memory next to 4B LLM on one 7.6 GB unit (~633 MB fp16 + workspace on top of Gemma 3 4B Q4 2.6-3.3 GB + 3× RTMPose + STT + TTS + VAD). Put on second Jetson.
- Sleeper: SenseVoice-Small (230M) replaces STT AND SER in one pass. MEASURED 77.1 ms for 5.59 s audio on Orin NX GPU (ORT FP32), RTF 0.0138, with emotion + audio-event heads built in. Lowest marginal SER latency = zero (ASR encoder runs anyway). SER accuracy self-reported on non-standard test sets → validate.
- Timing: SER runs at end-of-turn SERIALLY before LLM prefill → adds 1:1 to TTFT. Gemma 3 4B prefill 471 tok/s on Orin Nano Super → 400-token prompt ≈ 850 ms prefill. 50 ms SER ≈ 6% of TTFT (invisible); 150 ms ≈ 15% (noticeable).

## Measurements
| Model | Device | Precision | Audio | Latency | Memory | Source | M/E |
| SenseVoice-Small 230M | Orin NX MAXN_SUPER | FP32 ORT-CUDA | 5.59 s | 77.1 ms mean, 97.7 P95, RTF 0.0138 | n/r | github star-nexus/G1-Robot-Agents-Speech | Measured |
| SenseVoice-Small | Orin NX | INT8 CPU | 5.59 s | 205.9 ms | | same | M |
| Qwen3-ASR-0.6B | Orin NX | FP16 GPU | 5.59 s | 1218 ms RTF 0.218 | | same | M |
| Whisper base.en TRT | Orin Nano | FP16 | 20 s | 0.86 s enc+dec | 439 MB | NVIDIA-AI-IOT/whisper_trt | M |
| Whisper tiny.en TRT | Orin Nano | FP16 | 20 s | 0.64 s | 488 MB | same | M |
| Compressed wav2vec2 SER+SV | Xavier NX | quantized | | 151 ms | | Springer s11042-025-21057-w | M (paywalled) |
| wav2vec2-base | RPi 4 | INT8 | | RTF 0.98 | 207 MB | arXiv 2202.05993 | M |
| GEMM 8192³ | Orin Nano Super | FP16 | | 10 TFLOPS TRT / 13.1 CUTLASS vs 17 spec | | NVIDIA forum 359635 | M |
| Gemma 3 4B | Orin Nano Super | Ollama Q4 | | prefill 471.5 tok/s, decode 10.1 | | NVIDIA forum 337513 | M |
| Gemma 3n E2B | Orin Nano Super | Ollama | | prefill 135.7, decode 16.6 | | same | M |
| Gemma 4 E2B | Orin Nano | Q4 | | 25.5 tok/s | 3.6 GB | julien.cloud | M |
| Gemma 4 E4B | Orin Nano | Q4_K_M | | | 5533 MiB of 7619 | NVIDIA forum 365641 | M |
| Gemma 3 1B | Orin Nano Super 25 W | Q4_K_M llama.cpp | | 40.8 tok/s, TTFT ~220 ms | 769 MB | yuvrajsingh.io | M |
| Full VLM+voice stack | Orin Nano 8 GB | | | | 4.5/7.6 GB | Jetson AI Lab Reachy Mini | M |
| 4 concurrent GPU procs | Orin Nano | INT8 | | exec-context duration ×30; ×70 at 8 procs; SM util 15-30% | | arXiv 2508.08430 | M |
| WavLM-Base+ 94.7M | Orin Nano Super | FP16 TRT | 4 s | ~30-60 ms | ~250-350 MB | FLOP est | ESTIMATED |
| WavLM-Large 316.6M | Orin Nano Super | FP16 TRT | 4 s | ~80-150 ms | ~750-900 MB | FLOP est | ESTIMATED |
FLOPs: Base+ ≈ 13.4 GFLOP/audio-s (4 s ≈ 53 GFLOP); Large ≈ 35 GFLOP/s (4 s ≈ 140 GFLOP; cross-checks 126.3 GFLOP published for w2v2-large-robust). Realistic 2-3 TFLOPS effective, doubled for overhead.

## Two-Jetson partition
Unit A Vision: 3× RTMPose TRT FP16 batched, DLT, IK, BiLSTM, fault rules, diagnosis → structured cue JSON over gRPC/ZeroMQ.
Unit B Voice: wake word, Silero VAD, STT, SER encoder, Gemma LLM, TTS, LiveKit room.
Reason: opposite duty cycles (continuous vision vs bursty latency-critical voice); concurrency profiling shows ×30 exec-context inflation at 4 procs. Vision gone → ~7.6 GB for LLM 3.3 + STT 0.5 + TTS 0.2 + SER 0.9 ≈ 4.9 GB → WavLM-Large viable. Link cost ~1-3 ms GigE gRPC. Do NOT use LiveKit between Jetsons.

## Caveats
- Nobody has published WavLM/HuBERT/w2v2 on Orin Nano Super → measure yourself (1 day).
- 77 ms SenseVoice = Orin NX FP32; expect ~1.5-1.8× slower on Nano Super, FP16 claws back.
- emotion2vec has NO working ONNX export (FunASR #2291, emotion2vec #55: conv1d mismatch, masking failures) → PyTorch-only on Jetson = engineering risk.
- WavLM gated relative position bias = non-standard op; TRT FP16 conversion undocumented, overflow-class failure modes. HuBERT-Base/w2v2-base export cleanly. If export risk > 1-2 pt SUPERB-ER edge, ship HuBERT first.
- 17 TFLOPS FP16 spec not reachable; expect 60-70% SOL; carrier caps 5V/5.1A.
