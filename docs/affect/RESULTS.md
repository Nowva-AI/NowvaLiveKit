# Affect experiments

Rows are appended by `python -m training.affect.cli report <run_dir> --notes "..."`. Select on dev only;
run test splits only on promoted checkpoints and log every test evaluation here.

| date | run | git | seed | encoder | dev/macro_f1 | dev/avg_ccc | dev/ccc_arousal | dev/ccc_valence | dev/ccc_dominance | dev/ece | test_split | test/macro_f1 | test/avg_ccc | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

## V1 measurements (2026-09-17, dev Mac, Apple Silicon, onnxruntime 1.26 CPU EP unless noted)

Export parity (`training/affect/export.py::check_parity`, torch vs onnxruntime, random audio at 1/2/4/8 s):

| Graph | max AVD abs delta | min embedding cosine | padding delta (2 s zeros appended to 3 s) | ONNX size |
|---|---|---|---|---|
| V1a audEERING w2v2-L-12 A/D/V (`models/affect/bootstrap-audeering-v0`, promoted to `current`) | 2.1e-06 | 1.000 | 0.038 | 661 MB |
| WavLM-Base+ + random head (`models/affect/probe-wavlm-base`) | 0.0 | 1.000 | 0.0034 | 381 MB |
| HuBERT-Base + random head (`models/affect/probe-hubert-base`) | 0.0 | 1.000 | 0.0030 | 381 MB |

All three exported with the legacy TorchScript exporter (torch 2.12, opset 18, dynamic time axis). The padding delta is why the runtime never zero-pads.

Cartesia sonic-3 style delivery (`scripts/tools/affect_tts_probe.py --transcribe`, one line, Deepgram-transcribed):

| Method | speed 0.9 / 1.1 duration ratio (≥1.08) | volume 1.15 vs 0.9 (≥1.5 dB) | tag words spoken | verdict |
|---|---|---|---|---|
| inline SSML tags via LiveKit inference | 1.81 | +2.83 dB | none | **pass → default adapter `cartesia_inline`** |
| per-stream extra_kwargs via LiveKit inference | 1.42 | +2.20 dB | none | pass (two variants hit gateway 429s) |
| standalone cartesia plugin, generation_config | 1.02 | +2.68 dB | none | fail on speed |

Note: the gateway rate-limits bursts (429); space probe runs out or use `--methods plugin` for volume-only checks.

Runtime latency, V1a model, `python -m training.affect.cli bench` (onnxruntime CPU EP, `intra_op_threads: 4`, dev Mac):

| Utterance length | p50 | p95 |
|---|---|---|
| 1 s | 74 ms | 94 ms |
| 4 s | 263 ms | 271 ms |
| 8 s | 554 ms | 621 ms |

With 2 threads the 4 s figure was 524 ms p50. The default provider chain on the Mac tries CoreML, which fails to compile the dynamic-axis graph, and falls back to CPU (`AffectEngine._create_session`). Jetson TensorRT FP16 numbers are still to be measured (HANDOFF §11).

Replay smoke (`scripts/tools/affect_replay.py --no-baseline` on five cached TTS cue WAVs through real Silero + service + model): four clips skipped as under 1 s voiced, one scored at avd (0.64, 0.72, 0.40), population z (0.95, 1.44, −0.68) → `fresh/flat`, 95 ms inference; early trigger reused once.
