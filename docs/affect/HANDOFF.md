# Affect perception: handoff for the optimization loop

This is the map for the agent that takes the speech affect pipeline from V1 to state of the art.
Read it end to end once. Everything you need to run, measure, change and promote is here or linked.

## 0. Mission and non-negotiables

- **Goal.** Nova hears how the athlete sounds (arousal, valence, dominance, effort) on every utterance and adapts what it says and how it sounds, with no added reply latency. Rival or beat published SOTA on MSP-Podcast, and beat it on Nowva's own in-gym data.
- **Everything runs on the edge.** No cloud call may enter the perception path. The deployed model is an ONNX graph in `models/affect/current/` that must run on a Jetson Orin Nano Super next to the LLM.
- **The LLM gets at most two fields** (`[athlete: effort=… affect=…]`) and never writes voice or style tags. Do not widen this interface; a 2B local model cannot use more.
- **Squats only.** No new exercise-specific logic.
- **Never touch** `CuePriority.FAULT_CUE` in `src/agent/services/coaching_orchestrator.py` or the cached cue WAV path in `src/agent/services/audio_cue_service.py`.
- **Personalization is always evaluated with test-exclusive baselines** (`SpeakerBaseline.stats(exclude_ids=…)`). Test-inclusive baselines inflate results by 3-13 points.
- **No model selection on test splits.** Select on dev; run test only on promoted checkpoints; log every test evaluation in `docs/affect/RESULTS.md`.

## 1. System map

```
console mic (24 kHz, 10 ms frames) → livekit WebRTC APM (AEC, NS, HPF, AGC; hard-coded in the CLI)
  → TappedVAD (src/agent/services/affect_vad_tap.py) ─┬─→ Silero events to the session, unchanged
                                                       └─→ AffectService (src/agent/services/affect_service.py)
      push_audio: streaming 24k→16k resample, 0.3 s pre-roll ring buffer
      on_speech_window: early trigger at ≥0.25 s raw silence after ≥1.0 s voiced; cancelled if speech resumes
      on_speech_end: reuse the early result or run the final pass
      _run_utterance: trim to speech span (VAD probabilities) → crop last 8 s → skip <1 s voiced
        → AffectEngine.infer_async (src/affect/engine.py, onnxruntime, worker thread)
        → prosody_features (src/affect/prosody.py, numpy F0 / pauses)
        → SpeakerBaseline.observe + z_scores (src/affect/baseline.py, neutral-only enrollment)
        → AthleteStateTracker.observe_utterance (src/affect/state.py, hysteresis in utterances)
        → AthleteState {effort, affect, confident, fresh}
      on_rep_effort (from CoachingOrchestrator, ascent_time_s vs best rep) → effort state
  → AffectNodesMixin (src/agent/agents/shared/affect_mixin.py)
      llm_node: snapshot(max_wait ≤ 40 ms) → copy chat_ctx → upsert item id "nowva_athlete_state" → Agent.default.llm_node
      tts_node: normalize_stream → style adapter (src/affect/tts_adapters.py) → Agent.default.tts_node
  → CoachingOrchestrator gates: humor off when frustrated/strained, recap ATHLETE STATE line, calm motivation near limit
  → profiler category "affect" + turn fields affect_wait_ms / affect_fresh / affect_infer_ms (src/profiler/collector.py)
  → ContextViewer :8899 "affect" block; display "affect" pill (src/visual/display.html)
  → UtteranceRecorder (src/affect/recorder.py, AFFECT_RECORD=1) → data/affect/recordings/<user>/<session>/
```

Model contract (`src/affect/manifest.py`): `model.onnx` takes `waveform` float32 `[1, T]` at 16 kHz with a dynamic time axis and returns `embedding [1, D]`, `avd [1, 3]` in `[0, 1]` (order from `model.json.avd_order`), optionally `cat_logits [1, 8]`. Normalization is inside the graph. `model.json` carries labels, population norms and eval metrics. Swapping models is `AFFECT_MODEL_DIR=models/affect/<version>`.

Config: `config/affect.yaml` → `src/affect/config.py` (`AffectConfig`). Env overrides: `AFFECT_ENABLED`, `AFFECT_MODEL_DIR`, `AFFECT_RECORD`, `AFFECT_LLM_WAIT_MS`, `AFFECT_STYLE_ADAPTER`, `AFFECT_INJECT_AS`, `AFFECT_PROVIDERS`.

Training: `training/affect/` (plain PyTorch loop `train.py`, pydantic YAML `config.py`, typer `cli.py`). Runtime and training share `src/affect/baseline.py` so personalization evaluation uses the exact deployed arithmetic.

## 2. How to run

Main venv (Python 3.13, `./venv`):

```bash
PYTHONPATH=src ./venv/bin/pytest tests/test_affect tests/test_affect_vad_tap.py tests/test_affect_service.py tests/test_affect_mixin.py tests/test_coaching_orchestrator.py -q
PYTHONPATH=src:. ./venv/bin/pytest tests/test_training_affect -q          # CPU smoke pipeline, < 1 min
PYTHONPATH=src ./venv/bin/python -m benchmarks --include affect            # p50/p95 at 1/4/8 s
PYTHONPATH=src ./venv/bin/python scripts/tools/affect_replay.py --wav-dir <dir> --no-baseline
PYTHONPATH=src ./venv/bin/python scripts/tools/affect_tts_probe.py --transcribe
AFFECT_ENABLED=1 NOWVA_PROFILE=1 PYTHONPATH=src ./venv/bin/python -m agent.agents.voice_agent console
```

Training venv (GPU box, see `training/affect/README.md`):

```bash
python -m training.affect.cli export-audeering                       # V1a bootstrap runtime model
python -m training.affect.cli prep msp_podcast --root /data/MSP-Podcast
python -m training.affect.cli cache-teachers data/affect/manifests/naturalvoices.jsonl --out data/affect/teacher_cache/bootstrap
python -m training.affect.cli train training/affect/configs/teacher_wavlm_large_pft.yaml --seed 0
python -m training.affect.cli eval experiments/teacher-wavlm-large-pft-s0 --split dev
python -m training.affect.cli export experiments/<run> --version <name> --promote
python -m training.affect.cli report experiments/<run> --notes "..."
python training/affect/trt_build.py --model-dir models/affect/current    # on the Jetson
```

Deploy = copy `models/affect/<version>/` to the device, set `AFFECT_MODEL_DIR`, run `tests/test_affect/test_engine.py::TestRealModel`, run `trt_build.py`.

## 3. Data

See `docs/affect/DATASETS.md` for every corpus, its access status, the label mapping and contamination notes. Short version: MSP-Podcast v2.0 is the backbone (pending the data agreement), NaturalVoices is the unlabeled distillation pool, data_after_cardio trains the exertion head, and the runtime recorder is already collecting in-domain audio (`AFFECT_RECORD=1`, consenting testers only).

## 4. Metric definitions

| Metric | Definition | Where |
|---|---|---|
| macro-F1, UAR | over the 8 MSP classes, `other` excluded from scoring | `training/affect/metrics.py` |
| CCC per dimension | Lin's concordance on the label scale, per arousal / dominance / valence | `metrics.py::concordance_cc` |
| ECE | 15 bins, "correct" = soft-vote mass on the predicted class | `metrics.py::expected_calibration_error` |
| teacher fidelity | CCC(student, teacher) per dimension on unlabeled audio | `metrics.py::teacher_fidelity` |
| exertion | subject-independent accuracy and AUROC at 0.5 | `metrics.py::binary_metrics` |
| personalization gain | ΔCCC after per-speaker output z-scores with test-exclusive baselines, oracle-neutral and predicted-neutral enrollment | `training/affect/evaluate.py::personalization_gain` |
| latency | p50/p95 per provider at 1/4/8 s | `benchmarks/components/bench_affect.py`, `cli.py bench` |
| runtime freshness | `affect_fresh` share of turns whose state came from the current turn; `affect_wait_ms` | profiler summary `affect_fresh_rate`, `affect_avg_wait_ms` |
| CIs | 95% bootstrap over speakers | `metrics.py::speaker_bootstrap_ci` |

## 5. SOTA targets

MSP-Podcast Test1, Test2 and Test3 are **not comparable** (the same baseline scores macro-F1 0.297 / 0.206 / 0.356 on them). Always state the split. If Test3 labels are held out for the corpus release you receive, use dev, Test1 and Test2 and say so.

| Benchmark | Reference | Parity target | Above-SOTA target |
|---|---|---|---|
| MSP-Podcast Test3, 8-class macro-F1, audio-only single model | 0.3661 (Uniyal & Abrol, IS25) | ≥ 0.366 | ≥ 0.40 |
| Same, ensemble | 0.4316 (MEDUSA, 14-model multimodal) | ≥ 0.40 | ≥ 0.432 audio-only |
| Test3 CCC A / V / D | IS25 baseline .623 / .638 / .477; best avg 0.6076 (SAIL; verify per-dimension order) | avg ≥ 0.608 (teachers) | A ≥ .68, V ≥ .64, D ≥ .52 |
| Deployed student vs teacher ensemble | — | ΔF1 ≤ 0.02, ΔCCC ≤ 0.03 | — |
| audEERING bootstrap on Test1 (contaminated) | ~.745 / .655 / .638 A / D / V per model card | report only | — |
| data_after_cardio, binary exertion | 0.81-0.90 (treadmill speech) | ≥ 0.85 subject-independent | ≥ 0.90 |
| Personalization gain (valence CCC) | Tran et al. 2023: +0.029 A / +0.064 V | ≥ +0.03 | ≥ +0.06 |
| Calibration | — | ECE ≤ 0.05 | — |
| Jetson voice unit, 4 s utterance | — | p95 ≤ 80 ms | p95 ≤ 40 ms |
| Runtime | — | `affect_fresh_rate` ≥ 0.90, added TTFT p50 ≤ 20 ms | — |

## 6. Current numbers (V1)

See `docs/affect/RESULTS.md` (rows appended by `cli.py report`). V1a is the audEERING export; its parity and padding-sensitivity figures live in `models/affect/current/model.json` under `eval_metrics`. Mac CPU latency for the V1a model is recorded by `benchmarks --include affect`. No MSP-Podcast numbers exist yet because the corpus is pending.

## 7. Loop protocol

1. One hypothesis per experiment. Prefer config-only changes (`training/affect/configs/*.yaml`); code changes go in a separate commit from the experiment that uses them.
2. Every run lives in `experiments/<name>-s<seed>/` with `config.yaml`, `metrics.csv`, `metrics.json`, checkpoints and `<split>_metrics.json`. Keep a one-paragraph `NOTES.md` per run: hypothesis, result, decision.
3. Select on dev only. Promote a change only when the mean of 3 seeds beats the incumbent by more than the bootstrap CI half-width.
4. Run test splits only on promoted checkpoints; append the row to `RESULTS.md` and count test evaluations there.
5. Before swapping the deployed model: `cli.py export` parity passes (AVD Δ < 1e-3, embedding cosine > 0.999), `bench` p95 within budget on the target device, model dir ≤ 400 MB, and `affect_replay.py` on the in-domain recordings shows no state regressions.
6. One commit per experiment on the `emotion_detection` branch. Roll back after 3 non-improving attempts on the same lever.
7. Log GPU-hours per run in `NOTES.md` (WavLM-Large partial fine-tune ≈ 15 GPU-h on a 24 GB card).

## 8. Chokepoints and levers

| Lever | Path | Notes |
|---|---|---|
| Data mix, label mapping, splits | `training/affect/data/prepare.py`, `data/manifest.py` | per-corpus preparers; `Segment` schema |
| Train-serve parity augmentation | `training/affect/data/augment.py` | APM pass at 24 kHz with AGC warm-up, RIR, noise, speed |
| Sampling | `training/affect/data/sampler.py` | weighted stage 1, class-balanced stage 2 |
| Encoder, freeze depth, Whisper position slicing | `training/affect/models/encoder.py` | `freeze_bottom_layers`, `_forward_whisper` |
| Layer selection and pooling | `training/affect/models/pooling.py` | weighted sum, mean⊕std, attentive stats |
| Heads | `training/affect/models/heads.py` | categorical, sigmoid AVD, exertion |
| Losses and weights | `training/affect/losses.py`, `config.py::LossConfig` | soft-label KL with masks, 1−CCC, distillation quadrant term |
| Schedule, MixUp, early stop | `training/affect/train.py` | explicit loop; `arm_mixup` hooks an encoder layer |
| Teacher targets | `training/affect/teachers/` | audEERING, emotion2vec+ mapping, ensembling, `TargetCache` |
| Export and quantization | `training/affect/export.py`, `trt_build.py` | legacy then dynamo exporter; FP16/INT8 variants |
| Runtime trim / crop / min length / trigger timing | `src/affect/audio.py`, `src/agent/services/affect_service.py`, `config/affect.yaml` (`audio`, `trigger`) | |
| Neutral gating and shrinkage | `src/affect/baseline.py`, `config/affect.yaml` (`baseline`) | |
| State thresholds, dwell, effort ratios | `src/affect/state.py`, `config/affect.yaml` (`state`) | |
| Prompt line and injection mode | `src/agent/agents/shared/affect_mixin.py`, `src/affect/state.py::to_prompt_line` | `inject_as: user_prefix` for Gemma |
| Style policy and adapters | `src/affect/voice_style.py`, `src/affect/tts_adapters.py` | never mirror negative affect |
| Orchestrator gates | `src/agent/services/coaching_orchestrator.py` (`_humor_line`, `_athlete_state_line`, `_speak_llm_motivation`) | |
| Evaluation | `training/affect/evaluate.py`, `metrics.py` | personalization eval imports the runtime baseline |

## 9. Ranked backlog

1. MSP-Podcast teachers: `teacher_wavlm_large_pft.yaml` × 3 seeds, `teacher_whisper_large_v3_frozen.yaml` × 3 seeds (cache the frozen Whisper layer statistics once).
2. Ensemble the six teachers (`teachers/ensemble.py`), cache targets over MSP train + NaturalVoices subset + Nowva recordings, distill with `student_msp_distill.yaml`.
3. Speaker-normalized heads: train heads on per-speaker z-normalized pooled embeddings (MSP-Podcast has speaker ids), ship as `heads_spknorm.npz`, consume in `AffectEngine` (field already reserved in the manifest).
4. Attentive statistics pooling A/B (`pooling: attentive_stats`); evidence conflicts, validate on Nowva data.
5. Layer truncation: if learned layer weights above layer k sum below 5%, export only up to k (`export.py`).
6. Exertion head on data_after_cardio (`exertion_lr.yaml`), then fuse into `AthleteStateTracker` as the voice nudge.
7. A mic hub: one capture → APM → session input, wake word and affect, so effort vocalizations and rest-period breathing reach an EfficientAT event head during workouts (today nothing hears the user during sets; the wake-word tap needs `WAKE_WORD_LOCAL_MIC=1` in console mode).
8. True concentric velocity: timestamp `_build_trajectory_sample` in `src/biomechanics/pipeline.py`, compute in `SessionTracker.on_rep_complete`, add the message type to the forward list at `src/main.py` (`ipc_message_handler`).
9. Wav2Small-class tiny student for arousal/dominance if the Jetson voice unit is under contention.
10. Qwen3-TTS adapter on the Jetson (`Qwen3InstructAdapter` already exposes the 27 instruct strings).
11. Style the OpenAI `gpt-4o-mini-tts` preemptive coaching speech path (`audio_cue_service.py::generate_tts`) through its instructions field.
12. Disable AGC in the console audio chain (patched `AudioProcessingModule` construction) and A/B against the parity-trained model.

## 10. Known pitfalls

- Whisper's HF encoder expects 30 s (3000 mel frames); `models/encoder.py::_forward_whisper` slices the positional table. Heads must be trained with the same slicing.
- WavLM-Base+ group-norm makes outputs padding-sensitive; the runtime never zero-pads (tile-pad for static buckets only) and `export.py::padding_sensitivity` documents the delta.
- AGC is stateful: warm it with 1-2 s of audio before the clip (`augment.py::WebRTCProcessing`).
- Silero END_OF_SPEECH frames include 0.5 s prefix padding and ~0.55 s trailing silence; the service trims with the per-window probabilities instead.
- `inference.TTS.update_options` mutates one options dict shared by every stream; the extra-kwargs adapter swaps in a new dict per stream and the inline-tag adapter avoids shared state entirely.
- Decimal SSML attribute values can be split by sentence tokenizers; the inline adapter emits the whole tag as the first chunk and `tests/test_affect/test_tts_adapters.py` checks blingfire keeps it intact.
- Tool follow-ups reuse the `chat_ctx` object `llm_node` received; injection copies the context and upserts by item id.
- Gemma chat templates reject mid-conversation system messages: set `trigger.inject_as: user_prefix` for the local LLM.
- MSP-Podcast Test3 labels may be held out; Test1/2/3 numbers are not comparable.
- NaturalVoices labels are automatic; use the audio, not the labels.
- audEERING was trained on MSP-Podcast v1.7 train; emotion2vec+ likely saw IEMOCAP/MELD/MSP: report contamination.
- `user_id` can be None during onboarding; the service resolves it at save time.
- `src/main.py` drops IPC message types not on its forward allowlist.
- `src/profiler/collector.py::_build_output` drops categories it does not export; `affect` is exported.
- `.gitignore` covers `models/affect/` and `data/affect/`; never commit weights or recordings.
- The Hugging Face hub client's transfer path stalled on the dev Mac; curl to the CDN works. `teachers/audeering.py` prefers a local copy in `models/affect/_hf/audeering/`.
- The population norms in `model.json` (mean 0.5, std 0.15) are placeholders. In the first live session the audEERING model read the speaker's arousal at 0.19-0.35 through the Mac audio chain, so a fixed gate against those norms rejected every utterance and the baseline never enrolled. The gate now accepts the first `min_samples_before_gate` relaxed-context utterances and rejects outliers against the user's own running statistics. Measure real norms from Nowva recordings and write them into the manifest (lever: `training/affect/export.py`, `ModelManifest.population_avd_*`).
- The CoreML execution provider cannot compile the dynamic-time-axis graph ("ios18.conv: output size is too small") and onnxruntime raises at session creation instead of falling back. `AffectEngine._create_session` retries on CPU, so the Mac runs the CPU provider. Static-shape per-bucket exports would be needed for CoreML; the Jetson uses TensorRT and does not have this problem.
- On the dev Mac the V1a model (165M, fp32, CPU, 4 threads) takes ~260 ms for 4 s of audio, well above the 60 ms budget; the early trigger keeps it off the reply path but `affect_fresh` will be low there. The Jetson TensorRT FP16 path and the distilled Base+ student are where the budget is met; treat Mac latency as a smoke number (see RESULTS.md).

## 11. Edge checklist

- JetPack 6.x with the matching `onnxruntime-gpu` TensorRT build; verify `ort.get_available_providers()` lists `TensorrtExecutionProvider`.
- `trt_build.py` populates `models/affect/trt_cache/` with the 1/4/8 s profile; commit nothing from it.
- Warm-up runs in `voice_agent.py::prewarm` (`_load_affect`); the boot marker `[PREWARM] Affect engine pre-loaded` sits at 0.66.
- Run the voice stack as its own process on the voice unit; pin cores; keep the vision pipeline on the other unit (measured 30× execution-context inflation at four concurrent GPU processes on Orin Nano).
- 30-minute thermal test with the LLM resident before trusting p95 numbers.
