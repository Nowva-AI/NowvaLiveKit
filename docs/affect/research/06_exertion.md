# Exertion / breathing / effort from voice (agent report, 2026-09-16)

## Ship these (reliable)
- Coarse exertion/breathlessness from between-set speech — BINARY high-vs-low only: ~81-90% (data_after_cardio: 0.9048 on n=21 clips; 0.8102±0.04 baseline). Fine-grained 4-5 levels collapse to ~0.55 UAR. Best-evidenced audio signal for Nowva.
- Pause & breath-group structure — VAD 0.94, semantic pauses 0.89; pause rate/duration correlate with HR r≈0.47-0.51; approximates the clinically validated "talk test" (VT1/VT2 proxy). Cheap, robust.
- Vocal-effort events (grunt/groan/pant/gasp/sigh) — all 13 classes exist verbatim in AudioSet 527 output space (Groan=38, Grunt=39, Breathing=41, Gasp=44, Pant=45, Sigh=26, Wail/moan=25, Cough=47). Any AudioSet tagger emits them today, zero training (but long-tail: ~60 clips each).
- 8-14 Hz vocal tremor (~10 Hz) — Anikin et al. (PMC10078454, 33 speakers, L-sit under load): INVOLUNTARY byproduct of motor-unit recruitment under muscular load; fakers can't reproduce it. Near-free DSP feature; only audio signal specific to MUSCULAR effort (not cardio). Directly relevant to during-set squat grunts.

## Do NOT claim
- Heart rate as a number: r=0.418 best (Interspeech 2023). Unusable.
- Pain: all datasets lab thermal-pain, <100 subjects, speech-only (TAME Pain 51 subj). No basis for "pain detected".
- Fatigue: 81-94% figures are sleep/mental or vocal fatigue; do not transfer to neuromuscular set-to-set fatigue.
- Respiratory rate at gym distances: good numbers (MAE 0.5-1.7 bpm) are facemask/in-ear near-field; 3.8 bpm at 12 km/h w/ 78 dB ambient. Far-field gym mic far worse.
Better from biomechanics: rep-level fatigue (ascent-time lengthening / velocity loss = gold standard), depth/ROM degradation, tempo, rep count.

## Evidence (selected)
| Exertion binary | GRU/AlexNet+MFCC; W2V2 Emb-4/6 | data_after_cardio 15 s windows | Acc 0.9048 (n=21) | arXiv 2509.15473 |
| Exercise intensity 3-class | SVM+ComParE 6373-dim | 92 subj Mandarin | UAR 0.72 (rest-vs-high 0.96; 4-class 0.55) | Front. Physiol. 2025 PMC12066516 |
| HR from speech | deep prosodic+spectral | running corpus | r 0.418 (RPE 0.341) | Interspeech 2023 triantafyllopoulos23 |
| Pause type | 2-stage GRU MFB+Emb-4 | data_after_cardio | semantic 0.89, breathing-only 0.55, VAD 0.94 | arXiv 2509.15473 |
| Breathing signal from speech | ComParE 2020 winner | 17 spk + resp belt | r 0.76 | ISCA 2020 mendonca20 |
| Resp rate near-field exercise | multi-task Conv-LSTM mel | 21 subj | CCC 0.76 | Apple arXiv 2107.14028 |
| VocalSound 6-class | AST-style | 21k clips 3365 spk | 90.5% | ICASSP 2022 arXiv 2205.03433 (NO grunt/groan/pant; only sigh overlaps) |
| EfficientAT mn10_as | AudioSet-2M | mAP 47.1 (4.88 M params, 0.54 B MACs, MIT; <0.25 s RPi4) | github fschmid56/EfficientAT |
| EfficientAT mn04_as | | mAP 43.2 (0.98 M params, 0.11 B MACs) | |
| BEATs | | mAP 50.6 (ViT, marginal on edge) | ICML 2023 |
| voc2vec | w2v2 on 125 h non-verbal | beats OpenSMILE & emotion2vec on 6 benchmarks; 95 M | github koudounasalkis/voc2vec |
| ExVo 2022 | HUME-VB | MultiTask CCC 0.335→0.435; FewShot 0.444→0.739 | competitions.hume.ai/exvo2022 |
| FM robustness | w2v2/WavLM/HuBERT | post-exercise speech | significant ASR degradation on breathless speech | arXiv 2603.27508 |

## Recommended minimal audio branch
Backbone: EfficientAT mn10_as frozen (32 kHz, 25/10 ms mel). Use (a) 527 logits sliced to 13 indices, (b) penultimate embedding → small trained head. mn04_as if 5× less compute needed. Skip BEATs/AST/CLAP.
Shipped classes (6): speech, breath (Breathing+Pant+Gasp), effort_grunt (Grunt+Snort), strain (Groan+Wail/moan+Whimper), cough, gym_noise/other.
Two modes:
- During set: 2 s window / 0.5 s hop → event stream + DSP tremor feature (Hilbert envelope of voiced segments, FFT band power 8-14 Hz normalised to 2-6 Hz). Output per rep: effort_index ∈ [0,1].
- Between sets: VAD + pause stats (pause rate, mean pause dur, phrase length between breaths) + ΔF0 vs user's rested baseline → BINARY recovered/not_recovered, updated each second.
Latency: single-digit ms per window on Orin GPU; branch < 50 ms total.
Fusion: audio stays OUT of diagnosis engine; produces two scalars gating the coaching orchestrator:
1. recovery_readiness (audio-only) → WHEN to start next set, how much to talk. Biomech has no signal here — audio's unique contribution.
2. effort_index (audio) × velocity_loss (biomech) → agreement = genuine grind; disagreement = data problem. Audio confirms, never overrides.
Per-user baselining is MANDATORY (Interspeech work explicitly concluded exertion-from-speech needs personalization).

## Caveats
90.48% is 19/21 clips. AudioSet exertion classes ~60 clips each (high-variance AP). Every exertion result is treadmill/cardio — nothing on resistance training. Gym acoustics (music, plate clangs, far-field) unaddressed everywhere = largest risk. Breathing-only pause detection 0.55 — don't depend on it. ASR degrades on breathless speech — test local STT on post-set audio. HUME-VB has no exertion/effort class. No sports-tech product does exertion-from-voice (open space for YC narrative).
