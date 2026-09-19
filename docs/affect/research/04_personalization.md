# Per-user baseline personalization (agent report, 2026-09-16)

## Verdict
Neutral-referenced per-speaker normalization — computed from CALIBRATION (neutral) audio only, applied at feature AND output level — is best accuracy per complexity. ~20 lines of arithmetic, zero latency, no training; +1.4 to +9.7 absolute points.
KEY: normalizing with stats from ALL of a speaker's audio buys nothing; from NEUTRAL audio buys everything. Busso et al. ICASSP 2011: none 65.9%, all-data z-norm 65.8% (zero — normalizes away the emotion), neutral-only iterative (IFN) 75.6%, oracle neutral 78.1%. Nowva's calibration phase = the oracle neutral subset for free.
Second: per-speaker OUTPUT calibration (shift/scale predicted A/V by user stats). Tran et al. Interspeech 2023: +0.029 arousal, +0.064 valence CCC on MSP-Podcast from σ-shift alone, unsupervised post-inference.
Upper bound: HuBERT-Large IEMOCAP 71.9% WA speaker-independent vs 78.3% speaker-dependent (Gat et al. ICASSP 2022) → ~6-9 pts is what full speaker-dependence is worth; normalization captures a large fraction.
Heavier methods (LoRA/adapters per speaker, MAML, in-context enrollment) need LABELED emotional speech from the user → skip for v1.

## Evidence
| Per-speaker z-norm of SSL (w2v2 all-layer) features vs global | Pepino Interspeech 2021 | IEMOCAP UAR 65.8→67.2 (+1.4); RAVDESS 75.7→84.3 (+8.6) | speaker μ/σ (transductive!) |
| Neutral-only IFN | Busso ICASSP 2011 | 65.9/65.8 → 75.6 (78.1 oracle) | neutral ≥20% of speaker data |
| Adversarial speaker norm on SSL | Gat IBM ICASSP 2022 | SI WA 71.9→74.2; SD 78.3→81.0 | train-time |
| PAPT + per-speaker label σ-shift (PLDC) | Tran/Yin/Soleymani Interspeech 2023 | MSP-Podcast arousal CCC 0.512→0.541; valence 0.514→0.578; unseen +0.031/+0.042 | ~200 s/speaker | arXiv 2309.02418 |
| Unsupervised personalization via similar-speaker retrieval | Sridhar & Busso TAFFC 2022 | valence CCC +13.5% rel; arousal/dominance <1.9% | unlabeled | arXiv 2201.07876 |
| Few-shot in-context enrollment (speech-LM) | Sakurai 2025 | UA 67.5→75.7 (7-shot) | k labeled utts | arXiv 2509.08344 |
| MAML Meta-PerSER | Interspeech 2025 | +1-2 seen, +7 unseen (listener not speaker) | arXiv 2505.16220 |
| Backprop-free TTA benchmark (11 methods) | ICASSP 2026 | +0.2 to +3.1 acc (+7.6 F1 cross-corpus T3A); entropy-min & pseudo-label FAIL | arXiv 2601.16240 |
| EMO-TTA EM class-prior, bs=1 no backprop | 2025 | avg acc 31.4→38.0 (+6.7) | arXiv 2509.25495 |
| Per-user baseline norm (wearable affect) | Frontiers Digital Health 2026 | balanced acc +11-18 pts; test-inclusive baselines inflate 3-13 pts | resting window |
| Amazon patent US11545174B2 (2023) "Emotion detection using speaker baseline" | enrollment → neutral feature vector; multiple context-specific baselines; emotion as deviation |
| Exertion confound | Voice-as-biomarker 2025 PMC12066516 | F0 +14.6 Hz (d=1.24), pause count d=1.78, rest-vs-hard UAR 0.96 |

## Recipe for Nowva
A. Calibration (45-60 s relaxed speech; 200 s used in MSP-Podcast personalization; 30 s practical floor):
 1. SSL encoder over enrollment → per-dim mean/std of frame embeddings; z = (x − μ_user)/σ_user (Pepino gain).
 2. Neutral eGeMAPS anchors: F0 mean/std, loudness, HNR, jitter/shimmer, articulation rate, pause rate/duration (interpretable exertion channel).
 3. SER head over enrollment → output baselines μ_a,σ_a (arousal), μ_v,σ_v (valence); runtime signal z_a = (a_t − μ_a)/σ_a, not raw.
 4. ECAPA speaker embedding as profile key + context tag (mic/room) — re-estimate when context changes (Amazon multi-baseline design).
 5. Persist per user; EMA resting baselines across sessions α≈0.2/session.
B. Online — TWO baselines: resting (frozen during session) + exertion baseline (slow EMA half-life 60-90 s of NEUTRAL-classified utterances only; gate update to |z_a| < 1, else baseline eats the emotion = Busso's 65.8% failure). Report residuals: effort_z from eGeMAPS channel (F0 + pause rate strongest, d≈1.2-1.8), affect_z = arousal z after regressing out effort_z. Without split, a hard set reads as "excited/angry".
C. Smoothing/hysteresis: median or EMA over ~3 s; enter |z| > 1.5, exit |z| < 1.0, min dwell 5 s.
D. Agent receives 3 tokens: effort_state ∈ {fresh, working, near_limit}, affect_state ∈ {flat, engaged, strained, frustrated}, confidence ∈ {low, high} — thresholded z relative to THIS user. Never raw vectors.
E. Don't build yet: per-user LoRA, MAML, in-context enrollment.

## Caveats
Pepino stats transductive (test included) → optimistic; evaluate with baseline strictly excluded from test window. Busso +9.7 is F0-only binary acted; mechanism transfers, number doesn't. Arousal barely benefits from personalization; valence a lot — since primary signal is effort/arousal, expect low-end gains; the exertion-residual split is where accuracy comes from. Categorical SER in the wild tops ~0.4 macro-F1 (Interspeech 2025 naturalistic) → ship continuous effort/engagement, not labels. Two-baseline scheme is synthesis, not replicated — prototype on a labeled session. Product evidence thin (Hume population-level; Amazon patent strongest).
