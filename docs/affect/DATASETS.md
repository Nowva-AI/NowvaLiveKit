# Affect datasets

Status, paths, label mappings and contamination notes for every corpus the training package knows.
Manifests are built with `python -m training.affect.cli prep <corpus> --root <dir>` and land in
`data/affect/manifests/<corpus>.jsonl` (see `training/affect/data/manifest.py` for the row schema).

| Corpus | Role | Access | Status | Preparer |
|---|---|---|---|---|
| MSP-Podcast v2.0 (409 h, 3,641 spk) | Backbone: 8-class soft votes + A/D/V, Test1/2/3 | Academic data agreement, UT Dallas MSP lab | **Requested, pending** | `prepare_msp_podcast` (expects `Audios/`, `Labels/labels_consensus.csv`, optional `labels_detailed.csv`) |
| MSP-Conversation (>70 h) | Continuous A/V/D traces | Same agreement | Pending | not yet written (backlog: trace targets) |
| IEMOCAP (12 h, 10 spk) | Frustration class, LOSO reporting | USC SAIL agreement | Pending | `prepare_iemocap` (5 sessions; `fru` kept as `frustrated`, excluded from the 8-class head) |
| Switchboard-Affect (25 h) | Degraded-channel robustness | LDC + affect labels | Pending | backlog |
| AlloSat (16 h, French) | Continuous frustration trace | LIUM | Pending | backlog |
| NaturalVoices (5,049 h, automatic labels) | Unlabeled distillation pool (300-500 h subset) | Public | Pending download | `prepare_naturalvoices` (reads `metadata.jsonl` or globs WAVs) |
| data_after_cardio (Columbia ICSL, 59 subj) | Exertion head (binary high/low breathlessness) | GitHub, MIT | Pending download | `prepare_data_after_cardio` (needs `index.csv`: path,subject,exertion_level[,spontaneous]) |
| HUME-VB / ExVo (36.8 h) | Vocal bursts: triumph, distress | Zenodo request | Pending | backlog (event branch) |
| VocalSound (21 h) | Sigh / laughter / cough events | GitHub | Pending | backlog (event branch) |
| VIVAE (0.5 h) | Intensity calibration incl. physical pain | Zenodo | Pending | backlog |
| Nowva recordings | In-domain: gym, close-mic, post-APM | `AFFECT_RECORD=1` at runtime | **Live from V1** | `prepare_nowva_recordings` (reads `utterances.jsonl`; weak exertion labels from workout mode; `labels.jsonl` overrides) |

## Label conventions

- Categorical head: `["angry", "sad", "happy", "surprise", "fear", "disgust", "contempt", "neutral"]` (MSP-Podcast order). `other` / no-agreement segments are kept in training with `label="other"` and excluded from macro-F1.
- Dimensions: `[arousal, dominance, valence]` in `[0, 1]`. MSP-Podcast Likert 1-7 and IEMOCAP 1-5 are min-max scaled. audEERING outputs the same order natively.
- emotion2vec+ (9 classes) → MSP 8: `disgusted→disgust, fearful→fear, surprised→surprise`; `other`/`unknown` mass dropped and renormalized; `contempt` is masked in the KL loss (`training/affect/teachers/emotion2vec.py`).
- Exertion: binary. data_after_cardio levels 4-6 → 1.0. Nowva workout utterances with `effort ∈ {working, near_limit}` → 1.0 (weak).

## Contamination

- audEERING `wav2vec2-large-robust-12-ft-emotion-msp-dim` was trained on MSP-Podcast v1.7 **train**. Any V1a/V1b number on MSP-Podcast dev/test is optimistic; report it, but never select on it.
- emotion2vec+ pseudo-label training pool likely includes IEMOCAP, MELD and MSP-Podcast. Verify before claiming clean IEMOCAP numbers for anything distilled from it.
- NaturalVoices automatic labels come from an SER model; use only as unlabeled audio for distillation, never as ground truth.

## Splits

- MSP-Podcast: use the official `Split_Set` (Test1, Test2, Test3 are **not comparable** to each other; Test3 labels may be held out). State the split in every RESULTS.md row.
- IEMOCAP: leave-one-session-out; `extra.session` is on every segment.
- data_after_cardio: subject-disjoint (every 5th subject → dev).
- Nowva: never split a user across train and test.
