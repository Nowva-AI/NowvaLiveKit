# Best SER training datasets (agent report, 2026-09-15)

## Ranked for Nowva
1. MSP-Podcast v2.0 (409 h, 267,905 turns, 3,641 speakers; 8 primary + 16 secondary cats incl. frustration/annoyance/disappointment/excitement; VAD 1-7, >=5 raters, 1.45M annotations) — backbone for categorical + VAD heads. arxiv 2509.09791; lab-msp.com. Note: Interspeech 2025 challenge shipped 324 h (v1.12).
2. Columbia ICSL data_after_cardio (SenSys'25): 59 participants, 250 treadmill sessions 5-10 mph, 143 min read + 47 min spontaneous speech, respiration belt + HR + PCG, 6-level exertion scale defined by breathlessness/ability to speak. MIT, GitHub. Closest domain match for "talking while out of breath". github.com/Columbia-ICSL/data_after_cardio
3. MSP-Conversation (>70 h): time-continuous V/A/D traces over podcast conversation — supervises arousal trajectory. arxiv 2603.22536
4. HUME-VB / ExVo (36.8 h, 59,201 bursts, 1,702 speakers; 10 emotions self-rated 1-100 incl. Triumph & Distress) — vocal-burst head. zenodo 6308780
5. VocalSound (21,024 clips, 3,365 subjects; laughter/sigh/cough/throat-clear/sneeze/sniff) — sigh transfers to breathing detection. github YuanGongND/vocalsound
6. Deeply Nonverbal Vocalization Dataset (~57 h / 70k / 1,500 spk; 16 classes incl. panting, moaning, sighing, screaming) — only public NV corpus with panting; OpenSLR-99 has only ~1% (0.6 h), rest via Deeply Inc.
7. VIVAE (1,085 files, 11 spk; 6 affects × 4 intensities incl. achievement/triumph, physical pain) — intensity calibration set. zenodo 4066235
8. Switchboard-Affect (~25 h, 10k segs, 543 spk; 10 cats + A/V/D, 6 raters) — degraded-channel domain-shift set. arxiv 2510.13906
9. IEMOCAP (~12 h, 10 spk) — only for explicit frustration class + reporting.
10. AlloSat (303 calls ~16 h speech, FR) — continuous frustration↔satisfaction trace @0.25 s.
11. NaturalVoices (5,049 h, automatic labels; arxiv 2511.00256) / emotion2vec+ pseudo-label regime (40k of 160k h) — pretraining stage, not supervised heads.
Also: NonverbalTTS (17 h; groaning, grunting, breathing + 8 emotions; arxiv 2507.13155), ASVP-ESD (~10 h; pain(groan), excite(triumph); weak labels), Munich BioVoice (19 spk pre/post exercise, HR+SC), voc2vec (Voc125 ~125 h, ICASSP 2025) as burst-branch init.
SKIP: RAVDESS, CREMA-D, EmoV-DB, Expresso, EMNS, EmoNet-Voice (acted/synthetic → theatrical prosody); MELD/CMU-MOSEI eval only; SUSAS (35-word 1990s vocab); Coswara/COUGHVID; CLSE/CoLoSS; AudioSet grunt (214 clips, 50% label accuracy).

## Proposed multi-head combination (shared encoder)
- Encoder: WavLM-Large or emotion2vec+ large init; optional continued pretraining on NaturalVoices slice; burst branch init from voc2vec.
- Head A categorical 8-way: MSP-Podcast primary; cross-corpus eval MELD + Switchboard-Affect.
- Head B VAD regression: MSP-Podcast attributes + MSP-Conversation + Switchboard-Affect.
- Head C frustration (binary/ordinal): MSP-Podcast secondary frustration/annoyance/disappointment + IEMOCAP frustration + AlloSat.
- Head D non-speech events (grunt/gasp/pant/sigh/laugh): VocalSound + Deeply NVD + NonverbalTTS + HUME-VB.
- Head E effort/exertion ordinal 1-6: data_after_cardio primary + Munich BioVoice + VIVAE intensity + HUME-VB triumph/distress intensity aux.

## Gaps Nowva must fill
Nothing public = one speaker, close-mic, mid-recovery, addressing an agent, reverberant gym w/ iron noise. Missing: concentric-phase grunts under load (tennis SCORE! corpus only sports-grunt source), frustration CO-OCCURRING with high physiological arousal (all frustration corpora sedentary → model will confound frustrated with out-of-breath), far-field gym noise. Plan ~20-40 h in-situ from 8-user cohort, close-mic + far-field simultaneously.
Weak supervision supported: data_after_cardio defines exertion by breathlessness/ability to speak; Zhang et al. Frontiers Physiol 2025 (n=92) UAR 0.96 rest vs vigorous from duration, F0, F0 range, pause count/duration. Derive ordinal effort label from set number × reps-in-reserve × tempo collapse × rest elapsed → free target for Head E; failed rep / missed depth = weak positive for frustration; PR / clean set = weak positive for triumph; hand-verify held-out slice. Breathing-vs-semantic pause taxonomy (2,720 15-s snippets, 90.5% high-vs-low exertion) = edge-cheap feature.

## Caveats
MSP-Podcast 28% neutral, κ=0.411 primary; Switchboard-Affect consensus only 67%; valence generalizes cross-corpus far worse than arousal — expect arousal head to transfer to gym, valence head not to. BIIC-Podcast 157 h unverified. CMU-MOSEI hours vary. ASVP-ESD weak labels.
