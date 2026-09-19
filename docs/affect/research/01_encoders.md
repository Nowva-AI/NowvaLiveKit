# SOTA speech emotion encoders (agent report, 2026-09-16)

## Best accuracy
1. WavLM-Large fully/partially fine-tuned, layer-weighted-sum + attentive-stats pooling, multi-task head. Official baseline for Odyssey-24 & IS25; 70% of top-8 IS25 teams used WavLM. Best MSP-Podcast macro-F1 in EMO-SUPERB weighted-sum protocol (0.350), best WA in ParaLBench (.615). Audio-only WavLM-Large + MASP + focal + bottom-9 frozen = 0.3661 macro-F1 IS25 test-3 (beats IS25 baseline 0.3293 and every Odyssey-24 winner). + RoBERTa-Large text → ~0.39-0.40.
2. Whisper-Large-v3 encoder FROZEN + trained head. SAIL (IS25 2nd): beat fully fine-tuned WavLM-Large 0.383 vs 0.376 unimodal, 0.402 vs 0.389 multimodal (dev). 12 GB VRAM, ~15 GPU-h. EmoBox: top-1 on 23/32 corpora, cross-corpus 52.01% vs WavLM-Large 44.48% (BUT EmoBox used last-layer only → handicaps WavLM).
3. WavLM-Large + Whisper + RoBERTa-Large 14-model ensemble (NTUA 0.4316) = ceiling reference.

## Best accuracy per FLOP
1. Whisper-Small encoder (~88M, dim 768) + QKV attentive pooling + dual head: IEMOCAP 71.82 WA / 72.96 UA (Feb 2026, arXiv 2602.06000) — matches/beats frozen WavLM-Large (69.47 UA EmoBox) and emotion2vec (71.79 WA) at ~1/4 compute. Whisper-Tiny (~8M) 69.37/69.38. NOTE: Whisper encoder pads to 30 s unless positional embeddings sliced (audio_ctx trick) — train head with same truncation.
2. MERaLiON-SER-v1 (Whisper-Medium frozen + LoRA + ECAPA-TDNN head, 309M): off-the-shelf dual head (7-way categorical + sigmoid A/V/D, joint WCE+CCC). 60.2% UAR in-domain vs emotion2vec-seed 57.9%, GPT-4o-Audio ~55%; 64-70% UAR on MSP-Podcast/IEMOCAP/MELD. Zero-training baseline. Custom MERaLiON license.
3. emotion2vec base (93.79M, dim 768) frozen: IEMOCAP WA 71.79 linear probe (beats WavLM-Large 70.03 at 30% size) but ACTED wins only; MSP-Podcast 5-cls WA .586/UA .342 < WavLM-Large .615/.363. emotion2vec+ large: NO published table.
Floor: Wav2Small 72K params / 120 KB ONNX → arousal CCC 0.66 MSP-Podcast test-1 (valence ~0.37). MobileNetV4-S 3.12M → valence 0.42.

## Evidence (selected)
| WavLM-Large 316.6M/1024 | IEMOCAP SUPERB probe WA 70.03 | emotion2vec ACL24 |
| WavLM-Base+ 94.7M/768 | IEMOCAP WA 67.98 | same |
| HuBERT-Large | 67.62 | ; wav2vec2-Large 65.64; data2vec2.0-base 68.58; Vesper-12 70.70; emotion2vec 71.79 |
| WavLM-Large last-layer frozen | IEMOCAP EmoBox UA 69.47; MSP-Podcast UA/WA/F1 18.60/40.47/17.97; cross-corpus 44.48 |
| Whisper-large-v3 enc last-layer | IEMOCAP 73.54; MSP-Podcast 22.24/44.10/22.12; cross-corpus 52.01 |
| WavLM-Large weighted-sum frozen | MSP-Podcast EMO-SUPERB macro-F1 0.350 (best of 16); 9-cond avg 0.38334 (rank 2, XLS-R-1B 0.38352 rank 1) |
| ParaLBench MSP-Podcast 5-cls | WavLM-Large .615/.363/.576; Whisper-Large .606/.375/.574; emotion2vec-base .586/.342/.546 |
| IS25 baseline WavLM-L FT | macro-F1 0.329; CCC A/V/D 0.623/0.638/0.477 |
| WavLM-L + MASP audio-only | 0.3661 (Uniyal IS25) |
| NTUA 14-model ensemble | 0.4316 rank 1 |
| SAIL Whisper+RoBERTa | CCC 0.642/0.6829/0.498 avg 0.6076 rank 1; top-5 fusion valence 0.734 |
| audEERING Dawn w2v2-L-robust-12 165M | valence CCC 0.638 (TPAMI'23) |
| Wav2Small teacher 483.9M | valence 0.676 test-1; student 72K arousal 0.66 |
| Whisper-Small enc + QKV pooling | IEMOCAP 71.82/72.96 (arXiv 2602.06000) |
| SAIL ablation audio→+text | WavLM-L 0.376→0.389; Whisper-L-v3 0.383→0.402 |
CPU cost (Xeon 6226R, 5 s, FP32 ONNX): WavLM-Large 318.6M/95 GMAC/1284 MB/688 ms; Dawn 165M/55 G/697 MB/372 ms; MobileNetV4-S 3.12M/0.4 G/36 MB/5 ms; Wav2Small 0.072M/9 ms.

## Layer-wise
- DO NOT use last-layer features. Learnable softmax-weighted sum over all layers (challenge baselines, EMO-SUPERB, top teams). EmoBox's Whisper>WavLM is last-layer confounded.
- Emotion lives in the middle: ≈layer 6/12; ≈12-18/24 for WavLM-Large. No universal index (w2v2-robust peaks 17, data2vec 3, XLS-R-1B 30) → learn weights.
- Freeze bottom 9 of 24: dev F1 0.518 (full FT) → 0.5432 (4 trainable) → 0.6283 (12) → 0.6524 (15). CNN extractor frozen always.
- Pooling > classifier: attentive stats (mean⊕std) standard; QKV multi-head pooling +2.3 UA vs mean (IEMOCAP). One Odyssey team found STD > attention → validate.
- emotion2vec uses mean of last 4 layers.

## Text fusion for valence
Yes for valence: ~+0.05 CCC valence, +1-2 macro-F1 — only with a properly fine-tuned LM, not bolt-on sentence embeddings (additive SentenceTransformer LOST F1 0.32→0.31). IS25 baseline valence 0.6385; top teams with RoBERTa 0.679-0.694; top-5 ensemble 0.734 while arousal/dominance barely move. Wav2Small: valence accessible through language; 72K acoustic model arousal 0.66 but valence ~0.37. Replacing text with mid-layer audio: v/a/d 0.531/0.635/0.558 — valence collapses, a/d hold. ASR errors tolerable (~10-15% WER; ASR transcripts match ground-truth text CCC 0.616 vs 0.613). SRPOL: joint multi-task +5% macro-F1, +3% avg CCC vs separate. Nowva already has the transcript → small RoBERTa/DeBERTa text branch concatenated → 2-layer MLP → dual head is near-free at inference.

## Naturalistic winner
WavLM-Large and Whisper encoders are the two credible answers; emotion2vec is not (acted wins; trails WavLM-Large ~3 WA on MSP-Podcast; emotion2vec+ unbenchmarked). Cross-corpus (closest proxy to deployment): Whisper-large-v3 52.01 vs WavLM-Large 44.48 vs HuBERT-Large 42.52 → Whisper favoured for unseen speakers/rooms. Zero-shot speech LLMs trail supervised encoders.

## Caveats
EmoBox protocol-confounded. SUPERB ER official numbers unverified (±1-2 pts). 3loi multi-attribute card looks mislabelled (ordering scrambled). Valence CCC > arousal on test-3 is specific to its balanced construction. emotion2vec-base = 93.79M (not 19M). emotion2vec+ no eval table; 9 classes incl. other/unknown. MERaLiON CCC (v 0.450, a 0.651) on own SG set. Whisper-Small ≈ Large-v3 claim unverified (paper only compares Tiny vs Small). No measured Jetson latency; agent's arithmetic: WavLM-Large ~30-60 ms on Orin Nano Super at 30-40% util (other agent estimated 80-150 ms → measure). Whisper-Small enc ~10-20 ms; Wav2Small sub-ms. No ICASSP 2026 SER challenge.
