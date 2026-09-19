# SER fine-tuning recipe SOTA 2024-2026 (agent report, 2026-09-16)

## Recipe (audio-only, edge-deployable)
0. Data: MSP-Podcast v1.12 challenge split (84,260 train / 2,114 spk; 31,961 dev / 714 spk; test3 3,200 segs / 256 spk balanced 400/class). Keep per-annotator votes. Keep Other/No-agreement (X) segments (dropping = measurable loss for attributes).
1. Encoder: WavLM-Large (24 layers ~315M). Freeze CNN extractor AND bottom 9 transformer layers; fine-tune top 15. CKA shows layers 12-16 adapt most; beat full FT (dev F1 0.518→0.652; test 0.329→0.366, Uniyal & Abrol IS25). Don't use Whisper as audio-only encoder (underperforms WavLM at equal budget in audio-only; contradicted by SAILER 0.383 vs 0.376 → validate).
2. Aggregation: learnable softmax-weighted sum over ALL layers, each reduced by mean⊕std (multi-layer attentive statistics pooling). Never last-layer-only. mean→mean+std UA 69.44→72.56 IEMOCAP; + attentive correlation pooling → 75.60. Plain attention pooling worse than mean in Diatlova (0.28 vs 0.30) — add only if validated.
3. Heads/losses: one trunk two heads. Categorical 8-class: KL-div vs normalized annotator vote distribution (soft labels; −8.23% F1 when removed = biggest lever) + inverse-freq class weights or weighted focal γ=2. Attributes A/V/D: three 2-layer MLPs, 1−CCC loss (87% of top teams; MSE not competitive). L = 1.5·L_cat + 0.4·L_attr. Optional gender aux head (+0.2 CCC pts). Do NOT add speaker-ID aux (−2.3 CCC pts).
4. Regularisation: Manifold MixUp p=0.3 (−2.93% F1 without); audio mixing (overlap 2 utts 0-2 s, average label dists); annotation dropout 20%. Speed perturb / MUSAN+RIR used by top teams, no isolated effect size. Skip label smoothing (hurt: 0.31→0.28 at 0.2).
5. Schedule: two-stage: (a) full train, weighted loss, 15-20 epochs; (b) class-balanced epoch sampling 5 epochs (−1.20% F1 without). AdamW, encoder LR 1e-5 (1e-4 with 9 frozen) + cosine, batch 16-32 w/ grad accum, 10-15 s crops. No layer-wise LR decay ablation exists.
6. Ensemble then distill: 3 seeds average logits (+2.0 F1). Distill into deployment student using teacher soft outputs on augmented/unlabeled audio, CCC loss + quadrant-agreement L1. Student: WavLM-Base+ (fastest to teacher quality) or MobileNetV4-Small / Wav2Small for hard latency.
Expected (MSP-Podcast test3 audio-only): single macro-F1 0.36-0.40; 3-ensemble 0.40-0.43. Attributes single: arousal 0.62-0.68, valence 0.55-0.64, dominance 0.47-0.52 (avg 0.57-0.61). Arousal most reliable by wide margin. A/V/D more useful than 8-way for Nowva.

## Evidence (selected)
| Soft annotator-distribution targets | −8.23% F1 removed | MEDUSA arXiv 2506.09556 |
| Deep cross-modal fusion vs late | −6.31% F1 downgraded | MEDUSA |
| Multitask cat+attr | −2.18% F1 without | MEDUSA |
| Meta-classifier ensemble | 0.439→0.472 F1 dev | MEDUSA |
| Partial FT 15 layers + MASP + focal vs full FT | dev 0.518→0.652; test 0.329→0.366 | Uniyal & Abrol IS25 |
| Focal vs WCE vs balanced sampling | F1-micro 0.487/0.445→0.592 | Lessons Learnt arXiv 2508.07282 |
| Text (RoBERTa) fusion valence | CCC 0.700→0.7334 (+3.3 pts); avg +1.9 | SAIL arXiv 2506.10930 |
| Text fusion categorical | F1 0.383→0.402 (Whisper) | SAILER arXiv 2505.22133 |
| Class undersampling (attributes) | valence 0.5574→0.7334 | SAIL |
| Keeping O/X segments | avg CCC 0.6588→0.6852 | SAIL |
| Speaker-ID aux | 0.6588→0.6363 hurts | SAIL |
| PEFT (BA+LoRA+WS+WG 2-3% params) vs full FT | HuBERT-L 68.53→71.88 UA IEMOCAP only | Gao arXiv 2402.11747 |
| Distill teacher→72K student (Wav2Small) | arousal CCC 0.76→0.66; valence 0.68→0.37 | arXiv 2408.13920 |
| Ensemble top-5 valence | 0.642→0.734 test3 | IS25 naini25 |

## SOTA numbers
| MSP-Podcast test3 Odyssey-24 | macro-F1 0.3569 (Sheffield-MINI 7-model vote); avg CCC 0.5327 baseline won |
| MSP-Podcast test3 IS25 | baseline macro-F1 0.329 (WavLM-L + attentive stats); BEST macro-F1 0.4316 MEDUSA (14 DeepSER + meta soup); #2 SAILER 0.4281 (Whisper+RoBERTa KLD); #3 ABHINAYA 0.4181 |
| MSP-Podcast test3 IS25 | BEST avg CCC 0.6076 (A .6829 / V .6420 / D .4980) SAIL 2-system; baseline 0.579 (A .623 / V .638 / D .477); top-5 fusion valence 0.734 |
| MSP-Podcast Test1 | valence CCC 0.676 WavLM+Dawn teacher (Wav2Small 2024) |
| Same WavLM baseline Test1/2/3 | macro-F1 0.297/0.206/0.356; valence 0.722/0.549/0.632 → splits NOT comparable |
| IEMOCAP 4-class LOSO | UA 75.60% frozen WavLM-L + attentive correlation pooling + label smoothing (arXiv 2211.01756) |
| Cross-corpus MSP→IEMOCAP A/V/D | CCC .66/.50/.52 zero-shot (Wav2Small) |
No 2026 SER challenge (Odyssey 2026 Lisbon has none; ICASSP 2026 none).

## Compute
- WavLM-Large FT on MSP-Podcast challenge train: ~15 GPU-h on 12 GB GPU (SAILER 15 epochs 15 s crops). MEDUSA trained all 14 variants on one RTX 3090 24 GB. Uniyal: A100-40GB 50 epochs.
- VRAM: 12 GB OK batch 8-16 w/ accum; 24 GB comfortable batch 32. Freezing CNN + bottom 9 halves memory & time.
- Week budget = 5-10× what recipe needs: ~20 h sweep + 3×15 h seeds + ~20 h distill < 90 GPU-h. Spend surplus on distillation + in-gym data.
- Apple Silicon MPS NOT viable for training (no distributed, op gaps, SDPA instability, silent garbage; ~3× slower than 4090). Inference smoke tests only.
- Deployment (5 s audio ONNX fp32 Xeon): WavLM-Large 688 ms / 1284 MB; MobileNetV4-S 5 ms / 36 MB; Wav2Small 9 ms 72K params 120 KB. WavLM-Base+ student comfortably real-time on Jetson TRT FP16; WavLM-Large not a per-utterance edge target (per this agent; Jetson agent estimates 80-150 ms).

## Caveats
1. Test1/2/3 numbers not comparable — fix one split.
2. Text fusion (+3.3 valence CCC) requires Whisper-L + RoBERTa per utterance — contradicts edge; distill multimodal teacher into audio-only student (unverified how much survives).
3. Whisper vs WavLM and pooling evidence contradictory → validate on own data.
4. PEFT > full FT is IEMOCAP-only.
5. No layer-wise LR decay ablation; 15 GPU-h is SAILER-specific.
6. Calibration/ECE not standard in SER; conformal prediction exists (arXiv 2503.22712); measure ECE vs soft annotator dists.
7. Zero-shot speech LLMs don't beat fine-tuned encoders (VoxEmo).
8. Domain shift unmeasured: MSP-Podcast seated podcast vs exertion speech under load, close-mic, gym noise. Cross-corpus valence .68→.50. Collect + annotate a few hours in-gym speech = highest-value data.
