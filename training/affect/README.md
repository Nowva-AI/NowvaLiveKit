# Affect training

Trains the speech emotion / effort models that the runtime in `src/affect/` deploys.
Plain PyTorch, one explicit loop (`train.py`), pydantic YAML configs (`configs/`), typer CLI (`cli.py`).

## Environment (GPU box, Linux, 24 GB VRAM)

```bash
python3.11 -m venv .venv-train && source .venv-train/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchaudio==2.6.0
pip install -r training/affect/requirements.txt
python -m training.affect.cli --help
```

Disk budget: MSP-Podcast ~40 GB, NaturalVoices 300-500 h subset ~60 GB, teacher caches ~5 GB,
experiments ~2 GB per run (safetensors). Plan ~250 GB.

Apple Silicon (MPS) is for smoke tests only; the loop warns and continues on it.

## Commands

| Step | Command |
|---|---|
| Bootstrap runtime model (no data needed) | `python -m training.affect.cli export-audeering` |
| Manifest for a corpus | `python -m training.affect.cli prep msp_podcast --root /data/MSP-Podcast` |
| Teacher soft targets | `python -m training.affect.cli cache-teachers data/affect/manifests/naturalvoices.jsonl --out data/affect/teacher_cache/bootstrap` |
| Train / distill | `python -m training.affect.cli train training/affect/configs/teacher_wavlm_large_pft.yaml --seed 0` |
| Evaluate | `python -m training.affect.cli eval experiments/teacher-wavlm-large-pft-s0 --split test3` |
| Export | `python -m training.affect.cli export experiments/v2-wavlm-base-plus-student-s0 --version v2-student --promote` |
| Latency | `python -m training.affect.cli bench --providers cuda,cpu` |
| Leaderboard row | `python -m training.affect.cli report experiments/<run> --notes "..."` |
| Jetson engine | `python training/affect/trt_build.py --model-dir models/affect/current` |

Runs land in `experiments/<name>-s<seed>/` with `config.yaml`, `metrics.csv`, `metrics.json`,
`best.safetensors`, `last.safetensors` (resumable) and `<split>_metrics.json`.

## Recipes

- `configs/bootstrap_student_*.yaml`: V1b. Distill audEERING (A/D/V) + emotion2vec+ (categorical) into a 95M student. Capped at teacher quality; for speed and for the export path.
- `configs/teacher_wavlm_large_pft.yaml`: V2 teacher A. Partial fine-tune of WavLM-Large (top 15 layers), soft-label KL + 1−CCC, MixUp, two-stage sampling. ~15 GPU-h.
- `configs/teacher_whisper_large_v3_frozen.yaml`: V2 teacher B. Frozen Whisper-Large-v3 encoder, sliced positions, heads only.
- `configs/student_msp_distill.yaml`: V2 deployed student, distilled from the ensemble.
- `configs/exertion_lr.yaml`: exertion head on data_after_cardio.

See `docs/affect/HANDOFF.md` for targets, the loop protocol and the chokepoints.
