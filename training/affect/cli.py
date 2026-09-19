"""Affect training CLI: prep, cache-teachers, train, distill, eval, export, export-audeering, bench, report.

Run from the repository root with the training environment active:
    python -m training.affect.cli --help
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import typer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

app = typer.Typer(help=__doc__, no_args_is_help=True, add_completion=False)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("affect.cli")

MODELS_ROOT = REPO_ROOT / "models" / "affect"
CORPORA = ("msp_podcast", "iemocap", "naturalvoices", "data_after_cardio", "nowva_recordings")


@app.command()
def prep(
    corpus: str = typer.Argument(..., help=f"one of {', '.join(CORPORA)}"),
    root: Path = typer.Option(..., help="corpus root directory"),
    out: Path = typer.Option(None, help="output manifest path (default data/affect/manifests/<corpus>.jsonl)"),
    max_hours: float = typer.Option(None, help="subset cap in hours (NaturalVoices)"),
) -> None:
    """Build a segment manifest (jsonl) for one corpus."""
    from training.affect.data import prepare

    out = out or REPO_ROOT / "data" / "affect" / "manifests" / f"{corpus}.jsonl"
    count = prepare.run(corpus, root, out, max_hours=max_hours)
    typer.echo(f"wrote {count} segments → {out}")


@app.command("cache-teachers")
def cache_teachers(
    manifest: Path = typer.Argument(..., help="manifest to score"),
    out: Path = typer.Option(..., help="target cache directory"),
    audeering: bool = typer.Option(True, help="score A/D/V with the audEERING teacher"),
    emotion2vec: bool = typer.Option(True, help="score categories with emotion2vec+ large (needs funasr)"),
    device: str = typer.Option("cuda"),
    limit: int = typer.Option(None),
) -> None:
    """Score every segment with the off-the-shelf teachers and store soft targets."""
    from training.affect.teachers.run_cache import cache_targets

    written = cache_targets(manifest, out, use_audeering=audeering, use_emotion2vec=emotion2vec, device=device, limit=limit)
    typer.echo(f"cached {written} segments → {out}")


@app.command()
def train(
    config: Path = typer.Argument(..., help="experiment YAML"),
    seed: int = typer.Option(None, help="override seed"),
    no_resume: bool = typer.Option(False, help="ignore an existing last.safetensors"),
    max_segments: int = typer.Option(None, help="debug: cap segments per split"),
) -> None:
    """Train (or distill, when the config has a distill block) an AffectModel."""
    from training.affect.config import ExperimentConfig
    from training.affect.train import Trainer

    experiment = ExperimentConfig.load(config)
    if seed is not None:
        experiment.seed = seed
        experiment.name = f"{experiment.name}-s{seed}"
    trainer = Trainer(experiment, max_segments=max_segments)
    best = trainer.fit(resume=not no_resume)
    typer.echo(json.dumps(best, indent=2))


@app.command()
def distill(config: Path = typer.Argument(...), seed: int = typer.Option(None)) -> None:
    """Alias for train with a distill block; kept as a separate verb for discoverability."""
    train(config=config, seed=seed, no_resume=False, max_segments=None)


@app.command("eval")
def evaluate(
    run_dir: Path = typer.Argument(..., help="experiment run directory containing best.safetensors"),
    split: str = typer.Option("dev", help="dev | test | test1 | test2 | test3"),
    personalization: bool = typer.Option(True, help="also report test-exclusive personalization gains"),
    max_segments: int = typer.Option(None),
) -> None:
    """Evaluate the best checkpoint of a run on a manifest split."""
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from torch.utils.data import DataLoader

    from training.affect.config import ExperimentConfig
    from training.affect.data.dataset import AffectDataset, collate
    from training.affect.evaluate import collect_predictions, metrics_from_predictions, personalization_gain
    from training.affect.models.affect_model import AffectModel
    from training.affect.train import CHECKPOINT_BEST, _device

    experiment = ExperimentConfig.load(run_dir / "config.yaml")
    model = AffectModel(experiment.model)
    model.load_state_dict(load_file(str(run_dir / CHECKPOINT_BEST)), strict=False)
    device = _device()
    model.to(device)
    entries = experiment.data.dev if split == "dev" else experiment.data.test
    manifests = [(Path(e.manifest), e.weight) for e in entries]
    dataset = AffectDataset(manifests, experiment.data, len(experiment.model.class_labels), split, train=False, max_segments=max_segments)
    loader = DataLoader(dataset, batch_size=experiment.optim.batch_size, collate_fn=collate, num_workers=experiment.data.num_workers)
    predictions = collect_predictions(model, loader, device)
    metrics = metrics_from_predictions(predictions, experiment.model.class_labels, prefix=split)
    if personalization and "avd_pred" in predictions:
        metrics.update(personalization_gain(predictions, np.array([0.5, 0.5, 0.5]), np.array([0.15, 0.15, 0.15])))
    metrics["split"] = split
    (run_dir / f"{split}_metrics.json").write_text(json.dumps(metrics, indent=2))
    if split != "dev":
        (run_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    typer.echo(json.dumps(metrics, indent=2))


@app.command()
def export(
    run_dir: Path = typer.Argument(..., help="experiment run directory containing best.safetensors"),
    version: str = typer.Option(..., help="model version name, e.g. v1b-wavlm-base-plus-distill"),
    promote: bool = typer.Option(False, help="copy to models/affect/current after export"),
    fp16: bool = typer.Option(False),
    int8: bool = typer.Option(False),
) -> None:
    """Export a trained checkpoint to models/affect/<version>/ with parity checks and a manifest."""
    from safetensors.torch import load_file

    from affect.manifest import ModelManifest
    from training.affect.config import ExperimentConfig
    from training.affect.export import (
        check_parity,
        export_onnx,
        finalize_model_dir,
        padding_sensitivity,
        promote_to_current,
        write_fp16_variant,
        write_int8_variant,
    )
    from training.affect.models.affect_model import AffectModel, ExportWrapper
    from training.affect.train import CHECKPOINT_BEST

    experiment = ExperimentConfig.load(run_dir / "config.yaml")
    model = AffectModel(experiment.model)
    model.load_state_dict(load_file(str(run_dir / CHECKPOINT_BEST)), strict=False)
    wrapper = ExportWrapper(model.eval()).eval()
    output_names = ["embedding", "avd"] + (["cat_logits"] if experiment.model.categorical else [])
    out_dir = MODELS_ROOT / version
    onnx_path = export_onnx(wrapper, out_dir, output_names, experiment.data.sample_rate)
    parity = check_parity(wrapper, onnx_path, output_names, experiment.data.sample_rate)
    parity["padding_avd_delta_2s"] = padding_sensitivity(wrapper, experiment.data.sample_rate)
    metrics_path = run_dir / "metrics.json"
    best = json.loads(metrics_path.read_text()).get("best", {}) if metrics_path.exists() else {}
    manifest = ModelManifest(
        version=version,
        family=f"{experiment.model.encoder.name}:{experiment.model.encoder.pretrained}",
        sample_rate=experiment.data.sample_rate,
        min_seconds=experiment.data.min_seconds,
        max_seconds=8.0,
        output_cat_logits="cat_logits" if experiment.model.categorical else None,
        categorical_labels=list(experiment.model.class_labels) if experiment.model.categorical else [],
        embedding_dim=model.embedding_dim,
        input_normalized_in_graph=experiment.model.normalize_input_in_graph,
        eval_metrics={k: float(v) for k, v in best.items() if isinstance(v, (int, float))},
        source=str(run_dir),
        notes=experiment.notes,
    )
    finalize_model_dir(out_dir, manifest, parity)
    if fp16:
        write_fp16_variant(onnx_path)
    if int8:
        write_int8_variant(onnx_path)
    if promote:
        promote_to_current(out_dir, MODELS_ROOT)
    typer.echo(json.dumps({"model_dir": str(out_dir), **parity}, indent=2))


@app.command("export-audeering")
def export_audeering_cmd(
    version: str = typer.Option("bootstrap-audeering-v0"),
    promote: bool = typer.Option(True, help="copy to models/affect/current after export"),
) -> None:
    """Export the audEERING A/D/V model as the V1a bootstrap runtime model."""
    from training.affect.teachers.audeering import export_audeering

    out = export_audeering(MODELS_ROOT / version, promote=promote)
    typer.echo(json.dumps(json.loads((out / "model.json").read_text())["eval_metrics"], indent=2))


@app.command()
def bench(
    model_dir: Path = typer.Option(MODELS_ROOT / "current"),
    providers: str = typer.Option("", help="comma list, e.g. cpu or tensorrt,cuda,cpu"),
    iterations: int = typer.Option(30),
) -> None:
    """Measure per-utterance latency at 1/4/8 s on the local providers."""
    import numpy as np

    from affect.config import EngineConfig
    from affect.engine import AffectEngine

    config = EngineConfig(warmup=True)
    if providers:
        config.providers = [p.strip() for p in providers.split(",")]
    engine = AffectEngine(model_dir, config)
    engine.initialize()
    sr = engine.manifest.sample_rate
    rows = {}
    for seconds in (1.0, 4.0, 8.0):
        wave = np.tanh(0.3 * np.random.default_rng(0).standard_normal(int(seconds * sr))).astype(np.float32)
        runs = [engine.infer(wave).infer_ms for _ in range(iterations)]
        rows[f"{seconds:g}s"] = {"p50_ms": round(float(np.percentile(runs, 50)), 1), "p95_ms": round(float(np.percentile(runs, 95)), 1)}
    typer.echo(json.dumps({"model": engine.manifest.version, "provider": engine.provider, "static_buckets": engine.static_buckets, **rows}, indent=2))
    engine.release()


@app.command()
def report(run_dir: Path = typer.Argument(...), notes: str = typer.Option(""), test_split: str = typer.Option("")) -> None:
    """Append a run's metrics to docs/affect/RESULTS.md."""
    from training.affect.report import append_result

    typer.echo(append_result(run_dir, notes=notes, test_split=test_split))


if __name__ == "__main__":
    app()
