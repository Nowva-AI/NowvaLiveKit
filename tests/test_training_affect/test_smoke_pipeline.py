"""End-to-end smoke test on CPU: tiny WavLM → 2 train steps → eval → ONNX export → parity → report row."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

from training.affect.config import DataMixEntry, ExperimentConfig  # noqa: E402
from training.affect.evaluate import personalization_gain  # noqa: E402
from training.affect.export import check_parity, export_onnx, padding_sensitivity  # noqa: E402
from training.affect.models.affect_model import AffectModel, ExportWrapper  # noqa: E402
from training.affect.report import append_result  # noqa: E402
from training.affect.train import CHECKPOINT_BEST, Trainer  # noqa: E402

SMOKE_CONFIG = Path(__file__).parent.parent.parent / "training" / "affect" / "configs" / "smoke.yaml"


def _experiment(tiny_wavlm_dir: Path, manifest: Path, tmp_path: Path) -> ExperimentConfig:
    config = ExperimentConfig.load(SMOKE_CONFIG)
    config.model.encoder.pretrained = str(tiny_wavlm_dir)
    config.data.train = [DataMixEntry(manifest=str(manifest), weight=1.0)]
    config.data.dev = [DataMixEntry(manifest=str(manifest), weight=1.0)]
    config.output_dir = str(tmp_path / "experiments")
    return config


class TestSmokePipeline:
    def test_train_eval_export_report(self, tiny_wavlm_dir: Path, synthetic_manifest: Path, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("AFFECT_TRAIN_DEVICE", "cpu")
        config = _experiment(tiny_wavlm_dir, synthetic_manifest, tmp_path)
        trainer = Trainer(config)
        best = trainer.fit(resume=False)
        run_dir = trainer.run_dir
        assert (run_dir / CHECKPOINT_BEST).exists()
        assert (run_dir / "metrics.json").exists()
        assert "dev/avg_ccc" in best and "dev/macro_f1" in best and "dev/exertion_auroc" in best
        assert (run_dir / "metrics.csv").read_text().count("\n") >= 2

        resumed = Trainer(config)
        assert resumed.resume() is True
        assert resumed.step == trainer.step

        model = AffectModel(config.model).eval()
        from safetensors.torch import load_file

        model.load_state_dict(load_file(str(run_dir / CHECKPOINT_BEST)), strict=False)
        wrapper = ExportWrapper(model).eval()
        names = ["embedding", "avd", "cat_logits"]
        onnx_path = export_onnx(wrapper, tmp_path / "export", names)
        parity = check_parity(wrapper, onnx_path, names, lengths_seconds=(1.0, 2.0))
        assert parity["max_avd_abs_delta"] < 1e-3
        assert padding_sensitivity(wrapper, seconds=1.5, pad_seconds=1.0) >= 0.0

        import training.affect.report as report_module

        monkeypatch.setattr(report_module, "RESULTS_PATH", tmp_path / "RESULTS.md")
        line = append_result(run_dir, notes="smoke")
        assert "smoke" in line and (tmp_path / "RESULTS.md").exists()

    def test_personalization_eval_uses_runtime_baseline(self) -> None:
        rng = np.random.default_rng(0)
        n = 60
        speakers = np.array([f"s{i % 3}" for i in range(n)])
        offsets = np.array([[0.1, 0.0, -0.1], [-0.1, 0.05, 0.1], [0.0, -0.1, 0.0]])
        true = rng.uniform(0.3, 0.7, (n, 3))
        pred = true + offsets[[int(s[1]) for s in speakers]] + rng.normal(0, 0.02, (n, 3))
        result = personalization_gain(
            {"ids": [str(i) for i in range(n)], "speakers": speakers.tolist(), "avd_pred": pred, "avd_true": true, "has_avd": np.ones(n, dtype=bool)},
            np.array([0.5, 0.5, 0.5]), np.array([0.15, 0.15, 0.15]),
        )
        assert result["personalization/scored_utterances"] > 0
        assert result["personalization/delta_arousal"] > 0.0
